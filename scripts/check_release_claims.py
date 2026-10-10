#!/usr/bin/env python3
"""Versioned release closure: notes describe every selected cell with its real claim.

Use --release 2.3 for the current registry; the retained default is 2.2.

The claim vocabulary is the one in the header of parity/promotion_2_2.toml (`inference_claim`):

  point_only           exact or plug-in point, no interval
  calibrated           interval with a coverage record at its own coordinate
  structural_envelope  range over supplied structural scenarios; not a CI
  assumption_range     range over a declared sensitivity set; not a CI
  none                 no numerical claim (identification/inspection only)

plus the two release-notes phrases `nominal` (a pre-2.2 interval with no coverage record at its
own coordinate, `estimator_grid_not_measured`; 2.2 ships none) and `statistical interval` (the
reader-facing name of a `calibrated` interval).

For docs/release-notes/v2.2.0.md (or --notes PATH) this checks:

  1. the notes define all six terms in a glossary (outside the cell entries);
  2. every record of parity/promotion_2_2.toml has one entry, opened by a
     `<!-- cell: <record id> -->` marker, with exactly one `Claim: \\`<word>\\`` line;
  3. inside an entry only ONE vocabulary claim word appears (backticked token, the bare word
     `calibrated`, the phrase `statistical interval` for calibrated, `nominal`), and it equals the
     record's EFFECTIVE claim: the record's `inference_claim`, except that a `calibrated` record
     whose coverage records are not all in parity/coverage_records.toml, or which has
     no licensed uncertainty route, is point_only (the interval is withheld) and its
     entry may not use the word calibrated or `statistical
     interval` at all, and must say the interval is `withheld`; `nominal` never appears
     affirmatively (negations such as "no nominal" are fine); an entry whose effective claim
     is not `calibrated` may not promise a confidence / credible / bootstrap / prediction /
     percentile / sampling-uncertainty interval in prose either (negated or withheld
     mentions are fine);
  4. every 2.2B work package of B_PACKAGES has either a record entry or a
     `<!-- todo-cell: <package> -->` marker (the draft skeleton);
  5. X7 (GPU lane) is stated as carried forward with its rationale (CPU-only 2.1 baseline);
  6. the entry's `Bounds:` line equals the record's numeric bounds, any stated `record status`
     equals the record's status, and its `Guarantee:` string is the record's (stale otherwise: a warning while
     the notes are a DRAFT, an error outside draft mode and in `--final`);
  7. a record with a closed uncertainty route says its interval is `withheld` / `not offered`; an
     `assumption_range` or `structural_envelope` record says the range is never a confidence
     interval; a guarantee with a `complete` token (e.g.
     `exact_range_complete_within_declared_contamination_class`) is never printed without the
     record's qualifier (`within the declared contamination class`) stated beside it, and
     `sampling interval not offered` when no uncertainty route is licensed (rules 6-7: stale, as above).

Claim words are found in prose as well as backticks: `structural envelope` / `assumption range` (with
a space), and a plain `none` after the word claim (`Claim: none`). The `Claim:` line may omit the
backticks.

`--final` additionally fails on any DRAFT or TODO marker (the cut). `--write-draft` (re)generates
the draft skeleton from the records; it refuses to overwrite a notes file that is no longer
marked draft unless `--force`.

    python3 scripts/check_release_claims.py                 # check the draft/notes
    python3 scripts/check_release_claims.py --final         # at the cut: no draft, no TODO
    python3 scripts/check_release_claims.py --write-draft   # regenerate the skeleton
    python3 scripts/check_release_claims.py --self-test

Environment: RELEASE_CLAIMS_ROOT (tree to read, default the repo).
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("RELEASE_CLAIMS_ROOT", REPO))
RELEASE = "2.2"
RECORDS = "parity/promotion_2_2.toml"
COVERAGE = "parity/coverage_records.toml"
NOTES = "docs/release-notes/v2.2.0.md"

CLAIMS = ("point_only", "calibrated", "structural_envelope", "assumption_range", "none")
DRAFT_MARK = "<!-- draft:v2.2.0 -->"

# 2.2B work packages (TODO.md): package -> (workstream, title). A package disappears from the TODO
# list of the draft as soon as its record exists in parity/promotion_2_2.toml. Additive.
B_PACKAGES = {
    "B1": ("X2", "One latent-confounded ADMG transport row"),
    "B2": ("X4", "One smoothed dose-response transport grid"),
    "B3": ("X3", "Joint mechanism deviations and sampling uncertainty"),
    "B4": ("X6", "Planning over the new restricted catalogs and competing designs"),
    "B5": ("X8", "First bounded ADMG counterfactual-ID cell"),
    "B6": ("X10", "Exact binary causal observation recovery"),
}

GLOSSARY = (
    (
        "`point_only`",
        "An exact or plug-in point estimate. No interval of any kind is claimed.",
    ),
    (
        "`nominal`",
        "An interval whose coverage was never measured at its own coordinate "
        "(`estimator_grid_not_measured`). 2.2 cells publish none.",
    ),
    (
        "`calibrated`",
        "An interval with a coverage record measured at its own coordinate; shown to readers "
        "as a *statistical interval*. A cell is described this way only once its records exist.",
    ),
    (
        "`structural_envelope`",
        "The range of an answer over explicitly supplied structural scenarios. "
        "Not a confidence interval.",
    ),
    (
        "`assumption_range`",
        "The range of an answer over a declared sensitivity set. Not a confidence interval.",
    ),
    ("`none`", "No numerical claim: identification or inspection only."),
    (
        "statistical interval",
        "The reader-facing name of a `calibrated` interval; it appears only beside a measured "
        "coverage record.",
    ),
)

X7_LINE = (
    "X7 (GPU lane) is carried forward to 2.3 or later with no code in 2.2: the 2.1 neural "
    "cross-fit baseline (`benches/baselines/neural_crossfit.md`) is CPU-only and records no "
    "accelerator measurement, so the condition for starting the lane (a measured whole-analysis "
    "gain) is not met. The CPU provider stays the released provider; no 2.2 result depends on it."
)

TOKEN = re.compile(r"`(point_only|calibrated|structural_envelope|assumption_range|none)`")
BARE = re.compile(r"\b(point_only|structural[_ ]envelope|assumption[_ ]range)\b", re.I)
# a plain `none` counts as a claim word only right after the word claim ("Claim: none")
PLAIN_NONE = re.compile(r"\bclaims?\b[`:]?\s*(?:is |of )?\*{0,2}`?none\b", re.I)
BARE_CALIBRATED = re.compile(
    r"(?<![\w-])(?<!un)(?<!not )(?<!not yet )(?<!until )(?<!pending )calibrated\b", re.I
)
STAT = re.compile(r"(?<!not a )(?<!no )statistical interval", re.I)
NOMINAL = re.compile(r"(?<!no )(?<!not )(?<!never )(?<!any )\bnominal\b", re.I)
MARK = re.compile(r"<!-- (cell|todo-cell|carried-forward): ([^>]*?) -->")
CLAIM_LINE = re.compile(r"^Claim: `?(\w+)`?\s*$", re.M)
BOUNDS_LINE = re.compile(r"^- Bounds: (.*)\.\s*$", re.M)
STATUS_PART = re.compile(r"record status `(\w+)`")
GUARANTEE_PART = re.compile(r"^- Guarantee: `([^`]+)`", re.M)
NEVER_CI = re.compile(
    r"(never|not)\s+(a|an)\s+(confidence|statistical)\s+interval|never\s+a\s+CI", re.I
)
WITHHELD = re.compile(r"withheld|not offered", re.I)
# An interval a cell whose effective claim is not `calibrated` may not promise, however it is
# worded: "a 95% bootstrap interval is reported" is a calibrated claim in other words. Negated
# forms ("not a confidence interval", "never a CI", "no credible interval") are fine.
INTERVAL_PROMISE = re.compile(
    r"(?<!not a )(?<!not an )(?<!never a )(?<!no )(?<!without a )"
    r"\b(?:confidence|credible|bootstrap|prediction|percentile|sampling-uncertainty|sampling uncertainty)"
    r"\s+intervals?\b(?!\s+(?:is|are)\s+(?:not|never|withheld))",
    re.I,
)


def load(root: Path, rel: str) -> dict:
    p = root / rel
    return tomllib.loads(p.read_text()) if p.is_file() else {}


def effective_claim(rec: dict, present: set[str]) -> tuple[str, bool]:
    """(public claim, whether the allocated interval remains withheld)."""
    claim = rec["inference_claim"]
    if claim == "calibrated":
        owed = list(rec.get("coverage_records") or [])
        if not owed or not all(c in present for c in owed) or not licensed_uncertainty(rec):
            return "point_only", True
    return claim, False


def numeric_bounds(rec: dict) -> str:
    items = [
        f"{k} = {v}"
        for k, v in rec.get("bounds", {}).items()
        if isinstance(v, int) and not isinstance(v, bool)
    ]
    return ", ".join(items) or "none declared"


def closed_uncertainty(rec: dict) -> list[dict]:
    return [
        r
        for r in rec.get("routes") or []
        if r.get("stage") == "uncertainty" and r.get("status") == "closed"
    ]


def licensed_uncertainty(rec: dict) -> bool:
    return any(
        r.get("stage") == "uncertainty" and r.get("status") == "licensed"
        for r in rec.get("routes") or []
    )


def complete_qualifier(guarantee: str) -> str | None:
    """None when the guarantee has no `complete` token; else the words after it ('' if none)."""
    # a record may append its own note after the identifier ("id (range only; ...)")
    tokens = guarantee.split(" ", 1)[0].split("_")
    if "complete" not in tokens:
        return None
    return " ".join(tokens[tokens.index("complete") + 1 :])


def qualifier_regex(words: str) -> re.Pattern[str]:
    """The qualifier's words in order, allowing an inserted 'the' ('within the declared class')."""
    return re.compile(r"\s+(?:the\s+)?".join(re.escape(w) for w in words.split()), re.I)


def guarantee_text(rec: dict) -> str:
    g = rec["guarantee"]
    q = complete_qualifier(g)
    if q and q.startswith("within ") and not q.startswith("within the "):
        q = "within the " + q[len("within ") :]
    note = ""
    if q is not None:
        note = f" (complete only {q}" if q else " (complete as inherited from the cited theorem"
        if not licensed_uncertainty(rec):
            note += "; sampling interval not offered"
        note += ")"
    return f"`{g}`{note}"


def entry_text(rec: dict, present: set[str]) -> str:
    short = rec["id"].split(".")[2]
    name = rec["id"].split(".")[3].replace("_", " ")
    claim, pending = effective_claim(rec, present)
    lines = [
        f"<!-- cell: {rec['id']} -->",
        f"### {short}: {name}",
        "",
        f"Claim: `{claim}`",
        "",
        f"- Question: {rec['consumer_question']}",
        f"- Scope: {rec['graph_class']}",
        f"- Guarantee: {guarantee_text(rec)}.",
        f"- Bounds: {numeric_bounds(rec)}.",
    ]
    if pending:
        lines.append(
            "- Interval: withheld. The interval route is closed (`cell_not_licensed`) until its "
            "coverage records are measured at the cut; the result is the point only."
        )
    elif closed_uncertainty(rec):
        codes = ", ".join(
            sorted(
                {f"`{r['reason_code']}`" for r in closed_uncertainty(rec) if r.get("reason_code")}
            )
        )
        lines.append(
            "- Interval: withheld. The sampling-uncertainty interval is not offered"
            + (f" (its route is closed: {codes})." if codes else ".")
        )
    if claim in ("assumption_range", "structural_envelope"):
        lines.append(
            f"- Not an interval: the {claim.replace('_', ' ')} is never a confidence interval."
        )
    return "\n".join(lines) + "\n"


def draft(records: list[dict], present: set[str]) -> str:
    out = [
        f"# Antecedent {RELEASE}.0",
        "",
        DRAFT_MARK,
        " ".join(
            (
                "> **DRAFT.** Generated by `scripts/check_release_claims.py --write-draft` from",
                f"`{RECORDS}`. Not a release statement until the cut: the entries show",
                "each cell's claim as the records stand today, and TODO markers stand for cells not yet",
                "frozen. Regenerate, then hand-edit prose, and run `check_release_claims.py --final`.",
            )
        ),
        "",
        "## How to read the claims",
        "",
        "| Term | Meaning |",
        "| --- | --- |",
    ]
    if RELEASE == "2.3":
        start = out.index("## How to read the claims")
        out[start:start] = [
            "## Consumer workflows in this draft",
            "",
            "- Carry scientifically named quantities and aligned laws from original native producers "
            + "into supported decisions and inverse queries. A mean grid supplies affine mean utility; "
            + "probability, quantile and nonlinear-law questions require the separately declared joint law.",
            "- Recalculate declared adjusted, doubly robust, finite response/transport, Bayesian, "
            + "temporal and design-specific analyses through retained original execution state. "
            + "Actual component work is recorded; supported reuse preserves the producing uncertainty status.",
            "- Execute bounded external mean callbacks under explicit provider, environment, RNG "
            + "and side-effect contracts. Combined native/two-provider mean decisions share a receipt; "
            + "callback outputs keep their original external attestation.",
            "- Preserve original diagnostic scopes and semantic coordinates through decision and "
            + "inverse projections. Imported evidence retains unresolved dependencies; fresh source "
            + "resolution requires actual original analysis rather than caller metadata.",
            "- Inspect effect constancy in transport reviews, advisory prior-source ranking and "
            + "partition-specific point-policy comparisons. Non-rejection establishes neither "
            + "transportability nor pooling, and policy agreement supplies no generalization guarantee.",
            "- Bind rollout rankings to complete original laws, priors, terminal actions and utilities. "
            + "Delivered finite count data use the original proposal/repair and transport estimators; "
            + "a historical artifact alone restores no executable state.",
            "",
            "Calibration and final release gates remain pending. Reported candidate standard "
            + "errors, tests and posterior summaries remain explicitly unmeasured at their declared "
            + "coordinates. Discovery, adaptive policies and broader arbitrary callback/cross-world "
            + "execution retain their separate release targets in ROADMAP.md.",
            "",
        ]
    glossary = GLOSSARY
    if RELEASE == "2.3":
        glossary = tuple(
            (
                term,
                meaning.replace(
                    "2.2 cells publish none.", "Unmeasured diagnostics grant no coverage license."
                ),
            )
            for term, meaning in GLOSSARY
        )
    out += [f"| {t} | {m} |" for t, m in glossary]
    out += ["", f"## {RELEASE}A cells", ""]
    for rec in (r for r in records if r["milestone"] == "A"):
        out += [entry_text(rec, present)]
    out += [f"## {RELEASE}B cells", ""]
    have_b = {r["work_package"] for r in records if r["milestone"] == "B"}
    for rec in (r for r in records if r["milestone"] == "B"):
        out += [entry_text(rec, present)]
    for pkg, (ws, title) in B_PACKAGES.items():
        if pkg not in have_b:
            out += [
                f"<!-- todo-cell: {pkg} -->",
                f"### {ws}: {title}",
                "",
                f"TODO ({pkg}): describe this cell's result with its record's claim once its "
                "record is frozen.",
                "",
            ]
    if any(r["milestone"] == "C" for r in records):
        out += [f"## {RELEASE}C cells", ""]
        for rec in (r for r in records if r["milestone"] == "C"):
            out += [entry_text(rec, present)]
    out += [
        "## Carried forward",
        "",
        "<!-- carried-forward: X7 -->",
        X7_LINE,
        "",
        "## Explicit refusals",
        "",
        " ".join(
            (
                "TODO: copy the refusal section from the support-matrix generator at the cut",
                "(`gate_docs_support_matrix.sh` requires it, and the licensed-block markers).",
            )
        ),
        "",
        "See the [support matrix](../support-matrix.md).",
        "",
    ]
    return "\n".join(out)


def blocks(text: str) -> tuple[dict[str, str], str]:
    """Marker blocks keyed '<kind>:<name>', and the text outside every block (the preamble)."""
    marks = list(MARK.finditer(text))
    heads = [m.start() for m in re.finditer(r"^## ", text, re.M)]
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        ends = [h for h in heads if h > m.start()]
        subs = [h.start() for h in re.finditer(r"^### ", text, re.M) if h.start() > m.start()]
        nxt = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        # an entry owns its own first `### ` heading; the next one starts another entry
        end = min([nxt, *ends[:1], *subs[1:2]])
        out[f"{m.group(1)}:{m.group(2).strip()}"] = text[m.start() : end]
    return out, text


def claim_words(body: str) -> set[str]:
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    found: set[str] = set(TOKEN.findall(body))
    found |= {w.lower().replace(" ", "_") for w in BARE.findall(body)}
    if PLAIN_NONE.search(body):
        found.add("none")
    if BARE_CALIBRATED.search(body) or STAT.search(body):
        found.add("calibrated")
    if NOMINAL.search(body):
        found.add("nominal")
    return found


WARNINGS: list[str] = []


def check(root: Path, notes_rel: str = NOTES, final: bool = False) -> list[str]:
    """Errors (nonempty fails). While the notes are a DRAFT, a record or package the skeleton does
    not yet cover is a warning in WARNINGS (regenerate with --write-draft), not an error."""
    errors: list[str] = []
    WARNINGS.clear()
    records = load(root, RECORDS).get("record", [])
    present = {r.get("id") for r in load(root, COVERAGE).get("record", [])}
    p = root / notes_rel
    if not p.is_file():
        return [f"{notes_rel} is missing (generate with --write-draft)"]
    text = p.read_text()
    lenient = DRAFT_MARK in text and not final
    stale = WARNINGS if lenient else errors
    cells, _ = blocks(text)
    # 1. glossary
    gloss = text.split("## How to read the claims", 1)
    if len(gloss) < 2:
        errors.append("missing the '## How to read the claims' glossary")
    else:
        g = gloss[1].split("\n## ", 1)[0]
        for term in (
            "point_only",
            "nominal",
            "calibrated",
            "structural_envelope",
            "assumption_range",
            "statistical interval",
        ):
            if term not in g:
                errors.append(f"glossary does not define {term!r}")
    # 2-3. one entry per record with exactly its effective claim
    ids = set()
    for rec in records:
        ids.add(rec["id"])
        key = f"cell:{rec['id']}"
        label = rec["id"]
        if rec.get("inference_claim") not in CLAIMS:
            errors.append(
                f"{label}: record inference_claim {rec.get('inference_claim')!r} is not in the vocabulary"
            )
            continue
        if key not in cells:
            stale.append(f"{label}: no '<!-- cell: {rec['id']} -->' entry in the notes")
            continue
        body = cells[key]
        want, pending = effective_claim(rec, present)
        lines = CLAIM_LINE.findall(body)
        if len(lines) != 1:
            errors.append(
                f"{label}: entry needs exactly one 'Claim: `word`' line, found {len(lines)}"
            )
        elif lines[0] != want:
            errors.append(
                f"{label}: Claim line says {lines[0]!r}, record claim is {rec['inference_claim']!r} "
                f"(effective {want!r}" + (", coverage records absent" if pending else "") + ")"
            )
        words = claim_words(body)
        if "nominal" in words:
            errors.append(
                f"{label}: entry uses 'nominal' affirmatively; 2.2 publishes no nominal-only interval"
            )
            words.discard("nominal")
        if words != {want}:
            extra = sorted(words - {want})
            errors.append(
                f"{label}: entry uses claim word(s) {extra or sorted(words)} but the cell's claim is "
                f"exactly {want!r}"
                + (
                    " (a calibrated record is point_only until its coverage records exist)"
                    if pending
                    else ""
                )
            )
        if pending and "withheld" not in body:
            errors.append(
                f"{label}: coverage records are absent; the entry must say the interval is withheld"
            )
        if want != "calibrated":
            promise = INTERVAL_PROMISE.search(re.sub(r"<!--.*?-->", "", body, flags=re.S))
            if promise:
                errors.append(
                    f"{label}: entry promises a '{promise.group(0)}' but the cell's claim is {want!r}; "
                    "a cell without a measured coverage record publishes no interval"
                )
        # 6-7. the entry restates bounds, guarantee and interval stance. An optional
        # internal status must be accurate, but need not appear in public notes.
        bounds_lines = BOUNDS_LINE.findall(body)
        if (numeric_bounds(rec) != "none declared" or bounds_lines) and (
            len(bounds_lines) != 1 or bounds_lines[0] != numeric_bounds(rec)
        ):
            stale.append(
                f"{label}: Bounds line {bounds_lines[:1]} differs from the record's "
                f"[{numeric_bounds(rec)}]"
            )
        statuses = STATUS_PART.findall(body)
        if statuses and statuses != [rec.get("status")]:
            stale.append(
                f"{label}: entry says record status {statuses or 'nothing'} but the record is "
                f"`{rec.get('status')}`"
            )
        guarantees = GUARANTEE_PART.findall(body)
        if guarantees != [rec["guarantee"]]:
            stale.append(
                f"{label}: entry's Guarantee {guarantees or 'line'} is not the record's "
                f"`{rec['guarantee']}`"
            )
        qual = complete_qualifier(rec["guarantee"])
        if qual is not None:
            if qual and not qualifier_regex(qual).search(body):
                stale.append(
                    f"{label}: guarantee `{rec['guarantee']}` prints 'complete' without its "
                    f"qualifier ({qual!r}) stated in the entry"
                )
            if not qual and "inherited" not in body:
                stale.append(
                    f"{label}: guarantee `{rec['guarantee']}` prints 'complete' without the "
                    "paper-inherited note"
                )
            if not licensed_uncertainty(rec) and not re.search(
                r"sampling interval not offered|sampling-uncertainty interval is not offered",
                body,
            ):
                stale.append(
                    f"{label}: a 'complete' guarantee must say the sampling interval is not offered "
                    "(no uncertainty route is licensed)"
                )
        if closed_uncertainty(rec) and not WITHHELD.search(body):
            stale.append(
                f"{label}: the record has a closed uncertainty route; the entry must say the "
                "interval is withheld / not offered"
            )
        if rec["inference_claim"] in (
            "assumption_range",
            "structural_envelope",
        ) and not NEVER_CI.search(body):
            stale.append(
                f"{label}: the {rec['inference_claim']} entry must say the range is never a "
                "confidence interval"
            )
    # 4. B packages
    have_b = {r["work_package"] for r in records if r.get("milestone") == "B"}
    for pkg, (ws, _) in B_PACKAGES.items():
        if pkg in have_b:
            continue
        # The conditional accelerator package has an explicit negative release
        # decision, not a missing statistical record. Keep the exception tied
        # to this package and its checked carryover statement.
        if RELEASE == "2.3" and pkg == "B5" and "carried-forward:X7" in cells:
            decision = root / "parity/2_3_x7_decision.md"
            if not decision.is_file():
                errors.append("B5 carryover lacks parity/2_3_x7_decision.md")
            continue
        if f"todo-cell:{pkg}" not in cells:
            stale.append(
                f"{RELEASE}B package {pkg} ({ws}) has neither a record entry nor a todo-cell marker"
            )
    for key in cells:
        kind, _, name = key.partition(":")
        if kind == "cell" and name not in ids:
            stale.append(f"entry for unknown record {name!r}")
    # 5. X7
    x7 = cells.get("carried-forward:X7", "")
    for need in (
        "X7",
        "GPU",
        "carried forward",
        "CPU-only",
        "benches/baselines/neural_crossfit.md",
    ):
        if need not in x7:
            errors.append(f"X7 carried-forward statement lacks {need!r}")
    if final:
        if DRAFT_MARK in text or re.search(r"\bDRAFT\b", text):
            errors.append("the notes are still marked DRAFT")
        if "todo-cell:" in text or re.search(r"\bTODO\b", text):
            errors.append("TODO markers remain in the notes")
    return errors


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    global RELEASE, RECORDS, NOTES, DRAFT_MARK, B_PACKAGES, X7_LINE
    if "--release" in argv:
        index = argv.index("--release")
        if index + 1 >= len(argv) or argv[index + 1] not in ("2.2", "2.3"):
            print("--release requires 2.2 or 2.3")
            return 2
        RELEASE = argv[index + 1]
        RECORDS = f"parity/promotion_{RELEASE.replace('.', '_')}.toml"
        NOTES = f"docs/release-notes/v{RELEASE}.0.md"
        DRAFT_MARK = f"<!-- draft:v{RELEASE}.0 -->"
        if RELEASE == "2.3":
            B_PACKAGES = {
                "B1": ("X4", "Checked transport with Bayesian models"),
                "B2": ("X8", "Theorem-scoped recovery and counterfactuals"),
                "B3": ("F17", "Sensitivity and mechanism discrepancy"),
                "B4": ("F21", "Joint treatment and mediation effects"),
                "B5": ("X7", "Conditional accelerator lane"),
            }
            X7_LINE = (
                "X7 (GPU lane) remains carried forward. The CPU-only whole-workload "
                "baseline (`benches/baselines/neural_crossfit.md`) supplies no "
                "same-host accelerator comparison; no GPU backend is available. "
                "The required measured gain and distribution gates remain open. "
                "See the explicit decision in `parity/2_3_x7_decision.md`."
            )
    notes = NOTES
    if "--notes" in argv:
        notes = argv[argv.index("--notes") + 1]
    if "--write-draft" in argv:
        records = load(ROOT, RECORDS).get("record", [])
        present = {r.get("id") for r in load(ROOT, COVERAGE).get("record", [])}
        p = ROOT / notes
        if p.is_file() and DRAFT_MARK not in p.read_text() and "--force" not in argv:
            print(f"{notes} is no longer marked DRAFT; refusing to overwrite (use --force)")
            return 1
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(draft(records, present))
        print(f"wrote draft {notes} ({len(records)} record entries)")
        return 0
    errors = check(ROOT, notes, final="--final" in argv)
    for w in WARNINGS:
        print(f"warning: stale draft: {w} (run --write-draft)")
    if errors:
        print("release claims FAILED:")
        for e in errors:
            print(f" - {e}")
        return 1
    print(f"release claims ok ({notes})")
    return 0


# ------------------------------------------------------------------------------ self-test
def _rec(
    i: str,
    claim: str,
    cov: list[str] | None = None,
    milestone: str = "A",
    wp: str = "A1",
    guarantee: str = "sound_incomplete",
    closed: bool = False,
    status: str = "promoted",
) -> dict:
    rec = {
        "id": f"2.2{milestone}.{i}.cell_{i.lower()}",
        "milestone": milestone,
        "work_package": wp,
        "status": status,
        "consumer_question": "Q?",
        "graph_class": "G.",
        "guarantee": guarantee,
        "inference_claim": claim,
        "coverage_records": cov or [],
        "bounds": {"max_observed": 12, "cancellation": True},
    }
    if closed:
        rec["routes"] = [
            {
                "name": "r",
                "stage": "uncertainty",
                "status": "closed",
                "reason_code": "cell_not_licensed",
            }
        ]
    elif claim == "calibrated":
        rec["routes"] = [{"name": "r", "stage": "uncertainty", "status": "licensed"}]
    return rec


def _toml(recs: list[dict]) -> str:
    out = ["version = 1", 'release = "2.2"']
    for r in recs:
        out += ["", "[[record]]"]
        for k, v in r.items():
            if isinstance(v, dict):
                inner = ", ".join(
                    f"{a} = {str(b).lower() if isinstance(b, bool) else b}" for a, b in v.items()
                )
                out.append(f"{k} = {{ {inner} }}")
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                rows = [
                    "{ " + ", ".join(f'{a} = "{b}"' for a, b in row.items()) + " }" for row in v
                ]
                out.append(f"{k} = [" + ", ".join(rows) + "]")
            elif isinstance(v, list):
                out.append(f"{k} = [" + ", ".join(f'"{x}"' for x in v) + "]")
            else:
                out.append(f'{k} = "{v}"')
    return "\n".join(out) + "\n"


def self_test() -> int:
    global RELEASE, B_PACKAGES
    failures: list[str] = []
    recs = [
        _rec("X1", "calibrated", ["cov.x1"]),
        _rec("X2", "structural_envelope"),
        _rec("X5", "point_only"),
        _rec("X9", "none"),
    ]
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "parity").mkdir()
        (root / "docs/release-notes").mkdir(parents=True)
        (root / RECORDS).write_text(_toml(recs))
        (root / COVERAGE).write_text("")
        notes = root / NOTES

        def run(mutate=None, final=False, cov=False, records=None) -> list[str]:
            if records is not None:
                (root / RECORDS).write_text(_toml(records))
            (root / COVERAGE).write_text('[[record]]\nid = "cov.x1"\n' if cov else "")
            text = draft(load(root, RECORDS)["record"], {"cov.x1"} if cov else set())
            if mutate:
                text = mutate(text)
            notes.write_text(text)
            return check(root, final=final)

        def expect(name: str, errs: list[str], needle: str | None) -> None:
            if needle is None:
                if errs:
                    failures.append(f"{name}: expected clean, got {errs[:3]}")
            elif not any(needle in e for e in errs):
                failures.append(f"{name}: expected {needle!r}, got {errs[:3]}")

        expect("clean draft", run(), None)
        expect("final rejects draft", run(final=True), "DRAFT")
        expect(
            "wrong claim word",
            run(lambda s: s.replace("Claim: `structural_envelope`", "Claim: `point_only`")),
            "Claim line says",
        )
        expect(
            "calibrated prose while coverage absent",
            run(
                lambda s: s.replace(
                    "the result is the point only.",
                    "the result is a calibrated interval.",
                )
            ),
            "calibrated",
        )
        expect(
            "statistical interval while coverage absent",
            run(lambda s: s.replace("the point only.", "a statistical interval.")),
            "X1",
        )
        expect(
            "calibrated claim line while coverage absent",
            run(lambda s: s.replace("Claim: `point_only`", "Claim: `calibrated`", 1)),
            "effective",
        )
        expect("calibrated ok once coverage exists", run(cov=True), None)
        closed_record = _rec("X1", "calibrated", ["cov.x1"], closed=True)
        expect(
            "measured but closed interval stays point only",
            run(cov=True, records=[closed_record, *recs[1:]]),
            None,
        )
        expect(
            "closed interval cannot advertise calibration",
            run(
                cov=True,
                records=[closed_record, *recs[1:]],
                mutate=lambda s: s.replace("Claim: `point_only`", "Claim: `calibrated`", 1),
            ),
            "Claim line says",
        )
        expect(
            "calibrated entry wrongly point_only once coverage exists",
            run(
                cov=True,
                records=recs,
                mutate=lambda s: s.replace("Claim: `calibrated`", "Claim: `point_only`"),
            ),
            "Claim line says",
        )
        expect(
            "second claim word in an entry",
            run(
                lambda s: s.replace(
                    "### X5: cell x5",
                    "### X5: cell x5\n\nAlso a `structural_envelope`.",
                )
            ),
            "X5",
        )
        expect(
            "nominal affirmative",
            run(
                lambda s: s.replace(
                    "### X5: cell x5", "### X5: cell x5\n\nThe nominal interval covers."
                )
            ),
            "nominal",
        )
        expect(
            "nominal negated is fine",
            run(
                lambda s: s.replace(
                    "### X5: cell x5",
                    "### X5: cell x5\n\nThere is no nominal interval here.",
                )
            ),
            None,
        )
        expect(
            "a point_only entry promising a bootstrap interval in prose fails",
            run(
                lambda s: s.replace(
                    "### X5: cell x5",
                    "### X5: cell x5\n\nA 95% bootstrap interval is reported.",
                )
            ),
            "promises a 'bootstrap interval'",
        )
        expect(
            "a point_only entry promising a credible interval in prose fails",
            run(
                lambda s: s.replace(
                    "### X5: cell x5",
                    "### X5: cell x5\n\nBayesian inference publishes credible intervals.",
                )
            ),
            "promises a 'credible intervals'",
        )
        expect(
            "a negated or withheld interval mention is fine",
            run(
                lambda s: s.replace(
                    "### X5: cell x5",
                    "### X5: cell x5\n\nThis is not a confidence interval; the bootstrap interval "
                    "is withheld and no credible interval is offered.",
                )
            ),
            None,
        )
        nodraft = lambda s: s.replace(DRAFT_MARK, "")  # noqa: E731
        expect(
            "missing entry (not a draft)",
            run(lambda s: nodraft(s).replace("<!-- cell: 2.2A.X9.cell_x9 -->", "")),
            "X9",
        )
        expect(
            "missing entry in a draft is only a warning",
            run(lambda s: s.replace("<!-- cell: 2.2A.X9.cell_x9 -->", "")),
            None,
        )
        if not any("X9" in w for w in WARNINGS):
            failures.append("a draft missing an entry must warn")
        expect(
            "glossary term missing",
            run(lambda s: s.replace("assumption_range", "assume")),
            "glossary",
        )
        expect("x7 line missing", run(lambda s: s.replace("CPU-only", "cpu")), "X7")
        expect(
            "todo package marker missing",
            run(lambda s: nodraft(s).replace("<!-- todo-cell: B3 -->", "")),
            "B3",
        )
        expect(
            "todo markers fail final",
            run(
                final=True,
                mutate=lambda s: s.replace(DRAFT_MARK, "").replace("DRAFT", "x"),
            ),
            "TODO",
        )
        expect(
            "b record needs its entry",
            run(
                records=[*recs, _rec("X3", "assumption_range", None, "B", "B3")],
                mutate=lambda s: nodraft(s).replace("<!-- cell: 2.2B.X3.cell_x3 -->", ""),
            ),
            "X3",
        )
        expect(
            "b record entry generated",
            run(records=[*recs, _rec("X3", "assumption_range", None, "B", "B3")]),
            None,
        )

        # ---- D5: prose claim words, plain none, bounds/optional status/guarantee, interval stance
        expect(
            "'structural envelope' with a space is a claim word",
            run(
                lambda s: s.replace("### X5: cell x5", "### X5: cell x5\n\nA structural envelope.")
            ),
            "X5",
        )
        expect(
            "'assumption range' with a space is a claim word",
            run(lambda s: s.replace("### X5: cell x5", "### X5: cell x5\n\nThe assumption range.")),
            "X5",
        )
        expect(
            "a plain `none` claim in another entry is detected",
            run(lambda s: s.replace("### X5: cell x5", "### X5: cell x5\n\nThe claim is none.")),
            "X5",
        )
        expect(
            "a plain Claim: none line (no backticks) is read",
            run(lambda s: s.replace("Claim: `none`", "Claim: none")),
            None,
        )
        expect(
            "a plain Claim: none line for a non-none record fails",
            run(lambda s: s.replace("Claim: `point_only`", "Claim: none", 1)),
            "Claim line says",
        )
        if claim_words("Bounds: none declared.") or claim_words("publishes none."):
            failures.append("a bare 'none' outside a claim phrase must not be a claim word")
        expect(
            "bounds line edited (not a draft)",
            run(lambda s: nodraft(s).replace("max_observed = 12", "max_observed = 13", 1)),
            "Bounds line",
        )
        expect(
            "bounds line edited (final)",
            run(
                final=True,
                mutate=lambda s: (
                    s.replace(DRAFT_MARK, "")
                    .replace("DRAFT", "x")
                    .replace("max_observed = 12", "max_observed = 13", 1)
                ),
            ),
            "Bounds line",
        )
        expect(
            "bounds line edited (draft: only a warning)",
            run(lambda s: s.replace("max_observed = 12", "max_observed = 13", 1)),
            None,
        )
        if not any("Bounds line" in w for w in WARNINGS):
            failures.append("a stale bounds line in a draft must warn")
        expect(
            "status edited (not a draft)",
            run(
                lambda s: nodraft(s).replace(
                    "- Guarantee: `sound_incomplete`.",
                    "- Guarantee: `sound_incomplete`; record status `frozen`.",
                    1,
                )
            ),
            "record status",
        )
        expect(
            "guarantee edited (not a draft)",
            run(lambda s: nodraft(s).replace("`sound_incomplete`", "`sound_complete`", 1)),
            "Guarantee",
        )
        closed_recs = [
            *recs,
            _rec(
                "X3",
                "assumption_range",
                None,
                "B",
                "B3",
                "exact_range_complete_within_declared_contamination_class",
                closed=True,
            ),
        ]
        expect(
            "closed-route assumption_range record: generated entry is clean",
            run(records=closed_recs),
            None,
        )
        expect(
            "closed uncertainty route without a withheld sentence",
            run(
                records=closed_recs,
                mutate=lambda s: (
                    nodraft(s).replace("withheld", "shown").replace("not offered", "x")
                ),
            ),
            "closed uncertainty route",
        )
        expect(
            "assumption_range entry without the never-a-CI statement",
            run(
                records=closed_recs,
                mutate=lambda s: nodraft(s).replace("is never a confidence interval", "is wide"),
            ),
            "never a confidence interval",
        )
        expect(
            "'complete' guarantee printed without its qualifier",
            run(
                records=closed_recs,
                mutate=lambda s: nodraft(s).replace(
                    "complete only within the declared contamination class", "complete"
                ),
            ),
            "qualifier",
        )
        expect(
            "'complete' guarantee without 'sampling interval not offered'",
            run(
                records=closed_recs,
                mutate=lambda s: nodraft(s).replace("not offered", "on offer"),
            ),
            "sampling interval is not offered",
        )
        expect(
            "a record-carried qualifier may say 'the'",
            run(
                records=closed_recs,
                mutate=lambda s: nodraft(s).replace("within the declared", "within declared"),
            ),
            None,
        )
        expect(
            "a carried_forward record's entry is generated and checked like any other",
            run(
                records=[
                    *recs,
                    _rec(
                        "X3",
                        "assumption_range",
                        None,
                        "B",
                        "B3",
                        status="carried_forward",
                        closed=True,
                    ),
                ]
            ),
            None,
        )
        expect(
            "a structural_envelope entry without the never-a-CI statement",
            run(mutate=lambda s: nodraft(s).replace("is never a confidence interval", "is wide")),
            "never a confidence interval",
        )
        composition_record = _rec("C1", "point_only", milestone="C", wp="C1")
        expect(
            "composition records have generated and checked entries",
            run(records=[*recs, composition_record]),
            None,
        )
        expect(
            "composition record prose cannot invent an interval",
            run(
                records=[*recs, composition_record],
                mutate=lambda text: nodraft(text).replace(
                    f"<!-- cell: {composition_record['id']} -->",
                    f"<!-- cell: {composition_record['id']} -->\nA confidence interval is reported.",
                ),
            ),
            "promises",
        )
        (root / RECORDS).write_text(_toml(recs))
        saved_release, saved_packages = RELEASE, B_PACKAGES
        try:
            RELEASE = "2.3"
            B_PACKAGES = {"B5": ("X7", "Conditional accelerator lane")}
            carryover_notes = draft(recs, {"cov.x1"})
            begin = carryover_notes.index("<!-- todo-cell: B5 -->")
            end = carryover_notes.index("## Carried forward", begin)
            carryover_notes = carryover_notes[:begin] + carryover_notes[end:]
            carryover_notes = (
                carryover_notes.replace(DRAFT_MARK, "")
                .replace("DRAFT", "preparation")
                .replace("TODO", "pending")
            )
            notes.write_text(carryover_notes)
            (root / COVERAGE).write_text('[[record]]\nid = "cov.x1"\n')
            expect(
                "accelerator carryover requires its decision",
                check(root, final=True),
                "B5 carryover lacks",
            )
            (root / "parity/2_3_x7_decision.md").write_text("X7 remains deferred.\n")
            expect("explicit accelerator carryover can finalize", check(root, final=True), None)
            notes.write_text(carryover_notes.replace("<!-- carried-forward: X7 -->", ""))
            expect("accelerator carryover requires its marker", check(root, final=True), "B5")
            B_PACKAGES["B1"] = ("X4", "Transport")
            notes.write_text(carryover_notes)
            expect(
                "accelerator exception does not waive another package",
                check(root, final=True),
                "B1",
            )
        finally:
            RELEASE, B_PACKAGES = saved_release, saved_packages
        notes.write_text("# x\n")
        if not check(root):
            failures.append("a notes file without entries must fail")
    if failures:
        print("check_release_claims self-test FAILED:")
        for f in failures:
            print(f" - {f}")
        return 1
    print("check_release_claims self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
