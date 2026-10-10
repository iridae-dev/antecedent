#!/usr/bin/env python3
"""Release closure: the notes describe every registry cell with its real claim.

The claim vocabulary is the one in the header of parity/promotion.toml (`inference_claim`):

  point_only           exact or plug-in point, no interval
  calibrated           interval with a coverage record at its own coordinate
  structural_envelope  range over supplied structural scenarios; not a CI
  assumption_range     range over a declared sensitivity set; not a CI
  none                 no numerical claim (identification/inspection only)

plus the two release-notes phrases `nominal` (an interval with no coverage record at its own
coordinate, `estimator_grid_not_measured`; a release ships none) and `statistical interval` (the
reader-facing name of a `calibrated` interval).

For the notes of the documented release (docs/release-notes/preparation.toml `target_version`, else
the workspace version), docs/release-notes/v<version>.md, or --notes PATH, this checks:

  1. the notes define all six terms in a glossary (outside the cell entries);
  2. every record of parity/promotion.toml has one entry, opened by a
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
  4. every `<!-- carried-forward: <name> -->` block says it is carried forward, and a GPU lane
     block states its rationale (CPU-only baseline, `benches/baselines/neural_crossfit.md`);
  6. the entry's `Bounds:` line equals the record's numeric bounds, any stated `record status`
     equals the record's status, and its `Guarantee:` string is the record's (stale otherwise: a warning while
     the notes are a DRAFT, an error outside draft mode and in `--final`);
  7. a record with a closed uncertainty route says its interval is `withheld` / `not offered`; an
     `assumption_range` or `structural_envelope` record says the range is never a confidence
     interval; a guarantee with a `complete` token (e.g.
     `exact_range_complete_within_declared_contamination_class`) is never printed without the
     record's qualifier (`within the declared contamination class`) stated beside it, and
     `sampling interval not offered` when no uncertainty route is licensed (rules 6-7: stale, as above).

Release notes of earlier releases are checked for every record they carry, with the exact
restatement rules of 6 (Bounds line, status, Guarantee string, `complete` qualifier) waived: the
registry's wording moves on, shipped notes do not. Every record must have an entry in some notes
file.

Claim words are found in prose as well as backticks: `structural envelope` / `assumption range` (with
a space), and a plain `none` after the word claim (`Claim: none`). The `Claim:` line may omit the
backticks.

`--final` additionally fails on any DRAFT or TODO marker (release candidate). `--write-draft`
(re)generates the draft skeleton from the records; it refuses to overwrite a notes file that is
no longer marked draft unless `--force`.

    python3 scripts/check_release_claims.py                 # check the draft/notes
    python3 scripts/check_release_claims.py --final         # release candidate: no draft, no TODO
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
RECORDS = "parity/promotion.toml"
COVERAGE = "parity/coverage_records.toml"

CLAIMS = ("point_only", "calibrated", "structural_envelope", "assumption_range", "none")
DRAFT_MARK_RE = re.compile(r"<!-- draft:v[0-9][^ ]* -->")


def workspace_version(root: Path) -> str:
    """The version of the workspace under `root` (Cargo.toml `[workspace.package]`)."""
    return load(root, "Cargo.toml")["workspace"]["package"]["version"]


def documentation_version(root: Path) -> str:
    """The release being documented: docs/release-notes/preparation.toml `target_version` while a
    release is being prepared (as scripts/generate_support_matrix_docs.py reads it), else the
    workspace version."""
    preparation = load(root, "docs/release-notes/preparation.toml")
    target = preparation.get("target_version")
    if isinstance(target, str) and re.fullmatch(r"\d+\.\d+\.\d+", target):
        return target
    return workspace_version(root)


def notes_for(root: Path) -> str:
    return f"docs/release-notes/v{documentation_version(root)}.md"


def draft_mark(version: str) -> str:
    return f"<!-- draft:v{version} -->"


GLOSSARY = (
    (
        "`point_only`",
        "An exact or plug-in point estimate. No interval of any kind is claimed.",
    ),
    (
        "`nominal`",
        "An interval whose coverage was never measured at its own coordinate "
        "(`estimator_grid_not_measured`). Cells publish none.",
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

TOKEN = re.compile(r"`(point_only|calibrated|structural_envelope|assumption_range|none)`")
BARE = re.compile(r"\b(point_only|structural[_ ]envelope|assumption[_ ]range)\b", re.I)
# a plain `none` counts as a claim word only right after the word claim ("Claim: none")
PLAIN_NONE = re.compile(r"\bclaims?\b[`:]?\s*(?:is |of )?\*{0,2}`?none\b", re.I)
BARE_CALIBRATED = re.compile(
    r"(?<![\w-])(?<!un)(?<!not )(?<!not yet )(?<!until )(?<!pending )calibrated\b", re.I
)
STAT = re.compile(r"(?<!not a )(?<!no )statistical interval", re.I)
NOMINAL = re.compile(r"(?<!no )(?<!not )(?<!never )(?<!any )\bnominal\b", re.I)
MARK = re.compile(r"<!-- (cell|carried-forward): ([^>]*?) -->")
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
    name = rec["id"].split(".", 1)[-1].replace("_", " ")
    claim, pending = effective_claim(rec, present)
    lines = [
        f"<!-- cell: {rec['id']} -->",
        f"### {name}",
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
            "coverage records are measured; the result is the point only."
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


def draft(records: list[dict], present: set[str], version: str) -> str:
    out = [
        f"# Antecedent {version}",
        "",
        draft_mark(version),
        " ".join(
            (
                "> **DRAFT.** Generated by `scripts/check_release_claims.py --write-draft` from",
                f"`{RECORDS}`. Not a release statement until the release candidate: the entries show",
                "each cell's claim as the records stand today. Regenerate, then hand-edit prose,",
                "and run `check_release_claims.py --final`.",
            )
        ),
        "",
        "## How to read the claims",
        "",
        "| Term | Meaning |",
        "| --- | --- |",
    ]
    out += [f"| {t} | {m} |" for t, m in GLOSSARY]
    out += ["", "## Cells", ""]
    for rec in records:
        out += [entry_text(rec, present)]
    out += [
        "## Carried forward",
        "",
        "TODO: list every cell carried forward with its reason, each opened by a",
        "`carried-forward` marker comment (the checker requires each block to say it is carried forward).",
        "",
        "## Explicit refusals",
        "",
        " ".join(
            (
                "TODO: copy the refusal section from the support-matrix generator at the release",
                "candidate (`gate_docs_support_matrix.sh` requires it, and the licensed-block markers).",
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


def check_notes(
    root: Path, notes_rel: str, final: bool = False, restate: bool = True
) -> tuple[list[str], set[str]]:
    """(errors, ids of the records the notes carry an entry for). Errors nonempty fails. While the
    notes are a DRAFT every record must have an entry, and one the skeleton does not yet cover is
    a warning in WARNINGS (regenerate with --write-draft), not an error. Shipped notes are checked
    for the records they carry; `check` joins the files. `restate=False` (notes of an earlier
    release, whose wording the registry has since moved past) skips the exact restatement of the
    Bounds line, record status, Guarantee string and its `complete` qualifier; every claim-word and
    interval-promise rule still applies."""
    errors: list[str] = []
    registry = load(root, RECORDS)
    records = registry.get("record", [])
    present = {r.get("id") for r in load(root, COVERAGE).get("record", [])}
    p = root / notes_rel
    if not p.is_file():
        return [f"{notes_rel} is missing (generate with --write-draft)"], set()
    text = p.read_text()
    lenient = DRAFT_MARK_RE.search(text) is not None and not final
    stale = WARNINGS if lenient else errors
    cells, _ = blocks(text)
    noted = {key.split(":", 1)[1] for key in cells if key.startswith("cell:")}
    if not lenient:
        records = [rec for rec in records if rec.get("id") in noted]
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
                f"{label}: entry uses 'nominal' affirmatively; a release publishes no nominal-only interval"
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
        if restate:
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
    for key in cells:
        kind, _, name = key.partition(":")
        if kind == "cell" and name not in ids:
            stale.append(f"entry for unknown record {name!r}")
    # 4. carried-forward statements
    for key, body in cells.items():
        kind, _, name = key.partition(":")
        if kind != "carried-forward":
            continue
        if "carried forward" not in body:
            errors.append(f"carried-forward statement {name!r} does not say it is carried forward")
        if "GPU" in body:
            for need in ("CPU-only", "benches/baselines/neural_crossfit.md"):
                if need not in body:
                    errors.append(f"carried-forward statement {name!r} lacks {need!r}")
    if final:
        if DRAFT_MARK_RE.search(text) or re.search(r"\bDRAFT\b", text):
            errors.append("the notes are still marked DRAFT")
        if re.search(r"\bTODO\b", text):
            errors.append("TODO markers remain in the notes")
    return errors, noted


def check(root: Path, notes_rel: str | None = None, final: bool = False) -> list[str]:
    """Errors over the release notes of the tree: every record of parity/promotion.toml has an
    entry in some release-notes file and each entry states its record's real claim. `notes_rel`
    checks that one file only. `final` holds the notes of the documented version to the release
    candidate standard (not a draft, no TODO) and requires them to exist."""
    WARNINGS.clear()
    if notes_rel is not None:
        return check_notes(root, notes_rel, final)[0]
    current = notes_for(root)
    files = sorted(
        str(p.relative_to(root))
        for p in (root / "docs/release-notes").glob("v*.md")
        if "<!-- cell:" in p.read_text() or str(p.relative_to(root)) == current
    )
    errors: list[str] = []
    noted: set[str] = set()
    drafted = False
    for rel in files:
        file_errors, ids = check_notes(root, rel, final and rel == current, restate=rel == current)
        errors += [f"{rel}: {e}" for e in file_errors]
        noted |= ids
        drafted = drafted or DRAFT_MARK_RE.search((root / rel).read_text()) is not None
    if not (root / current).is_file():
        records = load(root, RECORDS).get("record", [])
        uncovered = [r["id"] for r in records if r["id"] not in noted]
        if final or uncovered:
            errors.append(f"{current} is missing (generate with --write-draft)")
    elif not drafted:
        for rec in load(root, RECORDS).get("record", []):
            if rec["id"] not in noted:
                errors.append(f"{rec['id']}: no '<!-- cell: {rec['id']} -->' entry in any release notes")
    return errors


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    explicit = argv[argv.index("--notes") + 1] if "--notes" in argv else None
    notes = explicit or notes_for(ROOT)
    if "--write-draft" in argv:
        records = load(ROOT, RECORDS).get("record", [])
        present = {r.get("id") for r in load(ROOT, COVERAGE).get("record", [])}
        p = ROOT / notes
        if p.is_file() and not DRAFT_MARK_RE.search(p.read_text()) and "--force" not in argv:
            print(f"{notes} is no longer marked DRAFT; refusing to overwrite (use --force)")
            return 1
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(draft(records, present, documentation_version(ROOT)))
        print(f"wrote draft {notes} ({len(records)} record entries)")
        return 0
    errors = check(ROOT, explicit, final="--final" in argv)
    for w in WARNINGS:
        print(f"warning: stale draft: {w} (run --write-draft)")
    if errors:
        print("release claims FAILED:")
        for e in errors:
            print(f" - {e}")
        return 1
    print(f"release claims ok ({explicit or 'docs/release-notes'})")
    return 0


# ------------------------------------------------------------------------------ self-test
def _rec(
    i: str,
    claim: str,
    cov: list[str] | None = None,
    guarantee: str = "sound_incomplete",
    closed: bool = False,
    status: str = "promoted",
) -> dict:
    rec = {
        "id": f"synth.cell_{i.lower()}",
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
    out = ["version = 1"]
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
    failures: list[str] = []
    recs = [
        _rec("a1", "calibrated", ["cov.a1"]),
        _rec("a2", "structural_envelope"),
        _rec("a5", "point_only"),
        _rec("a9", "none"),
    ]
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "parity").mkdir()
        (root / "docs/release-notes").mkdir(parents=True)
        (root / RECORDS).write_text(_toml(recs))
        (root / COVERAGE).write_text("")
        (root / "Cargo.toml").write_text('[workspace.package]\nversion = "9.9.9"\n')
        notes = root / notes_for(root)
        DRAFT_MARK = draft_mark("9.9.9")

        def run(mutate=None, final=False, cov=False, records=None) -> list[str]:
            if records is not None:
                (root / RECORDS).write_text(_toml(records))
            (root / COVERAGE).write_text('[[record]]\nid = "cov.a1"\n' if cov else "")
            text = draft(load(root, RECORDS)["record"], {"cov.a1"} if cov else set(), "9.9.9")
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
            "cell_a1",
        )
        expect(
            "calibrated claim line while coverage absent",
            run(lambda s: s.replace("Claim: `point_only`", "Claim: `calibrated`", 1)),
            "effective",
        )
        expect("calibrated ok once coverage exists", run(cov=True), None)
        closed_record = _rec("a1", "calibrated", ["cov.a1"], closed=True)
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
                    "### cell a5",
                    "### cell a5\n\nAlso a `structural_envelope`.",
                )
            ),
            "cell_a5",
        )
        expect(
            "nominal affirmative",
            run(
                lambda s: s.replace(
                    "### cell a5", "### cell a5\n\nThe nominal interval covers."
                )
            ),
            "nominal",
        )
        expect(
            "nominal negated is fine",
            run(
                lambda s: s.replace(
                    "### cell a5",
                    "### cell a5\n\nThere is no nominal interval here.",
                )
            ),
            None,
        )
        expect(
            "a point_only entry promising a bootstrap interval in prose fails",
            run(
                lambda s: s.replace(
                    "### cell a5",
                    "### cell a5\n\nA 95% bootstrap interval is reported.",
                )
            ),
            "promises a 'bootstrap interval'",
        )
        expect(
            "a point_only entry promising a credible interval in prose fails",
            run(
                lambda s: s.replace(
                    "### cell a5",
                    "### cell a5\n\nBayesian inference publishes credible intervals.",
                )
            ),
            "promises a 'credible intervals'",
        )
        expect(
            "a negated or withheld interval mention is fine",
            run(
                lambda s: s.replace(
                    "### cell a5",
                    "### cell a5\n\nThis is not a confidence interval; the bootstrap interval "
                    "is withheld and no credible interval is offered.",
                )
            ),
            None,
        )
        nodraft = lambda s: s.replace(DRAFT_MARK, "")  # noqa: E731
        expect(
            "missing entry (not a draft)",
            run(lambda s: nodraft(s).replace("<!-- cell: synth.cell_a9 -->", "")),
            "cell_a9",
        )
        expect(
            "missing entry in a draft is only a warning",
            run(lambda s: s.replace("<!-- cell: synth.cell_a9 -->", "")),
            None,
        )
        if not any("cell_a9" in w for w in WARNINGS):
            failures.append("a draft missing an entry must warn")
        expect(
            "glossary term missing",
            run(lambda s: s.replace("assumption_range", "assume")),
            "glossary",
        )
        carried = "\n## Carried forward\n\n<!-- carried-forward: gpu_lane -->\n"
        gpu_ok = (
            "The GPU lane is carried forward: the CPU-only baseline "
            "(`benches/baselines/neural_crossfit.md`) records no accelerator measurement.\n"
        )
        expect("a carried-forward GPU statement is accepted", run(lambda s: nodraft(s) + carried + gpu_ok), None)
        expect(
            "a carried-forward statement must say it is carried forward",
            run(lambda s: nodraft(s) + carried + "The GPU lane is deferred.\n"),
            "carried forward",
        )
        expect(
            "a GPU carried-forward statement needs its baseline rationale",
            run(lambda s: nodraft(s) + carried + gpu_ok.replace("CPU-only", "cpu")),
            "CPU-only",
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
            "an added record needs its entry",
            run(
                records=[*recs, _rec("a3", "assumption_range")],
                mutate=lambda s: nodraft(s).replace("<!-- cell: synth.cell_a3 -->", ""),
            ),
            "cell_a3",
        )
        expect(
            "an added record entry is generated",
            run(records=[*recs, _rec("a3", "assumption_range")]),
            None,
        )

        # ---- D5: prose claim words, plain none, bounds/optional status/guarantee, interval stance
        expect(
            "'structural envelope' with a space is a claim word",
            run(
                lambda s: s.replace("### cell a5", "### cell a5\n\nA structural envelope.")
            ),
            "cell_a5",
        )
        expect(
            "'assumption range' with a space is a claim word",
            run(lambda s: s.replace("### cell a5", "### cell a5\n\nThe assumption range.")),
            "cell_a5",
        )
        expect(
            "a plain `none` claim in another entry is detected",
            run(lambda s: s.replace("### cell a5", "### cell a5\n\nThe claim is none.")),
            "cell_a5",
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
                "a3",
                "assumption_range",
                None,
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
                        "a3",
                        "assumption_range",
                        None,
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
        composition_record = _rec("c1", "point_only")
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
