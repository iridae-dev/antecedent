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

Cells and Refusals are generated marker blocks. The checker compares those blocks to
the renderer. It does not read claim lines, glossary headings, or `## Explicit refusals` out of
hand-written prose, so rewording a heading in shipped notes does not fail the check.

  <!-- generated:release-claims:cells:begin --> ... <!-- generated:release-claims:cells:end -->
  <!-- generated:release-claims:refusals:begin --> ... <!-- generated:release-claims:refusals:end -->

The current draft (no snapshot yet) renders from the live registry: records not already frozen
in an earlier snapshot. At freeze, `--snapshot` writes `parity/release_snapshots/vX.Y.Z.toml`.
Once that file exists, the blocks render from it and the live registry is not consulted.
Shipped notes with no generated blocks are not validated at all — not against the snapshot, and
not against live data. Hand prose outside the markers is ignored.

`--final` fails on any DRAFT or TODO marker in the notes being released. `--write-draft` inserts
or replaces only the two marker blocks; it refuses to touch a notes file that is no longer
marked draft unless `--force`.

    python3 scripts/check_release_claims.py                 # check generated blocks
    python3 scripts/check_release_claims.py --final         # release candidate: no draft, no TODO
    python3 scripts/check_release_claims.py --write-draft   # insert or replace the marker blocks
    python3 scripts/check_release_claims.py --snapshot      # freeze the current draft's registry projection
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

from _repo import REPO_ROOT as REPO

ROOT = Path(os.environ.get("RELEASE_CLAIMS_ROOT", REPO))
RECORDS = "parity/promotion.toml"
COVERAGE = "parity/coverage_records.toml"
SNAPSHOT_DIR = "parity/release_snapshots"
CELLS_BEGIN = "<!-- generated:release-claims:cells:begin -->"
CELLS_END = "<!-- generated:release-claims:cells:end -->"
REFUSALS_BEGIN = "<!-- generated:release-claims:refusals:begin -->"
REFUSALS_END = "<!-- generated:release-claims:refusals:end -->"
NOTES_VERSION = re.compile(r"v(\d+\.\d+\.\d+)\.md$")

CLAIMS = ("point_only", "calibrated", "structural_envelope", "assumption_range", "none")
DRAFT_MARK_RE = re.compile(r"<!-- draft:v[0-9][^ ]* -->")


def workspace_version(root: Path) -> str:
    """The version of the workspace under `root` (Cargo.toml `[workspace.package]`)."""
    return load(root, "Cargo.toml")["workspace"]["package"]["version"]


def documentation_version(root: Path) -> str:
    """The release being documented.

    ``docs/release-notes/preparation.toml`` is optional. While a release is being
    prepared its ``target_version`` may be ahead of the package (the same read as
    scripts/generate_support_matrix_docs.py). A missing file, or a file with no
    valid ``target_version``, falls back to the workspace version and is not an
    error.
    """
    preparation = load(root, "docs/release-notes/preparation.toml")
    target = preparation.get("target_version")
    if isinstance(target, str) and re.fullmatch(r"\d+\.\d+\.\d+", target):
        return target
    return workspace_version(root)


def notes_for(root: Path) -> str:
    return f"docs/release-notes/v{documentation_version(root)}.md"


def draft_mark(version: str) -> str:
    return f"<!-- draft:v{version} -->"


def notes_version(rel: str) -> str | None:
    match = NOTES_VERSION.search(rel.replace("\\", "/"))
    return match.group(1) if match else None


def snapshot_rel(version: str) -> str:
    return f"{SNAPSHOT_DIR}/v{version}.toml"


def toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def project_record(rec: dict) -> dict:
    """The claim fields the Cells block renders. Theorems and surfaces stay in the registry."""
    bounds = {
        k: v
        for k, v in (rec.get("bounds") or {}).items()
        if isinstance(v, int) and not isinstance(v, bool)
    }
    routes = []
    for route in rec.get("routes") or []:
        if route.get("stage") != "uncertainty":
            continue
        row = {
            "name": str(route.get("name") or "uncertainty"),
            "stage": "uncertainty",
            "status": str(route.get("status") or ""),
        }
        if route.get("reason_code"):
            row["reason_code"] = str(route["reason_code"])
        routes.append(row)
    return {
        "id": rec["id"],
        "status": str(rec.get("status") or ""),
        "consumer_question": str(rec.get("consumer_question") or ""),
        "graph_class": str(rec.get("graph_class") or ""),
        "guarantee": str(rec.get("guarantee") or ""),
        "inference_claim": rec.get("inference_claim"),
        "coverage_records": list(rec.get("coverage_records") or []),
        "bounds": bounds,
        "routes": routes,
    }


def dump_snapshot(version: str, records: list[dict], present: set[str], queries: list[str]) -> str:
    lines = [
        f'version = "{version}"',
        "# Frozen claim projection for this release. Cells and Refusals render from here.",
        "# Shipped notes are not compared to the live registry.",
        "coverage_present = [" + ", ".join(f'"{toml_escape(x)}"' for x in sorted(present)) + "]",
        "zero_cell_queries = [" + ", ".join(f'"{toml_escape(x)}"' for x in queries) + "]",
    ]
    for rec in records:
        lines += ["", "[[record]]"]
        for key in (
            "id",
            "status",
            "consumer_question",
            "graph_class",
            "guarantee",
            "inference_claim",
        ):
            lines.append(f'{key} = "{toml_escape(str(rec.get(key) or ""))}"')
        lines.append(
            "coverage_records = ["
            + ", ".join(f'"{toml_escape(x)}"' for x in rec.get("coverage_records") or [])
            + "]"
        )
        bounds = rec.get("bounds") or {}
        if bounds:
            inner = ", ".join(f"{k} = {v}" for k, v in bounds.items())
            lines.append(f"bounds = {{ {inner} }}")
        else:
            lines.append("bounds = {}")
        rows = []
        for route in rec.get("routes") or []:
            parts = [
                f'name = "{toml_escape(route["name"])}"',
                'stage = "uncertainty"',
                f'status = "{toml_escape(route["status"])}"',
            ]
            if route.get("reason_code"):
                parts.append(f'reason_code = "{toml_escape(route["reason_code"])}"')
            rows.append("{ " + ", ".join(parts) + " }")
        lines.append("routes = [" + ", ".join(rows) + "]")
    return "\n".join(lines) + "\n"


def load_snapshot(root: Path, version: str) -> dict | None:
    path = root / snapshot_rel(version)
    if not path.is_file():
        return None
    return tomllib.loads(path.read_text(encoding="utf-8"))


def snapshot_ids(root: Path, skip: str | None = None) -> set[str]:
    directory = root / SNAPSHOT_DIR
    if not directory.is_dir():
        return set()
    ids: set[str] = set()
    for path in directory.glob("v*.toml"):
        version = path.name[1:-5]
        if skip is not None and version == skip:
            continue
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        ids.update(rec["id"] for rec in data.get("record", []) if rec.get("id"))
    return ids


def live_projection(root: Path) -> tuple[list[dict], set[str]]:
    records = [project_record(rec) for rec in load_required(root, RECORDS).get("record", [])]
    # A record ID alone does not establish the native matcher's construction,
    # scope, boundary and attestation requirements. This projection carries no
    # authoritative eligibility basis, so withhold live calibrated claims until
    # such a basis is available. Frozen snapshots retain their recorded policy.
    load_required(root, COVERAGE)  # Still require the current registry to exist and parse.
    present: set[str] = set()
    return records, present


def zero_cell_queries(root: Path) -> list[str]:
    axes = load_required(root, "parity/support_axes.toml").get("queries") or []
    licensed = {
        cell.get("query") for cell in load_required(root, "parity/support_licensed.toml").get("cell", [])
    }
    return [query for query in axes if query not in licensed]


def source_for(
    root: Path, version: str, current: str
) -> tuple[list[dict], set[str], list[str]] | None:
    """(records, coverage ids, zero-cell queries) to render.

    A snapshot wins over the live registry. The unfrozen current release renders the live
    records that no earlier snapshot already froze. Shipped notes with no snapshot are not
    rendered from live data."""
    snap = load_snapshot(root, version)
    if snap is not None:
        return (
            list(snap.get("record") or []),
            set(snap.get("coverage_present") or []),
            list(snap.get("zero_cell_queries") or []),
        )
    if version != current:
        return None
    records, present = live_projection(root)
    frozen = snapshot_ids(root)
    return [rec for rec in records if rec["id"] not in frozen], present, zero_cell_queries(root)


def cells_body(records: list[dict], present: set[str]) -> str:
    if not records:
        return "No cells are new in this release.\n"
    return "".join(entry_text(rec, present) for rec in records)


def refusals_body(queries: list[str]) -> str:
    if not queries:
        return "No root query is without a licensed cell.\n"
    lines = ["Root queries with no licensed cell:", ""]
    lines += [f"- `{query}`" for query in queries]
    return "\n".join(lines) + "\n"


def extract_block(text: str, begin: str, end: str) -> str | None:
    if begin not in text or end not in text:
        return None
    inner = text.split(begin, 1)[1].split(end, 1)[0]
    if inner.startswith("\n"):
        inner = inner[1:]
    return inner


def wrap_block(begin: str, body: str, end: str) -> str:
    if not body.endswith("\n"):
        body += "\n"
    return f"{begin}\n{body}{end}\n"


def replace_block(text: str, begin: str, end: str, body: str) -> str:
    block = wrap_block(begin, body, end).rstrip("\n")
    pre, rest = text.split(begin, 1)
    _, post = rest.split(end, 1)
    return pre + block + post


def insert_block(text: str, begin: str, end: str, body: str, anchor: str, *, after: bool) -> str:
    if begin in text and end in text:
        return replace_block(text, begin, end, body)
    block = wrap_block(begin, body, end)
    idx = text.find(anchor)
    if idx < 0:
        return text.rstrip() + "\n\n" + block
    if after:
        nl = text.find("\n", idx)
        if nl < 0:
            return text + "\n" + block
        return text[: nl + 1] + "\n" + block + text[nl + 1 :]
    return text[:idx] + block + "\n" + text[idx:]


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


def load(root: Path, rel: str) -> dict:
    p = root / rel
    return tomllib.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def load_required(root: Path, rel: str) -> dict:
    path = root / rel
    if not path.is_file():
        raise ValueError(f"required current-release registry is missing: {rel}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


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


def draft(
    records: list[dict], present: set[str], version: str, queries: list[str] | None = None
) -> str:
    cells = wrap_block(CELLS_BEGIN, cells_body(records, present), CELLS_END).rstrip("\n")
    refusals = wrap_block(REFUSALS_BEGIN, refusals_body(queries or []), REFUSALS_END).rstrip("\n")
    out = [
        f"# Antecedent {version}",
        "",
        draft_mark(version),
        " ".join(
            (
                "> **DRAFT.** The Cells and Refusals blocks are generated by",
                "`scripts/check_release_claims.py --write-draft`. Hand prose outside those",
                "markers is not claim data. Regenerate the blocks, then run",
                "`check_release_claims.py --final` at the release candidate.",
            )
        ),
        "",
        "## How to read the claims",
        "",
        "| Term | Meaning |",
        "| --- | --- |",
    ]
    out += [f"| {t} | {m} |" for t, m in GLOSSARY]
    out += [
        "",
        cells,
        "",
        "## Explicit refusals",
        "",
        refusals,
        "",
        "See the [support matrix](../support-matrix.md).",
        "",
    ]
    return "\n".join(out)


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


def _block_errors(
    text: str, records: list[dict], present: set[str], queries: list[str], frozen: bool
) -> list[str]:
    errors: list[str] = []
    for rec in records:
        if rec.get("inference_claim") not in CLAIMS:
            errors.append(
                f"{rec.get('id')}: record inference_claim {rec.get('inference_claim')!r} "
                "is not in the vocabulary"
            )
    which = "frozen snapshot" if frozen else "live registry"
    cells = extract_block(text, CELLS_BEGIN, CELLS_END)
    refusals = extract_block(text, REFUSALS_BEGIN, REFUSALS_END)
    if cells is None:
        errors.append("missing generated Cells block")
    elif cells != cells_body(records, present):
        errors.append(f"generated cells block does not match the {which}")
    if refusals is None:
        errors.append("missing generated Refusals block")
    elif refusals != refusals_body(queries):
        errors.append(f"generated refusals block does not match the {which}")
    return errors


def check_notes(root: Path, notes_rel: str, final: bool = False) -> list[str]:
    """Errors for one notes file.

    Shipped notes that have no generated blocks are not read as claim data and are not
    compared to the live registry. A file that has the blocks must match its snapshot, or
    the live registry when it is the unfrozen current release. Heading text is not consulted.
    """
    path = root / notes_rel
    if not path.is_file():
        return [f"{notes_rel} is missing (generate with --write-draft)"]
    text = path.read_text(encoding="utf-8")
    version = notes_version(notes_rel)
    if version is None:
        return [f"{notes_rel}: file name is not vX.Y.Z.md"]
    current = documentation_version(root)
    errors: list[str] = []
    if final and version == current:
        if DRAFT_MARK_RE.search(text) or re.search(r"\bDRAFT\b", text):
            errors.append("the notes are still marked DRAFT")
        if re.search(r"\bTODO\b", text):
            errors.append("TODO markers remain in the notes")
    cells = extract_block(text, CELLS_BEGIN, CELLS_END)
    refusals = extract_block(text, REFUSALS_BEGIN, REFUSALS_END)
    if version != current and cells is None and refusals is None:
        return errors
    source = source_for(root, version, current)
    if source is None:
        errors.append(
            f"generated blocks need {snapshot_rel(version)}; shipped notes are not checked "
            "against the live registry"
        )
        return errors
    records, present, queries = source
    frozen = load_snapshot(root, version) is not None
    return errors + _block_errors(text, records, present, queries, frozen)


def check(root: Path, notes_rel: str | None = None, final: bool = False) -> list[str]:
    """Errors for the documented notes, and for any other notes that carry generated blocks."""
    if notes_rel is not None:
        return check_notes(root, notes_rel, final)
    current = notes_for(root)
    directory = root / "docs/release-notes"
    files = (
        sorted(str(path.relative_to(root)) for path in directory.glob("v*.md"))
        if directory.is_dir()
        else []
    )
    errors: list[str] = []
    if current not in files:
        errors.append(f"{current} is missing (generate with --write-draft)")
    for rel in files:
        errors += [f"{rel}: {err}" for err in check_notes(root, rel, final and rel == current)]
    return errors


def write_blocks(
    path: Path, records: list[dict], present: set[str], queries: list[str], version: str
) -> None:
    """Insert or replace the Cells and Refusals blocks, leaving the surrounding prose."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(draft(records, present, version, queries), encoding="utf-8", newline="\n")
        return
    text = path.read_text(encoding="utf-8")
    text = insert_block(
        text,
        CELLS_BEGIN,
        CELLS_END,
        cells_body(records, present),
        "## Explicit refusals",
        after=False,
    )
    text = insert_block(
        text,
        REFUSALS_BEGIN,
        REFUSALS_END,
        refusals_body(queries),
        "## Explicit refusals",
        after=True,
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    explicit = argv[argv.index("--notes") + 1] if "--notes" in argv else None
    version = documentation_version(ROOT)
    notes = explicit or notes_for(ROOT)
    if "--snapshot" in argv:
        records, present, queries = source_for(ROOT, version, version) or ([], set(), [])
        path = ROOT / snapshot_rel(version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            dump_snapshot(version, records, present, queries), encoding="utf-8", newline="\n"
        )
        print(f"wrote {snapshot_rel(version)} ({len(records)} records)")
        return 0
    if "--write-draft" in argv:
        source = source_for(ROOT, notes_version(notes) or version, version)
        if source is None:
            print(
                f"{notes} is a shipped file with no snapshot; refusing to render it from live data"
            )
            return 1
        records, present, queries = source
        path = ROOT / notes
        if (
            path.is_file()
            and not DRAFT_MARK_RE.search(path.read_text(encoding="utf-8"))
            and "--force" not in argv
        ):
            print(f"{notes} is no longer marked DRAFT; refusing to overwrite (use --force)")
            return 1
        write_blocks(path, records, present, queries, notes_version(notes) or version)
        print(f"wrote generated blocks in {notes} ({len(records)} record entries)")
        return 0
    errors = check(ROOT, explicit, final="--final" in argv)
    if errors:
        print("release claims FAILED:")
        for err in errors:
            print(f" - {err}")
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

    def need(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(f"{name}: {detail}" if detail else name)

    withheld = _rec("a1", "calibrated", ["cov.a1"])
    body = entry_text(withheld, set())
    need(
        "coverage absent is point_only", "Claim: `point_only`" in body and "withheld" in body, body
    )
    need(
        "coverage absent does not say calibrated",
        "calibrated" not in body.lower() or "Claim: `point_only`" in body,
    )
    covered = entry_text(withheld, {"cov.a1"})
    need("coverage present is calibrated", "Claim: `calibrated`" in covered, covered)
    closed = entry_text(_rec("a1", "calibrated", ["cov.a1"], closed=True), {"cov.a1"})
    need(
        "closed interval stays point_only",
        "Claim: `point_only`" in closed and "withheld" in closed,
        closed,
    )
    envelope = entry_text(_rec("a2", "structural_envelope"), set())
    need("envelope is never a CI", "never a confidence interval" in envelope, envelope)
    ranged = _rec(
        "a3",
        "assumption_range",
        None,
        "exact_range_complete_within_declared_contamination_class",
        closed=True,
    )
    ranged_body = entry_text(ranged, set())
    need(
        "complete qualifier is rendered",
        "within the declared contamination class" in ranged_body,
        ranged_body,
    )
    need(
        "sampling interval not offered is rendered",
        "sampling interval not offered" in ranged_body,
        ranged_body,
    )
    need(
        "assumption range is never a CI", "never a confidence interval" in ranged_body, ranged_body
    )
    bounds = entry_text(_rec("a5", "point_only"), set())
    need("bounds line is rendered", "Bounds: max_observed = 12." in bounds, bounds)
    need(
        "bare none is not a claim word",
        not claim_words("Bounds: none declared.") and not claim_words("publishes none."),
    )

    recs = [
        _rec("a1", "calibrated", ["cov.a1"]),
        _rec("a2", "structural_envelope"),
        _rec("a5", "point_only"),
        _rec("a9", "none"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "parity").mkdir()
        (root / "docs/release-notes").mkdir(parents=True)
        (root / RECORDS).write_text(_toml(recs), encoding="utf-8", newline="\n")
        (root / COVERAGE).write_text("", encoding="utf-8", newline="\n")
        (root / "parity/support_axes.toml").write_text('queries = []\n', encoding="utf-8")
        (root / "parity/support_licensed.toml").write_text('', encoding="utf-8")
        (root / "Cargo.toml").write_text(
            '[workspace.package]\nversion = "9.9.9"\n', encoding="utf-8", newline="\n"
        )
        notes = root / notes_for(root)
        projected, present = live_projection(root)
        notes.write_text(draft(projected, present, "9.9.9"), encoding="utf-8", newline="\n")

        def expect(name: str, needle: str | None, errs: list[str]) -> None:
            if needle is None:
                if errs:
                    failures.append(f"{name}: expected clean, got {errs[:3]}")
            elif not any(needle in err for err in errs):
                failures.append(f"{name}: expected {needle!r}, got {errs[:3]}")

        expect("clean draft", None, check(root))
        expect("final rejects draft", "DRAFT", check(root, final=True))
        text = notes.read_text(encoding="utf-8")
        notes.write_text(
            text.replace("Claim: `structural_envelope`", "Claim: `point_only`", 1),
            encoding="utf-8",
            newline="\n",
        )
        expect("edited cells block", "generated cells block", check(root))
        notes.write_text(
            text.replace("## How to read the claims", "## Reading the words"),
            encoding="utf-8",
            newline="\n",
        )
        outside = notes.read_text(encoding="utf-8").replace(
            "See the [support matrix](../support-matrix.md).",
            "A bootstrap interval is reported. Claim: `calibrated`. The nominal interval covers.\n\n"
            "See the [support matrix](../support-matrix.md).",
        )
        notes.write_text(outside, encoding="utf-8", newline="\n")
        expect("prose outside the markers is not claim data", None, check(root))
        expect("reworded glossary heading is ignored", None, check(root))
        notes.write_text(text, encoding="utf-8", newline="\n")
        # Shipped notes keep a contradictory Claim line. No generated blocks, so the live
        # registry is not applied to them.
        shipped = root / "docs/release-notes/v1.0.0.md"
        shipped.write_text(
            "# Old\n\n## Claims glossary\n\n<!-- cell: synth.cell_a2 -->\nClaim: `point_only`\n",
            encoding="utf-8",
            newline="\n",
        )
        (root / RECORDS).write_text(
            _toml([_rec("a2", "calibrated", ["cov.a2"]), *recs[2:]]), encoding="utf-8", newline="\n"
        )
        (root / "parity/release_snapshots").mkdir()
        (root / snapshot_rel("1.0.0")).write_text(
            dump_snapshot("1.0.0", [project_record(_rec("a2", "structural_envelope"))], set(), []),
            encoding="utf-8",
            newline="\n",
        )
        # Current notes must follow the live delta (a2 is not in the 1.0.0 snapshot id...
        # project id is synth.cell_a2, which IS in the live registry). Freeze that id in the
        # snapshot and regenerate the current blocks from the remaining live records.
        snap_recs = [project_record(_rec("a2", "structural_envelope"))]
        (root / snapshot_rel("1.0.0")).write_text(
            dump_snapshot("1.0.0", snap_recs, set(), []), encoding="utf-8", newline="\n"
        )
        current_records, current_present, queries = source_for(root, "9.9.9", "9.9.9")
        notes.write_text(
            draft(current_records, current_present, "9.9.9", queries),
            encoding="utf-8",
            newline="\n",
        )
        expect("shipped hand prose is not checked against live data", None, check(root))
        # A generated block on a shipped file matches the snapshot, even when live data differs.
        shipped.write_text(
            "# Old\n\n## Whatever the heading says\n\n"
            + wrap_block(CELLS_BEGIN, cells_body(snap_recs, set()), CELLS_END)
            + wrap_block(REFUSALS_BEGIN, refusals_body([]), REFUSALS_END),
            encoding="utf-8",
            newline="\n",
        )
        expect("shipped blocks match the snapshot, not the live registry", None, check(root))
        shipped.write_text(
            shipped.read_text(encoding="utf-8").replace(
                "Claim: `structural_envelope`", "Claim: `calibrated`", 1
            ),
            encoding="utf-8",
            newline="\n",
        )
        expect("shipped block edited away from the snapshot", "frozen snapshot", check(root))
        shipped.write_text("# Old\n\nClaim: `calibrated`\n", encoding="utf-8", newline="\n")
        expect("shipped notes without blocks stay unchecked", None, check(root))
        notes.write_text("# x\n", encoding="utf-8", newline="\n")
        expect("current notes without blocks", "missing generated Cells block", check(root))

    if failures:
        print("check_release_claims self-test FAILED:")
        for item in failures:
            print(f" - {item}")
        return 1
    print("check_release_claims self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
