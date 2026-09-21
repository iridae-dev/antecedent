"""The one reader of the parity manifests for the feature gates.

Every feature gate (`gate_bayesian.sh`, `gate_gcm.sh`, ...) reads `[[capabilities]]`
rows and checks that each `done` row has evidence on disk. They all go through
this module, which parses with `tomllib`, so the gates see exactly the rows a real
TOML parser sees (multi-line strings, single-quoted literals, CRLF files,
indented keys and trailing comments included).

    from parity_rows import ...      # scripts/ is put on sys.path by the gates

* `rows(path)` / `find_row(path, id)`: the parsed rows.
* `honesty_problems(...)`: an inventory's retired/incomplete statuses and the
  evidence map for its `done` rows.
* `evidence_map_problems(...)`: the same evidence check for a curated subset of ids
  that may live in several manifests.
* `require_done(...)`, `exit_artifact_problems(...)`, `finish(...)`.

Evidence is the file (or fixture directory) that carries a row's behaviour. A test
file or a fixture directory must actually hold tests or fixtures, so deleting every
test in the cited file turns the gate red instead of leaving a path that merely
exists. The tests themselves are run, and counted, by the cargo lines each gate
issues through `scripts/counted_cargo.sh`.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

# The gates run from the repository root (or from an overlay of it in a self-test).
ROOT = Path.cwd()

_RUST_TEST = re.compile(r"#\[\s*(?:tokio::)?test\b")
_PY_TEST = re.compile(r"^(?:async\s+)?def\s+test\w*\s*\(", re.M)
RETIRED = ("intentional_deviation",)


def rows(path: str | Path, table: str = "capabilities") -> list[dict]:
    """Rows of `[[table]]` in `path` (relative to the repo root), parsed by tomllib."""
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    return tomllib.loads(p.read_text()).get(table, [])


def find_row(path: str | Path, cid: str) -> dict | None:
    return next((r for r in rows(path) if r.get("id") == cid), None)


def _evidence_problem(cid: str, rel: str) -> str | None:
    """Why `rel` cannot be evidence for row `cid`, or None."""
    p = ROOT / rel
    if not p.exists():
        return f"{cid} evidence missing: {rel}"
    if p.is_dir():
        if not any(f.is_file() for f in p.rglob("*")):
            return f"{cid} evidence directory is empty: {rel}"
        return None
    if p.stat().st_size == 0:
        return f"{cid} evidence file is empty: {rel}"
    text = p.read_text(errors="ignore")
    if p.suffix == ".rs" and "/tests/" in p.as_posix() and not _RUST_TEST.search(text):
        return f"{cid} evidence {rel} is a test file with no #[test] function"
    if p.suffix == ".py" and "/tests/" in p.as_posix() and not _PY_TEST.search(text):
        return f"{cid} evidence {rel} is a test module with no test function"
    return None


def honesty_problems(
    manifest: str,
    evidence: dict[str, str],
    *,
    blocked: tuple[str, ...] = (),
    all_done: bool = False,
) -> list[str]:
    """Problems with one inventory: retired statuses, and `done` rows without evidence.

    A `done` row must be named in `evidence`; a row whose status is retired (or in
    `blocked`) fails, and with `all_done` so does any status but `done`. Evidence
    ids that no longer name a row also fail, so the map cannot outlive the
    inventory it describes."""
    problems: list[str] = []
    seen: set[str] = set()
    for row in rows(manifest):
        cid, status = row.get("id"), row.get("status")
        seen.add(cid)
        if status in RETIRED:
            problems.append(f"{cid}: {status} is retired; use pending or done")
            continue
        if status in blocked:
            problems.append(f"{manifest}: {cid} still {status}")
            continue
        if status != "done":
            if all_done:
                problems.append(f"{manifest} {cid} status={status}")
            continue
        rel = evidence.get(cid)
        if not rel:
            problems.append(f"{cid} (status={status}) has no evidence mapping")
            continue
        if (msg := _evidence_problem(cid, rel)) is not None:
            problems.append(msg)
    for cid in sorted(set(evidence) - seen):
        problems.append(f"evidence map names {cid}, which is not a row of {manifest}")
    return problems


def evidence_map_problems(evidence: dict[str, str], manifests: list[str]) -> list[str]:
    """Each id in `evidence` must be a `done` row of one of `manifests` with real evidence."""
    by_id = {r.get("id"): r for m in manifests for r in rows(m)}
    problems: list[str] = []
    for cid, rel in evidence.items():
        row = by_id.get(cid)
        if row is None:
            problems.append(f"{cid} missing from {', '.join(manifests)}")
        elif row.get("status") != "done":
            problems.append(f"{cid} status={row.get('status')} (expected done)")
        elif (msg := _evidence_problem(cid, rel)) is not None:
            problems.append(msg)
    return problems


def require_done(manifest: str, ids: list[str] | tuple[str, ...], gate: str) -> list[str]:
    """Rows other inventories must have closed for `gate` to pass."""
    by_id = {r.get("id"): r for r in rows(manifest)}
    problems = []
    for cid in ids:
        row = by_id.get(cid)
        if row is None:
            problems.append(f"{cid} missing from {manifest}")
        elif row.get("status") != "done":
            problems.append(f"{cid} must be status=done when {gate} gate passes")
    return problems


def exit_artifact_problems(paths: list[str]) -> list[str]:
    return [f"required exit artifact missing: {p}" for p in paths if not (ROOT / p).exists()]


def finish(label: str, problems: list[str], ok_message: str) -> None:
    """Print the verdict and exit non-zero on any problem."""
    if problems:
        print(f"{label} gate FAILED:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print(ok_message)
