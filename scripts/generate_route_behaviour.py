#!/usr/bin/env python3
"""Generate `parity/generated/route_behaviour.toml`: the behaviour a record does not restate.

For every capability record the file lists the transport stage routes it owns (stage,
status, reason code) and, for a supported record, the `(code, detail)` refusal pairs of its
declared details: each detail must be registered in `parity/reason_details.toml` and emitted
by non-test source, and any code the source proves must be one the registry lists.

    python3 scripts/generate_route_behaviour.py           # write the file
    python3 scripts/generate_route_behaviour.py --check   # exit 1 when the file drifts

When behaviour changes (a refusal detail is added or removed, a route is added, closed or
reopened) edit the record's `refusals` / `routes` (a refusal row is `{ detail }`; a closed route
carries its `reason_code`), register a new detail in `parity/reason_details.toml`, rerun
this script and commit the regenerated file. `scripts/check_capability_records.py` fails when
the committed file differs from a fresh derivation.
"""

import sys
from pathlib import Path

from _repo import REPO_ROOT as root
from _repo import load_toml
from capability_source import scan_refusal_details
from route_behaviour import GENERATED, REASON_DETAILS, derive, load_details, render

# Non-test source scanned for refusal-detail literals (the checker scans the same corpus).
SOURCE_GLOBS = ("crates/*/src/**/*.rs", "python/src/**/*.rs", "python/antecedent/**/*.py")


def build(records: list[dict]) -> tuple[str, list[str]]:
    files = [(p, True) for pattern in SOURCE_GLOBS for p in sorted(root.glob(pattern))]
    namespaces = {
        row.get("detail", "").split(".")[0]
        for rec in records
        if rec.get("status") == "supported"
        for row in rec.get("refusals") or []
        if "." in row.get("detail", "")
    }
    texts: dict[Path, str] = {}

    def read(path: Path) -> str:
        if path not in texts:
            texts[path] = path.read_text(errors="ignore", encoding="utf-8")
        return texts[path]

    emitted, _ = scan_refusal_details(namespaces, files, read, lambda p: str(p.relative_to(root)))
    data, problems = derive(records, load_details(root / REASON_DETAILS), emitted)
    for rec in records:
        if rec.get("status") != "supported":
            continue
        for row in rec.get("refusals") or []:
            detail = row.get("detail", "")
            if detail not in emitted.get(detail.split(".")[0], {}):
                problems.append(f"{rec.get('id')}: refusal detail {detail} is not emitted by non-test source")
    return render(data), problems


def main(argv: list[str]) -> int:
    if any(a not in ("--check",) for a in argv):
        raise SystemExit("usage: generate_route_behaviour.py [--check]")
    records = load_toml(root / "parity/capability_records.toml").get("record", [])
    text, problems = build(records)
    if problems:
        print("route behaviour cannot be derived:", file=sys.stderr)
        for p in problems:
            print(f" - {p}", file=sys.stderr)
        return 1
    target = root / GENERATED
    if "--check" in argv:
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            print(f"{GENERATED} is out of date; run python3 scripts/generate_route_behaviour.py", file=sys.stderr)
            return 1
        print(f"{GENERATED}: up to date")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {GENERATED}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
