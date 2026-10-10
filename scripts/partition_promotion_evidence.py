"""Partition the fully enumerated promotion citations for parallel CI execution.

Every row belongs to exactly one group; the unsplit local gate remains the
reference. The two facade groups divide complete test files, so one assertion
is never run in two jobs.
"""

import sys
import tomllib
from collections import Counter
from pathlib import Path

GROUPS = (
    "python", "facade-0", "facade-1", "estimate", "identify", "other"
)


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    facade = Counter(
        row["evidence_test"] for row in rows
        if row["evidence_test"].startswith("crates/antecedent/")
    )
    facade_files: dict[str, str] = {}
    loads = [0, 0]
    for path, count in sorted(facade.items(), key=lambda item: (-item[1], item[0])):
        chosen = 0 if loads[0] <= loads[1] else 1
        facade_files[path] = f"facade-{chosen}"
        loads[chosen] += count

    groups: dict[str, list[dict]] = {name: [] for name in GROUPS}
    for row in rows:
        path = row["evidence_test"]
        if path.startswith("python/"):
            group = "python"
        elif path.startswith("crates/antecedent/"):
            group = facade_files[path]
        elif path.startswith("crates/antecedent-estimate/"):
            group = "estimate"
        elif path.startswith("crates/antecedent-identify/"):
            group = "identify"
        elif path.startswith("crates/"):
            group = "other"
        else:
            raise ValueError(f"unknown evidence path: {path}")
        groups[group].append(row)
    return groups


def toml_string(value: str) -> str:
    import json
    return json.dumps(value)


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[3] not in GROUPS:
        raise SystemExit(f"usage: {sys.argv[0]} INPUT OUTPUT {'|'.join(GROUPS)}")
    source, output, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    rows = tomllib.loads(source.read_text())["fixture_evidence"]
    groups = group_rows(rows)
    assert sum(map(len, groups.values())) == len(rows)
    selected = groups[name]
    if not selected:
        output.write_text("")
        print(f"promotion evidence {name}: 0/{len(rows)} rows")
        return
    output.write_text("".join(
        "[[fixture_evidence]]\n"
        + "".join(f"{key} = {toml_string(str(row[key]))}\n"
                  for key in ("id", "evidence_test", "evidence_assertion"))
        + "\n" for row in selected
    ))
    print(f"promotion evidence {name}: {len(selected)}/{len(rows)} rows")


if __name__ == "__main__":
    main()
