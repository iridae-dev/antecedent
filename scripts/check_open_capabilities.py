"""List every capability that is neither supported nor closed for a stated reason.

A release has two resting states for a capability: supported, with the evidence its claim
needs, or refused because the question has no answer. Everything else is open work, whatever
ledger records it:

  capability    a capability record that is not supported and names no closed question, or
                a capability unit the support gate (`gate_support.check_capability`) does
                not find supported, with the gate's missing evidence
  measurement   a calibration row still owed in `parity/pending_measurement.toml`
  statement     a published uncertainty statement no measurement covers
                (`parity/unmeasured_statements.toml`)

    python3 scripts/check_open_capabilities.py                 # print the open list
    python3 scripts/check_open_capabilities.py --require-zero  # release: fail while any is open
"""

from __future__ import annotations

import sys
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from _repo import load_toml


def gate_unsupported_capabilities() -> list[tuple[str, str, str]]:
    """Capability units the support gate does not find supported, with its missing evidence.

    This runs the gate's own rule over the registries (provenance and a resolvable independent
    check) instead of reading the embedded `unit_support_status.json` snapshot: that snapshot is
    rewritten only by the generator, so it can lag the registries, and a unit that lost its
    evidence would stay hidden until regeneration. Ledger currency is not applied here; owed
    scientific records are listed as measurements."""
    import gate_support

    result = gate_support.evaluate(evidence_only=True)
    return [
        ("capability", unit.ident, "; ".join(dict.fromkeys(unit.problems)))
        for unit in result["capabilities"]
        if not unit.supported
    ]


def open_items(
    root: Path = ROOT,
    gate_items: list[tuple[str, str, str]] | None = None,
) -> list[tuple[str, str, str]]:
    """Every open item as (kind, id, what is missing)."""
    parity = root / "parity"
    records = load_toml(parity / "capability_records.toml").get("record", [])
    items: list[tuple[str, str, str]] = []
    for row in load_toml(parity / "pending_measurement.toml").get("pending", []):
        items.append(("measurement", row["row"], f"{row['owed']} ({row['facet']})"))
    declared: set[str] = set()
    for record in records:
        if record.get("status") != "supported" and not record.get("closed_question"):
            items.append(("capability", record["id"], str(record.get("guarantee", ""))))
            declared.add(str(record["id"]))
    if gate_items is None:
        gate_items = gate_unsupported_capabilities()
    items.extend(item for item in gate_items if item[1] not in declared)
    statements = parity / "unmeasured_statements.toml"
    if statements.is_file():
        for row in load_toml(statements).get("statement", []):
            items.append(("statement", row["route"], row["statement"]))
    return items


def main(argv: list[str]) -> int:
    if [a for a in argv if a != "--require-zero"]:
        print(f"usage: {sys.argv[0]} [--require-zero]", file=sys.stderr)
        return 2
    items = open_items()
    counts: dict[str, int] = {}
    for kind, identifier, missing in items:
        counts[kind] = counts.get(kind, 0) + 1
        print(f"{kind:12s} {identifier}: {missing[:160]}")
    summary = ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items())) or "none"
    print(f"open capabilities: {len(items)} ({summary})")
    if "--require-zero" in argv and items:
        print("FAIL: a release has no open capability")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
