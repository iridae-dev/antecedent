#!/usr/bin/env python3
"""List every implemented capability that is neither supported nor closed for a stated reason.

A release has two resting states for a capability: supported, with the evidence its claim
needs, or refused because the question has no answer. Everything else is open work, whatever
ledger records it:

  route         a public route that executes without support (`execution_only`, `fitting_gap`)
  measurement   a scenario cell whose calibration is still owed
  attempt       a mandatory scope row whose attempt has no recorded outcome
  procedure     a registered whole-procedure measurement that grants no support
  capability    a capability record that is not supported and names no closed question

    python3 scripts/check_open_capabilities.py                 # print the open list
    python3 scripts/check_open_capabilities.py --require-zero  # release: fail while any is open
"""

from __future__ import annotations

import sys
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from _repo import load_toml

OPEN_ROUTE_STATUSES = ("execution_only", "fitting_gap")


def open_items(root: Path = ROOT) -> list[tuple[str, str, str]]:
    """Every open item as (kind, id, what is missing)."""
    parity = root / "parity"
    items: list[tuple[str, str, str]] = []
    for route in load_toml(parity / "capabilities/route_manifest.toml").get("route", []):
        if route.get("status") in OPEN_ROUTE_STATUSES:
            items.append(("route", route["id"], str(route.get("uncertainty", route["status"]))))
    for row in load_toml(parity / "pending_measurement.toml").get("pending", []):
        items.append(("measurement", row["row"], f"{row['owed']} ({row['facet']})"))
    for entry in load_toml(parity / "release_scope/scope.toml").get("entry", []):
        if entry.get("status") == "pending_attempt":
            items.append(("attempt", entry["id"], str(entry.get("implementation_status", ""))))
    registry = parity / "joint_response_calibration.toml"
    if registry.is_file():
        document = load_toml(registry)
        if not document.get("grants_release_support", False):
            for row in document.get("procedure", []):
                items.append(("procedure", row["id"], str(document.get("status", ""))))
    for record in load_toml(parity / "capability_records.toml").get("record", []):
        if record.get("status") != "supported" and not record.get("closed_question"):
            items.append(("capability", record["id"], str(record.get("guarantee", ""))))
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
