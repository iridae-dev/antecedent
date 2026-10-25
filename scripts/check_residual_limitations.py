#!/usr/bin/env python3
"""Check the residual-limitation ledger against the tracker and the live scan.

Rejected:

  unmapped         a limitation with neither an owner nor a scope decision
  ambiguous        a row naming both an owner and a decision, or a decision that is not `scope`
  no_rationale     a scope decision with an empty written rationale
  self_owned       an owner that is the limitation's own item
  closed_owner     an owner the tracker has already checked off
  absent_owner     an owner the tracker does not list
  missing_row      a limitation the scan finds that the ledger lacks (or the reverse: stale row)
  unused_rule      an owner rule that matches no scanned limitation

A row whose own item is checked off in the tracker is reported as a contradiction: the item claims
completion while its residual remains. It does not fail the check, since only the tracker's owner
may reopen the item.

    python3 scripts/check_residual_limitations.py
"""

from __future__ import annotations

import sys

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path, load_toml, read_text

ensure_scripts_on_path()
import residual_limitations as ledger  # noqa: E402


def problems(
    rows: list[dict],
    tracker: dict[str, bool],
    scanned: list[dict] | None = None,
    rules: list[dict] | None = None,
) -> list[str]:
    """Every rejection, as `code: id: detail`; `tracker` maps an item id to whether it is open."""
    errors: list[str] = []
    for row in rows:
        ident = row["id"]
        owner = row.get("owner", "")
        decision = row.get("decision", "")
        if not owner and not decision:
            errors.append(f"unmapped: {ident}: no owner and no scope decision")
            continue
        if owner and decision:
            errors.append(f"ambiguous: {ident}: names owner {owner} and decision {decision}")
        if decision and decision != "scope":
            errors.append(f"ambiguous: {ident}: decision {decision!r} is not 'scope'")
        if decision and not row.get("rationale", "").strip():
            errors.append(f"no_rationale: {ident}: scope decision without a written rationale")
        if owner:
            if owner == row.get("item", ""):
                errors.append(f"self_owned: {ident}: owner {owner} is the limitation's own item")
            elif owner not in tracker:
                errors.append(f"absent_owner: {ident}: {owner} is not listed in {ledger.TRACKER}")
            elif not tracker[owner]:
                errors.append(f"closed_owner: {ident}: {owner} is already closed")
    if scanned is not None:
        have = {row["id"] for row in rows}
        found = {row["id"] for row in scanned}
        for ident in sorted(found - have):
            errors.append(f"missing_row: {ident}: scanned but absent from the ledger")
        for ident in sorted(have - found):
            errors.append(f"missing_row: {ident}: in the ledger but no longer scanned (stale)")
    if rules is not None and scanned is not None:
        for rule in rules:
            if not any(ledger.rule_matches(rule, row) for row in scanned):
                errors.append(f"unused_rule: {rule.get('name', '?')}: matches no scanned limitation")
    return errors


def contradictions(rows: list[dict], tracker: dict[str, bool]) -> list[str]:
    return [
        f"{row['id']}: item {row['item']} is checked off but the limitation remains"
        for row in rows
        if row.get("item") and tracker.get(row["item"]) is False
    ]


def main() -> int:
    tracker = ledger.parse_tracker(read_text(ROOT / ledger.TRACKER))
    stored = load_toml(ROOT / ledger.LEDGER).get("limitation", [])
    rules = load_toml(ROOT / ledger.OWNERS).get("owner", [])
    errors = problems(stored, tracker, ledger.collect(ROOT), rules)
    for note in contradictions(stored, tracker):
        print(f"contradiction: {note}")
    if read_text(ROOT / ledger.LEDGER) != ledger.build(ROOT):
        errors.append(f"stale: {ledger.LEDGER} differs from the generator output")
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print(f"{len(errors)} residual-limitation problem(s)", file=sys.stderr)
        return 1
    print(f"{len(stored)} residual limitations, all owned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
