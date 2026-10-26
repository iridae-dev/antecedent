#!/usr/bin/env python3
"""Check the verified counterpart inventory (parity/counterparts.toml).

Fails when

  stale                 the generated inventory or its docs table differs from the sources
  vocabulary            a status, uncertainty kind or closure kind is outside the closed vocabulary
  missing_quadrant      a method has no row for one of static/temporal x frequentist/bayesian
  weaker_temporal       a temporal row carries weaker uncertainty than its static twin and
                        records no reason
  not_implemented       a reason reads as "not implemented" instead of a theory reason or a
                        failed attempt
  closure_kind          a closed_with_next_attempt row is not a narrowed_promise, or a narrowed
                        promise has no reason or no next attempt
  evidence              an executing row lacks a recorded target or sampling design, a fixture
                        that exists and is run, or an artifact consumer that reads the artifact,
                        or an uncertainty no method record backs

Per method it reports which side is missing and the next attempt.

    python3 scripts/check_counterparts.py
    python3 scripts/check_counterparts.py --report     # per-method missing sides and counts
    python3 scripts/check_counterparts.py --self-test
"""

from __future__ import annotations

import re
import sys
import tomllib
from collections import Counter

import generate_counterparts as gen
from _repo import REPO_ROOT as ROOT
from method_evidence import Evidence

INVENTORY = ROOT / "parity/counterparts.toml"
NOT_IMPLEMENTED = re.compile(
    r"\bnot\s+(?:yet\s+)?(?:implemented|built|supported)\b|\bunimplemented\b|\bnot\s+done\b|\btodo\b",
    re.I,
)


def evidence_problems(rows: list[dict], ev: Evidence) -> list[str]:
    """An executing row has a target, a sampling design, a verified fixture and a reading consumer."""
    errors: list[str] = []
    for row in rows:
        rid = row["id"]
        if row["uncertainty_basis"] not in gen.BASES:
            errors.append(
                f"evidence: {rid} uncertainty basis {row['uncertainty_basis']!r} is outside the vocabulary"
            )
        if row["status"] not in gen.EXECUTING:
            continue
        if row["uncertainty_basis"] == "no_method_record":
            errors.append(f"evidence: {rid} executes but no method record backs its uncertainty")
        if row["target"] in ("unlisted", "none", ""):
            errors.append(f"evidence: {rid} executes with no recorded target")
        if row["sampling_design"] in ("unlisted", ""):
            errors.append(f"evidence: {rid} executes with no recorded sampling design")
        for problem in (
            ev.fixture_findings(row["execution"])
            if row["execution"] != "unlisted"
            else ["no execution fixture"]
        ):
            errors.append(f"evidence: {rid} execution {row['execution']}: {problem}")
        if row["artifact"] == "unlisted":
            errors.append(f"evidence: {rid} executes with no artifact consumer")
        else:
            for problem in ev.consumer_findings(row["artifact"]):
                errors.append(f"evidence: {rid} artifact {row['artifact']}: {problem}")
            if ev.is_test_path(ev.split(row["artifact"])[0]):
                errors.append(f"evidence: {rid} artifact consumer {row['artifact']} is a test")
    return errors


def problems(rows: list[dict], vocab: dict | None = None, ev: Evidence | None = None) -> list[str]:
    errors: list[str] = evidence_problems(rows, ev) if ev is not None else []
    statuses = set((vocab or {}).get("statuses", gen.STATUSES))
    kinds = set((vocab or {}).get("uncertainty_kinds", gen.UNCERTAINTY_KINDS))
    closures = set((vocab or {}).get("closure_kinds", gen.CLOSURE_KINDS))
    for row in rows:
        rid = row["id"]
        if row["status"] not in statuses:
            errors.append(f"vocabulary: {rid} status {row['status']!r}")
        if row["uncertainty"] not in kinds:
            errors.append(f"vocabulary: {rid} uncertainty {row['uncertainty']!r}")
        if row["closure"] not in closures:
            errors.append(f"vocabulary: {rid} closure {row['closure']!r}")
        if NOT_IMPLEMENTED.search(row["reason"]):
            errors.append(
                f"not_implemented: {rid} reason reads as not implemented; name the failed attempt "
                f"or the theory reason and the next attempt: {row['reason'][:90]!r}"
            )
        if row["status"] == "closed_with_next_attempt" and row["closure"] != "narrowed_promise":
            errors.append(
                f"closure_kind: {rid} is closed_with_next_attempt but not a narrowed_promise"
            )
        if row["status"] == "not_applicable_with_theory_reason" and not row["reason"].strip():
            errors.append(f"closure_kind: {rid} is not applicable with no theory reason")
        if row["closure"] == "narrowed_promise" and not (
            row["reason"].strip() and row["next_attempt"].strip()
        ):
            errors.append(
                f"closure_kind: {rid} is a narrowed promise without a reason and a next attempt"
            )
    grouped = gen.methods(rows)
    for (kind, method), quads in sorted(grouped.items()):
        for quadrant in gen.QUADRANTS:
            if quadrant not in quads:
                errors.append(f"missing_quadrant: {kind} {method} has no row for {quadrant}")
        for inference in gen.INFERENCES:
            static = quads.get(f"static.{inference}")
            temporal = quads.get(f"temporal.{inference}")
            if not static or not temporal:
                continue
            twin = gen.best(static)
            for row in temporal:
                if row["status"] not in ("licensed", "executing_unlicensed"):
                    continue
                if twin["status"] not in ("licensed", "executing_unlicensed"):
                    continue
                if (
                    row["uncertainty"] not in gen.UNCERTAINTY_RANK
                    or twin["uncertainty"] not in gen.UNCERTAINTY_RANK
                ):
                    continue
                weaker = (
                    gen.UNCERTAINTY_RANK[row["uncertainty"]]
                    < gen.UNCERTAINTY_RANK[twin["uncertainty"]]
                )
                if weaker and not row["reason"].strip():
                    errors.append(
                        f"weaker_temporal: {row['id']} carries {row['uncertainty']} uncertainty against "
                        f"{twin['uncertainty']} on {twin['id']} with no recorded reason"
                    )
    return errors


def report(rows: list[dict]) -> str:
    lines = []
    totals: Counter[str] = Counter()
    for (kind, method), quads in sorted(gen.methods(rows).items()):
        missing = []
        for quadrant in gen.QUADRANTS:
            if quadrant not in quads:
                missing.append((quadrant, "no row records this quadrant"))
                continue
            row = gen.best(quads[quadrant])
            if row["status"] == "closed_with_next_attempt":
                missing.append((quadrant, row["next_attempt"]))
        totals[kind] += len(missing)
        for quadrant, attempt in missing:
            lines.append(f"{kind:6} {method}: {quadrant} missing; next attempt: {attempt}")
    lines.append("")
    lines.append(f"counts: {dict(totals)}; statuses: {dict(Counter(r['status'] for r in rows))}")
    lines.append(f"closure: {dict(Counter(r['closure'] for r in rows))}")
    return "\n".join(lines)


def load_rows() -> tuple[list[dict], dict]:
    with INVENTORY.open("rb") as handle:
        data = tomllib.load(handle)
    return data["row"], data


def self_test() -> int:
    base = {
        "kind": "method",
        "method": "M",
        "route": "r",
        "uncertainty": "analytic",
        "closure": "delivered",
        "status": "licensed",
        "reason": "",
        "next_attempt": "",
        "sampling_design": "x",
        "reuse_key": "k",
        "decision_port": "u",
        "artifact": "u",
        "scope_status": "mandatory",
        "gap": "none",
    }

    def full(**overrides: dict) -> list[dict]:
        rows = []
        for quadrant in gen.QUADRANTS:
            design, inference = quadrant.split(".")
            rows.append(base | {"id": f"M.{quadrant}", "design": design, "inference": inference})
        for row in rows:
            for key, value in overrides.get(row["id"], {}).items():
                row[key] = value
        return rows

    failures = 0

    def expect(label: str, rows: list[dict], needle: str | None) -> None:
        nonlocal failures
        found = problems(rows)
        ok = (not found) if needle is None else any(needle in line for line in found)
        if not ok:
            failures += 1
            print(f"self-test: {label}: {found}")

    expect("complete method passes", full(), None)
    expect(
        "missing quadrant fails",
        [r for r in full() if r["id"] != "M.temporal.bayesian"],
        "missing_quadrant",
    )
    expect(
        "weaker temporal fails",
        full(**{"M.temporal.frequentist": {"uncertainty": "none"}}),
        "weaker_temporal",
    )
    expect(
        "weaker temporal with a reason passes",
        full(
            **{
                "M.temporal.frequentist": {
                    "uncertainty": "none",
                    "reason": "no serial likelihood",
                    "closure": "narrowed_promise",
                    "next_attempt": "derive it",
                }
            }
        ),
        None,
    )
    expect(
        "not implemented fails",
        full(
            **{
                "M.static.bayesian": {
                    "reason": "Not implemented",
                    "closure": "narrowed_promise",
                    "next_attempt": "n",
                }
            }
        ),
        "not_implemented",
    )
    expect(
        "closed must be narrowed",
        full(
            **{
                "M.static.bayesian": {
                    "status": "closed_with_next_attempt",
                    "reason": "r",
                    "next_attempt": "n",
                }
            }
        ),
        "closure_kind",
    )
    expect("bad vocabulary fails", full(**{"M.static.bayesian": {"status": "done"}}), "vocabulary")
    expect(
        "narrowed needs next attempt",
        full(**{"M.static.bayesian": {"closure": "narrowed_promise", "reason": "r"}}),
        "closure_kind",
    )
    print("self-test: ok" if failures == 0 else f"self-test: {failures} failure(s)")
    return 1 if failures else 0


def main() -> int:
    args = sys.argv[1:]
    if "--self-test" in args:
        return self_test()
    if gen.main_check() != 0:
        return 1
    rows, data = load_rows()
    if "--report" in args:
        print(report(rows))
    errors = problems(rows, data, Evidence(ROOT))
    for line in errors:
        print(line)
    print(f"counterparts: {len(rows)} rows, {len(errors)} finding(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
