#!/usr/bin/env python3
"""Collect `calibration-record` lines into the coverage registry.

One command after a calibration run:

    python3 scripts/collect_coverage_records.py

It reads `target/calibration-records/*.log` (written by
`scripts/gate_calibration.sh`), writes `parity/coverage_records.toml` stamped
with the SHA of the commit the logs were measured at, rewrites the
`calibration` / `calibration_reason` pair for each already-bound licensed support
cell using only its existing record IDs, refreshes estimator rows, and regenerates
`crates/antecedent-io-core/data/coverage_records.json`
(loaded by `crates/antecedent-io-core/src/coverage_records_data.rs`;
`parity/coverage_records.toml` remains the registry).

Every record is measured at each point of its sample-size grid
(`ANTECEDENT_CALIBRATION_GRID_POINT`, `SampleGrid` in
crates/antecedent/tests/common/calibration.rs): the gate runs each group once
per point and writes `<group>.p<k>.log`. The lines of one record id, one per
point, are merged into one registry row by `merge_grid`: `grid` holds the
coverage measured at every point, `n_min..n_max` spans the points, and the row
is a boundary when any point is (a failing point is never averaged into a
pass). A record is refused unless every grid point is present and the points'
sample sizes strictly increase, so a design that does not scale its `n` with
the grid cannot claim a range.

A boundary record can recover the passing part of its range, but only after an
independent confirmation: the gate re-measures the passing points with fresh
seeds (`ANTECEDENT_CALIBRATION_CONFIRMATION=1`, logs `<group>.p<k>.confirm.log`),
and the collector keeps those tallies apart from the measurement's, in the
record's `confirmation`. When a contiguous run of at least two grid points, at
the top or the bottom of the grid, passed in the measurement and again in the
confirmation, the record states it as `confirmed_passing_range = [n_lo, n_hi]`.
The failing points and their evidence stay on the record, which remains a
boundary; the runtime (`calibration_slot_with`) calibrates only row counts
inside the confirmed range, governed by the worst passing point. A named
boundary never recovers a range. A confirmation point that failed leaves no
line and its log is dropped with a notice, so that point is unconfirmed.

A grid point that was rechecked at more replicates writes
`<group>.p<k>.recheck.log`; its lines replace that point's first run, because
the recheck's verdict is the one the gate takes. When the gate re-ran only some
of a group's tests (a whole-file group; the log names them in
`calibration-rechecked-test` lines), only those tests' records are replaced.

Every record is written with `facets`: the parts of the statistical surface it
depends on, derived by `scripts/calibration_facets.py` from the record itself.
A record stands until one of those facets changes, so a partial re-measurement
is enough when only some facets drifted:

    python3 scripts/collect_coverage_records.py --keep-attested

keeps every existing record the logs do not re-measure whose facets are
unchanged since its own `calibration_sha` (or that a valid replay waiver in
`parity/calibration_waivers.toml` attests), and drops (and names) the ones
that still owe a re-measurement. Without it the registry is exactly the logs.

`--retag` rewrites only the `facets` of the existing records (after a change
to `scripts/calibration_surface.list`); nothing is measured or re-stamped.

For a small independently measured set of cells while older records still owe
re-measurement, `--append-attested --only-cell QUERY:GRAPH:STRUCTURE:INFERENCE:VALIDATION=ID,ID`
adds only new IDs from valid, stamped logs, rejects collisions, preserves every
older record with its original SHA (including stale records), and syncs only
the named licensed cells to exactly the named record IDs after checking their
coordinate and declared estimator ownership. It leaves the global gate ledger and estimator rows
alone; preserving an older row does not attest or re-license it.

`--smoke --log-dir <dir> --out <file>` collects the lines of a wiring smoke run
(`ANTECEDENT_CALIBRATION_SMOKE=1` at a reduced replicate count) into a scratch
registry. Smoke lines measure nothing: every other invocation refuses them, and
`--smoke` refuses to write `parity/coverage_records.toml` or touch any registry.

Every log must show a passing `test result: ok. N passed` with N >= 1 and no
failure, and end with the `calibration-measured-at <sha>` line the gate wrote
(scripts/gate_calibration.sh); all logs must agree on that sha and it must equal
the stamped one. A run whose groups failed, matched no test, or was measured at
another commit than the one being stamped is refused. Without `--keep-attested`
the registry is exactly the logs, and the collector refuses to drop a record id
the registry already holds unless it is named with `--allow-removal <id>`.

`--sha <sha>` stamps that commit instead of HEAD (for logs measured at another
commit, which must say so in their `calibration-measured-at` line); `--no-cells` leaves the registries alone. Without `--sha` the
collector refuses to stamp HEAD while the worktree's statistical surface
differs from HEAD, because the logs would then describe uncommitted code.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import calibration_decision as coverage_decision
import calibration_decision_logs as coverage_decision_logs
import calibration_facets as facets
import calibration_family
from _repo import REPO_ROOT as ROOT
from _repo import read_text, write_text

# The schema reader consumes this module's historical configuration reader.
coverage_family = calibration_family

LOG_DIR = ROOT / "target" / "calibration-records"
OUT = ROOT / "parity" / "coverage_records.toml"
GATES = ROOT / "parity" / "calibration_gates.toml"
LICENSED = ROOT / "parity" / "support_licensed.toml"
ESTIMATE = ROOT / "parity" / "estimate.toml"
GENERATOR = ROOT / "scripts" / "generate_support_matrix_docs.py"

HEADER = """# Coverage records bound onto executions at claim time.
#
# Every row is a `calibration-record` line emitted by a coverage test through
# `CoverageTally::for_record` (crates/antecedent/tests/common/calibration.rs)
# and collected by `scripts/collect_coverage_records.py`, which stamps the SHA
# of the commit the logs were measured at. Do not edit rows by hand; re-run
# the collector. `scripts/gate_parity_schema.sh` checks every row. The pass/fail
# groups that emit no record are attested in parity/calibration_gates.toml. Each
# row's `surface_list_blob` is the git blob of scripts/calibration_surface.list at
# `calibration_sha`, so the surface a record was attested against is stored, not
# only re-derived from history.
"""

FIELDS = (
    "id",
    "query",
    "graph_class",
    "structure",
    "modality",
    "inference",
    "estimator",
    "interval_method",
    "se_kind",
    "dependence",
    "posterior",
    "functional",
    "identification",
    "nominal",
    "n_min",
    "n_max",
    "replicates_min",
    "posterior_draws_min",
    "unidentified_mass_max",
    "observed",
    "mcse",
    "replicates",
    "boundary",
    "role",
    "grid",
    "dgp",
    "test",
    "facets",
    "calibration_sha",
    "surface_list_blob",
)

# Fields only some records carry, written after `role` when present: a
# `one_sided` record (CoverageTally::assert_one_sided in
# crates/antecedent/tests/common/calibration.rs) states the side whose true
# extremal bound it covers and its target, `{ side = "lower"|"upper", target }`.
# A boundary record that passed an independent confirmation also states the
# confirmation's points (shaped like `grid`) and the `confirmed_passing_range`
# they establish (`confirmed_passing_range`).
OPTIONAL_FIELDS = ("decision_protocol", "one_sided", "conservative_point", "confirmation", "confirmed_passing_range")
# Coverage roles a grid point may carry. `gated` passes the two-sided band and
# precision floor/ceiling; `named_boundary` is a documented under-coverage;
# `reported_level` is an unasserted second level; `one_sided` passes a floor at
# the level with no upper band or ceiling (a conservative interval's
# at-least-nominal target). `conservative_point` is separately allowlisted
# for the theorem-supported scalar Gaussian study-mean construction.
ROLES = ("gated", "named_boundary", "reported_level", "one_sided", "conservative_point")
ONE_SIDED_SIDES = ("lower", "upper")

# A conservative point method is admitted by its scientific identity and a
# separately reviewed theorem, never by relabeling a failed nominal interval.
GAUSSIAN_STUDY_T_CLAIM = {
    "target": "common_marginal_study_mean",
    "justification": "independent_gaussian_study_t.ibragimov_mueller_2010.v1",
}


def conservative_point_problem(record: dict) -> str | None:
    """Validate the only currently supported conservative point construction."""
    claim = record.get("conservative_point")
    if record.get("role") != "conservative_point":
        return "conservative_point metadata on another role" if claim is not None else None
    if claim != GAUSSIAN_STUDY_T_CLAIM:
        return "conservative_point requires the Gaussian study-mean theorem and target"
    expected = {
        "query": "PooledEffect", "graph_class": "NoGraph", "structure": "fixed",
        "modality": "summary", "inference": "Frequentist",
        "estimator": "evidence.synthesis.equal_study_gaussian",
        "interval_method": "independent_gaussian_study_t",
        "se_kind": "empirical_study_variance", "dependence": "independent", "posterior": "",
        "identification": "point",
        "functional": "pooled_mean;equal_study_gaussian;studies=5-30",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        return "conservative_point is not the declared independent Gaussian study-t construction"
    if not 0.95 <= float(record.get("nominal", 0.0)) < 1.0:
        return "conservative_point Gaussian study theorem requires confidence at least 0.95"
    if record.get("posterior_draws_min", 0) != 0 or record.get("unidentified_mass_max", 0.0) != 0.0:
        return "conservative_point cannot borrow posterior or partial-identification evidence"
    if record.get("boundary"):
        return "conservative_point cannot be a boundary"
    return None


# Estimator rows of parity/estimate.toml and the resolved plan estimators whose
# records back them.
ESTIMATOR_ROW_IDS = {
    "estimate.linear_regression": {"linear.adjustment.ate", "temporal.linear.adjustment"},
    "estimate.glm": {"glm.adjustment"},
    "estimate.propensity": {"propensity.weighting", "propensity.matching", "propensity.stratification"},
    "estimate.matching": {"distance.matching"},
    "estimate.doubly_robust": {"aipw", "cell.aipw"},
    "estimate.iv": {"iv.wald", "iv.2sls"},
    "estimate.rd": {"rd.sharp"},
    "estimate.two_stage": {"frontdoor.linear_two_stage", "frontdoor.functional"},
    "estimate.conditional": {"conditional.linear.adjustment", "bayesian.conditional"},
    "estimate.temporal_sequential": {"temporal.sequential.gcomp"},
    "estimate.mediation.linear": {"mediation.linear", "temporal.mediation"},
}

NO_INTERVAL_QUERIES = {"Counterfactual", "AnomalyAttribution", "ChangeAttribution"}


# Sample-size grid points every record is measured at
# (`GRID_POINTS` in crates/antecedent/tests/common/calibration.rs).
GRID_POINTS = 3
# Fields measured per grid point; every other emitted field is the construction
# and provenance, which must be the same at every point.
POINT_FIELDS = (
    "decision_protocol",
    "n_min",
    "n_max",
    "replicates_min",
    "posterior_draws_min",
    "unidentified_mass_max",
    "observed",
    "mcse",
    "replicates",
    "bound_replicates",
    "boundary",
    "role",
    "grid_point",
)
GRID_ENTRY_FIELDS = (
    "point",
    "n_min",
    "n_max",
    "observed",
    "mcse",
    "replicates",
    "boundary",
    "role",
)


RECHECKED_TEST = re.compile(r"^calibration-rechecked-test (\S+)$", re.M)


def rechecked_tests(recheck_log: Path) -> set[str] | None:
    """The test functions a recheck log re-ran, when the gate re-ran only some
    of a group's tests (`calibration-rechecked-test` lines, written by
    scripts/gate_calibration.sh); `None` when it re-ran the whole group."""
    found = RECHECKED_TEST.findall(read_text(recheck_log, errors="ignore"))
    return {name.rpartition("::")[2] for name in found} or None


def record_lines(
    logs: list[Path], smoke: bool = False, sha: str | None = None, confirmation: bool = False
) -> dict[str, dict[int, dict]]:
    """`calibration-record` payloads by record id and grid point.

    A point's recheck log (`<group>.p<k>.recheck.log`) replaces that point's
    first run (`<group>.p<k>.log`): the recheck's verdict is the one the gate
    takes, and a point that failed it emitted no line at all, so the first
    run's line must not survive. A recheck that re-ran only some of a group's
    tests names them (`rechecked_tests`), and replaces only their records; the
    first run's lines of every other test stand."""
    by_name = {log.name: log for log in logs}
    out: dict[str, dict[int, dict]] = {}
    for log in sorted(logs):
        recheck = log.name.endswith(".recheck.log")
        superseded: set[str] | None = None
        if not recheck:
            partner = by_name.get(log.name.removesuffix(".log") + ".recheck.log")
            if partner is not None:
                superseded = rechecked_tests(partner)
                if superseded is None:
                    continue  # the whole group was re-run
        for line in read_text(log, errors="ignore").splitlines():
            if not line.startswith("calibration-record "):
                continue
            payload = json.loads(line.split(" ", 1)[1])
            rid = str(payload["id"])
            if superseded and str(payload["test"]).rpartition("::")[2] in superseded:
                continue
            if bool(payload.pop("smoke", False)) != smoke:
                raise SystemExit(
                    f"{log.name}: record {rid} is a wiring smoke line; it measured nothing and "
                    "only `--smoke` collects it, into a scratch registry"
                    if not smoke
                    else f"{log.name}: record {rid} is not a smoke line; --smoke collects only a "
                    "smoke run's logs"
                )
            if bool(payload.pop("confirmation", False)) != confirmation:
                raise SystemExit(
                    f"{log.name}: record {rid} "
                    + (
                        "is not a confirmation line; confirmation logs hold only the fresh-seed "
                        "re-measurement"
                        if confirmation
                        else "is a confirmation line; it is collected apart from the measurement"
                    )
                )
            measured_at = payload.pop("measured_at", "")
            if sha is not None and not smoke and measured_at != sha:
                raise SystemExit(
                    f"{log.name}: record {rid} says it was measured at {measured_at or '<nothing>'}, "
                    f"not at {sha}; each record carries the commit its harness measured, so a line "
                    "from another run or commit cannot be collected under this one"
                )
            if "grid_point" not in payload:
                raise SystemExit(
                    f"{log.name}: record {rid} carries no grid_point; it was measured before "
                    "the sample-size grid, so re-measure it with scripts/gate_calibration.sh"
                )
            validate_point_decision(payload)
            point = int(payload["grid_point"])
            if not 0 <= point < GRID_POINTS:
                raise SystemExit(f"{log.name}: record {rid} at grid point {point}")
            seen = out.setdefault(rid, {}).get(point)
            if seen is not None and not recheck:
                if seen["replicates"] == payload["replicates"]:
                    raise SystemExit(
                        f"duplicate record id {rid} at grid point {point} in {log.name}: "
                        "two tallies emit the same id"
                    )
                # The same measurement re-run at more replicates: keep the
                # more precise one (a precision recheck, however it was run).
                if seen["replicates"] > payload["replicates"]:
                    continue
            out[rid][point] = payload
    return out


def validate_point_decision(payload: dict, *, expected_config: dict | None = None,
                            final: bool = False) -> None:
    """Audit stored counts; only the governing final look claims compatibility."""
    value = payload.get("decision_protocol")
    if value is None:
        return
    try:
        coverage_decision.validate_decision(value, float(payload["nominal"]),
            observed=float(payload["observed"]), replicates=int(payload["replicates"]),
            expected_config=expected_config, role=str(payload["role"]) if final else None)
    except (ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"invalid coverage decision: {error}") from error


def grid_fields_valid(point: dict) -> bool:
    return set(point) in (set(GRID_ENTRY_FIELDS), set(GRID_ENTRY_FIELDS) | {"decision_protocol"})


def grid_entry(payload: dict) -> dict:
    """The registry's `grid` entry of one measured point."""
    return {
        **({"decision_protocol": payload["decision_protocol"]} if "decision_protocol" in payload else {}),
        "point": int(payload["grid_point"]),
        "n_min": int(payload["n_min"]),
        "n_max": int(payload["n_max"]),
        "observed": payload["observed"],
        "mcse": payload["mcse"],
        "replicates": int(payload["replicates"]),
        "boundary": bool(payload["boundary"]),
        "role": str(payload["role"]),
    }


def confirmed_passing_range(grid: list[dict], confirmation: list[dict]) -> list[int] | None:
    """`[n_lo, n_hi]` of the longest contiguous run of at least two grid points,
    touching the top or the bottom of the grid, that passed in the measurement
    and again in the confirmation; `None` when no run qualifies.

    A run in the middle is not a subrange: it would leave failing points on
    both sides of a claim. A tie goes to the upper run, where more data lies."""
    confirmed = {int(p["point"]) for p in confirmation if not p["boundary"]}
    passing = [not p["boundary"] and int(p["point"]) in confirmed for p in grid]
    runs = []
    for start in range(len(grid)):
        for end in range(start + 1, len(grid)):
            touches_end = start == 0 or end == len(grid) - 1
            if touches_end and all(passing[start : end + 1]):
                runs.append((end - start, end, start))
    if not runs:
        return None
    _, end, start = max(runs)
    return [int(grid[start]["n_min"]), int(grid[end]["n_max"])]


def attach_confirmation(rid: str, record: dict, points: dict[int, dict]) -> None:
    """Add a boundary record's independent confirmation and the passing range it
    establishes. Refused unless the confirmation measured the same construction,
    sample sizes and roles as the measurement; ignored for records with nothing
    to recover (not a boundary, or a named boundary, which documents a limit)."""
    if "decision_protocol" in record or any("decision_protocol" in p for p in points.values()):
        raise SystemExit(f"record {rid}: fresh-seed confirmation is not allocated risk in suite coverage protocol v1")
    if not record["boundary"] or record["role"] == "named_boundary":
        return
    entries = []
    for k in sorted(points):
        payload = points[k]
        measured = record["grid"][k]
        differs = sorted(
            key
            for key in payload
            if key not in POINT_FIELDS and payload[key] != record.get(key)
        )
        if differs:
            raise SystemExit(
                f"record {rid}: confirmation point {k} measured another construction "
                f"({', '.join(differs)} differ from the measurement)"
            )
        entry = grid_entry(payload)
        if (entry["n_min"], entry["n_max"], entry["role"]) != (
            measured["n_min"],
            measured["n_max"],
            measured["role"],
        ):
            raise SystemExit(
                f"record {rid}: confirmation point {k} measured {entry['n_min']}..{entry['n_max']} "
                f"rows as {entry['role']}, not the measurement's {measured['n_min']}.."
                f"{measured['n_max']} as {measured['role']}; a confirmation re-measures the same design"
            )
        entries.append(entry)
    record["confirmation"] = entries
    recovered = confirmed_passing_range(record["grid"], entries)
    if recovered is not None:
        record["confirmed_passing_range"] = recovered


def merge_grid(rid: str, points: dict[int, dict]) -> dict:
    """One registry row from a record's per-point payloads.

    The row's scope spans the points; it is a boundary when any point is, and
    its `observed` / `mcse` / `replicates` are the governing point's: the
    lowest-coverage failing point when one failed, otherwise the lowest-coverage
    point."""
    missing = [k for k in range(GRID_POINTS) if k not in points]
    if missing:
        raise SystemExit(
            f"record {rid} was not measured at grid point(s) {missing}: a record's range must "
            "span every point of its sample-size grid; run its group at every point"
        )
    ordered = [points[k] for k in range(GRID_POINTS)]
    first = ordered[0]
    configs = [{key: p["decision_protocol"][key] for key in coverage_decision.CONFIG_FIELDS}
               if "decision_protocol" in p else None for p in ordered]
    active_configs = [config for config in configs if config is not None]
    if (any(config != active_configs[0] for config in active_configs)
            or (active_configs and any(config is None and p['role']!='named_boundary'
                                      for config,p in zip(configs,ordered,strict=True)))):
        raise SystemExit(f"record {rid}: grid points mix coverage decision families or legacy protocols")
    for payload in ordered:
        validate_point_decision(payload, final=True)
    for payload in ordered[1:]:
        differs = sorted(
            key
            for key in set(first) | set(payload)
            if key not in POINT_FIELDS and first.get(key) != payload.get(key)
        )
        if differs:
            raise SystemExit(
                f"record {rid}: grid point {payload['grid_point']} measured another construction "
                f"({', '.join(differs)} differ from point 0)"
            )
    for low, high in zip(ordered, ordered[1:], strict=False):
        if not int(low["n_max"]) < int(high["n_min"]):
            raise SystemExit(
                f"record {rid}: grid point {high['grid_point']} measured {high['n_min']} rows, not "
                f"more than point {low['grid_point']}'s {low['n_max']}; the design does not scale "
                "its sample size with the grid (SampleGrid in tests/common/calibration.rs)"
            )
    roles = {str(p["role"]) for p in ordered}
    unknown = sorted(roles - set(ROLES))
    if unknown:
        raise SystemExit(f"record {rid}: unknown coverage role(s) {unknown}")
    for payload in ordered:
        if problem := conservative_point_problem(payload):
            raise SystemExit(f"record {rid}: {problem}")
    one_sided = first.get("one_sided")
    if ("one_sided" in roles) != (one_sided is not None):
        raise SystemExit(
            f"record {rid}: a one_sided record, and only one, states one_sided "
            "{ side, target } (CoverageTally::assert_one_sided)"
        )
    if one_sided is not None and (
        not isinstance(one_sided, dict)
        or sorted(one_sided) != ["side", "target"]
        or one_sided["side"] not in ONE_SIDED_SIDES
        or not str(one_sided["target"]).strip()
    ):
        raise SystemExit(
            f"record {rid}: one_sided must be {{ side = lower|upper, target = <non-empty> }}, "
            f"got {one_sided!r}"
        )
    if len(roles) == 1:
        role = roles.pop()
    elif roles == {"gated", "named_boundary"}:
        role = "named_boundary"
    else:
        raise SystemExit(f"record {rid}: grid points carry incompatible roles {sorted(roles)}")
    failing = [p for p in ordered if p["boundary"]]
    governing = min(failing or ordered, key=lambda p: (float(p["observed"]), int(p["grid_point"])))
    record = {key: value for key, value in first.items() if key not in POINT_FIELDS}
    record.update(
        n_min=min(int(p["n_min"]) for p in ordered),
        n_max=max(int(p["n_max"]) for p in ordered),
        replicates_min=min(int(p["replicates_min"]) for p in ordered),
        posterior_draws_min=min(int(p["posterior_draws_min"]) for p in ordered),
        unidentified_mass_max=max(float(p["unidentified_mass_max"]) for p in ordered),
        observed=governing["observed"],
        mcse=governing["mcse"],
        replicates=governing["replicates"],
        boundary=bool(failing),
        role=role,
        grid=[grid_entry(p) for p in ordered],
    )
    if "decision_protocol" in governing:
        record["decision_protocol"] = governing["decision_protocol"]
    return record


CONFIRMATION_LOG = re.compile(r"\.p\d+\.confirm(\.recheck)?\.log$")


def is_confirmation_log(log: Path) -> bool:
    """Whether a log is a fresh-seed confirmation (`<group>.p<k>.confirm[.recheck].log`)."""
    return CONFIRMATION_LOG.search(log.name) is not None


def merged_records(
    logs: list[Path], smoke: bool = False, sha: str | None = None
) -> dict[str, dict]:
    """Every record the logs measured, merged over its grid points, with the
    independent confirmation of each boundary record that has one."""
    measured = [log for log in logs if not is_confirmation_log(log)]
    records = {
        rid: merge_grid(rid, points)
        for rid, points in record_lines(measured, smoke, sha).items()
    }
    confirmations = record_lines(
        [log for log in logs if is_confirmation_log(log)], smoke, sha, confirmation=True
    )
    for rid, points in confirmations.items():
        if rid not in records:
            raise SystemExit(
                f"confirmation logs measured {rid}, which the measurement logs did not; a "
                "confirmation is collected beside the measurement it confirms"
            )
        attach_confirmation(rid, records[rid], points)
    return records


def drop_failed_confirmations(logs: list[Path]) -> list[Path]:
    """`logs` without the confirmation runs that failed.

    A confirmation point that failed its band measured nothing a record can
    state: its point stays unconfirmed, which is the answer, so the failure is
    reported and its logs (the run and its recheck) are left out instead of
    refusing the collection the way a failed measurement does."""
    failed = {
        log.name.replace(".recheck.log", ".log")
        for log in logs
        if is_confirmation_log(log) and log_problems(log)[0]
    }
    for name in sorted(failed):
        print(f"confirmation did not pass, point left unconfirmed: {name}")
    return [log for log in logs if log.name.replace(".recheck.log", ".log") not in failed]


CARGO_RESULT = re.compile(r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed;", re.M)
MEASURED_AT = re.compile(r"^calibration-measured-at ([0-9a-f]{40})$", re.M)


def log_problems(log: Path) -> tuple[list[str], set[str]]:
    """Why a group log cannot back a registry, and the shas it says it was measured at.

    The record-less groups (CI Type I, uniformity, discovery FPR, SBC) leave no
    artifact but their log, so a group that failed or matched no test would
    otherwise vanish from the registry without a trace."""
    text = read_text(log, errors="ignore")
    results = CARGO_RESULT.findall(text)
    problems = []
    if any(status == "FAILED" or int(failed) for status, _, failed in results):
        problems.append(f"{log.name}: a test failed")
    if sum(int(passed) for _, passed, _ in results) < 1:
        problems.append(f"{log.name}: no test passed (renamed or filtered away?)")
    return problems, set(MEASURED_AT.findall(text))


def check_logs(logs: list[Path], sha: str) -> None:
    """Refuse logs of a failed or empty group, or measured at another commit than `sha`."""
    problems: list[str] = []
    for log in logs:
        found, measured = log_problems(log)
        problems += found
        if not measured:
            problems.append(f"{log.name}: no calibration-measured-at line (not written by the gate)")
        elif measured != {sha}:
            problems.append(
                f"{log.name}: measured at {', '.join(sorted(measured))}, not at {sha}"
            )
    if problems:
        raise SystemExit("cannot collect these logs:\n  " + "\n  ".join(problems))
    try:
        coverage_decision_logs.audit_logs(logs, sha)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"invalid frozen-family decision log: {error}") from error


GATES_HEADER = """# Pass/fail calibration groups that emit no coverage record (CI Type I error,
# permutation uniformity, discovery FPR/power, SBC, response calibration).
#
# One row per group log `scripts/gate_calibration.sh` wrote and
# `scripts/collect_coverage_records.py` accepted: the group ran, passed at least
# one test and failed none, at `calibration_sha`. Do not edit rows by hand; re-run
# the collector. `scripts/gate_parity_schema.sh` checks every row.
"""


def gate_ledger(logs: list[Path], sha: str) -> dict[str, dict]:
    """A row per record-less group log: which group, how many tests passed, at which commit."""
    rows: dict[str, dict] = {}
    try:
        decisions = coverage_decision_logs.audit_logs(logs, sha)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"invalid frozen-family decision ledger: {error}") from error
    for log in logs:
        text = read_text(log, errors="ignore")
        if is_confirmation_log(log) or log.name.endswith(".recheck.log") or any(
            line.startswith("calibration-record ") for line in text.splitlines()
        ):
            continue
        passed = sum(int(n) for _, n, _ in CARGO_RESULT.findall(text))
        rows[log.name.removesuffix(".log")] = {
            "group": log.name.removesuffix(".log"),
            "passed": passed,
            "calibration_sha": sha,
        }
        raw = [{"log": name, **row, "record_id": row["record_id"] or ""} for name, entries in decisions.items()
               if name in {log.name, log.name.removesuffix(".log") + ".recheck.log"}
               for row in entries]
        contexts = [json.loads(line.partition(" ")[2]) for line in text.splitlines()
                    if line.startswith("calibration-decision-context ")]
        if contexts:
            rows[log.name.removesuffix(".log")]["coverage_decisions"] = raw
            rows[log.name.removesuffix(".log")]["decision_protocol"] = contexts[0]
        protocols = {line.removeprefix("CALIBRATION_GATE_PROTOCOL ").strip()
                     for line in text.splitlines() if line.startswith("CALIBRATION_GATE_PROTOCOL ")}
        if protocols:
            if len(protocols) != 1 or not re.fullmatch(r"[a-z][a-z0-9_]*", next(iter(protocols))):
                raise SystemExit(f"{log.name}: invalid or conflicting calibration protocol markers")
            rows[log.name.removesuffix(".log")]["protocol"] = next(iter(protocols))
    return rows


def write_gate_ledger(rows: dict[str, dict], keep: dict[str, dict] | None = None) -> None:
    merged = {**(keep or {}), **rows}
    lines = [GATES_HEADER]
    for group in sorted(merged):
        row = merged[group]
        lines += [
            "[[gate]]",
            f"group = {toml_value('group', group)}",
            f"passed = {int(row['passed'])}",
            f"calibration_sha = {toml_value('calibration_sha', row['calibration_sha'])}",
        ]
        for optional in ("decision_protocol", "coverage_decisions"):
            if optional in row:
                lines.append(f"{optional} = {toml_value(optional, row[optional])}")
        if "protocol" in row:
            lines.append(f"protocol = {toml_value('protocol', row['protocol'])}")
        lines.append("")
    write_text(GATES, "\n".join(lines))


def passing_group_logs(logs: list[Path], sha: str) -> list[Path]:
    """Harvest only complete passing groups; never recover selected grid points."""
    by_group: dict[str, list[Path]] = {}
    for log in logs:
        stem = log.name.removesuffix(".log").removesuffix(".recheck").removesuffix(".confirm")
        name = re.sub(r"\.p[012]$", "", stem)
        by_group.setdefault(name, []).append(log)
    accepted = []
    for group, members in sorted(by_group.items()):
        reasons = []
        points = {int(m.group(1)) for log in members
                  if (m := re.search(r"\.p([012])\.log$", log.name))}
        if points and points != {0, 1, 2}:
            reasons.append(f"incomplete sample grid: {sorted(points)}")
        if not any(not log.name.endswith((".recheck.log", ".confirm.log")) for log in members):
            reasons.append("no first-pass log")
        try:
            check_logs(members, sha)
            merged_records(members, sha=sha)
        except (SystemExit, ValueError, KeyError) as error:
            reasons.append(str(error))
        if reasons:
            print(f"passing-only skipped whole group {group}: {'; '.join(reasons)}")
        else:
            accepted.extend(members)
    return sorted(accepted)


def load_records(
    sha: str,
    log_dir: Path = LOG_DIR,
    smoke: bool = False,
    verify_logs: bool = True,
    allow_gate_only: bool = False,
    selected_logs: list[Path] | None = None,
    allow_empty: bool = False,
) -> dict[str, dict]:
    if not log_dir.is_dir():
        raise SystemExit(f"missing {log_dir}: run scripts/gate_calibration.sh first")
    logs = sorted(log_dir.glob("*.log")) if selected_logs is None else selected_logs
    if not logs and allow_empty:
        return {}
    if verify_logs and not smoke:
        logs = drop_failed_confirmations(logs)
        check_logs(logs, sha)
    if not verify_logs and not smoke:
        try:
            coverage_decision_logs.audit_logs(logs, sha)
        except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
            raise SystemExit(f"invalid frozen-family decision log: {error}") from error
    records = merged_records(logs, smoke, None if smoke or not verify_logs else sha)
    # A pass/fail-only re-run (the record-less groups) measures no record but still
    # refreshes the group ledger; --keep-attested keeps every record that stands.
    if not records and allow_gate_only and gate_ledger(logs, sha):
        return records
    if not records:
        raise SystemExit(
            f"no calibration-record lines in {log_dir}: nothing measured, so nothing to commit"
        )
    for record in records.values():
        record["calibration_sha"] = sha
        record["surface_list_blob"] = surface_list_blob(sha)
    return records


def surface_list_blob(sha: str) -> str:
    """The git blob of the surface list at `sha`: what the record's facets were derived from."""
    done = subprocess.run(
        ["git", "rev-parse", f"{sha}:scripts/calibration_surface.list"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise SystemExit(
            f"cannot read scripts/calibration_surface.list at {sha}: {done.stderr.strip()}"
        )
    return done.stdout.strip()


def toml_value(key: str, value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and key != "id":
        return repr(value)
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {toml_value(k, v)}" for k, v in value.items()) + " }"
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return "[\n" + "".join(f"  {toml_value(key, item)},\n" for item in value) + "]"
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(key, item) for item in value) + "]"
    return '"' + str(value).replace('"', '\\"') + '"'


def tag_facets(records: dict[str, dict]) -> None:
    """Set each record's `facets` from the record itself and the surface list."""
    if not records:
        return
    surface = facets.load_surface()
    if surface.errors:
        raise SystemExit("scripts/calibration_surface.list: " + "; ".join(surface.errors))
    refs = facets.references(surface)
    for rec in records.values():
        rec["facets"] = facets.record_facets(rec, surface, refs)


def keep_attested(measured: dict[str, dict]) -> dict[str, dict]:
    """Existing records the logs did not re-measure and that still stand."""
    surface = facets.load_surface()
    registry = facets.load_records(OUT)
    existing = [rec for rec in registry if rec["id"] not in measured]
    kept: dict[str, dict] = {}
    # A record attested_by_replay under a valid waiver is not stale, so it is
    # kept; the waiver's evidence is checked against the registry as it stood.
    for assessment in facets.assess(surface, existing, registry=registry):
        stale = {rec["id"] for rec in assessment.stale}
        for rec in assessment.records:
            if assessment.resolved and rec["id"] not in stale:
                kept[rec["id"]] = rec
            else:
                print(f"dropped {rec['id']}: owes a re-measurement the logs do not contain")
    return kept


def write_registry(records: dict[str, dict], out: Path = OUT, *, retag: bool = True) -> None:
    if retag:
        tag_facets(records)
    lines = [HEADER]
    for rid in sorted(records):
        rec = records[rid]
        rec.setdefault("surface_list_blob", surface_list_blob(rec["calibration_sha"]))
        lines.append("[[record]]")
        for key in FIELDS:
            if key not in rec:
                raise SystemExit(f"record {rid} is missing {key}; re-run the coverage tests")
            lines.append(f"{key} = {toml_value(key, rec[key])}")
            if key == "role":
                lines += [
                    f"{opt} = {toml_value(opt, rec[opt])}"
                    for opt in OPTIONAL_FIELDS
                    if rec.get(opt) is not None
                ]
        lines.append("")
    write_text(out, "\n".join(lines))


REPORTED_LEVEL = 0.95


def all_boundary_at_reported_level(ids: list[str], records: dict[str, dict]) -> bool:
    """Whether the cell's records measure the reported level and none of them passes it."""
    at_level = [
        records[rid] for rid in ids if abs(float(records[rid]["nominal"]) - REPORTED_LEVEL) < 1e-9
    ]
    return bool(at_level) and all(
        rec["boundary"] and "confirmed_passing_range" not in rec for rec in at_level
    )


def replace_pair(
    block: str, calibration: list[str] | None, reason: str | None, boundary_only: bool = False
) -> str:
    """Replace the calibration / calibration_reason pair inside one TOML block.

    A cell whose every reported-level record is a boundary keeps its record list
    and also states `boundary_record`: the list says what was measured, the
    reason says that none of it is a nominal pass."""
    kept = [
        line
        for line in block.splitlines()
        if not line.startswith("calibration = ") and not line.startswith("calibration_reason = ")
    ]
    while kept and not kept[-1].strip():
        kept.pop()
    if calibration:
        ids = ", ".join(f'"{rid}"' for rid in calibration)
        kept.append(f"calibration = [{ids}]")
        if boundary_only:
            kept.append('calibration_reason = "boundary_record"')
    else:
        kept.append(f'calibration_reason = "{reason}"')
    return "\n".join(kept) + "\n\n"


def conditional_record_dgp_labels(path: Path) -> set[str]:
    """Exact literal labels emitted by two-branch RecordKey DGP expressions.

    Comments and unrelated strings cannot establish a DGP reference. Both
    branches must be literal labels; arbitrary computed expressions do not resolve.
    """
    from test_evidence import rust_items

    code = rust_items(path)[0].code
    pattern = r'\bdgp\s*:\s*if\s+[^{};]+\{\s*"([^"\\]+)"\s*\}\s*else\s*\{\s*"([^"\\]+)"\s*\}'
    return {label for pair in re.findall(pattern, code) for label in pair}


CALIBRATION_OWNER_FIELDS = frozenset({
    "estimator", "source", "source_symbol", "evidence_test", "evidence_assertion",
})


def calibration_estimator_evidence_problems(cell: dict, root: Path = ROOT) -> list[str]:
    """Scientific owners require real native code and non-ignored asserting proof.

    This ownership declaration never changes selectable estimator names or grants
    calibration to another method. Native execution/current records remain required.
    """
    from promotion_source import assertion_problems, nontest_rust
    from test_evidence import static_rust_test

    entries = cell.get("calibration_estimators", [])
    if not isinstance(entries, list):
        return ["calibration_estimators must be an evidence list"]
    errors = []
    seen = set()
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != CALIBRATION_OWNER_FIELDS
                or any(not isinstance(value, str) or not value for value in entry.values())):
            errors.append("calibration_estimators entries require exact nonempty native/evidence fields")
            continue
        name = entry["estimator"]
        if name in seen or name in cell.get("estimators", []):
            errors.append(f"duplicate scientific calibration owner {name}")
        seen.add(name)
        source, test = root / entry["source"], root / entry["evidence_test"]
        if (not entry["source"].startswith("crates/") or "/src/" not in entry["source"]
                or source.suffix != ".rs" or not source.is_file()
                or not source.resolve().is_relative_to(root.resolve())):
            errors.append(f"{name}: scientific native source must exist inside crates/*/src")
        elif not re.search(r"\b" + re.escape(entry["source_symbol"]) + r"\b", nontest_rust(source)):
            errors.append(f"{name}: native source symbol is absent outside comments/tests")
        if (not entry["evidence_test"].startswith("crates/") or test.suffix != ".rs"
                or not test.is_file() or not test.resolve().is_relative_to(root.resolve())):
            errors.append(f"{name}: native scientific ownership evidence must exist")
        else:
            errors.extend(f"{name}: {error}" for error in
                          static_rust_test(test, entry["evidence_assertion"]).problems)
            errors.extend(f"{name}: {error}" for _, error in
                          assertion_problems(test, entry["evidence_assertion"]))
    return errors


def scientific_calibration_estimators(cell: dict) -> set[str]:
    """Declared algorithm identities, separate from public estimator selectors."""
    entries = cell.get("calibration_estimators", [])
    if not isinstance(entries, list):
        return set()
    return {entry["estimator"] for entry in entries if isinstance(entry, dict)
            and set(entry) == CALIBRATION_OWNER_FIELDS
            and all(isinstance(value, str) and value for value in entry.values())}


def licensed_record_matches(cell: dict, record: dict) -> bool:
    """Exact support coordinates and declared estimator ownership, never axes alone."""
    structure = "graph_posterior" if cell.get("structure") == "graph_posterior" else "fixed"
    return (
        record.get("query"), record.get("graph_class"), record.get("inference"),
        record.get("structure"),
    ) == (
        cell.get("query"), cell.get("graph_class"), cell.get("inference"), structure,
    ) and record.get("estimator") in (set(cell.get("estimators", [])) | scientific_calibration_estimators(cell))


def licensed_cell_calibration_problems(cell: dict, records: dict[str, dict]) -> list[str]:
    """Require all owned coordinate records and refuse foreign/missing bindings.

    Ownership is declared by the support cell, not inferred from a new passing
    record. Posterior engines of another estimator cannot upgrade that cell.
    """
    bound = cell.get("calibration", [])
    errors = []
    for rid in bound:
        record = records.get(rid)
        if record is None:
            errors.append(f"unknown bound record {rid}")
        elif not licensed_record_matches(cell, record):
            errors.append(f"bound record {rid} has foreign estimator or scientific coordinate")
    owned = {rid for rid, record in records.items() if licensed_record_matches(cell, record)}
    missing = sorted(owned - set(bound))
    if missing:
        errors.append(f"missing required owned records {missing}")
    return errors


def sync_licensed_cells(
    records: dict[str, dict],
    only_cells: dict[tuple[str, str, str, str, str], list[str]] | None = None,
) -> int:
    # Collection refreshes measurements, not public-route ownership. New bindings
    # require explicit --only-cell authorization and estimator compatibility.
    cells = tomllib.loads(read_text(LICENSED)).get("cell", [])
    text = read_text(LICENSED)
    blocks = text.split("[[cell]]")
    out = [blocks[0]]
    changed = 0
    seen: set[tuple[str, str, str, str, str]] = set()
    for block, cell in zip(blocks[1:], cells, strict=True):
        cell_key = (
            cell["query"], cell["graph_class"], cell["structure"],
            cell["inference"], cell["validation"],
        )
        if only_cells is not None and cell_key not in only_cells:
            out.append(block)
            continue
        seen.add(cell_key)
        evidence_errors = calibration_estimator_evidence_problems(cell)
        if evidence_errors:
            raise SystemExit(f"licensed cell {cell_key}: invalid scientific ownership: {evidence_errors}")
        structures = ["graph_posterior"] if cell["structure"] == "graph_posterior" else ["fixed"]
        if only_cells is None:
            ids = list(cell.get("calibration", []))
            missing = [rid for rid in ids if rid not in records]
            if missing:
                raise SystemExit(f"licensed cell {cell_key}: bound calibration records absent: {missing}")
            incompatible = [rid for rid in ids if not licensed_record_matches(cell, records[rid])]
            if incompatible:
                raise SystemExit(
                    f"licensed cell {cell_key}: bound records not owned by the licensed cell: {incompatible}"
                )
            if not ids:
                # An unmeasured cell must not acquire a claim from an unrelated
                # new estimator sharing broad support axes.
                out.append(block)
                continue
        else:
            ids = sorted(only_cells[cell_key])
            for rid in ids:
                rec = records.get(rid)
                if rec is None or (
                    rec["query"], rec["graph_class"], rec["inference"], rec["structure"]
                ) != (cell["query"], cell["graph_class"], cell["inference"], structures[0]):
                    raise SystemExit(f"--only-cell {cell_key}: incompatible or absent record {rid}")
                if not licensed_record_matches(cell, rec):
                    raise SystemExit(
                        f"--only-cell {cell_key}: record {rid} estimator {rec.get('estimator')!r} "
                        "is not owned by the licensed cell"
                    )
        # A point-only cell states `no_interval_reported` itself and keeps it: the
        # runtime publishes no interval there, so there is nothing to measure.
        point_only = (
            cell["query"] in NO_INTERVAL_QUERIES
            or cell.get("calibration_reason") == "no_interval_reported"
        )
        reason = (
            "no_interval_reported" if not ids and point_only else "estimator_grid_not_measured"
        )
        new_block = replace_pair(block, ids, reason, all_boundary_at_reported_level(ids, records))
        changed += int(new_block != block)
        out.append(new_block)
    if only_cells is not None and seen != only_cells.keys():
        raise SystemExit(f"unknown --only-cell coordinates: {sorted(only_cells.keys() - seen)}")
    write_text(LICENSED, "[[cell]]".join(out))
    return changed


def sync_estimator_rows(records: dict[str, dict]) -> int:
    text = read_text(ESTIMATE)
    rows = tomllib.loads(text).get("capabilities", [])
    blocks = text.split("[[capabilities]]")
    out = [blocks[0]]
    changed = 0
    for block, row in zip(blocks[1:], rows, strict=True):
        if row.get("id") not in ESTIMATOR_ROW_IDS:
            out.append(block)
            continue
        estimators = ESTIMATOR_ROW_IDS[row["id"]]
        ids = sorted(rid for rid, rec in records.items() if rec["estimator"] in estimators)
        new_block = replace_pair(block, ids, "estimator_grid_not_measured")
        changed += int(new_block != block)
        out.append(new_block)
    write_text(ESTIMATE, "[[capabilities]]".join(out))
    return changed


def append_attested_records(
    existing: dict[str, dict], measured: dict[str, dict], *,
    mode: str = "--append-attested", replace_ids: list[str] | None = None,
) -> dict[str, dict]:
    """Preserve old evidence except explicitly named, freshly measured records."""
    allowed = set(replace_ids or [])
    invalid = sorted(allowed - (existing.keys() & measured.keys()))
    if invalid:
        raise SystemExit("--replace-record requires each ID in both existing and new evidence:\n  " + "\n  ".join(invalid))
    collisions = sorted((existing.keys() & measured.keys()) - allowed)
    if collisions:
        raise SystemExit(f"{mode} refuses existing record ids:\n  " + "\n  ".join(collisions))
    return {**existing, **measured}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", help="commit the logs were measured at (default: HEAD)")
    parser.add_argument(
        "--append-attested",
        action="store_true",
        help="append new attested records without deleting existing, possibly stale rows; "
        "requires --only-cell and does not rewrite the global gate ledger",
    )
    parser.add_argument(
        "--only-cell",
        action="append",
        default=[],
        metavar="QUERY:GRAPH:STRUCTURE:INFERENCE:VALIDATION=ID,ID",
        help="exact licensed cell and record IDs to sync when appending (repeat per cell)",
    )
    parser.add_argument(
        "--defer-generated", action="store_true",
        help="with --no-cells, collect evidence only; regenerate support assets after the final harvest",
    )
    parser.add_argument(
        "--no-cells",
        action="store_true",
        help="do not rewrite support_licensed.toml / estimate.toml",
    )
    parser.add_argument(
        "--keep-existing", action="store_true",
        help="append only new record IDs, preserving existing rows and gate evidence even when stale",
    )
    parser.add_argument(
        "--passing-only", action="store_true",
        help="with --keep-existing, harvest only whole completed passing groups; retain failures in logs",
    )
    parser.add_argument(
        "--replace-record", action="append", default=[], metavar="ID",
        help="with --keep-existing, permit fresh evidence to replace this existing record (repeatable)",
    )
    parser.add_argument(
        "--keep-attested",
        action="store_true",
        help="keep existing records the logs do not re-measure whose facets have not drifted",
    )
    parser.add_argument(
        "--retag",
        action="store_true",
        help="only recompute the facets of the existing records",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="collect a wiring smoke run's lines into the scratch registry --out",
    )
    parser.add_argument(
        "--allow-removal",
        nargs="+",
        default=[],
        metavar="ID",
        help="record ids the registry holds that this run is allowed to drop "
        "(without --keep-attested the registry becomes exactly the logs)",
    )
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR, help="logs to collect")
    parser.add_argument("--out", type=Path, help="scratch registry path (with --smoke only)")
    args = parser.parse_args()
    if args.defer_generated and not args.no_cells:
        raise SystemExit("--defer-generated requires --no-cells")
    if args.passing_only and not args.keep_existing:
        raise SystemExit("--passing-only requires --keep-existing")
    if args.replace_record and not args.keep_existing:
        raise SystemExit("--replace-record requires --keep-existing")
    if args.keep_existing and (args.keep_attested or args.append_attested or args.retag or args.allow_removal):
        raise SystemExit("--keep-existing forbids --keep-attested/--append-attested/--retag/--allow-removal")
    if args.append_attested and (args.keep_attested or args.no_cells or not args.only_cell):
        raise SystemExit("--append-attested requires --only-cell and forbids --keep-attested/--no-cells")
    if args.only_cell and not args.append_attested:
        raise SystemExit("--only-cell is only valid with --append-attested")
    only_cells = {}
    for raw in args.only_cell:
        coordinate, separator, id_list = raw.partition("=")
        parts = tuple(coordinate.split(":"))
        ids = id_list.split(",") if separator else []
        if len(parts) != 5 or any(not part for part in parts) or not ids or any(not rid for rid in ids):
            raise SystemExit(f"bad --only-cell coordinate: {raw}")
        if parts in only_cells or len(set(ids)) != len(ids):
            raise SystemExit(f"duplicate --only-cell coordinate or record ID: {raw}")
        only_cells[parts] = ids
    if args.smoke or args.out:
        if not (args.smoke and args.out):
            raise SystemExit("--smoke and --out go together: a smoke run writes a scratch registry")
        if args.out.resolve() == OUT.resolve():
            raise SystemExit(f"--smoke never writes {OUT.relative_to(ROOT)}")
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        records = load_records(sha, args.log_dir, smoke=True)
        write_registry(records, args.out)
        print(
            f"wrote {len(records)} smoke records to {args.out} (not a registry; measures nothing)"
        )
        return 0
    if args.retag:
        existing = {rec["id"]: rec for rec in facets.load_records(OUT)}
        write_registry(existing)
        print(f"retagged the facets of {len(existing)} records in {OUT.relative_to(ROOT)}")
        return 0
    sha = args.sha or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit(f"calibration SHA must be 40 hex, got {sha!r}")
    collection_logs = sorted(args.log_dir.glob("*.log"))
    if args.passing_only:
        collection_logs = passing_group_logs(collection_logs, sha)
    records = load_records(sha, args.log_dir, allow_gate_only=args.keep_attested or args.keep_existing,
                           selected_logs=collection_logs, allow_empty=args.passing_only)
    if not args.sha:
        tag_facets(records)
        surface = facets.load_surface()
        depended = set().union(*(rec["facets"] for rec in records.values()))
        dirty = [
            rel
            for rel in facets.changed_paths(surface, sha)
            if (surface.facet_of(rel) or facets.CORE) in depended
        ]
        if dirty:
            raise SystemExit(
                "the statistical surface differs from HEAD, so these logs do not describe "
                "a commit; commit first (or pass --sha):\n  " + "\n  ".join(dirty)
            )
    if args.append_attested:
        tag_facets(records)
        existing = {rec["id"]: rec for rec in facets.load_records(OUT)}
        records = append_attested_records(existing, records)
    elif args.keep_existing:
        if args.sha:
            tag_facets(records)
        existing = {rec["id"]: rec for rec in facets.load_records(OUT)}
        replacements = args.replace_record
        if args.passing_only:
            unknown = set(replacements) - existing.keys()
            if unknown:
                raise SystemExit("--replace-record does not exist in old evidence: " + ", ".join(sorted(unknown)))
            deferred = set(replacements) - records.keys()
            if deferred:
                print(f"passing-only deferred {len(deferred)} replacements whose groups did not complete successfully")
            replacements = [rid for rid in replacements if rid in records]
        records = append_attested_records(existing, records, mode="--keep-existing", replace_ids=replacements)
        print(f"preserved {len(existing) - len(set(replacements))} existing records; replaced {len(set(replacements))} explicitly named records with fresh evidence")
    elif not args.keep_attested:
        dropped = sorted(
            {rec["id"] for rec in facets.load_records(OUT)} - set(records) - set(args.allow_removal)
        )
        if dropped:
            raise SystemExit(
                f"the logs do not re-measure {len(dropped)} record(s) the registry holds, so this "
                "run would silently delete them (a renamed or unrun group?). Run the missing "
                "groups, use --keep-attested, or name each id in --allow-removal:\n  "
                + "\n  ".join(dropped)
            )
    if args.keep_attested:
        kept = keep_attested(records)
        print(f"kept {len(kept)} attested records the logs did not re-measure")
        records = {**kept, **records}
    write_registry(records, retag=not (args.append_attested or args.keep_existing))
    print(f"wrote {len(records)} records to {OUT.relative_to(ROOT)} at {sha}")
    if args.append_attested:
        cells = sync_licensed_cells(records, only_cells)
        print(f"appended {len(records) - len(existing)} records; synced {cells} selected licensed cells")
        subprocess.run([sys.executable, str(GENERATOR)], cwd=ROOT, check=True)
        return 0
    previous = (
        {g["group"]: g for g in tomllib.loads(read_text(GATES)).get("gate", [])}
        if (args.keep_attested or args.keep_existing) and GATES.is_file()
        else None
    )
    write_gate_ledger(
        gate_ledger(collection_logs, sha),
        previous,
    )
    print(f"wrote the record-less group ledger to {GATES.relative_to(ROOT)}")
    if not args.no_cells:
        cells = sync_licensed_cells(records)
        rows = sync_estimator_rows(records)
        print(f"rewrote the calibration pair on {cells} licensed cells and {rows} estimator rows")
    if args.defer_generated:
        print("deferred generated support assets; evidence registries are preserved")
    else:
        subprocess.run([sys.executable, str(GENERATOR)], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
