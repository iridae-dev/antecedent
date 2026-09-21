#!/usr/bin/env python3
"""Collect `calibration-record` lines into the coverage registry.

One command after a calibration run:

    python3 scripts/collect_coverage_records.py

It reads `target/calibration-records/*.log` (written by
`scripts/gate_calibration.sh`), writes `parity/coverage_records.toml` stamped
with the SHA of the commit the logs were measured at, rewrites the
`calibration` / `calibration_reason` pair on every licensed support cell and
estimator row from those records, and regenerates
`crates/antecedent-io/src/coverage_records_data.rs`.

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

A grid point that was rechecked at more replicates writes
`<group>.p<k>.recheck.log`; its lines replace that point's first run, because
the recheck's verdict is the one the gate takes.

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

`--smoke --log-dir <dir> --out <file>` collects the lines of a wiring smoke run
(`ANTECEDENT_CALIBRATION_SMOKE=1` at a reduced replicate count) into a scratch
registry. Smoke lines measure nothing: every other invocation refuses them, and
`--smoke` refuses to write `parity/coverage_records.toml` or touch any registry.

`--sha <sha>` overrides the stamped SHA (for collecting logs measured at
another commit); `--no-cells` leaves the registries alone. Without `--sha` the
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import calibration_facets as facets  # noqa: E402

LOG_DIR = ROOT / "target" / "calibration-records"
OUT = ROOT / "parity" / "coverage_records.toml"
LICENSED = ROOT / "parity" / "support_licensed.toml"
ESTIMATE = ROOT / "parity" / "estimate.toml"
GENERATOR = ROOT / "scripts" / "generate_support_matrix_docs.py"

HEADER = """# Coverage records bound onto executions at claim time.
#
# Every row is a `calibration-record` line emitted by a coverage test through
# `CoverageTally::for_record` (crates/antecedent/tests/common/calibration.rs)
# and collected by `scripts/collect_coverage_records.py`, which stamps the SHA
# of the commit the logs were measured at. Do not edit rows by hand; re-run
# the collector. `scripts/gate_parity_schema.sh` checks every row.
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
)

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
    "estimate.two_stage": {"frontdoor.linear_two_stage"},
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


def record_lines(logs: list[Path], smoke: bool = False) -> dict[str, dict[int, dict]]:
    """`calibration-record` payloads by record id and grid point.

    A point's recheck log (`<group>.p<k>.recheck.log`) replaces that point's
    first run (`<group>.p<k>.log`): the recheck's verdict is the one the gate
    takes, and a point that failed it emitted no line at all, so the first
    run's line must not survive."""
    names = {log.name for log in logs}
    chosen = [
        log
        for log in sorted(logs)
        if log.name.endswith(".recheck.log")
        or log.name.removesuffix(".log") + ".recheck.log" not in names
    ]
    out: dict[str, dict[int, dict]] = {}
    for log in chosen:
        recheck = log.name.endswith(".recheck.log")
        for line in log.read_text(errors="ignore").splitlines():
            if not line.startswith("calibration-record "):
                continue
            payload = json.loads(line.split(" ", 1)[1])
            rid = str(payload["id"])
            if bool(payload.pop("smoke", False)) != smoke:
                raise SystemExit(
                    f"{log.name}: record {rid} is a wiring smoke line; it measured nothing and "
                    "only `--smoke` collects it, into a scratch registry"
                    if not smoke
                    else f"{log.name}: record {rid} is not a smoke line; --smoke collects only a "
                    "smoke run's logs"
                )
            if "grid_point" not in payload:
                raise SystemExit(
                    f"{log.name}: record {rid} carries no grid_point; it was measured before "
                    "the sample-size grid, so re-measure it with scripts/gate_calibration.sh"
                )
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
        grid=[
            {
                "point": int(p["grid_point"]),
                "n_min": int(p["n_min"]),
                "n_max": int(p["n_max"]),
                "observed": p["observed"],
                "mcse": p["mcse"],
                "replicates": int(p["replicates"]),
                "boundary": bool(p["boundary"]),
                "role": str(p["role"]),
            }
            for p in ordered
        ],
    )
    return record


def merged_records(logs: list[Path], smoke: bool = False) -> dict[str, dict]:
    """Every record the logs measured, merged over its grid points."""
    return {rid: merge_grid(rid, points) for rid, points in record_lines(logs, smoke).items()}


def load_records(sha: str, log_dir: Path = LOG_DIR, smoke: bool = False) -> dict[str, dict]:
    if not log_dir.is_dir():
        raise SystemExit(f"missing {log_dir}: run scripts/gate_calibration.sh first")
    records = merged_records(sorted(log_dir.glob("*.log")), smoke)
    if not records:
        raise SystemExit(
            f"no calibration-record lines in {log_dir}: nothing measured, so nothing to commit"
        )
    for record in records.values():
        record["calibration_sha"] = sha
    return records


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


def write_registry(records: dict[str, dict], out: Path = OUT) -> None:
    tag_facets(records)
    lines = [HEADER]
    for rid in sorted(records):
        rec = records[rid]
        lines.append("[[record]]")
        for key in FIELDS:
            if key not in rec:
                raise SystemExit(f"record {rid} is missing {key}; re-run the coverage tests")
            lines.append(f"{key} = {toml_value(key, rec[key])}")
        lines.append("")
    out.write_text("\n".join(lines))


def replace_pair(block: str, calibration: list[str] | None, reason: str | None) -> str:
    """Replace the calibration / calibration_reason pair inside one TOML block."""
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
    else:
        kept.append(f'calibration_reason = "{reason}"')
    return "\n".join(kept) + "\n\n"


def sync_licensed_cells(records: dict[str, dict]) -> int:
    cells = tomllib.loads(LICENSED.read_text()).get("cell", [])
    by_coordinate: dict[tuple[str, str, str, str], list[str]] = {}
    for rid, rec in records.items():
        key = (rec["query"], rec["graph_class"], rec["inference"], rec["structure"])
        by_coordinate.setdefault(key, []).append(rid)
    text = LICENSED.read_text()
    blocks = text.split("[[cell]]")
    out = [blocks[0]]
    changed = 0
    for block, cell in zip(blocks[1:], cells, strict=True):
        structures = ["graph_posterior"] if cell["structure"] == "graph_posterior" else ["fixed"]
        ids = sorted(
            rid
            for structure in structures
            for rid in by_coordinate.get(
                (cell["query"], cell["graph_class"], cell["inference"], structure), []
            )
        )
        reason = (
            "no_interval_reported"
            if not ids and cell["query"] in NO_INTERVAL_QUERIES
            else "estimator_grid_not_measured"
        )
        new_block = replace_pair(block, ids, reason)
        changed += int(new_block != block)
        out.append(new_block)
    LICENSED.write_text("[[cell]]".join(out))
    return changed


def sync_estimator_rows(records: dict[str, dict]) -> int:
    text = ESTIMATE.read_text()
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
    ESTIMATE.write_text("[[capabilities]]".join(out))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", help="commit the logs were measured at (default: HEAD)")
    parser.add_argument(
        "--no-cells",
        action="store_true",
        help="do not rewrite support_licensed.toml / estimate.toml",
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
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR, help="logs to collect")
    parser.add_argument("--out", type=Path, help="scratch registry path (with --smoke only)")
    args = parser.parse_args()
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
    records = load_records(sha, args.log_dir)
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
    if args.keep_attested:
        kept = keep_attested(records)
        print(f"kept {len(kept)} attested records the logs did not re-measure")
        records = {**kept, **records}
    write_registry(records)
    print(f"wrote {len(records)} records to {OUT.relative_to(ROOT)} at {sha}")
    if not args.no_cells:
        cells = sync_licensed_cells(records)
        rows = sync_estimator_rows(records)
        print(f"rewrote the calibration pair on {cells} licensed cells and {rows} estimator rows")
    subprocess.run([sys.executable, str(GENERATOR)], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
