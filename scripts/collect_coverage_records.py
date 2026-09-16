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

A group that was rechecked at more replicates writes
`<group>.recheck.log`; its records replace the first run's, because the
recheck's verdict is the one the gate takes.

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
    "estimate.two_stage": {"frontdoor.two_stage"},
    "estimate.conditional": {"conditional.linear.adjustment", "bayesian.conditional"},
    "estimate.temporal_sequential": {"temporal.sequential.gcomp"},
    "estimate.mediation.linear": {"mediation.linear", "temporal.mediation"},
}

NO_INTERVAL_QUERIES = {"Counterfactual", "AnomalyAttribution", "ChangeAttribution"}


def load_records(sha: str) -> dict[str, dict]:
    if not LOG_DIR.is_dir():
        raise SystemExit(f"missing {LOG_DIR}: run scripts/gate_calibration.sh first")
    records: dict[str, dict] = {}
    # A rechecked group's records come only from the recheck run: that run's
    # verdict is the one the gate takes, and a group that failed it emitted no
    # record at all, so the first run's rows must not survive.
    logs = []
    for log in sorted(LOG_DIR.glob("*.log")):
        if log.name.endswith(".recheck.log"):
            logs.append(log)
        elif not log.with_suffix(".recheck.log").exists():
            logs.append(log)
    for log in logs:
        recheck = log.name.endswith(".recheck.log")
        for line in log.read_text(errors="ignore").splitlines():
            if not line.startswith("calibration-record "):
                continue
            payload = json.loads(line.split(" ", 1)[1])
            rid = payload["id"]
            seen = records.get(rid)
            if seen is not None and not recheck:
                if seen["replicates"] == payload["replicates"]:
                    raise SystemExit(
                        f"duplicate record id {rid} in {log.name}: two tallies emit the same id"
                    )
                # The same measurement re-run at more replicates: keep the
                # more precise one (a precision recheck, however it was run).
                if seen["replicates"] > payload["replicates"]:
                    continue
            payload["calibration_sha"] = sha
            records[rid] = payload
    if not records:
        raise SystemExit(
            f"no calibration-record lines in {LOG_DIR}: nothing measured, so nothing to commit"
        )
    return records


def toml_value(key: str, value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and key != "id":
        return repr(value)
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


def write_registry(records: dict[str, dict]) -> None:
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
    OUT.write_text("\n".join(lines))


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
    args = parser.parse_args()
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
    records = load_records(sha)
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
