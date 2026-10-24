#!/usr/bin/env python3
"""Prepared query family: compile once, estimate, then retarget repeatedly.

The timed section is the whole prepared-family session on one table: `PreparedBatch.prepare`
over three average-effect queries that share nuisance fits, one `estimate` with the analytic
standard errors, then eight retargets of a declared claim family under different row weights.
Each retarget returns the joint score covariance, the contrast with its standard error and
the family-level simultaneous interval (20000 Monte Carlo draws). The weights are caller
inputs derived from the retained rows, so they are built inside the section in both roles.
Data synthesis sits outside it.

    python parity/performance_lanes/entry_points/prepared_query_families.py --role control
    python parity/performance_lanes/entry_points/prepared_query_families.py --role candidate

`--mode solo` is the independent-call baseline for the reuse lane: every one of the eight
retargets starts from nothing, so each repeats `prepare` and `estimate` instead of reusing
the prepared family. The data, queries, uncertainty, claims and contrast are the same.
"""

from __future__ import annotations

from typing import Any

import antecedent
import numpy as np
from _entry import arguments, digest_arrays, run_entry
from antecedent.estimation import PreparedBatch, RetargetClaim, RetargetContrast

DATA_SEED = 11
EXECUTION_SEED = 11
ROWS = 5000
TILTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
SIMULTANEOUS_DRAWS = 20_000
GRAPH = [
    ("z", "t1"),
    ("z", "t2"),
    ("z", "y1"),
    ("z", "y2"),
    ("t1", "y1"),
    ("t1", "y2"),
    ("t2", "y1"),
    ("t2", "y2"),
]
QUERIES = [
    antecedent.AverageEffect("t1", "y1"),
    antecedent.AverageEffect("t1", "y2"),
    antecedent.AverageEffect("t2", "y1"),
]


def synthesize(rows: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(DATA_SEED)
    z = rng.normal(size=rows)
    t1 = (rng.uniform(size=rows) < 1 / (1 + np.exp(-(-0.2 + 0.8 * z)))).astype(float)
    t2 = (rng.uniform(size=rows) < 1 / (1 + np.exp(-(0.1 - 0.6 * z)))).astype(float)
    y1 = 2.0 * t1 + 0.5 * t2 + z + 0.3 * rng.normal(size=rows)
    y2 = -t1 + 1.5 * t2 - 0.5 * z + 0.3 * rng.normal(size=rows)
    return {"t1": t1, "t2": t2, "y1": y1, "y2": y2, "z": z}


def main() -> None:
    args = arguments(
        __doc__ or "",
        lambda parser: parser.add_argument(
            "--mode",
            choices=("prepared", "solo"),
            default="prepared",
            help="solo repeats prepare and estimate for every retarget (independent calls)",
        ),
    )
    data = synthesize(args.rows or ROWS)

    def prepare() -> tuple[Any, Any]:
        batch = PreparedBatch.prepare(
            data,
            graph=GRAPH,
            queries=QUERIES,
            estimator="aipw",
            refute=False,
            bootstrap=0,
            seed=EXECUTION_SEED,
            threads=args.threads,
        )
        return batch, batch.estimate(data, seed=EXECUTION_SEED, threads=args.threads)

    def section() -> Any:
        batch, estimates = prepare()
        rows = list(batch.retarget_rows() or ())
        z = np.asarray(data["z"])[rows]
        snapshot = batch.retarget_snapshot()
        reports = []
        for index, tilt in enumerate(TILTS):
            if args.mode == "solo" and index > 0:
                batch, estimates = prepare()
                snapshot = batch.retarget_snapshot()
            claims = [
                RetargetClaim("up", 0, np.exp(tilt * z), depends_on=["z"]),
                RetargetClaim("down", 0, np.exp(-tilt * z), depends_on=["z"]),
                RetargetClaim("other_treatment", 2, np.exp(tilt * z), depends_on=["z"]),
            ]
            report = batch.retarget(
                claims,
                [RetargetContrast("up_minus_down", {"up": 1.0, "down": -1.0})],
                expected_snapshot=snapshot,
                simultaneous_draws=SIMULTANEOUS_DRAWS,
                simultaneous_seed=3,
            )
            reports.append((report, report.simultaneous_interval()))
        return estimates, reports

    def fields(result: Any) -> dict[str, Any]:
        estimates, reports = result
        return {
            "effects": [r.estimate.ate for r in estimates],
            "standard_errors": [r.estimate.se_analytic for r in estimates],
            "retarget_complete": [report.complete for report, _ in reports],
            "claim_values": [[m.value for m in report.claims] for report, _ in reports],
            "claim_standard_errors": [[m.std_error for m in report.claims] for report, _ in reports],
            "contrast": [report.contrasts[0].value for report, _ in reports],
            "contrast_standard_error": [report.contrasts[0].std_error for report, _ in reports],
            "simultaneous_critical_value": [band.critical_value for _, band in reports],
            "simultaneous_bounds": [
                [(m.lower, m.upper) for m in band.members] for _, band in reports
            ],
        }

    run_entry(
        "prepared_query_families",
        args,
        digest_arrays(data),
        section,
        fields,
        required=("standard_errors", "contrast_standard_error", "simultaneous_bounds"),
    )


if __name__ == "__main__":
    main()
