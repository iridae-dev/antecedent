#!/usr/bin/env python3
"""First answer for propensity-weighted adjustment, at a chosen row count.

The population, graph, estimator and requested uncertainty are those of
`examples/python/propensity_weighting.py` (z confounds treatment and outcome, effect 2, 30
bootstrap replicates, seed 11); only the number of rows is a parameter (1200 by default). One
`analyze` call returns the effect, the bootstrap standard error and the overlap diagnostics
(effective sample size, minimum propensity). Row synthesis sits outside the timed section.

    python parity/performance_lanes/entry_points/adjustment.py --role control
    python parity/performance_lanes/entry_points/adjustment.py --role candidate --rows 1000000
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _entry import arguments, digest_arrays, run_entry
from antecedent import AverageEffect, analyze

DATA_SEED = 7
EXECUTION_SEED = 11
ROWS = 1200
BOOTSTRAP = 30


def synthesize(rows: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(DATA_SEED)
    z = rng.normal(size=rows)
    t = (rng.uniform(size=rows) < 1.0 / (1.0 + np.exp(-(-0.4 + 0.9 * z)))).astype(
        np.float64
    )
    y = 2.0 * t + z + rng.normal(0.0, 0.4, size=rows)
    return {"t": t, "y": y, "z": z}


def main() -> None:
    args = arguments(__doc__ or "")
    data = synthesize(args.rows or ROWS)

    def section() -> Any:
        return analyze(
            data,
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=AverageEffect(treatment="t", outcome="y"),
            estimator="propensity.weighting",
            bootstrap=BOOTSTRAP,
            seed=EXECUTION_SEED,
            threads=args.threads,
        )

    def fields(result: Any) -> dict[str, Any]:
        return {
            "effect": result.answer.value,
            "answer_kind": result.answer.kind,
            "se_bootstrap": result.estimate.se_bootstrap,
            "estimator_id": result.estimate.estimator_id,
            "overlap_ess": result.estimate.overlap_ess,
            "overlap_propensity_min": result.estimate.overlap_propensity_min,
            "calibration": str(result.calibration.status),
        }

    run_entry(
        "adjustment",
        args,
        digest_arrays(data),
        section,
        fields,
        required=("effect", "se_bootstrap", "overlap_ess", "overlap_propensity_min"),
    )


if __name__ == "__main__":
    main()
