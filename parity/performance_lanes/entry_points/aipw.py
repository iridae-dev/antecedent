#!/usr/bin/env python3
"""First answer for the doubly robust workflow, at a chosen row count.

The table, graph, estimators and uncertainty are those of
`examples/python/ihdp_propensity_e2e.py`: the IHDP-like population (25 covariates, effect 4,
the first six confound), then inverse-probability weighting with the cheap refutation suite
and AIPW, each with 40 bootstrap replicates and seed 11. Only the row count is a parameter
(747 by default). Both analyses, their standard errors, the overlap diagnostics and the
refuters are inside the timed section; row synthesis is not.

    python parity/performance_lanes/entry_points/aipw.py --role control
    python parity/performance_lanes/entry_points/aipw.py --role candidate --rows 100000
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _entry import arguments, digest_arrays, run_entry
from antecedent import AverageEffect, analyze

TRUE_ATE = 4.0
DATA_SEED = 42
EXECUTION_SEED = 11
ROWS = 747
BOOTSTRAP = 40
ADJUSTMENT = [f"x{i}" for i in range(1, 7)]
COVARS = ADJUSTMENT + [f"x{i}" for i in range(7, 26)]


def synthesize(rows: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(DATA_SEED)
    cont = rng.normal(size=(rows, 6))
    cont[:, 1] = 0.35 * cont[:, 0] + np.sqrt(1.0 - 0.35**2) * cont[:, 1]
    binary = (rng.uniform(size=(rows, 19)) < 0.35).astype(np.float64)
    x = np.hstack([cont, binary])
    logit = -0.2 + 0.15 * (
        0.8 * x[:, 0] + 0.6 * x[:, 1] + 0.4 * x[:, 2] - 0.3 * x[:, 3]
    )
    treatment = (rng.uniform(size=rows) < 1.0 / (1.0 + np.exp(-logit))).astype(
        np.float64
    )
    outcome = (
        TRUE_ATE * treatment
        + 1.1 * x[:, 0]
        + 0.9 * x[:, 1]
        + 0.4 * x[:, 2]
        - 0.3 * x[:, 3]
        + 0.2 * x[:, 4]
        + 0.15 * x[:, 5]
        + rng.normal(0.0, 1.0, size=rows)
    )
    data = {name: x[:, i].copy() for i, name in enumerate(COVARS)}
    data["treatment"] = treatment
    data["outcome"] = outcome
    return data


def main() -> None:
    args = arguments(__doc__ or "")
    data = synthesize(args.rows or ROWS)
    graph = [(z, "treatment") for z in ADJUSTMENT] + [
        (z, "outcome") for z in ADJUSTMENT
    ]
    graph.append(("treatment", "outcome"))
    query = AverageEffect(treatment="treatment", outcome="outcome")

    def section() -> Any:
        common = {
            "graph": graph,
            "query": query,
            "bootstrap": BOOTSTRAP,
            "seed": EXECUTION_SEED,
        }
        ipw = analyze(
            data,
            estimator="propensity.weighting",
            refute="cheap",
            threads=args.threads,
            **common,
        )
        aipw = analyze(
            data, estimator="aipw", refute="none", threads=args.threads, **common
        )
        return ipw, aipw

    def fields(result: Any) -> dict[str, Any]:
        ipw, aipw = result
        se = ipw.estimate.se_bootstrap or ipw.estimate.se_analytic
        aipw_se = aipw.estimate.se_bootstrap or aipw.estimate.se_analytic
        return {
            "ipw_effect": ipw.answer.value,
            "ipw_standard_error": se,
            "ipw_interval": (
                ipw.answer.value - 1.96 * se,
                ipw.answer.value + 1.96 * se,
            ),
            "ipw_overlap_ess": ipw.estimate.overlap_ess,
            "ipw_refuters": [
                (r.refuter, bool(r.passed)) for r in ipw.validation.reports
            ],
            "ipw_identification": str(ipw.identification.status),
            "aipw_effect": aipw.answer.value,
            "aipw_standard_error": aipw_se,
            "aipw_interval": (
                aipw.answer.value - 1.96 * aipw_se,
                aipw.answer.value + 1.96 * aipw_se,
            ),
            "calibration": str(ipw.calibration.status),
        }

    run_entry(
        "aipw",
        args,
        digest_arrays(data),
        section,
        fields,
        required=("ipw_interval", "aipw_interval", "ipw_refuters", "ipw_overlap_ess"),
    )


if __name__ == "__main__":
    main()
