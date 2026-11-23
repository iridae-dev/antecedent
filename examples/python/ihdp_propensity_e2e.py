#!/usr/bin/env python3
"""Do propensity weighting and AIPW recover a known effect on a confounded benchmark table?

Data: a synthetic table in the layout of the Infant Health and Development Program (IHDP)
benchmark of Hill (2011): 747 rows, 25 covariates (6 continuous, 19 binary), a binary
``treatment`` and a continuous ``outcome``. The real IHDP microdata are not used or
redistributed. Rows are drawn from a known logit-treatment, linear-Gaussian-outcome mechanism
with true average effect ``TRUE_ATE = 4.0``; that value is a fixture truth, not a claim about
the real IHDP study.

Graph: declared, not discovered. ``x1..x6 -> treatment``, ``x1..x6 -> outcome`` and
``treatment -> outcome``, so the backdoor adjustment set is ``{x1, ..., x6}``. The binary
covariates ``x7..x25`` are in the table and outside the graph; that is a choice of this fixture,
not a claim that binary IHDP covariates are ignorable in a real analysis.

The script asserts, with ``seed=11`` and ``bootstrap=40``: identification is
``NonparametricallyIdentified`` by ``backdoor.adjustment`` with that set; the
``propensity.weighting`` and ``aipw`` estimates both lie in [3.7, 4.5]; the weighting estimate's
95% Wald interval contains 4.0; and its ``"cheap"`` refutation suite runs and passes. For this
untrimmed average effect ``aipw`` cross-fits its logistic propensity and least-squares outcome
regressions; it is derived from the doubly robust score and is not the exact estimator analyzed
by Chernozhukov et al. (2018), see ``provenance/estimate.aipw.toml``. One seeded table shows
agreement with a known truth under a correct graph, not interval coverage.

Run: ``python examples/python/ihdp_propensity_e2e.py``.
"""

from __future__ import annotations

import antecedent as _ant_api
import numpy as np
from antecedent import AverageEffect, analyze
from antecedent.outcomes import require_success

TRUE_ATE = 4.0
N_ROWS = 747
N_COVARS = 25
ADJUSTMENT = [f"x{i}" for i in range(1, 7)]  # continuous block only
BINARY = [f"x{i}" for i in range(7, 26)]
COVARS = ADJUSTMENT + BINARY


def make_ihdp_like(seed: int = 42) -> dict[str, np.ndarray]:
    """Simulate the IHDP-like table with a known average effect (see the module docstring)."""
    rng = np.random.default_rng(seed)
    cont = rng.normal(size=(N_ROWS, 6))
    # Correlate the first two confounders (0.35) so adjustment has to separate them.
    cont[:, 1] = 0.35 * cont[:, 0] + np.sqrt(1.0 - 0.35**2) * cont[:, 1]
    binary = (rng.uniform(size=(N_ROWS, 19)) < 0.35).astype(np.float64)
    x = np.hstack([cont, binary])

    # Treatment depends weakly on four confounders, so propensities stay away from 0 and 1 and
    # inverse-probability weights are stable (good overlap).
    logit = -0.2 + 0.15 * (0.8 * x[:, 0] + 0.6 * x[:, 1] + 0.4 * x[:, 2] - 0.3 * x[:, 3])
    propensity = 1.0 / (1.0 + np.exp(-logit))
    treatment = (rng.uniform(size=N_ROWS) < propensity).astype(np.float64)
    outcome = (
        TRUE_ATE * treatment
        + 1.1 * x[:, 0]
        + 0.9 * x[:, 1]
        + 0.4 * x[:, 2]
        - 0.3 * x[:, 3]
        + 0.2 * x[:, 4]
        + 0.15 * x[:, 5]
        + rng.normal(0.0, 1.0, size=N_ROWS)
    )

    data = {name: x[:, i].copy() for i, name in enumerate(COVARS)}
    data["treatment"] = treatment
    data["outcome"] = outcome
    return data


def assumed_dag_edges() -> list[tuple[str, str]]:
    """Edges of the assumed DAG: each confounder causes treatment and outcome."""
    return (
        [(z, "treatment") for z in ADJUSTMENT]
        + [(z, "outcome") for z in ADJUSTMENT]
        + [("treatment", "outcome")]
    )


def _wald_ci(estimate: float, se: float, z: float = 1.96) -> tuple[float, float]:
    return estimate - z * se, estimate + z * se


def main() -> None:
    data = make_ihdp_like(seed=42)
    assert set(data) >= set(COVARS) | {"treatment", "outcome"}
    assert data["treatment"].shape == (N_ROWS,)
    graph = assumed_dag_edges()
    query = AverageEffect(treatment="treatment", outcome="outcome")

    print("Assumed DAG edges:", graph)
    print("Assumed adjustment set:", ADJUSTMENT)
    print(f"IHDP-like table: n={N_ROWS}, covariates={N_COVARS} (binary x7–x25 not in DAG)")

    ipw = require_success(
        analyze(
            data,
            graph=graph,
            query=query,
            estimator="propensity.weighting",
            refute="cheap",
            inference=_ant_api.Frequentist(bootstrap=40),
            execution=_ant_api.Execution(seed=11),
        )
    )
    assert ipw.answer.kind == "point"
    assert ipw.identification.status == "NonparametricallyIdentified"
    assert ipw.identification.method == "backdoor.adjustment"
    assert list(ipw.identification.adjustment_set) == ADJUSTMENT
    assert ipw.estimate.estimator_id == "propensity.weighting"
    assert ipw.validation.ran and ipw.validation.count >= 1
    assert ipw.validation.passed, [
        (r.refuter, r.passed, r.failure_condition) for r in ipw.validation.reports
    ]

    se = ipw.estimate.se_bootstrap
    if se is None:
        se = ipw.estimate.se_analytic
    assert se is not None and se > 0.0
    lo, hi = _wald_ci(ipw.answer.value, se)
    assert 3.7 <= ipw.answer.value <= 4.5, ipw.answer.value
    assert lo <= TRUE_ATE <= hi, (lo, hi)

    print("Identification:", ipw.identification)
    print(
        f"IPW ATE={ipw.answer.value:.4f} "
        f"95% CI=[{lo:.4f}, {hi:.4f}] "
        f"estimator={ipw.estimate.estimator_id} "
        f"overlap_ess={ipw.estimate.overlap_ess}"
    )
    print("IPW refuters:")
    for report in ipw.validation.reports:
        print(
            f"  {report.refuter}: passed={report.passed} "
            f"comparison={report.comparison:.4g} replicates={report.replicates}"
        )

    # AIPW combines the propensity model with outcome regressions, so it stays consistent if
    # either is right. Its nuisance-fitting scheme is described in provenance/estimate.aipw.toml.
    aipw = require_success(
        analyze(
            data,
            graph=graph,
            query=query,
            estimator="aipw",
            refute="none",
            inference=_ant_api.Frequentist(bootstrap=40),
            execution=_ant_api.Execution(seed=11),
        )
    )
    assert aipw.answer.kind == "point"
    assert aipw.identification.status == "NonparametricallyIdentified"
    assert aipw.estimate.estimator_id == "aipw"
    assert 3.7 <= aipw.answer.value <= 4.5, aipw.answer.value
    aipw_se = aipw.estimate.se_bootstrap or aipw.estimate.se_analytic
    assert aipw_se is not None
    a_lo, a_hi = _wald_ci(aipw.answer.value, aipw_se)
    print(
        f"AIPW ATE={aipw.answer.value:.4f} "
        f"95% CI=[{a_lo:.4f}, {a_hi:.4f}] "
        f"(full-sample nuisances; not cross-fit DML)"
    )
    print("Calibration:", ipw.calibration.status)


if __name__ == "__main__":
    main()
