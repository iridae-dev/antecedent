#!/usr/bin/env python3
"""IHDP-style propensity / AIPW end-to-end: identify → estimate → refute.

Data provenance
---------------
This script does **not** redistribute the real Infant Health and Development
Program (IHDP) microdata. Redistributing that table is license-sensitive, and
public mirrors are flaky for CI. Instead it synthesizes an **IHDP-like** table
that matches the classic NPCI / Hill (2011) layout:

- ``n = 747`` rows (the usual IHDP analysis size),
- 25 covariates ``x1``…``x25`` (6 continuous, 19 binary),
- binary treatment ``treatment``,
- continuous outcome ``outcome``.

Covariates, treatment, and outcome are drawn from a known linear-Gaussian /
logit SCM under the DAG below. The ground-truth ATE is ``TRUE_ATE = 4.0``
(response-surface A order of magnitude, not a claim about real IHDP effects).

Assumed DAG / adjustment set (no discovery)
-------------------------------------------
We state the graph explicitly — Antecedent does not discover it:

    x1…x6 → treatment,  x1…x6 → outcome,  treatment → outcome

so the backdoor adjustment set is ``{x1,…,x6}``. Binary covariates ``x7``…``x25``
remain in the table for IHDP-like structure but are **not** edges in this DAG
(Antecedent's DAG backdoor search hits an identification budget well below 25
confounders). That is a synthetic design choice for this demo, not a claim that
binary IHDP covariates are ignorable in a real analysis.

Estimators
----------
- ``propensity.weighting`` — inverse-probability weighting with the cheap
  refute suite (overlap + E-value).
- ``aipw`` — doubly robust AIPW. Antecedent fits GLM propensity and OLS
  outcome nuisances on the **full sample** (no cross-fitting). Do **not** treat
  Chernozhukov et al. (2018) DML rate guarantees as transferring; see
  ``provenance/estimate.aipw.toml`` (Funk et al. 2011 is the uncross-fit cite).

Expected ranges (seeded; CI should not flake)
---------------------------------------------
With ``seed=11`` and ``bootstrap=40``:

- IPW ATE ∈ [3.7, 4.5], 95% Wald CI contains 4.0,
- AIPW ATE ∈ [3.7, 4.5],
- identification status ``NonparametricallyIdentified`` via ``backdoor.adjustment``,
- IPW cheap suite runs and passes (at least one refuter recorded).

Install with ``python -m pip install antecedent``; see examples/README.md.
"""

from __future__ import annotations

import numpy as np
from antecedent import AverageEffect, analyze

TRUE_ATE = 4.0
N_ROWS = 747
N_COVARS = 25
ADJUSTMENT = [f"x{i}" for i in range(1, 7)]  # continuous block only
BINARY = [f"x{i}" for i in range(7, 26)]
COVARS = ADJUSTMENT + BINARY


def make_ihdp_like(seed: int = 42) -> dict[str, np.ndarray]:
    """Synthetic IHDP-like covariates + known-ATE outcome (see module docstring)."""
    rng = np.random.default_rng(seed)
    cont = rng.normal(size=(N_ROWS, 6))
    # Mild correlation among the first two continuous covariates.
    cont[:, 1] = 0.35 * cont[:, 0] + np.sqrt(1.0 - 0.35**2) * cont[:, 1]
    binary = (rng.uniform(size=(N_ROWS, 19)) < 0.35).astype(np.float64)
    x = np.hstack([cont, binary])

    # Soft selection on the continuous confounders → good overlap for IPW.
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
    """Explicit edges for the assumed DAG (adjustment set = ADJUSTMENT)."""
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

    ipw = analyze(
        data,
        graph=graph,
        query=query,
        estimator="propensity.weighting",
        refute="cheap",
        bootstrap=40,
        seed=11,
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

    # AIPW: full-sample nuisances (no cross-fitting); see provenance/estimate.aipw.toml.
    aipw = analyze(
        data,
        graph=graph,
        query=query,
        estimator="aipw",
        refute="none",
        bootstrap=40,
        seed=11,
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
