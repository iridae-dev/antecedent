#!/usr/bin/env python3
"""Separate a treatment effect into direct and mediated parts.

The baseline variable Z affects treatment A, mediator M, and outcome Y.
For a change from 0.2 to 0.8, the simulated direct effect is 1.8 and the
indirect effect through M is 4.8. Their total is 6.6, which also equals the
mean individual effect in this simulation. Omitting Z would bias the analysis.

Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import numpy as np
from antecedent import Bayesian, Counterfactual, MediationEffect, analyze, prepare


def confounded_scm(n: int = 500) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    i = np.arange(n, dtype=float)
    z = np.cos(i * 0.41)
    a = np.sin(i * 0.71) + 0.4 * z
    m = 2 * a + 0.5 * z + np.cos(i * 1.13)
    y = 3 * a + 4 * m + 5 * z + 0.1 * np.sin(i * 0.31)
    graph = [
        ("a", "m"),
        ("a", "y"),
        ("m", "y"),
        ("z", "a"),
        ("z", "m"),
        ("z", "y"),
    ]
    return {"a": a, "m": m, "y": y, "z": z}, graph


def main() -> None:
    data, graph = confounded_scm()
    control, active = 0.2, 0.8
    for contrast, expected in (
        ("natural_direct", 1.8),
        ("natural_indirect", 4.8),
        ("total", 6.6),
    ):
        study = prepare(
            data,
            graph=graph,
            query=MediationEffect(
                "a",
                "y",
                mediators=["m"],
                contrast=contrast,
                control_level=control,
                active_level=active,
            ),
            refute="none",
            bootstrap=0,
        )
        result = study.estimate()
        print("Calibration:", result.calibration.status)
        print(
            f"{contrast}={result.effect:.4f} estimator={result.estimate.estimator_id}"
        )
        assert abs(result.effect - expected) < 0.03, result.effect
        assert result.estimate.estimator_id == "mediation.linear"

    ite = analyze(
        data,
        graph=graph,
        query=Counterfactual("a", "y", control_level=control, active_level=active),
        refute="none",
    )
    print(
        f"mean_ite={ite.mean_ite:.4f} n={len(ite.unit_effects)} "
        f"estimator={ite.estimate.estimator_id}"
    )
    assert abs(ite.mean_ite - 6.6) < 0.03, ite.mean_ite
    assert ite.estimate.estimator_id == "gcm.fit"
    assert len(ite.unit_effects) == len(data["a"])

    bayes = analyze(
        data,
        graph=graph,
        query=MediationEffect(
            "a",
            "y",
            mediators=["m"],
            contrast="natural_direct",
            control_level=control,
            active_level=active,
        ),
        inference=Bayesian(n_draws=64),
        refute="none",
        bootstrap=0,
    )
    print(f"bayesian_nde={bayes.effect:.4f} estimator={bayes.estimate.estimator_id}")
    assert abs(bayes.effect - 1.8) < 0.2, bayes.effect
    assert bayes.posterior is not None

    bayes_cf = analyze(
        data,
        graph=graph,
        query=Counterfactual("a", "y", control_level=control, active_level=active),
        inference=Bayesian(n_draws=64),
        refute="none",
    )
    print(
        f"bayesian_ite={bayes_cf.mean_ite:.4f} estimator={bayes_cf.estimate.estimator_id}"
    )
    assert abs(bayes_cf.mean_ite - 6.6) < 0.25, bayes_cf.mean_ite
    assert bayes_cf.posterior is not None


if __name__ == "__main__":
    main()
