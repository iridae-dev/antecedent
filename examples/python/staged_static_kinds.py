#!/usr/bin/env python3
"""How much of a treatment's effect runs directly, and how much through a mediator?

The data are 500 deterministic rows of a linear model: the baseline variable z affects treatment
a, mediator m (m = 2a + 0.5z + ...) and outcome y (y = 3a + 4m + 5z + ...), so z must be adjusted
for. For a change in a from 0.2 to 0.8 the true natural direct effect is 3 * 0.6 = 1.8, the
natural indirect effect through m is 4 * 2 * 0.6 = 4.8, and the total is 6.6, which in this
linear model also equals the mean individual counterfactual effect.

The script asserts each frequentist estimate within 0.03 of its truth with the expected estimator
(`mediation.linear`, `gcm.fit`), and Bayesian versions within 0.2 (direct) and 0.25 (individual).
The DAG is declared, not tested; with no unexplained noise these are fixture checks of the
estimators, not evidence about sampling error.

Run: `python examples/python/staged_static_kinds.py`."""

from __future__ import annotations

import antecedent as _ant_api
import numpy as np
from antecedent import Bayesian, Counterfactual, MediationEffect, analyze, prepare
from antecedent.outcomes import require_success


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
            inference=_ant_api.Frequentist(bootstrap=0),
        )
        result = require_success(_ant_api.estimate(study))
        print("Calibration:", result.calibration.status)
        print(f"{contrast}={result.effect:.4f} estimator={result.estimate.estimator_id}")
        assert abs(result.effect - expected) < 0.03, result.effect
        assert result.estimate.estimator_id == "mediation.linear"

    ite = require_success(
        analyze(
            data,
            graph=graph,
            query=Counterfactual("a", "y", control_level=control, active_level=active),
            refute="none",
        )
    )
    print(
        f"mean_ite={ite.mean_ite:.4f} n={len(ite.unit_effects)} "
        f"estimator={ite.estimate.estimator_id}"
    )
    assert abs(ite.mean_ite - 6.6) < 0.03, ite.mean_ite
    assert ite.estimate.estimator_id == "gcm.fit"
    assert len(ite.unit_effects) == len(data["a"])

    bayes = require_success(
        analyze(
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
        )
    )
    print(f"bayesian_nde={bayes.effect:.4f} estimator={bayes.estimate.estimator_id}")
    assert abs(bayes.effect - 1.8) < 0.2, bayes.effect
    assert bayes.posterior is not None

    bayes_cf = require_success(
        analyze(
            data,
            graph=graph,
            query=Counterfactual("a", "y", control_level=control, active_level=active),
            inference=Bayesian(n_draws=64),
            refute="none",
        )
    )
    print(f"bayesian_ite={bayes_cf.mean_ite:.4f} estimator={bayes_cf.estimate.estimator_id}")
    assert abs(bayes_cf.mean_ite - 6.6) < 0.25, bayes_cf.mean_ite
    assert bayes_cf.posterior is not None


if __name__ == "__main__":
    main()
