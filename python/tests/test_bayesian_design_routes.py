"""Bayesian panel DiD and fuzzy local ratio through the installed ``antecedent`` package.

The routes are explicitly unlicensed and uncalibrated; the posterior is compared with a
closed-form Normal-Inverse-Gamma reference computed here, independently of the library.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalError
from antecedent.quasi import FuzzyRegressionDiscontinuity, PanelDifferenceInDifferences


def _noise(i: int) -> float:
    return ((i * 7) % 11) / 10.0 - 0.5


def _panel():
    subjects, treated, post, outcome, changes, groups = [], [], [], [], [], []
    for j in range(24):
        t = j >= 12
        before = 1.0 + 0.3 * (j % 3) + _noise(j)
        change = 0.7 + 2.0 * float(t) + _noise(j + 3)
        for after in (False, True):
            subjects.append(f"s{j}")
            treated.append(t)
            post.append(after)
            outcome.append(before + change if after else before)
        changes.append(change)
        groups.append(t)
    data = {"y": outcome, "id": subjects, "group": treated, "after": post}
    return data, np.array(changes), np.array(groups, dtype=float)


def _nig_reference(y, g, scale=100.0, a0=1e-3, b0=1e-3):
    x = np.column_stack([np.ones_like(g), g])
    precision = x.T @ x + np.eye(2) / scale**2
    inv = np.linalg.inv(precision)
    mean = inv @ x.T @ y
    an = a0 + len(y) / 2
    bn = b0 + 0.5 * (y @ y - mean @ precision @ mean)
    return mean[1], float(np.sqrt(bn / (an - 1) * inv[1, 1]))


def test_bayesian_estimator_ids_are_public():
    assert antecedent.Estimator.BAYESIAN_PANEL_DID.value == "bayesian.panel_did"
    assert antecedent.Estimator.BAYESIAN_SYNTHETIC_CONTROL.value == "bayesian.synthetic_control"
    assert antecedent.Estimator.BAYESIAN_LOCAL_RATIO.value == "bayesian.local_ratio"


def test_bayesian_panel_did_matches_the_closed_form_and_stays_unlicensed():
    data, changes, groups = _panel()
    result = antecedent.analyze(
        data,
        query=PanelDifferenceInDifferences("y", "id", "group", "after"),
        inference=antecedent.Bayesian(backend="conjugate", n_draws=20000, prior_scale=100.0),
        seed=11,
        return_posterior_artifact=True,
    )
    mean, sd = _nig_reference(changes, groups)
    assert result.estimate.estimator_id == "bayesian.panel_did"
    assert result.posterior is not None and result.posterior.n_draws == 20000
    assert result.posterior.effect_mean == pytest.approx(mean, abs=0.02)
    assert result.posterior.effect_sd == pytest.approx(sd, rel=0.03)
    assert result.posterior.q025 < mean < result.posterior.q975
    assert result.evidence_status != "licensed"
    assert result.panel_did is None
    artifact = antecedent.inference.decode_posterior_artifact(result.posterior.artifact)
    assert artifact.n_draws == 20000
    assert "att" in artifact.quantity_names


def test_bayesian_panel_did_refuses_a_foreign_backend_and_shared_clusters():
    data, _, _ = _panel()
    query = PanelDifferenceInDifferences("y", "id", "group", "after")
    with pytest.raises(CausalError, match="conjugate"):
        antecedent.analyze(data, query=query, inference=antecedent.Bayesian(backend="laplace"))
    shared = dict(data)
    shared["cluster"] = [f"c{(i // 2) % 4}" for i in range(len(data["y"]))]
    with pytest.raises(CausalError, match="Next attempt"):
        antecedent.analyze(
            shared,
            query=PanelDifferenceInDifferences("y", "id", "group", "after", cluster="cluster"),
            inference=antecedent.Bayesian(backend="conjugate", n_draws=2000),
        )


def _ratio(strong: bool):
    x, t, y = [], [], []
    for i in range(200):
        r = (i - 100) / 100.0
        dose = 0.2 + 0.6 * float(r >= 0) + 0.05 * _noise(i) if strong else 0.3 * _noise(i)
        x.append(r)
        t.append(dose)
        y.append(1.0 + 2.0 * dose + 0.3 * r + 0.05 * _noise(i + 3))
    return {"x": x, "t": t, "y": y}


def test_bayesian_local_ratio_recovers_truth_and_withholds_a_weak_interval():
    query = FuzzyRegressionDiscontinuity("y", "t", "x", 0.0, 0.8)
    bayes = antecedent.Bayesian(backend="conjugate", n_draws=8000, prior_scale=10.0)
    strong = antecedent.analyze(_ratio(True), query=query, inference=bayes, seed=5)
    assert strong.estimate.estimator_id == "bayesian.local_ratio"
    assert strong.posterior.q025 < 2.0 < strong.posterior.q975
    weak = antecedent.analyze(_ratio(False), query=query, inference=bayes, seed=5)
    assert weak.posterior.q025 is None and weak.posterior.q975 is None
    assert any("weak_first_stage" in d for d in weak.diagnostics)
