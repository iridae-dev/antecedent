"""2.2 E2: the explicit ridge-logistic propensity of ``Aipw`` through the Python surface.

The route publishes a cross-fitted point and the score table and no interval; lasso, an ML
fallback and an interval request are closed with their registered reason codes. Truth is the
simulated effect 2; the numerical oracle for the score table, the penalty rule and the
retargeted covariance is the Rust integration test ``crates/antecedent-estimate/tests/
penalized_aipw.rs`` (an independent calculation written there), not this file.
"""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.estimators import Aipw, PropensityPenalty

from _refusal import assert_registered_refusal

QUERY = ant.AverageEffect("t", "y")
GRAPH = [("z0", "t"), ("z1", "t"), ("z0", "y"), ("z1", "y"), ("t", "y")]


def moderate(seed: int = 7, n: int = 400) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z0, z1 = rng.normal(size=n), rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.8 * z0 - 0.5 * z1)))).astype(float)
    y = 2.0 * t + z0 + 0.5 * z1 + 0.5 * rng.normal(size=n)
    return {"z0": z0, "z1": z1, "t": t, "y": y}


def separated(seed: int = 3, n: int = 300) -> dict[str, np.ndarray]:
    data = moderate(seed, n)
    noise = 0.5 * np.random.default_rng(seed + 1).normal(size=n)
    data["t"] = (data["z0"] > 0).astype(float)
    data["y"] = 2.0 * data["t"] + data["z0"] + 0.5 * data["z1"] + noise
    return data


def refusal_text(error: BaseException) -> str:
    return f"{getattr(error, 'reason_code', '')} {error}"


def test_the_dataclass_wires_the_declared_penalty_and_validates_it():
    cfg = Aipw(
        bootstrap=0,
        propensity_penalty=PropensityPenalty(lambdas=[10, 1.0], inner_folds=4),
        nuisance_fallback="ml",
    )
    assert cfg._wire() == {
        "bootstrap_replicates": 0,
        "propensity_penalty": {"kind": "ridge_logistic", "lambdas": [10.0, 1.0], "inner_folds": 4},
        "nuisance_fallback": "ml",
    }
    assert Aipw(bootstrap=0, propensity_penalty=PropensityPenalty())._wire() == {
        "bootstrap_replicates": 0,
        "propensity_penalty": {"kind": "ridge_logistic"},
    }
    assert Aipw()._wire() == {}
    for bad in (
        lambda: PropensityPenalty(kind="elastic_net"),
        lambda: PropensityPenalty(lambdas=[]),
        lambda: PropensityPenalty(lambdas=[0.0]),
        lambda: PropensityPenalty(lambdas=[-1.0]),
        lambda: PropensityPenalty(lambdas=[float("inf")]),
        lambda: PropensityPenalty(inner_folds=1),
        lambda: PropensityPenalty(inner_folds=21),
        lambda: PropensityPenalty(kind="lasso", lambdas=[1.0]),
        # No interval is published, so an interval request is refused up front.
        lambda: Aipw(propensity_penalty=PropensityPenalty()),
        lambda: Aipw(bootstrap=20, propensity_penalty=PropensityPenalty()),
        lambda: Aipw(bootstrap=0, se="hc1", propensity_penalty=PropensityPenalty()),
        lambda: Aipw(nuisance_fallback="auto"),
    ):
        with pytest.raises(CausalValueError):
            bad()


def test_a_penalized_aipw_reports_a_point_and_the_scores_but_no_interval():
    data = moderate()
    cfg = Aipw(bootstrap=0, propensity_penalty=PropensityPenalty(lambdas=[0.5, 5.0, 50.0]))
    result = ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=7)
    assert result.effect == pytest.approx(2.0, abs=0.3)
    estimate = result.estimate
    assert math.isnan(estimate.se_analytic)
    assert estimate.se_bootstrap is None
    assert estimate.joint_covariance is None and estimate.score_inference is None
    assert estimate.score_table is not None
    # Replay: the same seed gives the same point bit for bit.
    again = ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=7)
    assert again.effect == result.effect

    # The frozen scores retarget to a point; the retarget reports no standard error either.
    retargeted = result.study.retarget(np.exp(data["z0"] / 3), depends_on=["z0"])
    assert math.isfinite(retargeted.effect)
    assert math.isnan(retargeted.estimate.se_analytic)
    assert retargeted.estimate.joint_covariance is None


def test_high_dimensional_separation_is_recovered_by_the_penalty_not_by_the_plain_fit():
    data = separated()
    with pytest.raises(Exception, match="(?i)separation|converge|saturat|extreme"):
        ant.analyze(
            data, graph=GRAPH, query=QUERY, estimator=Aipw(bootstrap=0), refute=False, seed=2
        )
    cfg = Aipw(bootstrap=0, propensity_penalty=PropensityPenalty(lambdas=[1.0, 10.0, 100.0]))
    result = ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=2)
    assert result.effect == pytest.approx(2.0, abs=0.5)


def test_lasso_is_closed_with_a_typed_refusal():
    cfg = Aipw(propensity_penalty=PropensityPenalty(kind="lasso"))
    with pytest.raises(Exception) as error:
        ant.analyze(moderate(), graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=1)
    assert "selection_inference_not_licensed" in refusal_text(error.value)
    assert_registered_refusal(error.value)


def test_an_interval_request_is_refused_with_its_reason_code():
    with pytest.raises(Exception) as error:
        ant.analyze(
            moderate(),
            graph=GRAPH,
            query=QUERY,
            estimator="aipw",
            estimator_config={"propensity_penalty": {"kind": "ridge_logistic"}},
            refute=False,
            bootstrap=50,
            seed=1,
        )
    assert "penalized_interval_not_licensed" in refusal_text(error.value)
    assert_registered_refusal(error.value)


def test_a_failed_fit_under_a_declared_fallback_is_recorded_not_replaced():
    cfg = Aipw(bootstrap=0, nuisance_fallback="ml")
    with pytest.raises(Exception) as error:
        ant.analyze(separated(), graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=2)
    text = refusal_text(error.value)
    assert "nuisance_fallback_not_licensed" in text
    assert "primary nuisance fit failed" in text
    assert_registered_refusal(error.value)
    # A fit that succeeds is the plain GLM result: the fallback never silently switches.
    data = moderate()
    plain = ant.analyze(
        data, graph=GRAPH, query=QUERY, estimator=Aipw(bootstrap=0), refute=False, seed=2
    )
    armed = ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=2)
    assert armed.effect == plain.effect


@pytest.mark.parametrize(
    "penalty",
    [
        {"lambdas": [-1.0]},
        {"lambdas": []},
        {"inner_folds": 1},
        {"kind": "elastic_net"},
        {"unknown": 1},
    ],
)
def test_an_invalid_penalty_is_refused_before_any_fit(penalty):
    with pytest.raises(ValueError):
        ant.analyze(
            moderate(),
            graph=GRAPH,
            query=QUERY,
            estimator="aipw",
            estimator_config={"propensity_penalty": penalty, "bootstrap_replicates": 0},
            refute=False,
            seed=1,
        )
