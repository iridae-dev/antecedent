"""2.2 E2: the penalized propensities of ``Aipw`` through the Python surface.

Ridge and lasso propensities publish a cross-fitted point, the score table, the influence
function SE and, with ``bootstrap > 0``, a refit bootstrap SE whose replicates repeat penalty
selection; a declared GLM-to-penalized fallback runs and records the failed fit; a machine
learning fallback and a lasso outside the cross-fit are refused with their registered reason
codes. Truth is the simulated effect 2; the numerical oracle for the score table, the
penalty rule and the retargeted covariance is the Rust integration test
``crates/antecedent-estimate/tests/penalized_aipw.rs`` (an independent calculation written
there), not this file.
"""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalValueError
from antecedent.estimators import Aipw, Overlap, PropensityPenalty

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
    cfg = Aipw(bootstrap=0, propensity_penalty=PropensityPenalty(lambdas=[10, 1.0], inner_folds=4))
    assert cfg._wire() == {
        "bootstrap_replicates": 0,
        "propensity_penalty": {"kind": "ridge_logistic", "lambdas": [10.0, 1.0], "inner_folds": 4},
    }
    assert Aipw(bootstrap=0, propensity_penalty=PropensityPenalty())._wire() == {
        "bootstrap_replicates": 0,
        "propensity_penalty": {"kind": "ridge_logistic"},
    }
    # A bootstrap and a dependence-robust SE are accepted with a penalty: the interval routes.
    assert (
        Aipw(bootstrap=40, propensity_penalty=PropensityPenalty())._wire()["bootstrap_replicates"]
        == 40
    )
    assert Aipw(propensity_penalty=PropensityPenalty(kind="lasso", lambdas=[2.0, 8.0]))._wire() == {
        "propensity_penalty": {"kind": "lasso", "lambdas": [2.0, 8.0]}
    }
    # Fallback destinations: a name (default tuning) or a penalty declaration.
    assert Aipw(nuisance_fallback="ridge_logistic")._wire() == {
        "nuisance_fallback": "ridge_logistic"
    }
    declared = Aipw(nuisance_fallback=PropensityPenalty(kind="lasso", inner_folds=3))
    assert declared._wire() == {"nuisance_fallback": {"kind": "lasso", "inner_folds": 3}}
    assert Aipw(nuisance_fallback="ml")._wire() == {"nuisance_fallback": "ml"}
    assert Aipw()._wire() == {}
    for bad in (
        lambda: PropensityPenalty(kind="elastic_net"),
        lambda: PropensityPenalty(lambdas=[]),
        lambda: PropensityPenalty(lambdas=[0.0]),
        lambda: PropensityPenalty(lambdas=[-1.0]),
        lambda: PropensityPenalty(lambdas=[float("inf")]),
        lambda: PropensityPenalty(inner_folds=1),
        lambda: PropensityPenalty(inner_folds=21),
        lambda: PropensityPenalty(kind="lasso", lambdas=[]),
        lambda: Aipw(nuisance_fallback="auto"),
        # A fallback replaces a failed GLM fit; beside a penalized primary it could never run.
        lambda: Aipw(propensity_penalty=PropensityPenalty(), nuisance_fallback="ridge_logistic"),
    ):
        with pytest.raises(CausalValueError):
            bad()


def test_a_penalized_aipw_reports_a_point_the_scores_and_the_influence_interval():
    data = moderate()
    cfg = Aipw(bootstrap=0, propensity_penalty=PropensityPenalty(lambdas=[0.5, 5.0, 50.0]))
    result = ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=7)
    assert result.effect == pytest.approx(2.0, abs=0.3)
    estimate = result.estimate
    # The cross-fitted influence-function SE and the iid joint covariance are published.
    assert math.isfinite(estimate.se_analytic) and estimate.se_analytic > 0.0
    assert estimate.se_bootstrap is None
    assert estimate.joint_covariance is not None and estimate.score_inference is not None
    assert estimate.score_table is not None
    assert estimate.crossfit_folds == 5 and len(estimate.learner_provenance) == 5
    assert estimate.penalized_fallback is None and estimate.penalized_bootstrap is None
    # Replay: the same seed gives the same point bit for bit.
    again = ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=7)
    assert again.effect == result.effect
    assert again.estimate.se_analytic == estimate.se_analytic

    # The frozen scores retarget with their covariance, as for the unpenalized route.
    retargeted = result.study.retarget(np.exp(data["z0"] / 3), depends_on=["z0"])
    assert math.isfinite(retargeted.effect)
    assert math.isfinite(retargeted.estimate.se_analytic)
    assert retargeted.estimate.joint_covariance is not None


def test_the_refit_bootstrap_publishes_its_se_and_the_penalties_of_every_replicate():
    grid = [0.5, 5.0, 50.0]
    cfg = Aipw(
        bootstrap=16,
        propensity_penalty=PropensityPenalty(lambdas=grid, inner_folds=3),
    )
    data = moderate(n=200)
    result = ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=3)
    estimate = result.estimate
    report = estimate.penalized_bootstrap
    assert report is not None
    assert estimate.se_bootstrap is not None and estimate.se_bootstrap > 0.0
    assert report.se == estimate.se_bootstrap
    assert report.replicates_requested == 16
    assert report.replicates_ok + report.replicates_failed == 16
    assert len(report.replicate_penalties) == report.replicates_ok
    for replicate in report.replicate_penalties:
        assert len(replicate.lambdas) == 5 and all(v in grid for v in replicate.lambdas)
    assert report.influence_se == pytest.approx(estimate.se_analytic, rel=1e-9)
    assert "refit_bootstrap" in report.uncertainty_kind
    again = ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=3)
    assert again.estimate.se_bootstrap == estimate.se_bootstrap
    assert again.estimate.penalized_bootstrap == report


def test_high_dimensional_separation_is_recovered_by_the_penalty_not_by_the_plain_fit():
    data = separated()
    with pytest.raises(Exception, match="(?i)separation|converge|saturat|extreme"):
        ant.analyze(
            data, graph=GRAPH, query=QUERY, estimator=Aipw(bootstrap=0), refute=False, seed=2
        )
    for penalty in (
        PropensityPenalty(lambdas=[1.0, 10.0, 100.0]),
        PropensityPenalty(kind="lasso", lambdas=[5.0, 20.0, 80.0]),
    ):
        cfg = Aipw(bootstrap=0, propensity_penalty=penalty)
        result = ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=2)
        assert result.effect == pytest.approx(2.0, abs=0.5)


def test_a_lasso_propensity_records_its_selected_support_per_fold():
    cfg = Aipw(
        bootstrap=0,
        propensity_penalty=PropensityPenalty(kind="lasso", lambdas=[1.0, 5.0, 20.0]),
    )
    result = ant.analyze(
        moderate(n=500), graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=5
    )
    assert result.effect == pytest.approx(2.0, abs=0.35)
    support = result.estimate.penalized_support
    assert [fold.fold for fold in support] == [0, 1, 2, 3, 4]
    # z0 is the first data column (variable id 0), the strongest propensity covariate.
    assert all("V0" in fold.names for fold in support)
    assert all(set(fold.names) <= {"V0", "V1"} for fold in support)
    assert all(
        spec.startswith("lasso_logistic:") for spec, _, _ in result.estimate.learner_provenance
    )


def test_a_lasso_outside_the_cross_fit_is_refused_with_a_typed_refusal():
    cfg = Aipw(
        bootstrap=0,
        propensity_penalty=PropensityPenalty(kind="lasso"),
        overlap=Overlap(trim=0.05),
    )
    with pytest.raises(CausalError) as error:
        ant.analyze(moderate(), graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=1)
    assert "selection_inference_not_licensed" in refusal_text(error.value)
    assert_registered_refusal(error.value)


def test_a_ridge_penalty_outside_the_cross_fit_is_route_not_supported():
    cfg = Aipw(bootstrap=0, propensity_penalty=PropensityPenalty(), overlap=Overlap(trim=0.05))
    with pytest.raises(CausalError) as error:
        ant.analyze(moderate(), graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=1)
    assert "route_not_supported" in refusal_text(error.value)
    assert_registered_refusal(error.value)


def test_a_failed_glm_fit_runs_the_declared_fallback_and_records_both():
    data = separated()
    with pytest.raises(Exception, match="(?i)separation|converge|saturat|extreme"):
        ant.analyze(
            data, graph=GRAPH, query=QUERY, estimator=Aipw(bootstrap=0), refute=False, seed=2
        )
    destination = PropensityPenalty(lambdas=[1.0, 10.0, 100.0], inner_folds=3)
    cfg = Aipw(bootstrap=0, nuisance_fallback=destination)
    result = ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=2)
    assert result.effect == pytest.approx(2.0, abs=0.5)
    fallback = result.estimate.penalized_fallback
    assert fallback is not None
    assert fallback.stage == "propensity_fit"
    assert fallback.reason in {"separated", "non_converged", "boundary_saturated"}
    assert fallback.message
    assert fallback.destination.startswith("ridge_logistic.cv(")
    # The result is the destination's: the same point as declaring that route directly.
    direct = ant.analyze(
        data,
        graph=GRAPH,
        query=QUERY,
        estimator=Aipw(bootstrap=0, propensity_penalty=destination),
        refute=False,
        seed=2,
    )
    assert result.effect == direct.effect
    # A healthy fit under the same declaration is the plain GLM result and records nothing.
    healthy = moderate()
    plain = ant.analyze(
        healthy, graph=GRAPH, query=QUERY, estimator=Aipw(bootstrap=0), refute=False, seed=2
    )
    armed = ant.analyze(healthy, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=2)
    assert armed.effect == plain.effect
    assert armed.estimate.penalized_fallback is None


def test_a_machine_learning_fallback_is_refused_with_the_failed_fit_recorded():
    cfg = Aipw(bootstrap=0, nuisance_fallback="ml")
    with pytest.raises(CausalError) as error:
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
        {"kind": "lasso", "lambdas": [0.0]},
        {"unknown": 1},
    ],
)
def test_an_invalid_penalty_is_refused_before_any_fit(penalty):
    with pytest.raises(ValueError, match="penalty|propensity_penalty"):
        ant.analyze(
            moderate(),
            graph=GRAPH,
            query=QUERY,
            estimator="aipw",
            estimator_config={"propensity_penalty": penalty, "bootstrap_replicates": 0},
            refute=False,
            seed=1,
        )
