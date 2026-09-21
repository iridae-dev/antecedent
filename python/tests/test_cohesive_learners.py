"""Typed learner settings and independently consumed prediction models."""

import antecedent as ac
import numpy as np
import pytest
from antecedent.errors import CausalEstimateError
from antecedent.estimators import DML, CausalForest, DRLearner
from antecedent.learners import ElasticNet, GradientBoostedTrees, Linear, Logistic, NeuralNet, Ridge
from antecedent.prediction import FittedEffectModel


def data_fixture():
    rng = np.random.default_rng(19)
    z = rng.normal(size=160)
    t = rng.binomial(1, 0.5, size=160).astype(float)
    y = t * (2 + z) + z + rng.normal(size=160) * 0.1
    return {"z": z, "t": t, "y": y}


def analyze(estimator):
    return ac.analyze(
        data_fixture(),
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ac.AverageEffect(treatment="t", outcome="y"),
        estimator=estimator,
        bootstrap=0,
        refute=False,
        seed=3,
    )


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: Ridge(float("nan")), "penalty must be finite and nonnegative"),
        (lambda: ElasticNet(l1_ratio=2), "l1_ratio must be finite and between zero and one"),
        (lambda: GradientBoostedTrees(trees=0), "trees must be a positive 32-bit integer"),
        (lambda: NeuralNet(epochs=True), "epochs must be a positive 32-bit integer"),
    ],
)
def test_invalid_learner_settings_refuse(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_structured_learner_parameters_change_identity():
    a = analyze(DML(outcome=Ridge(0.1), treatment=Logistic()))
    b = analyze(DML(outcome=Ridge(20.0), treatment=Logistic()))
    from antecedent.artifacts import loads

    assert loads(a.export()).contract != loads(b.export()).contract


@pytest.mark.parametrize(
    "estimator",
    [
        DRLearner(outcome=Ridge(), treatment=Logistic(), final_learner=Linear()),
        DRLearner(final_learner=GradientBoostedTrees(trees=12, depth=2)),
        CausalForest(n_trees=20, min_leaf=4),
    ],
)
def test_fitted_effect_predicts_after_verified_reload(estimator):
    result = analyze(estimator)
    model = result.fitted_model
    x = {"z": np.array([-1.0, 0.0, 1.0])}
    expected = model.predict(x)
    loaded = FittedEffectModel.load(model.export())
    assert loaded.predict(x) == expected
    assert len(expected.values) == 3
    assert expected.parent_claim == model.parent_claim
    assert expected.to_dict()["uncertainty"]["status"] == "unavailable"
    with pytest.raises(ValueError, match="missing prediction features"):
        loaded.predict({"wrong": [1.0]})
    with pytest.raises(ValueError, match="must be finite one-dimensional columns"):
        loaded.predict({"z": [float("nan")]})


def test_model_payload_mutation_cannot_reuse_parent_claim():
    from antecedent.artifacts import dumps, loads

    result = analyze(DRLearner(final_learner=Linear()))
    artifact = loads(result.export())
    payload = dict(artifact.payload)
    payload["fitted_effect"]["predictor"]["model"]["coefficients"][0] += 1
    forged = dumps(
        "analysis_result", payload, variable_names=artifact.variable_names, artifact_id="forged"
    )
    with pytest.raises(Exception, match="verified parent claim"):
        FittedEffectModel.load(forged)


@pytest.mark.parametrize(
    "estimator",
    [DML(treatment=Ridge(9.0)), DRLearner(outcome=Logistic()), DRLearner(final_learner=Logistic())],
)
def test_explicit_learner_roles_refuse_task_mismatch(estimator):
    with pytest.raises(CausalEstimateError, match="task mismatch"):
        analyze(estimator)


def test_shared_learner_convenience_adapts_at_configuration():
    assert analyze(DML(learner=Ridge(2.0))).effect is not None
