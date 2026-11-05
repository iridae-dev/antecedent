"""Typed learner settings and independently consumed prediction models."""

import antecedent as ac
import numpy as np
import pytest
from antecedent.errors import CausalEstimateError, CausalSerializationError, CausalTypeError
from antecedent.estimators import DML, CausalForest, DRLearner
from antecedent.learners import ElasticNet, GradientBoostedTrees, Linear, Logistic, NeuralNet, Ridge
from antecedent.outcomes import require_success
from antecedent.prediction import FittedEffectModel


def data_fixture():
    rng = np.random.default_rng(19)
    z = rng.normal(size=160)
    t = rng.binomial(1, 0.5, size=160).astype(float)
    y = t * (2 + z) + z + rng.normal(size=160) * 0.1
    return {"z": z, "t": t, "y": y}


def analyze(estimator):
    return require_success(
        ac.analyze(
            data_fixture(),
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=ac.AverageEffect(treatment="t", outcome="y"),
            estimator=estimator,
            inference=ac.Frequentist(bootstrap=0),
            refute=False,
            execution=ac.Execution(seed=3),
        )
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

    assert loads(bytes(a.export())).contract != loads(bytes(b.export())).contract


def test_fitted_model_requires_explicit_matching_source_for_replay():
    result = analyze(DRLearner(final_learner=Linear()))
    with pytest.raises(CausalSerializationError, match="parent claim that replays"):
        _ = result.fitted_model
    source = data_fixture()
    model = result.fitted_model_from_source(source)
    assert len(require_success(model.predict({"z": [0.0]})).values) == 1
    assert FittedEffectModel.load(bytes(model.export())).parent_claim == model.parent_claim
    source["y"][0] += 1.0
    with pytest.raises(Exception, match="digest|snapshot"):
        result.fitted_model_from_source(source)


def test_fitted_effect_predicts_after_reload():
    estimator = DRLearner(outcome=Ridge(), treatment=Logistic(), final_learner=Linear())
    result = analyze(estimator)
    model = result.fitted_model_from_source(data_fixture())
    x = {"z": np.array([-1.0, 0.0, 1.0])}
    expected = require_success(model.predict(x))
    loaded = FittedEffectModel.load(bytes(model.export()))
    assert require_success(loaded.predict(x)) == expected
    assert len(expected.values) == 3
    assert expected.parent_claim == model.parent_claim
    assert expected.to_dict()["uncertainty"]["status"] == "unavailable"
    with pytest.raises(ValueError, match="missing prediction features"):
        require_success(loaded.predict({"wrong": [1.0]}))
    with pytest.raises(ValueError, match="must be finite one-dimensional columns"):
        require_success(loaded.predict({"z": [float("nan")]}))


@pytest.mark.parametrize(
    "estimator",
    [
        DRLearner(final_learner=GradientBoostedTrees(trees=12, depth=2)),
        CausalForest(n_trees=20, min_leaf=4),
    ],
)
def test_fitted_effect_refuses_reload_without_replay(estimator):
    # Neither boosted-tree DR nor causal forest has a checked replay declaration;
    # producing a fitted model alone does not establish a parent claim that replays.
    result = analyze(estimator)
    with pytest.raises(Exception, match="parent claim that replays"):
        result.fitted_model_from_source(data_fixture())


def test_fitted_effect_prediction_warns_on_digit_strings():
    # The strict ingest refuses digit strings; prediction still parses them for now
    # but says so, naming the column.
    model = analyze(DRLearner(final_learner=Linear())).fitted_model_from_source(data_fixture())
    with pytest.warns(FutureWarning, match="column 'z'"):
        parsed = require_success(model.predict({"z": np.array(["1", "2"])}))
    assert parsed == require_success(model.predict({"z": np.array([1.0, 2.0])}))
    with pytest.raises(CausalTypeError, match="'z'"):
        require_success(model.predict({"z": np.array(["one", "two"])}))


def test_model_payload_mutation_cannot_reuse_parent_claim():
    from antecedent.artifacts import dumps, loads

    result = analyze(DRLearner(final_learner=Linear()))
    artifact = loads(bytes(result.export()))
    payload = artifact.to_dict()["payload"]
    payload["fitted_effect"]["predictor"]["model"]["coefficients"][0] += 1
    forged = dumps(
        "analysis_result", payload, variable_names=artifact.variable_names, artifact_id="forged"
    )
    with pytest.raises(Exception, match="parent claim that replays"):
        FittedEffectModel.load(forged)


@pytest.mark.parametrize(
    "estimator",
    [DML(treatment=Ridge(9.0)), DRLearner(outcome=Logistic()), DRLearner(final_learner=Logistic())],
)
def test_explicit_learner_roles_refuse_task_mismatch(estimator):
    with pytest.raises(CausalEstimateError, match="task mismatch"):
        analyze(estimator)


def test_shared_learner_convenience_adapts_at_configuration():
    # The effect of t is 2 + z for each unit, so the sample ATE is 2 + mean(z) (1.989 here); the
    # shared ridge learner recovers it within 0.05 (observed deviation 0.008).
    sample_ate = 2.0 + float(data_fixture()["z"].mean())
    assert analyze(DML(learner=Ridge(2.0))).effect == pytest.approx(sample_ate, abs=0.05)


def test_neural_nuisance_executes_in_standard_python_wheel():
    result = analyze(
        DML(
            outcome=NeuralNet(hidden=4, epochs=3, learning_rate=0.01),
            treatment=Logistic(),
            folds=2,
        )
    )
    # Three epochs of a four-unit network is not a calibrated fit: only a finite executed answer is claimed.
    assert np.isfinite(result.effect)
