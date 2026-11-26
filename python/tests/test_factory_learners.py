"""Factory adapters over the existing nuisance-learner protocol."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.external_learners import (
    Learners,
    _learner_identity,
    _snapshot_learner,
    declare_portable_route,
)
from antecedent.learners import classifier, regressor


class MeanModel:
    def fit(self, x, y, sample_weight=None):
        self.mean_ = float(np.average(np.asarray(y, float), weights=sample_weight))
        return self

    def predict(self, x):
        return np.full(len(np.asarray(x)), self.mean_)


class PlainMean:
    def fit(self, x, y):
        self.mean_ = float(np.mean(np.asarray(y, float)))
        return self

    def predict(self, x):
        return np.full(len(np.asarray(x)), self.mean_)


class RateModel:
    def fit(self, x, y, sample_weight=None):
        self.classes_ = np.array([0.0, 1.0])
        self.p_ = float(np.average(np.asarray(y, float), weights=sample_weight))
        return self

    def predict_proba(self, x):
        p = np.full(len(np.asarray(x)), self.p_)
        return np.column_stack([1.0 - p, p])


class SwappedClasses:
    """Class 1 is the first probability column."""

    def fit(self, x, y):
        self.classes_ = np.array([1.0, 0.0])
        self.p_ = float(np.mean(np.asarray(y, float)))
        return self

    def predict_proba(self, x):
        positive = np.full(len(np.asarray(x)), self.p_)
        return np.column_stack([positive, 1.0 - positive])


class Counted:
    created = 0

    def __init__(self):
        Counted.created += 1
        self.mean_ = 0.0

    def fit(self, x, y):
        self.mean_ = float(np.mean(np.asarray(y, float)))
        return self

    def predict(self, x):
        return np.full(len(np.asarray(x)), self.mean_)


class SeededMean:
    def __init__(self):
        self.random_state = None

    def get_params(self):
        return {"random_state": self.random_state}

    def set_params(self, **params):
        self.random_state = params["random_state"]
        return self

    def fit(self, x, y):
        self.seen_seed = self.random_state
        self.mean_ = float(np.mean(np.asarray(y, float)))
        return self

    def predict(self, x):
        return np.full(len(np.asarray(x)), self.seen_seed)


def test_each_fit_builds_a_fresh_model_and_keeps_its_artifact():
    Counted.created = 0
    learner = regressor(Counted)
    assert Counted.created == 1
    first = learner.fit([[0.0], [1.0]], [1.0, 1.0])
    second = learner.fit([[0.0], [1.0]], [5.0, 5.0])
    assert Counted.created == 3
    assert first.predict([[0.0]]).estimate[0] == pytest.approx(1.0)
    assert second.predict([[0.0]]).estimate[0] == pytest.approx(5.0)


def test_weights_follow_the_fit_signature():
    weighted = regressor(MeanModel)
    assert weighted.supports_weights is True
    fitted = weighted.fit([[0.0], [1.0]], [1.0, 3.0], [1.0, 3.0])
    assert fitted.predict([[0.0]]).estimate[0] == pytest.approx(2.5)
    plain = regressor(PlainMean)
    assert plain.supports_weights is False
    with pytest.raises(CausalValueError, match="sample-weight"):
        plain.fit([[0.0], [1.0]], [1.0, 3.0], [1.0, 1.0])
    with pytest.raises(CausalValueError, match="sample_weight"):
        regressor(PlainMean, weights=True)


def test_seed_is_applied_before_fit_and_required_to_land():
    learner = regressor(SeededMean, seed=7)
    fitted = learner.fit([[0.0], [1.0]], [1.0, 1.0])
    assert fitted.predict([[0.0]]).estimate[0] == pytest.approx(7.0)
    with pytest.raises(CausalValueError, match="random_state"):
        regressor(PlainMean, seed=1)
    with pytest.raises(CausalTypeError, match="seed"):
        regressor(PlainMean, seed=True)


def test_classifier_reads_the_class_one_column():
    learner = classifier(SwappedClasses)
    fitted = learner.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 0.0, 0.0, 0.0])
    assert fitted.predict([[0.0]]).estimate[0] == pytest.approx(0.25)


def test_adapter_is_not_portable_and_configuration_changes_identity():
    learner = regressor(PlainMean, configuration={"variant": 1})
    assert learner.portable is False
    spec = learner.registry._entry(learner.name)[0].spec
    assert spec.artifact_codec == "live_python_estimator_v1"
    other = regressor(PlainMean, configuration={"variant": 2})
    assert _learner_identity(_snapshot_learner(learner)) != _learner_identity(
        _snapshot_learner(other)
    )
    models = Learners(learner, classifier(RateModel))
    for route in ("smoothed_dose", "trial_aipw", "joint_cells"):
        with pytest.raises(CausalUnsupportedError, match="live process") as refusal:
            declare_portable_route(route, models)
        assert refusal.value.reason_code == "route_not_supported"


def test_analyze_recovers_the_known_contrast():
    import antecedent as ant
    from antecedent.query import AverageEffect

    arms = [float(i % 2) for i in range(40)]
    data = {"z": [float(i) for i in range(40)], "t": arms, "y": [1.0 + 2.0 * arm for arm in arms]}
    models = Learners(regressor(MeanModel), classifier(RateModel))
    outcome = ant.analyze(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        learners=models,
        estimator="aipw",
    )
    assert outcome.refusal is None
    assert outcome.answer.value == pytest.approx(2.0)
