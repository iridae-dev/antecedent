#!/usr/bin/env python3
"""Estimate an average effect with models you supply.

Question: in a table where the outcome is ``1 + 2 * treatment`` and the treatment is
assigned independently of ``z``, what is the average effect of treatment? The fixture
answer is 2. The graph says ``z`` may affect both treatment and outcome. That
assumption is not tested here; in this table ``z`` is not a confounder, so adjustment
still recovers the structural effect.

Two declarations use the same protocol. The first pair is a least-squares outcome
model and a logistic propensity, the shape a scikit-learn estimator already has.
The second is an in-context mean: ``fit`` stores only the rows of that fit, and
``predict`` answers from them. A factory such as ``lambda: NoriRegressor(...)`` or
``lambda: KumoClassifier(...)`` is the same call. Neither vendor is imported.

Each fit builds a new model. The artifact is the fitted object, so these learners
are externally attested nuisance fits and are not portable prediction maps.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
from antecedent.external_learners import Learners
from antecedent.learners import classifier, regressor


class LeastSquares:
    """Ordinary least squares with an intercept. ``fit(x, y, sample_weight=None)``."""

    def fit(self, x, y, sample_weight=None):
        design = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
        target = np.asarray(y, float)
        if sample_weight is not None:
            scale = np.sqrt(np.asarray(sample_weight, float))
            design = design * scale[:, None]
            target = target * scale
        self.coef_ = np.linalg.lstsq(design, target, rcond=None)[0]
        return self

    def predict(self, x):
        design = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
        return design @ self.coef_


class LogisticProbability:
    """Logistic regression. ``predict_proba`` returns columns for classes 0 and 1."""

    def fit(self, x, y, sample_weight=None):
        design = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
        target = np.asarray(y, float)
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        coef = np.zeros(design.shape[1])
        for _ in range(25):
            eta = np.clip(design @ coef, -20.0, 20.0)
            probability = 1.0 / (1.0 + np.exp(-eta))
            variance = np.maximum(probability * (1.0 - probability), 1e-6)
            working = eta + (target - probability) / variance
            scale = np.sqrt(weight * variance)
            coef = np.linalg.lstsq(design * scale[:, None], working * scale, rcond=None)[0]
        self.coef_ = coef
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, x):
        design = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
        eta = np.clip(design @ self.coef_, -20.0, 20.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        return np.column_stack([1.0 - probability, probability])


class InContextMean:
    """A model that answers from the rows of this fit, not from a shared corpus."""

    def fit(self, x, y, sample_weight=None):
        self._mean = float(np.average(np.asarray(y, float), weights=sample_weight))
        return self

    def predict(self, x):
        return np.full(len(np.asarray(x)), self._mean)


class InContextRate:
    """The in-context propensity: the positive rate of this fit."""

    def fit(self, x, y, sample_weight=None):
        self.classes_ = np.array([0.0, 1.0])
        self._rate = float(np.average(np.asarray(y, float), weights=sample_weight))
        return self

    def predict_proba(self, x):
        probability = np.full(len(np.asarray(x)), self._rate)
        return np.column_stack([1.0 - probability, probability])


def _effect(models: Learners) -> float:
    n = 200
    treatment = (np.arange(n) % 2).astype(float)
    data = {
        "z": np.arange(n, dtype=float),
        "treatment": treatment,
        "outcome": 1.0 + 2.0 * treatment,
    }
    outcome = ant.analyze(
        data,
        graph=[("z", "treatment"), ("z", "outcome"), ("treatment", "outcome")],
        query=ant.AverageEffect("treatment", "outcome"),
        learners=models,
        estimator="aipw",
    )
    if outcome.refusal is not None:
        raise RuntimeError(f"{outcome.refusal.code}: {outcome.refusal.remedy}")
    return float(outcome.answer.value)


def main() -> None:
    tabular = _effect(Learners(regressor(LeastSquares), classifier(LogisticProbability)))
    in_context = _effect(Learners(regressor(InContextMean), classifier(InContextRate)))
    print(f"least squares and logistic: {tabular:.6f}")
    print(f"in-context mean: {in_context:.6f}")
    assert abs(tabular - 2.0) < 1e-6
    assert abs(in_context - 2.0) < 1e-8


if __name__ == "__main__":
    main()
