"""Native DML(auto) and DRLearner analyze paths."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.estimators import DML, DRLearner


def _linear_scm(n: int = 240, seed: int = 3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (0.6 * z + rng.normal(size=n) * 0.7 > 0).astype(float)
    y = 2.0 * t + z + rng.normal(size=n) * 0.4
    return {"t": t, "y": y, "z": z}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_dml_auto_recovers_ate():
    data, graph = _linear_scm()
    result = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        estimator=DML(learner="auto", folds=4),
        bootstrap=0,
        refute=False,
        seed=1,
    )
    assert result.estimate.estimator_id == "dml"
    assert abs(result.ate - 2.0) < 0.45
    assert result.estimate.outcome_oof_r2 is not None
    assert result.estimate.treatment_oof_logloss is not None
    assert result.estimate.crossfit_folds == 4
    assert result.estimate.crossfit_seed == 1
    assert len(result.estimate.learner_provenance) == 12
    assert all(len(item) == 3 for item in result.estimate.learner_provenance)


def test_drlearner_returns_cate():
    rng = np.random.default_rng(4)
    n = 280
    z = rng.normal(size=n)
    t = (0.4 * z + rng.normal(size=n) * 0.8 > 0).astype(float)
    y = (1.0 + z) * t + 0.3 * z + rng.normal(size=n) * 0.35
    data = {"t": t, "y": y, "z": z}
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    result = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        estimator=DRLearner(learner="ridge", folds=4),
        bootstrap=0,
        refute=False,
        seed=2,
    )
    assert result.estimate.estimator_id == "dr.learner"
    assert result.estimate.cate is not None
    assert len(result.estimate.cate) == n
    assert result.estimate.cate_se is not None
    assert len(result.estimate.cate_se) == n
    assert abs(result.ate - float(np.mean(result.estimate.cate))) < 1e-9


@pytest.mark.parametrize("estimator", [DML(), DRLearner()])
def test_staged_learner_artifact_roundtrip(estimator):
    data, graph = _linear_scm()
    prepared = antecedent.prepare(
        data,
        graph=graph,
        query=antecedent.AverageEffect("t", "y"),
        estimator=estimator,
        bootstrap=0,
        refute="none",
    )
    result = prepared.estimate(data)
    restored = antecedent.load(result.export())
    payload = restored.artifact.payload
    assert (
        tuple(tuple(p) for p in payload["learner_provenance"]) == result.estimate.learner_provenance
    )
    for field in ("outcome_oof_r2", "treatment_oof_logloss", "crossfit_folds", "crossfit_seed"):
        assert payload[field] == getattr(result.estimate, field)
    cate = payload.get("cate")
    assert (tuple(cate) if cate is not None else None) == result.estimate.cate
    cate_se = payload.get("cate_se")
    assert (tuple(cate_se) if cate_se is not None else None) == result.estimate.cate_se
    dispersion = payload.get("cate_leaf_dispersion")
    assert (
        tuple(dispersion) if dispersion is not None else None
    ) == result.estimate.cate_leaf_dispersion
    assert restored.as_point() == result.as_point()
