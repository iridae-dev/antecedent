"""Shared prepare/estimate/inspect/export/load lifecycle for ML and transport.

Run with the release-built Python package. Intervals below are not calibrated.
"""

import antecedent as ac
import numpy as np
from antecedent import transport as tr
from antecedent.estimators import DRLearner
from antecedent.learners import Linear, Logistic, Ridge
from antecedent.prediction import FittedEffectModel


def ml_prepare():
    rng = np.random.default_rng(18)
    z = rng.normal(size=300)
    treatment = rng.binomial(1, 0.5, size=300).astype(float)
    data = {
        "z": z,
        "a": treatment,
        "y": z + treatment * (2 + z) + rng.normal(size=300) * 0.1,
    }
    study = ac.prepare(
        data,
        graph=[("z", "a"), ("z", "y"), ("a", "y")],
        query=ac.AverageEffect(treatment="a", outcome="y"),
        estimator=DRLearner(outcome=Ridge(0.1), treatment=Logistic(), final_learner=Linear()),
        bootstrap=0,
        refute=False,
        seed=9,
        threads=1,
    )
    return study


def ml_example():
    study = ml_prepare()
    result = study.estimate()
    model = FittedEffectModel.load(result.fitted_model.export())
    predictions = model.predict({"z": [-1.0, 0.0, 1.0]})
    assert predictions == result.fitted_model.predict({"z": [-1.0, 0.0, 1.0]})
    return study, result, model


def transport_example():
    data = tr.TrialAipwData(
        covariates={},
        outcome=[1.0, 3.0] * 60 + [0.0] * 80,
        treatment=[False, True] * 60 + [False] * 80,
        source=[True] * 120 + [False] * 80,
        randomization=[0.5] * 200,
        sampling="independent_samples",
    )
    query = tr.TrialAipwQuery(
        ac.Admg.from_edges(["a", "y"], [("a", "y")]),
        tr.SelectionDiagram("trial", "target", []),
        "a",
        "y",
    )
    study = tr.prepare(
        query,
        data,
        provider=tr.TrialAipw(outcome=Linear(), folds=3),
        inference=tr.TransportInference(bootstrap=9, seed=8),
    )
    result = study.estimate()
    loaded = ac.load(result.export())
    assert loaded.estimate == result.estimate
    assert abs(result.estimate - 2.0) < 1e-10
    return study, result


if __name__ == "__main__":
    _, ml, model = ml_example()
    _, trial = transport_example()
    print(model.predict({"z": [-1.0, 0.0, 1.0]}).to_dict())
    print(trial.inspect().to_dict())
