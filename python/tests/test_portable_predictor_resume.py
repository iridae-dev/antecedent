"""Fresh-process prediction uses portable numerical state, not a live prepared study."""

import json
import subprocess
import sys

import antecedent as ac
import numpy as np
from antecedent.estimators import DRLearner
from antecedent.prediction import FittedEffectModel


def fitted_result(effect=2.0):
    rng = np.random.default_rng(19)
    z = np.tile(np.linspace(-1.0, 1.0, 21), 80)
    treatment = rng.binomial(1, 0.5, len(z)).astype(float)
    return ac.analyze(
        {"z": z, "t": treatment, "y": treatment * (effect + 0.4 * z) + z},
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ac.AverageEffect(treatment="t", outcome="y"),
        estimator=DRLearner(),
        bootstrap=0,
        refute=False,
        seed=3,
    )


def test_portable_predictor_fresh_process_matches_full_rerun_and_scm_truth(tmp_path):
    result = fitted_result()
    model = result.fitted_model
    features = {"z": [-1.0, 0.0, 1.0]}
    expected = model.predict(features)
    assert np.allclose(expected.values, [1.6, 2.0, 2.4], atol=0.12, rtol=0)
    assert fitted_result().fitted_model.predict(features) == expected
    artifact = tmp_path / "predictor.ant"
    artifact.write_bytes(model.export())
    code = """
import json, sys
from pathlib import Path
import antecedent as ac
from antecedent.prediction import FittedEffectModel

def forbidden(*args, **kwargs):
    raise AssertionError('resume cannot analyze or refit without data')
ac.analyze = forbidden
model = FittedEffectModel.load(Path(sys.argv[1]).read_bytes())
print(json.dumps(model.predict({'z': [-1., 0., 1.]}).to_dict()))
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(child.stdout) == expected.to_dict()
    assert expected.to_dict()["uncertainty"]["status"] == "unavailable"


def test_predictor_new_rows_preserve_parent_but_new_outcomes_need_a_new_fit():
    model = FittedEffectModel.load(fitted_result().export())
    changed_features = model.predict({"z": [0.5, -0.5]})
    assert changed_features.parent_claim == model.parent_claim
    assert np.allclose(changed_features.values, [2.2, 1.8], atol=0.12, rtol=0)
    changed_model = fitted_result(3.0).fitted_model
    assert changed_model.parent_claim != model.parent_claim
    assert np.allclose(changed_model.predict({"z": [0.0]}).values, [3.0], atol=0.12, rtol=0)
