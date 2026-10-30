"""Fresh-process prediction uses portable numerical state, not a live prepared study."""

import json
import subprocess
import sys

import antecedent as ac
import numpy as np
from antecedent.estimators import DRLearner
from antecedent.outcomes import require_success
from antecedent.prediction import FittedEffectModel


def fitted_result(effect=2.0):
    rng = np.random.default_rng(19)
    z = np.tile(np.linspace(-1.0, 1.0, 21), 80)
    treatment = rng.binomial(1, 0.5, len(z)).astype(float)
    return require_success(
        ac.analyze(
            {"z": z, "t": treatment, "y": treatment * (effect + 0.4 * z) + z},
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=ac.AverageEffect(treatment="t", outcome="y"),
            estimator=DRLearner(),
            inference=ac.Frequentist(bootstrap=0),
            refute=False,
            execution=ac.Execution(seed=3),
        )
    )


def test_portable_predictor_fresh_process_matches_full_rerun_and_scm_truth(tmp_path):
    result = fitted_result()
    model = result.fitted_model
    features = {"z": [-1.0, 0.0, 1.0]}
    expected = require_success(model.predict(features))
    assert np.allclose(expected.values, [1.6, 2.0, 2.4], atol=2e-3, rtol=0)
    assert require_success(fitted_result().fitted_model.predict(features)) == expected
    artifact = tmp_path / "predictor.ant"
    artifact.write_bytes(bytes(model.export()))
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


def test_native_predictor_snapshots_contiguous_and_strided_columns():
    model = FittedEffectModel.load(bytes(fitted_result().export()))
    contiguous = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    interleaved = np.array([-1.0, 99.0, 0.0, 99.0, 1.0, 99.0], dtype=np.float64)
    strided = interleaved[::2]
    assert not strided.flags.c_contiguous
    expected = np.asarray(model._native.predict([contiguous], len(contiguous)))
    actual = np.asarray(model._native.predict([strided], len(strided)))
    np.testing.assert_array_equal(actual, expected)
    contiguous[:] = 10.0
    interleaved[:] = 20.0
    np.testing.assert_array_equal(actual, expected)


def test_predictor_new_rows_preserve_parent_but_new_outcomes_need_a_new_fit():
    model = FittedEffectModel.load(bytes(fitted_result().export()))
    changed_features = require_success(model.predict({"z": [0.5, -0.5]}))
    assert changed_features.parent_claim == model.parent_claim
    assert np.allclose(changed_features.values, [2.2, 1.8], atol=2e-3, rtol=0)
    changed_model = fitted_result(3.0).fitted_model
    assert changed_model.parent_claim != model.parent_claim
    assert np.allclose(
        require_success(changed_model.predict({"z": [0.0]})).values, [3.0], atol=2e-3, rtol=0
    )
