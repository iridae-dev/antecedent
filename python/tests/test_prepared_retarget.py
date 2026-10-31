"""Retargeting honours a cancellation token like the other prepared executions."""

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalCancelledError
from antecedent.outcomes import require_success


def _data():
    t = np.tile([0.0, 1.0], 32)
    z = np.repeat(np.linspace(-1.0, 1.0, 32), 2)
    return {"t": t, "z": z, "y": 2.0 * t + z}


def test_retarget_accepts_cancel_token():
    outcome = ant.analyze(
        _data(),
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        estimator="aipw",
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    )
    study = require_success(outcome).study
    token = ant.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        study._engine.retarget(np.exp(_data()["z"] / 3), ["z"], cancel=token)
