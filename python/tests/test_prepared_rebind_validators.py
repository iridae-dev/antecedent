"""Rebinding validators reaches the study that the last execution ran on."""

import antecedent as ant
import numpy as np
from antecedent.outcomes import require_success


def _validator(calls, label):
    def run(*, ate, **_kwargs):
        calls.append(label)
        return {"passed": True, "refuted_ate": ate, "comparison": 0.0}

    return run


def test_rebind_reaches_bound_refutation():
    calls = []
    treatment = np.tile([0.0, 1.0], 80)
    covariate = np.repeat(np.linspace(-1.0, 1.0, 80), 2)
    study = ant.prepare(
        {"t": treatment, "z": covariate, "y": 2 * treatment + covariate},
        graph=[("t", "y"), ("z", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
        validators={"check": _validator(calls, "first")},
        execution=ant.Execution(seed=19, threads=1),
    )
    study._engine.estimate()
    calls.clear()
    study._engine.rebind_validators({"check": _validator(calls, "second")})
    study._engine.refute(suite="placebo")
    assert calls
    assert set(calls) == {"second"}
