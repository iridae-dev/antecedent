"""Validator inspection and branching keep callbacks bound to their own study."""

import antecedent as ant
import numpy as np
from antecedent.execution_attempt import observe_native_attempts
from antecedent.outcomes import require_success


def test_validator_names_are_nonexecuting_and_reconfiguration_preserves_original_callbacks():
    calls = []

    def original(*, ate, **_kwargs):
        calls.append(("original", ate))
        return {"passed": True, "refuted_ate": ate, "comparison": 0.0}

    def alternative(*, ate, **_kwargs):
        calls.append(("alternative", ate))
        return {"passed": True, "refuted_ate": ate, "comparison": 0.0}

    treatment = np.tile([0.0, 1.0], 80)
    covariate = np.repeat(np.linspace(-1.0, 1.0, 80), 2)
    bindings = {"original.check": original}
    study = ant.prepare(
        {"t": treatment, "z": covariate, "y": 2 * treatment + covariate},
        graph=[("t", "y"), ("z", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
        validators=bindings,
        execution=ant.Execution(seed=19, threads=1),
    )
    bindings.clear()
    read = observe_native_attempts(lambda: study.validator_names)
    assert read.unwrap() == ("original.check",)
    assert read.report.is_complete
    assert all(operation.attempted == 0 for operation in read.report.operations.values())
    assert calls == []

    branch = ant.reconfigure(study, validators={"alternative.check": alternative})
    assert study.validator_names == ("original.check",)
    assert branch.validator_names == ("alternative.check",)
    assert calls == []
    first = require_success(ant.estimate(study))
    assert np.isclose(first.effect, 2.0)
    assert {name for name, _ in calls} == {"original"}
    calls.clear()
    second = require_success(ant.estimate(branch))
    assert np.isclose(second.effect, 2.0)
    assert {name for name, _ in calls} == {"alternative"}
    assert study.validator_names == ("original.check",)
