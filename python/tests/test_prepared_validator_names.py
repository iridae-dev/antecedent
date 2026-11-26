"""Validator inspection and branching keep callbacks bound to their own study."""

import antecedent as ant
import numpy as np
import pytest
from antecedent.execution_attempt import observe_native_attempts
from antecedent.outcomes import require_success
from antecedent.validation import validator


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


def test_named_validator_in_a_sequence_keeps_the_declared_name_through_preparation():
    calls = []

    @validator(name="minimum_clinical_effect")
    def clinical_relevance(*, ate, se_analytic, **kwargs):
        calls.append((ate, kwargs["method"]))
        return {"passed": ate >= 0.5, "comparison": ate - 0.5}

    def plain_check(**kwargs):
        return {"passed": True}

    t = np.tile([0.0, 1.0], 80)
    z = np.repeat(np.linspace(-1.0, 1.0, 80), 2)
    arguments = {
        "data": {"t": t, "y": 2 * t + z},
        "graph": [("t", "y")],
        "query": ant.AverageEffect("t", "y"),
        "inference": ant.Frequentist(bootstrap=0),
        "refute": False,
    }
    study = ant.prepare(**arguments, validators=[clinical_relevance, plain_check])
    assert set(study.validator_names) == {
        "minimum_clinical_effect",
        f"{plain_check.__module__}.{plain_check.__qualname__}",
    }
    assert calls == []
    assert require_success(ant.estimate(study)).effect == pytest.approx(2.0)
    assert calls and calls[0][0] == pytest.approx(2.0)
    with pytest.raises(ant.errors.CausalValueError, match="validator name.*is not unique"):
        ant.prepare(**arguments, validators=[clinical_relevance, clinical_relevance])
