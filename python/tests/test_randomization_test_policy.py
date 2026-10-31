"""Fisher row-work policy reaches the public complete-randomization route."""

from dataclasses import FrozenInstanceError

import antecedent as ant
import pytest
from antecedent.errors import CausalResourceError, CausalTypeError, CausalValueError
from antecedent.outcomes import require_success


@pytest.mark.parametrize(
    "value,error",
    [
        (None, CausalTypeError),
        (True, CausalTypeError),
        (1.0, CausalTypeError),
        (-1, CausalValueError),
        (2**64, CausalValueError),
    ],
)
def test_limits_validate_finite_u64(value, error):
    with pytest.raises(error):
        ant.RandomizationTestLimits(max_row_visits=value)


def test_limits_are_frozen_and_passed_to_native(monkeypatch):
    import antecedent.execution as module

    captured = []
    monkeypatch.setattr(module._native, "_Execution", lambda **kw: captured.append(kw))
    limits = ant.RandomizationTestLimits(0)
    with pytest.raises(FrozenInstanceError):
        limits.max_row_visits = 1
    ant.Execution()
    ant.Execution(randomization_test=limits)
    assert captured[0]["randomization_test_max_row_visits"] is None
    assert captured[1]["randomization_test_max_row_visits"] == 0
    with pytest.raises(CausalTypeError):
        ant.Execution(randomization_test={})


def test_complete_design_refuses_work_over_limit_without_reducing_allocations():
    design = ant.experiment.ExperimentDesign(
        ant.interference.CompleteRandomization(2),
        [True, True, False, False],
        [f"u{i}" for i in range(4)],
        [f"y{i}" for i in range(4)],
    )
    query = ant.experiment.RandomizedEffect("y", design, exact_randomization_test=True)

    def run(limit):
        return require_success(
            ant.analyze(
                {"y": [1.0, 2.0, 3.0, 4.0]},
                query=query,
                refute="none",
                execution=ant.Execution(randomization_test=ant.RandomizationTestLimits(limit)),
            )
        )

    with pytest.raises(CausalResourceError, match="row visits"):
        run(27)
    result = run(28).randomized_effect
    assert result.randomization_allocations == 6
    assert result.randomization_p_value == pytest.approx(1 / 3)
