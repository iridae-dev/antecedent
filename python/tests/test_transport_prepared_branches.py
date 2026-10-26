"""Specialized transport preparations branch; completed executions own their artifacts."""

from dataclasses import FrozenInstanceError, replace

import pytest
from antecedent.errors import CausalUnsupportedError
from antecedent.execution import Execution
from antecedent.outcomes import require_success
from antecedent.transport import advanced

from test_learned_continuous_transport import fixture as continuous_fixture
from test_learned_continuous_transport import options as continuous_options
from test_smoothed_dose_transport import fixture as dose_fixture
from test_smoothed_dose_transport import options as dose_options


def test_learned_transport_branches_without_mutating_the_preparation_or_earlier_run():
    query, data = continuous_fixture()
    study = advanced.prepare_learned_continuous(
        query, data, options=continuous_options(), execution=Execution(seed=5)
    )
    with pytest.raises(CausalUnsupportedError):
        study._native.export()
    result = require_success(study.estimate())
    original = bytes(result.export())
    shifted = replace(
        data, outcome=tuple(y + 2.0 * a for y, a in zip(data.outcome, data.treatment, strict=True))
    )
    branch = study.bind_snapshot(shifted)
    assert branch is not study
    with pytest.raises(CausalUnsupportedError):
        branch._native.export()
    with pytest.raises(CausalUnsupportedError):
        study._native.export()
    changed = require_success(branch.estimate())
    assert changed.estimate == pytest.approx(result.estimate + 2.0, abs=1e-9)
    assert bytes(result.export()) == original
    replayed = advanced.consume_learned_continuous(result.export())
    assert replayed.estimate == result.estimate
    assert replayed.acceptance.verified and replayed.acceptance.replayable
    assert not replayed.acceptance.resumable
    with pytest.raises(FrozenInstanceError):
        study._execution = Execution(seed=9)


def test_smoothed_transport_branches_preserve_the_entire_grid_and_prior_execution():
    query, data = dose_fixture(n_trial=600, n_target=400)
    study = advanced.prepare_smoothed_dose(
        query, data, options=dose_options(), execution=Execution(seed=5)
    )
    result = require_success(study.estimate())
    original = bytes(result.export())
    branch = study.bind_snapshot(replace(data, outcome=tuple(y + 1.0 for y in data.outcome)))
    with pytest.raises(CausalUnsupportedError):
        branch._native.export()
    with pytest.raises(CausalUnsupportedError):
        study._native.export()
    changed = require_success(branch.estimate())
    assert tuple(point.dose for point in changed.grid) == tuple(point.dose for point in result.grid)
    assert tuple(point.estimate for point in changed.grid) == pytest.approx(
        tuple(point.estimate + 1.0 for point in result.grid), abs=1e-8
    )
    assert bytes(result.export()) == original
    replayed = advanced.consume_smoothed_dose(result.export())
    assert tuple(point.estimate for point in replayed.grid) == tuple(
        point.estimate for point in result.grid
    )
    assert replayed.acceptance.verified and replayed.acceptance.replayable
