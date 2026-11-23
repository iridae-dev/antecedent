"""Immutable lifecycle assertions through the canonical prepared implementation."""

from dataclasses import FrozenInstanceError

import antecedent as ant
import numpy as np
import pytest
from antecedent.execution import Execution
from antecedent.outcomes import require_success
from antecedent.prepared import estimate, prepare, rebind, reconfigure


def _data(effect=2.0):
    t = np.tile([0.0, 1.0], 32)
    z = np.repeat(np.linspace(-1.0, 1.0, 32), 2)
    return {"t": t, "z": z, "y": effect * t + z}


def _prepare(data=None):
    graph = ant.Dag.from_edges(["t", "z", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    return prepare(
        _data() if data is None else data,
        graph=graph,
        query=ant.AverageEffect("t", "y"),
        refute=False,
    )


def test_branches_leave_original_study_and_outcome_unchanged():
    parent = _prepare()
    answer = estimate(parent)
    branch = rebind(parent, data=_data(4.0))
    assert parent._engine._last_result is None
    assert branch._engine._last_result is None
    assert estimate(branch).answer.value == pytest.approx(4.0)
    assert estimate(parent).answer.value == pytest.approx(2.0)
    assert answer.answer.value == pytest.approx(2.0)
    assert answer.study is not None
    assert answer.study._engine is not parent._engine
    assert "fits" in branch.reuse.invalidated


def test_mutating_inputs_after_preparation_cannot_change_result_or_reconfiguration():
    data = _data()
    study = _prepare(data)
    data["y"][:] = 0.0
    assert estimate(study).answer.value == pytest.approx(2.0)
    branch = reconfigure(study, execution=Execution(seed=19, threads=1))
    assert branch._engine is not study._engine
    assert branch._data is study._data
    assert branch._engine._seed == 19
    assert estimate(branch).answer.value == pytest.approx(2.0)
    changed = reconfigure(study, query=ant.AverageEffect("t", "y"))
    assert estimate(changed).answer.value == pytest.approx(2.0)


def test_prepared_declarations_are_immutable_and_representations_do_not_execute(monkeypatch):
    study = _prepare()

    def unexpected(*args, **kwargs):
        raise AssertionError("representation executed an estimate")

    monkeypatch.setattr(type(study._engine), "estimate", unexpected)
    assert "stage='prepared'" in repr(study)
    assert "prepared" in study._repr_html_()
    assert study.inspect().stage == "prepared"
    branch = rebind(study, data=_data(3.0))
    assert branch.inspect().stage == "prepared"
    assert study._engine._last_result is None
    assert branch._engine._last_result is None
    with pytest.raises(FrozenInstanceError):
        study.execution = Execution(seed=2)


def test_unwrapped_result_retains_the_same_immutable_completed_study():
    outcome = estimate(_prepare())
    value = require_success(outcome)
    assert value.study is outcome.study
    assert not hasattr(value, "refresh")
    original = bytes(outcome.export())
    branch = value.study.rebind(data=_data(4.0))
    assert estimate(branch).answer.value == pytest.approx(4.0)
    assert estimate(value.study).answer.value == pytest.approx(2.0)
    assert bytes(outcome.export()) == original
    assert value.study.inspect().data_snapshot_id == value.data_snapshot_id


def test_population_retarget_uses_retained_scores_without_refitting(monkeypatch):
    data = _data()
    outcome = ant.analyze(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        estimator="aipw",
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    )
    value = require_success(outcome)

    def unexpected(*args, **kwargs):
        pytest.fail("population retarget refitted an estimator")

    monkeypatch.setattr(type(value.study._engine), "estimate", unexpected)
    targeted = value.study.retarget(np.exp(data["z"] / 3), depends_on=["z"])
    require_success(targeted)
    assert targeted.inspect().target_weights_id is not None
    assert outcome.inspect().target_weights_id is None
    assert targeted.inspect().data_snapshot_id == outcome.inspect().data_snapshot_id
    loaded = ant.load(targeted.export())
    assert loaded.study is None
    assert loaded.inspect().target_weights_id == targeted.inspect().target_weights_id


def test_refutation_executes_on_an_independent_completed_study():
    outcome = estimate(_prepare())
    value = require_success(outcome)
    before = bytes(outcome.export())
    original_inspection = outcome.study.inspect()
    checked = outcome.study.refute(suite="cheap")
    result = require_success(checked)
    assert result.validation.count > 0
    assert checked.answer.value == pytest.approx(outcome.answer.value)
    assert checked.study is result.study
    assert checked.study is not outcome.study
    assert outcome.study.inspect() == original_inspection
    assert bytes(outcome.export()) == before
    assert value.study is outcome.study


def test_declared_response_units_keep_the_same_native_record_after_loading():
    treatment = np.linspace(-1.0, 1.0, 128)
    covariate = np.sin(np.arange(128))
    outcome = ant.analyze(
        {"t": treatment, "z": covariate, "y": 2 * treatment + covariate},
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.ResponseCurve("t", "y", grid=[0.0, 0.5, 1.0]),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
        outcome_units="kg",
        quantity_population="clinic",
    )
    require_success(outcome)
    artifact = outcome.export()
    loaded = ant.load(artifact, expected=artifact.expectation)
    require_success(loaded)
    assert outcome.inspect().contract["format"] == 2
    assert outcome.inspect().contract == loaded.inspect().contract
    assert outcome.inspect().claim_id == loaded.inspect().claim_id
    assert outcome.value.quantities is not None
    assert all(quantity.units == "kg" for quantity in outcome.value.quantities)
