"""Immutable lifecycle assertions through the canonical prepared implementation."""

from dataclasses import FrozenInstanceError

import antecedent as ant
import numpy as np
import pytest
from antecedent.execution import Execution
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
