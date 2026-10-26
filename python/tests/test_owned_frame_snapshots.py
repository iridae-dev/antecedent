"""Frame constructors own mutable observations and share native snapshot partitions."""

import gc

import antecedent as ant
import numpy as np
import pytest
from antecedent import _native
from antecedent.prepared import estimate, prepare, rebind


def _series(effect=2.0, n=64):
    x = 1 + np.sin(np.arange(n) * 0.37)
    y = np.zeros(n)
    y[1:] = 3 + effect * x[:-1]
    return {"x": x, "y": y}


@pytest.mark.parametrize(
    "constructor,attribute", [(ant.data.panel, "unit_columns"), (ant.data.multi_env, "env_columns")]
)
def test_frame_partitions_own_inputs_once_and_reenter_native_without_copy(constructor, attribute):
    data = [_series(), _series(4.0)]
    frame = constructor(data)
    assert frame.names == ("x", "y")
    assert sum(part.bytes_copied for part in frame._snapshots) == 2 * 2 * 64 * 8
    original = getattr(frame, attribute)[0][0].copy()
    data[0]["x"][:] = -100
    np.testing.assert_array_equal(getattr(frame, attribute)[0][0], original)
    for columns in getattr(frame, attribute):
        assert all(not column.flags.writeable for column in columns)
        probe = _native.load_float64_columns(list(frame.names), list(columns))
        assert probe.bytes_copied == 0
    view = getattr(frame, attribute)[0][0]
    del frame
    gc.collect()
    np.testing.assert_array_equal(view, original)


def test_event_indices_and_values_are_owned_read_only_and_pin_native_allocations():
    data = _series()
    times = np.arange(64, dtype=np.int64) * 100
    frame = ant.data.event(data, times, align_interval_ns=100)
    data["x"][:] = -100
    times[:] = -100
    expected = np.arange(64, dtype=np.int64) * 100
    np.testing.assert_array_equal(frame.event_times_ns, expected)
    assert frame.event_times_ns.dtype == np.int64
    assert not frame.event_times_ns.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        frame.event_times_ns.setflags(write=True)
    times_view = frame.event_times_ns
    del frame
    gc.collect()
    np.testing.assert_array_equal(times_view, expected)


def test_event_preparation_and_branch_use_owned_snapshots_and_leave_prior_answers_unchanged():
    data = _series()
    times = np.arange(64, dtype=np.int64) * 100
    frame = ant.data.event(data, times, align_interval_ns=100)
    graph = ant.TemporalDag.from_lagged_edges(["x", "y"], [("x", 1, "y", 0)])
    study = prepare(
        frame,
        graph=graph,
        query=ant.PulseEffect("x", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    )
    data["y"][:] = 0
    times[:] = -1
    before = estimate(study)
    assert before.answer.value == pytest.approx(2.0)
    new_frame = ant.data.event(
        _series(4.0), np.arange(64, dtype=np.int64) * 100, align_interval_ns=100
    )
    branch = rebind(study, data=new_frame)
    assert branch.inspect().stage == "prepared"
    assert estimate(branch).answer.value == pytest.approx(4.0)
    assert estimate(study).answer.value == pytest.approx(2.0)
    assert before.answer.value == pytest.approx(2.0)
