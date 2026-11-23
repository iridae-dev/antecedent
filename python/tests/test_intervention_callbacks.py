"""Batch callbacks in intervention programs: execution, validation, failures and portability."""

from __future__ import annotations

from dataclasses import replace

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalUnsupportedError, CausalValueError
from antecedent.intervention_callbacks import (
    GENERATED,
    CallbackProgram,
    CallbackRecord,
    InterventionCallback,
    callback_records,
    recover_callback,
    sample_do,
)

N_DRAWS = 64


@pytest.fixture(scope="module")
def gcm():
    rng = np.random.default_rng(11)
    x = rng.normal(size=400)
    y = 2.0 * x + rng.normal(scale=0.5, size=400)
    return antecedent.model.fit_gcm(["x", "y"], [x, y], [("x", "y")])


def _column(result, model, name):
    return result.draws[model.names.index(name)]


def _refusal(code, program, model, **kwargs):
    with pytest.raises(CausalUnsupportedError) as raised:
        sample_do(model, [program], N_DRAWS, execution=antecedent.Execution(seed=3), **kwargs)
    assert raised.value.reason_code == code, raised.value
    return str(raised.value)


def _assign_y(function, **declaration):
    callback = InterventionCallback(function, inputs=["x"], name="f", **declaration)
    return CallbackProgram("y", callback)


def test_an_assignment_callback_is_the_function_of_its_inputs_on_every_row(gcm):
    program = _assign_y(lambda x: 3.0 * x + 1.0)
    result = sample_do(gcm, [program], N_DRAWS, execution=antecedent.Execution(seed=5))
    x, y = _column(result, gcm, "x"), _column(result, gcm, "y")
    np.testing.assert_allclose(y, 3.0 * x + 1.0, rtol=0, atol=1e-12)


def test_a_transform_callback_reads_the_generated_value_and_keeps_the_noise(gcm):
    floor = InterventionCallback(lambda g: np.maximum(g, 0.0), inputs=["x"], name="floor")
    program = CallbackProgram("x", floor, reads=[GENERATED], mode="transform")
    baseline = gcm.sample_do({}, N_DRAWS, execution=antecedent.Execution(seed=9))
    result = sample_do(gcm, [program], N_DRAWS, execution=antecedent.Execution(seed=9))
    before = baseline.draws[gcm.names.index("x")]
    after = _column(result, gcm, "x")
    np.testing.assert_array_equal(after, np.maximum(before, 0.0))
    assert (after >= 0.0).all() and (before < 0.0).any()


def test_the_callback_is_called_once_per_batch_with_whole_float64_columns(gcm):
    seen = []

    def record(x):
        seen.append((x.dtype, x.shape))
        return 2.0 * x

    result = sample_do(gcm, [_assign_y(record)], N_DRAWS, execution=antecedent.Execution(seed=1))
    assert result.n_draws == N_DRAWS
    assert seen == [(np.dtype("float64"), (N_DRAWS,))]


def test_draws_are_deterministic_per_seed_and_vary_across_seeds(gcm):
    program = _assign_y(np.tanh)
    first = sample_do(gcm, [program], N_DRAWS, execution=antecedent.Execution(seed=21)).draws
    again = sample_do(gcm, [program], N_DRAWS, execution=antecedent.Execution(seed=21)).draws
    other = sample_do(gcm, [program], N_DRAWS, execution=antecedent.Execution(seed=22)).draws
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


@pytest.mark.parametrize(
    "function",
    [
        lambda x: x[:-1],
        lambda x: np.stack([x, x]),
        lambda x: float(x.mean()),
        lambda x: np.array(["a"] * len(x)),
        lambda x: np.array([None] * len(x), dtype=object),
        lambda x: x.astype(complex),
    ],
    ids=["short", "two-dimensional", "scalar", "string", "object", "complex"],
)
def test_an_output_of_the_wrong_shape_or_dtype_is_refused(gcm, function):
    message = _refusal("policy_callback_output_invalid", _assign_y(function), gcm)
    assert "f" in message


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_output_is_refused_not_propagated(gcm, bad):
    def function(x):
        out = 2.0 * x
        out[7] = bad
        return out

    message = _refusal("policy_callback_output_invalid", _assign_y(function), gcm)
    assert "row 7" in message


def test_the_declared_range_and_levels_bind_the_output(gcm):
    message = _refusal(
        "policy_callback_output_invalid", _assign_y(lambda x: x, lower=-0.1, upper=0.1), gcm
    )
    assert "range" in message
    inside = sample_do(
        gcm,
        [_assign_y(np.tanh, lower=-1.0, upper=1.0)],
        N_DRAWS,
        execution=antecedent.Execution(seed=2),
    )
    assert np.abs(_column(inside, gcm, "y")).max() <= 1.0
    _refusal(
        "policy_callback_output_invalid",
        _assign_y(lambda x: (x > 0).astype(float) * 2.0, levels=[0.0, 1.0]),
        gcm,
    )
    ok = sample_do(
        gcm,
        [_assign_y(lambda x: (x > 0).astype(int), levels=[0, 1])],
        N_DRAWS,
        execution=antecedent.Execution(seed=2),
    )
    assert set(np.unique(_column(ok, gcm, "y"))) <= {0.0, 1.0}


def test_an_exception_in_the_callback_surfaces_with_its_message(gcm):
    def raising(x):
        raise ValueError("no demand model for region 7")

    message = _refusal("policy_callback_failed", _assign_y(raising), gcm)
    assert "no demand model for region 7" in message
    assert "ValueError" in message


def test_callback_inputs_are_dependency_checked_like_any_read(gcm):
    callback = InterventionCallback(lambda y: y, inputs=["y"], name="back")
    cyclic = CallbackProgram("x", callback)
    _refusal("policy_dependency_circular", cyclic, gcm)
    other = CallbackProgram("y", InterventionCallback(lambda x: x, inputs=["x"], name="g"))
    _refusal("policy_directive_conflict", other, gcm, interventions={"y": 1.0})
    unknown = CallbackProgram("y", callback, reads=["nope"])
    with pytest.raises(CausalError):
        sample_do(gcm, [unknown], N_DRAWS, execution=antecedent.Execution(seed=3))


def test_program_declarations_are_validated_before_anything_runs():
    callback = InterventionCallback(lambda x: x, inputs=["x"], name="f")
    with pytest.raises(CausalValueError, match="declares 1 inputs"):
        CallbackProgram("y", callback, reads=["x", "x"])
    with pytest.raises(CausalValueError, match="generated"):
        CallbackProgram("y", callback, reads=[GENERATED])
    with pytest.raises(CausalValueError, match="distinct"):
        InterventionCallback(lambda a, b: a, inputs=["x", "x"], name="dup")
    with pytest.raises(CausalValueError, match="lower"):
        InterventionCallback(lambda x: x, inputs=["x"], name="r", lower=0.0)
    with pytest.raises(CausalValueError, match="callable"):
        InterventionCallback("not callable", inputs=["x"], name="n")  # type: ignore[arg-type]


SOURCE = """
def scaled(x):
    return np.clip(2.0 * x, -3.0, 3.0)
"""


def test_a_runtime_binding_is_not_exportable_and_says_why():
    callback = InterventionCallback(lambda x: x, inputs=["x"], name="closure")
    assert not callback.recoverable
    program = CallbackProgram("y", callback)
    for target in (callback.recovery_record, lambda: callback_records([program])):
        with pytest.raises(CausalUnsupportedError) as raised:
            target()
        assert raised.value.reason_code == "policy_callback_not_portable"
        assert "closure" in str(raised.value)
    recoverable = InterventionCallback.from_source(
        SOURCE, entry_point="scaled", inputs=["x"], name="scaled"
    )
    mixed = [CallbackProgram("y", recoverable), program]
    with pytest.raises(CausalUnsupportedError, match="closure"):
        callback_records(mixed)


def test_a_source_declared_callback_round_trips_through_a_verified_record(gcm):
    original = InterventionCallback.from_source(
        SOURCE, entry_point="scaled", inputs=["x"], name="scaled", lower=-3.0, upper=3.0
    )
    assert original.recoverable
    (record,) = callback_records([CallbackProgram("y", original)])
    assert len(record.source_sha256) == 64
    assert dict(record.dependencies)["numpy"] == np.__version__
    restored_record = CallbackRecord.from_dict(record.to_dict())
    assert restored_record == record

    replayed = recover_callback(SOURCE, restored_record)
    first = sample_do(
        gcm, [CallbackProgram("y", original)], N_DRAWS, execution=antecedent.Execution(seed=4)
    ).draws
    again = sample_do(
        gcm, [CallbackProgram("y", replayed)], N_DRAWS, execution=antecedent.Execution(seed=4)
    ).draws
    np.testing.assert_array_equal(first, again)
    x, y = first[gcm.names.index("x")], first[gcm.names.index("y")]
    np.testing.assert_allclose(y, np.clip(2.0 * x, -3.0, 3.0))


@pytest.mark.parametrize(
    "tamper",
    [
        lambda s: s.replace("2.0", "2.5"),
        lambda s: s + "\n",
    ],
    ids=["edited", "whitespace"],
)
def test_replay_refuses_source_that_does_not_match_the_record(tamper):
    original = InterventionCallback.from_source(SOURCE, entry_point="scaled", inputs=["x"])
    record = original.recovery_record()
    with pytest.raises(CausalUnsupportedError) as raised:
        recover_callback(tamper(SOURCE), record)
    assert raised.value.reason_code == "policy_callback_unrecoverable"
    assert "source_sha256" in str(raised.value)


def test_replay_refuses_unavailable_pinned_dependencies():
    record = InterventionCallback.from_source(
        SOURCE, entry_point="scaled", inputs=["x"]
    ).recovery_record()
    stale = replace(record, dependencies=(("numpy", "0.0.1"),))
    with pytest.raises(CausalUnsupportedError) as raised:
        recover_callback(SOURCE, stale)
    assert raised.value.reason_code == "policy_callback_unrecoverable"
    assert "numpy" in str(raised.value)
    with pytest.raises(CausalUnsupportedError) as missing:
        InterventionCallback.from_source(
            SOURCE,
            entry_point="scaled",
            inputs=["x"],
            dependencies={"no-such-distribution-xyz": "1.0"},
        )
    assert missing.value.reason_code == "policy_callback_unrecoverable"


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef f(x):\n    return x\n",
        "def f(x):\n    return eval('x')\n",
        "def f(x):\n    return x.__class__\n",
        "def f(x):\n    global y\n    return x\n",
    ],
    ids=["import", "eval", "dunder", "global"],
)
def test_source_that_could_reach_outside_its_declared_dependencies_is_refused(source):
    with pytest.raises(CausalValueError):
        InterventionCallback.from_source(source, entry_point="f", inputs=["x"])


def test_source_must_define_its_entry_point():
    with pytest.raises(CausalValueError, match="missing"):
        InterventionCallback.from_source(SOURCE, entry_point="missing", inputs=["x"])
