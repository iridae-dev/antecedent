"""Retention refusals as values and as exceptions, on ``antecedent._native``.

A reduced retention mode names what it can no longer serve. The refusal of one later operation
is available as a ``RetentionRefusal`` value (``refusal``) and as the ``CausalStateError`` a
failed ``require`` raises; both carry the mode asked, the operation refused and everything
that mode can no longer serve. The cases match ``python/src/contract_refusals_api.rs``.
"""

from __future__ import annotations

import pytest
from antecedent import _native
from antecedent.errors import CausalStateError

OPERATIONS = (
    "row_reconstruction",
    "refit_from_retained_rows",
    "exact_row_eviction",
    "in_place_column_insert",
    "in_place_weight_or_mask_update",
    "reuse_full_state",
    "reuse_summary",
    "read_disk_artifact",
)


def _operation(name: str) -> _native.RetainedOperation:
    return getattr(_native.RetainedOperation, name)()


def _names(operations) -> list[str]:
    return [operation.as_str() for operation in operations]


def test_summary_only_refusal_is_a_value_naming_everything_it_cannot_serve():
    mode = _native.RetentionMode.summary_only()
    lost = _names(mode.no_longer_serves())
    assert lost, "summary-only drops operations"
    assert _operation("reuse_summary").as_str() not in lost
    refusal = mode.refusal(_native.RetainedOperation.refit_from_retained_rows())
    assert isinstance(refusal, _native.RetentionRefusal)
    assert refusal.mode().as_str() == "summary-only"
    assert refusal.operation().as_str() == _operation("refit_from_retained_rows").as_str()
    assert _names(refusal.no_longer_serves()) == lost
    assert refusal.message() == f"summary-only cannot serve {refusal.operation().as_str()}"
    assert refusal.message() == "summary-only cannot serve refit from retained rows"
    assert str(refusal) == refusal.message()
    assert mode.refusal(_native.RetainedOperation.reuse_summary()) is None


def test_every_operation_is_served_or_refused_consistently_by_every_mode():
    modes = [
        _native.RetentionMode.full_reusable(),
        _native.RetentionMode.summary_only(),
        _native.RetentionMode.disk_backed(),
    ]
    for mode in modes:
        lost = _names(mode.no_longer_serves())
        for name in OPERATIONS:
            operation = _operation(name)
            refusal = mode.refusal(operation)
            assert (refusal is None) == mode.serves(operation), (mode.as_str(), name)
            if refusal is None:
                mode.require(operation)
                continue
            with pytest.raises(CausalStateError):
                mode.require(operation)
            assert refusal.mode().as_str() == mode.as_str()
            assert _names(refusal.no_longer_serves()) == lost
            # A mode loses what full state served; an undeclared disk read was never served.
            if operation.as_str() not in lost:
                assert name == "read_disk_artifact"


def test_the_raised_refusal_carries_what_the_value_carries():
    mode = _native.RetentionMode.summary_only()
    operation = _native.RetainedOperation.exact_row_eviction()
    refusal = mode.refusal(operation)
    assert refusal is not None
    with pytest.raises(CausalStateError, match="summary-only cannot serve") as raised:
        mode.require(operation)
    assert raised.value.mode == refusal.mode().as_str()
    assert raised.value.operation == refusal.operation().as_str()
    assert raised.value.no_longer_serves == _names(refusal.no_longer_serves())
    with pytest.raises(CausalStateError) as again:
        refusal.raise_error()
    assert str(again.value) == str(raised.value)
    assert again.value.no_longer_serves == raised.value.no_longer_serves


def test_a_disk_backed_mode_serves_only_the_artifact_read():
    mode = _native.RetentionMode.disk_backed()
    assert mode.refusal(_native.RetainedOperation.read_disk_artifact()) is None
    refusal = mode.refusal(_native.RetainedOperation.reuse_full_state())
    assert refusal is not None
    assert refusal.mode().as_str() == "disk-backed output"
    lost = _names(refusal.no_longer_serves())
    assert _operation("read_disk_artifact").as_str() not in lost
    assert len(lost) == len(OPERATIONS) - 1
    with pytest.raises(CausalStateError) as raised:
        mode.require(_native.RetainedOperation.refit_from_retained_rows())
    assert raised.value.no_longer_serves == lost


def test_full_reusable_state_refuses_only_an_undeclared_disk_read():
    mode = _native.RetentionMode.default()
    refusal = mode.refusal(_native.RetainedOperation.read_disk_artifact())
    assert refusal is not None
    assert refusal.no_longer_serves() == []
    with pytest.raises(CausalStateError) as raised:
        mode.require(_native.RetainedOperation.read_disk_artifact())
    assert raised.value.no_longer_serves == []
    for name in OPERATIONS:
        if name != "read_disk_artifact":
            assert mode.refusal(_operation(name)) is None


def test_a_policy_refuses_through_its_result_mode():
    policy = _native.RetentionPolicy.sufficient_statistics()
    refusal = policy.refusal(_native.RetainedOperation.row_reconstruction())
    assert refusal is not None
    assert refusal.mode().as_str() == policy.result_mode().as_str()
    assert _names(refusal.no_longer_serves()) == _names(policy.no_longer_serves())
    assert policy.refusal(_native.RetainedOperation.reuse_summary()) is None
    raw = _native.RetentionPolicy.raw_history()
    assert raw.refusal(_native.RetainedOperation.row_reconstruction()) is None
    assert (
        _native.RetentionPolicy.bounded_window(0).refusal(
            _native.RetainedOperation.exact_row_eviction()
        )
        is not None
    )
