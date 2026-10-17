"""The recalculation bridge at scale and at its limits.

Wide numeric tables cross the Python/native boundary under memory budgets rather than fixed
ceilings: a table the budget allows is accepted and its recalculated result is the direct
native result; a table the budget (or the checked byte arithmetic) cannot allow is refused
before it is copied, and a cancelled run leaves the session untouched.
"""

from __future__ import annotations

import json
import time
import tracemalloc

import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import CausalValueError
from antecedent.recalc import RecalcRefusal, RecalcRequest, RecalcSession, Utility

TRUE_EFFECT = 2.0


def wide_table(rows: int, columns: int, seed: int) -> RecalcRequest:
    """`columns` variables in all: a binary treatment, an outcome and `columns - 2` adjusters."""
    rng = np.random.default_rng(seed)
    adjusters = {f"z{i}": rng.normal(size=rows) for i in range(columns - 2)}
    total = sum(adjusters.values())
    treatment = (rng.random(rows) < 1 / (1 + np.exp(-0.2 * total))).astype(np.float64)
    outcome = TRUE_EFFECT * treatment + 0.1 * total + rng.normal(size=rows)
    edges = [(name, "t") for name in adjusters] + [(name, "y") for name in adjusters]
    return RecalcRequest(
        data={**adjusters, "t": treatment, "y": outcome},
        edges=[*edges, ("t", "y")],
        treatment="t",
        outcome="y",
        utility=Utility(1.0, 0.0),
    )


def direct_native_ate(request: RecalcRequest, seed: int) -> float:
    """The average effect from the native session called without the Python request layer."""
    handle = _native.RecalcSessionHandle("licensed")
    names = list(request.data)
    columns = [np.ascontiguousarray(request.data[name], dtype=np.float64) for name in names]
    result, _, refusal = handle.execute(
        names,
        columns,
        [(str(a), str(b)) for a, b in request.edges],
        request.treatment,
        request.outcome,
        request.utility.benefit_per_unit,
        request.utility.cost,
        seed=seed,
    )
    assert refusal is None
    return json.loads(result)["law"]["ate"]


@pytest.mark.parametrize(
    ("rows", "columns", "seed"),
    [(100_000, 20, 5655702), (500_000, 20, 8117741), (1_000_000, 10, 1488588)],
)
def test_the_bridge_accepts_wide_tables_and_equals_the_direct_native_result(rows, columns, seed):
    request = wide_table(rows, columns, seed)
    assert len(request.data) == columns
    result = RecalcSession().execute(request, seed=3)
    # The recalculated result is the native result, bit for bit.
    assert result.law.ate == direct_native_ate(request, seed=3)
    # And it recovers the effect the table was generated with.
    assert result.law.std_error is not None
    assert abs(result.law.ate - TRUE_EFFECT) <= 6 * result.law.std_error


class Claimed:
    """A column that states a size and could not be materialised at it."""

    def __init__(self, size: int) -> None:
        self.size = size


def test_a_size_that_overflows_the_checked_byte_arithmetic_is_refused():
    # Four columns of 2**62 values: the value count fills 64 bits and the byte count passes it.
    data = {name: Claimed(2**62) for name in ("a", "b", "c", "d")}
    request = RecalcRequest(
        data=data, edges=[("a", "b")], treatment="a", outcome="b", utility=Utility(1.0)
    )
    session = RecalcSession()
    for call in (session.plan, session.execute):
        with pytest.raises(CausalValueError, match=r"recalc\.limits_exceeded.*overflows") as error:
            call(request)
        assert error.value.reason_code == "invalid_argument"
    assert not session.is_live


def test_a_declared_size_above_the_budget_is_refused_before_any_large_allocation():
    budget = 512 * 2**20
    declared = 2**40  # 8 TiB as float64, against data that holds five values
    request = RecalcRequest(
        data={"a": Claimed(declared), "b": np.zeros(5)},
        edges=[("a", "b")],
        treatment="a",
        outcome="b",
        utility=Utility(1.0),
    )
    session = RecalcSession()
    tracemalloc.start()
    try:
        for call in (session.plan, session.execute):
            with pytest.raises(CausalValueError, match="recalc.memory_budget_exceeded") as error:
                call(request, memory_limit_bytes=budget)
            assert error.value.reason_code == "invalid_argument"
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < budget
    assert not session.is_live


def test_a_column_that_holds_fewer_values_than_it_declared_is_a_size_mismatch():
    class Short:
        size = 1_000

        def __array__(self, dtype=None, copy=None):
            return np.zeros(5, dtype=dtype)

    request = RecalcRequest(
        data={"a": Short(), "b": np.zeros(1_000)},
        edges=[("a", "b")],
        treatment="a",
        outcome="b",
        utility=Utility(1.0),
    )
    with pytest.raises(CausalValueError, match=r"recalc\.declared_size_mismatch.*'a'") as error:
        RecalcSession().execute(request)
    assert error.value.reason_code == "invalid_argument"


def test_a_cancelled_run_stops_at_once_and_retains_no_partial_state():
    # A session that has run keeps exactly its previous state through a cancelled run.
    session = RecalcSession()
    first = session.execute(wide_table(2_000, 6, 11), seed=1)
    identities = session.identities
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(RecalcRefusal) as error:
        session.execute(wide_table(2_000, 6, 12), seed=1, cancel=token)
    assert (error.value.reason_code, error.value.detail) == (
        "cancelled_no_claim",
        "recalc.cancelled",
    )
    assert session.is_live and session.identities == identities
    assert session.execute(wide_table(2_000, 6, 11), seed=1).law.ate == first.law.ate

    # On the configured table size a cancelled run returns without fitting anything.
    request = wide_table(500_000, 4, 9554969)
    fresh = RecalcSession()
    started = time.perf_counter()
    with pytest.raises(RecalcRefusal) as error:
        fresh.execute(request, cancel=token)
    assert time.perf_counter() - started < 10.0
    assert error.value.detail == "recalc.cancelled"
    assert not fresh.is_live and not fresh.identities


def test_a_request_above_the_downstream_limit_names_the_limit_and_the_requested_size():
    request = wide_table(1_000, 4, 4333607)
    requested = 1_000 * 4 * 8
    limit = requested - 1
    with pytest.raises(CausalValueError) as error:
        RecalcSession().execute(request, memory_limit_bytes=limit)
    message = str(error.value)
    assert "recalc.memory_budget_exceeded" in message
    assert str(requested) in message and str(limit) in message
    assert error.value.reason_code == "invalid_argument"
