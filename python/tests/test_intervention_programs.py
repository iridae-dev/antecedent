"""Closed-form intervention programs from Python run the library's one executor.

Python builds program text; the library validates, compiles and executes it. The tests compare
the draws with hand computations on identical noise, so a divergence between the routes (or
between Rust and Python) would show as a numerical difference, and pin the program identity to
the digest the Rust test of the same program asserts.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalValueError
from antecedent.intervention_programs import (
    assign,
    counterfactual_policy,
    decode_policy_program,
    encode_policy_program,
    generated,
    lag,
    maximum,
    policy_program_identity,
    sample_policy,
    sample_temporal_policy,
    transform,
    var,
    where,
)

N_DRAWS = 512
NAMES = ["price", "demand"]
# The digest of this program as asserted by the Rust test
# `the_price_floor_program_has_a_pinned_identity`.
PRICE_FLOOR_IDENTITY = "4828012ed5bdb86f66847911617623031457a3373151fb18a2b4c54d3a051d04"
# The checksum of every drawn bit asserted by the Rust test
# `a_price_floor_draws_the_same_bits_from_rust_as_from_python` for the data built below.
PRICE_FLOOR_DRAW_CHECKSUM = 15_163_479_412_849_571_691


def _floor():
    return transform("price", maximum(10, 0.85 * generated()))


@pytest.fixture(scope="module")
def gcm():
    rng = np.random.default_rng(23)
    price = 11.0 + rng.normal(scale=2.0, size=600)
    demand = 50.0 - 0.8 * price + rng.normal(scale=1.0, size=600)
    return antecedent.model.fit_gcm(NAMES, [price, demand], [("price", "demand")]), price


def _column(result, name):
    return result.draws[NAMES.index(name)]


def test_a_price_floor_is_the_hand_applied_policy_on_the_same_noise(gcm):
    model, _ = gcm
    observed = _column(model.sample_do({}, N_DRAWS, seed=7), "price")
    policy = _column(sample_policy(model, [_floor()], N_DRAWS, seed=7), "price")
    np.testing.assert_array_equal(policy, np.maximum(10.0, 0.85 * observed))
    assert (policy >= 10.0).all() and (observed * 0.85 < 10.0).any()


def test_a_conditional_branch_selects_per_row(gcm):
    model, _ = gcm
    observed = _column(model.sample_do({}, N_DRAWS, seed=8), "price")
    branch = transform(
        "price",
        where(generated() > 12, 10 + 0.5 * generated(), maximum(10, 0.85 * generated())),
    )
    policy = _column(sample_policy(model, [branch], N_DRAWS, seed=8), "price")
    expected = np.where(observed > 12.0, 10.0 + 0.5 * observed, np.maximum(10.0, 0.85 * observed))
    np.testing.assert_array_equal(policy, expected)


def test_counterfactual_execution_applies_the_policy_to_each_rows_own_noise(gcm):
    model, price = gcm
    outcomes = counterfactual_policy(model, [_floor()], ["price", "demand"], seed=1)
    np.testing.assert_allclose(outcomes["price"], np.maximum(10.0, 0.85 * price), atol=1e-9)
    assert outcomes["demand"].shape == price.shape


def test_a_lagged_policy_matches_the_hand_recursion_over_time(gcm):
    model, _ = gcm
    n_rows, n_steps = 8, 10
    start = np.full(n_rows, 40.0)
    program = assign("price", maximum(10, 0.85 * lag("price", 1)))
    steps = sample_temporal_policy(
        model,
        [program],
        n_rows=n_rows,
        n_steps=n_steps,
        window=1,
        history=[{"price": start, "demand": np.zeros(n_rows)}],
        seed=3,
    )
    price = start
    for step in steps:
        price = np.maximum(10.0, 0.85 * price)
        np.testing.assert_array_equal(_column(step, "price"), price)


def test_a_look_back_before_the_first_step_is_refused(gcm):
    model, _ = gcm
    program = assign("price", lag("price", 2) + 1)
    with pytest.raises(CausalError) as raised:
        sample_temporal_policy(model, [program], n_rows=4, n_steps=3, window=2, seed=1)
    assert raised.value.reason_code == "policy_history_unavailable"
    with pytest.raises(CausalError) as raised:
        sample_policy(model, [program], 8, seed=1)
    assert raised.value.reason_code == "policy_history_unavailable"


def test_a_malformed_program_is_a_value_error(gcm):
    model, _ = gcm
    with pytest.raises(CausalValueError):
        sample_policy(model, [transform("supply", generated())], 8, seed=1)
    with pytest.raises(CausalValueError):
        generated() + True


def test_python_draws_the_same_bits_as_rust_for_the_same_model_policy_and_seed():
    rows = 600
    price = np.array([11.0 + (float((i * 37) % 101) - 50.0) * 0.04 for i in range(rows)])
    demand = np.array(
        [50.0 - 0.8 * p + (float((i * 13) % 23) - 11.0) * 0.05 for i, p in enumerate(price)]
    )
    model = antecedent.model.fit_gcm(NAMES, [price, demand], [("price", "demand")])
    draws = sample_policy(model, [_floor()], 512, seed=7).draws
    checksum = int(np.ascontiguousarray(draws).view(np.uint64).sum())
    assert checksum == PRICE_FLOOR_DRAW_CHECKSUM


def test_the_identity_is_the_one_rust_pins_and_moves_with_the_expression():
    assert policy_program_identity(NAMES, [_floor()]) == PRICE_FLOOR_IDENTITY
    other = transform("price", maximum(10, 0.9 * generated()))
    assert policy_program_identity(NAMES, [other]) != PRICE_FLOOR_IDENTITY
    kind = assign("price", maximum(10, 0.85 * var("demand")))
    assert policy_program_identity(NAMES, [kind]) != PRICE_FLOOR_IDENTITY


def test_an_encoded_program_round_trips_and_a_tampered_one_is_refused():
    programs = [_floor(), assign("demand", maximum(0, 60 - 0.8 * lag("price", 1)))]
    data = encode_policy_program(NAMES, programs)
    assert decode_policy_program(NAMES, data) == tuple(programs)
    assert encode_policy_program(NAMES, decode_policy_program(NAMES, data)) == data
    tampered = bytearray(data)
    tampered[-3] ^= 0x01
    with pytest.raises(CausalError):
        decode_policy_program(NAMES, bytes(tampered))
