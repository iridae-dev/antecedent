"""The whole-path uncertainty account of a cross-world execution through the Python surface:
derived from the sealed request and result under a declared unit sampling, bound to the execution
it stands on, and re-derived (never trusted) by a fresh consumer.

The response-type model (the Python spelling of
``crates/antecedent/tests/whole_path_account_facade.rs``): ``X = U_X`` with ``P(U_X = 1) = 0.4``;
``Y = f(X, U_Y)`` with ``U_Y`` over four response types (never, complier, defier, always) with
masses ``q = [0.2, 0.5, 0.1, 0.2]``. For ``P(Y_{x0} = 0 | evidence)`` over three units:
``case (X = 1, Y = 1)``: ``5/7``, variance ``10/49``; ``clean (X = 0, Y = 0)``: ``1``, variance
``0``; ``open``: ``0.7``, variance ``0.21``. The mean is ``169/210``; the realized-outcome variance
is ``2029/44100``; the sampling variance of the mean is ``2526/264600``.
"""

from __future__ import annotations

import pytest
from antecedent import cross_world_execution as cwx
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal

TOL = 1e-12


def scm() -> cwx.FiniteSCM:
    return cwx.FiniteSCM(
        population="pop-a",
        history="regime-1",
        exogenous=(
            cwx.ExogenousTerm("u_x", (0.6, 0.4)),
            cwx.ExogenousTerm("u_y", (0.2, 0.5, 0.1, 0.2)),
        ),
        variables=(
            cwx.FiniteVariable("x", 2, (), "u_x", (0, 1)),
            cwx.FiniteVariable("y", 2, ("x",), "u_y", (0, 0, 0, 1, 1, 0, 1, 1)),
        ),
    )


def request(units) -> cwx.CrossWorldRequest:
    return cwx.CrossWorldRequest(
        scm=scm(),
        units=tuple(units),
        worlds=(cwx.World("x0", {"x": 0}), cwx.World("x1", {"x": 1})),
        quantity=cwx.probability("x0", "y", 0),
    )


def three_units() -> cwx.CrossWorldRequest:
    return request(
        [
            cwx.FactualUnit("case", {"x": 1, "y": 1}),
            cwx.FactualUnit("clean", {"x": 0, "y": 0}),
            cwx.FactualUnit("open"),
        ]
    )


def test_the_account_is_the_enumerated_spread_bound_to_its_execution() -> None:
    sealed = require_success(cwx.seal(three_units()))
    account = require_success(sealed.whole_path_uncertainty("independent_sample"))
    assert account.label == "uncalibrated_spread"
    assert account.unit_sampling == "independent_sample"
    assert account.value == pytest.approx(169 / 210, abs=TOL)
    assert account.value == sealed.value
    assert account.realized_mean_predictive_variance == pytest.approx(2029 / 44100, abs=TOL)
    assert account.unit_mean_sampling_variance == pytest.approx(2526 / 264600, abs=TOL)
    assert account.realized_mean_standard_error == pytest.approx((2029 / 44100) ** 0.5, abs=TOL)
    assert account.unit_mean_standard_error is not None
    assert account.source("structural").treatment["kind"] == "premise"
    assert account.execution_receipt_digest == sealed.identity["receipt_digest"]

    # Hand values per unit: 5/7 with variance 10/49, 1 with 0, 0.7 with 0.21.
    spread = {u.unit: u for u in account.units}
    assert spread["case"].value == pytest.approx(5 / 7, abs=TOL)
    assert spread["case"].predictive_variance == pytest.approx(10 / 49, abs=TOL)
    assert spread["clean"].value == pytest.approx(1.0, abs=TOL)
    assert spread["clean"].predictive_variance == pytest.approx(0.0, abs=TOL)
    assert spread["open"].value == pytest.approx(0.7, abs=TOL)
    assert spread["open"].predictive_variance == pytest.approx(0.21, abs=TOL)

    assert [s.source for s in account.sources] == [
        "structural",
        "posterior",
        "sampling",
        "numerical",
    ]
    assert account.inspect().family == "WholePathAccount"

    # The reading stays closed: a spread is a labelled diagnostic, never an interval.
    for closed in (sealed.interval, sealed.posterior, account.interval, account.posterior):
        with pytest.raises(cwx.CrossWorldExecutionRefusal) as refused:
            closed()
        assert refused.value.reason_code == "cell_not_licensed"


def test_a_fresh_consumer_derives_the_same_account_from_the_artifact() -> None:
    sealed = require_success(cwx.seal(three_units()))
    account = require_success(sealed.whole_path_uncertainty("fixed_units"))
    assert account.unit_mean_sampling_variance is None
    assert account.unit_mean_standard_error is None

    data = sealed.to_bytes()
    consumed = require_success(
        cwx.consume_cross_world_execution_artifact(data, expected_identity=sealed.identity)
    )
    rebuilt = cwx.CrossWorldExecution(data, consumed)
    again = require_success(rebuilt.whole_path_uncertainty("fixed_units"))
    assert again.identity == account.identity
    assert again.value == account.value
    account.verify_against(rebuilt)

    # The declared unit sampling is part of the account: another declaration is another
    # account, not a relabelled one.
    sampled = require_success(rebuilt.whole_path_uncertainty("independent_sample"))
    assert sampled.identity != account.identity


def test_an_account_of_other_units_is_refused_rather_than_annotated() -> None:
    sealed = require_success(cwx.seal(three_units()))
    account = require_success(sealed.whole_path_uncertainty("independent_sample"))
    # New factual evidence is new units and a new execution: the earlier account does not
    # describe it.
    other = require_success(
        cwx.seal(
            request(
                [
                    cwx.FactualUnit("case", {"x": 1, "y": 1}),
                    cwx.FactualUnit("clean", {"x": 0, "y": 0}),
                ]
            )
        )
    )
    with pytest.raises(cwx.CrossWorldExecutionRefusal) as mismatch:
        account.verify_against(other)
    assert_registered_refusal(mismatch.value)
    assert mismatch.value.detail == "cross_world_uncertainty.account_mismatch"
    assert mismatch.value.remedy


def test_an_undeclared_unit_sampling_is_refused() -> None:
    sealed = require_success(cwx.seal(three_units()))
    with pytest.raises(cwx.CrossWorldExecutionRefusal) as undeclared:
        require_success(sealed.whole_path_uncertainty("undeclared"))  # type: ignore[arg-type]
    assert_registered_refusal(undeclared.value)
    assert undeclared.value.reason_code == "sampling_dependence_unknown"
    with pytest.raises(CausalValueError):
        require_success(sealed.whole_path_uncertainty("whole_population"))  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        require_success(sealed.whole_path_uncertainty("fixed_units")).verify_against(object())  # type: ignore[arg-type]
