"""A transported point decision is exported and a fresh consumer re-evaluates it.

Reference. The oracle is the hand arithmetic of ``test_transported_decision``: the transported
mean of ``effect`` in ``clinic_b`` is ``2.0`` and the utilities are ``4 x - 1.5`` (treat, ``6.5``),
``0 x`` (wait, ``0``) and ``x - 5`` (hold, ``-3``), so ``treat`` is uniquely optimal. The consumer
is checked against those hand values, not against the numbers stored in the artifact: the
decision it returns is the one it computed from the carried contract, point, receipt and support
policy. Values are exact products of binary doubles, so ``1e-12`` is the tolerance. A consumer
that retained the decision identity refuses an artifact of another decision; a damaged container
refuses.
"""

from __future__ import annotations

import pytest
from antecedent import _reason_details as reason_details
from antecedent import decision
from antecedent import transported_response as tr
from antecedent.errors import CausalSerializationError, CausalTypeError
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal
from _transport_fixtures import point, prepared_analysis
from test_transported_decision import TOL, accepting, contract


@pytest.fixture(scope="module")
def prepared():
    return prepared_analysis()


def decided(prepared, value=2.0, **overrides):
    return require_success(
        contract().evaluate_transported(
            point(value, **overrides), analysis=prepared, support_policy=accepting()
        )
    )


def test_a_consumer_re_evaluates_the_carried_decision_and_matches_the_hand_values(prepared):
    original = decided(prepared)
    exported = original.export()
    consumed = require_success(
        tr.consume_transported_decision(bytes(exported), original.decision_identity)
    )
    assert consumed.decision.outcome("treat").expected_utility == pytest.approx(6.5, abs=TOL)
    assert consumed.decision.outcome("wait").expected_utility == pytest.approx(0.0, abs=TOL)
    assert consumed.decision.outcome("hold").expected_utility == pytest.approx(-3.0, abs=TOL)
    assert consumed.verdict.kind == "uniquely_optimal"
    assert consumed.verdict.selected == "treat"
    # The carried point and the transport it travelled with come back with it, and the value is
    # never presented as native target data.
    assert consumed.response.means == pytest.approx((2.0,), abs=TOL)
    assert consumed.receipt == original.receipt
    assert consumed.receipt.target_population == "clinic_b"
    assert consumed.receipt.source_populations == ("trial",)
    assert consumed.inference == "point_only"
    assert consumed.native_target_data is False
    assert consumed.decision_identity == original.decision_identity
    # The consumer needs no retained identity to re-evaluate.
    assert require_success(tr.consume_transported_decision(bytes(exported))).verdict.selected == (
        "treat"
    )


def test_another_decision_has_another_identity_and_the_consumer_refuses_the_swap(prepared):
    original = decided(prepared, 2.0)
    other = decided(prepared, 2.0, diagram="diagram-2")
    assert other.decision_identity != original.decision_identity
    with pytest.raises(decision.DecisionRefusal) as caught:
        require_success(
            tr.consume_transported_decision(bytes(other.export()), original.decision_identity)
        )
    assert_registered_refusal(caught.value)
    assert caught.value.reason_code == "decision_contract_unsatisfied"
    assert caught.value.detail == reason_details.transported_decision_artifact.IDENTITY_EXPECTED
    # A different carried value is a different decision, and it re-evaluates to its own numbers.
    moved = require_success(tr.consume_transported_decision(bytes(decided(prepared, 0.5).export())))
    assert moved.decision.outcome("treat").expected_utility == pytest.approx(0.5, abs=TOL)
    assert moved.verdict.selected == "treat"


def test_a_damaged_container_or_the_wrong_type_is_refused(prepared):
    data = bytearray(bytes(decided(prepared).export()))
    for position in (len(data) // 2, len(data) - 8):
        damaged = bytearray(data)
        damaged[position] ^= 0x01
        with pytest.raises((decision.DecisionRefusal, CausalSerializationError)):
            require_success(tr.consume_transported_decision(bytes(damaged)))
    with pytest.raises((decision.DecisionRefusal, CausalSerializationError)):
        require_success(tr.consume_transported_decision(bytes(data[: len(data) // 2])))
    with pytest.raises(CausalTypeError, match="artifact is bytes"):
        tr.consume_transported_decision("bytes")  # type: ignore[arg-type]
