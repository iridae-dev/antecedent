"""Actual decision engines retain typed readings and explicit scientific refusals."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import composition, decision, decision_robust, inverse_query
from antecedent import decision_reliability as reliability
from antecedent._artifact_protocol import ExportedArtifact
from antecedent.errors import CausalTypeError
from antecedent.outcomes import Outcome, require_success

from test_composition import joint_contract, joint_input
from test_decision import _contract, _source
from test_decision_reliability import _draws
from test_decision_robust import _claims
from test_decision_robust import _contract as robust_contract
from test_inverse_query import _law, _query


def test_native_decision_outcome_retains_joint_source_and_replays_independently(monkeypatch):
    contract, source = _contract(), _source()
    outcome = contract.evaluate(source)
    assert isinstance(outcome, Outcome) and outcome.accepted
    result = require_success(outcome)
    assert result.source is source
    assert result.selected == ("safe",)
    assert result.actions[0].expected_utility == pytest.approx(2.0)
    assert result.evpi == pytest.approx(0.5)
    assert outcome.answer.structured["selected"] == ("safe",)

    def forbidden(*args, **kwargs):
        raise AssertionError("inspection cannot export a decision or its source")

    with monkeypatch.context() as patch:
        patch.setattr(type(result), "export", forbidden)
        patch.setattr(type(source), "export", forbidden)
        assert outcome.inspect().answer == outcome.answer
        assert "Decision" in repr(result)
    artifact = outcome.export()
    assert isinstance(artifact, ExportedArtifact)
    replay = decision.replay(artifact, contract=contract, source=source)
    assert isinstance(replay, Outcome) and replay.accepted
    restored = require_success(replay)
    assert restored.selected == result.selected
    assert restored.evpi == result.evpi
    assert restored.acceptance.verified and restored.acceptance.replayable
    assert restored.acceptance.identity_bound
    assert not restored.acceptance.resumable


def test_native_decision_refusal_is_a_reading_and_reraises_the_same_original():
    outcome = _contract().evaluate(_source(alignment="independent_marginals"))
    assert isinstance(outcome, Outcome) and not outcome.accepted
    assert outcome.answer.kind == "refused"
    assert outcome.refusal.code == "joint_law_required"
    original = outcome.refusal._error
    assert isinstance(original, decision.DecisionRefusal)
    with pytest.raises(decision.DecisionRefusal) as raised:
        require_success(outcome)
    assert raised.value is original
    with pytest.raises(CausalTypeError):
        _contract().evaluate(object())


def test_robust_decision_keeps_per_structure_claims_and_declared_support():
    contract = robust_contract()
    claims = _claims((5.0, 2.0), (4.0, 1.0))
    outcome = decision_robust.finite_scenarios(contract, claims)
    assert isinstance(outcome, Outcome) and outcome.accepted
    result = require_success(outcome)
    assert result.verdict.action == "A"
    assert result.verdict.kind == "structurally_robust"
    assert outcome.answer.structured["selected"] == "A"
    restored = require_success(
        decision_robust.replay(result.export(), contract=contract, claims=claims)
    )
    assert restored.verdict == result.verdict
    assert restored.acceptance.verified and restored.acceptance.identity_bound


def test_inverse_outcome_retains_finite_grid_feasibility_and_actual_consumer():
    query = _query(inverse_query.target_mean(0.5))
    outcome = query.evaluate(_law())
    assert isinstance(outcome, Outcome) and outcome.accepted
    result = require_success(outcome)
    assert result.feasible_actions == ("a1", "a2")
    assert result.selected == "a1" and result.selection_certified
    assert outcome.answer.structured["feasible_actions"] == ("a1", "a2")
    restored = require_success(
        inverse_query.consume(result.export(), expected_identity=result.identity)
    )
    assert restored.to_dict() == result.to_dict()
    assert restored.answer == result.answer
    assert restored.acceptance.verified and restored.acceptance.replayable
    assert restored.acceptance.identity_bound


def test_composed_functional_outcome_keeps_native_scalar_and_dependency_source():
    source = joint_input("enumerated")
    outcome = composition.evaluate_functional(
        joint_contract(), "wait", composition.Functional.expectation(), source
    )
    assert isinstance(outcome, Outcome) and outcome.accepted
    result = require_success(outcome)
    assert result.value == pytest.approx(2.0)
    assert outcome.answer.kind == "point" and outcome.answer.value == pytest.approx(2.0)
    other = joint_input("other", snapshot="different-snapshot")
    relation = composition.PairRelation(
        "enumerated", "other", composition.EvidenceRelation.unknown()
    )
    refused = composition.check_composition(
        [source, other], [relation], operation="statistical_pooling"
    )
    assert isinstance(refused, Outcome) and not refused.accepted
    assert refused.refusal.detail == "composition_boundary.unknown_dependence_is_not_independence"


def test_reliability_outcome_preserves_enumerated_probabilities_and_closed_publication():
    draws = _draws(
        {
            "A": np.array([3.0, 3.0, 2.0, 2.0]),
            "B": np.array([2.0, 1.0, 3.0, 2.0]),
            "C": np.array([1.0, 2.0, 1.0, 2.0]),
        }
    )
    outcome = reliability.rank(draws, reliability.RankingRequest(baseline="B"))
    assert isinstance(outcome, Outcome) and outcome.accepted
    report = require_success(outcome)
    assert {key: report.probability_best(key) for key in "ABC"} == pytest.approx(
        {"A": 7 / 12, "B": 1 / 3, "C": 1 / 12}, abs=1e-12
    )
    assert report.label["calibration"] == "uncalibrated"
    assessment = reliability.assess(
        _draws(
            {
                "A": np.tile([3.0, 3.0, 2.0, 2.0], 25),
                "B": np.tile([2.0, 1.0, 3.0, 2.0], 25),
                "C": np.tile([1.0, 2.0, 1.0, 2.0], 25),
            }
        )
    )
    assert isinstance(assessment, Outcome) and assessment.accepted
    published = require_success(assessment).publish("ranking_probability")
    assert isinstance(published, Outcome) and not published.accepted
    assert published.refusal.code == "cell_not_licensed"


@pytest.mark.parametrize("operation", ["rank", "assess"])
def test_reliability_execution_owns_actual_workspace_and_preserves_operational_errors(operation):
    from antecedent import CancellationToken, Execution, ResourceLimits
    from antecedent.errors import CausalCancelledError, CausalResourceError

    draws = _draws(
        {
            "A": np.tile([3.0, 3.0, 2.0, 2.0], 64),
            "B": np.tile([2.0, 1.0, 3.0, 2.0], 64),
            "C": np.tile([1.0, 2.0, 1.0, 2.0], 64),
        }
    )
    execute = getattr(reliability, operation)
    limited = Execution(limits=ResourceLimits(memory_bytes=128))
    with pytest.raises(CausalResourceError):
        execute(draws, execution=limited)
    assert limited._handle.available_memory_bytes == 128
    # Reusing the exact exhausted caller must fail identically after its leases return.
    with pytest.raises(CausalResourceError):
        execute(draws, execution=limited)
    assert limited._handle.available_memory_bytes == 128

    token = CancellationToken()
    token.cancel()
    cancelled = Execution(cancel=token, limits=ResourceLimits(memory_bytes=100_000))
    with pytest.raises(CausalCancelledError):
        execute(draws, execution=cancelled)
    assert cancelled._handle.available_memory_bytes == 100_000

    funded = Execution(seed=11, limits=ResourceLimits(memory_bytes=1_000_000))
    result = require_success(execute(draws, execution=funded))
    expected = require_success(execute(draws, execution=Execution(seed=11)))
    assert (result.report.raw if operation == "assess" else result.raw) == (
        expected.report.raw if operation == "assess" else expected.raw
    )
    assert funded._handle.available_memory_bytes == 1_000_000
    if operation == "assess":
        assert bytes(result.export()) == bytes(expected.export())
        artifact = result.export()
        with pytest.raises(CausalResourceError):
            reliability.consume_decision_reliability_artifact(artifact, execution=limited)
        assert limited._handle.available_memory_bytes == 128
