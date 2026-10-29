"""A transported response reaches the decision contract through the Python surface, bound to the
causal contract, under the contract's admissibility rules, with the transport receipt carried into
the result and a refusal for every request a transported point cannot answer.

Oracle (the Python spelling of ``crates/antecedent/tests/transported_decision_facade.rs``). The
transported mean of ``effect`` in ``clinic_b`` is ``2.0``. The utility of ``treat`` is
``4 x - 1.5`` so its expected utility is ``6.5``; ``wait`` is ``0 x = 0`` and ``hold`` is
``x - 5 = -3``.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import decision, decision_robust
from antecedent import transported_response as tr
from antecedent.composition import SupportPolicy
from antecedent.errors import CausalTypeError
from antecedent.joint_distribution import DistributionIdentity, JointDistributionArtifact
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal
from _transport_fixtures import (
    effect,
    point,
    prepared_analysis,
    transport_declaration,
    transported_law,
)

TOL = 1e-12


def contract() -> decision.Contract:
    def action(identifier: str, utility: decision.Expr) -> decision.Action:
        return decision.Action(identifier, inputs=(effect(),), utility=utility)

    return decision.Contract(
        actions=(
            action("treat", decision.x(0) * 4.0 - 1.5),
            action("wait", decision.x(0) * 0.0),
            action("hold", decision.x(0) - 5.0),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="clinic_b",
    )


@pytest.fixture(scope="module")
def prepared():
    return prepared_analysis()


def accepting() -> SupportPolicy:
    return SupportPolicy.require_all("extrapolative")


def admissible(**rules) -> decision_robust.AdmissibleContract:
    return decision_robust.admissible_contract(
        contract(), decision_robust.AdmissibilityRules(**rules)
    )


def refusal_of(call) -> decision.DecisionRefusal:
    with pytest.raises(decision.DecisionRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


def test_a_transported_point_decides_by_the_utility_of_the_mean_and_names_its_transport(
    prepared,
) -> None:
    decided = require_success(
        contract().evaluate_transported(point(2.0), analysis=prepared, support_policy=accepting())
    )
    assert decided.decision.outcome("treat").expected_utility == pytest.approx(6.5, abs=TOL)
    assert decided.decision.outcome("wait").expected_utility == pytest.approx(0.0, abs=TOL)
    assert decided.decision.outcome("hold").expected_utility == pytest.approx(-3.0, abs=TOL)
    assert decided.verdict.kind == "uniquely_optimal"
    assert decided.verdict.selected == "treat"
    # The transport travels beside every number; the value is never native target data.
    assert decided.receipt.target_population == "clinic_b"
    assert decided.receipt.source_populations == ("trial",)
    assert decided.receipt.support == "extrapolative"
    assert decided.inference == "point_only"
    assert decided.native_target_data is False
    assert any("not native target data" in line for line in decided.assumptions)
    assert "not native target data" in decided.explain()

    # Repeating the request reproduces the identity; another value or another transport at the
    # same shape changes it.
    again = require_success(
        contract().evaluate_transported(point(2.0), analysis=prepared, support_policy=accepting())
    )
    assert again.identity == decided.identity
    assert again.bound_identity == decided.bound_identity
    moved = require_success(
        contract().evaluate_transported(
            point(2.0, diagram="diagram-2"), analysis=prepared, support_policy=accepting()
        )
    )
    assert moved.identity != decided.identity
    assert moved.bound_identity == decided.bound_identity

    # An analysis result that retains its study binds the same identities.
    assert decided.causal_target == again.causal_target
    assert decided.causal_identification == again.causal_identification

    inspected = decided.inspect()
    assert inspected.family == "TransportedDecision"


def test_declared_exclusions_remove_actions_and_are_part_of_the_bound_decision(prepared) -> None:
    exclusions = decision_robust.DeclaredExclusion
    plain = admissible()
    excluding = admissible(declared_exclusions=(exclusions("hold", "not offered"),))
    assert plain.identity != excluding.identity

    full = require_success(
        plain.evaluate_transported(point(2.0), analysis=prepared, support_policy=accepting())
    )
    assert len(full.outcomes) == 3
    narrowed = require_success(
        excluding.evaluate_transported(point(2.0), analysis=prepared, support_policy=accepting())
    )
    assert [o.id for o in narrowed.outcomes] == ["treat", "wait"]
    assert narrowed.verdict.selected == "treat"
    assert narrowed.identity != full.identity

    # Exclusions that leave nothing to compare refuse with their own detail.
    nothing = admissible(
        declared_exclusions=(exclusions("hold", "not offered"), exclusions("wait", "not offered"))
    )
    refusal = refusal_of(
        lambda: require_success(
            nothing.evaluate_transported(point(2.0), analysis=prepared, support_policy=accepting())
        )
    )
    assert refusal.reason_code == "decision_contract_unsatisfied"
    assert refusal.detail == "transported_decision.exclusions_leave_no_comparison"


def test_a_support_rule_stronger_than_the_transported_label_refuses(prepared) -> None:
    strict = admissible(default_weakest_support="supported")
    refusal = refusal_of(
        lambda: require_success(
            strict.evaluate_transported(point(2.0), analysis=prepared, support_policy=accepting())
        )
    )
    assert refusal.reason_code == "transport_support_failure"
    assert refusal.detail == "transported_decision.support_rule_unmet"
    assert refusal.supplied == "extrapolative"
    assert refusal.remedy


def test_an_uncertainty_requirement_a_point_cannot_meet_refuses(prepared) -> None:
    for requirement in ("credible", "structural_envelope"):
        refusal = refusal_of(
            lambda requirement=requirement: require_success(
                admissible(uncertainty=requirement).evaluate_transported(
                    point(2.0), analysis=prepared, support_policy=accepting()
                )
            )
        )
        assert refusal.reason_code == "decision_contract_unsatisfied"
        assert refusal.detail == "transported_decision.uncertainty_requirement"
    # A point requirement is met by a transported point.
    met = require_success(
        admissible(uncertainty="point_only").evaluate_transported(
            point(2.0), analysis=prepared, support_policy=accepting()
        )
    )
    assert met.verdict.selected == "treat"


def test_a_point_refuses_laws_and_intervals_and_the_default_policy_refuses_the_label(
    prepared,
) -> None:
    response = point(2.0)
    law = refusal_of(
        lambda: require_success(
            contract().evaluate_transported(
                response, analysis=prepared, demand="law", support_policy=accepting()
            )
        )
    )
    assert law.reason_code == "decision_contract_unsatisfied"
    assert law.detail == "transported_decision.law_not_supplied"
    assert law.remedy

    interval = refusal_of(
        lambda: require_success(
            contract().evaluate_transported(
                response, analysis=prepared, demand="interval", support_policy=accepting()
            )
        )
    )
    assert interval.reason_code == "cell_not_licensed"

    # The default policy admits only `supported` coordinates: accepting a transported claim is the
    # consumer's explicit choice.
    default = refusal_of(
        lambda: require_success(contract().evaluate_transported(response, analysis=prepared))
    )
    assert default.reason_code == "cell_not_licensed"
    assert default.detail == "transported_decision.transported_not_accepted"

    with pytest.raises(tr.TransportedResponseRefusal) as closed:
        response.interval()
    assert closed.value.reason_code == "cell_not_licensed"


def test_a_response_is_declared_with_a_complete_transport_identity() -> None:
    with pytest.raises(tr.TransportedResponseRefusal) as same:
        tr.TransportedResponse.declared_point(
            [effect("trial")], [2.0], transport_declaration("trial", source="trial")
        )
    assert_registered_refusal(same.value)
    assert same.value.detail == "transported_decision.transport_identity_incomplete"

    # A support label stronger than extrapolative is refused at the declaration.
    with pytest.raises(tr.TransportedResponseRefusal) as inflated:
        tr.TransportedResponse.declared_point(
            [effect()],
            [2.0],
            tr.TransportDeclaration(
                route="declared",
                source_identity="transport-1",
                source_populations=("trial",),
                target_population="clinic_b",
                transport_identities=("diagram-1",),
                assumptions=("selection acts on age only",),
                provider_id="p",
                snapshot_id="s",
                causal_contract_id="c",
                support="supported",
            ),
        )
    assert inflated.value.detail == "transported_decision.support_label_inflated"

    # A coordinate must be stated in the target population.
    with pytest.raises(tr.TransportedResponseRefusal) as outside:
        tr.TransportedResponse.declared_point(
            [effect("elsewhere")], [2.0], transport_declaration("clinic_b")
        )
    assert outside.value.detail == "transported_decision.coordinate_not_in_target"

    # A route with a typed producer cannot be declared by hand.
    with pytest.raises(tr.TransportedResponseRefusal) as typed:
        tr.TransportedResponse.declared_point(
            [effect()], [2.0], transport_declaration("clinic_b", route="policy")
        )
    assert typed.value.detail == "transported_decision.route_needs_typed_source"


def test_a_law_carries_its_receipt_and_a_native_label_is_refused() -> None:
    response = transported_law()
    assert response.label == "law"
    assert response.means is None
    assert response.receipt.representation == "law"
    assert response.native_target_data is False

    # A law labelled exact is a calibration claim a transport never makes.
    exact = JointDistributionArtifact(
        DistributionIdentity(
            semantic="interventional_predictive",
            quantities=(effect("clinic_b", "y"),),
            alignment="joint",
            source_id="transport",
            provider_id="transport-provider",
            rng_id="deterministic_exact",
            snapshot_id="snapshot-1",
            causal_contract_id="contract-1",
        ),
        np.array([[1.0], [0.0]], dtype=np.float64),
        calibration="exact",
    )
    with pytest.raises(tr.TransportedResponseRefusal) as inflated:
        tr.TransportedResponse.declared_law(exact, transport_declaration("clinic_b"))
    assert inflated.value.detail == "transported_decision.calibration_label_inflated"


def test_only_a_transported_response_is_a_decision_source_of_this_route(prepared) -> None:
    with pytest.raises(CausalTypeError, match=r"response must be a TransportedResponse"):
        require_success(contract().evaluate_transported(object(), analysis=prepared))  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError, match=r"Contract\.evaluate needs a JointDistributionArtifact \(aligned draws\), a BoundExternalClaim or MeanSou"):
        require_success(contract().evaluate(point(2.0)))  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError, match=r"analysis must be a prepared analysis or an analysis result that retains one"):
        require_success(contract().evaluate_transported(point(2.0), analysis=object()))
