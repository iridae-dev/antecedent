"""2.3 C4: one public multi-source story, from graph to a consumed composition bundle.

The story uses only public Python modules. An explicit graph ``x -> a``, ``x -> y``, ``a -> y``
identifies the dose response ``E[y | do(a = d)]`` (mmHg) for ``d in {0, 1, 2}`` mg in the
``target`` population. Three sources answer it, each bound to the same program
(``program_claims.ProgramBinding``): a native mean response ``N``, an attested point-mean study
``E1`` (``lab-1`` / ``snap-e1``) and a second study ``E2`` (``lab-2`` / ``snap-e2``) that is
bound as point means and also carries two equally likely aligned joint draws.

Hand derivation (nothing below is read from the code under test)
-----------------------------------------------------------------
Mean grids ``(m0, m1, m2) = E[y | do(a = 0, 1, 2)]``::

    N  = (2, 4, 6)     dose 2 is outside empirical support, so it is withheld
    E1 = (1, 3, 5.5)
    E2 = (2, 3.5, 5)   the means of its two joint rows: (2+2)/2, (6+1)/2, (3+7)/2

Three actions: ``wait`` is worth ``m0``, ``treat`` is ``m1 - 1`` and ``extend`` is ``m2 - 3``::

    N alone:   wait 2, treat 4 - 1 = 3, extend unsupported (no coordinate)   -> treat
    E1 alone:  wait 1, treat 3 - 1 = 2, extend 5.5 - 3 = 2.5                 -> extend
    N then E1: wait 2 (N), treat 3 (N), extend 2.5 (E1; N has none)           -> treat, 3
    E2 means:  wait 2, treat 3.5 - 1 = 2.5, extend 5 - 3 = 2

E2's joint law has two equally likely rows ``(y0, y1, y2, theta)``: ``(2, 6, 3, 1/4)`` and
``(2, 1, 7, 3/4)``. The per-row utilities are::

    row 1: wait 2, treat 6 - 1 = 5, extend 3 - 3 = 0
    row 2: wait 2, treat 1 - 1 = 0, extend 7 - 3 = 4

so ``E[U] = (2, (5 + 0)/2, (0 + 4)/2) = (2, 2.5, 2)`` and ``treat`` is uniquely optimal. The best
action per row is 5 and 4, so ``E[max U] = 4.5`` and ``EVPI = 4.5 - 2.5 = 2``. Expected regrets are
``wait (3 + 2)/2 = 2.5``, ``treat (0 + 4)/2 = 2`` and ``extend (5 + 0)/2 = 2.5``. ``P(U_treat >= 5)
= 1/2``; the 3/4-quantile of ``U_treat`` over ``{0, 5}`` is 5 (``F(0) = 1/2 < 3/4``), of ``U_extend``
over ``{0, 4}`` is 4 and of ``U_wait`` is 2. Dropping ``extend`` (masked unsupported) leaves
``E[max(2, U_treat)] = (5 + 2)/2 = 3.5`` and ``EVPI = 3.5 - 2.5 = 1``.

Study ranking. The follow-up decision bets on the response rate ``theta`` (the law's own state
column, prior draws ``{1/4, 3/4}``): ``abstain`` pays 0 and ``bet`` pays ``theta - 1/2``. Both have
prior value 0 and ``EVPI = E[max(0, theta - 1/2)] = (0 + 1/4)/2 = 1/8``. A study of ``n = 2``
Bernoulli trials has ``P(k | theta = 1/4) = (9, 6, 1)/16`` and ``P(k | theta = 3/4) = (1, 6, 9)/16``,
so ``P(k) = (10, 12, 10)/32`` and ``E[theta | k] = (0.3, 0.5, 0.7)``; only ``k = 2`` makes the bet
worth taking, ``EVSI = 10/32 * (0.7 - 0.5) = 1/16``. A native binomial signal and an equivalent
attested posterior (posterior ``(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)``) share that value; with study
costs 0.02 and 0.05 in utility units the net values are ``1/16 - 0.02 = 0.0425`` and ``1/16 - 0.05
= 0.0125``.

Declared evidence: ``E1`` and ``E2`` are independent; ``N`` and ``E2`` both used registry
``registry-7``.

The native curve executes the original checked kernel engine on crossed observed data
Y=2+2A+X/2, with empirical A support [0,1] and symmetric X. Its causal means are
(2,4,6); the unsupported dose2 is never consumed. The fixed bandwidth2.1 permits
original-engine evaluation while preserving its support flags. Finite expected values
above are independent algebra, not result reconstruction.

The native claim enters composition through its actual issued execution authority.
Reconstructing its means or metadata grants no native authority. This response retains
no draws, so every joint-law answer here comes from the external ``E2`` law.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import antecedent as ac
import numpy as np
import pytest
from antecedent import composition as comp
from antecedent import composition_bundle as cb
from antecedent import decision, external, program_claims
from antecedent import inverse_query as iq
from antecedent.extensibility import ProviderTrust

import _c4_fixtures as fx
from _refusal import assert_registered_refusal

SNAPSHOT = fx.SNAPSHOT
NATIVE_DIGEST = fx.NATIVE_DIGEST
RELATION = fx.RELATION
_composed_builder = fx.composed_builder
_point_only_builder = fx.point_only_builder
IDS = ("wait", "treat", "extend")

CONSUMER = textwrap.dedent(
    """
    import json, sys
    from antecedent import composition_bundle as cb

    data = open(sys.argv[1], "rb").read()
    wanted = json.loads(sys.argv[3])
    supplied = None
    if len(sys.argv) > 4:
        supplied = cb.SuppliedSources().with_data(sys.argv[4], sys.argv[5])
    consumed = cb.consume_bundle(data, expected_identity=sys.argv[2], supplied=supplied)
    print(json.dumps({
        "all_verified": consumed.all_verified,
        "label": consumed.claim_label,
        "nodes": {n.id: n.verified for n in consumed.nodes},
        "kinds": {n.id: n.kind for n in consumed.nodes},
        "values": {f"{node}:{key}": consumed.value(node, key) for node, key in wanted},
    }))
    """
)


# ------------------------------------------------------------------ sources bind to a program


def test_c4_both_studies_and_the_native_claim_bind_to_the_same_program() -> None:
    program = fx.program()
    assert fx.spec().contract_id == program.identity
    assert len(program.identity) == 64
    expected = tuple(fx.coordinate(dose) for dose in fx.GRID)
    e1, e2 = fx.e1_claim(), fx.e2_claim()
    for claim in (e1, e2):
        assert claim.native is False
        assert claim.trust is ProviderTrust.EXTERNALLY_ATTESTED
        assert claim.identity_fields["causal_contract_id"] == program.identity
        assert claim.support == fx.SUPPORTED
        assert claim.quantities == expected
    e1_fields, e2_fields = e1.identity_fields, e2.identity_fields
    assert (e1_fields["provider_id"], e1_fields["snapshot_id"]) == ("lab-1", "snap-e1")
    assert (e2_fields["provider_id"], e2_fields["snapshot_id"]) == ("lab-2", "snap-e2")
    assert list(e1.values) == list(fx.E1_MEANS)
    assert list(e2.values) == list(fx.E2_MEANS)

    native = fx.native_claim()
    assert native.trust is ProviderTrust.NATIVE_LICENSED
    assert native.program_identity == program.identity
    assert native.coordinates == expected
    assert native.means == pytest.approx(fx.NATIVE_MEANS, abs=1e-9)
    assert native.support == ("supported", "supported", fx.OUTSIDE)
    assert native.support_status == fx.OUTSIDE
    assert native.calibration in {"point_only", "unmeasured"}
    assert native.has_joint_law is False

    law = fx.law()
    assert law.identity.causal_contract_id == program.identity
    assert law.n_draws == 2


def test_c4_a_fitted_native_response_binds_to_its_program() -> None:
    """An additional fitted response: ``E[y | do(a)] = 2a`` with seeded noise."""
    rng = np.random.default_rng(17)
    treatment = rng.normal(size=400)
    outcome = 2.0 * treatment + rng.normal(scale=0.2, size=400)
    grid = [-0.5, 0.0, 0.5]
    result = ac.analyze(
        {"a": treatment, "y": outcome},
        query=ac.ResponseCurve("a", "y", grid=grid),
        graph=[("a", "y")],
    )
    program = program_claims.ProgramBinding.from_response(
        result, outcome_units=fx.UNITS, dose_units="mg"
    )
    claim = program_claims.native_claim(result, program)
    assert claim.trust is ProviderTrust.NATIVE_LICENSED
    assert claim.program_identity == program.identity
    assert [q.regime_id for q in claim.coordinates] == ["do(a=-0.5)", "do(a=0)", "do(a=0.5)"]
    assert claim.support == ("supported",) * 3
    assert claim.calibration in {"point_only", "unmeasured"}
    assert claim.means == pytest.approx([-1.0, 0.0, 1.0], abs=0.3)

    low, high = claim.coordinates[0], claim.coordinates[2]
    contract = decision.Contract(
        actions=(
            decision.Action("low", inputs=(low,), utility=decision.x(0)),
            decision.Action("high", inputs=(high,), utility=decision.x(0)),
        ),
        utility_units="util",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    source = claim.as_decision_source(contract)
    assert source.source.provider_id.startswith("native:")
    assert contract.evaluate(source.source).selected == ("high",)


def test_c4_the_native_claim_enters_composition_with_issued_authority_and_supported_coordinates() -> (
    None
):
    native = fx.native_input()
    assert native.source == "mean"
    assert native.native is True
    assert native.provenance.trust == "native_licensed"
    assert native.provenance.calibration in {"point_only", "unmeasured"}
    assert [c.coordinate for c in native.support] == [fx.coordinate(0.0), fx.coordinate(1.0)]
    assert {c.status for c in native.support} == {"supported"}
    assert native.provenance.snapshot_id == SNAPSHOT
    with pytest.raises(comp.UnverifiedTrustRefusal) as refused:
        comp.DecisionInput.from_means(
            "native",
            [fx.coordinate(0.0)],
            [2.0],
            provider_id="native:op-n",
            snapshot_id=SNAPSHOT,
            causal_contract_id=fx.program().identity,
            requirement="native",
        )
    assert refused.value.detail == "composition_boundary.native_required"
    assert_registered_refusal(refused.value)


# -------------------------------------------------------- supported and unsupported actions


def test_c4_an_unsupported_action_is_reported_with_its_reason_and_the_rest_are_compared() -> None:
    contract = fx.mean_contract()
    decided = comp.evaluate_with_support(contract, [fx.native_input()])
    assert [o.id for o in decided.outcomes] == ["wait", "treat"]
    assert fx.near(decided.outcome("wait").expected_utility, 2.0)
    assert fx.near(decided.outcome("treat").expected_utility, 4.0 - 1.0)
    assert decided.verdict.kind == "uniquely_optimal" and decided.verdict.selected == "treat"
    assert decided.verdict.compared and decided.evpi is None
    assert decided.unsupported_actions == ("extend",)
    assert decided.disposition("extend").reasons == (
        comp.UnsupportedReason("native", 0, "composition_boundary.coordinate_missing", None),
    )
    with pytest.raises(comp.SupportRefusal) as refused:
        comp.evaluate_with_support(contract, [fx.native_input()], comp.SupportPolicy.require_all())
    assert refused.value.detail == "composition_boundary.unsupported_action_not_comparable"
    assert refused.value.offending == "extend"
    assert_registered_refusal(refused.value)


def test_c4_actions_unavailable_from_one_source_are_answered_by_another_by_value_only() -> None:
    contract = fx.mean_contract()
    alone = comp.evaluate_with_support(contract, [fx.e1_input()])
    assert fx.near(alone.outcome("wait").expected_utility, 1.0)
    assert fx.near(alone.outcome("treat").expected_utility, 3.0 - 1.0)
    assert fx.near(alone.outcome("extend").expected_utility, 5.5 - 3.0)
    assert alone.verdict.selected == "extend"

    both = comp.evaluate_with_support(contract, [fx.native_input(), fx.e1_input()])
    assert fx.near(both.outcome("wait").expected_utility, 2.0)
    assert fx.near(both.outcome("treat").expected_utility, 3.0)
    assert fx.near(both.outcome("extend").expected_utility, 2.5)
    assert both.verdict.kind == "uniquely_optimal" and both.verdict.selected == "treat"
    assert [both.disposition(i).input_id for i in IDS] == ["native", "native", "e1"]
    assert both.evpi is None, "sources are compared by value, never state by state"
    # `sources` counts source groups that compared two or more actions: the native source
    # compared wait and treat, while e1 alone answered extend and so forms no group.
    assert both.sources == 1
    assert all(o.expected_regret is None for o in both.outcomes)

    means = comp.evaluate_with_support(contract, [fx.e2_mean_input()])
    for action, utility in zip(IDS, (2.0, 2.5, 2.0), strict=True):
        assert fx.near(means.outcome(action).expected_utility, utility), action
    assert means.verdict.selected == "treat"


def test_c4_when_every_action_is_unsupported_the_answer_is_a_state_not_an_error() -> None:
    blind = comp.DecisionInput.from_means(
        "blind",
        [fx.coordinate(d) for d in fx.GRID],
        list(fx.E1_MEANS),
        provider_id="lab-1",
        snapshot_id="snap-e1",
        causal_contract_id=fx.program().identity,
        support="missing_evidence",
        evidence=comp.TrustEvidence.attested("lab-1"),
    )
    decided = comp.evaluate_with_support(fx.mean_contract(), [blind])
    assert decided.verdict.kind == "no_supported_action" and decided.verdict.selected is None
    assert decided.outcomes == () and decided.evpi is None and decided.sources == 0
    for disposition in decided.dispositions:
        assert disposition.status == "unsupported"
        assert [r.support for r in disposition.reasons] == ["missing_evidence"]
        assert [r.issue for r in disposition.reasons] == [
            "composition_boundary.coordinate_unsupported"
        ]
    with pytest.raises(comp.SupportRefusal):
        comp.evaluate_with_support(fx.mean_contract(), [blind], comp.SupportPolicy.require_all())

    nothing = fx.native_view(point_status=("missing_evidence",) * 3, support="missing_evidence")
    with pytest.raises(external.ExternalRefusal) as caught:
        fx.native_claim(nothing).as_decision_source(fx.mean_contract())
    assert caught.value.detail == "native_claims.projection_mismatch"


# ---------------------------------------------------- the joint-law decision, independently


def test_c4_the_joint_law_decision_matches_the_hand_derived_values() -> None:
    result = fx.law_contract().evaluate(fx.law())
    values = {a.id: a for a in result.actions}
    assert fx.near(values["wait"].expected_utility, 2.0)
    assert fx.near(values["treat"].expected_utility, (5.0 + 0.0) / 2.0)
    assert fx.near(values["extend"].expected_utility, (0.0 + 4.0) / 2.0)
    assert result.verdict == decision.Verdict("uniquely_optimal", ("treat",))
    assert fx.near(result.evpi, 4.5 - 2.5)
    assert fx.near(values["wait"].expected_regret, 2.5)
    assert fx.near(values["treat"].expected_regret, 2.0)
    assert fx.near(values["extend"].expected_regret, 2.5)
    assert result.n_draws == 2

    composed = comp.evaluate_with_support(fx.law_contract(), [fx.e2_law_input()])
    assert composed.verdict.selected == "treat" and composed.sources == 1
    assert fx.near(composed.evpi, 2.0)
    assert composed.unsupported_actions == ()
    assert fx.near(composed.outcome("treat").expected_regret, 2.0)

    # With the dose-2 coordinate masked unsupported only wait and treat are scored: E[max] = 3.5.
    masked = fx.e2_law_input(fx.law(supported=(True, True, False, True)))
    partial = comp.evaluate_with_support(fx.law_contract(), [masked])
    assert partial.unsupported_actions == ("extend",)
    (reason,) = partial.disposition("extend").reasons
    assert (reason.issue, reason.support) == (
        "composition_boundary.coordinate_unsupported",
        fx.OUTSIDE,
    )
    assert fx.near(partial.evpi, 3.5 - 2.5)
    assert partial.verdict.selected == "treat"


def test_c4_the_follow_up_decision_has_the_same_evpi_as_the_ranking() -> None:
    """EVPI of ``bet`` over the law's own state column is 1/8, as the ranking reports."""
    rollout = fx.rollout_contract().evaluate(fx.law())
    assert fx.near(rollout.evpi, 1.0 / 8.0)
    ranking = fx.rank()
    assert fx.near(ranking.evpi, 1.0 / 8.0)
    assert fx.near(ranking.prior_expected_utility, 0.0)


# ------------------------------------------------- point-only and joint-law answers distinct


def test_c4_a_probability_or_quantile_is_answered_only_by_the_joint_law() -> None:
    contract = fx.law_contract()
    law = fx.e2_law_input()
    probability = comp.evaluate_functional(
        contract, "treat", comp.Functional.probability(5.0, "upper"), law
    )
    assert fx.near(probability.value, 0.5)
    for action, quantile in zip(IDS, (2.0, 5.0, 4.0), strict=True):
        value = comp.evaluate_functional(contract, action, comp.Functional.quantile(0.75), law)
        assert fx.near(value.value, quantile), action

    # The same study's point means agree on the expectation and refuse everything else.
    mean = fx.e2_mean_input()
    on_means = comp.evaluate_functional(
        fx.mean_contract(), "treat", comp.Functional.expectation(), mean
    )
    assert fx.near(on_means.value, 3.5 - 1.0) and on_means.standard_error is None
    for functional in (comp.Functional.probability(2.0, "upper"), comp.Functional.quantile(0.5)):
        with pytest.raises(comp.SupportRefusal) as refused:
            comp.evaluate_functional(fx.mean_contract(), "treat", functional, mean)
        assert refused.value.detail == "composition_boundary.mean_is_not_a_distribution"
        assert refused.value.offending == "treat"
        assert_registered_refusal(refused.value)

    # A contract that ranks by a quantile cannot be evaluated on the mean claim either.
    quantile_contract = fx.mean_contract(decision.Criterion.quantile(0.5))
    with pytest.raises(decision.DecisionRefusal) as insufficient:
        quantile_contract.evaluate(fx.e2_claim())
    assert insufficient.value.detail == "decision_evaluation.mean_source_insufficient"
    assert insufficient.value.reason_code == "decision_contract_unsatisfied"
    with pytest.raises(external.ExternalRefusal) as unsupplied:
        fx.native_claim().as_decision_source(quantile_contract)
    assert unsupplied.value.detail == "native_claims.source_not_supplied"


def test_c4_the_uncertainty_sensitive_inverse_query_runs_on_the_law_and_refuses_the_mean() -> None:
    law = fx.law()
    probability = iq.InverseQuery(
        fx.law_contract(),
        IDS,
        (iq.probability_threshold(5.0, 0.25, tail="upper"),),
        selection="require_unique",
    ).evaluate(law)
    values = [a.point_values[0].value for a in probability.actions]
    assert values == pytest.approx([0.0, 0.5, 0.0], abs=1e-12)
    assert probability.feasible_actions == ("treat",) and probability.selected == "treat"

    quantile = iq.InverseQuery(
        fx.law_contract(), IDS, (iq.target_quantile(0.75, 4.5),), selection="require_unique"
    ).evaluate(law)
    values = [a.point_values[0].value for a in quantile.actions]
    assert values == pytest.approx([2.0, 5.0, 4.0], abs=1e-12)
    assert quantile.feasible_actions == ("treat",)

    # The same question on a point-mean study refuses with the typed detail.
    for constraint in (
        iq.probability_threshold(5.0, 0.25, tail="upper"),
        iq.target_quantile(0.75, 4.5),
    ):
        query = iq.InverseQuery(fx.mean_contract(), IDS, (constraint,))
        with pytest.raises(iq.InverseQueryRefusal) as refused:
            query.evaluate(fx.e2_claim())
        assert refused.value.detail == "decision_evaluation.mean_source_insufficient"
        assert refused.value.reason_code == "decision_contract_unsatisfied"
    # A target mean on affine utilities is answerable from means, without a sampling error.
    answered = iq.InverseQuery(fx.mean_contract(), IDS, (iq.target_mean(2.4),)).evaluate(
        fx.e2_claim()
    )
    assert answered.feasible_actions == ("treat",)
    assert answered.actions[1].point_values[0].standard_error is None


# ------------------------------------------------------- overlapping evidence is declared


def _relations() -> list[comp.PairRelation]:
    return [
        comp.PairRelation("native", "e1", comp.EvidenceRelation.independent()),
        comp.PairRelation("e1", "e2law", comp.EvidenceRelation.independent()),
        comp.PairRelation("native", "e2law", comp.EvidenceRelation.shared_data("registry-7")),
    ]


def test_c4_shared_data_refuses_independent_pooling_and_licensed_routes_lift_it() -> None:
    inputs = [fx.native_input(), fx.e1_input(), fx.e2_law_input()]
    with pytest.raises(comp.DependenceRefusal) as refused:
        comp.check_composition(inputs, _relations(), "statistical_pooling")
    assert refused.value.detail == "composition_boundary.shared_evidence_not_independent"
    assert refused.value.offending == "native~e2law"
    assert refused.value.reason_code == "scenario_aggregate_not_licensed"
    assert_registered_refusal(refused.value)

    # The two studies that really are independent pool, and the receipt says it assumed that.
    pair = comp.check_composition(
        [fx.e1_input(), fx.e2_law_input()], [_relations()[1]], "statistical_pooling"
    )
    assert pair.independence_assumed and pair.shared_evidence == () and pair.routes == ()

    # A declared non-pooling operation counts the shared registry once and assumes no independence.
    reuse = comp.check_composition(inputs, _relations(), "evidence_reuse")
    assert reuse.operation == "evidence_reuse" and not reuse.independence_assumed
    assert reuse.shared_evidence == ("registry-7",)
    transport = comp.check_composition(inputs, _relations(), "causal_transport")
    assert transport.shared_evidence == ("registry-7",)

    # An explicitly licensed joint-law route (one of the pair, an aligned law) lifts the refusal.
    route = comp.DependenceRoute("joint_law", "jl-1", "e2law")
    licensed = [
        comp.PairRelation("native", "e2law", comp.EvidenceRelation.shared_data("registry-7"), route)
    ]
    receipt = comp.check_composition(
        [fx.native_input(), fx.e2_law_input()], licensed, "statistical_pooling"
    )
    assert receipt.routes == ("jl-1",) and not receipt.independence_assumed
    assert receipt.shared_evidence == ("registry-7",)


def test_c4_paired_draws_refuse_a_mean_and_shared_data_without_a_licensed_route() -> None:
    reanalysis = fx.e2_law_input(fx.law(provider_id="lab-2b", snapshot="snap-e2b"), "reanalysis")
    pair = [fx.e2_law_input(), reanalysis]
    shared = comp.PairRelation(
        "e2law", "reanalysis", comp.EvidenceRelation.shared_data("registry-7")
    )
    with pytest.raises(comp.DependenceRefusal) as refused:
        comp.check_paired_draws(pair, [shared])
    assert refused.value.detail == "composition_boundary.shared_evidence_not_independent"

    route = comp.DependenceRoute("joint_law", "jl-1", "e2law")
    licensed = comp.PairRelation(
        "e2law", "reanalysis", comp.EvidenceRelation.shared_data("registry-7"), route
    )
    receipt = comp.check_paired_draws(pair, [licensed])
    assert receipt.routes == ("jl-1",) and receipt.shared_evidence == ("registry-7",)
    assert not receipt.independence_assumed

    with pytest.raises(comp.CompositionRefusal) as mean:
        comp.check_paired_draws([fx.e2_law_input(), fx.e1_input()], [])
    assert mean.value.reason_code == "joint_law_required"


def test_c4_unknown_dependence_is_never_independence_and_atoms_are_not_averaged() -> None:
    inputs = [fx.e1_input(), fx.e2_law_input()]
    for operation in comp.OPERATIONS:
        with pytest.raises(comp.DependenceRefusal) as refused:
            comp.check_composition(inputs, [], operation)
        assert refused.value.detail == "composition_boundary.unknown_dependence_is_not_independence"
    atoms = [comp.StructuralAtom("graph-a"), comp.StructuralAtom("graph-b")]
    with pytest.raises(comp.DependenceRefusal) as averaged:
        comp.check_atom_combination(atoms, "weighted_by_declared_probabilities")
    assert averaged.value.detail == "composition_boundary.conflicting_atoms_not_averaged"
    assert comp.check_atom_combination(atoms, "report_each").kind == "report_each"


# ------------------------------------------------------------------------- study ranking


def test_c4_the_ranking_matches_the_hand_derived_evsi_and_net_values() -> None:
    ranking = fx.rank()
    assert ranking.basis == "net_value"
    assert [c.id for c in ranking.candidates] == ["native-trial", "external-trial"]
    native, attested = ranking.candidates
    assert fx.near(native.evsi, 10.0 / 32.0 * (0.7 - 0.5))
    assert fx.near(attested.evsi, 1.0 / 16.0)
    assert fx.near(native.evpi, 1.0 / 8.0) and fx.near(attested.evpi, 1.0 / 8.0)
    assert fx.near(native.study_cost_utility, 0.02)
    assert fx.near(native.net_value, 1.0 / 16.0 - 0.02)
    assert fx.near(attested.net_value, 1.0 / 16.0 - 0.05)
    assert fx.near(native.net_value, 0.0425) and fx.near(attested.net_value, 0.0125)
    assert native.provider_trust == "native_licensed" and native.natively_replayed
    assert attested.provider_trust == "externally_attested" and not attested.natively_replayed
    assert ranking.calibration == "unmeasured"
    assert ranking.source_digests == (decision_source_digest(),)
    assert ranking.ties == ()
    # The order candidates are supplied in does not matter.
    reversed_ = fx.rank(candidate_list=list(reversed(fx.candidates())))
    assert reversed_.identity == ranking.identity


def decision_source_digest() -> str:
    return decision.source_digest(fx.law())


# ----------------------------------------------------------------- the composition bundle


JOINT_VALUES = [
    ["result_j", "wait.expected_utility"],
    ["result_j", "treat.expected_utility"],
    ["result_j", "extend.expected_utility"],
    ["ranking", "native-trial.net_value"],
    ["ranking", "external-trial.net_value"],
    ["ranking", "native-trial.evsi"],
]
JOINT_EXPECTED = {
    "result_j:wait.expected_utility": 2.0,
    "result_j:treat.expected_utility": 2.5,
    "result_j:extend.expected_utility": 2.0,
    "ranking:native-trial.net_value": 0.0425,
    "ranking:external-trial.net_value": 0.0125,
    "ranking:native-trial.evsi": 0.0625,
}


def test_c4_the_composed_bundle_is_consumed_in_process_with_the_hand_derived_values() -> None:
    bundle = _composed_builder().build()
    kinds = {n.id: n.kind for n in bundle.nodes}
    assert RELATION in kinds
    del kinds[RELATION]
    assert kinds == {
        "program": "causal_contract",
        "e1": "external_claim",
        "e2claim": "external_claim",
        "contract_j": "decision_contract",
        "e2law": "distribution",
        "result_j": "decision_result",
        "ranking": "study_ranking",
    }
    assert {n.id: n.embedded for n in bundle.nodes}["program"] is False
    consumed = cb.consume_bundle(bundle.export(), expected_identity=bundle.identity)
    assert consumed.claim_label == "joint_draw"
    assert consumed.node("result_j").claim_label == "joint_draw"
    assert isinstance(consumed.node("program").status, cb.ReferenceUnresolved)
    assert consumed.node("program").inspected == {"native.mean.do(a=1)": 4.0}
    assert consumed.value("program", "native.mean.do(a=1)") is None
    for node in consumed.nodes:
        if node.id != "program":
            assert node.verified, (node.id, node.status)
    for node_id, key in JOINT_VALUES:
        assert consumed.value(node_id, key) == pytest.approx(
            JOINT_EXPECTED[f"{node_id}:{key}"], abs=1e-12
        )
    assert consumed.node("e1").facts["law"] == "mean_only"
    assert consumed.node("e2law").facts["law"] == "joint_draw"
    with pytest.raises(cb.CallbackUnavailableRefusal):
        consumed.require_verified()
    supplied = cb.SuppliedSources().with_data(SNAPSHOT, NATIVE_DIGEST)
    resolved = cb.consume_bundle(
        bundle.export(), expected_identity=bundle.identity, supplied=supplied
    )
    resolved.require_verified()
    assert resolved.all_verified


def _consume_fresh(
    path: Path, identity: str, wanted: list[list[str]], *supplied: str
) -> dict[str, object]:
    done = subprocess.run(
        [sys.executable, "-c", CONSUMER, str(path), identity, json.dumps(wanted), *supplied],
        capture_output=True,
        text=True,
        check=True,
    )
    report: dict[str, object] = json.loads(done.stdout)
    return report


def test_c4_a_fresh_interpreter_consumes_the_joint_bundle_and_reads_the_values(
    tmp_path: Path,
) -> None:
    bundle = _composed_builder().build()
    path = tmp_path / "composed.bin"
    path.write_bytes(bundle.export())

    report = _consume_fresh(path, bundle.identity, JOINT_VALUES)
    assert report["label"] == "joint_draw"
    nodes = report["nodes"]
    assert isinstance(nodes, dict)
    program_replayed = nodes.pop("program")
    assert program_replayed is False, "an unsupplied data reference is never claimed replayed"
    assert all(nodes.values()), nodes
    assert set(nodes) == {"e1", "e2claim", RELATION, "contract_j", "e2law", "result_j", "ranking"}
    values = report["values"]
    assert isinstance(values, dict)
    for key, expected in JOINT_EXPECTED.items():
        assert values[key] == pytest.approx(expected, abs=1e-12), key
    kinds = report["kinds"]
    assert isinstance(kinds, dict) and kinds["ranking"] == "study_ranking"

    # With the supplied native snapshot (and its digest) every node verifies.
    full = _consume_fresh(path, bundle.identity, JOINT_VALUES, SNAPSHOT, NATIVE_DIGEST)
    assert full["all_verified"] is True

    # Another retained identity refuses the same bytes in the fresh interpreter.
    refused = subprocess.run(
        [sys.executable, "-c", CONSUMER, str(path), "0" * 64, "[]"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0 and "expected_identity_mismatch" in refused.stderr


def test_c4_the_point_only_decision_is_exported_as_an_attested_result_and_consumed_fresh(
    tmp_path: Path,
) -> None:
    bundle = _point_only_builder().build()
    path = tmp_path / "point_only.bin"
    path.write_bytes(bundle.export())
    wanted = [["result_m", f"{action}.expected_utility"] for action in IDS]
    report = _consume_fresh(path, bundle.identity, wanted)
    assert report["label"] == "point_only_attested" and report["all_verified"] is True
    values = report["values"]
    assert isinstance(values, dict)
    assert values["result_m:wait.expected_utility"] == pytest.approx(1.0, abs=1e-12)
    assert values["result_m:treat.expected_utility"] == pytest.approx(3.0 - 1.0, abs=1e-12)
    assert values["result_m:extend.expected_utility"] == pytest.approx(5.5 - 3.0, abs=1e-12)

    # A mean-only claim added upstream of a joint decision does not invalidate it, because the
    # aligned law it needs is still upstream; the label becomes the conservative one. (The
    # refusal of a joint-requiring decision over a mean-only claim ALONE is the C3 verifier
    # test `unsupported_law`.)
    law_bundle = _composed_builder().connect("e2claim", "result_j").build()
    consumed = cb.consume_bundle(law_bundle.export(), expected_identity=law_bundle.identity)
    assert consumed.node("result_j").verified
    assert consumed.node("result_j").claim_label == "point_only_attested"
    assert consumed.node("ranking").verified, "the unrelated ranking branch stays valid"

    # A bundle that mixes a mean decision with a joint one is labelled conservatively.
    mixed = _composed_builder()
    mixed.add_artifact("external_claim", fx.e1_claim(), node_id="claim")
    contract = fx.mean_contract()
    mixed.add_artifact(
        "decision_contract", contract.export(artifact_id="contract"), node_id="contract_m"
    )
    mixed.add_artifact(
        "decision_result", cb.mean_decision(contract, fx.e1_claim()), node_id="result_m"
    )
    mixed.connect("claim", "result_m").connect("contract_m", "result_m")
    sealed = mixed.build()
    assert (
        cb.consume_bundle(sealed.export(), expected_identity=sealed.identity).claim_label
        == "point_only_attested"
    )
