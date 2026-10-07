"""Admissibility rules, claim adapters and robustness verdicts for decisions.

Mirrors `crates/antecedent-design/tests/decision_robustness.rs`,
`decision_adapters.rs` and `decision_robust_artifact.rs`: each structure is the
exact law of the payoffs of actions A and B on two equally likely rows, so every
expected value is the hand-set payoff.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from antecedent import decision, decision_robust
from antecedent.errors import (
    CausalSerializationError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)


def _quantity(variable: str, regime: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units="units",
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id="outcome",
    )


COLUMNS = (_quantity("a", "do(a=1)"), _quantity("b", "do(a=0)"))


def _law(a: float, b: float) -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=COLUMNS,
        alignment="joint",
        source_id="structure",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id="enumeration",
        causal_contract_id="checked",
    )
    draws = np.array([[a, b], [a, b]], dtype=np.float64)
    return JointDistributionArtifact(identity, draws, calibration="exact")


def _base(
    policy: str = "require_invariant_best_action", *, reverse: bool = False
) -> decision.Contract:
    actions = (
        decision.Action("A", inputs=(COLUMNS[0],), utility=decision.x(0)),
        decision.Action("B", inputs=(COLUMNS[1],), utility=decision.x(0)),
    )
    return decision.Contract(
        actions=actions[::-1] if reverse else actions,
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        structural_policy=policy,  # type: ignore[arg-type]
    )


def _contract(
    policy: str = "require_invariant_best_action",
    rules: decision_robust.AdmissibilityRules | None = None,
) -> decision_robust.AdmissibleContract:
    return decision_robust.admissible_contract(_base(policy), rules)


def _require_supported(**kwargs: object) -> decision_robust.AdmissibilityRules:
    return decision_robust.AdmissibilityRules(
        default_weakest_support="supported",
        **kwargs,  # type: ignore[arg-type]
    )


def _claims(
    s1: tuple[float, float],
    s2: tuple[float, float],
    *,
    support: tuple[decision_robust.Support | None, decision_robust.Support | None] = (None, None),
    probabilities: tuple[float | None, float | None] = (None, None),
) -> list[decision_robust.Claim]:
    return [
        decision_robust.Claim.evaluated(
            "s1", _law(*s1), support=support[0], probability=probabilities[0]
        ),
        decision_robust.Claim.evaluated(
            "s2", _law(*s2), support=support[1], probability=probabilities[1]
        ),
    ]


SUPPORTED = decision_robust.Support("supported")


def test_f6_admissible_identity_ignores_order_and_changes_with_rules() -> None:
    rules = decision_robust.AdmissibilityRules(
        default_weakest_support="weak_overlap",
        support_rules=(
            decision_robust.SupportRule("A", 0, "supported"),
            decision_robust.SupportRule("B", 0, "extrapolative"),
        ),
        declared_exclusions=(
            decision_robust.DeclaredExclusion("A", "legal"),
            decision_robust.DeclaredExclusion("B", "not licensed"),
        ),
        uncertainty="structural_envelope",
    )
    contract = decision_robust.AdmissibleContract(_base(), rules)
    reordered = decision_robust.AdmissibleContract(
        _base(reverse=True),
        decision_robust.AdmissibilityRules(
            default_weakest_support="weak_overlap",
            support_rules=rules.support_rules[::-1],
            declared_exclusions=rules.declared_exclusions[::-1],
            uncertainty="structural_envelope",
        ),
    )
    assert contract.identity == reordered.identity
    assert contract.base_identity == _base().identity
    # Rules are part of the identity.
    assert contract.identity != contract.base_identity
    assert contract.identity != decision_robust.admissible_contract(_base()).identity
    assert contract.identity != contract.with_admissibility(_require_supported()).identity
    changed = decision_robust.AdmissibilityRules(
        default_weakest_support="weak_overlap",
        support_rules=(
            decision_robust.SupportRule("A", 0, "extrapolative"),
            decision_robust.SupportRule("B", 0, "extrapolative"),
        ),
        declared_exclusions=rules.declared_exclusions,
        uncertainty="structural_envelope",
    )
    assert contract.identity != contract.with_admissibility(changed).identity


def test_f6_admissible_contract_exports_loads_and_refuses_a_resealed_identity() -> None:
    rules = _require_supported(uncertainty="credible")
    contract = decision_robust.AdmissibleContract(_base(), rules)
    data = contract.export()
    loaded = decision_robust.AdmissibleContract.load(data, expected_identity=contract.identity)
    assert loaded.identity == contract.identity
    assert loaded.rules == rules
    assert loaded.contract.identity == _base().identity
    other = contract.with_admissibility(_require_supported(uncertainty="point_only"))
    with pytest.raises(CausalSerializationError, match="differs"):
        decision_robust.AdmissibleContract.load(data, expected_identity=other.identity)
    # The plain contract path is unchanged: it carries no rules and loads as before.
    plain = _base()
    assert decision.Contract.load(plain.export(), expected_identity=plain.identity) == plain


def test_f6_invalid_rules_refuse_with_registered_code_and_detail() -> None:
    ghost = decision_robust.AdmissibilityRules(
        support_rules=(decision_robust.SupportRule("ghost", 0, "supported"),)
    )
    with pytest.raises(decision.DecisionRefusal) as unknown:
        _ = decision_robust.admissible_contract(_base(), ghost).identity
    assert isinstance(unknown.value, CausalUnsupportedError)
    assert unknown.value.reason_code == "invalid_argument"
    assert unknown.value.detail == "decision_admissibility.unknown_action"
    assert unknown.value.offending == "ghost"
    bad_input = decision_robust.AdmissibilityRules(
        support_rules=(decision_robust.SupportRule("A", 7, "supported"),)
    )
    with pytest.raises(decision.DecisionRefusal) as missing:
        _ = decision_robust.admissible_contract(_base(), bad_input).identity
    assert missing.value.detail == "decision_admissibility.unknown_input"
    assert missing.value.offending == "A[7]"
    with pytest.raises(CausalValueError):
        decision_robust.AdmissibilityRules(uncertainty="sometimes")  # type: ignore[arg-type]


def test_f6_invariant_best_is_structurally_robust_and_graph_dependence_is_named() -> None:
    contract = _contract(rules=_require_supported())
    robust = decision_robust.finite_scenarios(
        contract, _claims((5.0, 3.0), (6.0, 2.0), support=(SUPPORTED, SUPPORTED))
    )
    assert robust.verdict == decision_robust.RobustVerdict("structurally_robust", action="A")
    assert robust.selected == "A"
    assert robust.shortfalls == ()
    # The ranges are the hand-set payoffs: A in {5, 6}, B in {2, 3}.
    by_id = {a.id: a for a in robust.actions}
    assert by_id["A"].range == (5.0, 6.0)
    assert by_id["B"].range == (2.0, 3.0)
    assert robust.native_verified
    assert "uniquely best in every structure" in robust.explain()

    # Structures that disagree: the choice is graph dependent, never averaged.
    plain = _contract()
    graph = decision_robust.graph_dependent(plain, _claims((5.0, 3.0), (1.0, 4.0)))
    assert graph.verdict.kind == "graph_dependent_choice"
    assert graph.verdict.leaders == (("s1", ("A",)), ("s2", ("B",)))
    assert graph.selected is None
    assert graph.profile.uncertainty == "structural_envelope"
    assert "depends on the structure" in graph.explain()
    assert graph.kind == "graph_dependent"
    assert {s["status"] for s in graph.structures} == {"evaluated"}


def test_f6_support_robust_support_dependent_and_unsupported_states() -> None:
    contract = _contract(rules=_require_supported())
    weak = decision_robust.Support("extrapolative")
    outside = decision_robust.Support("outside_empirical_support")

    robust = decision_robust.robust(
        contract, _claims((5.0, 3.0), (6.0, 2.0), support=(SUPPORTED, weak))
    )
    assert robust.verdict == decision_robust.RobustVerdict("support_robust", action="A")
    assert robust.unsupported_atoms == ("s2",)
    assert len(robust.shortfalls) == 2
    assert robust.shortfalls[0] == decision_robust.SupportShortfall(
        "s2", "A", 0, "extrapolative", "supported"
    )

    dependent = decision_robust.robust(
        contract, _claims((5.0, 3.0), (1.0, 4.0), support=(SUPPORTED, outside))
    )
    assert dependent.verdict.kind == "support_dependent"
    assert dependent.verdict.action == "A"
    assert dependent.verdict.unrestricted_choice is None
    assert "depends on the support rules" in dependent.explain()

    none = decision_robust.robust(
        contract, _claims((5.0, 3.0), (6.0, 2.0), support=(weak, outside))
    )
    assert none.verdict.kind == "unsupported_extrapolation"
    # Unassessed support counts as missing evidence once a rule needs it.
    unassessed = decision_robust.robust(contract, _claims((5.0, 3.0), (6.0, 2.0)))
    assert unassessed.verdict.kind == "unsupported_extrapolation"
    assert all(s.status == "missing_evidence" for s in unassessed.shortfalls)

    # Per-input support removes one action in one structure only.
    per_input = decision_robust.Support("supported", {("B", 0): "extrapolative"})
    mixed = decision_robust.robust(
        contract, _claims((5.0, 3.0), (1.0, 4.0), support=(SUPPORTED, per_input))
    )
    by_id = {a.id: a for a in mixed.actions}
    assert by_id["B"].unsupported_in == ("s2",)
    assert by_id["B"].range == (3.0, 4.0)
    assert by_id["B"].supported_range == (3.0, 3.0)


def test_f6_declared_exclusions_and_no_admissible_action_only_remove_actions() -> None:
    rules = decision_robust.AdmissibilityRules(
        declared_exclusions=(decision_robust.DeclaredExclusion("A", "legal"),)
    )
    removed = decision_robust.robust(_contract(rules=rules), _claims((5.0, 3.0), (6.0, 2.0)))
    # A has the higher payoff everywhere but is declared inadmissible; B wins and
    # A keeps its raw range rather than a penalized one.
    assert removed.verdict.action == "B"
    by_id = {a.id: a for a in removed.actions}
    assert by_id["A"].declared_exclusion == "legal"
    assert by_id["A"].range == (5.0, 6.0)
    assert "removed by declaration" in removed.explain()

    both = decision_robust.AdmissibilityRules(
        declared_exclusions=(
            decision_robust.DeclaredExclusion("A", "legal"),
            decision_robust.DeclaredExclusion("B", "ethics"),
        )
    )
    nothing = decision_robust.robust(_contract(rules=both), _claims((5.0, 3.0), (6.0, 2.0)))
    assert nothing.verdict.kind == "no_admissible_action"
    assert nothing.selected is None


def test_f6_uncertainty_requirement_is_met_only_by_its_own_kind() -> None:
    claims = _claims((5.0, 3.0), (6.0, 2.0))
    for required, supplied, kind, accepted in [
        ("none", "structural_envelope", "finite_scenarios", True),
        ("structural_envelope", "structural_envelope", "finite_scenarios", True),
        ("credible", "structural_envelope", "finite_scenarios", False),
        ("point_only", "structural_envelope", "finite_scenarios", False),
    ]:
        contract = _contract(rules=decision_robust.AdmissibilityRules(uncertainty=required))  # type: ignore[arg-type]
        result = decision_robust.robust(contract, claims, kind=kind)  # type: ignore[arg-type]
        assert result.uncertainty_supplied == supplied
        assert result.uncertainty_required == required
        assert (result.verdict.kind != "insufficient_claims") is accepted, (required, kind)
    # A single point claim is a point, not an envelope.
    point = decision_robust.point_claim(
        _contract(rules=decision_robust.AdmissibilityRules(uncertainty="point_only")),
        decision_robust.Claim.evaluated("only", _law(5.0, 3.0)),
    )
    assert point.uncertainty_supplied == "point"
    assert point.verdict.kind != "insufficient_claims"
    wrong = decision_robust.point_claim(
        _contract(rules=decision_robust.AdmissibilityRules(uncertainty="structural_envelope")),
        decision_robust.Claim.evaluated("only", _law(5.0, 3.0)),
    )
    assert wrong.verdict.kind == "insufficient_claims"
    assert "structural_envelope" in (wrong.verdict.reason or "")


def test_f6_weighted_atoms_keep_unresolved_mass_and_completion_counts_are_not_weights() -> None:
    # A = 0.5*5 + 0.5*1 = 3.0 and B = 0.5*3 + 0.5*4 = 3.5.
    bayes = _contract("bayes_over_structures")
    weighted = decision_robust.weighted_atoms(
        bayes, _claims((5.0, 3.0), (1.0, 4.0), probabilities=(0.5, 0.5))
    )
    assert weighted.verdict.kind == "bayes_choice"
    assert weighted.verdict.action == "B"
    assert weighted.verdict.evaluated_mass == pytest.approx(1.0)
    assert weighted.uncertainty_supplied == "credible"
    by_id = {a["id"]: a for a in weighted._body["structural"]["actions"]}
    assert by_id["A"]["weighted_value"] == pytest.approx(3.0)
    assert by_id["B"]["weighted_value"] == pytest.approx(3.5)

    # Completion counts are not probabilities.
    counts = [
        decision_robust.Claim.evaluated("s1", _law(5.0, 3.0), completion_count=3),
        decision_robust.Claim.evaluated("s2", _law(1.0, 4.0), completion_count=1),
    ]
    with pytest.raises(decision.DecisionRefusal) as refused:
        decision_robust.weighted_atoms(bayes, counts)
    assert refused.value.detail == "decision_adapters.completion_count_not_probability"
    assert refused.value.reason_code == "decision_contract_unsatisfied"
    # Under another policy they are retained and never used as weights.
    kept = decision_robust.weighted_atoms(_contract(), counts)
    assert kept.completion_counts == {"s1": 3, "s2": 1}
    assert kept.uncertainty_supplied == "structural_envelope"

    # Graph-dependent claims carry no probability.
    with pytest.raises(decision.DecisionRefusal) as graph:
        decision_robust.graph_dependent(
            _contract(), _claims((5.0, 3.0), (1.0, 4.0), probabilities=(0.5, 0.5))
        )
    assert graph.value.detail == "decision_adapters.graph_claims_carry_no_probability"

    # Unidentified mass stays unidentified: the Bayes choice is not made on the rest.
    partial = [
        decision_robust.Claim.evaluated("s1", _law(5.0, 3.0), probability=0.8),
        decision_robust.Claim.unidentified("s2", probability=0.2),
    ]
    held = decision_robust.weighted_atoms(bayes, partial)
    assert held.verdict.kind == "insufficient_claims"
    assert held.unidentified_mass == pytest.approx(0.2)
    assert held.evaluated_mass == pytest.approx(0.8)
    assert [s["status"] for s in held.structures] == ["evaluated", "unidentified"]


def test_f6_unresolved_structures_leave_invariance_unchecked() -> None:
    contract = _contract()
    for unresolved in (
        decision_robust.Claim.unidentified("s2"),
        decision_robust.Claim.unevaluated("s2", "budget"),
    ):
        result = decision_robust.robust(
            contract, [decision_robust.Claim.evaluated("s1", _law(5.0, 3.0)), unresolved]
        )
        assert result.verdict.kind == "insufficient_claims", unresolved
    with pytest.raises(decision.DecisionRefusal) as no_reason:
        decision_robust.robust(
            contract,
            [
                decision_robust.Claim.evaluated("s1", _law(5.0, 3.0)),
                decision_robust.Claim("s2", "unevaluated"),
            ],
        )
    assert no_reason.value.detail == "decision_adapters.unevaluated_needs_reason"
    with pytest.raises(decision.DecisionRefusal) as two:
        decision_robust.robust(contract, _claims((5.0, 3.0), (6.0, 2.0)), kind="point")
    assert two.value.detail == "decision_adapters.point_claim_is_single"


def test_f6_identified_sets_partial_identification_reports_an_interval_per_action() -> None:
    contract = _contract()

    def decide(
        a: tuple[float, float],
        b: tuple[float, float],
        under: decision_robust.AdmissibleContract = contract,
    ) -> decision_robust.IdentifiedDecision:
        return decision_robust.identified_sets(
            under,
            [
                decision_robust.IdentifiedUtility("A", *a),
                decision_robust.IdentifiedUtility("B", *b),
            ],
        )

    clear = decide((4.0, 6.0), (1.0, 3.0))
    assert clear.verdict == decision_robust.IdentifiedVerdict("necessarily_best", action="A")
    assert clear.selected == "A"
    assert not clear.conflicting_leaders
    by_id = {a.id: a for a in clear.actions}
    assert by_id["B"].dominated_by == ("A",)
    assert by_id["A"].necessarily_optimal and not by_id["B"].possibly_optimal

    # Overlapping intervals: no action is best everywhere; both stay possibly optimal.
    overlap = decide((2.0, 5.0), (3.0, 6.0))
    assert overlap.verdict.kind == "no_necessarily_best"
    assert set(overlap.verdict.actions) == {"A", "B"}
    assert overlap.selected is None
    assert overlap.lower_leader == "B" and overlap.upper_leader == "B"
    assert "possibly optimal" in overlap.explain()

    # Maximin picks the best worst case: B's lower bound 3 beats A's 2.
    maximin = decide((2.0, 9.0), (3.0, 6.0), _contract("maximin"))
    assert maximin.verdict == decision_robust.IdentifiedVerdict("worst_case_choice", action="B")
    assert maximin.conflicting_leaders
    assert "lower-bound leader" in maximin.explain()

    # An identified set has no probability law.
    with pytest.raises(decision.DecisionRefusal) as bayes:
        decide((2.0, 5.0), (3.0, 6.0), _contract("bayes_over_structures"))
    assert bayes.value.detail == "decision_adapters.bayes_over_identified_set"
    assert bayes.value.reason_code == "decision_contract_unsatisfied"
    with pytest.raises(decision.DecisionRefusal) as inverted:
        decide((5.0, 2.0), (3.0, 6.0))
    assert inverted.value.detail == "decision_adapters.invalid_interval"
    with pytest.raises(decision.DecisionRefusal) as missing:
        decision_robust.identified_sets(
            contract, [decision_robust.IdentifiedUtility("A", 2.0, 5.0)]
        )
    assert missing.value.detail == "decision_adapters.missing_action"


def test_f6_identified_sets_apply_support_hard_exclusions_and_interval_arithmetic() -> None:
    contract = _contract(rules=_require_supported())
    weak = decision_robust.Support("extrapolative")
    result = decision_robust.identified_sets(
        contract,
        [
            decision_robust.IdentifiedUtility("A", 8.0, 9.0, support=weak),
            decision_robust.IdentifiedUtility("B", 1.0, 2.0, support=SUPPORTED),
        ],
    )
    # A's interval is higher but its input lacks the required support, so it is
    # removed (never penalized) and B is the only eligible action.
    by_id = {a.id: a for a in result.actions}
    assert not by_id["A"].eligible
    assert by_id["A"].support_shortfalls == ((0, "extrapolative"),)
    assert by_id["A"].utility == (8.0, 9.0)
    assert result.verdict.action == "B"

    hard = decision_robust.identified_sets(
        _contract(),
        [
            decision_robust.IdentifiedUtility("A", 8.0, 9.0, hard_exclusions=("cap",)),
            decision_robust.IdentifiedUtility("B", 1.0, 2.0),
        ],
    )
    assert hard.verdict.action == "B"
    assert hard.actions[0].hard_exclusions == ("cap",)

    # An interval per input encloses the action's utility: (b - c) over b in [2, 5], c in [1, 2].
    two = decision.Contract(
        actions=(
            decision.Action(
                "treat",
                inputs=(COLUMNS[0], COLUMNS[1]),
                utility=decision.x(0) - decision.x(1),
            ),
            decision.Action("wait", inputs=(COLUMNS[1],), utility=decision.x(0)),
        ),
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        structural_policy="require_invariant_best_action",
    )
    wrapped = decision_robust.admissible_contract(two)
    treat = decision_robust.IdentifiedUtility.from_inputs(
        wrapped, "treat", [(2.0, 5.0), (1.0, 2.0)]
    )
    assert (treat.lower, treat.upper) == (0.0, 4.0)
    with pytest.raises(decision.DecisionRefusal) as unknown:
        decision_robust.IdentifiedUtility.from_inputs(wrapped, "ghost", [(0.0, 1.0)])
    assert unknown.value.detail == "decision_adapters.unknown_action"


def test_f6_external_callback_receipt_is_retained_and_never_labelled_native() -> None:
    contract = _contract(rules=_require_supported())
    receipt = decision_robust.ExternalReceipt(
        atom_id="s1",
        provider_id="lab-model",
        snapshot_id="snap-9",
        request_fingerprint="ab" * 32,
        attested_value=4.25,
        attestor="outside-lab",
    )
    result = decision_robust.robust(
        contract,
        _claims((5.0, 3.0), (6.0, 2.0), support=(SUPPORTED, SUPPORTED)),
        receipts=[receipt],
    )
    assert not result.native_verified
    assert result.receipts == (receipt,)
    assert result.receipts[0].attested_value == 4.25
    assert result.receipts[0].request_fingerprint == "ab" * 32
    assert result.receipts[0].trust == "externally_attested"
    assert "not natively verified" in result.explain()
    assert "externally_attested" in result.explain()
    stages = {link.stage for link in result.lineage}
    assert {"external_provider", "distribution_artifact", "decision_contract", "claim"} <= stages
    assert result.lineage[-1].id == decision_robust.RESULT_LINK_ID
    # A native-only result has no external provider behind it.
    native = decision_robust.robust(
        contract, _claims((5.0, 3.0), (6.0, 2.0), support=(SUPPORTED, SUPPORTED))
    )
    assert native.native_verified and native.receipts == ()
    assert "external_provider" not in {link.stage for link in native.lineage}
    # Changing the attested value changes the derivation.
    other = decision_robust.robust(
        contract,
        _claims((5.0, 3.0), (6.0, 2.0), support=(SUPPORTED, SUPPORTED)),
        receipts=[
            decision_robust.ExternalReceipt(
                "s1", "lab-model", "snap-9", "ab" * 32, 4.5, attestor="outside-lab"
            )
        ],
    )
    assert other.lineage[-1].digest != result.lineage[-1].digest

    # There is no native trust: it is refused, never accepted.
    forged = decision_robust.ExternalReceipt(
        "s1",
        "lab-model",
        "snap-9",
        "ab" * 32,
        4.25,
        trust="native",  # type: ignore[arg-type]
    )
    with pytest.raises(decision.DecisionRefusal) as native_claim:
        decision_robust.robust(contract, _claims((5.0, 3.0), (6.0, 2.0)), receipts=[forged])
    assert native_claim.value.detail == "decision_robust.external_trust_never_native"
    # A receipt must name a structure that has draws.
    with pytest.raises(decision.DecisionRefusal) as ghost:
        decision_robust.robust(
            contract,
            _claims((5.0, 3.0), (6.0, 2.0)),
            receipts=[
                decision_robust.ExternalReceipt(
                    "ghost", "lab-model", "snap-9", "ab" * 32, 4.25, attestor="outside-lab"
                )
            ],
        )
    assert "receipt_atom" in ghost.value.detail or "receipt_atom" in str(ghost.value.offending)


def test_f6_robust_result_exports_replays_and_refuses_changed_inputs() -> None:
    contract = _contract(rules=_require_supported())
    claims = _claims((5.0, 3.0), (6.0, 2.0), support=(SUPPORTED, SUPPORTED))
    result = decision_robust.robust(contract, claims)
    data = result.export()
    replayed = decision_robust.replay(data, contract=contract, claims=claims)
    assert replayed.replayed
    assert replayed.verdict == result.verdict
    assert replayed.native_verified and replayed.external_atoms == ()
    assert replayed.lineage == result.lineage

    # Different draws, contract or support refuse.
    with pytest.raises(decision.DecisionRefusal):
        decision_robust.replay(
            data,
            contract=contract,
            claims=_claims((5.5, 3.0), (6.0, 2.0), support=(SUPPORTED, SUPPORTED)),
        )
    with pytest.raises(decision.DecisionRefusal):
        decision_robust.replay(
            data, contract=_contract("maximin", _require_supported()), claims=claims
        )
    with pytest.raises(decision.DecisionRefusal) as profile:
        decision_robust.replay(
            data,
            contract=contract,
            claims=_claims(
                (5.0, 3.0),
                (6.0, 2.0),
                support=(SUPPORTED, decision_robust.Support("weak_overlap")),
            ),
        )
    assert profile.value.detail.endswith("replay_profile")
    # A changed byte refuses; it is never read as a different result.
    corrupt = bytearray(data)
    corrupt[-1] ^= 0x55
    with pytest.raises(CausalUnsupportedError):
        decision_robust.replay(bytes(corrupt), contract=contract, claims=claims)

    # An external callback leaves a receipt that survives export and replay.
    receipt = decision_robust.ExternalReceipt(
        "s1", "lab-model", "snap-9", "cd" * 32, 4.25, attestor="outside-lab"
    )
    external = decision_robust.robust(contract, claims, receipts=[receipt])
    stored = decision_robust.replay(external.export(), contract=contract, claims=claims)
    assert stored.receipts == (receipt,)
    assert not stored.native_verified
    assert stored.external_atoms == ("s1",)
    assert stored.replayed
    # The native artifact never loads as an external one and vice versa.
    assert json.loads(json.dumps(stored.receipts[0].attested_value)) == 4.25
