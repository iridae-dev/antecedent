"""C1: the composition boundary through the Python surface.

Oracles are hand-derived and mirror ``crates/antecedent-design/tests/composition_boundary.rs``.

The joint fixture has four equally weighted draws over ``do(a=0)``, ``do(a=1)`` and
``do(a=2)`` with ``x0 = [1, 2, 3, 2]`` (mean 2) and ``x1 = [4, 0, 2, 6]`` (mean 3), so with
identity utilities ``E[U(wait)] = 2``, ``E[U(treat)] = 3``, ``E[max] = (4 + 2 + 3 + 6) / 4 =
3.75`` and ``EVPI = 3.75 - 3 = 0.75``; ``P(x1 >= 2) = 3 / 4``. The mean fixture has
``E[Y | do(a=0)] = 1`` and ``E[Y | do(a=1)] = 3`` under the affine utility ``2x - 1``, so
``wait = 1`` and ``treat = 5``.

Python cannot assert a native provider: every input here is ``unverified`` or externally
attested, and a ``native`` requirement refuses a label that has no execution record behind it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from antecedent import composition as comp
from antecedent import decision
from antecedent.errors import CausalTypeError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)

from _refusal import assert_registered_refusal


def q(regime: str, functional: str = "outcome") -> ScientificQuantity:
    return ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units="units",
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id=functional,
    )


def joint(
    provider: str = "engine",
    snapshot: str = "snap-n",
    *,
    trust: Any = "unverified",
    calibration: Any = "unmeasured",
    supported: tuple[bool, ...] | None = None,
) -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=(q("do(a=0)"), q("do(a=1)"), q("do(a=2)")),
        alignment="joint",
        source_id=f"study-{snapshot}",
        provider_id=provider,
        rng_id="deterministic_exact",
        snapshot_id=snapshot,
        causal_contract_id="checked-contract",
    )
    x0 = [1.0, 2.0, 3.0, 2.0]
    x1 = [4.0, 0.0, 2.0, 6.0]
    draws = np.array([[a, b, 0.0] for a, b in zip(x0, x1, strict=True)], dtype=np.float64)
    if trust == "native_licensed":
        # Negative compatibility case: a historical caller-labelled artifact.
        # The public constructor now refuses such labels before copying rows.
        assert provider == "engine" and snapshot == "snap-n"
        assert calibration == "exact" and supported is None
        path = (
            Path(__file__).resolve().parents[2]
            / "conformance/composition/metadata_only_native/legacy.art"
        )
        return JointDistributionArtifact.load(path.read_bytes(), expected_identity=identity)
    return JointDistributionArtifact(
        identity, draws, supported=supported, calibration=calibration, trust=trust
    )


LABELLED_NATIVE: dict[str, Any] = {"trust": "native_licensed", "calibration": "exact"}


def joint_input(
    input_id: str,
    provider: str = "engine",
    snapshot: str = "snap-n",
    evidence: comp.TrustEvidence | None = None,
    **labels: Any,
) -> comp.DecisionInput:
    return comp.DecisionInput.from_distribution(
        input_id, joint(provider, snapshot, **labels), evidence=evidence
    )


def mean_input(
    input_id: str,
    snapshot: str,
    coordinates: list[tuple[str, float]],
    support: Any = "supported",
) -> comp.DecisionInput:
    return comp.DecisionInput.from_means(
        input_id,
        [q(regime, "mean") for regime, _ in coordinates],
        [mean for _, mean in coordinates],
        provider_id="lab",
        snapshot_id=snapshot,
        causal_contract_id="checked-contract",
        support=support,
        evidence=comp.TrustEvidence.attested("lab"),
    )


def action(action_id: str, regime: str, functional: str, utility: decision.Expr) -> decision.Action:
    return decision.Action(action_id, inputs=(q(regime, functional),), utility=utility)


def contract(
    actions: list[decision.Action], criterion: decision.Criterion | None = None
) -> decision.Contract:
    return decision.Contract(
        actions=tuple(actions),
        utility_units="utility",
        criterion=criterion or decision.Criterion.expected_utility(),
        target_population="target",
    )


def affine() -> decision.Expr:
    return decision.x(0) * 2.0 - 1.0


def mean_contract() -> decision.Contract:
    return contract(
        [
            action("wait", "do(a=0)", "mean", affine()),
            action("treat", "do(a=1)", "mean", affine()),
            action("extend", "do(a=2)", "mean", affine()),
        ]
    )


def joint_contract() -> decision.Contract:
    return contract(
        [
            action("wait", "do(a=0)", "outcome", decision.x(0)),
            action("treat", "do(a=1)", "outcome", decision.x(0)),
            action("extend", "do(a=2)", "outcome", decision.x(0)),
        ]
    )


def near(value: float | None, expected: float) -> bool:
    return value is not None and abs(value - expected) < 1e-12


# ---------------------------------------------------------------- trust carriage


def test_c1_boundary_caller_native_constructor_refuses_before_coercion() -> None:
    from antecedent.errors import CausalValueError

    # A poison object proves refusal precedes NumPy coercion or data access.
    class UnreadableDraws:
        def __array__(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("native claim refusal must precede array coercion")

    identity = joint().identity
    with pytest.raises(CausalValueError) as refused:
        JointDistributionArtifact(identity, UnreadableDraws(), trust="native_licensed")  # type: ignore[arg-type]
    assert refused.value.reason_code == "invalid_argument"
    assert "native_distribution.authority_required" in str(refused.value)


def test_c1_boundary_metadata_only_exact_artifact_is_refused_as_native() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "conformance/composition/metadata_only_native/legacy.art"
    )
    artifact = JointDistributionArtifact.load(path.read_bytes(), expected_identity=joint().identity)
    metadata = json.loads(artifact._native.metadata_json)
    assert metadata["trust"] == "native_licensed"
    assert metadata["calibration"] == "exact"
    np.testing.assert_array_equal(
        np.asarray(artifact), [[1.0, 4.0, 0.0], [2.0, 0.0, 0.0], [3.0, 2.0, 0.0], [2.0, 6.0, 0.0]]
    )
    assert artifact.trust == "native_licensed" and artifact.calibration == "exact"
    with pytest.raises(comp.UnverifiedTrustRefusal) as refused:
        comp.DecisionInput.from_distribution("native", artifact, requirement="native")
    assert refused.value.detail == "composition_boundary.metadata_only_native_claim"
    assert refused.value.reason_code == "attested_not_reverifiable"
    assert refused.value.offending == "native"
    assert_registered_refusal(refused.value)

    # An external attestation is not a native execution either.
    with pytest.raises(comp.UnverifiedTrustRefusal) as attested:
        comp.DecisionInput.from_distribution(
            "native",
            artifact,
            evidence=comp.TrustEvidence.attested("lab"),
            requirement="native",
        )
    assert attested.value.detail == "composition_boundary.metadata_only_native_claim"

    # An artifact that does not even claim native still is not native.
    with pytest.raises(comp.UnverifiedTrustRefusal) as plain:
        comp.DecisionInput.from_distribution("plain", joint(), requirement="native")
    assert plain.value.detail == "composition_boundary.native_required"


def test_c1_boundary_unbacked_label_is_stored_unverified_and_its_exact_claim_unused() -> None:
    labelled = comp.DecisionInput.from_distribution("native", joint(**LABELLED_NATIVE))
    provenance = labelled.provenance
    assert provenance.provider_kind == "external_attested"
    assert provenance.trust == "unverified"
    assert provenance.receipt is None
    assert provenance.calibration == "exact_claim_unverified"
    assert not provenance.native and not labelled.native
    assert labelled.source == "joint_law"
    assert {"sample", "mean", "covariance"} <= set(provenance.capabilities)

    # Attestation records who asserted, and nothing stronger.
    attested = comp.DecisionInput.from_distribution(
        "native",
        joint(**LABELLED_NATIVE),
        evidence=comp.TrustEvidence.attested("lab"),
    )
    assert attested.provenance.trust == "external_attested"
    assert attested.provenance.receipt == {"kind": "attestation", "attestor": "lab"}
    assert not attested.native
    assert attested.provenance.lineage_digest != provenance.lineage_digest


def test_c1_boundary_python_cannot_assert_a_native_or_verified_provider() -> None:
    assert comp.TrustEvidence.none().kind == "none"
    assert comp.TrustEvidence.attested("lab").kind == "externally_attested"
    # Only library-produced evidence is accepted, and no constructor names native.
    with pytest.raises(CausalTypeError):
        comp.TrustEvidence("native")  # type: ignore[arg-type]
    assert not hasattr(comp.TrustEvidence, "native")
    assert not hasattr(comp.TrustEvidence, "verified")
    with pytest.raises(CausalTypeError):
        comp.DecisionInput("native")  # type: ignore[arg-type]


def test_c1_boundary_exact_request_verification_needs_a_matching_receipt() -> None:
    claimed = joint("lab", "snap-x", trust="verified_extension", calibration="exact")
    for evidence in (None, comp.TrustEvidence.none(), comp.TrustEvidence.attested("lab")):
        with pytest.raises(comp.UnverifiedTrustRefusal) as refused:
            comp.DecisionInput.from_distribution(
                "ext", claimed, evidence=evidence, requirement="exact_request_verified"
            )
        assert refused.value.detail == "composition_boundary.verification_receipt_missing"
        assert refused.value.reason_code == "external_verification_failed"
        assert_registered_refusal(refused.value)


# ----------------------------------------------------------- per-action support


def test_c1_boundary_one_missing_coordinate_beside_a_supported_action() -> None:
    grid = mean_input("grid", "snap-x", [("do(a=0)", 1.0), ("do(a=1)", 3.0)])
    decided = comp.evaluate_with_support(mean_contract(), [grid])
    assert [o.id for o in decided.outcomes] == ["wait", "treat"]
    assert near(decided.outcome("wait").expected_utility, 1.0)
    assert near(decided.outcome("treat").expected_utility, 5.0)
    assert decided.verdict.kind == "uniquely_optimal" and decided.verdict.selected == "treat"
    assert decided.verdict.compared
    assert decided.evpi is None
    assert decided.evaluated_actions == ("wait", "treat")
    assert decided.unsupported_actions == ("extend",)
    extend = decided.disposition("extend")
    assert extend.status == "unsupported" and extend.input_id is None
    assert extend.reasons == (
        comp.UnsupportedReason("grid", 0, "composition_boundary.coordinate_missing", None),
    )
    assert decided.disposition("wait").input_id == "grid"

    # When the policy demands every action, the same inputs refuse.
    with pytest.raises(comp.SupportRefusal) as refused:
        comp.evaluate_with_support(mean_contract(), [grid], comp.SupportPolicy.require_all())
    assert refused.value.detail == "composition_boundary.unsupported_action_not_comparable"
    assert refused.value.offending == "extend"
    assert_registered_refusal(refused.value)


def test_c1_boundary_one_masked_coordinate_beside_a_supported_action_joint_source() -> None:
    law = joint_input("law", supported=(True, True, False))
    decided = comp.evaluate_with_support(joint_contract(), [law])
    assert near(decided.outcome("wait").expected_utility, 2.0)
    assert near(decided.outcome("treat").expected_utility, 3.0)
    assert decided.verdict.kind == "uniquely_optimal" and decided.verdict.selected == "treat"
    assert near(decided.evpi, 0.75)
    assert decided.sources == 1
    (reason,) = decided.disposition("extend").reasons
    assert reason.input_id == "law"
    assert reason.issue == "composition_boundary.coordinate_unsupported"
    assert reason.support == "outside_empirical_support"
    with pytest.raises(comp.SupportRefusal):
        comp.evaluate_with_support(joint_contract(), [law], comp.SupportPolicy.require_all())


def test_c1_boundary_a_missing_evidence_status_excludes_only_that_action() -> None:
    grid = mean_input(
        "grid", "snap-x", [("do(a=0)", 1.0), ("do(a=1)", 3.0)], ["supported", "missing_evidence"]
    )
    decided = comp.evaluate_with_support(mean_contract(), [grid])
    # Only `wait` is left: it is reported, but nothing was compared.
    assert decided.verdict.kind == "only_one_evaluated" and decided.verdict.selected == "wait"
    assert not decided.verdict.compared
    assert [o.id for o in decided.outcomes] == ["wait"]
    (reason,) = decided.disposition("treat").reasons
    assert reason.support == "missing_evidence"
    assert decided.disposition("treat").status == "unsupported"


def test_c1_boundary_weak_overlap_is_admitted_only_when_the_policy_allows_it() -> None:
    def make() -> comp.DecisionInput:
        return mean_input(
            "grid", "snap-x", [("do(a=0)", 1.0), ("do(a=1)", 3.0)], ["weak_overlap", "supported"]
        )

    strict = comp.evaluate_with_support(mean_contract(), [make()])
    assert strict.verdict.kind == "only_one_evaluated" and strict.verdict.selected == "treat"
    lenient = comp.evaluate_with_support(
        mean_contract(), [make()], comp.SupportPolicy.compare_supported("weak_overlap")
    )
    assert lenient.verdict.kind == "uniquely_optimal" and lenient.verdict.selected == "treat"
    assert len(lenient.outcomes) == 2


def test_c1_boundary_all_actions_unsupported_is_a_state_not_an_error() -> None:
    grid = mean_input("grid", "snap-x", [("do(a=9)", 7.0)])
    decided = comp.evaluate_with_support(mean_contract(), [grid])
    assert decided.verdict.kind == "no_supported_action"
    assert decided.verdict.selected is None and decided.verdict.actions == ()
    assert decided.outcomes == () and decided.evpi is None and decided.sources == 0
    assert len(decided.dispositions) == 3
    assert all(d.status == "unsupported" and d.input_id is None for d in decided.dispositions)

    # The same holds when every coordinate exists but is masked.
    masked = joint_input("law", supported=(False, False, False))
    assert comp.evaluate_with_support(joint_contract(), [masked]).verdict.kind == (
        "no_supported_action"
    )

    # Demanding every action turns it into a refusal.
    with pytest.raises(comp.SupportRefusal) as refused:
        comp.evaluate_with_support(mean_contract(), [grid], comp.SupportPolicy.require_all())
    assert refused.value.offending == "wait,treat,extend"


def test_c1_boundary_actions_answered_by_different_inputs_compare_by_value_only() -> None:
    first = mean_input("first", "snap-1", [("do(a=0)", 1.0)])
    second = mean_input("second", "snap-2", [("do(a=1)", 3.0)])
    decided = comp.evaluate_with_support(mean_contract(), [first, second])
    assert decided.verdict.kind == "uniquely_optimal" and decided.verdict.selected == "treat"
    assert decided.disposition("wait").input_id == "first"
    assert decided.disposition("treat").input_id == "second"
    assert decided.evpi is None
    assert all(o.expected_regret is None and o.max_regret is None for o in decided.outcomes)

    # A state-aligned criterion cannot span two sources.
    regret = contract(
        [
            action("wait", "do(a=0)", "mean", affine()),
            action("treat", "do(a=1)", "mean", affine()),
        ],
        decision.Criterion.expected_regret(),
    )
    with pytest.raises(comp.DependenceRefusal) as refused:
        comp.evaluate_with_support(regret, [first, second])
    assert refused.value.detail == "composition_boundary.paired_draws_across_sources"
    assert refused.value.reason_code == "joint_law_required"


# ------------------------------------------------ mean versus outcome law versus joint


def test_c1_boundary_a_probability_is_never_answered_from_a_mean() -> None:
    mean_only = contract(
        [
            action("wait", "do(a=0)", "mean", decision.x(0)),
            action("treat", "do(a=1)", "mean", decision.x(0)),
        ]
    )
    grid = mean_input("grid", "snap-x", [("do(a=0)", 1.0), ("do(a=1)", 3.0)])
    value = comp.evaluate_functional(mean_only, "treat", comp.Functional.expectation(), grid)
    assert near(value.value, 3.0) and value.standard_error is None
    for functional in (
        comp.Functional.probability(2.0, "upper"),
        comp.Functional.quantile(0.5),
        comp.Functional.variance(),
        comp.Functional.tail_expectation(0.5, "lower"),
    ):
        with pytest.raises(comp.SupportRefusal) as refused:
            comp.evaluate_functional(mean_only, "treat", functional, grid)
        assert refused.value.detail == "composition_boundary.mean_is_not_a_distribution"
        assert refused.value.offending == "treat"

    # A scalar claim is a mean too.
    scalar = comp.DecisionInput.from_scalar(
        "scalar",
        q("do(a=1)", "mean"),
        3.0,
        provider_id="lab",
        snapshot_id="snap-s",
        causal_contract_id="checked-contract",
        evidence=comp.TrustEvidence.attested("lab"),
    )
    assert scalar.source == "scalar"
    with pytest.raises(comp.SupportRefusal):
        comp.evaluate_functional(
            mean_only, "treat", comp.Functional.probability(2.0, "upper"), scalar
        )


def test_c1_boundary_an_aligned_joint_law_answers_the_probability_a_mean_cannot() -> None:
    outcome_only = contract(
        [
            action("wait", "do(a=0)", "outcome", decision.x(0)),
            action("treat", "do(a=1)", "outcome", decision.x(0)),
        ]
    )
    unbacked = comp.DecisionInput.from_distribution("law", joint(**LABELLED_NATIVE))
    # P(x1 >= 2) over [4, 0, 2, 6] is 3 / 4. The unbacked exact label is not used.
    probability = comp.evaluate_functional(
        outcome_only, "treat", comp.Functional.probability(2.0, "upper"), unbacked
    )
    assert near(probability.value, 0.75)
    expectation = comp.evaluate_functional(
        outcome_only, "treat", comp.Functional.expectation(), unbacked
    )
    assert near(expectation.value, 3.0)
    assert expectation.standard_error is None, "no exactness without a native record"


def test_c1_boundary_mean_sources_leave_nonlinear_and_outcome_law_actions_unevaluated() -> None:
    grid = mean_input("grid", "snap-x", [("do(a=0)", 1.0), ("do(a=1)", 3.0)])
    risky = decision.Action(
        "risky",
        inputs=(q("do(a=0)", "mean"), q("do(a=1)", "mean")),
        utility=decision.x(0) * decision.x(1),
    )
    mixed = contract(
        [
            action("wait", "do(a=0)", "mean", decision.x(0)),
            risky,
            action("law", "do(a=0)", "outcome", decision.x(0)),
        ]
    )
    decided = comp.evaluate_with_support(mixed, [grid])
    assert decided.verdict.kind == "only_one_evaluated" and decided.verdict.selected == "wait"
    nonlinear = decided.disposition("risky")
    assert nonlinear.status == "unevaluated"
    assert nonlinear.unevaluated_reason == "composition_boundary.non_affine_needs_joint_law"
    # The outcome-law coordinate is absent from a mean grid: unsupported, never read as a mean.
    assert decided.disposition("law").status == "unsupported"
    assert decided.unevaluated_actions == ("risky",)
    with pytest.raises(comp.SupportRefusal):
        comp.evaluate_with_support(mixed, [grid], comp.SupportPolicy.require_all())

    # A mean supplied for an outcome-law coordinate is present but cannot answer it.
    labelled = comp.DecisionInput.from_means(
        "outcome-grid",
        [q("do(a=0)", "outcome"), q("do(a=1)", "mean")],
        [1.0, 3.0],
        provider_id="lab",
        snapshot_id="snap-y",
        causal_contract_id="checked-contract",
        evidence=comp.TrustEvidence.attested("lab"),
    )
    law_and_treat = contract(
        [
            action("law", "do(a=0)", "outcome", decision.x(0)),
            action("treat", "do(a=1)", "mean", decision.x(0)),
        ]
    )
    decided = comp.evaluate_with_support(law_and_treat, [labelled])
    assert decided.verdict.selected == "treat"
    assert decided.disposition("law").unevaluated_reason == (
        "composition_boundary.outcome_law_needs_joint_law"
    )


def test_c1_boundary_a_mean_source_cannot_supply_paired_draws() -> None:
    with pytest.raises(comp.CompositionRefusal) as refused:
        comp.check_paired_draws(
            [joint_input("native"), mean_input("grid", "snap-x", [("do(a=0)", 1.0)])], []
        )
    assert refused.value.reason_code == "joint_law_required"
    assert_registered_refusal(refused.value)


# ----------------------------------------------------- source overlap and dependence


def shared_data(
    left: str, right: str, route: comp.DependenceRoute | None = None
) -> comp.PairRelation:
    return comp.PairRelation(left, right, comp.EvidenceRelation.shared_data("trial-1"), route)


def native_and_external_mean() -> list[comp.DecisionInput]:
    return [
        joint_input("native"),
        mean_input("external", "snap-x", [("do(a=0)", 1.0)]),
    ]


def test_c1_boundary_shared_data_refuses_independent_pooling() -> None:
    with pytest.raises(comp.DependenceRefusal) as refused:
        comp.check_composition(
            native_and_external_mean(),
            [shared_data("native", "external")],
            "statistical_pooling",
        )
    assert refused.value.detail == "composition_boundary.shared_evidence_not_independent"
    assert refused.value.reason_code == "scenario_aggregate_not_licensed"
    assert refused.value.offending == "native~external"
    assert_registered_refusal(refused.value)


def test_c1_boundary_shared_data_refuses_paired_draws_without_a_licensed_route() -> None:
    pair = [joint_input("native"), joint_input("external", "lab", "snap-x")]
    with pytest.raises(comp.DependenceRefusal):
        comp.check_paired_draws(pair, [shared_data("native", "external")])
    # The order of the pair does not matter.
    with pytest.raises(comp.DependenceRefusal):
        comp.check_paired_draws(pair, [shared_data("external", "native")])

    route = comp.DependenceRoute("joint_law", "joint-law-1", "native")
    receipt = comp.check_paired_draws(pair, [shared_data("native", "external", route)])
    assert receipt.operation == "statistical_pooling"
    assert not receipt.independence_assumed
    assert receipt.routes == ("joint-law-1",)
    assert receipt.shared_evidence == ("trial-1",)

    covariance = comp.DependenceRoute("covariance", "cov-1", "external")
    receipt = comp.check_paired_draws(pair, [shared_data("native", "external", covariance)])
    assert receipt.routes == ("cov-1",)


def test_c1_boundary_a_route_must_be_an_aligned_joint_law_of_the_pair() -> None:
    inputs = native_and_external_mean()
    for route in (
        comp.DependenceRoute("covariance", "cov-1", "external"),  # a mean has no covariance
        comp.DependenceRoute("joint_law", "joint-law-1", "elsewhere"),  # not one of the pair
        comp.DependenceRoute("joint_law", " ", "native"),  # a route needs an identity
    ):
        with pytest.raises(comp.DependenceRefusal) as refused:
            comp.check_composition(
                inputs, [shared_data("native", "external", route)], "statistical_pooling"
            )
        assert refused.value.detail == "composition_boundary.dependence_route_not_licensed"
    ok = comp.DependenceRoute("joint_law", "joint-law-1", "native")
    comp.check_composition(inputs, [shared_data("native", "external", ok)], "statistical_pooling")


def test_c1_boundary_declared_independence_is_overridden_by_a_shared_snapshot() -> None:
    first = mean_input("first", "same-snapshot", [("do(a=0)", 1.0)])
    second = mean_input("second", "same-snapshot", [("do(a=0)", 1.0)])
    relation = comp.PairRelation("first", "second", comp.EvidenceRelation.independent())
    with pytest.raises(comp.DependenceRefusal) as refused:
        comp.check_composition([first, second], [relation], "statistical_pooling")
    assert refused.value.detail == "composition_boundary.shared_evidence_not_independent"


def test_c1_boundary_unknown_dependence_is_never_independence() -> None:
    inputs = native_and_external_mean()
    unknown = comp.PairRelation("native", "external", comp.EvidenceRelation.unknown())
    for operation in comp.OPERATIONS:
        with pytest.raises(comp.DependenceRefusal) as refused:
            comp.check_composition(inputs, [unknown], operation)
        assert refused.value.detail == "composition_boundary.unknown_dependence_is_not_independence"
        assert refused.value.reason_code == "sampling_dependence_unknown"
        assert_registered_refusal(refused.value)
        # A pair that was never declared is the same as unknown.
        with pytest.raises(comp.DependenceRefusal) as silent:
            comp.check_composition(inputs, [], operation)
        assert silent.value.detail == refused.value.detail


def test_c1_boundary_an_undeclared_operation_is_refused() -> None:
    inputs = native_and_external_mean()
    independent = comp.PairRelation("native", "external", comp.EvidenceRelation.independent())
    with pytest.raises(comp.DependenceRefusal) as refused:
        comp.check_composition(inputs, [independent])
    assert refused.value.detail == "composition_boundary.operation_not_declared"
    # Independent sources pool, and pooling records that independence was assumed.
    receipt = comp.check_composition(inputs, [independent], "statistical_pooling")
    assert receipt.independence_assumed and receipt.routes == () and receipt.shared_evidence == ()

    # Malformed declarations are refused rather than guessed at.
    stranger = comp.PairRelation("native", "ghost", comp.EvidenceRelation.independent())
    with pytest.raises(comp.CompositionRefusal) as ghost:
        comp.check_composition(inputs, [stranger], "statistical_pooling")
    assert ghost.value.detail == "composition_boundary.unknown_input"
    with pytest.raises(comp.CompositionRefusal) as alone:
        comp.check_composition(inputs[:1], [], "statistical_pooling")
    assert alone.value.detail == "composition_boundary.invalid_input"


def test_c1_boundary_operations_are_declared_separately() -> None:
    inputs = native_and_external_mean()

    def relation(kind: comp.EvidenceRelation) -> list[comp.PairRelation]:
        return [comp.PairRelation("native", "external", kind)]

    # A shared prior is what Bayesian borrowing is; pooling would count it twice.
    prior = relation(comp.EvidenceRelation.shared_prior("prior-7"))
    with pytest.raises(comp.DependenceRefusal):
        comp.check_composition(inputs, prior, "statistical_pooling")
    borrowed = comp.check_composition(inputs, prior, "bayesian_borrowing")
    assert not borrowed.independence_assumed and borrowed.shared_evidence == ("prior-7",)
    assert comp.check_composition(inputs, prior, "causal_transport").operation == "causal_transport"
    assert comp.check_composition(inputs, prior, "evidence_reuse").operation == "evidence_reuse"

    # Shared data is double counting for borrowing and pooling, not for transport or reuse.
    statistical: tuple[comp.Operation, ...] = ("statistical_pooling", "bayesian_borrowing")
    data = relation(comp.EvidenceRelation.shared_data("trial-2", "trial-1"))
    for operation in statistical:
        with pytest.raises(comp.DependenceRefusal):
            comp.check_composition(inputs, data, operation)
    assert not comp.check_composition(inputs, data, "causal_transport").independence_assumed
    reused = comp.check_composition(inputs, data, "evidence_reuse")
    assert reused.shared_evidence == ("trial-1", "trial-2")

    # A shared fitted model is refused by both statistical operations.
    model = relation(comp.EvidenceRelation.shared_fitted_model("fit-3"))
    for operation in statistical:
        with pytest.raises(comp.DependenceRefusal):
            comp.check_composition(inputs, model, operation)
    comp.check_composition(inputs, model, "evidence_reuse")


# ------------------------------------------------------------ conflicting atoms


def test_c1_boundary_conflicting_atoms_never_become_a_silent_average() -> None:
    atoms = [comp.StructuralAtom("graph-a"), comp.StructuralAtom("graph-b")]
    with pytest.raises(comp.DependenceRefusal) as refused:
        comp.check_atom_combination(atoms, "weighted_by_declared_probabilities")
    assert refused.value.detail == "composition_boundary.conflicting_atoms_not_averaged"
    assert refused.value.reason_code == "scenario_aggregate_not_licensed"

    # Probabilities on only some atoms are still not a license to average.
    partial = [comp.StructuralAtom("graph-a", 0.5), comp.StructuralAtom("graph-b")]
    with pytest.raises(comp.DependenceRefusal):
        comp.check_atom_combination(partial, "weighted_by_declared_probabilities")

    # Reporting each atom and the worst case need no probability.
    assert comp.check_atom_combination(atoms, "report_each").kind == "report_each"
    assert comp.check_atom_combination(atoms, "worst_case").kind == "worst_case"

    # Declared probabilities license the weighted plan; missing mass is not renormalized.
    declared = [comp.StructuralAtom("graph-a", 0.25), comp.StructuralAtom("graph-b", 0.5)]
    plan = comp.check_atom_combination(declared, "weighted_by_declared_probabilities")
    assert plan.kind == "weighted" and plan.weights == (("graph-a", 0.25), ("graph-b", 0.5))

    # Invalid probabilities and repeated identities refuse.
    over = [comp.StructuralAtom("graph-a", 0.75), comp.StructuralAtom("graph-b", 0.5)]
    with pytest.raises(comp.DependenceRefusal) as invalid:
        comp.check_atom_combination(over, "weighted_by_declared_probabilities")
    assert invalid.value.detail == "composition_boundary.atom_probabilities_invalid"
    with pytest.raises(comp.CompositionRefusal) as repeated:
        comp.check_atom_combination([comp.StructuralAtom("graph-a")] * 2, "report_each")
    assert repeated.value.detail == "composition_boundary.invalid_input"
