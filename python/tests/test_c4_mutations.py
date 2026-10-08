"""2.3 C4 mutation table: what each change to the multi-source story must do.

Every row mutates one thing of the story in ``test_c4_multi_source.py`` (program, sources, law,
support, snapshot, provider, evidence overlap, utility) and asserts the expected receipt:

=============== ======================================= ==========================================
mutation        refusal / typed detail                   recalculation receipt
=============== ======================================= ==========================================
quantity        ``program_binding.*``,                   query recomputes identification, scores,
                ``external_response_binding.coordinate`` law, decision
law             ``mean_is_not_a_distribution``,          (derived stages only)
                ``joint_law_required``, ``unsupported_law``
structure       ``contract_identity_mismatch``,          graph recomputes identification chain;
                ``conflicting_atoms_not_averaged``       evidence also recomputes both priors
support         ``coordinate_unsupported``               external-study change: its provider,
                                                         prior and the decision; rest reused
data snapshot   shared snapshot overrides independence,  snapshot recomputes scores, law, decision;
                ``graph_or_snapshot_mismatch``           external branches reused
provider op     ``malformed_capability``,                provider request recomputes it and the
                ``provider_request_changed``,            decision only; unavailable in a fresh
                ``signal_provider.*``                    process
evidence        ``shared_evidence_not_independent``,     (declared by the caller)
overlap         ``swapped_evidence``, ``evsi.source_overlap``
utility         (none)                                   utility recomputes the decision only
=============== ======================================= ==========================================

Work counts come from a real :class:`~antecedent.recalc.RecalcSession` (the cross-fitted AIPW
average effect, 5 folds x 2 nuisance sets = 10 fold fits on a first run), so "no duplicate fits"
is a count and not a claim: a utility change does 0 fits, a snapshot change refits once, never
twice. The stage plans are the declared multi-source workflow whose stage identities are derived
from the story's real identities (program, claims, contract).

Evidence is never double counted: a shared registry is reported once, shared data refuses
independent pooling instead of being counted twice, and a study that reuses the prior's
observations refuses.

The planner binds external provider requests and priors to the causal program, evidence and
populations. Raw external-study identities can remain unchanged when the derived request
changes. External branch execution and provider-call counting remain unfinished; the
executed fit counts here cover the native AIPW route.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from antecedent import composition as comp
from antecedent import composition_bundle as cb
from antecedent import decision, external, program_claims, recalc
from antecedent import design as dr
from antecedent.joint_distribution import JointDistributionArtifact
from antecedent.recalc import (
    Capabilities,
    RecalcPlan,
    RecalcRequest,
    RecalcSession,
    ResumeContext,
    Stage,
    TargetWeights,
    Utility,
)

import _c4_fixtures as fx
from _refusal import assert_registered_refusal

RELATION = fx.RELATION
SNAPSHOT = fx.SNAPSHOT
NATIVE_DIGEST = fx.NATIVE_DIGEST
N = 600
SEED = 61
FIRST_RUN_FOLD_FITS = 10
BRANCHES = {
    "external_study.0",
    "provider_request.0",
    "prior.0",
    "external_study.1",
    "provider_request.1",
    "prior.1",
}


# ------------------------------------------------------------------------ stage plans


def _claim_identity(claim: external.BoundExternalClaim) -> str:
    return claim.identity


def workflow(**changes: str) -> dict[Stage, str]:
    """The declared workflow with stage identities taken from the story's real identities."""
    parts: dict[str, str] = {
        "graph": fx.spec().graph_id,
        "query": fx.program().identity,
        "regime": "do",
        "evidence": f"assumption:{fx.ASSUMPTION}",
        "source_population": "source",
        "target_population": "target",
        "data_snapshot": SNAPSHOT,
        "row_design": "rows-1",
        "treatment_grid": "0,1,2",
        "learner_folds_rng": "dml:5:seed-61",
        "utility": fx.mean_contract().identity,
        "external_study.0": _claim_identity(fx.e1_claim()),
        "external_study.1": _claim_identity(fx.e2_claim()),
        "provider_request.0": "lab-1@v1#req-e1",
        "provider_request.1": "lab-2@v1#req-e2",
        "prior.0": "prior:lab-1",
        "prior.1": "prior:lab-2",
        "identification": "v1",
        "score_artifact": "v1",
        "law": "v1",
        "decision": "v1",
    }
    parts.update(changes)
    return {Stage(label): recalc.stage_identity(label, value) for label, value in parts.items()}


def _recomputed(plan: RecalcPlan) -> set[str]:
    return {stage.value for stage in plan.recomputed}


def _plan(**changes: str) -> tuple[RecalcPlan, RecalcPlan]:
    """The plan of a mutated workflow and the plan of the unchanged one."""
    base = workflow()
    return recalc.plan_recalculation(base, workflow(**changes)), recalc.plan_recalculation(
        base, base
    )


def _assert_reused_unchanged(plan: RecalcPlan, unchanged: RecalcPlan, labels: set[str]) -> None:
    for label in labels:
        stage = Stage(label)
        assert plan.status(stage).reused, label
        assert plan.identity_of(stage) == unchanged.identity_of(stage), label


# --------------------------------------------------------------- the real native session


def columns(effect: float) -> dict[str, np.ndarray]:
    """Confounded binary treatment, an effect that varies in z, noise w and a second outcome."""
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(N)
    p = 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * z)))
    w = rng.standard_normal(N)
    t = (rng.random(N) < p).astype(np.float64)
    y = (effect + 0.8 * z) * t + z + 0.3 * rng.standard_normal(N)
    y2 = -t + 0.5 * z + 0.3 * rng.standard_normal(N)
    return {"t": t, "y": y, "z": z, "w": w, "y2": y2}


def session_request(effect: float = 2.0) -> RecalcRequest:
    return RecalcRequest(
        data=columns(effect),
        edges=(("z", "t"), ("z", "y"), ("t", "y"), ("z", "y2"), ("t", "y2")),
        treatment="t",
        outcome="y",
        utility=Utility(2.0, 0.5),
    )


def _rerun(request: RecalcRequest) -> recalc.RecalcResult:
    """An independent rerun in a brand-new session: nothing frozen or fitted is shared."""
    out = RecalcSession().execute(request, seed=SEED)
    assert out.receipt.totals.fold_fits == FIRST_RUN_FOLD_FITS
    return out


# --------------------------------------------------------------------------- quantity


def test_c4_mutation_quantity_is_refused_at_the_program_and_recomputes_the_native_chain() -> None:
    spec, program = fx.spec(), fx.program()
    override = fx.spec(
        quantities=tuple(
            replace(q, units="kPa") if i == 1 else q for i, q in enumerate(spec.quantities)
        )
    )
    with pytest.raises(external.ExternalRefusal) as units:
        program_claims.bind_to_program(override, program)
    assert units.value.detail == "program_binding.quantities_override_mismatch"
    assert units.value.reason_code == "quantity_semantics_mismatch"
    assert units.value.offending == "coordinate[1]"
    assert_registered_refusal(units.value)

    grams = replace(program, dose_units="g")
    with pytest.raises(external.ExternalRefusal) as grid:
        program_claims.bind_to_program(spec, grams)
    assert grid.value.detail == "program_binding.dose_grid_changed"
    assert grid.value.offending == "dose_units"
    with pytest.raises(external.ExternalRefusal) as native:
        fx.native_claim(fx.native_view(grid=(0.0, 1.0, 3.0)))
    assert native.value.detail == "native_claims.program_mismatch"

    wrong = tuple(replace(q, units="kPa") if i == 1 else q for i, q in enumerate(spec.quantities))
    provider = fx.provider_object("lab-1", "snap-e1", "req-e1")
    response = replace(fx.provider_response(provider, fx.E1_MEANS, "study:e1"), quantities=wrong)
    with pytest.raises(external.ExternalRefusal) as bound:
        program_claims.bind_to_program(spec, program).bind(response)
    assert bound.value.detail == "external_response_binding.coordinate_units"
    assert (bound.value.offending, bound.value.expected, bound.value.supplied) == (
        "coordinate[1]",
        "mmHg",
        "kPa",
    )

    # An action reading another unit's coordinate is unsupported on the source, not rescaled.
    kpa = replace(fx.coordinate(1.0), units="kPa")
    contract = decision.Contract(
        actions=(
            decision.Action("wait", inputs=(fx.coordinate(0.0),), utility=decision.x(0)),
            decision.Action("treat", inputs=(kpa,), utility=decision.x(0) - 1.0),
        ),
        utility_units="util",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    decided = comp.evaluate_with_support(contract, [fx.e1_input()])
    assert decided.unsupported_actions == ("treat",)
    assert decided.verdict.kind == "only_one_evaluated" and decided.verdict.selected == "wait"
    (reason,) = decided.disposition("treat").reasons
    assert reason.issue == "composition_boundary.coordinate_missing"

    plan, unchanged = _plan(query=replace(program, dose_units="g").identity)
    assert _recomputed(plan) == {
        "provider_request.0",
        "provider_request.1",
        "prior.0",
        "prior.1",
    } | {
        "query",
        "identification",
        "score_artifact",
        "law",
        "decision",
    }
    assert plan.recomputed_computations == (
        Stage.provider_request(0),
        Stage.provider_request(1),
        Stage.prior(0),
        Stage.prior(1),
        Stage.IDENTIFICATION,
        Stage.SCORE_ARTIFACT,
        Stage.LAW,
        Stage.DECISION,
    )
    assert dict(plan.table)["identification"] == "recomputed(upstream:query<-query:modified)"
    # Raw external studies remain unchanged; their derived requests and priors
    # cannot be reused under the changed causal program.
    _assert_reused_unchanged(plan, unchanged, {"external_study.0", "external_study.1"})


def test_c4_mutation_tampered_quantity_in_a_bundle_fails_the_dependent_decision() -> None:
    renamed = tuple(
        replace(c, variable_id="y2") if i == 1 else c for i, c in enumerate(fx.law_columns())
    )
    swapped = fx.composed_builder(embedded_law=fx.law(columns=renamed)).build()
    supplied = cb.SuppliedSources().with_data(SNAPSHOT, NATIVE_DIGEST)
    consumed = cb.consume_bundle(
        swapped.export(), expected_identity=swapped.identity, supplied=supplied
    )
    assert consumed.node("e2law").verified
    status = consumed.node("result_j").status
    assert isinstance(status, cb.Failed) and status.stage == "tampered_quantity"
    assert consumed.value("result_j", "treat.expected_utility") is None
    with pytest.raises(cb.TamperedQuantityRefusal) as refused:
        consumed.require_verified()
    assert refused.value.reason_code == "quantity_semantics_mismatch"


# ------------------------------------------------------------------------------- law


def test_c4_mutation_law_a_mean_a_marginal_or_another_meaning_never_stands_in() -> None:
    mean = fx.e2_mean_input()
    with pytest.raises(comp.SupportRefusal) as refused:
        comp.evaluate_functional(
            fx.mean_contract(), "treat", comp.Functional.probability(2.5, "upper"), mean
        )
    assert refused.value.detail == "composition_boundary.mean_is_not_a_distribution"

    product = decision.Contract(
        actions=(
            decision.Action("wait", inputs=(fx.coordinate(0.0, "outcome"),), utility=decision.x(0)),
            decision.Action(
                "product",
                inputs=(fx.coordinate(1.0, "outcome"), fx.coordinate(2.0, "outcome")),
                utility=decision.x(0) * decision.x(1),
            ),
        ),
        utility_units="util",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    marginals = fx.law(alignment="independent_marginals")
    with pytest.raises(decision.DecisionRefusal) as independent:
        product.evaluate(marginals)
    assert independent.value.detail == "decision_evaluation.joint_law_required"
    assert independent.value.reason_code == "joint_law_required"
    assert (independent.value.expected, independent.value.supplied) == (
        "joint",
        "independent_marginals",
    )
    # E[y1 * y2] over the joint rows is (6 * 3 + 1 * 7) / 2 = 12.5, a number marginals cannot give.
    joint = product.evaluate(fx.law())
    assert fx.near(joint.actions[1].expected_utility, 12.5)

    identity = replace(fx.law().identity, semantic="causal_functional_posterior")
    other_meaning = JointDistributionArtifact(identity, np.array(fx.LAW_ROWS, dtype=np.float64))
    with pytest.raises(decision.DecisionRefusal) as meaning:
        fx.law_contract().evaluate(other_meaning)
    assert meaning.value.reason_code == "distribution_meaning_mismatch"

    provider = fx.provider_object("lab-1", "snap-e1", "req-e1", meaning="posterior_predictive")
    with pytest.raises(external.ExternalRefusal) as bound:
        fx.bound_claim(fx.provider_response(provider, fx.E1_MEANS, "study:e1"))
    assert bound.value.reason_code == "distribution_meaning_mismatch"
    assert (bound.value.expected, bound.value.supplied) == (
        "InterventionalPredictive",
        "PosteriorPredictive",
    )

    # A mean-only claim added upstream of a joint decision that still has its aligned law
    # upstream does not invalidate the decision, but the label drops to the conservative
    # point-only one and the unrelated ranking branch stays verified. (A joint-requiring decision
    # over a mean-only claim ALONE fails as `unsupported_law`: the C3 verifier test.)
    unsupported = fx.composed_builder().connect("e2claim", "result_j").build()
    consumed = cb.consume_bundle(unsupported.export(), expected_identity=unsupported.identity)
    assert consumed.node("result_j").verified
    assert consumed.node("result_j").claim_label == "point_only_attested"
    assert consumed.node("ranking").verified


# ------------------------------------------------------------------ structural assumption


def test_c4_mutation_structure_a_changed_graph_or_premise_is_refused_and_reidentifies() -> None:
    spec, program = fx.spec(), fx.program()
    other = fx.identification(fx.EDGES + [("w", "y")], [*fx.NAMES, "w"])
    changed_spec, changed_program = fx.spec(other), fx.program(other)
    assert changed_spec.graph_id != spec.graph_id
    assert changed_program.identity != program.identity
    for call in (
        lambda: program_claims.bind_to_program(changed_spec, program),
        lambda: program_claims.bind_to_program(spec, changed_program),
    ):
        with pytest.raises(external.ExternalRefusal) as refused:
            call()
        assert refused.value.detail == "program_binding.contract_identity_mismatch"
        assert_registered_refusal(refused.value)

    premises = fx.spec(require_evidence=("factor:z",))
    with pytest.raises(external.ExternalRefusal) as premise:
        program_claims.bind_to_program(premises, program)
    assert premise.value.detail == "program_binding.contract_identity_mismatch"
    assert premise.value.offending == "contract_id"

    provider = fx.provider_object("lab-1", "snap-e1", "req-e1")
    missing = fx.provider_response(provider, fx.E1_MEANS, "study:e1", assumptions=())
    with pytest.raises(external.ExternalRefusal) as assumption:
        fx.bound_claim(missing)
    assert assumption.value.offending == fx.ASSUMPTION

    atoms = [comp.StructuralAtom("graph-a", 0.25), comp.StructuralAtom("graph-b")]
    with pytest.raises(comp.DependenceRefusal) as averaged:
        comp.check_atom_combination(atoms, "weighted_by_declared_probabilities")
    assert averaged.value.detail == "composition_boundary.conflicting_atoms_not_averaged"
    assert comp.check_atom_combination(atoms, "worst_case").kind == "worst_case"

    plan, unchanged = _plan(graph=changed_spec.graph_id)
    assert _recomputed(plan) == {
        "graph",
        "identification",
        "score_artifact",
        "law",
        "decision",
        "provider_request.0",
        "provider_request.1",
        "prior.0",
        "prior.1",
    }
    assert dict(plan.table)["identification"] == "recomputed(upstream:graph<-graph:modified)"
    _assert_reused_unchanged(plan, unchanged, {"external_study.0", "external_study.1"})

    # A changed evidence premise reaches priors and their provider requests.
    evidence_plan, evidence_unchanged = _plan(evidence="assumption:positivity")
    assert _recomputed(evidence_plan) == {
        "provider_request.0",
        "provider_request.1",
        "evidence",
        "identification",
        "prior.0",
        "prior.1",
        "score_artifact",
        "law",
        "decision",
    }
    _assert_reused_unchanged(
        evidence_plan,
        evidence_unchanged,
        {"external_study.0", "external_study.1"},
    )


# ----------------------------------------------------------------------------- support


def test_c4_mutation_support_withdrawn_from_a_study_excludes_only_the_affected_action() -> None:
    weak = (*fx.SUPPORTED[:2], "weak_overlap")
    claim = fx.e1_claim(support=(*fx.SUPPORTED[:2], fx.OUTSIDE))
    decided = comp.evaluate_with_support(fx.mean_contract(), [fx.e1_input(claim)])
    assert [o.id for o in decided.outcomes] == ["wait", "treat"]
    assert fx.near(decided.outcome("wait").expected_utility, 1.0)
    assert fx.near(decided.outcome("treat").expected_utility, 3.0 - 1.0)
    assert decided.verdict.selected == "treat"
    (reason,) = decided.disposition("extend").reasons
    assert (reason.input_id, reason.coordinate, reason.issue, reason.support) == (
        "e1",
        0,
        "composition_boundary.coordinate_unsupported",
        fx.OUTSIDE,
    )
    with pytest.raises(comp.SupportRefusal) as refused:
        comp.evaluate_with_support(
            fx.mean_contract(), [fx.e1_input(claim)], comp.SupportPolicy.require_all()
        )
    assert refused.value.detail == "composition_boundary.unsupported_action_not_comparable"

    # A weakly supported coordinate is admitted only when the policy says so.
    weak_input = fx.e1_input(fx.e1_claim(support=weak))
    strict = comp.evaluate_with_support(fx.mean_contract(), [weak_input])
    assert strict.unsupported_actions == ("extend",) and strict.verdict.selected == "treat"
    lenient = comp.evaluate_with_support(
        fx.mean_contract(), [weak_input], comp.SupportPolicy.compare_supported("weak_overlap")
    )
    assert fx.near(lenient.outcome("extend").expected_utility, 5.5 - 3.0)
    assert lenient.verdict.selected == "extend"

    # A native summary that hides a weaker coordinate is refused rather than trusted.
    hidden = fx.native_view(
        point_status=("supported", "weak_overlap", "supported"), support="supported"
    )
    with pytest.raises(external.ExternalRefusal) as summary:
        fx.native_claim(hidden)
    assert summary.value.detail == "native_claims.projection_mismatch"

    # The changed study invalidates its own branch and the decision; the rest stays valid.
    plan, unchanged = _plan(**{"external_study.0": _claim_identity(claim)})
    assert _recomputed(plan) == {"external_study.0", "provider_request.0", "prior.0", "decision"}
    assert dict(plan.table)["decision"] == "recomputed(upstream:prior.0<-external_study.0:modified)"
    assert plan.recomputed_computations == (
        Stage.provider_request(0),
        Stage.prior(0),
        Stage.DECISION,
    )
    _assert_reused_unchanged(
        plan,
        unchanged,
        {
            "external_study.1",
            "provider_request.1",
            "prior.1",
            "identification",
            "score_artifact",
            "law",
        },
    )


# ------------------------------------------------------------------------ data snapshot


def test_c4_mutation_snapshot_recomputes_dependents_and_keeps_the_external_branches() -> None:
    plan, unchanged = _plan(data_snapshot="snap-n-2")
    assert _recomputed(plan) == {"data_snapshot", "score_artifact", "law", "decision"}
    table = dict(plan.table)
    assert table["score_artifact"] == "recomputed(upstream:data_snapshot<-data_snapshot:modified)"
    assert table["decision"] == "recomputed(upstream:law<-data_snapshot:modified)"
    _assert_reused_unchanged(plan, unchanged, BRANCHES | {"identification"})

    # A study that carries the native snapshot cannot be declared independent of it.
    same = fx.e1_input(fx.e1_claim(snapshot=SNAPSHOT))
    independent = comp.PairRelation("native", "e1", comp.EvidenceRelation.independent())
    with pytest.raises(comp.DependenceRefusal) as shared:
        comp.check_composition([fx.native_input(), same], [independent], "statistical_pooling")
    assert shared.value.detail == "composition_boundary.shared_evidence_not_independent"
    receipt = comp.check_composition(
        [fx.native_input(), fx.e1_input()], [independent], "statistical_pooling"
    )
    assert receipt.independence_assumed and receipt.shared_evidence == ()

    # The bundle's reference to the native snapshot fails for another digest and keeps the
    # inspected decision unchanged.
    bundle = fx.composed_builder().build()
    changed = cb.SuppliedSources().with_data(SNAPSHOT, "another-digest")
    consumed = cb.consume_bundle(
        bundle.export(), expected_identity=bundle.identity, supplied=changed
    )
    status = consumed.node("program").status
    assert isinstance(status, cb.Failed) and status.stage == "graph_or_snapshot_mismatch"
    assert consumed.value("result_j", "treat.expected_utility") == pytest.approx(2.5, abs=1e-12)
    exact = cb.SuppliedSources().with_data(SNAPSHOT, NATIVE_DIGEST)
    cb.consume_bundle(
        bundle.export(), expected_identity=bundle.identity, supplied=exact
    ).require_verified()


def test_c4_mutation_snapshot_refits_once_and_matches_an_independent_rerun() -> None:
    session = RecalcSession()
    first = session.execute(session_request(), seed=SEED)
    assert first.receipt.totals.as_tuple() == (1, FIRST_RUN_FOLD_FITS, 1, 1, 1)

    fresh = session_request(1.5)
    second = session.execute(fresh, seed=SEED)
    assert {s.value for s in second.plan.recomputed} == {
        "data_snapshot",
        "score_artifact",
        "law",
        "decision",
    }
    totals = second.receipt.totals
    assert (totals.identifications, totals.score_computations, totals.reweights) == (0, 1, 1)
    assert totals.fold_fits == first.receipt.totals.fold_fits, "one refit, never a duplicate"
    assert second.plan.status(Stage.IDENTIFICATION).reused
    assert abs(second.law.ate - first.law.ate) > 0.05
    assert second.law.ate == pytest.approx(_rerun(fresh).law.ate, abs=1e-12)


# --------------------------------------------------------------------- provider operation


def test_c4_mutation_provider_operation_unknown_changed_or_unavailable() -> None:
    provider = fx.provider_object("lab-1", "snap-e1", "req-e1", capabilities=("mean", "teleport"))
    with pytest.raises(external.ExternalRefusal) as unknown:
        fx.bound_claim(fx.provider_response(provider, fx.E1_MEANS, "study:e1"))
    assert unknown.value.reason_code == "invalid_argument"
    assert unknown.value.detail == "external_response_binding.malformed_capability"

    # A reference that must answer one exact request refuses another and an absent provider.
    builder = cb.Bundle.builder()
    builder.add_reference(
        "provider",
        "external_claim",
        identity="claim-e1",
        requires=cb.ProviderRequirement("lab-1", "snap-e1", "req-e1"),
        facts={"law": "mean_only"},
    )
    bundle = builder.build()
    unresolved = cb.consume_bundle(bundle.export(), expected_identity=bundle.identity)
    assert isinstance(unresolved.node("provider").status, cb.ReferenceUnresolved)
    with pytest.raises(cb.CallbackUnavailableRefusal) as unavailable:
        unresolved.require_verified()
    assert unavailable.value.reason_code == "external_capability_missing"
    exact = cb.SuppliedSources().with_provider("lab-1", "snap-e1", "req-e1")
    cb.consume_bundle(
        bundle.export(), expected_identity=bundle.identity, supplied=exact
    ).require_verified()
    changed = cb.SuppliedSources().with_provider("lab-1", "snap-e1", "req-CHANGED")
    refused = cb.consume_bundle(
        bundle.export(), expected_identity=bundle.identity, supplied=changed
    )
    status = refused.node("provider").status
    assert isinstance(status, cb.Failed) and status.stage == "provider_request_changed"
    with pytest.raises(cb.ProviderRequestChangedRefusal):
        refused.require_verified()

    # The ranking's signal provider answers only the candidate, prior and law it was asked for.
    mismatched = dr.ExternalSignal(
        "lab-2",
        "trial-2",
        "v1",
        "snap-e2",
        "lab-qa",
        fx.EQUIVALENT_POSTERIOR,
        attested_candidate_id="someone-else",
    )
    with pytest.raises(dr.SignalProviderRefusal) as candidate:
        fx.rank(candidate_list=[dr.Candidate("external-trial", 2, mismatched, cost=0.05)])
    assert candidate.value.detail == "signal_provider.candidate_mismatch"
    assert candidate.value.reason_code == "design_signal_invalid"
    incoherent = dr.ExternalLaw.posterior(
        states=[0.25, 0.75],
        statistics=[0.0, 1.0, 2.0],
        predictive=[0.3125, 0.375, 0.3125],
        posterior=[[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],
    )
    with pytest.raises(dr.SignalProviderRefusal) as coherence:
        fx.rank(candidate_list=fx.candidates(law_of_signal=incoherent))
    assert coherence.value.detail == "signal_provider.posterior_incoherent"

    # A changed request recomputes that provider branch and the decision, not its prior.
    plan, unchanged = _plan(**{"provider_request.0": "lab-1@v1#req-CHANGED"})
    assert _recomputed(plan) == {"provider_request.0", "decision"}
    assert dict(plan.table)["decision"] == (
        "recomputed(upstream:provider_request.0<-provider_request.0:modified)"
    )
    _assert_reused_unchanged(plan, unchanged, {"prior.0", "external_study.0", "law", "prior.1"})

    # In a fresh process with data but no provider callback, the provider stage is unavailable.
    declared = workflow()
    without = recalc.plan_recalculation(
        declared, declared, Capabilities(resume=ResumeContext(supplied_data=True))
    )
    assert without.status(Stage.provider_request(0)).text == "refused(recalc.unavailable_provider)"
    assert not without.is_executable
    supplied = recalc.plan_recalculation(
        declared,
        declared,
        Capabilities(resume=ResumeContext(supplied_data=True, supplied_provider=True)),
    )
    assert supplied.status(Stage.provider_request(0)).text == "recomputed(fresh_process)"


def test_c4_mutation_external_study_change_leaves_unrelated_branches_valid() -> None:
    renewed = _claim_identity(fx.e2_claim(snapshot="snap-e2-new"))
    plan, unchanged = _plan(**{"external_study.1": renewed})
    assert _recomputed(plan) == {"external_study.1", "provider_request.1", "prior.1", "decision"}
    table = dict(plan.table)
    assert table["external_study.1"] == "recomputed(own:external_study.1:modified)"
    assert table["provider_request.1"] == (
        "recomputed(upstream:external_study.1<-external_study.1:modified)"
    )
    assert table["decision"] == "recomputed(upstream:prior.1<-external_study.1:modified)"
    _assert_reused_unchanged(
        plan,
        unchanged,
        {
            "external_study.0",
            "provider_request.0",
            "prior.0",
            "identification",
            "score_artifact",
            "law",
        },
    )
    assert plan.explain().count("recomputed:") >= 4


# ---------------------------------------------------------------------- evidence overlap


def _shared(left: str, right: str, evidence: str = "registry-7") -> comp.PairRelation:
    return comp.PairRelation(left, right, comp.EvidenceRelation.shared_data(evidence))


def test_c4_mutation_overlap_is_never_pooled_or_counted_twice() -> None:
    inputs = [fx.native_input(), fx.e1_input(), fx.e2_law_input()]
    relations = [
        _shared("native", "e1"),
        _shared("native", "e2law"),
        comp.PairRelation("e1", "e2law", comp.EvidenceRelation.independent()),
    ]
    for operation in ("statistical_pooling", "bayesian_borrowing"):
        with pytest.raises(comp.DependenceRefusal) as refused:
            comp.check_composition(inputs, relations, operation)
        assert refused.value.detail == "composition_boundary.shared_evidence_not_independent"
        assert refused.value.offending == "native~e1"
        assert_registered_refusal(refused.value)
    # Declared reuse and transport report the registry once, however many pairs share it.
    for operation in ("evidence_reuse", "causal_transport"):
        receipt = comp.check_composition(inputs, relations, operation)
        assert receipt.shared_evidence == ("registry-7",), operation
        assert not receipt.independence_assumed
    # A shared prior is borrowing, not pooling.
    prior = [comp.PairRelation("e1", "e2law", comp.EvidenceRelation.shared_prior("prior-7"))]
    pair = [fx.e1_input(), fx.e2_law_input()]
    with pytest.raises(comp.DependenceRefusal):
        comp.check_composition(pair, prior, "statistical_pooling")
    borrowed = comp.check_composition(pair, prior, "bayesian_borrowing")
    assert borrowed.shared_evidence == ("prior-7",) and not borrowed.independence_assumed
    # A shared fitted model refuses both statistical operations.
    model = [comp.PairRelation("e1", "e2law", comp.EvidenceRelation.shared_fitted_model("fit-3"))]
    for operation in ("statistical_pooling", "bayesian_borrowing"):
        with pytest.raises(comp.DependenceRefusal):
            comp.check_composition(pair, model, operation)

    # The bundle's declared independence is checked against the studies' own evidence.
    overlapping = fx.composed_builder(e2_evidence="study:e1").build()
    supplied = cb.SuppliedSources().with_data(SNAPSHOT, NATIVE_DIGEST)
    consumed = cb.consume_bundle(
        overlapping.export(), expected_identity=overlapping.identity, supplied=supplied
    )
    status = consumed.node(RELATION).status
    assert isinstance(status, cb.Failed) and status.stage == "swapped_evidence"
    with pytest.raises(cb.SwappedEvidenceRefusal):
        consumed.require_verified()
    assert consumed.node("result_j").verified, "the decision's own branch is untouched"

    # A candidate study that reuses the prior's observations would count them twice.
    with pytest.raises(dr.SourceOverlapRefusal) as overlap:
        fx.rank(candidate_list=fx.candidates(external_reuses=("obs:e2-registry",)))
    assert overlap.value.detail == "evsi.source_overlap"
    assert overlap.value.offending == "obs:e2-registry"
    clean = fx.rank(candidate_list=fx.candidates(external_reuses=("obs:future",)))
    assert clean.candidate("external-trial").source_overlap.overlapping == ()


# --------------------------------------------------------------------------- utility


def test_c4_mutation_utility_recomputes_only_the_decision_with_no_fit_work() -> None:
    base = fx.mean_contract()
    shifted = replace(
        base,
        actions=(
            base.actions[0],
            replace(base.actions[1], utility=decision.x(0) - 2.0),
            base.actions[2],
        ),
    )
    assert shifted.identity != base.identity
    plan, unchanged = _plan(utility=shifted.identity)
    assert _recomputed(plan) == {"utility", "decision"}
    assert plan.recomputed_computations == (Stage.DECISION,)
    assert dict(plan.table)["decision"] == "recomputed(upstream:utility<-utility:modified)"
    _assert_reused_unchanged(
        plan, unchanged, BRANCHES | {"identification", "score_artifact", "law"}
    )

    # Same sources, new utility: treat is now worth 4 - 2 = 2, so extend (E1: 5.5 - 3) wins.
    again = comp.evaluate_with_support(shifted, [fx.native_input(), fx.e1_input()])
    assert fx.near(again.outcome("treat").expected_utility, 2.0)
    assert fx.near(again.outcome("extend").expected_utility, 2.5)
    assert again.verdict.selected == "extend"
    original = comp.evaluate_with_support(base, [fx.native_input(), fx.e1_input()])
    assert original.verdict.selected == "treat"

    # The real session counts it: no identification, no fold fit, no score, no reweight.
    request = session_request()
    session = RecalcSession()
    first = session.execute(request, seed=SEED)
    second = session.execute(replace(request, utility=Utility(3.0, 0.1)), seed=SEED)
    assert second.receipt.totals.as_tuple() == (0, 0, 0, 0, 1)
    assert second.plan.recomputed_computations == (Stage.DECISION,)
    assert second.law == first.law
    assert second.decision.net_benefit == pytest.approx(3.0 * first.law.ate - 0.1, abs=1e-12)


def test_c4_mutation_target_weights_reuse_frozen_scores_and_the_dose_grid_refuses_off_grid() -> (
    None
):
    request = session_request()
    session = RecalcSession()
    first = session.execute(request, seed=SEED)
    z = np.asarray(request.data["z"])
    changed = replace(request, target=TargetWeights(np.exp(0.4 * z), ("z",)))
    retargeted = session.execute(changed, seed=SEED)
    assert retargeted.receipt.totals.as_tuple() == (0, 0, 0, 1, 1)
    assert {s.value for s in retargeted.plan.recomputed} == {"target_population", "law", "decision"}
    assert abs(retargeted.law.ate - first.law.ate) > 0.05
    assert retargeted.law.ate == pytest.approx(_rerun(changed).law.ate, abs=1e-12)

    cold = RecalcSession()
    cold.set_request_support("off_grid", "transport.smoothed_dose")
    with pytest.raises(recalc.RecalcRefusal) as off_grid:
        cold.execute(request, seed=SEED)
    assert (off_grid.value.stage, off_grid.value.detail) == (
        "treatment_grid",
        "recalc.off_grid_request",
    )
    assert off_grid.value.plan is not None
    assert off_grid.value.plan.status(Stage.TREATMENT_GRID).text == (
        "refused(recalc.off_grid_request[transport.smoothed_dose])"
    )
    assert not cold.is_live
