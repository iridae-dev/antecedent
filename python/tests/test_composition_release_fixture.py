"""Release-boundary composition fixture, through public Python APIs only.

One scenario whose numbers are checked by hand (``conformance/composition/release_boundary``):

* a binary exposure with a declared confusion model (Se 0.9, Sp 0.8) is recorded as exposed for
  410 of 1000, so the corrected prevalence is ``(0.41 + 0.8 - 1) / 0.7 = 0.3``;
* two disjoint studies report a risk difference of 0.10 and 0.20 (variance 0.0004 each): the pool
  is 0.15 with variance 0.0002;
* with baseline risk 0.10 the composed finite model has ``P(Y=1 | do(X=1)) = 0.25`` and
  ``P(Y=1) = 0.145``, so the excess risk is ``0.145 - 0.10 = 0.3 * 0.15 = 0.045``;
* the affine utility ``1000 * excess - 30`` is 15 for ``act`` and 0 for ``hold``;
* a larger second study (0.07, variance 0.0002) makes the pool ``600 / 7500 = 0.08``, the excess
  risk 0.024 and ``act`` worth -6.

Every stage is point-only: each request for an interval, a posterior decision or a ranking
guarantee is refused with ``cell_not_licensed``. Nothing here uses an internal constructor or a
test-only hook. The fresh-process consumer is ``consumer.py`` from the fixture directory; the
oracle in it is a closed form written there.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from antecedent import composition_verification as cv
from antecedent import decision
from antecedent import evidence_ledger as el
from antecedent import interface_contract as ic
from antecedent import joint_law_propagation as jl
from antecedent import measurement_models as mm
from antecedent.errors import StructuredRefusal
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success

import _release_boundary as rb
from _refusal import assert_registered_refusal
from _release_boundary import DECISION, EXACT, FIXTURE, SCENARIO, World

VALUES = FIXTURE["values"]
REVISED = FIXTURE["revised_values"]
MUTATIONS = FIXTURE["mutations"]
CONVERSIONS = FIXTURE["conversions"]
VARIANTS = FIXTURE["variants"]


def assert_refusal(error: StructuredRefusal, entry: dict[str, Any]) -> None:
    assert_registered_refusal(error)
    assert error.reason_code == entry["code"], (error.reason_code, str(error))
    if "detail" in entry:
        assert error.detail == entry["detail"], (error.detail, str(error))


def refused(call: Any, kind: type[StructuredRefusal] = StructuredRefusal) -> StructuredRefusal:
    with pytest.raises(kind) as caught:
        require_success(call())
    return caught.value


# --------------------------------------------------------------------------- 1 and 2


def test_a_measured_exposure_enters_the_causal_model_through_the_declared_confusion_model():
    world = World.build()
    m = SCENARIO["measurement"]
    # Closed form: p* = Se pi + (1 - Sp)(1 - pi), J = Se + Sp - 1 = 0.7.
    j = m["sensitivity"] + m["specificity"] - 1.0
    p_star = m["events"] / m["total"]
    pi = (p_star + m["specificity"] - 1.0) / j
    assert world.measured.point == pytest.approx(pi, abs=EXACT)
    assert world.measured.point == pytest.approx(VALUES["prevalence"], abs=EXACT)
    assert world.measured.naive == pytest.approx(VALUES["naive_prevalence"], abs=1e-15)
    variance = (
        p_star * (1.0 - p_star) / m["total"]
        + (1.0 - pi) ** 2 * m["specificity_variance"]
        + pi**2 * m["sensitivity_variance"]
    ) / j**2
    assert world.measured.standard_error**2 == pytest.approx(variance, abs=EXACT)

    # The pooled study evidence: inverse variance over disjoint rows.
    pooled = world.pooled
    assert pooled.estimate == pytest.approx(VALUES["pooled_effect"], abs=EXACT)
    assert pooled.variance == pytest.approx(VALUES["pooled_variance"], abs=EXACT)
    assert pooled.declared_slots == VALUES["declared_slots"]
    assert pooled.distinct_observations == VALUES["distinct_observations"]

    # The composed finite model reads the measured prevalence and the pooled effect.
    c = world.composition
    assert rb.outcome_risk(c, {"X": "1"}) == pytest.approx(
        VALUES["outcome_risk_exposed"], abs=EXACT
    )
    assert rb.outcome_risk(c, {"X": "0"}) == pytest.approx(
        VALUES["outcome_risk_unexposed"], abs=EXACT
    )
    assert rb.outcome_risk(c) == pytest.approx(VALUES["outcome_risk"], abs=EXACT)
    assert rb.excess_risk(c) == pytest.approx(VALUES["excess_risk"], abs=EXACT)
    assert rb.excess_risk(c) == pytest.approx(pi * pooled.estimate, abs=EXACT)


# --------------------------------------------------------------------------- 3


def _artifact(snapshot: str, alignment: str, x, y) -> JointDistributionArtifact:
    quantities = tuple(
        ScientificQuantity.outcome(name, units="risk", population="pop", regime="observational")
        for name in ("prevalence", "effect")
    )
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=quantities,
        alignment=alignment,  # type: ignore[arg-type]
        source_id=f"study-{snapshot}",
        provider_id="release_boundary",
        rng_id=f"enumerated-{snapshot}",
        snapshot_id=snapshot,
        causal_contract_id="checked-contract",
    )
    return JointDistributionArtifact(identity, np.column_stack([x, y]).astype(np.float64))


def test_shared_evidence_is_counted_once_and_dependence_is_declared_or_refused():
    studies = SCENARIO["studies"]
    overlapping = SCENARIO["overlapping_study"]
    ledger = el.EvidenceLedger()
    a = ledger.add_source(rb.study("study_a", "cohort", studies[0]["rows"]))
    c = ledger.add_source(rb.study("study_c", "cohort", overlapping["rows"]))
    vendor = ledger.add_source(el.SourceDeclaration("vendor", "external_claim"))
    est_a, var_a = studies[0]["estimate"], studies[0]["variance"]
    est_c, var_c = overlapping["estimate"], overlapping["variance"]
    cov = overlapping["covariance"]

    # The ledger reads the overlap from the rows: 200 observations are shared.
    account = ledger.account_observations([a, c])
    assert account.pairs[0].overlap == "shared_observations"
    assert account.pairs[0].shared == 200
    assert account.mass.declared_slots == 1000
    assert account.mass.distinct_observations == 800
    assert account.mass.repeated_slots == 200

    # Declared independent, the shared rows would count twice: refused before any number.
    head = ledger.head
    error = refused(lambda: rb.pool(ledger, "independent", [(a, est_a, var_a), (c, est_c, var_c)]))
    assert_refusal(error, MUTATIONS["evidence_overlap"])
    error = refused(
        lambda: rb.pool(
            ledger, "diagonal", [(a, est_a, var_a), (c, est_c, var_c)],
            covariance=[var_a, 0.0, 0.0, var_c],
        )
    )  # fmt: skip
    assert (error.reason_code, error.detail) == (
        "evidence_double_counted",
        "evidence_ledger.diagonal_covariance_shared_rows",
    )
    assert ledger.head == head

    # Undeclared observations are an unknown overlap, never independent.
    error = refused(lambda: rb.pool(ledger, "vendor", [(a, est_a, var_a), (vendor, est_c, var_c)]))
    assert_refusal(error, VARIANTS["dependence_gap_ledger"])

    # A declared covariance with cross terms licenses the shared pair: generalized least
    # squares of y = (0.10, 0.20), S = [[0.0004, 0.0002], [0.0002, 0.0004]]: weights (1/2, 1/2),
    # estimate 0.15, variance (0.0004 + 0.0002) / 2 = 0.0003.
    pooled = rb.pool(
        ledger,
        "shared",
        [(a, est_a, var_a), (c, est_c, var_c)],
        covariance=[var_a, cov, cov, var_c],
    )
    assert pooled.estimate == pytest.approx(VALUES["overlap_pooled_estimate"], abs=EXACT)
    assert pooled.variance == pytest.approx(VALUES["overlap_pooled_variance"], abs=EXACT)
    assert pooled.declared_slots == VALUES["overlap_declared_slots"]
    assert pooled.distinct_observations == VALUES["overlap_distinct_observations"]

    # The joint-law rule reads the same overlap: the shared rows carry one evidence identity.
    shared = [jl.MomentInput("study_a", a, est_a), jl.MomentInput("study_c", a, est_c)]
    error = refused(
        lambda: require_success(jl.linear_moments(
            "weighted_sum", shared, [0.5, 0.5], [[var_a, 0.0], [0.0, var_c]],
            dependence="declared_independent",
        )),
        jl.JointLawPropagationRefusal,
    )  # fmt: skip
    assert_refusal(error, VARIANTS["shared_evidence_declared_independent"])
    claim = require_success(jl.linear_moments(
        "weighted_sum", shared, [0.5, 0.5], [[var_a, cov], [cov, var_c]],
        dependence="covariance", covariance_id="shared-rows-300-500",
    ))  # fmt: skip
    assert claim.value("mean") == pytest.approx(0.15, abs=EXACT)
    # The rule's variance is the ledger's generalized-least-squares variance.
    assert claim.value("variance") == pytest.approx(pooled.variance, abs=EXACT)
    assert claim.label == "point_only"
    error = refused(claim.interval, jl.JointLawPropagationRefusal)
    assert error.reason_code == VARIANTS["interval"]["code"]

    # Unknown dependence is a gap, not an independence.
    distinct = [
        jl.MomentInput("study_a", "evidence-a", est_a),
        jl.MomentInput("study_b", "evidence-b", 0.20),
    ]
    error = refused(
        lambda: require_success(
            jl.linear_moments(
                "weighted_sum",
                distinct,
                [0.5, 0.5],
                [[var_a, 0.0], [0.0, var_a]],
                dependence="unknown",
            )
        ),
        jl.JointLawPropagationRefusal,
    )
    assert_refusal(error, VARIANTS["dependence_gap_rule"])

    # The claim graph reports the same overlap from the ledger's rows.
    graph = cv.CompositionGraph(
        [
            cv.Node("study_a", "cohort_source", identity=cv.digest("study_a"), evidence=(a,)),
            cv.Node("study_c", "cohort_source", identity=cv.digest("study_c"), evidence=(c,)),
            cv.Node("overlap_check", "claim", identity=cv.digest("overlap_check")),
        ],
        [
            cv.Edge("overlap_check", "study_a", "evidence"),
            cv.Edge("overlap_check", "study_c", "evidence"),
        ],
    )
    relations = cv.shared_data_from_ledger(graph, ledger)
    assert [(r.first, r.second, r.relation, r.shared_observations) for r in relations] == [
        ("study_a", "study_c", "shared_data", 200)
    ]


def test_unpaired_marginals_are_never_paired_by_index_and_an_enumerated_joint_law_is():
    prevalence = SCENARIO["marginal_draws"]["prevalence"]
    effect = SCENARIO["marginal_draws"]["effect"]
    # Marginals with distinct evidence and no joint law: index pairing is refused.
    x = np.tile(prevalence, 2)
    y = np.tile(effect, 2)
    marginals = _artifact("marginals", "independent_marginals", x, y)
    mx = jl.PropagationDraws.from_artifact_column("prevalence", marginals, 0)
    my = jl.PropagationDraws.from_artifact_column("effect", marginals, 1)
    error = refused(
        lambda: require_success(jl.aligned_functional("nonlinear_functional", mx, my, "minimum")),
        jl.JointLawPropagationRefusal,
    )
    assert_refusal(error, VARIANTS["unpaired_marginals"])

    # The declared independent coupling, enumerated as the full product lattice, is one
    # aligned joint law: E[pi d] = E[pi] E[d] = 0.3 * 0.15 = 0.045.
    lattice = [(p, d) for p in prevalence for d in effect]
    joint = _artifact("lattice", "joint", [p for p, _ in lattice], [d for _, d in lattice])
    jx = jl.PropagationDraws.from_artifact_column("prevalence", joint, 0)
    jy = jl.PropagationDraws.from_artifact_column("effect", joint, 1)
    claim = require_success(jl.aligned_functional("product", jx, jy, "product"))
    assert claim.value("mean") == pytest.approx(VALUES["aligned_product_mean"], abs=EXACT)
    assert claim.route == "aligned_joint_law"
    assert claim.label == "point_only"
    # The aligned moment is the point claim of the composed model.
    assert claim.value("mean") == pytest.approx(World.build().excess_risk, abs=DECISION)
    assert (
        refused(claim.posterior, jl.JointLawPropagationRefusal).reason_code == "cell_not_licensed"
    )


# --------------------------------------------------------------------------- 4


def test_the_decision_reads_the_licensed_mean_and_matches_the_hand_computed_utilities():
    world = World.build()
    result = require_success(world.result)
    # 1000 * 0.045 - 30 = 15 for act; the hold action is worth 0.
    assert rb.value_of(result, "act") == pytest.approx(VALUES["act"], abs=DECISION)
    assert rb.value_of(result, "hold") == pytest.approx(VALUES["hold"], abs=DECISION)
    assert result.verdict == decision.Verdict("uniquely_optimal", (VALUES["selected"],))
    # A mean source carries no sampling error and no regret or EVPI.
    assert result.evpi is None
    assert all(a.standard_error is None and a.expected_regret is None for a in result.actions)
    assert result.n_draws == 0

    # The decision follows the declaration: a programme costing 50 is worth 1000 * 0.045 - 50 = -5.
    dearer = require_success(
        rb.decide(
            rb.declaration(cost=50.0),
            world.excess_risk,
            world.ledger.state_digest,
            world.plan_identity,
        )
    )
    assert rb.value_of(dearer, "act") == pytest.approx(-5.0, abs=DECISION)
    assert dearer.selected == ("hold",)


# --------------------------------------------------------------------------- 5


def test_new_observations_revalidate_the_claim_graph_and_the_decision_is_reevaluated(tmp_path):
    world = World.build()
    before = world.pooled.result

    # Nothing has changed: every claim is still valid and the decision is usable.
    steady = rb.verify(world.graph, world.graph, world.ledger, before)
    assert steady.keys_of("recompute") == ()
    assert world.gate().classify("decision").klass == "still_valid"

    # Study B grows: the ledger classifies the pooled result for recomputation and the claim
    # graph carries that to everything that rests on it.
    step = world.revise()
    assert world.ledger.standing(before) == "recompute"
    error = refused(world.gate, cv.CompositionVerificationRefusal)
    assert_refusal(error, VARIANTS["revalidation_pending"])
    stale = rb.verify(world.graph, world.graph, world.ledger, before)
    assert stale.keys_of("recompute") == ("claim", "decision", "mechanisms", "pooled_effect")
    assert stale.keys_of("still_valid") == ("measurement", "study_a", "study_b", "survey")
    assert stale.keys_of("withdrawn") == ()
    # The stale pooled numbers are not served.
    error = refused(lambda: world.ledger.synthesis(before), el.EvidenceLedgerRefusal)
    assert (error.reason_code, error.detail) == (
        "evidence_source_withdrawn",
        "evidence_ledger.result_not_valid",
    )

    # Recompute on the grown study, naming the result replaced:
    # (0.10 / 0.0004 + 0.07 / 0.0002) / (1 / 0.0004 + 1 / 0.0002) = 600 / 7500 = 0.08.
    first = SCENARIO["studies"][0]
    pooled = rb.pool(
        world.ledger,
        "pooled_effect_recomputed",
        [(world.study_a, first["estimate"], first["variance"]), (step.grown, *step.summary)],
        replaces=before,
    )
    assert pooled.estimate == pytest.approx(REVISED["pooled_effect"], abs=EXACT)
    assert pooled.variance == pytest.approx(REVISED["pooled_variance"], abs=EXACT)
    assert world.ledger.standing(before) == "withdrawn"
    assert world.ledger.standing(pooled.result) == "still_valid"

    # Everything downstream of the pool is recomputed from the new evidence.
    revised = world.downstream_of(pooled, step.grown)
    assert revised.excess_risk == pytest.approx(REVISED["excess_risk"], abs=EXACT)
    assert rb.outcome_risk(revised.composition, {"X": "1"}) == pytest.approx(
        REVISED["outcome_risk_exposed"], abs=EXACT
    )
    assert rb.outcome_risk(revised.composition) == pytest.approx(REVISED["outcome_risk"], abs=EXACT)
    assert rb.value_of(revised.result, "act") == pytest.approx(REVISED["act"], abs=DECISION)
    assert rb.value_of(revised.result, "hold") == pytest.approx(REVISED["hold"], abs=DECISION)
    # The decision flips: the programme no longer pays for itself.
    assert require_success(revised.result).selected == (REVISED["selected"],)
    assert revised.plan_identity != world.plan_identity

    # The classification against the earlier graph: what rested on study B is recomputed;
    # what did not is still valid and keeps its numbers.
    moved = rb.verify(world.graph, revised.graph, revised.ledger, pooled.result)
    assert moved.keys_of("recompute") == (
        "claim", "decision", "mechanisms", "pooled_effect", "study_b",
    )  # fmt: skip
    assert moved.keys_of("still_valid") == ("measurement", "study_a", "survey")
    # And the recomputed decision is usable against its own graph.
    assert revised.gate().classify("decision").klass == "still_valid"

    # A fresh process consumes the revised evidence state and reaches the same decision.
    bundle = tmp_path / "revised"
    revised.write_bundle(bundle, studies=rb.revised_studies())
    outcome = _consume(bundle)
    assert outcome["ok"], outcome
    report = outcome["report"]
    assert report["pooled_effect"] == pytest.approx(REVISED["pooled_effect"], abs=EXACT)
    assert report["act"] == pytest.approx(REVISED["act"], abs=DECISION)
    assert report["selected"] == REVISED["selected"]
    assert report["snapshot_bound"] and report["claim_graph_valid"]


# --------------------------------------------------------------------------- 6


def test_duplicated_evidence_cannot_be_counted_twice(tmp_path):
    first = SCENARIO["studies"][0]
    est, var = first["estimate"], first["variance"]
    ledger = el.EvidenceLedger()
    a = ledger.add_source(rb.study("study_a", "cohort", first["rows"]))
    # The same study re-entering through its lineage.
    copy = ledger.add_source(
        el.SourceDeclaration("study_a_again", "observation_set", lineage=el.Lineage.duplicate(a))
    )
    head = ledger.head
    error = refused(lambda: rb.pool(ledger, "twice", [(a, est, var), (copy, est, var)]))
    assert (error.reason_code, error.detail) == (
        "evidence_double_counted",
        "evidence_ledger.reentry_duplicate",
    )
    # The same source listed twice.
    error = refused(lambda: rb.pool(ledger, "listed_twice", [(a, est, var), (a, est, var)]))
    assert (error.reason_code, error.detail) == (
        "evidence_double_counted",
        "evidence_ledger.repeated_source",
    )
    assert ledger.head == head
    assert ledger.results == ()

    # A consumer re-deriving the pool from declarations in which the second study re-uses the
    # first study's rows is refused the same way.
    world = World.build()
    bundle = tmp_path / "bundle"
    studies = world.studies()
    studies[1]["rows"] = list(studies[0]["rows"])
    world.write_bundle(bundle, studies=studies)
    outcome = _consume(bundle)
    assert not outcome["ok"]
    assert_refusal_dict(outcome["refusal"], MUTATIONS["evidence_overlap"])


# --------------------------------------------------------------------------- 7


def test_unsupported_semantic_conversions_fail_closed():
    world = World.build()
    model = world.measured.identity

    # The measured column offered as the construct it proxies.
    error = refused(lambda: rb.measure(SCENARIO, rb.INDICATOR), mm.MeasurementModelRefusal)
    assert_refusal(error, CONVERSIONS["measured_as_construct"])

    # The value used without the model it was corrected under.
    components, connections = rb.interface_components(model, omit_measurement_model=True)
    error = refused(
        lambda: ic.check_composition(components, connections), ic.InterfaceContractRefusal
    )
    assert (error.reason_code, error.detail) == (
        CONVERSIONS["measurement_model_missing"]["code"],
        CONVERSIONS["measurement_model_missing"]["detail"],
    )

    # A mean offered where the consumer requires a law.
    components, connections = rb.interface_components(model, decision_requires_law=True)
    error = refused(
        lambda: ic.check_composition(components, connections), ic.InterfaceContractRefusal
    )
    assert_refusal(error, CONVERSIONS["mean_for_law"])

    # The honest declarations connect, producers first.
    components, connections = rb.interface_components(model)
    assert ic.check_composition(components, connections).order == (
        "measurement", "mechanisms", "decision",
    )  # fmt: skip

    # A mean source never answers a criterion that needs a law.
    error = refused(
        lambda: rb.decide(rb.declaration(needs_law=True), world.excess_risk, "snapshot", "plan"),
        decision.DecisionRefusal,
    )
    assert_refusal(error, MUTATIONS["law"])


# --------------------------------------------------------------------------- 8


def _consume(directory: Path) -> dict[str, Any]:
    done = subprocess.run(
        [sys.executable, str(rb.CONSUMER), str(directory)],
        capture_output=True,
        text=True,
        cwd=directory,
        check=True,
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


def assert_refusal_dict(refusal: dict[str, str], entry: dict[str, Any]) -> None:
    assert refusal["stage"] == entry.get("stage", refusal["stage"])
    assert refusal["code"] == entry["code"], refusal
    if "detail" in entry:
        assert refusal["detail"] == entry["detail"], refusal


def _mutated(base: Path, tmp_path: Path, name: str, edit) -> Path:
    target = tmp_path / name
    shutil.copytree(base, target)
    path = target / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    edit(target, manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return target


def test_a_fresh_process_consumer_recomputes_the_decision_and_every_mutation_is_refused(tmp_path):
    # The producer is gone before the consumer starts: only a bundle on disk survives.
    base = tmp_path / "base"
    other = tmp_path / "other"
    verified_here = {}
    world = World.build()
    world.write_bundle(base)
    verified_here["act"] = rb.value_of(world.result, "act")
    # Another history: the same evidence plus a late arrival.
    moved = World.build()
    moved.ledger.add_source(rb.study("late_arrival", "late", [0, 10]))
    moved.write_bundle(other)
    del world

    outcome = _consume(base)
    assert outcome["ok"], outcome
    report = outcome["report"]
    assert report["prevalence"] == pytest.approx(VALUES["prevalence"], abs=EXACT)
    assert report["pooled_effect"] == pytest.approx(VALUES["pooled_effect"], abs=EXACT)
    assert report["pooled_variance"] == pytest.approx(VALUES["pooled_variance"], abs=EXACT)
    assert report["exposed_risk"] == pytest.approx(VALUES["outcome_risk_exposed"], abs=EXACT)
    assert report["unexposed_risk"] == pytest.approx(VALUES["outcome_risk_unexposed"], abs=EXACT)
    assert report["excess_risk"] == pytest.approx(VALUES["excess_risk"], abs=DECISION)
    assert report["act"] == pytest.approx(VALUES["act"], abs=DECISION)
    assert report["hold"] == pytest.approx(VALUES["hold"], abs=DECISION)
    assert report["selected"] == VALUES["selected"]
    assert report["snapshot_bound"] and report["claim_graph_valid"]
    # The producer's report and the consumer's closed form agree.
    assert verified_here["act"] == pytest.approx(report["act"], abs=DECISION)

    def quantity(_dir, manifest):
        manifest["decision"]["mean_units"] = "percent"

    def law(_dir, manifest):
        manifest["decision"]["needs_law"] = True

    def support(_dir, manifest):
        manifest["interface"]["producer_support_real"] = True

    def snapshot(dir_, manifest):
        # The ledger bytes are another history's under the retained identity.
        shutil.copyfile(other / "ledger.bin", dir_ / "ledger.bin")

    def evidence_overlap(_dir, manifest):
        first = manifest["studies"][0]["rows"]
        manifest["studies"][1]["rows"] = [first[0], first[0] + 100]

    for name, edit in (
        ("quantity", quantity),
        ("law", law),
        ("support", support),
        ("snapshot", snapshot),
        ("evidence_overlap", evidence_overlap),
    ):
        outcome = _consume(_mutated(base, tmp_path, name, edit))
        assert not outcome["ok"], name
        assert_refusal_dict(outcome["refusal"], MUTATIONS[name])

    # The unmutated bundle is still consumed: no mutation reached it.
    assert _consume(base)["ok"]


def test_a_claim_graph_verified_at_another_ledger_state_is_refused_by_the_consumer():
    world = World.build()
    verification = rb.verify(world.graph, world.graph, world.ledger, world.pooled.result)
    exported = verification.export()
    consumed = cv.consume_composition_verification_artifact(
        exported.data, exported.expectation.identity
    )
    consumed.verify_ledger(world.graph, world.ledger)  # the same state: accepted
    world.ledger.add_source(rb.study("late_arrival", "late", [0, 10]))
    error = refused(
        lambda: consumed.verify_ledger(world.graph, world.ledger), cv.CompositionVerificationRefusal
    )
    assert (error.reason_code, error.detail) == (
        "invalid_argument",
        "composition_verification.ledger_mismatch",
    )


# --------------------------------------------------------------------------- 10


def test_an_upstream_withdrawal_makes_the_downstream_decision_refuse_end_to_end():
    world = World.build()
    world.gate()
    # Consent for study A is withdrawn.
    require_success(
        world.ledger.revise_source(
            require_success(el.SourceRevision.withdraw(world.study_a, "consent withdrawn"))
        )
    )
    pooled = world.pooled.result
    assert world.ledger.standing(pooled) == "withdrawn"

    # The claim graph withdraws everything resting on the pool, with no numerical reason;
    # the unrelated measurement stays valid.
    verification = rb.verify(world.graph, world.graph, world.ledger, pooled)
    assert verification.keys_of("withdrawn") == ("claim", "decision", "mechanisms", "pooled_effect")
    assert "measurement" in verification.keys_of("still_valid")
    assert all(not verification.classify(k).numerical for k in verification.keys_of("withdrawn"))
    error = refused(world.gate, cv.CompositionVerificationRefusal)
    assert_refusal(error, VARIANTS["withdrawal"])
    # No number is served from the withdrawn pool, and new work cannot rest on it.
    error = refused(lambda: world.ledger.synthesis(pooled), el.EvidenceLedgerRefusal)
    assert error.reason_code == "evidence_source_withdrawn"
    first, second = SCENARIO["studies"]
    error = refused(
        lambda: rb.pool(
            world.ledger,
            "again",
            [
                (world.study_a, first["estimate"], first["variance"]),
                (world.study_b, second["estimate"], second["variance"]),
            ],
        )
    )
    assert (error.reason_code, error.detail) == (
        "evidence_source_withdrawn",
        "evidence_ledger.source_withdrawn",
    )


def test_a_dependence_gap_makes_the_downstream_decision_refuse_end_to_end():
    first = SCENARIO["studies"][0]
    vendor_claim = SCENARIO["vendor_claim"]
    ledger = el.EvidenceLedger()
    a = ledger.add_source(rb.study("study_a", "cohort", first["rows"]))
    vendor = ledger.add_source(el.SourceDeclaration("vendor", "external_claim"))

    # The second contribution is a vendor claim whose observations are not declared: its
    # overlap with the first study is unknown. There is no pooled result, hence no
    # composition, claim or decision.
    error = refused(
        lambda: rb.pool(
            ledger,
            "pooled_effect",
            [
                (a, first["estimate"], first["variance"]),
                (vendor, vendor_claim["estimate"], vendor_claim["variance"]),
            ],
        )
    )
    assert_refusal(error, VARIANTS["dependence_gap_ledger"])
    assert ledger.results == ()

    # The joint-law rule refuses the same gap when asked for the combination.
    inputs = [jl.MomentInput("study_a", a, 0.10), jl.MomentInput("vendor", vendor, 0.20)]
    error = refused(
        lambda: require_success(
            jl.linear_moments(
                "weighted_sum",
                inputs,
                [0.5, 0.5],
                [[0.0004, 0.0], [0.0, 0.0004]],
                dependence="unknown",
            )
        ),
        jl.JointLawPropagationRefusal,
    )
    assert error.reason_code == VARIANTS["dependence_gap_rule"]["code"]

    # Declaring the dependence is what licenses the combination: the vendor discloses its
    # rows, and a declared covariance pools it.
    covered = ledger.add_source(rb.study("vendor_rows", "cohort", [400, 900]))
    licensed = rb.pool(
        ledger,
        "pooled_with_declared_dependence",
        [
            (a, first["estimate"], first["variance"]),
            (covered, vendor_claim["estimate"], vendor_claim["variance"]),
        ],
        covariance=[0.0004, 0.0002, 0.0002, 0.0004],
    )
    assert licensed.variance == pytest.approx(0.0003, abs=EXACT)


# --------------------------------------------------------------------------- point-only


def test_every_stage_is_point_only_and_no_interval_or_posterior_decision_ships():
    world = World.build()
    # The ledger: a pooled interval is never licensed.
    error = refused(
        lambda: world.ledger.pooled_interval(world.pooled.result), el.EvidenceLedgerRefusal
    )
    assert error.reason_code == VARIANTS["interval"]["code"]
    # The measurement: an interval is never licensed.
    prepared = mm.prepare_measurement_model(
        rb.measurement_model(),
        mm.AnalysisBinding(
            construct_id=rb.CONSTRUCT,
            population_id="pop",
            regime_id="observational",
            representation=rb.SCALAR,
        ),
        mm.BinaryPrevalenceData(
            SCENARIO["measurement"]["events"], SCENARIO["measurement"]["total"]
        ),
    )
    error = refused(require_success(prepared.estimate()).interval, mm.MeasurementModelRefusal)
    assert error.reason_code == VARIANTS["interval"]["code"]
    # The composition: the variance carries no measured coverage.
    error = refused(world.composition.interval)
    assert error.reason_code == VARIANTS["interval"]["code"]
    # The claim graph holds no calibration record: the decision is a point value.
    assert world.gate().classify("decision").calibration.state == "not_applicable"
    # A criterion beyond a mean is refused, not approximated.
    refusal = refused(
        lambda: rb.decide(rb.declaration(needs_law=True), world.excess_risk, "s", "p"),
        decision.DecisionRefusal,
    )
    assert refusal.reason_code == "decision_contract_unsatisfied"
    assert FIXTURE["calibration"] == "unmeasured"
    assert FIXTURE["label"] == "point_only"
    assert require_success(world.result).evpi is None
