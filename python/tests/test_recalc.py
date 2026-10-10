"""2.3 C2: selective recalculation with a visible receipt, through the Python facade.

The route is the cross-fitted AIPW average effect through the prepared-study architecture.
Each mutation asserts the whole per-stage status table, the counts of work that actually ran
(nuisance fold fits read from the estimator's own per-call fit cache, identifications, score
builds, reweights, decisions) and the value against an independent rerun in a fresh session
plus a plain-sum oracle over the frozen score table. The first run performs 5 folds x 2
nuisance sets (propensity, outcome) = 10 fold fits.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from antecedent import recalc
from antecedent.errors import CausalSerializationError, CausalValueError
from antecedent.recalc import (
    Capabilities,
    RecalcNoLiveState,
    RecalcReceipt,
    RecalcReceiptRefusal,
    RecalcRefusal,
    RecalcRequest,
    RecalcResult,
    RecalcSession,
    RecalcUnavailable,
    ResumeContext,
    Stage,
    TargetWeights,
    Utility,
)

from _refusal import assert_registered_refusal

N = 600
SEED = 61
FIRST_RUN_FOLD_FITS = 10

STAGES = (
    Stage.GRAPH,
    Stage.QUERY,
    Stage.REGIME,
    Stage.EVIDENCE,
    Stage.SOURCE_POPULATION,
    Stage.TARGET_POPULATION,
    Stage.DATA_SNAPSHOT,
    Stage.ROW_DESIGN,
    Stage.TREATMENT_GRID,
    Stage.LEARNER_FOLDS_RNG,
    Stage.UTILITY,
    Stage.IDENTIFICATION,
    Stage.SCORE_ARTIFACT,
    Stage.LAW,
    Stage.DECISION,
)
# The primary (first declared) dependency of each derived stage; an input is its own.
PRIMARY = {
    Stage.IDENTIFICATION: Stage.GRAPH,
    Stage.SCORE_ARTIFACT: Stage.IDENTIFICATION,
    Stage.LAW: Stage.SCORE_ARTIFACT,
    Stage.DECISION: Stage.LAW,
}


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


BASE_EDGES = (("z", "t"), ("z", "y"), ("t", "y"), ("z", "y2"), ("t", "y2"))


def request(effect: float = 2.0) -> RecalcRequest:
    return RecalcRequest(
        data=columns(effect),
        edges=BASE_EDGES,
        treatment="t",
        outcome="y",
        utility=Utility(2.0, 0.5),
    )


def z_weights(req: RecalcRequest) -> TargetWeights:
    z = np.asarray(req.data["z"])
    return TargetWeights(np.exp(0.4 * z), ("z",))


def reused_text(stage: Stage) -> str:
    return f"reused({PRIMARY.get(stage, stage).value})"


def expected(overrides: dict[Stage, str]) -> tuple[tuple[str, str], ...]:
    return tuple((s.value, overrides.get(s, reused_text(s))) for s in STAGES)


def rerun(req: RecalcRequest, seed: int = SEED) -> RecalcResult:
    """An independent rerun: a brand-new session, so no frozen score or fit is shared."""
    out = RecalcSession().execute(req, seed=seed)
    assert out.receipt.totals.fold_fits == FIRST_RUN_FOLD_FITS, "a rerun fits afresh"
    return out


def work(out: RecalcResult) -> tuple[int, int, int, int, int]:
    return out.receipt.totals.as_tuple()


def weighted_mean(session: RecalcSession, weights: np.ndarray) -> float:
    contrast = session.score_contrast()
    assert contrast is not None
    return float(np.sum(weights * contrast) / np.sum(weights))


def assert_reused_identities_equal(a: RecalcResult, b: RecalcResult) -> None:
    for stage in b.plan.reused:
        assert a.plan.identity_of(stage) == b.plan.identity_of(stage), stage
        assert a.receipt.entry(stage).identity == b.receipt.entry(stage).identity


def test_c2_first_run_fits_and_a_rerun_in_a_fresh_session_fits_again() -> None:
    req = request()
    session = RecalcSession()
    first = session.execute(req, seed=SEED)
    # Nothing is previously known: every stage is recomputed and none reused.
    assert first.plan.reused == ()
    assert len(first.plan.recomputed) == len(STAGES)
    assert work(first) == (1, FIRST_RUN_FOLD_FITS, 1, 1, 1)
    # The per-call batch cache is not a persistent fit cache: a second session fits again.
    again = rerun(req)
    assert again.law.ate == first.law.ate
    assert again.receipt.identity == first.receipt.identity
    # The law is the plain-sum oracle of the frozen scores, the decision the declared rule.
    assert first.law.ate == pytest.approx(weighted_mean(session, np.ones(N)), abs=1e-12)
    assert first.decision.net_benefit == pytest.approx(2.0 * first.law.ate - 0.5, abs=1e-12)
    assert first.decision.treat == (first.decision.net_benefit > 0.0)
    assert first.receipt.loaded is False and session.is_live


def test_c2_utility_only_change_recomputes_the_decision_with_zero_fit_work() -> None:
    base = request()
    session = RecalcSession()
    first = session.execute(base, seed=SEED)

    changed = replace(base, utility=Utility(3.0, 0.1))
    second = session.execute(changed, seed=SEED)

    assert second.plan.table == expected(
        {
            Stage.UTILITY: "recomputed(own:utility:modified)",
            Stage.DECISION: "recomputed(upstream:utility<-utility:modified)",
        }
    )
    assert second.receipt.status_table == second.plan.table
    assert second.plan.recomputed_computations == (Stage.DECISION,)
    assert work(second) == (0, 0, 0, 0, 1)
    assert_reused_identities_equal(first, second)
    assert second.law == first.law, "the law was not touched"

    independent = rerun(changed)
    assert second.decision.net_benefit == pytest.approx(independent.decision.net_benefit, abs=1e-12)
    assert second.decision.net_benefit == pytest.approx(3.0 * independent.law.ate - 0.1, abs=1e-12)
    assert second.decision.treat == independent.decision.treat


def test_c2_compatible_target_weight_change_reuses_frozen_scores_with_equal_value() -> None:
    base = request()
    session = RecalcSession()
    first = session.execute(base, seed=SEED)

    changed = replace(base, target=z_weights(base))
    second = session.execute(changed, seed=SEED)

    assert second.plan.table == expected(
        {
            Stage.TARGET_POPULATION: "recomputed(own:target_population:modified)",
            Stage.LAW: "recomputed(upstream:target_population<-target_population:modified)",
            Stage.DECISION: "recomputed(upstream:law<-target_population:modified)",
        }
    )
    # No identification, no fit, no score build: one reweight and one decision.
    assert work(second) == (0, 0, 0, 1, 1)
    assert_reused_identities_equal(first, second)
    assert abs(second.law.ate - first.law.ate) > 0.05, "the target must change the law"

    weights = np.asarray(changed.target.weights)  # type: ignore[union-attr]
    assert second.law.ate == pytest.approx(weighted_mean(session, weights), abs=1e-12)
    independent = rerun(changed)
    assert second.law.ate == pytest.approx(independent.law.ate, abs=1e-12)
    assert second.law.std_error == pytest.approx(independent.law.std_error, abs=1e-12)
    assert second.decision.net_benefit == pytest.approx(independent.decision.net_benefit, abs=1e-12)


def test_c2_changed_folds_and_new_outcomes_refit_without_reidentifying() -> None:
    base = request()
    session = RecalcSession()
    first = session.execute(base, seed=SEED)

    # Changed folds/RNG: the same data under another master seed.
    seed = SEED + 1
    folds = session.execute(base, seed=seed)
    assert folds.plan.table == expected(
        {
            Stage.LEARNER_FOLDS_RNG: "recomputed(own:learner_folds_rng:modified)",
            Stage.SCORE_ARTIFACT: (
                "recomputed(upstream:learner_folds_rng<-learner_folds_rng:modified)"
            ),
            Stage.LAW: "recomputed(upstream:score_artifact<-learner_folds_rng:modified)",
            Stage.DECISION: "recomputed(upstream:law<-learner_folds_rng:modified)",
        }
    )
    c = folds.receipt.totals
    assert (c.identifications, c.score_computations, c.reweights) == (0, 1, 1)
    assert c.fold_fits >= FIRST_RUN_FOLD_FITS and c.fold_fits % 5 == 0
    assert folds.law.ate != first.law.ate, "other folds, other scores"
    assert folds.law.ate == pytest.approx(rerun(base, seed).law.ate, abs=1e-12)

    # New outcomes: same seed, different outcome column.
    fresh = request(1.5)
    outcomes = session.execute(fresh, seed=seed)
    assert outcomes.plan.table == expected(
        {
            Stage.DATA_SNAPSHOT: "recomputed(own:data_snapshot:modified)",
            Stage.SCORE_ARTIFACT: "recomputed(upstream:data_snapshot<-data_snapshot:modified)",
            Stage.LAW: "recomputed(upstream:score_artifact<-data_snapshot:modified)",
            Stage.DECISION: "recomputed(upstream:law<-data_snapshot:modified)",
        }
    )
    o = outcomes.receipt.totals
    assert (o.identifications, o.score_computations, o.reweights) == (0, 1, 1)
    assert o.fold_fits == c.fold_fits, "both refit the same cross-fitted design"
    assert abs(outcomes.law.ate - folds.law.ate) > 0.05
    assert outcomes.law.ate == pytest.approx(rerun(fresh, seed).law.ate, abs=1e-12)


def test_c2_graph_or_query_change_reidentifies() -> None:
    base = request()
    session = RecalcSession()
    first = session.execute(base, seed=SEED)

    # Graph change: an extra edge from the independent column.
    graph = replace(base, edges=(*BASE_EDGES, ("w", "y")))
    regraphed = session.execute(graph, seed=SEED)
    assert regraphed.plan.table == expected(
        {
            Stage.GRAPH: "recomputed(own:graph:modified)",
            Stage.IDENTIFICATION: "recomputed(upstream:graph<-graph:modified)",
            Stage.SCORE_ARTIFACT: "recomputed(upstream:identification<-graph:modified)",
            Stage.LAW: "recomputed(upstream:score_artifact<-graph:modified)",
            Stage.DECISION: "recomputed(upstream:law<-graph:modified)",
        }
    )
    c = regraphed.receipt.totals
    assert (c.identifications, c.score_computations, c.reweights) == (1, 1, 1)
    assert c.fold_fits >= FIRST_RUN_FOLD_FITS and c.fold_fits % 5 == 0
    assert regraphed.law.ate == pytest.approx(rerun(graph).law.ate, abs=1e-12)

    # Query change: the second outcome on the same graph and data.
    query = replace(graph, outcome="y2")
    requeried = session.execute(query, seed=SEED)
    assert requeried.plan.table == expected(
        {
            Stage.QUERY: "recomputed(own:query:modified)",
            Stage.IDENTIFICATION: "recomputed(upstream:query<-query:modified)",
            Stage.SCORE_ARTIFACT: "recomputed(upstream:identification<-query:modified)",
            Stage.LAW: "recomputed(upstream:score_artifact<-query:modified)",
            Stage.DECISION: "recomputed(upstream:law<-query:modified)",
        }
    )
    assert requeried.receipt.totals.identifications == 1
    assert requeried.law.ate == pytest.approx(rerun(query).law.ate, abs=1e-12)
    assert abs(requeried.law.ate - first.law.ate) > 0.5, "another outcome, another law"


def test_c2_an_undeclared_retarget_refits_instead_of_reusing_scores() -> None:
    base = request()
    session = RecalcSession(retarget="not_declared")
    session.execute(base, seed=SEED)

    changed = replace(base, target=z_weights(base))
    second = session.execute(changed, seed=SEED)
    assert second.plan.table == expected(
        {
            Stage.TARGET_POPULATION: "recomputed(own:target_population:modified)",
            Stage.SCORE_ARTIFACT: (
                "recomputed(upstream:target_population<-target_population:modified)"
            ),
            Stage.LAW: "recomputed(upstream:score_artifact<-target_population:modified)",
            Stage.DECISION: "recomputed(upstream:law<-target_population:modified)",
        }
    )
    c = second.receipt.totals
    assert (c.identifications, c.score_computations, c.reweights) == (0, 1, 1)
    assert c.fold_fits >= FIRST_RUN_FOLD_FITS, "the refit really ran"
    assert second.law.ate == pytest.approx(rerun(changed).law.ate, abs=1e-12)


def test_c2_external_study_change_leaves_unrelated_branches_reused() -> None:
    """Planning over declared external studies: only the changed branch's chain recomputes."""

    def workflow(study_one: str) -> dict[Stage, str]:
        declared = {s: recalc.stage_identity(s, "v1") for s in STAGES}
        for branch in (0, 1):
            declared[Stage.provider_request(branch)] = recalc.stage_identity(
                Stage.provider_request(branch), "v1"
            )
            declared[Stage.prior(branch)] = recalc.stage_identity(Stage.prior(branch), "v1")
        declared[Stage.external_study(0)] = recalc.stage_identity(Stage.external_study(0), "s0")
        declared[Stage.external_study(1)] = recalc.stage_identity(
            Stage.external_study(1), study_one
        )
        return declared

    previous, requested = workflow("s1"), workflow("s1-revised")
    unchanged = recalc.plan_recalculation(previous, previous)
    assert unchanged.recomputed == ()
    plan = recalc.plan_recalculation(previous, requested)
    table = dict(plan.table)
    assert table["external_study.1"] == "recomputed(own:external_study.1:modified)"
    assert table["provider_request.1"] == (
        "recomputed(upstream:external_study.1<-external_study.1:modified)"
    )
    assert table["prior.1"] == "recomputed(upstream:external_study.1<-external_study.1:modified)"
    assert table["decision"] == "recomputed(upstream:prior.1<-external_study.1:modified)"
    # The unrelated branch and the whole native chain stay valid, with unchanged identities.
    for stage in (
        Stage.external_study(0),
        Stage.provider_request(0),
        Stage.prior(0),
        Stage.IDENTIFICATION,
        Stage.SCORE_ARTIFACT,
        Stage.LAW,
    ):
        assert plan.status(stage).reused, stage
        assert plan.identity_of(stage) == unchanged.identity_of(stage), stage
    assert set(plan.recomputed) == {
        Stage.external_study(1),
        Stage.provider_request(1),
        Stage.prior(1),
        Stage.DECISION,
    }
    assert plan.recomputed_computations == (
        Stage.provider_request(1),
        Stage.prior(1),
        Stage.DECISION,
    )
    text = plan.explain()
    assert "recomputed: its own input `external_study.1` was modified" in text
    assert "reused: its dependency `external_study.0` is unchanged" in text

    # Declaration identities are length-prefixed and typed.
    assert len(recalc.stage_identity(Stage.GRAPH, "ab", "c")) == 64
    assert recalc.stage_identity(Stage.GRAPH, "ab", "c") != recalc.stage_identity(
        Stage.GRAPH, "a", "bc"
    )
    with pytest.raises(CausalValueError, match="unknown recalculation stage"):
        recalc.plan_recalculation({"no_such_stage": "0" * 64}, {})
    with pytest.raises(CausalValueError, match="external-study branch"):
        Stage.external_study(8)


def test_c2_refusals_run_no_work_and_leave_the_session_untouched() -> None:
    base = request()
    session = RecalcSession()

    # Off-grid on a cold session: nothing runs and nothing becomes live.
    session.set_request_support("off_grid", "transport.smoothed_dose")
    with pytest.raises(RecalcRefusal) as caught:
        session.execute(base, seed=SEED)
    refusal = caught.value
    assert_registered_refusal(refusal)
    assert (refusal.stage, refusal.detail) == ("treatment_grid", "recalc.off_grid_request")
    assert refusal.reason_code == "route_not_supported"
    assert refusal.remedy is not None and "transport.smoothed_dose" in refusal.remedy
    assert refusal.plan is not None
    assert refusal.plan.status(Stage.TREATMENT_GRID).text == (
        "refused(recalc.off_grid_request[transport.smoothed_dose])"
    )
    for stage in (Stage.SCORE_ARTIFACT, Stage.LAW, Stage.DECISION):
        status = refusal.plan.status(stage)
        assert status.refused and status.detail.startswith("recalc.blocked_by_refused_dependency")
    assert not session.is_live
    assert session.identities == {}

    # A good run, then an unsupported request, then the good request again.
    session.set_request_support("on_grid")
    first = session.execute(base, seed=SEED)
    known = session.identities
    session.set_request_support("unsupported")
    with pytest.raises(RecalcRefusal) as caught:
        session.execute(base, seed=SEED)
    assert caught.value.detail == "recalc.unsupported_request"
    assert_registered_refusal(caught.value)
    assert session.is_live and session.identities == known
    session.set_request_support("on_grid")
    again = session.execute(base, seed=SEED)
    assert again.plan.table == expected({})
    assert work(again) == (0, 0, 0, 0, 0)
    assert again.law == first.law

    # Weights that are not a licensed function of the adjustment set refuse at the law.
    session.set_retarget_support("incompatible")
    changed = replace(base, target=z_weights(base))
    with pytest.raises(RecalcRefusal) as caught:
        session.execute(changed, seed=SEED)
    assert caught.value.detail == "recalc.retarget_incompatible"
    assert caught.value.reason_code == "construction_not_licensed"
    assert caught.value.plan is not None
    assert caught.value.plan.status(Stage.LAW).text == "refused(recalc.retarget_incompatible)"
    assert session.identities == known, "a refusal changes nothing"

    # Malformed requests are value errors, not refusals.
    with pytest.raises(CausalValueError):
        session.plan(replace(base, treatment="no_such_column"))


def test_c2_a_loaded_result_cannot_claim_cache_reuse_in_a_fresh_process() -> None:
    base = request()
    original = RecalcSession()
    first = original.execute(base, seed=SEED)
    # What an ordinary loaded result keeps: the identities, with no prepared study.
    loaded = original.identities

    # Nothing supplied: the missing data snapshot is the specific unavailable result.
    bare = RecalcSession.resume(loaded)
    assert not bare.is_live
    assert bare.capabilities.boundary == "fresh_process"
    with pytest.raises(RecalcUnavailable) as caught:
        bare.execute(base, seed=SEED)
    assert_registered_refusal(caught.value)
    assert (caught.value.stage, caught.value.missing) == ("data_snapshot", "data")
    assert caught.value.reason_code == "score_table_unavailable"
    assert caught.value.plan is not None
    assert caught.value.plan.status(Stage.SCORE_ARTIFACT).text == "refused(recalc.unavailable_fit)"
    assert caught.value.plan.reused and not any(not s.is_input for s in caught.value.plan.reused), (
        "no derived stage is reused"
    )
    # Resuming from a loaded receipt behaves the same.
    with pytest.raises(RecalcUnavailable):
        RecalcSession.resume(first.receipt).execute(base, seed=SEED)

    # Portable scores are declared but nothing loads them: no reuse is claimed.
    claimed = ResumeContext(portable_scores=True, supplied_data=True)
    with pytest.raises(RecalcNoLiveState) as caught_live:
        RecalcSession.resume(loaded, claimed).execute(base, seed=SEED)
    assert caught_live.value.detail == "recalc.no_live_state"
    assert caught_live.value.stage == "score_artifact"
    assert_registered_refusal(caught_live.value)

    # Supplied data resumes by recomputing every derived stage, with the same fit work as the
    # first run, and never by reuse.
    resumed = RecalcSession.resume(loaded, ResumeContext(supplied_data=True))
    second = resumed.execute(base, seed=SEED)
    for entry in second.plan.entries:
        if not entry.stage.is_input:
            assert entry.status.text == "recomputed(fresh_process)", entry.stage
    assert work(second) == work(first)
    assert second.law.ate == first.law.ate
    assert second.receipt.capabilities.boundary == "fresh_process"
    assert not second.receipt.claims_derived_reuse
    # Once live, the resumed session reuses like any in-process one.
    third = resumed.execute(replace(base, utility=Utility(1.0, 0.0)), seed=SEED)
    assert work(third) == (0, 0, 0, 0, 1)
    assert third.receipt.capabilities.boundary == "in_process"

    # A missing provider callback is its own specific result, and a supplied one is not.
    declared = {s: recalc.stage_identity(s, "v1") for s in STAGES}
    declared[Stage.external_study(0)] = recalc.stage_identity(Stage.external_study(0), "s0")
    declared[Stage.provider_request(0)] = recalc.stage_identity(Stage.provider_request(0), "p0")
    without = recalc.plan_recalculation(
        declared, declared, Capabilities(resume=ResumeContext(supplied_data=True))
    )
    assert without.status(Stage.provider_request(0)).text == "refused(recalc.unavailable_provider)"
    assert not without.is_executable
    with_provider = recalc.plan_recalculation(
        declared,
        declared,
        Capabilities(resume=ResumeContext(supplied_data=True, supplied_provider=True)),
    )
    assert with_provider.is_executable
    assert with_provider.status(Stage.provider_request(0)).text == "recomputed(fresh_process)"


def test_c2_the_receipt_is_portable_canonical_and_refuses_edits() -> None:
    base = request()
    session = RecalcSession()
    first = session.execute(base, seed=SEED)
    second = session.execute(replace(base, utility=Utility(3.0, 0.1)), seed=SEED)

    blob = second.receipt.export()
    loaded = RecalcReceipt.consume(blob, expected_identity=second.receipt.identity)
    assert loaded.loaded and not second.receipt.loaded
    assert loaded.identity == second.receipt.identity
    assert loaded.plan_identity == second.plan.identity
    assert loaded.status_table == second.receipt.status_table
    assert loaded.totals == second.receipt.totals and loaded.totals.total == 1
    assert loaded.plan == second.plan
    assert loaded.capabilities == Capabilities()
    assert loaded.requested == session.identities
    assert loaded.export() == blob
    assert recalc.consume(blob).identity == loaded.identity
    assert "recomputed(own:utility:modified)" in loaded.explain()

    # A receipt of another run, however consistent, is refused against the retained identity.
    assert first.receipt.identity != second.receipt.identity
    with pytest.raises(RecalcReceiptRefusal) as caught:
        RecalcReceipt.consume(first.receipt.export(), expected_identity=second.receipt.identity)
    assert caught.value.detail == "recalc_receipt.identity_mismatch"
    assert_registered_refusal(caught.value)

    # Corruption is a serialization error, never a refusal or a receipt.
    damaged = bytearray(blob)
    damaged[len(damaged) // 2] ^= 0xFF
    with pytest.raises(CausalSerializationError):
        RecalcReceipt.consume(bytes(damaged))
    with pytest.raises(CausalSerializationError):
        RecalcReceipt.consume(blob[: len(blob) // 2])

    # A fresh-process receipt records the boundary and claims no derived reuse.
    resumed = RecalcSession.resume(session.identities, ResumeContext(supplied_data=True))
    out = resumed.execute(base, seed=SEED)
    fresh = RecalcReceipt.consume(out.receipt.export())
    assert fresh.capabilities.boundary == "fresh_process"
    assert fresh.capabilities.resume == ResumeContext(supplied_data=True)
    assert not fresh.claims_derived_reuse
    assert fresh.reused == tuple(s for s in fresh.reused if s.is_input)
    assert fresh.identity == out.receipt.identity


def test_c2_the_plan_explains_each_status_with_its_determining_dependency() -> None:
    base = request()
    session = RecalcSession()
    session.execute(base, seed=SEED)
    known = session.identities
    plan = session.prepare(replace(base, target=z_weights(base)), seed=SEED)
    assert plan == session.plan(replace(base, target=z_weights(base)), seed=SEED)
    assert plan.recomputed_computations == (Stage.LAW, Stage.DECISION)
    assert plan.status(Stage.SCORE_ARTIFACT).reused
    assert plan.is_executable and plan.first_refusal is None
    text = plan.explain()
    assert (
        "recomputed: `target_population` changed because `target_population` was modified" in text
    )
    assert "reused: its dependency `identification` is unchanged" in text
    # Planning ran nothing: the session still reuses against its last run.
    assert session.identities == known
    assert session.is_live
