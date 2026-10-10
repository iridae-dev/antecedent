"""2.3 C2 remainder: cell-AIPW score reuse and portable score resume, through the Python facade.

The route is the cell-saturated AIPW interaction contrast over two binary treatments. Each
mutation asserts the whole per-stage status table, the work that actually ran (cell-model fits
read from the estimator's own instrument, identifications, score builds, reweights, decisions)
and the value against an independent rerun in a fresh session plus a plain-sum oracle over the
frozen scores. The first run fits 5 folds x (1 multinomial propensity + 4 cell outcome
regressions) = 25 cell models; a first cross-fit run is 10 fold fits.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from antecedent import recalc_cell as rc
from antecedent.errors import CausalError, CausalSerializationError
from antecedent.recalc import (
    RecalcNoLiveState,
    RecalcReceipt,
    RecalcRefusal,
    RecalcRequest,
    RecalcResult,
    Stage,
    TargetWeights,
    Utility,
)
from antecedent.recalc_cell import (
    CellRequest,
    CellSession,
    CrossfitSession,
    ScoreResumeRefusal,
    ScoreResumeUnavailable,
)

from _refusal import assert_registered_refusal

N = 1200
# Seed 71 at this size draws a fold on which the multinomial propensity fit does not converge
# (an estimator-robustness edge, recorded in TODO.md); this seed converges on every fold.
SEED = 72
FIRST_RUN_FITS = 25
VARIABLES = ("a", "d", "y", "z", "w", "y2")
BASE_EDGES = (
    ("z", "a"),
    ("z", "d"),
    ("z", "y"),
    ("a", "y"),
    ("d", "y"),
    ("z", "y2"),
    ("a", "y2"),
    ("d", "y2"),
)
UTILITY = Utility(2.0, 0.5)

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
TARGET_ONLY = {
    Stage.TARGET_POPULATION: "recomputed(own:target_population:modified)",
    Stage.LAW: "recomputed(upstream:target_population<-target_population:modified)",
    Stage.DECISION: "recomputed(upstream:law<-target_population:modified)",
}


def columns(effect: float) -> dict[str, np.ndarray]:
    """Two confounded binary treatments with an interaction, noise ``w`` and a second outcome."""
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(N)
    w = rng.standard_normal(N)
    p = 1.0 / (1.0 + np.exp(-0.5 * z))
    a = (rng.random(N) < p).astype(np.float64)
    d = (rng.random(N) < 0.5).astype(np.float64)
    y = effect * a * d + 0.5 * a + 0.2 * z + 0.25 * rng.standard_normal(N)
    y2 = -a + 0.5 * d + 0.3 * z + 0.25 * rng.standard_normal(N)
    return {"a": a, "d": d, "y": y, "z": z, "w": w, "y2": y2}


def request(effect: float = 1.5) -> CellRequest:
    return CellRequest(
        data=columns(effect),
        edges=BASE_EDGES,
        treatments=("a", "d"),
        outcome="y",
        utility=UTILITY,
        adjustment=("z",),
    )


def z_weights(req: CellRequest) -> TargetWeights:
    return TargetWeights(np.exp(0.4 * np.asarray(req.data["z"])), ("z",))


def reused_text(stage: Stage) -> str:
    return f"reused({PRIMARY.get(stage, stage).value})"


def expected(overrides: dict[Stage, str]) -> tuple[tuple[str, str], ...]:
    return tuple((s.value, overrides.get(s, reused_text(s))) for s in STAGES)


def work(out: RecalcResult) -> tuple[int, int, int, int, int]:
    return out.receipt.totals.as_tuple()


def rerun(req: CellRequest, seed: int = SEED) -> RecalcResult:
    """An independent rerun: a brand-new session, so no frozen score or fit is shared."""
    out = CellSession().execute(req, seed=seed)
    assert out.receipt.totals.fold_fits == FIRST_RUN_FITS, "a rerun fits afresh"
    return out


def interaction_mean(scores: dict[int, np.ndarray], weights: np.ndarray | None) -> float:
    """Plain-sum oracle: the weighted mean of ``s0 - s1 - s2 + s3`` over the frozen scores."""
    contrast = scores[0] - scores[1] - scores[2] + scores[3]
    w = np.ones_like(contrast) if weights is None else weights
    return float(np.sum(w * contrast) / np.sum(w))


def live_scores(session: CellSession) -> dict[int, np.ndarray]:
    scores = session.score_columns()
    assert scores is not None
    return scores


def test_c2_cell_first_run_counts_the_cell_fits_and_matches_a_plain_sum_oracle() -> None:
    req = request()
    session = CellSession()
    first = session.execute(req, seed=SEED)

    assert first.plan.reused == ()
    assert len(first.plan.recomputed) == len(STAGES)
    assert work(first) == (1, FIRST_RUN_FITS, 1, 1, 1)
    assert first.receipt.loaded is False and session.is_live
    assert first.law.ate == pytest.approx(interaction_mean(live_scores(session), None), abs=1e-10)
    assert abs(first.law.ate - 1.5) < 0.5, first.law.ate
    assert first.decision.net_benefit == pytest.approx(2.0 * first.law.ate - 0.5, abs=1e-12)
    assert first.decision.treat == (first.decision.net_benefit > 0.0)

    # The receipt is a portable recalc_receipt_v1 artifact that recomputes.
    consumed = RecalcReceipt.consume(
        first.receipt.export(), expected_identity=first.receipt.identity
    )
    assert consumed.status_table == first.receipt.status_table
    assert consumed.totals == first.receipt.totals

    # A second session fits again: there is no persistent fit cache.
    again = rerun(req)
    assert again.law.ate == first.law.ate
    assert again.receipt.identity == first.receipt.identity


def test_c2_cell_utility_only_change_recomputes_the_decision_with_zero_fits() -> None:
    base = request()
    session = CellSession()
    first = session.execute(base, seed=SEED)

    changed = replace(base, utility=Utility(3.0, 0.1))
    second = session.execute(changed, seed=SEED)

    assert second.receipt.status_table == expected(
        {
            Stage.UTILITY: "recomputed(own:utility:modified)",
            Stage.DECISION: "recomputed(upstream:utility<-utility:modified)",
        }
    )
    assert second.plan.recomputed_computations == (Stage.DECISION,)
    assert work(second) == (0, 0, 0, 0, 1)
    assert second.receipt.totals.fold_fits == 0
    assert second.law == first.law
    assert second.decision.net_benefit == pytest.approx(
        rerun(changed).decision.net_benefit, abs=1e-12
    )
    assert second.decision.net_benefit == pytest.approx(3.0 * first.law.ate - 0.1, abs=1e-12)


def test_c2_cell_compatible_target_weight_change_reuses_the_frozen_cell_scores() -> None:
    base = request()
    session = CellSession()
    first = session.execute(base, seed=SEED)
    frozen = {arm: scores.copy() for arm, scores in live_scores(session).items()}

    changed = replace(base, target=z_weights(base))
    second = session.execute(changed, seed=SEED)

    assert second.receipt.status_table == expected(TARGET_ONLY)
    # Zero refits, no identification, no score build: one reweight and one decision.
    assert work(second) == (0, 0, 0, 1, 1)
    assert second.receipt.totals.fold_fits == 0
    for arm, scores in live_scores(session).items():
        assert np.array_equal(scores, frozen[arm]), "the scores were not touched"
    # The retarget really applied: the law moves by far more than numerical noise (1e-12).
    assert abs(second.law.ate - first.law.ate) > 1e-3, "the target must move the law"

    weights = np.asarray(changed.target.weights)  # type: ignore[union-attr]
    assert second.law.ate == pytest.approx(
        interaction_mean(live_scores(session), weights), abs=1e-10
    )
    fresh = rerun(changed)
    assert second.law.ate == pytest.approx(fresh.law.ate, abs=1e-12)
    assert second.law.std_error == pytest.approx(fresh.law.std_error, abs=1e-12)
    assert second.decision.net_benefit == pytest.approx(fresh.decision.net_benefit, abs=1e-12)


def test_c2_cell_changed_folds_and_new_outcomes_refit_without_reidentifying() -> None:
    base = request()
    session = CellSession()
    first = session.execute(base, seed=SEED)

    folds = session.execute(base, seed=SEED + 1)
    assert folds.receipt.status_table == expected(
        {
            Stage.LEARNER_FOLDS_RNG: "recomputed(own:learner_folds_rng:modified)",
            Stage.SCORE_ARTIFACT: (
                "recomputed(upstream:learner_folds_rng<-learner_folds_rng:modified)"
            ),
            Stage.LAW: "recomputed(upstream:score_artifact<-learner_folds_rng:modified)",
            Stage.DECISION: "recomputed(upstream:law<-learner_folds_rng:modified)",
        }
    )
    assert work(folds) == (0, FIRST_RUN_FITS, 1, 1, 1)
    assert folds.law.ate != first.law.ate, "other folds, other scores"
    assert folds.law.ate == pytest.approx(rerun(base, SEED + 1).law.ate, abs=1e-12)

    other = replace(base, data=columns(1.0))
    outcomes = session.execute(other, seed=SEED + 1)
    assert outcomes.receipt.status_table == expected(
        {
            Stage.DATA_SNAPSHOT: "recomputed(own:data_snapshot:modified)",
            Stage.SCORE_ARTIFACT: "recomputed(upstream:data_snapshot<-data_snapshot:modified)",
            Stage.LAW: "recomputed(upstream:score_artifact<-data_snapshot:modified)",
            Stage.DECISION: "recomputed(upstream:law<-data_snapshot:modified)",
        }
    )
    assert work(outcomes) == (0, FIRST_RUN_FITS, 1, 1, 1)
    assert abs(outcomes.law.ate - folds.law.ate) > 0.2
    assert outcomes.law.ate == pytest.approx(rerun(other, SEED + 1).law.ate, abs=1e-12)


def test_c2_cell_graph_or_query_change_reidentifies_and_refits() -> None:
    base = request()
    session = CellSession()
    first = session.execute(base, seed=SEED)

    graph = replace(base, edges=(*BASE_EDGES, ("w", "y")))
    regraphed = session.execute(graph, seed=SEED)
    assert regraphed.receipt.status_table == expected(
        {
            Stage.GRAPH: "recomputed(own:graph:modified)",
            Stage.IDENTIFICATION: "recomputed(upstream:graph<-graph:modified)",
            Stage.SCORE_ARTIFACT: "recomputed(upstream:identification<-graph:modified)",
            Stage.LAW: "recomputed(upstream:score_artifact<-graph:modified)",
            Stage.DECISION: "recomputed(upstream:law<-graph:modified)",
        }
    )
    assert work(regraphed) == (1, FIRST_RUN_FITS, 1, 1, 1)
    assert regraphed.law.ate == pytest.approx(first.law.ate, abs=1e-12), "same scores"

    query = replace(graph, outcome="y2")
    requeried = session.execute(query, seed=SEED)
    assert requeried.receipt.status_table == expected(
        {
            Stage.QUERY: "recomputed(own:query:modified)",
            Stage.IDENTIFICATION: "recomputed(upstream:query<-query:modified)",
            Stage.SCORE_ARTIFACT: "recomputed(upstream:identification<-query:modified)",
            Stage.LAW: "recomputed(upstream:score_artifact<-query:modified)",
            Stage.DECISION: "recomputed(upstream:law<-query:modified)",
        }
    )
    assert work(requeried) == (1, FIRST_RUN_FITS, 1, 1, 1)
    assert requeried.law.ate == pytest.approx(rerun(query).law.ate, abs=1e-12)
    assert abs(requeried.law.ate - first.law.ate) > 0.5, "another outcome, another law"


def test_c2_cell_an_undeclared_retarget_refits_instead_of_reusing_scores() -> None:
    base = request()
    session = CellSession(retarget="not_declared")
    session.execute(base, seed=SEED)

    changed = replace(base, target=z_weights(base))
    second = session.execute(changed, seed=SEED)
    assert second.receipt.status_table == expected(
        {
            Stage.TARGET_POPULATION: "recomputed(own:target_population:modified)",
            Stage.SCORE_ARTIFACT: (
                "recomputed(upstream:target_population<-target_population:modified)"
            ),
            Stage.LAW: "recomputed(upstream:score_artifact<-target_population:modified)",
            Stage.DECISION: "recomputed(upstream:law<-target_population:modified)",
        }
    )
    assert work(second) == (0, FIRST_RUN_FITS, 1, 1, 1)
    assert second.law.ate == pytest.approx(rerun(changed).law.ate, abs=1e-12)

    # An unlicensed route cannot be exported either.
    with pytest.raises(RecalcRefusal) as info:
        session.export_frozen_scores()
    assert info.value.detail == "recalc.retarget_not_licensed"
    assert_registered_refusal(info.value)


def test_c2_cell_refusals_run_no_work_and_leave_the_session_usable() -> None:
    base = request()

    # A descendant of a treatment is not an adjustment set: nothing is fitted or kept.
    bad = replace(base, edges=(*BASE_EDGES, ("a", "w")), adjustment=("z", "w"))
    cold = CellSession()
    with pytest.raises(RecalcRefusal) as info:
        cold.execute(bad, seed=SEED)
    assert info.value.detail == "recalc.invalid_adjustment_set"
    assert_registered_refusal(info.value)
    assert not cold.is_live
    assert Stage.GRAPH not in cold.identities

    # An outcome in the adjustment set is refused the same way.
    with pytest.raises(RecalcRefusal) as info:
        CellSession().execute(replace(base, adjustment=("z", "y")), seed=SEED)
    assert info.value.detail == "recalc.invalid_adjustment_set"

    # Nothing is exportable before a run.
    with pytest.raises(RecalcNoLiveState):
        CellSession().export_frozen_scores()

    # An incompatible retarget is refused from the plan, with no work and the session intact.
    session = CellSession()
    first = session.execute(base, seed=SEED)
    session.set_retarget_support("incompatible")
    with pytest.raises(RecalcRefusal) as info:
        session.execute(replace(base, target=z_weights(base)), seed=SEED)
    assert info.value.plan is not None
    refusal = info.value.plan.first_refusal
    assert refusal is not None and refusal[0] is Stage.LAW
    assert session.is_live
    after = session.execute(base, seed=SEED)
    assert work(after) == (0, 0, 0, 0, 0), "the session still holds the base run"
    assert after.law.ate == first.law.ate


def _z_retarget_inputs() -> tuple[CellRequest, CellRequest]:
    base = request()
    return base, replace(base, target=z_weights(base))


def test_c2_cell_resume_in_a_fresh_interpreter_retargets_exported_scores_with_zero_fits(
    tmp_path: Path,
) -> None:
    base, changed = _z_retarget_inputs()
    session = CellSession()
    session.execute(base, seed=SEED)
    frozen = session.export_frozen_scores()
    assert len(frozen.identity) == 64
    assert session.export_frozen_scores().export() == frozen.export(), "an export is deterministic"

    # The in-process retarget of the live session is the reference.
    in_process = session.execute(changed, seed=SEED)
    weights = np.asarray(changed.target.weights)  # type: ignore[union-attr]

    artifact = tmp_path / "scores.bin"
    artifact.write_bytes(frozen.export())
    weights_path = tmp_path / "weights.npy"
    np.save(weights_path, weights)
    script = textwrap.dedent(
        """
        import json, sys
        import numpy as np
        from antecedent import recalc_cell as rc
        from antecedent.recalc import Stage, TargetWeights, Utility

        path, weights_path, identity, variables, edges = sys.argv[1:6]
        resumed = rc.resume_from_scores(
            open(path, "rb").read(),
            variables=json.loads(variables),
            edges=[tuple(e) for e in json.loads(edges)],
            utility=Utility(2.0, 0.5),
            expected_identity=identity,
        )
        assert not resumed.has_run
        target = TargetWeights(np.load(weights_path), ("z",))
        first = resumed.retarget(target, row_ids=resumed.row_ids)
        score = first.receipt.entry(Stage.SCORE_ARTIFACT)
        scores = resumed.score_columns()
        contrast = scores[0] - scores[1] - scores[2] + scores[3]
        w = target.weights
        later = resumed.retarget(target, utility=Utility(4.0, 0.2))
        print(json.dumps({
            "identity": resumed.identity,
            "table": list(first.receipt.status_table),
            "work": list(first.receipt.totals.as_tuple()),
            "score_status": score.status.text,
            "score_work": score.counts.total,
            "ate": first.law.ate,
            "se": first.law.std_error,
            "net": first.decision.net_benefit,
            "treat": first.decision.treat,
            "oracle": float(np.sum(w * contrast) / np.sum(w)),
            "later_work": list(later.receipt.totals.as_tuple()),
            "later_ate": later.law.ate,
            "later_net": later.decision.net_benefit,
        }))
        """
    )
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(artifact),
            str(weights_path),
            frozen.identity,
            json.dumps(VARIABLES),
            json.dumps(BASE_EDGES),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(done.stdout)

    assert report["identity"] == frozen.identity
    assert report["work"] == [1, 0, 0, 1, 1], "zero fits, zero score builds"
    assert report["score_status"] == "reused(identification)"
    assert report["score_work"] == 0, "the reused score artifact did no work"
    assert [tuple(row) for row in report["table"]] == list(
        expected(
            {
                Stage.IDENTIFICATION: "recomputed(fresh_process)",
                **TARGET_ONLY,
                Stage.LAW: "recomputed(own:law:modified)",
                Stage.DECISION: "recomputed(upstream:law<-law:modified)",
            }
        )
    )
    assert report["ate"] == pytest.approx(in_process.law.ate, abs=1e-12)
    assert report["se"] == pytest.approx(in_process.law.std_error, abs=1e-12)
    assert report["net"] == pytest.approx(in_process.decision.net_benefit, abs=1e-12)
    assert report["treat"] == in_process.decision.treat
    assert report["ate"] == pytest.approx(report["oracle"], abs=1e-10)
    fresh = rerun(changed)
    assert report["ate"] == pytest.approx(fresh.law.ate, abs=1e-12)
    assert report["se"] == pytest.approx(fresh.law.std_error, abs=1e-12)

    # The resumed session now holds its law: a utility change recomputes the decision only.
    assert report["later_work"] == [0, 0, 0, 0, 1]
    assert report["later_ate"] == report["ate"]
    assert report["later_net"] == pytest.approx(4.0 * report["ate"] - 0.2, abs=1e-12)


def test_c2_cell_resume_with_an_unchanged_request_reproduces_the_first_run() -> None:
    base = request()
    session = CellSession()
    first = session.execute(base, seed=SEED)
    resumed = session.export_frozen_scores().resume(
        variables=VARIABLES, edges=BASE_EDGES, utility=UTILITY
    )
    assert not resumed.has_run
    plan = resumed.plan()
    assert plan.status(Stage.SCORE_ARTIFACT).reused

    out = resumed.retarget()
    assert resumed.has_run
    # A fresh process never reuses a derived stage but the portable scores.
    assert out.receipt.status_table == expected(
        {
            Stage.IDENTIFICATION: "recomputed(fresh_process)",
            Stage.LAW: "recomputed(own:law:modified)",
            Stage.DECISION: "recomputed(upstream:law<-law:modified)",
        }
    )
    assert work(out) == (1, 0, 0, 1, 1)
    assert out.law.ate == first.law.ate
    assert out.law.std_error == first.law.std_error
    assert out.decision.net_benefit == first.decision.net_benefit
    assert isinstance(out.receipt, rc.ResumeReceipt)
    with pytest.raises(CausalError):
        out.receipt.export()


def test_c2_cell_resume_refuses_every_request_that_needs_data_or_a_fit() -> None:
    base = request()
    session = CellSession()
    session.execute(base, seed=SEED)
    resumed = session.export_frozen_scores().resume(
        variables=VARIABLES, edges=BASE_EDGES, utility=UTILITY
    )
    anchor = resumed.identities

    refused = {
        "changed outcome": {"declared_changes": {Stage.QUERY: "outcome=y2"}},
        "changed folds": {"declared_changes": {Stage.LEARNER_FOLDS_RNG: "seed+1"}},
        "new data": {"declared_changes": {Stage.DATA_SNAPSHOT: "other rows"}},
        "new rows": {"declared_changes": {Stage.ROW_DESIGN: "complete_case.rows=5000"}},
        "changed grid": {"declared_changes": {Stage.TREATMENT_GRID: "binary.cells.k=3"}},
        "changed graph": {"edges": (*BASE_EDGES, ("w", "y"))},
    }
    for what, arguments in refused.items():
        with pytest.raises(ScoreResumeUnavailable) as info:
            resumed.retarget(**arguments)  # type: ignore[arg-type]
        error = info.value
        assert error.detail == "recalc.unavailable_data", what
        assert error.missing == "data", what
        assert error.reason_code == "score_table_unavailable", what
        assert error.stage == "data_snapshot", what
        assert_registered_refusal(error)
        assert isinstance(error, RecalcRefusal) and isinstance(error, ScoreResumeRefusal)
        assert error.plan is not None
        first = error.plan.first_refusal
        assert first is not None and first[0] is Stage.DATA_SNAPSHOT, what
        assert error.plan.status(Stage.SCORE_ARTIFACT).refused, (
            f"{what}: the score artifact is not claimed as reused"
        )
        assert resumed.identities == anchor, f"{what}: the session is unchanged"
        assert not resumed.has_run, what
        # `plan` reports the same refusal without running anything.
        with pytest.raises(ScoreResumeUnavailable):
            resumed.plan(**arguments)  # type: ignore[arg-type]

    # New data supplied as a whole request is refused too: the session holds no data.
    with pytest.raises(ScoreResumeUnavailable) as info:
        resumed.execute(replace(base, data=columns(1.0)))
    assert info.value.missing == "data"
    assert not resumed.has_run and resumed.identities == anchor

    # Weights over other rows, or of another length, are refused up front.
    ones = np.ones(resumed.n_rows)
    swapped = resumed.row_ids.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    with pytest.raises(ScoreResumeRefusal) as info:
        resumed.retarget(TargetWeights(ones, ("z",)), row_ids=swapped)
    assert info.value.detail == "recalc.row_ids_mismatch"
    with pytest.raises(ScoreResumeRefusal) as info:
        resumed.retarget(TargetWeights(np.ones(10), ("z",)))
    assert info.value.detail == "recalc.row_count_mismatch"

    # A derived stage cannot be declared as a changed input.
    with pytest.raises(ScoreResumeRefusal) as info:
        resumed.retarget(declared_changes={Stage.LAW: "cheat"})
    assert info.value.detail == "recalc.invalid_changed_input"

    # Weights that depend on a treatment are refused by the estimator's own gate, and the
    # session returns to the artifact.
    with pytest.raises(CausalError):
        resumed.retarget(TargetWeights(ones, ("a",)))
    assert resumed.identities == anchor and not resumed.has_run

    # None of that poisoned the session: the licensed operation still runs.
    good = resumed.retarget()
    assert work(good) == (1, 0, 0, 1, 1)


def test_c2_cell_resume_refuses_resealed_foreign_or_corrupt_artifact_bytes() -> None:
    base = request()
    session = CellSession()
    session.execute(base, seed=SEED)
    frozen = session.export_frozen_scores()
    resume = {"variables": VARIABLES, "edges": BASE_EDGES, "utility": UTILITY}
    assert (
        rc.resume_from_scores(frozen.export(), expected_identity=frozen.identity, **resume).identity
        == frozen.identity
    )

    # Another run's artifact, consistently sealed, under the identity this consumer retained.
    other = CellSession()
    other.execute(replace(base, data=columns(1.0)), seed=SEED)
    foreign = other.export_frozen_scores()
    assert foreign.identity != frozen.identity
    assert rc.resume_from_scores(foreign, **resume).identity == foreign.identity
    with pytest.raises(ScoreResumeRefusal) as info:
        rc.resume_from_scores(foreign, expected_identity=frozen.identity, **resume)
    assert info.value.detail == "frozen_scores.identity_mismatch"
    assert_registered_refusal(info.value)
    with pytest.raises(ScoreResumeRefusal):
        rc.FrozenScores.load(foreign.export(), expected_identity=frozen.identity)

    # An edited byte is refused by recomputation alone; so is a truncated artifact.
    raw = bytearray(frozen.export())
    raw[len(raw) // 2] ^= 0xFF
    with pytest.raises((CausalSerializationError, ScoreResumeRefusal)):
        rc.resume_from_scores(bytes(raw), **resume)
    with pytest.raises((CausalSerializationError, ScoreResumeRefusal)):
        rc.resume_from_scores(frozen.export()[:-64], **resume)
    with pytest.raises(CausalSerializationError):
        rc.resume_from_scores(b"not an artifact", **resume)


def crossfit_columns() -> dict[str, np.ndarray]:
    n = 600
    rng = np.random.default_rng(61)
    z = rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * z)))
    t = (rng.random(n) < p).astype(np.float64)
    y = (2.0 + 0.8 * z) * t + z + 0.3 * rng.standard_normal(n)
    return {"t": t, "y": y, "z": z}


def test_c2_cell_crossfit_scores_resume_the_same_way() -> None:
    edges = (("z", "t"), ("z", "y"), ("t", "y"))
    base = RecalcRequest(
        data=crossfit_columns(), edges=edges, treatment="t", outcome="y", utility=UTILITY
    )
    session = CrossfitSession()
    first = session.execute(base, seed=61)
    assert first.receipt.totals.fold_fits == 10
    frozen = session.export_frozen_scores()

    weights = TargetWeights(np.exp(0.4 * np.asarray(base.data["z"])), ("z",))
    in_process = session.execute(replace(base, target=weights), seed=61)
    assert in_process.receipt.totals.fold_fits == 0

    resumed = rc.resume_from_scores(
        frozen,
        variables=("t", "y", "z"),
        edges=edges,
        utility=UTILITY,
        quantity="average_effect",
        expected_identity=frozen.identity,
    )
    out = resumed.retarget(weights, row_ids=resumed.row_ids)
    assert work(out) == (1, 0, 0, 1, 1)
    assert out.receipt.entry(Stage.SCORE_ARTIFACT).status.reused
    assert out.receipt.entry(Stage.SCORE_ARTIFACT).counts.total == 0
    assert out.law.ate == pytest.approx(in_process.law.ate, abs=1e-12)
    assert out.law.std_error == pytest.approx(in_process.law.std_error, abs=1e-12)
    assert out.decision.net_benefit == pytest.approx(in_process.decision.net_benefit, abs=1e-12)

    # Nothing is exportable before a run.
    with pytest.raises(RecalcNoLiveState):
        CrossfitSession().export_frozen_scores()
