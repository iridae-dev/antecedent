//! 2.3 C2: selective recalculation proven by instrumented counts.
//!
//! The route is the cross-fitted AIPW average effect through `PreparedStudy`. Each mutation
//! asserts the whole per-stage status table, the counts of work that actually ran (nuisance
//! fold fits read from the estimator's own `CrossfitNuisanceCache`, identifications, score
//! builds, reweights, decisions), and the value against an independent rerun in a fresh
//! session plus a plain-sum oracle over the frozen score table.
//!
//! The first run performs 5 folds x 2 nuisance sets (propensity, outcome) = 10 fold fits.

use antecedent::analysis::recalc_receipt::{
    RecalcOutcome, RecalcReceipt, RecalcRequest, RecalcRunError, RecalcSession, ReceiptEntry,
    ReceiptError, ReceiptRecorder, StageCounts, TargetWeights, UtilitySpec, execute_with_receipt,
};
use antecedent_core::recalc::{
    ChangedDependency, MissingDependency, RefusalReason, RequestSupport, ResumeContext,
    RetargetSupport, Stage, StageIdentities, StageIdentity, StageStatus,
};
use antecedent_core::{ExecutionContext, StreamDomain, VariableId};
use antecedent_estimate::{AipwAte, ScoreTable};
use antecedent_kernels::standard_normal;

const T: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
const W: u32 = 3;
const Y2: u32 = 4;
const N: usize = 600;
const SEED: u64 = 61;
const FIRST_RUN_FOLD_FITS: u64 = 10;

const STAGES: [Stage; 15] = [
    Stage::Graph,
    Stage::Query,
    Stage::Regime,
    Stage::Evidence,
    Stage::SourcePopulation,
    Stage::TargetPopulation,
    Stage::DataSnapshot,
    Stage::RowDesign,
    Stage::TreatmentGrid,
    Stage::LearnerFoldsRng,
    Stage::Utility,
    Stage::Identification,
    Stage::ScoreArtifact,
    Stage::Law,
    Stage::Decision,
];

type Columns = Vec<(String, Vec<f64>)>;
type Table = Vec<(String, String)>;

/// Confounded binary treatment with an effect that varies in `z`, plus an independent noise
/// column `w` and a second outcome `y2`.
fn columns(effect: f64) -> Columns {
    let mut rng = ExecutionContext::for_tests(SEED).rng.stream_for(StreamDomain::Test, 0xC2);
    let (mut t, mut y, mut z, mut w, mut y2) =
        (vec![0.0; N], vec![0.0; N], vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    for i in 0..N {
        let zi = standard_normal(&mut rng);
        let p = 1.0 / (1.0 + (-(-0.2 + 0.8 * zi)).exp());
        z[i] = zi;
        w[i] = standard_normal(&mut rng);
        t[i] = f64::from(rng.next_f64() < p);
        y[i] = (effect + 0.8 * zi) * t[i] + zi + 0.3 * standard_normal(&mut rng);
        y2[i] = -t[i] + 0.5 * zi + 0.3 * standard_normal(&mut rng);
    }
    vec![("t".into(), t), ("y".into(), y), ("z".into(), z), ("w".into(), w), ("y2".into(), y2)]
}

fn base_edges() -> Vec<(u32, u32)> {
    vec![(Z, T), (Z, Y), (T, Y), (Z, Y2), (T, Y2)]
}

fn utility() -> UtilitySpec {
    UtilitySpec { benefit_per_unit: 2.0, cost: 0.5 }
}

fn request() -> RecalcRequest {
    RecalcRequest {
        columns: columns(2.0),
        edges: base_edges(),
        treatment: T,
        outcome: Y,
        estimator: AipwAte::new().with_bootstrap_replicates(0),
        target: None,
        utility: utility(),
    }
}

fn z_weights(request: &RecalcRequest) -> TargetWeights {
    let z = &request.columns[Z as usize].1;
    TargetWeights {
        weights: z.iter().map(|v| (0.4 * v).exp()).collect(),
        depends_on: vec![VariableId::from_raw(Z)],
    }
}

fn ctx(seed: u64) -> ExecutionContext {
    ExecutionContext::for_tests(seed)
}

fn go(session: &mut RecalcSession, request: &RecalcRequest, seed: u64) -> RecalcOutcome {
    execute_with_receipt(session, request, &ctx(seed)).unwrap()
}

/// An independent rerun: a brand-new session, so no frozen score or nuisance fit is shared.
fn rerun(request: &RecalcRequest, seed: u64) -> RecalcOutcome {
    let outcome = go(&mut RecalcSession::new(), request, seed);
    assert_eq!(outcome.receipt.totals().fold_fits, FIRST_RUN_FOLD_FITS, "rerun fits afresh");
    outcome
}

fn table(outcome: &RecalcOutcome) -> Table {
    outcome.plan.entries().iter().map(|e| (e.stage.label(), e.status.to_string())).collect()
}

fn reused_string(stage: Stage) -> String {
    let dep = stage.dependencies(RetargetSupport::Licensed).first().copied().unwrap_or(stage);
    format!("reused({dep})")
}

fn expected(overrides: &[(Stage, &str)]) -> Table {
    STAGES
        .iter()
        .map(|stage| {
            let status = overrides
                .iter()
                .find(|(s, _)| s == stage)
                .map_or_else(|| reused_string(*stage), |(_, text)| (*text).to_string());
            (stage.label(), status)
        })
        .collect()
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-12, "{what}: {a} vs {b}");
}

/// Plain-sum oracle over the frozen scores: `sum w (phi_1 - phi_0) / sum w`.
fn oracle(table: &ScoreTable, weights: &[f64]) -> f64 {
    let column = |arm: u32| {
        let at = table.columns.iter().position(|c| c.arm == arm && c.threshold.is_none()).unwrap();
        table.column(at).unwrap().to_vec()
    };
    let (control, active) = (column(0), column(1));
    let total: f64 = weights.iter().sum();
    active.iter().zip(&control).zip(weights).map(|((a, c), w)| (a - c) * w).sum::<f64>() / total
}

fn ones() -> Vec<f64> {
    vec![1.0; N]
}

fn counts(outcome: &RecalcOutcome) -> StageCounts {
    *outcome.receipt.totals()
}

fn assert_no_work(c: &StageCounts, decisions: u64) {
    assert_eq!(c.identifications, 0);
    assert_eq!(c.fold_fits, 0);
    assert_eq!(c.score_computations, 0);
    assert_eq!(c.reweights, 0);
    assert_eq!(c.decisions, decisions);
}

fn assert_reused_identities_equal(a: &RecalcOutcome, b: &RecalcOutcome) {
    for stage in b.plan.reused() {
        assert_eq!(a.plan.identity(stage), b.plan.identity(stage), "{stage}");
        assert_eq!(
            a.receipt.entry(stage).unwrap().identity,
            b.receipt.entry(stage).unwrap().identity
        );
    }
}

fn refused_plan(error: RecalcRunError) -> antecedent_core::recalc::RecalcPlan {
    match error {
        RecalcRunError::Refused(plan) => *plan,
        other => panic!("expected a refused plan, got {other:?}"),
    }
}

#[test]
fn c2_recalc_first_run_fits_and_a_rerun_in_a_fresh_session_fits_again() {
    let request = request();
    let mut session = RecalcSession::new();
    let first = go(&mut session, &request, SEED);
    // Nothing is previously known: every stage is recomputed and none reused.
    assert!(first.plan.reused().is_empty());
    assert_eq!(first.plan.recomputed().len(), STAGES.len());
    let c = counts(&first);
    assert_eq!(
        (c.identifications, c.fold_fits, c.score_computations, c.reweights, c.decisions),
        (1, FIRST_RUN_FOLD_FITS, 1, 1, 1)
    );
    // The per-call batch cache is not a persistent fit cache: a second session fits again.
    let again = rerun(&request, SEED);
    assert_eq!(again.law.ate.to_bits(), first.law.ate.to_bits());
    assert_eq!(again.receipt.identity(), first.receipt.identity());
    // The law is the plain-sum oracle of the frozen scores.
    close(first.law.ate, oracle(session.score_table().unwrap(), &ones()), "law vs oracle");
    close(
        first.decision.net_benefit,
        utility().benefit_per_unit * first.law.ate - utility().cost,
        "decision rule",
    );
    assert_eq!(first.decision.treat, first.decision.net_benefit > 0.0);
}

#[test]
fn c2_recalc_utility_only_change_recomputes_the_decision_with_zero_fit_work() {
    let base = request();
    let mut session = RecalcSession::new();
    let first = go(&mut session, &base, SEED);

    let mut changed = base.clone();
    changed.utility = UtilitySpec { benefit_per_unit: 3.0, cost: 0.1 };
    let second = go(&mut session, &changed, SEED);

    let overrides = [
        (Stage::Utility, "recomputed(own:utility:modified)"),
        (Stage::Decision, "recomputed(upstream:utility<-utility:modified)"),
    ];
    assert_eq!(table(&second), expected(&overrides));
    assert_eq!(second.plan.recomputed_computations(), vec![Stage::Decision]);
    assert_no_work(&counts(&second), 1);
    assert_reused_identities_equal(&first, &second);
    assert_eq!(second.law.ate.to_bits(), first.law.ate.to_bits());
    assert_eq!(second.law.std_error.to_bits(), first.law.std_error.to_bits());

    let independent = rerun(&changed, SEED);
    close(second.decision.net_benefit, independent.decision.net_benefit, "net benefit");
    close(second.decision.net_benefit, 3.0 * independent.law.ate - 0.1, "decision oracle");
    assert_eq!(second.decision.treat, independent.decision.treat);
}

#[test]
fn c2_recalc_compatible_target_weight_change_reuses_frozen_scores() {
    let base = request();
    let mut session = RecalcSession::new();
    let first = go(&mut session, &base, SEED);

    let mut changed = base.clone();
    changed.target = Some(z_weights(&base));
    let second = go(&mut session, &changed, SEED);

    let overrides = [
        (Stage::TargetPopulation, "recomputed(own:target_population:modified)"),
        (Stage::Law, "recomputed(upstream:target_population<-target_population:modified)"),
        (Stage::Decision, "recomputed(upstream:law<-target_population:modified)"),
    ];
    assert_eq!(table(&second), expected(&overrides));
    // No identification, no fit, no score build: one reweight and one decision.
    let c = counts(&second);
    assert_eq!(
        (c.identifications, c.fold_fits, c.score_computations, c.reweights, c.decisions),
        (0, 0, 0, 1, 1)
    );
    assert_reused_identities_equal(&first, &second);
    assert!((second.law.ate - first.law.ate).abs() > 0.05, "the target must change the law");

    let weights = &changed.target.as_ref().unwrap().weights;
    close(second.law.ate, oracle(session.score_table().unwrap(), weights), "plain-sum oracle");
    let independent = rerun(&changed, SEED);
    close(second.law.ate, independent.law.ate, "retargeted law vs independent rerun");
    close(second.law.std_error, independent.law.std_error, "retargeted se vs independent rerun");
    close(second.decision.net_benefit, independent.decision.net_benefit, "decision");
}

#[test]
fn c2_recalc_changed_folds_and_new_outcomes_refit_without_reidentifying() {
    let base = request();
    let mut session = RecalcSession::new();
    let first = go(&mut session, &base, SEED);

    // Changed folds/RNG: the same data under another master seed.
    let seed = SEED + 1;
    let folds = go(&mut session, &base, seed);
    let overrides = [
        (Stage::LearnerFoldsRng, "recomputed(own:learner_folds_rng:modified)"),
        (
            Stage::ScoreArtifact,
            "recomputed(upstream:learner_folds_rng<-learner_folds_rng:modified)",
        ),
        (Stage::Law, "recomputed(upstream:score_artifact<-learner_folds_rng:modified)"),
        (Stage::Decision, "recomputed(upstream:law<-learner_folds_rng:modified)"),
    ];
    assert_eq!(table(&folds), expected(&overrides));
    let c = counts(&folds);
    assert_eq!((c.identifications, c.score_computations, c.reweights), (0, 1, 1));
    assert!(c.fold_fits >= FIRST_RUN_FOLD_FITS && c.fold_fits % 5 == 0, "{}", c.fold_fits);
    assert_ne!(folds.law.ate.to_bits(), first.law.ate.to_bits(), "other folds, other scores");
    close(folds.law.ate, rerun(&base, seed).law.ate, "changed folds vs independent rerun");

    // New outcomes: same seed, different outcome column.
    let mut fresh = base.clone();
    fresh.columns = columns(1.5);
    let outcomes = go(&mut session, &fresh, seed);
    let overrides = [
        (Stage::DataSnapshot, "recomputed(own:data_snapshot:modified)"),
        (Stage::ScoreArtifact, "recomputed(upstream:data_snapshot<-data_snapshot:modified)"),
        (Stage::Law, "recomputed(upstream:score_artifact<-data_snapshot:modified)"),
        (Stage::Decision, "recomputed(upstream:law<-data_snapshot:modified)"),
    ];
    assert_eq!(table(&outcomes), expected(&overrides));
    let o = counts(&outcomes);
    assert_eq!((o.identifications, o.score_computations, o.reweights), (0, 1, 1));
    assert_eq!(o.fold_fits, c.fold_fits, "both refit the same cross-fitted design");
    assert!((outcomes.law.ate - folds.law.ate).abs() > 0.05);
    close(outcomes.law.ate, rerun(&fresh, seed).law.ate, "new outcomes vs independent rerun");
}

#[test]
fn c2_recalc_graph_or_query_change_reidentifies() {
    let base = request();
    let mut session = RecalcSession::new();
    let first = go(&mut session, &base, SEED);

    // Graph change: an extra edge from the independent column.
    let mut graph = base.clone();
    graph.edges.push((W, Y));
    let regraphed = go(&mut session, &graph, SEED);
    let overrides = [
        (Stage::Graph, "recomputed(own:graph:modified)"),
        (Stage::Identification, "recomputed(upstream:graph<-graph:modified)"),
        (Stage::ScoreArtifact, "recomputed(upstream:identification<-graph:modified)"),
        (Stage::Law, "recomputed(upstream:score_artifact<-graph:modified)"),
        (Stage::Decision, "recomputed(upstream:law<-graph:modified)"),
    ];
    assert_eq!(table(&regraphed), expected(&overrides));
    let c = counts(&regraphed);
    assert_eq!((c.identifications, c.score_computations, c.reweights), (1, 1, 1));
    assert!(c.fold_fits >= FIRST_RUN_FOLD_FITS && c.fold_fits % 5 == 0);
    close(regraphed.law.ate, rerun(&graph, SEED).law.ate, "graph change vs independent rerun");

    // Query change: the second outcome on the same graph and data.
    let mut query = graph.clone();
    query.outcome = Y2;
    let requeried = go(&mut session, &query, SEED);
    let overrides = [
        (Stage::Query, "recomputed(own:query:modified)"),
        (Stage::Identification, "recomputed(upstream:query<-query:modified)"),
        (Stage::ScoreArtifact, "recomputed(upstream:identification<-query:modified)"),
        (Stage::Law, "recomputed(upstream:score_artifact<-query:modified)"),
        (Stage::Decision, "recomputed(upstream:law<-query:modified)"),
    ];
    assert_eq!(table(&requeried), expected(&overrides));
    assert_eq!(counts(&requeried).identifications, 1);
    close(requeried.law.ate, rerun(&query, SEED).law.ate, "query change vs independent rerun");
    assert!((requeried.law.ate - first.law.ate).abs() > 0.5, "another outcome, another law");
}

#[test]
fn c2_recalc_an_undeclared_retarget_refits_instead_of_reusing_scores() {
    let base = request();
    let mut session = RecalcSession::new();
    session.set_retarget_support(RetargetSupport::NotDeclared);
    go(&mut session, &base, SEED);

    let mut changed = base.clone();
    changed.target = Some(z_weights(&base));
    let second = go(&mut session, &changed, SEED);
    let overrides = [
        (Stage::TargetPopulation, "recomputed(own:target_population:modified)"),
        (
            Stage::ScoreArtifact,
            "recomputed(upstream:target_population<-target_population:modified)",
        ),
        (Stage::Law, "recomputed(upstream:score_artifact<-target_population:modified)"),
        (Stage::Decision, "recomputed(upstream:law<-target_population:modified)"),
    ];
    assert_eq!(table(&second), expected(&overrides));
    let c = counts(&second);
    assert_eq!((c.identifications, c.score_computations, c.reweights), (0, 1, 1));
    assert!(c.fold_fits >= FIRST_RUN_FOLD_FITS, "the refit really ran");
    close(second.law.ate, rerun(&changed, SEED).law.ate, "refit vs independent rerun");
}

#[test]
fn c2_recalc_refusals_run_no_work_and_leave_the_session_untouched() {
    let base = request();
    let mut session = RecalcSession::new();

    // Off-grid on a cold session: nothing runs and nothing becomes live.
    session.set_request_support(RequestSupport::OffGrid {
        licensed_route: Some("transport.smoothed_dose"),
    });
    let plan = refused_plan(execute_with_receipt(&mut session, &base, &ctx(SEED)).unwrap_err());
    assert_eq!(
        plan.status(Stage::TreatmentGrid),
        Some(&StageStatus::Refused {
            reason: RefusalReason::OffGrid { licensed_route: Some("transport.smoothed_dose") }
        })
    );
    for stage in [Stage::ScoreArtifact, Stage::Law, Stage::Decision] {
        assert!(matches!(
            plan.status(stage),
            Some(StageStatus::Refused { reason: RefusalReason::Blocked { .. } })
        ));
    }
    assert!(!session.is_live());
    assert_eq!(session.identities(), &StageIdentities::new());

    // A good run, then an unsupported request, then the good request again.
    session.set_request_support(RequestSupport::OnGrid);
    let first = go(&mut session, &base, SEED);
    let known = session.identities().clone();
    session.set_request_support(RequestSupport::Unsupported { licensed_route: None });
    let plan = refused_plan(execute_with_receipt(&mut session, &base, &ctx(SEED)).unwrap_err());
    assert_eq!(plan.first_refusal().unwrap().1.detail(), "recalc.unsupported_request");
    assert!(session.is_live());
    assert_eq!(session.identities(), &known);
    session.set_request_support(RequestSupport::OnGrid);
    let again = go(&mut session, &base, SEED);
    assert_eq!(table(&again), expected(&[]));
    assert_no_work(&counts(&again), 0);
    assert_eq!(again.law.ate.to_bits(), first.law.ate.to_bits());

    // Weights that are not a licensed function of the adjustment set refuse at the law.
    session.set_retarget_support(RetargetSupport::Incompatible);
    let mut changed = base.clone();
    changed.target = Some(z_weights(&base));
    let plan = refused_plan(execute_with_receipt(&mut session, &changed, &ctx(SEED)).unwrap_err());
    assert_eq!(
        plan.status(Stage::Law),
        Some(&StageStatus::Refused { reason: RefusalReason::RetargetIncompatible })
    );
    assert_eq!(session.identities(), &known, "a refusal changes nothing");
}

#[test]
fn c2_recalc_a_loaded_result_cannot_claim_cache_reuse_in_a_fresh_process() {
    let base = request();
    let mut original = RecalcSession::new();
    let first = go(&mut original, &base, SEED);
    // What an ordinary loaded result keeps: the identities, with no prepared study.
    let loaded = original.identities().clone();

    // Nothing supplied: the missing data snapshot is the specific unavailable result.
    let mut bare = RecalcSession::resume(loaded.clone(), ResumeContext::default());
    let plan = refused_plan(execute_with_receipt(&mut bare, &base, &ctx(SEED)).unwrap_err());
    assert_eq!(
        plan.first_refusal(),
        Some((
            Stage::DataSnapshot,
            RefusalReason::Unavailable { missing: MissingDependency::Data }
        ))
    );
    assert_eq!(
        plan.status(Stage::ScoreArtifact),
        Some(&StageStatus::Refused {
            reason: RefusalReason::Unavailable { missing: MissingDependency::Fit }
        })
    );

    // Portable scores are declared but this wrapper cannot load them: no reuse is claimed.
    let claimed =
        ResumeContext { portable_scores: true, supplied_data: true, ..Default::default() };
    let mut scores = RecalcSession::resume(loaded.clone(), claimed);
    let error = execute_with_receipt(&mut scores, &base, &ctx(SEED)).unwrap_err();
    assert!(matches!(error, RecalcRunError::NoLiveState(Stage::ScoreArtifact)), "{error:?}");

    // Supplied data resumes by recomputing every derived stage, with the same fit work as the
    // first run, and never by reuse.
    let supplied = ResumeContext { supplied_data: true, ..ResumeContext::default() };
    let mut resumed = RecalcSession::resume(loaded, supplied);
    let second = go(&mut resumed, &base, SEED);
    for entry in second.plan.entries().iter().filter(|e| !e.stage.is_input()) {
        assert_eq!(
            entry.status,
            StageStatus::Recomputed { because: ChangedDependency::FreshProcess },
            "{}",
            entry.stage
        );
    }
    assert_eq!(counts(&second), counts(&first));
    assert_eq!(second.law.ate.to_bits(), first.law.ate.to_bits());
    // Once live, the resumed session reuses like any in-process one.
    let mut changed = base.clone();
    changed.utility = UtilitySpec { benefit_per_unit: 1.0, cost: 0.0 };
    let third = go(&mut resumed, &changed, SEED);
    assert_no_work(&counts(&third), 1);
}

fn entry(stage: Stage, status: StageStatus, counts: StageCounts) -> ReceiptEntry {
    ReceiptEntry { stage, status, identity: StageIdentity::of(&stage.label(), &[b"x"]), counts }
}

#[test]
fn c2_recalc_receipt_identity_is_order_independent_and_counts_must_match_status() {
    let base = request();
    let first = go(&mut RecalcSession::new(), &base, SEED);
    let mut shuffled: Vec<ReceiptEntry> = first.receipt.entries().to_vec();
    shuffled.reverse();
    shuffled.rotate_left(4);
    let rebuilt = RecalcReceipt::new(shuffled).unwrap();
    assert_eq!(rebuilt.identity(), first.receipt.identity());
    assert_eq!(rebuilt.status_table(), first.receipt.status_table());
    assert_eq!(rebuilt.totals(), first.receipt.totals());

    // A utility-only change has a different canonical receipt.
    let mut session = RecalcSession::new();
    go(&mut session, &base, SEED);
    let mut changed = base.clone();
    changed.utility = UtilitySpec { benefit_per_unit: 3.0, cost: 0.1 };
    let second = go(&mut session, &changed, SEED);
    assert_ne!(second.receipt.identity(), first.receipt.identity());

    let reused = StageStatus::Reused { dependency: Stage::Graph };
    let recomputed = StageStatus::Recomputed { because: ChangedDependency::FreshProcess };
    let none = StageCounts::default();
    let fit = StageCounts { fold_fits: 10, ..StageCounts::default() };
    let decided = StageCounts { decisions: 1, ..StageCounts::default() };
    // Reuse that did work is a contradiction, not a receipt.
    assert_eq!(
        RecalcReceipt::new(vec![entry(Stage::ScoreArtifact, reused, fit)]),
        Err(ReceiptError::UnexpectedWork { stage: Stage::ScoreArtifact })
    );
    // A recomputed decision that never ran the decision is a contradiction as well.
    assert!(matches!(
        RecalcReceipt::new(vec![entry(Stage::Decision, recomputed, none)]),
        Err(ReceiptError::MissingWork { stage: Stage::Decision, .. })
    ));
    assert!(RecalcReceipt::new(vec![entry(Stage::Decision, recomputed, decided)]).is_ok());
    // Work counted against a stage that does not own it is refused.
    assert_eq!(
        RecalcReceipt::new(vec![entry(Stage::Decision, recomputed, fit)]),
        Err(ReceiptError::UnexpectedWork { stage: Stage::Decision })
    );
    assert_eq!(
        RecalcReceipt::new(vec![
            entry(Stage::Graph, reused, none),
            entry(Stage::Graph, reused, none)
        ]),
        Err(ReceiptError::DuplicateStage(Stage::Graph))
    );
}

#[test]
fn c2_recalc_recorder_counts_only_fits_a_scope_actually_ran() {
    let mut recorder = ReceiptRecorder::new();
    let out = recorder.measured(|| 7);
    assert_eq!(out, 7);
    assert_eq!(recorder.counts(Stage::ScoreArtifact).fold_fits, 0);
    assert_eq!(recorder.counts(Stage::Law).total(), 0);
}
