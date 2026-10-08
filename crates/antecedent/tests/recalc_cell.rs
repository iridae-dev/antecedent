//! 2.3 C2 remainder: cell-AIPW score reuse and portable score resume, proven by counts.
//!
//! The route is the cell-saturated AIPW interaction contrast over two binary treatments. Each
//! mutation asserts the whole per-stage status table, the work that actually ran (cell-model
//! fits read from the estimator's own instrument and cross-checked by measuring the call from
//! outside, identifications, score builds, reweights, decisions) and the value against an
//! independent fit-and-summarize in this file.
//!
//! The first run fits 5 folds x (1 multinomial propensity + 4 cell outcome regressions) = 25
//! cell models.

use antecedent::analysis::recalc_cell::{
    CellRequest, CellSession, CellSpec, ScoreQuantity, ScoreResumeRequest, ScoreResumeSession,
    execute_cell_with_receipt, execute_resumed_retarget, freeze_crossfit_scores,
};
use antecedent::analysis::recalc_receipt::{
    RecalcOutcome, RecalcRequest, RecalcRunError, RecalcSession, TargetWeights, UtilitySpec,
    execute_with_receipt,
};
use antecedent_core::recalc::{
    MissingDependency, RecalcPlan, RefusalReason, RetargetSupport, Stage, StageIdentity,
    StageStatus,
};
use antecedent_core::{ExecutionContext, OutcomeFunctional, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::cell_aipw::{count_cell_model_fits, summarize_with_contrast};
use antecedent_estimate::{AipwAte, CellSaturatedAipw, ScoreTable};
use antecedent_io::frozen_scores_artifact::{FrozenScoreTable, decode_parts, encode_parts};
use antecedent_kernels::standard_normal;

const A: u32 = 0;
const D: u32 = 1;
const Y: u32 = 2;
const Z: u32 = 3;
const W: u32 = 4;
const Y2: u32 = 5;
const N: usize = 1200;
const SEED: u64 = 71;
const FIRST_RUN_FITS: u64 = 25;

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

/// Two confounded binary treatments with an interaction of size `effect`, an isolated noise
/// column `w` and a second outcome `y2`.
fn columns(effect: f64) -> Columns {
    let mut rng = ExecutionContext::for_tests(SEED).rng.stream_for(StreamDomain::Test, 0xC3);
    let (mut a, mut d, mut y, mut z, mut w, mut y2) =
        (vec![0.0; N], vec![0.0; N], vec![0.0; N], vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    for i in 0..N {
        let zi = standard_normal(&mut rng);
        let p = 1.0 / (1.0 + (-(0.5 * zi)).exp());
        z[i] = zi;
        w[i] = standard_normal(&mut rng);
        a[i] = f64::from(rng.next_f64() < p);
        d[i] = f64::from(rng.next_f64() < 0.5);
        y[i] = effect * a[i] * d[i] + 0.5 * a[i] + 0.2 * zi + 0.25 * standard_normal(&mut rng);
        y2[i] = -a[i] + 0.5 * d[i] + 0.3 * zi + 0.25 * standard_normal(&mut rng);
    }
    vec![
        ("a".into(), a),
        ("d".into(), d),
        ("y".into(), y),
        ("z".into(), z),
        ("w".into(), w),
        ("y2".into(), y2),
    ]
}

fn base_edges() -> Vec<(u32, u32)> {
    vec![(Z, A), (Z, D), (Z, Y), (A, Y), (D, Y), (Z, Y2), (A, Y2), (D, Y2)]
}

fn utility() -> UtilitySpec {
    UtilitySpec { benefit_per_unit: 2.0, cost: 0.5 }
}

fn request() -> CellRequest {
    CellRequest {
        columns: columns(1.5),
        spec: CellSpec {
            edges: base_edges(),
            treatments: vec![A, D],
            outcome: Y,
            adjustment: vec![Z],
            estimator: CellSaturatedAipw::new(),
            quantity: ScoreQuantity::Interaction,
            target: None,
            utility: utility(),
        },
    }
}

fn z_weights(request: &CellRequest) -> TargetWeights {
    let z = &request.columns[Z as usize].1;
    TargetWeights {
        weights: z.iter().map(|v| (0.4 * v).exp()).collect(),
        depends_on: vec![VariableId::from_raw(Z)],
    }
}

fn ctx(seed: u64) -> ExecutionContext {
    ExecutionContext::for_tests(seed)
}

/// Run one call and cross-check the receipt's fit count against the same instrument read from
/// outside the executor.
fn go(session: &mut CellSession, request: &CellRequest, seed: u64) -> RecalcOutcome {
    let (result, measured) =
        count_cell_model_fits(|| execute_cell_with_receipt(session, request, &ctx(seed)));
    let outcome = result.unwrap();
    assert_eq!(measured, outcome.receipt.totals().fold_fits, "receipt fits are the measured fits");
    outcome
}

/// An independent rerun in a brand-new session: nothing is shared with the caller's session.
fn rerun(request: &CellRequest, seed: u64) -> RecalcOutcome {
    let outcome = go(&mut CellSession::new(), request, seed);
    assert_eq!(outcome.receipt.totals().fold_fits, FIRST_RUN_FITS, "the rerun fits afresh");
    outcome
}

/// Independent oracle: fit the cell scores directly and summarize the interaction under
/// `weights`, with no session, plan or receipt involved.
fn independent(columns: &Columns, seed: u64, weights: Option<&[f64]>) -> (f64, f64, ScoreTable) {
    let borrowed: Vec<(&str, &[f64])> =
        columns.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let table = CellSaturatedAipw::new()
        .with_fold_seed(seed)
        .fit_scores(
            &data,
            &[VariableId::from_raw(A), VariableId::from_raw(D)],
            VariableId::from_raw(Y),
            &[VariableId::from_raw(Z)],
            &OutcomeFunctional::Mean,
            None,
        )
        .unwrap();
    let ones = vec![1.0; table.n_rows];
    let weights = weights.unwrap_or(&ones);
    let (_, contrast) = summarize_with_contrast(&table, "interaction", Some(weights)).unwrap();
    (contrast.value, contrast.se, table)
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

fn counts(outcome: &RecalcOutcome) -> [u64; 5] {
    let c = outcome.receipt.totals();
    [c.identifications, c.fold_fits, c.score_computations, c.reweights, c.decisions]
}

fn refused_plan(error: RecalcRunError) -> RecalcPlan {
    match error {
        RecalcRunError::Refused(plan) => *plan,
        other => panic!("expected a refused plan, got {other:?}"),
    }
}

fn request_detail(error: &RecalcRunError) -> &'static str {
    match error {
        RecalcRunError::Request(detail) => detail,
        other => panic!("expected a request refusal, got {other:?}"),
    }
}

const TARGET_ONLY: [(Stage, &str); 3] = [
    (Stage::TargetPopulation, "recomputed(own:target_population:modified)"),
    (Stage::Law, "recomputed(upstream:target_population<-target_population:modified)"),
    (Stage::Decision, "recomputed(upstream:law<-target_population:modified)"),
];

#[test]
fn c2_cell_first_run_counts_the_cell_fits_and_matches_an_independent_fit() {
    let request = request();
    let mut session = CellSession::new();
    let first = go(&mut session, &request, SEED);

    assert!(first.plan.reused().is_empty());
    assert_eq!(first.plan.recomputed().len(), STAGES.len());
    assert_eq!(counts(&first), [1, FIRST_RUN_FITS, 1, 1, 1]);

    // The same fit, measured with no executor around it, is 25 cell models.
    let (oracle, fits) = count_cell_model_fits(|| independent(&request.columns, SEED, None));
    assert_eq!(fits, FIRST_RUN_FITS);
    close(first.law.ate, oracle.0, "interaction vs independent fit");
    close(first.law.std_error, oracle.1, "se vs independent fit");
    assert_eq!(session.score_table().unwrap(), &oracle.2, "the frozen scores are the fit's");
    assert!((first.law.ate - 1.5).abs() < 0.4, "recovers the interaction: {}", first.law.ate);
    close(first.decision.net_benefit, 2.0 * first.law.ate - 0.5, "decision rule");
    assert_eq!(first.decision.treat, first.decision.net_benefit > 0.0);

    // A second session fits again: the executor keeps no persistent fit cache.
    let again = rerun(&request, SEED);
    assert_eq!(again.law.ate.to_bits(), first.law.ate.to_bits());
    assert_eq!(again.receipt.identity(), first.receipt.identity());
}

#[test]
fn c2_cell_utility_only_change_recomputes_the_decision_with_zero_fits() {
    let base = request();
    let mut session = CellSession::new();
    let first = go(&mut session, &base, SEED);

    let mut changed = base.clone();
    changed.spec.utility = UtilitySpec { benefit_per_unit: 3.0, cost: 0.1 };
    let second = go(&mut session, &changed, SEED);

    let overrides = [
        (Stage::Utility, "recomputed(own:utility:modified)"),
        (Stage::Decision, "recomputed(upstream:utility<-utility:modified)"),
    ];
    assert_eq!(table(&second), expected(&overrides));
    assert_eq!(second.plan.recomputed_computations(), vec![Stage::Decision]);
    assert_eq!(counts(&second), [0, 0, 0, 0, 1]);
    assert_eq!(second.law.ate.to_bits(), first.law.ate.to_bits());
    assert_eq!(second.law.std_error.to_bits(), first.law.std_error.to_bits());
    let independent_run = rerun(&changed, SEED);
    close(second.decision.net_benefit, independent_run.decision.net_benefit, "net benefit");
    close(second.decision.net_benefit, 3.0 * first.law.ate - 0.1, "decision oracle");
}

#[test]
fn c2_cell_compatible_target_weight_change_reuses_the_frozen_cell_scores() {
    let base = request();
    let mut session = CellSession::new();
    let first = go(&mut session, &base, SEED);
    let frozen = session.score_table().unwrap().clone();

    let mut changed = base.clone();
    changed.spec.target = Some(z_weights(&base));
    let second = go(&mut session, &changed, SEED);

    assert_eq!(table(&second), expected(&TARGET_ONLY));
    // Zero refits, no identification, no score build: one reweight and one decision.
    assert_eq!(counts(&second), [0, 0, 0, 1, 1]);
    assert_eq!(session.score_table().unwrap(), &frozen, "the scores were not touched");
    assert!((second.law.ate - first.law.ate).abs() > 0.02, "the target must move the law");

    let weights = changed.spec.target.as_ref().unwrap().weights.clone();
    let oracle = independent(&base.columns, SEED, Some(&weights));
    close(second.law.ate, oracle.0, "retargeted law vs independent fit");
    close(second.law.std_error, oracle.1, "retargeted se vs independent fit");
    let fresh = rerun(&changed, SEED);
    close(second.law.ate, fresh.law.ate, "retargeted law vs independent rerun");
    close(second.law.std_error, fresh.law.std_error, "retargeted se vs independent rerun");
    close(second.decision.net_benefit, fresh.decision.net_benefit, "decision");
}

#[test]
fn c2_cell_changed_folds_and_new_outcomes_refit_without_reidentifying() {
    let base = request();
    let mut session = CellSession::new();
    let first = go(&mut session, &base, SEED);

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
    assert_eq!(counts(&folds), [0, FIRST_RUN_FITS, 1, 1, 1]);
    assert_ne!(folds.law.ate.to_bits(), first.law.ate.to_bits(), "other folds, other scores");
    close(folds.law.ate, independent(&base.columns, seed, None).0, "folds vs independent fit");

    let mut fresh = base.clone();
    fresh.columns = columns(1.0);
    let outcomes = go(&mut session, &fresh, seed);
    let overrides = [
        (Stage::DataSnapshot, "recomputed(own:data_snapshot:modified)"),
        (Stage::ScoreArtifact, "recomputed(upstream:data_snapshot<-data_snapshot:modified)"),
        (Stage::Law, "recomputed(upstream:score_artifact<-data_snapshot:modified)"),
        (Stage::Decision, "recomputed(upstream:law<-data_snapshot:modified)"),
    ];
    assert_eq!(table(&outcomes), expected(&overrides));
    assert_eq!(counts(&outcomes), [0, FIRST_RUN_FITS, 1, 1, 1]);
    assert!((outcomes.law.ate - folds.law.ate).abs() > 0.2);
    close(outcomes.law.ate, independent(&fresh.columns, seed, None).0, "outcomes vs fit");
    close(outcomes.law.ate, rerun(&fresh, seed).law.ate, "outcomes vs independent rerun");
}

#[test]
fn c2_cell_graph_or_query_change_reidentifies_and_refits() {
    let base = request();
    let mut session = CellSession::new();
    let first = go(&mut session, &base, SEED);

    let mut graph = base.clone();
    graph.spec.edges.push((W, Y));
    let regraphed = go(&mut session, &graph, SEED);
    let overrides = [
        (Stage::Graph, "recomputed(own:graph:modified)"),
        (Stage::Identification, "recomputed(upstream:graph<-graph:modified)"),
        (Stage::ScoreArtifact, "recomputed(upstream:identification<-graph:modified)"),
        (Stage::Law, "recomputed(upstream:score_artifact<-graph:modified)"),
        (Stage::Decision, "recomputed(upstream:law<-graph:modified)"),
    ];
    assert_eq!(table(&regraphed), expected(&overrides));
    assert_eq!(counts(&regraphed), [1, FIRST_RUN_FITS, 1, 1, 1]);
    close(regraphed.law.ate, first.law.ate, "an irrelevant edge leaves the scores the same");

    let mut query = graph.clone();
    query.spec.outcome = Y2;
    let requeried = go(&mut session, &query, SEED);
    let overrides = [
        (Stage::Query, "recomputed(own:query:modified)"),
        (Stage::Identification, "recomputed(upstream:query<-query:modified)"),
        (Stage::ScoreArtifact, "recomputed(upstream:identification<-query:modified)"),
        (Stage::Law, "recomputed(upstream:score_artifact<-query:modified)"),
        (Stage::Decision, "recomputed(upstream:law<-query:modified)"),
    ];
    assert_eq!(table(&requeried), expected(&overrides));
    assert_eq!(counts(&requeried), [1, FIRST_RUN_FITS, 1, 1, 1]);
    close(requeried.law.ate, rerun(&query, SEED).law.ate, "query change vs independent rerun");
    assert!((requeried.law.ate - first.law.ate).abs() > 0.5, "another outcome, another law");
}

#[test]
fn c2_cell_an_undeclared_retarget_refits_instead_of_reusing_scores() {
    let base = request();
    let mut session = CellSession::new();
    session.set_retarget_support(RetargetSupport::NotDeclared);
    go(&mut session, &base, SEED);

    let mut changed = base.clone();
    changed.spec.target = Some(z_weights(&base));
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
    assert_eq!(counts(&second), [0, FIRST_RUN_FITS, 1, 1, 1]);
    close(second.law.ate, rerun(&changed, SEED).law.ate, "refit vs independent rerun");

    // An unlicensed route cannot be exported either.
    let error = session.export_scores().unwrap_err();
    assert_eq!(request_detail(&error), "recalc.retarget_not_licensed");
}

#[test]
fn c2_cell_refusals_run_no_work_and_leave_the_session_usable() {
    let base = request();

    // A descendant of a treatment is not an adjustment set: nothing is fitted or kept.
    let mut bad = base.clone();
    bad.spec.edges.push((A, W));
    bad.spec.adjustment = vec![Z, W];
    let mut cold = CellSession::new();
    let (result, fits) =
        count_cell_model_fits(|| execute_cell_with_receipt(&mut cold, &bad, &ctx(SEED)));
    assert_eq!(request_detail(&result.unwrap_err()), "recalc.invalid_adjustment_set");
    assert_eq!(fits, 0, "the screen refuses before any fit");
    assert!(!cold.is_live());
    assert!(!cold.identities().contains(Stage::Graph));

    // An outcome in the adjustment set is refused the same way.
    let mut outcome_adjusted = base.clone();
    outcome_adjusted.spec.adjustment = vec![Z, Y];
    let error = execute_cell_with_receipt(&mut CellSession::new(), &outcome_adjusted, &ctx(SEED))
        .unwrap_err();
    assert_eq!(request_detail(&error), "recalc.invalid_adjustment_set");

    // An incompatible retarget is refused from the plan, with no work and the session intact.
    let mut session = CellSession::new();
    let first = go(&mut session, &base, SEED);
    session.set_retarget_support(RetargetSupport::Incompatible);
    let mut changed = base.clone();
    changed.spec.target = Some(z_weights(&base));
    let (result, fits) =
        count_cell_model_fits(|| execute_cell_with_receipt(&mut session, &changed, &ctx(SEED)));
    let plan = refused_plan(result.unwrap_err());
    assert_eq!(fits, 0);
    assert_eq!(
        plan.first_refusal(),
        Some((Stage::Law, RefusalReason::RetargetIncompatible)),
        "{plan:?}"
    );
    assert!(session.is_live());
    let after = go(&mut session, &base, SEED);
    assert_eq!(counts(&after), [0, 0, 0, 0, 0], "the session still holds the base run");
    assert_eq!(after.law.ate.to_bits(), first.law.ate.to_bits());
}

#[test]
fn c2_cell_resume_in_a_fresh_session_retargets_the_exported_scores_with_zero_fits() {
    let base = request();
    let mut session = CellSession::new();
    let first = go(&mut session, &base, SEED);
    let artifact = session.export_scores().unwrap();
    let identity = artifact.identity().to_owned();
    assert_eq!(artifact.input_rows(), N as u64);
    let again = session.export_scores().unwrap();
    assert_eq!(again.identity(), identity, "an export is deterministic");
    assert_eq!(
        again.to_bytes("c2-cell-scores").unwrap(),
        artifact.to_bytes("c2-cell-scores").unwrap()
    );
    let bytes = artifact.to_bytes("c2-cell-scores").unwrap();

    // In-process retarget of the live session: the reference.
    let mut changed = base.clone();
    changed.spec.target = Some(z_weights(&base));
    let in_process = go(&mut session, &changed, SEED);

    // A fresh session built only from the artifact bytes: no data, no fit, no study.
    let mut fresh = ScoreResumeSession::resume_from_score_bytes(&bytes, Some(&identity)).unwrap();
    assert_eq!(fresh.artifact_identity(), identity);
    assert!(!fresh.has_run());
    let row_ids = fresh.score_table().row_index.to_vec();
    let request = ScoreResumeRequest {
        n_variables: 6,
        edges: base_edges(),
        target: changed.spec.target.clone(),
        target_row_ids: Some(row_ids.clone()),
        utility: utility(),
        quantity: ScoreQuantity::Interaction,
        changed_inputs: Vec::new(),
    };
    let (result, measured) =
        count_cell_model_fits(|| execute_resumed_retarget(&mut fresh, &request));
    let resumed = result.unwrap();

    assert_eq!(measured, 0, "no cell model was fitted");
    assert_eq!(counts(&resumed), [1, 0, 0, 1, 1]);
    let overrides = [
        (Stage::Identification, "recomputed(fresh_process)"),
        (Stage::Law, "recomputed(upstream:target_population<-target_population:modified)"),
        (Stage::Decision, "recomputed(upstream:law<-target_population:modified)"),
        (Stage::TargetPopulation, "recomputed(own:target_population:modified)"),
    ];
    assert_eq!(table(&resumed), expected(&overrides));
    let scores = resumed.receipt.entry(Stage::ScoreArtifact).unwrap();
    assert!(matches!(scores.status, StageStatus::Reused { dependency: Stage::Identification }));
    assert_eq!(scores.counts.total(), 0, "the reused score artifact did no work");

    // Bit for bit the in-process retarget, and equal to an independent rerun within 1e-12.
    assert_eq!(resumed.law.ate.to_bits(), in_process.law.ate.to_bits());
    assert_eq!(resumed.law.std_error.to_bits(), in_process.law.std_error.to_bits());
    assert_eq!(resumed.decision.net_benefit.to_bits(), in_process.decision.net_benefit.to_bits());
    assert_eq!(resumed.decision.treat, in_process.decision.treat);
    assert_eq!(fresh.score_table(), session.score_table().unwrap());
    let weights = &changed.spec.target.as_ref().unwrap().weights;
    let oracle = independent(&base.columns, SEED, Some(weights));
    close(resumed.law.ate, oracle.0, "resumed law vs independent fit");
    close(resumed.law.std_error, oracle.1, "resumed se vs independent fit");
    assert!((resumed.law.ate - first.law.ate).abs() > 0.02);

    // The resumed session now holds its law: a utility change recomputes the decision only.
    let mut utility_only = request.clone();
    utility_only.utility = UtilitySpec { benefit_per_unit: 4.0, cost: 0.2 };
    let (result, measured) =
        count_cell_model_fits(|| execute_resumed_retarget(&mut fresh, &utility_only));
    let decided = result.unwrap();
    assert_eq!(measured, 0);
    assert_eq!(counts(&decided), [0, 0, 0, 0, 1]);
    assert_eq!(decided.law.ate.to_bits(), resumed.law.ate.to_bits());
    close(decided.decision.net_benefit, 4.0 * resumed.law.ate - 0.2, "decision");
}

#[test]
fn c2_cell_resume_with_an_unchanged_request_reproduces_the_first_run() {
    let base = request();
    let mut session = CellSession::new();
    let first = go(&mut session, &base, SEED);
    let artifact = session.export_scores().unwrap();
    let mut fresh = ScoreResumeSession::resume_from_scores(&artifact).unwrap();
    let request = ScoreResumeRequest {
        n_variables: 6,
        edges: base_edges(),
        target: None,
        target_row_ids: None,
        utility: utility(),
        quantity: ScoreQuantity::Interaction,
        changed_inputs: Vec::new(),
    };
    let plan = fresh.plan(&request).unwrap();
    assert!(matches!(plan.status(Stage::ScoreArtifact), Some(StageStatus::Reused { .. })));
    let resumed = execute_resumed_retarget(&mut fresh, &request).unwrap();
    // A fresh process never reuses a derived stage but the portable scores.
    let overrides = [
        (Stage::Identification, "recomputed(fresh_process)"),
        (Stage::Law, "recomputed(fresh_process)"),
        (Stage::Decision, "recomputed(fresh_process)"),
    ];
    assert_eq!(table(&resumed), expected(&overrides));
    assert_eq!(counts(&resumed), [1, 0, 0, 1, 1]);
    assert_eq!(resumed.law.ate.to_bits(), first.law.ate.to_bits());
    assert_eq!(resumed.law.std_error.to_bits(), first.law.std_error.to_bits());
    assert_eq!(resumed.decision.net_benefit.to_bits(), first.decision.net_benefit.to_bits());
}

#[test]
fn c2_cell_resume_refuses_every_request_that_needs_data_or_a_fit() {
    let base = request();
    let mut session = CellSession::new();
    go(&mut session, &base, SEED);
    let artifact = session.export_scores().unwrap();
    let mut fresh = ScoreResumeSession::resume_from_scores(&artifact).unwrap();
    let anchor = fresh.identities().clone();
    let ok = ScoreResumeRequest {
        n_variables: 6,
        edges: base_edges(),
        target: None,
        target_row_ids: None,
        utility: utility(),
        quantity: ScoreQuantity::Interaction,
        changed_inputs: Vec::new(),
    };
    let changed = |stage: Stage, text: &str| {
        let mut r = ok.clone();
        r.changed_inputs = vec![(stage, StageIdentity::of(&stage.label(), &[text.as_bytes()]))];
        r
    };
    let mut regraph = ok.clone();
    regraph.edges.push((W, Y));
    let mut more_variables = ok.clone();
    more_variables.n_variables = 7;

    let refused = [
        ("changed outcome", changed(Stage::Query, "outcome=y2")),
        ("changed folds", changed(Stage::LearnerFoldsRng, "seed+1")),
        ("new data", changed(Stage::DataSnapshot, "other rows")),
        ("new rows", changed(Stage::RowDesign, "complete_case.rows=5000")),
        ("changed grid", changed(Stage::TreatmentGrid, "binary.cells.k=3")),
        ("changed graph", regraph),
        ("changed variable count", more_variables),
    ];
    for (what, request) in refused {
        let (result, measured) =
            count_cell_model_fits(|| execute_resumed_retarget(&mut fresh, &request));
        assert_eq!(measured, 0, "{what}: no fit ran");
        let plan = refused_plan(result.unwrap_err());
        let unavailable = RefusalReason::Unavailable { missing: MissingDependency::Data };
        assert_eq!(plan.first_refusal(), Some((Stage::DataSnapshot, unavailable)), "{what}");
        assert_eq!(
            plan.status(Stage::ScoreArtifact),
            Some(&StageStatus::Refused { reason: unavailable }),
            "{what}: the score artifact is not claimed as reused"
        );
        assert_eq!(unavailable.detail(), "recalc.unavailable_data");
        assert_eq!(unavailable.reason_code(), "score_table_unavailable");
        assert_eq!(fresh.identities(), &anchor, "{what}: the session is unchanged");
        assert!(!fresh.has_run());
        // `plan` reports the same refusal without running anything.
        assert!(matches!(fresh.plan(&request), Err(RecalcRunError::Refused(_))), "{what}");
    }

    // Row ids other than the frozen ones, or weights of another length, are refused up front.
    let weights = TargetWeights {
        weights: vec![1.0; fresh.score_table().n_rows],
        depends_on: vec![VariableId::from_raw(Z)],
    };
    let mut wrong_ids = ok.clone();
    wrong_ids.target = Some(weights.clone());
    let mut ids = fresh.score_table().row_index.to_vec();
    ids.swap(0, 1);
    wrong_ids.target_row_ids = Some(ids);
    let error = execute_resumed_retarget(&mut fresh, &wrong_ids).unwrap_err();
    assert_eq!(request_detail(&error), "recalc.row_ids_mismatch");
    let mut fewer = ok.clone();
    fewer.target = Some(TargetWeights { weights: vec![1.0; 10], depends_on: weights.depends_on });
    let error = execute_resumed_retarget(&mut fresh, &fewer).unwrap_err();
    assert_eq!(request_detail(&error), "recalc.row_count_mismatch");

    // A derived stage cannot be declared as a changed input.
    let derived = changed(Stage::Law, "cheat");
    let error = execute_resumed_retarget(&mut fresh, &derived).unwrap_err();
    assert_eq!(request_detail(&error), "recalc.invalid_changed_input");

    // Weights that depend on the treatment are refused by the estimator's own gate, and the
    // session returns to the artifact.
    let mut illegal = ok.clone();
    illegal.target = Some(TargetWeights {
        weights: vec![1.0; fresh.score_table().n_rows],
        depends_on: vec![VariableId::from_raw(A)],
    });
    assert!(matches!(
        execute_resumed_retarget(&mut fresh, &illegal).unwrap_err(),
        RecalcRunError::Execution(_)
    ));
    assert_eq!(fresh.identities(), &anchor);
    assert!(!fresh.has_run());

    // None of that poisoned the session: the licensed operation still runs.
    let good = execute_resumed_retarget(&mut fresh, &ok).unwrap();
    assert_eq!(counts(&good), [1, 0, 0, 1, 1]);
}

#[test]
fn c2_cell_resume_refuses_resealed_or_foreign_artifact_bytes() {
    let base = request();
    let mut session = CellSession::new();
    go(&mut session, &base, SEED);
    let artifact = session.export_scores().unwrap();
    let identity = artifact.identity().to_owned();
    let bytes = artifact.to_bytes("c2-cell-scores").unwrap();
    assert!(ScoreResumeSession::resume_from_score_bytes(&bytes, Some(&identity)).is_ok());

    // Another run's artifact, consistently sealed, under the identity this consumer retained.
    let mut other = request();
    other.columns = columns(1.0);
    let mut other_session = CellSession::new();
    go(&mut other_session, &other, SEED);
    let foreign = other_session.export_scores().unwrap().to_bytes("c2-cell-scores").unwrap();
    assert!(ScoreResumeSession::resume_from_score_bytes(&foreign, None).is_ok());
    assert!(ScoreResumeSession::resume_from_score_bytes(&foreign, Some(&identity)).is_err());

    // A resealed edit of one score: valid on its own, refused by the retained identity.
    let mut meta = decode_parts(&bytes).unwrap();
    meta.table.scores[0] += 0.25;
    FrozenScoreTable::reseal_identity(&mut meta).unwrap();
    let resealed = encode_parts(&meta, "c2-cell-scores").unwrap();
    assert!(ScoreResumeSession::resume_from_score_bytes(&resealed, Some(&identity)).is_err());
    // An unresealed edit is refused by recomputation alone.
    let mut stale = decode_parts(&bytes).unwrap();
    stale.table.scores[0] += 0.25;
    let stale_bytes = encode_parts(&stale, "c2-cell-scores").unwrap();
    assert!(ScoreResumeSession::resume_from_score_bytes(&stale_bytes, None).is_err());
}

fn crossfit_columns() -> Columns {
    let n = 600;
    let mut rng = ExecutionContext::for_tests(61).rng.stream_for(StreamDomain::Test, 0xC2);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let p = 1.0 / (1.0 + (-(-0.2 + 0.8 * zi)).exp());
        z[i] = zi;
        t[i] = f64::from(rng.next_f64() < p);
        y[i] = (2.0 + 0.8 * zi) * t[i] + zi + 0.3 * standard_normal(&mut rng);
    }
    vec![("t".into(), t), ("y".into(), y), ("z".into(), z)]
}

#[test]
fn c2_cell_crossfit_scores_resume_the_same_way() {
    // Columns t=0, y=1, z=2; edges z->t, z->y, t->y.
    let edges = vec![(2, 0), (2, 1), (0, 1)];
    let base = RecalcRequest {
        columns: crossfit_columns(),
        edges: edges.clone(),
        treatment: 0,
        outcome: 1,
        estimator: AipwAte::new().with_bootstrap_replicates(0),
        target: None,
        utility: utility(),
    };
    let mut session = RecalcSession::new();
    let first = execute_with_receipt(&mut session, &base, &ctx(61)).unwrap();
    assert_eq!(first.receipt.totals().fold_fits, 10);
    let artifact = freeze_crossfit_scores(&session, 600, RetargetSupport::Licensed).unwrap();
    let bytes = artifact.to_bytes("c2-crossfit-scores").unwrap();

    let mut changed = base.clone();
    let z = &base.columns[2].1;
    changed.target = Some(antecedent::analysis::recalc_receipt::TargetWeights {
        weights: z.iter().map(|v| (0.4 * v).exp()).collect(),
        depends_on: vec![VariableId::from_raw(2)],
    });
    let in_process = execute_with_receipt(&mut session, &changed, &ctx(61)).unwrap();
    assert_eq!(in_process.receipt.totals().fold_fits, 0);

    let mut fresh =
        ScoreResumeSession::resume_from_score_bytes(&bytes, Some(artifact.identity())).unwrap();
    let request = ScoreResumeRequest {
        n_variables: 3,
        edges,
        target: changed.target.clone(),
        target_row_ids: Some(fresh.score_table().row_index.to_vec()),
        utility: utility(),
        quantity: ScoreQuantity::AverageEffect,
        changed_inputs: Vec::new(),
    };
    let resumed = execute_resumed_retarget(&mut fresh, &request).unwrap();
    assert_eq!(counts(&resumed), [1, 0, 0, 1, 1]);
    assert!(matches!(
        resumed.receipt.entry(Stage::ScoreArtifact).unwrap().status,
        StageStatus::Reused { .. }
    ));
    close(resumed.law.ate, in_process.law.ate, "resumed cross-fit law vs in-process retarget");
    close(resumed.law.std_error, in_process.law.std_error, "resumed cross-fit se");
    close(
        resumed.decision.net_benefit,
        in_process.decision.net_benefit,
        "resumed cross-fit decision",
    );
    // An unlicensed export is refused.
    let error = freeze_crossfit_scores(&session, 600, RetargetSupport::NotDeclared).unwrap_err();
    assert_eq!(request_detail(&error), "recalc.retarget_not_licensed");
}
