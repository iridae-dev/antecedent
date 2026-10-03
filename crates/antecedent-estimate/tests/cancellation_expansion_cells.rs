//! Cancellation of the 2.2 E-cells that loop (cost and cancellation checks): a context
//! cancelled before the call yields the cell's typed cancellation refusal and no partial
//! result, a context tripped in the middle of the bounded loop is observed at that loop's
//! granularity, and the same inputs complete under a live context.
//!
//! The loops covered here: the penalty grid of the penalized AIPW route (per outer fold and
//! penalty), the whole-cluster folds of the one-way and two-way clustered DML route, the cell,
//! conditional-fit and learner-fold loops of the factorized joint cells (ridge and learner
//! routes), the Newton iterations of the matched case-control solve and the row pass of the
//! descriptive comparison. A tripping token (`cancel_after_checks`) observes cancellation at
//! its `k + 1`-th poll, so each cell is stopped after `k` clean polls.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "test fixtures index small literals and build tables element by element"
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CancellationToken, CausalSchemaBuilder, ExecutionContext,
    MeasurementSpec, RoleHint, SmallRoleSet, StreamDomain, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};
use antecedent_estimate::{
    AdjustedEstimate, AipwAte, AipwWorkspace, ClusterDml, DmlAte, DmlScore, EffectEstimate,
    EstimationError, FactorizedJointConfig, FactorizedJointFit, LearnerSpec, LinearSpec,
    PropensityNuisance, RidgeTuning, compare_raw_adjusted, conditional_odds_ratio,
    fit_factorized_joint_cells, orderings_for, raw_contrast,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::standard_normal;

fn tripping(checks: usize) -> ExecutionContext {
    let mut ctx = ExecutionContext::for_tests(5);
    ctx.cancellation = CancellationToken::cancel_after_checks(checks);
    ctx
}

fn pre_cancelled() -> ExecutionContext {
    let ctx = ExecutionContext::for_tests(5);
    ctx.cancellation.cancel();
    ctx
}

/// The registered code and the message of a coded refusal.
fn refused(error: &EstimationError) -> (&'static str, &str) {
    match error {
        EstimationError::Refused { code, message }
        | EstimationError::RefusedWithFields { code, message, .. } => (*code, message.as_str()),
        other => panic!("expected a coded refusal, got {other:?}"),
    }
}

/// A refusal that is the cell's cancellation stop, never a verdict.
fn assert_cancelled(error: &EstimationError, code: &str, detail: &str) {
    let (got, message) = refused(error);
    assert_eq!(got, code, "{error}");
    assert!(message.contains(detail), "{message}");
}

// ---- penalized and clustered AIPW -----------------------------------------------------------

/// Columns of one synthetic AIPW data set; `cluster` and `second` are row labels.
#[derive(Clone)]
struct Raw {
    t: Vec<f64>,
    y: Vec<f64>,
    z: Vec<Vec<f64>>,
    cluster: Vec<u32>,
    second: Vec<u32>,
}

/// `groups` clusters of `size` rows, `Y = 2 T + z0 + 0.5 z1 + shock_g + noise`.
fn draw(groups: usize, size: usize, seed: u64) -> Raw {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xD1);
    let n = groups * size;
    let mut raw = Raw {
        t: vec![0.0; n],
        y: vec![0.0; n],
        z: vec![vec![0.0; n]; 2],
        cluster: vec![0; n],
        second: Vec::new(),
    };
    for g in 0..groups {
        let shock = 1.5 * standard_normal(&mut rng);
        for k in 0..size {
            let i = g * size + k;
            let z0 = standard_normal(&mut rng);
            let z1 = standard_normal(&mut rng);
            let eta = 0.6 * z0 - 0.4 * z1;
            let t = f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-eta).exp())));
            raw.t[i] = t;
            raw.z[0][i] = z0;
            raw.z[1][i] = z1;
            raw.y[i] = 2.0 * t + z0 + 0.5 * z1 + shock + 0.5 * standard_normal(&mut rng);
            raw.cluster[i] = u32::try_from(g).unwrap();
        }
    }
    raw
}

/// `blocks` components, each the full `na` by `nb` grid of first and second endpoints.
fn draw_two_way(blocks: usize, na: usize, nb: usize, seed: u64) -> Raw {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xD2);
    let mut raw =
        Raw { t: vec![], y: vec![], z: vec![vec![], vec![]], cluster: vec![], second: vec![] };
    for k in 0..blocks {
        let u: Vec<f64> = (0..na).map(|_| 1.2 * standard_normal(&mut rng)).collect();
        let v: Vec<f64> = (0..nb).map(|_| 1.2 * standard_normal(&mut rng)).collect();
        for i in 0..na {
            for j in 0..nb {
                let z0 = 0.5 * u[i] + standard_normal(&mut rng);
                let z1 = standard_normal(&mut rng);
                let eta = 0.6 * z0 - 0.4 * z1;
                let t = f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-eta).exp())));
                raw.t.push(t);
                raw.z[0].push(z0);
                raw.z[1].push(z1);
                raw.y.push(2.0 * t + z0 + 0.5 * z1 + u[i] + v[j] + 0.5 * standard_normal(&mut rng));
                raw.cluster.push(u32::try_from(k * na + i).unwrap());
                raw.second.push(1_000_000 + u32::try_from(k * nb + j).unwrap());
            }
        }
    }
    raw
}

fn build(raw: &Raw) -> (TabularData, IdentifiedEstimand, AverageEffectQuery) {
    let n = raw.t.len();
    let p = raw.z.len();
    let mut builder = CausalSchemaBuilder::new();
    for index in 0..p + 2 {
        let (name, hint) = match index {
            0 => ("t".to_string(), RoleHint::TreatmentCandidate),
            1 => ("y".to_string(), RoleHint::OutcomeCandidate),
            j => (format!("z{}", j - 2), RoleHint::Context),
        };
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let mut values = vec![raw.t.clone(), raw.y.clone()];
    values.extend(raw.z.iter().cloned());
    let columns = values
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(v),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let adjustment: Vec<VariableId> =
        (0..p).map(|j| VariableId::from_raw(u32::try_from(j + 2).unwrap())).collect();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(adjustment),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), estimand, query)
}

fn fit(
    est: &AipwAte,
    raw: &Raw,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, EstimationError> {
    let (data, estimand, query) = build(raw);
    let problem = est.prepare(&data, &estimand, &query)?;
    est.fit(&problem, &mut AipwWorkspace::default(), ctx, AssumptionSet::new())
}

fn penalized(grid: &[f64]) -> AipwAte {
    AipwAte {
        bootstrap_replicates: 0,
        propensity: PropensityNuisance::ridge_logistic(RidgeTuning::new(grid, 3).unwrap()),
        ..AipwAte::new()
    }
}

/// Penalty selection polls once per penalty on every outer fold (5 folds x 2 penalties): a
/// pre-cancelled context stops it with `cancelled_no_claim` and no estimate, a context tripped
/// after the first penalty stops it at the next, and the same call completes when live.
#[test]
fn penalized_aipw_stops_at_a_penalty_without_a_partial_estimate() {
    let raw = draw(40, 6, 3);
    let est = penalized(&[1.0, 10.0]);
    let error = fit(&est, &raw, &pre_cancelled()).unwrap_err();
    assert_cancelled(&error, "cancelled_no_claim", "penalized_propensity.cancelled");
    for checks in [1, 2, 5] {
        let error = fit(&est, &raw, &tripping(checks)).unwrap_err();
        assert_cancelled(&error, "cancelled_no_claim", "penalized_propensity.cancelled");
    }
    let done = fit(&est, &raw, &ExecutionContext::for_tests(5)).unwrap();
    assert!(done.ate.is_finite() && done.score_table.is_some());
}

fn one_way(raw: &Raw) -> AipwAte {
    AipwAte {
        bootstrap_replicates: 0,
        cluster_ids: Some(raw.cluster.clone()),
        cluster_dml: Some(ClusterDml::new(20).unwrap()),
        ..AipwAte::new()
    }
}

fn two_way(raw: &Raw) -> AipwAte {
    AipwAte {
        bootstrap_replicates: 0,
        cluster_ids: Some(raw.cluster.clone()),
        cluster_ids_second: Some(raw.second.clone()),
        cluster_dml: Some(ClusterDml::dyadic(20, 4).unwrap()),
        ..AipwAte::new()
    }
}

/// The whole-cluster fold loop polls once per fold, for the one-way and the two-way unit.
#[test]
fn clustered_dml_stops_at_a_fold_without_a_partial_estimate() {
    let one = draw(40, 6, 4);
    let two = draw_two_way(30, 3, 4, 4);
    for (name, est, raw) in [("one-way", one_way(&one), &one), ("two-way", two_way(&two), &two)] {
        let error = fit(&est, raw, &pre_cancelled()).unwrap_err();
        assert_cancelled(&error, "cancelled_no_claim", "cluster_dml.cancelled");
        for checks in [1, 3] {
            let error = fit(&est, raw, &tripping(checks)).unwrap_err();
            assert_cancelled(&error, "cancelled_no_claim", "cluster_dml.cancelled");
        }
        let done = fit(&est, raw, &ExecutionContext::for_tests(5)).unwrap();
        assert!(done.ate.is_finite() && done.score_table.is_some(), "{name}");
    }
}

// ---- factorized joint cells -----------------------------------------------------------------

struct Sample {
    z: Vec<f64>,
    t: Vec<Vec<f64>>,
    y: Vec<f64>,
}

/// Two binary components from `P(cell | z)` with every cell populated.
fn joint_sample(n: usize, seed: u64) -> Sample {
    let law = [[0.4, 0.2, 0.2, 0.2], [0.1, 0.3, 0.2, 0.4]];
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xD3);
    let mut sample = Sample { z: Vec::new(), t: vec![Vec::new(); 2], y: Vec::new() };
    for _ in 0..n {
        let z = usize::from(rng.next_f64() < 0.5);
        let u = rng.next_f64();
        let mut cumulative = 0.0;
        let mut cell = 3;
        for (c, &p) in law[z].iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                cell = c;
                break;
            }
        }
        sample.z.push(z as f64);
        for (j, column) in sample.t.iter_mut().enumerate() {
            column.push(f64::from(u8::try_from((cell >> j) & 1).unwrap()));
        }
        sample.y.push(0.4 * cell as f64 + 0.5 * z as f64 + 0.3 * standard_normal(&mut rng));
    }
    sample
}

fn joint_frame(sample: &Sample) -> TabularData {
    let columns: Vec<(&str, &[f64])> = vec![
        ("t0", sample.t[0].as_slice()),
        ("t1", sample.t[1].as_slice()),
        ("y", sample.y.as_slice()),
        ("z", sample.z.as_slice()),
    ];
    TabularData::from_f64_columns(columns).unwrap()
}

fn joint_fit(
    data: &TabularData,
    config: &FactorizedJointConfig,
    ctx: &ExecutionContext,
) -> Result<FactorizedJointFit, EstimationError> {
    let id = |name: &str| data.schema().id_of(name).unwrap();
    let orderings = orderings_for(2, &[0, 1], true).unwrap();
    fit_factorized_joint_cells(
        data,
        &[id("t0"), id("t1")],
        id("y"),
        &[id("z")],
        &orderings,
        config,
        ctx,
    )
}

/// The cell loop, every conditional fit, every penalty and every learner fold poll: both
/// routes stop with `cancelled_no_claim` (`joint_cells.cancelled`) when cancelled before the
/// call or after `k` clean polls, with no fit and no score table, and complete when live.
#[test]
fn factorized_joint_cells_stop_in_both_routes_without_a_partial_fit() {
    let data = joint_frame(&joint_sample(600, 9));
    let ridge =
        FactorizedJointConfig { seed: 7, ..FactorizedJointConfig::new(RidgeTuning::default()) };
    let learner = FactorizedJointConfig {
        learner: Some(LearnerSpec::Linear(LinearSpec {})),
        ..FactorizedJointConfig { seed: 7, ..FactorizedJointConfig::new(RidgeTuning::default()) }
    };
    for (name, config) in [("ridge", ridge), ("learner", learner)] {
        let error = joint_fit(&data, &config, &pre_cancelled()).unwrap_err();
        assert_cancelled(&error, "cancelled_no_claim", "joint_cells.cancelled");
        for checks in [1, 3, 8] {
            let error = joint_fit(&data, &config, &tripping(checks)).unwrap_err();
            assert_cancelled(&error, "cancelled_no_claim", "joint_cells.cancelled");
        }
        let live = joint_fit(&data, &config, &ExecutionContext::for_tests(5)).unwrap();
        let spared = joint_fit(&data, &config, &tripping(100_000)).unwrap();
        assert_eq!(live.scores, spared.scores, "{name}: a token that never trips changes nothing");
    }
}

// ---- matched case-control -------------------------------------------------------------------

/// `discordant_cases` pairs whose case alone is exposed and `discordant_controls` pairs whose
/// control alone is exposed; the conditional odds ratio is their ratio.
fn pairs(discordant_cases: usize, discordant_controls: usize) -> (Vec<String>, Vec<f64>, Vec<f64>) {
    let (mut stratum, mut case, mut exposed) = (Vec::new(), Vec::new(), Vec::new());
    for p in 0..discordant_cases + discordant_controls {
        let case_exposed = f64::from(u8::from(p < discordant_cases));
        for (is_case, x) in [(1.0, case_exposed), (0.0, 1.0 - case_exposed)] {
            stratum.push(format!("set{p}"));
            case.push(is_case);
            exposed.push(x);
        }
    }
    (stratum, case, exposed)
}

/// Entry and every Newton iteration poll. With every pair concordant the solve is never
/// reached, so a pre-cancelled context stopping it proves the entry poll; a trip after the
/// entry poll proves the per-iteration one.
#[test]
fn matched_case_control_stops_at_entry_and_at_an_iteration() {
    let (s, c, x) = pairs(7, 3);
    let error = conditional_odds_ratio(&s, &c, &x, &pre_cancelled()).unwrap_err();
    assert_cancelled(&error, "transport_budget_cancel", "matched_case_control.budget");
    let concordant: Vec<f64> = vec![1.0; s.len()];
    let error = conditional_odds_ratio(&s, &c, &concordant, &pre_cancelled()).unwrap_err();
    assert_cancelled(&error, "transport_budget_cancel", "matched_case_control.budget");
    for checks in [1, 2] {
        let error = conditional_odds_ratio(&s, &c, &x, &tripping(checks)).unwrap_err();
        assert_cancelled(&error, "transport_budget_cancel", "matched_case_control.budget");
    }
    let fitted = conditional_odds_ratio(&s, &c, &x, &ExecutionContext::for_tests(5)).unwrap();
    assert!((fitted.odds_ratio - 7.0 / 3.0).abs() < 1e-8, "{}", fitted.odds_ratio);
}

// ---- descriptive comparison -----------------------------------------------------------------

/// The row pass polls at the first row, every 4096 rows and once after the pass.
#[test]
fn descriptive_comparison_stops_in_the_row_pass_without_a_contrast() {
    let n = 10_000;
    let outcome: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    let treatment: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 3 == 0))).collect();
    let error = raw_contrast(&outcome, &treatment, &pre_cancelled()).unwrap_err();
    assert_cancelled(&error, "cancelled_no_claim", "descriptive_comparison.cancelled");
    // Polls fall at rows 0, 4096 and 8192 and once after the pass, so a token that allows one,
    // two or three clean polls stops at row 4096, row 8192 and after the pass respectively.
    for checks in [1, 2, 3] {
        let error = raw_contrast(&outcome, &treatment, &tripping(checks)).unwrap_err();
        assert_cancelled(&error, "cancelled_no_claim", "descriptive_comparison.cancelled");
    }
    let adjusted = AdjustedEstimate::mean_difference(0.1, Some(0.02));
    let error = compare_raw_adjusted(&outcome, &treatment, adjusted, &tripping(1)).unwrap_err();
    assert_cancelled(&error, "cancelled_no_claim", "descriptive_comparison.cancelled");
    let live = ExecutionContext::for_tests(5);
    let raw = raw_contrast(&outcome, &treatment, &live).unwrap();
    assert_eq!(raw.active.n + raw.control.n, n);
    assert!(compare_raw_adjusted(&outcome, &treatment, adjusted, &live).is_ok());
}

// ---- poll counts equal the planned counts (cost checks) --------------------------------------

/// The number of polls a call makes: the smallest budget of clean polls under which it
/// completes (a tripping token only ever stops a call that polls past its budget, so the
/// search is monotone).
fn polls<T, E>(call: impl Fn(&ExecutionContext) -> Result<T, E>) -> usize {
    let mut hi = 1;
    while call(&tripping(hi)).is_err() {
        hi *= 2;
        assert!(hi < 1 << 20, "the call never completes");
    }
    let mut lo = hi / 2;
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if call(&tripping(mid)).is_ok() {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// Penalty selection polls once per penalty on every outer fold, so the polls an executed
/// ridge or lasso fit makes equal the outer folds times the declared grid, and it records one
/// selected penalty per fold: the plan's `folds x penalties x (inner folds + 1)` fits are the
/// declared grid and inner folds, repeated on the folds the run records.
#[test]
fn penalized_selection_polls_equal_the_planned_penalty_candidates() {
    let raw = draw(40, 6, 3);
    let tuning = RidgeTuning::new(&[1.0, 10.0, 100.0], 3).unwrap();
    for (name, propensity) in [
        ("ridge", PropensityNuisance::ridge_logistic(tuning.clone())),
        ("lasso", PropensityNuisance::lasso_with(tuning.clone())),
    ] {
        let est = AipwAte { bootstrap_replicates: 0, propensity, ..AipwAte::new() };
        assert_eq!(est.propensity.planned_fits_per_fold(), 3 * (3 + 1), "{name}");
        let done = fit(&est, &raw, &ExecutionContext::for_tests(5)).unwrap();
        let folds = done.crossfit_folds.expect("a cross-fitted route records its folds");
        assert_eq!(done.learner_provenance.len(), folds, "{name}: one selection per fold");
        let planned_candidates = folds as u64 * tuning.planned_penalties();
        let measured = polls(|ctx| fit(&est, &raw, ctx));
        assert_eq!(measured as u64, planned_candidates, "{name}");
    }
}

/// The whole-cluster fold loop polls once per fold: the polls of a one-way and a two-way fit
/// equal the folds the run records.
#[test]
fn clustered_dml_polls_equal_the_planned_folds() {
    let one = draw(40, 6, 4);
    let two = draw_two_way(30, 3, 4, 4);
    for (est, raw) in [(one_way(&one), &one), (two_way(&two), &two)] {
        let done = fit(&est, raw, &ExecutionContext::for_tests(5)).unwrap();
        let folds = done.crossfit_folds.expect("a cross-fitted route records its folds");
        assert_eq!(polls(|ctx| fit(&est, raw, ctx)), folds);
    }
}

fn ridge_joint_config() -> FactorizedJointConfig {
    FactorizedJointConfig {
        seed: 7,
        ..FactorizedJointConfig::new(RidgeTuning::new(&[1.0, 10.0], 2).unwrap())
    }
}

/// The ridge joint cell fits one conditional per planned stratum (the provenance records one
/// selected penalty each) and polls once per cell, once per conditional and once per penalty of
/// every conditional: exactly the plan's cells, conditional fits and penalty grid.
#[test]
fn joint_cell_ridge_conditionals_and_polls_equal_the_plan() {
    let data = joint_frame(&joint_sample(600, 9));
    let config = ridge_joint_config();
    let orderings = orderings_for(2, &[0, 1], true).unwrap();
    let plan = config.planned_fits(2, &orderings).unwrap();
    assert_eq!((plan.conditional_strata, plan.conditional_fits), (6, 30));
    let done = joint_fit(&data, &config, &ExecutionContext::for_tests(5)).unwrap();
    let provenance: &str = &done.scores.nuisance_provenance;
    let selected =
        provenance.split("selected_lambda=").nth(1).expect("ridge selections").split(',');
    assert_eq!(selected.count() as u64, plan.conditional_fits);
    let penalties = config.tuning.planned_penalties();
    let expected = plan.cells + plan.conditional_fits * (1 + penalties);
    assert_eq!(polls(|ctx| joint_fit(&data, &config, ctx)) as u64, expected);
}

/// The declared-learner joint cell polls once per cell, once per cell and fold of the outcome
/// models and once per conditional: the plan's cells, outcome fits and conditional fits.
#[test]
fn joint_cell_learner_polls_equal_the_plan() {
    let data = joint_frame(&joint_sample(600, 9));
    let config = FactorizedJointConfig {
        learner: Some(LearnerSpec::Linear(LinearSpec {})),
        ..ridge_joint_config()
    };
    let orderings = orderings_for(2, &[0, 1], true).unwrap();
    let plan = config.planned_fits(2, &orderings).unwrap();
    assert_eq!(plan.propensity_fits, plan.conditional_fits);
    let expected = plan.cells + plan.outcome_fits + plan.conditional_fits;
    assert_eq!(polls(|ctx| joint_fit(&data, &config, ctx)) as u64, expected);
}

// ---- DML score route ------------------------------------------------------------------------

/// The cross-fit observes cancellation before every nuisance fit of the AIPW score (once per
/// fold) and before each nuisance of the partially linear score; every stop under a tripping
/// token is the typed `cancelled_no_claim` (`dml.cancelled`) stop and returns no estimate. Each call
/// prepares a fresh problem, so the nuisance cache never answers in place of the fit.
#[test]
fn dml_stops_at_a_fold_and_every_stop_is_the_typed_cancellation() {
    let raw = draw(40, 6, 6);
    let (data, estimand, query) = build(&raw);
    for (score, minimum_polls) in [(DmlScore::Aipw, 4 + 5), (DmlScore::PartiallyLinear, 2)] {
        let est = DmlAte::new().with_score(score);
        let call = |ctx: &ExecutionContext| -> Result<EffectEstimate, EstimationError> {
            let problem = est.prepare(&data, &estimand, &query)?;
            est.fit(&problem, ctx, AssumptionSet::new())
        };
        let stop = |error: &EstimationError| {
            let (code, message) = refused(error);
            code == "cancelled_no_claim" && message.contains("dml.cancelled")
        };
        assert!(stop(&call(&pre_cancelled()).unwrap_err()));
        let made = polls(call);
        assert!(made >= minimum_polls, "{score:?}: {made} polls");
        for checks in 0..made {
            assert!(stop(&call(&tripping(checks)).unwrap_err()), "{score:?} at {checks}");
        }
        assert!(call(&tripping(made)).unwrap().ate.is_finite());
    }
}

/// Plain (unpenalized, unclustered) cross-fitted AIPW polls once per fold: every stop is
/// `cancelled_no_claim` (`aipw.cancelled`) with no estimate, the polls equal the folds the run
/// records, and a token that never trips leaves the estimate bit-identical.
#[test]
fn plain_aipw_stops_at_a_fold_and_polling_changes_no_number() {
    let raw = draw(40, 6, 8);
    let est = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
    let live = fit(&est, &raw, &ExecutionContext::for_tests(5)).unwrap();
    let folds = live.score_table.as_ref().map(|t| t.n_folds as usize).unwrap();
    let error = fit(&est, &raw, &pre_cancelled()).unwrap_err();
    assert_cancelled(&error, "cancelled_no_claim", "aipw.cancelled");
    for checks in 0..folds {
        let error = fit(&est, &raw, &tripping(checks)).unwrap_err();
        assert_cancelled(&error, "cancelled_no_claim", "aipw.cancelled");
    }
    assert_eq!(polls(|ctx| fit(&est, &raw, ctx)), folds);
    let spared = fit(&est, &raw, &tripping(folds)).unwrap();
    assert_eq!(spared.ate.to_bits(), live.ate.to_bits());
    assert_eq!(spared.score_table, live.score_table);
}
