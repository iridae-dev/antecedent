//! The graphless support rows of the sequential contrast, the Bayesian regime and policy
//! values and the within-period history routes, through the study facade.
//!
//! A frequentist section states its row in its own coded diagnostic, recomputed by the
//! artifact check from the stored query and section (the section's `supported` field stays
//! false); a posterior states it in the diagnostic
//! `estimate.design_posterior.graphless_support`, whose fields carry the row key and the
//! observed support. Each test runs one route on a design that meets its row,
//! checks the retained program, the statement and a known value, exports the result, forges the
//! claim (refused by the artifact check), and runs the same route outside the row, where it
//! states no support.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "simulation dimensions are small counts"
)]

#[path = "common/retained_graphless.rs"]
mod retained_graphless;

use std::sync::Arc;

use antecedent::{
    BayesianConfig, CausalError, EstimatorId, InferenceMode, Study, StudyBuilder, StudyResult,
};
use antecedent_core::{
    CausalQuery, ExecutionContext, HistoryMatchQuery, LongitudinalRegimeMethod,
    LongitudinalRegimeQuery, PolicyValueQuery, SequentialContrastQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_io::analysis_result_artifact::AnalysisResultWire;

const ROW_CODE: &str = "estimate.design_posterior.graphless_support";
const DRAWS: usize = 2_000;

/// A fixed-seed generator for the simulated designs.
struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * (1.0 - self.uniform()).ln()).sqrt();
        radius * (std::f64::consts::TAU * self.uniform()).cos()
    }

    /// Mean zero, unit variance, bounded.
    fn bounded(&mut self) -> f64 {
        3.0_f64.sqrt() * (2.0 * self.uniform() - 1.0)
    }
}

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn bayesian(draws: usize) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws))
}

fn has(result: &StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == code)
}

/// The statement of the support row `code` names in an exported body, when it is stated.
fn statement(body: &AnalysisResultWire, code: &str) -> Option<antecedent_io::DiagnosticWire> {
    body.diagnostics.iter().find(|diagnostic| diagnostic.code == code).cloned()
}

/// `body` without the statement of the row `code` names.
fn without_statement(body: &AnalysisResultWire, code: &str) -> AnalysisResultWire {
    let mut unclaimed = body.clone();
    unclaimed.diagnostics.retain(|diagnostic| diagnostic.code != code);
    unclaimed
}

const CONTRAST_ROW: &str = antecedent_io::graphless_support::CROSSFIT_CONTRAST_ROW_CODE;
const HISTORY_ROW: &str = antecedent_io::graphless_support::PERIOD_HISTORY_ROW_CODE;

/// Mean and equal-tailed 95% interval of the posterior's effect column.
fn effect(result: &StudyResult) -> (f64, f64, f64) {
    let posterior = result.posterior.as_ref().expect("a posterior");
    let column = posterior.effect_column().unwrap_or(0);
    let summaries = &posterior.summaries;
    (summaries.mean[column], summaries.q025[column], summaries.q975[column])
}

/// The error an exported body is refused with, or `None` when it is accepted.
fn refusal(wire: &AnalysisResultWire, names: &[String]) -> Option<String> {
    antecedent_io::encode_analysis_result_artifact(wire, names.to_vec(), "forged-support")
        .err()
        .map(|error| error.to_string())
}

/// Run a study through the retained path and export it: the result, the decoded body and
/// the artifact's variable names.
fn run_and_export(
    label: &str,
    builder: StudyBuilder,
    data: &TabularData,
) -> (StudyResult, AnalysisResultWire, Vec<String>) {
    let ctx = ExecutionContext::for_tests(0x7E57);
    let prepared = builder.build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);
    let bytes = prepared.encode_contracted_result(&result, label, &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    (result, body, header.variable_names.clone())
}

/// Run a posterior design that meets its row: the retained program, the stated row, every
/// forged field and another method refused, and a body that states no row accepted.
fn assert_posterior_row(
    label: &str,
    builder: StudyBuilder,
    data: &TabularData,
    method: &str,
    forged: &[(&str, &str)],
) -> (StudyResult, AnalysisResultWire, Vec<String>) {
    let ctx = ExecutionContext::for_tests(0x7E57);
    let prepared = builder.build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);
    let bytes = prepared.encode_contracted_result(&result, label, &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let names = header.variable_names.clone();
    let row = result
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code.as_ref() == ROW_CODE)
        .unwrap_or_else(|| panic!("{label}: the fit states its support row"));
    assert!(
        row.fields
            .iter()
            .any(|(name, value)| name.as_ref() == "method" && value.as_ref() == method),
        "{label}: {:?}",
        row.fields
    );
    let index = body
        .diagnostics
        .iter()
        .position(|diagnostic| diagnostic.code == ROW_CODE)
        .unwrap_or_else(|| panic!("{label}: the statement is exported"));
    assert_eq!(refusal(&body, &names), None, "{label}: the exported statement is admitted");
    for (name, value) in forged.iter().chain(&[("method", "another_method")]) {
        let mut wire = body.clone();
        let slot = wire.diagnostics[index]
            .fields
            .iter_mut()
            .find(|slot| slot.0 == *name)
            .unwrap_or_else(|| panic!("{label}: field {name}"));
        (*value).clone_into(&mut slot.1);
        assert!(refusal(&wire, &names).is_some(), "{label}: forged {name}={value}");
    }
    let mut unclaimed = body.clone();
    unclaimed.diagnostics.remove(index);
    assert_eq!(refusal(&unclaimed, &names), None, "{label}: a body that claims no row");
    (result, body, names)
}

/// Run a posterior design outside its row: the route executes and states no support.
fn assert_no_posterior_row(label: &str, builder: StudyBuilder, data: &TabularData) {
    let (result, body, names) = run_and_export(label, builder, data);
    assert!(!has(&result, ROW_CODE), "{label}: no support row is stated");
    assert_eq!(refusal(&body, &names), None, "{label}: the unsupported result round trips");
}

// ------------------------------------------------------------ sequential contrast

/// `E[Y(1, 1)] - E[Y(0, 0)]` of the panel below.
const CONTRAST_TRUTH: f64 = 1.32 + 1.5;

/// Columns `l0, a0, l1, a1, y`: bounded histories, logistic action probabilities close to one
/// half, `l1 = .5 l0 + .8 a0 + .5 u` and `y = 1 + .5 l0 + a0 + 1.5 a1 + .4 l1 + e`.
fn contrast_panel(n: usize, seed: u64) -> TabularData {
    let mut stream = Stream(seed);
    let (mut l0s, mut a0s, mut l1s, mut a1s, mut ys) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let l0 = stream.bounded();
        let a0 = f64::from(stream.uniform() < expit(0.2 * l0));
        let l1 = 0.5 * l0 + 0.8 * a0 + 0.5 * stream.bounded();
        let a1 = f64::from(stream.uniform() < expit(0.2 * (l1 - 0.4)));
        ys.push(1.0 + 0.5 * l0 + a0 + 1.5 * a1 + 0.4 * l1 + stream.normal());
        l0s.push(l0);
        a0s.push(a0);
        l1s.push(l1);
        a1s.push(a1);
    }
    TabularData::from_f64_columns([
        ("l0", l0s.as_slice()),
        ("a0", a0s.as_slice()),
        ("l1", l1s.as_slice()),
        ("a1", a1s.as_slice()),
        ("y", ys.as_slice()),
    ])
    .unwrap()
}

fn contrast_query(n: usize, folds: usize) -> SequentialContrastQuery {
    SequentialContrastQuery {
        outcome: v(4),
        first_treatment: v(1),
        second_treatment: v(3),
        baseline: Arc::from([v(0)]),
        intermediate: Arc::from([v(2)]),
        active: [1, 1],
        control: [0, 0],
        fold_of_row: (0..n).map(|row| (row % folds) as u32).collect(),
        fold_count: folds as u32,
        min_probability: 0.05,
    }
}

fn contrast_study(
    data: &TabularData,
    n: usize,
    folds: usize,
    draws: Option<usize>,
) -> StudyBuilder {
    let builder = Study::tabular(data.clone())
        .query(CausalQuery::SequentialContrast(contrast_query(n, folds)));
    match draws {
        Some(draws) => builder.inference(bayesian(draws)),
        None => builder,
    }
}

/// The cross-fit Wald interval of the always-treat against never-treat contrast.
#[test]
fn supported_sequential_contrast_wald_interval_round_trips_and_refuses_forgery() {
    let data = contrast_panel(400, 31);
    let (result, body, names) =
        run_and_export("sequential_contrast_wald", contrast_study(&data, 400, 5, None), &data);
    let fit = result.sequential_contrast().unwrap();
    assert!(
        result.states_section_row(),
        "five folds, 400 subjects and probabilities inside the floor"
    );
    assert!(has(&result, CONTRAST_ROW));
    assert!(!has(&result, "estimate.sequential_contrast.unsupported"));
    assert!((fit.contrast - CONTRAST_TRUTH).abs() < 0.8, "{}", fit.contrast);
    let [lower, upper] = fit.interval.unwrap();
    assert!(lower < fit.contrast && fit.contrast < upper);

    // The exported claim is recomputed from the query's folds and the retained scores.
    let claim = statement(&body, CONTRAST_ROW).expect("the exported body states the row");
    assert!(!body.sequential_contrast.as_ref().unwrap().supported, "the flag is not the claim");
    assert_eq!(refusal(&body, &names), None);
    let unclaimed = without_statement(&body, CONTRAST_ROW);
    assert_eq!(refusal(&unclaimed, &names), None, "a result that states no row is accepted");
    let mut low_probability = body.clone();
    low_probability.sequential_contrast.as_mut().unwrap().minimum_action_probability = 0.04;
    assert!(refusal(&low_probability, &names).is_some(), "a probability below the row");
    let mut other_method = body;
    other_method.sequential_contrast.as_mut().unwrap().uncertainty =
        "modular_bayesian_bootstrap_posterior_no_interval".to_owned();
    assert!(refusal(&other_method, &names).is_some(), "another method's label");

    // Two folds, fewer subjects than the row, and another pair of regimes each run and state
    // no support; claiming the row on any of them is refused.
    let small = contrast_panel(190, 32);
    let mut other_regimes = contrast_query(400, 5);
    other_regimes.control = [1, 0];
    let outside = [
        ("two_folds", contrast_study(&data, 400, 2, None), &data),
        ("few_subjects", contrast_study(&small, 190, 5, None), &small),
        (
            "other_regimes",
            Study::tabular(data.clone()).query(CausalQuery::SequentialContrast(other_regimes)),
            &data,
        ),
    ];
    for (label, builder, table) in outside {
        let (result, body, names) = run_and_export(label, builder, table);
        assert!(!result.states_section_row(), "{label}");
        assert!(has(&result, "estimate.sequential_contrast.unsupported"), "{label}");
        assert_eq!(refusal(&body, &names), None, "{label}: the unsupported result round trips");
        let mut stated = body.clone();
        stated.diagnostics.push(claim.clone());
        assert!(
            refusal(&stated, &names).is_some_and(|error| error.contains("states a graphless")),
            "{label}: a forged statement is refused"
        );
        let mut forged = body;
        forged.sequential_contrast.as_mut().unwrap().supported = true;
        assert!(refusal(&forged, &names).is_some(), "{label}: a forged flag is refused");
    }
}

/// The modular Bayesian-bootstrap posterior of the same contrast.
#[test]
fn supported_sequential_contrast_posterior_round_trips_and_refuses_forgery() {
    let data = contrast_panel(400, 31);
    let (result, body, names) = assert_posterior_row(
        "sequential_contrast_posterior",
        contrast_study(&data, 400, 5, Some(DRAWS)),
        &data,
        "crossfit_augmented_scores_bayesian_bootstrap",
        &[
            ("rows", "199"),
            ("rows", "401"),
            ("min_probability", "0.04"),
            ("min_probability", "0.9"),
            ("interval_resamples", "1999"),
        ],
    );
    assert!(!has(&result, "estimate.sequential_contrast.unsupported"));
    assert!(!result.states_section_row(), "the Wald row is not the posterior's");
    let (mean, lower, upper) = effect(&result);
    assert!((mean - CONTRAST_TRUTH).abs() < 0.8, "{mean}");
    assert!(lower < mean && mean < upper);

    // Fewer draws than the row and two folds each run and state no support, and the supported
    // fit's statement moved onto the two-fold result is refused.
    assert_no_posterior_row(
        "sequential_contrast_few_draws",
        contrast_study(&data, 400, 5, Some(1_000)),
        &data,
    );
    let (two_folds, mut forged, _) = run_and_export(
        "sequential_contrast_posterior_two_folds",
        contrast_study(&data, 400, 2, Some(DRAWS)),
        &data,
    );
    assert!(!has(&two_folds, ROW_CODE));
    let statement =
        body.diagnostics.iter().find(|diagnostic| diagnostic.code == ROW_CODE).unwrap().clone();
    forged.diagnostics.push(statement);
    assert!(refusal(&forged, &names).is_some(), "a two-fold result cannot state the row");
}

// ---------------------------------------------------------------- sequential regime

/// `E[Y(1, 1)]` of the trial below.
const REGIME_TRUTH: f64 = 3.5 + 0.32;

/// A two-decision trial with fair-coin sequential randomization and frozen Q predictions that
/// are functions of each subject's own history (the true Q functions times 0.75).
fn regime_trial(n: usize, seed: u64) -> (TabularData, LongitudinalRegimeQuery) {
    let mut stream = Stream(seed);
    let mut outcome = Vec::with_capacity(n);
    let mut treatment = Vec::with_capacity(2 * n);
    let mut q = Vec::with_capacity(2 * n);
    for _ in 0..n {
        let l0 = stream.normal();
        let a0 = stream.uniform() < 0.5;
        let l1 = 0.5 * l0 + 0.8 * f64::from(a0) + 0.5 * stream.normal();
        let a1 = stream.uniform() < 0.5;
        outcome.push(
            1.0 + 0.5 * l0 + f64::from(a0) + 1.5 * f64::from(a1) + 0.4 * l1 + stream.normal(),
        );
        treatment.extend([a0, a1]);
        q.push(0.75 * (3.82 + 0.7 * l0));
        q.push(0.75 * (2.5 + 0.5 * l0 + f64::from(a0) + 0.4 * l1));
    }
    let folds: Vec<u32> = (0..n).map(|i| u32::from(i % 2 == 1)).collect();
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = LongitudinalRegimeQuery {
        outcome: v(0),
        method: LongitudinalRegimeMethod::SequentialDoublyRobust,
        period_outcome_predictions: None,
        known_fixed_outcome_predictions: false,
        stabilizing_numerator_probabilities: None,
        q_predictions: Some(Arc::from(q)),
        observation_history: Some(Arc::from(vec![true; 2 * n])),
        prediction_fold_ids: Some(Arc::from(folds.clone())),
        periods: 2,
        treatment_history: Arc::from(treatment),
        regime_actions: Arc::from(vec![true; 2 * n]),
        treatment_probabilities: Arc::from(vec![0.5; 2 * n]),
        censoring_probabilities: Arc::from(vec![1.0; 2 * n]),
        outcome_observed: Arc::from(vec![true; n]),
        subject_ids: (0..n).map(|i| Arc::<str>::from(format!("subject-{i}"))).collect(),
        fold_ids: Arc::from(folds),
        excluded_fold_predictions: true,
        probabilities_known_by_design: true,
        minimum_probability: 0.1,
        rule_id: None,
        rule_version: None,
        rule_provenance: None,
    };
    (data, query)
}

fn regime_study(data: &TabularData, query: LongitudinalRegimeQuery, draws: usize) -> StudyBuilder {
    Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(query))
        .inference(bayesian(draws))
}

/// The Bayesian-bootstrap posterior of the two-period regime value under a fair coin.
#[test]
fn supported_sequential_regime_posterior_round_trips_and_refuses_forgery() {
    let (data, query) = regime_trial(400, 41);
    let (result, _, _) = assert_posterior_row(
        "sequential_regime_posterior",
        regime_study(&data, query.clone(), DRAWS),
        &data,
        "subject_excluded_q_sequential_dr_bayesian_bootstrap",
        &[("rows", "199"), ("min_probability", "0.4"), ("interval_resamples", "1999")],
    );
    assert!(!has(&result, "estimate.longitudinal.bayesian_sequential_regime_value.unsupported"));
    assert!(result.support_status().is_none(), "the matrix status stays off the matrix");
    let (mean, lower, upper) = effect(&result);
    assert!((mean - REGIME_TRUTH).abs() < 0.8, "{mean}");
    assert!(lower < mean && mean < upper);

    // Fewer draws and fewer subjects each run and state no support.
    assert_no_posterior_row(
        "sequential_regime_few_draws",
        regime_study(&data, query.clone(), 1_000),
        &data,
    );
    let (small, small_query) = regime_trial(150, 42);
    assert_no_posterior_row(
        "sequential_regime_few_subjects",
        regime_study(&small, small_query, DRAWS),
        &small,
    );
    // Predictions owned by other folds than the subjects' never reach the route: the query is
    // refused when the study is compiled, so no result can state the row for them.
    let mut other_owner = query;
    other_owner.prediction_fold_ids =
        Some((0..400).map(|i| u32::from(i % 2 == 0)).collect::<Vec<u32>>().into());
    let ctx = ExecutionContext::for_tests(0x7E57);
    let error = regime_study(&data, other_owner, DRAWS)
        .build()
        .and_then(|study| study.prepare(&ctx).map(|_| ()))
        .expect_err("predictions owned by other folds are refused");
    let CausalError::Rejected(failure) = error.peeled() else {
        panic!("expected a compile refusal, got {error}");
    };
    assert!(failure.message().contains("subject-owned folds"), "{}", failure.message());
}

// ---------------------------------------------------------------------- policy value

/// Incremental value of treating when `x > 0` against treating no one, `.5 + .5 phi(0)`.
const POLICY_TRUTH: f64 = 0.5 + 0.5 * 0.398_942_280_401_432_7;

/// A fair-coin trial with `y = 1 + x + a (1 + .5 x) + e`; the policy treats when `x` exceeds
/// `threshold` and the reference treats no one.
fn policy_trial(n: usize, threshold: f64, seed: u64) -> (TabularData, PolicyValueQuery) {
    let mut stream = Stream(seed);
    let (mut outcome, mut assignment, mut actions) = (vec![], vec![], vec![]);
    for _ in 0..n {
        let x = stream.normal();
        let assigned = stream.uniform() < 0.5;
        outcome.push(1.0 + x + f64::from(assigned) * (1.0 + 0.5 * x) + stream.normal());
        assignment.push(assigned);
        actions.push(x > threshold);
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = PolicyValueQuery {
        outcome: v(0),
        assignment: Arc::from(assignment),
        propensity: Arc::from([0.5]),
        actions: Arc::from(actions),
        reference: Arc::from(vec![false; n]),
        mu0: Arc::from([]),
        mu1: Arc::from([]),
        costs: Arc::from([0.0]),
        reference_costs: Arc::from([0.0]),
        evaluation_subject_ids: (0..n).map(|i| Arc::<str>::from(format!("subject-{i}"))).collect(),
        disjoint_training_subjects: false,
        crossfit_fold_ownership_valid: false,
        global_constraints_present: false,
        multi_action: None,
        uplift_bins: Arc::from([]),
        uplift_bin_count: 0,
        uplift_training_subject_ids: Arc::from([]),
        regret: None,
    };
    (data, query)
}

fn policy_study(data: &TabularData, query: PolicyValueQuery, draws: usize) -> StudyBuilder {
    Study::tabular(data.clone()).query(CausalQuery::PolicyValue(query)).inference(bayesian(draws))
}

/// The Bayesian-bootstrap posterior of a fixed binary policy's incremental value.
#[test]
fn supported_binary_policy_posterior_round_trips_and_refuses_forgery() {
    let (data, query) = policy_trial(400, 0.0, 51);
    let (result, _, _) = assert_posterior_row(
        "binary_policy_posterior",
        policy_study(&data, query.clone(), DRAWS),
        &data,
        "fixed_randomized_ipw_scores_bayesian_bootstrap",
        &[
            ("rows", "199"),
            ("treated", "79"),
            ("control", "79"),
            ("min_probability", "0.4"),
            ("policy_matches", "79"),
            ("reference_matches", "79"),
            ("interval_resamples", "1999"),
        ],
    );
    assert!(!has(&result, "estimate.policy_value.bayesian.unsupported"));
    assert!(result.support_status().is_none(), "the matrix status stays off the matrix");
    let (mean, lower, upper) = effect(&result);
    assert!((mean - POLICY_TRUTH).abs() < 0.9, "{mean}");
    assert!(lower < mean && mean < upper);

    // Fewer draws, fewer rows, and a policy that differs from its reference on few rows each
    // run and state no support.
    assert_no_posterior_row("binary_policy_few_draws", policy_study(&data, query, 1_000), &data);
    let (small, small_query) = policy_trial(150, 0.0, 52);
    assert_no_posterior_row(
        "binary_policy_few_rows",
        policy_study(&small, small_query, DRAWS),
        &small,
    );
    let (rare, rare_query) = policy_trial(400, 1.5, 53);
    assert_no_posterior_row("binary_policy_rare", policy_study(&rare, rare_query, DRAWS), &rare);
}

// ------------------------------------------------------------------ history routes

/// A panel of calendar periods of independent units: columns `h0, h1, y, treated, period`.
/// A unit's score `s ~ N(0, 1)` is the mean of its two history values, treatment is
/// `Bernoulli(expit(.8 s))`, and `y = .5 period + .5 s + treated (1 + 2 s) + e`. Returns the
/// table and the mean effect `1 + 2 s` over the treated units.
fn history_panel(periods: usize, per_period: usize, seed: u64) -> (TabularData, f64) {
    let mut stream = Stream(seed);
    let (mut h0, mut h1, mut y, mut t, mut p) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut effects, mut treated_units) = (0.0, 0.0);
    for index in 0..periods {
        for _ in 0..per_period {
            let s = stream.normal();
            let spread = 0.3 * stream.normal();
            let arm = stream.uniform() < expit(0.8 * s);
            let effect = if arm { 1.0 + 2.0 * s } else { 0.0 };
            h0.push(s + spread);
            h1.push(s - spread);
            y.push(0.5 * index as f64 + 0.5 * s + effect + stream.normal());
            t.push(f64::from(arm));
            p.push(index as f64);
            if arm {
                effects += effect;
                treated_units += 1.0;
            }
        }
    }
    let data = TabularData::from_f64_columns([
        ("h0", h0.as_slice()),
        ("h1", h1.as_slice()),
        ("y", y.as_slice()),
        ("treated", t.as_slice()),
        ("period", p.as_slice()),
    ])
    .unwrap();
    (data, effects / treated_units)
}

fn history_query(caliper: Option<f64>) -> HistoryMatchQuery {
    HistoryMatchQuery {
        outcome: v(2),
        treated: v(3),
        history: Arc::from([v(0), v(1)]),
        period: Some(v(4)),
        weight: None,
        caliper,
    }
}

fn match_study(data: &TabularData, caliper: Option<f64>) -> StudyBuilder {
    Study::tabular(data.clone()).query(CausalQuery::HistoryMatch(history_query(caliper)))
}

/// Population mean effect on the treated of the panel: `1 + 2 E[s | treated]`.
const MATCH_TRUTH: f64 = 1.0 + 2.0 * 0.350_452_056_355;

/// The fixed-b period interval of the within-period history match.
#[test]
fn supported_period_history_interval_round_trips_and_refuses_forgery() {
    let (data, _) = history_panel(8, 100, 61);
    let (result, body, names) =
        run_and_export("period_history_match", match_study(&data, None), &data);
    let fit = result.history_match().unwrap();
    assert!(result.states_section_row(), "eight matched periods of about fifty pairs each");
    assert!(has(&result, HISTORY_ROW));
    assert!(!has(&result, "estimate.history_match.unsupported"));
    assert!((fit.contrast - MATCH_TRUTH).abs() < 0.8, "{}", fit.contrast);
    let (lower, upper) = (fit.interval_lower.unwrap(), fit.interval_upper.unwrap());
    assert!(lower < fit.contrast && fit.contrast < upper);

    // The exported claim is recomputed from the query and the retained pairs and periods.
    let claim = statement(&body, HISTORY_ROW).expect("the exported body states the row");
    assert!(!body.history_match.as_ref().unwrap().supported, "the flag is not the claim");
    assert_eq!(refusal(&body, &names), None);
    let unclaimed = without_statement(&body, HISTORY_ROW);
    assert_eq!(refusal(&unclaimed, &names), None, "a result that states no row is accepted");
    let mut other_method = body;
    other_method.history_match.as_mut().unwrap().uncertainty = "another_method".to_owned();
    assert!(refusal(&other_method, &names).is_some(), "another method's label");

    // Fewer periods, fewer pairs in a period, and a caliper each run and state no support;
    // claiming the row on any of them is refused.
    let (few_periods, _) = history_panel(6, 100, 62);
    let (thin_periods, _) = history_panel(8, 30, 63);
    let outside = [
        ("few_periods", match_study(&few_periods, None), &few_periods),
        ("thin_periods", match_study(&thin_periods, None), &thin_periods),
        ("caliper", match_study(&data, Some(5.0)), &data),
    ];
    for (label, builder, table) in outside {
        let (result, body, names) = run_and_export(label, builder, table);
        assert!(!result.states_section_row(), "{label}");
        assert!(has(&result, "estimate.history_match.unsupported"), "{label}");
        assert_eq!(refusal(&body, &names), None, "{label}: the unsupported result round trips");
        let mut stated = body.clone();
        stated.diagnostics.push(claim.clone());
        assert!(
            refusal(&stated, &names).is_some_and(|error| error.contains("states a graphless")),
            "{label}: a forged statement is refused"
        );
        let mut forged = body;
        forged.history_match.as_mut().unwrap().supported = true;
        assert!(refusal(&forged, &names).is_some(), "{label}: a forged flag is refused");
    }
}

fn outcome_model_study(data: &TabularData, config: BayesianConfig) -> StudyBuilder {
    Study::tabular(data.clone())
        .query(CausalQuery::HistoryMatch(history_query(None)))
        .inference(InferenceMode::Bayesian(config))
        .estimator(EstimatorId::BayesianPeriodHistoryOutcomeModel)
}

/// The conjugate posterior of the within-period history outcome model.
#[test]
fn supported_period_history_outcome_model_round_trips_and_refuses_forgery() {
    let (data, truth) = history_panel(3, 60, 71);
    let (result, _, _) = assert_posterior_row(
        "period_history_outcome_model",
        outcome_model_study(&data, BayesianConfig::conjugate().n_draws(DRAWS)),
        &data,
        "conjugate_gaussian_period_outcome_model_prior_scale_10",
        &[
            ("rows", "134"),
            ("blocks", "2"),
            ("treated", "49"),
            ("control", "49"),
            ("interval_resamples", "1999"),
        ],
    );
    assert!(result.support_status().is_none(), "the matrix status stays off the matrix");
    let (mean, lower, upper) = effect(&result);
    assert!((mean - truth).abs() < 0.8, "{mean} vs {truth}");
    assert!(lower < mean && mean < upper);

    // Fewer draws, another prior scale and fewer units each run and state no support.
    assert_no_posterior_row(
        "period_history_outcome_model_few_draws",
        outcome_model_study(&data, BayesianConfig::conjugate().n_draws(1_000)),
        &data,
    );
    assert_no_posterior_row(
        "period_history_outcome_model_other_prior",
        outcome_model_study(&data, BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(4.0)),
        &data,
    );
    let (small, _) = history_panel(3, 30, 72);
    assert_no_posterior_row(
        "period_history_outcome_model_few_units",
        outcome_model_study(&small, BayesianConfig::conjugate().n_draws(DRAWS)),
        &small,
    );
}
