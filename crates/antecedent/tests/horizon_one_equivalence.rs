//! Horizon-one equivalence of temporal estimators with their static twins.
//!
//! A one-decision-time specialization of a temporal estimator is compared with the
//! static route on the same scientific target, the same selected rows, the same
//! design columns, the same functional and the same reading of uncertainty. Every
//! reference value is computed in this file from the raw rows (closed form,
//! normal equations, Newton iteration, or enumerated scores). Routes are never
//! checked by setting one equal to the other alone.
//!
//! Where a temporal route is not equivalent at horizon one the difference is pinned
//! with its cause, so it cannot change silently: the lag-aligned cohort differs
//! from the raw rows, a tempered temporal posterior is a generalized posterior and
//! not an ordinary one, a cross-world temporal effect is a population average with
//! no unit-level payload, a regime value is not licensed where the static
//! randomized contrast is, and a repeated-measures procedure refuses one
//! measurement per unit.
//!
//! The families whose counterparts are not linear regressions live in the sibling modules:
//! `dose` (kernel and local-quadratic dose targets, the average derivative), `design`
//! (instruments, sequential contrasts, history matching, regime survival), `quasi`
//! (discontinuities), `identified` (graph-identified effects, attribution, transport),
//! `meta` (discovery, refutation, decisions, refresh, program search) and `panel` (joint
//! interventions and programs). The inventory test at the end requires every owed family to name
//! a test here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    reason = "integration test builds deterministic fixtures from small counts and solves tiny dense systems by index"
)]

use std::sync::Arc;

use antecedent::analysis::adaptive_procedure_report::{
    RepeatedMeasuresProcedure, RepeatedMeasuresReportBody, RepeatedMeasuresReportRequest,
};
use antecedent::{
    BayesianConfig, CausalQuery, Dag, DenseNodeId, EstimatorId, InferenceMode, PanelDidQuery,
    RandomizedEffectQuery, RefuteSuite, Study, StudyBuilder, StudyResult, TemporalDag,
};
use antecedent_core::{
    AverageEffectQuery, ContinuousDomain, CounterfactualQuery, ExecutionContext, GridSpec,
    IntervalMethod, Intervention, Lag, LongitudinalRegimeMethod, LongitudinalRegimeQuery,
    MediationContrast, MediationQuery, OutcomeMisclassification, RandomizationDesign,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue, TemporalAssignment,
    TemporalDesignFunctional, TemporalDesignQuery, TemporalEffectQuery, TemporalPolicy,
    TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_estimate::{temporal_link, temporal_penalized_score};
use antecedent_graph::ensure_lagged;
use antecedent_stats::adaptive_inference::{
    AnalysisEstimator, FinalAnalysis, OutcomeStructure, RepeatedMeasuresDeclaration, StoppingRule,
    UnitAssignment, UnitSummary, WithinUnitCorrelation,
};

#[path = "horizon_one_equivalence/design.rs"]
mod design;
#[path = "horizon_one_equivalence/dose.rs"]
mod dose;
#[path = "horizon_one_equivalence/identified.rs"]
mod identified;
#[path = "horizon_one_equivalence/meta.rs"]
mod meta;
#[path = "horizon_one_equivalence/panel.rs"]
mod panel;
#[path = "horizon_one_equivalence/quasi.rs"]
mod quasi;

const N: usize = 160;
const TOLERANCE: f64 = 1e-8;
/// `Phi^{-1}(0.975)`.
const NORMAL_Q975: f64 = 1.959_963_984_540_054;
/// Student `t` quantile at 0.975 with 99 degrees of freedom.
const STUDENT_T975_DF99: f64 = 1.984_216_951_586_417_2;

fn id(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= TOLERANCE
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn dag(nodes: u32, edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(nodes);
    for &(from, to) in edges {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    graph
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "callers build the study in the argument and this function drops it after the run"
)]
fn run(study: Study) -> StudyResult {
    study.run(&ctx()).unwrap()
}

fn frequentist(builder: StudyBuilder) -> Study {
    builder
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn bayesian(builder: StudyBuilder) -> Study {
    builder
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(4096).prior_scale(1.0e6),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn has_diagnostic(result: &StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == code)
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for column in 0..n {
        let mut pivot = column;
        for row in column + 1..n {
            if a[row][column].abs() > a[pivot][column].abs() {
                pivot = row;
            }
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        for row in column + 1..n {
            let factor = a[row][column] / a[column][column];
            for k in column..n {
                a[row][k] -= factor * a[column][k];
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut rest = b[row];
        for k in row + 1..n {
            rest -= a[row][k] * x[k];
        }
        x[row] = rest / a[row][row];
    }
    x
}

/// Least-squares coefficients of `y ~ 1 + columns`, intercept first, from the normal equations.
fn ols(columns: &[&[f64]], y: &[f64]) -> Vec<f64> {
    let p = columns.len() + 1;
    let mut gram = vec![vec![0.0; p]; p];
    let mut moment = vec![0.0; p];
    for (row, &outcome) in y.iter().enumerate() {
        let mut x = vec![1.0];
        x.extend(columns.iter().map(|column| column[row]));
        for j in 0..p {
            moment[j] += x[j] * outcome;
            for k in 0..p {
                gram[j][k] += x[j] * x[k];
            }
        }
    }
    solve(gram, moment)
}

/// Treated mean minus control mean. Arms are the 0/1 values in `treatment`.
fn mean_contrast(treatment: &[f64], outcome: &[f64]) -> f64 {
    let mut treated = (0.0, 0.0);
    let mut control = (0.0, 0.0);
    for (&arm, &value) in treatment.iter().zip(outcome) {
        let bucket = if arm > 0.5 { &mut treated } else { &mut control };
        bucket.0 += value;
        bucket.1 += 1.0;
    }
    treated.0 / treated.1 - control.0 / control.1
}

/// Classical homoskedastic SE of a binary mean contrast, `RSS / (n - 2)`.
fn homoskedastic_contrast_se(treatment: &[f64], outcome: &[f64]) -> f64 {
    let mut count = [0.0_f64; 2];
    let mut sum = [0.0_f64; 2];
    for (&arm, &value) in treatment.iter().zip(outcome) {
        let g = usize::from(arm > 0.5);
        count[g] += 1.0;
        sum[g] += value;
    }
    let mean = [sum[0] / count[0], sum[1] / count[1]];
    let rss = treatment
        .iter()
        .zip(outcome)
        .fold(0.0, |acc, (&arm, &value)| acc + (value - mean[usize::from(arm > 0.5)]).powi(2));
    let n = count[0] + count[1];
    (rss / (n - 2.0) * (1.0 / count[0] + 1.0 / count[1])).sqrt()
}

/// A binary treatment series that is not periodic over the sample.
fn treatment_series(n: usize) -> Vec<f64> {
    (0..n).map(|i| if (i * 37 + (i / 4) * 11) % 7 < 3 { 1.0 } else { 0.0 }).collect()
}

/// Deterministic disturbance in `[-0.65, 0.65]`.
fn disturbance(i: usize) -> f64 {
    0.4 * (1.3 * i as f64).sin() + 0.25 * (0.77 * ((i * i) % 97) as f64).cos()
}

/// `y_i = 1 + 2 t_{i-1} + e_i`. Row zero has no lag.
fn pulse_series() -> (Vec<f64>, Vec<f64>) {
    let t = treatment_series(N);
    let mut y = vec![0.0; N];
    y[0] = disturbance(0);
    for i in 1..N {
        y[i] = 1.0 + 2.0 * t[i - 1] + disturbance(i);
    }
    (t, y)
}

fn table(columns: &[(&str, &[f64])]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().copied()).unwrap()
}

fn series(columns: &[(&str, &[f64])]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(columns.iter().copied(), 1).unwrap()
}

/// `t@1 -> y`.
fn pulse_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let treatment_lag = ensure_lagged(&mut graph, id(0), Lag::from_raw(1)).unwrap();
    let outcome_now = ensure_lagged(&mut graph, id(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(treatment_lag, outcome_now).unwrap();
    graph
}

fn pulse_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(id(0), id(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

fn ate_query() -> CausalQuery {
    CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(id(0), id(1)))
}

fn static_pulse_study(treatment: &[f64], outcome: &[f64]) -> StudyBuilder {
    Study::tabular(table(&[("t", treatment), ("y", outcome)]))
        .graph(dag(2, &[(0, 1)]))
        .query(ate_query())
}

fn draws_of(result: &StudyResult) -> Vec<f64> {
    let post = result.posterior.as_ref().unwrap();
    post.draws.column(post.effect_column().unwrap()).unwrap().to_vec()
}

fn mean_sd(draws: &[f64]) -> (f64, f64) {
    let n = draws.len() as f64;
    let mean = draws.iter().sum::<f64>() / n;
    let variance = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, variance.sqrt())
}

fn tempering_kappas(result: &StudyResult) -> Vec<f64> {
    let post = result.posterior.as_ref().unwrap();
    post.diagnostics
        .notes
        .iter()
        .filter_map(|note| {
            antecedent_estimate::tempering_kappa_from_notes(std::slice::from_ref(note))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Linear adjustment (frequentist): selected rows and the sampling interval.
// ---------------------------------------------------------------------------

/// outcome: `not_equivalent` on the raw series, equivalent on the shared cohort.
///
/// A lag-one window cannot form a row for the first time, so the temporal fit
/// selects `N - 1` rows. The static route on the same lag-aligned pairs selects
/// the same rows and reproduces the contrast and the homoskedastic SE. The static
/// route on the unshifted raw rows is another cohort and another target. The
/// temporal interval is a circular-block interval, not the iid analytic interval.
#[test]
fn linear_pulse_warm_up_drops_a_row_and_the_matched_static_cohort_reproduces_it() {
    let (t, y) = pulse_series();
    let aligned_t = &t[..N - 1];
    let aligned_y = &y[1..];
    let contrast = mean_contrast(aligned_t, aligned_y);
    let se = homoskedastic_contrast_se(aligned_t, aligned_y);
    assert!((contrast - 2.0).abs() < 0.5, "the fixture must carry the effect: {contrast}");

    let data = series(&[("t", &t), ("y", &y)]);
    let temporal = run(frequentist(
        Study::series(data.clone())
            .graph(pulse_graph())
            .query(CausalQuery::TemporalEffect(pulse_query())),
    ));
    let matched = run(frequentist(static_pulse_study(aligned_t, aligned_y)));
    let raw = run(frequentist(static_pulse_study(&t, &y)));

    let temporal_effect = temporal.estimate.as_effect().unwrap();
    let matched_effect = matched.estimate.as_effect().unwrap();
    let raw_effect = raw.estimate.as_effect().unwrap();
    // Same functional on the same cohort.
    assert!(close(temporal_effect.ate, contrast), "{}", temporal_effect.ate);
    assert!(close(matched_effect.ate, contrast), "{}", matched_effect.ate);
    assert_eq!(temporal_effect.provenance.n_obs, Some(u64::try_from(N - 1).unwrap()));
    assert_eq!(matched_effect.provenance.n_obs, Some(u64::try_from(N - 1).unwrap()));
    // Same nuisance structure: no adjustment, no learned nuisance on either side.
    assert!(temporal.estimand.adjustment_set.is_empty());
    assert!(matched.estimand.adjustment_set.is_empty());
    assert!(temporal_effect.crossfit.learner_provenance.is_empty());
    assert!(matched_effect.crossfit.learner_provenance.is_empty());
    // Same static twin SE as the closed form; the temporal fit withholds the iid one.
    assert!(close(matched_effect.se_analytic(), se), "{}", matched_effect.se_analytic());
    assert!(temporal_effect.se_analytic().is_nan());

    // The raw rows are another cohort (all N) and another target (contemporaneous).
    assert_eq!(raw_effect.provenance.n_obs, Some(u64::try_from(N).unwrap()));
    assert!((raw_effect.ate - contrast).abs() > 0.5, "raw static effect {}", raw_effect.ate);

    // Uncertainty interpretation: the static iid analytic interval stays an interval of
    // that kind; the temporal route replaces it by a circular-block interval.
    let static_interval = matched.primary_interval_binding(false);
    assert_eq!(static_interval.method, IntervalMethod::AnalyticSe);
    assert_eq!(static_interval.dependence, "iid");
    let blocked = run(Study::series(data)
        .graph(pulse_graph())
        .query(CausalQuery::TemporalEffect(pulse_query()))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(24)
        .build()
        .unwrap());
    let blocked_interval = blocked.primary_interval_binding(false);
    assert_eq!(blocked_interval.method, IntervalMethod::CircularBlockSe);
    assert_ne!(blocked_interval.dependence, "iid");
    let blocked_effect = blocked.estimate.as_effect().unwrap();
    assert!(close(blocked_effect.ate, contrast));
    assert!(blocked_effect.se_bootstrap().is_some_and(|se| se.is_finite() && se > 0.0));
    assert!(blocked_effect.se_analytic().is_nan());
}

// ---------------------------------------------------------------------------
// Linear adjustment (Bayesian): tempered generalized posterior vs ordinary posterior.
// ---------------------------------------------------------------------------

/// outcome: equivalent point functional, `not_equivalent` posterior interpretation.
///
/// Both posteriors are centred on the least-squares contrast of the shared rows.
/// The temporal one tempers the Gaussian likelihood for serial dependence
/// (`kappa >= 1`), so it is a generalized (power) posterior with a recorded
/// correction; the static one is the ordinary conjugate posterior with none. The
/// temporal posterior is never narrower than the ordinary one.
#[test]
fn bayesian_linear_pulse_is_a_tempered_generalized_posterior_not_the_ordinary_one() {
    let (t, y) = pulse_series();
    let aligned_t = &t[..N - 1];
    let aligned_y = &y[1..];
    let contrast = mean_contrast(aligned_t, aligned_y);
    let se = homoskedastic_contrast_se(aligned_t, aligned_y);

    let temporal = bayesian(
        Study::series(series(&[("t", &t), ("y", &y)]))
            .graph(pulse_graph())
            .query(CausalQuery::TemporalEffect(pulse_query())),
    )
    .run(&ExecutionContext::for_tests(7))
    .unwrap();
    let ordinary = bayesian(static_pulse_study(aligned_t, aligned_y))
        .run(&ExecutionContext::for_tests(7))
        .unwrap();

    let (temporal_mean, temporal_sd) = mean_sd(&draws_of(&temporal));
    let (ordinary_mean, ordinary_sd) = mean_sd(&draws_of(&ordinary));
    let kappas = tempering_kappas(&temporal);
    assert_eq!(kappas.len(), 1, "one mechanism, one tempering note");
    let kappa = kappas[0];
    assert!(kappa >= 1.0 - 1e-9, "tempering never sharpens the likelihood: {kappa}");

    // Same point functional: the posterior mean is the least-squares contrast.
    assert!((ordinary_mean - contrast).abs() < 0.25 * se, "{ordinary_mean} vs {contrast}");
    assert!(
        (temporal_mean - contrast).abs() < 0.25 * se * kappa.sqrt(),
        "{temporal_mean} vs {contrast}"
    );
    // Not the same posterior: the temporal one carries the tempering.
    assert!(temporal_sd > 0.9 * ordinary_sd * kappa.sqrt(), "{temporal_sd} {ordinary_sd}");
    assert!(temporal_sd < 1.5 * ordinary_sd * kappa.sqrt(), "{temporal_sd} {ordinary_sd}");

    assert!(has_diagnostic(&temporal, "estimate.bayesian.temporal.dependence_correction"));
    assert!(!has_diagnostic(&ordinary, "estimate.bayesian.temporal.dependence_correction"));
    assert!(tempering_kappas(&ordinary).is_empty());
    // Both are posterior quantile intervals, so the interval kind agrees; the label of
    // the generalized posterior is the tempering record, not a different interval method.
    assert_eq!(temporal.primary_interval_binding(true).method, IntervalMethod::PosteriorQuantile);
    assert_eq!(ordinary.primary_interval_binding(true).method, IntervalMethod::PosteriorQuantile);
}

// ---------------------------------------------------------------------------
// Mediation (Bayesian): product of two tempered mechanisms vs ordinary product.
// ---------------------------------------------------------------------------

/// `m_i = 0.5 t_{i-1} + u_i`, `y_i = 0.3 t_{i-1} + 0.8 m_i + e_i`.
fn mediation_series() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = treatment_series(N);
    let mut m = vec![0.1; N];
    let mut y = vec![0.2; N];
    for i in 1..N {
        let x = i as f64;
        m[i] = 0.5 * t[i - 1] + 0.3 * (0.9 * x).sin();
        y[i] = 0.3 * t[i - 1] + 0.8 * m[i] + 0.2 * (1.7 * x + 0.5).cos();
    }
    (t, m, y)
}

/// outcome: equivalent point functional (path product), `not_equivalent` posterior label.
///
/// The mediated effect is the product of the treatment-to-mediator slope and the
/// mediator-to-outcome slope fitted on the shared rows. Both posteriors are
/// centred on that product (independent mechanisms, flat prior). The temporal one
/// tempers each of its two mechanisms; the static one is untempered.
#[test]
fn bayesian_mediation_path_product_is_tempered_per_mechanism_in_the_temporal_route() {
    let (t, m, y) = mediation_series();
    let (lag_t, now_m, now_y) = (&t[..N - 1], &m[1..], &y[1..]);
    let a = ols(&[lag_t], now_m)[1];
    let b = ols(&[lag_t, now_m], now_y)[2];
    let product = a * b;
    assert!(product.abs() > 0.2, "the mediated effect must be present: {product}");

    let static_query =
        MediationQuery::binary(id(0), id(2), [id(1)], MediationContrast::NaturalIndirect);
    let ordinary = bayesian(
        Study::tabular(table(&[("t", lag_t), ("m", now_m), ("y", now_y)]))
            .graph(dag(3, &[(0, 1), (1, 2), (0, 2)]))
            .query(CausalQuery::Mediation(static_query)),
    )
    .run(&ExecutionContext::for_tests(5))
    .unwrap();

    let mut temporal_graph = TemporalDag::empty();
    let treatment_lag = ensure_lagged(&mut temporal_graph, id(0), Lag::from_raw(1)).unwrap();
    let mediator_now = ensure_lagged(&mut temporal_graph, id(1), Lag::CONTEMPORANEOUS).unwrap();
    let outcome_now = ensure_lagged(&mut temporal_graph, id(2), Lag::CONTEMPORANEOUS).unwrap();
    temporal_graph.insert_directed(treatment_lag, mediator_now).unwrap();
    temporal_graph.insert_directed(treatment_lag, outcome_now).unwrap();
    temporal_graph.insert_directed(mediator_now, outcome_now).unwrap();
    let temporal_query = MediationQuery::binary(id(0), id(2), [id(1)], MediationContrast::Mediated)
        .with_horizons(vec![1])
        .unwrap();
    let temporal = bayesian(
        Study::series(series(&[("t", &t), ("m", &m), ("y", &y)]))
            .graph(temporal_graph)
            .query(CausalQuery::Mediation(temporal_query)),
    )
    .run(&ExecutionContext::for_tests(5))
    .unwrap();

    for (label, result) in [("static", &ordinary), ("temporal", &temporal)] {
        let (mean, sd) = mean_sd(&draws_of(result));
        assert!(
            (mean - product).abs() < 0.2 * sd,
            "{label}: mean {mean} vs a*b {product}, sd {sd}"
        );
    }
    let kappas = tempering_kappas(&temporal);
    assert_eq!(kappas.len(), 2, "one tempering note per mechanism");
    assert!(kappas.iter().all(|kappa| *kappa >= 1.0 - 1e-9), "{kappas:?}");
    assert!(has_diagnostic(&temporal, "estimate.bayesian.temporal.dependence_correction"));
    assert!(tempering_kappas(&ordinary).is_empty());
    assert!(!has_diagnostic(&ordinary, "estimate.bayesian.temporal.dependence_correction"));
}

// ---------------------------------------------------------------------------
// Response curve: horizon-one temporal curve vs the static contrast.
// ---------------------------------------------------------------------------

/// outcome: equivalent target and functional; estimators differ by design.
///
/// The temporal curve is the parametric g-computation of the mean outcome under a
/// pulse at each dose, one step before the outcome. With one lag and no adjustment
/// that mean is `a + b d` for the least-squares fit of the shared rows. The static
/// binary contrast on the same rows is `b`, so the two-point curve difference
/// equals it. The static dose-response curve route is a nonparametric doubly
/// robust estimator and is not asserted equal here.
#[test]
fn horizon_one_response_curve_reproduces_the_least_squares_mean_and_the_static_contrast() {
    let (t, y) = pulse_series();
    let aligned_t = &t[..N - 1];
    let aligned_y = &y[1..];
    let coefficients = ols(&[aligned_t], aligned_y);
    let (intercept, slope) = (coefficients[0], coefficients[1]);

    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: id(1),
        treatment: ContinuousDomain::new(id(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap());
    let temporal = run(frequentist(
        Study::series(series(&[("t", &t), ("y", &y)]))
            .graph(pulse_graph())
            .query(CausalQuery::Response(query)),
    ));
    let static_contrast = run(frequentist(static_pulse_study(aligned_t, aligned_y)));

    let response = temporal.response().expect("temporal response payload");
    assert_eq!(response.provenance_id.as_ref(), "estimate.temporal_response.gcomp");
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected a point-identified temporal response surface");
    };
    assert_eq!(mean.len(), 2);
    assert!(close(mean[0], intercept), "{} vs {intercept}", mean[0]);
    assert!(close(mean[1], intercept + slope), "{} vs {}", mean[1], intercept + slope);
    let static_effect = static_contrast.estimate.as_effect().unwrap().ate;
    assert!(close(static_effect, slope), "{static_effect} vs {slope}");
    assert!(close(mean[1] - mean[0], static_effect));
}

// ---------------------------------------------------------------------------
// Counterfactual: temporal cross-world effect vs static unit-level counterfactual.
// ---------------------------------------------------------------------------

/// outcome: `not_equivalent`.
///
/// The static counterfactual abducts each unit's noise and publishes a unit-level
/// payload (one effect per row, their mean) and, under Bayesian inference, a
/// posterior. The temporal cross-world effect is a population average read off the
/// unfolded linear mechanism: no unit payload; under Frequentist inference a
/// circular-block interval of the refit slope when replicates are requested (none by
/// default); the Bayesian twin publishes the posterior of the slope times the contrast.
/// The mean values agree (both equal the fitted slope) but the objects do not.
#[test]
fn temporal_counterfactual_is_a_population_average_without_unit_effects_or_posterior() {
    let (t, y) = pulse_series();
    let aligned_t = &t[..N - 1];
    let aligned_y = &y[1..];
    let slope = ols(&[aligned_t], aligned_y)[1];
    let query = CounterfactualQuery::new(id(1), [Intervention::set(id(0), Value::f64(1.0))])
        .with_control_level(0.0);

    let static_data = table(&[("t", aligned_t), ("y", aligned_y)]);
    let static_result = Study::tabular(static_data.clone())
        .graph(dag(2, &[(0, 1)]))
        .query(CausalQuery::Counterfactual(query.clone()))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap();
    let unit = static_result.counterfactual().expect("static unit-level payload");
    assert_eq!(unit.unit_effects.len(), N - 1);
    assert!((unit.mean_ite - slope).abs() < 0.1, "{} vs {slope}", unit.mean_ite);

    let context = ctx();
    let static_bayes = Study::tabular(static_data.clone())
        .graph(dag(2, &[(0, 1)]))
        .query(CausalQuery::Counterfactual(query.clone()))
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(64).prior_scale(1000.0),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap()
        .estimate(&static_data, &context)
        .unwrap();
    assert!(static_bayes.posterior.is_some());

    let data = series(&[("t", &t), ("y", &y)]);
    for bayes in [false, true] {
        let mut builder = Study::series(data.clone())
            .graph(pulse_graph())
            .query(CausalQuery::Counterfactual(query.clone()))
            .refute(RefuteSuite::None);
        if bayes {
            builder = builder.inference(InferenceMode::Bayesian(
                BayesianConfig::conjugate().n_draws(8).prior_scale(1000.0),
            ));
        }
        let context = ctx();
        let result = builder
            .build()
            .unwrap()
            .prepare(&context)
            .unwrap()
            .estimate_retained(&context)
            .unwrap();
        // The Bayesian twin publishes the posterior mean of the slope times the contrast
        // from eight draws, so it agrees with the fitted slope to Monte Carlo error only.
        let tolerance = if bayes { 0.2 } else { TOLERANCE };
        let ate = result.estimate.as_effect().unwrap().ate;
        assert!((ate - slope).abs() <= tolerance, "bayes={bayes}: {ate} vs {slope}");
        assert!(has_diagnostic(&result, "estimate.temporal.cross_world"));
        assert!(result.counterfactual().is_none(), "bayes={bayes}: no unit-level payload");
        assert_eq!(result.posterior.is_some(), bayes, "bayes={bayes}: posterior iff Bayesian");
        // A frequentist counterfactual resamples only when replicates are requested, so
        // with the omitted default it is point only and carries no interval binding; the
        // Bayesian twin carries its credible interval.
        let method = result.interval.as_ref().map(|binding| binding.method);
        assert_eq!(
            method,
            if bayes { Some(IntervalMethod::PosteriorQuantile) } else { None },
            "bayes={bayes}"
        );
    }
}

// ---------------------------------------------------------------------------
// Randomized assignment: one-period switchback vs Bernoulli.
// ---------------------------------------------------------------------------

const TRIAL_ROWS: usize = 100;

/// Fifty treated and fifty control rows, outcomes with a 1.4 treatment effect.
fn trial_rows() -> (Vec<bool>, Vec<f64>) {
    let assigned: Vec<bool> = (0..TRIAL_ROWS).map(|i| (i * 13 + 5) % 10 < 5).collect();
    let outcomes = (0..TRIAL_ROWS)
        .map(|i| {
            let x = i as f64;
            1.0 + 0.5 * (0.7 * x).sin()
                + 0.3 * (0.4 * x).cos()
                + if assigned[i] { 1.4 } else { 0.0 }
        })
        .collect();
    (assigned, outcomes)
}

fn labels(prefix: &str, n: usize) -> Vec<Arc<str>> {
    (0..n).map(|i| Arc::<str>::from(format!("{prefix}-{i}"))).collect()
}

/// outcome: equivalent point, variance and interval kind; the critical value differs.
///
/// A switchback with one period per sequence has the same unit-period
/// Horvitz-Thompson contrast and the same sample score sandwich variance as the
/// Bernoulli design. The only differences are the Student `t` critical value on
/// the sequences (vs the normal) and the meaning of the variance field.
#[test]
fn one_period_switchback_reproduces_the_bernoulli_horvitz_thompson_contrast() {
    let (assigned, outcomes) = trial_rows();
    let n = TRIAL_ROWS as f64;
    let scores: Vec<f64> = assigned
        .iter()
        .zip(&outcomes)
        .map(|(&arm, &value)| if arm { value / 0.5 } else { -value / 0.5 })
        .collect();
    let effect = scores.iter().sum::<f64>() / n;
    let sandwich = scores.iter().map(|s| (s - effect).powi(2)).sum::<f64>() / ((n - 1.0) * n);
    let se = sandwich.sqrt();
    let design_variance = outcomes.iter().map(|v| 2.0 * v * v).sum::<f64>() / (n * n);

    let data = table(&[("outcome", &outcomes)]);
    let bernoulli_query = RandomizedEffectQuery::bernoulli_itt(
        id(0),
        assigned.clone(),
        vec![0.5; TRIAL_ROWS],
        labels("unit", TRIAL_ROWS),
        labels("unit", TRIAL_ROWS),
        ("off", "on"),
    );
    let switchback_query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Switchback { periods: labels("period", TRIAL_ROWS).into() },
        id(0),
        assigned,
        vec![0.5; TRIAL_ROWS],
        labels("sequence", TRIAL_ROWS),
        labels("row", TRIAL_ROWS),
        ("off", "on"),
    );
    let bernoulli =
        Study::tabular(data.clone()).query(bernoulli_query).build().unwrap().run(&ctx());
    let switchback = Study::tabular(data).query(switchback_query).build().unwrap().run(&ctx());
    let (bernoulli, switchback) = (bernoulli.unwrap(), switchback.unwrap());
    let b = bernoulli.randomized_effect().unwrap();
    let s = switchback.randomized_effect().unwrap();

    assert_eq!(b.assignment_design.as_ref(), "bernoulli");
    assert_eq!(s.assignment_design.as_ref(), "switchback");
    // Same selected units and arms.
    assert_eq!((b.control_units, b.treatment_units), (50, 50));
    assert_eq!((s.control_units, s.treatment_units), (50, 50));
    // Same functional.
    assert!((b.effect - effect).abs() < 1e-12, "{} vs {effect}", b.effect);
    assert!((s.effect - effect).abs() < 1e-12, "{} vs {effect}", s.effect);
    // Same standard error: the score sandwich.
    assert!((b.standard_error.unwrap() - se).abs() < 1e-12);
    assert!((s.standard_error.unwrap() - se).abs() < 1e-12);
    // Same interval kind (design-based pointwise interval around the HT contrast),
    // different critical value.
    let [b_lower, b_upper] = b.interval_95.expect("Bernoulli score interval");
    let [s_lower, s_upper] = s.interval_95.expect("switchback sequence interval");
    assert!((b_upper - effect - NORMAL_Q975 * se).abs() < 1e-9);
    assert!((effect - b_lower - NORMAL_Q975 * se).abs() < 1e-9);
    assert!((s_upper - effect - STUDENT_T975_DF99 * se).abs() < 1e-9);
    assert!((effect - s_lower - STUDENT_T975_DF99 * se).abs() < 1e-9);
    assert_eq!(b.uncertainty.as_ref(), "bernoulli_ht_score_normal_interval");
    assert_eq!(s.uncertainty.as_ref(), "switchback_independent_sequence_student_interval");
    // The variance field means different things: a Bernoulli design variance bound
    // against the switchback sequence sandwich.
    assert!((b.variance_upper_bound - design_variance).abs() < 1e-12);
    assert!((s.variance_upper_bound - sandwich).abs() < 1e-12);
    assert!((b.variance_upper_bound - s.variance_upper_bound).abs() > 0.01);
    for result in [&bernoulli, &switchback] {
        assert_eq!(result.interval.as_ref().unwrap().method, IntervalMethod::AnalyticSe);
        assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    }
}

// ---------------------------------------------------------------------------
// Regime value: one-period IPW regime vs the static randomized contrast.
// ---------------------------------------------------------------------------

fn ipw_regime(outcomes: &[f64], assigned: &[bool], action: bool) -> LongitudinalRegimeQuery {
    let n = outcomes.len();
    LongitudinalRegimeQuery {
        outcome: id(0),
        method: LongitudinalRegimeMethod::Ipw,
        period_outcome_predictions: None,
        known_fixed_outcome_predictions: false,
        stabilizing_numerator_probabilities: None,
        q_predictions: None,
        observation_history: None,
        prediction_fold_ids: None,
        periods: 1,
        treatment_history: Arc::from(assigned),
        regime_actions: Arc::from(vec![action; n]),
        treatment_probabilities: Arc::from(vec![0.5; n]),
        censoring_probabilities: Arc::from(vec![1.0; n]),
        outcome_observed: Arc::from(vec![true; n]),
        subject_ids: labels("subject", n).into(),
        fold_ids: Arc::from(vec![0; n]),
        excluded_fold_predictions: false,
        probabilities_known_by_design: true,
        minimum_probability: 0.01,
        rule_id: None,
        rule_version: None,
        rule_provenance: None,
    }
}

/// outcome: equivalent point contrast, `not_equivalent` uncertainty and licence.
///
/// With one period the always-treat and never-treat regime values are the two
/// Horvitz-Thompson arm means, and their difference is the static Bernoulli
/// contrast. Each regime value is published alone, with its own score SE. The
/// contrast's SE needs the cross-regime covariance, which two separate regime
/// values do not carry: adding the two variances is wrong by exactly
/// `2 m1 m0 / (n - 1)`. The regime value is also unlicensed, while the static
/// randomized contrast is licensed.
#[test]
fn one_period_regime_values_difference_to_the_static_randomized_contrast() {
    let (assigned, outcomes) = trial_rows();
    let n = TRIAL_ROWS as f64;
    let data = table(&[("outcome", &outcomes)]);

    let arm_scores = |action: bool| -> Vec<f64> {
        assigned
            .iter()
            .zip(&outcomes)
            .map(|(&arm, &value)| if arm == action { value / 0.5 } else { 0.0 })
            .collect()
    };
    let summary = |scores: &[f64]| -> (f64, f64) {
        let mean = scores.iter().sum::<f64>() / n;
        let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n * (n - 1.0));
        (mean, var)
    };
    let (m1, v1) = summary(&arm_scores(true));
    let (m0, v0) = summary(&arm_scores(false));

    let regime = |action: bool| {
        run(Study::tabular(data.clone())
            .query(CausalQuery::LongitudinalRegime(ipw_regime(&outcomes, &assigned, action)))
            .build()
            .unwrap())
    };
    let always = regime(true);
    let never = regime(false);
    let (a, z) = (always.longitudinal_regime().unwrap(), never.longitudinal_regime().unwrap());
    assert!((a.value - m1).abs() < 1e-12 && (z.value - m0).abs() < 1e-12);
    assert!((a.value_standard_error.unwrap() - v1.sqrt()).abs() < 1e-12);
    assert!((z.value_standard_error.unwrap() - v0.sqrt()).abs() < 1e-12);
    assert_eq!(a.value_interval_95, None);
    assert_eq!(a.graphless_support_status, None);

    let contrast = run(Study::tabular(data.clone())
        .query(RandomizedEffectQuery::bernoulli_itt(
            id(0),
            assigned.clone(),
            vec![0.5; TRIAL_ROWS],
            labels("unit", TRIAL_ROWS),
            labels("unit", TRIAL_ROWS),
            ("off", "on"),
        ))
        .build()
        .unwrap());
    let fit = contrast.randomized_effect().unwrap();
    assert!((fit.effect - (a.value - z.value)).abs() < 1e-12);
    assert_eq!(contrast.support_status, Some(antecedent::support::CellStatus::Licensed));

    // The contrast variance includes the covariance of the two arm scores.
    let contrast_variance = fit.standard_error.unwrap().powi(2);
    let covariance_term = 2.0 * m1 * m0 / (n - 1.0);
    assert!((contrast_variance - (v1 + v0 + covariance_term)).abs() < 1e-12);
    assert!(covariance_term.abs() > 1e-3);
    assert!((contrast_variance - (v1 + v0)).abs() > 1e-3, "the naive sum is not the contrast SE");
}

// ---------------------------------------------------------------------------
// Difference-in-differences: panel, repeated cross-section, staggered group-time.
// ---------------------------------------------------------------------------

struct DidRows {
    outcome: Vec<f64>,
    treated: Vec<bool>,
    post: Vec<bool>,
    unit: Vec<usize>,
}

/// Eighty units, forty treated, two periods each, unit-major.
fn did_rows() -> DidRows {
    let mut rows =
        DidRows { outcome: Vec::new(), treated: Vec::new(), post: Vec::new(), unit: Vec::new() };
    for i in 0..80_usize {
        let treated = i % 2 == 0;
        let x = i as f64;
        let level = 5.0 + 0.3 * x.sin() + if treated { 2.0 } else { 0.0 };
        let change = 1.0 + if treated { 1.5 } else { 0.0 } + 0.4 * (1.3 * x).cos();
        for after in [false, true] {
            rows.outcome.push(if after { level + change } else { level });
            rows.treated.push(treated);
            rows.post.push(after);
            rows.unit.push(i);
        }
    }
    rows
}

/// outcome: equivalent for the balanced two-period case; typed refusals mark the rest.
///
/// A two-period balanced panel is the horizon-one case of every temporal
/// difference-in-differences design. The change-score panel route, the four-cell
/// repeated-cross-section route with one cluster per unit, and the staggered
/// group-time route with one cohort all produce the same contrast and the same
/// cluster score sandwich SE. They differ in what they count and publish: the
/// repeated cross-section counts rows where the panel counts subjects, and the
/// group-time route never publishes an interval. The repeated cross-section
/// refuses panel rows (a subject appearing twice).
#[test]
fn two_period_did_designs_agree_on_the_contrast_and_cluster_sandwich() {
    let rows = did_rows();
    let n_rows = rows.outcome.len();
    // Closed form from the unit change scores.
    let mut changes = [Vec::new(), Vec::new()];
    for unit in 0..80 {
        let pre = rows.outcome[2 * unit];
        let post = rows.outcome[2 * unit + 1];
        let group = usize::from(rows.treated[2 * unit]);
        changes[group].push((unit, post - pre));
    }
    let mean = |g: usize| changes[g].iter().map(|c| c.1).sum::<f64>() / changes[g].len() as f64;
    let (m0, m1) = (mean(0), mean(1));
    let effect = m1 - m0;
    let mut score_square = 0.0;
    for g in 0..2 {
        let sign = if g == 1 { 1.0 } else { -1.0 };
        for &(_, change) in &changes[g] {
            let score = sign * (change - mean(g)) / changes[g].len() as f64;
            score_square += score * score;
        }
    }
    let clusters = 80.0_f64;
    let se = (clusters / (clusters - 1.0) * score_square).sqrt();

    let data = table(&[("outcome", &rows.outcome)]);
    let subject_per_unit: Vec<Arc<str>> =
        rows.unit.iter().map(|u| Arc::<str>::from(format!("s{u}"))).collect();
    let subject_per_row: Vec<Arc<str>> =
        (0..n_rows).map(|r| Arc::<str>::from(format!("row{r}"))).collect();
    let cluster_per_unit: Vec<Arc<str>> =
        rows.unit.iter().map(|u| Arc::<str>::from(format!("c{u}"))).collect();
    let run_did = |query: PanelDidQuery| {
        Study::tabular(data.clone()).query(query).build().and_then(|study| study.run(&ctx()))
    };

    let panel = run_did(PanelDidQuery::new(
        id(0),
        rows.treated.clone(),
        rows.post.clone(),
        subject_per_unit.clone(),
        cluster_per_unit.clone(),
    ))
    .unwrap();
    let cross_section = run_did(PanelDidQuery::repeated_cross_section(
        id(0),
        rows.treated.clone(),
        rows.post.clone(),
        subject_per_row,
        cluster_per_unit.clone(),
    ))
    .unwrap();
    let periods: Vec<i64> = rows.post.iter().map(|&after| if after { 2 } else { 1 }).collect();
    let cohorts: Vec<i64> =
        rows.treated.iter().map(|&treated| if treated { 2 } else { 0 }).collect();
    let group_time = run_did(PanelDidQuery::staggered_group_time(
        id(0),
        subject_per_unit.clone(),
        cluster_per_unit.clone(),
        periods,
        cohorts,
        2,
        2,
    ))
    .unwrap();

    let p = panel.panel_did().unwrap();
    let c = cross_section.panel_did().unwrap();
    let g = group_time.panel_did().unwrap();
    for (label, fit) in [("panel", p), ("cross-section", c), ("group-time", g)] {
        assert!((fit.effect - effect).abs() < 1e-10, "{label}: {} vs {effect}", fit.effect);
        assert!((fit.standard_error - se).abs() < 1e-10, "{label}: {} vs {se}", fit.standard_error);
        assert_eq!(fit.clusters, 80, "{label}");
    }
    // What each design counts and publishes.
    assert_eq!((p.treated_subjects, p.comparison_subjects), (40, 40));
    assert_eq!((g.treated_subjects, g.comparison_subjects), (40, 40));
    assert_eq!((c.treated_subjects, c.comparison_subjects), (80, 80), "rows, not subjects");
    let radius = NORMAL_Q975 * se;
    for (label, fit) in [("panel", p), ("cross-section", c)] {
        let [lower, upper] = fit.interval_95.unwrap_or_else(|| panic!("{label} interval"));
        assert!((upper - effect - radius).abs() < 1e-9 && (effect - lower - radius).abs() < 1e-9);
        assert_eq!(fit.uncertainty.as_ref(), "cluster_robust_normal_interval_independent_clusters");
    }
    assert_eq!(g.interval_95, None);
    assert_eq!(g.uncertainty.as_ref(), "cluster_robust_standard_error_no_interval");

    // A repeated cross-section has one row per subject: panel rows are refused.
    let refused = run_did(PanelDidQuery::repeated_cross_section(
        id(0),
        rows.treated.clone(),
        rows.post.clone(),
        subject_per_unit,
        cluster_per_unit,
    ));
    assert!(refused.is_err(), "a subject observed in both periods is not a cross-section");
}

// ---------------------------------------------------------------------------
// Repeated-measures adaptive procedure: one measurement per unit.
// ---------------------------------------------------------------------------

fn repeated_measures(measurements: u32, rho: f64) -> RepeatedMeasuresDeclaration {
    RepeatedMeasuresDeclaration {
        arms: 2,
        assignment: Some(UnitAssignment::Fixed { probabilities: vec![0.5, 0.5] }),
        enrolment: Some(vec![2]),
        lag_looks: 0,
        structure: Some(OutcomeStructure::RepeatedMeasures {
            measurements,
            variance: 2.0,
            correlation: Some(WithinUnitCorrelation::CompoundSymmetric { rho }),
        }),
        analysis: Some(FinalAnalysis {
            estimator: AnalysisEstimator::ClusterRobust { summary: UnitSummary::Mean },
            plus: 1,
            minus: 0,
        }),
        stopping: Some(StoppingRule::NoStopping),
    }
}

/// outcome: `not_equivalent` (typed refusal) at one measurement; limit case at independence.
///
/// One measurement per unit is not a repeated-measures cell: the declaration is
/// refused with the invalid measurement count, and the one-outcome-per-unit procedures
/// own that case. Two uncorrelated measurements recover the independent case: the
/// design effect is 1, so the unit mean has variance `sigma^2 / m` (the closed form),
/// and the two-unit fixed-allocation contrast has variance `2 sigma^2 / m`.
#[test]
fn repeated_measures_refuses_one_measurement_and_reduces_to_independence_at_zero_correlation() {
    let refusals = RepeatedMeasuresProcedure::declare(repeated_measures(1, 0.0))
        .expect_err("a single measurement per unit has no within-unit dependence to declare");
    assert_eq!(refusals[0].code, antecedent_core::reason_code!("invalid_argument"));
    assert_eq!(refusals[0].detail, "adaptive_inference.measurements_invalid");

    let procedure = RepeatedMeasuresProcedure::declare(repeated_measures(2, 0.0))
        .expect("uncorrelated pairs are a valid declaration");
    let report = procedure
        .report(&RepeatedMeasuresReportRequest::ClosedForm { arm_means: vec![0.0, 0.5] }, &ctx())
        .expect("the closed form is derived");
    let RepeatedMeasuresReportBody::ClosedForm {
        marginal_variance,
        unit_summary_variance,
        design_effect,
        exact_law,
        ..
    } = report.body
    else {
        panic!("a closed-form body");
    };
    assert!((marginal_variance - 2.0).abs() < 1e-9);
    assert!((design_effect - 1.0).abs() < 1e-9, "no correlation, no design effect");
    assert!((unit_summary_variance - 2.0 / 2.0).abs() < 1e-9);
    let law = exact_law.expect("a fixed two-arm allocation has a closed form");
    assert!((law.components[0].variance - 2.0 * (2.0 / 2.0)).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Temporal GLM contrast and ridge propensity: independent trajectories only.
// ---------------------------------------------------------------------------

fn fractional(value: f64) -> f64 {
    value - value.floor()
}

fn sigmoid(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

/// Newton iteration for the logistic maximum likelihood, with a ridge penalty on the slopes
/// (every column after the first). `rows` hold the design row with the intercept first.
fn logistic_mode(rows: &[Vec<f64>], y: &[f64], ridge: f64) -> Vec<f64> {
    let p = rows[0].len();
    let mut beta = vec![0.0; p];
    for _ in 0..100 {
        let mut gradient = vec![0.0; p];
        let mut hessian = vec![vec![0.0; p]; p];
        for (x, &outcome) in rows.iter().zip(y) {
            let eta: f64 = x.iter().zip(&beta).map(|(a, b)| a * b).sum();
            let mu = sigmoid(eta);
            let weight = mu * (1.0 - mu);
            for j in 0..p {
                gradient[j] += (outcome - mu) * x[j];
                for k in 0..p {
                    hessian[j][k] += weight * x[j] * x[k];
                }
            }
        }
        for j in 1..p {
            gradient[j] -= ridge * beta[j];
            hessian[j][j] += ridge;
        }
        let step = solve(hessian, gradient);
        for j in 0..p {
            beta[j] += step[j];
        }
    }
    beta
}

/// Average probability contrast of `t` (column 1) and its delta-method SE at `beta`, holding the
/// lag (column 2) at its observed value.
fn glm_contrast_and_se(rows: &[Vec<f64>], beta: &[f64]) -> (f64, f64) {
    let n = rows.len() as f64;
    let p = beta.len();
    let eta = |lag: f64, treatment: f64| beta[0] + beta[1] * treatment + beta[2] * lag;
    let mut contrast = 0.0;
    let mut gradient = vec![0.0; p];
    let mut fisher = vec![vec![0.0; p]; p];
    for x in rows {
        let (mu1, mu0) = (sigmoid(eta(x[2], 1.0)), sigmoid(eta(x[2], 0.0)));
        contrast += (mu1 - mu0) / n;
        let (w1, w0) = (mu1 * (1.0 - mu1), mu0 * (1.0 - mu0));
        let (x1, x0) = ([1.0, 1.0, x[2]], [1.0, 0.0, x[2]]);
        for j in 0..p {
            gradient[j] += (w1 * x1[j] - w0 * x0[j]) / n;
        }
        let observed = sigmoid(beta.iter().zip(x).map(|(b, v)| b * v).sum());
        let weight = observed * (1.0 - observed);
        for j in 0..p {
            for k in 0..p {
                fisher[j][k] += weight * x[j] * x[k];
            }
        }
    }
    let solved = solve(fisher, gradient.clone());
    let variance: f64 = gradient.iter().zip(&solved).map(|(g, s)| g * s).sum();
    (contrast, variance.sqrt())
}

/// outcome: equivalent for independent trajectories; refused for one dependent series.
///
/// One row per subject at one decision time with a lagged covariate held at its
/// observed value is exactly the static adjusted logistic contrast. The temporal
/// link estimator, the static `glm.adjustment` route, and an independent Newton
/// iteration with the delta-method SE agree. A single dependent series has no
/// independent-trajectory information matrix and is refused.
#[test]
fn temporal_link_contrast_on_independent_subjects_reproduces_the_static_glm_route() {
    let n = 300_usize;
    let lag: Vec<f64> = (0..n).map(|i| 1.5 * (0.37 * i as f64).sin()).collect();
    let treatment: Vec<f64> = (0..n)
        .map(|i| {
            let draw = fractional(i as f64 * 0.618_033_988_749_894_9);
            f64::from(draw < sigmoid(-0.3 + 0.8 * lag[i]))
        })
        .collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            let draw = fractional(i as f64 * 0.754_877_666_246_692_7 + 0.31);
            f64::from(draw < sigmoid(-0.5 + 1.2 * treatment[i] + 0.8 * lag[i]))
        })
        .collect();
    let rows: Vec<Vec<f64>> = (0..n).map(|i| vec![1.0, treatment[i], lag[i]]).collect();
    let beta = logistic_mode(&rows, &outcome, 0.0);
    let (contrast, se) = glm_contrast_and_se(&rows, &beta);
    assert!(contrast.abs() > 0.05, "the fixture must carry an effect: {contrast}");

    let design = |dependence: temporal_link::TrajectoryDependence| {
        let mut x = vec![1.0; n];
        x.extend_from_slice(&treatment);
        x.extend_from_slice(&lag);
        temporal_link::TemporalContrastDesign::try_new(
            x,
            outcome.clone(),
            n,
            vec![temporal_link::RegimeColumn {
                column: 1,
                active: vec![1.0; n],
                control: vec![0.0; n],
            }],
            dependence,
        )
        .unwrap()
    };
    let temporal = temporal_link::FrequentistTemporalLink::new()
        .fit(
            &design(temporal_link::TrajectoryDependence::IndependentSubjects),
            temporal_link::TemporalOutcomeLink::BernoulliLogit,
        )
        .unwrap();

    let static_result =
        run(Study::tabular(table(&[("t", &treatment), ("y", &outcome), ("z", &lag)]))
            .graph(dag(3, &[(2, 0), (2, 1), (0, 1)]))
            .query(ate_query())
            .estimator(EstimatorId::GlmAdjustment)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap());
    let static_effect = static_result.estimate.as_effect().unwrap();
    assert_eq!(static_result.logical_plan.estimator.as_deref(), Some("glm.adjustment"));

    assert!((temporal.estimate - contrast).abs() < 1e-6, "{} vs {contrast}", temporal.estimate);
    assert!((static_effect.ate - contrast).abs() < 1e-6, "{} vs {contrast}", static_effect.ate);
    assert!(
        (temporal.standard_error - se).abs() < 1e-5 * se,
        "{} vs {se}",
        temporal.standard_error
    );
    assert!(
        (static_effect.se_analytic() - se).abs() < 1e-5 * se,
        "{} vs {se}",
        static_effect.se_analytic()
    );
    // The link-scale coefficient is not the mean-scale contrast.
    assert!((temporal.regime_coefficients[0] - beta[1]).abs() < 1e-6);
    assert!((temporal.regime_coefficients[0] - temporal.estimate).abs() > 0.1);
    assert_eq!(temporal.trajectories, n);

    let dependent = temporal_link::FrequentistTemporalLink::new().fit(
        &design(temporal_link::TrajectoryDependence::OneDependentSeries),
        temporal_link::TemporalOutcomeLink::BernoulliLogit,
    );
    assert!(dependent.unwrap_err().to_string().contains("dependent series"));
}

/// outcome: equivalent to the stated ridge objective; no equivalent for one dependent series.
///
/// The temporal propensity is the unstandardized ridge-logistic mode of one decision
/// time on independent subjects (intercept free). It matches an independent Newton
/// iteration of that objective and is shrunk relative to the unpenalized fit. It is
/// not the standardized penalty inside static cross-fitted AIPW; a dependent series
/// is refused.
#[test]
fn temporal_ridge_propensity_matches_the_penalized_mode_and_refuses_one_series() {
    let n = 240_usize;
    let lag_one: Vec<f64> = (0..n).map(|i| 2.0 * (0.29 * i as f64).sin()).collect();
    let lag_two: Vec<f64> = (0..n).map(|i| f64::from(fractional(i as f64 * 0.413) < 0.5)).collect();
    let treatment: Vec<f64> = (0..n)
        .map(|i| {
            let draw = fractional(i as f64 * 0.618_033_988_749_894_9 + 0.2);
            f64::from(draw < sigmoid(0.2 + 0.9 * lag_one[i] - 0.7 * lag_two[i]))
        })
        .collect();
    let rows: Vec<Vec<f64>> = (0..n).map(|i| vec![1.0, lag_one[i], lag_two[i]]).collect();
    let penalty = 5.0;
    let ridge = logistic_mode(&rows, &treatment, penalty);
    let unpenalized = logistic_mode(&rows, &treatment, 0.0);
    assert!(ridge[1].abs() < unpenalized[1].abs(), "the penalty must shrink the slope");

    let design = |dependence: temporal_penalized_score::TrajectoryDependence| {
        let mut x = vec![1.0; n];
        x.extend_from_slice(&lag_one);
        x.extend_from_slice(&lag_two);
        temporal_penalized_score::TemporalPropensityDesign::try_new(
            x,
            treatment.clone(),
            n,
            dependence,
        )
        .unwrap()
    };
    let score = temporal_penalized_score::FrequentistTemporalPropensity::new(penalty)
        .unwrap()
        .fit(&design(temporal_penalized_score::TrajectoryDependence::IndependentSubjects))
        .unwrap();
    assert_eq!(score.coefficients.len(), 3);
    for (coefficient, expected) in score.coefficients.iter().zip(&ridge) {
        assert!((coefficient - expected).abs() < 1e-6, "{coefficient} vs {expected}");
    }
    let history = [0.7, 1.0];
    let reference = sigmoid(ridge[0] + ridge[1] * history[0] + ridge[2] * history[1]);
    assert!((score.at(&history).unwrap() - reference).abs() < 1e-6);

    let dependent = temporal_penalized_score::FrequentistTemporalPropensity::new(penalty)
        .unwrap()
        .fit(&design(temporal_penalized_score::TrajectoryDependence::OneDependentSeries));
    assert!(dependent.unwrap_err().to_string().contains("dependent series"));
}

// ---------------------------------------------------------------------------
// Sequentially randomized designs: one decision per unit vs the static twins.
// ---------------------------------------------------------------------------

fn design_query(
    n: usize,
    assignment: TemporalAssignment,
    functional: TemporalDesignFunctional,
) -> TemporalDesignQuery {
    TemporalDesignQuery {
        outcome: id(0),
        units: labels("unit", n).into(),
        periods: vec![0_i64; n].into(),
        lag: 0,
        assignment,
        functional,
        min_probability: 0.05,
        outcome_misclassification: None,
    }
}

fn run_design(
    data: TabularData,
    query: TemporalDesignQuery,
    bootstrap: u32,
) -> antecedent::StudyResult {
    let mut builder = Study::tabular(data).query(CausalQuery::TemporalDesign(query));
    if bootstrap > 0 {
        builder = builder.bootstrap_replicates(bootstrap);
    }
    builder.build().unwrap().run(&ctx()).unwrap()
}

/// outcome: equivalent point, standard error, interval and selected rows.
///
/// A two-arm excursion with one decision per unit, lag zero and a constant recorded
/// probability is the static Bernoulli Horvitz-Thompson contrast: the same arms, the same
/// functional, the same score sandwich and the same normal interval. The temporal route adds
/// the two arm means as coordinates; the static route reports the contrast alone.
#[test]
fn one_decision_two_arm_excursion_reproduces_the_static_bernoulli_contrast() {
    let (assigned, outcomes) = trial_rows();
    let n = TRIAL_ROWS;
    let nf = n as f64;
    let scores: Vec<f64> = assigned
        .iter()
        .zip(&outcomes)
        .map(|(&arm, &value)| if arm { value / 0.5 } else { -value / 0.5 })
        .collect();
    let effect = scores.iter().sum::<f64>() / nf;
    let se = (scores.iter().map(|s| (s - effect).powi(2)).sum::<f64>() / ((nf - 1.0) * nf)).sqrt();

    let data = table(&[("outcome", &outcomes)]);
    let static_query = RandomizedEffectQuery::bernoulli_itt(
        id(0),
        assigned.clone(),
        vec![0.5; n],
        labels("unit", n),
        labels("unit", n),
        ("off", "on"),
    );
    let static_result =
        Study::tabular(data.clone()).query(static_query).build().unwrap().run(&ctx()).unwrap();
    let twin = static_result.randomized_effect().unwrap();

    let temporal = run_design(
        data,
        design_query(
            n,
            TemporalAssignment::TwoArm {
                assignment: assigned.as_slice().into(),
                probability: vec![0.5; n].into(),
            },
            TemporalDesignFunctional::Mean,
        ),
        0,
    );
    let design = temporal.temporal_design().unwrap();
    assert_eq!(design.design.as_ref(), "two_arm");
    // Same selected rows: every unit is one decision, 50 in each arm.
    assert_eq!((design.pairs, design.clusters, design.unpaired), (n as u64, n as u64, 0));
    assert_eq!((twin.control_units, twin.treatment_units), (50, 50));
    // Same functional and the same score sandwich.
    assert!(close(design.estimate[0], effect), "{} vs {effect}", design.estimate[0]);
    assert!(close(twin.effect, effect));
    assert!(close(design.standard_error[0], se), "{} vs {se}", design.standard_error[0]);
    assert!(close(twin.standard_error.unwrap(), se));
    // Same normal pointwise interval.
    let [twin_lower, twin_upper] = twin.interval_95.unwrap();
    assert!(close(design.lower[0], twin_lower) && close(design.upper[0], twin_upper));
    // The arm means are the single-arm Horvitz-Thompson means of the same scores.
    let treated_mean: f64 =
        assigned.iter().zip(&outcomes).filter(|(a, _)| **a).map(|(_, y)| y / 0.5).sum::<f64>() / nf;
    let control_mean: f64 =
        assigned.iter().zip(&outcomes).filter(|(a, _)| !**a).map(|(_, y)| y / 0.5).sum::<f64>()
            / nf;
    assert!(close(design.estimate[1], treated_mean));
    assert!(close(design.estimate[2], control_mean));
    assert!(close(treated_mean - control_mean, effect));
    assert_eq!(design.calibration.as_ref(), "unmeasured");
    assert!(!design.licensed);
}

/// outcome: equivalent arm means and reference contrasts; the Holm family is added.
///
/// Three arms with one decision per unit: each arm mean is the Horvitz-Thompson mean with that
/// arm's recorded probability, and each contrast against the reference arm is the difference
/// of arm means. The categorical declaration reads the same coordinates and adds the Holm
/// family over the contrasts.
#[test]
fn one_decision_multi_arm_and_categorical_contrasts_reproduce_horvitz_thompson_arm_means() {
    let n = 150_usize;
    let nf = n as f64;
    let arm: Vec<usize> = (0..n).map(|i| (i * 7 + 2) % 3).collect();
    let rows: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let wobble = 0.1 * (0.9 * i as f64).sin();
            [0.3 + wobble, 0.4 - wobble, 0.3]
        })
        .collect();
    let outcomes: Vec<f64> =
        (0..n).map(|i| 1.0 + [0.0, 0.8, 1.5][arm[i]] + 0.4 * (0.31 * i as f64).sin()).collect();
    let mean_of = |k: usize| {
        (0..n).filter(|&i| arm[i] == k).map(|i| outcomes[i] / rows[i][k]).sum::<f64>() / nf
    };
    let means = [mean_of(0), mean_of(1), mean_of(2)];
    let data = table(&[("outcome", &outcomes)]);
    let arms: Arc<[Arc<str>]> = Arc::from([Arc::from("a"), Arc::from("b"), Arc::from("c")]);
    for categorical in [false, true] {
        let result = run_design(
            data.clone(),
            design_query(
                n,
                TemporalAssignment::MultiArm {
                    assignment: arm.as_slice().into(),
                    probabilities: rows.iter().map(|row| row.to_vec()).collect::<Vec<_>>().into(),
                    arms: arms.clone(),
                    categorical,
                    ordered: false,
                },
                TemporalDesignFunctional::Mean,
            ),
            0,
        );
        let design = result.temporal_design().unwrap();
        assert_eq!(design.design.as_ref(), if categorical { "categorical" } else { "multi_arm" });
        // Coordinates: contrasts against the reference arm, then the three arm means.
        assert!(close(design.estimate[0], means[1] - means[0]), "{:?}", design.estimate);
        assert!(close(design.estimate[1], means[2] - means[0]));
        for k in 0..3 {
            assert!(close(design.estimate[2 + k], means[k]));
        }
        assert!(design.holm_adjusted_p.is_some());
        assert!(design.standard_error.iter().all(|se| *se > 0.0));
    }
}

/// Weighted left-continuous quantile of `values` at `tau`.
fn weighted_quantile(values: &[f64], weights: &[f64], tau: f64) -> f64 {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let total: f64 = weights.iter().sum();
    let mut cumulative = 0.0;
    for &i in &order {
        cumulative += weights[i];
        if cumulative >= tau * total - 1e-12 * total {
            return values[i];
        }
    }
    values[*order.last().unwrap()]
}

/// outcome: equivalent arm quantiles of the inverse-probability weighted distribution.
///
/// With one decision per unit the pooled outcome law under an arm is the Hajek weighted
/// distribution of the units, so the temporal quantile excursion is the difference of the
/// weighted arm quantiles. The interval is a whole-procedure unit bootstrap, and the route
/// refuses fewer than 99 replicates.
#[test]
fn one_decision_quantile_excursion_reproduces_the_weighted_arm_quantiles() {
    let (assigned, outcomes) = trial_rows();
    let n = TRIAL_ROWS;
    let weights = |arm: bool| -> Vec<f64> {
        assigned.iter().map(|a| if *a == arm { 1.0 / 0.5 } else { 0.0 }).collect()
    };
    let treated = weighted_quantile(&outcomes, &weights(true), 0.5);
    let control = weighted_quantile(&outcomes, &weights(false), 0.5);
    let data = table(&[("outcome", &outcomes)]);
    let query = design_query(
        n,
        TemporalAssignment::TwoArm {
            assignment: assigned.as_slice().into(),
            probability: vec![0.5; n].into(),
        },
        TemporalDesignFunctional::Quantile { level: 0.5 },
    );
    let result = run_design(data.clone(), query.clone(), 99);
    let design = result.temporal_design().unwrap();
    assert!(close(design.estimate[1], treated), "{} vs {treated}", design.estimate[1]);
    assert!(close(design.estimate[2], control));
    assert!(close(design.estimate[0], treated - control));
    assert!(design.replicates.is_some_and(|(done, failed)| done + failed == 99));
    let refused = Study::tabular(data)
        .query(CausalQuery::TemporalDesign(query))
        .build()
        .and_then(|study| study.run(&ctx()));
    assert!(refused.is_err(), "the quantile interval needs replicates");
}

/// outcome: equivalent corrected contrast.
///
/// A non-differentially misclassified binary outcome with sensitivity `se` and specificity `sp`
/// has contrast `HT contrast / (se + sp - 1)`; with one decision per unit the temporal route
/// reproduces that correction exactly.
#[test]
fn one_decision_misclassification_correction_divides_the_contrast_by_the_youden_index() {
    let n = 200_usize;
    let nf = n as f64;
    let assigned: Vec<bool> = (0..n).map(|i| (i * 11 + 3) % 10 < 5).collect();
    let recorded: Vec<f64> = (0..n)
        .map(|i| {
            let rate = if assigned[i] { 0.65 } else { 0.35 };
            f64::from(fractional(i as f64 * 0.754_877_666_246_692_7 + 0.1) < rate)
        })
        .collect();
    let observed: f64 = assigned
        .iter()
        .zip(&recorded)
        .map(|(&arm, &value)| if arm { value / 0.5 } else { -value / 0.5 })
        .sum::<f64>()
        / nf;
    let (true_positive, false_negative, true_negative, false_positive) = (90_u64, 10, 85, 15);
    let youden = 90.0 / 100.0 + 85.0 / 100.0 - 1.0;
    let mut query = design_query(
        n,
        TemporalAssignment::TwoArm {
            assignment: assigned.as_slice().into(),
            probability: vec![0.5; n].into(),
        },
        TemporalDesignFunctional::Mean,
    );
    query.outcome_misclassification = Some(OutcomeMisclassification {
        true_positive,
        false_negative,
        true_negative,
        false_positive,
    });
    let result = run_design(table(&[("outcome", &recorded)]), query, 0);
    let design = result.temporal_design().unwrap();
    assert!(
        close(design.estimate[0], observed / youden),
        "{} vs {}",
        design.estimate[0],
        observed / youden
    );
    assert!(design.functional.contains("misclassification"));
    assert!(design.standard_error[0] > 0.0);
}

/// outcome: equivalent point estimate and selected rows; the interval refits the assignment law.
///
/// The observational excursion with one decision per unit is the Horvitz-Thompson contrast with
/// the fitted assignment probability. The reference fits the logistic maximum likelihood by
/// Newton iteration on the raw rows and evaluates the same ratio; the route must reproduce it,
/// and the same fitted probabilities handed over as a recorded law give the same point
/// estimate, so only the identification and the interval kind differ.
#[test]
fn one_decision_observational_excursion_reproduces_the_fitted_propensity_contrast() {
    let n = 400_usize;
    let nf = n as f64;
    let history: Vec<f64> = (0..n).map(|i| 1.5 * (0.37 * i as f64).sin()).collect();
    let assigned: Vec<bool> = (0..n)
        .map(|i| fractional(i as f64 * 0.618_033_988_749_894_9) < sigmoid(-0.2 + 0.9 * history[i]))
        .collect();
    let outcomes: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 1.5 * f64::from(assigned[i]) + 0.8 * history[i] + 0.3 * (0.53 * i as f64).cos()
        })
        .collect();
    let rows: Vec<Vec<f64>> = (0..n).map(|i| vec![1.0, history[i]]).collect();
    let indicator: Vec<f64> = assigned.iter().map(|a| f64::from(*a)).collect();
    let beta = logistic_mode(&rows, &indicator, 0.0);
    let fitted: Vec<f64> = rows.iter().map(|x| sigmoid(beta[0] + beta[1] * x[1])).collect();
    let scores: Vec<f64> =
        (0..n)
            .map(|i| {
                if assigned[i] {
                    outcomes[i] / fitted[i]
                } else {
                    -outcomes[i] / (1.0 - fitted[i])
                }
            })
            .collect();
    let effect = scores.iter().sum::<f64>() / nf;
    let treated_mean: f64 =
        (0..n).filter(|&i| assigned[i]).map(|i| outcomes[i] / fitted[i]).sum::<f64>() / nf;

    let data = table(&[("outcome", &outcomes), ("treated", &indicator), ("history", &history)]);
    let query = design_query(
        n,
        TemporalAssignment::ObservationalExcursion {
            treatment: id(1),
            history: Arc::from([id(2)]),
        },
        TemporalDesignFunctional::Mean,
    );
    let result = run_design(data.clone(), query, 99);
    let design = result.temporal_design().unwrap();
    assert_eq!(design.design.as_ref(), "observational_excursion");
    assert_eq!((design.pairs, design.clusters), (n as u64, n as u64));
    assert!((design.estimate[0] - effect).abs() < 1e-6, "{} vs {effect}", design.estimate[0]);
    assert!((design.estimate[1] - treated_mean).abs() < 1e-6);
    assert!(design.replicates.is_some_and(|(done, failed)| done + failed == 99));

    // The same contrast with the fitted probabilities treated as a recorded law has the same
    // point estimate; its sandwich interval ignores the estimation of the assignment model.
    let recorded = run_design(
        table(&[("outcome", &outcomes)]),
        design_query(
            n,
            TemporalAssignment::TwoArm {
                assignment: assigned.as_slice().into(),
                probability: fitted.as_slice().into(),
            },
            TemporalDesignFunctional::Mean,
        ),
        0,
    );
    let recorded = recorded.temporal_design().unwrap();
    assert!((recorded.estimate[0] - effect).abs() < 1e-6);
    assert!(design.standard_error[0] > 0.0 && recorded.standard_error[0] > 0.0);

    // The history is declared, not inferred: with no replicates the interval is refused.
    let refused = Study::tabular(data)
        .query(CausalQuery::TemporalDesign(design_query(
            n,
            TemporalAssignment::ObservationalExcursion {
                treatment: id(1),
                history: Arc::from([id(2)]),
            },
            TemporalDesignFunctional::Mean,
        )))
        .build()
        .and_then(|study| study.run(&ctx()));
    assert!(refused.is_err());
}

/// outcome: equivalent cell means, main effects and interaction.
///
/// Two independently assigned binary factors with one decision per unit: each cell mean is the
/// Horvitz-Thompson mean with the product of the two recorded factor probabilities, the main
/// effect of a factor averages its contrast over the other factor's two levels, and the
/// interaction is the difference of differences. Reference values come from the raw rows.
#[test]
fn one_decision_factorial_reproduces_the_cell_means_main_effects_and_interaction() {
    let n = 240_usize;
    let nf = n as f64;
    let first: Vec<bool> =
        (0..n).map(|i| fractional(i as f64 * 0.618_033_988_749_894_9) < 0.5).collect();
    let second: Vec<bool> =
        (0..n).map(|i| fractional(i as f64 * 0.754_877_666_246_692_7 + 0.3) < 0.4).collect();
    let first_probability: Vec<f64> = vec![0.5; n];
    let second_probability: Vec<f64> = vec![0.4; n];
    let outcomes: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 0.7 * f64::from(first[i])
                + 1.1 * f64::from(second[i])
                + 0.5 * f64::from(first[i] && second[i])
                + 0.4 * (0.43 * i as f64).sin()
        })
        .collect();
    let mut mu = [0.0_f64; 4];
    for i in 0..n {
        let cell = usize::from(first[i]) + 2 * usize::from(second[i]);
        let pb = if second[i] { 0.4 } else { 0.6 };
        mu[cell] += outcomes[i] / (0.5 * pb) / nf;
    }
    let main_first = 0.5 * ((mu[1] - mu[0]) + (mu[3] - mu[2]));
    let main_second = 0.5 * ((mu[2] - mu[0]) + (mu[3] - mu[1]));
    let interaction = mu[3] - mu[1] - mu[2] + mu[0];
    let result = run_design(
        table(&[("outcome", &outcomes)]),
        design_query(
            n,
            TemporalAssignment::Factorial {
                first: first.as_slice().into(),
                first_probability: first_probability.as_slice().into(),
                second: second.as_slice().into(),
                second_probability: second_probability.as_slice().into(),
            },
            TemporalDesignFunctional::Mean,
        ),
        0,
    );
    let design = result.temporal_design().unwrap();
    assert_eq!(design.design.as_ref(), "factorial");
    let expected = [main_first, main_second, interaction, mu[0], mu[1], mu[2], mu[3]];
    assert_eq!(design.estimate.len(), expected.len());
    for (actual, wanted) in design.estimate.iter().zip(expected) {
        assert!(close(*actual, wanted), "{actual} vs {wanted}");
    }
    assert!(design.holm_adjusted_p.is_some() && design.band_critical_value.is_some());
}

/// outcome: equivalent exposure means and direct, spillover and total effects.
///
/// Units in clusters of three decide once; a unit's exposure is its own assignment and whether
/// any peer of its cluster was treated, with the peer-exposure probability the product of the
/// peers' recorded probabilities. One decision per unit makes the temporal route the static
/// partial-interference exposure contrast; inference clusters are the interference clusters.
#[test]
fn one_decision_interference_reproduces_the_partial_interference_exposure_contrasts() {
    let clusters = 80_usize;
    let n = clusters * 3;
    let nf = n as f64;
    let probability: Vec<f64> = (0..n).map(|i| 0.35 + 0.3 * fractional(i as f64 * 0.382)).collect();
    let assigned: Vec<bool> = (0..n)
        .map(|i| fractional(i as f64 * 0.618_033_988_749_894_9 + 0.11) < probability[i])
        .collect();
    let outcomes: Vec<f64> = (0..n)
        .map(|i| {
            let peers_treated =
                (0..3).map(|k| (i / 3) * 3 + k).any(|peer| peer != i && assigned[peer]);
            1.0 + 0.9 * f64::from(assigned[i])
                + 0.6 * f64::from(peers_treated)
                + 0.3 * (0.29 * i as f64).cos()
        })
        .collect();
    let mut mu = [0.0_f64; 4];
    for i in 0..n {
        let peers: Vec<usize> = (0..3).map(|k| (i / 3) * 3 + k).filter(|peer| *peer != i).collect();
        let none_treated: f64 = peers.iter().map(|peer| 1.0 - probability[*peer]).product();
        let any_treated = peers.iter().any(|peer| assigned[*peer]);
        let own = if assigned[i] { probability[i] } else { 1.0 - probability[i] };
        let others = if any_treated { 1.0 - none_treated } else { none_treated };
        mu[usize::from(assigned[i]) + 2 * usize::from(any_treated)] +=
            outcomes[i] / (own * others) / nf;
    }
    let cluster_labels: Vec<Arc<str>> =
        (0..n).map(|i| Arc::<str>::from(format!("cluster-{}", i / 3))).collect();
    let periods = vec![0_i64; n];
    let query = TemporalDesignQuery {
        outcome: id(0),
        units: labels("unit", n).into(),
        periods: periods.into(),
        lag: 0,
        assignment: TemporalAssignment::Interference {
            assignment: assigned.as_slice().into(),
            probability: probability.as_slice().into(),
            clusters: cluster_labels.into(),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.01,
        outcome_misclassification: None,
    };
    let result = run_design(table(&[("outcome", &outcomes)]), query, 0);
    let design = result.temporal_design().unwrap();
    assert_eq!(design.design.as_ref(), "interference");
    assert_eq!(design.clusters, clusters as u64);
    let expected = [mu[1] - mu[0], mu[2] - mu[0], mu[3] - mu[0], mu[0], mu[1], mu[2], mu[3]];
    for (actual, wanted) in design.estimate.iter().zip(expected) {
        assert!(close(*actual, wanted), "{actual} vs {wanted}");
    }
    assert!(design.standard_error.iter().all(|se| *se > 0.0));
}

// ---------------------------------------------------------------------------
// Obligations generated from the counterpart inventory.
// ---------------------------------------------------------------------------

const INVENTORY: &str = include_str!("../../../parity/counterparts.toml");
/// Every file of this suite: the tests named by [`ESTABLISHED`] are looked up in all of them.
const SUITE_FILES: [&str; 7] = [
    include_str!("horizon_one_equivalence.rs"),
    include_str!("horizon_one_equivalence/design.rs"),
    include_str!("horizon_one_equivalence/dose.rs"),
    include_str!("horizon_one_equivalence/identified.rs"),
    include_str!("horizon_one_equivalence/meta.rs"),
    include_str!("horizon_one_equivalence/panel.rs"),
    include_str!("horizon_one_equivalence/quasi.rs"),
];

/// One method family of the inventory: its static and temporal frequentist rows.
#[derive(Debug, Default, Clone)]
struct InventoryFamily {
    static_status: Option<String>,
    temporal_status: Option<String>,
    temporal_sampling_design: String,
}

fn quoted(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?.trim_start().strip_prefix('=')?.trim();
    let inner = rest.strip_prefix('"')?;
    Some(inner[..inner.rfind('"')?].to_owned())
}

/// Every method family of `parity/counterparts.toml` with the status of its static and temporal
/// frequentist rows. The rows are read from the generated inventory, so a new method family
/// appears here without anyone editing this file.
fn inventory_families() -> std::collections::BTreeMap<String, InventoryFamily> {
    let mut families = std::collections::BTreeMap::<String, InventoryFamily>::new();
    let flush =
        |row: &std::collections::HashMap<String, String>,
         families: &mut std::collections::BTreeMap<String, InventoryFamily>| {
            if row.get("kind").map(String::as_str) != Some("method")
                || row.get("inference").map(String::as_str) != Some("frequentist")
            {
                return;
            }
            let method = row["method"].clone();
            let family = families.entry(method).or_default();
            match row.get("design").map(String::as_str) {
                Some("static") => family.static_status = row.get("status").cloned(),
                Some("temporal") => {
                    family.temporal_status = row.get("status").cloned();
                    family.temporal_sampling_design =
                        row.get("sampling_design").cloned().unwrap_or_default();
                }
                _ => {}
            }
        };
    let mut row = std::collections::HashMap::<String, String>::new();
    for line in INVENTORY.lines() {
        if line.starts_with("[[row]]") {
            flush(&row, &mut families);
            row.clear();
            continue;
        }
        for key in ["kind", "method", "design", "inference", "status", "sampling_design"] {
            if let Some(value) = quoted(line, key) {
                row.insert(key.to_owned(), value);
            }
        }
    }
    flush(&row, &mut families);
    families
}

/// The test in this file that establishes each family's horizon-one equivalence (or its pinned
/// difference). A family appears here only with the names of tests that exist in this file.
const ESTABLISHED: &[(&str, &[&str])] = &[
    (
        "P01.linear",
        &[
            "linear_pulse_warm_up_drops_a_row_and_the_matched_static_cohort_reproduces_it",
            "bayesian_linear_pulse_is_a_tempered_generalized_posterior_not_the_ordinary_one",
        ],
    ),
    (
        "P02.glm",
        &["temporal_link_contrast_on_independent_subjects_reproduces_the_static_glm_route"],
    ),
    (
        "P02.penalized_propensity",
        &["temporal_ridge_propensity_matches_the_penalized_mode_and_refuses_one_series"],
    ),
    ("P03.policy", &["one_period_regime_values_difference_to_the_static_randomized_contrast"]),
    (
        "P07.curve",
        &["horizon_one_response_curve_reproduces_the_least_squares_mean_and_the_static_contrast"],
    ),
    ("P07.quantile", &["one_decision_quantile_excursion_reproduces_the_weighted_arm_quantiles"]),
    ("P08", &["bayesian_mediation_path_product_is_tempered_per_mechanism_in_the_temporal_route"]),
    (
        "P09.adaptive",
        &[
            "repeated_measures_refuses_one_measurement_and_reduces_to_independence_at_zero_correlation",
        ],
    ),
    (
        "P09.categorical",
        &["one_decision_multi_arm_and_categorical_contrasts_reproduce_horvitz_thompson_arm_means"],
    ),
    (
        "P09.factorial",
        &["one_decision_factorial_reproduces_the_cell_means_main_effects_and_interaction"],
    ),
    (
        "P09.interference",
        &["one_decision_interference_reproduces_the_partial_interference_exposure_contrasts"],
    ),
    (
        "P09.multiarm",
        &["one_decision_multi_arm_and_categorical_contrasts_reproduce_horvitz_thompson_arm_means"],
    ),
    (
        "P09.switchback",
        &["one_period_switchback_reproduces_the_bernoulli_horvitz_thompson_contrast"],
    ),
    ("P09.two_arm", &["one_decision_two_arm_excursion_reproduces_the_static_bernoulli_contrast"]),
    ("P11.did", &["two_period_did_designs_agree_on_the_contrast_and_cluster_sandwich"]),
    (
        "P12.counterfactual",
        &["temporal_counterfactual_is_a_population_average_without_unit_effects_or_posterior"],
    ),
    (
        "P16",
        &["one_decision_misclassification_correction_divides_the_contrast_by_the_youden_index"],
    ),    (
        "P01.iv",
        &["horizon_one_lagged_instrument_reproduces_the_static_two_stage_ratio_with_a_different_uncertainty_product"],
    ),
    (
        "P03.aipw",
        &["a_sequential_contrast_has_no_one_decision_member_and_differs_from_the_static_aipw_of_the_first_action"],
    ),
    (
        "P04",
        &["horizon_one_history_match_is_the_unadjusted_matched_difference_of_the_static_distance_matching"],
    ),
    (
        "P05",
        &["horizon_one_discovery_finds_the_static_adjacency_and_adds_only_the_time_orientation"],
    ),
    (
        "P06",
        &["horizon_one_cpdag_average_effect_has_the_static_estimate_and_envelope_but_not_its_licence"],
    ),
    (
        "P07.continuous_dose",
        &["horizon_one_smoothed_dose_contrast_is_gaussian_horvitz_thompson_while_the_static_policy_value_is_epanechnikov_hajek"],
    ),
    (
        "P07.derivative",
        &["horizon_one_average_derivative_is_the_least_squares_slope_where_the_static_route_is_semiparametric"],
    ),
    (
        "P07.dose_grid",
        &["horizon_one_local_quadratic_dose_grid_reproduces_the_static_randomized_dose_row"],
    ),
    (
        "P07.smoothed_dose",
        &["horizon_one_smoothed_curve_is_gaussian_and_the_static_transport_target_fixes_the_epanechnikov_kernel"],
    ),
    (
        "P10",
        &["horizon_one_regime_survival_is_the_arm_event_share_of_the_static_two_arm_survival"],
    ),
    (
        "P11.local_ratio",
        &["horizon_one_time_discontinuity_local_ratio_matches_the_static_fit_with_block_uncertainty"],
    ),
    (
        "P11.rd",
        &["horizon_one_sharp_rd_uniform_window_fit_is_biased_where_the_temporal_local_ratio_recovers_the_jump"],
    ),
    (
        "P12.intervention",
        &["horizon_one_intervention_response_is_linear_gcomputation_where_the_static_route_is_additive_smooth"],
    ),
    (
        "P13.attribution",
        &["horizon_one_change_attribution_reproduces_the_static_decomposition_with_block_uncertainty"],
    ),
    (
        "P13.refuter",
        &["horizon_one_refutation_gives_identical_reports_in_both_quadrants_and_refuses_the_pre_trend"],
    ),
    (
        "P14",
        &["horizon_one_transport_is_the_static_contrast_with_a_long_run_variance"],
    ),
    (
        "P15.conditional",
        &["horizon_one_conditional_effect_is_the_interaction_regression_marginal_over_the_modifier"],
    ),
    (
        "P15.projection",
        &["horizon_one_admg_average_effect_is_the_static_functional_effect"],
    ),
    (
        "P17",
        &["horizon_one_transported_policy_value_is_the_static_expected_utility_but_needs_a_selection_diagram"],
    ),
    (
        "P18",
        &["horizon_one_period_refresh_is_the_static_least_squares_refresh_with_a_standing_declaration"],
    ),
    (
        "X01",
        &["horizon_one_joint_interaction_is_the_static_decision_world_contrast_but_not_its_licence"],
    ),
    (
        "X04.1",
        &["horizon_one_finite_program_search_ranks_exactly_as_the_static_search"],
    ),
    (
        "X04.2",
        &["horizon_one_continuous_program_search_polls_static_decision_worlds_on_one_reused_fit"],
    ),
];

/// Families that owe an equivalence test and have none yet. It is empty: every executing temporal
/// counterpart with a static twin names a test in [`ESTABLISHED`]. A new inventory family, or a
/// counterpart that starts to execute, appears in the inventory as an owed family with no entry
/// in either list, and the inventory test below fails until its test is written and named.
const NOT_YET_ESTABLISHED: &[&str] = &[];

/// A family owes a horizon-one equivalence obligation when its temporal counterpart executes
/// (or is licensed), it is a statistical method rather than an audit over records, and its
/// static twin is a route and not a theory-based "does not denote".
fn owes_obligation(family: &InventoryFamily) -> bool {
    let executes = |status: &Option<String>| {
        matches!(status.as_deref(), Some("licensed" | "executing_unlicensed"))
    };
    executes(&family.temporal_status)
        && executes(&family.static_status)
        && !family.temporal_sampling_design.starts_with("none:")
}

#[test]
fn every_executing_temporal_counterpart_with_a_static_twin_owes_an_established_obligation() {
    let families = inventory_families();
    assert!(families.len() >= 50, "the inventory lists {} method families", families.len());
    let owed: std::collections::BTreeSet<&str> = families
        .iter()
        .filter(|(_, family)| owes_obligation(family))
        .map(|(method, _)| method.as_str())
        .collect();
    let established: std::collections::BTreeSet<&str> =
        ESTABLISHED.iter().map(|(method, _)| *method).collect();
    let open: std::collections::BTreeSet<&str> = NOT_YET_ESTABLISHED.iter().copied().collect();
    assert_eq!(established.len(), ESTABLISHED.len(), "an established family is listed twice");
    assert!(established.is_disjoint(&open), "a family cannot be both established and open");

    // Every established family names tests that exist in this file.
    for (method, tests) in ESTABLISHED {
        assert!(!tests.is_empty(), "{method} names no test");
        for test in *tests {
            assert!(
                SUITE_FILES.iter().any(|file| file.contains(&format!("#[test]\nfn {test}("))),
                "{method} names {test}, which is not a test of this suite"
            );
        }
    }
    // Every owed obligation is established or explicitly open; nothing is silently missing.
    let unlisted: Vec<&&str> = owed
        .iter()
        .filter(|method| !established.contains(*method) && !open.contains(*method))
        .collect();
    assert!(unlisted.is_empty(), "inventory families with no obligation entry: {unlisted:?}");
    // An open entry that no longer owes anything, or that is established, is stale.
    let stale: Vec<&&str> = open.iter().filter(|method| !owed.contains(*method)).collect();
    assert!(stale.is_empty(), "open entries that no longer owe an obligation: {stale:?}");
    // An established family that is no longer an inventory family is stale, except the switchback,
    // whose static twin is the Bernoulli design (the static switchback does not denote).
    for method in &established {
        assert!(families.contains_key(*method), "{method} is not an inventory family");
    }
    // No obligation is left open.
    assert!(open.is_empty(), "{} open obligations: {open:?}", open.len());
}
