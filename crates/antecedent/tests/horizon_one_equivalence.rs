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
    MediationContrast, MediationQuery, RandomizationDesign, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, TemporalEffectQuery, TemporalPolicy,
    TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_estimate::{temporal_link, temporal_penalized_score};
use antecedent_graph::ensure_lagged;
use antecedent_stats::adaptive_inference::{
    AnalysisEstimator, FinalAnalysis, OutcomeStructure, RepeatedMeasuresDeclaration, StoppingRule,
    UnitAssignment, UnitSummary, WithinUnitCorrelation,
};

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
/// unfolded linear mechanism: no unit payload and, under Frequentist inference, no
/// interval; the Bayesian twin publishes the posterior of the slope times the contrast.
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
        if !bayes {
            if let Some(binding) = &result.interval {
                assert_eq!(binding.method, IntervalMethod::None);
            }
        }
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
