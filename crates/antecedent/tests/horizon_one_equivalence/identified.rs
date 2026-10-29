//! Graph-identified families at horizon one.
//!
//! Every test compares a temporal graph whose window holds one decision with the static graph on
//! the lag-aligned rows of the same series (or on the same table, for the contemporaneous
//! projection of an ADMG). The references are closed forms from the raw rows. Where a family
//! differs, the cause is pinned: the unlicensed temporal cell, the serial-dependence
//! uncertainty, or the function class of the outcome model.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent::{AcceptedGraph, CellStatus, TransportTrialSpec};
use antecedent_core::{
    AllocationMethod, ChangeAttributionQuery, ConditionalEffectQuery, Intervention,
    PopulationSelector, ShapleyConfig, TransportQuery, Value,
};
use antecedent_estimate::AnalyticSeKind;
use antecedent_graph::{Admg, Cpdag, TemporalAdmg, TemporalCpdag};

// ---------------------------------------------------------------------------
// CPDAG average effect.
// ---------------------------------------------------------------------------

const LAG_ROWS: usize = 600;

/// `z` is exogenous, `x_t = 0.6 z_{t-1} + e_t`, `y_t = 2 x_t + 0.8 z_{t-1} + u_t` (columns `x, y, z`).
fn confounded_lag_zero_series() -> [Vec<f64>; 3] {
    let z: Vec<f64> =
        (0..LAG_ROWS).map(|t| (0.7 * t as f64).sin() + 0.4 * (1.9 * t as f64).cos()).collect();
    let mut x = vec![0.0; LAG_ROWS];
    let mut y = vec![0.0; LAG_ROWS];
    for t in 1..LAG_ROWS {
        let s = t as f64;
        x[t] = 0.6 * z[t - 1] + 0.7 * (2.3 * s + 1.0).sin();
        y[t] = 2.0 * x[t] + 0.8 * z[t - 1] + 0.3 * (3.1 * s).cos();
    }
    [x, y, z]
}

fn lag_zero_cpdag() -> TemporalCpdag {
    let mut graph = TemporalCpdag::empty();
    let z1 = graph.add_lagged(id(2), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(id(0), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(id(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(z1, x0).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(x0, y0).unwrap();
    graph
}

fn static_cpdag() -> Cpdag {
    let mut graph = Cpdag::with_variables(3);
    graph.insert_directed(node(2), node(0)).unwrap();
    graph.insert_directed(node(2), node(1)).unwrap();
    graph.insert_directed(node(0), node(1)).unwrap();
    graph
}

/// Homoskedastic standard error of coefficient `index` (intercept first) of `y ~ 1 + columns`.
fn ols_standard_error(columns: &[&[f64]], y: &[f64], index: usize) -> f64 {
    let p = columns.len() + 1;
    let n = y.len();
    let beta = ols(columns, y);
    let mut gram = vec![vec![0.0; p]; p];
    let mut rss = 0.0;
    for (row, &outcome) in y.iter().enumerate() {
        let mut x = vec![1.0];
        x.extend(columns.iter().map(|column| column[row]));
        rss += (outcome - x.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>()).powi(2);
        for j in 0..p {
            for k in 0..p {
                gram[j][k] += x[j] * x[k];
            }
        }
    }
    let mut unit = vec![0.0; p];
    unit[index] = 1.0;
    let inverse_column = solve(gram, unit);
    (rss / (n - p) as f64 * inverse_column[index]).sqrt()
}

/// outcome: equivalent point and structural envelope; `not_equivalent` licence and uncertainty.
///
/// A temporal CPDAG whose window is the contemporaneous projection of a static CPDAG has the same
/// compiled adjustment: the effect of `x` on `y` adjusting for the lagged cause `z`, the least
/// squares coefficient of `y ~ 1 + x + z_lag` on the aligned rows. The static cell is licensed and
/// publishes the influence-function standard error of that fit (within a percent of the
/// homoskedastic one). The temporal `AverageEffect` is the
/// lag-zero pulse: it executes unlicensed (refused status), discloses the reading, and does not
/// publish the iid standard error for a dependent series.
#[test]
fn horizon_one_cpdag_average_effect_has_the_static_estimate_and_envelope_but_not_its_licence() {
    let [x, y, z] = confounded_lag_zero_series();
    let (aligned_x, aligned_y, aligned_z) = (&x[1..], &y[1..], &z[..LAG_ROWS - 1]);
    let coefficients = ols(&[aligned_x, aligned_z], aligned_y);
    let reference_se = ols_standard_error(&[aligned_x, aligned_z], aligned_y, 1);
    assert!((coefficients[1] - 2.0).abs() < 0.1, "{}", coefficients[1]);

    let query = CausalQuery::AverageEffect(AverageEffectQuery::with_levels(id(0), id(1), 0.0, 1.0));
    let temporal = run(frequentist(
        Study::series(series(&[("x", &x), ("y", &y), ("z", &z)]))
            .graph(AcceptedGraph::from(lag_zero_cpdag()))
            .query(query.clone()),
    ));
    let static_result = run(frequentist(
        Study::tabular(table(&[("x", aligned_x), ("y", aligned_y), ("z", aligned_z)]))
            .graph(AcceptedGraph::from(static_cpdag()))
            .query(query),
    ));

    let temporal_effect = temporal.estimate().as_effect().unwrap();
    let static_effect = static_result.estimate().as_effect().unwrap();
    assert!(close(static_effect.ate, coefficients[1]), "{}", static_effect.ate);
    assert!(close(temporal_effect.ate, coefficients[1]), "{}", temporal_effect.ate);
    // A class envelope publishes the standard error of the mixed influence function, the
    // heteroskedasticity-consistent form of the same fit: it matches the homoskedastic
    // reference to the finite-sample factor, not to rounding.
    assert!(
        (static_effect.se_analytic() / reference_se - 1.0).abs() < 0.01,
        "{} vs {reference_se}",
        static_effect.se_analytic()
    );
    // The same single completion on both sides.
    if let (Some(temporal_mix), Some(static_mix)) =
        (temporal.structural_response.as_ref(), static_result.structural_response.as_ref())
    {
        assert_eq!(temporal_mix.atoms.len(), static_mix.atoms.len());
        assert!(close(temporal_mix.identified_mass, static_mix.identified_mass));
        assert!(close(temporal_mix.unidentified_mass, static_mix.unidentified_mass));
    }
    // What differs: the licence and the uncertainty reading.
    assert_eq!(static_result.support_status(), Some(CellStatus::Licensed));
    assert_eq!(temporal.support_status(), Some(CellStatus::Refused));
    assert!(has_diagnostic(&temporal, "estimate.temporal_class.lag_zero_average_effect"));
    assert!(!has_diagnostic(&static_result, "estimate.temporal_class.lag_zero_average_effect"));
}

// ---------------------------------------------------------------------------
// Intervention response (g-computation).
// ---------------------------------------------------------------------------

fn response_scalar(result: &StudyResult) -> f64 {
    match &result.response().expect("a response payload").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            assert_eq!(mean.len(), 1);
            mean[0]
        }
        other => panic!("expected a point-identified level, got {other:?}"),
    }
}

fn intervention_response(level: f64) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: id(1),
        interventions: Arc::from([Intervention::set(id(0), Value::f64(level))]),
    })
}

/// outcome: equivalent target and linear closed form; the outcome model class is
/// `not_equivalent` under curvature.
///
/// The temporal intervention response is the parametric g-computation of the lag-aligned linear
/// regression, `a + b d` at the intervened dose. The static response is the additive-smooth
/// g-computation of the same rows. On a linear mean they agree within the smoother's error; on
/// a convex mean the temporal route returns the least-squares projection (its closed form from
/// the rows) and misses the curvature that the static smooth recovers.
#[test]
fn horizon_one_intervention_response_is_linear_gcomputation_where_the_static_route_is_additive_smooth()
 {
    let dose = 1.5;
    for convex in [false, true] {
        let (x, mut y) = super::dose::derivative_series();
        if convex {
            for t in 1..N {
                y[t] = 1.0 + 0.5 * x[t - 1] * x[t - 1] + 0.05 * disturbance(t);
            }
        }
        let (aligned_x, aligned_y) = (&x[..N - 1], &y[1..]);
        let coefficients = ols(&[aligned_x], aligned_y);
        let projection = coefficients[0] + coefficients[1] * dose;
        let truth = if convex { 1.0 + 0.5 * dose * dose } else { 3.0 + 2.0 * dose };

        let temporal = run(frequentist(
            Study::series(series(&[("x", &x), ("y", &y)])).graph(pulse_graph()).query(
                CausalQuery::Response(intervention_response(dose).with_temporal(
                    TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap(),
                )),
            ),
        ));
        let static_result = run(frequentist(
            Study::tabular(table(&[("x", aligned_x), ("y", aligned_y)]))
                .graph(dag(2, &[(0, 1)]))
                .query(CausalQuery::Response(intervention_response(dose))),
        ));
        let temporal_level = response_scalar(&temporal);
        let static_level = response_scalar(&static_result);
        let label = format!("convex={convex}");
        assert!(
            (temporal_level - projection).abs() < 1e-8,
            "{label}: {temporal_level} vs {projection}"
        );
        assert!((static_level - truth).abs() < 0.06, "{label}: {static_level} vs {truth}");
        if convex {
            assert!((temporal_level - truth).abs() > 0.08, "{label}: curvature must show");
            assert!((temporal_level - static_level).abs() > 0.08, "{label}");
        } else {
            assert!((temporal_level - static_level).abs() < 0.06, "{label}");
        }
    }
}

// ---------------------------------------------------------------------------
// Change attribution.
// ---------------------------------------------------------------------------

/// `x_t = 0.5 x_{t-1} + e`, `y_t = c + 2 x_{t-1} + u` with the intercept `c` jumping by 5 midway.
fn shifted_chain() -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; N];
    let mut y = vec![0.0; N];
    for t in 1..N {
        x[t] = 0.5 * x[t - 1] + disturbance(2 * t);
        let intercept = if t < N / 2 { 1.0 } else { 6.0 };
        y[t] = intercept + 2.0 * x[t - 1] + disturbance(2 * t + 1);
    }
    (x, y)
}

fn exact_change(baseline: (usize, usize), comparison: (usize, usize)) -> CausalQuery {
    CausalQuery::ChangeAttribution(
        ChangeAttributionQuery::new(
            id(1),
            PopulationSelector::TimeRange { start: baseline.0, end: baseline.1 },
            PopulationSelector::TimeRange { start: comparison.0, end: comparison.1 },
        )
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() }),
    )
}

/// outcome: equivalent total, contributions and observed change on the shared cohort; the interval
/// is `not_equivalent` (circular block against iid row bootstrap).
///
/// Mechanism attribution on `x_{t-1} -> y_t` unfolds the lagged graph into a static SCM on a table
/// whose incomplete lag prefix copies the first observation, so the temporal cohort is the
/// series length, not the aligned length. The static attribution on that same padded table, split
/// at the same row, is the temporal one: the exact Shapley decomposition is deterministic given
/// the fitted mechanisms, so the total and every component agree to rounding. With replicates the
/// temporal fit resamples circular blocks of the series and records that dependence, where the
/// static fit resamples iid rows.
#[test]
fn horizon_one_change_attribution_reproduces_the_static_decomposition_with_block_uncertainty() {
    let (x, y) = shifted_chain();
    let half = N / 2;
    // The unfolded table: the lagged column copies the first observation on the first row.
    let mut padded_x = vec![x[0]];
    padded_x.extend_from_slice(&x[..N - 1]);
    let query = exact_change((0, half), (half, N));

    let temporal = run(frequentist(
        Study::series(series(&[("x", &x), ("y", &y)])).graph(pulse_graph()).query(query.clone()),
    ));
    let static_result = run(frequentist(
        Study::tabular(table(&[("x", &padded_x), ("y", &y)]))
            .graph(dag(2, &[(0, 1)]))
            .query(query.clone()),
    ));
    let (a, b) =
        (temporal.change_attribution().unwrap(), static_result.change_attribution().unwrap());
    assert!((a.total_change - 5.0).abs() < 0.5, "the window mean shifts by 5: {}", a.total_change);
    // Independent reference: the model reproduces the observed mean shift of the outcome
    // between the two windows (the mechanisms are linear with an intercept), so the total
    // change is that raw difference of means up to Monte Carlo error.
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let raw_shift = mean(&y[half..N]) - mean(&y[0..half]);
    assert!(
        (a.total_change - raw_shift).abs() < 0.1,
        "total change {} vs raw mean shift {raw_shift}",
        a.total_change
    );
    // The fitted mechanisms are the same, but the unfolded graph holds one more root (the
    // unused lag-zero treatment column), so the two Monte Carlo outcome laws draw their
    // exogenous noise on different streams: the totals and contributions agree to Monte Carlo
    // accuracy, while the observed change is a deterministic functional of the rows.
    let monte_carlo = |left: f64, right: f64| (left - right).abs() <= 0.05;
    assert!(
        monte_carlo(a.total_change, b.total_change),
        "{} vs {}",
        a.total_change,
        b.total_change
    );
    assert_eq!(a.contributions.len(), b.contributions.len());
    let sorted = |result: &antecedent_attribution::ChangeAttributionResult| {
        let mut values: Vec<f64> = result.contributions.iter().map(|c| c.contribution).collect();
        values.sort_by(f64::total_cmp);
        values
    };
    for (left, right) in sorted(a).iter().zip(sorted(b)) {
        assert!(monte_carlo(*left, right), "{left} vs {right}");
    }
    assert_eq!(a.observed_change.is_some(), b.observed_change.is_some());
    if let (Some(left), Some(right)) = (a.observed_change, b.observed_change) {
        assert!(close(left, right));
    }

    // Uncertainty: block resampling of the series against iid rows.
    let with_replicates = |builder: StudyBuilder| {
        builder
            .inference(InferenceMode::Frequentist)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(99)
            .build()
            .unwrap()
            .run(&ctx())
            .unwrap()
    };
    let blocked = with_replicates(
        Study::series(series(&[("x", &x), ("y", &y)])).graph(pulse_graph()).query(query.clone()),
    );
    let rows = with_replicates(
        Study::tabular(table(&[("x", &padded_x), ("y", &y)])).graph(dag(2, &[(0, 1)])).query(query),
    );
    assert_eq!(
        blocked.interval().as_ref().map(|binding| binding.method),
        Some(IntervalMethod::CircularBlockSe)
    );
    assert_ne!(
        rows.interval().as_ref().map(|binding| binding.method),
        Some(IntervalMethod::CircularBlockSe)
    );
    assert!(blocked.estimate().as_effect().unwrap().se_bootstrap().is_some_and(|se| se > 0.0));
}

// ---------------------------------------------------------------------------
// Transport and ADMG projections.
// ---------------------------------------------------------------------------

/// `a, y, trial, s, e`: an arm coin, an outcome with effect 1 and independent noise, a trial coin and
/// the recorded 0.5 selection and treatment probabilities.
fn transport_rows(n: usize) -> TabularData {
    let (a, y, trial) = transport_vectors(n);
    let half = vec![0.5; n];
    table(&[("a", &a), ("y", &y), ("trial", &trial), ("s", &half), ("e", &half)])
}

/// The arm, outcome and trial columns of [`transport_rows`].
fn transport_vectors(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Serially independent draws (splitmix64 of the row and a stream salt): the rows of this
    // fixture must carry no autocorrelation for the two uncertainty products to be comparable.
    let unit = |i: usize, salt: u64| -> f64 {
        let mut z = (i as u64).wrapping_add(salt).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    };
    let a: Vec<f64> = (0..n).map(|i| f64::from(unit(i, 1) < 0.5)).collect();
    let y: Vec<f64> = (0..n).map(|i| a[i] + 0.5 * (unit(i, 2) - 0.5) * 2.0).collect();
    let trial: Vec<f64> = (0..n).map(|i| f64::from(unit(i, 3) < 0.5)).collect();
    (a, y, trial)
}

fn transport_result(data: &TabularData, graph: impl antecedent::IntoGraphInput) -> StudyResult {
    let curve = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: id(1),
        treatment: ContinuousDomain::new(id(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
    });
    Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Transport(TransportQuery::new(curve, "trial", "target", [id(0)])))
        .selection_targets(Arc::from([]))
        .transport_trial(TransportTrialSpec {
            trial: id(2),
            selection_probability: id(3),
            treatment_probability: id(4),
        })
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap()
}

/// outcome: equivalent transported contrast; the interval is `not_equivalent` (long-run
/// variance against iid rows).
///
/// A temporal ADMG whose every variable appears once at lag zero is the contemporaneous
/// projection of the static ADMG, so the transported contrast is the same inverse-probability
/// weighted number. The temporal route reads the rows as one series and reports a Newey-West
/// long-run standard error where the static route reports the iid score standard error; on
/// serially independent rows the two are of the same size but they are different estimators.
#[test]
fn horizon_one_transport_is_the_static_contrast_with_a_long_run_variance() {
    let data = transport_rows(1000);
    let mut admg = Admg::with_variables(5);
    admg.insert_directed(node(0), node(1)).unwrap();
    let temporal = TemporalAdmg::from_static_contemporaneous(&admg).unwrap();

    let series_result = transport_result(&data, temporal);
    let rows_result = transport_result(&data, admg);
    let series_effect = series_result.estimate().as_effect().expect("a temporal transport effect");
    let rows_effect = rows_result.estimate().as_effect().expect("a static transport effect");
    assert!((series_effect.ate - rows_effect.ate).abs() < 1e-12);
    assert!(
        (series_result.transport().unwrap().ipw - rows_result.transport().unwrap().ipw).abs()
            < 1e-12
    );
    assert!(
        (rows_effect.ate - 1.0).abs() < 0.2,
        "the fixture carries the effect: {}",
        rows_effect.ate
    );
    assert!(matches!(
        series_effect.uncertainty.analytic.kind,
        Some(AnalyticSeKind::NeweyWest { lag: 10 })
    ));
    assert!(!matches!(
        rows_effect.uncertainty.analytic.kind,
        Some(AnalyticSeKind::NeweyWest { .. })
    ));
    let (series_se, rows_se) =
        (series_effect.uncertainty.analytic.se, rows_effect.uncertainty.analytic.se);
    assert!(series_se.is_finite() && rows_se.is_finite() && rows_se > 0.0);
    assert!(series_se > 0.5 * rows_se && series_se < 2.0 * rows_se, "{series_se} vs {rows_se}");
}

/// The front-door-free confounded fixture: `z -> t`, `z -> y`, `t -> y` plus a disconnected latent pair.
fn functional_admg_fixture() -> (TabularData, Admg) {
    let (mut treatment, mut outcome, mut confounder) = (Vec::new(), Vec::new(), Vec::new());
    let (mut latent_left, mut latent_right) = (Vec::new(), Vec::new());
    // P(z=1) = 1/2; the outcome risks are .2/.8 when z=0 and .4/.6 when z=1 for t=0/1, so the
    // adjustment formula gives E[Y | do(t=1)] - E[Y | do(t=0)] = .4.
    for (z, t, o, count) in [
        (0.0, 0.0, 0.0, 80),
        (0.0, 0.0, 1.0, 20),
        (0.0, 1.0, 0.0, 20),
        (0.0, 1.0, 1.0, 80),
        (1.0, 0.0, 0.0, 60),
        (1.0, 0.0, 1.0, 40),
        (1.0, 1.0, 0.0, 40),
        (1.0, 1.0, 1.0, 60),
    ] {
        for _ in 0..count {
            for latent in [0.0, 1.0] {
                treatment.push(t);
                outcome.push(o);
                confounder.push(z);
                latent_left.push(latent);
                latent_right.push(latent);
            }
        }
    }
    let data = table(&[
        ("t", &treatment),
        ("y", &outcome),
        ("z", &confounder),
        ("u", &latent_left),
        ("v", &latent_right),
    ]);
    let mut graph = Admg::with_variables(5);
    graph.insert_directed(node(2), node(0)).unwrap();
    graph.insert_directed(node(2), node(1)).unwrap();
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_bidirected(node(3), node(4)).unwrap();
    (data, graph)
}

fn functional_effect(data: &TabularData, graph: impl antecedent::IntoGraphInput) -> StudyResult {
    run(frequentist(
        Study::tabular(data.clone())
            .graph(graph)
            .query(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(id(0), id(1))))
            .identifier(antecedent::IdentifierId::GeneralId)
            .estimator(EstimatorId::FunctionalEffect),
    ))
}

/// outcome: equivalent.
///
/// The temporal ADMG average effect on one contemporaneous window is the static ADMG functional
/// effect: the same identifier (`general.id`), the same plug-in functional on the same rows, so
/// the effect, its analytic standard error and the estimator are the static ones. The adjusted
/// risk difference is counted from the contingency table: `.4`.
#[test]
fn horizon_one_admg_average_effect_is_the_static_functional_effect() {
    let (data, admg) = functional_admg_fixture();
    let temporal = TemporalAdmg::from_static_contemporaneous(&admg).unwrap();
    let static_result = functional_effect(&data, admg);
    let temporal_result = functional_effect(&data, temporal);
    let (static_effect, temporal_effect) = (
        static_result.estimate().as_effect().unwrap(),
        temporal_result.estimate().as_effect().unwrap(),
    );
    assert!((static_effect.ate - 0.4).abs() < 1e-9, "{}", static_effect.ate);
    assert!((temporal_effect.ate - static_effect.ate).abs() < 1e-12);
    assert!(
        (temporal_effect.se_analytic() - static_effect.se_analytic()).abs() < 1e-12
            || (temporal_effect.se_analytic().is_nan() && static_effect.se_analytic().is_nan())
    );
    assert_eq!(
        temporal_result.logical_plan.estimator.as_deref(),
        static_result.logical_plan.estimator.as_deref()
    );
    assert_eq!(temporal_result.support_status(), static_result.support_status());
}

// ---------------------------------------------------------------------------
// Conditional effect.
// ---------------------------------------------------------------------------

/// `t` continuous and exogenous, `w` an exogenous modifier, `y_i = 1 + 1.2 t_{i-1} + 0.5 w_i + 0.9 t_{i-1} w_i + e`.
fn conditional_series() -> [Vec<f64>; 3] {
    let t: Vec<f64> =
        (0..N).map(|i| 0.8 * (0.61 * i as f64).sin() + 0.4 * (1.7 * i as f64).cos()).collect();
    let w: Vec<f64> = (0..N).map(|i| 1.0 + 0.7 * (0.43 * i as f64 + 0.5).sin()).collect();
    let mut y = vec![0.0; N];
    for i in 1..N {
        y[i] = 1.0 + 1.2 * t[i - 1] + 0.5 * w[i] + 0.9 * t[i - 1] * w[i] + 0.1 * disturbance(i);
    }
    [t, y, w]
}

/// outcome: equivalent point (marginal effect over the modifier law); `not_equivalent`
/// uncertainty.
///
/// With the dose levels 0 and 2 (so that the static route does not substitute its binary AIPW
/// scores) the static conditional effect and the temporal pulse with one modifier are the same
/// interaction regression `y ~ 1 + t + w + t w` on the aligned rows, averaged over the observed
/// modifier: `(b_t + b_tw wbar) * 2`. The static route publishes the delta-method standard error
/// including the sampling variance of the modifier mean; the temporal route is point-only.
#[test]
fn horizon_one_conditional_effect_is_the_interaction_regression_marginal_over_the_modifier() {
    let [t, y, w] = conditional_series();
    let (aligned_t, aligned_y, aligned_w) = (&t[..N - 1], &y[1..], &w[1..]);
    let interaction: Vec<f64> = aligned_t.iter().zip(aligned_w).map(|(a, b)| a * b).collect();
    let coefficients = ols(&[aligned_t, aligned_w, &interaction], aligned_y);
    let mean_w = aligned_w.iter().sum::<f64>() / aligned_w.len() as f64;
    let expected = (coefficients[1] + coefficients[3] * mean_w) * 2.0;
    assert!(expected.abs() > 0.5, "the fixture must carry a conditional effect: {expected}");

    let mut graph = TemporalDag::empty();
    let treatment_lag = ensure_lagged(&mut graph, id(0), Lag::from_raw(1)).unwrap();
    let outcome_now = ensure_lagged(&mut graph, id(1), Lag::CONTEMPORANEOUS).unwrap();
    let modifier_now = ensure_lagged(&mut graph, id(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(treatment_lag, outcome_now).unwrap();
    graph.insert_directed(modifier_now, outcome_now).unwrap();
    let temporal = run(frequentist(
        Study::series(series(&[("t", &t), ("y", &y), ("w", &w)])).graph(graph).query(
            CausalQuery::TemporalEffect(
                TemporalEffectQuery::pulse(id(0), id(1), 2.0)
                    .with_policy(TemporalPolicy::pulse(-1))
                    .with_horizon_steps(1)
                    .with_max_history_lag(Some(1))
                    .with_effect_modifiers([id(2)]),
            ),
        ),
    ));
    let static_result = run(frequentist(
        Study::tabular(table(&[("t", aligned_t), ("y", aligned_y), ("w", aligned_w)]))
            .graph(dag(3, &[(0, 1), (2, 1)]))
            .query(CausalQuery::ConditionalEffect(
                ConditionalEffectQuery::try_new(
                    AverageEffectQuery::with_levels(id(0), id(1), 0.0, 2.0)
                        .with_effect_modifiers([id(2)]),
                )
                .unwrap(),
            )),
    ));
    let temporal_effect = temporal.estimate().as_effect().unwrap();
    let static_effect = static_result.estimate().as_effect().unwrap();
    assert!(close(temporal_effect.ate, expected), "{} vs {expected}", temporal_effect.ate);
    assert!(close(static_effect.ate, expected), "{} vs {expected}", static_effect.ate);
    assert!(static_effect.se_analytic().is_finite() && static_effect.se_analytic() > 0.0);
    assert!(
        temporal_effect.se_analytic().is_nan(),
        "the temporal conditional effect is point-only"
    );
}
