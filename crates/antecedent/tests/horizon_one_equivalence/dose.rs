//! Continuous-dose and derivative families at horizon one.
//!
//! A one-decision, lag-zero dose design with a constant recorded density is a randomized-dose
//! sample of independent rows. The local-quadratic temporal target is then the static
//! randomized-dose row: same fit, same selected rows, same smoothing-bias bound. The kernel
//! targets are different objects whose kernels, normalization and nuisance structure are part
//! of the estimand, so each is pinned against its own closed form and against the other.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_core::{DerivativeWeighting, TemporalDoseTarget};
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_estimate::continuous_dose::{
    DoseRows, DoseRules, DoseWindow, fixed_dose_policy_value,
};
use antecedent_estimate::dose_grid_functional::{
    ClaimSet, DoseDesign, DoseFunctional, DoseGridRequest, IntervalCalibration,
    estimate_dose_grid_bias_aware, estimate_dose_grid_functional,
};

const DOSE_UNITS: usize = 1200;
const DOSE_BANDWIDTH: f64 = 0.3;

/// Equispaced doses on `(0, 2)` (density 0.5) and a smooth outcome with deterministic noise.
fn dose_fixture(n: usize) -> (Vec<f64>, Vec<f64>) {
    let dose: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64 + 0.5) / n as f64).collect();
    let outcome = dose
        .iter()
        .enumerate()
        .map(|(i, d)| 1.0 + d + 0.5 * d * d + 0.3 * (1.7 * i as f64).sin())
        .collect();
    (dose, outcome)
}

/// One decision per unit, lag zero, constant recorded density 0.5.
fn dose_query(dose: &[f64], target: TemporalDoseTarget) -> TemporalDesignQuery {
    let n = dose.len();
    TemporalDesignQuery {
        outcome: id(0),
        units: labels("unit", n).into(),
        periods: vec![0_i64; n].into(),
        lag: 0,
        assignment: TemporalAssignment::Dose {
            dose: dose.into(),
            density: vec![0.5; n].into(),
            bandwidth: DOSE_BANDWIDTH,
            target,
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    }
}

/// Gaussian-kernel local quadratic at `at` from the normal equations: the level and the slope,
/// each with its HC0 sandwich standard error `sqrt(sum (l_i e_i)^2)`.
fn local_quadratic_reference(dose: &[f64], y: &[f64], at: f64, h: f64) -> ([f64; 2], [f64; 2]) {
    let n = dose.len();
    let weight: Vec<f64> = dose.iter().map(|d| (-0.5 * ((d - at) / h).powi(2)).exp()).collect();
    let mut gram = vec![vec![0.0; 3]; 3];
    let mut moment = vec![0.0; 3];
    for i in 0..n {
        let u = dose[i] - at;
        let x = [1.0, u, u * u];
        for a in 0..3 {
            moment[a] += weight[i] * x[a] * y[i];
            for b in 0..3 {
                gram[a][b] += weight[i] * x[a] * x[b];
            }
        }
    }
    let beta = solve(gram.clone(), moment);
    let mut squares = [0.0; 2];
    for i in 0..n {
        let u = dose[i] - at;
        let x = [1.0, u, u * u];
        let smoother = solve(gram.clone(), x.to_vec());
        let residual = y[i] - (beta[0] + beta[1] * u + beta[2] * u * u);
        for c in 0..2 {
            squares[c] += (weight[i] * smoother[c] * residual).powi(2);
        }
    }
    ([beta[0], beta[1]], [squares[0].sqrt(), squares[1].sqrt()])
}

/// outcome: equivalent point, selected rows and smoothing-bias bound; the standard error carries
/// the cluster finite-sample factor.
///
/// With one decision per unit and a constant density the inverse-density weights drop out of the
/// local quadratic, so the temporal level and slope are the static randomized-dose row (and its
/// bias-aware extension) on the same rows. The only differences are the cluster correction
/// `g / (g - 1)` on the sandwich variance (every row is its own cluster), so the temporal
/// interval is never narrower than the static one, and the fact that both are unlicensed with
/// calibration unmeasured.
#[test]
fn horizon_one_local_quadratic_dose_grid_reproduces_the_static_randomized_dose_row() {
    let (dose, outcome) = dose_fixture(DOSE_UNITS);
    let nf = DOSE_UNITS as f64;
    let grid = [0.6, 1.0, 1.4];
    let bound = 0.5;
    let data = table(&[("y", &outcome)]);
    for derivative in [false, true] {
        let temporal = run_design(
            data.clone(),
            dose_query(
                &dose,
                TemporalDoseTarget::LocalQuadratic {
                    grid: Arc::from(grid),
                    derivative,
                    curvature_bound: Some(bound),
                },
            ),
            0,
        );
        let fit = temporal.temporal_design().unwrap();
        assert_eq!(fit.design.as_ref(), "dose");
        assert_eq!((fit.pairs, fit.clusters), (DOSE_UNITS as u64, DOSE_UNITS as u64));

        let request = DoseGridRequest {
            dose: &dose,
            outcome: &outcome,
            grid: &grid,
            bandwidth: DOSE_BANDWIDTH,
            bandwidth_range: (0.05, 1.0),
            minimum_local_ess: 10.0,
            functional: if derivative { DoseFunctional::Derivative } else { DoseFunctional::Level },
            claims: ClaimSet { pointwise_level: !derivative, derivative, simultaneous_band: false },
            design: DoseDesign::RandomizedDose,
        };
        let row = estimate_dose_grid_functional(&request).unwrap();
        let aware = estimate_dose_grid_bias_aware(&request, bound, 200, &ctx()).unwrap();
        assert_eq!(row.n_rows, DOSE_UNITS);
        assert!(aware.band.is_some() && fit.band_critical_value.is_some());

        let correction = (nf / (nf - 1.0)).sqrt();
        for (j, &at) in grid.iter().enumerate() {
            let (value, se) = local_quadratic_reference(&dose, &outcome, at, DOSE_BANDWIDTH);
            let (expected, expected_se) =
                if derivative { (value[1], se[1]) } else { (value[0], se[0]) };
            let (static_value, static_se) = if derivative {
                (row.derivatives[j].derivative, row.derivatives[j].standard_error)
            } else {
                (row.levels[j].level, row.levels[j].standard_error)
            };
            let label = format!("derivative={derivative} dose={at}");
            // Same functional on the same rows, against the normal equations.
            assert!(close(fit.estimate[j], expected), "{label}: {} vs {expected}", fit.estimate[j]);
            assert!(close(static_value, expected), "{label}: {static_value} vs {expected}");
            // The static sandwich is the HC0 influence norm; the temporal one is clustered.
            assert!(close(static_se, expected_se), "{label}: {static_se} vs {expected_se}");
            assert!(
                close(fit.standard_error[j], expected_se * correction),
                "{label}: {} vs {}",
                fit.standard_error[j],
                expected_se * correction
            );
            // Same linear smoother, same worst-case bias under the declared bound.
            let temporal_bias = fit.smoothing_bias_bound.as_ref().unwrap()[j];
            assert!(close(temporal_bias, aware.smoothing_bias_bound[j]), "{label}");
            assert!(temporal_bias > 0.0, "{label}: the bound must bite");
            // The bias-aware interval is wider for the larger standard error.
            let temporal_half = fit.upper[j] - fit.estimate[j];
            let static_half = aware.intervals[j].upper - static_value;
            assert!(temporal_half >= static_half - 1e-12, "{label}: {temporal_half} {static_half}");
            assert!(temporal_half - static_half < 0.1 * static_half, "{label}");
            assert_eq!(aware.intervals[j].calibration, IntervalCalibration::Unmeasured);
        }
        assert!(!fit.licensed);
        assert_eq!(fit.calibration.as_ref(), "unmeasured");
    }
}

/// `int_0^2 kernel(u) m(u) du` by the midpoint rule.
fn dose_integral(kernel: impl Fn(f64) -> f64, m: impl Fn(f64) -> f64) -> f64 {
    let steps = 400_000_usize;
    let step = 2.0 / steps as f64;
    (0..steps)
        .map(|k| {
            let u = (k as f64 + 0.5) * step;
            kernel(u) * m(u)
        })
        .sum::<f64>()
        * step
}

fn gaussian_kernel(at: f64) -> impl Fn(f64) -> f64 {
    move |u| {
        let z = (u - at) / DOSE_BANDWIDTH;
        (-0.5 * z * z).exp() / (DOSE_BANDWIDTH * (2.0 * std::f64::consts::PI).sqrt())
    }
}

fn epanechnikov_kernel(at: f64) -> impl Fn(f64) -> f64 {
    move |u| {
        let z = (u - at) / DOSE_BANDWIDTH;
        if z.abs() < 1.0 { 0.75 * (1.0 - z * z) } else { 0.0 }
    }
}

/// Gaussian-smoothed response (what the temporal kernel route targets).
fn gaussian_smoothed(at: f64, m: impl Fn(f64) -> f64) -> f64 {
    dose_integral(gaussian_kernel(at), m)
}

/// Epanechnikov-smoothed response, normalized by its window mass (what the static route targets).
fn epanechnikov_smoothed(at: f64, m: impl Fn(f64) -> f64) -> f64 {
    dose_integral(epanechnikov_kernel(at), m) / dose_integral(epanechnikov_kernel(at), |_| 1.0)
}

fn cubic_dose_fixture() -> (Vec<f64>, Vec<f64>) {
    let (dose, _) = dose_fixture(DOSE_UNITS);
    let outcome = dose.iter().map(|d| d.powi(3)).collect();
    (dose, outcome)
}

/// outcome: `not_equivalent` (kernel and normalization are part of the estimand); equivalent
/// design and rows.
///
/// The temporal dose contrast is the Gaussian-kernel Horvitz-Thompson contrast of the smoothed
/// response `theta_h(a) = int phi_h(u - a) m(u) du`. The static fixed dose policy is the
/// Epanechnikov-kernel self-normalized (Hajek) local average of the same rows. For a mean that is
/// not quadratic the two smoothing biases `h^2 m''(a) / 2` and `h^2 m''(a) / 10` do not cancel in
/// a contrast, so on `m(d) = d^3` the two routes estimate different numbers; each reproduces its
/// own closed form, and the point-only static answer carries no interval where the temporal one
/// carries a cluster sandwich.
#[test]
fn horizon_one_smoothed_dose_contrast_is_gaussian_horvitz_thompson_while_the_static_policy_value_is_epanechnikov_hajek()
 {
    let (dose, outcome) = cubic_dose_fixture();
    let (active, control) = (1.0, 0.6);
    let m = |d: f64| d.powi(3);

    let temporal = run_design(
        table(&[("y", &outcome)]),
        dose_query(&dose, TemporalDoseTarget::SmoothedContrast { active, control }),
        0,
    );
    let fit = temporal.temporal_design().unwrap();
    let gaussian_contrast = gaussian_smoothed(active, m) - gaussian_smoothed(control, m);
    assert!(
        (fit.estimate[0] - gaussian_contrast).abs() < 1e-6,
        "{} vs {gaussian_contrast}",
        fit.estimate[0]
    );
    assert!(fit.standard_error[0].is_finite());

    let groups = vec!["group".to_owned(); dose.len()];
    let density = vec![0.5; dose.len()];
    let policy = [("group".to_owned(), active)];
    let reference = [("group".to_owned(), control)];
    let value = fixed_dose_policy_value(
        &DoseRows { outcome: &outcome, dose: &dose, groups: &groups, density: &density },
        &DoseRules { policy: &policy, reference: &reference },
        &DoseWindow { bandwidth: DOSE_BANDWIDTH, min_local_support: 2, density_known: true },
    )
    .unwrap();
    let epanechnikov_contrast =
        epanechnikov_smoothed(active, m) - epanechnikov_smoothed(control, m);
    assert!(
        (value.incremental_value - epanechnikov_contrast).abs() < 1e-4,
        "{} vs {epanechnikov_contrast}",
        value.incremental_value
    );
    assert!((value.policy_value - epanechnikov_smoothed(active, m)).abs() < 1e-4);

    // The Gaussian smoothing bias of the cubic is 3 a h^2; the Epanechnikov one is a fifth of it.
    let bias_gap = gaussian_contrast - epanechnikov_contrast;
    assert!((bias_gap - 0.8 * 3.0 * DOSE_BANDWIDTH.powi(2) * (active - control)).abs() < 0.02);
    assert!(bias_gap > 0.05, "the two smoothed targets must differ: {bias_gap}");
    assert!((fit.estimate[0] - value.incremental_value).abs() > 0.05);
}

/// outcome: `not_equivalent` (kernel and target population).
///
/// The static smoothed dose-response transport fixes the Epanechnikov kernel and averages the
/// fitted curve over a target population with cross-fitted nuisances; any other kernel name is a
/// refusal. The temporal smoothed curve is the Gaussian-kernel inverse-density score of the
/// trial doses alone. On `m(d) = d^3` the Gaussian target exceeds the Epanechnikov one by
/// `0.8 * 3 a h^2`, and the temporal curve reproduces the Gaussian value at every grid dose.
#[test]
fn horizon_one_smoothed_curve_is_gaussian_and_the_static_transport_target_fixes_the_epanechnikov_kernel()
 {
    use antecedent_core::SmoothingKernel;
    use antecedent_estimate::smoothed_dose::parse_smoothing_kernel;

    let (dose, outcome) = cubic_dose_fixture();
    let grid = [0.7, 1.0];
    let m = |d: f64| d.powi(3);
    let temporal = run_design(
        table(&[("y", &outcome)]),
        dose_query(&dose, TemporalDoseTarget::SmoothedCurve { grid: Arc::from(grid) }),
        0,
    );
    let fit = temporal.temporal_design().unwrap();
    assert_eq!(fit.estimate.len(), grid.len());
    for (j, &at) in grid.iter().enumerate() {
        let gaussian = gaussian_smoothed(at, m);
        let epanechnikov = epanechnikov_smoothed(at, m);
        assert!(
            (fit.estimate[j] - gaussian).abs() < 1e-6,
            "{at}: {} vs {gaussian}",
            fit.estimate[j]
        );
        assert!((fit.estimate[j] - epanechnikov).abs() > 0.05, "{at}: kernels must differ");
        assert!(
            (gaussian - epanechnikov - 0.8 * 3.0 * at * DOSE_BANDWIDTH.powi(2)).abs() < 0.02,
            "{at}"
        );
    }
    assert_eq!(
        parse_smoothing_kernel("epanechnikov").unwrap().name(),
        SmoothingKernel::Epanechnikov.name()
    );
    assert!(parse_smoothing_kernel("gaussian").is_err(), "the estimand fixes the kernel");
}

/// `x_t = 2 + sin(0.37 t) + 0.3 cos(1.1 t)`, `y_t = 3 + 2 x_{t-1} + noise`.
pub(super) fn derivative_series() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..N)
        .map(|t| {
            let t = t as f64;
            2.0 + (0.37 * t).sin() + 0.3 * (1.1 * t).cos()
        })
        .collect();
    let mut y = vec![0.05 * disturbance(0); N];
    for t in 1..N {
        y[t] = 3.0 + 2.0 * x[t - 1] + 0.05 * disturbance(t);
    }
    (x, y)
}

fn average_derivative() -> ResponseFunctional {
    ResponseFunctional::AverageDerivative {
        outcome: id(1),
        treatment: id(0),
        weighting: DerivativeWeighting::Observed,
    }
}

fn scalar_response(result: &StudyResult) -> f64 {
    let response = result.response().expect("a response payload");
    match &response.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        other => panic!("expected a point-identified scalar, got {other:?}"),
    }
}

/// outcome: equivalent target; estimator class is `not_equivalent` (parametric linear against
/// cross-fitted semiparametric).
///
/// The temporal average derivative at one lag is the slope of the unfolded linear mechanism, so on
/// the shared lag-aligned rows it is exactly the least-squares slope. The static average
/// derivative estimates the same population derivative with a cross-fitted additive nuisance and
/// a Riesz correction, so it does not reproduce the least-squares closed form. Its value is pinned
/// to an independent evaluation of the documented estimator on these exact rows (index-modulo
/// five folds, a six-function cubic B-spline outcome smooth at sample-quantile knots with the
/// integrated squared second-derivative penalty at lambda 1, the derivative of that smooth at
/// each held-out dose, plus the Gaussian-score Riesz residual correction with the training mean
/// and the n - 1 variance; the exact rational pipeline of
/// `estimate_response_riesz_ade_reference.py` run on these rows, agreeing with the product to
/// well under 1e-6). The knot-aware penalty is zero on a straight line, so the smooth carries the
/// slope 2.001 of this linear mean and the remaining 1.1% shortfall is the documented boundary
/// rule of the cross-fit: a held-out dose at or beyond its fold's training maximum (or below its
/// minimum) has derivative zero and only the Riesz residual term recovers part of it. The
/// earlier coefficient second-difference penalty read 2.1315 here, 6.5% above the slope, because
/// quantile knots are not equispaced. The temporal route discloses the linear-mechanism premise
/// it rests on.
#[test]
fn horizon_one_average_derivative_is_the_least_squares_slope_where_the_static_route_is_semiparametric()
 {
    let (x, y) = derivative_series();
    let (aligned_x, aligned_y) = (&x[..N - 1], &y[1..]);
    let slope = ols(&[aligned_x], aligned_y)[1];
    assert!((slope - 2.0).abs() < 0.05, "the fixture must carry the derivative: {slope}");

    let temporal = run(frequentist(
        Study::series(series(&[("x", &x), ("y", &y)])).graph(pulse_graph()).query(
            CausalQuery::Response(ResponseQuery::new(average_derivative()).with_temporal(
                TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap(),
            )),
        ),
    ));
    let static_result = run(frequentist(
        Study::tabular(table(&[("x", aligned_x), ("y", aligned_y)]))
            .graph(dag(2, &[(0, 1)]))
            .query(CausalQuery::Response(ResponseQuery::new(average_derivative())))
            .response_options(ContinuousResponseOptions {
                bandwidth: Some(0.35),
                ..Default::default()
            }),
    ));

    let temporal_value = scalar_response(&temporal);
    let static_value = scalar_response(&static_result);
    assert!((temporal_value - slope).abs() < 1e-8, "{temporal_value} vs {slope}");
    const STATIC_REFERENCE: f64 = 1.978_809_921_286_839;
    assert!((static_value - STATIC_REFERENCE).abs() < 1e-6, "{static_value} vs {STATIC_REFERENCE}");
    // A linear mean is not shrunk by the penalty: only the boundary rows of the cross-fit fall
    // short of the slope, by under 2%, and the estimate never overshoots it.
    assert!(slope - static_value > 0.0 && slope - static_value < 0.04, "{static_value}");
    assert!(has_diagnostic(&temporal, "estimate.temporal.linear_derivative"));
    assert!(!has_diagnostic(&static_result, "estimate.temporal.linear_derivative"));
}
