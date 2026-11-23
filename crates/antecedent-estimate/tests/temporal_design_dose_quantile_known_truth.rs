//! The dose and quantile temporal designs against values worked out by hand.
//!
//! The arm designs have an exact-rational reference; these designs do not share its scores, so
//! each interval construction is checked here on panels small enough to solve directly:
//!
//! * the kernel-smoothed dose targets against the inverse-density kernel ratio and its
//!   linearised cluster-robust normal interval, summed row by row;
//! * the local-quadratic level against a weighted least-squares solve by Cramer's rule, with
//!   the cluster sandwich of its smoother weights, and against a quadratic response, which a
//!   local quadratic reproduces exactly;
//! * the dose posteriors on two units that share their doses: the smoothed ratio and the
//!   local-quadratic level are then linear in one uniform Dirichlet weight, so each draw is
//!   uniform between the two single-unit values;
//! * the weighted quantile on two units of four decisions: the unit bootstrap has three
//!   resamples with probabilities 1/4, 1/2, 1/4 and the Dirichlet posterior changes value at
//!   known weights, so the percentile interval, the bootstrap spread and the posterior law are
//!   exact.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::temporal_design::dose::{DoseFit, DosePairs, DoseTarget, fit_dose};
use antecedent_estimate::temporal_design::quantile::{QuantileFit, fit_quantile};
use antecedent_estimate::temporal_design::{PanelRows, lagged_pairs};

/// The 97.5% point of the standard normal law.
const Z: f64 = 1.959_963_984_540_054;
/// Posterior draws of the exact-law checks.
const DRAWS: usize = 4000;

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{what}: {actual} vs {expected}"
    );
}

/// Gaussian density with standard deviation `h` at `u`.
fn dose_kernel_density(u: f64, h: f64) -> f64 {
    let z = u / h;
    (-0.5 * z * z).exp() / (h * (2.0 * std::f64::consts::PI).sqrt())
}

#[derive(Default)]
struct DosePanel {
    unit: Vec<usize>,
    period: Vec<i64>,
    dose: Vec<f64>,
    density: Vec<f64>,
    outcome: Vec<f64>,
}

/// Four units of five decisions: doses step from 0.6 to 1.36, the recorded density alternates
/// between 0.5 and 0.8, and `response` gives the outcome of a decision from its index and dose.
fn graded_panel(response: impl Fn(u32, f64) -> f64) -> DosePanel {
    let mut panel = DosePanel::default();
    for i in 0..20_u32 {
        let d = 0.6 + 0.04 * f64::from(i);
        panel.unit.push(usize::try_from(i / 5).unwrap());
        panel.period.push(i64::from(i % 5));
        panel.dose.push(d);
        panel.density.push(if i % 2 == 0 { 0.5 } else { 0.8 });
        panel.outcome.push(response(i, d));
    }
    panel
}

/// Two units with the same ten doses (0.55 to 1.45) at density 0.5, each with its own response.
fn twin_panel(first: impl Fn(f64) -> f64, second: impl Fn(f64) -> f64) -> DosePanel {
    let mut panel = DosePanel::default();
    for unit in 0..2_usize {
        for i in 0..10_u32 {
            let d = 0.55 + 0.1 * f64::from(i);
            panel.unit.push(unit);
            panel.period.push(i64::from(i));
            panel.dose.push(d);
            panel.density.push(0.5);
            panel.outcome.push(if unit == 0 { first(d) } else { second(d) });
        }
    }
    panel
}

fn quadratic(d: f64) -> f64 {
    1.0 + d + 0.5 * d * d
}

fn other_quadratic(d: f64) -> f64 {
    3.0 - d + 2.0 * d * d
}

fn fit(panel: &DosePanel, bandwidth: f64, target: &DoseTarget, draws: Option<usize>) -> DoseFit {
    let pairs = lagged_pairs(PanelRows { unit: &panel.unit, period: &panel.period }, 0).unwrap();
    let input = DosePairs {
        pairs: &pairs,
        outcome: &panel.outcome,
        dose: &panel.dose,
        density: &panel.density,
        cluster: &panel.unit,
    };
    fit_dose(input, bandwidth, target, draws, 200, &ExecutionContext::for_tests(17)).unwrap()
}

/// Inverse-density kernel score of every decision at dose `at`.
fn kernel_scores(panel: &DosePanel, at: f64, h: f64) -> Vec<f64> {
    (0..panel.dose.len())
        .map(|p| dose_kernel_density(panel.dose[p] - at, h) * panel.outcome[p] / panel.density[p])
        .collect()
}

/// The ratio `theta = sum_p s_p / n` and its standard error
/// `sqrt(G / (G - 1) * sum_g (S_g - n_g theta)^2) / n` over `units` clusters.
fn hand_ratio(scores: &[f64], unit: &[usize], units: usize) -> (f64, f64) {
    let n = scores.len() as f64;
    let theta = scores.iter().sum::<f64>() / n;
    let mut residual = vec![0.0; units];
    for (score, g) in scores.iter().zip(unit) {
        residual[*g] += score - theta;
    }
    let g = units as f64;
    let variance = g / (g - 1.0) * residual.iter().map(|r| r * r).sum::<f64>() / (n * n);
    (theta, variance.sqrt())
}

#[test]
fn smoothed_dose_targets_match_the_hand_kernel_ratio_and_its_cluster_interval() {
    let panel = graded_panel(|i, d| quadratic(d) + 0.1 * (f64::from(i * 7 % 5) - 2.0));
    let h = 0.5;
    let check = |fit: &DoseFit, expected: &[Vec<f64>]| {
        let summary = fit.sampling.as_ref().expect("a frequentist summary");
        assert_eq!(summary.method, "cluster_robust_ratio_normal");
        assert!(summary.smoothing_bias_bound.is_none() && fit.smoothing_bias_bound.is_none());
        assert_eq!((fit.pairs, fit.clusters), (20, 4));
        assert_eq!(summary.estimate.len(), expected.len());
        for (j, scores) in expected.iter().enumerate() {
            let name = &fit.coordinates[j];
            let (theta, se) = hand_ratio(scores, &panel.unit, 4);
            assert!(se > 0.0, "{name}");
            close(summary.estimate[j], theta, 1e-10, name);
            close(summary.standard_error[j], se, 1e-10, name);
            close(summary.lower[j], theta - Z * se, 1e-10, name);
            close(summary.upper[j], theta + Z * se, 1e-10, name);
        }
    };

    let contrast =
        fit(&panel, h, &DoseTarget::SmoothedContrast { active: 1.1, control: 0.9 }, None);
    assert_eq!(
        contrast.coordinates,
        ["smoothed_contrast:1.1-0.9", "smoothed_response@1.1", "smoothed_response@0.9"]
    );
    let (active, control) = (kernel_scores(&panel, 1.1, h), kernel_scores(&panel, 0.9, h));
    let difference: Vec<f64> = active.iter().zip(&control).map(|(a, c)| a - c).collect();
    check(&contrast, &[difference, active, control]);

    let grid = [0.8, 1.0, 1.2];
    let curve = fit(&panel, h, &DoseTarget::SmoothedCurve { grid: grid.to_vec() }, None);
    assert_eq!(curve.coordinates.len(), 3);
    let expected: Vec<Vec<f64>> = grid.iter().map(|at| kernel_scores(&panel, *at, h)).collect();
    check(&curve, &expected);
}

/// Draws of `w a + (1 - w) b` under a uniform `w`: every draw lies between the two values, and
/// the mean, the spread and the equal-tailed 95% endpoints are those of the uniform law.
fn assert_uniform_between(draws: &[f64], reported: [f64; 4], a: f64, b: f64, what: &str) {
    let (low, high) = (a.min(b), a.max(b));
    let width = high - low;
    assert!(width > 0.1, "{what}: the two values must differ");
    let n = draws.len() as f64;
    let sd = width / 12.0_f64.sqrt();
    let [mean, spread, lower, upper] = reported;
    assert!(
        draws.iter().all(|x| *x >= low - 1e-9 * width && *x <= high + 1e-9 * width),
        "{what}: a draw left [{low}, {high}]"
    );
    assert!((mean - 0.5 * (low + high)).abs() <= 5.0 * sd / n.sqrt(), "{what}: mean {mean}");
    assert!((spread - sd).abs() <= 0.05 * sd, "{what}: spread {spread} vs {sd}");
    // Five binomial standard errors of the tail share, as a distance on the uniform scale.
    let slack = 5.0 * (0.025 * 0.975 / n).sqrt() * width;
    assert!((lower - (low + 0.025 * width)).abs() <= slack, "{what}: lower {lower}");
    assert!((upper - (high - 0.025 * width)).abs() <= slack, "{what}: upper {upper}");
}

fn check_uniform_posterior(fit: &DoseFit, method: &str, ends: &[(f64, f64)]) {
    let posterior = fit.posterior.as_ref().expect("a posterior summary");
    assert_eq!(posterior.method, method);
    assert_eq!(posterior.draws.len(), ends.len());
    for (j, (a, b)) in ends.iter().enumerate() {
        assert_eq!(posterior.draws[j].len(), DRAWS);
        assert_uniform_between(
            &posterior.draws[j],
            [
                posterior.mean[j],
                posterior.standard_deviation[j],
                posterior.lower[j],
                posterior.upper[j],
            ],
            *a,
            *b,
            &fit.coordinates[j],
        );
    }
}

#[test]
fn smoothed_dose_posterior_is_uniform_between_the_two_unit_ratios() {
    let panel = twin_panel(quadratic, other_quadratic);
    let h = 0.4;
    // Both units hold ten decisions, so the Dirichlet-weighted ratio is the weighted mean of
    // the two single-unit ratios.
    let unit_ratio = |at: f64, unit: usize| -> f64 {
        kernel_scores(&panel, at, h)
            .iter()
            .zip(&panel.unit)
            .filter(|(_, g)| **g == unit)
            .map(|(score, _)| score)
            .sum::<f64>()
            / 10.0
    };
    let ends = |at: f64| (unit_ratio(at, 0), unit_ratio(at, 1));
    let contrast =
        fit(&panel, h, &DoseTarget::SmoothedContrast { active: 1.2, control: 0.8 }, Some(DRAWS));
    let (active, control) = (ends(1.2), ends(0.8));
    check_uniform_posterior(
        &contrast,
        "bayesian_bootstrap_over_clusters",
        &[(active.0 - control.0, active.1 - control.1), active, control],
    );
    let grid = [0.8, 1.0, 1.2];
    let curve = fit(&panel, h, &DoseTarget::SmoothedCurve { grid: grid.to_vec() }, Some(DRAWS));
    let expected: Vec<(f64, f64)> = grid.iter().map(|at| ends(*at)).collect();
    check_uniform_posterior(&curve, "bayesian_bootstrap_over_clusters", &expected);
}

fn determinant(a: &[[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// Solve the symmetric 3 x 3 system `m x = rhs` by Cramer's rule.
fn cramer(m: &[[f64; 3]; 3], rhs: [f64; 3]) -> [f64; 3] {
    let whole = determinant(m);
    let mut solution = [0.0; 3];
    for (column, value) in solution.iter_mut().enumerate() {
        let mut replaced = *m;
        for (row, entry) in replaced.iter_mut().zip(rhs) {
            row[column] = entry;
        }
        *value = determinant(&replaced) / whole;
    }
    solution
}

/// Inverse-density kernel-weighted least squares of the outcome on `[1, u, u^2]`, `u` the dose
/// minus `at`: the fitted level and its cluster-robust standard error
/// `sqrt(G / (G - 1) * sum_g (sum_{p in g} l_p r_p)^2)` from the smoother weights `l_p` of the
/// level and the residuals `r_p`.
fn hand_local_level(panel: &DosePanel, at: f64, h: f64, units: usize) -> (f64, f64) {
    let n = panel.dose.len();
    let u: Vec<f64> = panel.dose.iter().map(|d| d - at).collect();
    let omega: Vec<f64> = (0..n).map(|p| dose_kernel_density(u[p], h) / panel.density[p]).collect();
    let moment = |power: i32| -> f64 { (0..n).map(|p| omega[p] * u[p].powi(power)).sum() };
    let cross = |power: i32| -> f64 {
        (0..n).map(|p| omega[p] * u[p].powi(power) * panel.outcome[p]).sum()
    };
    let m = [
        [moment(0), moment(1), moment(2)],
        [moment(1), moment(2), moment(3)],
        [moment(2), moment(3), moment(4)],
    ];
    let beta = cramer(&m, [cross(0), cross(1), cross(2)]);
    let level_row = cramer(&m, [1.0, 0.0, 0.0]);
    let mut residual = vec![0.0; units];
    for p in 0..n {
        let basis = [1.0, u[p], u[p] * u[p]];
        let smoother: f64 =
            omega[p] * basis.iter().zip(&level_row).map(|(x, r)| x * r).sum::<f64>();
        let fitted: f64 = basis.iter().zip(&beta).map(|(x, b)| x * b).sum();
        residual[panel.unit[p]] += smoother * (panel.outcome[p] - fitted);
    }
    let g = units as f64;
    (beta[0], (g / (g - 1.0) * residual.iter().map(|r| r * r).sum::<f64>()).sqrt())
}

#[test]
fn local_quadratic_level_matches_an_independent_weighted_least_squares() {
    let grid = [0.8, 1.0, 1.2];
    let h = 0.5;
    let target = DoseTarget::LocalQuadratic {
        grid: grid.to_vec(),
        derivative: false,
        curvature_bound: Some(0.0),
    };
    let panel = graded_panel(|i, d| quadratic(d) + 0.1 * (f64::from(i * 7 % 5) - 2.0));
    let fitted = fit(&panel, h, &target, None);
    let summary = fitted.sampling.as_ref().expect("a frequentist summary");
    assert_eq!(summary.method, "cluster_robust_local_quadratic_bias_aware");
    assert_eq!(fitted.coordinates, ["level@0.8", "level@1", "level@1.2"]);
    // A zero curvature bound declares no smoothing bias, so the bias-aware interval is the
    // normal interval of the cluster-robust standard error.
    assert_eq!(summary.smoothing_bias_bound.as_deref(), Some(&[0.0, 0.0, 0.0][..]));
    for (j, at) in grid.iter().enumerate() {
        let name = &fitted.coordinates[j];
        let (level, se) = hand_local_level(&panel, *at, h, 4);
        assert!(se > 0.0, "{name}");
        close(summary.estimate[j], level, 1e-8, name);
        close(summary.standard_error[j], se, 1e-7, name);
        assert!((summary.lower[j] - (level - Z * se)).abs() <= 1e-5 * se + 1e-8, "{name} lower");
        assert!((summary.upper[j] - (level + Z * se)).abs() <= 1e-5 * se + 1e-8, "{name} upper");
    }
    // A quadratic response is reproduced exactly at every dose, whatever the weights.
    let exact = fit(&graded_panel(|_, d| quadratic(d)), h, &target, None);
    let exact = exact.sampling.as_ref().expect("a frequentist summary");
    for (j, at) in grid.iter().enumerate() {
        close(exact.estimate[j], quadratic(*at), 1e-9, "the quadratic level");
    }
}

#[test]
fn local_quadratic_posterior_is_uniform_between_the_two_unit_curves() {
    // Each unit's outcomes lie on its own quadratic at shared doses, so a Dirichlet-weighted
    // local quadratic returns the weighted mean of the two curves at every grid dose.
    let panel = twin_panel(quadratic, other_quadratic);
    let grid = [0.8, 1.0, 1.2];
    let target = DoseTarget::LocalQuadratic {
        grid: grid.to_vec(),
        derivative: false,
        curvature_bound: Some(0.0),
    };
    let fitted = fit(&panel, 0.4, &target, Some(DRAWS));
    let ends: Vec<(f64, f64)> =
        grid.iter().map(|at| (quadratic(*at), other_quadratic(*at))).collect();
    check_uniform_posterior(&fitted, "bayesian_bootstrap_local_quadratic_projection", &ends);
    assert_eq!(fitted.smoothing_bias_bound.as_deref(), Some(&[0.0, 0.0, 0.0][..]));
}

/// Two units of four decisions. Unit 0: treated outcomes 10 (probability 0.1, weight 10) and 11
/// (weight 2), control outcomes 1 and 2 (weight 2 each). Unit 1: treated 20 and 21, control 5
/// and 6, every weight 2.
fn fit_two_unit_quantile(replicates: usize, draws: Option<usize>) -> QuantileFit {
    let rows = [
        (0_usize, true, 0.1, 10.0),
        (0, true, 0.5, 11.0),
        (0, false, 0.5, 1.0),
        (0, false, 0.5, 2.0),
        (1, true, 0.5, 20.0),
        (1, true, 0.5, 21.0),
        (1, false, 0.5, 5.0),
        (1, false, 0.5, 6.0),
    ];
    let unit: Vec<usize> = rows.iter().map(|row| row.0).collect();
    let period: Vec<i64> = (0..4).chain(0..4).collect();
    let assignment: Vec<bool> = rows.iter().map(|row| row.1).collect();
    let probability: Vec<f64> = rows.iter().map(|row| row.2).collect();
    let outcome: Vec<f64> = rows.iter().map(|row| row.3).collect();
    let pairs = lagged_pairs(PanelRows { unit: &unit, period: &period }, 0).unwrap();
    fit_quantile(
        &pairs,
        &outcome,
        &assignment,
        &probability,
        &unit,
        0.05,
        0.5,
        replicates,
        draws,
        &ExecutionContext::for_tests(23),
    )
    .unwrap()
}

fn exactly(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-9
}

#[test]
fn weighted_median_and_its_unit_bootstrap_match_the_exact_two_unit_law() {
    let fitted = fit_two_unit_quantile(999, None);
    // Treated weights 10, 2, 2, 2 on 10, 11, 20, 21: half the total weight is reached at 10.
    // Control weights 2 each on 1, 2, 5, 6: half is reached at 2.
    let [effect, treated, control] = fitted.estimate;
    assert!(exactly(effect, 8.0) && exactly(treated, 10.0) && exactly(control, 2.0));
    assert_eq!(fitted.replicates, (999, 0));
    assert_eq!((fitted.pairs, fitted.clusters), (8, 2));
    // Resampling two units gives unit 0 twice (1/4): 10 - 1; both (1/2): 10 - 2; unit 1 twice
    // (1/4): 20 - 5. Each end carries a quarter of the mass, so the percentile interval is
    // the range of every coordinate.
    let interval = fitted.interval.expect("a percentile interval");
    for ((lower, upper), (low, high)) in
        interval.iter().zip([(8.0, 15.0), (10.0, 20.0), (1.0, 5.0)])
    {
        assert!(exactly(*lower, low) && exactly(*upper, high), "({lower}, {upper})");
    }
    // Exact bootstrap standard deviations: sqrt(8.5), sqrt(18.75) and 1.5.
    let spread = fitted.standard_error.expect("a bootstrap standard error");
    for (got, want) in spread.iter().zip([8.5_f64.sqrt(), 18.75_f64.sqrt(), 1.5]) {
        assert!((got - want).abs() <= 0.1 * want, "{got} vs {want}");
    }
}

#[test]
fn weighted_median_posterior_matches_the_exact_dirichlet_law() {
    let fitted = fit_two_unit_quantile(0, Some(DRAWS));
    let posterior = fitted.posterior.as_ref().expect("a posterior summary");
    assert_eq!(posterior.method, "bayesian_bootstrap_weighted_quantile");
    assert_eq!(posterior.draws[0].len(), DRAWS);
    // With w the uniform share of unit 0: the treated median is 10 for w >= 1/3, 11 for
    // 1/4 <= w < 1/3 and 20 below; the control median is 2 for w >= 1/2 and 5 below.
    let laws: [&[(f64, f64)]; 3] = [
        &[(8.0, 1.0 / 2.0), (5.0, 1.0 / 6.0), (6.0, 1.0 / 12.0), (15.0, 1.0 / 4.0)],
        &[(10.0, 2.0 / 3.0), (11.0, 1.0 / 12.0), (20.0, 1.0 / 4.0)],
        &[(2.0, 1.0 / 2.0), (5.0, 1.0 / 2.0)],
    ];
    let n = DRAWS as f64;
    for (j, law) in laws.iter().enumerate() {
        let draws = &posterior.draws[j];
        for (value, probability) in *law {
            let share = draws.iter().filter(|x| exactly(**x, *value)).count() as f64 / n;
            let slack = 5.0 * (probability * (1.0 - probability) / n).sqrt();
            assert!((share - probability).abs() <= slack, "coordinate {j} at {value}: {share}");
        }
        let mean: f64 = law.iter().map(|(value, p)| value * p).sum();
        let variance: f64 = law.iter().map(|(value, p)| (value - mean).powi(2) * p).sum();
        assert!(
            (posterior.mean[j] - mean).abs() <= 5.0 * (variance / n).sqrt(),
            "coordinate {j} mean {}",
            posterior.mean[j]
        );
        assert!((posterior.standard_deviation[j] - variance.sqrt()).abs() <= 0.1 * variance.sqrt());
        // Every end value carries far more than 2.5% of the mass.
        let low = law.iter().map(|(value, _)| *value).fold(f64::INFINITY, f64::min);
        let high = law.iter().map(|(value, _)| *value).fold(f64::NEG_INFINITY, f64::max);
        assert!(exactly(posterior.lower[j], low) && exactly(posterior.upper[j], high));
    }
}
