//! Continuous-dose excursions under a known dose density.
//!
//! At each decision the dose `D_t` is drawn from a law the experimenter records; the density
//! `f_t(D_t | history)` of the realized dose is known. Three targets of the dose response
//! `m(d) = E[Y_{t+lag}(d)]` (the excursion of setting the dose at `t` to `d`):
//!
//! * the kernel-smoothed response `theta_h(d) = int K_h(u - d) m(u) du` over the dose support,
//!   with a Gaussian kernel of declared bandwidth `h`. The bandwidth is part of the estimand,
//!   so there is no smoothing bias to account for. Its estimator is the inverse-density
//!   kernel score `K_h(D_t - d) Y_{t+lag} / f_t(D_t)`, a ratio of cluster scores;
//! * the smoothed contrast `theta_h(active) - theta_h(control)` of two doses (the continuous
//!   dose excursion);
//! * the local-quadratic level or derivative of `m` itself on a grid. The inverse-density
//!   weighted local quadratic estimates `m(d)` with a smoothing bias that is *not* part of the
//!   target. Under a declared bound `M` on `|m'''|` the worst-case bias of the linear smoother
//!   `sum_p l_p Y_p` is `M / 6 * sum_p |l_p| |D_p - d|^3`, exactly, and the pointwise interval
//!   is the bias-aware (folded-normal) interval of Armstrong and Kolesar; the simultaneous band
//!   adds the bias bound to the sup-t half-width. Without a declared bound the grid is refused.
//!
//! The Bayesian counterparts reweight whole clusters with Dirichlet weights and re-evaluate
//! the same functional: the kernel ratio, or the weighted local quadratic. The local-quadratic
//! posterior is a posterior of the bandwidth-`h` local-quadratic projection of `m`; its
//! smoothing bias is reported beside it, never inside the credible interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use super::linalg::{cholesky, cholesky_solve};
use super::{
    ClusterScores, LaggedPairs, NOMINAL_LEVEL, PosteriorSummary, SamplingSummary, attach_band,
    folded_normal_critical, invalid, posterior_by_cluster_weights, sup_t_critical,
};
use crate::error::EstimationError;
use crate::util::{count, refuse};

/// Smallest Kish effective sample size of the kernel weights at a requested dose.
pub const MIN_LOCAL_ESS: f64 = 10.0;

/// One dose target.
#[derive(Clone, Debug, PartialEq)]
pub enum DoseTarget {
    /// Kernel-smoothed response at every grid dose.
    SmoothedCurve {
        /// Grid doses.
        grid: Vec<f64>,
    },
    /// Contrast of the kernel-smoothed response between two doses.
    SmoothedContrast {
        /// Active dose.
        active: f64,
        /// Control dose.
        control: f64,
    },
    /// Local-quadratic level (or first derivative) of the dose response on a grid.
    LocalQuadratic {
        /// Grid doses.
        grid: Vec<f64>,
        /// Report the first derivative instead of the level.
        derivative: bool,
        /// Declared bound on the absolute third derivative of the dose response.
        curvature_bound: Option<f64>,
    },
}

/// Paired dose rows of one design.
#[derive(Clone, Copy, Debug)]
pub struct DosePairs<'a> {
    /// Decision and outcome rows.
    pub pairs: &'a LaggedPairs,
    /// Outcome per row.
    pub outcome: &'a [f64],
    /// Realized dose per row.
    pub dose: &'a [f64],
    /// Density of the realized dose given the history, per row.
    pub density: &'a [f64],
    /// Independent cluster per row.
    pub cluster: &'a [usize],
}

/// A fitted dose target.
#[derive(Clone, Debug, PartialEq)]
pub struct DoseFit {
    /// Coordinate names.
    pub coordinates: Vec<String>,
    /// Frequentist summary, under frequentist inference.
    pub sampling: Option<SamplingSummary>,
    /// Posterior summary, under Bayesian inference.
    pub posterior: Option<PosteriorSummary>,
    /// Kish effective sample size of the kernel weights at each requested dose.
    pub local_ess: Vec<f64>,
    /// Worst-case smoothing bias per coordinate, when a curvature bound was declared.
    pub smoothing_bias_bound: Option<Vec<f64>>,
    /// Number of pairs.
    pub pairs: usize,
    /// Number of independent clusters.
    pub clusters: usize,
}

/// Gaussian kernel `phi(u / h) / h`.
fn kernel(u: f64, h: f64) -> f64 {
    let z = u / h;
    (-0.5 * z * z).exp() / (h * (2.0 * std::f64::consts::PI).sqrt())
}

struct Prepared {
    dose: Vec<f64>,
    outcome: Vec<f64>,
    inverse_density: Vec<f64>,
    cluster: Vec<usize>,
    clusters: usize,
}

fn prepare(input: DosePairs<'_>, bandwidth: f64) -> Result<Prepared, EstimationError> {
    let rows = input.outcome.len();
    if input.dose.len() != rows || input.density.len() != rows || input.cluster.len() != rows {
        return Err(invalid("dose, density and cluster arrays must be aligned with the rows"));
    }
    if !(bandwidth.is_finite() && bandwidth > 0.0) {
        return Err(invalid("the bandwidth must be finite and positive"));
    }
    let n = input.pairs.decision.len();
    let mut dose = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut inverse_density = Vec::with_capacity(n);
    let mut raw_cluster = Vec::with_capacity(n);
    for (&t, &o) in input.pairs.decision.iter().zip(&input.pairs.outcome) {
        let (d, f, y) = (input.dose[t], input.density[t], input.outcome[o]);
        if !d.is_finite() || !y.is_finite() || !f.is_finite() || f <= 0.0 {
            return Err(invalid(
                "every paired dose and outcome must be finite and every density positive",
            ));
        }
        dose.push(d);
        outcome.push(y);
        inverse_density.push(1.0 / f);
        raw_cluster.push(input.cluster[t]);
    }
    let (cluster, clusters) = super::dense_labels(&raw_cluster);
    if clusters < super::MIN_CLUSTERS {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "temporal_design.too_few_clusters",
            "the dose design needs at least two independent units",
        ));
    }
    Ok(Prepared { dose, outcome, inverse_density, cluster, clusters })
}

/// Kish ESS of the kernel-times-inverse-density weights at `at`; refused below the floor or
/// outside the observed dose range.
fn local_support(prepared: &Prepared, at: f64, bandwidth: f64) -> Result<f64, EstimationError> {
    let (low, high) = prepared
        .dose
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), d| (lo.min(*d), hi.max(*d)));
    let weights: Vec<f64> = prepared
        .dose
        .iter()
        .zip(&prepared.inverse_density)
        .map(|(d, w)| kernel(d - at, bandwidth) * w)
        .collect();
    let ess = antecedent_kernels::kish_ess(&weights);
    if !at.is_finite() || at < low || at > high {
        return Err(refuse(
            antecedent_core::reason_code!("cell_not_licensed"),
            "temporal_design.dose_outside_support",
            &format!("dose {at} is outside the observed dose range"),
        ));
    }
    if ess.is_nan() || ess < MIN_LOCAL_ESS {
        return Err(refuse(
            antecedent_core::reason_code!("cell_not_licensed"),
            "temporal_design.insufficient_local_weight",
            &format!(
                "dose {at} has local effective sample size {ess} below {MIN_LOCAL_ESS}. Next \
                 attempt: a wider bandwidth or a dose inside the well-supported range"
            ),
        ));
    }
    Ok(ess)
}

/// Fit one dose target.
///
/// `posterior_draws` selects Bayesian inference (the cluster Bayesian bootstrap); otherwise
/// the frequentist summary is computed, with a sup-t band over the grid of a curve target.
///
/// # Errors
/// Misaligned inputs, an unsupported dose, a local-quadratic grid without a curvature bound, a
/// singular local fit, admission or cancellation.
pub fn fit_dose(
    input: DosePairs<'_>,
    bandwidth: f64,
    target: &DoseTarget,
    posterior_draws: Option<usize>,
    band_replicates: usize,
    ctx: &ExecutionContext,
) -> Result<DoseFit, EstimationError> {
    let prepared = prepare(input, bandwidth)?;
    match target {
        DoseTarget::SmoothedCurve { grid } => {
            smoothed(&prepared, bandwidth, grid, None, posterior_draws, band_replicates, ctx)
        }
        DoseTarget::SmoothedContrast { active, control } => smoothed(
            &prepared,
            bandwidth,
            &[*active, *control],
            Some((*active, *control)),
            posterior_draws,
            band_replicates,
            ctx,
        ),
        DoseTarget::LocalQuadratic { grid, derivative, curvature_bound } => local_quadratic(
            &prepared,
            bandwidth,
            grid,
            *derivative,
            *curvature_bound,
            posterior_draws,
            band_replicates,
            ctx,
        ),
    }
}

fn smoothed(
    prepared: &Prepared,
    bandwidth: f64,
    points: &[f64],
    contrast: Option<(f64, f64)>,
    posterior_draws: Option<usize>,
    band_replicates: usize,
    ctx: &ExecutionContext,
) -> Result<DoseFit, EstimationError> {
    if points.is_empty() {
        return Err(invalid("a dose target needs at least one dose"));
    }
    let local_ess = points
        .iter()
        .map(|&at| local_support(prepared, at, bandwidth))
        .collect::<Result<Vec<_>, _>>()?;
    let offset = usize::from(contrast.is_some());
    let k = points.len() + offset;
    let n = prepared.dose.len();
    let _lease = crate::util::admit_array::<f64>(ctx, n, k)?;
    let mut scores = Vec::with_capacity(n * k);
    for p in 0..n {
        let base = prepared.outcome[p] * prepared.inverse_density[p];
        let values: Vec<f64> =
            points.iter().map(|&at| kernel(prepared.dose[p] - at, bandwidth) * base).collect();
        if contrast.is_some() {
            scores.push(values[0] - values[1]);
        }
        scores.extend_from_slice(&values);
    }
    let accumulated = ClusterScores::accumulate(&scores, k, &prepared.cluster)?;
    let mut coordinates = Vec::with_capacity(k);
    if let Some((active, control)) = contrast {
        coordinates.push(format!("smoothed_contrast:{active}-{control}"));
    }
    coordinates.extend(points.iter().map(|at| format!("smoothed_response@{at}")));
    let (sampling, posterior) = if let Some(draws) = posterior_draws {
        (None, Some(super::bayesian_bootstrap(&accumulated, draws, ctx)?))
    } else {
        let mut summary = super::sampling_summary(&accumulated)?;
        if contrast.is_none() && points.len() > 1 {
            let band: Vec<usize> = (0..points.len()).collect();
            super::sup_t_band(&accumulated, &mut summary, &band, band_replicates, ctx)?;
        }
        (Some(summary), None)
    };
    Ok(DoseFit {
        coordinates,
        sampling,
        posterior,
        local_ess,
        smoothing_bias_bound: None,
        pairs: n,
        clusters: prepared.clusters,
    })
}

/// The local-quadratic linear smoother at one dose under row weights `base * cluster weight`.
struct LocalSmoother {
    /// Smoother weights `l_p` of the requested coordinate (level or slope).
    weights: Vec<f64>,
    /// The fitted coordinate `sum_p l_p Y_p`.
    value: f64,
    /// Local residuals `Y_p - x_p' beta`.
    residuals: Vec<f64>,
}

fn local_smoother(
    prepared: &Prepared,
    bandwidth: f64,
    at: f64,
    derivative: bool,
    cluster_weights: Option<&[f64]>,
) -> Result<LocalSmoother, EstimationError> {
    let n = prepared.dose.len();
    let mut information = [0.0; 9];
    let mut moment = [0.0; 3];
    let mut omega = Vec::with_capacity(n);
    for p in 0..n {
        let u = prepared.dose[p] - at;
        let w = kernel(u, bandwidth)
            * prepared.inverse_density[p]
            * cluster_weights.map_or(1.0, |weights| weights[prepared.cluster[p]]);
        let x = [1.0, u, u * u];
        for a in 0..3 {
            moment[a] += w * x[a] * prepared.outcome[p];
            for b in 0..3 {
                information[a * 3 + b] += w * x[a] * x[b];
            }
        }
        omega.push(w);
    }
    let factor = cholesky(&information, 3).ok_or_else(|| {
        refuse(
            antecedent_core::reason_code!("cell_not_licensed"),
            "temporal_design.singular_local_fit",
            &format!("the local quadratic at dose {at} is singular"),
        )
    })?;
    let beta = cholesky_solve(&factor, 3, &moment);
    let coordinate = usize::from(derivative);
    let mut weights = Vec::with_capacity(n);
    let mut residuals = Vec::with_capacity(n);
    for (p, &omega_p) in omega.iter().enumerate() {
        let u = prepared.dose[p] - at;
        let x = [1.0, u, u * u];
        let row = cholesky_solve(&factor, 3, &x);
        weights.push(row[coordinate] * omega_p);
        residuals.push(prepared.outcome[p] - (beta[0] + beta[1] * u + beta[2] * u * u));
    }
    Ok(LocalSmoother { weights, value: beta[coordinate], residuals })
}

#[allow(
    clippy::too_many_arguments,
    reason = "one grid fit: data, bandwidth, grid, coordinate, bias bound, inference and budget"
)]
#[allow(
    clippy::too_many_lines,
    reason = "one grid fit: pointwise local fits, the cluster covariance and the intervals share the residual matrix"
)]
fn local_quadratic(
    prepared: &Prepared,
    bandwidth: f64,
    grid: &[f64],
    derivative: bool,
    curvature_bound: Option<f64>,
    posterior_draws: Option<usize>,
    band_replicates: usize,
    ctx: &ExecutionContext,
) -> Result<DoseFit, EstimationError> {
    if grid.is_empty() {
        return Err(invalid("a local-quadratic grid needs at least one dose"));
    }
    let bound = match curvature_bound {
        Some(bound) if bound.is_finite() && bound >= 0.0 => bound,
        Some(_) => return Err(invalid("the curvature bound must be finite and non-negative")),
        None if posterior_draws.is_some() => f64::NAN,
        None => {
            return Err(refuse(
                antecedent_core::reason_code!("required_option_missing"),
                "temporal_design.curvature_bound_missing",
                "the local-quadratic grid estimates m(d) with a smoothing bias that is not part of \
                 the target, so its interval needs a declared bound on |m'''|. Next attempt: \
                 declare curvature_bound, or ask for the smoothed curve, whose bandwidth is part \
                 of the estimand",
            ));
        }
    };
    let local_ess = grid
        .iter()
        .map(|&at| local_support(prepared, at, bandwidth))
        .collect::<Result<Vec<_>, _>>()?;
    let label = if derivative { "derivative" } else { "level" };
    let coordinates: Vec<String> = grid.iter().map(|at| format!("{label}@{at}")).collect();
    let k = grid.len();
    let g_count = prepared.clusters;
    let _lease = crate::util::admit_array::<f64>(ctx, g_count + prepared.dose.len(), k)?;
    let mut estimate = Vec::with_capacity(k);
    let mut residuals = vec![0.0; g_count * k];
    let mut bias = Vec::with_capacity(k);
    for (j, &at) in grid.iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_stats::StatsError::Cancelled.into());
        }
        let smoother = local_smoother(prepared, bandwidth, at, derivative, None)?;
        estimate.push(smoother.value);
        let mut worst = 0.0;
        for p in 0..prepared.dose.len() {
            residuals[prepared.cluster[p] * k + j] += smoother.weights[p] * smoother.residuals[p];
            worst += smoother.weights[p].abs() * (prepared.dose[p] - at).abs().powi(3);
        }
        bias.push(bound / 6.0 * worst);
    }
    let smoothing_bias_bound = bound.is_finite().then(|| bias.clone());
    if let Some(draws) = posterior_draws {
        let posterior = posterior_by_cluster_weights(g_count, k, draws, ctx, |weights| {
            grid.iter()
                .map(|&at| {
                    local_smoother(prepared, bandwidth, at, derivative, Some(weights))
                        .map(|smoother| smoother.value)
                })
                .collect()
        })
        .map(|mut summary| {
            summary.method = "bayesian_bootstrap_local_quadratic_projection";
            summary
        })?;
        return Ok(DoseFit {
            coordinates,
            sampling: None,
            posterior: Some(posterior),
            local_ess,
            smoothing_bias_bound,
            pairs: prepared.dose.len(),
            clusters: g_count,
        });
    }
    let factor = count(g_count) / (count(g_count) - 1.0);
    let standard_error: Vec<f64> = (0..k)
        .map(|j| (factor * (0..g_count).map(|g| residuals[g * k + j].powi(2)).sum::<f64>()).sqrt())
        .collect();
    let mut covariance = vec![0.0; k * k];
    for g in 0..g_count {
        for a in 0..k {
            for b in 0..k {
                covariance[a * k + b] += factor * residuals[g * k + a] * residuals[g * k + b];
            }
        }
    }
    let mut lower = Vec::with_capacity(k);
    let mut upper = Vec::with_capacity(k);
    for j in 0..k {
        let se = standard_error[j];
        let critical = if se > 0.0 {
            folded_normal_critical(bias[j] / se, NOMINAL_LEVEL)
        } else {
            antecedent_stats::NORMAL_Q975
        };
        let half = if se > 0.0 { critical * se } else { bias[j] };
        lower.push(estimate[j] - half);
        upper.push(estimate[j] + half);
    }
    let mut summary = SamplingSummary {
        estimate,
        standard_error: standard_error.clone(),
        lower,
        upper,
        covariance,
        band: None,
        holm_adjusted_p: None,
        smoothing_bias_bound: Some(bias),
        method: "cluster_robust_local_quadratic_bias_aware",
    };
    if k > 1 {
        let band: Vec<usize> = (0..k).collect();
        let critical = sup_t_critical(
            &residuals,
            k,
            factor.sqrt(),
            &standard_error,
            &band,
            band_replicates,
            ctx,
        )?;
        attach_band(&mut summary, &band, critical, band_replicates);
    }
    Ok(DoseFit {
        coordinates,
        sampling: Some(summary),
        posterior: None,
        local_ess,
        smoothing_bias_bound,
        pairs: prepared.dose.len(),
        clusters: g_count,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{PanelRows, lagged_pairs};
    use super::*;

    fn uniform(seed: u64) -> f64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2545_F491_4F6C_DD1D;
        state ^= state >> 31;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 29;
        (state >> 11) as f64 / (1_u64 << 53) as f64
    }

    type DosePanel = (Vec<usize>, Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>);

    /// Uniform(0, 2) doses (density 0.5), outcome `1 + d + 0.5 d^2` plus noise at the same
    /// period.
    fn panel(units: u64, periods: u64) -> DosePanel {
        let (mut unit, mut period, mut dose, mut density, mut outcome) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..units {
            for t in 0..periods {
                let d = 2.0 * uniform(i * 977 + t);
                let noise = uniform(i * 977 + t + 1_000_003) - 0.5;
                unit.push(usize::try_from(i).unwrap());
                period.push(i64::try_from(t).unwrap());
                dose.push(d);
                density.push(0.5);
                outcome.push(1.0 + d + 0.5 * d * d + noise);
            }
        }
        (unit, period, dose, density, outcome)
    }

    #[test]
    fn the_bias_aware_local_quadratic_recovers_the_quadratic_curve_and_band() {
        let (unit, period, dose, density, outcome) = panel(300, 8);
        let pairs = lagged_pairs(PanelRows { unit: &unit, period: &period }, 0).unwrap();
        let input = DosePairs {
            pairs: &pairs,
            outcome: &outcome,
            dose: &dose,
            density: &density,
            cluster: &unit,
        };
        let grid = [0.5, 1.0, 1.5];
        let target = DoseTarget::LocalQuadratic {
            grid: grid.to_vec(),
            derivative: false,
            curvature_bound: Some(0.0),
        };
        let ctx = ExecutionContext::for_tests(7);
        let fit = fit_dose(input, 0.3, &target, None, 500, &ctx).unwrap();
        let summary = fit.sampling.unwrap();
        for (j, &d) in grid.iter().enumerate() {
            let truth = 1.0 + d + 0.5 * d * d;
            assert!((summary.estimate[j] - truth).abs() < 4.0 * summary.standard_error[j]);
            assert!(summary.lower[j] <= summary.estimate[j]);
        }
        let band = summary.band.unwrap();
        assert!(band.critical_value > antecedent_stats::NORMAL_Q975);
        let missing = DoseTarget::LocalQuadratic {
            grid: grid.to_vec(),
            derivative: false,
            curvature_bound: None,
        };
        assert!(fit_dose(input, 0.3, &missing, None, 500, &ctx).is_err());
    }

    #[test]
    fn the_smoothed_contrast_matches_the_kernel_smoothed_truth() {
        let (unit, period, dose, density, outcome) = panel(300, 8);
        let pairs = lagged_pairs(PanelRows { unit: &unit, period: &period }, 0).unwrap();
        let input = DosePairs {
            pairs: &pairs,
            outcome: &outcome,
            dose: &dose,
            density: &density,
            cluster: &unit,
        };
        let target = DoseTarget::SmoothedContrast { active: 1.2, control: 0.8 };
        let ctx = ExecutionContext::for_tests(9);
        let fit = fit_dose(input, 0.1, &target, None, 500, &ctx).unwrap();
        let summary = fit.sampling.unwrap();
        // With h = 0.1 well inside (0, 2) the smoothed curve is m(d) + 0.5 h^2 m''(d).
        let smoothed = |d: f64| 1.0 + d + 0.5 * d * d + 0.5 * 0.01;
        let truth = smoothed(1.2) - smoothed(0.8);
        assert!((summary.estimate[0] - truth).abs() < 4.0 * summary.standard_error[0]);
        let posterior = fit_dose(input, 0.1, &target, Some(300), 0, &ctx).unwrap();
        let draws = posterior.posterior.unwrap();
        assert!(
            (draws.mean[0] - summary.estimate[0]).abs() < 0.5 * summary.standard_error[0] + 0.05
        );
    }

    #[test]
    fn the_folded_normal_critical_value_widens_with_bias() {
        let unbiased = folded_normal_critical(0.0, 0.95);
        assert!((unbiased - antecedent_stats::NORMAL_Q975).abs() < 1e-6);
        assert!(folded_normal_critical(1.0, 0.95) > unbiased);
        assert!(folded_normal_critical(5.0, 0.95) - 5.0 < antecedent_stats::NORMAL_Q975);
    }
}
