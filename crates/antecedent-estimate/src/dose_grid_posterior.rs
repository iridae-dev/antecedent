//! Joint Bayesian posterior over a randomized-dose response grid.
//!
//! # The row
//!
//! * **Target.** The dose response `m(d) = E[Y(d)]` of a randomized continuous dose, read
//!   through one named functional exactly as in [`crate::dose_grid_functional`]:
//!   [`DoseFunctional::Level`] (`m(d)` at each grid dose), [`DoseFunctional::Derivative`]
//!   (`m'(d)`) or [`DoseFunctional::Contrast`] (`m(to) - m(from)`). Unlike the fixed-bandwidth
//!   smoother, the target is `m` itself: no bandwidth is part of the estimand and no smoothing
//!   bias is added to it.
//! * **Likelihood.** `y_i = m(d_i) + e_i`, `e_i ~ N(0, sigma^2)`, independent rows, with the
//!   dose independent of the potential-outcome process (a caller attestation, as in the
//!   frequentist row; an observational dose is refused here too).
//! * **Prior.** `m(d) = sum_j beta_j z^j` is a polynomial in `z = (2d - lo - hi) / (hi - lo)`,
//!   the dose rescaled to `[-1, 1]` on the declared support. The coefficients carry
//!   `beta_j | sigma^2 ~ N(0, sigma^2 s_j^2)`: the scale vector `s` is the smoothing prior
//!   (a decaying `s_j` shrinks the higher-order, rougher terms). A smoothing parameter alone
//!   is not a posterior; the vector `s`, the degree and the residual prior are the declared
//!   prior and are reported with the request. The residual variance is known or has an
//!   inverse-gamma prior.
//! * **Joint posterior.** One coefficient draw gives every grid dose at once, so draw `k` is
//!   the same posterior sample at every grid point and the covariance across the grid is
//!   preserved. Levels and derivatives are evaluated for all draws by
//!   [`crate::tiled_prediction`] (the polynomial encoding is formed once per grid point), and
//!   the joint covariance is its [`Reduction::DrawCovariance`]. Posterior means alone are
//!   not the predictive law of a new outcome: this is a posterior for `m`, not for `Y`.
//! * **Support.** A dose outside the observed dose range is refused, never extrapolated.
//!
//! With a known residual variance the posterior mean and covariance of every functional are
//! also reported in closed form, from the exact Gaussian posterior of the coefficients.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{QuantileRule, equal_tail_interval_sorted};

use crate::dose_basis_regression::{BasisPosterior, ResidualPrior, fit_basis_regression};
use crate::dose_grid_functional::{DoseDesign, DoseFunctional, refuse};
use crate::error::EstimationError;
use crate::tiled_prediction::{
    AxisOrder, Encoding, GridPoint, OutcomeDraws, Reduction, StreamedValue, TileRequest,
    stream_tile,
};

/// Reference grid size. Not enforced: grid tiles are streamed under the context's memory
/// admission.
pub const MAX_POSTERIOR_GRID: usize = 64;

/// One request: rows, the declared polynomial prior, and the named functional.
#[derive(Clone, Copy, Debug)]
pub struct DoseGridPosteriorRequest<'a> {
    /// Observed doses.
    pub dose: &'a [f64],
    /// Observed outcomes, aligned with `dose`.
    pub outcome: &'a [f64],
    /// Grid doses for a level or derivative; ignored for a contrast.
    pub grid: &'a [f64],
    /// Declared dose support `(lo, hi)` the polynomial is rescaled on.
    pub support: (f64, f64),
    /// Prior scale `s_j` of every coefficient `j = 0..=degree`.
    pub coefficient_scales: &'a [f64],
    /// Prior of the residual variance.
    pub residual: ResidualPrior,
    /// Named functional.
    pub functional: DoseFunctional,
    /// Design attestation.
    pub design: DoseDesign,
    /// Posterior draws.
    pub draws: usize,
    /// Seed of the draws.
    pub seed: u64,
}

/// Joint posterior of one functional over its evaluation points.
#[derive(Clone, Debug, PartialEq)]
pub struct DoseGridPosterior {
    retention: DoseOutputRetention,
    /// Requested functional.
    pub functional: DoseFunctional,
    /// Evaluation doses: the grid, or `[from, to]` for a contrast.
    pub points: Vec<f64>,
    /// Number of functional values: the grid size, or one for a contrast.
    pub values: usize,
    /// Posterior mean of each value, from the draws.
    pub mean: Vec<f64>,
    /// Pointwise equal-tailed 95% credible interval of each value.
    pub interval_95: Vec<[f64; 2]>,
    /// Joint covariance of the values across the draws, row-major `values x values`.
    pub covariance: Vec<f64>,
    /// Exact posterior mean of each value (known residual variance only).
    pub exact_mean: Option<Vec<f64>>,
    /// Exact posterior covariance, row-major `values x values` (known residual variance only).
    pub exact_covariance: Option<Vec<f64>>,
    /// Draw-major functional draws: draw `k` occupies `[k * values, (k + 1) * values)`.
    pub draws: Vec<f64>,
    /// Number of draws.
    pub n_draws: usize,
}

#[derive(Clone, Debug, Default)]
struct DoseOutputRetention(Option<std::sync::Arc<antecedent_core::execution::AdmissionLease>>);
impl PartialEq for DoseOutputRetention {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl DoseGridPosterior {
    /// Shared reservations funding the retained numerical output, not process resident memory.
    #[must_use]
    pub fn retained_admitted_bytes(&self) -> u64 {
        self.retention.0.as_ref().map_or(0, |lease| lease.remainder().bytes())
    }
}

/// Posterior of the requested dose-response functional under the declared polynomial prior.
///
/// # Errors
///
/// An observational dose design, misaligned or non-finite rows, a dose outside the declared
/// support, a request point outside the observed dose range, an invalid prior, a derivative
/// of a degree-zero curve, or a draw count outside `200..=100000`.
pub fn estimate_dose_grid_posterior(
    request: &DoseGridPosteriorRequest<'_>,
) -> Result<DoseGridPosterior, EstimationError> {
    let points = validate(request)?;
    let posterior = fit_dose_basis(request)?;
    evaluate_frozen_basis(request, points, &posterior)
}

fn fit_dose_basis(
    request: &DoseGridPosteriorRequest<'_>,
) -> Result<BasisPosterior, EstimationError> {
    let (lo, hi) = request.support;
    let width = request.coefficient_scales.len();
    let rows = request.dose.len();
    let mut design = vec![0.0; rows * width];
    for (i, dose) in request.dose.iter().enumerate() {
        if antecedent_core::numerical_admission::is_cancelled() {
            return Err(execution_refusal(
                "dose_grid.cancelled",
                "dose design construction cancelled",
            ));
        }
        let z = rescale(*dose, lo, hi);
        let mut power = 1.0;
        for j in 0..width {
            design[j * rows + i] = power;
            power *= z;
        }
    }
    fit_basis_regression(
        &design,
        rows,
        request.outcome,
        request.coefficient_scales,
        request.residual,
        request.draws,
        request.seed,
    )
}

fn evaluate_frozen_basis(
    request: &DoseGridPosteriorRequest<'_>,
    points: Vec<f64>,
    posterior: &BasisPosterior,
) -> Result<DoseGridPosterior, EstimationError> {
    let (lo, hi) = request.support;
    let width = request.coefficient_scales.len();
    let derivative = matches!(request.functional, DoseFunctional::Derivative);
    let (coefficients, degree) = if derivative {
        (derivative_coefficients(&posterior.coefficients, width, lo, hi), width - 2)
    } else {
        (posterior.coefficients.clone(), width - 1)
    };
    let z_points: Vec<f64> = points.iter().map(|d| rescale(*d, lo, hi)).collect();
    let levels = tile_levels(&coefficients, request.draws, degree, &z_points)?;
    let contrast = matches!(request.functional, DoseFunctional::Contrast { .. });
    let (values, draws) = if contrast {
        let diff: Vec<f64> =
            (0..request.draws).map(|k| levels[k * 2 + 1] - levels[k * 2]).collect();
        (1, diff)
    } else {
        (points.len(), levels)
    };
    let covariance = if contrast {
        let mean = draws.iter().sum::<f64>() / request.draws as f64;
        vec![draws.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / request.draws as f64]
    } else {
        tile_covariance(&coefficients, request.draws, degree, &z_points)?
    };
    let mean: Vec<f64> = (0..values)
        .map(|g| {
            (0..request.draws).map(|k| draws[k * values + g]).sum::<f64>() / request.draws as f64
        })
        .collect();
    let interval_95 = (0..values)
        .map(|g| {
            let mut column: Vec<f64> = (0..request.draws).map(|k| draws[k * values + g]).collect();
            column.sort_by(f64::total_cmp);
            let (lower, upper) =
                equal_tail_interval_sorted(&column, 0.95, QuantileRule::ExchangeableRank);
            [lower, upper]
        })
        .collect();
    let rows_b = functional_rows(request.functional, &points, width, lo, hi);
    let (exact_mean, exact_covariance) = exact_moments(posterior, &rows_b);
    Ok(DoseGridPosterior {
        retention: DoseOutputRetention::default(),
        functional: request.functional,
        points,
        values,
        mean,
        interval_95,
        covariance,
        exact_mean,
        exact_covariance,
        draws,
        n_draws: request.draws,
    })
}

fn rescale(dose: f64, lo: f64, hi: f64) -> f64 {
    (2.0 * dose - lo - hi) / (hi - lo)
}

fn validate(request: &DoseGridPosteriorRequest<'_>) -> Result<Vec<f64>, EstimationError> {
    let invalid = |message: &str| {
        refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "dose_grid.invalid_request",
            message,
        )
    };
    if request.design == DoseDesign::Observational {
        return Err(refuse(
            antecedent_core::reason_code!("effect_not_identified"),
            "dose_grid.graph_not_certified",
            "this row does not adjust; it requires a randomized dose",
        ));
    }
    let n = request.dose.len();
    let (lo, hi) = request.support;
    if n < 3
        || request.outcome.len() != n
        || request.dose.iter().chain(request.outcome).any(|v| !v.is_finite())
        || !lo.is_finite()
        || !hi.is_finite()
        || lo >= hi
        || request.dose.iter().any(|d| *d < lo || *d > hi)
    {
        return Err(invalid(
            "rows must be aligned, finite, at least three, and inside the declared support",
        ));
    }
    if request.coefficient_scales.len() < 2 && request.functional == DoseFunctional::Derivative {
        return Err(invalid("a derivative needs a polynomial of degree at least one"));
    }
    let points: Vec<f64> = match request.functional {
        DoseFunctional::Contrast { from, to } => vec![from, to],
        DoseFunctional::Level | DoseFunctional::Derivative => request.grid.to_vec(),
    };
    if points.is_empty() || points.iter().any(|v| !v.is_finite()) {
        return Err(invalid("the requested doses must be at least one finite value"));
    }
    let (minimum, maximum) = request
        .dose
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), d| (a.min(*d), b.max(*d)));
    if let Some(at) = points.iter().find(|d| **d < minimum || **d > maximum) {
        return Err(refuse(
            antecedent_core::reason_code!("cell_not_licensed"),
            "dose_grid.unsupported_dose",
            &format!("dose {at} is outside the observed dose support"),
        ));
    }
    crate::dose_basis_regression::checked_basis_prior(
        request.coefficient_scales,
        request.residual,
        request.draws,
    )?;
    Ok(points)
}

/// `d/dd sum_j beta_j z^j = (2 / (hi - lo)) sum_j j beta_j z^(j-1)`, draw by draw.
fn derivative_coefficients(coefficients: &[f64], width: usize, lo: f64, hi: f64) -> Vec<f64> {
    let scale = 2.0 / (hi - lo);
    let mut out = Vec::with_capacity(coefficients.len() / width * (width - 1));
    for draw in coefficients.chunks(width) {
        for (j, beta) in draw.iter().enumerate().skip(1) {
            #[allow(clippy::cast_precision_loss, reason = "the degree is at most 63")]
            out.push(scale * j as f64 * beta);
        }
    }
    out
}

fn grid_points(z_points: &[f64]) -> Vec<GridPoint> {
    z_points
        .iter()
        .enumerate()
        .map(|(id, z)| GridPoint { id: id as u64, coordinate: *z, factor: 1.0 })
        .collect()
}

fn tile_request<'a>(
    failed: &'a [bool],
    grid: &'a [GridPoint],
    outcomes: &'a [OutcomeDraws<'a>],
    degree: usize,
    reduction: Reduction,
) -> TileRequest<'a> {
    TileRequest {
        row_features: &[],
        row_start: 0,
        row_count: 1,
        row_width: 0,
        outcomes,
        failed_draws: failed,
        draw_start: 0,
        draw_count: failed.len() as u64,
        grid,
        grid_start: 0,
        grid_count: grid.len() as u64,
        encoding: Encoding::Polynomial { degree: degree as u64 },
        order: AxisOrder::row_major(),
        reduction,
    }
}

fn tile_levels(
    coefficients: &[f64],
    draws: usize,
    degree: usize,
    z_points: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    let failed = vec![false; draws];
    let grid = grid_points(z_points);
    let outcomes = [OutcomeDraws { id: 0, coefficients }];
    let request = tile_request(&failed, &grid, &outcomes, degree, Reduction::Full);
    let mut out = vec![0.0; draws * z_points.len()];
    stream_tile(&request, |value| {
        if let StreamedValue::Full { draw, grid_id, value, .. } = value {
            if let (Ok(k), Ok(g)) = (usize::try_from(draw), usize::try_from(grid_id)) {
                if let Some(slot) = out.get_mut(k * z_points.len() + g) {
                    *slot = value;
                }
            }
        }
    })?;
    Ok(out)
}

fn tile_covariance(
    coefficients: &[f64],
    draws: usize,
    degree: usize,
    z_points: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    let failed = vec![false; draws];
    let grid = grid_points(z_points);
    let outcomes = [OutcomeDraws { id: 0, coefficients }];
    let request = tile_request(&failed, &grid, &outcomes, degree, Reduction::DrawCovariance);
    let size = z_points.len();
    let mut out = vec![0.0; size * size];
    stream_tile(&request, |value| {
        if let StreamedValue::DrawCovariance { left_grid_id, right_grid_id, value, .. } = value {
            if let (Ok(i), Ok(j)) = (usize::try_from(left_grid_id), usize::try_from(right_grid_id))
            {
                if let Some(slot) = out.get_mut(i * size + j) {
                    *slot = value;
                }
            }
        }
    })?;
    Ok(out)
}

/// Rows `B` of the linear map from coefficients to functional values.
fn functional_rows(
    functional: DoseFunctional,
    points: &[f64],
    width: usize,
    lo: f64,
    hi: f64,
) -> Vec<Vec<f64>> {
    let level = |dose: f64| -> Vec<f64> {
        let z = rescale(dose, lo, hi);
        let mut power = 1.0;
        (0..width)
            .map(|_| {
                let value = power;
                power *= z;
                value
            })
            .collect()
    };
    let slope = |dose: f64| -> Vec<f64> {
        let z = rescale(dose, lo, hi);
        let scale = 2.0 / (hi - lo);
        (0..width)
            .map(|j| {
                if j == 0 {
                    0.0
                } else {
                    #[allow(clippy::cast_precision_loss, reason = "the degree is at most 63")]
                    let order = j as f64;
                    scale * order * z.powi(i32::try_from(j - 1).unwrap_or(0))
                }
            })
            .collect()
    };
    match functional {
        DoseFunctional::Level => points.iter().map(|d| level(*d)).collect(),
        DoseFunctional::Derivative => points.iter().map(|d| slope(*d)).collect(),
        DoseFunctional::Contrast { .. } => {
            let to = level(points[1]);
            let from = level(points[0]);
            vec![to.iter().zip(&from).map(|(a, b)| a - b).collect()]
        }
    }
}

/// `B m` and `B S B'` from the exact Gaussian posterior of the coefficients.
fn exact_moments(
    posterior: &BasisPosterior,
    rows: &[Vec<f64>],
) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
    let Some(covariance) = posterior.covariance.as_ref() else {
        return (None, None);
    };
    let w = posterior.width;
    let mean = rows
        .iter()
        .map(|row| row.iter().zip(&posterior.mean).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    let map = |row: &[f64]| -> Vec<f64> {
        (0..w).map(|i| (0..w).map(|j| covariance[i * w + j] * row[j]).sum::<f64>()).collect()
    };
    let mut out = Vec::with_capacity(rows.len() * rows.len());
    for left in rows {
        let spread = map(left);
        for right in rows {
            out.push(spread.iter().zip(right).map(|(a, b)| a * b).sum::<f64>());
        }
    }
    (Some(mean), Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOSE: [f64; 10] = [0.1, 0.5, 0.9, 1.4, 1.8, 2.2, 2.7, 3.1, 3.5, 3.9];
    const OUTCOME: [f64; 10] = [1.2, 1.9, 2.1, 3.0, 3.2, 3.1, 3.0, 2.6, 2.2, 1.5];
    const SCALES: [f64; 3] = [3.0, 2.0, 1.0];

    fn request<'a>(
        functional: DoseFunctional,
        grid: &'a [f64],
        dose: &'a [f64],
        outcome: &'a [f64],
    ) -> DoseGridPosteriorRequest<'a> {
        DoseGridPosteriorRequest {
            dose,
            outcome,
            grid,
            support: (0.0, 4.0),
            coefficient_scales: &SCALES,
            residual: ResidualPrior::KnownVariance(0.25),
            functional,
            design: DoseDesign::RandomizedDose,
            draws: 40_000,
            seed: 21,
        }
    }

    fn close(left: f64, right: f64, tolerance: f64) {
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }

    #[test]
    fn level_posterior_matches_the_conjugate_reference_and_keeps_the_covariance() {
        // Reference mean and covariance from numpy: solve (V0^-1 + X'X) m = X'y on the
        // rescaled polynomial design and form B C B' for the level rows at 0.5, 2.0, 3.5.
        let grid = [0.5, 2.0, 3.5];
        let posterior =
            estimate_dose_grid_posterior(&request(DoseFunctional::Level, &grid, &DOSE, &OUTCOME))
                .unwrap();
        let mean = posterior.exact_mean.as_ref().unwrap();
        for (got, want) in mean.iter().zip([2.018_449_65, 2.714_270_59, 2.318_891_91]) {
            close(*got, want, 1e-7);
        }
        let exact = posterior.exact_covariance.as_ref().unwrap();
        let reference = [
            0.064_536_70,
            0.016_412_15,
            -0.005_911_84,
            0.016_412_15,
            0.041_541_01,
            0.015_904_12,
            -0.005_911_84,
            0.015_904_12,
            0.063_660_80,
        ];
        for (got, want) in exact.iter().zip(reference) {
            close(*got, want, 1e-7);
        }
        // The draws reproduce the exact joint moments, off-diagonals included: independent
        // per-point draws would leave these covariances near zero.
        for g in 0..3 {
            close(posterior.mean[g], mean[g], 0.01);
            for h in 0..3 {
                let scale = (exact[g * 3 + g] * exact[h * 3 + h]).sqrt();
                close(posterior.covariance[g * 3 + h], exact[g * 3 + h], 0.05 * scale);
            }
        }
        assert!(posterior.covariance[1] > 0.01 && posterior.covariance[5] > 0.01);
        for (interval, value) in posterior.interval_95.iter().zip(&posterior.mean) {
            assert!(interval[0] < *value && *value < interval[1]);
        }
    }

    #[test]
    fn derivative_and_contrast_posteriors_use_the_same_joint_draws() {
        let grid = [0.5, 2.0, 3.5];
        let derivative = estimate_dose_grid_posterior(&request(
            DoseFunctional::Derivative,
            &grid,
            &DOSE,
            &OUTCOME,
        ))
        .unwrap();
        let want = [0.827_613_84, 0.100_147_42, -0.627_319_00];
        for (got, want) in derivative.exact_mean.as_ref().unwrap().iter().zip(want) {
            close(*got, want, 1e-7);
        }
        for (got, want) in derivative.mean.iter().zip(want) {
            close(*got, want, 0.01);
        }
        close(derivative.covariance[0], 0.083_616_85, 0.004);
        close(derivative.covariance[2], -0.052_563_33, 0.004);

        let contrast = estimate_dose_grid_posterior(&request(
            DoseFunctional::Contrast { from: 1.0, to: 3.0 },
            &[],
            &DOSE,
            &OUTCOME,
        ))
        .unwrap();
        assert_eq!(contrast.values, 1);
        close(contrast.exact_mean.as_ref().unwrap()[0], 0.200_294_84, 1e-7);
        close(contrast.exact_covariance.as_ref().unwrap()[0], 0.062_231_64, 1e-7);
        close(contrast.mean[0], 0.200_294_84, 0.01);
        // The contrast variance is the joint one, 0.0622, not the 0.0842 that independent
        // draws at the two doses would give.
        close(contrast.covariance[0], 0.062_231_64, 0.004);
        assert!(contrast.covariance[0] < 0.07);
    }

    #[test]
    fn unknown_variance_posterior_recovers_a_noisy_quadratic_truth() {
        // y = 1 + 2d - 0.3 d^2 plus deterministic noise of standard deviation about 0.1.
        let n = 200;
        let dose: Vec<f64> = (0..n).map(|i| 4.0 * (f64::from(i) + 0.5) / f64::from(n)).collect();
        let outcome: Vec<f64> = dose
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let noise = crate::splitmix::keyed_unit(3, i as u64) - 0.5;
                1.0 + 2.0 * d - 0.3 * d * d + 0.35 * noise
            })
            .collect();
        let grid = [1.0, 2.0, 3.0];
        let mut level = request(DoseFunctional::Level, &grid, &dose, &outcome);
        level.residual = ResidualPrior::InverseGamma { shape: 2.0, scale: 0.1 };
        level.draws = 4000;
        let posterior = estimate_dose_grid_posterior(&level).unwrap();
        assert!(posterior.exact_mean.is_none());
        for (g, d) in grid.iter().enumerate() {
            let truth = 1.0 + 2.0 * d - 0.3 * d * d;
            close(posterior.mean[g], truth, 0.05);
            let [lower, upper] = posterior.interval_95[g];
            assert!(lower < posterior.mean[g] && posterior.mean[g] < upper && upper - lower < 0.15);
        }
        let mut slope = request(DoseFunctional::Derivative, &grid, &dose, &outcome);
        slope.residual = level.residual;
        slope.draws = 4000;
        let derivative = estimate_dose_grid_posterior(&slope).unwrap();
        for (g, d) in grid.iter().enumerate() {
            close(derivative.mean[g], 2.0 - 0.6 * d, 0.1);
        }
    }

    #[test]
    fn unlicensed_requests_are_refused() {
        let grid = [0.5];
        let mut observational = request(DoseFunctional::Level, &grid, &DOSE, &OUTCOME);
        observational.design = DoseDesign::Observational;
        let refused = estimate_dose_grid_posterior(&observational).unwrap_err();
        assert!(refused.to_string().contains("dose_grid.graph_not_certified"));

        let outside = [0.01];
        let extrapolated = request(DoseFunctional::Level, &outside, &DOSE, &OUTCOME);
        let refused = estimate_dose_grid_posterior(&extrapolated).unwrap_err();
        assert!(refused.to_string().contains("dose_grid.unsupported_dose"));

        let mut flat = request(DoseFunctional::Derivative, &grid, &DOSE, &OUTCOME);
        flat.coefficient_scales = &[1.0];
        assert!(estimate_dose_grid_posterior(&flat).is_err());

        let mut thin = request(DoseFunctional::Level, &grid, &DOSE, &OUTCOME);
        thin.draws = 10;
        assert!(estimate_dose_grid_posterior(&thin).is_err());
    }
}

type SharedDoseBasis = std::sync::Arc<
    std::sync::Mutex<Option<(antecedent_core::KernelPolicy, std::sync::Arc<FrozenDoseBasis>)>>,
>;

/// Typed, checked randomized-dose posterior program. Compilation performs no fit or simulation.
#[derive(Clone, Debug)]
pub struct CompiledDoseGridPosterior {
    dose: std::sync::Arc<[f64]>,
    outcome: std::sync::Arc<[f64]>,
    grid: std::sync::Arc<[f64]>,
    scales: std::sync::Arc<[f64]>,
    support: (f64, f64),
    residual: ResidualPrior,
    functional: DoseFunctional,
    draws: usize,
    seed: u64,
    source_bytes: u64,
    basis: SharedDoseBasis,
    source_retention: std::sync::Arc<antecedent_core::execution::AdmissionLease>,
    grid_retention: std::sync::Arc<antecedent_core::execution::AdmissionLease>,
}
#[derive(Debug)]
struct FrozenDoseBasis {
    posterior: BasisPosterior,
    retention: antecedent_core::execution::AdmissionLease,
}
fn execution_refusal(detail: &'static str, message: &str) -> EstimationError {
    refuse(antecedent_core::reason_code!("transport_budget_cancel"), detail, message)
}
impl CompiledDoseGridPosterior {
    /// Validate the scientific target, randomized-design premise, support and prior; retain inputs.
    pub fn compile(
        request: &DoseGridPosteriorRequest<'_>,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, EstimationError> {
        validate(request)?;
        if ctx.cancellation.is_cancelled() {
            return Err(execution_refusal("dose_grid.cancelled", "compilation cancelled"));
        }
        let cells = request
            .dose
            .len()
            .checked_add(request.outcome.len())
            .and_then(|n| n.checked_add(request.grid.len()))
            .and_then(|n| n.checked_add(request.coefficient_scales.len()))
            .ok_or_else(|| {
                execution_refusal("dose_grid.resource_refused", "input size overflow")
            })?;
        let bytes = u64::try_from(cells).ok().and_then(|n| n.checked_mul(8)).ok_or_else(|| {
            execution_refusal("dose_grid.resource_refused", "input byte size overflow")
        })?;
        if ctx.memory_limit_bytes().is_some_and(|limit| bytes > limit) {
            return Err(execution_refusal(
                "dose_grid.resource_refused",
                "retained inputs exceed memory limit",
            ));
        }
        let grid_bytes =
            u64::try_from(request.grid.len()).ok().and_then(|n| n.checked_mul(8)).ok_or_else(
                || execution_refusal("dose_grid.resource_refused", "grid byte size overflow"),
            )?;
        let source_bytes = bytes.checked_sub(grid_bytes).ok_or_else(|| {
            execution_refusal("dose_grid.resource_refused", "input byte size overflow")
        })?;
        let retained = ctx
            .acquire_admission(0, source_bytes)
            .map_err(|e| execution_refusal("dose_grid.resource_refused", &e.to_string()))?;
        let grid_retention = ctx
            .acquire_admission(0, grid_bytes)
            .map_err(|e| execution_refusal("dose_grid.resource_refused", &e.to_string()))?;
        Ok(Self {
            dose: request.dose.into(),
            outcome: request.outcome.into(),
            grid: request.grid.into(),
            scales: request.coefficient_scales.into(),
            support: request.support,
            residual: request.residual,
            functional: request.functional,
            draws: request.draws,
            seed: request.seed,
            source_bytes: bytes,
            basis: std::sync::Arc::default(),
            grid_retention: std::sync::Arc::new(grid_retention),
            source_retention: std::sync::Arc::new(retained),
        })
    }
    /// Exact numeric input bytes retained by this program.
    #[must_use]
    pub fn retained_source_bytes(&self) -> u64 {
        self.source_bytes
    }
    fn request(&self) -> DoseGridPosteriorRequest<'_> {
        DoseGridPosteriorRequest {
            dose: &self.dose,
            outcome: &self.outcome,
            grid: &self.grid,
            coefficient_scales: &self.scales,
            support: self.support,
            residual: self.residual,
            functional: self.functional,
            draws: self.draws,
            seed: self.seed,
            design: DoseDesign::RandomizedDose,
        }
    }
    /// Retarget grid/functionals without changing the frozen data, prior or draws.
    pub fn retarget(
        &self,
        grid: &[f64],
        functional: DoseFunctional,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, EstimationError> {
        if ctx.cancellation.is_cancelled() {
            return Err(execution_refusal("dose_grid.cancelled", "retargeting cancelled"));
        }
        let mut request = self.request();
        request.grid = grid;
        request.functional = functional;
        validate(&request)?;
        let bytes =
            u64::try_from(grid.len()).ok().and_then(|n| n.checked_mul(8)).ok_or_else(|| {
                execution_refusal("dose_grid.resource_refused", "grid byte size overflow")
            })?;
        let retained = ctx
            .acquire_admission(0, bytes)
            .map_err(|e| execution_refusal("dose_grid.resource_refused", &e.to_string()))?;
        let mut next = self.clone();
        next.source_bytes = self
            .source_bytes
            .checked_sub((self.grid.len() as u64) * 8)
            .and_then(|n| n.checked_add(bytes))
            .ok_or_else(|| {
                execution_refusal("dose_grid.resource_refused", "retargeted input size overflow")
            })?;
        next.grid = grid.into();
        next.functional = functional;
        next.grid_retention = std::sync::Arc::new(retained);
        Ok(next)
    }
    fn output_bytes(&self, points: usize) -> Result<u64, EstimationError> {
        let values =
            if matches!(self.functional, DoseFunctional::Contrast { .. }) { 1 } else { points };
        let cells = values
            .checked_mul(self.draws)
            .and_then(|n| n.checked_add(values.checked_mul(values)?.checked_mul(2)?))
            .and_then(|n| n.checked_add(values.checked_mul(4)?))
            .and_then(|n| n.checked_add(points))
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_add(std::mem::size_of::<DoseGridPosterior>()));
        u64::try_from(cells.ok_or_else(|| {
            execution_refusal("dose_grid.resource_refused", "output size overflow")
        })?)
        .map_err(|e| execution_refusal("dose_grid.resource_refused", &e.to_string()))
    }
    /// Execute the original conjugate engine and retained polynomial lowering.
    pub fn execute(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<DoseGridPosterior, EstimationError> {
        use antecedent_core::execution_attempt::{PhysicalEvent, record_physical};
        if ctx.cancellation.is_cancelled() {
            return Err(execution_refusal("dose_grid.cancelled", "execution cancelled"));
        }
        self.source_retention.observe_current();
        self.grid_retention.observe_current();
        let request = self.request();
        let points = validate(&request)?;
        let width = self.scales.len() as u64;
        let draws = self.draws as u64;
        let encoded = (self.dose.len() as u64).checked_mul(width);
        let coefficients = draws.checked_mul(width).and_then(|n| n.checked_mul(4));
        let prediction = draws.checked_mul(points.len() as u64).and_then(|n| n.checked_mul(4));
        let scratch = width.checked_mul(width).and_then(|n| n.checked_mul(16));
        let bytes = encoded
            .and_then(|n| n.checked_add(coefficients?))
            .and_then(|n| n.checked_add(prediction?))
            .and_then(|n| n.checked_add(scratch?))
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_add(self.output_bytes(points.len()).ok()?))
            .ok_or_else(|| {
                execution_refusal("dose_grid.resource_refused", "execution byte estimate overflow")
            })?;
        if ctx.memory_limit_bytes().is_some_and(|limit| bytes > limit) {
            return Err(execution_refusal(
                "dose_grid.resource_refused",
                "execution exceeds memory limit",
            ));
        }
        let reservation = std::sync::Arc::new(
            ctx.acquire_admission(0, bytes)
                .map_err(|e| execution_refusal("dose_grid.resource_refused", &e.to_string()))?,
        );
        antecedent_core::numerical_admission::with_context(ctx, || {
            antecedent_core::numerical_admission::with_reservation(
                std::sync::Arc::clone(&reservation),
                || {
                    let mut frozen = self.basis.lock().map_err(|_| {
                        execution_refusal(
                            "dose_grid.resource_refused",
                            "frozen basis lock poisoned",
                        )
                    })?;
                    let posterior = if let Some((_, fit)) =
                        frozen.as_ref().filter(|(kernel, _)| *kernel == ctx.kernel_policy)
                    {
                        record_physical(PhysicalEvent::CacheHit, 1);
                        record_physical(PhysicalEvent::ModelFit, 0);
                        {
                            fit.retention.observe_current();
                            std::sync::Arc::clone(fit)
                        }
                    } else {
                        let fitted = fit_dose_basis(&request)?;
                        let bytes = ((fitted.coefficients.len()
                            + fitted.mean.len()
                            + fitted.covariance.as_ref().map_or(0, Vec::len))
                            as u64)
                            .checked_mul(8)
                            .ok_or_else(|| {
                                execution_refusal(
                                    "dose_grid.resource_refused",
                                    "fit retention overflow",
                                )
                            })?;
                        let retention = reservation.acquire_child(0, bytes).map_err(|e| {
                            execution_refusal("dose_grid.resource_refused", &e.to_string())
                        })?;
                        record_physical(PhysicalEvent::RetainedBytes, bytes);
                        let posterior =
                            std::sync::Arc::new(FrozenDoseBasis { posterior: fitted, retention });
                        *frozen = Some((ctx.kernel_policy, std::sync::Arc::clone(&posterior)));
                        posterior
                    };
                    drop(frozen);
                    let output_bytes = self.output_bytes(points.len())?;
                    let output_retention =
                        reservation.acquire_child(0, output_bytes).map_err(|e| {
                            execution_refusal("dose_grid.resource_refused", &e.to_string())
                        })?;
                    let mut result = evaluate_frozen_basis(&request, points, &posterior.posterior)?;
                    result.retention =
                        DoseOutputRetention(Some(std::sync::Arc::new(output_retention)));
                    record_physical(PhysicalEvent::RetainedBytes, output_bytes);
                    record_physical(PhysicalEvent::RetainedFull, 1);
                    if ctx.cancellation.is_cancelled() {
                        return Err(execution_refusal(
                            "dose_grid.cancelled",
                            "execution cancelled",
                        ));
                    }
                    Ok(result)
                },
            )
        })
    }
}

#[cfg(test)]
mod retained_resource_tests {
    use super::*;
    use antecedent_core::{ExecutionContext, execution::AdmissionPool};
    fn compiled(ctx: &ExecutionContext) -> CompiledDoseGridPosterior {
        CompiledDoseGridPosterior::compile(
            &DoseGridPosteriorRequest {
                dose: &[0., 0.25, 0.5, 0.75, 1.],
                outcome: &[0., 0.4, 1.1, 1.4, 2.],
                grid: &[0.25, 0.75],
                support: (0., 1.),
                coefficient_scales: &[2., 1.],
                residual: ResidualPrior::KnownVariance(0.5),
                functional: DoseFunctional::Level,
                design: DoseDesign::RandomizedDose,
                draws: 200,
                seed: 3,
            },
            ctx,
        )
        .unwrap()
    }
    #[test]
    fn dose_output_remains_admitted_after_compiler_and_cache_are_dropped() {
        let pool = AdmissionPool::fixed(1, 1_000_000);
        let ctx = ExecutionContext::for_tests(3).with_admission(pool.clone());
        let program = compiled(&ctx);
        let result = program.execute(&ctx).unwrap();
        assert!(result.retained_admitted_bytes() > 0);
        drop(program);
        assert_eq!(pool.byte_budget() - pool.available().bytes(), result.retained_admitted_bytes());
        let copy = result.clone();
        drop(result);
        assert!(pool.available().bytes() < pool.byte_budget());
        drop(copy);
        assert_eq!(pool.available().bytes(), pool.byte_budget());
    }
    #[test]
    fn dose_retarget_replaces_grid_reservation_instead_of_accumulating_history() {
        let pool = AdmissionPool::fixed(1, 1_000_000);
        let ctx = ExecutionContext::for_tests(3).with_admission(pool.clone());
        let mut program = compiled(&ctx);
        for _ in 0..100 {
            program = program.retarget(&[0.5], DoseFunctional::Level, &ctx).unwrap();
        }
        assert_eq!(pool.byte_budget() - pool.available().bytes(), program.retained_source_bytes());
        drop(program);
        assert_eq!(pool.available().bytes(), pool.byte_budget());
    }
    #[test]
    fn dose_replacing_policy_keeps_an_in_flight_frozen_fit_funded() {
        let pool = AdmissionPool::fixed(1, 1_000_000);
        let ctx = ExecutionContext::for_tests(3).with_admission(pool.clone());
        let program = compiled(&ctx);
        let earlier = program.execute(&ctx).unwrap();
        let in_flight = std::sync::Arc::clone(&program.basis.lock().unwrap().as_ref().unwrap().1);
        let old_bytes = in_flight.retention.remainder().bytes();
        let mut changed = ctx.clone();
        changed.kernel_policy = if ctx.kernel_policy == antecedent_core::KernelPolicy::scalar_only()
        {
            antecedent_core::KernelPolicy::default_policy()
        } else {
            antecedent_core::KernelPolicy::scalar_only()
        };
        let later = program.execute(&changed).unwrap();
        let available = pool.available().bytes();
        drop(in_flight);
        assert_eq!(pool.available().bytes(), available + old_bytes);
        let mut refused = ctx.clone();
        refused.set_memory_limit(Some(1));
        assert!(program.execute(&refused).is_err());
        assert_eq!(earlier.n_draws, later.n_draws);
        drop(program);
        drop(earlier);
        drop(later);
        assert_eq!(pool.available().bytes(), pool.byte_budget());
    }
}
