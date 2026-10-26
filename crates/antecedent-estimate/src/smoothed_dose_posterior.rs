//! Joint Bayesian posterior over a smoothed dose-response transport grid.
//!
//! # The cell
//!
//! The target is the one of [`crate::smoothed_dose`]: for each grid dose `a`, one declared
//! bandwidth `h` and the Epanechnikov kernel,
//!
//! ```text
//! psi_h(a) = E_target[ integral K_h(a - t) mu(t, X) dt ],   mu(t, x) = E(Y | X = x, A = t, S = 1),
//! ```
//!
//! read through the same direct or baseline-standardization certificate. The bandwidth and
//! the kernel are part of the estimand.
//!
//! # Likelihood and prior
//!
//! `y_i = mu(a_i, x_i) + e_i`, `e_i ~ N(0, sigma^2)`, independent source rows, with `mu` linear
//! in the declared dose-by-covariate basis ([`DoseBasis`]) and a Gaussian coefficient prior
//! `beta_j | sigma^2 ~ N(0, sigma^2 s_j^2)`: one scale for the intercept and one for every other
//! coefficient (the covariates enter the basis as given, so their units set the prior's
//! meaning). The residual variance is known or inverse-gamma. The ridge scale is the
//! smoothing prior of the fitted curve and is part of the answer.
//!
//! The randomized dose and the source-membership model are ancillary for a likelihood of
//! `Y` given `(A, X, S = 1)`, so the posterior does not use the known dose density or a
//! membership model; those are what the doubly robust frequentist score needs. The price is
//! that this is an outcome-model posterior: it is not robust to a misspecified basis, which
//! the frequentist cell's weighting term protects against. The target-covariate law is the
//! observed target sample, held fixed, exactly as in the frequentist plug-in term.
//!
//! # Computation
//!
//! The basis is linear in the covariates, so the target mean of a basis row equals the basis
//! row at the mean target covariates. The window integral of each basis function is a
//! Gauss-Legendre rule split at the declared knots, exact for the polynomial pieces up to the rule's degree. Each grid
//! dose then reads a fixed linear functional of the coefficients, and one coefficient draw
//! gives every grid dose at once: the covariance across the grid is preserved. With a known
//! residual variance the exact Gaussian mean and covariance are reported alongside.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::SmoothedDoseTransportQuery;
use antecedent_identify::{TransportFormula, TransportIdentification};
use antecedent_stats::special::gauss_legendre;
use antecedent_stats::{QuantileRule, equal_tail_interval_sorted};

use crate::dose_basis_regression::{ResidualPrior, fit_basis_regression};
use crate::error::EstimationError;
use crate::util::dot;
use crate::smoothed_dose::{
    DoseBasis, SmoothedDoseInput, check_grid_windows, outcome_design, split_rule,
};

/// Gauss-Legendre nodes per window piece.
const WINDOW_NODES: usize = 16;

/// The declared model of the source outcome curve and the posterior sample.
#[derive(Clone, Copy, Debug)]
pub struct SmoothedDosePosteriorModel<'a> {
    /// Dose-by-covariate basis of `mu(t, x)`.
    pub basis: &'a DoseBasis,
    /// Prior scale `s` of the intercept.
    pub intercept_scale: f64,
    /// Prior scale `s` of every other coefficient.
    pub coefficient_scale: f64,
    /// Prior of the residual variance.
    pub residual: ResidualPrior,
    /// Fewest source rows inside every grid window (at least two).
    pub min_local_rows: usize,
    /// Posterior draws.
    pub draws: usize,
    /// Seed of the draws.
    pub seed: u64,
}

/// Joint posterior of the smoothed dose response at the declared grid.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothedDosePosterior {
    /// Grid doses in the query's order.
    pub grid: Vec<f64>,
    /// Posterior mean of `psi_h` at each grid dose, from the draws.
    pub mean: Vec<f64>,
    /// Pointwise equal-tailed 95% credible interval at each grid dose.
    pub interval_95: Vec<[f64; 2]>,
    /// Joint covariance across the grid, row-major `grid x grid`, from the draws.
    pub covariance: Vec<f64>,
    /// Exact posterior mean (known residual variance only).
    pub exact_mean: Option<Vec<f64>>,
    /// Exact posterior covariance, row-major (known residual variance only).
    pub exact_covariance: Option<Vec<f64>>,
    /// Draw-major draws: draw `k` occupies `[k * grid, (k + 1) * grid)`.
    pub draws: Vec<f64>,
    /// Number of draws.
    pub n_draws: usize,
    /// Source rows used by the likelihood.
    pub source_rows: usize,
    /// Target rows whose covariate mean enters the functional.
    pub target_rows: usize,
    /// Fewest source rows in any grid window.
    pub minimum_local_rows: usize,
}

/// The covariates a transport certificate standardizes over, sorted.
///
/// # Errors
///
/// A certificate that is not a direct or baseline-standardization formula.
pub fn smoothed_dose_certified_features(
    id: &TransportIdentification,
) -> Result<Vec<u32>, EstimationError> {
    match id {
        TransportIdentification::Transportable { formula: TransportFormula::Direct(_), .. } => {
            Ok(Vec::new())
        }
        TransportIdentification::Transportable {
            formula: TransportFormula::Standardize { over, .. },
            ..
        } => {
            let mut features: Vec<u32> = over.iter().map(|v| v.raw()).collect();
            features.sort_unstable();
            Ok(features)
        }
        _ => Err(EstimationError::data_msg(
            "the smoothed dose posterior requires a direct or standardization certificate",
        )),
    }
}

/// Posterior of `psi_h(a)` over the query's grid under the declared outcome model.
///
/// # Errors
///
/// An uncertified identification, a malformed query, rows or basis, a grid window outside
/// the declared dose support, fewer than `min_local_rows` source rows in a window, no target
/// rows, or an invalid prior or draw count.
pub fn smoothed_dose_posterior(
    id: &TransportIdentification,
    query: &SmoothedDoseTransportQuery,
    input: &SmoothedDoseInput,
    model: &SmoothedDosePosteriorModel<'_>,
) -> Result<SmoothedDosePosterior, EstimationError> {
    let certified = smoothed_dose_certified_features(id)?;
    query.validate()?;
    check_grid_windows(query)?;
    let (source, target) = split_rows(&certified, query, input, model)?;
    let features = input.covariates.len();
    let basis = model.basis;
    let width = basis.width(features);
    let design = outcome_design(basis, &input.covariates, &source, |r| input.dose[source[r]]);
    let outcome: Vec<f64> = source.iter().map(|i| input.outcome[*i]).collect();
    let mut scales = vec![model.coefficient_scale; width];
    scales[0] = model.intercept_scale;
    let fit = fit_basis_regression(
        &design,
        source.len(),
        &outcome,
        &scales,
        model.residual,
        model.draws,
        model.seed,
    )?;
    let functionals = grid_functionals(query, input, model, &target);
    let minimum_local_rows = local_rows(query, input, &source, model.min_local_rows)?;
    let size = functionals.len();
    let _admitted = crate::util::admit_bound_array::<f64>(model.draws, size)?;
    let mut draws = vec![0.0; model.draws * size];
    for (k, coefficients) in fit.coefficients.chunks(width).enumerate() {
        for (g, row) in functionals.iter().enumerate() {
            draws[k * size + g] = dot(row, coefficients);
        }
    }
    #[allow(clippy::cast_precision_loss, reason = "draw counts are far below 2^52")]
    let count = model.draws as f64;
    let column = |g: usize| -> Vec<f64> { (0..model.draws).map(|k| draws[k * size + g]).collect() };
    let mean: Vec<f64> = (0..size).map(|g| column(g).iter().sum::<f64>() / count).collect();
    let mut covariance = vec![0.0; size * size];
    for g in 0..size {
        for h in 0..size {
            covariance[g * size + h] = (0..model.draws)
                .map(|k| (draws[k * size + g] - mean[g]) * (draws[k * size + h] - mean[h]))
                .sum::<f64>()
                / count;
        }
    }
    let interval_95 = (0..size)
        .map(|g| {
            let mut values = column(g);
            values.sort_by(f64::total_cmp);
            let (lower, upper) =
                equal_tail_interval_sorted(&values, 0.95, QuantileRule::ExchangeableRank);
            [lower, upper]
        })
        .collect();
    let (exact_mean, exact_covariance) = match fit.covariance.as_ref() {
        Some(posterior_covariance) => {
            let spread: Vec<Vec<f64>> = functionals
                .iter()
                .map(|row| {
                    (0..width)
                        .map(|i| {
                            (0..width).map(|j| posterior_covariance[i * width + j] * row[j]).sum()
                        })
                        .collect()
                })
                .collect();
            let exact_mean = functionals.iter().map(|row| dot(row, &fit.mean)).collect();
            let exact_covariance = spread
                .iter()
                .flat_map(|s| functionals.iter().map(move |row| dot(s, row)))
                .collect();
            (Some(exact_mean), Some(exact_covariance))
        }
        None => (None, None),
    };
    Ok(SmoothedDosePosterior {
        grid: query.grid.to_vec(),
        mean,
        interval_95,
        covariance,
        exact_mean,
        exact_covariance,
        draws,
        n_draws: model.draws,
        source_rows: source.len(),
        target_rows: target.len(),
        minimum_local_rows,
    })
}

/// Source and target row indices, after the structural checks of rows, basis and model.
fn split_rows(
    certified: &[u32],
    query: &SmoothedDoseTransportQuery,
    input: &SmoothedDoseInput,
    model: &SmoothedDosePosteriorModel<'_>,
) -> Result<(Vec<usize>, Vec<usize>), EstimationError> {
    let n = input.source.len();
    let (lo, hi) = query.dose_support;
    let basis = model.basis;
    let mut actual = input.features.clone();
    actual.sort_unstable();
    let invalid = |message: &str| EstimationError::data_msg(message.to_owned());
    if actual != certified
        || input.covariates.len() != actual.len()
        || n == 0
        || input.outcome.len() != n
        || input.dose.len() != n
        || input.covariates.iter().any(|col| col.len() != n || col.iter().any(|v| !v.is_finite()))
        || basis.degree == 0
        // The fixed rule integrates the basis times the degree-2 kernel exactly up to degree 2Q - 1.
        || usize::from(basis.degree) + 2 >= 2 * WINDOW_NODES
        || basis.knots.iter().any(|k| !k.is_finite() || *k <= lo || *k >= hi)
        || model.min_local_rows < 2
    {
        return Err(invalid(
            "invalid certified smoothed dose posterior input: rows, covariates, basis or support",
        ));
    }
    if !model.intercept_scale.is_finite()
        || model.intercept_scale <= 0.0
        || !model.coefficient_scale.is_finite()
        || model.coefficient_scale <= 0.0
    {
        return Err(invalid("the coefficient prior scales must be finite and positive"));
    }
    let source: Vec<usize> = (0..n).filter(|i| input.source[*i]).collect();
    let target: Vec<usize> = (0..n).filter(|i| !input.source[*i]).collect();
    if source.len() < 2 || target.is_empty() {
        return Err(invalid("the posterior needs source rows for the fit and target rows"));
    }
    for &i in &source {
        let (dose, outcome) = (input.dose[i], input.outcome[i]);
        if !dose.is_finite() || !outcome.is_finite() || dose < lo || dose > hi {
            return Err(invalid(
                "source doses and outcomes must be finite with doses inside the declared support",
            ));
        }
    }
    Ok((source, target))
}

/// Fewest source rows inside any grid window; refuses a window below the declared floor.
fn local_rows(
    query: &SmoothedDoseTransportQuery,
    input: &SmoothedDoseInput,
    source: &[usize],
    floor: usize,
) -> Result<usize, EstimationError> {
    let h = query.bandwidth;
    let mut minimum = usize::MAX;
    for &a in query.grid.iter() {
        let rows = source.iter().filter(|i| (input.dose[**i] - a).abs() < h).count();
        if rows < floor {
            return Err(EstimationError::data_msg(format!(
                "smoothed dose local support failure: {rows} source rows in the window of grid dose {a}, below {floor}"
            )));
        }
        minimum = minimum.min(rows);
    }
    Ok(minimum)
}

/// One coefficient functional per grid dose: the kernel-window average of every basis
/// function at the mean target covariates.
fn grid_functionals(
    query: &SmoothedDoseTransportQuery,
    input: &SmoothedDoseInput,
    model: &SmoothedDosePosteriorModel<'_>,
    target: &[usize],
) -> Vec<Vec<f64>> {
    let (nodes, weights) = gauss_legendre(WINDOW_NODES);
    #[allow(clippy::cast_precision_loss, reason = "row counts are far below 2^52")]
    let rows = target.len() as f64;
    let mean_covariates: Vec<Vec<f64>> = input
        .covariates
        .iter()
        .map(|col| vec![target.iter().map(|i| col[*i]).sum::<f64>() / rows])
        .collect();
    let width = model.basis.width(input.covariates.len());
    let h = query.bandwidth;
    query
        .grid
        .iter()
        .map(|&a| {
            let (u, w) =
                split_rule(&model.basis.knots, (a, h), (nodes.as_slice(), weights.as_slice()));
            let q = u.len();
            let design =
                outcome_design(model.basis, &mean_covariates, &vec![0; q], |r| a + h * u[r]);
            (0..width)
                .map(|j| {
                    (0..q)
                        .map(|r| w[r] * query.kernel.density(u[r]) * design[j * q + r])
                        .sum::<f64>()
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned_trial::TrialSampling;
    use antecedent_core::{SmoothingKernel, VariableId};
    use antecedent_identify::{NotCertifiedCertificate, PopulationFactor, TransportCertificate};
    use std::sync::Arc;

    fn input() -> SmoothedDoseInput {
        let n = 24;
        let index = |i: usize| i as f64;
        SmoothedDoseInput {
            features: vec![7],
            covariates: vec![(0..n).map(|i| (index(i) * 0.37).sin()).collect()],
            outcome: (0..n)
                .map(|i| {
                    let dose = 3.0 * (index(i) * 0.618_033_988_7).fract();
                    let x = (index(i) * 0.37).sin();
                    1.0 + 0.8 * dose - 0.2 * dose * dose
                        + 0.5 * x
                        + 0.3 * dose * x
                        + 0.1 * (7.0 * index(i)).sin()
                })
                .collect(),
            dose: (0..n).map(|i| 3.0 * (index(i) * 0.618_033_988_7).fract()).collect(),
            dose_density: vec![1.0 / 3.0; n],
            source: (0..n).map(|i| i < 18).collect(),
            sampling: TrialSampling::IndependentSamples,
        }
    }

    fn query() -> SmoothedDoseTransportQuery {
        SmoothedDoseTransportQuery {
            outcome: VariableId::from_raw(2),
            dose: VariableId::from_raw(1),
            source_population: Arc::from("trial"),
            target_population: Arc::from("target"),
            grid: Arc::from([1.0, 2.0]),
            bandwidth: 0.5,
            kernel: SmoothingKernel::Epanechnikov,
            dose_support: (0.0, 3.0),
            density_provenance: Arc::from("known"),
        }
    }

    fn model(basis: &DoseBasis) -> SmoothedDosePosteriorModel<'_> {
        SmoothedDosePosteriorModel {
            basis,
            intercept_scale: 5.0,
            coefficient_scale: 2.0,
            residual: ResidualPrior::KnownVariance(0.04),
            min_local_rows: 3,
            draws: 40_000,
            seed: 13,
        }
    }

    fn factor() -> PopulationFactor {
        PopulationFactor {
            population: Arc::from("trial"),
            regime: None,
            variables: Arc::from([]),
            conditioned_on: Arc::from([]),
            interventions: Arc::from([]),
        }
    }

    /// A standardization certificate over the single covariate with raw id 7.
    fn standardize() -> TransportIdentification {
        TransportIdentification::Transportable {
            formula: TransportFormula::Standardize {
                over: Arc::from([VariableId::from_raw(7)]),
                source_response: factor(),
                target_law: factor(),
            },
            certificate: TransportCertificate {
                rule: Arc::from("test"),
                selection_targets: Arc::from([]),
                premises: Arc::from([]),
            },
        }
    }

    #[test]
    fn grid_posterior_matches_the_conjugate_reference_and_keeps_the_covariance() {
        // Reference: numpy ridge posterior on the same basis, with the window average taken
        // as the mean over target rows of scipy-quadrature integrals (no mean-covariate
        // shortcut and no Gauss-Legendre rule).
        let basis = DoseBasis::default();
        let posterior =
            smoothed_dose_posterior(&standardize(), &query(), &input(), &model(&basis)).unwrap();
        let mean = posterior.exact_mean.as_ref().unwrap();
        assert!((mean[0] - 2.166_769_53).abs() < 1e-7 && (mean[1] - 2.616_840_59).abs() < 1e-7);
        let covariance = posterior.exact_covariance.as_ref().unwrap();
        for (got, want) in
            covariance.iter().zip([0.008_745_18, 0.004_698_84, 0.004_698_84, 0.008_106_85])
        {
            assert!((got - want).abs() < 1e-7, "{got} != {want}");
        }
        for (got, want) in posterior.mean.iter().zip(mean) {
            assert!((got - want).abs() < 0.005);
        }
        // Joint draws keep the cross-dose covariance (about 0.0047), which independent
        // per-dose draws would drop.
        assert!((posterior.covariance[1] - 0.004_698_84).abs() < 0.0006);
        assert_eq!(posterior.source_rows, 18);
        assert_eq!(posterior.target_rows, 6);
        assert_eq!(posterior.minimum_local_rows, 6);
    }

    #[test]
    fn uncertified_windows_and_thin_support_are_refused() {
        let basis = DoseBasis::default();
        let not_certified = TransportIdentification::NotCertified(NotCertifiedCertificate {
            reason: Arc::from("none"),
            witness: Arc::from(Vec::new()),
            message: Arc::from("none"),
        });
        assert!(
            smoothed_dose_posterior(&not_certified, &query(), &input(), &model(&basis)).is_err()
        );

        let mut wide = query();
        wide.bandwidth = 1.2;
        let refused =
            smoothed_dose_posterior(&standardize(), &wide, &input(), &model(&basis)).unwrap_err();
        assert!(refused.to_string().contains("dose_response.grid_outside_dose_support"));

        let mut strict = model(&basis);
        strict.min_local_rows = 12;
        let refused =
            smoothed_dose_posterior(&standardize(), &query(), &input(), &strict).unwrap_err();
        assert!(refused.to_string().contains("local support failure"));

        let mut wrong = input();
        wrong.features = vec![9];
        assert!(smoothed_dose_posterior(&standardize(), &query(), &wrong, &model(&basis)).is_err());
    }
}
