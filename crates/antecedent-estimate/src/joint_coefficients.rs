//! Joint draws of the coefficients of several regressions fitted to one series.
//!
//! Regressions read from one series (the horizons of a temporal response, the mediator and
//! outcome mechanisms of a mediation) have posteriors that are not independent: they share
//! innovations, regressors and the calendar. Fitting each separately and pairing draws by index
//! samples a product of marginals. Here each regression keeps the mean and covariance of its own
//! posterior draws and the regressions are tied by the long-run covariance of the stacked
//! normal-equation scores on the shared calendar
//! ([`crate::temporal_slope_design::joint_dependence_factor`]); one draw is one standard-normal
//! vector for the whole stack, correlated by that structure and mapped back through each
//! regression's own covariance factor.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_prob::PosteriorQuantityKind;
use antecedent_stats::CompiledDesign;

use crate::bayesian::CausalPosterior;
use crate::error::EstimationError;
use crate::temporal_slope_design::{DependencePart, joint_dependence_factor};

/// Stream domain base of the joint draws.
const JOINT_STREAM: u64 = 0x7B17_4A1C;

/// The coefficient draws of one regression with the scores that tie it to the others.
pub(crate) struct CoefficientDraws {
    /// `coef[j]` is the posterior draws of design coefficient `j`.
    pub(crate) coef: Vec<Vec<f64>>,
    /// Normal-equation scores of the regression's least-squares fit, one series per column.
    pub(crate) scores: Vec<Vec<f64>>,
    /// Series time of each design row.
    pub(crate) times: Vec<usize>,
}

impl CoefficientDraws {
    /// Retain the coefficient draws of `posterior` and the scores of `design`, whose rows sit
    /// at series `times`.
    ///
    /// # Errors
    ///
    /// A posterior without a coefficient of the design, a design whose rows do not match
    /// `times`, or a least-squares fit that fails.
    pub(crate) fn of(
        posterior: &CausalPosterior,
        design: &CompiledDesign,
        times: Vec<usize>,
    ) -> Result<Self, EstimationError> {
        let (n, p) = (design.nrows, design.ncols);
        if times.len() != n {
            return Err(EstimationError::unsupported(
                "the design rows do not map one-to-one onto series times",
            ));
        }
        let mut coef = Vec::with_capacity(p);
        for index in 0..p {
            let column = posterior
                .draws
                .schema
                .quantities
                .iter()
                .position(|q| {
                    matches!(q, PosteriorQuantityKind::Coefficient { index: i, .. } if *i == index)
                })
                .ok_or_else(|| {
                    EstimationError::stats_msg("response posterior missing coefficient")
                })?;
            coef.push(posterior.draws.column(column)?.to_vec());
        }
        let scores =
            crate::temporal_block::normal_equation_scores(&design.matrix, n, p, &design.outcome)
                .ok_or_else(|| {
                    EstimationError::stats_msg("the regression scores could not be formed")
                })?;
        Ok(Self { coef, scores, times })
    }
}

/// Joint draws: `theta[k][j]` is draw column `j` of regression `k`, and draw `r` of every column
/// is one draw of the whole stack. Returns them with the dependence block length.
///
/// # Errors
///
/// Fewer than two draws, a posterior covariance that is not positive definite, or a joint
/// dependence structure no bounded shrinkage makes positive definite.
pub(crate) fn joint_draws(
    fits: &[&CoefficientDraws],
    structural_span: usize,
    seed: u64,
    rng: &antecedent_core::RngFactory,
) -> Result<(Vec<Vec<Vec<f64>>>, usize), EstimationError> {
    let n_draws = fits.iter().filter_map(|f| f.coef.first().map(Vec::len)).min().unwrap_or(0);
    if n_draws < 2 {
        return Err(EstimationError::stats_msg("a joint law needs at least two draws"));
    }
    struct Moments {
        mean: Vec<f64>,
        covariance: Vec<f64>,
        factor: Vec<f64>,
    }
    let mut moments = Vec::with_capacity(fits.len());
    for fit in fits {
        let p = fit.coef.len();
        let mean: Vec<f64> =
            fit.coef.iter().map(|c| c[..n_draws].iter().sum::<f64>() / n_draws as f64).collect();
        let mut covariance = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..=i {
                let value: f64 = (0..n_draws)
                    .map(|r| (fit.coef[i][r] - mean[i]) * (fit.coef[j][r] - mean[j]))
                    .sum::<f64>()
                    / (n_draws - 1) as f64;
                covariance[i * p + j] = value;
                covariance[j * p + i] = value;
            }
        }
        let scale = (0..p).map(|i| covariance[i * p + i]).fold(0.0_f64, f64::max);
        let mut factor = None;
        for ridge in [0.0, 1e-12, 1e-10, 1e-8] {
            let mut jittered = covariance.clone();
            for i in 0..p {
                jittered[i * p + i] += ridge * scale;
            }
            if let Some(found) = antecedent_stats::cholesky_spd(&jittered, p) {
                factor = Some(found);
                break;
            }
        }
        let factor = factor.ok_or_else(|| {
            EstimationError::stats_msg("a posterior covariance is not positive definite")
        })?;
        moments.push(Moments { mean, covariance, factor });
    }
    let parts: Vec<DependencePart<'_>> = fits
        .iter()
        .zip(&moments)
        .map(|(fit, m)| DependencePart {
            inverse: &m.covariance,
            scores: &fit.scores,
            times: &fit.times,
        })
        .collect();
    let (joint, block) = joint_dependence_factor(&parts, structural_span)?;
    let widths: Vec<usize> = fits.iter().map(|f| f.coef.len()).collect();
    let total: usize = widths.iter().sum();
    let mut theta: Vec<Vec<Vec<f64>>> =
        widths.iter().map(|&w| vec![Vec::with_capacity(n_draws); w]).collect();
    let base = antecedent_core::derive_stream(seed, JOINT_STREAM);
    let mut z = vec![0.0; total];
    for r in 0..n_draws {
        let mut stream: antecedent_core::CausalRng = rng.stream_for(
            antecedent_core::StreamDomain::Bayesian,
            antecedent_core::derive_stream(base, r as u64),
        );
        for slot in &mut z {
            *slot = antecedent_kernels::standard_normal(&mut stream);
        }
        let v: Vec<f64> =
            (0..total).map(|i| (0..=i).map(|j| joint[i * total + j] * z[j]).sum()).collect();
        let mut offset = 0;
        for (k, (m, &width)) in moments.iter().zip(&widths).enumerate() {
            for j in 0..width {
                let value = m.mean[j]
                    + (0..=j).map(|c| m.factor[j * width + c] * v[offset + c]).sum::<f64>();
                theta[k][j].push(value);
            }
            offset += width;
        }
    }
    Ok((theta, block))
}

/// `posterior` with its coefficient columns replaced by `columns` (and every other quantity cut
/// to the same number of draws), and its summaries recomputed.
///
/// # Errors
///
/// A column count that does not match the posterior's coefficients.
pub(crate) fn with_coefficient_columns(
    posterior: &CausalPosterior,
    columns: &[Vec<f64>],
) -> Result<CausalPosterior, EstimationError> {
    let n = columns.first().map_or(0, Vec::len);
    let mut values = Vec::with_capacity(n * posterior.draws.schema.quantities.len());
    for (q, kind) in posterior.draws.schema.quantities.iter().enumerate() {
        match kind {
            PosteriorQuantityKind::Coefficient { index, .. } => {
                let column = columns.get(*index).ok_or_else(|| {
                    EstimationError::stats_msg("a joint column is missing for a coefficient")
                })?;
                values.extend_from_slice(&column[..n]);
            }
            _ => values.extend_from_slice(&posterior.draws.column(q)?[..n]),
        }
    }
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        posterior.draws.schema.clone(),
        n,
        values,
    )?;
    let mut out = posterior.clone();
    out.summaries = draws.summarize();
    out.draws = draws;
    Ok(out)
}
