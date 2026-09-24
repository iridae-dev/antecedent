//! Conjugate Bayesian quadratic-basis g-computation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, ConjugateGaussianBackend,
    GaussianCoefficientPrior, InferenceBackend, InvGammaPrior, LaplaceWorkspace, PriorSet,
    PriorSpec,
};

use crate::{LearnError, PosteriorPrediction, PosteriorPredictionProvenance, PredictionTask};

/// Target population requested for Bayesian basis g-computation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BasisTargetPopulation {
    /// Empirical covariate distribution among complete observed rows.
    AllObserved,
    /// Any separately selected population; currently refused pending a target contract.
    Other,
}

/// Declared prior and draw settings for quadratic basis regression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BayesianBasisSpec {
    /// Standard deviation of the zero-centered, sigma-scaled coefficient prior.
    /// Every basis coefficient, including the intercept, uses this scale.
    pub prior_sd: f64,
    /// Posterior draws retained for all row-level predictions.
    pub n_draws: usize,
    /// Seed for deterministic conjugate posterior draws.
    pub seed: u64,
}

impl Default for BayesianBasisSpec {
    fn default() -> Self {
        Self { prior_sd: 2.0, n_draws: 1000, seed: 0 }
    }
}

/// Native Gaussian Bayesian basis-regression g-computation estimator.
#[derive(Clone, Copy, Debug, Default)]
pub struct BayesianBasisGComputation;

/// Posterior g-computation result with shared potential-outcome and CATE draws.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianBasisEffect {
    /// Joint outcome predictions for control, active, and individual effect.
    pub predictions: PosteriorPrediction,
    /// ATE posterior draws over the declared empirical `AllObserved` population.
    pub ate_draws: Arc<[f64]>,
    /// Posterior mean of the ATE draws.
    pub ate_mean: f64,
}

impl BayesianBasisGComputation {
    /// Fit a quadratic outcome basis and evaluate treatment at 0 and 1.
    ///
    /// Covariates use the basis `z_j` and `z_j²`; treatment interactions use
    /// `t × z_j`. This yields a row-level CATE `β_t + Σ β_tz_j z_j` and an ATE
    /// averaged over the same complete observed rows. Each posterior draw is
    /// shared across every row and both intervention levels.
    ///
    /// Covariates are column-major (`covariates[j * n_rows + row]`). Row ids
    /// default to `0..n_rows`; fold ids may be supplied by an upstream
    /// cross-fitting stage and are copied to the returned posterior prediction.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid dimensions/data, absent treatment
    /// variation, invalid shrinkage settings, or a target other than
    /// [`BasisTargetPopulation::AllObserved`].
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &self,
        treatment: &[f64],
        outcome: &[f64],
        covariates_colmajor: &[f64],
        n_covariates: usize,
        row_ids: Option<&[u32]>,
        fold_ids: Option<&[u16]>,
        population: BasisTargetPopulation,
        spec: BayesianBasisSpec,
        ctx: &ExecutionContext,
    ) -> Result<BayesianBasisEffect, LearnError> {
        if population != BasisTargetPopulation::AllObserved {
            return Err(LearnError::Unsupported {
                message: "Bayesian basis g-computation supports AllObserved only",
            });
        }
        let n = treatment.len();
        if n < 4
            || outcome.len() != n
            || covariates_colmajor.len() != n.saturating_mul(n_covariates)
        {
            return Err(LearnError::Shape {
                message: "basis g-computation input dimensions do not match rows and covariates",
            });
        }
        if n_covariates == 0 {
            return Err(LearnError::Shape { message: "basis g-computation requires covariates" });
        }
        if row_ids.is_some_and(|ids| ids.len() != n) {
            return Err(LearnError::Shape {
                message: "basis g-computation row ids length != rows",
            });
        }
        if fold_ids.is_some_and(|ids| ids.len() != n) {
            return Err(LearnError::Shape {
                message: "basis g-computation fold ids length != rows",
            });
        }
        if treatment.iter().any(|t| *t != 0.0 && *t != 1.0)
            || !treatment.iter().any(|t| *t == 0.0)
            || !treatment.iter().any(|t| *t == 1.0)
        {
            return Err(LearnError::Unsupported {
                message: "basis g-computation requires binary treatment with both levels 0 and 1",
            });
        }
        if treatment.iter().chain(outcome).chain(covariates_colmajor).any(|v| !v.is_finite()) {
            return Err(LearnError::Shape { message: "basis g-computation inputs must be finite" });
        }
        if spec.n_draws == 0 || !(spec.prior_sd > 0.0) || !spec.prior_sd.is_finite() {
            return Err(LearnError::Shape {
                message: "basis g-computation requires positive finite prior_sd and n_draws",
            });
        }

        let p = 2 + 3 * n_covariates;
        let mut design = vec![0.0; n * p];
        for r in 0..n {
            design[r] = 1.0;
            design[n + r] = treatment[r];
        }
        for j in 0..n_covariates {
            let z_col = &covariates_colmajor[j * n..(j + 1) * n];
            let z_offset = (2 + j) * n;
            let tz_offset = (2 + n_covariates + j) * n;
            let z2_offset = (2 + 2 * n_covariates + j) * n;
            for r in 0..n {
                design[z_offset + r] = z_col[r];
                design[tz_offset + r] = z_col[r] * treatment[r];
                design[z2_offset + r] = z_col[r] * z_col[r];
            }
        }

        let prior = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                    p,
                    spec.prior_sd,
                )),
                PriorSpec::ResidualInvGamma(InvGammaPrior::weakly_informative()),
            ],
            ..PriorSet::default()
        };
        let fit = ConjugateGaussianBackend.fit(
            BayesLikelihood::GaussianIdentity,
            BayesDesignRef {
                x_colmajor: &design,
                nrows: n,
                ncols: p,
                y: outcome,
                weights: None,
                offsets: None,
            },
            &prior,
            &BayesFitOptions { n_draws: spec.n_draws, seed: spec.seed, ..Default::default() },
            &mut LaplaceWorkspace::default(),
            ctx,
        )?;

        let row_ids: Vec<u32> = match row_ids {
            Some(ids) => ids.to_vec(),
            None => (0..n)
                .map(|i| {
                    u32::try_from(i).map_err(|_| LearnError::Shape {
                        message: "basis g-computation row count exceeds row id capacity",
                    })
                })
                .collect::<Result<_, _>>()?,
        };
        let beta_cols: Vec<&[f64]> =
            (0..p).map(|coefficient| fit.draws.column(coefficient)).collect::<Result<_, _>>()?;
        let mut values = vec![0.0; spec.n_draws * 3 * n];
        let mut ate_draws = vec![0.0; spec.n_draws];
        for draw in 0..spec.n_draws {
            let base = draw * 3 * n;
            let (control, remaining) = values[base..base + 3 * n].split_at_mut(n);
            let (active, effect) = remaining.split_at_mut(n);
            for r in 0..n {
                let mut y0 = beta_cols[0][draw];
                let mut y1 = y0 + beta_cols[1][draw];
                for j in 0..n_covariates {
                    let z = covariates_colmajor[j * n + r];
                    y0 += beta_cols[2 + j][draw] * z
                        + beta_cols[2 + 2 * n_covariates + j][draw] * z * z;
                    y1 += beta_cols[2 + j][draw] * z
                        + beta_cols[2 + n_covariates + j][draw] * z
                        + beta_cols[2 + 2 * n_covariates + j][draw] * z * z;
                }
                control[r] = y0;
                active[r] = y1;
                effect[r] = y1 - y0;
                ate_draws[draw] += effect[r] / n as f64;
            }
        }
        let provenance = PosteriorPredictionProvenance {
            model_id: Arc::from("bayesian_basis_gcomp.quadratic_tz_z2"),
            prior_id: Arc::from(format!("gaussian_coefficients_isotropic_sd_{}", spec.prior_sd)),
            model: crate::LearnerProvenance {
                spec: "bayesian_basis_gcomp".into(),
                implementation: "antecedent_prob::conjugate_gaussian".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
        };
        let predictions = PosteriorPrediction::new(
            spec.n_draws,
            vec![
                Arc::from("outcome_control"),
                Arc::from("outcome_active"),
                Arc::from("individual_effect"),
            ],
            row_ids,
            fold_ids.map(<[u16]>::to_vec),
            values,
            provenance,
        )?;
        let ate_mean = ate_draws.iter().sum::<f64>() / ate_draws.len() as f64;
        Ok(BayesianBasisEffect { predictions, ate_draws: Arc::from(ate_draws), ate_mean })
    }

    /// Stable public estimator identity.
    #[must_use]
    pub const fn estimator_id(&self) -> &'static str {
        "bayesian.basis_gcomp"
    }

    /// Model task; the estimator fits a continuous outcome regression.
    #[must_use]
    pub const fn task(&self) -> PredictionTask {
        PredictionTask::Regression
    }
}
