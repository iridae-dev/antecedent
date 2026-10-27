//! Posterior of a static logistic propensity under a Gaussian coefficient prior.
//!
//! A ridge penalty is the log density of a Gaussian prior, and the penalized fit is that prior's
//! posterior mode. The mode is a point. A Laplace posterior around it is a proper posterior of
//! the propensity coefficients, and each row's score has a posterior of its own: its mean, its
//! spread, and the probability that it falls outside a positivity band. That is a posterior of a
//! nuisance, and it is all this module publishes.
//!
//! It does not publish a posterior of the weighting effect. Under a likelihood that factors into
//! a treatment model and an outcome model with independent priors, the propensity coefficients
//! are ancillary for the average effect: the effect's posterior is a function of the outcome
//! model and the covariate distribution only. Pushing propensity draws through the weighting
//! identity with the outcomes held fixed carries propensity uncertainty alone, and is not that
//! posterior. [`PropensityCoefficientPosterior::weighting_effect_posterior`] refuses, and names
//! the objects that are posteriors of the same effect.
//!
//! The prior is `N(0, prior_sd²)` on every coefficient, the intercept included. It is not the
//! standardized, intercept-free ridge of the cross-fitted penalized propensity, and the posterior
//! mean score is not the penalized-fit score.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::float_cmp,
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_kernels::sigmoid;
use antecedent_prob::laplace::fit_laplace_glm_owned;
use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, GaussianCoefficientPrior, PriorSet, PriorSpec,
};

use crate::error::EstimationError;
use crate::util::dot;
use crate::util::refuse;

/// Column-major propensity design: an intercept of ones, then covariates. Treatment is 0 or 1.
#[derive(Clone, Debug)]
pub struct StaticPropensityDesign {
    x_colmajor: Vec<f64>,
    treatment: Vec<f64>,
    nrows: usize,
    ncols: usize,
}

/// A validated immutable propensity design borrowed from a retained native request.
#[derive(Clone, Copy, Debug)]
pub struct StaticPropensityDesignRef<'a> {
    x_colmajor: &'a [f64],
    treatment: &'a [f64],
    nrows: usize,
    ncols: usize,
}

impl<'a> StaticPropensityDesignRef<'a> {
    /// Validate without copying numerical storage.
    ///
    /// # Errors
    /// The same shape, finite-value, intercept and binary-treatment conditions as the owned design.
    pub fn try_new(
        x_colmajor: &'a [f64],
        treatment: &'a [f64],
        nrows: usize,
    ) -> Result<Self, EstimationError> {
        if nrows == 0 || treatment.len() != nrows {
            return Err(shape("the treatment must have one value per row"));
        }
        let Some(ncols) = x_colmajor.len().checked_div(nrows) else {
            return Err(shape("the design length is not a multiple of the row count"));
        };
        if ncols == 0 || nrows * ncols != x_colmajor.len() {
            return Err(shape("the design length is not a multiple of the row count"));
        }
        if !x_colmajor.iter().all(|value| value.is_finite()) {
            return Err(shape("design values must be finite"));
        }
        if !x_colmajor.iter().take(nrows).all(|&value| is_exactly(value, 1.0)) {
            return Err(shape("column 0 is the intercept and must be identically one"));
        }
        if !treatment.iter().all(|&value| is_exactly(value, 0.0) || is_exactly(value, 1.0)) {
            return Err(shape("treatment must be coded 0 or 1"));
        }
        Ok(Self { x_colmajor, treatment, nrows, ncols })
    }
}

impl StaticPropensityDesign {
    /// Assemble a design.
    ///
    /// # Errors
    ///
    /// A shape mismatch, an intercept column that is not identically one, a treatment that is
    /// not coded 0 or 1, or a non-finite entry.
    pub fn try_new(
        x_colmajor: Vec<f64>,
        treatment: Vec<f64>,
        nrows: usize,
    ) -> Result<Self, EstimationError> {
        let borrowed = StaticPropensityDesignRef::try_new(&x_colmajor, &treatment, nrows)?;
        let ncols = borrowed.ncols;
        Ok(Self { x_colmajor, treatment, nrows, ncols })
    }
    /// Borrow the same validated numerical storage without another ownership copy.
    #[must_use]
    pub fn as_ref(&self) -> StaticPropensityDesignRef<'_> {
        StaticPropensityDesignRef {
            x_colmajor: &self.x_colmajor,
            treatment: &self.treatment,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

/// Gaussian prior and draw settings of the propensity posterior.
#[derive(Clone, Copy, Debug)]
pub struct PropensityCoefficientPrior {
    prior_sd: f64,
    n_draws: usize,
    seed: u64,
}

impl PropensityCoefficientPrior {
    /// State the prior standard deviation of every coefficient, the draw count and the seed.
    ///
    /// # Errors
    ///
    /// A standard deviation that is not finite and positive, or a draw count of zero.
    pub fn new(prior_sd: f64, n_draws: usize, seed: u64) -> Result<Self, EstimationError> {
        if !(prior_sd.is_finite() && prior_sd > 0.0) {
            return Err(shape("prior standard deviation must be finite and positive"));
        }
        if n_draws == 0 {
            return Err(shape("the propensity posterior needs at least one draw"));
        }
        Ok(Self { prior_sd, n_draws, seed })
    }

    /// Fit the Laplace posterior of the logit coefficients.
    ///
    /// # Errors
    ///
    /// A Laplace fit that does not publish a posterior.
    pub fn fit(
        &self,
        design: &StaticPropensityDesign,
    ) -> Result<PropensityCoefficientPosterior, EstimationError> {
        self.fit_borrowed(&design.as_ref())
    }

    /// Fit the same scientific model on validated borrowed native storage.
    ///
    /// # Errors
    /// The same fit and resource refusals as the owned design entry point.
    pub fn fit_borrowed(
        &self,
        design: &StaticPropensityDesignRef<'_>,
    ) -> Result<PropensityCoefficientPosterior, EstimationError> {
        let prior = PriorSet {
            specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                design.ncols,
                self.prior_sd,
            ))],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        prior.validate()?;
        let fit = fit_laplace_glm_owned(
            BayesLikelihood::BernoulliLogit,
            BayesDesignRef {
                x_colmajor: design.x_colmajor,
                nrows: design.nrows,
                ncols: design.ncols,
                y: design.treatment,
                weights: None,
                offsets: None,
            },
            &prior,
            &BayesFitOptions {
                n_draws: self.n_draws,
                max_iter: 50,
                grad_tol: 1e-8,
                seed: self.seed,
            },
        )?;
        if fit.map.len() != design.ncols {
            return Err(EstimationError::stats_msg(
                "Laplace mode length does not match the design",
            ));
        }
        let _admitted = crate::util::admit_bound_array::<f64>(self.n_draws, design.ncols)?;
        let mut draws = Vec::with_capacity(self.n_draws * design.ncols);
        for draw in 0..self.n_draws {
            if antecedent_core::numerical_admission::is_cancelled() {
                return Err(EstimationError::data_msg("propensity posterior cancelled"));
            }
            for column in 0..design.ncols {
                draws.push(fit.draws.get(draw, column)?);
            }
        }
        Ok(PropensityCoefficientPosterior {
            prior_sd: self.prior_sd,
            mode: fit.map,
            draws,
            n_draws: self.n_draws,
            ncols: design.ncols,
        })
    }
}

/// Posterior of the logit coefficients of a static propensity.
#[derive(Clone, Debug)]
#[must_use = "a posterior of the propensity is not a posterior of the weighting effect"]
pub struct PropensityCoefficientPosterior {
    /// Standard deviation of the Gaussian prior on every coefficient.
    pub prior_sd: f64,
    /// Laplace mode, which is the ridge fit with penalty `1 / prior_sd²` on every coefficient.
    pub mode: Vec<f64>,
    /// Draw-major coefficients: index `draw * ncols + coefficient`.
    pub draws: Vec<f64>,
    /// Posterior draws.
    pub n_draws: usize,
    /// Intercept plus covariates.
    pub ncols: usize,
}

/// Posterior summary of one row's propensity score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropensityScoreSummary {
    /// Mean of the inverse link over the posterior draws.
    pub mean: f64,
    /// Inverse link at the Laplace mode. Not the posterior mean.
    pub at_mode: f64,
    /// Posterior standard deviation of the score.
    pub sd: f64,
    /// Posterior probability that the score lies below `clip` or above `1 - clip`.
    pub outside_band: f64,
}

impl PropensityCoefficientPosterior {
    /// Summarize the score of one design row, `[1, x₁, …]`.
    ///
    /// # Errors
    ///
    /// A row whose length is not the number of coefficients, a non-finite entry, or a band
    /// `clip` outside `(0, 0.5)`.
    pub fn score_at(
        &self,
        row: &[f64],
        clip: f64,
    ) -> Result<PropensityScoreSummary, EstimationError> {
        if row.len() != self.ncols || !row.iter().all(|value| value.is_finite()) {
            return Err(shape("the row must be finite with one value per coefficient"));
        }
        if !(clip > 0.0 && clip < 0.5) {
            return Err(shape("the positivity band must lie strictly inside (0, 0.5)"));
        }
        let mut sum = 0.0_f64;
        let mut sum_squares = 0.0_f64;
        let mut outside = 0usize;
        for draw in 0..self.n_draws {
            if antecedent_core::numerical_admission::is_cancelled() {
                return Err(EstimationError::data_msg("propensity summary cancelled"));
            }
            let start = draw * self.ncols;
            let score = sigmoid(dot(&self.draws[start..start + self.ncols], row));
            sum += score;
            sum_squares += score * score;
            if score < clip || score > 1.0 - clip {
                outside += 1;
            }
        }
        let n = self.n_draws as f64;
        let mean = sum / n;
        let variance = if self.n_draws > 1 {
            ((sum_squares - n * mean * mean) / (n - 1.0)).max(0.0)
        } else {
            0.0
        };
        Ok(PropensityScoreSummary {
            mean,
            at_mode: sigmoid(dot(&self.mode, row)),
            sd: variance.sqrt(),
            outside_band: outside as f64 / n,
        })
    }

    /// The weighting effect's posterior from these draws. It does not exist.
    ///
    /// # Errors
    ///
    /// Always. The refusal names the missing premise and the objects that answer the same
    /// effect.
    pub fn weighting_effect_posterior(&self) -> Result<Vec<f64>, EstimationError> {
        Err(weighting_effect_posterior_refusal())
    }
}

/// The typed refusal of a weighting-effect posterior built from propensity draws.
///
/// The refusal names the missing premise (an outcome law) and the objects that are posteriors of
/// the same effect.
#[must_use]
pub fn weighting_effect_posterior_refusal() -> EstimationError {
    refuse(
        antecedent_core::reason_code!("route_not_supported"),
        "penalized_propensity.effect_posterior",
        "a posterior of the propensity coefficients is not a posterior of the weighting \
         effect: the outcome law is unmodelled, and under a factorised likelihood the \
         propensity is ancillary for the average effect, so propensity draws pushed through \
         the weighting identity carry propensity uncertainty alone. Next attempt: the \
         Bayesian-bootstrap posterior of the cross-fitted ridge-orthogonal score, or the \
         Bayesian g-computation posterior of an outcome model",
    )
}

fn shape(message: &str) -> EstimationError {
    EstimationError::data_msg(message)
}

#[allow(clippy::float_cmp, reason = "the intercept and treatment are exact codes")]
fn is_exactly(value: f64, code: f64) -> bool {
    value == code
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 100 rows at `x = 0` with 30 treated and 100 rows at `x = 1` with 60 treated.
    fn two_cell_design() -> StaticPropensityDesign {
        let nrows = 200usize;
        let mut x = vec![1.0_f64; nrows];
        x.extend((0..nrows).map(|row| if row >= 100 { 1.0 } else { 0.0 }));
        let treatment: Vec<f64> = (0..nrows)
            .map(|row| {
                let treated = if row >= 100 { row - 100 < 60 } else { row < 30 };
                if treated { 1.0 } else { 0.0 }
            })
            .collect();
        StaticPropensityDesign::try_new(x, treatment, nrows).expect("two-cell design")
    }

    fn fit(prior_sd: f64) -> PropensityCoefficientPosterior {
        PropensityCoefficientPrior::new(prior_sd, 4000, 13)
            .expect("prior")
            .fit(&two_cell_design())
            .expect("Laplace posterior")
    }

    #[test]
    fn a_wide_prior_recovers_the_cell_rates_with_their_posterior_spread() {
        // Reference: Newton mode and Laplace covariance of the two-parameter model, with the
        // score moments from a large draw. Cell rates are 0.30 and 0.60. N(0, 2^2) on both
        // coefficients puts the mode at 0.3051 and 0.5970, the posterior mean score at 0.3069
        // and 0.5960, and the posterior sd at 0.0453 and 0.0484.
        let posterior = fit(2.0);
        let low = posterior.score_at(&[1.0, 0.0], 0.35).expect("low cell");
        let high = posterior.score_at(&[1.0, 1.0], 0.35).expect("high cell");
        assert!((low.at_mode - 0.305_098).abs() < 1e-4, "mode {}", low.at_mode);
        assert!((high.at_mode - 0.596_960).abs() < 1e-4, "mode {}", high.at_mode);
        assert!((low.mean - 0.306_894).abs() < 0.004, "mean {}", low.mean);
        assert!((high.mean - 0.596_026).abs() < 0.004, "mean {}", high.mean);
        assert!((low.sd / 0.045_336 - 1.0).abs() < 0.06, "sd {}", low.sd);
        assert!((high.sd / 0.048_367 - 1.0).abs() < 0.06, "sd {}", high.sd);
        // The low cell sits below the 0.35 band edge with posterior probability 0.830.
        assert!((low.outside_band - 0.829).abs() < 0.03, "outside {}", low.outside_band);
        // The high cell exceeds 0.65 with posterior probability 0.132.
        assert!((high.outside_band - 0.132).abs() < 0.03, "outside {}", high.outside_band);
    }

    #[test]
    fn a_tight_prior_is_a_posterior_not_a_large_penalty() {
        // N(0, 0.1^2) holds both scores near one half: modes 0.4793 and 0.5034, spread 0.0207.
        let posterior = fit(0.1);
        let low = posterior.score_at(&[1.0, 0.0], 0.35).expect("low cell");
        let high = posterior.score_at(&[1.0, 1.0], 0.35).expect("high cell");
        assert!((low.at_mode - 0.479_320).abs() < 1e-4, "mode {}", low.at_mode);
        assert!((high.at_mode - 0.503_447).abs() < 1e-4, "mode {}", high.at_mode);
        assert!((low.sd / 0.020_665 - 1.0).abs() < 0.06, "sd {}", low.sd);
        assert!(low.sd > 0.0);
        assert_eq!(low.outside_band, 0.0);
    }

    #[test]
    fn the_weighting_effect_has_no_posterior_from_propensity_draws() {
        let posterior = fit(2.0);
        let message = posterior.weighting_effect_posterior().unwrap_err().to_string();
        assert!(message.contains("penalized_propensity.effect_posterior"), "{message}");
        assert!(message.contains("ancillary"), "{message}");
        assert!(message.contains("Next attempt"), "{message}");
    }

    #[test]
    fn invalid_designs_priors_and_rows_are_refused() {
        assert!(PropensityCoefficientPrior::new(0.0, 10, 1).is_err());
        assert!(PropensityCoefficientPrior::new(1.0, 0, 1).is_err());
        assert!(StaticPropensityDesign::try_new(vec![1.0, 0.5], vec![0.0, 1.0], 2).is_err());
        assert!(StaticPropensityDesign::try_new(vec![1.0, 1.0], vec![0.0, 2.0], 2).is_err());
        let posterior = fit(2.0);
        assert!(posterior.score_at(&[1.0], 0.1).is_err());
        assert!(posterior.score_at(&[1.0, 0.0], 0.5).is_err());
    }
}

#[cfg(test)]
mod borrowed_design_regression {
    use super::*;
    #[test]
    fn borrowed_and_owned_laplace_paths_share_inputs_and_identical_mode_draws() {
        let mut x = vec![1.0; 200];
        x.extend((0..200).map(|i| if i >= 100 { 1.0 } else { 0.0 }));
        let y =
            (0..200).map(|i| if i < 30 || (100..160).contains(&i) { 1.0 } else { 0.0 }).collect();
        let design = StaticPropensityDesign::try_new(x, y, 200).unwrap();
        let borrowed = design.as_ref();
        assert_eq!(borrowed.x_colmajor.as_ptr(), design.x_colmajor.as_ptr());
        assert_eq!(borrowed.treatment.as_ptr(), design.treatment.as_ptr());
        let prior = PropensityCoefficientPrior::new(2.0, 8, 13).unwrap();
        let owned = prior.fit(&design).unwrap();
        let viewed = prior.fit_borrowed(&borrowed).unwrap();
        assert_eq!(owned.mode, viewed.mode);
        assert_eq!(owned.draws, viewed.draws);
        assert!((viewed.score_at(&[1.0, 0.0], 0.05).unwrap().at_mode - 0.305_098).abs() < 1e-4);
    }
}
