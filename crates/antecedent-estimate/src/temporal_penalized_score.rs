//! Temporal propensity of treatment at a decision time given lagged history.
//!
//! Each row is one independent subject. Column 0 is an intercept of ones. Every later column is
//! a lag. The outcome is the treatment at that decision time.
//!
//! The frequentist object maximises the Bernoulli log likelihood minus `(penalty / 2)` times the
//! squared Euclidean norm of the slopes, on the column scale as supplied (slopes are not
//! standardized). The intercept is unpenalized. The score is the inverse link at that mode.
//! This is the deliberate ridge fit, not a separation rescue and not the standardized penalty
//! inside cross-fitted AIPW.
//!
//! The Bayesian object is a Laplace posterior under a Gaussian prior on every coefficient,
//! including the intercept. The score is the mean of the inverse link over posterior draws. A
//! prior standard deviation is not a penalty, and that posterior mean is not the penalized mode.
//!
//! A single dependent series is refused. Nothing here registers a licensed route.
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
use antecedent_stats::{
    FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace, fit_glm_ridge,
};

use crate::error::EstimationError;

/// How the lag-aligned rows were sampled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrajectoryDependence {
    /// One row, one subject, one decision time. Rows are independent.
    IndependentSubjects,
    /// One serially dependent series. No propensity posterior or penalty is fit as if it were iid.
    OneDependentSeries,
}

/// Lag-aligned treatment design: intercept, then lags, outcome equal to treatment.
#[derive(Clone, Debug)]
pub struct TemporalPropensityDesign {
    x_colmajor: Vec<f64>,
    y: Vec<f64>,
    nrows: usize,
    ncols: usize,
    dependence: TrajectoryDependence,
}

/// A validated immutable propensity design borrowed from a retained native request.
#[derive(Clone, Copy, Debug)]
pub struct TemporalPropensityDesignRef<'a> {
    x_colmajor: &'a [f64],
    y: &'a [f64],
    nrows: usize,
    ncols: usize,
    dependence: TrajectoryDependence,
}

impl<'a> TemporalPropensityDesignRef<'a> {
    /// Validate without copying numerical storage.
    ///
    /// # Errors
    /// The same shape, finite-value, intercept and binary-treatment conditions as the owned design.
    pub fn try_new(
        x_colmajor: &'a [f64],
        y: &'a [f64],
        nrows: usize,
        dependence: TrajectoryDependence,
    ) -> Result<Self, EstimationError> {
        if nrows == 0 || y.len() != nrows {
            return Err(shape("the treatment must have one value per row"));
        }
        let Some(ncols) = x_colmajor.len().checked_div(nrows) else {
            return Err(shape("the design length is not a multiple of the row count"));
        };
        if ncols < 2 || nrows * ncols != x_colmajor.len() {
            return Err(shape("a temporal propensity needs an intercept and a lag"));
        }
        if !column_is_ones(&x_colmajor[..nrows]) {
            return Err(shape("column 0 is the intercept and must be identically one"));
        }
        binary_outcomes(&y)?;
        finite_column("design", &x_colmajor)?;
        Ok(Self { x_colmajor, y, nrows, ncols, dependence })
    }
    const fn refuse_series(&self) -> Result<(), EstimationError> {
        match self.dependence {
            TrajectoryDependence::IndependentSubjects => Ok(()),
            TrajectoryDependence::OneDependentSeries => Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("route_not_supported"),
                "a single dependent series is not an independent-trajectory propensity sample",
            )),
        }
    }
}

impl TemporalPropensityDesign {
    /// Assemble a column-major design with an intercept and at least one lag.
    ///
    /// # Errors
    ///
    /// A shape mismatch, a non-binary treatment, a missing lag, or a non-finite entry.
    pub fn try_new(
        x_colmajor: Vec<f64>,
        y: Vec<f64>,
        nrows: usize,
        dependence: TrajectoryDependence,
    ) -> Result<Self, EstimationError> {
        let borrowed = TemporalPropensityDesignRef::try_new(&x_colmajor, &y, nrows, dependence)?;
        let ncols = borrowed.ncols;
        Ok(Self { x_colmajor, y, nrows, ncols, dependence })
    }

    /// Borrow the same validated numerical storage without another ownership copy.
    #[must_use]
    pub fn as_ref(&self) -> TemporalPropensityDesignRef<'_> {
        TemporalPropensityDesignRef {
            x_colmajor: &self.x_colmajor,
            y: &self.y,
            nrows: self.nrows,
            ncols: self.ncols,
            dependence: self.dependence,
        }
    }
}

/// Ridge penalty on the temporal propensity. The intercept is unpenalized.
///
/// `penalty` multiplies `½ ‖slopes‖²` in a sum-scale Bernoulli objective. It is not a prior
/// variance and not a prior precision.
#[derive(Clone, Copy, Debug)]
pub struct FrequentistTemporalPropensity {
    penalty: f64,
}

impl FrequentistTemporalPropensity {
    /// Declare a finite positive penalty.
    ///
    /// # Errors
    ///
    /// A penalty that is not finite and positive.
    pub fn new(penalty: f64) -> Result<Self, EstimationError> {
        positive_finite(penalty, "ridge penalty must be finite and positive")?;
        Ok(Self { penalty })
    }

    /// Fit the penalized score.
    ///
    /// # Errors
    ///
    /// A dependent series, or a ridge fit that does not converge.
    pub fn fit(
        &self,
        design: &TemporalPropensityDesign,
    ) -> Result<PenalizedPropensityScore, EstimationError> {
        self.fit_borrowed(&design.as_ref())
    }

    /// Fit the same scientific model on validated borrowed native storage.
    ///
    /// # Errors
    /// The same fit and resource refusals as the owned design entry point.
    pub fn fit_borrowed(
        &self,
        design: &TemporalPropensityDesignRef<'_>,
    ) -> Result<PenalizedPropensityScore, EstimationError> {
        design.refuse_series()?;
        let fit = fit_glm_ridge(
            GlmFamily::BinomialLogit,
            GlmDesignRef {
                x_colmajor: design.x_colmajor,
                nrows: design.nrows,
                ncols: design.ncols,
                y: design.y,
            },
            &FaerBackend,
            &mut LeastSquaresWorkspace::default(),
            &GlmOptions::new(80, 1e-12),
            self.penalty,
        )?;
        fit.require_ok()?;
        if !fit.penalized || fit.coefficients.len() != design.ncols {
            return Err(EstimationError::stats_msg(
                "ridge propensity did not return a penalized mode",
            ));
        }
        Ok(PenalizedPropensityScore { penalty: self.penalty, coefficients: fit.coefficients })
    }
}

/// Inverse link of the penalized mode. This is not a posterior probability.
#[derive(Clone, Debug)]
#[must_use = "the penalized score is the inverse link of the mode, not a posterior mean"]
pub struct PenalizedPropensityScore {
    /// Penalty that was maximised.
    pub penalty: f64,
    /// Intercept, then lag slopes, at the penalized mode.
    pub coefficients: Vec<f64>,
}

impl PenalizedPropensityScore {
    /// Propensity at one lag history. `lags` lines up with the slope columns.
    ///
    /// # Errors
    ///
    /// A history whose length is not the number of lags.
    pub fn at(&self, lags: &[f64]) -> Result<f64, EstimationError> {
        Ok(sigmoid(linear_predictor(&self.coefficients, lags)?))
    }
}

/// Gaussian prior for the temporal propensity posterior.
///
/// `intercept_sd` and `slope_sd` are standard deviations of independent `N(0, sd²)` priors.
/// They are not a ridge penalty. The intercept is penalized by this prior; a ridge penalty
/// leaves it alone.
#[derive(Clone, Copy, Debug)]
pub struct BayesianTemporalPropensity {
    intercept_sd: f64,
    slope_sd: f64,
    n_draws: usize,
    seed: u64,
}

impl BayesianTemporalPropensity {
    /// State the two prior widths, the draw count, and the draw seed.
    ///
    /// # Errors
    ///
    /// A standard deviation that is not finite and positive, or a draw count of zero.
    pub fn new(
        intercept_sd: f64,
        slope_sd: f64,
        n_draws: usize,
        seed: u64,
    ) -> Result<Self, EstimationError> {
        positive_finite(
            intercept_sd,
            "intercept prior standard deviation must be finite and positive",
        )?;
        positive_finite(slope_sd, "slope prior standard deviation must be finite and positive")?;
        if n_draws == 0 {
            return Err(shape("the temporal propensity posterior needs at least one draw"));
        }
        Ok(Self { intercept_sd, slope_sd, n_draws, seed })
    }

    /// Fit the Laplace posterior of the logit coefficients.
    ///
    /// # Errors
    ///
    /// A dependent series, or a Laplace fit that does not publish a posterior.
    pub fn fit(
        &self,
        design: &TemporalPropensityDesign,
    ) -> Result<PosteriorPropensity, EstimationError> {
        self.fit_borrowed(&design.as_ref())
    }

    /// Fit the same scientific model on validated borrowed native storage.
    ///
    /// # Errors
    /// The same fit and resource refusals as the owned design entry point.
    pub fn fit_borrowed(
        &self,
        design: &TemporalPropensityDesignRef<'_>,
    ) -> Result<PosteriorPropensity, EstimationError> {
        design.refuse_series()?;
        let mut variance = Vec::with_capacity(design.ncols);
        variance.push(self.intercept_sd * self.intercept_sd);
        variance.extend(std::iter::repeat_n(self.slope_sd * self.slope_sd, design.ncols - 1));
        let coefficients = GaussianCoefficientPrior::from_absolute_variance(
            vec![0.0_f64; design.ncols],
            variance,
            1.0,
        )?;
        let prior = PriorSet {
            specs: vec![PriorSpec::GaussianCoefficients(coefficients)],
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
                y: design.y,
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
                return Err(EstimationError::data_msg("temporal propensity posterior cancelled"));
            }
            for column in 0..design.ncols {
                draws.push(fit.draws.get(draw, column)?);
            }
        }
        Ok(PosteriorPropensity {
            intercept_sd: self.intercept_sd,
            slope_sd: self.slope_sd,
            mode: fit.map,
            draws,
            n_draws: self.n_draws,
            ncols: design.ncols,
        })
    }
}

/// Posterior of the logit coefficients. Propensity at a history is a functional of these draws.
#[derive(Clone, Debug)]
#[must_use = "the posterior mean of the propensity is not the probability at the mode"]
pub struct PosteriorPropensity {
    /// Prior standard deviation of the intercept.
    pub intercept_sd: f64,
    /// Prior standard deviation of each lag slope.
    pub slope_sd: f64,
    /// Laplace mode.
    pub mode: Vec<f64>,
    /// Draw-major coefficients: index `draw * ncols + coefficient`.
    pub draws: Vec<f64>,
    /// Posterior draws.
    pub n_draws: usize,
    /// Intercept plus lags.
    pub ncols: usize,
}

impl PosteriorPropensity {
    /// Mean of the inverse link over posterior draws.
    ///
    /// # Errors
    ///
    /// A history whose length is not the number of lags.
    pub fn at(&self, lags: &[f64]) -> Result<f64, EstimationError> {
        let mut sum = 0.0_f64;
        for draw in 0..self.n_draws {
            if antecedent_core::numerical_admission::is_cancelled() {
                return Err(EstimationError::data_msg("temporal propensity summary cancelled"));
            }
            let start = draw * self.ncols;
            sum += sigmoid(linear_predictor(&self.draws[start..start + self.ncols], lags)?);
        }
        Ok(sum / self.n_draws as f64)
    }

    /// Inverse link at the Laplace mode. The published propensity is [`Self::at`].
    ///
    /// # Errors
    ///
    /// A history whose length is not the number of lags.
    pub fn at_mode(&self, lags: &[f64]) -> Result<f64, EstimationError> {
        Ok(sigmoid(linear_predictor(&self.mode, lags)?))
    }
}

fn linear_predictor(beta: &[f64], lags: &[f64]) -> Result<f64, EstimationError> {
    if lags.len() + 1 != beta.len() {
        return Err(shape("lag history length must equal the number of slope columns"));
    }
    if !lags.iter().all(|value| value.is_finite()) {
        return Err(shape("lag history must be finite"));
    }
    let mut eta = beta[0];
    for (slope, &lag) in beta[1..].iter().zip(lags) {
        eta += slope * lag;
    }
    Ok(eta)
}

fn shape(message: &str) -> EstimationError {
    EstimationError::data_msg(message)
}

fn positive_finite(value: f64, message: &str) -> Result<(), EstimationError> {
    if value.is_finite() && value > 0.0 { Ok(()) } else { Err(shape(message)) }
}

#[allow(
    clippy::float_cmp,
    reason = "the intercept column and Bernoulli outcomes are exact codes, not measurements"
)]
fn column_is_ones(column: &[f64]) -> bool {
    column.iter().all(|&value| value == 1.0)
}

#[allow(clippy::float_cmp, reason = "treatment is coded exactly 0 or 1")]
fn binary_outcomes(y: &[f64]) -> Result<(), EstimationError> {
    if y.iter().all(|&value| value == 0.0 || value == 1.0) {
        Ok(())
    } else {
        Err(shape("treatment must be coded 0 or 1"))
    }
}

fn finite_column(name: &str, values: &[f64]) -> Result<(), EstimationError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(shape(&format!("{name} values must be finite")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eight rows at lag 0 with one treated, eight at lag 1 with three treated.
    /// Marginal rate `4/16 = 0.25`. Cell rates `1/8` and `3/8`.
    fn lagged_treatment(dependence: TrajectoryDependence) -> TemporalPropensityDesign {
        let nrows = 16usize;
        let mut x = vec![0.0_f64; nrows * 2];
        let mut y = vec![0.0_f64; nrows];
        for row in 0..8 {
            x[row] = 1.0;
            x[nrows + row] = 0.0;
            y[row] = if row == 0 { 1.0 } else { 0.0 };
        }
        for row in 8..16 {
            x[row] = 1.0;
            x[nrows + row] = 1.0;
            y[row] = if row < 11 { 1.0 } else { 0.0 };
        }
        TemporalPropensityDesign::try_new(x, y, nrows, dependence).expect("lagged treatment")
    }

    #[test]
    fn a_design_without_a_lag_is_refused() {
        let err = TemporalPropensityDesign::try_new(
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.0, 1.0, 0.0, 1.0],
            4,
            TrajectoryDependence::IndependentSubjects,
        )
        .unwrap_err();
        assert!(err.to_string().contains("lag"));
    }

    #[test]
    fn one_dependent_series_is_refused() {
        let design = lagged_treatment(TrajectoryDependence::OneDependentSeries);
        let penalty = FrequentistTemporalPropensity::new(1.0).unwrap();
        assert!(penalty.fit(&design).unwrap_err().to_string().contains("dependent series"));
        let prior = BayesianTemporalPropensity::new(1.0, 1.0, 4, 0).unwrap();
        assert!(prior.fit(&design).is_err());
    }

    #[test]
    fn a_nonpositive_penalty_is_refused() {
        assert!(FrequentistTemporalPropensity::new(0.0).is_err());
        assert!(FrequentistTemporalPropensity::new(f64::NAN).is_err());
        assert!(BayesianTemporalPropensity::new(0.0, 1.0, 4, 0).is_err());
    }

    #[test]
    fn large_penalty_keeps_the_marginal_rate() {
        let score = FrequentistTemporalPropensity::new(1.0e4)
            .unwrap()
            .fit(&lagged_treatment(TrajectoryDependence::IndependentSubjects))
            .unwrap();
        let at_zero = score.at(&[0.0]).unwrap();
        let at_one = score.at(&[1.0]).unwrap();
        assert!((at_zero - 0.25).abs() < 0.02, "{at_zero}");
        assert!((at_one - 0.25).abs() < 0.02, "{at_one}");
        assert!((at_zero - 0.5).abs() > 0.15);
        assert!((score.penalty - 1.0e4).abs() < 1e-9);
    }

    #[test]
    fn small_penalty_keeps_the_lag_specific_rates() {
        let score = FrequentistTemporalPropensity::new(1.0e-4)
            .unwrap()
            .fit(&lagged_treatment(TrajectoryDependence::IndependentSubjects))
            .unwrap();
        let at_zero = score.at(&[0.0]).unwrap();
        let at_one = score.at(&[1.0]).unwrap();
        assert!(at_zero > 0.10 && at_zero < 0.16, "{at_zero}");
        assert!(at_one > 0.34 && at_one < 0.42, "{at_one}");
    }

    #[test]
    fn score_is_the_inverse_link_of_the_penalized_mode() {
        let score = FrequentistTemporalPropensity::new(1.0)
            .unwrap()
            .fit(&lagged_treatment(TrajectoryDependence::IndependentSubjects))
            .unwrap();
        let eta = score.coefficients[0] + score.coefficients[1];
        assert!((score.at(&[1.0]).unwrap() - sigmoid(eta)).abs() < 1e-12);
    }

    #[test]
    fn a_tight_prior_is_not_a_large_penalty() {
        let design = lagged_treatment(TrajectoryDependence::IndependentSubjects);
        let penalized = FrequentistTemporalPropensity::new(1.0e4).unwrap().fit(&design).unwrap();
        let posterior =
            BayesianTemporalPropensity::new(0.02, 0.02, 40, 1).unwrap().fit(&design).unwrap();
        let penalty_score = penalized.at(&[0.0]).unwrap();
        let prior_score = posterior.at(&[0.0]).unwrap();
        assert!((penalty_score - 0.25).abs() < 0.02, "{penalty_score}");
        assert!(prior_score > 0.45 && prior_score < 0.55, "{prior_score}");
        assert!((prior_score - penalty_score).abs() > 0.15);
    }

    #[test]
    fn posterior_mean_propensity_is_not_the_mode_probability() {
        let posterior = BayesianTemporalPropensity::new(1.0, 1.0, 200, 3)
            .unwrap()
            .fit(&lagged_treatment(TrajectoryDependence::IndependentSubjects))
            .unwrap();
        let mean = posterior.at(&[0.0]).unwrap();
        let mode = posterior.at_mode(&[0.0]).unwrap();
        assert_ne!(mean.to_bits(), mode.to_bits());
        assert!((posterior.intercept_sd - 1.0).abs() < 1e-15);
        assert!((posterior.slope_sd - 1.0).abs() < 1e-15);
    }
}

#[cfg(test)]
mod borrowed_design_regression {
    use super::*;
    #[test]
    fn borrowed_temporal_fit_matches_owned_ridge_and_posterior_without_design_copies() {
        let mut x = vec![1.0; 200];
        x.extend((0..200).map(|i| if i >= 100 { 1.0 } else { 0.0 }));
        let y =
            (0..200).map(|i| if i < 30 || (100..160).contains(&i) { 1.0 } else { 0.0 }).collect();
        let design =
            TemporalPropensityDesign::try_new(x, y, 200, TrajectoryDependence::IndependentSubjects)
                .unwrap();
        let borrowed = design.as_ref();
        assert_eq!(borrowed.x_colmajor.as_ptr(), design.x_colmajor.as_ptr());
        assert_eq!(borrowed.y.as_ptr(), design.y.as_ptr());
        let ridge = FrequentistTemporalPropensity::new(2.0).unwrap();
        assert_eq!(
            ridge.fit(&design).unwrap().coefficients,
            ridge.fit_borrowed(&borrowed).unwrap().coefficients
        );
        let prior = BayesianTemporalPropensity::new(1.0, 0.5, 8, 13).unwrap();
        let owned = prior.fit(&design).unwrap();
        let viewed = prior.fit_borrowed(&borrowed).unwrap();
        assert_eq!(owned.mode, viewed.mode);
        assert_eq!(owned.draws, viewed.draws);
    }
}
