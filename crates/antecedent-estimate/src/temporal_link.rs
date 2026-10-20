//! Decision-time g-computation for a temporal GLM with a non-identity link.
//!
//! Each row is one independent subject at one decision time. Lag columns stay at their observed
//! values. Regime columns — the treatment, and any column built from it — take the active value
//! and then the control value. The estimand is the average mean-scale contrast. The coefficient
//! on treatment is a link-scale parameter, not that contrast.
//!
//! The identity link belongs to temporal linear adjustment and is refused. A ridge penalty is
//! neither the frequentist fit nor the Bayesian object. The frequentist fit is the unpenalized
//! GLM maximum likelihood estimate and its Fisher delta-method standard error. The Bayesian
//! object is a Laplace posterior under an explicit Gaussian prior on the coefficients; the
//! published summary is the mean of the contrast over those draws.
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

use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, GaussianCoefficientPrior, LaplaceWorkspace,
    PriorSet, PriorSpec, fit_laplace_glm,
};
use antecedent_stats::{
    FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace, fit_glm, invert_square,
};

use crate::error::EstimationError;
use crate::glm_adjustment::FamilyKernel;

/// How the lag-aligned rows were sampled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrajectoryDependence {
    /// One row, one subject, one decision time. Rows are independent.
    IndependentSubjects,
    /// One serially dependent series. No iid information matrix is reported for it.
    OneDependentSeries,
}

/// Outcome link of the temporal GLM.
///
/// Identity is absent: that estimator is temporal linear adjustment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TemporalOutcomeLink {
    /// Bernoulli outcome, logit link.
    BernoulliLogit,
    /// Bernoulli outcome, probit link.
    BernoulliProbit,
}

impl TemporalOutcomeLink {
    /// Map a GLM family onto this contrast.
    ///
    /// # Errors
    ///
    /// The identity link, and every family other than Bernoulli logit or probit.
    pub const fn from_family(family: GlmFamily) -> Result<Self, EstimationError> {
        match family {
            GlmFamily::BinomialLogit => Ok(Self::BernoulliLogit),
            GlmFamily::BinomialProbit => Ok(Self::BernoulliProbit),
            GlmFamily::GaussianIdentity => Err(EstimationError::unsupported(
                "an identity link belongs to temporal linear adjustment",
            )),
            GlmFamily::PoissonLog | GlmFamily::NegativeBinomial => Err(EstimationError::unsupported(
                "temporal GLM contrast supports Bernoulli logit and Bernoulli probit",
            )),
        }
    }

    const fn family(self) -> GlmFamily {
        match self {
            Self::BernoulliLogit => GlmFamily::BinomialLogit,
            Self::BernoulliProbit => GlmFamily::BinomialProbit,
        }
    }

    const fn likelihood(self) -> BayesLikelihood {
        match self {
            Self::BernoulliLogit => BayesLikelihood::BernoulliLogit,
            Self::BernoulliProbit => BayesLikelihood::BernoulliProbit,
        }
    }

    const fn kernel(self) -> FamilyKernel {
        FamilyKernel::new(self.family())
    }
}

/// One design column that changes with the intervention.
#[derive(Clone, Debug)]
pub struct RegimeColumn {
    /// Column index in the column-major design. Column 0 is the intercept and cannot be a regime.
    pub column: usize,
    /// Value written into this column under the active regime, one entry per row.
    pub active: Vec<f64>,
    /// Value written into this column under the control regime, one entry per row.
    pub control: Vec<f64>,
}

/// Lag-aligned windows for one decision-time GLM contrast.
#[derive(Clone, Debug)]
pub struct TemporalContrastDesign {
    x_colmajor: Vec<f64>,
    y: Vec<f64>,
    nrows: usize,
    ncols: usize,
    /// `None` keeps the observed column. `Some` replaces it under each regime.
    regimes: Vec<Option<(Vec<f64>, Vec<f64>)>>,
    lags: usize,
    dependence: TrajectoryDependence,
}

impl TemporalContrastDesign {
    /// Assemble a column-major design.
    ///
    /// Column 0 must be an intercept of ones. At least one other column must stay observed: that
    /// column is the lag held fixed. `regimes` names every column the intervention replaces,
    /// including interactions built from treatment.
    ///
    /// # Errors
    ///
    /// A shape mismatch, a non-binary outcome, a missing lag, a regime on the intercept, or a
    /// non-finite entry.
    pub fn try_new(
        x_colmajor: Vec<f64>,
        y: Vec<f64>,
        nrows: usize,
        regimes: Vec<RegimeColumn>,
        dependence: TrajectoryDependence,
    ) -> Result<Self, EstimationError> {
        if nrows == 0 || y.len() != nrows {
            return Err(shape("the outcome must have one value per row"));
        }
        let Some(ncols) = x_colmajor.len().checked_div(nrows) else {
            return Err(shape("the design length is not a multiple of the row count"));
        };
        if ncols < 2 || nrows * ncols != x_colmajor.len() {
            return Err(shape("a temporal contrast needs an intercept, a regime, and a lag"));
        }
        if !column_is_ones(&x_colmajor[..nrows]) {
            return Err(shape("column 0 is the intercept and must be identically one"));
        }
        binary_outcomes(&y)?;
        finite_column("design", &x_colmajor)?;
        let mut slots: Vec<Option<(Vec<f64>, Vec<f64>)>> = (0..ncols).map(|_| None).collect();
        for regime in regimes {
            if regime.column == 0 || regime.column >= ncols {
                return Err(shape("a regime column must be a non-intercept column of the design"));
            }
            if regime.active.len() != nrows || regime.control.len() != nrows {
                return Err(shape("regime values must have one entry per row"));
            }
            finite_column("active regime", &regime.active)?;
            finite_column("control regime", &regime.control)?;
            if slots[regime.column].is_some() {
                return Err(shape("a regime column was named twice"));
            }
            slots[regime.column] = Some((regime.active, regime.control));
        }
        if !slots.iter().skip(1).any(std::option::Option::is_some) {
            return Err(shape("a temporal contrast needs a column the intervention replaces"));
        }
        let lags = slots.iter().skip(1).filter(|slot| slot.is_none()).count();
        if lags == 0 {
            return Err(shape(
                "a temporal GLM contrast requires a lag column held at its observed value",
            ));
        }
        Ok(Self { x_colmajor, y, nrows, ncols, regimes: slots, lags, dependence })
    }

    const fn refuse_series(&self) -> Result<(), EstimationError> {
        match self.dependence {
            TrajectoryDependence::IndependentSubjects => Ok(()),
            TrajectoryDependence::OneDependentSeries => Err(EstimationError::unsupported(
                "a single dependent series has no independent-trajectory information matrix",
            )),
        }
    }

    #[allow(
        clippy::needless_range_loop,
        reason = "the column index addresses both the design and the coefficient"
    )]
    fn observed_eta(&self, beta: &[f64], row: usize) -> f64 {
        let mut eta = 0.0_f64;
        for column in 0..self.ncols {
            eta += self.x_colmajor[column * self.nrows + row] * beta[column];
        }
        eta
    }

    fn column_value(&self, column: usize, row: usize, active: bool) -> f64 {
        match &self.regimes[column] {
            Some((active_values, control_values)) => {
                if active { active_values[row] } else { control_values[row] }
            }
            None => self.x_colmajor[column * self.nrows + row],
        }
    }

    #[allow(
        clippy::needless_range_loop,
        reason = "the column index addresses both the regime value and the coefficient"
    )]
    fn regime_eta(&self, beta: &[f64], row: usize, active: bool) -> f64 {
        let mut eta = 0.0_f64;
        for column in 0..self.ncols {
            eta += self.column_value(column, row, active) * beta[column];
        }
        eta
    }

    fn mean_contrast(&self, link: TemporalOutcomeLink, beta: &[f64]) -> f64 {
        let mut sum = 0.0_f64;
        for row in 0..self.nrows {
            let active = link.family().mean_from_eta(self.regime_eta(beta, row, true));
            let control = link.family().mean_from_eta(self.regime_eta(beta, row, false));
            sum += active - control;
        }
        sum / self.nrows as f64
    }

    /// Coefficients of regime columns, in increasing column index.
    fn regime_coefficients(&self, beta: &[f64]) -> Vec<f64> {
        self.regimes
            .iter()
            .enumerate()
            .filter_map(|(column, regime)| regime.as_ref().map(|_| beta[column]))
            .collect()
    }

    /// Fisher delta-method standard error of [`Self::mean_contrast`] for independent rows.
    ///
    /// Information is the sum of observed-data Fisher weights. The gradient is the derivative
    /// of the row-averaged regime contrast, so a saturated model reproduces the sampling
    /// variance of the cell-mean contrast.
    #[allow(
        clippy::needless_range_loop,
        reason = "the column index addresses the regime values and the gradient"
    )]
    fn delta_method_se(
        &self,
        link: TemporalOutcomeLink,
        beta: &[f64],
    ) -> Result<f64, EstimationError> {
        let kernel = link.kernel();
        let n = self.nrows;
        let p = self.ncols;
        let mut weights = vec![0.0_f64; n];
        let mut derivative_active = vec![0.0_f64; n];
        let mut derivative_control = vec![0.0_f64; n];
        for row in 0..n {
            let weight = kernel.fisher_weight(self.observed_eta(beta, row), 0.0);
            let active = self.regime_eta(beta, row, true);
            let control = self.regime_eta(beta, row, false);
            derivative_active[row] = kernel.mean_derivative(active);
            derivative_control[row] = kernel.mean_derivative(control);
            if !weight.is_finite()
                || !derivative_active[row].is_finite()
                || !derivative_control[row].is_finite()
            {
                return Err(stats("temporal GLM information or regime predictor is not finite"));
            }
            weights[row] = weight;
        }
        let mut gram = vec![0.0_f64; p * p];
        for j in 0..p {
            for k in j..p {
                let left = &self.x_colmajor[j * n..(j + 1) * n];
                let right = &self.x_colmajor[k * n..(k + 1) * n];
                let mut acc = 0.0_f64;
                for row in 0..n {
                    acc += weights[row] * left[row] * right[row];
                }
                gram[j * p + k] = acc;
                gram[k * p + j] = acc;
            }
        }
        let Some(covariance) = invert_square(&gram, p) else {
            return Err(stats("temporal GLM Fisher information is singular"));
        };
        let mut gradient = vec![0.0_f64; p];
        for row in 0..n {
            for column in 0..p {
                let active = self.column_value(column, row, true);
                let control = self.column_value(column, row, false);
                gradient[column] +=
                    derivative_active[row] * active - derivative_control[row] * control;
            }
        }
        let scale = 1.0 / n as f64;
        for slot in &mut gradient {
            *slot *= scale;
        }
        let mut variance = 0.0_f64;
        for j in 0..p {
            let mut row_sum = 0.0_f64;
            for k in 0..p {
                row_sum += covariance[j * p + k] * gradient[k];
            }
            variance += gradient[j] * row_sum;
        }
        if !(variance.is_finite() && variance >= 0.0) {
            return Err(stats("temporal GLM delta-method variance is not finite"));
        }
        Ok(variance.sqrt())
    }
}

/// Unpenalized maximum-likelihood temporal GLM contrast.
#[derive(Clone, Copy, Debug)]
pub struct FrequentistTemporalLink {
    max_iter: u32,
    tolerance: f64,
}

impl Default for FrequentistTemporalLink {
    fn default() -> Self {
        Self::new()
    }
}

impl FrequentistTemporalLink {
    /// Eighty IRLS iterations and a coefficient tolerance of `1e-12`.
    ///
    /// Separation is a refusal. The fit does not fall back to a ridge penalty.
    #[must_use]
    pub const fn new() -> Self {
        Self { max_iter: 80, tolerance: 1e-12 }
    }

    /// Fit the contrast on independent trajectories.
    ///
    /// # Errors
    ///
    /// A dependent series, an unsupported link, a failed or separated fit, or singular information.
    pub fn fit(
        &self,
        design: &TemporalContrastDesign,
        link: TemporalOutcomeLink,
    ) -> Result<FrequentistLinkContrast, EstimationError> {
        design.refuse_series()?;
        let mut options = GlmOptions::new(self.max_iter, self.tolerance);
        options.ridge_on_separation = None;
        let fit = fit_glm(
            link.family(),
            GlmDesignRef {
                x_colmajor: &design.x_colmajor,
                nrows: design.nrows,
                ncols: design.ncols,
                y: &design.y,
            },
            &FaerBackend,
            &mut LeastSquaresWorkspace::default(),
            &options,
        )?;
        fit.require_ok()?;
        if fit.penalized {
            return Err(EstimationError::unsupported(
                "a ridge penalty is not the temporal GLM contrast",
            ));
        }
        if fit.coefficients.len() != design.ncols {
            return Err(stats("GLM coefficient count does not match the design"));
        }
        let estimate = design.mean_contrast(link, &fit.coefficients);
        if !estimate.is_finite() {
            return Err(stats("temporal GLM contrast is not finite"));
        }
        Ok(FrequentistLinkContrast {
            estimate,
            standard_error: design.delta_method_se(link, &fit.coefficients)?,
            regime_coefficients: design.regime_coefficients(&fit.coefficients),
            link,
            trajectories: design.nrows,
            lags: design.lags,
            iterations: fit.iterations,
        })
    }
}

/// Probability-scale (or mean-scale) contrast and its independent-trajectory standard error.
#[derive(Clone, Debug)]
#[must_use = "the mean-scale contrast is not the treatment coefficient"]
pub struct FrequentistLinkContrast {
    /// Average `μ(active, lag) − μ(control, lag)`.
    pub estimate: f64,
    /// Fisher delta-method standard error under independent trajectories.
    pub standard_error: f64,
    /// Coefficients of the regime columns, in increasing column index.
    pub regime_coefficients: Vec<f64>,
    /// Link that was fit.
    pub link: TemporalOutcomeLink,
    /// Independent subjects.
    pub trajectories: usize,
    /// Lag columns held at their observed values.
    pub lags: usize,
    /// IRLS iterations.
    pub iterations: u32,
}

/// Laplace posterior of the same contrast under a Gaussian coefficient prior.
///
/// `prior_sd` is the standard deviation of `N(0, prior_sd²)` on every coefficient, including
/// the intercept. It is not a ridge penalty.
#[derive(Clone, Copy, Debug)]
pub struct BayesianTemporalLink {
    prior_sd: f64,
    n_draws: usize,
    seed: u64,
}

impl BayesianTemporalLink {
    /// State the prior width, the draw count, and the draw seed.
    ///
    /// # Errors
    ///
    /// A prior standard deviation that is not finite and positive, or a draw count of zero.
    pub fn new(prior_sd: f64, n_draws: usize, seed: u64) -> Result<Self, EstimationError> {
        positive_finite(prior_sd, "Gaussian prior standard deviation must be finite and positive")?;
        if n_draws == 0 {
            return Err(shape("the temporal GLM posterior needs at least one draw"));
        }
        Ok(Self { prior_sd, n_draws, seed })
    }

    /// Draw the posterior of the mean-scale contrast.
    ///
    /// # Errors
    ///
    /// A dependent series, or a Laplace fit that does not publish a posterior.
    #[allow(
        clippy::needless_range_loop,
        reason = "the column index addresses the draw matrix and the coefficient buffer"
    )]
    pub fn fit(
        &self,
        design: &TemporalContrastDesign,
        link: TemporalOutcomeLink,
    ) -> Result<BayesianLinkPosterior, EstimationError> {
        design.refuse_series()?;
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
        let fit = fit_laplace_glm(
            link.likelihood(),
            BayesDesignRef {
                x_colmajor: &design.x_colmajor,
                nrows: design.nrows,
                ncols: design.ncols,
                y: &design.y,
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
            &mut LaplaceWorkspace::default(),
        )?;
        if fit.map.len() != design.ncols {
            return Err(stats("Laplace mode length does not match the design"));
        }
        let functional_at_mode = design.mean_contrast(link, &fit.map);
        let mut beta = vec![0.0_f64; design.ncols];
        let mut draws = Vec::with_capacity(self.n_draws);
        for draw in 0..self.n_draws {
            for column in 0..design.ncols {
                beta[column] = fit.draws.get(draw, column)?;
            }
            let value = design.mean_contrast(link, &beta);
            if !value.is_finite() {
                return Err(stats("a posterior draw of the temporal GLM contrast is not finite"));
            }
            draws.push(value);
        }
        let posterior_mean = draws.iter().sum::<f64>() / self.n_draws as f64;
        Ok(BayesianLinkPosterior {
            posterior_mean,
            functional_at_mode,
            draws,
            prior_sd: self.prior_sd,
            link,
            trajectories: design.nrows,
            lags: design.lags,
        })
    }
}

/// Posterior of the mean-scale contrast. The mean of the draws is not the contrast at the mode.
#[derive(Clone, Debug)]
#[must_use = "the posterior mean of the contrast is not the contrast at the mode"]
pub struct BayesianLinkPosterior {
    /// Mean of [`Self::draws`].
    pub posterior_mean: f64,
    /// Contrast evaluated at the Laplace mode.
    pub functional_at_mode: f64,
    /// One contrast per posterior draw.
    pub draws: Vec<f64>,
    /// Standard deviation of the Gaussian coefficient prior.
    pub prior_sd: f64,
    /// Link that was fit.
    pub link: TemporalOutcomeLink,
    /// Independent subjects.
    pub trajectories: usize,
    /// Lag columns held at their observed values.
    pub lags: usize,
}

fn shape(message: &str) -> EstimationError {
    EstimationError::data_msg(message)
}

fn stats(message: &str) -> EstimationError {
    EstimationError::stats_msg(message)
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

#[allow(
    clippy::float_cmp,
    reason = "Bernoulli outcomes are coded exactly 0 or 1"
)]
fn binary_outcomes(y: &[f64]) -> Result<(), EstimationError> {
    if y.iter().all(|&value| value == 0.0 || value == 1.0) {
        Ok(())
    } else {
        Err(shape("a Bernoulli outcome must be coded 0 or 1"))
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

    fn saturated() -> TemporalContrastDesign {
        let n_cell = 10usize;
        let nrows = n_cell * 4;
        let mut x = vec![0.0_f64; nrows * 4];
        let mut y = vec![0.0_f64; nrows];
        let cells = [(0.0_f64, 0.0_f64, 2usize), (1.0, 0.0, 6), (0.0, 1.0, 4), (1.0, 1.0, 5)];
        let mut lag = vec![0.0_f64; nrows];
        for (cell, &(treatment, level, successes)) in cells.iter().enumerate() {
            for row_in_cell in 0..n_cell {
                let row = cell * n_cell + row_in_cell;
                x[row] = 1.0;
                x[nrows + row] = treatment;
                x[2 * nrows + row] = level;
                x[3 * nrows + row] = treatment * level;
                lag[row] = level;
                y[row] = if row_in_cell < successes { 1.0 } else { 0.0 };
            }
        }
        TemporalContrastDesign::try_new(
            x,
            y,
            nrows,
            vec![
                RegimeColumn {
                    column: 1,
                    active: vec![1.0_f64; nrows],
                    control: vec![0.0_f64; nrows],
                },
                RegimeColumn { column: 3, active: lag, control: vec![0.0_f64; nrows] },
            ],
            TrajectoryDependence::IndependentSubjects,
        )
        .expect("saturated lag design")
    }

    fn cell_mean_se() -> f64 {
        let probabilities = [0.2_f64, 0.6, 0.4, 0.5];
        let sum_var: f64 = probabilities.iter().map(|p| p * (1.0 - p) / 10.0).sum();
        (0.25 * sum_var).sqrt()
    }

    #[test]
    fn identity_link_is_temporal_linear_adjustment() {
        let err = TemporalOutcomeLink::from_family(GlmFamily::GaussianIdentity).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("identity"));
        assert!(message.contains("temporal linear adjustment"));
    }

    #[test]
    fn other_links_are_refused() {
        let err = TemporalOutcomeLink::from_family(GlmFamily::PoissonLog).unwrap_err();
        assert!(!err.to_string().contains("temporal linear adjustment"));
    }

    #[test]
    fn a_design_without_a_lag_is_refused() {
        let nrows = 4usize;
        let x = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let err = TemporalContrastDesign::try_new(
            x,
            vec![0.0, 1.0, 0.0, 1.0],
            nrows,
            vec![RegimeColumn {
                column: 1,
                active: vec![1.0_f64; nrows],
                control: vec![0.0_f64; nrows],
            }],
            TrajectoryDependence::IndependentSubjects,
        )
        .unwrap_err();
        assert!(err.to_string().contains("lag"));
    }

    #[test]
    fn one_dependent_series_is_refused() {
        let mut design = saturated();
        design.dependence = TrajectoryDependence::OneDependentSeries;
        let err = FrequentistTemporalLink::new()
            .fit(&design, TemporalOutcomeLink::BernoulliLogit)
            .unwrap_err();
        assert!(err.to_string().contains("dependent series"));
        let bayes = BayesianTemporalLink::new(1.0, 4, 0).unwrap();
        assert!(bayes.fit(&design, TemporalOutcomeLink::BernoulliLogit).is_err());
    }

    #[test]
    fn separated_sample_is_refused() {
        let nrows = 8usize;
        let mut x = vec![0.0_f64; nrows * 3];
        let mut y = vec![0.0_f64; nrows];
        for row in 0..nrows {
            let treatment = if row < 4 { 0.0 } else { 1.0 };
            x[row] = 1.0;
            x[nrows + row] = treatment;
            x[2 * nrows + row] = if row % 2 == 0 { 0.0 } else { 1.0 };
            y[row] = treatment;
        }
        let design = TemporalContrastDesign::try_new(
            x,
            y,
            nrows,
            vec![RegimeColumn {
                column: 1,
                active: vec![1.0_f64; nrows],
                control: vec![0.0_f64; nrows],
            }],
            TrajectoryDependence::IndependentSubjects,
        )
        .unwrap();
        let err = FrequentistTemporalLink::new()
            .fit(&design, TemporalOutcomeLink::BernoulliLogit)
            .unwrap_err();
        let message = err.to_string();
        assert!(!message.contains("ridge"), "{message}");
        assert!(
            message.contains("separation")
                || message.contains("converge")
                || message.contains("1e-8"),
            "{message}"
        );
    }

    #[test]
    fn saturated_logit_contrast_matches_the_cell_mean_difference() {
        let fit = FrequentistTemporalLink::new()
            .fit(&saturated(), TemporalOutcomeLink::BernoulliLogit)
            .unwrap();
        assert!((fit.estimate - 0.25).abs() < 1e-8);
        assert!((fit.regime_coefficients[0] - 6.0_f64.ln()).abs() < 1e-6);
        assert!((fit.regime_coefficients[0] - fit.estimate).abs() > 1.0);
        assert!((fit.standard_error - cell_mean_se()).abs() < 1e-5);
        assert_eq!(fit.lags, 1);
        assert_eq!(fit.trajectories, 40);
    }

    #[test]
    fn saturated_probit_contrast_matches_the_same_probability_scale() {
        let design = saturated();
        let logit = FrequentistTemporalLink::new()
            .fit(&design, TemporalOutcomeLink::BernoulliLogit)
            .unwrap();
        let probit = FrequentistTemporalLink::new()
            .fit(&design, TemporalOutcomeLink::BernoulliProbit)
            .unwrap();
        assert!((probit.estimate - 0.25).abs() < 1e-8);
        assert!((probit.standard_error - cell_mean_se()).abs() < 1e-5);
        assert!((probit.regime_coefficients[0] - logit.regime_coefficients[0]).abs() > 0.2);
    }

    #[test]
    fn tight_prior_shrinks_the_contrast_toward_zero() {
        let posterior = BayesianTemporalLink::new(0.05, 32, 0)
            .unwrap()
            .fit(&saturated(), TemporalOutcomeLink::BernoulliLogit)
            .unwrap();
        assert!(posterior.posterior_mean < 0.05);
        assert!(posterior.functional_at_mode < 0.05);
        assert!((posterior.prior_sd - 0.05).abs() < 1e-15);
        let mean = posterior.draws.iter().sum::<f64>() / posterior.draws.len() as f64;
        assert!((mean - posterior.posterior_mean).abs() < 1e-12);
    }

    #[test]
    fn posterior_mean_is_not_the_maximum_likelihood_contrast() {
        let frequentist = FrequentistTemporalLink::new()
            .fit(&saturated(), TemporalOutcomeLink::BernoulliLogit)
            .unwrap();
        let posterior = BayesianTemporalLink::new(1.0, 200, 11)
            .unwrap()
            .fit(&saturated(), TemporalOutcomeLink::BernoulliLogit)
            .unwrap();
        assert!((posterior.posterior_mean - frequentist.estimate).abs() > 0.01);
        assert_ne!(posterior.posterior_mean.to_bits(), posterior.functional_at_mode.to_bits());
    }
}
