//! Known-variance Bayesian basis regression: the outcome-mechanism model object of the
//! learned joint source-target transport row (2.3A cell X4 remainder, B1 model provider).
//!
//! `antecedent-learn` owns the **basis declaration** ([`PolynomialBasis`]) and the **fitted
//! Gaussian model** ([`KnownVarianceBasisRegression`]): a dense Gaussian coefficient prior,
//! a per-row known noise precision, and the exact conjugate posterior computed by the
//! `antecedent-prob` Gaussian backend (known residual variance, dense prior correlation).
//! The posterior is exactly Gaussian, so it is reported as an exact mean and covariance
//! together with iid draws from that Gaussian; there is no sampler to diagnose.
//!
//! The module only fits `y_i ~ Normal(x_i' beta, 1 / precision_i)` with `beta ~ N(m0, S0)`.
//! Which columns the design holds (treatment interactions, population-varying blocks) and
//! which estimand is read off the posterior belong to the caller.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::needless_range_loop,
    reason = "small dense Gram algebra indexes several buffers by one position"
)]

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, CoefficientCorrelation,
    ConjugateGaussianBackend, GaussianCoefficientPrior, InferenceBackend, LaplaceWorkspace,
    PriorSet, PriorSpec,
};

use crate::{LearnError, LearnerProvenance};

/// Highest polynomial degree the basis accepts.
pub const BASIS_MAX_DEGREE: usize = 6;
/// Relative pivot at or below which the weighted Gram matrix counts as rank deficient.
pub const BASIS_PIVOT_TOLERANCE: f64 = 1e-10;

/// Per-covariate polynomial basis `x_j, x_j^2, ..., x_j^degree` (no cross terms, no
/// constant: the intercept is the caller's).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolynomialBasis {
    degree: usize,
}

impl PolynomialBasis {
    /// Declare a basis of the given degree, `1..=`[`BASIS_MAX_DEGREE`].
    ///
    /// # Errors
    /// [`LearnError::Shape`] for a degree outside the supported range.
    pub fn new(degree: usize) -> Result<Self, LearnError> {
        if degree == 0 || degree > BASIS_MAX_DEGREE {
            return Err(LearnError::Shape { message: "polynomial basis degree must be 1..=6" });
        }
        Ok(Self { degree })
    }

    /// Declared degree.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Number of basis terms for `n_covariates` covariates.
    #[must_use]
    pub const fn width(&self, n_covariates: usize) -> usize {
        n_covariates * self.degree
    }

    /// Stable identity of the declaration.
    #[must_use]
    pub fn id(&self) -> String {
        format!("polynomial_degree_{}", self.degree)
    }

    /// Evaluate the basis at one covariate row; `out` is cleared and refilled in the order
    /// `x_0, x_0^2, .., x_1, x_1^2, ..`.
    pub fn evaluate(&self, x: &[f64], out: &mut Vec<f64>) {
        out.clear();
        for value in x {
            let mut power = 1.0;
            for _ in 0..self.degree {
                power *= value;
                out.push(power);
            }
        }
    }

    /// Term names in [`Self::evaluate`] order: `<name>` then `<name>^k`.
    #[must_use]
    pub fn term_names(&self, covariate_names: &[String]) -> Vec<String> {
        let mut names = Vec::with_capacity(self.width(covariate_names.len()));
        for name in covariate_names {
            names.push(name.clone());
            for k in 2..=self.degree {
                names.push(format!("{name}^{k}"));
            }
        }
        names
    }
}

/// Borrowed column-major design with per-row known noise precision.
#[derive(Clone, Copy, Debug)]
pub struct BasisDesign<'a> {
    /// Column-major design, `nrows * ncols` entries.
    pub x_colmajor: &'a [f64],
    /// Rows.
    pub nrows: usize,
    /// Columns (coefficients).
    pub ncols: usize,
    /// Outcome per row.
    pub y: &'a [f64],
    /// Known noise precision `1 / sigma_i^2` per row, positive.
    pub precision_weights: &'a [f64],
}

/// Declared Gaussian prior and draw request.
#[derive(Clone, Copy, Debug)]
pub struct KnownVarianceBasisSpec<'a> {
    /// Prior mean, one entry per coefficient.
    pub prior_mean: &'a [f64],
    /// Row-major symmetric positive definite prior covariance (absolute, not scaled).
    pub prior_covariance: &'a [f64],
    /// Posterior draws to retain, at least one.
    pub n_draws: usize,
    /// Seed of the deterministic draw stream.
    pub seed: u64,
}

/// Exact Gaussian posterior of a fitted basis regression.
#[derive(Clone, Debug, PartialEq)]
pub struct BasisRegressionPosterior {
    /// Exact posterior mean.
    pub mean: Vec<f64>,
    /// Exact posterior covariance, row-major `p * p`.
    pub covariance: Vec<f64>,
    /// Draws, coefficient-major: coefficient `c` occupies `c * n_draws .. (c + 1) * n_draws`.
    pub draws: Arc<[f64]>,
    /// Number of draws.
    pub n_draws: usize,
    /// Condition lower bound of the equilibrated posterior precision.
    pub precision_condition: f64,
    /// Stable model identity, including the declared noise model.
    pub model_id: Arc<str>,
    /// Implementation provenance.
    pub provenance: LearnerProvenance,
}

impl BasisRegressionPosterior {
    /// Number of coefficients.
    #[must_use]
    pub fn n_coefficients(&self) -> usize {
        self.mean.len()
    }

    /// One draw of one coefficient, or `None` out of range.
    #[must_use]
    pub fn draw(&self, draw: usize, coefficient: usize) -> Option<f64> {
        if draw >= self.n_draws || coefficient >= self.mean.len() {
            return None;
        }
        self.draws.get(coefficient * self.n_draws + draw).copied()
    }
}

/// Known-variance conjugate Gaussian basis regression (the learned mechanism object).
#[derive(Clone, Copy, Debug, Default)]
pub struct KnownVarianceBasisRegression;

impl KnownVarianceBasisRegression {
    /// Stable estimator identity.
    #[must_use]
    pub const fn estimator_id(&self) -> &'static str {
        "bayesian.basis_known_variance"
    }

    /// Fit the regression exactly and draw from its Gaussian posterior.
    ///
    /// The per-row precisions enter as likelihood weights of a unit-variance Gaussian, so
    /// the posterior precision is `S0^-1 + X' W X` with `W = diag(precision_weights)` and
    /// the prior covariance is used as declared.
    ///
    /// # Errors
    /// [`LearnError::Shape`] for inconsistent dimensions or a non-positive precision,
    /// [`LearnError::Probability`] for an invalid prior or a failed solve, and
    /// [`LearnError::Backend`] when the posterior precision is too ill-conditioned to publish.
    pub fn fit(
        &self,
        design: BasisDesign<'_>,
        spec: &KnownVarianceBasisSpec<'_>,
        ctx: &ExecutionContext,
    ) -> Result<BasisRegressionPosterior, LearnError> {
        let (n, p) = (design.nrows, design.ncols);
        validate(&design, spec)?;
        let variance: Vec<f64> = (0..p).map(|i| spec.prior_covariance[i * p + i]).collect();
        let mut specs = vec![
            PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
                mean: Arc::from(spec.prior_mean.to_vec()),
                variance: Arc::from(variance),
            }),
            PriorSpec::KnownResidualVariance(1.0),
        ];
        if let Some(correlation) =
            CoefficientCorrelation::from_covariance(spec.prior_covariance, p)?
        {
            specs.push(PriorSpec::CoefficientCorrelation(correlation));
        }
        let prior = PriorSet { specs, ..PriorSet::default() };
        let fit = ConjugateGaussianBackend.fit(
            BayesLikelihood::GaussianIdentity,
            BayesDesignRef {
                x_colmajor: design.x_colmajor,
                nrows: n,
                ncols: p,
                y: design.y,
                weights: Some(design.precision_weights),
                offsets: None,
            },
            &prior,
            &BayesFitOptions { n_draws: spec.n_draws, seed: spec.seed, ..Default::default() },
            &mut LaplaceWorkspace::default(),
            ctx,
        )?;
        if !fit.diagnostics.allows_posterior() {
            return Err(LearnError::Backend(format!(
                "posterior precision condition {} exceeds the publication ceiling",
                fit.diagnostics.hessian_condition
            )));
        }
        let covariance = fit.cov.ok_or_else(|| {
            LearnError::Backend("the known-variance fit returned no exact covariance".to_owned())
        })?;
        let mut draws = Vec::with_capacity(spec.n_draws * p);
        for coefficient in 0..p {
            draws.extend_from_slice(fit.draws.column(coefficient)?);
        }
        Ok(BasisRegressionPosterior {
            mean: fit.map,
            covariance,
            draws: Arc::from(draws),
            n_draws: spec.n_draws,
            precision_condition: fit.diagnostics.hessian_condition,
            model_id: Arc::from("bayesian_basis.known_variance_gaussian"),
            provenance: LearnerProvenance {
                spec: "bayesian_basis_known_variance".into(),
                implementation: "antecedent_prob::conjugate_gaussian".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
        })
    }
}

fn validate(design: &BasisDesign<'_>, spec: &KnownVarianceBasisSpec<'_>) -> Result<(), LearnError> {
    let (n, p) = (design.nrows, design.ncols);
    if n == 0
        || p == 0
        || design.x_colmajor.len() != n.saturating_mul(p)
        || design.y.len() != n
        || design.precision_weights.len() != n
        || spec.prior_mean.len() != p
        || spec.prior_covariance.len() != p.saturating_mul(p)
    {
        return Err(LearnError::Shape {
            message: "basis regression design, weights and prior dimensions disagree",
        });
    }
    if spec.n_draws == 0 {
        return Err(LearnError::Shape { message: "basis regression needs at least one draw" });
    }
    let finite = design.x_colmajor.iter().chain(design.y).all(|v| v.is_finite());
    let precisions = design.precision_weights.iter().all(|w| w.is_finite() && *w > 0.0);
    if !finite || !precisions {
        return Err(LearnError::Shape {
            message: "basis regression needs finite data and positive finite noise precisions",
        });
    }
    Ok(())
}

/// Rank diagnostics of the weighted information matrix `X' W X`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InformationRank {
    /// Smallest relative Cholesky pivot (negative or zero when rank deficient).
    pub min_pivot_ratio: f64,
    /// First column whose pivot is at or below [`BASIS_PIVOT_TOLERANCE`], if any.
    pub deficient_column: Option<usize>,
}

/// Weighted Gram rank check of a design, independent of any prior.
///
/// A proper prior makes the posterior exist for any design, so a design whose data
/// information is singular would otherwise be answered by the prior alone; the caller
/// refuses it instead.
///
/// # Errors
/// [`LearnError::Shape`] for inconsistent dimensions.
pub fn information_rank(design: &BasisDesign<'_>) -> Result<InformationRank, LearnError> {
    let (n, p) = (design.nrows, design.ncols);
    if n == 0
        || p == 0
        || design.x_colmajor.len() != n.saturating_mul(p)
        || design.precision_weights.len() != n
    {
        return Err(LearnError::Shape { message: "information rank design dimensions disagree" });
    }
    let mut gram = vec![0.0; p * p];
    for i in 0..p {
        for j in i..p {
            let mut acc = 0.0;
            for r in 0..n {
                acc += design.precision_weights[r]
                    * design.x_colmajor[i * n + r]
                    * design.x_colmajor[j * n + r];
            }
            gram[i * p + j] = acc;
            gram[j * p + i] = acc;
        }
    }
    let mut factor = vec![0.0; p * p];
    let mut smallest = f64::INFINITY;
    for j in 0..p {
        let mut pivot = gram[j * p + j];
        for k in 0..j {
            pivot -= factor[j * p + k] * factor[j * p + k];
        }
        let ratio = pivot / gram[j * p + j].abs().max(f64::MIN_POSITIVE);
        smallest = smallest.min(ratio);
        if ratio.is_nan() || ratio <= BASIS_PIVOT_TOLERANCE {
            return Ok(InformationRank { min_pivot_ratio: ratio, deficient_column: Some(j) });
        }
        let root = pivot.sqrt();
        factor[j * p + j] = root;
        for i in (j + 1)..p {
            let mut s = gram[i * p + j];
            for k in 0..j {
                s -= factor[i * p + k] * factor[j * p + k];
            }
            factor[i * p + j] = s / root;
        }
    }
    Ok(InformationRank { min_pivot_ratio: smallest, deficient_column: None })
}
