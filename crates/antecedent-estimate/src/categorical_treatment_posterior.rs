//! Bayesian counterpart of the dummy-coded categorical-treatment regression.
//!
//! # Target
//!
//! The same coefficients as [`crate::categorical_treatment`]: in
//! `y = b0 + A g + sum_{L != ref} b_L 1[level = L] + e` the effect of level `L` against the
//! declared reference is `b_L`, and the effect of `to` relative to `from` is `b_to - b_from`.
//! The frequentist route reports these with a (possibly robust) sandwich covariance and Holm
//! multiplicity; this route reports their posterior.
//!
//! # Likelihood and prior
//!
//! Gaussian homoskedastic errors, `e ~ N(0, sigma2)`, with the prior
//! `b_j | sigma2 ~ N(0, sigma2 / precision_j)` and `sigma2 ~ InvGamma(a0, b0)`. The prior is
//! declared by the caller and has no default: a prior alone never identifies a level contrast,
//! the rows do, and the prior only regularizes. Every contrast `w' b` then has an exact
//! Student-t posterior ([`crate::gaussian_linear_posterior`]).
//!
//! # Estimand
//!
//! The estimand is the superpopulation coefficient vector of the stated linear model, i.e. the
//! covariate-adjusted conditional level contrast. It is not the finite-population average of the
//! row-level potential-outcome differences of the experiment's own rows, which is a different
//! target under heterogeneity (a randomization-design interval for it is not a posterior and is
//! not relabelled one). The two coincide when the level effects are constant across rows and
//! the linear model is correct; with heterogeneous effects the dummy coefficient is a
//! variance-weighted average over covariate strata, and the posterior is for that weighted
//! average. The covariance kind of the frequentist spec is not read: the model-based likelihood
//! is the posterior's, and a robust sandwich has no likelihood.
//!
//! # Monotonicity
//!
//! For ordered levels the posterior probability that every adjacent step lies in the declared
//! direction is reported as the interval `[max(0, 1 - sum_j P(step_j violates)), min_j
//! P(step_j satisfies)]` (Bonferroni and Frechet bounds, valid under any dependence of the
//! steps); the bounds coincide for a single step. A point value would need the joint
//! multivariate-t probability of an orthant, which is not computed here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_code;
use antecedent_stats::invert_square;

use crate::categorical_treatment::{
    CategoricalSummary, CategoricalTreatmentInput, CategoricalTreatmentSpec, LevelCount,
    MonotonicityDirection, summarize_categorical_design, validate_spec,
};
use crate::error::EstimationError;
use crate::gaussian_linear_posterior::{
    GaussianLinearPosterior, GaussianLinearPrior, StudentPosterior, conjugate_posterior_from_residuals,
};
use crate::util::refuse;
use crate::vector_treatment::DesignSummary;

/// Likelihood of the posterior, for evidence records.
pub const CATEGORICAL_POSTERIOR_LIKELIHOOD: &str = "gaussian_homoskedastic_dummy_regression";

/// Estimand of the posterior, for evidence records.
pub const CATEGORICAL_POSTERIOR_ESTIMAND: &str =
    "superpopulation_dummy_coefficient_contrast_under_the_stated_linear_model";

/// Declared prior of the categorical posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CategoricalPosteriorPrior {
    /// Prior precision (relative to `sigma2`) of each level coefficient; positive.
    pub level_precision: f64,
    /// Prior precision of the intercept and every adjustment coefficient; positive.
    pub nuisance_precision: f64,
    /// Inverse-Gamma shape of `sigma2`; positive.
    pub variance_shape: f64,
    /// Inverse-Gamma scale of `sigma2`; positive.
    pub variance_scale: f64,
}

/// Posterior of one contrast `effect(to) - effect(from)`.
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastPosterior {
    /// Baseline level.
    pub from: String,
    /// Compared level.
    pub to: String,
    /// Marginal Student-t posterior.
    pub posterior: StudentPosterior,
    /// Posterior mean (the location; the degrees of freedom exceed one for any admitted prior).
    pub mean: f64,
    /// Posterior standard deviation; `NaN` when the degrees of freedom are at most two.
    pub standard_deviation: f64,
    /// Equal-tailed credible interval at the requested level.
    pub interval: [f64; 2],
    /// Posterior probability that the contrast is positive.
    pub probability_positive: f64,
}

/// Posterior probability that every adjacent step moves in the declared direction.
#[derive(Clone, Debug, PartialEq)]
pub struct MonotonicityPosterior {
    /// Declared direction.
    pub direction: MonotonicityDirection,
    /// Per-step marginal posterior probability of lying in the declared direction.
    pub step_probabilities: Vec<f64>,
    /// Lower bound of the joint probability, `max(0, 1 - sum_j (1 - p_j))`.
    pub lower_bound: f64,
    /// Upper bound of the joint probability, `min_j p_j`.
    pub upper_bound: f64,
}

/// The categorical posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalTreatmentPosterior {
    /// Reference level.
    pub reference: String,
    /// Canonical level order.
    pub level_order: Vec<String>,
    /// Rows per level, in canonical order.
    pub counts: Vec<LevelCount>,
    /// Posterior mean of the dummy coefficients, non-reference levels in canonical order.
    pub coefficient_means: Vec<f64>,
    /// Per-level contrasts against the reference, in canonical order.
    pub level_contrasts: Vec<ContrastPosterior>,
    /// Requested pairwise contrasts, in request order.
    pub pairwise: Vec<ContrastPosterior>,
    /// Declared monotonicity statement, when requested.
    pub monotonicity: Option<MonotonicityPosterior>,
    /// Posterior Inverse-Gamma shape and scale of the error variance.
    pub variance_shape_and_scale: (f64, f64),
    /// Credible level of every interval.
    pub credible_level: f64,
    /// Always [`CATEGORICAL_POSTERIOR_LIKELIHOOD`].
    pub likelihood: &'static str,
    /// Always [`CATEGORICAL_POSTERIOR_ESTIMAND`].
    pub estimand: &'static str,
}

struct Coding<'a> {
    nonreference: Vec<&'a String>,
    first_treatment: usize,
    width: usize,
}

impl Coding<'_> {
    fn weights(&self, from: &str, to: &str, reference: &str) -> Vec<f64> {
        let mut weights = vec![0.0; self.width];
        for (level, sign) in [(to, 1.0), (from, -1.0)] {
            let position = self.nonreference.iter().position(|l| l.as_str() == level);
            if let (true, Some(position)) = (level != reference, position) {
                weights[self.first_treatment + position] += sign;
            }
        }
        weights
    }
}

fn summarize(
    posterior: &GaussianLinearPosterior,
    weights: &[f64],
    names: (&str, &str),
    level: f64,
) -> Result<ContrastPosterior, EstimationError> {
    let student = posterior.linear_functional(weights).ok_or_else(|| {
        refuse(
            reason_code!("invalid_argument"),
            "categorical_treatment_posterior.degenerate_contrast",
            "the contrast has zero posterior scale",
        )
    })?;
    Ok(ContrastPosterior {
        from: names.0.to_string(),
        to: names.1.to_string(),
        mean: student.location,
        standard_deviation: student.standard_deviation(),
        interval: student.equal_tailed_interval(level),
        probability_positive: student.probability_above(0.0),
        posterior: student,
    })
}

/// Posterior of the dummy-coded coefficients from the stored sufficient statistics.
fn posterior_from_summary(
    design: &DesignSummary,
    prior: &CategoricalPosteriorPrior,
) -> Result<GaussianLinearPosterior, EstimationError> {
    let width = design.xty.len();
    // |y - X m|^2 = RSS + (m - b)' G (m - b) for the OLS fit b = G^-1 X'y that produced the stored
    // RSS (the OLS residual is orthogonal to the columns of X). Using it avoids the cancellation
    // in y'y - m . X'y when the outcome sits far from zero.
    let gram_inverse = invert_square(&design.gram, width).ok_or_else(|| {
        refuse(
            reason_code!("invalid_argument"),
            "gaussian_linear_posterior.singular_precision",
            "the stored Gram matrix is singular",
        )
    })?;
    let ols: Vec<f64> = (0..width)
        .map(|i| (0..width).map(|j| gram_inverse[i * width + j] * design.xty[j]).sum::<f64>())
        .collect();
    let precision = (0..width)
        .map(|j| {
            if j >= design.first_treatment {
                prior.level_precision
            } else {
                prior.nuisance_precision
            }
        })
        .collect();
    let model_prior = GaussianLinearPrior {
        coefficient_precision: precision,
        variance_shape: prior.variance_shape,
        variance_scale: prior.variance_scale,
    };
    conjugate_posterior_from_residuals(
        &design.gram,
        &design.xty,
        design.n_rows,
        &model_prior,
        |mean| {
            let shift: Vec<f64> = mean.iter().zip(&ols).map(|(m, b)| m - b).collect();
            let curvature: f64 = (0..width)
                .map(|i| {
                    shift[i] * (0..width).map(|j| design.gram[i * width + j] * shift[j]).sum::<f64>()
                })
                .sum();
            design.rss + curvature
        },
    )
}

/// Per-step and joint-bound posterior probabilities of the declared monotone direction.
fn monotonicity_posterior(
    direction: MonotonicityDirection,
    order: &[String],
    (coding, posterior): (&Coding<'_>, &GaussianLinearPosterior),
    reference: &str,
    credible_level: f64,
) -> Result<MonotonicityPosterior, EstimationError> {
    let mut probabilities = Vec::with_capacity(order.len().saturating_sub(1));
    for window in order.windows(2) {
        let weights = coding.weights(&window[0], &window[1], reference);
        let step = summarize(
            posterior,
            &weights,
            (window[0].as_str(), window[1].as_str()),
            credible_level,
        )?;
        let up = step.probability_positive;
        probabilities.push(match direction {
            MonotonicityDirection::NonDecreasing => up,
            MonotonicityDirection::NonIncreasing => 1.0 - up,
        });
    }
    let violation: f64 = probabilities.iter().map(|p| 1.0 - p).sum();
    Ok(MonotonicityPosterior {
        direction,
        lower_bound: (1.0 - violation).max(0.0),
        upper_bound: probabilities.iter().copied().fold(1.0, f64::min),
        step_probabilities: probabilities,
    })
}

/// Posterior of the declared categorical-treatment contrasts.
///
/// # Errors
///
/// The declared-design and level-count refusals of
/// [`crate::categorical_treatment::fit_categorical_treatment`];
/// `invalid_argument` with `categorical_treatment_posterior.invalid_level` when the credible
/// level is outside `(0, 1)`, and with the `gaussian_linear_posterior.*` details for an invalid
/// prior or a singular posterior precision.
pub fn posterior_categorical_treatment(
    input: &CategoricalTreatmentInput,
    spec: &CategoricalTreatmentSpec,
    prior: &CategoricalPosteriorPrior,
    credible_level: f64,
) -> Result<CategoricalTreatmentPosterior, EstimationError> {
    check_level(credible_level)?;
    validate_spec(spec)?;
    let summary = summarize_categorical_design(input, spec)?;
    posterior_categorical_from_summary(&summary, spec, prior, credible_level)
}

fn check_level(credible_level: f64) -> Result<(), EstimationError> {
    if credible_level.is_finite() && credible_level > 0.0 && credible_level < 1.0 {
        return Ok(());
    }
    Err(refuse(
        reason_code!("invalid_argument"),
        "categorical_treatment_posterior.invalid_level",
        "the credible level must lie strictly between zero and one",
    ))
}

/// Posterior of the declared contrasts from the stored sufficient statistics of the dummy-coded
/// design, as an artifact holds them: no rows are needed, and the result equals
/// [`posterior_categorical_treatment`] on the rows the summary came from.
///
/// # Errors
///
/// As [`posterior_categorical_treatment`], for the declared design and the prior.
pub fn posterior_categorical_from_summary(
    summary: &CategoricalSummary,
    spec: &CategoricalTreatmentSpec,
    prior: &CategoricalPosteriorPrior,
    credible_level: f64,
) -> Result<CategoricalTreatmentPosterior, EstimationError> {
    check_level(credible_level)?;
    let order = validate_spec(spec)?;
    let design = &summary.design;
    let width = design.xty.len();
    let coding = Coding {
        nonreference: order.iter().filter(|l| **l != spec.reference).collect(),
        first_treatment: design.first_treatment,
        width,
    };
    let posterior = posterior_from_summary(design, prior)?;
    let reference = spec.reference.as_str();
    let level_contrasts = coding
        .nonreference
        .iter()
        .map(|level| {
            let weights = coding.weights(reference, level, reference);
            summarize(&posterior, &weights, (reference, level.as_str()), credible_level)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let pairwise = spec
        .pairwise
        .iter()
        .map(|(from, to)| {
            let weights = coding.weights(from, to, reference);
            summarize(&posterior, &weights, (from.as_str(), to.as_str()), credible_level)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let monotonicity = spec
        .monotonicity
        .map(|direction| {
            monotonicity_posterior(
                direction,
                &order,
                (&coding, &posterior),
                reference,
                credible_level,
            )
        })
        .transpose()?;
    Ok(CategoricalTreatmentPosterior {
        reference: spec.reference.clone(),
        level_order: order,
        counts: summary.counts.clone(),
        coefficient_means: posterior.mean()[design.first_treatment..].to_vec(),
        level_contrasts,
        pairwise,
        monotonicity,
        variance_shape_and_scale: posterior.variance_shape_and_scale(),
        credible_level,
        likelihood: CATEGORICAL_POSTERIOR_LIKELIHOOD,
        estimand: CATEGORICAL_POSTERIOR_ESTIMAND,
    })
}
