//! Optional posterior wire forms for categorical treatment artifacts.

use super::{CATEGORICAL_TREATMENT_CALIBRATION, MonotonicityDirectionWire};
use antecedent_estimate::categorical_treatment_posterior::{
    CATEGORICAL_POSTERIOR_ESTIMAND, CATEGORICAL_POSTERIOR_LIKELIHOOD,
    CategoricalTreatmentPosterior, ContrastPosterior,
};
use serde::{Deserialize, Serialize};

/// The declared prior and credible level of the optional posterior section.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalPosteriorDeclaration {
    /// Prior precision (relative to the error variance) of each level coefficient.
    pub level_precision: f64,
    /// Prior precision of the intercept and every adjustment coefficient.
    pub nuisance_precision: f64,
    /// Inverse-Gamma shape of the error variance.
    pub variance_shape: f64,
    /// Inverse-Gamma scale of the error variance.
    pub variance_scale: f64,
    /// Credible level in `(0, 1)` of every interval.
    pub credible_level: f64,
}

/// Student-t posterior of one level or pairwise contrast, on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PosteriorContrastWire {
    /// Baseline level.
    pub from: String,
    /// Compared level.
    pub to: String,
    /// Student-t location (the posterior mean).
    pub location: f64,
    /// Student-t scale.
    pub scale: f64,
    /// Student-t degrees of freedom.
    pub degrees_of_freedom: f64,
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation; absent when the degrees of freedom are at most two.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_deviation: Option<f64>,
    /// Equal-tailed credible interval at the declared level; the consumer verifies it against
    /// the replay, and an artifact without it is malformed.
    pub interval: [f64; 2],
    /// Posterior probability that the contrast is positive.
    pub probability_positive: f64,
}

/// Posterior probability that every adjacent step moves in the declared direction, on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PosteriorMonotonicityWire {
    /// Direction of the declared monotonicity.
    pub direction: MonotonicityDirectionWire,
    /// Per-step marginal posterior probability of lying in the declared direction.
    pub step_probabilities: Vec<f64>,
    /// Lower (Bonferroni) bound of the joint probability.
    pub lower_bound: f64,
    /// Upper (Frechet) bound of the joint probability.
    pub upper_bound: f64,
}

/// The optional posterior section: a Bayesian counterpart of the stored fit under a declared
/// prior, recomputed from the stored design on consumption. Its estimand is the superpopulation
/// dummy coefficient contrast of the stated linear model; it is unsupported and its calibration is
/// `not_measured`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalPosteriorWire {
    /// The declared prior and level.
    pub declaration: CategoricalPosteriorDeclaration,
    /// The likelihood the posterior is exact under.
    pub likelihood: String,
    /// What the contrasts are the contrasts of.
    pub estimand: String,
    /// Always `not_measured`.
    pub calibration: String,
    /// Always `false`: no support covers the posterior.
    pub supported: bool,
    /// Posterior Inverse-Gamma shape of the error variance.
    pub variance_shape: f64,
    /// Posterior Inverse-Gamma scale of the error variance.
    pub variance_scale: f64,
    /// Posterior mean of the dummy coefficients, non-reference levels in canonical order.
    pub coefficient_means: Vec<f64>,
    /// Per-level contrasts against the reference.
    pub level_contrasts: Vec<PosteriorContrastWire>,
    /// Requested pairwise contrasts, in request order.
    pub pairwise: Vec<PosteriorContrastWire>,
    /// Declared monotonicity statement, when requested.
    pub monotonicity: Option<PosteriorMonotonicityWire>,
}

fn contrast_wire(c: &ContrastPosterior) -> PosteriorContrastWire {
    PosteriorContrastWire {
        from: c.from.clone(),
        to: c.to.clone(),
        location: c.posterior.location,
        scale: c.posterior.scale,
        degrees_of_freedom: c.posterior.degrees_of_freedom,
        mean: c.mean,
        standard_deviation: c.standard_deviation.is_finite().then_some(c.standard_deviation),
        interval: c.interval,
        probability_positive: c.probability_positive,
    }
}

pub(super) fn posterior_section(
    declaration: &CategoricalPosteriorDeclaration,
    posterior: &CategoricalTreatmentPosterior,
) -> CategoricalPosteriorWire {
    CategoricalPosteriorWire {
        declaration: *declaration,
        likelihood: CATEGORICAL_POSTERIOR_LIKELIHOOD.to_owned(),
        estimand: CATEGORICAL_POSTERIOR_ESTIMAND.to_owned(),
        calibration: CATEGORICAL_TREATMENT_CALIBRATION.to_owned(),
        supported: false,
        variance_shape: posterior.variance_shape_and_scale.0,
        variance_scale: posterior.variance_shape_and_scale.1,
        coefficient_means: posterior.coefficient_means.clone(),
        level_contrasts: posterior.level_contrasts.iter().map(contrast_wire).collect(),
        pairwise: posterior.pairwise.iter().map(contrast_wire).collect(),
        monotonicity: posterior.monotonicity.as_ref().map(|m| PosteriorMonotonicityWire {
            direction: m.direction.into(),
            step_probabilities: m.step_probabilities.clone(),
            lower_bound: m.lower_bound,
            upper_bound: m.upper_bound,
        }),
    }
}
