//! The exact posterior of the arm means of an adaptive trial whose units contribute several
//! correlated measurements, under a declared within-unit dependence.
//!
//! # The model and the target
//!
//! A unit assigned to arm `k` contributes `m` measurements `y_i ~ N(mu_k 1, sigma2 R)`, where
//! `sigma2` and the correlation `R` (compound symmetric, autoregressive, or a unit-level random
//! effect) are the declared [`OutcomeStructure`] of the frequentist route in
//! [`super::procedure_repeated_measures`]. Units are independent given the arm means. The
//! arm means have independent `N(prior_mean_k, prior_variance_k)` priors. The target is the
//! same as the frequentist final analysis': the contrast `mu_plus - mu_minus` of arm means,
//! the population mean measurement of each arm.
//!
//! The generalized-least-squares unit summary `ybar_i = w' y_i` (weights `w = R^-1 1 / 1' R^-1 1`)
//! is sufficient for `mu_k` and has variance `v = sigma2 / (1' R^-1 1)`, so with `n_k` units
//! in arm `k`
//!
//! ```text
//! precision_k = 1 / prior_variance_k + n_k / v
//! mean_k      = (prior_mean_k / prior_variance_k + sum_i ybar_i / v) / precision_k
//! ```
//!
//! The arms are independent a posteriori, so the contrast is exactly normal with the difference
//! of means and the sum of variances. The variance and the correlation are declared, not
//! estimated: a posterior that marginalized over them needs a prior on them and is a separate
//! model.
//!
//! # Adaptation
//!
//! If the assignment of unit `i` depends on the data of earlier units only (and on independent
//! randomization), the probability of the realized assignment is a function of observed
//! quantities and factors out of the likelihood of the arm means (the likelihood principle).
//! The posterior above is therefore the posterior given the adaptively collected data with no
//! correction for the adaptation, and it is the same under any such policy. That statement is a
//! property of the *posterior*; it is not inherited by the frequentist analysis, which must be
//! calibrated to the whole procedure (the fixed-sample analysis after adaptive assignment is
//! refused by [`super::procedure_repeated_measures::admit_repeated_measures`], and the
//! procedure-derived uncertainty is the route for calibration). A policy that depends on an
//! unobserved quantity ([`SelectionMechanism::DependsOnUnobserved`]) breaks the premise and is
//! refused. The recorded assignment probabilities are returned in
//! [`SelectionEvidence`] unchanged and unused, so later synthesis can see what the assignment
//! mechanism was.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::norm_cdf;

use super::AdaptiveInferenceError as Error;
use super::procedure_contract::MAX_PROCEDURE_ARMS;
use super::procedure_repeated_measures::{OutcomeStructure, gls_unit_summary};
use crate::special::normal_ppf;

/// Most units one posterior accepts.
pub const MAX_POSTERIOR_UNITS: usize = 4_000_000;

/// Likelihood label for evidence records.
pub const REPEATED_MEASURES_POSTERIOR_LIKELIHOOD: &str =
    "gaussian_units_declared_within_unit_dependence_known_variance";

/// What the assignment of a unit may depend on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SelectionMechanism {
    /// Probabilities fixed in advance.
    Fixed,
    /// Probabilities that depend on the observed data of earlier units and on independent
    /// randomization only.
    DependsOnObservedHistory,
    /// Probabilities that depend on something unobserved. Refused: the likelihood does not
    /// factor and the posterior is not the posterior of the arm means alone.
    DependsOnUnobserved,
}

/// Normal prior of one arm mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalMeanPrior {
    /// Prior mean.
    pub mean: f64,
    /// Prior variance; positive.
    pub variance: f64,
}

/// One unit's arm, measurements and the probability the policy gave its realized arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitRecord<'a> {
    /// The unit's arm.
    pub arm: usize,
    /// The unit's measurements, in order; as many as the declared structure.
    pub measurements: &'a [f64],
    /// The probability the policy assigned to the realized arm, retained as evidence.
    pub assignment_probability: f64,
}

/// A repeated-measures posterior request.
#[derive(Clone, Copy, Debug)]
pub struct RepeatedMeasuresPosteriorRequest<'a> {
    /// The declared within-unit structure (variance and dependence).
    pub structure: &'a OutcomeStructure,
    /// What the assignment depended on.
    pub selection: SelectionMechanism,
    /// One prior per arm.
    pub priors: &'a [NormalMeanPrior],
    /// The completed units.
    pub units: &'a [UnitRecord<'a>],
    /// Arm entering with a plus sign.
    pub plus: usize,
    /// Arm entering with a minus sign.
    pub minus: usize,
    /// Credible level in `(0, 1)`.
    pub credible_level: f64,
}

/// Posterior of one arm mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmMeanPosterior {
    /// Units observed under the arm.
    pub units: usize,
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub standard_deviation: f64,
}

/// Posterior of the contrast of two arm means.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContrastPosterior {
    /// Posterior mean of `mu_plus - mu_minus`.
    pub mean: f64,
    /// Posterior standard deviation.
    pub standard_deviation: f64,
    /// Equal-tailed credible interval.
    pub interval: [f64; 2],
    /// Posterior probability that the contrast is positive.
    pub probability_positive: f64,
}

/// The assignment evidence of the data, for later synthesis.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionEvidence {
    /// The declared selection mechanism.
    pub selection: SelectionMechanism,
    /// Whether assignment adapted to observed data.
    pub adaptive: bool,
    /// The probability of each unit's realized arm, in unit order, as supplied.
    pub assignment_probabilities: Vec<f64>,
    /// The premise under which the posterior needs no correction for the adaptation.
    pub premise: &'static str,
}

/// The posterior of an adaptive repeated-measures trial.
#[derive(Clone, Debug, PartialEq)]
pub struct RepeatedMeasuresPosterior {
    /// Posterior of each arm mean.
    pub arms: Vec<ArmMeanPosterior>,
    /// Posterior of the declared contrast.
    pub contrast: ContrastPosterior,
    /// Variance `v` of one generalized-least-squares unit summary.
    pub unit_summary_variance: f64,
    /// Credible level of the interval.
    pub credible_level: f64,
    /// Always [`REPEATED_MEASURES_POSTERIOR_LIKELIHOOD`].
    pub likelihood: &'static str,
    /// Assignment evidence.
    pub evidence: SelectionEvidence,
}

/// Premise recorded in [`SelectionEvidence::premise`].
pub const SELECTION_IGNORABILITY_PREMISE: &str = "assignment depends only on observed earlier \
    data and independent randomization, so the assignment probabilities factor out of the \
    likelihood of the arm means";

fn premise_failure(detail: &'static str, message: &'static str) -> Error {
    Error::PremiseFailed { detail, message }
}

/// Units and summed generalized-least-squares summaries per arm, validating each unit.
fn accumulate_units(
    units: &[UnitRecord<'_>],
    weights: &[f64],
    arms: usize,
) -> Result<(Vec<usize>, Vec<f64>), Error> {
    let mut counts = vec![0_usize; arms];
    let mut sums = vec![0.0_f64; arms];
    for unit in units {
        let probability = unit.assignment_probability;
        if unit.arm >= arms
            || unit.measurements.len() != weights.len()
            || !unit.measurements.iter().all(|v| v.is_finite())
            || !(probability.is_finite() && probability > 0.0 && probability <= 1.0)
        {
            return Err(Error::Shape(
                "each unit needs a declared arm, the declared number of finite measurements and an assignment probability in (0, 1]",
            ));
        }
        counts[unit.arm] += 1;
        sums[unit.arm] += unit.measurements.iter().zip(weights).map(|(y, w)| y * w).sum::<f64>();
    }
    Ok((counts, sums))
}

/// Posterior of the arm means and of the declared contrast.
///
/// # Errors
///
/// [`Error::PremiseFailed`] with `adaptive_inference.selection_not_ignorable` for an assignment
/// that depends on an unobserved quantity (the next attempt: model the selection jointly with the
/// outcome, or randomize), `adaptive_inference.arm_not_observed` when the plus or minus arm has
/// no unit, and `adaptive_inference.structure_incomplete` for an incomplete or inadmissible
/// declared structure; [`Error::Shape`] and [`Error::InvalidParameter`] for malformed units,
/// priors, arms or level.
pub fn repeated_measures_posterior(
    request: &RepeatedMeasuresPosteriorRequest<'_>,
) -> Result<RepeatedMeasuresPosterior, Error> {
    if request.selection == SelectionMechanism::DependsOnUnobserved {
        return Err(premise_failure(
            "adaptive_inference.selection_not_ignorable",
            "the assignment depends on an unobserved quantity, so the likelihood of the arm means \
             does not factor from the assignment mechanism; model the selection jointly with the \
             outcome, or randomize independently of unobserved quantities",
        ));
    }
    let arms = request.priors.len();
    if !(2..=MAX_PROCEDURE_ARMS).contains(&arms) || request.plus >= arms || request.minus >= arms {
        return Err(Error::Shape(
            "a posterior needs between 2 and the maximum arms and valid contrast arms",
        ));
    }
    if request.plus == request.minus {
        return Err(Error::InvalidParameter("the contrast needs two different arms"));
    }
    if request.units.is_empty() || request.units.len() > MAX_POSTERIOR_UNITS {
        return Err(Error::Limit("a posterior needs between one and four million units"));
    }
    let level = request.credible_level;
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(Error::InvalidParameter(
            "the credible level must lie strictly between zero and one",
        ));
    }
    if !request
        .priors
        .iter()
        .all(|p| p.mean.is_finite() && p.variance.is_finite() && p.variance > 0.0)
    {
        return Err(Error::InvalidParameter(
            "a prior mean must be finite and a prior variance positive",
        ));
    }
    let (weights, summary_variance) = gls_unit_summary(request.structure).ok_or_else(|| {
        premise_failure(
            "adaptive_inference.structure_incomplete",
            "the declared within-unit structure is incomplete or not a valid covariance",
        )
    })?;
    let (counts, sums) = accumulate_units(request.units, &weights, arms)?;
    if counts[request.plus] == 0 || counts[request.minus] == 0 {
        return Err(premise_failure(
            "adaptive_inference.arm_not_observed",
            "a contrast arm has no completed unit; a prior alone does not identify the contrast",
        ));
    }
    let posteriors: Vec<ArmMeanPosterior> = request
        .priors
        .iter()
        .zip(counts.iter().zip(&sums))
        .map(|(prior, (count, sum))| {
            let precision = 1.0 / prior.variance + *count as f64 / summary_variance;
            ArmMeanPosterior {
                units: *count,
                mean: (prior.mean / prior.variance + sum / summary_variance) / precision,
                standard_deviation: (1.0 / precision).sqrt(),
            }
        })
        .collect();
    let (plus, minus) = (posteriors[request.plus], posteriors[request.minus]);
    let mean = plus.mean - minus.mean;
    let standard_deviation = plus.standard_deviation.hypot(minus.standard_deviation);
    let radius = standard_deviation * normal_ppf(0.5 + 0.5 * level);
    Ok(RepeatedMeasuresPosterior {
        arms: posteriors,
        contrast: ContrastPosterior {
            mean,
            standard_deviation,
            interval: [mean - radius, mean + radius],
            probability_positive: norm_cdf(mean / standard_deviation),
        },
        unit_summary_variance: summary_variance,
        credible_level: level,
        likelihood: REPEATED_MEASURES_POSTERIOR_LIKELIHOOD,
        evidence: SelectionEvidence {
            selection: request.selection,
            adaptive: request.selection == SelectionMechanism::DependsOnObservedHistory,
            assignment_probabilities: request
                .units
                .iter()
                .map(|u| u.assignment_probability)
                .collect(),
            premise: SELECTION_IGNORABILITY_PREMISE,
        },
    })
}
