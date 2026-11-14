//! Wire section of a noncausal association result.
//!
//! The section is optional on the result body and absent on every other result, so their bodies
//! and digests are unchanged. An `Association` query requires it. The result describes the
//! observed law of the columns: the body carries no scalar effect estimate, standard error or
//! interval beside it, the section states that it is not an intervention effect and ranks no
//! action, and its calibration label is `not_measured`. A loader checks that every reported
//! number is finite, that the selected analysis is one the declared selection rule can name,
//! that the uncertainty label names only what the section holds, and that a claimed support is
//! the one its graphless support row grants.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use super::AnalysisResultWire;
use crate::{CausalQueryWire, IoError};

/// Uncertainty label of a result that reports no standard error, spread or posterior.
pub const ASSOCIATION_NO_UNCERTAINTY: &str = "none";
/// Uncertainty label of a result that reports analytic standard errors.
pub const ASSOCIATION_ANALYTIC_UNCERTAINTY: &str = "analytic_standard_error";
/// Uncertainty label of a result that reports resampling standard errors and a resampling
/// spread.
pub const ASSOCIATION_RESAMPLING_UNCERTAINTY: &str = "resampling_standard_error";
/// Uncertainty label of a result that reports a posterior.
pub const ASSOCIATION_POSTERIOR_UNCERTAINTY: &str = "posterior";

/// The analyses the declared selection rule can name.
pub const ASSOCIATION_ANALYSES: [&str; 10] = [
    "stratified_mean",
    "stratified_probability",
    "linear_regression",
    "logistic_regression",
    "poisson_regression",
    "pearson_correlation",
    "distance_correlation",
    "predictive_linear",
    "predictive_logistic",
    "predictive_poisson",
];

/// One reported quantity: a stratum value, a coefficient, a correlation or a score.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AssociationTermWire {
    /// Name of the quantity: `intercept`, a predictor name, `stratum:x=1`, or the measure.
    pub label: String,
    /// Scale the estimate is on: `outcome_mean`, `probability`, `log_odds`, `log_rate`,
    /// `correlation`, `distance_correlation`, `outcome_units` for a linear slope.
    pub scale: String,
    /// The estimate.
    pub estimate: f64,
    /// Standard error of the construction named by `standard_error_method`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_error: Option<f64>,
    /// Name of the standard-error construction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_error_method: Option<String>,
    /// Lower end of the resampling spread: the studentized interval of a regression coefficient
    /// under a row bootstrap, the percentile range of the replicates otherwise (widened by the
    /// fixed-b and kernel-bias factors of the block length for a series).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spread_lower: Option<f64>,
    /// Upper end of the resampling spread.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spread_upper: Option<f64>,
    /// Posterior mean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior_mean: Option<f64>,
    /// Posterior standard deviation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior_sd: Option<f64>,
    /// Lower end of the equal-tailed posterior interval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior_lower: Option<f64>,
    /// Upper end of the equal-tailed posterior interval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior_upper: Option<f64>,
    /// Rows behind this quantity.
    pub rows: u64,
}

/// Resampling behind the standard errors and spreads of a frequentist result.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AssociationResamplingWire {
    /// `bootstrap` or `block_bootstrap`.
    pub method: String,
    /// Unit resampled: `row` or `calendar_block`.
    pub unit: String,
    /// Replicates requested.
    pub replicates_requested: u64,
    /// Replicates that fit.
    pub replicates_ok: u64,
    /// Block length of a circular-block resample.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_length: Option<u64>,
    /// Level of the resampling spread.
    pub spread_level: f64,
}

/// Posterior behind the quantities of a Bayesian result.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AssociationPosteriorWire {
    /// `conjugate_normal_inverse_gamma` or `bayesian_bootstrap`.
    pub method: String,
    /// Draws behind the summaries (zero for the closed-form conjugate marginals).
    pub draws: u64,
    /// Level of the equal-tailed interval.
    pub level: f64,
}

/// Out-of-fold scores of a predictive model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AssociationPredictionWire {
    /// Held-out folds.
    pub folds: u64,
    /// `random_assignment` or `consecutive_blocks`.
    pub fold_design: String,
    /// Held-out cases scored.
    pub cases: u64,
    /// Name of the proper score.
    pub score: String,
    /// Mean score of the fitted model on the held-out cases (lower is better).
    pub mean_score: f64,
    /// Mean score of the training-mean baseline on the same cases.
    pub baseline_score: f64,
    /// Mean continuous ranked probability score (real outcome) or `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_crps: Option<f64>,
    /// Always `not_measured`: a score is a point value with no coverage record.
    pub calibration: String,
}

/// Noncausal association of the observed data (calibration `not_measured`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AssociationWire {
    /// The analysis the selection rule picked.
    pub analysis: String,
    /// Identifier of the declared selection rule.
    pub rule: String,
    /// The functional the query requested (`auto` lets the rule pick).
    pub requested: String,
    /// Outcome scale the rule read: `continuous`, `binary` or `count`.
    pub outcome_kind: String,
    /// `cross_section` or `series`.
    pub modality: String,
    /// The reported quantities.
    pub terms: Vec<AssociationTermWire>,
    /// Complete rows behind the answer.
    pub rows: u64,
    /// Rows left out for a missing cell or weight.
    pub rows_dropped: u64,
    /// Resampling behind the standard errors and spreads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resampling: Option<AssociationResamplingWire>,
    /// Posterior behind the summaries of a Bayesian result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior: Option<AssociationPosteriorWire>,
    /// Out-of-fold scores of a predictive model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction: Option<AssociationPredictionWire>,
    /// The premises the description rests on and the readings it does not support.
    pub assumptions: Vec<String>,
    /// Reason code of the refused causal query this answer follows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refused_after: Option<String>,
    /// Uncertainty label.
    pub uncertainty: String,
    /// Always `not_measured`.
    pub calibration: String,
    /// Whether the fit matched a row of the graphless support table; the claim is the
    /// uncertainty of the slope term.
    pub supported: bool,
    /// Always `false`: an association ranks no action.
    pub ranks_actions: bool,
    /// Always `false`: an association is not the effect of an intervention.
    pub is_intervention_effect: bool,
}

/// Check an association section against its query.
pub(in crate::analysis_result_artifact) fn validate_association(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let fail = |message: &str| Err(IoError::Convert(format!("association section: {message}")));
    let (query, section) = match (&result.query, &result.association) {
        (CausalQueryWire::Association(query), Some(section)) => (query, section),
        (CausalQueryWire::Association(_), None) => {
            return fail("an association artifact is missing its section");
        }
        (_, None) => return Ok(()),
        (_, Some(_)) => return fail("the section is attached to another query"),
    };
    // A stored support is recomputed from the query and the section; a section that states
    // none is accepted as written.
    if (section.supported && !association_slope_row_admits(query, section))
        || section.calibration != "not_measured"
        || section.ranks_actions
        || section.is_intervention_effect
        || result.estimate.is_some()
        || result.standard_error.is_some()
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
    {
        return fail(
            "an association describes the observed law: it claims no support its graphless row \
             does not grant, no calibration, no intervention effect, no ranking of actions, and \
             no scalar estimate or interval",
        );
    }
    if !ASSOCIATION_ANALYSES.contains(&section.analysis.as_str()) {
        return fail("the analysis is not one the declared selection rule names");
    }
    if !matches!(section.outcome_kind.as_str(), "continuous" | "binary" | "count")
        || !matches!(section.modality.as_str(), "cross_section" | "series")
    {
        return fail("the outcome kind or the modality is not one the rule reads");
    }
    if section.terms.is_empty() || section.rows == 0 || section.assumptions.is_empty() {
        return fail("an association reports at least one quantity, its rows and its assumptions");
    }
    validate_terms(section)?;
    if let Some(resampling) = &section.resampling
        && (resampling.replicates_ok > resampling.replicates_requested
            || !(0.0..1.0).contains(&resampling.spread_level)
            || !matches!(resampling.method.as_str(), "bootstrap" | "block_bootstrap")
            || !matches!(resampling.unit.as_str(), "row" | "calendar_block"))
    {
        return fail("the resampling record is inconsistent");
    }
    if let Some(posterior) = &section.posterior
        && (!(0.0..1.0).contains(&posterior.level)
            || !matches!(
                posterior.method.as_str(),
                "conjugate_normal_inverse_gamma" | "bayesian_bootstrap"
            ))
    {
        return fail("the posterior record is inconsistent");
    }
    if let Some(prediction) = &section.prediction
        && (prediction.folds < 2
            || prediction.cases == 0
            || prediction.calibration != "not_measured"
            || !prediction.mean_score.is_finite()
            || !prediction.baseline_score.is_finite()
            || !prediction.mean_crps.is_none_or(f64::is_finite))
    {
        return fail("the held-out scores are not finite or claim a calibration");
    }
    let expected = if section.posterior.is_some() {
        ASSOCIATION_POSTERIOR_UNCERTAINTY
    } else if section.resampling.is_some() {
        ASSOCIATION_RESAMPLING_UNCERTAINTY
    } else if section.terms.iter().any(|term| term.standard_error.is_some()) {
        ASSOCIATION_ANALYTIC_UNCERTAINTY
    } else {
        ASSOCIATION_NO_UNCERTAINTY
    };
    if section.uncertainty != expected {
        return fail("the uncertainty label names an uncertainty the section does not hold");
    }
    Ok(())
}

/// Whether the section's slope interval, rows and resamples match a supported row of the
/// graphless table. Recomputed from the stored query and section, never taken from the
/// producer. Rows and resamples are thresholds of the row; a linear regression of a real
/// outcome on one numeric predictor, nothing held fixed, unweighted, no dropped row, the 95%
/// level, every requested replicate fitted and a conjugate prior that moves the slope by at
/// most a hundredth of its posterior standard deviation are checked here.
fn association_slope_row_admits(
    query: &crate::AssociationQueryWire,
    section: &AssociationWire,
) -> bool {
    if section.analysis != "linear_regression"
        || section.outcome_kind != "continuous"
        || query.predictors.len() != 1
        || !query.conditioning.is_empty()
        || query.weight.is_some()
        || section.rows_dropped != 0
        || section.terms.len() != 2
        || section.terms[0].label != "intercept"
        || section.terms[1].label.contains('=')
        || section.terms[1].scale != "outcome_units"
    {
        return false;
    }
    let slope = &section.terms[1];
    // One series is one dependent sequence; table rows are independent units.
    let (design, assignment_unit, blocks) = match section.modality.as_str() {
        "cross_section" => ("cross_section_linear_slope", "unit", 0),
        "series" => ("series_linear_slope", "sequence", 1),
        _ => return false,
    };
    let at_95 = |level: f64| (level - 0.95).abs() <= 1e-12;
    let (method, inference_claim, band, resamples) = if let Some(record) = &section.posterior {
        let (Some(lower), Some(upper)) = (slope.posterior_lower, slope.posterior_upper) else {
            return false;
        };
        let moved = slope
            .posterior_mean
            .zip(slope.posterior_sd)
            .is_none_or(|(mean, sd)| (mean - slope.estimate).abs() > 0.01 * sd);
        if !at_95(record.level) || (record.method == "conjugate_normal_inverse_gamma" && moved) {
            return false;
        }
        (
            record.method.as_str(),
            "slope_95_posterior_equal_tailed_interval",
            (lower, upper),
            record.draws,
        )
    } else if let Some(record) = &section.resampling {
        let (Some(lower), Some(upper)) = (slope.spread_lower, slope.spread_upper) else {
            return false;
        };
        let (method, inference_claim) = match record.unit.as_str() {
            "row" => ("bootstrap_row", "slope_95_studentized_spread"),
            "calendar_block" => ("bootstrap_circular_block", "slope_95_fixed_b_percentile_spread"),
            _ => return false,
        };
        if !at_95(record.spread_level)
            || record.replicates_ok != record.replicates_requested
            || slope.standard_error_method.as_deref() != Some(method)
        {
            return false;
        }
        (method, inference_claim, (lower, upper), record.replicates_ok)
    } else {
        let (Some(error), Some(method)) =
            (slope.standard_error, slope.standard_error_method.as_deref())
        else {
            return false;
        };
        (
            method,
            "slope_standard_error_95_normal_reference_interval",
            (slope.estimate - error, slope.estimate + error),
            0,
        )
    };
    crate::graphless_support::graphless_admits(
        crate::graphless_support::GraphlessSupportKey {
            family: "association",
            design,
            method,
            inference_claim,
        },
        crate::graphless_support::GraphlessAssignmentSupport {
            assignment_unit,
            rows: usize::try_from(section.rows).unwrap_or(usize::MAX),
            blocks,
            interval_95_published: band.0.is_finite() && band.1.is_finite() && band.0 < band.1,
            reported_intervals: 1,
            all_reported_intervals: true,
            interval_resamples: usize::try_from(resamples).unwrap_or(usize::MAX),
            ..Default::default()
        },
    )
}

/// Check numerical term summaries and their links to section-level uncertainty.
fn validate_terms(section: &AssociationWire) -> Result<(), IoError> {
    let fail = |message: &str| Err(IoError::Convert(format!("association section: {message}")));
    let finite = |value: Option<f64>| value.is_none_or(f64::is_finite);
    for term in &section.terms {
        if !term.estimate.is_finite()
            || !finite(term.standard_error)
            || term.standard_error.is_some_and(|se| se < 0.0)
            || !finite(term.spread_lower)
            || !finite(term.spread_upper)
            || term.spread_lower.zip(term.spread_upper).is_some_and(|(lower, upper)| lower > upper)
            || !finite(term.posterior_mean)
            || !finite(term.posterior_sd)
            || term.posterior_sd.is_some_and(|sd| sd < 0.0)
            || !finite(term.posterior_lower)
            || !finite(term.posterior_upper)
            || term
                .posterior_lower
                .zip(term.posterior_upper)
                .is_some_and(|(lower, upper)| lower > upper)
        {
            return fail("a reported quantity is not finite and ordered");
        }
        if term.standard_error.is_some() != term.standard_error_method.is_some() {
            return fail("a standard error comes with its construction and the reverse");
        }
        let has_posterior = term.posterior_mean.is_some();
        if has_posterior != section.posterior.is_some() {
            return fail("a posterior summary comes with the posterior record and the reverse");
        }
        let has_spread = term.spread_lower.is_some() || term.spread_upper.is_some();
        if has_spread && section.resampling.is_none() {
            return fail("a percentile spread comes from resampling the section records");
        }
    }
    Ok(())
}
