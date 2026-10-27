//! Wire section of a counterfactual result that contrasts several outcomes.
//!
//! The section is optional on the result body and absent on every other result, so their bodies
//! and digests are unchanged. A counterfactual with one outcome keeps publishing its per-unit
//! effects as [`super::UnitEffectsWire`]; one with several outcomes, or with simultaneous
//! interventions that the single form does not state, publishes one contrast per outcome here,
//! in the order the query lists its outcomes. A loader checks that the section belongs to a
//! counterfactual query, that its outcomes are the query's, that every contrast covers the
//! same units, and that each contrast's per-unit effects are well formed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use super::{AnalysisResultWire, UnitEffectsWire};
use crate::{CausalQueryWire, IoError};

/// The unit-level contrast of one outcome between the active and control worlds.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OutcomeContrastWire {
    /// Raw id of the outcome the contrast reads.
    pub outcome: u32,
    /// Mean effect over the observed units (the posterior mean when the execution is Bayesian).
    pub mean: f64,
    /// Standard error of [`Self::mean`] the execution reported: the posterior standard deviation
    /// or the bootstrap standard error. Absent for a point result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_error: Option<f64>,
    /// Per-unit effects, their intervals and extrapolation flags.
    pub unit_effects: UnitEffectsWire,
}

/// Contrasts of several outcomes that share one abduction of the exogenous state.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualContrastsWire {
    /// How the exogenous noise was obtained (the engine's noise-inference kind label).
    pub noise_inference: String,
    /// One contrast per query outcome, in the query's order.
    pub contrasts: Vec<OutcomeContrastWire>,
}

/// Check the section against the result body that carries it.
///
/// # Errors
///
/// A section on a result whose query is not a counterfactual, outcomes other than the query's,
/// contrasts over different unit counts, a non-finite mean or standard error, a single-form
/// `unit_effects` beside it, or malformed per-unit effects.
pub(super) fn validate(
    result: &AnalysisResultWire,
    section: &CounterfactualContrastsWire,
    variable_count: usize,
) -> Result<(), IoError> {
    let CausalQueryWire::Counterfactual { outcomes, .. } = &result.query else {
        return Err(IoError::Convert(
            "counterfactual contrasts require a counterfactual query".into(),
        ));
    };
    if result.unit_effects.is_some() {
        return Err(IoError::Convert(
            "counterfactual contrasts replace the single-outcome unit effects".into(),
        ));
    }
    if section.noise_inference.trim().is_empty() {
        return Err(IoError::Convert(
            "counterfactual contrasts need a noise-inference kind".into(),
        ));
    }
    let published: Vec<u32> = section.contrasts.iter().map(|c| c.outcome).collect();
    if &published != outcomes {
        return Err(IoError::Convert(
            "counterfactual contrasts must name the query's outcomes in order".into(),
        ));
    }
    let n_units = section.contrasts.first().map(|c| c.unit_effects.effects.len());
    for contrast in &section.contrasts {
        if usize::try_from(contrast.outcome).map_or(true, |id| id >= variable_count) {
            return Err(IoError::Convert("counterfactual contrast outcome is out of range".into()));
        }
        if !contrast.mean.is_finite()
            || contrast.standard_error.is_some_and(|se| !se.is_finite() || se < 0.0)
        {
            return Err(IoError::Convert(
                "counterfactual contrast mean and standard error must be finite and nonnegative"
                    .into(),
            ));
        }
        if Some(contrast.unit_effects.effects.len()) != n_units {
            return Err(IoError::Convert(
                "counterfactual contrasts must cover the same units".into(),
            ));
        }
        super::validate::validate_unit_effects(&contrast.unit_effects)?;
    }
    Ok(())
}
