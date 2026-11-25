//! Admission and source projection for the original static average effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_details::native_claims::{
    NATIVE_STATE_UNAVAILABLE, PROJECTION_MISMATCH, UNSUPPORTED_ESTIMAND,
};
use antecedent_core::{
    AverageEffectQuery, IdentificationStatus, Intervention, OutcomeFunctional, TargetPopulation,
    VariableId,
};
use antecedent_design::decision_eval::{MeanEstimate, MeanSource};

use super::{
    DecisionProjectionError, OUTCOME_NAME, TREATMENT_NAME, causal_source_identity,
    mean_contrast_quantity, static_refusal,
};
use crate::result::{ExecutedContract, StudyResult};
use crate::support::CellStatus;

pub(super) const UNITS_POPULATION: &str = "effect units must be declared (at most 256 bytes) and population must be target (the original observed sample)";

/// The original result fields needed to admit a static effect after execution.
#[derive(Clone, Debug)]
pub struct NativeEffectAuthority {
    query: AverageEffectQuery,
    status: IdentificationStatus,
    ate: f64,
    has_structural_response: bool,
    support_status: Option<CellStatus>,
    executed_contract: Option<ExecutedContract>,
}

impl NativeEffectAuthority {
    /// Read the effect and query from the result before it crosses a binding.
    #[must_use]
    pub fn from_result(result: &StudyResult) -> Option<Self> {
        Some(Self {
            query: result.identification.average_effect()?.clone(),
            status: result.identification.status,
            ate: result.estimate().as_effect()?.ate,
            has_structural_response: result.structural_response.is_some(),
            support_status: result.support_status(),
            executed_contract: result.executed_contract().copied(),
        })
    }
}

/// A mean source and the original query fields read by scalar-result consumers.
#[derive(Debug)]
pub struct StaticEffectSource {
    /// Mean of the original contrast and its execution provenance.
    pub source: MeanSource,
    /// Original treatment label.
    pub treatment: String,
    /// Original outcome label.
    pub outcome: String,
    /// Original control intervention value.
    pub control: f64,
    /// Original active intervention value.
    pub active: f64,
}

/// Admit the original checked static effect and project it to a single contrast coordinate.
///
/// # Errors
///
/// Returns a declaration error for invalid units, population or missing schema labels, and a
/// structured bind refusal for unsupported evidence, a changed point or missing execution state.
#[allow(
    clippy::result_large_err,
    reason = "ExternalRefusal is the structured refusal of the external contract, returned by value on a cold once-per-decision refusal path"
)]
pub fn project_static_effect(
    authority: Option<&NativeEffectAuthority>,
    units: &str,
    population: &str,
    supplied: f64,
    name: impl Fn(VariableId) -> Option<String>,
) -> Result<StaticEffectSource, DecisionProjectionError> {
    if units.trim().is_empty() || units.len() > 256 || population != "target" {
        return Err(DecisionProjectionError::Declaration(UNITS_POPULATION));
    }
    let Some(authority) = authority else {
        return Err(static_refusal(UNSUPPORTED_ESTIMAND).into());
    };
    let query = &authority.query;
    if authority.status != IdentificationStatus::NonparametricallyIdentified
        || query.target_population != TargetPopulation::AllObserved
        || query.outcome_functional != OutcomeFunctional::Mean
        || !query.effect_modifiers.is_empty()
        || authority.has_structural_response
        || authority.support_status.is_some_and(|status| status != CellStatus::Supported)
    {
        return Err(static_refusal(UNSUPPORTED_ESTIMAND).into());
    }
    if !authority.ate.is_finite() || supplied.to_bits() != authority.ate.to_bits() {
        return Err(static_refusal(PROJECTION_MISMATCH).into());
    }
    let Some(executed) = &authority.executed_contract else {
        return Err(static_refusal(NATIVE_STATE_UNAVAILABLE).into());
    };
    let level = |intervention: &Intervention| match intervention {
        Intervention::Set { variable, value } if *variable == query.treatment => value.as_f64(),
        _ => None,
    };
    let (Some(control), Some(active)) = (level(&query.control), level(&query.active)) else {
        return Err(static_refusal(UNSUPPORTED_ESTIMAND).into());
    };
    let treatment =
        name(query.treatment).ok_or(DecisionProjectionError::Declaration(TREATMENT_NAME))?;
    let outcome = name(query.outcome).ok_or(DecisionProjectionError::Declaration(OUTCOME_NAME))?;
    let coordinate =
        mean_contrast_quantity(&treatment, &outcome, units, population, control, active);
    Ok(StaticEffectSource {
        source: MeanSource {
            coordinates: vec![coordinate],
            means: vec![authority.ate],
            estimate: MeanEstimate::Frequentist,
            provider_id: "native:static_ate".to_owned(),
            snapshot_id: executed.identities.data_snapshot.to_hex(),
            causal_contract_id: causal_source_identity(executed),
            rng_id: "none:mean_grid".to_owned(),
        },
        treatment,
        outcome,
        control,
        active,
    })
}
