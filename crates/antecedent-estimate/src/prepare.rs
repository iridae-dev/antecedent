//! Shared estimation prepare helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AverageEffectQuery, Intervention, TargetPopulation};
use antecedent_expr::{EstimandMethod, IdentifiedEstimand};

use crate::adjustment::intervention_f64;
use crate::error::EstimationError;

/// Require the estimand method to be one of `allowed`.
///
/// # Errors
///
/// Unknown method string or incompatible estimand.
pub fn require_method(
    estimand: &IdentifiedEstimand,
    allowed: &[EstimandMethod],
    message: &'static str,
) -> Result<EstimandMethod, EstimationError> {
    let kind = estimand.method_kind().map_err(EstimationError::UnsupportedQuery)?;
    if !allowed.contains(&kind) {
        return Err(EstimationError::IncompatibleEstimand { message });
    }
    Ok(kind)
}

/// Validate an ATE query allowing `AllObserved` / Treated / Untreated targets.
pub fn validate_ate_query_with_targets(query: &AverageEffectQuery) -> Result<(), EstimationError> {
    query.validate()?;
    if !query.effect_modifiers.is_empty() {
        return Err(EstimationError::EffectModifiers);
    }
    if !matches!(
        query.target_population,
        TargetPopulation::AllObserved
            | TargetPopulation::Treated
            | TargetPopulation::Untreated
            | TargetPopulation::Predicate(_)
    ) {
        return Err(EstimationError::TargetPopulation);
    }
    Ok(())
}

/// Validate a Phase-1 ATE query (no effect modifiers, all-observed population).
///
/// # Errors
///
/// Unsupported query options.
pub fn validate_simple_ate_query(query: &AverageEffectQuery) -> Result<(), EstimationError> {
    query.validate()?;
    if !query.effect_modifiers.is_empty() {
        return Err(EstimationError::EffectModifiers);
    }
    if query.target_population != TargetPopulation::AllObserved {
        return Err(EstimationError::TargetPopulation);
    }
    Ok(())
}

/// Extract numeric active/control levels and nonzero treatment delta.
///
/// # Errors
///
/// Non-numeric / non-Set interventions or identical levels.
pub fn treatment_contrast(
    active: &Intervention,
    control: &Intervention,
) -> Result<(f64, f64, f64), EstimationError> {
    let a = intervention_f64(active)?;
    let c = intervention_f64(control)?;
    let delta = a - c;
    if delta == 0.0 {
        return Err(EstimationError::unsupported(
            "active and control treatment levels must differ",
        ));
    }
    Ok((a, c, delta))
}
