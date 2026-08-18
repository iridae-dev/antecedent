//! Shared estimation prepare helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AverageEffectQuery, Intervention, PopulationRegistry, TargetPopulation};
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
    let kind = estimand.method_kind().map_err(EstimationError::data_msg)?;
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

/// Intersect a complete-case mask with a [`TargetPopulation::Predicate`] selection.
///
/// Other target populations are left unchanged (ATT/ATC are applied at g-computation
/// time; custom distributions carry weights rather than dropping rows).
///
/// # Errors
///
/// Named predicates without a registry, or out-of-range row indices.
pub(crate) fn intersect_predicate_mask(
    row_mask: &mut [bool],
    target: &TargetPopulation,
    n_full: usize,
    registry: Option<&PopulationRegistry>,
) -> Result<(), EstimationError> {
    if !matches!(target, TargetPopulation::Predicate(_)) {
        return Ok(());
    }
    let sel = target
        .resolve(n_full, None, registry)
        .map_err(|e| EstimationError::data_msg(e.to_string()))?;
    for (i, slot) in row_mask.iter_mut().enumerate() {
        *slot = *slot && sel.keep.get(i).copied().unwrap_or(false);
    }
    Ok(())
}
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
