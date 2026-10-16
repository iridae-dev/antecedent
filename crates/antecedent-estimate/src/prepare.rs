//! Shared estimation prepare helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AverageEffectQuery, Intervention, PopulationRegistry, TargetPopulation, VariableId,
};
use antecedent_expr::{EstimandMethod, IdentifiedEstimand};
use antecedent_identify::{EstimandClaim, IdentificationResult};

use crate::adjustment::intervention_f64;
use crate::error::EstimationError;

/// Require the estimand method to be one of `allowed`.
///
/// # Errors
///
/// Unknown method string or incompatible estimand.
pub(crate) fn require_method(
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

/// The selected estimand and its identification claim.
///
/// The two lookup failures use the strings every checked prepare already returns.
pub(crate) struct CheckedSelection<'a> {
    /// Estimand at `estimand_index`.
    pub estimand: &'a IdentifiedEstimand,
    /// Claim returned by [`IdentificationResult::claim`] for that index.
    pub claim: EstimandClaim,
}

impl<'a> CheckedSelection<'a> {
    /// Look up `estimand_index` on `identification`.
    ///
    /// # Errors
    ///
    /// `selected estimand index is out of range` or `selected estimand has no identification claim`.
    pub(crate) fn resolve(
        identification: &'a IdentificationResult,
        estimand_index: usize,
    ) -> Result<Self, EstimationError> {
        let estimand = identification
            .estimands
            .get(estimand_index)
            .ok_or_else(|| EstimationError::data_msg("selected estimand index is out of range"))?;
        let claim = identification.claim(estimand_index).ok_or_else(|| {
            EstimationError::data_msg("selected estimand has no identification claim")
        })?;
        Ok(Self { estimand, claim })
    }
}

/// Refuse duplicate roles or a role that is also treatment or outcome.
///
/// `message` is the caller's existing refusal text. Membership is tested after the same
/// sort-and-dedup the checked prepares used.
///
/// # Errors
///
/// `message`, when a role repeats or equals an excluded id.
pub(crate) fn require_unique_roles(
    roles: &[VariableId],
    excluded: &[VariableId],
    message: &'static str,
) -> Result<(), EstimationError> {
    let mut sorted = roles.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != roles.len() || excluded.iter().any(|id| sorted.contains(id)) {
        return Err(EstimationError::data_msg(message));
    }
    Ok(())
}

/// Require an adjustment-shaped estimand (`backdoor.*` / `generalized.adjustment*` / `tiered.*`).
///
/// # Errors
///
/// Incompatible estimand.
pub(crate) fn require_adjustment_shaped(
    estimand: &IdentifiedEstimand,
    message: &'static str,
) -> Result<(), EstimationError> {
    if !estimand.is_adjustment_shaped() {
        return Err(EstimationError::IncompatibleEstimand { message });
    }
    Ok(())
}

/// Validate an ATE query allowing `AllObserved` / Treated / Untreated targets.
pub(crate) fn validate_ate_query_with_targets(
    query: &AverageEffectQuery,
) -> Result<(), EstimationError> {
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
