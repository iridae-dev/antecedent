//! Shared estimation prepare helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AverageEffectQuery, Intervention, PopulationRegistry, TargetPopulation, VariableId,
};
use antecedent_data::materialize::{
    EstimatorStreamNeed, PreparationControls, PreparationProduct, PreparationReceipt, prepare_table,
};
use antecedent_data::{MissingPolicy, TableView, TabularData};
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

/// What every estimator in this crate reads: one finite table, never a scan.
///
/// An estimator declares this to [`admit_estimator_columns`]. A frame it cannot
/// stream is refused or explicitly materialized before it gets here, and the
/// receipt says which.
pub(crate) const FINITE_TABLE_ONLY: EstimatorStreamNeed = EstimatorStreamNeed::CannotStream;

/// The finite table an estimator reads, with the products of preparing it.
pub(crate) struct AdmittedColumns<'a> {
    /// The table the estimator reads.
    pub data: &'a TabularData,
    /// Complete-case mask over `data`'s rows, computed once by preparation.
    pub row_mask: Vec<bool>,
    /// Preparation receipt. No byte budget is compared here, so the route is not
    /// described as bounded.
    pub receipt: PreparationReceipt,
}

/// The summary an estimate records for the receipt its columns were admitted under.
pub(crate) fn preparation_summary(
    receipt: &PreparationReceipt,
) -> std::sync::Arc<crate::PreparationSummary> {
    std::sync::Arc::new(crate::PreparationSummary::from_receipt(receipt))
}

/// Test check that an estimate recorded the preparation of a table of `rows` rows.
#[cfg(test)]
pub(crate) fn assert_records_preparation(estimate: &crate::EffectEstimate, rows: u64) {
    let summary = estimate
        .provenance
        .preparation
        .as_ref()
        .expect("the estimate must record its preparation receipt");
    assert_eq!(summary.backend, "direct_arrow");
    assert_eq!(summary.rows_admitted, Some(rows));
    assert_eq!(summary.route_bound, "rows_only");
    assert!(!summary.receipt_digest.is_empty());
}

/// Run the accepted preparation path, then hand the estimator the same finite table.
///
/// `stream` is the estimator's own declaration. An estimator that declares it can
/// stream is refused here because it is being handed a table. The receipt must
/// prove the table's row count.
///
/// # Errors
///
/// Schema, missing-value, or admission failure, an estimator that declares it
/// reads a scan, or a receipt that does not prove this table.
pub(crate) fn admit_estimator_columns<'a>(
    data: &'a TabularData,
    ids: &[VariableId],
    stream: EstimatorStreamNeed,
) -> Result<AdmittedColumns<'a>, EstimationError> {
    let mut controls = PreparationControls::finite_resident();
    controls.stream = stream;
    let PreparationProduct { row_mask, receipt, .. } =
        prepare_table(data, ids, MissingPolicy::CompleteCase, &[], controls)?;
    let rows = u64::try_from(data.row_count())
        .map_err(|_| EstimationError::data_msg("admitted row count does not fit u64"))?;
    if receipt.proved_rows() != Some(rows) {
        return Err(EstimationError::unsupported(
            "the preparation receipt does not prove the finite table the estimator reads",
        ));
    }
    Ok(AdmittedColumns { data, row_mask, receipt })
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
