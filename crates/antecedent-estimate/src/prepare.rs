//! Shared estimation prepare helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::execution_attempt::{PhysicalEvent, record_physical};
use antecedent_core::{
    AverageEffectQuery, Intervention, PopulationRegistry, TargetPopulation, VariableId,
};
use antecedent_data::materialize::{
    EstimatorStreamNeed, PreparationBudget, PreparationControls, PreparationProduct,
    PreparationReceipt, prepare_table,
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

thread_local! {
    static AMBIENT_BUDGET: std::cell::Cell<Option<PreparationBudget>> =
        const { std::cell::Cell::new(None) };
}

/// Run `prepare` with every estimator preparation on this thread admitted against `budget`.
///
/// Inside the call, [`admit_estimator_columns`] compares the payload with the budget: a
/// payload over it is refused, and one inside it yields a receipt whose route is a proved
/// bound (`route_bound = "proved"`, `budget_bytes` recorded). Outside it the payload is
/// unmetered and only the rows are proved. The previous budget is restored on return or
/// unwind, so nesting is safe.
///
/// The planner wraps an estimator's prepare in this with
/// [`PreparationBudget::from_limits`] of the study's execution context.
pub fn with_preparation_budget<T>(budget: PreparationBudget, prepare: impl FnOnce() -> T) -> T {
    struct Restore(Option<PreparationBudget>);
    impl Drop for Restore {
        fn drop(&mut self) {
            AMBIENT_BUDGET.with(|slot| slot.set(self.0));
        }
    }
    let _restore = Restore(AMBIENT_BUDGET.with(|slot| slot.replace(Some(budget))));
    prepare()
}

/// The budget [`with_preparation_budget`] set on this thread, if any.
#[must_use]
pub fn ambient_preparation_budget() -> Option<PreparationBudget> {
    AMBIENT_BUDGET.with(std::cell::Cell::get)
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
    /// Preparation receipt. A byte budget is compared only under a workflow context;
    /// without one the route is not described as bounded.
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
    // The resource facts are the receipt's: a finite payload that no budget was compared with.
    assert!(summary.admitted_bytes.is_some_and(|bytes| bytes > 0));
    assert_eq!(summary.budget_bytes, None);
    assert_eq!(summary.delivery.as_deref(), Some("finite"));
}

/// Most distinct treatment values a column may take and still be read as discrete levels
/// rather than as a continuous dose.
pub(crate) const MAX_DISCRETE_LEVELS: usize = 10;

/// Restrict the rows of a regression-based contrast to the two contrasted levels when the
/// treatment is a discrete column with further levels.
///
/// A regression that enters the treatment as one numeric column reads a column such as
/// `0/1/2` as a dose, so "level 1 against level 0" would be the fitted slope, not the contrast
/// of the two levels. A column counts as discrete when it takes at most
/// [`MAX_DISCRETE_LEVELS`] distinct values, holds both contrasted levels, and holds some other
/// value. Such rows are refused with `treatment_level_undeclared`, naming the observed levels,
/// unless `other_levels` declares the restriction to the rows at the two levels, in which case
/// the mask is narrowed and the exclusion is returned for the result to report. Any other
/// column (two-valued, or continuous) is left alone and returns `None`.
///
/// # Errors
///
/// A column read failure, or `treatment_level_undeclared` as above.
#[allow(
    clippy::float_cmp,
    reason = "declared treatment levels are exact category labels, not computed values"
)]
pub(crate) fn restrict_to_contrast_levels(
    data: &TabularData,
    treatment: VariableId,
    row_mask: &mut [bool],
    (active, control): (f64, f64),
    other_levels: crate::propensity::levels::OtherLevels,
) -> Result<Option<crate::propensity::levels::DeclaredTreatmentLevels>, EstimationError> {
    let values = data.float64_selected(treatment, row_mask).map_err(EstimationError::from)?;
    if !is_discrete_with_other_levels(&values, active, control) {
        return Ok(None);
    }
    let (keep, levels) = crate::propensity::levels::classify_treatment_levels(
        &values,
        active,
        control,
        other_levels,
    )?;
    let mut decisions = keep.iter().copied();
    for slot in row_mask.iter_mut().filter(|slot| **slot) {
        *slot = decisions.next().unwrap_or(false);
    }
    Ok(Some(levels))
}

/// Whether a treatment column is discrete (at most [`MAX_DISCRETE_LEVELS`] distinct values),
/// holds both contrasted levels, and holds some other value.
#[allow(
    clippy::float_cmp,
    reason = "declared treatment levels are exact category labels, not computed values"
)]
pub(crate) fn is_discrete_with_other_levels(values: &[f64], active: f64, control: f64) -> bool {
    let mut distinct: Vec<f64> = Vec::new();
    for &value in values {
        if !distinct.iter().any(|seen| *seen == value) {
            distinct.push(value);
            if distinct.len() > MAX_DISCRETE_LEVELS {
                return false;
            }
        }
    }
    let has = |level: f64| distinct.iter().any(|seen| *seen == level);
    distinct.len() >= 3 && has(active) && has(control)
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
    let mut controls = ambient_preparation_budget()
        .map_or_else(PreparationControls::finite_resident, PreparationControls::finite_within);
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
    let sel = target.resolve(n_full, None, registry)?;
    for (i, slot) in row_mask.iter_mut().enumerate() {
        *slot = *slot && sel.keep.get(i);
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

/// A source-ordered cohort, constructed once after all role restrictions.
pub(crate) struct SelectedRows {
    pub(crate) indices: Vec<usize>,
    source_len: usize,
}
impl SelectedRows {
    pub(crate) fn from_mask(mask: &[bool]) -> Self {
        Self {
            indices: mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect(),
            source_len: mask.len(),
        }
    }
    fn copy_into(&self, values: &[f64], output: &mut [f64]) -> Result<(), EstimationError> {
        if values.len() != self.source_len || output.len() != self.indices.len() {
            return Err(EstimationError::data_msg("selected column length mismatch"));
        }
        record_physical(PhysicalEvent::Scan, 1);
        record_physical(PhysicalEvent::Copy, 1);
        record_physical(PhysicalEvent::CopiedBytes, size_of_val(output) as u64);
        if self.indices.len() == self.source_len {
            output.copy_from_slice(values);
        } else {
            for (slot, &row) in output.iter_mut().zip(&self.indices) {
                *slot = values[row];
            }
        }
        Ok(())
    }
    pub(crate) fn gather(
        &self,
        data: &TabularData,
        id: VariableId,
    ) -> Result<Vec<f64>, EstimationError> {
        let antecedent_data::ColumnView::Float64(column) = data.column(id)? else {
            return Err(EstimationError::data_msg("expected float64 selected column"));
        };
        let mut output = vec![0.0; self.indices.len()];
        self.copy_into(column.values.as_slice(), &mut output)?;
        Ok(output)
    }
}

/// Materialize selected columns before assembling the unchanged linear design kernel.
pub(crate) fn materialized_adjustment_design(
    data: &TabularData,
    treatment: VariableId,
    outcome: VariableId,
    adjustment: &[VariableId],
    expanded: &[antecedent_data::categorical::ExpandedContrast],
    mask: &[bool],
) -> Result<antecedent_stats::CompiledDesign, EstimationError> {
    if mask.len() != data.row_count() {
        return Err(EstimationError::data_msg("design selection length mismatch"));
    }
    let selected = SelectedRows::from_mask(mask);
    let treatment = selected.gather(data, treatment)?;
    let outcome = selected.gather(data, outcome)?;
    let mut columns = Vec::new();
    for &id in adjustment {
        if let Some(contrast) = expanded.iter().find(|c| c.variable == id) {
            for column in &contrast.columns {
                let mut output = vec![0.0; selected.indices.len()];
                selected.copy_into(column, &mut output)?;
                columns.push((id, output));
            }
        } else {
            columns.push((id, selected.gather(data, id)?));
        }
    }
    let refs: Vec<_> = columns.iter().map(|(id, column)| (*id, column.as_slice())).collect();
    let design = antecedent_stats::CompiledDesign::linear_adjustment(
        &treatment,
        &refs,
        &outcome,
        &selected.indices,
    )?;
    Ok(design)
}

/// Gather the admitted cohort directly into its final design. Expanded contrasts
/// remain source-row aligned; selection is applied exactly once to every column.
pub(crate) fn direct_adjustment_design(
    data: &TabularData,
    treatment: VariableId,
    outcome: VariableId,
    adjustment: &[VariableId],
    expanded: &[antecedent_data::categorical::ExpandedContrast],
    mask: &[bool],
) -> Result<antecedent_stats::CompiledDesign, EstimationError> {
    use antecedent_data::ColumnView;
    use antecedent_stats::{
        CompiledDesign, DesignColumnMap, DesignColumnRole, StandardizationRecord,
    };
    use std::sync::Arc;
    let column = |id| -> Result<&[f64], EstimationError> {
        match data.column(id)? {
            ColumnView::Float64(c) => Ok(c.values.as_slice()),
            _ => Err(EstimationError::data_msg("expected a float64 design column")),
        }
    };
    if mask.len() != data.row_count() {
        return Err(EstimationError::data_msg("design selection length mismatch"));
    }
    let selected = SelectedRows::from_mask(mask);
    if selected.indices.is_empty() {
        return Err(EstimationError::data_msg("empty design"));
    }
    let mut columns = vec![(DesignColumnRole::Treatment, column(treatment)?)];
    for &id in adjustment {
        if let Some(contrast) = expanded.iter().find(|c| c.variable == id) {
            columns.extend(
                contrast.columns.iter().map(|c| (DesignColumnRole::Covariate(id), c.as_slice())),
            );
        } else {
            columns.push((DesignColumnRole::Covariate(id), column(id)?));
        }
    }
    let nrows = selected.indices.len();
    let ncols = columns.len() + 1;
    let mut matrix = vec![
        0.0;
        nrows
            .checked_mul(ncols)
            .ok_or_else(|| EstimationError::data_msg("design size overflow"))?
    ];
    record_physical(PhysicalEvent::Allocation, 1);
    record_physical(PhysicalEvent::AllocatedBytes, size_of_val(matrix.as_slice()) as u64);
    matrix[..nrows].fill(1.0);
    for (dest, (_, values)) in matrix[nrows..].chunks_exact_mut(nrows).zip(&columns) {
        selected.copy_into(values, dest)?;
    }
    let mut y = vec![0.0; nrows];
    record_physical(PhysicalEvent::Allocation, 1);
    record_physical(PhysicalEvent::AllocatedBytes, size_of_val(y.as_slice()) as u64);
    selected.copy_into(column(outcome)?, &mut y)?;
    let roles =
        std::iter::once(DesignColumnRole::Intercept).chain(columns.iter().map(|(role, _)| *role));
    // Vec-to-Arc ownership transfer copies payloads into their shared allocation.
    record_physical(PhysicalEvent::Copy, 3);
    record_physical(
        PhysicalEvent::CopiedBytes,
        (size_of_val(matrix.as_slice())
            + size_of_val(y.as_slice())
            + size_of_val(selected.indices.as_slice())) as u64,
    );
    record_physical(PhysicalEvent::Allocation, 3);
    record_physical(
        PhysicalEvent::AllocatedBytes,
        (size_of_val(matrix.as_slice())
            + size_of_val(y.as_slice())
            + size_of_val(selected.indices.as_slice())) as u64,
    );
    Ok(CompiledDesign {
        nrows,
        ncols,
        matrix: Arc::from(matrix),
        outcome: Arc::from(y),
        columns: DesignColumnMap::from_roles(roles),
        row_selection: Arc::from(selected.indices),
        contrasts: Vec::new(),
        smooths: Vec::new(),
        standardization: StandardizationRecord::default(),
    })
}

#[cfg(test)]
mod metered_preparation_tests {
    use super::*;

    fn table() -> TabularData {
        TabularData::from_f64_columns([
            ("t", &[0.0, 1.0, 0.0, 1.0][..]),
            ("y", &[2.0, 4.0, 8.0, 16.0][..]),
        ])
        .unwrap()
    }

    #[test]
    fn a_budget_turns_the_row_bound_into_a_proved_bound_and_a_tight_one_refuses() {
        let data = table();
        let ids = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let unmetered = admit_estimator_columns(&data, &ids, FINITE_TABLE_ONLY).unwrap();
        assert!(!unmetered.receipt.is_bounded_end_to_end_route());
        assert_eq!(preparation_summary(&unmetered.receipt).budget_bytes, None);

        let metered = with_preparation_budget(PreparationBudget::new(1 << 20), || {
            assert_eq!(ambient_preparation_budget(), Some(PreparationBudget::new(1 << 20)));
            admit_estimator_columns(&data, &ids, FINITE_TABLE_ONLY).map(|admitted| admitted.receipt)
        })
        .unwrap();
        assert!(metered.is_bounded_end_to_end_route());
        let summary = preparation_summary(&metered);
        assert_eq!(summary.route_bound, "proved");
        assert_eq!(summary.budget_bytes, Some(1 << 20));
        assert!(summary.admitted_bytes.is_some_and(|bytes| bytes > 0 && bytes <= 1 << 20));

        let refused = with_preparation_budget(PreparationBudget::new(1), || {
            admit_estimator_columns(&data, &ids, FINITE_TABLE_ONLY)
                .map(|admitted| admitted.row_mask)
        });
        assert!(refused.is_err(), "a payload over the budget is refused");
        assert_eq!(ambient_preparation_budget(), None, "the budget does not outlive the call");
    }
}

#[cfg(test)]
mod direct_design_tests {
    use super::*;
    #[test]
    fn direct_gather_matches_the_independent_masked_design() {
        let data = TabularData::from_f64_columns([
            ("t", &[0.0, 1.0, 0.0, 1.0][..]),
            ("y", &[2.0, 4.0, 8.0, 16.0][..]),
            ("z", &[3.0, 5.0, 7.0, 11.0][..]),
        ])
        .unwrap();
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let z = VariableId::from_raw(2);
        for mask in [[true, true, true, true], [true, false, true, false]] {
            let actual = direct_adjustment_design(&data, t, y, &[z], &[], &mask).unwrap();
            let rows: Vec<_> =
                mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();
            let ts = data.float64_masked(t, &mask).unwrap();
            let ys = data.float64_masked(y, &mask).unwrap();
            let zs = data.float64_masked(z, &mask).unwrap();
            let reference =
                antecedent_stats::CompiledDesign::linear_adjustment(&ts, &[(z, &zs)], &ys, &rows)
                    .unwrap();
            assert_eq!(actual.matrix, reference.matrix);
            assert_eq!(actual.outcome, reference.outcome);
            assert_eq!(actual.row_selection, reference.row_selection);
            assert_eq!(actual.columns, reference.columns);
            let materialized =
                materialized_adjustment_design(&data, t, y, &[z], &[], &mask).unwrap();
            assert_eq!(actual.matrix, materialized.matrix);
            assert_eq!(actual.outcome, materialized.outcome);
            assert_eq!(actual.row_selection, materialized.row_selection);
            assert_eq!(actual.columns, materialized.columns);
        }
    }
}

#[cfg(test)]
mod design_measurement {
    use super::*;
    #[test]
    #[ignore = "explicit performance measurement; requires a quiet machine"]
    fn measure_direct_design() {
        assert_eq!(std::env::var("CLASSIC_MEASURE").as_deref(), Ok("1"));
        let number = |name: &str, default| {
            std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
        };
        let n = number("CLASSIC_ROWS", 100_000);
        let width = number("CLASSIC_COLUMNS", 32);
        let repeats = number("CLASSIC_REPEATS", 9);
        let names: Vec<_> = (0..width).map(|i| format!("v{i}")).collect();
        let columns: Vec<Vec<f64>> = (0..width)
            .map(|c| (0..n).map(|r| ((r * (c + 1)) as f64 * 0.013).sin()).collect())
            .collect();
        let data = TabularData::from_f64_columns(
            names.iter().zip(&columns).map(|(name, values)| (name.as_str(), values.as_slice())),
        )
        .unwrap();
        let ids: Vec<_> =
            (2..width).map(|i| VariableId::from_raw(u32::try_from(i).unwrap())).collect();
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        for filtered in [false, true] {
            let mask: Vec<_> = (0..n).map(|i| !filtered || i % 10 != 0).collect();
            let reference = || {
                let t = data.float64_masked(t, &mask).unwrap();
                let y = data.float64_masked(y, &mask).unwrap();
                let covs: Vec<_> =
                    ids.iter().map(|&id| (id, data.float64_masked(id, &mask).unwrap())).collect();
                let refs: Vec<_> = covs.iter().map(|(id, c)| (*id, c.as_slice())).collect();
                let rows: Vec<_> =
                    mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();
                antecedent_stats::CompiledDesign::linear_adjustment(&t, &refs, &y, &rows).unwrap()
            };
            let candidate = || direct_adjustment_design(&data, t, y, &ids, &[], &mask).unwrap();
            let expected = reference();
            let actual = candidate();
            assert_eq!(actual.matrix, expected.matrix);
            assert_eq!(actual.outcome, expected.outcome);
            assert_eq!(actual.row_selection, expected.row_selection);
            let mut modes = ["reference", "direct"];
            if std::env::var_os("CLASSIC_REVERSE").is_some() {
                modes.reverse();
            }
            for mode in modes {
                let mut samples = Vec::new();
                for _ in 0..repeats {
                    let start = std::time::Instant::now();
                    let value = if mode == "reference" { reference() } else { candidate() };
                    samples.push(start.elapsed().as_nanos());
                    std::hint::black_box(value);
                }
                println!(
                    "{{\"experiment\":\"design\",\"mode\":\"{mode}\",\"rows\":{n},\"columns\":{width},\"filtered\":{filtered},\"samples_ns\":{samples:?}}}"
                );
            }
        }
    }
}
