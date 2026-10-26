//! Row accounting for declared missing-data handling.
//!
//! Every route that drops, weights or restricts rows reports the same record: how many rows
//! the table had, how many were analysed, why the others were not, and a digest of exactly
//! which rows they were. The digest is taken over original row positions, so two analyses
//! that excluded the same rows agree and an analysis that excluded different rows does not.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;

use crate::column::ValidityBitmap;
use crate::error::DataError;
use crate::sample_policy::MissingPolicy;
use crate::table::TableView;

/// Domain-separated BLAKE3 context of the dropped-row digest.
const DROPPED_DIGEST_CONTEXT: &str = "antecedent.data.row_accounting.v1";

/// A missing cell: the first (by variable order, then row) invalid value of a requested column.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MissingCell {
    /// Column id.
    pub id: VariableId,
    /// Column name.
    pub variable: Arc<str>,
    /// Zero-based row position in the submitted table.
    pub row: usize,
}

/// Reason a row left the analysed sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DropReason {
    /// The row was hidden by the analysis mask.
    AnalysisMask,
    /// A requested column was missing on the row.
    MissingValue,
    /// The row was complete but its unit was removed by a panel restriction.
    UnitIncomplete,
}

impl DropReason {
    /// Stable name used in receipts.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::AnalysisMask => "analysis_mask",
            Self::MissingValue => "missing_value",
            Self::UnitIncomplete => "unit_incomplete",
        }
    }
}

/// Rows and kept rows of one labelled group (an arm, a treatment group).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupAccounting {
    /// Group label.
    pub label: Arc<str>,
    /// Rows in the group.
    pub total: usize,
    /// Rows of the group that were analysed.
    pub kept: usize,
    /// Rows of the group whose analysed variables are all observed. Equal to `kept` unless
    /// a route keeps rows with a missing value (responder weighting keeps every assigned
    /// row and weights the observed ones).
    pub observed: usize,
}

/// Units removed by a panel restriction.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnitAccounting {
    /// Units in the panel.
    pub total: usize,
    /// Units removed.
    pub dropped: usize,
}

/// The row accounting of one analysis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RowAccounting {
    /// Declared policy the accounting was produced under.
    pub policy: MissingPolicy,
    /// Rows in the submitted table.
    pub rows_total: usize,
    /// Rows whose analysed values entered the estimate.
    pub rows_kept: usize,
    /// Rows the estimate's denominators count. Equal to `rows_kept` unless a route keeps
    /// rows it cannot observe: responder weighting counts every assigned participant in its
    /// arm's denominator and corrects for the non-responders through the weights.
    pub rows_in_denominator: usize,
    /// Rows missing each analysed variable, in the order the variables were requested.
    pub missing_by_variable: Arc<[(Arc<str>, usize)]>,
    /// Dropped rows by first applicable reason.
    pub reasons: Arc<[(DropReason, usize)]>,
    /// Zero-based positions of the dropped rows in the submitted table, ascending.
    pub dropped_rows: Arc<[u64]>,
    /// Lower-case hex BLAKE3 of the row total and the dropped positions.
    pub dropped_digest: Arc<str>,
    /// Per-group totals, when the route has groups (arms, treatment groups).
    pub groups: Arc<[GroupAccounting]>,
    /// Units removed, for panel routes.
    pub units: Option<UnitAccounting>,
}

impl RowAccounting {
    /// Accounting of a selection `keep` over `variables`.
    ///
    /// A dropped row is attributed to the analysis mask when it hides the row, else to a
    /// missing value when a requested column is invalid on it, else to its unit.
    ///
    /// # Errors
    ///
    /// Unknown variables, or a `keep` mask whose length differs from the table's.
    pub fn from_selection<T: TableView>(
        table: &T,
        analysis_mask: Option<&ValidityBitmap>,
        policy: MissingPolicy,
        variables: &[VariableId],
        keep: &ValidityBitmap,
    ) -> Result<Self, DataError> {
        let n = table.row_count();
        if keep.len() != n {
            return Err(DataError::LengthMismatch {
                expected: n,
                actual: keep.len(),
                context: "row accounting keep mask",
            });
        }
        let mut validities = Vec::with_capacity(variables.len());
        let mut missing_by_variable = Vec::with_capacity(variables.len());
        for &id in variables {
            let validity = table.column(id)?.validity();
            let name = Arc::clone(&table.schema().get(id)?.name);
            let missing = n - validity.count_valid();
            missing_by_variable.push((name, missing));
            validities.push(validity);
        }
        let mut counts = [0usize; 3];
        let mut dropped_rows = Vec::new();
        for row in keep.iter_invalid() {
            let reason = if analysis_mask.is_some_and(|mask| !mask.is_valid(row)) {
                DropReason::AnalysisMask
            } else if validities.iter().any(|validity| !validity.is_valid(row)) {
                DropReason::MissingValue
            } else {
                DropReason::UnitIncomplete
            };
            counts[reason as usize] += 1;
            dropped_rows.push(row as u64);
        }
        let reasons =
            [DropReason::AnalysisMask, DropReason::MissingValue, DropReason::UnitIncomplete]
                .into_iter()
                .map(|reason| (reason, counts[reason as usize]))
                .filter(|(_, count)| *count > 0)
                .collect::<Vec<_>>();
        let dropped_digest = dropped_row_digest(n, &dropped_rows);
        Ok(Self {
            policy,
            rows_total: n,
            rows_kept: n - dropped_rows.len(),
            rows_in_denominator: n - dropped_rows.len(),
            missing_by_variable: missing_by_variable.into(),
            reasons: reasons.into(),
            dropped_rows: dropped_rows.into(),
            dropped_digest: Arc::from(dropped_digest),
            groups: Arc::from([]),
            units: None,
        })
    }

    /// Accounting of rows a route dropped for a missing value and identified by position,
    /// for routes whose rows are not a table's rows (the lag-aligned windows of a series).
    ///
    /// `dropped_rows` are ascending positions (original series times for windows).
    #[must_use]
    pub fn from_dropped(
        policy: MissingPolicy,
        rows_total: usize,
        dropped_rows: Vec<u64>,
        missing_by_variable: Vec<(Arc<str>, usize)>,
    ) -> Self {
        let dropped = dropped_rows.len();
        let reasons: Vec<(DropReason, usize)> =
            if dropped == 0 { Vec::new() } else { vec![(DropReason::MissingValue, dropped)] };
        let dropped_digest = dropped_row_digest(rows_total, &dropped_rows);
        Self {
            policy,
            rows_total,
            rows_kept: rows_total - dropped,
            rows_in_denominator: rows_total - dropped,
            missing_by_variable: missing_by_variable.into(),
            reasons: reasons.into(),
            dropped_rows: dropped_rows.into(),
            dropped_digest: Arc::from(dropped_digest),
            groups: Arc::from([]),
            units: None,
        }
    }

    /// Accounting of the complete-case selection over `variables`: rows the analysis mask
    /// hides or where a requested column is missing are dropped.
    ///
    /// # Errors
    ///
    /// Unknown variables.
    pub fn complete_case<T: TableView>(
        table: &T,
        analysis_mask: Option<&ValidityBitmap>,
        variables: &[VariableId],
    ) -> Result<Self, DataError> {
        let keep = complete_rows(table, analysis_mask, variables)?;
        Self::from_selection(table, analysis_mask, MissingPolicy::CompleteCase, variables, &keep)
    }

    /// Accounting of a multiple-imputation analysis over `variables`: every row the
    /// analysis mask exposes is kept (its missing cells are imputed), and only the rows the
    /// mask hides are dropped. `missing_by_variable` still counts the missing cells of each
    /// variable, which are the cells that were imputed.
    ///
    /// # Errors
    ///
    /// Unknown variables, or a policy other than [`MissingPolicy::MultipleImputation`].
    pub fn imputed<T: TableView>(
        table: &T,
        analysis_mask: Option<&ValidityBitmap>,
        policy: MissingPolicy,
        variables: &[VariableId],
    ) -> Result<Self, DataError> {
        if policy.imputation().is_none() {
            return Err(DataError::precondition(
                "imputed row accounting",
                "the multiple_imputation policy",
            ));
        }
        let keep = complete_rows(table, analysis_mask, &[])?;
        Self::from_selection(table, analysis_mask, policy, variables, &keep)
    }

    /// Rows removed from the analysis.
    #[must_use]
    pub fn rows_dropped(&self) -> usize {
        self.rows_total - self.rows_kept
    }

    /// Add per-group totals from one label per table row, the rows kept and the rows whose
    /// analysed variables are all observed.
    ///
    /// # Errors
    ///
    /// `labels`, `keep` or `observed` differ in length from the table.
    pub fn with_groups(
        mut self,
        labels: &[Arc<str>],
        keep: &[bool],
        observed: &[bool],
    ) -> Result<Self, DataError> {
        if labels.len() != self.rows_total
            || keep.len() != self.rows_total
            || observed.len() != self.rows_total
        {
            return Err(DataError::LengthMismatch {
                expected: self.rows_total,
                actual: labels.len().min(keep.len()).min(observed.len()),
                context: "row accounting group labels",
            });
        }
        let mut groups: Vec<GroupAccounting> = Vec::new();
        for ((label, kept), seen) in labels.iter().zip(keep).zip(observed) {
            let slot = if let Some(slot) = groups.iter().position(|g| g.label == *label) {
                slot
            } else {
                groups.push(GroupAccounting {
                    label: Arc::clone(label),
                    total: 0,
                    kept: 0,
                    observed: 0,
                });
                groups.len() - 1
            };
            groups[slot].total += 1;
            groups[slot].kept += usize::from(*kept);
            groups[slot].observed += usize::from(*seen);
        }
        self.groups = groups.into();
        Ok(self)
    }

    /// Record that the estimate's denominators count `rows` rows.
    #[must_use]
    pub fn with_denominator(mut self, rows: usize) -> Self {
        self.rows_in_denominator = rows;
        self
    }

    /// Record the units a panel restriction removed.
    #[must_use]
    pub fn with_units(mut self, total: usize, dropped: usize) -> Self {
        self.units = Some(UnitAccounting { total, dropped });
        self
    }
}

/// Rows the analysis mask exposes and where every requested column is observed, as a bitmap
/// (the word-wise AND of the mask and the requested columns' validity).
///
/// Unlike `TabularData::complete_case_mask`, an empty selection is not an error: the caller
/// is accounting for a selection, not drawing a sample from it.
///
/// # Errors
///
/// Unknown variables.
pub fn complete_rows<T: TableView>(
    table: &T,
    analysis_mask: Option<&ValidityBitmap>,
    variables: &[VariableId],
) -> Result<ValidityBitmap, DataError> {
    let n = table.row_count();
    let mut parts: Vec<&ValidityBitmap> = Vec::with_capacity(variables.len() + 1);
    parts.extend(analysis_mask.filter(|mask| !mask.is_all_valid()));
    for &id in variables {
        let validity = table.column(id)?.validity();
        if !validity.is_all_valid() {
            parts.push(validity);
        }
    }
    if parts.is_empty() {
        return Ok(ValidityBitmap::all_valid(n));
    }
    ValidityBitmap::intersection(n, &parts)
}

/// A copy of `table` whose `Float64` columns named in `replacements` hold the supplied
/// values, all observed. Every other column, the schema, the analysis mask and the weights
/// are kept as they are, so a completed table of a multiple-imputation draw has the
/// schema and row order of the table it completes.
///
/// # Errors
///
/// A replaced variable that is not a `Float64` column of the table, a replacement whose
/// length differs from the row count, a non-finite replacement value, or a storage
/// construction failure.
pub fn complete_float_columns(
    table: &crate::dataset::TabularData,
    replacements: &[(VariableId, Arc<[f64]>)],
) -> Result<crate::dataset::TabularData, DataError> {
    use crate::column::{Float64Column, OwnedColumn};
    let storage = table.storage();
    let n = table.row_count();
    for (id, values) in replacements {
        if values.len() != n {
            return Err(DataError::LengthMismatch {
                expected: n,
                actual: values.len(),
                context: "completed column",
            });
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(DataError::invalid_column(*id, "holds a non-finite completed value"));
        }
        if !storage.columns().iter().any(|c| matches!(c, OwnedColumn::Float64(f) if f.id == *id)) {
            return Err(DataError::TypeMismatch { id: *id, expected: "float64" });
        }
    }
    let mut columns = Vec::with_capacity(storage.columns().len());
    for column in storage.columns() {
        let replaced = replacements.iter().find(|(id, _)| *id == column.id());
        match (column, replaced) {
            (OwnedColumn::Float64(float), Some((_, values))) => {
                columns.push(OwnedColumn::Float64(Float64Column::new(
                    float.id,
                    Arc::clone(values),
                    ValidityBitmap::all_valid(n),
                )?));
            }
            _ => columns.push(column.clone()),
        }
    }
    let rebuilt = crate::storage::OwnedColumnarStorage::try_new(
        table.schema().clone(),
        columns,
        storage.analysis_mask().cloned(),
        storage.weights().map(Arc::from),
    )?;
    Ok(crate::dataset::TabularData::new(rebuilt))
}

/// Lower-case hex BLAKE3 of the row total and the ascending dropped positions.
#[must_use]
pub fn dropped_row_digest(rows_total: usize, dropped: &[u64]) -> String {
    let mut hasher = blake3::Hasher::new_derive_key(DROPPED_DIGEST_CONTEXT);
    hasher.update(&(rows_total as u64).to_le_bytes());
    hasher.update(&(dropped.len() as u64).to_le_bytes());
    for row in dropped {
        hasher.update(&row.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

/// The first missing cell of `variables` in `table`: the first requested variable with an
/// invalid row, and its first such row.
///
/// # Errors
///
/// Unknown variables.
pub fn first_missing_cell<T: TableView>(
    table: &T,
    variables: &[VariableId],
) -> Result<Option<MissingCell>, DataError> {
    for &id in variables {
        let validity = table.column(id)?.validity();
        if validity.is_all_valid() {
            continue;
        }
        if let Some(row) = (0..table.row_count()).find(|&row| !validity.is_valid(row)) {
            return Ok(Some(MissingCell {
                id,
                variable: Arc::clone(&table.schema().get(id)?.name),
                row,
            }));
        }
    }
    Ok(None)
}

/// One chunk of a missing-value conversion.
///
/// Positions in [`Self::dropped_digest`] are local to the chunk. A scan of several
/// chunks does not promote their sum into an end-to-end population bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkMissing {
    /// Rows in the chunk.
    pub rows_total: usize,
    /// Rows kept under the policy.
    pub rows_kept: usize,
    /// Digest of the chunk's dropped local positions.
    pub dropped_digest: Arc<str>,
}

/// Missing-value conversion for a preparation plan.
///
/// [`Self::population_bound`] is true only when the chunks are the whole finite
/// table the estimator will read. A streaming scan leaves it false.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MissingConversion {
    /// Policy that was executed.
    pub policy: MissingPolicy,
    /// Per-chunk accounting.
    pub chunks: Arc<[ChunkMissing]>,
    /// Whether `chunks` is the whole population, not a prefix of a scan.
    pub population_bound: bool,
}

impl MissingConversion {
    /// Convert `variables` under `policy`.
    ///
    /// Complete-case drops incomplete rows and records them. Error-on-missing
    /// refuses the first missing cell. Other policies are not executed here.
    ///
    /// # Errors
    ///
    /// Unknown variables, a policy this conversion does not execute, or a missing
    /// cell under [`MissingPolicy::ErrorOnMissing`].
    pub fn convert<T: TableView>(
        table: &T,
        analysis_mask: Option<&ValidityBitmap>,
        variables: &[VariableId],
        policy: MissingPolicy,
    ) -> Result<Self, DataError> {
        Self::convert_and_select(table, analysis_mask, variables, policy)
            .map(|(conversion, _)| conversion)
    }

    // Preparation consumes both products. Keep one selection rather than rescanning
    // the same columns to reconstruct a mask after producing the receipt.
    pub(crate) fn convert_and_select<T: TableView>(
        table: &T,
        analysis_mask: Option<&ValidityBitmap>,
        variables: &[VariableId],
        policy: MissingPolicy,
    ) -> Result<(Self, ValidityBitmap), DataError> {
        executable_policy(policy)?;
        if policy == MissingPolicy::ErrorOnMissing {
            refuse_error_on_missing(table, variables)?;
        }
        let keep = complete_rows(table, analysis_mask, variables)?;
        let accounting = RowAccounting::from_selection(
            table,
            analysis_mask,
            MissingPolicy::CompleteCase,
            variables,
            &keep,
        )?;
        Ok((Self::from_accounting(policy, &accounting, true), keep))
    }

    /// A scan prefix. `population_bound` stays false, including when every chunk
    /// seen so far is complete.
    ///
    /// # Errors
    ///
    /// A policy this conversion does not execute, or error-on-missing with a dropped row.
    pub fn from_scan(
        policy: MissingPolicy,
        chunks: impl Into<Arc<[ChunkMissing]>>,
    ) -> Result<Self, DataError> {
        executable_policy(policy)?;
        let chunks = chunks.into();
        if policy == MissingPolicy::ErrorOnMissing
            && chunks.iter().any(|chunk| chunk.rows_kept != chunk.rows_total)
        {
            return Err(DataError::precondition(
                "missing-value conversion under error-on-missing",
                "no missing cell",
            ));
        }
        Ok(Self { policy, chunks, population_bound: false })
    }

    fn from_accounting(policy: MissingPolicy, accounting: &RowAccounting, bound: bool) -> Self {
        let chunk = ChunkMissing {
            rows_total: accounting.rows_total,
            rows_kept: accounting.rows_kept,
            dropped_digest: Arc::clone(&accounting.dropped_digest),
        };
        Self { policy, chunks: Arc::from([chunk]), population_bound: bound }
    }

    /// Rows kept, when this conversion is the whole population.
    #[must_use]
    pub fn proved_rows_kept(&self) -> Option<usize> {
        if !self.population_bound || self.chunks.len() != 1 {
            return None;
        }
        Some(self.chunks[0].rows_kept)
    }
}

fn executable_policy(policy: MissingPolicy) -> Result<(), DataError> {
    match policy {
        MissingPolicy::CompleteCase | MissingPolicy::ErrorOnMissing => Ok(()),
        other => Err(DataError::unsupported(
            "missing-value conversion",
            format!(
                "the policy `{}` is not executed; executable policies are complete_case, error",
                other.name()
            ),
        )),
    }
}

fn refuse_error_on_missing<T: TableView>(
    table: &T,
    variables: &[VariableId],
) -> Result<(), DataError> {
    let Some(cell) = first_missing_cell(table, variables)? else {
        return Ok(());
    };
    Err(DataError::MissingPolicyRequired {
        route: "missing-value conversion",
        variable: Arc::clone(&cell.variable),
        row: cell.row,
        available: vec![MissingPolicy::CompleteCase.name(), MissingPolicy::ErrorOnMissing.name()],
    })
}

/// The text every route's refusal of an undeclared policy shares: the first missing cell and
/// the policies the route can execute.
#[must_use]
pub fn policy_required_text(
    cell: &MissingCell,
    route: &str,
    available: &[MissingPolicy],
) -> String {
    format!(
        "{route} found a missing value in column `{}` at row {} and no missing-data policy is \
         declared; declare one of: {}",
        cell.variable,
        cell.row,
        available.iter().map(|policy| policy.name()).collect::<Vec<_>>().join(", "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TabularData;

    fn table() -> TabularData {
        let x = [1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0];
        let y = [1.0, 2.0, f64::NAN, 4.0, f64::NAN, 6.0];
        TabularData::from_f64_columns([("x", &x[..]), ("y", &y[..])]).unwrap()
    }

    fn ids() -> [VariableId; 2] {
        [VariableId::from_raw(0), VariableId::from_raw(1)]
    }

    #[test]
    fn validity_fast_paths_match_scalar_selection_and_accounting() {
        for n in [1usize, 2, 7, 8, 9, 63, 64, 65] {
            for missing_x in [false, true] {
                for missing_y in [false, true] {
                    let mut x = vec![1.0; n];
                    let mut y = vec![2.0; n];
                    for i in 0..n {
                        if missing_x && i % 7 == 1 {
                            x[i] = f64::NAN;
                        }
                        if missing_y && i % 11 == 2 {
                            y[i] = f64::NAN;
                        }
                    }
                    for masked in [false, true] {
                        let table = TabularData::from_f64_columns([
                            ("x", x.as_slice()),
                            ("y", y.as_slice()),
                        ])
                        .unwrap();
                        let bytes = vec![if masked { 0x55 } else { 0xff }; n.div_ceil(8)];
                        let mask = ValidityBitmap::from_bytes(bytes, n).unwrap();
                        let table = table.with_analysis_mask(mask).unwrap();
                        let expected: Vec<bool> = (0..n)
                            .map(|i| (!masked || i % 2 == 0) && !x[i].is_nan() && !y[i].is_nan())
                            .collect();
                        let actual =
                            complete_rows(&table, table.storage().analysis_mask(), &ids()).unwrap();
                        assert_eq!(actual.to_bools(), expected);
                        if expected.iter().any(|keep| *keep) {
                            assert_eq!(table.complete_case_mask(&ids()).unwrap(), expected);
                        } else {
                            assert!(table.complete_case_mask(&ids()).is_err());
                        }
                        let accounting = RowAccounting::complete_case(
                            &table,
                            table.storage().analysis_mask(),
                            &ids(),
                        )
                        .unwrap();
                        let dropped: Vec<u64> = (0..n)
                            .filter(|i| !expected[*i])
                            .map(|i| u64::try_from(i).unwrap())
                            .collect();
                        assert_eq!(accounting.dropped_rows.as_ref(), dropped.as_slice());
                        assert_eq!(
                            accounting.rows_kept,
                            expected.iter().filter(|keep| **keep).count()
                        );
                        assert_eq!(
                            accounting.missing_by_variable[0].1,
                            x.iter().filter(|v| v.is_nan()).count()
                        );
                        assert_eq!(
                            accounting.missing_by_variable[1].1,
                            y.iter().filter(|v| v.is_nan()).count()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn first_missing_cell_names_the_variable_and_row() {
        let data = table();
        let cell = first_missing_cell(&data, &ids()).unwrap().unwrap();
        assert_eq!((cell.variable.as_ref(), cell.row), ("x", 1));
        let only_y = first_missing_cell(&data, &ids()[1..]).unwrap().unwrap();
        assert_eq!((only_y.variable.as_ref(), only_y.row), ("y", 2));
        let complete = TabularData::from_f64_columns([("z", &[1.0, 2.0][..])]).unwrap();
        assert_eq!(first_missing_cell(&complete, &[VariableId::from_raw(0)]).unwrap(), None);
    }

    #[test]
    fn accounting_counts_dropped_rows_by_variable_and_reason() {
        let data = table();
        let keep = ValidityBitmap::from_bools(&data.complete_case_mask(&ids()).unwrap());
        let accounting =
            RowAccounting::from_selection(&data, None, MissingPolicy::CompleteCase, &ids(), &keep)
                .unwrap();
        assert_eq!((accounting.rows_total, accounting.rows_kept), (6, 3));
        assert_eq!(accounting.rows_dropped(), 3);
        let by_variable: Vec<_> =
            accounting.missing_by_variable.iter().map(|(n, c)| (n.as_ref(), *c)).collect();
        assert_eq!(by_variable, [("x", 1), ("y", 2)]);
        assert_eq!(accounting.dropped_rows.as_ref(), [1, 2, 4]);
        assert_eq!(accounting.reasons.as_ref(), [(DropReason::MissingValue, 3)]);
    }

    #[test]
    fn the_refusal_text_names_the_cell_and_the_policies() {
        let data = table();
        let cell = first_missing_cell(&data, &ids()).unwrap().unwrap();
        let text = policy_required_text(
            &cell,
            "a route",
            &[MissingPolicy::CompleteCase, MissingPolicy::BalancedPanel],
        );
        assert!(text.contains("column `x` at row 1"), "{text}");
        assert!(text.ends_with("declare one of: complete_case, balanced_panel"), "{text}");
    }

    #[test]
    fn digest_identifies_which_rows_were_dropped() {
        let one = dropped_row_digest(6, &[1, 2, 4]);
        assert_eq!(one, dropped_row_digest(6, &[1, 2, 4]));
        assert_ne!(one, dropped_row_digest(6, &[1, 2, 5]));
        assert_ne!(one, dropped_row_digest(7, &[1, 2, 4]));
        assert_eq!(one.len(), 64);
    }

    #[test]
    fn complete_rows_removed_with_their_unit_are_attributed_to_the_unit() {
        let data = table();
        let mut flags = data.complete_case_mask(&ids()).unwrap();
        flags[3] = false;
        let keep = ValidityBitmap::from_bools(&flags);
        let accounting =
            RowAccounting::from_selection(&data, None, MissingPolicy::BalancedPanel, &ids(), &keep)
                .unwrap()
                .with_units(3, 1);
        assert_eq!(
            accounting.reasons.as_ref(),
            [(DropReason::MissingValue, 3), (DropReason::UnitIncomplete, 1)]
        );
        assert_eq!(accounting.units, Some(UnitAccounting { total: 3, dropped: 1 }));
    }

    #[test]
    fn group_totals_follow_the_labels() {
        let data = table();
        let flags = data.complete_case_mask(&ids()).unwrap();
        let keep = ValidityBitmap::from_bools(&flags);
        let labels: Vec<Arc<str>> = ["a", "a", "a", "b", "b", "b"].map(Arc::from).to_vec();
        let accounting =
            RowAccounting::from_selection(&data, None, MissingPolicy::CompleteCase, &ids(), &keep)
                .unwrap()
                .with_groups(&labels, &flags, &flags)
                .unwrap();
        let groups: Vec<_> = accounting
            .groups
            .iter()
            .map(|g| (g.label.as_ref(), g.total, g.kept, g.observed))
            .collect();
        assert_eq!(groups, [("a", 3, 1, 1), ("b", 3, 2, 2)]);
    }

    #[test]
    fn complete_case_conversion_records_the_population_and_error_on_missing_refuses() {
        let data = table();
        let converted =
            MissingConversion::convert(&data, None, &ids(), MissingPolicy::CompleteCase).unwrap();
        assert!(converted.population_bound);
        assert_eq!(converted.proved_rows_kept(), Some(3));
        assert_eq!(converted.policy, MissingPolicy::CompleteCase);
        let refused =
            MissingConversion::convert(&data, None, &ids(), MissingPolicy::ErrorOnMissing);
        assert!(refused.is_err());
        let pairwise =
            MissingConversion::convert(&data, None, &ids(), MissingPolicy::PairwiseAvailable);
        let message = pairwise.expect_err("pairwise is not executed").to_string();
        assert!(message.contains("`pairwise_available` is not executed"), "{message}");
    }

    #[test]
    fn a_scan_conversion_is_not_a_population_bound() {
        let scan = MissingConversion::from_scan(
            MissingPolicy::CompleteCase,
            [ChunkMissing {
                rows_total: 4,
                rows_kept: 4,
                dropped_digest: Arc::from(dropped_row_digest(4, &[])),
            }],
        )
        .unwrap();
        assert!(!scan.population_bound);
        assert_eq!(scan.proved_rows_kept(), None);
    }
}
