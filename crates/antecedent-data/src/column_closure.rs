//! Required-column closure, computed before projection.
//!
//! Validation, support and positivity, weights, splits, and complete-case rules
//! name columns the scientific target may not. Those columns are part of the
//! cohort: projecting to the target first, then applying complete-case, can
//! keep a row the closure would have dropped. A later projection or filter
//! pushdown that would do that is a failed attempt. The diagnostic names the
//! column that must be retained, and the attempt does not return a narrower
//! column set.
//!
//! The file-backed table source plans every read from this closure: only the
//! closed columns are decoded, and a projection that would drop one of them is
//! refused before a byte is read.
//!
//! Column identity is an owned name ([`ColumnRole`]). Snapshot schema names are
//! the same kind of string; a dense variable id is assigned when a schema is
//! built and is remapped by projection, so it is the wrong key for this check.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

/// Owned column name. Equality is the name, not the allocation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ColumnRole(Arc<str>);

impl ColumnRole {
    /// Column named `name`.
    #[must_use]
    pub fn new(name: impl AsRef<str>) -> Self {
        Self(Arc::from(name.as_ref()))
    }

    /// Borrow the name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ColumnRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for ColumnRole {
    fn from(name: &str) -> Self {
        Self::new(name)
    }
}

impl From<Arc<str>> for ColumnRole {
    fn from(name: Arc<str>) -> Self {
        Self(name)
    }
}

impl AsRef<str> for ColumnRole {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// Why a column is in the closure. This is not a schema role hint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ColumnDuty {
    /// Named by the scientific target.
    Target,
    /// Mandatory validator. A row missing it is out of the cohort.
    Validator,
    /// Support or positivity column.
    Support,
    /// Sampling or observation weight.
    Weight,
    /// Split or fold assignment.
    Split,
    /// Any other column the complete-case rule reads.
    CompleteCase,
}

impl ColumnDuty {
    /// Stable name for diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Target => "target",
            Self::Validator => "validator",
            Self::Support => "support",
            Self::Weight => "weight",
            Self::Split => "split",
            Self::CompleteCase => "complete_case",
        }
    }
}

/// One column the closure retains, with the duty that first required it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequiredColumn {
    /// Column name.
    pub role: ColumnRole,
    /// First duty that named this column. Later duties do not replace it.
    pub duty: ColumnDuty,
}

/// Obligations that pull columns into the closure. Build with [`Self::new`],
/// then [`Self::close`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClosureInputs {
    target: Vec<ColumnRole>,
    validators: Vec<ColumnRole>,
    support: Vec<ColumnRole>,
    weights: Vec<ColumnRole>,
    splits: Vec<ColumnRole>,
    complete_case: Vec<ColumnRole>,
}

impl ClosureInputs {
    /// Scientific target. Other duties start empty.
    #[must_use]
    pub fn new(target: impl IntoIterator<Item = ColumnRole>) -> Self {
        Self {
            target: target.into_iter().collect(),
            validators: Vec::new(),
            support: Vec::new(),
            weights: Vec::new(),
            splits: Vec::new(),
            complete_case: Vec::new(),
        }
    }

    /// Require a validator column.
    #[must_use]
    pub fn validator(mut self, column: ColumnRole) -> Self {
        self.validators.push(column);
        self
    }

    /// Require a support or positivity column.
    #[must_use]
    pub fn support(mut self, column: ColumnRole) -> Self {
        self.support.push(column);
        self
    }

    /// Require a weight column.
    #[must_use]
    pub fn weight(mut self, column: ColumnRole) -> Self {
        self.weights.push(column);
        self
    }

    /// Require a split column.
    #[must_use]
    pub fn split(mut self, column: ColumnRole) -> Self {
        self.splits.push(column);
        self
    }

    /// Require a column the complete-case rule reads.
    #[must_use]
    pub fn complete_case(mut self, column: ColumnRole) -> Self {
        self.complete_case.push(column);
        self
    }

    /// Union the duties. Order is target, validator, support, weight, split,
    /// then complete-case. A repeated name keeps the earliest duty.
    #[must_use]
    pub fn close(self) -> ColumnClosure {
        let mut seen = HashSet::new();
        let mut columns = Vec::new();
        remember(&mut seen, &mut columns, ColumnDuty::Target, self.target);
        remember(&mut seen, &mut columns, ColumnDuty::Validator, self.validators);
        remember(&mut seen, &mut columns, ColumnDuty::Support, self.support);
        remember(&mut seen, &mut columns, ColumnDuty::Weight, self.weights);
        remember(&mut seen, &mut columns, ColumnDuty::Split, self.splits);
        remember(&mut seen, &mut columns, ColumnDuty::CompleteCase, self.complete_case);
        ColumnClosure { columns }
    }
}

/// Columns that must be present before projection and before complete-case.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColumnClosure {
    columns: Vec<RequiredColumn>,
}

impl ColumnClosure {
    /// Required columns in closure order.
    #[must_use]
    pub fn columns(&self) -> &[RequiredColumn] {
        &self.columns
    }

    /// Whether `role` is required.
    #[must_use]
    pub fn contains(&self, role: &ColumnRole) -> bool {
        self.duty(role).is_some()
    }

    /// Duty that first required `role`.
    #[must_use]
    pub fn duty(&self, role: &ColumnRole) -> Option<ColumnDuty> {
        self.columns.iter().find(|column| &column.role == role).map(|column| column.duty)
    }

    /// Keep this closure only when `keep` contains every required column.
    ///
    /// Success returns `self`. It does not build a narrower column list.
    /// Failure names the earliest omitted column in closure order.
    ///
    /// # Errors
    ///
    /// [`ClosureError::Refused`] when a required column is absent from `keep`.
    pub fn project(&self, keep: &[ColumnRole]) -> Result<&Self, ClosureError> {
        for column in &self.columns {
            if !keep.contains(&column.role) {
                return Err(ClosureError::Refused(PushdownRefusal { retain: column.role.clone() }));
            }
        }
        Ok(self)
    }

    /// Complete-case cohort on every column in this closure.
    ///
    /// A row is kept when every required column is observed. Call this before
    /// projecting. Projecting to the target and then completing cases is a
    /// different cohort; [`ObservationFrame::complete_case`] on the target
    /// names alone is that different cohort.
    ///
    /// # Errors
    ///
    /// [`ClosureError::NoColumns`] when the closure is empty.
    /// [`ClosureError::UnknownColumn`] when a required column is not in `frame`.
    /// [`ClosureError::EmptyCohort`] when no row is observed on every required column.
    pub fn complete_case(&self, frame: &ObservationFrame) -> Result<Cohort, ClosureError> {
        let roles: Vec<ColumnRole> =
            self.columns.iter().map(|column| column.role.clone()).collect();
        frame.complete_case(&roles)
    }

    /// Push `filter` only when it drops no row this closure's complete-case keeps.
    ///
    /// The returned cohort is that complete-case cohort, so an allowed filter
    /// does not change it. A filter that drops a kept row is refused. The
    /// diagnostic names the filter column: that column still decides membership,
    /// and this method does not project it away.
    ///
    /// # Errors
    ///
    /// [`ClosureError::LengthMismatch`] when the drop mask length is not the
    /// frame's row count. [`ClosureError::Refused`] when a kept row would be
    /// dropped. Also the errors of [`Self::complete_case`].
    pub fn push_filter(
        &self,
        frame: &ObservationFrame,
        filter: &ColumnFilter,
    ) -> Result<Cohort, ClosureError> {
        if filter.drop_row.len() != frame.rows {
            return Err(ClosureError::LengthMismatch {
                column: filter.column.clone(),
                expected: frame.rows,
                actual: filter.drop_row.len(),
            });
        }
        let cohort = self.complete_case(frame)?;
        for &row in cohort.rows() {
            if filter.drop_row[row] {
                return Err(ClosureError::Refused(PushdownRefusal {
                    retain: filter.column.clone(),
                }));
            }
        }
        Ok(cohort)
    }
}

/// Observed (`true`) or missing (`false`) cells, one mask per column.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationFrame {
    columns: Vec<ColumnMask>,
    rows: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ColumnMask {
    role: ColumnRole,
    observed: Vec<bool>,
}

impl ObservationFrame {
    /// Build a frame. Every column has the same number of rows.
    ///
    /// # Errors
    ///
    /// [`ClosureError::NoColumns`] when `columns` is empty.
    /// [`ClosureError::DuplicateColumn`] when a name is repeated.
    /// [`ClosureError::LengthMismatch`] when a column's length differs from the first.
    pub fn try_new(columns: Vec<(ColumnRole, Vec<bool>)>) -> Result<Self, ClosureError> {
        let mut rows = None;
        let mut seen = HashSet::new();
        let mut stored = Vec::with_capacity(columns.len());
        for (role, observed) in columns {
            if !seen.insert(role.clone()) {
                return Err(ClosureError::DuplicateColumn { column: role });
            }
            if let Some(expected) = rows {
                if expected != observed.len() {
                    return Err(ClosureError::LengthMismatch {
                        column: role,
                        expected,
                        actual: observed.len(),
                    });
                }
            } else {
                rows = Some(observed.len());
            }
            stored.push(ColumnMask { role, observed });
        }
        let Some(rows) = rows else {
            return Err(ClosureError::NoColumns);
        };
        Ok(Self { columns: stored, rows })
    }

    /// Number of rows.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.rows
    }

    /// Rows where every named column is observed, in ascending row index.
    ///
    /// Passing only the scientific target is the cohort you get by projecting
    /// those columns away from the rest and then applying complete-case. That
    /// can be larger than [`ColumnClosure::complete_case`].
    ///
    /// # Errors
    ///
    /// [`ClosureError::NoColumns`] when `columns` is empty.
    /// [`ClosureError::UnknownColumn`] when a name is not in this frame.
    /// [`ClosureError::EmptyCohort`] when every row is missing at least one named column.
    pub fn complete_case(&self, columns: &[ColumnRole]) -> Result<Cohort, ClosureError> {
        if columns.is_empty() {
            return Err(ClosureError::NoColumns);
        }
        let mut masks = Vec::with_capacity(columns.len());
        for column in columns {
            masks.push(self.mask(column)?);
        }
        let mut kept = Vec::new();
        for row in 0..self.rows {
            let mut observed = true;
            for mask in &masks {
                if !mask[row] {
                    observed = false;
                    break;
                }
            }
            if observed {
                kept.push(row);
            }
        }
        if kept.is_empty() {
            return Err(ClosureError::EmptyCohort);
        }
        Ok(Cohort { rows: kept })
    }

    fn mask(&self, role: &ColumnRole) -> Result<&[bool], ClosureError> {
        for column in &self.columns {
            if &column.role == role {
                return Ok(column.observed.as_slice());
            }
        }
        Err(ClosureError::UnknownColumn { column: role.clone() })
    }
}

/// Ascending row indexes kept by complete-case.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cohort {
    rows: Vec<usize>,
}

impl Cohort {
    /// Kept rows.
    #[must_use]
    pub fn rows(&self) -> &[usize] {
        &self.rows
    }
}

/// A declared row filter attributed to one column.
///
/// `drop_row[i]` drops storage row `i`. The column is the one the predicate
/// reads; a refusal names it instead of applying the filter and projecting.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColumnFilter {
    column: ColumnRole,
    drop_row: Vec<bool>,
}

impl ColumnFilter {
    /// Filter that drops rows marked `true`, read off `column`.
    #[must_use]
    pub fn new(column: ColumnRole, drop_row: Vec<bool>) -> Self {
        Self { column, drop_row }
    }

    /// Column the predicate reads.
    #[must_use]
    pub fn column(&self) -> &ColumnRole {
        &self.column
    }
}

/// A projection or filter pushdown that would change the cohort.
///
/// The attempt failed. `retain` is the column that must stay. No projected
/// column list is produced.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PushdownRefusal {
    /// Column that must be retained.
    pub retain: ColumnRole,
}

/// Failure while building a frame, closing a cohort, or refusing a pushdown.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ClosureError {
    /// A column's row count, or a filter mask, disagrees with the frame.
    LengthMismatch {
        /// Column whose length was wrong, or the filter column.
        column: ColumnRole,
        /// Length the frame already fixed.
        expected: usize,
        /// Length that was supplied.
        actual: usize,
    },
    /// Complete-case was asked for a column the frame does not have.
    UnknownColumn {
        /// Missing name.
        column: ColumnRole,
    },
    /// The frame named one column twice.
    DuplicateColumn {
        /// Repeated name.
        column: ColumnRole,
    },
    /// A closure or complete-case list named no columns.
    NoColumns,
    /// Complete-case kept nothing.
    EmptyCohort,
    /// The pushdown is not legal. See [`PushdownRefusal`].
    Refused(PushdownRefusal),
}

impl fmt::Display for ClosureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { column, expected, actual } => {
                write!(f, "{column}: expected length {expected}, got {actual}")
            }
            Self::UnknownColumn { column } => write!(f, "unknown column '{column}'"),
            Self::DuplicateColumn { column } => write!(f, "duplicate column '{column}'"),
            Self::NoColumns => write!(f, "column closure has no columns"),
            Self::EmptyCohort => write!(f, "complete-case kept no rows"),
            Self::Refused(PushdownRefusal { retain }) => {
                write!(f, "pushdown refused: column '{retain}' must be retained")
            }
        }
    }
}

impl std::error::Error for ClosureError {}

fn remember(
    seen: &mut HashSet<ColumnRole>,
    out: &mut Vec<RequiredColumn>,
    duty: ColumnDuty,
    roles: Vec<ColumnRole>,
) {
    for role in roles {
        if seen.insert(role.clone()) {
            out.push(RequiredColumn { role, duty });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four rows. Complete-case on the closure keeps only row 0.
    ///
    /// | row | outcome | treatment | validator | weight |
    /// | 0   | yes     | yes       | yes       | yes    |
    /// | 1   | yes     | yes       | missing   | yes    |
    /// | 2   | yes     | missing   | yes       | yes    |
    /// | 3   | yes     | yes       | yes       | missing|
    ///
    /// `support`, `split`, and `covariate` are observed on every row. They are
    /// still required. Projecting to outcome and treatment first keeps rows
    /// 0, 1, and 3.
    struct Sample {
        frame: ObservationFrame,
        closure: ColumnClosure,
        outcome: ColumnRole,
        treatment: ColumnRole,
        validator: ColumnRole,
    }

    fn role(name: &str) -> ColumnRole {
        ColumnRole::new(name)
    }

    fn sample() -> Sample {
        let outcome = role("outcome");
        let treatment = role("treatment");
        let validator = role("validator");
        let support = role("support");
        let weight = role("weight");
        let split = role("split");
        let covariate = role("covariate");
        let yes = vec![true, true, true, true];
        let frame = ObservationFrame::try_new(vec![
            (outcome.clone(), yes.clone()),
            (treatment.clone(), vec![true, true, false, true]),
            (validator.clone(), vec![true, false, true, true]),
            (support.clone(), yes.clone()),
            (weight.clone(), vec![true, true, true, false]),
            (split.clone(), yes.clone()),
            (covariate.clone(), yes),
        ])
        .unwrap();
        let closure = ClosureInputs::new([outcome.clone(), treatment.clone()])
            .validator(validator.clone())
            .support(support)
            .weight(weight)
            .split(split)
            .complete_case(covariate)
            .close();
        Sample { frame, closure, outcome, treatment, validator }
    }

    #[test]
    fn projecting_to_the_target_before_complete_case_changes_the_cohort() {
        let sample = sample();
        let target_only = sample.frame.complete_case(&[sample.outcome, sample.treatment]).unwrap();
        let closed = sample.closure.complete_case(&sample.frame).unwrap();
        assert_eq!(target_only.rows(), &[0, 1, 3]);
        assert_eq!(closed.rows(), &[0]);
        assert_ne!(target_only, closed);
    }

    #[test]
    fn closure_includes_the_validator_so_a_later_projection_cannot_omit_it() {
        let sample = sample();
        assert_eq!(sample.closure.duty(&sample.validator), Some(ColumnDuty::Validator));
        let err = sample.closure.project(&[sample.outcome, sample.treatment]).unwrap_err();
        assert_eq!(
            err,
            ClosureError::Refused(PushdownRefusal { retain: ColumnRole::new("validator") })
        );
        assert!(err.to_string().contains("validator"));
        assert!(sample.closure.contains(&sample.validator));

        let keep: Vec<ColumnRole> =
            sample.closure.columns().iter().map(|column| column.role.clone()).collect();
        let retained = sample.closure.project(&keep).unwrap();
        assert!(std::ptr::eq(retained, &sample.closure));
        assert!(retained.contains(&sample.validator));
    }

    #[test]
    fn filter_pushdown_that_drops_a_kept_row_is_refused() {
        let sample = sample();
        let filter = ColumnFilter::new(role("weight"), vec![true, false, false, false]);
        let err = sample.closure.push_filter(&sample.frame, &filter).unwrap_err();
        assert_eq!(
            err,
            ClosureError::Refused(PushdownRefusal { retain: ColumnRole::new("weight") })
        );
        let text = err.to_string();
        assert!(text.contains("weight"));
        assert!(text.contains("must be retained"));
        assert!(sample.closure.contains(&sample.validator));
        assert_eq!(sample.closure.columns().len(), 7);
    }

    #[test]
    fn filter_that_drops_only_excluded_rows_keeps_the_cohort() {
        let sample = sample();
        let closed = sample.closure.complete_case(&sample.frame).unwrap();
        let filter = ColumnFilter::new(role("weight"), vec![false, true, true, true]);
        let pushed = sample.closure.push_filter(&sample.frame, &filter).unwrap();
        assert_eq!(pushed.rows(), &[0]);
        assert_eq!(pushed, closed);
    }

    #[test]
    fn closure_includes_support_weights_splits_and_complete_case_columns() {
        let sample = sample();
        let expected = [
            ("outcome", ColumnDuty::Target),
            ("treatment", ColumnDuty::Target),
            ("validator", ColumnDuty::Validator),
            ("support", ColumnDuty::Support),
            ("weight", ColumnDuty::Weight),
            ("split", ColumnDuty::Split),
            ("covariate", ColumnDuty::CompleteCase),
        ];
        assert_eq!(sample.closure.columns().len(), expected.len());
        for (index, (name, duty)) in expected.iter().enumerate() {
            let column = &sample.closure.columns()[index];
            assert_eq!(column.role.as_str(), *name);
            assert_eq!(column.duty, *duty);
            assert_eq!(column.duty.name(), duty.name());
        }
        assert!(!sample.closure.contains(&role("noise")));
    }
}
