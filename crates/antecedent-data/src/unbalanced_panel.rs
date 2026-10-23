//! Unbalanced panels: gaps, late entry and attrition.
//!
//! A long panel table carries one row per observed unit-period. [`PanelLayout::assess`]
//! reads its unit labels, its periods and which rows are complete, and reports the
//! calendar grid, each unit's coverage of it, and the attrition across it. The report
//! prepares the rows for analysis without crossing a unit or a calendar boundary:
//! whole units, single complete unit-periods, or each unit's runs of consecutive grid
//! periods. Designs that need a balanced panel are refused with the units and periods
//! that break balance and the attempt to try instead.
//!
//! The grid is the sorted set of distinct periods in the table. "Consecutive" means
//! adjacent on that grid; [`PanelBalanceReport::grid_step`] says whether the grid is
//! regular in calendar time.
//!
//! A unit-period counts as observed only when its row is present and complete. A
//! present row with a missing analysed value is a gap like an absent row.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::error::Error;
use std::fmt;
use std::fmt::Write as _;
use std::ops::Range;
use std::sync::Arc;

use antecedent_core::named_variants;
use antecedent_core::refusal::NextAttempt;

use crate::error::DataError;
use crate::missing::{DropReason, RowAccounting};
use crate::panel_window::PopulationUnit;
use crate::sample_policy::MissingPolicy;

/// Domain-separated BLAKE3 context of the panel layout digest.
const LAYOUT_DIGEST_CONTEXT: &str = "antecedent.data.panel_layout";

/// Units or cells named in one refusal message, at most.
const MAX_LISTED: usize = 5;

/// Characters of a unit label quoted in a refusal message, at most.
const MAX_LABEL_CHARS: usize = 64;

const ARGUMENT_CODE: &str = antecedent_core::reason_code!("invalid_argument");
const POLICY_CODE: &str = antecedent_core::reason_code!("missing_policy_not_applicable");

const LENGTH_DETAIL: &str = "panel_preparation.length_mismatch";
const EMPTY_DETAIL: &str = "panel_preparation.empty";
const DUPLICATE_DETAIL: &str = "panel_preparation.duplicate_unit_period";
const TOO_MANY_UNITS_DETAIL: &str = "panel_preparation.too_many_units";
const NOTHING_KEPT_DETAIL: &str = "panel_preparation.nothing_kept";
const GRID_SHAPE_DETAIL: &str = "panel_preparation.grid_shape";
const UNBALANCED_DETAIL: &str = "panel_preparation.unbalanced";
const LAG_RANGE_DETAIL: &str = "panel_preparation.lag_out_of_range";
const IRREGULAR_GRID_DETAIL: &str = "panel_preparation.irregular_grid";
const LAG_GAP_DETAIL: &str = "panel_preparation.lag_crosses_gap";

/// Drop every unit that is incomplete in any period; single unit-periods are never dropped.
pub const DECLARE_BALANCED_PANEL: NextAttempt = NextAttempt::new(
    "balanced_panel",
    "declare missing=balanced_panel to drop incomplete units whole",
);

/// Let each contrast use the units observed in the periods it needs.
pub const DECLARE_AVAILABLE_CASE: NextAttempt = NextAttempt::new(
    "available_case",
    "declare missing=available_case so each contrast uses the units observed in its own periods",
);

/// Restrict units and periods to a window in which every kept unit is complete.
pub const RESTRICT_TO_COMMON_WINDOW: NextAttempt = NextAttempt::new(
    "common_window",
    "restrict the units and periods to a common window in which every kept unit is complete",
);

/// Balance the panel before building row-aligned nuisance columns.
pub const BALANCE_BEFORE_NUISANCE: NextAttempt = NextAttempt::new(
    "balanced_units_before_nuisance",
    "prepare balanced units first and build the row-aligned nuisance columns on the kept rows",
);

/// Take lags inside each contiguous run of one unit.
pub const LAG_INSIDE_CONTIGUOUS_RUNS: NextAttempt = NextAttempt::new(
    "contiguous_runs",
    "prepare contiguous runs and lag inside each run; a longer span of the same unit, never \
     a borrow across a gap",
);

/// Submit the rows on a grid whose periods are evenly spaced in calendar time.
pub const SUBMIT_REGULAR_GRID: NextAttempt = NextAttempt::new(
    "regular_grid",
    "submit periods on an evenly spaced calendar grid so a lag counts calendar steps",
);

/// Every next attempt this module names.
pub const PANEL_PREPARATION_NEXT_ATTEMPTS: [NextAttempt; 6] = [
    DECLARE_BALANCED_PANEL,
    DECLARE_AVAILABLE_CASE,
    RESTRICT_TO_COMMON_WINDOW,
    BALANCE_BEFORE_NUISANCE,
    LAG_INSIDE_CONTIGUOUS_RUNS,
    SUBMIT_REGULAR_GRID,
];

/// A refused panel layout, preparation, design or lag.
#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use = "a refused panel preparation names the next attempt"]
pub struct PanelPreparationError {
    code: &'static str,
    detail: &'static str,
    message: String,
    next: Option<NextAttempt>,
}

impl PanelPreparationError {
    fn new(
        code: &'static str,
        detail: &'static str,
        message: String,
        next: Option<NextAttempt>,
    ) -> Self {
        let message = match next {
            Some(attempt) => attempt.append_to(&message),
            None => message,
        };
        Self { code, detail, message, next }
    }

    fn argument(detail: &'static str, message: String) -> Self {
        Self::new(ARGUMENT_CODE, detail, message, None)
    }

    /// Registered reason code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }

    /// Namespaced detail, `panel_preparation.<slot>`.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }

    /// What was refused, ending with the next attempt when there is one.
    #[must_use]
    pub fn message(&self) -> &str {
        self.message.as_str()
    }

    /// What to try instead, when the refusal names one.
    #[must_use]
    pub const fn next_attempt(&self) -> Option<NextAttempt> {
        self.next
    }
}

impl fmt::Display for PanelPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.detail, self.message)
    }
}

impl Error for PanelPreparationError {}

antecedent_core::impl_refused!(PanelPreparationError);

impl From<PanelPreparationError> for DataError {
    fn from(error: PanelPreparationError) -> Self {
        Self::InvalidArgument { message: error.to_string() }
    }
}

/// Inclusive run of consecutive grid periods.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PeriodRun {
    /// First grid period of the run.
    pub first: i64,
    /// Last grid period of the run, inclusive.
    pub last: i64,
    /// Grid periods in the run.
    pub periods: usize,
}

/// One unit's coverage of the calendar grid.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnitSummary {
    /// The unit. Its id is the rank of its label in byte order.
    pub unit: PopulationUnit,
    /// Unit label as submitted.
    pub label: Arc<str>,
    /// First period with a complete row, if any.
    pub first_observed: Option<i64>,
    /// Last period with a complete row, if any.
    pub last_observed: Option<i64>,
    /// Periods with a row.
    pub periods_present: usize,
    /// Periods with a complete row.
    pub periods_complete: usize,
    /// Runs of grid periods without a complete row between the first and last observed
    /// periods.
    pub gaps: Box<[PeriodRun]>,
    /// The first observed period is after the grid start.
    pub late_entry: bool,
    /// The last observed period is before the grid end.
    pub attrited: bool,
    /// Complete in every grid period.
    pub balanced: bool,
    /// Input row of the first observed period, if any.
    pub first_observed_row: Option<usize>,
}

/// Units at risk, observed, entering and leaving in one grid period.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AttritionRow {
    /// Grid period.
    pub period: i64,
    /// Units whose first observed period is at or before this period and whose last is at
    /// or after it.
    pub at_risk: usize,
    /// Units with a complete row in this period.
    pub observed: usize,
    /// Units whose first observed period is this period.
    pub entering: usize,
    /// Units whose last observed period is this period, when it is not the grid end.
    pub leaving: usize,
}

impl AttritionRow {
    /// Share of the units at risk that leave after this period, or `None` when none are at
    /// risk.
    #[must_use]
    pub fn hazard(&self) -> Option<f64> {
        (self.at_risk > 0).then(|| self.leaving as f64 / self.at_risk as f64)
    }
}

/// A baseline variable at each unit's first observed period, compared between units that
/// attrite and units observed to the grid end.
///
/// A descriptive diagnostic of whether attrition is related to baseline levels. It makes
/// no causal claim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttritionBaseline {
    /// Units observed to the grid end with a finite baseline value.
    pub stayers: usize,
    /// Units that attrite with a finite baseline value.
    pub attriters: usize,
    /// Observed units whose baseline value is missing or not finite.
    pub missing_baseline: usize,
    /// Mean baseline of the stayers.
    pub mean_stayers: Option<f64>,
    /// Mean baseline of the attriters.
    pub mean_attriters: Option<f64>,
    /// Attriters' mean minus stayers' mean, over the root of the average of the two sample
    /// variances. `None` when a group has fewer than two units or both variances are zero.
    pub standardized_difference: Option<f64>,
}

/// Entry point of the panel layout assessment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PanelLayout;

impl PanelLayout {
    /// Assess a long panel table: `subjects[i]` and `periods[i]` identify row `i`, and
    /// `observed[i]` says whether its analysed values are complete (see
    /// [`complete_rows`](crate::missing::complete_rows)).
    ///
    /// Time and memory are proportional to the rows (plus a sort).
    ///
    /// # Errors
    ///
    /// Slices of different lengths, an empty table, a repeated (unit, period) row, or more
    /// than `u32::MAX` units.
    pub fn assess<S: AsRef<str>>(
        subjects: &[S],
        periods: &[i64],
        observed: &[bool],
    ) -> Result<PanelBalanceReport, PanelPreparationError> {
        let n = subjects.len();
        let order = canonical_row_order(subjects, periods, observed)?;
        let mut grid: Vec<i64> = periods.to_vec();
        grid.sort_unstable();
        grid.dedup();
        let grid_step = regular_step(&grid);
        let row_grid: Vec<usize> = order
            .iter()
            .map(|&row| match grid.binary_search(&periods[row]) {
                Ok(index) | Err(index) => index,
            })
            .collect();
        let mut units: Vec<UnitSummary> = Vec::new();
        let mut unit_starts: Vec<usize> = Vec::new();
        let mut pos = 0;
        while pos < n {
            let label = subjects[order[pos]].as_ref();
            let start = pos;
            while pos < n && subjects[order[pos]].as_ref() == label {
                pos += 1;
            }
            let unit_id = u32::try_from(units.len()).map_err(|_| {
                PanelPreparationError::argument(
                    TOO_MANY_UNITS_DETAIL,
                    format!("the panel has more than {} units", u32::MAX),
                )
            })?;
            units.push(summarize_unit(
                PopulationUnit::panel(unit_id),
                label,
                &order[start..pos],
                &row_grid[start..pos],
                observed,
                &grid,
            ));
            unit_starts.push(start);
        }
        unit_starts.push(n);
        let count = |keep: fn(&UnitSummary) -> bool| units.iter().filter(|u| keep(u)).count();
        let balanced_units = count(|u| u.balanced);
        let gapped_units = count(|u| !u.gaps.is_empty());
        let late_entry_units = count(|u| u.late_entry);
        let attrited_units = count(|u| u.attrited);
        let never_observed_units = count(|u| u.periods_complete == 0);
        let rows_complete = observed.iter().filter(|&&seen| seen).count();
        let cells_expected = (units.len() as u64).saturating_mul(grid.len() as u64);
        let digest = layout_digest(&units, &unit_starts, &order, periods, observed);
        Ok(PanelBalanceReport {
            grid: grid.into_boxed_slice(),
            grid_step,
            units: units.into_boxed_slice(),
            rows_total: n,
            rows_complete,
            balanced_units,
            gapped_units,
            late_entry_units,
            attrited_units,
            never_observed_units,
            cells_expected,
            digest: Arc::from(digest),
            canonical_rows: order.into_boxed_slice(),
            row_grid: row_grid.into_boxed_slice(),
            unit_starts: unit_starts.into_boxed_slice(),
            observed: observed.into(),
        })
    }
}

/// The rows sorted by (unit, period, row), after checking the slices agree in length, the
/// table is not empty and no (unit, period) pair repeats.
fn canonical_row_order<S: AsRef<str>>(
    subjects: &[S],
    periods: &[i64],
    observed: &[bool],
) -> Result<Vec<usize>, PanelPreparationError> {
    let n = subjects.len();
    if periods.len() != n || observed.len() != n {
        return Err(PanelPreparationError::argument(
            LENGTH_DETAIL,
            format!(
                "{n} subjects, {} periods and {} observed flags; the panel needs one of \
                 each per row",
                periods.len(),
                observed.len()
            ),
        ));
    }
    if n == 0 {
        return Err(PanelPreparationError::argument(
            EMPTY_DETAIL,
            "the panel has no rows".to_owned(),
        ));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        subjects[a]
            .as_ref()
            .cmp(subjects[b].as_ref())
            .then(periods[a].cmp(&periods[b]))
            .then(a.cmp(&b))
    });
    for pair in order.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if periods[a] == periods[b] && subjects[a].as_ref() == subjects[b].as_ref() {
            return Err(PanelPreparationError::argument(
                DUPLICATE_DETAIL,
                format!(
                    "rows {a} and {b} both carry unit '{}' at period {}; a panel has one \
                     row per unit-period",
                    short_label(subjects[a].as_ref()),
                    periods[a]
                ),
            ));
        }
    }
    Ok(order)
}

/// How a panel's rows are kept for analysis.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PreparationMode {
    /// Keep the units complete in every grid period, whole; drop every other unit whole.
    /// Single unit-periods are never dropped. The `balanced_panel` policy.
    BalancedUnits,
    /// Keep every complete unit-period. Unit and period identities are preserved and
    /// nothing is relabelled. The `available_case` policy.
    AvailableUnitPeriods,
    /// Keep every complete unit-period and expose each unit's maximal runs of consecutive
    /// grid periods, so a lag or difference taken inside a run never spans a gap or
    /// crosses a unit.
    ContiguousRuns,
}

named_variants! {
    PreparationMode {
        BalancedUnits => "balanced_units",
        AvailableUnitPeriods => "available_unit_periods",
        ContiguousRuns => "contiguous_runs",
    }
}

impl PreparationMode {
    /// The declared missing-data policy this mode executes.
    #[must_use]
    pub const fn policy(self) -> MissingPolicy {
        match self {
            Self::BalancedUnits => MissingPolicy::BalancedPanel,
            Self::AvailableUnitPeriods | Self::ContiguousRuns => MissingPolicy::AvailableCase,
        }
    }
}

/// A design that needs every unit complete in every grid period.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BalanceRequirement {
    /// Two-period panel difference-in-differences: a grid of exactly two periods.
    TwoPeriodPanel,
    /// Synthetic control: a donor pool over a common grid.
    SyntheticControl,
    /// Event study whose event-time contrasts share one balanced set of units.
    BalancedEventStudy,
    /// Augmented panel whose nuisance columns are aligned to the submitted rows.
    AugmentedPanel,
}

named_variants! {
    BalanceRequirement {
        TwoPeriodPanel => "two_period_panel",
        SyntheticControl => "synthetic_control",
        BalancedEventStudy => "balanced_event_study",
        AugmentedPanel => "augmented_panel",
    }
}

impl BalanceRequirement {
    /// What to try when the panel is not balanced.
    #[must_use]
    pub const fn next_attempt(self) -> NextAttempt {
        match self {
            Self::TwoPeriodPanel => DECLARE_BALANCED_PANEL,
            Self::SyntheticControl => RESTRICT_TO_COMMON_WINDOW,
            Self::BalancedEventStudy => DECLARE_AVAILABLE_CASE,
            Self::AugmentedPanel => BALANCE_BEFORE_NUISANCE,
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::TwoPeriodPanel => "a two-period panel difference-in-differences",
            Self::SyntheticControl => "a synthetic control donor pool",
            Self::BalancedEventStudy => "a balanced event study",
            Self::AugmentedPanel => "an augmented panel with row-aligned nuisance columns",
        }
    }
}

/// The layout of an unbalanced panel: grid, unit coverage, attrition and a digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PanelBalanceReport {
    grid: Box<[i64]>,
    grid_step: Option<i64>,
    units: Box<[UnitSummary]>,
    rows_total: usize,
    rows_complete: usize,
    balanced_units: usize,
    gapped_units: usize,
    late_entry_units: usize,
    attrited_units: usize,
    never_observed_units: usize,
    cells_expected: u64,
    digest: Arc<str>,
    /// Input rows sorted by (unit label, period).
    canonical_rows: Box<[usize]>,
    /// Grid index of each canonical row.
    row_grid: Box<[usize]>,
    /// Start of each unit's canonical rows, then the row total.
    unit_starts: Box<[usize]>,
    /// Completeness of each input row, in input order.
    observed: Box<[bool]>,
}

impl PanelBalanceReport {
    /// Sorted distinct periods of the table.
    #[must_use]
    pub fn grid(&self) -> &[i64] {
        &self.grid
    }

    /// The common difference of consecutive grid periods, when the grid has at least two
    /// periods and they are evenly spaced.
    #[must_use]
    pub const fn grid_step(&self) -> Option<i64> {
        self.grid_step
    }

    /// Units in byte order of their labels.
    #[must_use]
    pub fn units(&self) -> &[UnitSummary] {
        &self.units
    }

    /// Number of units.
    #[must_use]
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Rows in the table.
    #[must_use]
    pub const fn rows_total(&self) -> usize {
        self.rows_total
    }

    /// Rows whose analysed values are complete.
    #[must_use]
    pub const fn rows_complete(&self) -> usize {
        self.rows_complete
    }

    /// Units complete in every grid period.
    #[must_use]
    pub const fn balanced_units(&self) -> usize {
        self.balanced_units
    }

    /// Units with at least one internal gap.
    #[must_use]
    pub const fn gapped_units(&self) -> usize {
        self.gapped_units
    }

    /// Units first observed after the grid start.
    #[must_use]
    pub const fn late_entry_units(&self) -> usize {
        self.late_entry_units
    }

    /// Units last observed before the grid end.
    #[must_use]
    pub const fn attrited_units(&self) -> usize {
        self.attrited_units
    }

    /// Units without a complete row.
    #[must_use]
    pub const fn never_observed_units(&self) -> usize {
        self.never_observed_units
    }

    /// Units times grid periods (saturating).
    #[must_use]
    pub const fn cells_expected(&self) -> u64 {
        self.cells_expected
    }

    /// Unit-periods with a row. Equal to the row total, since rows are unique.
    #[must_use]
    pub const fn cells_present(&self) -> usize {
        self.rows_total
    }

    /// Whether every unit is complete in every grid period.
    #[must_use]
    pub fn is_balanced(&self) -> bool {
        self.balanced_units == self.units.len()
    }

    /// Lower-case hex BLAKE3 of the canonical (unit, period, complete) set. The order of
    /// the submitted rows does not change it.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// Units at risk, observed, entering and leaving in each grid period.
    #[must_use]
    pub fn attrition(&self) -> Vec<AttritionRow> {
        let len = self.grid.len();
        let mut entering = vec![0usize; len];
        let mut leaving = vec![0usize; len];
        let mut last_seen = vec![0usize; len];
        let mut observed = vec![0usize; len];
        for unit_index in 0..self.units.len() {
            let mut first: Option<usize> = None;
            let mut last: Option<usize> = None;
            for slot in self.unit_range(unit_index) {
                if self.observed[self.canonical_rows[slot]] {
                    let g = self.row_grid[slot];
                    observed[g] += 1;
                    first.get_or_insert(g);
                    last = Some(g);
                }
            }
            if let (Some(first), Some(last)) = (first, last) {
                entering[first] += 1;
                last_seen[last] += 1;
                if last + 1 < len {
                    leaving[last] += 1;
                }
            }
        }
        let mut at_risk = 0usize;
        let mut rows = Vec::with_capacity(len);
        for (g, &period) in self.grid.iter().enumerate() {
            at_risk += entering[g];
            rows.push(AttritionRow {
                period,
                at_risk,
                observed: observed[g],
                entering: entering[g],
                leaving: leaving[g],
            });
            at_risk -= last_seen[g];
        }
        rows
    }

    /// Compare `baseline` (row-aligned, `NaN` missing) at each unit's first observed period
    /// between units that attrite and units observed to the grid end.
    ///
    /// # Errors
    ///
    /// `baseline` differs in length from the table.
    pub fn attrition_baseline(
        &self,
        baseline: &[f64],
    ) -> Result<AttritionBaseline, PanelPreparationError> {
        if baseline.len() != self.rows_total {
            return Err(PanelPreparationError::argument(
                LENGTH_DETAIL,
                format!(
                    "the baseline has {} values for {} panel rows",
                    baseline.len(),
                    self.rows_total
                ),
            ));
        }
        let mut stayers = Moments::default();
        let mut attriters = Moments::default();
        let mut missing_baseline = 0usize;
        for unit in &*self.units {
            let Some(row) = unit.first_observed_row else {
                continue;
            };
            let value = baseline[row];
            if !value.is_finite() {
                missing_baseline += 1;
            } else if unit.attrited {
                attriters.push(value);
            } else {
                stayers.push(value);
            }
        }
        let standardized_difference = match (stayers.variance(), attriters.variance()) {
            (Some(vs), Some(va)) if vs + va > 0.0 => {
                Some((attriters.mean - stayers.mean) / ((vs + va) / 2.0).sqrt())
            }
            _ => None,
        };
        Ok(AttritionBaseline {
            stayers: stayers.count,
            attriters: attriters.count,
            missing_baseline,
            mean_stayers: stayers.mean(),
            mean_attriters: attriters.mean(),
            standardized_difference,
        })
    }

    /// Keep the rows `mode` licenses.
    ///
    /// # Errors
    ///
    /// The mode keeps no row: no unit is balanced (`BalancedUnits`) or no row is complete.
    pub fn prepare(&self, mode: PreparationMode) -> Result<PreparedPanel, PanelPreparationError> {
        let mut keep = vec![false; self.rows_total];
        let mut runs = Vec::new();
        let mut units_kept = 0usize;
        for (unit_index, summary) in self.units.iter().enumerate() {
            let kept = match mode {
                PreparationMode::BalancedUnits => summary.balanced,
                PreparationMode::AvailableUnitPeriods | PreparationMode::ContiguousRuns => {
                    summary.periods_complete > 0
                }
            };
            if !kept {
                continue;
            }
            units_kept += 1;
            match mode {
                PreparationMode::BalancedUnits | PreparationMode::ContiguousRuns => {
                    self.push_runs(unit_index, summary, &mut keep, &mut runs);
                }
                PreparationMode::AvailableUnitPeriods => {
                    for slot in self.unit_range(unit_index) {
                        let row = self.canonical_rows[slot];
                        keep[row] = self.observed[row];
                    }
                }
            }
        }
        if units_kept == 0 {
            let (message, next) = match mode {
                PreparationMode::BalancedUnits => (
                    format!(
                        "none of the {} units is complete in every one of the {} grid periods",
                        self.units.len(),
                        self.grid.len()
                    ),
                    Some(RESTRICT_TO_COMMON_WINDOW),
                ),
                PreparationMode::AvailableUnitPeriods | PreparationMode::ContiguousRuns => {
                    ("no row of the panel is complete".to_owned(), None)
                }
            };
            return Err(PanelPreparationError::new(
                POLICY_CODE,
                NOTHING_KEPT_DETAIL,
                message,
                next,
            ));
        }
        let mut dropped_rows = Vec::new();
        let mut dropped_missing_value = 0usize;
        let mut dropped_unit_incomplete = 0usize;
        for (row, &kept) in keep.iter().enumerate() {
            if kept {
                continue;
            }
            if self.observed[row] {
                dropped_unit_incomplete += 1;
            } else {
                dropped_missing_value += 1;
            }
            dropped_rows.push(row as u64);
        }
        Ok(PreparedPanel {
            mode,
            rows_total: self.rows_total,
            keep: keep.into_boxed_slice(),
            dropped_rows: dropped_rows.into_boxed_slice(),
            dropped_missing_value,
            dropped_unit_incomplete,
            units_total: self.units.len(),
            units_kept,
            runs: runs.into_boxed_slice(),
        })
    }

    /// Refuse `design` unless every unit is complete in every grid period.
    ///
    /// # Errors
    ///
    /// A grid too short for the design (a two-period design needs exactly two periods), or
    /// units that break balance; the refusal names the first few with the first period each
    /// is incomplete in, and the design's next attempt.
    pub fn require_balance(&self, design: BalanceRequirement) -> Result<(), PanelPreparationError> {
        let periods = self.grid.len();
        let shape_ok = match design {
            BalanceRequirement::TwoPeriodPanel => periods == 2,
            BalanceRequirement::SyntheticControl
            | BalanceRequirement::BalancedEventStudy
            | BalanceRequirement::AugmentedPanel => periods >= 2,
        };
        if !shape_ok {
            return Err(PanelPreparationError::new(
                ARGUMENT_CODE,
                GRID_SHAPE_DETAIL,
                format!("{} cannot run on a grid of {periods} periods", design.description()),
                matches!(design, BalanceRequirement::TwoPeriodPanel)
                    .then_some(RESTRICT_TO_COMMON_WINDOW),
            ));
        }
        let broken = self.units.len() - self.balanced_units;
        if broken == 0 {
            return Ok(());
        }
        let mut listing = String::new();
        let offenders =
            self.units.iter().enumerate().filter(|(_, unit)| !unit.balanced).take(MAX_LISTED);
        for (listed, (unit_index, unit)) in offenders.enumerate() {
            if listed > 0 {
                listing.push_str(", ");
            }
            let label = short_label(&unit.label);
            match self.first_incomplete_period(unit_index) {
                Some(period) => {
                    let _ = write!(listing, "unit '{label}' at period {period}");
                }
                None => {
                    let _ = write!(listing, "unit '{label}'");
                }
            }
        }
        if broken > MAX_LISTED {
            let _ = write!(listing, ", and {} more", broken - MAX_LISTED);
        }
        Err(PanelPreparationError::new(
            POLICY_CODE,
            UNBALANCED_DETAIL,
            format!(
                "{} needs every unit complete in every one of the {periods} grid periods; {broken} \
                 of {} units are not ({listing})",
                design.description(),
                self.units.len()
            ),
            Some(design.next_attempt()),
        ))
    }

    /// Refuse a lag or difference of `lag` grid periods that would read across a gap.
    ///
    /// A complete unit-period whose lag falls before the unit's first observed period only
    /// loses its lag; one whose lag lands on a period of the same unit without a complete
    /// row would cross a gap.
    ///
    /// # Errors
    ///
    /// `lag` is zero or not shorter than the grid, the grid is not evenly spaced, or a lag
    /// would cross a gap of some unit (the refusal names the first few).
    pub fn require_lag(&self, lag: usize) -> Result<(), PanelPreparationError> {
        let periods = self.grid.len();
        if lag == 0 || lag >= periods {
            return Err(PanelPreparationError::argument(
                LAG_RANGE_DETAIL,
                format!(
                    "a lag of {lag} periods on a grid of {periods}; it must be from 1 to {}",
                    periods.saturating_sub(1)
                ),
            ));
        }
        if self.grid_step.is_none() {
            return Err(PanelPreparationError::new(
                ARGUMENT_CODE,
                IRREGULAR_GRID_DETAIL,
                "the grid periods are not evenly spaced, so a lag of grid periods is not a \
                 calendar lag"
                    .to_owned(),
                Some(SUBMIT_REGULAR_GRID),
            ));
        }
        let mut complete: Vec<usize> = Vec::new();
        let mut crossing_units = 0usize;
        let mut listing = String::new();
        for (unit_index, unit) in self.units.iter().enumerate() {
            complete.clear();
            complete.extend(
                self.unit_range(unit_index)
                    .filter(|&slot| self.observed[self.canonical_rows[slot]])
                    .map(|slot| self.row_grid[slot]),
            );
            let Some(&first) = complete.first() else {
                continue;
            };
            let crossing = complete
                .iter()
                .copied()
                .find(|&g| g >= first + lag && complete.binary_search(&(g - lag)).is_err());
            if let Some(g) = crossing {
                crossing_units += 1;
                if crossing_units <= MAX_LISTED {
                    if crossing_units > 1 {
                        listing.push_str(", ");
                    }
                    let _ = write!(
                        listing,
                        "unit '{}' at period {} needs period {}",
                        short_label(&unit.label),
                        self.grid[g],
                        self.grid[g - lag]
                    );
                }
            }
        }
        if crossing_units == 0 {
            return Ok(());
        }
        if crossing_units > MAX_LISTED {
            let _ = write!(listing, ", and {} more", crossing_units - MAX_LISTED);
        }
        Err(PanelPreparationError::new(
            POLICY_CODE,
            LAG_GAP_DETAIL,
            format!(
                "a lag of {lag} grid periods would cross a gap in {crossing_units} units \
                 ({listing})"
            ),
            Some(LAG_INSIDE_CONTIGUOUS_RUNS),
        ))
    }

    fn unit_range(&self, unit_index: usize) -> Range<usize> {
        self.unit_starts[unit_index]..self.unit_starts[unit_index + 1]
    }

    fn first_incomplete_period(&self, unit_index: usize) -> Option<i64> {
        let mut expected = 0usize;
        for slot in self.unit_range(unit_index) {
            if self.row_grid[slot] != expected || !self.observed[self.canonical_rows[slot]] {
                return self.grid.get(expected).copied();
            }
            expected += 1;
        }
        self.grid.get(expected).copied()
    }

    /// Mark the unit's complete rows kept and push its maximal runs of consecutive grid
    /// periods.
    fn push_runs(
        &self,
        unit_index: usize,
        summary: &UnitSummary,
        keep: &mut [bool],
        runs: &mut Vec<UnitRun>,
    ) {
        let mut current: Vec<usize> = Vec::new();
        let mut start = 0usize;
        let mut prev: Option<usize> = None;
        for slot in self.unit_range(unit_index) {
            let row = self.canonical_rows[slot];
            let g = self.row_grid[slot];
            if !self.observed[row] {
                self.flush_run(summary, &mut current, start, prev, runs);
                prev = None;
                continue;
            }
            keep[row] = true;
            if prev.is_none_or(|p| p + 1 != g) {
                self.flush_run(summary, &mut current, start, prev, runs);
                start = g;
            }
            current.push(row);
            prev = Some(g);
        }
        self.flush_run(summary, &mut current, start, prev, runs);
    }

    fn flush_run(
        &self,
        summary: &UnitSummary,
        current: &mut Vec<usize>,
        start: usize,
        end: Option<usize>,
        runs: &mut Vec<UnitRun>,
    ) {
        if current.is_empty() {
            return;
        }
        let Some(end) = end else {
            current.clear();
            return;
        };
        runs.push(UnitRun {
            unit: summary.unit,
            label: Arc::clone(&summary.label),
            start_period: self.grid[start],
            end_period: self.grid[end],
            rows: Arc::from(current.as_slice()),
        });
        current.clear();
    }
}

/// A maximal run of consecutive grid periods of one unit, every one with a complete row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnitRun {
    /// The unit.
    pub unit: PopulationUnit,
    /// Unit label.
    pub label: Arc<str>,
    /// First grid period of the run.
    pub start_period: i64,
    /// Last grid period of the run, inclusive.
    pub end_period: i64,
    /// Input rows of the run, in period order.
    pub rows: Arc<[usize]>,
}

/// The rows a [`PreparationMode`] keeps, with the reasons the others were dropped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedPanel {
    mode: PreparationMode,
    rows_total: usize,
    keep: Box<[bool]>,
    dropped_rows: Box<[u64]>,
    dropped_missing_value: usize,
    dropped_unit_incomplete: usize,
    units_total: usize,
    units_kept: usize,
    runs: Box<[UnitRun]>,
}

impl PreparedPanel {
    /// The mode that prepared the rows.
    #[must_use]
    pub const fn mode(&self) -> PreparationMode {
        self.mode
    }

    /// One flag per input row: whether it is kept.
    #[must_use]
    pub fn keep(&self) -> &[bool] {
        &self.keep
    }

    /// Rows in the table.
    #[must_use]
    pub const fn rows_total(&self) -> usize {
        self.rows_total
    }

    /// Rows kept.
    #[must_use]
    pub fn rows_kept(&self) -> usize {
        self.rows_total - self.dropped_rows.len()
    }

    /// Input positions of the dropped rows, ascending.
    #[must_use]
    pub fn dropped_rows(&self) -> &[u64] {
        &self.dropped_rows
    }

    /// Dropped rows by reason: an incomplete row is a missing value; a complete row of a
    /// unit dropped whole is an incomplete unit. Reasons with no rows are left out.
    #[must_use]
    pub fn reasons(&self) -> Vec<(DropReason, usize)> {
        [
            (DropReason::MissingValue, self.dropped_missing_value),
            (DropReason::UnitIncomplete, self.dropped_unit_incomplete),
        ]
        .into_iter()
        .filter(|(_, count)| *count > 0)
        .collect()
    }

    /// Units in the panel.
    #[must_use]
    pub const fn units_total(&self) -> usize {
        self.units_total
    }

    /// Units with at least one kept row.
    #[must_use]
    pub const fn units_kept(&self) -> usize {
        self.units_kept
    }

    /// Units without a kept row.
    #[must_use]
    pub const fn units_dropped(&self) -> usize {
        self.units_total - self.units_kept
    }

    /// Runs of consecutive grid periods of one unit. One run per kept unit spanning the
    /// grid under `BalancedUnits`, each maximal run under `ContiguousRuns`, and none under
    /// `AvailableUnitPeriods`, which licenses no lag.
    #[must_use]
    pub fn runs(&self) -> &[UnitRun] {
        &self.runs
    }

    /// The row accounting a route reports for this preparation, under the policy the mode
    /// executes. `missing_by_variable` counts rows missing each analysed variable.
    #[must_use]
    pub fn row_accounting(&self, missing_by_variable: Vec<(Arc<str>, usize)>) -> RowAccounting {
        let mut accounting = RowAccounting::from_dropped(
            self.mode.policy(),
            self.rows_total,
            self.dropped_rows.to_vec(),
            missing_by_variable,
        );
        accounting.reasons = self.reasons().into();
        accounting.with_units(self.units_total, self.units_dropped())
    }
}

/// Count, mean and sum of squared deviations (Welford).
#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    count: usize,
    mean: f64,
    squares: f64,
}

impl Moments {
    fn push(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.squares += delta * (value - self.mean);
    }

    fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    fn variance(&self) -> Option<f64> {
        (self.count > 1).then(|| self.squares / (self.count - 1) as f64)
    }
}

/// The common difference of consecutive periods, if there are two or more and it is
/// constant.
fn regular_step(grid: &[i64]) -> Option<i64> {
    let step = grid.get(1)?.checked_sub(grid[0])?;
    grid.windows(2).all(|pair| pair[1].checked_sub(pair[0]) == Some(step)).then_some(step)
}

/// One unit's coverage from its rows in period order and their grid indexes.
fn summarize_unit(
    unit: PopulationUnit,
    label: &str,
    rows: &[usize],
    row_grid: &[usize],
    observed: &[bool],
    grid: &[i64],
) -> UnitSummary {
    let mut first: Option<(usize, usize)> = None;
    let mut prev: Option<usize> = None;
    let mut complete = 0usize;
    let mut gaps = Vec::new();
    for (&row, &g) in rows.iter().zip(row_grid) {
        if !observed[row] {
            continue;
        }
        complete += 1;
        first.get_or_insert((g, row));
        if let Some(p) = prev.filter(|&p| g > p + 1) {
            gaps.push(PeriodRun { first: grid[p + 1], last: grid[g - 1], periods: g - p - 1 });
        }
        prev = Some(g);
    }
    let last = prev;
    UnitSummary {
        unit,
        label: Arc::from(label),
        first_observed: first.map(|(g, _)| grid[g]),
        last_observed: last.map(|g| grid[g]),
        periods_present: rows.len(),
        periods_complete: complete,
        gaps: gaps.into_boxed_slice(),
        late_entry: first.is_some_and(|(g, _)| g > 0),
        attrited: last.is_some_and(|g| g + 1 < grid.len()),
        balanced: complete == grid.len(),
        first_observed_row: first.map(|(_, row)| row),
    }
}

fn short_label(label: &str) -> String {
    if label.chars().count() <= MAX_LABEL_CHARS {
        label.to_owned()
    } else {
        let mut short: String = label.chars().take(MAX_LABEL_CHARS).collect();
        short.push_str("...");
        short
    }
}

fn layout_digest(
    units: &[UnitSummary],
    unit_starts: &[usize],
    order: &[usize],
    periods: &[i64],
    observed: &[bool],
) -> String {
    let mut hasher = blake3::Hasher::new_derive_key(LAYOUT_DIGEST_CONTEXT);
    hasher.update(&(units.len() as u64).to_le_bytes());
    for (unit_index, unit) in units.iter().enumerate() {
        let rows = &order[unit_starts[unit_index]..unit_starts[unit_index + 1]];
        hasher.update(&(unit.label.len() as u64).to_le_bytes());
        hasher.update(unit.label.as_bytes());
        hasher.update(&(rows.len() as u64).to_le_bytes());
        for &row in rows {
            hasher.update(&periods[row].to_le_bytes());
            hasher.update(&[u8::from(observed[row])]);
        }
    }
    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use antecedent_core::refusal::Refused;

    use super::*;

    /// Units a (balanced), b (absent at 3), c (enters at 2), d (leaves after 2) and e
    /// (incomplete at 3), over periods 1 to 4, in shuffled row order.
    const ROWS: [(&str, i64, bool); 16] = [
        ("c", 2, true),
        ("a", 1, true),
        ("b", 4, true),
        ("a", 2, true),
        ("d", 1, true),
        ("e", 3, false),
        ("a", 3, true),
        ("b", 1, true),
        ("c", 3, true),
        ("e", 1, true),
        ("a", 4, true),
        ("d", 2, true),
        ("b", 2, true),
        ("e", 2, true),
        ("c", 4, true),
        ("e", 4, true),
    ];

    fn assess(rows: &[(&str, i64, bool)]) -> Result<PanelBalanceReport, PanelPreparationError> {
        let subjects: Vec<&str> = rows.iter().map(|r| r.0).collect();
        let periods: Vec<i64> = rows.iter().map(|r| r.1).collect();
        let observed: Vec<bool> = rows.iter().map(|r| r.2).collect();
        PanelLayout::assess(&subjects, &periods, &observed)
    }

    fn unit<'a>(report: &'a PanelBalanceReport, label: &str) -> &'a UnitSummary {
        report.units().iter().find(|u| u.label.as_ref() == label).unwrap()
    }

    #[test]
    fn report_describes_gaps_entry_and_attrition() {
        let report = assess(&ROWS).unwrap();
        assert_eq!(report.grid(), &[1, 2, 3, 4]);
        assert_eq!(report.grid_step(), Some(1));
        assert_eq!(report.unit_count(), 5);
        assert_eq!(report.rows_total(), 16);
        assert_eq!(report.rows_complete(), 15);
        assert_eq!(report.cells_expected(), 20);
        assert_eq!(report.cells_present(), 16);
        assert_eq!(report.balanced_units(), 1);
        assert_eq!(report.gapped_units(), 2);
        assert_eq!(report.late_entry_units(), 1);
        assert_eq!(report.attrited_units(), 1);
        assert_eq!(report.never_observed_units(), 0);
        assert!(!report.is_balanced());

        let a = unit(&report, "a");
        assert_eq!(a.unit, PopulationUnit::panel(0));
        assert!(a.balanced && a.gaps.is_empty() && !a.late_entry && !a.attrited);
        assert_eq!(a.first_observed_row, Some(1));

        let b = unit(&report, "b");
        assert_eq!((b.first_observed, b.last_observed), (Some(1), Some(4)));
        assert_eq!((b.periods_present, b.periods_complete), (3, 3));
        assert_eq!(&*b.gaps, &[PeriodRun { first: 3, last: 3, periods: 1 }]);
        assert!(!b.balanced);

        let c = unit(&report, "c");
        assert_eq!(c.first_observed, Some(2));
        assert!(c.late_entry && !c.attrited && c.gaps.is_empty());

        let d = unit(&report, "d");
        assert_eq!(d.last_observed, Some(2));
        assert!(d.attrited && !d.late_entry);

        let e = unit(&report, "e");
        assert_eq!((e.periods_present, e.periods_complete), (4, 3));
        assert_eq!(&*e.gaps, &[PeriodRun { first: 3, last: 3, periods: 1 }]);
    }

    #[test]
    fn malformed_panels_are_refused() {
        let duplicate = assess(&[("a", 1, true), ("b", 1, true), ("a", 1, false)]).unwrap_err();
        assert_eq!(duplicate.detail(), DUPLICATE_DETAIL);
        assert_eq!(duplicate.refusal_code(), "invalid_argument");
        assert!(duplicate.message().contains("unit 'a' at period 1"));

        let empty = assess(&[]).unwrap_err();
        assert_eq!(empty.detail(), EMPTY_DETAIL);

        let mismatch = PanelLayout::assess(&["a", "b"], &[1], &[true, true]).unwrap_err();
        assert_eq!(mismatch.detail(), LENGTH_DETAIL);
        let data: DataError = mismatch.into();
        assert!(matches!(data, DataError::InvalidArgument { .. }));
    }

    #[test]
    fn balanced_units_keeps_complete_units_whole() {
        let report = assess(&ROWS).unwrap();
        let prepared = report.prepare(PreparationMode::BalancedUnits).unwrap();
        let kept: Vec<usize> = (0..ROWS.len()).filter(|&row| prepared.keep()[row]).collect();
        assert_eq!(kept, vec![1, 3, 6, 10]);
        assert_eq!(prepared.rows_kept(), 4);
        assert_eq!(
            prepared.reasons(),
            vec![(DropReason::MissingValue, 1), (DropReason::UnitIncomplete, 11)]
        );
        assert_eq!((prepared.units_kept(), prepared.units_dropped()), (1, 4));
        assert_eq!(prepared.runs().len(), 1);
        assert_eq!((prepared.runs()[0].start_period, prepared.runs()[0].end_period), (1, 4));

        let accounting = prepared.row_accounting(Vec::new());
        assert_eq!(accounting.policy, MissingPolicy::BalancedPanel);
        assert_eq!((accounting.rows_total, accounting.rows_kept), (16, 4));
        assert_eq!(accounting.units, Some(crate::missing::UnitAccounting { total: 5, dropped: 4 }));
        assert_eq!(&*accounting.reasons, prepared.reasons().as_slice());
    }

    #[test]
    fn available_unit_periods_preserve_identities() {
        let report = assess(&ROWS).unwrap();
        let prepared = report.prepare(PreparationMode::AvailableUnitPeriods).unwrap();
        for (row, &(_, _, observed)) in ROWS.iter().enumerate() {
            assert_eq!(prepared.keep()[row], observed, "row {row}");
        }
        assert_eq!(prepared.dropped_rows(), &[5]);
        assert_eq!(prepared.reasons(), vec![(DropReason::MissingValue, 1)]);
        assert_eq!(prepared.units_kept(), 5);
        assert!(prepared.runs().is_empty());
        assert_eq!(prepared.row_accounting(Vec::new()).policy, MissingPolicy::AvailableCase);
    }

    #[test]
    fn contiguous_runs_never_span_a_gap() {
        let report = assess(&ROWS).unwrap();
        let prepared = report.prepare(PreparationMode::ContiguousRuns).unwrap();
        let grid = report.grid();
        let mut covered = 0usize;
        for run in prepared.runs() {
            let first = grid.iter().position(|&p| p == run.start_period).unwrap();
            for (offset, &row) in run.rows.iter().enumerate() {
                let (label, period, observed) = ROWS[row];
                assert_eq!(label, run.label.as_ref());
                assert!(observed);
                assert_eq!(period, grid[first + offset]);
            }
            assert_eq!(grid[first + run.rows.len() - 1], run.end_period);
            covered += run.rows.len();
        }
        assert_eq!(covered, 15);
        let spans: Vec<(&str, i64, i64)> = prepared
            .runs()
            .iter()
            .map(|run| (run.label.as_ref(), run.start_period, run.end_period))
            .collect();
        assert_eq!(
            spans,
            vec![
                ("a", 1, 4),
                ("b", 1, 2),
                ("b", 4, 4),
                ("c", 2, 4),
                ("d", 1, 2),
                ("e", 1, 2),
                ("e", 4, 4),
            ]
        );
    }

    #[test]
    fn designs_that_need_balance_refuse_with_the_next_attempt() {
        let report = assess(&ROWS).unwrap();
        let refused = report.require_balance(BalanceRequirement::SyntheticControl).unwrap_err();
        assert_eq!(refused.code(), "missing_policy_not_applicable");
        assert_eq!(refused.detail(), UNBALANCED_DETAIL);
        assert_eq!(refused.next_attempt(), Some(RESTRICT_TO_COMMON_WINDOW));
        assert!(refused.message().contains("unit 'b' at period 3"));
        assert!(refused.message().contains("unit 'c' at period 1"));
        assert!(refused.message().contains("4 of 5 units"));

        let study = report.require_balance(BalanceRequirement::BalancedEventStudy).unwrap_err();
        assert_eq!(study.next_attempt(), Some(DECLARE_AVAILABLE_CASE));

        let two = report.require_balance(BalanceRequirement::TwoPeriodPanel).unwrap_err();
        assert_eq!(two.detail(), GRID_SHAPE_DETAIL);

        let balanced =
            assess(&[("x", 1, true), ("x", 2, true), ("y", 2, true), ("y", 1, true)]).unwrap();
        assert!(balanced.is_balanced());
        for design in BalanceRequirement::ALL {
            assert_eq!(balanced.require_balance(design), Ok(()));
        }
        assert_eq!(balanced.require_lag(1), Ok(()));
    }

    #[test]
    fn a_lag_across_a_gap_is_refused() {
        let report = assess(&ROWS).unwrap();
        let refused = report.require_lag(1).unwrap_err();
        assert_eq!(refused.detail(), LAG_GAP_DETAIL);
        assert_eq!(refused.next_attempt(), Some(LAG_INSIDE_CONTIGUOUS_RUNS));
        assert!(refused.message().contains("unit 'b' at period 4 needs period 3"));
        assert_eq!(report.require_lag(0).unwrap_err().detail(), LAG_RANGE_DETAIL);
        assert_eq!(report.require_lag(4).unwrap_err().detail(), LAG_RANGE_DETAIL);

        let irregular = assess(&[("x", 1, true), ("x", 2, true), ("x", 4, true)]).unwrap();
        assert_eq!(irregular.grid_step(), None);
        assert_eq!(irregular.require_lag(1).unwrap_err().next_attempt(), Some(SUBMIT_REGULAR_GRID));
    }

    #[test]
    fn digest_ignores_row_order_and_tracks_cells() {
        let report = assess(&ROWS).unwrap();
        let mut reversed = ROWS;
        reversed.reverse();
        assert_eq!(assess(&reversed).unwrap().digest(), report.digest());
        let mut changed = ROWS;
        changed[0].2 = false;
        assert_ne!(assess(&changed).unwrap().digest(), report.digest());
        assert_eq!(report.digest().len(), 64);
    }

    #[test]
    fn attrition_table_counts_units_per_period() {
        let report = assess(&ROWS).unwrap();
        let table = report.attrition();
        let counts: Vec<(i64, usize, usize, usize, usize)> = table
            .iter()
            .map(|r| (r.period, r.at_risk, r.observed, r.entering, r.leaving))
            .collect();
        assert_eq!(
            counts,
            vec![(1, 4, 4, 4, 0), (2, 5, 5, 1, 1), (3, 4, 2, 0, 0), (4, 4, 4, 0, 0)]
        );
        assert!((table[1].hazard().unwrap() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn attrition_baseline_compares_first_observed_levels() {
        let rows = [
            ("p", 1, true),
            ("p", 2, true),
            ("q", 1, true),
            ("q", 2, true),
            ("r", 1, true),
            ("s", 1, true),
        ];
        let report = assess(&rows).unwrap();
        let baseline = [1.0, 10.0, 3.0, 10.0, 5.0, 7.0];
        let balance = report.attrition_baseline(&baseline).unwrap();
        assert_eq!((balance.stayers, balance.attriters, balance.missing_baseline), (2, 2, 0));
        assert!((balance.mean_stayers.unwrap() - 2.0).abs() < 1e-12);
        assert!((balance.mean_attriters.unwrap() - 6.0).abs() < 1e-12);
        let expected = 4.0 / 2.0_f64.sqrt();
        assert!((balance.standardized_difference.unwrap() - expected).abs() < 1e-12);

        let sparse = report.attrition_baseline(&[f64::NAN, 1.0, 3.0, 1.0, 5.0, 7.0]).unwrap();
        assert_eq!((sparse.stayers, sparse.missing_baseline), (1, 1));
        assert_eq!(sparse.standardized_difference, None);
        assert_eq!(report.attrition_baseline(&[1.0]).unwrap_err().detail(), LENGTH_DETAIL);
    }
}
