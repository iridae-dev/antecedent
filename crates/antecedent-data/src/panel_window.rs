// SPDX-License-Identifier: MIT OR Apache-2.0

//! Windows over one panel unit or one environment.
//!
//! A window is one unit, one declared split, and one calendar span. A lag may
//! borrow earlier rows of that same unit when they stay inside the span. It
//! does not borrow across a unit, a split, or a calendar boundary. Local row
//! indexes are offsets inside that unit.
//!
//! The eager reference is every local row of the whole unit that lies in the
//! split and the span, in unit order. A chunk boundary is a cut of that slice:
//! the chunks concatenate to it, and a lag that crosses the cut reads the same
//! row the slice reads.

// The parent adds this module and the re-exports. Until a call site uses the
// views, only the tests do.
#![allow(dead_code, reason = "the parent wires panel_window into the crate root")]

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use crate::error::DataError;

/// Next attempt when a lag would read another unit.
///
/// The legal attempt is a longer span of this unit. The neighbor is not a source.
pub const LONGER_SPAN_OF_SAME_UNIT: &str =
    "a longer span of the same unit, not a borrow from the neighbor";

/// One panel unit or one environment.
///
/// The id is the unit's identity, not a row index.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PopulationUnit {
    /// A panel unit.
    Panel {
        /// [`PanelUnit::unit_id`](crate::panel::PanelUnit::unit_id).
        unit_id: u32,
    },
    /// One environment in an environment list.
    Environment {
        /// Index of the environment in that list.
        env_index: u32,
    },
}

impl PopulationUnit {
    /// Panel unit `unit_id`.
    #[must_use]
    pub const fn panel(unit_id: u32) -> Self {
        Self::Panel { unit_id }
    }

    /// Environment `env_index`.
    #[must_use]
    pub const fn environment(env_index: u32) -> Self {
        Self::Environment { env_index }
    }
}

impl fmt::Display for PopulationUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Panel { unit_id } => write!(formatter, "panel unit {unit_id}"),
            Self::Environment { env_index } => write!(formatter, "environment {env_index}"),
        }
    }
}

/// Half-open calendar interval `[start, end)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CalendarSpan {
    start: i64,
    end: i64,
}

impl CalendarSpan {
    /// Interval `[start, end)`.
    ///
    /// # Errors
    ///
    /// `end <= start`. An empty span covers no row.
    #[must_use = "an empty span is not a window"]
    pub fn try_new(start: i64, end: i64) -> Result<Self, DataError> {
        if end <= start {
            return Err(invalid("calendar span must be non-empty"));
        }
        Ok(Self { start, end })
    }

    /// Inclusive start time.
    #[must_use]
    pub const fn start(self) -> i64 {
        self.start
    }

    /// Exclusive end time.
    #[must_use]
    pub const fn end(self) -> i64 {
        self.end
    }

    /// Whether `time` lies in the interval.
    #[must_use]
    pub const fn contains(self, time: i64) -> bool {
        time >= self.start && time < self.end
    }
}

/// A named half-open row range inside one unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeclaredSplit {
    name: Arc<str>,
    rows: Range<usize>,
}

impl DeclaredSplit {
    /// Split `name` over local rows `rows`.
    ///
    /// # Errors
    ///
    /// An empty name or an empty range.
    #[must_use = "an invalid split is not a window boundary"]
    pub fn try_new(name: impl Into<Arc<str>>, rows: Range<usize>) -> Result<Self, DataError> {
        let name = name.into();
        if name.as_ref().is_empty() {
            return Err(invalid("split name must be non-empty"));
        }
        if rows.is_empty() {
            return Err(invalid(format!("split '{name}' must be non-empty")));
        }
        Ok(Self { name, rows })
    }

    /// Split name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    /// Local rows the split covers.
    #[must_use]
    pub const fn rows(&self) -> Range<usize> {
        self.rows.start..self.rows.end
    }
}

/// One unit's calendar, segment breaks, and declared splits.
///
/// Built once and reused. Each [`UnitPartition::window`] is a view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnitPartition {
    unit: PopulationUnit,
    calendar: Arc<[i64]>,
    /// Local rows at which a new calendar segment starts. Row 0 is not listed.
    boundaries: Arc<[usize]>,
    splits: Arc<[DeclaredSplit]>,
}

impl UnitPartition {
    /// Partition `unit` by `calendar`, `step`, and `splits`.
    ///
    /// `calendar[i]` is the time of local row `i`. Times must strictly increase.
    /// `step` is the distance between adjacent rows of one segment. A larger or
    /// smaller positive gap starts a segment a lag must not cross.
    ///
    /// # Errors
    ///
    /// An empty calendar, a non-positive step, a time that does not increase,
    /// a split past the unit, a duplicate split name, two splits that share a row,
    /// or no splits.
    #[must_use = "an invalid partition is not a unit view"]
    pub fn try_new(
        unit: PopulationUnit,
        calendar: impl Into<Arc<[i64]>>,
        step: i64,
        splits: impl IntoIterator<Item = DeclaredSplit>,
    ) -> Result<Self, DataError> {
        let calendar = calendar.into();
        if calendar.is_empty() {
            return Err(invalid("a unit needs at least one row"));
        }
        if step <= 0 {
            return Err(invalid("calendar step must be positive"));
        }
        let splits: Arc<[DeclaredSplit]> = splits.into_iter().collect::<Vec<_>>().into();
        if splits.is_empty() {
            return Err(invalid("a unit needs at least one declared split"));
        }
        validate_splits(calendar.len(), &splits)?;
        let boundaries = segment_boundaries(&calendar, step)?;
        Ok(Self { unit, calendar, boundaries, splits })
    }

    /// Unit this partition describes.
    #[must_use]
    pub const fn unit(&self) -> PopulationUnit {
        self.unit
    }

    /// Local row count.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.calendar.len()
    }

    /// View the rows of `split_name` whose times lie in `span`.
    ///
    /// # Errors
    ///
    /// Unknown split, or the span meets none of that split's rows.
    #[must_use = "a refused window is not a view of the unit"]
    pub fn window(&self, split_name: &str, span: CalendarSpan) -> Result<PanelWindow, DataError> {
        let split = find_split(&self.splits, split_name)?;
        let rows = intersecting_rows(&self.calendar, split.rows(), span)?;
        Ok(PanelWindow {
            unit: self.unit,
            split_name: Arc::clone(&split.name),
            split_rows: split.rows(),
            span,
            rows,
            calendar: Arc::clone(&self.calendar),
            boundaries: Arc::clone(&self.boundaries),
        })
    }
}

/// A view of one unit, one split, and one calendar span.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PanelWindow {
    unit: PopulationUnit,
    split_name: Arc<str>,
    split_rows: Range<usize>,
    span: CalendarSpan,
    rows: Range<usize>,
    calendar: Arc<[i64]>,
    boundaries: Arc<[usize]>,
}

impl PanelWindow {
    /// Unit the view reads.
    #[must_use]
    pub const fn unit(&self) -> PopulationUnit {
        self.unit
    }

    /// Declared split name.
    #[must_use]
    pub fn split_name(&self) -> &str {
        self.split_name.as_ref()
    }

    /// Calendar span.
    #[must_use]
    pub const fn span(&self) -> CalendarSpan {
        self.span
    }

    /// Local rows of this window, in unit order.
    #[must_use]
    pub const fn local_rows(&self) -> Range<usize> {
        self.rows.start..self.rows.end
    }

    /// Eager slice of the whole unit for this split and span.
    ///
    /// The same rows [`Self::local_rows`] names. A chunk boundary concatenates
    /// back to this slice.
    #[must_use = "the eager slice is the reference a chunk boundary must reproduce"]
    pub fn eager_rows(&self) -> Vec<usize> {
        self.local_rows().collect()
    }

    /// Borrow the row `lag` steps before `target`.
    ///
    /// `lag == 0` is `target`. A positive lag reads an earlier local row of
    /// this unit when that row is inside the split and the span and no calendar
    /// segment starts between the two rows. A chunk cut is not a boundary.
    ///
    /// # Errors
    ///
    /// `target` is outside the window, or the lagged row would leave the unit,
    /// the split, or the calendar span. Leaving the unit names
    /// [`LONGER_SPAN_OF_SAME_UNIT`].
    #[must_use = "a refused lag names the next attempt and is not a neighbor row"]
    pub fn borrow_lag(&self, target: usize, lag: u32) -> Result<usize, BorrowError> {
        if !self.rows.contains(&target) {
            return Err(BorrowError::TargetOutsideWindow { target, rows: self.local_rows() });
        }
        let Some(source) = target.checked_sub(lag as usize) else {
            return Err(self.refuse_unit(target, lag));
        };
        let Some(&source_time) = self.calendar.get(source) else {
            return Err(self.refuse_unit(target, lag));
        };
        if !self.split_rows.contains(&source) {
            return Err(BorrowError::Refused(LagRefusal {
                target,
                lag,
                boundary: WindowBoundary::Split,
                next_attempt: WindowNextAttempt::StayInsideSplit {
                    name: Arc::clone(&self.split_name),
                },
            }));
        }
        if !self.span.contains(source_time) || !self.rows.contains(&source) {
            return Err(self.refuse_span(target, lag));
        }
        if crosses_calendar(&self.boundaries, source, target) {
            return Err(BorrowError::Refused(LagRefusal {
                target,
                lag,
                boundary: WindowBoundary::Calendar,
                next_attempt: WindowNextAttempt::StayInsideCalendarSegment,
            }));
        }
        Ok(source)
    }

    /// Cut the window at a local row strictly inside it.
    ///
    /// The two chunks concatenate to [`Self::eager_rows`]. The cut does not
    /// change which row a lag reads.
    ///
    /// # Errors
    ///
    /// `boundary` is not strictly inside [`Self::local_rows`].
    #[must_use = "dropping the chunks drops one side of the eager slice"]
    pub fn chunk_at(&self, boundary: usize) -> Result<(WindowChunk, WindowChunk), DataError> {
        if boundary <= self.rows.start || boundary >= self.rows.end {
            return Err(invalid(format!(
                "chunk boundary {boundary} is not strictly inside {}..{}",
                self.rows.start, self.rows.end
            )));
        }
        Ok((
            WindowChunk { rows: self.rows.start..boundary, window: self.clone() },
            WindowChunk { rows: boundary..self.rows.end, window: self.clone() },
        ))
    }

    fn refuse_unit(&self, target: usize, lag: u32) -> BorrowError {
        BorrowError::Refused(LagRefusal {
            target,
            lag,
            boundary: WindowBoundary::Unit,
            next_attempt: WindowNextAttempt::LongerSpanOfSameUnit { unit: self.unit },
        })
    }

    fn refuse_span(&self, target: usize, lag: u32) -> BorrowError {
        BorrowError::Refused(LagRefusal {
            target,
            lag,
            boundary: WindowBoundary::Calendar,
            next_attempt: WindowNextAttempt::LongerSpanOfSameUnit { unit: self.unit },
        })
    }
}

/// One side of a chunk cut. The rows are a subrange of the eager slice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowChunk {
    rows: Range<usize>,
    /// The whole window the chunk was cut from. Shared storage, so a clone is cheap.
    window: PanelWindow,
}

impl WindowChunk {
    /// Local rows on this side of the cut.
    #[must_use]
    pub const fn local_rows(&self) -> Range<usize> {
        self.rows.start..self.rows.end
    }

    /// Borrow the row `lag` steps before `target`, where `target` is a row of this chunk.
    ///
    /// The cut is not a boundary: a lag from the first rows of a later chunk reads the
    /// row in the earlier chunk that [`PanelWindow::borrow_lag`] reads over the whole
    /// window, and is refused exactly when that call is.
    ///
    /// # Errors
    ///
    /// `target` is not a row of this chunk, or the whole-window borrow is refused.
    #[must_use = "a refused lag names the next attempt and is not a neighbor row"]
    pub fn borrow_lag(&self, target: usize, lag: u32) -> Result<usize, BorrowError> {
        if !self.rows.contains(&target) {
            return Err(BorrowError::TargetOutsideWindow { target, rows: self.local_rows() });
        }
        self.window.borrow_lag(target, lag)
    }
}

/// Which boundary a lag tried to cross.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WindowBoundary {
    /// The lagged index is before this unit's first row.
    Unit,
    /// The lagged row belongs to another declared split.
    Split,
    /// The lagged row leaves the span or crosses a calendar segment.
    Calendar,
}

/// The attempt that follows a refused lag.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WindowNextAttempt {
    /// Widen the span of this unit. Do not read the neighbor.
    LongerSpanOfSameUnit {
        /// Unit whose span can grow.
        unit: PopulationUnit,
    },
    /// Keep the lag inside the split that contains the target.
    StayInsideSplit {
        /// Split that contains the target.
        name: Arc<str>,
    },
    /// Do not cross the calendar break. A longer span still cannot.
    StayInsideCalendarSegment,
}

impl WindowNextAttempt {
    /// Stable name of the attempt.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::LongerSpanOfSameUnit { .. } => LONGER_SPAN_OF_SAME_UNIT,
            Self::StayInsideSplit { .. } => "keep the lag inside the declared split",
            Self::StayInsideCalendarSegment => "stay inside the calendar segment",
        }
    }
}

/// A lag that would leave the window's unit, split, or calendar span.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagRefusal {
    target: usize,
    lag: u32,
    boundary: WindowBoundary,
    next_attempt: WindowNextAttempt,
}

impl LagRefusal {
    /// Row the lag was read from.
    #[must_use]
    pub const fn target(&self) -> usize {
        self.target
    }

    /// Requested lag, in earlier local rows.
    #[must_use]
    pub const fn lag(&self) -> u32 {
        self.lag
    }

    /// Boundary the lag would cross.
    #[must_use]
    pub const fn boundary(&self) -> WindowBoundary {
        self.boundary
    }

    /// What to try instead.
    #[must_use]
    pub const fn next_attempt(&self) -> &WindowNextAttempt {
        &self.next_attempt
    }
}

impl fmt::Display for LagRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.next_attempt {
            WindowNextAttempt::LongerSpanOfSameUnit { unit } => match self.boundary {
                WindowBoundary::Unit => write!(
                    formatter,
                    "lag would cross {unit}; next attempt: {LONGER_SPAN_OF_SAME_UNIT}"
                ),
                WindowBoundary::Calendar => write!(
                    formatter,
                    "lag would leave the calendar span of {unit}; next attempt: \
                     {LONGER_SPAN_OF_SAME_UNIT}"
                ),
                WindowBoundary::Split => write!(
                    formatter,
                    "lag would leave {unit}; next attempt: {LONGER_SPAN_OF_SAME_UNIT}"
                ),
            },
            WindowNextAttempt::StayInsideSplit { name } => {
                let name = name.as_ref();
                write!(
                    formatter,
                    "lag would cross split '{name}'; next attempt: keep the lag inside that split"
                )
            }
            WindowNextAttempt::StayInsideCalendarSegment => write!(
                formatter,
                "lag would cross a calendar boundary; next attempt: stay inside the calendar \
                 segment"
            ),
        }
    }
}

impl std::error::Error for LagRefusal {}

/// Why [`PanelWindow::borrow_lag`] returned no row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BorrowError {
    /// `target` is not a row of the window.
    TargetOutsideWindow {
        /// Requested row.
        target: usize,
        /// Rows the window holds.
        rows: Range<usize>,
    },
    /// The lag would cross a unit, a split, or a calendar boundary.
    Refused(LagRefusal),
}

impl BorrowError {
    /// Boundary refusal, when the target was inside the window.
    #[must_use = "none means the target was outside the window, not a boundary refusal"]
    pub const fn refusal(&self) -> Option<&LagRefusal> {
        match self {
            Self::Refused(refusal) => Some(refusal),
            Self::TargetOutsideWindow { .. } => None,
        }
    }
}

impl fmt::Display for BorrowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetOutsideWindow { target, rows } => {
                write!(formatter, "target row {target} is outside {}..{}", rows.start, rows.end)
            }
            Self::Refused(refusal) => write!(formatter, "{refusal}"),
        }
    }
}

impl std::error::Error for BorrowError {}

fn invalid(message: impl Into<String>) -> DataError {
    DataError::InvalidArgument { message: message.into() }
}

fn validate_splits(row_count: usize, splits: &[DeclaredSplit]) -> Result<(), DataError> {
    for (index, split) in splits.iter().enumerate() {
        if split.rows.end > row_count {
            return Err(invalid(format!("split '{}' is outside the unit", split.name)));
        }
        for earlier in splits.iter().take(index) {
            if earlier.name.as_ref() == split.name.as_ref() {
                return Err(invalid(format!("duplicate split '{}'", split.name)));
            }
            if earlier.rows.start < split.rows.end && split.rows.start < earlier.rows.end {
                return Err(invalid(format!(
                    "splits '{}' and '{}' share rows; declared splits must be disjoint",
                    earlier.name, split.name
                )));
            }
        }
    }
    Ok(())
}

fn segment_boundaries(calendar: &[i64], step: i64) -> Result<Arc<[usize]>, DataError> {
    let mut boundaries = Vec::new();
    for index in 1..calendar.len() {
        let Some(&later) = calendar.get(index) else {
            return Err(invalid("calendar index is outside the unit"));
        };
        let Some(&earlier) = calendar.get(index - 1) else {
            return Err(invalid("calendar index is outside the unit"));
        };
        let gap = match later.checked_sub(earlier) {
            Some(gap) if gap > 0 => gap,
            Some(_) => return Err(invalid("calendar times must increase")),
            None => return Err(invalid("calendar times overflow")),
        };
        if gap != step {
            boundaries.push(index);
        }
    }
    Ok(Arc::from(boundaries))
}

fn find_split<'a>(splits: &'a [DeclaredSplit], name: &str) -> Result<&'a DeclaredSplit, DataError> {
    splits
        .iter()
        .find(|split| split.name.as_ref() == name)
        .ok_or_else(|| invalid(format!("unknown split '{name}'")))
}

fn intersecting_rows(
    calendar: &[i64],
    split: Range<usize>,
    span: CalendarSpan,
) -> Result<Range<usize>, DataError> {
    let mut found_start = None;
    let mut found_end = split.start;
    for row in split.start..split.end {
        let Some(&time) = calendar.get(row) else {
            return Err(invalid("split row is outside the unit"));
        };
        if span.contains(time) {
            if found_start.is_none() {
                found_start = Some(row);
            }
            found_end = row.checked_add(1).ok_or_else(|| invalid("row index overflow"))?;
        } else if found_start.is_some() {
            break;
        }
    }
    match found_start {
        Some(start) => Ok(start..found_end),
        None => Err(invalid("calendar span does not meet the split")),
    }
}

fn crosses_calendar(boundaries: &[usize], source: usize, target: usize) -> bool {
    for &boundary in boundaries {
        if source < boundary && boundary <= target {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eager_of_whole_unit(calendar: &[i64], split: Range<usize>, span: Range<i64>) -> Vec<usize> {
        (0..calendar.len())
            .filter(|&row| split.contains(&row) && span.contains(&calendar[row]))
            .collect()
    }

    fn fit_window() -> (Vec<usize>, PanelWindow) {
        let calendar: Vec<i64> = (0..12u32).map(|row| i64::from(row) + 100).collect();
        let eager = eager_of_whole_unit(&calendar, 0..12, 104..110);
        let split = DeclaredSplit::try_new("fit", 0..12).expect("split covers the unit");
        let partition =
            UnitPartition::try_new(PopulationUnit::environment(3), calendar, 1, [split])
                .expect("partition");
        let span = CalendarSpan::try_new(104, 110).expect("span");
        let window = partition.window("fit", span).expect("window");
        assert_eq!(partition.row_count(), 12);
        assert_eq!(partition.unit(), PopulationUnit::environment(3));
        (eager, window)
    }

    #[test]
    fn lag_across_a_unit_names_a_longer_span_of_the_same_unit() {
        let neighbor = full_unit(PopulationUnit::panel(0), 6);
        let current = full_unit(PopulationUnit::panel(1), 6);
        assert_eq!(neighbor.borrow_lag(5, 1).expect("neighbor lag stays in its unit"), 4);
        assert_ne!(current.unit(), neighbor.unit());

        let error = current.borrow_lag(0, 1).expect_err("lag leaves the unit");
        let refusal = error.refusal().expect("leaving the unit is a boundary refusal");
        assert_eq!(refusal.target(), 0);
        assert_eq!(refusal.lag(), 1);
        assert_eq!(refusal.boundary(), WindowBoundary::Unit);
        assert_eq!(current.split_name(), "all");
        assert_eq!(current.span().start(), 0);
        assert_eq!(current.span().end(), 6);
        assert_eq!(
            refusal.next_attempt(),
            &WindowNextAttempt::LongerSpanOfSameUnit { unit: PopulationUnit::panel(1) }
        );
        assert_eq!(refusal.next_attempt().name(), LONGER_SPAN_OF_SAME_UNIT);
        let text = refusal.to_string();
        assert!(text.contains(LONGER_SPAN_OF_SAME_UNIT));
        assert!(text.contains("panel unit 1"));
        assert!(!text.contains("panel unit 0"));
        assert_eq!(current.borrow_lag(2, 1).expect("lag stays in this unit"), 1);
    }

    #[test]
    fn within_unit_lag_matches_the_eager_slice() {
        let (eager, window) = fit_window();
        assert_eq!(eager, (4..10).collect::<Vec<_>>());
        assert_eq!(window.eager_rows(), eager);
        assert_eq!(window.local_rows(), 4..10);
        assert_eq!(window.split_name(), "fit");
        assert_eq!(window.span().start(), 104);
        assert_eq!(window.span().end(), 110);
        assert_eq!(window.borrow_lag(6, 2).expect("lag stays inside the span"), 4);
        assert!(eager.contains(&4));

        let mut saw_borrow = false;
        for &target in &eager {
            for lag in 0..4u32 {
                let Some(source) = target.checked_sub(lag as usize) else {
                    continue;
                };
                let borrowed = window.borrow_lag(target, lag);
                if eager.contains(&source) {
                    assert_eq!(borrowed.expect("source is in the eager slice"), source);
                    saw_borrow = true;
                } else {
                    let refusal = borrowed.expect_err("source is outside the eager slice");
                    assert!(refusal.refusal().is_some());
                }
            }
        }
        assert!(saw_borrow);

        let outside = window.borrow_lag(4, 1).expect_err("row 3 is outside the span");
        let refusal = outside.refusal().expect("span miss is a refusal");
        assert_eq!(refusal.boundary(), WindowBoundary::Calendar);
        assert_eq!(
            refusal.next_attempt(),
            &WindowNextAttempt::LongerSpanOfSameUnit { unit: PopulationUnit::environment(3) }
        );
    }

    #[test]
    fn chunk_boundary_concatenates_to_the_eager_slice() {
        let (eager, window) = fit_window();
        assert_eq!(eager, (4..10).collect::<Vec<_>>());
        assert!(window.chunk_at(4).is_err());
        assert!(window.chunk_at(10).is_err());

        let (left, right) = window.chunk_at(7).expect("cut inside the span");
        assert_eq!(left.local_rows(), 4..7);
        assert_eq!(right.local_rows(), 7..10);
        let mut concatenated = Vec::new();
        concatenated.extend(left.local_rows());
        concatenated.extend(right.local_rows());
        assert_eq!(concatenated, eager);
        assert_eq!(concatenated, window.eager_rows());

        let across = window.borrow_lag(7, 1).expect("chunk cut is not a lag boundary");
        let previous = eager[eager.iter().position(|row| *row == 7).expect("row 7") - 1];
        assert_eq!(across, 6);
        assert_eq!(across, previous);
        assert_eq!(right.borrow_lag(7, 1).expect("lag from the later chunk"), previous);
        assert!(right.borrow_lag(6, 1).is_err(), "row 6 is not a row of the later chunk");
    }

    /// A lag read through a chunk equals the whole-window read and an independent
    /// recomputation from the calendar, at every cut, target, and lag.
    #[test]
    fn chunk_borrow_matches_the_whole_window_at_every_cut() {
        // Rows 0..12 with a calendar break between rows 6 and 7, and a split 2..12.
        let calendar: Vec<i64> = vec![100, 101, 102, 103, 104, 105, 106, 110, 111, 112, 113, 114];
        let split = DeclaredSplit::try_new("fit", 2..12).expect("split");
        let partition =
            UnitPartition::try_new(PopulationUnit::panel(7), calendar.clone(), 1, [split])
                .expect("partition");
        let span = CalendarSpan::try_new(102, 114).expect("span");
        let window = partition.window("fit", span).expect("window");
        let eager = window.eager_rows();
        assert_eq!(eager, (2..11).collect::<Vec<_>>());
        let expected = |target: usize, lag: u32| -> Option<usize> {
            let source = target.checked_sub(lag as usize)?;
            let in_window = eager.contains(&source);
            let same_segment = (source..=target).all(|row| {
                i64::try_from(row - source).is_ok_and(|gap| calendar[row] - calendar[source] == gap)
            });
            (in_window && same_segment).then_some(source)
        };
        for cut in (window.local_rows().start + 1)..window.local_rows().end {
            let (left, right) = window.chunk_at(cut).expect("cut");
            for chunk in [&left, &right] {
                for target in chunk.local_rows() {
                    for lag in 0..6u32 {
                        let through_chunk = chunk.borrow_lag(target, lag).ok();
                        assert_eq!(through_chunk, window.borrow_lag(target, lag).ok());
                        assert_eq!(
                            through_chunk,
                            expected(target, lag),
                            "cut {cut} target {target} lag {lag}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn overlapping_splits_are_refused() {
        let calendar: Vec<i64> = (0..10u32).map(i64::from).collect();
        let discovery = DeclaredSplit::try_new("discovery", 0..6).expect("discovery");
        let estimation = DeclaredSplit::try_new("estimation", 4..10).expect("estimation");
        let overlap = UnitPartition::try_new(
            PopulationUnit::panel(1),
            calendar.clone(),
            1,
            [discovery, estimation],
        );
        assert!(overlap.is_err());
        let discovery = DeclaredSplit::try_new("discovery", 0..5).expect("discovery");
        let estimation = DeclaredSplit::try_new("estimation", 5..10).expect("estimation");
        UnitPartition::try_new(PopulationUnit::panel(1), calendar, 1, [discovery, estimation])
            .expect("adjacent splits share no row");
    }

    #[test]
    fn lag_across_a_split_stays_inside_the_split() {
        let calendar: Vec<i64> = (0..10u32).map(i64::from).collect();
        let eager = eager_of_whole_unit(&calendar, 4..10, 0..10);
        let discovery = DeclaredSplit::try_new("discovery", 0..4).expect("discovery");
        let estimation = DeclaredSplit::try_new("estimation", 4..10).expect("estimation");
        assert_eq!(estimation.name(), "estimation");
        assert_eq!(estimation.rows(), 4..10);
        let partition =
            UnitPartition::try_new(PopulationUnit::panel(4), calendar, 1, [discovery, estimation])
                .expect("partition");
        let span = CalendarSpan::try_new(0, 10).expect("span");
        let window = partition.window("estimation", span).expect("window");
        assert_eq!(eager, (4..10).collect::<Vec<_>>());
        assert_eq!(window.eager_rows(), eager);

        let error = window.borrow_lag(4, 1).expect_err("row 3 is in the other split");
        let refusal = error.refusal().expect("split miss is a refusal");
        assert_eq!(refusal.boundary(), WindowBoundary::Split);
        assert_eq!(
            refusal.next_attempt(),
            &WindowNextAttempt::StayInsideSplit { name: Arc::from("estimation") }
        );
        assert_eq!(window.borrow_lag(5, 1).expect("both rows are in the split"), 4);
    }

    #[test]
    fn lag_across_a_calendar_break_stays_inside_the_segment() {
        let calendar = vec![0, 1, 2, 5, 6, 7];
        let eager = eager_of_whole_unit(&calendar, 0..6, 0..8);
        let split = DeclaredSplit::try_new("all", 0..6).expect("split");
        let partition =
            UnitPartition::try_new(PopulationUnit::panel(2), calendar, 1, [split]).expect("unit");
        let span = CalendarSpan::try_new(0, 8).expect("span");
        let window = partition.window("all", span).expect("window");
        assert_eq!(eager, (0..6).collect::<Vec<_>>());
        assert_eq!(window.eager_rows(), eager);

        let (left, right) = window.chunk_at(3).expect("cut on the calendar break");
        let mut concatenated = Vec::new();
        concatenated.extend(left.local_rows());
        concatenated.extend(right.local_rows());
        assert_eq!(concatenated, eager);

        let error = window.borrow_lag(3, 1).expect_err("lag crosses the break");
        let refusal = error.refusal().expect("calendar break is a refusal");
        assert_eq!(refusal.boundary(), WindowBoundary::Calendar);
        assert_eq!(refusal.next_attempt(), &WindowNextAttempt::StayInsideCalendarSegment);
        assert_eq!(window.borrow_lag(4, 1).expect("same segment"), 3);
    }

    fn full_unit(unit: PopulationUnit, rows: usize) -> PanelWindow {
        let calendar: Vec<i64> = (0..rows).map(|row| i64::try_from(row).expect("row")).collect();
        let end = i64::try_from(rows).expect("length");
        let split = DeclaredSplit::try_new("all", 0..rows).expect("split");
        let partition = UnitPartition::try_new(unit, calendar, 1, [split]).expect("partition");
        let span = CalendarSpan::try_new(0, end).expect("span");
        partition.window("all", span).expect("window")
    }
}
