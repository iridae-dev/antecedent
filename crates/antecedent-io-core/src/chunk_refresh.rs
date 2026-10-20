//! Append, refresh, and window a ledger of data chunks and summaries.
//!
//! Each mutation updates a cached snapshot and leaves the retained chunks as
//! they are. [`ChunkLedger::fresh_snapshot`] rebuilds the complete result, the
//! diagnostics, the source identity, and the ledger state from those chunks
//! alone. A summary is an exact count, sum, and extrema over a whole chunk.
//! When rows were not retained, a window that cuts a chunk is refused. The
//! refusal is the fallback: the missing rows are not invented, and a gap is
//! not filled by treating the series as stationary.
//!
//! A lag is an earlier row of this series. A chunk cut is not a boundary. The
//! row a chunk reads is the row the eager slice reads. Overlapping window
//! splits are refused.
//!
//! Limits. One series, integer rows, and additive summaries. A caller-supplied
//! summary is trusted when the rows are gone. Nothing is written to disk.
//! Stationarity is not inferred and is not tested.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, reason = "the parent wires chunk_refresh into the crate root")]

use std::ops::Range;
use std::sync::Arc;

use antecedent_core::{CanonicalWriter, SemanticDigest};

const SOURCE_TAG: &str = "antecedent.chunk_refresh.source";
const LEDGER_TAG: &str = "antecedent.chunk_refresh.ledger";

/// Next attempt when a lag would leave this series.
pub const LONGER_SPAN_OF_SAME_UNIT: &str =
    "a longer span of the same unit, not a borrow from the neighbor";

/// What a gap fill would have claimed, and does not.
pub const STATIONARITY_NOT_INFERRED: &str = "stationarity was not inferred";

/// How chunk rows are kept after a mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ChunkRetention {
    /// Rows stay in memory.
    FullReusable,
    /// Per-chunk summaries stay. Rows do not.
    SummaryOnly,
    /// Declared disk-backed output. No path is stored and no bytes are written.
    DiskBacked,
}

impl ChunkRetention {
    /// Stable name used in refusals.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullReusable => "full reusable state",
            Self::SummaryOnly => "summary-only",
            Self::DiskBacked => "disk-backed output",
        }
    }
}

impl std::fmt::Display for ChunkRetention {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One row of one series.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkRow {
    source: u64,
    time: i64,
    value: i64,
}

impl ChunkRow {
    /// Row `source` at `time` with integer `value`.
    #[must_use]
    pub const fn new(source: u64, time: i64, value: i64) -> Self {
        Self { source, time, value }
    }

    /// Source identity of this row.
    #[must_use]
    pub const fn source(self) -> u64 {
        self.source
    }

    /// Time of this row.
    #[must_use]
    pub const fn time(self) -> i64 {
        self.time
    }

    /// Integer value.
    #[must_use]
    pub const fn value(self) -> i64 {
        self.value
    }
}

/// Half-open time split `[start, end)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowSplit {
    name: Arc<str>,
    start: i64,
    end: i64,
}

impl WindowSplit {
    /// Split `name` over times `[start, end)`.
    ///
    /// # Errors
    ///
    /// An empty name or an empty range.
    #[must_use = "an empty split is not a window"]
    pub fn try_new(name: impl Into<Arc<str>>, start: i64, end: i64) -> Result<Self, RefreshError> {
        let name = name.into();
        if name.as_ref().is_empty() {
            return Err(RefreshError::EmptySplitName);
        }
        if end <= start {
            return Err(RefreshError::EmptySplit { name });
        }
        Ok(Self { name, start, end })
    }

    /// Split name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    /// Inclusive start time.
    #[must_use]
    pub const fn start(&self) -> i64 {
        self.start
    }

    /// Exclusive end time.
    #[must_use]
    pub const fn end(&self) -> i64 {
        self.end
    }

    /// Whether `time` lies in the split.
    #[must_use]
    pub const fn contains(&self, time: i64) -> bool {
        time >= self.start && time < self.end
    }

    const fn range(&self) -> Range<i64> {
        self.start..self.end
    }
}

/// Disjoint named splits. Overlap is refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeclaredWindows {
    splits: Arc<[WindowSplit]>,
}

impl DeclaredWindows {
    /// Splits in caller order.
    ///
    /// # Errors
    ///
    /// No splits, a duplicate name, or two ranges that share a time.
    #[must_use = "overlapping splits are refused"]
    pub fn try_new(splits: impl IntoIterator<Item = WindowSplit>) -> Result<Self, RefreshError> {
        let splits: Vec<WindowSplit> = splits.into_iter().collect();
        if splits.is_empty() {
            return Err(RefreshError::NoSplits);
        }
        for (index, split) in splits.iter().enumerate() {
            for earlier in splits.iter().take(index) {
                if earlier.name.as_ref() == split.name.as_ref() {
                    return Err(RefreshError::DuplicateSplit { name: split.name.to_string() });
                }
                if earlier.start < split.end && split.start < earlier.end {
                    return Err(RefreshError::OverlappingSplits {
                        first: earlier.name.to_string(),
                        second: split.name.to_string(),
                    });
                }
            }
        }
        Ok(Self { splits: splits.into() })
    }

    /// The splits, in declaration order.
    #[must_use]
    pub fn splits(&self) -> &[WindowSplit] {
        &self.splits
    }
}

/// Additive result over the rows a view still includes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompleteResult {
    /// Rows included.
    pub count: u64,
    /// Sum of included values.
    pub sum: i64,
    /// Smallest included value.
    pub minimum: Option<i64>,
    /// Largest included value.
    pub maximum: Option<i64>,
}

/// What the cached view records about how it was built.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Diagnostics {
    /// Always false. No operation sets this.
    pub stationarity_inferred: bool,
    /// True when an included chunk contributed a summary because its rows are gone.
    pub summary_fallback: bool,
}

/// One retained chunk as the ledger last committed it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkState {
    /// Stable id assigned at append.
    pub id: u64,
    /// Whether the chunk still holds rows.
    pub retains_rows: bool,
    /// Rows the summary covers, which is the whole chunk.
    pub count: u64,
    /// Inclusive start time of the whole chunk.
    pub time_start: i64,
    /// Exclusive end time of the whole chunk.
    pub time_end: i64,
    /// Whether the active window includes any of the chunk.
    pub included: bool,
}

/// Retention, window, and chunk list.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerState {
    /// Retention last committed.
    pub retention: ChunkRetention,
    /// Calendar step, in time units per adjacent row of one segment.
    pub step: i64,
    /// Selected split, or the whole series when absent.
    pub selected_split: Option<Arc<str>>,
    /// Chunks in append order.
    pub chunks: Vec<ChunkState>,
}

/// Cached or reconstructed view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerSnapshot {
    /// Count, sum, and extrema.
    pub result: CompleteResult,
    /// Stationarity flag and summary fallback.
    pub diagnostics: Diagnostics,
    /// Digest of the included source identities, in chunk order.
    pub source_identity: SemanticDigest,
    /// Retention, selection, and per-chunk flags.
    pub state: LedgerState,
}

/// Why a ledger mutation was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RefreshError {
    /// A chunk or a filtered window contained no row.
    #[error("a chunk must contain at least one row")]
    EmptyChunk,
    /// Times went backwards, repeated, or ran into a neighbor chunk.
    #[error("chunk times must increase and must not share a time with a neighbor")]
    TimeOrder,
    /// No chunk has this id.
    #[error("unknown chunk {id}")]
    UnknownChunk {
        /// Requested id.
        id: u64,
    },
    /// Two splits share a time.
    #[error("splits '{first}' and '{second}' share times; declared splits must be disjoint")]
    OverlappingSplits {
        /// Earlier split.
        first: String,
        /// Later split.
        second: String,
    },
    /// A split name was empty.
    #[error("split name must be non-empty")]
    EmptySplitName,
    /// A split range was empty.
    #[error("split '{name}' must be non-empty")]
    EmptySplit {
        /// Split name.
        name: Arc<str>,
    },
    /// The same split name was declared twice.
    #[error("duplicate split '{name}'")]
    DuplicateSplit {
        /// Split name.
        name: String,
    },
    /// No splits were declared.
    #[error("a series needs at least one declared split")]
    NoSplits,
    /// The selected name is not declared.
    #[error("unknown split '{name}'")]
    UnknownSplit {
        /// Requested name.
        name: String,
    },
    /// The selected split meets no retained chunk.
    #[error("window does not meet a retained chunk")]
    WindowMiss,
    /// The calendar step was not positive.
    #[error("calendar step must be positive")]
    Step,
    /// A window cut a chunk that no longer has rows.
    #[error("{retention} cannot serve row reconstruction")]
    PartialWindowNeedsRows {
        /// Retention that dropped the rows.
        retention: ChunkRetention,
    },
    /// A lag or an eager slice needed rows that were not retained.
    #[error("{retention} cannot serve row reconstruction")]
    RowsNotRetained {
        /// Retention that dropped the rows.
        retention: ChunkRetention,
    },
    /// A gap fill was requested. The series is not treated as stationary.
    #[error("{STATIONARITY_NOT_INFERRED}")]
    StationarityNotInferred,
    /// A count, a sum, or a time bound did not fit.
    #[error("chunk summary overflowed")]
    Overflow,
    /// Eager indexes were not a contiguous range, so a chunk cut is undefined.
    #[error("eager slice is not a contiguous index range")]
    EagerSliceNotContiguous,
    /// The cut was not strictly inside the eager slice.
    #[error("chunk boundary {boundary} is not strictly inside {start}..{end}")]
    ChunkBoundary {
        /// Requested cut.
        boundary: usize,
        /// First eager index.
        start: usize,
        /// One past the last eager index.
        end: usize,
    },
}

/// Which boundary a lag tried to cross.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LagBoundary {
    /// The lagged index is before the first row.
    Unit,
    /// The lagged row belongs to another declared split.
    Split,
    /// The lagged row crosses a calendar segment.
    Calendar,
}

/// What to try after a refused lag.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LagNextAttempt {
    /// Widen this series. Do not read a neighbor.
    LongerSpanOfSameUnit,
    /// Keep the lag inside the split that contains the target.
    StayInsideSplit {
        /// Split that contains the target.
        name: Arc<str>,
    },
    /// Do not cross the calendar break.
    StayInsideCalendarSegment,
}

impl LagNextAttempt {
    /// Stable name of the attempt.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::LongerSpanOfSameUnit => LONGER_SPAN_OF_SAME_UNIT,
            Self::StayInsideSplit { .. } => "keep the lag inside the declared split",
            Self::StayInsideCalendarSegment => "stay inside the calendar segment",
        }
    }
}

/// A lag that would leave the series, the split, or the calendar segment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagRefusal {
    target: usize,
    lag: u32,
    boundary: LagBoundary,
    next_attempt: LagNextAttempt,
}

impl LagRefusal {
    /// Row the lag was read from.
    #[must_use]
    pub const fn target(&self) -> usize {
        self.target
    }

    /// Requested lag, in earlier rows.
    #[must_use]
    pub const fn lag(&self) -> u32 {
        self.lag
    }

    /// Boundary the lag would cross.
    #[must_use]
    pub const fn boundary(&self) -> LagBoundary {
        self.boundary
    }

    /// What to try instead.
    #[must_use]
    pub const fn next_attempt(&self) -> &LagNextAttempt {
        &self.next_attempt
    }
}

impl std::fmt::Display for LagRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.next_attempt {
            LagNextAttempt::LongerSpanOfSameUnit => write!(
                formatter,
                "lag would leave this series; next attempt: {LONGER_SPAN_OF_SAME_UNIT}"
            ),
            LagNextAttempt::StayInsideSplit { name } => {
                let name = name.as_ref();
                write!(
                    formatter,
                    "lag would cross split '{name}'; next attempt: keep the lag inside that split"
                )
            }
            LagNextAttempt::StayInsideCalendarSegment => write!(
                formatter,
                "lag would cross a calendar boundary; next attempt: stay inside the calendar \
                 segment"
            ),
        }
    }
}

impl std::error::Error for LagRefusal {}

/// Why a lag returned no row.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum BorrowError {
    /// `target` is not a row of the eager slice.
    #[error("target row {target} is outside {start}..{end}")]
    TargetOutsideWindow {
        /// Requested row.
        target: usize,
        /// First index of the window.
        start: usize,
        /// One past the last index of the window.
        end: usize,
    },
    /// `target` is not a row of this chunk.
    #[error("target row {target} is outside chunk {start}..{end}")]
    TargetOutsideChunk {
        /// Requested row.
        target: usize,
        /// First index of the chunk.
        start: usize,
        /// One past the last index of the chunk.
        end: usize,
    },
    /// Rows were not retained, so the eager slice cannot be built.
    #[error("{retention} cannot serve row reconstruction")]
    RowsNotRetained {
        /// Retention that dropped the rows.
        retention: ChunkRetention,
    },
    /// The lag would cross a unit, a split, or a calendar boundary.
    #[error("{0}")]
    Refused(LagRefusal),
}

impl BorrowError {
    /// Boundary refusal, when the target was inside the window.
    #[must_use = "none means the target was outside the window, not a boundary refusal"]
    pub const fn refusal(&self) -> Option<&LagRefusal> {
        match self {
            Self::Refused(refusal) => Some(refusal),
            Self::TargetOutsideWindow { .. }
            | Self::TargetOutsideChunk { .. }
            | Self::RowsNotRetained { .. } => None,
        }
    }
}

/// One side of a chunk cut. The range is a subrange of the eager slice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexChunk {
    rows: Arc<[ChunkRow]>,
    range: Range<usize>,
    split_name: Arc<str>,
    split: Range<i64>,
    boundaries: Arc<[usize]>,
}

impl IndexChunk {
    /// Unit indexes on this side of the cut.
    #[must_use]
    pub const fn indexes(&self) -> Range<usize> {
        self.range.start..self.range.end
    }

    /// Borrow the row `lag` steps before `target`.
    ///
    /// `target` has to be an index of this chunk. The cut is not a boundary:
    /// the row is the one a borrow on the eager slice returns.
    ///
    /// # Errors
    ///
    /// `target` is outside this chunk, or the series borrow is refused.
    #[must_use = "a refused lag names the next attempt and is not a neighbor row"]
    pub fn borrow_lag(&self, target: usize, lag: u32) -> Result<usize, BorrowError> {
        if !self.range.contains(&target) {
            return Err(BorrowError::TargetOutsideChunk {
                target,
                start: self.range.start,
                end: self.range.end,
            });
        }
        borrow_in_series(
            &self.rows,
            &self.split_name,
            self.split.start..self.split.end,
            &self.boundaries,
            target,
            lag,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct ChunkSummary {
    count: u64,
    sum: i64,
    minimum: i64,
    maximum: i64,
    time_start: i64,
    time_end: i64,
    source: SemanticDigest,
}

#[derive(Clone, Debug)]
struct RetainedChunk {
    id: u64,
    rows: Option<Arc<[ChunkRow]>>,
    summary: ChunkSummary,
}

#[derive(Clone, Copy)]
struct Parts<'a> {
    retention: ChunkRetention,
    step: i64,
    windows: &'a DeclaredWindows,
    selected: Option<&'a str>,
    chunks: &'a [RetainedChunk],
}

/// Ledger of chunks, a cached snapshot, and the retention mode.
#[derive(Clone, Debug)]
pub struct ChunkLedger {
    retention: ChunkRetention,
    step: i64,
    windows: DeclaredWindows,
    selected: Option<Arc<str>>,
    chunks: Vec<RetainedChunk>,
    next_id: u64,
    cached: LedgerSnapshot,
}

impl ChunkLedger {
    /// Empty ledger. `step` is the calendar distance between adjacent rows of one segment.
    ///
    /// # Errors
    ///
    /// `step` is not positive, or `windows` is not a set of disjoint splits.
    #[must_use = "an invalid ledger is not a series"]
    pub fn try_new(step: i64, windows: DeclaredWindows) -> Result<Self, RefreshError> {
        if step <= 0 {
            return Err(RefreshError::Step);
        }
        let retention = ChunkRetention::FullReusable;
        let chunks = Vec::new();
        let cached = cache_snapshot(Parts {
            retention,
            step,
            windows: &windows,
            selected: None,
            chunks: &chunks,
        })?;
        Ok(Self { retention, step, windows, selected: None, chunks, next_id: 1, cached })
    }

    /// Snapshot last committed by a mutation.
    #[must_use]
    pub const fn cached_snapshot(&self) -> &LedgerSnapshot {
        &self.cached
    }

    /// Snapshot rebuilt from the retained chunks, ignoring the cache.
    ///
    /// # Errors
    ///
    /// A selected window cuts a chunk that has no rows, or a sum overflows.
    #[must_use = "a reconstruction that fails did not see the rows"]
    pub fn fresh_snapshot(&self) -> Result<LedgerSnapshot, RefreshError> {
        fresh_snapshot(Parts {
            retention: self.retention,
            step: self.step,
            windows: &self.windows,
            selected: self.selected.as_deref(),
            chunks: &self.chunks,
        })
    }

    /// Append rows. Summary-only and disk-backed keep the summary and drop the rows.
    ///
    /// # Errors
    ///
    /// The rows are empty, out of order, overlap a neighbor, or overflow.
    #[must_use = "a refused append leaves the ledger unchanged"]
    pub fn append_rows(&mut self, rows: Arc<[ChunkRow]>) -> Result<u64, RefreshError> {
        let summary = summarize(&rows)?;
        let retained = match self.retention {
            ChunkRetention::FullReusable => Some(rows),
            ChunkRetention::SummaryOnly | ChunkRetention::DiskBacked => None,
        };
        self.push_chunk(retained, summary)
    }

    /// Append an existing summary. No rows are stored.
    ///
    /// # Errors
    ///
    /// The summary's time range is empty, unordered, or overlaps a neighbor.
    #[must_use = "a refused summary leaves the ledger unchanged"]
    pub fn append_summary(&mut self, summary: ChunkSummaryInput) -> Result<u64, RefreshError> {
        let summary = checked_summary(summary)?;
        self.push_chunk(None, summary)
    }

    /// Replace a chunk with rows. Reduced retention stores only the new summary.
    ///
    /// # Errors
    ///
    /// Unknown id, empty or unordered rows, a time range that hits a neighbor, or overflow.
    #[must_use = "a refused refresh leaves the ledger unchanged"]
    pub fn refresh_rows(&mut self, id: u64, rows: Arc<[ChunkRow]>) -> Result<(), RefreshError> {
        let summary = summarize(&rows)?;
        let retained = match self.retention {
            ChunkRetention::FullReusable => Some(rows),
            ChunkRetention::SummaryOnly | ChunkRetention::DiskBacked => None,
        };
        self.replace_chunk(id, retained, summary)
    }

    /// Replace a chunk with an existing summary and drop any rows it held.
    ///
    /// # Errors
    ///
    /// Unknown id, an unordered summary, or a time range that hits a neighbor.
    #[must_use = "a refused refresh leaves the ledger unchanged"]
    pub fn refresh_summary(
        &mut self,
        id: u64,
        summary: ChunkSummaryInput,
    ) -> Result<(), RefreshError> {
        let summary = checked_summary(summary)?;
        self.replace_chunk(id, None, summary)
    }

    /// Restrict the cached view to `name`.
    ///
    /// # Errors
    ///
    /// Unknown split, a split that meets no chunk, or a partial chunk without rows.
    #[must_use = "a refused window leaves the selection unchanged"]
    pub fn select_split(&mut self, name: &str) -> Result<(), RefreshError> {
        let selected = Some(Arc::<str>::from(name));
        self.commit(self.chunks.clone(), self.next_id, selected, self.retention)
    }

    /// View the whole series again.
    ///
    /// # Errors
    ///
    /// The retained summaries overflow.
    #[must_use = "a refused clear leaves the selection unchanged"]
    pub fn clear_split(&mut self) -> Result<(), RefreshError> {
        self.commit(self.chunks.clone(), self.next_id, None, self.retention)
    }

    /// Drop every row and keep per-chunk summaries.
    ///
    /// # Errors
    ///
    /// The active window cuts a chunk, which summaries cannot serve.
    #[must_use = "a refused retention change leaves the rows in place"]
    pub fn retain_summary_only(&mut self) -> Result<(), RefreshError> {
        self.drop_rows(ChunkRetention::SummaryOnly)
    }

    /// Drop every row, keep summaries, and record disk-backed output.
    ///
    /// No path is stored and nothing is written.
    ///
    /// # Errors
    ///
    /// The active window cuts a chunk.
    #[must_use = "a refused retention change leaves the rows in place"]
    pub fn retain_disk_backed(&mut self) -> Result<(), RefreshError> {
        self.drop_rows(ChunkRetention::DiskBacked)
    }

    /// Refuse to fill a gap.
    ///
    /// Every retention mode returns [`RefreshError::StationarityNotInferred`].
    /// The snapshot is unchanged because this does not write.
    #[must_use = "a gap stays open; stationarity was not inferred"]
    pub const fn impute_gap(&self) -> Result<(), RefreshError> {
        match self.retention {
            ChunkRetention::FullReusable
            | ChunkRetention::SummaryOnly
            | ChunkRetention::DiskBacked => Err(RefreshError::StationarityNotInferred),
        }
    }

    /// Unit indexes whose times lie in the active window, in series order.
    ///
    /// # Errors
    ///
    /// A chunk has no rows.
    #[must_use = "without the indexes there is no eager slice"]
    pub fn eager_indexes(&self) -> Result<Vec<usize>, RefreshError> {
        let rows = self.unit_rows()?;
        let (_split_name, split) = self.active_split(&rows)?;
        Ok((0..rows.len()).filter(|&index| split.contains(&rows[index].time)).collect())
    }

    /// Borrow the row `lag` steps before `target` on the eager slice.
    ///
    /// # Errors
    ///
    /// Rows are gone, `target` is outside the window, or the lag leaves the
    /// series, the split, or the calendar segment.
    #[must_use = "a refused lag names the next attempt and is not a neighbor row"]
    pub fn borrow_lag(&self, target: usize, lag: u32) -> Result<usize, BorrowError> {
        let rows = match self.unit_rows() {
            Ok(rows) => rows,
            Err(RefreshError::RowsNotRetained { retention }) => {
                return Err(BorrowError::RowsNotRetained { retention });
            }
            Err(_) => {
                return Err(BorrowError::TargetOutsideWindow { target, start: 0, end: 0 });
            }
        };
        if rows.is_empty() {
            return Err(BorrowError::TargetOutsideWindow { target, start: 0, end: 0 });
        }
        let Ok((split_name, split)) = self.active_split(&rows) else {
            return Err(BorrowError::TargetOutsideWindow { target, start: 0, end: rows.len() });
        };
        let Ok(boundaries) = segment_boundaries(&rows, self.step) else {
            return Err(BorrowError::TargetOutsideWindow { target, start: 0, end: rows.len() });
        };
        borrow_in_series(&rows, &split_name, split, &boundaries, target, lag)
    }

    /// Cut the eager slice at a unit index strictly inside it.
    ///
    /// # Errors
    ///
    /// Rows are gone, the slice is empty or not contiguous, or `boundary` is
    /// not strictly inside it.
    #[must_use = "dropping the chunks drops one side of the eager slice"]
    pub fn chunk_at(&self, boundary: usize) -> Result<(IndexChunk, IndexChunk), RefreshError> {
        let rows: Arc<[ChunkRow]> = self.unit_rows()?.into();
        let (split_name, split) = self.active_split(&rows)?;
        let indexes: Vec<usize> =
            (0..rows.len()).filter(|&index| split.contains(&rows[index].time)).collect();
        let range = contiguous_range(&indexes)?;
        if boundary <= range.start || boundary >= range.end {
            return Err(RefreshError::ChunkBoundary {
                boundary,
                start: range.start,
                end: range.end,
            });
        }
        let boundaries: Arc<[usize]> = segment_boundaries(&rows, self.step)?.into();
        let left = IndexChunk {
            rows: Arc::clone(&rows),
            range: range.start..boundary,
            split_name: Arc::clone(&split_name),
            split: split.start..split.end,
            boundaries: Arc::clone(&boundaries),
        };
        let right = IndexChunk {
            rows,
            range: boundary..range.end,
            split_name,
            split: split.start..split.end,
            boundaries,
        };
        Ok((left, right))
    }

    fn push_chunk(
        &mut self,
        rows: Option<Arc<[ChunkRow]>>,
        summary: ChunkSummary,
    ) -> Result<u64, RefreshError> {
        if let Some(previous) = self.chunks.last() {
            if summary.time_start < previous.summary.time_end {
                return Err(RefreshError::TimeOrder);
            }
        }
        let id = self.next_id;
        let next_id = id.checked_add(1).ok_or(RefreshError::Overflow)?;
        let mut chunks = self.chunks.clone();
        chunks.push(RetainedChunk { id, rows, summary });
        self.commit(chunks, next_id, self.selected.clone(), self.retention)?;
        Ok(id)
    }

    fn replace_chunk(
        &mut self,
        id: u64,
        rows: Option<Arc<[ChunkRow]>>,
        summary: ChunkSummary,
    ) -> Result<(), RefreshError> {
        let index = self.chunk_index(id)?;
        if let Some(previous) = index.checked_sub(1).and_then(|prior| self.chunks.get(prior)) {
            if summary.time_start < previous.summary.time_end {
                return Err(RefreshError::TimeOrder);
            }
        }
        if let Some(next) = index.checked_add(1).and_then(|next| self.chunks.get(next)) {
            if summary.time_end > next.summary.time_start {
                return Err(RefreshError::TimeOrder);
            }
        }
        let mut chunks = self.chunks.clone();
        chunks[index] = RetainedChunk { id, rows, summary };
        self.commit(chunks, self.next_id, self.selected.clone(), self.retention)
    }

    fn drop_rows(&mut self, retention: ChunkRetention) -> Result<(), RefreshError> {
        let chunks = self
            .chunks
            .iter()
            .map(|chunk| RetainedChunk { id: chunk.id, rows: None, summary: chunk.summary })
            .collect();
        self.commit(chunks, self.next_id, self.selected.clone(), retention)
    }

    fn commit(
        &mut self,
        chunks: Vec<RetainedChunk>,
        next_id: u64,
        selected: Option<Arc<str>>,
        retention: ChunkRetention,
    ) -> Result<(), RefreshError> {
        let cached = cache_snapshot(Parts {
            retention,
            step: self.step,
            windows: &self.windows,
            selected: selected.as_deref(),
            chunks: &chunks,
        })?;
        self.chunks = chunks;
        self.next_id = next_id;
        self.selected = selected;
        self.retention = retention;
        self.cached = cached;
        Ok(())
    }

    fn chunk_index(&self, id: u64) -> Result<usize, RefreshError> {
        self.chunks.iter().position(|chunk| chunk.id == id).ok_or(RefreshError::UnknownChunk { id })
    }

    fn unit_rows(&self) -> Result<Vec<ChunkRow>, RefreshError> {
        let mut rows = Vec::new();
        for chunk in &self.chunks {
            let Some(stored) = &chunk.rows else {
                return Err(RefreshError::RowsNotRetained { retention: self.retention });
            };
            rows.extend(stored.iter().copied());
        }
        Ok(rows)
    }

    fn active_split(&self, rows: &[ChunkRow]) -> Result<(Arc<str>, Range<i64>), RefreshError> {
        if let Some(name) = &self.selected {
            let split = find_split(&self.windows, name)?;
            return Ok((Arc::clone(&split.name), split.range()));
        }
        let Some(first) = rows.first() else {
            return Err(RefreshError::WindowMiss);
        };
        let Some(last) = rows.last() else {
            return Err(RefreshError::WindowMiss);
        };
        let end = last.time.checked_add(1).ok_or(RefreshError::Overflow)?;
        Ok((Arc::from("series"), first.time..end))
    }
}

/// Caller-supplied summary for a chunk whose rows are not in this ledger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkSummaryInput {
    /// Row count. Must be positive.
    pub count: u64,
    /// Sum of the values.
    pub sum: i64,
    /// Smallest value.
    pub minimum: i64,
    /// Largest value.
    pub maximum: i64,
    /// Inclusive start time.
    pub time_start: i64,
    /// Exclusive end time.
    pub time_end: i64,
    /// Source identity the caller already sealed for these rows.
    pub source: SemanticDigest,
}

fn checked_summary(input: ChunkSummaryInput) -> Result<ChunkSummary, RefreshError> {
    if input.count == 0 || input.time_end <= input.time_start || input.minimum > input.maximum {
        return Err(RefreshError::EmptyChunk);
    }
    Ok(ChunkSummary {
        count: input.count,
        sum: input.sum,
        minimum: input.minimum,
        maximum: input.maximum,
        time_start: input.time_start,
        time_end: input.time_end,
        source: input.source,
    })
}

fn summarize(rows: &[ChunkRow]) -> Result<ChunkSummary, RefreshError> {
    let Some(first) = rows.first() else {
        return Err(RefreshError::EmptyChunk);
    };
    let mut count = 0u64;
    let mut sum = 0i64;
    let mut minimum = first.value;
    let mut maximum = first.value;
    let mut sources = Vec::with_capacity(rows.len());
    let mut previous: Option<i64> = None;
    for row in rows {
        if let Some(time) = previous {
            if row.time <= time {
                return Err(RefreshError::TimeOrder);
            }
        }
        previous = Some(row.time);
        count = count.checked_add(1).ok_or(RefreshError::Overflow)?;
        sum = sum.checked_add(row.value).ok_or(RefreshError::Overflow)?;
        minimum = minimum.min(row.value);
        maximum = maximum.max(row.value);
        sources.push(row.source);
    }
    let last = rows.last().ok_or(RefreshError::EmptyChunk)?;
    let time_end = last.time.checked_add(1).ok_or(RefreshError::Overflow)?;
    Ok(ChunkSummary {
        count,
        sum,
        minimum,
        maximum,
        time_start: first.time,
        time_end,
        source: digest_source_ids(&sources),
    })
}

fn digest_source_ids(sources: &[u64]) -> SemanticDigest {
    let mut writer = CanonicalWriter::new(SOURCE_TAG);
    let count = u64::try_from(sources.len()).unwrap_or(u64::MAX);
    writer.word(count);
    for source in sources {
        writer.word(*source);
    }
    SemanticDigest::from_bytes(writer.finish())
}

fn ledger_source(digests: &[SemanticDigest], split: Option<&str>) -> SemanticDigest {
    let mut writer = CanonicalWriter::new(LEDGER_TAG);
    writer.opt_text(split);
    let count = u64::try_from(digests.len()).unwrap_or(u64::MAX);
    writer.word(count);
    for digest in digests {
        writer.bytes(digest.as_bytes());
    }
    SemanticDigest::from_bytes(writer.finish())
}

fn find_split<'a>(
    windows: &'a DeclaredWindows,
    name: &str,
) -> Result<&'a WindowSplit, RefreshError> {
    windows
        .splits
        .iter()
        .find(|split| split.name.as_ref() == name)
        .ok_or_else(|| RefreshError::UnknownSplit { name: name.to_string() })
}

fn coverage<'a>(summary: &ChunkSummary, split: Option<&'a WindowSplit>) -> Coverage<'a> {
    let Some(split) = split else {
        return Coverage::Whole;
    };
    if summary.time_end <= split.start || summary.time_start >= split.end {
        Coverage::None
    } else if summary.time_start >= split.start && summary.time_end <= split.end {
        Coverage::Whole
    } else {
        Coverage::Partial(split)
    }
}

enum Coverage<'a> {
    Whole,
    Partial(&'a WindowSplit),
    None,
}

fn cache_snapshot(parts: Parts<'_>) -> Result<LedgerSnapshot, RefreshError> {
    let split = match parts.selected {
        Some(name) => Some(find_split(parts.windows, name)?),
        None => None,
    };
    let mut pieces = Vec::new();
    let mut digests = Vec::new();
    let mut fallback = false;
    let mut included = Vec::new();
    for chunk in parts.chunks {
        match coverage(&chunk.summary, split) {
            Coverage::None => included.push(false),
            Coverage::Whole => {
                pieces.push(chunk.summary);
                digests.push(chunk.summary.source);
                if chunk.rows.is_none() {
                    fallback = true;
                }
                included.push(true);
            }
            Coverage::Partial(window) => {
                let Some(rows) = &chunk.rows else {
                    return Err(RefreshError::PartialWindowNeedsRows {
                        retention: parts.retention,
                    });
                };
                let filtered = filter_rows(rows, window);
                if filtered.is_empty() {
                    included.push(false);
                    continue;
                }
                let summary = summarize(&filtered)?;
                digests.push(summary.source);
                pieces.push(summary);
                included.push(true);
            }
        }
    }
    finish_snapshot(parts, &pieces, &digests, fallback, &included)
}

fn fresh_snapshot(parts: Parts<'_>) -> Result<LedgerSnapshot, RefreshError> {
    let split = match parts.selected {
        Some(name) => Some(find_split(parts.windows, name)?),
        None => None,
    };
    let mut count = 0u64;
    let mut sum = 0i64;
    let mut minimum: Option<i64> = None;
    let mut maximum: Option<i64> = None;
    let mut digests = Vec::new();
    let mut fallback = false;
    let mut included = Vec::new();
    for chunk in parts.chunks {
        match coverage(&chunk.summary, split) {
            Coverage::None => included.push(false),
            Coverage::Whole => {
                included.push(true);
                if let Some(rows) = &chunk.rows {
                    fold_into(rows, &mut count, &mut sum, &mut minimum, &mut maximum)?;
                    let sources: Vec<u64> = rows.iter().copied().map(ChunkRow::source).collect();
                    digests.push(digest_source_ids(&sources));
                } else {
                    fallback = true;
                    fold_summary(&chunk.summary, &mut count, &mut sum, &mut minimum, &mut maximum)?;
                    digests.push(chunk.summary.source);
                }
            }
            Coverage::Partial(window) => {
                let Some(rows) = &chunk.rows else {
                    return Err(RefreshError::PartialWindowNeedsRows {
                        retention: parts.retention,
                    });
                };
                let filtered = filter_rows(rows, window);
                if filtered.is_empty() {
                    included.push(false);
                    continue;
                }
                fold_into(&filtered, &mut count, &mut sum, &mut minimum, &mut maximum)?;
                let sources: Vec<u64> = filtered.iter().copied().map(ChunkRow::source).collect();
                digests.push(digest_source_ids(&sources));
                included.push(true);
            }
        }
    }
    if parts.selected.is_some() && !included.contains(&true) {
        return Err(RefreshError::WindowMiss);
    }
    Ok(LedgerSnapshot {
        result: CompleteResult { count, sum, minimum, maximum },
        diagnostics: Diagnostics { stationarity_inferred: false, summary_fallback: fallback },
        source_identity: ledger_source(&digests, parts.selected),
        state: ledger_state(&parts, &included),
    })
}

fn finish_snapshot(
    parts: Parts<'_>,
    pieces: &[ChunkSummary],
    digests: &[SemanticDigest],
    fallback: bool,
    included: &[bool],
) -> Result<LedgerSnapshot, RefreshError> {
    if parts.selected.is_some() && !included.contains(&true) {
        return Err(RefreshError::WindowMiss);
    }
    Ok(LedgerSnapshot {
        result: combine(pieces)?,
        diagnostics: Diagnostics { stationarity_inferred: false, summary_fallback: fallback },
        source_identity: ledger_source(digests, parts.selected),
        state: ledger_state(&parts, included),
    })
}

fn combine(pieces: &[ChunkSummary]) -> Result<CompleteResult, RefreshError> {
    let mut count = 0u64;
    let mut sum = 0i64;
    let mut minimum: Option<i64> = None;
    let mut maximum: Option<i64> = None;
    for piece in pieces {
        count = count.checked_add(piece.count).ok_or(RefreshError::Overflow)?;
        sum = sum.checked_add(piece.sum).ok_or(RefreshError::Overflow)?;
        minimum = Some(minimum.map_or(piece.minimum, |current| current.min(piece.minimum)));
        maximum = Some(maximum.map_or(piece.maximum, |current| current.max(piece.maximum)));
    }
    Ok(CompleteResult { count, sum, minimum, maximum })
}

fn fold_into(
    rows: &[ChunkRow],
    count: &mut u64,
    sum: &mut i64,
    minimum: &mut Option<i64>,
    maximum: &mut Option<i64>,
) -> Result<(), RefreshError> {
    for row in rows {
        *count = count.checked_add(1).ok_or(RefreshError::Overflow)?;
        *sum = sum.checked_add(row.value).ok_or(RefreshError::Overflow)?;
        *minimum = Some(minimum.map_or(row.value, |current| current.min(row.value)));
        *maximum = Some(maximum.map_or(row.value, |current| current.max(row.value)));
    }
    Ok(())
}

fn fold_summary(
    summary: &ChunkSummary,
    count: &mut u64,
    sum: &mut i64,
    minimum: &mut Option<i64>,
    maximum: &mut Option<i64>,
) -> Result<(), RefreshError> {
    *count = count.checked_add(summary.count).ok_or(RefreshError::Overflow)?;
    *sum = sum.checked_add(summary.sum).ok_or(RefreshError::Overflow)?;
    *minimum = Some(minimum.map_or(summary.minimum, |current| current.min(summary.minimum)));
    *maximum = Some(maximum.map_or(summary.maximum, |current| current.max(summary.maximum)));
    Ok(())
}

fn filter_rows(rows: &[ChunkRow], split: &WindowSplit) -> Vec<ChunkRow> {
    rows.iter().copied().filter(|row| split.contains(row.time)).collect()
}

fn ledger_state(parts: &Parts<'_>, included: &[bool]) -> LedgerState {
    let chunks = parts
        .chunks
        .iter()
        .zip(included.iter().copied())
        .map(|(chunk, included)| ChunkState {
            id: chunk.id,
            retains_rows: chunk.rows.is_some(),
            count: chunk.summary.count,
            time_start: chunk.summary.time_start,
            time_end: chunk.summary.time_end,
            included,
        })
        .collect();
    LedgerState {
        retention: parts.retention,
        step: parts.step,
        selected_split: parts.selected.map(Arc::from),
        chunks,
    }
}

fn segment_boundaries(rows: &[ChunkRow], step: i64) -> Result<Vec<usize>, RefreshError> {
    let mut boundaries = Vec::new();
    for index in 1..rows.len() {
        let later = rows[index].time;
        let earlier = rows[index - 1].time;
        let gap = later.checked_sub(earlier).ok_or(RefreshError::TimeOrder)?;
        if gap != step {
            boundaries.push(index);
        }
    }
    Ok(boundaries)
}

fn contiguous_range(indexes: &[usize]) -> Result<Range<usize>, RefreshError> {
    let Some(&start) = indexes.first() else {
        return Err(RefreshError::WindowMiss);
    };
    let Some(&last) = indexes.last() else {
        return Err(RefreshError::WindowMiss);
    };
    let end = last.checked_add(1).ok_or(RefreshError::Overflow)?;
    if end.saturating_sub(start) != indexes.len() {
        return Err(RefreshError::EagerSliceNotContiguous);
    }
    Ok(start..end)
}

fn borrow_in_series(
    rows: &[ChunkRow],
    split_name: &Arc<str>,
    split: Range<i64>,
    boundaries: &[usize],
    target: usize,
    lag: u32,
) -> Result<usize, BorrowError> {
    let Some(row) = rows.get(target) else {
        return Err(BorrowError::TargetOutsideWindow { target, start: 0, end: rows.len() });
    };
    if !split.contains(&row.time) {
        let start = (0..rows.len()).find(|index| split.contains(&rows[*index].time)).unwrap_or(0);
        let end = (0..rows.len())
            .rev()
            .find(|index| split.contains(&rows[*index].time))
            .map_or(start, |index| index.saturating_add(1));
        return Err(BorrowError::TargetOutsideWindow { target, start, end });
    }
    let Ok(lag_rows) = usize::try_from(lag) else {
        return Err(BorrowError::Refused(LagRefusal {
            target,
            lag,
            boundary: LagBoundary::Unit,
            next_attempt: LagNextAttempt::LongerSpanOfSameUnit,
        }));
    };
    let Some(source) = target.checked_sub(lag_rows) else {
        return Err(BorrowError::Refused(LagRefusal {
            target,
            lag,
            boundary: LagBoundary::Unit,
            next_attempt: LagNextAttempt::LongerSpanOfSameUnit,
        }));
    };
    let Some(source_row) = rows.get(source) else {
        return Err(BorrowError::Refused(LagRefusal {
            target,
            lag,
            boundary: LagBoundary::Unit,
            next_attempt: LagNextAttempt::LongerSpanOfSameUnit,
        }));
    };
    if !split.contains(&source_row.time) {
        return Err(BorrowError::Refused(LagRefusal {
            target,
            lag,
            boundary: LagBoundary::Split,
            next_attempt: LagNextAttempt::StayInsideSplit { name: Arc::clone(split_name) },
        }));
    }
    if crosses_calendar(boundaries, source, target) {
        return Err(BorrowError::Refused(LagRefusal {
            target,
            lag,
            boundary: LagBoundary::Calendar,
            next_attempt: LagNextAttempt::StayInsideCalendarSegment,
        }));
    }
    Ok(source)
}

fn crosses_calendar(boundaries: &[usize], source: usize, target: usize) -> bool {
    boundaries.iter().any(|&boundary| source < boundary && boundary <= target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn windows() -> DeclaredWindows {
        let fit = WindowSplit::try_new("fit", 0, 10).expect("fit");
        let hold = WindowSplit::try_new("hold", 10, 20).expect("hold");
        DeclaredWindows::try_new([fit, hold]).expect("disjoint splits")
    }

    fn row(source: u64, time: i64, value: i64) -> ChunkRow {
        ChunkRow::new(source, time, value)
    }

    fn series(rows: &[ChunkRow]) -> (u64, i64, i64, i64) {
        let count = u64::try_from(rows.len()).expect("len");
        let sum = rows.iter().copied().map(ChunkRow::value).sum();
        let minimum = rows.iter().copied().map(ChunkRow::value).min().expect("rows");
        let maximum = rows.iter().copied().map(ChunkRow::value).max().expect("rows");
        (count, sum, minimum, maximum)
    }

    fn check(ledger: &ChunkLedger) {
        let fresh = ledger.fresh_snapshot().expect("fresh reconstruction");
        let cached = ledger.cached_snapshot();
        assert_eq!(cached.result, fresh.result);
        assert_eq!(cached.diagnostics, fresh.diagnostics);
        assert!(!cached.diagnostics.stationarity_inferred);
        assert_eq!(cached.source_identity, fresh.source_identity);
        assert_eq!(cached.state, fresh.state);
    }

    fn expect_result(ledger: &ChunkLedger, rows: &[ChunkRow]) {
        let (count, sum, minimum, maximum) = series(rows);
        let result = &ledger.cached_snapshot().result;
        assert_eq!(result.count, count);
        assert_eq!(result.sum, sum);
        assert_eq!(result.minimum, Some(minimum));
        assert_eq!(result.maximum, Some(maximum));
        check(ledger);
    }

    #[test]
    fn appends_refreshes_and_windows_match_a_fresh_reconstruction() {
        let mut ledger = ChunkLedger::try_new(1, windows()).expect("ledger");
        check(&ledger);
        assert!(!ledger.cached_snapshot().diagnostics.summary_fallback);

        let first = Arc::<[ChunkRow]>::from(vec![row(1, 0, 1), row(2, 1, 2), row(3, 2, 3)]);
        let second = Arc::<[ChunkRow]>::from(vec![row(4, 3, 4), row(5, 4, 5)]);
        let first_id = ledger.append_rows(Arc::clone(&first)).expect("append");
        expect_result(&ledger, &first);
        let second_id = ledger.append_rows(Arc::clone(&second)).expect("append");
        let mut all = first.to_vec();
        all.extend(second.iter().copied());
        expect_result(&ledger, &all);

        let refreshed = Arc::<[ChunkRow]>::from(vec![row(9, 0, 10), row(2, 1, 2), row(3, 2, 3)]);
        let before = ledger.cached_snapshot().source_identity;
        ledger.refresh_rows(first_id, Arc::clone(&refreshed)).expect("refresh");
        assert_ne!(ledger.cached_snapshot().source_identity, before);
        let mut after = refreshed.to_vec();
        after.extend(second.iter().copied());
        expect_result(&ledger, &after);

        let overlap = Arc::<[ChunkRow]>::from(vec![row(4, 3, 1)]);
        let snapshot = ledger.cached_snapshot().clone();
        assert!(ledger.refresh_rows(first_id, overlap).is_err());
        assert_eq!(ledger.cached_snapshot(), &snapshot);
        check(&ledger);

        ledger.select_split("fit").expect("both chunks sit inside fit");
        expect_result(&ledger, &after);
        assert!(ledger.select_split("hold").is_err());
        expect_result(&ledger, &after);

        let third = Arc::<[ChunkRow]>::from(vec![row(6, 10, 7), row(7, 11, 8)]);
        ledger.clear_split().expect("clear");
        let third_id = ledger.append_rows(Arc::clone(&third)).expect("append");
        ledger.select_split("hold").expect("hold");
        expect_result(&ledger, &third);
        assert!(!ledger.cached_snapshot().diagnostics.summary_fallback);

        ledger.select_split("fit").expect("fit");
        expect_result(&ledger, &after);
        assert_ne!(first_id, second_id);
        assert_ne!(second_id, third_id);
    }

    #[test]
    fn summary_fallback_matches_reconstruction_and_drops_rows() {
        let mut ledger = ChunkLedger::try_new(1, windows()).expect("ledger");
        let rows = Arc::<[ChunkRow]>::from(vec![row(1, 0, 4), row(2, 1, 6)]);
        let id = ledger.append_rows(Arc::clone(&rows)).expect("append");
        assert!(Arc::strong_count(&rows) >= 2);
        ledger.retain_summary_only().expect("summary");
        assert_eq!(Arc::strong_count(&rows), 1);
        assert!(ledger.cached_snapshot().diagnostics.summary_fallback);
        assert!(!ledger.cached_snapshot().state.chunks[0].retains_rows);
        assert_eq!(ledger.cached_snapshot().state.retention, ChunkRetention::SummaryOnly);
        expect_result(&ledger, &rows);

        let identity = ledger.cached_snapshot().source_identity;
        ledger
            .refresh_summary(
                id,
                ChunkSummaryInput {
                    count: 2,
                    sum: 9,
                    minimum: 4,
                    maximum: 5,
                    time_start: 0,
                    time_end: 2,
                    source: identity,
                },
            )
            .expect("summary refresh");
        let result = &ledger.cached_snapshot().result;
        assert_eq!(
            (result.count, result.sum, result.minimum, result.maximum),
            (2, 9, Some(4), Some(5))
        );
        check(&ledger);

        ledger
            .append_summary(ChunkSummaryInput {
                count: 1,
                sum: 8,
                minimum: 8,
                maximum: 8,
                time_start: 10,
                time_end: 11,
                source: identity,
            })
            .expect("existing summary");
        let result = &ledger.cached_snapshot().result;
        assert_eq!(
            (result.count, result.sum, result.minimum, result.maximum),
            (3, 17, Some(4), Some(8))
        );
        assert!(ledger.cached_snapshot().diagnostics.summary_fallback);
        check(&ledger);
    }

    #[test]
    fn partial_window_needs_rows_after_summary_retention() {
        let mut ledger = ChunkLedger::try_new(
            1,
            DeclaredWindows::try_new([
                WindowSplit::try_new("early", 0, 2).expect("early"),
                WindowSplit::try_new("late", 2, 6).expect("late"),
            ])
            .expect("splits"),
        )
        .expect("ledger");
        ledger
            .append_rows(Arc::from(vec![row(1, 0, 1), row(2, 1, 1), row(3, 2, 1), row(4, 3, 1)]))
            .expect("append");
        ledger.select_split("early").expect("early takes a prefix of the chunk");
        expect_result(&ledger, &[row(1, 0, 1), row(2, 1, 1)]);
        assert!(!ledger.cached_snapshot().diagnostics.summary_fallback);
        let refusal = ledger.retain_summary_only().expect_err("partial window still needs rows");
        assert!(matches!(
            refusal,
            RefreshError::PartialWindowNeedsRows { retention: ChunkRetention::SummaryOnly }
        ));
        assert!(ledger.cached_snapshot().state.chunks[0].retains_rows);
        check(&ledger);
        let kept = ledger.cached_snapshot().clone();
        ledger.clear_split().expect("clear");
        ledger.retain_summary_only().expect("rows dropped while the whole chunk is in view");
        let refusal = ledger.select_split("early").expect_err("partial summary");
        assert!(matches!(
            refusal,
            RefreshError::PartialWindowNeedsRows { retention: ChunkRetention::SummaryOnly }
        ));
        assert_eq!(refusal.to_string(), "summary-only cannot serve row reconstruction");
        check(&ledger);
        assert_eq!(ledger.cached_snapshot().state.selected_split, None);
        assert_ne!(ledger.cached_snapshot().result, kept.result);
    }

    #[test]
    fn disk_backed_retention_matches_summaries_and_refuses_lag() {
        let mut ledger = ChunkLedger::try_new(1, windows()).expect("ledger");
        let rows = Arc::<[ChunkRow]>::from(vec![row(1, 0, 3), row(2, 1, 4)]);
        ledger.append_rows(Arc::clone(&rows)).expect("append");
        ledger.retain_disk_backed().expect("disk");
        assert_eq!(Arc::strong_count(&rows), 1);
        assert_eq!(ledger.cached_snapshot().state.retention, ChunkRetention::DiskBacked);
        assert!(ledger.cached_snapshot().diagnostics.summary_fallback);
        expect_result(&ledger, &rows);
        let refusal = ledger.borrow_lag(1, 1).expect_err("no rows");
        assert!(matches!(
            refusal,
            BorrowError::RowsNotRetained { retention: ChunkRetention::DiskBacked }
        ));
        assert!(ledger.eager_indexes().is_err());
    }

    #[test]
    fn gap_imputation_does_not_infer_stationarity() {
        let mut ledger = ChunkLedger::try_new(1, windows()).expect("ledger");
        ledger.append_rows(Arc::from(vec![row(1, 0, 1), row(2, 2, 1)])).expect("gap at time 1");
        let before = ledger.cached_snapshot().clone();
        let refusal = ledger.impute_gap().expect_err("no stationary fill");
        assert_eq!(refusal, RefreshError::StationarityNotInferred);
        assert_eq!(refusal.to_string(), STATIONARITY_NOT_INFERRED);
        assert_eq!(ledger.cached_snapshot(), &before);
        assert!(!ledger.cached_snapshot().diagnostics.stationarity_inferred);
        check(&ledger);
    }

    #[test]
    fn overlapping_window_splits_are_refused() {
        let discovery = WindowSplit::try_new("discovery", 0, 6).expect("discovery");
        let estimation = WindowSplit::try_new("estimation", 4, 10).expect("estimation");
        let overlap = DeclaredWindows::try_new([discovery, estimation]).expect_err("overlap");
        assert!(matches!(overlap, RefreshError::OverlappingSplits { .. }));
        assert!(overlap.to_string().contains("discovery"));
        assert!(overlap.to_string().contains("estimation"));
        let discovery = WindowSplit::try_new("discovery", 0, 5).expect("discovery");
        let estimation = WindowSplit::try_new("estimation", 5, 10).expect("estimation");
        DeclaredWindows::try_new([discovery, estimation]).expect("adjacent splits share no time");
    }

    #[test]
    fn chunk_lag_matches_the_eager_slice() {
        let calendar = [0_i64, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14];
        let rows: Vec<ChunkRow> = calendar
            .iter()
            .enumerate()
            .map(|(index, time)| row(u64::try_from(index).expect("index"), *time, 1))
            .collect();
        let mut ledger = ChunkLedger::try_new(
            1,
            DeclaredWindows::try_new([
                WindowSplit::try_new("prefix", 0, 2).expect("prefix"),
                WindowSplit::try_new("fit", 2, 14).expect("fit"),
            ])
            .expect("splits"),
        )
        .expect("ledger");
        ledger.append_rows(Arc::from(rows.clone())).expect("one chunk");
        ledger.select_split("fit").expect("fit");
        let eager = ledger.eager_indexes().expect("eager");
        assert_eq!(eager, (2..11).collect::<Vec<_>>());
        let unit = rows;
        let expected = |target: usize, lag: u32| -> Option<usize> {
            if !eager.contains(&target) {
                return None;
            }
            let source = target.checked_sub(lag as usize)?;
            if !eager.contains(&source) {
                return None;
            }
            let same_segment = (source + 1..=target)
                .all(|index| unit[index].time.checked_sub(unit[index - 1].time) == Some(1));
            same_segment.then_some(source)
        };
        assert_eq!(ledger.borrow_lag(4, 2).expect("inside the segment"), 2);
        assert_eq!(expected(4, 2), Some(2));
        let split_miss = ledger.borrow_lag(2, 1).expect_err("time 1 is in the other split");
        let refusal = split_miss.refusal().expect("split");
        assert_eq!(refusal.boundary(), LagBoundary::Split);
        assert_eq!(refusal.next_attempt().name(), "keep the lag inside the declared split");
        let unit_miss = ledger.borrow_lag(2, 3).expect_err("before the first row");
        let unit_refusal = unit_miss.refusal().expect("unit");
        assert_eq!(unit_refusal.boundary(), LagBoundary::Unit);
        assert_eq!(unit_refusal.next_attempt().name(), LONGER_SPAN_OF_SAME_UNIT);
        let calendar_miss = ledger.borrow_lag(7, 1).expect_err("calendar break");
        assert_eq!(calendar_miss.refusal().expect("calendar").boundary(), LagBoundary::Calendar);

        let last = eager[eager.len() - 1];
        for cut in (eager[0] + 1)..=last {
            let (left, right) = ledger.chunk_at(cut).expect("cut");
            let mut concatenated = Vec::new();
            concatenated.extend(left.indexes());
            concatenated.extend(right.indexes());
            assert_eq!(concatenated, eager);
            for chunk in [&left, &right] {
                for target in chunk.indexes() {
                    for lag in 0..6u32 {
                        let through_chunk = chunk.borrow_lag(target, lag).ok();
                        let through_series = ledger.borrow_lag(target, lag).ok();
                        assert_eq!(through_chunk, through_series);
                        assert_eq!(
                            through_chunk,
                            expected(target, lag),
                            "cut {cut} target {target} lag {lag}"
                        );
                    }
                }
            }
        }
        let outside = right_outside(&ledger);
        assert!(outside.is_err());
    }

    fn right_outside(ledger: &ChunkLedger) -> Result<usize, BorrowError> {
        let (left, right) = ledger.chunk_at(6).expect("cut");
        assert!(left.borrow_lag(6, 1).is_err());
        right.borrow_lag(left.indexes().start, 1)
    }
}
