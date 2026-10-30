//! Tiled evaluation of a frozen linear prediction over rows, draws, and grid or horizon points.
//!
//! Each tile streams one reduction the caller requested. Invariant encodings and their factors
//! are formed once for that tile's grid points and then reused for every row. Draw `k` is the
//! same index on every outcome. A failed draw stays at that index and is not dropped from the
//! full output or from a reduction. Output order is row-major in the caller's declared axis
//! order: the last axis varies fastest. [`Reduction::Full`] returns every row, draw, and grid
//! point in the tile. This module does not fit a model.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use antecedent_contract::rewrite_legality::{
    FrameLedger, PreservationAxis, Rewrite, check_executed, combine_ids,
};
use antecedent_core::ExecutionContext;
use antecedent_core::execution::{MapExecutionError, PipelineReceipt, PipelineStep};
use thiserror::Error;

/// Largest draw count whose integer value is exact as `f64` (`2^53`).
const EXACT_DRAW_COUNT: u64 = 1 << 53;

/// Largest working set, in bytes, a covariance tile may allocate: the draw-by-component
/// scratch plus the component-by-component matrix.
const MAX_COVARIANCE_BYTES: u64 = 1 << 30;

/// Axis of a row × draw × grid × outcome evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EvalAxis {
    /// Sample row, in the caller's declared row order.
    Row,
    /// Posterior or replicate draw. Index `k` is shared by every outcome.
    Draw,
    /// Dose grid point or horizon, in the caller's declared site order.
    Grid,
    /// Outcome whose coefficients share the draw index.
    Outcome,
}

impl EvalAxis {
    const fn index(self) -> usize {
        match self {
            Self::Row => 0,
            Self::Draw => 1,
            Self::Grid => 2,
            Self::Outcome => 3,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Row => "row",
            Self::Draw => "draw",
            Self::Grid => "grid",
            Self::Outcome => "outcome",
        }
    }
}

impl fmt::Display for EvalAxis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Caller-declared output order.
///
/// The last axis varies fastest (row-major). [`AxisOrder::row_major`] is row, draw, grid,
/// outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AxisOrder {
    axes: [EvalAxis; 4],
}

impl AxisOrder {
    /// Row, then draw, then grid, then outcome.
    #[must_use]
    pub const fn row_major() -> Self {
        Self { axes: [EvalAxis::Row, EvalAxis::Draw, EvalAxis::Grid, EvalAxis::Outcome] }
    }

    /// Accept `axes` when it lists each axis once.
    ///
    /// # Errors
    ///
    /// [`TileError::AxisOrder`] when an axis is repeated or missing.
    #[cfg(test)]
    #[must_use = "an axis order that repeats an axis is refused"]
    pub fn try_new(axes: [EvalAxis; 4]) -> Result<Self, TileError> {
        let mut seen = [false; 4];
        for axis in axes {
            let slot = axis.index();
            if seen[slot] {
                return Err(TileError::AxisOrder);
            }
            seen[slot] = true;
        }
        Ok(Self { axes })
    }

    /// Declared axes, first to last. The last varies fastest.
    #[must_use]
    pub const fn axes(self) -> [EvalAxis; 4] {
        self.axes
    }

    #[cfg(test)]
    fn without_draw(self) -> [EvalAxis; 3] {
        let mut kept = [EvalAxis::Row; 3];
        let mut written = 0;
        for axis in self.axes {
            if axis != EvalAxis::Draw {
                kept[written] = axis;
                written += 1;
            }
        }
        kept
    }

    /// Grid and outcome in their relative declared order. The first is the slower component.
    fn component_axes(self) -> [EvalAxis; 2] {
        let mut kept = [EvalAxis::Grid, EvalAxis::Outcome];
        let mut written = 0;
        for axis in self.axes {
            if axis == EvalAxis::Grid || axis == EvalAxis::Outcome {
                kept[written] = axis;
                written += 1;
            }
        }
        kept
    }
}

/// Invariant map from a grid coordinate or horizon to coefficients.
///
/// The encoding and its [`GridPoint::factor`] are formed once per grid point in the tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Encoding {
    /// `factor * [1, z, z², …, z^degree]` for coordinate `z`.
    Polynomial {
        /// Highest power. Width is `degree + 1`.
        degree: u64,
    },
    /// `factor` in the horizon column and zero elsewhere.
    ///
    /// The coordinate is the horizon index `0..width`. Exercised by this module's tests only:
    /// no estimator streams a horizon indicator.
    #[cfg(test)]
    HorizonIndicator {
        /// Number of horizon columns.
        width: u64,
    },
}

impl Encoding {
    /// Columns this encoding writes.
    ///
    /// # Errors
    ///
    /// A refusal when a horizon indicator has width zero (test-only encoding).
    /// [`TileError::SizeOverflow`] when `degree + 1` does not fit in `u64`.
    #[must_use = "a zero horizon width is refused"]
    pub const fn width(self) -> Result<u64, TileError> {
        match self {
            Self::Polynomial { degree } => match degree.checked_add(1) {
                Some(width) => Ok(width),
                None => Err(TileError::SizeOverflow),
            },
            #[cfg(test)]
            Self::HorizonIndicator { width } => {
                if width == 0 {
                    Err(TileError::HorizonWidth)
                } else {
                    Ok(width)
                }
            }
        }
    }
}

/// One dose or horizon in the caller's declared order.
///
/// Output follows slice order. [`Self::id`] is not sorted and need not be dense.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridPoint {
    /// Caller-declared identity emitted with every value at this site.
    pub id: u64,
    /// Dose coordinate (a horizon index for a horizon indicator encoding).
    pub coordinate: f64,
    /// Factor applied to the encoding. It does not scale row features.
    pub factor: f64,
}

/// Coefficient draws for one outcome.
///
/// Draw `k` occupies `[k * width, (k + 1) * width)`. Inside a draw the encoding coefficients
/// come first and the row-feature coefficients follow. `k` is the same index the other
/// outcomes use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutcomeDraws<'a> {
    /// Caller-declared outcome identity. Emitted in slice order.
    pub id: u64,
    /// Draw-major coefficients for every draw, not only the tile.
    pub coefficients: &'a [f64],
}

/// The one reduction a tile streams.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Reduction {
    /// Every row, draw, grid point, and outcome in the tile.
    Full,
    /// Arithmetic mean over every draw. A failed draw makes the mean non-finite and is not
    /// removed from the divisor. Exercised by this module's tests only: no estimator streams
    /// a draw mean.
    #[cfg(test)]
    DrawMean,
    /// Population covariance `Σ_k (y_k − mean)(y_k − mean)ᵀ / n` across the tile's grid points
    /// and outcomes.
    ///
    /// `n` is the draw count, failed draws included. A non-finite component makes every
    /// covariance entry that uses it non-finite. Component order follows the relative declared
    /// order of grid and outcome. One row-major matrix is emitted per row, rows in caller order.
    DrawCovariance,
}

/// What to evaluate. The tile does not fit coefficients.
#[derive(Clone, Copy, Debug)]
pub struct TileRequest<'a> {
    /// Row features in caller order, `row_width` values per row. Unused when `row_width` is zero.
    pub row_features: &'a [f64],
    /// First row index. Later rows follow in declaration order.
    pub row_start: u64,
    /// Rows in this tile.
    pub row_count: u64,
    /// Feature columns after the invariant encoding.
    pub row_width: u64,
    /// Outcomes in caller order. Every slice uses the same draw index.
    pub outcomes: &'a [OutcomeDraws<'a>],
    /// Joint failure mask, one flag per draw. `true` fails draw `k` for every outcome.
    pub failed_draws: &'a [bool],
    /// First draw index included in a [`Reduction::Full`] tile.
    pub draw_start: u64,
    /// Draws in this tile. Mean and covariance must cover every draw.
    pub draw_count: u64,
    /// Grid or horizon sites in caller order.
    pub grid: &'a [GridPoint],
    /// First site included in the tile.
    pub grid_start: u64,
    /// Sites in this tile.
    pub grid_count: u64,
    /// Encoding shared by every outcome and formed once per site in the tile.
    pub encoding: Encoding,
    /// Row-major axis order for [`Reduction::Full`].
    pub order: AxisOrder,
    /// Reduction this tile streams.
    pub reduction: Reduction,
}

/// One streamed value, in emission order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StreamedValue {
    /// One cell of [`Reduction::Full`].
    Full {
        /// Absolute row index (`row_start + local`).
        row: u64,
        /// Absolute draw index (`draw_start + local`).
        draw: u64,
        /// [`GridPoint::id`] in tile order.
        grid_id: u64,
        /// [`OutcomeDraws::id`] in slice order.
        outcome: u64,
        /// Linear prediction, or NaN when the draw is failed.
        value: f64,
    },
    /// One cell of a draw-mean reduction.
    #[cfg(test)]
    DrawMean {
        /// Absolute row index.
        row: u64,
        /// [`GridPoint::id`].
        grid_id: u64,
        /// [`OutcomeDraws::id`].
        outcome: u64,
        /// Mean over every draw, or NaN when any draw in that cell is non-finite.
        value: f64,
    },
    /// One entry of [`Reduction::DrawCovariance`].
    DrawCovariance {
        /// Absolute row index.
        row: u64,
        /// Grid id of the row of the covariance matrix.
        left_grid_id: u64,
        /// Outcome id of the row of the covariance matrix.
        left_outcome: u64,
        /// Grid id of the column of the covariance matrix.
        right_grid_id: u64,
        /// Outcome id of the column of the covariance matrix.
        right_outcome: u64,
        /// Population covariance of the aligned draws, or NaN when a paired draw is non-finite.
        value: f64,
    },
}

/// Counts from one streamed tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct StreamSummary {
    /// Reduction that was streamed. Read by this module's tests only.
    #[cfg(test)]
    pub reduction: Reduction,
    /// Values passed to the sink.
    pub emitted: u64,
    /// Grid points whose encoding and factor were formed.
    ///
    /// This equals the tile's grid count, not `rows * grid count`.
    pub invariant_evaluations: u64,
}

/// Refusal of a tile. Nothing is streamed when this is returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Error)]
#[non_exhaustive]
pub enum TileError {
    /// Worker admission could not fund this tile.
    #[error(transparent)]
    Admission(antecedent_core::execution::AdmissionError),
    /// An axis length was zero.
    #[error("tiled prediction refuses an empty {axis}; next attempt: declare at least one {axis}")]
    Empty {
        /// Axis that was empty.
        axis: &'static str,
    },
    /// The axis list was not a permutation of row, draw, grid, and outcome.
    #[cfg(test)]
    #[error(
        "tiled prediction refuses an axis order that repeats or omits an axis; next attempt: \
         list row, draw, grid, and outcome once each"
    )]
    AxisOrder,
    /// A product or width does not fit in `u64`.
    #[error(
        "tiled prediction refuses a size that overflows u64; next attempt: tile fewer rows, \
         draws, grid points, or a narrower encoding"
    )]
    SizeOverflow,
    /// An index does not fit in `usize`.
    #[error(
        "tiled prediction refuses index {value}, which does not fit usize; next attempt: tile \
         a smaller range"
    )]
    Index {
        /// Index that did not fit.
        value: u64,
    },
    /// The row-feature slice does not cover the requested rows.
    #[error(
        "tiled prediction needs {expected} row-feature values and found {actual}; next \
         attempt: pass row-major features through row_start + row_count"
    )]
    RowFeatures {
        /// Values required to cover the last requested row.
        expected: u64,
        /// Values present.
        actual: u64,
    },
    /// An outcome's coefficient slice does not match the shared draw count and width.
    #[error(
        "outcome {outcome} has {actual} coefficients; next attempt: supply {expected} \
         draw-major values, encoding width then row width"
    )]
    Coefficients {
        /// Outcome whose slice was the wrong length.
        outcome: u64,
        /// Required length.
        expected: u64,
        /// Actual length.
        actual: u64,
    },
    /// The draw window is outside the joint draw vector.
    #[error(
        "tiled prediction refuses draws {start}..{start}+{count} outside 0..{full}; next \
         attempt: choose a window inside the joint draw vector"
    )]
    DrawRange {
        /// First draw.
        start: u64,
        /// Window length.
        count: u64,
        /// Joint draw count.
        full: u64,
    },
    /// The grid window is outside the declared sites.
    #[error(
        "tiled prediction refuses grid points {start}..{start}+{count} outside 0..{full}; next \
         attempt: choose a window inside the declared grid"
    )]
    GridRange {
        /// First site.
        start: u64,
        /// Window length.
        count: u64,
        /// Declared site count.
        full: u64,
    },
    /// `row_start + row_count` overflowed.
    #[error(
        "tiled prediction refuses rows starting at {start} with count {count}; next attempt: \
         choose a row window that fits in u64"
    )]
    RowRange {
        /// First row.
        start: u64,
        /// Window length.
        count: u64,
    },
    /// A coordinate or factor was not finite.
    #[error(
        "tiled prediction refuses a non-finite coordinate or factor at grid id {grid_id}; next \
         attempt: supply a finite dose or horizon and a finite factor"
    )]
    NonFinitePoint {
        /// Site that was refused.
        grid_id: u64,
    },
    /// A horizon coordinate was not an index in `0..width`.
    #[cfg(test)]
    #[error(
        "tiled prediction refuses a horizon at grid id {grid_id} outside the indicator width; \
         next attempt: pass an integer horizon in 0..width"
    )]
    Horizon {
        /// Site that was refused.
        grid_id: u64,
    },
    /// A horizon indicator declared width zero.
    #[cfg(test)]
    #[error(
        "tiled prediction refuses a horizon indicator of width zero; next attempt: declare one \
         column per horizon"
    )]
    HorizonWidth,
    /// Mean or covariance was asked to see only some draws.
    #[error(
        "draw mean and draw covariance keep all {full} draws, including failed draws; next \
         attempt: cover draws 0..{full} or request the full output (this tile started at \
         {start} with count {count})"
    )]
    PartialDraws {
        /// Joint draw count.
        full: u64,
        /// Requested start.
        start: u64,
        /// Requested count.
        count: u64,
    },
    /// The covariance working set would exceed the allocation budget.
    #[error(
        "tiled prediction refuses a covariance tile needing {bytes} bytes, over the {limit} \
         byte budget; next attempt: tile fewer grid points or outcomes"
    )]
    CovarianceBudget {
        /// Bytes the tile would allocate.
        bytes: u64,
        /// Budget.
        limit: u64,
    },
    /// The draw count is not an exact `f64`, so a divisor would round.
    #[error(
        "tiled prediction refuses draw count {count}, which is not exact as f64; next attempt: \
         use at most 2^53 draws"
    )]
    DrawCountNotExact {
        /// Draw count that would have rounded.
        count: u64,
    },
    /// The values the tile streamed are not the cells the request declared.
    ///
    /// A cell was dropped, repeated, or emitted for another row, draw, grid
    /// point or outcome, or a failed draw was not kept as a failed draw.
    #[error(
        "tiled prediction streamed cells that alter the {axis}; next attempt: retain the \
         declared tile"
    )]
    Altered {
        /// Earliest axis the streamed cells alter.
        axis: PreservationAxis,
    },
    /// The execution context was cancelled between row chunks.
    #[error("tiled prediction was cancelled between row chunks")]
    Cancelled,
    /// The admission pool could not hold the live row chunks of a pipelined tile.
    #[error(
        "tiled prediction needs {bytes} bytes for its live row chunks; next attempt: use fewer \
         rows per chunk, a smaller window, or a larger pool"
    )]
    ChunkBudget {
        /// Bytes the live chunks would hold.
        bytes: u64,
    },
}

/// Stream [`TileRequest::reduction`] in declared order.
///
/// Invariant encodings and factors for the tile's grid points are formed before any row is
/// read. The sink sees only the requested reduction. A refusal emits nothing.
///
/// # Errors
///
/// [`TileError`] when the window, lengths, encoding, axis order, or draw policy is refused.
#[must_use = "a refused tile is not an empty reduction"]
pub fn stream_tile(
    request: &TileRequest<'_>,
    mut emit: impl FnMut(StreamedValue),
) -> Result<StreamSummary, TileError> {
    let frame = Frame::prepare(request)?;
    let hoisted = hoist(&frame)?;
    let mut audit = TileAudit::new(&frame)?;
    let mut audited = |value: StreamedValue| {
        audit.see(value);
        emit(value);
    };
    let emitted = match frame.reduction {
        Reduction::Full => stream_full(&frame, &hoisted, &mut audited)?,
        #[cfg(test)]
        Reduction::DrawMean => stream_mean(&frame, &hoisted, &mut audited)?,
        Reduction::DrawCovariance => stream_covariance(&frame, &hoisted, &mut audited)?,
    };
    audit.admit(&frame)?;
    Ok(StreamSummary {
        #[cfg(test)]
        reduction: frame.reduction,
        emitted,
        invariant_evaluations: hoisted.evaluations,
    })
}

/// A pipelined tile: the stream's summary and what the pipeline kept alive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelinedTile {
    /// Counts of the whole tile; `invariant_evaluations` is per row chunk, as each chunk
    /// forms its own encodings.
    pub summary: StreamSummary,
    /// Row chunks consumed, the most alive at once, and the producers used. Read by this
    /// module's tests only.
    #[cfg(test)]
    pub receipt: PipelineReceipt,
}

impl PipelinedTile {
    fn new(summary: StreamSummary, receipt: PipelineReceipt) -> Self {
        #[cfg(not(test))]
        let _ = receipt;
        Self {
            summary,
            #[cfg(test)]
            receipt,
        }
    }
}

/// [`stream_tile`] with row chunks produced by leased workers ahead of the sink.
///
/// Rows are independent, so the tile is cut into chunks of `rows_per_chunk` rows; workers
/// evaluate chunks while `emit` receives them in declared order on the caller. At most
/// `window` chunks are alive at once, so a sink that stalls stops the producers instead of
/// letting evaluated cells pile up. The bytes of `window` chunks are leased from the context's
/// admission pool first. The emitted values equal [`stream_tile`]'s, bit for bit.
///
/// A tile whose axis order does not put the row axis first for its reduction is not
/// row-separable in emission order; it is streamed serially as one chunk.
///
/// # Errors
///
/// The refusals of [`stream_tile`] (each chunk is validated and audited as a tile),
/// [`TileError::ChunkBudget`], and [`TileError::Cancelled`].
pub fn stream_tile_pipelined(
    request: &TileRequest<'_>,
    ctx: &ExecutionContext,
    rows_per_chunk: u64,
    window: usize,
    mut emit: impl FnMut(StreamedValue),
) -> Result<PipelinedTile, TileError> {
    let rows_first = match request.reduction {
        Reduction::Full => request.order.axes()[0] == EvalAxis::Row,
        #[cfg(test)]
        Reduction::DrawMean => request.order.without_draw()[0] == EvalAxis::Row,
        Reduction::DrawCovariance => true,
    };
    let rows_per_chunk = rows_per_chunk.max(1);
    if !rows_first || request.row_count <= rows_per_chunk {
        let summary = stream_tile(request, emit)?;
        let receipt = PipelineReceipt { items_consumed: 1, peak_live: 1, workers: 1 };
        return Ok(PipelinedTile::new(summary, receipt));
    }
    // Validate the whole request once so a refusal emits nothing.
    Frame::prepare(request)?;
    let outcomes = len_u64(request.outcomes)?;
    let cells_per_row = match request.reduction {
        Reduction::Full => product(&[request.draw_count, request.grid_count, outcomes])?,
        #[cfg(test)]
        Reduction::DrawMean => product(&[request.grid_count, outcomes])?,
        Reduction::DrawCovariance => {
            let component = product(&[request.grid_count, outcomes])?;
            product(&[component, component])?
        }
    };
    let chunk_bytes = product(&[
        rows_per_chunk,
        cells_per_row,
        size_of::<StreamedValue>() as u64,
        window.max(1) as u64,
    ])?;
    let _held = ctx
        .acquire_admission(0, chunk_bytes)
        .map_err(|_| TileError::ChunkBudget { bytes: chunk_bytes })?;
    let chunks = request.row_count.div_ceil(rows_per_chunk);
    let chunks = as_usize(chunks)?;
    let mut emitted = 0u64;
    let mut invariant = 0u64;
    let piped = ctx.pipeline_indexed(
        chunks,
        window,
        |index, _| {
            let offset = index as u64 * rows_per_chunk;
            let sub = TileRequest {
                row_start: request.row_start + offset,
                row_count: rows_per_chunk.min(request.row_count - offset),
                ..*request
            };
            let mut values = Vec::new();
            let summary = stream_tile(&sub, |value| values.push(value))?;
            Ok::<_, TileError>((values, summary))
        },
        |_, (values, summary)| {
            emitted += summary.emitted;
            invariant = summary.invariant_evaluations;
            for value in values {
                emit(value);
            }
            Ok(PipelineStep::Continue)
        },
    );
    let receipt = match piped {
        Ok(receipt) => receipt,
        Err(MapExecutionError::Work(error)) => return Err(error),
        Err(MapExecutionError::Admission(
            antecedent_core::execution::AdmissionError::Cancelled,
        )) => return Err(TileError::Cancelled),
        Err(MapExecutionError::Admission(error)) => return Err(TileError::Admission(error)),
    };
    let summary = StreamSummary {
        #[cfg(test)]
        reduction: request.reduction,
        emitted,
        invariant_evaluations: invariant,
    };
    Ok(PipelinedTile::new(summary, receipt))
}

/// Cells the tile actually emitted, checked against the cells its request declares.
///
/// The declared side is enumerated by plain nested loops over the request, not by the
/// streaming walk, so a walk that skips, repeats or mislabels a cell does not agree with it.
struct TileAudit<'a> {
    failed: &'a [bool],
    /// Whether any draw in the tile's window is failed.
    #[cfg(test)]
    any_failed: bool,
    executed: FrameLedger,
}

/// State of a streamed cell: kept as declared, or a failed draw that was not kept failed.
const STATE_KEPT: u64 = 0;
const STATE_FAILED_KEPT: u64 = 2;
const STATE_FAILED_LOST: u64 = 3;

impl<'a> TileAudit<'a> {
    fn new(frame: &Frame<'a>) -> Result<Self, TileError> {
        let window = as_usize(frame.draw_start)?
            ..as_usize(checked_add(frame.draw_start, frame.draw_count)?)?;
        #[cfg(test)]
        let any_failed = frame.failed.get(window).is_some_and(|draws| draws.contains(&true));
        #[cfg(not(test))]
        let _ = window;
        Ok(Self {
            failed: frame.failed,
            #[cfg(test)]
            any_failed,
            executed: tile_ledger()?,
        })
    }

    fn see(&mut self, value: StreamedValue) {
        match value {
            StreamedValue::Full { row, draw, grid_id, outcome, value } => {
                let failed = usize::try_from(draw)
                    .ok()
                    .and_then(|index| self.failed.get(index))
                    .copied()
                    .unwrap_or(false);
                let state = match (failed, value.is_nan()) {
                    (false, _) => STATE_KEPT,
                    (true, true) => STATE_FAILED_KEPT,
                    (true, false) => STATE_FAILED_LOST,
                };
                self.executed.record_numeric(
                    combine_ids(&[row, grid_id, outcome]),
                    draw,
                    draw,
                    state,
                );
            }
            #[cfg(test)]
            StreamedValue::DrawMean { row, grid_id, outcome, value } => {
                let state = match (self.any_failed, value.is_nan()) {
                    (false, _) => STATE_KEPT,
                    (true, true) => STATE_FAILED_KEPT,
                    (true, false) => STATE_FAILED_LOST,
                };
                self.executed.record_numeric(combine_ids(&[row, grid_id, outcome]), 0, 0, state);
            }
            StreamedValue::DrawCovariance {
                row,
                left_grid_id,
                left_outcome,
                right_grid_id,
                right_outcome,
                ..
            } => {
                self.executed.record_numeric(
                    combine_ids(&[row, left_grid_id, left_outcome, right_grid_id, right_outcome]),
                    0,
                    0,
                    STATE_KEPT,
                );
            }
        }
    }

    fn admit(&self, frame: &Frame<'_>) -> Result<(), TileError> {
        let mut declared = tile_ledger()?;
        let grid_ids: Vec<u64> =
            (0..frame.grid_count).map(|local| grid_id(frame, local)).collect::<Result<_, _>>()?;
        let outcome_ids: Vec<u64> = frame.outcomes.iter().map(|outcome| outcome.id).collect();
        let draws = frame.draw_start..frame.draw_start + frame.draw_count;
        for row in frame.row_start..frame.row_start + frame.row_count {
            match frame.reduction {
                Reduction::Full => {
                    for draw in draws.clone() {
                        let failed = frame.failed[as_usize(draw)?];
                        let state = if failed { STATE_FAILED_KEPT } else { STATE_KEPT };
                        for &grid in &grid_ids {
                            for &outcome in &outcome_ids {
                                declared.record_numeric(
                                    combine_ids(&[row, grid, outcome]),
                                    draw,
                                    draw,
                                    state,
                                );
                            }
                        }
                    }
                }
                #[cfg(test)]
                Reduction::DrawMean => {
                    let state = if self.any_failed { STATE_FAILED_KEPT } else { STATE_KEPT };
                    for &outcome in &outcome_ids {
                        for &grid in &grid_ids {
                            declared.record_numeric(
                                combine_ids(&[row, grid, outcome]),
                                0,
                                0,
                                state,
                            );
                        }
                    }
                }
                Reduction::DrawCovariance => {
                    for &left_grid in &grid_ids {
                        for &left_outcome in &outcome_ids {
                            for &right_grid in &grid_ids {
                                for &right_outcome in &outcome_ids {
                                    declared.record_numeric(
                                        combine_ids(&[
                                            row,
                                            left_grid,
                                            left_outcome,
                                            right_grid,
                                            right_outcome,
                                        ]),
                                        0,
                                        0,
                                        STATE_KEPT,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        check_executed(
            Rewrite::TILED_EVALUATION,
            &declared.fingerprint(),
            &[self.executed.fingerprint()],
        )
        .map_err(|refusal| TileError::Altered { axis: refusal.axis() })
    }
}

fn tile_ledger() -> Result<FrameLedger, TileError> {
    FrameLedger::try_new("declared_tile", "static_cross_section", "joint_draws_failures_kept")
        .map_err(|refusal| TileError::Altered { axis: refusal.axis() })
}

struct Frame<'a> {
    features: &'a [f64],
    row_start: u64,
    row_count: u64,
    row_width: usize,
    outcomes: &'a [OutcomeDraws<'a>],
    failed: &'a [bool],
    draw_start: u64,
    draw_count: u64,
    grid: &'a [GridPoint],
    grid_start: usize,
    grid_count: u64,
    encoding: Encoding,
    enc_width: usize,
    order: AxisOrder,
    reduction: Reduction,
    /// Whether every coefficient of draw `draw_start + k` is finite, for outcome `o` at
    /// `o * draw_count + k`. Checked once here so `predict` does not rescan.
    finite_coefficients: Vec<bool>,
}

impl<'a> Frame<'a> {
    #[allow(clippy::too_many_lines, reason = "one validation pass over the request")]
    fn prepare(request: &'a TileRequest<'a>) -> Result<Self, TileError> {
        if request.outcomes.is_empty() {
            return Err(TileError::Empty { axis: EvalAxis::Outcome.as_str() });
        }
        if request.row_count == 0 {
            return Err(TileError::Empty { axis: EvalAxis::Row.as_str() });
        }
        if request.draw_count == 0 {
            return Err(TileError::Empty { axis: EvalAxis::Draw.as_str() });
        }
        if request.grid_count == 0 {
            return Err(TileError::Empty { axis: EvalAxis::Grid.as_str() });
        }
        let enc_width_u64 = request.encoding.width()?;
        let row_width = as_usize(request.row_width)?;
        let enc_width = as_usize(enc_width_u64)?;
        let coef_width = checked_add(enc_width_u64, request.row_width)?;
        let full_draws = len_u64(request.failed_draws)?;
        if full_draws == 0 {
            return Err(TileError::Empty { axis: EvalAxis::Draw.as_str() });
        }
        let expected = product(&[full_draws, coef_width])?;
        for outcome in request.outcomes {
            let actual = len_u64(outcome.coefficients)?;
            if actual != expected {
                return Err(TileError::Coefficients { outcome: outcome.id, expected, actual });
            }
        }
        let draw_end = checked_add(request.draw_start, request.draw_count)?;
        if draw_end > full_draws {
            return Err(TileError::DrawRange {
                start: request.draw_start,
                count: request.draw_count,
                full: full_draws,
            });
        }
        if !matches!(request.reduction, Reduction::Full)
            && (request.draw_start != 0 || request.draw_count != full_draws)
        {
            return Err(TileError::PartialDraws {
                full: full_draws,
                start: request.draw_start,
                count: request.draw_count,
            });
        }
        if !matches!(request.reduction, Reduction::Full) && request.draw_count > EXACT_DRAW_COUNT {
            return Err(TileError::DrawCountNotExact { count: request.draw_count });
        }
        let row_end = checked_add(request.row_start, request.row_count).map_err(|_| {
            TileError::RowRange { start: request.row_start, count: request.row_count }
        })?;
        if request.row_width > 0 {
            let expected_features = product(&[row_end, request.row_width])?;
            let actual = len_u64(request.row_features)?;
            if actual < expected_features {
                return Err(TileError::RowFeatures { expected: expected_features, actual });
            }
        }
        let grid_full = len_u64(request.grid)?;
        let grid_end = checked_add(request.grid_start, request.grid_count)?;
        if grid_end > grid_full {
            return Err(TileError::GridRange {
                start: request.grid_start,
                count: request.grid_count,
                full: grid_full,
            });
        }
        let window = as_usize(request.draw_count)?;
        let first = as_usize(request.draw_start)?;
        let width = as_usize(coef_width)?;
        let mut finite_coefficients =
            Vec::with_capacity(window.saturating_mul(request.outcomes.len()));
        for outcome in request.outcomes {
            for offset in 0..window {
                let start = (first + offset) * width;
                let draw = outcome
                    .coefficients
                    .get(start..start + width)
                    .ok_or(TileError::Index { value: request.draw_start + offset as u64 })?;
                finite_coefficients.push(draw.iter().all(|coef| coef.is_finite()));
            }
        }
        Ok(Self {
            features: request.row_features,
            row_start: request.row_start,
            row_count: request.row_count,
            row_width,
            outcomes: request.outcomes,
            failed: request.failed_draws,
            draw_start: request.draw_start,
            draw_count: request.draw_count,
            grid: request.grid,
            grid_start: as_usize(request.grid_start)?,
            grid_count: request.grid_count,
            encoding: request.encoding,
            enc_width,
            order: request.order,
            reduction: request.reduction,
            finite_coefficients,
        })
    }
}

struct Hoisted {
    values: Vec<f64>,
    width: usize,
    evaluations: u64,
}

impl Hoisted {
    fn factors(&self, grid_local: usize) -> &[f64] {
        let start = grid_local * self.width;
        &self.values[start..start + self.width]
    }
}

fn hoist(frame: &Frame<'_>) -> Result<Hoisted, TileError> {
    let grid_count = as_usize(frame.grid_count)?;
    let cells = grid_count.checked_mul(frame.enc_width).ok_or(TileError::SizeOverflow)?;
    let mut values = vec![0.0; cells];
    let end = frame.grid_start.checked_add(grid_count).ok_or(TileError::SizeOverflow)?;
    let sites = frame.grid.get(frame.grid_start..end).ok_or(TileError::SizeOverflow)?;
    for (local, point) in sites.iter().enumerate() {
        let start = local * frame.enc_width;
        let stop = start + frame.enc_width;
        write_encoding(&mut values[start..stop], point, frame.encoding)?;
    }
    Ok(Hoisted { values, width: frame.enc_width, evaluations: frame.grid_count })
}

fn write_encoding(out: &mut [f64], point: &GridPoint, encoding: Encoding) -> Result<(), TileError> {
    if !point.coordinate.is_finite() || !point.factor.is_finite() {
        return Err(TileError::NonFinitePoint { grid_id: point.id });
    }
    match encoding {
        Encoding::Polynomial { .. } => {
            let mut power = 1.0;
            for slot in out.iter_mut() {
                *slot = point.factor * power;
                power *= point.coordinate;
            }
        }
        #[cfg(test)]
        Encoding::HorizonIndicator { width } => {
            let horizon = horizon_index(point, width)?;
            out[as_usize(horizon)?] = point.factor;
        }
    }
    Ok(())
}

#[cfg(test)]
fn horizon_index(point: &GridPoint, width: u64) -> Result<u64, TileError> {
    if point.coordinate < 0.0 {
        return Err(TileError::Horizon { grid_id: point.id });
    }
    let mut horizon = 0;
    while horizon < width {
        // Numeric equality, so `-0.0` is horizon zero like `0.0`.
        if point.coordinate.partial_cmp(&(horizon as f64)) == Some(std::cmp::Ordering::Equal) {
            return Ok(horizon);
        }
        horizon += 1;
    }
    Err(TileError::Horizon { grid_id: point.id })
}

fn stream_full(
    frame: &Frame<'_>,
    hoisted: &Hoisted,
    emit: &mut impl FnMut(StreamedValue),
) -> Result<u64, TileError> {
    let limits = [frame.row_count, frame.draw_count, frame.grid_count, len_u64(frame.outcomes)?];
    let order = frame.order.axes();
    let mut emitted = 0;
    walk(&order, &limits, |local| {
        let row = frame.row_start + local[EvalAxis::Row.index()];
        let draw = frame.draw_start + local[EvalAxis::Draw.index()];
        let grid_local = local[EvalAxis::Grid.index()];
        let outcome_local = local[EvalAxis::Outcome.index()];
        let value = predict(frame, hoisted, row, draw, grid_local, outcome_local)?;
        emit(StreamedValue::Full {
            row,
            draw,
            grid_id: grid_id(frame, grid_local)?,
            outcome: frame.outcomes[as_usize(outcome_local)?].id,
            value,
        });
        emitted = checked_add(emitted, 1)?;
        Ok(())
    })?;
    Ok(emitted)
}

#[cfg(test)]
fn stream_mean(
    frame: &Frame<'_>,
    hoisted: &Hoisted,
    emit: &mut impl FnMut(StreamedValue),
) -> Result<u64, TileError> {
    let outcomes = len_u64(frame.outcomes)?;
    let order = frame.order.without_draw();
    let extents = [frame.row_count, frame.draw_count, frame.grid_count, outcomes];
    let divisor = frame.draw_count as f64;
    let mut emitted = 0;
    walk(&order, &extents, |local| {
        let row = frame.row_start + local[EvalAxis::Row.index()];
        let grid_local = local[EvalAxis::Grid.index()];
        let outcome_local = local[EvalAxis::Outcome.index()];
        let mut sum = 0.0;
        let mut finite = true;
        for offset in 0..frame.draw_count {
            let draw = frame.draw_start + offset;
            let value = predict(frame, hoisted, row, draw, grid_local, outcome_local)?;
            if finite && value.is_finite() {
                sum += value;
            } else {
                finite = false;
            }
        }
        emit(StreamedValue::DrawMean {
            row,
            grid_id: grid_id(frame, grid_local)?,
            outcome: frame.outcomes[as_usize(outcome_local)?].id,
            value: if finite { sum / divisor } else { f64::NAN },
        });
        emitted = checked_add(emitted, 1)?;
        Ok(())
    })?;
    Ok(emitted)
}

fn stream_covariance(
    frame: &Frame<'_>,
    hoisted: &Hoisted,
    emit: &mut impl FnMut(StreamedValue),
) -> Result<u64, TileError> {
    let n_outcome = len_u64(frame.outcomes)?;
    let component_n = product(&[frame.grid_count, n_outcome])?;
    let draws = frame.draw_count;
    let scratch_cells = product(&[draws, component_n])?;
    let matrix_cells = product(&[component_n, component_n])?;
    let bytes = product(&[checked_add(scratch_cells, matrix_cells)?, 8])?;
    if bytes > MAX_COVARIANCE_BYTES {
        return Err(TileError::CovarianceBudget { bytes, limit: MAX_COVARIANCE_BYTES });
    }
    let mut scratch = vec![0.0; as_usize(scratch_cells)?];
    let mut matrix = vec![0.0; as_usize(matrix_cells)?];
    let components = frame.order.component_axes();
    let divisor = draws as f64;
    let mut emitted = 0;
    for row_offset in 0..frame.row_count {
        let row = frame.row_start + row_offset;
        fill_row(frame, hoisted, row, components, component_n, &mut scratch)?;
        let component_us = as_usize(component_n)?;
        let draw_us = as_usize(draws)?;
        // The matrix is symmetric: compute the upper triangle and mirror it.
        for left in 0..component_us {
            for right in left..component_us {
                let value =
                    population_covariance(&scratch, draw_us, component_us, left, right, divisor);
                matrix[left * component_us + right] = value;
                matrix[right * component_us + left] = value;
            }
        }
        for left in 0..component_n {
            for right in 0..component_n {
                let value = matrix[as_usize(left)? * component_us + as_usize(right)?];
                let (left_grid, left_outcome) =
                    component_ids(frame, components, left, frame.grid_count, n_outcome)?;
                let (right_grid, right_outcome) =
                    component_ids(frame, components, right, frame.grid_count, n_outcome)?;
                emit(StreamedValue::DrawCovariance {
                    row,
                    left_grid_id: left_grid,
                    left_outcome,
                    right_grid_id: right_grid,
                    right_outcome,
                    value,
                });
                emitted = checked_add(emitted, 1)?;
            }
        }
    }
    Ok(emitted)
}

fn fill_row(
    frame: &Frame<'_>,
    hoisted: &Hoisted,
    row: u64,
    components: [EvalAxis; 2],
    component_n: u64,
    scratch: &mut [f64],
) -> Result<(), TileError> {
    let n_outcome = len_u64(frame.outcomes)?;
    for draw_offset in 0..frame.draw_count {
        let draw = frame.draw_start + draw_offset;
        for grid_local in 0..frame.grid_count {
            for outcome_local in 0..n_outcome {
                let component = component_index(
                    components,
                    grid_local,
                    outcome_local,
                    frame.grid_count,
                    n_outcome,
                )?;
                let value = predict(frame, hoisted, row, draw, grid_local, outcome_local)?;
                let slot = checked_add(product(&[draw_offset, component_n])?, component)?;
                scratch[as_usize(slot)?] = value;
            }
        }
    }
    Ok(())
}

fn population_covariance(
    scratch: &[f64],
    draws: usize,
    components: usize,
    left: usize,
    right: usize,
    divisor: f64,
) -> f64 {
    let mut left_sum = 0.0;
    let mut right_sum = 0.0;
    for draw in 0..draws {
        let left_value = scratch[draw * components + left];
        let right_value = scratch[draw * components + right];
        if !left_value.is_finite() || !right_value.is_finite() {
            return f64::NAN;
        }
        left_sum += left_value;
        right_sum += right_value;
    }
    let left_mean = left_sum / divisor;
    let right_mean = right_sum / divisor;
    let mut acc = 0.0;
    for draw in 0..draws {
        let left_value = scratch[draw * components + left];
        let right_value = scratch[draw * components + right];
        acc += (left_value - left_mean) * (right_value - right_mean);
    }
    acc / divisor
}

fn predict(
    frame: &Frame<'_>,
    hoisted: &Hoisted,
    row: u64,
    draw: u64,
    grid_local: u64,
    outcome_local: u64,
) -> Result<f64, TileError> {
    let draw_index = as_usize(draw)?;
    if frame.failed[draw_index] {
        return Ok(f64::NAN);
    }
    let outcome = as_usize(outcome_local)?;
    let window = as_usize(frame.draw_count)?;
    let draw_offset = draw_index - as_usize(frame.draw_start)?;
    if !frame.finite_coefficients[outcome * window + draw_offset] {
        return Ok(f64::NAN);
    }
    let width = frame.enc_width.checked_add(frame.row_width).ok_or(TileError::SizeOverflow)?;
    let start = draw_index.checked_mul(width).ok_or(TileError::SizeOverflow)?;
    let stop = start.checked_add(width).ok_or(TileError::SizeOverflow)?;
    let coefficients = frame.outcomes[outcome]
        .coefficients
        .get(start..stop)
        .ok_or(TileError::Index { value: draw })?;
    let factors = hoisted.factors(as_usize(grid_local)?);
    let mut value = 0.0;
    for (factor, coef) in factors.iter().zip(coefficients) {
        value += factor * coef;
    }
    let features = row_features(frame, row)?;
    for (feature, coef) in features.iter().zip(&coefficients[frame.enc_width..]) {
        value += feature * coef;
    }
    Ok(value)
}

fn row_features<'a>(frame: &Frame<'a>, row: u64) -> Result<&'a [f64], TileError> {
    if frame.row_width == 0 {
        return Ok(&[]);
    }
    let row_index = as_usize(row)?;
    let start = row_index.checked_mul(frame.row_width).ok_or(TileError::SizeOverflow)?;
    let stop = start.checked_add(frame.row_width).ok_or(TileError::SizeOverflow)?;
    frame.features.get(start..stop).ok_or(TileError::Index { value: row })
}

fn grid_id(frame: &Frame<'_>, grid_local: u64) -> Result<u64, TileError> {
    let index =
        frame.grid_start.checked_add(as_usize(grid_local)?).ok_or(TileError::SizeOverflow)?;
    frame.grid.get(index).map(|point| point.id).ok_or(TileError::Index { value: grid_local })
}

fn component_index(
    axes: [EvalAxis; 2],
    grid_local: u64,
    outcome_local: u64,
    n_grid: u64,
    n_outcome: u64,
) -> Result<u64, TileError> {
    let (slow, fast, fast_extent) = if axes[0] == EvalAxis::Outcome {
        (outcome_local, grid_local, n_grid)
    } else {
        (grid_local, outcome_local, n_outcome)
    };
    checked_add(product(&[slow, fast_extent])?, fast)
}

fn component_ids(
    frame: &Frame<'_>,
    axes: [EvalAxis; 2],
    index: u64,
    n_grid: u64,
    n_outcome: u64,
) -> Result<(u64, u64), TileError> {
    let outcome_slow = axes[0] == EvalAxis::Outcome;
    let fast_extent = if outcome_slow { n_grid } else { n_outcome };
    let slow = index / fast_extent;
    let fast = index % fast_extent;
    let (grid_local, outcome_local) = if outcome_slow { (fast, slow) } else { (slow, fast) };
    let outcome = frame.outcomes[as_usize(outcome_local)?].id;
    Ok((grid_id(frame, grid_local)?, outcome))
}

fn walk(
    order: &[EvalAxis],
    extent_by_axis: &[u64; 4],
    mut visit: impl FnMut([u64; 4]) -> Result<(), TileError>,
) -> Result<(), TileError> {
    let mut limits = [0; 4];
    for (place, axis) in order.iter().enumerate() {
        limits[place] = extent_by_axis[axis.index()];
    }
    let total = product(&limits[..order.len()])?;
    let mut index = [0; 4];
    for _ in 0..total {
        let mut local = [0; 4];
        for (place, axis) in order.iter().enumerate() {
            local[axis.index()] = index[place];
        }
        visit(local)?;
        bump(&mut index[..order.len()], &limits[..order.len()]);
    }
    Ok(())
}

fn bump(index: &mut [u64], limits: &[u64]) {
    for place in (0..index.len()).rev() {
        index[place] += 1;
        if index[place] < limits[place] {
            return;
        }
        index[place] = 0;
    }
}

fn product(values: &[u64]) -> Result<u64, TileError> {
    let mut acc = 1_u64;
    for value in values {
        acc = acc.checked_mul(*value).ok_or(TileError::SizeOverflow)?;
    }
    Ok(acc)
}

fn checked_add(left: u64, right: u64) -> Result<u64, TileError> {
    left.checked_add(right).ok_or(TileError::SizeOverflow)
}

fn len_u64<T>(values: &[T]) -> Result<u64, TileError> {
    u64::try_from(values.len()).map_err(|_| TileError::SizeOverflow)
}

fn as_usize(value: u64) -> Result<usize, TileError> {
    usize::try_from(value).map_err(|_| TileError::Index { value })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(value: f64) -> u64 {
        value.to_bits()
    }

    fn point(id: u64, coordinate: f64, factor: f64) -> GridPoint {
        GridPoint { id, coordinate, factor }
    }

    fn request<'a>(
        features: &'a [f64],
        outcomes: &'a [OutcomeDraws<'a>],
        failed: &'a [bool],
        grid: &'a [GridPoint],
        encoding: Encoding,
        order: AxisOrder,
        reduction: Reduction,
    ) -> TileRequest<'a> {
        TileRequest {
            row_features: features,
            row_start: 0,
            row_count: 1,
            row_width: 0,
            outcomes,
            failed_draws: failed,
            draw_start: 0,
            draw_count: u64::try_from(failed.len()).unwrap(),
            grid,
            grid_start: 0,
            grid_count: u64::try_from(grid.len()).unwrap(),
            encoding,
            order,
            reduction,
        }
    }

    fn collect(
        request: &TileRequest<'_>,
    ) -> Result<(Vec<StreamedValue>, StreamSummary), TileError> {
        let mut values = Vec::new();
        let summary = stream_tile(request, |value| values.push(value))?;
        Ok((values, summary))
    }

    #[test]
    fn invariant_encodings_and_factors_are_computed_once_per_tile() {
        let features = [0.0, 1.0, 2.0, 3.0];
        let coefficients = [3.0, 1.0];
        let outcomes = [OutcomeDraws { id: 7, coefficients: &coefficients }];
        let failed = [false];
        let grid = [point(0, 0.0, 2.0), point(1, 4.0, 2.0)];
        let mut request = request(
            &features,
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            AxisOrder::row_major(),
            Reduction::Full,
        );
        request.row_count = 4;
        request.row_width = 1;
        let (values, summary) = collect(&request).unwrap();
        assert_eq!(summary.invariant_evaluations, 2);
        assert_eq!(values.len(), 8);
        // y = factor * beta_enc + feature * beta_row = 2 * 3 + feature.
        let expected = [6.0, 7.0, 8.0, 9.0];
        for (row, expected) in expected.iter().enumerate() {
            let cell = values[row * 2];
            match cell {
                StreamedValue::Full { value, grid_id, .. } => {
                    assert_eq!(grid_id, 0);
                    assert_eq!(bits(value), bits(*expected));
                }
                other => panic!("full cell, got {other:?}"),
            }
        }
    }

    #[test]
    fn joint_draw_index_aligns_outcomes() {
        // Draw 0 is (A=1, B=10), draw 1 is (A=3, B=-2). Pairing draw 0 of A with draw 1 of B
        // would flip the covariance sign.
        let outcomes = [
            OutcomeDraws { id: 1, coefficients: &[1.0, 3.0] },
            OutcomeDraws { id: 2, coefficients: &[10.0, -2.0] },
        ];
        let failed = [false, false];
        let grid = [point(4, 0.0, 1.0)];
        let request = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            AxisOrder::row_major(),
            Reduction::DrawCovariance,
        );
        let (values, _) = collect(&request).unwrap();
        let cross = values.iter().find_map(|value| match value {
            StreamedValue::DrawCovariance { left_outcome: 1, right_outcome: 2, value, .. } => {
                Some(*value)
            }
            _ => None,
        });
        assert_eq!(bits(cross.unwrap()), bits(-6.0));
    }

    #[test]
    fn failed_draw_stays_in_place_and_is_not_dropped() {
        let outcomes = [OutcomeDraws { id: 1, coefficients: &[1.0, 99.0, 3.0] }];
        let failed = [false, true, false];
        let grid = [point(0, 0.0, 1.0)];
        let full = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            AxisOrder::row_major(),
            Reduction::Full,
        );
        let (values, summary) = collect(&full).unwrap();
        assert_eq!(summary.emitted, 3);
        let cell = values[1];
        match cell {
            StreamedValue::Full { draw: 1, value, .. } => assert!(value.is_nan()),
            other => panic!("failed slot, got {other:?}"),
        }
        let cell = values[0];
        match cell {
            StreamedValue::Full { draw: 0, value, .. } => assert_eq!(bits(value), bits(1.0)),
            other => panic!("first draw, got {other:?}"),
        }
        let mut mean_request = full;
        mean_request.reduction = Reduction::DrawMean;
        let (means, _) = collect(&mean_request).unwrap();
        let cell = means[0];
        match cell {
            StreamedValue::DrawMean { value, .. } => assert!(value.is_nan()),
            other => panic!("mean, got {other:?}"),
        }
        let mut covariance = full;
        covariance.reduction = Reduction::DrawCovariance;
        let (covariances, _) = collect(&covariance).unwrap();
        let cell = covariances[0];
        match cell {
            StreamedValue::DrawCovariance { value, .. } => assert!(value.is_nan()),
            other => panic!("covariance, got {other:?}"),
        }
    }

    #[test]
    fn per_outcome_non_finite_coefficient_stays_at_its_draw() {
        let outcomes = [
            OutcomeDraws { id: 1, coefficients: &[1.0, f64::NAN] },
            OutcomeDraws { id: 2, coefficients: &[4.0, 8.0] },
        ];
        let failed = [false, false];
        let grid = [point(0, 0.0, 1.0)];
        let request = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            AxisOrder::row_major(),
            Reduction::Full,
        );
        let (values, _) = collect(&request).unwrap();
        assert_eq!(values.len(), 4);
        let cell = values[1];
        match cell {
            StreamedValue::Full { draw: 0, outcome: 2, value, .. } => {
                assert_eq!(bits(value), bits(4.0));
            }
            other => panic!("aligned partner, got {other:?}"),
        }
        let cell = values[3];
        match cell {
            StreamedValue::Full { draw: 1, outcome: 2, value, .. } => {
                assert_eq!(bits(value), bits(8.0));
            }
            other => panic!("second partner, got {other:?}"),
        }
        let cell = values[2];
        match cell {
            StreamedValue::Full { draw: 1, outcome: 1, value, .. } => assert!(value.is_nan()),
            other => panic!("failed outcome draw, got {other:?}"),
        }
    }

    #[test]
    fn output_is_row_major_in_declared_axis_order() {
        // y = 10 * draw + 100 * coordinate + feature, one outcome.
        // Order grid, row, draw, outcome: grid slowest, draw fastest.
        let features = [0.0, 1.0];
        let coefficients = [
            0.0, 100.0, 1.0, // draw 0
            10.0, 100.0, 1.0, // draw 1
        ];
        let outcomes = [OutcomeDraws { id: 9, coefficients: &coefficients }];
        let failed = [false, false];
        let grid = [point(0, 0.0, 1.0), point(1, 1.0, 1.0)];
        let order =
            AxisOrder::try_new([EvalAxis::Grid, EvalAxis::Row, EvalAxis::Draw, EvalAxis::Outcome])
                .unwrap();
        let mut request = request(
            &features,
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 1 },
            order,
            Reduction::Full,
        );
        request.row_count = 2;
        request.row_width = 1;
        let (values, _) = collect(&request).unwrap();
        let got: Vec<(u64, u64, u64, f64)> = values
            .iter()
            .map(|value| match value {
                StreamedValue::Full { grid_id, row, draw, value, .. } => {
                    (*grid_id, *row, *draw, *value)
                }
                other => panic!("full, got {other:?}"),
            })
            .collect();
        let expected = [
            (0, 0, 0, 0.0),
            (0, 0, 1, 10.0),
            (0, 1, 0, 1.0),
            (0, 1, 1, 11.0),
            (1, 0, 0, 100.0),
            (1, 0, 1, 110.0),
            (1, 1, 0, 101.0),
            (1, 1, 1, 111.0),
        ];
        assert_eq!(got.len(), expected.len());
        for (got, expected) in got.iter().zip(expected) {
            assert_eq!(got.0, expected.0);
            assert_eq!(got.1, expected.1);
            assert_eq!(got.2, expected.2);
            assert_eq!(bits(got.3), bits(expected.3));
        }
    }

    #[test]
    fn full_output_returns_every_row_draw_and_grid_point() {
        let features = [1.0, 2.0];
        let coefficients = [1.0, 0.0, 5.0, 0.0, 7.0, 0.0];
        let outcomes = [OutcomeDraws { id: 3, coefficients: &coefficients }];
        let failed = [false, false, false];
        let grid = [point(10, 0.0, 1.0), point(11, 1.0, 1.0)];
        let mut request = request(
            &features,
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            AxisOrder::row_major(),
            Reduction::Full,
        );
        request.row_count = 2;
        request.row_width = 1;
        let (values, summary) = collect(&request).unwrap();
        assert_eq!(summary.emitted, 2 * 3 * 2);
        let mut seen = [[[false; 2]; 3]; 2];
        for value in &values {
            match value {
                StreamedValue::Full { row, draw, grid_id, .. } => {
                    let grid_slot = usize::from(*grid_id != 10);
                    let row_slot = usize::try_from(*row).unwrap();
                    let draw_slot = usize::try_from(*draw).unwrap();
                    assert!(!seen[row_slot][draw_slot][grid_slot]);
                    seen[row_slot][draw_slot][grid_slot] = true;
                }
                other => panic!("full, got {other:?}"),
            }
        }
        assert!(seen.iter().flatten().flatten().all(|present| *present));
        request.draw_start = 1;
        request.draw_count = 2;
        let (tiled, _) = collect(&request).unwrap();
        assert_eq!(tiled.len(), 2 * 2 * 2);
        let cell = tiled[0];
        match cell {
            StreamedValue::Full { draw: 1, value, .. } => assert_eq!(bits(value), bits(5.0)),
            other => panic!("draw tile, got {other:?}"),
        }
    }

    #[test]
    fn draw_mean_keeps_every_finite_draw() {
        let outcomes = [OutcomeDraws { id: 1, coefficients: &[1.0, 2.0, 3.0] }];
        let failed = [false, false, false];
        let grid = [point(0, 0.0, 1.0), point(1, 0.0, 3.0)];
        let order =
            AxisOrder::try_new([EvalAxis::Grid, EvalAxis::Row, EvalAxis::Draw, EvalAxis::Outcome])
                .unwrap();
        let request = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            order,
            Reduction::DrawMean,
        );
        let (values, _) = collect(&request).unwrap();
        match (&values[0], &values[1]) {
            (
                StreamedValue::DrawMean { grid_id: 0, value: first, .. },
                StreamedValue::DrawMean { grid_id: 1, value: second, .. },
            ) => {
                assert_eq!(bits(*first), bits(2.0));
                assert_eq!(bits(*second), bits(6.0));
            }
            other => panic!("mean order, got {other:?}"),
        }
    }

    #[test]
    fn draw_covariance_uses_the_aligned_draw_index() {
        // Degree 1, row width 0, factor 1. Draw 0 predicts z. Draw 1 predicts 2 + 3z.
        let coefficients = [0.0, 1.0, 2.0, 3.0];
        let outcomes = [OutcomeDraws { id: 1, coefficients: &coefficients }];
        let failed = [false, false];
        let grid = [point(0, 0.0, 1.0), point(1, 1.0, 1.0)];
        let request = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 1 },
            AxisOrder::row_major(),
            Reduction::DrawCovariance,
        );
        let (values, _) = collect(&request).unwrap();
        let got: Vec<f64> = values
            .iter()
            .map(|value| match value {
                StreamedValue::DrawCovariance { value, .. } => *value,
                other => panic!("covariance, got {other:?}"),
            })
            .collect();
        let expected = [1.0, 2.0, 2.0, 4.0];
        assert_eq!(got.len(), expected.len());
        for (got, expected) in got.iter().zip(expected) {
            assert_eq!(bits(*got), bits(expected));
        }
    }

    #[test]
    fn covariance_component_order_follows_declared_axes() {
        let outcomes = [
            OutcomeDraws { id: 8, coefficients: &[1.0] },
            OutcomeDraws { id: 4, coefficients: &[1.0] },
        ];
        let failed = [false];
        let grid = [point(2, 0.0, 1.0), point(6, 0.0, 1.0)];
        let order =
            AxisOrder::try_new([EvalAxis::Row, EvalAxis::Outcome, EvalAxis::Grid, EvalAxis::Draw])
                .unwrap();
        let request = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            order,
            Reduction::DrawCovariance,
        );
        let (values, _) = collect(&request).unwrap();
        let pairs: Vec<(u64, u64)> = values
            .iter()
            .map(|value| match value {
                StreamedValue::DrawCovariance { left_outcome, left_grid_id, .. } => {
                    (*left_outcome, *left_grid_id)
                }
                other => panic!("covariance, got {other:?}"),
            })
            .collect();
        // Outcome is slower than grid, so the left index runs (8,2), (8,6), (4,2), (4,6),
        // and each of those is repeated across the four right-hand entries.
        assert_eq!(pairs[0], (8, 2));
        assert_eq!(pairs[4], (8, 6));
        assert_eq!(pairs[8], (4, 2));
        assert_eq!(pairs[12], (4, 6));
    }

    #[test]
    fn row_and_grid_tiles_match_the_full_frame() {
        let features = [1.0, 2.0, 3.0, 4.0];
        let coefficients = [1.0, 1.0];
        let outcomes = [OutcomeDraws { id: 1, coefficients: &coefficients }];
        let failed = [false];
        let grid = [point(0, 0.0, 1.0), point(1, 0.0, 1.0), point(2, 0.0, 1.0)];
        let mut full = request(
            &features,
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            AxisOrder::row_major(),
            Reduction::Full,
        );
        full.row_count = 4;
        full.row_width = 1;
        let (all, _) = collect(&full).unwrap();
        let mut rows = full;
        rows.row_count = 2;
        let (first_rows, _) = collect(&rows).unwrap();
        rows.row_start = 2;
        let (second_rows, _) = collect(&rows).unwrap();
        assert_eq!(first_rows.len() + second_rows.len(), all.len());
        for (tiled, full) in first_rows.iter().chain(&second_rows).zip(&all) {
            match (tiled, full) {
                (
                    StreamedValue::Full { value: left, row: left_row, .. },
                    StreamedValue::Full { value: right, row: right_row, .. },
                ) => {
                    assert_eq!(left_row, right_row);
                    assert_eq!(bits(*left), bits(*right));
                }
                _ => panic!("row tiles"),
            }
        }
        let mut grids = full;
        grids.grid_count = 2;
        let (leading, _) = collect(&grids).unwrap();
        let mut covariance = full;
        covariance.reduction = Reduction::DrawCovariance;
        let (full_covariance, _) = collect(&covariance).unwrap();
        covariance.grid_count = 2;
        let (tiled_covariance, summary) = collect(&covariance).unwrap();
        assert_eq!(summary.invariant_evaluations, 2);
        assert_eq!(leading.len(), 4 * 2);
        assert_eq!(tiled_covariance.len(), 4 * 2 * 2);
        match (&tiled_covariance[0], &full_covariance[0]) {
            (
                StreamedValue::DrawCovariance { value: left, .. },
                StreamedValue::DrawCovariance { value: right, .. },
            ) => assert_eq!(bits(*left), bits(*right)),
            _ => panic!("grid covariance tile"),
        }
    }

    #[test]
    fn mean_and_covariance_refuse_a_partial_draw_tile() {
        let outcomes = [OutcomeDraws { id: 1, coefficients: &[1.0, 2.0, 3.0] }];
        let failed = [false, false, false];
        let grid = [point(0, 0.0, 1.0)];
        let mut request = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            AxisOrder::row_major(),
            Reduction::DrawMean,
        );
        request.draw_start = 1;
        request.draw_count = 2;
        let error = stream_tile(&request, |_| {}).unwrap_err();
        assert_eq!(error, TileError::PartialDraws { full: 3, start: 1, count: 2 });
        request.reduction = Reduction::DrawCovariance;
        assert!(matches!(stream_tile(&request, |_| {}), Err(TileError::PartialDraws { .. })));
    }

    #[test]
    fn polynomial_and_horizon_encodings_match_a_hand_evaluation() {
        let outcomes = [OutcomeDraws { id: 1, coefficients: &[1.0, 1.0, 1.0] }];
        let failed = [false];
        let grid = [point(5, 3.0, 2.0)];
        let poly = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 2 },
            AxisOrder::row_major(),
            Reduction::Full,
        );
        let (values, _) = collect(&poly).unwrap();
        let cell = values[0];
        match cell {
            // factor * (1 + z + z²) = 2 * (1 + 3 + 9) = 26, times coefficient 1.
            StreamedValue::Full { value, .. } => assert_eq!(bits(value), bits(26.0)),
            other => panic!("polynomial, got {other:?}"),
        }
        let outcomes = [OutcomeDraws { id: 1, coefficients: &[5.0, 7.0, 9.0] }];
        let grid = [point(1, 1.0, 4.0)];
        let horizon = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::HorizonIndicator { width: 3 },
            AxisOrder::row_major(),
            Reduction::Full,
        );
        let (values, _) = collect(&horizon).unwrap();
        let cell = values[0];
        match cell {
            StreamedValue::Full { value, .. } => assert_eq!(bits(value), bits(28.0)),
            other => panic!("horizon, got {other:?}"),
        }
    }

    #[test]
    fn caller_grid_order_is_preserved() {
        let outcomes = [OutcomeDraws { id: 1, coefficients: &[0.0, 1.0] }];
        let failed = [false];
        let grid = [point(5, 3.0, 1.0), point(1, 1.0, 1.0)];
        let request = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 1 },
            AxisOrder::row_major(),
            Reduction::Full,
        );
        let (values, _) = collect(&request).unwrap();
        match (&values[0], &values[1]) {
            (
                StreamedValue::Full { grid_id: 5, value: first, .. },
                StreamedValue::Full { grid_id: 1, value: second, .. },
            ) => {
                assert_eq!(bits(*first), bits(3.0));
                assert_eq!(bits(*second), bits(1.0));
            }
            _ => panic!("grid order {values:?}"),
        }
    }

    #[test]
    fn non_finite_site_and_empty_axes_are_refused() {
        let outcomes = [OutcomeDraws { id: 1, coefficients: &[1.0] }];
        let failed = [false];
        let grid = [point(4, f64::NAN, 1.0)];
        let request = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            AxisOrder::row_major(),
            Reduction::Full,
        );
        let mut emitted = 0;
        let error = stream_tile(&request, |_| emitted += 1).unwrap_err();
        assert_eq!(emitted, 0);
        assert_eq!(error, TileError::NonFinitePoint { grid_id: 4 });
        let mut empty = request;
        let grid_point = [point(0, 0.0, 1.0)];
        empty.grid = &grid_point;
        empty.grid_count = 0;
        assert_eq!(stream_tile(&empty, |_| {}).unwrap_err(), TileError::Empty { axis: "grid" });
        assert_eq!(
            AxisOrder::try_new([EvalAxis::Row, EvalAxis::Row, EvalAxis::Draw, EvalAxis::Grid])
                .unwrap_err(),
            TileError::AxisOrder
        );
        let horizon = [point(9, 1.5, 1.0)];
        let wide = [OutcomeDraws { id: 1, coefficients: &[0.0, 0.0, 0.0] }];
        let mut bad_horizon = request;
        bad_horizon.outcomes = &wide;
        bad_horizon.grid = &horizon;
        bad_horizon.grid_count = 1;
        bad_horizon.encoding = Encoding::HorizonIndicator { width: 3 };
        assert_eq!(
            stream_tile(&bad_horizon, |_| {}).unwrap_err(),
            TileError::Horizon { grid_id: 9 }
        );
        assert_eq!(Encoding::HorizonIndicator { width: 0 }.width(), Err(TileError::HorizonWidth));
    }

    #[test]
    fn negative_zero_is_horizon_zero() {
        let outcomes = [OutcomeDraws { id: 1, coefficients: &[5.0, 7.0] }];
        let failed = [false];
        for coordinate in [0.0, -0.0] {
            let grid = [point(3, coordinate, 2.0)];
            let request = request(
                &[],
                &outcomes,
                &failed,
                &grid,
                Encoding::HorizonIndicator { width: 2 },
                AxisOrder::row_major(),
                Reduction::Full,
            );
            let (values, _) = collect(&request).unwrap();
            match values.first() {
                Some(&StreamedValue::Full { value, .. }) => assert_eq!(bits(value), bits(10.0)),
                other => panic!("horizon zero, got {other:?}"),
            }
        }
    }

    #[test]
    fn covariance_matches_the_hand_computed_symmetric_matrix() {
        // Two sites with factors 1 and 2 on one intercept coefficient, draws 1, 2, 6:
        // y0 = [1, 2, 6], y1 = 2 * y0. Population variance of y0 is 14 / 3.
        let outcomes = [OutcomeDraws { id: 1, coefficients: &[1.0, 2.0, 6.0] }];
        let failed = [false, false, false];
        let grid = [point(0, 0.0, 1.0), point(1, 0.0, 2.0)];
        let request = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            AxisOrder::row_major(),
            Reduction::DrawCovariance,
        );
        let (values, summary) = collect(&request).unwrap();
        assert_eq!(summary.emitted, 4);
        let expected = [14.0 / 3.0, 28.0 / 3.0, 28.0 / 3.0, 56.0 / 3.0];
        for (cell, want) in values.iter().zip(expected) {
            match cell {
                StreamedValue::DrawCovariance { value, .. } => {
                    assert!((value - want).abs() < 1e-12, "{value} vs {want}");
                }
                other => panic!("covariance, got {other:?}"),
            }
        }
    }

    #[test]
    fn oversized_covariance_tile_is_refused_before_allocating() {
        let outcomes = [OutcomeDraws { id: 1, coefficients: &[1.0] }];
        let failed = [false];
        let grid: Vec<GridPoint> = (0..12_000).map(|id| point(id, 0.0, 1.0)).collect();
        let request = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            AxisOrder::row_major(),
            Reduction::DrawCovariance,
        );
        let mut emitted = 0;
        let error = stream_tile(&request, |_| emitted += 1).unwrap_err();
        assert_eq!(emitted, 0);
        assert!(
            matches!(error, TileError::CovarianceBudget { limit, .. } if limit == MAX_COVARIANCE_BYTES)
        );
    }

    #[test]
    fn a_walk_that_drops_repeats_or_unfails_a_cell_is_refused() {
        let outcomes = [
            OutcomeDraws { id: 1, coefficients: &[1.0, 2.0, 3.0] },
            OutcomeDraws { id: 2, coefficients: &[4.0, 5.0, 6.0] },
        ];
        let failed = [false, true, false];
        let grid = [point(0, 0.0, 1.0), point(1, 1.0, 1.0)];
        let mut full = request(
            &[],
            &outcomes,
            &failed,
            &grid,
            Encoding::Polynomial { degree: 0 },
            AxisOrder::row_major(),
            Reduction::Full,
        );
        full.row_count = 2;
        let (values, _) = collect(&full).unwrap();
        let frame = Frame::prepare(&full).unwrap();
        let audit_of = |cells: &[StreamedValue]| {
            let mut audit = TileAudit::new(&frame).unwrap();
            for cell in cells {
                audit.see(*cell);
            }
            audit.admit(&frame)
        };
        assert_eq!(audit_of(&values), Ok(()));

        let mut dropped = values.clone();
        dropped.remove(5);
        assert_eq!(
            audit_of(&dropped),
            Err(TileError::Altered { axis: PreservationAxis::SampleSelection })
        );

        let mut repeated = values.clone();
        repeated.push(values[3]);
        assert_eq!(
            audit_of(&repeated),
            Err(TileError::Altered { axis: PreservationAxis::SampleSelection })
        );

        let mut relabeled = values.clone();
        if let StreamedValue::Full { row, .. } = &mut relabeled[0] {
            *row = 7;
        }
        assert!(audit_of(&relabeled).is_err());

        let mut unfailed = values.clone();
        let target = unfailed
            .iter()
            .position(|cell| matches!(cell, StreamedValue::Full { draw: 1, .. }))
            .unwrap();
        if let StreamedValue::Full { value, .. } = &mut unfailed[target] {
            *value = 0.0;
        }
        assert_eq!(
            audit_of(&unfailed),
            Err(TileError::Altered { axis: PreservationAxis::Uncertainty })
        );

        let mut mean = full;
        mean.reduction = Reduction::DrawMean;
        let (means, _) = collect(&mean).unwrap();
        let mean_frame = Frame::prepare(&mean).unwrap();
        let mut audit = TileAudit::new(&mean_frame).unwrap();
        for cell in &means[1..] {
            audit.see(*cell);
        }
        assert!(audit.admit(&mean_frame).is_err());
    }

    #[test]
    fn a_stalled_sink_keeps_live_row_chunks_at_the_window_and_values_match_the_serial_stream() {
        use antecedent_core::ExecutionContext;
        use antecedent_core::execution::AdmissionPool;

        let rows = 64usize;
        let features: Vec<f64> = (0..rows).map(|i| i as f64 * 0.25).collect();
        let coefficients = [1.0, 0.5, 2.0, 1.5, -1.0, 0.25];
        let outcomes = [OutcomeDraws { id: 1, coefficients: &coefficients }];
        let failed = [false, false, false];
        let grid = [point(0, 0.0, 1.0), point(1, 1.0, 1.0)];
        for reduction in [Reduction::Full, Reduction::DrawMean, Reduction::DrawCovariance] {
            let mut whole = request(
                &features,
                &outcomes,
                &failed,
                &grid,
                Encoding::Polynomial { degree: 0 },
                AxisOrder::row_major(),
                reduction,
            );
            whole.row_count = rows as u64;
            whole.row_width = 1;
            // Two coefficients per draw: one encoding term, one row feature.
            let coefficients = [1.0, 0.5, 2.0, 1.5, -1.0, 0.25];
            let outcomes = [OutcomeDraws { id: 1, coefficients: &coefficients }];
            whole.outcomes = &outcomes;
            let (serial, _) = collect(&whole).unwrap();

            let pool = AdmissionPool::fixed(4, 1 << 24);
            let ctx = ExecutionContext::production(1, 4).with_admission(pool.clone());
            let mut got = Vec::new();
            let mut first = true;
            let tile = stream_tile_pipelined(&whole, &ctx, 4, 3, |value| {
                if std::mem::take(&mut first) {
                    std::thread::sleep(std::time::Duration::from_millis(80));
                }
                got.push(value);
            })
            .unwrap();
            assert_eq!(got.len(), serial.len(), "{reduction:?}");
            for (left, right) in got.iter().zip(&serial) {
                assert_eq!(format!("{left:?}"), format!("{right:?}"));
            }
            assert_eq!(tile.summary.emitted, serial.len() as u64);
            assert!(tile.receipt.peak_live <= 3, "{reduction:?}: {}", tile.receipt.peak_live);
            assert_eq!(tile.receipt.items_consumed, 16);
            assert_eq!(pool.available().bytes(), 1 << 24);
        }
    }
}
