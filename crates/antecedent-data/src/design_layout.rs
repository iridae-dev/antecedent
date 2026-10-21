// SPDX-License-Identifier: MIT OR Apache-2.0

//! Column-major numeric design layout.
//!
//! Selected rows of selected columns land in one declared layout. An in-order
//! contiguous run of a contiguous `f64` column is borrowed from that buffer, so
//! the receipt lists no ownership copy. A cast into `f64`, or a gather of a
//! selection that is not an in-order contiguous run, owns one buffer. The
//! receipt names that copy. A cast of a noncontiguous selection is still one
//! copy: the cast is the owned buffer.
//!
//! `NumPy` is outside this crate. This module does not read Arrow arrays. A
//! float64 column that already borrows a foreign buffer is reused through its
//! contiguous `f64` slice. A nanosecond-epoch column is not cast into that
//! layout: [`DesignLayout::with_time_index`] keeps the selected epochs as `i64`.

use std::borrow::Cow;
use std::fmt;

use crate::column::{ColumnView, ValidityBitmap};
use crate::global_index::GlobalId;

/// Empty `f64` slice borrowed when a numeric column contributes no rows.
const EMPTY_F64: &[f64] = &[];

/// Column-major `f64` layout of selected columns and rows.
///
/// Column `c`, row `r` occupies index `c * n_rows + r`. Each column is its own
/// contiguous run. A borrowed run is the source buffer (or a subslice of it).
/// An owned run is one receipt entry.
#[derive(Clone, Debug)]
pub struct DesignLayout<'a> {
    n_rows: usize,
    columns: Vec<Cow<'a, [f64]>>,
    receipt: OwnershipReceipt,
    /// Selected nanosecond epochs. Not an `f64` column and not a cast.
    time_index_ns: Option<Cow<'a, [i64]>>,
}

/// Every ownership copy the layout still performed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnershipReceipt {
    copies: Vec<OwnershipCopy>,
}

/// One owned numeric buffer.
///
/// [`Self::name`] is [`Self::CAST`] or [`Self::GATHER`]. A noncontiguous
/// selection of an `f64` column is that gather: one named copy, not one copy
/// for the gather and another for the selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OwnershipCopy {
    name: &'static str,
    column: usize,
}

/// A row id is past a column, the column is not numeric, an `i64` level is not an
/// exact `f64`, or a nanosecond epoch was offered as a design cast.
///
/// The error is `Copy` and holds no buffer. [`DesignLayout::from_views`]
/// returns it before building any column vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayoutError {
    column: usize,
    kind: LayoutFailure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LayoutFailure {
    PastColumn {
        id: GlobalId,
        len: usize,
    },
    NotNumeric,
    InexactInteger {
        value: i64,
    },
    /// Timestamp column passed to [`DesignLayout::from_views`].
    EpochColumn,
    /// [`DesignLayout::with_time_index`] was not given an epoch column.
    NotAnEpoch,
    /// A selected time-index row is invalid.
    InvalidEpoch {
        id: GlobalId,
    },
}

/// Largest magnitude below which every `i64` is an exact `f64`.
const EXACT_F64_INTEGER: u64 = 1 << 53;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RowRun {
    Contiguous { start: usize },
    Gather,
}

impl OwnershipCopy {
    /// Receipt name for a cast into `f64`.
    pub const CAST: &'static str = "cast";

    /// Receipt name for a gathered `f64` selection.
    ///
    /// This is the copy a noncontiguous selection records.
    pub const GATHER: &'static str = "gather";

    const fn cast(column: usize) -> Self {
        Self { name: Self::CAST, column }
    }

    const fn gather(column: usize) -> Self {
        Self { name: Self::GATHER, column }
    }

    /// `"cast"` or `"gather"`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// Column in the layout that owns the buffer.
    #[must_use]
    pub const fn column(self) -> usize {
        self.column
    }
}

impl OwnershipReceipt {
    /// Ownership copies, in column order.
    #[must_use]
    pub fn copies(&self) -> &[OwnershipCopy] {
        &self.copies
    }

    /// No owned numeric buffer was required.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.copies.is_empty()
    }
}

impl LayoutError {
    const fn past_column(id: GlobalId, len: usize) -> Self {
        Self { column: 0, kind: LayoutFailure::PastColumn { id, len } }
    }

    const fn not_numeric(column: usize) -> Self {
        Self { column, kind: LayoutFailure::NotNumeric }
    }

    const fn inexact_integer(value: i64) -> Self {
        Self { column: 0, kind: LayoutFailure::InexactInteger { value } }
    }

    const fn epoch_column(column: usize) -> Self {
        Self { column, kind: LayoutFailure::EpochColumn }
    }

    const fn not_an_epoch() -> Self {
        Self { column: 0, kind: LayoutFailure::NotAnEpoch }
    }

    const fn invalid_epoch(id: GlobalId) -> Self {
        Self { column: 0, kind: LayoutFailure::InvalidEpoch { id } }
    }

    /// The `i64` value that is not an exact `f64`, when that is the refusal.
    #[must_use]
    pub const fn inexact_value(self) -> Option<i64> {
        match self.kind {
            LayoutFailure::InexactInteger { value } => Some(value),
            LayoutFailure::PastColumn { .. }
            | LayoutFailure::NotNumeric
            | LayoutFailure::EpochColumn
            | LayoutFailure::NotAnEpoch
            | LayoutFailure::InvalidEpoch { .. } => None,
        }
    }

    const fn in_column(mut self, column: usize) -> Self {
        self.column = column;
        self
    }

    /// Column that refused the selection.
    #[must_use]
    pub const fn column_index(self) -> usize {
        self.column
    }

    /// Row id that does not fit the column, or the invalid time-index row.
    #[must_use]
    pub const fn global_id(self) -> Option<GlobalId> {
        match self.kind {
            LayoutFailure::PastColumn { id, .. } | LayoutFailure::InvalidEpoch { id } => Some(id),
            LayoutFailure::NotNumeric
            | LayoutFailure::InexactInteger { .. }
            | LayoutFailure::EpochColumn
            | LayoutFailure::NotAnEpoch => None,
        }
    }

    /// Length of the column the id did not fit, when the refusal is a past-the-end id.
    #[must_use]
    pub const fn column_len(self) -> Option<usize> {
        match self.kind {
            LayoutFailure::PastColumn { len, .. } => Some(len),
            LayoutFailure::NotNumeric
            | LayoutFailure::InexactInteger { .. }
            | LayoutFailure::EpochColumn
            | LayoutFailure::NotAnEpoch
            | LayoutFailure::InvalidEpoch { .. } => None,
        }
    }
}

impl fmt::Display for LayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            LayoutFailure::PastColumn { id, len } => {
                let id = id.get();
                let column = self.column;
                write!(formatter, "global id {id} is past column {column} of length {len}")
            }
            LayoutFailure::NotNumeric => {
                let column = self.column;
                write!(formatter, "column {column} is not an f64, int64, or boolean column")
            }
            LayoutFailure::InexactInteger { value } => {
                let column = self.column;
                write!(
                    formatter,
                    "int64 value {value} in column {column} is not exact as f64 (limit 2^53); \
                     a nanosecond-epoch column uses DesignLayout::with_time_index and is not cast"
                )
            }
            LayoutFailure::EpochColumn => {
                let column = self.column;
                write!(
                    formatter,
                    "column {column} is a nanosecond-epoch timestamp; use \
                     DesignLayout::with_time_index instead of casting it into the design"
                )
            }
            LayoutFailure::NotAnEpoch => write!(
                formatter,
                "DesignLayout::with_time_index takes a nanosecond-epoch \
                 int64 or timestamp column"
            ),
            LayoutFailure::InvalidEpoch { id } => {
                let id = id.get();
                write!(
                    formatter,
                    "time-index row {id} is invalid; DesignLayout::with_time_index does not cast \
                     a missing epoch"
                )
            }
        }
    }
}

impl std::error::Error for LayoutError {}

impl<'a> DesignLayout<'a> {
    /// Place `columns` at `rows` into a column-major `f64` layout.
    ///
    /// `rows` is the same [`GlobalId`] list for every column. An in-order
    /// contiguous run of an `f64` column borrows that buffer. Any other `f64`
    /// selection gathers one owned column named [`OwnershipCopy::GATHER`]. An
    /// `i64` or boolean column is cast into one owned column named
    /// [`OwnershipCopy::CAST`], including when the selection is also
    /// noncontiguous.
    ///
    /// # Errors
    ///
    /// [`LayoutError`] when a [`GlobalId`] is past a column, a column is not
    /// `f64`, `i64`, or boolean, a timestamp column is passed here, or a valid
    /// `i64` value has magnitude above 2^53 and would round when cast. A
    /// nanosecond epoch uses [`Self::with_time_index`] and is not cast. The
    /// refusal allocates no layout buffer.
    #[must_use = "a row past the column is refused before any layout allocation"]
    pub fn from_views(columns: &[ColumnView<'a>], rows: &[GlobalId]) -> Result<Self, LayoutError> {
        validate(columns, rows)?;
        let mut stored = Vec::with_capacity(columns.len());
        let mut copies = Vec::new();
        for (column_index, view) in columns.iter().copied().enumerate() {
            stored.push(place_column(view, rows, column_index, &mut copies)?);
        }
        Ok(Self {
            n_rows: rows.len(),
            columns: stored,
            receipt: OwnershipReceipt { copies },
            time_index_ns: None,
        })
    }

    /// Numeric design plus a nanosecond-epoch time index.
    ///
    /// `time_index` is an `i64` or timestamp column. Selected epochs stay `i64`
    /// nanoseconds. They are not cast into the `f64` columns, and a value above
    /// 2^53 is kept exactly. A contiguous in-order run of valid epochs is borrowed.
    ///
    /// # Errors
    ///
    /// [`Self::from_views`] refuses `columns`. `time_index` is not an epoch
    /// column, a selected epoch row is invalid, or a row id is past that column.
    /// An invalid epoch allocates no layout buffer.
    #[must_use = "an epoch that is not a time index is refused before any layout allocation"]
    pub fn with_time_index(
        columns: &[ColumnView<'a>],
        rows: &[GlobalId],
        time_index: ColumnView<'a>,
    ) -> Result<Self, LayoutError> {
        let time_index_ns = place_time_index(time_index, rows)?;
        let mut layout = Self::from_views(columns, rows)?;
        layout.time_index_ns = Some(time_index_ns);
        Ok(layout)
    }

    /// Selected row count. Every column has this length.
    #[must_use]
    pub const fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Selected column count.
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// Contiguous `f64` values of one selected column.
    #[must_use]
    pub fn column(&self, index: usize) -> Option<&[f64]> {
        self.columns.get(index).map(Cow::as_ref)
    }

    /// Value at column-major index `column * n_rows + row`.
    #[must_use]
    pub fn get(&self, column_major_index: usize) -> Option<f64> {
        let n_rows = self.n_rows;
        if n_rows == 0 {
            return None;
        }
        let column = column_major_index / n_rows;
        let row = column_major_index % n_rows;
        self.column(column)?.get(row).copied()
    }

    /// Ownership copies performed while building this layout.
    #[must_use]
    pub const fn receipt(&self) -> &OwnershipReceipt {
        &self.receipt
    }

    /// Selected nanosecond epochs, when [`Self::with_time_index`] supplied them.
    ///
    /// This is not an `f64` design column.
    #[must_use]
    pub fn time_index_ns(&self) -> Option<&[i64]> {
        self.time_index_ns.as_deref()
    }
}

/// Check every column before any layout vector exists.
fn validate(columns: &[ColumnView<'_>], rows: &[GlobalId]) -> Result<(), LayoutError> {
    for (column_index, view) in columns.iter().copied().enumerate() {
        ensure_numeric(view, column_index)?;
        row_run(view.len(), rows).map_err(|error| error.in_column(column_index))?;
    }
    Ok(())
}

const fn ensure_numeric(view: ColumnView<'_>, column_index: usize) -> Result<(), LayoutError> {
    match view {
        ColumnView::Float64(_) | ColumnView::Int64(_) | ColumnView::Boolean(_) => Ok(()),
        ColumnView::Timestamp(_) => Err(LayoutError::epoch_column(column_index)),
        ColumnView::Categorical(_) | ColumnView::FixedVector(_) => {
            Err(LayoutError::not_numeric(column_index))
        }
    }
}

fn row_index(id: GlobalId, len: usize) -> Result<usize, LayoutError> {
    let index = usize::try_from(id.get()).map_err(|_| LayoutError::past_column(id, len))?;
    if index >= len {
        return Err(LayoutError::past_column(id, len));
    }
    Ok(index)
}

/// `Contiguous` when `rows` is `start, start + 1, ...` inside the column.
///
/// Every id is checked before `Gather` is returned, so a later past-the-end id
/// refuses the layout before a gather buffer exists.
fn row_run(len: usize, rows: &[GlobalId]) -> Result<RowRun, LayoutError> {
    let Some(first) = rows.first() else {
        return Ok(RowRun::Contiguous { start: 0 });
    };
    let start = row_index(*first, len)?;
    let mut previous = start;
    let mut contiguous = true;
    for id in rows.iter().skip(1) {
        let index = row_index(*id, len)?;
        if previous.checked_add(1) != Some(index) {
            contiguous = false;
        }
        previous = index;
    }
    if contiguous { Ok(RowRun::Contiguous { start }) } else { Ok(RowRun::Gather) }
}

fn place_column<'a>(
    view: ColumnView<'a>,
    rows: &[GlobalId],
    column_index: usize,
    copies: &mut Vec<OwnershipCopy>,
) -> Result<Cow<'a, [f64]>, LayoutError> {
    if rows.is_empty() {
        return Ok(Cow::Borrowed(empty_slice(view)));
    }
    let run = row_run(view.len(), rows).map_err(|error| error.in_column(column_index))?;
    match view {
        ColumnView::Float64(column) => {
            place_f64(column.values.as_slice(), rows, column_index, run, copies)
        }
        ColumnView::Int64(column) => {
            let owned = cast_i64(column.values.as_ref(), &column.validity, rows)
                .map_err(|error| error.in_column(column_index))?;
            copies.push(OwnershipCopy::cast(column_index));
            Ok(Cow::Owned(owned))
        }
        ColumnView::Boolean(column) => {
            let owned = cast_bool(column.values.as_ref(), &column.validity, rows)?;
            copies.push(OwnershipCopy::cast(column_index));
            Ok(Cow::Owned(owned))
        }
        ColumnView::Timestamp(_) => Err(LayoutError::epoch_column(column_index)),
        ColumnView::Categorical(_) | ColumnView::FixedVector(_) => {
            Err(LayoutError::not_numeric(column_index))
        }
    }
}

fn empty_slice(view: ColumnView<'_>) -> &[f64] {
    match view {
        ColumnView::Float64(column) => column.values.as_slice().get(..0).unwrap_or(EMPTY_F64),
        ColumnView::Int64(_)
        | ColumnView::Boolean(_)
        | ColumnView::Categorical(_)
        | ColumnView::Timestamp(_)
        | ColumnView::FixedVector(_) => EMPTY_F64,
    }
}

fn place_f64<'a>(
    values: &'a [f64],
    rows: &[GlobalId],
    column_index: usize,
    run: RowRun,
    copies: &mut Vec<OwnershipCopy>,
) -> Result<Cow<'a, [f64]>, LayoutError> {
    match run {
        RowRun::Contiguous { start } => {
            Ok(Cow::Borrowed(borrow_contiguous(values, start, rows.len())?))
        }
        RowRun::Gather => {
            let owned = gather_f64(values, rows)?;
            copies.push(OwnershipCopy::gather(column_index));
            Ok(Cow::Owned(owned))
        }
    }
}

fn borrow_contiguous(values: &[f64], start: usize, n_rows: usize) -> Result<&[f64], LayoutError> {
    let end = start
        .checked_add(n_rows)
        .ok_or_else(|| LayoutError::past_column(GlobalId::new(u64::MAX), values.len()))?;
    let id = GlobalId::new(u64::try_from(start).unwrap_or(u64::MAX));
    values.get(start..end).ok_or_else(|| LayoutError::past_column(id, values.len()))
}

fn gather_f64(values: &[f64], rows: &[GlobalId]) -> Result<Vec<f64>, LayoutError> {
    selected_f64(values.len(), rows, |index| values.get(index).copied().unwrap_or(f64::NAN))
}

fn place_time_index<'a>(
    view: ColumnView<'a>,
    rows: &[GlobalId],
) -> Result<Cow<'a, [i64]>, LayoutError> {
    let (values, validity) = epoch_values(view)?;
    if rows.is_empty() {
        return Ok(Cow::Borrowed(&values[..0]));
    }
    for id in rows {
        let index = row_index(*id, values.len())?;
        if !validity.is_valid(index) {
            return Err(LayoutError::invalid_epoch(*id));
        }
    }
    let run = row_run(values.len(), rows)?;
    match run {
        RowRun::Contiguous { start } => Ok(Cow::Borrowed(borrow_epoch(values, start, rows.len())?)),
        RowRun::Gather => {
            let mut owned = Vec::with_capacity(rows.len());
            for id in rows {
                let index = row_index(*id, values.len())?;
                let Some(value) = values.get(index).copied() else {
                    return Err(LayoutError::past_column(*id, values.len()));
                };
                owned.push(value);
            }
            Ok(Cow::Owned(owned))
        }
    }
}

fn epoch_values(view: ColumnView<'_>) -> Result<(&[i64], &ValidityBitmap), LayoutError> {
    match view {
        ColumnView::Int64(column) => Ok((column.values.as_ref(), &column.validity)),
        ColumnView::Timestamp(column) => Ok((column.values_ns.as_ref(), &column.validity)),
        ColumnView::Float64(_)
        | ColumnView::Boolean(_)
        | ColumnView::Categorical(_)
        | ColumnView::FixedVector(_) => Err(LayoutError::not_an_epoch()),
    }
}

fn borrow_epoch(values: &[i64], start: usize, n_rows: usize) -> Result<&[i64], LayoutError> {
    let end = start
        .checked_add(n_rows)
        .ok_or_else(|| LayoutError::past_column(GlobalId::new(u64::MAX), values.len()))?;
    let id = GlobalId::new(u64::try_from(start).unwrap_or(u64::MAX));
    values.get(start..end).ok_or_else(|| LayoutError::past_column(id, values.len()))
}

fn cast_i64(
    values: &[i64],
    validity: &ValidityBitmap,
    rows: &[GlobalId],
) -> Result<Vec<f64>, LayoutError> {
    let mut out = Vec::with_capacity(rows.len());
    for id in rows {
        let index = row_index(*id, values.len())?;
        out.push(match values.get(index) {
            Some(value) if validity.is_valid(index) => {
                if value.unsigned_abs() > EXACT_F64_INTEGER {
                    return Err(LayoutError::inexact_integer(*value));
                }
                *value as f64
            }
            Some(_) | None => f64::NAN,
        });
    }
    Ok(out)
}

fn cast_bool(
    values: &[u8],
    validity: &ValidityBitmap,
    rows: &[GlobalId],
) -> Result<Vec<f64>, LayoutError> {
    selected_f64(values.len(), rows, |index| match values.get(index) {
        Some(value) if validity.is_valid(index) => f64::from(*value),
        Some(_) | None => f64::NAN,
    })
}

fn selected_f64(
    len: usize,
    rows: &[GlobalId],
    mut at: impl FnMut(usize) -> f64,
) -> Result<Vec<f64>, LayoutError> {
    let mut values = Vec::with_capacity(rows.len());
    for id in rows {
        let index = row_index(*id, len)?;
        values.push(at(index));
    }
    Ok(values)
}

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "the layout keeps the selected f64 bits")]
mod tests {
    use antecedent_core::VariableId;

    use super::{DesignLayout, OwnershipCopy};
    use crate::column::{
        BooleanColumn, ColumnView, Float64Column, Int64Column, TimestampColumn, ValidityBitmap,
    };
    use crate::global_index::GlobalId;

    fn float_column(values: &[f64]) -> Float64Column {
        Float64Column::new(
            VariableId::from_raw(0),
            values.to_vec(),
            ValidityBitmap::all_valid(values.len()),
        )
        .expect("float column")
    }

    #[test]
    fn in_order_contiguous_f64_has_an_empty_copy_list() {
        let x = float_column(&[1.0, 2.0, 3.0]);
        let y = float_column(&[4.0, 5.0, 6.0]);
        let rows = [GlobalId::new(0), GlobalId::new(1), GlobalId::new(2)];
        let layout =
            DesignLayout::from_views(&[ColumnView::Float64(&x), ColumnView::Float64(&y)], &rows)
                .expect("full prefix");
        assert!(layout.receipt().is_empty());
        assert!(layout.receipt().copies().is_empty());
        assert_eq!(layout.n_rows(), 3);
        assert_eq!(layout.n_columns(), 2);
        assert_eq!(layout.column(0).expect("x").as_ptr(), x.values.as_slice().as_ptr());
        assert_eq!(layout.column(1).expect("y").as_ptr(), y.values.as_slice().as_ptr());
        assert_eq!(layout.get(0), Some(1.0));
        assert_eq!(layout.get(3), Some(4.0));
        assert_eq!(layout.get(5), Some(6.0));

        let window = [GlobalId::new(1), GlobalId::new(2)];
        let layout = DesignLayout::from_views(&[ColumnView::Float64(&x)], &window).expect("window");
        assert!(layout.receipt().copies().is_empty());
        let source = x.values.as_slice().get(1..).expect("source window");
        assert_eq!(layout.column(0).expect("window").as_ptr(), source.as_ptr());
        assert_eq!(layout.column(0).expect("window"), &[2.0, 3.0]);

        let layout = DesignLayout::from_views(&[ColumnView::Float64(&x)], &[]).expect("empty");
        assert!(layout.receipt().is_empty());
        assert_eq!(layout.n_rows(), 0);
        assert_eq!(layout.column(0).expect("empty prefix").len(), 0);
        assert!(layout.get(0).is_none());
    }

    #[test]
    fn gathered_subset_names_exactly_one_copy() {
        let column = float_column(&[10.0, 20.0, 30.0, 40.0]);
        let rows = [GlobalId::new(0), GlobalId::new(2)];
        let layout =
            DesignLayout::from_views(&[ColumnView::Float64(&column)], &rows).expect("gather");
        let copies = layout.receipt().copies();
        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].name(), OwnershipCopy::GATHER);
        assert_eq!(copies[0].name(), "gather");
        assert_eq!(copies[0].column(), 0);
        let gathered = layout.column(0).expect("gathered");
        assert_eq!(gathered, &[10.0, 30.0]);
        assert_ne!(gathered.as_ptr(), column.values.as_slice().as_ptr());
    }

    #[test]
    fn global_id_past_the_column_length_is_refused_and_allocates_nothing() {
        let column = float_column(&[1.0, 2.0, 3.0]);
        let rows = [GlobalId::new(3)];
        let error = DesignLayout::from_views(&[ColumnView::Float64(&column)], &rows)
            .expect_err("index 3 is past length 3");
        assert_eq!(error.global_id(), Some(GlobalId::new(3)));
        assert_eq!(error.column_index(), 0);
        assert_eq!(error.column_len(), Some(3));
        let again = error;
        assert_eq!(again, error);

        let huge = [GlobalId::new(u64::MAX)];
        let error = DesignLayout::from_views(&[ColumnView::Float64(&column)], &huge)
            .expect_err("u64::MAX is past the column");
        assert_eq!(error.global_id(), Some(GlobalId::new(u64::MAX)));
        assert_eq!(error.column_len(), Some(3));

        let wide = float_column(&[1.0, 2.0, 3.0, 4.0]);
        let short = float_column(&[5.0, 6.0]);
        let rows = [GlobalId::new(0), GlobalId::new(2)];
        let error = DesignLayout::from_views(
            &[ColumnView::Float64(&wide), ColumnView::Float64(&short)],
            &rows,
        )
        .expect_err("column 1 rejects id 2 before a gather buffer");
        assert_eq!(error.column_index(), 1);
        assert_eq!(error.global_id(), Some(GlobalId::new(2)));
        assert_eq!(error.column_len(), Some(2));
    }

    #[test]
    fn int64_above_two_to_the_53_is_refused_not_rounded() {
        let exact = 1_i64 << 53;
        let ints = |values: Vec<i64>| {
            let n = values.len();
            Int64Column::new(VariableId::from_raw(1), values, ValidityBitmap::all_valid(n))
                .expect("int column")
        };
        let rows = [GlobalId::new(0), GlobalId::new(1)];
        // 2^53 is exact; 2^53 + 1 would round to 2^53 and collide with it.
        let ok = ints(vec![exact, -exact]);
        let layout = DesignLayout::from_views(&[ColumnView::Int64(&ok)], &rows).expect("exact");
        assert_eq!(layout.column(0).expect("cast").first().copied(), Some(9_007_199_254_740_992.0));
        let bad = ints(vec![1, exact + 1]);
        let error = DesignLayout::from_views(&[ColumnView::Int64(&bad)], &rows)
            .expect_err("2^53 + 1 is not an exact f64");
        assert_eq!(error.inexact_value(), Some(exact + 1));
        assert_eq!(error.column_index(), 0);
        let message = error.to_string();
        assert!(message.contains("DesignLayout::with_time_index"), "{message}");
        assert!(message.contains("is not cast"), "{message}");
        let negative = ints(vec![-exact - 1, 0]);
        assert_eq!(
            DesignLayout::from_views(&[ColumnView::Int64(&negative)], &rows)
                .expect_err("negative side")
                .inexact_value(),
            Some(-exact - 1)
        );
        // An invalid slot is NaN whatever its payload, so it is not refused.
        let masked = Int64Column::new(
            VariableId::from_raw(1),
            vec![1_i64, i64::MAX],
            ValidityBitmap::from_bytes(vec![0b01], 2).expect("validity"),
        )
        .expect("masked");
        DesignLayout::from_views(&[ColumnView::Int64(&masked)], &rows).expect("invalid slot");
        // The column index is that of the refusing column.
        let floats = float_column(&[1.0, 2.0]);
        let second = DesignLayout::from_views(
            &[ColumnView::Float64(&floats), ColumnView::Int64(&bad)],
            &rows,
        )
        .expect_err("second column");
        assert_eq!(second.column_index(), 1);
    }

    #[test]
    fn cast_names_one_ownership_copy() {
        let floats = float_column(&[8.0, 9.0, 10.0]);
        let validity = ValidityBitmap::from_bytes(vec![0b101], 3).expect("validity");
        let ints = Int64Column::new(VariableId::from_raw(1), vec![1_i64, 2, 3], validity)
            .expect("int column");
        let rows = [GlobalId::new(0), GlobalId::new(1), GlobalId::new(2)];
        let layout = DesignLayout::from_views(
            &[ColumnView::Float64(&floats), ColumnView::Int64(&ints)],
            &rows,
        )
        .expect("cast");
        let copies = layout.receipt().copies();
        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].name(), OwnershipCopy::CAST);
        assert_eq!(copies[0].column(), 1);
        assert_eq!(layout.column(0).expect("borrowed").as_ptr(), floats.values.as_slice().as_ptr());
        let cast = layout.column(1).expect("cast");
        assert_eq!(cast.first().copied(), Some(1.0));
        assert!(cast.get(1).copied().is_some_and(f64::is_nan));
        assert_eq!(cast.get(2).copied(), Some(3.0));

        let flags = BooleanColumn::new(
            VariableId::from_raw(2),
            vec![1_u8, 0],
            ValidityBitmap::all_valid(2),
        )
        .expect("bool column");
        let rows = [GlobalId::new(1), GlobalId::new(0)];
        let layout =
            DesignLayout::from_views(&[ColumnView::Boolean(&flags)], &rows).expect("bool layout");
        assert_eq!(layout.receipt().copies().len(), 1);
        assert_eq!(layout.receipt().copies()[0].name(), OwnershipCopy::CAST);
        assert_eq!(layout.column(0).expect("bool").first().copied(), Some(0.0));
        assert_eq!(layout.column(0).expect("bool").get(1).copied(), Some(1.0));
    }

    #[test]
    fn nanosecond_epoch_uses_the_time_index_path_and_is_not_cast() {
        let epoch = 1_700_000_000_000_000_000_i64;
        let beyond = (1_i64 << 53) + 1;
        let stamps = TimestampColumn::new(
            VariableId::from_raw(3),
            vec![epoch, epoch + 1, epoch + 2],
            ValidityBitmap::all_valid(3),
        )
        .expect("timestamp column");
        let rows = [GlobalId::new(0), GlobalId::new(1), GlobalId::new(2)];
        let refused = DesignLayout::from_views(&[ColumnView::Timestamp(&stamps)], &rows)
            .expect_err("timestamp is not a design cast");
        let message = refused.to_string();
        assert!(message.contains("DesignLayout::with_time_index"), "{message}");
        assert!(message.contains("instead of casting"), "{message}");
        assert_eq!(refused.column_index(), 0);

        let ints = Int64Column::new(
            VariableId::from_raw(1),
            vec![beyond, epoch],
            ValidityBitmap::all_valid(2),
        )
        .expect("epoch ints");
        let int_rows = [GlobalId::new(0)];
        let cast_refused = DesignLayout::from_views(&[ColumnView::Int64(&ints)], &int_rows)
            .expect_err("above 2^53");
        assert!(cast_refused.to_string().contains("DesignLayout::with_time_index"));

        let kept = DesignLayout::with_time_index(&[], &rows, ColumnView::Timestamp(&stamps))
            .expect("time index");
        assert!(kept.receipt().is_empty());
        assert_eq!(kept.n_columns(), 0);
        assert_eq!(kept.time_index_ns(), Some(stamps.values_ns.as_ref()));
        assert_eq!(kept.time_index_ns().expect("epochs").as_ptr(), stamps.values_ns.as_ptr());

        let gathered_rows = [GlobalId::new(2), GlobalId::new(0)];
        let gathered =
            DesignLayout::with_time_index(&[], &gathered_rows, ColumnView::Timestamp(&stamps))
                .expect("gathered epochs");
        assert_eq!(gathered.time_index_ns(), Some(&[epoch + 2, epoch][..]));
        assert!(gathered.column(0).is_none());
        let int_kept =
            DesignLayout::with_time_index(&[], &int_rows, ColumnView::Int64(&ints)).expect("ints");
        assert_eq!(int_kept.time_index_ns(), Some(&[beyond][..]));

        let masked = TimestampColumn::new(
            VariableId::from_raw(3),
            vec![epoch, epoch + 1],
            ValidityBitmap::from_bytes(vec![0b01], 2).expect("validity"),
        )
        .expect("masked timestamp");
        let invalid = [GlobalId::new(1)];
        let error = DesignLayout::with_time_index(&[], &invalid, ColumnView::Timestamp(&masked))
            .expect_err("missing epoch");
        assert_eq!(error.global_id(), Some(GlobalId::new(1)));
        assert!(error.to_string().contains("does not cast"));

        let floats = float_column(&[1.0, 2.0]);
        let wrong = DesignLayout::with_time_index(&[], &int_rows, ColumnView::Float64(&floats))
            .expect_err("float is not an epoch");
        assert!(wrong.to_string().contains("DesignLayout::with_time_index"));
    }
}
