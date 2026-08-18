//! Per-unit panel slice template for refute refits.
//!
//! Stacked refute mutations (`with_replaced_float`, RCC append, analysis-mask subset)
//! Arc-clone unmutated columns. Rebuilding the panel by copying every float column
//! per unit per replicate is wasted work: unmutated unit buffers can be reused and
//! only mutated / appended columns need a slice from the stacked table.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, PanelData, PanelUnit, TableView, TabularData,
    TimeSeriesData, ValidityBitmap,
};

use crate::error::ValidationError;

/// Per-unit offsets into a stacked panel table, plus payload pointers of the
/// baseline stacked columns so a later mutation can reuse original unit Arcs.
#[derive(Clone, Debug)]
pub struct PanelSliceTemplate<'a> {
    original: &'a PanelData,
    /// Payload pointer of each baseline stacked column (`None` for empty).
    baseline_ptrs: Vec<Option<*const u8>>,
    /// Number of columns on the original panel schema (appended RCC columns sit after this).
    original_ncols: usize,
    offsets: Vec<usize>,
    lengths: Vec<usize>,
}

impl<'a> PanelSliceTemplate<'a> {
    /// Compile offsets and baseline stacked-column identity from `original` + `stacked`.
    ///
    /// `stacked` must be the pre-mutation concatenation of `original` (same row count).
    ///
    /// # Errors
    ///
    /// Row-count mismatch, or a unit whose column count disagrees with the panel schema.
    pub fn from_panel(
        original: &'a PanelData,
        stacked: &TabularData,
    ) -> Result<Self, ValidationError> {
        let expected = original.total_rows();
        if stacked.row_count() != expected {
            return Err(ValidationError::data_msg(format!(
                "stacked panel refute rows {} != panel total_rows {expected}",
                stacked.row_count()
            )));
        }
        let original_ncols = original.schema().len();
        let mut offsets = Vec::with_capacity(original.unit_count());
        let mut lengths = Vec::with_capacity(original.unit_count());
        let mut offset = 0usize;
        for u in original.units() {
            let n = u.series.row_count();
            if u.series.storage().columns().len() != original_ncols {
                return Err(ValidationError::data_msg(
                    "panel unit column count disagrees with panel schema",
                ));
            }
            offsets.push(offset);
            lengths.push(n);
            offset += n;
        }
        let baseline_ptrs = stacked.storage().columns().iter().map(payload_ptr).collect();
        Ok(Self { original, baseline_ptrs, original_ncols, offsets, lengths })
    }

    /// Rebuild a panel from a (possibly mutated / column-appended) stacked table.
    ///
    /// Unmutated columns whose stacked payload pointer still matches the baseline
    /// reuse the original per-unit `OwnedColumn` Arc. Mutated and appended columns
    /// are sliced from `stacked`.
    ///
    /// # Errors
    ///
    /// Row-count mismatch, out-of-range slice, or a non-float mutated column.
    pub fn apply_stacked(&self, stacked: &TabularData) -> Result<PanelData, ValidationError> {
        if stacked.row_count() != self.original.total_rows() {
            return Err(ValidationError::data_msg(format!(
                "stacked panel refute rows {} != panel total_rows {}",
                stacked.row_count(),
                self.original.total_rows()
            )));
        }
        let stacked_storage = stacked.storage();
        let stacked_cols = stacked_storage.columns();
        let schema = stacked_storage.schema().clone();
        let mut units = Vec::with_capacity(self.original.unit_count());
        for (unit_idx, u) in self.original.units().iter().enumerate() {
            let start = self.offsets[unit_idx];
            let len = self.lengths[unit_idx];
            let mut cols = Vec::with_capacity(stacked_cols.len());
            for (j, col) in stacked_cols.iter().enumerate() {
                let reuse = j < self.original_ncols
                    && j < self.baseline_ptrs.len()
                    && payload_ptr(col) == self.baseline_ptrs[j];
                if reuse {
                    cols.push(u.series.storage().columns()[j].clone());
                } else {
                    cols.push(slice_column(col, start, len)?);
                }
            }
            let mask = stacked_storage
                .analysis_mask()
                .map(|m| slice_validity(m, start, len))
                .transpose()
                .map_err(ValidationError::from)?;
            let weights = stacked_storage
                .weights()
                .map(|w| Arc::<[f64]>::from(w[start..start + len].to_vec()));
            let storage = OwnedColumnarStorage::try_new(schema.clone(), cols, mask, weights)
                .map_err(ValidationError::from)?;
            let series = TimeSeriesData::try_new(storage, u.series.time_index().clone())
                .map_err(ValidationError::from)?;
            units.push(PanelUnit { unit_id: u.unit_id, series });
        }
        PanelData::try_new(Arc::from(units)).map_err(ValidationError::from)
    }
}

fn payload_ptr(col: &OwnedColumn) -> Option<*const u8> {
    match col {
        OwnedColumn::Float64(c) => {
            let s = c.values.as_slice();
            (!s.is_empty()).then(|| s.as_ptr().cast())
        }
        OwnedColumn::Int64(c) => (!c.values.is_empty()).then(|| c.values.as_ptr().cast()),
        OwnedColumn::Boolean(c) => (!c.values.is_empty()).then(|| c.values.as_ptr().cast()),
        OwnedColumn::Categorical(c) => (!c.codes.is_empty()).then(|| c.codes.as_ptr().cast()),
        OwnedColumn::Timestamp(c) => (!c.values_ns.is_empty()).then(|| c.values_ns.as_ptr().cast()),
        OwnedColumn::FixedVector(c) => (!c.values.is_empty()).then(|| c.values.as_ptr().cast()),
    }
}

fn slice_validity(
    src: &ValidityBitmap,
    start: usize,
    len: usize,
) -> Result<ValidityBitmap, antecedent_data::DataError> {
    let mut bytes = vec![0u8; len.div_ceil(8)];
    for i in 0..len {
        if src.is_valid(start + i) {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    ValidityBitmap::from_bytes(bytes, len)
}

fn slice_column(
    col: &OwnedColumn,
    start: usize,
    len: usize,
) -> Result<OwnedColumn, ValidationError> {
    let end = start
        .checked_add(len)
        .ok_or(ValidationError::NotApplicable { message: "panel slice out of range" })?;
    match col {
        OwnedColumn::Float64(c) => {
            if end > c.values.len() {
                return Err(ValidationError::NotApplicable { message: "panel slice out of range" });
            }
            let values: Arc<[f64]> = Arc::from(c.values.as_slice()[start..end].to_vec());
            let validity =
                slice_validity(&c.validity, start, len).map_err(ValidationError::from)?;
            Ok(OwnedColumn::Float64(
                Float64Column::new(c.id, values, validity).map_err(ValidationError::from)?,
            ))
        }
        _ => Err(ValidationError::NotApplicable {
            message: "panel refute slice requires float64 columns",
        }),
    }
}

/// Old path: copy every stacked column into per-unit buffers (differential tests).
#[cfg(test)]
pub(crate) fn copy_all_panel_from_stacked(
    original: &PanelData,
    stacked: &TabularData,
) -> Result<PanelData, ValidationError> {
    let expected = original.total_rows();
    if stacked.row_count() != expected {
        return Err(ValidationError::data_msg(format!(
            "stacked panel refute rows {} != panel total_rows {expected}",
            stacked.row_count()
        )));
    }
    let mut offset = 0usize;
    let mut units = Vec::with_capacity(original.unit_count());
    for u in original.units() {
        let n = u.series.row_count();
        let mut cols = Vec::with_capacity(stacked.storage().columns().len());
        for col in stacked.storage().columns() {
            cols.push(slice_column(col, offset, n)?);
        }
        let mask = stacked
            .storage()
            .analysis_mask()
            .map(|m| slice_validity(m, offset, n))
            .transpose()
            .map_err(ValidationError::from)?;
        let weights =
            stacked.storage().weights().map(|w| Arc::<[f64]>::from(w[offset..offset + n].to_vec()));
        let storage =
            OwnedColumnarStorage::try_new(stacked.storage().schema().clone(), cols, mask, weights)
                .map_err(ValidationError::from)?;
        let series = TimeSeriesData::try_new(storage, u.series.time_index().clone())
            .map_err(ValidationError::from)?;
        units.push(PanelUnit { unit_id: u.unit_id, series });
        offset += n;
    }
    PanelData::try_new(Arc::from(units)).map_err(ValidationError::from)
}
