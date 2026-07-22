//! General sample request → plan → prepare path.
//!
//! Temporal lag-only gathers remain in [`crate::sample`] as
//! [`LaggedSamplePlan`](crate::LaggedSamplePlan).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::{KernelPolicy, Lag, NodeRef, VariableId};
use causal_kernels::{F64MatrixView, F64VectorView, gather};

use crate::aligned_buffer::AlignedBuffer;
use crate::column::ColumnView;
use crate::dataset::{TabularData, TimeSeriesData};
use crate::error::DataError;
use crate::reference::ReferencePointPolicy;
use crate::sample::{DropSummary, LagMap};
use crate::sample_policy::{MaskPolicy, MissingPolicy, WeightPolicy};
use crate::table::TableView;
use crate::temporal::SamplingRegularity;

/// Alias for the design-matrix view returned by [`PreparedSample`].
pub type MatrixRef<'a> = F64MatrixView<'a>;

/// Borrowed row-index selection.
#[derive(Clone, Copy, Debug)]
pub struct RowSelectionRef<'a> {
    /// Selected raw row indexes (`u32`).
    pub indexes: &'a [u32],
}

/// Column-role partitions within a prepared sample matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SamplePartitions {
    /// Number of X columns (treatment / predictors of interest).
    pub n_x: usize,
    /// Number of Y columns (outcomes).
    pub n_y: usize,
    /// Number of Z columns (conditioning / covariates).
    pub n_z: usize,
}

impl SamplePartitions {
    /// Total planned columns.
    #[must_use]
    pub const fn ncols(self) -> usize {
        self.n_x + self.n_y + self.n_z
    }
}

/// How prepared values are laid out in the workspace buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum SampleLayout {
    /// Column-major blocks of length `effective_n` (default).
    #[default]
    ColumnMajor,
}

/// One resolved column in a [`SamplePlan`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PreparedColumn {
    /// Source variable.
    pub variable: VariableId,
    /// Lag to apply (`0` / contemporaneous for static tabular).
    pub lag: Lag,
    /// Role partition index: 0 = X, 1 = Y, 2 = Z.
    pub role: u8,
}

/// Row-selection recipe baked into a plan (applied at prepare time).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedRowSelector {
    /// Candidate base rows before missingness filtering (empty = all in-range).
    pub candidate_rows: Arc<[u32]>,
    /// Missingness policy.
    pub missing: MissingPolicy,
    /// Analysis-mask policy.
    pub mask: MaskPolicy,
}

/// Cache key for reusable plans (same request + data/policy version → same rows).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SampleCacheKey {
    /// Ordered node fingerprints: (variable raw, lag raw, role).
    pub nodes: Arc<[(u32, u32, u8)]>,
    /// Reference policy discriminant + origin.
    pub reference: (u8, u64),
    /// Missing / mask / weight policy discriminants.
    pub policies: (u8, u8, u8),
    /// Row count / series length.
    pub data_len: usize,
    /// Time-index version proxy (0 for tabular; series length for temporal).
    pub time_index_version: u64,
}

/// Request describing which nodes and policies to materialize.
#[derive(Clone, Copy, Debug)]
pub struct SampleRequest<'a> {
    /// Predictors / treatments.
    pub x: &'a [NodeRef],
    /// Outcomes.
    pub y: &'a [NodeRef],
    /// Conditioning set.
    pub z: &'a [NodeRef],
    /// Temporal reference-point policy.
    pub reference: ReferencePointPolicy,
    /// Missing-value policy.
    pub missing: MissingPolicy,
    /// Analysis-mask policy.
    pub mask: MaskPolicy,
    /// Weight policy.
    pub weights: WeightPolicy,
}

impl<'a> SampleRequest<'a> {
    /// Construct a request with default policies (complete-case, honor mask/weights,
    /// series-origin reference).
    #[must_use]
    pub const fn new(x: &'a [NodeRef], y: &'a [NodeRef], z: &'a [NodeRef]) -> Self {
        Self {
            x,
            y,
            z,
            reference: ReferencePointPolicy::SeriesOrigin,
            missing: MissingPolicy::CompleteCase,
            mask: MaskPolicy::Honor,
            weights: WeightPolicy::Honor,
        }
    }
}

/// Reusable compiled sample plan.
#[derive(Clone, Debug)]
pub struct SamplePlan {
    /// Prepared column descriptors in matrix order (X then Y then Z).
    pub columns: Vec<PreparedColumn>,
    /// Row selector recipe.
    pub row_selector: PreparedRowSelector,
    /// Output layout.
    pub output_layout: SampleLayout,
    /// Cache key.
    pub cache_key: SampleCacheKey,
    /// Partitions.
    pub partitions: SamplePartitions,
    /// Shared lag map when any column is lagged; `None` for pure tabular.
    lag_map: Option<Arc<LagMap>>,
    /// Weight policy (applied at prepare).
    weight_policy: WeightPolicy,
}

/// Caller-owned scratch for [`SamplePlan::prepare`].
#[derive(Clone, Debug, Default)]
pub struct SampleWorkspace {
    /// Selected raw row indexes (`u32`).
    pub row_indexes: Vec<u32>,
    /// Column-major values.
    pub values: AlignedBuffer<f64>,
    /// Packed validity words for selected rows (optional scratch).
    pub validity_words: Vec<u64>,
    /// Extra scratch.
    pub scratch: AlignedBuffer<f64>,
    /// Gather index scratch (`usize` for kernel gather).
    gather_indexes: Vec<usize>,
    /// Optional weights for selected rows.
    weights: Vec<f64>,
}

impl SampleWorkspace {
    /// Ensure capacity for `n` selected rows and `ncols` columns.
    pub fn prepare(&mut self, n: usize, ncols: usize) {
        if self.row_indexes.len() < n {
            self.row_indexes.resize(n, 0);
        }
        self.values.resize(n.saturating_mul(ncols));
        self.gather_indexes.resize(n, 0);
        let words = n.div_ceil(64);
        if self.validity_words.len() < words {
            self.validity_words.resize(words, 0);
        }
        if self.weights.len() < n {
            self.weights.resize(n, 1.0);
        }
    }
}

/// Borrowed prepared sample.
#[derive(Clone, Copy, Debug)]
pub struct PreparedSample<'a> {
    /// Column-major design matrix.
    pub matrix: MatrixRef<'a>,
    /// X/Y/Z partitions.
    pub partitions: SamplePartitions,
    /// Selected raw rows.
    pub selected_rows: RowSelectionRef<'a>,
    /// Effective sample size.
    pub effective_n: usize,
    /// Drop summary.
    pub dropped: DropSummary,
    /// Per-row weights when policy requests them.
    pub weights: Option<&'a [f64]>,
}

impl SamplePlan {
    /// Compile a plan against tabular (IID) data. Lagged nodes are rejected.
    ///
    /// # Errors
    ///
    /// Empty request, lagged nodes, unknown variables, or empty selection.
    pub fn compile_tabular(
        data: &TabularData,
        request: &SampleRequest<'_>,
    ) -> Result<Self, DataError> {
        compile_inner(data, None, request, data.row_count(), 0)
    }

    /// Compile a plan against a regular time series (supports lagged nodes).
    ///
    /// # Errors
    ///
    /// Irregular series, empty request, unknown variables, or invalid lags.
    pub fn compile_timeseries(
        data: &TimeSeriesData,
        request: &SampleRequest<'_>,
    ) -> Result<Self, DataError> {
        if matches!(data.time_index().regularity, SamplingRegularity::Irregular) {
            return Err(DataError::InvalidArgument {
                message: "integer-lag SampleRequest requires regular time series".into(),
            });
        }
        let max_lag = max_lag_in_request(request);
        let lag_map = if max_lag > 0 || has_lagged(request) {
            Some(Arc::new(LagMap::with_reference(data.row_count(), max_lag, request.reference)?))
        } else {
            None
        };
        compile_inner(data, lag_map, request, data.row_count(), data.row_count() as u64)
    }

    /// Planned columns.
    #[must_use]
    pub fn columns(&self) -> &[PreparedColumn] {
        &self.columns
    }

    /// Cache key.
    #[must_use]
    pub fn cache_key(&self) -> &SampleCacheKey {
        &self.cache_key
    }

    /// Materialize into `workspace`.
    ///
    /// # Errors
    ///
    /// Missing columns, empty selection after policies, or type mismatches.
    pub fn prepare<'a>(
        &'a self,
        data: &impl TableView,
        storage_mask: Option<&crate::column::ValidityBitmap>,
        storage_weights: Option<&[f64]>,
        workspace: &'a mut SampleWorkspace,
        policy: &KernelPolicy,
    ) -> Result<PreparedSample<'a>, DataError> {
        let ncols = self.partitions.ncols();
        if ncols == 0 {
            return Err(DataError::InvalidArgument {
                message: "sample plan has no columns".into(),
            });
        }

        let candidates = resolve_candidates(self)?;
        let mut selected = Vec::with_capacity(candidates.len());
        for &row in &candidates {
            let row_usize = row as usize;
            if !row_passes_mask(storage_mask, self.row_selector.mask, row_usize) {
                continue;
            }
            match row_missing_status(data, &self.columns, self.lag_map.as_deref(), row)? {
                RowStatus::Ok => selected.push(row),
                RowStatus::Missing => match self.row_selector.missing {
                    MissingPolicy::CompleteCase => {}
                    MissingPolicy::ErrorOnMissing => {
                        return Err(DataError::IncompleteSeries {
                            id: None,
                            message: "missing values under ErrorOnMissing policy",
                        });
                    }
                },
            }
        }
        if selected.is_empty() {
            return Err(DataError::EmptySelection {
                context: "sample prepare after mask/missingness",
            });
        }

        let n = selected.len();
        let requested = candidates.len();
        workspace.prepare(n, ncols);
        workspace.row_indexes[..n].copy_from_slice(&selected);

        for (c, col) in self.columns.iter().enumerate() {
            let ColumnView::Float64(src) = data.column(col.variable)? else {
                return Err(DataError::TypeMismatch { id: col.variable, expected: "float64" });
            };
            // Complete-case / ErrorOnMissing already filtered `selected` via
            // `row_missing_status` on resolved raw rows (including lagged).
            for (i, &row) in selected.iter().enumerate() {
                let raw = resolve_raw_row(self.lag_map.as_deref(), col.lag, row)?;
                workspace.gather_indexes[i] = raw;
            }
            let dst = workspace.values.prepare_mut(n * ncols);
            let col_dst = &mut dst[c * n..(c + 1) * n];
            gather(
                policy,
                F64VectorView::contiguous(src.values.as_ref()),
                &workspace.gather_indexes[..n],
                col_dst,
            );
        }

        let weights = match self.weight_policy {
            WeightPolicy::Ignore => None,
            WeightPolicy::Unit => {
                workspace.weights[..n].fill(1.0);
                Some(&workspace.weights[..n])
            }
            WeightPolicy::Honor => {
                if let Some(w) = storage_weights {
                    for (i, &row) in selected.iter().enumerate() {
                        workspace.weights[i] = w[row as usize];
                    }
                } else {
                    workspace.weights[..n].fill(1.0);
                }
                Some(&workspace.weights[..n])
            }
        };

        let values = workspace.values.as_slice(n * ncols);
        let matrix = F64MatrixView::column_major(values, n, ncols).map_err(|_| {
            DataError::InvalidArgument { message: "prepared matrix shape rejected".into() }
        })?;

        Ok(PreparedSample {
            matrix,
            partitions: self.partitions,
            selected_rows: RowSelectionRef { indexes: &workspace.row_indexes[..n] },
            effective_n: n,
            dropped: DropSummary { requested, retained: n },
            weights,
        })
    }

    /// Prepare from [`TabularData`].
    ///
    /// # Errors
    ///
    /// Propagates [`Self::prepare`] errors.
    pub fn prepare_tabular<'a>(
        &'a self,
        data: &TabularData,
        workspace: &'a mut SampleWorkspace,
        policy: &KernelPolicy,
    ) -> Result<PreparedSample<'a>, DataError> {
        self.prepare(
            data,
            data.storage().analysis_mask(),
            data.storage().weights(),
            workspace,
            policy,
        )
    }

    /// Prepare from [`TimeSeriesData`].
    ///
    /// # Errors
    ///
    /// Propagates [`Self::prepare`] errors.
    pub fn prepare_timeseries<'a>(
        &'a self,
        data: &TimeSeriesData,
        workspace: &'a mut SampleWorkspace,
        policy: &KernelPolicy,
    ) -> Result<PreparedSample<'a>, DataError> {
        self.prepare(
            data,
            data.storage().analysis_mask(),
            data.storage().weights(),
            workspace,
            policy,
        )
    }
}

#[derive(Clone, Copy)]
enum RowStatus {
    Ok,
    Missing,
}

fn has_lagged(request: &SampleRequest<'_>) -> bool {
    request
        .x
        .iter()
        .chain(request.y.iter())
        .chain(request.z.iter())
        .any(|n| matches!(n, NodeRef::Lagged { .. }))
}

fn max_lag_in_request(request: &SampleRequest<'_>) -> u32 {
    request
        .x
        .iter()
        .chain(request.y.iter())
        .chain(request.z.iter())
        .map(|n| match n {
            NodeRef::Lagged { lag, .. } => lag.raw(),
            NodeRef::Static(_) | NodeRef::Context { .. } => 0,
        })
        .max()
        .unwrap_or(0)
}

fn node_to_prepared(node: NodeRef, role: u8) -> Result<PreparedColumn, DataError> {
    match node {
        NodeRef::Static(variable) => {
            Ok(PreparedColumn { variable, lag: Lag::CONTEMPORANEOUS, role })
        }
        NodeRef::Lagged { variable, lag } => Ok(PreparedColumn { variable, lag, role }),
        NodeRef::Context { variable, environment } => {
            if environment.is_some() {
                return Err(DataError::InvalidArgument {
                    message: "SampleRequest Context nodes with environment are not supported yet"
                        .into(),
                });
            }
            Ok(PreparedColumn { variable, lag: Lag::CONTEMPORANEOUS, role })
        }
    }
}

fn compile_inner(
    data: &impl TableView,
    lag_map: Option<Arc<LagMap>>,
    request: &SampleRequest<'_>,
    data_len: usize,
    time_index_version: u64,
) -> Result<SamplePlan, DataError> {
    let mut columns = Vec::new();
    let mut nodes_key = Vec::new();
    for (role, slice) in [(0u8, request.x), (1, request.y), (2, request.z)] {
        for &node in slice {
            if lag_map.is_none() {
                if let NodeRef::Lagged { .. } = node {
                    return Err(DataError::InvalidArgument {
                        message: "lagged nodes require TimeSeriesData".into(),
                    });
                }
            }
            let prep = node_to_prepared(node, role)?;
            // Ensure column exists and is float64 at compile time.
            let ColumnView::Float64(_) = data.column(prep.variable)? else {
                return Err(DataError::TypeMismatch { id: prep.variable, expected: "float64" });
            };
            nodes_key.push((prep.variable.raw(), prep.lag.raw(), role));
            columns.push(prep);
        }
    }
    if columns.is_empty() {
        return Err(DataError::InvalidArgument {
            message: "SampleRequest needs ≥1 node in x, y, or z".into(),
        });
    }

    let partitions =
        SamplePartitions { n_x: request.x.len(), n_y: request.y.len(), n_z: request.z.len() };

    let candidate_rows: Arc<[u32]> = if let Some(map) = lag_map.as_ref() {
        let n = map.n_effective();
        let base = map.row_index(Lag::CONTEMPORANEOUS, 0) as u32;
        Arc::from((0..n).map(|i| base + i as u32).collect::<Vec<_>>())
    } else {
        Arc::from((0..data_len as u32).collect::<Vec<_>>())
    };

    let reference_key = match request.reference {
        ReferencePointPolicy::SeriesOrigin => (0u8, 0u64),
        ReferencePointPolicy::AbsoluteOrigin { origin_row } => (1u8, origin_row as u64),
    };
    let policies = (
        match request.missing {
            MissingPolicy::CompleteCase => 0u8,
            MissingPolicy::ErrorOnMissing => 1u8,
        },
        match request.mask {
            MaskPolicy::Honor => 0u8,
            MaskPolicy::Ignore => 1u8,
        },
        match request.weights {
            WeightPolicy::Honor => 0u8,
            WeightPolicy::Unit => 1u8,
            WeightPolicy::Ignore => 2u8,
        },
    );

    Ok(SamplePlan {
        columns,
        row_selector: PreparedRowSelector {
            candidate_rows,
            missing: request.missing,
            mask: request.mask,
        },
        output_layout: SampleLayout::ColumnMajor,
        cache_key: SampleCacheKey {
            nodes: Arc::from(nodes_key),
            reference: reference_key,
            policies,
            data_len,
            time_index_version,
        },
        partitions,
        lag_map,
        weight_policy: request.weights,
    })
}

fn resolve_candidates(plan: &SamplePlan) -> Result<Vec<u32>, DataError> {
    if plan.row_selector.candidate_rows.is_empty() {
        return Err(DataError::EmptySelection { context: "sample plan candidates" });
    }
    Ok(plan.row_selector.candidate_rows.to_vec())
}

fn row_passes_mask(
    mask: Option<&crate::column::ValidityBitmap>,
    policy: MaskPolicy,
    row: usize,
) -> bool {
    match policy {
        MaskPolicy::Ignore => true,
        MaskPolicy::Honor => mask.is_none_or(|m| m.is_valid(row)),
    }
}

fn resolve_raw_row(lag_map: Option<&LagMap>, lag: Lag, base_row: u32) -> Result<usize, DataError> {
    let Some(map) = lag_map else {
        return Ok(base_row as usize);
    };
    // base_row is a contemporaneous raw row in the effective window.
    let base_t = map.row_index(Lag::CONTEMPORANEOUS, 0);
    if (base_row as usize) < base_t {
        return Err(DataError::InvalidArgument { message: "sample row before lag base".into() });
    }
    let sample_i = (base_row as usize) - base_t;
    if sample_i >= map.n_effective() {
        return Err(DataError::InvalidArgument { message: "sample row past effective n".into() });
    }
    Ok(map.row_index(lag, sample_i))
}

fn row_missing_status(
    data: &impl TableView,
    columns: &[PreparedColumn],
    lag_map: Option<&LagMap>,
    base_row: u32,
) -> Result<RowStatus, DataError> {
    for col in columns {
        let raw = resolve_raw_row(lag_map, col.lag, base_row)?;
        let view = data.column(col.variable)?;
        let validity = view.validity();
        if !validity.is_valid(raw) {
            return Ok(RowStatus::Missing);
        }
    }
    Ok(RowStatus::Ok)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use causal_core::{Lag, NodeRef, VariableId};

    use super::*;
    use crate::testing::float_series;

    #[test]
    fn tabular_complete_case_prepare() {
        let series = float_series(20, 2);
        let tabular = TabularData::new(series.storage().clone());
        let x = [NodeRef::Static(VariableId::from_raw(0))];
        let y = [NodeRef::Static(VariableId::from_raw(1))];
        let req = SampleRequest::new(&x, &y, &[]);
        let plan = SamplePlan::compile_tabular(&tabular, &req).unwrap();
        let mut ws = SampleWorkspace::default();
        let prep =
            plan.prepare_tabular(&tabular, &mut ws, &KernelPolicy::default_policy()).unwrap();
        assert_eq!(prep.effective_n, 20);
        assert_eq!(prep.partitions.ncols(), 2);
        assert_eq!(prep.matrix.nrows(), 20);
        assert_eq!(prep.matrix.ncols(), 2);
    }

    #[test]
    fn timeseries_lagged_request() {
        let data = float_series(30, 2);
        let x = [NodeRef::Lagged { variable: VariableId::from_raw(0), lag: Lag::from_raw(2) }];
        let y = [NodeRef::Lagged { variable: VariableId::from_raw(1), lag: Lag::CONTEMPORANEOUS }];
        let req = SampleRequest::new(&x, &y, &[]);
        let plan = SamplePlan::compile_timeseries(&data, &req).unwrap();
        let mut ws = SampleWorkspace::default();
        let prep =
            plan.prepare_timeseries(&data, &mut ws, &KernelPolicy::default_policy()).unwrap();
        assert_eq!(prep.effective_n, 28);
        assert!((prep.matrix.get(0, 0).unwrap() - 0.0).abs() < 1e-12);
        assert!((prep.matrix.get(0, 1).unwrap() - 102.0).abs() < 1e-12);
    }

    #[test]
    fn same_request_same_cache_key() {
        let data = float_series(10, 1);
        let x = [NodeRef::Static(VariableId::from_raw(0))];
        let req = SampleRequest::new(&x, &[], &[]);
        let a = SamplePlan::compile_timeseries(&data, &req).unwrap();
        let b = SamplePlan::compile_timeseries(&data, &req).unwrap();
        assert_eq!(a.cache_key, b.cache_key);
    }

    #[test]
    fn lagged_rejected_on_tabular() {
        let series = float_series(10, 1);
        let tabular = TabularData::new(series.storage().clone());
        let x = [NodeRef::Lagged { variable: VariableId::from_raw(0), lag: Lag::from_raw(1) }];
        let req = SampleRequest::new(&x, &[], &[]);
        assert!(SamplePlan::compile_tabular(&tabular, &req).is_err());
    }
}
