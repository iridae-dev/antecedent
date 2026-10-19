//! Library-owned causal data views and storage.
//!
//! Arrow adapters are optional (`arrow` feature) and never leak Arrow types
//! into the public causal API (ADR 0004).
//!
//! # Building tabular data
//!
//! ```
//! use antecedent_data::{TableView, TabularData};
//!
//! let data = TabularData::from_f64_columns([
//!     ("x", &[1.0_f64, 2.0, 3.0][..]),
//! ]).unwrap();
//! assert_eq!(data.row_count(), 3);
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

pub mod aligned_buffer;
#[cfg(feature = "arrow")]
pub mod arrow_adapter;
#[cfg(feature = "arrow")]
pub mod arrow_ffi;
pub mod buffer;
pub mod categorical;
pub mod column;
mod column_closure;
mod content_identity;
pub mod dataset;
pub mod discrete;
pub mod error;
pub mod event;
pub mod global_index;
pub mod lagged_frame;
mod lazy_identity;
pub mod materialize;
pub mod missing;
pub mod multi_env;
pub mod network;
pub mod panel;
pub mod panel_window;
pub mod pooled_frame;
pub mod project;
pub mod reference;
pub mod resample;
pub mod sample;
pub mod sample_policy;
pub mod selection;
mod source;
mod source_snapshot;
pub mod split;
pub mod storage;
pub mod surrogate;
pub mod table;
pub mod temporal;
pub mod vector_vars;

#[cfg(test)]
mod differential;
#[cfg(test)]
mod testing;

#[cfg(feature = "arrow")]
pub use arrow_adapter::{ArrowLoadResult, tabular_from_arrow_c_columns, tabular_from_record_batch};
#[cfg(feature = "arrow")]
pub use arrow_ffi::{ArrowCColumn, FfiArrowArray, FfiArrowSchema};
pub use buffer::{F64Buffer, ForeignBufferOwner, ForeignF64Buffer};
pub use categorical::{
    CategoricalColumn, CategoricalView, CategoryCode, CategoryDomain, CategoryLevel, Contrast,
    ContrastMatrix, UnknownCategoryPolicy, compile_contrast_matrix,
};
pub use column::{
    BooleanColumn, ColumnView, FixedVectorColumn, Float64Column, Int64Column, OwnedColumn,
    TimestampColumn, ValidityBitmap,
};
pub use dataset::{TabularData, TimeSeriesData};
pub use discrete::DiscreteColumn;
pub use error::DataError;
pub use event::EventData;
pub use global_index::{ChunkIndex, GlobalId, GlobalIndexError};
pub use lagged_frame::{LaggedFrame, LaggedFrameOptions};
pub use missing::{
    DropReason, GroupAccounting, MissingCell, RowAccounting, UnitAccounting, complete_rows,
    dropped_row_digest, first_missing_cell, policy_required_text,
};
pub use multi_env::MultiEnvironmentData;
pub use network::{NetworkData, NetworkEdge};
pub use panel::{PanelData, PanelUnit, PanelUnitView};
pub use panel_window::{
    BorrowError, CalendarSpan, DeclaredSplit, LONGER_SPAN_OF_SAME_UNIT, LagRefusal, PanelWindow,
    PopulationUnit, UnitPartition, WindowBoundary, WindowChunk, WindowNextAttempt,
};
pub use pooled_frame::{
    DEFAULT_MAX_TIME_ONE_HOT_LEVELS, DummyOptions, PooledLaggedFrame, TimeDummyEncoding,
    pool_multi_env_lagged_frame,
};
pub use project::{IdRemap, dedupe_variable_ids};
pub use reference::ReferencePointPolicy;
pub use resample::{
    PermutationScheme, ResamplingPlan, circular_block_length, fill_circular_block_indexes,
    fill_resample_index_batch, fill_resample_indexes, resample_timeseries,
};
pub use sample::{
    DropSummary, GappedSample, LagMap, LaggedColumn, LaggedPreparedSample, LaggedSamplePlan,
    LaggedSampleWorkspace,
};
pub use sample_policy::{MaskPolicy, MissingDeclaration, MissingPolicy};
pub use split::{DiscoveryEstimationSplit, EnvHoldoutSplit, TimeRange};
pub use storage::OwnedColumnarStorage;
pub use surrogate::{surrogate_permute_columns, surrogate_phase_randomize};
pub use table::TableView;
pub use temporal::{SamplingRegularity, TemporalIndexer, TemporalNodeKey, TimeIndex};
pub use vector_vars::{VectorVariableGroups, column_blocks_for_frame};

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };

    use super::*;

    fn two_col_table() -> OwnedColumnarStorage {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let n = 1_000usize;
        let x = Float64Column::new(
            VariableId::from_raw(0),
            Arc::<[f64]>::from((0..n).map(|i| i as f64).collect::<Vec<_>>()),
            ValidityBitmap::all_valid(n),
        )
        .unwrap();
        let y = Float64Column::new(
            VariableId::from_raw(1),
            Arc::<[f64]>::from((0..n).map(|i| (i * 2) as f64).collect::<Vec<_>>()),
            ValidityBitmap::all_valid(n),
        )
        .unwrap();
        OwnedColumnarStorage::try_new(
            schema,
            vec![OwnedColumn::Float64(x), OwnedColumn::Float64(y)],
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn table_view_returns_columns_by_id() {
        let table = two_col_table();
        assert_eq!(table.row_count(), 1000);
        let col = table.column(VariableId::from_raw(0)).unwrap();
        assert_eq!(col.len(), 1000);
        match col {
            ColumnView::Float64(c) => {
                assert!((c.values[10] - 10.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected float64"),
        }
    }

    #[test]
    fn prepared_column_view_does_not_reallocate() {
        let table = two_col_table();
        let col = table.column(VariableId::from_raw(0)).unwrap();
        let ColumnView::Float64(c) = col else {
            panic!("expected float");
        };
        let ptr = c.values.as_ptr();
        for _ in 0..100 {
            let again = table.column(VariableId::from_raw(0)).unwrap();
            let ColumnView::Float64(c2) = again else {
                panic!("expected float");
            };
            assert_eq!(c2.values.as_ptr(), ptr);
            let view = c2.as_f64_view();
            assert_eq!(view.len(), 1000);
        }
    }

    #[test]
    fn timeseries_wraps_storage() {
        let storage = two_col_table();
        let ts = TimeSeriesData::try_new(
            storage,
            TimeIndex {
                regularity: SamplingRegularity::Regular { interval_ns: 1_000 },
                length: 1000,
            },
        )
        .unwrap();
        assert_eq!(ts.row_count(), 1000);
    }
}
