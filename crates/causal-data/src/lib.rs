//! Library-owned causal data views and storage.
//!
//! Arrow adapters are optional (`arrow` feature) and never leak Arrow types
//! into the public causal API (ADR 0004).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "arrow")]
pub mod arrow_adapter;
pub mod categorical;
pub mod column;
pub mod dataset;
pub mod error;
pub mod lagged_frame;
pub mod materialize;
pub mod multi_env;
pub mod reference;
pub mod resample;
pub mod sample;
pub mod selection;
pub mod sim;
pub mod split;
pub mod storage;
pub mod table;
pub mod temporal;
pub mod transforms;

#[cfg(test)]
mod testing;

#[cfg(feature = "arrow")]
pub use arrow_adapter::{ArrowLoadResult, tabular_from_record_batch};
pub use categorical::{
    CategoricalColumn, CategoricalView, CategoryCode, CategoryDomain, CategoryLevel, Contrast,
    ContrastMatrix, UnknownCategoryPolicy, compile_contrast_matrix,
};
pub use column::{
    BooleanColumn, ColumnView, FixedVectorColumn, Float64Column, Int64Column, OwnedColumn,
    TimestampColumn, ValidityBitmap,
};
pub use dataset::{TabularData, TimeSeriesData};
pub use error::DataError;
pub use lagged_frame::LaggedFrame;
pub use materialize::{MaterializationReason, materialization_diagnostic};
pub use multi_env::MultiEnvironmentData;
pub use reference::ReferencePointPolicy;
pub use resample::{ResamplingPlan, fill_resample_indexes, resample_timeseries};
pub use sample::{DropSummary, LagMap, LaggedColumn, PreparedSample, SamplePlan, SampleWorkspace};
pub use sim::{KnownLaggedParent, LaggedLinearPair};
pub use split::{DiscoveryEstimationSplit, TimeRange};
pub use storage::OwnedColumnarStorage;
pub use table::TableView;
pub use temporal::{SamplingRegularity, TemporalIndexer, TemporalNodeKey, TimeIndex};
pub use transforms::{equal_width_bin, moving_average, ordinal_patterns};

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
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
