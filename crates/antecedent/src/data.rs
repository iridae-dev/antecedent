//! Data containers a study reads: tabular, panel, time-series, event and network data, typed
//! columns, missingness declarations and the Arrow boundary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Isolated Arrow C Data Interface FFI.
pub use antecedent_data::ArrowCColumn;
/// Data-layer errors.
pub use antecedent_data::DataError;
/// Split strategies for discovery / estimation.
pub use antecedent_data::EnvHoldoutSplit;
/// Contiguous f64 column buffers — owned or foreign-backed.
pub use antecedent_data::F64Buffer;
/// `antecedent-data` items from `global_index.rs`.
pub use antecedent_data::GlobalId;
/// Lag-aligned sample planning for temporal discovery.
pub use antecedent_data::LaggedColumn;
/// Owned tabular storage implementing `TableView`.
pub use antecedent_data::OwnedColumnarStorage;
/// Row accounting for declared missing-data handling.
pub use antecedent_data::RowAccounting;
/// `TableView` trait — public causal table API (ADR 0004).
pub use antecedent_data::TableView;
/// Pooled multi-environment lagged frames with space/time dummies (J-PCMCI+).
pub use antecedent_data::TimeDummyEncoding;
/// Column conversion: Arrow arrays and record batches to owned library columns.
pub use antecedent_data::tabular_from_arrow_c_columns;
/// Columnar storage and typed column views.
pub use antecedent_data::{
    BooleanColumn, Float64Column, Int64Column, OwnedColumn, TimestampColumn, ValidityBitmap,
};
/// Dictionary categoricals and explicit contrasts (ADR 0003).
pub use antecedent_data::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, UnknownCategoryPolicy,
};
/// `antecedent-data` items at the crate root.
pub use antecedent_data::{FfiArrowArray, FfiArrowSchema};
/// Sample construction policies.
pub use antecedent_data::{
    ImputationDomain, ImputationModel, ImputationSpec, MissingDeclaration, MissingPolicy,
    MissingnessAssumption,
};
/// Temporal sampling metadata (series length / regularity).
pub use antecedent_data::{SamplingRegularity, TimeIndex};
/// Arrow-backed adapters.
pub mod arrow_adapter {
    pub use antecedent_data::arrow_adapter::ArrowFrameView;
}
/// Isolated Arrow C Data Interface FFI.
pub mod arrow_ffi {
    pub use antecedent_data::arrow_ffi::{ArrowCStream, FfiArrowArrayStream};
}
