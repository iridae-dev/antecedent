//! Data containers a study reads: tabular, panel, time-series, event and network data, typed
//! columns, missingness declarations and the Arrow boundary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
/// Columnar storage and typed column views.
pub use antecedent_data::{
    BooleanColumn, Float64Column, Int64Column, OwnedColumn, TimestampColumn, ValidityBitmap,
};
/// Dictionary categoricals and explicit contrasts (ADR 0003).
pub use antecedent_data::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, UnknownCategoryPolicy,
};
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
/// Import from the Arrow C Data Interface: one entry point from exporter columns to a table.
///
/// The FFI structs are the capsule layouts a host moves an exporter's arrays out of; only the
/// types needed to build an [`ArrowCColumn`] or an [`ArrowCStream`] are published.
pub mod arrow_c {
    pub use antecedent_data::arrow_ffi::{ArrowCColumn, ArrowCStream, FfiArrowArrayStream};
    pub use antecedent_data::{ArrowLoadResult, FfiArrowArray, FfiArrowSchema};

    /// Copy exporter columns into library-owned storage (borrowing contiguous float64 value
    /// buffers where the exporter allows), joining a column's chunks into one buffer.
    ///
    /// # Errors
    /// Empty input, an unsupported type, a C Data Interface or schema failure, a category or
    /// timestamp that cannot be kept exact, or more than `u32::MAX` columns.
    pub fn load_columns(
        columns: Vec<ArrowCColumn>,
    ) -> Result<ArrowLoadResult, antecedent_data::DataError> {
        antecedent_data::tabular_from_arrow_c_columns(columns)
    }
}

/// `data` holding `retention`, so the admitted memory is released only when the last holder of
/// the table drops.
#[must_use]
pub fn with_retention(
    data: antecedent_data::TabularData,
    retention: &crate::execution::Retention,
) -> antecedent_data::TabularData {
    data.with_retention(retention.shared_lease())
}

/// `buffer` holding `retention` for as long as any view of the buffer lives.
#[must_use]
pub fn buffer_with_retention(
    buffer: F64Buffer,
    retention: &crate::execution::Retention,
) -> F64Buffer {
    buffer.with_admission(retention.shared_lease())
}
