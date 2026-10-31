//! Optional full-source witness of an analysis-result artifact.
//!
//! The section carries the complete original table of a tabular analysis so a
//! later verifier can replay a fit from the artifact alone. It is bound to the
//! contract by the data-snapshot partition digest: the witness is accepted
//! only when the storage rebuilt from it has exactly the content digest the
//! snapshot records. The section embeds every column of the table, including
//! columns the analysis did not use, and may expose sensitive rows.
//!
//! Only finite, all-valid `Float64` columns with no analysis mask and no
//! weights are carried. Nothing is normalised: a table outside that shape is
//! refused instead of being rewritten into one the digest would then describe
//! differently.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use antecedent_core::{CausalSchemaBuilder, MeasurementSpec, SmallRoleSet, ValueType, VariableId};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage};
use serde::{Deserialize, Serialize};

use crate::{DataSnapshotIdentityWire, IoError};

/// Artifact section id of the full-source witness.
pub const SOURCE_WITNESS_SECTION: &str = "analysis_result.source_witness";

/// Wire format of [`SourceWitnessWire`].
pub const SOURCE_WITNESS_FORMAT: u16 = 1;

/// Bytes one stored value occupies once rebuilt as storage.
const VALUE_BYTES: u64 = 8;

/// One column of the witnessed table.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SourceColumnWire {
    /// Dense variable id of the column.
    pub variable: u32,
    /// Every cell in row order; all finite.
    pub values: Vec<f64>,
}

/// The complete original table of a tabular analysis.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SourceWitnessWire {
    /// Witness format ([`SOURCE_WITNESS_FORMAT`]).
    pub format: u16,
    /// Rows of the table.
    pub row_count: u64,
    /// Columns in dense variable-id order.
    pub columns: Vec<SourceColumnWire>,
}

/// Bounds a reader holds a witness to before it allocates a table for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceWitnessLimits {
    /// Largest encoded section, and largest rebuilt table, in bytes.
    pub max_bytes: u64,
    /// Most rows a witness may declare.
    pub max_rows: u64,
    /// Most columns a witness may declare.
    pub max_columns: u64,
}

impl Default for SourceWitnessLimits {
    fn default() -> Self {
        Self { max_bytes: 256 * 1024 * 1024, max_rows: 50_000_000, max_columns: 4096 }
    }
}

/// Why a source witness was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceWitnessError {
    /// The table is outside the shape a witness can carry; the text names what.
    Unsupported(&'static str),
    /// A declared or computed size exceeds the limits (or overflows).
    TooLarge,
    /// The witness is not a well-formed table; the text names the fault.
    Malformed(&'static str),
    /// The witness disagrees with the snapshot it must bind to.
    Mismatch(&'static str),
}

impl fmt::Display for SourceWitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "source witness cannot carry {what}"),
            Self::TooLarge => write!(f, "source witness exceeds its size limits"),
            Self::Malformed(what) => write!(f, "malformed source witness: {what}"),
            Self::Mismatch(what) => write!(f, "source witness does not match the snapshot: {what}"),
        }
    }
}

impl std::error::Error for SourceWitnessError {}

impl From<SourceWitnessError> for IoError {
    fn from(error: SourceWitnessError) -> Self {
        Self::Convert(error.to_string())
    }
}

impl SourceWitnessWire {
    /// Witness of a table the format can carry exactly.
    ///
    /// # Errors
    ///
    /// [`SourceWitnessError::Unsupported`] for an analysis mask, weights, an
    /// invalid cell, a column that is not `Float64`, a non-finite value or a
    /// table without columns; [`SourceWitnessError::TooLarge`] when a count
    /// does not fit its wire integer.
    pub fn from_storage(storage: &OwnedColumnarStorage) -> Result<Self, SourceWitnessError> {
        if storage.analysis_mask().is_some() {
            return Err(SourceWitnessError::Unsupported("an analysis mask"));
        }
        if storage.weights().is_some() {
            return Err(SourceWitnessError::Unsupported("observation weights"));
        }
        if storage.columns().is_empty() {
            return Err(SourceWitnessError::Unsupported("a table without columns"));
        }
        let mut columns = Vec::with_capacity(storage.columns().len());
        let mut row_count = 0_usize;
        for column in storage.columns() {
            let OwnedColumn::Float64(column) = column else {
                return Err(SourceWitnessError::Unsupported("a column that is not Float64"));
            };
            if !column.validity.is_all_valid() {
                return Err(SourceWitnessError::Unsupported("an invalid (missing) cell"));
            }
            let values = column.values.as_slice();
            if values.iter().any(|value| !value.is_finite()) {
                return Err(SourceWitnessError::Unsupported("a non-finite value"));
            }
            row_count = values.len();
            columns.push(SourceColumnWire { variable: column.id.raw(), values: values.to_vec() });
        }
        let row_count = u64::try_from(row_count).map_err(|_| SourceWitnessError::TooLarge)?;
        Ok(Self { format: SOURCE_WITNESS_FORMAT, row_count, columns })
    }

    /// Check the declared sizes against `limits` without reading any value.
    ///
    /// # Errors
    ///
    /// [`SourceWitnessError::Malformed`] for an unknown format;
    /// [`SourceWitnessError::TooLarge`] when the rows, the columns or the
    /// table they imply exceed the limits, or when the product overflows.
    pub fn check_declared(&self, limits: &SourceWitnessLimits) -> Result<(), SourceWitnessError> {
        if self.format != SOURCE_WITNESS_FORMAT {
            return Err(SourceWitnessError::Malformed("unsupported format"));
        }
        let columns =
            u64::try_from(self.columns.len()).map_err(|_| SourceWitnessError::TooLarge)?;
        if self.row_count > limits.max_rows || columns > limits.max_columns {
            return Err(SourceWitnessError::TooLarge);
        }
        let bytes = self
            .row_count
            .checked_mul(columns)
            .and_then(|cells| cells.checked_mul(VALUE_BYTES))
            .ok_or(SourceWitnessError::TooLarge)?;
        if bytes > limits.max_bytes {
            return Err(SourceWitnessError::TooLarge);
        }
        Ok(())
    }

    /// Rebuild the witnessed table under a schema of continuous variables
    /// named `variable_names`.
    ///
    /// The schema's role hints are not part of the storage content digest, so
    /// the default role is used for every variable.
    ///
    /// # Errors
    ///
    /// [`SourceWitnessError::TooLarge`] before any table is allocated when the
    /// declared sizes exceed `limits`; [`SourceWitnessError::Malformed`] for
    /// ids that are not exactly `0..k` ascending with `k` the number of names,
    /// a column whose length is not the row count, a non-finite value or names
    /// the schema refuses.
    pub fn to_storage(
        &self,
        variable_names: &[String],
        limits: &SourceWitnessLimits,
    ) -> Result<OwnedColumnarStorage, SourceWitnessError> {
        self.check_declared(limits)?;
        if self.columns.is_empty() {
            return Err(SourceWitnessError::Malformed("no columns"));
        }
        if self.columns.len() != variable_names.len() {
            return Err(SourceWitnessError::Malformed("column count differs from variable names"));
        }
        let rows = usize::try_from(self.row_count).map_err(|_| SourceWitnessError::TooLarge)?;
        for (index, column) in self.columns.iter().enumerate() {
            let expected = u32::try_from(index).map_err(|_| SourceWitnessError::TooLarge)?;
            if column.variable != expected {
                return Err(SourceWitnessError::Malformed(
                    "variable ids must be 0..k in ascending order",
                ));
            }
            if column.values.len() != rows {
                return Err(SourceWitnessError::Malformed("column length differs from row count"));
            }
            if column.values.iter().any(|value| !value.is_finite()) {
                return Err(SourceWitnessError::Malformed("non-finite value"));
            }
        }
        let mut builder = CausalSchemaBuilder::new();
        for name in variable_names {
            builder
                .add_variable(
                    Arc::<str>::from(name.as_str()),
                    ValueType::Continuous,
                    SmallRoleSet::empty(),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .map_err(|_| SourceWitnessError::Malformed("variable names refused by schema"))?;
        }
        let schema =
            builder.build().map_err(|_| SourceWitnessError::Malformed("schema refused"))?;
        let mut columns = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            let built = Float64Column::new(
                VariableId::from_raw(column.variable),
                Arc::<[f64]>::from(column.values.as_slice()),
                ValidityBitmap::all_valid(rows),
            )
            .map_err(|_| SourceWitnessError::Malformed("column refused by storage"))?;
            columns.push(OwnedColumn::Float64(built));
        }
        OwnedColumnarStorage::try_new(schema, columns, None, None)
            .map_err(|_| SourceWitnessError::Malformed("table refused by storage"))
    }
}

/// Accept `witness` only when it rebuilds the exact table `snapshot` records.
///
/// The rebuilt storage's content digest, the existing storage digest, must
/// equal the snapshot's single partition digest, and the row counts must
/// agree. Observation and schema-role identity are not checked here.
///
/// # Errors
///
/// Any [`SourceWitnessError`] from rebuilding the table, and
/// [`SourceWitnessError::Mismatch`] for a snapshot without exactly one
/// partition, a different row count or a different content digest.
pub fn verify_source_witness(
    witness: &SourceWitnessWire,
    variable_names: &[String],
    snapshot: &DataSnapshotIdentityWire,
    limits: &SourceWitnessLimits,
) -> Result<(), SourceWitnessError> {
    let storage = witness.to_storage(variable_names, limits)?;
    let [partition] = snapshot.partitions.as_slice() else {
        return Err(SourceWitnessError::Mismatch("snapshot must have exactly one partition"));
    };
    if partition.row_count != witness.row_count {
        return Err(SourceWitnessError::Mismatch("row count"));
    }
    if storage.content_digest() != partition.content {
        return Err(SourceWitnessError::Mismatch("content digest"));
    }
    Ok(())
}
