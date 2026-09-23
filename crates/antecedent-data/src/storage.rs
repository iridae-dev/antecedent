//! Owned tabular storage implementing [`TableView`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalSchema, VariableId};

use crate::column::{ColumnView, OwnedColumn};
use crate::error::DataError;
use crate::table::TableView;

/// Owned columnar table with optional analysis mask and weights.
#[derive(Clone, Debug)]
pub struct OwnedColumnarStorage {
    schema: CausalSchema,
    columns: Arc<[OwnedColumn]>,
    row_count: usize,
    /// Optional analysis inclusion mask (1 = include), independent of validity.
    analysis_mask: Option<crate::column::ValidityBitmap>,
    /// Optional observation weights.
    weights: Option<Arc<[f64]>>,
    content_digest: [u8; 32],
}

impl OwnedColumnarStorage {
    /// Build a table from schema-aligned columns.
    ///
    /// # Errors
    ///
    /// Length mismatches, missing schema variables, duplicate columns, or weights that
    /// are not finite and non-negative with positive total over the analysis rows.
    pub fn try_new(
        schema: CausalSchema,
        columns: Vec<OwnedColumn>,
        analysis_mask: Option<crate::column::ValidityBitmap>,
        weights: Option<Arc<[f64]>>,
    ) -> Result<Self, DataError> {
        if columns.len() != schema.len() {
            return Err(DataError::LengthMismatch {
                expected: schema.len(),
                actual: columns.len(),
                context: "column count vs schema",
            });
        }
        let row_count = columns.first().map_or(0, OwnedColumn::len);
        for (i, col) in columns.iter().enumerate() {
            let expected_id = VariableId::from_raw(u32::try_from(i).map_err(|_| {
                DataError::InvalidArgument { message: "schema exceeds VariableId range".into() }
            })?);
            if col.id() != expected_id {
                return Err(DataError::UnknownVariable { id: col.id() });
            }
            if col.len() != row_count {
                return Err(DataError::LengthMismatch {
                    expected: row_count,
                    actual: col.len(),
                    context: "column row count",
                });
            }
        }
        if let Some(mask) = &analysis_mask {
            if mask.len() != row_count {
                return Err(DataError::LengthMismatch {
                    expected: row_count,
                    actual: mask.len(),
                    context: "analysis mask",
                });
            }
        }
        if let Some(w) = &weights {
            if w.len() != row_count {
                return Err(DataError::LengthMismatch {
                    expected: row_count,
                    actual: w.len(),
                    context: "weights",
                });
            }
            validate_weights(w, analysis_mask.as_ref())?;
        }
        let content_digest = crate::content_identity::storage_digest(
            &columns,
            row_count,
            analysis_mask.as_ref(),
            weights.as_deref(),
        );
        Ok(Self {
            schema,
            columns: Arc::from(columns),
            row_count,
            analysis_mask,
            weights,
            content_digest,
        })
    }

    /// Version-1 BLAKE3 digest of typed column contents, validity, mask, and weights.
    ///
    /// Computed once at construction without copying cell buffers; this accessor
    /// and storage clones never rescan rows. The enclosing snapshot must also bind
    /// schema, modality, temporal metadata, and unit/environment partitions.
    ///
    /// Encoding uses the `antecedent.data.storage.v1` derive-key domain, little-endian
    /// integers, exact float bits, length-prefixed strings, and dense column/row
    /// order. Validity padding is ignored; missing payloads and explicit optional
    /// mask/weight presence are retained. Equality is representational, not a claim
    /// of equivalent statistical meaning or proof against hash collisions.
    #[must_use]
    pub const fn content_digest(&self) -> [u8; 32] {
        self.content_digest
    }

    /// Optional analysis mask.
    #[must_use]
    pub fn analysis_mask(&self) -> Option<&crate::column::ValidityBitmap> {
        self.analysis_mask.as_ref()
    }

    /// Optional weights.
    #[must_use]
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// Borrow owned columns in dense id order.
    #[must_use]
    pub fn columns(&self) -> &[OwnedColumn] {
        &self.columns
    }

    /// Shared column Arc (identity for copy-avoidance checks).
    #[must_use]
    pub fn columns_arc(&self) -> &Arc<[OwnedColumn]> {
        &self.columns
    }
}

/// Observation weights must be a finite, non-negative measure with positive
/// total mass over the analysis rows; anything else makes weighted means and
/// least squares meaningless (NaN, or sign-flipped contributions).
fn validate_weights(
    weights: &[f64],
    analysis_mask: Option<&crate::column::ValidityBitmap>,
) -> Result<(), DataError> {
    let mut total = 0.0_f64;
    let mut analysis_rows = 0usize;
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(DataError::InvalidWeights { index: Some(i), reason: "not finite" });
        }
        if w < 0.0 {
            return Err(DataError::InvalidWeights { index: Some(i), reason: "negative" });
        }
        if analysis_mask.is_none_or(|m| m.is_valid(i)) {
            total += w;
            analysis_rows += 1;
        }
    }
    if analysis_rows > 0 && (total <= 0.0 || !total.is_finite()) {
        return Err(DataError::InvalidWeights {
            index: None,
            reason: "total weight over the analysis rows is not positive and finite",
        });
    }
    Ok(())
}

impl TableView for OwnedColumnarStorage {
    fn schema(&self) -> &CausalSchema {
        &self.schema
    }

    fn row_count(&self) -> usize {
        self.row_count
    }

    fn column(&self, id: VariableId) -> Result<ColumnView<'_>, DataError> {
        self.columns
            .get(id.as_usize())
            .map(OwnedColumn::as_view)
            .ok_or(DataError::UnknownVariable { id })
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };

    use super::*;
    use crate::column::{Float64Column, ValidityBitmap};

    fn build(
        weights: Vec<f64>,
        mask: Option<ValidityBitmap>,
    ) -> Result<OwnedColumnarStorage, DataError> {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "v0".to_owned(),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let n = weights.len();
        let col = OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(vec![1.0; n]),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        );
        OwnedColumnarStorage::try_new(schema, vec![col], mask, Some(Arc::from(weights)))
    }

    #[test]
    fn weights_must_be_finite_and_non_negative() {
        for (bad, reason) in [
            (f64::NAN, "not finite"),
            (f64::INFINITY, "not finite"),
            (f64::NEG_INFINITY, "not finite"),
            (-0.5, "negative"),
        ] {
            let err = build(vec![1.0, bad, 2.0], None).unwrap_err();
            assert_eq!(err, DataError::InvalidWeights { index: Some(1), reason });
        }
    }

    #[test]
    fn weights_need_positive_mass_over_analysis_rows() {
        assert!(matches!(
            build(vec![0.0, 0.0, 0.0], None),
            Err(DataError::InvalidWeights { index: None, .. })
        ));
        // Row 0 is the only analysis row (LSB-first) and it carries zero weight.
        let mask = ValidityBitmap::from_bytes(vec![0b001u8], 3).unwrap();
        assert!(matches!(
            build(vec![0.0, 5.0, 5.0], Some(mask)),
            Err(DataError::InvalidWeights { index: None, .. })
        ));
        // Zero weights on some rows are legitimate (Bayesian-bootstrap draws).
        assert!(build(vec![0.0, 1.0, 3.0], None).is_ok());
    }
}
