//! Owned tabular storage implementing [`TableView`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, OnceLock};

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
    /// Memoised on first read. Shared across clones: every clone views the same
    /// immutable columns, mask, and weights, so one computation serves all.
    content_digest: Arc<OnceLock<[u8; 32]>>,
    /// Lifetime guard: the admission that funded the bytes this table retains, held until
    /// the last clone of the storage (and of every table derived from it by
    /// [`Self::derive`]) drops. Not part of the table's value: no identity, digest or
    /// comparison reads it.
    retention: Option<Arc<antecedent_core::execution::AdmissionLease>>,
}

impl OwnedColumnarStorage {
    /// Build a table from schema-aligned columns.
    ///
    /// # Errors
    ///
    /// Length mismatches, missing schema variables, duplicate columns, or weights that
    /// are not finite and non-negative with positive total over the analysis rows.
    /// The row count is the column length. It is not capped at `2^32 - 1`.
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
            let expected_id = VariableId::from_raw(
                u32::try_from(i)
                    .map_err(|_| DataError::overflow("schema variable count as VariableId"))?,
            );
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
        if let Some(mask) = &analysis_mask
            && mask.len() != row_count
        {
            return Err(DataError::LengthMismatch {
                expected: row_count,
                actual: mask.len(),
                context: "analysis mask",
            });
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
        Ok(Self {
            schema,
            columns: Arc::from(columns),
            row_count,
            analysis_mask,
            weights,
            content_digest: Arc::new(OnceLock::new()),
            retention: None,
        })
    }

    /// This table holding `lease` until it, its clones and the tables [`Self::derive`]d from
    /// it have all dropped, whatever the column types. Attach before publishing the table:
    /// an alias made earlier does not hold the lease.
    #[must_use]
    pub fn with_retention(
        mut self,
        lease: Arc<antecedent_core::execution::AdmissionLease>,
    ) -> Self {
        self.retention = Some(lease);
        self
    }

    /// Whether an admission lease is held for this table's retained bytes.
    #[must_use]
    pub const fn is_retained(&self) -> bool {
        self.retention.is_some()
    }

    /// A table built from this one's columns, mask or weights (a reweighting, a mask, a
    /// projection of the same buffers). As [`Self::try_new`], and the result holds this
    /// table's retention lease, so the bytes stay funded while any derived view shares them.
    ///
    /// # Errors
    ///
    /// As [`Self::try_new`].
    pub fn derive(
        &self,
        schema: CausalSchema,
        columns: Vec<OwnedColumn>,
        analysis_mask: Option<crate::column::ValidityBitmap>,
        weights: Option<Arc<[f64]>>,
    ) -> Result<Self, DataError> {
        let mut derived = Self::try_new(schema, columns, analysis_mask, weights)?;
        derived.retention.clone_from(&self.retention);
        Ok(derived)
    }

    /// The same columns, mask and weights under another schema (a rename, a role or a unit
    /// change). The content digest does not cover the schema, so a source snapshot binds it
    /// separately.
    ///
    /// # Errors
    ///
    /// A schema whose variable count is not the column count.
    pub fn with_schema(&self, schema: CausalSchema) -> Result<Self, DataError> {
        self.derive(schema, self.columns.to_vec(), self.analysis_mask.clone(), self.weights.clone())
    }

    /// Version-1 BLAKE3 digest of typed column contents, validity, mask, and weights.
    ///
    /// Computed on first use without copying cell buffers and memoised (storage is
    /// immutable, so the value equals what construction would have produced);
    /// later calls and clones never rescan rows. The enclosing snapshot must also bind
    /// schema, modality, temporal metadata, and unit/environment partitions.
    ///
    /// Encoding uses the `antecedent.data.storage.v1` derive-key domain, little-endian
    /// integers, exact float bits, length-prefixed strings, and dense column/row
    /// order. Validity padding is ignored; missing payloads and explicit optional
    /// mask/weight presence are retained. Equality is representational, not a claim
    /// of equivalent statistical meaning or proof against hash collisions.
    #[must_use]
    pub fn content_digest(&self) -> [u8; 32] {
        *self.content_digest.get_or_init(|| {
            crate::content_identity::storage_digest(
                &self.columns,
                self.row_count,
                self.analysis_mask.as_ref(),
                self.weights.as_deref(),
            )
        })
    }

    /// Compute or reuse the content identity with the caller's admission and cancellation.
    ///
    /// # Errors
    /// Cancellation while streaming the storage contents.
    pub fn content_digest_with_context(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<[u8; 32], antecedent_core::execution::AdmissionError> {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_core::execution::AdmissionError::Cancelled);
        }
        if let Some(digest) = self.content_digest.get() {
            return Ok(*digest);
        }
        let digest = crate::content_identity::storage_digest_with_context(
            &self.columns,
            self.row_count,
            self.analysis_mask.as_ref(),
            self.weights.as_deref(),
            ctx,
        )?;
        let _ = self.content_digest.set(digest);
        Ok(digest)
    }

    /// Whether the digest has been computed yet (laziness probe).
    #[cfg(test)]
    pub(crate) fn digest_is_computed(&self) -> bool {
        self.content_digest.get().is_some()
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

    #[test]
    fn a_retention_guard_lives_exactly_as_long_as_the_storage_and_its_derivations() {
        use antecedent_core::ExecutionContext;
        use antecedent_core::execution::AdmissionPool;

        let pool = AdmissionPool::fixed(1, 64);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let lease = Arc::new(ctx.acquire_admission(0, 16).unwrap());
        let storage = build(vec![1.0; 3], None).unwrap().with_retention(lease);
        assert!(storage.is_retained());
        assert_eq!(pool.available().bytes(), 48);

        // A rebuild that shares the columns keeps the guard; a plain `try_new` does not.
        let derived = storage
            .derive(storage.schema().clone(), storage.columns().to_vec(), None, None)
            .unwrap();
        assert!(derived.is_retained());
        drop(storage);
        assert_eq!(pool.available().bytes(), 48);
        drop(derived);
        assert_eq!(pool.available().bytes(), 64);
    }
}
