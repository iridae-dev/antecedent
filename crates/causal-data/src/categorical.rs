//! Dictionary categoricals and explicit contrasts (ADR 0003).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::CategoryDomainId;

use crate::column::ValidityBitmap;
use crate::error::DataError;

/// Dictionary-encoded category code (`u32`). Never treated as a magnitude.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CategoryCode(u32);

impl CategoryCode {
    /// Construct from raw code.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Underlying code.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// One level in a category domain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CategoryLevel {
    /// Stable level label.
    pub label: Arc<str>,
}

/// Policy for unseen category codes at inference time.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum UnknownCategoryPolicy {
    /// Fail when an unknown code appears.
    Fail,
    /// Map to a declared `Other` level (must exist in the domain).
    MapToOther {
        /// Code of the Other level.
        other: CategoryCode,
    },
}

/// Immutable category domain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CategoryDomain {
    /// Domain id within a schema registry.
    pub id: CategoryDomainId,
    /// Ordered levels.
    pub levels: Arc<[CategoryLevel]>,
    /// Whether levels are ordered (ordinal).
    pub ordered: bool,
    /// Optional default reference level for treatment coding.
    pub reference: Option<CategoryCode>,
    /// Unknown-code policy.
    pub unknown_policy: UnknownCategoryPolicy,
}

impl CategoryDomain {
    /// Construct a domain.
    ///
    /// # Errors
    ///
    /// Empty levels, invalid reference, or invalid Other mapping.
    pub fn try_new(
        id: CategoryDomainId,
        levels: impl Into<Arc<[CategoryLevel]>>,
        ordered: bool,
        reference: Option<CategoryCode>,
        unknown_policy: UnknownCategoryPolicy,
    ) -> Result<Self, DataError> {
        let levels = levels.into();
        if levels.is_empty() {
            return Err(DataError::InvalidValidity {
                message: "category domain requires at least one level",
            });
        }
        let n = u32::try_from(levels.len())
            .map_err(|_| DataError::InvalidValidity { message: "too many category levels" })?;
        if let Some(r) = reference {
            if r.raw() >= n {
                return Err(DataError::InvalidValidity { message: "reference code out of range" });
            }
        }
        if let UnknownCategoryPolicy::MapToOther { other } = unknown_policy {
            if other.raw() >= n {
                return Err(DataError::InvalidValidity { message: "Other code out of range" });
            }
        }
        Ok(Self { id, levels, ordered, reference, unknown_policy })
    }

    /// Number of levels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// Whether empty (always false after construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }
}

/// Owned categorical column (codes + domain + validity).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CategoricalColumn {
    /// Variable id (dense).
    pub id: causal_core::VariableId,
    /// Codes.
    pub codes: Arc<[CategoryCode]>,
    /// Validity (missing ≠ a category).
    pub validity: ValidityBitmap,
    /// Domain.
    pub domain: Arc<CategoryDomain>,
}

impl CategoricalColumn {
    /// Construct a categorical column.
    ///
    /// Out-of-range codes fail under [`UnknownCategoryPolicy::Fail`] and are
    /// remapped to the domain's `Other` level under
    /// [`UnknownCategoryPolicy::MapToOther`].
    ///
    /// # Errors
    ///
    /// Length mismatch or out-of-range codes under Fail policy.
    pub fn try_new(
        id: causal_core::VariableId,
        codes: impl Into<Arc<[CategoryCode]>>,
        validity: ValidityBitmap,
        domain: Arc<CategoryDomain>,
    ) -> Result<Self, DataError> {
        let codes = codes.into();
        if validity.len() != codes.len() {
            return Err(DataError::LengthMismatch {
                expected: codes.len(),
                actual: validity.len(),
                context: "categorical validity",
            });
        }
        let n_levels = u32::try_from(domain.len()).expect("checked");
        let mut remapped: Option<Vec<CategoryCode>> = None;
        for (i, code) in codes.iter().enumerate() {
            if !validity.is_valid(i) {
                continue;
            }
            if code.raw() >= n_levels {
                match domain.unknown_policy {
                    UnknownCategoryPolicy::Fail => {
                        return Err(DataError::InvalidValidity {
                            message: "unknown category code under Fail policy",
                        });
                    }
                    UnknownCategoryPolicy::MapToOther { other } => {
                        // `other` is validated in-range at domain construction.
                        remapped.get_or_insert_with(|| codes.to_vec())[i] = other;
                    }
                }
            }
        }
        let codes = remapped.map_or(codes, Arc::from);
        Ok(Self { id, codes, validity, domain })
    }

    /// Row count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Borrowed view.
    #[must_use]
    pub fn as_view(&self) -> CategoricalView<'_> {
        CategoricalView { codes: &self.codes, validity: &self.validity, domain: &self.domain }
    }
}

/// Borrowed categorical view.
#[derive(Clone, Copy, Debug)]
pub struct CategoricalView<'a> {
    /// Codes.
    pub codes: &'a [CategoryCode],
    /// Validity.
    pub validity: &'a ValidityBitmap,
    /// Domain.
    pub domain: &'a CategoryDomain,
}

/// Explicit contrast coding.
#[derive(Clone, Debug, PartialEq)]
pub enum Contrast {
    /// Treatment coding relative to a reference level.
    Treatment {
        /// Reference category.
        reference: CategoryCode,
    },
    /// Sum-to-zero.
    SumToZero,
    /// Helmert.
    Helmert,
    /// Polynomial.
    Polynomial,
    /// Full-rank indicator (no drop).
    FullRankIndicator,
    /// Caller-supplied contrast matrix (`levels × columns`).
    Custom(ContrastMatrix),
}

/// Dense contrast matrix in column-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastMatrix {
    /// Number of levels (rows).
    pub n_levels: usize,
    /// Number of generated columns.
    pub n_columns: usize,
    /// Column-major entries.
    pub values: Arc<[f64]>,
}

impl ContrastMatrix {
    /// Construct a matrix.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn try_new(
        n_levels: usize,
        n_columns: usize,
        values: impl Into<Arc<[f64]>>,
    ) -> Result<Self, DataError> {
        let values = values.into();
        if values.len()
            != n_levels
                .checked_mul(n_columns)
                .ok_or(DataError::InvalidValidity { message: "contrast shape overflow" })?
        {
            return Err(DataError::LengthMismatch {
                expected: n_levels * n_columns,
                actual: values.len(),
                context: "contrast matrix",
            });
        }
        Ok(Self { n_levels, n_columns, values })
    }
}

/// Compile a contrast into a design matrix for valid rows.
///
/// Builds the contrast matrix for all contrast variants used by /1 design
/// compilation (`Treatment`, `SumToZero`, `Helmert`, `Polynomial`,
/// `FullRankIndicator`, `Custom`).
///
/// # Errors
///
/// Unsupported domain/contrast combination or invalid reference.
pub fn compile_contrast_matrix(
    domain: &CategoryDomain,
    contrast: &Contrast,
) -> Result<ContrastMatrix, DataError> {
    let k = domain.len();
    match contrast {
        Contrast::FullRankIndicator => {
            let mut values = vec![0.0; k * k];
            for i in 0..k {
                values[i + i * k] = 1.0;
            }
            ContrastMatrix::try_new(k, k, values)
        }
        Contrast::Treatment { reference } => {
            if reference.raw() as usize >= k {
                return Err(DataError::InvalidValidity {
                    message: "treatment reference out of range",
                });
            }
            let cols = k - 1;
            let mut values = vec![0.0; k * cols];
            let mut col = 0usize;
            for level in 0..k {
                if level == reference.raw() as usize {
                    continue;
                }
                values[level + col * k] = 1.0;
                col += 1;
            }
            ContrastMatrix::try_new(k, cols, values)
        }
        Contrast::Custom(m) => {
            if m.n_levels != k {
                return Err(DataError::LengthMismatch {
                    expected: k,
                    actual: m.n_levels,
                    context: "custom contrast levels",
                });
            }
            Ok(m.clone())
        }
        Contrast::SumToZero => {
            // k levels → k-1 columns; last level is -1 on every column.
            let cols = k - 1;
            let mut values = vec![0.0; k * cols];
            for c in 0..cols {
                values[c + c * k] = 1.0;
                values[(k - 1) + c * k] = -1.0;
            }
            ContrastMatrix::try_new(k, cols, values)
        }
        Contrast::Helmert => {
            // Forward Helmert: each column contrasts a level with the mean of subsequent levels.
            let cols = k - 1;
            let mut values = vec![0.0; k * cols];
            for c in 0..cols {
                let remaining = k - c;
                let weight = 1.0 / remaining as f64;
                values[c + c * k] = 1.0 - weight;
                for r in (c + 1)..k {
                    values[r + c * k] = -weight;
                }
            }
            ContrastMatrix::try_new(k, cols, values)
        }
        Contrast::Polynomial => {
            if !domain.ordered {
                return Err(DataError::InvalidValidity {
                    message: "polynomial contrasts require an ordered domain",
                });
            }
            // Orthogonal polynomial contrasts on equally spaced scores 1..=k
            // (R's `contr.poly`): Vandermonde basis, then modified Gram-Schmidt
            // of each degree against all lower degrees (including the constant),
            // then unit-length normalization.
            let cols = k - 1;
            let mut basis = vec![0.0; k * k];
            for d in 0..k {
                for r in 0..k {
                    basis[r + d * k] = ((r + 1) as f64).powi(i32::try_from(d).expect("small k"));
                }
            }
            for d in 0..k {
                for p in 0..d {
                    // Lower columns are already unit-norm, so the projection
                    // coefficient is a plain dot product.
                    let dot: f64 = (0..k).map(|r| basis[r + d * k] * basis[r + p * k]).sum();
                    for r in 0..k {
                        basis[r + d * k] -= dot * basis[r + p * k];
                    }
                }
                let norm: f64 = (0..k).map(|r| basis[r + d * k].powi(2)).sum::<f64>().sqrt();
                if norm > 0.0 {
                    for r in 0..k {
                        basis[r + d * k] /= norm;
                    }
                }
            }
            // Drop the constant column: degrees 1..k-1.
            ContrastMatrix::try_new(k, cols, basis[k..].to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{CategoryDomainId, VariableId};

    fn domain() -> Arc<CategoryDomain> {
        Arc::new(
            CategoryDomain::try_new(
                CategoryDomainId::from_raw(0),
                Arc::<[CategoryLevel]>::from(vec![
                    CategoryLevel { label: Arc::from("a") },
                    CategoryLevel { label: Arc::from("b") },
                    CategoryLevel { label: Arc::from("c") },
                ]),
                false,
                Some(CategoryCode::from_raw(0)),
                UnknownCategoryPolicy::Fail,
            )
            .unwrap(),
        )
    }

    #[test]
    fn treatment_contrast_drops_reference() {
        let d = domain();
        let m = compile_contrast_matrix(
            &d,
            &Contrast::Treatment { reference: CategoryCode::from_raw(0) },
        )
        .unwrap();
        assert_eq!(m.n_levels, 3);
        assert_eq!(m.n_columns, 2);
    }

    #[test]
    fn sum_to_zero_and_helmert_shapes() {
        let d = domain();
        let s = compile_contrast_matrix(&d, &Contrast::SumToZero).unwrap();
        assert_eq!((s.n_levels, s.n_columns), (3, 2));
        let h = compile_contrast_matrix(&d, &Contrast::Helmert).unwrap();
        assert_eq!((h.n_levels, h.n_columns), (3, 2));
    }

    #[test]
    fn polynomial_requires_ordered_domain() {
        let d = domain();
        let err = compile_contrast_matrix(&d, &Contrast::Polynomial).unwrap_err();
        assert!(matches!(err, DataError::InvalidValidity { .. }));
        let ordered = Arc::new(
            CategoryDomain::try_new(
                CategoryDomainId::from_raw(0),
                Arc::<[CategoryLevel]>::from(vec![
                    CategoryLevel { label: Arc::from("a") },
                    CategoryLevel { label: Arc::from("b") },
                    CategoryLevel { label: Arc::from("c") },
                ]),
                true,
                Some(CategoryCode::from_raw(0)),
                UnknownCategoryPolicy::Fail,
            )
            .unwrap(),
        );
        let m = compile_contrast_matrix(&ordered, &Contrast::Polynomial).unwrap();
        assert_eq!(m.n_columns, 2);
    }

    fn ordered_domain(k: usize) -> Arc<CategoryDomain> {
        let levels: Vec<CategoryLevel> =
            (0..k).map(|i| CategoryLevel { label: Arc::from(format!("l{i}")) }).collect();
        Arc::new(
            CategoryDomain::try_new(
                CategoryDomainId::from_raw(0),
                Arc::<[CategoryLevel]>::from(levels),
                true,
                None,
                UnknownCategoryPolicy::Fail,
            )
            .unwrap(),
        )
    }

    #[test]
    fn polynomial_matches_contr_poly_for_k3() {
        let m = compile_contrast_matrix(&ordered_domain(3), &Contrast::Polynomial).unwrap();
        let expected = [
            -std::f64::consts::FRAC_1_SQRT_2,
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
            0.408_248_290_463_863,
            -0.816_496_580_927_726,
            0.408_248_290_463_863,
        ];
        for (got, want) in m.values.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn polynomial_columns_orthonormal_for_k4_and_k5() {
        for k in [4usize, 5] {
            let m = compile_contrast_matrix(&ordered_domain(k), &Contrast::Polynomial).unwrap();
            assert_eq!((m.n_levels, m.n_columns), (k, k - 1));
            for a in 0..m.n_columns {
                let norm: f64 = (0..k).map(|r| m.values[r + a * k].powi(2)).sum();
                assert!((norm - 1.0).abs() < 1e-12, "k={k} column {a} norm {norm}");
                // Orthogonal to the constant (columns sum to zero).
                let sum: f64 = (0..k).map(|r| m.values[r + a * k]).sum();
                assert!(sum.abs() < 1e-12, "k={k} column {a} sum {sum}");
                for b in (a + 1)..m.n_columns {
                    let dot: f64 = (0..k).map(|r| m.values[r + a * k] * m.values[r + b * k]).sum();
                    assert!(dot.abs() < 1e-12, "k={k} columns {a},{b} dot {dot}");
                }
            }
        }
    }

    #[test]
    fn map_to_other_remaps_out_of_range_codes() {
        let d = Arc::new(
            CategoryDomain::try_new(
                CategoryDomainId::from_raw(0),
                Arc::<[CategoryLevel]>::from(vec![
                    CategoryLevel { label: Arc::from("a") },
                    CategoryLevel { label: Arc::from("b") },
                    CategoryLevel { label: Arc::from("other") },
                ]),
                false,
                None,
                UnknownCategoryPolicy::MapToOther { other: CategoryCode::from_raw(2) },
            )
            .unwrap(),
        );
        let col = CategoricalColumn::try_new(
            VariableId::from_raw(0),
            Arc::<[CategoryCode]>::from(vec![
                CategoryCode::from_raw(1),
                CategoryCode::from_raw(9),
                CategoryCode::from_raw(0),
            ]),
            ValidityBitmap::all_valid(3),
            d,
        )
        .unwrap();
        assert_eq!(col.codes[0], CategoryCode::from_raw(1));
        assert_eq!(col.codes[1], CategoryCode::from_raw(2));
        assert_eq!(col.codes[2], CategoryCode::from_raw(0));
        // Every stored code now indexes the domain safely.
        assert!(col.codes.iter().all(|c| (c.raw() as usize) < col.domain.len()));
    }

    #[test]
    fn categorical_rejects_unknown_under_fail() {
        let d = domain();
        let err = CategoricalColumn::try_new(
            VariableId::from_raw(0),
            Arc::<[CategoryCode]>::from(vec![CategoryCode::from_raw(9)]),
            ValidityBitmap::all_valid(1),
            d,
        )
        .unwrap_err();
        assert!(matches!(err, DataError::InvalidValidity { .. }));
    }

    #[test]
    fn missing_skips_code_validation() {
        let d = domain();
        let bytes = vec![0u8];
        // invalid bit 0
        let col = CategoricalColumn::try_new(
            VariableId::from_raw(0),
            Arc::<[CategoryCode]>::from(vec![CategoryCode::from_raw(9)]),
            ValidityBitmap::from_bytes(bytes, 1).unwrap(),
            d,
        )
        .unwrap();
        assert_eq!(col.len(), 1);
        assert!(!col.validity.is_valid(0));
    }
}
