//! Design compilation helpers that expand categoricals into float columns.
//!
//! Contrast generation lives in `causal-data`; provenance records live on
//! [`causal_stats::CompiledDesign`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::{KernelPolicy, VariableId};
use causal_data::{
    CategoryCode, CategoryDomain, Contrast, ContrastMatrix, compile_contrast_matrix,
};
use causal_stats::{
    CompiledDesign, ContrastCodingKind, DesignColumn, DesignColumnMap, DesignColumnRole,
    RecordedContrast, StandardizationRecord, StandardizedColumn, standardize_columns, StatsError,
};

use crate::error::EstimationError;

/// One covariate input for [`compile_adjustment_design`].
#[derive(Clone, Debug)]
pub enum CovariateSpec<'a> {
    /// Continuous float column (already masked to analysis rows).
    Float {
        /// Variable id.
        id: VariableId,
        /// Values length = nrows.
        values: &'a [f64],
    },
    /// Dictionary categorical expanded via `contrast`.
    Categorical {
        /// Variable id.
        id: VariableId,
        /// Category codes length = nrows.
        codes: &'a [u32],
        /// Domain for levels / reference.
        domain: &'a CategoryDomain,
        /// Explicit contrast coding.
        contrast: Contrast,
    },
}

/// Compile `[1 | T | Z…]` with optional categorical contrasts and float standardization.
///
/// # Errors
///
/// Shape mismatches, contrast compilation failure, or empty design.
pub fn compile_adjustment_design(
    treatment: &[f64],
    covariates: &[CovariateSpec<'_>],
    outcome: &[f64],
    row_selection: &[usize],
    standardize_float_covariates: bool,
    policy: &KernelPolicy,
) -> Result<CompiledDesign, EstimationError> {
    let nrows = outcome.len();
    if nrows == 0 {
        return Err(EstimationError::from(StatsError::Shape { message: "empty design" }));
    }
    if treatment.len() != nrows {
        return Err(EstimationError::from(StatsError::Shape {
            message: "treatment length mismatch",
        }));
    }

    let selection: Arc<[usize]> = if row_selection.is_empty() {
        Arc::from((0..nrows).collect::<Vec<_>>())
    } else if row_selection.len() == nrows {
        Arc::from(row_selection.to_vec())
    } else {
        return Err(EstimationError::from(StatsError::Shape {
            message: "row_selection length mismatch",
        }));
    };

    let mut columns: Vec<DesignColumn> = vec![
        DesignColumn::from_role(DesignColumnRole::Intercept),
        DesignColumn::from_role(DesignColumnRole::Treatment),
    ];
    let mut matrix_cols: Vec<Vec<f64>> = Vec::new();
    // intercept
    matrix_cols.push(vec![1.0; nrows]);
    // treatment
    matrix_cols.push(treatment.to_vec());

    let mut contrasts: Vec<RecordedContrast> = Vec::new();
    let mut float_cov_col_idxs: Vec<usize> = Vec::new();

    for spec in covariates {
        match spec {
            CovariateSpec::Float { id, values } => {
                if values.len() != nrows {
                    return Err(EstimationError::from(StatsError::Shape {
                        message: "covariate length mismatch",
                    }));
                }
                float_cov_col_idxs.push(matrix_cols.len());
                matrix_cols.push(values.to_vec());
                columns.push(DesignColumn::from_role(DesignColumnRole::Covariate(*id)));
            }
            CovariateSpec::Categorical { id, codes, domain, contrast } => {
                if codes.len() != nrows {
                    return Err(EstimationError::from(StatsError::Shape {
                        message: "categorical codes length mismatch",
                    }));
                }
                let cm = compile_contrast_matrix(domain, contrast).map_err(EstimationError::from)?;
                let start = matrix_cols.len();
                let contrast_idx = contrasts.len();
                expand_contrast_columns(
                    &mut matrix_cols,
                    &mut columns,
                    *id,
                    codes,
                    &cm,
                    contrast_idx,
                )?;
                let end = matrix_cols.len();
                let labels: Arc<[Arc<str>]> =
                    Arc::from(domain.levels.iter().map(|l| Arc::clone(&l.label)).collect::<Vec<_>>());
                contrasts.push(RecordedContrast {
                    variable: *id,
                    coding: coding_kind(contrast),
                    level_labels: labels,
                    column_range: (start, end),
                    matrix: Arc::clone(&cm.values),
                });
            }
        }
    }

    let ncols = matrix_cols.len();
    let mut matrix = vec![0.0; nrows * ncols];
    for (c, col) in matrix_cols.into_iter().enumerate() {
        let base = c * nrows;
        matrix[base..base + nrows].copy_from_slice(&col);
    }

    let standardization = if standardize_float_covariates && !float_cov_col_idxs.is_empty() {
        standardize_columns(&mut matrix, nrows, ncols, &float_cov_col_idxs, 1e-12, policy)
            .map_err(EstimationError::from)?
    } else {
        StandardizationRecord::default()
    };

    let columns = DesignColumnMap::from(columns).with_standardization_links(&standardization);

    Ok(CompiledDesign {
        nrows,
        ncols,
        matrix: Arc::from(matrix),
        columns,
        outcome: Arc::from(outcome.to_vec()),
        row_selection: selection,
        contrasts,
        standardization,
        smooths: Vec::new(),
    })
}

fn coding_kind(contrast: &Contrast) -> ContrastCodingKind {
    match contrast {
        Contrast::Treatment { reference } => {
            ContrastCodingKind::Treatment { reference: reference.raw() }
        }
        Contrast::SumToZero => ContrastCodingKind::SumToZero,
        Contrast::Helmert => ContrastCodingKind::Helmert,
        Contrast::Polynomial => ContrastCodingKind::Polynomial,
        Contrast::FullRankIndicator => ContrastCodingKind::FullRankIndicator,
        Contrast::Custom(_) => ContrastCodingKind::Custom,
    }
}

fn expand_contrast_columns(
    matrix_cols: &mut Vec<Vec<f64>>,
    columns: &mut Vec<DesignColumn>,
    id: VariableId,
    codes: &[u32],
    cm: &ContrastMatrix,
    contrast_idx: usize,
) -> Result<(), EstimationError> {
    let nrows = codes.len();
    for c in 0..cm.n_columns {
        let mut col = vec![0.0; nrows];
        for (r, &code) in codes.iter().enumerate() {
            let level = code as usize;
            if level >= cm.n_levels {
                return Err(EstimationError::from(StatsError::Shape {
                    message: "category code out of domain range",
                }));
            }
            col[r] = cm.values[c * cm.n_levels + level];
        }
        matrix_cols.push(col);
        columns.push(DesignColumn {
            role: DesignColumnRole::Covariate(id),
            contrast_idx: Some(contrast_idx),
            standardization_idx: None,
            smooth_idx: None,
        });
    }
    Ok(())
}

/// Map a data-layer contrast reference helper for tests / callers.
#[must_use]
pub fn treatment_contrast_ref(reference: CategoryCode) -> Contrast {
    Contrast::Treatment { reference }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use causal_core::{CategoryDomainId, VariableId};
    use causal_data::{CategoryDomain, CategoryLevel, Contrast, compile_contrast_matrix};

    use super::*;

    fn two_level_domain() -> CategoryDomain {
        CategoryDomain::try_new(
            CategoryDomainId::from_raw(0),
            vec![
                CategoryLevel { label: Arc::from("a") },
                CategoryLevel { label: Arc::from("b") },
            ],
            false,
            Some(CategoryCode::from_raw(0)),
            causal_data::UnknownCategoryPolicy::Fail,
        )
        .unwrap()
    }

    #[test]
    fn categorical_covariate_records_contrast_provenance() {
        let t = vec![0.0, 1.0, 0.0, 1.0];
        let y = vec![1.0, 2.0, 1.5, 2.5];
        let codes = [0u32, 1, 0, 1];
        let domain = two_level_domain();
        let design = compile_adjustment_design(
            &t,
            &[CovariateSpec::Categorical {
                id: VariableId::from_raw(3),
                codes: &codes,
                domain: &domain,
                contrast: Contrast::Treatment { reference: CategoryCode::from_raw(0) },
            }],
            &y,
            &[],
            false,
            &KernelPolicy::default_policy(),
        )
        .unwrap();
        assert_eq!(design.ncols, 3); // intercept, T, one treatment dummy
        assert_eq!(design.contrasts.len(), 1);
        assert_eq!(design.contrasts[0].column_range, (2, 3));
        assert!(matches!(
            design.contrasts[0].coding,
            ContrastCodingKind::Treatment { reference: 0 }
        ));
        let expected = compile_contrast_matrix(
            &domain,
            &Contrast::Treatment { reference: CategoryCode::from_raw(0) },
        )
        .unwrap();
        assert_eq!(&*design.contrasts[0].matrix, &*expected.values);
        assert_eq!(design.columns.get(2).unwrap().contrast_idx, Some(0));
    }

    #[test]
    fn float_standardization_recorded() {
        let t = vec![0.0, 1.0, 0.0, 1.0];
        let z = vec![0.0, 2.0, 4.0, 6.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let design = compile_adjustment_design(
            &t,
            &[CovariateSpec::Float { id: VariableId::from_raw(2), values: &z }],
            &y,
            &[],
            true,
            &KernelPolicy::default_policy(),
        )
        .unwrap();
        assert_eq!(design.standardization.entries.len(), 1);
        let e: &StandardizedColumn = &design.standardization.entries[0];
        assert_eq!(e.col_idx, 2);
        assert!((e.mean - 3.0).abs() < 1e-12);
        // Column mean ≈ 0 after standardization.
        let base = 2 * design.nrows;
        let mean: f64 =
            design.matrix[base..base + design.nrows].iter().sum::<f64>() / design.nrows as f64;
        assert!(mean.abs() < 1e-12);
        assert_eq!(design.columns.get(2).unwrap().standardization_idx, Some(0));
    }
}
