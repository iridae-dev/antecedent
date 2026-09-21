//! Fold-aware preprocessing contract. Learned transforms fit inside each
//! nuisance-training fold.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop)]

use antecedent_core::ExecutionContext;

use crate::design::{DesignView, RowSelection};
use crate::error::LearnError;

/// Unfitted transformer.
pub trait TransformerFactory: Send + Sync {
    /// Fit on the selected training rows of `x`.
    ///
    /// # Errors
    ///
    /// Shape or backend failure.
    fn fit(
        &self,
        x: DesignView<'_>,
        rows: RowSelection<'_>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedTransformer>, LearnError>;
}

/// Fitted transformer. Applies to any compatible design view.
pub trait FittedTransformer: Send + Sync {
    /// Transform `x` into a new owned column-major buffer and view it.
    ///
    /// The returned buffer is packed column-major with the same logical shape
    /// as `x` unless the transformer expands features.
    ///
    /// # Errors
    ///
    /// Shape or backend failure.
    fn transform(
        &self,
        x: DesignView<'_>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<f64>, usize, usize), LearnError>;
}

/// Stateless identity. Fit ignores rows; transform copies logical rows.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl TransformerFactory for Identity {
    fn fit(
        &self,
        _x: DesignView<'_>,
        _rows: RowSelection<'_>,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedTransformer>, LearnError> {
        Ok(Box::new(IdentityTransform))
    }
}

struct IdentityTransform;

impl FittedTransformer for IdentityTransform {
    fn transform(
        &self,
        x: DesignView<'_>,
        _ctx: &ExecutionContext,
    ) -> Result<(Vec<f64>, usize, usize), LearnError> {
        copy_logical_colmajor(x)
    }
}

/// Stateless `ln(1 + ·)` applied to every entry except an exact all-ones first column.
/// Does not fit on data.
///
/// A constant first column of exactly `1.0` is the design's intercept: it passes through
/// unchanged. Transforming it would turn it into `ln 2`, after which no learner recognises
/// an intercept column and an elastic net would penalize it as a slope.
#[derive(Clone, Copy, Debug, Default)]
pub struct Log1p;

impl TransformerFactory for Log1p {
    fn fit(
        &self,
        _x: DesignView<'_>,
        _rows: RowSelection<'_>,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedTransformer>, LearnError> {
        Ok(Box::new(Log1pTransform))
    }
}

struct Log1pTransform;

impl FittedTransformer for Log1pTransform {
    fn transform(
        &self,
        x: DesignView<'_>,
        _ctx: &ExecutionContext,
    ) -> Result<(Vec<f64>, usize, usize), LearnError> {
        let (mut values, nrows, ncols) = copy_logical_colmajor(x)?;
        let skip = usize::from(antecedent_stats::first_col_is_exact_ones(&values, nrows));
        for v in values.iter_mut().skip(skip * nrows) {
            // NaN must be refused explicitly: `v < 0` alone lets it through.
            if v.is_nan() || *v < 0.0 {
                return Err(LearnError::Unsupported {
                    message: "log1p requires non-negative, non-NaN entries",
                });
            }
            // `ln_1p` keeps precision for small entries where `(v + 1).ln()` rounds to 0.
            *v = v.ln_1p();
        }
        Ok((values, nrows, ncols))
    }
}

fn copy_logical_colmajor(x: DesignView<'_>) -> Result<(Vec<f64>, usize, usize), LearnError> {
    let nrows = x.nrows();
    let ncols = x.ncols();
    let mut out = vec![0.0; nrows.saturating_mul(ncols)];
    for c in 0..ncols {
        for r in 0..nrows {
            out[c * nrows + r] = x.get(r, c)?;
        }
    }
    Ok((out, nrows, ncols))
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn transform_all(values: &[f64], nrows: usize, ncols: usize) -> Result<Vec<f64>, LearnError> {
        let view = DesignView::from_column_major(values, nrows, ncols).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        Log1pTransform.transform(view, &ctx).map(|(v, _, _)| v)
    }

    /// The intercept survives, small entries keep their precision, and NaN is refused.
    #[test]
    fn log1p_keeps_the_intercept_uses_ln_1p_and_refuses_nan() {
        // Column 0 is all ones (intercept); column 1 holds a tiny entry, where
        // `(1e-20 + 1).ln()` is exactly 0 but `ln_1p(1e-20)` is 1e-20.
        let values = [1.0, 1.0, 1.0, 1e-20, 3.0, 0.0];
        let out = transform_all(&values, 3, 2).unwrap();
        assert_eq!(&out[..3], &[1.0, 1.0, 1.0]);
        assert_eq!(out[3], 1e-20);
        assert!((out[4] - 4.0_f64.ln()).abs() < 1e-15);
        assert_eq!(out[5], 0.0);

        // Without an intercept every column is transformed.
        let no_intercept = transform_all(&[2.0, 2.0], 2, 1).unwrap();
        assert!((no_intercept[0] - 3.0_f64.ln()).abs() < 1e-15);

        assert!(transform_all(&[1.0, f64::NAN], 2, 1).is_err());
        assert!(transform_all(&[1.0, -0.5], 2, 1).is_err());
    }
}
