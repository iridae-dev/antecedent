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

/// Stateless `ln(1 + ·)` applied to every entry. Does not fit on data.
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
        for v in &mut values {
            if *v < 0.0 {
                return Err(LearnError::Unsupported {
                    message: "log1p requires non-negative entries",
                });
            }
            *v = (*v + 1.0).ln();
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
