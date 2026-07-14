//! Pairwise multivariate CI wrapper (DESIGN.md §12).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::ExecutionContext;

use super::parcorr_variants::MultivariatePartialCorrelation;
use super::types::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    SignificanceMethod,
};
use crate::error::StatsError;

/// Pairwise multivariate wrapper: for each scalar query `(x,y|Z)`, treats configured
/// multivariate blocks containing `x` and `y` (or falls back to singleton blocks) and
/// runs [`MultivariatePartialCorrelation::test_blocks`].
#[derive(Clone, Debug)]
pub struct PairwiseMultivariateCi {
    inner: MultivariatePartialCorrelation,
    /// Optional explicit X-blocks / Y-blocks (column index lists). When empty, each
    /// query uses singleton `{x}` / `{y}`.
    pub x_blocks: Arc<[Arc<[usize]>]>,
    /// Y blocks aligned with [`Self::x_blocks`] (or empty for singleton Y).
    pub y_blocks: Arc<[Arc<[usize]>]>,
}

impl Default for PairwiseMultivariateCi {
    fn default() -> Self {
        Self::new()
    }
}

impl PairwiseMultivariateCi {
    /// Singleton-block wrapper (equivalent to scalar `ParCorr`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: MultivariatePartialCorrelation::new(),
            x_blocks: Arc::from([]),
            y_blocks: Arc::from([]),
        }
    }

    /// Construct with explicit paired blocks.
    #[must_use]
    pub fn with_blocks(x_blocks: Arc<[Arc<[usize]>]>, y_blocks: Arc<[Arc<[usize]>]>) -> Self {
        Self { inner: MultivariatePartialCorrelation::new(), x_blocks, y_blocks }
    }
}

impl ConditionalIndependenceTest for PairwiseMultivariateCi {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        let mut results = Vec::with_capacity(request.queries.len());
        for (qi, q) in request.queries.iter().enumerate() {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let (x_cols, y_cols) = if self.x_blocks.is_empty() {
                (vec![q.x], vec![q.y])
            } else {
                let i = qi.min(self.x_blocks.len().saturating_sub(1));
                let xb = self.x_blocks[i].to_vec();
                let yb = if self.y_blocks.is_empty() {
                    vec![q.y]
                } else {
                    self.y_blocks[i.min(self.y_blocks.len() - 1)].to_vec()
                };
                (xb, yb)
            };
            let r = self.inner.test_blocks(
                request.columns,
                &x_cols,
                &y_cols,
                z,
                request.significance,
                workspace,
                ctx,
            )?;
            results.push(r);
        }
        Ok(CiBatchResult { results })
    }
}

/// Convenience: one block-pair test.
pub fn pairwise_multivariate_test(
    columns: &[&[f64]],
    x_cols: &[usize],
    y_cols: &[usize],
    z_flat: &[usize],
    significance: SignificanceMethod,
    workspace: &mut CiWorkspace,
    ctx: &ExecutionContext,
) -> Result<CiResult, StatsError> {
    MultivariatePartialCorrelation::new().test_blocks(
        columns,
        x_cols,
        y_cols,
        z_flat,
        significance,
        workspace,
        ctx,
    )
}
