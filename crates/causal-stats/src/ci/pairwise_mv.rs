//! Pairwise multivariate CI wrapper.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::ExecutionContext;

use super::parcorr_variants::MultivariatePartialCorrelation;
use super::types::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, SignificanceMethod,
};
use crate::error::StatsError;

/// Pairwise multivariate wrapper: for each scalar query `(x,y|Z)`, expands endpoints
/// that belong to a configured column block and runs
/// [`MultivariatePartialCorrelation::test_blocks`].
///
/// Empty [`Self::column_blocks`] ⇒ singleton `{x}` / `{y}` / unchanged `Z` (scalar `ParCorr`).
#[derive(Clone, Debug)]
pub struct PairwiseMultivariateCi {
    inner: MultivariatePartialCorrelation,
    /// Column-membership blocks. If a query column appears in a block, that endpoint
    /// (or Z member) expands to the full block.
    pub column_blocks: Arc<[Arc<[usize]>]>,
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
        Self { inner: MultivariatePartialCorrelation::new(), column_blocks: Arc::from([]) }
    }

    /// Construct with column-membership blocks (pinned baseline `vector_vars` style).
    #[must_use]
    pub fn with_column_blocks(column_blocks: Arc<[Arc<[usize]>]>) -> Self {
        Self { inner: MultivariatePartialCorrelation::new(), column_blocks }
    }

    /// Legacy alias for [`Self::with_column_blocks`] using only the X-side lists as blocks.
    #[must_use]
    pub fn with_blocks(x_blocks: Arc<[Arc<[usize]>]>, _y_blocks: Arc<[Arc<[usize]>]>) -> Self {
        Self::with_column_blocks(x_blocks)
    }

    fn expand_endpoint(&self, col: usize) -> Vec<usize> {
        for block in self.column_blocks.iter() {
            if block.iter().any(|&c| c == col) {
                return block.to_vec();
            }
        }
        vec![col]
    }

    fn expand_z(&self, z: &[usize]) -> Vec<usize> {
        if self.column_blocks.is_empty() {
            return z.to_vec();
        }
        let mut out = Vec::with_capacity(z.len());
        for &c in z {
            for m in self.expand_endpoint(c) {
                if !out.contains(&m) {
                    out.push(m);
                }
            }
        }
        out
    }
}

impl ConditionalIndependenceTest for PairwiseMultivariateCi {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        let mut results = Vec::with_capacity(request.queries.len());
        for q in request.queries {
            let z_raw = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let x_cols = self.expand_endpoint(q.x);
            let y_cols = self.expand_endpoint(q.y);
            let z = self.expand_z(z_raw);
            let r = self.inner.test_blocks(
                request.columns,
                &x_cols,
                &y_cols,
                &z,
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

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::ci::types::{CiQuery, ConfidenceMethod};

    #[test]
    fn column_block_expands_x_endpoint() {
        let n = 400usize;
        // Shared latent drives two dummy dims and y; single dim alone is weak.
        let mut d0 = vec![0.0; n];
        let mut d1 = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let z = (i as f64 * 0.017).sin();
            d0[i] = z + 0.05 * ((i % 7) as f64);
            d1[i] = 0.8 * z + 0.05 * ((i % 11) as f64);
            y[i] = 1.2 * z + 0.05 * ((i % 13) as f64);
        }
        let cols: [&[f64]; 3] = [&d0, &d1, &y];
        let queries = [CiQuery { x: 0, y: 2, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);

        let scalar = PairwiseMultivariateCi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        let block = PairwiseMultivariateCi::with_column_blocks(Arc::from([Arc::from([0usize, 1])]))
            .test_batch_adhoc(&req, &mut ws, &ctx)
            .unwrap();
        assert!(
            block.results[0].p_value <= scalar.results[0].p_value,
            "block p={} scalar p={}",
            block.results[0].p_value,
            scalar.results[0].p_value
        );
        assert!(
            block.results[0].p_value < 1e-3,
            "block should detect dependence p={}",
            block.results[0].p_value
        );
    }

    #[test]
    fn column_block_expands_z_conditioner() {
        let n = 300usize;
        let mut d0 = vec![0.0; n];
        let mut d1 = vec![0.0; n];
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let z = (i as f64 * 0.02).cos();
            d0[i] = z;
            d1[i] = 0.9 * z;
            x[i] = z + 0.05 * ((i % 5) as f64);
            y[i] = z + 0.05 * ((i % 9) as f64);
        }
        let cols: [&[f64]; 4] = [&x, &y, &d0, &d1];
        // Condition only on d0; block expands to {d0,d1} and should kill the dependence.
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(4);
        let out = PairwiseMultivariateCi::with_column_blocks(Arc::from([Arc::from([2usize, 3])]))
            .test_batch_adhoc(&req, &mut ws, &ctx)
            .unwrap();
        assert!(
            out.results[0].p_value > 0.05,
            "expanded Z should screen x⊥y; p={}",
            out.results[0].p_value
        );
    }
}
