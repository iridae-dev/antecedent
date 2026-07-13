//! CI request / result types and workspace.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::ExecutionContext;
use causal_kernels::ParCorrWorkspace;

use crate::error::StatsError;

/// Reusable kNN index + permutation plan for CMI.
#[derive(Clone, Debug, Default)]
pub struct KnnCmiWorkspace {
    /// Built neighbor index (reused across queries when geometry unchanged).
    pub index_generation: u64,
    /// Last built feature dim.
    pub last_dim: usize,
    /// Last n.
    pub last_n: usize,
    /// Permutation plan (row indexes).
    pub perm: Vec<usize>,
    /// Distance scratch.
    pub distances: Vec<f64>,
}

/// Significance method for a CI statistic.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SignificanceMethod {
    /// Analytic Fisher-z / Student-t for partial correlation.
    Analytic,
    /// Block-shuffle null distribution.
    BlockShuffle {
        /// Number of null replicates.
        replicates: u32,
        /// Block length for shuffling.
        block_size: usize,
    },
}

/// One CI query over column indexes into a shared matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CiQuery {
    /// X column index.
    pub x: usize,
    /// Y column index.
    pub y: usize,
    /// Start into flat conditioning indexes.
    pub z_start: usize,
    /// Conditioning arity.
    pub z_len: usize,
}

/// Batch of CI queries (deterministic output order).
#[derive(Clone, Debug)]
pub struct CiBatchRequest<'a> {
    /// Column-major / list of equal-length float columns.
    pub columns: &'a [&'a [f64]],
    /// Queries.
    pub queries: &'a [CiQuery],
    /// Flat conditioning column indexes.
    pub z_flat: &'a [usize],
    /// Significance.
    pub significance: SignificanceMethod,
}

/// One CI result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CiResult {
    /// Test statistic (partial correlation for partial-correlation CI).
    pub statistic: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Residual degrees of freedom (analytic path).
    pub df: f64,
    /// Optional analytic confidence interval `(lower, upper)` for the statistic.
    pub ci: Option<(f64, f64)>,
}

/// Batch results aligned with request queries.
#[derive(Clone, Debug, Default)]
pub struct CiBatchResult {
    /// Per-query results.
    pub results: Vec<CiResult>,
}

/// Conditional independence test.
pub trait ConditionalIndependence {
    /// Evaluate a batch of queries.
    ///
    /// # Errors
    ///
    /// Shape / numerical failures.
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError>;
}

impl CiBatchRequest<'_> {
    /// Validate non-empty equal-length columns; returns `n`.
    ///
    /// # Errors
    ///
    /// Empty column list or length mismatch.
    pub fn nrows(&self) -> Result<usize, StatsError> {
        if self.columns.is_empty() {
            return Err(StatsError::Shape { message: "no columns" });
        }
        let n = self.columns[0].len();
        for col in self.columns {
            if col.len() != n {
                return Err(StatsError::Shape { message: "column length mismatch" });
            }
        }
        Ok(n)
    }
}

/// Shared scratch for CI batches.
#[derive(Clone, Debug, Default)]
pub struct CiWorkspace {
    /// Partial-correlation residualization workspace.
    pub parcorr: ParCorrWorkspace,
    /// Temporary statistic buffer.
    pub stats: Vec<Option<f64>>,
    /// Block-shuffle column scratch.
    pub shuffled: Vec<f64>,
    /// Block starts for shuffle / reusable permutation plan.
    pub block_perm: Vec<usize>,
    /// kNN CMI index / permutation reuse state.
    pub knn: KnnCmiWorkspace,
}

impl CiWorkspace {
    /// Prepare for `n_queries` results.
    pub fn prepare_queries(&mut self, n_queries: usize) {
        if self.stats.len() < n_queries {
            self.stats.resize(n_queries, None);
        }
    }
}
