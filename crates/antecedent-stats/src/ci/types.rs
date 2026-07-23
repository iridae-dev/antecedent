//! CI request / result types and workspace.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_kernels::ParCorrWorkspace;

use crate::error::StatsError;

/// Reusable kNN index + permutation plan for CMI.
#[derive(Clone, Debug, Default)]
pub struct KnnDependenceWorkspace {
    /// Built neighbor index generation (bumps only on rebuild).
    pub index_generation: u64,
    /// Number of times a new [`MatchingIndex`] was constructed.
    pub index_builds: u32,
    /// Last built feature dim.
    pub last_dim: usize,
    /// Last n.
    pub last_n: usize,
    /// Fingerprint of the (x, y, z) inputs behind the cached index (pointer, length,
    /// and sampled-content hash), so different pairs in one batch rebuild correctly.
    pub last_fingerprint: u64,
    /// Cached joint features (row-major `n * dim`).
    pub features: Vec<f64>,
    /// Cached nearest-neighbor index over [`Self::features`].
    pub index: Option<crate::matching::MatchingIndex>,
    /// Permutation plan (row indexes) reused for null shuffles.
    pub perm: Vec<usize>,
    /// Distance scratch (per-query kth distances).
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

/// Confidence interval method for a CI statistic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConfidenceMethod {
    /// No interval.
    None,
    /// Analytic Fisher-z interval at the given level in `(0, 1)`.
    Analytic {
        /// Confidence level (e.g. `0.95`).
        level: f64,
    },
}

impl Default for ConfidenceMethod {
    fn default() -> Self {
        Self::Analytic { level: 0.95 }
    }
}

/// How many null replicates a nonparametric CI test should draw.
///
/// [`SignificanceMethod::Analytic`] has no closed-form null for distance / MI proxies;
/// those tests use a documented default of 49. [`SignificanceMethod::BlockShuffle`]
/// honors the caller's `replicates`.
#[must_use]
pub fn nonparametric_permutation_count(significance: SignificanceMethod) -> usize {
    match significance {
        SignificanceMethod::Analytic => 49,
        SignificanceMethod::BlockShuffle { replicates, .. } => replicates.max(1) as usize,
    }
}

/// Confidence level for analytic intervals, if requested.
#[must_use]
pub fn analytic_confidence_level(confidence: ConfidenceMethod) -> Option<f64> {
    match confidence {
        ConfidenceMethod::None => None,
        ConfidenceMethod::Analytic { level } => Some(level),
    }
}

/// Preparation plan for a CI session.
#[derive(Clone, Debug)]
pub struct CiPreparationPlan {
    /// Significance method applied to subsequent queries.
    pub significance: SignificanceMethod,
    /// Confidence method applied when analytic intervals are available.
    pub confidence: ConfidenceMethod,
}

impl Default for CiPreparationPlan {
    fn default() -> Self {
        Self { significance: SignificanceMethod::Analytic, confidence: ConfidenceMethod::default() }
    }
}

/// Prepared CI state after [`ConditionalIndependenceTest::prepare`].
#[derive(Clone, Debug)]
pub struct PreparedCiTest {
    /// Row count observed at prepare time.
    pub n: usize,
    /// Column count observed at prepare time.
    pub ncols: usize,
    /// Plan used for preparation.
    pub plan: CiPreparationPlan,
}

impl PreparedCiTest {
    /// Ensure a batch request matches this prepare (row/column counts).
    ///
    /// # Errors
    ///
    /// Shape mismatch vs prepare-time `n` / `ncols`.
    pub fn ensure_compatible(&self, request: &CiBatchRequest<'_>) -> Result<(), StatsError> {
        let n = request.nrows()?;
        if n != self.n {
            return Err(StatsError::Shape { message: "CI batch row count differs from prepare()" });
        }
        if request.columns.len() != self.ncols {
            return Err(StatsError::Shape {
                message: "CI batch column count differs from prepare()",
            });
        }
        Ok(())
    }

    /// Copy of `request` with significance/confidence taken from this prepare plan.
    #[must_use]
    pub fn bind_request<'a>(&self, request: &CiBatchRequest<'a>) -> CiBatchRequest<'a> {
        CiBatchRequest {
            columns: request.columns,
            queries: request.queries,
            z_flat: request.z_flat,
            significance: self.plan.significance,
            confidence: self.plan.confidence,
        }
    }
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
    /// Confidence intervals (when the test supports analytic intervals).
    pub confidence: ConfidenceMethod,
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
///
/// Numeric kernels live in `antecedent-stats`; discovery owns the algorithm surface and
/// re-exports this trait.
pub trait ConditionalIndependenceTest {
    /// Prepare once for a data view / plan (sample planning, caches).
    ///
    /// # Errors
    ///
    /// Shape failures.
    fn prepare(
        &self,
        columns: &[&[f64]],
        plan: &CiPreparationPlan,
        _ctx: &ExecutionContext,
    ) -> Result<PreparedCiTest, StatsError> {
        if columns.is_empty() {
            return Err(StatsError::Shape { message: "no columns" });
        }
        let n = columns[0].len();
        for col in columns {
            if col.len() != n {
                return Err(StatsError::Shape { message: "column length mismatch" });
            }
        }
        Ok(PreparedCiTest { n, ncols: columns.len(), plan: plan.clone() })
    }

    /// Single query (convenience over [`Self::test_batch`]).
    ///
    /// # Errors
    ///
    /// Shape / numerical failures.
    fn test(
        &self,
        prepared: &PreparedCiTest,
        columns: &[&[f64]],
        query: CiQuery,
        z_flat: &[usize],
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiResult, StatsError> {
        let req = CiBatchRequest {
            columns,
            queries: std::slice::from_ref(&query),
            z_flat,
            significance: prepared.plan.significance,
            confidence: prepared.plan.confidence,
        };
        let out = self.test_batch(prepared, &req, workspace, ctx)?;
        out.results
            .into_iter()
            .next()
            .ok_or(StatsError::Shape { message: "CI test returned no results" })
    }

    /// Evaluate a batch against a prior [`Self::prepare`].
    ///
    /// Implementations must call [`PreparedCiTest::ensure_compatible`] and should
    /// honor `prepared.plan` for significance / confidence.
    ///
    /// # Errors
    ///
    /// Shape / numerical failures.
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError>;

    /// Ad-hoc batch: prepare from the request plan, then [`Self::test_batch`].
    ///
    /// Prefer an explicit prepare-once session for discovery / repeated queries.
    ///
    /// # Errors
    ///
    /// Shape / numerical failures.
    fn test_batch_adhoc(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        let plan = CiPreparationPlan {
            significance: request.significance,
            confidence: request.confidence,
        };
        let prepared = self.prepare(request.columns, &plan, ctx)?;
        self.test_batch(&prepared, request, workspace, ctx)
    }
}

/// Conditional independence test.
///
/// Prefer this name; [`ConditionalIndependenceTest`] is the same trait.
pub use ConditionalIndependenceTest as ConditionalIndependence;

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
    /// Contingency X level-code scratch (G² / discrete CI).
    pub contingency_x_codes: Vec<u32>,
    /// Contingency Y level-code scratch (G² / discrete CI).
    pub contingency_y_codes: Vec<u32>,
    /// kNN CMI index / permutation reuse state.
    pub knn: KnnDependenceWorkspace,
}

impl CiWorkspace {
    /// Prepare for `n_queries` results.
    pub fn prepare_queries(&mut self, n_queries: usize) {
        if self.stats.len() < n_queries {
            self.stats.resize(n_queries, None);
        }
    }
}
