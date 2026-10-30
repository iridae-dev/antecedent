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
    /// Number of times a new [`MatchingIndex`](crate::matching::MatchingIndex) was constructed.
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

/// Null replicates a nonparametric CI test draws when the caller asks for
/// [`SignificanceMethod::Analytic`], which has no closed-form null for distance / MI proxies.
pub const DEFAULT_PERMUTATIONS: usize = 49;

/// How many null replicates a nonparametric CI test should draw.
///
/// [`SignificanceMethod::Analytic`] has no closed-form null for distance / MI proxies, so those
/// tests draw `DEFAULT_PERMUTATIONS` (49) replicates and their smallest attainable p-value is
/// `1/50 = 0.02`: with `alpha <= 0.02` no dependence can ever be declared, and Benjamini–
/// Hochberg over a family of such p-values needs at least 40% of the family at that minimum.
/// Discovery must compare [`ConditionalIndependenceTest::min_attainable_p`] with its `alpha`
/// and ask for [`SignificanceMethod::BlockShuffle`] with more replicates when the test cannot
/// resolve it. [`SignificanceMethod::BlockShuffle`] honors the caller's `replicates`.
#[must_use]
pub(crate) fn nonparametric_permutation_count(significance: SignificanceMethod) -> usize {
    match significance {
        SignificanceMethod::Analytic => DEFAULT_PERMUTATIONS,
        SignificanceMethod::BlockShuffle { replicates, .. } => replicates.max(1) as usize,
    }
}

/// Smallest p-value an add-one permutation test with `replicates` draws can report:
/// `1 / (replicates + 1)`.
#[must_use]
pub(crate) fn permutation_min_p(replicates: usize) -> f64 {
    1.0 / (replicates as f64 + 1.0)
}

/// Resolution of the exact block-permutation reference, including the identity.
/// A short final block is still one independently shuffled block. Once the
/// factorial exceeds `u64`, its reciprocal is below any `u32` replicate floor.
fn arrangement_min_p(rows: usize, block_size: usize) -> f64 {
    let blocks = rows.div_ceil(block_size.max(1));
    let mut arrangements = 1_u64;
    for factor in 2..=blocks {
        let Ok(factor) = u64::try_from(factor) else { return 0.0 };
        let Some(product) = arrangements.checked_mul(factor) else { return 0.0 };
        arrangements = product;
    }
    1.0 / arrangements as f64
}

/// Which null sets [`ConditionalIndependenceTest::min_attainable_p`].
///
/// The trait default is [`AttainableP::Zero`], so a test that does not override
/// [`ConditionalIndependenceTest::attainable_p`] keeps reporting `0`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum AttainableP {
    /// Always `0`, including when the caller asked for a block shuffle.
    Zero,
    /// `0` on the analytic path; `1 / (replicates + 1)` under block shuffle.
    AnalyticOrBlockShuffle,
    /// Permutation floor, using the default replicate count when the method is analytic.
    NonparametricPermutation,
}

/// Smallest attainable p-value of the block-permutation null (`0` under the analytic path).
pub(crate) fn block_shuffle_min_p(significance: SignificanceMethod) -> f64 {
    match significance {
        SignificanceMethod::Analytic => 0.0,
        SignificanceMethod::BlockShuffle { replicates, .. } => {
            permutation_min_p(replicates as usize)
        }
    }
}

/// Floor reported by [`AttainableP`].
pub(crate) fn attainable_p_value(kind: AttainableP, significance: SignificanceMethod) -> f64 {
    match kind {
        AttainableP::Zero => 0.0,
        AttainableP::AnalyticOrBlockShuffle => block_shuffle_min_p(significance),
        AttainableP::NonparametricPermutation => {
            permutation_min_p(nonparametric_permutation_count(significance))
        }
    }
}

/// Refuse an `alpha` at or below the test's Monte Carlo or exact-null resolution.
///
/// More Monte Carlo draws do not repair an exact reference with too few block
/// arrangements. Callers that know their `alpha` (discovery) call this with
/// [`ConditionalIndependenceTest::min_resolvable_p`].
///
/// # Errors
///
/// [`StatsError::Unsupported`] when `min_p >= alpha`.
pub fn ensure_alpha_resolvable(min_p: f64, alpha: f64) -> Result<(), StatsError> {
    if min_p >= alpha {
        return Err(StatsError::Unsupported {
            message: "the CI test's smallest attainable p-value or exact-null resolution \
                      is not below alpha: use enough \
                      permutation replicates and distinct block arrangements (more rows \
                      or smaller blocks), or a larger alpha",
        });
    }
    Ok(())
}

/// The block length the caller asked for, or `1` when no blocking was requested.
///
/// `1` means "permute exchangeably": [`SignificanceMethod::Analytic`] has no blocking concept,
/// and `BlockShuffle { block_size: 0 | 1, .. }` imposes none.
#[must_use]
pub fn requested_block_size(significance: SignificanceMethod) -> usize {
    match significance {
        SignificanceMethod::Analytic => 1,
        SignificanceMethod::BlockShuffle { block_size, .. } => block_size.max(1),
    }
}

/// Reject a block-preserving request from a test that can only permute exchangeably.
///
/// [`SignificanceMethod::BlockShuffle`]'s `block_size` exists so the null keeps the serial
/// dependence of the observed series; an exchangeable permutation under-disperses that null
/// and inflates Type I error for exactly the autocorrelated data the parameter is meant to
/// protect. Tests whose null generation cannot honour `block_size` must say so rather than
/// accept the argument and discard it, which silently returns an anticonservative p-value.
///
/// `block_size <= 1` imposes no blocking, so it is always accepted.
///
/// Call this per query, not per batch: for a test that builds its null by stratifying on Z,
/// whether `block_size` can be honoured depends on whether *that query's* conditioning set is
/// empty, which varies across a batch.
///
/// # Errors
///
/// [`StatsError::Unsupported`] when `block_size > 1`.
pub fn reject_unsupported_block_size(
    significance: SignificanceMethod,
    test_name: &'static str,
) -> Result<(), crate::error::StatsError> {
    if let SignificanceMethod::BlockShuffle { block_size, .. } = significance {
        if block_size > 1 {
            let _ = test_name;
            return Err(crate::error::StatsError::Unsupported {
                message: "block-preserving permutation (block_size > 1) is not supported for \
                          this CI test when the conditioning set is non-empty: its null is a \
                          within-Z-stratum exchange over strata whose members are scattered \
                          across time, which is structurally incompatible with a contiguous-block \
                          permutation and would be invalid for autocorrelated data if silently \
                          substituted. This is a structural limit, not an unimplemented feature. \
                          Options: query with an empty conditioning set, where blocking is \
                          supported because the null is a single time-ordered stratum; use \
                          ParCorr / WeightedParCorr / MultivariateParCorr / Gpdc, which \
                          residualize on Z and so support block-preserving nulls with \
                          conditioning; or set block_size = 1 to accept an exchangeable null.",
            });
        }
    }
    Ok(())
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
    /// Ensure a batch request matches this prepare and every query index is in range.
    ///
    /// # Errors
    ///
    /// Shape mismatch vs prepare-time `n` / `ncols`, or invalid query/conditioning indices.
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
        for query in request.queries {
            if query.x >= self.ncols || query.y >= self.ncols {
                return Err(StatsError::Shape { message: "CI query column index out of range" });
            }
            let conditioning = query
                .z_start
                .checked_add(query.z_len)
                .and_then(|end| request.z_flat.get(query.z_start..end))
                .ok_or(StatsError::Shape { message: "CI conditioning range out of bounds" })?;
            if conditioning.iter().any(|&column| column >= self.ncols) {
                return Err(StatsError::Shape {
                    message: "CI conditioning column index out of range",
                });
            }
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
    /// Two-sided p-value. For the Bayesian tests this is instead the posterior probability of
    /// independence (see [`ConditionalIndependenceTest::p_value_is_frequentist`]), which is
    /// not uniform under the null and must not be multiplicity-adjusted.
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
    /// Optional borrow-scoped cross-products for repeated analytic queries.
    /// Tests with different nulls or statistics retain the ordinary route.
    fn covariance_session<'a>(
        &self,
        _columns: &'a [&'a [f64]],
        _ctx: &ExecutionContext,
    ) -> Option<super::CovarianceCiSession<'a>> {
        None
    }

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

    /// Which null this test's smallest p-value comes from.
    ///
    /// The default is [`AttainableP::Zero`]. Analytic partial-correlation tests override
    /// with [`AttainableP::AnalyticOrBlockShuffle`]; nonparametric permutation tests override
    /// with [`AttainableP::NonparametricPermutation`]. A test whose floor depends on instance
    /// state overrides [`Self::min_attainable_p`] instead.
    fn attainable_p(&self) -> AttainableP {
        AttainableP::Zero
    }

    /// Smallest p-value this test can report under `significance`.
    ///
    /// `0.0` for tests with a continuous analytic reference. Permutation tests return
    /// `1 / (replicates + 1)`; a caller with a significance level `alpha <= min_attainable_p`
    /// cannot get a rejection from them (see [`ensure_alpha_resolvable`]).
    ///
    /// The default reads [`Self::attainable_p`]. Override this method only when the floor
    /// depends on values stored on `self`.
    fn min_attainable_p(&self, significance: SignificanceMethod) -> f64 {
        attainable_p_value(self.attainable_p(), significance)
    }

    /// Smallest significance level the null can resolve on `rows` observations.
    ///
    /// Permutation tests need both enough Monte Carlo draws and enough distinct
    /// block arrangements. The latter include the identity and number
    /// `ceil(rows / block_size)!`, even with a short final block. Repeated data
    /// values or conditional strata can further coarsen the reference, so this
    /// is a lower bound on its resolution. A sampled p-value can fall below the
    /// exact arrangement floor by chance; adding replicates does not make that
    /// reference more informative. Discovery should use this method for its
    /// alpha guard, while [`Self::min_attainable_p`] describes only the reported
    /// Monte Carlo floor.
    fn min_resolvable_p(&self, significance: SignificanceMethod, rows: usize) -> f64 {
        let monte_carlo = self.min_attainable_p(significance);
        let permutes = match self.attainable_p() {
            AttainableP::Zero => false,
            AttainableP::AnalyticOrBlockShuffle => {
                matches!(significance, SignificanceMethod::BlockShuffle { .. })
            }
            AttainableP::NonparametricPermutation => true,
        };
        if permutes {
            monte_carlo.max(arrangement_min_p(rows, requested_block_size(significance)))
        } else {
            monte_carlo
        }
    }

    /// Length of the whole series this test's per-row weights were supplied for, when they are
    /// aligned to the *end* of a series (lagged frames drop leading rows). `None` for tests
    /// without series-aligned side data.
    ///
    /// A caller that builds the frames must check that its frame is exactly the series minus
    /// the leading lag rows, with no rows removed elsewhere, before running the test: a wrong
    /// length or a frame with interior gaps would pair weights with the wrong observations.
    fn series_aligned_weights_len(&self) -> Option<usize> {
        None
    }

    /// Whether [`CiResult::p_value`] is a frequentist p-value (super-uniform under the null).
    ///
    /// `false` for the Bayesian tests, whose `p_value` is a posterior probability of
    /// independence: thresholding it at `alpha` is a decision rule with its own operating
    /// characteristics, and Benjamini–Hochberg / FDR adjustment of it is meaningless.
    /// Multiplicity-adjusting callers must refuse tests that return `false`.
    fn p_value_is_frequentist(&self) -> bool {
        true
    }

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

#[cfg(test)]
mod resolution_tests {
    use super::*;
    use crate::ci::PartialCorrelation;

    #[test]
    fn block_resolution_counts_arrangements_including_a_short_final_block() {
        let test = PartialCorrelation::new();
        let significance = SignificanceMethod::BlockShuffle { replicates: 999, block_size: 50 };
        assert_eq!(test.min_attainable_p(significance), 0.001);
        assert_eq!(test.min_resolvable_p(significance, 100), 0.5);
        assert_eq!(test.min_resolvable_p(significance, 101), 1.0 / 6.0);
        assert_eq!(test.min_resolvable_p(significance, 50), 1.0);
        assert!(ensure_alpha_resolvable(test.min_resolvable_p(significance, 100), 0.05).is_err());
        assert!(ensure_alpha_resolvable(test.min_resolvable_p(significance, 1000), 0.05).is_ok());
        assert_eq!(test.min_resolvable_p(SignificanceMethod::Analytic, 2), 0.0);
    }

    #[test]
    fn huge_arrangement_counts_do_not_overflow_or_loop_over_all_rows() {
        assert_eq!(arrangement_min_p(usize::MAX, 1), 0.0);
        assert_eq!(arrangement_min_p(0, 1), 1.0);
    }
}

#[cfg(test)]
mod request_validation_tests {
    use super::*;
    use crate::ci::{GSquared, PartialCorrelation};

    #[test]
    fn every_query_index_is_validated_before_continuous_or_discrete_ci_runs() {
        let first = [0.0, 1.0, 0.0, 1.0];
        let second = [1.0, 0.0, 1.0, 0.0];
        let columns = [&first[..], &second[..]];
        let continuous = PartialCorrelation::new();
        let discrete = GSquared::new();
        let ctx = ExecutionContext::for_tests(1);
        let invalid = [
            CiQuery { x: 2, y: 0, z_start: 0, z_len: 0 },
            CiQuery { x: 0, y: 2, z_start: 0, z_len: 0 },
            CiQuery { x: 0, y: 1, z_start: usize::MAX, z_len: 1 },
            CiQuery { x: 0, y: 1, z_start: 2, z_len: 0 },
            CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 },
        ];
        for test in [&continuous as &dyn ConditionalIndependenceTest, &discrete] {
            let prepared = test.prepare(&columns, &CiPreparationPlan::default(), &ctx).unwrap();
            for query in invalid {
                assert!(matches!(
                    test.test(&prepared, &columns, query, &[2], &mut CiWorkspace::default(), &ctx),
                    Err(StatsError::Shape { .. })
                ));
            }
        }
    }
}
