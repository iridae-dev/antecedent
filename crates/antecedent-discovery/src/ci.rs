//! Conditional-independence surface owned by discovery.
//!
//! Numeric kernels remain in `antecedent-stats`; this module re-exports the DESIGN
//! trait contract so discovery algorithms depend on a discovery-owned CI API.
//! Concrete test constructors live in `antecedent-stats` (use
//! [`ci_from_name`] or import the type from that crate).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{StatsError, ensure_alpha_resolvable};

use crate::error::DiscoveryError;

/// Smallest number of queries worth a thread of their own: below it the spawn and the fresh
/// per-thread workspace cost more than the tests.
const MIN_QUERIES_PER_THREAD: usize = 4;

/// Evaluate `request` against `prepared`, splitting the queries across the context's threads
/// when there are enough of them.
///
/// Every query is independent of its batch position (permutation streams are keyed by the
/// query, never by its index), so the results are bit-identical for any thread count and
/// arrive in query order.
///
/// # Errors
///
/// The first failing part's error, as `test_batch` would report it.
pub(crate) fn run_ci_batch(
    ci: &(dyn ConditionalIndependence + Sync),
    prepared: &PreparedCiTest,
    request: &CiBatchRequest<'_>,
    workspace: &mut CiWorkspace,
    ctx: &ExecutionContext,
) -> Result<Vec<CiResult>, StatsError> {
    let n = request.queries.len();
    let threads = (ctx.parallelism.max_threads.get() as usize).min(n / MIN_QUERIES_PER_THREAD);
    if threads < 2 {
        return ci.test_batch(prepared, request, workspace, ctx).map(|out| out.results);
    }
    let per_part = n.div_ceil(threads);
    let parts = n.div_ceil(per_part);
    let chunks = ctx.map_indexed(parts, |part, inner| {
        let lo = part * per_part;
        let hi = (lo + per_part).min(n);
        let sub = CiBatchRequest { queries: &request.queries[lo..hi], ..request.clone() };
        ci.test_batch(prepared, &sub, &mut CiWorkspace::default(), inner).map(|out| out.results)
    })?;
    Ok(chunks.into_iter().flatten().collect())
}

/// The statistic and p-value of `result`, refused when either is not finite.
///
/// # Errors
///
/// [`DiscoveryError::Stats`] for a non-finite statistic or p-value: a NaN must not pass as
/// evidence either way.
pub(crate) fn finite_outcome(result: &CiResult) -> Result<(f64, f64), DiscoveryError> {
    if result.statistic.is_finite() && result.p_value.is_finite() {
        Ok((result.statistic, result.p_value))
    } else {
        Err(DiscoveryError::stats_msg("non-finite CI statistic or p-value"))
    }
}

/// Whether `error` says the CI test has no answer for this query (a constant variable or a
/// collinear conditioning set), as opposed to the run being broken.
///
/// An unanswerable query carries no information about independence, so constraint-based
/// search must neither delete the edge on it nor count it as a rejection.
#[must_use]
pub(crate) fn is_untestable(error: &DiscoveryError) -> bool {
    matches!(
        error,
        DiscoveryError::Stats(StatsError::Unsupported { message })
            if message.starts_with(UNDEFINED_PARTIAL_CORRELATION)
    )
}

/// Leading words of the statistics crate's refusal of an undefined partial correlation.
const UNDEFINED_PARTIAL_CORRELATION: &str = "partial correlation is undefined";

/// One single-query CI batch, its only result checked for finiteness.
///
/// # Errors
///
/// A CI failure (including [`is_untestable`] queries), an empty result, or a non-finite
/// statistic or p-value.
pub(crate) fn single_finite(
    ci: &(dyn ConditionalIndependence + Sync),
    prepared: &PreparedCiTest,
    request: &CiBatchRequest<'_>,
    workspace: &mut CiWorkspace,
    ctx: &ExecutionContext,
) -> Result<(f64, f64), DiscoveryError> {
    let results =
        run_ci_batch(ci, prepared, request, workspace, ctx).map_err(DiscoveryError::from)?;
    let result =
        results.first().ok_or_else(|| DiscoveryError::stats_msg("CI batch returned no results"))?;
    finite_outcome(result)
}

/// [`single_finite`] where a query the test cannot answer ([`is_untestable`]) is `Ok(None)`.
///
/// # Errors
///
/// Every failure of [`single_finite`] other than an untestable query.
pub(crate) fn single_finite_or_untestable(
    ci: &(dyn ConditionalIndependence + Sync),
    prepared: &PreparedCiTest,
    request: &CiBatchRequest<'_>,
    workspace: &mut CiWorkspace,
    ctx: &ExecutionContext,
) -> Result<Option<(f64, f64)>, DiscoveryError> {
    match single_finite(ci, prepared, request, workspace, ctx) {
        Ok(outcome) => Ok(Some(outcome)),
        Err(error) if is_untestable(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

/// [`run_ci_batch`] where a query the test cannot answer ([`is_untestable`]) is `None`.
///
/// One unanswerable query fails its whole batch, so on that failure each query is run by
/// itself and only the unanswerable ones are `None`. The results are what the batch would
/// have returned, because a query's result never depends on its batch.
///
/// # Errors
///
/// Every failure other than an untestable query, or a non-finite statistic or p-value.
pub(crate) fn run_ci_batch_or_untestable(
    ci: &(dyn ConditionalIndependence + Sync),
    prepared: &PreparedCiTest,
    request: &CiBatchRequest<'_>,
    workspace: &mut CiWorkspace,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<(f64, f64)>>, DiscoveryError> {
    match run_ci_batch(ci, prepared, request, workspace, ctx) {
        Ok(results) => results.iter().map(|result| finite_outcome(result).map(Some)).collect(),
        Err(error) => {
            let error = DiscoveryError::from(error);
            if !is_untestable(&error) {
                return Err(error);
            }
            (0..request.queries.len())
                .map(|index| {
                    let one = CiBatchRequest {
                        queries: &request.queries[index..=index],
                        ..request.clone()
                    };
                    single_finite_or_untestable(ci, prepared, &one, workspace, ctx)
                })
                .collect()
        }
    }
}

/// Sizes of the successive chunks of a scan that stops at its first hit.
///
/// A serial context evaluates one test at a time, exactly as an unbatched loop would. A
/// threaded one starts with a single test (most edges separate or fail immediately) and then
/// widens geometrically, so the tests evaluated past the hit stay within a small multiple of
/// the tests that were needed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChunkSchedule {
    next: usize,
    threads: usize,
    cap: usize,
}

impl ChunkSchedule {
    pub(crate) fn new(ctx: &ExecutionContext) -> Self {
        let threads = ctx.parallelism.max_threads.get() as usize;
        Self { next: 1, threads, cap: if threads <= 1 { 1 } else { threads * 16 } }
    }

    pub(crate) fn next_size(&mut self) -> usize {
        let size = self.next;
        self.next = if self.threads <= 1 {
            1
        } else if size == 1 {
            self.threads * MIN_QUERIES_PER_THREAD
        } else {
            (size * 2).min(self.cap)
        };
        size
    }
}

/// Refuse a CI test / `alpha` / multiplicity combination whose decisions would not mean what
/// they say.
///
/// - A permutation test cannot report a p-value below `1/(replicates+1)`; with `alpha` at or
///   below that floor every test "fails to reject" whatever the data show, which reads as
///   evidence of independence (default 49 replicates: nothing rejects at `alpha <= 0.02`).
/// - A test whose `p_value` is a posterior probability (the Bayesian tests) is not a
///   frequentist p-value, so Benjamini–Hochberg / FDR adjustment of it is meaningless.
///
/// # Errors
///
/// [`DiscoveryError::Stats`] when `alpha` is unresolvable;
/// [`DiscoveryError::Unsupported`] when FDR is requested for a non-frequentist p-value.
pub(crate) fn ensure_ci_decisions_meaningful(
    ci: &dyn ConditionalIndependence,
    significance: SignificanceMethod,
    alpha: f64,
    fdr_requested: bool,
) -> Result<(), DiscoveryError> {
    ensure_alpha_resolvable(ci.min_attainable_p(significance), alpha)?;
    if fdr_requested && !ci.p_value_is_frequentist() {
        return Err(DiscoveryError::unsupported(
            "FDR adjustment requires frequentist p-values; this CI test reports a posterior \
             probability of independence in p_value",
        ));
    }
    Ok(())
}

/// Diagnostic for a multiple-testing family finer than the CI test can resolve.
///
/// Benjamini–Hochberg (and Holm / Bonferroni) keep the smallest p-value only when it is at most
/// `alpha / m` for a family of `m` hypotheses. A permutation test cannot report a p-value below
/// its floor, so when that floor exceeds `alpha / m` a lone dependent link cannot survive: at
/// least `ceil(m · floor / alpha)` links must reach the floor together. Sparse graphs then come
/// back empty, which reads as evidence of independence. The run is not refused — dense graphs
/// can still clear the bar — but the result says so.
#[must_use]
pub(crate) fn fdr_resolution_diagnostic(
    ci: &dyn ConditionalIndependence,
    significance: SignificanceMethod,
    alpha: f64,
    family_size: usize,
) -> Option<crate::result::DiscoveryDiagnostic> {
    let floor = ci.min_attainable_p(significance);
    if family_size < 2 || floor <= alpha / family_size as f64 || !floor.is_finite() {
        return None;
    }
    let needed = (family_size as f64 * floor / alpha).ceil();
    let replicates = (family_size as f64 / alpha).ceil() - 1.0;
    Some(crate::result::DiscoveryDiagnostic {
        code: std::sync::Arc::from("ci.fdr_resolution"),
        message: std::sync::Arc::from(format!(
            "multiple-testing correction over {family_size} hypotheses keeps a lone link only at \
             p <= {:.3e}, below this CI test's smallest attainable p-value {floor:.3e}; at least \
             {needed} links must reach that floor together for any to survive. Use at least \
             {replicates} permutation replicates or disable the correction",
            alpha / family_size as f64
        )),
    })
}

/// A test whose per-row weights are aligned to the end of a series may only run on a frame that
/// is exactly that series minus its leading lag rows, with no interior rows removed.
///
/// Trailing alignment is only correct then: a weight vector of another length, a stacked
/// multi-environment frame, or a frame with masked / missing rows (which discovery compacts
/// before testing) would silently pair weights with the wrong observations.
///
/// # Errors
///
/// [`DiscoveryError::Unsupported`] when the weights do not fit the frame.
pub(crate) fn ensure_ci_fits_frame(
    ci: &dyn ConditionalIndependence,
    frame: &antecedent_data::LaggedFrame,
) -> Result<(), DiscoveryError> {
    let Some(weights) = ci.series_aligned_weights_len() else {
        return Ok(());
    };
    let series_rows = frame.n_effective() + frame.max_lag() as usize;
    if weights != series_rows || !frame.is_fully_valid() {
        return Err(DiscoveryError::unsupported(
            "observation weights must have one entry per series row and the lagged frame must be \
             the complete series (no masked, missing or pooled rows) to pair them with \
             observations",
        ));
    }
    Ok(())
}

/// Static counterpart of [`ensure_ci_fits_frame`]: a static table has no leading lag rows to
/// drop, so series-aligned weights must hold exactly one entry per row. Trailing alignment
/// would otherwise pair a longer vector's last `n_rows` entries with the rows.
///
/// # Errors
///
/// [`DiscoveryError::Unsupported`] when the weight count differs from `n_rows`.
pub(crate) fn ensure_ci_fits_rows(
    ci: &dyn ConditionalIndependence,
    n_rows: usize,
) -> Result<(), DiscoveryError> {
    match ci.series_aligned_weights_len() {
        Some(weights) if weights != n_rows => Err(DiscoveryError::unsupported(
            "observation weights must have exactly one entry per data row",
        )),
        _ => Ok(()),
    }
}

/// Prepare `columns` once for the unmasked fast path.
///
/// Every algorithm uses the caller's significance method and the default confidence
/// method. Masked queries still re-prepare on the compacted columns.
///
/// # Errors
///
/// The CI implementation's prepare error.
pub(crate) fn prepare_ci(
    ci: &(dyn ConditionalIndependence + Sync),
    columns: &[&[f64]],
    significance: SignificanceMethod,
    ctx: &ExecutionContext,
) -> Result<PreparedCiTest, DiscoveryError> {
    let plan = CiPreparationPlan { significance, confidence: ConfidenceMethod::default() };
    ci.prepare(columns, &plan, ctx).map_err(DiscoveryError::from)
}

/// One batch request at the caller's significance level and the default confidence method.
pub(crate) fn ci_batch_request<'a>(
    columns: &'a [&'a [f64]],
    queries: &'a [CiQuery],
    z_flat: &'a [usize],
    significance: SignificanceMethod,
) -> CiBatchRequest<'a> {
    CiBatchRequest {
        columns,
        queries,
        z_flat,
        significance,
        confidence: ConfidenceMethod::default(),
    }
}

pub use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery, CiResult, CiWorkspace,
    ConditionalIndependence, ConditionalIndependenceTest, ConfidenceMethod, PartialCorrelation,
    PreparedCiTest, SignificanceMethod, ci_from_name,
};

#[cfg(test)]
mod tests {
    use antecedent_core::VariableId;
    use antecedent_data::{LaggedFrame, TimeSeriesData};
    use antecedent_stats::{
        BayesFactorCi, PartialCorrelation, PosteriorDependenceCi, WeightedPartialCorrelation,
    };

    use super::*;

    #[test]
    fn series_aligned_weights_must_cover_exactly_the_series() {
        let x: Vec<f64> = (0..20).map(|i| f64::from(i).sin()).collect();
        let y: Vec<f64> = (0..20).map(|i| f64::from(i).cos()).collect();
        let data = TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
            .unwrap();
        let variables = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let frame = LaggedFrame::from_series(
            &data,
            &variables,
            2,
            &antecedent_core::KernelPolicy::default_policy(),
        )
        .unwrap();
        // One weight per series row: the frame is that series minus its two leading rows.
        assert!(
            ensure_ci_fits_frame(
                &WeightedPartialCorrelation::aligned_to_series_end(vec![1.0; 20]),
                &frame
            )
            .is_ok()
        );
        // Too long or too short: the trailing window would pair weights with other rows.
        for len in [19, 21, 40] {
            let ci = WeightedPartialCorrelation::aligned_to_series_end(vec![1.0; len]);
            assert!(ensure_ci_fits_frame(&ci, &frame).is_err(), "{len}");
        }
        // Tests without series-aligned weights are unaffected.
        assert!(ensure_ci_fits_frame(&PartialCorrelation, &frame).is_ok());
    }

    #[test]
    fn posterior_probability_tests_are_refused_under_fdr_but_not_without_it() {
        let significance = SignificanceMethod::Analytic;
        for ci in
            [&BayesFactorCi::new() as &dyn ConditionalIndependence, &PosteriorDependenceCi::new()]
        {
            assert!(ensure_ci_decisions_meaningful(ci, significance, 0.05, true).is_err());
            assert!(ensure_ci_decisions_meaningful(ci, significance, 0.05, false).is_ok());
        }
        assert!(
            ensure_ci_decisions_meaningful(&PartialCorrelation, significance, 0.05, true).is_ok()
        );
    }

    #[test]
    fn undefined_partial_correlation_is_untestable_not_independent() {
        let n = 60usize;
        let constant = vec![1.5; n];
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31).sin()).collect();
        let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17).cos()).collect();
        let columns: [&[f64]; 3] = [&constant, &y, &z];
        let ctx = ExecutionContext::for_tests(5);
        let ci = PartialCorrelation;
        let plan = CiPreparationPlan {
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let prepared = ci.prepare(&columns, &plan, &ctx).unwrap();
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let request = ci_batch_request(&columns, &queries, &z_flat, SignificanceMethod::Analytic);
        let mut workspace = CiWorkspace::default();
        let outcome =
            single_finite_or_untestable(&ci, &prepared, &request, &mut workspace, &ctx).unwrap();
        assert!(outcome.is_none(), "a constant variable has no partial correlation: {outcome:?}");
    }
}
