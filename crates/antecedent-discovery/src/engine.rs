//! Shared PCMCI engine: PC-style parents + MCI.
//!
//! Hot path: one [`LaggedFrame`] per run; CI tests index columns and reuse
//! workspace scratch (no per-test sample-plan rebuild).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::{LaggedFrame, TimeSeriesData, VectorVariableGroups, column_blocks_for_frame};
use antecedent_graph::TemporalGraphReview;
use antecedent_stats::{
    CiQuery, CiWorkspace, ConditionalIndependence, PairwiseMultivariateCi, PartialCorrelation,
    PreparedCiTest, SignificanceMethod,
};

use crate::constraints::{CompiledConstraints, DiscoveryConstraints};
use crate::error::DiscoveryError;
use crate::evidence::graph_evidence_from_scored_with_sepsets;
use crate::result::{
    AlgorithmRecord, DagDiscoveryResult, DiscoveryIteration, DiscoveryPerformanceRecord,
    LaggedLink, PcSepsets, ScoredLink, discovery_assumptions,
};

/// A tested link and its conditioning set.
type CiKey = (LaggedLink, Vec<(VariableId, Lag)>);

/// The lagged frame and the link a scan of conditioning sets tests on it.
#[derive(Clone, Copy)]
pub(crate) struct CiProbe<'a> {
    pub frame: &'a LaggedFrame,
    pub link: LaggedLink,
}

/// CI outcomes of one run keyed by tested link and conditioning set.
///
/// A test is a pure function of its link, conditioning set and the frame, so a repeated one
/// is answered from here instead of being run again.
#[derive(Debug, Default)]
pub(crate) struct CiMemo {
    outcomes: HashMap<CiKey, (f64, f64)>,
    /// Tests actually run (memo misses).
    pub computed: u64,
}

/// Live bytes of one partial-correlation CI query on `n_rows` rows with `z_len`
/// conditioning columns: the design matrix (the conditioning columns plus the two tested
/// ones), the Gram matrix, the two coefficient vectors and the two residual vectors, in `f64`.
fn ci_query_bytes(n_rows: usize, z_len: usize) -> u64 {
    let (n, p) = (n_rows as u128, z_len as u128);
    let cells = n * (p + 4) + p * p + 2 * p;
    u64::try_from(cells.saturating_mul(8)).unwrap_or(u64::MAX)
}

/// Most conditioning columns one CI query on `frame` may carry within the context's hard
/// memory limit; unbounded when no limit is set. The kernel allocates its workspace from the
/// query, so memory is the only ceiling on the conditioning-set size.
pub(crate) fn ci_conditioning_capacity(frame: &LaggedFrame, ctx: &ExecutionContext) -> usize {
    let Some(hard) = ctx.memory_limit_bytes() else {
        return usize::MAX;
    };
    let n_rows = frame_rows(frame);
    // The query needs at least 8 bytes per conditioning column, which bounds the search.
    let (mut fits, mut over) = (0usize, usize::try_from(hard / 8).unwrap_or(usize::MAX));
    while fits < over {
        let mid = fits + (over - fits).div_ceil(2);
        if ci_query_bytes(n_rows, mid) <= hard {
            fits = mid;
        } else {
            over = mid - 1;
        }
    }
    fits
}

/// Refuse a CI query whose workspace would exceed the context's hard memory limit.
fn ensure_ci_query_fits(
    n_rows: usize,
    z_len: usize,
    ctx: &ExecutionContext,
) -> Result<(), DiscoveryError> {
    match ctx.memory_limit_bytes() {
        Some(hard) if ci_query_bytes(n_rows, z_len) > hard => {
            Err(DiscoveryError::Resource(format!(
                "a CI query with {z_len} conditioning columns on {n_rows} rows needs {} bytes, \
                 over the memory hard limit of {hard}",
                ci_query_bytes(n_rows, z_len)
            )))
        }
        _ => Ok(()),
    }
}

/// Rows of a lagged frame (every column has the same length).
fn frame_rows(frame: &LaggedFrame) -> usize {
    if frame.ncols() == 0 { 0 } else { frame.column(0).len() }
}

type ParentSet = Vec<(VariableId, Lag)>;
type TargetParents = (VariableId, ParentSet);

struct ParentSelectOut {
    target: VariableId,
    parents: ParentSet,
    tests: u64,
    sepsets: PcSepsets,
}

/// The lagged frame, searched variables and compiled constraints one PC/MCI phase reads.
#[derive(Clone, Copy)]
pub(crate) struct LaggedSearch<'a> {
    pub frame: &'a LaggedFrame,
    pub variables: &'a [VariableId],
    pub compiled: &'a CompiledConstraints,
}

/// One PC1 conditioning level for `target`: its current parents and the conditioning size.
struct ParentLevel<'a> {
    target: VariableId,
    parents: &'a [(VariableId, Lag)],
    cond_size: usize,
}

struct MciChunkOut {
    scored: Vec<ScoredLink>,
    tests: u64,
    truncated: u64,
}

/// Reusable target-local discovery workspace.
#[derive(Clone, Debug, Default)]
pub struct DiscoveryWorkspace {
    /// Prepare-once CI session for the active lagged frame.
    pub prepared_ci: Option<PreparedCiTest>,
    /// CI workspace (parcorr residuals / shuffle scratch).
    pub ci: CiWorkspace,
    /// Scratch parents list.
    pub parents: Vec<(VariableId, Lag)>,
    /// Scratch combination buffer for PC conditioning sets.
    pub combo: Vec<(VariableId, Lag)>,
    /// Dense column indexes into the lagged frame for the active CI query.
    pub col_idxs: Vec<usize>,
    /// Flat conditioning indexes into the active CI column list (`2..`).
    pub z_flat: Vec<usize>,
    /// Scratch “others” list while iterating PC candidates.
    pub others: Vec<(VariableId, Lag)>,
    /// Scratch removal list for one conditioning-size pass.
    pub removed: Vec<(VariableId, Lag)>,
    /// Separating sets recorded when PC removes a candidate parent.
    ///
    /// Key: `(source, source_lag, target, target_lag)` with `target_lag` contemporaneous
    /// in the PC phase. Value: conditioning set that rendered the pair independent.
    pub sepsets: PcSepsets,
    /// Compacted column-major values for masked CI (`ncols * n_keep`).
    pub compact_values: Vec<f64>,
    /// Involved-column scratch for keep-mask construction.
    pub involved_cols: Vec<usize>,
    /// Cache: (frame row count, sorted involved column indexes) → keep mask.
    ///
    /// The row count is part of the key, not just the columns: this workspace is reused across
    /// calls whose lagged frames can differ in row count (e.g. one bootstrap replicate's frame
    /// against the next in [`crate::rpcmci::Rpcmci::run`]), and a mask cached for a larger frame
    /// would otherwise be replayed against a smaller one and read past its rows.
    pub keep_cache: HashMap<(usize, Vec<usize>), Arc<[bool]>>,
}

/// Shared PCMCI engine core.
#[derive(Clone)]
pub struct PcmciEngine {
    /// Constraints / alpha / lags.
    pub constraints: DiscoveryConstraints,
    /// Pluggable CI test (defaults to partial correlation).
    pub ci: Arc<dyn ConditionalIndependence + Send + Sync>,
    /// Optional pairwise-MV column blocks (pinned baseline `vector_vars` / space-dummy MV).
    /// Empty ⇒ no block expansion when building masked keep masks.
    pub column_blocks: Arc<[Arc<[usize]>]>,
}

impl std::fmt::Debug for PcmciEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PcmciEngine")
            .field("constraints", &self.constraints)
            .finish_non_exhaustive()
    }
}

impl Default for PcmciEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PcmciEngine {
    /// Default engine (`ParCorr` CI).
    #[must_use]
    pub fn new() -> Self {
        Self {
            constraints: DiscoveryConstraints::default(),
            ci: Arc::new(PartialCorrelation::new()),
            column_blocks: Arc::from([]),
        }
    }

    /// With constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Replace the CI test (e.g. [`antecedent_stats::OracleCi`], GPDC, …).
    #[must_use]
    pub fn with_ci(mut self, ci: Arc<dyn ConditionalIndependence + Send + Sync>) -> Self {
        self.ci = ci;
        self
    }

    /// Install pairwise-MV column blocks used for masked keep-mask expansion.
    #[must_use]
    pub fn with_column_blocks(mut self, column_blocks: Arc<[Arc<[usize]>]>) -> Self {
        self.column_blocks = column_blocks;
        self
    }

    /// PC1 parent selection for one target (pinned baseline `run_pc_stable`, `max_combinations=1`).
    ///
    /// Candidates are tested unconditionally, then at each conditioning size `q` against the
    /// single strongest-`q` set of the *other* surviving candidates, ranked by the minimum
    /// absolute test statistic seen so far. Returned parents are sorted strongest-first.
    ///
    /// # Errors
    ///
    /// Data or CI failures.
    #[allow(
        clippy::too_many_lines,
        reason = "PC1 levels in one loop: batch or sequential test, record outcomes, prune, re-rank; every step reads the surviving parents"
    )]
    pub fn select_parents(
        &self,
        frame: &LaggedFrame,
        target: VariableId,
        variables: &[VariableId],
        compiled: &CompiledConstraints,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<(VariableId, Lag)>, u64), DiscoveryError> {
        let mut parents = self.constraints.candidate_sources(variables, target);
        parents.retain(|&(src, slag)| {
            let link = LaggedLink {
                source: src,
                source_lag: slag,
                target,
                target_lag: Lag::CONTEMPORANEOUS,
            };
            compiled.allows(link)
        });
        let mut ci_tests = 0u64;
        let max_cond = self.constraints.max_cond_size;
        // PC1 strength ranking: minimum |statistic| across the levels a candidate survived.
        let mut min_stat = vec![f64::INFINITY; parents.len()];
        for cond_size in 0..=max_cond {
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            // A level-q test needs q other candidates to condition on.
            if parents.is_empty() || parents.len() <= cond_size {
                break;
            }
            if let Some(p) = &ctx.progress {
                p.report(cond_size as f64 / (max_cond.max(1) as f64 + 1.0), "discovery.ci");
            }
            workspace.removed.clear();
            // Permutation nulls are keyed by the query (x, y, sorted Z), not by its position in
            // a batch, so a batched and a sequential BlockShuffle p-value agree. BlockShuffle
            // stays on the sequential path here; Analytic benefits from the batched statistic
            // path.
            let use_batch = frame.is_fully_valid()
                && !matches!(
                    self.constraints.significance,
                    SignificanceMethod::BlockShuffle { .. }
                );
            if use_batch {
                let search = LaggedSearch { frame, variables, compiled };
                let level = ParentLevel { target, parents: &parents, cond_size };
                ci_tests +=
                    self.pc_parent_level_batch(search, &level, &mut min_stat, workspace, ctx)?;
            } else {
                for pi in 0..parents.len() {
                    if ctx.cancellation.is_cancelled() {
                        return Err(DiscoveryError::Cancelled);
                    }
                    let (src, slag) = parents[pi];
                    // Single strongest-q conditioning set: candidates are kept sorted by
                    // descending min |stat|, so the top q others are the strongest.
                    let mut combo = std::mem::take(&mut workspace.combo);
                    combo.clear();
                    combo.extend(pc_parent_combo(&parents, pi, cond_size));
                    let link = LaggedLink {
                        source: src,
                        source_lag: slag,
                        target,
                        target_lag: Lag::CONTEMPORANEOUS,
                    };
                    let result = self.ci_statistic(frame, link, &combo, workspace, ctx);
                    let outcome = match result {
                        Ok(pair) => pair,
                        Err(e) => {
                            workspace.combo = combo;
                            return Err(e);
                        }
                    };
                    ci_tests += 1;
                    record_pc_parent_outcome(
                        link,
                        outcome,
                        &combo,
                        compiled,
                        self.constraints.alpha,
                        &mut min_stat[pi],
                        workspace,
                    );
                    workspace.combo = combo;
                }
            }
            if !workspace.removed.is_empty() {
                let removed = std::mem::take(&mut workspace.removed);
                let mut keep_stats = Vec::with_capacity(parents.len());
                let mut keep_parents = Vec::with_capacity(parents.len());
                for (p, s) in parents.iter().copied().zip(min_stat.iter().copied()) {
                    if !removed.contains(&p) {
                        keep_parents.push(p);
                        keep_stats.push(s);
                    }
                }
                parents = keep_parents;
                min_stat = keep_stats;
                workspace.removed = removed;
            }
            sort_by_strength(&mut parents, &mut min_stat);
        }
        // `max_parents` caps the *surviving* parents, strongest first (the reference PCMCI+
        // implementation's `max_conds_py`), never the candidate list: truncating candidates by enumeration
        // order would decide by variable index which links are ever tested. Required
        // parents are exempt from the cap.
        if let Some(max_p) = self.constraints.max_parents {
            let n_required = parents
                .iter()
                .filter(|&&(src, slag)| compiled.requires(link_to_target(src, slag, target)))
                .count();
            let mut room = max_p.saturating_sub(n_required);
            parents.retain(|&(src, slag)| {
                compiled.requires(link_to_target(src, slag, target)) || {
                    let keep = room > 0;
                    room = room.saturating_sub(1);
                    keep
                }
            });
        }
        Ok((parents, ci_tests))
    }

    /// One PC1 conditioning level: batch every remaining candidate on a fully-valid frame.
    fn pc_parent_level_batch(
        &self,
        search: LaggedSearch<'_>,
        level: &ParentLevel<'_>,
        min_stat: &mut [f64],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<u64, DiscoveryError> {
        let LaggedSearch { frame, .. } = search;
        let ParentLevel { target, parents, cond_size } = *level;
        if ctx.cancellation.is_cancelled() {
            return Err(DiscoveryError::Cancelled);
        }
        let mut queries = Vec::with_capacity(parents.len());
        let mut z_flat = Vec::new();
        let mut combos = Vec::with_capacity(parents.len());
        for pi in 0..parents.len() {
            let (src, slag) = parents[pi];
            let combo: Vec<(VariableId, Lag)> = pc_parent_combo(parents, pi, cond_size);
            ensure_ci_query_fits(frame_rows(frame), combo.len(), ctx)?;
            let xi = frame.column_index(src, slag).ok_or_else(|| {
                DiscoveryError::data_msg(format!("missing lagged column for {src:?} lag {slag:?}"))
            })?;
            let yi = frame.column_index(target, Lag::CONTEMPORANEOUS).ok_or_else(|| {
                DiscoveryError::data_msg(format!("missing lagged column for {target:?}"))
            })?;
            let z_start = z_flat.len();
            for &(v, l) in &combo {
                let zi = frame.column_index(v, l).ok_or_else(|| {
                    DiscoveryError::data_msg(format!("missing lagged column for {v:?} lag {l:?}"))
                })?;
                z_flat.push(zi);
            }
            queries.push(CiQuery { x: xi, y: yi, z_start, z_len: combo.len() });
            combos.push(combo);
        }
        let cols: Vec<&[f64]> = (0..frame.ncols()).map(|i| frame.column(i)).collect();
        self.ensure_prepared_ci(frame, workspace, ctx)?;
        let req =
            crate::ci::ci_batch_request(&cols, &queries, &z_flat, self.constraints.significance);
        let prepared = workspace
            .prepared_ci
            .as_ref()
            .ok_or(DiscoveryError::Unsupported { message: "CI test used before prepare()" })?;
        let results = crate::ci::run_ci_batch(&*self.ci, prepared, &req, &mut workspace.ci, ctx)
            .map_err(DiscoveryError::from)?;
        if results.len() != parents.len() {
            return Err(DiscoveryError::stats_msg("CI batch result length mismatch"));
        }
        for (pi, (result, combo)) in results.into_iter().zip(combos).enumerate() {
            if !result.statistic.is_finite() || !result.p_value.is_finite() {
                return Err(DiscoveryError::stats_msg("non-finite CI statistic or p-value"));
            }
            let (src, slag) = parents[pi];
            let link = LaggedLink {
                source: src,
                source_lag: slag,
                target,
                target_lag: Lag::CONTEMPORANEOUS,
            };
            record_pc_parent_outcome(
                link,
                (result.statistic, result.p_value),
                &combo,
                search.compiled,
                self.constraints.alpha,
                &mut min_stat[pi],
                workspace,
            );
        }
        Ok(parents.len() as u64)
    }

    /// MCI test for a candidate link given parent sets.
    ///
    /// `parents_source` is keyed to the source at lag 0 and is shifted by the link's source
    /// lag τ so the conditioning set is `pa(X_{t−τ})`, matching pinned baseline's MCI phase. The
    /// frame must therefore materialize lags up to `2 · max_lag`.
    ///
    /// Returns the scored link and the number of conditioning parents truncated by
    /// the MCI size cap (same count the batch path surfaces as `mci.conditioning_truncated`).
    ///
    /// # Errors
    ///
    /// Data or CI failures.
    pub fn mci_test(
        &self,
        frame: &LaggedFrame,
        link: LaggedLink,
        parents_target: &[(VariableId, Lag)],
        parents_source: &[(VariableId, Lag)],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(ScoredLink, u64), DiscoveryError> {
        let capacity = ci_conditioning_capacity(frame, ctx);
        let truncated =
            mci_conditioning(link, parents_target, parents_source, capacity, &mut workspace.others);
        refuse_truncated_mci(truncated)?;
        let cond = std::mem::take(&mut workspace.others);
        let result = self.ci_statistic(frame, link, &cond, workspace, ctx);
        workspace.others = cond;
        let (stat, p) = result?;
        Ok((ScoredLink { link, statistic: stat, p_value: p, adjusted_p_value: None }, truncated))
    }

    /// Run PC parents for all targets then MCI on the full constrained candidate family.
    ///
    /// PC parent sets are used only for MCI conditioning (`pa(Y_t)` and time-shifted
    /// `pa(X_{t−τ})`). MCI scores every allowed `(X_{t−τ}, Y_t)` pair (Runge et al. 2019 /
    /// pinned baseline `run_mci`), not only PC survivors. Returns **unthresholded** scores;
    /// callers apply alpha and optional FDR over that full family.
    ///
    /// When [`DiscoveryConstraints::vector_groups`] is non-empty, the lagged frame still
    /// materializes all component columns, search runs over logical nodes only, and CI
    /// uses pairwise multivariate expansion.
    ///
    /// # Errors
    ///
    /// Data / CI / graph construction / memory-budget failures.
    pub fn run_pc_mci(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DagDiscoveryResult, DiscoveryError> {
        let frame =
            LaggedFrame::from_series(data, variables, self.frame_depth(), &ctx.kernel_policy)
                .map_err(DiscoveryError::from)?;
        self.run_pc_mci_on_frame(&frame, variables, workspace, ctx)
    }

    /// [`Self::run_pc_mci`] over the units of a panel: each unit's lag windows are built inside
    /// that unit ([`LaggedFrame::from_panel`]) and the rows are pooled, so a unit boundary never
    /// supplies a lagged parent. The pooled design is the lagged design of one structural model
    /// shared by exchangeable units.
    ///
    /// # Errors
    ///
    /// Data / CI / graph construction / memory-budget failures.
    pub fn run_pc_mci_panel(
        &self,
        units: &[TimeSeriesData],
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DagDiscoveryResult, DiscoveryError> {
        let frame =
            LaggedFrame::from_panel(units, variables, self.frame_depth(), &ctx.kernel_policy)
                .map_err(DiscoveryError::from)?;
        self.run_pc_mci_on_frame(&frame, variables, workspace, ctx)
    }

    /// Depth of the lagged frame PC and MCI both run on.
    ///
    /// Aligned with the pinned baseline's default `cut_off='2xtau_max'`: both use a frame
    /// materializing lags up to `2·max_lag` (same effective sample count).
    pub(crate) fn frame_depth(&self) -> u32 {
        2 * self.constraints.temporal.max_lag.raw()
    }

    /// [`Self::run_pc_mci`] on a pre-built lagged frame of depth `Self::frame_depth`.
    ///
    /// # Errors
    ///
    /// Data / CI / graph construction / memory-budget failures.
    pub fn run_pc_mci_on_frame(
        &self,
        frame: &LaggedFrame,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DagDiscoveryResult, DiscoveryError> {
        if let Some(hard) = ctx.memory_limit_bytes() {
            if frame.values_bytes() > hard {
                return Err(DiscoveryError::Unsupported {
                    message: "lagged frames exceed ExecutionContext memory hard limit",
                });
            }
        }

        let (ci, column_blocks, search_vars) = resolve_vector_ci(
            &self.ci,
            &self.column_blocks,
            &self.constraints.vector_groups,
            frame,
            variables,
        )?;
        crate::ci::ensure_ci_decisions_meaningful(
            &*ci,
            self.constraints.significance,
            self.constraints.alpha,
            false,
        )?;
        crate::ci::ensure_ci_fits_frame(&*ci, frame)?;
        let engine = PcmciEngine { constraints: self.constraints.clone(), ci, column_blocks };

        let threads = ctx.parallelism.max_threads.get().max(1);
        let compiled = engine.constraints.compile(&search_vars)?;

        // : prepare CI once for the lagged frame (unmasked fast path).
        if frame.is_fully_valid() {
            let cols: Vec<&[f64]> = (0..frame.ncols()).map(|i| frame.column(i)).collect();
            workspace.prepared_ci = Some(crate::ci::prepare_ci(
                &*engine.ci,
                &cols,
                engine.constraints.significance,
                ctx,
            )?);
        } else {
            workspace.prepared_ci = None;
            workspace.keep_cache.clear();
        }

        let (all_parents, iterations, mut ci_tests) =
            engine.select_parents_all(frame, &search_vars, &compiled, workspace, ctx, threads)?;

        let mut scored = Vec::new();
        // MCI conditioning that overflows the column cap is refused (`refuse_truncated_mci`),
        // so a successful run never carries a truncated test.
        let search = LaggedSearch { frame, variables: &search_vars, compiled: &compiled };
        let (mci_tests, _) =
            engine.mci_all(search, &all_parents, &mut scored, workspace, ctx, threads)?;
        ci_tests += mci_tests;

        let sepsets = std::mem::take(&mut workspace.sepsets);
        let evidence = graph_evidence_from_scored_with_sepsets(scored, &sepsets)?;
        let algorithm = AlgorithmRecord {
            id: Arc::from("pcmci.engine.pc_mci"),
            config: Arc::from(format!(
                "alpha={},max_lag={}",
                self.constraints.alpha,
                self.constraints.temporal.max_lag.raw()
            )),
        };
        let review = TemporalGraphReview::from_graph(evidence.graph.clone(), algorithm.id.clone());
        let n_links = evidence.links.len() as u64;
        let diagnostics = Vec::new();
        Ok(DagDiscoveryResult {
            evidence,
            review,
            algorithm,
            assumptions: discovery_assumptions("pcmci", true),
            iterations,
            diagnostics,
            performance: DiscoveryPerformanceRecord {
                ci_tests,
                links_retained: n_links,
                targets: search_vars.len() as u64,
                lagged_frame_bytes: frame.values_bytes(),
                worker_threads: threads,
            },
            sepsets,
        })
    }

    pub(crate) fn select_parents_all(
        &self,
        frame: &LaggedFrame,
        variables: &[VariableId],
        compiled: &CompiledConstraints,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
        threads: u32,
    ) -> Result<(Vec<TargetParents>, Vec<DiscoveryIteration>, u64), DiscoveryError> {
        if threads <= 1 || variables.len() <= 1 {
            let mut all_parents = Vec::with_capacity(variables.len());
            // One record per variable, so graph-sized, not data-sized.
            let mut iterations = Vec::with_capacity(variables.len());
            let mut ci_tests = 0u64;
            for &target in variables {
                if ctx.cancellation.is_cancelled() {
                    return Err(DiscoveryError::Cancelled);
                }
                let (parents, tests) =
                    self.select_parents(frame, target, variables, compiled, workspace, ctx)?;
                ci_tests += tests;
                iterations.push(DiscoveryIteration {
                    label: Arc::from(format!("pc_parents:{target}")),
                    ci_tests: tests,
                });
                all_parents.push((target, parents));
                if let Some(p) = &ctx.progress {
                    p.report(
                        all_parents.len() as f64 / variables.len().max(1) as f64,
                        "discovery.ci",
                    );
                }
            }
            return Ok((all_parents, iterations, ci_tests));
        }

        let n = variables.len();
        let prepared_ci = workspace.prepared_ci.clone();
        let outs = ctx.map_indexed(n, |i, inner| {
            if inner.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            let target = variables[i];
            let mut local_ws = DiscoveryWorkspace {
                prepared_ci: prepared_ci.clone(),
                ..DiscoveryWorkspace::default()
            };
            self.select_parents(frame, target, variables, compiled, &mut local_ws, inner).map(
                |(parents, tests)| ParentSelectOut {
                    target,
                    parents,
                    tests,
                    sepsets: std::mem::take(&mut local_ws.sepsets),
                },
            )
        })?;

        let mut all_parents = Vec::with_capacity(n);
        let mut iterations = Vec::with_capacity(n);
        let mut ci_tests = 0u64;
        for (i, out) in outs.into_iter().enumerate() {
            debug_assert_eq!(out.target, variables[i]);
            ci_tests += out.tests;
            workspace.sepsets.extend(out.sepsets);
            iterations.push(DiscoveryIteration {
                label: Arc::from(format!("pc_parents:{}", out.target)),
                ci_tests: out.tests,
            });
            all_parents.push((out.target, out.parents));
        }
        Ok((all_parents, iterations, ci_tests))
    }

    fn mci_all(
        &self,
        search: LaggedSearch<'_>,
        all_parents: &[TargetParents],
        scored: &mut Vec<ScoredLink>,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
        threads: u32,
    ) -> Result<(u64, u64), DiscoveryError> {
        if threads <= 1 || all_parents.len() <= 1 {
            let mut tests = 0u64;
            let mut truncated = 0u64;
            for (target, parents) in all_parents {
                let (batch, trunc) = self.mci_batch_for_target(
                    search,
                    *target,
                    parents,
                    all_parents,
                    workspace,
                    ctx,
                )?;
                tests += batch.len() as u64;
                truncated += trunc;
                scored.extend(batch);
            }
            return Ok((tests, truncated));
        }

        let n = all_parents.len();
        let prepared_ci = workspace.prepared_ci.clone();
        let partials = ctx.map_indexed(n, |i, inner| {
            let (target, parents) = &all_parents[i];
            let mut local_ws = DiscoveryWorkspace {
                prepared_ci: prepared_ci.clone(),
                ..DiscoveryWorkspace::default()
            };
            let (batch, trunc) = self.mci_batch_for_target(
                search,
                *target,
                parents,
                all_parents,
                &mut local_ws,
                inner,
            )?;
            let tests = batch.len() as u64;
            Ok::<_, DiscoveryError>(MciChunkOut { scored: batch, tests, truncated: trunc })
        })?;

        let mut tests = 0u64;
        let mut truncated = 0u64;
        for chunk in partials {
            tests += chunk.tests;
            truncated += chunk.truncated;
            scored.extend(chunk.scored);
        }
        Ok((tests, truncated))
    }

    /// MCI-test the full constrained candidate family for one target.
    ///
    /// `parents` are the PC-estimated parents of `target` and are used only for
    /// conditioning, together with PC parents of each candidate source. Candidates are
    /// every allowed `(src, τ)` from [`DiscoveryConstraints::candidate_sources`].
    ///
    /// Returns the scored links and the number of conditioning columns dropped by the
    /// memory-derived column capacity (0 unless a hard memory limit is set).
    fn mci_batch_for_target(
        &self,
        search: LaggedSearch<'_>,
        target: VariableId,
        parents: &[(VariableId, Lag)],
        all_parents: &[TargetParents],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<ScoredLink>, u64), DiscoveryError> {
        let LaggedSearch { frame, variables, compiled } = search;
        if ctx.cancellation.is_cancelled() {
            return Err(DiscoveryError::Cancelled);
        }
        let mut candidates = self.constraints.candidate_sources(variables, target);
        candidates.retain(|&(src, slag)| compiled.allows(link_to_target(src, slag, target)));
        if candidates.is_empty() {
            return Ok((Vec::new(), 0));
        }

        // Masked frames: per-query keep masks differ, so fall back to serial ci_statistic.
        if !frame.is_fully_valid() {
            let mut scored = Vec::with_capacity(candidates.len());
            let mut truncated = 0u64;
            for &(src, slag) in &candidates {
                if ctx.cancellation.is_cancelled() {
                    return Err(DiscoveryError::Cancelled);
                }
                let link = link_to_target(src, slag, target);
                let src_parents = parents_of(all_parents, src);
                let (s, trunc) =
                    self.mci_test(frame, link, parents, src_parents, workspace, ctx)?;
                truncated += trunc;
                scored.push(s);
            }
            return Ok((scored, truncated));
        }

        let cols: Vec<&[f64]> = (0..frame.ncols()).map(|i| frame.column(i)).collect();
        self.ensure_prepared_ci(frame, workspace, ctx)?;
        let mut queries = Vec::with_capacity(candidates.len());
        let mut z_flat = Vec::new();
        let mut links = Vec::with_capacity(candidates.len());

        let capacity = ci_conditioning_capacity(frame, ctx);
        let mut truncated = 0u64;
        for &(src, slag) in &candidates {
            let link = link_to_target(src, slag, target);
            let src_parents = parents_of(all_parents, src);
            truncated +=
                mci_conditioning(link, parents, src_parents, capacity, &mut workspace.others);
            refuse_truncated_mci(truncated)?;

            let xi = frame.column_index(link.source, link.source_lag).ok_or_else(|| {
                DiscoveryError::data_msg(format!("missing lagged column for {:?}", link.source))
            })?;
            let yi = frame.column_index(link.target, link.target_lag).ok_or_else(|| {
                DiscoveryError::data_msg(format!("missing lagged column for {:?}", link.target))
            })?;
            let z_start = z_flat.len();
            for &(v, l) in &workspace.others {
                let zi = frame.column_index(v, l).ok_or_else(|| {
                    DiscoveryError::data_msg(format!("missing lagged column for {v:?} lag {l:?}"))
                })?;
                z_flat.push(zi);
            }
            queries.push(CiQuery { x: xi, y: yi, z_start, z_len: workspace.others.len() });
            links.push(link);
        }

        let req =
            crate::ci::ci_batch_request(&cols, &queries, &z_flat, self.constraints.significance);
        let prepared = workspace
            .prepared_ci
            .as_ref()
            .ok_or(DiscoveryError::Unsupported { message: "CI test used before prepare()" })?;
        let results = crate::ci::run_ci_batch(&*self.ci, prepared, &req, &mut workspace.ci, ctx)
            .map_err(DiscoveryError::from)?;
        if results.len() != links.len() {
            return Err(DiscoveryError::stats_msg("CI batch result length mismatch"));
        }
        let mut scored = Vec::with_capacity(links.len());
        for (link, result) in links.into_iter().zip(results) {
            if !result.statistic.is_finite() || !result.p_value.is_finite() {
                return Err(DiscoveryError::stats_msg("non-finite CI statistic or p-value"));
            }
            scored.push(ScoredLink {
                link,
                statistic: result.statistic,
                p_value: result.p_value,
                adjusted_p_value: None,
            });
        }
        Ok((scored, truncated))
    }

    /// Ensure [`DiscoveryWorkspace::prepared_ci`] matches `frame`.
    fn ensure_prepared_ci(
        &self,
        frame: &LaggedFrame,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(), DiscoveryError> {
        let n = frame.n_effective();
        let ncols = frame.ncols();
        let needs = match workspace.prepared_ci.as_ref() {
            None => true,
            Some(p) => p.n != n || p.ncols != ncols,
        };
        if needs {
            let cols: Vec<&[f64]> = (0..ncols).map(|i| frame.column(i)).collect();
            workspace.prepared_ci =
                Some(crate::ci::prepare_ci(&*self.ci, &cols, self.constraints.significance, ctx)?);
        }
        Ok(())
    }

    /// CI tests of `link` under each conditioning set of `conds` as one (possibly threaded)
    /// batch, or `None` when the sets cannot be batched (masked frame, oversized or unknown
    /// columns, a failing batch). `None` sends the caller down the one-test-at-a-time path,
    /// which reports each failure at the test that causes it.
    fn ci_batch(
        &self,
        frame: &LaggedFrame,
        link: LaggedLink,
        conds: &[&[(VariableId, Lag)]],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Option<Vec<antecedent_stats::CiResult>> {
        if conds.len() < 2 || !frame.is_fully_valid() {
            return None;
        }
        let xi = frame.column_index(link.source, link.source_lag)?;
        let yi = frame.column_index(link.target, link.target_lag)?;
        let mut queries = Vec::with_capacity(conds.len());
        let mut z_flat = Vec::new();
        for cond in conds {
            if ensure_ci_query_fits(frame_rows(frame), cond.len(), ctx).is_err() {
                return None;
            }
            let z_start = z_flat.len();
            for &(v, l) in *cond {
                z_flat.push(frame.column_index(v, l)?);
            }
            queries.push(CiQuery { x: xi, y: yi, z_start, z_len: cond.len() });
        }
        self.ensure_prepared_ci(frame, workspace, ctx).ok()?;
        let cols: Vec<&[f64]> = (0..frame.ncols()).map(|i| frame.column(i)).collect();
        let req =
            crate::ci::ci_batch_request(&cols, &queries, &z_flat, self.constraints.significance);
        let prepared = workspace.prepared_ci.as_ref()?;
        let results =
            crate::ci::run_ci_batch(&*self.ci, prepared, &req, &mut workspace.ci, ctx).ok()?;
        (results.len() == conds.len()).then_some(results)
    }

    /// Visit the CI outcome of `link` under each conditioning set of `sets`, in order, until
    /// `visit` breaks.
    ///
    /// Equivalent to calling [`Self::ci_statistic`] on each set in turn and stopping at the
    /// first break: upcoming sets are evaluated in speculative batches (see
    /// [`crate::ci::ChunkSchedule`]) whose results past the break are only kept in `memo`,
    /// and an error is reported at the set that causes it, after every earlier set was
    /// visited.
    ///
    /// # Errors
    ///
    /// The first failing set's error (a failure of the generator counts at its position).
    pub(crate) fn scan_ci_sets<B>(
        &self,
        probe: CiProbe<'_>,
        mut sets: impl Iterator<Item = Result<Vec<(VariableId, Lag)>, DiscoveryError>>,
        memo: &mut CiMemo,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
        mut visit: impl FnMut(Vec<(VariableId, Lag)>, f64, f64) -> std::ops::ControlFlow<B>,
    ) -> Result<Option<B>, DiscoveryError> {
        let CiProbe { frame, link } = probe;
        let mut schedule = crate::ci::ChunkSchedule::new(ctx);
        loop {
            let mut chunk: Vec<Vec<(VariableId, Lag)>> = Vec::new();
            let mut pending: Option<DiscoveryError> = None;
            for item in sets.by_ref().take(schedule.next_size()) {
                match item {
                    Ok(cond) => chunk.push(cond),
                    Err(e) => {
                        pending = Some(e);
                        break;
                    }
                }
            }
            if chunk.is_empty() && pending.is_none() {
                return Ok(None);
            }

            let mut missing: Vec<&[(VariableId, Lag)]> = Vec::new();
            for cond in &chunk {
                let known = memo.outcomes.contains_key(&(link, cond.clone()))
                    || missing.iter().any(|m| *m == cond.as_slice());
                if !known {
                    missing.push(cond);
                }
            }
            if let Some(results) = self.ci_batch(frame, link, &missing, workspace, ctx) {
                for (cond, r) in missing.iter().zip(results) {
                    // A non-finite outcome is left out so that it errors where it is visited.
                    if r.statistic.is_finite() && r.p_value.is_finite() {
                        memo.outcomes.insert((link, cond.to_vec()), (r.statistic, r.p_value));
                        memo.computed += 1;
                    }
                }
            }

            for cond in chunk {
                let key = (link, cond);
                let (stat, p) = if let Some(&hit) = memo.outcomes.get(&key) {
                    hit
                } else {
                    let out = self.ci_statistic(frame, link, &key.1, workspace, ctx)?;
                    memo.outcomes.insert(key.clone(), out);
                    memo.computed += 1;
                    out
                };
                if let std::ops::ControlFlow::Break(b) = visit(key.1, stat, p) {
                    return Ok(Some(b));
                }
            }
            if let Some(e) = pending {
                return Err(e);
            }
        }
    }

    /// CI test of `link.source ⊥ link.target | cond` on `frame`.
    #[allow(
        clippy::too_many_lines,
        reason = "fast unmasked path, then the masked complete-case path; both share the column indexes and workspace"
    )]
    pub(crate) fn ci_statistic(
        &self,
        frame: &LaggedFrame,
        link: LaggedLink,
        cond: &[(VariableId, Lag)],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(f64, f64), DiscoveryError> {
        let LaggedLink { source: x, source_lag: x_lag, target: y, target_lag: y_lag } = link;
        ensure_ci_query_fits(frame_rows(frame), cond.len(), ctx)?;
        let xi = frame.column_index(x, x_lag).ok_or_else(|| {
            DiscoveryError::data_msg(format!("missing lagged column for {x:?} lag {x_lag:?}"))
        })?;
        let yi = frame.column_index(y, y_lag).ok_or_else(|| {
            DiscoveryError::data_msg(format!("missing lagged column for {y:?} lag {y_lag:?}"))
        })?;
        workspace.z_flat.clear();
        for &(v, l) in cond {
            let zi = frame.column_index(v, l).ok_or_else(|| {
                DiscoveryError::data_msg(format!("missing lagged column for {v:?} lag {l:?}"))
            })?;
            workspace.z_flat.push(zi);
        }

        if frame.is_fully_valid() {
            let cols: Vec<&[f64]> = (0..frame.ncols()).map(|i| frame.column(i)).collect();
            self.ensure_prepared_ci(frame, workspace, ctx)?;
            let queries = [CiQuery { x: xi, y: yi, z_start: 0, z_len: workspace.z_flat.len() }];
            let req = crate::ci::ci_batch_request(
                &cols,
                &queries,
                &workspace.z_flat,
                self.constraints.significance,
            );
            let prepared = workspace
                .prepared_ci
                .as_ref()
                .ok_or(DiscoveryError::Unsupported { message: "CI test used before prepare()" })?;
            let result = crate::ci::run_ci_batch(&*self.ci, prepared, &req, &mut workspace.ci, ctx)
                .map_err(DiscoveryError::from)?
                .into_iter()
                .next()
                .ok_or_else(|| DiscoveryError::stats_msg("CI batch returned no results"))?;
            if !result.statistic.is_finite() || !result.p_value.is_finite() {
                return Err(DiscoveryError::stats_msg("non-finite CI statistic or p-value"));
            }
            return Ok((result.statistic, result.p_value));
        }

        // Masked / incomplete series: complete-case over selected (y,x,z) roles per mask_type,
        // expanding through pairwise-MV blocks when configured.
        workspace.involved_cols.clear();
        let mask_type = self.constraints.mask_type;
        if mask_type.includes_x() {
            expand_into(&mut workspace.involved_cols, xi, &self.column_blocks);
        }
        if mask_type.includes_y() {
            expand_into(&mut workspace.involved_cols, yi, &self.column_blocks);
        }
        if mask_type.includes_z() {
            let z_copy: Vec<usize> = workspace.z_flat.clone();
            for &zi in &z_copy {
                expand_into(&mut workspace.involved_cols, zi, &self.column_blocks);
            }
        }
        if workspace.involved_cols.is_empty() {
            // Degenerate mask_type with no participants: keep all rows.
            workspace.involved_cols.clear();
        }
        workspace.involved_cols.sort_unstable();
        workspace.involved_cols.dedup();

        let frame_n = if frame.ncols() == 0 { 0 } else { frame.column(0).len() };
        let keep = if workspace.involved_cols.is_empty() {
            Arc::from(vec![true; frame_n])
        } else {
            let key = (frame_n, workspace.involved_cols.clone());
            if let Some(cached) = workspace.keep_cache.get(&key) {
                Arc::clone(cached)
            } else {
                let mask = frame
                    .keep_mask_for_columns(&workspace.involved_cols)
                    .map_err(DiscoveryError::from)?;
                let arc: Arc<[bool]> = Arc::from(mask);
                workspace.keep_cache.insert(key, Arc::clone(&arc));
                arc
            }
        };
        let n_keep = keep.iter().filter(|&&k| k).count();
        if n_keep < 3 {
            return Err(DiscoveryError::stats_msg(
                "insufficient complete-case samples after mask/missingness",
            ));
        }

        // Compact only the columns this query reads — x, y, then Z, with the
        // query remapped onto that dense layout. The historical loop compacted
        // all `frame.ncols()` columns per CI test (O(ncols·n) copy per test);
        // the CI implementations read exactly `columns[q.x]`, `columns[q.y]`,
        // and the Z columns, so the selected sub-matrix yields bit-identical
        // statistics (the prepared plan records only n/ncols/plan).
        let z_len = workspace.z_flat.len();
        let selected: Vec<usize> =
            [xi, yi].into_iter().chain(workspace.z_flat.iter().copied()).collect();
        let n_sel = selected.len();
        let need = n_sel.saturating_mul(n_keep);
        if workspace.compact_values.len() < need {
            workspace.compact_values.resize(need, 0.0);
        }
        for (c, &src_col) in selected.iter().enumerate() {
            let src = frame.column(src_col);
            let dst = &mut workspace.compact_values[c * n_keep..(c + 1) * n_keep];
            let mut j = 0usize;
            for (i, &k) in keep.iter().enumerate() {
                if k {
                    dst[j] = src[i];
                    j += 1;
                }
            }
        }
        {
            let cols: Vec<&[f64]> = (0..n_sel)
                .map(|c| &workspace.compact_values[c * n_keep..(c + 1) * n_keep])
                .collect();
            workspace.prepared_ci =
                Some(crate::ci::prepare_ci(&*self.ci, &cols, self.constraints.significance, ctx)?);
        }
        let cols: Vec<&[f64]> =
            (0..n_sel).map(|c| &workspace.compact_values[c * n_keep..(c + 1) * n_keep]).collect();
        let z_remap: Vec<usize> = (2..2 + z_len).collect();
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len }];
        let req =
            crate::ci::ci_batch_request(&cols, &queries, &z_remap, self.constraints.significance);
        let prepared = workspace
            .prepared_ci
            .as_ref()
            .ok_or(DiscoveryError::Unsupported { message: "CI test used before prepare()" })?;
        let result = crate::ci::run_ci_batch(&*self.ci, prepared, &req, &mut workspace.ci, ctx)
            .map_err(DiscoveryError::from)?
            .into_iter()
            .next()
            .ok_or_else(|| DiscoveryError::stats_msg("CI batch returned no results"))?;
        if !result.statistic.is_finite() || !result.p_value.is_finite() {
            return Err(DiscoveryError::stats_msg("non-finite CI statistic or p-value"));
        }
        Ok((result.statistic, result.p_value))
    }
}

fn pc_parent_combo(
    parents: &[(VariableId, Lag)],
    pi: usize,
    cond_size: usize,
) -> Vec<(VariableId, Lag)> {
    parents.iter().enumerate().filter(|(j, _)| *j != pi).map(|(_, x)| *x).take(cond_size).collect()
}

/// Fold one PC1 outcome for `link` into the candidate's `min_stat` slot, queueing it for
/// removal (with `combo` as its separating set) when the test accepts independence.
fn record_pc_parent_outcome(
    link: LaggedLink,
    outcome: (f64, f64),
    combo: &[(VariableId, Lag)],
    compiled: &CompiledConstraints,
    alpha: f64,
    min_stat: &mut f64,
    workspace: &mut DiscoveryWorkspace,
) {
    let (stat, p) = outcome;
    // Always record |stat| so required edges that fail independence still rank by
    // observed association (not as +∞ / strongest).
    *min_stat = min_stat.min(stat.abs());
    // pinned baseline retains links with p <= alpha (independence when p > alpha).
    if p > alpha && !compiled.requires(link) {
        workspace.removed.push((link.source, link.source_lag));
        workspace.sepsets.insert(
            (link.source, link.source_lag, link.target, Lag::CONTEMPORANEOUS),
            Arc::from(combo),
        );
    }
}

#[allow(clippy::type_complexity, reason = "one-off tuple returned from a single private helper")]
fn resolve_vector_ci(
    base_ci: &Arc<dyn ConditionalIndependence + Send + Sync>,
    base_blocks: &Arc<[Arc<[usize]>]>,
    groups: &VectorVariableGroups,
    frame: &LaggedFrame,
    variables: &[VariableId],
) -> Result<
    (Arc<dyn ConditionalIndependence + Send + Sync>, Arc<[Arc<[usize]>]>, Vec<VariableId>),
    DiscoveryError,
> {
    let search_vars = groups.filter_search_variables(variables);
    let group_blocks = column_blocks_for_frame(groups, frame).map_err(DiscoveryError::from)?;
    if group_blocks.is_empty() {
        return Ok((Arc::clone(base_ci), Arc::clone(base_blocks), search_vars));
    }
    // Prefer explicit vector-group blocks; merge with any pre-installed blocks.
    let mut merged: Vec<Arc<[usize]>> = group_blocks.iter().cloned().collect();
    for b in base_blocks.iter() {
        merged.push(Arc::clone(b));
    }
    let blocks: Arc<[Arc<[usize]>]> = Arc::from(merged);
    let ci: Arc<dyn ConditionalIndependence + Send + Sync> =
        Arc::new(PairwiseMultivariateCi::with_column_blocks(Arc::clone(&blocks)));
    Ok((ci, blocks, search_vars))
}

fn expand_into(out: &mut Vec<usize>, col: usize, blocks: &[Arc<[usize]>]) {
    for block in blocks {
        if block.iter().any(|&c| c == col) {
            for &m in block.iter() {
                if !out.contains(&m) {
                    out.push(m);
                }
            }
            return;
        }
    }
    if !out.contains(&col) {
        out.push(col);
    }
}

fn parents_of(all_parents: &[TargetParents], src: VariableId) -> &[(VariableId, Lag)] {
    all_parents.iter().find(|(t, _)| *t == src).map_or(&[][..], |(_, p)| p.as_slice())
}

pub(crate) fn parents_of_target(
    all_parents: &[(VariableId, Vec<(VariableId, Lag)>)],
    src: VariableId,
) -> &[(VariableId, Lag)] {
    parents_of(all_parents, src)
}

/// Sort candidates by descending min |statistic|, tie-broken by (variable, lag) for
/// determinism across runs and thread counts.
fn sort_by_strength(parents: &mut Vec<(VariableId, Lag)>, min_stat: &mut Vec<f64>) {
    let mut order: Vec<usize> = (0..parents.len()).collect();
    order.sort_by(|&i, &j| {
        min_stat[j].partial_cmp(&min_stat[i]).unwrap_or(std::cmp::Ordering::Equal).then_with(|| {
            let (vi, li) = parents[i];
            let (vj, lj) = parents[j];
            (vi.raw(), li.raw()).cmp(&(vj.raw(), lj.raw()))
        })
    });
    *parents = order.iter().map(|&i| parents[i]).collect();
    *min_stat = order.iter().map(|&i| min_stat[i]).collect();
}

fn link_to_target(src: VariableId, slag: Lag, target: VariableId) -> LaggedLink {
    LaggedLink { source: src, source_lag: slag, target, target_lag: Lag::CONTEMPORANEOUS }
}

/// Build the MCI conditioning set for `link` into `out`: parents of the target and of the
/// source, each *time-shifted by its endpoint's lag* (a parent `(v, l)` of `X_t` is
/// `(v, l+τ)` for `X_{t−τ}`), minus the link endpoints. Parent sets are keyed to lag 0. The
/// target shift is zero for every skeleton-phase link (`Y_t`); it matters when a caller tests a
/// pair whose later node is the source. Both inputs are strongest-first, so truncation to
/// `budget` columns keeps the strongest conditions; returns how many were dropped.
///
/// `budget` is the most conditioning columns the caller can use, normally
/// [`ci_conditioning_capacity`]. Callers that add further conditions of their own (PCMCI+'s
/// contemporaneous candidate set `S`) must reserve room for them *here*, before the lagged
/// block is built. Truncating the combined set afterwards drops from the tail, which is the
/// lagged MCI block — the fixed control set that makes the test valid under autocorrelation —
/// while keeping the optional conditions that were prepended.
pub(crate) fn mci_conditioning(
    link: LaggedLink,
    parents_target: &[(VariableId, Lag)],
    parents_source: &[(VariableId, Lag)],
    budget: usize,
    out: &mut Vec<(VariableId, Lag)>,
) -> u64 {
    out.clear();
    let src_key = (link.source, link.source_lag);
    let tgt_key = (link.target, link.target_lag);
    for &(v, l) in parents_target {
        let shifted = (v, Lag::from_raw(l.raw() + link.target_lag.raw()));
        if !out.contains(&shifted) && shifted != src_key && shifted != tgt_key {
            out.push(shifted);
        }
    }
    for &(v, l) in parents_source {
        let shifted = (v, Lag::from_raw(l.raw() + link.source_lag.raw()));
        if !out.contains(&shifted) && shifted != src_key && shifted != tgt_key {
            out.push(shifted);
        }
    }
    if out.len() > budget {
        let dropped = out.len() - budget;
        out.truncate(budget);
        dropped as u64
    } else {
        0
    }
}

/// Fail closed when an MCI conditioning set was truncated. A weaker test under
/// autocorrelation is not a valid MCI test.
pub(crate) fn refuse_truncated_mci(dropped: u64) -> Result<(), DiscoveryError> {
    if dropped == 0 {
        Ok(())
    } else {
        Err(DiscoveryError::Resource(format!(
            "MCI conditioning dropped {dropped} lagged parents over the memory-derived column \
             budget; refusing a truncated test"
        )))
    }
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
