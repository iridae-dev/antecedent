//! Shared PCMCI engine: PC-style parents + MCI.
//!
//! Hot path: one [`LaggedFrame`] per run; CI tests index columns and reuse
//! workspace scratch (no per-test sample-plan rebuild).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use causal_data::{LaggedFrame, TimeSeriesData, VectorVariableGroups, column_blocks_for_frame};
use causal_graph::TemporalGraphReview;
use causal_stats::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependence, ConfidenceMethod,
    PairwiseMultivariateCi, PreparedCiTest, PartialCorrelation,
};

use crate::constraints::{CompiledConstraints, DiscoveryConstraints};
use crate::error::DiscoveryError;
use crate::evidence::graph_evidence_from_scored_with_sepsets;
use crate::result::{
    AlgorithmRecord, DagDiscoveryResult, DiscoveryIteration, DiscoveryPerformanceRecord,
    LaggedLink, PcSepsets, ScoredLink,
};

/// Maximum columns in one CI query (X, Y, + conditioning). Stack-backed refs.
const MAX_CI_COLS: usize = 32;

type ParentSet = Vec<(VariableId, Lag)>;
type TargetParents = (VariableId, ParentSet);

struct ParentSelectOut {
    target: VariableId,
    parents: ParentSet,
    tests: u64,
    sepsets: PcSepsets,
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
    /// Cache: sorted involved column indexes → keep mask.
    pub keep_cache: HashMap<Vec<usize>, Arc<[bool]>>,
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

    /// Replace the CI test (e.g. [`causal_stats::OracleCi`], GPDC, …).
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
        if let Some(max_p) = self.constraints.max_parents {
            // Never drop required parents when truncating.
            let mut required: Vec<_> = parents
                .iter()
                .copied()
                .filter(|&(src, slag)| {
                    compiled.requires(LaggedLink {
                        source: src,
                        source_lag: slag,
                        target,
                        target_lag: Lag::CONTEMPORANEOUS,
                    })
                })
                .collect();
            let mut optional: Vec<_> =
                parents.iter().copied().filter(|p| !required.contains(p)).collect();
            let room = max_p.saturating_sub(required.len());
            optional.truncate(room);
            required.extend(optional);
            parents = required;
        }
        let mut ci_tests = 0u64;
        let max_cond = self.constraints.max_cond_size;
        // PC1 strength ranking: minimum |statistic| across the levels a candidate survived.
        let mut min_stat = vec![f64::INFINITY; parents.len()];
        for cond_size in 0..=max_cond {
            // A level-q test needs q other candidates to condition on.
            if parents.is_empty() || parents.len() <= cond_size {
                break;
            }
            workspace.removed.clear();
            for pi in 0..parents.len() {
                let (src, slag) = parents[pi];
                // Single strongest-q conditioning set: candidates are kept sorted by
                // descending min |stat|, so the top q others are the strongest.
                let mut combo = std::mem::take(&mut workspace.combo);
                combo.clear();
                combo.extend(
                    parents
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != pi)
                        .map(|(_, x)| *x)
                        .take(cond_size),
                );
                let result = self.ci_statistic(
                    frame,
                    src,
                    slag,
                    target,
                    Lag::CONTEMPORANEOUS,
                    &combo,
                    workspace,
                    ctx,
                );
                let outcome = match result {
                    Ok(pair) => pair,
                    Err(e) => {
                        workspace.combo = combo;
                        return Err(e);
                    }
                };
                ci_tests += 1;
                let (stat, p) = outcome;
                // Always record |stat| so required edges that fail independence still rank by
                // observed association (not as +∞ / strongest).
                min_stat[pi] = min_stat[pi].min(stat.abs());
                // pinned baseline retains links with p <= alpha (independence when p > alpha).
                if p > self.constraints.alpha {
                    let link = LaggedLink {
                        source: src,
                        source_lag: slag,
                        target,
                        target_lag: Lag::CONTEMPORANEOUS,
                    };
                    if !compiled.requires(link) {
                        workspace.removed.push((src, slag));
                        workspace.sepsets.insert(
                            (src, slag, target, Lag::CONTEMPORANEOUS),
                            Arc::from(combo.clone().into_boxed_slice()),
                        );
                    }
                }
                workspace.combo = combo;
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
        Ok((parents, ci_tests))
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
        let truncated =
            mci_conditioning(link, parents_target, parents_source, &mut workspace.others);
        let cond = std::mem::take(&mut workspace.others);
        let result = self.ci_statistic(
            frame,
            link.source,
            link.source_lag,
            link.target,
            link.target_lag,
            &cond,
            workspace,
            ctx,
        );
        workspace.others = cond;
        let (stat, p) = result?;
        Ok((
            ScoredLink { link, statistic: stat, p_value: p, adjusted_p_value: None },
            truncated,
        ))
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
        let max_lag = self.constraints.temporal.max_lag.raw();
        // Align with pinned baseline's default `cut_off='2xtau_max'`: both PC and MCI use a
        // frame materializing lags up to 2·max_lag (same effective sample count).
        let frame_depth = 2 * max_lag;
        let frame =
            LaggedFrame::from_series(data, variables, frame_depth, &ctx.kernel_policy)
                .map_err(DiscoveryError::from)?;
        if let Some(hard) = ctx.memory.hard_limit_bytes {
            if frame.values_bytes() > hard {
                return Err(DiscoveryError::Unsupported {
                    message: "lagged frames exceed ExecutionContext memory hard limit",
                });
            }
        }

        let (ci, column_blocks, search_vars) =
            resolve_vector_ci(&self.ci, &self.column_blocks, &self.constraints.vector_groups, &frame, variables)?;
        let engine = PcmciEngine {
            constraints: self.constraints.clone(),
            ci,
            column_blocks,
        };

        let threads = ctx.parallelism.max_threads.get().max(1);
        let compiled = engine.constraints.compile(&search_vars)?;

        // : prepare CI once for the lagged frame (unmasked fast path).
        if frame.is_fully_valid() {
            let cols: Vec<&[f64]> = (0..frame.ncols()).map(|i| frame.column(i)).collect();
            let plan = causal_stats::CiPreparationPlan {
                significance: engine.constraints.significance,
                confidence: ConfidenceMethod::default(),
            };
            workspace.prepared_ci =
                Some(engine.ci.prepare(&cols, &plan, ctx).map_err(DiscoveryError::from)?);
        } else {
            workspace.prepared_ci = None;
            workspace.keep_cache.clear();
        }

        let (all_parents, iterations, mut ci_tests) =
            engine.select_parents_all(&frame, &search_vars, &compiled, workspace, ctx, threads)?;

        let mut scored = Vec::new();
        let (mci_tests, truncated) = engine.mci_all(
            &frame,
            &search_vars,
            &compiled,
            &all_parents,
            &mut scored,
            workspace,
            ctx,
            threads,
        )?;
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
        let mut diagnostics = Vec::new();
        if truncated > 0 {
            diagnostics.push(crate::result::DiscoveryDiagnostic {
                code: Arc::from("mci.conditioning_truncated"),
                message: Arc::from(format!(
                    "MCI conditioning sets dropped {truncated} weakest condition(s) at the \
                     {MAX_CI_COLS}-column cap; statistics for the affected links use a \
                     reduced conditioning set"
                )),
            });
        }
        Ok(DagDiscoveryResult {
            evidence,
            review,
            algorithm,
            assumptions: AssumptionSet::new(),
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
            let mut iterations = Vec::with_capacity(variables.len());
            let mut ci_tests = 0u64;
            for &target in variables {
                let (parents, tests) =
                    self.select_parents(frame, target, variables, compiled, workspace, ctx)?;
                ci_tests += tests;
                iterations.push(DiscoveryIteration {
                    label: Arc::from(format!("pc_parents:{target}")),
                    ci_tests: tests,
                });
                all_parents.push((target, parents));
            }
            return Ok((all_parents, iterations, ci_tests));
        }

        let n = variables.len();
        let mut slots: Vec<Option<Result<ParentSelectOut, DiscoveryError>>> =
            (0..n).map(|_| None).collect();

        let prepared_ci = workspace.prepared_ci.clone();
        std::thread::scope(|scope| {
            let mut rest = slots.as_mut_slice();
            let mut cursor = 0usize;
            for (start, end) in chunk_ranges(n, threads) {
                let (this, next) = rest.split_at_mut(end - start);
                debug_assert_eq!(cursor, start);
                let chunk_vars = &variables[start..end];
                let engine = self;
                let prepared_ci = prepared_ci.clone();
                scope.spawn(move || {
                    let mut local_ws = DiscoveryWorkspace {
                        prepared_ci,
                        ..DiscoveryWorkspace::default()
                    };
                    for (i, &target) in chunk_vars.iter().enumerate() {
                        this[i] = Some(
                            engine
                                .select_parents(
                                    frame,
                                    target,
                                    variables,
                                    compiled,
                                    &mut local_ws,
                                    ctx,
                                )
                                .map(|(parents, tests)| ParentSelectOut {
                                    target,
                                    parents,
                                    tests,
                                    sepsets: std::mem::take(&mut local_ws.sepsets),
                                }),
                        );
                    }
                });
                rest = next;
                cursor = end;
            }
        });

        let mut all_parents = Vec::with_capacity(n);
        let mut iterations = Vec::with_capacity(n);
        let mut ci_tests = 0u64;
        for (i, slot) in slots.into_iter().enumerate() {
            let out = slot.ok_or(DiscoveryError::Unsupported {
                message: "parallel PC worker left empty slot",
            })??;
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

    #[allow(clippy::too_many_arguments)]
    fn mci_all(
        &self,
        frame: &LaggedFrame,
        variables: &[VariableId],
        compiled: &CompiledConstraints,
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
                    frame,
                    variables,
                    compiled,
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
        let ranges = chunk_ranges(n, threads);
        let mut partials: Vec<Option<Result<MciChunkOut, DiscoveryError>>> =
            Vec::with_capacity(ranges.len());

        let prepared_ci = workspace.prepared_ci.clone();
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(ranges.len());
            for &(start, end) in &ranges {
                let chunk_parents = &all_parents[start..end];
                let engine = self;
                let prepared_ci = prepared_ci.clone();
                handles.push(scope.spawn(move || {
                    let mut local_ws = DiscoveryWorkspace {
                        prepared_ci,
                        ..DiscoveryWorkspace::default()
                    };
                    let mut local_scored = Vec::new();
                    let mut tests = 0u64;
                    let mut truncated = 0u64;
                    for (target, parents) in chunk_parents {
                        let (batch, trunc) = engine.mci_batch_for_target(
                            frame,
                            variables,
                            compiled,
                            *target,
                            parents,
                            all_parents,
                            &mut local_ws,
                            ctx,
                        )?;
                        tests += batch.len() as u64;
                        truncated += trunc;
                        local_scored.extend(batch);
                    }
                    Ok(MciChunkOut { scored: local_scored, tests, truncated })
                }));
            }
            for h in handles {
                partials.push(Some(h.join().expect("MCI worker panicked")));
            }
        });

        let mut tests = 0u64;
        let mut truncated = 0u64;
        for slot in partials {
            let chunk = slot.ok_or(DiscoveryError::Unsupported {
                message: "parallel MCI worker left empty slot",
            })??;
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
    /// `MAX_CI_COLS` cap (0 in the common case).
    #[allow(clippy::too_many_arguments)]
    fn mci_batch_for_target(
        &self,
        frame: &LaggedFrame,
        variables: &[VariableId],
        compiled: &CompiledConstraints,
        target: VariableId,
        parents: &[(VariableId, Lag)],
        all_parents: &[TargetParents],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<ScoredLink>, u64), DiscoveryError> {
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

        let mut truncated = 0u64;
        for &(src, slag) in &candidates {
            let link = link_to_target(src, slag, target);
            let src_parents = parents_of(all_parents, src);
            truncated += mci_conditioning(link, parents, src_parents, &mut workspace.others);

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

        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: self.constraints.significance,
            confidence: ConfidenceMethod::default(),
        };
        let prepared = workspace.prepared_ci.as_ref().ok_or_else(|| {
            DiscoveryError::Unsupported { message: "CI test used before prepare()" }
        })?;
        let out = self
            .ci
            .test_batch(prepared, &req, &mut workspace.ci, ctx)
            .map_err(DiscoveryError::from)?;
        if out.results.len() != links.len() {
            return Err(DiscoveryError::stats_msg("CI batch result length mismatch"));
        }
        let mut scored = Vec::with_capacity(links.len());
        for (link, result) in links.into_iter().zip(out.results) {
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
            let plan = causal_stats::CiPreparationPlan {
                significance: self.constraints.significance,
                confidence: ConfidenceMethod::default(),
            };
            workspace.prepared_ci =
                Some(self.ci.prepare(&cols, &plan, ctx).map_err(DiscoveryError::from)?);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ci_statistic(
        &self,
        frame: &LaggedFrame,
        x: VariableId,
        x_lag: Lag,
        y: VariableId,
        y_lag: Lag,
        cond: &[(VariableId, Lag)],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(f64, f64), DiscoveryError> {
        if 2 + cond.len() > MAX_CI_COLS {
            return Err(DiscoveryError::Unsupported {
                message: "conditioning set exceeds MAX_CI_COLS",
            });
        }
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
            let queries = [CiQuery {
                x: xi,
                y: yi,
                z_start: 0,
                z_len: workspace.z_flat.len(),
            }];
            let req = CiBatchRequest {
                columns: &cols,
                queries: &queries,
                z_flat: &workspace.z_flat,
                significance: self.constraints.significance,
                confidence: ConfidenceMethod::default(),
            };
            let prepared = workspace.prepared_ci.as_ref().ok_or_else(|| {
                DiscoveryError::Unsupported { message: "CI test used before prepare()" }
            })?;
            let out = self
                .ci
                .test_batch(prepared, &req, &mut workspace.ci, ctx)
                .map_err(DiscoveryError::from)?;
            let result = out
                .results
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

        let keep = if workspace.involved_cols.is_empty() {
            let n = if frame.ncols() == 0 { 0 } else { frame.column(0).len() };
            Arc::from(vec![true; n])
        } else {
            let key = workspace.involved_cols.clone();
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

        let ncols = frame.ncols();
        let need = ncols.saturating_mul(n_keep);
        if workspace.compact_values.len() < need {
            workspace.compact_values.resize(need, 0.0);
        }
        for c in 0..ncols {
            let src = frame.column(c);
            let dst = &mut workspace.compact_values[c * n_keep..(c + 1) * n_keep];
            let mut j = 0usize;
            for (i, &k) in keep.iter().enumerate() {
                if k {
                    dst[j] = src[i];
                    j += 1;
                }
            }
        }
        let plan = causal_stats::CiPreparationPlan {
            significance: self.constraints.significance,
            confidence: ConfidenceMethod::default(),
        };
        {
            let cols: Vec<&[f64]> = (0..ncols)
                .map(|c| &workspace.compact_values[c * n_keep..(c + 1) * n_keep])
                .collect();
            workspace.prepared_ci =
                Some(self.ci.prepare(&cols, &plan, ctx).map_err(DiscoveryError::from)?);
        }
        let cols: Vec<&[f64]> = (0..ncols)
            .map(|c| &workspace.compact_values[c * n_keep..(c + 1) * n_keep])
            .collect();
        let queries = [CiQuery {
            x: xi,
            y: yi,
            z_start: 0,
            z_len: workspace.z_flat.len(),
        }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &workspace.z_flat,
            significance: self.constraints.significance,
            confidence: ConfidenceMethod::default(),
        };
        let prepared = workspace.prepared_ci.as_ref().ok_or_else(|| {
            DiscoveryError::Unsupported { message: "CI test used before prepare()" }
        })?;
        let out = self
            .ci
            .test_batch(prepared, &req, &mut workspace.ci, ctx)
            .map_err(DiscoveryError::from)?;
        let result = out
            .results
            .into_iter()
            .next()
            .ok_or_else(|| DiscoveryError::stats_msg("CI batch returned no results"))?;
        if !result.statistic.is_finite() || !result.p_value.is_finite() {
            return Err(DiscoveryError::stats_msg("non-finite CI statistic or p-value"));
        }
        Ok((result.statistic, result.p_value))
    }
}

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

/// Build the MCI conditioning set for `link` into `out`: parents of the target minus the
/// link endpoints, then parents of the source *time-shifted by the source lag* (a parent
/// `(v, l)` of `X_t` is `(v, l+τ)` for `X_{t−τ}`). Both inputs are strongest-first, so the
/// `MAX_CI_COLS` truncation keeps the strongest conditions; returns how many were dropped.
pub(crate) fn mci_conditioning(
    link: LaggedLink,
    parents_target: &[(VariableId, Lag)],
    parents_source: &[(VariableId, Lag)],
    out: &mut Vec<(VariableId, Lag)>,
) -> u64 {
    out.clear();
    let src_key = (link.source, link.source_lag);
    let tgt_key = (link.target, link.target_lag);
    out.extend(parents_target.iter().copied().filter(|p| *p != src_key && *p != tgt_key));
    for &(v, l) in parents_source {
        let shifted = (v, Lag::from_raw(l.raw() + link.source_lag.raw()));
        if !out.contains(&shifted) && shifted != src_key && shifted != tgt_key {
            out.push(shifted);
        }
    }
    if out.len() > MAX_CI_COLS - 2 {
        let dropped = out.len() - (MAX_CI_COLS - 2);
        out.truncate(MAX_CI_COLS - 2);
        dropped as u64
    } else {
        0
    }
}

/// Inclusive-exclusive index ranges for target-wise parallel work.
fn chunk_ranges(n: usize, threads: u32) -> Vec<(usize, usize)> {
    if n == 0 {
        return Vec::new();
    }
    let n_threads = (threads as usize).min(n).max(1);
    let chunk = n.div_ceil(n_threads);
    let mut out = Vec::with_capacity(n_threads);
    let mut start = 0;
    while start < n {
        let end = (start + chunk).min(n);
        out.push((start, end));
        start = end;
    }
    out
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
