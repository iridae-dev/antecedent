//! Shared PCMCI engine: PC-style parents + MCI (DESIGN.md §13.4 / §13.8).
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

use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use causal_data::{LaggedFrame, TimeSeriesData};
use causal_graph::TemporalGraphReview;
use causal_stats::{
    CiBatchRequest, CiQuery, CiResult, CiWorkspace, ConditionalIndependence, ConfidenceMethod, PartialCorrelation,
};

use crate::combinations::for_each_combination;
use crate::constraints::{CompiledConstraints, DiscoveryConstraints};
use crate::error::DiscoveryError;
use crate::evidence::graph_evidence_from_scored_with_sepsets;
use crate::result::{
        AlgorithmRecord, DagDiscoveryResult, DiscoveryIteration, DiscoveryPerformanceRecord,
        LaggedLink,
    PcSepsets, ScoredLink,
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
}

/// Reusable target-local discovery workspace.
#[derive(Clone, Debug, Default)]
pub struct DiscoveryWorkspace {
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
}

/// Shared PCMCI engine core.
#[derive(Clone)]
pub struct PcmciEngine {
    /// Constraints / alpha / lags.
    pub constraints: DiscoveryConstraints,
    /// Pluggable CI test (defaults to partial correlation).
    pub ci: Arc<dyn ConditionalIndependence + Send + Sync>,
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

    /// PC-style parent selection for one target.
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
        for cond_size in 0..=max_cond {
            workspace.removed.clear();
            for pi in 0..parents.len() {
                let (src, slag) = parents[pi];
                let mut others = std::mem::take(&mut workspace.others);
                others.clear();
                others
                    .extend(parents.iter().enumerate().filter(|(j, _)| *j != pi).map(|(_, x)| *x));
                if others.len() < cond_size {
                    workspace.others = others;
                    continue;
                }

                let mut combo = std::mem::take(&mut workspace.combo);
                let mut indep = false;
                let mut sep_for_removal: Option<Arc<[(VariableId, Lag)]>> = None;
                let mut err: Option<DiscoveryError> = None;
                for_each_combination(&others, cond_size, &mut combo, |cond| {
                    match self.ci_independent(
                        frame,
                        src,
                        slag,
                        target,
                        Lag::CONTEMPORANEOUS,
                        cond,
                        workspace,
                        ctx,
                    ) {
                        Ok(true) => {
                            ci_tests += 1;
                            indep = true;
                            sep_for_removal = Some(Arc::from(cond.to_vec().into_boxed_slice()));
                            false
                        }
                        Ok(false) => {
                            ci_tests += 1;
                            true
                        }
                        Err(e) => {
                            err = Some(e);
                            false
                        }
                    }
                });
                workspace.combo = combo;
                workspace.others = others;
                if let Some(e) = err {
                    return Err(e);
                }
                if indep {
                    let link = LaggedLink {
                        source: src,
                        source_lag: slag,
                        target,
                        target_lag: Lag::CONTEMPORANEOUS,
                    };
                    if !compiled.requires(link) {
                        workspace.removed.push((src, slag));
                        if let Some(sep) = sep_for_removal {
                            workspace
                                .sepsets
                                .insert((src, slag, target, Lag::CONTEMPORANEOUS), sep);
                        }
                    }
                }
            }
            parents.retain(|p| !workspace.removed.contains(p));
            if parents.is_empty() {
                break;
            }
        }
        Ok((parents, ci_tests))
    }

    /// MCI test for a candidate link given parent sets.
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
    ) -> Result<ScoredLink, DiscoveryError> {
        workspace.others.clear();
        let src_key = (link.source, link.source_lag);
        let tgt_key = (link.target, link.target_lag);
        workspace
            .others
            .extend(parents_target.iter().copied().filter(|p| *p != src_key && *p != tgt_key));
        for p in parents_source {
            if !workspace.others.contains(p) && *p != src_key && *p != tgt_key {
                workspace.others.push(*p);
            }
        }
        if workspace.others.len() > MAX_CI_COLS - 2 {
            workspace.others.truncate(MAX_CI_COLS - 2);
        }
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
        Ok(ScoredLink { link, statistic: stat, p_value: p })
    }

    /// Run PC parents for all targets then MCI on surviving links.
    ///
    /// Returns **unthresholded** MCI scores (full MCI family). Callers apply
    /// alpha and optional FDR.
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
        let frame = LaggedFrame::from_series(data, variables, max_lag)
            .map_err(|e| DiscoveryError::Data(e.to_string()))?;
        if let Some(hard) = ctx.memory.hard_limit_bytes {
            if frame.values_bytes() > hard {
                return Err(DiscoveryError::Unsupported {
                    message: "lagged frame exceeds ExecutionContext memory hard limit",
                });
            }
        }

        let threads = ctx.parallelism.max_threads.get().max(1);
        let compiled = self.constraints.compile(variables)?;

        // DESIGN.md §12: prepare CI once for the lagged frame.
        {
            let cols: Vec<&[f64]> = (0..frame.ncols()).map(|i| frame.column(i)).collect();
            let plan = causal_stats::CiPreparationPlan {
                significance: self.constraints.significance,
                confidence: ConfidenceMethod::default(),
            };
            let _prepared = self
                .ci
                .prepare(&cols, &plan, ctx)
                .map_err(|e| DiscoveryError::Stats(e.to_string()))?;
        }

        let (all_parents, iterations, mut ci_tests) =
            self.select_parents_all(&frame, variables, &compiled, workspace, ctx, threads)?;

        let mut scored = Vec::new();
        let mci_tests = self.mci_all(&frame, &all_parents, &mut scored, workspace, ctx, threads)?;
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
        Ok(DagDiscoveryResult {
            evidence,
            review,
            algorithm,
            assumptions: AssumptionSet::new(),
            iterations,
            diagnostics: Vec::new(),
            performance: DiscoveryPerformanceRecord {
                ci_tests,
                links_retained: n_links,
                targets: variables.len() as u64,
                lagged_frame_bytes: frame.values_bytes(),
                worker_threads: threads,
            },
            sepsets,
        })
    }

    fn select_parents_all(
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

        std::thread::scope(|scope| {
            let mut rest = slots.as_mut_slice();
            let mut cursor = 0usize;
            for (start, end) in chunk_ranges(n, threads) {
                let (this, next) = rest.split_at_mut(end - start);
                debug_assert_eq!(cursor, start);
                let chunk_vars = &variables[start..end];
                let engine = self;
                scope.spawn(move || {
                    let mut local_ws = DiscoveryWorkspace::default();
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

    fn mci_all(
        &self,
        frame: &LaggedFrame,
        all_parents: &[TargetParents],
        scored: &mut Vec<ScoredLink>,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
        threads: u32,
    ) -> Result<u64, DiscoveryError> {
        if threads <= 1 || all_parents.len() <= 1 {
            let mut tests = 0u64;
            for (target, parents) in all_parents {
                let batch = self.mci_batch_for_target(
                    frame,
                    *target,
                    parents,
                    all_parents,
                    workspace,
                    ctx,
                )?;
                tests += batch.len() as u64;
                scored.extend(batch);
            }
            return Ok(tests);
        }

        let n = all_parents.len();
        let ranges = chunk_ranges(n, threads);
        let mut partials: Vec<Option<Result<MciChunkOut, DiscoveryError>>> =
            Vec::with_capacity(ranges.len());

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(ranges.len());
            for &(start, end) in &ranges {
                let chunk_parents = &all_parents[start..end];
                let engine = self;
                handles.push(scope.spawn(move || {
                    let mut local_ws = DiscoveryWorkspace::default();
                    let mut local_scored = Vec::new();
                    let mut tests = 0u64;
                    for (target, parents) in chunk_parents {
                        let batch = engine.mci_batch_for_target(
                            frame,
                            *target,
                            parents,
                            all_parents,
                            &mut local_ws,
                            ctx,
                        )?;
                        tests += batch.len() as u64;
                        local_scored.extend(batch);
                    }
                    Ok(MciChunkOut {
                        scored: local_scored,
                        tests,
                    })
                }));
            }
            for h in handles {
                partials.push(Some(h.join().expect("MCI worker panicked")));
            }
        });

        let mut tests = 0u64;
        for slot in partials {
            let chunk = slot.ok_or(DiscoveryError::Unsupported {
                message: "parallel MCI worker left empty slot",
            })??;
            tests += chunk.tests;
            scored.extend(chunk.scored);
        }
        Ok(tests)
    }

    /// MCI-test all parents of one target in a single CI batch (DESIGN.md §12.1).
    fn mci_batch_for_target(
        &self,
        frame: &LaggedFrame,
        target: VariableId,
        parents: &[(VariableId, Lag)],
        all_parents: &[TargetParents],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Vec<ScoredLink>, DiscoveryError> {
        if parents.is_empty() {
            return Ok(Vec::new());
        }
        let cols: Vec<&[f64]> = (0..frame.ncols()).map(|i| frame.column(i)).collect();
        let mut queries = Vec::with_capacity(parents.len());
        let mut z_flat = Vec::new();
        let mut links = Vec::with_capacity(parents.len());

        for &(src, slag) in parents {
            let link = link_to_target(src, slag, target);
            let src_parents = parents_of(all_parents, src);
            workspace.others.clear();
            let src_key = (link.source, link.source_lag);
            let tgt_key = (link.target, link.target_lag);
            workspace
                .others
                .extend(parents.iter().copied().filter(|p| *p != src_key && *p != tgt_key));
            for p in src_parents {
                if !workspace.others.contains(p) && *p != src_key && *p != tgt_key {
                    workspace.others.push(*p);
                }
            }
            if workspace.others.len() > MAX_CI_COLS - 2 {
                workspace.others.truncate(MAX_CI_COLS - 2);
            }

            let xi = frame.column_index(link.source, link.source_lag).ok_or_else(|| {
                DiscoveryError::Data(format!("missing lagged column for {:?}", link.source))
            })?;
            let yi = frame.column_index(link.target, link.target_lag).ok_or_else(|| {
                DiscoveryError::Data(format!("missing lagged column for {:?}", link.target))
            })?;
            let z_start = z_flat.len();
            for &(v, l) in &workspace.others {
                let zi = frame.column_index(v, l).ok_or_else(|| {
                    DiscoveryError::Data(format!("missing lagged column for {v:?} lag {l:?}"))
                })?;
                z_flat.push(zi);
            }
            queries.push(CiQuery {
                x: xi,
                y: yi,
                z_start,
                z_len: workspace.others.len(),
            });
            links.push(link);
        }

        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: self.constraints.significance,
            confidence: ConfidenceMethod::default(),
        };
        let out = self
            .ci
            .test_batch(&req, &mut workspace.ci, ctx)
            .map_err(|e| DiscoveryError::Stats(e.to_string()))?;
        if out.results.len() != links.len() {
            return Err(DiscoveryError::Stats("CI batch result length mismatch".into()));
        }
        let mut scored = Vec::with_capacity(links.len());
        for (link, result) in links.into_iter().zip(out.results) {
            if !result.statistic.is_finite() || !result.p_value.is_finite() {
                return Err(DiscoveryError::Stats("non-finite CI statistic or p-value".into()));
            }
            scored.push(ScoredLink {
                link,
                statistic: result.statistic,
                p_value: result.p_value,
            });
        }
        Ok(scored)
    }

    #[allow(clippy::too_many_arguments)]
    fn ci_independent(
        &self,
        frame: &LaggedFrame,
        x: VariableId,
        x_lag: Lag,
        y: VariableId,
        y_lag: Lag,
        cond: &[(VariableId, Lag)],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<bool, DiscoveryError> {
        let (_, p) = self.ci_statistic(frame, x, x_lag, y, y_lag, cond, workspace, ctx)?;
        Ok(p >= self.constraints.alpha)
    }

    #[allow(clippy::too_many_arguments)]
    fn ci_statistic(
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
        workspace.col_idxs.clear();
        let xi = frame.column_index(x, x_lag).ok_or_else(|| {
            DiscoveryError::Data(format!("missing lagged column for {x:?} lag {x_lag:?}"))
        })?;
        let yi = frame.column_index(y, y_lag).ok_or_else(|| {
            DiscoveryError::Data(format!("missing lagged column for {y:?} lag {y_lag:?}"))
        })?;
        workspace.col_idxs.push(xi);
        workspace.col_idxs.push(yi);
        for &(v, l) in cond {
            let zi = frame.column_index(v, l).ok_or_else(|| {
                DiscoveryError::Data(format!("missing lagged column for {v:?} lag {l:?}"))
            })?;
            workspace.col_idxs.push(zi);
        }
        workspace.z_flat.clear();
        workspace.z_flat.extend(2..workspace.col_idxs.len());

        let mut col_buf: [&[f64]; MAX_CI_COLS] = [&[]; MAX_CI_COLS];
        let ncols = workspace.col_idxs.len();
        for (i, &idx) in workspace.col_idxs.iter().enumerate() {
            col_buf[i] = frame.column(idx);
        }
        let col_refs = &col_buf[..ncols];

        let result: CiResult = {
            let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: workspace.z_flat.len() }];
            let req = CiBatchRequest {
                columns: col_refs,
                queries: &queries,
                z_flat: &workspace.z_flat,
                significance: self.constraints.significance,
                confidence: ConfidenceMethod::default(),
            };
            let out = self
                .ci
                .test_batch(&req, &mut workspace.ci, ctx)
                .map_err(|e| DiscoveryError::Stats(e.to_string()))?;
            out.results
                .into_iter()
                .next()
                .ok_or_else(|| DiscoveryError::Stats("CI batch returned no results".into()))?
        };
        if !result.statistic.is_finite() || !result.p_value.is_finite() {
            return Err(DiscoveryError::Stats("non-finite CI statistic or p-value".into()));
        }
        Ok((result.statistic, result.p_value))
    }
}

fn parents_of(all_parents: &[TargetParents], src: VariableId) -> &[(VariableId, Lag)] {
    all_parents.iter().find(|(t, _)| *t == src).map_or(&[][..], |(_, p)| p.as_slice())
}

fn link_to_target(src: VariableId, slag: Lag, target: VariableId) -> LaggedLink {
    LaggedLink { source: src, source_lag: slag, target, target_lag: Lag::CONTEMPORANEOUS }
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
