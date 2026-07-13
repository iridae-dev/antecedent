//! Shared PCMCI engine: PC-style parents + MCI (DESIGN.md §13.4).
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
use causal_data::{LaggedColumn, SampleWorkspace, TimeSeriesData};
use causal_stats::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependence, PartialCorrelation,
    SignificanceMethod,
};

use crate::combinations::combinations;
use crate::constraints::DiscoveryConstraints;
use crate::error::DiscoveryError;
use crate::evidence::graph_evidence_from_scored;
use crate::result::{
    AlgorithmRecord, DiscoveryIteration, DiscoveryPerformanceRecord, DiscoveryResult, LaggedLink,
    ScoredLink,
};

/// Reusable target-local discovery workspace.
#[derive(Clone, Debug, Default)]
pub struct DiscoveryWorkspace {
    /// Sample gather workspace.
    pub sample: SampleWorkspace,
    /// CI workspace.
    pub ci: CiWorkspace,
    /// Scratch parents list.
    pub parents: Vec<(VariableId, Lag)>,
    /// Scratch combinations of parent indexes.
    pub combo: Vec<usize>,
}

/// Shared PCMCI engine core.
#[derive(Clone, Debug)]
pub struct PcmciEngine {
    /// Constraints / alpha / lags.
    pub constraints: DiscoveryConstraints,
    /// CI test.
    pub ci: PartialCorrelation,
}

impl Default for PcmciEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PcmciEngine {
    /// Default engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            constraints: DiscoveryConstraints::default(),
            ci: PartialCorrelation::new(),
        }
    }

    /// With constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// PC-style parent selection for one target (contemporaneous).
    ///
    /// # Errors
    ///
    /// Data or CI failures.
    pub fn select_parents(
        &self,
        data: &TimeSeriesData,
        target: VariableId,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<(VariableId, Lag)>, u64), DiscoveryError> {
        let mut parents = self.constraints.candidate_sources(variables, target);
        if let Some(max_p) = self.constraints.max_parents {
            if parents.len() > max_p {
                parents.truncate(max_p);
            }
        }
        let mut ci_tests = 0u64;
        let max_cond = self.constraints.max_cond_size;
        for cond_size in 0..=max_cond {
            let mut removed = Vec::new();
            for (pi, &(src, slag)) in parents.iter().enumerate() {
                let others: Vec<(VariableId, Lag)> =
                    parents.iter().enumerate().filter(|(j, _)| *j != pi).map(|(_, x)| *x).collect();
                if others.len() < cond_size {
                    continue;
                }
                let combos = combinations(&others, cond_size);
                for cond in combos.iter().take(32) {
                    let indep = self.ci_independent(
                        data,
                        src,
                        slag,
                        target,
                        Lag::CONTEMPORANEOUS,
                        cond,
                        workspace,
                        ctx,
                    )?;
                    ci_tests += 1;
                    if indep {
                        removed.push((src, slag));
                        break;
                    }
                }
            }
            parents.retain(|p| !removed.contains(p));
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
        data: &TimeSeriesData,
        link: LaggedLink,
        parents_target: &[(VariableId, Lag)],
        parents_source: &[(VariableId, Lag)],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<ScoredLink, DiscoveryError> {
        let mut cond: Vec<(VariableId, Lag)> = parents_target
            .iter()
            .copied()
            .filter(|p| *p != (link.source, link.source_lag))
            .collect();
        for p in parents_source {
            if !cond.contains(p) && *p != (link.source, link.source_lag) {
                cond.push(*p);
            }
        }
        let (stat, p) = self.ci_statistic(
            data,
            link.source,
            link.source_lag,
            link.target,
            link.target_lag,
            &cond,
            workspace,
            ctx,
        )?;
        Ok(ScoredLink { link, statistic: stat, p_value: p })
    }

    /// Run PC parents for all targets then MCI on surviving links.
    ///
    /// # Errors
    ///
    /// Data / CI / graph construction failures.
    pub fn run_pc_mci(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DiscoveryResult, DiscoveryError> {
        let mut all_parents: Vec<(VariableId, Vec<(VariableId, Lag)>)> = Vec::new();
        let mut iterations = Vec::new();
        let mut ci_tests = 0u64;
        for &target in variables {
            let (parents, tests) =
                self.select_parents(data, target, variables, workspace, ctx)?;
            ci_tests += tests;
            iterations.push(DiscoveryIteration {
                label: Arc::from(format!("pc_parents:{target}")),
                ci_tests: tests,
            });
            all_parents.push((target, parents));
        }

        let mut scored = Vec::new();
        for (target, parents) in &all_parents {
            for &(src, slag) in parents {
                let link = LaggedLink {
                    source: src,
                    source_lag: slag,
                    target: *target,
                    target_lag: Lag::CONTEMPORANEOUS,
                };
                let src_parents = all_parents
                    .iter()
                    .find(|(t, _)| *t == src)
                    .map_or(&[][..], |(_, p)| p.as_slice());
                let s = self.mci_test(data, link, parents, src_parents, workspace, ctx)?;
                ci_tests += 1;
                if s.p_value < self.constraints.alpha {
                    scored.push(s);
                }
            }
        }

        let n_links = scored.len() as u64;
        Ok(DiscoveryResult {
            evidence: graph_evidence_from_scored(scored)?,
            algorithm: AlgorithmRecord {
                id: Arc::from("pcmci.engine.pc_mci"),
                config: Arc::from(format!(
                    "alpha={},max_lag={}",
                    self.constraints.alpha,
                    self.constraints.temporal.max_lag.raw()
                )),
            },
            assumptions: AssumptionSet::new(),
            iterations,
            diagnostics: Vec::new(),
            performance: DiscoveryPerformanceRecord {
                ci_tests,
                links_retained: n_links,
                targets: variables.len() as u64,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn ci_independent(
        &self,
        data: &TimeSeriesData,
        x: VariableId,
        x_lag: Lag,
        y: VariableId,
        y_lag: Lag,
        cond: &[(VariableId, Lag)],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<bool, DiscoveryError> {
        let (_, p) = self.ci_statistic(data, x, x_lag, y, y_lag, cond, workspace, ctx)?;
        Ok(p >= self.constraints.alpha)
    }

    #[allow(clippy::too_many_arguments)]
    fn ci_statistic(
        &self,
        data: &TimeSeriesData,
        x: VariableId,
        x_lag: Lag,
        y: VariableId,
        y_lag: Lag,
        cond: &[(VariableId, Lag)],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(f64, f64), DiscoveryError> {
        let max_lag = self.constraints.temporal.max_lag.raw();
        let mut cols = Vec::with_capacity(2 + cond.len());
        cols.push(LaggedColumn { variable: x, lag: x_lag });
        cols.push(LaggedColumn { variable: y, lag: y_lag });
        for &(v, l) in cond {
            cols.push(LaggedColumn { variable: v, lag: l });
        }
        let plan = data
            .plan_lagged_sample(max_lag, Arc::<[LaggedColumn]>::from(cols))
            .map_err(|e| DiscoveryError::Data(e.to_string()))?;
        let prep = plan
            .prepare(data, &mut workspace.sample)
            .map_err(|e| DiscoveryError::Data(e.to_string()))?;
        let col_refs: Vec<&[f64]> = (0..prep.ncols).map(|c| prep.column(c)).collect();
        let z_flat: Vec<usize> = (2..prep.ncols).collect();
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: z_flat.len() }];
        let req = CiBatchRequest {
            columns: &col_refs,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
        };
        let Ok(out) = self.ci.test_batch(&req, &mut workspace.ci, ctx) else {
            return Ok((0.0, 1.0));
        };
        let Some(r) = out.results.first() else {
            return Ok((0.0, 1.0));
        };
        if !r.statistic.is_finite() || !r.p_value.is_finite() {
            return Ok((0.0, 1.0));
        }
        Ok((r.statistic, r.p_value))
    }
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
