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
use causal_graph::{TemporalDag, ensure_lagged};
use causal_stats::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependence, PartialCorrelation,
    SignificanceMethod,
};

use crate::constraints::DiscoveryConstraints;
use crate::error::DiscoveryError;
use crate::result::{
    AlgorithmRecord, DiscoveryIteration, DiscoveryPerformanceRecord, DiscoveryResult,
    GraphEvidence, LaggedLink, ScoredLink,
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
                // Test against each combination of size cond_size (cap combinations).
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

        let mut graph = TemporalDag::empty();
        for s in &scored {
            let from = ensure_lagged(&mut graph, s.link.source, s.link.source_lag)
                .map_err(|e| DiscoveryError::Data(e.to_string()))?;
            let to = ensure_lagged(&mut graph, s.link.target, s.link.target_lag)
                .map_err(|e| DiscoveryError::Data(e.to_string()))?;
            let _ = graph.insert_directed(from, to);
        }

        let n_links = scored.len() as u64;
        Ok(DiscoveryResult {
            evidence: GraphEvidence { graph, links: Arc::from(scored) },
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
        let out = self
            .ci
            .test_batch(&req, &mut workspace.ci, ctx)
            .map_err(|e| DiscoveryError::Stats(e.to_string()))?;
        let r = out.results[0];
        Ok((r.statistic, r.p_value))
    }
}

fn combinations(items: &[(VariableId, Lag)], k: usize) -> Vec<Vec<(VariableId, Lag)>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if k > items.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.iter().map(|&i| items[i]).collect());
        // next combination
        let mut i = k;
        while i > 0 {
            i -= 1;
            if idx[i] != i + items.len() - k {
                idx[i] += 1;
                for j in i + 1..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return out;
            }
        }
        if k == 0 {
            break;
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };

    use super::*;
    use crate::constraints::{DiscoveryConstraints, TemporalConstraints};

    fn var_series() -> (TimeSeriesData, Vec<VariableId>) {
        // Y_t = 0.8 X_{t-1} + noise; X_t = noise
        let n = 400usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = ((t as f64) * 0.01).sin();
            y[t] = 0.8 * x[t - 1] + 0.01 * ((t as f64) * 0.03).cos();
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        (data, vec![VariableId::from_raw(0), VariableId::from_raw(1)])
    }

    #[test]
    fn recovers_lagged_parent() {
        let (data, vars) = var_series();
        let engine = PcmciEngine::new().with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(2),
                min_lag: Lag::from_raw(1),
            },
            alpha: 0.05,
            max_cond_size: 2,
            ..DiscoveryConstraints::default()
        });
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(9);
        let result = engine.run_pc_mci(&data, &vars, &mut ws, &ctx).unwrap();
        let has = result.evidence.links.iter().any(|s| {
            s.link.source == VariableId::from_raw(0)
                && s.link.target == VariableId::from_raw(1)
                && s.link.source_lag.raw() == 1
        });
        assert!(has, "links={:?}", result.evidence.links);
    }
}
