//! Structure MCMC for DAG posteriors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::sync::Arc;

use causal_core::{CausalRng, ExecutionContext, VariableId};
use causal_data::TabularData;
use causal_state::{GraphScoreCacheKey, GraphScoreFamily, LocalScoreCache};

use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::graph_mcmc::{FinishMaskPosterior, GraphMcmcSchedule, run_parallel_mask_chains};
use crate::graph_posterior::{
    GraphPosterior, GraphPosteriorEngine, GraphPrior, has_edge, mask_is_dag, n_directed_edges,
    set_edge,
};
use crate::graph_score::{score_dag_mask, tabular_score_data};

/// Structure MCMC with single-edge add / delete / reverse proposals.
#[derive(Clone, Debug)]
pub struct StructureMcmc {
    /// Number of chains (≥ 2 for R-hat).
    pub n_chains: u32,
    /// Warmup draws discarded per chain.
    pub n_warmup: u32,
    /// Post-warmup draws retained per chain (before thinning).
    pub n_draws: u32,
    /// Keep every `thin`-th post-warmup draw.
    pub thin: u32,
    /// Optional undirected candidate pairs `(lo, hi)` from CI screening.
    pub candidate_pairs: Option<Arc<[(u32, u32)]>>,
    /// When true (default), refuse if [`crate::graph_posterior::allows_graph_posterior`] fails.
    pub require_diagnostics_gate: bool,
}

impl Default for StructureMcmc {
    fn default() -> Self {
        Self::new()
    }
}

impl StructureMcmc {
    /// Default sampler (4 chains × 500 warmup × 1000 draws, thin 1).
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_chains: 4,
            n_warmup: 500,
            n_draws: 1000,
            thin: 1,
            candidate_pairs: None,
            require_diagnostics_gate: true,
        }
    }

    /// Configure chain lengths.
    #[must_use]
    pub fn with_schedule(mut self, n_chains: u32, n_warmup: u32, n_draws: u32, thin: u32) -> Self {
        let s = GraphMcmcSchedule::normalize(n_chains, n_warmup, n_draws, thin);
        self.n_chains = s.n_chains;
        self.n_warmup = s.n_warmup;
        self.n_draws = s.n_draws;
        self.thin = s.thin;
        self
    }

    /// Restrict proposals to undirected candidate pairs.
    #[must_use]
    pub fn with_candidate_pairs(mut self, pairs: impl Into<Arc<[(u32, u32)]>>) -> Self {
        self.candidate_pairs = Some(pairs.into());
        self
    }

    /// Enable / disable the multi-chain diagnostics publish gate.
    #[must_use]
    pub fn with_diagnostics_gate(mut self, require: bool) -> Self {
        self.require_diagnostics_gate = require;
        self
    }

    fn schedule(&self) -> GraphMcmcSchedule {
        GraphMcmcSchedule {
            n_chains: self.n_chains,
            n_warmup: self.n_warmup,
            n_draws: self.n_draws,
            thin: self.thin,
        }
    }

    /// Run structure MCMC.
    ///
    /// # Errors
    ///
    /// Data/score failures or diagnostics gate refusal.
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        prior: &GraphPrior,
        score_family: GraphScoreFamily,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosterior, DiscoveryError> {
        let _ = workspace;
        prior.constraints.validate()?;
        let n = variables.len();
        if n == 0 {
            return Err(DiscoveryError::unsupported(
                "structure MCMC requires at least one variable",
            ));
        }
        if n_directed_edges(n) > 63 {
            return Err(DiscoveryError::unsupported(
                "structure MCMC adjacency exceeds 63 directed edges",
            ));
        }
        if !matches!(score_family, GraphScoreFamily::GaussianBic) {
            return Err(DiscoveryError::unsupported(
                "structure MCMC currently supports GaussianBic only",
            ));
        }
        let schedule = self.schedule();
        schedule.require_min_chains(2, "structure MCMC requires at least 2 chains for R-hat")?;

        let score_data = tabular_score_data(data, variables)?;
        let pairs = candidate_directed_list(n, self.candidate_pairs.as_deref());
        if pairs.is_empty() {
            return Err(DiscoveryError::unsupported(
                "structure MCMC has empty candidate edge list",
            ));
        }

        let (n_chains, n_warmup, n_draws, thin) = schedule.as_usize();
        let n_params = pairs.len();
        let threads = ctx.parallelism.max_threads.get().max(1) as usize;

        let (traces, sample_masks, rejected) =
            run_parallel_mask_chains(n_chains, n_draws, n_params, threads, |start, end| {
                let mut local_traces = vec![0.0f64; (end - start) * n_draws * n_params];
                let mut local_samples = vec![Vec::new(); end - start];
                let mut local_rej = 0u64;
                for li in 0..(end - start) {
                    let chain = start + li;
                    let mut rng = ctx.rng.stream(1000 + chain as u64);
                    let mut cache = LocalScoreCache::new(GraphScoreCacheKey {
                        data_version: 1,
                        family: score_family,
                        var_fingerprint: n as u64,
                        penalty_fingerprint: score_data.n_rows as u64,
                    });
                    let mut mask = 0u64;
                    let mut cur =
                        score_dag_mask(mask, n, &score_data, &mut cache, prior, variables)
                            .unwrap_or(f64::NEG_INFINITY);
                    let total_steps = n_warmup + n_draws * thin;
                    let mut kept = 0usize;
                    for step in 0..total_steps {
                        let (prop, q_ratio, rej_inc) = propose_structure(mask, n, &pairs, &mut rng);
                        local_rej += rej_inc;
                        let prop_score =
                            score_dag_mask(prop, n, &score_data, &mut cache, prior, variables);
                        let accept = match prop_score {
                            Some(ps) if cur.is_finite() => {
                                let log_r = ps - cur + q_ratio.ln();
                                log_r >= 0.0 || rng.next_f64() < log_r.exp()
                            }
                            Some(ps) if !cur.is_finite() => {
                                cur = ps;
                                mask = prop;
                                false
                            }
                            _ => false,
                        };
                        if accept {
                            if let Some(ps) = prop_score {
                                mask = prop;
                                cur = ps;
                            }
                        }
                        if step >= n_warmup && (step - n_warmup) % thin == 0 && kept < n_draws {
                            for (p, &(i, j)) in pairs.iter().enumerate() {
                                let v = if has_edge(mask, n, i, j) { 1.0 } else { 0.0 };
                                local_traces[(li * n_draws + kept) * n_params + p] = v;
                            }
                            local_samples[li].push(mask);
                            kept += 1;
                        }
                    }
                }
                (start, local_traces, local_samples, local_rej)
            });

        FinishMaskPosterior {
            n,
            schedule: &schedule,
            traces: &traces,
            sample_masks: &sample_masks,
            rejected,
            n_params,
            require_gate: self.require_diagnostics_gate,
            refuse_msg: "structure MCMC diagnostics gate refused posterior",
            empty_msg: "structure MCMC produced no samples",
        }
        .run()
    }
}

impl GraphPosteriorEngine for StructureMcmc {
    fn infer_graphs(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        prior: &GraphPrior,
        score_family: GraphScoreFamily,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosterior, DiscoveryError> {
        let mut ws = DiscoveryWorkspace::default();
        self.run(data, variables, prior, score_family, &mut ws, ctx)
    }
}

fn candidate_directed_list(n: usize, undirected: Option<&[(u32, u32)]>) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    match undirected {
        None => {
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        out.push((i, j));
                    }
                }
            }
        }
        Some(pairs) => {
            for &(lo, hi) in pairs {
                let a = lo as usize;
                let b = hi as usize;
                if a < n && b < n && a != b {
                    out.push((a, b));
                    out.push((b, a));
                }
            }
        }
    }
    out
}

/// Propose add/delete/reverse; returns `(new_mask, q(old|new)/q(new|old) approx, rejected)`.
fn propose_structure(
    mask: u64,
    n: usize,
    pairs: &[(usize, usize)],
    rng: &mut CausalRng,
) -> (u64, f64, u64) {
    if pairs.is_empty() {
        return (mask, 1.0, 0);
    }
    let idx = (rng.next_u64() as usize) % pairs.len();
    let (i, j) = pairs[idx];
    let forward = has_edge(mask, n, i, j);
    let backward = has_edge(mask, n, j, i);
    let u = rng.next_f64();
    if !forward && !backward {
        let prop = set_edge(mask, n, i, j, true);
        if mask_is_dag(prop, n) { (prop, 1.0, 0) } else { (mask, 1.0, 1) }
    } else if forward && !backward {
        if u < 0.5 {
            (set_edge(mask, n, i, j, false), 1.0, 0)
        } else {
            let mut prop = set_edge(mask, n, i, j, false);
            prop = set_edge(prop, n, j, i, true);
            if mask_is_dag(prop, n) { (prop, 1.0, 0) } else { (mask, 1.0, 1) }
        }
    } else if backward && !forward {
        if u < 0.5 {
            (set_edge(mask, n, j, i, false), 1.0, 0)
        } else {
            let mut prop = set_edge(mask, n, j, i, false);
            prop = set_edge(prop, n, i, j, true);
            if mask_is_dag(prop, n) { (prop, 1.0, 0) } else { (mask, 1.0, 1) }
        }
    } else {
        (set_edge(mask, n, i, j, false), 1.0, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use causal_core::{CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };

    use crate::graph_posterior::GraphPrior;

    fn fork_data(n_rows: usize) -> (TabularData, Vec<VariableId>) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "y", "z"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let vars: Vec<_> = (0..3).map(VariableId::from_raw).collect();
        let mut rng = causal_core::CausalRng::from_seed(11);
        let mut x = vec![0.0; n_rows];
        let mut y = vec![0.0; n_rows];
        let mut z = vec![0.0; n_rows];
        for i in 0..n_rows {
            x[i] = rng.next_f64() * 2.0 - 1.0;
            y[i] = 1.4 * x[i] + 0.2 * (rng.next_f64() * 2.0 - 1.0);
            z[i] = 1.1 * x[i] + 0.2 * (rng.next_f64() * 2.0 - 1.0);
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(vars[0], Arc::from(x), ValidityBitmap::all_valid(n_rows))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(vars[1], Arc::from(y), ValidityBitmap::all_valid(n_rows))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(vars[2], Arc::from(z), ValidityBitmap::all_valid(n_rows))
                    .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        (TabularData::new(storage), vars)
    }

    #[test]
    fn structure_mcmc_fork_edges() {
        let (data, vars) = fork_data(250);
        let eng = StructureMcmc::new().with_schedule(2, 200, 400, 1);
        let ctx = ExecutionContext::for_tests(42);
        let mut ws = DiscoveryWorkspace::default();
        let post = eng
            .run(&data, &vars, &GraphPrior::uniform(), GraphScoreFamily::GaussianBic, &mut ws, &ctx)
            .unwrap();
        let n = 3;
        let sk_xy = post.edge_marginals[1] + post.edge_marginals[n];
        let sk_xz = post.edge_marginals[2] + post.edge_marginals[2 * n];
        assert!(sk_xy > 0.4, "P(X—Y)={sk_xy}");
        assert!(sk_xz > 0.4, "P(X—Z)={sk_xz}");
        assert!(crate::graph_posterior::allows_graph_posterior(&post.diagnostics));
    }
}
