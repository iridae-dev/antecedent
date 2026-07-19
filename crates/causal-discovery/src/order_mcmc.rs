//! Order MCMC for DAG posteriors.
//!
//! State = topological order + forward-edge subset. Proposals: adjacent
//! transposition (dropping conflicting edges) and forward-edge flips.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

use std::collections::HashMap;

use causal_core::{CausalRng, ExecutionContext, VariableId};
use causal_data::TabularData;
use causal_state::{GraphScoreCacheKey, GraphScoreFamily, LocalScoreCache};

use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::exact_enumeration::{score_dag_mask, tabular_score_data};
use crate::graph_posterior::{
    accumulate_marginals, allows_graph_posterior, graph_chain_diagnostics, has_edge, kish_ess,
    mcmc_graph_diagnostics, n_directed_edges, set_edge, GraphPosterior, GraphPosteriorEngine,
    GraphPrior,
};

/// Order MCMC over topological orders and compatible forward edges.
#[derive(Clone, Debug)]
pub struct OrderMcmc {
    /// Number of chains (≥ 2 for R-hat).
    pub n_chains: u32,
    /// Warmup draws discarded per chain.
    pub n_warmup: u32,
    /// Post-warmup draws retained per chain (before thinning).
    pub n_draws: u32,
    /// Keep every `thin`-th post-warmup draw.
    pub thin: u32,
    /// When true (default), refuse if [`allows_graph_posterior`] fails.
    pub require_diagnostics_gate: bool,
}

impl Default for OrderMcmc {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderMcmc {
    /// Default sampler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_chains: 4,
            n_warmup: 500,
            n_draws: 1000,
            thin: 1,
            require_diagnostics_gate: true,
        }
    }

    /// Configure chain lengths.
    #[must_use]
    pub fn with_schedule(mut self, n_chains: u32, n_warmup: u32, n_draws: u32, thin: u32) -> Self {
        self.n_chains = n_chains.max(1);
        self.n_warmup = n_warmup;
        self.n_draws = n_draws.max(4);
        self.thin = thin.max(1);
        self
    }

    /// Enable / disable the multi-chain diagnostics publish gate.
    #[must_use]
    pub fn with_diagnostics_gate(mut self, require: bool) -> Self {
        self.require_diagnostics_gate = require;
        self
    }

    /// Run order MCMC.
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
                "order MCMC requires at least one variable",
            ));
        }
        if n_directed_edges(n) > 63 {
            return Err(DiscoveryError::unsupported(
                "order MCMC adjacency exceeds 63 directed edges",
            ));
        }
        if !matches!(score_family, GraphScoreFamily::GaussianBic) {
            return Err(DiscoveryError::unsupported(
                "order MCMC currently supports GaussianBic only",
            ));
        }
        if self.n_chains < 2 {
            return Err(DiscoveryError::unsupported(
                "order MCMC requires at least 2 chains for R-hat",
            ));
        }

        let score_data = tabular_score_data(data, variables)?;
        let n_chains = self.n_chains as usize;
        let n_warmup = self.n_warmup as usize;
        let n_draws = self.n_draws as usize;
        let thin = self.thin as usize;
        let n_params = n_directed_edges(n);
        let mut traces = vec![0.0f64; n_chains * n_draws * n_params];
        let mut sample_masks: Vec<Vec<u64>> = vec![Vec::new(); n_chains];
        let mut rejected = 0u64;

        let threads = ctx.parallelism.max_threads.get().max(1) as usize;
        let chunk = (n_chains / threads).max(1);

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for start in (0..n_chains).step_by(chunk) {
                let end = (start + chunk).min(n_chains);
                let score_ref = &score_data;
                let vars_ref = variables;
                handles.push(scope.spawn(move || {
                    let mut local_traces = vec![0.0f64; (end - start) * n_draws * n_params];
                    let mut local_samples = vec![Vec::new(); end - start];
                    let mut local_rej = 0u64;
                    for li in 0..(end - start) {
                        let chain = start + li;
                        let mut rng = ctx.rng.stream(2000 + chain as u64);
                        let mut cache = LocalScoreCache::new(GraphScoreCacheKey {
                            data_version: 1,
                            family: score_family,
                            var_fingerprint: n as u64,
                            penalty_fingerprint: score_data.n_rows as u64,
                        });
                        let mut order: Vec<usize> = (0..n).collect();
                        // Shuffle initial order.
                        for i in (1..n).rev() {
                            let j = (rng.next_u64() as usize) % (i + 1);
                            order.swap(i, j);
                        }
                        // Seed a random forward-edge subset under this order.
                        let mut mask = 0u64;
                        let pos = position_map(&order);
                        for i in 0..n {
                            for j in 0..n {
                                if i != j && pos[i] < pos[j] && rng.next_f64() < 0.3 {
                                    mask = set_edge(mask, n, i, j, true);
                                }
                            }
                        }
                        let mut cur =
                            score_dag_mask(mask, n, score_ref, &mut cache, prior, vars_ref)
                                .unwrap_or(f64::NEG_INFINITY);
                        let total_steps = n_warmup + n_draws * thin;
                        let mut kept = 0usize;
                        for step in 0..total_steps {
                            let (new_order, new_mask, rej) =
                                propose_order(&order, mask, n, &mut rng);
                            local_rej += rej;
                            let prop_score = score_dag_mask(
                                new_mask, n, score_ref, &mut cache, prior, vars_ref,
                            );
                            let accept = match prop_score {
                                Some(ps) if cur.is_finite() => {
                                    let log_r = ps - cur;
                                    log_r >= 0.0 || rng.next_f64() < log_r.exp()
                                }
                                Some(ps) => {
                                    order = new_order.clone();
                                    mask = new_mask;
                                    cur = ps;
                                    false
                                }
                                None => false,
                            };
                            if accept {
                                if let Some(ps) = prop_score {
                                    order = new_order;
                                    mask = new_mask;
                                    cur = ps;
                                }
                            }
                            if step >= n_warmup && (step - n_warmup) % thin == 0 && kept < n_draws {
                                let mut p = 0usize;
                                for i in 0..n {
                                    for j in 0..n {
                                        if i == j {
                                            continue;
                                        }
                                        local_traces[(li * n_draws + kept) * n_params + p] =
                                            if has_edge(mask, n, i, j) { 1.0 } else { 0.0 };
                                        p += 1;
                                    }
                                }
                                local_samples[li].push(mask);
                                kept += 1;
                            }
                        }
                    }
                    (start, local_traces, local_samples, local_rej)
                }));
            }
            for h in handles {
                let (start, local_traces, local_samples, local_rej) =
                    h.join().expect("order mcmc worker");
                rejected += local_rej;
                let n_local = local_samples.len();
                for li in 0..n_local {
                    let chain = start + li;
                    sample_masks[chain] = local_samples[li].clone();
                    for d in 0..n_draws {
                        for p in 0..n_params {
                            traces[(chain * n_draws + d) * n_params + p] =
                                local_traces[(li * n_draws + d) * n_params + p];
                        }
                    }
                }
            }
        });

        let (rhat, ess_bulk) = graph_chain_diagnostics(&traces, n_chains, n_draws, n_params);
        let diagnostics = mcmc_graph_diagnostics(
            self.n_chains,
            self.n_warmup,
            self.n_draws,
            ess_bulk,
            rhat,
            0,
            true,
        );
        if !allows_graph_posterior(&diagnostics) {
            return Err(DiscoveryError::unsupported(
                "order MCMC diagnostics gate refused posterior",
            ));
        }

        let mut counts: HashMap<u64, u64> = HashMap::new();
        for chain_samples in &sample_masks {
            for &m in chain_samples {
                *counts.entry(m).or_insert(0) += 1;
            }
        }
        let total: f64 = counts.values().map(|&c| c as f64).sum();
        if !(total > 0.0) {
            return Err(DiscoveryError::unsupported("order MCMC produced no samples"));
        }
        let mut masks = Vec::with_capacity(counts.len());
        let mut weights = Vec::with_capacity(counts.len());
        for (m, c) in counts {
            masks.push(m);
            weights.push(c as f64 / total);
        }
        let ess = kish_ess(&weights);
        let (edge, orient) = accumulate_marginals(n, &weights, &masks);
        GraphPosterior::new(n, weights, masks, edge, orient, ess, diagnostics, rejected)
    }
}

impl GraphPosteriorEngine for OrderMcmc {
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

fn propose_order(
    order: &[usize],
    mask: u64,
    n: usize,
    rng: &mut CausalRng,
) -> (Vec<usize>, u64, u64) {
    let mut new_order = order.to_vec();
    let mut new_mask = mask;
    let mut rejected = 0u64;
    let u = rng.next_f64();
    if u < 0.35 && n >= 2 {
        // Adjacent transposition: drop edges that violate the new order.
        let k = (rng.next_u64() as usize) % (n - 1);
        new_order.swap(k, k + 1);
        let pos = position_map(&new_order);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if has_edge(new_mask, n, i, j) && pos[i] > pos[j] {
                    new_mask = set_edge(new_mask, n, i, j, false);
                    rejected += 1;
                }
            }
        }
    } else if u < 0.55 && n >= 3 {
        // Random transposition of two positions.
        let a = (rng.next_u64() as usize) % n;
        let b = (rng.next_u64() as usize) % n;
        if a != b {
            new_order.swap(a, b);
            let pos = position_map(&new_order);
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    if has_edge(new_mask, n, i, j) && pos[i] > pos[j] {
                        new_mask = set_edge(new_mask, n, i, j, false);
                        rejected += 1;
                    }
                }
            }
        }
    } else {
        // Flip a random forward edge under the current order.
        let a = (rng.next_u64() as usize) % n;
        let b = (rng.next_u64() as usize) % n;
        if a != b {
            let pos = position_map(&new_order);
            let (from, to) = if pos[a] < pos[b] { (a, b) } else { (b, a) };
            let on = has_edge(new_mask, n, from, to);
            new_mask = set_edge(new_mask, n, from, to, !on);
        }
    }
    (new_order, new_mask, rejected)
}

fn position_map(order: &[usize]) -> Vec<usize> {
    let mut pos = vec![0; order.len()];
    for (p, &node) in order.iter().enumerate() {
        pos[node] = p;
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};

    use crate::graph_posterior::GraphPrior;

    fn chain_data(n_rows: usize) -> (TabularData, Vec<VariableId>) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["a", "b", "c"] {
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
        let mut rng = causal_core::CausalRng::from_seed(3);
        let mut a = vec![0.0; n_rows];
        let mut bb = vec![0.0; n_rows];
        let mut c = vec![0.0; n_rows];
        for i in 0..n_rows {
            a[i] = rng.next_f64() * 2.0 - 1.0;
            bb[i] = 1.6 * a[i] + 0.15 * (rng.next_f64() * 2.0 - 1.0);
            c[i] = 1.3 * bb[i] + 0.15 * (rng.next_f64() * 2.0 - 1.0);
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(vars[0], Arc::from(a), ValidityBitmap::all_valid(n_rows))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(vars[1], Arc::from(bb), ValidityBitmap::all_valid(n_rows))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(vars[2], Arc::from(c), ValidityBitmap::all_valid(n_rows))
                    .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        (TabularData::new(storage), vars)
    }

    #[test]
    fn order_mcmc_chain_signal() {
        let (data, vars) = chain_data(220);
        let eng = OrderMcmc::new()
            .with_schedule(2, 300, 600, 1)
            .with_diagnostics_gate(false);
        let ctx = ExecutionContext::for_tests(9);
        let mut ws = DiscoveryWorkspace::default();
        let post = eng
            .run(
                &data,
                &vars,
                &GraphPrior::uniform(),
                GraphScoreFamily::GaussianBic,
                &mut ws,
                &ctx,
            )
            .unwrap();
        let n = 3;
        let sk01 = post.edge_marginals[0 * n + 1] + post.edge_marginals[1 * n + 0];
        assert!(sk01 > 0.25, "P(A—B)={sk01}");
        assert!(crate::graph_posterior::allows_graph_posterior(&post.diagnostics));
    }
}
