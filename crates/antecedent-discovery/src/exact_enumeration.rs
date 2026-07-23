//! Exact DAG posterior enumeration for very small graphs.
//!
//! Hard product limit: labeled DAGs on **`n ≤ `[`EXACT_ENUM_MAX_NODES`]** nodes
//! (Gaussian BIC). Larger graphs must use order / structure / CI-screened MCMC
//! (`OrderMcmc`, `StructureMcmc`, `CiScreenedPosterior`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::too_many_lines)]

use std::collections::HashSet;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_state::{GraphScoreCacheKey, GraphScoreFamily, LocalScoreCache};

use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::graph_posterior::{
    EXACT_ENUM_MAX_NODES, GraphPosterior, GraphPosteriorEngine, GraphPrior, accumulate_marginals,
    analytic_graph_diagnostics, kish_ess, mask_is_dag, n_directed_edges, normalize_log_weights,
    set_edge,
};
use crate::graph_score::{score_dag_mask, tabular_score_data};

/// Exact enumeration engine for labeled DAGs on `n ≤ `[`EXACT_ENUM_MAX_NODES`] nodes.
///
/// This is a hard combinatorial cap (not a soft heuristic). Prefer MCMC engines
/// when `variables.len() > EXACT_ENUM_MAX_NODES`.
#[derive(Clone, Debug, Default)]
pub struct ExactDagPosterior {
    /// Optional hard node cap (defaults to [`EXACT_ENUM_MAX_NODES`]).
    pub max_nodes: usize,
}

impl ExactDagPosterior {
    /// Default exact engine.
    #[must_use]
    pub fn new() -> Self {
        Self { max_nodes: EXACT_ENUM_MAX_NODES }
    }

    /// Override max nodes (still capped at [`EXACT_ENUM_MAX_NODES`]).
    #[must_use]
    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes.min(EXACT_ENUM_MAX_NODES);
        self
    }

    /// Run exact posterior inference.
    ///
    /// # Errors
    ///
    /// Unsupported size, data/score failures, empty valid support, or memory budget.
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
                "exact DAG posterior requires at least one variable",
            ));
        }
        if n > self.max_nodes {
            return Err(DiscoveryError::unsupported(
                "exact DAG posterior requires n ≤ 6 variables (EXACT_ENUM_MAX_NODES); \
                 use order_mcmc, structure_mcmc, or ci_screened_posterior",
            ));
        }
        if n_directed_edges(n) > 63 {
            return Err(DiscoveryError::unsupported(
                "exact DAG posterior adjacency exceeds 63 directed edges",
            ));
        }

        let score_data = tabular_score_data(data, variables)?;
        if !matches!(score_family, GraphScoreFamily::GaussianBic) {
            return Err(DiscoveryError::unsupported(
                "exact DAG posterior currently supports GaussianBic only",
            ));
        }

        let masks = enumerate_unique_dags(n);
        let est_bytes = (masks.len() as u64).saturating_mul(32);
        if let Some(hard) = ctx.memory.hard_limit_bytes {
            if est_bytes > hard {
                return Err(DiscoveryError::Resource(format!(
                    "exact DAG enumeration needs ~{est_bytes} bytes; hard limit {hard}"
                )));
            }
        }

        let threads = ctx.parallelism.max_threads.get().max(1) as usize;
        let chunk = (masks.len() / threads).max(1);
        let mut log_w = vec![f64::NEG_INFINITY; masks.len()];
        let mut rejected = 0u64;

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (chunk_id, start) in (0..masks.len()).step_by(chunk).enumerate() {
                let end = (start + chunk).min(masks.len());
                let masks_ref = &masks;
                let score_ref = &score_data;
                let vars_ref = variables;
                handles.push(scope.spawn(move || {
                    let mut cache = LocalScoreCache::new(GraphScoreCacheKey {
                        data_version: 1,
                        family: score_family,
                        var_fingerprint: n as u64,
                        penalty_fingerprint: score_data.n_rows as u64,
                    });
                    let mut local_log = vec![f64::NEG_INFINITY; end - start];
                    let mut local_rej = 0u64;
                    for (k, &mask) in masks_ref[start..end].iter().enumerate() {
                        match score_dag_mask(mask, n, score_ref, &mut cache, prior, vars_ref) {
                            Some(lw) => local_log[k] = lw,
                            None => local_rej += 1,
                        }
                    }
                    (chunk_id, start, local_log, local_rej)
                }));
            }
            for h in handles {
                let (_id, start, local_log, local_rej) = h.join().expect("exact enum worker");
                rejected += local_rej;
                for (i, lw) in local_log.into_iter().enumerate() {
                    log_w[start + i] = lw;
                }
            }
        });

        // Keep only finite-weight graphs.
        let mut kept_masks = Vec::new();
        let mut kept_log = Vec::new();
        for (mask, lw) in masks.into_iter().zip(log_w.into_iter()) {
            if lw.is_finite() {
                kept_masks.push(mask);
                kept_log.push(lw);
            }
        }
        if kept_masks.is_empty() {
            return Err(DiscoveryError::unsupported("no constraint-valid DAGs under prior"));
        }

        let weights = normalize_log_weights(&kept_log)?;
        let ess = kish_ess(&weights);
        let (edge, orient) = accumulate_marginals(n, &weights, &kept_masks);
        let diagnostics = analytic_graph_diagnostics(kept_masks.len(), ess);
        GraphPosterior::new(n, weights, kept_masks, edge, orient, ess, diagnostics, rejected)
    }
}

impl GraphPosteriorEngine for ExactDagPosterior {
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

/// Unique labeled DAGs on `n` nodes via topo-order × forward-edge subsets + dedup.
pub(crate) fn enumerate_unique_dags(n: usize) -> Vec<u64> {
    let mut seen = HashSet::new();
    let mut order: Vec<usize> = (0..n).collect();
    permute_and_collect(n, &mut order, 0, &mut seen);
    let mut out: Vec<u64> = seen.into_iter().collect();
    out.sort_unstable();
    out
}

fn permute_and_collect(n: usize, order: &mut [usize], k: usize, seen: &mut HashSet<u64>) {
    if k == n {
        let n_fwd = n * (n.saturating_sub(1)) / 2;
        let limit = 1u64 << n_fwd;
        for edge_mask in 0..limit {
            let mut adj = 0u64;
            let mut bit = 0u32;
            for a in 0..n {
                for b in (a + 1)..n {
                    if (edge_mask >> bit) & 1 == 1 {
                        let from = order[a];
                        let to = order[b];
                        adj = set_edge(adj, n, from, to, true);
                    }
                    bit += 1;
                }
            }
            debug_assert!(mask_is_dag(adj, n));
            seen.insert(adj);
        }
        return;
    }
    for i in k..n {
        order.swap(k, i);
        permute_and_collect(n, order, k + 1, seen);
        order.swap(k, i);
    }
}

/// Empty graph always included; also used by tests.
#[allow(dead_code)]
pub(crate) fn empty_mask() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_state::GraphScoreFamily;

    use crate::graph_posterior::{GraphPrior, edge_bit, has_edge, set_edge};

    fn chain_tabular(n_rows: usize, seed: u64) -> (TabularData, Vec<VariableId>) {
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
        let mut rng = antecedent_core::CausalRng::from_seed(seed);
        let mut a = vec![0.0; n_rows];
        let mut bb = vec![0.0; n_rows];
        let mut c = vec![0.0; n_rows];
        for i in 0..n_rows {
            a[i] = rng.next_f64() * 2.0 - 1.0;
            bb[i] = 1.5 * a[i] + 0.15 * (rng.next_f64() * 2.0 - 1.0);
            c[i] = 1.2 * bb[i] + 0.15 * (rng.next_f64() * 2.0 - 1.0);
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
    fn enumerates_known_dag_counts() {
        assert_eq!(enumerate_unique_dags(1).len(), 1);
        assert_eq!(enumerate_unique_dags(2).len(), 3);
        assert_eq!(enumerate_unique_dags(3).len(), 25);
        assert_eq!(enumerate_unique_dags(4).len(), 543);
    }

    #[test]
    fn chain_recovers_adjacent_edges() {
        let (data, vars) = chain_tabular(200, 7);
        let eng = ExactDagPosterior::new();
        let prior = GraphPrior::uniform();
        let ctx = ExecutionContext::for_tests(1);
        let mut ws = DiscoveryWorkspace::default();
        let post =
            eng.run(&data, &vars, &prior, GraphScoreFamily::GaussianBic, &mut ws, &ctx).unwrap();
        assert!(post.diagnostics.converged);
        let n = 3;
        // Chain A→B→C is Markov-equivalent to A←B←C and A←B→C; assert skeleton
        // mass, not a single orientation.
        let sk01 = post.edge_marginals[1] + post.edge_marginals[n];
        let sk12 = post.edge_marginals[n + 2] + post.edge_marginals[2 * n + 1];
        let sk02 = post.edge_marginals[2] + post.edge_marginals[2 * n];
        assert!(sk01 > 0.5, "P(A—B)={sk01}");
        assert!(sk12 > 0.5, "P(B—C)={sk12}");
        assert!(sk02 < 0.3, "P(A—C)={sk02}");
        let samples = post.to_weighted_samples().unwrap();
        assert!((samples.total_weight() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_oversized() {
        let mut b = CausalSchemaBuilder::new();
        for i in 0..7 {
            b.add_variable(
                format!("v{i}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let vars: Vec<_> = (0..7).map(VariableId::from_raw).collect();
        let cols: Vec<_> = vars
            .iter()
            .map(|&id| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        id,
                        Arc::from([0.0_f64, 1.0, 2.0]),
                        ValidityBitmap::all_valid(3),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let err = ExactDagPosterior::new()
            .infer_graphs(
                &data,
                &vars,
                &GraphPrior::uniform(),
                GraphScoreFamily::GaussianBic,
                &ExecutionContext::for_tests(0),
            )
            .unwrap_err();
        assert!(matches!(err, DiscoveryError::Unsupported { .. }));
    }

    #[test]
    fn cycle_mask_excluded_from_enumeration() {
        let n = 3;
        let mut cyc = 0u64;
        cyc = set_edge(cyc, n, 0, 1, true);
        cyc = set_edge(cyc, n, 1, 2, true);
        cyc = set_edge(cyc, n, 2, 0, true);
        assert!(!enumerate_unique_dags(n).contains(&cyc));
        assert!(!has_edge(0, n, 0, 1));
        let _ = edge_bit(n, 0, 1);
    }
}
