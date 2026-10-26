//! Exact DAG posterior enumeration for very small graphs.
//!
//! Enumeration lists every labeled DAG, so it reaches **`n ≤ `[`EXACT_ENUM_MAX_NODES`]**
//! nodes (Gaussian BIC). Larger problems are routed to order MCMC, and the exact edge
//! marginals of up to [`crate::EXACT_MARGINAL_MAX_NODES`] variables come from the subset
//! recursion of [`crate::exact_dag_marginals`], which does not list graphs.
//!
//! "Exact" means every DAG is enumerated, not that the weights are exact posterior
//! probabilities: each graph's weight is `exp(BIC + log prior)`, the BIC approximation to the
//! marginal likelihood. Graphs with no finite score — constraint-violating ones, and ones
//! containing a deterministic relation (zero residual variance), which BIC cannot score — are
//! excluded from the support; their count is `rejected_invalid` and is stated in the
//! diagnostics notes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_prob::kish_ess;
use antecedent_state::GraphScoreFamily;

use crate::adjacency_mask::AdjacencyMask;
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::graph_posterior::{
    EXACT_ENUM_MAX_NODES, GraphPosterior, GraphPosteriorEngine, GraphPrior, SamplingSummary,
    accumulate_marginals, analytic_graph_diagnostics, edge_bit, normalize_log_weights,
};
use crate::graph_score::{DagScorer, tabular_score_data};

/// Exact enumeration engine for labeled DAGs on `n ≤ `[`EXACT_ENUM_MAX_NODES`] nodes.
///
/// Enumeration cannot list the graphs of a larger problem (seven nodes have 1.1 billion), so
/// above the cap the run is routed to [`crate::OrderMcmc`] when [`Self::route_larger`] is set
/// (the default), and the posterior says so in its diagnostics notes and algorithm tag.
/// Without routing the run is refused.
#[derive(Clone, Debug)]
pub struct ExactDagPosterior {
    /// Node cap for enumeration (defaults to [`EXACT_ENUM_MAX_NODES`]).
    pub max_nodes: usize,
    /// Sample larger problems with order MCMC instead of refusing them.
    pub route_larger: bool,
}

impl Default for ExactDagPosterior {
    fn default() -> Self {
        Self::new()
    }
}

impl ExactDagPosterior {
    /// Default exact engine.
    #[must_use]
    pub fn new() -> Self {
        Self { max_nodes: EXACT_ENUM_MAX_NODES, route_larger: true }
    }

    /// Refuse, rather than sample, a problem above the enumeration cap.
    #[must_use]
    pub fn with_route_larger(mut self, route_larger: bool) -> Self {
        self.route_larger = route_larger;
        self
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
    #[allow(
        clippy::too_many_lines,
        reason = "one algorithm pipeline over shared mutable graph state; the phases are marked in place"
    )]
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        prior: &GraphPrior,
        score_family: GraphScoreFamily,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosterior, DiscoveryError> {
        prior.constraints.validate()?;
        let n = variables.len();
        if n == 0 {
            return Err(DiscoveryError::unsupported(
                "exact DAG posterior requires at least one variable",
            ));
        }
        let cap = self.max_nodes.min(EXACT_ENUM_MAX_NODES);
        if n > cap {
            if !self.route_larger {
                return Err(DiscoveryError::unsupported(
                    "exact DAG posterior requires n ≤ 6 variables (EXACT_ENUM_MAX_NODES); \
                     use order_mcmc, structure_mcmc, or ci_screened_posterior",
                ));
            }
            return Self::sample_instead(
                data, variables, prior, score_family, workspace, ctx, cap,
            );
        }

        let score_data = tabular_score_data(data, variables)?;
        if !matches!(score_family, GraphScoreFamily::GaussianBic) {
            return Err(DiscoveryError::unsupported(
                "exact DAG posterior currently supports GaussianBic only",
            ));
        }

        // Check before allocating/enumerating millions of DAGs. These are the
        // exact labeled DAG counts for the supported sizes 0..=6.
        let dag_counts: [u64; 7] = [1, 1, 3, 25, 543, 29_281, 3_781_503];
        let est_bytes = dag_counts[n].saturating_mul(32);
        if let Some(hard) = ctx.memory_limit_bytes() {
            if est_bytes > hard {
                return Err(DiscoveryError::Resource(format!(
                    "exact DAG enumeration needs ~{est_bytes} bytes; hard limit {hard}"
                )));
            }
        }

        let masks = enumerate_unique_dags(n);
        let scorer = DagScorer::new(n, &score_data, score_family, prior, variables);
        let scored = ctx.map_indexed(masks.len(), |index, inner| {
            if inner.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            Ok(match scorer.score(&masks[index]) {
                Some(lw) => (lw, 0u64),
                None => (f64::NEG_INFINITY, 1u64),
            })
        })?;
        let mut log_w = Vec::with_capacity(scored.len());
        let mut rejected = 0u64;
        for (lw, rej) in scored {
            log_w.push(lw);
            rejected += rej;
        }
        if ctx.cancellation.is_cancelled() {
            return Err(DiscoveryError::Cancelled);
        }
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
        let mut diagnostics = analytic_graph_diagnostics(kept_masks.len(), ess);
        if rejected > 0 {
            diagnostics.notes.push(std::sync::Arc::from(format!(
                "{rejected} graph(s) excluded: constraint-violating or no finite BIC score \
                 (e.g. a deterministic relation)"
            )));
        }
        GraphPosterior::from_atoms(
            n,
            weights,
            kept_masks,
            edge,
            orient,
            SamplingSummary { ess, diagnostics, rejected_invalid: rejected },
        )
        .map(|posterior| posterior.with_algorithm("exact_dag_posterior"))
    }
}

impl ExactDagPosterior {
    /// A problem above the enumeration cap, sampled by order MCMC and labelled as such.
    fn sample_instead(
        data: &TabularData,
        variables: &[VariableId],
        prior: &GraphPrior,
        score_family: GraphScoreFamily,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
        cap: usize,
    ) -> Result<GraphPosterior, DiscoveryError> {
        let mut posterior = crate::OrderMcmc::new()
            .run(data, variables, prior, score_family, workspace, ctx)?
            .with_algorithm("order_mcmc");
        posterior.push_note(format!(
            "{} variables exceed the enumeration limit of {cap}: the posterior was sampled by \
             order MCMC, not enumerated; its marginals carry Monte-Carlo error",
            variables.len()
        ));
        Ok(posterior)
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

/// Every labeled DAG on `n` nodes exactly once, as packed adjacency masks in ascending order.
///
/// At most [`MAX_ENUM_NODES`] nodes, so every mask lives in word 0 of its [`AdjacencyMask`].
///
/// Built by adding node `k` to each DAG on nodes `0..k`: choose its children `T`, then its
/// parents from the nodes that are not `T` or a descendant of `T` (those would close a cycle).
/// Removing the highest node maps each DAG back to a unique smaller one, so no deduplication
/// is needed.
pub(crate) fn enumerate_unique_dags(n: usize) -> Vec<AdjacencyMask> {
    assert!(n <= MAX_ENUM_NODES, "DAG enumeration packs masks into one word");
    let mut bit = [[0u64; MAX_ENUM_NODES]; MAX_ENUM_NODES];
    for (from, row) in bit.iter_mut().enumerate().take(n) {
        for (to, slot) in row.iter_mut().enumerate().take(n) {
            if from != to {
                *slot = 1u64 << edge_bit(n, from, to);
            }
        }
    }
    let mut words = Vec::new();
    grow_dags(n, 0, [0u8; MAX_ENUM_NODES], 0, &bit, &mut words);
    words.sort_unstable();
    words.into_iter().map(AdjacencyMask::from_word).collect()
}

const MAX_ENUM_NODES: usize = 8;

/// `descendants[v]` holds the strict descendants of `v` among the first `k` nodes.
fn grow_dags(
    n: usize,
    k: usize,
    descendants: [u8; MAX_ENUM_NODES],
    mask: u64,
    bit: &[[u64; MAX_ENUM_NODES]; MAX_ENUM_NODES],
    out: &mut Vec<u64>,
) {
    if k == n {
        out.push(mask);
        return;
    }
    // `k < n <= 8`, so a node set over the first `k` nodes fits a byte.
    let all = (1u8 << k) - 1;
    for children in 0..=all {
        let mut reach = children;
        let mut with_children = mask;
        for (v, &below) in descendants.iter().enumerate().take(k) {
            if children & (1 << v) != 0 {
                reach |= below;
                with_children |= bit[k][v];
            }
        }
        let free = !reach & all;
        let mut parents = free;
        loop {
            let mut next = descendants;
            let mut next_mask = with_children;
            for u in 0..k {
                if parents & (1 << u) != 0 {
                    next_mask |= bit[u][k];
                }
                if parents & (1 << u) != 0 || descendants[u] & parents != 0 {
                    next[u] |= reach | (1 << k);
                }
            }
            next[k] = reach;
            grow_dags(n, k + 1, next, next_mask, bit, out);
            if parents == 0 {
                break;
            }
            parents = (parents - 1) & free;
        }
    }
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

    use crate::adjacency_mask::{has_edge, set_edge};
    use crate::graph_posterior::{GraphPrior, edge_bit};

    fn chain_tabular(n_rows: usize, seed: u64) -> (TabularData, Vec<VariableId>) {
        let mut rng = antecedent_core::CausalRng::from_seed(seed);
        let mut a = vec![0.0; n_rows];
        let mut bb = vec![0.0; n_rows];
        let mut c = vec![0.0; n_rows];
        for i in 0..n_rows {
            a[i] = rng.next_f64() * 2.0 - 1.0;
            bb[i] = 1.5 * a[i] + 0.15 * (rng.next_f64() * 2.0 - 1.0);
            c[i] = 1.2 * bb[i] + 0.15 * (rng.next_f64() * 2.0 - 1.0);
        }
        let data = crate::test_tables::continuous_table(&["a", "b", "c"], vec![a, bb, c]);
        let vars: Vec<_> = (0..3).map(VariableId::from_raw).collect();
        (data, vars)
    }

    #[test]
    fn enumerates_known_dag_counts() {
        assert_eq!(enumerate_unique_dags(1).len(), 1);
        assert_eq!(enumerate_unique_dags(2).len(), 3);
        assert_eq!(enumerate_unique_dags(3).len(), 25);
        assert_eq!(enumerate_unique_dags(4).len(), 543);
    }

    #[test]
    fn enumeration_is_the_sorted_set_of_acyclic_masks() {
        assert_eq!(enumerate_unique_dags(0), vec![AdjacencyMask::EMPTY]);
        for n in 1..=4usize {
            let bits = n * (n - 1);
            let expected: Vec<AdjacencyMask> = (0..(1u64 << bits))
                .filter(|&m| crate::adjacency_mask::mask_is_dag(m, n))
                .map(AdjacencyMask::from_word)
                .collect();
            assert_eq!(enumerate_unique_dags(n), expected, "n={n}");
        }
        assert_eq!(enumerate_unique_dags(5).len(), 29_281);
    }

    #[test]
    fn dag_scorer_matches_per_mask_scoring() {
        let (data, vars) = chain_tabular(60, 3);
        let n = 3;
        let score_data = tabular_score_data(&data, &vars).unwrap();
        let prior = GraphPrior::bernoulli_edges(0.3).unwrap().with_constraints(
            crate::DiscoveryConstraints { max_parents: Some(1), ..Default::default() },
        );
        let scorer = DagScorer::new(n, &score_data, GraphScoreFamily::GaussianBic, &prior, &vars);
        let mut cache =
            antecedent_state::LocalScoreCache::new(crate::graph_score::graph_score_cache_key(
                GraphScoreFamily::GaussianBic,
                n,
                score_data.n_rows,
            ));
        for word in 0..(1u64 << (n * (n - 1))) {
            let mask = AdjacencyMask::from_word(word);
            let direct =
                crate::graph_score::score_dag_mask(&mask, n, &score_data, &mut cache, &prior, &vars);
            assert_eq!(scorer.score(&mask).map(f64::to_bits), direct.map(f64::to_bits));
        }
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
        assert!(post.diagnostics().converged);
        assert_eq!(post.algorithm().map(|a| &**a), Some("exact_dag_posterior"));
        let n = 3;
        // Chain A→B→C is Markov-equivalent to A←B←C and A←B→C; assert skeleton
        // mass, not a single orientation.
        let sk01 = post.edge_marginals()[1] + post.edge_marginals()[n];
        let sk12 = post.edge_marginals()[n + 2] + post.edge_marginals()[2 * n + 1];
        let sk02 = post.edge_marginals()[2] + post.edge_marginals()[2 * n];
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
        let mut cyc = AdjacencyMask::EMPTY;
        cyc = set_edge(cyc, n, 0, 1, true);
        cyc = set_edge(cyc, n, 1, 2, true);
        cyc = set_edge(cyc, n, 2, 0, true);
        assert!(!enumerate_unique_dags(n).contains(&cyc));
        assert!(!has_edge(AdjacencyMask::EMPTY, n, 0, 1));
        let _ = edge_bit(n, 0, 1);
    }
    #[test]
    fn exact_enumeration_honors_memory_budget_and_hard_node_cap() {
        let (data, variables) = chain_tabular(20, 5);
        let mut context = ExecutionContext::for_tests(1);
        context.set_memory_limit(Some(1));
        let result = ExactDagPosterior::new().run(
            &data,
            &variables,
            &GraphPrior::uniform(),
            GraphScoreFamily::GaussianBic,
            &mut DiscoveryWorkspace::default(),
            &context,
        );
        assert!(matches!(result, Err(DiscoveryError::Resource(_))));
        // Public field mutation must not bypass the documented hard cap.
        let engine = ExactDagPosterior { max_nodes: 100, route_larger: false };
        let variables: Vec<_> = (0..7).map(VariableId::from_raw).collect();
        let error = engine
            .run(
                &data,
                &variables,
                &GraphPrior::uniform(),
                GraphScoreFamily::GaussianBic,
                &mut DiscoveryWorkspace::default(),
                &context,
            )
            .unwrap_err();
        assert!(error.to_string().contains("n ≤ 6"));
    }

    /// `cargo test --profile bench -p antecedent-discovery --lib exact_enumeration::tests::timing -- --ignored --nocapture`
    #[test]
    #[ignore = "timing harness"]
    fn timing_enumeration_and_scoring() {
        use std::time::Instant;
        let digest = |masks: &[AdjacencyMask]| {
            masks.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, m| {
                (h ^ m.key()).wrapping_mul(0x0100_0000_01b3)
            })
        };
        for n in [5usize, 6] {
            let t = Instant::now();
            let masks = enumerate_unique_dags(n);
            println!(
                "n={n} enumerate: {} dags, digest {:016x}, {:?}",
                masks.len(),
                digest(&masks),
                t.elapsed()
            );
            let vars: Vec<_> = (0..u32::try_from(n).unwrap()).map(VariableId::from_raw).collect();
            let prior = GraphPrior::bernoulli_edges(0.3).unwrap();
            let t = Instant::now();
            let mut acc = 0.0;
            let mut dags = 0usize;
            for m in &masks {
                if let Some(lp) = crate::graph_posterior::log_prior_mask(m, n, &prior, &vars) {
                    acc += lp;
                    dags += 1;
                }
            }
            println!(
                "n={n} log_prior_mask over all: {dags} valid, sum {acc:.6}, {:?}",
                t.elapsed()
            );
        }
    }

    #[test]
    #[ignore = "timing harness"]
    fn timing_exact_run() {
        use std::time::Instant;
        for n in [5usize, 6] {
            let mut b = CausalSchemaBuilder::new();
            for i in 0..n {
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
            let vars: Vec<_> = (0..u32::try_from(n).unwrap()).map(VariableId::from_raw).collect();
            let mut rng = antecedent_core::CausalRng::from_seed(11);
            let rows = 300;
            let mut cols_v: Vec<Vec<f64>> = vec![vec![0.0; rows]; n];
            for r in 0..rows {
                for c in 0..n {
                    let prev = if c > 0 { 0.7 * cols_v[c - 1][r] } else { 0.0 };
                    cols_v[c][r] = prev + rng.next_f64() - 0.5;
                }
            }
            let cols: Vec<_> = cols_v
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    OwnedColumn::Float64(
                        Float64Column::new(vars[i], Arc::from(v), ValidityBitmap::all_valid(rows))
                            .unwrap(),
                    )
                })
                .collect();
            let data =
                TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
            let ctx = ExecutionContext::production(1, 4);
            let t = Instant::now();
            let post = ExactDagPosterior::new()
                .run(
                    &data,
                    &vars,
                    &GraphPrior::bernoulli_edges(0.3).unwrap(),
                    GraphScoreFamily::GaussianBic,
                    &mut DiscoveryWorkspace::default(),
                    &ctx,
                )
                .unwrap();
            let dig = post.weights().iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, w| {
                (h ^ w.to_bits()).wrapping_mul(0x0100_0000_01b3)
            });
            let edig = post.edge_marginals().iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, w| {
                (h ^ w.to_bits()).wrapping_mul(0x0100_0000_01b3)
            });
            println!(
                "n={n} run: {} graphs, weights {dig:016x}, edges {edig:016x}, {:?}",
                post.weights().len(),
                t.elapsed()
            );
        }
    }
}
