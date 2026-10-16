//! Calibration design for the edge marginals of the order- and structure-MCMC posteriors at ten
//! and twelve variables.
//!
//! The claim under test is that, under the published conditions (at most three parents, four
//! chains, the schedules below) and a published posterior, the skeleton marginal
//! `P(i — j) = P(i → j) + P(j → i)` of a pair is calibrated: among pairs reported at marginal
//! `p`, a fraction `p` are adjacent in the generating graph. The skeleton, not the direction,
//! is the claim: a direction inside a Markov equivalence class is not identified, so its
//! marginal has no truth to be calibrated against.
//!
//! Each replicate draws a random sparse linear-Gaussian SEM (random order, zero to two parents
//! per node, coefficients of magnitude 0.5–1), simulates 400 rows, runs the sampler with its
//! publication gate on, and bins the `n (n - 1) / 2` skeleton marginals into ten equal-width
//! bins. A replicate the gate refuses is counted, not scored: the claim is about published
//! posteriors, and the test fails when fewer than 80% are published. Per bin the pooled
//! residual `observed - predicted` must be within 3.5 cluster-robust standard errors of zero
//! (clusters are replicates, since the pairs of one dataset are dependent).
//!
//! Each test runs via `scripts/gate_calibration.sh` and is `#[ignore]`d otherwise.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalRng, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_state::GraphScoreFamily;

use crate::adjacency_mask::AdjacencyMask;
use crate::constraints::DiscoveryConstraints;
use crate::engine::DiscoveryWorkspace;
use crate::graph_posterior::{GraphPosterior, GraphPrior, has_edge, set_edge};
use crate::order_mcmc::OrderMcmc;
use crate::structure_mcmc::StructureMcmc;

/// Rows per simulated dataset.
const N_ROWS: usize = 400;
/// Replicates (datasets) per test.
const N_SIM: u64 = 100;
/// Equal-width bins of the predicted skeleton marginal.
const N_BINS: usize = 10;
/// A bin enters the check with at least this many pairs, from at least [`MIN_BIN_CLUSTERS`]
/// replicates.
const MIN_BIN_PAIRS: f64 = 100.0;
const MIN_BIN_CLUSTERS: usize = 20;
/// Cluster-robust standard errors of slack on a bin's residual.
const Z_BAND: f64 = 3.5;
/// Fraction of replicates the publication gate must pass.
const MIN_PUBLISHED_FRACTION: f64 = 0.8;
/// Published cap on the parents of any node.
const MAX_PARENTS: usize = 3;

fn normal(rng: &mut CausalRng) -> f64 {
    let (u, v) = (rng.next_f64().max(f64::MIN_POSITIVE), rng.next_f64());
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// A random sparse linear-Gaussian SEM on `n` nodes and its rows: columns and the true graph.
fn sparse_sem(n: usize, seed: u64) -> (Vec<Vec<f64>>, AdjacencyMask) {
    let mut rng = CausalRng::from_seed(seed);
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, crate::indexing::bounded_index(rng.next_u64(), i + 1));
    }
    let mut truth = AdjacencyMask::EMPTY;
    let mut parents: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for pos in 1..n {
        let u = rng.next_f64();
        let want = if u < 0.3 {
            0
        } else if u < 0.7 {
            1
        } else {
            2
        };
        let mut pool: Vec<usize> = order[..pos].to_vec();
        for _ in 0..want.min(pos) {
            let pick = pool.swap_remove(crate::indexing::bounded_index(rng.next_u64(), pool.len()));
            let sign = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
            parents[order[pos]].push((pick, sign * (0.5 + 0.5 * rng.next_f64())));
            truth = set_edge(truth, n, pick, order[pos], true);
        }
    }
    let mut cols = vec![vec![0.0; N_ROWS]; n];
    for row in 0..N_ROWS {
        for &node in &order {
            let mean: f64 = parents[node].iter().map(|&(p, b)| b * cols[p][row]).sum();
            cols[node][row] = mean + normal(&mut rng);
        }
    }
    (cols, truth)
}

/// Per bin: pairs, sum of predicted, sum of observed.
type BinCells = [(f64, f64, f64); N_BINS];

/// One published replicate's skeleton marginals against the generating graph.
fn bin_skeleton(n: usize, post: &GraphPosterior, truth: AdjacencyMask) -> BinCells {
    let mut cells = [(0.0, 0.0, 0.0); N_BINS];
    for i in 0..n {
        for j in (i + 1)..n {
            let p = (post.edge_marginals[i * n + j] + post.edge_marginals[j * n + i]).min(1.0);
            let scaled = p * N_BINS as f64;
            let bin = (1..N_BINS)
                .filter(|&b| scaled >= f64::from(u32::try_from(b).unwrap_or(u32::MAX)))
                .count();
            let adjacent = has_edge(truth, n, i, j) || has_edge(truth, n, j, i);
            cells[bin].0 += 1.0;
            cells[bin].1 += p;
            cells[bin].2 += f64::from(u8::from(adjacent));
        }
    }
    cells
}

/// Run `N_SIM` replicates of `sampler` on `n` variables and check the binned residuals.
fn assert_skeleton_marginals_calibrated(
    label: &str,
    n: usize,
    sampler: impl Fn(
        &TabularData,
        &[VariableId],
        &GraphPrior,
        &ExecutionContext,
    ) -> Option<GraphPosterior>,
) {
    let vars: Vec<VariableId> =
        (0..n).map(|i| VariableId::from_raw(u32::try_from(i).unwrap())).collect();
    let prior = GraphPrior::uniform().with_constraints(DiscoveryConstraints {
        max_parents: Some(MAX_PARENTS),
        ..Default::default()
    });
    let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    let mut published: Vec<BinCells> = Vec::new();
    for rep in 0..N_SIM {
        let (cols, truth) = sparse_sem(n, 0x00ED_6E00 + rep);
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let data = crate::test_tables::continuous_table(&name_refs, cols);
        if let Some(post) = sampler(&data, &vars, &prior, &ExecutionContext::for_tests(9_000 + rep))
        {
            published.push(bin_skeleton(n, &post, truth));
        }
    }
    let rate = published.len() as f64 / N_SIM as f64;
    println!("edge-marginal-calibration {label} n={n} published={}/{N_SIM}", published.len());
    assert!(
        rate >= MIN_PUBLISHED_FRACTION,
        "{label} n={n}: the publication gate passed {rate:.2} of replicates, below \
         {MIN_PUBLISHED_FRACTION}; the published conditions do not support the claim"
    );
    let mut checked = 0;
    for bin in 0..N_BINS {
        let clusters: Vec<(f64, f64)> = published
            .iter()
            .map(|cells| (cells[bin].0, cells[bin].2 - cells[bin].1))
            .filter(|&(m, _)| m > 0.0)
            .collect();
        let pairs: f64 = clusters.iter().map(|c| c.0).sum();
        if pairs < MIN_BIN_PAIRS || clusters.len() < MIN_BIN_CLUSTERS {
            continue;
        }
        let residual = clusters.iter().map(|c| c.1).sum::<f64>() / pairs;
        let k = clusters.len() as f64;
        let spread: f64 = clusters.iter().map(|&(m, s)| (s - residual * m).powi(2)).sum();
        let se = (spread * k / (k - 1.0)).sqrt() / pairs;
        let predicted = published.iter().map(|c| c[bin].1).sum::<f64>() / pairs;
        println!(
            "edge-marginal-calibration {label} n={n} bin={bin} pairs={pairs} \
             predicted={predicted:.4} observed={:.4} se={se:.4}",
            predicted + residual
        );
        assert!(
            residual.abs() <= Z_BAND * se,
            "{label} n={n} bin {bin}: observed adjacency {:.4} vs predicted {predicted:.4} \
             (residual {residual:.4}, {Z_BAND} x se = {:.4})",
            predicted + residual,
            Z_BAND * se
        );
        checked += 1;
    }
    assert!(checked >= 3, "{label} n={n}: only {checked} bins had enough pairs to check");
}

fn structure_sampler(
    data: &TabularData,
    vars: &[VariableId],
    prior: &GraphPrior,
    ctx: &ExecutionContext,
) -> Option<GraphPosterior> {
    StructureMcmc::new()
        .with_schedule(4, 3000, 12_000, 2)
        .run(
            data,
            vars,
            prior,
            GraphScoreFamily::GaussianBic,
            &mut DiscoveryWorkspace::default(),
            ctx,
        )
        .ok()
}

fn order_sampler(
    data: &TabularData,
    vars: &[VariableId],
    prior: &GraphPrior,
    ctx: &ExecutionContext,
) -> Option<GraphPosterior> {
    OrderMcmc::new()
        .with_schedule(4, 3000, 12_000, 2)
        .run(
            data,
            vars,
            prior,
            GraphScoreFamily::GaussianBic,
            &mut DiscoveryWorkspace::default(),
            ctx,
        )
        .ok()
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn structure_mcmc_skeleton_marginals_calibrated_at_ten_variables() {
    assert_skeleton_marginals_calibrated("structure_mcmc", 10, structure_sampler);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn structure_mcmc_skeleton_marginals_calibrated_at_twelve_variables() {
    assert_skeleton_marginals_calibrated("structure_mcmc", 12, structure_sampler);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn order_mcmc_skeleton_marginals_calibrated_at_ten_variables() {
    assert_skeleton_marginals_calibrated("order_mcmc", 10, order_sampler);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn order_mcmc_skeleton_marginals_calibrated_at_twelve_variables() {
    assert_skeleton_marginals_calibrated("order_mcmc", 12, order_sampler);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The design's generator is the graph it reports: every node has at most two parents,
    /// the graph is acyclic, and the rows are not degenerate.
    #[test]
    fn sparse_sem_returns_its_generating_graph() {
        for n in [10usize, 12] {
            let (cols, truth) = sparse_sem(n, 17);
            assert!(crate::graph_posterior::mask_is_dag(truth, n));
            assert!((0..n).all(|v| crate::graph_posterior::parents_of(truth, n, v).len() <= 2));
            assert!(cols.iter().all(|c| c.len() == N_ROWS && c.iter().all(|x| x.is_finite())));
        }
    }

    /// A pair lands in the bin of its skeleton marginal, the sum of both directions.
    #[test]
    fn binning_places_a_pair_by_its_skeleton_marginal() {
        let n = 3;
        let truth = set_edge(AdjacencyMask::EMPTY, n, 0, 1, true);
        let mut edge = vec![0.0; n * n];
        edge[1] = 0.25;
        edge[n] = 0.5; // P(1 -> 0) = 0.5: skeleton pair (0, 1) at 0.75
        edge[2] = 0.05; // pair (0, 2) at 0.05
        let post = GraphPosterior::new(
            n,
            vec![1.0],
            vec![truth],
            edge.clone(),
            edge,
            crate::graph_posterior::SamplingSummary {
                ess: 1.0,
                diagnostics: antecedent_prob::InferenceDiagnostics::analytic("test"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let cells = bin_skeleton(n, &post, truth);
        assert_eq!(cells[7], (1.0, 0.75, 1.0));
        assert_eq!(cells[0], (2.0, 0.05, 0.0));
    }
}
