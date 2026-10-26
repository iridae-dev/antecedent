//! Designs for the edge marginals of the order- and structure-MCMC posteriors at ten and twelve
//! variables.
//!
//! Two kinds of claim, kept apart.
//!
//! *The samplers reproduce their target.* Under the published conditions (a parent cap that is a
//! design input, three by default and checked at two and four as well; a sparse Bernoulli edge
//! prior, four chains, the schedules below) the publication gate passes
//! on at least 80% of replicates, and on a published replicate every directed-edge and skeleton
//! marginal agrees with [`crate::exact_dag_marginals`] (same data, score and prior) within a
//! tolerance derived from the marginal's reported Monte-Carlo standard error. The
//! `*_marginals_match_the_exact_posterior_*` tests assert this. Alongside, they print the binned
//! truth-versus-predicted table of the published skeleton marginals as a diagnostic: that table
//! measures the Gaussian-BIC score, not the sampler, and is not asserted.
//!
//! *The skeleton marginal is calibrated against the generating graph*: among pairs reported at
//! `p = P(i → j) + P(j → i)`, a fraction `p` are adjacent in the truth. The practitioner-scale
//! acceptance designs (`graph_posterior_*_variable_claims_calibrated`) assert it with the pooled
//! residual of each bin within 3.5 cluster-robust standard errors of zero (clusters are
//! replicates, since the pairs of one dataset are dependent).
//!
//! Each replicate draws a random sparse linear-Gaussian SEM (random order, zero to two parents
//! per node, coefficients of magnitude 0.5–1) and simulates the rows. A replicate the gate
//! refuses is counted, not scored.
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
use crate::graph_posterior::{GraphPosterior, GraphPrior};
use crate::order_mcmc::OrderMcmc;
use crate::structure_mcmc::StructureMcmc;

/// Rows per simulated dataset of the standing designs.
const N_ROWS: usize = 400;
/// Replicates (datasets) per standing design.
const N_SIM: u64 = 100;
/// Equal-width bins of the predicted skeleton marginal.
const N_BINS: usize = 10;
/// Fraction of replicates the publication gate must pass.
const MIN_PUBLISHED_FRACTION: f64 = 0.8;
/// Default cap on the parents of any node; the designs take the cap as an input and the
/// parent-cap designs below run others.
const MAX_PARENTS: usize = 3;
/// Other parent caps the standing designs are run at.
const OTHER_MAX_PARENTS: [usize; 2] = [2, 4];
/// Bernoulli edge-inclusion probability of the published prior. A posterior is calibrated for
/// data drawn from its own prior, so the prior states the generator's sparsity: a node has one
/// parent on average, which at ten to twelve variables is about 0.19 to 0.16 of the pairs
/// adjacent, or `0.09` per direction. The uniform prior over DAGs puts roughly even odds on each
/// pair and so reports about 0.075 for pairs the generator never connects.
const EDGE_INCLUSION: f64 = 0.09;

fn normal(rng: &mut CausalRng) -> f64 {
    let (u, v) = (rng.next_f64().max(f64::MIN_POSITIVE), rng.next_f64());
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// A random sparse linear-Gaussian SEM on `n` nodes and its rows: columns and the true graph.
fn sparse_sem(n: usize, rows: usize, seed: u64) -> (Vec<Vec<f64>>, AdjacencyMask) {
    sparse_sem_with_magnitudes(n, rows, seed, (0.5, 0.5))
}

/// [`sparse_sem`] with coefficient magnitudes drawn uniformly from `[floor, floor + span]`: the
/// same graph and random stream for every `(floor, span)`; the design's is `(0.5, 0.5)`.
fn sparse_sem_with_magnitudes(
    n: usize,
    rows: usize,
    seed: u64,
    (floor, span): (f64, f64),
) -> (Vec<Vec<f64>>, AdjacencyMask) {
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
            parents[order[pos]].push((pick, sign * (floor + span * rng.next_f64())));
            truth.set_edge(n, pick, order[pos], true);
        }
    }
    let mut cols = vec![vec![0.0; rows]; n];
    for row in 0..rows {
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
fn bin_skeleton(n: usize, post: &GraphPosterior, truth: &AdjacencyMask) -> BinCells {
    bin_edge_marginals(n, post.edge_marginals(), truth)
}

/// The skeleton marginals of directed-edge marginals `edge` (row-major `from * n + to`) against
/// the generating graph.
fn bin_edge_marginals(n: usize, edge: &[f64], truth: &AdjacencyMask) -> BinCells {
    let mut cells = [(0.0, 0.0, 0.0); N_BINS];
    for i in 0..n {
        for j in (i + 1)..n {
            let p = (edge[i * n + j] + edge[j * n + i]).min(1.0);
            let scaled = p * N_BINS as f64;
            let bin = (1..N_BINS)
                .filter(|&b| scaled >= f64::from(u32::try_from(b).unwrap_or(u32::MAX)))
                .count();
            let adjacent = truth.has_edge(n, i, j) || truth.has_edge(n, j, i);
            cells[bin].0 += 1.0;
            cells[bin].1 += p;
            cells[bin].2 += f64::from(u8::from(adjacent));
        }
    }
    cells
}

/// The prior of the designs, with at most `max_parents` parents per node.
fn design_prior(max_parents: usize) -> GraphPrior {
    GraphPrior::bernoulli_edges(EDGE_INCLUSION).unwrap().with_constraints(DiscoveryConstraints {
        max_parents: Some(max_parents),
        ..Default::default()
    })
}

/// Standard errors of slack on a sampler-versus-exact marginal. A design makes about
/// `N_SIM * n (n - 1)` such comparisons (up to some 20,000), so 4 standard errors (two-sided
/// 6e-5) would be expected to fail once per run on pure sampling noise; 5 (6e-7) leaves a
/// per-run false-failure rate near 1%, and a bias of that size is still far below any
/// reported-interval half-width.
const EXACT_MATCH_Z: f64 = 5.0;
/// Absolute slack on a sampler-versus-exact marginal, for chains whose reported error is zero.
const EXACT_MATCH_FLOOR: f64 = 1e-3;

/// Effective draws of a chain: the median edge ESS over the edges whose exact marginal lies in
/// `[0.05, 0.95]`, where an indicator flips often enough for its ESS to be estimated. `None`
/// when no edge is in that range.
fn chain_effective_draws(post: &GraphPosterior, exact: &[f64]) -> Option<f64> {
    let ess = post.edge_ess()?;
    let mut informative: Vec<f64> = exact
        .iter()
        .zip(ess.iter())
        .filter(|(e, s)| (0.05..=0.95).contains(*e) && s.is_finite() && **s > 0.0)
        .map(|(_, s)| *s)
        .collect();
    if informative.is_empty() {
        return None;
    }
    informative.sort_by(f64::total_cmp);
    Some(informative[informative.len() / 2])
}

/// Monte Carlo error of a sampled marginal whose exact value is `exact`: the larger of the
/// chain's reported MCSE and the binomial error `sqrt(p (1 - p) / n_eff)` at the chain's
/// effective draws. A batch-means MCSE of an indicator that rarely flips (an edge whose exact
/// marginal is near 0 or 1) is estimated from too few transitions and collapses toward zero;
/// the exact marginal gives the indicator's true variance.
fn marginal_error(reported: f64, exact: f64, effective_draws: Option<f64>) -> f64 {
    let binomial = effective_draws.map_or(0.0, |n_eff| (exact * (1.0 - exact) / n_eff).sqrt());
    reported.max(binomial)
}

/// Run `N_SIM` replicates of `sampler` on `n` variables. The publication gate must pass on
/// [`MIN_PUBLISHED_FRACTION`] of them, and on each published replicate every directed-edge
/// marginal must equal the exact one within `EXACT_MATCH_Z * error + EXACT_MATCH_FLOOR` (the error
/// of [`marginal_error`]), and every skeleton marginal within
/// `EXACT_MATCH_Z * (error_ij + error_ji) + EXACT_MATCH_FLOOR` (the two
/// directions are mutually exclusive events, so their errors are not independent and the sum of
/// the standard errors bounds the standard error of the sum).
///
/// The sampler runs with its gate off so every replicate can be compared with the exact posterior:
/// a replicate counts as published when the gate would have passed it, and each replicate prints
/// its worst difference from the exact marginals and the outcome of each gate criterion, so a
/// disagreement is visibly either withheld (correct) or published (a failure). All published
/// disagreements are listed before the test fails.
#[allow(clippy::too_many_lines, reason = "one replicate loop printing every comparison in order")]
fn assert_marginals_match_the_exact_posterior(
    label: &str,
    n: usize,
    max_parents: usize,
    sampler: GatedSampler,
) {
    let vars: Vec<VariableId> =
        (0..n).map(|i| VariableId::from_raw(u32::try_from(i).unwrap())).collect();
    let prior = design_prior(max_parents);
    let label = &format!("{label} max_parents={max_parents}");
    let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut truth_bins: Vec<BinCells> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for rep in 0..N_SIM {
        let (cols, truth) = sparse_sem(n, N_ROWS, 0x00ED_6E00 + rep);
        let data = crate::test_tables::continuous_table(&name_refs, cols);
        let ctx = ExecutionContext::for_tests(9_000 + rep);
        let Some(post) = sampler(false, &data, &vars, &prior, &ctx) else {
            println!("exact-posterior-match {label} n={n} rep={rep} sampler error");
            continue;
        };
        let published = post.diagnostics().converged;
        let exact =
            crate::exact_dag_marginals(&data, &vars, &prior, GraphScoreFamily::GaussianBic, &ctx)
                .unwrap();
        let mcse = post.edge_mcse().expect("a posterior reports its mcse");
        let n_eff = chain_effective_draws(&post, &exact.edge_marginals);
        // A rarely-flipping edge has fewer effective draws than the chain's median; its own ESS,
        // when finite, bounds the draws its marginal is estimated from.
        let edge_draws = |k: usize| {
            let own =
                post.edge_ess().map(|ess| ess[k]).filter(|e| e.is_finite() && *e > 0.0);
            match (n_eff, own) {
                (Some(chain), Some(own)) => Some(chain.min(own)),
                (chain, own) => chain.or(own),
            }
        };
        let mut worst = (0.0_f64, 0usize, 0usize);
        let mut violation: Option<String> = None;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let k = i * n + j;
                let difference = (post.edge_marginals()[k] - exact.edge_marginals[k]).abs();
                if difference > worst.0 {
                    worst = (difference, i, j);
                }
                let error_k = marginal_error(mcse[k], exact.edge_marginals[k], edge_draws(k));
                if violation.is_none() && difference > EXACT_MATCH_Z * error_k + EXACT_MATCH_FLOOR {
                    violation = Some(format!(
                        "{label} n={n} rep={rep} edge {i}->{j}: sampler {:.5} vs exact {:.5} \
                         (error {error_k:.5})",
                        post.edge_marginals()[k], exact.edge_marginals[k],
                    ));
                }
                if i < j {
                    let r = j * n + i;
                    let (sampled, expected) = (
                        post.edge_marginals()[k] + post.edge_marginals()[r],
                        exact.edge_marginals[k] + exact.edge_marginals[r],
                    );
                    let error = marginal_error(mcse[k], exact.edge_marginals[k], edge_draws(k))
                        + marginal_error(mcse[r], exact.edge_marginals[r], edge_draws(r));
                    if violation.is_none()
                        && (sampled - expected).abs() > EXACT_MATCH_Z * error + EXACT_MATCH_FLOOR
                    {
                        violation = Some(format!(
                            "{label} n={n} rep={rep} skeleton {i}-{j}: sampler {sampled:.5} vs \
                             exact {expected:.5} (error {error:.5})"
                        ));
                    }
                }
            }
        }
        let gate: Vec<&str> =
            post.diagnostics().notes.iter().filter_map(|note| note.strip_prefix("gate_")).collect();
        println!(
            "exact-posterior-match {label} n={n} rep={rep} published={published} \
             worst_error={:.5} at {}->{} outside_tolerance={} gate=[{}]",
            worst.0,
            worst.1,
            worst.2,
            violation.is_some(),
            gate.join("; ")
        );
        if !published {
            continue;
        }
        if let Some(message) = violation {
            failures.push(message);
        }
        truth_bins.push(bin_skeleton(n, &post, &truth));
    }
    let rate = truth_bins.len() as f64 / N_SIM as f64;
    println!("exact-posterior-match {label} n={n} published={}/{N_SIM}", truth_bins.len());
    assert!(
        rate >= MIN_PUBLISHED_FRACTION,
        "{label} n={n}: the publication gate passed {rate:.2} of replicates, below \
         {MIN_PUBLISHED_FRACTION}; the published conditions do not support the claim"
    );
    assert!(
        failures.is_empty(),
        "{label} n={n}: {} published replicates disagree with the exact posterior:\n{}",
        failures.len(),
        failures.join("\n")
    );
    for (bin, &(pairs, predicted, observed)) in pool(&truth_bins).iter().enumerate() {
        if pairs > 0.0 {
            println!(
                "exact-posterior-match {label} n={n} truth-diagnostic bin={bin} pairs={pairs} \
                 predicted={:.4} observed={:.4}",
                predicted / pairs,
                observed / pairs
            );
        }
    }
}

/// A design sampler: `(publication gate on, data, variables, prior, context)`.
type GatedSampler =
    fn(bool, &TabularData, &[VariableId], &GraphPrior, &ExecutionContext) -> Option<GraphPosterior>;

fn structure_sampler_gated(
    gate: bool,
    data: &TabularData,
    vars: &[VariableId],
    prior: &GraphPrior,
    ctx: &ExecutionContext,
) -> Option<GraphPosterior> {
    StructureMcmc::new()
        .with_schedule(4, 3000, 12_000, 2)
        .with_diagnostics_gate(gate)
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

fn order_sampler_gated(
    gate: bool,
    data: &TabularData,
    vars: &[VariableId],
    prior: &GraphPrior,
    ctx: &ExecutionContext,
) -> Option<GraphPosterior> {
    OrderMcmc::new()
        .with_schedule(4, 6000, 24_000, 2)
        .with_diagnostics_gate(gate)
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
fn structure_mcmc_marginals_match_the_exact_posterior_at_ten_variables() {
    assert_marginals_match_the_exact_posterior(
        "structure_mcmc",
        10,
        MAX_PARENTS,
        structure_sampler_gated,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn structure_mcmc_marginals_match_the_exact_posterior_at_twelve_variables() {
    assert_marginals_match_the_exact_posterior(
        "structure_mcmc",
        12,
        MAX_PARENTS,
        structure_sampler_gated,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn order_mcmc_marginals_match_the_exact_posterior_at_ten_variables() {
    assert_marginals_match_the_exact_posterior("order_mcmc", 10, MAX_PARENTS, order_sampler_gated);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn order_mcmc_marginals_match_the_exact_posterior_at_twelve_variables() {
    assert_marginals_match_the_exact_posterior("order_mcmc", 12, MAX_PARENTS, order_sampler_gated);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn structure_mcmc_marginals_match_the_exact_posterior_at_other_parent_caps() {
    for max_parents in OTHER_MAX_PARENTS {
        assert_marginals_match_the_exact_posterior(
            "structure_mcmc",
            10,
            max_parents,
            structure_sampler_gated,
        );
    }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn order_mcmc_marginals_match_the_exact_posterior_at_other_parent_caps() {
    for max_parents in OTHER_MAX_PARENTS {
        assert_marginals_match_the_exact_posterior(
            "order_mcmc",
            10,
            max_parents,
            order_sampler_gated,
        );
    }
}

/// Replicates of the diagnostic below that run the exact computation only (cheap).
const DIAGNOSTIC_EXACT_REPLICATES: u64 = 100;
/// Replicates of the diagnostic below that also run each sampler.
const DIAGNOSTIC_SAMPLER_REPLICATES: u64 = 20;

fn pool(cells: &[BinCells]) -> BinCells {
    let mut pooled = [(0.0, 0.0, 0.0); N_BINS];
    for replicate in cells {
        for (bin, cell) in replicate.iter().enumerate() {
            pooled[bin].0 += cell.0;
            pooled[bin].1 += cell.1;
            pooled[bin].2 += cell.2;
        }
    }
    pooled
}

/// Where the bin-0 overstatement of the skeleton marginals comes from, and which publication
/// criterion refuses a replicate (ten variables).
///
/// The exact marginals (dynamic programming, same Gaussian-BIC score, same prior, same parent
/// cap) are the posterior the samplers target. Three questions, all printed:
///
/// 1. Is the posterior itself calibrated on the design's data? The exact skeleton marginals are
///    binned against the generating graph for the design's coefficients (magnitudes 0.5-1.0)
///    and for coefficients that reach zero (0.0-1.0), which is what the diffuse unit-information
///    effect prior of the BIC score assumes of an edge. An exact bin-0 prediction near 0.017 with
///    an observed adjacency near 0.006 under the design, closing under the second generator,
///    says the overstatement is the score's implicit effect-size prior (it keeps mass on
///    near-zero edges that the design never generates), not the sampler.
/// 2. Does each sampler reproduce the exact marginals? The mean difference of sampler and exact
///    skeleton marginals over all pairs and over the pairs the exact posterior puts below 0.1,
///    and the largest difference of any pair, per replicate.
/// 3. Which criterion of the publication gate refuses a replicate (R-hat, bulk ESS, edge
///    Monte-Carlo error, movement), with the worst edge and its exact marginals, so a mixing
///    change is judged on the criterion that fails.
#[test]
#[ignore = "diagnostic: cargo test --release -p antecedent-discovery --lib exact_marginals_locate -- --ignored --nocapture"]
#[allow(clippy::too_many_lines, reason = "one diagnostic printing every comparison in order")]
fn exact_marginals_locate_the_skeleton_overstatement() {
    let n = 10;
    let vars: Vec<VariableId> =
        (0..n).map(|i| VariableId::from_raw(u32::try_from(i).unwrap())).collect();
    let prior = design_prior(MAX_PARENTS);
    let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let draw = |magnitudes: (f64, f64), rep: u64| {
        let (cols, truth) = sparse_sem_with_magnitudes(n, N_ROWS, 0x00ED_6E00 + rep, magnitudes);
        (crate::test_tables::continuous_table(&name_refs, cols), truth)
    };
    let exact_of = |data: &TabularData, rep: u64| {
        crate::exact_dag_marginals(
            data,
            &vars,
            &prior,
            GraphScoreFamily::GaussianBic,
            &ExecutionContext::for_tests(9_000 + rep),
        )
        .unwrap()
    };

    for (label, magnitudes) in
        [("design_0.5-1.0", (0.5, 0.5)), ("zero_reaching_0.0-1.0", (0.0, 1.0))]
    {
        let cells: Vec<BinCells> = (0..DIAGNOSTIC_EXACT_REPLICATES)
            .map(|rep| {
                let (data, truth) = draw(magnitudes, rep);
                bin_edge_marginals(n, &exact_of(&data, rep).edge_marginals, &truth)
            })
            .collect();
        for (bin, &(pairs, predicted, observed)) in pool(&cells).iter().enumerate() {
            if pairs > 0.0 {
                println!(
                    "skeleton-diagnostic exact {label} bin={bin} pairs={pairs} \
                     predicted={:.4} observed={:.4}",
                    predicted / pairs,
                    observed / pairs
                );
            }
        }
    }

    let samplers: [(&str, GatedSampler); 2] =
        [("structure_mcmc", structure_sampler_gated), ("order_mcmc", order_sampler_gated)];
    for (label, sampler) in samplers {
        let (mut gap_all, mut gap_low, mut exact_low, mut pairs_low) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut refused = 0u64;
        for rep in 0..DIAGNOSTIC_SAMPLER_REPLICATES {
            let (data, _) = draw((0.5, 0.5), rep);
            let exact = exact_of(&data, rep);
            let ctx = ExecutionContext::for_tests(9_000 + rep);
            let post = sampler(false, &data, &vars, &prior, &ctx)
                .expect("the ungated sampler returns its posterior");
            let mut worst_gap = 0.0f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    let exact_skeleton =
                        exact.edge_marginals[i * n + j] + exact.edge_marginals[j * n + i];
                    let sampled = post.edge_marginals()[i * n + j] + post.edge_marginals()[j * n + i];
                    worst_gap = worst_gap.max((sampled - exact_skeleton).abs());
                    gap_all += sampled - exact_skeleton;
                    if exact_skeleton < 1.0 / N_BINS as f64 {
                        gap_low += sampled - exact_skeleton;
                        exact_low += exact_skeleton;
                        pairs_low += 1.0;
                    }
                }
            }
            let (mcse, rhat, ess) = (
                post.edge_mcse().unwrap(),
                post.edge_rhat().unwrap(),
                post.edge_ess().unwrap(),
            );
            let worst = (0..n * n).max_by(|&a, &b| mcse[a].total_cmp(&mcse[b])).unwrap();
            let (wi, wj) = (worst / n, worst % n);
            let d = post.diagnostics();
            refused += u64::from(!d.converged);
            println!(
                "skeleton-diagnostic {label} rep={rep} published={} rhat_max={:?} \
                 ess_bulk_min={:?} moved={:?} max_skeleton_gap={worst_gap:.4} \
                 worst_mcse_edge={wi}->{wj} mcse={:.4} rhat={:.3} ess={:.1} \
                 exact_forward={:.3} exact_backward={:.3}",
                d.converged,
                d.rhat_max,
                d.ess_bulk_min,
                d.all_chains_moved,
                mcse[worst],
                rhat[worst],
                ess[worst],
                exact.edge_marginals[worst],
                exact.edge_marginals[wj * n + wi],
            );
        }
        let all_pairs =
            (usize::try_from(DIAGNOSTIC_SAMPLER_REPLICATES).unwrap() * n * (n - 1) / 2) as f64;
        println!(
            "skeleton-diagnostic {label} summary refused={refused}/{DIAGNOSTIC_SAMPLER_REPLICATES} \
             mean_sampler_minus_exact_all={:.5} \
             mean_sampler_minus_exact_where_exact_below_0.1={:.5} \
             mean_exact_where_exact_below_0.1={:.5}",
            gap_all / all_pairs,
            gap_low / pairs_low.max(1.0),
            exact_low / pairs_low.max(1.0),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The design's generator is the graph it reports: every node has at most two parents,
    /// the graph is acyclic, and the rows are not degenerate.
    #[test]
    fn sparse_sem_returns_its_generating_graph() {
        for n in [10usize, 12] {
            let (cols, truth) = sparse_sem(n, N_ROWS, 17);
            assert!(crate::adjacency_mask::mask_is_dag(&truth, n));
            assert!((0..n).all(|v| crate::adjacency_mask::parents_of(&truth, n, v).len() <= 2));
            assert!(cols.iter().all(|c| c.len() == N_ROWS && c.iter().all(|x| x.is_finite())));
        }
    }

    /// A pair lands in the bin of its skeleton marginal, the sum of both directions.
    #[test]
    fn binning_places_a_pair_by_its_skeleton_marginal() {
        let n = 3;
        let truth = crate::adjacency_mask::set_edge(AdjacencyMask::EMPTY, n, 0, 1, true);
        let mut edge = vec![0.0; n * n];
        edge[1] = 0.25;
        edge[n] = 0.5; // P(1 -> 0) = 0.5: skeleton pair (0, 1) at 0.75
        edge[2] = 0.05; // pair (0, 2) at 0.05
        let post = GraphPosterior::from_atoms(
            n,
            vec![1.0],
            vec![truth.clone()],
            edge.clone(),
            edge,
            crate::graph_posterior::SamplingSummary {
                ess: 1.0,
                diagnostics: antecedent_prob::InferenceDiagnostics::analytic("test"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let cells = bin_skeleton(n, &post, &truth);
        assert_eq!(cells[7], (1.0, 0.75, 1.0));
        assert_eq!(cells[0], (2.0, 0.05, 0.0));
    }
}
