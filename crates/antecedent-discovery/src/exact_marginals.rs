//! Exact edge marginals of the DAG posterior by dynamic programming over node subsets.
//!
//! [`crate::ExactDagPosterior`] lists every labelled DAG at its default width of six variables
//! (3.8 million graphs; seven variables have 1.1 billion). The Gaussian-BIC posterior with a
//! uniform or independent-Bernoulli edge prior and a parent cap is a product of per-node
//! factors over the DAGs, `P(G) ∝ Π_v f_v(Pa_v(G))`, and for such a product the sum over all
//! DAGs, and the sum over the DAGs that contain a given edge, are computed without enumerating
//! them. The result is the exact `P(i → j)` of every ordered pair, the quantity the samplers
//! estimate. The subset-pair visits are quoted against the execution; the default operation
//! declaration is [`EXACT_MARGINAL_WORK_BUDGET`] (fifteen variables is the largest `n` that fits it).
//!
//! # The recursion
//!
//! Let `A(S)` be the sum of `Π_{v ∈ S} f_v(Pa_v)` over the DAGs on the node set `S`, parents
//! taken within `S`, and `g_v(U) = Σ_{P ⊆ U} f_v(P)` the mass of the parent sets of `v` inside
//! `U`. Choosing a non-empty set `T` of sinks and removing it (Robinson's recursion, with
//! inclusion–exclusion over which nodes of the DAG are sinks):
//!
//! ```text
//! A(∅) = 1,    A(S) = Σ_{∅ ≠ T ⊆ S} (-1)^{|T|+1} · A(S \ T) · Π_{v ∈ T} g_v(S \ T).
//! ```
//!
//! `A(V)` is the normalizing constant. The same recursion with `f_v` replaced by its restriction
//! to the parent sets that contain `i` gives the mass of the DAGs with the edge `i → v`, and
//! `P(i → v)` is the ratio. Each pass costs `O(3^n n)` and there are `n (n - 1) + 1` of them.
//!
//! # Numerics
//!
//! Local BIC scores run to tens of thousands of nats, and the factors of a node differ by
//! hundreds, so nothing is held in linear form. The parent-set masses `g_v` are kept as
//! logarithms (a log-sum-exp subset transform). `A(S)` is carried relative to `best(S)`, the
//! largest weight of any DAG on `S` (a max-plus recursion), so `Ã(S) = A(S) / exp(best(S))`
//! lies between one and the number of DAGs. Every term of the alternating sum is the mass of
//! the DAGs in which the nodes of `T` are sinks, which is at most `A(S)`; the sum therefore
//! loses at most `log2(2^n)` bits to cancellation however concentrated the posterior is, and
//! each marginal is accurate to about `1e-12`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_state::GraphScoreFamily;

use crate::adjacency_mask::AdjacencyMask;
use crate::error::DiscoveryError;
use crate::graph_posterior::{GraphPrior, GraphPriorTable};
use crate::graph_score::{ChainTarget, ScoredTarget, tabular_score_data};

/// Default operation declaration of one exact-marginal computation, in subset-pair visits:
/// `3^n` pairs per pass, `n` factors per pair, and `n (n - 1) + 1` passes.
const EXACT_MARGINAL_WORK_BUDGET: u128 = 1 << 36;

/// The subset-pair visits an exact-marginal run on `n` variables costs.
///
/// A product that does not fit in `u128` is reported as [`u128::MAX`]: it does not fit the
/// default declaration.
const fn exact_marginal_work(n: usize) -> u128 {
    if n >= 128 {
        return u128::MAX;
    }
    let Some(base) = 3u128.checked_pow(n as u32) else {
        return u128::MAX;
    };
    let n = n as u128;
    let passes = n * n.saturating_sub(1) + 1;
    base.saturating_mul(n).saturating_mul(passes)
}

/// Largest `n` whose subset-pair visits fit the default declaration
/// [`EXACT_MARGINAL_WORK_BUDGET`]: `3^15 * 15 * 211` at fifteen. A larger `n` is an operations
/// receipt against that declaration. This constant is not a second refusal: the visit quote is
/// what stops the recursion.
pub const EXACT_MARGINAL_MAX_NODES: usize = 15;

/// Bytes of the subset tables (`3n + 3` arrays of `2^n` floats) before they are allocated.
///
/// `None` when `n` is at least the width of a `usize` bit index. That width is the index of a
/// parent-set bitset, not a work cap.
fn marginal_table_bytes(n: usize) -> Option<u64> {
    if n >= usize::BITS as usize {
        return None;
    }
    let sets = 1u64 << n;
    let arrays = (3 * n as u64).checked_add(3)?;
    sets.checked_mul(8)?.checked_mul(arrays)
}

/// Quote subset-pair visits and table bytes against the execution. A miss is a receipt and the
/// recursion is not entered.
pub(crate) fn admit_exact_marginals(
    n: usize,
    ctx: &ExecutionContext,
) -> Result<(), DiscoveryError> {
    let work = u64::try_from(exact_marginal_work(n)).unwrap_or(u64::MAX);
    let bytes = marginal_table_bytes(n).unwrap_or(u64::MAX);
    let operations = usize::try_from(EXACT_MARGINAL_WORK_BUDGET).unwrap_or(usize::MAX);
    crate::exact_enumeration::admit_quote(ctx, operations, bytes, work, "exact marginal recursion")
}

/// The exact edge marginals of a DAG posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactDagMarginals {
    /// Number of variables.
    pub n_vars: usize,
    /// `P(i → j)`, row-major `from * n_vars + to`, zero on the diagonal.
    pub edge_marginals: Arc<[f64]>,
    /// Natural logarithm of the pseudo-posterior normalizer
    /// `Σ_G exp(score(G) + log prior(G))` over constraint-valid DAGs with a finite score.
    /// The Gaussian score is -BIC/2 and counts each fitted residual variance. This is
    /// not an exact marginal likelihood.
    pub log_evidence: f64,
    /// The variables the marginals index, in order, when they are a selection of a larger
    /// screened set; `None` when they cover every variable of the call.
    pub variables: Option<Arc<[VariableId]>>,
    /// Statement that the marginals are conditional on a CI screen, when they are.
    pub conditional_on_screen: Option<Arc<str>>,
}

impl ExactDagMarginals {
    /// The marginals pooled across imputations: the means of the per-imputation edge
    /// marginals (the marginals of the equal-weight mixture of the per-imputation posteriors)
    /// and the log of the mean of the per-imputation normalizers.
    ///
    /// # Errors
    ///
    /// No marginals, or marginals over different variable counts or selections.
    pub fn pooled_over_imputations(parts: &[Self]) -> Result<Self, DiscoveryError> {
        let first = parts.first().ok_or_else(|| {
            DiscoveryError::unsupported("pooling over imputations needs at least one marginal set")
        })?;
        if parts.iter().any(|p| p.n_vars != first.n_vars || p.variables != first.variables) {
            return Err(DiscoveryError::unsupported(
                "the imputations' marginals are over different variables",
            ));
        }
        let share = 1.0 / parts.len() as f64;
        let mut edge = vec![0.0; first.edge_marginals.len()];
        for part in parts {
            for (cell, value) in edge.iter_mut().zip(part.edge_marginals.iter()) {
                *cell += share * value;
            }
        }
        let top = parts.iter().map(|p| p.log_evidence).fold(f64::NEG_INFINITY, f64::max);
        let mean: f64 = parts.iter().map(|p| share * (p.log_evidence - top).exp()).sum();
        Ok(Self {
            n_vars: first.n_vars,
            edge_marginals: Arc::from(edge),
            log_evidence: top + mean.ln(),
            variables: first.variables.clone(),
            conditional_on_screen: first.conditional_on_screen.clone(),
        })
    }
}

/// Exact `P(i → j)` of the specified DAG posterior of `variables`.
///
/// The prior, its edge probabilities and every constraint (forbidden and required edges,
/// tiers, the parent cap) are those of [`crate::ExactDagPosterior`] and the samplers: the
/// factor of a node is `exp(local BIC + log prior)` of its parent set, zero where the
/// constraints forbid the set or the score is not finite.
///
/// # Errors
///
/// Invalid constraints, a visit or byte quote the execution does not hold, a score family outside the supported Gaussian families, data that cannot be scored, no constraint-valid DAG with a finite score,
/// or cancellation.
pub fn exact_dag_marginals(
    data: &TabularData,
    variables: &[VariableId],
    prior: &GraphPrior,
    score_family: GraphScoreFamily,
    ctx: &ExecutionContext,
) -> Result<ExactDagMarginals, DiscoveryError> {
    prior.validate_for(variables)?;
    prior.constraints.validate()?;
    let n = variables.len();
    if n == 0 {
        return Err(DiscoveryError::unsupported(
            "exact DAG marginals require at least one variable",
        ));
    }
    admit_exact_marginals(n, ctx)?;
    if score_family == GraphScoreFamily::GaussianSignedUniform {
        let data = tabular_score_data(data, variables)?;
        let (prior_table, mut factors) =
            crate::graph_score::signed_uniform_local_table(&data, prior, variables, ctx)?;
        for (node, row) in factors.iter_mut().enumerate() {
            for (subset, score) in row.iter_mut().enumerate().filter(|(_, s)| s.is_finite()) {
                let parents = AdjacencyMask::from_word(subset as u64);
                *score += prior_table.node_log_prior(node, &parents).expect("admissible family");
            }
        }
        return marginals_from_factors(n, &factors, ctx);
    }
    if !matches!(score_family, GraphScoreFamily::GaussianBic) {
        return Err(DiscoveryError::unsupported(
            "exact DAG marginals support GaussianBic or GaussianSignedUniform",
        ));
    }
    let score_data = tabular_score_data(data, variables)?;
    let table = GraphPriorTable::new(n, prior, variables);
    let mut target = ScoredTarget::new(n, &score_data, score_family, &table);

    // The log factor of every (node, parent set); parent sets are node bitsets.
    let sets = 1usize << n;
    let mut log_factor = vec![vec![f64::NEG_INFINITY; sets]; n];
    for (node, row) in log_factor.iter_mut().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(DiscoveryError::Cancelled);
        }
        for (parents, slot) in row.iter_mut().enumerate() {
            if parents & (1 << node) != 0 {
                continue;
            }
            let set = AdjacencyMask::from_word(u64::try_from(parents).unwrap_or(u64::MAX));
            if let Some(score) = target.node_score(node, &set) {
                *slot = score;
            }
        }
    }
    marginals_from_factors(n, &log_factor, ctx)
}

/// `a + b` in the log domain.
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (high, low) = if a >= b { (a, b) } else { (b, a) };
    high + (low - high).exp().ln_1p()
}

/// In place, `values[U] = combine over P ⊆ U of the original values[P]` (the subset transform).
fn fold_subsets(values: &mut [f64], n: usize, combine: impl Fn(f64, f64) -> f64) {
    for bit in 0..n {
        let mask = 1usize << bit;
        for set in 0..values.len() {
            if set & mask != 0 {
                values[set] = combine(values[set], values[set ^ mask]);
            }
        }
    }
}

/// The marginals of the posterior whose factors are `exp(log_factor[v][P])`.
fn marginals_from_factors(
    n: usize,
    log_factor: &[Vec<f64>],
    ctx: &ExecutionContext,
) -> Result<ExactDagMarginals, DiscoveryError> {
    let sets = 1usize << n;
    // best(S): the largest log weight of a DAG on S, removing one sink at a time.
    let best_parent: Vec<Vec<f64>> = log_factor
        .iter()
        .map(|row| {
            let mut best = row.clone();
            fold_subsets(&mut best, n, f64::max);
            best
        })
        .collect();
    let mut best = vec![f64::NEG_INFINITY; sets];
    best[0] = 0.0;
    for set in 1..sets {
        let mut top = f64::NEG_INFINITY;
        for (node, parents) in best_parent.iter().enumerate() {
            if set & (1 << node) != 0 {
                let rest = set ^ (1 << node);
                top = top.max(best[rest] + parents[rest]);
            }
        }
        best[set] = top;
    }
    if !best[sets - 1].is_finite() {
        return Err(DiscoveryError::unsupported(
            "no DAG satisfies the constraints with a finite score",
        ));
    }
    // log g_v(U) = log Σ_{P ⊆ U} f_v(P).
    let mut log_mass: Vec<Vec<f64>> = log_factor
        .iter()
        .map(|row| {
            let mut mass = row.clone();
            fold_subsets(&mut mass, n, log_add);
            mass
        })
        .collect();

    let total = scaled_total(n, &best, &log_mass);
    if !(total.is_finite() && total > 0.0) {
        return Err(DiscoveryError::unsupported(
            "the exact DAG recursion produced no positive normalizing constant",
        ));
    }
    let mut marginals = vec![0.0; n * n];
    for target in 0..n {
        for source in 0..n {
            if source == target {
                continue;
            }
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            // The mass of the DAGs in which `source` is a parent of `target`.
            let mut with_source: Vec<f64> = (0..sets)
                .map(|parents| {
                    if parents & (1 << source) == 0 {
                        f64::NEG_INFINITY
                    } else {
                        log_factor[target][parents]
                    }
                })
                .collect();
            fold_subsets(&mut with_source, n, log_add);
            let unrestricted = std::mem::replace(&mut log_mass[target], with_source);
            let mass = scaled_total(n, &best, &log_mass);
            log_mass[target] = unrestricted;
            marginals[source * n + target] = (mass / total).clamp(0.0, 1.0);
        }
    }
    Ok(ExactDagMarginals {
        n_vars: n,
        edge_marginals: Arc::from(marginals),
        log_evidence: total.ln() + best[sets - 1],
        variables: None,
        conditional_on_screen: None,
    })
}

/// `A(V) / exp(best(V))` for the parent-set masses `log_mass`, by the sink recursion, with every
/// term scaled by the best weight of the node set it belongs to.
fn scaled_total(n: usize, best: &[f64], log_mass: &[Vec<f64>]) -> f64 {
    let sets = 1usize << n;
    let mut scaled = vec![0.0_f64; sets];
    scaled[0] = 1.0;
    for set in 1..sets {
        if !best[set].is_finite() {
            continue;
        }
        let mut sum = 0.0;
        let mut sinks = set;
        while sinks != 0 {
            let rest = set ^ sinks;
            if scaled[rest] != 0.0 {
                let mut exponent = best[rest] - best[set];
                let mut members = sinks;
                while members != 0 {
                    let node = members.trailing_zeros() as usize;
                    exponent += log_mass[node][rest];
                    members &= members - 1;
                }
                if exponent.is_finite() {
                    let term = scaled[rest] * exponent.exp();
                    if sinks.count_ones() % 2 == 1 {
                        sum += term;
                    } else {
                        sum -= term;
                    }
                }
            }
            sinks = (sinks - 1) & set;
        }
        scaled[set] = sum;
    }
    scaled[sets - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEG: f64 = f64::NEG_INFINITY;

    /// Two nodes: the DAGs are the empty graph and the two single edges.
    #[test]
    fn the_advertised_size_is_the_largest_within_the_work_budget() {
        assert!(exact_marginal_work(EXACT_MARGINAL_MAX_NODES) <= EXACT_MARGINAL_WORK_BUDGET);
        assert!(exact_marginal_work(EXACT_MARGINAL_MAX_NODES + 1) > EXACT_MARGINAL_WORK_BUDGET);
    }

    #[test]
    fn two_nodes_have_the_three_dags_of_the_closed_form() {
        // Parent sets are node bitsets: node 0 has {} (0) and {1} (2); node 1 has {} (0) and
        // {0} (1). The log factors are a, b for node 0 and c, d for node 1.
        let (a, b, c, d) = (0.2_f64, 1.1, -0.4, 0.7);
        let factors = vec![vec![a, NEG, b, NEG], vec![c, d, NEG, NEG]];
        let result = marginals_from_factors(2, &factors, &ExecutionContext::for_tests(0)).unwrap();
        // The empty graph weighs e^(a + c), the edge 0 -> 1 e^(a + d), the edge 1 -> 0 e^(b + c).
        let (empty, forward, backward) = ((a + c).exp(), (a + d).exp(), (b + c).exp());
        let z = empty + forward + backward;
        assert!((result.edge_marginals[1] - forward / z).abs() < 1e-14);
        assert!((result.edge_marginals[2] - backward / z).abs() < 1e-14);
        assert!(result.edge_marginals[0].abs() < 1e-15 && result.edge_marginals[3].abs() < 1e-15);
        assert!((result.log_evidence - z.ln()).abs() < 1e-13);
    }

    /// Scores a thousand nats from zero, with the empty parent set five thousand nats below the
    /// best of every node, underflow any linear-domain recursion; the marginals are still exact.
    #[test]
    fn a_posterior_concentrated_by_thousands_of_nats_keeps_its_marginals() {
        // Three nodes. Allowed parent sets: node 0 {} (-5000) and {1} (1999); node 1 {} (-5000),
        // {0} (2000) and {2} (1999); node 2 {} (-5000) and {1} (2000). The DAGs with a weight
        // that matters are the chain 0 -> 1 -> 2 (e^-1000), the fork 0 <- 1 -> 2 (e^-1001) and
        // the reversed chain 2 -> 1 -> 0 (e^-1002); every other acyclic choice is more than
        // 6000 nats below them.
        let n = 3;
        let mut factors = vec![vec![NEG; 1 << n]; n];
        for row in &mut factors {
            row[0] = -5000.0;
        }
        factors[0][0b010] = 1999.0;
        factors[1][0b001] = 2000.0;
        factors[1][0b100] = 1999.0;
        factors[2][0b010] = 2000.0;
        let result = marginals_from_factors(n, &factors, &ExecutionContext::for_tests(0)).unwrap();
        let r = (-1.0_f64).exp();
        let z = 1.0 + r + r * r;
        let cell = |from: usize, to: usize| result.edge_marginals[from * n + to];
        assert!((cell(0, 1) - 1.0 / z).abs() < 1e-9, "{:?}", result.edge_marginals);
        assert!((cell(1, 2) - (1.0 + r) / z).abs() < 1e-9);
        assert!((cell(1, 0) - (r + r * r) / z).abs() < 1e-9);
        assert!((cell(2, 1) - r * r / z).abs() < 1e-9);
        assert!(cell(0, 2).abs() < 1e-9 && cell(2, 0).abs() < 1e-9);
        assert!((result.log_evidence - (-1000.0 + z.ln())).abs() < 1e-9);
    }

    #[test]
    fn a_visit_quote_past_the_declaration_is_an_operations_receipt() {
        let ctx = ExecutionContext::for_tests(0);
        let text =
            admit_exact_marginals(EXACT_MARGINAL_MAX_NODES + 1, &ctx).unwrap_err().to_string();
        assert!(text.contains("search.operations"), "{text}");
        assert!(text.contains("unevaluated [exact marginal recursion]"), "{text}");
        assert!(!text.contains("at most"), "{text}");
        assert!(!text.contains("order_mcmc"), "{text}");
        let mut tight = ExecutionContext::for_tests(0);
        tight.set_memory_limit(Some(1));
        let text = admit_exact_marginals(2, &tight).unwrap_err().to_string();
        assert!(text.contains("search.memory"), "{text}");
        assert!(text.contains("exact marginal recursion"), "{text}");
    }
}
