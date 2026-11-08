use super::{CandidateDesign, observation_reliability};
use antecedent_core::CausalRng;
use antecedent_prob::WeightedGraphSamples;

#[allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
)]
fn shannon_entropy(weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    if !(total > 0.0) {
        return 0.0;
    }
    let mut h = 0.0;
    for w in weights {
        let p = w / total;
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

/// Discrete soft-observation channel over the posterior's graph categories, with the
/// candidate-independent parts (category index per graph, prior entropy) computed once.
pub(super) struct EntropyChannel {
    weights: Vec<f64>,
    /// Category index of every posterior graph.
    cat_of: Vec<usize>,
    /// Number of distinct categories (≥ 2).
    pub(super) k: usize,
    pub(super) prior_h: f64,
}

impl EntropyChannel {
    /// `None` when the posterior is empty, the features do not line up with it, or there
    /// are fewer than two categories — no observation can then change the entropy.
    pub(super) fn new(
        graphs: &WeightedGraphSamples,
        graph_features: Option<&[u32]>,
    ) -> Option<Self> {
        let n = graphs.n_samples;
        if n == 0 {
            return None;
        }
        if let Some(feat) = graph_features
            && feat.len() != n
        {
            return None;
        }
        let labels: Vec<u64> = if let Some(feat) = graph_features {
            feat.iter().map(|&label| u64::from(label)).collect()
        } else {
            graphs.graph_keys.to_vec()
        };
        let mut categories = labels.clone();
        categories.sort_unstable();
        categories.dedup();
        let k = categories.len();
        if k < 2 {
            return None;
        }
        let cat_of = labels.iter().map(|l| categories.binary_search(l).unwrap_or(0)).collect();
        Some(Self {
            weights: graphs.weights.to_vec(),
            cat_of,
            k,
            prior_h: shannon_entropy(&graphs.weights),
        })
    }

    /// One Monte Carlo draw of information gain for the discrete observation channel:
    /// sample `y ~ P(y | G★, design)`, then return the signed reduction
    /// `H(prior) − H(p(G|y))`.
    ///
    /// Individual draws may be negative when an observation increases posterior entropy;
    /// expectation is taken over draws without pointwise clipping (MM-012).
    pub(super) fn draw(
        &self,
        candidate: &CandidateDesign,
        graph_idx: usize,
        rng: &mut CausalRng,
    ) -> f64 {
        let k = self.k;
        let reliability = observation_reliability(candidate);
        // Zero reliability is the channel's uninformative point: y is then independent of
        // the graph and the posterior equals the prior, so the gain is exactly zero.
        if reliability <= 0.0 {
            return 0.0;
        }
        let true_cat = self.cat_of[graph_idx];

        // Sample the soft observation of the true graph's categorical feature: with
        // probability `reliability` the true category is reported, otherwise a category
        // drawn uniformly from all `k` (the true one included).
        let y = if rng.next_f64() < reliability {
            true_cat
        } else {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the floored value lies in [0, k) because next_f64 is in [0, 1)"
            )]
            let u = (rng.next_f64() * k as f64).floor() as usize;
            u.min(k - 1)
        };

        self.prior_h - self.posterior_entropy(reliability, y)
    }

    /// `p(y | category, reliability)` for the symmetric channel: the reliability-weighted
    /// mixture of a perfect report and a uniform one,
    /// `reliability·1[y = category] + (1 − reliability)/k`. Reliability 0 is uninformative
    /// and 1 is perfect; a lower reliability is a garbling of a higher one, so the
    /// expected information gain is non-decreasing in reliability.
    fn likelihood(&self, reliability: f64, y: usize, cat: usize) -> f64 {
        let uniform = (1.0 - reliability) / self.k as f64;
        if cat == y { reliability + uniform } else { uniform }
    }

    /// Entropy of the graph posterior after observing category `y`.
    pub(super) fn posterior_entropy(&self, reliability: f64, y: usize) -> f64 {
        let mut total = 0.0;
        for (&w, &cat) in self.weights.iter().zip(&self.cat_of) {
            total += w * self.likelihood(reliability, y, cat);
        }
        if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return 0.0;
        }
        let mut h = 0.0;
        for (&w, &cat) in self.weights.iter().zip(&self.cat_of) {
            let p = w * self.likelihood(reliability, y, cat) / total;
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        h
    }

    /// Exact expected information gain `Σ_g p(g) Σ_y p(y|g) [H(G) − H(G|y)]` by
    /// enumerating the observation. The expectation depends only on reliability, so
    /// a ranking reports it directly instead of Monte Carlo draws.
    pub(super) fn exact_gain(&self, reliability: f64) -> f64 {
        let total: f64 = self.weights.iter().sum();
        (0..self.k)
            .map(|y| {
                let p_y: f64 = self
                    .weights
                    .iter()
                    .zip(&self.cat_of)
                    .map(|(w, &cat)| w / total * self.likelihood(reliability, y, cat))
                    .sum();
                p_y * (self.prior_h - self.posterior_entropy(reliability, y))
            })
            .sum()
    }
}

/// Single-draw entropy channel for tests that need the raw per-draw value.
#[cfg(test)]
pub(super) fn eig_graph_entropy(
    candidate: &CandidateDesign,
    graphs: &WeightedGraphSamples,
    graph_idx: usize,
    graph_features: Option<&[u32]>,
    rng: &mut CausalRng,
) -> f64 {
    EntropyChannel::new(graphs, graph_features)
        .map_or(0.0, |channel| channel.draw(candidate, graph_idx, rng))
}
