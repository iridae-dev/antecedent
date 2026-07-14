//! Exact and approximate Shapley value estimation (DESIGN.md §17.4).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{CausalRng, ComponentId, ExecutionContext, ShapleyConfig, ShapleyMode};

use crate::coalition::{CoalitionCache, CoalitionKey};
use crate::error::AttributionError;
use crate::result::{
    CacheStats, ComponentContribution, ComputeBudget, InteractionTerm,
};

/// Payoff function `v(S)` for a coalition bitmask (`bit i` ⇒ player `i` present).
pub trait CoalitionPayoff {
    /// Evaluate the value of coalition `mask`.
    ///
    /// # Errors
    ///
    /// Model / sampling failures.
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError>;
}

/// Shapley estimation output.
#[derive(Clone, Debug)]
pub struct ShapleyEstimate {
    /// Ordered player ids.
    pub players: Vec<ComponentId>,
    /// Shapley values φ_i.
    pub values: Vec<f64>,
    /// Optional pairwise interactions from sequential residuals (empty for pure Shapley).
    pub interactions: Vec<InteractionTerm>,
    /// Compute budget.
    pub budget: ComputeBudget,
    /// Mean Monte Carlo stderr across players (approx modes only).
    pub monte_carlo_stderr: Option<f64>,
    /// Per-player MC stderr.
    pub component_mc_stderr: Option<Vec<f64>>,
    /// Cache stats.
    pub cache_stats: CacheStats,
}

impl ShapleyEstimate {
    /// Convert to component contributions.
    #[must_use]
    pub fn into_contributions(self) -> Vec<ComponentContribution> {
        let stderrs = self.component_mc_stderr;
        self.players
            .into_iter()
            .zip(self.values)
            .enumerate()
            .map(|(i, (component, contribution))| {
                let stderr = stderrs.as_ref().and_then(|s| s.get(i).copied());
                ComponentContribution {
                    component,
                    contribution,
                    stderr,
                    ci_low: stderr.map(|s| contribution - 1.96 * s),
                    ci_high: stderr.map(|s| contribution + 1.96 * s),
                }
            })
            .collect()
    }
}

/// Enforce exact Shapley size limits (DESIGN.md §17.4).
///
/// # Errors
///
/// [`AttributionError::ExactShapleyRejected`] when Exact exceeds the limit
/// without override, or component count exceeds 64 (bitset width).
pub fn check_shapley_size(
    n_components: usize,
    config: &ShapleyConfig,
) -> Result<(), AttributionError> {
    if n_components > 64 {
        return Err(AttributionError::SizeLimit {
            kind: "components",
            requested: n_components,
            max: 64,
        });
    }
    if matches!(config.mode, ShapleyMode::Exact)
        && n_components > config.max_exact_components
        && !config.allow_exact_override
    {
        return Err(AttributionError::ExactShapleyRejected {
            n_components,
            max: config.max_exact_components,
        });
    }
    Ok(())
}

/// Estimate Shapley values for `players` under `payoff`.
///
/// Uses a semantic [`CoalitionCache`] keyed by coalition mask. Approximate
/// modes always populate `monte_carlo_stderr`.
///
/// # Errors
///
/// Size limits, empty players, or payoff failures.
pub fn estimate_shapley<P: CoalitionPayoff>(
    players: &[ComponentId],
    config: &ShapleyConfig,
    payoff: &mut P,
    ctx: &ExecutionContext,
) -> Result<ShapleyEstimate, AttributionError> {
    config.validate()?;
    if players.is_empty() {
        return Err(AttributionError::Message("Shapley requires ≥1 player".into()));
    }
    let n = players.len();
    check_shapley_size(n, config)?;

    let mut cache = CoalitionCache::from_policy(ctx.cache_policy);
    let mut budget = ComputeBudget::default();

    let eval = |mask: u64,
                    payoff: &mut P,
                    cache: &mut CoalitionCache,
                    budget: &mut ComputeBudget|
     -> Result<f64, AttributionError> {
        let key = CoalitionKey { mask, tag: 0 };
        if let Some(v) = cache.get(key) {
            return Ok(v);
        }
        let v = payoff.value(mask)?;
        budget.evaluations += 1;
        cache.insert(key, v);
        Ok(v)
    };

    match config.mode {
        ShapleyMode::Exact => {
            let n_coalitions = 1u64 << n;
            budget.exact_coalitions = n_coalitions;
            let mut phi = vec![0.0; n];
            let fact = factorial_weights(n);
            for mask in 0..n_coalitions {
                let v_s = eval(mask, payoff, &mut cache, &mut budget)?;
                for i in 0..n {
                    let bit = 1u64 << i;
                    if mask & bit != 0 {
                        continue;
                    }
                    let v_si = eval(mask | bit, payoff, &mut cache, &mut budget)?;
                    let s = (mask.count_ones()) as usize;
                    phi[i] += fact[s] * (v_si - v_s);
                }
            }
            Ok(ShapleyEstimate {
                players: players.to_vec(),
                values: phi,
                interactions: Vec::new(),
                budget,
                monte_carlo_stderr: None,
                component_mc_stderr: None,
                cache_stats: cache.stats(),
            })
        }
        ShapleyMode::MonteCarlo { n_samples } => {
            budget.samples = n_samples as u64;
            let mut rng = CausalRng::from_seed(config.seed);
            let mut phi = vec![0.0; n];
            let mut phi2 = vec![0.0; n];
            for _ in 0..n_samples {
                let mut order: Vec<usize> = (0..n).collect();
                shuffle(&mut order, &mut rng);
                let mut mask = 0u64;
                let mut v_prev = eval(0, payoff, &mut cache, &mut budget)?;
                let mut sample_phi = vec![0.0; n];
                for &i in &order {
                    let bit = 1u64 << i;
                    mask |= bit;
                    let v_new = eval(mask, payoff, &mut cache, &mut budget)?;
                    sample_phi[i] = v_new - v_prev;
                    v_prev = v_new;
                }
                for i in 0..n {
                    phi[i] += sample_phi[i];
                    phi2[i] += sample_phi[i] * sample_phi[i];
                }
            }
            let ns = n_samples as f64;
            for i in 0..n {
                phi[i] /= ns;
                phi2[i] = ((phi2[i] / ns) - phi[i] * phi[i]).max(0.0).sqrt() / ns.sqrt();
            }
            let mean_se = phi2.iter().sum::<f64>() / n as f64;
            Ok(ShapleyEstimate {
                players: players.to_vec(),
                values: phi,
                interactions: Vec::new(),
                budget,
                monte_carlo_stderr: Some(mean_se),
                component_mc_stderr: Some(phi2),
                cache_stats: cache.stats(),
            })
        }
        ShapleyMode::Permutation { n_permutations } => {
            let mut cfg = *config;
            cfg.mode = ShapleyMode::MonteCarlo {
                n_samples: n_permutations,
            };
            estimate_shapley(players, &cfg, payoff, ctx)
        }
        _ => Err(AttributionError::Message(
            "unsupported ShapleyMode variant".into(),
        )),
    }
}

/// Sequential (path-dependent) allocation along a fixed order.
///
/// Returns contributions and explicit pairwise consecutive interaction residuals
/// relative to the Shapley-incomparable sequential path (nonadditive marker).
///
/// # Errors
///
/// Payoff failures.
pub fn sequential_allocate<P: CoalitionPayoff>(
    order: &[ComponentId],
    player_index: &dyn Fn(ComponentId) -> Option<usize>,
    payoff: &mut P,
    ctx: &ExecutionContext,
) -> Result<ShapleyEstimate, AttributionError> {
    if order.is_empty() {
        return Err(AttributionError::Message("sequential order is empty".into()));
    }
    let mut cache = CoalitionCache::from_policy(ctx.cache_policy);
    let mut budget = ComputeBudget::default();
    let mut mask = 0u64;
    let mut v_prev = {
        let key = CoalitionKey { mask: 0, tag: 0 };
        if let Some(v) = cache.get(key) {
            v
        } else {
            let v = payoff.value(0)?;
            budget.evaluations += 1;
            cache.insert(key, v);
            v
        }
    };
    let mut values = Vec::with_capacity(order.len());
    let mut interactions = Vec::new();
    let mut prev_component: Option<ComponentId> = None;
    for &comp in order {
        let idx = player_index(comp).ok_or_else(|| {
            AttributionError::Message(format!("component {comp} not in player set"))
        })?;
        let bit = 1u64 << idx;
        mask |= bit;
        let key = CoalitionKey { mask, tag: 0 };
        let v_new = if let Some(v) = cache.get(key) {
            v
        } else {
            let v = payoff.value(mask)?;
            budget.evaluations += 1;
            cache.insert(key, v);
            v
        };
        let marginal = v_new - v_prev;
        values.push(marginal);
        if let Some(a) = prev_component {
            // Residual vs independent sum is zero for pure sequential; record
            // consecutive pair with the marginal as an explicit nonadditive path term.
            interactions.push(InteractionTerm { a, b: comp, value: 0.0 });
        }
        prev_component = Some(comp);
        v_prev = v_new;
        let _ = interactions; // keep interactions vec for API; values stay 0 for additive path
    }
    // Re-enable interaction recording meaningfully: store consecutive marginals as path terms.
    let interactions: Vec<InteractionTerm> = order
        .windows(2)
        .enumerate()
        .map(|(i, w)| InteractionTerm {
            a: w[0],
            b: w[1],
            value: values[i + 1], // next marginal along the ordered path
        })
        .collect();

    Ok(ShapleyEstimate {
        players: order.to_vec(),
        values,
        interactions,
        budget,
        monte_carlo_stderr: None,
        component_mc_stderr: None,
        cache_stats: cache.stats(),
    })
}

fn factorial_weights(n: usize) -> Vec<f64> {
    // w(s) = s! * (n-s-1)! / n!
    let mut fact = vec![1.0; n + 1];
    for i in 1..=n {
        fact[i] = fact[i - 1] * i as f64;
    }
    let n_fact = fact[n];
    let mut w = vec![0.0; n];
    for s in 0..n {
        w[s] = fact[s] * fact[n - s - 1] / n_fact;
    }
    w
}

fn shuffle(xs: &mut [usize], rng: &mut CausalRng) {
    for i in (1..xs.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        xs.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{CachePolicy, ComponentId, Parallelism};

    struct AdditivePayoff {
        weights: Vec<f64>,
    }

    impl CoalitionPayoff for AdditivePayoff {
        fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
            let mut s = 0.0;
            for (i, &w) in self.weights.iter().enumerate() {
                if mask & (1u64 << i) != 0 {
                    s += w;
                }
            }
            Ok(s)
        }
    }

    #[test]
    fn exact_shapley_recovers_additive_game() {
        let players: Vec<_> = (0..3).map(ComponentId::from_raw).collect();
        let mut payoff = AdditivePayoff { weights: vec![1.0, 2.0, 3.0] };
        let cfg = ShapleyConfig::exact();
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.cache_policy = CachePolicy::enabled(Some(1_000_000));
        let est = estimate_shapley(&players, &cfg, &mut payoff, &ctx).unwrap();
        assert!((est.values[0] - 1.0).abs() < 1e-9);
        assert!((est.values[1] - 2.0).abs() < 1e-9);
        assert!((est.values[2] - 3.0).abs() < 1e-9);
        assert!(est.cache_stats.hits > 0 || est.budget.evaluations > 0);
    }

    #[test]
    fn exact_rejects_above_limit() {
        let players: Vec<_> = (0..5).map(ComponentId::from_raw).collect();
        let mut payoff = AdditivePayoff { weights: vec![1.0; 5] };
        let cfg = ShapleyConfig::exact().with_max_exact_components(3);
        let ctx = ExecutionContext::for_tests(1);
        let err = estimate_shapley(&players, &cfg, &mut payoff, &ctx).unwrap_err();
        assert!(matches!(err, AttributionError::ExactShapleyRejected { .. }));
    }

    #[test]
    fn monte_carlo_reports_stderr() {
        let players: Vec<_> = (0..4).map(ComponentId::from_raw).collect();
        let mut payoff = AdditivePayoff {
            weights: vec![1.0, 1.0, 1.0, 1.0],
        };
        let cfg = ShapleyConfig::monte_carlo(200).with_seed(7);
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.cache_policy = CachePolicy::enabled(None);
        ctx.parallelism = Parallelism::serial();
        let est = estimate_shapley(&players, &cfg, &mut payoff, &ctx).unwrap();
        assert!(est.monte_carlo_stderr.is_some());
        assert!(est.component_mc_stderr.is_some());
        for (v, w) in est.values.iter().zip([1.0, 1.0, 1.0, 1.0]) {
            assert!((v - w).abs() < 0.15, "v={v}");
        }
    }
}
