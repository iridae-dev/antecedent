//! Exact and approximate Shapley value estimation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalRng, ComponentId, ExecutionContext, ShapleyConfig, ShapleyMode};
use antecedent_kernels::shuffle;
use antecedent_stats::student_t_ppf;

use crate::coalition::{CoalitionCache, CoalitionKey};
use crate::error::AttributionError;
use crate::result::{CacheStats, ComponentContribution, ComputeBudget, InteractionTerm};

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
    /// Shapley values `φ_i`.
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
    ///
    /// CIs use a Student-t critical value with `n_samples - 1` degrees of freedom (from
    /// [`ComputeBudget::samples`]) rather than a fixed asymptotic `z = 1.96`: Monte Carlo /
    /// permutation Shapley runs often use modest sample counts, where the normal
    /// approximation understates interval width. Exact mode never populates `stderr`
    /// (`component_mc_stderr` is `None`), so `ci_low`/`ci_high` stay `None` there regardless
    /// of the (unused) critical value. `n_samples == 1` still yields `stderr = +inf`
    /// (`df == 0` also yields `t_crit = +inf`, i.e. `+inf * +inf = +inf`, not NaN) —
    /// preserving the existing single-sample behavior.
    #[must_use]
    pub fn into_contributions(self) -> Vec<ComponentContribution> {
        let stderrs = self.component_mc_stderr;
        #[allow(clippy::cast_precision_loss)]
        let df = self.budget.samples.saturating_sub(1) as f64;
        let t_crit = student_t_ppf(0.975, df);
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
                    ci_low: stderr.map(|s| contribution - t_crit * s),
                    ci_high: stderr.map(|s| contribution + t_crit * s),
                }
            })
            .collect()
    }
}

/// Enforce exact Shapley size limits.
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
    if matches!(config.mode, ShapleyMode::Exact) && n_components >= 64 {
        return Err(AttributionError::SizeLimit {
            kind: "exact Shapley components",
            requested: n_components,
            max: 63,
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
        return Err(AttributionError::unsupported("Shapley requires ≥1 player"));
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
                if ctx.cancellation.is_cancelled() {
                    return Err(AttributionError::Cancelled);
                }
                if let Some(p) = &ctx.progress {
                    #[allow(clippy::cast_precision_loss)]
                    p.report(mask as f64 / n_coalitions as f64, "shapley");
                }
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
            // Welford's online sum-of-squared-deviations (from the running mean),
            // accumulated per sample. Numerically stable vs. the naive two-pass
            // `E[X^2] - E[X]^2` form, which suffers catastrophic cancellation when
            // the mean is large relative to the variance.
            let mut phi_m2 = vec![0.0; n];
            let mut completed = 0u64;
            // Permutation scratch reused across samples (was two fresh Vecs
            // per permutation).
            let mut order: Vec<usize> = Vec::with_capacity(n);
            let mut sample_phi = vec![0.0; n];
            for _ in 0..n_samples {
                if ctx.cancellation.is_cancelled() {
                    break;
                }
                order.clear();
                order.extend(0..n);
                shuffle(&mut rng, &mut order);
                let mut mask = 0u64;
                let mut v_prev = eval(0, payoff, &mut cache, &mut budget)?;
                sample_phi.fill(0.0);
                for &i in &order {
                    let bit = 1u64 << i;
                    mask |= bit;
                    let v_new = eval(mask, payoff, &mut cache, &mut budget)?;
                    sample_phi[i] = v_new - v_prev;
                    v_prev = v_new;
                }
                completed += 1;
                let cf = completed as f64;
                for i in 0..n {
                    let delta = sample_phi[i] - phi[i];
                    phi[i] += delta / cf;
                    let delta2 = sample_phi[i] - phi[i];
                    phi_m2[i] += delta * delta2;
                }
                if let Some(p) = &ctx.progress {
                    #[allow(clippy::cast_precision_loss)]
                    p.report(completed as f64 / n_samples as f64, "shapley");
                }
            }
            if completed == 0 {
                return Err(AttributionError::Cancelled);
            }
            budget.samples = completed;
            let ns = completed as f64;
            let phi2: Vec<f64> = phi_m2.iter().map(|&m2| mc_stderr_of_mean(m2, ns)).collect();
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
            cfg.mode = ShapleyMode::MonteCarlo { n_samples: n_permutations };
            estimate_shapley(players, &cfg, payoff, ctx)
        }
        _ => Err(AttributionError::unsupported("unsupported ShapleyMode variant")),
    }
}

/// Sequential (path-dependent) allocation along a fixed order.
///
/// Returns ordered path contributions (successive marginals) and consecutive
/// pairwise interaction residuals
/// `v(S∪{a,b}) − v(S∪{a}) − v(S∪{b}) + v(S)` along that path.
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
        return Err(AttributionError::invalid_input("sequential order is empty"));
    }
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
    let mut mask = 0u64;
    let mut v_prev = eval(0, payoff, &mut cache, &mut budget)?;
    let mut values = Vec::with_capacity(order.len());
    let mut interactions = Vec::new();
    let mut prev_component: Option<(ComponentId, usize)> = None;
    let mut seen = 0u64;
    for &comp in order {
        let idx = player_index(comp).ok_or(AttributionError::UnknownPlayer)?;
        if idx >= 64 {
            return Err(AttributionError::SizeLimit {
                kind: "components",
                requested: idx + 1,
                max: 64,
            });
        }
        let bit = 1u64 << idx;
        if seen & bit != 0 {
            return Err(AttributionError::invalid_input(
                "sequential allocation order contains duplicate components",
            ));
        }
        seen |= bit;
        let s_mask = mask; // coalition before adding `comp`
        let v_s = v_prev;
        mask |= bit;
        let v_new = eval(mask, payoff, &mut cache, &mut budget)?;
        let marginal = v_new - v_prev;
        values.push(marginal);
        if let Some((a, a_idx)) = prev_component {
            // Interaction residual for consecutive path pair (a, comp):
            // v(S∪{a,b}) - v(S∪{a}) - v(S∪{b}) + v(S), with S before a.
            let bit_a = 1u64 << a_idx;
            let s_before_a = s_mask & !bit_a;
            let v_s_before_a = eval(s_before_a, payoff, &mut cache, &mut budget)?;
            let v_sb = eval(s_before_a | bit, payoff, &mut cache, &mut budget)?;
            let residual = v_new - v_s - v_sb + v_s_before_a;
            interactions.push(InteractionTerm { a, b: comp, value: residual });
        }
        prev_component = Some((comp, idx));
        v_prev = v_new;
        let _ = v_s;
    }

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

/// Bessel-corrected standard error of a Monte Carlo mean, given the running
/// sum of squared deviations from the mean (`m2`, e.g. Welford's `M2`) and the
/// completed sample count `ns`.
///
/// The sample variance is `m2 / (ns - 1)` (Bessel-corrected, unbiased); the SE of
/// the mean is `sqrt(sample_var / ns) == sqrt(m2 / (ns * (ns - 1)))`. Undefined for
/// `ns < 2` (a single draw has no sample variance) — returns `f64::INFINITY`,
/// matching `antecedent_design::ranker::mc_stderr`'s convention for `n < 2`.
fn mc_stderr_of_mean(m2: f64, ns: f64) -> f64 {
    if ns < 2.0 {
        return f64::INFINITY;
    }
    (m2 / (ns * (ns - 1.0))).sqrt()
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

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{CachePolicy, ComponentId, Parallelism};

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
    fn exact_rejects_64_before_payoff_evaluation() {
        struct PanickingPayoff;
        impl CoalitionPayoff for PanickingPayoff {
            fn value(&mut self, _mask: u64) -> Result<f64, AttributionError> {
                panic!("payoff must not be evaluated");
            }
        }
        let players: Vec<_> = (0..64).map(ComponentId::from_raw).collect();
        let mut payoff = PanickingPayoff;
        let cfg = ShapleyConfig::exact().with_exact_override(true);
        let err = estimate_shapley(&players, &cfg, &mut payoff, &ExecutionContext::for_tests(1))
            .unwrap_err();
        assert!(matches!(err, AttributionError::SizeLimit { requested: 64, max: 63, .. }));
    }

    #[test]
    fn monte_carlo_supports_64_players() {
        let players: Vec<_> = (0..64).map(ComponentId::from_raw).collect();
        let mut payoff = AdditivePayoff { weights: vec![1.0; 64] };
        let cfg = ShapleyConfig::monte_carlo(2).with_seed(9);
        let estimate =
            estimate_shapley(&players, &cfg, &mut payoff, &ExecutionContext::for_tests(1)).unwrap();
        assert_eq!(estimate.values.len(), 64);
        assert!((estimate.values.iter().sum::<f64>() - 64.0).abs() <= 1e-12);
    }

    #[test]
    fn monte_carlo_reports_stderr() {
        let players: Vec<_> = (0..4).map(ComponentId::from_raw).collect();
        let mut payoff = AdditivePayoff { weights: vec![1.0, 1.0, 1.0, 1.0] };
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

    /// E1: `mc_stderr_of_mean` against a known, hand-computed Bessel-corrected
    /// reference (not derived from the estimator itself).
    #[test]
    fn monte_carlo_se_matches_bessel_corrected_reference() {
        // Reference sample: s^2 (Bessel-corrected) = 2.5 for [1,2,3,4,5], so the
        // SE of the mean is sqrt(2.5 / 5).
        let draws = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let ns = draws.len() as f64;
        let mean = draws.iter().sum::<f64>() / ns;
        let m2: f64 = draws.iter().map(|x| (x - mean) * (x - mean)).sum();
        let sample_var_ref = m2 / (ns - 1.0); // Bessel-corrected, textbook two-pass.
        assert!((sample_var_ref - 2.5).abs() < 1e-12);
        let se_ref = (sample_var_ref / ns).sqrt();
        assert!((mc_stderr_of_mean(m2, ns) - se_ref).abs() < 1e-12);

        // The pre-fix formula (population variance, no Bessel correction, i.e.
        // `sqrt(m2 / ns) / sqrt(ns)`) would instead give `sqrt(m2 / ns^2)` — a
        // strictly smaller, wrong value. Confirm the fixed helper does not match it.
        let biased_old = (m2 / ns).sqrt() / ns.sqrt();
        assert!(mc_stderr_of_mean(m2, ns) > biased_old);

        // n < 2: sample SE is undefined. The biased pre-fix formula would (wrongly)
        // report exactly 0.0 for a single draw (E[X^2] - E[X]^2 == 0 for one point);
        // the fix must report +inf instead.
        assert!(mc_stderr_of_mean(0.0, 1.0).is_infinite());
    }

    /// E1 end-to-end: with a single Monte Carlo sample, the pre-fix formula
    /// (`sqrt(E[X^2] - E[X]^2) / sqrt(n)`) is exactly `0.0` — a single draw has no
    /// meaningful sample variance. The fix must report `+inf` instead.
    #[test]
    fn monte_carlo_single_sample_reports_infinite_stderr() {
        let players: Vec<_> = (0..3).map(ComponentId::from_raw).collect();
        let mut payoff = AdditivePayoff { weights: vec![1.0, 2.0, 3.0] };
        let cfg = ShapleyConfig::monte_carlo(1).with_seed(1);
        let est =
            estimate_shapley(&players, &cfg, &mut payoff, &ExecutionContext::for_tests(1)).unwrap();
        let stderrs = est.component_mc_stderr.expect("stderr reported");
        for se in stderrs {
            assert!(se.is_infinite(), "expected +inf for a single MC sample, got {se}");
        }
        assert!(est.monte_carlo_stderr.expect("mean stderr").is_infinite());
    }

    /// `into_contributions` must widen CIs using a Student-t critical value at
    /// `n_samples - 1` degrees of freedom, not the fixed asymptotic `z = 1.96` — at a
    /// small sample count the two are materially different.
    #[test]
    fn into_contributions_uses_t_critical_not_fixed_z() {
        let players: Vec<_> = (0..2).map(ComponentId::from_raw).collect();
        let mut payoff = XorPayoff;
        let cfg = ShapleyConfig::monte_carlo(5).with_seed(11);
        let est =
            estimate_shapley(&players, &cfg, &mut payoff, &ExecutionContext::for_tests(1)).unwrap();
        let stderr = est.component_mc_stderr.clone().expect("stderr reported")[0];
        let contribution = est.values[0];
        let n_samples = est.budget.samples;
        let df = (n_samples - 1) as f64;
        let expected_t = antecedent_stats::student_t_ppf(0.975, df);
        assert!(
            (expected_t - 1.96).abs() > 0.05,
            "test setup needs a small-df t/z gap: t={expected_t}"
        );
        let contributions = est.into_contributions();
        let c0 = &contributions[0];
        assert!(
            (c0.ci_high.unwrap() - (contribution + expected_t * stderr)).abs() < 1e-9,
            "ci_high={:?} expected={}",
            c0.ci_high,
            contribution + expected_t * stderr
        );
        assert!(
            (c0.ci_low.unwrap() - (contribution - expected_t * stderr)).abs() < 1e-9,
            "ci_low={:?} expected={}",
            c0.ci_low,
            contribution - expected_t * stderr
        );
    }

    /// E1 end-to-end statistical check: for a 2-player XOR payoff, each player's
    /// per-permutation marginal contribution is Bernoulli(0.5) (the two possible
    /// orderings are equally likely), with known closed-form variance `0.25`. The
    /// reported MC stderr should track `sqrt(0.25 / n_samples)`.
    #[test]
    fn monte_carlo_stderr_tracks_known_bernoulli_variance() {
        let players: Vec<_> = (0..2).map(ComponentId::from_raw).collect();
        let mut payoff = XorPayoff;
        let n_samples: usize = 2_000;
        let cfg = ShapleyConfig::monte_carlo(n_samples).with_seed(3);
        let est =
            estimate_shapley(&players, &cfg, &mut payoff, &ExecutionContext::for_tests(1)).unwrap();
        let expected_se = (0.25 / n_samples as f64).sqrt();
        for se in est.component_mc_stderr.expect("stderr reported") {
            let ratio = se / expected_se;
            assert!((0.7..1.3).contains(&ratio), "se={se} expected≈{expected_se} ratio={ratio}");
        }
    }

    #[test]
    fn sequential_interactions_zero_for_additive() {
        let players: Vec<_> = (0..3).map(ComponentId::from_raw).collect();
        let mut payoff = AdditivePayoff { weights: vec![1.0, 2.0, 3.0] };
        let ctx = ExecutionContext::for_tests(1);
        let est = sequential_allocate(
            &players,
            &|c| players.iter().position(|&p| p == c),
            &mut payoff,
            &ctx,
        )
        .unwrap();
        assert!((est.values[0] - 1.0).abs() < 1e-12);
        assert!((est.values[1] - 2.0).abs() < 1e-12);
        assert!((est.values[2] - 3.0).abs() < 1e-12);
        assert_eq!(est.interactions.len(), 2);
        for term in &est.interactions {
            assert!(term.value.abs() < 1e-12, "residual={}", term.value);
        }
    }

    #[test]
    fn sequential_rejects_duplicate_player() {
        let players: Vec<_> = (0..2).map(ComponentId::from_raw).collect();
        let order = [players[0], players[0]];
        let mut payoff = AdditivePayoff { weights: vec![1.0, 2.0] };
        let err = sequential_allocate(
            &order,
            &|c| players.iter().position(|&p| p == c),
            &mut payoff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert!(matches!(err, AttributionError::InvalidInput { .. }));
    }

    #[test]
    fn sequential_is_efficient_for_every_three_player_permutation() {
        struct InteractivePayoff;
        impl CoalitionPayoff for InteractivePayoff {
            fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
                let additive = f64::from((mask & 0b001).count_ones())
                    + 2.0 * f64::from((mask & 0b010).count_ones())
                    + 3.0 * f64::from((mask & 0b100).count_ones());
                let interaction = if mask & 0b011 == 0b011 { 4.0 } else { 0.0 };
                Ok(additive + interaction)
            }
        }
        let players: Vec<_> = (0..3).map(ComponentId::from_raw).collect();
        let permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
        for permutation in permutations {
            let order: Vec<_> = permutation.iter().map(|&i| players[i]).collect();
            let estimate = sequential_allocate(
                &order,
                &|c| players.iter().position(|&p| p == c),
                &mut InteractivePayoff,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
            assert!((estimate.values.iter().sum::<f64>() - 10.0).abs() <= 1e-12);
        }
    }

    struct XorPayoff;

    impl CoalitionPayoff for XorPayoff {
        fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
            // Non-additive: value is 1 iff both bit0 and bit1 are set.
            Ok(f64::from((mask & 0b11) == 0b11))
        }
    }

    #[test]
    fn sequential_records_nonzero_interaction_for_xor() {
        let players: Vec<_> = (0..2).map(ComponentId::from_raw).collect();
        let mut payoff = XorPayoff;
        let ctx = ExecutionContext::for_tests(1);
        let est = sequential_allocate(
            &players,
            &|c| players.iter().position(|&p| p == c),
            &mut payoff,
            &ctx,
        )
        .unwrap();
        assert_eq!(est.interactions.len(), 1);
        assert!((est.interactions[0].value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn exact_shapley_efficiency_on_interactive_game() {
        // v(S) = |S| + 1{both bit0 and bit1}: non-additive interaction.
        struct Interactive;
        impl CoalitionPayoff for Interactive {
            fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
                let size = f64::from((mask & 0b111).count_ones());
                let bonus = f64::from((mask & 0b11) == 0b11);
                Ok(size + bonus)
            }
        }
        let players: Vec<_> = (0..3).map(ComponentId::from_raw).collect();
        let mut payoff = Interactive;
        let v_empty = payoff.value(0).unwrap();
        let v_full = payoff.value(0b111).unwrap();
        let cfg = ShapleyConfig::exact();
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.cache_policy = CachePolicy::enabled(Some(1_000_000));
        let est = estimate_shapley(&players, &cfg, &mut payoff, &ctx).unwrap();
        let sum: f64 = est.values.iter().sum();
        assert!(
            (sum - (v_full - v_empty)).abs() < 1e-9,
            "efficiency: sum={sum} v(N)-v(∅)={}",
            v_full - v_empty
        );
    }
}
