//! Exact and approximate Shapley value estimation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::sync::Mutex;

use antecedent_core::{
    AllocationMethod, ComponentId, ExecutionContext, ShapleyConfig, ShapleyMode,
};
use antecedent_kernels::numeric::CompensatedSum;
use antecedent_kernels::shuffle;
use antecedent_stats::{Welford, student_t_ppf};

use crate::change_common::{SHAPLEY_STREAM, attribution_stream};
use crate::coalition::{
    Coalition, CoalitionCache, CoalitionKey, MAX_EXACT_COALITION_PLAYERS, ensure_player_bitset,
};
use crate::error::AttributionError;
use crate::result::{CacheStats, ComponentContribution, ComputeBudget, InteractionTerm};

/// Payoff function `v(S)` for a coalition of players (a [`Coalition`], any width).
pub trait CoalitionPayoff {
    /// Evaluate the value of `coalition`.
    ///
    /// # Errors
    ///
    /// Model / sampling failures.
    fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError>;

    /// An independent evaluator of the same `v(S)`, which [`estimate_shapley`] may run
    /// on worker threads to evaluate several coalitions at once.
    ///
    /// A fork's `value(c)` must be bit-identical to `self.value(c)` for every coalition `c`
    /// regardless of what either evaluated before, i.e. the payoff is a pure function of the
    /// coalition. The default `None` keeps every evaluation on `self`.
    fn fork<'a>(&'a self) -> Option<Box<dyn CoalitionPayoff + Send + 'a>> {
        None
    }

    /// Sampled draws one evaluation of `v(S)` performs (`1` for a closed-form payoff).
    ///
    /// [`estimate_shapley`] multiplies the planned number of coalition evaluations by this
    /// count and refuses a plan above `MAX_COALITION_SAMPLE_BUDGET` before evaluating
    /// anything, so every caller is guarded without having to remember a separate check.
    fn samples_per_evaluation(&self) -> usize {
        1
    }
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
    /// Mean across players of the permutation-sampling standard error (approx modes only).
    pub monte_carlo_stderr: Option<f64>,
    /// Per-player permutation-sampling standard error: the spread of one player's marginal
    /// contribution across the sampled permutations, divided by `√permutations`.
    ///
    /// It quantifies only the error from sampling permutations *given the payoff values*.
    /// It excludes Monte Carlo noise inside each coalition value `v(S)` and, for
    /// data-fitted payoffs, all estimation uncertainty from fitting the mechanisms; an
    /// additive game has zero permutation variance however noisy `v` is. `None` (exact
    /// mode, sequential allocation) means *not quantified*, not zero.
    pub component_mc_stderr: Option<Vec<f64>>,
    /// Numerical efficiency diagnostic of exact mode: `Σφᵢ − (v(N) − v(∅))`, which is zero
    /// in exact arithmetic. `None` when not computed (sampled and sequential modes).
    pub efficiency_residual: Option<f64>,
    /// Cache stats.
    pub cache_stats: CacheStats,
}

impl ShapleyEstimate {
    /// Convert to component contributions.
    ///
    /// `ci_low`/`ci_high` are the *permutation-sampling* interval
    /// `contribution ± t · component_mc_stderr` (see that field for what it excludes); they
    /// are not a confidence interval for the underlying change attribution. The interval
    /// uses a Student-t critical value with `n_samples - 1` degrees of freedom (from
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
/// without override, or [`AttributionError::SizeLimit`] when the component count exceeds the
/// player sanity bound, or the exact-enumeration width (63 players).
pub(crate) fn check_shapley_size(
    n_components: usize,
    config: &ShapleyConfig,
) -> Result<(), AttributionError> {
    ensure_player_bitset(n_components)?;
    // Exact mode enumerates `2^k` coalitions by `u64` index, so `k == 64` is a legal
    // permutation-sampling width and an illegal exact width: `1u64 << 64` does not fit.
    if matches!(config.mode, ShapleyMode::Exact) && n_components > MAX_EXACT_COALITION_PLAYERS {
        return Err(AttributionError::SizeLimit {
            kind: "exact Shapley components",
            requested: n_components,
            max: MAX_EXACT_COALITION_PLAYERS,
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

/// Default ceiling on `planned coalition evaluations × per-coalition sample draws` for the
/// sampled change-attribution payoffs (`distribution_change`, `structure_change`);
/// [`ShapleyConfig::sample_budget`] replaces it per query.
///
/// Rationale: each coalition evaluation on those paths ancestrally samples
/// `n_samples` rows through every mechanism in the model, so total work scales with
/// the product of the two knobs. [`check_shapley_size`] bounds the *player count*
/// but not that product: exact mode caps at 12 components by default yet is
/// overridable, `max_components` defaults to 64 for Monte Carlo, and the Shapley
/// sample count multiplies the per-coalition draw count silently. At roughly
/// 10–100 ns per sampled value, `1e8` planned draws is already seconds-to-minutes
/// of single-core work. Every supported default stays far below the ceiling
/// (exact `k = 12` with the default `2_000` draws/coalition plans
/// `2^12 × 2_000 ≈ 8.2e6`; Monte Carlo at 64 players with `200`
/// permutations plans `200 × 65 × 2_000 = 2.6e7`), while an exact override at
/// `k = 20` (`2^20 × 2_000 ≈ 2.1e9`) or an oversized Monte Carlo schedule is
/// refused up front instead of silently running for hours.
pub(crate) const MAX_COALITION_SAMPLE_BUDGET: u64 = 100_000_000;

/// Fail-closed guard on the planned sampling volume of a coalition game.
///
/// `planned coalition evaluations` counts the *distinct* coalition masks the
/// estimator schedules (exact: `2^k`; Monte Carlo / permutation: `samples × (k+1)`
/// marginal-chain evaluations) — the irreducible minimum with a warm coalition
/// cache. Multiplied by the per-coalition draw count `n_samples_per_eval` and
/// compared against the configured budget (default [`MAX_COALITION_SAMPLE_BUDGET`]). Non-Shapley allocations pass:
/// sequential allocation is `O(k)` evaluations and path-based allocation performs
/// no coalition sampling at all.
///
/// # Errors
///
/// [`AttributionError::SizeLimit`] when the planned volume exceeds the ceiling
/// (mirrors [`check_shapley_size`]'s posture: refuse up front, clearly, rather
/// than start a run that cannot finish in reasonable time).
pub(crate) fn check_coalition_sample_budget(
    n_players: usize,
    allocation: &AllocationMethod,
    n_samples_per_eval: usize,
) -> Result<(), AttributionError> {
    let AllocationMethod::Shapley { approximation } = allocation else {
        return Ok(());
    };
    check_shapley_sample_budget(n_players, approximation, n_samples_per_eval)
}

/// [`check_coalition_sample_budget`] for a Shapley configuration.
pub(crate) fn check_shapley_sample_budget(
    n_players: usize,
    approximation: &ShapleyConfig,
    n_samples_per_eval: usize,
) -> Result<(), AttributionError> {
    let planned: u128 = match approximation.mode {
        // Distinct coalitions. Exact mode rejects `k > MAX_EXACT_COALITION_PLAYERS`
        // in `check_shapley_size`; the shift stays inside the 128-bit width.
        ShapleyMode::Exact => 1u128 << n_players.min(MAX_EXACT_COALITION_PLAYERS + 1),
        ShapleyMode::MonteCarlo { n_samples }
        | ShapleyMode::Permutation { n_permutations: n_samples } => {
            (n_samples as u128).saturating_mul(n_players as u128 + 1)
        }
        // Unknown modes are rejected later by `estimate_shapley`.
        _ => return Ok(()),
    };
    let total = planned.saturating_mul(n_samples_per_eval.max(1) as u128);
    let budget = approximation.sample_budget.unwrap_or(MAX_COALITION_SAMPLE_BUDGET);
    if total > u128::from(budget) {
        return Err(AttributionError::SizeLimit {
            kind: "planned coalition sample draws",
            requested: usize::try_from(total).unwrap_or(usize::MAX),
            max: usize::try_from(budget).unwrap_or(usize::MAX),
        });
    }
    Ok(())
}

/// Largest player count for which exact mode keeps a dense `2^k` value table (`8 · 2^k`
/// bytes, 128 MiB at the limit). Beyond it the walk is lazy and leans on the cache.
const MAX_DENSE_TABLE_PLAYERS: usize = 24;

/// Value of `coalition`, through the semantic cache.
fn evaluate_coalition<P: CoalitionPayoff>(
    coalition: &Coalition,
    payoff: &mut P,
    cache: &mut CoalitionCache,
    budget: &mut ComputeBudget,
) -> Result<f64, AttributionError> {
    let key = CoalitionKey { mask: coalition.clone(), tag: 0 };
    if let Some(v) = cache.get(&key) {
        return Ok(v);
    }
    let v = payoff.value(coalition)?;
    budget.evaluations += 1;
    cache.insert(key, v);
    Ok(v)
}

/// Coalition values the serial estimator is about to request, computed ahead of it on
/// worker threads from forks of the payoff.
///
/// The estimator's own loop is untouched: it still requests every value in its original
/// order through the cache, so budgets, cache statistics, error ordering and the numbers
/// themselves match a serial run exactly. A coalition that was not (or could not be)
/// prefetched falls through to the payoff (or, once it has been forked, to its first fork).
///
/// The payoff is forked once per estimate: the forks borrow the payoff for the rest of the run,
/// keep their memoized state between blocks, and also answer the serial requests, which the
/// fork contract ("bit-identical to the payoff") permits.
struct Prefetched<'p, P: CoalitionPayoff> {
    state: Evaluator<'p, P>,
    known: HashMap<Coalition, f64>,
}

/// Where a [`Prefetched`] payoff evaluates coalitions that were not prefetched.
enum Evaluator<'p, P: CoalitionPayoff> {
    /// The payoff itself; whether it forks is not yet known.
    Own(&'p mut P),
    /// The payoff does not fork; every evaluation stays on it.
    Serial(&'p mut P),
    /// Worker forks of the payoff, at least one.
    Forked(Vec<Mutex<Box<dyn CoalitionPayoff + Send + 'p>>>),
    /// Transient state while the payoff is being replaced by its forks.
    Moved,
}

/// Prefetched values kept at once; the table is dropped when it grows past this.
const MAX_PREFETCHED: usize = 1 << 20;
/// Fewest distinct unevaluated coalitions worth a thread hand-off.
const MIN_PARALLEL_COALITIONS: usize = 8;

impl<'p, P: CoalitionPayoff> Prefetched<'p, P> {
    fn new(inner: &'p mut P) -> Self {
        Self { state: Evaluator::Own(inner), known: HashMap::new() }
    }

    /// Worker threads the estimator may use for coalition values under `ctx`.
    fn threads(ctx: &ExecutionContext) -> usize {
        ctx.parallelism.max_threads.get() as usize
    }

    /// Replace the payoff by `threads` forks of it, or mark it as serial when it cannot fork.
    ///
    /// Whether the payoff can fork is a property of its type, probed once per estimate (a fork
    /// can be costly) and before any value is evaluated out of order.
    fn fork_workers(&mut self, threads: usize) {
        match std::mem::replace(&mut self.state, Evaluator::Moved) {
            Evaluator::Own(inner) => {
                if inner.fork().is_none() {
                    self.state = Evaluator::Serial(inner);
                    return;
                }
                let shared: &'p P = inner;
                let workers: Vec<_> =
                    (0..threads).filter_map(|_| shared.fork()).map(Mutex::new).collect();
                self.state = Evaluator::Forked(workers);
            }
            other => self.state = other,
        }
    }

    /// Evaluate the distinct, not yet known `coalitions` in parallel into [`Self::known`].
    ///
    /// Best effort: a payoff that cannot fork, a cancelled context, or a failing
    /// coalition leaves the affected coalitions to the serial path, which then reports the
    /// failure at the same point a serial run would.
    fn prefetch(
        &mut self,
        coalitions: impl IntoIterator<Item = Coalition>,
        ctx: &ExecutionContext,
    ) {
        let threads = Self::threads(ctx);
        if threads < 2 || ctx.cancellation.is_cancelled() {
            return;
        }
        if matches!(self.state, Evaluator::Serial(_)) {
            return;
        }
        if self.known.len() > MAX_PREFETCHED {
            self.known.clear();
        }
        let mut seen = HashSet::new();
        let todo: Vec<Coalition> = coalitions
            .into_iter()
            .filter(|c| !self.known.contains_key(c) && seen.insert(c.clone()))
            .collect();
        if todo.len() < MIN_PARALLEL_COALITIONS {
            return;
        }
        // The first value is computed on the payoff itself so that its memoized state
        // (fitted mechanisms, baseline law) is built once and then cloned into the forks.
        let mut head_known = false;
        if let Evaluator::Own(inner) = &mut self.state {
            let Ok(head) = inner.value(&todo[0]) else { return };
            self.known.insert(todo[0].clone(), head);
            head_known = true;
        }
        self.fork_workers(threads);
        let Evaluator::Forked(workers) = &self.state else { return };
        let rest = if head_known { &todo[1..] } else { &todo[..] };
        if rest.is_empty() || workers.is_empty() {
            return;
        }
        let chunk_len = rest.len().div_ceil(workers.len().min(rest.len()));
        let evaluated = ctx.map_indexed(rest.chunks(chunk_len).count(), |t, _| {
            let Ok(mut worker) = workers[t].lock() else {
                return Ok(Vec::new());
            };
            let mut out = Vec::new();
            for coalition in rest.chunks(chunk_len).nth(t).unwrap_or(&[]) {
                match worker.value(coalition) {
                    Ok(v) => out.push((coalition.clone(), v)),
                    Err(_) => break,
                }
            }
            Ok::<_, Infallible>(out)
        });
        let Ok(values) = evaluated;
        self.known.extend(values.into_iter().flatten());
    }
}

impl<P: CoalitionPayoff> CoalitionPayoff for Prefetched<'_, P> {
    fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
        if let Some(&v) = self.known.get(coalition) {
            return Ok(v);
        }
        match &mut self.state {
            Evaluator::Own(inner) | Evaluator::Serial(inner) => inner.value(coalition),
            Evaluator::Forked(workers) => match workers.first_mut() {
                Some(worker) => match worker.get_mut() {
                    Ok(worker) => worker.value(coalition),
                    Err(poisoned) => poisoned.into_inner().value(coalition),
                },
                None => Err(AttributionError::unsupported("payoff fork unavailable")),
            },
            Evaluator::Moved => Err(AttributionError::unsupported("payoff fork unavailable")),
        }
    }
}

/// Progress reports emitted over one exact enumeration (cancellation is still polled for
/// every coalition).
const PROGRESS_REPORTS: u64 = 1024;
/// Coalitions per worker thread handed out in one exact-mode prefetch block.
const PREFETCH_BLOCK: u64 = 64;
/// Permutations per worker thread handed out in one Monte Carlo prefetch block.
const PREFETCH_PERMUTATIONS: usize = 8;

/// The `n + 1` prefix coalitions (`∅` through the full set) of every permutation in
/// `orders`, in the order the estimator requests them.
fn permutation_masks(orders: &[usize], n: usize) -> impl Iterator<Item = Coalition> + '_ {
    orders.chunks_exact(n).flat_map(|order| {
        let mut coalition = Coalition::empty();
        std::iter::once(Coalition::empty()).chain(order.iter().map(move |&i| {
            coalition.insert(i);
            coalition.clone()
        }))
    })
}

/// Estimate Shapley values for `players` under `payoff`.
///
/// Uses a semantic [`CoalitionCache`] keyed by coalition mask. Approximate
/// modes always populate `monte_carlo_stderr`.
///
/// # Errors
///
/// Size limits, empty players, or payoff failures.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
#[allow(
    clippy::too_many_lines,
    reason = "exact and permutation-sampled estimators are selected by one match; each arm is a self-contained estimator over the same payoff"
)]
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
    check_shapley_sample_budget(n, config, payoff.samples_per_evaluation())?;
    let mut payoff = Prefetched::new(payoff);
    let payoff = &mut payoff;

    let mut cache = CoalitionCache::from_policy(ctx.cache_policy);
    if matches!(config.mode, ShapleyMode::Exact) && n > MAX_DENSE_TABLE_PLAYERS {
        cache.enable_dense_index(n);
    }
    let mut budget = ComputeBudget::default();

    let eval = evaluate_coalition::<Prefetched<'_, P>>;

    match config.mode {
        ShapleyMode::Exact => {
            let n_coalitions = 1u64 << n;
            budget.exact_coalitions = n_coalitions;
            // Weighted marginals are summed with compensation: each player accumulates up to
            // 2^(n-1) terms of mixed sign.
            let mut phi = vec![CompensatedSum::default(); n];
            let fact = factorial_weights(n);
            let (mut v_empty, mut v_full) = (0.0, 0.0);
            let report_every = (n_coalitions / PROGRESS_REPORTS).max(1);
            if n <= MAX_DENSE_TABLE_PLAYERS {
                // Evaluate every coalition exactly once into a dense table, whatever the
                // cache policy: with the cache off (or saturated) the lazy walk below
                // re-evaluates each `v(S ∪ {i})` and costs up to `2^k (1 + k/2)` payoff calls,
                // which `check_coalition_sample_budget` (2^k) does not account for. The table
                // already holds every value, so nothing is retained in the coalition cache.
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "this branch runs only for n <= MAX_DENSE_TABLE_PLAYERS (24), so masks are below 2^24 and fit usize"
                )]
                let mut table = Vec::with_capacity(n_coalitions as usize);
                let block = PREFETCH_BLOCK * Prefetched::<P>::threads(ctx) as u64;
                for mask in 0..n_coalitions {
                    if mask % block == 0 {
                        payoff.prefetch(
                            (mask..(mask + block).min(n_coalitions)).map(Coalition::from_bits),
                            ctx,
                        );
                    }
                    if ctx.cancellation.is_cancelled() {
                        return Err(AttributionError::Cancelled);
                    }
                    if mask % report_every == 0 {
                        if let Some(p) = &ctx.progress {
                            p.report(mask as f64 / n_coalitions as f64, "shapley");
                        }
                    }
                    table.push(payoff.value(&Coalition::from_bits(mask))?);
                    budget.evaluations += 1;
                }
                for mask in 0..n_coalitions {
                    #[allow(
                        clippy::cast_possible_truncation,
                        reason = "this branch runs only for n <= MAX_DENSE_TABLE_PLAYERS (24), so masks are below 2^24 and fit usize"
                    )]
                    let v_s = table[mask as usize];
                    let s = (mask.count_ones()) as usize;
                    for i in 0..n {
                        let bit = 1u64 << i;
                        if mask & bit != 0 {
                            continue;
                        }
                        #[allow(
                            clippy::cast_possible_truncation,
                            reason = "this branch runs only for n <= MAX_DENSE_TABLE_PLAYERS (24), so masks are below 2^24 and fit usize"
                        )]
                        let v_with_i = table[(mask | bit) as usize];
                        phi[i].add(fact[s] * (v_with_i - v_s));
                    }
                }
                v_empty = table[0];
                v_full = table[table.len() - 1];
            } else {
                for mask in 0..n_coalitions {
                    if ctx.cancellation.is_cancelled() {
                        return Err(AttributionError::Cancelled);
                    }
                    if mask % report_every == 0 {
                        if let Some(p) = &ctx.progress {
                            p.report(mask as f64 / n_coalitions as f64, "shapley");
                        }
                    }
                    let v_s = eval(&Coalition::from_bits(mask), payoff, &mut cache, &mut budget)?;
                    if mask == 0 {
                        v_empty = v_s;
                    }
                    if mask == n_coalitions - 1 {
                        v_full = v_s;
                    }
                    for i in 0..n {
                        let bit = 1u64 << i;
                        if mask & bit != 0 {
                            continue;
                        }
                        let v_si = eval(
                            &Coalition::from_bits(mask | bit),
                            payoff,
                            &mut cache,
                            &mut budget,
                        )?;
                        let s = (mask.count_ones()) as usize;
                        phi[i].add(fact[s] * (v_si - v_s));
                    }
                }
            }
            let phi: Vec<f64> = phi.iter().map(|acc| acc.total()).collect();
            let efficiency_residual = CompensatedSum::of(phi.iter().copied()) - (v_full - v_empty);
            Ok(ShapleyEstimate {
                players: players.to_vec(),
                values: phi,
                interactions: Vec::new(),
                budget,
                monte_carlo_stderr: None,
                component_mc_stderr: None,
                efficiency_residual: Some(efficiency_residual),
                cache_stats: cache.stats(),
            })
        }
        ShapleyMode::MonteCarlo { n_samples }
        | ShapleyMode::Permutation { n_permutations: n_samples } => {
            budget.samples = n_samples as u64;
            let mut rng = attribution_stream(ctx, SHAPLEY_STREAM, config.seed);
            // Per-player Welford accumulators over the sampled permutations (stable when a
            // marginal's mean is large relative to its spread).
            let mut acc = vec![Welford::new(); n];
            let mut completed = 0u64;
            // Permutations of one prefetch block, `n` player indices per sample.
            let block = Prefetched::<P>::threads(ctx).max(1) * PREFETCH_PERMUTATIONS;
            let mut orders: Vec<usize> = Vec::with_capacity(block * n);
            let mut sample_phi = vec![0.0; n];
            let mut next = 0usize;
            'samples: while next < n_samples {
                let in_block = block.min(n_samples - next);
                orders.clear();
                for _ in 0..in_block {
                    let start = orders.len();
                    orders.extend(0..n);
                    shuffle(&mut rng, &mut orders[start..]);
                }
                payoff.prefetch(permutation_masks(&orders, n), ctx);
                for order in orders.chunks_exact(n) {
                    if ctx.cancellation.is_cancelled() {
                        break 'samples;
                    }
                    let mut coalition = Coalition::empty();
                    let mut v_prev = eval(&coalition, payoff, &mut cache, &mut budget)?;
                    sample_phi.fill(0.0);
                    for &i in order {
                        coalition.insert(i);
                        let v_new = eval(&coalition, payoff, &mut cache, &mut budget)?;
                        sample_phi[i] = v_new - v_prev;
                        v_prev = v_new;
                    }
                    completed += 1;
                    for (a, &x) in acc.iter_mut().zip(&sample_phi) {
                        a.push(x);
                    }
                    if let Some(p) = &ctx.progress {
                        p.report(completed as f64 / n_samples as f64, "shapley");
                    }
                }
                next += in_block;
            }
            // Exact mode refuses on cancellation; a permutation estimate stopped early is
            // likewise not the estimate that was asked for, so it is refused rather than
            // returned as `Ok` with a smaller sample count nobody is told to check.
            if completed < n_samples as u64 {
                return Err(AttributionError::Cancelled);
            }
            budget.samples = completed;
            let phi: Vec<f64> = acc.iter().map(Welford::mean).collect();
            let phi2: Vec<f64> = acc.iter().map(Welford::stderr_of_mean).collect();
            let mean_se = phi2.iter().sum::<f64>() / n as f64;
            Ok(ShapleyEstimate {
                players: players.to_vec(),
                values: phi,
                interactions: Vec::new(),
                budget,
                monte_carlo_stderr: Some(mean_se),
                component_mc_stderr: Some(phi2),
                efficiency_residual: None,
                cache_stats: cache.stats(),
            })
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
pub(crate) fn sequential_allocate<P: CoalitionPayoff>(
    order: &[ComponentId],
    player_index: &dyn Fn(ComponentId) -> Option<usize>,
    payoff: &mut P,
    ctx: &ExecutionContext,
) -> Result<ShapleyEstimate, AttributionError> {
    if order.is_empty() {
        return Err(AttributionError::invalid_input("sequential order is empty"));
    }
    let mut cache = CoalitionCache::from_policy(ctx.cache_policy);
    cache.enable_dense_index(order.len());
    let mut budget = ComputeBudget::default();
    let eval = evaluate_coalition::<P>;
    let mut coalition = Coalition::empty();
    let mut v_prev = eval(&coalition, payoff, &mut cache, &mut budget)?;
    let mut values = Vec::with_capacity(order.len());
    let mut interactions = Vec::new();
    let mut prev_component: Option<(ComponentId, usize)> = None;
    for &comp in order {
        let idx = player_index(comp).ok_or(AttributionError::UnknownPlayer)?;
        ensure_player_bitset(idx + 1)?;
        if coalition.contains(idx) {
            return Err(AttributionError::invalid_input(
                "sequential allocation order contains duplicate components",
            ));
        }
        let before = coalition.clone(); // coalition before adding `comp`
        let v_s = v_prev;
        coalition.insert(idx);
        let v_new = eval(&coalition, payoff, &mut cache, &mut budget)?;
        let marginal = v_new - v_prev;
        values.push(marginal);
        if let Some((a, a_idx)) = prev_component {
            // Interaction residual for consecutive path pair (a, comp):
            // v(S∪{a,b}) - v(S∪{a}) - v(S∪{b}) + v(S), with S before a.
            let s_before_a = before.without(a_idx);
            let v_s_before_a = eval(&s_before_a, payoff, &mut cache, &mut budget)?;
            let v_sb = eval(&s_before_a.with(idx), payoff, &mut cache, &mut budget)?;
            let residual = v_new - v_s - v_sb + v_s_before_a;
            interactions.push(InteractionTerm { a, b: comp, value: residual });
        }
        prev_component = Some((comp, idx));
        v_prev = v_new;
    }

    Ok(ShapleyEstimate {
        players: order.to_vec(),
        values,
        interactions,
        budget,
        monte_carlo_stderr: None,
        component_mc_stderr: None,
        efficiency_residual: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{CachePolicy, ComponentId, Parallelism};

    struct AdditivePayoff {
        weights: Vec<f64>,
    }

    impl CoalitionPayoff for AdditivePayoff {
        fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
            let mask = coalition.low_bits();
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
        // 3 players: 2^3 = 8 distinct coalitions, each evaluated exactly once.
        assert_eq!(est.budget.evaluations, 8);
    }

    /// Glove game: players 0 and 1 hold left gloves, player 2 a right glove; a coalition is worth
    /// one pair. Enumerating the six orderings by hand gives Shapley values (1/6, 1/6, 2/3), a
    /// non-additive game where a wrong permutation weighting or a plain marginal-from-empty rule
    /// (which would give 0, 0, 0) cannot pass.
    #[test]
    fn exact_shapley_matches_the_hand_enumerated_glove_game_claims_06() {
        struct Glove;
        impl CoalitionPayoff for Glove {
            fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
                let mask = coalition.low_bits();
                Ok(f64::from(mask & 0b100 != 0 && mask & 0b011 != 0))
            }
        }
        let players: Vec<_> = (0..3).map(ComponentId::from_raw).collect();
        let ctx = ExecutionContext::for_tests(1);
        let est = estimate_shapley(&players, &ShapleyConfig::exact(), &mut Glove, &ctx).unwrap();
        assert!((est.values[0] - 1.0 / 6.0).abs() < 1e-12, "{:?}", est.values);
        assert!((est.values[1] - 1.0 / 6.0).abs() < 1e-12, "{:?}", est.values);
        assert!((est.values[2] - 2.0 / 3.0).abs() < 1e-12, "{:?}", est.values);
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
            fn value(&mut self, _coalition: &Coalition) -> Result<f64, AttributionError> {
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
    fn permutation_matches_monte_carlo_draws() {
        let players: Vec<_> = (0..4).map(ComponentId::from_raw).collect();
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let ctx = ExecutionContext::for_tests(1);
        let mut monte = AdditivePayoff { weights: weights.clone() };
        let mut permutation = AdditivePayoff { weights };
        let sampled = estimate_shapley(
            &players,
            &ShapleyConfig::monte_carlo(30).with_seed(11),
            &mut monte,
            &ctx,
        )
        .unwrap();
        let ordered = estimate_shapley(
            &players,
            &ShapleyConfig::permutation(30).with_seed(11),
            &mut permutation,
            &ctx,
        )
        .unwrap();
        assert_eq!(sampled.values, ordered.values);
        assert_eq!(sampled.component_mc_stderr, ordered.component_mc_stderr);
        assert_eq!(sampled.budget.samples, ordered.budget.samples);
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

    /// The per-player standard error against a known, hand-computed Bessel-corrected
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
        let mut acc = Welford::new();
        draws.iter().for_each(|&x| acc.push(x));
        assert!((acc.stderr_of_mean() - se_ref).abs() < 1e-12);

        // The pre-fix formula (population variance, no Bessel correction, i.e.
        // `sqrt(m2 / ns) / sqrt(ns)`) would instead give `sqrt(m2 / ns^2)` — a
        // strictly smaller, wrong value. Confirm the accumulator does not match it.
        let biased_old = (m2 / ns).sqrt() / ns.sqrt();
        assert!(acc.stderr_of_mean() > biased_old);

        // n < 2: sample SE is undefined and must be +inf, not 0.
        let mut one = Welford::new();
        one.push(3.0);
        assert!(one.stderr_of_mean().is_infinite());
    }

    /// End-to-end: with a single Monte Carlo sample, the pre-fix formula
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

    /// End-to-end statistical check: for a 2-player XOR payoff, each player's
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
    fn coalition_sample_budget_guard_refuses_runaway_plans() {
        // Exact k=20 with the default 2_000 draws/coalition plans ~2.1e9 draws.
        let exact = AllocationMethod::Shapley { approximation: ShapleyConfig::exact() };
        let err = check_coalition_sample_budget(20, &exact, 2_000).unwrap_err();
        assert!(
            matches!(
                err,
                AttributionError::SizeLimit { kind: "planned coalition sample draws", .. }
            ),
            "unexpected error: {err:?}"
        );

        // Oversized Monte Carlo schedule: 1e6 permutations × 65 chain evals × 2_000 draws.
        let mc = AllocationMethod::Shapley { approximation: ShapleyConfig::monte_carlo(1_000_000) };
        assert!(check_coalition_sample_budget(64, &mc, 2_000).is_err());

        // Supported defaults pass with wide margin.
        assert!(check_coalition_sample_budget(12, &exact, 2_000).is_ok());
        let mc_default =
            AllocationMethod::Shapley { approximation: ShapleyConfig::monte_carlo(200) };
        assert!(check_coalition_sample_budget(64, &mc_default, 2_000).is_ok());

        // Non-Shapley allocations never coalition-sample at scale; always pass.
        let seq =
            AllocationMethod::Sequential { order: std::sync::Arc::from(Vec::<ComponentId>::new()) };
        assert!(check_coalition_sample_budget(64, &seq, usize::MAX).is_ok());
    }

    #[test]
    fn estimate_shapley_enforces_the_sample_budget_for_every_caller() {
        struct Heavy;
        impl CoalitionPayoff for Heavy {
            fn value(&mut self, _coalition: &Coalition) -> Result<f64, AttributionError> {
                panic!("a plan beyond the budget must be refused before any evaluation");
            }
            fn samples_per_evaluation(&self) -> usize {
                10_000_000
            }
        }
        let players: Vec<_> = (0..20).map(ComponentId::from_raw).collect();
        let cfg = ShapleyConfig::exact().with_exact_override(true);
        let err = estimate_shapley(&players, &cfg, &mut Heavy, &ExecutionContext::for_tests(1))
            .unwrap_err();
        assert!(matches!(err, AttributionError::SizeLimit { .. }), "{err:?}");
    }

    #[test]
    fn exact_mode_reports_a_vanishing_efficiency_residual() {
        let players: Vec<_> = (0..5).map(ComponentId::from_raw).collect();
        let mut payoff = AdditivePayoff { weights: vec![0.1, 0.2, 0.3, 0.4, 1e6] };
        let est = estimate_shapley(
            &players,
            &ShapleyConfig::exact(),
            &mut payoff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!(est.efficiency_residual.unwrap().abs() < 1e-9);
        assert!(est.cache_stats.entries == 0, "the dense table retains nothing in the cache");
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
            fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
                let mask = coalition.low_bits();
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
        fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
            let mask = coalition.low_bits();
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
            fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
                let mask = coalition.low_bits();
                let size = f64::from((mask & 0b111).count_ones());
                let bonus = f64::from((mask & 0b11) == 0b11);
                Ok(size + bonus)
            }
        }
        let players: Vec<_> = (0..3).map(ComponentId::from_raw).collect();
        let mut payoff = Interactive;
        let v_empty = payoff.value(&Coalition::empty()).unwrap();
        let v_full = payoff.value(&Coalition::from_bits(0b111)).unwrap();
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

    /// Exact mode evaluates each of the `2^k` coalitions exactly once even with the
    /// coalition cache off (the `for_tests` default). The lazy walk it replaced made
    /// `2^k (1 + k/2)` payoff calls there (48 for k = 4), so the sample-budget guard's
    /// `2^k` assumption held only with a warm cache.
    #[test]
    fn exact_mode_evaluates_each_coalition_once_without_a_cache() {
        struct Counting {
            calls: u64,
        }
        impl CoalitionPayoff for Counting {
            fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
                let mask = coalition.low_bits();
                self.calls += 1;
                Ok(f64::from(mask.count_ones()).powi(2))
            }
        }
        let players: Vec<_> = (0..4).map(ComponentId::from_raw).collect();
        let mut payoff = Counting { calls: 0 };
        let ctx = ExecutionContext::for_tests(1);
        let est = estimate_shapley(&players, &ShapleyConfig::exact(), &mut payoff, &ctx).unwrap();
        assert_eq!(payoff.calls, 16);
        assert_eq!(est.budget.evaluations, 16);
        // v(S) = |S|² is symmetric: every player gets v(N)/4 = 16/4 = 4.
        for v in &est.values {
            assert!((v - 4.0).abs() < 1e-12, "φ={v}");
        }
    }

    /// A Monte Carlo run cancelled before all requested permutations complete is refused,
    /// like exact mode, instead of returned as `Ok` with fewer samples than asked for.
    #[test]
    fn monte_carlo_cancellation_is_refused_not_returned_partial() {
        struct CancelAfterFirstChain {
            token: antecedent_core::CancellationToken,
        }
        impl CoalitionPayoff for CancelAfterFirstChain {
            fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
                let mask = coalition.low_bits();
                if mask == 0b1111 {
                    self.token.cancel();
                }
                Ok(f64::from(mask.count_ones()))
            }
        }
        let players: Vec<_> = (0..4).map(ComponentId::from_raw).collect();
        let ctx = ExecutionContext::for_tests(1);
        let mut payoff = CancelAfterFirstChain { token: ctx.cancellation.clone() };
        let cfg = ShapleyConfig::monte_carlo(200).with_seed(7);
        let err = estimate_shapley(&players, &cfg, &mut payoff, &ctx).unwrap_err();
        assert_eq!(err, AttributionError::Cancelled);
    }

    /// A nonlinear game that is a pure function of the mask and can fork. `forked`
    /// counts the coalitions evaluated on forks.
    #[derive(Clone)]
    struct Game {
        weights: Vec<f64>,
        fail_at: Option<u64>,
        is_fork: bool,
        forked: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        forks: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Game {
        fn new(n: usize, fail_at: Option<u64>) -> Self {
            Self {
                weights: (0..n).map(|i| 1.0 + 0.37 * i as f64).collect(),
                fail_at,
                is_fork: false,
                forked: std::sync::Arc::default(),
                forks: std::sync::Arc::default(),
            }
        }
    }

    impl CoalitionPayoff for Game {
        fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
            let mask = coalition.low_bits();
            if self.is_fork {
                self.forked.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if self.fail_at == Some(mask) {
                return Err(AttributionError::invalid_input("game fails at this coalition"));
            }
            let mut total = 0.0;
            let mut size: f64 = 0.0;
            for (i, &w) in self.weights.iter().enumerate() {
                if mask & (1u64 << i) != 0 {
                    total += w;
                    size += 1.0;
                }
            }
            Ok(total * total + (size * 1.3).sin() + f64::from(mask.count_ones() % 3))
        }

        fn fork<'a>(&'a self) -> Option<Box<dyn CoalitionPayoff + Send + 'a>> {
            self.forks.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(Box::new(Self { is_fork: true, ..self.clone() }))
        }
    }

    /// The same game without `fork`: the serial reference.
    struct NoFork(Game);

    impl CoalitionPayoff for NoFork {
        fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
            self.0.value(coalition)
        }
    }

    fn ctx_with_threads(threads: u32, cache: bool) -> ExecutionContext {
        let mut ctx = ExecutionContext::for_tests(11);
        ctx.parallelism = Parallelism::bounded(
            antecedent_core::NonZeroThreadCount::new(threads).expect("positive threads"),
        );
        ctx.cache_policy =
            if cache { CachePolicy::enabled(Some(1_000_000)) } else { CachePolicy::disabled() };
        ctx
    }

    fn bits(est: &ShapleyEstimate) -> (Vec<u64>, Option<Vec<u64>>, u64, u64, u64, u64) {
        (
            est.values.iter().map(|v| v.to_bits()).collect(),
            est.component_mc_stderr.as_ref().map(|s| s.iter().map(|v| v.to_bits()).collect()),
            est.budget.evaluations,
            est.budget.samples,
            est.cache_stats.hits,
            est.cache_stats.misses,
        )
    }

    /// Coalition values computed on worker threads change nothing: values, standard
    /// errors, budgets and cache statistics are bit-identical to the serial run at every
    /// thread count, with the cache on or off, for exact and sampled estimation.
    #[test]
    fn shapley_does_not_depend_on_thread_count() {
        let players: Vec<_> = (0..8).map(ComponentId::from_raw).collect();
        let configs = [
            ShapleyConfig::exact(),
            ShapleyConfig::monte_carlo(150).with_seed(3),
            ShapleyConfig::monte_carlo(7).with_seed(5),
        ];
        for cfg in configs {
            for cache in [false, true] {
                let mut reference = NoFork(Game::new(8, None));
                let serial =
                    estimate_shapley(&players, &cfg, &mut reference, &ctx_with_threads(1, cache))
                        .unwrap();
                for threads in [1u32, 2, 3, 8] {
                    let mut game = Game::new(8, None);
                    let est = estimate_shapley(
                        &players,
                        &cfg,
                        &mut game,
                        &ctx_with_threads(threads, cache),
                    )
                    .unwrap();
                    assert_eq!(bits(&est), bits(&serial), "{cfg:?} cache={cache} t={threads}");
                    let forked = game.forked.load(std::sync::atomic::Ordering::Relaxed);
                    assert_eq!(forked > 0, threads > 1, "t={threads}: forked evaluations {forked}");
                }
            }
        }
    }

    /// A failing coalition surfaces the same error whether or not it was evaluated ahead
    /// of the estimator on a worker.
    #[test]
    fn failing_coalition_reports_the_same_error_in_parallel() {
        let players: Vec<_> = (0..6).map(ComponentId::from_raw).collect();
        for cfg in [ShapleyConfig::exact(), ShapleyConfig::monte_carlo(40).with_seed(2)] {
            let mut serial = NoFork(Game::new(6, Some(0b10_1101)));
            let expected =
                estimate_shapley(&players, &cfg, &mut serial, &ctx_with_threads(1, false))
                    .unwrap_err();
            let mut game = Game::new(6, Some(0b10_1101));
            let err = estimate_shapley(&players, &cfg, &mut game, &ctx_with_threads(4, false))
                .unwrap_err();
            assert_eq!(err, expected);
        }
    }

    /// Permutations are drawn from the context's attribution stream: a given master seed
    /// reproduces, another master seed gives a different sample, and each stays within
    /// its own standard error of the exact value.
    #[test]
    fn sampled_shapley_follows_the_context_seed_and_converges_to_exact() {
        let players: Vec<_> = (0..6).map(ComponentId::from_raw).collect();
        let exact = {
            let mut game = NoFork(Game::new(6, None));
            estimate_shapley(
                &players,
                &ShapleyConfig::exact(),
                &mut game,
                &ctx_with_threads(1, false),
            )
            .unwrap()
        };
        let cfg = ShapleyConfig::monte_carlo(4_000).with_seed(9);
        let run = |master: u64| {
            let mut ctx = ctx_with_threads(1, true);
            ctx.rng = antecedent_core::RngFactory::from_seed(master);
            let mut game = NoFork(Game::new(6, None));
            estimate_shapley(&players, &cfg, &mut game, &ctx).unwrap()
        };
        let first = run(1);
        assert_eq!(bits(&first), bits(&run(1)));
        assert_ne!(bits(&first), bits(&run(2)));
        for master in 1..=5u64 {
            let est = run(master);
            let stderr = est.component_mc_stderr.as_ref().unwrap();
            for ((phi, truth), se) in est.values.iter().zip(&exact.values).zip(stderr) {
                assert!(
                    (phi - truth).abs() <= 4.0 * se,
                    "master {master}: {phi} vs exact {truth} (se {se})"
                );
            }
        }
    }

    /// The payoff is forked once per estimate (a probe plus one fork per worker), however many
    /// prefetch blocks the run needs.
    #[test]
    fn the_payoff_is_forked_once_per_estimate() {
        let players: Vec<_> = (0..8).map(ComponentId::from_raw).collect();
        let mut game = Game::new(8, None);
        let threads = 4u32;
        estimate_shapley(
            &players,
            &ShapleyConfig::exact(),
            &mut game,
            &ctx_with_threads(threads, false),
        )
        .unwrap();
        let forks = game.forks.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(forks, 1 + threads as usize, "one probe and one fork per worker");
    }

    /// An additive game over more players than a machine word: permutation sampling has no
    /// `2^n` structure, so the width is bounded only by the sample budget.
    #[test]
    fn permutation_sampling_runs_beyond_64_players() {
        struct Wide(Vec<f64>);
        impl CoalitionPayoff for Wide {
            fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
                Ok(coalition.players().map(|i| self.0[i]).sum())
            }
        }
        let n = 150;
        let players: Vec<_> = (0..n).map(|i| ComponentId::from_raw(i as u32)).collect();
        let weights: Vec<f64> = (0..n).map(|i| 1.0 + 0.25 * i as f64).collect();
        let mut payoff = Wide(weights.clone());
        let cfg = ShapleyConfig::permutation(6).with_seed(4);
        let est =
            estimate_shapley(&players, &cfg, &mut payoff, &ExecutionContext::for_tests(1)).unwrap();
        for (phi, w) in est.values.iter().zip(&weights) {
            assert!((phi - w).abs() <= 1e-9, "{phi} vs {w}");
        }
        let exact = estimate_shapley(
            &players,
            &ShapleyConfig::exact().with_exact_override(true),
            &mut payoff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert!(matches!(exact, AttributionError::SizeLimit { requested: 150, max: 63, .. }));
    }

    #[test]
    fn the_sample_budget_is_configurable() {
        let cfg = ShapleyConfig::permutation(10);
        assert!(check_shapley_sample_budget(10, &cfg, 1_000).is_ok());
        assert!(check_shapley_sample_budget(10, &cfg.with_sample_budget(1_000), 1_000).is_err());
        assert!(
            check_shapley_sample_budget(10, &cfg.with_sample_budget(u64::MAX), 1 << 40).is_ok()
        );
    }
}
