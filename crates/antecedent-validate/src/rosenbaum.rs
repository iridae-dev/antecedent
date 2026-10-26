//! Rosenbaum sensitivity bounds for matched designs.
//!
//! In a matched design, treatment is as if randomized within matched sets when two units of
//! one set have the same odds of treatment. Rosenbaum's sensitivity model lets those odds
//! differ by at most a factor `Γ ≥ 1` and bounds the randomization test under every
//! assignment law that satisfies it (Rosenbaum 1987; Rosenbaum 2002, *Observational
//! Studies*, ch. 4).
//!
//! * **Matched pairs, Wilcoxon signed rank.** With the `n` non-zero pair differences ranked by
//!   absolute value (average ranks for ties), the statistic `T⁺` is the sum of the ranks of
//!   the positive differences. Under `Γ`, `T⁺` is stochastically at most
//!   `Σ rank_i · B_i` with independent `B_i ~ Bernoulli(Γ / (1 + Γ))`. The upper-bound
//!   p-value is computed **exactly** by a dynamic program over the (doubled, hence integer)
//!   rank sums while the dynamic program stays within [`EXACT_WORK_LIMIT`] updates (about
//!   `pairs × Σ doubled ranks`, i.e. cubic in the pair count), and by the normal approximation
//!   with the tie-corrected variance `p(1 − p) Σ rank_i²` beyond it. Which law was used is
//!   reported: [`SignedRankStatistic::upper_p_value`] returns it and
//!   [`SignedRankSensitivity::exact`] records it.
//! * **Sensitivity value.** [`signed_rank_sensitivity`] reports the upper-bound p-value on a
//!   grid of `Γ` and the break-even `Γ*`: the largest `Γ` at which the upper bound is still
//!   at most `α`.
//! * **Matched sets with several controls.** [`matched_sets_upper_p`] bounds a sum statistic
//!   of within-set scores (the outcomes, or Hodges–Lehmann aligned ranks) by the separable
//!   approximation of Gastwirth, Krieger and Rosenbaum (2000): in each set the worst-case
//!   assignment law puts odds `Γ` on the `a` highest scores, with `a` chosen to maximize the
//!   set's expectation (then its variance), and the statistic is referred to the normal law
//!   with the summed moments.
//! * **Amplification.** [`amplify`] decomposes `Γ` into an unobserved covariate's effect `Λ`
//!   on treatment and `Δ` on the outcome, `Γ = (ΛΔ + 1) / (Λ + Δ)` (Rosenbaum and Silber 2009).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_kernels::norm_sf;

use crate::ValidationError;
use crate::assumption_range::{AssumptionRange, RangePoint, SensitivityAssumption};
use crate::common::admit_cells;
use crate::target_native::RefutationTarget;

/// Largest dynamic-program work (`pairs × Σ doubled ranks` table updates) for which the
/// upper-bound distribution is computed exactly; about 460 untied pairs.
pub const EXACT_WORK_LIMIT: u64 = 100_000_000;

/// Default grid of `Γ` values probed by [`signed_rank_sensitivity`].
pub const DEFAULT_GAMMA_GRID: [f64; 10] = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0];

/// Average ranks of `|values|`, doubled so every rank is an integer.
#[allow(clippy::float_cmp, reason = "exact ties share an average rank")]
fn doubled_abs_ranks(values: &[f64]) -> Vec<u64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].abs().total_cmp(&values[b].abs()));
    let mut ranks = vec![0_u64; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]].abs() == values[order[start]].abs() {
            end += 1;
        }
        // Positions start+1 ..= end share the average rank (start + 1 + end) / 2; doubled.
        let doubled = (start + 1 + end) as u64;
        for &index in &order[start..end] {
            ranks[index] = doubled;
        }
        start = end;
    }
    ranks
}

/// The signed-rank statistic of the non-zero differences.
#[derive(Clone, Debug, PartialEq)]
pub struct SignedRankStatistic {
    /// Non-zero differences used (zeros are dropped).
    pub pairs: usize,
    /// Doubled average ranks of `|d|`.
    doubled_ranks: Vec<u64>,
    /// Doubled `T⁺`.
    doubled_statistic: u64,
}

impl SignedRankStatistic {
    /// Rank the non-zero pair differences (treated minus control).
    ///
    /// # Errors
    ///
    /// A non-finite difference or no non-zero difference.
    pub fn new(differences: &[f64]) -> Result<Self, ValidationError> {
        if differences.iter().any(|d| !d.is_finite()) {
            return Err(ValidationError::NotApplicable {
                message: "matched-pair differences must be finite",
            });
        }
        let nonzero: Vec<f64> = differences.iter().copied().filter(|&d| d != 0.0).collect();
        if nonzero.is_empty() {
            return Err(ValidationError::NotApplicable {
                message: "every matched-pair difference is zero; the signed-rank test is \
                          undefined",
            });
        }
        let doubled_ranks = doubled_abs_ranks(&nonzero);
        let doubled_statistic =
            nonzero.iter().zip(&doubled_ranks).filter(|(d, _)| **d > 0.0).map(|(_, r)| *r).sum();
        Ok(Self { pairs: nonzero.len(), doubled_ranks, doubled_statistic })
    }

    /// `T⁺`, the sum of the ranks of the positive differences.
    #[must_use]
    pub fn statistic(&self) -> f64 {
        self.doubled_statistic as f64 / 2.0
    }

    /// Upper bound on the one-sided p-value (large `T⁺` is evidence of a positive effect)
    /// under hidden bias at most `gamma`. `exact` reports whether the exact law was used.
    ///
    /// # Errors
    ///
    /// `gamma < 1`, cancellation, or an admission refusal for the exact table.
    pub fn upper_p_value(
        &self,
        gamma: f64,
        ctx: Option<&ExecutionContext>,
    ) -> Result<(f64, bool), ValidationError> {
        if !(gamma.is_finite() && gamma >= 1.0) {
            return Err(ValidationError::NotApplicable {
                message: "gamma must be finite and >= 1",
            });
        }
        let p = gamma / (1.0 + gamma);
        if self.exact_work() <= EXACT_WORK_LIMIT {
            return Ok((self.exact_upper(p, ctx)?, true));
        }
        let sum_r: f64 = self.doubled_ranks.iter().map(|&r| r as f64 / 2.0).sum();
        let sum_r2: f64 = self.doubled_ranks.iter().map(|&r| (r as f64 / 2.0).powi(2)).sum();
        let mean = p * sum_r;
        let sd = (p * (1.0 - p) * sum_r2).sqrt();
        Ok((norm_sf((self.statistic() - mean) / sd), false))
    }

    /// Table updates the exact dynamic program performs: `pairs × Σ doubled ranks`.
    fn exact_work(&self) -> u64 {
        let total: u64 = self.doubled_ranks.iter().sum();
        total.saturating_mul(self.pairs as u64)
    }

    fn exact_upper(&self, p: f64, ctx: Option<&ExecutionContext>) -> Result<f64, ValidationError> {
        let total: u64 = self.doubled_ranks.iter().sum();
        let size = usize::try_from(total)
            .ok()
            .and_then(|t| t.checked_add(1))
            .ok_or_else(|| ValidationError::estimation_msg("signed-rank table size overflow"))?;
        let _lease = admit_cells(ctx, 2, size)?;
        let mut dp = vec![0.0_f64; size];
        let mut next = vec![0.0_f64; size];
        dp[0] = 1.0;
        let mut reach = 0_usize;
        for &rank in &self.doubled_ranks {
            if let Some(ctx) = ctx {
                crate::common::check_cancelled(ctx)?;
            }
            let r = usize::try_from(rank).unwrap_or(usize::MAX);
            let new_reach = reach + r;
            next[..=new_reach].fill(0.0);
            for s in 0..=reach {
                let mass = dp[s];
                if mass != 0.0 {
                    next[s] += mass * (1.0 - p);
                    next[s + r] += mass * p;
                }
            }
            std::mem::swap(&mut dp, &mut next);
            reach = new_reach;
        }
        let from = usize::try_from(self.doubled_statistic).unwrap_or(usize::MAX);
        Ok(dp[from.min(size)..].iter().sum::<f64>().min(1.0))
    }
}

/// Upper-bound p-values on a grid of `Γ` and the break-even `Γ*`.
#[derive(Clone, Debug, PartialEq)]
pub struct SignedRankSensitivity {
    /// Non-zero pairs.
    pub pairs: usize,
    /// `T⁺`.
    pub statistic: f64,
    /// `(Γ, upper-bound p-value)` on the probed grid.
    pub grid: Vec<(f64, f64)>,
    /// Largest `Γ` with upper-bound p-value at most `alpha`; `None` when the test does not
    /// reject even at `Γ = 1`.
    pub gamma_break_even: Option<f64>,
    /// Level.
    pub alpha: f64,
    /// Whether the exact law was used.
    pub exact: bool,
}

/// Signed-rank sensitivity analysis of matched-pair differences.
///
/// # Errors
///
/// As [`SignedRankStatistic::new`] and [`SignedRankStatistic::upper_p_value`], or `alpha`
/// outside `(0, 1)`.
pub fn signed_rank_sensitivity(
    differences: &[f64],
    gammas: &[f64],
    alpha: f64,
    ctx: Option<&ExecutionContext>,
) -> Result<SignedRankSensitivity, ValidationError> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(ValidationError::NotApplicable { message: "alpha must lie in (0, 1)" });
    }
    let stat = SignedRankStatistic::new(differences)?;
    let mut grid = Vec::with_capacity(gammas.len());
    let mut exact = true;
    for &gamma in gammas {
        let (p, was_exact) = stat.upper_p_value(gamma, ctx)?;
        exact &= was_exact;
        grid.push((gamma, p));
    }
    let gamma_break_even =
        break_even_gamma(alpha, |gamma| stat.upper_p_value(gamma, ctx).map(|(p, _)| p))?;
    Ok(SignedRankSensitivity {
        pairs: stat.pairs,
        statistic: stat.statistic(),
        grid,
        gamma_break_even,
        alpha,
        exact,
    })
}

/// Largest `Γ ≥ 1` with `upper(Γ) ≤ alpha`, by bisection on `log Γ` (the bound is
/// non-decreasing in `Γ`); `None` when `upper(1) > alpha`.
fn break_even_gamma(
    alpha: f64,
    mut upper: impl FnMut(f64) -> Result<f64, ValidationError>,
) -> Result<Option<f64>, ValidationError> {
    if upper(1.0)? > alpha {
        return Ok(None);
    }
    let mut lo = 1.0_f64;
    let mut hi = 2.0_f64;
    while upper(hi)? <= alpha {
        lo = hi;
        hi *= 2.0;
        if hi > 1e12 {
            return Ok(Some(f64::INFINITY));
        }
    }
    for _ in 0..60 {
        let mid = (lo * hi).sqrt();
        if upper(mid)? <= alpha {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi / lo < 1.0 + 1e-10 {
            break;
        }
    }
    Ok(Some(lo))
}

/// One matched set: outcomes and the index of the treated unit.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchedSet {
    /// Outcomes of every unit in the set (at least two).
    pub outcomes: Vec<f64>,
    /// Index of the treated unit in [`Self::outcomes`].
    pub treated: usize,
}

/// Scores the matched-set statistic sums.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MatchedSetScore {
    /// The outcomes themselves.
    Outcomes,
    /// Hodges–Lehmann aligned ranks: outcomes minus their set mean, ranked across all sets.
    AlignedRanks,
}

/// Worst-case moments of one set's score under `Γ` (separable approximation).
fn worst_case_set_moments(scores: &[f64], gamma: f64) -> (f64, f64) {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let n = sorted.len();
    let mut best = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for a in 1..n {
        let denom = a as f64 * gamma + (n - a) as f64;
        let (mut s1, mut s2) = (0.0, 0.0);
        for (i, q) in sorted.iter().enumerate() {
            let w = if i < a { gamma } else { 1.0 };
            s1 += w * q;
            s2 += w * q * q;
        }
        let mean = s1 / denom;
        let var = (s2 / denom - mean * mean).max(0.0);
        // Largest expectation; ties broken by the larger variance.
        if mean > best.0 + 1e-12 * mean.abs().max(1.0)
            || ((mean - best.0).abs() <= 1e-12 * mean.abs().max(1.0) && var > best.1)
        {
            best = (mean, var);
        }
    }
    best
}

/// Upper-bound one-sided p-value of the matched-set sum statistic under `Γ`, with the
/// observed statistic and its worst-case expectation and variance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchedSetBound {
    /// `Γ`.
    pub gamma: f64,
    /// Observed sum of the treated units' scores.
    pub statistic: f64,
    /// Worst-case expectation.
    pub expectation: f64,
    /// Worst-case variance.
    pub variance: f64,
    /// Upper-bound p-value (normal approximation).
    pub p_value: f64,
}

/// Bound the matched-set statistic under hidden bias at most `gamma`.
///
/// # Errors
///
/// A set with fewer than two units, a treated index out of range, non-finite outcomes,
/// `gamma < 1`, or zero worst-case variance.
pub fn matched_sets_upper_p(
    sets: &[MatchedSet],
    score: MatchedSetScore,
    gamma: f64,
) -> Result<MatchedSetBound, ValidationError> {
    if !(gamma.is_finite() && gamma >= 1.0) {
        return Err(ValidationError::NotApplicable { message: "gamma must be finite and >= 1" });
    }
    if sets.is_empty()
        || sets.iter().any(|s| s.outcomes.len() < 2 || s.treated >= s.outcomes.len())
        || sets.iter().flat_map(|s| &s.outcomes).any(|y| !y.is_finite())
    {
        return Err(ValidationError::NotApplicable {
            message: "matched sets need at least two finite outcomes each and a treated index \
                      inside the set",
        });
    }
    let scored = set_scores(sets, score);
    let (mut statistic, mut expectation, mut variance) = (0.0, 0.0, 0.0);
    for (set, scores) in sets.iter().zip(&scored) {
        statistic += scores[set.treated];
        let (m, v) = worst_case_set_moments(scores, gamma);
        expectation += m;
        variance += v;
    }
    if variance.is_nan() || variance <= 0.0 {
        return Err(ValidationError::NotApplicable {
            message: "every matched set has constant scores; the test is undefined",
        });
    }
    let p_value = norm_sf((statistic - expectation) / variance.sqrt());
    Ok(MatchedSetBound { gamma, statistic, expectation, variance, p_value })
}

#[allow(clippy::float_cmp, reason = "exact ties share an average rank")]
fn set_scores(sets: &[MatchedSet], score: MatchedSetScore) -> Vec<Vec<f64>> {
    match score {
        MatchedSetScore::Outcomes => sets.iter().map(|s| s.outcomes.clone()).collect(),
        MatchedSetScore::AlignedRanks => {
            let aligned: Vec<Vec<f64>> = sets
                .iter()
                .map(|s| {
                    let mean = s.outcomes.iter().sum::<f64>() / s.outcomes.len() as f64;
                    s.outcomes.iter().map(|y| y - mean).collect()
                })
                .collect();
            let flat: Vec<f64> = aligned.iter().flatten().copied().collect();
            let mut order: Vec<usize> = (0..flat.len()).collect();
            order.sort_by(|&a, &b| flat[a].total_cmp(&flat[b]));
            let mut ranks = vec![0.0; flat.len()];
            let mut start = 0;
            while start < order.len() {
                let mut end = start + 1;
                while end < order.len() && flat[order[end]] == flat[order[start]] {
                    end += 1;
                }
                let average = (start + 1 + end) as f64 / 2.0;
                for &index in &order[start..end] {
                    ranks[index] = average;
                }
                start = end;
            }
            let mut offset = 0;
            aligned
                .iter()
                .map(|set| {
                    let out = ranks[offset..offset + set.len()].to_vec();
                    offset += set.len();
                    out
                })
                .collect()
        }
    }
}

/// Break-even `Γ` of the matched-set statistic at level `alpha`.
///
/// # Errors
///
/// As [`matched_sets_upper_p`].
pub fn matched_sets_break_even(
    sets: &[MatchedSet],
    score: MatchedSetScore,
    alpha: f64,
) -> Result<Option<f64>, ValidationError> {
    break_even_gamma(alpha, |gamma| matched_sets_upper_p(sets, score, gamma).map(|b| b.p_value))
}

/// The outcome effect `Δ` an unobserved covariate with treatment effect `lambda` needs to
/// produce hidden bias `gamma`: `Δ = (ΓΛ − 1) / (Λ − Γ)`, defined for `Λ > Γ`.
#[must_use]
pub fn amplify(gamma: f64, lambda: f64) -> Option<f64> {
    (gamma >= 1.0 && lambda > gamma).then(|| (gamma * lambda - 1.0) / (lambda - gamma))
}

/// `Γ = (ΛΔ + 1) / (Λ + Δ)`.
#[must_use]
pub fn gamma_of(lambda: f64, delta: f64) -> f64 {
    (lambda * delta + 1.0) / (lambda + delta)
}

/// Signed-rank sensitivity of matched-pair differences as an [`AssumptionRange`] on `target`.
///
/// The tested range is `gammas`; at each `Γ` the target is the upper-bound p-value. The
/// break-even (significance) value is `Γ*`.
///
/// # Errors
///
/// As [`signed_rank_sensitivity`].
pub fn rosenbaum_range(
    target: RefutationTarget,
    differences: &[f64],
    gammas: &[f64],
    alpha: f64,
    ctx: Option<&ExecutionContext>,
) -> Result<(SignedRankSensitivity, AssumptionRange), ValidationError> {
    let analysis = signed_rank_sensitivity(differences, gammas, alpha, ctx)?;
    let tested = analysis
        .grid
        .iter()
        .map(|&(gamma, p)| RangePoint { parameter: gamma, target: p, test: Some(p) })
        .collect();
    let range = AssumptionRange::new(
        target,
        SensitivityAssumption::HiddenBiasInMatchedAssignment,
        "gamma",
        1.0,
        tested,
        None,
        analysis.gamma_break_even,
    );
    Ok((analysis, range))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upper-bound p-value by brute force over all `2^n` sign vectors.
    fn brute_force(differences: &[f64], gamma: f64) -> f64 {
        let nonzero: Vec<f64> = differences.iter().copied().filter(|d| *d != 0.0).collect();
        let ranks = doubled_abs_ranks(&nonzero);
        let observed: u64 =
            nonzero.iter().zip(&ranks).filter(|(d, _)| **d > 0.0).map(|(_, r)| r).sum();
        let p = gamma / (1.0 + gamma);
        let n = nonzero.len();
        let mut total = 0.0;
        for mask in 0_u32..(1 << n) {
            let mut sum = 0;
            let mut prob = 1.0;
            for (i, r) in ranks.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    sum += r;
                    prob *= p;
                } else {
                    prob *= 1.0 - p;
                }
            }
            if sum >= observed {
                total += prob;
            }
        }
        total
    }

    #[test]
    fn exact_law_matches_enumeration_with_ties_and_zeros() {
        let d = [1.5, -0.5, 2.0, 2.0, 0.0, 3.1, -1.5, 0.7, 4.2, 2.6];
        let stat = SignedRankStatistic::new(&d).unwrap();
        assert_eq!(stat.pairs, 9);
        for gamma in [1.0, 1.5, 2.0, 3.7] {
            let (p, exact) = stat.upper_p_value(gamma, None).unwrap();
            assert!(exact);
            assert!((p - brute_force(&d, gamma)).abs() < 1e-12, "gamma {gamma}");
        }
    }

    #[test]
    fn all_positive_pairs_have_the_closed_form_bound() {
        let d = [1.0, 2.0, 3.0, 4.0, 5.0];
        let stat = SignedRankStatistic::new(&d).unwrap();
        assert!((stat.upper_p_value(1.0, None).unwrap().0 - 1.0 / 32.0).abs() < 1e-15);
        let gamma = 2.0_f64;
        let expected = (gamma / (1.0 + gamma)).powi(5);
        assert!((stat.upper_p_value(gamma, None).unwrap().0 - expected).abs() < 1e-15);
    }

    #[test]
    fn break_even_gamma_is_where_the_bound_crosses_alpha() {
        let d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -0.5, 9.0];
        let s = signed_rank_sensitivity(&d, &DEFAULT_GAMMA_GRID, 0.05, None).unwrap();
        let g = s.gamma_break_even.unwrap();
        let stat = SignedRankStatistic::new(&d).unwrap();
        assert!(stat.upper_p_value(g, None).unwrap().0 <= 0.05);
        assert!(stat.upper_p_value(g * 1.001, None).unwrap().0 > 0.05);
        // The bound grows with Gamma.
        assert!(s.grid.windows(2).all(|w| w[0].1 <= w[1].1));
    }

    #[test]
    fn amplification_reproduces_rosenbaum_and_silber() {
        // Gamma 1.25 is (Lambda, Delta) = (2, 2); Gamma 2 is (3, 5).
        assert!((amplify(1.25, 2.0).unwrap() - 2.0).abs() < 1e-12);
        assert!((amplify(2.0, 3.0).unwrap() - 5.0).abs() < 1e-12);
        assert!((gamma_of(3.0, 5.0) - 2.0).abs() < 1e-12);
        assert!(amplify(2.0, 1.5).is_none());
    }

    #[test]
    fn worst_case_set_mean_is_the_maximum_over_hidden_bias_vectors() {
        // For one set, enumerate every binary unobserved covariate u (not all equal): the
        // treated unit is i with probability Gamma^u_i / sum_j Gamma^u_j.
        let scores = [3.0, -1.0, 0.5, 2.0];
        let gamma = 2.5;
        let mut best = f64::NEG_INFINITY;
        for mask in 1_u32..((1 << scores.len()) - 1) {
            let weights: Vec<f64> =
                (0..scores.len()).map(|i| if mask & (1 << i) != 0 { gamma } else { 1.0 }).collect();
            let total: f64 = weights.iter().sum();
            let mean: f64 = weights.iter().zip(&scores).map(|(w, q)| w * q).sum::<f64>() / total;
            best = best.max(mean);
        }
        let (mean, _) = worst_case_set_moments(&scores, gamma);
        assert!((mean - best).abs() < 1e-12);
    }

    #[test]
    fn matched_sets_at_gamma_one_use_randomization_moments() {
        let sets = vec![
            MatchedSet { outcomes: vec![5.0, 1.0, 2.0], treated: 0 },
            MatchedSet { outcomes: vec![3.0, 4.0, 0.0], treated: 1 },
            MatchedSet { outcomes: vec![2.0, 2.5], treated: 1 },
        ];
        let b = matched_sets_upper_p(&sets, MatchedSetScore::Outcomes, 1.0).unwrap();
        let mean: f64 =
            sets.iter().map(|s| s.outcomes.iter().sum::<f64>() / s.outcomes.len() as f64).sum();
        assert!((b.expectation - mean).abs() < 1e-12);
        assert!((b.statistic - 11.5).abs() < 1e-12);
        let bigger = matched_sets_upper_p(&sets, MatchedSetScore::AlignedRanks, 3.0).unwrap();
        let aligned = matched_sets_upper_p(&sets, MatchedSetScore::AlignedRanks, 1.0).unwrap();
        assert!(bigger.expectation >= aligned.expectation);
        assert!(bigger.p_value.is_finite() && aligned.p_value.is_finite());
    }
}
