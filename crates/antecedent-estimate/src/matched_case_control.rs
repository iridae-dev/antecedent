//! Conditional odds ratio from matched case-control sets (2.2 E7).
//!
//! # Design and estimand
//!
//! The data are matched sets sampled on the outcome: each set holds `k` cases and
//! `n - k` controls matched on the declared stratum, with a binary exposure. The
//! estimand is the **conditional odds ratio** `theta = exp(beta)`, the common
//! within-set odds ratio of exposure between cases and controls under the model
//! `logit P(case | set i, x) = alpha_i + beta x` for set-specific, unrestricted
//! `alpha_i`. Outcome-dependent sampling changes only the `alpha_i`, so `beta` is
//! the same in the sampled and source populations. Nothing else is identified: a
//! population risk, risk difference or risk ratio needs the outcome prevalence (or
//! selection fractions), which a case-control sample does not carry, and is refused.
//!
//! # Exact conditional likelihood
//!
//! Conditioning set `i` on its size `n`, its number of cases `k` and its exposure
//! total `t` removes `alpha_i`. The number `A` of exposed cases is then
//! noncentral hypergeometric:
//! `P(A = a) = C(k, a) C(n - k, t - a) e^{beta a} / sum_j C(k, j) C(n - k, t - j) e^{beta j}`
//! on `max(0, t - (n - k)) <= a <= min(k, t)`. With `E_i`, `V_i` the mean and
//! variance of `A` at `beta`, the log likelihood `l(beta) = sum_i log P(A_i = a_i)`
//! has score `U(beta) = sum_i (a_i - E_i(beta))` and observed (= expected, an
//! exponential family) information `I(beta) = -U'(beta) = sum_i V_i(beta)`. Since
//! `I > 0` for every informative set, `l` is strictly concave and the maximizer is
//! unique when it exists.
//!
//! A set with no case or no control (singletons included), or whose members all
//! share one exposure (`t = 0` or `t = n`), has a single possible `A`: it
//! contributes `0` to `U` and `I` and is counted, never silently dropped. The
//! estimate is finite iff `sum a_i` lies strictly between `sum lo_i` and
//! `sum hi_i` over the informative sets (with 1:1 matching: at least one pair
//! discordant each way); otherwise it is refused as not finite.
//!
//! With 1:1 matching the informative sets are the discordant pairs, `lo = 0`,
//! `hi = 1`, `U(beta) = b - (b + c) e^beta / (1 + e^beta)` and the solution is the
//! closed form `theta = b / c` (`b` pairs with the case exposed, `c` with the
//! control exposed).
//!
//! # Inference claim: point only
//!
//! No interval is reported. A Wald interval `beta_hat +- z / sqrt(I(beta_hat))`
//! would rest on asymptotics (many informative sets, bounded set size, a fixed
//! `beta`, independent sets given the matching) that this cell neither checks nor
//! measures, so the interval route is closed with a typed refusal until a coverage
//! record exists.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::EstimationError;
use antecedent_core::ExecutionContext;
use std::collections::BTreeMap;

/// The one declared sampling design: matched sets sampled on the outcome.
pub const MATCHED_SAMPLING: &str = "matched_case_control";

/// The one estimand of the cell.
pub const MATCHED_ESTIMAND: &str = "conditional_odds_ratio";

/// Largest `|log odds ratio|` the bracket search will look for before the estimate
/// is declared numerically not finite (an odds ratio beyond `e^60`).
const BETA_LIMIT: f64 = 60.0;

/// Iteration cap of the safeguarded Newton solve (bisection fallback bounds it).
const MAX_ITERATIONS: usize = 500;

/// Counts of the declared matched sets by what they contribute to the likelihood.
///
/// Three classes partition `total`: `outcome_degenerate` (no case or no control,
/// singletons included) is decided first, then `exposure_concordant` (every member
/// exposed or every member unexposed), the rest are `informative`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatchedSetCounts {
    /// Distinct matched sets (strata) in the data.
    pub total: usize,
    /// Sets that contribute to the conditional likelihood.
    pub informative: usize,
    /// Sets with a case and a control whose members all share one exposure value.
    pub exposure_concordant: usize,
    /// Sets with no case or no control (includes sets of one member).
    pub outcome_degenerate: usize,
    /// Sets of exactly one member (a subset of `outcome_degenerate`).
    pub singleton: usize,
}

/// The conditional odds ratio from matched sets, point only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConditionalOddsRatio {
    /// Conditional maximum likelihood estimate of `beta = log(theta)`.
    pub log_odds_ratio: f64,
    /// `exp(log_odds_ratio)`.
    pub odds_ratio: f64,
    /// What every declared set contributed.
    pub counts: MatchedSetCounts,
}

fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

/// Accept only the declared sampling design.
///
/// # Errors
/// `invalid_argument` (`matched_case_control.sampling_design`).
pub fn parse_matched_sampling(name: &str) -> Result<(), EstimationError> {
    if name == MATCHED_SAMPLING {
        Ok(())
    } else {
        Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "matched_case_control.sampling_design",
            &format!(
                "sampling '{name}' is not the declared design; the cell takes matched sets \
                 sampled on the outcome ('{MATCHED_SAMPLING}')"
            ),
        ))
    }
}

/// Accept the conditional odds ratio and refuse every risk-scale estimand.
///
/// # Errors
/// `effect_not_identified` (`matched_case_control.absolute_risk_not_identified`) for a
/// population risk, risk difference, risk ratio or absolute risk: outcome-dependent
/// sampling fixes the case fraction, so these need prevalence or selection
/// information the cell does not take. `route_not_supported`
/// (`matched_case_control.estimand_not_supported`) for any other name.
pub fn parse_matched_estimand(name: &str) -> Result<(), EstimationError> {
    match name {
        MATCHED_ESTIMAND => Ok(()),
        "population_risk" | "absolute_risk" | "risk_difference" | "risk_ratio" => Err(refuse(
            antecedent_core::reason_code!("effect_not_identified"),
            "matched_case_control.absolute_risk_not_identified",
            &format!(
                "'{name}' is not identified from matched sets sampled on the outcome: the case \
                 fraction is fixed by design, so absolute risks need outcome prevalence or \
                 selection information this cell does not take; only the conditional odds \
                 ratio is licensed"
            ),
        )),
        other => Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "matched_case_control.estimand_not_supported",
            &format!("estimand '{other}' is not licensed; only '{MATCHED_ESTIMAND}' is"),
        )),
    }
}

/// The refusal of the closed interval route: the point is kept, no interval exists.
#[must_use]
pub fn refuse_matched_interval() -> EstimationError {
    refuse(
        antecedent_core::reason_code!("cell_not_licensed"),
        "matched_case_control.interval_withheld",
        "no interval is reported: a Wald interval needs asymptotic conditions (many informative \
         sets, bounded set size, a fixed odds ratio) that no coverage record has measured; the \
         point estimate is retained",
    )
}

/// Accumulated informative sets of one `(n, k, t)` type.
struct SetType {
    /// Sets of this type.
    sets: f64,
    /// Sum of the observed exposed-case counts over those sets.
    sum_a: f64,
    /// `(a, ln C(k, a) + ln C(n - k, t - a))` over the support of `A`.
    support: Vec<(f64, f64)>,
}

impl SetType {
    fn new(n: usize, k: usize, t: usize, ln_fact: &[f64]) -> Self {
        let ln_choose = |m: usize, r: usize| ln_fact[m] - ln_fact[r] - ln_fact[m - r];
        let lo = t.saturating_sub(n - k);
        let hi = k.min(t);
        let support =
            (lo..=hi).map(|a| (a as f64, ln_choose(k, a) + ln_choose(n - k, t - a))).collect();
        Self { sets: 0.0, sum_a: 0.0, support }
    }

    /// Mean and variance of `A` at `beta`, by a max-shifted sum of the weights.
    fn moments(&self, beta: f64) -> (f64, f64) {
        let logs: Vec<f64> = self.support.iter().map(|&(a, base)| base + beta * a).collect();
        let shift = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = logs.iter().map(|l| (l - shift).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mean = self.support.iter().zip(&weights).map(|(&(a, _), w)| a * w).sum::<f64>() / total;
        let var = self
            .support
            .iter()
            .zip(&weights)
            .map(|(&(a, _), w)| (a - mean) * (a - mean) * w)
            .sum::<f64>()
            / total;
        (mean, var)
    }
}

/// Score `U(beta) = sum_i (a_i - E_i)` and information `I(beta) = sum_i V_i`.
fn score_information(types: &[SetType], beta: f64) -> (f64, f64) {
    types.iter().fold((0.0, 0.0), |(u, info), ty| {
        let (mean, var) = ty.moments(beta);
        (u + (ty.sum_a - ty.sets * mean), info + ty.sets * var)
    })
}

fn cancelled(ctx: &ExecutionContext) -> Result<(), EstimationError> {
    if ctx.cancellation.is_cancelled() {
        Err(refuse(
            antecedent_core::reason_code!("transport_budget_cancel"),
            "matched_case_control.budget",
            "conditional odds ratio solve cancelled; no estimate is reported, which is not a \
             verdict on the data",
        ))
    } else {
        Ok(())
    }
}

/// Per-stratum tallies while the rows are read.
#[derive(Default)]
struct Tally {
    members: usize,
    cases: usize,
    exposed: usize,
    exposed_cases: usize,
}

#[allow(clippy::float_cmp)] // exact 0/1 coding is the contract; anything else is refused
fn binary(value: f64) -> Option<bool> {
    if value == 0.0 {
        Some(false)
    } else if value == 1.0 {
        Some(true)
    } else {
        None
    }
}

/// The sets classified by what they contribute, with the informative ones grouped by type.
struct Classified {
    counts: MatchedSetCounts,
    types: Vec<SetType>,
    /// Observed exposed cases, and the least and most the sets allow, summed over the
    /// informative sets: the estimate is finite iff the first is strictly between the others.
    sum_a: usize,
    sum_lo: usize,
    sum_hi: usize,
}

fn classify(
    stratum: &[String],
    case: &[f64],
    exposed: &[f64],
) -> Result<Classified, EstimationError> {
    let invalid = |message: &str| {
        refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "matched_case_control.invalid_data",
            message,
        )
    };
    if stratum.is_empty() || stratum.len() != case.len() || stratum.len() != exposed.len() {
        return Err(invalid("stratum, case and exposed must be non-empty and of equal length"));
    }
    let mut tallies: BTreeMap<&str, Tally> = BTreeMap::new();
    for ((label, &y), &x) in stratum.iter().zip(case).zip(exposed) {
        let (Some(y), Some(x)) = (binary(y), binary(x)) else {
            return Err(invalid("case and exposed must be exactly 0 or 1 (no missing values)"));
        };
        let tally = tallies.entry(label.as_str()).or_default();
        tally.members += 1;
        tally.cases += usize::from(y);
        tally.exposed += usize::from(x);
        tally.exposed_cases += usize::from(x && y);
    }
    let max_members = tallies.values().map(|t| t.members).max().unwrap_or(0);
    let mut ln_fact = vec![0.0_f64];
    for m in 1..=max_members {
        ln_fact.push(ln_fact[m - 1] + (m as f64).ln());
    }
    let mut counts = MatchedSetCounts {
        total: tallies.len(),
        informative: 0,
        exposure_concordant: 0,
        outcome_degenerate: 0,
        singleton: 0,
    };
    let mut by_type: BTreeMap<(usize, usize, usize), SetType> = BTreeMap::new();
    let (mut sum_a, mut sum_lo, mut sum_hi) = (0_usize, 0_usize, 0_usize);
    for tally in tallies.values() {
        let (n, k, t) = (tally.members, tally.cases, tally.exposed);
        if n == 1 {
            counts.singleton += 1;
        }
        if k == 0 || k == n {
            counts.outcome_degenerate += 1;
        } else if t == 0 || t == n {
            counts.exposure_concordant += 1;
        } else {
            counts.informative += 1;
            sum_a += tally.exposed_cases;
            sum_lo += t.saturating_sub(n - k);
            sum_hi += k.min(t);
            let ty = by_type.entry((n, k, t)).or_insert_with(|| SetType::new(n, k, t, &ln_fact));
            ty.sets += 1.0;
            ty.sum_a += tally.exposed_cases as f64;
        }
    }
    if counts.informative == 0 {
        return Err(refuse(
            antecedent_core::reason_code!("effect_not_identified"),
            "matched_case_control.no_informative_sets",
            &format!(
                "none of the {} matched sets is informative ({} outcome-degenerate, {} exposure-\
                 concordant): concordant sets carry no information about the odds ratio",
                counts.total, counts.outcome_degenerate, counts.exposure_concordant
            ),
        ));
    }
    Ok(Classified { counts, types: by_type.into_values().collect(), sum_a, sum_lo, sum_hi })
}

fn not_finite(informative: usize, which: &str) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("route_not_supported"),
        "matched_case_control.estimate_not_finite",
        &format!(
            "the conditional likelihood is monotone in the log odds ratio over the {informative} \
             informative sets ({which}): the maximum likelihood estimate does not exist"
        ),
    )
}

/// Root of the monotone score: bracket it, then a safeguarded Newton iteration.
fn solve(
    types: &[SetType],
    informative: usize,
    ctx: &ExecutionContext,
) -> Result<f64, EstimationError> {
    let (mut lo, mut hi) = (-1.0_f64, 1.0_f64);
    while score_information(types, lo).0 <= 0.0 {
        lo *= 2.0;
        if lo < -BETA_LIMIT {
            return Err(not_finite(informative, "the estimate is below e^-60, numerically zero"));
        }
    }
    while score_information(types, hi).0 >= 0.0 {
        hi *= 2.0;
        if hi > BETA_LIMIT {
            return Err(not_finite(
                informative,
                "the estimate is above e^60, numerically infinite",
            ));
        }
    }
    let tolerance = 1e-10 * informative as f64;
    let mut beta = 0.5 * (lo + hi);
    for _ in 0..MAX_ITERATIONS {
        cancelled(ctx)?;
        let (u, info) = score_information(types, beta);
        if u.abs() <= tolerance || hi - lo <= 1e-14 {
            break;
        }
        if u > 0.0 {
            lo = beta;
        } else {
            hi = beta;
        }
        let newton = beta + u / info;
        beta = if newton > lo && newton < hi { newton } else { 0.5 * (lo + hi) };
    }
    Ok(beta)
}

/// The conditional maximum likelihood odds ratio of matched case-control sets.
///
/// `stratum`, `case` (1 = case, 0 = control) and `exposed` (1 = exposed) are parallel
/// rows. The score is monotone, so the root is bracketed and found by a safeguarded
/// Newton iteration (the information supplies the step, bisection the guarantee);
/// `ctx` is observed on entry and once per iteration.
///
/// # Errors
/// `invalid_argument` (`matched_case_control.invalid_data`) for empty or unequal
/// inputs or a value other than exactly 0 or 1; `effect_not_identified`
/// (`matched_case_control.no_informative_sets`) when every set is concordant or
/// outcome-degenerate; `route_not_supported`
/// (`matched_case_control.estimate_not_finite`) when the conditional likelihood has no
/// finite maximizer; `transport_budget_cancel` (`matched_case_control.budget`) when
/// cancelled.
pub fn conditional_odds_ratio(
    stratum: &[String],
    case: &[f64],
    exposed: &[f64],
    ctx: &ExecutionContext,
) -> Result<ConditionalOddsRatio, EstimationError> {
    cancelled(ctx)?;
    let sets = classify(stratum, case, exposed)?;
    let informative = sets.counts.informative;
    if sets.sum_a == sets.sum_lo {
        return Err(not_finite(
            informative,
            "every case is as unexposed as its set allows, so the estimate is 0",
        ));
    }
    if sets.sum_a == sets.sum_hi {
        return Err(not_finite(
            informative,
            "every case is as exposed as its set allows, so the estimate is infinite",
        ));
    }
    let beta = solve(&sets.types, informative, ctx)?;
    Ok(ConditionalOddsRatio { log_odds_ratio: beta, odds_ratio: beta.exp(), counts: sets.counts })
}

#[cfg(test)]
mod tests {
    use super::{SetType, score_information};

    /// One informative 1:2 set type (n = 3, k = 1, t = 1) for the derivative check.
    fn type_with(sets: f64, sum_a: f64) -> SetType {
        let ln_fact = [0.0, 0.0, std::f64::consts::LN_2, (6.0_f64).ln()];
        let mut ty = SetType::new(3, 1, 1, &ln_fact);
        ty.sets = sets;
        ty.sum_a = sum_a;
        ty
    }

    /// The information is minus the derivative of the score (central difference), and the
    /// closed form of the n = 3, k = 1, t = 1 set: P(A = 1) = e^b / (e^b + 2).
    #[test]
    fn information_is_minus_the_score_derivative() {
        let types = [type_with(5.0, 3.0)];
        for beta in [-1.3, 0.0, 0.7, 2.1] {
            let h = 1e-5;
            let derivative = (score_information(&types, beta + h).0
                - score_information(&types, beta - h).0)
                / (2.0 * h);
            let (u, info) = score_information(&types, beta);
            assert!((info + derivative).abs() < 1e-6, "beta {beta}: {info} vs {}", -derivative);
            let p = beta.exp() / (beta.exp() + 2.0);
            assert!((u - (3.0 - 5.0 * p)).abs() < 1e-12, "beta {beta}: score {u}");
            assert!((info - 5.0 * p * (1.0 - p)).abs() < 1e-12, "beta {beta}: info {info}");
        }
    }
}
