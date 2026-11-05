//! Decision value of further choice studies: the expected value of sample information (EVSI) of
//! a candidate number of additional choices about a Dirichlet posterior of option shares, net of
//! cost.
//!
//! The posterior comes from the choice inference ([`antecedent_estimate`'s
//! `choice_inference::choice_posterior` on the qualified prior]); this module only values the
//! next study. The terminal decision picks the option with the largest `value_k * E[share_k]`,
//! the Dirichlet-multinomial predictive is enumerated exactly up to a bound, and beyond it a
//! Monte Carlo over the predictive carries its standard error. For two options the exact EVSI
//! is the scalar binomial EVSI the decision engine's `BinomialSignal` evaluates.
//!
//! The application owns the candidate designs, the acquisition of observations and the
//! authorization of fieldwork.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalRng;
use antecedent_kernels::rng::sample_gamma;
use antecedent_stats::special::ln_gamma;

/// Largest outcome space the exact EVSI enumerates.
pub const MAX_EXACT_COMPOSITIONS: u64 = 250_000;

/// Monte Carlo draws used beyond the enumeration cap.
pub const MONTE_CARLO_DRAWS: usize = 20_000;

/// Largest future study the Monte Carlo evaluates.
pub const MAX_MONTE_CARLO_STUDY: u32 = 10_000;

/// Why a ranking request is refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ChoiceEvsiError {
    /// Values or candidates outside their domain.
    #[error("invalid ranking request: {0}")]
    Invalid(String),
    /// The caller cancelled the requested ranking.
    #[error("choice ranking cancelled")]
    Cancelled,
    /// The ranking exceeds a work or admission limit.
    #[error("choice ranking resource limit: {0}")]
    Resource(String),
}

impl ChoiceEvsiError {
    /// Registered refusal code (`parity/reason_codes.toml`).
    #[must_use]
    pub const fn registered_code(&self) -> &'static str {
        match self {
            Self::Invalid(_) => "choice_evidence_invalid",
            Self::Cancelled => "execution_cancelled",
            Self::Resource(_) => "execution_resource_limit",
        }
    }
}

/// A candidate additional study.
#[derive(Clone, Debug, PartialEq)]
pub struct StudyCandidate {
    /// Label.
    pub label: String,
    /// Choices it would observe.
    pub observations: u32,
    /// Cost in the same units as the option values.
    pub cost: f64,
}

/// How an EVSI was computed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvsiMethod {
    /// Sum over every possible outcome.
    Exact {
        /// Outcomes summed.
        compositions: u64,
    },
    /// Exhaustive outcomes with deterministic hierarchical posterior quadrature.
    HierarchicalQuadrature {
        /// Outcome count vectors summed.
        compositions: u64,
        /// Largest reported posterior-summary change across refits (not a coverage bound).
        max_reported_change: f64,
        /// Absolute deviation of predictive outcome mass from one.
        predictive_mass_error: f64,
    },
    /// Monte Carlo over the predictive distribution.
    MonteCarlo {
        /// Draws.
        draws: usize,
    },
}

/// A ranked candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct StudyRank {
    /// The candidate.
    pub candidate: StudyCandidate,
    /// Expected value of sample information per decision, in option-value units.
    pub evsi: f64,
    /// Monte Carlo standard error; zero for deterministic enumeration, not a numerical error bound.
    pub standard_error: f64,
    /// Method.
    pub method: EvsiMethod,
    /// EVSI net of cost.
    pub net_value: f64,
    /// Whether the order against the next candidate is not established by the method.
    pub rank_uncertain: bool,
    /// Calibration of this study-value estimate; always `not_measured`.
    pub calibration: &'static str,
}

fn compositions(n: u32, k: usize) -> u64 {
    // C(n + k - 1, k - 1), saturating.
    let mut c: u128 = 1;
    for i in 1..k as u128 {
        c = c.saturating_mul(u128::from(n) + i) / i;
        if c > u128::from(u64::MAX) {
            return u64::MAX;
        }
    }
    u64::try_from(c).unwrap_or(u64::MAX)
}

fn best_value(values: &[f64], alpha: &[f64], counts: &[f64], total: f64) -> f64 {
    values
        .iter()
        .zip(alpha.iter().zip(counts))
        .map(|(v, (a, c))| v * ((a + c) / total))
        .fold(f64::MIN, f64::max)
}

struct Enumeration<'a> {
    alpha: &'a [f64],
    values: &'a [f64],
    base: f64,
    post_total: f64,
    counts: Vec<f64>,
    acc: f64,
    cancelled: &'a dyn Fn() -> bool,
}

impl Enumeration<'_> {
    fn go(&mut self, i: usize, left: u32) -> Result<(), ChoiceEvsiError> {
        if (self.cancelled)() {
            return Err(ChoiceEvsiError::Cancelled);
        }
        if i + 1 == self.counts.len() {
            self.counts[i] = f64::from(left);
            let ln_p = self.base
                + self
                    .counts
                    .iter()
                    .zip(self.alpha)
                    .map(|(c, a)| ln_gamma(a + c) - ln_gamma(c + 1.0))
                    .sum::<f64>();
            self.acc +=
                ln_p.exp() * best_value(self.values, self.alpha, &self.counts, self.post_total);
            return Ok(());
        }
        for c in 0..=left {
            self.counts[i] = f64::from(c);
            self.go(i + 1, left - c)?;
        }
        Ok(())
    }
}

fn exact_evsi(
    alpha: &[f64],
    values: &[f64],
    n: u32,
    cancelled: &dyn Fn() -> bool,
) -> Result<f64, ChoiceEvsiError> {
    let a0: f64 = alpha.iter().sum();
    let nf = f64::from(n);
    let base = ln_gamma(nf + 1.0) + ln_gamma(a0)
        - ln_gamma(a0 + nf)
        - alpha.iter().map(|a| ln_gamma(*a)).sum::<f64>();
    let mut walk = Enumeration {
        alpha,
        values,
        base,
        post_total: a0 + nf,
        counts: vec![0.0; alpha.len()],
        acc: 0.0,
        cancelled,
    };
    walk.go(0, n)?;
    Ok(walk.acc)
}

fn monte_carlo_evsi(
    alpha: &[f64],
    values: &[f64],
    n: u32,
    seed: u64,
    cancelled: &dyn Fn() -> bool,
) -> Result<(f64, f64), ChoiceEvsiError> {
    let a0: f64 = alpha.iter().sum();
    let total = a0 + f64::from(n);
    let mut rng = CausalRng::from_seed(seed);
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    let mut theta = vec![0.0; alpha.len()];
    let mut counts = vec![0.0; alpha.len()];
    for _ in 0..MONTE_CARLO_DRAWS {
        if cancelled() {
            return Err(ChoiceEvsiError::Cancelled);
        }
        let mut s = 0.0;
        for (t, a) in theta.iter_mut().zip(alpha) {
            *t = sample_gamma(*a, 1.0, &mut rng);
            s += *t;
        }
        counts.iter_mut().for_each(|c| *c = 0.0);
        for i in 0..n {
            if i % 256 == 0 && cancelled() {
                return Err(ChoiceEvsiError::Cancelled);
            }
            let mut u = rng.next_f64() * s;
            let mut pick = theta.len() - 1;
            for (j, t) in theta.iter().enumerate() {
                if u < *t {
                    pick = j;
                    break;
                }
                u -= t;
            }
            counts[pick] += 1.0;
        }
        let v = best_value(values, alpha, &counts, total);
        sum += v;
        sum_sq += v * v;
    }
    let m = MONTE_CARLO_DRAWS as f64;
    let mean = sum / m;
    let var = ((sum_sq / m) - mean * mean).max(0.0) * m / (m - 1.0);
    Ok((mean, (var / m).sqrt()))
}

/// Rank candidate studies by the expected value of sample information net of cost.
///
/// `alpha` is the Dirichlet posterior of the shares, as the choice inference reports it.
/// The terminal decision picks the option with the largest `value_k * E[share_k]`. The
/// application owns the candidate designs, the acquisition of observations and the authorization
/// of fieldwork; this computes their decision value under the posterior.
///
/// # Errors
/// Values or candidates outside their domain.
pub fn rank_studies(
    alpha: &[f64],
    values: &[f64],
    candidates: &[StudyCandidate],
    seed: u64,
) -> Result<Vec<StudyRank>, ChoiceEvsiError> {
    let ctx = antecedent_core::ExecutionContext::production_default(seed);
    rank_studies_with_context(alpha, values, candidates, seed, &ctx)
}

/// Rank candidate studies with caller-owned admission and cancellation.
///
/// # Errors
/// Invalid inputs, an excessive cumulative work request, cancellation or resource admission failure.
#[allow(
    clippy::too_many_lines,
    reason = "the bounded decision procedure keeps its scientific checks together"
)]
pub fn rank_studies_with_context(
    alpha: &[f64],
    values: &[f64],
    candidates: &[StudyCandidate],
    seed: u64,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<StudyRank>, ChoiceEvsiError> {
    if ctx.cancellation.is_cancelled() {
        return Err(ChoiceEvsiError::Cancelled);
    }
    if !(2..=64).contains(&alpha.len())
        || alpha.iter().any(|a| !a.is_finite() || *a <= 0.0)
        || !alpha.iter().sum::<f64>().is_finite()
    {
        return Err(ChoiceEvsiError::Invalid(
            "two to 64 finite positive posterior shapes are required".into(),
        ));
    }
    if candidates.len() > 1024 {
        return Err(ChoiceEvsiError::Resource("too many candidate studies".into()));
    }
    let bytes = candidates
        .len()
        .checked_mul(std::mem::size_of::<StudyRank>())
        .and_then(|n| n.checked_add(alpha.len().saturating_mul(64)))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| ChoiceEvsiError::Resource("admission size overflow".into()))?;
    let _lease = ctx.acquire_admission(0, bytes).map_err(|e| {
        if ctx.cancellation.is_cancelled() {
            ChoiceEvsiError::Cancelled
        } else {
            ChoiceEvsiError::Resource(e.to_string())
        }
    })?;
    let mut work = 0u64;
    if values.len() != alpha.len() || values.iter().any(|v| !v.is_finite()) {
        return Err(ChoiceEvsiError::Invalid("one finite value per option is required".into()));
    }
    if candidates.is_empty()
        || candidates.iter().any(|c| {
            c.label.trim().is_empty() || c.observations == 0 || !c.cost.is_finite() || c.cost < 0.0
        })
        || candidates
            .iter()
            .map(|c| c.label.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != candidates.len()
    {
        return Err(ChoiceEvsiError::Invalid(
            "candidates need a positive observation count and a finite nonnegative cost".into(),
        ));
    }
    let a0: f64 = alpha.iter().sum();
    let zero = vec![0.0; alpha.len()];
    let now = best_value(values, alpha, &zero, a0);
    let mut rows: Vec<StudyRank> = Vec::with_capacity(candidates.len());
    for c in candidates {
        let comps = compositions(c.observations, alpha.len());
        let candidate_work = if comps <= MAX_EXACT_COMPOSITIONS {
            comps.saturating_mul(alpha.len() as u64)
        } else {
            u64::from(c.observations)
                .saturating_mul(alpha.len() as u64)
                .saturating_mul(MONTE_CARLO_DRAWS as u64)
        };
        work = work.saturating_add(candidate_work);
        if work > 100_000_000 {
            return Err(ChoiceEvsiError::Resource(
                "cumulative candidate work exceeds 100000000 operations".into(),
            ));
        }
        let cancelled = || ctx.cancellation.is_cancelled();
        let (value, se, method) = if comps <= MAX_EXACT_COMPOSITIONS {
            (
                exact_evsi(alpha, values, c.observations, &cancelled)?,
                0.0,
                EvsiMethod::Exact { compositions: comps },
            )
        } else if c.observations <= MAX_MONTE_CARLO_STUDY {
            let (v, se) = monte_carlo_evsi(alpha, values, c.observations, seed, &cancelled)?;
            (v, se, EvsiMethod::MonteCarlo { draws: MONTE_CARLO_DRAWS })
        } else {
            return Err(ChoiceEvsiError::Invalid(format!(
                "candidate `{}` exceeds the supported study size",
                c.label
            )));
        };
        if !value.is_finite() || !se.is_finite() || !now.is_finite() || !(value - now).is_finite() {
            return Err(ChoiceEvsiError::Invalid("ranking produced a non-finite value".into()));
        }
        let evsi = (value - now).max(0.0);
        rows.push(StudyRank {
            candidate: c.clone(),
            evsi,
            standard_error: se,
            method,
            net_value: evsi - c.cost,
            rank_uncertain: false,
            calibration: "not_measured",
        });
    }
    rows.sort_by(|a, b| b.net_value.total_cmp(&a.net_value));
    for i in 0..rows.len().saturating_sub(1) {
        let gap = rows[i].net_value - rows[i + 1].net_value;
        rows[i].rank_uncertain = gap <= 2.0 * (rows[i].standard_error + rows[i + 1].standard_error);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_evsi_matches_the_scalar_binomial_closed_form_for_two_options() {
        // Two options with values (1, 0): EVSI of n choices is the expected posterior-mean max
        // over a beta-binomial outcome minus the current max, computed here independently.
        let (a, b, n) = (3.0_f64, 4.0_f64, 12u32);
        let got = exact_evsi(&[a, b], &[1.0, 0.0], n, &|| false).unwrap();
        let ln_beta = |x: f64, y: f64| ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y);
        let mut want = 0.0;
        for s in 0..=n {
            let s = f64::from(s);
            let ln_choose =
                ln_gamma(f64::from(n) + 1.0) - ln_gamma(s + 1.0) - ln_gamma(f64::from(n) - s + 1.0);
            let p = (ln_choose + ln_beta(a + s, b + f64::from(n) - s) - ln_beta(a, b)).exp();
            // Terminal value of option 0 is its posterior mean; option 1 has value zero.
            want += p * ((a + s) / (a + b + f64::from(n))).max(0.0);
        }
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        // Equal values: the terminal value is the larger posterior mean of the two options.
        let both = exact_evsi(&[a, b], &[1.0, 1.0], n, &|| false).unwrap();
        let mut want_both = 0.0;
        for s in 0..=n {
            let s = f64::from(s);
            let nf = f64::from(n);
            let ln_choose = ln_gamma(nf + 1.0) - ln_gamma(s + 1.0) - ln_gamma(nf - s + 1.0);
            let p = (ln_choose + ln_beta(a + s, b + nf - s) - ln_beta(a, b)).exp();
            want_both += p * ((a + s).max(b + nf - s) / (a + b + nf));
        }
        assert!((both - want_both).abs() < 1e-12, "{both} vs {want_both}");
    }

    #[test]
    fn monte_carlo_agrees_with_exact_within_its_standard_error() {
        let alpha = [4.0, 5.0, 3.0];
        let values = [1.0, 1.1, 0.9];
        let exact = exact_evsi(&alpha, &values, 30, &|| false).unwrap();
        let (mc, se) = monte_carlo_evsi(&alpha, &values, 30, 7, &|| false).unwrap();
        assert!(se > 0.0 && (mc - exact).abs() < 4.0 * se, "{mc} vs {exact} ± {se}");
    }

    #[test]
    fn ranking_prefers_more_information_until_cost_dominates() {
        // Uniform prior plus counts (6, 5, 4).
        let alpha = [7.0, 6.0, 5.0];
        let cands = [
            StudyCandidate { label: "small".into(), observations: 10, cost: 0.0 },
            StudyCandidate { label: "large".into(), observations: 60, cost: 0.0 },
            StudyCandidate { label: "large-costly".into(), observations: 60, cost: 10.0 },
        ];
        let rows = rank_studies(&alpha, &[1.0, 1.0, 1.0], &cands, 1).unwrap();
        assert_eq!(rows[0].candidate.label, "large");
        assert_eq!(rows[2].candidate.label, "large-costly");
        assert!(rows.iter().all(|r| r.evsi >= 0.0));
        // An option that dominates at any plausible share cannot be displaced: no value.
        let flat = rank_studies(&alpha, &[1.0, -100.0, -100.0], &cands, 1).unwrap();
        assert!(flat[0].evsi.abs() < 1e-9);
    }
}

#[cfg(test)]
mod resource_tests {
    use super::*;

    #[test]
    fn ranking_refuses_invalid_shapes_excessive_work_and_cancellation() {
        let candidate = StudyCandidate { label: "study".into(), observations: 10, cost: 0.0 };
        assert!(matches!(
            rank_studies(&[], &[], &[candidate.clone()], 1),
            Err(ChoiceEvsiError::Invalid(_))
        ));
        assert!(matches!(
            rank_studies(&[f64::NAN, 1.0], &[1.0, 1.0], &[candidate.clone()], 1),
            Err(ChoiceEvsiError::Invalid(_))
        ));
        let huge = StudyCandidate { observations: 10_000, ..candidate.clone() };
        assert!(matches!(
            rank_studies(&[1.0; 4], &[1.0; 4], &[huge], 1),
            Err(ChoiceEvsiError::Resource(_))
        ));
        let ctx = antecedent_core::ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        assert!(matches!(
            rank_studies_with_context(&[1.0, 1.0], &[1.0, 1.0], &[candidate], 1, &ctx),
            Err(ChoiceEvsiError::Cancelled)
        ));
    }
}

/// Rank studies of one new respondent who makes one or two repeated choices.
///
/// Each predictive outcome is a complete respondent count vector. The hierarchical posterior is
/// refitted for every outcome, retaining dependence within that respondent. The outcome mass is
/// the multinomial coefficient times the ratio of the refitted and current marginal likelihoods.
/// This is deterministic quadrature, not a calibrated interval or a guarantee of fieldwork value.
///
/// # Errors
/// Invalid inputs, unsupported design, a quadrature or mass check failure, cancellation or admission.
#[allow(
    clippy::too_many_lines,
    reason = "the bounded decision procedure keeps its scientific checks together"
)]
pub fn rank_hierarchical_studies(
    prior: &antecedent_stats::adaptive_inference::HierarchicalChoicePrior,
    respondents: &[Vec<u32>],
    values: &[f64],
    candidates: &[StudyCandidate],
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<StudyRank>, ChoiceEvsiError> {
    use antecedent_stats::adaptive_inference::HierarchicalChoicePosterior;
    let k = prior.shapes.len();
    if !(2..=3).contains(&k)
        || values.len() != k
        || values.iter().any(|v| !v.is_finite())
        || candidates.is_empty()
        || candidates.len() > 64
        || candidates.iter().any(|c| {
            c.label.trim().is_empty()
                || !(1..=2).contains(&c.observations)
                || !c.cost.is_finite()
                || c.cost < 0.0
        })
        || candidates
            .iter()
            .map(|c| c.label.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != candidates.len()
    {
        return Err(ChoiceEvsiError::Invalid("hierarchical candidates require distinct identities, one or two choices by one new respondent and finite values and costs".into()));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(ChoiceEvsiError::Cancelled);
    }
    let cells = respondents
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_mul(k))
        .and_then(|n| n.checked_mul(64))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| ChoiceEvsiError::Resource("hierarchical ranking size overflow".into()))?;
    let _lease = ctx.acquire_admission(0, cells).map_err(|e| {
        if ctx.cancellation.is_cancelled() {
            ChoiceEvsiError::Cancelled
        } else {
            ChoiceEvsiError::Resource(e.to_string())
        }
    })?;
    let cancelled = || ctx.cancellation.is_cancelled();
    let convert = |e: antecedent_stats::adaptive_inference::AdaptiveInferenceError| {
        if cancelled() {
            ChoiceEvsiError::Cancelled
        } else {
            ChoiceEvsiError::Invalid(e.to_string())
        }
    };
    let base = HierarchicalChoicePosterior::new(prior, respondents)
        .map_err(convert)?
        .summary(&cancelled)
        .map_err(convert)?;
    let now = values.iter().zip(&base.mean).map(|(v, p)| v * p).fold(f64::NEG_INFINITY, f64::max);
    let mut cache = std::collections::BTreeMap::new();
    for n in candidates.iter().map(|c| c.observations).collect::<std::collections::BTreeSet<_>>() {
        let mut gain = 0.0;
        let mut mass = 0.0;
        let mut count = 0u64;
        let mut quadrature_change = base.quadrature_error;
        let mut data = respondents.to_vec();
        for first in 0..=n {
            for second in 0..=n - first {
                if k == 2 && first + second != n {
                    continue;
                }
                let mut observed = vec![first, second];
                if k == 3 {
                    observed.push(n - first - second);
                }
                data.push(observed.clone());
                let updated = HierarchicalChoicePosterior::new(prior, &data)
                    .map_err(convert)?
                    .summary(&cancelled)
                    .map_err(convert)?;
                data.pop();
                quadrature_change = quadrature_change.max(updated.quadrature_error);
                let log_mult = ln_gamma(f64::from(n) + 1.0)
                    - observed.iter().map(|c| ln_gamma(f64::from(*c) + 1.0)).sum::<f64>();
                let p = (updated.log_marginal_likelihood - base.log_marginal_likelihood + log_mult)
                    .exp();
                let value = values
                    .iter()
                    .zip(&updated.mean)
                    .map(|(v, p)| v * p)
                    .fold(f64::NEG_INFINITY, f64::max);
                if !p.is_finite() || !value.is_finite() {
                    return Err(ChoiceEvsiError::Invalid(
                        "hierarchical predictive value is non-finite".into(),
                    ));
                }
                mass += p;
                gain += p * value;
                count += 1;
            }
        }
        if !mass.is_finite() || (mass - 1.0).abs() > 1e-5 {
            return Err(ChoiceEvsiError::Invalid(
                "hierarchical predictive outcome mass did not resolve".into(),
            ));
        }
        let evsi = gain - now;
        if !evsi.is_finite() || evsi < -1e-6 {
            return Err(ChoiceEvsiError::Invalid(
                "hierarchical expected information value did not resolve".into(),
            ));
        }
        cache.insert(n, (evsi.max(0.0), count, quadrature_change, (mass - 1.0).abs()));
    }
    let mut rows: Vec<_> = candidates
        .iter()
        .map(|candidate| {
            let (evsi, compositions, max_reported_change, predictive_mass_error) =
                cache[&candidate.observations];
            StudyRank {
                candidate: candidate.clone(),
                evsi,
                standard_error: 0.0,
                method: EvsiMethod::HierarchicalQuadrature {
                    compositions,
                    max_reported_change,
                    predictive_mass_error,
                },
                net_value: evsi - candidate.cost,
                rank_uncertain: false,
                calibration: "not_measured",
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        b.net_value.total_cmp(&a.net_value).then_with(|| a.candidate.label.cmp(&b.candidate.label))
    });
    // The summary convergence check is not a bound on the marginal-likelihood ratio,
    // so it cannot certify any pairwise ordering of distinct quadrature values.
    for i in 0..rows.len().saturating_sub(1) {
        rows[i].rank_uncertain = true;
    }
    Ok(rows)
}

#[cfg(test)]
mod hierarchical_reference_tests {
    use super::*;
    use antecedent_stats::adaptive_inference::HierarchicalChoicePrior;

    #[test]
    fn hierarchical_evsi_matches_independent_three_option_grid() {
        let prior = HierarchicalChoicePrior {
            shapes: vec![1.0; 3],
            log_concentration_mean: 5.0_f64.ln(),
            log_concentration_sd: 1.0,
        };
        let respondents = vec![
            vec![2, 1, 0],
            vec![0, 2, 1],
            vec![1, 0, 2],
            vec![1, 1, 1],
            vec![2, 1, 0],
            vec![0, 2, 1],
            vec![1, 0, 2],
        ];
        let candidates = vec![
            StudyCandidate { label: "one".into(), observations: 1, cost: 0.0 },
            StudyCandidate { label: "two".into(), observations: 2, cost: 0.0 },
        ];
        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../../antecedent/tests/oracle_data/hierarchical_choice_evsi_three_reference.json"
        ))
        .unwrap();
        let result = rank_hierarchical_studies(
            &prior,
            &respondents,
            &[1.0, 1.0, 1.0],
            &candidates,
            &antecedent_core::ExecutionContext::for_tests(9),
        )
        .unwrap();
        for row in &result {
            let index = row.candidate.observations as usize - 1;
            assert!((row.evsi - expected["evsi"][index].as_f64().unwrap()).abs() < 2e-5, "{row:?}");
        }
    }

    #[test]
    fn hierarchical_evsi_matches_independent_two_dimensional_quadrature() {
        let prior = HierarchicalChoicePrior {
            shapes: vec![1.0, 1.0],
            log_concentration_mean: 5.0_f64.ln(),
            log_concentration_sd: 1.0,
        };
        let respondents = vec![
            vec![3, 1],
            vec![1, 3],
            vec![2, 2],
            vec![4, 0],
            vec![0, 4],
            vec![3, 1],
            vec![2, 2],
        ];
        let candidates = vec![
            StudyCandidate { label: "one".into(), observations: 1, cost: 0.0 },
            StudyCandidate { label: "two".into(), observations: 2, cost: 0.0 },
        ];
        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../../antecedent/tests/oracle_data/hierarchical_choice_evsi_reference.json"
        ))
        .unwrap();
        let result = rank_hierarchical_studies(
            &prior,
            &respondents,
            &[1.0, 1.0],
            &candidates,
            &antecedent_core::ExecutionContext::for_tests(7),
        )
        .unwrap();
        for row in &result {
            let index = row.candidate.observations as usize - 1;
            assert!((row.evsi - expected["evsi"][index].as_f64().unwrap()).abs() < 2e-6, "{row:?}");
            assert!(matches!(row.method, EvsiMethod::HierarchicalQuadrature { .. }));
        }
        assert_eq!(result[0].candidate.label, "two");
        assert!(result[0].rank_uncertain);
    }
}
