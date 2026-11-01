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
}

impl ChoiceEvsiError {
    /// Registered refusal code (`parity/reason_codes.toml`).
    #[must_use]
    pub const fn registered_code(&self) -> &'static str {
        "choice_evidence_invalid"
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
    /// Monte Carlo standard error; zero when exact.
    pub standard_error: f64,
    /// Method.
    pub method: EvsiMethod,
    /// EVSI net of cost.
    pub net_value: f64,
    /// Whether the order against the next candidate is within combined standard errors.
    pub rank_uncertain: bool,
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
        .map(|(v, (a, c))| v * (a + c) / total)
        .fold(f64::MIN, f64::max)
}

struct Enumeration<'a> {
    alpha: &'a [f64],
    values: &'a [f64],
    base: f64,
    post_total: f64,
    counts: Vec<f64>,
    acc: f64,
}

impl Enumeration<'_> {
    fn go(&mut self, i: usize, left: u32) {
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
            return;
        }
        for c in 0..=left {
            self.counts[i] = f64::from(c);
            self.go(i + 1, left - c);
        }
    }
}

fn exact_evsi(alpha: &[f64], values: &[f64], n: u32) -> f64 {
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
    };
    walk.go(0, n);
    walk.acc
}

fn monte_carlo_evsi(alpha: &[f64], values: &[f64], n: u32, seed: u64) -> (f64, f64) {
    let a0: f64 = alpha.iter().sum();
    let total = a0 + f64::from(n);
    let mut rng = CausalRng::from_seed(seed);
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    let mut theta = vec![0.0; alpha.len()];
    let mut counts = vec![0.0; alpha.len()];
    for _ in 0..MONTE_CARLO_DRAWS {
        let mut s = 0.0;
        for (t, a) in theta.iter_mut().zip(alpha) {
            *t = sample_gamma(*a, 1.0, &mut rng);
            s += *t;
        }
        counts.iter_mut().for_each(|c| *c = 0.0);
        for _ in 0..n {
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
    (mean, (var / m).sqrt())
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
    if values.len() != alpha.len() || values.iter().any(|v| !v.is_finite()) {
        return Err(ChoiceEvsiError::Invalid("one finite value per option is required".into()));
    }
    if candidates.is_empty()
        || candidates.iter().any(|c| c.observations == 0 || !c.cost.is_finite() || c.cost < 0.0)
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
        let (value, se, method) = if comps <= MAX_EXACT_COMPOSITIONS {
            (
                exact_evsi(alpha, values, c.observations),
                0.0,
                EvsiMethod::Exact { compositions: comps },
            )
        } else if c.observations <= MAX_MONTE_CARLO_STUDY {
            let (v, se) = monte_carlo_evsi(alpha, values, c.observations, seed);
            (v, se, EvsiMethod::MonteCarlo { draws: MONTE_CARLO_DRAWS })
        } else {
            return Err(ChoiceEvsiError::Invalid(format!(
                "candidate `{}` exceeds the supported study size",
                c.label
            )));
        };
        let evsi = (value - now).max(0.0);
        rows.push(StudyRank {
            candidate: c.clone(),
            evsi,
            standard_error: se,
            method,
            net_value: evsi - c.cost,
            rank_uncertain: false,
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
        let got = exact_evsi(&[a, b], &[1.0, 0.0], n);
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
        let both = exact_evsi(&[a, b], &[1.0, 1.0], n);
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
        let exact = exact_evsi(&alpha, &values, 30);
        let (mc, se) = monte_carlo_evsi(&alpha, &values, 30, 7);
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
