//! Replay decision values and retained signal laws for ranking artifacts.

use crate::evsi::IntegrationMethod;
use crate::preposterior::expected_max_affine;

use super::{
    DecisionWire, EvsiCandidateWire, LawWire, MASS_TOLERANCE, PriorWire, ReplayKind, UtilityWire,
};

/// What the stored decision implies before any information.
pub(super) struct Baseline {
    pub(super) bayes: usize,
    pub(super) prior_best: f64,
    pub(super) evpi: f64,
    /// Lines `(c_a, d_a)` of the conjugate-normal analysis (normal belief only).
    lines: Vec<(f64, f64)>,
}

fn finite_rows(decision: &DecisionWire) -> Option<(&[Vec<f64>], usize)> {
    match (&decision.prior, &decision.utility) {
        (PriorWire::Draws { states }, UtilityWire::Table { rows }) => {
            Some((rows.as_slice(), states.len()))
        }
        _ => None,
    }
}

fn admissible_indices(decision: &DecisionWire) -> Vec<usize> {
    decision.admissible.iter().enumerate().filter(|(_, ok)| **ok).map(|(a, _)| a).collect()
}

pub(super) fn baseline(decision: &DecisionWire) -> Baseline {
    if let Some((rows, k)) = finite_rows(decision) {
        let kf = k as f64;
        let adm = admissible_indices(decision);
        let mut bayes = adm[0];
        let mut prior_best = f64::NEG_INFINITY;
        for &a in &adm {
            let eu = rows[a].iter().sum::<f64>() / kf;
            if eu > prior_best {
                prior_best = eu;
                bayes = a;
            }
        }
        let evpi = (0..k)
            .map(|draw| {
                adm.iter()
                    .map(|&a| rows[a][draw] - rows[bayes][draw])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .sum::<f64>()
            / kf;
        return Baseline { bayes, prior_best, evpi, lines: Vec::new() };
    }
    let (PriorWire::Normal { mean, variance }, UtilityWire::Affine { coefficients }) =
        (&decision.prior, &decision.utility)
    else {
        return Baseline { bayes: 0, prior_best: 0.0, evpi: 0.0, lines: Vec::new() };
    };
    let mut bayes = 0;
    let mut prior_best = f64::NEG_INFINITY;
    for (a, (alpha, beta)) in coefficients.iter().enumerate() {
        let eu = alpha + beta * mean;
        if eu > prior_best {
            prior_best = eu;
            bayes = a;
        }
    }
    let (alpha_star, beta_star) = coefficients[bayes];
    let lines: Vec<(f64, f64)> = coefficients
        .iter()
        .map(|&(alpha, beta)| {
            let slope = beta - beta_star;
            ((alpha - alpha_star + slope * mean).min(0.0), slope)
        })
        .collect();
    let evpi = expected_max_affine(&lines, variance.sqrt());
    Baseline { bayes, prior_best, evpi, lines }
}

/// Binomial likelihood `[y][k]` over the prior draws.
fn binomial_table(states: &[f64], n: usize) -> Vec<Vec<f64>> {
    let mut ln_choose = 0.0_f64;
    let mut rows = Vec::with_capacity(n + 1);
    for y in 0..=n {
        if y > 0 {
            ln_choose += ((n - y + 1) as f64 / y as f64).ln();
        }
        rows.push(
            states
                .iter()
                .map(|&theta| {
                    let success = if y > 0 { y as f64 * theta.ln() } else { 0.0 };
                    let failure = if y < n { (n - y) as f64 * (1.0 - theta).ln() } else { 0.0 };
                    (ln_choose + success + failure).exp()
                })
                .collect(),
        );
    }
    rows
}

pub(super) fn rectangular(rows: &[Vec<f64>], n_rows: usize, n_cols: usize) -> bool {
    rows.len() == n_rows && rows.iter().all(|r| r.len() == n_cols)
}

fn probabilities(values: &[f64]) -> bool {
    values.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p))
}

/// Expected value of the information, `Σ_y max_{a ∈ F} E[P(y | θ) (U(a, θ) − U(a*, θ))]` for the
/// Bayes action `a*`, with the likelihood mass checked.
///
/// Utilities enter only as gaps to the Bayes action, so a common utility offset cannot enter the
/// cancellation `E[max after] − max E[prior]` (the evaluator that produced the stored value uses
/// the same form); `bayes` is the Bayes action of [`baseline`].
fn value_after(decision: &DecisionWire, likelihood: &[Vec<f64>], bayes: usize) -> Option<f64> {
    let (rows, k) = finite_rows(decision)?;
    let kf = k as f64;
    let adm = admissible_indices(decision);
    let mut total = 0.0;
    let mut mass = 0.0;
    for prow in likelihood {
        mass += prow.iter().sum::<f64>() / kf;
        total += adm
            .iter()
            .map(|&a| {
                rows[a]
                    .iter()
                    .zip(&rows[bayes])
                    .zip(prow)
                    .map(|((u, best), p)| (u - best) * p)
                    .sum::<f64>()
                    / kf
            })
            .fold(f64::NEG_INFINITY, f64::max);
    }
    ((mass - 1.0).abs() <= MASS_TOLERANCE).then_some(total)
}

fn column_masses_ok(likelihood: &[Vec<f64>], k: usize) -> bool {
    (0..k).all(|col| {
        let mass: f64 = likelihood.iter().map(|row| row[col]).sum();
        (mass - 1.0).abs() <= MASS_TOLERANCE
    })
}

/// Likelihood `[y][k]` implied by an attested posterior, with coherence checked.
fn posterior_to_likelihood(
    k: usize,
    predictive: &[f64],
    posterior: &[Vec<f64>],
) -> Option<Vec<Vec<f64>>> {
    let kf = k as f64;
    if !probabilities(predictive)
        || !rectangular(posterior, predictive.len(), k)
        || !posterior.iter().all(|row| probabilities(row))
        || (predictive.iter().sum::<f64>() - 1.0).abs() > MASS_TOLERANCE
        || !posterior.iter().all(|row| (row.iter().sum::<f64>() - 1.0).abs() <= MASS_TOLERANCE)
    {
        return None;
    }
    for col in 0..k {
        let recovered: f64 = predictive.iter().zip(posterior).map(|(p, row)| p * row[col]).sum();
        if (recovered - 1.0 / kf).abs() > MASS_TOLERANCE {
            return None;
        }
    }
    Some(
        predictive
            .iter()
            .zip(posterior)
            .map(|(p, row)| row.iter().map(|q| kf * p * q).collect())
            .collect(),
    )
}

/// What the consumer could recompute for one candidate.
pub(super) struct Recomputed {
    /// Recomputed EVSI, when the law is replayable.
    pub(super) evsi: Option<f64>,
    pub(super) evpi: f64,
    pub(super) replay: ReplayKind,
}

pub(super) fn is_native(law: &LawWire) -> bool {
    matches!(law, LawWire::GaussianMean { .. } | LawWire::Binomial)
}

pub(super) fn recompute(
    candidate: &EvsiCandidateWire,
    decision: &DecisionWire,
    base: &Baseline,
    method: IntegrationMethod,
    max_states: usize,
) -> std::result::Result<Recomputed, &'static str> {
    let n = candidate.sample_size;
    let k = finite_rows(decision).map_or(0, |(_, k)| k);
    match (&candidate.law, method) {
        (LawWire::GaussianMean { noise_variance }, IntegrationMethod::Exact) => {
            let PriorWire::Normal { variance, .. } = &decision.prior else {
                return Err("law");
            };
            if !(noise_variance.is_finite() && *noise_variance > 0.0) {
                return Err("law");
            }
            let nf = n as f64;
            let preposterior = variance * (nf * variance) / (nf * variance + noise_variance);
            let after = expected_max_affine(&base.lines, preposterior.sqrt());
            Ok(Recomputed {
                evsi: Some(after),
                evpi: base.evpi,
                replay: ReplayKind::NativeExactRecomputed,
            })
        }
        (LawWire::GaussianMean { noise_variance }, IntegrationMethod::MonteCarlo) => {
            if !(noise_variance.is_finite() && *noise_variance > 0.0) || k == 0 {
                return Err("law");
            }
            Ok(Recomputed {
                evsi: None,
                evpi: base.evpi,
                replay: ReplayKind::MonteCarloNotReplayed,
            })
        }
        (LawWire::Binomial, IntegrationMethod::Exact) => {
            let PriorWire::Draws { states } = &decision.prior else {
                return Err("law");
            };
            let Ok(n_us) = usize::try_from(n) else {
                return Err("law");
            };
            if !states.iter().all(|s| (0.0..=1.0).contains(s)) || n_us >= max_states {
                return Err("law");
            }
            let table = binomial_table(states, n_us);
            let gain = value_after(decision, &table, base.bayes).ok_or("law")?;
            Ok(Recomputed {
                evsi: Some(gain),
                evpi: base.evpi,
                replay: ReplayKind::NativeExactRecomputed,
            })
        }
        (
            LawWire::FiniteLikelihood { statistics, probabilities: table },
            IntegrationMethod::Exact,
        ) => {
            if statistics.is_empty()
                || !rectangular(table, statistics.len(), k)
                || !table.iter().all(|row| probabilities(row))
                || !column_masses_ok(table, k)
            {
                return Err("law");
            }
            let gain = value_after(decision, table, base.bayes).ok_or("law")?;
            Ok(Recomputed {
                evsi: Some(gain),
                evpi: base.evpi,
                replay: ReplayKind::AttestedLawArithmetic,
            })
        }
        (
            LawWire::FinitePosterior { statistics, predictive, posterior },
            IntegrationMethod::Exact,
        ) => {
            if statistics.len() != predictive.len() || statistics.is_empty() {
                return Err("law");
            }
            let table = posterior_to_likelihood(k, predictive, posterior).ok_or("law")?;
            let gain = value_after(decision, &table, base.bayes).ok_or("law")?;
            Ok(Recomputed {
                evsi: Some(gain),
                evpi: base.evpi,
                replay: ReplayKind::AttestedLawArithmetic,
            })
        }
        (
            LawWire::DecisionValues { branch_probabilities, action_ids, values },
            IntegrationMethod::ExternallyComputed,
        ) => {
            let (rows, kk) = finite_rows(decision).ok_or("law")?;
            let n_actions = decision.action_ids.len();
            let mut left: Vec<&String> = action_ids.iter().collect();
            let mut right: Vec<&String> = decision.action_ids.iter().collect();
            left.sort();
            right.sort();
            if left != right {
                return Err("law");
            }
            if branch_probabilities.is_empty()
                || !probabilities(branch_probabilities)
                || !rectangular(values, branch_probabilities.len(), n_actions)
                || !values.iter().all(|row| row.iter().all(|v| v.is_finite()))
                || (branch_probabilities.iter().sum::<f64>() - 1.0).abs() > MASS_TOLERANCE
            {
                return Err("law");
            }
            let column: Vec<usize> = decision
                .action_ids
                .iter()
                .filter_map(|id| action_ids.iter().position(|v| v == id))
                .collect();
            for action in 0..n_actions {
                let averaged: f64 = branch_probabilities
                    .iter()
                    .enumerate()
                    .map(|(y, p)| p * values[y][column[action]])
                    .sum();
                let prior_eu = rows[action].iter().sum::<f64>() / kk as f64;
                if (averaged - prior_eu).abs() > MASS_TOLERANCE * (1.0 + prior_eu.abs()) {
                    return Err("law");
                }
            }
            let adm = admissible_indices(decision);
            let after: f64 = branch_probabilities
                .iter()
                .enumerate()
                .map(|(y, p)| {
                    p * adm.iter().map(|&a| values[y][column[a]]).fold(f64::NEG_INFINITY, f64::max)
                })
                .sum();
            Ok(Recomputed {
                evsi: Some(after - base.prior_best),
                evpi: base.evpi,
                replay: ReplayKind::AttestedValuesCombined,
            })
        }
        _ => Err("integration"),
    }
}
