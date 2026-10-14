//! Proper scoring rules for predictive laws.
//!
//! Every score is a loss: smaller is better, and each is *strictly proper* on
//! the laws it is defined for (the true law has the smallest expected score).
//!
//! | score | defined for | form |
//! |-------|-------------|------|
//! | log score | Gaussian, Bernoulli, categorical, finite, provider-evaluated | `-ln p(y)` |
//! | CRPS | Gaussian, Bernoulli, finite, empirical sample | `integral (F(t) - 1{t >= y})^2 dt` |
//! | Brier | Bernoulli, categorical | `sum_k (p_k - 1{y = k})^2` (binary: `(p - y)^2`) |
//! | energy | joint finite, joint sample | `E\|X - y\| - 1/2 E\|X - X'\|` |
//!
//! # Closed forms used as oracles
//!
//! * Gaussian CRPS (Gneiting and Raftery 2007, eq. 20):
//!   `sigma (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))`, `z = (y - mu) / sigma`.
//! * Bernoulli `p = P(Y = 1)`: Brier `(p - y)^2`; the CRPS of a binary law equals the Brier
//!   score; log score `-ln p` for `y = 1`, `-ln (1 - p)` for `y = 0`.
//!
//! # Ensemble CRPS and the finite-sample note
//!
//! A sample of `m` draws is scored as the law it is, its empirical distribution. The CRPS of
//! that law is the energy form `(1/m) sum |x_i - y| - (1/(2 m^2)) sum_ij |x_i - x_j|`, and it
//! is computed as the integral of the squared difference of distribution functions, a sum of
//! non-negative terms. As a score of the *empirical law* it is proper.
//!
//! As an estimator of the CRPS of the law the draws were taken from it is biased upward by
//! `E|X - X'| / (2 m)` (the diagonal terms of the double sum are zero but counted in the
//! divisor). [`EnsembleCrpsForm::Fair`] replaces `m^2` by `m (m - 1)` (Ferro 2014), which is
//! unbiased for iid draws from the forecast law but is not a proper score of the empirical
//! law; it is offered as the estimator of the generating law's CRPS and is labelled as such.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::{log_norm_pdf, norm_cdf, norm_pdf};

use super::law::{ForecastLaw, LawKind, Outcome};
use super::refusal::{PredictiveAssessmentError, detail, invalid, unsupported};

/// `1 / sqrt(pi)`.
const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;

/// A proper scoring rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum ScoreKind {
    /// Logarithmic score `-ln p(y)`.
    Log,
    /// Continuous ranked probability score.
    Crps,
    /// Brier score.
    Brier,
    /// Energy score of a joint law (`beta = 1`).
    Energy,
}

impl ScoreKind {
    /// Every score, in report order.
    pub const ALL: [Self; 4] = [Self::Log, Self::Crps, Self::Brier, Self::Energy];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Log => "log_score",
            Self::Crps => "crps",
            Self::Brier => "brier",
            Self::Energy => "energy_score",
        }
    }

    /// Parse a stable name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}

/// How an empirical sample's CRPS is formed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum EnsembleCrpsForm {
    /// The CRPS of the empirical law of the draws: proper for that law. The default.
    #[default]
    EmpiricalLaw,
    /// The fair (unbiased for iid draws) estimator of the generating law's CRPS.
    /// Not a proper score of the empirical law; needs at least two draws.
    Fair,
}

impl EnsembleCrpsForm {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::EmpiricalLaw => "empirical_law",
            Self::Fair => "fair",
        }
    }
}

/// Logarithmic score `-ln N(y; mean, sd^2)` of a Gaussian law.
#[must_use]
pub fn log_score_gaussian(mean: f64, sd: f64, y: f64) -> f64 {
    let z = (y - mean) / sd;
    sd.ln() - log_norm_pdf(z)
}

/// Closed-form CRPS of `N(mean, sd^2)` at `y`:
/// `sd (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))`.
#[must_use]
pub fn crps_gaussian(mean: f64, sd: f64, y: f64) -> f64 {
    let z = (y - mean) / sd;
    sd * (z * (2.0 * norm_cdf(z) - 1.0) + 2.0 * norm_pdf(z) - INV_SQRT_PI)
}

/// Brier score of a Bernoulli forecast `P(Y = 1) = probability` at a `0/1` outcome.
#[must_use]
pub fn brier_binary(probability: f64, outcome: bool) -> f64 {
    let target = if outcome { 1.0 } else { 0.0 };
    (probability - target).powi(2)
}

/// Brier score `sum_k (p_k - 1{y = k})^2` of a categorical forecast.
///
/// For two categories this is twice [`brier_binary`] of the second category's
/// probability; the binary form is the Bernoulli Brier score.
#[must_use]
pub fn brier_categorical(probabilities: &[f64], outcome: usize) -> f64 {
    let mut total = 0.0;
    for (k, p) in probabilities.iter().enumerate() {
        let target = if k == outcome { 1.0 } else { 0.0 };
        total += (p - target).powi(2);
    }
    total
}

/// CRPS of a finite law with `atoms` sorted ascending by value (probabilities
/// summing to one) at `y`: the integral of `(F(t) - 1{t >= y})^2`, a sum of
/// non-negative terms between consecutive breakpoints.
#[must_use]
pub fn crps_weighted_atoms(atoms: &[(f64, f64)], y: f64) -> f64 {
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(atoms.len() + 1);
    points.extend_from_slice(atoms);
    points.push((y, 0.0));
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut cumulative = 0.0;
    let mut total = 0.0;
    for (k, point) in points.iter().enumerate() {
        cumulative += point.1;
        if let Some(next) = points.get(k + 1) {
            let width = next.0 - point.0;
            let indicator = if point.0 >= y { 1.0 } else { 0.0 };
            total += width * (cumulative - indicator).powi(2);
        }
    }
    total
}

/// Sum over ordered pairs `sum_ij |x_i - x_j|` of an ascending-sorted slice.
fn pairwise_abs_sum_sorted(sorted: &[f64]) -> f64 {
    let m = sorted.len();
    let mut total = 0.0;
    for (k, x) in sorted.iter().enumerate() {
        total += x * (2.0 * k as f64 - m as f64 + 1.0);
    }
    2.0 * total
}

/// Energy-form CRPS of an ascending-sorted sample at `y`.
///
/// `fair = false` is `(1/m) sum |x_i - y| - (1/(2 m^2)) sum_ij |x_i - x_j|`, the
/// CRPS of the empirical law. `fair = true` divides the pair sum by
/// `2 m (m - 1)` instead (needs `m >= 2`).
#[must_use]
pub fn crps_ensemble_energy_sorted(sorted: &[f64], y: f64, fair: bool) -> f64 {
    let m = sorted.len() as f64;
    let first: f64 = sorted.iter().map(|x| (x - y).abs()).sum::<f64>() / m;
    let pairs = pairwise_abs_sum_sorted(sorted);
    let divisor = if fair { 2.0 * m * (m - 1.0) } else { 2.0 * m * m };
    first - pairs / divisor
}

/// Energy score `sum_i w_i |x_i - y| - 1/2 sum_ij w_i w_j |x_i - x_j|` (Euclidean norm,
/// `beta = 1`) of a weighted set of `dimension`-vectors (row-major `atoms`) at `y`.
#[must_use]
pub fn energy_score_weighted(atoms: &[f64], weights: &[f64], dimension: usize, y: &[f64]) -> f64 {
    let row = |i: usize| &atoms[i * dimension..(i + 1) * dimension];
    let distance = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b).map(|(u, v)| (u - v) * (u - v)).sum::<f64>().sqrt()
    };
    let mut first = 0.0;
    for (i, weight) in weights.iter().enumerate() {
        first += weight * distance(row(i), y);
    }
    let mut second = 0.0;
    for (i, weight_i) in weights.iter().enumerate() {
        for (j, weight_j) in weights.iter().enumerate().skip(i + 1) {
            second += weight_i * weight_j * distance(row(i), row(j));
        }
    }
    // The double sum over ordered pairs is twice the sum over i < j, so the 1/2 cancels.
    first - second
}

fn outcome_mismatch(law: LawKind, expected: &str, got: &Outcome) -> PredictiveAssessmentError {
    invalid(
        detail::OUTCOME_KIND,
        format!(
            "a {} law predicts a {expected} outcome but the realized outcome is a {} outcome",
            law.name(),
            got.family().name()
        ),
    )
}

fn not_defined(law: LawKind, score: ScoreKind, supports: &str) -> PredictiveAssessmentError {
    unsupported(
        detail::SCORE_NOT_DEFINED,
        format!(
            "the {} is not defined for a {} law; that law is scored by {supports}. Request a \
             score the law supports, or supply the law in a family that supports this score",
            score.name(),
            law.name()
        ),
    )
}

fn infinite_log(law: LawKind) -> PredictiveAssessmentError {
    invalid(
        detail::LOG_SCORE_INFINITE,
        format!(
            "the {} law gives the realized outcome probability zero, so its log score is \
             infinite and no mean or difference of scores exists. Score it by the CRPS or Brier \
             score, or supply a law with the outcome in its support",
            law.name()
        ),
    )
}

/// Score one case: `law` against the realized `outcome`.
///
/// # Errors
/// An outcome of another kind than the law predicts, a score the law does not
/// support, or an infinite log score.
pub fn score_case(
    kind: ScoreKind,
    form: EnsembleCrpsForm,
    law: &ForecastLaw,
    outcome: &Outcome,
) -> Result<f64, PredictiveAssessmentError> {
    law.validate()?;
    match outcome {
        Outcome::Real(value) if !value.is_finite() => {
            return Err(invalid(detail::INVALID_OBSERVATIONS, "a realized outcome is not finite"));
        }
        Outcome::Vector(values)
            if values.is_empty() || values.iter().any(|value| !value.is_finite()) =>
        {
            return Err(invalid(
                detail::INVALID_OBSERVATIONS,
                "a realized vector outcome is empty or contains a non-finite value",
            ));
        }
        _ => {}
    }
    score_case_validated(kind, form, law, outcome)
}

/// Score a law and outcome already checked by the enclosing assessment.
///
/// Keeping this separate avoids revalidating every draw in an empirical law for
/// every realized case; [`score_case`] remains safe for standalone callers.
#[allow(
    clippy::too_many_lines,
    reason = "one exhaustive (law, outcome) match keeps every defined score and refusal in one place"
)]
pub(crate) fn score_case_validated(
    kind: ScoreKind,
    form: EnsembleCrpsForm,
    law: &ForecastLaw,
    outcome: &Outcome,
) -> Result<f64, PredictiveAssessmentError> {
    let score = match (law, outcome) {
        (ForecastLaw::Gaussian { mean, sd }, Outcome::Real(y)) => match kind {
            ScoreKind::Log => Ok(log_score_gaussian(*mean, *sd, *y)),
            ScoreKind::Crps => Ok(crps_gaussian(*mean, *sd, *y)),
            _ => Err(not_defined(LawKind::Gaussian, kind, "the log score and the CRPS")),
        },
        (ForecastLaw::Bernoulli { probability }, Outcome::Category(c)) => {
            let hit = match c {
                0 => false,
                1 => true,
                other => {
                    return Err(invalid(
                        detail::INVALID_OBSERVATIONS,
                        format!("a Bernoulli outcome is 0 or 1, not {other}"),
                    ));
                }
            };
            match kind {
                ScoreKind::Brier | ScoreKind::Crps => Ok(brier_binary(*probability, hit)),
                ScoreKind::Log => {
                    let p = if hit { *probability } else { 1.0 - probability };
                    if p <= 0.0 {
                        return Err(infinite_log(LawKind::Bernoulli));
                    }
                    Ok(-p.ln())
                }
                ScoreKind::Energy => Err(not_defined(
                    LawKind::Bernoulli,
                    kind,
                    "the log score, the Brier score and the CRPS",
                )),
            }
        }
        (ForecastLaw::Categorical { probabilities }, Outcome::Category(c)) => {
            let index = usize::try_from(*c).unwrap_or(usize::MAX);
            if index >= probabilities.len() {
                return Err(invalid(
                    detail::INVALID_OBSERVATIONS,
                    format!(
                        "category {c} is outside the {} categories of the law",
                        probabilities.len()
                    ),
                ));
            }
            match kind {
                ScoreKind::Brier => Ok(brier_categorical(probabilities, index)),
                ScoreKind::Log => {
                    let p = probabilities[index];
                    if p <= 0.0 {
                        return Err(infinite_log(LawKind::Categorical));
                    }
                    Ok(-p.ln())
                }
                _ => Err(not_defined(
                    LawKind::Categorical,
                    kind,
                    "the log score and the Brier score (an unordered law has no CRPS)",
                )),
            }
        }
        (ForecastLaw::FiniteReal { support, probabilities }, Outcome::Real(y)) => match kind {
            ScoreKind::Crps => {
                let atoms: Vec<(f64, f64)> =
                    support.iter().copied().zip(probabilities.iter().copied()).collect();
                Ok(crps_weighted_atoms(&atoms, *y))
            }
            ScoreKind::Log => {
                let mass: f64 = support
                    .iter()
                    .zip(probabilities)
                    .filter(|(x, _)| x.partial_cmp(&y) == Some(std::cmp::Ordering::Equal))
                    .map(|(_, p)| *p)
                    .sum();
                if mass <= 0.0 {
                    return Err(infinite_log(LawKind::FiniteReal));
                }
                Ok(-mass.ln())
            }
            _ => Err(not_defined(LawKind::FiniteReal, kind, "the log score and the CRPS")),
        },
        (ForecastLaw::EmpiricalSample { draws }, Outcome::Real(y)) => match kind {
            ScoreKind::Crps => {
                let mut sorted = draws.clone();
                sorted.sort_by(f64::total_cmp);
                let fair = form == EnsembleCrpsForm::Fair;
                if fair && sorted.len() < 2 {
                    return Err(invalid(
                        detail::INVALID_LAW,
                        "the fair ensemble CRPS needs at least two draws",
                    ));
                }
                if fair {
                    Ok(crps_ensemble_energy_sorted(&sorted, *y, true))
                } else {
                    let weight = 1.0 / sorted.len() as f64;
                    let atoms: Vec<(f64, f64)> = sorted.iter().map(|x| (*x, weight)).collect();
                    Ok(crps_weighted_atoms(&atoms, *y))
                }
            }
            _ => Err(not_defined(
                LawKind::EmpiricalSample,
                kind,
                "the CRPS (a sample has no density; supply a density to score it by the log score)",
            )),
        },
        (ForecastLaw::PointEvaluated { log_density, .. }, Outcome::Real(_)) => {
            match (kind, log_density) {
                (ScoreKind::Log, Some(value)) => Ok(-value),
                (ScoreKind::Log, None) => Err(not_defined(
                    LawKind::PointEvaluated,
                    kind,
                    "the log score only when the provider supplies the log density at the outcome",
                )),
                _ => Err(not_defined(
                    LawKind::PointEvaluated,
                    kind,
                    "the log score and the PIT (only values at the outcome are available)",
                )),
            }
        }
        (ForecastLaw::JointFinite { dimension, atoms, probabilities }, Outcome::Vector(y)) => {
            check_dimension(*dimension, y.len())?;
            match kind {
                ScoreKind::Energy => Ok(energy_score_weighted(atoms, probabilities, *dimension, y)),
                _ => Err(not_defined(LawKind::JointFinite, kind, "the energy score")),
            }
        }
        (ForecastLaw::JointSample { dimension, draws }, Outcome::Vector(y)) => {
            check_dimension(*dimension, y.len())?;
            match kind {
                ScoreKind::Energy => {
                    let m = draws.len() / dimension;
                    let weights = vec![1.0 / m as f64; m];
                    Ok(energy_score_weighted(draws, &weights, *dimension, y))
                }
                _ => Err(not_defined(LawKind::JointSample, kind, "the energy score")),
            }
        }
        (law, outcome) => {
            let expected = law.outcome_family().name();
            Err(outcome_mismatch(law.kind(), expected, outcome))
        }
    }?;
    if score.is_finite() {
        Ok(score)
    } else {
        Err(invalid(
            detail::INVALID_LAW,
            "the requested score overflowed the finite numeric range; rescale the outcome and forecast law",
        ))
    }
}

fn check_dimension(law: usize, outcome: usize) -> Result<(), PredictiveAssessmentError> {
    if law == outcome {
        Ok(())
    } else {
        Err(invalid(
            detail::OUTCOME_KIND,
            format!(
                "a joint law of dimension {law} was scored against a vector of length {outcome}"
            ),
        ))
    }
}
