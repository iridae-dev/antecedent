//! Exact expected scores of a predictive law under a finite true law.
//!
//! When the generating law of the outcome is a finite law `P`, the expected
//! score of a forecast `Q` is a finite sum `sum_k P(y_k) S(Q, y_k)`. There is no
//! sampling, so no sampling interval exists or is reported: the number is exact
//! up to floating point. This is the oracle behind the proper-score property
//! (the true law attains the smallest expected score) and the reference value for
//! the sample-based assessments.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::law::{
    ForecastLaw, MAX_JOINT_DIMENSION, MAX_LAW_ATOMS, Outcome, PROBABILITY_SUM_TOLERANCE,
};
use super::refusal::{PredictiveAssessmentError, detail, invalid};
use super::score::{EnsembleCrpsForm, ScoreKind, score_case_validated};

/// A finite true law of the outcome.
#[derive(Clone, Debug, PartialEq)]
pub enum FiniteTruth {
    /// A real-valued outcome on a finite support.
    Real {
        /// Support points.
        support: Vec<f64>,
        /// Probabilities summing to one.
        probabilities: Vec<f64>,
    },
    /// A category outcome.
    Category {
        /// Category probabilities summing to one.
        probabilities: Vec<f64>,
    },
    /// A vector outcome on a finite set of atoms.
    Vector {
        /// Dimension.
        dimension: usize,
        /// Atoms, row-major.
        atoms: Vec<f64>,
        /// Probabilities summing to one.
        probabilities: Vec<f64>,
    },
}

fn check_masses(probabilities: &[f64]) -> Result<(), PredictiveAssessmentError> {
    let total: f64 = probabilities.iter().sum();
    if probabilities.is_empty()
        || probabilities.len() > MAX_LAW_ATOMS
        || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
        || (total - 1.0).abs() > PROBABILITY_SUM_TOLERANCE
    {
        return Err(invalid(
            detail::INVALID_LAW,
            "a finite truth needs between one and the maximum number of non-negative \
             probabilities summing to one",
        ));
    }
    Ok(())
}

/// The exact expected `kind` score of `forecast` when the outcome follows
/// `truth`: `sum_k truth(y_k) * score(forecast, y_k)`, summed in support order.
///
/// Outcomes of probability zero are skipped, so a forecast need not cover them.
///
/// # Errors
/// A malformed truth, or any refusal of [`score_case`] at an outcome of positive
/// probability (for example an infinite log score).
pub fn exact_expected_score(
    kind: ScoreKind,
    forecast: &ForecastLaw,
    truth: &FiniteTruth,
) -> Result<f64, PredictiveAssessmentError> {
    forecast.validate()?;
    let form = EnsembleCrpsForm::EmpiricalLaw;
    let mut total = 0.0;
    match truth {
        FiniteTruth::Real { support, probabilities } => {
            check_masses(probabilities)?;
            if support.len() != probabilities.len() {
                return Err(invalid(detail::INVALID_LAW, "one probability per support point"));
            }
            if support.iter().any(|x| !x.is_finite()) {
                return Err(invalid(
                    detail::INVALID_LAW,
                    "a finite truth support point is not finite",
                ));
            }
            for (x, p) in support.iter().zip(probabilities) {
                if *p > 0.0 {
                    total += p * score_case_validated(kind, form, forecast, &Outcome::Real(*x))?;
                }
            }
        }
        FiniteTruth::Category { probabilities } => {
            check_masses(probabilities)?;
            for (k, p) in probabilities.iter().enumerate() {
                if *p > 0.0 {
                    let category = u32::try_from(k).unwrap_or(u32::MAX);
                    total += p * score_case_validated(
                        kind,
                        form,
                        forecast,
                        &Outcome::Category(category),
                    )?;
                }
            }
        }
        FiniteTruth::Vector { dimension, atoms, probabilities } => {
            check_masses(probabilities)?;
            if *dimension == 0
                || *dimension > MAX_JOINT_DIMENSION
                || probabilities.len().checked_mul(*dimension) != Some(atoms.len())
                || atoms.iter().any(|x| !x.is_finite())
            {
                return Err(invalid(
                    detail::INVALID_LAW,
                    "a vector truth needs a bounded positive dimension and atoms * dimension finite reals",
                ));
            }
            for (row, p) in atoms.chunks_exact(*dimension).zip(probabilities) {
                if *p > 0.0 {
                    total += p * score_case_validated(
                        kind,
                        form,
                        forecast,
                        &Outcome::Vector(row.to_vec()),
                    )?;
                }
            }
        }
    }
    if total.is_finite() {
        Ok(total)
    } else {
        Err(invalid(
            detail::INVALID_LAW,
            "the expected score overflowed the finite numeric range; rescale the outcome and forecast law",
        ))
    }
}
