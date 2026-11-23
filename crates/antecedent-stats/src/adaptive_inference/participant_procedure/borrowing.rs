//! Historical borrowing receipts and prior-data conflict.

use super::super::historical_prior::HistoricalPrior;
use super::{
    ArmEvidence, Error, ParticipantHistory, ParticipantSpec, PosteriorRequest, exact_posterior,
};

// -- Borrowing receipt ----------------------------------------------------------------------------

/// What one arm borrowed from its historical sources, and whether the current
/// data of the arm agree with what it borrowed.
#[derive(Clone, Debug, PartialEq)]
pub struct ArmBorrowing {
    /// The arm (0 is the control).
    pub arm: usize,
    /// Every historical source with its counts and its discount.
    pub sources: Vec<HistoricalPrior>,
    /// Historical responses the discounts are worth: the sum over the sources
    /// of their trials times their discounts.
    pub effective_historical_sample_size: f64,
    /// The effective sample size `alpha + beta` of the resolved prior, the
    /// declared initial prior included.
    pub prior_effective_sample_size: f64,
    /// Participants of the arm whose responses had completed.
    pub current_participants: u64,
    /// The prior-data conflict diagnostic; `None` for an arm that borrowed
    /// nothing or has no completed participant.
    pub conflict: Option<PriorDataConflict>,
}

/// Agreement between the resolved (borrowed) prior of an arm and its current
/// data, read from the exact posteriors of the two.
///
/// The current-data posterior is the arm mean under the declared initial prior
/// and the participant-aware likelihood of the completed participants alone;
/// the borrowed prior is the arm's resolved Beta prior. Both are independent
/// distributions of the arm mean, so `current_exceeds_prior` is the exact
/// probability that a draw of the former exceeds a draw of the latter. It is a
/// diagnostic of how far the borrowed information sits from the new data, not a
/// test with an error rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PriorDataConflict {
    /// Probability that the arm mean drawn from the current-data posterior
    /// exceeds the one drawn from the borrowed prior.
    pub current_exceeds_prior: f64,
    /// `2 min(p, 1 - p)` of that probability: near zero when the current data
    /// sit entirely on one side of what was borrowed, one when they agree.
    pub agreement: f64,
}

impl ParticipantSpec {
    /// The borrowing receipt of a retained history: per arm, every source with
    /// its discount, the effective historical sample size, and the prior-data
    /// conflict between the borrowed prior and the participants whose
    /// responses had completed.
    ///
    /// # Errors
    /// A history that disagrees with the declaration in shape or counts, a
    /// posterior the quadrature cannot resolve, or cancellation.
    pub fn borrowing_receipt(
        &self,
        history: &ParticipantHistory,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<ArmBorrowing>, Error> {
        let mut counts = vec![vec![0_u64; self.responses as usize + 1]; self.arms()];
        for (&arm, success) in history.arms.iter().zip(&history.successes) {
            let Some(successes) = *success else { continue };
            if arm >= self.arms() || successes > self.responses {
                return Err(Error::Shape(
                    "a participant's arm or success count is outside the declaration",
                ));
            }
            counts[arm][successes as usize] += 1;
        }
        let mut receipt = Vec::with_capacity(self.arms());
        for (arm, ((resolved, declared), histogram)) in
            self.priors.iter().zip(&self.borrowing.arms).zip(&counts).enumerate()
        {
            let current_participants: u64 = histogram.iter().sum();
            let conflict = if resolved.sources.is_empty() || current_participants == 0 {
                None
            } else {
                let silent = vec![0_u64; histogram.len()];
                let pair = [
                    ArmEvidence {
                        prior_alpha: resolved.prior.alpha,
                        prior_beta: resolved.prior.beta,
                        counts: &silent,
                    },
                    ArmEvidence {
                        prior_alpha: declared.initial.alpha,
                        prior_beta: declared.initial.beta,
                        counts: histogram,
                    },
                ];
                let posterior = exact_posterior(
                    &PosteriorRequest {
                        responses: self.responses,
                        correlation: self.correlation,
                        arms: &pair,
                        exceedance_margin: Some(0.0),
                    },
                    cancelled,
                )?;
                let exceeds = *posterior
                    .exceedance
                    .first()
                    .ok_or(Error::Numerical("the conflict comparison returned no exceedance"))?;
                Some(PriorDataConflict {
                    current_exceeds_prior: exceeds,
                    agreement: 2.0 * exceeds.min(1.0 - exceeds),
                })
            };
            receipt.push(ArmBorrowing {
                arm,
                sources: resolved.sources.clone(),
                effective_historical_sample_size: resolved.borrowed_sample_size,
                prior_effective_sample_size: resolved.effective_sample_size,
                current_participants,
                conflict,
            });
        }
        Ok(receipt)
    }
}
