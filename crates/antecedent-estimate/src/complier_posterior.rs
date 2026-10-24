//! Bayesian-bootstrap posterior of the complier effect under known Bernoulli encouragement.
//!
//! The frequentist row ([`crate::randomized_scores::complier_wald_effect`]) divides the mean
//! intention-to-treat score by the mean first-stage score. Here both score means are
//! functionals of one Dirichlet(1, ..., 1) weight vector per draw (a Dirichlet-process prior of
//! vanishing concentration on the subject law, see [`crate::score_posterior`]), so a draw of the
//! complier effect is the ratio of that draw's intention-to-treat draw to its first-stage draw.
//! The declared assignment probabilities identify both means; exclusion and monotonicity are
//! declared, not tested, and the prior identifies nothing.
//!
//! **Weak first stage.** This implementation refuses if any retained first-stage draw is
//! nonpositive. This is a finite-draw instability guard, not proof that the full score-law
//! posterior excludes zero. A ratio law can be proper without finite moments; no ratio
//! expectation or general support certificate follows from passing this guard. These are
//! empirical score-law weights, not a principal-strata likelihood enforcing monotonicity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use crate::EstimationError;
use crate::score_posterior::{
    ScorePosterior, ScorePosteriorOptions, bayesian_bootstrap_score_posteriors, summarize,
};

/// Joint posterior of the intention-to-treat effect, the first stage and their ratio.
///
/// The three posteriors share every draw's weights, so `complier.draws[k]` equals
/// `intention_to_treat.draws[k] / first_stage.draws[k]`.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplierEffectPosterior {
    /// Posterior of the complier effect (the Wald ratio).
    pub complier: ScorePosterior,
    /// Posterior of the effect of assignment on the outcome.
    pub intention_to_treat: ScorePosterior,
    /// Posterior of the effect of assignment on receipt.
    pub first_stage: ScorePosterior,
}

/// Posterior of the complier effect from a known Bernoulli encouragement design.
///
/// # Errors
/// Misaligned or non-finite inputs, probabilities outside `(0, 1)`, an arm with no rows, a
/// weak first stage (any first-stage draw at or below zero), and the option refusals of
/// [`bayesian_bootstrap_score_posteriors`].
pub fn bayesian_complier_effect(
    outcomes: &[f64],
    assignment: &[bool],
    received: &[bool],
    probabilities: &[f64],
    options: ScorePosteriorOptions,
    ctx: &ExecutionContext,
) -> Result<ComplierEffectPosterior, EstimationError> {
    let n = outcomes.len();
    if n < 2 || assignment.len() != n || received.len() != n || probabilities.len() != n {
        return Err(EstimationError::data_msg(
            "complier posterior needs aligned outcome, assignment, receipt and probability rows",
        ));
    }
    let mut outcome_scores = Vec::with_capacity(n);
    let mut receipt_scores = Vec::with_capacity(n);
    for i in 0..n {
        let p = probabilities[i];
        if !outcomes[i].is_finite() || !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(EstimationError::data_msg(
                "complier posterior needs finite outcomes and probabilities strictly inside (0, 1)",
            ));
        }
        let sign = if assignment[i] { 1.0 / p } else { -1.0 / (1.0 - p) };
        outcome_scores.push(sign * outcomes[i]);
        receipt_scores.push(sign * f64::from(received[i]));
    }
    if assignment.iter().all(|z| *z) || assignment.iter().all(|z| !*z) {
        return Err(EstimationError::data_msg(
            "complier posterior needs rows assigned to both arms",
        ));
    }
    let mut columns =
        bayesian_bootstrap_score_posteriors(&[&outcome_scores, &receipt_scores], options, ctx)?
            .into_iter();
    let (Some(intention_to_treat), Some(first_stage)) = (columns.next(), columns.next()) else {
        return Err(EstimationError::stats_msg("complier posterior lost a score column"));
    };
    if first_stage.draws.iter().any(|draw| !draw.is_finite() || *draw <= f64::EPSILON) {
        return Err(EstimationError::unsupported(
            "weak first stage: a retained empirical score-law receipt draw is nonpositive; \
             the ratio procedure refuses this unstable realization. This is a finite-draw \
             guard, not proof of posterior support or nonexistence of a proper ratio law. \
             Next attempt: stronger encouragement or weak-instrument-robust inference",
        ));
    }
    let ratio = intention_to_treat
        .draws
        .iter()
        .zip(&first_stage.draws)
        .map(|(itt, stage)| itt / stage)
        .collect::<Vec<_>>();
    let point = intention_to_treat.score_mean / first_stage.score_mean;
    let complier = summarize(ratio, point, options.coverage);
    Ok(ComplierEffectPosterior { complier, intention_to_treat, first_stage })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Perfect compliers with constant effect 2 and known p = 0.5, alternating assignment.
    fn fixture(n: usize, compliance: f64) -> (Vec<f64>, Vec<bool>, Vec<bool>, Vec<f64>) {
        let mut outcome = Vec::new();
        let mut assigned = Vec::new();
        let mut received = Vec::new();
        for i in 0..n {
            let z = i % 2 == 0;
            let complier = (i % 10) as f64 / 10.0 < compliance;
            let d = z && complier;
            assigned.push(z);
            received.push(d);
            outcome.push(1.0 + 0.1 * ((i % 7) as f64) + if d { 2.0 } else { 0.0 });
        }
        (outcome, assigned, received, vec![0.5; n])
    }

    #[test]
    fn posterior_recovers_the_known_complier_effect_and_ties_to_its_components() {
        let (y, z, d, p) = fixture(2_000, 0.7);
        let ctx = ExecutionContext::for_tests(3);
        let options = ScorePosteriorOptions { draws: 2_000, coverage: 0.95 };
        let fit = bayesian_complier_effect(&y, &z, &d, &p, options, &ctx).unwrap();
        assert!((fit.complier.posterior_mean - 2.0).abs() < 0.3, "{}", fit.complier.posterior_mean);
        assert!(fit.complier.interval.0 < 2.0 && 2.0 < fit.complier.interval.1);
        for k in 0..50 {
            let ratio = fit.intention_to_treat.draws[k] / fit.first_stage.draws[k];
            assert!((fit.complier.draws[k] - ratio).abs() < 1e-9);
        }
        let frequentist = crate::randomized_scores::complier_wald_effect(&y, &z, &d, &p).unwrap();
        assert!((fit.complier.score_mean - frequentist.effect).abs() < 1e-9);
    }

    #[test]
    fn a_weak_first_stage_is_refused_with_its_next_attempt() {
        // Receipt is unrelated to assignment, so the first stage is centred on zero.
        let (y, z, _, p) = fixture(200, 0.7);
        let d: Vec<bool> = (0..200).map(|i| i % 3 == 0).collect();
        let ctx = ExecutionContext::for_tests(3);
        let options = ScorePosteriorOptions { draws: 500, coverage: 0.95 };
        let error = bayesian_complier_effect(&y, &z, &d, &p, options, &ctx).unwrap_err();
        assert!(error.to_string().contains("weak first stage"), "{error}");
    }
}
