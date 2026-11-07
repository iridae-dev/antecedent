use super::{CandidateDesign, DesignError, PreposteriorAnalysis, ScoreEvaluation};
use antecedent_core::CausalRng;

/// Per-candidate preposterior scoring: the design's sample size and, where
/// available, its exact EVSI (computed once and reused by every replicate).
pub(super) struct DecisionScoring<'a, O> {
    analysis: PreposteriorAnalysis<'a, O>,
    /// `(n, exact EVSI)` by candidate index; `None` for inactive candidates.
    plans: Vec<Option<(u64, Option<f64>)>>,
}

impl<'a, O> DecisionScoring<'a, O> {
    pub(super) fn new(
        analysis: PreposteriorAnalysis<'a, O>,
        candidates: &[CandidateDesign],
        active: &[usize],
    ) -> Result<Self, DesignError> {
        let mut plans = vec![None; candidates.len()];
        for &cand_i in active {
            if let Some(n) = analysis.signal().sample_size(&candidates[cand_i]) {
                plans[cand_i] = Some((n, analysis.exact_evsi(n)?));
            }
        }
        Ok(Self { analysis, plans })
    }

    pub(super) fn evaluation(&self, cand_i: usize) -> ScoreEvaluation {
        match self.plans.get(cand_i).copied().flatten() {
            Some((_, Some(_))) => ScoreEvaluation::Exact,
            _ => ScoreEvaluation::MonteCarlo,
        }
    }

    pub(super) fn score(
        &self,
        cand_i: usize,
        crn_seed: u64,
        log_lik: &mut Vec<f64>,
        weights: &mut Vec<f64>,
    ) -> Result<f64, DesignError> {
        match self.plans.get(cand_i).copied().flatten() {
            Some((_, Some(exact))) => Ok(exact),
            Some((n, None)) => self.analysis.draw_evsi_replicate(
                n,
                &mut CausalRng::from_seed(crn_seed),
                log_lik,
                weights,
            ),
            None => Err(DesignError::Config(format!(
                "candidate {cand_i} has no decision sample size and cannot be scored"
            ))),
        }
    }
}
