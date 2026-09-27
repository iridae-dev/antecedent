//! Longitudinal regime value contract for one outcome row per subject.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;
use std::sync::Arc;

/// Point estimator for a prescribed subject-history regime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LongitudinalRegimeMethod {
    /// Sequential inverse-probability value using realized trajectories.
    Ipw,
    /// Plug-in g-formula using supplied period reward predictions.
    GFormula,
    /// Backward-recursive augmented value using supplied subject-owned Q scores.
    SequentialDoublyRobust,
}

/// Prespecified binary static or history-dependent actions, resolved before
/// analysis on the frozen subject histories. Nuisance probabilities are supplied
/// by the caller and subject-level prediction ownership remains explicit.
#[derive(Clone, Debug, PartialEq)]
pub struct LongitudinalRegimeQuery {
    /// Complete finite endpoint outcome column, one row per subject.
    pub outcome: VariableId,
    /// Evaluation method; both routes are point-only.
    pub method: LongitudinalRegimeMethod,
    /// Prespecified conditional period rewards under the regime, subject-major.
    /// Required for g-formula and absent for IPW.
    pub period_outcome_predictions: Option<Arc<[f64]>>,
    /// Period Q scores for sequential augmentation.
    pub q_predictions: Option<Arc<[f64]>>,
    /// Whether the subject remains observed at each decision.
    pub observation_history: Option<Arc<[bool]>>,
    /// Fold used to generate each subject's Q trajectory.
    pub prediction_fold_ids: Option<Arc<[u32]>>,
    /// Number of treatment decisions per subject.
    pub periods: usize,
    /// Realized binary actions, flattened subject-major.
    pub treatment_history: Arc<[bool]>,
    /// Prescribed regime actions, flattened subject-major. This permits a
    /// history-dependent policy resolved using each subject's predecision history.
    pub regime_actions: Arc<[bool]>,
    /// Conditional probability of observed action 1, flattened subject-major.
    pub treatment_probabilities: Arc<[f64]>,
    /// Conditional probability of remaining uncensored at each decision.
    pub censoring_probabilities: Arc<[f64]>,
    /// Whether the endpoint outcome was observed for each subject.
    pub outcome_observed: Arc<[bool]>,
    /// Unique IDs, one per outcome row, binding whole histories to a fold.
    pub subject_ids: Arc<[Arc<str>]>,
    /// One fold ID per subject. A subject's entire history has one owner.
    pub fold_ids: Arc<[u32]>,
    /// Caller declaration that each nuisance prediction excludes its subject's fold.
    pub excluded_fold_predictions: bool,
    /// Probabilities are known from a sequential randomization mechanism.
    pub probabilities_known_by_design: bool,
    /// Minimum accepted conditional action and censoring probability.
    pub minimum_probability: f64,
}

impl LongitudinalRegimeQuery {
    /// Validate shape, subject ownership, and sequential positivity metadata.
    pub fn validate(&self) -> Result<(), QueryError> {
        let n = self.subject_ids.len();
        let cells = n
            .checked_mul(self.periods)
            .ok_or_else(|| QueryError::InvalidLongitudinalRegime("dimensions overflow".into()))?;
        if n < 2
            || self.periods == 0
            || self.treatment_history.len() != cells
            || self.regime_actions.len() != cells
            || self.treatment_probabilities.len() != cells
            || self.censoring_probabilities.len() != cells
            || self.outcome_observed.len() != n
            || self.fold_ids.len() != n
        {
            return Err(QueryError::InvalidLongitudinalRegime(
                "histories, probabilities, outcomes, subjects, and folds must align".into(),
            ));
        }
        match (self.method, &self.period_outcome_predictions) {
            (LongitudinalRegimeMethod::Ipw, None) => {}
            (LongitudinalRegimeMethod::GFormula, Some(q))
                if q.len() == cells && q.iter().all(|v| v.is_finite()) => {}
            (LongitudinalRegimeMethod::SequentialDoublyRobust, None)
                if self.q_predictions.as_ref().is_some_and(|q| q.len() == cells && q.iter().all(|v| v.is_finite()))
                    && self.observation_history.as_ref().is_some_and(|o| o.len() == cells)
                    && self.prediction_fold_ids.as_ref().is_some_and(|f| f.as_ref() == self.fold_ids.as_ref())
                    && self.excluded_fold_predictions
                    && self.fold_ids.iter().collect::<std::collections::HashSet<_>>().len() >= 2
                    && self.outcome_observed.iter().any(|&x| x) => {}
            _ => return Err(QueryError::InvalidLongitudinalRegime(
                "method requires aligned finite predictions, observation history, and subject-owned folds"
                    .into(),
            )),
        }
        if self.method != LongitudinalRegimeMethod::SequentialDoublyRobust
            && (self.q_predictions.is_some() || self.observation_history.is_some() || self.prediction_fold_ids.is_some()) {
            return Err(QueryError::InvalidLongitudinalRegime("sequential DR fields require the sequential doubly robust method".into()));
        }
        if let Some(observed) = &self.observation_history {
            for i in 0..n {
                if observed[i * self.periods + self.periods - 1] != self.outcome_observed[i]
                    || (1..self.periods).any(|t| !observed[i * self.periods + t - 1] && observed[i * self.periods + t]) {
                    return Err(QueryError::InvalidLongitudinalRegime("observation history must be monotone and agree with endpoint observation".into()));
                }
            }
        }
        if self.subject_ids.iter().any(|id| id.trim().is_empty())
            || self.subject_ids.iter().collect::<std::collections::HashSet<_>>().len() != n
        {
            return Err(QueryError::InvalidLongitudinalRegime(
                "subject IDs must be unique and nonempty; one outcome row owns the whole history"
                    .into(),
            ));
        }
        if !self.probabilities_known_by_design && !self.excluded_fold_predictions {
            return Err(QueryError::InvalidLongitudinalRegime("probabilities require a known sequential design or subject-level excluded-fold ownership".into()));
        }
        if self.excluded_fold_predictions
            && self.fold_ids.iter().collect::<std::collections::HashSet<_>>().len() < 2
        {
            return Err(QueryError::InvalidLongitudinalRegime(
                "excluded-fold prediction ownership requires at least two subject folds".into(),
            ));
        }
        let floor = self.minimum_probability;
        if !floor.is_finite()
            || !(0.0 < floor && floor <= 0.5)
            || self
                .treatment_probabilities
                .iter()
                .any(|p| !p.is_finite() || *p < floor || *p > 1.0 - floor)
            || self.censoring_probabilities.iter().any(|p| !p.is_finite() || *p < floor || *p > 1.0)
        {
            return Err(QueryError::InvalidLongitudinalRegime(
                "sequential treatment or censoring positivity fails at the declared floor".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_split_subject_ownership_and_bad_probability() {
        let mut q = LongitudinalRegimeQuery {
            outcome: VariableId::from_raw(0),
            method: LongitudinalRegimeMethod::Ipw,
            period_outcome_predictions: None,
            q_predictions: None,
            observation_history: None,
            prediction_fold_ids: None,
            periods: 2,
            treatment_history: Arc::from([true, true, false, false]),
            regime_actions: Arc::from([true, true, true, true]),
            treatment_probabilities: Arc::from([0.5; 4]),
            censoring_probabilities: Arc::from([1.0; 4]),
            outcome_observed: Arc::from([true; 2]),
            subject_ids: Arc::from([Arc::<str>::from("a"), Arc::<str>::from("b")]),
            fold_ids: Arc::from([0, 1]),
            excluded_fold_predictions: true,
            probabilities_known_by_design: false,
            minimum_probability: 0.01,
        };
        assert!(q.validate().is_ok());
        q.subject_ids = Arc::from([Arc::<str>::from("a"), Arc::<str>::from("a")]);
        assert!(q.validate().is_err());
        q.subject_ids = Arc::from([Arc::<str>::from("a"), Arc::<str>::from("b")]);
        q.treatment_probabilities = Arc::from([0.0, 0.5, 0.5, 0.5]);
        assert!(q.validate().is_err());
    }
}
