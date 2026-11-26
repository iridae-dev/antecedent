//! Longitudinal regime value contract for one outcome row per subject.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use super::fault::LongitudinalRegimeFault;
use crate::canonical_encode::{CanonicalEncode, count, encode_flags, encode_u32s, flag};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;
use std::sync::Arc;

/// Point estimator for a prescribed subject-history regime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LongitudinalRegimeMethod {
    /// Sequential inverse-probability value using realized trajectories.
    Ipw,
    /// Plug-in g-formula using supplied period reward predictions.
    GFormula,
    /// Fit a baseline-stratified terminal Q model and refit it in each subject bootstrap.
    FittedGFormula(FittedRegimeQ),
    /// Compare a declared class of schedules or learn one on a separate subject fold.
    ScheduleFamily(LongitudinalScheduleFamily),
    /// Backward-recursive augmented value using supplied subject-owned Q scores.
    SequentialDoublyRobust,
    /// Additive binary marginal structural mean with stabilized IPTW.
    MarginalStructuralModel,
}

/// Compiler-owned two-period cell-mean Q regression and its resampling design.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FittedRegimeQ {
    /// Complete binary pre-treatment baseline column.
    pub baseline: VariableId,
    /// If set, this fold is an independent external training sample and the other fold is the target.
    /// Otherwise each fold is held out in turn and scored using the other folds.
    pub external_training_fold: Option<u32>,
    /// Number of whole-subject bootstrap replicates, preserving the declared folds.
    pub bootstrap_replicates: u32,
}

/// Two-decision policy class, declared before estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LongitudinalScheduleFamily {
    /// Bitmask of candidate schedules: bit k encodes the two ordered actions in k.
    pub schedules: u8,
    /// Fixed comparison coordinates or a held-out learning rule.
    pub target: LongitudinalScheduleTarget,
    /// Whole-subject multiplier draws and, for learning, training/evaluation bootstrap refits.
    pub draws: u32,
}

/// The declared reduction of a two-decision schedule family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LongitudinalScheduleTarget {
    /// Simultaneously compare a declared subset to its reference schedule.
    Contrasts {
        /// Two-bit reference schedule.
        reference: u8,
        /// Mask of the schedules contrasted with that reference.
        contrasts: u8,
    },
    /// Choose the maximum training score, with ties going to the smallest schedule code.
    Learn {
        /// Training subject fold; the other fold is an independent evaluation sample.
        training_fold: u32,
    },
}

impl CanonicalEncode for LongitudinalRegimeMethod {
    fn encode(&self, writer: &mut CanonicalWriter) {
        let tag = match self {
            Self::Ipw => 1,
            Self::GFormula => 2,
            Self::SequentialDoublyRobust => 3,
            Self::MarginalStructuralModel => 4,
            Self::FittedGFormula(fit) => {
                writer.word(5);
                fit.baseline.encode(writer);
                match fit.external_training_fold {
                    Some(fold) => {
                        writer.word(1).word(u64::from(fold));
                    }
                    None => {
                        writer.word(0);
                    }
                }
                writer.word(u64::from(fit.bootstrap_replicates));
                return;
            }
            Self::ScheduleFamily(family) => {
                writer.word(6).word(u64::from(family.schedules)).word(u64::from(family.draws));
                match family.target {
                    LongitudinalScheduleTarget::Contrasts { reference, contrasts } => {
                        writer.word(1).word(u64::from(reference)).word(u64::from(contrasts));
                    }
                    LongitudinalScheduleTarget::Learn { training_fold } => {
                        writer.word(2).word(u64::from(training_fold));
                    }
                }
                return;
            }
        };
        writer.word(tag);
    }
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
    /// Caller declaration that the g-formula reward law is prespecified,
    /// fixed, and known rather than fitted or selected on these subjects.
    pub known_fixed_outcome_predictions: bool,
    /// One marginal treatment-one numerator probability per period for MSM.
    pub stabilizing_numerator_probabilities: Option<Arc<[f64]>>,
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
    /// Stable caller identity of a Python history-adaptive rule, if used.
    pub rule_id: Option<Arc<str>>,
    /// Stable caller-declared rule version.
    pub rule_version: Option<Arc<str>>,
    /// Caller-declared source or provenance of the rule implementation.
    pub rule_provenance: Option<Arc<str>>,
}

/// Every field is written, the caller declarations and rule provenance
/// included. Every array is subject-, period- or subject-major-aligned and
/// keeps its order; an absent optional array is a presence word.
impl CanonicalEncode for LongitudinalRegimeQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.outcome.encode(writer);
        self.method.encode(writer);
        opt_reals(writer, self.period_outcome_predictions.as_deref());
        writer.word(flag(self.known_fixed_outcome_predictions));
        opt_reals(writer, self.stabilizing_numerator_probabilities.as_deref());
        opt_reals(writer, self.q_predictions.as_deref());
        match self.observation_history.as_deref() {
            Some(history) => {
                writer.word(1);
                encode_flags(writer, history);
            }
            None => {
                writer.word(0);
            }
        }
        match self.prediction_fold_ids.as_deref() {
            Some(folds) => {
                writer.word(1);
                encode_u32s(writer, folds);
            }
            None => {
                writer.word(0);
            }
        }
        writer.word(count(self.periods));
        encode_flags(writer, &self.treatment_history);
        encode_flags(writer, &self.regime_actions);
        writer.reals(&self.treatment_probabilities).reals(&self.censoring_probabilities);
        encode_flags(writer, &self.outcome_observed);
        writer.texts(&self.subject_ids);
        encode_u32s(writer, &self.fold_ids);
        writer
            .word(flag(self.excluded_fold_predictions))
            .word(flag(self.probabilities_known_by_design))
            .real(self.minimum_probability)
            .opt_text(self.rule_id.as_deref())
            .opt_text(self.rule_version.as_deref())
            .opt_text(self.rule_provenance.as_deref());
    }
}

/// An optional real array: a presence word, then the reals in order.
fn opt_reals(writer: &mut CanonicalWriter, values: Option<&[f64]>) {
    match values {
        Some(values) => {
            writer.word(1).reals(values);
        }
        None => {
            writer.word(0);
        }
    }
}

impl LongitudinalRegimeQuery {
    /// Validate shape, subject ownership, and sequential positivity metadata.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::InvalidLongitudinalRegime`] when the rule identity
    /// is partially specified or attached to an MSM, the trajectory shapes or
    /// subject ownership are misaligned, excluded-fold prediction ownership
    /// lacks at least two subject folds, or the positivity floor and declared
    /// treatment or censoring probabilities are non-finite or out of range.
    // A flat sequence of independent guard checks; splitting it would only scatter
    // the query's validity contract across helpers with no shared logic.
    #[allow(
        clippy::too_many_lines,
        reason = "one validation per query field, in declaration order"
    )]
    pub fn validate(&self) -> Result<(), QueryError> {
        let identity = [&self.rule_id, &self.rule_version, &self.rule_provenance];
        if identity.iter().any(|value| value.is_some())
            && (identity
                .iter()
                .any(|value| value.as_ref().is_none_or(|text| text.trim().is_empty()))
                || self.method == LongitudinalRegimeMethod::MarginalStructuralModel)
        {
            return Err(QueryError::InvalidLongitudinalRegime(
                LongitudinalRegimeFault::DynamicRuleIdentity,
            ));
        }
        let n = self.subject_ids.len();
        let cells = n.checked_mul(self.periods).ok_or(QueryError::InvalidLongitudinalRegime(
            LongitudinalRegimeFault::DimensionsOverflow,
        ))?;
        if n < 2
            || self.periods == 0
            || self.treatment_history.len() != cells
            || self.regime_actions.len() != cells
            || self.treatment_probabilities.len() != cells
            || self.censoring_probabilities.len() != cells
            || self.outcome_observed.len() != n
            || self.fold_ids.len() != n
        {
            return Err(QueryError::InvalidLongitudinalRegime(LongitudinalRegimeFault::Alignment));
        }
        match (self.method, &self.period_outcome_predictions) {
            (LongitudinalRegimeMethod::Ipw, None) => {}
            (LongitudinalRegimeMethod::GFormula, Some(q))
                if q.len() == cells && q.iter().all(|v| v.is_finite()) => {}
            (LongitudinalRegimeMethod::FittedGFormula(fit), None)
                if self.periods == 2
                    && fit.baseline != self.outcome
                    && self.rule_id.is_none()
                    && self.probabilities_known_by_design
                    && self.outcome_observed.iter().all(|&observed| observed)
                    && self
                        .censoring_probabilities
                        .iter()
                        .all(|p| p.to_bits() == 1_f64.to_bits())
                    && self
                        .treatment_probabilities
                        .iter()
                        .all(|p| p.to_bits() == 0.5_f64.to_bits())
                    && self
                        .regime_actions
                        .chunks_exact(2)
                        .all(|actions| actions == &self.regime_actions[..2])
                    && (99..=100_000).contains(&fit.bootstrap_replicates)
                    && fitted_fold_ownership(&self.fold_ids, fit.external_training_fold) => {}
            (LongitudinalRegimeMethod::ScheduleFamily(family), None)
                if self.periods == 2
                    && self.rule_id.is_none()
                    && self.probabilities_known_by_design
                    && self.outcome_observed.iter().all(|&observed| observed)
                    && self
                        .censoring_probabilities
                        .iter()
                        .all(|p| p.to_bits() == 1_f64.to_bits())
                    && self
                        .treatment_probabilities
                        .iter()
                        .all(|p| p.to_bits() == 0.5_f64.to_bits())
                    && self
                        .regime_actions
                        .chunks_exact(2)
                        .all(|actions| actions == &self.regime_actions[..2])
                    && family.schedules <= 0b1111
                    && family.schedules.count_ones() >= 2
                    && (199..=100_000).contains(&family.draws)
                    && self.valid_schedule_target(family) => {}
            (LongitudinalRegimeMethod::MarginalStructuralModel, None)
                if self.stabilizing_numerator_probabilities.as_ref().is_some_and(|v| {
                    v.len() == self.periods
                        && v.iter().all(|p| {
                            p.is_finite()
                                && *p >= self.minimum_probability
                                && *p <= 1.0 - self.minimum_probability
                        })
                }) && self.outcome_observed.iter().filter(|&&x| x).count()
                    > self.periods + 1 => {}
            (LongitudinalRegimeMethod::SequentialDoublyRobust, None)
                if self
                    .q_predictions
                    .as_ref()
                    .is_some_and(|q| q.len() == cells && q.iter().all(|v| v.is_finite()))
                    && self.observation_history.as_ref().is_some_and(|o| o.len() == cells)
                    && self
                        .prediction_fold_ids
                        .as_ref()
                        .is_some_and(|f| f.as_ref() == self.fold_ids.as_ref())
                    && self.excluded_fold_predictions
                    && self.fold_ids.iter().collect::<std::collections::HashSet<_>>().len()
                        >= 2
                    && self.outcome_observed.iter().any(|&x| x) => {}
            _ => {
                return Err(QueryError::InvalidLongitudinalRegime(
                    LongitudinalRegimeFault::MethodPredictions,
                ));
            }
        }
        if self.known_fixed_outcome_predictions && self.method != LongitudinalRegimeMethod::GFormula
        {
            return Err(QueryError::InvalidLongitudinalRegime(
                LongitudinalRegimeFault::FixedPredictionsNeedGFormula,
            ));
        }
        if self.method != LongitudinalRegimeMethod::SequentialDoublyRobust
            && (self.q_predictions.is_some()
                || self.observation_history.is_some()
                || self.prediction_fold_ids.is_some())
        {
            return Err(QueryError::InvalidLongitudinalRegime(
                LongitudinalRegimeFault::SequentialDrFields,
            ));
        }
        if self.method != LongitudinalRegimeMethod::MarginalStructuralModel
            && self.stabilizing_numerator_probabilities.is_some()
        {
            return Err(QueryError::InvalidLongitudinalRegime(
                LongitudinalRegimeFault::StabilizingNumerator,
            ));
        }
        if let Some(observed) = &self.observation_history {
            for i in 0..n {
                if observed[i * self.periods + self.periods - 1] != self.outcome_observed[i]
                    || (1..self.periods).any(|t| {
                        !observed[i * self.periods + t - 1] && observed[i * self.periods + t]
                    })
                {
                    return Err(QueryError::InvalidLongitudinalRegime(
                        LongitudinalRegimeFault::ObservationHistory,
                    ));
                }
            }
        }
        if self.subject_ids.iter().any(|id| id.trim().is_empty())
            || self.subject_ids.iter().collect::<std::collections::HashSet<_>>().len() != n
        {
            return Err(QueryError::InvalidLongitudinalRegime(LongitudinalRegimeFault::SubjectIds));
        }
        if !self.probabilities_known_by_design && !self.excluded_fold_predictions {
            return Err(QueryError::InvalidLongitudinalRegime(
                LongitudinalRegimeFault::ProbabilityOwnership,
            ));
        }
        if self.excluded_fold_predictions
            && self.fold_ids.iter().collect::<std::collections::HashSet<_>>().len() < 2
        {
            return Err(QueryError::InvalidLongitudinalRegime(
                LongitudinalRegimeFault::ExcludedFoldCount,
            ));
        }
        let floor = self.minimum_probability;
        if !(floor.is_finite() && 0.0 < floor && floor <= 0.5)
            || self
                .treatment_probabilities
                .iter()
                .any(|p| !p.is_finite() || *p < floor || *p > 1.0 - floor)
            || self.censoring_probabilities.iter().any(|p| !p.is_finite() || *p < floor || *p > 1.0)
        {
            return Err(QueryError::InvalidLongitudinalRegime(
                LongitudinalRegimeFault::PositivityFloor,
            ));
        }
        Ok(())
    }

    fn valid_schedule_target(&self, family: LongitudinalScheduleFamily) -> bool {
        match family.target {
            LongitudinalScheduleTarget::Contrasts { reference, contrasts } => {
                reference < 4
                    && family.schedules & (1 << reference) != 0
                    && contrasts != 0
                    && contrasts & !family.schedules == 0
                    && contrasts & (1 << reference) == 0
                    && self.regime_actions[..2] == [reference & 2 != 0, reference & 1 != 0]
            }
            LongitudinalScheduleTarget::Learn { training_fold } => {
                fitted_fold_ownership(&self.fold_ids, Some(training_fold))
            }
        }
    }
}

fn fitted_fold_ownership(folds: &[u32], external: Option<u32>) -> bool {
    let assigned: std::collections::HashSet<_> = folds.iter().copied().collect();
    match external {
        Some(training_fold) => assigned.len() == 2 && assigned.contains(&training_fold),
        None => assigned.len() >= 2,
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
            known_fixed_outcome_predictions: false,
            stabilizing_numerator_probabilities: None,
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
            rule_id: None,
            rule_version: None,
            rule_provenance: None,
        };
        assert!(q.validate().is_ok());
        q.subject_ids = Arc::from([Arc::<str>::from("a"), Arc::<str>::from("a")]);
        assert!(q.validate().is_err());
        q.subject_ids = Arc::from([Arc::<str>::from("a"), Arc::<str>::from("b")]);
        q.treatment_probabilities = Arc::from([0.0, 0.5, 0.5, 0.5]);
        assert!(q.validate().is_err());
    }
}
