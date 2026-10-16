//! The declared whole procedure of an adaptive experiment: assignment rule,
//! enrolment schedule, outcome observation (delay and loss), exposure
//! (exploration failures), stopping rule and final selection.
//!
//! The uncertainty methods of this module tree are derived for the complete
//! procedure, so every element that shapes the data-collection process must be
//! stated. A [`ProcedureDeclaration`] holds each element as an `Option` so a
//! missing element is representable and refused with its own detail
//! ([`ContractGap`]); nothing is defaulted. A validated declaration is a
//! [`ProcedureSpec`], the only input of the law, simulation and uncertainty
//! routes.
//!
//! The procedure is a batched sequential design. Looks `0..L` follow one
//! another; at look `j` the policy (a function of the *observed* history
//! alone) fixes its distribution over arms, `enrolment[j]` units are
//! enrolled under it, the outcomes that have aged `lag_looks` looks are
//! revealed, and the stopping rule is evaluated on the observed counts. One
//! unit per look is a fully sequential design; larger batches are batched
//! adaptive designs; a zero enrolment is a follow-up look that only reveals
//! pending outcomes.
//!
//! Outcomes are Bernoulli. Contexts, when more than one, are discrete with
//! known probabilities and the policy runs independently within each context
//! (stratified bandit).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::group_sequential::Sidedness;

/// Most arms of one procedure.
pub const MAX_PROCEDURE_ARMS: usize = 4;
/// Most contexts of one procedure.
pub const MAX_PROCEDURE_CONTEXTS: usize = 4;
/// Most arm-by-context cells of one procedure.
pub const MAX_PROCEDURE_CELLS: usize = 8;
/// Most looks of one procedure.
pub const MAX_PROCEDURE_LOOKS: usize = 24;
/// Most units one procedure enrols in total.
pub const MAX_PROCEDURE_UNITS: u32 = 400;
/// Longest outcome delay, in looks.
pub const MAX_OUTCOME_LAG: usize = 4;

/// Tolerance for probability vectors summing to one.
const SUM_TOLERANCE: f64 = 1e-9;

/// A `Beta(alpha, beta)` prior of one Bernoulli cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaPrior {
    /// First shape, at least one.
    pub alpha: f64,
    /// Second shape, at least one.
    pub beta: f64,
}

impl BetaPrior {
    /// The uniform prior.
    pub const UNIFORM: Self = Self { alpha: 1.0, beta: 1.0 };

    pub(crate) fn is_valid(self) -> bool {
        self.alpha.is_finite() && self.beta.is_finite() && self.alpha >= 1.0 && self.beta >= 1.0
    }
}

/// How units are assigned to arms, as a function of the observed history.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum AssignmentRule {
    /// Probabilities fixed in advance.
    Fixed {
        /// One probability per arm.
        probabilities: Vec<f64>,
    },
    /// Epsilon-greedy over prior-shrunk running arm means. Arms tied for the
    /// highest mean share the greedy mass equally.
    EpsilonGreedy {
        /// Exploration probability in `[0, 1]`.
        epsilon: f64,
        /// Prior that shrinks the running means.
        prior: BetaPrior,
    },
    /// Beta-Bernoulli Thompson sampling: arm probabilities are the posterior
    /// probabilities that each arm is best.
    ThompsonBeta {
        /// Prior of every arm.
        prior: BetaPrior,
    },
    /// Deterministic UCB1: an untried arm first (lowest index), otherwise the
    /// arm maximizing `mean + sqrt(exploration ln(t) / n)`, ties to the lowest
    /// index. The assignment probabilities are zero or one.
    UpperConfidenceBound {
        /// Exploration constant, positive (UCB1 uses 2).
        exploration: f64,
    },
}

impl AssignmentRule {
    /// Whether the rule depends on the observed history.
    #[must_use]
    pub const fn is_adaptive(&self) -> bool {
        !matches!(self, Self::Fixed { .. })
    }
}

/// How outcome availability relates to the outcome values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelayMechanism {
    /// No delay and no loss.
    Immediate,
    /// A delay and loss that do not depend on the outcome value.
    IgnorableAdministrative,
    /// Whether an outcome is ever observed depends on its value.
    OutcomeDependent,
}

impl DelayMechanism {
    /// Stable `snake_case` name (the contract's vocabulary).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Immediate => "immediate",
            Self::IgnorableAdministrative => "ignorable_administrative",
            Self::OutcomeDependent => "outcome_dependent",
        }
    }
}

/// When and whether each enrolled unit's outcome is observed.
///
/// A unit enrolled at look `j` has its outcome revealed at look
/// `j + lag_looks`. A revealed success is observed with probability
/// `retain_success` and a failure with `retain_failure`; an unobserved
/// outcome is lost for good.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutcomeObservation {
    /// Delay in looks.
    pub lag_looks: usize,
    /// Probability that a success is observed.
    pub retain_success: f64,
    /// Probability that a failure is observed.
    pub retain_failure: f64,
}

impl OutcomeObservation {
    /// No delay and no loss.
    pub const COMPLETE: Self = Self { lag_looks: 0, retain_success: 1.0, retain_failure: 1.0 };

    /// A delay of `lag_looks` looks that does not depend on the outcome.
    #[must_use]
    pub const fn delayed(lag_looks: usize) -> Self {
        Self { lag_looks, retain_success: 1.0, retain_failure: 1.0 }
    }

    /// Which mechanism this observation model is.
    #[must_use]
    pub fn mechanism(&self) -> DelayMechanism {
        if self.retain_success.to_bits() != self.retain_failure.to_bits() {
            DelayMechanism::OutcomeDependent
        } else if self.lag_looks == 0 && self.retain_success.to_bits() == 1.0_f64.to_bits() {
            DelayMechanism::Immediate
        } else {
            DelayMechanism::IgnorableAdministrative
        }
    }
}

/// Which arm an observed outcome is attributed to when exposure can differ
/// from assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attribution {
    /// The assigned arm (intention to treat).
    Assigned,
    /// The arm actually delivered.
    Delivered,
}

/// Exploration failures: a unit assigned to an arm may receive another arm.
#[derive(Clone, Debug, PartialEq)]
pub struct ExposureModel {
    /// Per assigned arm, the probability that the arm is not delivered and the
    /// fallback arm is delivered instead.
    pub failure_probability: Vec<f64>,
    /// The arm delivered on a failure.
    pub fallback_arm: usize,
    /// Which arm the analysis attributes the outcome to.
    pub attribution: Attribution,
}

impl ExposureModel {
    /// Every assigned arm is delivered.
    #[must_use]
    pub fn none(arms: usize) -> Self {
        Self {
            failure_probability: vec![0.0; arms],
            fallback_arm: 0,
            attribution: Attribution::Delivered,
        }
    }

    pub(crate) fn has_failures(&self) -> bool {
        self.failure_probability.iter().any(|&p| p > 0.0)
    }
}

/// What a look can decide.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum StoppingRule {
    /// Nothing: the planned horizon is run.
    NoStopping,
    /// Stop when the posterior probability that an arm is best reaches the
    /// threshold; the decision is that arm.
    PosteriorProbabilityBest {
        /// Threshold in `(0.5, 1)`.
        threshold: f64,
        /// Prior of every arm.
        prior: BetaPrior,
    },
    /// Stop when the smallest posterior expected loss of choosing an arm falls
    /// below the threshold; the decision is that arm.
    PosteriorExpectedLoss {
        /// Loss threshold, in units of the success probability.
        loss_threshold: f64,
        /// Prior of every arm.
        prior: BetaPrior,
    },
    /// Stop when the unpooled Wald statistic of arm 1 against arm 0 crosses a
    /// per-look critical value (two arms, one context). The statistic is
    /// undefined when an arm has no observation or the standard error is
    /// zero, and the trial then continues.
    WaldBoundaries {
        /// One critical value per look.
        critical: Vec<f64>,
        /// Which crossings stop the trial.
        sidedness: Sidedness,
    },
    /// Stop at the first look at which the enrolled units reach the cap.
    ResourceCap {
        /// Units at which the trial stops.
        max_units: u32,
    },
}

/// How the procedure selects an arm when no stopping rule decided.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum FinalSelection {
    /// No arm is selected.
    NoSelection,
    /// The arm with the highest observed mean (arms with no observation rank
    /// last; ties to the lowest index).
    HighestObservedMean,
    /// The arm with the highest posterior mean under the prior.
    HighestPosteriorMean(BetaPrior),
}

/// Why a procedure declaration was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContractGap {
    /// No assignment rule (policy history and realized probabilities).
    AssignmentPolicyMissing,
    /// No enrolment schedule (the interim looks).
    EnrolmentScheduleMissing,
    /// No outcome availability and delay model.
    OutcomeObservationMissing,
    /// No exposure model.
    ExposureModelMissing,
    /// No stopping rule.
    StoppingRuleMissing,
    /// No statement of how the final arm is selected.
    FinalSelectionMissing,
    /// No outcome structure of a unit with several measurements.
    OutcomeStructureMissing,
    /// No declared within-unit correlation of repeated measures.
    WithinUnitCorrelationMissing,
    /// No declared between-unit variance of a clustered outcome.
    BetweenUnitVarianceMissing,
    /// No declared within-unit variance of a clustered outcome.
    WithinUnitVarianceMissing,
    /// No declared final-analysis estimator.
    FinalAnalysisMissing,
    /// An element is present and not valid; the slug names which.
    Invalid(&'static str),
}

impl ContractGap {
    /// Namespaced detail of the refusal.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::AssignmentPolicyMissing => "adaptive_inference.policy_history_missing",
            Self::EnrolmentScheduleMissing => "adaptive_inference.interim_analyses_missing",
            Self::OutcomeObservationMissing => "adaptive_inference.outcome_availability_missing",
            Self::ExposureModelMissing => "adaptive_inference.exposure_history_missing",
            Self::StoppingRuleMissing => "adaptive_inference.stopping_rule_missing",
            Self::FinalSelectionMissing => "adaptive_inference.policy_selection_missing",
            Self::OutcomeStructureMissing => "adaptive_inference.outcome_structure_missing",
            Self::WithinUnitCorrelationMissing => {
                "adaptive_inference.within_unit_correlation_missing"
            }
            Self::BetweenUnitVarianceMissing => "adaptive_inference.between_unit_variance_missing",
            Self::WithinUnitVarianceMissing => "adaptive_inference.within_unit_variance_missing",
            Self::FinalAnalysisMissing => "adaptive_inference.final_analysis_missing",
            Self::Invalid(detail) => detail,
        }
    }

    /// Registered reason code of the refusal.
    #[must_use]
    pub fn reason_code(self) -> &'static str {
        if matches!(self, Self::Invalid(_)) {
            antecedent_core::reason_code!("invalid_argument")
        } else {
            antecedent_core::reason_code!("adaptive_contract_incomplete")
        }
    }

    /// What would license the declaration.
    #[must_use]
    pub const fn remedy(self) -> &'static str {
        match self {
            Self::AssignmentPolicyMissing => {
                "declare the assignment rule that governed every unit, as a function of the observed history"
            }
            Self::EnrolmentScheduleMissing => {
                "declare the interim looks: the units enrolled between consecutive looks"
            }
            Self::OutcomeObservationMissing => {
                "declare when outcomes become observable and how loss relates to the outcome value"
            }
            Self::ExposureModelMissing => {
                "declare the exposure model: the probability that an assigned arm is not delivered and what is delivered instead"
            }
            Self::StoppingRuleMissing => {
                "declare the stopping rule, its parameters and what it decides at each look"
            }
            Self::FinalSelectionMissing => {
                "declare how the procedure selects an arm when no stopping rule decided, or declare no selection"
            }
            Self::OutcomeStructureMissing => {
                "declare the outcome structure of a unit: the measurements per unit and their dependence"
            }
            Self::WithinUnitCorrelationMissing => {
                "declare the within-unit correlation of the repeated measures (compound symmetric or autoregressive), or declare independence explicitly with a correlation of zero"
            }
            Self::BetweenUnitVarianceMissing => {
                "declare the between-unit variance of the unit-level random effect (zero declares no clustering)"
            }
            Self::WithinUnitVarianceMissing => {
                "declare the within-unit variance of the clustered outcome"
            }
            Self::FinalAnalysisMissing => {
                "declare the final-analysis estimator: a cluster-robust contrast of unit summaries, or the adaptively weighted score"
            }
            Self::Invalid(_) => "correct the declared element",
        }
    }
}

/// A declaration that was refused: one gap per defect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcedureContractError {
    /// The gaps, in the order they were found.
    pub gaps: Vec<ContractGap>,
}

impl std::fmt::Display for ProcedureContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let details: Vec<&str> = self.gaps.iter().map(|g| g.detail()).collect();
        write!(f, "adaptive procedure declaration refused: {}", details.join(", "))
    }
}

impl std::error::Error for ProcedureContractError {}

/// Everything a producer states about the data-collection procedure.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcedureDeclaration {
    /// Number of arms.
    pub arms: usize,
    /// Probabilities of the discrete contexts; `None` is a single context.
    pub context_probabilities: Option<Vec<f64>>,
    /// The assignment rule.
    pub assignment: Option<AssignmentRule>,
    /// Units enrolled between consecutive looks, one entry per look.
    pub enrolment: Option<Vec<u32>>,
    /// Outcome availability, delay and loss.
    pub observation: Option<OutcomeObservation>,
    /// Exposure failures.
    pub exposure: Option<ExposureModel>,
    /// The stopping rule.
    pub stopping: Option<StoppingRule>,
    /// How the final arm is selected.
    pub selection: Option<FinalSelection>,
}

/// A validated procedure.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcedureSpec {
    arms: usize,
    contexts: Vec<f64>,
    assignment: AssignmentRule,
    enrolment: Vec<u32>,
    observation: OutcomeObservation,
    exposure: ExposureModel,
    stopping: StoppingRule,
    selection: FinalSelection,
}

fn probability_vector(values: &[f64]) -> bool {
    values.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p))
        && (values.iter().sum::<f64>() - 1.0).abs() <= SUM_TOLERANCE
}

impl ProcedureSpec {
    /// Validate a declaration.
    ///
    /// # Errors
    /// One gap per missing element and per invalid element; nothing is
    /// defaulted.
    #[allow(
        clippy::too_many_lines,
        reason = "one flat list of the element checks keeps each refusal next to its condition"
    )]
    pub fn new(declaration: ProcedureDeclaration) -> Result<Self, ProcedureContractError> {
        let mut gaps = Vec::new();
        if declaration.assignment.is_none() {
            gaps.push(ContractGap::AssignmentPolicyMissing);
        }
        if declaration.enrolment.is_none() {
            gaps.push(ContractGap::EnrolmentScheduleMissing);
        }
        if declaration.observation.is_none() {
            gaps.push(ContractGap::OutcomeObservationMissing);
        }
        if declaration.exposure.is_none() {
            gaps.push(ContractGap::ExposureModelMissing);
        }
        if declaration.stopping.is_none() {
            gaps.push(ContractGap::StoppingRuleMissing);
        }
        if declaration.selection.is_none() {
            gaps.push(ContractGap::FinalSelectionMissing);
        }
        let arms = declaration.arms;
        let contexts = declaration.context_probabilities.clone().unwrap_or_else(|| vec![1.0]);
        if !(2..=MAX_PROCEDURE_ARMS).contains(&arms) {
            gaps.push(ContractGap::Invalid("adaptive_inference.arms_invalid"));
        }
        if contexts.is_empty()
            || contexts.len() > MAX_PROCEDURE_CONTEXTS
            || contexts.iter().any(|&p| !(p.is_finite() && p > 0.0))
            || !probability_vector(&contexts)
            || contexts.len() * arms > MAX_PROCEDURE_CELLS
        {
            gaps.push(ContractGap::Invalid("adaptive_inference.contexts_invalid"));
        }
        let single_context = contexts.len() == 1;
        if let Some(rule) = &declaration.assignment {
            let valid = match rule {
                AssignmentRule::Fixed { probabilities } => {
                    probabilities.len() == arms && probability_vector(probabilities)
                }
                AssignmentRule::EpsilonGreedy { epsilon, prior } => {
                    epsilon.is_finite() && (0.0..=1.0).contains(epsilon) && prior.is_valid()
                }
                AssignmentRule::ThompsonBeta { prior } => prior.is_valid(),
                AssignmentRule::UpperConfidenceBound { exploration } => {
                    exploration.is_finite() && *exploration > 0.0
                }
            };
            if !valid {
                gaps.push(ContractGap::Invalid("adaptive_inference.rule_mismatch"));
            }
        }
        let mut looks = 0;
        if let Some(schedule) = &declaration.enrolment {
            looks = schedule.len();
            let total: u64 = schedule.iter().map(|&n| u64::from(n)).sum();
            let ordered = !schedule.is_empty()
                && schedule.len() <= MAX_PROCEDURE_LOOKS
                && total > 0
                && total <= u64::from(MAX_PROCEDURE_UNITS)
                && schedule[0] > 0;
            if !ordered {
                gaps.push(ContractGap::Invalid("adaptive_inference.looks_not_ordered"));
            }
        }
        if let Some(observation) = &declaration.observation {
            let retention = |p: f64| p.is_finite() && p > 0.0 && p <= 1.0;
            if observation.lag_looks > MAX_OUTCOME_LAG
                || !retention(observation.retain_success)
                || !retention(observation.retain_failure)
            {
                gaps.push(ContractGap::Invalid("adaptive_inference.outcome_availability_invalid"));
            }
        }
        if let Some(exposure) = &declaration.exposure {
            if exposure.failure_probability.len() != arms
                || exposure
                    .failure_probability
                    .iter()
                    .any(|p| !(p.is_finite() && (0.0..1.0).contains(p)))
                || exposure.fallback_arm >= arms
            {
                gaps.push(ContractGap::Invalid("adaptive_inference.exposure_invalid"));
            }
        }
        if let Some(rule) = &declaration.stopping {
            let valid = match rule {
                StoppingRule::NoStopping => true,
                StoppingRule::PosteriorProbabilityBest { threshold, prior } => {
                    single_context && *threshold > 0.5 && *threshold < 1.0 && prior.is_valid()
                }
                StoppingRule::PosteriorExpectedLoss { loss_threshold, prior } => {
                    single_context
                        && loss_threshold.is_finite()
                        && *loss_threshold > 0.0
                        && prior.is_valid()
                }
                StoppingRule::WaldBoundaries { critical, .. } => {
                    single_context
                        && arms == 2
                        && critical.len() == looks
                        && critical.iter().all(|c| c.is_finite())
                }
                StoppingRule::ResourceCap { max_units } => *max_units > 0,
            };
            if !valid {
                gaps.push(ContractGap::Invalid("adaptive_inference.rule_mismatch"));
            }
        }
        if let Some(selection) = &declaration.selection {
            let valid = match selection {
                FinalSelection::NoSelection => true,
                FinalSelection::HighestObservedMean => single_context,
                FinalSelection::HighestPosteriorMean(prior) => single_context && prior.is_valid(),
            };
            if !valid {
                gaps.push(ContractGap::Invalid("adaptive_inference.rule_mismatch"));
            }
        }
        if !gaps.is_empty() {
            return Err(ProcedureContractError { gaps });
        }
        // Every option is present once the gap list is empty.
        match (
            declaration.assignment,
            declaration.enrolment,
            declaration.observation,
            declaration.exposure,
            declaration.stopping,
            declaration.selection,
        ) {
            (
                Some(assignment),
                Some(enrolment),
                Some(observation),
                Some(exposure),
                Some(stopping),
                Some(selection),
            ) => Ok(Self {
                arms,
                contexts,
                assignment,
                enrolment,
                observation,
                exposure,
                stopping,
                selection,
            }),
            _ => Err(ProcedureContractError { gaps }),
        }
    }

    /// Number of arms.
    #[must_use]
    pub const fn arms(&self) -> usize {
        self.arms
    }

    /// Context probabilities (one entry for a single context).
    #[must_use]
    pub fn contexts(&self) -> &[f64] {
        &self.contexts
    }

    /// Arm-by-context cells, context-major: cell `c * arms + a`.
    #[must_use]
    pub fn cells(&self) -> usize {
        self.arms * self.contexts.len()
    }

    /// The assignment rule.
    #[must_use]
    pub const fn assignment(&self) -> &AssignmentRule {
        &self.assignment
    }

    /// Units enrolled at each look.
    #[must_use]
    pub fn enrolment(&self) -> &[u32] {
        &self.enrolment
    }

    /// Cumulative units enrolled through each look.
    #[must_use]
    pub fn cumulative_enrolment(&self) -> Vec<u32> {
        self.enrolment
            .iter()
            .scan(0_u32, |total, &n| {
                *total += n;
                Some(*total)
            })
            .collect()
    }

    /// Outcome observation model.
    #[must_use]
    pub const fn observation(&self) -> &OutcomeObservation {
        &self.observation
    }

    /// Exposure model.
    #[must_use]
    pub const fn exposure(&self) -> &ExposureModel {
        &self.exposure
    }

    /// Stopping rule.
    #[must_use]
    pub const fn stopping(&self) -> &StoppingRule {
        &self.stopping
    }

    /// Final selection.
    #[must_use]
    pub const fn selection(&self) -> &FinalSelection {
        &self.selection
    }

    /// The assignment-family name of the procedure matrix this procedure
    /// belongs to: contexts make a contextual bandit, an adaptive rule with
    /// batches larger than one is a batched adaptive design.
    #[must_use]
    pub fn family_name(&self) -> &'static str {
        if self.contexts.len() > 1 {
            return "contextual_bandit";
        }
        let batched = self.enrolment.iter().any(|&n| n > 1);
        match &self.assignment {
            AssignmentRule::Fixed { .. } => "fixed_randomization",
            _ if batched => "batched_adaptive",
            AssignmentRule::EpsilonGreedy { .. } => "epsilon_greedy",
            AssignmentRule::ThompsonBeta { .. } => "thompson_beta_bernoulli",
            AssignmentRule::UpperConfidenceBound { .. } => "upper_confidence_bound",
        }
    }

    /// The monitoring-strategy name of the procedure matrix.
    #[must_use]
    pub fn monitoring_name(&self) -> &'static str {
        if self.enrolment.len() == 1 {
            "final_analysis_only"
        } else if matches!(self.stopping, StoppingRule::WaldBoundaries { .. }) {
            "group_sequential"
        } else if self.enrolment.iter().all(|&n| n == 1) {
            "continuous_monitoring"
        } else {
            "fixed_interim_looks"
        }
    }

    /// The stopping-kind name of the procedure matrix.
    #[must_use]
    pub const fn stopping_name(&self) -> &'static str {
        match &self.stopping {
            StoppingRule::NoStopping => "no_stopping",
            StoppingRule::PosteriorProbabilityBest { .. } | StoppingRule::WaldBoundaries { .. } => {
                "efficacy"
            }
            StoppingRule::PosteriorExpectedLoss { .. } => "decision_theoretic",
            StoppingRule::ResourceCap { .. } => "resource",
        }
    }
}
