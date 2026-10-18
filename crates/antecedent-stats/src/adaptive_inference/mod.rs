//! Numerics for inference from adaptively collected data.
//!
//! Four methods, one per inferential paradigm, each over a different part of
//! the retained assignment history. None of them analyses the final sample as
//! if the design had been fixed, and none reports a calibrated interval: every
//! interval-like number is labelled uncalibrated by its caller and the
//! interval route stays closed until whole-procedure coverage is measured.
//!
//! * [`beta_bernoulli`]: the conjugate posterior of Bernoulli arms, the
//!   posterior probability that an arm is best and the posterior expected loss
//!   of choosing an arm. The posterior is a function of the observed counts
//!   alone, so it is the same under any assignment rule that depends on the
//!   observed history and independent randomization only (the likelihood
//!   principle); that premise is the caller's to state.
//! * [`weighting`]: augmented inverse-probability scores with a *predictable*
//!   outcome model and predictable weights, combined by an adaptively weighted
//!   mean (Hadad, Hirshberg, Zhan, Wager and Athey 2021). Each score is a
//!   martingale difference under the assignment history, so the weighted sum
//!   is a martingale however the policy adapts.
//! * [`confidence_sequence`]: time-uniform confidence sequences for the mean
//!   of a `[0, 1]`-bounded arm, valid at every sample size at once and so
//!   under data-dependent stopping: a normal-mixture Hoeffding sequence
//!   (Robbins 1970; Howard, Ramdas, `McAuliffe` and Sekhon 2021) and the
//!   variance-adaptive predictable plug-in empirical Bernstein betting
//!   sequence (Waudby-Smith and Ramdas 2024).
//! * [`group_sequential`]: alpha-spending boundaries (Lan and `DeMets`) for a
//!   pre-specified set of interim looks, by recursive numerical integration of
//!   the canonical joint distribution of the interim statistics (Armitage,
//!   `McPherson` and Rowe 1969; Lan and `DeMets` 1983; Jennison and Turnbull 2000),
//!   and stagewise-ordered p-values and intervals for the stopped statistic.
//!
//! Those four are components. The uncertainty of a *procedure* is derived by
//! the `procedure_*` modules: [`procedure_contract`] declares the whole
//! data-collection procedure (policy, looks, delay and loss, exposure
//! failures, stopping, selection) and refuses a missing element;
//! [`procedure_law`] computes its exact law; [`procedure_simulation`] runs
//! it end to end for order-dependent estimators; [`procedure_uncertainty`]
//! derives each paradigm's statement from them, separately.
//! [`procedure_repeated_measures`] derives the same for units that contribute
//! several correlated measurements (repeated measures, or members of a
//! cluster with a unit-level random effect), under a declared within-unit
//! dependence, by simulation with an exact closed-form oracle.
//! [`participant_procedure`] is the participant-level family: several binary
//! responses per participant (a beta-binomial count with a declared or
//! jointly estimated correlation, and its exact posterior in
//! [`participant_posterior`]), arm priors that borrow declared historical sources
//! ([`historical_prior`]), Thompson allocation with a minimum exploration
//! probability, and expected-loss, futility and recruitment-cap decisions at
//! completed-participant batch boundaries, with a posterior claim and a
//! time-uniform confidence-sequence claim read at every look.
//! [`capacity`] holds the resource bounds of all of these routes, one class of
//! limit per refusal, and the receipt a simulation returns.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub mod beta_bernoulli;
pub mod capacity;
pub mod confidence_sequence;
pub mod group_sequential;
pub mod historical_prior;
pub mod participant_posterior;
pub mod participant_procedure;
pub mod procedure_contract;
pub mod procedure_law;
pub mod procedure_repeated_measures;
pub mod procedure_simulation;
pub mod procedure_uncertainty;
pub mod weighting;

use thiserror::Error;

pub use beta_bernoulli::{
    BetaPosterior, MAX_ARM_TRIALS, MAX_BANDIT_ARMS, best_arm_probabilities, expected_losses,
    probability_exceeds, probability_superior, regularized_incomplete_beta,
};
pub use capacity::{
    CapacityError, MAX_RETAINED_LOOKS, MAX_SIMULATION_MEMORY_BYTES, MAX_SIMULATION_WORK,
    ResourceBudget, ResourceReceipt,
};
pub use confidence_sequence::{
    ArmSequence, MAX_SEQUENCE_OBSERVATIONS, PLUG_IN_LAMBDA_CAP, SequenceKind, SequenceState,
    bounded_mean_confidence_sequence, contrast_bounds, mixture_hoeffding_radius,
    plug_in_confidence_sequence,
};
pub use group_sequential::{
    Boundaries, MAX_LOOKS, Sidedness, SpendingFunction, StagewiseInference, StagewiseLook,
    boundaries, crossing_probabilities, first_crossing, stagewise_inference,
    stagewise_upper_p_value,
};
pub use historical_prior::{
    ArmPrior, BorrowingPlan, HistoricalPrior, MAX_HISTORICAL_SOURCES, MAX_HISTORICAL_TRIALS,
    MAX_SOURCE_LABEL_BYTES, ResolvedArmPrior,
};
pub use participant_posterior::{
    ArmEvidence, CorrelationModel, CorrelationPrior, ExactPosterior, PosteriorRequest,
    exact_posterior,
};
pub use participant_procedure::{
    AnytimeSettings, ArmBorrowing, ArmReading, ComparisonReading, DecisionRules, FrequentistClaim,
    FutilityRule, LookDecision, LookReading, MAX_PARTICIPANT_LOOKS, MAX_PARTICIPANT_RESPONSES,
    PARTICIPANT_PROCEDURE_VERSION, ParticipantDeclaration, ParticipantHistory,
    ParticipantReplicate, ParticipantRun, ParticipantSimulation, ParticipantSimulationRequest,
    ParticipantSimulationSummary, ParticipantSpec, PosteriorClaim, PriorDataConflict,
    REPLAY_TOLERANCE, StopFractions, StopReason, beta_binomial_pmf, design_effect,
    run_participant_procedure_within, simulate_participant_history, simulate_participant_procedure,
    simulate_participant_procedure_within,
};
pub use procedure_contract::{
    AssignmentRule, Attribution, BetaPrior, ContractGap, DelayMechanism, ExposureModel,
    FinalSelection, MAX_OUTCOME_LAG, MAX_PROCEDURE_ARMS, MAX_PROCEDURE_CELLS,
    MAX_PROCEDURE_CONTEXTS, MAX_PROCEDURE_LOOKS, MAX_PROCEDURE_UNITS, OutcomeObservation,
    ProcedureContractError, ProcedureDeclaration, ProcedureSpec, StoppingRule,
};
pub use procedure_law::{
    MAX_LAW_STATES, MAX_LAW_WORK, OutcomeKernel, ProcedureLaw, TerminalAtom,
    apply_exploration_floor, enumerate_law,
};
pub use procedure_repeated_measures::{
    AnalysisEstimator, CellLabel, FinalAnalysis, GaussianMixtureLaw, MAX_MEASUREMENTS_PER_UNIT,
    MAX_UNIT_BATCHES, MAX_UNIT_LAG, MixtureComponent, OutcomeStructure,
    RepeatedMeasuresDeclaration, RepeatedMeasuresDerivation, RepeatedMeasuresLaw,
    RepeatedMeasuresReplicate, RepeatedMeasuresRequest, RepeatedMeasuresSpec,
    RepeatedMeasuresSummary, UnitAssignment, UnitSummary, WithinUnitCorrelation,
    admit_repeated_measures, derive_repeated_measures, derive_repeated_measures_within,
    exact_fixed_contrast_law, simulate_repeated_measures, simulate_repeated_measures_within,
};
pub use procedure_simulation::{
    MAX_SIMULATION_REPLICATIONS, SimulatedLaw, SimulatedReplicate, SimulationRequest,
    SimulationSummary, simulate_procedure, simulate_procedure_within,
};
pub use procedure_uncertainty::{
    AnytimeCellBound, AnytimeReport, BayesianReport, DERIVED_UNMEASURED, DecisionReport,
    EstimandStatement, EstimatorLaw, Functional, InversionGrid, InversionInterval,
    MAX_INVERSION_LAWS, ObservedTerminal, OrderingPValues, PosteriorMeanGap, UncertaintyParadigm,
    admit, anytime_exclusion_probability, anytime_valid_report, bayesian_report, decision_report,
    inversion_interval, ordering_p_values, posterior_mean_gaps,
};
pub use weighting::{
    AdaptiveEstimate, Estimand, MAX_ADAPTIVE_UNITS, MAX_WEIGHTED_ARMS, OutcomeModelPrior,
    WeightScheme, adaptive_scores, weighted_estimate,
};

/// Why an adaptive-inference routine refused its input.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum AdaptiveInferenceError {
    /// Lengths or counts do not agree, or a bound is exceeded.
    #[error("adaptive inference shape error: {0}")]
    Shape(&'static str),
    /// A parameter is outside its range or not finite.
    #[error("adaptive inference parameter is invalid: {0}")]
    InvalidParameter(&'static str),
    /// An arm of the estimand had assignment probability below the declared
    /// floor at a unit, so its inverse-probability score is not supported.
    #[error(
        "arm {arm} had assignment probability {probability} below the floor {floor} at unit {unit}"
    )]
    Positivity {
        /// Unit (enrollment order) at which the floor failed.
        unit: usize,
        /// Arm index.
        arm: usize,
        /// The probability the policy gave the arm.
        probability: f64,
        /// The declared floor.
        floor: f64,
    },
    /// A paradigm's validity premise does not hold for the declared
    /// procedure. `detail` is the namespaced `adaptive_inference.*` slug.
    #[error("adaptive inference premise failed ({detail}): {message}")]
    PremiseFailed {
        /// Namespaced detail slug.
        detail: &'static str,
        /// Which premise failed and why.
        message: &'static str,
    },
    /// A size or work bound was exceeded.
    #[error("adaptive inference limit exceeded: {0}")]
    Limit(&'static str),
    /// A resource bound of a procedure route was exceeded; the class of bound
    /// carries its own detail and remedy.
    #[error("adaptive inference capacity exceeded: {0}")]
    Capacity(#[from] CapacityError),
    /// An iteration or root search did not converge or left its bracket.
    #[error("adaptive inference numerical failure: {0}")]
    Numerical(&'static str),
    /// The caller cancelled the computation.
    #[error("adaptive inference was cancelled")]
    Cancelled,
}

/// Number of Gauss-Legendre nodes per integration panel.
pub(crate) const PANEL_NODES: usize = 12;

/// Gauss-Legendre rule shared by the panels of the quadrature routines.
pub(crate) fn panel_rule() -> (Vec<f64>, Vec<f64>) {
    crate::special::gauss_legendre(PANEL_NODES)
}

/// Nodes and weights of a composite Gauss-Legendre rule on `[lo, hi]` with
/// `panels` equal panels.
pub(crate) fn composite_rule(
    rule: &(Vec<f64>, Vec<f64>),
    lo: f64,
    hi: f64,
    panels: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = Vec::with_capacity(panels * rule.0.len());
    let mut weights = Vec::with_capacity(panels * rule.0.len());
    let width = (hi - lo) / panels as f64;
    for panel in 0..panels {
        let left = lo + width * panel as f64;
        let centre = left + 0.5 * width;
        for (u, w) in rule.0.iter().zip(&rule.1) {
            nodes.push(centre + 0.5 * width * u);
            weights.push(0.5 * width * w);
        }
    }
    (nodes, weights)
}
