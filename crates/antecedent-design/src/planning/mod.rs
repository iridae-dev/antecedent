//! Experiment planning: power, sample size and minimum detectable effect.
//!
//! A [`PlanningRequest`] declares a randomized design, the assumptions its
//! operating characteristics rest on (outcome law, effect, level and tails,
//! allocation ratio, attrition, covariate adjustment, intracluster correlation,
//! serial dependence, priors) and a criterion, and solves in one of three
//! directions ([`PlanningGoal`]): the smallest size whose criterion reaches its
//! target, the criterion at a given size, or the minimum detectable effect at a
//! given size.
//!
//! Each design is planned for the analysis that will read it:
//!
//! | design | analysis | reference law of the test |
//! | --- | --- | --- |
//! | Bernoulli | difference in means (ANCOVA with covariates) | binomial mixture over realized arm sizes of noncentral t |
//! | complete | difference in means (ANCOVA with covariates) | noncentral t, `n − 2 − p` df |
//! | stratified | block-weighted difference in means | noncentral t, `n − 2H` df |
//! | cluster | cluster-total contrast | noncentral t, `k − 2` df, design effect `1 + ((1 + CV²) m − 1) ρ` |
//! | multi-arm | many-to-one contrasts (Bonferroni, Holm, Dunnett) or one-way F | equicorrelated multivariate t; noncentral F |
//! | factorial | two-by-two cell contrasts | noncentral t, `n − 4` df |
//! | switchback | sequence-clustered Horvitz–Thompson | noncentral t, `S − 1` df, block-autocorrelation inflation |
//! | group sequential | z statistic at each look | recursive integration of the canonical joint law |
//!
//! A binary outcome is tested by its Wald statistic against the normal law.
//!
//! The frequentist form is analytic (here) and simulation-based (in the
//! `antecedent` crate, through the compiler). The Bayesian form reports
//! assurance over a design prior (the planned test, or a posterior criterion
//! under a conjugate normal analysis prior), a posterior-precision target and
//! the probability of a correct decision; the posterior forms are conjugate
//! closed forms on the planned estimator's normal law.
//!
//! Designs whose operating characteristics cannot be established are refused
//! with the next attempt named: interference without an exposure model, an
//! outcome-adaptive design without a pre-specified rule, carryover without a
//! washout. Planned characteristics are point statements under the declared
//! assumptions; no coverage or calibration claim is attached to them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod analytic;
mod request;
mod result;

use antecedent_core::NextAttempt;

pub use analytic::{allocate, plan_analytic, try_allocate};
pub use request::{
    ArmComparison, FactorialContrast, FamilyPower, InterferenceExposure, MAX_SIMULATION_REPLICATES,
    Multiplicity, NormalPrior, OutcomeModel, PlannedDesign,
    PlanningAssumptions, PlanningCriterion, PlanningGoal, PlanningRefusal, PlanningRequest,
    SimulationSpec, StratumSpec, SuccessCriterion, TestSidedness,
};
pub use result::{
    AnalyticCharacteristics, FamilyCharacteristics, FamilySimulation, PlanningResult,
    SequentialCharacteristics, SequentialSimulation, SimulatedCharacteristics,
};

/// Declare how units are exposed to one another's assignment.
pub const DECLARE_EXPOSURE_MODEL: NextAttempt = NextAttempt::new(
    "declare_exposure_model",
    "declare that interference is contained in clusters that are randomized whole \
     (InterferenceExposure::ContainedInClusters), or randomize whole clusters and plan the \
     cluster design",
);

/// Pre-specify the stopping rule as a group-sequential boundary.
pub const PRESPECIFY_STOPPING_RULE: NextAttempt = NextAttempt::new(
    "prespecify_stopping_rule",
    "pre-specify the looks and an alpha-spending efficacy boundary and plan \
     PlannedDesign::GroupSequential",
);

/// Derive the operating characteristics of the whole adaptive procedure.
pub const SIMULATE_ADAPTIVE_PROCEDURE: NextAttempt = NextAttempt::new(
    "simulate_adaptive_procedure",
    "declare the whole adaptive procedure as an adaptive procedure contract and derive its \
     operating characteristics with the adaptive procedure simulation",
);

/// Exclude the periods that follow a switch.
pub const DECLARE_WASHOUT: NextAttempt = NextAttempt::new(
    "declare_washout",
    "declare at least one washout period per switch block so that every analysed period \
     follows a period with the same assignment",
);

/// Declare the control-arm outcome level.
pub const DECLARE_CONTROL_MEAN: NextAttempt = NextAttempt::new(
    "declare_control_mean",
    "declare the control-arm outcome mean (OutcomeModel::Continuous control_mean); the analysis \
     this design uses depends on the outcome level",
);

/// Plan a design whose analysis adjusts for covariates.
pub const COVARIATE_ADJUSTED_DESIGN: NextAttempt = NextAttempt::new(
    "covariate_adjusted_design",
    "plan Bernoulli or complete randomization of a continuous outcome, whose analysis adjusts \
     for baseline covariates, or declare no covariate adjustment",
);

/// Plan a single contrast for a posterior criterion.
pub const SINGLE_CONTRAST_BAYESIAN: NextAttempt = NextAttempt::new(
    "single_contrast_bayesian",
    "plan the posterior criterion for a fixed two-arm, stratified, cluster, factorial or \
     switchback contrast, or plan assurance with the significance criterion",
);

/// Relax the target, raise the search bound, or plan a larger effect.
pub const RELAX_TARGET: NextAttempt = NextAttempt::new(
    "relax_target",
    "raise max_units, lower the target, or plan a larger effect; a target above the limit the \
     design approaches as it grows is never reached",
);

/// Every attempt a planning refusal can name.
pub const PLANNING_NEXT_ATTEMPTS: [NextAttempt; 8] = [
    DECLARE_EXPOSURE_MODEL,
    PRESPECIFY_STOPPING_RULE,
    SIMULATE_ADAPTIVE_PROCEDURE,
    DECLARE_WASHOUT,
    DECLARE_CONTROL_MEAN,
    COVARIATE_ADJUSTED_DESIGN,
    SINGLE_CONTRAST_BAYESIAN,
    RELAX_TARGET,
];

/// Calibration status of every planned characteristic: the simulation
/// procedures are not yet measured against a coverage record.
pub const PLANNING_CALIBRATION: &str = "unmeasured";

#[cfg(test)]
mod tests;
