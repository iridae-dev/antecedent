//! The declared planning request: design, assumptions, criterion and goal.
//!
//! Every number that changes an operating characteristic is a field here and is
//! written to [`PlanningRequest::identity`]. Validation refuses an assumption
//! outside its domain, an option the design's analysis cannot use, and a
//! design whose operating characteristics cannot be established, each with the
//! next attempt named.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use antecedent_core::{CanonicalWriter, NextAttempt, SemanticDigest};
use antecedent_stats::adaptive_inference::group_sequential::{MAX_LOOKS, SpendingFunction};
use antecedent_stats::power::MAX_COMPARISONS;

use super::{
    COVARIATE_ADJUSTED_DESIGN, DECLARE_CONTROL_MEAN, DECLARE_EXPOSURE_MODEL, DECLARE_WASHOUT,
    PRESPECIFY_STOPPING_RULE, SIMULATE_ADAPTIVE_PROCEDURE, SINGLE_CONTRAST_BAYESIAN,
};

const REQUEST_TAG: &str = "antecedent.planning.request";

/// Most strata a stratified plan declares.
pub const MAX_STRATA: usize = 64;
/// Most Monte Carlo replicates one simulation runs.
pub const MAX_SIMULATION_REPLICATES: u32 = 1_000_000;
/// Most covariates an adjusted analysis declares.
pub const MAX_COVARIATES: u32 = 16;

/// Which tails of the test statistic reject.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TestSidedness {
    /// Reject for a large statistic in either direction.
    TwoSided,
    /// Reject for a large positive statistic only.
    OneSided,
}

antecedent_core::named_variants! {
    TestSidedness {
        TwoSided => "two_sided",
        OneSided => "one_sided",
    }
}

impl TestSidedness {
    /// Whether both tails reject.
    #[must_use]
    pub const fn is_two_sided(self) -> bool {
        matches!(self, Self::TwoSided)
    }
}

/// Outcome law the plan assumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OutcomeModel {
    /// A continuous outcome with a common standard deviation in every arm.
    Continuous {
        /// Outcome standard deviation `σ`, before covariate adjustment.
        sd: f64,
        /// Control-arm outcome mean. Required where the analysis depends on the
        /// outcome level (cluster totals with unequal sizes, switchback
        /// Horvitz–Thompson scores).
        control_mean: Option<f64>,
    },
    /// A binary outcome; the effect is a difference of proportions.
    Binary {
        /// Control-arm event probability.
        control_rate: f64,
    },
}

impl OutcomeModel {
    /// Stable name: `continuous` or `binary`.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Continuous { .. } => "continuous",
            Self::Binary { .. } => "binary",
        }
    }

    /// Whether the outcome is binary.
    #[must_use]
    pub const fn is_binary(&self) -> bool {
        matches!(self, Self::Binary { .. })
    }
}

/// The assumptions every planned operating characteristic rests on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanningAssumptions {
    /// Outcome law.
    pub outcome: OutcomeModel,
    /// Assumed effect (difference of means or of proportions). `None` when the
    /// goal solves for it or a design prior replaces it.
    pub effect: Option<f64>,
    /// Level of the test (total two-sided level for a two-sided test).
    pub alpha: f64,
    /// Rejection tails.
    pub sidedness: TestSidedness,
    /// Units in each treatment arm per unit in the control arm.
    pub allocation_ratio: f64,
    /// Fraction of enrolled units whose outcome is missing at analysis.
    pub attrition: f64,
    /// Share of outcome variance explained by baseline covariates the analysis adjusts for.
    pub covariate_r_squared: f64,
    /// Number of adjustment covariates (residual degrees of freedom).
    pub covariates: u32,
}

impl PlanningAssumptions {
    /// A continuous outcome of standard deviation `sd`: two-sided level 0.05,
    /// equal allocation, no attrition and no covariate adjustment.
    #[must_use]
    pub const fn continuous(sd: f64) -> Self {
        Self::with_outcome(OutcomeModel::Continuous { sd, control_mean: None })
    }

    /// A binary outcome with control rate `control_rate`, with the defaults of
    /// [`Self::continuous`].
    #[must_use]
    pub const fn binary(control_rate: f64) -> Self {
        Self::with_outcome(OutcomeModel::Binary { control_rate })
    }

    const fn with_outcome(outcome: OutcomeModel) -> Self {
        Self {
            outcome,
            effect: None,
            alpha: 0.05,
            sidedness: TestSidedness::TwoSided,
            allocation_ratio: 1.0,
            attrition: 0.0,
            covariate_r_squared: 0.0,
            covariates: 0,
        }
    }

    /// Declare the assumed effect.
    #[must_use]
    pub const fn with_effect(mut self, effect: f64) -> Self {
        self.effect = Some(effect);
        self
    }

    /// Declare the level and the rejection tails.
    #[must_use]
    pub const fn with_test(mut self, alpha: f64, sidedness: TestSidedness) -> Self {
        self.alpha = alpha;
        self.sidedness = sidedness;
        self
    }

    /// Declare the treatment-to-control allocation ratio.
    #[must_use]
    pub const fn with_allocation_ratio(mut self, ratio: f64) -> Self {
        self.allocation_ratio = ratio;
        self
    }

    /// Declare the attrition fraction.
    #[must_use]
    pub const fn with_attrition(mut self, attrition: f64) -> Self {
        self.attrition = attrition;
        self
    }

    /// Declare covariate adjustment: `covariates` baseline covariates
    /// explaining `r_squared` of the outcome variance.
    #[must_use]
    pub const fn with_covariates(mut self, r_squared: f64, covariates: u32) -> Self {
        self.covariate_r_squared = r_squared;
        self.covariates = covariates;
        self
    }

    /// Declare the control-arm mean of a continuous outcome.
    #[must_use]
    pub const fn with_control_mean(mut self, mean: f64) -> Self {
        if let OutcomeModel::Continuous { sd, .. } = self.outcome {
            self.outcome = OutcomeModel::Continuous { sd, control_mean: Some(mean) };
        }
        self
    }
}

/// Familywise error control of many-to-one comparisons.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Multiplicity {
    /// Single-step Bonferroni: each comparison at `alpha / K`.
    Bonferroni,
    /// Holm step-down over the Bonferroni levels.
    Holm,
    /// Dunnett single-step from the joint law of the comparisons.
    Dunnett,
}

antecedent_core::named_variants! {
    Multiplicity {
        Bonferroni => "bonferroni",
        Holm => "holm",
        Dunnett => "dunnett",
    }
}

/// What power means for a family of comparisons.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FamilyPower {
    /// Probability that a given comparison rejects.
    PerComparison,
    /// Probability that at least one comparison rejects (disjunctive).
    Any,
    /// Probability that every comparison rejects (conjunctive).
    All,
}

antecedent_core::named_variants! {
    FamilyPower {
        PerComparison => "per_comparison",
        Any => "any",
        All => "all",
    }
}

/// How a multi-arm trial is tested.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ArmComparison {
    /// Each treatment arm against the shared control, with familywise control.
    ManyToOne {
        /// Familywise error control.
        multiplicity: Multiplicity,
        /// What power means across the comparisons.
        power: FamilyPower,
    },
    /// One-way analysis-of-variance F test of equal arm means.
    Omnibus,
}

/// Contrast of a two-by-two factorial the plan powers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FactorialContrast {
    /// Main effect of the primary factor, averaged over the secondary factor.
    MainEffect,
    /// Difference of the primary factor's effects between secondary levels.
    Interaction,
}

antecedent_core::named_variants! {
    FactorialContrast {
        MainEffect => "main_effect",
        Interaction => "interaction",
    }
}

/// One stratum of a blocked design.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StratumSpec {
    /// Share of the enrolled units in this stratum.
    pub share: f64,
    /// Within-stratum outcome standard deviation (continuous outcome) or
    /// control-arm rate (binary outcome).
    pub within: f64,
}

/// Exposure model that makes interference plannable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InterferenceExposure {
    /// Interference is contained in clusters that are randomized whole, so the
    /// exposure contrast is the cluster-level assignment contrast.
    ContainedInClusters {
        /// Mean units per cluster.
        mean_cluster_size: f64,
        /// Intracluster correlation of the outcome.
        icc: f64,
        /// Coefficient of variation of cluster sizes.
        cluster_size_cv: f64,
    },
}

/// The randomized design whose operating characteristics are planned.
#[derive(Clone, Debug, PartialEq)]
pub enum PlannedDesign {
    /// Each unit is treated independently with probability `r / (1 + r)`.
    Bernoulli,
    /// Fixed arm sizes in the allocation ratio.
    Complete,
    /// Complete randomization within strata.
    Stratified {
        /// Strata with their shares and within-stratum outcome scale.
        strata: Vec<StratumSpec>,
    },
    /// Whole clusters are randomized.
    Cluster {
        /// Mean units per cluster.
        mean_cluster_size: f64,
        /// Intracluster correlation of the outcome.
        icc: f64,
        /// Coefficient of variation of cluster sizes (0 for equal sizes).
        cluster_size_cv: f64,
    },
    /// `treatment_arms` arms against one shared control.
    MultiArm {
        /// Number of treatment arms.
        treatment_arms: usize,
        /// Test of the family.
        comparison: ArmComparison,
    },
    /// Balanced two-by-two factorial.
    Factorial {
        /// Contrast the plan powers.
        contrast: FactorialContrast,
    },
    /// Each sequence switches between arms in randomized blocks of periods.
    Switchback {
        /// Independently randomized sequences (markets, sites, units).
        sequences: u64,
        /// Periods in one switch block.
        periods_per_block: u64,
        /// First-order autocorrelation of the outcome between periods.
        autocorrelation: f64,
        /// Share of the direct effect carried into the next period.
        carryover: f64,
        /// Periods at the start of each block excluded from the analysis.
        washout_periods: u64,
    },
    /// Two-arm complete randomization with pre-specified interim looks and an
    /// alpha-spending efficacy boundary.
    GroupSequential {
        /// Information fractions of the looks; the last is 1.
        information_fractions: Vec<f64>,
        /// Spending function of the boundary.
        spending: SpendingFunction,
    },
    /// Units interfere with one another.
    Interference {
        /// Exposure model; without one no operating characteristic is defined.
        exposure: Option<InterferenceExposure>,
    },
    /// Allocation or stopping adapts to outcomes.
    OutcomeAdaptive {
        /// Identity of the pre-specified adaptation rule, when there is one.
        rule: Option<String>,
    },
}

impl PlannedDesign {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Bernoulli => "bernoulli",
            Self::Complete => "complete",
            Self::Stratified { .. } => "stratified",
            Self::Cluster { .. } => "cluster",
            Self::MultiArm { .. } => "multi_arm",
            Self::Factorial { .. } => "factorial",
            Self::Switchback { .. } => "switchback",
            Self::GroupSequential { .. } => "group_sequential",
            Self::Interference { .. } => "interference",
            Self::OutcomeAdaptive { .. } => "outcome_adaptive",
        }
    }

    /// The randomization unit the planned size counts.
    #[must_use]
    pub const fn unit(&self) -> &'static str {
        match self {
            Self::Cluster { .. } | Self::Interference { .. } => "cluster",
            Self::Switchback { .. } => "switch_block_per_sequence",
            _ => "unit",
        }
    }

    /// Name of the analysis the plan's operating characteristics are for.
    #[must_use]
    pub fn analysis(&self, assumptions: &PlanningAssumptions) -> &'static str {
        let adjusted = assumptions.covariate_r_squared > 0.0;
        match self {
            Self::Bernoulli | Self::Complete if adjusted => "randomized_ancova_hc0",
            Self::Bernoulli | Self::Complete => "complete_unit_itt",
            Self::Stratified { .. } => "blocked_unit_itt",
            Self::Cluster { .. } | Self::Interference { .. } => "complete_cluster_itt",
            Self::MultiArm { comparison: ArmComparison::Omnibus, .. } => "one_way_anova_f",
            Self::MultiArm { .. } => "complete_unit_itt_many_to_one",
            Self::Factorial { .. } => "factorial_2x2",
            Self::Switchback { .. } => "switchback_itt",
            Self::GroupSequential { .. } => "complete_unit_itt_group_sequential",
            Self::OutcomeAdaptive { .. } => "unplanned",
        }
    }
}

/// A normal prior on the effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalPrior {
    /// Prior mean.
    pub mean: f64,
    /// Prior standard deviation; 0 is a point prior (design prior only).
    pub sd: f64,
}

/// What counts as a successful trial in an assurance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SuccessCriterion {
    /// The planned frequentist test rejects (hybrid assurance).
    Significance,
    /// The posterior probability that the effect exceeds `threshold` reaches
    /// `probability`, under a conjugate normal analysis prior.
    Posterior {
        /// Analysis prior.
        analysis_prior: NormalPrior,
        /// Effect the posterior must exceed.
        threshold: f64,
        /// Posterior probability that declares success.
        probability: f64,
    },
}

/// The operating characteristic a plan targets or reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlanningCriterion {
    /// Frequentist power of the planned test at the assumed effect.
    Power {
        /// Target power.
        target: f64,
    },
    /// Probability of success averaged over a design prior on the effect.
    Assurance {
        /// Design prior on the true effect.
        design_prior: NormalPrior,
        /// What a success is.
        success: SuccessCriterion,
        /// Target assurance.
        target: f64,
    },
    /// Posterior standard deviation at or below a target.
    PosteriorPrecision {
        /// Conjugate normal analysis prior.
        analysis_prior: NormalPrior,
        /// Largest acceptable posterior standard deviation.
        max_posterior_sd: f64,
    },
    /// Probability that the posterior decision (declare the effect above
    /// `threshold` when its posterior probability reaches `probability`) is right.
    CorrectDecision {
        /// Design prior on the true effect.
        design_prior: NormalPrior,
        /// Conjugate normal analysis prior.
        analysis_prior: NormalPrior,
        /// Decision threshold on the effect.
        threshold: f64,
        /// Posterior probability that declares the effect above the threshold.
        probability: f64,
        /// Target probability of a correct decision.
        target: f64,
    },
}

impl PlanningCriterion {
    /// A posterior-precision target from a credible-interval half-width:
    /// `half_width / z_{(1 + level) / 2}` bounds the posterior standard deviation.
    #[must_use]
    pub fn credible_half_width(analysis_prior: NormalPrior, level: f64, half_width: f64) -> Self {
        let z = antecedent_kernels::norm_inv(0.5 * (1.0 + level));
        Self::PosteriorPrecision { analysis_prior, max_posterior_sd: half_width / z }
    }

    /// Stable name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Power { .. } => "power",
            Self::Assurance { .. } => "assurance",
            Self::PosteriorPrecision { .. } => "posterior_precision",
            Self::CorrectDecision { .. } => "correct_decision",
        }
    }

    /// Whether the criterion is a Bayesian one (it declares a prior).
    #[must_use]
    pub const fn is_bayesian(&self) -> bool {
        !matches!(self, Self::Power { .. })
    }

    /// Whether a larger value is better (`false` for a posterior standard deviation).
    #[must_use]
    pub const fn larger_is_better(&self) -> bool {
        !matches!(self, Self::PosteriorPrecision { .. })
    }

    /// The target value.
    #[must_use]
    pub const fn target(&self) -> f64 {
        match *self {
            Self::Power { target }
            | Self::Assurance { target, .. }
            | Self::CorrectDecision { target, .. } => target,
            Self::PosteriorPrecision { max_posterior_sd, .. } => max_posterior_sd,
        }
    }
}

/// Which direction the plan solves in.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PlanningGoal {
    /// Smallest size whose criterion reaches its target.
    SampleSize {
        /// Largest size searched.
        max_units: u64,
    },
    /// The criterion at a given size.
    Evaluate {
        /// Randomization units.
        units: u64,
    },
    /// Smallest effect the planned test detects with the target power at a given size.
    MinimumDetectableEffect {
        /// Randomization units.
        units: u64,
    },
}

impl PlanningGoal {
    /// Stable name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SampleSize { .. } => "sample_size",
            Self::Evaluate { .. } => "evaluate",
            Self::MinimumDetectableEffect { .. } => "minimum_detectable_effect",
        }
    }
}

/// Whole-procedure simulation of the planned design and analysis.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SimulationSpec {
    /// Monte Carlo replicates.
    pub replicates: u32,
    /// Seed; each replicate opens its own keyed stream from it.
    pub seed: u64,
}

/// A complete planning request.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanningRequest {
    /// Planned design.
    pub design: PlannedDesign,
    /// Declared assumptions.
    pub assumptions: PlanningAssumptions,
    /// Operating characteristic targeted or reported.
    pub criterion: PlanningCriterion,
    /// Direction of the plan.
    pub goal: PlanningGoal,
    /// Simulation of the whole procedure, when requested.
    pub simulation: Option<SimulationSpec>,
}

/// Why a planning request is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanningRefusal {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Namespaced detail.
    pub detail: &'static str,
    /// What was refused.
    pub message: String,
    /// What to try instead.
    pub next_attempt: Option<NextAttempt>,
}

impl PlanningRefusal {
    pub(crate) fn new(
        code: &'static str,
        detail: &'static str,
        message: impl Into<String>,
        next_attempt: Option<NextAttempt>,
    ) -> Self {
        Self { code, detail, message: message.into(), next_attempt }
    }

    pub(crate) fn invalid(detail: &'static str, message: impl Into<String>) -> Self {
        Self::new(antecedent_core::reason_code!("invalid_argument"), detail, message, None)
    }
}

impl fmt::Display for PlanningRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}{}: {}: {}",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.message
        )?;
        if let Some(attempt) = self.next_attempt {
            write!(formatter, "; next attempt ({}): {}", attempt.name(), attempt.sentence())?;
        }
        Ok(())
    }
}

impl std::error::Error for PlanningRefusal {}

antecedent_core::impl_refused!(PlanningRefusal);

fn probability(value: f64) -> bool {
    value.is_finite() && value > 0.0 && value < 1.0
}

fn check_prior(prior: NormalPrior, what: &str, allow_point: bool) -> Result<(), PlanningRefusal> {
    let sd_ok = prior.sd.is_finite() && (prior.sd > 0.0 || (allow_point && prior.sd == 0.0));
    if !prior.mean.is_finite() || !sd_ok {
        return Err(PlanningRefusal::invalid(
            "planning.prior",
            format!(
                "the {what} prior needs a finite mean and a {} standard deviation (got {} and {})",
                if allow_point { "finite nonnegative" } else { "finite positive" },
                prior.mean,
                prior.sd
            ),
        ));
    }
    Ok(())
}

impl PlanningRequest {
    /// A request without simulation.
    #[must_use]
    pub const fn new(
        design: PlannedDesign,
        assumptions: PlanningAssumptions,
        criterion: PlanningCriterion,
        goal: PlanningGoal,
    ) -> Self {
        Self { design, assumptions, criterion, goal, simulation: None }
    }

    /// Request a whole-procedure simulation of `replicates` replicates from `seed`.
    #[must_use]
    pub const fn with_simulation(mut self, replicates: u32, seed: u64) -> Self {
        self.simulation = Some(SimulationSpec { replicates, seed });
        self
    }

    /// The design whose operating characteristics are computed: an interference
    /// design contained in clusters is planned as the cluster design it is.
    #[must_use]
    pub fn effective_design(&self) -> PlannedDesign {
        match &self.design {
            PlannedDesign::Interference {
                exposure:
                    Some(InterferenceExposure::ContainedInClusters {
                        mean_cluster_size,
                        icc,
                        cluster_size_cv,
                    }),
            } => PlannedDesign::Cluster {
                mean_cluster_size: *mean_cluster_size,
                icc: *icc,
                cluster_size_cv: *cluster_size_cv,
            },
            design => design.clone(),
        }
    }

    /// Check every declared value and refuse a design whose operating
    /// characteristics cannot be established.
    ///
    /// # Errors
    ///
    /// A [`PlanningRefusal`] naming what was refused and, where one exists, the
    /// next attempt.
    pub fn validate(&self) -> Result<(), PlanningRefusal> {
        self.check_operability()?;
        self.check_assumptions()?;
        self.check_criterion()?;
        self.check_design()?;
        self.check_goal_and_simulation()
    }

    fn check_operability(&self) -> Result<(), PlanningRefusal> {
        match &self.design {
            PlannedDesign::Interference { exposure: None } => Err(PlanningRefusal::new(
                antecedent_core::reason_code!("required_option_missing"),
                "planning.interference",
                "units interfere and no exposure model is declared, so the exposure contrast \
                 and its variance are undefined",
                Some(DECLARE_EXPOSURE_MODEL),
            )),
            PlannedDesign::OutcomeAdaptive { rule: None } => Err(PlanningRefusal::new(
                antecedent_core::reason_code!("adaptive_contract_incomplete"),
                "planning.adaptive",
                "an outcome-adaptive design without a pre-specified adaptation rule has no \
                 operating characteristics: the stopping rule is missing",
                Some(PRESPECIFY_STOPPING_RULE),
            )),
            PlannedDesign::OutcomeAdaptive { rule: Some(rule) } => Err(PlanningRefusal::new(
                antecedent_core::reason_code!("route_not_supported"),
                "planning.adaptive",
                format!(
                    "the adaptation rule {rule:?} is not a group-sequential boundary; its \
                     operating characteristics are those of the whole adaptive procedure"
                ),
                Some(SIMULATE_ADAPTIVE_PROCEDURE),
            )),
            _ => Ok(()),
        }
    }

    fn check_assumptions(&self) -> Result<(), PlanningRefusal> {
        let a = &self.assumptions;
        if !(a.alpha.is_finite() && a.alpha > 0.0 && a.alpha <= 0.5) {
            return Err(PlanningRefusal::invalid(
                "planning.alpha",
                format!("alpha must lie in (0, 0.5] (got {})", a.alpha),
            ));
        }
        if !(a.allocation_ratio.is_finite() && a.allocation_ratio > 0.0) {
            return Err(PlanningRefusal::invalid(
                "planning.allocation_ratio",
                format!("the allocation ratio must be finite and > 0 (got {})", a.allocation_ratio),
            ));
        }
        if !(a.attrition.is_finite() && (0.0..1.0).contains(&a.attrition)) {
            return Err(PlanningRefusal::invalid(
                "planning.attrition",
                format!("attrition must lie in [0, 1) (got {})", a.attrition),
            ));
        }
        if !(a.covariate_r_squared.is_finite() && (0.0..1.0).contains(&a.covariate_r_squared)) {
            return Err(PlanningRefusal::invalid(
                "planning.covariates",
                format!("covariate R² must lie in [0, 1) (got {})", a.covariate_r_squared),
            ));
        }
        if (a.covariate_r_squared > 0.0) != (a.covariates > 0) || a.covariates > MAX_COVARIATES {
            return Err(PlanningRefusal::invalid(
                "planning.covariates",
                format!(
                    "a covariate R² above zero needs between 1 and {MAX_COVARIATES} declared \
                     covariates, and covariates need an R² above zero (got R² {} and {} \
                     covariates)",
                    a.covariate_r_squared, a.covariates
                ),
            ));
        }
        match a.outcome {
            OutcomeModel::Continuous { sd, control_mean } => {
                if !(sd.is_finite() && sd > 0.0) || control_mean.is_some_and(|m| !m.is_finite()) {
                    return Err(PlanningRefusal::invalid(
                        "planning.outcome",
                        format!(
                            "the outcome sd must be finite and > 0 and a declared control mean finite (got sd {sd})"
                        ),
                    ));
                }
            }
            OutcomeModel::Binary { control_rate } => {
                if !probability(control_rate) {
                    return Err(PlanningRefusal::invalid(
                        "planning.outcome",
                        format!("the control rate must lie in (0, 1) (got {control_rate})"),
                    ));
                }
                if a.covariate_r_squared > 0.0 {
                    return Err(PlanningRefusal::new(
                        antecedent_core::reason_code!("option_not_applicable"),
                        "planning.covariates",
                        "covariate adjustment is planned for a continuous outcome",
                        Some(COVARIATE_ADJUSTED_DESIGN),
                    ));
                }
            }
        }
        if let Some(effect) = a.effect {
            if !effect.is_finite() {
                return Err(PlanningRefusal::invalid(
                    "planning.effect",
                    format!("the effect must be finite (got {effect})"),
                ));
            }
            if let OutcomeModel::Binary { control_rate } = a.outcome {
                if !probability(control_rate + effect) {
                    return Err(PlanningRefusal::invalid(
                        "planning.effect",
                        format!(
                            "control rate {control_rate} plus effect {effect} is not a probability"
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    fn check_criterion(&self) -> Result<(), PlanningRefusal> {
        let a = &self.assumptions;
        let target_ok = |target: f64| probability(target);
        match self.criterion {
            PlanningCriterion::Power { target } => {
                if !target_ok(target) || target <= a.alpha {
                    return Err(PlanningRefusal::invalid(
                        "planning.power",
                        format!("target power must lie in (alpha, 1) (got {target})"),
                    ));
                }
                let needs_effect =
                    !matches!(self.goal, PlanningGoal::MinimumDetectableEffect { .. });
                match (needs_effect, a.effect) {
                    (true, None) => Err(PlanningRefusal::new(
                        antecedent_core::reason_code!("required_option_missing"),
                        "planning.effect",
                        "power is computed at a declared effect; none is declared",
                        None,
                    )),
                    (true, Some(effect))
                        if effect == 0.0 && !matches!(self.goal, PlanningGoal::Evaluate { .. }) =>
                    {
                        Err(PlanningRefusal::invalid(
                            "planning.effect",
                            "a sample size cannot be planned to detect a zero effect",
                        ))
                    }
                    (false, Some(_)) => Err(PlanningRefusal::invalid(
                        "planning.effect",
                        "a minimum detectable effect is solved for; declare no effect",
                    )),
                    _ => Ok(()),
                }
            }
            PlanningCriterion::Assurance { design_prior, success, target } => {
                self.no_effect_with_prior()?;
                check_prior(design_prior, "design", true)?;
                if !target_ok(target) {
                    return Err(PlanningRefusal::invalid(
                        "planning.assurance",
                        format!("target assurance must lie in (0, 1) (got {target})"),
                    ));
                }
                if let SuccessCriterion::Posterior { analysis_prior, threshold, probability: p } =
                    success
                {
                    check_prior(analysis_prior, "analysis", false)?;
                    if !threshold.is_finite() || !probability(p) {
                        return Err(PlanningRefusal::invalid(
                            "planning.success",
                            "the success threshold must be finite and its probability in (0, 1)",
                        ));
                    }
                }
                Ok(())
            }
            PlanningCriterion::PosteriorPrecision { analysis_prior, max_posterior_sd } => {
                self.no_effect_with_prior()?;
                check_prior(analysis_prior, "analysis", false)?;
                if !(max_posterior_sd.is_finite() && max_posterior_sd > 0.0) {
                    return Err(PlanningRefusal::invalid(
                        "planning.precision",
                        format!(
                            "the posterior sd target must be finite and > 0 (got {max_posterior_sd})"
                        ),
                    ));
                }
                Ok(())
            }
            PlanningCriterion::CorrectDecision {
                design_prior,
                analysis_prior,
                threshold,
                probability: p,
                target,
            } => {
                self.no_effect_with_prior()?;
                check_prior(design_prior, "design", true)?;
                check_prior(analysis_prior, "analysis", false)?;
                if !threshold.is_finite() || !probability(p) || !target_ok(target) {
                    return Err(PlanningRefusal::invalid(
                        "planning.decision",
                        "the decision threshold must be finite and its probability and target \
                         in (0, 1)",
                    ));
                }
                Ok(())
            }
        }
    }

    fn no_effect_with_prior(&self) -> Result<(), PlanningRefusal> {
        if self.assumptions.effect.is_some() {
            return Err(PlanningRefusal::invalid(
                "planning.effect",
                "a Bayesian criterion takes the effect from its design prior; declare no effect",
            ));
        }
        if matches!(self.goal, PlanningGoal::MinimumDetectableEffect { .. }) {
            return Err(PlanningRefusal::invalid(
                "planning.goal",
                "a minimum detectable effect is a property of the frequentist power criterion",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines, reason = "one arm per design, each a short list of checks")]
    fn check_design(&self) -> Result<(), PlanningRefusal> {
        let a = &self.assumptions;
        let adjusted = a.covariate_r_squared > 0.0;
        let design = self.effective_design();
        if adjusted && !matches!(design, PlannedDesign::Bernoulli | PlannedDesign::Complete) {
            return Err(PlanningRefusal::new(
                antecedent_core::reason_code!("option_not_applicable"),
                "planning.covariates",
                format!("the {} analysis does not adjust for covariates", design.as_str()),
                Some(COVARIATE_ADJUSTED_DESIGN),
            ));
        }
        let single_contrast_posterior = matches!(
            self.criterion,
            PlanningCriterion::Assurance { success: SuccessCriterion::Posterior { .. }, .. }
                | PlanningCriterion::PosteriorPrecision { .. }
                | PlanningCriterion::CorrectDecision { .. }
        );
        match &design {
            PlannedDesign::Bernoulli
            | PlannedDesign::Complete
            | PlannedDesign::Interference { .. }
            | PlannedDesign::OutcomeAdaptive { .. } => Ok(()),
            PlannedDesign::Stratified { strata } => {
                if strata.is_empty() || strata.len() > MAX_STRATA {
                    return Err(PlanningRefusal::invalid(
                        "planning.stratified",
                        format!("a stratified plan declares between 1 and {MAX_STRATA} strata"),
                    ));
                }
                let total: f64 = strata.iter().map(|s| s.share).sum();
                if strata.iter().any(|s| !(s.share.is_finite() && s.share > 0.0))
                    || (total - 1.0).abs() > 1e-9
                {
                    return Err(PlanningRefusal::invalid(
                        "planning.stratified",
                        format!("stratum shares must be positive and sum to 1 (sum {total})"),
                    ));
                }
                for stratum in strata {
                    let ok = match a.outcome {
                        OutcomeModel::Continuous { .. } => {
                            stratum.within.is_finite() && stratum.within > 0.0
                        }
                        OutcomeModel::Binary { .. } => {
                            probability(stratum.within)
                                && a.effect
                                    .is_none_or(|effect| probability(stratum.within + effect))
                        }
                    };
                    if !ok {
                        return Err(PlanningRefusal::invalid(
                            "planning.stratified",
                            format!(
                                "stratum value {} is not a positive sd (continuous) or a rate \
                                 that stays a probability under the effect (binary)",
                                stratum.within
                            ),
                        ));
                    }
                }
                Ok(())
            }
            PlannedDesign::Cluster { mean_cluster_size, icc, cluster_size_cv } => {
                if !(mean_cluster_size.is_finite()
                    && *mean_cluster_size >= 1.0
                    && icc.is_finite()
                    && (0.0..1.0).contains(icc)
                    && cluster_size_cv.is_finite()
                    && *cluster_size_cv >= 0.0)
                {
                    return Err(PlanningRefusal::invalid(
                        "planning.cluster",
                        format!(
                            "cluster size must be >= 1, ICC in [0, 1) and size CV >= 0 (got {mean_cluster_size}, {icc}, {cluster_size_cv})"
                        ),
                    ));
                }
                if *cluster_size_cv > 0.0
                    && matches!(a.outcome, OutcomeModel::Continuous { control_mean: None, .. })
                {
                    return Err(PlanningRefusal::new(
                        antecedent_core::reason_code!("required_option_missing"),
                        "planning.cluster",
                        "the cluster-total analysis with unequal cluster sizes depends on the \
                         outcome level; no control mean is declared",
                        Some(DECLARE_CONTROL_MEAN),
                    ));
                }
                Ok(())
            }
            PlannedDesign::MultiArm { treatment_arms, comparison } => {
                if !(2..=MAX_COMPARISONS).contains(treatment_arms) {
                    return Err(PlanningRefusal::invalid(
                        "planning.multi_arm",
                        format!(
                            "a multi-arm plan has between 2 and {MAX_COMPARISONS} treatment arms"
                        ),
                    ));
                }
                if single_contrast_posterior {
                    return Err(PlanningRefusal::new(
                        antecedent_core::reason_code!("route_not_supported"),
                        "planning.multi_arm",
                        "a posterior criterion is planned for a single contrast; a family of \
                         comparisons has none",
                        Some(SINGLE_CONTRAST_BAYESIAN),
                    ));
                }
                if *comparison == ArmComparison::Omnibus
                    && (a.outcome.is_binary() || !a.sidedness.is_two_sided())
                {
                    return Err(PlanningRefusal::new(
                        antecedent_core::reason_code!("option_not_applicable"),
                        "planning.multi_arm",
                        "the omnibus F test is planned for a continuous outcome and is two-sided",
                        None,
                    ));
                }
                Ok(())
            }
            PlannedDesign::Factorial { .. } => {
                if (a.allocation_ratio - 1.0).abs() > 1e-12 {
                    return Err(PlanningRefusal::new(
                        antecedent_core::reason_code!("option_not_applicable"),
                        "planning.factorial",
                        "the factorial plan has four equal cells; the allocation ratio is fixed at 1",
                        None,
                    ));
                }
                Ok(())
            }
            PlannedDesign::Switchback {
                sequences,
                periods_per_block,
                autocorrelation,
                carryover,
                washout_periods,
            } => {
                if *sequences < 2
                    || *periods_per_block == 0
                    || washout_periods >= periods_per_block
                    || !(autocorrelation.is_finite() && autocorrelation.abs() < 1.0)
                    || !carryover.is_finite()
                    || (1.0 + carryover).abs() < 1e-9
                {
                    return Err(PlanningRefusal::invalid(
                        "planning.switchback",
                        "a switchback plan needs at least 2 sequences, a block longer than its \
                         washout, autocorrelation in (-1, 1) and a finite carryover other than -1",
                    ));
                }
                if a.attrition > 0.0 {
                    return Err(PlanningRefusal::new(
                        antecedent_core::reason_code!("option_not_applicable"),
                        "planning.switchback",
                        "switchback periods are scheduled, not enrolled; attrition does not apply",
                        None,
                    ));
                }
                if *carryover != 0.0 && *washout_periods == 0 {
                    return Err(PlanningRefusal::new(
                        antecedent_core::reason_code!("effect_not_identified"),
                        "planning.switchback",
                        "with carryover and no washout the first period of a block mixes the \
                         previous block's assignment, so the period contrast does not estimate \
                         the global effect",
                        Some(DECLARE_WASHOUT),
                    ));
                }
                match a.outcome {
                    OutcomeModel::Binary { .. } => Err(PlanningRefusal::new(
                        antecedent_core::reason_code!("option_not_applicable"),
                        "planning.switchback",
                        "switchback planning models a continuous outcome with autoregressive \
                         period dependence",
                        None,
                    )),
                    OutcomeModel::Continuous { control_mean: None, .. } => {
                        Err(PlanningRefusal::new(
                            antecedent_core::reason_code!("required_option_missing"),
                            "planning.switchback",
                            "the switchback Horvitz–Thompson analysis depends on the outcome \
                             level; no control mean is declared",
                            Some(DECLARE_CONTROL_MEAN),
                        ))
                    }
                    OutcomeModel::Continuous { .. } => Ok(()),
                }
            }
            PlannedDesign::GroupSequential { information_fractions, spending: _ } => {
                let increasing = information_fractions.windows(2).all(|w| w[1] > w[0]);
                let last_is_one =
                    information_fractions.last().is_some_and(|last| (*last - 1.0).abs() < 1e-12);
                if information_fractions.is_empty()
                    || information_fractions.len() > MAX_LOOKS
                    || !increasing
                    || !last_is_one
                    || information_fractions.iter().any(|f| !(f.is_finite() && *f > 0.0))
                {
                    return Err(PlanningRefusal::invalid(
                        "planning.group_sequential",
                        format!(
                            "information fractions must be strictly increasing in (0, 1], end \
                             at 1 and number at most {MAX_LOOKS}"
                        ),
                    ));
                }
                let hybrid = matches!(
                    self.criterion,
                    PlanningCriterion::Power { .. }
                        | PlanningCriterion::Assurance {
                            success: SuccessCriterion::Significance,
                            ..
                        }
                );
                if !hybrid {
                    return Err(PlanningRefusal::new(
                        antecedent_core::reason_code!("route_not_supported"),
                        "planning.group_sequential",
                        "a group-sequential plan targets power or the assurance of crossing its \
                         efficacy boundary; a posterior criterion of the stopped trial is not \
                         planned",
                        Some(SINGLE_CONTRAST_BAYESIAN),
                    ));
                }
                Ok(())
            }
        }
    }

    fn check_goal_and_simulation(&self) -> Result<(), PlanningRefusal> {
        match self.goal {
            PlanningGoal::SampleSize { max_units: 0 } => {
                return Err(PlanningRefusal::invalid("planning.goal", "max_units must be > 0"));
            }
            PlanningGoal::Evaluate { units } | PlanningGoal::MinimumDetectableEffect { units }
                if units == 0 =>
            {
                return Err(PlanningRefusal::invalid("planning.goal", "units must be > 0"));
            }
            _ => {}
        }
        if let Some(spec) = self.simulation {
            if spec.replicates == 0 || spec.replicates > MAX_SIMULATION_REPLICATES {
                return Err(PlanningRefusal::invalid(
                    "planning.simulation",
                    format!(
                        "simulation replicates must lie in [1, {MAX_SIMULATION_REPLICATES}] (got {})",
                        spec.replicates
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Canonical identity of the request: every declared assumption, the
    /// design, the criterion, the goal and the simulation seed and size.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut w = CanonicalWriter::new(REQUEST_TAG);
        write_design(&mut w, &self.design);
        write_assumptions(&mut w, &self.assumptions);
        write_criterion(&mut w, &self.criterion);
        w.text(self.goal.as_str());
        match self.goal {
            PlanningGoal::SampleSize { max_units } => w.word(max_units),
            PlanningGoal::Evaluate { units } | PlanningGoal::MinimumDetectableEffect { units } => {
                w.word(units)
            }
        };
        match self.simulation {
            Some(spec) => w.word(1).word(u64::from(spec.replicates)).word(spec.seed),
            None => w.word(0),
        };
        SemanticDigest::from_bytes(w.finish())
    }
}

fn write_prior(w: &mut CanonicalWriter, prior: NormalPrior) {
    w.real(prior.mean).real(prior.sd);
}

fn write_assumptions(w: &mut CanonicalWriter, a: &PlanningAssumptions) {
    w.text(a.outcome.as_str());
    match a.outcome {
        OutcomeModel::Continuous { sd, control_mean } => {
            w.real(sd).opt_real(control_mean);
        }
        OutcomeModel::Binary { control_rate } => {
            w.real(control_rate);
        }
    }
    w.opt_real(a.effect)
        .real(a.alpha)
        .text(a.sidedness.as_str())
        .real(a.allocation_ratio)
        .real(a.attrition)
        .real(a.covariate_r_squared)
        .word(u64::from(a.covariates));
}

fn write_criterion(w: &mut CanonicalWriter, criterion: &PlanningCriterion) {
    w.text(criterion.as_str());
    match *criterion {
        PlanningCriterion::Power { target } => {
            w.real(target);
        }
        PlanningCriterion::Assurance { design_prior, success, target } => {
            write_prior(w, design_prior);
            match success {
                SuccessCriterion::Significance => {
                    w.text("significance");
                }
                SuccessCriterion::Posterior { analysis_prior, threshold, probability } => {
                    w.text("posterior");
                    write_prior(w, analysis_prior);
                    w.real(threshold).real(probability);
                }
            }
            w.real(target);
        }
        PlanningCriterion::PosteriorPrecision { analysis_prior, max_posterior_sd } => {
            write_prior(w, analysis_prior);
            w.real(max_posterior_sd);
        }
        PlanningCriterion::CorrectDecision {
            design_prior,
            analysis_prior,
            threshold,
            probability,
            target,
        } => {
            write_prior(w, design_prior);
            write_prior(w, analysis_prior);
            w.real(threshold).real(probability).real(target);
        }
    }
}

fn write_design(w: &mut CanonicalWriter, design: &PlannedDesign) {
    w.text(design.as_str());
    match design {
        PlannedDesign::Bernoulli | PlannedDesign::Complete => {}
        PlannedDesign::Stratified { strata } => {
            w.word(strata.len() as u64);
            for stratum in strata {
                w.real(stratum.share).real(stratum.within);
            }
        }
        PlannedDesign::Cluster { mean_cluster_size, icc, cluster_size_cv } => {
            w.real(*mean_cluster_size).real(*icc).real(*cluster_size_cv);
        }
        PlannedDesign::MultiArm { treatment_arms, comparison } => {
            w.word(*treatment_arms as u64);
            match comparison {
                ArmComparison::ManyToOne { multiplicity, power } => {
                    w.text("many_to_one").text(multiplicity.as_str()).text(power.as_str());
                }
                ArmComparison::Omnibus => {
                    w.text("omnibus");
                }
            }
        }
        PlannedDesign::Factorial { contrast } => {
            w.text(contrast.as_str());
        }
        PlannedDesign::Switchback {
            sequences,
            periods_per_block,
            autocorrelation,
            carryover,
            washout_periods,
        } => {
            w.word(*sequences)
                .word(*periods_per_block)
                .real(*autocorrelation)
                .real(*carryover)
                .word(*washout_periods);
        }
        PlannedDesign::GroupSequential { information_fractions, spending } => {
            w.reals(information_fractions).text(spending.as_str());
        }
        PlannedDesign::Interference { exposure } => match exposure {
            Some(InterferenceExposure::ContainedInClusters {
                mean_cluster_size,
                icc,
                cluster_size_cv,
            }) => {
                w.text("contained_in_clusters")
                    .real(*mean_cluster_size)
                    .real(*icc)
                    .real(*cluster_size_cv);
            }
            None => {
                w.text("undeclared");
            }
        },
        PlannedDesign::OutcomeAdaptive { rule } => {
            w.opt_text(rule.as_deref());
        }
    }
}
