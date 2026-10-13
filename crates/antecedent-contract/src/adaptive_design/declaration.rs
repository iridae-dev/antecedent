//! The retained analysis contract of an adaptive experiment: declarations,
//! completeness and consistency checks, canonical identity.
//!
//! Inference from adaptively collected data is licensed only when the whole
//! data-generating and monitoring contract is represented. The contract keeps
//! six elements beyond the realized assignments themselves:
//!
//! 1. the assignment-policy history (which policy governed which units);
//! 2. the realized assignment probabilities (the policy's distribution over
//!    arms at each unit's assignment);
//! 3. the exposure history (the arm each unit actually received);
//! 4. outcome availability and delay (when each outcome became observable);
//! 5. the interim analyses (how many units were enrolled at each look and the
//!    information fraction); and
//! 6. the stopping rule (its kind, parameters, and whether it ended the trial).
//!
//! Each missing element is its own typed refusal. The final sample is never
//! analysed as if the design had been fixed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, SemanticDigest};

use super::matrix::{
    AssignmentFamily, CellStatus, InferenceParadigm, MonitoringStrategy, ProcedureCell,
    StoppingRuleKind, procedure_status,
};
use crate::ExternalRefusal;

/// Version word of the canonical contract encoding. Bump when the encoding
/// below changes.
pub const ADAPTIVE_DESIGN_VERSION: u64 = 1;
/// Most arms a contract declares.
pub const MAX_CONTRACT_ARMS: usize = 64;
/// Most units a contract declares.
pub const MAX_CONTRACT_UNITS: usize = 1 << 20;
/// Most interim analyses a contract declares.
pub const MAX_CONTRACT_LOOKS: usize = 64;
/// Most measurements one unit carries.
pub const MAX_UNIT_MEASUREMENTS: usize = 1024;

const STAGE_DECLARE: &str = "declare";
const STAGE_BIND: &str = "bind";
const PROBABILITY_ROW_TOLERANCE: f64 = 1e-8;

/// One policy state and the first unit it governed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolicyEpoch {
    /// First unit (enrollment order) assigned under this policy state.
    pub first_unit: u32,
    /// Identity of the policy state: its parameters and the history it
    /// conditioned on.
    pub policy_identity: SemanticDigest,
}

/// How outcome availability relates to the outcomes themselves.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum DelayMechanism {
    /// Every outcome is available when its unit is enrolled.
    Immediate,
    /// Outcomes arrive after a delay that does not depend on their values.
    IgnorableAdministrative,
    /// Whether and when an outcome arrives depends on its value.
    OutcomeDependent,
}

impl DelayMechanism {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Immediate => "immediate",
            Self::IgnorableAdministrative => "ignorable_administrative",
            Self::OutcomeDependent => "outcome_dependent",
        }
    }
}

/// When each outcome first became available, in terms of interim looks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutcomeAvailability {
    /// Per unit: the first look at which its outcome was available, or `None`
    /// when it never became available within the trial.
    pub available_at_look: Vec<Option<u32>>,
    /// How availability relates to the outcome values.
    pub delay: DelayMechanism,
}

/// One interim analysis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterimAnalysis {
    /// Units enrolled when the analysis was performed (a prefix of the
    /// enrollment order).
    pub units_enrolled: u32,
    /// Information fraction the analysis had reached, in `(0, 1]`.
    pub information_fraction: f64,
}

/// Which alpha-spending function a group-sequential rule uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum SpendingKind {
    /// `2 - 2 Phi(z_{alpha/2} / sqrt(t))`.
    OBrienFlemingType,
    /// `alpha ln(1 + (e - 1) t)`.
    PocockType,
}

impl SpendingKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OBrienFlemingType => "obrien_fleming_type",
            Self::PocockType => "pocock_type",
        }
    }

    /// Parse a wire name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::OBrienFlemingType, Self::PocockType].into_iter().find(|k| k.as_str() == name)
    }
}

/// Which crossings of an efficacy boundary stop the trial.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum BoundarySidedness {
    /// `Z >= c`.
    OneSidedUpper,
    /// `|Z| >= c`.
    TwoSidedSymmetric,
}

impl BoundarySidedness {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OneSidedUpper => "one_sided_upper",
            Self::TwoSidedSymmetric => "two_sided_symmetric",
        }
    }

    /// Parse a wire name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::OneSidedUpper, Self::TwoSidedSymmetric].into_iter().find(|k| k.as_str() == name)
    }
}

/// Parameters of a stopping rule.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum StoppingSpec {
    /// No rule: the planned horizon is run.
    NoStopping,
    /// Error-rate boundaries by alpha spending.
    AlphaSpending {
        /// Spending function.
        spending: SpendingKind,
        /// Total alpha the rule spends.
        total_alpha: f64,
        /// Which crossings stop the trial.
        sidedness: BoundarySidedness,
    },
    /// Stop when the posterior probability that an arm is best exceeds a
    /// threshold.
    PosteriorProbabilityBest {
        /// Threshold in `(0.5, 1)`.
        threshold: f64,
    },
    /// Stop when the posterior expected loss of the best arm falls below a
    /// threshold.
    PosteriorExpectedLoss {
        /// Loss threshold, in units of the success probability.
        loss_threshold: f64,
    },
    /// Stop at a resource cap.
    ResourceCap {
        /// Most units that may be enrolled.
        max_units: u32,
    },
}

/// A stopping rule and whether it ended the trial.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StoppingRuleDeclaration {
    /// What a look can decide.
    pub kind: StoppingRuleKind,
    /// Its parameters.
    pub spec: StoppingSpec,
    /// Whether the rule, rather than the planned horizon, ended the trial at
    /// the final look.
    pub ended_by_rule: bool,
}

/// Everything a producer states about one adaptive experiment.
///
/// The four optional elements and the stopping rule are `Option` so that a
/// missing element is representable and refused with its own detail, never
/// silently defaulted.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveDesignDeclaration {
    /// Human label; part of the identity.
    pub label: String,
    /// The procedure cell the experiment ran under.
    pub cell: ProcedureCell,
    /// Arm labels, in arm-index order.
    pub arm_labels: Vec<String>,
    /// Realized assigned arm per unit, in enrollment order.
    pub assigned_arm: Vec<u32>,
    /// Assignment-policy history.
    pub policy_history: Option<Vec<PolicyEpoch>>,
    /// Per unit, the policy's probability of every arm at assignment.
    pub assignment_probabilities: Option<Vec<Vec<f64>>>,
    /// Per unit, the arm actually received.
    pub exposure: Option<Vec<u32>>,
    /// Outcome availability and delay.
    pub availability: Option<OutcomeAvailability>,
    /// Per unit, the observed measurements (empty when never observed).
    pub outcomes: Vec<Vec<f64>>,
    /// The interim analyses, in order; the last is the final analysis.
    pub interim_analyses: Option<Vec<InterimAnalysis>>,
    /// The stopping rule.
    pub stopping_rule: Option<StoppingRuleDeclaration>,
    /// Smallest assignment probability an analysed arm may have had.
    pub probability_floor: f64,
}

/// A contract that was refused: one typed refusal per defect.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveContractError {
    /// The refusals, in the order the defects were found.
    pub refusals: Vec<ExternalRefusal>,
}

impl AdaptiveContractError {
    /// The namespaced detail of each refusal.
    #[must_use]
    pub fn details(&self) -> Vec<&str> {
        self.refusals.iter().map(|r| r.detail.as_str()).collect()
    }
}

impl std::fmt::Display for AdaptiveContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "adaptive design contract refused: {}", self.details().join(", "))
    }
}

impl std::error::Error for AdaptiveContractError {}

fn is_binary(value: f64) -> bool {
    value.to_bits() == 0.0_f64.to_bits() || value.to_bits() == 1.0_f64.to_bits()
}

fn declare(code: &'static str, detail: &'static str, remedy: &'static str) -> ExternalRefusal {
    ExternalRefusal::new(code, STAGE_DECLARE, detail).with_remedy(remedy)
}

fn missing(detail: &'static str, remedy: &'static str) -> ExternalRefusal {
    declare(antecedent_core::reason_code!("adaptive_contract_incomplete"), detail, remedy)
}

fn invalid(detail: &'static str, remedy: &'static str) -> ExternalRefusal {
    declare(antecedent_core::reason_code!("invalid_argument"), detail, remedy)
}

fn required_elements(declaration: &AdaptiveDesignDeclaration, refusals: &mut Vec<ExternalRefusal>) {
    if declaration.policy_history.is_none() {
        refusals.push(missing(
            "adaptive_inference.policy_history_missing",
            "retain the assignment-policy history: the policy state that governed each stretch of units",
        ));
    }
    if declaration.assignment_probabilities.is_none() {
        refusals.push(missing(
            "adaptive_inference.assignment_probabilities_missing",
            "retain each unit's realized assignment probabilities (the policy's distribution over arms at that unit)",
        ));
    }
    if declaration.exposure.is_none() {
        refusals.push(missing(
            "adaptive_inference.exposure_history_missing",
            "retain the arm each unit actually received, so assignment and exposure can be compared",
        ));
    }
    if declaration.availability.is_none() {
        refusals.push(missing(
            "adaptive_inference.outcome_availability_missing",
            "retain when each outcome became available and how that delay relates to the outcome",
        ));
    }
    if declaration.interim_analyses.is_none() {
        refusals.push(missing(
            "adaptive_inference.interim_analyses_missing",
            "retain the interim analyses performed: units enrolled and information fraction at each look",
        ));
    }
    if declaration.stopping_rule.is_none() {
        refusals.push(missing(
            "adaptive_inference.stopping_rule_missing",
            "retain the stopping rule, its parameters and whether it ended the trial",
        ));
    }
}

fn check_shapes(declaration: &AdaptiveDesignDeclaration, refusals: &mut Vec<ExternalRefusal>) {
    let arms = declaration.arm_labels.len();
    let units = declaration.assigned_arm.len();
    let remedy =
        "declare between 2 and 64 distinct arm labels, and as many per-unit records as units";
    let distinct = {
        let mut labels: Vec<&str> = declaration.arm_labels.iter().map(String::as_str).collect();
        labels.sort_unstable();
        labels.windows(2).all(|w| w[0] != w[1]) && labels.iter().all(|l| !l.trim().is_empty())
    };
    if !(2..=MAX_CONTRACT_ARMS).contains(&arms) || !distinct {
        refusals.push(invalid("adaptive_inference.arms_invalid", remedy));
    }
    if !(2..=MAX_CONTRACT_UNITS).contains(&units) || declaration.outcomes.len() != units {
        refusals.push(invalid("adaptive_inference.shape_mismatch", remedy));
    }
    if declaration.assigned_arm.iter().any(|&a| a as usize >= arms) {
        refusals.push(invalid(
            "adaptive_inference.arm_out_of_range",
            "every assigned arm must index a declared arm label",
        ));
    }
    let bad_outcome = declaration
        .outcomes
        .iter()
        .any(|m| m.len() > MAX_UNIT_MEASUREMENTS || m.iter().any(|y| !y.is_finite()));
    if bad_outcome {
        refusals.push(invalid(
            "adaptive_inference.outcome_invalid",
            "outcomes must be finite, at most 1024 measurements per unit",
        ));
    }
    let single = declaration.cell.outcome == super::matrix::OutcomeStructure::SingleOutcome;
    if single && declaration.outcomes.iter().any(|m| m.len() > 1) {
        refusals.push(invalid(
            "adaptive_inference.outcome_invalid",
            "a single-outcome procedure carries at most one measurement per unit",
        ));
    }
    let bernoulli = declaration.cell.assignment == AssignmentFamily::ThompsonBetaBernoulli;
    if bernoulli && declaration.outcomes.iter().flatten().any(|&y| !is_binary(y)) {
        refusals.push(invalid(
            "adaptive_inference.outcome_invalid",
            "Beta-Bernoulli Thompson sampling is defined for outcomes in {0, 1}",
        ));
    }
    if !(declaration.probability_floor > 0.0 && declaration.probability_floor <= 1.0) {
        refusals.push(invalid(
            "adaptive_inference.probability_floor_invalid",
            "declare a probability floor in (0, 1]",
        ));
    }
}

fn check_history(declaration: &AdaptiveDesignDeclaration, refusals: &mut Vec<ExternalRefusal>) {
    let units = declaration.assigned_arm.len();
    let arms = declaration.arm_labels.len();
    if let Some(epochs) = &declaration.policy_history {
        let ordered = epochs.first().is_some_and(|e| e.first_unit == 0)
            && epochs.windows(2).all(|w| w[0].first_unit < w[1].first_unit)
            && epochs.last().is_some_and(|e| (e.first_unit as usize) < units);
        if !ordered {
            refusals.push(invalid(
                "adaptive_inference.policy_history_invalid",
                "policy epochs must start at unit 0, increase strictly and lie within the units",
            ));
        }
    }
    if let Some(rows) = &declaration.assignment_probabilities {
        let mut malformed = rows.len() != units;
        let mut zero_assigned = false;
        for (row, &arm) in rows.iter().zip(&declaration.assigned_arm) {
            if row.len() != arms
                || row.iter().any(|p| !(p.is_finite() && (0.0..=1.0).contains(p)))
                || (row.iter().sum::<f64>() - 1.0).abs() > PROBABILITY_ROW_TOLERANCE
            {
                malformed = true;
            } else if row.get(arm as usize).is_some_and(|&p| p <= 0.0) {
                zero_assigned = true;
            }
        }
        if malformed {
            refusals.push(invalid(
                "adaptive_inference.probabilities_invalid",
                "each unit needs one probability per arm, each in [0, 1], summing to one",
            ));
        }
        if zero_assigned {
            refusals.push(invalid(
                "adaptive_inference.assigned_probability_zero",
                "a unit was assigned an arm the policy gave probability zero: the recorded probabilities are not the policy's",
            ));
        }
        if declaration.cell.assignment == AssignmentFamily::FixedRandomization
            && !malformed
            && rows.windows(2).any(|w| w[0] != w[1])
        {
            refusals.push(invalid(
                "adaptive_inference.rule_mismatch",
                "fixed randomization declares constant probabilities, but the recorded probabilities vary between units",
            ));
        }
    }
    if let Some(exposure) = &declaration.exposure {
        if exposure.len() != units || exposure.iter().any(|&a| a as usize >= arms) {
            refusals.push(invalid(
                "adaptive_inference.shape_mismatch",
                "exposure needs one declared arm per unit",
            ));
        }
    }
}

fn check_monitoring(declaration: &AdaptiveDesignDeclaration, refusals: &mut Vec<ExternalRefusal>) {
    let units = declaration.assigned_arm.len();
    let Some(looks) = &declaration.interim_analyses else { return };
    let ordered = !looks.is_empty()
        && looks.len() <= MAX_CONTRACT_LOOKS
        && looks.windows(2).all(|w| {
            w[0].units_enrolled < w[1].units_enrolled
                && w[0].information_fraction < w[1].information_fraction
        })
        && looks.iter().all(|l| l.information_fraction > 0.0 && l.information_fraction <= 1.0)
        && looks.last().is_some_and(|l| l.units_enrolled as usize == units)
        && looks.first().is_some_and(|l| l.units_enrolled > 0);
    if !ordered {
        refusals.push(invalid(
            "adaptive_inference.looks_not_ordered",
            "interim analyses must increase strictly in units and information, with information in (0, 1], and the last look must include every unit",
        ));
    }
    let wanted_single = declaration.cell.monitoring == MonitoringStrategy::FinalAnalysisOnly;
    if ordered && ((looks.len() == 1) != wanted_single) {
        refusals.push(invalid(
            "adaptive_inference.rule_mismatch",
            "the number of interim analyses disagrees with the declared monitoring strategy",
        ));
    }
    if let Some(availability) = &declaration.availability {
        let consistent = availability.available_at_look.len() == units
            && availability.available_at_look.iter().zip(&declaration.outcomes).all(
                |(look, outcome)| match look {
                    Some(l) => (*l as usize) < looks.len() && !outcome.is_empty(),
                    None => outcome.is_empty(),
                },
            );
        if !consistent {
            refusals.push(invalid(
                "adaptive_inference.shape_mismatch",
                "availability needs one entry per unit, a look within the declared looks exactly where an outcome was observed",
            ));
        } else if ordered {
            let enrolled_by =
                |unit: usize| looks.iter().position(|l| unit < l.units_enrolled as usize);
            let early = availability.available_at_look.iter().enumerate().any(|(unit, look)| {
                look.zip(enrolled_by(unit)).is_some_and(|(l, e)| (l as usize) < e)
            });
            if early {
                refusals.push(invalid(
                    "adaptive_inference.shape_mismatch",
                    "an outcome cannot be available at a look before its unit was enrolled",
                ));
            }
        }
    }
}

fn check_rule(declaration: &AdaptiveDesignDeclaration, refusals: &mut Vec<ExternalRefusal>) {
    let Some(rule) = &declaration.stopping_rule else { return };
    let kind_matches_cell = rule.kind == declaration.cell.stopping;
    let spec_ok = match (rule.kind, rule.spec) {
        (StoppingRuleKind::NoStopping, StoppingSpec::NoStopping) => true,
        (StoppingRuleKind::Efficacy, StoppingSpec::AlphaSpending { total_alpha, .. }) => {
            total_alpha > 0.0
                && total_alpha < 1.0
                && declaration.cell.monitoring == MonitoringStrategy::GroupSequential
        }
        (StoppingRuleKind::Efficacy, StoppingSpec::PosteriorProbabilityBest { threshold }) => {
            threshold > 0.5
                && threshold < 1.0
                && declaration.cell.monitoring != MonitoringStrategy::GroupSequential
        }
        (
            StoppingRuleKind::DecisionTheoretic,
            StoppingSpec::PosteriorExpectedLoss { loss_threshold },
        ) => loss_threshold.is_finite() && loss_threshold > 0.0,
        (StoppingRuleKind::Resource, StoppingSpec::ResourceCap { max_units }) => max_units > 0,
        (StoppingRuleKind::Futility, spec) => !matches!(spec, StoppingSpec::NoStopping),
        _ => false,
    };
    if !kind_matches_cell || !spec_ok {
        refusals.push(invalid(
            "adaptive_inference.rule_mismatch",
            "the stopping rule disagrees with the procedure cell: its kind, parameters and the monitoring strategy must agree",
        ));
    }
}

/// A validated contract: a declaration that is complete and consistent, with
/// its canonical identity.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveDesignContract {
    declaration: AdaptiveDesignDeclaration,
    identity: SemanticDigest,
}

/// Authorization to run one inferential paradigm on a contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteAuthorization {
    /// The procedure cell.
    pub cell: ProcedureCell,
    /// The paradigm authorized.
    pub paradigm: InferenceParadigm,
    /// Identity of the contract it was granted for.
    pub contract_identity: SemanticDigest,
}

impl AdaptiveDesignContract {
    /// Validate a declaration.
    ///
    /// # Errors
    /// One refusal per missing element and per inconsistency; nothing is
    /// defaulted.
    pub fn new(declaration: AdaptiveDesignDeclaration) -> Result<Self, AdaptiveContractError> {
        let mut refusals = Vec::new();
        required_elements(&declaration, &mut refusals);
        check_shapes(&declaration, &mut refusals);
        check_history(&declaration, &mut refusals);
        check_monitoring(&declaration, &mut refusals);
        check_rule(&declaration, &mut refusals);
        if !refusals.is_empty() {
            return Err(AdaptiveContractError { refusals });
        }
        let identity = identity_of(&declaration);
        Ok(Self { declaration, identity })
    }

    /// The validated declaration.
    #[must_use]
    pub const fn declaration(&self) -> &AdaptiveDesignDeclaration {
        &self.declaration
    }

    /// Canonical identity (typed, length-delimited, floats as bit patterns).
    #[must_use]
    pub const fn identity(&self) -> SemanticDigest {
        self.identity
    }

    /// The procedure cell.
    #[must_use]
    pub const fn cell(&self) -> ProcedureCell {
        self.declaration.cell
    }

    /// Number of arms.
    #[must_use]
    pub fn arms(&self) -> usize {
        self.declaration.arm_labels.len()
    }

    /// Number of units.
    #[must_use]
    pub fn units(&self) -> usize {
        self.declaration.assigned_arm.len()
    }

    /// Number of interim analyses.
    #[must_use]
    pub fn looks(&self) -> usize {
        self.declaration.interim_analyses.as_ref().map_or(0, Vec::len)
    }

    /// Units whose outcomes were observed by look `look` and which were
    /// enrolled by then, in enrollment order. Nested in `look`.
    ///
    /// # Panics
    /// Never for a validated contract and `look < looks()`.
    #[must_use]
    pub fn analysis_units(&self, look: usize) -> Vec<usize> {
        let looks = self.declaration.interim_analyses.as_ref().expect("validated contract");
        let availability = self.declaration.availability.as_ref().expect("validated contract");
        let enrolled = looks[look].units_enrolled as usize;
        (0..enrolled)
            .filter(|&unit| {
                availability.available_at_look[unit].is_some_and(|l| (l as usize) <= look)
            })
            .collect()
    }

    /// Whether the fixed-sample paradigm may be asked of this contract's
    /// data: only when nothing about the design adapted or stopped.
    #[must_use]
    pub fn is_adaptive(&self) -> bool {
        self.declaration.cell.is_adaptive()
    }

    /// Authorize an inferential paradigm for this contract's data.
    ///
    /// # Errors
    /// A typed refusal: fixed-sample inference on data from an adaptive
    /// design; a cell that is closed or carried forward; a paradigm with no
    /// route for the cell; exposure that differs from assignment; or a delay
    /// that depends on the outcome.
    ///
    /// # Panics
    /// Never for a contract built by [`Self::new`], which guarantees the
    /// elements this reads.
    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
    )]
    pub fn authorize(
        &self,
        paradigm: InferenceParadigm,
    ) -> Result<RouteAuthorization, ExternalRefusal> {
        let cell = self.declaration.cell;
        if paradigm == InferenceParadigm::FrequentistFixedSample && cell.is_adaptive() {
            return Err(ExternalRefusal::new(
                antecedent_core::reason_code!("adaptive_design_ignored"),
                STAGE_BIND,
                "adaptive_inference.fixed_sample_on_adaptive_data",
            )
            .with_expected("a design fixed in advance: fixed randomization, one final analysis, no stopping rule")
            .with_supplied(cell.identifier())
            .with_remedy(
                "analyse the data with an adaptive-design route: a posterior, adaptively weighted scores, group-sequential boundaries or a confidence sequence",
            ));
        }
        match procedure_status(&cell) {
            CellStatus::Licensed { paradigms } => {
                if !paradigms.contains(&paradigm) {
                    return Err(ExternalRefusal::new(
                        antecedent_core::reason_code!("cell_not_licensed"),
                        STAGE_BIND,
                        "adaptive_inference.paradigm_not_licensed_for_cell",
                    )
                    .with_offending(paradigm.as_str())
                    .with_supplied(cell.identifier())
                    .with_remedy(
                        "use a paradigm the cell is licensed for; a guarantee for one paradigm never transfers to another",
                    ));
                }
            }
            CellStatus::ClosedIncoherent { reason } => {
                return Err(ExternalRefusal::new(
                    antecedent_core::reason_code!("cell_not_licensed"),
                    STAGE_BIND,
                    "adaptive_inference.cell_closed",
                )
                .with_supplied(cell.identifier())
                .with_expected(reason));
            }
            CellStatus::ClosedPairing { reason } => {
                return Err(ExternalRefusal::new(
                    antecedent_core::reason_code!("cell_not_licensed"),
                    STAGE_BIND,
                    "adaptive_inference.pairing_not_licensed",
                )
                .with_supplied(cell.identifier())
                .with_expected(reason));
            }
            CellStatus::CarriedForward { owner, reason } => {
                return Err(ExternalRefusal::new(
                    antecedent_core::reason_code!("cell_not_licensed"),
                    STAGE_BIND,
                    "adaptive_inference.cell_carried_forward",
                )
                .with_supplied(cell.identifier())
                .with_offending(owner)
                .with_expected(reason));
            }
        }
        let exposure = self.declaration.exposure.as_ref().expect("validated contract");
        if exposure != &self.declaration.assigned_arm {
            return Err(declare_bind(
                antecedent_core::reason_code!("adaptive_assumption_violated"),
                "adaptive_inference.exposure_differs_from_assignment",
                "analyse by assignment (intention to treat) with a compliance model, or restrict to units that received the assigned arm; noncompliance is not modelled",
            ));
        }
        let availability = self.declaration.availability.as_ref().expect("validated contract");
        if availability.delay == DelayMechanism::OutcomeDependent {
            return Err(declare_bind(
                antecedent_core::reason_code!("adaptive_assumption_violated"),
                "adaptive_inference.delay_not_ignorable",
                "declare a delay that does not depend on the outcome, or model the delay mechanism; units whose outcomes are missing for outcome-related reasons cannot be dropped",
            ));
        }
        Ok(RouteAuthorization { cell, paradigm, contract_identity: self.identity })
    }
}

fn declare_bind(code: &'static str, detail: &'static str, remedy: &'static str) -> ExternalRefusal {
    ExternalRefusal::new(code, STAGE_BIND, detail).with_remedy(remedy)
}

#[allow(
    clippy::too_many_lines,
    reason = "one canonical-encoding pass over every declared field, in declaration order"
)]
fn identity_of(declaration: &AdaptiveDesignDeclaration) -> SemanticDigest {
    let mut writer = CanonicalWriter::new("antecedent.adaptive_design.contract");
    writer.word(ADAPTIVE_DESIGN_VERSION);
    writer.text(&declaration.label);
    let cell = &declaration.cell;
    writer
        .text(cell.assignment.as_str())
        .text(cell.outcome.as_str())
        .text(cell.monitoring.as_str())
        .text(cell.stopping.as_str());
    writer.texts(&declaration.arm_labels);
    writer.word(declaration.assigned_arm.len() as u64);
    for &arm in &declaration.assigned_arm {
        writer.word(u64::from(arm));
    }
    writer.real(declaration.probability_floor);
    match &declaration.policy_history {
        Some(epochs) => {
            writer.word(1).word(epochs.len() as u64);
            for epoch in epochs {
                writer.word(u64::from(epoch.first_unit)).bytes(epoch.policy_identity.as_bytes());
            }
        }
        None => {
            writer.word(0);
        }
    }
    match &declaration.assignment_probabilities {
        Some(rows) => {
            writer.word(1).word(rows.len() as u64);
            for row in rows {
                writer.reals(row);
            }
        }
        None => {
            writer.word(0);
        }
    }
    match &declaration.exposure {
        Some(exposure) => {
            writer.word(1).word(exposure.len() as u64);
            for &arm in exposure {
                writer.word(u64::from(arm));
            }
        }
        None => {
            writer.word(0);
        }
    }
    match &declaration.availability {
        Some(availability) => {
            writer.word(1).text(availability.delay.as_str());
            writer.word(availability.available_at_look.len() as u64);
            for look in &availability.available_at_look {
                writer.opt_word(look.map(u64::from));
            }
        }
        None => {
            writer.word(0);
        }
    }
    writer.word(declaration.outcomes.len() as u64);
    for measurements in &declaration.outcomes {
        writer.reals(measurements);
    }
    match &declaration.interim_analyses {
        Some(looks) => {
            writer.word(1).word(looks.len() as u64);
            for look in looks {
                writer.word(u64::from(look.units_enrolled)).real(look.information_fraction);
            }
        }
        None => {
            writer.word(0);
        }
    }
    match &declaration.stopping_rule {
        Some(rule) => {
            writer.word(1).text(rule.kind.as_str()).word(u64::from(rule.ended_by_rule));
            match rule.spec {
                StoppingSpec::NoStopping => {
                    writer.text("no_stopping");
                }
                StoppingSpec::AlphaSpending { spending, total_alpha, sidedness } => {
                    writer
                        .text("alpha_spending")
                        .text(spending.as_str())
                        .real(total_alpha)
                        .text(sidedness.as_str());
                }
                StoppingSpec::PosteriorProbabilityBest { threshold } => {
                    writer.text("posterior_probability_best").real(threshold);
                }
                StoppingSpec::PosteriorExpectedLoss { loss_threshold } => {
                    writer.text("posterior_expected_loss").real(loss_threshold);
                }
                StoppingSpec::ResourceCap { max_units } => {
                    writer.text("resource_cap").word(u64::from(max_units));
                }
            }
        }
        None => {
            writer.word(0);
        }
    }
    SemanticDigest::from_bytes(writer.finish())
}
