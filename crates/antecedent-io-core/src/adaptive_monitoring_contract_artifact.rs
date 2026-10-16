//! Portable adaptive monitoring contract (`adaptive_monitoring_contract_v1`).
//!
//! A bounded, checksummed container holding the whole analysis contract of one
//! adaptive experiment and the inferential paradigm an inference from it
//! belongs to:
//!
//! * the procedure: assignment policy, outcome structure, monitoring strategy
//!   and stopping rule, with the stopping rule's parameters and whether it
//!   ended the trial;
//! * the retained history: assignment-policy epochs, every unit's realized
//!   assignment probabilities, exposure, outcomes and availability, and the
//!   interim analyses;
//! * the paradigm (Bayesian posterior, decision-theoretic stopping,
//!   frequentist fixed-sample, adaptively weighted frequentist, group
//!   sequential frequentist or anytime-valid frequentist);
//! * for an alpha-spending rule, the efficacy boundaries it spends; and
//! * the observation accounting of the interim analyses: how many observations
//!   each look used, how many distinct observations that is, and that pooling
//!   the looks as independent evidence is refused; and
//! * optionally, the declared participant-level procedure (a procedure
//!   section): the arms' priors with every historical source, discount and
//!   effective sample size, the responses per participant and their
//!   correlation, the allocation floor, the decisions and the confidence-sequence
//!   settings, and the reading at every performed look (the prior updates,
//!   posterior claim, decision and anytime-valid claim). The section is
//!   additive: a contract without it is encoded exactly as before, and a
//!   reader that does not know the section refuses it.
//!
//! A consumer trusts none of the stored derived values. It rebuilds the
//! contract through the same completeness and consistency checks (so a missing
//! element or an inconsistency is refused, one detail per defect), recomputes
//! the canonical identity, asks the contract to admit the stored paradigm
//! (so a fixed-sample analysis of an adaptive design, or a paradigm the
//! procedure cell is not licensed for, is refused), recomputes the boundaries
//! by numerical integration and replays the observation accounting through the
//! evidence ledger. Given a procedure section it also rebuilds the declared
//! procedure through its own checks, recomputes the allocation of every batch
//! from the completed participants and the declared (frozen) priors and
//! compares it with the retained probabilities, replays the decisions so the
//! procedure neither continued past nor stopped without them, and recomputes
//! every stored reading. A resealed edit of the stopping rule, of a
//! probability, of the paradigm, of a stored boundary, of a borrowed source
//! or discount, or of a stored reading is refused, and a consumer that
//! retained the identity independently refuses a consistently resealed
//! contract of another experiment.
//!
//! The artifact makes no statistical claim. It carries no interval, posterior
//! or coverage statement; the guarantee routes of every cell are closed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ExternalRefusal;
use antecedent_contract::adaptive_design::{
    AdaptiveContractError, AdaptiveDesignContract, AdaptiveDesignDeclaration, AssignmentFamily,
    BoundarySidedness, DelayMechanism, InferenceParadigm, InterimAnalysis, MonitoringStrategy,
    OutcomeAvailability, OutcomeStructure, PolicyEpoch, ProcedureCell, SpendingKind,
    StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
};
use antecedent_contract::evidence_ledger::parse_digest_hex;
use antecedent_core::{CanonicalWriter, SemanticDigest};
use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError, AnytimeSettings, ArmPrior, BetaPrior, BorrowingPlan, Boundaries,
    CapacityError, CorrelationPrior, DecisionRules, FutilityRule, HistoricalPrior, LookDecision,
    LookReading, MAX_RETAINED_LOOKS, ParticipantDeclaration, ParticipantHistory, ParticipantSpec,
    Sidedness, SpendingFunction, boundaries as spend_boundaries,
};
use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;

/// The artifact major version this reader writes and accepts.
pub const ADAPTIVE_MONITORING_CONTRACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const ADAPTIVE_MONITORING_CONTRACT_FEATURE: &str = "adaptive_monitoring_contract_v1";
/// Most bytes a monitoring contract artifact may occupy.
pub const MAX_ADAPTIVE_MONITORING_CONTRACT_BYTES: usize = 256 * 1024 * 1024;
/// Absolute tolerance when comparing a stored boundary with its recomputation
/// (transcendental functions may differ in the last bits across platforms).
pub const BOUNDARY_REPLAY_TOLERANCE: f64 = 1e-9;

/// The kind of procedure section this reader replays.
pub const PROCEDURE_KIND: &str = "beta_binomial_participants";

const ARTIFACT_KIND: &str = crate::artifact_kinds::ADAPTIVE_MONITORING_CONTRACT;
const META_SECTION: &str = "adaptive_monitoring_contract_meta";

crate::sectioned_artifact_enum! {
    /// Why a monitoring contract artifact was refused or could not be produced.
    #[derive(Clone, Debug, PartialEq, Eq)]
    AdaptiveMonitoringContractError {
        noun: "adaptive monitoring contract",
        identity: "adaptive_inference.wrong_contract",
        variants {
            /// The contract or paradigm checks refused the stored declaration.
            Refused {
                /// Registered reason code of the first refusal.
                code: &'static str,
                /// Namespaced detail of the first refusal.
                detail: String,
                /// Every refusal's detail and its remedy.
                message: String,
            },
            /// A stored derived value differs from its recomputation.
            ReplayMismatch(&'static str),
        }
        display(f) {
            Self::Refused { detail, message, .. } => write!(f, "{detail}: {message}"),
            Self::ReplayMismatch(what) => write!(
                f,
                "adaptive_inference.report_replay_mismatch: stored {what} does not replay"
            ),
        }
    }
}

impl AdaptiveMonitoringContractError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a contract or paradigm refusal, a changed identity and a
    /// stored value that does not replay; absent for corruption, unsupported
    /// versions, limits and encoding failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, String, String)> {
        match self {
            Self::Refused { code, detail, message } => {
                Some((*code, detail.clone(), message.clone()))
            }
            Self::IdentityMismatch { field } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "adaptive_inference.wrong_contract".to_owned(),
                format!("{field} changed"),
            )),
            Self::ReplayMismatch(what) => Some((
                antecedent_core::reason_code!("invalid_argument"),
                "adaptive_inference.report_replay_mismatch".to_owned(),
                format!("the stored {what} does not replay"),
            )),
            _ => None,
        }
    }
}

type Error = AdaptiveMonitoringContractError;

fn malformed(text: impl Into<String>) -> Error {
    Error::Malformed(text.into())
}

fn refused_with(refusals: &[ExternalRefusal]) -> Error {
    let first = refusals.first();
    Error::Refused {
        code: first.map_or(antecedent_core::reason_code!("invalid_argument"), |r| r.code),
        detail: first.map_or_else(String::new, |r| r.detail.clone()),
        message: refusals
            .iter()
            .map(|r| format!("{}: {}", r.detail, r.remedy.unwrap_or("")))
            .collect::<Vec<_>>()
            .join("; "),
    }
}

impl From<AdaptiveContractError> for Error {
    fn from(error: AdaptiveContractError) -> Self {
        refused_with(&error.refusals)
    }
}

// -- wire types ---------------------------------------------------------------------------

/// One policy epoch on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyEpochWire {
    /// First unit governed by the policy state.
    pub first_unit: u32,
    /// Policy state identity (hex).
    pub identity: String,
}

/// Outcome availability on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvailabilityWire {
    /// First look at which each outcome was available.
    pub available_at_look: Vec<Option<u32>>,
    /// `immediate`, `ignorable_administrative` or `outcome_dependent`.
    pub delay: String,
}

/// One interim analysis on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookWire {
    /// Units enrolled at the look.
    pub units_enrolled: u32,
    /// Information fraction, as `f64` bits.
    pub information_fraction_bits: u64,
}

/// A stopping rule on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleWire {
    /// Kind name.
    pub kind: String,
    /// Whether the rule ended the trial.
    pub ended_by_rule: bool,
    /// Specification name.
    pub spec: String,
    /// Spending function name, for alpha spending.
    pub spending: Option<String>,
    /// Sidedness name, for alpha spending.
    pub sidedness: Option<String>,
    /// Total alpha, threshold, loss threshold or futility margin, as `f64`
    /// bits.
    pub parameter_bits: Option<u64>,
    /// The futility threshold, as `f64` bits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub second_parameter_bits: Option<u64>,
    /// Resource cap in units.
    pub max_units: Option<u32>,
}

/// The declaration on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclarationWire {
    /// Label.
    pub label: String,
    /// Assignment family name.
    pub assignment: String,
    /// Outcome structure name.
    pub outcome: String,
    /// Monitoring strategy name.
    pub monitoring: String,
    /// Stopping rule kind name.
    pub stopping: String,
    /// Arm labels.
    pub arm_labels: Vec<String>,
    /// Assigned arm per unit.
    pub assigned_arm: Vec<u32>,
    /// Policy history; absent when it was not retained.
    pub policy_history: Option<Vec<PolicyEpochWire>>,
    /// Per-unit assignment probabilities as `f64` bits; absent when not retained.
    pub assignment_probability_bits: Option<Vec<Vec<u64>>>,
    /// Received arm per unit; absent when not retained.
    pub exposure: Option<Vec<u32>>,
    /// Outcome availability; absent when not retained.
    pub availability: Option<AvailabilityWire>,
    /// Per-unit measurements as `f64` bits.
    pub outcome_bits: Vec<Vec<u64>>,
    /// Interim analyses; absent when not retained.
    pub interim_analyses: Option<Vec<LookWire>>,
    /// Stopping rule; absent when not retained.
    pub stopping_rule: Option<RuleWire>,
    /// Probability floor as `f64` bits.
    pub probability_floor_bits: u64,
}

/// Observation accounting of the interim analyses on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceWire {
    /// Observations each look analysed.
    pub look_sizes: Vec<u64>,
    /// Observation slots across the looks.
    pub declared_slots: u64,
    /// Distinct observations across the looks.
    pub distinct_observations: u64,
    /// Slots a naive count would double count.
    pub repeated_slots: u64,
    /// Ledger state digest (hex).
    pub state_digest: String,
    /// Repeated-use result identity (hex).
    pub reuse_result: String,
    /// Detail of the refusal to pool the looks as independent, when the
    /// ledger refuses; absent when there is no overlap to refuse.
    pub independent_pooling_detail: Option<String>,
}

/// Boundaries on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryWire {
    /// Critical values as `f64` bits.
    pub critical_bits: Vec<u64>,
    /// Alpha spent at each look as `f64` bits.
    pub spend_increment_bits: Vec<u64>,
}

/// A historical source on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalWire {
    /// Source label.
    pub source: String,
    /// Successes among the historical responses.
    pub successes: u64,
    /// Historical responses.
    pub trials: u64,
    /// Power-prior discount, as `f64` bits.
    pub discount_bits: u64,
}

/// The prior of one arm on the wire: the declared initial prior and sources,
/// and the resolved prior with its effective sample size (recomputed by a
/// consumer).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmPriorWire {
    /// First shape of the initial prior, as `f64` bits.
    pub initial_alpha_bits: u64,
    /// Second shape of the initial prior, as `f64` bits.
    pub initial_beta_bits: u64,
    /// Historical sources borrowed from.
    pub historical: Vec<HistoricalWire>,
    /// First shape of the resolved prior, as `f64` bits.
    pub resolved_alpha_bits: u64,
    /// Second shape of the resolved prior, as `f64` bits.
    pub resolved_beta_bits: u64,
    /// Prior effective sample size, as `f64` bits.
    pub effective_sample_size_bits: u64,
    /// Part of it borrowed from history, as `f64` bits.
    pub borrowed_sample_size_bits: u64,
}

/// A futility rule on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FutilityWire {
    /// Margin, as `f64` bits.
    pub margin_bits: u64,
    /// Threshold, as `f64` bits.
    pub threshold_bits: u64,
}

/// The reading at one performed look on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadingWire {
    /// Every number of the reading, as `f64` bits, in the order of
    /// `LookReading::numbers`.
    pub numbers_bits: Vec<u64>,
    /// `continue`, or the name of the reason the procedure stopped.
    pub decision: String,
    /// The arm decided for or selected, when the look stopped the procedure.
    pub arm: Option<u32>,
}

/// The declared participant-level procedure and its readings on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureWire {
    /// Procedure kind, [`PROCEDURE_KIND`].
    pub kind: String,
    /// Binary responses per participant.
    pub responses: u32,
    /// Declared intra-participant correlation, as `f64` bits (zero bits when
    /// the correlation is estimated).
    pub correlation_bits: u64,
    /// The `Beta` prior of an estimated correlation, as `f64` bits of its two
    /// shapes; absent when the correlation is declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_prior_bits: Option<[u64; 2]>,
    /// The prior of each arm.
    pub arms: Vec<ArmPriorWire>,
    /// Participants enrolled at each look of the schedule.
    pub enrolment: Vec<u32>,
    /// Looks between enrolment and completed follow-up.
    pub lag_looks: u32,
    /// Minimum allocation probability, as `f64` bits.
    pub min_exploration_bits: u64,
    /// Expected-loss threshold, as `f64` bits.
    pub loss_threshold_bits: u64,
    /// Futility rule, when declared.
    pub futility: Option<FutilityWire>,
    /// Recruitment cap, when declared.
    pub participant_cap: Option<u32>,
    /// Miscoverage of the confidence-sequence family, as `f64` bits.
    pub anytime_alpha_bits: u64,
    /// Tuning sample size of the sequences, as `f64` bits.
    pub anytime_tuning_bits: u64,
    /// Identity (hex) of the declared procedure.
    pub procedure_identity: String,
    /// The reading at every performed look.
    pub looks: Vec<ReadingWire>,
}

/// Stored metadata of a monitoring contract artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MonitoringContractMeta {
    /// Artifact major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// The retained declaration.
    pub declaration: DeclarationWire,
    /// The inferential paradigm.
    pub paradigm: String,
    /// Contract identity (hex).
    pub contract_identity: String,
    /// Procedure cell identifier.
    pub procedure: String,
    /// Class of the cell in the procedure matrix (`licensed`).
    pub cell_class: String,
    /// Stopping boundaries of an alpha-spending rule.
    pub boundaries: Option<BoundaryWire>,
    /// Observation accounting.
    pub evidence: EvidenceWire,
    /// The declared participant-level procedure and its readings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub participant_procedure: Option<ProcedureWire>,
    /// BLAKE3 identity (hex) of everything above.
    pub identity: String,
}

// -- declaration <-> wire -------------------------------------------------------------------

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn reals(values: &[u64]) -> Vec<f64> {
    values.iter().map(|v| f64::from_bits(*v)).collect()
}

fn rule_to_wire(rule: &StoppingRuleDeclaration) -> RuleWire {
    let mut wire = RuleWire {
        kind: rule.kind.as_str().to_owned(),
        ended_by_rule: rule.ended_by_rule,
        spec: String::new(),
        spending: None,
        sidedness: None,
        parameter_bits: None,
        second_parameter_bits: None,
        max_units: None,
    };
    match rule.spec {
        StoppingSpec::NoStopping => "no_stopping".clone_into(&mut wire.spec),
        StoppingSpec::AlphaSpending { spending, total_alpha, sidedness } => {
            "alpha_spending".clone_into(&mut wire.spec);
            wire.spending = Some(spending.as_str().to_owned());
            wire.sidedness = Some(sidedness.as_str().to_owned());
            wire.parameter_bits = Some(total_alpha.to_bits());
        }
        StoppingSpec::PosteriorProbabilityBest { threshold } => {
            "posterior_probability_best".clone_into(&mut wire.spec);
            wire.parameter_bits = Some(threshold.to_bits());
        }
        StoppingSpec::PosteriorExpectedLoss { loss_threshold } => {
            "posterior_expected_loss".clone_into(&mut wire.spec);
            wire.parameter_bits = Some(loss_threshold.to_bits());
        }
        StoppingSpec::PosteriorFutility { margin, threshold } => {
            "posterior_futility".clone_into(&mut wire.spec);
            wire.parameter_bits = Some(margin.to_bits());
            wire.second_parameter_bits = Some(threshold.to_bits());
        }
        StoppingSpec::ResourceCap { max_units } => {
            "resource_cap".clone_into(&mut wire.spec);
            wire.max_units = Some(max_units);
        }
        _ => "unrecognized".clone_into(&mut wire.spec),
    }
    wire
}

fn rule_from_wire(wire: &RuleWire) -> Result<StoppingRuleDeclaration, Error> {
    let kind = StoppingRuleKind::from_name(&wire.kind)
        .ok_or(Error::UnsupportedSemantics("stopping rule kind"))?;
    let parameter = || {
        wire.parameter_bits
            .map(f64::from_bits)
            .ok_or_else(|| malformed("the stopping rule is missing its parameter"))
    };
    let spec = match wire.spec.as_str() {
        "no_stopping" => StoppingSpec::NoStopping,
        "alpha_spending" => StoppingSpec::AlphaSpending {
            spending: wire
                .spending
                .as_deref()
                .and_then(SpendingKind::from_name)
                .ok_or(Error::UnsupportedSemantics("spending function"))?,
            total_alpha: parameter()?,
            sidedness: wire
                .sidedness
                .as_deref()
                .and_then(BoundarySidedness::from_name)
                .ok_or(Error::UnsupportedSemantics("boundary sidedness"))?,
        },
        "posterior_probability_best" => {
            StoppingSpec::PosteriorProbabilityBest { threshold: parameter()? }
        }
        "posterior_expected_loss" => {
            StoppingSpec::PosteriorExpectedLoss { loss_threshold: parameter()? }
        }
        "posterior_futility" => StoppingSpec::PosteriorFutility {
            margin: parameter()?,
            threshold: wire
                .second_parameter_bits
                .map(f64::from_bits)
                .ok_or_else(|| malformed("a futility rule is missing its threshold"))?,
        },
        "resource_cap" => StoppingSpec::ResourceCap {
            max_units: wire.max_units.ok_or_else(|| malformed("a resource cap needs its units"))?,
        },
        _ => return Err(Error::UnsupportedSemantics("stopping specification")),
    };
    Ok(StoppingRuleDeclaration { kind, spec, ended_by_rule: wire.ended_by_rule })
}

/// The declaration of a contract on the wire.
#[must_use]
pub fn declaration_to_wire(declaration: &AdaptiveDesignDeclaration) -> DeclarationWire {
    let cell = &declaration.cell;
    DeclarationWire {
        label: declaration.label.clone(),
        assignment: cell.assignment.as_str().to_owned(),
        outcome: cell.outcome.as_str().to_owned(),
        monitoring: cell.monitoring.as_str().to_owned(),
        stopping: cell.stopping.as_str().to_owned(),
        arm_labels: declaration.arm_labels.clone(),
        assigned_arm: declaration.assigned_arm.clone(),
        policy_history: declaration.policy_history.as_ref().map(|epochs| {
            epochs
                .iter()
                .map(|e| PolicyEpochWire {
                    first_unit: e.first_unit,
                    identity: e.policy_identity.to_hex(),
                })
                .collect()
        }),
        assignment_probability_bits: declaration
            .assignment_probabilities
            .as_ref()
            .map(|rows| rows.iter().map(|row| bits(row)).collect()),
        exposure: declaration.exposure.clone(),
        availability: declaration.availability.as_ref().map(|a| AvailabilityWire {
            available_at_look: a.available_at_look.clone(),
            delay: a.delay.as_str().to_owned(),
        }),
        outcome_bits: declaration.outcomes.iter().map(|m| bits(m)).collect(),
        interim_analyses: declaration.interim_analyses.as_ref().map(|looks| {
            looks
                .iter()
                .map(|l| LookWire {
                    units_enrolled: l.units_enrolled,
                    information_fraction_bits: l.information_fraction.to_bits(),
                })
                .collect()
        }),
        stopping_rule: declaration.stopping_rule.as_ref().map(rule_to_wire),
        probability_floor_bits: declaration.probability_floor.to_bits(),
    }
}

/// The declaration a wire states, not yet validated.
///
/// # Errors
/// An unknown procedure axis, delay mechanism or stopping specification, or a
/// malformed digest.
pub fn declaration_from_wire(
    wire: &DeclarationWire,
) -> Result<AdaptiveDesignDeclaration, AdaptiveMonitoringContractError> {
    let cell = ProcedureCell {
        assignment: AssignmentFamily::from_name(&wire.assignment)
            .ok_or(Error::UnsupportedSemantics("assignment family"))?,
        outcome: OutcomeStructure::from_name(&wire.outcome)
            .ok_or(Error::UnsupportedSemantics("outcome structure"))?,
        monitoring: MonitoringStrategy::from_name(&wire.monitoring)
            .ok_or(Error::UnsupportedSemantics("monitoring strategy"))?,
        stopping: StoppingRuleKind::from_name(&wire.stopping)
            .ok_or(Error::UnsupportedSemantics("stopping rule kind"))?,
    };
    let policy_history = wire
        .policy_history
        .as_ref()
        .map(|epochs| {
            epochs
                .iter()
                .map(|e| {
                    parse_digest_hex(&e.identity)
                        .map(|policy_identity| PolicyEpoch {
                            first_unit: e.first_unit,
                            policy_identity,
                        })
                        .ok_or_else(|| malformed("a policy identity is not 64 hex digits"))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let availability = match &wire.availability {
        Some(a) => {
            let delay = [
                DelayMechanism::Immediate,
                DelayMechanism::IgnorableAdministrative,
                DelayMechanism::OutcomeDependent,
            ]
            .into_iter()
            .find(|d| d.as_str() == a.delay)
            .ok_or(Error::UnsupportedSemantics("delay mechanism"))?;
            Some(OutcomeAvailability { available_at_look: a.available_at_look.clone(), delay })
        }
        None => None,
    };
    Ok(AdaptiveDesignDeclaration {
        label: wire.label.clone(),
        cell,
        arm_labels: wire.arm_labels.clone(),
        assigned_arm: wire.assigned_arm.clone(),
        policy_history,
        assignment_probabilities: wire
            .assignment_probability_bits
            .as_ref()
            .map(|rows| rows.iter().map(|row| reals(row)).collect()),
        exposure: wire.exposure.clone(),
        availability,
        outcomes: wire.outcome_bits.iter().map(|m| reals(m)).collect(),
        interim_analyses: wire.interim_analyses.as_ref().map(|looks| {
            looks
                .iter()
                .map(|l| InterimAnalysis {
                    units_enrolled: l.units_enrolled,
                    information_fraction: f64::from_bits(l.information_fraction_bits),
                })
                .collect()
        }),
        stopping_rule: wire.stopping_rule.as_ref().map(rule_from_wire).transpose()?,
        probability_floor: f64::from_bits(wire.probability_floor_bits),
    })
}

// -- derived values -----------------------------------------------------------------------

/// The efficacy boundaries an alpha-spending rule spends over the contract's
/// interim analyses; `None` for any other rule.
///
/// # Errors
/// The spending rule cannot be solved (a boundary outside the search bracket).
pub fn contract_boundaries(
    contract: &AdaptiveDesignContract,
) -> Result<Option<Boundaries>, AdaptiveMonitoringContractError> {
    let declaration = contract.declaration();
    let Some(rule) = &declaration.stopping_rule else { return Ok(None) };
    let StoppingSpec::AlphaSpending { spending, total_alpha, sidedness } = rule.spec else {
        return Ok(None);
    };
    let fractions: Vec<f64> = declaration
        .interim_analyses
        .as_ref()
        .map(|looks| looks.iter().map(|l| l.information_fraction).collect())
        .unwrap_or_default();
    let spending = match spending {
        SpendingKind::OBrienFlemingType => SpendingFunction::OBrienFlemingType,
        SpendingKind::PocockType => SpendingFunction::PocockType,
        _ => return Err(Error::UnsupportedSemantics("spending function")),
    };
    let sidedness = match sidedness {
        BoundarySidedness::OneSidedUpper => Sidedness::OneSidedUpper,
        BoundarySidedness::TwoSidedSymmetric => Sidedness::TwoSidedSymmetric,
        _ => return Err(Error::UnsupportedSemantics("boundary sidedness")),
    };
    spend_boundaries(&fractions, total_alpha, spending, sidedness, &|| false)
        .map(Some)
        .map_err(|_| Error::ReplayMismatch("boundaries"))
}

fn evidence_of(contract: &AdaptiveDesignContract) -> Result<EvidenceWire, Error> {
    let evidence = contract.interim_evidence().map_err(|r| refused_with(&[r]))?;
    let look_sizes =
        (0..contract.looks()).map(|l| contract.analysis_units(l).len() as u64).collect();
    let independent_pooling_detail =
        contract.refuse_independent_pooling_of_looks(&evidence).err().map(|refusal| refusal.detail);
    Ok(EvidenceWire {
        look_sizes,
        declared_slots: evidence.mass.declared_slots,
        distinct_observations: evidence.mass.distinct_observations,
        repeated_slots: evidence.mass.repeated_slots,
        state_digest: evidence.ledger.state_digest().to_hex(),
        reuse_result: evidence.reuse.to_hex(),
        independent_pooling_detail,
    })
}

fn boundaries_to_wire(boundaries: &Boundaries) -> BoundaryWire {
    BoundaryWire {
        critical_bits: bits(&boundaries.critical),
        spend_increment_bits: bits(&boundaries.spend_increments),
    }
}

fn compute_identity(meta: &MonitoringContractMeta) -> String {
    let mut writer = CanonicalWriter::new("antecedent.adaptive_inference.monitoring_contract");
    writer.word(u64::from(meta.version));
    writer.text(&meta.feature);
    writer.text(&meta.contract_identity);
    writer.text(&meta.paradigm);
    writer.text(&meta.procedure);
    writer.text(&meta.cell_class);
    match &meta.boundaries {
        Some(b) => {
            writer.word(1);
            for list in [&b.critical_bits, &b.spend_increment_bits] {
                writer.word(list.len() as u64);
                for value in list {
                    writer.word(*value);
                }
            }
        }
        None => {
            writer.word(0);
        }
    }
    let evidence = &meta.evidence;
    writer.word(evidence.look_sizes.len() as u64);
    for size in &evidence.look_sizes {
        writer.word(*size);
    }
    writer
        .word(evidence.declared_slots)
        .word(evidence.distinct_observations)
        .word(evidence.repeated_slots)
        .text(&evidence.state_digest)
        .text(&evidence.reuse_result)
        .opt_text(evidence.independent_pooling_detail.as_deref());
    if let Some(procedure) = &meta.participant_procedure {
        writer
            .word(1)
            .text(&procedure.kind)
            .text(&procedure.procedure_identity)
            .word(procedure.looks.len() as u64);
        for look in &procedure.looks {
            writer.word(look.numbers_bits.len() as u64);
            for value in &look.numbers_bits {
                writer.word(*value);
            }
            writer.text(&look.decision).opt_word(look.arm.map(u64::from));
        }
    }
    writer.finish_hex()
}

// -- participant-level procedure -----------------------------------------------------------

/// The participant-level procedure of a monitoring contract, replayed: the
/// declared procedure and its reading at every performed look.
#[derive(Clone, Debug)]
pub struct ProcedureRecord {
    spec: ParticipantSpec,
    readings: Vec<LookReading>,
}

impl ProcedureRecord {
    /// The declared procedure.
    #[must_use]
    pub const fn spec(&self) -> &ParticipantSpec {
        &self.spec
    }

    /// The recomputed reading at every performed look.
    #[must_use]
    pub fn readings(&self) -> &[LookReading] {
        &self.readings
    }
}

fn procedure_error(error: AdaptiveInferenceError) -> Error {
    let (code, detail, message) = match error {
        AdaptiveInferenceError::PremiseFailed { detail, message } => (
            antecedent_stats::adaptive_inference::participant_procedure::premise_reason_code(),
            detail.to_owned(),
            message.to_owned(),
        ),
        AdaptiveInferenceError::Capacity(capacity) => (
            capacity.reason_code(),
            capacity.detail().to_owned(),
            format!("{capacity}; {}", capacity.remedy()),
        ),
        other => (
            antecedent_core::reason_code!("invalid_argument"),
            "adaptive_inference.input_invalid".to_owned(),
            other.to_string(),
        ),
    };
    Error::Refused { code, detail, message }
}

fn procedure_refusal(detail: &str, message: &str) -> Error {
    Error::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail: detail.to_owned(),
        message: message.to_owned(),
    }
}

fn decision_to_wire(decision: LookDecision) -> (String, Option<u32>) {
    match decision {
        LookDecision::Continue => ("continue".to_owned(), None),
        LookDecision::Stop { reason, arm } => {
            (reason.as_str().to_owned(), Some(u32::try_from(arm).unwrap_or(u32::MAX)))
        }
    }
}

fn procedure_to_wire(spec: &ParticipantSpec, readings: &[LookReading]) -> ProcedureWire {
    let decisions = spec.decisions();
    ProcedureWire {
        kind: PROCEDURE_KIND.to_owned(),
        responses: spec.responses(),
        correlation_bits: spec.correlation().map_or(0, f64::to_bits),
        correlation_prior_bits: spec
            .correlation_prior()
            .map(|prior| [prior.alpha.to_bits(), prior.beta.to_bits()]),
        arms: spec
            .borrowing()
            .arms
            .iter()
            .zip(spec.priors())
            .map(|(declared, resolved)| ArmPriorWire {
                initial_alpha_bits: declared.initial.alpha.to_bits(),
                initial_beta_bits: declared.initial.beta.to_bits(),
                historical: declared
                    .historical
                    .iter()
                    .map(|h| HistoricalWire {
                        source: h.source.clone(),
                        successes: h.successes,
                        trials: h.trials,
                        discount_bits: h.discount.to_bits(),
                    })
                    .collect(),
                resolved_alpha_bits: resolved.prior.alpha.to_bits(),
                resolved_beta_bits: resolved.prior.beta.to_bits(),
                effective_sample_size_bits: resolved.effective_sample_size.to_bits(),
                borrowed_sample_size_bits: resolved.borrowed_sample_size.to_bits(),
            })
            .collect(),
        enrolment: spec.enrolment().to_vec(),
        lag_looks: u32::try_from(spec.lag_looks()).unwrap_or(u32::MAX),
        min_exploration_bits: spec.min_exploration().to_bits(),
        loss_threshold_bits: decisions.loss_threshold.to_bits(),
        futility: decisions.futility.map(|f| FutilityWire {
            margin_bits: f.margin.to_bits(),
            threshold_bits: f.threshold.to_bits(),
        }),
        participant_cap: decisions.participant_cap,
        anytime_alpha_bits: spec.anytime().alpha.to_bits(),
        anytime_tuning_bits: spec.anytime().tuning_samples.to_bits(),
        procedure_identity: spec.identity(),
        looks: readings
            .iter()
            .map(|reading| {
                let (decision, arm) = decision_to_wire(reading.decision);
                ReadingWire { numbers_bits: bits(&reading.numbers()), decision, arm }
            })
            .collect(),
    }
}

/// The declared procedure a wire states, validated through the procedure's
/// own checks.
fn spec_from_wire(wire: &ProcedureWire, arms: usize) -> Result<ParticipantSpec, Error> {
    if wire.kind != PROCEDURE_KIND {
        return Err(Error::UnsupportedSemantics("procedure kind"));
    }
    let borrowing = BorrowingPlan {
        arms: wire
            .arms
            .iter()
            .map(|arm| ArmPrior {
                initial: BetaPrior {
                    alpha: f64::from_bits(arm.initial_alpha_bits),
                    beta: f64::from_bits(arm.initial_beta_bits),
                },
                historical: arm
                    .historical
                    .iter()
                    .map(|h| HistoricalPrior {
                        source: h.source.clone(),
                        successes: h.successes,
                        trials: h.trials,
                        discount: f64::from_bits(h.discount_bits),
                    })
                    .collect(),
            })
            .collect(),
    };
    let declaration = ParticipantDeclaration {
        arms,
        responses: Some(wire.responses),
        correlation: match wire.correlation_prior_bits {
            Some(_) => None,
            None => Some(f64::from_bits(wire.correlation_bits)),
        },
        correlation_prior: wire.correlation_prior_bits.map(|[alpha, beta]| CorrelationPrior {
            alpha: f64::from_bits(alpha),
            beta: f64::from_bits(beta),
        }),
        borrowing: Some(borrowing),
        enrolment: Some(wire.enrolment.clone()),
        lag_looks: wire.lag_looks as usize,
        min_exploration: Some(f64::from_bits(wire.min_exploration_bits)),
        decisions: Some(DecisionRules {
            loss_threshold: f64::from_bits(wire.loss_threshold_bits),
            futility: wire.futility.as_ref().map(|f| FutilityRule {
                margin: f64::from_bits(f.margin_bits),
                threshold: f64::from_bits(f.threshold_bits),
            }),
            participant_cap: wire.participant_cap,
        }),
        anytime: Some(AnytimeSettings {
            alpha: f64::from_bits(wire.anytime_alpha_bits),
            tuning_samples: f64::from_bits(wire.anytime_tuning_bits),
        }),
    };
    ParticipantSpec::new(declaration).map_err(|error| Error::Refused {
        code: error
            .gaps
            .first()
            .map_or(antecedent_core::reason_code!("invalid_argument"), |gap| gap.reason_code()),
        detail: error.gaps.first().map_or_else(String::new, |gap| gap.detail().to_owned()),
        message: error
            .gaps
            .iter()
            .map(|gap| format!("{}: {}", gap.detail(), gap.remedy()))
            .collect::<Vec<_>>()
            .join("; "),
    })
}

/// The retained history of a contract in the terms of the procedure.
fn history_of(
    contract: &AdaptiveDesignContract,
    spec: &ParticipantSpec,
) -> Result<ParticipantHistory, Error> {
    let declaration = contract.declaration();
    let (Some(looks), Some(availability), Some(rows)) = (
        declaration.interim_analyses.as_ref(),
        declaration.availability.as_ref(),
        declaration.assignment_probabilities.as_ref(),
    ) else {
        return Err(procedure_refusal(
            "adaptive_inference.interim_analyses_missing",
            "the contract is not complete",
        ));
    };
    let mut batches = Vec::with_capacity(looks.len());
    let mut previous = 0_u32;
    for look in looks {
        batches.push(look.units_enrolled - previous);
        previous = look.units_enrolled;
    }
    let lag = spec.lag_looks();
    let mut successes = Vec::with_capacity(declaration.outcomes.len());
    let mut unit = 0_usize;
    for (look, &size) in batches.iter().enumerate() {
        for _ in 0..size {
            let measurements = &declaration.outcomes[unit];
            let available = availability.available_at_look[unit];
            if available.is_some_and(|at| at as usize != look + lag) {
                return Err(procedure_refusal(
                    "adaptive_inference.outcome_availability_invalid",
                    "an outcome became available at a look other than its enrolment look plus the declared delay",
                ));
            }
            successes.push(if measurements.is_empty() {
                None
            } else if measurements.len() == spec.responses() as usize {
                let ones = measurements.iter().filter(|&&y| y.to_bits() == 1.0_f64.to_bits()).count();
                Some(u32::try_from(ones).unwrap_or(u32::MAX))
            } else {
                return Err(procedure_refusal(
                    "adaptive_inference.measurements_invalid",
                    "a participant's responses are not the declared number of responses per participant",
                ));
            });
            unit += 1;
        }
    }
    Ok(ParticipantHistory {
        batches,
        arms: declaration.assigned_arm.iter().map(|&a| a as usize).collect(),
        successes,
        assignment: rows.clone(),
    })
}

/// Replay a procedure section against the replayed contract.
fn replay_procedure(
    contract: &AdaptiveDesignContract,
    wire: &ProcedureWire,
) -> Result<ProcedureRecord, Error> {
    if wire.looks.len() > MAX_RETAINED_LOOKS {
        return Err(procedure_error(AdaptiveInferenceError::Capacity(
            CapacityError::RetainedRecord {
                limit: MAX_RETAINED_LOOKS as u64,
                requested: wire.looks.len() as u64,
            },
        )));
    }
    let cell = contract.cell();
    if (cell.assignment, cell.outcome, cell.monitoring, cell.stopping)
        != (
            AssignmentFamily::ThompsonBetaBernoulli,
            OutcomeStructure::RepeatedMeasures,
            MonitoringStrategy::FixedInterimLooks,
            StoppingRuleKind::DecisionTheoretic,
        )
    {
        return Err(procedure_refusal(
            "adaptive_inference.rule_mismatch",
            "a procedure section describes Thompson allocation of participants with repeated binary responses, interim looks and decision-theoretic stopping",
        ));
    }
    let spec = spec_from_wire(wire, contract.arms())?;
    if spec.identity() != wire.procedure_identity {
        return Err(Error::ReplayMismatch("procedure identity"));
    }
    let resolved_matches = wire.arms.len() == spec.priors().len()
        && wire.arms.iter().zip(spec.priors()).all(|(stored, resolved)| {
            within(
                &[
                    stored.resolved_alpha_bits,
                    stored.resolved_beta_bits,
                    stored.effective_sample_size_bits,
                    stored.borrowed_sample_size_bits,
                ],
                &[
                    resolved.prior.alpha,
                    resolved.prior.beta,
                    resolved.effective_sample_size,
                    resolved.borrowed_sample_size,
                ],
            )
        });
    if !resolved_matches {
        return Err(Error::ReplayMismatch("resolved priors"));
    }
    let declaration = contract.declaration();
    let rule_agrees = declaration.stopping_rule.is_some_and(|rule| {
        rule.kind == StoppingRuleKind::DecisionTheoretic
            && matches!(
                rule.spec,
                StoppingSpec::PosteriorExpectedLoss { loss_threshold }
                    if loss_threshold.to_bits() == spec.decisions().loss_threshold.to_bits()
            )
    });
    if !rule_agrees {
        return Err(procedure_refusal(
            "adaptive_inference.rule_mismatch",
            "the contract's stopping rule is not the procedure's expected-loss decision",
        ));
    }
    let history = history_of(contract, &spec)?;
    let readings = spec.read_history(&history, &|| false).map_err(procedure_error)?;
    let ended_by_rule = readings.last().is_some_and(|reading| {
        matches!(reading.decision, LookDecision::Stop { reason, .. } if reason.ended_by_rule())
    });
    if declaration.stopping_rule.is_some_and(|rule| rule.ended_by_rule != ended_by_rule) {
        return Err(procedure_refusal(
            "adaptive_inference.rule_mismatch",
            "whether a decision ended the procedure disagrees with the contract's stopping rule",
        ));
    }
    let stored_matches = wire.looks.len() == readings.len()
        && wire.looks.iter().zip(&readings).all(|(stored, reading)| {
            let (decision, arm) = decision_to_wire(reading.decision);
            stored.decision == decision
                && stored.arm == arm
                && within(&stored.numbers_bits, &reading.numbers())
        });
    if !stored_matches {
        return Err(Error::ReplayMismatch("procedure readings"));
    }
    Ok(ProcedureRecord { spec, readings })
}

// -- consumption --------------------------------------------------------------------------

fn within(stored: &[u64], recomputed: &[f64]) -> bool {
    stored.len() == recomputed.len()
        && stored
            .iter()
            .zip(recomputed)
            .all(|(s, r)| (f64::from_bits(*s) - r).abs() <= BOUNDARY_REPLAY_TOLERANCE)
}

fn check(
    meta: &MonitoringContractMeta,
    expected_identity: Option<&str>,
) -> Result<AdaptiveMonitoringContractArtifact, Error> {
    if meta.version != ADAPTIVE_MONITORING_CONTRACT_VERSION {
        return Err(Error::UnsupportedVersion { version: meta.version });
    }
    if meta.feature != ADAPTIVE_MONITORING_CONTRACT_FEATURE {
        return Err(Error::UnsupportedSemantics("feature marker"));
    }
    let contract = AdaptiveDesignContract::new(declaration_from_wire(&meta.declaration)?)?;
    if contract.identity().to_hex() != meta.contract_identity {
        return Err(Error::IdentityMismatch { field: "contract_identity" });
    }
    let paradigm = InferenceParadigm::from_name(&meta.paradigm)
        .ok_or(Error::UnsupportedSemantics("inferential paradigm"))?;
    contract.admit_paradigm(paradigm).map_err(|r| refused_with(&[r]))?;
    if meta.procedure != contract.cell().identifier() {
        return Err(Error::ReplayMismatch("procedure"));
    }
    if meta.cell_class != "licensed" {
        return Err(Error::ReplayMismatch("cell class"));
    }
    let boundaries = contract_boundaries(&contract)?;
    match (&meta.boundaries, &boundaries) {
        (None, None) => {}
        (Some(stored), Some(computed)) => {
            if !within(&stored.critical_bits, &computed.critical)
                || !within(&stored.spend_increment_bits, &computed.spend_increments)
            {
                return Err(Error::ReplayMismatch("boundaries"));
            }
        }
        _ => return Err(Error::ReplayMismatch("boundaries")),
    }
    if evidence_of(&contract)? != meta.evidence {
        return Err(Error::ReplayMismatch("observation accounting"));
    }
    let procedure = meta
        .participant_procedure
        .as_ref()
        .map(|wire| replay_procedure(&contract, wire))
        .transpose()?;
    let identity = compute_identity(meta);
    if meta.identity != identity {
        return Err(Error::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(Error::IdentityMismatch { field: "retained_identity" });
    }
    Ok(AdaptiveMonitoringContractArtifact { contract, paradigm, boundaries, procedure, identity })
}

fn encode_meta(meta: &MonitoringContractMeta, artifact_id: &str) -> Result<Vec<u8>, Error> {
    if artifact_id.trim().is_empty() {
        return Err(Error::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(|error| Error::Encode(error.to_string()))?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "adaptive_monitoring_contract",
            max_bytes: MAX_ADAPTIVE_MONITORING_CONTRACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection {
            id: META_SECTION,
            content_type: crate::CONTENT_CBOR,
            bytes: &meta_bytes,
        }],
    )
    .map_err(|error| crate::map_artifact_check!(error, AdaptiveMonitoringContractError))
}

fn decode_meta(bytes: &[u8]) -> Result<MonitoringContractMeta, Error> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_ADAPTIVE_MONITORING_CONTRACT_BYTES,
        &[META_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, AdaptiveMonitoringContractError))?;
    let Ok([meta_bytes]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(Error::Malformed("unsupported container layout".into()));
    };
    crate::decode_with_version(&meta_bytes, |version| {
        if version == ADAPTIVE_MONITORING_CONTRACT_VERSION {
            Ok(())
        } else {
            Err(Error::UnsupportedVersion { version })
        }
    })
}

// -- public type -----------------------------------------------------------------------------

/// A validated monitoring contract: the replayed contract, its paradigm, the
/// recomputed boundaries and the identity that binds them.
#[derive(Clone, Debug)]
pub struct AdaptiveMonitoringContractArtifact {
    contract: AdaptiveDesignContract,
    paradigm: InferenceParadigm,
    boundaries: Option<Boundaries>,
    procedure: Option<ProcedureRecord>,
    identity: String,
}

impl AdaptiveMonitoringContractArtifact {
    /// Seal a contract for a paradigm. The result passes the same checks a
    /// consumer applies; the sealed metadata is returned alongside it.
    ///
    /// # Errors
    /// A paradigm the contract's cell is not licensed for, a delay or exposure
    /// that fails the authorization, a boundary that does not solve, or a
    /// ledger bound.
    pub fn seal(
        contract: &AdaptiveDesignContract,
        paradigm: InferenceParadigm,
    ) -> Result<(Self, MonitoringContractMeta), Error> {
        Self::seal_with(contract, paradigm, None)
    }

    /// Seal a contract for a paradigm together with the declared
    /// participant-level procedure it was collected under.
    ///
    /// The retained history is read under the procedure first, so a contract
    /// whose allocation probabilities, decisions or outcome availability the
    /// procedure does not replay cannot be sealed with it.
    ///
    /// # Errors
    /// As [`Self::seal`], and a history that the procedure does not replay.
    pub fn seal_with_procedure(
        contract: &AdaptiveDesignContract,
        paradigm: InferenceParadigm,
        procedure: &ParticipantSpec,
    ) -> Result<(Self, MonitoringContractMeta), Error> {
        Self::seal_with(contract, paradigm, Some(procedure))
    }

    fn seal_with(
        contract: &AdaptiveDesignContract,
        paradigm: InferenceParadigm,
        procedure: Option<&ParticipantSpec>,
    ) -> Result<(Self, MonitoringContractMeta), Error> {
        contract.admit_paradigm(paradigm).map_err(|r| refused_with(&[r]))?;
        let boundaries = contract_boundaries(contract)?;
        let procedure_wire = procedure
            .map(|spec| {
                let history = history_of(contract, spec)?;
                let readings = spec.read_history(&history, &|| false).map_err(procedure_error)?;
                Ok::<_, Error>(procedure_to_wire(spec, &readings))
            })
            .transpose()?;
        let mut meta = MonitoringContractMeta {
            version: ADAPTIVE_MONITORING_CONTRACT_VERSION,
            feature: ADAPTIVE_MONITORING_CONTRACT_FEATURE.to_owned(),
            declaration: declaration_to_wire(contract.declaration()),
            paradigm: paradigm.as_str().to_owned(),
            contract_identity: contract.identity().to_hex(),
            procedure: contract.cell().identifier(),
            cell_class: "licensed".to_owned(),
            boundaries: boundaries.as_ref().map(boundaries_to_wire),
            evidence: evidence_of(contract)?,
            participant_procedure: procedure_wire,
            identity: String::new(),
        };
        meta.identity = compute_identity(&meta);
        let artifact = check(&meta, None)?;
        Ok((artifact, meta))
    }

    /// The replayed contract.
    #[must_use]
    pub const fn contract(&self) -> &AdaptiveDesignContract {
        &self.contract
    }

    /// The inferential paradigm the contract was sealed for.
    #[must_use]
    pub const fn paradigm(&self) -> InferenceParadigm {
        self.paradigm
    }

    /// The procedure cell the experiment ran under.
    #[must_use]
    pub const fn procedure(&self) -> ProcedureCell {
        self.contract.cell()
    }

    /// The stopping rule the inference used, with its parameters.
    #[must_use]
    pub fn stopping_rule(&self) -> Option<StoppingRuleDeclaration> {
        self.contract.declaration().stopping_rule
    }

    /// The recomputed boundaries of an alpha-spending rule.
    #[must_use]
    pub const fn boundaries(&self) -> Option<&Boundaries> {
        self.boundaries.as_ref()
    }

    /// The replayed participant-level procedure and its readings, when the
    /// contract was sealed with one.
    #[must_use]
    pub const fn procedure_record(&self) -> Option<&ProcedureRecord> {
        self.procedure.as_ref()
    }

    /// BLAKE3 identity (hex).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Serialize a sealed contract through the checksummed container.
    ///
    /// # Errors
    /// A contract that does not seal, an encoding failure or an oversized
    /// payload.
    pub fn to_bytes(
        contract: &AdaptiveDesignContract,
        paradigm: InferenceParadigm,
        artifact_id: &str,
    ) -> Result<Vec<u8>, Error> {
        let (_, meta) = Self::seal(contract, paradigm)?;
        encode_meta(&meta, artifact_id)
    }

    /// Serialize a contract sealed with its participant-level procedure.
    ///
    /// # Errors
    /// As [`Self::seal_with_procedure`], an encoding failure or an oversized
    /// payload.
    pub fn to_bytes_with_procedure(
        contract: &AdaptiveDesignContract,
        paradigm: InferenceParadigm,
        procedure: &ParticipantSpec,
        artifact_id: &str,
    ) -> Result<Vec<u8>, Error> {
        let (_, meta) = Self::seal_with_procedure(contract, paradigm, procedure)?;
        encode_meta(&meta, artifact_id)
    }

    /// Consume an artifact by replay and recomputation.
    ///
    /// `expected_identity` is an identity (hex) the consumer retained
    /// independently; when given, a consistently resealed artifact of another
    /// experiment is refused.
    ///
    /// # Errors
    /// Corruption, another major version, a missing or inconsistent contract
    /// element, a paradigm the cell is not licensed for, a stored derived value
    /// that does not replay, or an identity that does not match.
    pub fn from_bytes(bytes: &[u8], expected_identity: Option<&str>) -> Result<Self, Error> {
        check(&decode_meta(bytes)?, expected_identity)
    }

    /// Validate metadata as a consumer would, for tests that reseal edits.
    ///
    /// # Errors
    /// The same refusals as [`Self::from_bytes`].
    #[doc(hidden)]
    pub fn from_meta(
        meta: &MonitoringContractMeta,
        expected_identity: Option<&str>,
    ) -> Result<Self, Error> {
        check(meta, expected_identity)
    }

    /// Recompute the identity of `meta` after an edit, for tests that reseal.
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut MonitoringContractMeta) {
        meta.identity = compute_identity(meta);
    }

    /// Decode stored metadata without validating it, for tests that edit it.
    ///
    /// # Errors
    /// Corruption, an unsupported layout or version.
    #[doc(hidden)]
    pub fn decode_unchecked(bytes: &[u8]) -> Result<MonitoringContractMeta, Error> {
        decode_meta(bytes)
    }
}

/// The digest of a contract's identity, for callers that retain it.
#[must_use]
pub fn contract_identity_hex(digest: SemanticDigest) -> String {
    digest.to_hex()
}
