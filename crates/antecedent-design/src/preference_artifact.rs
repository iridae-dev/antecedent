//! Durable records of accepted preferences and of decisions across a preference set.
//!
//! [`AcceptedPreferencesArtifact`] stores accepted preferences: the declared value model, the
//! basis (the learned set with its relations and ranges, or the estimate with its measure), the
//! learning options and decider selection needed to relearn it, the version, the acknowledged
//! review items and the identity. [`PreferenceDecisionArtifact`] stores a preference contract,
//! the measure it was read with (if any), the evaluation options, the route that answered, the
//! result and the digest of the joint draws it was computed from (the same source digest the
//! decision result artifact retains).
//!
//! Both follow the decision artifacts: a checksummed container with a versioned kind
//! (`accepted_preferences_v1`, `preference_decision_v1`), a CBOR body whose reals are written as
//! IEEE-754 values and read back bit for bit, loading only under identities the consumer
//! retained separately from the bytes (a resealed edit is refused because every identity is
//! recomputed from the decoded content), and replay by recomputation. Replay compares labels,
//! verdicts, identities, counts and statuses exactly, and every real within
//! [`REPLAY_RELATIVE_TOLERANCE`] of the stored value relative to the larger of one and the two
//! magnitudes, since linear-program and fitting outputs need not agree to the last bit across
//! platforms. A declared basis replays without records.

// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{ExternalRefusal, ScientificQuantity};
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_io_core::convert::to_cbor;
use antecedent_io_core::distribution_artifact::{DistributionArtifact, DistributionCalibration};
use antecedent_io_core::error::IoError;
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use antecedent_stats::ConditionalLogitOptions;
use antecedent_stats::linear_program::LpOptions;
use serde::{Deserialize, Serialize};

use crate::accepted_preferences::{
    AcceptedPreferences, AcceptedPreferencesError, LearningOptions, PendingReview, PreferenceBasis,
    select_estimate, select_learned, stated_in_scope,
};
use crate::choice_record::ChoiceRecordSet;
use crate::decision_artifact::{decode_section, encode_section, source_digest};
use crate::decision_contract::{
    ActionKind, DecisionCriterion, HardConstraint, StructuralPolicy, UtilityExpr,
};
use crate::decision_eval::{ConstraintExclusion, SourceReceipt};
use crate::preference_certified::{
    CertifiedActionOutcome, CertifiedOptions, CertifiedPreferenceResult, CertifiedRegion,
    CertifiedVerdict, PreferenceCertifiedError, RegionStatus,
};
use crate::preference_choice_model::{
    ChoiceModelError, ChoiceModelEstimate, ChoiceModelOptions, CovarianceKind, ParameterEstimate,
};
use crate::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferenceContractError, PreferencePolicy,
};
use crate::preference_decision::{
    MeasureOrigin, PreferenceActionOutcome, PreferenceDecisionError, PreferenceDecisionOptions,
    PreferenceDecisionResult, PreferenceMeasure, PreferenceSwitch, PreferenceVerdict,
    SwitchThreshold,
};
use crate::preference_evaluation::{
    PreferenceEvaluation, PreferenceEvaluationError, PreferenceEvaluationOptions, PreferenceRoute,
    decide_on_draws, decide_preferences,
};
use crate::preference_model::{
    Attribute, LinearRelation, PreferenceExpr, PreferenceParameter, PreferenceSet, RelationOrigin,
    RelationSense, ValueModel,
};
use crate::preference_refusal::{CONSUME, EXPORT, refusal};
use crate::preference_set_learning::{
    BindingChoice, LearnedPreferenceSet, ParameterRange, SetLearningError, SetLearningOptions,
};
use crate::signal::at_stage;

/// Relative tolerance of every real a replay compares: `|stored - fresh|` may be at most this
/// times the larger of one, `|stored|` and `|fresh|`.
pub const REPLAY_RELATIVE_TOLERANCE: f64 = 1e-9;

const ACCEPTED_KIND: &str = "accepted_preferences_v1";
const ACCEPTED_SECTION: &str = "accepted_preferences_body";
const DECISION_KIND: &str = "preference_decision_v1";
const DECISION_SECTION: &str = "preference_decision_body";
/// Version word of every body this reader decodes; the shared decoder refuses any other.
const BODY_VERSION: u16 = 1;

/// Why a preference artifact is refused.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum PreferenceArtifactError {
    /// The body could not be encoded or the container could not be written.
    #[error("encoding: {0}")]
    Encode(IoError),
    /// The container or body is corrupt, of another kind or an unsupported version.
    #[error("container: {0}")]
    Container(IoError),
    /// A decoded field names nothing this reader knows.
    #[error("malformed field: {what}")]
    Malformed {
        /// What is malformed.
        what: String,
    },
    /// The decoded or relearned accepted preferences are refused.
    #[error("accepted preferences: {0}")]
    Accepted(AcceptedPreferencesError),
    /// The stored preference contract is refused.
    #[error("preference contract: {0}")]
    Contract(PreferenceContractError),
    /// The evaluation is refused.
    #[error("evaluation: {0}")]
    Evaluation(PreferenceEvaluationError),
    /// The identity recomputed from the decoded content differs from the stored identity.
    #[error("the stored identity does not match the stored content")]
    StoredIdentityMismatch,
    /// The identity differs from the consumer's retained identity.
    #[error("the identity differs from the consumer's retained identity")]
    IdentityExpected,
    /// The result was computed under a different contract than the consumer retained.
    #[error("the result was computed under a different preference contract")]
    ContractExpected,
    /// The result was computed from different draws than the consumer retained.
    #[error("the result was computed from different draws")]
    SourceExpected,
    /// The result names a different contract than the one stored with it.
    #[error("the result was computed under another contract than the one stored")]
    ResultContractMismatch,
    /// Replay was given different draws than the result was computed from.
    #[error("the replay source differs from the stored result's")]
    ReplaySourceDiffers,
    /// Replay was given different records than the preferences were learned from.
    #[error("the replay records differ from the ones the preferences were learned from")]
    ReplayRecordsDiffer,
    /// Recomputation does not reproduce the stored content.
    #[error("recomputation does not reproduce the stored {field}")]
    ReplayMismatch {
        /// The first stored field that differs.
        field: &'static str,
    },
    /// The execution context was cancelled during a replay; nothing was verified.
    #[error("the replay was cancelled")]
    Cancelled,
}

impl PreferenceArtifactError {
    /// Stable detail id, prefixed `preference_artifact.`; a wrapped refusal of another module
    /// keeps its own detail.
    #[must_use]
    pub fn detail(&self) -> &'static str {
        use antecedent_core::reason_details;
        match self {
            Self::Encode(_) => reason_details::preference_artifact::ENCODE_FAILED,
            Self::Container(_) => reason_details::preference_artifact::CONTAINER_REFUSED,
            Self::Malformed { .. } => reason_details::preference_artifact::WIRE_MALFORMED,
            Self::Accepted(inner) => inner.detail(),
            Self::Contract(inner) => inner.detail(),
            Self::Evaluation(inner) => inner.detail(),
            Self::StoredIdentityMismatch => reason_details::preference_artifact::IDENTITY_RESEALED,
            Self::IdentityExpected => reason_details::preference_artifact::IDENTITY_EXPECTED,
            Self::ContractExpected => reason_details::preference_artifact::CONTRACT_EXPECTED,
            Self::SourceExpected => reason_details::preference_artifact::SOURCE_EXPECTED,
            Self::ResultContractMismatch => {
                reason_details::preference_artifact::RESULT_CONTRACT_MISMATCH
            }
            Self::ReplaySourceDiffers => reason_details::preference_artifact::REPLAY_SOURCE_DIFFERS,
            Self::ReplayRecordsDiffer => reason_details::preference_artifact::REPLAY_RECORDS_DIFFER,
            Self::ReplayMismatch { .. } => reason_details::preference_artifact::REPLAY_MISMATCH,
            Self::Cancelled => reason_details::preference_artifact::CANCELLED,
        }
    }
}

impl From<PreferenceArtifactError> for IoError {
    /// A container error stays itself; every other refusal keeps the registered code and
    /// detail of its structured refusal, with the detail id leading the message.
    fn from(error: PreferenceArtifactError) -> Self {
        match error {
            PreferenceArtifactError::Encode(inner) | PreferenceArtifactError::Container(inner) => {
                inner
            }
            other => {
                let refusal = other.to_refusal();
                Self::Refused {
                    code: refusal.code,
                    message: format!("{}: {other}", refusal.detail),
                }
            }
        }
    }
}

use PreferenceArtifactError as E;

/// `Err(Cancelled)` when a context is given and its token is set.
fn observe(ctx: Option<&ExecutionContext>) -> Result<(), PreferenceArtifactError> {
    if ctx.is_some_and(|ctx| ctx.cancellation.is_cancelled()) {
        return Err(E::Cancelled);
    }
    Ok(())
}

/// A refused relearning, with the learner's cancellation reported as the replay's.
fn relearn_error(error: AcceptedPreferencesError) -> PreferenceArtifactError {
    match error {
        AcceptedPreferencesError::SetLearning(SetLearningError::Cancelled)
        | AcceptedPreferencesError::ChoiceModel(ChoiceModelError::Cancelled) => E::Cancelled,
        other => E::Accepted(other),
    }
}

/// A refused re-evaluation, with the route's cancellation reported as the replay's.
fn reevaluation_error(error: PreferenceEvaluationError) -> PreferenceArtifactError {
    match error {
        PreferenceEvaluationError::Exact(PreferenceDecisionError::Cancelled)
        | PreferenceEvaluationError::Certified(PreferenceCertifiedError::Cancelled) => E::Cancelled,
        other => E::Evaluation(other),
    }
}

fn malformed(what: impl Into<String>) -> PreferenceArtifactError {
    E::Malformed { what: what.into() }
}

// ---------------------------------------------------------------------------------------------
// Value model wire
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExprWire {
    Const(f64),
    Attribute(usize),
    Parameter(usize),
    Neg(Box<ExprWire>),
    Add(Box<ExprWire>, Box<ExprWire>),
    Sub(Box<ExprWire>, Box<ExprWire>),
    Mul(Box<ExprWire>, Box<ExprWire>),
    Min(Box<ExprWire>, Box<ExprWire>),
    Max(Box<ExprWire>, Box<ExprWire>),
    Exp(Box<ExprWire>),
    Ln(Box<ExprWire>),
}

impl From<&PreferenceExpr> for ExprWire {
    fn from(expr: &PreferenceExpr) -> Self {
        let b = |e: &PreferenceExpr| Box::new(Self::from(e));
        match expr {
            PreferenceExpr::Const(v) => Self::Const(*v),
            PreferenceExpr::Attribute(k) => Self::Attribute(*k),
            PreferenceExpr::Parameter(j) => Self::Parameter(*j),
            PreferenceExpr::Neg(a) => Self::Neg(b(a)),
            PreferenceExpr::Add(x, y) => Self::Add(b(x), b(y)),
            PreferenceExpr::Sub(x, y) => Self::Sub(b(x), b(y)),
            PreferenceExpr::Mul(x, y) => Self::Mul(b(x), b(y)),
            PreferenceExpr::Min(x, y) => Self::Min(b(x), b(y)),
            PreferenceExpr::Max(x, y) => Self::Max(b(x), b(y)),
            PreferenceExpr::Exp(a) => Self::Exp(b(a)),
            PreferenceExpr::Ln(a) => Self::Ln(b(a)),
        }
    }
}

impl From<ExprWire> for PreferenceExpr {
    fn from(wire: ExprWire) -> Self {
        let b = |e: Box<ExprWire>| Box::new(Self::from(*e));
        match wire {
            ExprWire::Const(v) => Self::Const(v),
            ExprWire::Attribute(k) => Self::Attribute(k),
            ExprWire::Parameter(j) => Self::Parameter(j),
            ExprWire::Neg(a) => Self::Neg(b(a)),
            ExprWire::Add(x, y) => Self::Add(b(x), b(y)),
            ExprWire::Sub(x, y) => Self::Sub(b(x), b(y)),
            ExprWire::Mul(x, y) => Self::Mul(b(x), b(y)),
            ExprWire::Min(x, y) => Self::Min(b(x), b(y)),
            ExprWire::Max(x, y) => Self::Max(b(x), b(y)),
            ExprWire::Exp(a) => Self::Exp(b(a)),
            ExprWire::Ln(a) => Self::Ln(b(a)),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AttributeWire {
    id: String,
    units: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterWire {
    id: String,
    units: String,
    lower: f64,
    upper: f64,
    assumption_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TermWire {
    parameter: usize,
    coefficient: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SenseWire {
    AtLeast,
    AtMost,
    Equal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OriginWire {
    Declared,
    ShapeRule,
    Choice { record_id: String, rejected_option: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RelationWire {
    id: String,
    terms: Vec<TermWire>,
    sense: SenseWire,
    bound: f64,
    origin: OriginWire,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SetWire {
    parameters: Vec<ParameterWire>,
    relations: Vec<RelationWire>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelWire {
    attributes: Vec<AttributeWire>,
    preferences: SetWire,
    value: ExprWire,
    utility_units: String,
}

fn set_wire(set: &PreferenceSet) -> SetWire {
    SetWire {
        parameters: set
            .parameters
            .iter()
            .map(|p| ParameterWire {
                id: p.id.clone(),
                units: p.units.clone(),
                lower: p.lower,
                upper: p.upper,
                assumption_id: p.assumption_id.clone(),
            })
            .collect(),
        relations: set
            .relations
            .iter()
            .map(|r| RelationWire {
                id: r.id.clone(),
                terms: r
                    .terms
                    .iter()
                    .map(|(parameter, coefficient)| TermWire {
                        parameter: *parameter,
                        coefficient: *coefficient,
                    })
                    .collect(),
                sense: match r.sense {
                    RelationSense::AtLeast => SenseWire::AtLeast,
                    RelationSense::AtMost => SenseWire::AtMost,
                    RelationSense::Equal => SenseWire::Equal,
                },
                bound: r.bound,
                origin: match &r.origin {
                    RelationOrigin::Declared => OriginWire::Declared,
                    RelationOrigin::ShapeRule => OriginWire::ShapeRule,
                    RelationOrigin::Choice { record_id, rejected_option } => OriginWire::Choice {
                        record_id: record_id.clone(),
                        rejected_option: rejected_option.clone(),
                    },
                },
            })
            .collect(),
    }
}

fn set_from(wire: SetWire) -> PreferenceSet {
    PreferenceSet {
        parameters: wire
            .parameters
            .into_iter()
            .map(|p| PreferenceParameter {
                id: p.id,
                units: p.units,
                lower: p.lower,
                upper: p.upper,
                assumption_id: p.assumption_id,
            })
            .collect(),
        relations: wire
            .relations
            .into_iter()
            .map(|r| LinearRelation {
                id: r.id,
                terms: r.terms.into_iter().map(|t| (t.parameter, t.coefficient)).collect(),
                sense: match r.sense {
                    SenseWire::AtLeast => RelationSense::AtLeast,
                    SenseWire::AtMost => RelationSense::AtMost,
                    SenseWire::Equal => RelationSense::Equal,
                },
                bound: r.bound,
                origin: match r.origin {
                    OriginWire::Declared => RelationOrigin::Declared,
                    OriginWire::ShapeRule => RelationOrigin::ShapeRule,
                    OriginWire::Choice { record_id, rejected_option } => {
                        RelationOrigin::Choice { record_id, rejected_option }
                    }
                },
            })
            .collect(),
    }
}

fn model_wire(model: &ValueModel) -> ModelWire {
    ModelWire {
        attributes: model
            .attributes
            .iter()
            .map(|a| AttributeWire { id: a.id.clone(), units: a.units.clone() })
            .collect(),
        preferences: set_wire(&model.preferences),
        value: ExprWire::from(&model.value),
        utility_units: model.utility_units.clone(),
    }
}

fn model_from(wire: ModelWire) -> ValueModel {
    ValueModel {
        attributes: wire
            .attributes
            .into_iter()
            .map(|a| Attribute { id: a.id, units: a.units })
            .collect(),
        preferences: set_from(wire.preferences),
        value: wire.value.into(),
        utility_units: wire.utility_units,
    }
}

// ---------------------------------------------------------------------------------------------
// Measure, learned set, estimate, options and review items
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MeasureOriginWire {
    Declared { assumption_id: String },
    ChoiceEstimate { estimate_identity: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeasureWire {
    points: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
    labels: Option<Vec<String>>,
    origin: MeasureOriginWire,
    unresolved: Vec<usize>,
}

fn measure_wire(measure: &PreferenceMeasure) -> MeasureWire {
    MeasureWire {
        points: measure.points.clone(),
        weights: measure.weights.clone(),
        labels: measure.labels.clone(),
        origin: match &measure.origin {
            MeasureOrigin::Declared { assumption_id } => {
                MeasureOriginWire::Declared { assumption_id: assumption_id.clone() }
            }
            MeasureOrigin::ChoiceEstimate { estimate_identity } => {
                MeasureOriginWire::ChoiceEstimate { estimate_identity: estimate_identity.clone() }
            }
        },
        unresolved: measure.unresolved.clone(),
    }
}

fn measure_from(wire: MeasureWire) -> PreferenceMeasure {
    PreferenceMeasure {
        points: wire.points,
        weights: wire.weights,
        labels: wire.labels,
        origin: match wire.origin {
            MeasureOriginWire::Declared { assumption_id } => {
                MeasureOrigin::Declared { assumption_id }
            }
            MeasureOriginWire::ChoiceEstimate { estimate_identity } => {
                MeasureOrigin::ChoiceEstimate { estimate_identity }
            }
        },
        unresolved: wire.unresolved,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RangeWire {
    parameter_id: String,
    declared: (f64, f64),
    learned: (f64, f64),
    varied_in_records: bool,
    narrowed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingWire {
    record_id: String,
    rejected_option: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LearnedWire {
    model_identity: String,
    records_digest: String,
    decider: Option<String>,
    set: SetWire,
    ranges: Vec<RangeWire>,
    binding: Vec<BindingWire>,
    tolerance: f64,
    n_records: usize,
    assumptions: Vec<String>,
}

fn learned_wire(learned: &LearnedPreferenceSet) -> LearnedWire {
    LearnedWire {
        model_identity: learned.model_identity.clone(),
        records_digest: learned.records_digest.clone(),
        decider: learned.decider.clone(),
        set: set_wire(&learned.set),
        ranges: learned
            .ranges
            .iter()
            .map(|r| RangeWire {
                parameter_id: r.parameter_id.clone(),
                declared: r.declared,
                learned: r.learned,
                varied_in_records: r.varied_in_records,
                narrowed: r.narrowed,
            })
            .collect(),
        binding: learned
            .binding
            .iter()
            .map(|b| BindingWire {
                record_id: b.record_id.clone(),
                rejected_option: b.rejected_option.clone(),
            })
            .collect(),
        tolerance: learned.tolerance,
        n_records: learned.n_records,
        assumptions: learned.assumptions.clone(),
    }
}

fn learned_from(wire: LearnedWire) -> LearnedPreferenceSet {
    LearnedPreferenceSet {
        model_identity: wire.model_identity,
        records_digest: wire.records_digest,
        decider: wire.decider,
        set: set_from(wire.set),
        ranges: wire
            .ranges
            .into_iter()
            .map(|r| ParameterRange {
                parameter_id: r.parameter_id,
                declared: r.declared,
                learned: r.learned,
                varied_in_records: r.varied_in_records,
                narrowed: r.narrowed,
            })
            .collect(),
        binding: wire
            .binding
            .into_iter()
            .map(|b| BindingChoice { record_id: b.record_id, rejected_option: b.rejected_option })
            .collect(),
        tolerance: wire.tolerance,
        n_records: wire.n_records,
        assumptions: wire.assumptions,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EstimateParameterWire {
    parameter_id: String,
    estimate: f64,
    standard_error: f64,
    interval: (f64, f64),
    declared: (f64, f64),
    inside_declared: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CovarianceWire {
    Model,
    ClusteredByDecider,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EstimateWire {
    identity: String,
    model_identity: String,
    records_digest: String,
    decider: Option<String>,
    consistency: f64,
    consistency_standard_error: f64,
    parameters: Vec<EstimateParameterWire>,
    not_learnable: Vec<String>,
    log_likelihood: f64,
    null_log_likelihood: f64,
    n_records: usize,
    n_deciders: usize,
    covariance_kind: CovarianceWire,
    coefficient_covariance: Vec<f64>,
    level: f64,
    interval_calibration: DistributionCalibration,
    measure: MeasureWire,
    share_outside_declared: f64,
    assumptions: Vec<String>,
}

fn estimate_wire(estimate: &ChoiceModelEstimate) -> EstimateWire {
    EstimateWire {
        identity: estimate.identity.clone(),
        model_identity: estimate.model_identity.clone(),
        records_digest: estimate.records_digest.clone(),
        decider: estimate.decider.clone(),
        consistency: estimate.consistency,
        consistency_standard_error: estimate.consistency_standard_error,
        parameters: estimate
            .parameters
            .iter()
            .map(|p| EstimateParameterWire {
                parameter_id: p.parameter_id.clone(),
                estimate: p.estimate,
                standard_error: p.standard_error,
                interval: p.interval,
                declared: p.declared,
                inside_declared: p.inside_declared,
            })
            .collect(),
        not_learnable: estimate.not_learnable.clone(),
        log_likelihood: estimate.log_likelihood,
        null_log_likelihood: estimate.null_log_likelihood,
        n_records: estimate.n_records,
        n_deciders: estimate.n_deciders,
        covariance_kind: match estimate.covariance_kind {
            CovarianceKind::Model => CovarianceWire::Model,
            CovarianceKind::ClusteredByDecider => CovarianceWire::ClusteredByDecider,
        },
        coefficient_covariance: estimate.coefficient_covariance.clone(),
        level: estimate.level,
        interval_calibration: estimate.interval_calibration,
        measure: measure_wire(&estimate.measure),
        share_outside_declared: estimate.share_outside_declared,
        assumptions: estimate.assumptions.clone(),
    }
}

fn estimate_from(wire: EstimateWire) -> ChoiceModelEstimate {
    ChoiceModelEstimate {
        identity: wire.identity,
        model_identity: wire.model_identity,
        records_digest: wire.records_digest,
        decider: wire.decider,
        consistency: wire.consistency,
        consistency_standard_error: wire.consistency_standard_error,
        parameters: wire
            .parameters
            .into_iter()
            .map(|p| ParameterEstimate {
                parameter_id: p.parameter_id,
                estimate: p.estimate,
                standard_error: p.standard_error,
                interval: p.interval,
                declared: p.declared,
                inside_declared: p.inside_declared,
            })
            .collect(),
        not_learnable: wire.not_learnable,
        log_likelihood: wire.log_likelihood,
        null_log_likelihood: wire.null_log_likelihood,
        n_records: wire.n_records,
        n_deciders: wire.n_deciders,
        covariance_kind: match wire.covariance_kind {
            CovarianceWire::Model => CovarianceKind::Model,
            CovarianceWire::ClusteredByDecider => CovarianceKind::ClusteredByDecider,
        },
        coefficient_covariance: wire.coefficient_covariance,
        level: wire.level,
        interval_calibration: wire.interval_calibration,
        measure: measure_from(wire.measure),
        share_outside_declared: wire.share_outside_declared,
        assumptions: wire.assumptions,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BasisWire {
    Declared,
    CompatibleSet(Box<LearnedWire>),
    ChoiceEstimate(Box<EstimateWire>),
}

fn basis_wire(basis: &PreferenceBasis) -> BasisWire {
    match basis {
        PreferenceBasis::Declared => BasisWire::Declared,
        PreferenceBasis::CompatibleSet(learned) => {
            BasisWire::CompatibleSet(Box::new(learned_wire(learned)))
        }
        PreferenceBasis::ChoiceEstimate(estimate) => {
            BasisWire::ChoiceEstimate(Box::new(estimate_wire(estimate)))
        }
    }
}

fn basis_from(wire: BasisWire) -> PreferenceBasis {
    match wire {
        BasisWire::Declared => PreferenceBasis::Declared,
        BasisWire::CompatibleSet(learned) => {
            PreferenceBasis::CompatibleSet(Box::new(learned_from(*learned)))
        }
        BasisWire::ChoiceEstimate(estimate) => {
            PreferenceBasis::ChoiceEstimate(Box::new(estimate_from(*estimate)))
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LpWire {
    feasibility_tolerance: f64,
    optimality_tolerance: f64,
    max_pivots: usize,
}

fn lp_wire(lp: LpOptions) -> LpWire {
    LpWire {
        feasibility_tolerance: lp.feasibility_tolerance,
        optimality_tolerance: lp.optimality_tolerance,
        max_pivots: lp.max_pivots,
    }
}

fn lp_from(wire: LpWire) -> LpOptions {
    LpOptions {
        feasibility_tolerance: wire.feasibility_tolerance,
        optimality_tolerance: wire.optimality_tolerance,
        max_pivots: wire.max_pivots,
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SetOptionsWire {
    tolerance: f64,
    conflict_search_size: usize,
    lp: LpWire,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FitWire {
    max_iterations: usize,
    gradient_tolerance: f64,
    separation_bound: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChoiceOptionsWire {
    level: f64,
    n_draws: usize,
    seed: u64,
    fit: FitWire,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LearningWire {
    CompatibleSet(SetOptionsWire),
    ChoiceEstimate(ChoiceOptionsWire),
}

fn learning_wire(learning: &LearningOptions) -> LearningWire {
    match learning {
        LearningOptions::CompatibleSet(o) => LearningWire::CompatibleSet(SetOptionsWire {
            tolerance: o.tolerance,
            conflict_search_size: o.conflict_search_size,
            lp: lp_wire(o.lp),
        }),
        LearningOptions::ChoiceEstimate(o) => LearningWire::ChoiceEstimate(ChoiceOptionsWire {
            level: o.level,
            n_draws: o.n_draws,
            seed: o.seed,
            fit: FitWire {
                max_iterations: o.fit.max_iterations,
                gradient_tolerance: o.fit.gradient_tolerance,
                separation_bound: o.fit.separation_bound,
            },
        }),
    }
}

fn learning_from(wire: LearningWire) -> LearningOptions {
    match wire {
        LearningWire::CompatibleSet(o) => LearningOptions::CompatibleSet(SetLearningOptions {
            tolerance: o.tolerance,
            conflict_search_size: o.conflict_search_size,
            lp: lp_from(o.lp),
        }),
        LearningWire::ChoiceEstimate(o) => LearningOptions::ChoiceEstimate(ChoiceModelOptions {
            level: o.level,
            n_draws: o.n_draws,
            seed: o.seed,
            fit: ConditionalLogitOptions {
                max_iterations: o.fit.max_iterations,
                gradient_tolerance: o.fit.gradient_tolerance,
                separation_bound: o.fit.separation_bound,
            },
        }),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PendingWire {
    ParameterNotVaried { parameter_id: String },
    ParameterNotNarrowed { parameter_id: String },
    ToleranceApplied { tolerance: f64 },
    ParameterNotLearnable { parameter_id: String },
    EstimateOutsideDeclared { parameter_id: String },
    DrawsOutsideDeclared { share: f64 },
    StatedComparisonsUsed { count: usize },
}

fn pending_wire(item: &PendingReview) -> PendingWire {
    match item {
        PendingReview::ParameterNotVaried { parameter_id } => {
            PendingWire::ParameterNotVaried { parameter_id: parameter_id.clone() }
        }
        PendingReview::ParameterNotNarrowed { parameter_id } => {
            PendingWire::ParameterNotNarrowed { parameter_id: parameter_id.clone() }
        }
        PendingReview::ToleranceApplied { tolerance } => {
            PendingWire::ToleranceApplied { tolerance: *tolerance }
        }
        PendingReview::ParameterNotLearnable { parameter_id } => {
            PendingWire::ParameterNotLearnable { parameter_id: parameter_id.clone() }
        }
        PendingReview::EstimateOutsideDeclared { parameter_id } => {
            PendingWire::EstimateOutsideDeclared { parameter_id: parameter_id.clone() }
        }
        PendingReview::DrawsOutsideDeclared { share } => {
            PendingWire::DrawsOutsideDeclared { share: *share }
        }
        PendingReview::StatedComparisonsUsed { count } => {
            PendingWire::StatedComparisonsUsed { count: *count }
        }
    }
}

fn pending_from(wire: PendingWire) -> PendingReview {
    match wire {
        PendingWire::ParameterNotVaried { parameter_id } => {
            PendingReview::ParameterNotVaried { parameter_id }
        }
        PendingWire::ParameterNotNarrowed { parameter_id } => {
            PendingReview::ParameterNotNarrowed { parameter_id }
        }
        PendingWire::ToleranceApplied { tolerance } => {
            PendingReview::ToleranceApplied { tolerance }
        }
        PendingWire::ParameterNotLearnable { parameter_id } => {
            PendingReview::ParameterNotLearnable { parameter_id }
        }
        PendingWire::EstimateOutsideDeclared { parameter_id } => {
            PendingReview::EstimateOutsideDeclared { parameter_id }
        }
        PendingWire::DrawsOutsideDeclared { share } => {
            PendingReview::DrawsOutsideDeclared { share }
        }
        PendingWire::StatedComparisonsUsed { count } => {
            PendingReview::StatedComparisonsUsed { count }
        }
    }
}

/// Canonical identity of accepted preferences: the declared model identity, then the CBOR wire
/// of the basis and of the learning options, the number of stated comparisons, the version and
/// the CBOR wire of the acknowledged items.
pub(crate) fn accepted_identity(
    declared_identity: &str,
    basis: &PreferenceBasis,
    learning: Option<&LearningOptions>,
    stated_comparisons: usize,
    version: u32,
    acknowledged: &[PendingReview],
) -> String {
    let acknowledged: Vec<PendingWire> = acknowledged.iter().map(pending_wire).collect();
    let mut out = CanonicalWriter::new("antecedent.accepted_preferences.v1");
    out.text(declared_identity)
        .bytes(&to_cbor(&basis_wire(basis)).unwrap_or_default())
        .bytes(&to_cbor(&learning.map(learning_wire)).unwrap_or_default())
        .word(u64::try_from(stated_comparisons).unwrap_or(u64::MAX))
        .word(u64::from(version))
        .bytes(&to_cbor(&acknowledged).unwrap_or_default());
    out.finish_hex()
}

// ---------------------------------------------------------------------------------------------
// Accepted preferences artifact
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcceptedBody {
    version: u16,
    identity: String,
    declared: ModelWire,
    basis: BasisWire,
    learning: Option<LearningWire>,
    stated_comparisons: usize,
    accepted_version: u32,
    acknowledged: Vec<PendingWire>,
}

fn accepted_body(accepted: &AcceptedPreferences) -> AcceptedBody {
    AcceptedBody {
        version: BODY_VERSION,
        identity: accepted.identity().to_owned(),
        declared: model_wire(accepted.declared_model()),
        basis: basis_wire(accepted.basis()),
        learning: accepted.learning_options().as_ref().map(learning_wire),
        stated_comparisons: accepted.stated_comparisons(),
        accepted_version: accepted.version(),
        acknowledged: accepted.acknowledged().iter().map(pending_wire).collect(),
    }
}

fn accepted_from_body(body: AcceptedBody) -> Result<AcceptedPreferences, PreferenceArtifactError> {
    let acknowledged: Vec<PendingReview> =
        body.acknowledged.into_iter().map(pending_from).collect();
    let accepted = AcceptedPreferences::from_parts(
        model_from(body.declared),
        basis_from(body.basis),
        body.learning.map(learning_from),
        body.stated_comparisons,
        body.accepted_version,
        &acknowledged,
    )
    .map_err(E::Accepted)?;
    if accepted.identity() != body.identity {
        return Err(E::StoredIdentityMismatch);
    }
    Ok(accepted)
}

fn encode(
    kind: &str,
    section: &str,
    artifact_id: &str,
    note: &str,
    body: &impl Serialize,
) -> Result<Vec<u8>, PreferenceArtifactError> {
    let bytes = to_cbor(body).map_err(E::Encode)?;
    encode_section(kind, section, artifact_id, note, &bytes).map_err(E::Encode)
}

/// Accepted preferences in a durable, identity-bound record.
#[derive(Clone, Debug, PartialEq)]
pub struct AcceptedPreferencesArtifact {
    accepted: AcceptedPreferences,
}

impl AcceptedPreferencesArtifact {
    /// Hold accepted preferences for storage.
    #[must_use]
    pub fn new(accepted: AcceptedPreferences) -> Self {
        Self { accepted }
    }

    /// The accepted preferences.
    #[must_use]
    pub fn accepted(&self) -> &AcceptedPreferences {
        &self.accepted
    }

    /// Identity of the accepted preferences; the identity a consumer retains.
    #[must_use]
    pub fn identity(&self) -> &str {
        self.accepted.identity()
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An encoding failure, an oversized payload or a blank artifact id.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, PreferenceArtifactError> {
        encode(
            ACCEPTED_KIND,
            ACCEPTED_SECTION,
            artifact_id,
            "accepted_preferences",
            &accepted_body(&self.accepted),
        )
    }

    /// Load under the consumer's independently retained identity.
    ///
    /// # Errors
    /// A corrupt container, another kind or an unsupported version; content whose parts do not
    /// belong together; an identity recomputed from the content that differs from the stored
    /// one (a resealed edit) or from `expected_identity`.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: &str,
    ) -> Result<Self, PreferenceArtifactError> {
        let body: AcceptedBody =
            decode_section(bytes, ACCEPTED_KIND, ACCEPTED_SECTION).map_err(E::Container)?;
        let accepted = accepted_from_body(body)?;
        if accepted.identity() != expected_identity {
            return Err(E::IdentityExpected);
        }
        Ok(Self { accepted })
    }

    /// Relearn from `records` with the stored options and decider selection and require the
    /// stored basis to be reproduced: the records digest, the number of stated comparisons in
    /// the selected scope and every label, count and identity exactly, and every real within
    /// [`REPLAY_RELATIVE_TOLERANCE`]. A declared basis replays without reading the records.
    ///
    /// # Errors
    /// Records other than the ones the basis was learned from; a refused relearning; any
    /// differing field.
    pub fn replay(&self, records: &ChoiceRecordSet) -> Result<(), PreferenceArtifactError> {
        self.replay_in(records, None)
    }

    /// [`Self::replay`] under an execution context: the relearning reads the context's
    /// cancellation token as the learner or the estimator does.
    ///
    /// # Errors
    /// As [`Self::replay`]; [`PreferenceArtifactError::Cancelled`] once the token is set.
    pub fn replay_with_context(
        &self,
        records: &ChoiceRecordSet,
        ctx: &ExecutionContext,
    ) -> Result<(), PreferenceArtifactError> {
        self.replay_in(records, Some(ctx))
    }

    fn replay_in(
        &self,
        records: &ChoiceRecordSet,
        ctx: Option<&ExecutionContext>,
    ) -> Result<(), PreferenceArtifactError> {
        observe(ctx)?;
        let model = self.accepted.declared_model();
        let decider = self.accepted.decider();
        let stated = || {
            ensure(
                stated_in_scope(records, decider) == self.accepted.stated_comparisons(),
                "stated_comparisons",
            )
        };
        match (self.accepted.basis(), self.accepted.learning_options()) {
            (PreferenceBasis::Declared, _) => Ok(()),
            (PreferenceBasis::CompatibleSet(stored), Some(LearningOptions::CompatibleSet(o))) => {
                let fresh =
                    select_learned(model, records, decider, o, ctx).map_err(relearn_error)?;
                compare_learned(stored, &fresh)?;
                stated()
            }
            (PreferenceBasis::ChoiceEstimate(stored), Some(LearningOptions::ChoiceEstimate(o))) => {
                let fresh =
                    select_estimate(model, records, decider, o, ctx).map_err(relearn_error)?;
                compare_estimate(stored, &fresh)?;
                stated()
            }
            _ => Err(E::Accepted(AcceptedPreferencesError::OptionsMismatch { option: "basis" })),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Contract wire
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum UtilityWire {
    Const(f64),
    Input(usize),
    Neg(Box<UtilityWire>),
    Add(Box<UtilityWire>, Box<UtilityWire>),
    Sub(Box<UtilityWire>, Box<UtilityWire>),
    Mul(Box<UtilityWire>, Box<UtilityWire>),
    Min(Box<UtilityWire>, Box<UtilityWire>),
    Max(Box<UtilityWire>, Box<UtilityWire>),
}

impl From<&UtilityExpr> for UtilityWire {
    fn from(expr: &UtilityExpr) -> Self {
        let b = |e: &UtilityExpr| Box::new(Self::from(e));
        match expr {
            UtilityExpr::Const(v) => Self::Const(*v),
            UtilityExpr::Input(k) => Self::Input(*k),
            UtilityExpr::Neg(a) => Self::Neg(b(a)),
            UtilityExpr::Add(x, y) => Self::Add(b(x), b(y)),
            UtilityExpr::Sub(x, y) => Self::Sub(b(x), b(y)),
            UtilityExpr::Mul(x, y) => Self::Mul(b(x), b(y)),
            UtilityExpr::Min(x, y) => Self::Min(b(x), b(y)),
            UtilityExpr::Max(x, y) => Self::Max(b(x), b(y)),
        }
    }
}

impl From<UtilityWire> for UtilityExpr {
    fn from(wire: UtilityWire) -> Self {
        let b = |e: Box<UtilityWire>| Box::new(Self::from(*e));
        match wire {
            UtilityWire::Const(v) => Self::Const(v),
            UtilityWire::Input(k) => Self::Input(k),
            UtilityWire::Neg(a) => Self::Neg(b(a)),
            UtilityWire::Add(x, y) => Self::Add(b(x), b(y)),
            UtilityWire::Sub(x, y) => Self::Sub(b(x), b(y)),
            UtilityWire::Mul(x, y) => Self::Mul(b(x), b(y)),
            UtilityWire::Min(x, y) => Self::Min(b(x), b(y)),
            UtilityWire::Max(x, y) => Self::Max(b(x), b(y)),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CriterionWire {
    PosteriorExpectedUtility,
    PosteriorExpectedLoss,
    ThresholdProbability { threshold: f64 },
    Quantile { p: f64 },
    MinimaxOverIdentifiedSet,
    MaximinOverStructures,
    Regret,
    ExpectedRegret,
}

impl From<DecisionCriterion> for CriterionWire {
    fn from(c: DecisionCriterion) -> Self {
        match c {
            DecisionCriterion::PosteriorExpectedUtility => Self::PosteriorExpectedUtility,
            DecisionCriterion::PosteriorExpectedLoss => Self::PosteriorExpectedLoss,
            DecisionCriterion::ThresholdProbability { threshold } => {
                Self::ThresholdProbability { threshold }
            }
            DecisionCriterion::Quantile { p } => Self::Quantile { p },
            DecisionCriterion::MinimaxOverIdentifiedSet => Self::MinimaxOverIdentifiedSet,
            DecisionCriterion::MaximinOverStructures => Self::MaximinOverStructures,
            DecisionCriterion::Regret => Self::Regret,
            DecisionCriterion::ExpectedRegret => Self::ExpectedRegret,
        }
    }
}

impl From<CriterionWire> for DecisionCriterion {
    fn from(c: CriterionWire) -> Self {
        match c {
            CriterionWire::PosteriorExpectedUtility => Self::PosteriorExpectedUtility,
            CriterionWire::PosteriorExpectedLoss => Self::PosteriorExpectedLoss,
            CriterionWire::ThresholdProbability { threshold } => {
                Self::ThresholdProbability { threshold }
            }
            CriterionWire::Quantile { p } => Self::Quantile { p },
            CriterionWire::MinimaxOverIdentifiedSet => Self::MinimaxOverIdentifiedSet,
            CriterionWire::MaximinOverStructures => Self::MaximinOverStructures,
            CriterionWire::Regret => Self::Regret,
            CriterionWire::ExpectedRegret => Self::ExpectedRegret,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum InputWire {
    Quantity(ScientificQuantityWire),
    Fixed { value: f64, units: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionWire {
    id: String,
    kind: String,
    inputs: Vec<InputWire>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConstraintWire {
    id: String,
    expr: UtilityWire,
    bound: f64,
    min_probability: f64,
    units: String,
    applies_to: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContractWire {
    model: ModelWire,
    actions: Vec<ActionWire>,
    criterion: CriterionWire,
    constraints: Vec<ConstraintWire>,
    target_population: String,
    horizon: u32,
    structural_policy: String,
    preference_policy: String,
}

fn kind_name(kind: ActionKind) -> &'static str {
    match kind {
        ActionKind::Intervention => "intervention",
        ActionKind::Policy => "policy",
        ActionKind::Regime => "regime",
        ActionKind::Study => "study",
        ActionKind::External => "external",
    }
}

fn kind_from(name: &str) -> Result<ActionKind, PreferenceArtifactError> {
    Ok(match name {
        "intervention" => ActionKind::Intervention,
        "policy" => ActionKind::Policy,
        "regime" => ActionKind::Regime,
        "study" => ActionKind::Study,
        "external" => ActionKind::External,
        _ => return Err(malformed(format!("action kind `{name}`"))),
    })
}

fn structural_name(policy: StructuralPolicy) -> &'static str {
    match policy {
        StructuralPolicy::RequireInvariantBestAction => "require_invariant_best_action",
        StructuralPolicy::Maximin => "maximin",
        StructuralPolicy::BayesOverStructures => "bayes_over_structures",
        StructuralPolicy::ReportOnly => "report_only",
    }
}

fn structural_from(name: &str) -> Result<StructuralPolicy, PreferenceArtifactError> {
    Ok(match name {
        "require_invariant_best_action" => StructuralPolicy::RequireInvariantBestAction,
        "maximin" => StructuralPolicy::Maximin,
        "bayes_over_structures" => StructuralPolicy::BayesOverStructures,
        "report_only" => StructuralPolicy::ReportOnly,
        _ => return Err(malformed(format!("structural policy `{name}`"))),
    })
}

fn preference_policy_name(policy: PreferencePolicy) -> &'static str {
    match policy {
        PreferencePolicy::RequireInvariantBestAction => "require_invariant_best_action",
        PreferencePolicy::Maximin => "maximin",
        PreferencePolicy::MinimaxRegret => "minimax_regret",
        PreferencePolicy::Weighted => "weighted",
        PreferencePolicy::ReportOnly => "report_only",
    }
}

fn preference_policy_from(name: &str) -> Result<PreferencePolicy, PreferenceArtifactError> {
    Ok(match name {
        "require_invariant_best_action" => PreferencePolicy::RequireInvariantBestAction,
        "maximin" => PreferencePolicy::Maximin,
        "minimax_regret" => PreferencePolicy::MinimaxRegret,
        "weighted" => PreferencePolicy::Weighted,
        "report_only" => PreferencePolicy::ReportOnly,
        _ => return Err(malformed(format!("preference policy `{name}`"))),
    })
}

fn contract_wire(contract: &PreferenceContract) -> ContractWire {
    ContractWire {
        model: model_wire(&contract.model),
        actions: contract
            .actions
            .iter()
            .map(|a| ActionWire {
                id: a.id.clone(),
                kind: kind_name(a.kind).to_owned(),
                inputs: a
                    .inputs
                    .iter()
                    .map(|input| match input {
                        ActionInput::Quantity(q) => {
                            InputWire::Quantity(ScientificQuantityWire::from(q))
                        }
                        ActionInput::Fixed { value, units } => {
                            InputWire::Fixed { value: *value, units: units.clone() }
                        }
                    })
                    .collect(),
            })
            .collect(),
        criterion: contract.criterion.into(),
        constraints: contract
            .constraints
            .iter()
            .map(|c| ConstraintWire {
                id: c.id.clone(),
                expr: UtilityWire::from(&c.expr),
                bound: c.bound,
                min_probability: c.min_probability,
                units: c.units.clone(),
                applies_to: c.applies_to.clone(),
            })
            .collect(),
        target_population: contract.target_population.clone(),
        horizon: contract.horizon,
        structural_policy: structural_name(contract.structural_policy).to_owned(),
        preference_policy: preference_policy_name(contract.preference_policy).to_owned(),
    }
}

fn contract_from(wire: ContractWire) -> Result<PreferenceContract, PreferenceArtifactError> {
    let actions = wire
        .actions
        .into_iter()
        .map(|a| {
            let inputs = a
                .inputs
                .into_iter()
                .map(|input| match input {
                    InputWire::Quantity(q) => ScientificQuantity::try_from(q)
                        .map(ActionInput::Quantity)
                        .map_err(|why| malformed(format!("quantity of action `{}`: {why}", a.id))),
                    InputWire::Fixed { value, units } => Ok(ActionInput::Fixed { value, units }),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(PreferenceAction { kind: kind_from(&a.kind)?, id: a.id, inputs })
        })
        .collect::<Result<Vec<_>, PreferenceArtifactError>>()?;
    Ok(PreferenceContract {
        model: model_from(wire.model),
        actions,
        criterion: wire.criterion.into(),
        constraints: wire
            .constraints
            .into_iter()
            .map(|c| HardConstraint {
                id: c.id,
                expr: c.expr.into(),
                bound: c.bound,
                min_probability: c.min_probability,
                units: c.units,
                applies_to: c.applies_to,
            })
            .collect(),
        target_population: wire.target_population,
        horizon: wire.horizon,
        structural_policy: structural_from(&wire.structural_policy)?,
        preference_policy: preference_policy_from(&wire.preference_policy)?,
    })
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluationOptionsWire {
    tie_tolerance: f64,
    vertex_cap: usize,
    lp: LpWire,
    max_boxes: usize,
    min_width_fraction: f64,
    certified_tie_tolerance: f64,
}

fn options_wire(options: &PreferenceEvaluationOptions) -> EvaluationOptionsWire {
    EvaluationOptionsWire {
        tie_tolerance: options.exact.tie_tolerance,
        vertex_cap: options.exact.vertex_cap,
        lp: lp_wire(options.exact.lp),
        max_boxes: options.certified.max_boxes,
        min_width_fraction: options.certified.min_width_fraction,
        certified_tie_tolerance: options.certified.tie_tolerance,
    }
}

fn options_from(wire: EvaluationOptionsWire) -> PreferenceEvaluationOptions {
    PreferenceEvaluationOptions {
        exact: PreferenceDecisionOptions {
            tie_tolerance: wire.tie_tolerance,
            vertex_cap: wire.vertex_cap,
            lp: lp_from(wire.lp),
        },
        certified: CertifiedOptions {
            max_boxes: wire.max_boxes,
            min_width_fraction: wire.min_width_fraction,
            tie_tolerance: wire.certified_tie_tolerance,
        },
    }
}

// ---------------------------------------------------------------------------------------------
// Result wire
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExclusionWire {
    constraint_id: String,
    probability: f64,
    required: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_field_names, reason = "wire names mirror the source receipt fields")]
struct ReceiptWire {
    provider_id: String,
    snapshot_id: String,
    rng_id: String,
    causal_contract_id: String,
}

fn exclusions_wire(exclusions: &[ConstraintExclusion]) -> Vec<ExclusionWire> {
    exclusions
        .iter()
        .map(|e| ExclusionWire {
            constraint_id: e.constraint_id.clone(),
            probability: e.probability,
            required: e.required,
        })
        .collect()
}

fn exclusions_from(wire: Vec<ExclusionWire>) -> Vec<ConstraintExclusion> {
    wire.into_iter()
        .map(|e| ConstraintExclusion {
            constraint_id: e.constraint_id,
            probability: e.probability,
            required: e.required,
        })
        .collect()
}

fn receipt_wire(receipt: &SourceReceipt) -> ReceiptWire {
    ReceiptWire {
        provider_id: receipt.provider_id.clone(),
        snapshot_id: receipt.snapshot_id.clone(),
        rng_id: receipt.rng_id.clone(),
        causal_contract_id: receipt.causal_contract_id.clone(),
    }
}

fn receipt_from(wire: ReceiptWire) -> SourceReceipt {
    SourceReceipt {
        provider_id: wire.provider_id,
        snapshot_id: wire.snapshot_id,
        rng_id: wire.rng_id,
        causal_contract_id: wire.causal_contract_id,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactOutcomeWire {
    id: String,
    admissible: bool,
    exclusions: Vec<ExclusionWire>,
    coefficients: Option<Vec<f64>>,
    value_range: Option<(f64, f64)>,
    necessarily_best: bool,
    uniquely_best: bool,
    dominated_by: Vec<String>,
    possibly_best: Option<bool>,
    worst_case_regret: Option<f64>,
    measure_share: Option<f64>,
    measure_value: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThresholdWire {
    parameter_id: String,
    value: f64,
    first_leads_above: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SwitchWire {
    first: String,
    second: String,
    margin_coefficients: Vec<f64>,
    margin_range: (f64, f64),
    favours_first_at: Vec<f64>,
    favours_second_at: Vec<f64>,
    threshold: Option<ThresholdWire>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExactVerdictWire {
    NecessarilyBest(String),
    NeverBeaten { action: String, ties_with: Vec<String> },
    PreferenceDependent { possibly_best: Vec<String> },
    WorstCaseChoice(String),
    MinimaxRegretChoice(String),
    MeasureChoice { action: String, share: f64 },
    Tied(Vec<String>),
    ReportOnly,
    NoAdmissibleAction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactWire {
    contract_identity: String,
    policy: String,
    parameter_ids: Vec<String>,
    parameter_ranges: Vec<(f64, f64)>,
    actions: Vec<ExactOutcomeWire>,
    switches: Vec<SwitchWire>,
    verdict: ExactVerdictWire,
    draw_share_best_everywhere: Option<f64>,
    unresolved_measure_share: Option<f64>,
    assumptions: Vec<String>,
    source: ReceiptWire,
    n_draws: usize,
}

fn exact_wire(result: &PreferenceDecisionResult) -> ExactWire {
    ExactWire {
        contract_identity: result.contract_identity.clone(),
        policy: preference_policy_name(result.policy).to_owned(),
        parameter_ids: result.parameter_ids.clone(),
        parameter_ranges: result.parameter_ranges.clone(),
        actions: result
            .actions
            .iter()
            .map(|a| ExactOutcomeWire {
                id: a.id.clone(),
                admissible: a.admissible,
                exclusions: exclusions_wire(&a.exclusions),
                coefficients: a.coefficients.clone(),
                value_range: a.value_range,
                necessarily_best: a.necessarily_best,
                uniquely_best: a.uniquely_best,
                dominated_by: a.dominated_by.clone(),
                possibly_best: a.possibly_best,
                worst_case_regret: a.worst_case_regret,
                measure_share: a.measure_share,
                measure_value: a.measure_value,
            })
            .collect(),
        switches: result
            .switches
            .iter()
            .map(|s| SwitchWire {
                first: s.first.clone(),
                second: s.second.clone(),
                margin_coefficients: s.margin_coefficients.clone(),
                margin_range: s.margin_range,
                favours_first_at: s.favours_first_at.clone(),
                favours_second_at: s.favours_second_at.clone(),
                threshold: s.threshold.as_ref().map(|t| ThresholdWire {
                    parameter_id: t.parameter_id.clone(),
                    value: t.value,
                    first_leads_above: t.first_leads_above,
                }),
            })
            .collect(),
        verdict: match &result.verdict {
            PreferenceVerdict::NecessarilyBest(a) => ExactVerdictWire::NecessarilyBest(a.clone()),
            PreferenceVerdict::NeverBeaten { action, ties_with } => ExactVerdictWire::NeverBeaten {
                action: action.clone(),
                ties_with: ties_with.clone(),
            },
            PreferenceVerdict::PreferenceDependent { possibly_best } => {
                ExactVerdictWire::PreferenceDependent { possibly_best: possibly_best.clone() }
            }
            PreferenceVerdict::WorstCaseChoice(a) => ExactVerdictWire::WorstCaseChoice(a.clone()),
            PreferenceVerdict::MinimaxRegretChoice(a) => {
                ExactVerdictWire::MinimaxRegretChoice(a.clone())
            }
            PreferenceVerdict::MeasureChoice { action, share } => {
                ExactVerdictWire::MeasureChoice { action: action.clone(), share: *share }
            }
            PreferenceVerdict::Tied(ids) => ExactVerdictWire::Tied(ids.clone()),
            PreferenceVerdict::ReportOnly => ExactVerdictWire::ReportOnly,
            PreferenceVerdict::NoAdmissibleAction => ExactVerdictWire::NoAdmissibleAction,
        },
        draw_share_best_everywhere: result.draw_share_best_everywhere,
        unresolved_measure_share: result.unresolved_measure_share,
        assumptions: result.assumptions.clone(),
        source: receipt_wire(&result.source),
        n_draws: result.n_draws,
    }
}

fn exact_from(wire: ExactWire) -> Result<PreferenceDecisionResult, PreferenceArtifactError> {
    Ok(PreferenceDecisionResult {
        contract_identity: wire.contract_identity,
        policy: preference_policy_from(&wire.policy)?,
        parameter_ids: wire.parameter_ids,
        parameter_ranges: wire.parameter_ranges,
        actions: wire
            .actions
            .into_iter()
            .map(|a| PreferenceActionOutcome {
                id: a.id,
                admissible: a.admissible,
                exclusions: exclusions_from(a.exclusions),
                coefficients: a.coefficients,
                value_range: a.value_range,
                necessarily_best: a.necessarily_best,
                uniquely_best: a.uniquely_best,
                dominated_by: a.dominated_by,
                possibly_best: a.possibly_best,
                worst_case_regret: a.worst_case_regret,
                measure_share: a.measure_share,
                measure_value: a.measure_value,
            })
            .collect(),
        switches: wire
            .switches
            .into_iter()
            .map(|s| PreferenceSwitch {
                first: s.first,
                second: s.second,
                margin_coefficients: s.margin_coefficients,
                margin_range: s.margin_range,
                favours_first_at: s.favours_first_at,
                favours_second_at: s.favours_second_at,
                threshold: s.threshold.map(|t| SwitchThreshold {
                    parameter_id: t.parameter_id,
                    value: t.value,
                    first_leads_above: t.first_leads_above,
                }),
            })
            .collect(),
        verdict: match wire.verdict {
            ExactVerdictWire::NecessarilyBest(a) => PreferenceVerdict::NecessarilyBest(a),
            ExactVerdictWire::NeverBeaten { action, ties_with } => {
                PreferenceVerdict::NeverBeaten { action, ties_with }
            }
            ExactVerdictWire::PreferenceDependent { possibly_best } => {
                PreferenceVerdict::PreferenceDependent { possibly_best }
            }
            ExactVerdictWire::WorstCaseChoice(a) => PreferenceVerdict::WorstCaseChoice(a),
            ExactVerdictWire::MinimaxRegretChoice(a) => PreferenceVerdict::MinimaxRegretChoice(a),
            ExactVerdictWire::MeasureChoice { action, share } => {
                PreferenceVerdict::MeasureChoice { action, share }
            }
            ExactVerdictWire::Tied(ids) => PreferenceVerdict::Tied(ids),
            ExactVerdictWire::ReportOnly => PreferenceVerdict::ReportOnly,
            ExactVerdictWire::NoAdmissibleAction => PreferenceVerdict::NoAdmissibleAction,
        },
        draw_share_best_everywhere: wire.draw_share_best_everywhere,
        unresolved_measure_share: wire.unresolved_measure_share,
        assumptions: wire.assumptions,
        source: receipt_from(wire.source),
        n_draws: wire.n_draws,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CertifiedOutcomeWire {
    id: String,
    admissible: bool,
    exclusions: Vec<ExclusionWire>,
    best_fraction: f64,
    value_bounds: Option<(f64, f64)>,
    worst_case_bounds: Option<(f64, f64)>,
    worst_case_regret_bounds: Option<(f64, f64)>,
    measure_value: Option<f64>,
    measure_share: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RegionStatusWire {
    Best(String),
    Undecided { candidates: Vec<String> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegionWire {
    lower: Vec<f64>,
    upper: Vec<f64>,
    status: RegionStatusWire,
    touches_relation_boundary: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CertifiedVerdictWire {
    NecessarilyBest(String),
    PreferenceDependent { certified_somewhere: Vec<String> },
    WorstCaseChoice(String),
    MinimaxRegretChoice(String),
    MeasureChoice { action: String, share: f64 },
    Undetermined { candidates: Vec<String>, undecided_fraction: f64 },
    ReportOnly,
    NoAdmissibleAction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CertifiedWire {
    contract_identity: String,
    policy: String,
    parameter_ids: Vec<String>,
    actions: Vec<CertifiedOutcomeWire>,
    regions: Vec<RegionWire>,
    undecided_fraction: f64,
    boxes_examined: usize,
    verdict: CertifiedVerdictWire,
    assumptions: Vec<String>,
    source: ReceiptWire,
    n_draws: usize,
}

fn certified_wire(result: &CertifiedPreferenceResult) -> CertifiedWire {
    CertifiedWire {
        contract_identity: result.contract_identity.clone(),
        policy: preference_policy_name(result.policy).to_owned(),
        parameter_ids: result.parameter_ids.clone(),
        actions: result
            .actions
            .iter()
            .map(|a| CertifiedOutcomeWire {
                id: a.id.clone(),
                admissible: a.admissible,
                exclusions: exclusions_wire(&a.exclusions),
                best_fraction: a.best_fraction,
                value_bounds: a.value_bounds,
                worst_case_bounds: a.worst_case_bounds,
                worst_case_regret_bounds: a.worst_case_regret_bounds,
                measure_value: a.measure_value,
                measure_share: a.measure_share,
            })
            .collect(),
        regions: result
            .regions
            .iter()
            .map(|r| RegionWire {
                lower: r.lower.clone(),
                upper: r.upper.clone(),
                status: match &r.status {
                    RegionStatus::Best(a) => RegionStatusWire::Best(a.clone()),
                    RegionStatus::Undecided { candidates } => {
                        RegionStatusWire::Undecided { candidates: candidates.clone() }
                    }
                },
                touches_relation_boundary: r.touches_relation_boundary,
            })
            .collect(),
        undecided_fraction: result.undecided_fraction,
        boxes_examined: result.boxes_examined,
        verdict: match &result.verdict {
            CertifiedVerdict::NecessarilyBest(a) => {
                CertifiedVerdictWire::NecessarilyBest(a.clone())
            }
            CertifiedVerdict::PreferenceDependent { certified_somewhere } => {
                CertifiedVerdictWire::PreferenceDependent {
                    certified_somewhere: certified_somewhere.clone(),
                }
            }
            CertifiedVerdict::WorstCaseChoice(a) => {
                CertifiedVerdictWire::WorstCaseChoice(a.clone())
            }
            CertifiedVerdict::MinimaxRegretChoice(a) => {
                CertifiedVerdictWire::MinimaxRegretChoice(a.clone())
            }
            CertifiedVerdict::MeasureChoice { action, share } => {
                CertifiedVerdictWire::MeasureChoice { action: action.clone(), share: *share }
            }
            CertifiedVerdict::Undetermined { candidates, undecided_fraction } => {
                CertifiedVerdictWire::Undetermined {
                    candidates: candidates.clone(),
                    undecided_fraction: *undecided_fraction,
                }
            }
            CertifiedVerdict::ReportOnly => CertifiedVerdictWire::ReportOnly,
            CertifiedVerdict::NoAdmissibleAction => CertifiedVerdictWire::NoAdmissibleAction,
        },
        assumptions: result.assumptions.clone(),
        source: receipt_wire(&result.source),
        n_draws: result.n_draws,
    }
}

fn certified_from(
    wire: CertifiedWire,
) -> Result<CertifiedPreferenceResult, PreferenceArtifactError> {
    Ok(CertifiedPreferenceResult {
        contract_identity: wire.contract_identity,
        policy: preference_policy_from(&wire.policy)?,
        parameter_ids: wire.parameter_ids,
        actions: wire
            .actions
            .into_iter()
            .map(|a| CertifiedActionOutcome {
                id: a.id,
                admissible: a.admissible,
                exclusions: exclusions_from(a.exclusions),
                best_fraction: a.best_fraction,
                value_bounds: a.value_bounds,
                worst_case_bounds: a.worst_case_bounds,
                worst_case_regret_bounds: a.worst_case_regret_bounds,
                measure_value: a.measure_value,
                measure_share: a.measure_share,
            })
            .collect(),
        regions: wire
            .regions
            .into_iter()
            .map(|r| CertifiedRegion {
                lower: r.lower,
                upper: r.upper,
                status: match r.status {
                    RegionStatusWire::Best(a) => RegionStatus::Best(a),
                    RegionStatusWire::Undecided { candidates } => {
                        RegionStatus::Undecided { candidates }
                    }
                },
                touches_relation_boundary: r.touches_relation_boundary,
            })
            .collect(),
        undecided_fraction: wire.undecided_fraction,
        boxes_examined: wire.boxes_examined,
        verdict: match wire.verdict {
            CertifiedVerdictWire::NecessarilyBest(a) => CertifiedVerdict::NecessarilyBest(a),
            CertifiedVerdictWire::PreferenceDependent { certified_somewhere } => {
                CertifiedVerdict::PreferenceDependent { certified_somewhere }
            }
            CertifiedVerdictWire::WorstCaseChoice(a) => CertifiedVerdict::WorstCaseChoice(a),
            CertifiedVerdictWire::MinimaxRegretChoice(a) => {
                CertifiedVerdict::MinimaxRegretChoice(a)
            }
            CertifiedVerdictWire::MeasureChoice { action, share } => {
                CertifiedVerdict::MeasureChoice { action, share }
            }
            CertifiedVerdictWire::Undetermined { candidates, undecided_fraction } => {
                CertifiedVerdict::Undetermined { candidates, undecided_fraction }
            }
            CertifiedVerdictWire::ReportOnly => CertifiedVerdict::ReportOnly,
            CertifiedVerdictWire::NoAdmissibleAction => CertifiedVerdict::NoAdmissibleAction,
        },
        assumptions: wire.assumptions,
        source: receipt_from(wire.source),
        n_draws: wire.n_draws,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EvaluationWire {
    Exact(Box<ExactWire>),
    Certified(Box<CertifiedWire>),
}

fn evaluation_wire(evaluation: &PreferenceEvaluation) -> EvaluationWire {
    match evaluation {
        PreferenceEvaluation::Exact(result) => EvaluationWire::Exact(Box::new(exact_wire(result))),
        PreferenceEvaluation::Certified(result) => {
            EvaluationWire::Certified(Box::new(certified_wire(result)))
        }
    }
}

fn evaluation_from(wire: EvaluationWire) -> Result<PreferenceEvaluation, PreferenceArtifactError> {
    Ok(match wire {
        EvaluationWire::Exact(result) => {
            PreferenceEvaluation::Exact(Box::new(exact_from(*result)?))
        }
        EvaluationWire::Certified(result) => {
            PreferenceEvaluation::Certified(Box::new(certified_from(*result)?))
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Preference decision artifact
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionBody {
    version: u16,
    contract_identity: String,
    source_digest: String,
    contract: ContractWire,
    measure: Option<MeasureWire>,
    options: EvaluationOptionsWire,
    evaluation: EvaluationWire,
}

/// A decision across a preference set bound to its contract and to the draws it was computed
/// from.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceDecisionArtifact {
    contract: PreferenceContract,
    contract_identity: String,
    measure: Option<PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    evaluation: PreferenceEvaluation,
    source_digest: String,
}

impl PreferenceDecisionArtifact {
    /// Bind an evaluation to its contract, measure, options and source.
    ///
    /// # Errors
    /// An invalid contract, or an evaluation computed under another contract.
    pub fn new(
        contract: PreferenceContract,
        measure: Option<PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        evaluation: PreferenceEvaluation,
        source: &DistributionArtifact,
    ) -> Result<Self, PreferenceArtifactError> {
        let contract_identity = contract.identity().map_err(E::Contract)?;
        if evaluation.contract_identity() != contract_identity {
            return Err(E::ResultContractMismatch);
        }
        Ok(Self {
            contract,
            contract_identity,
            measure,
            options,
            evaluation,
            source_digest: source_digest(source),
        })
    }

    /// Decide `contract` on `source` by the route rule and bind the result.
    ///
    /// # Errors
    /// A refused evaluation, or an invalid contract.
    pub fn evaluate(
        contract: PreferenceContract,
        source: &DistributionArtifact,
        measure: Option<PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
    ) -> Result<Self, PreferenceArtifactError> {
        let evaluation = decide_preferences(&contract, source, measure.as_ref(), options)
            .map_err(E::Evaluation)?;
        Self::new(contract, measure, options, evaluation, source)
    }

    /// [`Self::evaluate`] under an execution context: the answering route reads the context's
    /// cancellation token at its work boundaries.
    ///
    /// # Errors
    /// As [`Self::evaluate`]; a cancelled run is the answering route's `Cancelled` refusal.
    pub fn evaluate_with_context(
        contract: PreferenceContract,
        source: &DistributionArtifact,
        measure: Option<PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        ctx: &ExecutionContext,
    ) -> Result<Self, PreferenceArtifactError> {
        let evaluation = decide_on_draws(&contract, source, measure.as_ref(), options, Some(ctx))
            .map_err(E::Evaluation)?;
        Self::new(contract, measure, options, evaluation, source)
    }

    /// The preference contract.
    #[must_use]
    pub fn contract(&self) -> &PreferenceContract {
        &self.contract
    }

    /// Identity of the preference contract; the identity a consumer retains.
    #[must_use]
    pub fn contract_identity(&self) -> &str {
        &self.contract_identity
    }

    /// The measure the evaluation read, if any.
    #[must_use]
    pub fn measure(&self) -> Option<&PreferenceMeasure> {
        self.measure.as_ref()
    }

    /// The evaluation options.
    #[must_use]
    pub fn options(&self) -> PreferenceEvaluationOptions {
        self.options
    }

    /// The stored evaluation.
    #[must_use]
    pub fn evaluation(&self) -> &PreferenceEvaluation {
        &self.evaluation
    }

    /// The route that answered.
    #[must_use]
    pub fn route(&self) -> PreferenceRoute {
        self.evaluation.route()
    }

    /// Digest of the draws the evaluation was computed from, as
    /// [`crate::decision_artifact::source_digest`] computes it.
    #[must_use]
    pub fn source_digest(&self) -> &str {
        &self.source_digest
    }

    fn body(&self) -> DecisionBody {
        DecisionBody {
            version: BODY_VERSION,
            contract_identity: self.contract_identity.clone(),
            source_digest: self.source_digest.clone(),
            contract: contract_wire(&self.contract),
            measure: self.measure.as_ref().map(measure_wire),
            options: options_wire(&self.options),
            evaluation: evaluation_wire(&self.evaluation),
        }
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An encoding failure, an oversized payload or a blank artifact id.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, PreferenceArtifactError> {
        encode(DECISION_KIND, DECISION_SECTION, artifact_id, "preference_decision", &self.body())
    }

    /// Load under the consumer's retained contract identity and source digest.
    ///
    /// # Errors
    /// A corrupt container, another kind or an unsupported version; a contract or source other
    /// than the retained ones; a stored contract whose recomputed identity differs from the
    /// stored one (a resealed edit); a result computed under another contract; a malformed
    /// field.
    pub fn from_bytes(
        bytes: &[u8],
        expected_contract_identity: &str,
        expected_source_digest: &str,
    ) -> Result<Self, PreferenceArtifactError> {
        let body: DecisionBody =
            decode_section(bytes, DECISION_KIND, DECISION_SECTION).map_err(E::Container)?;
        if body.contract_identity != expected_contract_identity {
            return Err(E::ContractExpected);
        }
        if body.source_digest != expected_source_digest {
            return Err(E::SourceExpected);
        }
        let contract = contract_from(body.contract)?;
        let contract_identity = contract.identity().map_err(E::Contract)?;
        if contract_identity != body.contract_identity {
            return Err(E::StoredIdentityMismatch);
        }
        let evaluation = evaluation_from(body.evaluation)?;
        if evaluation.contract_identity() != contract_identity {
            return Err(E::ResultContractMismatch);
        }
        Ok(Self {
            contract,
            contract_identity,
            measure: body.measure.map(measure_from),
            options: options_from(body.options),
            evaluation,
            source_digest: body.source_digest,
        })
    }

    /// Re-evaluate on `source` with the stored contract, measure and options and require the
    /// stored evaluation to be reproduced: the same route, verdicts, labels, region statuses,
    /// identities and counts exactly, and every real within [`REPLAY_RELATIVE_TOLERANCE`].
    ///
    /// # Errors
    /// Draws other than the ones the result was computed from; a refused evaluation; any
    /// differing field.
    pub fn replay(&self, source: &DistributionArtifact) -> Result<(), PreferenceArtifactError> {
        self.replay_in(source, None)
    }

    /// [`Self::replay`] under an execution context: the re-evaluation reads the context's
    /// cancellation token as the answering route does.
    ///
    /// # Errors
    /// As [`Self::replay`]; [`PreferenceArtifactError::Cancelled`] once the token is set.
    pub fn replay_with_context(
        &self,
        source: &DistributionArtifact,
        ctx: &ExecutionContext,
    ) -> Result<(), PreferenceArtifactError> {
        self.replay_in(source, Some(ctx))
    }

    fn replay_in(
        &self,
        source: &DistributionArtifact,
        ctx: Option<&ExecutionContext>,
    ) -> Result<(), PreferenceArtifactError> {
        if source_digest(source) != self.source_digest {
            return Err(E::ReplaySourceDiffers);
        }
        observe(ctx)?;
        let measure = self.measure.as_ref();
        let fresh = decide_on_draws(&self.contract, source, measure, self.options, ctx)
            .map_err(reevaluation_error)?;
        match (&self.evaluation, &fresh) {
            (PreferenceEvaluation::Exact(stored), PreferenceEvaluation::Exact(fresh)) => {
                compare_exact(stored, fresh)
            }
            (PreferenceEvaluation::Certified(stored), PreferenceEvaluation::Certified(fresh)) => {
                compare_certified(stored, fresh)
            }
            _ => Err(E::ReplayMismatch { field: "route" }),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Replay comparison
// ---------------------------------------------------------------------------------------------

fn close(a: f64, b: f64) -> bool {
    a.total_cmp(&b).is_eq()
        || (a - b).abs() <= REPLAY_RELATIVE_TOLERANCE * a.abs().max(b.abs()).max(1.0)
}

fn close_all(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
}

fn close_pair(a: (f64, f64), b: (f64, f64)) -> bool {
    close(a.0, b.0) && close(a.1, b.1)
}

fn close_opt(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => close(x, y),
        _ => false,
    }
}

fn close_opt_pair(a: Option<(f64, f64)>, b: Option<(f64, f64)>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => close_pair(x, y),
        _ => false,
    }
}

fn close_exclusions(a: &[ConstraintExclusion], b: &[ConstraintExclusion]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.constraint_id == y.constraint_id
                && close(x.probability, y.probability)
                && close(x.required, y.required)
        })
}

fn ensure(holds: bool, field: &'static str) -> Result<(), PreferenceArtifactError> {
    if holds { Ok(()) } else { Err(E::ReplayMismatch { field }) }
}

fn compare_exact(
    stored: &PreferenceDecisionResult,
    fresh: &PreferenceDecisionResult,
) -> Result<(), PreferenceArtifactError> {
    ensure(stored.contract_identity == fresh.contract_identity, "contract_identity")?;
    ensure(stored.policy == fresh.policy, "policy")?;
    ensure(stored.parameter_ids == fresh.parameter_ids, "parameter_ids")?;
    ensure(
        stored.parameter_ranges.len() == fresh.parameter_ranges.len()
            && stored
                .parameter_ranges
                .iter()
                .zip(&fresh.parameter_ranges)
                .all(|(s, f)| close_pair(*s, *f)),
        "parameter_ranges",
    )?;
    ensure(stored.verdict == fresh.verdict, "verdict")?;
    ensure(stored.actions.len() == fresh.actions.len(), "actions")?;
    for (s, f) in stored.actions.iter().zip(&fresh.actions) {
        ensure(s.id == f.id && s.admissible == f.admissible, "actions")?;
        ensure(s.necessarily_best == f.necessarily_best, "necessarily_best")?;
        ensure(s.uniquely_best == f.uniquely_best, "uniquely_best")?;
        ensure(s.dominated_by == f.dominated_by, "dominated_by")?;
        ensure(s.possibly_best == f.possibly_best, "possibly_best")?;
        ensure(close_exclusions(&s.exclusions, &f.exclusions), "exclusions")?;
        let coefficients = match (&s.coefficients, &f.coefficients) {
            (None, None) => true,
            (Some(x), Some(y)) => close_all(x, y),
            _ => false,
        };
        ensure(coefficients, "coefficients")?;
        ensure(close_opt_pair(s.value_range, f.value_range), "value_range")?;
        ensure(close_opt(s.worst_case_regret, f.worst_case_regret), "worst_case_regret")?;
        ensure(close_opt(s.measure_share, f.measure_share), "measure_share")?;
        ensure(close_opt(s.measure_value, f.measure_value), "measure_value")?;
    }
    ensure(stored.switches.len() == fresh.switches.len(), "switches")?;
    for (s, f) in stored.switches.iter().zip(&fresh.switches) {
        ensure(s.first == f.first && s.second == f.second, "switches")?;
        ensure(close_all(&s.margin_coefficients, &f.margin_coefficients), "margin_coefficients")?;
        ensure(close_pair(s.margin_range, f.margin_range), "margin_range")?;
        ensure(close_all(&s.favours_first_at, &f.favours_first_at), "favours_first_at")?;
        ensure(close_all(&s.favours_second_at, &f.favours_second_at), "favours_second_at")?;
        let threshold = match (&s.threshold, &f.threshold) {
            (None, None) => true,
            (Some(x), Some(y)) => {
                x.parameter_id == y.parameter_id
                    && x.first_leads_above == y.first_leads_above
                    && close(x.value, y.value)
            }
            _ => false,
        };
        ensure(threshold, "threshold")?;
    }
    ensure(
        close_opt(stored.draw_share_best_everywhere, fresh.draw_share_best_everywhere),
        "draw_share_best_everywhere",
    )?;
    ensure(
        close_opt(stored.unresolved_measure_share, fresh.unresolved_measure_share),
        "unresolved_measure_share",
    )?;
    ensure(stored.assumptions == fresh.assumptions, "assumptions")?;
    ensure(stored.source == fresh.source, "source")?;
    ensure(stored.n_draws == fresh.n_draws, "n_draws")
}

fn compare_certified(
    stored: &CertifiedPreferenceResult,
    fresh: &CertifiedPreferenceResult,
) -> Result<(), PreferenceArtifactError> {
    ensure(stored.contract_identity == fresh.contract_identity, "contract_identity")?;
    ensure(stored.policy == fresh.policy, "policy")?;
    ensure(stored.parameter_ids == fresh.parameter_ids, "parameter_ids")?;
    ensure(stored.verdict == fresh.verdict, "verdict")?;
    ensure(stored.actions.len() == fresh.actions.len(), "actions")?;
    for (s, f) in stored.actions.iter().zip(&fresh.actions) {
        ensure(s.id == f.id && s.admissible == f.admissible, "actions")?;
        ensure(close_exclusions(&s.exclusions, &f.exclusions), "exclusions")?;
        ensure(close(s.best_fraction, f.best_fraction), "best_fraction")?;
        ensure(close_opt_pair(s.value_bounds, f.value_bounds), "value_bounds")?;
        ensure(close_opt_pair(s.worst_case_bounds, f.worst_case_bounds), "worst_case_bounds")?;
        ensure(
            close_opt_pair(s.worst_case_regret_bounds, f.worst_case_regret_bounds),
            "worst_case_regret_bounds",
        )?;
        ensure(close_opt(s.measure_value, f.measure_value), "measure_value")?;
        ensure(close_opt(s.measure_share, f.measure_share), "measure_share")?;
    }
    ensure(stored.regions.len() == fresh.regions.len(), "regions")?;
    for (s, f) in stored.regions.iter().zip(&fresh.regions) {
        ensure(s.status == f.status, "region_status")?;
        ensure(s.touches_relation_boundary == f.touches_relation_boundary, "region_boundary")?;
        ensure(close_all(&s.lower, &f.lower) && close_all(&s.upper, &f.upper), "region_box")?;
    }
    ensure(close(stored.undecided_fraction, fresh.undecided_fraction), "undecided_fraction")?;
    ensure(stored.boxes_examined == fresh.boxes_examined, "boxes_examined")?;
    ensure(stored.assumptions == fresh.assumptions, "assumptions")?;
    ensure(stored.source == fresh.source, "source")?;
    ensure(stored.n_draws == fresh.n_draws, "n_draws")
}

fn compare_set(
    stored: &PreferenceSet,
    fresh: &PreferenceSet,
) -> Result<(), PreferenceArtifactError> {
    ensure(stored.parameters == fresh.parameters, "parameters")?;
    ensure(stored.relations.len() == fresh.relations.len(), "relations")?;
    for (s, f) in stored.relations.iter().zip(&fresh.relations) {
        let same_terms = s.terms.len() == f.terms.len()
            && s.terms.iter().zip(&f.terms).all(|(x, y)| x.0 == y.0 && close(x.1, y.1));
        ensure(s.id == f.id && s.sense == f.sense && s.origin == f.origin, "relations")?;
        ensure(same_terms && close(s.bound, f.bound), "relation_terms")?;
    }
    Ok(())
}

fn compare_learned(
    stored: &LearnedPreferenceSet,
    fresh: &LearnedPreferenceSet,
) -> Result<(), PreferenceArtifactError> {
    if stored.records_digest != fresh.records_digest {
        return Err(E::ReplayRecordsDiffer);
    }
    ensure(stored.model_identity == fresh.model_identity, "model_identity")?;
    ensure(stored.decider == fresh.decider, "decider")?;
    ensure(stored.tolerance.total_cmp(&fresh.tolerance).is_eq(), "tolerance")?;
    ensure(stored.n_records == fresh.n_records, "n_records")?;
    compare_set(&stored.set, &fresh.set)?;
    ensure(stored.ranges.len() == fresh.ranges.len(), "ranges")?;
    for (s, f) in stored.ranges.iter().zip(&fresh.ranges) {
        ensure(s.parameter_id == f.parameter_id, "ranges")?;
        ensure(
            s.varied_in_records == f.varied_in_records && s.narrowed == f.narrowed,
            "range_labels",
        )?;
        ensure(close_pair(s.declared, f.declared) && close_pair(s.learned, f.learned), "ranges")?;
    }
    ensure(stored.binding == fresh.binding, "binding")?;
    ensure(stored.assumptions == fresh.assumptions, "assumptions")
}

fn compare_estimate(
    stored: &ChoiceModelEstimate,
    fresh: &ChoiceModelEstimate,
) -> Result<(), PreferenceArtifactError> {
    if stored.records_digest != fresh.records_digest {
        return Err(E::ReplayRecordsDiffer);
    }
    ensure(stored.identity == fresh.identity, "estimate_identity")?;
    ensure(stored.model_identity == fresh.model_identity, "model_identity")?;
    ensure(stored.decider == fresh.decider, "decider")?;
    ensure(close(stored.consistency, fresh.consistency), "consistency")?;
    ensure(
        close(stored.consistency_standard_error, fresh.consistency_standard_error),
        "consistency_standard_error",
    )?;
    ensure(stored.parameters.len() == fresh.parameters.len(), "parameters")?;
    for (s, f) in stored.parameters.iter().zip(&fresh.parameters) {
        ensure(s.parameter_id == f.parameter_id, "parameters")?;
        ensure(s.inside_declared == f.inside_declared, "inside_declared")?;
        let numbers = close(s.estimate, f.estimate)
            && close(s.standard_error, f.standard_error)
            && close_pair(s.interval, f.interval)
            && close_pair(s.declared, f.declared);
        ensure(numbers, "parameter_estimates")?;
    }
    ensure(stored.not_learnable == fresh.not_learnable, "not_learnable")?;
    ensure(close(stored.log_likelihood, fresh.log_likelihood), "log_likelihood")?;
    ensure(close(stored.null_log_likelihood, fresh.null_log_likelihood), "null_log_likelihood")?;
    ensure(stored.n_records == fresh.n_records && stored.n_deciders == fresh.n_deciders, "counts")?;
    ensure(stored.covariance_kind == fresh.covariance_kind, "covariance_kind")?;
    ensure(
        close_all(&stored.coefficient_covariance, &fresh.coefficient_covariance),
        "coefficient_covariance",
    )?;
    ensure(stored.level.total_cmp(&fresh.level).is_eq(), "level")?;
    ensure(stored.interval_calibration == fresh.interval_calibration, "interval_calibration")?;
    let (kept, redrawn) = (&stored.measure, &fresh.measure);
    ensure(
        kept.points.len() == redrawn.points.len()
            && kept.points.iter().zip(&redrawn.points).all(|(x, y)| close_all(x, y)),
        "measure_points",
    )?;
    ensure(
        kept.weights == redrawn.weights
            && kept.labels == redrawn.labels
            && kept.origin == redrawn.origin
            && kept.unresolved == redrawn.unresolved,
        "measure",
    )?;
    ensure(
        close(stored.share_outside_declared, fresh.share_outside_declared),
        "share_outside_declared",
    )?;
    ensure(stored.assumptions == fresh.assumptions, "assumptions")
}

impl PreferenceArtifactError {
    /// Structured refusal for a failure to write, read, check or replay a preference artifact.
    ///
    /// An accepted-preferences, contract or evaluation refusal keeps its own refusal; an
    /// encoding or container failure carries the container's message as the supplied text.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |stage, remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), stage, self.detail())
                .remedy(remedy)
        };
        let unsatisfied = |remedy| {
            refusal(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                CONSUME,
                self.detail(),
            )
            .remedy(remedy)
        };
        match self {
            Self::Encode(error) => {
                refusal(antecedent_core::reason_code!("invalid_argument"), EXPORT, self.detail())
                    .supplied(error.to_string())
                    .build()
            }
            Self::Container(error) => invalid(CONSUME, "supply bytes written by this library")
                .supplied(error.to_string())
                .build(),
            Self::Malformed { what } => invalid(CONSUME, "supply bytes written by this library")
                .offending(what.clone())
                .build(),
            Self::Accepted(error) => at_stage(error.to_refusal(), CONSUME),
            Self::Contract(error) => at_stage(error.to_refusal(), CONSUME),
            Self::Evaluation(error) => at_stage(error.to_refusal(), CONSUME),
            Self::StoredIdentityMismatch => {
                invalid(CONSUME, "re-export from the producer instead of editing the bytes").build()
            }
            Self::IdentityExpected => {
                unsatisfied("supply the artifact whose identity was retained").build()
            }
            Self::ContractExpected => {
                unsatisfied("supply the result of the retained preference contract").build()
            }
            Self::SourceExpected => {
                unsatisfied("supply the result computed from the retained draws").build()
            }
            Self::ResultContractMismatch => {
                unsatisfied("re-export the result from the producer").build()
            }
            Self::ReplaySourceDiffers => {
                unsatisfied("replay with the draws the result was computed from").build()
            }
            Self::ReplayRecordsDiffer => {
                unsatisfied("replay with the records the preferences were learned from").build()
            }
            Self::ReplayMismatch { field } => {
                unsatisfied("re-export from the producer and replay again")
                    .offending(*field)
                    .build()
            }
            Self::Cancelled => {
                refusal(antecedent_core::reason_code!("cancelled_no_claim"), CONSUME, self.detail())
                    .build()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_names_are_malformed() {
        assert_eq!(
            kind_from("treatment").unwrap_err().detail(),
            antecedent_core::reason_details::preference_artifact::WIRE_MALFORMED
        );
        assert!(structural_from("bayes").is_err());
        assert!(preference_policy_from("weighted_sum").is_err());
        for policy in [
            PreferencePolicy::RequireInvariantBestAction,
            PreferencePolicy::Maximin,
            PreferencePolicy::MinimaxRegret,
            PreferencePolicy::Weighted,
            PreferencePolicy::ReportOnly,
        ] {
            assert_eq!(preference_policy_from(preference_policy_name(policy)).unwrap(), policy);
        }
    }

    #[test]
    fn replay_closeness_is_relative_to_the_larger_magnitude() {
        assert!(close(1e12, 1e12 + 1.0));
        assert!(!close(1.0, 1.0 + 1e-6));
        assert!(close(0.0, 1e-10));
        assert!(close_opt(None, None) && !close_opt(Some(1.0), None));
        assert!(!close_all(&[1.0], &[1.0, 2.0]));
    }

    #[test]
    fn every_expression_variant_survives_the_wire() {
        use PreferenceExpr as P;
        let x = P::Attribute(0);
        let t = P::Parameter(1);
        let expr = P::sum(
            P::difference(P::product(t.clone(), x.clone()), P::negate(P::Const(-0.0))),
            P::maximum(
                P::minimum(P::exp(x.clone()), P::ln(P::Const(0.1))),
                P::Const(f64::MIN_POSITIVE / 4.0),
            ),
        );
        let bytes = to_cbor(&ExprWire::from(&expr)).unwrap();
        let back: ExprWire = antecedent_io_core::convert::from_cbor(&bytes).unwrap();
        assert_eq!(PreferenceExpr::from(back), expr);
    }

    #[test]
    fn a_resealed_accepted_body_is_refused() {
        use crate::preference_model::{Attribute as A, PreferenceParameter as Q};
        let model = ValueModel {
            attributes: vec![A { id: "benefit".into(), units: "events".into() }],
            preferences: PreferenceSet {
                parameters: vec![Q {
                    id: "w".into(),
                    units: "utility/event".into(),
                    lower: 0.0,
                    upper: 1.0,
                    assumption_id: "w-judgment".into(),
                }],
                relations: vec![],
            },
            value: PreferenceExpr::product(
                PreferenceExpr::Parameter(0),
                PreferenceExpr::Attribute(0),
            ),
            utility_units: "utility".into(),
        };
        let accepted = AcceptedPreferences::declared(model).unwrap();
        let mut body = accepted_body(&accepted);
        body.accepted_version = 7;
        let bytes =
            encode(ACCEPTED_KIND, ACCEPTED_SECTION, "accepted", "accepted_preferences", &body)
                .unwrap();
        let error =
            AcceptedPreferencesArtifact::from_bytes(&bytes, accepted.identity()).unwrap_err();
        assert_eq!(error, E::StoredIdentityMismatch);
        assert_eq!(
            error.detail(),
            antecedent_core::reason_details::preference_artifact::IDENTITY_RESEALED
        );
        let io: IoError = error.into();
        assert!(matches!(io, IoError::Refused { .. }));
    }
}
