//! Durable records of decisions across a preference set.
//!
//! [`PreferenceDecisionArtifact`] stores a preference contract, the measure it was read with (if
//! any), the evaluation options, the route that answered, the result and the digest of the joint
//! draws it was computed from (the same source digest the decision result artifact retains).
//!
//! It follows the decision artifacts: a checksummed container with a versioned kind
//! (`preference_decision_v1`), a CBOR body whose reals are written as IEEE-754 values and read
//! back bit for bit, loading only under identities the consumer retained separately from the
//! bytes (a resealed edit is refused because every identity is recomputed from the decoded
//! content), and replay by recomputation. Replay compares labels, verdicts, identities, counts and
//! statuses exactly, and every real within [`REPLAY_RELATIVE_TOLERANCE`] of the stored value
//! relative to the larger of one and the two magnitudes, since linear-program outputs need not
//! agree to the last bit across platforms.
//!
//! The value-model, measure and option wires here are shared with the record of accepted
//! preferences, which builds on this module; this module reads no code that learns preferences
//! from recorded choices.

// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::ExecutionContext;
use antecedent_io_core::convert::to_cbor;
use antecedent_io_core::distribution_artifact::DistributionArtifact;
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use antecedent_stats::linear_program::LpOptions;
use serde::{Deserialize, Serialize};

use crate::decision_artifact::{decode_section, encode_section, source_digest};
use crate::decision_contract::{ActionKind, HardConstraint, StructuralPolicy};
use crate::decision_eval::{ConstraintExclusion, SourceReceipt};
use crate::decision_wire::{
    ConstraintWire, CriterionWire, ExclusionWire, kind_name, kind_named, structural_name,
    structural_named,
};
use crate::preference_artifact_error::PreferenceArtifactError;
use crate::preference_certified::{
    CertifiedActionOutcome, CertifiedOptions, CertifiedPreferenceResult, CertifiedRegion,
    CertifiedVerdict, PreferenceCertifiedError, RegionStatus,
};
use crate::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferencePolicy,
};
use crate::preference_decision::{
    MeasureOrigin, PreferenceActionOutcome, PreferenceDecisionError, PreferenceDecisionOptions,
    PreferenceDecisionResult, PreferenceMeasure, PreferenceSwitch, PreferenceVerdict,
    SwitchThreshold,
};
use crate::preference_model::{
    Attribute, BeliefRows, LinearRelation, PreferenceExpr, PreferenceParameter, PreferenceSet,
    RelationOrigin, RelationSense, ValueModel, ValueRelation,
};
use crate::preference_route_choice::{
    PreferenceEvaluation, PreferenceEvaluationError, PreferenceEvaluationOptions, PreferenceRoute,
    decide_on_draws, decide_preferences,
};

/// Relative tolerance of every real a replay compares: `|stored - fresh|` may be at most this
/// times the larger of one, `|stored|` and `|fresh|`.
pub const REPLAY_RELATIVE_TOLERANCE: f64 = 1e-9;

const DECISION_KIND: &str = "preference_decision_v1";
const DECISION_SECTION: &str = "preference_decision_body";
/// Version word of every body this reader decodes; the shared decoder refuses any other.
pub(crate) const BODY_VERSION: u16 = 1;

use PreferenceArtifactError as E;

/// `Err(Cancelled)` when a context is given and its token is set.
pub(crate) fn observe(ctx: Option<&ExecutionContext>) -> Result<(), PreferenceArtifactError> {
    if ctx.is_some_and(|ctx| ctx.cancellation.is_cancelled()) {
        return Err(E::Cancelled);
    }
    Ok(())
}

/// A refused re-evaluation, with the route's cancellation reported as the replay's.
fn reevaluation_error(error: PreferenceEvaluationError) -> PreferenceArtifactError {
    match error {
        PreferenceEvaluationError::Exact(PreferenceDecisionError::Cancelled)
        | PreferenceEvaluationError::Certified(PreferenceCertifiedError::Cancelled) => E::Cancelled,
        other => E::Evaluation(Box::new(other)),
    }
}

pub(crate) fn malformed(what: impl Into<String>) -> PreferenceArtifactError {
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
struct BeliefRowsWire {
    rows: Vec<f64>,
    weights: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ValueRelationWire {
    id: String,
    origin: OriginWire,
    chosen: BeliefRowsWire,
    rejected: BeliefRowsWire,
    tolerance: f64,
}

/// A set without value relations writes no `value_relations` field, so its wire (and every
/// identity read from it) is what it was before value relations existed.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SetWire {
    parameters: Vec<ParameterWire>,
    relations: Vec<RelationWire>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    value_relations: Vec<ValueRelationWire>,
}

fn origin_wire(origin: &RelationOrigin) -> OriginWire {
    match origin {
        RelationOrigin::Declared => OriginWire::Declared,
        RelationOrigin::ShapeRule => OriginWire::ShapeRule,
        RelationOrigin::Choice { record_id, rejected_option } => OriginWire::Choice {
            record_id: record_id.clone(),
            rejected_option: rejected_option.clone(),
        },
    }
}

fn origin_from(wire: OriginWire) -> RelationOrigin {
    match wire {
        OriginWire::Declared => RelationOrigin::Declared,
        OriginWire::ShapeRule => RelationOrigin::ShapeRule,
        OriginWire::Choice { record_id, rejected_option } => {
            RelationOrigin::Choice { record_id, rejected_option }
        }
    }
}

fn rows_wire(rows: &BeliefRows) -> BeliefRowsWire {
    BeliefRowsWire { rows: rows.rows.clone(), weights: rows.weights.clone() }
}

fn rows_from(wire: BeliefRowsWire) -> BeliefRows {
    BeliefRows { rows: wire.rows, weights: wire.weights }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelWire {
    attributes: Vec<AttributeWire>,
    preferences: SetWire,
    value: ExprWire,
    utility_units: String,
}

pub(crate) fn set_wire(set: &PreferenceSet) -> SetWire {
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
                origin: origin_wire(&r.origin),
            })
            .collect(),
        value_relations: set
            .value_relations
            .iter()
            .map(|r| ValueRelationWire {
                id: r.id.clone(),
                origin: origin_wire(&r.origin),
                chosen: rows_wire(&r.chosen),
                rejected: rows_wire(&r.rejected),
                tolerance: r.tolerance,
            })
            .collect(),
    }
}

pub(crate) fn set_from(wire: SetWire) -> PreferenceSet {
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
                origin: origin_from(r.origin),
            })
            .collect(),
        value_relations: wire
            .value_relations
            .into_iter()
            .map(|r| ValueRelation {
                id: r.id,
                origin: origin_from(r.origin),
                chosen: rows_from(r.chosen),
                rejected: rows_from(r.rejected),
                tolerance: r.tolerance,
            })
            .collect(),
    }
}

pub(crate) fn model_wire(model: &ValueModel) -> ModelWire {
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

pub(crate) fn model_from(wire: ModelWire) -> ValueModel {
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
pub(crate) struct MeasureWire {
    points: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
    labels: Option<Vec<String>>,
    origin: MeasureOriginWire,
    unresolved: Vec<usize>,
}

pub(crate) fn measure_wire(measure: &PreferenceMeasure) -> MeasureWire {
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

pub(crate) fn measure_from(wire: MeasureWire) -> PreferenceMeasure {
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LpWire {
    feasibility_tolerance: f64,
    optimality_tolerance: f64,
    max_pivots: usize,
}

pub(crate) fn lp_wire(lp: LpOptions) -> LpWire {
    LpWire {
        feasibility_tolerance: lp.feasibility_tolerance,
        optimality_tolerance: lp.optimality_tolerance,
        max_pivots: lp.max_pivots,
    }
}

pub(crate) fn lp_from(wire: LpWire) -> LpOptions {
    LpOptions {
        feasibility_tolerance: wire.feasibility_tolerance,
        optimality_tolerance: wire.optimality_tolerance,
        max_pivots: wire.max_pivots,
    }
}

pub(crate) fn encode(
    kind: &str,
    section: &str,
    artifact_id: &str,
    note: &str,
    body: &impl Serialize,
) -> Result<Vec<u8>, PreferenceArtifactError> {
    let bytes = to_cbor(body).map_err(E::Encode)?;
    encode_section(kind, section, artifact_id, note, &bytes).map_err(E::Encode)
}

// ---------------------------------------------------------------------------------------------
// Contract wire
// ---------------------------------------------------------------------------------------------

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

fn kind_from(name: &str) -> Result<ActionKind, PreferenceArtifactError> {
    kind_named(name).ok_or_else(|| malformed(format!("action kind `{name}`")))
}

fn structural_from(name: &str) -> Result<StructuralPolicy, PreferenceArtifactError> {
    structural_named(name).ok_or_else(|| malformed(format!("structural policy `{name}`")))
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
        constraints: contract.constraints.iter().map(ConstraintWire::from).collect(),
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
        constraints: wire.constraints.into_iter().map(HardConstraint::from).collect(),
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
#[allow(clippy::struct_field_names, reason = "wire names mirror the source receipt fields")]
struct ReceiptWire {
    provider_id: String,
    snapshot_id: String,
    rng_id: String,
    causal_contract_id: String,
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
    /// The tie tolerance the verdict applied, stored only when it differs from the declared
    /// one (a numerical bound widened it), so artifacts without a widening keep their bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    effective_tie_tolerance: Option<f64>,
}

/// `Some(effective)` when it differs from the declared tie tolerance.
fn widened(effective: f64, declared: f64) -> Option<f64> {
    (effective.to_bits() != declared.to_bits()).then_some(effective)
}

fn exact_wire(result: &PreferenceDecisionResult, declared_tie: f64) -> ExactWire {
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
                exclusions: a.exclusions.iter().map(ExclusionWire::from).collect(),
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
        effective_tie_tolerance: widened(result.effective_tie_tolerance, declared_tie),
    }
}

fn exact_from(
    wire: ExactWire,
    declared_tie: f64,
) -> Result<PreferenceDecisionResult, PreferenceArtifactError> {
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
                exclusions: a.exclusions.into_iter().map(ConstraintExclusion::from).collect(),
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
        effective_tie_tolerance: wire.effective_tie_tolerance.unwrap_or(declared_tie),
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
    /// As [`ExactWire::effective_tie_tolerance`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    effective_tie_tolerance: Option<f64>,
}

fn certified_wire(result: &CertifiedPreferenceResult, declared_tie: f64) -> CertifiedWire {
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
                exclusions: a.exclusions.iter().map(ExclusionWire::from).collect(),
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
        effective_tie_tolerance: widened(result.effective_tie_tolerance, declared_tie),
    }
}

fn certified_from(
    wire: CertifiedWire,
    declared_tie: f64,
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
                exclusions: a.exclusions.into_iter().map(ConstraintExclusion::from).collect(),
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
        effective_tie_tolerance: wire.effective_tie_tolerance.unwrap_or(declared_tie),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EvaluationWire {
    Exact(Box<ExactWire>),
    Certified(Box<CertifiedWire>),
}

fn evaluation_wire(
    evaluation: &PreferenceEvaluation,
    options: &PreferenceEvaluationOptions,
) -> EvaluationWire {
    match evaluation {
        PreferenceEvaluation::Exact(result) => {
            EvaluationWire::Exact(Box::new(exact_wire(result, options.exact.tie_tolerance)))
        }
        PreferenceEvaluation::Certified(result) => EvaluationWire::Certified(Box::new(
            certified_wire(result, options.certified.tie_tolerance),
        )),
    }
}

/// The stored evaluation; an effective tie tolerance not stored is the declared one.
fn evaluation_from(
    wire: EvaluationWire,
    options: &PreferenceEvaluationOptions,
) -> Result<PreferenceEvaluation, PreferenceArtifactError> {
    Ok(match wire {
        EvaluationWire::Exact(result) => {
            PreferenceEvaluation::Exact(Box::new(exact_from(*result, options.exact.tie_tolerance)?))
        }
        EvaluationWire::Certified(result) => PreferenceEvaluation::Certified(Box::new(
            certified_from(*result, options.certified.tie_tolerance)?,
        )),
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
            .map_err(|error| E::Evaluation(Box::new(error)))?;
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
            .map_err(|error| E::Evaluation(Box::new(error)))?;
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
            evaluation: evaluation_wire(&self.evaluation, &self.options),
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
        let options = options_from(body.options);
        let evaluation = evaluation_from(body.evaluation, &options)?;
        if evaluation.contract_identity() != contract_identity {
            return Err(E::ResultContractMismatch);
        }
        Ok(Self {
            contract,
            contract_identity,
            measure: body.measure.map(measure_from),
            options,
            evaluation,
            source_digest: body.source_digest,
        })
    }

    /// [`Self::from_bytes`], admitting only the artifact the consumer retained: the bytes bind
    /// to `retained` ([`antecedent_io_core::bind_retained_artifact`]) before they are read.
    ///
    /// # Errors
    /// As [`Self::from_bytes`], and [`PreferenceArtifactError::Retained`]
    /// (`external_binding_mismatch`) when the bytes are not the retained artifact.
    pub fn from_bytes_expecting(
        bytes: &[u8],
        expected_contract_identity: &str,
        expected_source_digest: &str,
        retained: &antecedent_io_core::RetainedArtifactIdentity,
    ) -> Result<Self, PreferenceArtifactError> {
        antecedent_io_core::bind_retained_artifact(bytes, retained).map_err(E::Retained)?;
        Self::from_bytes(bytes, expected_contract_identity, expected_source_digest)
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

pub(crate) fn close(a: f64, b: f64) -> bool {
    a.total_cmp(&b).is_eq()
        || (a - b).abs() <= REPLAY_RELATIVE_TOLERANCE * a.abs().max(b.abs()).max(1.0)
}

pub(crate) fn close_all(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
}

pub(crate) fn close_pair(a: (f64, f64), b: (f64, f64)) -> bool {
    close(a.0, b.0) && close(a.1, b.1)
}

pub(crate) fn close_opt(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => close(x, y),
        _ => false,
    }
}

pub(crate) fn close_opt_pair(a: Option<(f64, f64)>, b: Option<(f64, f64)>) -> bool {
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

pub(crate) fn ensure(holds: bool, field: &'static str) -> Result<(), PreferenceArtifactError> {
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
    fn an_unwidened_tie_tolerance_leaves_the_result_wire_unchanged() {
        let key: &[u8] = b"effective_tie_tolerance";
        let holds = |bytes: &[u8]| bytes.windows(key.len()).any(|window| window == key);
        let receipt = || ReceiptWire {
            provider_id: "p".into(),
            snapshot_id: "s".into(),
            rng_id: "r".into(),
            causal_contract_id: "c".into(),
        };
        let wire = |effective: Option<f64>| CertifiedWire {
            contract_identity: "contract".into(),
            policy: "report_only".into(),
            parameter_ids: vec!["w".into()],
            actions: Vec::new(),
            regions: Vec::new(),
            undecided_fraction: 0.0,
            boxes_examined: 1,
            verdict: CertifiedVerdictWire::ReportOnly,
            assumptions: Vec::new(),
            source: receipt(),
            n_draws: 2,
            effective_tie_tolerance: effective,
        };
        // Equal to the declared tolerance: not stored, so the encoding is the one a record
        // written without the field has, and that encoding reads back as the declared value.
        assert_eq!(widened(1e-9, 1e-9), None);
        let plain = to_cbor(&wire(widened(1e-9, 1e-9))).unwrap();
        assert!(!holds(&plain));
        let back: CertifiedWire = antecedent_io_core::convert::from_cbor(&plain).unwrap();
        let result = certified_from(back, 1e-9).unwrap();
        assert_eq!(result.effective_tie_tolerance.to_bits(), 1e-9_f64.to_bits());
        // Widened: stored and read back exactly.
        let stored = to_cbor(&wire(widened(6.4e-8, 1e-9))).unwrap();
        assert!(holds(&stored));
        let back: CertifiedWire = antecedent_io_core::convert::from_cbor(&stored).unwrap();
        let result = certified_from(back, 1e-9).unwrap();
        assert_eq!(result.effective_tie_tolerance.to_bits(), 6.4e-8_f64.to_bits());
    }
}

#[cfg(test)]
mod retained_tests {
    use super::PreferenceDecisionArtifact;
    use crate::retained_test_support::{other_artifact, refused_as_not_retained, same_outcome};
    use antecedent_io_core::RetainedArtifactIdentity;

    #[test]
    fn preference_decision_artifact_expecting_binds_the_retained_bytes() {
        let bytes = b"not an artifact".to_vec();
        let retained = RetainedArtifactIdentity::of_bytes(&bytes);
        same_outcome(
            &PreferenceDecisionArtifact::from_bytes(&bytes, "contract", "source"),
            &PreferenceDecisionArtifact::from_bytes_expecting(
                &bytes, "contract", "source", &retained,
            ),
        );
        refused_as_not_retained(&PreferenceDecisionArtifact::from_bytes_expecting(
            &bytes,
            "contract",
            "source",
            &other_artifact(),
        ));
    }
}
