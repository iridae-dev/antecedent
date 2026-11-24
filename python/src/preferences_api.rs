//! Python bridge for decisions across a range of preferences, and for learning those
//! preferences from recorded choices.
//!
//! Python declares the value model, the actions, an optional measure over the preference set
//! and the recorded choices as JSON. The validation, the route, every verdict, the learning,
//! the accepted-preferences lifecycle, the artifacts and every refusal are Rust's: a refusal
//! comes back as the structured JSON of the underlying error, with its registered code and
//! detail. A malformed wire (a bridge fault, not a scientific refusal) raises a value error.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod learning;

use antecedent::analysis::inverse_query::{AtomEvidence, DistributionArtifact, MeanSource};
use antecedent::analysis::native_claims::NativeDecisionSource;
use antecedent::analysis::preference_decision::{
    ActionInput, ActionKind, Attribute, BeliefRows, CertifiedPreferenceResult, CertifiedVerdict,
    DecisionCriterion, HardConstraint, LinearRelation, MeasureOrigin, PreferenceAction,
    PreferenceContract, PreferenceDecisionArtifact, PreferenceDecisionResult, PreferenceEvaluation,
    PreferenceEvaluationOptions, PreferenceExpr, PreferenceMeasure, PreferenceParameter,
    PreferencePolicy, PreferenceRoute, PreferenceSet, PreferenceVerdict, RegionStatus,
    RelationOrigin, RelationSense, StructuralCertifiedResult, StructuralPolicy,
    StructuralPreferenceEvaluation, StructuralPreferenceResult, StructuralPreferenceVerdict,
    ValueModel, ValueRelation, chain_joint_law, decide_across_structures_with_context,
    decide_preferences_on_means_with_context, decide_preferences_with_context,
    experiment_arm_quantity, experiment_mean_source, joint_law_of, preference_route,
    source_requirement,
};
use antecedent::analysis::preference_decision::{
    ClassSampling, ClassSize, CompiledPreferenceDecision, CompiledPreferenceEvaluation,
    CompiledStructure, CrossStudyCovariance, MassEstimate, OutcomeDependence,
    PreferenceDecisionExecution, PreferenceJointLaw, PreferenceRouteError,
    SampledStructuralEvaluation, SampledStructuralVerdict, StructureIdentification,
    compile_preference_decision_across_prepared_studies,
};
use antecedent::analysis::propagation_chain::{PropagationChain, PropagationChainRecordWire};
use antecedent::contract::ExternalRefusal;
use antecedent::design::decision_adapters::{
    AdaptedClaims, ClaimProbability, SuppliedClaim, adapt_finite_scenarios,
    adapt_graph_dependent_claims, adapt_point_claim, adapt_weighted_graph_atoms,
};
use antecedent::design::decision_artifact::source_digest;
use antecedent::design::decision_contract::{SourceRepresentation, UtilityExpr};
use antecedent::design::decision_eval::MeanEstimate;
use antecedent::design::decision_robustness::{AtomSupport, InputSupport};
use antecedent::io::quantity_wire::ScientificQuantityWire;
use antecedent::planner::PlanView;
use antecedent::planner::lowered::CarrierDemand;
use antecedent::scientific::ScientificQuantity;
use antecedent::scientific::SemanticDigest;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

use crate::bridge::{check_declared_size, external_refusal_json};
use crate::distribution_api::PyJointDistributionArtifact;

/// Largest JSON declaration accepted (records may carry joint draws of every option).
const MAX_PREFERENCE_JSON_BYTES: usize = 64 * 1024 * 1024;

/// Most draw bytes the structures of one decision may copy between them.
const MAX_CLAIM_DRAW_BYTES: usize = 512 * 1024 * 1024;

/// `(result JSON, refusal JSON)`: exactly one is set.
type Reply = (Option<String>, Option<String>);

fn parse<T: DeserializeOwned>(text: &str, what: &str) -> PyResult<T> {
    check_declared_size(text, MAX_PREFERENCE_JSON_BYTES, what)?;
    serde_json::from_str(text).map_err(|e| crate::value_err(format!("invalid {what}: {e}")))
}

fn refusal(value: &ExternalRefusal) -> String {
    external_refusal_json(value)
}

fn refusal_value(value: &ExternalRefusal) -> Value {
    serde_json::from_str(&external_refusal_json(value)).unwrap_or(Value::Null)
}

fn reply(outcome: Result<Value, String>) -> Reply {
    match outcome {
        Ok(value) => (Some(value.to_string()), None),
        Err(refused) => (None, Some(refused)),
    }
}

/// `CamelCase` variant name of a `Debug` rendering, as `snake_case`.
fn debug_kind(debug: &str) -> String {
    let name: String = debug.chars().take_while(char::is_ascii_alphanumeric).collect();
    let mut out = String::with_capacity(name.len() + 4);
    for (index, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if index > 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Declaration wires.
// ---------------------------------------------------------------------------

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

#[derive(Clone, Debug, Deserialize)]
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
    assumption: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SenseWire {
    AtLeast,
    AtMost,
    Equal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RelationOriginWire {
    Declared,
    ShapeRule,
    Choice { record_id: String, rejected_option: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RelationWire {
    id: String,
    terms: Vec<(usize, f64)>,
    sense: SenseWire,
    bound: f64,
    origin: RelationOriginWire,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelWire {
    attributes: Vec<AttributeWire>,
    parameters: Vec<ParameterWire>,
    relations: Vec<RelationWire>,
    /// Recorded choices under a value with shape parameters: a learned set holds them, and a
    /// decision on it must read them rather than the declared box.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    value_relations: Vec<ValueRelationWire>,
    value: ExprWire,
    utility_units: String,
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
    origin: RelationOriginWire,
    chosen: BeliefRowsWire,
    rejected: BeliefRowsWire,
    tolerance: f64,
}

fn origin_from(w: RelationOriginWire) -> RelationOrigin {
    match w {
        RelationOriginWire::Declared => RelationOrigin::Declared,
        RelationOriginWire::ShapeRule => RelationOrigin::ShapeRule,
        RelationOriginWire::Choice { record_id, rejected_option } => {
            RelationOrigin::Choice { record_id, rejected_option }
        }
    }
}

fn origin_to(o: &RelationOrigin) -> RelationOriginWire {
    match o {
        RelationOrigin::Declared => RelationOriginWire::Declared,
        RelationOrigin::ShapeRule => RelationOriginWire::ShapeRule,
        RelationOrigin::Choice { record_id, rejected_option } => RelationOriginWire::Choice {
            record_id: record_id.clone(),
            rejected_option: rejected_option.clone(),
        },
    }
}

fn value_relation_from(w: ValueRelationWire) -> ValueRelation {
    ValueRelation {
        id: w.id,
        origin: origin_from(w.origin),
        chosen: BeliefRows { rows: w.chosen.rows, weights: w.chosen.weights },
        rejected: BeliefRows { rows: w.rejected.rows, weights: w.rejected.weights },
        tolerance: w.tolerance,
    }
}

fn value_relation_to(r: &ValueRelation) -> ValueRelationWire {
    ValueRelationWire {
        id: r.id.clone(),
        origin: origin_to(&r.origin),
        chosen: BeliefRowsWire { rows: r.chosen.rows.clone(), weights: r.chosen.weights.clone() },
        rejected: BeliefRowsWire {
            rows: r.rejected.rows.clone(),
            weights: r.rejected.weights.clone(),
        },
        tolerance: r.tolerance,
    }
}

fn attribute_from(w: AttributeWire) -> Attribute {
    Attribute { id: w.id, units: w.units }
}

fn attribute_to(a: &Attribute) -> AttributeWire {
    AttributeWire { id: a.id.clone(), units: a.units.clone() }
}

fn parameter_to(p: &PreferenceParameter) -> ParameterWire {
    ParameterWire {
        id: p.id.clone(),
        units: p.units.clone(),
        lower: p.lower,
        upper: p.upper,
        assumption: p.assumption_id.clone(),
    }
}

fn relation_from(w: RelationWire) -> LinearRelation {
    LinearRelation {
        id: w.id,
        terms: w.terms,
        sense: match w.sense {
            SenseWire::AtLeast => RelationSense::AtLeast,
            SenseWire::AtMost => RelationSense::AtMost,
            SenseWire::Equal => RelationSense::Equal,
        },
        bound: w.bound,
        origin: origin_from(w.origin),
    }
}

fn relation_to(r: &LinearRelation) -> RelationWire {
    RelationWire {
        id: r.id.clone(),
        terms: r.terms.clone(),
        sense: match r.sense {
            RelationSense::AtLeast => SenseWire::AtLeast,
            RelationSense::AtMost => SenseWire::AtMost,
            RelationSense::Equal => SenseWire::Equal,
        },
        bound: r.bound,
        origin: origin_to(&r.origin),
    }
}

fn model_from(w: ModelWire) -> ValueModel {
    ValueModel {
        attributes: w.attributes.into_iter().map(attribute_from).collect(),
        preferences: PreferenceSet {
            parameters: w
                .parameters
                .into_iter()
                .map(|p| PreferenceParameter {
                    id: p.id,
                    units: p.units,
                    lower: p.lower,
                    upper: p.upper,
                    assumption_id: p.assumption,
                })
                .collect(),
            relations: w.relations.into_iter().map(relation_from).collect(),
            value_relations: w.value_relations.into_iter().map(value_relation_from).collect(),
        },
        value: w.value.into(),
        utility_units: w.utility_units,
    }
}

fn model_to(m: &ValueModel) -> ModelWire {
    ModelWire {
        attributes: m.attributes.iter().map(attribute_to).collect(),
        parameters: m.preferences.parameters.iter().map(parameter_to).collect(),
        relations: m.preferences.relations.iter().map(relation_to).collect(),
        value_relations: m.preferences.value_relations.iter().map(value_relation_to).collect(),
        value: ExprWire::from(&m.value),
        utility_units: m.utility_units.clone(),
    }
}

fn model_value(m: &ValueModel) -> Value {
    serde_json::to_value(model_to(m)).unwrap_or(Value::Null)
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum InputWire {
    Quantity(ScientificQuantityWire),
    Fixed { value: f64, units: String },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionWire {
    id: String,
    kind: String,
    inputs: Vec<InputWire>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
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

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConstraintWire {
    id: String,
    expr: UtilityWire,
    bound: f64,
    min_probability: f64,
    units: String,
    applies_to: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
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

fn quantity_from(wire: ScientificQuantityWire) -> PyResult<ScientificQuantity> {
    ScientificQuantity::try_from(wire).map_err(|e| crate::value_err(String::from(e)))
}

fn kind_from(name: &str) -> PyResult<ActionKind> {
    Ok(match name {
        "intervention" => ActionKind::Intervention,
        "policy" => ActionKind::Policy,
        "regime" => ActionKind::Regime,
        "study" => ActionKind::Study,
        "external" => ActionKind::External,
        other => return Err(crate::value_err(format!("unknown action kind {other:?}"))),
    })
}

fn structural_from(name: &str) -> PyResult<StructuralPolicy> {
    Ok(match name {
        "require_invariant_best_action" => StructuralPolicy::RequireInvariantBestAction,
        "maximin" => StructuralPolicy::Maximin,
        "bayes_over_structures" => StructuralPolicy::BayesOverStructures,
        "report_only" => StructuralPolicy::ReportOnly,
        other => return Err(crate::value_err(format!("unknown structural policy {other:?}"))),
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

fn policy_from(name: &str) -> PyResult<PreferencePolicy> {
    Ok(match name {
        "require_invariant_best_action" => PreferencePolicy::RequireInvariantBestAction,
        "maximin" => PreferencePolicy::Maximin,
        "minimax_regret" => PreferencePolicy::MinimaxRegret,
        "weighted" => PreferencePolicy::Weighted,
        "report_only" => PreferencePolicy::ReportOnly,
        other => return Err(crate::value_err(format!("unknown preference policy {other:?}"))),
    })
}

fn policy_name(policy: PreferencePolicy) -> &'static str {
    match policy {
        PreferencePolicy::RequireInvariantBestAction => "require_invariant_best_action",
        PreferencePolicy::Maximin => "maximin",
        PreferencePolicy::MinimaxRegret => "minimax_regret",
        PreferencePolicy::Weighted => "weighted",
        PreferencePolicy::ReportOnly => "report_only",
    }
}

fn contract_from(json: &str) -> PyResult<PreferenceContract> {
    let wire: ContractWire = parse(json, "preference contract")?;
    let actions = wire
        .actions
        .into_iter()
        .map(|action| {
            let inputs = action
                .inputs
                .into_iter()
                .map(|input| {
                    Ok(match input {
                        InputWire::Quantity(q) => ActionInput::Quantity(quantity_from(q)?),
                        InputWire::Fixed { value, units } => ActionInput::Fixed { value, units },
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PreferenceAction { id: action.id, kind: kind_from(&action.kind)?, inputs })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let constraints = wire
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
        .collect();
    Ok(PreferenceContract {
        model: model_from(wire.model),
        actions,
        criterion: wire.criterion.into(),
        constraints,
        target_population: wire.target_population,
        horizon: wire.horizon,
        structural_policy: structural_from(&wire.structural_policy)?,
        preference_policy: policy_from(&wire.preference_policy)?,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum MeasureOriginWire {
    Declared { assumption: String },
    ChoiceEstimate { estimate: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeasureWire {
    points: Vec<Vec<f64>>,
    #[serde(default)]
    weights: Option<Vec<f64>>,
    #[serde(default)]
    labels: Option<Vec<String>>,
    origin: MeasureOriginWire,
    #[serde(default)]
    unresolved: Vec<usize>,
}

fn measure_from(w: MeasureWire) -> PreferenceMeasure {
    PreferenceMeasure {
        points: w.points,
        weights: w.weights,
        labels: w.labels,
        origin: match w.origin {
            MeasureOriginWire::Declared { assumption } => {
                MeasureOrigin::Declared { assumption_id: assumption }
            }
            MeasureOriginWire::ChoiceEstimate { estimate } => {
                MeasureOrigin::ChoiceEstimate { estimate_identity: estimate }
            }
        },
        unresolved: w.unresolved,
    }
}

fn measure_value(m: &PreferenceMeasure) -> Value {
    let origin = match &m.origin {
        MeasureOrigin::Declared { assumption_id } => {
            MeasureOriginWire::Declared { assumption: assumption_id.clone() }
        }
        MeasureOrigin::ChoiceEstimate { estimate_identity } => {
            MeasureOriginWire::ChoiceEstimate { estimate: estimate_identity.clone() }
        }
    };
    serde_json::to_value(MeasureWire {
        points: m.points.clone(),
        weights: m.weights.clone(),
        labels: m.labels.clone(),
        origin,
        unresolved: m.unresolved.clone(),
    })
    .unwrap_or(Value::Null)
}

fn measure_opt(json: Option<&str>) -> PyResult<Option<PreferenceMeasure>> {
    json.map(|text| parse::<MeasureWire>(text, "measure").map(measure_from)).transpose()
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct OptionsWire {
    /// Both routes' tie tolerance.
    tie_tolerance: Option<f64>,
    /// The certified route's tie tolerance, when it differs from the exact route's.
    certified_tie_tolerance: Option<f64>,
    vertex_cap: Option<usize>,
    max_boxes: Option<usize>,
    min_width_fraction: Option<f64>,
    lp_feasibility_tolerance: Option<f64>,
    lp_optimality_tolerance: Option<f64>,
    lp_max_pivots: Option<usize>,
}

fn options_from(json: &str) -> PyResult<PreferenceEvaluationOptions> {
    let wire: OptionsWire = parse(json, "decision options")?;
    let mut options = PreferenceEvaluationOptions::default();
    if let Some(tolerance) = wire.tie_tolerance {
        options.exact.tie_tolerance = tolerance;
        options.certified.tie_tolerance = tolerance;
    }
    if let Some(tolerance) = wire.certified_tie_tolerance {
        options.certified.tie_tolerance = tolerance;
    }
    if let Some(tolerance) = wire.lp_feasibility_tolerance {
        options.exact.lp.feasibility_tolerance = tolerance;
    }
    if let Some(tolerance) = wire.lp_optimality_tolerance {
        options.exact.lp.optimality_tolerance = tolerance;
    }
    if let Some(pivots) = wire.lp_max_pivots {
        options.exact.lp.max_pivots = pivots;
    }
    if let Some(cap) = wire.vertex_cap {
        options.exact.vertex_cap = cap;
    }
    if let Some(boxes) = wire.max_boxes {
        options.certified.max_boxes = boxes;
    }
    if let Some(fraction) = wire.min_width_fraction {
        options.certified.min_width_fraction = fraction;
    }
    Ok(options)
}

/// The options in the wire [`options_from`] reads, every field set: reading it back gives the
/// same options, the certified tie tolerance and the linear-program tolerances included.
fn options_value(options: PreferenceEvaluationOptions) -> Value {
    json!({
        "tie_tolerance": options.exact.tie_tolerance,
        "certified_tie_tolerance": options.certified.tie_tolerance,
        "vertex_cap": options.exact.vertex_cap,
        "max_boxes": options.certified.max_boxes,
        "min_width_fraction": options.certified.min_width_fraction,
        "lp_feasibility_tolerance": options.exact.lp.feasibility_tolerance,
        "lp_optimality_tolerance": options.exact.lp.optimality_tolerance,
        "lp_max_pivots": options.exact.lp.max_pivots,
    })
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeanSourceWire {
    coordinates: Vec<ScientificQuantityWire>,
    means: Vec<f64>,
    provider_id: String,
    snapshot_id: String,
    causal_contract_id: String,
    rng_id: String,
}

fn mean_source_from(json: &str) -> PyResult<MeanSource> {
    let wire: MeanSourceWire = parse(json, "mean source")?;
    Ok(MeanSource {
        coordinates: wire.coordinates.into_iter().map(quantity_from).collect::<PyResult<_>>()?,
        means: wire.means,
        estimate: MeanEstimate::Frequentist,
        provider_id: wire.provider_id,
        snapshot_id: wire.snapshot_id,
        causal_contract_id: wire.causal_contract_id,
        rng_id: wire.rng_id,
    })
}

// ---------------------------------------------------------------------------
// Structures (the same claim wire `decision_robust` declares).
// ---------------------------------------------------------------------------

fn invalid_json(detail: &str, offending: Option<String>) -> String {
    refusal(&ExternalRefusal {
        code: antecedent::reason::reason_code!("invalid_argument"),
        stage: "adapt",
        detail: detail.to_owned(),
        offending,
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    })
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbabilityWire {
    kind: String,
    #[serde(default)]
    value: Option<f64>,
    #[serde(default)]
    count: Option<u64>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct InputSupportWire {
    action_id: String,
    input: usize,
    status: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SupportWire {
    overall: String,
    #[serde(default)]
    per_input: Vec<InputSupportWire>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ClaimWire {
    id: String,
    #[serde(default)]
    probability: Option<ProbabilityWire>,
    status: String,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    support: Option<SupportWire>,
}

fn support_status(name: &str) -> Result<antecedent::query::SupportStatus, String> {
    antecedent::query::SupportStatus::from_name(name).ok_or_else(|| {
        invalid_json(
            antecedent::reason::reason_details::decision_adapters::UNKNOWN_SUPPORT_STATUS,
            Some(name.into()),
        )
    })
}

fn support_from(wire: Option<SupportWire>) -> Result<AtomSupport, String> {
    let Some(wire) = wire else {
        return Ok(AtomSupport::unassessed());
    };
    Ok(AtomSupport {
        overall: support_status(&wire.overall)?,
        per_input: wire
            .per_input
            .into_iter()
            .map(|p| {
                Ok(InputSupport {
                    action_id: p.action_id,
                    input: p.input,
                    status: support_status(&p.status)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    })
}

fn probability_from(wire: Option<ProbabilityWire>, id: &str) -> Result<ClaimProbability, String> {
    Ok(match wire {
        None => ClaimProbability::Unspecified,
        Some(p) => match (p.kind.as_str(), p.value, p.count) {
            ("unspecified", None, None) => ClaimProbability::Unspecified,
            ("genuine", Some(value), None) => ClaimProbability::Genuine(value),
            ("completion_count", None, Some(count)) => ClaimProbability::CompletionCount(count),
            _ => {
                return Err(invalid_json(
                    antecedent::reason::reason_details::decision_adapters::INVALID_PROBABILITY,
                    Some(id.to_owned()),
                ));
            }
        },
    })
}

fn claim_from(
    wire: ClaimWire,
    source: Option<&PyJointDistributionArtifact>,
) -> Result<SuppliedClaim, String> {
    use antecedent::reason::reason_details::decision_adapters as details;
    let bad = |slot: &str| invalid_json(slot, Some(wire.id.clone()));
    let evidence = match (wire.status.as_str(), source) {
        ("evaluated", Some(source)) => AtomEvidence::Evaluated(Box::new(source.inner().clone())),
        ("evaluated", None) => return Err(bad(details::EVALUATED_NEEDS_DRAWS)),
        ("unidentified", _) => AtomEvidence::Unidentified,
        ("unevaluated", _) => match wire.reason.as_deref() {
            Some(reason) if !reason.trim().is_empty() => {
                AtomEvidence::Unevaluated(reason.to_owned())
            }
            _ => return Err(bad(details::UNEVALUATED_NEEDS_REASON)),
        },
        _ => return Err(bad(details::UNKNOWN_CLAIM_STATUS)),
    };
    let probability = probability_from(wire.probability, &wire.id)?;
    Ok(SuppliedClaim { id: wire.id, probability, evidence, support: support_from(wire.support)? })
}

fn adapt(
    policy: StructuralPolicy,
    kind: &str,
    claims_json: &str,
    sources: &[Option<PyRef<'_, PyJointDistributionArtifact>>],
) -> Result<AdaptedClaims, String> {
    use antecedent::reason::reason_details::decision_adapters as details;
    let wires: Vec<ClaimWire> = serde_json::from_str(claims_json)
        .map_err(|e| invalid_json(details::INVALID_CLAIMS, Some(e.to_string())))?;
    if wires.len() != sources.len() {
        return Err(invalid_json(details::CLAIMS_SOURCES_MISMATCH, None));
    }
    // Each evaluated structure owns a copy of its draws: bound the total before cloning.
    let cloned_bytes = wires
        .iter()
        .zip(sources)
        .filter(|(wire, _)| wire.status == "evaluated")
        .filter_map(|(_, source)| source.as_deref())
        .try_fold(0usize, |total, source| {
            total.checked_add(source.inner().draws().len().checked_mul(size_of::<f64>())?)
        });
    if cloned_bytes.is_none_or(|bytes| bytes > MAX_CLAIM_DRAW_BYTES) {
        return Err(invalid_json(
            details::CLAIM_DRAWS_TOO_LARGE,
            Some(format!("the structures copy more than {MAX_CLAIM_DRAW_BYTES} bytes of draws")),
        ));
    }
    let claims = wires
        .into_iter()
        .zip(sources)
        .map(|(wire, source)| claim_from(wire, source.as_deref()))
        .collect::<Result<Vec<_>, String>>()?;
    match kind {
        "point" => {
            let Ok([claim]) = <[SuppliedClaim; 1]>::try_from(claims) else {
                return Err(invalid_json(details::POINT_CLAIM_IS_SINGLE, None));
            };
            adapt_point_claim(&claim.id, claim.evidence, claim.support, policy)
        }
        "graph_dependent" => adapt_graph_dependent_claims(claims, policy),
        "weighted_graph_posterior" => adapt_weighted_graph_atoms(claims, policy),
        "finite_scenarios" => adapt_finite_scenarios(claims, policy),
        other => return Err(invalid_json(details::UNKNOWN_CLAIM_KIND, Some(other.into()))),
    }
    .map_err(|e| refusal(&e.to_refusal()))
}

// ---------------------------------------------------------------------------
// Result JSON.
// ---------------------------------------------------------------------------

macro_rules! exclusions_json {
    ($list:expr) => {
        $list
            .iter()
            .map(|e| {
                json!({
                    "constraint_id": e.constraint_id,
                    "probability": e.probability,
                    "required": e.required,
                })
            })
            .collect::<Vec<Value>>()
    };
}

macro_rules! receipt_json {
    ($receipt:expr) => {
        json!({
            "provider_id": $receipt.provider_id,
            "snapshot_id": $receipt.snapshot_id,
            "rng_id": $receipt.rng_id,
            "causal_contract_id": $receipt.causal_contract_id,
        })
    };
}

fn exact_verdict_json(verdict: &PreferenceVerdict) -> Value {
    match verdict {
        PreferenceVerdict::NecessarilyBest(a) => json!({"kind": "necessarily_best", "action": a}),
        PreferenceVerdict::NeverBeaten { action, ties_with } => {
            json!({"kind": "never_beaten", "action": action, "ties_with": ties_with})
        }
        PreferenceVerdict::PreferenceDependent { possibly_best } => {
            json!({"kind": "preference_dependent", "actions": possibly_best})
        }
        PreferenceVerdict::WorstCaseChoice(a) => json!({"kind": "worst_case_choice", "action": a}),
        PreferenceVerdict::MinimaxRegretChoice(a) => {
            json!({"kind": "minimax_regret_choice", "action": a})
        }
        PreferenceVerdict::MeasureChoice { action, share } => {
            json!({"kind": "measure_choice", "action": action, "share": share})
        }
        PreferenceVerdict::Tied(actions) => json!({"kind": "tied", "actions": actions}),
        PreferenceVerdict::ReportOnly => json!({"kind": "report_only"}),
        PreferenceVerdict::NoAdmissibleAction => json!({"kind": "no_admissible_action"}),
    }
}

fn certified_verdict_json(verdict: &CertifiedVerdict) -> Value {
    match verdict {
        CertifiedVerdict::NecessarilyBest(a) => json!({"kind": "necessarily_best", "action": a}),
        CertifiedVerdict::PreferenceDependent { certified_somewhere } => {
            json!({"kind": "preference_dependent", "actions": certified_somewhere})
        }
        CertifiedVerdict::WorstCaseChoice(a) => json!({"kind": "worst_case_choice", "action": a}),
        CertifiedVerdict::MinimaxRegretChoice(a) => {
            json!({"kind": "minimax_regret_choice", "action": a})
        }
        CertifiedVerdict::MeasureChoice { action, share } => {
            json!({"kind": "measure_choice", "action": action, "share": share})
        }
        CertifiedVerdict::Undetermined { candidates, undecided_fraction } => json!({
            "kind": "undetermined",
            "actions": candidates,
            "undecided_fraction": undecided_fraction,
        }),
        CertifiedVerdict::ReportOnly => json!({"kind": "report_only"}),
        CertifiedVerdict::NoAdmissibleAction => json!({"kind": "no_admissible_action"}),
    }
}

fn exact_json(r: &PreferenceDecisionResult) -> Value {
    let actions: Vec<Value> = r
        .actions
        .iter()
        .map(|a| {
            json!({
                "id": a.id,
                "admissible": a.admissible,
                "exclusions": exclusions_json!(a.exclusions),
                "coefficients": a.coefficients,
                "value_range": a.value_range,
                "necessarily_best": a.necessarily_best,
                "uniquely_best": a.uniquely_best,
                "dominated_by": a.dominated_by,
                "possibly_best": a.possibly_best,
                "worst_case_regret": a.worst_case_regret,
                "measure_share": a.measure_share,
                "measure_value": a.measure_value,
            })
        })
        .collect();
    let switches: Vec<Value> = r
        .switches
        .iter()
        .map(|s| {
            json!({
                "first": s.first,
                "second": s.second,
                "margin_coefficients": s.margin_coefficients,
                "margin_range": s.margin_range,
                "favours_first_at": s.favours_first_at,
                "favours_second_at": s.favours_second_at,
                "threshold": s.threshold.as_ref().map(|t| json!({
                    "parameter_id": t.parameter_id,
                    "value": t.value,
                    "first_leads_above": t.first_leads_above,
                })),
            })
        })
        .collect();
    json!({
        "route": "exact",
        "contract_identity": r.contract_identity,
        "policy": policy_name(r.policy),
        "parameter_ids": r.parameter_ids,
        "parameter_ranges": r.parameter_ranges,
        "statement": r.statement(),
        "actions": actions,
        "switches": switches,
        "verdict": exact_verdict_json(&r.verdict),
        "draw_share_best_everywhere": r.draw_share_best_everywhere,
        "unresolved_measure_share": r.unresolved_measure_share,
        "assumptions": r.assumptions,
        "source": receipt_json!(r.source),
        "n_draws": r.n_draws,
        "effective_tie_tolerance": r.effective_tie_tolerance,
        "undecided_fraction": Value::Null,
        "regions": Vec::<Value>::new(),
    })
}

fn certified_json(r: &CertifiedPreferenceResult) -> Value {
    let actions: Vec<Value> = r
        .actions
        .iter()
        .map(|a| {
            json!({
                "id": a.id,
                "admissible": a.admissible,
                "exclusions": exclusions_json!(a.exclusions),
                "best_fraction": a.best_fraction,
                "value_bounds": a.value_bounds,
                "worst_case_bounds": a.worst_case_bounds,
                "worst_case_regret_bounds": a.worst_case_regret_bounds,
                "measure_value": a.measure_value,
                "measure_share": a.measure_share,
            })
        })
        .collect();
    let regions: Vec<Value> = r
        .regions
        .iter()
        .map(|region| {
            let status = match &region.status {
                RegionStatus::Best(action) => json!({"kind": "best", "action": action}),
                RegionStatus::Undecided { candidates } => {
                    json!({"kind": "undecided", "candidates": candidates})
                }
            };
            json!({
                "lower": region.lower,
                "upper": region.upper,
                "status": status,
                "touches_relation_boundary": region.touches_relation_boundary,
            })
        })
        .collect();
    json!({
        "route": "certified",
        "contract_identity": r.contract_identity,
        "policy": policy_name(r.policy),
        "parameter_ids": r.parameter_ids,
        "actions": actions,
        "switches": Vec::<Value>::new(),
        "regions": regions,
        "undecided_fraction": r.undecided_fraction,
        "boxes_examined": r.boxes_examined,
        "verdict": certified_verdict_json(&r.verdict),
        "assumptions": r.assumptions,
        "source": receipt_json!(r.source),
        "n_draws": r.n_draws,
        "effective_tie_tolerance": r.effective_tie_tolerance,
    })
}

/// The route label of `contract` when it names the route that answered.
fn route_reason(contract: &PreferenceContract, route: PreferenceRoute) -> Value {
    match preference_route(contract) {
        Ok(choice) if choice.route == route => json!(choice.reason),
        _ => Value::Null,
    }
}

fn evaluation_json(evaluation: &PreferenceEvaluation, contract: &PreferenceContract) -> Value {
    let mut value = match evaluation {
        PreferenceEvaluation::Exact(result) => exact_json(result),
        PreferenceEvaluation::Certified(result) => certified_json(result),
    };
    value["route_reason"] = route_reason(contract, evaluation.route());
    value["statement"] = json!(evaluation.statement());
    value
}

fn atom_json(
    id: &str,
    probability: Option<f64>,
    status: &str,
    reason: Option<&str>,
    result: Option<Value>,
) -> Value {
    json!({
        "id": id,
        "probability": probability,
        "status": status,
        "reason": reason,
        "result": result,
    })
}

fn structural_exact_json(r: &StructuralPreferenceResult) -> Value {
    use antecedent::analysis::preference_decision::PreferenceAtomStatus as S;
    let atoms: Vec<Value> = r
        .atoms
        .iter()
        .map(|atom| match &atom.status {
            S::Evaluated(inner) => {
                atom_json(&atom.id, atom.probability, "evaluated", None, Some(exact_json(inner)))
            }
            S::Unidentified => atom_json(&atom.id, atom.probability, "unidentified", None, None),
            S::Unevaluated(why) => {
                atom_json(&atom.id, atom.probability, "unevaluated", Some(why.as_str()), None)
            }
        })
        .collect();
    let verdict = match &r.verdict {
        StructuralPreferenceVerdict::Combined(v) => {
            json!({"kind": "combined", "verdict": exact_verdict_json(v)})
        }
        StructuralPreferenceVerdict::StructuresDisagree(choices) => {
            let structures: Vec<Value> = choices
                .iter()
                .map(|c| json!({"structure": c.atom, "verdict": exact_verdict_json(&c.verdict)}))
                .collect();
            json!({"kind": "structures_disagree", "structures": structures})
        }
        StructuralPreferenceVerdict::InsufficientScience(why) => {
            json!({"kind": "insufficient_science", "reason": why})
        }
        StructuralPreferenceVerdict::ReportOnly => json!({"kind": "report_only"}),
    };
    json!({
        "route": "exact",
        "contract_identity": r.contract_identity,
        "structural_policy": structural_name(r.structural_policy),
        "policy": policy_name(r.preference_policy),
        "structures": atoms,
        "combined": r.combined.as_ref().map(exact_json),
        "unidentified_mass": r.unidentified_mass,
        "unevaluated_mass": r.unevaluated_mass,
        "evaluated_mass": r.evaluated_mass,
        "structural_verdict": verdict,
        "statement": r.statement(),
    })
}

fn structural_certified_json(r: &StructuralCertifiedResult) -> Value {
    use antecedent::analysis::preference_decision::CertifiedAtomStatus as S;
    use antecedent::analysis::preference_decision::StructuralCertifiedVerdict as V;
    let atoms: Vec<Value> = r
        .atoms
        .iter()
        .map(|atom| match &atom.status {
            S::Evaluated(inner) => atom_json(
                &atom.id,
                atom.probability,
                "evaluated",
                None,
                Some(certified_json(inner)),
            ),
            S::Unidentified => atom_json(&atom.id, atom.probability, "unidentified", None, None),
            S::Unevaluated(why) => {
                atom_json(&atom.id, atom.probability, "unevaluated", Some(why.as_str()), None)
            }
        })
        .collect();
    let verdict = match &r.verdict {
        V::Combined(v) => json!({"kind": "combined", "verdict": certified_verdict_json(v)}),
        V::StructuresDisagree(choices) => {
            let structures: Vec<Value> = choices
                .iter()
                .map(
                    |c| json!({"structure": c.atom, "verdict": certified_verdict_json(&c.verdict)}),
                )
                .collect();
            json!({"kind": "structures_disagree", "structures": structures})
        }
        V::InsufficientScience(why) => json!({"kind": "insufficient_science", "reason": why}),
        V::ReportOnly => json!({"kind": "report_only"}),
    };
    json!({
        "route": "certified",
        "contract_identity": r.contract_identity,
        "structural_policy": structural_name(r.structural_policy),
        "policy": policy_name(r.preference_policy),
        "structures": atoms,
        "combined": r.combined.as_ref().map(certified_json),
        "unidentified_mass": r.unidentified_mass,
        "unevaluated_mass": r.unevaluated_mass,
        "evaluated_mass": r.evaluated_mass,
        "structural_verdict": verdict,
    })
}

fn structural_json(
    evaluation: &StructuralPreferenceEvaluation,
    contract: &PreferenceContract,
) -> Value {
    let mut value = match evaluation {
        StructuralPreferenceEvaluation::Exact(r) => structural_exact_json(r),
        StructuralPreferenceEvaluation::Certified(r) => structural_certified_json(r),
    };
    value["route_reason"] = route_reason(contract, evaluation.route());
    value
}

// ---------------------------------------------------------------------------
// Declarations and decisions.
// ---------------------------------------------------------------------------

/// Validate a value model; returns `{identity, shape_parameters, affine}` or a refusal.
/// The content identity of a whole table, as a study of that table carries it: the identity
/// a column projection names as the table its rows came from, so studies of different columns
/// of one table are judged as sharing its rows.
#[pyfunction]
fn table_content_identity(names: Vec<String>, columns: Vec<Bound<'_, PyAny>>) -> PyResult<String> {
    crate::catch_ffi(|| {
        let data = crate::tabular_from_py_columns(names, columns)?;
        Ok(antecedent::analysis::preference_decision::RowProvenance::of_table(&data)
            .table()
            .to_owned())
    })
}

#[pyfunction]
fn preference_model_describe(model_json: &str) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let model = model_from(parse(model_json, "value model")?);
        Ok(reply(match model.identity() {
            Ok(identity) => {
                let shape: Vec<String> = model
                    .shape_parameters()
                    .into_iter()
                    .filter_map(|j| model.preferences.parameters.get(j).map(|p| p.id.clone()))
                    .collect();
                Ok(json!({
                    "identity": identity,
                    "affine": model.linear_form().is_some(),
                    "shape_parameters": shape,
                }))
            }
            Err(error) => Err(refusal(&error.to_refusal())),
        }))
    })
}

/// Validate a preference contract; returns `{identity, route, route_reason, needs}` (`needs`
/// is `means` or `joint_draws`) or a refusal.
#[pyfunction]
fn preference_contract_describe(contract_json: &str) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let contract = contract_from(contract_json)?;
        let run = || -> Result<Value, String> {
            let identity = contract.identity().map_err(|e| refusal(&e.to_refusal()))?;
            let choice = preference_route(&contract).map_err(|e| refusal(&e.to_refusal()))?;
            let requirement = source_requirement(&contract).map_err(|e| refusal(&e))?;
            let means = requirement.any_of.contains(&SourceRepresentation::Mean);
            Ok(json!({
                "identity": identity,
                "route": match choice.route {
                    PreferenceRoute::Exact => "exact",
                    PreferenceRoute::Certified => "certified",
                },
                "route_reason": choice.reason,
                "needs": if means { "means" } else { "joint_draws" },
            }))
        };
        Ok(reply(run()))
    })
}

/// Decide a preference contract on aligned joint draws, by the route rule. The decision reads
/// `cancel` at its work boundaries.
#[pyfunction]
#[pyo3(signature = (contract_json, measure_json, options_json, source, cancel=None))]
fn decide_preferences_joint(
    contract_json: &str,
    measure_json: Option<&str>,
    options_json: &str,
    source: &PyJointDistributionArtifact,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let contract = contract_from(contract_json)?;
        let measure = measure_opt(measure_json)?;
        let options = options_from(options_json)?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        Ok(reply(
            decide_preferences_with_context(
                &contract,
                source.inner(),
                measure.as_ref(),
                options,
                &ctx,
            )
            .map(|e| evaluation_json(&e, &contract))
            .map_err(|e| refusal(&e.to_refusal())),
        ))
    })
}

/// Decide a preference contract on a mean-only source (exact route only). The decision reads
/// `cancel` at its work boundaries.
#[pyfunction]
#[pyo3(signature = (contract_json, measure_json, options_json, mean_source_json, cancel=None))]
fn decide_preferences_means(
    contract_json: &str,
    measure_json: Option<&str>,
    options_json: &str,
    mean_source_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let contract = contract_from(contract_json)?;
        let measure = measure_opt(measure_json)?;
        let options = options_from(options_json)?;
        let source = mean_source_from(mean_source_json)?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        Ok(reply(
            decide_preferences_on_means_with_context(
                &contract,
                &source,
                measure.as_ref(),
                options,
                &ctx,
            )
            .map(|e| evaluation_json(&e, &contract))
            .map_err(|e| refusal(&e.to_refusal())),
        ))
    })
}

/// Decide a preference contract across structures (`kind` as the robust decision's claims:
/// `point`, `graph_dependent`, `weighted_graph_posterior`, `finite_scenarios`) and its
/// preference set together; `sources` aligns with the claims. The decision reads `cancel` at
/// its work boundaries.
#[pyfunction]
#[pyo3(signature = (contract_json, measure_json, options_json, kind, claims_json, sources, cancel=None))]
fn decide_preferences_structures(
    contract_json: &str,
    measure_json: Option<&str>,
    options_json: &str,
    kind: &str,
    claims_json: &str,
    sources: Vec<Option<PyRef<'_, PyJointDistributionArtifact>>>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let contract = contract_from(contract_json)?;
        let measure = measure_opt(measure_json)?;
        let options = options_from(options_json)?;
        check_declared_size(claims_json, MAX_PREFERENCE_JSON_BYTES, "structures")?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        let run = || -> Result<Value, String> {
            let claims = adapt(contract.structural_policy, kind, claims_json, &sources)?;
            let evaluation = decide_across_structures_with_context(
                &contract,
                &claims,
                measure.as_ref(),
                options,
                &ctx,
            )
            .map_err(|e| refusal(&e))?;
            Ok(structural_json(&evaluation, &contract))
        };
        Ok(reply(run()))
    })
}

/// Decide on aligned joint draws and export the bound decision record; returns the bytes or
/// a refusal. The decision reads `cancel` at its work boundaries.
#[pyfunction]
#[pyo3(signature = (contract_json, measure_json, options_json, source, artifact_id, cancel=None))]
fn export_preference_decision<'py>(
    py: Python<'py>,
    contract_json: &str,
    measure_json: Option<&str>,
    options_json: &str,
    source: &PyJointDistributionArtifact,
    artifact_id: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<Bound<'py, PyBytes>>, Option<String>)> {
    crate::catch_ffi(|| {
        let contract = contract_from(contract_json)?;
        let measure = measure_opt(measure_json)?;
        let options = options_from(options_json)?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        let outcome = PreferenceDecisionArtifact::evaluate_with_context(
            contract,
            source.inner(),
            measure,
            options,
            &ctx,
        )
        .and_then(|artifact| artifact.to_bytes(artifact_id));
        Ok(match outcome {
            Ok(bytes) => (Some(PyBytes::new(py, &bytes)), None),
            Err(error) => (None, Some(refusal(&error.to_refusal()))),
        })
    })
}

/// Replay a stored decision under the consumer's contract and draws: load it under the
/// contract's identity and the draws' digest, recompute, and require the stored result.
/// Returns the result JSON (with `measure` and `options` as stored) or a refusal. The
/// recomputation reads `cancel` at its work boundaries.
#[pyfunction]
#[pyo3(signature = (data, contract_json, source, cancel=None))]
fn replay_preference_decision(
    data: &[u8],
    contract_json: &str,
    source: &PyJointDistributionArtifact,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let contract = contract_from(contract_json)?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        let run = || -> Result<Value, String> {
            let identity = contract.identity().map_err(|e| refusal(&e.to_refusal()))?;
            let artifact = PreferenceDecisionArtifact::from_bytes(
                data,
                &identity,
                &source_digest(source.inner()),
            )
            .map_err(|e| refusal(&e.to_refusal()))?;
            artifact
                .replay_with_context(source.inner(), &ctx)
                .map_err(|e| refusal(&e.to_refusal()))?;
            let mut value = evaluation_json(artifact.evaluation(), artifact.contract());
            value["measure"] = artifact.measure().map_or(Value::Null, measure_value);
            value["options"] = options_value(artifact.options());
            Ok(value)
        };
        Ok(reply(run()))
    })
}

// ---------------------------------------------------------------------------
// Several estimated outcomes as one joint law.
// ---------------------------------------------------------------------------

/// A joint law or the refusal JSON.
type LawReply = (Option<PyJointDistributionArtifact>, Option<String>);

fn law(outcome: Result<DistributionArtifact, String>) -> LawReply {
    match outcome {
        Ok(artifact) => (Some(PyJointDistributionArtifact { artifact, source_claim: None }), None),
        Err(refused) => (None, Some(refused)),
    }
}

/// The aligned joint output law of a sealed propagation chain: its record is resealed (every
/// stage recomputed under the declared dependence) and its joint outputs read as one law.
#[pyfunction]
#[pyo3(signature = (record_json, cancel))]
fn preference_chain_joint_law(
    record_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<LawReply> {
    crate::catch_ffi(|| {
        let record: PropagationChainRecordWire = parse(record_json, "propagation chain record")?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        let outcome = PropagationChain::seal(record, &ctx)
            .map_err(|e| crate::chain_decision_api::chain_refusal_json(&e, "propagation_chain"))
            .and_then(|chain| chain_joint_law(&chain).map_err(|e| refusal(&e)));
        Ok(law(outcome))
    })
}

/// One joint law from laws whose rows already pair (the same source, provider, generator,
/// snapshot, contract and number of draws), or the refusal naming why they do not.
#[pyfunction]
fn preference_joint_law_of(
    sources: Vec<PyRef<'_, PyJointDistributionArtifact>>,
) -> PyResult<LawReply> {
    crate::catch_ffi(|| {
        let laws: Vec<_> = sources.iter().map(|source| source.inner()).collect();
        Ok(law(joint_law_of(&laws).map_err(|e| refusal(&e))))
    })
}

/// The experiment the `decision_evidence` wire declares, rebuilt and checked against its
/// compiled source exactly as the decision-evidence bridge does, or that bridge's refusal JSON.
fn experiment(
    evidence_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Result<crate::decision_evidence_api::ParsedEvidence, String>> {
    let wire: Value = parse(evidence_json, "experiment evidence")?;
    let ctx = crate::joint_panel_api::context(seed, threads, max_bytes, cancel)?;
    Ok(crate::decision_evidence_api::evidence_with_context(&wire, &ctx)?.map_err(|refused| {
        crate::bridge::compact_refusal_json(
            refused.registered_code(),
            "decision_evidence",
            refused.reason_code(),
            &refused.to_string(),
            Some(refused.next_attempt()),
        )
    }))
}

/// The arm means of one experiment's outcomes as a mean source (`{coordinates, means,
/// provider_id, snapshot_id, causal_contract_id, rng_id}`), or a refusal. Needs exposure
/// declared free of between-unit interference.
#[pyfunction]
#[pyo3(signature = (evidence_json, *, seed=0, threads=1, max_bytes=None, cancel=None))]
fn preference_experiment_source(
    evidence_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let evidence = match experiment(evidence_json, seed, threads, max_bytes, cancel)? {
            Ok(evidence) => evidence,
            Err(refused) => return Ok((None, Some(refused))),
        };
        Ok(reply(experiment_mean_source(&evidence).map_err(|e| refusal(&e)).map(|source| {
            json!({
                "coordinates": source
                    .coordinates
                    .iter()
                    .map(ScientificQuantityWire::from)
                    .collect::<Vec<_>>(),
                "means": source.means,
                "provider_id": source.provider_id,
                "snapshot_id": source.snapshot_id,
                "causal_contract_id": source.causal_contract_id,
                "rng_id": source.rng_id,
            })
        })))
    })
}

/// The coordinate of one outcome of an experiment in one arm (functional `mean`), as the
/// quantity wire, or a refusal.
#[pyfunction]
#[pyo3(signature = (evidence_json, outcome, treated, *, seed=0, threads=1, max_bytes=None, cancel=None))]
fn preference_experiment_quantity(
    evidence_json: &str,
    outcome: &str,
    treated: bool,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let evidence = match experiment(evidence_json, seed, threads, max_bytes, cancel)? {
            Ok(evidence) => evidence,
            Err(refused) => return Ok((None, Some(refused))),
        };
        Ok(reply(
            experiment_arm_quantity(&evidence, outcome, treated)
                .map(|q| json!(ScientificQuantityWire::from(&q)))
                .map_err(|e| refusal(&e)),
        ))
    })
}

// ---------------------------------------------------------------------------
// The compiled route: bind, inspect the plan, execute.
// ---------------------------------------------------------------------------

/// A decision across preferences compiled with the analyses that estimate its quantities:
/// the plan shows the decision before anything is fitted, and executing it runs every
/// compiled plan once.
#[pyclass(name = "_CompiledPreferenceDecision", frozen)]
pub(crate) struct PyCompiledPreferenceDecision {
    inner: Arc<CompiledPreferenceDecision>,
}

/// `(compiled handle, refusal JSON)`: exactly one is set.
pub(crate) type CompiledReply = (Option<PyCompiledPreferenceDecision>, Option<String>);

/// `(result JSON, refusal JSON, aligned law the decision read)`.
type ExecutedReply = (Option<String>, Option<String>, Option<PyJointDistributionArtifact>);

/// A preference contract from its declaration JSON, for the bridges that compile it.
///
/// # Errors
/// A malformed declaration.
pub(crate) fn preference_contract_from(json: &str) -> PyResult<PreferenceContract> {
    contract_from(json)
}

/// A compile outcome as the handle or the refusal JSON (see [`route_error`]).
///
/// # Errors
/// A compiler or design error that is not a refusal, raised as its Python exception.
pub(crate) fn compiled_reply(
    outcome: Result<CompiledPreferenceDecision, PreferenceRouteError>,
) -> PyResult<CompiledReply> {
    match outcome {
        Ok(compiled) => {
            Ok((Some(PyCompiledPreferenceDecision { inner: Arc::new(compiled) }), None))
        }
        Err(error) => route_error(error, "compile").map(|refused| (None, Some(refused))),
    }
}

/// Stage of a compiled decision's cancellation refusal.
type RouteStage = &'static str;

/// The refusal JSON of a route error: a binding, source or decision refusal as it is, and a
/// cancellation of the compile or the execution as `preference_compile.cancelled`.
///
/// # Errors
/// Any other error of the analysis behind the decision raises its own Python exception, as
/// every other route does, carrying the error's registered reason code when it has one.
fn route_error(error: PreferenceRouteError, stage: RouteStage) -> PyResult<String> {
    let code = error.code().map(str::to_owned);
    match error {
        PreferenceRouteError::Refused(refused) => Ok(refusal(&refused)),
        PreferenceRouteError::Compiler(error) => compiler_refusal(*error, stage),
        PreferenceRouteError::Design(error) => design_refusal(*error, code, stage),
    }
}

fn cancelled_refusal(stage: RouteStage, message: &str) -> String {
    crate::bridge::RefusalJson::new(
        antecedent::reason::reason_code!("cancelled_no_claim"),
        stage,
        antecedent::reason::reason_details::preference_compile::CANCELLED,
        message,
    )
    .remedy(Some("rerun without cancellation; the stop is not a verdict on the analysis"))
    .json()
}

fn compiler_refusal(error: antecedent::CausalError, stage: RouteStage) -> PyResult<String> {
    if matches!(error.peeled(), antecedent::CausalError::Cancelled { .. }) {
        return Ok(cancelled_refusal(stage, &error.to_string()));
    }
    Err(crate::py_err(error))
}

/// A compiled design's error: the compiler's own error as the compiler route reads it, a
/// cancellation as `preference_compile.cancelled`, and every other error as the design route
/// raises it, carrying the registered reason code the compiled route reports for it (`code`).
fn design_refusal(
    error: antecedent::analysis::joint_panel::JointPanelError,
    code: Option<String>,
    stage: RouteStage,
) -> PyResult<String> {
    use antecedent::analysis::joint_panel::JointPanelError;
    use antecedent::estimate::joint_panel::PanelError;
    match error {
        JointPanelError::Compiler(error) => compiler_refusal(error, stage),
        JointPanelError::Engine(PanelError::Cancelled) => {
            Ok(cancelled_refusal(stage, &error.to_string()))
        }
        other => {
            let raised = crate::joint_panel_api::joint_panel_error(other);
            Err(match code {
                Some(code) => crate::errors::with_reason_code(raised, &code),
                None => raised,
            })
        }
    }
}

const fn representation_name(representation: SourceRepresentation) -> &'static str {
    match representation {
        SourceRepresentation::Mean => "mean",
        SourceRepresentation::MeanAndCovariance => "mean_and_covariance",
        SourceRepresentation::Cdf => "cdf",
        SourceRepresentation::QuantileFunction => "quantile_function",
        SourceRepresentation::MarginalDraws => "marginal_draws",
        SourceRepresentation::JointDraws => "joint_draws",
    }
}

fn dependence_json(dependence: OutcomeDependence) -> Value {
    match dependence {
        OutcomeDependence::Undeclared => json!({"kind": "undeclared"}),
        OutcomeDependence::DistinctEvidence { draws, seed } => {
            json!({"kind": "distinct_evidence", "draws": draws, "seed": seed})
        }
        OutcomeDependence::SharedEvidence { draws, seed } => {
            json!({"kind": "shared_evidence", "draws": draws, "seed": seed})
        }
    }
}

fn joint_law_json(joint: &PreferenceJointLaw) -> Value {
    let fact = joint.fact();
    json!({
        "dependence": fact.dependence(),
        "draws": fact.draws(),
        "seed": fact.seed(),
        "reading": fact.reading(),
        "label": fact.label(),
        "declaration": fact.declaration(),
        "cross_blocks": fact.cross_blocks(),
        "pairs": fact
            .pairs()
            .iter()
            .map(|(first, second, basis)| {
                json!({"first": first, "second": second, "basis": &**basis})
            })
            .collect::<Vec<_>>(),
        "inputs": fact
            .inputs()
            .iter()
            .map(|input| json!({
                "source": input.source(),
                "stage": input.stage(),
                "study": input.study(),
                "quantity": input.quantity(),
                "factor": input.factor(),
            }))
            .collect::<Vec<_>>(),
        "stages": joint
            .plan()
            .stages
            .iter()
            .map(|stage| json!({
                "id": stage.id,
                "kind": stage.kind.as_str(),
                "representation": stage.representation.as_str(),
                "sources": stage.sources,
            }))
            .collect::<Vec<_>>(),
    })
}

fn structure_json(structure: &CompiledStructure) -> Value {
    let identification = match structure.identification() {
        StructureIdentification::Identified(status) => {
            json!({"kind": "identified", "status": status.as_str()})
        }
        StructureIdentification::IdentifiedAtPreparation => {
            json!({"kind": "identified_at_preparation"})
        }
        StructureIdentification::NotIdentified(reason) => {
            json!({"kind": "not_identified", "reason": reason})
        }
    };
    let member = structure.member();
    json!({
        "key": structure.key(),
        "class_key": structure.class_key(),
        "fingerprint": structure.fingerprint(),
        "multiplicity": member.map(|m| m.multiplicity),
        "sampled": member.is_some_and(|m| m.sampling.is_some()),
        "weight": structure.weight(),
        "source": structure.source(),
        "identification": identification,
        "plan_identity": structure.plan_view().map(|view| view.identity().to_hex()),
    })
}

/// A class size as JSON: exact counts can exceed a JSON number, so they are decimal text.
fn class_size_json(size: ClassSize) -> Value {
    match size {
        ClassSize::Exact(count) => json!({"kind": "exact", "count": count.to_string()}),
        ClassSize::AtLeast(count) => json!({"kind": "at_least", "count": count.to_string()}),
    }
}

fn class_sampling_json(sampling: &ClassSampling) -> Value {
    json!({
        "class_key": sampling.class_key,
        "scheme": sampling.scheme.as_str(),
        "class_size": class_size_json(sampling.class_size),
        "drawn": sampling.drawn,
        "distinct": sampling.distinct,
        "seed": sampling.seed,
        "attempts": sampling.attempts,
        "assumption": sampling.assumption,
    })
}

fn mass_estimate_json(estimate: MassEstimate) -> Value {
    json!({"estimate": estimate.estimate, "lower": estimate.lower, "upper": estimate.upper})
}

fn sampled_verdict_json(verdict: &SampledStructuralVerdict) -> Value {
    match verdict {
        SampledStructuralVerdict::OnSampledMeasure {
            best_on_sampled_measure: Some(action),
            ..
        } => json!({"kind": "best_on_sampled_measure", "action": action}),
        SampledStructuralVerdict::OnSampledMeasure { candidates, .. } => {
            json!({"kind": "undetermined_on_sampled_measure", "actions": candidates})
        }
        SampledStructuralVerdict::PerSampledMember { best_on_every_sampled_member, other_best } => {
            let kind = if best_on_every_sampled_member.is_some() {
                "best_on_every_sampled_member"
            } else {
                "per_sampled_member"
            };
            let bounds: Vec<Value> = other_best
                .iter()
                .map(|bound| {
                    json!({
                        "class_key": bound.class_key,
                        "disagreeing": bound.disagreeing,
                        "drawn": bound.drawn,
                        "upper": bound.upper,
                    })
                })
                .collect();
            json!({"kind": kind, "action": best_on_every_sampled_member, "other_best": bounds})
        }
        SampledStructuralVerdict::ReportOnly => json!({"kind": "report_only"}),
    }
}

/// A decision on sampled class members: the structural body with the verdict relabelled.
/// The joint table on the sampled measure keeps its ranges, but no action in it is flagged
/// best everywhere and its verdict is the sampled one.
fn sampled_json(evaluation: &SampledStructuralEvaluation, contract: &PreferenceContract) -> Value {
    let mut body = structural_json(&evaluation.evaluation, contract);
    let verdict = sampled_verdict_json(&evaluation.verdict);
    if let Some(combined) = body.get_mut("combined").and_then(Value::as_object_mut) {
        if let Some(actions) = combined.get_mut("actions").and_then(Value::as_array_mut) {
            for action in actions.iter_mut().filter_map(Value::as_object_mut) {
                action.insert("necessarily_best".into(), Value::Null);
                action.insert("uniquely_best".into(), Value::Null);
            }
        }
        combined.insert("draw_share_best_everywhere".into(), Value::Null);
        combined.insert("verdict".into(), verdict.clone());
        combined.insert("statement".into(), json!(evaluation.statement()));
    }
    body["structural_verdict"] = verdict;
    body["statement"] = json!(evaluation.statement());
    body["sampled"] = json!({
        "level": evaluation.level,
        "sampling": evaluation.sampling.iter().map(class_sampling_json).collect::<Vec<_>>(),
        "masses": evaluation
            .masses
            .iter()
            .map(|mass| json!({
                "class_key": mass.class_key,
                "class_mass": mass.class_mass,
                "drawn": mass.drawn,
                "identified": mass_estimate_json(mass.identified),
                "unidentified": mass_estimate_json(mass.unidentified),
                "unevaluated": mass_estimate_json(mass.unevaluated),
            }))
            .collect::<Vec<_>>(),
        "margins": evaluation
            .margins
            .iter()
            .map(|margin| json!({
                "first": margin.first,
                "second": margin.second,
                "vertex": margin.vertex,
                "margin": margin.margin,
                "standard_error": margin.standard_error,
            }))
            .collect::<Vec<_>>(),
    });
    body
}

/// What the compile states, before anything is fitted.
fn describe_json(compiled: &CompiledPreferenceDecision) -> Value {
    let demand = compiled.demand();
    let route = match demand.route.route {
        PreferenceRoute::Exact => "exact",
        PreferenceRoute::Certified => "certified",
    };
    json!({
        "bound_identity": compiled.bound_identity(),
        "contract_identity": compiled.contract().identity().ok(),
        "causal": compiled.causal_bindings(),
        "dependence": dependence_json(compiled.dependence()),
        "cross_study_covariance": compiled
            .cross_study_covariance()
            .iter()
            .map(|block| json!({
                "first": block.first,
                "second": block.second,
                "covariance": block.covariance,
            }))
            .collect::<Vec<_>>(),
        "demand": {
            "route": route,
            "route_reason": json!(demand.route.reason),
            "representation": representation_name(demand.representation),
            "accepted": demand
                .requirement
                .any_of
                .iter()
                .map(|r| representation_name(*r))
                .collect::<Vec<_>>(),
            "carrier": demand.carrier.map(CarrierDemand::as_str),
            "quantities": demand
                .quantities
                .iter()
                .map(|q| json!({
                    "identity": q.quantity.canonical_identity(),
                    "quantity": ScientificQuantityWire::from(&q.quantity),
                    "source": q.source,
                    "read_by": q.read_by,
                }))
                .collect::<Vec<_>>(),
        },
        "structures": compiled.structures().iter().map(structure_json).collect::<Vec<_>>(),
        "class_sampling": compiled
            .class_sampling()
            .iter()
            .map(|sampling| class_sampling_json(sampling))
            .collect::<Vec<_>>(),
        "unidentified_mass": compiled.unidentified_mass(),
        "joint_law": compiled.joint_law().map(joint_law_json),
        "plans": compiled.plan_views().iter().map(PlanView::to_json).collect::<Vec<_>>(),
    })
}

fn executed_json(
    compiled: &CompiledPreferenceDecision,
    executed: &PreferenceDecisionExecution,
) -> Value {
    let contract = compiled.contract();
    let mut body = match &executed.evaluation {
        CompiledPreferenceEvaluation::Single(evaluation) => evaluation_json(evaluation, contract),
        CompiledPreferenceEvaluation::Structures(evaluation) => {
            structural_json(evaluation, contract)
        }
        CompiledPreferenceEvaluation::Sampled(evaluation) => sampled_json(evaluation, contract),
    };
    let receipt = &executed.receipt;
    body["compiled"] = json!({
        "bound_identity": receipt.bound_identity,
        "contract_identity": receipt.contract_identity,
        "causal": receipt.causal,
        "programs": receipt.programs.iter().map(SemanticDigest::to_hex).collect::<Vec<_>>(),
        // Aligned with `programs`; `null` where no executed result reports an identity.
        "executed": receipt
            .executed
            .iter()
            .map(|digest| digest.as_ref().map(SemanticDigest::to_hex))
            .collect::<Vec<_>>(),
        "measure": receipt.measure,
        "options": receipt.options,
        "mixture": executed.mixture.as_ref().map(|m| json!({
            "weight_basis": m.weight_basis.as_str(),
            "identified_mass": m.identified_mass,
            "unidentified_mass": m.unidentified_mass,
            "unevaluable_mass": m.unevaluable_mass,
        })),
    });
    body
}

#[pymethods]
impl PyCompiledPreferenceDecision {
    /// What the compile states (the bound identity, the demand, every structure with its
    /// identification and plan identity, the masses, the declared joint law and every plan
    /// view), as JSON. Nothing is fitted.
    fn describe(&self) -> String {
        describe_json(&self.inner).to_string()
    }

    /// Identity of the preference contract bound to the analyses' causal contracts.
    fn bound_identity(&self) -> String {
        self.inner.bound_identity().to_owned()
    }

    /// Run every compiled plan once and decide across the preference set; the decision reads
    /// `cancel` at its work boundaries.
    #[pyo3(signature = (measure_json, options_json, *, seed=0, memory_bytes=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        measure_json: Option<&str>,
        options_json: &str,
        seed: u64,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<ExecutedReply> {
        let measure = measure_opt(measure_json)?;
        let options = options_from(options_json)?;
        let ctx = crate::transport_common::execution_context(seed, memory_bytes, cancel);
        let compiled = Arc::clone(&self.inner);
        crate::detach_catch(py, move || match compiled.execute(measure.as_ref(), options, &ctx) {
            Ok(executed) => {
                let law = match &executed.source {
                    Some(NativeDecisionSource::JointLaw(law)) => {
                        Some(PyJointDistributionArtifact {
                            artifact: law.as_ref().clone(),
                            source_claim: None,
                        })
                    }
                    _ => None,
                };
                Ok((Some(executed_json(&compiled, &executed).to_string()), None, law))
            }
            Err(error) => route_error(error, "execute").map(|refused| (None, Some(refused), None)),
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CrossWire {
    first: usize,
    second: usize,
    covariance: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DependenceWire {
    kind: String,
    #[serde(default)]
    draws: usize,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    cross: Vec<CrossWire>,
}

fn dependence_from(json: &str) -> PyResult<(OutcomeDependence, Vec<CrossStudyCovariance>)> {
    let wire: DependenceWire = parse(json, "dependence")?;
    let dependence = match wire.kind.as_str() {
        "undeclared" => OutcomeDependence::Undeclared,
        "distinct_evidence" => {
            OutcomeDependence::DistinctEvidence { draws: wire.draws, seed: wire.seed }
        }
        "shared_evidence" => {
            OutcomeDependence::SharedEvidence { draws: wire.draws, seed: wire.seed }
        }
        other => {
            return Err(crate::value_err(format!(
                "dependence is undeclared, distinct_evidence or shared_evidence, not {other}"
            )));
        }
    };
    let cross = wire
        .cross
        .into_iter()
        .map(|block| CrossStudyCovariance {
            first: block.first,
            second: block.second,
            covariance: block.covariance,
        })
        .collect();
    Ok((dependence, cross))
}

/// Compile a decision across preferences with one prepared study: bind the contract to the
/// study's query, lower the decision into the study's plan and check identification under
/// every structure it carries. Nothing is fitted.
#[pyfunction]
#[pyo3(signature = (study, contract_json, *, seed=0, memory_bytes=None, cancel=None))]
fn preference_compile_for_study(
    py: Python<'_>,
    study: PyRef<'_, crate::prepared_api::PyPreparedAnalysis>,
    contract_json: &str,
    seed: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<CompiledReply> {
    let contract = contract_from(contract_json)?;
    let ctx = crate::transport_common::execution_context(seed, memory_bytes, cancel);
    let prepared = Arc::clone(&study.inner);
    drop(study);
    let outcome = crate::detach_catch(py, move || {
        Ok(prepared.compile_for_preference_decision(&contract, &ctx))
    })?;
    compiled_reply(outcome)
}

/// Compile a decision across preferences whose quantities are the outcomes of several
/// prepared studies, one outcome per study, under the declared dependence (`{kind, draws,
/// seed, cross}`).
#[pyfunction]
#[pyo3(signature = (studies, contract_json, dependence_json, *, seed=0, memory_bytes=None, cancel=None))]
fn preference_compile_across_studies(
    py: Python<'_>,
    studies: Vec<PyRef<'_, crate::prepared_api::PyPreparedAnalysis>>,
    contract_json: &str,
    dependence_json: &str,
    seed: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<CompiledReply> {
    let contract = contract_from(contract_json)?;
    let (dependence, cross) = dependence_from(dependence_json)?;
    let ctx = crate::transport_common::execution_context(seed, memory_bytes, cancel);
    let prepared: Vec<_> = studies.iter().map(|study| Arc::clone(&study.inner)).collect();
    drop(studies);
    let outcome = crate::detach_catch(py, move || {
        let handles: Vec<&antecedent::PreparedStudy> = prepared.iter().map(AsRef::as_ref).collect();
        Ok(compile_preference_decision_across_prepared_studies(
            &handles,
            &contract,
            (dependence, &cross),
            &ctx,
        ))
    })?;
    compiled_reply(outcome)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    learning::register(m)?;
    m.add_class::<PyCompiledPreferenceDecision>()?;
    m.add_function(wrap_pyfunction!(preference_compile_for_study, m)?)?;
    m.add_function(wrap_pyfunction!(preference_compile_across_studies, m)?)?;
    m.add_function(wrap_pyfunction!(preference_experiment_source, m)?)?;
    m.add_function(wrap_pyfunction!(preference_experiment_quantity, m)?)?;
    m.add_function(wrap_pyfunction!(preference_chain_joint_law, m)?)?;
    m.add_function(wrap_pyfunction!(preference_joint_law_of, m)?)?;
    m.add_function(wrap_pyfunction!(preference_model_describe, m)?)?;
    m.add_function(wrap_pyfunction!(preference_contract_describe, m)?)?;
    m.add_function(wrap_pyfunction!(decide_preferences_joint, m)?)?;
    m.add_function(wrap_pyfunction!(decide_preferences_means, m)?)?;
    m.add_function(wrap_pyfunction!(decide_preferences_structures, m)?)?;
    m.add_function(wrap_pyfunction!(export_preference_decision, m)?)?;
    m.add_function(wrap_pyfunction!(replay_preference_decision, m)?)?;
    m.add_function(wrap_pyfunction!(table_content_identity, m)?)
}
