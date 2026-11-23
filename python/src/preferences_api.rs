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

use antecedent::analysis::inverse_query::{AtomEvidence, DistributionArtifact, MeanSource};
use antecedent::analysis::native_claims::NativeDecisionSource;
use antecedent::analysis::preference_decision::{
    AcceptedPreferences, AcceptedPreferencesArtifact, ActionInput, ActionKind, Attribute,
    BeliefSource, CertifiedPreferenceResult, CertifiedVerdict, ChoiceModelEstimate,
    ChoiceModelOptions, ChoiceModelResult, ChoiceOption, ChoiceRecord, ChoiceRecordSet,
    ConflictKind, CovarianceKind, DecisionCriterion, EstimateOutcome, HardConstraint,
    LearnedPreferenceSet, LearningOptions, LinearRelation, MeasureOrigin, OptionBelief,
    PendingReview, Pooling, PreferenceAction, PreferenceBasis, PreferenceContract,
    PreferenceDecisionArtifact, PreferenceDecisionResult, PreferenceEvaluation,
    PreferenceEvaluationOptions, PreferenceExpr, PreferenceMeasure, PreferenceParameter,
    PreferencePolicy, PreferenceRoute, PreferenceSet, PreferenceVerdict, RegionStatus,
    RelationOrigin, RelationSense, SetLearningOptions, SetLearningResult, SetOutcome,
    StructuralCertifiedResult, StructuralPolicy, StructuralPreferenceEvaluation,
    StructuralPreferenceResult, StructuralPreferenceVerdict, ValueModel, analysis_options,
    chain_joint_law, decide_across_structures_with_context,
    decide_preferences_on_means_with_context, decide_preferences_with_context,
    estimate_accepted_with_context, estimate_choice_model_with_context, experiment_arm_quantity,
    experiment_mean_source, joint_law_of, learn_compatible_set_accepted_with_context,
    learn_compatible_set_with_context, preference_route, source_requirement,
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
use antecedent::scientific::ScientificQuantity;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::bridge::{check_declared_size, external_refusal_json};
use crate::distribution_api::PyJointDistributionArtifact;

/// Largest JSON declaration accepted (records may carry joint draws of every option).
const MAX_PREFERENCE_JSON_BYTES: usize = 64 * 1024 * 1024;

/// Most draw bytes the structures of one decision may copy between them.
const MAX_CLAIM_DRAW_BYTES: usize = 512 * 1024 * 1024;

/// `(result JSON, refusal JSON)`: exactly one is set.
type Reply = (Option<String>, Option<String>);

/// An accepted-preferences handle or the refusal JSON.
type HandleReply = (Option<PyAcceptedPreferences>, Option<String>);

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
    value: ExprWire,
    utility_units: String,
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
        origin: match w.origin {
            RelationOriginWire::Declared => RelationOrigin::Declared,
            RelationOriginWire::ShapeRule => RelationOrigin::ShapeRule,
            RelationOriginWire::Choice { record_id, rejected_option } => {
                RelationOrigin::Choice { record_id, rejected_option }
            }
        },
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
        origin: match &r.origin {
            RelationOrigin::Declared => RelationOriginWire::Declared,
            RelationOrigin::ShapeRule => RelationOriginWire::ShapeRule,
            RelationOrigin::Choice { record_id, rejected_option } => RelationOriginWire::Choice {
                record_id: record_id.clone(),
                rejected_option: rejected_option.clone(),
            },
        },
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
    tie_tolerance: Option<f64>,
    vertex_cap: Option<usize>,
    max_boxes: Option<usize>,
    min_width_fraction: Option<f64>,
}

fn options_from(json: &str) -> PyResult<PreferenceEvaluationOptions> {
    let wire: OptionsWire = parse(json, "decision options")?;
    let mut options = PreferenceEvaluationOptions::default();
    if let Some(tolerance) = wire.tie_tolerance {
        options.exact.tie_tolerance = tolerance;
        options.certified.tie_tolerance = tolerance;
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
// Recorded choices.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BeliefWire {
    Means { values: Vec<f64> },
    Draws { rows: Vec<f64>, weights: Option<Vec<f64>> },
    Unavailable { constraint: String },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SourceWire {
    Forecast { source_id: String },
    AnalysisAsOf { result_identity: String, as_of: String },
    Realized { assumption: String },
    Stated { elicitation: String, assumption: String },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OptionWire {
    id: String,
    belief: BeliefWire,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordWire {
    id: String,
    decider: String,
    options: Vec<OptionWire>,
    chosen: String,
    source: SourceWire,
    observation_rows: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PoolingWire {
    PerDecider,
    Pooled { assumption: String },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordsWire {
    attributes: Vec<AttributeWire>,
    population: String,
    records: Vec<RecordWire>,
    pooling: PoolingWire,
}

fn belief_from(w: BeliefWire) -> OptionBelief {
    match w {
        BeliefWire::Means { values } => OptionBelief::Means(values),
        BeliefWire::Draws { rows, weights } => OptionBelief::Draws { rows, weights },
        BeliefWire::Unavailable { constraint } => {
            OptionBelief::Unavailable { constraint_id: constraint }
        }
    }
}

fn belief_to(b: &OptionBelief) -> BeliefWire {
    match b {
        OptionBelief::Means(values) => BeliefWire::Means { values: values.clone() },
        OptionBelief::Draws { rows, weights } => {
            BeliefWire::Draws { rows: rows.clone(), weights: weights.clone() }
        }
        OptionBelief::Unavailable { constraint_id } => {
            BeliefWire::Unavailable { constraint: constraint_id.clone() }
        }
    }
}

fn records_from(json: &str) -> PyResult<ChoiceRecordSet> {
    let wire: RecordsWire = parse(json, "choice records")?;
    Ok(ChoiceRecordSet {
        attributes: wire.attributes.into_iter().map(attribute_from).collect(),
        population: wire.population,
        records: wire
            .records
            .into_iter()
            .map(|r| ChoiceRecord {
                id: r.id,
                decider: r.decider,
                options: r
                    .options
                    .into_iter()
                    .map(|o| ChoiceOption { id: o.id, belief: belief_from(o.belief) })
                    .collect(),
                chosen: r.chosen,
                belief_source: match r.source {
                    SourceWire::Forecast { source_id } => {
                        BeliefSource::RecordedForecast { source_id }
                    }
                    SourceWire::AnalysisAsOf { result_identity, as_of } => {
                        BeliefSource::AnalysisAsOf { result_identity, as_of }
                    }
                    SourceWire::Realized { assumption } => {
                        BeliefSource::RealizedOutcome { assumption_id: assumption }
                    }
                    SourceWire::Stated { elicitation, assumption } => BeliefSource::Stated {
                        elicitation_id: elicitation,
                        assumption_id: assumption,
                    },
                },
                observation_rows: r.observation_rows,
            })
            .collect(),
        pooling: match wire.pooling {
            PoolingWire::PerDecider => Pooling::PerDecider,
            PoolingWire::Pooled { assumption } => Pooling::Pooled { assumption_id: assumption },
        },
    })
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct SetOptionsWire {
    tolerance: Option<f64>,
    conflict_search_size: Option<usize>,
}

fn set_options_from(json: &str) -> PyResult<SetLearningOptions> {
    let wire: SetOptionsWire = parse(json, "compatible-set options")?;
    let mut options = SetLearningOptions::default();
    if let Some(tolerance) = wire.tolerance {
        options.tolerance = tolerance;
    }
    if let Some(size) = wire.conflict_search_size {
        options.conflict_search_size = size;
    }
    Ok(options)
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ChoiceOptionsWire {
    level: Option<f64>,
    n_draws: Option<usize>,
    seed: Option<u64>,
}

fn choice_options_from(json: &str) -> PyResult<ChoiceModelOptions> {
    let wire: ChoiceOptionsWire = parse(json, "choice-model options")?;
    let mut options = ChoiceModelOptions::default();
    if let Some(level) = wire.level {
        options.level = level;
    }
    if let Some(n_draws) = wire.n_draws {
        options.n_draws = n_draws;
    }
    if let Some(seed) = wire.seed {
        options.seed = seed;
    }
    Ok(options)
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

fn learned_json(l: &LearnedPreferenceSet) -> Value {
    let ranges: Vec<Value> = l
        .ranges
        .iter()
        .map(|r| {
            json!({
                "parameter_id": r.parameter_id,
                "declared": r.declared,
                "learned": r.learned,
                "varied_in_records": r.varied_in_records,
                "narrowed": r.narrowed,
            })
        })
        .collect();
    let binding: Vec<Value> = l
        .binding
        .iter()
        .map(|b| json!({"record_id": b.record_id, "rejected_option": b.rejected_option}))
        .collect();
    json!({
        "model_identity": l.model_identity,
        "records_digest": l.records_digest,
        "decider": l.decider,
        "parameters": l.set.parameters.iter().map(parameter_to).collect::<Vec<_>>(),
        "relations": l.set.relations.iter().map(relation_to).collect::<Vec<_>>(),
        "ranges": ranges,
        "binding": binding,
        "tolerance": l.tolerance,
        "n_records": l.n_records,
        "assumptions": l.assumptions,
    })
}

fn outcome_json(outcome: &SetOutcome) -> Value {
    match outcome {
        SetOutcome::Learned(learned) => {
            json!({"kind": "learned", "learned": learned_json(learned)})
        }
        SetOutcome::Contradictory(conflict) => json!({
            "kind": "contradictory",
            "conflict": {
                "records": conflict.records,
                "kind": match conflict.kind {
                    ConflictKind::Smallest => "smallest",
                    ConflictKind::Irreducible => "irreducible",
                },
                "tolerance_needed": conflict.tolerance_needed,
            },
        }),
    }
}

fn set_result_json(result: &SetLearningResult) -> Value {
    json!({
        "pooled": result.pooled.as_ref().map(outcome_json),
        "per_decider": result
            .per_decider
            .iter()
            .map(|d| json!({
                "decider": d.decider,
                "n_records": d.n_records,
                "outcome": outcome_json(&d.outcome),
            }))
            .collect::<Vec<_>>(),
        "overlap": result.overlap.as_ref().map(|o| json!({
            "common": outcome_json(&o.common),
            "disagreeing": o.disagreeing,
        })),
    })
}

fn estimate_json(e: &ChoiceModelEstimate) -> Value {
    let parameters: Vec<Value> = e
        .parameters
        .iter()
        .map(|p| {
            json!({
                "parameter_id": p.parameter_id,
                "estimate": p.estimate,
                "standard_error": p.standard_error,
                "interval": p.interval,
                "declared": p.declared,
                "inside_declared": p.inside_declared,
            })
        })
        .collect();
    json!({
        "identity": e.identity,
        "model_identity": e.model_identity,
        "records_digest": e.records_digest,
        "decider": e.decider,
        "consistency": e.consistency,
        "consistency_standard_error": e.consistency_standard_error,
        "parameters": parameters,
        "not_learnable": e.not_learnable,
        "log_likelihood": e.log_likelihood,
        "null_log_likelihood": e.null_log_likelihood,
        "n_records": e.n_records,
        "n_deciders": e.n_deciders,
        "covariance_kind": match e.covariance_kind {
            CovarianceKind::Model => "model",
            CovarianceKind::ClusteredByDecider => "clustered_by_decider",
        },
        "coefficient_covariance": e.coefficient_covariance,
        "level": e.level,
        "interval_calibration": debug_kind(&format!("{:?}", e.interval_calibration)),
        "measure": measure_value(&e.measure),
        "share_outside_declared": e.share_outside_declared,
        "assumptions": e.assumptions,
    })
}

fn choice_result_json(result: &ChoiceModelResult) -> Value {
    json!({
        "pooled": result.pooled.as_deref().map(estimate_json),
        "per_decider": result
            .per_decider
            .iter()
            .map(|(decider, outcome)| match outcome {
                EstimateOutcome::Estimated(estimate) => json!({
                    "decider": decider,
                    "estimate": estimate_json(estimate),
                    "refusal": Value::Null,
                }),
                EstimateOutcome::Refused(error) => json!({
                    "decider": decider,
                    "estimate": Value::Null,
                    "refusal": refusal_value(&error.to_refusal()),
                }),
            })
            .collect::<Vec<_>>(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PendingWire {
    ParameterNotVaried { parameter_id: String },
    ParameterNotNarrowed { parameter_id: String },
    ToleranceApplied { tolerance: f64 },
    ParameterNotLearnable { parameter_id: String },
    EstimateOutsideDeclared { parameter_id: String },
    DrawsOutsideDeclared { share: f64 },
    StatedComparisonsUsed { count: usize },
}

fn pending_to(item: &PendingReview) -> Value {
    let wire = match item {
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
    };
    serde_json::to_value(wire).unwrap_or(Value::Null)
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

fn learning_json(options: Option<LearningOptions>) -> Value {
    match options {
        None => Value::Null,
        Some(LearningOptions::CompatibleSet(o)) => json!({
            "kind": "compatible_set",
            "tolerance": o.tolerance,
            "conflict_search_size": o.conflict_search_size,
        }),
        Some(LearningOptions::ChoiceEstimate(o)) => json!({
            "kind": "choice_model",
            "level": o.level,
            "n_draws": o.n_draws,
            "seed": o.seed,
        }),
    }
}

fn accepted_json(a: &AcceptedPreferences) -> Value {
    let basis = match a.basis() {
        PreferenceBasis::Declared => json!({"kind": "declared"}),
        PreferenceBasis::CompatibleSet(learned) => {
            json!({"kind": "compatible_set", "learned": learned_json(learned)})
        }
        PreferenceBasis::ChoiceEstimate(estimate) => {
            json!({"kind": "choice_estimate", "estimate": estimate_json(estimate)})
        }
    };
    json!({
        "identity": a.identity(),
        "version": a.version(),
        "declared_identity": a.declared_identity(),
        "decider": a.decider(),
        "basis": basis,
        "model": model_value(a.model()),
        "declared_model": model_value(a.declared_model()),
        "measure": a.measure().map(measure_value),
        "pending": a.pending().iter().map(pending_to).collect::<Vec<_>>(),
        "acknowledged": a.acknowledged().iter().map(pending_to).collect::<Vec<_>>(),
        "assumptions": a.assumptions(),
        "stated_comparisons": a.stated_comparisons(),
        "learning": learning_json(a.learning_options()),
    })
}

// ---------------------------------------------------------------------------
// Declarations and decisions.
// ---------------------------------------------------------------------------

/// Validate a value model; returns `{identity, shape_parameters, affine}` or a refusal.
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

/// What was expected of each action of a contract, read from a source, as the options of a
/// recorded choice: `{options: [{id, belief}]}`. Exactly one of `source` and
/// `mean_source_json` is given.
#[pyfunction]
#[pyo3(signature = (contract_json, source, mean_source_json))]
fn preference_analysis_options(
    contract_json: &str,
    source: Option<&PyJointDistributionArtifact>,
    mean_source_json: Option<&str>,
) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let contract = contract_from(contract_json)?;
        let native = match (source, mean_source_json) {
            (Some(joint), None) => NativeDecisionSource::JointLaw(Box::new(joint.inner().clone())),
            (None, Some(json)) => NativeDecisionSource::Mean(mean_source_from(json)?),
            _ => return Err(crate::value_err("pass joint draws or a mean source, not both")),
        };
        Ok(reply(
            analysis_options(&contract, &native)
                .map(|options| {
                    json!({
                        "options": options
                            .iter()
                            .map(|o| json!({"id": o.id, "belief": belief_to(&o.belief)}))
                            .collect::<Vec<_>>(),
                    })
                })
                .map_err(|e| refusal(&e)),
        ))
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
            Ok(value)
        };
        Ok(reply(run()))
    })
}

// ---------------------------------------------------------------------------
// Learning.
// ---------------------------------------------------------------------------

/// Learn the set of preference parameters compatible with recorded choices; returns
/// `{pooled, per_decider, overlap}` or a refusal. The learner reads `cancel` before every
/// linear program.
#[pyfunction]
#[pyo3(signature = (model_json, records_json, options_json, cancel=None))]
fn learn_preference_set(
    model_json: &str,
    records_json: &str,
    options_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let model = model_from(parse(model_json, "value model")?);
        let records = records_from(records_json)?;
        let options = set_options_from(options_json)?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        Ok(reply(
            learn_compatible_set_with_context(&model, &records, options, &ctx)
                .map(|result| set_result_json(&result))
                .map_err(|e| refusal(&e.to_refusal())),
        ))
    })
}

/// Estimate the preference parameters from choices read as noisy; returns
/// `{pooled, per_decider}` or a refusal. The estimator reads `cancel` before every fit,
/// Newton step and block of measure draws.
#[pyfunction]
#[pyo3(signature = (model_json, records_json, options_json, cancel=None))]
fn estimate_preference_choice_model(
    model_json: &str,
    records_json: &str,
    options_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let model = model_from(parse(model_json, "value model")?);
        let records = records_from(records_json)?;
        let options = choice_options_from(options_json)?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        Ok(reply(
            estimate_choice_model_with_context(&model, &records, options, &ctx)
                .map(|result| choice_result_json(&result))
                .map_err(|e| refusal(&e.to_refusal())),
        ))
    })
}

/// Reviewed, versioned preferences: an immutable native handle.
#[pyclass(name = "_AcceptedPreferences", frozen, skip_from_py_object)]
pub(crate) struct PyAcceptedPreferences {
    inner: AcceptedPreferences,
}

fn handle(outcome: Result<AcceptedPreferences, ExternalRefusal>) -> HandleReply {
    match outcome {
        Ok(inner) => (Some(PyAcceptedPreferences { inner }), None),
        Err(error) => (None, Some(refusal(&error))),
    }
}

#[pymethods]
impl PyAcceptedPreferences {
    #[getter]
    fn identity(&self) -> String {
        self.inner.identity().to_owned()
    }

    #[getter]
    fn version(&self) -> u32 {
        self.inner.version()
    }

    /// Everything the handle holds, as JSON: identity, version, basis, accepted model,
    /// measure, pending and acknowledged review items, assumptions and learning options.
    fn summary_json(&self) -> String {
        accepted_json(&self.inner).to_string()
    }

    /// A new handle with the review items acknowledged, or a refusal.
    fn acknowledge(&self, items_json: &str) -> PyResult<HandleReply> {
        crate::catch_ffi(|| {
            let items: Vec<PendingWire> = parse(items_json, "review items")?;
            let items: Vec<PendingReview> = items.into_iter().map(pending_from).collect();
            Ok(handle(self.inner.acknowledge(&items).map_err(|e| e.to_refusal())))
        })
    }

    /// Relearn the compatible set from `records`: version + 1, acknowledgements cleared. The
    /// learner reads `cancel` before every linear program.
    #[pyo3(signature = (records_json, decider, options_json, cancel=None))]
    fn relearn_compatible_set(
        &self,
        records_json: &str,
        decider: Option<&str>,
        options_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<HandleReply> {
        crate::catch_ffi(|| {
            let records = records_from(records_json)?;
            let options = set_options_from(options_json)?;
            let ctx = crate::transport_common::execution_context(0, None, cancel);
            Ok(handle(
                self.inner
                    .relearn_compatible_set_with_context(&records, decider, options, &ctx)
                    .map_err(|e| e.to_refusal()),
            ))
        })
    }

    /// Re-estimate from `records`: version + 1, acknowledgements cleared. The estimator reads
    /// `cancel` before every fit, Newton step and block of measure draws.
    #[pyo3(signature = (records_json, decider, options_json, cancel=None))]
    fn relearn_choice_model(
        &self,
        records_json: &str,
        decider: Option<&str>,
        options_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<HandleReply> {
        crate::catch_ffi(|| {
            let records = records_from(records_json)?;
            let options = choice_options_from(options_json)?;
            let ctx = crate::transport_common::execution_context(0, None, cancel);
            Ok(handle(
                self.inner
                    .relearn_estimate_with_context(&records, decider, options, &ctx)
                    .map_err(|e| e.to_refusal()),
            ))
        })
    }

    /// The checksummed record of the handle, or a refusal.
    fn export<'py>(
        &self,
        py: Python<'py>,
        artifact_id: &str,
    ) -> PyResult<(Option<Bound<'py, PyBytes>>, Option<String>)> {
        crate::catch_ffi(|| {
            Ok(match AcceptedPreferencesArtifact::new(self.inner.clone()).to_bytes(artifact_id) {
                Ok(bytes) => (Some(PyBytes::new(py, &bytes)), None),
                Err(error) => (None, Some(refusal(&error.to_refusal()))),
            })
        })
    }
}

/// Accept the declared preference set as it stands (version 1, nothing pending).
#[pyfunction]
fn accept_preferences_declared(model_json: &str) -> PyResult<HandleReply> {
    crate::catch_ffi(|| {
        let model = model_from(parse(model_json, "value model")?);
        Ok(handle(AcceptedPreferences::declared(model).map_err(|e| e.to_refusal())))
    })
}

/// Learn the compatible set and accept it; `decider` selects one decider's outcome, `None`
/// the pooled one. The learner reads `cancel` before every linear program.
#[pyfunction]
#[pyo3(signature = (model_json, records_json, decider, options_json, cancel=None))]
fn accept_preferences_compatible_set(
    model_json: &str,
    records_json: &str,
    decider: Option<&str>,
    options_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<HandleReply> {
    crate::catch_ffi(|| {
        let model = model_from(parse(model_json, "value model")?);
        let records = records_from(records_json)?;
        let options = set_options_from(options_json)?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        Ok(handle(
            learn_compatible_set_accepted_with_context(&model, &records, decider, options, &ctx)
                .map_err(|e| e.to_refusal()),
        ))
    })
}

/// Estimate from choices read as noisy and accept the estimate. The estimator reads `cancel`
/// before every fit, Newton step and block of measure draws.
#[pyfunction]
#[pyo3(signature = (model_json, records_json, decider, options_json, cancel=None))]
fn accept_preferences_choice_model(
    model_json: &str,
    records_json: &str,
    decider: Option<&str>,
    options_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<HandleReply> {
    crate::catch_ffi(|| {
        let model = model_from(parse(model_json, "value model")?);
        let records = records_from(records_json)?;
        let options = choice_options_from(options_json)?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        Ok(handle(
            estimate_accepted_with_context(&model, &records, decider, options, &ctx)
                .map_err(|e| e.to_refusal()),
        ))
    })
}

/// Load accepted preferences under an identity the consumer retained; with records, a
/// learned basis is relearned from them and must be reproduced, reading `cancel` as the
/// learner or the estimator does.
#[pyfunction]
#[pyo3(signature = (data, expected_identity, records_json, cancel=None))]
fn consume_accepted_preferences(
    data: &[u8],
    expected_identity: &str,
    records_json: Option<&str>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<HandleReply> {
    crate::catch_ffi(|| {
        let records = records_json.map(records_from).transpose()?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        let outcome = AcceptedPreferencesArtifact::from_bytes(data, expected_identity)
            .and_then(|artifact| match &records {
                Some(records) => artifact.replay_with_context(records, &ctx).map(|()| artifact),
                None => Ok(artifact),
            })
            .map(|artifact| artifact.accepted().clone())
            .map_err(|e| e.to_refusal());
        Ok(handle(outcome))
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAcceptedPreferences>()?;
    m.add_function(wrap_pyfunction!(preference_experiment_source, m)?)?;
    m.add_function(wrap_pyfunction!(preference_experiment_quantity, m)?)?;
    m.add_function(wrap_pyfunction!(preference_chain_joint_law, m)?)?;
    m.add_function(wrap_pyfunction!(preference_joint_law_of, m)?)?;
    m.add_function(wrap_pyfunction!(preference_model_describe, m)?)?;
    m.add_function(wrap_pyfunction!(preference_contract_describe, m)?)?;
    m.add_function(wrap_pyfunction!(decide_preferences_joint, m)?)?;
    m.add_function(wrap_pyfunction!(decide_preferences_means, m)?)?;
    m.add_function(wrap_pyfunction!(decide_preferences_structures, m)?)?;
    m.add_function(wrap_pyfunction!(preference_analysis_options, m)?)?;
    m.add_function(wrap_pyfunction!(export_preference_decision, m)?)?;
    m.add_function(wrap_pyfunction!(replay_preference_decision, m)?)?;
    m.add_function(wrap_pyfunction!(learn_preference_set, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_preference_choice_model, m)?)?;
    m.add_function(wrap_pyfunction!(accept_preferences_declared, m)?)?;
    m.add_function(wrap_pyfunction!(accept_preferences_compatible_set, m)?)?;
    m.add_function(wrap_pyfunction!(accept_preferences_choice_model, m)?)?;
    m.add_function(wrap_pyfunction!(consume_accepted_preferences, m)?)
}
