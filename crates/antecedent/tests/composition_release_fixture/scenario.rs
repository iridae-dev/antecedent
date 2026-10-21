//! The release-boundary scenario, built only from public Rust APIs.
//!
//! One uncertain measured exposure (a binary indicator with a declared confusion
//! model) enters a finite causal model; two study summaries are pooled through
//! the evidence ledger; the licensed mean of the resulting attributable excess
//! risk feeds an affine decision through a caller-declared mean source; a claim
//! graph revalidates what a revision of the evidence touches.
//!
//! Everything in this module is a producer or consumer helper. The consumer
//! half ([`consume`]) takes only bytes, retained identities and declarations:
//! it holds no producer object, and its oracle is a closed form written here,
//! not a library call.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, reason = "two test binaries share this module and each uses a subset")]
#![allow(clippy::too_many_lines, reason = "scenario builders list declarations in order")]

use std::collections::BTreeMap;
use std::ops::Range;

use antecedent::analysis::composition_verification::{
    CompositionGraph, CompositionNode, CompositionVerification, DependencyKind, NodeKind,
    ObservedEffect, consume_composition_verification_artifact, ledger_effects,
};
use antecedent::analysis::evidence_ledger::{
    ObservationId, ResultId, SourceDeclaration, SourceId, SourceKind,
};
use antecedent::analysis::interface_contract::{
    ComponentConnection, ConsumedPort, ContractOperation, InterfaceContract, ProducedPort,
    SupportDomain,
};
use antecedent::analysis::mechanism_composition::CompositionQueryWire;
use antecedent::analysis::{check_interface_composition, check_interface_connection};
use antecedent::{
    DependenceInput, EvidenceLedgerSession, MeasurementAnalysisBinding, MeasurementInput,
    MechanismCompositionRequest, MechanismInput, MechanismSource, PoolingDeclaration,
    PoolingRequest, PreparedMechanismComposition, StudyBuilder, StudySummary, SynthesisValue,
    consume_evidence_ledger_artifact, consume_measurement_model_artifact,
    consume_mechanism_composition_artifact,
};
use antecedent_contract::{
    DistributionMeaning, MechanismDeclaration, MechanismOriginKind, QuantityRole,
    ScientificQuantity,
};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_core::{CancellationToken, ExecutionContext, SemanticDigest};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{
    DecisionResult, MeanEstimate, MeanSource, Verdict, evaluate_contract_on_means,
};
use antecedent_model::compose::{Domain, TabularRow};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Declarations read from the fixture file
// ---------------------------------------------------------------------------

/// The committed fixture: declared inputs, hand-derived values and refusals.
pub fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../../conformance/composition/release_boundary/expected.json"
    ))
    .expect("expected.json parses")
}

pub fn num(value: &Value) -> f64 {
    value.as_f64().expect("a number")
}

pub fn text(value: &Value) -> String {
    value.as_str().expect("a string").to_owned()
}

pub fn rows_of(value: &Value) -> Range<u32> {
    let bound = |i: usize| u32::try_from(value[i].as_u64().expect("row bound")).unwrap();
    bound(0)..bound(1)
}

pub fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

/// One typed refusal, whatever layer raised it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Refused {
    pub stage: String,
    pub code: String,
    pub detail: String,
}

impl Refused {
    #[allow(
        clippy::needless_pass_by_value,
        reason = "callers pass borrowed and owned displayable values alike; the signature is shared by two test targets"
    )]
    pub fn of(stage: &str, code: impl ToString, detail: impl ToString) -> Self {
        Self { stage: stage.to_owned(), code: code.to_string(), detail: detail.to_string() }
    }
}

// ---------------------------------------------------------------------------
// Evidence ledger
// ---------------------------------------------------------------------------

pub fn observations(domain: &str, range: Range<u32>) -> Vec<ObservationId> {
    range.map(|i| ObservationId::from_key(domain, &i.to_le_bytes())).collect()
}

pub fn study_source(label: &str, domain: &str, range: Range<u32>) -> SourceDeclaration {
    SourceDeclaration::new(label, SourceKind::ObservationSet)
        .with_observations(observations(domain, range))
}

/// A pooled result with the numbers it carries and the ledger head it was
/// registered at.
#[derive(Clone, Debug)]
pub struct Pooled {
    pub result: ResultId,
    pub estimate: f64,
    pub variance: f64,
    pub declared_slots: u64,
    pub distinct_observations: u64,
    pub head: String,
}

pub fn pool(
    session: &mut EvidenceLedgerSession,
    label: &str,
    summaries: &[(SourceId, f64, f64)],
    covariance: Option<Vec<f64>>,
    replaces: Option<ResultId>,
) -> Result<Pooled, Refused> {
    let request = PoolingRequest {
        label: label.to_owned(),
        summaries: summaries
            .iter()
            .map(|&(source, estimate, variance)| StudySummary { source, estimate, variance })
            .collect(),
        replaces,
    };
    let declaration = match covariance {
        None => PoolingDeclaration::Independent,
        Some(covariance) => PoolingDeclaration::DeclaredCovariance { covariance },
    };
    let outcome = session
        .pool_summaries(request, declaration, &ctx())
        .map_err(|e| Refused::of("evidence", e.code, &e.detail))?;
    let SynthesisValue::FixedEffect(fit) = &outcome.value else {
        panic!("a pool of summaries is a fixed-effect pool");
    };
    Ok(Pooled {
        result: outcome.result,
        estimate: fit.estimate,
        variance: fit.variance,
        declared_slots: outcome.mass.declared_slots,
        distinct_observations: outcome.mass.distinct_observations,
        head: session.head_hex(),
    })
}

// ---------------------------------------------------------------------------
// Measurement model
// ---------------------------------------------------------------------------

pub const INDICATOR: &str = "schema:exposure_recorded";
pub const CONSTRUCT: &str = "construct:exposure";

fn scalar() -> RepresentationVersion {
    RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 }
}

pub fn measurement_model(sc: &Value) -> MeasurementModel {
    let m = &sc["measurement"];
    MeasurementModel {
        model_id: "exposure-misclassification".into(),
        route: MeasurementRoute::BinaryMisclassification,
        indicator: MeasurementVariable {
            id: INDICATOR.into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "indicator".into(),
        },
        construct: MeasurementVariable {
            id: CONSTRUCT.into(),
            kind: MeasurementKind::LatentScientific,
            units: "indicator".into(),
        },
        binding: MeasurementBinding {
            population_id: "pop".into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: num(&m["sensitivity"]),
                specificity: num(&m["specificity"]),
                sensitivity_variance: num(&m["sensitivity_variance"]),
                specificity_variance: num(&m["specificity_variance"]),
            }],
            min_youden: num(&m["min_youden"]),
        },
        representation: scalar(),
        evidence: ParameterEvidence::Declared { declarer: "release_fixture".into() },
    }
}

pub fn measurement_binding(construct: &str) -> MeasurementAnalysisBinding {
    MeasurementAnalysisBinding {
        construct_id: construct.into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        arm_id: None,
        representation: scalar(),
        representation_mapping: None,
        assumed_structure: ErrorStructure::NonDifferential,
    }
}

/// The corrected prevalence with the exported artifact and the identity a
/// consumer retains.
#[derive(Clone, Debug)]
pub struct Measured {
    pub point: f64,
    pub naive: f64,
    pub standard_error: f64,
    pub bytes: Vec<u8>,
    pub identity: String,
}

pub fn measure(sc: &Value, construct: &str) -> Result<Measured, Refused> {
    let m = &sc["measurement"];
    let input = MeasurementInput::BinaryPrevalence {
        events: m["events"].as_u64().unwrap(),
        total: m["total"].as_u64().unwrap(),
    };
    let prepared = StudyBuilder::measurement_model(
        measurement_model(sc),
        measurement_binding(construct),
        input,
        None,
        &ctx(),
    )
    .map_err(|e| Refused::of("measurement", e.code, &e.detail))?;
    let estimate =
        prepared.estimate(&ctx()).map_err(|e| Refused::of("measurement", e.code, &e.detail))?;
    let bytes = estimate
        .export("release-boundary-measurement")
        .map_err(|e| Refused::of("measurement", e.code, &e.detail))?;
    Ok(Measured {
        point: estimate.point(),
        naive: estimate.naive_point().expect("a naive point"),
        standard_error: estimate.standard_error().expect("an uncalibrated standard error"),
        bytes,
        identity: prepared.model_identity_hex(),
    })
}

// ---------------------------------------------------------------------------
// Mechanism composition (finite): X exposure, Y outcome
// ---------------------------------------------------------------------------

fn mechanism_quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: format!("label of {variable}"),
        role: if variable == "X" { QuantityRole::Treatment } else { QuantityRole::Outcome },
        units: "category".into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn table_input(
    variable: &str,
    tag: &str,
    parents: &[&str],
    rows: &[(&[&str], [f64; 2])],
) -> MechanismInput {
    let levels = vec!["0".to_owned(), "1".to_owned()];
    MechanismInput {
        declaration: MechanismDeclaration {
            output: mechanism_quantity(variable),
            parents: parents.iter().map(|p| mechanism_quantity(p)).collect(),
            conditioning_variable_ids: parents.iter().map(|p| (*p).to_owned()).collect(),
            origin: MechanismOriginKind::Fitted,
            artifact_id: format!("declared:{variable}:{tag}"),
            evidence_id: format!("evidence:{variable}:{tag}"),
            latent_causes: Vec::new(),
        },
        source: MechanismSource::DeclaredTable {
            levels: levels.clone(),
            rows: rows
                .iter()
                .map(|(config, p)| TabularRow {
                    parent_levels: parents
                        .iter()
                        .zip(config.iter())
                        .map(|(parent, level)| ((*parent).to_owned(), (*level).to_owned()))
                        .collect(),
                    probabilities: vec![("0".to_owned(), p[0]), ("1".to_owned(), p[1])],
                })
                .collect(),
        },
        output_support: Domain::Levels(levels.clone()),
        parent_domains: parents
            .iter()
            .map(|p| ((*p).to_owned(), Domain::Levels(levels.clone())))
            .collect(),
        uncertainty: None,
        latent_loadings: Vec::new(),
    }
}

/// `X ~ Bernoulli(prevalence)`; `Y | X ~ Bernoulli(baseline + effect * X)`.
pub fn compose(
    prevalence: f64,
    baseline: f64,
    effect: f64,
) -> Result<PreparedMechanismComposition, Refused> {
    let exposed = baseline + effect;
    let mechanisms = vec![
        table_input("X", &bits(prevalence), &[], &[(&[], [1.0 - prevalence, prevalence])]),
        table_input(
            "Y",
            &format!("{}:{}", bits(baseline), bits(effect)),
            &["X"],
            &[(&["0"], [1.0 - baseline, baseline]), (&["1"], [1.0 - exposed, exposed])],
        ),
    ];
    let request = MechanismCompositionRequest {
        target_population_id: "pop".into(),
        target_regime_id: "observational".into(),
        mechanisms,
        edges: vec![("X".into(), "Y".into())],
        admissions: Vec::new(),
        dependence: DependenceInput::Unknown,
        admit_declared_mechanisms: true,
    };
    StudyBuilder::mechanism_composition(request, &ctx())
        .map_err(|e| Refused::of("mechanisms", e.code(), e.details().join(",")))
}

pub fn outcome_risk(
    composition: &PreparedMechanismComposition,
    interventions: &[(&str, &str)],
) -> f64 {
    composition
        .finite_law(interventions, &ctx())
        .expect("a finite law")
        .probability(&[("Y", "1")])
        .expect("a probability")
}

/// The attributable excess risk `P(Y = 1) - P(Y = 1 | do(X = 0))`.
pub fn excess_risk(composition: &PreparedMechanismComposition) -> f64 {
    outcome_risk(composition, &[]) - outcome_risk(composition, &[("X", "0")])
}

/// The two interventional artifacts a consumer recomputes from.
pub fn export_interventional(
    composition: &PreparedMechanismComposition,
    level: &str,
) -> (Vec<u8>, String) {
    let export = composition
        .export(
            CompositionQueryWire::FiniteInterventionDistribution {
                interventions: vec![("X".into(), level.into())],
                outcome: "Y".into(),
            },
            &ctx(),
        )
        .expect("an interventional export");
    (export.bytes, export.premises_digest)
}

// ---------------------------------------------------------------------------
// Decision through a caller-declared mean source
// ---------------------------------------------------------------------------

pub fn excess_quantity(units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "excess_risk".into(),
        variable_name: "attributable excess risk".into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "attributable_excess_risk".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

/// What the consumer of a decision declares: the units the mean source claims,
/// and whether the criterion asks for more than a mean.
#[derive(Clone, Debug)]
pub struct DecisionDeclaration {
    pub benefit: f64,
    pub cost: f64,
    pub mean_units: String,
    pub needs_law: bool,
}

impl DecisionDeclaration {
    pub fn from_fixture(sc: &Value) -> Self {
        Self {
            benefit: num(&sc["decision"]["benefit_per_unit_risk"]),
            cost: num(&sc["decision"]["programme_cost"]),
            mean_units: "risk".into(),
            needs_law: false,
        }
    }
}

pub fn decision_contract(d: &DecisionDeclaration) -> DecisionContract {
    let input = || vec![excess_quantity("risk")];
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "act".into(),
                kind: ActionKind::Intervention,
                inputs: input(),
                utility: UtilityExpr::difference(
                    UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(d.benefit)),
                    UtilityExpr::Const(d.cost),
                ),
            },
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::Intervention,
                inputs: input(),
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
            },
        ],
        utility_units: "cases_averted_per_1000_net_of_cost".into(),
        criterion: if d.needs_law {
            DecisionCriterion::ThresholdProbability { threshold: 0.0 }
        } else {
            DecisionCriterion::PosteriorExpectedUtility
        },
        constraints: vec![],
        target_population: "pop".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

/// A mean the caller declares from a licensed mean, bound to the evidence state
/// (`snapshot`) and to the composition it was computed from (`plan`).
pub fn mean_source(units: &str, mean: f64, snapshot: &str, plan: &str) -> MeanSource {
    MeanSource {
        coordinates: vec![excess_quantity(units)],
        means: vec![mean],
        provider_id: "release_boundary:composed_finite_model".into(),
        snapshot_id: snapshot.into(),
        causal_contract_id: plan.into(),
        rng_id: "none:mean_grid".into(),
        estimate: MeanEstimate::Frequentist,
    }
}

pub fn decide(
    d: &DecisionDeclaration,
    mean: f64,
    snapshot: &str,
    plan: &str,
) -> Result<DecisionResult, Refused> {
    evaluate_contract_on_means(
        &decision_contract(d),
        &mean_source(&d.mean_units, mean, snapshot, plan),
    )
    .map_err(|e| {
        let refusal = e.to_refusal();
        Refused::of("decision", refusal.code, &refusal.detail)
    })
}

pub fn selected(result: &DecisionResult) -> String {
    match &result.verdict {
        Verdict::UniquelyOptimal(id) => id.clone(),
        other => panic!("expected a unique action, got {other:?}"),
    }
}

pub fn value_of(result: &DecisionResult, action: &str) -> f64 {
    result.actions.iter().find(|a| a.id == action).expect("action").expected_utility
}

// ---------------------------------------------------------------------------
// Interface contracts
// ---------------------------------------------------------------------------

/// Which declaration of the three components to break.
#[derive(Clone, Debug, Default)]
pub struct InterfaceOptions {
    /// The producer declares a real support where the consumer needs `[0, 1]`.
    pub producer_support_real: bool,
    /// The producer does not name the measurement model its value was corrected under.
    pub omit_measurement_model: bool,
    /// The decision requires a law where a mean is offered.
    pub decision_requires_law: bool,
}

pub fn prevalence_quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: CONSTRUCT.into(),
        variable_name: "exposure prevalence".into(),
        role: QuantityRole::Covariate,
        units: "proportion".into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

pub fn interface_components(
    model_identity: &str,
    options: &InterfaceOptions,
) -> (Vec<InterfaceContract>, Vec<ComponentConnection>) {
    let produced_support = if options.producer_support_real {
        SupportDomain::Real
    } else {
        SupportDomain::interval(0.0, 1.0)
    };
    let produced_models: Vec<String> =
        if options.omit_measurement_model { Vec::new() } else { vec![model_identity.to_owned()] };
    let measurement = InterfaceContract {
        component: "measurement".into(),
        produces: vec![
            ProducedPort::new(
                "prevalence",
                prevalence_quantity(),
                "estimand:exposure_prevalence",
                produced_support,
                [ContractOperation::Point, ContractOperation::Mean],
            )
            .with_measurement_models(produced_models),
        ],
        consumes: vec![],
    };
    let mechanisms = InterfaceContract {
        component: "mechanisms".into(),
        consumes: vec![
            ConsumedPort::new(
                "prevalence",
                prevalence_quantity(),
                "estimand:exposure_prevalence",
                SupportDomain::interval(0.0, 1.0),
                [ContractOperation::Mean],
            )
            .with_measurement_models([model_identity.to_owned()]),
        ],
        produces: vec![ProducedPort::new(
            "excess_risk",
            excess_quantity("risk"),
            "estimand:attributable_excess_risk",
            SupportDomain::interval(-1.0, 1.0),
            [ContractOperation::Mean],
        )],
    };
    let mut wanted = ConsumedPort::new(
        "excess_risk",
        excess_quantity("risk"),
        "estimand:attributable_excess_risk",
        SupportDomain::interval(-1.0, 1.0),
        [if options.decision_requires_law {
            ContractOperation::Law
        } else {
            ContractOperation::Mean
        }],
    );
    if options.decision_requires_law {
        wanted = wanted.with_uncertainty(DistributionMeaning::InterventionalPredictive);
    }
    let decision = InterfaceContract {
        component: "decision".into(),
        consumes: vec![wanted],
        produces: vec![],
    };
    let connections = vec![
        ComponentConnection::new("measurement", "prevalence", "mechanisms", "prevalence"),
        ComponentConnection::new("mechanisms", "excess_risk", "decision", "excess_risk"),
    ];
    (vec![measurement, mechanisms, decision], connections)
}

/// The order producers first, or the first refusal of the composition.
pub fn check_interfaces(
    model_identity: &str,
    options: &InterfaceOptions,
) -> Result<Vec<String>, Refused> {
    let (components, connections) = interface_components(model_identity, options);
    check_interface_composition(&components, &connections)
        .map(|composed| composed.order.clone())
        .map_err(|refusals| {
            // Report the defects in order; the stage is the first one.
            let first = &refusals[0];
            Refused::of("interface", first.code.to_string(), first.detail.to_string())
        })
}

/// Every defect of one pair, for tests that name several.
pub fn interface_pair_details(
    model_identity: &str,
    options: &InterfaceOptions,
) -> Vec<(String, String)> {
    let (components, _) = interface_components(model_identity, options);
    match check_interface_connection(&components[0], "prevalence", &components[1], "prevalence") {
        Ok(_) => Vec::new(),
        Err(refusals) => {
            refusals.iter().map(|r| (r.code.to_string(), r.detail.to_string())).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// The claim graph
// ---------------------------------------------------------------------------

/// A declaration identity from named parts (length-framed so parts cannot run together).
pub fn ident(parts: &[&str]) -> SemanticDigest {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(&u64::try_from(part.len()).unwrap().to_le_bytes());
        hasher.update(part.as_bytes());
    }
    SemanticDigest::from_bytes(*hasher.finalize().as_bytes())
}

pub fn bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

/// Everything the claim graph needs to name.
#[derive(Clone, Debug)]
pub struct GraphInputs {
    pub survey: SourceId,
    pub study_a: SourceId,
    pub study_b: SourceId,
    pub measurement_identity: String,
    pub prevalence: f64,
    pub pooled_head: String,
    pub plan_identity: String,
    pub excess_risk: f64,
    pub contract_identity: String,
    pub act: f64,
}

pub fn claim_graph(i: &GraphInputs) -> CompositionGraph {
    let mut g = CompositionGraph::new();
    let mut add = |node: CompositionNode| g.add_node(node).expect("node");
    add(CompositionNode::new(
        "survey",
        NodeKind::CohortSource,
        ident(&["survey", &i.survey.to_hex()]),
    )
    .with_evidence(vec![i.survey]));
    add(CompositionNode::new(
        "study_a",
        NodeKind::CohortSource,
        ident(&["study", &i.study_a.to_hex()]),
    )
    .with_evidence(vec![i.study_a]));
    add(CompositionNode::new(
        "study_b",
        NodeKind::CohortSource,
        ident(&["study", &i.study_b.to_hex()]),
    )
    .with_evidence(vec![i.study_b]));
    add(CompositionNode::new(
        "measurement",
        NodeKind::MeasurementModel,
        ident(&["measurement", &i.measurement_identity, &bits(i.prevalence)]),
    )
    .with_representation("scalar/1")
    .with_assumptions(vec!["non-differential misclassification".to_owned()]));
    add(CompositionNode::new(
        "pooled_effect",
        NodeKind::LedgerResult,
        ident(&["pool", &i.pooled_head]),
    ));
    add(CompositionNode::new(
        "mechanisms",
        NodeKind::MechanismComposition,
        ident(&["mechanisms", &i.plan_identity]),
    )
    .with_assumptions(vec!["declared finite mechanisms".to_owned()]));
    add(CompositionNode::new(
        "claim",
        NodeKind::Claim,
        ident(&["claim", &i.plan_identity, &bits(i.excess_risk)]),
    ));
    add(CompositionNode::new(
        "decision",
        NodeKind::Decision,
        ident(&["decision", &i.contract_identity, &bits(i.act)]),
    ));
    for (downstream, upstream, kind) in [
        ("measurement", "survey", DependencyKind::Data),
        ("pooled_effect", "study_a", DependencyKind::Evidence),
        ("pooled_effect", "study_b", DependencyKind::Evidence),
        ("mechanisms", "measurement", DependencyKind::ModelVersion),
        ("mechanisms", "pooled_effect", DependencyKind::Data),
        ("claim", "mechanisms", DependencyKind::Data),
        ("decision", "claim", DependencyKind::Data),
    ] {
        g.add_edge(downstream, upstream, kind).expect("edge");
    }
    g
}

/// Verify `current` against `previous` under the effects of the ledger for the
/// pooled result, with the ledger's overlap read from its rows.
pub fn verify(
    previous: Option<&CompositionGraph>,
    current: &CompositionGraph,
    session: &EvidenceLedgerSession,
    pooled: ResultId,
) -> CompositionVerification {
    let effects: BTreeMap<String, ObservedEffect> =
        ledger_effects(session.ledger(), &[("pooled_effect", pooled)]).expect("effects");
    CompositionVerification::verify(
        previous,
        current,
        &effects,
        Some((session.ledger(), &CancellationToken::new())),
    )
    .expect("verification")
}

/// The refusal that stops a decision whose claim is not usable as it stands,
/// or `None` when it may be evaluated.
pub fn gate(
    graph: &CompositionGraph,
    session: &EvidenceLedgerSession,
    pooled: ResultId,
) -> Option<Refused> {
    verify(Some(graph), graph, session, pooled)
        .require_valid("decision")
        .err()
        .map(|e| Refused::of("claim_graph", e.code, &e.detail))
}

// ---------------------------------------------------------------------------
// The whole pipeline, producer side
// ---------------------------------------------------------------------------

/// The producer's objects after one full run of the scenario.
pub struct World {
    pub session: EvidenceLedgerSession,
    pub survey: SourceId,
    pub study_a: SourceId,
    pub study_b: SourceId,
    pub pooled: Pooled,
    pub measured: Measured,
    pub composition: PreparedMechanismComposition,
    pub plan_identity: String,
    pub excess_risk: f64,
    pub declaration: DecisionDeclaration,
    pub result: DecisionResult,
    pub graph: CompositionGraph,
}

impl World {
    pub fn graph_inputs(&self) -> GraphInputs {
        GraphInputs {
            survey: self.survey,
            study_a: self.study_a,
            study_b: self.study_b,
            measurement_identity: self.measured.identity.clone(),
            prevalence: self.measured.point,
            pooled_head: self.pooled.head.clone(),
            plan_identity: self.plan_identity.clone(),
            excess_risk: self.excess_risk,
            contract_identity: decision_contract(&self.declaration).identity().unwrap(),
            act: value_of(&self.result, "act"),
        }
    }

    /// Recompute everything downstream of a pooled result: the composition, the
    /// claim, the decision and the graph.
    pub fn downstream_of(
        sc: &Value,
        session: EvidenceLedgerSession,
        sources: (SourceId, SourceId, SourceId),
        pooled: Pooled,
        measured: Measured,
    ) -> Self {
        let composition = compose(measured.point, num(&sc["baseline_risk"]), pooled.estimate)
            .expect("the composition compiles");
        let excess = excess_risk(&composition);
        let plan_identity = composition.plan_identity_hex();
        let declaration = DecisionDeclaration::from_fixture(sc);
        let result = decide(&declaration, excess, &session.state_digest_hex(), &plan_identity)
            .expect("a decision");
        let mut world = Self {
            session,
            survey: sources.0,
            study_a: sources.1,
            study_b: sources.2,
            pooled,
            measured,
            composition,
            plan_identity,
            excess_risk: excess,
            declaration,
            result,
            graph: CompositionGraph::new(),
        };
        world.graph = claim_graph(&world.graph_inputs());
        world
    }

    /// Run the scenario from the declared inputs.
    pub fn build(sc: &Value) -> Self {
        let mut session = StudyBuilder::evidence_ledger();
        let survey = session
            .add_source(study_source("survey", "survey", rows_of(&sc["survey_rows"])))
            .expect("survey");
        let studies = sc["studies"].as_array().expect("studies");
        let source_of = |session: &mut EvidenceLedgerSession, study: &Value| {
            session
                .add_source(study_source(&text(&study["label"]), "cohort", rows_of(&study["rows"])))
                .expect("study")
        };
        let a = source_of(&mut session, &studies[0]);
        let b = source_of(&mut session, &studies[1]);
        let pooled = pool(
            &mut session,
            "pooled_effect",
            &[
                (a, num(&studies[0]["estimate"]), num(&studies[0]["variance"])),
                (b, num(&studies[1]["estimate"]), num(&studies[1]["variance"])),
            ],
            None,
            None,
        )
        .expect("the studies are disjoint");
        let measured = measure(sc, CONSTRUCT).expect("the measurement model prepares");
        Self::downstream_of(sc, session, (survey, a, b), pooled, measured)
    }

    /// The revised study summary, after the study grew.
    pub fn revise(&mut self, sc: &Value) -> RevisionStep {
        let revision = &sc["revision"];
        let report = self
            .session
            .revise_source(antecedent::SourceRevision::Update {
                target: self.study_b,
                replacement: study_source("study_b_grown", "cohort", rows_of(&revision["rows"])),
            })
            .expect("the study grew");
        let grown = report.created_source.expect("a successor source");
        RevisionStep { grown, summary: (num(&revision["estimate"]), num(&revision["variance"])) }
    }
}

/// What a revision of the study leaves to recompute.
pub struct RevisionStep {
    pub grown: SourceId,
    pub summary: (f64, f64),
}

// ---------------------------------------------------------------------------
// Bundle and consumer
// ---------------------------------------------------------------------------

/// A study as the consumer re-derives it: declared rows and summary.
#[derive(Clone, Debug)]
pub struct StudyDeclaration {
    pub label: String,
    pub rows: Range<u32>,
    pub estimate: f64,
    pub variance: f64,
}

/// Everything a consumer holds: bytes, retained identities and declarations.
/// It holds no producer object.
#[derive(Clone)]
pub struct Bundle {
    pub scenario: Value,
    pub ledger_bytes: Vec<u8>,
    pub ledger_identity: String,
    pub pooled: ResultId,
    pub measurement_bytes: Vec<u8>,
    pub measurement_identity: String,
    pub exposed_bytes: Vec<u8>,
    pub exposed_digest: String,
    pub unexposed_bytes: Vec<u8>,
    pub unexposed_digest: String,
    pub verification_bytes: Vec<u8>,
    pub verification_identity: String,
    pub snapshot_id: String,
    pub reported_act: f64,
    pub reported_hold: f64,
    pub reported_selected: String,
    pub studies: Vec<StudyDeclaration>,
    pub decision: DecisionDeclaration,
    pub interface: InterfaceOptions,
}

pub fn study_declarations(sc: &Value) -> Vec<StudyDeclaration> {
    sc["studies"]
        .as_array()
        .expect("studies")
        .iter()
        .map(|s| StudyDeclaration {
            label: text(&s["label"]),
            rows: rows_of(&s["rows"]),
            estimate: num(&s["estimate"]),
            variance: num(&s["variance"]),
        })
        .collect()
}

/// The declared studies after the second study grew: the consumer's
/// re-derivation of the revised pool starts from these.
pub fn revised_study_declarations(sc: &Value) -> Vec<StudyDeclaration> {
    let mut studies = study_declarations(sc);
    let revision = &sc["revision"];
    studies[1] = StudyDeclaration {
        label: "study_b_grown".to_owned(),
        rows: rows_of(&revision["rows"]),
        estimate: num(&revision["estimate"]),
        variance: num(&revision["variance"]),
    };
    studies
}

impl World {
    /// Seal the producer's artifacts and the numbers it reports.
    pub fn bundle(&self, sc: &Value) -> Bundle {
        let export = self.session.export("release-boundary-ledger").expect("ledger export");
        let (exposed_bytes, exposed_digest) = export_interventional(&self.composition, "1");
        let (unexposed_bytes, unexposed_digest) = export_interventional(&self.composition, "0");
        let verification =
            verify(Some(&self.graph), &self.graph, &self.session, self.pooled.result);
        let verification_bytes =
            verification.export("release-boundary-verification").expect("verification export");
        Bundle {
            scenario: sc.clone(),
            ledger_bytes: export.bytes,
            ledger_identity: export.identity,
            pooled: self.pooled.result,
            measurement_bytes: self.measured.bytes.clone(),
            measurement_identity: self.measured.identity.clone(),
            exposed_bytes,
            exposed_digest,
            unexposed_bytes,
            unexposed_digest,
            verification_bytes,
            verification_identity: verification.identity().to_owned(),
            snapshot_id: self.session.state_digest_hex(),
            reported_act: value_of(&self.result, "act"),
            reported_hold: value_of(&self.result, "hold"),
            reported_selected: selected(&self.result),
            studies: study_declarations(sc),
            decision: self.declaration.clone(),
            interface: InterfaceOptions::default(),
        }
    }
}

/// What the consumer established.
#[derive(Clone, Debug)]
pub struct ConsumerReport {
    pub prevalence: f64,
    pub pooled_effect: f64,
    pub pooled_variance: f64,
    pub exposed_risk: f64,
    pub unexposed_risk: f64,
    pub excess_risk: f64,
    pub act: f64,
    pub hold: f64,
    pub selected: String,
    pub snapshot_bound: bool,
    pub claim_graph_valid: bool,
}

/// Hand-written closed forms: the consumer's oracle. No library call.
pub struct Oracle {
    pub prevalence: f64,
    pub pooled: f64,
    pub pooled_variance: f64,
    pub excess_risk: f64,
    pub act: f64,
}

pub fn oracle(sc: &Value, studies: &[StudyDeclaration]) -> Oracle {
    let m = &sc["measurement"];
    let (se, sp) = (num(&m["sensitivity"]), num(&m["specificity"]));
    let observed = m["events"].as_f64().unwrap() / m["total"].as_f64().unwrap();
    // Rogan-Gladen: p* = Se pi + (1 - Sp)(1 - pi).
    let prevalence = (observed + sp - 1.0) / (se + sp - 1.0);
    // Inverse-variance pool: sum(w y) / sum(w), variance 1 / sum(w).
    let weights: Vec<f64> = studies.iter().map(|s| 1.0 / s.variance).collect();
    let total: f64 = weights.iter().sum();
    let pooled = studies.iter().zip(&weights).map(|(s, w)| w * s.estimate).sum::<f64>() / total;
    // Y | X is Bernoulli(q0 + d X): P(Y=1) - P(Y=1 | do(X=0)) = pi * d.
    let excess_risk = prevalence * pooled;
    let d = &sc["decision"];
    let act = num(&d["benefit_per_unit_risk"]) * excess_risk - num(&d["programme_cost"]);
    Oracle { prevalence, pooled, pooled_variance: 1.0 / total, excess_risk, act }
}

/// Consume a bundle: every artifact against its retained identity, the
/// pool re-derived from the declared studies, the decision re-evaluated from
/// the consumed numbers, everything compared with the closed forms.
pub fn consume(bundle: &Bundle) -> Result<ConsumerReport, Refused> {
    let sc = &bundle.scenario;
    let truth = oracle(sc, &bundle.studies);

    // The evidence state: the ledger replays under the retained identity.
    let ledger =
        consume_evidence_ledger_artifact(&bundle.ledger_bytes, Some(&bundle.ledger_identity))
            .map_err(|e| Refused::of("ledger", e.code, &e.detail))?;
    let synthesis = ledger.synthesis_of(bundle.pooled).expect("the pooled result is carried");
    let (pooled, variance) = (synthesis.values[0], synthesis.values[1]);

    // The pool re-derived from the declared studies: one observation counts once.
    let mut fresh = StudyBuilder::evidence_ledger();
    let mut sources = Vec::new();
    for study in &bundle.studies {
        let id = fresh
            .add_source(study_source(&study.label, "cohort", study.rows.clone()))
            .map_err(|e| Refused::of("evidence", e.code, &e.detail))?;
        sources.push((id, study.estimate, study.variance));
    }
    let rederived = pool(&mut fresh, "rederived", &sources, None, None)?;

    // The corrected prevalence, recomputed from its artifact.
    let measurement =
        consume_measurement_model_artifact(&bundle.measurement_bytes, &bundle.measurement_identity)
            .map_err(|e| Refused::of("measurement", e.code, &e.detail))?;
    let prevalence = measurement.replayed().expect("a replayed value").point;

    // The interventional risks, recomputed from the composition artifacts.
    let risk = |bytes: &[u8], digest: &str| -> Result<f64, Refused> {
        let consumed = consume_mechanism_composition_artifact(bytes, Some(digest), &ctx())
            .map_err(|e| Refused::of("mechanisms", e.code(), e.details().join(",")))?;
        Ok(consumed.finite_distribution().expect("a finite distribution")[1].1)
    };
    let exposed = risk(&bundle.exposed_bytes, &bundle.exposed_digest)?;
    let unexposed = risk(&bundle.unexposed_bytes, &bundle.unexposed_digest)?;
    let excess = prevalence * (exposed - unexposed);

    // The interfaces connect before anything is believed.
    check_interfaces(&bundle.measurement_identity, &bundle.interface)?;

    // The decision is re-evaluated from the consumed mean through the
    // consumer's own declaration.
    let result =
        decide(&bundle.decision, excess, &ledger.ledger().state_digest().to_hex(), "consumer")?;

    // The claim graph replays and is valid at the ledger state consumed.
    let verification = consume_composition_verification_artifact(
        &bundle.verification_bytes,
        Some(&bundle.verification_identity),
    )
    .map_err(|e| Refused::of("claim_graph", e.code, &e.detail))?;
    verification
        .verify_ledger(ledger.ledger(), &CancellationToken::new())
        .map_err(|e| Refused::of("claim_graph", e.code, &e.detail))?;

    // The comparison with the closed forms, at predeclared tolerances.
    let (exact, loose) = (1e-12, 1e-9);
    let close = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol;
    assert!(close(prevalence, truth.prevalence, exact), "prevalence {prevalence}");
    assert!(close(pooled, truth.pooled, exact), "pooled {pooled}");
    assert!(close(rederived.estimate, truth.pooled, exact), "re-derived pool");
    assert!(close(variance, truth.pooled_variance, exact), "pooled variance");
    assert!(close(excess, truth.excess_risk, loose), "excess {excess}");
    assert!(close(value_of(&result, "act"), truth.act, loose), "act");
    assert!(close(bundle.reported_act, truth.act, loose), "reported act");

    Ok(ConsumerReport {
        prevalence,
        pooled_effect: pooled,
        pooled_variance: variance,
        exposed_risk: exposed,
        unexposed_risk: unexposed,
        excess_risk: excess,
        act: value_of(&result, "act"),
        hold: value_of(&result, "hold"),
        selected: selected(&result),
        snapshot_bound: ledger.ledger().state_digest().to_hex() == bundle.snapshot_id,
        claim_graph_valid: verification.require_valid("decision").is_ok(),
    })
}
