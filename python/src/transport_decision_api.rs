//! Python bridge for transported decisions, time-ordered path transport and the
//! composition bundles that replay a transported fit.
//!
//! Python declares the graph, the transport query, the raw source and target
//! evidence, the decision contract with its coordinates and the time-ordered
//! path; Rust re-identifies, fits, decides, seals the portable bytes and replays
//! them under the program identity the consumer retained. Every number, every
//! identity and every refusal is Rust's. No route here earns interval coverage
//! or decision reliability: a result says `not_measured` where its route does.

use std::sync::Arc;

use antecedent::CounterfactualTransportOptions;
use antecedent::analysis::composition::{
    BundleError, BundleLimits, ChainBundleRecomputeError, CompiledJointTransportEffectNodeBundle,
    CompiledTrialIpwNodeBundle, JointTransportEffectBundleError,
    JointTransportEffectNodeBundleReceipt, TrialIpwBundleError, TrialIpwNodeBundleReceipt,
    compile_joint_transport_effect_node_bundle_recomputation,
    compile_trial_ipw_node_bundle_recomputation,
    consume_portable_joint_transport_effect_node_bundle_recomputation,
    consume_portable_trial_ipw_node_bundle_recomputation, joint_transport_effect_bundle_consumer,
    trial_ipw_bundle_consumer,
};
use antecedent::analysis::inverse_query::DecisionContract;
use antecedent::analysis::joint_transport_effect_decision::{
    CompiledJointTransportEffectDecision, JointTransportEffectDecisionError,
    JointTransportEffectDecisionRequest, JointTransportEffectDecisionResult,
    compile_joint_transport_effect_decision, consume_joint_transport_effect_decision,
};
use antecedent::analysis::joint_transport_predictive_decision::{
    CompiledJointTransportPredictiveDecision, JointTransportPredictiveDecisionError,
    JointTransportPredictiveDecisionRequest, JointTransportPredictiveDecisionResult,
    compile_joint_transport_predictive_decision, consume_joint_transport_predictive_decision,
};
use antecedent::analysis::propagation_chain::ChainRefusal;
use antecedent::analysis::temporal_path_transport::{
    CompiledContinuousChainTransport, CompiledSampledTemporalPathTransport,
    CompiledTemporalPathTransport, ContinuousChainTransportRequest, ContinuousChainTransportResult,
    SampledTemporalPathTransportResult, TemporalPathSpec, TemporalPathTransportError,
    TemporalPathTransportResult, compile_continuous_chain_transport,
    compile_sampled_temporal_path_transport, compile_temporal_path_transport,
    consume_continuous_chain_transport, consume_sampled_temporal_path_transport,
    consume_temporal_path_transport,
};
use antecedent::analysis::trial_ipw_predictive_decision::{
    CompiledTrialIpwPredictiveDecision, TrialIpwPredictiveDecisionError,
    TrialIpwPredictiveDecisionRequest, TrialIpwPredictiveDecisionResult,
    compile_trial_ipw_predictive_decision, consume_trial_ipw_predictive_decision,
};
use antecedent::analysis::trial_ipw_transport_decision::{
    CompiledTrialIpwTransportDecision, TrialIpwTransportDecisionError,
    TrialIpwTransportDecisionRequest, TrialIpwTransportDecisionResult,
    compile_trial_ipw_transport_decision, consume_trial_ipw_transport_decision,
};
use antecedent::analysis::{trial_ipw_predictive_decision, trial_ipw_transport_decision};
use antecedent::design::decision_artifact::contract_from_json_refusal;
use antecedent::design::decision_eval::{DecisionResult, Verdict};
use antecedent::estimate::ExactEvaluationLimits;
use antecedent::estimate::counterfactual_transport_sampled::{
    SampledTransportLimits, SampledTransportRequest,
};
use antecedent::estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JointPriors, JointTransportModel, JointTransportOptions,
    PriorProvenance, SourceData, SourceDependence, SourceSharing, TargetData, TransportGraphClass,
    VaryingBlock,
};
use antecedent::execution::SearchLimits;
use antecedent::graph::SelectionDiagram;
use antecedent::identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportProblem,
    CounterfactualTransportSource,
};
use antecedent::io::IoError;
use antecedent::io::joint_bayesian_transport_artifact::{
    DataIdentityWire, GaussianPriorWire, PriorProvenanceWire,
};
use antecedent::prelude::{ExecutionContext, VariableId};
use antecedent::query::{EvidenceCatalog, TransportQuery};
use antecedent::reason::reason_code;
use antecedent::scientific::{ScientificQuantity, SemanticDigest};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::{Value as Json, json};

use crate::bridge::{
    MAX_CONFIG_DECLARATION_BYTES, MAX_DATA_DECLARATION_BYTES, check_declared_size,
    invalid_joint_data_identity,
};
use crate::composition_bundle_api::{PyBundleBuilder, PyCompositionBundle};
use crate::graphs::Admg;
use crate::propagation_chain_api::{ChainParts, Refused, chain_parts};
use crate::transport_common::{RegimeCheck, error as transport_error, parse_laws, resolve};
use crate::transport_counterfactual_sampled_api::{
    TableKeyTuple, parse_count_table, parse_overlap, parse_unseen_rule,
};
use crate::transport_interference_api::{ResponseArgs, parse_catalog, transport_query};
use crate::transported_response_api::{quantities_of, quantity_of};

const TRIAL_EFFECT: &str = "trial_ipw_effect";
const TRIAL_PREDICTIVE: &str = "trial_ipw_predictive";
const JOINT_EFFECT: &str = "joint_transport_effect";
const JOINT_PREDICTIVE: &str = "joint_transport_predictive";
const TRIAL_FIT: &str = "trial_ipw";
const JOINT_FIT: &str = "joint_transport_effect";
const PATH_EXACT: &str = "exact";
const PATH_SAMPLED: &str = "sampled";
const PATH_CONTINUOUS: &str = "continuous_chain";

/// One sampled overlap declaration: default relation and declared table pairs.
type OverlapTuple = (String, Vec<(TableKeyTuple, TableKeyTuple, String)>);

// ---------------------------------------------------------------------------
// Execution and refusals.
// ---------------------------------------------------------------------------

/// The context one call runs under: the caller's seed, threads, memory limit and token.
fn context(
    seed: u64,
    threads: Option<u32>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> ExecutionContext {
    let mut ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    ctx.set_memory_limit(memory_bytes);
    ctx
}

/// Where a refusal arose: an invalid declaration is the caller's input, an
/// execution refusal is the route's, and an export or consume mismatch means the
/// bytes do not bind to what the consumer retained.
#[derive(Clone, Copy)]
enum Stage {
    Compile,
    Run,
    Bind,
}

/// The refusal classes the routes of this module share.
enum Failure {
    Invalid(String),
    Identification(String),
    Contract(String),
    Decision(String),
    Chain(ChainRefusal),
    Io(IoError),
    Query(antecedent::query::QueryError),
    Estimate(antecedent::estimate::EstimationError),
    NonTransportable(String),
    NotSupported(String),
    Bundle(BundleError),
}

/// A refusal carrying its own registered code keeps it; any other takes `default`.
fn coded(default: &'static str, message: String) -> PyErr {
    if crate::reason_code_in_message(&message).is_some() {
        crate::unsupported_message(message)
    } else {
        crate::refusal(default, message)
    }
}

fn raise(failure: Failure, stage: Stage, ctx: Option<&ExecutionContext>) -> PyErr {
    if ctx.is_some_and(|ctx| ctx.cancellation.is_cancelled()) {
        return crate::CausalCancelledError::new_err("transported route cancelled");
    }
    let binding = reason_code!("external_binding_mismatch");
    match failure {
        Failure::Invalid(message) => match stage {
            Stage::Compile => crate::value_err_code(reason_code!("invalid_argument"), message),
            Stage::Run => crate::refusal(reason_code!("invalid_argument"), message),
            Stage::Bind => crate::refusal(binding, message),
        },
        Failure::Identification(message) => coded(reason_code!("transport_not_certified"), message),
        Failure::Contract(message) => {
            crate::refusal(reason_code!("decision_contract_unsatisfied"), message)
        }
        Failure::Decision(message) => match stage {
            Stage::Bind => coded(binding, message),
            Stage::Compile | Stage::Run => {
                coded(reason_code!("decision_contract_unsatisfied"), message)
            }
        },
        Failure::Chain(refusal) => crate::refusal(refusal.code(), refusal.to_string()),
        Failure::Io(error) => transport_error(error),
        Failure::Query(error) => transport_error(error),
        Failure::Estimate(error) => transport_error(error),
        Failure::NonTransportable(message) => {
            crate::refusal(reason_code!("transport_proven_non_transportable"), message)
        }
        Failure::NotSupported(message) => crate::unsupported_py_err(message),
        Failure::Bundle(error) => match error.to_refusal() {
            Some(refusal) => crate::refusal(refusal.code, format!("{}: {error}", refusal.detail)),
            None => crate::refusal(reason_code!("invalid_argument"), error.to_string()),
        },
    }
}

impl From<TrialIpwTransportDecisionError> for Failure {
    fn from(error: TrialIpwTransportDecisionError) -> Self {
        use TrialIpwTransportDecisionError as E;
        let message = error.to_string();
        match error {
            E::Invalid(_) => Self::Invalid(message),
            E::Producer(_) => Self::Identification(message),
            E::Contract(_) => Self::Contract(message),
            E::Chain(_) => Self::Decision(message),
            E::Io(io) => Self::Io(io),
        }
    }
}

impl From<TrialIpwPredictiveDecisionError> for Failure {
    fn from(error: TrialIpwPredictiveDecisionError) -> Self {
        use TrialIpwPredictiveDecisionError as E;
        let message = error.to_string();
        match error {
            E::Invalid(_) => Self::Invalid(message),
            E::Producer(_) => Self::Identification(message),
            E::Contract(_) => Self::Contract(message),
            E::Chain(_) => Self::Decision(message),
            E::Io(io) => Self::Io(io),
        }
    }
}

impl From<JointTransportEffectDecisionError> for Failure {
    fn from(error: JointTransportEffectDecisionError) -> Self {
        use JointTransportEffectDecisionError as E;
        let message = error.to_string();
        match error {
            E::Invalid(_) => Self::Invalid(message),
            E::Identify(_) => Self::Identification(message),
            E::Contract(_) => Self::Contract(message),
            E::Chain(_) => Self::Decision(message),
            E::Io(io) => Self::Io(io),
        }
    }
}

impl From<JointTransportPredictiveDecisionError> for Failure {
    fn from(error: JointTransportPredictiveDecisionError) -> Self {
        use JointTransportPredictiveDecisionError as E;
        let message = error.to_string();
        match error {
            E::Invalid(_) => Self::Invalid(message),
            E::Identify(_) => Self::Identification(message),
            E::Contract(_) => Self::Contract(message),
            E::Chain(_) => Self::Decision(message),
            E::Io(io) => Self::Io(io),
        }
    }
}

impl From<TemporalPathTransportError> for Failure {
    fn from(error: TemporalPathTransportError) -> Self {
        use TemporalPathTransportError as E;
        let message = error.to_string();
        match error {
            E::Invalid(_) => Self::Invalid(message),
            E::Query(query) => Self::Query(query),
            E::Io(io) => Self::Io(io),
            E::Estimate(estimate) => Self::Estimate(estimate),
            E::ProvenNonTransportable => Self::NonTransportable(message),
        }
    }
}

impl From<ChainBundleRecomputeError> for Failure {
    fn from(error: ChainBundleRecomputeError) -> Self {
        use ChainBundleRecomputeError as E;
        let message = error.to_string();
        match error {
            E::Bundle(bundle) => Self::Bundle(bundle),
            E::Chain(chain) => Self::Chain(chain),
            E::Io(io) => Self::Io(io),
            E::Invalid(_) => Self::Invalid(message),
        }
    }
}

impl From<TrialIpwBundleError> for Failure {
    fn from(error: TrialIpwBundleError) -> Self {
        use TrialIpwBundleError as E;
        let message = error.to_string();
        match error {
            E::Bundle(bundle) => bundle.into(),
            E::Invalid(_) => Self::Invalid(message),
            E::NotSupported(_) => Self::NotSupported(message),
        }
    }
}

impl From<JointTransportEffectBundleError> for Failure {
    fn from(error: JointTransportEffectBundleError) -> Self {
        use JointTransportEffectBundleError as E;
        let message = error.to_string();
        match error {
            E::Bundle(bundle) => bundle.into(),
            E::Invalid(_) => Self::Invalid(message),
            E::NotSupported(_) => Self::NotSupported(message),
        }
    }
}

fn invalid_declaration(message: &str) -> PyErr {
    crate::value_err_code(reason_code!("invalid_argument"), message)
}

fn unknown_route(route: &str, known: &[&str]) -> PyErr {
    invalid_declaration(&format!("unknown route {route:?}; use one of {known:?}"))
}

fn expected_digest(hex: &str) -> PyResult<SemanticDigest> {
    crate::digest::digest(hex, "expected_identity")
}

fn json_err(what: &str) -> impl Fn(serde_json::Error) -> PyErr {
    move |error| invalid_declaration(&format!("invalid {what}: {error}"))
}

// ---------------------------------------------------------------------------
// Shared declarations.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponseIn {
    kind: String,
    treatments: Vec<String>,
    outcomes: Vec<String>,
    grid: Option<Vec<f64>>,
    at: Option<Vec<f64>>,
    direction: Option<Vec<f64>>,
    order: u8,
    scale: String,
    weighting: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructureIn {
    selections: Vec<String>,
    source: String,
    target: String,
    source_experiments: Vec<String>,
    response: ResponseIn,
}

/// The selection diagram and transport query, built from the graph's names the
/// way the identification stage builds them.
fn structure(
    graph: &Admg,
    structure_json: &str,
    catalog: Option<&Bound<'_, PyAny>>,
) -> PyResult<(SelectionDiagram, TransportQuery)> {
    check_declared_size(structure_json, MAX_CONFIG_DECLARATION_BYTES, "transport structure")?;
    let declared: StructureIn =
        serde_json::from_str(structure_json).map_err(json_err("transport structure"))?;
    let names = &graph.names;
    let resolve_all = |list: &[String]| -> PyResult<Arc<[VariableId]>> {
        list.iter().map(|name| resolve(names, name)).collect::<PyResult<Vec<_>>>().map(Arc::from)
    };
    let response = declared.response;
    let mut query = transport_query(
        ResponseArgs {
            kind: response.kind,
            treatments: response.treatments,
            outcomes: response.outcomes,
            grid: response.grid,
            at: response.at,
            direction: response.direction,
            order: response.order,
            scale: response.scale,
            weighting: response.weighting,
        },
        declared.source,
        declared.target,
        &declared.source_experiments,
        resolve_all,
    )?;
    if let Some(catalog) = catalog {
        query = query.with_catalog(parse_catalog(catalog, graph)?).map_err(transport_error)?;
    }
    let diagram = SelectionDiagram::try_new(
        graph.aligned_to_names(names)?,
        resolve_all(&declared.selections)?,
    )
    .map_err(crate::py_err)?;
    Ok((diagram, query))
}

fn contract_of(contract_json: &str) -> PyResult<DecisionContract> {
    check_declared_size(contract_json, MAX_CONFIG_DECLARATION_BYTES, "decision contract")?;
    contract_from_json_refusal(contract_json).map_err(|refusal| {
        let at = match &refusal.offending {
            Some(offending) => format!(" at {offending}"),
            None => String::new(),
        };
        crate::refusal(refusal.code, format!("{}{at}", refusal.detail))
    })
}

/// The effect coordinate or the two outcome worlds; exactly one must be declared.
enum Target {
    Effect(ScientificQuantity),
    Worlds([ScientificQuantity; 2]),
}

fn target_of(effect_json: Option<&str>, worlds_json: Option<&str>) -> PyResult<Target> {
    match (effect_json, worlds_json) {
        (Some(effect), None) => Ok(Target::Effect(quantity_of(effect, "effect coordinate")?)),
        (None, Some(worlds)) => {
            let worlds = quantities_of(worlds, "outcome worlds")?;
            let worlds: [ScientificQuantity; 2] = worlds.try_into().map_err(|_| {
                invalid_declaration(
                    "outcome worlds are exactly the untreated and the treated world",
                )
            })?;
            Ok(Target::Worlds(worlds))
        }
        _ => Err(invalid_declaration(
            "declare either the effect coordinate or the two outcome worlds",
        )),
    }
}

// ---------------------------------------------------------------------------
// Transported decisions.
// ---------------------------------------------------------------------------

enum CompiledDecision {
    TrialEffect(Box<CompiledTrialIpwTransportDecision>),
    TrialPredictive(Box<CompiledTrialIpwPredictiveDecision>),
    JointEffect(Box<CompiledJointTransportEffectDecision>),
    JointPredictive(Box<CompiledJointTransportPredictiveDecision>),
}

enum DecisionOutcome {
    TrialEffect(Box<TrialIpwTransportDecisionResult>),
    TrialPredictive(Box<TrialIpwPredictiveDecisionResult>),
    JointEffect(Box<JointTransportEffectDecisionResult>),
    JointPredictive(Box<JointTransportPredictiveDecisionResult>),
}

/// A compiled transported decision: the six-layer program, before any fit runs.
#[pyclass(name = "CompiledTransportDecisionHandle", unsendable)]
struct CompiledTransportDecisionNative {
    inner: CompiledDecision,
}

/// One fitted transported decision; exported only through the program that made it.
#[pyclass(name = "TransportDecisionResultHandle", unsendable)]
struct TransportDecisionResultNative {
    inner: DecisionOutcome,
}

impl CompiledDecision {
    const fn route(&self) -> &'static str {
        match self {
            Self::TrialEffect(_) => TRIAL_EFFECT,
            Self::TrialPredictive(_) => TRIAL_PREDICTIVE,
            Self::JointEffect(_) => JOINT_EFFECT,
            Self::JointPredictive(_) => JOINT_PREDICTIVE,
        }
    }

    fn identity(&self) -> SemanticDigest {
        match self {
            Self::TrialEffect(plan) => plan.lowered().identity(),
            Self::TrialPredictive(plan) => plan.lowered().identity(),
            Self::JointEffect(plan) => plan.lowered().identity(),
            Self::JointPredictive(plan) => plan.lowered().identity(),
        }
    }

    fn plan(&self) -> Json {
        match self {
            Self::TrialEffect(plan) => plan.plan_view().to_json(),
            Self::TrialPredictive(plan) => plan.plan_view().to_json(),
            Self::JointEffect(plan) => plan.plan_view().to_json(),
            Self::JointPredictive(plan) => plan.plan_view().to_json(),
        }
    }

    fn execute(&self, ctx: &ExecutionContext) -> PyResult<DecisionOutcome> {
        let run = Some(ctx);
        Ok(match self {
            Self::TrialEffect(plan) => DecisionOutcome::TrialEffect(Box::new(
                plan.execute(ctx).map_err(|error| raise(error.into(), Stage::Run, run))?,
            )),
            Self::TrialPredictive(plan) => DecisionOutcome::TrialPredictive(Box::new(
                plan.execute(ctx).map_err(|error| raise(error.into(), Stage::Run, run))?,
            )),
            Self::JointEffect(plan) => DecisionOutcome::JointEffect(Box::new(
                plan.execute(ctx).map_err(|error| raise(error.into(), Stage::Run, run))?,
            )),
            Self::JointPredictive(plan) => DecisionOutcome::JointPredictive(Box::new(
                plan.execute(ctx).map_err(|error| raise(error.into(), Stage::Run, run))?,
            )),
        })
    }

    fn export(
        &self,
        outcome: &DecisionOutcome,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> PyResult<Vec<u8>> {
        let bind = Some(ctx);
        match (self, outcome) {
            (Self::TrialEffect(plan), DecisionOutcome::TrialEffect(result)) => plan
                .export(result, artifact_id, ctx)
                .map_err(|error| raise(error.into(), Stage::Bind, bind)),
            (Self::TrialPredictive(plan), DecisionOutcome::TrialPredictive(result)) => plan
                .export(result, artifact_id, ctx)
                .map_err(|error| raise(error.into(), Stage::Bind, bind)),
            (Self::JointEffect(plan), DecisionOutcome::JointEffect(result)) => plan
                .export(result, artifact_id, ctx)
                .map_err(|error| raise(error.into(), Stage::Bind, bind)),
            (Self::JointPredictive(plan), DecisionOutcome::JointPredictive(result)) => plan
                .export(result, artifact_id, ctx)
                .map_err(|error| raise(error.into(), Stage::Bind, bind)),
            _ => Err(invalid_declaration(&format!(
                "the result is not one of this {} program's results",
                self.route()
            ))),
        }
    }
}

fn verdict_json(verdict: &Verdict) -> Json {
    match verdict {
        Verdict::UniquelyOptimal(action) => json!({ "uniquely_optimal": action }),
        Verdict::Indistinguishable(actions) => json!({ "indistinguishable": actions }),
        Verdict::NoAdmissibleAction => json!("no_admissible_action"),
    }
}

fn decision_json(result: &DecisionResult) -> Json {
    json!({
        "contract_identity": result.contract_identity,
        "criterion_label": result.criterion_label,
        "actions": result.actions.iter().map(|action| json!({
            "id": action.id,
            "admissible": action.admissible,
            "exclusions": action.exclusions.iter().map(|exclusion| json!({
                "constraint_id": exclusion.constraint_id,
                "probability": exclusion.probability,
                "required": exclusion.required,
            })).collect::<Vec<_>>(),
            "expected_utility": action.expected_utility,
            "value": action.value,
            "standard_error": action.standard_error,
            "expected_regret": action.expected_regret,
            "max_regret": action.max_regret,
        })).collect::<Vec<_>>(),
        "verdict": verdict_json(&result.verdict),
        "evpi": result.evpi,
        "n_draws": result.n_draws,
        "effective_draws": result.effective_draws,
        "assumptions": result.assumptions,
    })
}

impl DecisionOutcome {
    const fn route(&self) -> &'static str {
        match self {
            Self::TrialEffect(_) => TRIAL_EFFECT,
            Self::TrialPredictive(_) => TRIAL_PREDICTIVE,
            Self::JointEffect(_) => JOINT_EFFECT,
            Self::JointPredictive(_) => JOINT_PREDICTIVE,
        }
    }

    fn chain(&self) -> &antecedent::analysis::propagation_chain::PropagationChain {
        match self {
            Self::TrialEffect(result) => &result.decision.chain,
            Self::TrialPredictive(result) => &result.decision.chain,
            Self::JointEffect(result) => &result.decision.chain,
            Self::JointPredictive(result) => &result.decision.chain,
        }
    }

    fn summary(&self) -> Json {
        let chain_columns = |chain: &antecedent::analysis::propagation_chain::PropagationChain| {
            chain
                .record()
                .sources
                .iter()
                .take(2)
                .map(|source| source.values.iter().copied().map(f64::from_bits).collect())
                .collect::<Vec<Vec<f64>>>()
        };
        let (decision, program, calibration, effect, predictive, fit) = match self {
            Self::TrialEffect(result) => (
                &result.decision,
                result.lowered_program_identity,
                result.calibration,
                Some(result.effect_draws.clone()),
                None,
                Json::Null,
            ),
            Self::TrialPredictive(result) => (
                &result.decision,
                result.lowered_program_identity,
                result.calibration,
                None,
                Some(result.predictive_draws.to_vec()),
                Json::Null,
            ),
            Self::JointEffect(result) => (
                &result.decision,
                result.lowered_program_identity,
                result.calibration,
                result
                    .fit
                    .draws
                    .column("effect.target")
                    .map(|column| result.fit.draws.coordinate(column)),
                None,
                json!({
                    "target_effect_mean": result.fit.target_effect_mean,
                    "target_effect_variance": result.fit.target_effect_variance,
                    "model_identity": result.fit.model_identity,
                }),
            ),
            Self::JointPredictive(result) => (
                &result.decision,
                result.lowered_program_identity,
                result.calibration,
                None,
                Some(chain_columns(&result.decision.chain)),
                json!({
                    "target_effect_mean": result.fit.target_effect_mean,
                    "target_effect_variance": result.fit.target_effect_variance,
                    "model_identity": result.fit.model_identity,
                }),
            ),
        };
        json!({
            "route": self.route(),
            "program_identity": program.to_hex(),
            "decision_program_identity": decision.lowered_program_identity.to_hex(),
            "chain_identity": decision.chain.identity(),
            "calibration": calibration,
            "decision": decision_json(&decision.decision),
            "effect_draws": effect,
            "predictive_draws": predictive,
            "fit": fit,
        })
    }
}

#[pymethods]
impl CompiledTransportDecisionNative {
    /// `trial_ipw_effect`, `trial_ipw_predictive`, `joint_transport_effect` or
    /// `joint_transport_predictive`.
    #[getter]
    fn route(&self) -> &'static str {
        self.inner.route()
    }

    /// The six-layer program identity a consumer retains, as lowercase hex.
    #[getter]
    fn identity(&self) -> String {
        self.inner.identity().to_hex()
    }

    /// Every compiler layer of the program, as JSON, without fitting.
    fn plan_json(&self) -> String {
        self.inner.plan().to_string()
    }

    /// Fit the raw evidence under the program and evaluate the decision.
    #[pyo3(signature = (*, seed=1, threads=None, memory_bytes=None, cancel=None))]
    fn execute(
        &self,
        seed: u64,
        threads: Option<u32>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<TransportDecisionResultNative> {
        crate::catch_ffi(|| {
            let ctx = context(seed, threads, memory_bytes, cancel);
            Ok(TransportDecisionResultNative { inner: self.inner.execute(&ctx)? })
        })
    }

    /// Re-execute, require the same result and seal the raw evidence, the
    /// contract and the decision as portable bytes.
    #[pyo3(signature = (result, artifact_id, *, seed=1, threads=None, memory_bytes=None, cancel=None))]
    #[allow(clippy::too_many_arguments, reason = "the result plus the execution controls")]
    fn export(
        &self,
        py: Python<'_>,
        result: PyRef<'_, TransportDecisionResultNative>,
        artifact_id: &str,
        seed: u64,
        threads: Option<u32>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        crate::catch_ffi(|| {
            let ctx = context(seed, threads, memory_bytes, cancel);
            let bytes = self.inner.export(&result.inner, artifact_id, &ctx)?;
            Ok(PyBytes::new(py, &bytes).unbind())
        })
    }
}

#[pymethods]
impl TransportDecisionResultNative {
    /// The route that produced this result.
    #[getter]
    fn route(&self) -> &'static str {
        self.inner.route()
    }

    /// Draws, decision, identities and calibration label, as JSON.
    fn summary_json(&self) -> String {
        self.inner.summary().to_string()
    }

    /// The propagation chain the decision read: `(parts, refusal)`.
    fn chain(&self, py: Python<'_>) -> Refused<ChainParts> {
        chain_parts(py, self.inner.chain())
    }
}

/// Compile a trial-to-target IPW decision on the target average effect
/// (`effect_json`) or on the two target outcome worlds (`worlds_json`).
#[pyfunction]
#[pyo3(signature = (graph, structure_json, rows_json, contract_json, effect_json=None, worlds_json=None, independent_potential_outcomes=false, *, catalog=None))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python declaration")]
fn trial_ipw_decision_compile(
    graph: PyRef<'_, Admg>,
    structure_json: &str,
    rows_json: &str,
    contract_json: &str,
    effect_json: Option<&str>,
    worlds_json: Option<&str>,
    independent_potential_outcomes: bool,
    catalog: Option<&Bound<'_, PyAny>>,
) -> PyResult<CompiledTransportDecisionNative> {
    crate::catch_ffi(|| {
        let (diagram, query) = structure(&graph, structure_json, catalog)?;
        check_declared_size(rows_json, MAX_DATA_DECLARATION_BYTES, "trial rows")?;
        let decision = contract_of(contract_json)?;
        let inner = match target_of(effect_json, worlds_json)? {
            Target::Effect(quantity) => {
                let rows: trial_ipw_transport_decision::TrialIpwRows =
                    serde_json::from_str(rows_json).map_err(json_err("trial rows"))?;
                CompiledDecision::TrialEffect(Box::new(
                    compile_trial_ipw_transport_decision(TrialIpwTransportDecisionRequest {
                        diagram,
                        query,
                        rows,
                        quantity,
                        decision,
                    })
                    .map_err(|error| raise(error.into(), Stage::Compile, None))?,
                ))
            }
            Target::Worlds(worlds) => {
                let rows: trial_ipw_predictive_decision::TrialIpwRows =
                    serde_json::from_str(rows_json).map_err(json_err("trial rows"))?;
                CompiledDecision::TrialPredictive(Box::new(
                    compile_trial_ipw_predictive_decision(TrialIpwPredictiveDecisionRequest {
                        diagram,
                        query,
                        rows,
                        worlds,
                        independent_potential_outcomes_given_mass: independent_potential_outcomes,
                        decision,
                    })
                    .map_err(|error| raise(error.into(), Stage::Compile, None))?,
                ))
            }
        };
        Ok(CompiledTransportDecisionNative { inner })
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JointSourceIn {
    id: String,
    population: String,
    identity: DataIdentityWire,
    treatment: Vec<bool>,
    outcome: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    noise_variance: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JointTargetIn {
    population: String,
    identity: DataIdentityWire,
    rows: usize,
    covariates: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JointRequestIn {
    features: Vec<String>,
    treatment: String,
    outcome: String,
    varying: String,
    sharing: String,
    dependence: String,
    invariant: GaussianPriorWire,
    varying_prior: GaussianPriorWire,
    sources: Vec<JointSourceIn>,
    target: JointTargetIn,
    draws: usize,
    seed: u64,
    basis_degree: usize,
    max_unsupported_mass: f64,
    conflict_z_threshold: f64,
}

fn data_identity(wire: DataIdentityWire, rows: usize) -> PyResult<DataIdentity> {
    if invalid_joint_data_identity(&wire, rows) {
        return Err(invalid_declaration("joint_transport.observation_identity_mismatch"));
    }
    Ok(DataIdentity { snapshot_digest: wire.snapshot_digest, datum_ids: wire.datum_ids })
}

fn gaussian_prior(wire: GaussianPriorWire) -> PyResult<GaussianPrior> {
    let provenance = match wire.provenance {
        PriorProvenanceWire::Declared => PriorProvenance::Declared,
        PriorProvenanceWire::Bank { bank_id, consumed } => {
            if bank_id.trim().is_empty() || consumed.is_empty() {
                return Err(invalid_declaration("joint_transport.prior_bank_lineage_missing"));
            }
            let consumed = consumed
                .into_iter()
                .map(|identity| {
                    let rows = identity.datum_ids.len();
                    data_identity(identity, rows)
                })
                .collect::<PyResult<Vec<_>>>()?;
            PriorProvenance::Bank { bank_id, consumed }
        }
    };
    Ok(GaussianPrior { mean: wire.mean, covariance: wire.covariance, provenance })
}

/// The joint Gaussian model, source trials, target frame and draw schedule of a
/// request, checked against the transport query it serves.
fn joint_inputs(
    graph: &Admg,
    query: &TransportQuery,
    request_json: &str,
) -> PyResult<(JointTransportModel, Vec<SourceData>, TargetData, JointTransportOptions)> {
    check_declared_size(request_json, MAX_DATA_DECLARATION_BYTES, "joint transport request")?;
    let request: JointRequestIn =
        serde_json::from_str(request_json).map_err(json_err("joint transport request"))?;
    // The model class is a fixed DAG; a graph with a latent district is another route.
    if (0..graph.admg.node_count()).any(|i| {
        antecedent::graph::DenseNodeId::try_from_usize(i)
            .is_ok_and(|node| !graph.admg.bidirected_neighbors(node).is_empty())
    }) {
        return Err(crate::refusal(
            reason_code!("route_not_supported"),
            "joint_transport.unsupported_graph",
        ));
    }
    let names = &graph.names;
    let Some((treatment, outcome)) = query.response.functional.primary_pair() else {
        return Err(invalid_declaration("joint_transport.query_mismatch"));
    };
    if resolve(names, &request.treatment)? != treatment
        || resolve(names, &request.outcome)? != outcome
    {
        return Err(invalid_declaration("joint_transport.query_mismatch"));
    }
    let source_population: &str = &query.source_population;
    let target_population: &str = &query.target_population;
    if request.target.population != target_population
        || request.sources.iter().any(|source| source.population != source_population)
    {
        return Err(invalid_declaration("joint_transport.population_mismatch"));
    }
    if request.basis_degree != 1 {
        return Err(invalid_declaration("joint_transport.basis_degree_invalid"));
    }
    let features = request
        .features
        .iter()
        .map(|name| resolve(names, name).map(VariableId::raw))
        .collect::<PyResult<Vec<_>>>()?;
    let varying = match request.varying.as_str() {
        "intercept" => VaryingBlock::Intercept,
        "intercept_and_covariates" => VaryingBlock::InterceptAndCovariates,
        _ => return Err(invalid_declaration("joint_transport.varying_invalid")),
    };
    let sharing = match request.sharing.as_str() {
        "independent_varying_blocks" => SourceSharing::IndependentVaryingBlocks,
        "shared_varying_block" => SourceSharing::SharedVaryingBlock,
        _ => return Err(invalid_declaration("joint_transport.sharing_invalid")),
    };
    let dependence = match request.dependence.as_str() {
        "independent_samples" => SourceDependence::IndependentSamples,
        "overlapping_units" => SourceDependence::OverlappingUnits,
        "unknown" => SourceDependence::Unknown,
        _ => return Err(invalid_declaration("joint_transport.dependence_invalid")),
    };
    let sources = request
        .sources
        .into_iter()
        .map(|source| {
            let rows = source.outcome.len();
            Ok(SourceData {
                id: source.id,
                identity: data_identity(source.identity, rows)?,
                treatment: source.treatment,
                outcome: source.outcome,
                covariates: source.covariates,
                noise_variance: source.noise_variance,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let target = TargetData {
        identity: data_identity(request.target.identity, request.target.rows)?,
        rows: request.target.rows,
        covariates: request.target.covariates,
    };
    let model = JointTransportModel {
        graph: TransportGraphClass::FixedDag,
        features,
        varying,
        sharing,
        dependence,
        priors: JointPriors {
            invariant: gaussian_prior(request.invariant)?,
            varying: gaussian_prior(request.varying_prior)?,
        },
        max_unsupported_mass: request.max_unsupported_mass,
        conflict_z_threshold: request.conflict_z_threshold,
    };
    let options = JointTransportOptions { draws: request.draws, seed: request.seed };
    Ok((model, sources, target, options))
}

/// Compile a joint source-target Gaussian transport decision on the target
/// average effect, or on the two target outcome worlds under a declared target
/// residual variance and cross-world disturbance correlation.
#[pyfunction]
#[pyo3(signature = (graph, structure_json, request_json, contract_json, effect_json=None, worlds_json=None, target_noise_variance=None, target_noise_correlation=None, *, catalog=None))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python declaration")]
fn joint_transport_decision_compile(
    graph: PyRef<'_, Admg>,
    structure_json: &str,
    request_json: &str,
    contract_json: &str,
    effect_json: Option<&str>,
    worlds_json: Option<&str>,
    target_noise_variance: Option<f64>,
    target_noise_correlation: Option<f64>,
    catalog: Option<&Bound<'_, PyAny>>,
) -> PyResult<CompiledTransportDecisionNative> {
    crate::catch_ffi(|| {
        let (diagram, query) = structure(&graph, structure_json, catalog)?;
        let (model, sources, target, options) = joint_inputs(&graph, &query, request_json)?;
        let decision = contract_of(contract_json)?;
        let declared = target_of(effect_json, worlds_json)?;
        let inner = match (declared, target_noise_variance, target_noise_correlation) {
            (Target::Effect(quantity), None, None) => CompiledDecision::JointEffect(Box::new(
                compile_joint_transport_effect_decision(JointTransportEffectDecisionRequest {
                    diagram,
                    query,
                    model,
                    sources,
                    target,
                    options,
                    quantity,
                    decision,
                })
                .map_err(|error| raise(error.into(), Stage::Compile, None))?,
            )),
            (Target::Worlds(worlds), Some(variance), Some(correlation)) => {
                CompiledDecision::JointPredictive(Box::new(
                    compile_joint_transport_predictive_decision(
                        JointTransportPredictiveDecisionRequest {
                            diagram,
                            query,
                            model,
                            sources,
                            target,
                            options,
                            target_noise_variance: variance,
                            target_noise_correlation: correlation,
                            worlds,
                            decision,
                        },
                    )
                    .map_err(|error| raise(error.into(), Stage::Compile, None))?,
                ))
            }
            (Target::Effect(_), _, _) => {
                return Err(invalid_declaration(
                    "an effect decision takes no target noise variance or correlation",
                ));
            }
            (Target::Worlds(_), _, _) => {
                return Err(invalid_declaration(
                    "outcome worlds need the target noise variance and the cross-world correlation",
                ));
            }
        };
        Ok(CompiledTransportDecisionNative { inner })
    })
}

/// Independently re-identify, refit and re-decide a sealed transported decision
/// under the program identity the consumer retained.
#[pyfunction]
#[pyo3(signature = (data, route, expected_identity, *, seed=1, threads=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "the declaration plus the execution controls")]
fn transport_decision_consume(
    data: &[u8],
    route: &str,
    expected_identity: &str,
    seed: u64,
    threads: Option<u32>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<TransportDecisionResultNative> {
    crate::catch_ffi(|| {
        let expected = expected_digest(expected_identity)?;
        let ctx = context(seed, threads, memory_bytes, cancel);
        let bind = Some(&ctx);
        let inner = match route {
            TRIAL_EFFECT => DecisionOutcome::TrialEffect(Box::new(
                consume_trial_ipw_transport_decision(data, expected, &ctx)
                    .map_err(|error| raise(error.into(), Stage::Bind, bind))?,
            )),
            TRIAL_PREDICTIVE => DecisionOutcome::TrialPredictive(Box::new(
                consume_trial_ipw_predictive_decision(data, expected, &ctx)
                    .map_err(|error| raise(error.into(), Stage::Bind, bind))?,
            )),
            JOINT_EFFECT => DecisionOutcome::JointEffect(Box::new(
                consume_joint_transport_effect_decision(data, expected, &ctx)
                    .map_err(|error| raise(error.into(), Stage::Bind, bind))?,
            )),
            JOINT_PREDICTIVE => DecisionOutcome::JointPredictive(Box::new(
                consume_joint_transport_predictive_decision(data, expected, &ctx)
                    .map_err(|error| raise(error.into(), Stage::Bind, bind))?,
            )),
            other => {
                return Err(unknown_route(
                    other,
                    &[TRIAL_EFFECT, TRIAL_PREDICTIVE, JOINT_EFFECT, JOINT_PREDICTIVE],
                ));
            }
        };
        Ok(TransportDecisionResultNative { inner })
    })
}

// ---------------------------------------------------------------------------
// Bundles that replay a transported fit.
// ---------------------------------------------------------------------------

/// Embed a sealed transported decision as a fit node: the builder re-identifies
/// and refits it under the program identity before the node is accepted.
#[pyfunction]
#[pyo3(signature = (builder, route, node_id, data, program_identity, *, seed=1, threads=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "the declaration plus the execution controls")]
fn transport_bundle_add_fit(
    mut builder: PyRefMut<'_, PyBundleBuilder>,
    route: &str,
    node_id: &str,
    data: &[u8],
    program_identity: &str,
    seed: u64,
    threads: Option<u32>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<()> {
    crate::catch_ffi(|| {
        let program = expected_digest(program_identity)?;
        let ctx = context(seed, threads, memory_bytes, cancel);
        let bind = Some(&ctx);
        match route {
            TRIAL_FIT => builder
                .inner
                .add_trial_ipw_transport_fit(node_id, data, program, &ctx)
                .map_err(|error| raise(error.into(), Stage::Bind, bind)),
            JOINT_FIT => builder
                .inner
                .add_joint_transport_effect_fit(node_id, data, program, &ctx)
                .map_err(|error| raise(error.into(), Stage::Bind, bind)),
            other => Err(unknown_route(other, &[TRIAL_FIT, JOINT_FIT])),
        }
    })
}

enum CompiledBundle {
    Trial(Box<CompiledTrialIpwNodeBundle>),
    Joint(Box<CompiledJointTransportEffectNodeBundle>),
}

/// A compiled bundle whose transported fit node feeds a live chain terminal.
#[pyclass(name = "CompiledTransportNodeBundleHandle", unsendable)]
struct CompiledTransportNodeBundleNative {
    inner: CompiledBundle,
}

fn hex(bytes: &[u8; 32]) -> String {
    SemanticDigest::from_bytes(*bytes).to_hex()
}

fn receipt_view(
    bundle: &antecedent::analysis::composition::ChainBundleRecomputationReceipt,
    producer: &[u8; 32],
    program: &[u8; 32],
) -> Json {
    json!({
        "bundle_identity": bundle.bundle_identity,
        "terminal_node": bundle.terminal_node,
        "terminal_identity": bundle.terminal_identity,
        "chain_program_identity": hex(&bundle.lowered_program_identity),
        "dependencies": bundle.dependencies.iter().map(|dependency| json!({
            "node_id": dependency.node_id,
            "artifact_identity": dependency.artifact_identity,
            "chain_digest": dependency.chain_digest,
        })).collect::<Vec<_>>(),
        "producer_program_identity": hex(producer),
        "program_identity": hex(program),
    })
}

fn trial_receipt(receipt: &TrialIpwNodeBundleReceipt) -> PyResult<(String, String)> {
    let view = receipt_view(
        &receipt.bundle,
        &receipt.producer_program_identity,
        &receipt.lowered_program_identity,
    );
    let wire = serde_json::to_string(receipt)
        .map_err(|error| crate::serialization_err(error.to_string()))?;
    Ok((view.to_string(), wire))
}

fn joint_receipt(receipt: &JointTransportEffectNodeBundleReceipt) -> PyResult<(String, String)> {
    let view = receipt_view(
        &receipt.bundle,
        &receipt.producer_program_identity,
        &receipt.lowered_program_identity,
    );
    let wire = serde_json::to_string(receipt)
        .map_err(|error| crate::serialization_err(error.to_string()))?;
    Ok((view.to_string(), wire))
}

#[pymethods]
impl CompiledTransportNodeBundleNative {
    /// `trial_ipw` or `joint_transport_effect`: the producer the bundle refits.
    #[getter]
    fn route(&self) -> &'static str {
        match self.inner {
            CompiledBundle::Trial(_) => TRIAL_FIT,
            CompiledBundle::Joint(_) => JOINT_FIT,
        }
    }

    /// The outer program identity a consumer retains, as lowercase hex.
    #[getter]
    fn identity(&self) -> String {
        match &self.inner {
            CompiledBundle::Trial(plan) => plan.lowered().identity().to_hex(),
            CompiledBundle::Joint(plan) => plan.lowered().identity().to_hex(),
        }
    }

    /// Every compiler layer of the outer program, as JSON, without fitting.
    fn plan_json(&self) -> String {
        match &self.inner {
            CompiledBundle::Trial(plan) => plan.plan_view().to_json().to_string(),
            CompiledBundle::Joint(plan) => plan.plan_view().to_json().to_string(),
        }
    }

    /// Refit the fit node through the bundle consumer and run the chain
    /// terminal. Returns `(receipt view json, receipt wire json)`.
    #[pyo3(signature = (*, seed=1, threads=None, memory_bytes=None, cancel=None))]
    fn execute(
        &self,
        seed: u64,
        threads: Option<u32>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(String, String)> {
        crate::catch_ffi(|| {
            let ctx = context(seed, threads, memory_bytes, cancel);
            let run = Some(&ctx);
            match &self.inner {
                CompiledBundle::Trial(plan) => trial_receipt(
                    &plan.execute(&ctx).map_err(|error| raise(error.into(), Stage::Run, run))?,
                ),
                CompiledBundle::Joint(plan) => joint_receipt(
                    &plan.execute(&ctx).map_err(|error| raise(error.into(), Stage::Run, run))?,
                ),
            }
        })
    }

    /// Re-execute, require the receipt and seal the bundle with it.
    #[pyo3(signature = (receipt_json, *, seed=1, threads=None, memory_bytes=None, cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        receipt_json: &str,
        seed: u64,
        threads: Option<u32>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        crate::catch_ffi(|| {
            check_declared_size(receipt_json, MAX_CONFIG_DECLARATION_BYTES, "bundle receipt")?;
            let ctx = context(seed, threads, memory_bytes, cancel);
            let bind = Some(&ctx);
            let bytes = match &self.inner {
                CompiledBundle::Trial(plan) => {
                    let receipt: TrialIpwNodeBundleReceipt =
                        serde_json::from_str(receipt_json).map_err(json_err("bundle receipt"))?;
                    plan.export_portable(&receipt, &ctx)
                        .map_err(|error| raise(error.into(), Stage::Bind, bind))?
                }
                CompiledBundle::Joint(plan) => {
                    let receipt: JointTransportEffectNodeBundleReceipt =
                        serde_json::from_str(receipt_json).map_err(json_err("bundle receipt"))?;
                    plan.export_portable(&receipt, &ctx)
                        .map_err(|error| raise(error.into(), Stage::Bind, bind))?
                }
            };
            Ok(PyBytes::new(py, &bytes).unbind())
        })
    }
}

/// Compile a bundle whose transported fit node is a direct dependency of a
/// propagation-chain terminal.
#[pyfunction]
#[pyo3(signature = (bundle, route, producer_node, terminal_node, *, seed=1, threads=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "the declaration plus the execution controls")]
fn transport_node_bundle_compile(
    bundle: PyRef<'_, PyCompositionBundle>,
    route: &str,
    producer_node: &str,
    terminal_node: &str,
    seed: u64,
    threads: Option<u32>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<CompiledTransportNodeBundleNative> {
    crate::catch_ffi(|| {
        let ctx = context(seed, threads, memory_bytes, cancel);
        let compile = Some(&ctx);
        let graph = bundle.bundle.clone();
        let inner = match route {
            TRIAL_FIT => CompiledBundle::Trial(Box::new(
                compile_trial_ipw_node_bundle_recomputation(
                    graph,
                    producer_node,
                    terminal_node,
                    &ctx,
                )
                .map_err(|error| raise(error.into(), Stage::Compile, compile))?,
            )),
            JOINT_FIT => CompiledBundle::Joint(Box::new(
                compile_joint_transport_effect_node_bundle_recomputation(
                    graph,
                    producer_node,
                    terminal_node,
                    &ctx,
                )
                .map_err(|error| raise(error.into(), Stage::Compile, compile))?,
            )),
            other => return Err(unknown_route(other, &[TRIAL_FIT, JOINT_FIT])),
        };
        Ok(CompiledTransportNodeBundleNative { inner })
    })
}

/// Independently recompile and replay a sealed transported-fit bundle under the
/// bundle identity and program identity the consumer retained. Returns
/// `(receipt view json, receipt wire json)`.
#[pyfunction]
#[pyo3(signature = (data, route, expected_bundle, expected_identity, *, seed=1, threads=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "the declaration plus the execution controls")]
fn transport_node_bundle_consume(
    data: &[u8],
    route: &str,
    expected_bundle: &str,
    expected_identity: &str,
    seed: u64,
    threads: Option<u32>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(String, String)> {
    crate::catch_ffi(|| {
        let expected = expected_digest(expected_identity)?;
        let ctx = context(seed, threads, memory_bytes, cancel);
        let bind = Some(&ctx);
        match route {
            TRIAL_FIT => trial_receipt(
                &consume_portable_trial_ipw_node_bundle_recomputation(
                    data,
                    expected_bundle,
                    expected,
                    &ctx,
                )
                .map_err(|error| raise(error.into(), Stage::Bind, bind))?,
            ),
            JOINT_FIT => joint_receipt(
                &consume_portable_joint_transport_effect_node_bundle_recomputation(
                    data,
                    expected_bundle,
                    expected,
                    &ctx,
                )
                .map_err(|error| raise(error.into(), Stage::Bind, bind))?,
            ),
            other => Err(unknown_route(other, &[TRIAL_FIT, JOINT_FIT])),
        }
    })
}

/// Consume an exported bundle with a consumer that re-identifies and refits its
/// transported fit nodes. Returns `(consumed json, refusal json)`.
#[pyfunction]
#[pyo3(signature = (data, route, expected_identity, supplied_json=None, *, seed=1, threads=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "the declaration plus the execution controls")]
fn transport_fit_bundle_consume(
    data: &[u8],
    route: &str,
    expected_identity: &str,
    supplied_json: Option<&str>,
    seed: u64,
    threads: Option<u32>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<String>> {
    crate::catch_ffi(|| {
        let supplied = crate::composition_bundle_api::supplied_from(supplied_json)?;
        let ctx = context(seed, threads, memory_bytes, cancel);
        let consumer = match route {
            TRIAL_FIT => trial_ipw_bundle_consumer(&ctx),
            JOINT_FIT => joint_transport_effect_bundle_consumer(&ctx),
            other => return Err(unknown_route(other, &[TRIAL_FIT, JOINT_FIT])),
        };
        Ok(match consumer.consume(data, &BundleLimits::default(), expected_identity, &supplied) {
            Ok(consumed) => (Some(crate::composition_bundle_api::consumed_json(&consumed)), None),
            Err(error) => (None, Some(crate::composition_bundle_api::error_refusal(&error))),
        })
    })
}

// ---------------------------------------------------------------------------
// Time-ordered path transport.
// ---------------------------------------------------------------------------

enum CompiledPath {
    Exact(Box<CompiledTemporalPathTransport>),
    Sampled(Box<CompiledSampledTemporalPathTransport>),
    Continuous(Box<CompiledContinuousChainTransport>),
}

enum PathOutcome {
    Exact(Box<TemporalPathTransportResult>),
    Sampled(Box<SampledTemporalPathTransportResult>),
    Continuous(Box<ContinuousChainTransportResult>),
}

/// A compiled time-ordered path contrast, before any law is evaluated.
#[pyclass(name = "CompiledTemporalPathHandle", unsendable)]
struct CompiledTemporalPathNative {
    inner: CompiledPath,
}

/// One evaluated path contrast; exported only through the program that made it.
#[pyclass(name = "TemporalPathResultHandle", unsendable)]
struct TemporalPathResultNative {
    inner: PathOutcome,
}

fn world_json(world: &antecedent::CounterfactualTransportResult) -> PyResult<Json> {
    let identity = world.identity().map_err(transport_error)?;
    Ok(json!({
        "probability": world.probability(),
        "numerator": world.numerator(),
        "denominator": world.denominator(),
        "cells_read": world.cells_read(),
        "premises_digest": identity.premises_digest,
        "data_digest": identity.data_digest,
    }))
}

impl PathOutcome {
    const fn kind(&self) -> &'static str {
        match self {
            Self::Exact(_) => PATH_EXACT,
            Self::Sampled(_) => PATH_SAMPLED,
            Self::Continuous(_) => PATH_CONTINUOUS,
        }
    }

    fn summary(&self) -> PyResult<Json> {
        Ok(match self {
            Self::Exact(result) => json!({
                "kind": PATH_EXACT,
                "program_identity": result.lowered_program_identity.to_hex(),
                "active": result.active.probability(),
                "baseline": result.baseline.probability(),
                "contrast": result.contrast,
                "worlds": {
                    "active": world_json(&result.active)?,
                    "baseline": world_json(&result.baseline)?,
                },
            }),
            Self::Sampled(result) => {
                let estimate = &result.estimate;
                json!({
                    "kind": PATH_SAMPLED,
                    "program_identity": result.lowered_program_identity.to_hex(),
                    "active": estimate.active,
                    "baseline": estimate.baseline,
                    "contrast": estimate.contrast,
                    "standard_error": estimate.standard_error.as_ref().map(|se| json!({
                        "value": se.value,
                        "label": se.label,
                        "method": se.method,
                        "gradient_relative_step": se.gradient_relative_step,
                        "gradient_evaluations": se.gradient_evaluations,
                        "datasets": se.datasets,
                    })),
                    "standard_error_withheld_reason": estimate.standard_error_withheld_reason,
                    "evidence_identity": estimate.evidence_identity,
                    "sampled_units": estimate.sampled_units,
                    "calibration": estimate.calibration,
                })
            }
            Self::Continuous(result) => json!({
                "kind": PATH_CONTINUOUS,
                "program_identity": result.lowered_program_identity.to_hex(),
                "active": result.active,
                "baseline": result.baseline,
                "contrast": result.contrast,
                "interval": [result.interval.0, result.interval.1],
                "calibration": result.calibration,
            }),
        })
    }
}

impl CompiledPath {
    const fn kind(&self) -> &'static str {
        match self {
            Self::Exact(_) => PATH_EXACT,
            Self::Sampled(_) => PATH_SAMPLED,
            Self::Continuous(_) => PATH_CONTINUOUS,
        }
    }

    fn identity(&self) -> SemanticDigest {
        match self {
            Self::Exact(plan) => plan.lowered().identity(),
            Self::Sampled(plan) => plan.lowered().identity(),
            Self::Continuous(plan) => plan.lowered().identity(),
        }
    }

    fn plan(&self) -> Json {
        match self {
            Self::Exact(plan) => plan.plan_view().to_json(),
            Self::Sampled(plan) => plan.plan_view().to_json(),
            Self::Continuous(plan) => plan.plan_view().to_json(),
        }
    }
}

#[pymethods]
impl CompiledTemporalPathNative {
    /// `exact`, `sampled` or `continuous_chain`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    /// The six-layer program identity a consumer retains, as lowercase hex.
    #[getter]
    fn identity(&self) -> String {
        self.inner.identity().to_hex()
    }

    /// Every compiler layer of the program, as JSON, without evaluating a law.
    fn plan_json(&self) -> String {
        self.inner.plan().to_string()
    }

    /// Evaluate both path worlds and their contrast.
    #[pyo3(signature = (*, seed=1, threads=None, memory_bytes=None, cancel=None))]
    fn execute(
        &self,
        seed: u64,
        threads: Option<u32>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<TemporalPathResultNative> {
        crate::catch_ffi(|| {
            let ctx = context(seed, threads, memory_bytes, cancel);
            let run = Some(&ctx);
            let inner = match &self.inner {
                CompiledPath::Exact(plan) => PathOutcome::Exact(Box::new(
                    plan.execute(&ctx).map_err(|error| raise(error.into(), Stage::Run, run))?,
                )),
                CompiledPath::Sampled(plan) => PathOutcome::Sampled(Box::new(
                    plan.execute(&ctx).map_err(|error| raise(error.into(), Stage::Run, run))?,
                )),
                CompiledPath::Continuous(plan) => PathOutcome::Continuous(Box::new(
                    plan.execute(&ctx).map_err(|error| raise(error.into(), Stage::Run, run))?,
                )),
            };
            Ok(TemporalPathResultNative { inner })
        })
    }

    /// Re-execute, require the same result and seal it as portable bytes.
    #[pyo3(signature = (result, *, seed=1, threads=None, memory_bytes=None, cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        result: PyRef<'_, TemporalPathResultNative>,
        seed: u64,
        threads: Option<u32>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        crate::catch_ffi(|| {
            let ctx = context(seed, threads, memory_bytes, cancel);
            let bind = Some(&ctx);
            let bytes = match (&self.inner, &result.inner) {
                (CompiledPath::Exact(plan), PathOutcome::Exact(outcome)) => plan
                    .export(outcome, &ctx)
                    .map_err(|error| raise(error.into(), Stage::Bind, bind))?,
                (CompiledPath::Sampled(plan), PathOutcome::Sampled(outcome)) => plan
                    .export(outcome, &ctx)
                    .map_err(|error| raise(error.into(), Stage::Bind, bind))?,
                (CompiledPath::Continuous(plan), PathOutcome::Continuous(outcome)) => plan
                    .export(outcome, &ctx)
                    .map_err(|error| raise(error.into(), Stage::Bind, bind))?,
                _ => {
                    return Err(crate::value_err_code(
                        reason_code!("invalid_argument"),
                        format!(
                            "the result is not one of this {} program's results",
                            self.inner.kind()
                        ),
                    ));
                }
            };
            Ok(PyBytes::new(py, &bytes).unbind())
        })
    }
}

#[pymethods]
impl TemporalPathResultNative {
    /// `exact`, `sampled` or `continuous_chain`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    /// Both worlds, the contrast, its uncertainty and identities, as JSON.
    fn summary_json(&self) -> PyResult<String> {
        Ok(self.inner.summary()?.to_string())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PathIn {
    treatment: String,
    outcome: String,
    control: f64,
    active: f64,
    outcome_level: f64,
    times: Vec<(String, u32)>,
    active_edges: Vec<(String, String)>,
}

/// The raw-id path specification; time labels are ordered by variable id.
fn path_spec(names: &[String], path_json: &str) -> PyResult<TemporalPathSpec> {
    check_declared_size(path_json, MAX_CONFIG_DECLARATION_BYTES, "temporal path")?;
    let path: PathIn = serde_json::from_str(path_json).map_err(json_err("temporal path"))?;
    let raw = |name: &str| resolve(names, name).map(VariableId::raw);
    let bits = |level: f64| (level + 0.0).to_bits();
    let mut times = path
        .times
        .iter()
        .map(|(name, time)| Ok((raw(name.as_str())?, *time)))
        .collect::<PyResult<Vec<_>>>()?;
    times.sort_unstable();
    Ok(TemporalPathSpec {
        treatment: raw(path.treatment.as_str())?,
        outcome: raw(path.outcome.as_str())?,
        control_bits: bits(path.control),
        active_bits: bits(path.active),
        outcome_level_bits: bits(path.outcome_level),
        times,
        active_edges: path
            .active_edges
            .iter()
            .map(|(from, to)| Ok((raw(from.as_str())?, raw(to.as_str())?)))
            .collect::<PyResult<Vec<_>>>()?,
    })
}

/// The finite problem, target, sources and catalog of a counterfactual
/// transport query, named against the graph.
fn finite_problem(
    graph: &Admg,
    levels: Vec<(String, Vec<f64>)>,
    sources: &[(String, Vec<String>, Vec<String>)],
    catalog: &Bound<'_, PyAny>,
) -> PyResult<(CounterfactualTransportProblem, Arc<[CounterfactualTransportSource]>, EvidenceCatalog)>
{
    let names = &graph.names;
    let declared = levels
        .into_iter()
        .map(|(name, values)| Ok((resolve(names, &name)?, values)))
        .collect::<PyResult<Vec<_>>>()?;
    let problem =
        CounterfactualTransportProblem::from_admg(&graph.aligned_to_names(names)?, &declared)
            .map_err(transport_error)?;
    let coordinates = |variables: &[String]| -> PyResult<Arc<[VariableId]>> {
        variables
            .iter()
            .map(|name| resolve(names, name))
            .collect::<PyResult<Vec<_>>>()
            .map(Arc::from)
    };
    let sources = sources
        .iter()
        .map(|(population, controllable, selections)| {
            Ok(CounterfactualTransportSource {
                population: Arc::from(population.as_str()),
                selection_targets: coordinates(selections)?,
                controllable: coordinates(controllable)?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok((problem, Arc::from(sources), parse_catalog(catalog, graph)?))
}

/// Compile a target-population path contrast on a time-ordered finite graph,
/// on exact regime laws (`laws`) or on sampled count tables (`tables`).
#[pyfunction]
#[pyo3(signature = (graph, levels, target, sources, catalog, path_json, *, laws=None, tables=None, overlap=None, unseen_rule=None, min_table_total=None, max_unseen_fraction=None, max_envelope_evaluations=None, max_operations=None, max_depth=None, max_search_operations=None, max_search_depth=None, max_support_rows=1_000_000, seed=1, threads=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python declaration")]
#[allow(clippy::too_many_lines, reason = "one decode of each declared argument, then one compile")]
fn temporal_path_compile(
    graph: PyRef<'_, Admg>,
    levels: Vec<(String, Vec<f64>)>,
    target: &str,
    sources: Vec<(String, Vec<String>, Vec<String>)>,
    catalog: &Bound<'_, PyAny>,
    path_json: &str,
    laws: Option<&Bound<'_, PyAny>>,
    tables: Option<&Bound<'_, PyAny>>,
    overlap: Option<OverlapTuple>,
    unseen_rule: Option<(String, Option<f64>)>,
    min_table_total: Option<u64>,
    max_unseen_fraction: Option<f64>,
    max_envelope_evaluations: Option<u32>,
    max_operations: Option<usize>,
    max_depth: Option<usize>,
    max_search_operations: Option<usize>,
    max_search_depth: Option<usize>,
    max_support_rows: usize,
    seed: u64,
    threads: Option<u32>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<CompiledTemporalPathNative> {
    crate::catch_ffi(|| {
        let (problem, sources, catalog) = finite_problem(&graph, levels, &sources, catalog)?;
        let spec = path_spec(&graph.names, path_json)?;
        let evaluation = ExactEvaluationLimits::default();
        let evaluation = ExactEvaluationLimits {
            operations: max_operations.unwrap_or(evaluation.operations),
            depth: max_depth.unwrap_or(evaluation.depth),
        };
        let options = CounterfactualTransportOptions {
            search: SearchLimits {
                operations: max_search_operations
                    .unwrap_or(COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS.operations),
                depth: max_search_depth.unwrap_or(COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS.depth),
            },
            evaluation,
            variable_names: graph.names.clone(),
        };
        let ctx = context(seed, threads, memory_bytes, cancel);
        let compile = Some(&ctx);
        let target = Arc::from(target);
        let inner = match (laws, tables) {
            (Some(laws), None) => {
                if overlap.is_some() || unseen_rule.is_some() {
                    return Err(invalid_declaration(
                        "exact laws take no sampling overlap or unseen-cell rule",
                    ));
                }
                let data =
                    parse_laws(laws, &catalog, &graph, max_support_rows, RegimeCheck::Strict)?;
                CompiledPath::Exact(Box::new(
                    compile_temporal_path_transport(
                        problem, target, sources, catalog, data, spec, options, &ctx,
                    )
                    .map_err(|error| raise(error.into(), Stage::Compile, compile))?,
                ))
            }
            (None, Some(tables)) => {
                let mut parsed = Vec::new();
                for table in tables.try_iter()? {
                    parsed.push(parse_count_table(&table?, &catalog, &graph)?);
                }
                let overlap = overlap
                    .ok_or_else(|| invalid_declaration("sampled tables need a declared overlap"))?;
                let defaults = SampledTransportLimits::default();
                let request = SampledTransportRequest {
                    tables: parsed,
                    overlap: parse_overlap(&overlap, &catalog, &graph)?,
                    unseen_rule: parse_unseen_rule(
                        &unseen_rule.unwrap_or_else(|| ("refuse".to_owned(), None)),
                    )?,
                    limits: SampledTransportLimits {
                        min_table_total: min_table_total.unwrap_or(defaults.min_table_total),
                        max_unseen_fraction: max_unseen_fraction
                            .unwrap_or(defaults.max_unseen_fraction),
                        max_envelope_evaluations: max_envelope_evaluations
                            .unwrap_or(defaults.max_envelope_evaluations),
                        evaluation,
                    },
                };
                CompiledPath::Sampled(Box::new(
                    compile_sampled_temporal_path_transport(
                        problem, target, sources, catalog, request, spec, options, &ctx,
                    )
                    .map_err(|error| raise(error.into(), Stage::Compile, compile))?,
                ))
            }
            _ => {
                return Err(invalid_declaration(
                    "supply either exact laws or sampled count tables",
                ));
            }
        };
        Ok(CompiledTemporalPathNative { inner })
    })
}

/// Compile the restricted continuous sequential-edge chain and its paired
/// bootstrap schedule; the seed becomes part of the program.
#[pyfunction]
#[pyo3(signature = (request_json, *, seed=1, threads=None, memory_bytes=None, cancel=None))]
fn continuous_chain_compile(
    request_json: &str,
    seed: u64,
    threads: Option<u32>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<CompiledTemporalPathNative> {
    crate::catch_ffi(|| {
        check_declared_size(request_json, MAX_DATA_DECLARATION_BYTES, "continuous chain")?;
        let request: ContinuousChainTransportRequest =
            serde_json::from_str(request_json).map_err(json_err("continuous chain"))?;
        let ctx = context(seed, threads, memory_bytes, cancel);
        let plan = compile_continuous_chain_transport(request, &ctx)
            .map_err(|error| raise(error.into(), Stage::Compile, Some(&ctx)))?;
        Ok(CompiledTemporalPathNative { inner: CompiledPath::Continuous(Box::new(plan)) })
    })
}

/// Independently replay a sealed path contrast under the program identity the
/// consumer retained. A continuous chain replays only under its compile seed.
#[pyfunction]
#[pyo3(signature = (data, kind, expected_identity, *, seed=1, threads=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "the declaration plus the execution controls")]
fn temporal_path_consume(
    data: &[u8],
    kind: &str,
    expected_identity: &str,
    seed: u64,
    threads: Option<u32>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<TemporalPathResultNative> {
    crate::catch_ffi(|| {
        let expected = expected_digest(expected_identity)?;
        let ctx = context(seed, threads, memory_bytes, cancel);
        let bind = Some(&ctx);
        let inner = match kind {
            PATH_EXACT => PathOutcome::Exact(Box::new(
                consume_temporal_path_transport(data, expected, &ctx)
                    .map_err(|error| raise(error.into(), Stage::Bind, bind))?,
            )),
            PATH_SAMPLED => PathOutcome::Sampled(Box::new(
                consume_sampled_temporal_path_transport(data, expected, &ctx)
                    .map_err(|error| raise(error.into(), Stage::Bind, bind))?,
            )),
            PATH_CONTINUOUS => PathOutcome::Continuous(Box::new(
                consume_continuous_chain_transport(data, expected, &ctx)
                    .map_err(|error| raise(error.into(), Stage::Bind, bind))?,
            )),
            other => {
                return Err(unknown_route(other, &[PATH_EXACT, PATH_SAMPLED, PATH_CONTINUOUS]));
            }
        };
        Ok(TemporalPathResultNative { inner })
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CompiledTransportDecisionNative>()?;
    module.add_class::<TransportDecisionResultNative>()?;
    module.add_class::<CompiledTransportNodeBundleNative>()?;
    module.add_class::<CompiledTemporalPathNative>()?;
    module.add_class::<TemporalPathResultNative>()?;
    module.add_function(wrap_pyfunction!(trial_ipw_decision_compile, module)?)?;
    module.add_function(wrap_pyfunction!(joint_transport_decision_compile, module)?)?;
    module.add_function(wrap_pyfunction!(transport_decision_consume, module)?)?;
    module.add_function(wrap_pyfunction!(transport_bundle_add_fit, module)?)?;
    module.add_function(wrap_pyfunction!(transport_node_bundle_compile, module)?)?;
    module.add_function(wrap_pyfunction!(transport_node_bundle_consume, module)?)?;
    module.add_function(wrap_pyfunction!(transport_fit_bundle_consume, module)?)?;
    module.add_function(wrap_pyfunction!(temporal_path_compile, module)?)?;
    module.add_function(wrap_pyfunction!(continuous_chain_compile, module)?)?;
    module.add_function(wrap_pyfunction!(temporal_path_consume, module)?)?;
    Ok(())
}
