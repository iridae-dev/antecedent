// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cyclic decisions: connected feedback components, paired action worlds and bundle refits.
//!
//! Python declares components, couplings, worlds, output coordinates, decision contracts and
//! posterior schedules as JSON; every port check, fit, certified solve, decision, artifact and
//! refusal is Rust's. A compiled decision binds the execution seed when it is compiled, so its
//! execution, its export and an independent consumer all run under that seed. The fitted child
//! chains compile under the caller's context, which is passed through unchanged. Nothing here
//! claims an interval, a coverage or a calibrated decision.

use std::sync::Arc;

use antecedent::analysis::composition::{
    BundleLimits, CompiledCyclicPosteriorBundleRecomputation, CompiledCyclicPosteriorNodeBundle,
    CyclicPosteriorBundleError, CyclicPosteriorBundleReceipt,
    compile_cyclic_posterior_bundle_recomputation,
    compile_cyclic_posterior_node_bundle_recomputation,
    consume_portable_cyclic_posterior_bundle_recomputation,
    consume_portable_cyclic_posterior_node_bundle_recomputation, cyclic_posterior_bundle_consumer,
    native_fit_bundle_consumer,
};
use antecedent::analysis::cyclic_action_world_decision::{
    CompiledCyclicActionDecision, CyclicActionDecisionResult, CyclicDecisionOutput,
    compile_cyclic_action_world_decision, consume_cyclic_action_world_decision,
};
use antecedent::analysis::cyclic_component_composition::{
    BoundedComponentPosteriorDecisionResult, CompiledBoundedComponentPosteriorDecision,
    CompiledJointComponentPosteriorDecision, CompiledJointVectorComponentPosteriorDecision,
    ComponentJointPosteriorSchedule, JointComponentPosteriorDecisionResult,
    compile_bounded_component_posterior_decision, compile_joint_component_posterior_decision,
    compile_joint_vector_component_posterior_decision,
    consume_bounded_component_posterior_decision, consume_joint_component_posterior_decision,
    consume_joint_vector_component_posterior_decision, prepare_component_feedback,
    prepare_component_feedback_with_ledger, prepare_vector_component_feedback,
    prepare_vector_component_feedback_with_ledger,
};
use antecedent::analysis::cyclic_posterior_action_decision::{
    CompiledCyclicPosteriorDecision, CyclicPosteriorDecisionRequest, CyclicPosteriorDecisionResult,
    compile_cyclic_posterior_action_decision, consume_cyclic_posterior_action_decision,
};
use antecedent::analysis::cyclic_target_feedback_decision::{
    CompiledCyclicTargetFeedbackDecision, CyclicTargetFeedbackRequest, CyclicTargetFeedbackResult,
    TargetFeedbackFrame, compile_cyclic_target_feedback_decision,
    consume_cyclic_target_feedback_decision,
};
use antecedent::analysis::interface_contract::{ComponentConnection, InterfaceContract};
use antecedent::analysis::{
    ChainDecisionResult, ComponentFeedbackRequest, ComponentFeedbackWorld, FeedbackConnection,
    PreparedComponentFeedback, ScalarFeedbackComponent, VectorFeedbackComponent,
    VectorFeedbackConnection, VectorFeedbackOutput, VectorFeedbackRequest,
};
use antecedent::design::decision_eval::{ActionOutcome, Verdict};
use antecedent::prelude::ExecutionContext;
use antecedent::scientific::SemanticDigest;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::{Value as Json, json};

use crate::composition_bundle_api::{PyBundleBuilder, PyCompositionBundle};
use crate::cyclic_system_api::PyPreparedCyclicSystem;
use crate::evidence_ledger_api::EvidenceLedgerHandle;

type Contract = antecedent::analysis::inverse_query::DecisionContract;
type Failure = Box<dyn std::error::Error + Send + 'static>;

/// Feature names the portable decision and bundle artifacts declare, read only to select the
/// consumer that then verifies the whole artifact.
const ACTION_WORLD: &str = "cyclic_derived_action_world_decision_v1";
const POSTERIOR: &str = "cyclic_posterior_action_decision_v1";
const TARGET_FEEDBACK: &str = "cyclic_target_feedback_decision_v1";
const COMPONENT: &str = "joint_component_posterior_decision_v1";
const VECTOR_COMPONENT: &str = "joint_vector_component_posterior_decision_v1";
const BOUNDED: &str = "bounded_joint_component_posterior_decision_v1";
const BUNDLE: &str = "cyclic_posterior_bundle_recomputation_v1";
const NODE_BUNDLE: &str = "cyclic_posterior_node_bundle_recomputation_v1";

fn fail<E: std::error::Error + Send + 'static>(error: E) -> Failure {
    Box::new(error)
}

fn other(message: &'static str) -> Failure {
    Box::new(std::io::Error::other(message))
}

fn context(
    seed: u64,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> ExecutionContext {
    let mut ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(None), cancel);
    if max_bytes.is_some() {
        ctx.set_memory_limit(max_bytes);
    }
    ctx
}

fn parse<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> PyResult<T> {
    crate::bridge::check_declared_size(text, crate::bridge::MAX_DATA_DECLARATION_BYTES, what)?;
    serde_json::from_str(text).map_err(|error| crate::value_err(format!("invalid {what}: {error}")))
}

fn contract_of(text: &str) -> PyResult<Contract> {
    crate::bridge::check_declared_size(
        text,
        crate::bridge::MAX_DECLARED_ARTIFACT_BYTES,
        "decision contract",
    )?;
    antecedent::design::decision_artifact::contract_from_json_refusal(text).map_err(|refusal| {
        let message = refusal.offending.as_ref().map_or_else(
            || refusal.detail.clone(),
            |offending| format!("{} at {offending}", refusal.detail),
        );
        crate::with_remedy(crate::refusal(refusal.code, message), refusal.remedy)
    })
}

fn invalid(what: &'static str) -> impl Fn(serde_json::Error) -> PyErr {
    move |error| crate::value_err(format!("invalid {what}: {error}"))
}

fn scientific_refusal(code: &str, detail: &str, message: &str, next_attempt: &str) -> PyErr {
    Python::attach(|py| {
        let build = || -> PyResult<PyErr> {
            let fields = pyo3::types::PyDict::new(py);
            fields.set_item("code", code)?;
            fields.set_item("detail", detail)?;
            fields.set_item("message", message)?;
            fields.set_item("remedy", next_attempt)?;
            fields.set_item("next_attempt", next_attempt)?;
            let class = pyo3::types::PyModule::import(py, "antecedent.errors")?
                .getattr("StructuredRefusal")?;
            Ok(PyErr::from_value(class.call1((fields,))?))
        };
        build().unwrap_or_else(|error| error)
    })
}

/// The Python error of a refused compile, run or replay: cancellation, resource and
/// release-support refusals keep their own classes; any other refusal of a declaration or run
/// is an invalid value, and any other refusal of a replay is a serialization failure.
fn classify(
    error: &(dyn std::error::Error + 'static),
    ctx: &ExecutionContext,
    replay: bool,
) -> PyErr {
    use antecedent::gcm::nonlinear_equilibrium::EquilibriumError;
    use antecedent::io::cyclic_system_artifact::CyclicArtifactError;
    let message = error.to_string();
    if ctx.cancellation.is_cancelled() {
        return crate::CausalCancelledError::new_err(message);
    }
    let mut cause = Some(error);
    while let Some(current) = cause {
        if let Some(refusal) =
            current.downcast_ref::<antecedent::execution::support_gate::NotSupported>()
        {
            return crate::not_supported_err(refusal);
        }
        match current.downcast_ref::<CyclicArtifactError>() {
            Some(CyclicArtifactError::ScientificRefusal {
                code,
                detail,
                message: why,
                next_attempt,
            }) => {
                return scientific_refusal(code, detail, why, next_attempt);
            }
            Some(CyclicArtifactError::Cancelled) => {
                return crate::CausalCancelledError::new_err(message);
            }
            Some(
                CyclicArtifactError::Admission(_)
                | CyclicArtifactError::Execution(EquilibriumError::Resource(_)),
            ) => return crate::CausalResourceError::new_err(message),
            _ => {}
        }
        if antecedent::execution::is_admission_refusal(current)
            || matches!(
                current.downcast_ref::<EquilibriumError>(),
                Some(EquilibriumError::Resource(_))
            )
        {
            return crate::CausalResourceError::new_err(message);
        }
        cause = current.source();
    }
    if replay { crate::serialization_err(message) } else { crate::value_err(message) }
}

#[derive(Deserialize)]
struct FeaturePeek {
    #[serde(default)]
    required_feature: Option<String>,
    #[serde(default)]
    feature: Option<String>,
}

/// The feature a portable artifact declares; the selected consumer checks everything else.
fn declared_feature(bytes: &[u8]) -> Result<String, Failure> {
    let peek: FeaturePeek = antecedent::io::from_cbor(bytes).map_err(fail)?;
    Ok(peek.required_feature.or(peek.feature).unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Connected feedback components.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FeedbackIn {
    components: Vec<ComponentIn>,
    connections: Vec<ConnectionIn>,
    noise_covariance: Vec<f64>,
    noise: Vec<Vec<f64>>,
    initial: Vec<f64>,
    worlds: Vec<Vec<(String, f64)>>,
    policy: Json,
    #[serde(default)]
    factual: Option<Vec<f64>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentIn {
    interface: Json,
    #[serde(default)]
    equation: Option<Json>,
    #[serde(default)]
    outputs: Option<Vec<OutputIn>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputIn {
    produced_port: String,
    equation: Json,
    consumed_ports: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConnectionIn {
    producer: String,
    produced_port: String,
    consumer: String,
    consumed_port: String,
    coefficient: f64,
    #[serde(default)]
    consumer_output: Option<String>,
}

#[allow(
    clippy::large_enum_variant,
    reason = "one owner per Python handle; boxing a variant would only add an indirection"
)]
enum FeedbackRequest {
    Scalar(ComponentFeedbackRequest),
    Vector(VectorFeedbackRequest),
}

fn interface_of(value: &Json) -> PyResult<InterfaceContract> {
    crate::interface_contract_api::contract_of(&value.to_string(), "component interface")
}

fn feedback_request(text: &str) -> PyResult<FeedbackRequest> {
    let wire: FeedbackIn = parse(text, "component feedback declaration")?;
    if wire
        .components
        .iter()
        .any(|component| component.equation.is_some() == component.outputs.is_some())
    {
        return Err(crate::value_err(
            "each component declares exactly one of a scalar equation or its output equations",
        ));
    }
    let multi = wire.components.iter().filter(|component| component.outputs.is_some()).count();
    if multi != 0 && multi != wire.components.len() {
        return Err(crate::value_err(
            "components are either all scalar or all multi-output in one connected system",
        ));
    }
    let worlds: Vec<ComponentFeedbackWorld> =
        wire.worlds.into_iter().map(|clamps| ComponentFeedbackWorld { clamps }).collect();
    if multi == 0 {
        let components = wire
            .components
            .into_iter()
            .map(|component| -> PyResult<ScalarFeedbackComponent> {
                Ok(ScalarFeedbackComponent {
                    interface: interface_of(&component.interface)?,
                    equation: serde_json::from_value(component.equation.unwrap_or(Json::Null))
                        .map_err(invalid("component equation"))?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let connections = wire
            .connections
            .into_iter()
            .map(|connection| -> PyResult<FeedbackConnection> {
                if connection.consumer_output.is_some() {
                    return Err(crate::value_err(
                        "a scalar component connection names no consumer output",
                    ));
                }
                Ok(FeedbackConnection {
                    ports: ComponentConnection::new(
                        connection.producer,
                        connection.produced_port,
                        connection.consumer,
                        connection.consumed_port,
                    ),
                    coefficient: connection.coefficient,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(FeedbackRequest::Scalar(ComponentFeedbackRequest {
            components,
            connections,
            noise_covariance: wire.noise_covariance,
            noise: wire.noise,
            initial: wire.initial,
            worlds,
            policy: serde_json::from_value(wire.policy).map_err(invalid("solve policy"))?,
            factual: wire.factual,
        }));
    }
    let components = wire
        .components
        .into_iter()
        .map(|component| -> PyResult<VectorFeedbackComponent> {
            let outputs = component
                .outputs
                .unwrap_or_default()
                .into_iter()
                .map(|output| -> PyResult<VectorFeedbackOutput> {
                    Ok(VectorFeedbackOutput {
                        produced_port: output.produced_port,
                        equation: serde_json::from_value(output.equation)
                            .map_err(invalid("output equation"))?,
                        consumed_ports: output.consumed_ports,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(VectorFeedbackComponent { interface: interface_of(&component.interface)?, outputs })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let connections = wire
        .connections
        .into_iter()
        .map(|connection| -> PyResult<VectorFeedbackConnection> {
            let consumer_output = connection.consumer_output.ok_or_else(|| {
                crate::value_err("a multi-output connection names the consumer output it feeds")
            })?;
            Ok(VectorFeedbackConnection {
                ports: ComponentConnection::new(
                    connection.producer,
                    connection.produced_port,
                    connection.consumer,
                    connection.consumed_port,
                ),
                consumer_output,
                coefficient: connection.coefficient,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(FeedbackRequest::Vector(VectorFeedbackRequest {
        components,
        connections,
        noise_covariance: wire.noise_covariance,
        noise: wire.noise,
        initial: wire.initial,
        worlds,
        policy: serde_json::from_value(wire.policy).map_err(invalid("solve policy"))?,
        factual: wire.factual,
    }))
}

/// A connected system of feedback components prepared through the central cyclic compiler,
/// with the declaration it was prepared from.
#[pyclass(name = "_PreparedComponentFeedback", frozen)]
pub(crate) struct PyPreparedComponentFeedback {
    request: FeedbackRequest,
    prepared: PreparedComponentFeedback,
}

#[pymethods]
impl PyPreparedComponentFeedback {
    /// Dense variable order of the connected system (`component` or `component::port`).
    fn component_order(&self) -> Vec<String> {
        self.prepared.component_order().to_vec()
    }

    /// Identity of the checked interfaces and structural couplings, as lowercase hex.
    fn composition_identity(&self) -> String {
        self.prepared.composition_identity().to_hex()
    }

    /// Identity of the connected six-layer cyclic program, as lowercase hex.
    fn identity(&self) -> String {
        self.prepared.prepared().lowered().identity().to_hex()
    }

    /// Whether the components declare several output equations each.
    fn multi_output(&self) -> bool {
        matches!(self.request, FeedbackRequest::Vector(_))
    }

    /// The connected system as an ordinary prepared cyclic system handle.
    fn system(&self) -> PyPreparedCyclicSystem {
        PyPreparedCyclicSystem { inner: self.prepared.prepared().clone() }
    }
}

/// Check every component port and coupling, then prepare the connected system as one cyclic
/// program (`prepare_component_feedback`, `prepare_vector_component_feedback` and their ledger
/// forms). A ledger proves disjoint-evidence inputs and binds its state into the identity.
#[pyfunction]
#[pyo3(signature = (declaration_json, ledger=None, *, seed, max_bytes=None, cancel=None))]
fn cyclic_component_feedback_prepare(
    py: Python<'_>,
    declaration_json: &str,
    ledger: Option<PyRef<'_, EvidenceLedgerHandle>>,
    seed: u64,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedComponentFeedback> {
    crate::catch_ffi(|| {
        let request = feedback_request(declaration_json)?;
        let ctx = context(seed, max_bytes, cancel);
        let ledger = ledger.as_deref().map(EvidenceLedgerHandle::ledger);
        let prepared = match &request {
            FeedbackRequest::Scalar(declared) => {
                let declared = declared.clone();
                py.detach(|| match ledger {
                    Some(ledger) => prepare_component_feedback_with_ledger(declared, ledger, &ctx),
                    None => prepare_component_feedback(declared, &ctx),
                })
            }
            FeedbackRequest::Vector(declared) => {
                let declared = declared.clone();
                py.detach(|| match ledger {
                    Some(ledger) => {
                        prepare_vector_component_feedback_with_ledger(declared, ledger, &ctx)
                    }
                    None => prepare_vector_component_feedback(declared, &ctx),
                })
            }
        }
        .map_err(|error| classify(&error, &ctx, false))?;
        Ok(PyPreparedComponentFeedback { request, prepared })
    })
}

// ---------------------------------------------------------------------------
// Compiled decisions and their results.
// ---------------------------------------------------------------------------

#[allow(
    clippy::large_enum_variant,
    reason = "one owner per Python handle; boxing a variant would only add an indirection"
)]
enum Compiled {
    ActionWorld(CompiledCyclicActionDecision),
    Posterior(CompiledCyclicPosteriorDecision),
    TargetFeedback(CompiledCyclicTargetFeedbackDecision),
    Component(CompiledJointComponentPosteriorDecision),
    VectorComponent(CompiledJointVectorComponentPosteriorDecision),
    Bounded(CompiledBoundedComponentPosteriorDecision),
}

impl Compiled {
    fn identity(&self) -> SemanticDigest {
        match self {
            Self::ActionWorld(plan) => plan.lowered().identity(),
            Self::Posterior(plan) => plan.lowered().identity(),
            Self::TargetFeedback(plan) => plan.lowered().identity(),
            Self::Component(plan) => plan.lowered().identity(),
            Self::VectorComponent(plan) => plan.lowered().identity(),
            Self::Bounded(plan) => plan.lowered().identity(),
        }
    }

    fn plan_view(&self) -> antecedent::planner::PlanView {
        match self {
            Self::ActionWorld(plan) => plan.plan_view(),
            Self::Posterior(plan) => plan.plan_view(),
            Self::TargetFeedback(plan) => plan.plan_view(),
            Self::Component(plan) => plan.plan_view(),
            Self::VectorComponent(plan) => plan.plan_view(),
            Self::Bounded(plan) => plan.plan_view(),
        }
    }

    fn route(&self) -> &'static str {
        self.plan_view().route().as_str()
    }

    const fn feature(&self) -> &'static str {
        match self {
            Self::ActionWorld(_) => ACTION_WORLD,
            Self::Posterior(_) => POSTERIOR,
            Self::TargetFeedback(_) => TARGET_FEEDBACK,
            Self::Component(_) => COMPONENT,
            Self::VectorComponent(_) => VECTOR_COMPONENT,
            Self::Bounded(_) => BOUNDED,
        }
    }

    fn execute(&self, ctx: &ExecutionContext) -> Result<Outcome, Failure> {
        Ok(match self {
            Self::ActionWorld(plan) => Outcome::ActionWorld(plan.execute(ctx).map_err(fail)?),
            Self::Posterior(plan) => Outcome::Posterior(plan.execute(ctx).map_err(fail)?),
            Self::TargetFeedback(plan) => Outcome::TargetFeedback(plan.execute(ctx).map_err(fail)?),
            Self::Component(plan) => Outcome::Component(plan.execute(ctx).map_err(fail)?),
            Self::VectorComponent(plan) => Outcome::Component(plan.execute(ctx).map_err(fail)?),
            Self::Bounded(plan) => Outcome::Bounded(plan.execute(ctx).map_err(fail)?),
        })
    }

    fn export(
        &self,
        outcome: &Outcome,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, Failure> {
        match (self, outcome) {
            (Self::ActionWorld(plan), Outcome::ActionWorld(result)) => {
                plan.export(result, artifact_id, ctx).map_err(fail)
            }
            (Self::Posterior(plan), Outcome::Posterior(result)) => {
                plan.export(result, artifact_id, ctx).map_err(fail)
            }
            (Self::TargetFeedback(plan), Outcome::TargetFeedback(result)) => {
                plan.export(result, artifact_id, ctx).map_err(fail)
            }
            (Self::Component(plan), Outcome::Component(result)) => {
                plan.export(result, ctx).map_err(fail)
            }
            (Self::VectorComponent(plan), Outcome::Component(result)) => {
                plan.export(result, ctx).map_err(fail)
            }
            (Self::Bounded(plan), Outcome::Bounded(result)) => {
                plan.export(result, ctx).map_err(fail)
            }
            _ => Err(other("the result was not produced by this compiled decision")),
        }
    }
}

#[allow(
    clippy::large_enum_variant,
    reason = "one owner per Python handle; boxing a variant would only add an indirection"
)]
enum Outcome {
    ActionWorld(CyclicActionDecisionResult),
    Posterior(CyclicPosteriorDecisionResult),
    TargetFeedback(CyclicTargetFeedbackResult),
    Component(JointComponentPosteriorDecisionResult),
    Bounded(BoundedComponentPosteriorDecisionResult),
}

fn outcome_value(outcome: &ActionOutcome) -> Json {
    json!({
        "id": outcome.id,
        "admissible": outcome.admissible,
        "exclusions": outcome
            .exclusions
            .iter()
            .map(|exclusion| json!({
                "constraint_id": exclusion.constraint_id,
                "probability": exclusion.probability,
                "required": exclusion.required,
            }))
            .collect::<Vec<_>>(),
        "expected_utility": outcome.expected_utility,
        "value": outcome.value,
        "standard_error": outcome.standard_error,
        "expected_regret": outcome.expected_regret,
        "max_regret": outcome.max_regret,
    })
}

fn verdict_value(verdict: &Verdict) -> Json {
    match verdict {
        Verdict::UniquelyOptimal(action) => {
            json!({"kind": "uniquely_optimal", "actions": [action]})
        }
        Verdict::Indistinguishable(actions) => {
            json!({"kind": "indistinguishable", "actions": actions})
        }
        Verdict::NoAdmissibleAction => json!({"kind": "no_admissible_action", "actions": []}),
    }
}

fn decision_value(result: &ChainDecisionResult) -> Json {
    let decision = &result.decision;
    json!({
        "contract_identity": decision.contract_identity,
        "criterion": decision.criterion_label,
        "outcomes": decision.actions.iter().map(outcome_value).collect::<Vec<_>>(),
        "verdict": verdict_value(&decision.verdict),
        "evpi": decision.evpi,
        "draws": decision.n_draws,
        "effective_draws": decision.effective_draws,
        "assumptions": decision.assumptions,
        "chain_identity": result.chain.identity(),
        "program_identity": result.lowered_program_identity.to_hex(),
    })
}

/// The posterior's chain counts and convergence diagnostics; the draws are read separately.
fn posterior_value<P: serde::Serialize>(posterior: &P) -> PyResult<Json> {
    let mut value = serde_json::to_value(posterior)
        .map_err(|error| crate::serialization_err(error.to_string()))?;
    if let Some(map) = value.as_object_mut() {
        map.remove("samples");
    }
    Ok(value)
}

impl Outcome {
    fn decision(&self) -> &ChainDecisionResult {
        match self {
            Self::ActionWorld(result) => &result.decision,
            Self::Posterior(result) => &result.decision,
            Self::TargetFeedback(result) => &result.decision,
            Self::Component(result) => &result.child.decision,
            Self::Bounded(result) => &result.decision,
        }
    }

    fn identity(&self) -> SemanticDigest {
        match self {
            Self::ActionWorld(result) => result.lowered_program_identity,
            Self::Posterior(result) => result.lowered_program_identity,
            Self::TargetFeedback(result) => result.lowered_program_identity,
            Self::Component(result) => result.lowered_program_identity,
            Self::Bounded(result) => result.lowered_program_identity,
        }
    }

    fn calibration(&self) -> &str {
        match self {
            Self::ActionWorld(result) => &result.cyclic.calibration,
            Self::TargetFeedback(result) => result.calibration,
            Self::Bounded(result) => result.calibration,
            Self::Posterior(_) | Self::Component(_) => "not_measured",
        }
    }

    fn posterior(&self) -> PyResult<Json> {
        match self {
            Self::ActionWorld(_) => Ok(Json::Null),
            Self::Posterior(result) => posterior_value(&result.posterior),
            Self::TargetFeedback(result) => posterior_value(&result.posterior),
            Self::Component(result) => posterior_value(&result.child.posterior),
            Self::Bounded(result) => posterior_value(&result.inner.child.posterior),
        }
    }

    fn posterior_samples(&self) -> Option<Vec<Vec<f64>>> {
        match self {
            Self::ActionWorld(_) => None,
            Self::Posterior(result) => Some(result.posterior.samples.clone()),
            Self::TargetFeedback(result) => Some(result.posterior.samples.clone()),
            Self::Component(result) => Some(result.child.posterior.samples.clone()),
            Self::Bounded(result) => Some(result.inner.child.posterior.samples.clone()),
        }
    }

    fn summary(&self, feature: &str) -> PyResult<Json> {
        let real_valued = match self {
            Self::Bounded(result) => decision_value(&result.inner.child.decision),
            _ => Json::Null,
        };
        let cyclic = match self {
            Self::ActionWorld(result) => json!({
                "identity": result.cyclic.identity,
                "world_count": result.cyclic.worlds.len(),
                "claim": result.cyclic.claim,
            }),
            _ => Json::Null,
        };
        Ok(json!({
            "kind": feature,
            "identity": self.identity().to_hex(),
            "decision": decision_value(self.decision()),
            "real_valued_decision": real_valued,
            "posterior": self.posterior()?,
            "calibration": self.calibration(),
            "cyclic": cyclic,
        }))
    }
}

/// A compiled cyclic decision: its six-layer program and the execution seed it binds.
#[pyclass(name = "_CompiledCyclicDecision", frozen)]
pub(crate) struct PyCompiledCyclicDecision {
    compiled: Arc<Compiled>,
    seed: u64,
}

#[pymethods]
impl PyCompiledCyclicDecision {
    /// Identity of the complete compiled program, as lowercase hex.
    fn identity(&self) -> String {
        self.compiled.identity().to_hex()
    }

    /// The compiled route name of the program.
    fn route(&self) -> &'static str {
        self.compiled.route()
    }

    /// The support decision of the unit the program runs under, as JSON, or `None` while no
    /// unit is bound to its route.
    fn unit_support(&self) -> Option<String> {
        crate::cyclic_system_api::unit_support_json(&self.compiled.plan_view())
    }

    /// The feature its portable artifact declares.
    fn kind(&self) -> &'static str {
        self.compiled.feature()
    }

    /// The execution seed bound at compilation.
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Fit, solve every paired world and evaluate the decision under the bound seed.
    #[pyo3(signature = (*, max_bytes=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyCyclicDecisionResult> {
        let ctx = context(self.seed, max_bytes, cancel);
        let compiled = &self.compiled;
        let outcome =
            py.detach(|| compiled.execute(&ctx)).map_err(|error| classify(&*error, &ctx, false))?;
        Ok(PyCyclicDecisionResult {
            feature: self.compiled.feature(),
            outcome,
            compiled: Some((Arc::clone(&self.compiled), self.seed)),
            artifact: None,
        })
    }

    /// Decide on the logistic transform of each fitted output of a multi-output component
    /// decision (`compile_bounded_component_posterior_decision`).
    fn logistic(&self, quantities_json: &str, contract_json: &str) -> PyResult<Self> {
        crate::catch_ffi(|| {
            let Compiled::VectorComponent(inner) = self.compiled.as_ref() else {
                return Err(crate::value_err(
                    "logistic outputs apply to a multi-output component posterior decision",
                ));
            };
            let quantities = crate::transported_response_api::quantities_of(
                quantities_json,
                "logistic output quantities",
            )?;
            let contract = contract_of(contract_json)?;
            let ctx = context(self.seed, None, None);
            let plan =
                compile_bounded_component_posterior_decision(inner.clone(), quantities, contract)
                    .map_err(|error| classify(&error, &ctx, false))?;
            Ok(Self { compiled: Arc::new(Compiled::Bounded(plan)), seed: self.seed })
        })
    }
}

/// The result of one compiled or consumed cyclic decision.
#[pyclass(name = "_CyclicDecisionResult", frozen)]
pub(crate) struct PyCyclicDecisionResult {
    feature: &'static str,
    outcome: Outcome,
    compiled: Option<(Arc<Compiled>, u64)>,
    artifact: Option<Vec<u8>>,
}

#[pymethods]
impl PyCyclicDecisionResult {
    /// Kind, program identity, decision, posterior diagnostics and calibration status, as JSON.
    fn summary_json(&self) -> PyResult<String> {
        Ok(self.outcome.summary(self.feature)?.to_string())
    }

    /// The paired joint outcome columns the decision read, one column per action world.
    fn draws(&self) -> Vec<Vec<f64>> {
        self.outcome
            .decision()
            .chain
            .joint_output_law()
            .map(|(columns, _)| {
                columns
                    .iter()
                    .map(|column| column.iter().copied().map(f64::from_bits).collect())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The retained posterior parameter vectors, chain-major, or `None` for a specified model.
    fn posterior_samples(&self) -> Option<Vec<Vec<f64>>> {
        self.outcome.posterior_samples()
    }

    /// The propagation chain the decision read, as its portable artifact bytes.
    fn chain_artifact(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let bytes = self
            .outcome
            .decision()
            .chain
            .export()
            .map_err(|error| crate::serialization_err(error.to_string()))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    /// Re-execute, check this result and seal the portable artifact; a consumed result
    /// returns the bytes it was consumed from.
    #[pyo3(signature = (artifact_id, *, max_bytes=None, cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        artifact_id: &str,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        if let Some(bytes) = &self.artifact {
            return Ok(PyBytes::new(py, bytes).unbind());
        }
        let Some((compiled, seed)) = &self.compiled else {
            return Err(crate::value_err("this result retains neither a program nor an artifact"));
        };
        let ctx = context(*seed, max_bytes, cancel);
        let outcome = &self.outcome;
        let bytes = py
            .detach(|| compiled.export(outcome, artifact_id, &ctx))
            .map_err(|error| classify(&*error, &ctx, false))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

/// Compile paired hard-clamp action worlds of a specified cyclic system and a decision contract
/// (`compile_cyclic_action_world_decision`).
#[pyfunction]
#[pyo3(signature = (system, outputs_json, contract_json, *, seed, max_bytes=None, cancel=None))]
fn cyclic_action_world_decision_compile(
    py: Python<'_>,
    system: PyRef<'_, PyPreparedCyclicSystem>,
    outputs_json: &str,
    contract_json: &str,
    seed: u64,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyCompiledCyclicDecision> {
    crate::catch_ffi(|| {
        let outputs: Vec<CyclicDecisionOutput> = parse(outputs_json, "decision outputs")?;
        let contract = contract_of(contract_json)?;
        let request = system.inner.request().clone();
        let ctx = context(seed, max_bytes, cancel);
        let plan = py
            .detach(|| compile_cyclic_action_world_decision(request, outputs, contract, &ctx))
            .map_err(|error| classify(&error, &ctx, false))?;
        Ok(PyCompiledCyclicDecision { compiled: Arc::new(Compiled::ActionWorld(plan)), seed })
    })
}

/// Compile a jointly fitted cyclic posterior and its paired action worlds, standardized over a
/// target frame when one is given (`compile_cyclic_posterior_action_decision`,
/// `compile_cyclic_target_feedback_decision`). The system supplies the model, the fitted
/// coordinates, the training rows, the worlds, the initial state and the solve policy.
#[pyfunction]
#[pyo3(signature = (system, outputs_json, contract_json, prior_sd, chains, warmup, draws, proposal_scale, target_json=None, *, seed, max_bytes=None, cancel=None))]
fn cyclic_posterior_decision_compile(
    py: Python<'_>,
    system: PyRef<'_, PyPreparedCyclicSystem>,
    outputs_json: &str,
    contract_json: &str,
    prior_sd: f64,
    chains: usize,
    warmup: usize,
    draws: usize,
    proposal_scale: f64,
    target_json: Option<&str>,
    seed: u64,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyCompiledCyclicDecision> {
    crate::catch_ffi(|| {
        let outputs: Vec<CyclicDecisionOutput> = parse(outputs_json, "decision outputs")?;
        let contract = contract_of(contract_json)?;
        let target: Option<TargetFeedbackFrame> =
            target_json.map(|text| parse(text, "target frame")).transpose()?;
        let request = system.inner.request();
        let fit = request.fit.as_ref().ok_or_else(|| {
            crate::value_err(
                "a posterior decision needs a system prepared with fitted parameters and rows",
            )
        })?;
        let source = CyclicPosteriorDecisionRequest {
            model: request.model.clone(),
            parameters: fit.parameters.clone(),
            training: fit.rows.clone(),
            prior_sd,
            chains,
            warmup,
            draws_per_chain: draws,
            proposal_scale,
            worlds: request.worlds.clone(),
            initial: request.initial.clone(),
            solve_policy: request.policy,
        };
        let ctx = context(seed, max_bytes, cancel);
        let compiled = match target {
            Some(target) => Compiled::TargetFeedback(
                py.detach(|| {
                    compile_cyclic_target_feedback_decision(
                        CyclicTargetFeedbackRequest { source, target, outputs },
                        contract,
                        &ctx,
                    )
                })
                .map_err(|error| classify(&error, &ctx, false))?,
            ),
            None => Compiled::Posterior(
                py.detach(|| {
                    compile_cyclic_posterior_action_decision(source, outputs, contract, &ctx)
                })
                .map_err(|error| classify(&error, &ctx, false))?,
            ),
        };
        Ok(PyCompiledCyclicDecision { compiled: Arc::new(compiled), seed })
    })
}

/// Compile the checked connected components, one shared posterior schedule and the paired
/// action worlds (`compile_joint_component_posterior_decision`,
/// `compile_joint_vector_component_posterior_decision`).
#[pyfunction]
#[pyo3(signature = (feedback, schedule_json, outputs_json, contract_json, *, seed, max_bytes=None, cancel=None))]
fn cyclic_component_decision_compile(
    py: Python<'_>,
    feedback: PyRef<'_, PyPreparedComponentFeedback>,
    schedule_json: &str,
    outputs_json: &str,
    contract_json: &str,
    seed: u64,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyCompiledCyclicDecision> {
    crate::catch_ffi(|| {
        let schedule: ComponentJointPosteriorSchedule = parse(schedule_json, "posterior schedule")?;
        let outputs: Vec<CyclicDecisionOutput> = parse(outputs_json, "decision outputs")?;
        let contract = contract_of(contract_json)?;
        let ctx = context(seed, max_bytes, cancel);
        let compiled = match &feedback.request {
            FeedbackRequest::Scalar(request) => {
                let request = request.clone();
                Compiled::Component(
                    py.detach(|| {
                        compile_joint_component_posterior_decision(
                            request, schedule, outputs, contract, &ctx,
                        )
                    })
                    .map_err(|error| classify(&error, &ctx, false))?,
                )
            }
            FeedbackRequest::Vector(request) => {
                let request = request.clone();
                Compiled::VectorComponent(
                    py.detach(|| {
                        compile_joint_vector_component_posterior_decision(
                            request, schedule, outputs, contract, &ctx,
                        )
                    })
                    .map_err(|error| classify(&error, &ctx, false))?,
                )
            }
        };
        Ok(PyCompiledCyclicDecision { compiled: Arc::new(compiled), seed })
    })
}

fn consume_outcome(
    bytes: &[u8],
    expected: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<(&'static str, Outcome), Failure> {
    let feature = declared_feature(bytes)?;
    Ok(match feature.as_str() {
        ACTION_WORLD => (
            ACTION_WORLD,
            Outcome::ActionWorld(
                consume_cyclic_action_world_decision(bytes, expected, ctx).map_err(fail)?,
            ),
        ),
        POSTERIOR => (
            POSTERIOR,
            Outcome::Posterior(
                consume_cyclic_posterior_action_decision(bytes, expected, ctx).map_err(fail)?,
            ),
        ),
        TARGET_FEEDBACK => (
            TARGET_FEEDBACK,
            Outcome::TargetFeedback(
                consume_cyclic_target_feedback_decision(bytes, expected, ctx).map_err(fail)?,
            ),
        ),
        COMPONENT => (
            COMPONENT,
            Outcome::Component(
                consume_joint_component_posterior_decision(bytes, expected, ctx).map_err(fail)?,
            ),
        ),
        VECTOR_COMPONENT => (
            VECTOR_COMPONENT,
            Outcome::Component(
                consume_joint_vector_component_posterior_decision(bytes, expected, ctx)
                    .map_err(fail)?,
            ),
        ),
        BOUNDED => (
            BOUNDED,
            Outcome::Bounded(
                consume_bounded_component_posterior_decision(bytes, expected, ctx).map_err(fail)?,
            ),
        ),
        _ => return Err(other("the artifact declares no cyclic decision feature")),
    })
}

/// Independently recompile, refit, re-solve and re-evaluate a portable cyclic decision under a
/// program identity retained outside the artifact, with the producer's seed.
#[pyfunction]
#[pyo3(signature = (data, expected_identity, *, seed, max_bytes=None, cancel=None))]
fn cyclic_decision_consume(
    py: Python<'_>,
    data: &[u8],
    expected_identity: &str,
    seed: u64,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyCyclicDecisionResult> {
    crate::catch_ffi(|| {
        let expected = crate::digest::digest(expected_identity, "expected_identity")?;
        let ctx = context(seed, max_bytes, cancel);
        let (feature, outcome) = py
            .detach(|| consume_outcome(data, expected, &ctx))
            .map_err(|error| classify(&*error, &ctx, true))?;
        Ok(PyCyclicDecisionResult {
            feature,
            outcome,
            compiled: None,
            artifact: Some(data.to_vec()),
        })
    })
}

// ---------------------------------------------------------------------------
// Bundles whose terminal chain is regenerated by a refitted cyclic posterior.
// ---------------------------------------------------------------------------

#[allow(
    clippy::large_enum_variant,
    reason = "one owner per Python handle; boxing a variant would only add an indirection"
)]
enum CompiledBundle {
    Producer(CompiledCyclicPosteriorBundleRecomputation),
    Node(CompiledCyclicPosteriorNodeBundle),
}

impl CompiledBundle {
    fn identity(&self) -> SemanticDigest {
        match self {
            Self::Producer(plan) => plan.lowered().identity(),
            Self::Node(plan) => plan.lowered().identity(),
        }
    }

    fn unit_support(&self) -> Option<String> {
        match self {
            Self::Producer(plan) => crate::errors::unit_support_json(&plan.plan_view()),
            Self::Node(plan) => crate::errors::unit_support_json(&plan.plan_view()),
        }
    }

    const fn feature(&self) -> &'static str {
        match self {
            Self::Producer(_) => BUNDLE,
            Self::Node(_) => NODE_BUNDLE,
        }
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicPosteriorBundleReceipt, CyclicPosteriorBundleError> {
        match self {
            Self::Producer(plan) => plan.execute(ctx),
            Self::Node(plan) => plan.execute(ctx),
        }
    }

    fn export(
        &self,
        receipt: &CyclicPosteriorBundleReceipt,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CyclicPosteriorBundleError> {
        match self {
            Self::Producer(plan) => plan.export_portable(receipt, ctx),
            Self::Node(plan) => plan.export_portable(receipt, ctx),
        }
    }
}

fn receipt_value(receipt: &CyclicPosteriorBundleReceipt, feature: &str) -> Json {
    let bundle = &receipt.bundle;
    json!({
        "kind": feature,
        "identity": SemanticDigest::from_bytes(receipt.lowered_program_identity).to_hex(),
        "producer_program_identity":
            SemanticDigest::from_bytes(receipt.producer_program_identity).to_hex(),
        "bundle_identity": bundle.bundle_identity,
        "terminal_node": bundle.terminal_node,
        "terminal_identity": bundle.terminal_identity,
        "chain_program_identity": SemanticDigest::from_bytes(bundle.lowered_program_identity).to_hex(),
        "dependencies": bundle
            .dependencies
            .iter()
            .map(|dependency| json!({
                "node_id": dependency.node_id,
                "artifact_identity": dependency.artifact_identity,
                "chain_digest": dependency.chain_digest,
            }))
            .collect::<Vec<_>>(),
    })
}

/// A compiled bundle recomputation: its six-layer program and the execution seed it binds.
#[pyclass(name = "_CompiledCyclicBundle", frozen)]
pub(crate) struct PyCompiledCyclicBundle {
    compiled: Arc<CompiledBundle>,
    seed: u64,
}

#[pymethods]
impl PyCompiledCyclicBundle {
    /// Identity of the complete bundle and producer program, as lowercase hex.
    fn identity(&self) -> String {
        self.compiled.identity().to_hex()
    }

    /// The feature its portable artifact declares.
    fn kind(&self) -> &'static str {
        self.compiled.feature()
    }

    /// The support decision of the unit the compiled program runs under, as JSON; `None` while
    /// no unit is bound to its route.
    fn unit_support(&self) -> Option<String> {
        self.compiled.unit_support()
    }

    /// Refit the cyclic producer, execute the terminal chain and require both to agree.
    #[pyo3(signature = (*, max_bytes=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyCyclicBundleReceipt> {
        let ctx = context(self.seed, max_bytes, cancel);
        let compiled = &self.compiled;
        let receipt =
            py.detach(|| compiled.execute(&ctx)).map_err(|error| classify(&error, &ctx, false))?;
        Ok(PyCyclicBundleReceipt {
            feature: self.compiled.feature(),
            receipt,
            compiled: Some((Arc::clone(&self.compiled), self.seed)),
            artifact: None,
        })
    }
}

/// The receipt of one executed or consumed bundle recomputation.
#[pyclass(name = "_CyclicBundleReceipt", frozen)]
pub(crate) struct PyCyclicBundleReceipt {
    feature: &'static str,
    receipt: CyclicPosteriorBundleReceipt,
    compiled: Option<(Arc<CompiledBundle>, u64)>,
    artifact: Option<Vec<u8>>,
}

#[pymethods]
impl PyCyclicBundleReceipt {
    /// Program, producer, bundle, terminal and dependency identities, as JSON.
    fn receipt_json(&self) -> String {
        receipt_value(&self.receipt, self.feature).to_string()
    }

    /// Re-execute, check this receipt and seal the portable bundle recomputation; a consumed
    /// receipt returns the bytes it was consumed from.
    #[pyo3(signature = (*, max_bytes=None, cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        if let Some(bytes) = &self.artifact {
            return Ok(PyBytes::new(py, bytes).unbind());
        }
        let Some((compiled, seed)) = &self.compiled else {
            return Err(crate::value_err("this receipt retains neither a program nor an artifact"));
        };
        let ctx = context(*seed, max_bytes, cancel);
        let receipt = &self.receipt;
        let bytes = py
            .detach(|| compiled.export(receipt, &ctx))
            .map_err(|error| classify(&error, &ctx, false))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

/// Add a refitted cyclic posterior producer node to a bundle under construction
/// (`BundleBuilder::add_cyclic_posterior_fit`); the producer is replayed before it is accepted.
#[pyfunction]
#[pyo3(signature = (builder, node_id, artifact, producer_identity, *, seed, max_bytes=None, cancel=None))]
fn cyclic_bundle_add_posterior_fit(
    py: Python<'_>,
    mut builder: PyRefMut<'_, PyBundleBuilder>,
    node_id: &str,
    artifact: &[u8],
    producer_identity: &str,
    seed: u64,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<()> {
    crate::catch_ffi(|| {
        let program = crate::digest::digest(producer_identity, "producer_identity")?;
        let ctx = context(seed, max_bytes, cancel);
        let inner = &mut builder.inner;
        py.detach(|| inner.add_cyclic_posterior_fit(node_id, artifact, program, &ctx))
            .map_err(|error| classify(&error, &ctx, false))
    })
}

/// Compile a bundle whose propagation-chain terminal is regenerated by a cyclic posterior
/// producer: either the producer artifact with its program identity
/// (`compile_cyclic_posterior_bundle_recomputation`) or an embedded producer node with a direct
/// edge to the terminal (`compile_cyclic_posterior_node_bundle_recomputation`).
#[pyfunction]
#[pyo3(signature = (bundle, terminal_node, producer=None, producer_identity=None, producer_node=None, *, seed, max_bytes=None, cancel=None))]
fn cyclic_bundle_compile(
    py: Python<'_>,
    bundle: PyRef<'_, PyCompositionBundle>,
    terminal_node: &str,
    producer: Option<&[u8]>,
    producer_identity: Option<&str>,
    producer_node: Option<&str>,
    seed: u64,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyCompiledCyclicBundle> {
    crate::catch_ffi(|| {
        let graph = bundle.bundle.clone();
        let ctx = context(seed, max_bytes, cancel);
        let compiled = match (producer_node, producer, producer_identity) {
            (Some(node), None, None) => CompiledBundle::Node(
                py.detach(|| {
                    compile_cyclic_posterior_node_bundle_recomputation(
                        graph,
                        node,
                        terminal_node,
                        &ctx,
                    )
                })
                .map_err(|error| classify(&error, &ctx, false))?,
            ),
            (None, Some(artifact), Some(identity)) => {
                let program = crate::digest::digest(identity, "producer_identity")?;
                let artifact = artifact.to_vec();
                CompiledBundle::Producer(
                    py.detach(|| {
                        compile_cyclic_posterior_bundle_recomputation(
                            graph,
                            terminal_node,
                            artifact,
                            program,
                            &ctx,
                        )
                    })
                    .map_err(|error| classify(&error, &ctx, false))?,
                )
            }
            _ => {
                return Err(crate::value_err(
                    "name either the producer node or the producer artifact with its identity",
                ));
            }
        };
        Ok(PyCompiledCyclicBundle { compiled: Arc::new(compiled), seed })
    })
}

/// Independently reconstruct, recompile and refit a portable bundle recomputation under the
/// bundle and program identities retained outside the artifact
/// (`consume_portable_cyclic_posterior_bundle_recomputation`,
/// `consume_portable_cyclic_posterior_node_bundle_recomputation`).
#[pyfunction]
#[pyo3(signature = (data, expected_bundle, expected_identity, *, seed, max_bytes=None, cancel=None))]
fn cyclic_bundle_consume(
    py: Python<'_>,
    data: &[u8],
    expected_bundle: &str,
    expected_identity: &str,
    seed: u64,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyCyclicBundleReceipt> {
    crate::catch_ffi(|| {
        let program = crate::digest::digest(expected_identity, "expected_identity")?;
        let ctx = context(seed, max_bytes, cancel);
        let (feature, receipt) = py
            .detach(|| -> Result<(&'static str, CyclicPosteriorBundleReceipt), Failure> {
                match declared_feature(data)?.as_str() {
                    BUNDLE => consume_portable_cyclic_posterior_bundle_recomputation(
                        data,
                        expected_bundle,
                        program,
                        &ctx,
                    )
                    .map(|receipt| (BUNDLE, receipt))
                    .map_err(fail),
                    NODE_BUNDLE => consume_portable_cyclic_posterior_node_bundle_recomputation(
                        data,
                        expected_bundle,
                        program,
                        &ctx,
                    )
                    .map(|receipt| (NODE_BUNDLE, receipt))
                    .map_err(fail),
                    _ => Err(other("the artifact declares no cyclic bundle recomputation feature")),
                }
            })
            .map_err(|error| classify(&*error, &ctx, true))?;
        Ok(PyCyclicBundleReceipt {
            feature,
            receipt,
            compiled: None,
            artifact: Some(data.to_vec()),
        })
    })
}

/// Consume a bundle with the native fit verifiers, each of which recompiles and refits its
/// embedded producer under this context (`native_fit_bundle_consumer`, or with `cyclic` the
/// cyclic posterior route's `cyclic_posterior_bundle_consumer`). Returns
/// `(consumed_json, refusal)`.
#[pyfunction]
#[pyo3(signature = (data, expected_identity, supplied_json=None, cyclic=false, *, seed, max_bytes=None, cancel=None))]
fn cyclic_bundle_consume_refitting(
    py: Python<'_>,
    data: &[u8],
    expected_identity: &str,
    supplied_json: Option<&str>,
    cyclic: bool,
    seed: u64,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<String>, Option<String>)> {
    crate::catch_ffi(|| {
        let supplied = crate::composition_bundle_api::supplied_from(supplied_json)?;
        let ctx = context(seed, max_bytes, cancel);
        let consumed = py.detach(|| {
            let consumer = if cyclic {
                cyclic_posterior_bundle_consumer(&ctx)
            } else {
                native_fit_bundle_consumer(&ctx)
            };
            consumer.consume(data, &BundleLimits::default(), expected_identity, &supplied)
        });
        Ok(match consumed {
            Ok(consumed) => (Some(crate::composition_bundle_api::consumed_json(&consumed)), None),
            Err(error) => (None, Some(crate::composition_bundle_api::error_refusal(&error))),
        })
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPreparedComponentFeedback>()?;
    m.add_class::<PyCompiledCyclicDecision>()?;
    m.add_class::<PyCyclicDecisionResult>()?;
    m.add_class::<PyCompiledCyclicBundle>()?;
    m.add_class::<PyCyclicBundleReceipt>()?;
    m.add_function(wrap_pyfunction!(cyclic_component_feedback_prepare, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_action_world_decision_compile, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_posterior_decision_compile, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_component_decision_compile, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_decision_consume, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_bundle_add_posterior_fit, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_bundle_compile, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_bundle_consume, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_bundle_consume_refitting, m)?)?;
    Ok(())
}
