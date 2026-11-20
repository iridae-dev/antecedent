//! Python bridge for live recomputation of a composition bundle's terminal.
//!
//! Python hands over a sealed bundle and names its terminal (a propagation chain, a decision
//! result, or a producer -> chain -> decision path), with the contract and source law a
//! decision terminal is scored on. The six-layer compiler, the dependency replay, the live
//! terminal, the portable artifact and the independent consumer are Rust's. The builder methods
//! embed the native predictive producers and chain-decision terminals such a path reads. A
//! refusal comes back as structured JSON for the Python layer to raise as a typed exception; a
//! cancelled run raises `CausalCancelledError`.

use antecedent::analysis::composition::{
    BundleBuilder, BundleError, ChainBundleRecomputationReceipt, ChainBundleRecomputeError,
    CompiledChainBundleRecomputation, CompiledPredictiveFitDecisionBundle, CompositionBundle,
    NodeKind, NodeSource, PredictiveFitDecisionBundleError, PredictiveFitDecisionBundleReceipt,
    ReplayedBundleDependency, compile_assessed_bridge_posterior_node_decision_bundle,
    compile_chain_bundle_recomputation, compile_cyclic_target_feedback_node_decision_bundle,
    compile_graph_imputation_predictive_node_decision_bundle,
    compile_joint_transport_predictive_node_decision_bundle,
    compile_trial_ipw_predictive_node_decision_bundle, consume_portable_chain_bundle_recomputation,
    consume_portable_predictive_fit_decision_bundle,
};
use antecedent::analysis::composition_decision_terminal::{
    CompiledDecisionBundleRecomputation, DecisionBundleRecomputationReceipt,
    DecisionBundleRecomputeError, compile_decision_bundle_recomputation,
    compile_posterior_decision_bundle_recomputation,
    consume_portable_decision_bundle_recomputation,
    consume_portable_posterior_decision_bundle_recomputation,
};
use antecedent::analysis::propagation_chain::{
    PropagationChainRecordWire, compile_propagation_chain,
};
use antecedent::design::decision_artifact::contract_from_json_refusal;
use antecedent::prelude::ExecutionContext;
use antecedent::reason::reason_code;
use antecedent::scientific::SemanticDigest;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Value as Json, json};

use crate::PyCancellationToken;
use crate::bridge::{MAX_DATA_DECLARATION_BYTES, RefusalJson, external_refusal_json};
use crate::chain_decision_api::{chain_refusal_json, context, io_refusal_json};
use crate::composition_bundle_api::{PyBundleBuilder, PyCompositionBundle, error_refusal};
use crate::distribution_api::PyJointDistributionArtifact;

type Refused<T> = (Option<T>, Option<String>);
/// The shared signature of the five predictive-fit decision bundle compilers.
type CompilePredictive =
    fn(
        CompositionBundle,
        &str,
        &str,
        &str,
        &ExecutionContext,
    ) -> Result<CompiledPredictiveFitDecisionBundle, PredictiveFitDecisionBundleError>;
/// The shared signature of the five builder methods that embed a predictive producer.
type AddPredictive = fn(
    &mut BundleBuilder,
    &str,
    &[u8],
    SemanticDigest,
    &ExecutionContext,
) -> Result<(), PredictiveFitDecisionBundleError>;

const STAGE: &str = "bundle_recomputation";
// Family-local details of this bridge; a bundle, chain or contract refusal keeps its own.
const INVALID_REQUEST: &str = "bundle_recomputation.invalid_request";
const COMPILE_REFUSED: &str = "bundle_recomputation.refused";
const REPLAY_REFUSED: &str = "bundle_recomputation.replay_refused";
const EVALUATION_REFUSED: &str = "bundle_recomputation.evaluation_refused";
const ARTIFACT_INVALID: &str = "bundle_recomputation.artifact_invalid";

const CHAIN_ROUTE: &str = "chain";
const PREDICTIVE_DECISION_ROUTE: &str = "predictive_decision";
const POSTERIOR_DECISION_ROUTE: &str = "posterior_decision";
const PREDICTIVE_FIT_ROUTE: &str = "predictive_fit_decision";

fn cancelled() -> PyErr {
    crate::CausalCancelledError::new_err("bundle recomputation cancelled")
}

/// Where a binding failed: while compiling (under the route's own code) or while executing or
/// consuming, where the embedded graph does not bind the live result.
#[derive(Clone, Copy)]
enum Phase {
    Compile(&'static str),
    Replay,
}

fn invalid_json(why: &str, phase: Phase) -> String {
    match phase {
        Phase::Compile(code) => RefusalJson::new(code, STAGE, COMPILE_REFUSED, why).json(),
        Phase::Replay => {
            RefusalJson::new(reason_code!("external_binding_mismatch"), STAGE, REPLAY_REFUSED, why)
                .json()
        }
    }
}

fn request_json(message: &str) -> String {
    RefusalJson::new(reason_code!("invalid_argument"), STAGE, INVALID_REQUEST, message).json()
}

fn bundle_json(error: &BundleError) -> String {
    error_refusal(error)
}

fn chain_bundle_json(error: &ChainBundleRecomputeError, phase: Phase) -> String {
    match error {
        ChainBundleRecomputeError::Bundle(bundle) => bundle_json(bundle),
        ChainBundleRecomputeError::Chain(refusal) => chain_refusal_json(refusal, STAGE),
        ChainBundleRecomputeError::Io(io) => io_refusal_json(io, STAGE, ARTIFACT_INVALID),
        ChainBundleRecomputeError::Invalid(why) => invalid_json(why, phase),
    }
}

fn decision_bundle_json(error: &DecisionBundleRecomputeError, phase: Phase) -> String {
    match error {
        DecisionBundleRecomputeError::Bundle(bundle) => bundle_json(bundle),
        DecisionBundleRecomputeError::Contract(contract) => {
            external_refusal_json(&contract.to_refusal())
        }
        DecisionBundleRecomputeError::Evaluation(message) => RefusalJson::new(
            reason_code!("decision_contract_unsatisfied"),
            STAGE,
            EVALUATION_REFUSED,
            message,
        )
        .json(),
        DecisionBundleRecomputeError::Io(io) => io_refusal_json(io, STAGE, ARTIFACT_INVALID),
        DecisionBundleRecomputeError::Invalid(why) => invalid_json(why, phase),
        DecisionBundleRecomputeError::NotSupported(refused) => {
            RefusalJson::new(refused.code(), STAGE, refused.detail, &refused.to_string()).json()
        }
    }
}

fn predictive_bundle_json(error: &PredictiveFitDecisionBundleError, phase: Phase) -> String {
    match error {
        PredictiveFitDecisionBundleError::Bundle(inner) => chain_bundle_json(inner, phase),
        PredictiveFitDecisionBundleError::Invalid(why) => invalid_json(why, phase),
        PredictiveFitDecisionBundleError::NotSupported(refused) => {
            RefusalJson::new(refused.code(), STAGE, refused.detail, &refused.to_string()).json()
        }
    }
}

fn hex(bytes: [u8; 32]) -> String {
    SemanticDigest::from_bytes(bytes).to_hex()
}

fn dependencies_json(dependencies: &[ReplayedBundleDependency]) -> Json {
    Json::Array(
        dependencies
            .iter()
            .map(|dependency| {
                json!({
                    "node_id": dependency.node_id,
                    "artifact_identity": dependency.artifact_identity,
                    "chain_digest": dependency.chain_digest,
                })
            })
            .collect(),
    )
}

enum Plan {
    Chain(CompiledChainBundleRecomputation),
    Decision { plan: CompiledDecisionBundleRecomputation, posterior: bool },
    Predictive(CompiledPredictiveFitDecisionBundle),
}

#[derive(Clone)]
enum Receipt {
    Chain(ChainBundleRecomputationReceipt),
    Decision(DecisionBundleRecomputationReceipt),
    Predictive(PredictiveFitDecisionBundleReceipt),
}

impl Plan {
    const fn route(&self) -> &'static str {
        match self {
            Self::Chain(_) => CHAIN_ROUTE,
            Self::Decision { posterior: false, .. } => PREDICTIVE_DECISION_ROUTE,
            Self::Decision { posterior: true, .. } => POSTERIOR_DECISION_ROUTE,
            Self::Predictive(_) => PREDICTIVE_FIT_ROUTE,
        }
    }

    fn identity(&self) -> SemanticDigest {
        match self {
            Self::Chain(plan) => plan.lowered().identity(),
            Self::Decision { plan, .. } => plan.lowered().identity(),
            Self::Predictive(plan) => plan.lowered().identity(),
        }
    }

    fn plan_view_json(&self) -> String {
        match self {
            Self::Chain(plan) => plan.plan_view().to_json().to_string(),
            Self::Decision { plan, .. } => plan.plan_view().to_json().to_string(),
            Self::Predictive(plan) => plan.plan_view().to_json().to_string(),
        }
    }

    fn execute(&self, ctx: &ExecutionContext) -> Result<Receipt, String> {
        match self {
            Self::Chain(plan) => plan
                .execute(ctx)
                .map(Receipt::Chain)
                .map_err(|error| chain_bundle_json(&error, Phase::Replay)),
            Self::Decision { plan, .. } => plan
                .execute(ctx)
                .map(Receipt::Decision)
                .map_err(|error| decision_bundle_json(&error, Phase::Replay)),
            Self::Predictive(plan) => plan
                .execute(ctx)
                .map(Receipt::Predictive)
                .map_err(|error| predictive_bundle_json(&error, Phase::Replay)),
        }
    }

    fn export_portable(
        &self,
        receipt: &Receipt,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, String> {
        match (self, receipt) {
            (Self::Chain(plan), Receipt::Chain(receipt)) => plan
                .export_portable(receipt, ctx)
                .map_err(|error| chain_bundle_json(&error, Phase::Replay)),
            (Self::Decision { plan, .. }, Receipt::Decision(receipt)) => plan
                .export_portable(receipt, ctx)
                .map_err(|error| decision_bundle_json(&error, Phase::Replay)),
            (Self::Predictive(plan), Receipt::Predictive(receipt)) => plan
                .export_portable(receipt, ctx)
                .map_err(|error| predictive_bundle_json(&error, Phase::Replay)),
            _ => Err(invalid_json(
                "the receipt was produced by another recomputation route",
                Phase::Compile(reason_code!("invalid_argument")),
            )),
        }
    }
}

impl Receipt {
    fn json(&self) -> Json {
        match self {
            Self::Chain(receipt) => json!({
                "route": CHAIN_ROUTE,
                "bundle_identity": receipt.bundle_identity,
                "program_identity": hex(receipt.lowered_program_identity),
                "terminal_node": receipt.terminal_node,
                "terminal_identity": receipt.terminal_identity,
                "chain_identity": Json::Null,
                "decision_fingerprint": Json::Null,
                "producer_program_identity": Json::Null,
                "dependencies": dependencies_json(&receipt.dependencies),
            }),
            Self::Decision(receipt) => json!({
                "route": Json::Null,
                "bundle_identity": receipt.bundle_identity,
                "program_identity": hex(receipt.lowered_program_identity),
                "terminal_node": receipt.terminal_node,
                "terminal_identity": receipt.terminal_identity,
                "chain_identity": Json::Null,
                "decision_fingerprint": receipt.decision_fingerprint,
                "producer_program_identity": Json::Null,
                "dependencies": dependencies_json(&receipt.dependencies),
            }),
            Self::Predictive(receipt) => json!({
                "route": PREDICTIVE_FIT_ROUTE,
                "bundle_identity": receipt.bundle_identity,
                "program_identity": hex(receipt.lowered_program_identity),
                "terminal_node": Json::Null,
                "terminal_identity": Json::Null,
                "chain_identity": receipt.chain_identity,
                "decision_fingerprint": receipt.decision_fingerprint,
                "producer_program_identity": hex(receipt.producer_program_identity),
                "dependencies": dependencies_json(&receipt.dependencies),
            }),
        }
    }
}

/// A bundle and its live terminal compiled into one six-layer program.
#[pyclass(name = "BundleRecomputationPlanHandle", skip_from_py_object)]
pub(crate) struct PyBundleRecomputationPlan {
    plan: Plan,
}

#[pymethods]
impl PyBundleRecomputationPlan {
    /// `chain`, `predictive_decision`, `posterior_decision` or `predictive_fit_decision`.
    #[getter]
    fn route(&self) -> &'static str {
        self.plan.route()
    }

    /// Identity of the compiled program (lowercase hex).
    #[getter]
    fn program_identity(&self) -> String {
        self.plan.identity().to_hex()
    }

    /// Every compiler layer as JSON, with nothing executed.
    fn plan_view_json(&self) -> String {
        self.plan.plan_view_json()
    }

    /// Replay every dependency through its own consumer and execute the live terminal.
    #[pyo3(signature = (*, seed=1, threads=None, cancel=None, memory_bytes=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> PyResult<Refused<PyBundleRecomputationRun>> {
        let ctx = context(seed, threads, cancel, memory_bytes);
        let plan = &self.plan;
        let (executed, stopped) = crate::detach_catch(py, move || {
            let executed = plan.execute(&ctx);
            Ok((executed, ctx.cancellation.is_cancelled()))
        })?;
        match executed {
            Ok(receipt) => {
                Ok((Some(PyBundleRecomputationRun { route: plan.route(), receipt }), None))
            }
            Err(_) if stopped => Err(cancelled()),
            Err(refusal) => Ok((None, Some(refusal))),
        }
    }

    /// Re-execute, require the same receipt, and embed the graph with it for a fresh process.
    #[pyo3(signature = (run, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
    fn export_portable(
        &self,
        py: Python<'_>,
        run: PyRef<'_, PyBundleRecomputationRun>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> PyResult<Refused<Py<PyBytes>>> {
        let ctx = context(seed, threads, cancel, memory_bytes);
        let plan = &self.plan;
        let receipt = run.receipt.clone();
        let (exported, stopped) = crate::detach_catch(py, move || {
            let exported = plan.export_portable(&receipt, &ctx);
            Ok((exported, ctx.cancellation.is_cancelled()))
        })?;
        match exported {
            Ok(bytes) => Ok((Some(PyBytes::new(py, &bytes).unbind()), None)),
            Err(_) if stopped => Err(cancelled()),
            Err(refusal) => Ok((None, Some(refusal))),
        }
    }
}

/// The receipt of one live recomputation or of an independent portable replay.
#[pyclass(name = "BundleRecomputationRunHandle", skip_from_py_object)]
pub(crate) struct PyBundleRecomputationRun {
    route: &'static str,
    receipt: Receipt,
}

#[pymethods]
impl PyBundleRecomputationRun {
    /// The receipt as JSON: route, identities, fingerprints and every replayed dependency.
    fn receipt_json(&self) -> String {
        let mut wire = self.receipt.json();
        if let Some(object) = wire.as_object_mut() {
            object.insert("route".to_owned(), Json::String(self.route.to_owned()));
        }
        wire.to_string()
    }
}

fn embedded_chain(
    bundle: &CompositionBundle,
    terminal: &str,
) -> Result<PropagationChainRecordWire, String> {
    let node = bundle.nodes().iter().find(|node| node.id == terminal).ok_or_else(|| {
        invalid_json(
            "terminal node is absent from the bundle",
            Phase::Compile(reason_code!("invalid_argument")),
        )
    })?;
    match (&node.kind, &node.source) {
        (NodeKind::PropagationChain, NodeSource::Embedded { bytes }) => {
            PropagationChainRecordWire::decode(bytes)
                .map_err(|error| chain_refusal_json(&error.to_refusal(), STAGE))
        }
        _ => Err(invalid_json(
            "the executable terminal must be an embedded propagation chain",
            Phase::Compile(reason_code!("invalid_argument")),
        )),
    }
}

/// Compile a fully embedded bundle with its propagation-chain terminal into one program. The
/// chain is the terminal's own embedded record unless `record_json` declares the live chain
/// the terminal must equal.
#[pyfunction]
#[pyo3(signature = (bundle, terminal_node, record_json=None, *, cancel=None))]
fn chain_bundle_recomputation_compile(
    bundle: PyRef<'_, PyCompositionBundle>,
    terminal_node: &str,
    record_json: Option<&str>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<PyBundleRecomputationPlan>> {
    crate::catch_ffi(|| {
        let record = match record_json {
            Some(text) if text.len() > MAX_DATA_DECLARATION_BYTES => {
                return Ok((None, Some(request_json("the chain record exceeds its byte bound"))));
            }
            Some(text) => match serde_json::from_str::<PropagationChainRecordWire>(text) {
                Ok(record) => record,
                Err(error) => return Ok((None, Some(request_json(&error.to_string())))),
            },
            None => match embedded_chain(&bundle.bundle, terminal_node) {
                Ok(record) => record,
                Err(refusal) => return Ok((None, Some(refusal))),
            },
        };
        let ctx = context(0, Some(1), cancel, None);
        let chain = match compile_propagation_chain(record, &ctx) {
            Ok(chain) => chain,
            Err(_) if ctx.cancellation.is_cancelled() => return Err(cancelled()),
            Err(refusal) => return Ok((None, Some(chain_refusal_json(&refusal, STAGE)))),
        };
        Ok(match compile_chain_bundle_recomputation(bundle.bundle.clone(), terminal_node, &chain) {
            Ok(plan) => (Some(PyBundleRecomputationPlan { plan: Plan::Chain(plan) }), None),
            Err(error) => (
                None,
                Some(chain_bundle_json(&error, Phase::Compile(reason_code!("invalid_argument")))),
            ),
        })
    })
}

/// Compile a bundle whose terminal is an embedded decision result, recomputed from the contract
/// on the aligned source law: predictive by default, a parameter posterior when `posterior`.
#[pyfunction]
#[pyo3(signature = (bundle, terminal_node, contract_json, source, *, posterior=false))]
fn decision_bundle_recomputation_compile(
    bundle: PyRef<'_, PyCompositionBundle>,
    terminal_node: &str,
    contract_json: &str,
    source: PyRef<'_, PyJointDistributionArtifact>,
    posterior: bool,
) -> PyResult<Refused<PyBundleRecomputationPlan>> {
    crate::catch_ffi(|| {
        let contract = match contract_from_json_refusal(contract_json) {
            Ok(contract) => contract,
            Err(refusal) => return Ok((None, Some(external_refusal_json(&refusal)))),
        };
        let compiled = if posterior {
            compile_posterior_decision_bundle_recomputation(
                bundle.bundle.clone(),
                terminal_node,
                contract,
                source.inner().clone(),
            )
        } else {
            compile_decision_bundle_recomputation(
                bundle.bundle.clone(),
                terminal_node,
                contract,
                source.inner().clone(),
            )
        };
        Ok(match compiled {
            Ok(plan) => {
                (Some(PyBundleRecomputationPlan { plan: Plan::Decision { plan, posterior } }), None)
            }
            Err(error) => (
                None,
                Some(decision_bundle_json(
                    &error,
                    Phase::Compile(reason_code!("decision_contract_unsatisfied")),
                )),
            ),
        })
    })
}

/// Compile a producer -> chain -> decision bundle. The producer kind is the kind of the
/// `producer` node; the chain and the predictive chain-decision terminal follow it by edges.
#[pyfunction]
#[pyo3(signature = (bundle, producer, chain, terminal, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
#[allow(clippy::too_many_arguments, reason = "the three bound nodes and the execution controls")]
fn predictive_fit_decision_bundle_compile(
    bundle: PyRef<'_, PyCompositionBundle>,
    producer: &str,
    chain: &str,
    terminal: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Refused<PyBundleRecomputationPlan>> {
    crate::catch_ffi(|| {
        let kind =
            bundle.bundle.nodes().iter().find(|node| node.id == producer).map(|node| node.kind);
        let compile: CompilePredictive = match kind {
            Some(NodeKind::JointTransportPredictiveFit) => {
                compile_joint_transport_predictive_node_decision_bundle
            }
            Some(NodeKind::TrialIpwPredictiveFit) => {
                compile_trial_ipw_predictive_node_decision_bundle
            }
            Some(NodeKind::CyclicTargetFeedbackFit) => {
                compile_cyclic_target_feedback_node_decision_bundle
            }
            Some(NodeKind::GraphImputationPredictiveFit) => {
                compile_graph_imputation_predictive_node_decision_bundle
            }
            Some(NodeKind::AssessedBridgePosteriorFit) => {
                compile_assessed_bridge_posterior_node_decision_bundle
            }
            _ => {
                return Ok((
                    None,
                    Some(invalid_json(
                        "the producer must be an embedded predictive fit node of the bundle",
                        Phase::Compile(reason_code!("invalid_argument")),
                    )),
                ));
            }
        };
        let ctx = context(seed, threads, cancel, memory_bytes);
        Ok(match compile(bundle.bundle.clone(), producer, chain, terminal, &ctx) {
            Ok(plan) => (Some(PyBundleRecomputationPlan { plan: Plan::Predictive(plan) }), None),
            Err(_) if ctx.cancellation.is_cancelled() => return Err(cancelled()),
            Err(error) => (
                None,
                Some(predictive_bundle_json(
                    &error,
                    Phase::Compile(reason_code!("invalid_argument")),
                )),
            ),
        })
    })
}

fn consume_run(
    py: Python<'_>,
    route: &'static str,
    ctx: ExecutionContext,
    consume: impl FnOnce(&ExecutionContext) -> Result<Receipt, String> + Send,
) -> PyResult<Refused<PyBundleRecomputationRun>> {
    let (consumed, stopped) = crate::detach_catch(py, move || {
        let consumed = consume(&ctx);
        Ok((consumed, ctx.cancellation.is_cancelled()))
    })?;
    match consumed {
        Ok(receipt) => Ok((Some(PyBundleRecomputationRun { route, receipt }), None)),
        Err(_) if stopped => Err(cancelled()),
        Err(refusal) => Ok((None, Some(refusal))),
    }
}

/// Rebuild a portable chain recomputation, recompile its terminal program, replay every
/// dependency and require the live receipt under the retained bundle and program identities.
#[pyfunction]
#[pyo3(signature = (data, expected_bundle, expected_program, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
#[allow(clippy::too_many_arguments, reason = "the artifact, its two identities and the controls")]
fn chain_bundle_recomputation_consume(
    py: Python<'_>,
    data: &[u8],
    expected_bundle: &str,
    expected_program: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Refused<PyBundleRecomputationRun>> {
    let program = crate::digest::digest(expected_program, "expected_program")?;
    let ctx = context(seed, threads, cancel, memory_bytes);
    consume_run(py, CHAIN_ROUTE, ctx, move |ctx| {
        consume_portable_chain_bundle_recomputation(data, expected_bundle, program, ctx)
            .map(Receipt::Chain)
            .map_err(|error| chain_bundle_json(&error, Phase::Replay))
    })
}

/// Rebuild a portable decision recomputation from its embedded contract and source law,
/// recompile it and require the live receipt; `posterior` names the parameter-posterior route.
#[pyfunction]
#[pyo3(signature = (data, expected_bundle, expected_program, *, posterior=false, seed=1, threads=None, cancel=None, memory_bytes=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "the artifact, its identities, its route and the controls"
)]
fn decision_bundle_recomputation_consume(
    py: Python<'_>,
    data: &[u8],
    expected_bundle: &str,
    expected_program: &str,
    posterior: bool,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Refused<PyBundleRecomputationRun>> {
    let program = crate::digest::digest(expected_program, "expected_program")?;
    let ctx = context(seed, threads, cancel, memory_bytes);
    let route = if posterior { POSTERIOR_DECISION_ROUTE } else { PREDICTIVE_DECISION_ROUTE };
    consume_run(py, route, ctx, move |ctx| {
        let consumed = if posterior {
            consume_portable_posterior_decision_bundle_recomputation(
                data,
                expected_bundle,
                program,
                ctx,
            )
        } else {
            consume_portable_decision_bundle_recomputation(data, expected_bundle, program, ctx)
        };
        consumed.map(Receipt::Decision).map_err(|error| decision_bundle_json(&error, Phase::Replay))
    })
}

/// Rebuild a portable producer -> chain -> decision bundle, refit the producer from its raw
/// evidence, rescore the decision and require the live receipt.
#[pyfunction]
#[pyo3(signature = (data, expected_bundle, expected_program, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
#[allow(clippy::too_many_arguments, reason = "the artifact, its two identities and the controls")]
fn predictive_fit_decision_bundle_consume(
    py: Python<'_>,
    data: &[u8],
    expected_bundle: &str,
    expected_program: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Refused<PyBundleRecomputationRun>> {
    let program = crate::digest::digest(expected_program, "expected_program")?;
    let ctx = context(seed, threads, cancel, memory_bytes);
    consume_run(py, PREDICTIVE_FIT_ROUTE, ctx, move |ctx| {
        consume_portable_predictive_fit_decision_bundle(data, expected_bundle, program, ctx)
            .map(Receipt::Predictive)
            .map_err(|error| predictive_bundle_json(&error, Phase::Replay))
    })
}

#[pymethods]
impl PyBundleBuilder {
    /// Embed a native predictive producer artifact of `kind` under its producer program
    /// identity. The producer is reidentified and refitted from its raw evidence before the
    /// node is accepted; `seed` is recorded with the node and every replay refits under it.
    /// Returns the refusal, or `None`.
    #[pyo3(signature = (kind, node_id, artifact, program, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
    #[allow(clippy::too_many_arguments, reason = "the node declaration and the execution controls")]
    fn add_predictive_fit(
        &mut self,
        py: Python<'_>,
        kind: &str,
        node_id: &str,
        artifact: &[u8],
        program: &str,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> PyResult<Option<String>> {
        let program = crate::digest::digest(program, "program")?;
        let add: AddPredictive = match NodeKind::from_name(kind) {
            Some(NodeKind::JointTransportPredictiveFit) => {
                BundleBuilder::add_joint_transport_predictive_fit
            }
            Some(NodeKind::TrialIpwPredictiveFit) => BundleBuilder::add_trial_ipw_predictive_fit,
            Some(NodeKind::CyclicTargetFeedbackFit) => {
                BundleBuilder::add_cyclic_target_feedback_fit
            }
            Some(NodeKind::GraphImputationPredictiveFit) => {
                BundleBuilder::add_graph_imputation_predictive_fit
            }
            Some(NodeKind::AssessedBridgePosteriorFit) => {
                BundleBuilder::add_assessed_bridge_posterior_fit
            }
            _ => {
                return Ok(Some(request_json(&format!(
                    "`{kind}` is not a predictive fit node kind"
                ))));
            }
        };
        let ctx = context(seed, threads, cancel, memory_bytes);
        let builder = &mut self.inner;
        let (added, stopped) = crate::detach_catch(py, move || {
            let added = add(builder, node_id, artifact, program, &ctx);
            Ok((added, ctx.cancellation.is_cancelled()))
        })?;
        match added {
            Ok(()) => Ok(None),
            Err(_) if stopped => Err(cancelled()),
            Err(error) => Ok(Some(predictive_bundle_json(&error, Phase::Replay))),
        }
    }

    /// Embed an exported chain-decision artifact as a terminal that is recompiled and scored
    /// live; `seed` is recorded with the node. Returns the refusal, or `None`.
    #[pyo3(signature = (node_id, artifact, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
    #[allow(clippy::too_many_arguments, reason = "the node declaration and the execution controls")]
    fn add_predictive_chain_decision(
        &mut self,
        py: Python<'_>,
        node_id: &str,
        artifact: &[u8],
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> PyResult<Option<String>> {
        let ctx = context(seed, threads, cancel, memory_bytes);
        let builder = &mut self.inner;
        let (added, stopped) = crate::detach_catch(py, move || {
            let added = builder.add_predictive_chain_decision(node_id, artifact, &ctx);
            Ok((added, ctx.cancellation.is_cancelled()))
        })?;
        match added {
            Ok(()) => Ok(None),
            Err(_) if stopped => Err(cancelled()),
            Err(error) => Ok(Some(predictive_bundle_json(&error, Phase::Replay))),
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBundleRecomputationPlan>()?;
    module.add_class::<PyBundleRecomputationRun>()?;
    module.add_function(wrap_pyfunction!(chain_bundle_recomputation_compile, module)?)?;
    module.add_function(wrap_pyfunction!(chain_bundle_recomputation_consume, module)?)?;
    module.add_function(wrap_pyfunction!(decision_bundle_recomputation_compile, module)?)?;
    module.add_function(wrap_pyfunction!(decision_bundle_recomputation_consume, module)?)?;
    module.add_function(wrap_pyfunction!(predictive_fit_decision_bundle_compile, module)?)?;
    module.add_function(wrap_pyfunction!(predictive_fit_decision_bundle_consume, module)?)?;
    Ok(())
}
