//! Python bridge for compiled policy programs and their portable replay.
//!
//! Four capabilities share the shape compile, execute, seal, consume: an observational policy
//! value sealed with its complete request, the full-refit subsampling candidate of a
//! graph-checked cross-fitted policy learner, the reliability diagnostic of fixed sequential
//! policies, and the bundle that carries that diagnostic's producer, draw carrier and
//! diagnostic for independent recomputation. Python supplies the columns and declarations;
//! the compiler programs, every refit, the diagnostics, the artifacts and every refusal are
//! Rust's. No interval here claims coverage: each carries its own calibration standing.

use antecedent::analysis::composition::{
    BundleBuilder, BundleError, BundleStage, ChainBundleRecomputeError,
    CompiledSequentialReliabilityBundle, SequentialReliabilityBundleError,
    SequentialReliabilityBundleReceipt, compile_sequential_reliability_node_bundle,
    consume_portable_sequential_reliability_node_bundle,
};
use antecedent::analysis::observational_policy::{
    CompiledPolicySubsampling, CrossFittedPolicyValueCandidate, ObservationalPolicyReportWire,
    compile_observational_policy, compile_policy_subsampling,
    consume_compiled_observational_policy, consume_policy_subsampling,
};
use antecedent::analysis::sequential_evaluation as seq;
use antecedent::analysis::sequential_policy_reliability::{
    SequentialPolicyReliabilityPlan, SequentialPolicyReliabilityResult, SequentialReliabilityError,
    SequentialReliabilityMode, compile_sequential_policy_reliability,
    consume_sequential_policy_reliability,
};
use antecedent::prelude::ExecutionContext;
use antecedent::scientific::SemanticDigest;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::{Value as Json, json};

use crate::PyCancellationToken;
use crate::bridge::{MAX_DATA_DECLARATION_BYTES, check_declared_size, serialization};
use crate::composition_bundle_api::{PyBundleBuilder, PyCompositionBundle, error_refusal};
use crate::observational_policy_api::policy_request;
use crate::sequential_evaluation_api::{
    PolicyIn, SpecIn, declaration, model_of, policy_from, spec_from, trajectories_of,
};
use crate::transport_common::execution_context;

type Refused<T> = (Option<T>, Option<String>);

const RELIABILITY_STAGE: &str = "sequential_policy_reliability";
const BUNDLE_STAGE: &str = "composition_bundle";

fn cancelled(what: &str) -> PyErr {
    crate::CausalCancelledError::new_err(format!("{what} cancelled"))
}

// ---------------------------------------------------------------------------
// Observational policy value
// ---------------------------------------------------------------------------

/// A policy refusal, or the cancellation that stopped the run.
fn policy_err(ctx: &ExecutionContext, error: antecedent::CausalError) -> PyErr {
    if ctx.cancellation.is_cancelled() {
        cancelled("observational policy")
    } else {
        crate::py_err(error)
    }
}

/// Compile the declared policy request under its split seed, check that the compiled program
/// reproduces `report_json` and seal both. Returns `(bytes, program_identity)`.
#[pyfunction]
#[pyo3(signature = (outcome, treatment, adjustment_names, adjustment_columns, declaration_json, report_json, *, memory_bytes=None, cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "one request: columns, declaration, report and bounds"
)]
fn observational_policy_export(
    py: Python<'_>,
    outcome: Vec<f64>,
    treatment: Vec<f64>,
    adjustment_names: Vec<String>,
    adjustment_columns: Vec<Vec<f64>>,
    declaration_json: &str,
    report_json: &str,
    memory_bytes: Option<u64>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<(Py<PyBytes>, String)> {
    let request =
        policy_request(outcome, treatment, adjustment_names, adjustment_columns, declaration_json)?;
    check_declared_size(report_json, MAX_DATA_DECLARATION_BYTES, "observational policy report")?;
    let report: ObservationalPolicyReportWire =
        serde_json::from_str(report_json).map_err(|error| {
            crate::value_err(format!("invalid observational policy report: {error}"))
        })?;
    let ctx = execution_context(request.spec.seed, memory_bytes, cancel);
    let (bytes, identity) = crate::detach_catch(py, move || {
        let plan = compile_observational_policy(request, &ctx).map_err(|e| policy_err(&ctx, e))?;
        let bytes = plan.export(&report, &ctx).map_err(|e| policy_err(&ctx, e))?;
        Ok((bytes, plan.lowered().identity().to_hex()))
    })?;
    Ok((PyBytes::new(py, &bytes).unbind(), identity))
}

/// Independently recompile and execute sealed policy bytes under the split seed they were
/// produced with, and under the retained program identity when one is given. Returns the
/// replayed report as JSON.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None, seed=1, *, memory_bytes=None, cancel=None))]
fn observational_policy_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity: Option<&str>,
    seed: u64,
    memory_bytes: Option<u64>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<String> {
    let expected =
        expected_identity.map(|hex| crate::digest::digest(hex, "expected_identity")).transpose()?;
    let ctx = execution_context(seed, memory_bytes, cancel);
    crate::detach_catch(py, move || {
        let report = consume_compiled_observational_policy(artifact, expected, &ctx)
            .map_err(|e| policy_err(&ctx, e))?;
        serde_json::to_string(&report).map_err(serialization)
    })
}

/// The candidate as JSON: the report, the interval, its calibration basis and slot, and the
/// identity of the subsampling program.
fn candidate_wire(
    candidate: &CrossFittedPolicyValueCandidate,
    program: SemanticDigest,
) -> PyResult<String> {
    let report = serde_json::to_value(&candidate.report).map_err(serialization)?;
    let basis = serde_json::to_value(&candidate.basis).map_err(serialization)?;
    let slot = serde_json::to_value(&candidate.slot).map_err(serialization)?;
    Ok(json!({
        "report": report,
        "lower": candidate.interval.lower,
        "upper": candidate.interval.upper,
        "level": candidate.interval.level,
        "calibration": candidate.interval.calibration,
        "basis": basis,
        "slot": slot,
        "program_identity": program.to_hex(),
    })
    .to_string())
}

/// A compiled and executed policy subsampling schedule, retained so the candidate it
/// reproduces can be sealed.
#[pyclass(name = "PolicySubsamplingHandle", skip_from_py_object)]
pub(crate) struct PyPolicySubsampling {
    plan: CompiledPolicySubsampling,
    candidate: CrossFittedPolicyValueCandidate,
    seed: u64,
}

#[pymethods]
impl PyPolicySubsampling {
    /// The candidate as JSON (report, interval, calibration basis and slot, program identity).
    #[getter]
    fn candidate_json(&self) -> PyResult<String> {
        candidate_wire(&self.candidate, self.plan.lowered().identity())
    }

    /// The support decision of the unit the compiled program runs under, as JSON; `None` while
    /// no unit is bound to its route.
    fn unit_support(&self) -> Option<String> {
        crate::errors::unit_support_json(&self.plan.plan_view())
    }

    /// Seal the source rows and the candidate after re-executing every refit.
    #[pyo3(signature = (*, memory_bytes=None, cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        memory_bytes: Option<u64>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        let ctx = execution_context(self.seed, memory_bytes, cancel);
        let bytes = crate::detach_catch(py, || {
            self.plan.export(&self.candidate, &ctx).map_err(|e| policy_err(&ctx, e))
        })?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

/// Compile the graph-checked cross-fitted policy learner and its full-refit
/// `subsample_rows`-of-n schedule of `replicates` subsamples, then execute every refit.
#[pyfunction]
#[pyo3(signature = (outcome, treatment, adjustment_names, adjustment_columns, declaration_json, subsample_rows, replicates, *, memory_bytes=None, cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "one request: columns, declaration, schedule and bounds"
)]
fn policy_subsampling_compile(
    py: Python<'_>,
    outcome: Vec<f64>,
    treatment: Vec<f64>,
    adjustment_names: Vec<String>,
    adjustment_columns: Vec<Vec<f64>>,
    declaration_json: &str,
    subsample_rows: usize,
    replicates: usize,
    memory_bytes: Option<u64>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<PyPolicySubsampling> {
    let request =
        policy_request(outcome, treatment, adjustment_names, adjustment_columns, declaration_json)?;
    let seed = request.spec.seed;
    let ctx = execution_context(seed, memory_bytes, cancel);
    let (plan, candidate) = crate::detach_catch(py, move || {
        let plan = compile_policy_subsampling(request, subsample_rows, replicates, &ctx)
            .map_err(|e| policy_err(&ctx, e))?;
        let candidate = plan.execute(&ctx).map_err(|e| policy_err(&ctx, e))?;
        Ok((plan, candidate))
    })?;
    Ok(PyPolicySubsampling { plan, candidate, seed })
}

/// Independently recompile and rerun a sealed subsampling schedule under the retained program
/// identity and the split seed it was produced with. Returns the candidate as JSON.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity, seed=1, *, memory_bytes=None, cancel=None))]
fn policy_subsampling_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity: &str,
    seed: u64,
    memory_bytes: Option<u64>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<String> {
    let expected = crate::digest::digest(expected_identity, "expected_identity")?;
    let ctx = execution_context(seed, memory_bytes, cancel);
    crate::detach_catch(py, move || {
        let candidate = consume_policy_subsampling(artifact, expected, &ctx)
            .map_err(|e| policy_err(&ctx, e))?;
        candidate_wire(&candidate, expected)
    })
}

// ---------------------------------------------------------------------------
// Sequential policy reliability
// ---------------------------------------------------------------------------

/// How the fixed policies are evaluated: common-stream rollouts from the declared model, or a
/// whole-trajectory bootstrap of a logged behaviour policy.
#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
enum ModeIn {
    Rollout {
        seed: u64,
        rollouts: u64,
    },
    OffPolicy {
        behaviour: PolicyIn,
        log: Json,
        seed: u64,
        replicates: u64,
        minimum_effective_sample_size: f64,
    },
}

fn mode_of(
    text: &str,
) -> PyResult<Result<SequentialReliabilityMode, seq::SequentialEvaluationError>> {
    let mode: ModeIn = declaration(text, "reliability mode")?;
    Ok(match mode {
        ModeIn::Rollout { seed, rollouts } => {
            Ok(SequentialReliabilityMode::Rollout(seq::RolloutConfig { seed, rollouts }))
        }
        ModeIn::OffPolicy { behaviour, log, seed, replicates, minimum_effective_sample_size } => {
            let log = trajectories_of(&log.to_string())?;
            policy_from(behaviour).map(|behaviour| SequentialReliabilityMode::OffPolicy {
                behaviour,
                log,
                seed,
                replicates,
                minimum_effective_sample_size,
            })
        }
    })
}

/// A refused declaration, or on replay a refused binding of the bytes to the retained
/// program identity or the recomputed diagnostic.
fn reliability_invalid_json(message: &str, replay: bool) -> String {
    let (code, detail) = if replay {
        (
            antecedent::reason::reason_code!("external_binding_mismatch"),
            "sequential_policy_reliability.replay_mismatch",
        )
    } else {
        (
            antecedent::reason::reason_code!("invalid_argument"),
            "sequential_policy_reliability.invalid_declaration",
        )
    };
    json!({
        "code": code,
        "stage": RELIABILITY_STAGE,
        "detail": detail,
        "message": message,
        "offending": null,
        "remedy": null,
    })
    .to_string()
}

/// The refusal JSON of a reliability error; an unreadable artifact raises its serialization
/// error.
fn reliability_refusal(error: SequentialReliabilityError, replay: bool) -> PyResult<String> {
    match error {
        SequentialReliabilityError::Invalid(message)
            if message.starts_with(antecedent::reason::reason_code::PREFIX)
                && message[antecedent::reason::reason_code::PREFIX.len()..]
                    .starts_with("not_supported:") =>
        {
            Err(crate::errors::unsupported_message(message))
        }
        SequentialReliabilityError::Invalid(message) => {
            Ok(reliability_invalid_json(message, replay))
        }
        SequentialReliabilityError::Evaluation(error) => {
            Ok(crate::sequential_evaluation_api::refusal(&error))
        }
        SequentialReliabilityError::Reliability(error) => {
            Ok(crate::decision_reliability_api::error_json(&error))
        }
        SequentialReliabilityError::Io(error) => Err(crate::py_err(error)),
    }
}

/// The aligned draws, the diagnostic report and the program identity of one evaluation.
fn reliability_parts(
    result: &SequentialPolicyReliabilityResult,
) -> PyResult<(String, String, String)> {
    Ok((
        serde_json::to_string(&result.draws).map_err(serialization)?,
        serde_json::to_string(&result.report).map_err(serialization)?,
        result.lowered_program_identity.to_hex(),
    ))
}

/// A compiled and executed sequential reliability program, retained so its result can be
/// sealed or embedded in a bundle.
#[pyclass(name = "SequentialPolicyReliabilityHandle", skip_from_py_object)]
pub(crate) struct PySequentialPolicyReliability {
    plan: SequentialPolicyReliabilityPlan,
    result: SequentialPolicyReliabilityResult,
    seed: u64,
}

#[pymethods]
impl PySequentialPolicyReliability {
    /// `(draws_json, report_json, program_identity)` of the executed program.
    #[getter]
    fn parts(&self) -> PyResult<(String, String, String)> {
        reliability_parts(&self.result)
    }

    /// The compiled route the program was lowered for.
    #[getter]
    fn route(&self) -> &'static str {
        self.plan.lowered().route()
    }

    /// The support decision of the unit the compiled program runs under, as JSON; `None` while
    /// no unit is bound to its route.
    fn unit_support(&self) -> Option<String> {
        crate::errors::unit_support_json(&self.plan.plan_view())
    }

    /// Seal the raw declaration after re-evaluating it. Returns `(bytes, refusal)`.
    #[pyo3(signature = (cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<Refused<Py<PyBytes>>> {
        let ctx = execution_context(self.seed, None, cancel);
        let exported = crate::detach_catch(py, || Ok(self.plan.export(&self.result, &ctx)))?;
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled(RELIABILITY_STAGE));
        }
        match exported {
            Ok(bytes) => Ok((Some(PyBytes::new(py, &bytes).unbind()), None)),
            Err(error) => Ok((None, Some(reliability_refusal(error, false)?))),
        }
    }
}

/// Compile fixed sequential policies on a declared model, evaluate them and assess the
/// decision among them. `policies_json` is a list of policy specifications; `mode_json` is
/// `{"rollout": {...}}` or `{"off_policy": {...}}`. Returns `(handle, refusal)`.
#[pyfunction]
#[pyo3(signature = (model_json, policies_json, unit, mode_json, seed=1, cancel=None))]
fn sequential_policy_reliability_compile(
    py: Python<'_>,
    model_json: &str,
    policies_json: &str,
    unit: String,
    mode_json: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<PySequentialPolicyReliability>> {
    let model = model_of(model_json)?;
    let specs: Vec<SpecIn> = declaration(policies_json, "sequential policies")?;
    let mut policies = Vec::with_capacity(specs.len());
    for spec in specs {
        match spec_from(spec) {
            Ok(policy) => policies.push(policy),
            Err(error) => {
                return Ok((None, Some(crate::sequential_evaluation_api::refusal(&error))));
            }
        }
    }
    let mode = match mode_of(mode_json)? {
        Ok(mode) => mode,
        Err(error) => return Ok((None, Some(crate::sequential_evaluation_api::refusal(&error)))),
    };
    let ctx = execution_context(seed, None, cancel);
    let outcome = crate::detach_catch(py, || {
        Ok(compile_sequential_policy_reliability(model, policies, unit, mode)
            .and_then(|plan| plan.execute(&ctx).map(|result| (plan, result))))
    })?;
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled(RELIABILITY_STAGE));
    }
    match outcome {
        Ok((plan, result)) => {
            Ok((Some(PySequentialPolicyReliability { plan, result, seed }), None))
        }
        Err(error) => Ok((None, Some(reliability_refusal(error, false)?))),
    }
}

/// Independently consume sealed reliability bytes by recompiling and re-evaluating the raw
/// declaration, under the retained program identity when one is given. Returns
/// `((draws_json, report_json, program_identity), refusal)`.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None, seed=1, cancel=None))]
fn sequential_policy_reliability_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity: Option<&str>,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<(String, String, String)>> {
    let expected =
        expected_identity.map(|hex| crate::digest::digest(hex, "expected_identity")).transpose()?;
    let ctx = execution_context(seed, None, cancel);
    let outcome = crate::detach_catch(py, || {
        Ok(consume_sequential_policy_reliability(artifact, expected, &ctx))
    })?;
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled(RELIABILITY_STAGE));
    }
    match outcome {
        Ok(result) => Ok((Some(reliability_parts(&result)?), None)),
        Err(error) => Ok((None, Some(reliability_refusal(error, true)?))),
    }
}

// ---------------------------------------------------------------------------
// Sequential reliability bundle
// ---------------------------------------------------------------------------

/// A bundle refusal at `stage` for a binding the sequential route checks itself.
fn stage_refusal(stage: BundleStage, message: &str) -> String {
    error_refusal(&BundleError::Refused { stage, reason: message.to_owned(), offending: None })
}

/// The refusal JSON of a bundle error; `stage` names where an unnamed binding failed. A
/// release build's unsupported route and an unreadable artifact raise.
fn bundle_refusal(error: SequentialReliabilityBundleError, stage: BundleStage) -> PyResult<String> {
    match error {
        SequentialReliabilityBundleError::NotSupported(refusal) => {
            Err(crate::not_supported_err(&refusal))
        }
        SequentialReliabilityBundleError::Invalid(message)
        | SequentialReliabilityBundleError::Bundle(ChainBundleRecomputeError::Invalid(message)) => {
            Ok(stage_refusal(stage, message))
        }
        SequentialReliabilityBundleError::Bundle(ChainBundleRecomputeError::Bundle(error)) => {
            Ok(error_refusal(&error))
        }
        SequentialReliabilityBundleError::Bundle(ChainBundleRecomputeError::Chain(refusal)) => {
            Ok(crate::sampling_assumption_union_api::chain_refusal_json(&refusal, BUNDLE_STAGE))
        }
        SequentialReliabilityBundleError::Bundle(ChainBundleRecomputeError::Io(error)) => {
            Err(crate::py_err(error))
        }
    }
}

/// Why the producer, carrier and diagnostic nodes could not be staged.
enum StageFailure {
    Producer(SequentialReliabilityError),
    Node(SequentialReliabilityBundleError),
    Graph(BundleError),
}

/// Stage the producer, its aligned draws and its diagnostic on a copy of the builder, each
/// verified as it is embedded, connected producer to draws to diagnostic.
fn stage_sequential(
    mut builder: BundleBuilder,
    reliability: &PySequentialPolicyReliability,
    ids: [&str; 3],
    ctx: &ExecutionContext,
) -> Result<BundleBuilder, StageFailure> {
    let [producer, draws, diagnostic] = ids;
    let artifact =
        reliability.plan.export(&reliability.result, ctx).map_err(StageFailure::Producer)?;
    builder
        .add_sequential_policy_reliability_fit(
            producer,
            &artifact,
            reliability.result.lowered_program_identity,
            ctx,
        )
        .map_err(StageFailure::Node)?;
    builder
        .add_sequential_decision_draws(draws, &reliability.result.draws)
        .map_err(StageFailure::Node)?;
    builder
        .add_sequential_reliability_diagnostic(
            diagnostic,
            &reliability.result.draws,
            &reliability.result.report,
            ctx,
        )
        .map_err(StageFailure::Node)?;
    builder.connect(producer, draws).map_err(StageFailure::Graph)?;
    builder.connect(draws, diagnostic).map_err(StageFailure::Graph)?;
    Ok(builder)
}

/// Embed a sequential reliability in a bundle under construction: its sealed producer, its
/// aligned decision draws and its diagnostic, connected in that order. Every node is verified
/// by recomputation before it is accepted; on a refusal the builder is unchanged. Returns the
/// refusal, if any.
#[pyfunction]
#[pyo3(signature = (builder, reliability, producer, draws, diagnostic, cancel=None))]
fn composition_add_sequential_policy_reliability(
    py: Python<'_>,
    mut builder: PyRefMut<'_, PyBundleBuilder>,
    reliability: PyRef<'_, PySequentialPolicyReliability>,
    producer: &str,
    draws: &str,
    diagnostic: &str,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Option<String>> {
    let ctx = execution_context(reliability.seed, None, cancel);
    let staged_from = builder.inner.clone();
    let handle: &PySequentialPolicyReliability = &reliability;
    let staged = crate::detach_catch(py, || {
        Ok(stage_sequential(staged_from, handle, [producer, draws, diagnostic], &ctx))
    })?;
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled(BUNDLE_STAGE));
    }
    match staged {
        Ok(inner) => {
            builder.inner = inner;
            Ok(None)
        }
        Err(StageFailure::Producer(error)) => Ok(Some(reliability_refusal(error, false)?)),
        Err(StageFailure::Node(error)) => {
            Ok(Some(bundle_refusal(error, BundleStage::SwappedEvidence)?))
        }
        Err(StageFailure::Graph(error)) => Ok(Some(error_refusal(&error))),
    }
}

fn receipt_json(receipt: &SequentialReliabilityBundleReceipt) -> String {
    let dependencies: Vec<Json> = receipt
        .dependencies
        .iter()
        .map(|dependency| {
            json!({
                "node_id": dependency.node_id,
                "artifact_identity": dependency.artifact_identity,
                "chain_digest": dependency.chain_digest,
            })
        })
        .collect();
    json!({
        "bundle_identity": receipt.bundle_identity,
        "draws_digest": receipt.draws_digest,
        "diagnostic_digest": receipt.diagnostic_digest,
        "producer_program_identity":
            SemanticDigest::from_bytes(receipt.producer_program_identity).to_hex(),
        "program_identity": SemanticDigest::from_bytes(receipt.lowered_program_identity).to_hex(),
        "dependencies": dependencies,
    })
    .to_string()
}

/// A compiled and executed sequential reliability bundle, retained so its receipt can be
/// sealed with the bundle.
#[pyclass(name = "SequentialReliabilityBundleHandle", skip_from_py_object)]
pub(crate) struct PySequentialReliabilityBundle {
    plan: CompiledSequentialReliabilityBundle,
    receipt: SequentialReliabilityBundleReceipt,
    seed: u64,
}

#[pymethods]
impl PySequentialReliabilityBundle {
    /// The live receipt as JSON: bundle, draw and diagnostic digests, the producer and
    /// bundle program identities and every replayed dependency.
    #[getter]
    fn receipt_json(&self) -> String {
        receipt_json(&self.receipt)
    }

    /// The compiled route the program was lowered for.
    #[getter]
    fn route(&self) -> &'static str {
        self.plan.lowered().route()
    }

    /// The support decision of the unit the compiled program runs under, as JSON; `None` while
    /// no unit is bound to its route.
    fn unit_support(&self) -> Option<String> {
        crate::errors::unit_support_json(&self.plan.plan_view())
    }

    /// Seal the bundle and its receipt after recomputing it. Returns `(bytes, refusal)`.
    #[pyo3(signature = (cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<Refused<Py<PyBytes>>> {
        let ctx = execution_context(self.seed, None, cancel);
        let exported =
            crate::detach_catch(py, || Ok(self.plan.export_portable(&self.receipt, &ctx)))?;
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled(BUNDLE_STAGE));
        }
        match exported {
            Ok(bytes) => Ok((Some(PyBytes::new(py, &bytes).unbind()), None)),
            Err(error) => Ok((None, Some(bundle_refusal(error, BundleStage::SwappedEvidence)?))),
        }
    }
}

/// Compile the direct producer, draw carrier and diagnostic graph of a built bundle and
/// recompute it: the producer is replayed from its raw declaration and the diagnostic is
/// reassessed from the retained draws. Returns `(handle, refusal)`.
#[pyfunction]
#[pyo3(signature = (bundle, producer, draws, diagnostic, seed=1, cancel=None))]
fn sequential_reliability_bundle_compile(
    py: Python<'_>,
    bundle: PyRef<'_, PyCompositionBundle>,
    producer: &str,
    draws: &str,
    diagnostic: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<PySequentialReliabilityBundle>> {
    let graph = bundle.bundle.clone();
    let ctx = execution_context(seed, None, cancel);
    let outcome = crate::detach_catch(py, || {
        Ok(match compile_sequential_reliability_node_bundle(graph, producer, draws, diagnostic) {
            Ok(plan) => plan
                .execute(&ctx)
                .map(|receipt| (plan, receipt))
                .map_err(|error| (error, BundleStage::SwappedEvidence)),
            Err(error) => Err((error, BundleStage::EdgeDigestMismatch)),
        })
    })?;
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled(BUNDLE_STAGE));
    }
    match outcome {
        Ok((plan, receipt)) => {
            Ok((Some(PySequentialReliabilityBundle { plan, receipt, seed }), None))
        }
        Err((error, stage)) => Ok((None, Some(bundle_refusal(error, stage)?))),
    }
}

/// Independently reconstruct a sealed sequential reliability bundle under the retained bundle
/// identity and program identity, and recompute its producer and diagnostic. Returns
/// `(receipt_json, refusal)`.
#[pyfunction]
#[pyo3(signature = (artifact, expected_bundle, expected_identity, seed=1, cancel=None))]
fn sequential_reliability_bundle_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_bundle: &str,
    expected_identity: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<String>> {
    let expected = crate::digest::digest(expected_identity, "expected_identity")?;
    let ctx = execution_context(seed, None, cancel);
    let outcome = crate::detach_catch(py, || {
        Ok(consume_portable_sequential_reliability_node_bundle(
            artifact,
            expected_bundle,
            expected,
            &ctx,
        ))
    })?;
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled(BUNDLE_STAGE));
    }
    match outcome {
        Ok(receipt) => Ok((Some(receipt_json(&receipt)), None)),
        Err(error) => {
            Ok((None, Some(bundle_refusal(error, BundleStage::ExpectedIdentityMismatch)?)))
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPolicySubsampling>()?;
    m.add_class::<PySequentialPolicyReliability>()?;
    m.add_class::<PySequentialReliabilityBundle>()?;
    m.add_function(wrap_pyfunction!(observational_policy_export, m)?)?;
    m.add_function(wrap_pyfunction!(observational_policy_consume, m)?)?;
    m.add_function(wrap_pyfunction!(policy_subsampling_compile, m)?)?;
    m.add_function(wrap_pyfunction!(policy_subsampling_consume, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_policy_reliability_compile, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_policy_reliability_consume, m)?)?;
    m.add_function(wrap_pyfunction!(composition_add_sequential_policy_reliability, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_reliability_bundle_compile, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_reliability_bundle_consume, m)?)?;
    Ok(())
}
