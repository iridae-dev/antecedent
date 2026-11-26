//! Python bridge for decisions compiled together with a propagation chain.
//!
//! Python hands over the chain record wire (the JSON of a declared or sealed chain), the
//! decision contract declaration and, for a transported law, the transported response with its
//! support policy. The joint compiler, the six-layer program identity, the execution, the
//! artifact and the independent replay are Rust's. A refusal comes back as structured JSON for
//! the Python layer to raise as a typed exception; a cancelled run raises
//! `CausalCancelledError`.

use antecedent::analysis::propagation_chain::{ChainRefusal, PropagationChainRecordWire};
use antecedent::analysis::{
    ChainDecisionError, ChainDecisionPlan, ChainDecisionResult, ChainRobustDecisionPlan,
    ChainRobustDecisionResult, compile_action_world_chain_decision, compile_chain_decision,
    compile_chain_robust_decision, compile_transported_law_chain_decision, consume_chain_decision,
    consume_chain_robust_decision,
};
use antecedent::design::decision_artifact::contract_from_json_refusal;
use antecedent::design::decision_eval::{DecisionResult, Verdict};
use antecedent::io::IoError;
use antecedent::reason::reason_code;
use antecedent::scientific::SemanticDigest;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Value as Json, json};

use crate::PyCancellationToken;
use crate::bridge::{MAX_DATA_DECLARATION_BYTES, RefusalJson, external_refusal_json};
use crate::propagation_chain_api::{ChainParts, chain_parts};
use crate::transported_response_api::{TransportedResponseHandle, support_policy_of};

type Refused<T> = (Option<T>, Option<String>);

const STAGE: &str = "chain_decision";
// Family-local details of this bridge; a chain or contract refusal keeps its own detail.
const INVALID_REQUEST: &str = "chain_decision.invalid_request";
const COMPILE_REFUSED: &str = "chain_decision.refused";
const REPLAY_REFUSED: &str = "chain_decision.replay_refused";
const EVALUATION_REFUSED: &str = "chain_decision.evaluation_refused";
const ARTIFACT_INVALID: &str = "chain_decision.artifact_invalid";
const NOT_EXPORTABLE: &str = "chain_decision.not_exportable";

fn cancelled() -> PyErr {
    crate::CausalCancelledError::new_err("chain decision cancelled")
}

/// The refusal JSON of a chain refusal: its code and detail, the refused chain stage as
/// `offending`, and the next attempt that would carry the uncertainty.
pub(crate) fn chain_refusal_json(refusal: &ChainRefusal, stage: &str) -> String {
    json!({
        "code": refusal.code(),
        "stage": stage,
        "detail": refusal.detail(),
        "message": refusal.to_string(),
        "offending": refusal.stage(),
        "next_attempt": refusal.next_attempt().name(),
        "remedy": refusal.next_attempt().sentence(),
    })
    .to_string()
}

/// The refusal JSON of an artifact error: a coded refusal keeps its code and detail, any other
/// decoding or encoding failure is an invalid argument under `detail`.
pub(crate) fn io_refusal_json(error: &IoError, stage: &str, detail: &str) -> String {
    match error {
        IoError::Refused { code, message } => {
            RefusalJson::from_coded_message(code, stage, message).json()
        }
        other => {
            RefusalJson::new(reason_code!("invalid_argument"), stage, detail, &other.to_string())
                .json()
        }
    }
}

/// The refusal JSON of a compiler, execution or replay error. `replay` marks the independent
/// consumer, where an unmet binding is a mismatch with the retained artifact rather than a
/// contract the source cannot answer.
fn error_json(error: &ChainDecisionError, replay: bool) -> String {
    match error {
        ChainDecisionError::Chain(refusal) => chain_refusal_json(refusal, STAGE),
        ChainDecisionError::Contract(contract) => external_refusal_json(&contract.to_refusal()),
        ChainDecisionError::Evaluation(evaluation) => RefusalJson::new(
            reason_code!("decision_contract_unsatisfied"),
            STAGE,
            EVALUATION_REFUSED,
            &evaluation.to_string(),
        )
        .json(),
        ChainDecisionError::Io(io) => io_refusal_json(io, STAGE, ARTIFACT_INVALID),
        ChainDecisionError::Coordinate(why) => {
            if replay {
                RefusalJson::new(
                    reason_code!("external_binding_mismatch"),
                    STAGE,
                    REPLAY_REFUSED,
                    why,
                )
                .json()
            } else {
                RefusalJson::new(
                    reason_code!("decision_contract_unsatisfied"),
                    STAGE,
                    COMPILE_REFUSED,
                    why,
                )
                .json()
            }
        }
    }
}

fn invalid_json(message: &str) -> String {
    RefusalJson::new(reason_code!("invalid_argument"), STAGE, INVALID_REQUEST, message).json()
}

fn parse_record(record_json: &str) -> Result<PropagationChainRecordWire, String> {
    if record_json.len() > MAX_DATA_DECLARATION_BYTES {
        return Err(invalid_json("the chain record exceeds its byte bound"));
    }
    serde_json::from_str(record_json).map_err(|error| invalid_json(&error.to_string()))
}

fn expected_digest(expected: Option<&str>) -> PyResult<Option<SemanticDigest>> {
    expected.map(|hex| crate::digest::digest(hex, "expected_identity")).transpose()
}

/// An execution context from the loose controls an `Execution` hands a native entry.
pub(crate) fn context(
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> antecedent::prelude::ExecutionContext {
    let mut ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    ctx.set_memory_limit(memory_bytes);
    ctx
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
        "method_provenance": result.method_provenance(),
        "source": {
            "provider_id": result.source.provider_id,
            "snapshot_id": result.source.snapshot_id,
            "rng_id": result.source.rng_id,
            "causal_contract_id": result.source.causal_contract_id,
        },
    })
}

fn robust_json(result: &ChainRobustDecisionResult) -> Json {
    json!({
        "contract_identity": result.contract_identity,
        "dominant_action": result.dominant_action,
        "actions": result.actions.iter().map(|action| json!({
            "id": action.id,
            "utility_lower": action.utility_lower,
            "utility_upper": action.utility_upper,
            "certainly_admissible": action.certainly_admissible,
            "certainly_excluded": action.certainly_excluded,
            "constraints": action.constraints.iter().map(|constraint| json!({
                "id": constraint.id,
                "probability_lower": constraint.probability_lower,
                "probability_upper": constraint.probability_upper,
                "required": constraint.required,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

/// A chain law and a multi-action decision contract compiled into one program.
#[pyclass(name = "ChainDecisionPlanHandle", skip_from_py_object)]
pub(crate) struct PyChainDecisionPlan {
    plan: ChainDecisionPlan,
}

#[pymethods]
impl PyChainDecisionPlan {
    /// Identity of the six-layer program binding the chain and the contract (lowercase hex).
    #[getter]
    fn program_identity(&self) -> String {
        self.plan.lowered().identity().to_hex()
    }

    /// Every compiler layer as JSON, with nothing executed.
    fn plan_view_json(&self) -> String {
        self.plan.plan_view().to_json().to_string()
    }

    /// Execute the retained chain and score the contract on its aligned output law.
    #[pyo3(signature = (*, seed=1, threads=None, cancel=None, memory_bytes=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> PyResult<Refused<PyChainDecisionRun>> {
        let ctx = context(seed, threads, cancel, memory_bytes);
        let plan = &self.plan;
        let (executed, stopped) = crate::detach_catch(py, move || {
            let executed = plan.execute(&ctx);
            Ok((executed, ctx.cancellation.is_cancelled()))
        })?;
        match executed {
            Ok(result) => {
                Ok((Some(PyChainDecisionRun { plan: Some(self.plan.clone()), result }), None))
            }
            Err(_) if stopped => Err(cancelled()),
            Err(error) => Ok((None, Some(error_json(&error, false)))),
        }
    }
}

/// An executed or independently replayed chain decision.
#[pyclass(name = "ChainDecisionRunHandle", skip_from_py_object)]
pub(crate) struct PyChainDecisionRun {
    plan: Option<ChainDecisionPlan>,
    result: ChainDecisionResult,
}

#[pymethods]
impl PyChainDecisionRun {
    /// The program that executed the decision (lowercase hex).
    #[getter]
    fn program_identity(&self) -> String {
        self.result.lowered_program_identity.to_hex()
    }

    /// The scored decision as JSON.
    fn decision_json(&self) -> String {
        decision_json(&self.result.decision).to_string()
    }

    /// The executed chain: record bytes, record JSON, program identity, law meaning, alignment.
    fn chain(&self, py: Python<'_>) -> Refused<ChainParts> {
        chain_parts(py, &self.result.chain)
    }

    /// The versioned artifact a consumer replays; only the producing plan exports it.
    fn export(&self, py: Python<'_>, artifact_id: &str) -> Refused<Py<PyBytes>> {
        let Some(plan) = &self.plan else {
            return (
                None,
                Some(
                    RefusalJson::new(
                        reason_code!("route_not_supported"),
                        STAGE,
                        NOT_EXPORTABLE,
                        "a replayed chain decision is forwarded as the bytes it was read from",
                    )
                    .json(),
                ),
            );
        };
        match plan.export(&self.result, artifact_id) {
            Ok(bytes) => (Some(PyBytes::new(py, &bytes).unbind()), None),
            Err(error) => (None, Some(error_json(&error, false))),
        }
    }
}

/// Compile a chain record and a decision contract into one program. `action_worlds` declares
/// that each action reads its own intervention regime of one jointly aligned draw table.
#[pyfunction]
#[pyo3(signature = (record_json, contract_json, *, action_worlds=false, cancel=None))]
fn chain_decision_compile(
    record_json: &str,
    contract_json: &str,
    action_worlds: bool,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<PyChainDecisionPlan>> {
    crate::catch_ffi(|| {
        let draft = match parse_record(record_json) {
            Ok(draft) => draft,
            Err(refusal) => return Ok((None, Some(refusal))),
        };
        let contract = match contract_from_json_refusal(contract_json) {
            Ok(contract) => contract,
            Err(refusal) => return Ok((None, Some(external_refusal_json(&refusal)))),
        };
        let ctx = context(0, Some(1), cancel, None);
        let compiled = if action_worlds {
            compile_action_world_chain_decision(draft, contract, &ctx)
        } else {
            compile_chain_decision(draft, contract, &ctx)
        };
        match compiled {
            Ok(plan) => Ok((Some(PyChainDecisionPlan { plan }), None)),
            Err(_) if ctx.cancellation.is_cancelled() => Err(cancelled()),
            Err(error) => Ok((None, Some(error_json(&error, false)))),
        }
    })
}

/// Compile a checked transported law and a decision contract through the chain compiler,
/// accepted under the declared support policy.
#[pyfunction]
#[pyo3(signature = (response, contract_json, unsupported, weakest_support, *, cancel=None))]
fn chain_decision_compile_transported(
    response: PyRef<'_, TransportedResponseHandle>,
    contract_json: &str,
    unsupported: &str,
    weakest_support: &str,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<PyChainDecisionPlan>> {
    crate::catch_ffi(|| {
        let policy = support_policy_of(unsupported, weakest_support)?;
        let contract = match contract_from_json_refusal(contract_json) {
            Ok(contract) => contract,
            Err(refusal) => return Ok((None, Some(external_refusal_json(&refusal)))),
        };
        let ctx = context(0, Some(1), cancel, None);
        match compile_transported_law_chain_decision(&response.inner, contract, &policy, &ctx) {
            Ok(plan) => Ok((Some(PyChainDecisionPlan { plan }), None)),
            Err(_) if ctx.cancellation.is_cancelled() => Err(cancelled()),
            Err(error) => Ok((None, Some(error_json(&error, false)))),
        }
    })
}

/// Independently consume a chain decision artifact: the chain record and contract are
/// consumed, recompiled into their joint program, re-executed and compared bit for bit.
#[pyfunction]
#[pyo3(signature = (data, expected_identity=None, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
fn chain_decision_consume(
    py: Python<'_>,
    data: &[u8],
    expected_identity: Option<&str>,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Refused<PyChainDecisionRun>> {
    let expected = expected_digest(expected_identity)?;
    let ctx = context(seed, threads, cancel, memory_bytes);
    let (consumed, stopped) = crate::detach_catch(py, move || {
        let consumed = consume_chain_decision(data, expected, &ctx);
        Ok((consumed, ctx.cancellation.is_cancelled()))
    })?;
    match consumed {
        Ok(result) => Ok((Some(PyChainDecisionRun { plan: None, result }), None)),
        Err(_) if stopped => Err(cancelled()),
        Err(error) => Ok((None, Some(error_json(&error, true)))),
    }
}

/// A conservative dominance decision over a chain's draw-indexed assumption envelope.
#[pyclass(name = "ChainRobustDecisionPlanHandle", skip_from_py_object)]
pub(crate) struct PyChainRobustDecisionPlan {
    plan: ChainRobustDecisionPlan,
}

#[pymethods]
impl PyChainRobustDecisionPlan {
    /// Identity of the six-layer program binding the chain and the contract (lowercase hex).
    #[getter]
    fn program_identity(&self) -> String {
        self.plan.lowered().identity().to_hex()
    }

    /// Every compiler layer as JSON, with nothing executed.
    fn plan_view_json(&self) -> String {
        self.plan.plan_view().to_json().to_string()
    }

    /// Replay the rowwise assumption bounds and bound every action's utility and constraints.
    #[pyo3(signature = (*, seed=1, threads=None, cancel=None, memory_bytes=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> PyResult<Refused<PyChainRobustDecisionRun>> {
        let ctx = context(seed, threads, cancel, memory_bytes);
        let plan = &self.plan;
        let (executed, stopped) = crate::detach_catch(py, move || {
            let executed = plan.execute(&ctx);
            Ok((executed, ctx.cancellation.is_cancelled()))
        })?;
        match executed {
            Ok(result) => {
                Ok((Some(PyChainRobustDecisionRun { plan: Some(self.plan.clone()), result }), None))
            }
            Err(_) if stopped => Err(cancelled()),
            Err(error) => Ok((None, Some(error_json(&error, false)))),
        }
    }
}

/// An executed or independently replayed robust chain decision.
#[pyclass(name = "ChainRobustDecisionRunHandle", skip_from_py_object)]
pub(crate) struct PyChainRobustDecisionRun {
    plan: Option<ChainRobustDecisionPlan>,
    result: ChainRobustDecisionResult,
}

#[pymethods]
impl PyChainRobustDecisionRun {
    /// The program that executed the decision (lowercase hex).
    #[getter]
    fn program_identity(&self) -> String {
        self.result.lowered_program_identity.to_hex()
    }

    /// The action bounds and the dominant action as JSON.
    fn decision_json(&self) -> String {
        robust_json(&self.result).to_string()
    }

    /// The executed chain: record bytes, record JSON, program identity, law meaning, alignment.
    fn chain(&self, py: Python<'_>) -> Refused<ChainParts> {
        chain_parts(py, &self.result.chain)
    }

    /// The versioned artifact a consumer replays; only the producing plan exports it.
    fn export(&self, py: Python<'_>, artifact_id: &str) -> Refused<Py<PyBytes>> {
        let Some(plan) = &self.plan else {
            return (
                None,
                Some(
                    RefusalJson::new(
                        reason_code!("route_not_supported"),
                        STAGE,
                        NOT_EXPORTABLE,
                        "a replayed robust decision is forwarded as the bytes it was read from",
                    )
                    .json(),
                ),
            );
        };
        match plan.export(&self.result, artifact_id) {
            Ok(bytes) => (Some(PyBytes::new(py, &bytes).unbind()), None),
            Err(error) => (None, Some(error_json(&error, false))),
        }
    }
}

/// Compile a maximin decision over a chain's assumption envelope into one program.
#[pyfunction]
#[pyo3(signature = (record_json, contract_json, *, cancel=None))]
fn chain_robust_decision_compile(
    record_json: &str,
    contract_json: &str,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<PyChainRobustDecisionPlan>> {
    crate::catch_ffi(|| {
        let draft = match parse_record(record_json) {
            Ok(draft) => draft,
            Err(refusal) => return Ok((None, Some(refusal))),
        };
        let contract = match contract_from_json_refusal(contract_json) {
            Ok(contract) => contract,
            Err(refusal) => return Ok((None, Some(external_refusal_json(&refusal)))),
        };
        let ctx = context(0, Some(1), cancel, None);
        match compile_chain_robust_decision(draft, contract, &ctx) {
            Ok(plan) => Ok((Some(PyChainRobustDecisionPlan { plan }), None)),
            Err(_) if ctx.cancellation.is_cancelled() => Err(cancelled()),
            Err(error) => Ok((None, Some(error_json(&error, false)))),
        }
    })
}

/// Independently consume a robust chain decision artifact under the retained program identity.
#[pyfunction]
#[pyo3(signature = (data, expected_identity=None, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
fn chain_robust_decision_consume(
    py: Python<'_>,
    data: &[u8],
    expected_identity: Option<&str>,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Refused<PyChainRobustDecisionRun>> {
    let expected = expected_digest(expected_identity)?;
    let ctx = context(seed, threads, cancel, memory_bytes);
    let (consumed, stopped) = crate::detach_catch(py, move || {
        let consumed = consume_chain_robust_decision(data, expected, &ctx);
        Ok((consumed, ctx.cancellation.is_cancelled()))
    })?;
    match consumed {
        Ok(result) => Ok((Some(PyChainRobustDecisionRun { plan: None, result }), None)),
        Err(_) if stopped => Err(cancelled()),
        Err(error) => Ok((None, Some(error_json(&error, true)))),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyChainDecisionPlan>()?;
    module.add_class::<PyChainDecisionRun>()?;
    module.add_class::<PyChainRobustDecisionPlan>()?;
    module.add_class::<PyChainRobustDecisionRun>()?;
    module.add_function(wrap_pyfunction!(chain_decision_compile, module)?)?;
    module.add_function(wrap_pyfunction!(chain_decision_compile_transported, module)?)?;
    module.add_function(wrap_pyfunction!(chain_decision_consume, module)?)?;
    module.add_function(wrap_pyfunction!(chain_robust_decision_compile, module)?)?;
    module.add_function(wrap_pyfunction!(chain_robust_decision_consume, module)?)?;
    Ok(())
}
