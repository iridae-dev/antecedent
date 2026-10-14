//! Python bridge for cross-world selective execution.
//!
//! Python hands over a declaration as JSON (the wire form of the request) and
//! gets back JSON. Which requests are executed, the abduction, the shared
//! exogenous states, the reuse decision, the pairing rule and every refusal are
//! Rust's. A refusal comes back as structured JSON for the Python layer to
//! raise as a typed exception. Nothing here pairs draws.

use antecedent::analysis::cross_world_execution::{
    CrossWorldExecution, CrossWorldExecutionArtifactError, CrossWorldExecutionError,
    CrossWorldExecutionIdentity, FiniteExecutionResult, FiniteScmExecutor, SourceTreatment,
    UnitSampling, WholePathAccount, capability_matrix,
};
use antecedent_io::cross_world_execution_artifact::{
    CrossWorldExecutionArtifact, CrossWorldRequestWire, request_of,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Value as Json, json};

use crate::PyCancellationToken;
use crate::transport_common::execution_context;

type Refused<T> = (Option<T>, Option<String>);

const STAGE: &str = "execute";

fn invalid_json(detail: &str, message: String) -> String {
    json!({
        "code": antecedent_core::reason_code!("invalid_argument"),
        "stage": STAGE,
        "detail": detail,
        "message": message,
        "offending": Json::Null,
        "remedy": Json::Null,
        "missing": Vec::<String>::new(),
    })
    .to_string()
}

fn core_refusal(error: &CrossWorldExecutionError) -> String {
    match error {
        CrossWorldExecutionError::Refused(r) => json!({
            "code": r.refusal.code,
            "stage": r.refusal.stage,
            "detail": r.refusal.detail,
            "message": error.to_string(),
            "offending": r.refusal.offending,
            "remedy": r.refusal.remedy,
            "missing": r.missing,
        })
        .to_string(),
        _ => json!({
            "code": antecedent_core::reason_code!("transport_budget_cancel"),
            "stage": STAGE,
            "detail": "cross_world_execution.cancelled",
            "message": error.to_string(),
            "offending": Json::Null,
            "remedy": Json::Null,
            "missing": Vec::<String>::new(),
        })
        .to_string(),
    }
}

fn artifact_refusal(error: &CrossWorldExecutionArtifactError) -> String {
    if let Some(w) = error.refusal() {
        json!({
            "code": w.code, "stage": w.stage, "detail": w.detail, "message": error.to_string(),
            "offending": w.offending, "remedy": w.remedy, "missing": w.missing,
        })
        .to_string()
    } else {
        let detail = match error {
            CrossWorldExecutionArtifactError::Cancelled => "cross_world_execution.cancelled",
            CrossWorldExecutionArtifactError::ResultMismatch(_) => {
                "cross_world_execution.artifact_changed"
            }
            _ => "cross_world_execution.invalid_artifact",
        };
        let code = if matches!(error, CrossWorldExecutionArtifactError::Cancelled) {
            antecedent_core::reason_code!("transport_budget_cancel")
        } else if matches!(error, CrossWorldExecutionArtifactError::ResultMismatch(_)) {
            antecedent_core::reason_code!("route_not_supported")
        } else {
            antecedent_core::reason_code!("invalid_argument")
        };
        json!({
            "code": code, "stage": "consume", "detail": detail, "message": error.to_string(),
            "offending": Json::Null, "remedy": Json::Null, "missing": Vec::<String>::new(),
        })
        .to_string()
    }
}

fn parse_request(
    request_json: &str,
) -> Result<antecedent::analysis::cross_world_execution::FiniteExecutionRequest, String> {
    let wire: CrossWorldRequestWire = serde_json::from_str(request_json)
        .map_err(|e| invalid_json("cross_world_execution.invalid_request", e.to_string()))?;
    request_of(&wire).map_err(|e| artifact_refusal(&e))
}

fn result_json(result: &FiniteExecutionResult) -> Json {
    let w = &result.work;
    json!({
        "matrix_row": result.matrix_row,
        "value": result.value,
        "label": result.label,
        "units": result.units.iter().map(|u| json!({
            "unit": u.unit, "value": u.value, "evidence_probability": u.evidence_probability,
        })).collect::<Vec<_>>(),
        "stages": result.stages.iter().map(|s| json!({
            "stage": s.stage.name(),
            "identity": s.identity,
            "status": match s.status {
                antecedent::analysis::cross_world_execution::StageStatus::Declared => "declared",
                antecedent::analysis::cross_world_execution::StageStatus::Computed => "computed",
                antecedent::analysis::cross_world_execution::StageStatus::Reused => "reused",
            },
        })).collect::<Vec<_>>(),
        "work": {
            "mechanism_fits": w.mechanism_fits,
            "abductions": w.abductions,
            "abduction_states_enumerated": w.abduction_states_enumerated,
            "abductions_reused": w.abductions_reused,
            "shared_draws": w.shared_draws,
            "world_predictions": w.world_predictions,
            "posterior_states_predicted": w.posterior_states_predicted,
            "summaries": w.summaries,
        },
        "receipt": {
            "mechanism_digest": result.receipt.mechanism_digest,
            "abduction_digest": result.receipt.abduction_digest,
            "exogenous_digest": result.receipt.exogenous_digest,
            "world_digest": result.receipt.world_digest,
            "prediction_digest": result.receipt.prediction_digest,
            "summary_digest": result.receipt.summary_digest,
            "coupling": result.receipt.coupling,
            "digest": result.receipt.digest,
        },
    })
}

fn identity_json(identity: &CrossWorldExecutionIdentity) -> Json {
    serde_json::to_value(identity).unwrap_or(Json::Null)
}

/// The capability matrix as JSON rows.
#[pyfunction]
fn cross_world_capability_matrix() -> String {
    let rows: Vec<Json> = capability_matrix()
        .iter()
        .map(|r| {
            json!({
                "id": r.id,
                "quantity": r.quantity.name(),
                "model": r.model.name(),
                "conditioning": r.conditioning.iter().map(|c| c.name()).collect::<Vec<_>>(),
                "history": r.history.iter().map(|c| c.name()).collect::<Vec<_>>(),
                "evidence": r.evidence.iter().map(|c| c.name()).collect::<Vec<_>>(),
                "population": r.population.name(),
                "coupling": r.coupling.name(),
                "status": r.status.name(),
                "route": r.route,
                "theorem": r.theorem,
                "refusal_code": r.refusal_code,
                "refusal_detail": r.refusal_detail,
                "remedy": r.remedy,
            })
        })
        .collect();
    Json::Array(rows).to_string()
}

fn pick<T: Copy>(name: &str, all: &[(&str, T)], what: &str) -> Result<T, String> {
    all.iter().find(|(n, _)| *n == name).map(|(_, v)| *v).ok_or_else(|| {
        invalid_json("cross_world_execution.invalid_request", format!("unknown {what} `{name}`"))
    })
}

/// Decide a matrix cell. Returns the licensed row's JSON or the refusal JSON.
#[pyfunction]
#[allow(clippy::too_many_arguments, reason = "one argument per matrix dimension")]
#[allow(
    clippy::too_many_lines,
    reason = "one flat decode of each declared argument, then a single decision call"
)]
fn cross_world_decide(
    quantity: &str,
    model: &str,
    conditioning: &str,
    interventions: &str,
    population: &str,
    history: &str,
    evidence: &str,
) -> Refused<String> {
    use antecedent::analysis::cross_world_execution::{
        ConditioningKind as C, EvidenceKind as E, HistoryKind as H, InterventionKind as I,
        MatrixRequest, ModelClass as M, PopulationKind as P, QuantityClass as Q, decide_capability,
    };
    macro_rules! tri {
        ($e:expr) => {
            match $e {
                Ok(v) => v,
                Err(r) => return (None, Some(r)),
            }
        };
    }
    let request = MatrixRequest {
        quantity: tri!(pick(
            quantity,
            &[
                ("counterfactual_marginal", Q::CounterfactualMarginal),
                ("counterfactual_joint", Q::CounterfactualJoint),
                ("natural_path_specific", Q::NaturalPathSpecific),
                ("temporal", Q::Temporal),
                ("transported_cross_world", Q::TransportedCrossWorld),
            ],
            "quantity"
        )),
        model: tri!(pick(
            model,
            &[
                ("finite_markovian_scm", M::FiniteMarkovianScm),
                ("additive_linear_two_slice", M::AdditiveLinearTwoSlice),
                ("additive_linear_static_transported", M::AdditiveLinearStaticTransported),
                ("semi_markovian", M::SemiMarkovian),
                ("nonparametric_admg", M::NonparametricAdmg),
            ],
            "model"
        )),
        conditioning: tri!(pick(
            conditioning,
            &[("no_evidence", C::NoEvidence), ("factual_evidence", C::FactualEvidence)],
            "conditioning"
        )),
        interventions: tri!(pick(
            interventions,
            &[("hard", I::Hard), ("soft", I::Soft)],
            "interventions"
        )),
        population: tri!(pick(
            population,
            &[("single", P::Single), ("transported", P::Transported)],
            "population"
        )),
        history: tri!(pick(
            history,
            &[
                ("no_history", H::NoHistory),
                ("fixed_population_two_slice", H::FixedPopulationTwoSlice),
                ("incompatible", H::Incompatible),
            ],
            "history"
        )),
        evidence: tri!(pick(
            evidence,
            &[
                ("declared_mechanisms", E::DeclaredMechanisms),
                ("supplied_fitted_identity", E::SuppliedFittedIdentity),
                ("data_to_fit", E::DataToFit),
            ],
            "evidence"
        )),
    };
    match decide_capability(&request) {
        Ok(row) => (
            Some(
                json!({
                    "id": row.id, "status": row.status.name(), "route": row.route,
                    "coupling": row.coupling.name(),
                })
                .to_string(),
            ),
            None,
        ),
        Err(r) => (
            None,
            Some(
                json!({
                    "code": r.code, "stage": STAGE, "detail": r.detail,
                    "message": format!("cross-world capability refused: {}", r.detail),
                    "offending": r.subject, "remedy": r.remedy, "missing": r.missing,
                })
                .to_string(),
            ),
        ),
    }
}

/// A selective executor retaining abductions across calls.
#[pyclass(name = "CrossWorldExecutor")]
pub(crate) struct PyCrossWorldExecutor {
    inner: FiniteScmExecutor,
}

#[pymethods]
impl PyCrossWorldExecutor {
    #[new]
    fn new() -> Self {
        Self { inner: FiniteScmExecutor::new() }
    }

    /// Execute a request; reuses retained abductions whose identity matches.
    #[pyo3(signature = (request_json, cancel=None))]
    fn execute(
        &mut self,
        request_json: &str,
        cancel: Option<PyCancellationToken>,
    ) -> Refused<String> {
        let request = match parse_request(request_json) {
            Ok(r) => r,
            Err(e) => return (None, Some(e)),
        };
        let ctx = execution_context(0, None, cancel);
        match self.inner.execute(&request, &ctx) {
            Ok(result) => (Some(result_json(&result).to_string()), None),
            Err(error) => (None, Some(core_refusal(&error))),
        }
    }

    fn retained_len(&self) -> usize {
        self.inner.retained_len()
    }

    /// Cumulative observed work.
    fn counters(&self) -> String {
        let c = self.inner.counters();
        json!({
            "mechanism_fits": c.mechanism_fits, "abductions": c.abductions,
            "abduction_states_enumerated": c.abduction_states_enumerated,
            "abductions_reused": c.abductions_reused, "shared_draws": c.shared_draws,
            "world_predictions": c.world_predictions,
            "posterior_states_predicted": c.posterior_states_predicted,
            "summaries": c.summaries,
        })
        .to_string()
    }

    /// Drop every retained abduction (a process boundary).
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Consume an artifact and return an executor warmed by its verified
    /// abductions together with the summary.
    #[staticmethod]
    #[pyo3(signature = (data, expected_identity_json=None, cancel=None))]
    fn from_artifact(
        py: Python<'_>,
        data: &[u8],
        expected_identity_json: Option<&str>,
        cancel: Option<PyCancellationToken>,
    ) -> Refused<(Py<PyCrossWorldExecutor>, String)> {
        let expected = match expected_identity_json {
            Some(text) => match serde_json::from_str::<CrossWorldExecutionIdentity>(text) {
                Ok(v) => Some(v),
                Err(e) => {
                    return (
                        None,
                        Some(invalid_json("cross_world_execution.invalid_identity", e.to_string())),
                    );
                }
            },
            None => None,
        };
        let ctx = execution_context(0, None, cancel);
        match CrossWorldExecutionArtifact::from_bytes(data, expected.as_ref(), &ctx) {
            Ok(artifact) => {
                let summary = summary_json(&artifact);
                let executor = PyCrossWorldExecutor { inner: artifact.into_executor() };
                match Py::new(py, executor) {
                    Ok(obj) => (Some((obj, summary)), None),
                    Err(e) => (
                        None,
                        Some(invalid_json("cross_world_execution.invalid_artifact", e.to_string())),
                    ),
                }
            }
            Err(error) => (None, Some(artifact_refusal(&error))),
        }
    }
}

fn summary_json(artifact: &CrossWorldExecutionArtifact) -> String {
    json!({
        "result": result_json(artifact.result()),
        "identity": identity_json(artifact.identity()),
        "matrix_row": artifact.meta().matrix_row,
        "inference_claim": artifact.meta().inference_claim,
        "shared_draws": artifact.meta().shared_draws.iter().map(|d| json!({
            "unit": d.unit, "identity": d.identity, "states": d.states,
            "draws_digest": d.draws_digest,
        })).collect::<Vec<_>>(),
    })
    .to_string()
}

/// Execute with a fresh executor and seal. Returns `(bytes, summary_json)`.
#[pyfunction]
#[pyo3(signature = (request_json, artifact_id, cancel=None))]
fn cross_world_seal(
    py: Python<'_>,
    request_json: &str,
    artifact_id: &str,
    cancel: Option<PyCancellationToken>,
) -> Refused<(Py<PyBytes>, String)> {
    let request = match parse_request(request_json) {
        Ok(r) => r,
        Err(e) => return (None, Some(e)),
    };
    let ctx = execution_context(0, None, cancel);
    let artifact = match CrossWorldExecutionArtifact::seal(&request, &ctx) {
        Ok(a) => a,
        Err(e) => return (None, Some(artifact_refusal(&e))),
    };
    match artifact.to_bytes(artifact_id) {
        Ok(bytes) => (Some((PyBytes::new(py, &bytes).unbind(), summary_json(&artifact))), None),
        Err(e) => (None, Some(artifact_refusal(&e))),
    }
}

/// Consume artifact bytes by replay; returns the summary JSON.
#[pyfunction]
#[pyo3(signature = (data, expected_identity_json=None, cancel=None))]
fn cross_world_consume(
    data: &[u8],
    expected_identity_json: Option<&str>,
    cancel: Option<PyCancellationToken>,
) -> Refused<String> {
    let expected = match expected_identity_json {
        Some(text) => match serde_json::from_str::<CrossWorldExecutionIdentity>(text) {
            Ok(v) => Some(v),
            Err(e) => {
                return (
                    None,
                    Some(invalid_json("cross_world_execution.invalid_identity", e.to_string())),
                );
            }
        },
        None => None,
    };
    let ctx = execution_context(0, None, cancel);
    match CrossWorldExecutionArtifact::from_bytes(data, expected.as_ref(), &ctx) {
        Ok(artifact) => (Some(summary_json(&artifact)), None),
        Err(error) => (None, Some(artifact_refusal(&error))),
    }
}

fn parse_unit_sampling(name: &str) -> Result<UnitSampling, String> {
    match name {
        "undeclared" => Ok(UnitSampling::Undeclared),
        "fixed_units" => Ok(UnitSampling::FixedUnits),
        "independent_sample" => Ok(UnitSampling::IndependentSample),
        other => Err(invalid_json(
            "cross_world_execution.invalid_request",
            format!("unknown unit sampling `{other}`"),
        )),
    }
}

fn treatment_json(treatment: &SourceTreatment) -> Json {
    match treatment {
        SourceTreatment::Premise => json!({"kind": "premise"}),
        SourceTreatment::Exact { observed_weight_defect } => {
            json!({"kind": "exact", "observed_weight_defect": observed_weight_defect})
        }
        SourceTreatment::Propagated { variance } => {
            json!({"kind": "propagated", "variance": variance})
        }
        SourceTreatment::NotApplicable(reason) => {
            json!({"kind": "not_applicable", "reason": reason})
        }
        SourceTreatment::Unavailable(reason) => json!({"kind": "unavailable", "reason": reason}),
    }
}

fn account_json(account: &WholePathAccount) -> String {
    let u = account.uncertainty();
    let inspection = account.inspect();
    json!({
        "label": u.label,
        "unit_sampling": u.unit_sampling.name(),
        "value": u.value,
        "units": u.units.iter().map(|unit| json!({
            "unit": unit.unit,
            "value": unit.value,
            "predictive_variance": unit.predictive_variance,
        })).collect::<Vec<_>>(),
        "realized_mean_predictive_variance": u.realized_mean_predictive_variance,
        "realized_mean_standard_error": inspection.realized_mean_standard_error,
        "unit_mean_sampling_variance": u.unit_mean_sampling_variance,
        "unit_mean_standard_error": inspection.unit_mean_standard_error,
        "sources": u.sources.iter().map(|s| json!({
            "source": s.source.name(),
            "treatment": treatment_json(&s.treatment),
        })).collect::<Vec<_>>(),
        "routes": u.routes,
        "identity": u.identity,
        "execution_receipt_digest": account.execution_receipt_digest(),
    })
    .to_string()
}

fn consume_for_account(
    data: &[u8],
    expected_identity_json: Option<&str>,
    cancel: Option<PyCancellationToken>,
) -> Result<(CrossWorldExecution, antecedent_core::ExecutionContext), String> {
    let expected = match expected_identity_json {
        Some(text) => match serde_json::from_str::<CrossWorldExecutionIdentity>(text) {
            Ok(v) => Some(v),
            Err(e) => {
                return Err(invalid_json("cross_world_execution.invalid_identity", e.to_string()));
            }
        },
        None => None,
    };
    let ctx = execution_context(0, None, cancel);
    CrossWorldExecution::consume(data, expected.as_ref(), &ctx)
        .map(|execution| (execution, ctx))
        .map_err(|error| artifact_refusal(&error))
}

/// The whole-path uncertainty account of a sealed execution, derived again from its bytes
/// under a declared unit sampling: an uncalibrated spread, never an interval.
#[pyfunction]
#[pyo3(signature = (data, unit_sampling, expected_identity_json=None, cancel=None))]
fn cross_world_whole_path(
    data: &[u8],
    unit_sampling: &str,
    expected_identity_json: Option<&str>,
    cancel: Option<PyCancellationToken>,
) -> Refused<String> {
    let sampling = match parse_unit_sampling(unit_sampling) {
        Ok(s) => s,
        Err(e) => return (None, Some(e)),
    };
    let (execution, ctx) = match consume_for_account(data, expected_identity_json, cancel) {
        Ok(pair) => pair,
        Err(e) => return (None, Some(e)),
    };
    match execution.whole_path_uncertainty(sampling, &ctx) {
        Ok(account) => (Some(account_json(&account)), None),
        Err(error) => (None, Some(core_refusal(&error))),
    }
}

/// Derive the account again from the bytes and require it to equal a stored one: `None` when
/// it does, the refusal JSON otherwise. A stored account is never trusted.
#[pyfunction]
#[pyo3(signature = (data, unit_sampling, account_identity, receipt_digest, expected_identity_json=None, cancel=None))]
fn cross_world_whole_path_verify(
    data: &[u8],
    unit_sampling: &str,
    account_identity: &str,
    receipt_digest: &str,
    expected_identity_json: Option<&str>,
    cancel: Option<PyCancellationToken>,
) -> Option<String> {
    let sampling = match parse_unit_sampling(unit_sampling) {
        Ok(s) => s,
        Err(e) => return Some(e),
    };
    let (execution, ctx) = match consume_for_account(data, expected_identity_json, cancel) {
        Ok(pair) => pair,
        Err(e) => return Some(e),
    };
    match execution.whole_path_uncertainty(sampling, &ctx) {
        Ok(account) => {
            if account.uncertainty().identity == account_identity
                && account.execution_receipt_digest() == receipt_digest
            {
                None
            } else {
                Some(
                    json!({
                        "code": antecedent_core::reason_code!("invalid_argument"),
                        "stage": "cross_world_uncertainty",
                        "detail": "cross_world_uncertainty.account_mismatch",
                        "message": "the stored account differs from the derivation",
                        "offending": Json::Null,
                        "remedy": "derive the account from the consumed execution; a stored account is never trusted",
                        "missing": Vec::<String>::new(),
                    })
                    .to_string(),
                )
            }
        }
        Err(error) => Some(core_refusal(&error)),
    }
}

/// The refusal of an interval or posterior reading (always `cell_not_licensed`).
#[pyfunction]
fn cross_world_closed_reading(reading: &str) -> String {
    let detail = match reading {
        "interval" => "cross_world_execution.interval_route_closed",
        "posterior" => "cross_world_execution.posterior_route_closed",
        "sampled_abduction" => "cross_world_execution.sampled_abduction_not_licensed",
        _ => {
            return invalid_json(
                "cross_world_execution.invalid_request",
                format!("unknown reading `{reading}`"),
            );
        }
    };
    json!({
        "code": antecedent_core::reason_code!("cell_not_licensed"),
        "stage": STAGE,
        "detail": detail,
        "message": format!("cross-world reading refused: {detail}"),
        "offending": reading,
        "remedy": "the whole abduction-action-prediction path has no measured coverage record; the point value from exact enumeration is the result, and the calibration design that would open this reading is recorded in docs/cross-world-execution.md",
        "missing": ["whole_path_calibration_record"],
    })
    .to_string()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCrossWorldExecutor>()?;
    module.add_function(wrap_pyfunction!(cross_world_capability_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(cross_world_decide, module)?)?;
    module.add_function(wrap_pyfunction!(cross_world_seal, module)?)?;
    module.add_function(wrap_pyfunction!(cross_world_consume, module)?)?;
    module.add_function(wrap_pyfunction!(cross_world_whole_path, module)?)?;
    module.add_function(wrap_pyfunction!(cross_world_whole_path_verify, module)?)?;
    module.add_function(wrap_pyfunction!(cross_world_closed_reading, module)?)?;
    Ok(())
}
