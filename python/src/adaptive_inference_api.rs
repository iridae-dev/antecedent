//! Python bridge for inference from adaptive experiments.
//!
//! Python declares the retained analysis contract of one adaptive experiment
//! (the assigned arms, the policy's realized assignment probabilities, the
//! exposure history, outcome availability, the interim analyses and the
//! stopping rule) and asks for one inferential paradigm at a time. The
//! contract checks, the paradigm gate, every number, the exported artifacts
//! and every refusal are Rust's; a refusal comes back as structured JSON for
//! the Python layer to raise as a typed exception.
//!
//! Every result is point-only. No interval, credible interval or coverage
//! statement is made: the guarantee route refuses with `cell_not_licensed`.

use antecedent::analysis::adaptive_inference::{
    AdaptiveInferenceRefusal, AdaptiveInferenceSession, AdaptiveRequest, CellStatus,
    InferenceParadigm, consume_adaptive_inference_result_artifact,
    consume_adaptive_monitoring_contract_artifact, procedure_cells, procedure_status,
};
use antecedent_io::adaptive_monitoring_contract_artifact::{
    AdaptiveMonitoringContractArtifact, DeclarationWire, contract_boundaries,
    declaration_from_wire, declaration_to_wire,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Value as Json, json};

use crate::measurement_model_api::{object, opt_flag, parse, present, real, small, text};
use crate::transport_common::execution_context;

pub(crate) type Refused<T> = (Option<T>, Option<String>);

fn refusal_value(refusal: &AdaptiveInferenceRefusal) -> Json {
    json!({
        "code": refusal.code,
        "stage": "adaptive_inference",
        "detail": refusal.detail,
        "message": refusal.message,
        "offending": Json::Null,
        "remedy": Json::Null,
    })
}

pub(crate) fn refused<T>(refusal: &AdaptiveInferenceRefusal) -> Refused<T> {
    (None, Some(refusal_value(refusal).to_string()))
}

pub(crate) fn refused_all<T>(refusals: &[AdaptiveInferenceRefusal]) -> Refused<T> {
    let mut first = refusals.first().map_or_else(|| json!({}), refusal_value);
    first["refusals"] = Json::Array(refusals.iter().map(refusal_value).collect());
    (None, Some(first.to_string()))
}

fn paradigm_of(value: &str) -> PyResult<InferenceParadigm> {
    InferenceParadigm::from_name(value)
        .ok_or_else(|| crate::value_err(format!("unknown inferential paradigm `{value}`")))
}

fn parse_request(value: &Json) -> PyResult<AdaptiveRequest> {
    let map = object(value, "adaptive inference request")?;
    let look = small(map, "look")?;
    Ok(match text(map, "kind")?.as_str() {
        "bayesian_posterior" => AdaptiveRequest::BayesianPosterior {
            look,
            prior_alpha: real(map, "prior_alpha")?,
            prior_beta: real(map, "prior_beta")?,
        },
        "posterior_stopping" => AdaptiveRequest::PosteriorStopping {
            look,
            prior_alpha: real(map, "prior_alpha")?,
            prior_beta: real(map, "prior_beta")?,
        },
        "adaptive_weighting" => AdaptiveRequest::AdaptiveWeighting {
            look,
            plus_arm: small(map, "plus_arm")?,
            minus_arm: present(map, "minus_arm").map(|_| small(map, "minus_arm")).transpose()?,
            scheme: text(map, "scheme")?,
            prior_mean: real(map, "prior_mean")?,
            prior_strength: real(map, "prior_strength")?,
            size_weighted: opt_flag(map, "size_weighted")?.unwrap_or(false),
        },
        "confidence_sequence" => AdaptiveRequest::ConfidenceSequence {
            look,
            alpha: real(map, "alpha")?,
            tuning_samples: real(map, "tuning_samples")?,
            contrast: match (present(map, "plus_arm"), present(map, "minus_arm")) {
                (Some(_), Some(_)) => Some((small(map, "plus_arm")?, small(map, "minus_arm")?)),
                (None, None) => None,
                _ => {
                    return Err(crate::value_err(
                        "a contrast names both `plus_arm` and `minus_arm`",
                    ));
                }
            },
        },
        "group_sequential" => AdaptiveRequest::GroupSequential {
            look,
            control_arm: small(map, "control_arm")?,
            treatment_arm: small(map, "treatment_arm")?,
        },
        other => return Err(crate::value_err(format!("unknown request kind `{other}`"))),
    })
}

fn rule_json(
    rule: Option<antecedent::analysis::adaptive_inference::StoppingRuleDeclaration>,
) -> Json {
    let Some(rule) = rule else { return Json::Null };
    let mut declaration = None;
    // The wire spelling of a rule is the one place its parameters are named.
    if let Some(wire) = declaration_for_rule(rule) {
        declaration = serde_json::to_value(wire).ok();
    }
    declaration.unwrap_or(Json::Null)
}

fn declaration_for_rule(
    rule: antecedent::analysis::adaptive_inference::StoppingRuleDeclaration,
) -> Option<antecedent_io::adaptive_monitoring_contract_artifact::RuleWire> {
    use antecedent::analysis::adaptive_inference::{
        AdaptiveDesignDeclaration, AssignmentFamily, MonitoringStrategy, OutcomeStructure,
        ProcedureCell, StoppingRuleKind,
    };
    let shell = AdaptiveDesignDeclaration {
        label: String::new(),
        cell: ProcedureCell {
            assignment: AssignmentFamily::FixedRandomization,
            outcome: OutcomeStructure::SingleOutcome,
            monitoring: MonitoringStrategy::FinalAnalysisOnly,
            stopping: StoppingRuleKind::NoStopping,
        },
        arm_labels: Vec::new(),
        assigned_arm: Vec::new(),
        policy_history: None,
        assignment_probabilities: None,
        exposure: None,
        availability: None,
        outcomes: Vec::new(),
        interim_analyses: None,
        stopping_rule: Some(rule),
        probability_floor: 1.0,
    };
    declaration_to_wire(&shell).stopping_rule
}

fn status_json(status: &CellStatus) -> Json {
    match status {
        CellStatus::Licensed { paradigms } => json!({
            "class": "licensed",
            "paradigms": paradigms.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
        }),
        CellStatus::ClosedIncoherent { reason } => {
            json!({"class": "closed_incoherent", "reason": reason})
        }
        CellStatus::ClosedPairing { reason } => {
            json!({"class": "closed_pairing", "reason": reason})
        }
        CellStatus::CarriedForward { owner, reason } => {
            json!({"class": "carried_forward", "owner": owner, "reason": reason})
        }
        _ => json!({"class": "unrecognized"}),
    }
}

/// A prepared adaptive-experiment analysis.
#[pyclass(name = "AdaptiveInferenceHandle", skip_from_py_object)]
pub(crate) struct AdaptiveInferenceHandle {
    pub(crate) inner: AdaptiveInferenceSession,
}

/// A consumed monitoring contract.
#[pyclass(name = "ConsumedMonitoringContractHandle", skip_from_py_object)]
pub(crate) struct ConsumedMonitoringContractHandle {
    summary: String,
}

/// A consumed adaptive inference result.
#[pyclass(name = "ConsumedAdaptiveResultHandle", skip_from_py_object)]
pub(crate) struct ConsumedAdaptiveResultHandle {
    summary: String,
}

#[pymethods]
impl AdaptiveInferenceHandle {
    /// Identity (hex) of the validated contract.
    fn contract_identity(&self) -> String {
        self.inner.contract_identity()
    }

    /// Units, arms, looks, procedure cell and its matrix status, as JSON.
    fn summary_json(&self) -> String {
        let contract = self.inner.contract();
        json!({
            "identity": self.inner.contract_identity(),
            "procedure": contract.cell().identifier(),
            "units": contract.units(),
            "arms": contract.arms(),
            "looks": contract.looks(),
            "adaptive": contract.is_adaptive(),
            "status": status_json(&self.inner.cell_status()),
            "stopping_rule": rule_json(contract.declaration().stopping_rule),
        })
        .to_string()
    }

    /// Run the paradigm a request names.
    #[pyo3(signature = (request_json, cancel=None))]
    fn analyse(
        &self,
        request_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let request = parse_request(&parse(request_json, "adaptive inference request")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.analyse(&request, &ctx) {
            Ok(outcome) => (
                Some(
                    json!({
                        "paradigm": outcome.paradigm.as_str(),
                        "procedure": outcome.procedure.identifier(),
                        "method": outcome.result.method,
                        "values": outcome.result.values,
                        "closed_guarantee": outcome.result.closed_guarantee,
                        "contract_identity": outcome.contract_identity,
                        "look": outcome.request.look(),
                        "stopping_rule": rule_json(outcome.stopping_rule),
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Fixed-sample inference of the data: refused for an adaptive design.
    fn fixed_sample(&self) -> Option<String> {
        self.inner.fixed_sample().err().map(|r| refusal_value(&r).to_string())
    }

    /// An interval or coverage statement for the whole procedure: refused.
    fn interval(&self) -> Option<String> {
        self.inner.interval().err().map(|r| refusal_value(&r).to_string())
    }

    /// The interval of one result of a measured procedure cell, bound to
    /// currently attesting whole-procedure coverage.
    #[pyo3(signature = (request_json, cancel=None))]
    fn measured_interval(
        &self,
        request_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<crate::measured_inference_api::NativeMeasuredInference>, Option<String>)>
    {
        let request = parse_request(&parse(request_json, "adaptive inference request")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.measured_interval(&request, "measured-adaptive-interval", &ctx) {
            Ok(measured) => (
                Some(crate::measured_inference_api::NativeMeasuredInference::from_io(measured)?),
                None,
            ),
            Err(refusal) => (None, Some(refusal_value(&refusal).to_string())),
        })
    }

    /// The observations behind each look, counted once, as JSON.
    fn interim_evidence(&self) -> Refused<String> {
        match self.inner.interim_evidence() {
            Ok(evidence) => {
                let contract = self.inner.contract();
                let sizes: Vec<usize> = match (0..contract.looks())
                    .map(|l| contract.analysis_units(l).map(|units| units.len()))
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(sizes) => sizes,
                    Err(error) => {
                        return refused(&AdaptiveInferenceRefusal {
                            code: error.code,
                            detail: error.detail,
                            message: error.remedy.unwrap_or("").to_owned(),
                        });
                    }
                };
                let pooling = contract
                    .refuse_independent_pooling_of_looks(&evidence)
                    .err()
                    .map(|r| json!({"code": r.code, "detail": r.detail}));
                (
                    Some(
                        json!({
                            "look_sizes": sizes,
                            "declared_slots": evidence.mass.declared_slots,
                            "distinct_observations": evidence.mass.distinct_observations,
                            "repeated_slots": evidence.mass.repeated_slots,
                            "state_digest": evidence.ledger.state_digest().to_hex(),
                            "reuse_result": evidence.reuse.to_hex(),
                            "independent_pooling_refusal": pooling,
                        })
                        .to_string(),
                    ),
                    None,
                )
            }
            Err(refusal) => refused(&refusal),
        }
    }

    /// Export the monitoring contract for a paradigm.
    fn export_contract(
        &self,
        py: Python<'_>,
        paradigm: &str,
        artifact_id: &str,
    ) -> PyResult<Refused<(Py<PyBytes>, String)>> {
        let paradigm = paradigm_of(paradigm)?;
        Ok(match self.inner.export_contract(paradigm, artifact_id) {
            Ok(bytes) => {
                // These are freshly produced bytes, not an untrusted consumer input.
                // Return the artifact's identity, which also binds the selected paradigm.
                let identity = AdaptiveMonitoringContractArtifact::decode_unchecked(&bytes)
                    .map_err(|error| crate::serialization_err(error.to_string()))?
                    .identity;
                (Some((PyBytes::new(py, &bytes).unbind(), identity)), None)
            }
            Err(refusal) => refused(&refusal),
        })
    }

    /// Export the monitoring contract with the participant-level procedure it
    /// was collected under: the priors with their historical sources, the
    /// decisions and the reading at every performed look.
    fn export_contract_with_procedure(
        &self,
        py: Python<'_>,
        paradigm: &str,
        procedure: PyRef<'_, crate::adaptive_participant_api::ParticipantProcedureHandle>,
        artifact_id: &str,
    ) -> PyResult<Refused<(Py<PyBytes>, String)>> {
        let paradigm = paradigm_of(paradigm)?;
        Ok(
            match self.inner.export_contract_with_procedure(paradigm, &procedure.inner, artifact_id)
            {
                Ok(bytes) => {
                    // These are freshly produced bytes, not an untrusted consumer input.
                    // Return the artifact's identity, which also binds the selected paradigm.
                    let identity = AdaptiveMonitoringContractArtifact::decode_unchecked(&bytes)
                        .map_err(|error| crate::serialization_err(error.to_string()))?
                        .identity;
                    (Some((PyBytes::new(py, &bytes).unbind(), identity)), None)
                }
                Err(refusal) => refused(&refusal),
            },
        )
    }

    /// Export a result bound to its monitoring contract.
    #[pyo3(signature = (request_json, artifact_id, cancel=None))]
    fn export_result(
        &self,
        py: Python<'_>,
        request_json: &str,
        artifact_id: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<(Py<PyBytes>, String)>> {
        let request = parse_request(&parse(request_json, "adaptive inference request")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.export_result(&request, artifact_id, &ctx) {
            Ok(bytes) => {
                // The result identity also binds the request and recomputed values;
                // the session's contract identity is not a substitute for it.
                let identity = antecedent_io::adaptive_inference_result_artifact::AdaptiveInferenceResultArtifact::decode_unchecked(&bytes)
                    .map_err(|error| crate::serialization_err(error.to_string()))?.identity;
                (Some((PyBytes::new(py, &bytes).unbind(), identity)), None)
            }
            Err(refusal) => refused(&refusal),
        })
    }
}

#[pymethods]
impl ConsumedMonitoringContractHandle {
    /// The replayed procedure, rule, paradigm and boundaries, as JSON.
    fn summary_json(&self) -> String {
        self.summary.clone()
    }
}

#[pymethods]
impl ConsumedAdaptiveResultHandle {
    /// The replayed procedure, rule, paradigm and recomputed values, as JSON.
    fn summary_json(&self) -> String {
        self.summary.clone()
    }
}

/// Validate a retained analysis contract and prepare the analysis.
#[pyfunction]
fn adaptive_inference_prepare(
    declaration_json: &str,
) -> PyResult<Refused<AdaptiveInferenceHandle>> {
    let value = parse(declaration_json, "adaptive design")?;
    let wire: DeclarationWire = serde_json::from_value(value).map_err(|error| {
        crate::value_err(format!("invalid adaptive design declaration: {error}"))
    })?;
    let declaration = match declaration_from_wire(&wire) {
        Ok(declaration) => declaration,
        Err(error) => return Err(crate::value_err(error.to_string())),
    };
    Ok(match AdaptiveInferenceSession::prepare(declaration) {
        Ok(inner) => (Some(AdaptiveInferenceHandle { inner }), None),
        Err(refusals) => refused_all(&refusals),
    })
}

/// Consume a monitoring contract artifact by replay.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None))]
fn adaptive_monitoring_contract_consume(
    artifact: &[u8],
    expected_identity: Option<&str>,
) -> Refused<ConsumedMonitoringContractHandle> {
    match consume_adaptive_monitoring_contract_artifact(artifact, expected_identity) {
        Ok(consumed) => {
            let boundaries = contract_boundaries(consumed.contract()).ok().flatten().map(|b| {
                json!({
                    "critical": b.critical,
                    "spend_increments": b.spend_increments,
                    "cumulative_spend": b.cumulative_spend,
                })
            });
            let summary = json!({
                "identity": consumed.identity(),
                "procedure": consumed.procedure().identifier(),
                "paradigm": consumed.paradigm().as_str(),
                "stopping_rule": rule_json(consumed.stopping_rule()),
                "contract_identity": consumed.contract().identity().to_hex(),
                "units": consumed.contract().units(),
                "arms": consumed.contract().arms(),
                "looks": consumed.contract().looks(),
                "boundaries": boundaries,
                "declaration": serde_json::to_value(declaration_to_wire(
                    consumed.contract().declaration(),
                ))
                .unwrap_or(Json::Null),
                "participant_procedure": consumed.procedure_record().map(|record| json!({
                    "procedure_identity": record.spec().identity(),
                    "priors": crate::adaptive_participant_api::priors_json(record.spec().priors()),
                    "readings": crate::adaptive_participant_api::readings_json(record.readings()),
                })),
            });
            (Some(ConsumedMonitoringContractHandle { summary: summary.to_string() }), None)
        }
        Err(refusal) => refused(&refusal),
    }
}

/// Consume an adaptive inference result artifact by replay and recomputation.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None, cancel=None))]
fn adaptive_inference_result_consume(
    artifact: &[u8],
    expected_identity: Option<&str>,
    cancel: Option<crate::PyCancellationToken>,
) -> Refused<ConsumedAdaptiveResultHandle> {
    let ctx = execution_context(0, None, cancel);
    match consume_adaptive_inference_result_artifact(artifact, expected_identity, &ctx) {
        Ok(consumed) => {
            let result = consumed.result();
            let summary = json!({
                "identity": consumed.identity(),
                "procedure": consumed.procedure().identifier(),
                "paradigm": consumed.paradigm().as_str(),
                "stopping_rule": rule_json(consumed.stopping_rule()),
                "method": result.method,
                "values": result.values,
                "closed_guarantee": result.closed_guarantee,
            });
            (Some(ConsumedAdaptiveResultHandle { summary: summary.to_string() }), None)
        }
        Err(refusal) => refused(&refusal),
    }
}

/// Every cell of the procedure matrix with its status, as JSON.
#[pyfunction]
fn adaptive_procedure_matrix() -> String {
    Json::Array(
        procedure_cells()
            .iter()
            .map(|cell| {
                json!({
                    "id": cell.identifier(),
                    "assignment": cell.assignment.as_str(),
                    "outcome": cell.outcome.as_str(),
                    "monitoring": cell.monitoring.as_str(),
                    "stopping": cell.stopping.as_str(),
                    "status": status_json(&procedure_status(cell)),
                })
            })
            .collect(),
    )
    .to_string()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<AdaptiveInferenceHandle>()?;
    module.add_class::<ConsumedMonitoringContractHandle>()?;
    module.add_class::<ConsumedAdaptiveResultHandle>()?;
    module.add_function(wrap_pyfunction!(adaptive_inference_prepare, module)?)?;
    module.add_function(wrap_pyfunction!(adaptive_monitoring_contract_consume, module)?)?;
    module.add_function(wrap_pyfunction!(adaptive_inference_result_consume, module)?)?;
    module.add_function(wrap_pyfunction!(adaptive_procedure_matrix, module)?)?;
    Ok(())
}
