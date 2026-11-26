//! Python bridge for decision reliability.
//!
//! Python builds the draws carrier (utilities with stream identities and declared
//! dependence) and a request as JSON; the validation of every declaration, the Monte
//! Carlo error, the ranking diagnostics, the selection-aware evaluation, the artifact and
//! every refusal are Rust's. A refusal comes back as structured JSON for the Python layer
//! to raise as a typed exception. Nothing here estimates dependence: it is declared or it
//! refuses, and no ranking probability or interval is published.

use std::fmt::Write as _;

use antecedent::analysis::decision_reliability::{
    CalibrationCoordinate, CohortUtilityDraws, DecisionDraws, DecisionReliabilityExpectation,
    DecisionReliabilityInspection, DecisionReliabilityRefusal, DerivedUncertainty, IntervalRequest,
    JointLawDeclaration, JointLawStanding, OffPolicyReturnDraws, PointOnlyOutput, ProducerCarrier,
    ProducerOutput, PublishedRoute, ReliabilityReport, ReliabilityRequest, RolloutNoise,
    RolloutUtilityDraws, SelectionUncertainty, SourceParadigm, SourceUncertainty, StreamId,
    StreamPurpose, WithinStreamDependence, consume_decision_reliability_artifact,
    consume_derived_uncertainty, decision_reliability_coverage_design, publish_derived_route,
    reach_decision_carrier,
};
use antecedent::analysis::decision_reliability_programs::{
    CompiledDecisionReliabilityAssessment, compile_decision_reliability_assessment,
    compile_decision_reliability_derivation, compile_producer_carrier_assessment,
    compile_producer_carrier_derivation,
};
use antecedent::analysis::input_decision_programs::{
    TerminalError, compile_decision_draws_ranking, compile_producer_carrier_ranking,
};
use antecedent::design::decision_reliability::{RankingRequest, ReliabilityError, support_route};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Value as Json, json};

use crate::PyCancellationToken;
use crate::transport_common::execution_context;

type Refused<T> = (Option<T>, Option<String>);
/// Exported bytes, report JSON and inspection JSON.
type AssessParts = (Py<PyBytes>, String, String);

/// Execute numerical work off the GIL and preserve the operational meaning of
/// the lower owner's typed refusal fields across the historical JSON boundary.
fn run_worker<T: Send>(
    py: Python<'_>,
    ctx: antecedent::prelude::ExecutionContext,
    work: impl FnOnce(&antecedent::prelude::ExecutionContext) -> Refused<T> + Send,
) -> PyResult<Refused<T>> {
    crate::detach_catch(py, move || {
        let outcome = work(&ctx);
        if ctx.cancellation.is_cancelled() {
            return Err(crate::py_err(antecedent::CausalError::Cancelled {
                stage: "decision_reliability",
            }));
        }
        if let Some(refusal) = &outcome.1 {
            let fields: Json = serde_json::from_str(refusal)
                .map_err(|error| crate::serialization_err(error.to_string()))?;
            if fields["code"] == "cancelled_no_claim" {
                return Err(crate::py_err(antecedent::CausalError::Cancelled {
                    stage: "decision_reliability",
                }));
            }
            if fields["detail"]
                == antecedent::reason::reason_details::decision_reliability::MEMORY_BUDGET_EXCEEDED
            {
                return Err(crate::CausalResourceError::new_err(refusal.clone()));
            }
        }
        Ok(outcome)
    })
}

fn refusal_json(refusal: &DecisionReliabilityRefusal) -> String {
    crate::bridge::compact_refusal_json(
        refusal.code,
        "decision_reliability",
        &refusal.detail,
        &refusal.message,
        refusal.remedy,
    )
}

pub(crate) fn error_json(error: &ReliabilityError) -> String {
    let refusal = error.to_refusal();
    json!({
        "code": refusal.code,
        "stage": "decision_reliability",
        "detail": refusal.detail,
        "message": error.to_string(),
        "offending": refusal.offending,
        "remedy": refusal.remedy,
    })
    .to_string()
}

/// Display of `AdmissionError::Cancelled`. This crate cannot name that type; the facade
/// compares the variant, and this is the message that variant prints.
const ADMISSION_CANCELLED: &str = "admission stopped because the study was cancelled";

type CompilerSlot = std::sync::Mutex<Option<antecedent::CausalError>>;

fn store_compiler(slot: &CompilerSlot, error: antecedent::CausalError) {
    *slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
}

fn compiler_error(slot: CompilerSlot) -> Option<antecedent::CausalError> {
    slot.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The facade's `DecisionReliabilityRefusal::from_error` message, which is private.
fn reliability_refusal_json(error: &ReliabilityError) -> String {
    let refusal = error.to_refusal();
    let mut message = refusal.detail.clone();
    if let Some(offending) = &refusal.offending {
        let _ = write!(message, ": {offending}");
    }
    if let Some(expected) = &refusal.expected {
        let _ = write!(message, "; expected {expected}");
    }
    if let Some(remedy) = refusal.remedy {
        let _ = write!(message, "; {remedy}");
    }
    crate::bridge::compact_refusal_json(
        refusal.code,
        "decision_reliability",
        &refusal.detail,
        &message,
        refusal.remedy,
    )
}

/// The facade's `DecisionReliabilityRefusal::from_io` message, which is private.
fn io_refusal_json(error: &antecedent::io::IoError) -> String {
    use antecedent::io::IoError;
    use antecedent::reason::reason_details::decision_reliability as detail;
    if let IoError::Admission(admission) = error {
        let cancelled = admission.to_string() == ADMISSION_CANCELLED;
        let code = if cancelled {
            antecedent::reason::reason_code!("cancelled_no_claim")
        } else {
            antecedent::reason::reason_code!("invalid_argument")
        };
        let key = if cancelled { detail::CANCELLED } else { detail::MEMORY_BUDGET_EXCEEDED };
        return crate::bridge::compact_refusal_json(
            code,
            "decision_reliability",
            key,
            &admission.to_string(),
            None,
        );
    }
    if let IoError::Refused { code, message } = error
        && let Some((detail::ARTIFACT_KIND_MISMATCH, text)) = message.split_once(": ")
    {
        return crate::bridge::compact_refusal_json(
            code,
            "decision_reliability",
            detail::ARTIFACT_KIND_MISMATCH,
            text,
            None,
        );
    }
    crate::bridge::compact_refusal_json(
        error.reason_code().unwrap_or(antecedent::reason::reason_code!("invalid_argument")),
        "decision_reliability",
        detail::ARTIFACT,
        &error.to_string(),
        None,
    )
}

/// Same text as [`PreparedDecisionReliability::inspect`].
fn inspection_of(report: &ReliabilityReport) -> DecisionReliabilityInspection {
    let source = match &report.source_uncertainty {
        SourceUncertainty::KnownLaw => {
            "none: the law was declared known; draws carry numerical error only".to_owned()
        }
        SourceUncertainty::Spread { paradigm, candidates } => format!(
            "spread of utility under the supplied {} law over {} candidate(s)",
            paradigm.as_str(),
            candidates.len()
        ),
    };
    let mc = &report.monte_carlo_error.decision;
    let selection = match &report.selection_uncertainty {
        SelectionUncertainty::NotSelected => {
            "none: the decision was fixed independently of the draws".to_owned()
        }
        SelectionUncertainty::Selected(assessment) => format!(
            "in_sample maximum over {} considered candidate(s); status {:?}",
            assessment.n_considered, assessment.status
        ),
    };
    DecisionReliabilityInspection {
        decided_id: report.decided_id.clone(),
        standing: "uncalibrated_point_diagnostic",
        source_uncertainty: source,
        monte_carlo_error: format!(
            "value {} with standard error {} (effective sample size {:.1}, {:?})",
            mc.value, mc.standard_error, mc.effective_sample_size, mc.method
        ),
        selection_uncertainty: selection,
        coordinate: report.coordinate.describe(),
    }
}

fn seal_assessment(
    program: &CompiledDecisionReliabilityAssessment,
    artifact_id: &str,
    ctx: &antecedent::prelude::ExecutionContext,
    refused: &CompilerSlot,
) -> Refused<(Vec<u8>, String, String)> {
    let artifact = match program.seal(ctx) {
        Ok(artifact) => artifact,
        Err(TerminalError::Kernel(error)) => return (None, Some(io_refusal_json(&error))),
        Err(TerminalError::Refused(error)) => {
            store_compiler(refused, error);
            return (None, None);
        }
    };
    let bytes = match artifact.to_bytes(artifact_id, ctx) {
        Ok(value) => value,
        Err(error) => return (None, Some(io_refusal_json(&error))),
    };
    let report = match serde_json::to_string(&artifact.report) {
        Ok(value) => value,
        Err(error) => return (None, Some(invalid_json(error.to_string()))),
    };
    (Some((bytes, report, inspection_json(&inspection_of(&artifact.report)))), None)
}

fn invalid_json(message: String) -> String {
    crate::bridge::compact_refusal_json(
        antecedent::reason::reason_code!("invalid_argument"),
        "decision_reliability",
        antecedent::reason::reason_details::decision_reliability::INVALID_INPUT,
        &message,
        None,
    )
}

fn parse<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> Result<T, String> {
    serde_json::from_str(text).map_err(|error| invalid_json(format!("{what}: {error}")))
}

fn inspection_json(inspection: &DecisionReliabilityInspection) -> String {
    json!({
        "decided_id": inspection.decided_id,
        "standing": inspection.standing,
        "source_uncertainty": inspection.source_uncertainty,
        "monte_carlo_error": inspection.monte_carlo_error,
        "selection_uncertainty": inspection.selection_uncertainty,
        "coordinate": inspection.coordinate,
    })
    .to_string()
}

fn parse_expectation(text: &str) -> Result<DecisionReliabilityExpectation, String> {
    let value: Json = parse(text, "expectation")?;
    let opt_text = |key: &str| value.get(key).and_then(Json::as_str).map(str::to_owned);
    let stream = |key: &str| -> Result<Option<StreamId>, String> {
        match value.get(key) {
            None | Some(Json::Null) => Ok(None),
            Some(inner) => serde_json::from_value(inner.clone())
                .map(Some)
                .map_err(|error| invalid_json(format!("{key}: {error}"))),
        }
    };
    Ok(DecisionReliabilityExpectation {
        draws_digest: opt_text("draws_digest"),
        candidate_set_digest: opt_text("candidate_set_digest"),
        selection_stream: stream("selection_stream")?,
        validation_stream: stream("validation_stream")?,
        utility_unit: opt_text("utility_unit"),
        decided_id: opt_text("decided_id"),
    })
}

/// Assess the reliability of a decision and export `decision_reliability_v1`.
#[pyfunction]
#[pyo3(signature = (draws_json, validation_json, request_json, artifact_id, seed=0, cancel=None))]
fn decision_reliability_assess(
    py: Python<'_>,
    draws_json: &str,
    validation_json: Option<&str>,
    request_json: &str,
    artifact_id: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<AssessParts>> {
    let ctx = execution_context(seed, None, cancel);
    let refused = std::sync::Mutex::new(None);
    let outcome = run_worker(py, ctx, |ctx| {
        let draws: DecisionDraws = match parse(draws_json, "draws") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let validation: Option<DecisionDraws> =
            match validation_json.map(|t| parse(t, "validation")) {
                Some(Ok(value)) => Some(value),
                Some(Err(refusal)) => return (None, Some(refusal)),
                None => None,
            };
        let request: ReliabilityRequest = match parse(request_json, "request") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let program = match compile_decision_reliability_assessment(draws, validation, request, ctx)
        {
            Ok(program) => program,
            Err(error) => {
                store_compiler(&refused, error);
                return (None, None);
            }
        };
        if let Err(error) = program.execute(ctx) {
            return match error {
                TerminalError::Kernel(error) => (None, Some(reliability_refusal_json(&error))),
                TerminalError::Refused(error) => {
                    store_compiler(&refused, error);
                    (None, None)
                }
            };
        }
        seal_assessment(&program, artifact_id, ctx, &refused)
    })?;
    match compiler_error(refused) {
        Some(error) => Err(crate::py_err(error)),
        None => Ok((
            outcome.0.map(|(bytes, report, inspection)| {
                (PyBytes::new(py, &bytes).unbind(), report, inspection)
            }),
            outcome.1,
        )),
    }
}

/// Independently consume `decision_reliability_v1` bytes by recomputation.
#[pyfunction]
#[pyo3(signature = (artifact, expected_json=None, seed=0, cancel=None))]
fn decision_reliability_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_json: Option<&str>,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<(String, String)>> {
    let ctx = execution_context(seed, None, cancel);
    run_worker(py, ctx, move |ctx| {
        let expected = match expected_json.map(parse_expectation) {
            Some(Ok(value)) => value,
            Some(Err(refusal)) => return (None, Some(refusal)),
            None => DecisionReliabilityExpectation::default(),
        };
        match consume_decision_reliability_artifact(artifact, &expected, ctx) {
            Ok(prepared) => match serde_json::to_string(prepared.report()) {
                Ok(report) => (Some((report, inspection_json(&prepared.inspect()))), None),
                Err(error) => (None, Some(invalid_json(error.to_string()))),
            },
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        }
    })
}

/// Ranking statements over a finite candidate set.
#[pyfunction]
#[pyo3(signature = (draws_json, request_json, seed=0, cancel=None))]
fn decision_reliability_rank(
    py: Python<'_>,
    draws_json: &str,
    request_json: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<String>> {
    let ctx = execution_context(seed, None, cancel);
    // A compiler refusal is raised as the Python error it is, after the worker returns.
    let refused = std::sync::Mutex::new(None);
    let outcome = run_worker(py, ctx, |ctx| {
        let draws: DecisionDraws = match parse(draws_json, "draws") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let request: RankingRequest = match parse(request_json, "ranking request") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let executed = compile_decision_draws_ranking(draws, request, ctx)
            .map_err(TerminalError::Refused)
            .and_then(|program| program.execute(ctx));
        match executed {
            Ok(report) => match serde_json::to_string(&report) {
                Ok(text) => (Some(text), None),
                Err(error) => (None, Some(invalid_json(error.to_string()))),
            },
            Err(TerminalError::Kernel(error)) => (None, Some(error_json(&error))),
            Err(TerminalError::Refused(error)) => {
                *refused.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
                (None, None)
            }
        }
    })?;
    match refused.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner) {
        Some(error) => Err(crate::py_err(error)),
        None => Ok(outcome),
    }
}

/// Digests of a carrier (draws and candidate set), after validating its declarations.
#[doc(hidden)]
#[pyfunction]
fn decision_reliability_digests(draws_json: &str) -> Refused<(String, String)> {
    let draws: DecisionDraws = match parse(draws_json, "draws") {
        Ok(value) => value,
        Err(refusal) => return (None, Some(refusal)),
    };
    match draws.validate() {
        Ok(()) => (Some((draws.digest_hex(), draws.candidate_set_digest())), None),
        Err(error) => (None, Some(error_json(&error))),
    }
}

/// Publish a ranking probability or interval route at a coordinate. The route is always
/// published; the result is its calibration slot.
#[pyfunction]
fn decision_reliability_publish(route: &str, coordinate_json: &str) -> Refused<String> {
    let route: PublishedRoute = match serde_json::from_value(Json::String(route.to_owned())) {
        Ok(value) => value,
        Err(error) => return (None, Some(invalid_json(format!("route: {error}")))),
    };
    let coordinate: CalibrationCoordinate = match parse(coordinate_json, "coordinate") {
        Ok(value) => value,
        Err(refusal) => return (None, Some(refusal)),
    };
    match support_route(route, &coordinate) {
        Ok(slot) => match serde_json::to_string(&slot) {
            Ok(text) => (Some(text), None),
            Err(error) => (None, Some(invalid_json(format!("slot: {error}")))),
        },
        Err(error) => (None, Some(error_json(&error))),
    }
}

/// Derive the three uncertainty components from draws (derived, `not_measured`).
#[pyfunction]
#[pyo3(signature = (draws_json, validation_json, request_json, interval_json, seed=0, cancel=None))]
fn decision_reliability_derive(
    py: Python<'_>,
    draws_json: &str,
    validation_json: Option<&str>,
    request_json: &str,
    interval_json: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<String>> {
    let ctx = execution_context(seed, None, cancel);
    let refused = std::sync::Mutex::new(None);
    let outcome = run_worker(py, ctx, |ctx| {
        let draws: DecisionDraws = match parse(draws_json, "draws") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let validation: Option<DecisionDraws> =
            match validation_json.map(|t| parse(t, "validation")) {
                Some(Ok(value)) => Some(value),
                Some(Err(refusal)) => return (None, Some(refusal)),
                None => None,
            };
        let request: ReliabilityRequest = match parse(request_json, "request") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let interval: IntervalRequest = match parse(interval_json, "interval request") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let program = match compile_decision_reliability_derivation(
            draws, validation, request, interval, ctx,
        ) {
            Ok(program) => program,
            Err(error) => {
                store_compiler(&refused, error);
                return (None, None);
            }
        };
        match program.execute(ctx) {
            Ok(derived) => derived_json(&derived),
            Err(TerminalError::Kernel(error)) => (None, Some(reliability_refusal_json(&error))),
            Err(TerminalError::Refused(error)) => {
                store_compiler(&refused, error);
                (None, None)
            }
        }
    })?;
    match compiler_error(refused) {
        Some(error) => Err(crate::py_err(error)),
        None => Ok(outcome),
    }
}

fn derived_json(derived: &DerivedUncertainty) -> Refused<String> {
    match serde_json::to_string(derived) {
        Ok(text) => (Some(text), None),
        Err(error) => (None, Some(invalid_json(error.to_string()))),
    }
}

/// Consume `decision_reliability_v1` bytes and re-derive the three components from the
/// carried draws.
#[pyfunction]
#[pyo3(signature = (artifact, expected_json, interval_json, seed=0, cancel=None))]
fn decision_reliability_consume_derived(
    py: Python<'_>,
    artifact: &[u8],
    expected_json: Option<&str>,
    interval_json: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<String>> {
    let ctx = execution_context(seed, None, cancel);
    run_worker(py, ctx, move |ctx| {
        let expected = match expected_json.map(parse_expectation) {
            Some(Ok(value)) => value,
            Some(Err(refusal)) => return (None, Some(refusal)),
            None => DecisionReliabilityExpectation::default(),
        };
        let interval: IntervalRequest = match parse(interval_json, "interval request") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        match consume_derived_uncertainty(artifact, &expected, &interval, ctx) {
            Ok(derived) => derived_json(&derived),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        }
    })
}

/// Re-derive a stored derivation from an artifact's carried draws and compare.
#[pyfunction]
#[pyo3(signature = (artifact, derived_json, expected_json, seed=0, cancel=None))]
fn decision_reliability_verify_derived(
    py: Python<'_>,
    artifact: &[u8],
    derived_json: &str,
    expected_json: Option<&str>,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Option<String>> {
    let ctx = execution_context(seed, None, cancel);
    let outcome = run_worker(py, ctx, move |ctx| {
        let expected = match expected_json.map(parse_expectation) {
            Some(Ok(value)) => value,
            Some(Err(refusal)) => return (None, Some(refusal)),
            None => DecisionReliabilityExpectation::default(),
        };
        let derived: DerivedUncertainty = match parse(derived_json, "derived uncertainty") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let prepared = match consume_decision_reliability_artifact(artifact, &expected, ctx) {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal_json(&refusal))),
        };
        (
            Some(()),
            prepared.verify_derived(&derived, ctx).err().map(|refusal| refusal_json(&refusal)),
        )
    })?;
    Ok(outcome.1)
}

/// Publish a route from a derivation; the result is its calibration slot.
#[pyfunction]
fn decision_reliability_publish_derived(route: &str, derived_json: &str) -> Refused<String> {
    let route: PublishedRoute = match serde_json::from_value(Json::String(route.to_owned())) {
        Ok(value) => value,
        Err(error) => return (None, Some(invalid_json(format!("route: {error}")))),
    };
    let derived: DerivedUncertainty = match parse(derived_json, "derived uncertainty") {
        Ok(value) => value,
        Err(refusal) => return (None, Some(refusal)),
    };
    match publish_derived_route(&derived, route) {
        Ok(slot) => match serde_json::to_string(&slot) {
            Ok(text) => (Some(text), None),
            Err(error) => (None, Some(invalid_json(format!("slot: {error}")))),
        },
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

/// The coverage-measurement design as JSON (nothing is measured).
#[pyfunction]
fn decision_reliability_coverage_design_json() -> Refused<String> {
    match serde_json::to_string(&decision_reliability_coverage_design()) {
        Ok(text) => (Some(text), None),
        Err(error) => (None, Some(invalid_json(error.to_string()))),
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CohortWire {
    source_identity: String,
    utility_unit: String,
    run_seed: u64,
    purpose: StreamPurpose,
    replicate: u64,
    label: String,
    source: SourceParadigm,
    joint_law: JointLawDeclaration,
    within_stream: WithinStreamDependence,
    actions: Vec<(String, Vec<f64>)>,
    #[serde(default)]
    reported_means: Vec<(String, f64)>,
    #[serde(default = "default_tolerance")]
    max_standard_errors: f64,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RolloutWire {
    model_identity: String,
    stream_scheme: String,
    seed: u64,
    utility_unit: String,
    purpose: StreamPurpose,
    replicate: u64,
    noise: RolloutNoise,
    policies: Vec<(String, Vec<f64>)>,
    #[serde(default)]
    reported_means: Vec<(String, f64)>,
    #[serde(default = "default_tolerance")]
    max_standard_errors: f64,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct OffPolicyWire {
    model_identity: String,
    log_identity: String,
    run_seed: u64,
    utility_unit: String,
    purpose: StreamPurpose,
    replicate: u64,
    policies: Vec<(String, Vec<f64>)>,
    #[serde(default)]
    reported_means: Vec<(String, f64)>,
    #[serde(default = "default_tolerance")]
    max_standard_errors: f64,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProducerWire {
    Cohort(CohortWire),
    Rollout(RolloutWire),
    OffPolicy(OffPolicyWire),
    PointOnly(PointOnlyOutput),
}

const fn default_tolerance() -> f64 {
    4.0
}

fn producer_output(text: &str) -> Result<ProducerOutput, String> {
    let wire: ProducerWire = parse(text, "producer output")?;
    Ok(match wire {
        ProducerWire::Cohort(w) => ProducerOutput::Cohort(CohortUtilityDraws {
            source_identity: w.source_identity,
            utility_unit: w.utility_unit,
            run_seed: w.run_seed,
            purpose: w.purpose,
            replicate: w.replicate,
            label: w.label,
            source: w.source,
            joint_law: w.joint_law,
            within_stream: w.within_stream,
            actions: w.actions,
            reported_means: w.reported_means,
            max_standard_errors: w.max_standard_errors,
        }),
        ProducerWire::Rollout(w) => ProducerOutput::Rollout(RolloutUtilityDraws {
            model_identity: w.model_identity,
            stream_scheme: w.stream_scheme,
            seed: w.seed,
            utility_unit: w.utility_unit,
            purpose: w.purpose,
            replicate: w.replicate,
            noise: w.noise,
            policies: w.policies,
            reported_means: w.reported_means,
            max_standard_errors: w.max_standard_errors,
        }),
        ProducerWire::OffPolicy(w) => ProducerOutput::OffPolicy(OffPolicyReturnDraws {
            model_identity: w.model_identity,
            log_identity: w.log_identity,
            run_seed: w.run_seed,
            utility_unit: w.utility_unit,
            purpose: w.purpose,
            replicate: w.replicate,
            policies: w.policies,
            reported_means: w.reported_means,
            max_standard_errors: w.max_standard_errors,
        }),
        ProducerWire::PointOnly(point) => ProducerOutput::PointOnly(point),
    })
}

fn carrier_from(text: &str) -> Result<ProducerCarrier, String> {
    let output = producer_output(text)?;
    reach_decision_carrier(output).map_err(|refusal| refusal_json(&refusal))
}

fn standing_label(standing: JointLawStanding) -> Json {
    match standing {
        JointLawStanding::OneSharedLaw => {
            json!({"standing": "one_shared_law", "reason": Json::Null})
        }
        JointLawStanding::MarginalOnly { reason } => {
            json!({"standing": "marginal_only", "reason": reason})
        }
    }
}

/// Take a producer output to the reliability carrier: its draws, producer and standing.
#[pyfunction]
fn decision_reliability_reach_carrier(output_json: &str) -> Refused<String> {
    let carrier = match carrier_from(output_json) {
        Ok(value) => value,
        Err(refusal) => return (None, Some(refusal)),
    };
    let body = json!({
        "producer": carrier.producer().as_str(),
        "joint_law": standing_label(carrier.joint_law()),
        "draws": carrier.draws(),
        "draws_digest": carrier.draws().digest_hex(),
        "candidate_set_digest": carrier.draws().candidate_set_digest(),
    });
    (Some(body.to_string()), None)
}

/// Assess a producer output (carrier standing honoured) and export the artifact.
#[pyfunction]
#[pyo3(signature = (output_json, validation_json, request_json, artifact_id, seed=0, cancel=None))]
fn decision_reliability_assess_producer(
    py: Python<'_>,
    output_json: &str,
    validation_json: Option<&str>,
    request_json: &str,
    artifact_id: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<AssessParts>> {
    let ctx = execution_context(seed, None, cancel);
    let refused = std::sync::Mutex::new(None);
    let outcome = run_worker(py, ctx, |ctx| {
        let carrier = match carrier_from(output_json) {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let validation: Option<DecisionDraws> =
            match validation_json.map(|t| parse(t, "validation")) {
                Some(Ok(value)) => Some(value),
                Some(Err(refusal)) => return (None, Some(refusal)),
                None => None,
            };
        let request: ReliabilityRequest = match parse(request_json, "request") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let draws = carrier.draws().clone();
        let checked = match compile_producer_carrier_assessment(
            carrier,
            validation.clone(),
            request.clone(),
            ctx,
        ) {
            Ok(program) => program,
            Err(error) => {
                store_compiler(&refused, error);
                return (None, None);
            }
        };
        let sealed = match compile_decision_reliability_assessment(draws, validation, request, ctx)
        {
            Ok(program) => program,
            Err(error) => {
                store_compiler(&refused, error);
                return (None, None);
            }
        };
        if let Err(error) = checked.execute(ctx) {
            return match error {
                TerminalError::Kernel(error) => (None, Some(reliability_refusal_json(&error))),
                TerminalError::Refused(error) => {
                    store_compiler(&refused, error);
                    (None, None)
                }
            };
        }
        seal_assessment(&sealed, artifact_id, ctx, &refused)
    })?;
    match compiler_error(refused) {
        Some(error) => Err(crate::py_err(error)),
        None => Ok((
            outcome.0.map(|(bytes, report, inspection)| {
                (PyBytes::new(py, &bytes).unbind(), report, inspection)
            }),
            outcome.1,
        )),
    }
}

/// Rank a producer carrier. The program carries joint-law standing, so a marginal-only
/// carrier is refused and a shared law ranks that carrier's draws.
#[pyfunction]
#[pyo3(signature = (output_json, request_json, seed=0, cancel=None))]
fn decision_reliability_rank_producer(
    py: Python<'_>,
    output_json: &str,
    request_json: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<String>> {
    let ctx = execution_context(seed, None, cancel);
    let refused = std::sync::Mutex::new(None);
    let outcome = run_worker(py, ctx, |ctx| {
        let carrier = match carrier_from(output_json) {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let request: RankingRequest = match parse(request_json, "ranking request") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let program = match compile_producer_carrier_ranking(carrier, request, ctx) {
            Ok(program) => program,
            Err(error) => {
                store_compiler(&refused, error);
                return (None, None);
            }
        };
        match program.execute(ctx) {
            Ok(report) => match serde_json::to_string(&report) {
                Ok(text) => (Some(text), None),
                Err(error) => (None, Some(invalid_json(error.to_string()))),
            },
            Err(TerminalError::Kernel(error)) => (None, Some(reliability_refusal_json(&error))),
            Err(TerminalError::Refused(error)) => {
                store_compiler(&refused, error);
                (None, None)
            }
        }
    })?;
    match compiler_error(refused) {
        Some(error) => Err(crate::py_err(error)),
        None => Ok(outcome),
    }
}

/// Derive the three components for a producer output (carrier standing honoured).
#[pyfunction]
#[pyo3(signature = (output_json, validation_json, request_json, interval_json, seed=0, cancel=None))]
fn decision_reliability_derive_producer(
    py: Python<'_>,
    output_json: &str,
    validation_json: Option<&str>,
    request_json: &str,
    interval_json: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<String>> {
    let ctx = execution_context(seed, None, cancel);
    let refused = std::sync::Mutex::new(None);
    let outcome = run_worker(py, ctx, |ctx| {
        let carrier = match carrier_from(output_json) {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let validation: Option<DecisionDraws> =
            match validation_json.map(|t| parse(t, "validation")) {
                Some(Ok(value)) => Some(value),
                Some(Err(refusal)) => return (None, Some(refusal)),
                None => None,
            };
        let request: ReliabilityRequest = match parse(request_json, "request") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let interval: IntervalRequest = match parse(interval_json, "interval request") {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        };
        let program = match compile_producer_carrier_derivation(
            carrier, validation, request, interval, ctx,
        ) {
            Ok(program) => program,
            Err(error) => {
                store_compiler(&refused, error);
                return (None, None);
            }
        };
        match program.execute(ctx) {
            Ok(derived) => derived_json(&derived),
            Err(TerminalError::Kernel(error)) => (None, Some(reliability_refusal_json(&error))),
            Err(TerminalError::Refused(error)) => {
                store_compiler(&refused, error);
                (None, None)
            }
        }
    })?;
    match compiler_error(refused) {
        Some(error) => Err(crate::py_err(error)),
        None => Ok(outcome),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decision_reliability_assess, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_consume, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_rank, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_digests, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_publish, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_derive, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_consume_derived, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_verify_derived, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_publish_derived, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_coverage_design_json, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_reach_carrier, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_assess_producer, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_rank_producer, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_derive_producer, module)?)?;
    Ok(())
}
