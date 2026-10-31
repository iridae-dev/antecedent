//! Python bridge for transport across a discovered class refined by a randomized binary trial.
//!
//! Python declares the same static discovery contract, columns, query and selection targets as
//! ordinary discovered transport, the level of the one shared no-effect implication test, and
//! (for a confidence set) the scientific support of every potential outcome. The exact test,
//! every member's graph decision, the retained estimates, the whole-procedure set, the replay
//! and every refusal are Rust's. The source treatment must be completely randomized; that is
//! asserted by the caller and never inferred from the rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::NumericColumn;
use antecedent::analysis::discovered_transport::{
    DEFAULT_ENVELOPE_MEMBER_BOUND, DiscoveredTransportError, DiscoveredTransportReport,
    DiscoveredTransportRun, discover_and_transport,
};
use antecedent::analysis::discovered_transport_estimate::TransportEvidence;
use antecedent::analysis::discovered_transport_trial::{
    TransportOutcomeSupport, TrialCompatibilityOptions, TrialCompatibilityProof,
    TrialCompatibleTransportConfidenceSet, TrialConfidenceSetKind, TrialLogicalBoundReason,
    TrialRandomization, estimate_trial_compatible_discovered_transport,
    estimate_trial_compatible_transport_confidence_set,
    verify_trial_compatible_transport_confidence_set_against,
    verify_trial_compatible_transport_estimate,
};
use antecedent_core::{ExecutionContext, TransportQuery, VariableId};
use antecedent_discovery::selective::SelectiveDiscoveryError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Value as Json, json};

use crate::discovered_transport_api::{answer_json, estimate_json, query_of, refusal_of};
use crate::selective_discovery_api as sd;

/// `(result, transport refusal, discovery, discovery refusal)`.
type TrialReply<T> = (Option<T>, Option<String>, Option<String>, Option<String>);

/// `(body, transport refusal, discovery, discovery refusal, artifact)`.
type ArtifactReply =
    (Option<String>, Option<String>, Option<String>, Option<String>, Option<Py<PyBytes>>);

fn estimation(error: antecedent_io::IoError) -> DiscoveredTransportError {
    antecedent::analysis::discovered_transport_estimate::discovered_transport_io_failure(error)
}

fn options(alpha: f64) -> TrialCompatibilityOptions {
    TrialCompatibilityOptions { alpha, randomization: TrialRandomization::CompletelyRandomized }
}

fn proof_json(proof: &TrialCompatibilityProof) -> Result<Json, antecedent_io::IoError> {
    let mut value = serde_json::to_value(proof)
        .map_err(|error| antecedent_io::IoError::Convert(error.to_string()))?;
    if let Some(map) = value.as_object_mut() {
        map.insert("retained_members".to_owned(), json!(proof.retained_members()));
        map.insert("identity".to_owned(), json!(sd::hex(&proof.identity())));
    }
    Ok(value)
}

fn kind_label(kind: TrialConfidenceSetKind) -> &'static str {
    match kind {
        TrialConfidenceSetKind::IdentifiedSamplingInterval => "identified_sampling_interval",
        TrialConfidenceSetKind::LogicalSupportBound => "logical_support_bound",
    }
}

fn reason_label(reason: TrialLogicalBoundReason) -> &'static str {
    match reason {
        TrialLogicalBoundReason::TrialRejectsEveryMember => "trial_rejects_every_member",
        TrialLogicalBoundReason::ClassNotIdentified => "class_not_identified",
        TrialLogicalBoundReason::StructuralRange => "structural_range",
    }
}

/// Typed refusal retained by the post-discovery estimation stage.
type TrialWorkResult<T> = Result<T, Box<DiscoveredTransportError>>;

/// The set as JSON together with its exported receipt.
fn set_reply(set: &TrialCompatibleTransportConfidenceSet) -> TrialWorkResult<(String, Vec<u8>)> {
    let receipt = set.export().map_err(estimation)?;
    let identity = set.identity().map_err(estimation)?;
    let (lower, upper) = set.bounds();
    let support = set.outcome_support();
    let compatibility = proof_json(set.compatibility()).map_err(estimation)?;
    let body = json!({
        "method": set.method(),
        "kind": kind_label(set.kind()),
        "lower": lower,
        "upper": upper,
        "logical_reason": set.logical_reason().map(reason_label),
        "outcome_support": { "lower": support.lower, "upper": support.upper },
        "calibration": set.calibration(),
        "execution_seed": set.execution_seed(),
        "input_identity": sd::hex(set.input_identity()),
        "identity": sd::hex(&identity),
        "compatibility": compatibility,
        "estimate": set.retained_estimate().map(|e| estimate_json(e.estimate())),
    })
    .to_string();
    Ok((body, receipt))
}

/// Discover on the source columns, enumerate the class and run `work` over the report, the query
/// and the source and target evidence.
#[allow(clippy::too_many_arguments, reason = "one request: contract, two tables, query, bounds")]
fn run_trial<T, F>(
    py: Python<'_>,
    contract_json: &str,
    names: &[String],
    source_columns: &[NumericColumn<'_>],
    target_columns: &[NumericColumn<'_>],
    query_json: &str,
    selection_targets: Vec<u32>,
    member_bound: Option<usize>,
    bootstrap_replicates: u32,
    discovery_columns: Option<&[NumericColumn<'_>]>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    work: F,
) -> PyResult<TrialReply<T>>
where
    T: Send,
    F: FnOnce(
            &DiscoveredTransportReport,
            &TransportQuery,
            &TransportEvidence<'_>,
            &ExecutionContext,
        ) -> TrialWorkResult<T>
        + Send,
{
    let parsed = sd::parse_json(contract_json)?;
    let map = sd::contract_map(&parsed, "static")?;
    let seed = sd::get_u64(map, "seed")?;
    let contract = sd::static_contract(map, names.len())?;
    let query = query_of(query_json)?;
    let ids: Vec<VariableId> = selection_targets.into_iter().map(VariableId::from_raw).collect();
    let bound = member_bound.unwrap_or(DEFAULT_ENVELOPE_MEMBER_BOUND);
    let source_batch = crate::numeric_columns_to_batch(names, source_columns)?;
    let target_batch = crate::numeric_columns_to_batch(names, target_columns)?;
    let discovery_batch = discovery_columns
        .map(|columns| crate::numeric_columns_to_batch(names, columns))
        .transpose()?;
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let ran: Result<(DiscoveredTransportRun, Option<TrialWorkResult<T>>), SelectiveDiscoveryError> =
        crate::detach_catch(py, move || {
            let (source, _): (antecedent_data::TabularData, _) =
                crate::discovery_api::tabular_from_batch(&source_batch)?;
            let (target, _): (antecedent_data::TabularData, _) =
                crate::discovery_api::tabular_from_batch(&target_batch)?;
            let discovery_data = match &discovery_batch {
                Some(batch) => {
                    let (data, _): (antecedent_data::TabularData, _) =
                        crate::discovery_api::tabular_from_batch(batch)?;
                    Some(data)
                }
                None => None,
            };
            let basis = discovery_data.as_ref().unwrap_or(&source);
            let run = match discover_and_transport(&contract, basis, &ids, &query, bound, &ctx) {
                Ok(run) => run,
                Err(error) => return Ok(Err(error)),
            };
            let result = run.transport.as_ref().ok().map(|report| {
                let mut evidence = TransportEvidence::new(&source, &target);
                evidence.bootstrap_replicates = bootstrap_replicates;
                work(report, &query, &evidence, &ctx)
            });
            Ok(Ok((run, result)))
        })?;
    let (run, result) = match ran {
        Err(error) => {
            return match sd::error_json(&error) {
                Some(refusal) => Ok((None, None, None, Some(refusal))),
                None => Err(crate::CausalDiscoveryError::new_err(error.to_string())),
            };
        }
        Ok(pair) => pair,
    };
    let discovery =
        sd::discovery_value(&sd::report_json(&run.discovery), &sd::outcome_json(&run.outcome))?
            .to_string();
    match (&run.transport, result) {
        (Err(error), _) => Ok((None, Some(refusal_of(error)?), Some(discovery), None)),
        (Ok(_), Some(Ok(value))) => Ok((Some(value), None, Some(discovery), None)),
        (Ok(_), Some(Err(error))) => {
            Ok((None, Some(refusal_of(error.as_ref())?), Some(discovery), None))
        }
        (Ok(_), None) => Err(crate::value_err("the transport request produced no result")),
    }
}

fn with_artifact(py: Python<'_>, reply: TrialReply<(String, Vec<u8>)>) -> ArtifactReply {
    let (result, refusal, discovery, discovery_refusal) = reply;
    match result {
        Some((body, bytes)) => (
            Some(body),
            refusal,
            discovery,
            discovery_refusal,
            Some(PyBytes::new(py, &bytes).unbind()),
        ),
        None => (None, refusal, discovery, discovery_refusal, None),
    }
}

/// Refine the discovered class with the randomized binary source trial, then estimate every
/// retained member from the source and target columns.
///
/// The body carries the estimate and the compatibility proof (the test, its p-value and every
/// original member's decision); `artifact` is the exported proof with every retained grid.
#[pyfunction]
#[pyo3(signature = (contract_json, names, source_columns, target_columns, query_json, selection_targets, alpha, member_bound=None, bootstrap_replicates=0, *, discovery_columns=None, threads=None, cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "one request: contract, two tables, query, test level"
)]
fn discovered_transport_trial_estimate_run(
    py: Python<'_>,
    contract_json: &str,
    names: Vec<String>,
    source_columns: Vec<NumericColumn<'_>>,
    target_columns: Vec<NumericColumn<'_>>,
    query_json: &str,
    selection_targets: Vec<u32>,
    alpha: f64,
    member_bound: Option<usize>,
    bootstrap_replicates: u32,
    discovery_columns: Option<Vec<NumericColumn<'_>>>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ArtifactReply> {
    let options = options(alpha);
    let reply = run_trial(
        py,
        contract_json,
        &names,
        &source_columns,
        &target_columns,
        query_json,
        selection_targets,
        member_bound,
        bootstrap_replicates,
        discovery_columns.as_deref(),
        threads,
        cancel,
        move |report, query, evidence, ctx| {
            let estimate = estimate_trial_compatible_discovered_transport(
                report, query, evidence, options, ctx,
            )?;
            let bytes = estimate.export().map_err(estimation)?;
            let compatibility = proof_json(estimate.compatibility()).map_err(estimation)?;
            let body = json!({
                "estimate": estimate_json(estimate.estimate()),
                "compatibility": compatibility,
            })
            .to_string();
            Ok((body, bytes))
        },
    )?;
    Ok(with_artifact(py, reply))
}

/// Replay an exported trial-compatible estimate: the exact test, every graph decision and every
/// retained member grid. Returns the recomputed answer and the replayed proof as JSON.
#[pyfunction]
#[pyo3(signature = (artifact, *, seed=0))]
fn verify_trial_compatible_transport_estimate_artifact(
    artifact: &[u8],
    seed: u64,
) -> PyResult<String> {
    let ctx = crate::py_execution_context_cancel(seed, crate::resolve_user_threads(None), None);
    let (proof, verified) = verify_trial_compatible_transport_estimate(
        artifact,
        antecedent_expr::ExactEvaluationLimits::default(),
        &ctx,
    )
    .map_err(|e| crate::value_err(e.to_string()))?;
    let compatibility =
        proof_json(&proof).map_err(|error| crate::serialization_err(error.to_string()))?;
    Ok(json!({
        "answer": answer_json(&verified.answer),
        "report_identity": sd::hex(&verified.report_identity),
        "members": verified.members,
        "compatibility": compatibility,
    })
    .to_string())
}

/// Compute the bounded-outcome whole-procedure confidence set of the binary treatment mean
/// contrast, with an explicit logical-support fallback.
///
/// `support_lower` and `support_upper` are the caller's scientific bounds on every potential
/// outcome in both populations; observed rows are checked against them but never learn them.
#[pyfunction]
#[pyo3(signature = (contract_json, names, source_columns, target_columns, query_json, selection_targets, alpha, support_lower, support_upper, member_bound=None, bootstrap_replicates=200, *, discovery_columns=None, threads=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one request: contract, two tables, query, support")]
fn discovered_transport_trial_confidence_set_run(
    py: Python<'_>,
    contract_json: &str,
    names: Vec<String>,
    source_columns: Vec<NumericColumn<'_>>,
    target_columns: Vec<NumericColumn<'_>>,
    query_json: &str,
    selection_targets: Vec<u32>,
    alpha: f64,
    support_lower: f64,
    support_upper: f64,
    member_bound: Option<usize>,
    bootstrap_replicates: u32,
    discovery_columns: Option<Vec<NumericColumn<'_>>>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ArtifactReply> {
    let options = options(alpha);
    let support = TransportOutcomeSupport { lower: support_lower, upper: support_upper };
    let reply = run_trial(
        py,
        contract_json,
        &names,
        &source_columns,
        &target_columns,
        query_json,
        selection_targets,
        member_bound,
        bootstrap_replicates,
        discovery_columns.as_deref(),
        threads,
        cancel,
        move |report, query, evidence, ctx| {
            let set = estimate_trial_compatible_transport_confidence_set(
                report, query, evidence, options, support, ctx,
            )?;
            set_reply(&set)
        },
    )?;
    Ok(with_artifact(py, reply))
}

/// Replay a confidence-set receipt against the original columns, query, support, test level,
/// bootstrap size and execution seed. A receipt that does not replay is refused.
#[pyfunction]
#[pyo3(signature = (receipt, contract_json, names, source_columns, target_columns, query_json, selection_targets, alpha, support_lower, support_upper, member_bound=None, bootstrap_replicates=200, *, discovery_columns=None, threads=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "replay needs the complete original request")]
fn verify_trial_confidence_set_against_inputs(
    py: Python<'_>,
    receipt: Vec<u8>,
    contract_json: &str,
    names: Vec<String>,
    source_columns: Vec<NumericColumn<'_>>,
    target_columns: Vec<NumericColumn<'_>>,
    query_json: &str,
    selection_targets: Vec<u32>,
    alpha: f64,
    support_lower: f64,
    support_upper: f64,
    member_bound: Option<usize>,
    bootstrap_replicates: u32,
    discovery_columns: Option<Vec<NumericColumn<'_>>>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ArtifactReply> {
    let options = options(alpha);
    let support = TransportOutcomeSupport { lower: support_lower, upper: support_upper };
    let reply = run_trial(
        py,
        contract_json,
        &names,
        &source_columns,
        &target_columns,
        query_json,
        selection_targets,
        member_bound,
        bootstrap_replicates,
        discovery_columns.as_deref(),
        threads,
        cancel,
        move |report, query, evidence, ctx| {
            let set = verify_trial_compatible_transport_confidence_set_against(
                &receipt, report, query, evidence, options, support, ctx,
            )
            .map_err(estimation)?;
            set_reply(&set)
        },
    )?;
    Ok(with_artifact(py, reply))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(discovered_transport_trial_estimate_run, module)?)?;
    module.add_function(wrap_pyfunction!(
        verify_trial_compatible_transport_estimate_artifact,
        module
    )?)?;
    module
        .add_function(wrap_pyfunction!(discovered_transport_trial_confidence_set_run, module)?)?;
    module.add_function(wrap_pyfunction!(verify_trial_confidence_set_against_inputs, module)?)?;
    Ok(())
}
