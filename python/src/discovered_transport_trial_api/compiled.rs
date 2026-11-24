//! Transport across a discovered class through its compiled program.
//!
//! Python declares the same static contract, columns, query, selection targets and stage
//! settings as the direct discovered-transport bridges. The request lowers to the
//! `DiscoveredClassTransport` program, composed over the compiled `DiscoverStatic` search of the
//! declared table; the search, the envelope, every stage kernel and every refusal are Rust's.
//! The bodies are the JSON the direct bridges return, so the Python layer builds the same result
//! objects. A refusal of the transport request or of the stage comes back as the same structured
//! refusal; a refusal of the search itself is raised as the compiler's error.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use antecedent::analysis::compiled_discovered_transport::{
    DiscoveredClassEvidence, DiscoveredClassStage, DiscoveredClassStageResult,
    DiscoveredClassTransportError, DiscoveredClassTransportRequest, DiscoveredClassTransportRun,
};
use antecedent::analysis::discovered_transport::{
    DEFAULT_ENVELOPE_MEMBER_BOUND, DiscoveredTransportError,
};
use antecedent::analysis::discovered_transport_trial::TransportOutcomeSupport;
use antecedent::prelude::VariableId;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::json;

use super::{estimation, options, proof_json, set_reply};
use crate::NumericColumn;
use crate::discovered_transport_api::{envelope_json, estimate_json, query_of, refusal_of};
use crate::selective_discovery_api as sd;
use crate::{py_err, value_err};

/// `(body, transport refusal, discovery, program identity, artifact)` of a compiled request.
type CompiledReply = (Option<String>, Option<String>, String, String, Option<Py<PyBytes>>);

/// A stage body and its exported artifact, or the stage's typed refusal.
type StageBody = Result<(String, Option<Vec<u8>>), DiscoveredTransportError>;

fn program_error(error: DiscoveredClassTransportError) -> PyErr {
    match error {
        DiscoveredClassTransportError::Program(error)
        | DiscoveredClassTransportError::Search(error) => py_err(error),
        other => value_err(other.to_string()),
    }
}

fn stage_body(run: &DiscoveredClassTransportRun) -> PyResult<StageBody> {
    let report = match &run.transport {
        Ok(report) => report,
        Err(error) => return Ok(Err(error.clone())),
    };
    let Some(stage) = &run.stage else {
        return Ok(Ok((envelope_json(report).to_string(), None)));
    };
    let result = match stage {
        Ok(result) => result,
        Err(error) => return Ok(Err(error.clone())),
    };
    Ok(match result {
        DiscoveredClassStageResult::Estimate(estimate) => estimate
            .export()
            .map_err(estimation)
            .map(|bytes| (estimate_json(estimate).to_string(), Some(bytes))),
        DiscoveredClassStageResult::TrialCompatibleEstimate(estimate) => {
            match (estimate.export(), proof_json(estimate.compatibility())) {
                (Ok(bytes), Ok(compatibility)) => Ok((
                    json!({
                        "estimate": estimate_json(estimate.estimate()),
                        "compatibility": compatibility,
                    })
                    .to_string(),
                    Some(bytes),
                )),
                (Err(error), _) | (_, Err(error)) => Err(estimation(error)),
            }
        }
        DiscoveredClassStageResult::TrialConfidenceSet(set) => {
            set_reply(set).map(|(body, receipt)| (body, Some(receipt))).map_err(|error| *error)
        }
        _ => return Err(value_err("the compiled transport stage returned no known result")),
    })
}

/// Run a transport request over the class a static selective search returns, through the
/// compiled `DiscoveredClassTransport` program.
///
/// `stage` is `envelope`, `estimate`, `trial_compatible_estimate`, `trial_confidence_set` or
/// `trial_confidence_set_replay`; every stage but the envelope reads `source_columns` and
/// `target_columns`, the trial stages `alpha`, the confidence-set stages the support bounds and
/// the replay its `receipt`. `columns` are the searched table's columns.
#[pyfunction]
#[pyo3(signature = (stage, contract_json, names, columns, query_json, selection_targets, member_bound=None, *, source_columns=None, target_columns=None, bootstrap_replicates=0, alpha=None, support_lower=None, support_upper=None, receipt=None, threads=None, cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "one request: contract, tables, query, stage settings"
)]
fn discovered_class_transport_compiled(
    py: Python<'_>,
    stage: &str,
    contract_json: &str,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    query_json: &str,
    selection_targets: Vec<u32>,
    member_bound: Option<usize>,
    source_columns: Option<Vec<NumericColumn<'_>>>,
    target_columns: Option<Vec<NumericColumn<'_>>>,
    bootstrap_replicates: u32,
    alpha: Option<f64>,
    support_lower: Option<f64>,
    support_upper: Option<f64>,
    receipt: Option<&[u8]>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<CompiledReply> {
    let parsed = sd::parse_json(contract_json)?;
    let map = sd::contract_map(&parsed, "static")?;
    let seed = sd::get_u64(map, "seed")?;
    let contract = sd::static_contract(map, names.len())?;
    let query = query_of(query_json)?;
    let ids: Vec<VariableId> = selection_targets.into_iter().map(VariableId::from_raw).collect();
    let bound = member_bound.unwrap_or(DEFAULT_ENVELOPE_MEMBER_BOUND);
    let batch = crate::numeric_columns_to_batch(&names, &columns)?;
    let evidence_batches = match (&source_columns, &target_columns) {
        (Some(source), Some(target)) => Some((
            crate::numeric_columns_to_batch(&names, source)?,
            crate::numeric_columns_to_batch(&names, target)?,
        )),
        _ => None,
    };
    let needs_evidence = stage != "envelope";
    if needs_evidence && evidence_batches.is_none() {
        return Err(value_err(format!("the {stage} stage reads source and target columns")));
    }
    let trial = || alpha.map(options).ok_or_else(|| value_err("the trial stages need alpha"));
    let support = || match (support_lower, support_upper) {
        (Some(lower), Some(upper)) => Ok(TransportOutcomeSupport { lower, upper }),
        _ => Err(value_err("the confidence-set stages need support_lower and support_upper")),
    };
    // The stage without its tables, which are read off the GIL.
    enum Declared {
        Envelope,
        Estimate,
        Trial(antecedent::analysis::discovered_transport_trial::TrialCompatibilityOptions),
        Set(
            antecedent::analysis::discovered_transport_trial::TrialCompatibilityOptions,
            TransportOutcomeSupport,
        ),
        Replay(
            Vec<u8>,
            antecedent::analysis::discovered_transport_trial::TrialCompatibilityOptions,
            TransportOutcomeSupport,
        ),
    }
    let declared = match stage {
        "envelope" => Declared::Envelope,
        "estimate" => Declared::Estimate,
        "trial_compatible_estimate" => Declared::Trial(trial()?),
        "trial_confidence_set" => Declared::Set(trial()?, support()?),
        "trial_confidence_set_replay" => Declared::Replay(
            receipt
                .map(<[u8]>::to_vec)
                .ok_or_else(|| value_err("the replay stage needs the receipt"))?,
            trial()?,
            support()?,
        ),
        other => return Err(value_err(format!("unknown discovered-transport stage {other:?}"))),
    };
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let (run, program) = crate::detach_catch(py, move || {
        let (data, _) = crate::discovery_api::tabular_from_batch(&batch)?;
        let evidence = match &evidence_batches {
            Some((source, target)) => {
                let (source, _) = crate::discovery_api::tabular_from_batch(source)?;
                let (target, _) = crate::discovery_api::tabular_from_batch(target)?;
                let mut evidence = DiscoveredClassEvidence::new(source, target);
                evidence.bootstrap_replicates = bootstrap_replicates;
                Some(evidence)
            }
            None => None,
        };
        let evidence =
            || evidence.clone().ok_or_else(|| value_err("the stage reads source and target"));
        let stage = match declared {
            Declared::Envelope => DiscoveredClassStage::Envelope,
            Declared::Estimate => DiscoveredClassStage::Estimate { evidence: evidence()? },
            Declared::Trial(options) => {
                DiscoveredClassStage::TrialCompatibleEstimate { evidence: evidence()?, options }
            }
            Declared::Set(options, support) => {
                DiscoveredClassStage::TrialConfidenceSet { evidence: evidence()?, options, support }
            }
            Declared::Replay(receipt, options, support) => {
                DiscoveredClassStage::TrialConfidenceSetReplay {
                    receipt,
                    evidence: evidence()?,
                    options,
                    support,
                }
            }
        };
        let request = DiscoveredClassTransportRequest::new(contract, data, ids, query, stage)
            .with_member_bound(bound);
        let compiled = Study::compile_discovered_class_transport(request, &ctx).map_err(py_err)?;
        let program = compiled.lowered().identity().to_hex();
        let run = compiled.execute(&ctx).map_err(program_error)?;
        Ok((run, program))
    })?;
    let discovery =
        sd::discovery_value(&sd::report_json(&run.discovery), &sd::outcome_json(&run.outcome))?
            .to_string();
    match stage_body(&run)? {
        Ok((body, artifact)) => Ok((
            Some(body),
            None,
            discovery,
            program,
            artifact.map(|bytes| PyBytes::new(py, &bytes).unbind()),
        )),
        Err(error) => Ok((None, Some(refusal_of(&error)?), discovery, program, None)),
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(discovered_class_transport_compiled, module)?)?;
    Ok(())
}
