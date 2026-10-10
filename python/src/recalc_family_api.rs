//! Shared receipt sealing and typed boundaries for checked posterior/history adapters.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::recalc_api::{
    RunPayload, artifact_invalid, counts_wire, parse_json, plan_refusal_json, refusal_json,
};
use antecedent::analysis::recalc_receipt::{RecalcReceipt, RecalcRunError};
use antecedent_core::recalc::{
    Boundary, RecalcCapabilities, RecalcPlan, RequestSupport, ResumeContext, RetargetSupport,
    StageIdentities,
};
use antecedent_io::recalc_receipt_artifact::{
    DeclaredStageWire, RecalcReceiptArtifact, ResumeWire, identities_from_wire, plan_to_wire,
};
use pyo3::prelude::*;
use std::collections::BTreeMap;

pub(crate) fn caps(boundary: Boundary) -> RecalcCapabilities {
    RecalcCapabilities {
        boundary,
        request: RequestSupport::OnGrid,
        retarget: RetargetSupport::NotDeclared,
    }
}
pub(crate) fn resume(previous: &str, context: &str) -> PyResult<(StageIdentities, ResumeContext)> {
    let previous: Vec<DeclaredStageWire> = parse_json(previous, "previous")?;
    let ids = identities_from_wire(&previous).map_err(|error| artifact_invalid(&error))?;
    let wire: ResumeWire = parse_json(context, "resume")?;
    Ok((ids, ResumeContext { supplied_data: wire.supplied_data, ..ResumeContext::default() }))
}
pub(crate) fn payload(
    previous: &StageIdentities,
    current: &StageIdentities,
    capabilities: &RecalcCapabilities,
    plan: &RecalcPlan,
    receipt: &RecalcReceipt,
    mut body: serde_json::Value,
) -> PyResult<RunPayload> {
    let counts = receipt
        .entries()
        .iter()
        .map(|entry| (entry.stage, counts_wire(&entry.counts)))
        .collect::<BTreeMap<_, _>>();
    let artifact = RecalcReceiptArtifact::seal(previous, current, capabilities, &counts)
        .map_err(crate::py_msg)?;
    if artifact.plan() != plan {
        return Err(crate::py_msg("family receipt differs from actual executed plan"));
    }
    body["plan"] = serde_json::to_value(plan_to_wire(plan)).map_err(crate::py_msg)?;
    body["receipt"] = serde_json::to_value(artifact.meta()).map_err(crate::py_msg)?;
    let bytes = artifact
        .to_bytes(&format!("recalc-family-{}", &artifact.receipt_identity()[..16]))
        .map_err(crate::py_msg)?;
    Ok((Some(body.to_string()), Some(bytes), None))
}
pub(crate) fn run_error(error: &RecalcRunError, fallback: &'static str) -> String {
    match error {
        RecalcRunError::Refused(plan) => plan_refusal_json(plan),
        RecalcRunError::NoLiveState(stage) => refusal_json(
            antecedent_core::reason_code!("score_table_unavailable"),
            &stage.label(),
            "recalc.no_live_state",
            Some(&stage.label()),
            None,
            &error.to_string(),
            None,
        ),
        RecalcRunError::Request(detail) => {
            let (code, stage) = match *detail {
                "recalc.cancelled" => {
                    (antecedent_core::reason_code!("cancelled_no_claim"), "score_artifact")
                }
                "recalc.bayesian_invalid_inference" => {
                    (antecedent_core::reason_code!("invalid_argument"), "learner_folds_rng")
                }
                "recalc.bayesian_fit_refused" => {
                    (antecedent_core::reason_code!("route_not_supported"), "score_artifact")
                }
                "recalc.bayesian_inference_unsupported"
                | "recalc.bayesian_basis_prior_unsupported" => {
                    (antecedent_core::reason_code!("route_not_supported"), "learner_folds_rng")
                }
                "recalc.bayesian_external_compose_unsupported"
                | "recalc.bayesian_prior_source_unverified"
                | "recalc.bayesian_prior_source_unsupported" => {
                    (antecedent_core::reason_code!("route_not_supported"), "prior.0")
                }
                "recalc.bayesian_prior_source_conflict"
                | "recalc.bayesian_prior_invalid"
                | "recalc.bayesian_prior_likelihood_double_use" => {
                    (antecedent_core::reason_code!("invalid_argument"), "prior.0")
                }
                "recalc.bayesian_invalid_graph" | "recalc.invalid_graph" => {
                    (antecedent_core::reason_code!("invalid_argument"), "graph")
                }
                "recalc.bayesian_invalid_query"
                | "recalc.invalid_temporal_window"
                | "recalc.invalid_evidence" => {
                    (antecedent_core::reason_code!("invalid_argument"), "query")
                }
                "recalc.bayesian_invalid_data" => {
                    (antecedent_core::reason_code!("invalid_argument"), "data_snapshot")
                }
                "recalc.invalid_initial_state_law" => {
                    (antecedent_core::reason_code!("invalid_argument"), "target_population")
                }
                "recalc.bayesian_invalid_summary" => {
                    (antecedent_core::reason_code!("invalid_argument"), "law")
                }
                "recalc.bayesian_effect_unavailable"
                | "recalc.bayesian_workspace_limit"
                | "recalc.memory_budget_exceeded"
                | "recalc.invalid_temporal_provider"
                | "recalc.temporal_artifact_mismatch"
                | "recalc.temporal_artifact_seed_mismatch" => {
                    (antecedent_core::reason_code!("invalid_argument"), "score_artifact")
                }
                _ => (antecedent_core::reason_code!("invalid_argument"), "score_artifact"),
            };
            refusal_json(code, stage, detail, None, None, &error.to_string(), None)
        }
        RecalcRunError::Execution(cause) => {
            use antecedent::CausalError;
            use antecedent_estimate::EstimationError;
            let (code, detail, stage) = match cause.peeled() {
                CausalError::NotIdentified { .. } => (
                    antecedent_core::reason_code!("effect_not_identified"),
                    "recalc.family_not_identified",
                    "identification",
                ),
                CausalError::Estimate(EstimationError::Data(_) | EstimationError::Query(_)) => (
                    antecedent_core::reason_code!("invalid_argument"),
                    "recalc.family_model_data_invalid",
                    "data_snapshot",
                ),
                CausalError::Estimate(
                    EstimationError::Refused { code, .. }
                    | EstimationError::RefusedWithFields { code, .. },
                ) if *code == "cancelled_no_claim" => (
                    antecedent_core::reason_code!("cancelled_no_claim"),
                    "recalc.cancelled",
                    "score_artifact",
                ),
                _ => (
                    antecedent_core::reason_code!("route_not_supported"),
                    fallback,
                    "score_artifact",
                ),
            };
            refusal_json(code, stage, detail, None, None, &error.to_string(), None)
        }
        RecalcRunError::Receipt(_) => refusal_json(
            antecedent_core::reason_code!("invalid_argument"),
            "score_artifact",
            "recalc.family_receipt_invalid",
            None,
            None,
            &error.to_string(),
            None,
        ),
    }
}
