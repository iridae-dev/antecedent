//! Shared receipt sealing and typed boundaries for checked posterior/history adapters.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::recalc_api::{
    RunPayload, artifact_invalid, counts_wire, parse_json, plan_refusal_json, refusal_json,
};
use antecedent::analysis::recalc_receipt::{RecalcReceipt, RecalcRunError};
use antecedent::execution::recalc::{
    Boundary, RecalcCapabilities, RecalcPlan, RequestSupport, ResumeContext, RetargetSupport,
    StageIdentities,
};
use antecedent::io::recalc_receipt_artifact::{
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
            antecedent::reason::reason_code!("score_table_unavailable"),
            &stage.label(),
            antecedent::reason::reason_details::recalc::NO_LIVE_STATE,
            Some(&stage.label()),
            None,
            &error.to_string(),
            None,
        ),
        RecalcRunError::Request(detail) => {
            let detail = detail.as_str();
            let (code, stage) = match detail {
                antecedent::reason::reason_details::recalc::CANCELLED => {
                    (antecedent::reason::reason_code!("cancelled_no_claim"), "score_artifact")
                }
                antecedent::reason::reason_details::recalc::BAYESIAN_INVALID_INFERENCE => {
                    (antecedent::reason::reason_code!("invalid_argument"), "learner_folds_rng")
                }
                antecedent::reason::reason_details::recalc::BAYESIAN_FIT_REFUSED
                | antecedent::reason::reason_details::recalc::JOINT_BAYESIAN_REFUSED => {
                    (antecedent::reason::reason_code!("route_not_supported"), "score_artifact")
                }
                antecedent::reason::reason_details::recalc::BAYESIAN_INFERENCE_UNSUPPORTED
                | antecedent::reason::reason_details::recalc::BAYESIAN_BASIS_PRIOR_UNSUPPORTED => {
                    (antecedent::reason::reason_code!("route_not_supported"), "learner_folds_rng")
                }
                antecedent::reason::reason_details::recalc::BAYESIAN_EXTERNAL_COMPOSE_UNSUPPORTED
                | antecedent::reason::reason_details::recalc::BAYESIAN_PRIOR_SOURCE_UNVERIFIED
                | antecedent::reason::reason_details::recalc::BAYESIAN_PRIOR_SOURCE_UNSUPPORTED => {
                    (antecedent::reason::reason_code!("route_not_supported"), "prior.0")
                }
                antecedent::reason::reason_details::recalc::BAYESIAN_PRIOR_SOURCE_CONFLICT
                | antecedent::reason::reason_details::recalc::BAYESIAN_PRIOR_INVALID
                | antecedent::reason::reason_details::recalc::BAYESIAN_PRIOR_LIKELIHOOD_DOUBLE_USE => {
                    (antecedent::reason::reason_code!("invalid_argument"), "prior.0")
                }
                antecedent::reason::reason_details::recalc::BAYESIAN_INVALID_GRAPH
                | antecedent::reason::reason_details::recalc::INVALID_GRAPH => {
                    (antecedent::reason::reason_code!("invalid_argument"), "graph")
                }
                antecedent::reason::reason_details::recalc::BAYESIAN_INVALID_QUERY
                | antecedent::reason::reason_details::recalc::INVALID_TEMPORAL_WINDOW
                | antecedent::reason::reason_details::recalc::INVALID_EVIDENCE => {
                    (antecedent::reason::reason_code!("invalid_argument"), "query")
                }
                antecedent::reason::reason_details::recalc::BAYESIAN_INVALID_DATA => {
                    (antecedent::reason::reason_code!("invalid_argument"), "data_snapshot")
                }
                antecedent::reason::reason_details::recalc::INVALID_INITIAL_STATE_LAW => {
                    (antecedent::reason::reason_code!("invalid_argument"), "target_population")
                }
                antecedent::reason::reason_details::recalc::BAYESIAN_INVALID_SUMMARY => {
                    (antecedent::reason::reason_code!("invalid_argument"), "law")
                }
                antecedent::reason::reason_details::recalc::BAYESIAN_EFFECT_UNAVAILABLE
                | antecedent::reason::reason_details::recalc::BAYESIAN_WORKSPACE_LIMIT
                | antecedent::reason::reason_details::recalc::MEMORY_BUDGET_EXCEEDED
                | antecedent::reason::reason_details::recalc::INVALID_TEMPORAL_PROVIDER
                | antecedent::reason::reason_details::recalc::TEMPORAL_ARTIFACT_MISMATCH
                | antecedent::reason::reason_details::recalc::TEMPORAL_ARTIFACT_SEED_MISMATCH => {
                    (antecedent::reason::reason_code!("invalid_argument"), "score_artifact")
                }
                _ => (antecedent::reason::reason_code!("invalid_argument"), "score_artifact"),
            };
            refusal_json(code, stage, detail, None, None, &error.to_string(), None)
        }
        RecalcRunError::Execution(cause) => {
            use antecedent::CausalError;
            use antecedent::estimate::EstimationError;
            let (code, detail, stage) = match cause.peeled() {
                CausalError::NotIdentified { .. } => (
                    antecedent::reason::reason_code!("effect_not_identified"),
                    antecedent::reason::reason_details::recalc::FAMILY_NOT_IDENTIFIED,
                    "identification",
                ),
                CausalError::Estimate(EstimationError::Data(_) | EstimationError::Query(_)) => (
                    antecedent::reason::reason_code!("invalid_argument"),
                    antecedent::reason::reason_details::recalc::FAMILY_MODEL_DATA_INVALID,
                    "data_snapshot",
                ),
                CausalError::Estimate(
                    EstimationError::Refused { code, .. }
                    | EstimationError::RefusedWithFields { code, .. },
                ) if *code == "cancelled_no_claim" => (
                    antecedent::reason::reason_code!("cancelled_no_claim"),
                    antecedent::reason::reason_details::recalc::CANCELLED,
                    "score_artifact",
                ),
                _ => (
                    antecedent::reason::reason_code!("route_not_supported"),
                    fallback,
                    "score_artifact",
                ),
            };
            refusal_json(code, stage, detail, None, None, &error.to_string(), None)
        }
        RecalcRunError::Receipt(_) => refusal_json(
            antecedent::reason::reason_code!("invalid_argument"),
            "score_artifact",
            antecedent::reason::reason_details::recalc::FAMILY_RECEIPT_INVALID,
            None,
            None,
            &error.to_string(),
            None,
        ),
    }
}
