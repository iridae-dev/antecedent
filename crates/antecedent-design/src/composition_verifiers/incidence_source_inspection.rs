//! Source-bound competing-incidence intervals at the bundle boundary.

use antecedent_io_core::{AnalysisResultWire, CausalQueryWire, replay_analysis_result};

use super::{
    FACT_REQUEST, FACT_SNAPSHOT, Inspected, MAX_SNIFF_BYTES, failure, io_failure, publish,
    push_value,
};
use crate::composition_bundle::{BundleStage, NodeFailure, NodeKind};

pub(super) fn is_delayed_entry_incidence(body: &AnalysisResultWire) -> bool {
    matches!(&body.query, CausalQueryWire::Survival(query)
        if query.target_cause.is_some() && query.delayed_entry.is_some())
        && body.survival.as_ref().is_some_and(|curve| curve.difference_at_tau_interval.is_some())
}

pub(super) fn inspect(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::DelayedEntryIncidenceResult;
    if bytes.len() > MAX_SNIFF_BYTES {
        return Err(failure(BundleStage::Oversized, "incidence artifact exceeds the size bound"));
    }
    let consumed = replay_analysis_result(bytes).map_err(|error| io_failure(kind, &error))?;
    if !is_delayed_entry_incidence(&consumed.body) {
        return Err(failure(
            BundleStage::SwappedEvidence,
            "a delayed-entry incidence interval is required",
        ));
    }
    if !consumed.acceptance.accepts_as_claim() {
        let missing =
            consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.checked_survival_incidence_operation"
            });
        return Err(failure(
            if missing { BundleStage::CallbackUnavailable } else { BundleStage::TamperedQuantity },
            format!("delayed-entry incidence replay: {:?}", consumed.acceptance.unresolved),
        ));
    }
    let (Some(curve), Some(contract)) = (&consumed.body.survival, &consumed.contract) else {
        return Err(failure(
            BundleStage::SwappedEvidence,
            "incidence result has no source contract",
        ));
    };
    let (Some(cause), Some([lower, upper]), Some(claim)) =
        (curve.target_cause, curve.difference_at_tau_interval, contract.claim.as_ref())
    else {
        return Err(failure(BundleStage::TamperedQuantity, "incidence claim is incomplete"));
    };
    let digest = |bytes| antecedent_core::SemanticDigest::from_bytes(bytes).to_hex();
    let mut inspected = Inspected { identity: digest(claim.claim_id), ..Inspected::default() };
    publish(&mut inspected.facts, FACT_SNAPSHOT, digest(contract.identities.data_snapshot));
    publish(&mut inspected.facts, FACT_REQUEST, digest(contract.identities.target));
    publish(&mut inspected.facts, "program_identity", digest(contract.identities.program));
    publish(&mut inspected.facts, "target_cause", cause.to_string());
    publish(&mut inspected.facts, "calibration", claim.calibration.status.clone());
    publish(&mut inspected.facts, "inference_scope", curve.uncertainty.clone());
    publish(
        &mut inspected.facts,
        "support",
        curve.graphless_support_status.as_deref().unwrap_or("off_axis"),
    );
    push_value(&mut inspected.values, "horizon".into(), curve.tau);
    if let (Some(active), Some(reference)) = (curve.treated.last(), curve.control.last()) {
        push_value(&mut inspected.values, "incidence_difference".into(), active - reference);
    }
    push_value(&mut inspected.values, "lower_95".into(), lower);
    push_value(&mut inspected.values, "upper_95".into(), upper);
    Ok(inspected)
}
