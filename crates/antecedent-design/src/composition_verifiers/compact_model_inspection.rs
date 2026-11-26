//! Compact model prediction facts and explicit refit dependencies.

use antecedent_io_core::cohort_decision::RetainedEffectModel;
use antecedent_io_core::compact_export::{CompactExport, ExportLimits};

use super::{FACT_COORDINATES, Inspected, coordinate_id, failure, publish};
use crate::composition_bundle::{BundleStage, NodeFailure};

pub(super) fn inspect(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let limits = ExportLimits::default();
    let export = CompactExport::inspect_for_bundle(bytes, &limits).map_err(|error| {
        use antecedent_core::reason_details::compact_export as details;
        let stage = match error.detail {
            details::OVERSIZED => BundleStage::Oversized,
            details::UNSUPPORTED_VERSION => BundleStage::IncompatibleVersion,
            _ => BundleStage::SwappedEvidence,
        };
        failure(stage, error.to_string())
    })?;
    let body = export.body();
    let mut inspected = Inspected { identity: export.identity().into(), ..Inspected::default() };
    publish(&mut inspected.facts, "compact_data_digest", &body.data_blake3);
    publish(&mut inspected.facts, FACT_COORDINATES, coordinate_id(&body.response));
    publish(&mut inspected.facts, "compact_premises_digest", &body.premises_blake3);
    publish(&mut inspected.facts, "prediction", "retained_basis");
    publish(&mut inspected.facts, "refit", "training_data_and_learner_required");
    if let Ok(model) = RetainedEffectModel::from_compact_export(&export) {
        publish(&mut inspected.facts, "model_identity", model.identity());
    }
    Ok(inspected)
}

/// Check the refit capability of an independently verified compact model.
///
/// # Errors
/// An invalid artifact, or a missing declaration of original training data and learner.
#[doc(hidden)]
pub fn require_compact_model_refit_inputs(bytes: &[u8]) -> Result<(), NodeFailure> {
    inspect(bytes)?;
    Err(failure(
        BundleStage::CallbackUnavailable,
        "compact refit requires the original training data snapshot and fitted learner declaration",
    ))
}
