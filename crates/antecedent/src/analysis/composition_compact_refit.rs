//! Independently replay the original native fit behind a compact effect model.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, execution::AdmissionError};
use antecedent_estimate::cohort_linear_effect::{
    FIT_PROCEDURE_ID, FIT_PROCEDURE_VARIABLE, FIT_TRAINING_VARIABLE, MAX_FIT_FEATURES,
    MAX_FIT_ROWS, NativeCohortFit, NativeFitRows,
};
use antecedent_io_core::compact_export::{CompactExport, ExportLimits};

use super::{
    BundleError, BundleLimits, BundleStage, CompositionBundle, NodeKind, NodeSource, refused,
};

fn admission_refusal(error: AdmissionError) -> BundleError {
    let stage = if matches!(error, AdmissionError::Cancelled) {
        BundleStage::ExecutionCancelled
    } else {
        BundleStage::ResourceRefused
    };
    refused(stage, &error.to_string(), None)
}

/// Replay the original fitting procedure against an independently retained compact identity.
///
/// The initial native OLS/HC3 route records a digest of the ordered training columns and
/// modifier ids. Refit validates that digest, the explicit learner, and the entire reproduced
/// model, including its coefficient covariance, feature support and mask.
///
/// # Errors
/// Missing historical training binding, wrong learner or snapshot, a differing compact model,
/// a fitting failure, or an execution stop.
pub fn refit_compact_model(
    model: &[u8],
    expected_identity: &str,
    training: NativeFitRows,
    learner: &str,
    ctx: &ExecutionContext,
) -> Result<NativeCohortFit, BundleError> {
    let numeric_values =
        training.features.iter().fold(
            training.outcome.len().saturating_add(training.treatment.len()),
            |count, column| count.saturating_add(column.len()),
        );
    let max_values = MAX_FIT_ROWS * (MAX_FIT_FEATURES + 2);
    if training.outcome.len() > MAX_FIT_ROWS
        || training.features.len() > MAX_FIT_FEATURES
        || training.feature_ids.len() > MAX_FIT_FEATURES
        || numeric_values > max_values
    {
        return Err(refused(
            BundleStage::ResourceRefused,
            "compact refit training exceeds the bounded native fit size",
            None,
        ));
    }
    let bytes = (numeric_values as u64)
        .saturating_mul(32)
        .saturating_add((model.len() as u64).saturating_mul(4))
        .saturating_add(512 * 1024);
    let _fit = ctx.acquire_serial_numerical_admission(bytes).map_err(admission_refusal)?;
    let compact = CompactExport::consume(model, &ExportLimits::default(), expected_identity)
        .map_err(|error| {
            use antecedent_core::reason_details::compact_export as details;
            let stage = match error.detail {
                details::OVERSIZED => BundleStage::Oversized,
                details::UNSUPPORTED_VERSION => BundleStage::IncompatibleVersion,
                _ => BundleStage::SwappedEvidence,
            };
            refused(stage, &error.to_string(), None)
        })?;
    let binding = |id| {
        compact
            .body()
            .response
            .conditioning
            .iter()
            .find(|condition| condition.variable_id == id)
            .map(|condition| condition.value_id.as_str())
    };
    if binding(FIT_PROCEDURE_VARIABLE) != Some(FIT_PROCEDURE_ID)
        || binding(FIT_TRAINING_VARIABLE).is_none()
    {
        return Err(refused(
            BundleStage::CallbackUnavailable,
            "this compact export does not bind the original training snapshot and native learner",
            None,
        ));
    }
    if learner != FIT_PROCEDURE_ID {
        return Err(refused(
            BundleStage::ProviderRequestChanged,
            "the supplied learner differs from the compact model's native OLS/HC3 learner",
            None,
        ));
    }
    if binding(FIT_TRAINING_VARIABLE) != Some(training.training_identity().as_str()) {
        return Err(refused(
            BundleStage::GraphOrSnapshotMismatch,
            "the supplied training rows or their order differ from the original snapshot",
            None,
        ));
    }
    let replayed =
        NativeCohortFit::replay_controlled(training, &compact, || ctx.cancellation.is_cancelled());
    if ctx.cancellation.is_cancelled() {
        return Err(admission_refusal(AdmissionError::Cancelled));
    }
    replayed.map_err(|error| refused(BundleStage::TamperedQuantity, &error.to_string(), None))
}

/// Resolve a named embedded compact model in an independently identified bundle and refit it.
///
/// # Errors
/// A different or invalid bundle, an absent or unresolved model, or a compact-refit refusal.
pub fn refit_compact_bundle_model(
    bundle_bytes: &[u8],
    expected_identity: &str,
    model_node: &str,
    training: NativeFitRows,
    learner: &str,
    ctx: &ExecutionContext,
) -> Result<NativeCohortFit, BundleError> {
    let _bundle = ctx
        .acquire_admission(0, (bundle_bytes.len() as u64).saturating_mul(4))
        .map_err(admission_refusal)?;
    let bundle =
        CompositionBundle::from_bytes(bundle_bytes, expected_identity, &BundleLimits::default())?;
    let node = bundle.nodes().iter().find(|node| node.id == model_node).ok_or_else(|| {
        refused(
            BundleStage::CallbackUnavailable,
            "the compact model node is absent",
            Some(model_node),
        )
    })?;
    let NodeSource::Embedded { bytes } = &node.source else {
        return Err(refused(
            BundleStage::CallbackUnavailable,
            "embed the compact model before supplying its original training and learner",
            Some(model_node),
        ));
    };
    if node.kind != NodeKind::CompactFittedEffectModel {
        return Err(refused(
            BundleStage::SwappedEvidence,
            "the selected bundle node is not a compact fitted-effect model",
            Some(model_node),
        ));
    }
    refit_compact_model(bytes, &node.identity, training, learner, ctx)
}
