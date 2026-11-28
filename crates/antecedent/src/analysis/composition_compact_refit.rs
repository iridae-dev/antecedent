//! Independently replay the original native fit behind a compact effect model.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, ExecutionContext, execution::AdmissionError};
use antecedent_estimate::cohort_linear_effect::{
    FIT_PROCEDURE_ID, FIT_PROCEDURE_VARIABLE, FIT_TRAINING_VARIABLE, NativeCohortFit, NativeFitRows,
};
use antecedent_io_core::compact_export::{CompactExport, ExportLimits};

use super::{
    BundleError, BundleLimits, BundleStage, CompositionBundle, NodeKind, NodeSource, refused,
};
use crate::analysis::compiled_native_cohort_fit::{CohortModelFitRun, lower_fit, working_bytes};
use crate::analysis::execute::Study;
use crate::analysis::program_handle::{CompiledProgramHandle, gate_program};
use crate::planner::PlanView;
use crate::planner::lowered::{LoweredProgram, ProgramCell};

const PLAN_ID: &str = "compact_model_refit.v1";

/// A verified compact model and its original training retained together for controlled refit.
#[derive(Clone, Debug)]
pub struct CompiledCompactModelRefit {
    compact: CompactExport,
    training: NativeFitRows,
    working_bytes: u64,
    lowered: Arc<LoweredProgram>,
}

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
    let plan =
        Study::compile_compact_model_refit(model, expected_identity, training, learner, ctx)?;
    plan.execute(ctx).map(|run| run.fit)
}

fn compile_refit(
    model: &[u8],
    expected_identity: &str,
    training: NativeFitRows,
    learner: &str,
    bundle_node: Option<(&str, &str)>,
    ctx: &ExecutionContext,
) -> Result<CompiledCompactModelRefit, BundleError> {
    let training_bytes = working_bytes(
        training.outcome.len(),
        training.treatment.len(),
        training.features.iter().map(Vec::len),
    )
    .map_err(admission_refusal)?;
    let working_bytes = (model.len() as u64)
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(training_bytes))
        .ok_or_else(|| admission_refusal(AdmissionError::Overflow))?;
    let _intake =
        ctx.acquire_serial_numerical_admission(working_bytes).map_err(admission_refusal)?;
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
    let training_identity = training.training_identity();
    if binding(FIT_TRAINING_VARIABLE) != Some(training_identity.as_str()) {
        return Err(refused(
            BundleStage::GraphOrSnapshotMismatch,
            "the supplied training rows or their order differ from the original snapshot",
            None,
        ));
    }
    let mut question = CanonicalWriter::new("antecedent.compact_model_refit.question.v1");
    question.text(compact.identity()).text(learner).presence(bundle_node.is_some());
    if let Some((bundle, node)) = bundle_node {
        question.text(bundle).text(node);
    }
    let mut source = CanonicalWriter::new("antecedent.compact_model_refit.source.v1");
    source.text(&training_identity);
    let lowered = lower_fit(
        ProgramCell::CompactModelRefit,
        PLAN_ID,
        question.finish(),
        source.finish(),
        training.outcome.len(),
        training.features.len(),
    )
    .map_err(|error| refused(BundleStage::UnsupportedLaw, &error.to_string(), None))?;
    if ctx.cancellation.is_cancelled() {
        return Err(admission_refusal(AdmissionError::Cancelled));
    }
    Ok(CompiledCompactModelRefit { compact, training, working_bytes, lowered })
}

impl CompiledProgramHandle for CompiledCompactModelRefit {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledCompactModelRefit {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// View the compiled plan without fitting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }

    /// Refit the original learner and require the same complete compact model.
    ///
    /// # Errors
    /// A differing model, fitting failure, resource admission or cancellation.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<CohortModelFitRun, BundleError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let publication = gate_program(&scoped, self)
            .map_err(|error| refused(BundleStage::UnsupportedLaw, &error.to_string(), None))?;
        let _fit = scoped
            .acquire_serial_numerical_admission(self.working_bytes)
            .map_err(admission_refusal)?;
        let replayed =
            NativeCohortFit::replay_controlled(self.training.clone(), &self.compact, || {
                scoped.cancellation.is_cancelled()
            });
        if scoped.cancellation.is_cancelled() {
            return Err(admission_refusal(AdmissionError::Cancelled));
        }
        let fit = replayed
            .map_err(|error| refused(BundleStage::TamperedQuantity, &error.to_string(), None))?;
        publication.finish(Ok(CohortModelFitRun {
            fit,
            lowered_program_identity: self.lowered.identity(),
        }))
    }
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
    let plan = Study::compile_compact_bundle_refit(
        bundle_bytes,
        expected_identity,
        model_node,
        training,
        learner,
        ctx,
    )?;
    plan.execute(ctx).map(|run| run.fit)
}

impl Study {
    /// Verify and bind a compact model to its original ordered training and learner.
    ///
    /// # Errors
    /// A model, learner or snapshot that fails validation, or a resource or cancellation stop.
    pub fn compile_compact_model_refit(
        model: &[u8],
        expected_identity: &str,
        training: NativeFitRows,
        learner: &str,
        ctx: &ExecutionContext,
    ) -> Result<CompiledCompactModelRefit, BundleError> {
        compile_refit(model, expected_identity, training, learner, None, ctx)
    }

    /// Verify the caller-retained bundle identity and bind its named model to a refit plan.
    ///
    /// # Errors
    /// A different or invalid bundle, an absent or unresolved model, or compact intake refusal.
    pub fn compile_compact_bundle_refit(
        bundle_bytes: &[u8],
        expected_identity: &str,
        model_node: &str,
        training: NativeFitRows,
        learner: &str,
        ctx: &ExecutionContext,
    ) -> Result<CompiledCompactModelRefit, BundleError> {
        let byte_bound = (bundle_bytes.len() as u64)
            .checked_mul(4)
            .ok_or_else(|| admission_refusal(AdmissionError::Overflow))?;
        let _bundle = ctx.acquire_admission(0, byte_bound).map_err(admission_refusal)?;
        let bundle = CompositionBundle::from_bytes(
            bundle_bytes,
            expected_identity,
            &BundleLimits::default(),
        )?;
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
        compile_refit(
            bytes,
            &node.identity,
            training,
            learner,
            Some((expected_identity, model_node)),
            ctx,
        )
    }
}
