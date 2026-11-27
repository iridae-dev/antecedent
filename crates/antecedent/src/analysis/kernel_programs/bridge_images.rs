//! Exact set and joint-outcome images through the causal compiler.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::{ExecutionContext, SemanticDigest};
pub use antecedent_io::bridge_image_artifact::{
    BridgeImageError, BridgeImageReplayBinding, BridgeImageRequest, BridgeImageResult,
    StoredBridgeImage,
};
pub use antecedent_io::bridge_joint_outcomes::{
    JointOutcomeBridgeError, JointOutcomeMap, JointOutcomeTerm,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

pub use antecedent_io::bridge_posterior_interval::{
    PosteriorAffineBridge, PosteriorBridgeInterval,
};
pub use antecedent_io::bridge_trial_interval::{
    TrialBridgeConfidenceInterval, TrialBridgeExperiment, TrialBridgeReleaseDesign, TrialMeanStudy,
};

use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::Study;
use crate::planner::{
    PlanView,
    lowered::{EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading},
};

/// A refusal at compilation, numerical mapping, or release admission.
#[derive(Debug, thiserror::Error)]
pub enum ExactImageError {
    /// The source, mapping, or persisted artifact failed validation.
    #[error(transparent)]
    Image(#[from] BridgeImageError),
    /// The causal compiler refused the operation.
    #[error(transparent)]
    Compile(#[from] crate::CausalError),
    /// The release gate refused the route.
    #[error(transparent)]
    Support(#[from] antecedent_core::NotSupported),
}

impl ExactImageError {
    /// Whether the caller stopped this operation.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        match self {
            Self::Image(error) => error.is_cancelled(),
            Self::Compile(crate::CausalError::Cancelled { .. }) => true,
            _ => false,
        }
    }

    /// Whether the bridge image exceeded a caller or structural resource limit.
    #[must_use]
    pub fn is_resource_refusal(&self) -> bool {
        match self {
            Self::Image(error) => error.is_resource_refusal(),
            Self::Compile(crate::CausalError::Resource { .. }) => true,
            _ => false,
        }
    }
}

/// The complete source and mapping retained by an exact-image program.
#[derive(Clone, Debug)]
pub struct CompiledBridgeImage(KernelProgram<BridgeImageRequest>);

impl Study {
    /// Compile a scalar identified-set image or the image of a finite joint outcome law.
    ///
    /// # Errors
    /// Invalid input encoding, compiler refusal or cancellation.
    pub fn compile_bridge_image(
        request: BridgeImageRequest,
        ctx: &ExecutionContext,
    ) -> Result<CompiledBridgeImage, ExactImageError> {
        let shape = match &request {
            BridgeImageRequest::IntervalThreshold { interval, .. } => {
                interval.require_interval()?;
                KernelProgramShape {
                    cell: ProgramCell::BridgeIntervalThreshold,
                    plan_id: "scientific_bridge.interval_threshold",
                    functional: FunctionalKind::PropagatedChain,
                    evaluator: "strict_interval_threshold",
                    realization: KernelRealization::FixedImage,
                    uncertainty: UncertaintyReading::None,
                    role: EvidenceRole::ChainSource,
                    graph_class: "none",
                    data: DataClassification::Tabular,
                    design: "compare both declared interval endpoints to the cutoff in the target coordinate",
                }
            }
            BridgeImageRequest::TrialConfidence(experiment) => KernelProgramShape {
                cell: ProgramCell::TrialBridgePercentileInterval,
                plan_id: "scientific_bridge.trial_bootstrap_interval",
                functional: FunctionalKind::PropagatedChain,
                evaluator: "whole_trial_ols_source_mean_percentile",
                realization: KernelRealization::ReplicateMapping {
                    validation_refits: Some(experiment.replicates),
                },
                uncertainty: UncertaintyReading::Sampling,
                role: EvidenceRole::ChainSource,
                graph_class: "none",
                data: DataClassification::Tabular,
                design: match &experiment.study {
                    TrialMeanStudy::Shared => {
                        "same trial-pairs resample feeds both the source mean and the estimated line"
                    }
                    TrialMeanStudy::Independent { .. } => {
                        "independent source trials and validation trial pairs refitted separately on each replicate"
                    }
                },
            },
            BridgeImageRequest::JointPosterior { .. } => KernelProgramShape {
                cell: ProgramCell::PosteriorAffineBridge,
                plan_id: "scientific_bridge.joint_posterior_interval",
                functional: FunctionalKind::PropagatedChain,
                evaluator: "weighted_joint_posterior_equal_tails",
                realization: KernelRealization::Posterior,
                uncertainty: UncertaintyReading::Posterior,
                role: EvidenceRole::ChainSource,
                graph_class: "none",
                data: DataClassification::Tabular,
                design: "posterior over source and both affine coefficients evaluated together on each supplied row",
            },
            BridgeImageRequest::IdentifiedSet { .. } => KernelProgramShape {
                cell: ProgramCell::IdentifiedSetBridge,
                plan_id: "scientific_bridge.identified_set_image",
                functional: FunctionalKind::IdentifiedSet,
                evaluator: "sharp_affine_interval_image",
                realization: KernelRealization::FixedImage,
                uncertainty: UncertaintyReading::Envelope,
                role: EvidenceRole::ChainSource,
                graph_class: "none",
                data: DataClassification::Tabular,
                design: "image of the complete identified interval on the declared mapping domain",
            },
            BridgeImageRequest::JointOutcomes { source, .. } => KernelProgramShape {
                cell: ProgramCell::JointOutcomeBridge,
                plan_id: "scientific_bridge.joint_outcome_image",
                functional: FunctionalKind::PropagatedChain,
                evaluator: "paired_affine_vector_image",
                realization: KernelRealization::FixedImage,
                uncertainty: match source.semantic() {
                    DistributionMeaningWire::EmpiricalOutcome => UncertaintyReading::Empirical,
                    DistributionMeaningWire::PosteriorPredictive
                    | DistributionMeaningWire::InterventionalPredictive => {
                        UncertaintyReading::Predictive
                    }
                    _ => {
                        return Err(BridgeImageError::Mapping(
                            antecedent_contract::bridge::draws_unusable(
                                "source law",
                                "supply paired observed or predictive outcomes",
                            )
                            .into(),
                        )
                        .into());
                    }
                },
                role: EvidenceRole::ChainSource,
                graph_class: "none",
                data: DataClassification::Tabular,
                design: "row-wise vector image of an aligned finite law retaining its row weights",
            },
        };
        let identity = request.input_identity()?;
        Ok(CompiledBridgeImage(KernelProgram::compile(&shape, request, identity.as_bytes(), ctx)?))
    }
}

impl CompiledBridgeImage {
    /// Compile a decision which preserves this interval's meaning and source.
    ///
    /// # Errors
    ///
    /// The image is not an interval, or the threshold decision does not compile.
    pub fn compile_threshold(
        &self,
        quantity: antecedent_contract::ScientificQuantity,
        threshold: f64,
        ctx: &ExecutionContext,
    ) -> Result<Self, ExactImageError> {
        self.0.input().require_interval()?;
        Study::compile_bridge_image(
            BridgeImageRequest::IntervalThreshold {
                interval: Box::new(self.0.input().clone()),
                quantity,
                threshold,
            },
            ctx,
        )
    }

    /// Inspect the scientific target, operation, carrier and support scope.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.0.view()
    }
    /// All six program layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.0.program_layers()
    }
    /// Compute the exact image from the bound source.
    ///
    /// # Errors
    ///
    /// The execution gate refuses, or the image evaluation fails or is cancelled.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<BridgeImageResult, ExactImageError> {
        self.0.gate(ctx)?;
        Ok(self.0.input().evaluate(ctx)?)
    }
    /// Execute and export the checked result together with its source and declarations.
    ///
    /// # Errors
    ///
    /// Execution fails, or the result cannot be encoded.
    pub fn export(
        &self,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, ExactImageError> {
        Ok(self.export_with_identity(artifact_id, ctx)?.0)
    }

    /// Retain the producer's numerical result identity alongside its bytes in one execution.
    ///
    /// # Errors
    ///
    /// Execution fails, or the result cannot be encoded.
    pub fn export_with_identity(
        &self,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<u8>, SemanticDigest), ExactImageError> {
        let result = self.execute(ctx)?;
        let bytes = self.0.input().export(&result, self.lowered().identity(), artifact_id)?;
        Ok((bytes, result.identity()))
    }
}

/// Recompile the retained inputs and verify both the expected identities and the numerical output.
///
/// # Errors
///
/// The bytes do not decode, an expected identity differs, or the replay refuses.
pub fn consume_bridge_image(
    bytes: &[u8],
    program: SemanticDigest,
    result: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<BridgeImageResult, ExactImageError> {
    let (request, binding) =
        StoredBridgeImage::open_with_context(bytes, program, result, ctx)?.into_parts();
    Ok(replay_stored_image(request, &binding, ctx)?.1)
}

pub(crate) fn replay_stored_image(
    request: BridgeImageRequest,
    binding: &BridgeImageReplayBinding,
    ctx: &ExecutionContext,
) -> Result<(SemanticDigest, BridgeImageResult), ExactImageError> {
    let plan = Study::compile_bridge_image(request, ctx)?;
    plan.0.gate(ctx)?;
    let program = plan.lowered().identity();
    let live = plan.0.input().replay_image(ctx)?;
    let _readout = live.admit_readout(ctx)?;
    binding.verify(program, &live)?;
    if ctx.cancellation.is_cancelled() {
        return Err(BridgeImageError::from(antecedent_io::IoError::StreamCancelled).into());
    }
    Ok((program, live))
}
