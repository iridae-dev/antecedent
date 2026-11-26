//! Exact set and joint-outcome images through the causal compiler.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::{ExecutionContext, SemanticDigest};
pub use antecedent_io::bridge_image_artifact::{
    BridgeImageError, BridgeImageRequest, BridgeImageResult, StoredBridgeImage,
};
pub use antecedent_io::bridge_joint_outcomes::{
    JointOutcomeBridgeError, JointOutcomeMap, JointOutcomeTerm,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

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
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<BridgeImageResult, ExactImageError> {
        self.0.gate(ctx)?;
        Ok(self.0.input().evaluate(ctx)?)
    }
    /// Execute and export the checked result together with its source and declarations.
    pub fn export(
        &self,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, ExactImageError> {
        Ok(self.export_with_identity(artifact_id, ctx)?.0)
    }

    /// Retain the producer's numerical result identity alongside its bytes in one execution.
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
pub fn consume_bridge_image(
    bytes: &[u8],
    program: SemanticDigest,
    result: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<BridgeImageResult, ExactImageError> {
    let stored = StoredBridgeImage::open(bytes, program, result)?;
    let plan = Study::compile_bridge_image(stored.request.clone(), ctx)?;
    let live = plan.execute(ctx)?;
    stored.verify(plan.lowered().identity(), &live)?;
    Ok(live)
}
