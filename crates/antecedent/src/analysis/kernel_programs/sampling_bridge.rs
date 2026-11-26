//! Compiler-owned pushforward of estimator sampling laws and trial-pairs bootstraps.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::scientific_bridge::{
    PreparedScientificBridgeDistribution, ScientificBridgeDistributionResult,
    ScientificBridgeRefusal,
};
use crate::planner::{
    PlanView,
    lowered::{EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading},
};
use antecedent_contract::bridge::BridgeMapping;
use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_io::{IoError, distribution_artifact::DistributionArtifact};

const SHAPE: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SamplingDistributionBridge,
    plan_id: "scientific_bridge.sampling_distribution",
    functional: FunctionalKind::PropagatedChain,
    evaluator: "replicatewise_fixed_map_or_validation_trial_refit",
    realization: KernelRealization::ReplicateMapping { validation_refits: None },
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::ChainSource,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: "source replicate identity and trial resampling design declared before mapping",
};

#[derive(Clone, Debug)]
struct Inputs {
    bridge: PreparedScientificBridgeDistribution,
    source: DistributionArtifact,
    kind: Option<ClaimKind>,
    stream: u64,
}

/// Failure of the substantive declaration, compiler, release gate, or caller resource budget.
#[derive(Debug, thiserror::Error)]
pub enum SamplingBridgeError {
    /// Native scientific refusal.
    #[error(transparent)]
    Scientific(#[from] ScientificBridgeRefusal),
    /// Causal program lowering failed.
    #[error(transparent)]
    Compile(#[from] crate::CausalError),
    /// Artifact encoding or execution admission failed.
    #[error(transparent)]
    Artifact(#[from] IoError),
    /// The capability unit has yet to earn release support.
    #[error(transparent)]
    NotSupported(#[from] antecedent_core::NotSupported),
}

/// A declared map, source replicates and bootstrap stream bound to an inspectable causal program.
#[derive(Clone, Debug)]
pub struct CompiledSamplingBridge(KernelProgram<Inputs>);

impl PreparedScientificBridgeDistribution {
    /// Bind the exact source distribution, map and replicate stream before execution.
    ///
    /// # Errors
    /// A declaration of another scientific law, source encoding failure, cancellation or compiler refusal.
    pub fn compile_sampling(
        &self,
        source: DistributionArtifact,
        kind: Option<ClaimKind>,
        stream: u64,
        ctx: &ExecutionContext,
    ) -> Result<CompiledSamplingBridge, SamplingBridgeError> {
        if !self.contract().source_value.is_sampling() {
            return Err(ScientificBridgeRefusal::from_contract(
                &antecedent_contract::bridge::draws_unusable(
                    "source law",
                    "declare a sampling or bootstrap bridge",
                ),
            )
            .into());
        }
        let mut key =
            CanonicalWriter::new("antecedent.scientific_bridge.sampling_program_input.v1");
        key.bytes(self.contract_identity().as_bytes())
            .bytes(&source.to_bytes("sampling-source")?)
            .text(kind.unwrap_or(ClaimKind::Mixture).as_str())
            .word(stream);
        let mut shape = SHAPE;
        if matches!(self.contract().mapping, BridgeMapping::TrialLevelLinear(_)) {
            let replicates = u32::try_from(source.n_draws()).map_err(|_| IoError::TooLarge)?;
            shape.realization =
                KernelRealization::ReplicateMapping { validation_refits: Some(replicates) };
            shape.design = match self.contract().source_dependence {
                Some(antecedent_contract::bridge::SourceMappingDependence::SharedTrialMean {
                    ..
                }) => "source mean and validation map refitted on identical whole-trial pairs",
                _ => "independent validation-study trial pairs refitted for each source replicate",
            };
        }
        let inputs = Inputs { bridge: self.clone(), source, kind, stream };
        Ok(CompiledSamplingBridge(KernelProgram::compile(&shape, inputs, &key.finish(), ctx)?))
    }
}

impl CompiledSamplingBridge {
    /// The six lowered layers of the program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.0.program_layers()
    }

    /// The route, evidence bindings and public support status.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.0.view()
    }

    /// Replay each mapped replicate under the caller's resource budget.
    ///
    /// # Errors
    /// The capability gate, caller resource admission, or numerical validation of the refits.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ScientificBridgeDistributionResult, SamplingBridgeError> {
        self.0.gate(ctx)?;
        let inputs = self.0.input();
        let trials = match &inputs.bridge.contract().mapping {
            BridgeMapping::TrialLevelLinear(v) => v.trials.len(),
            _ => 0,
        };
        let bytes = (inputs.source.draws().len() as u64)
            .saturating_mul(48)
            .saturating_add((inputs.source.n_draws() as u64).saturating_mul(64))
            .saturating_add((trials as u64).saturating_mul(64));
        let _lease = ctx.acquire_serial_numerical_admission(bytes).map_err(IoError::from)?;
        Ok(inputs.bridge.derive_cancellable(
            &inputs.source,
            inputs.kind,
            inputs.stream,
            &ctx.cancellation,
        )?)
    }
}
