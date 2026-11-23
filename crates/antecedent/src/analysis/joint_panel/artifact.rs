//! Full native compiled/frozen execution interchange with independent historical replay.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{ExecutionContext, JointPanelFit, JointPanelResult};
use antecedent_io::joint_panel_artifact::{
    FrozenPanelFit, PanelArtifactError, PanelExecutionProgram, PanelExecutionSource,
    PanelSourceMetadata, PanelUnitMetadata,
};

impl JointPanelResult {
    /// Preserve exact compiled inputs, original proof and frozen shared numerical law.
    ///
    /// This authenticates/replays evaluation of a stored mechanism. It does not attest
    /// that its stored fit/draws originated from the source data or grant current permission.
    /// # Errors
    /// Resource limits, unsupported metadata or original compiler/proof errors.
    pub fn artifact_program(&self) -> Result<PanelExecutionProgram, PanelArtifactError> {
        let source = self.prepared().source_data();
        let metadata = PanelSourceMetadata {
            schema: antecedent_io::convert::schema_to_wire(source.schema()),
            units: source
                .units()
                .iter()
                .map(|unit| PanelUnitMetadata {
                    id: unit.unit_id,
                    interval_ns: match unit.series.time_index().regularity {
                        antecedent_data::SamplingRegularity::Regular { interval_ns } => interval_ns,
                        antecedent_data::SamplingRegularity::Irregular => 0,
                    },
                    rows: unit.series.time_index().length,
                    content_digest: unit.series.storage().content_digest(),
                })
                .collect(),
        };
        let frozen = match &self.fit {
            JointPanelFit::Invariant(fit) => FrozenPanelFit::invariant(fit),
            JointPanelFit::Varying(fit) => FrozenPanelFit::varying(fit),
        };
        PanelExecutionProgram::from_source(PanelExecutionSource {
            source: metadata,
            data: self.prepared().model_data(),
            options: &self.request().options,
            temporal: self.request().temporal.as_ref(),
            programs: self.request().worlds.iter().map(|world| world.program.clone()).collect(),
            world_ids: self.request().worlds.iter().map(|world| world.world.raw()).collect(),
            target: &self.request().target,
            proof: self.identification(),
            source_identity: self.program.state.source_identity,
            dependency: self.dependency_identity(),
            frozen,
            context: &self.context,
        })
    }
    /// Encode the already fitted/executed compiled product without refitting any mechanism.
    /// # Errors
    /// Original metadata/proof, serialization, size, resource or cancellation refusal.
    pub fn export(&self, ctx: &ExecutionContext) -> Result<Vec<u8>, PanelArtifactError> {
        antecedent_io::joint_panel_artifact::encode_panel_execution(
            self.artifact_program()?,
            &self.point,
            &self.draws,
            ctx,
        )
    }
    /// Consume against the caller's actual current compiled request and frozen law.
    ///
    /// Independent replay checks the stored evaluation; this additional comparison uses
    /// the caller's original current target/proof/source/options/fit/draw identities.
    /// # Errors
    /// Replay failure or any changed current dependency/target/frozen law is refused.
    pub fn consume_current(
        &self,
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<Self, PanelArtifactError> {
        let replay = antecedent_io::joint_panel_artifact::consume_panel_execution(bytes, ctx)?;
        if replay.program().identity()? != self.artifact_program()?.identity()? {
            return Err(PanelArtifactError::Mismatch);
        }
        Ok(self.clone())
    }
}
