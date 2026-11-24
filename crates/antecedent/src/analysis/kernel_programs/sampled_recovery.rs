//! The sampled observation recovery and its measured source replay as compiled program families.
//!
//! A sampled recovery evaluates the checked observation-recovery formula of a derivation on
//! counted whole observation rows and bootstraps the whole recovery and effect evaluation
//! (`antecedent_estimate::estimate_sampled_recovery`). Compiling the derivation, the rows and the
//! request configuration binds their digest into a six-layer program under the
//! `SampledObservationRecovery` route; estimating passes the support gate of the route's unit
//! and then calls `estimate_sampled_recovery` with the same arguments, so the point, the
//! interval, the replicate receipt and the calibration basis are the direct path's.
//!
//! A sampled-recovery source artifact is replayed and bound to its coverage records by
//! `SampledRecoveryArtifactWire::measure`. Compiling the source bytes, the expected identity and
//! the consumer limits binds their digest under the `MeasuredSampledObservationRecovery` route;
//! measuring passes the gate and then calls `SampledRecoveryArtifactWire::measure` with the same
//! arguments, so the measured envelope is the direct path's byte for byte.
//!
//! Both routes are bound to the `recovery.sampled_observation` unit.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{CanonicalWriter, ExecutionContext, NotSupported};
use antecedent_estimate::{
    SampledObservationInput, SampledRecoveryConfig, SampledRecoveryDetail, SampledRecoveryError,
    SampledRecoveryResult, estimate_sampled_recovery,
};
use antecedent_identify::RecoveryDerivation;
use antecedent_io::IoError;
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::sampled_recovery_artifact::{
    SampledRecoveryArtifactWire, SampledRecoveryConsumeLimits, SampledRecoveryExpectation,
};

use super::{KernelProgram, KernelProgramShape, KernelRealization, io_refusal};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const SAMPLED_RECOVERY: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SampledObservationRecovery,
    plan_id: "recovery.sampled_observation",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "whole_row_bootstrap_bca_observation_recovery",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::Observations,
    graph_class: "binary_m_graph",
    data: DataClassification::Tabular,
    design: "checked observation-recovery formula on counted whole rows, whole-row bootstrap",
};

const MEASURED_SAMPLED_RECOVERY: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::MeasuredSampledObservationRecovery,
    plan_id: "recovery.sampled_observation.measured_source",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "whole_row_bootstrap_bca_observation_recovery_replay",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::Observations,
    graph_class: "binary_m_graph",
    data: DataClassification::Tabular,
    design: "replay of a sampled-recovery source bound to its coverage records",
};

/// The derivation, rows and configuration a sampled recovery evaluates.
#[derive(Clone, Debug)]
struct SampledRecoveryRequest {
    derivation: RecoveryDerivation,
    input: SampledObservationInput,
    config: SampledRecoveryConfig,
}

/// The source bytes, expected identity and consumer limits a measured replay reads.
#[derive(Clone, Debug)]
struct MeasuredSampledRecoveryRequest {
    source: Arc<[u8]>,
    expected: SampledRecoveryExpectation,
    limits: SampledRecoveryConsumeLimits,
}

/// Canonical bytes of a sampled-recovery request: the derivation identity, the snapshot, every
/// row in order and every configuration field.
fn request_bytes(request: &SampledRecoveryRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.kernel_program.sampled_observation_recovery");
    writer.text(&request.derivation.identity());
    writer.text(&request.input.snapshot_id);
    writer.word(request.input.rows.len() as u64);
    for row in &request.input.rows {
        writer.word(row.id);
        writer.bytes(&[row.pattern.responses, row.pattern.proxies, row.pattern.fully]);
    }
    let config = &request.config;
    writer
        .text(config.interval_method.name())
        .word(config.replicates as u64)
        .word(config.seed)
        .real(config.max_failed_fraction)
        .real(config.normalization_tolerance)
        .real(config.small_cell_count)
        .real(config.treated_level)
        .real(config.control_level);
    writer.finish()
}

/// Canonical bytes of a measured replay: the source bytes, the expected digests and the limits.
fn measured_bytes(request: &MeasuredSampledRecoveryRequest) -> [u8; 32] {
    let mut writer =
        CanonicalWriter::new("antecedent.kernel_program.measured_sampled_observation_recovery");
    writer.bytes(&request.source);
    for expected in [&request.expected.premises_digest, &request.expected.data_digest] {
        match expected {
            Some(digest) => writer.word(1).text(digest),
            None => writer.word(0),
        };
    }
    let limits = &request.limits;
    writer
        .word(limits.decision.search.operations as u64)
        .word(limits.decision.search.depth as u64)
        .word(limits.decision.memory_bytes)
        .word(limits.max_rows as u64)
        .word(limits.max_replay_work);
    writer.finish()
}

/// The support refusal as a sampled-recovery refusal: the route is outside what this build
/// supports, carried through the kernel's `route_not_supported` detail.
fn sampled_refusal(refusal: &NotSupported) -> SampledRecoveryError {
    SampledRecoveryError {
        detail: SampledRecoveryDetail::BoundsExceeded,
        message: refusal.message().to_owned(),
        patterns: Vec::new(),
        recovery: None,
    }
}

/// A compiled sampled observation recovery: the derivation, the rows, the configuration and the
/// program that binds them.
#[derive(Clone, Debug)]
pub struct CompiledSampledObservationRecovery {
    program: KernelProgram<SampledRecoveryRequest>,
}

/// A compiled replay of a sampled-recovery source artifact against its coverage records.
#[derive(Clone, Debug)]
pub struct CompiledMeasuredSampledObservationRecovery {
    program: KernelProgram<MeasuredSampledRecoveryRequest>,
}

impl Study {
    /// Compile a sampled observation recovery (the inputs of
    /// `antecedent_estimate::estimate_sampled_recovery`) under the `SampledObservationRecovery`
    /// route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_sampled_observation_recovery(
        derivation: RecoveryDerivation,
        input: SampledObservationInput,
        config: SampledRecoveryConfig,
        ctx: &ExecutionContext,
    ) -> Result<CompiledSampledObservationRecovery, CausalError> {
        let request = SampledRecoveryRequest { derivation, input, config };
        let canonical = request_bytes(&request);
        let program = KernelProgram::compile(&SAMPLED_RECOVERY, request, &canonical, ctx)?;
        Ok(CompiledSampledObservationRecovery { program })
    }

    /// Compile the measured replay of a sampled-recovery source artifact (the inputs of
    /// `SampledRecoveryArtifactWire::measure`) under the `MeasuredSampledObservationRecovery`
    /// route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_measured_sampled_observation_recovery(
        source_artifact: &[u8],
        expected: SampledRecoveryExpectation,
        limits: SampledRecoveryConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<CompiledMeasuredSampledObservationRecovery, CausalError> {
        let request =
            MeasuredSampledRecoveryRequest { source: Arc::from(source_artifact), expected, limits };
        let canonical = measured_bytes(&request);
        let program = KernelProgram::compile(&MEASURED_SAMPLED_RECOVERY, request, &canonical, ctx)?;
        Ok(CompiledMeasuredSampledObservationRecovery { program })
    }
}

impl CompiledSampledObservationRecovery {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// The checked recovery derivation the program binds.
    #[must_use]
    pub fn derivation(&self) -> &RecoveryDerivation {
        &self.program.input().derivation
    }

    /// The counted observation rows the program binds.
    #[must_use]
    pub fn observations(&self) -> &SampledObservationInput {
        &self.program.input().input
    }

    /// The request configuration the program binds.
    #[must_use]
    pub fn config(&self) -> &SampledRecoveryConfig {
        &self.program.input().config
    }

    /// Pass the support gate, then evaluate the recovery and its whole-row bootstrap as
    /// `estimate_sampled_recovery` does.
    ///
    /// # Errors
    ///
    /// The support refusal (reason `not_supported` in the message, detail
    /// `sampled_recovery.bounds_exceeded`) in a release build when the unit is not supported,
    /// and every refusal of `estimate_sampled_recovery`.
    pub fn estimate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SampledRecoveryResult, SampledRecoveryError> {
        self.program.gate(ctx).map_err(|refusal| sampled_refusal(&refusal))?;
        let request = self.program.input();
        estimate_sampled_recovery(&request.derivation, &request.input, &request.config, ctx)
    }
}

impl CompiledMeasuredSampledObservationRecovery {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// The source artifact bytes the program binds.
    #[must_use]
    pub fn source_artifact(&self) -> &[u8] {
        &self.program.input().source
    }

    /// Pass the support gate, replay the source and bind its scalar to the coverage records, as
    /// `SampledRecoveryArtifactWire::measure` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not supported,
    /// and the refusals of the replay. Where no current attesting record measured this
    /// construction, the interval is still reported with calibration `not_measured`.
    pub fn measure(&self, ctx: &ExecutionContext) -> Result<MeasuredInference, IoError> {
        self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        let request = self.program.input();
        SampledRecoveryArtifactWire::measure(
            &request.source,
            &request.expected,
            request.limits,
            ctx,
        )
    }
}
