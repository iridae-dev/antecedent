//! The measured sample-split effect and parent-set effect as compiled program families.
//!
//! A split-effect source (`sample_split_effect::SplitEffect::source_artifact`) and a
//! parent-set source (`sample_split_parent_set_effect::ParentSetEffect::source_artifact`)
//! carry everything a replay needs: the columns, the split seed, the treatment and outcome.
//! Compiling one binds its digest into a six-layer program under the `SampleSplitEffect` or
//! `SampleSplitParentSetEffect` route. Measuring passes the support gate of the route's unit
//! and then replays the source exactly as `SplitEffect::measure` and `ParentSetEffect::measure`
//! do: the row split, the PC discovery, the acceptance of the structure, the adjusted
//! regression and the binding of the interval to its coverage record. The numbers, the
//! calibration basis and the coverage record are the direct path's.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::ExecutionContext;
use antecedent_io::IoError;
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::measured_inference::{split_effect, split_parent_set_effect};

use super::{KernelProgram, KernelProgramShape, KernelRealization, io_refusal};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const SPLIT_EFFECT: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SampleSplitEffect,
    plan_id: "discovery.sample_split_effect",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "sample_split_pc_adjusted_ols",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::Observations,
    graph_class: "discovered_cpdag",
    data: DataClassification::Tabular,
    design: "discovery on one part of a seeded row split, adjusted regression on the other",
};

const PARENT_SET_EFFECT: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SampleSplitParentSetEffect,
    plan_id: "discovery.sample_split_parent_set_effect",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "sample_split_pc_parent_set_ols",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::Observations,
    graph_class: "discovered_cpdag",
    data: DataClassification::Tabular,
    design: "discovery on one part of a seeded row split, parent-set regressions on the other",
};

/// A compiled split-effect source: the canonical source bytes and the program that binds them.
#[derive(Clone, Debug)]
pub struct CompiledSplitEffect {
    program: KernelProgram<Arc<[u8]>>,
}

/// A compiled parent-set effect source: the canonical source bytes and the program that binds
/// them.
#[derive(Clone, Debug)]
pub struct CompiledParentSetEffect {
    program: KernelProgram<Arc<[u8]>>,
}

impl Study {
    /// Compile a sample-split effect source (`SplitEffect::source_artifact`) under the
    /// `SampleSplitEffect` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_split_effect_source(
        source_artifact: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<CompiledSplitEffect, CausalError> {
        let program = KernelProgram::compile(
            &SPLIT_EFFECT,
            Arc::from(source_artifact),
            source_artifact,
            ctx,
        )?;
        Ok(CompiledSplitEffect { program })
    }

    /// Compile a sample-split parent-set effect source (`ParentSetEffect::source_artifact`)
    /// under the `SampleSplitParentSetEffect` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_split_parent_set_effect_source(
        source_artifact: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<CompiledParentSetEffect, CausalError> {
        let program = KernelProgram::compile(
            &PARENT_SET_EFFECT,
            Arc::from(source_artifact),
            source_artifact,
            ctx,
        )?;
        Ok(CompiledParentSetEffect { program })
    }
}

impl CompiledSplitEffect {
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

    /// The canonical source bytes the program binds.
    #[must_use]
    pub fn source_artifact(&self) -> &[u8] {
        self.program.input()
    }

    /// Pass the support gate, replay the source and attest the interval from the coverage
    /// records, as `SplitEffect::measure` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not supported,
    /// and the refusals of the replay. Where no current attesting record measured this
    /// construction, the interval is still reported with calibration `not_measured` or
    /// `outside_measured_scope`.
    pub fn measure(&self, ctx: &ExecutionContext) -> Result<MeasuredInference, IoError> {
        self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        split_effect::measure(self.program.input(), ctx)
    }
}

impl CompiledParentSetEffect {
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

    /// The canonical source bytes the program binds.
    #[must_use]
    pub fn source_artifact(&self) -> &[u8] {
        self.program.input()
    }

    /// Pass the support gate, replay the source and attest every interval from the coverage
    /// records, as `ParentSetEffect::measure` does.
    ///
    /// # Errors
    ///
    /// As [`CompiledSplitEffect::measure`].
    pub fn measure(&self, ctx: &ExecutionContext) -> Result<MeasuredInference, IoError> {
        self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        split_parent_set_effect::measure(self.program.input(), ctx)
    }
}
