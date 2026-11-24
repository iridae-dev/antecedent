//! Compiled measured whole-unit interval of a finite two-step temporal response.
//!
//! [`Study::compile_temporal_dependent_interval`] binds a [`TemporalDependentIntervalRequest`]
//! (the panel snapshot and its repeated units with their complete histories, the two-step
//! sequence, the estimand and the resampling design) into a lowered program: the sequence,
//! estimand and design are the program's question, the unit histories its source.
//! [`CompiledTemporalDependentInterval::execute`] runs what the deprecated
//! `temporal_dependent_interval` runs: the original whole-unit candidate is built (panel and
//! design validation, the studentized whole-unit resampling seeded from the design's seed) and
//! its artifact replayed by `TemporalIntervalArtifactWire::measure`, which reports the response
//! interval as a measured scalar with the calibration slot of its construction.
//!
//! The direct entry runs under the production context of the design's seed; the compiled route
//! runs under the caller's context. The resampling stream is seeded from the design's seed in
//! both, so the context changes cancellation and admission, never a number.
//!
//! The coverage of this interval is measured for the route `temporal_interval` (the coverage
//! records cited by the `temporal.dependent_interval` capability record);
//! [`MeasuredIntervalRun::calibration_route`] names it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::temporal_dependent_interval::{DependentIntervalConfig, UnitHistories};
use antecedent_io::IoError;

use super::compiled_measured_interval::{
    MeasuredIntervalProgram, MeasuredIntervalRun, frequentist,
};
use super::program_handle::{CompiledProgramHandle, gate_program};
use super::temporal_extensions::{IntervalEstimand, measure_dependent_interval};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};
use crate::planner::{PlanStage, PlanView};

const PLAN_ID: &str = "temporal_dependent_interval.v1";
/// The measured route the coverage of this interval is recorded under.
const CALIBRATION_ROUTE: &str = "temporal_interval";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// A whole-unit dependent interval request: the arguments `temporal_dependent_interval` takes,
/// owned.
#[derive(Clone, Debug)]
pub struct TemporalDependentIntervalRequest {
    snapshot_id: String,
    units: Option<Vec<UnitHistories>>,
    sequence: [u32; 2],
    estimand: IntervalEstimand,
    config: DependentIntervalConfig,
}

impl TemporalDependentIntervalRequest {
    /// Retain the panel snapshot, its units, the two-step sequence, the estimand and the
    /// resampling design. Nothing is validated here: the measured consumer validates the panel
    /// and design at execution, with the refusals of the direct entry.
    #[must_use]
    pub fn new(
        snapshot_id: &str,
        units: Option<Vec<UnitHistories>>,
        sequence: [u32; 2],
        estimand: IntervalEstimand,
        config: DependentIntervalConfig,
    ) -> Self {
        Self { snapshot_id: snapshot_id.to_owned(), units, sequence, estimand, config }
    }

    /// The panel snapshot id.
    #[must_use]
    pub fn snapshot_id(&self) -> &str {
        &self.snapshot_id
    }

    /// The two-step sequence.
    #[must_use]
    pub const fn sequence(&self) -> [u32; 2] {
        self.sequence
    }

    /// The estimand.
    #[must_use]
    pub const fn estimand(&self) -> &IntervalEstimand {
        &self.estimand
    }

    /// The resampling design.
    #[must_use]
    pub const fn config(&self) -> &DependentIntervalConfig {
        &self.config
    }
}

/// A whole-unit dependent interval request bound to its lowered program.
#[derive(Clone, Debug)]
pub struct CompiledTemporalDependentInterval {
    request: TemporalDependentIntervalRequest,
    lowered: Arc<LoweredProgram>,
}

fn question_digest(request: &TemporalDependentIntervalRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.temporal_dependent_interval.question.v1");
    let config = &request.config;
    writer.text(&request.snapshot_id).u32s(&request.sequence);
    // The estimand's debug form spells the state or the law's population, snapshot and
    // masses, floats in their exact round-trip form.
    writer.text(&format!("{:?}", request.estimand));
    writer
        .word(config.replicates as u64)
        .word(config.seed)
        .real(config.level)
        .text(&format!("{:?}", config.method))
        .word(config.min_units as u64)
        .real(config.max_failed_fraction);
    writer.finish()
}

fn source_digest(request: &TemporalDependentIntervalRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.temporal_dependent_interval.units.v1");
    let Some(units) = &request.units else {
        writer.word(0);
        return writer.finish();
    };
    writer.word(1).word(units.len() as u64);
    for unit in units {
        writer.word(unit.unit_id).word(unit.histories.len() as u64);
        for history in &unit.histories {
            writer
                .word(history.time_id)
                .word(u64::from(history.s0))
                .u32s(&history.actions)
                .u32s(&history.covariates)
                .real(history.y);
        }
    }
    writer.finish()
}

fn lower(request: &TemporalDependentIntervalRequest) -> Result<Arc<LoweredProgram>, CausalError> {
    let histories = request
        .units
        .as_ref()
        .map_or(0, |units| units.iter().map(|unit| unit.histories.len() as u64).sum());
    MeasuredIntervalProgram {
        cell: ProgramCell::TemporalDependentInterval,
        plan_id: PLAN_ID,
        kind: FunctionalKind::InterventionalDistribution,
        graph_class: "selected_two_step_binary_temporal_admg",
        question: question_digest(request),
        source: source_digest(request),
        source_role: EvidenceRole::Observations,
        realization: frequentist("whole_unit_studentized_bootstrap", PLAN_ID)?,
        uncertainty: UncertaintyReading::Sampling,
        rows: histories,
        columns: 6,
        claim: ClaimKind::Response,
    }
    .lower()
}

impl Study {
    /// Bind a whole-unit dependent interval request into a compiled program.
    ///
    /// # Errors
    ///
    /// A score or physical operator that cannot be declared.
    pub fn compile_temporal_dependent_interval(
        request: TemporalDependentIntervalRequest,
    ) -> Result<CompiledTemporalDependentInterval, CausalError> {
        let lowered = lower(&request)?;
        Ok(CompiledTemporalDependentInterval { request, lowered })
    }
}

impl CompiledProgramHandle for CompiledTemporalDependentInterval {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledTemporalDependentInterval {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the plan, including the support decision of its unit, without evaluating.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// The bound request.
    #[must_use]
    pub const fn request(&self) -> &TemporalDependentIntervalRequest {
        &self.request
    }

    /// Build the whole-unit candidate, replay it and report the measured interval.
    ///
    /// # Errors
    ///
    /// Every refusal of the direct entry (an invalid panel, design or estimand, an unsupported
    /// history, a retired method, absent unit dependence), cancellation, admission, or the
    /// support gate once it is on.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<MeasuredIntervalRun, IoError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self).map_err(|refusal| IoError::Refused {
                code: refusal.code(),
                message: format!("{}: {refusal}", refusal.detail),
            })?;
        }
        let request = &self.request;
        let measured = measure_dependent_interval(
            &request.snapshot_id,
            request.units.clone(),
            request.sequence,
            request.estimand.clone(),
            &request.config,
            &scoped,
        )?;
        Ok(MeasuredIntervalRun {
            measured,
            lowered_program_identity: self.lowered.identity(),
            calibration_route: CALIBRATION_ROUTE,
        })
    }
}
