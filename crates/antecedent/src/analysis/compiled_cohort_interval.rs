//! Compiled measured interval of a declared cohort's mean conditional effect.
//!
//! [`Study::compile_cohort_interval`] binds a native linear-effect fit ([`NativeCohortFit`]: the
//! OLS fit with its HC3 covariance, its export and the rows it was fitted from), the cohort
//! decision source built from it ([`CohortDecisionSource`]) and a nominal level into a lowered
//! program: the export, the cohort source and the level are the program's question, the fitted
//! rows its source. [`CompiledCohortInterval::execute`] runs the measured consumer the
//! deprecated `measured_cohort_interval` runs: both artifacts are encoded and replayed by
//! `CohortIntervalArtifactWire::measure`, which refits from the retained rows, requires the
//! export to reproduce, and reports the Student-t interval of the cohort mean as a measured
//! scalar with the calibration slot of its construction. A declared (caller-built) export or a
//! shifted cohort is refused exactly as on the direct entry.
//!
//! The coverage of this interval is measured for the route `cohort_interval` (the coverage
//! records cited by the `cohort_decision.interval` capability record);
//! [`MeasuredIntervalRun::calibration_route`] names it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_io::IoError;

use super::cohort_decision::{CohortDecisionSource, NativeCohortFit, measure_cohort_interval};
use super::compiled_measured_interval::{
    MeasuredIntervalProgram, MeasuredIntervalRun, frequentist,
};
use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};
use crate::planner::{PlanStage, PlanView};

const PLAN_ID: &str = "cohort_mean_interval.v1";
/// The measured route the coverage of this interval is recorded under.
const CALIBRATION_ROUTE: &str = antecedent_io::measured_inference::cohort_interval::ROUTE;
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// A native cohort fit, its cohort source and a nominal level bound to their lowered program.
#[derive(Clone, Debug)]
pub struct CompiledCohortInterval {
    native: NativeCohortFit,
    source: CohortDecisionSource,
    level: f64,
    lowered: Arc<LoweredProgram>,
}

fn question_digest(
    native: &NativeCohortFit,
    source: &CohortDecisionSource,
    level: f64,
) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.cohort_mean_interval.question.v1");
    writer.text(native.export().identity()).text(source.identity()).real(level);
    writer.finish()
}

fn source_digest(native: &NativeCohortFit) -> [u8; 32] {
    let rows = native.rows();
    let mut writer = CanonicalWriter::new("antecedent.cohort_mean_interval.rows.v1");
    writer.reals(&rows.outcome).reals(&rows.treatment).texts(&rows.feature_ids);
    writer.word(rows.features.len() as u64);
    for column in &rows.features {
        writer.reals(column);
    }
    writer.finish()
}

fn lower(
    native: &NativeCohortFit,
    source: &CohortDecisionSource,
    level: f64,
) -> Result<Arc<LoweredProgram>, CausalError> {
    let rows = native.rows();
    MeasuredIntervalProgram {
        cell: ProgramCell::CohortMeanInterval,
        plan_id: PLAN_ID,
        kind: FunctionalKind::AssignmentContrast,
        graph_class: "as_the_cohort_mean_contrast_route",
        question: question_digest(native, source, level),
        source: source_digest(native),
        source_role: EvidenceRole::Observations,
        realization: frequentist("ols_hc3_treatment_by_basis_student_t", PLAN_ID)?,
        uncertainty: UncertaintyReading::Sampling,
        rows: rows.outcome.len() as u64,
        columns: 2 + rows.features.len() as u64,
        claim: ClaimKind::Point,
    }
    .lower()
}

impl Study {
    /// Bind a native cohort fit, the cohort source built from it and a nominal level into a
    /// compiled program.
    ///
    /// # Errors
    ///
    /// A score or physical operator that cannot be declared.
    pub fn compile_cohort_interval(
        native: NativeCohortFit,
        source: CohortDecisionSource,
        level: f64,
    ) -> Result<CompiledCohortInterval, CausalError> {
        let lowered = lower(&native, &source, level)?;
        Ok(CompiledCohortInterval { native, source, level, lowered })
    }
}

impl CompiledProgramHandle for CompiledCohortInterval {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledCohortInterval {
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

    /// The bound native fit.
    #[must_use]
    pub const fn native(&self) -> &NativeCohortFit {
        &self.native
    }

    /// The bound cohort source.
    #[must_use]
    pub const fn source(&self) -> &CohortDecisionSource {
        &self.source
    }

    /// The bound nominal level.
    #[must_use]
    pub const fn level(&self) -> f64 {
        self.level
    }

    /// Replay the fit and the cohort source and report the measured cohort-mean interval.
    ///
    /// # Errors
    ///
    /// Every refusal of the measured consumer (a source not built from this fit, a declared
    /// export, a shifted cohort, an invalid level, a changed artifact), cancellation, or the
    /// support gate once it is on.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<MeasuredIntervalRun, IoError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self).map_err(|refusal| IoError::Refused {
                code: refusal.code(),
                message: format!("{}: {refusal}", refusal.detail),
            })?;
        }
        let measured = measure_cohort_interval(&self.native, &self.source, self.level, &scoped)?;
        Ok(MeasuredIntervalRun {
            measured,
            lowered_program_identity: self.lowered.identity(),
            calibration_route: CALIBRATION_ROUTE,
        })
    }
}
