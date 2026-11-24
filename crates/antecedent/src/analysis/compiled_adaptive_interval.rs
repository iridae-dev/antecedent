//! Compiled measured interval of one result of an adaptive experiment.
//!
//! [`Study::compile_adaptive_interval`] binds a prepared [`AdaptiveInferenceSession`] (its
//! validated contract: assignment history, realized probabilities, exposure, outcome
//! availability, looks and stopping rule), one named result ([`AdaptiveRequest`]) and the
//! artifact id the result is sealed under into a lowered program: the request and artifact id
//! are the program's question, the contract its source. [`CompiledAdaptiveInterval::execute`]
//! runs what the deprecated `AdaptiveInferenceSession::measured_interval` runs: the result is
//! sealed, replayed by a fresh reader and measured by
//! `AdaptiveInferenceResultArtifact::measure_interval`, which builds the interval with the
//! construction whose whole-procedure coverage was measured (the Thompson credible interval of
//! the second arm, the weighted contrast, the confidence sequence or the repeated confidence
//! interval) and reports it as a measured scalar with the calibration slot of its procedure
//! cell. A procedure outside the measured cells still reports its interval, annotated.
//!
//! The coverage of these intervals is measured for the route `adaptive_interval` (the
//! coverage records the `adaptive_inference` records cite);
//! [`MeasuredIntervalRun::calibration_route`] names it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{CanonicalWriter, ExecutionContext};

use super::adaptive_inference::{
    AdaptiveInferenceRefusal, AdaptiveInferenceSession, AdaptiveRequest, InferenceParadigm,
};
use super::compiled_measured_interval::{
    MeasuredIntervalProgram, MeasuredIntervalRun, bayesian, frequentist,
};
use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};
use crate::planner::{PlanStage, PlanView};

const PLAN_ID: &str = "adaptive_result_interval.v1";
/// The measured route the coverage of these intervals is recorded under.
const CALIBRATION_ROUTE: &str = "adaptive_interval";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// One named result of a prepared adaptive experiment bound to its lowered program.
#[derive(Clone, Debug)]
pub struct CompiledAdaptiveInterval<'s> {
    session: &'s AdaptiveInferenceSession,
    request: AdaptiveRequest,
    artifact_id: String,
    lowered: Arc<LoweredProgram>,
}

fn question_digest(request: &AdaptiveRequest, artifact_id: &str) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.adaptive_result_interval.question.v1");
    // The request's debug form spells every variant field, floats in their exact round-trip
    // form; the sealed result the consumer replays binds the same request canonically.
    writer.text(&format!("{request:?}")).text(artifact_id);
    writer.finish()
}

fn source_digest(session: &AdaptiveInferenceSession) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.adaptive_result_interval.contract.v1");
    writer.bytes(session.contract().identity().as_bytes());
    writer.finish()
}

fn lower(
    session: &AdaptiveInferenceSession,
    request: &AdaptiveRequest,
    artifact_id: &str,
) -> Result<Arc<LoweredProgram>, CausalError> {
    let posterior = matches!(
        request.paradigm(),
        InferenceParadigm::BayesianPosterior | InferenceParadigm::DecisionTheoreticStopping
    );
    let (realization, uncertainty) = if posterior {
        (
            bayesian(
                "bernoulli_arms_under_adaptive_assignment",
                "declared_beta_arm_priors".to_owned(),
                PLAN_ID,
            )?,
            UncertaintyReading::Posterior,
        )
    } else {
        (
            frequentist("adaptive_procedure_whole_procedure_interval", PLAN_ID)?,
            UncertaintyReading::Sampling,
        )
    };
    let declaration = session.contract().declaration();
    MeasuredIntervalProgram {
        cell: ProgramCell::AdaptiveResultInterval,
        plan_id: PLAN_ID,
        kind: FunctionalKind::AssignmentContrast,
        graph_class: "caller_attested_adaptive_assignment",
        question: question_digest(request, artifact_id),
        source: source_digest(session),
        source_role: EvidenceRole::Trial,
        realization,
        uncertainty,
        rows: declaration.assigned_arm.len() as u64,
        columns: 3,
        claim: ClaimKind::Point,
    }
    .lower()
}

impl Study {
    /// Bind one named result of a prepared adaptive experiment, sealed under `artifact_id`, into
    /// a compiled program.
    ///
    /// # Errors
    ///
    /// A likelihood, prior, score or physical operator that cannot be declared.
    pub fn compile_adaptive_interval<'s>(
        session: &'s AdaptiveInferenceSession,
        request: AdaptiveRequest,
        artifact_id: &str,
    ) -> Result<CompiledAdaptiveInterval<'s>, CausalError> {
        let lowered = lower(session, &request, artifact_id)?;
        Ok(CompiledAdaptiveInterval {
            session,
            request,
            artifact_id: artifact_id.to_owned(),
            lowered,
        })
    }
}

impl CompiledProgramHandle for CompiledAdaptiveInterval<'_> {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledAdaptiveInterval<'_> {
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
    pub const fn request(&self) -> &AdaptiveRequest {
        &self.request
    }

    /// The id the result is sealed under.
    #[must_use]
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    /// Seal the result, replay it and report its measured interval.
    ///
    /// # Errors
    ///
    /// A result the contract cannot produce, an interval of a kind the replayed result does not
    /// carry (`route_not_supported`), cancellation, or the support gate once it is on.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<MeasuredIntervalRun, AdaptiveInferenceRefusal> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self).map_err(|refusal| AdaptiveInferenceRefusal {
                code: refusal.code(),
                detail: refusal.detail.to_owned(),
                message: refusal.to_string(),
            })?;
        }
        let measured =
            self.session.measure_result_interval(&self.request, &self.artifact_id, &scoped)?;
        Ok(MeasuredIntervalRun {
            measured,
            lowered_program_identity: self.lowered.identity(),
            calibration_route: CALIBRATION_ROUTE,
        })
    }
}
