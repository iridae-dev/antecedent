//! Compiled measured interval of one result of a choice study.
//!
//! [`Study::compile_choice_interval`] binds a prepared [`ChoiceInferenceSession`] (its validated
//! choice contract: options, meaning, population, respondents, prior, retained sources, looks
//! and stopping rule), one named result ([`ChoiceRequest`]) and the artifact id the result is
//! sealed under into a lowered program: the request and artifact id are the program's question,
//! the contract its source. [`CompiledChoiceInterval::execute`] runs what the deprecated
//! `ChoiceInferenceSession::measured_interval` runs: the result is sealed, replayed by a fresh
//! reader and measured by `ChoiceInferenceResultArtifact::measure_interval`, which reports the
//! equal-tailed credible interval of the first option's (population) share as a measured scalar
//! with the calibration slot of its choice cell. A study outside the measured cells still
//! reports its interval, annotated; a stopping decision has no interval and is refused.
//!
//! The coverage of these intervals is measured for the route `choice_interval` (the candidate
//! coverage records of the `choice_inference.interval_candidates` capability record);
//! [`MeasuredIntervalRun::calibration_route`] names it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{CanonicalWriter, ExecutionContext};

use super::choice_inference::{ChoiceInferenceRefusal, ChoiceInferenceSession, ChoiceRequest};
use super::compiled_measured_interval::{MeasuredIntervalProgram, MeasuredIntervalRun, bayesian};
use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};
use crate::planner::{PlanStage, PlanView};

const PLAN_ID: &str = "choice_result_interval.v1";
/// The measured route the coverage of these intervals is recorded under.
const CALIBRATION_ROUTE: &str = "choice_interval";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// One named result of a prepared choice study bound to its lowered program.
#[derive(Clone, Debug)]
pub struct CompiledChoiceInterval<'s> {
    session: &'s ChoiceInferenceSession,
    request: ChoiceRequest,
    artifact_id: String,
    lowered: Arc<LoweredProgram>,
}

fn question_digest(request: &ChoiceRequest, artifact_id: &str) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.choice_result_interval.question.v1");
    // The request's debug form spells every variant field, floats in their exact round-trip
    // form; the sealed result the consumer replays binds the same request canonically.
    writer.text(&format!("{request:?}")).text(artifact_id);
    writer.finish()
}

fn source_digest(session: &ChoiceInferenceSession) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.choice_result_interval.contract.v1");
    writer.bytes(session.contract().identity().as_bytes());
    writer.finish()
}

fn lower(
    session: &ChoiceInferenceSession,
    request: &ChoiceRequest,
    artifact_id: &str,
) -> Result<Arc<LoweredProgram>, CausalError> {
    let declaration = session.contract().declaration();
    MeasuredIntervalProgram {
        cell: ProgramCell::ChoiceResultInterval,
        plan_id: PLAN_ID,
        kind: FunctionalKind::Association,
        graph_class: "not_applicable_descriptive_choice_shares",
        question: question_digest(request, artifact_id),
        source: source_digest(session),
        source_role: EvidenceRole::Observations,
        realization: bayesian(
            "multinomial_choices_of_exchangeable_respondents",
            "declared_dirichlet_or_hierarchical_share_prior".to_owned(),
            PLAN_ID,
        )?,
        uncertainty: UncertaintyReading::Posterior,
        rows: declaration.respondents.len() as u64,
        columns: declaration.option_labels.len() as u64,
        claim: ClaimKind::Point,
    }
    .lower()
}

impl Study {
    /// Bind one named result of a prepared choice study, sealed under `artifact_id`, into a
    /// compiled program.
    ///
    /// # Errors
    ///
    /// A likelihood, prior or physical operator that cannot be declared.
    pub fn compile_choice_interval<'s>(
        session: &'s ChoiceInferenceSession,
        request: ChoiceRequest,
        artifact_id: &str,
    ) -> Result<CompiledChoiceInterval<'s>, CausalError> {
        let lowered = lower(session, &request, artifact_id)?;
        Ok(CompiledChoiceInterval {
            session,
            request,
            artifact_id: artifact_id.to_owned(),
            lowered,
        })
    }
}

impl CompiledProgramHandle for CompiledChoiceInterval<'_> {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledChoiceInterval<'_> {
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
    pub const fn request(&self) -> &ChoiceRequest {
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
    /// A result the contract cannot produce, a stopping decision, which has no interval
    /// (`route_not_supported`), cancellation, or the support gate once it is on.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<MeasuredIntervalRun, ChoiceInferenceRefusal> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self).map_err(|refusal| ChoiceInferenceRefusal {
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
