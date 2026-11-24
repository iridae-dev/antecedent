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
use super::supplied_identity::variant;
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

/// The request's canonical encoding: its variant tag, then every field in declaration order,
/// reals by their bits. The Dirichlet shapes are ordered by category, so reordering them
/// changes the identity.
fn write_request(writer: &mut CanonicalWriter, request: &ChoiceRequest) {
    match request {
        ChoiceRequest::Posterior { look } => {
            variant(writer, "posterior").word(u64::from(*look));
        }
        ChoiceRequest::Stopping { look } => {
            variant(writer, "stopping").word(u64::from(*look));
        }
        ChoiceRequest::Hierarchical {
            look,
            shapes,
            log_concentration_mean,
            log_concentration_sd,
        } => {
            variant(writer, "hierarchical")
                .word(u64::from(*look))
                .reals(shapes)
                .real(*log_concentration_mean)
                .real(*log_concentration_sd);
        }
        ChoiceRequest::HierarchicalStopping {
            look,
            shapes,
            log_concentration_mean,
            log_concentration_sd,
        } => {
            variant(writer, "hierarchical_stopping")
                .word(u64::from(*look))
                .reals(shapes)
                .real(*log_concentration_mean)
                .real(*log_concentration_sd);
        }
        // The request enum is non-exhaustive; a variant added later is written under its own
        // tag with the fields every variant exposes until it gains its own arm here.
        other => {
            variant(writer, "unrecognized_variant")
                .word(u64::from(other.look()))
                .text(other.paradigm().as_str());
        }
    }
}

fn question_digest(request: &ChoiceRequest, artifact_id: &str) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.choice_result_interval.question.v2");
    write_request(&mut writer, request);
    writer.text(artifact_id);
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

#[cfg(test)]
mod identity_tests {
    use super::{ChoiceRequest, question_digest};

    fn hierarchical(shapes: Vec<f64>, log_concentration_mean: f64) -> ChoiceRequest {
        ChoiceRequest::Hierarchical {
            look: 3,
            shapes,
            log_concentration_mean,
            log_concentration_sd: 1.0,
        }
    }

    #[test]
    fn the_same_request_has_the_same_identity() {
        assert_eq!(
            question_digest(&hierarchical(vec![1.0, 2.0], 0.5), "a"),
            question_digest(&hierarchical(vec![1.0, 2.0], 0.5), "a")
        );
    }

    #[test]
    fn a_changed_field_or_variant_changes_the_identity() {
        let base = question_digest(&hierarchical(vec![1.0, 2.0], 0.5), "a");
        assert_ne!(base, question_digest(&hierarchical(vec![1.0, 2.0, 3.0], 0.5), "a"));
        assert_ne!(base, question_digest(&hierarchical(vec![1.0, 2.0], 0.25), "a"));
        assert_ne!(base, question_digest(&hierarchical(vec![1.0, 2.0], 0.5), "b"));
        let stopping = ChoiceRequest::HierarchicalStopping {
            look: 3,
            shapes: vec![1.0, 2.0],
            log_concentration_mean: 0.5,
            log_concentration_sd: 1.0,
        };
        assert_ne!(base, question_digest(&stopping, "a"));
        assert_ne!(
            question_digest(&ChoiceRequest::Posterior { look: 1 }, "a"),
            question_digest(&ChoiceRequest::Stopping { look: 1 }, "a")
        );
    }

    #[test]
    fn reordering_the_shapes_changes_the_identity() {
        assert_ne!(
            question_digest(&hierarchical(vec![1.0, 2.0], 0.5), "a"),
            question_digest(&hierarchical(vec![2.0, 1.0], 0.5), "a")
        );
    }

    #[test]
    fn negative_zero_is_not_zero() {
        assert_ne!(
            question_digest(&hierarchical(vec![1.0, 2.0], 0.0), "a"),
            question_digest(&hierarchical(vec![1.0, 2.0], -0.0), "a")
        );
    }
}
