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
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};

use super::adaptive_inference::{
    AdaptiveInferenceRefusal, AdaptiveInferenceSession, AdaptiveRequest, InferenceParadigm,
};
use super::compiled_measured_interval::{
    MeasuredIntervalProgram, MeasuredIntervalRun, bayesian, frequentist,
};
use super::program_handle::{CompiledProgramHandle, gate_program_when};
use super::supplied_identity::{unencoded_variant, variant};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

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

/// The request's canonical encoding: its variant tag, then every field in declaration order,
/// reals by their bits. The arms of a contrast are ordered (plus then minus), so swapping them
/// changes the identity.
fn write_request(
    writer: &mut CanonicalWriter,
    request: &AdaptiveRequest,
) -> Result<(), CausalError> {
    match request {
        AdaptiveRequest::BayesianPosterior { look, prior_alpha, prior_beta } => {
            variant(writer, "bayesian_posterior")
                .word(u64::from(*look))
                .real(*prior_alpha)
                .real(*prior_beta);
        }
        AdaptiveRequest::PosteriorStopping { look, prior_alpha, prior_beta } => {
            variant(writer, "posterior_stopping")
                .word(u64::from(*look))
                .real(*prior_alpha)
                .real(*prior_beta);
        }
        AdaptiveRequest::AdaptiveWeighting {
            look,
            plus_arm,
            minus_arm,
            scheme,
            prior_mean,
            prior_strength,
            size_weighted,
        } => {
            variant(writer, "adaptive_weighting")
                .word(u64::from(*look))
                .word(u64::from(*plus_arm))
                .opt_word(minus_arm.map(u64::from))
                .text(scheme)
                .real(*prior_mean)
                .real(*prior_strength)
                .word(u64::from(*size_weighted));
        }
        AdaptiveRequest::ConfidenceSequence { look, alpha, tuning_samples, contrast } => {
            variant(writer, "confidence_sequence")
                .word(u64::from(*look))
                .real(*alpha)
                .real(*tuning_samples);
            match contrast {
                Some((plus, minus)) => {
                    writer.word(1).word(u64::from(*plus)).word(u64::from(*minus));
                }
                None => {
                    writer.word(0);
                }
            }
        }
        AdaptiveRequest::GroupSequential { look, control_arm, treatment_arm } => {
            variant(writer, "group_sequential")
                .word(u64::from(*look))
                .word(u64::from(*control_arm))
                .word(u64::from(*treatment_arm));
        }
        // The request enum is non-exhaustive: a variant added later refuses to compile until
        // it gains its own arm here, rather than sharing a fallback identity with another.
        _ => return Err(unencoded_variant(PLAN_ID, "adaptive result request")),
    }
    Ok(())
}

fn question_digest(request: &AdaptiveRequest, artifact_id: &str) -> Result<[u8; 32], CausalError> {
    let mut writer = CanonicalWriter::new("antecedent.adaptive_result_interval.question.v2");
    write_request(&mut writer, request)?;
    writer.text(artifact_id);
    Ok(writer.finish())
}

/// The source: the identity of the validated adaptive contract the result is replayed from.
fn source_digest(contract: &SemanticDigest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.adaptive_result_interval.contract.v1");
    writer.bytes(contract.as_bytes());
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
        question: question_digest(request, artifact_id)?,
        source: source_digest(&session.contract().identity()),
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
        CompiledProgramHandle::plan_view(self)
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
        gate_program_when(GATE_ON_UNIT_SUPPORT, &scoped, self).map_err(|refusal| {
            AdaptiveInferenceRefusal {
                code: refusal.code(),
                detail: refusal.detail.to_owned(),
                message: refusal.to_string(),
            }
        })?;
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
    use antecedent_core::SemanticDigest;

    use super::{AdaptiveRequest, question_digest, source_digest};
    use crate::identity_laws::{
        ADVERSARIAL_TEXT, all_distinct, bit_exact, injective, ordered_laws,
    };

    #[test]
    fn site_compiled_adaptive_interval_question_orders_the_arms_and_frames_the_request() {
        ordered_laws(&[1_u32, 0], |arms| {
            question_digest(&sequence(Some((arms[0], arms[1]))), "a").ok()
        });
        ordered_laws(&[1_u32, 0], |arms| {
            question_digest(&weighting(arms[0], Some(arms[1]), 0.5), "a").ok()
        });
        bit_exact(|prior_mean| question_digest(&weighting(1, Some(0), prior_mean), "a").ok());
        injective(
            &ADVERSARIAL_TEXT,
            |parts| *parts,
            |(scheme, artifact_id)| {
                let request = AdaptiveRequest::AdaptiveWeighting {
                    look: 2,
                    plus_arm: 1,
                    minus_arm: None,
                    scheme: (*scheme).to_owned(),
                    prior_mean: 0.5,
                    prior_strength: 1.0,
                    size_weighted: false,
                };
                question_digest(&request, artifact_id).ok()
            },
        );
        let variants = [
            AdaptiveRequest::BayesianPosterior { look: 2, prior_alpha: 1.0, prior_beta: 1.0 },
            AdaptiveRequest::PosteriorStopping { look: 2, prior_alpha: 1.0, prior_beta: 1.0 },
            AdaptiveRequest::GroupSequential { look: 2, control_arm: 0, treatment_arm: 1 },
            AdaptiveRequest::GroupSequential { look: 2, control_arm: 1, treatment_arm: 0 },
            sequence(None),
            sequence(Some((0, 1))),
            weighting(1, None, 0.5),
        ];
        all_distinct(
            &variants
                .iter()
                .map(|request| question_digest(request, "a").expect("encoded"))
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn site_compiled_adaptive_interval_source_binds_the_contract_identity() {
        all_distinct(&[
            source_digest(&SemanticDigest::from_bytes([0; 32])),
            source_digest(&SemanticDigest::from_bytes([1; 32])),
        ]);
    }

    fn weighting(plus_arm: u32, minus_arm: Option<u32>, prior_mean: f64) -> AdaptiveRequest {
        AdaptiveRequest::AdaptiveWeighting {
            look: 2,
            plus_arm,
            minus_arm,
            scheme: "uniform".to_owned(),
            prior_mean,
            prior_strength: 1.0,
            size_weighted: false,
        }
    }

    fn sequence(contrast: Option<(u32, u32)>) -> AdaptiveRequest {
        AdaptiveRequest::ConfidenceSequence {
            look: 1,
            alpha: 0.05,
            tuning_samples: 100.0,
            contrast,
        }
    }

    #[test]
    fn the_same_request_has_the_same_identity() {
        assert_eq!(
            question_digest(&weighting(1, Some(0), 0.5), "a").unwrap(),
            question_digest(&weighting(1, Some(0), 0.5), "a").unwrap()
        );
    }

    #[test]
    fn a_changed_field_or_artifact_id_changes_the_identity() {
        let base = question_digest(&weighting(1, Some(0), 0.5), "a").unwrap();
        assert_ne!(base, question_digest(&weighting(1, None, 0.5), "a").unwrap());
        assert_ne!(base, question_digest(&weighting(1, Some(0), 0.25), "a").unwrap());
        assert_ne!(base, question_digest(&weighting(1, Some(0), 0.5), "b").unwrap());
        let posterior =
            AdaptiveRequest::BayesianPosterior { look: 2, prior_alpha: 1.0, prior_beta: 1.0 };
        let stopping =
            AdaptiveRequest::PosteriorStopping { look: 2, prior_alpha: 1.0, prior_beta: 1.0 };
        assert_ne!(
            question_digest(&posterior, "a").unwrap(),
            question_digest(&stopping, "a").unwrap()
        );
    }

    #[test]
    fn swapping_the_contrast_arms_changes_the_identity() {
        assert_ne!(
            question_digest(&sequence(Some((1, 0))), "a").unwrap(),
            question_digest(&sequence(Some((0, 1))), "a").unwrap()
        );
        assert_ne!(
            question_digest(&weighting(1, Some(0), 0.5), "a").unwrap(),
            question_digest(&weighting(0, Some(1), 0.5), "a").unwrap()
        );
    }

    #[test]
    fn negative_zero_is_not_zero() {
        assert_ne!(
            question_digest(&weighting(1, Some(0), 0.0), "a").unwrap(),
            question_digest(&weighting(1, Some(0), -0.0), "a").unwrap()
        );
    }
}
