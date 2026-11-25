//! Compiled assessment and derivation of a sampled decision.
//!
//! Each entry compiles a computation a caller states over draws it supplies into a program of
//! its own family, and executes the same kernel the direct entry calls, on the same inputs:
//!
//! | entry | family | kernel |
//! |---|---|---|
//! | [`compile_decision_reliability_assessment`] | `DecisionReliabilityAssessment` | `decision_reliability::assess` |
//! | [`compile_producer_carrier_assessment`] | `DecisionReliabilityAssessment` | `decision_reliability::ProducerCarrier::assess` |
//! | [`compile_decision_reliability_derivation`] | `DecisionReliabilityDerivation` | `decision_reliability::derive_uncertainty` |
//! | [`compile_producer_carrier_derivation`] | `DecisionReliabilityDerivation` | `decision_reliability::ProducerCarrier::derive_uncertainty` |
//!
//! [`CompiledDecisionReliabilityAssessment::seal`] gates and then calls
//! [`DecisionReliabilityArtifact::new`], which recomputes the report. A refusal of the kernel is
//! returned unchanged as [`TerminalError::Kernel`]; the compiler's own refusals are
//! [`TerminalError::Refused`].
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_design::decision_reliability::{
    DecisionDraws, DecisionReliabilityArtifact, DerivedUncertainty, IntervalMultiplicity,
    IntervalRequest, JointLawStanding, ProducerCarrier, ReliabilityError, ReliabilityReport,
    ReliabilityRequest, assess, derive_uncertainty,
};
use antecedent_io::IoError;

use crate::analysis::supplied_identity::{Canonical, Declaration, Wire, variant};
use crate::analysis::supplied_terminal::{
    Reading, TerminalError, TerminalSpec, admit_compile, compiled_terminal, lower,
};
use crate::error::CausalError;
use crate::planner::lowered::{
    CarrierDemand, Composition, DecisionProgram, EvidenceRole, FunctionalKind, ProgramCell,
};

/// Joint-law standing of a producer carrier. The tags match
/// [`crate::analysis::input_decision_programs`] carrier ranking.
struct CarrierJointLaw(JointLawStanding);

impl Canonical for CarrierJointLaw {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0 {
            JointLawStanding::OneSharedLaw => {
                variant(out, "one_shared_law");
            }
            JointLawStanding::MarginalOnly { reason } => {
                variant(out, "marginal_only").text(reason);
            }
        }
    }
}

/// Level by bits and the multiplicity tag. The derivation reads both and has no collection here.
struct IntervalId(IntervalRequest);

impl Canonical for IntervalId {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.real(self.0.level);
        match self.0.multiplicity {
            IntervalMultiplicity::Marginal => {
                variant(out, "marginal");
            }
            IntervalMultiplicity::Simultaneous => {
                variant(out, "simultaneous");
            }
        }
    }
}

fn decision_layer(draws: &DecisionDraws) -> DecisionProgram {
    DecisionProgram {
        carrier: CarrierDemand::DecisionDraws,
        utility: DecisionProgram::utility_form(None),
        actions: draws.candidates.len(),
        constraints: 0,
        chance_threshold: 0.0,
        utility_unit: Some(Arc::from(draws.utility_unit.as_str())),
        contract_identity: None,
        information_value: None,
        scenario_source: Some(Arc::from("decision_draws")),
    }
}

fn declared(
    tag: &str,
    draws: &DecisionDraws,
    validation: &Option<DecisionDraws>,
    request: &ReliabilityRequest,
) -> Declaration {
    let mut question = Declaration::new(tag);
    question
        .field("draws", &draws.digest_hex())
        .field("validation", &validation.as_ref().map(DecisionDraws::digest_hex))
        .field("request", &Wire(request));
    question
}

compiled_terminal! {
    /// A compiled assessment of a sampled decision from declared draws.
    #[derive(Clone, Debug)]
    CompiledDecisionReliabilityAssessment {
        draws: DecisionDraws,
        validation: Option<DecisionDraws>,
        request: ReliabilityRequest,
    }
}

/// Compile the assessment of `draws` under `request`.
///
/// The draws and any validation draws enter by [`DecisionDraws::digest_hex`]: the assessment
/// reads each candidate set in canonical order and refuses a repeated id. A ranking request's
/// `top_k` stays in declaration order inside the request wire.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_decision_reliability_assessment(
    draws: DecisionDraws,
    validation: Option<DecisionDraws>,
    request: ReliabilityRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledDecisionReliabilityAssessment, CausalError> {
    admit_compile(ctx)?;
    let question =
        declared("antecedent.decision_reliability_assessment.v1", &draws, &validation, &request)
            .finish();
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::DecisionReliabilityAssessment,
        plan_id: "supplied.decision_reliability_assessment.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_draws",
        question,
        evaluator: "decision_reliability::assess",
        reading: Reading::MonteCarlo,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision_layer(&draws)),
        claim: Some(ClaimKind::Point),
    })?;
    Ok(CompiledDecisionReliabilityAssessment { lowered, draws, validation, request })
}

impl CompiledDecisionReliabilityAssessment {
    /// Assess the draws.
    ///
    /// # Errors
    /// The assessment's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ReliabilityReport, TerminalError<ReliabilityError>> {
        let scoped = self.enter(ctx)?;
        assess(&self.draws, self.validation.as_ref(), &self.request, &scoped)
            .map_err(TerminalError::Kernel)
    }

    /// Gate, then seal the artifact. [`DecisionReliabilityArtifact::new`] recomputes the report
    /// from the same inputs.
    ///
    /// # Errors
    /// Anything [`DecisionReliabilityArtifact::new`] returns, or the compiler's.
    pub fn seal(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DecisionReliabilityArtifact, TerminalError<IoError>> {
        let scoped = self.enter(ctx)?;
        DecisionReliabilityArtifact::new(
            self.draws.clone(),
            self.validation.clone(),
            self.request.clone(),
            &scoped,
        )
        .map_err(TerminalError::Kernel)
    }
}

compiled_terminal! {
    /// A compiled assessment of a producer carrier.
    #[derive(Clone, Debug)]
    CompiledProducerCarrierAssessment {
        carrier: ProducerCarrier,
        validation: Option<DecisionDraws>,
        request: ReliabilityRequest,
    }
}

/// Compile the assessment of `carrier` under `request`.
///
/// The declaration includes the carrier's joint-law standing, so a marginal-only carrier is a
/// different program from one under one shared law. The draws enter by
/// [`DecisionDraws::digest_hex`].
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_producer_carrier_assessment(
    carrier: ProducerCarrier,
    validation: Option<DecisionDraws>,
    request: ReliabilityRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledProducerCarrierAssessment, CausalError> {
    admit_compile(ctx)?;
    let standing = carrier.joint_law();
    let mut question = declared(
        "antecedent.producer_carrier_assessment.v1",
        carrier.draws(),
        &validation,
        &request,
    );
    question.field("joint_law", &CarrierJointLaw(standing));
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::DecisionReliabilityAssessment,
        plan_id: "supplied.producer_carrier_assessment.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_draws",
        question: question.finish(),
        evaluator: "decision_reliability::ProducerCarrier::assess",
        reading: Reading::MonteCarlo,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision_layer(carrier.draws())),
        claim: Some(ClaimKind::Point),
    })?;
    Ok(CompiledProducerCarrierAssessment { lowered, carrier, validation, request })
}

impl CompiledProducerCarrierAssessment {
    /// Assess the carrier: its standing checks, then the assessment kernel on its draws.
    ///
    /// # Errors
    /// A ranking request on a marginal-only carrier, a functional the producer does not
    /// support, the assessment's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ReliabilityReport, TerminalError<ReliabilityError>> {
        let scoped = self.enter(ctx)?;
        self.carrier
            .assess(self.validation.as_ref(), &self.request, &scoped)
            .map_err(TerminalError::Kernel)
    }
}

compiled_terminal! {
    /// A compiled derivation of the three uncertainty components from declared draws.
    #[derive(Clone, Debug)]
    CompiledDecisionReliabilityDerivation {
        draws: DecisionDraws,
        validation: Option<DecisionDraws>,
        request: ReliabilityRequest,
        interval: IntervalRequest,
    }
}

/// Compile the derivation of `draws` under `request` and `interval`.
///
/// The result is the whole [`DerivedUncertainty`]: the numerical interval, the optimality
/// confidence set and the selection-corrected interval. The draws enter by
/// [`DecisionDraws::digest_hex`]. `interval` is the level by bits and the multiplicity tag.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_decision_reliability_derivation(
    draws: DecisionDraws,
    validation: Option<DecisionDraws>,
    request: ReliabilityRequest,
    interval: IntervalRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledDecisionReliabilityDerivation, CausalError> {
    admit_compile(ctx)?;
    let mut question =
        declared("antecedent.decision_reliability_derivation.v1", &draws, &validation, &request);
    question.field("interval", &IntervalId(interval));
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::DecisionReliabilityDerivation,
        plan_id: "supplied.decision_reliability_derivation.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_draws",
        question: question.finish(),
        evaluator: "decision_reliability::derive_uncertainty",
        reading: Reading::MonteCarlo,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision_layer(&draws)),
        claim: Some(ClaimKind::Point),
    })?;
    Ok(CompiledDecisionReliabilityDerivation { lowered, draws, validation, request, interval })
}

impl CompiledDecisionReliabilityDerivation {
    /// Derive the three components.
    ///
    /// # Errors
    /// The derivation's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DerivedUncertainty, TerminalError<ReliabilityError>> {
        let scoped = self.enter(ctx)?;
        derive_uncertainty(
            &self.draws,
            self.validation.as_ref(),
            &self.request,
            &self.interval,
            &scoped,
        )
        .map_err(TerminalError::Kernel)
    }
}

compiled_terminal! {
    /// A compiled derivation of the three uncertainty components of a producer carrier.
    #[derive(Clone, Debug)]
    CompiledProducerCarrierDerivation {
        carrier: ProducerCarrier,
        validation: Option<DecisionDraws>,
        request: ReliabilityRequest,
        interval: IntervalRequest,
    }
}

/// Compile the derivation of `carrier` under `request` and `interval`.
///
/// The declaration includes the carrier's joint-law standing. The result is the whole
/// [`DerivedUncertainty`].
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_producer_carrier_derivation(
    carrier: ProducerCarrier,
    validation: Option<DecisionDraws>,
    request: ReliabilityRequest,
    interval: IntervalRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledProducerCarrierDerivation, CausalError> {
    admit_compile(ctx)?;
    let standing = carrier.joint_law();
    let mut question = declared(
        "antecedent.producer_carrier_derivation.v1",
        carrier.draws(),
        &validation,
        &request,
    );
    question
        .field("interval", &IntervalId(interval))
        .field("joint_law", &CarrierJointLaw(standing));
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::DecisionReliabilityDerivation,
        plan_id: "supplied.producer_carrier_derivation.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_draws",
        question: question.finish(),
        evaluator: "decision_reliability::ProducerCarrier::derive_uncertainty",
        reading: Reading::MonteCarlo,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision_layer(carrier.draws())),
        claim: Some(ClaimKind::Point),
    })?;
    Ok(CompiledProducerCarrierDerivation { lowered, carrier, validation, request, interval })
}

impl CompiledProducerCarrierDerivation {
    /// Derive the carrier's three components: its standing checks, then the derivation kernel.
    ///
    /// # Errors
    /// A ranking request on a marginal-only carrier, a functional the producer does not
    /// support, the derivation's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DerivedUncertainty, TerminalError<ReliabilityError>> {
        let scoped = self.enter(ctx)?;
        self.carrier
            .derive_uncertainty(self.validation.as_ref(), &self.request, &self.interval, &scoped)
            .map_err(TerminalError::Kernel)
    }
}

#[cfg(test)]
mod site_tests {
    use super::*;
    use crate::identity_laws::{bit_exact, ordered_laws, set_laws};

    use antecedent_design::decision_reliability::{
        BatchPlan, CandidateCoupling, CandidateDraws, CandidateOrigin, DrawSegment, ProducerOutput,
        RankingRequest, RolloutNoise, RolloutUtilityDraws, SourceParadigm, StreamId, StreamPurpose,
        WithinStreamDependence, reach_carrier,
    };

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(7)
    }

    fn draws_of(columns: &[(&str, [f64; 4])]) -> DecisionDraws {
        let shared = DrawSegment::whole(
            StreamId::new(20_240, StreamPurpose::Selection, 0, "test_stream"),
            4,
        );
        DecisionDraws {
            utility_unit: "utility".into(),
            producer: "test".into(),
            source: SourceParadigm::KnownLaw,
            coupling: CandidateCoupling::SharedSimulation,
            within_stream: WithinStreamDependence::Independent,
            weights: None,
            repeated_evaluations: 0,
            origin: CandidateOrigin::Enumerated,
            search_record: None,
            candidates: columns
                .iter()
                .map(|(id, utilities)| CandidateDraws {
                    id: (*id).to_owned(),
                    utilities: utilities.to_vec(),
                    segment: shared.clone(),
                })
                .collect(),
        }
    }

    fn request(top_k: Vec<u32>) -> ReliabilityRequest {
        ReliabilityRequest {
            ranking: Some(RankingRequest {
                baseline: None,
                top_k,
                batch_plan: BatchPlan::SquareRoot,
            }),
            ..ReliabilityRequest::expected_utility()
        }
    }

    fn accepted_draws(columns: &[(&str, [f64; 4])]) -> Option<DecisionDraws> {
        let draws = draws_of(columns);
        draws.validate().ok()?;
        Some(draws)
    }

    const CANDIDATES: [(&str, [f64; 4]); 3] =
        [("A", [3.0, 3.0, 2.0, 2.0]), ("B", [2.0, 1.0, 3.0, 2.0]), ("C", [1.0, 2.0, 1.0, 2.0])];

    #[test]
    fn site_decision_reliability_programs_assessment_candidates_are_a_set_and_top_k_ordered() {
        let context = ctx();
        let identity = |columns: &[(&str, [f64; 4])],
                        validation: Option<&[(&str, [f64; 4])]>,
                        top_k: &[u32]| {
            let draws = accepted_draws(columns)?;
            let validation = match validation {
                Some(columns) => Some(accepted_draws(columns)?),
                None => None,
            };
            Some(
                compile_decision_reliability_assessment(
                    draws,
                    validation,
                    request(top_k.to_vec()),
                    &context,
                )
                .expect("compiles")
                .identity(),
            )
        };
        set_laws(&CANDIDATES, |columns| identity(columns, None, &[1]));
        set_laws(&CANDIDATES, |columns| identity(&CANDIDATES, Some(columns), &[1]));
        ordered_laws(&[1_u32, 2, 3], |top_k| identity(&CANDIDATES, None, top_k));
        bit_exact(|bit| {
            let mut columns = CANDIDATES;
            columns[0].1[0] = bit;
            identity(&columns, None, &[1])
        });
    }

    fn rollout(columns: &[(&str, [f64; 4])], noise: RolloutNoise) -> Option<ProducerCarrier> {
        reach_carrier(ProducerOutput::Rollout(RolloutUtilityDraws {
            model_identity: "model".into(),
            stream_scheme: "keyed".into(),
            seed: 3,
            utility_unit: "utility".into(),
            purpose: StreamPurpose::Selection,
            replicate: 0,
            noise,
            policies: columns
                .iter()
                .map(|(id, utilities)| ((*id).to_owned(), utilities.to_vec()))
                .collect(),
            reported_means: Vec::new(),
            max_standard_errors: 4.0,
        }))
        .ok()
    }

    #[test]
    fn site_decision_reliability_programs_carrier_assessment_includes_joint_law() {
        let context = ctx();
        let identity = |columns: &[(&str, [f64; 4])], top_k: &[u32]| {
            let carrier = rollout(columns, RolloutNoise::StructuralExogenous)?;
            Some(
                compile_producer_carrier_assessment(
                    carrier,
                    None,
                    request(top_k.to_vec()),
                    &context,
                )
                .expect("compiles")
                .identity(),
            )
        };
        set_laws(&CANDIDATES, |columns| identity(columns, &[1]));
        ordered_laws(&[1_u32, 2, 3], |top_k| identity(&CANDIDATES, top_k));
        let shared = rollout(&CANDIDATES, RolloutNoise::StructuralExogenous).unwrap();
        let marginal = rollout(&CANDIDATES, RolloutNoise::ImplementationKeyed).unwrap();
        assert_eq!(shared.draws().digest_hex(), marginal.draws().digest_hex());
        let asked = request(vec![1]);
        assert_ne!(
            compile_producer_carrier_assessment(shared, None, asked.clone(), &context)
                .unwrap()
                .identity(),
            compile_producer_carrier_assessment(marginal, None, asked, &context)
                .unwrap()
                .identity(),
        );
    }

    #[test]
    fn site_decision_reliability_programs_derivation_candidates_are_a_set_and_top_k_ordered() {
        let context = ctx();
        let identity = |columns: &[(&str, [f64; 4])], top_k: &[u32], level: f64| {
            let draws = accepted_draws(columns)?;
            Some(
                compile_decision_reliability_derivation(
                    draws,
                    None,
                    request(top_k.to_vec()),
                    IntervalRequest::marginal(level),
                    &context,
                )
                .expect("compiles")
                .identity(),
            )
        };
        set_laws(&CANDIDATES, |columns| identity(columns, &[1], 0.95));
        ordered_laws(&[1_u32, 2, 3], |top_k| identity(&CANDIDATES, top_k, 0.95));
        bit_exact(|level| identity(&CANDIDATES, &[1], level));
        bit_exact(|bit| {
            let mut columns = CANDIDATES;
            columns[0].1[0] = bit;
            identity(&columns, &[1], 0.95)
        });
    }

    #[test]
    fn site_decision_reliability_programs_carrier_derivation_includes_joint_law() {
        let context = ctx();
        let identity = |columns: &[(&str, [f64; 4])]| {
            let carrier = rollout(columns, RolloutNoise::StructuralExogenous)?;
            Some(
                compile_producer_carrier_derivation(
                    carrier,
                    None,
                    request(vec![1]),
                    IntervalRequest::marginal(0.95),
                    &context,
                )
                .expect("compiles")
                .identity(),
            )
        };
        set_laws(&CANDIDATES, identity);
        let shared = rollout(&CANDIDATES, RolloutNoise::StructuralExogenous).unwrap();
        let marginal = rollout(&CANDIDATES, RolloutNoise::ImplementationKeyed).unwrap();
        assert_eq!(shared.draws().digest_hex(), marginal.draws().digest_hex());
        let interval = IntervalRequest::marginal(0.95);
        assert_ne!(
            compile_producer_carrier_derivation(shared, None, request(vec![1]), interval, &context)
                .unwrap()
                .identity(),
            compile_producer_carrier_derivation(
                marginal,
                None,
                request(vec![1]),
                interval,
                &context,
            )
            .unwrap()
            .identity(),
        );
    }
}
