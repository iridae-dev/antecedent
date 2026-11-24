//! Compiled decisions, rankings and plans over declared inputs.
//!
//! Each entry compiles a computation a caller states over evidence it supplies into a program
//! of its own family, and executes the same kernel the direct entry calls, on the same inputs:
//!
//! | entry | family | kernel |
//! |---|---|---|
//! | [`compile_supported_input_decision`] | `SupportedInputDecision` | `composition_boundary::evaluate_with_support` |
//! | [`compile_input_functional`] | `SupportedInputDecision` | `composition_boundary::evaluate_functional_on_input` |
//! | [`compile_sensitivity_decision`] | `SensitivityDecision` | `sensitivity_decision::evaluate_sensitivity_decision` |
//! | [`compile_inverse_query`] | `InverseQueryDecision` | `inverse_query::evaluate_inverse_query` |
//! | [`compile_finite_enumeration_baseline`] | `InverseQueryDecision` | `inverse_query::finite_enumeration_baseline` |
//! | [`compile_decision_draws_ranking`] | `DecisionDrawsRanking` | `decision_reliability::rank` |
//! | [`compile_prior_signal_adaptation`] | `PriorSignalAdaptation` | `prior_signal::adapt_prior_to_signal` |
//! | [`compile_evidence_obligations`] | `EvidenceObligations` | `obligation_adapters::identification_failure_obligations` |
//! | [`compile_finite_action_decision`] | `FiniteActionDecision` | `decision::evaluate_decision` |
//! | [`compile_adaptive_policy_search`] | `AdaptivePolicySearch` | `PreparedAdaptivePolicySearch::execute` |
//!
//! A refusal of the kernel is returned unchanged as [`TerminalError::Kernel`]; the compiler's
//! own refusals (re-entry, cancellation, a release build's support gate) are
//! [`TerminalError::Refused`].
//!
//! A declared input that is a live callback (a utility or constraint of a finite decision
//! problem, an external signal provider, the response source of a policy search) is a runtime
//! binding: the program identity records that one is bound, not what it computes.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_contract::evidence_obligation::EvidenceObligation;
use antecedent_core::{CanonicalWriter, ExecutionContext, SupportStatus, VariableId};
use antecedent_design::DecisionFunctional;
use antecedent_design::adaptive_policy_search::{
    AdaptivePolicySearchError, AdaptivePolicySearchResult, PolicyRolloutSource,
    PreparedAdaptivePolicySearch,
};
use antecedent_design::composition_boundary::{
    BoundaryError, DecisionInput, SupportPolicy, SupportedDecision, evaluate_functional_on_input,
    evaluate_with_support,
};
use antecedent_design::decision_contract::DecisionContract;
use antecedent_design::decision_eval::FunctionalValue;
use antecedent_design::decision_reliability::{
    DecisionDraws, RankingReport, RankingRequest, ReliabilityError, rank,
};
use antecedent_design::inverse_query::{
    Comparison, EnumeratedForwardValue, FeasibilityStatus, ForwardEvidence, InverseQuery,
    InverseQueryError, InverseResult, evaluate_inverse_query, finite_enumeration_baseline,
};
use antecedent_design::obligation_adapters::identification_failure_obligations;
use antecedent_design::prior_signal::{
    CandidateSignalPrior, CandidateSignalSource, PriorSignalError, PriorSignalRequest,
    adapt_prior_to_signal,
};
use antecedent_design::sensitivity_decision::{
    SensitivityArtifact, SensitivityDecisionError, SensitivityDecisionResult,
    SensitivityDecisionSpec, evaluate_sensitivity_decision,
};
use antecedent_design::{
    DecisionEvaluation, DecisionProblem, DesignError, RepairError, SensitivityFinding,
    evaluate_decision,
};
use antecedent_graph::Admg;
use antecedent_identify::IdentificationResult;
use antecedent_io_core::{PriorCatalog, TargetDesign};

pub use crate::analysis::supplied_terminal::TerminalError;

use crate::analysis::program_handle::CompiledProgramHandle;
use crate::analysis::supplied_terminal::{
    Reading, TerminalSpec, admit_compile, compiled_terminal, declaration_digest, lower,
};
use crate::error::CausalError;
use crate::planner::lowered::{
    CarrierDemand, Composition, DecisionProgram, EvidenceRole, FunctionalKind, LoweredProgram,
    ProgramCell,
};

/// The decision layer of a contract's program: its actions, constraints, strictest chance
/// threshold, utility unit and identity (absent when the contract does not validate; the
/// kernel then refuses it at execution with its own error).
fn contract_program(
    contract: &DecisionContract,
    carrier: CarrierDemand,
    source: Option<&str>,
) -> DecisionProgram {
    DecisionProgram {
        carrier,
        utility: DecisionProgram::utility_form(None),
        actions: contract.actions.len(),
        constraints: contract.constraints.len(),
        chance_threshold: contract
            .constraints
            .iter()
            .map(|constraint| constraint.min_probability)
            .fold(0.0_f64, f64::max),
        utility_unit: Some(Arc::from(contract.utility_units.as_str())),
        contract_identity: contract.identity().ok().map(Arc::from),
        information_value: None,
        scenario_source: source.map(Arc::from),
    }
}

// --- decisions over several supported inputs ------------------------------------------------

compiled_terminal! {
    /// A compiled decision over several inputs, with support decided per action.
    #[derive(Clone, Debug)]
    CompiledSupportedInputDecision {
        contract: DecisionContract,
        inputs: Vec<DecisionInput>,
        policy: SupportPolicy,
    }
}

/// Compile the decision of `contract` over `inputs` under the support `policy`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_supported_input_decision(
    contract: DecisionContract,
    inputs: Vec<DecisionInput>,
    policy: SupportPolicy,
    ctx: &ExecutionContext,
) -> Result<CompiledSupportedInputDecision, CausalError> {
    admit_compile(ctx)?;
    let question = declaration_digest(
        "antecedent.supported_input_decision.v1",
        &[&contract, &inputs, &policy],
    );
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::SupportedInputDecision,
        plan_id: "supported_input_decision.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_inputs",
        question,
        evaluator: "composition_boundary.evaluate_with_support",
        reading: Reading::Predictive,
        role: EvidenceRole::ChainSource,
        composition: Composition::Pooled { sources: inputs.len() },
        decision: Some(contract_program(&contract, CarrierDemand::PredictiveLaw, None)),
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledSupportedInputDecision { lowered, contract, inputs, policy })
}

impl CompiledSupportedInputDecision {
    /// Decide over the inputs.
    ///
    /// # Errors
    /// The composition boundary's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SupportedDecision, TerminalError<BoundaryError>> {
        let _scoped = self.enter(ctx)?;
        evaluate_with_support(&self.contract, &self.inputs, &self.policy)
            .map_err(TerminalError::Kernel)
    }
}

compiled_terminal! {
    /// A compiled functional of one action's utility from one supported input.
    #[derive(Clone, Debug)]
    CompiledInputFunctional {
        contract: DecisionContract,
        action_id: String,
        functional: DecisionFunctional,
        input: DecisionInput,
        weakest: SupportStatus,
    }
}

/// Compile `functional` of `action_id`'s utility from `input`, refusing a coordinate below
/// `weakest`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_input_functional(
    contract: DecisionContract,
    action_id: &str,
    functional: DecisionFunctional,
    input: DecisionInput,
    weakest: SupportStatus,
    ctx: &ExecutionContext,
) -> Result<CompiledInputFunctional, CausalError> {
    admit_compile(ctx)?;
    let question = declaration_digest(
        "antecedent.input_functional.v1",
        &[&contract, &action_id, &functional, &input, &weakest],
    );
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::SupportedInputDecision,
        plan_id: "input_functional.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_inputs",
        question,
        evaluator: "composition_boundary.evaluate_functional_on_input",
        reading: Reading::Predictive,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(contract_program(&contract, CarrierDemand::PredictiveLaw, None)),
        claim: Some(ClaimKind::Point),
    })?;
    Ok(CompiledInputFunctional {
        lowered,
        contract,
        action_id: action_id.to_owned(),
        functional,
        input,
        weakest,
    })
}

impl CompiledInputFunctional {
    /// Compute the functional.
    ///
    /// # Errors
    /// The composition boundary's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<FunctionalValue, TerminalError<BoundaryError>> {
        let _scoped = self.enter(ctx)?;
        evaluate_functional_on_input(
            &self.contract,
            &self.action_id,
            self.functional.clone(),
            &self.input,
            self.weakest,
        )
        .map_err(TerminalError::Kernel)
    }
}

// --- decisions over an assumption-sensitivity surface --------------------------------------

compiled_terminal! {
    /// A compiled decision over the surface of an assumption-sensitivity artifact.
    #[derive(Clone, Debug)]
    CompiledSensitivityDecision {
        contract: DecisionContract,
        artifact: SensitivityArtifact,
        spec: SensitivityDecisionSpec,
    }
}

/// Compile the decision of `contract` over the surface of `artifact`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_sensitivity_decision(
    contract: DecisionContract,
    artifact: SensitivityArtifact,
    spec: SensitivityDecisionSpec,
    ctx: &ExecutionContext,
) -> Result<CompiledSensitivityDecision, CausalError> {
    admit_compile(ctx)?;
    let question =
        declaration_digest("antecedent.sensitivity_decision.v1", &[&contract, &artifact, &spec]);
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::SensitivityDecision,
        plan_id: "sensitivity_decision.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_assumption_surface",
        question,
        evaluator: "sensitivity_decision.evaluate_sensitivity_decision",
        reading: Reading::Envelope,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(contract_program(
            &contract,
            CarrierDemand::IdentifiedSet,
            Some("assumption_surface"),
        )),
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledSensitivityDecision { lowered, contract, artifact, spec })
}

impl CompiledSensitivityDecision {
    /// Decide over the surface.
    ///
    /// # Errors
    /// The sensitivity decision's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SensitivityDecisionResult, TerminalError<SensitivityDecisionError>> {
        let _scoped = self.enter(ctx)?;
        evaluate_sensitivity_decision(&self.contract, &self.artifact, &self.spec)
            .map_err(TerminalError::Kernel)
    }
}

// --- inverse decision queries ---------------------------------------------------------------

compiled_terminal! {
    /// A compiled inverse decision query over forward evidence.
    #[derive(Clone, Debug)]
    CompiledInverseQuery {
        query: InverseQuery,
        evidence: ForwardEvidence,
    }
}

/// Compile the inverse `query` over `evidence`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_inverse_query(
    query: InverseQuery,
    evidence: ForwardEvidence,
    ctx: &ExecutionContext,
) -> Result<CompiledInverseQuery, CausalError> {
    admit_compile(ctx)?;
    let question = declaration_digest("antecedent.inverse_query.v1", &[&query, &evidence]);
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::InverseQueryDecision,
        plan_id: "inverse_query.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_forward_evidence",
        question,
        evaluator: "inverse_query.evaluate_inverse_query",
        reading: Reading::Predictive,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(contract_program(&query.contract, CarrierDemand::PredictiveLaw, None)),
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledInverseQuery { lowered, query, evidence })
}

impl CompiledInverseQuery {
    /// Answer the inverse query.
    ///
    /// # Errors
    /// The inverse query's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<InverseResult, TerminalError<InverseQueryError>> {
        let _scoped = self.enter(ctx)?;
        evaluate_inverse_query(&self.query, &self.evidence).map_err(TerminalError::Kernel)
    }
}

compiled_terminal! {
    /// The compiled finite-enumeration baseline of an inverse mean query.
    #[derive(Clone, Debug)]
    CompiledFiniteEnumerationBaseline {
        points: Vec<EnumeratedForwardValue>,
        target: f64,
        comparison: Comparison,
        tolerance: f64,
    }
}

/// Compile the classification of each enumerated action by its forward mean against `target`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_finite_enumeration_baseline(
    points: Vec<EnumeratedForwardValue>,
    target: f64,
    comparison: Comparison,
    tolerance: f64,
    ctx: &ExecutionContext,
) -> Result<CompiledFiniteEnumerationBaseline, CausalError> {
    admit_compile(ctx)?;
    let question = declaration_digest(
        "antecedent.finite_enumeration_baseline.v1",
        &[&points, &target.to_bits(), &comparison, &tolerance.to_bits()],
    );
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::InverseQueryDecision,
        plan_id: "finite_enumeration_baseline.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_forward_means",
        question,
        evaluator: "inverse_query.finite_enumeration_baseline",
        reading: Reading::Exact,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: None,
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledFiniteEnumerationBaseline { lowered, points, target, comparison, tolerance })
}

impl CompiledFiniteEnumerationBaseline {
    /// Classify each enumerated action.
    ///
    /// # Errors
    /// The baseline's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, FeasibilityStatus)>, TerminalError<InverseQueryError>> {
        let _scoped = self.enter(ctx)?;
        finite_enumeration_baseline(&self.points, self.target, self.comparison, self.tolerance)
            .map_err(TerminalError::Kernel)
    }
}

// --- ranking statements over decision draws ------------------------------------------------

compiled_terminal! {
    /// Compiled ranking statements over aligned decision draws.
    #[derive(Clone, Debug)]
    CompiledDecisionDrawsRanking {
        draws: DecisionDraws,
        request: RankingRequest,
    }
}

/// Compile the ranking statements of `draws` under `request`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_decision_draws_ranking(
    draws: DecisionDraws,
    request: RankingRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledDecisionDrawsRanking, CausalError> {
    admit_compile(ctx)?;
    let question = declaration_digest("antecedent.decision_draws_ranking.v1", &[&draws, &request]);
    let decision = DecisionProgram {
        carrier: CarrierDemand::DecisionDraws,
        utility: DecisionProgram::utility_form(None),
        actions: draws.candidates.len(),
        constraints: 0,
        chance_threshold: 0.0,
        utility_unit: Some(Arc::from(draws.utility_unit.as_str())),
        contract_identity: None,
        information_value: None,
        scenario_source: Some(Arc::from("decision_draws")),
    };
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::DecisionDrawsRanking,
        plan_id: "decision_draws_ranking.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_draws",
        question,
        evaluator: "decision_reliability.rank",
        reading: Reading::MonteCarlo,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision),
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledDecisionDrawsRanking { lowered, draws, request })
}

impl CompiledDecisionDrawsRanking {
    /// State the ranking.
    ///
    /// # Errors
    /// The ranking's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<RankingReport, TerminalError<ReliabilityError>> {
        let scoped = self.enter(ctx)?;
        rank(&self.draws, &self.request, &scoped).map_err(TerminalError::Kernel)
    }
}

// --- a prior adapted to a future signal ----------------------------------------------------

compiled_terminal! {
    /// A compiled adaptation of catalogued posterior sources to a future candidate signal.
    #[derive(Clone)]
    CompiledPriorSignalAdaptation {
        catalog: PriorCatalog,
        target: TargetDesign,
        request: PriorSignalRequest,
    }
}

impl std::fmt::Debug for CompiledPriorSignalAdaptation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompiledPriorSignalAdaptation")
            .field("identity", &self.lowered.identity())
            .field("catalog", &self.catalog)
            .field("target", &self.target)
            .field("prior_id", &self.request.prior_id)
            .finish_non_exhaustive()
    }
}

/// The digest of a prior-signal declaration. A signal provider is a runtime binding.
fn prior_signal_digest(
    catalog: &PriorCatalog,
    target: &TargetDesign,
    request: &PriorSignalRequest,
) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.prior_signal_adaptation.v1");
    writer
        .text(&format!("{catalog:?}"))
        .text(&format!("{target:?}"))
        .text(&request.prior_id)
        .text(&format!("{:?}", request.sources))
        .text(&request.target_population)
        .text(&format!("{:?}", request.transport))
        .texts(&request.candidate_observation_ids)
        .word(request.resolution as u64);
    match &request.signal {
        CandidateSignalSource::Native(family) => {
            writer.text("native").text(&format!("{family:?}"));
        }
        CandidateSignalSource::Provider(_) => {
            writer.text("provider");
            writer.note_runtime_callback();
        }
        CandidateSignalSource::CoefficientPriorOnly => {
            writer.text("coefficient_prior_only");
        }
        CandidateSignalSource::UnlockListOnly(unlocks) => {
            writer.text("unlock_list_only").texts(unlocks);
        }
    }
    writer.finish()
}

/// Compile the adaptation of `catalog`'s sources named in `request` to its future signal.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_prior_signal_adaptation(
    catalog: PriorCatalog,
    target: TargetDesign,
    request: PriorSignalRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledPriorSignalAdaptation, CausalError> {
    admit_compile(ctx)?;
    let question = prior_signal_digest(&catalog, &target, &request);
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::PriorSignalAdaptation,
        plan_id: "prior_signal_adaptation.v1",
        functional: FunctionalKind::PropagatedChain,
        graph_class: "catalogued_posterior_sources",
        question,
        evaluator: "prior_signal.adapt_prior_to_signal",
        reading: Reading::Posterior,
        role: EvidenceRole::PriorArtifact,
        composition: Composition::Pooled { sources: request.sources.len() },
        decision: None,
        claim: None,
    })?;
    Ok(CompiledPriorSignalAdaptation { lowered, catalog, target, request })
}

impl CompiledPriorSignalAdaptation {
    /// Adapt the sources to the signal.
    ///
    /// # Errors
    /// The adapter's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CandidateSignalPrior, TerminalError<PriorSignalError>> {
        let _scoped = self.enter(ctx)?;
        adapt_prior_to_signal(&self.catalog, &self.target, &self.request)
            .map_err(TerminalError::Kernel)
    }
}

// --- evidence a failed identification owes -------------------------------------------------

compiled_terminal! {
    /// The compiled evidence obligations of a failed identification.
    #[derive(Clone, Debug)]
    CompiledEvidenceObligations {
        result: IdentificationResult,
        graph: Admg,
        observed: Vec<VariableId>,
        population: String,
        sensitivity: Option<SensitivityFinding>,
    }
}

/// Compile the evidence obligations a failed identification `result` owes on `graph`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_evidence_obligations(
    result: IdentificationResult,
    graph: Admg,
    observed: Vec<VariableId>,
    population: &str,
    sensitivity: Option<SensitivityFinding>,
    ctx: &ExecutionContext,
) -> Result<CompiledEvidenceObligations, CausalError> {
    admit_compile(ctx)?;
    let question = declaration_digest(
        "antecedent.evidence_obligations.v1",
        &[&result, &graph, &observed, &population, &sensitivity],
    );
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::EvidenceObligations,
        plan_id: "evidence_obligations.v1",
        functional: FunctionalKind::InterventionalDistribution,
        graph_class: "admg_identification_failure",
        question,
        evaluator: "obligation_adapters.identification_failure_obligations",
        reading: Reading::Exact,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: None,
        claim: None,
    })?;
    Ok(CompiledEvidenceObligations {
        lowered,
        result,
        graph,
        observed,
        population: population.to_owned(),
        sensitivity,
    })
}

impl CompiledEvidenceObligations {
    /// State the obligations.
    ///
    /// # Errors
    /// The adapter's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Vec<EvidenceObligation>, TerminalError<RepairError>> {
        let _scoped = self.enter(ctx)?;
        identification_failure_obligations(
            &self.result,
            &self.graph,
            &self.observed,
            &self.population,
            self.sensitivity.as_ref(),
        )
        .map_err(TerminalError::Kernel)
    }
}

// --- a finite decision problem under draws of the state ------------------------------------

/// A compiled finite decision problem under equally weighted draws of the decision state.
///
/// The utility and the constraints are live callbacks: the program identity records the
/// problem's shape (actions, draws, constraints, chance threshold) and that callbacks are
/// bound, not what they compute.
pub struct CompiledFiniteActionDecision<A, O> {
    lowered: Arc<LoweredProgram>,
    problem: DecisionProblem<A, O>,
    outcomes: Vec<O>,
}

impl<A, O> CompiledProgramHandle for CompiledFiniteActionDecision<A, O> {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl<A, O> std::fmt::Debug for CompiledFiniteActionDecision<A, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompiledFiniteActionDecision")
            .field("identity", &self.lowered.identity())
            .field("actions", &self.problem.actions.len())
            .field("outcomes", &self.outcomes.len())
            .finish_non_exhaustive()
    }
}

/// Compile the Bayes action and expected regret of `problem` under the equally weighted
/// draws `outcomes`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_finite_action_decision<A, O>(
    problem: DecisionProblem<A, O>,
    outcomes: Vec<O>,
    ctx: &ExecutionContext,
) -> Result<CompiledFiniteActionDecision<A, O>, CausalError> {
    admit_compile(ctx)?;
    let mut writer = CanonicalWriter::new("antecedent.finite_action_decision.v1");
    writer
        .word(problem.actions.len() as u64)
        .word(outcomes.len() as u64)
        .word(problem.constraints.len() as u64)
        .real(problem.chance_threshold);
    writer.note_runtime_callback();
    let decision = DecisionProgram {
        carrier: CarrierDemand::ParameterPosterior,
        utility: DecisionProgram::utility_form(None),
        actions: problem.actions.len(),
        constraints: problem.constraints.len(),
        chance_threshold: problem.chance_threshold,
        utility_unit: None,
        contract_identity: None,
        information_value: None,
        scenario_source: Some(Arc::from("state_draws")),
    };
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::FiniteActionDecision,
        plan_id: "finite_action_decision.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_state_draws",
        question: writer.finish(),
        evaluator: "decision.evaluate_decision",
        reading: Reading::Posterior,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision),
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledFiniteActionDecision { lowered, problem, outcomes })
}

impl<A, O> CompiledFiniteActionDecision<A, O> {
    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// The plan view of the compiled program. Reading it runs nothing.
    #[must_use]
    pub fn plan_view(&self) -> crate::planner::PlanView {
        crate::planner::PlanView::of_program(
            Arc::clone(&self.lowered),
            crate::planner::PlanStage::Compiled,
        )
    }

    /// Identity of the compiled program.
    #[must_use]
    pub fn identity(&self) -> antecedent_core::SemanticDigest {
        self.lowered.identity()
    }

    /// Evaluate the Bayes action and its expected regret.
    ///
    /// # Errors
    /// The evaluation's own refusals (a callback failure, no actions or draws, no admissible
    /// action), or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DecisionEvaluation, TerminalError<DesignError>> {
        let _scoped = crate::analysis::supplied_terminal::enter(ctx, self)?;
        evaluate_decision(&self.problem, &self.outcomes).map_err(TerminalError::Kernel)
    }
}

// --- a search over adaptive policy families -------------------------------------------------

/// Whether a release build refuses an adaptive policy search whose support unit is not
/// supported.
///
/// The unit is the capability record bound to the `AdaptivePolicySearch` route in
/// `parity/compiled_units.toml`. The gate stays closed until that record's evidence is earned,
/// so no valid use refuses before then; flipping this constant is the only change that turns
/// it on.
pub(crate) const ADAPTIVE_POLICY_SEARCH_GATE_LIVE: bool = false;

/// A compiled bounded search over adaptive policy families.
#[derive(Clone, Debug)]
pub struct CompiledAdaptivePolicySearch {
    lowered: Arc<LoweredProgram>,
    search: PreparedAdaptivePolicySearch,
}

impl CompiledProgramHandle for CompiledAdaptivePolicySearch {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

/// Compile a prepared adaptive policy search. The response source is bound at execution: the
/// program identity covers the families, constraints, objective and options, and the result
/// carries the source's identity.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_adaptive_policy_search(
    search: PreparedAdaptivePolicySearch,
    ctx: &ExecutionContext,
) -> Result<CompiledAdaptivePolicySearch, CausalError> {
    admit_compile(ctx)?;
    let mut writer = CanonicalWriter::new("antecedent.adaptive_policy_search.v1");
    writer.bytes(&search.identity());
    writer.note_runtime_callback();
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::AdaptivePolicySearch,
        plan_id: "adaptive_policy_search.v1",
        functional: FunctionalKind::RegimeValue,
        graph_class: "declared_policy_families",
        question: writer.finish(),
        evaluator: "adaptive_policy_search.execute",
        reading: Reading::MonteCarlo,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: None,
        claim: Some(ClaimKind::PolicyValue),
    })?;
    Ok(CompiledAdaptivePolicySearch { lowered, search })
}

impl CompiledAdaptivePolicySearch {
    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// The plan view of the compiled program. Reading it runs nothing.
    #[must_use]
    pub fn plan_view(&self) -> crate::planner::PlanView {
        crate::planner::PlanView::of_program(
            Arc::clone(&self.lowered),
            crate::planner::PlanStage::Compiled,
        )
    }

    /// Identity of the compiled program.
    #[must_use]
    pub fn identity(&self) -> antecedent_core::SemanticDigest {
        self.lowered.identity()
    }

    /// The prepared search.
    #[must_use]
    pub const fn search(&self) -> &PreparedAdaptivePolicySearch {
        &self.search
    }

    /// Run the search against `source`. Cancellation is the search's own refusal, as on the
    /// direct path.
    ///
    /// # Errors
    /// The search's own refusals, or the compiler's support gate once it is live.
    pub fn execute(
        &self,
        source: &dyn PolicyRolloutSource,
        ctx: &ExecutionContext,
    ) -> Result<AdaptivePolicySearchResult, TerminalError<AdaptivePolicySearchError>> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if ADAPTIVE_POLICY_SEARCH_GATE_LIVE {
            crate::analysis::program_handle::gate_program(&scoped, self)
                .map_err(CausalError::from)?;
        }
        self.search.execute(source, &scoped).map_err(TerminalError::Kernel)
    }
}
