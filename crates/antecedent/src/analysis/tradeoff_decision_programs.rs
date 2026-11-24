//! Compiled decisions across a range of preferences over evidence a caller supplies.
//!
//! A preference contract weighs several outcome attributes through trade-off parameters that
//! range over a declared or learned preference set. Decided on a law, a grid of means or a set
//! of structure claims the caller already holds, it compiles here to a program of its own
//! family and runs the same kernel the design crate exposes, on the same inputs, so every
//! verdict and number is the kernel's:
//!
//! | entry | family | kernel |
//! |---|---|---|
//! | [`compile_tradeoff_decision`] | `TradeoffLawDecision` (exact route) or `TradeoffCertifiedLawDecision` (certified route) | `decide_preferences_with_context` on a law, `decide_preferences_on_means_with_context` on means |
//! | [`compile_tradeoff_decision_with_accepted`] | the same two | the same, on the contract and measure of accepted preferences (`decide_with_accepted_with_context`) |
//! | [`compile_tradeoff_structural_decision`] | `TradeoffStructuralDecision` or `TradeoffCertifiedStructuralDecision` | the supplied claims' adapter, then `decide_preferences_structural_with_context` |
//!
//! A family binds one support unit, so the exact and the certified route are separate
//! families (`preference_decision.exact`, `preference_decision.certified`). Which one a
//! contract compiles to is read from the contract alone: the route rule
//! of [`preference_route`] and the exact-route refusals that send a contract to the certified
//! route (a part of the value that is not affine in its quantities, the worst case of expected
//! regret). Those are the only exact-route refusals the kernel falls through on, so the route
//! that answers is always the family's.
//!
//! A decision record ([`CompiledTradeoffDecision::record`]) is the law family's decision
//! sealed with its contract, measure, options and source digest
//! ([`PreferenceDecisionArtifact::evaluate_with_context`]); replaying a record is a consumer of
//! the record, not a decision.
//!
//! The contract is validated at compile, as the kernel validates it first: an invalid contract
//! is refused at compile with the kernel's own error, [`TerminalError::Kernel`]. The
//! compiler's own refusals (re-entry, a release build's support gate) are
//! [`TerminalError::Refused`].
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use antecedent_contract::ExternalRefusal;
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_design::accepted_preferences::{AcceptedPreferences, PreferenceContractParts};
use antecedent_design::decision_adapters::AdapterError;
use antecedent_design::decision_contract::{DecisionCriterion, StructuralPolicy};
use antecedent_design::preference_artifact::{PreferenceArtifactError, PreferenceDecisionArtifact};
use antecedent_design::preference_certified::CertifiedOptions;
use antecedent_design::preference_contract::{PreferenceContract, PreferencePolicy};
use antecedent_design::preference_decision::{
    MeasureOrigin, PreferenceDecisionOptions, PreferenceMeasure,
};
use antecedent_design::preference_evaluation::{
    PreferenceEvaluation, PreferenceEvaluationError, PreferenceEvaluationOptions, PreferenceRoute,
    StructuralPreferenceEvaluation, decide_preferences_on_means_with_context,
    decide_preferences_structural_with_context, decide_preferences_with_context, preference_route,
};
use antecedent_stats::linear_program::LpOptions;

use crate::analysis::claim_decision_programs::{ContractSource, SuppliedClaimSet};
use crate::analysis::supplied_identity::{Canonical, Declaration, variant};
pub use crate::analysis::supplied_terminal::TerminalError;
use crate::analysis::supplied_terminal::{
    Reading, TerminalSpec, admit_compile, compiled_terminal, lower,
};
use crate::error::CausalError;
use crate::planner::lowered::{
    CarrierDemand, Composition, DecisionProgram, EvidenceRole, FunctionalKind, ProgramCell,
};
use crate::planner::{PlanView, lowered::LoweredProgram};

/// Versioned tags of the families' declarations.
const LAW_TAG: &str = "antecedent.tradeoff_law_decision.v1";
const CERTIFIED_LAW_TAG: &str = "antecedent.tradeoff_certified_law_decision.v1";
const STRUCTURAL_TAG: &str = "antecedent.tradeoff_structural_decision.v1";
const CERTIFIED_STRUCTURAL_TAG: &str = "antecedent.tradeoff_certified_structural_decision.v1";

// --- the route a contract compiles to --------------------------------------------------------

/// The route that answers `contract`, read from the contract alone: the route rule of
/// [`preference_route`], then the exact-route refusals the kernel falls through on, which are
/// properties of the contract (a part of the value not affine in its quantities, so no term
/// contract states it; the worst case of expected regret under the preference set's or, across
/// structures, the structures' maximin policy).
///
/// # Errors
/// An invalid contract, with the kernel's own error.
pub(crate) fn answering_route(
    contract: &PreferenceContract,
    across_structures: bool,
) -> Result<PreferenceRoute, PreferenceEvaluationError> {
    if preference_route(contract)?.route == PreferenceRoute::Certified {
        return Ok(PreferenceRoute::Certified);
    }
    let regret = matches!(contract.criterion, DecisionCriterion::ExpectedRegret);
    let worst_case_regret = regret
        && (matches!(contract.preference_policy, PreferencePolicy::Maximin)
            || (across_structures
                && matches!(contract.structural_policy, StructuralPolicy::Maximin)));
    if worst_case_regret
        || contract.term_contract().map_err(PreferenceEvaluationError::Contract)?.is_none()
    {
        return Ok(PreferenceRoute::Certified);
    }
    Ok(PreferenceRoute::Exact)
}

// --- the decision layer and the declaration ---------------------------------------------------

/// The decision layer of a preference contract's program: its actions, constraints, strictest
/// chance threshold, value unit and identity. The value names trade-offs no affine utility
/// states, so the utility form is opaque and the contract identity carries it.
fn decision_program(
    contract: &PreferenceContract,
    identity: &str,
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
        utility_unit: Some(Arc::from(contract.model.utility_units.as_str())),
        contract_identity: Some(Arc::from(identity)),
        information_value: None,
        scenario_source: source.map(Arc::from),
    }
}

/// A measure over the preference set, as it is declared.
struct Measure<'a>(&'a PreferenceMeasure);

impl Canonical for Measure<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let PreferenceMeasure { points, weights, labels, origin, unresolved } = self.0;
        // Points, weights and labels are parallel and ordered.
        out.word(points.len() as u64);
        for point in points {
            out.reals(point);
        }
        match weights {
            Some(weights) => {
                out.word(1).reals(weights);
            }
            None => {
                out.word(0);
            }
        }
        match labels {
            Some(labels) => {
                out.word(1).texts(labels.as_slice());
            }
            None => {
                out.word(0);
            }
        }
        match origin {
            MeasureOrigin::Declared { assumption_id } => {
                variant(out, "declared").text(assumption_id);
            }
            MeasureOrigin::ChoiceEstimate { estimate_identity } => {
                variant(out, "choice_estimate").text(estimate_identity);
            }
        }
        unresolved.encode(out);
    }
}

/// The options of both routes. Every field is destructured, so an option added to either route
/// cannot be left out of the identity.
struct Options<'a>(&'a PreferenceEvaluationOptions);

impl Canonical for Options<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let PreferenceEvaluationOptions { exact, certified } = self.0;
        let PreferenceDecisionOptions { tie_tolerance: exact_tie, vertex_cap, lp } = exact;
        let LpOptions { feasibility_tolerance, optimality_tolerance, max_pivots } = lp;
        let CertifiedOptions { max_boxes, min_width_fraction, tie_tolerance: certified_tie } =
            certified;
        out.real(*exact_tie)
            .word(*vertex_cap as u64)
            .real(*feasibility_tolerance)
            .real(*optimality_tolerance)
            .word(*max_pivots as u64)
            .word(*max_boxes as u64)
            .real(*min_width_fraction)
            .real(*certified_tie);
    }
}

/// The declaration of a decision across a preference set: the contract's identity, then its
/// action and constraint ids in declaration order (results are reported in action order), the
/// measure, the options and, for accepted preferences, their identity.
fn declare(
    declaration: &mut Declaration,
    contract: &PreferenceContract,
    identity: &str,
    measure: Option<&PreferenceMeasure>,
    options: &PreferenceEvaluationOptions,
    accepted: Option<&str>,
) {
    let actions: Vec<&str> = contract.actions.iter().map(|action| action.id.as_str()).collect();
    let constraints: Vec<&str> = contract.constraints.iter().map(|c| c.id.as_str()).collect();
    declaration
        .field("contract", identity)
        .field("actions", &actions)
        .field("constraints", &constraints)
        .field("measure", &measure.map(Measure))
        .field("options", &Options(options))
        .field("accepted", &accepted);
}

/// The contract's identity, after the route rule validated it.
fn contract_identity(contract: &PreferenceContract) -> Result<String, PreferenceEvaluationError> {
    contract.identity().map_err(PreferenceEvaluationError::Contract)
}

// --- decisions on a law or a grid of means ----------------------------------------------------

compiled_terminal! {
    /// A compiled decision across a preference set on a law or a grid of means, answered by
    /// linear programs over the admissible set.
    #[derive(Clone, Debug)]
    CompiledTradeoffLawDecision {
        contract: PreferenceContract,
        measure: Option<PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        source: ContractSource,
    }
}

compiled_terminal! {
    /// A compiled decision across a preference set on a law, answered by certified
    /// subdivision of the admissible set.
    #[derive(Clone, Debug)]
    CompiledTradeoffCertifiedLawDecision {
        contract: PreferenceContract,
        measure: Option<PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        source: ContractSource,
    }
}

/// A compiled decision across a preference set on a law or a grid of means, of the family of
/// the route that answers it.
#[derive(Clone, Debug)]
pub enum CompiledTradeoffDecision {
    /// The exact route (`TradeoffLawDecision`, unit `preference_decision.exact`).
    Exact(CompiledTradeoffLawDecision),
    /// The certified route (`TradeoffCertifiedLawDecision`, unit
    /// `preference_decision.certified`).
    Certified(CompiledTradeoffCertifiedLawDecision),
}

/// Compile the decision of `contract` across its preference set on `source`, with an optional
/// measure over the set.
///
/// # Errors
/// An invalid contract (`TerminalError::Kernel`, the kernel's own error); `public_builder_reentry`
/// inside a lowered dispatch.
pub fn compile_tradeoff_decision(
    contract: PreferenceContract,
    measure: Option<PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    source: ContractSource,
    ctx: &ExecutionContext,
) -> Result<CompiledTradeoffDecision, TerminalError<PreferenceEvaluationError>> {
    compile_law(contract, measure, options, source, None, ctx)
}

/// Compile the decision with accepted preferences: the contract over the accepted model from
/// `parts`, with the accepted measure when there is one. The accepted identity is part of the
/// program's declaration.
///
/// # Errors
/// As [`compile_tradeoff_decision`].
pub fn compile_tradeoff_decision_with_accepted(
    accepted: &AcceptedPreferences,
    parts: PreferenceContractParts,
    options: PreferenceEvaluationOptions,
    source: ContractSource,
    ctx: &ExecutionContext,
) -> Result<CompiledTradeoffDecision, TerminalError<PreferenceEvaluationError>> {
    let contract = accepted.contract(parts);
    let measure = accepted.measure().cloned();
    compile_law(contract, measure, options, source, Some(accepted.identity()), ctx)
}

fn compile_law(
    contract: PreferenceContract,
    measure: Option<PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    source: ContractSource,
    accepted: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<CompiledTradeoffDecision, TerminalError<PreferenceEvaluationError>> {
    admit_compile(ctx)?;
    let route = answering_route(&contract, false).map_err(TerminalError::Kernel)?;
    let identity = contract_identity(&contract).map_err(TerminalError::Kernel)?;
    let (tag, cell, plan_id) = match route {
        PreferenceRoute::Exact => {
            (LAW_TAG, ProgramCell::TradeoffLawDecision, "supplied.tradeoff_law_decision.v1")
        }
        PreferenceRoute::Certified => (
            CERTIFIED_LAW_TAG,
            ProgramCell::TradeoffCertifiedLawDecision,
            "supplied.tradeoff_certified_law_decision.v1",
        ),
    };
    let mut declaration = Declaration::new(tag);
    declare(&mut declaration, &contract, &identity, measure.as_ref(), &options, accepted);
    declaration.field("source", &source);
    let decision = decision_program(&contract, &identity, source.carrier(), None);
    let lowered = lower(TerminalSpec {
        cell,
        plan_id,
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_law",
        question: declaration.finish(),
        evaluator: match &source {
            ContractSource::Law(_) => "preference_evaluation::decide_preferences_with_context",
            ContractSource::Means(_) => {
                "preference_evaluation::decide_preferences_on_means_with_context"
            }
        },
        reading: source.reading(),
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision),
        claim: Some(antecedent_contract::ClaimKind::PolicyValue),
    })?;
    Ok(match route {
        PreferenceRoute::Exact => CompiledTradeoffDecision::Exact(CompiledTradeoffLawDecision {
            lowered,
            contract,
            measure,
            options,
            source,
        }),
        PreferenceRoute::Certified => {
            CompiledTradeoffDecision::Certified(CompiledTradeoffCertifiedLawDecision {
                lowered,
                contract,
                measure,
                options,
                source,
            })
        }
    })
}

/// Decide on the declared source with the kernel the direct entry calls.
fn decide_on_source(
    contract: &PreferenceContract,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    source: &ContractSource,
    ctx: &ExecutionContext,
) -> Result<PreferenceEvaluation, PreferenceEvaluationError> {
    match source {
        ContractSource::Law(law) => {
            decide_preferences_with_context(contract, law, measure, options, ctx)
        }
        ContractSource::Means(means) => {
            decide_preferences_on_means_with_context(contract, means, measure, options, ctx)
        }
    }
}

/// A decision record needs the aligned joint draws it was computed from.
fn record_needs_law(plan_id: &str) -> CausalError {
    crate::compile_reason!(
        "route_not_supported",
        "{plan_id}.record: a decision record binds the aligned joint draws it was decided on, \
         and a grid of means is not a law; decide on joint draws to record the decision"
    )
}

macro_rules! law_family {
    ($name:ident, $route:expr, $plan:literal) => {
        impl $name {
            /// The declared contract.
            #[must_use]
            pub const fn contract(&self) -> &PreferenceContract {
                &self.contract
            }

            /// The declared measure over the preference set.
            #[must_use]
            pub const fn measure(&self) -> Option<&PreferenceMeasure> {
                self.measure.as_ref()
            }

            /// The declared options of both routes.
            #[must_use]
            pub const fn options(&self) -> PreferenceEvaluationOptions {
                self.options
            }

            /// The declared source.
            #[must_use]
            pub const fn source(&self) -> &ContractSource {
                &self.source
            }

            /// Decide the contract across its preference set on the source. The route reads
            /// the context's cancellation token at its work boundaries.
            ///
            /// # Errors
            /// The kernel's own refusals (an invalid measure or options, a contract a mean
            /// source cannot answer, an empty preference set, a linear-program or subdivision
            /// failure, cancellation), or the compiler's.
            pub fn execute(
                &self,
                ctx: &ExecutionContext,
            ) -> Result<PreferenceEvaluation, TerminalError<PreferenceEvaluationError>> {
                let scoped = self.enter(ctx)?;
                let evaluation = decide_on_source(
                    &self.contract,
                    self.measure.as_ref(),
                    self.options,
                    &self.source,
                    &scoped,
                )
                .map_err(TerminalError::Kernel)?;
                debug_assert_eq!(evaluation.route(), $route, "the family's route answered");
                Ok(evaluation)
            }

            /// Decide on the declared law and seal the decision with its contract, measure,
            /// options and the law's digest, for export.
            ///
            /// # Errors
            /// A grid of means (`route_not_supported`: a record binds joint draws); the
            /// kernel's refusals, as the record's error; or the compiler's.
            pub fn record(
                &self,
                ctx: &ExecutionContext,
            ) -> Result<PreferenceDecisionArtifact, TerminalError<PreferenceArtifactError>> {
                let ContractSource::Law(law) = &self.source else {
                    return Err(TerminalError::Refused(record_needs_law($plan)));
                };
                let scoped = self.enter(ctx)?;
                PreferenceDecisionArtifact::evaluate_with_context(
                    self.contract.clone(),
                    law,
                    self.measure.clone(),
                    self.options,
                    &scoped,
                )
                .map_err(TerminalError::Kernel)
            }
        }
    };
}

law_family!(
    CompiledTradeoffLawDecision,
    PreferenceRoute::Exact,
    "supplied.tradeoff_law_decision.v1"
);
law_family!(
    CompiledTradeoffCertifiedLawDecision,
    PreferenceRoute::Certified,
    "supplied.tradeoff_certified_law_decision.v1"
);

impl CompiledTradeoffDecision {
    /// The route that answers the contract, and so the family it compiled to.
    #[must_use]
    pub const fn route(&self) -> PreferenceRoute {
        match self {
            Self::Exact(_) => PreferenceRoute::Exact,
            Self::Certified(_) => PreferenceRoute::Certified,
        }
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        match self {
            Self::Exact(program) => program.lowered(),
            Self::Certified(program) => program.lowered(),
        }
    }

    /// The plan view of the compiled program, with its route and the support decision of the
    /// route's unit. Reading it runs nothing.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        match self {
            Self::Exact(program) => program.plan_view(),
            Self::Certified(program) => program.plan_view(),
        }
    }

    /// Identity of the compiled program.
    #[must_use]
    pub fn identity(&self) -> antecedent_core::SemanticDigest {
        match self {
            Self::Exact(program) => program.identity(),
            Self::Certified(program) => program.identity(),
        }
    }

    /// The declared contract.
    #[must_use]
    pub const fn contract(&self) -> &PreferenceContract {
        match self {
            Self::Exact(program) => program.contract(),
            Self::Certified(program) => program.contract(),
        }
    }

    /// Decide the contract on its source.
    ///
    /// # Errors
    /// As the family's `execute`.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PreferenceEvaluation, TerminalError<PreferenceEvaluationError>> {
        match self {
            Self::Exact(program) => program.execute(ctx),
            Self::Certified(program) => program.execute(ctx),
        }
    }

    /// Decide on the declared law and seal the decision for export.
    ///
    /// # Errors
    /// As the family's `record`.
    pub fn record(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PreferenceDecisionArtifact, TerminalError<PreferenceArtifactError>> {
        match self {
            Self::Exact(program) => program.record(ctx),
            Self::Certified(program) => program.record(ctx),
        }
    }
}

// --- decisions across structures --------------------------------------------------------------

/// A refusal of a decision across structures and a preference set: the claims' adapter or the
/// evaluation, each with its own error.
#[derive(Clone, Debug, PartialEq)]
pub enum TradeoffStructuralError {
    /// The adapter refused the claims.
    Adapter(AdapterError),
    /// The evaluation across the structures and the preference set refused.
    Evaluation(PreferenceEvaluationError),
}

impl TradeoffStructuralError {
    /// The structured refusal of the error, with its registered code and detail.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        match self {
            Self::Adapter(error) => error.to_refusal(),
            Self::Evaluation(error) => error.to_refusal(),
        }
    }
}

impl fmt::Display for TradeoffStructuralError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let refusal = self.to_refusal();
        write!(formatter, "{}: {} at {}", refusal.code, refusal.detail, refusal.stage)
    }
}

impl std::error::Error for TradeoffStructuralError {}

compiled_terminal! {
    /// A compiled decision across supplied structure claims and a preference set together,
    /// answered by linear programs over the admissible set.
    #[derive(Clone, Debug)]
    CompiledTradeoffStructuralDecision {
        contract: PreferenceContract,
        measure: Option<PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        claims: SuppliedClaimSet,
    }
}

compiled_terminal! {
    /// A compiled decision across supplied structure claims and a preference set together,
    /// answered by certified subdivision of the admissible set.
    #[derive(Clone, Debug)]
    CompiledTradeoffCertifiedStructuralDecision {
        contract: PreferenceContract,
        measure: Option<PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        claims: SuppliedClaimSet,
    }
}

/// A compiled decision across supplied structures and a preference set, of the family of the
/// route that answers it.
#[derive(Clone, Debug)]
pub enum CompiledTradeoffStructures {
    /// The exact route (`TradeoffStructuralDecision`, unit `preference_decision.exact`).
    Exact(CompiledTradeoffStructuralDecision),
    /// The certified route (`TradeoffCertifiedStructuralDecision`, unit
    /// `preference_decision.certified`).
    Certified(CompiledTradeoffCertifiedStructuralDecision),
}

/// Compile the decision of `contract` across the supplied structure claims and its preference
/// set together. The claims are read under the contract's own structural policy.
///
/// # Errors
/// An invalid contract (`TerminalError::Kernel`, the evaluation's own error);
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_tradeoff_structural_decision(
    contract: PreferenceContract,
    measure: Option<PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    claims: SuppliedClaimSet,
    ctx: &ExecutionContext,
) -> Result<CompiledTradeoffStructures, TerminalError<TradeoffStructuralError>> {
    admit_compile(ctx)?;
    // The direct path adapts the claims before the evaluation validates the contract, so a
    // contract refused here reports the adapter's refusal first when the claims are refused too.
    let refuse =
        |error: PreferenceEvaluationError| {
            let adapter = claims.clone().adapt(contract.structural_policy).err();
            TerminalError::Kernel(adapter.map_or(
                TradeoffStructuralError::Evaluation(error),
                TradeoffStructuralError::Adapter,
            ))
        };
    let route = answering_route(&contract, true).map_err(refuse)?;
    let identity = contract_identity(&contract).map_err(refuse)?;
    let (tag, cell, plan_id) = match route {
        PreferenceRoute::Exact => (
            STRUCTURAL_TAG,
            ProgramCell::TradeoffStructuralDecision,
            "supplied.tradeoff_structural_decision.v1",
        ),
        PreferenceRoute::Certified => (
            CERTIFIED_STRUCTURAL_TAG,
            ProgramCell::TradeoffCertifiedStructuralDecision,
            "supplied.tradeoff_certified_structural_decision.v1",
        ),
    };
    let mut declaration = Declaration::new(tag);
    declare(&mut declaration, &contract, &identity, measure.as_ref(), &options, None);
    declaration.field("claims", &claims);
    let decision =
        decision_program(&contract, &identity, CarrierDemand::IdentifiedSet, Some(claims.kind()));
    let lowered = lower(TerminalSpec {
        cell,
        plan_id,
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_structural_claims",
        question: declaration.finish(),
        evaluator: "preference_evaluation::decide_preferences_structural_with_context",
        reading: Reading::Envelope,
        role: EvidenceRole::JointScenarios,
        composition: Composition::ScenarioSet { scenarios: claims.len() },
        decision: Some(decision),
        claim: Some(antecedent_contract::ClaimKind::PolicyValue),
    })?;
    Ok(match route {
        PreferenceRoute::Exact => {
            CompiledTradeoffStructures::Exact(CompiledTradeoffStructuralDecision {
                lowered,
                contract,
                measure,
                options,
                claims,
            })
        }
        PreferenceRoute::Certified => {
            CompiledTradeoffStructures::Certified(CompiledTradeoffCertifiedStructuralDecision {
                lowered,
                contract,
                measure,
                options,
                claims,
            })
        }
    })
}

macro_rules! structural_family {
    ($name:ident, $route:expr) => {
        impl $name {
            /// The declared contract.
            #[must_use]
            pub const fn contract(&self) -> &PreferenceContract {
                &self.contract
            }

            /// The declared claims.
            #[must_use]
            pub const fn claims(&self) -> &SuppliedClaimSet {
                &self.claims
            }

            /// Adapt the claims under the contract's structural policy and decide across the
            /// structures and the preference set together. The route reads the context's
            /// cancellation token before every structure and at its work boundaries.
            ///
            /// # Errors
            /// The adapter's and the evaluation's own refusals, or the compiler's.
            pub fn execute(
                &self,
                ctx: &ExecutionContext,
            ) -> Result<StructuralPreferenceEvaluation, TerminalError<TradeoffStructuralError>>
            {
                let scoped = self.enter(ctx)?;
                let adapted = self
                    .claims
                    .clone()
                    .adapt(self.contract.structural_policy)
                    .map_err(|e| TerminalError::Kernel(TradeoffStructuralError::Adapter(e)))?;
                let evaluation = decide_preferences_structural_with_context(
                    &self.contract,
                    &adapted.atoms,
                    self.measure.as_ref(),
                    self.options,
                    &scoped,
                )
                .map_err(|e| TerminalError::Kernel(TradeoffStructuralError::Evaluation(e)))?;
                debug_assert_eq!(evaluation.route(), $route, "the family's route answered");
                Ok(evaluation)
            }
        }
    };
}

structural_family!(CompiledTradeoffStructuralDecision, PreferenceRoute::Exact);
structural_family!(CompiledTradeoffCertifiedStructuralDecision, PreferenceRoute::Certified);

impl CompiledTradeoffStructures {
    /// The route that answers the contract, and so the family it compiled to.
    #[must_use]
    pub const fn route(&self) -> PreferenceRoute {
        match self {
            Self::Exact(_) => PreferenceRoute::Exact,
            Self::Certified(_) => PreferenceRoute::Certified,
        }
    }

    /// The declared contract.
    #[must_use]
    pub const fn contract(&self) -> &PreferenceContract {
        match self {
            Self::Exact(program) => program.contract(),
            Self::Certified(program) => program.contract(),
        }
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        match self {
            Self::Exact(program) => program.lowered(),
            Self::Certified(program) => program.lowered(),
        }
    }

    /// The plan view of the compiled program. Reading it runs nothing.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        match self {
            Self::Exact(program) => program.plan_view(),
            Self::Certified(program) => program.plan_view(),
        }
    }

    /// Identity of the compiled program.
    #[must_use]
    pub fn identity(&self) -> antecedent_core::SemanticDigest {
        match self {
            Self::Exact(program) => program.identity(),
            Self::Certified(program) => program.identity(),
        }
    }

    /// Decide across the structures and the preference set.
    ///
    /// # Errors
    /// As the family's `execute`.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<StructuralPreferenceEvaluation, TerminalError<TradeoffStructuralError>> {
        match self {
            Self::Exact(program) => program.execute(ctx),
            Self::Certified(program) => program.execute(ctx),
        }
    }
}
