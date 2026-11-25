//! Compiled decisions over supplied claims, identified sets and declared laws.
//!
//! Each entry compiles a decision a caller states over evidence it supplies (claims adapted to
//! structural atoms, per-action identified sets, an aligned law or a grid of means) into a
//! program of its own family, and executes the same kernel the direct entry calls:
//!
//! | entry | family | kernel |
//! |---|---|---|
//! | [`compile_supplied_claim_decision`] | `SuppliedClaimDecision` | `adapt_point_claim`, `adapt_graph_dependent_claims`, `adapt_weighted_graph_atoms`, `adapt_finite_scenarios`, then `evaluate_adapted` |
//! | [`compile_identified_set_decision`] | `IdentifiedSetDecision` | `utility_interval` per declared action, then `evaluate_identified_sets` |
//! | [`compile_contract_decision`] | `ContractLawDecision` | `evaluate_contract` on a law, `evaluate_contract_on_means` on means |
//! | [`compile_decision_functional`] | `DecisionFunctionalValue` | `evaluate_functional` |
//!
//! The claims are read under the contract's own structural policy, so a claim set adapted
//! under one policy is never evaluated under another. An identified set is a structural
//! envelope: the decision compares intervals and reads no probability, and the interval of an
//! action declared by its inputs is the outer enclosure `utility_interval` computes.
//!
//! A refusal of the kernel is returned unchanged as [`TerminalError::Kernel`]; the compiler's
//! own refusals (re-entry, a release build's support gate) are
//! [`TerminalError::Refused`].
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::response::IdentifiedSet;
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_design::DecisionFunctional;
use antecedent_design::decision_adapters::{
    AdaptedClaims, AdaptedDecision, AdapterError, IdentifiedSetDecision, IdentifiedUtility,
    SuppliedClaim, adapt_finite_scenarios, adapt_graph_dependent_claims, adapt_point_claim,
    adapt_weighted_graph_atoms, evaluate_adapted, evaluate_identified_sets, utility_interval,
};
use antecedent_design::decision_contract::{
    AdmissibleDecisionContract, DecisionContract, StructuralPolicy,
};
use antecedent_design::decision_eval::{
    DecisionEvalError, DecisionResult, FunctionalValue, MeanEstimate, MeanSource,
    evaluate_contract, evaluate_contract_on_means, evaluate_functional,
};
use antecedent_design::decision_robustness::AtomSupport;
use antecedent_design::decision_structural::AtomEvidence;
use antecedent_io::distribution_artifact::DistributionArtifact;
use std::sync::Arc;

use crate::analysis::supplied_identity::{Canonical, Declaration, Wire, variant};
pub use crate::analysis::supplied_terminal::TerminalError;
use crate::analysis::supplied_terminal::{
    Reading, TerminalSpec, admit_compile, compiled_terminal, lower,
};
use crate::error::CausalError;
use crate::planner::lowered::{
    CarrierDemand, Composition, DecisionProgram, EvidenceRole, FunctionalKind, ProgramCell,
};

/// The decision layer of a contract's program: its actions, constraints, strictest chance
/// threshold, utility unit and identity (absent when the contract does not validate; the
/// kernel then refuses it at execution with its own error).
fn decision_program(
    contract: &DecisionContract,
    identity: Option<String>,
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
        contract_identity: identity.map(Arc::from),
        information_value: None,
        scenario_source: source.map(Arc::from),
    }
}

// --- supplied claims -------------------------------------------------------------------------

/// Claims a caller supplies for a structural decision, by how they are to be read.
#[derive(Clone, Debug)]
pub enum SuppliedClaimSet {
    /// One exact law or answer.
    Point {
        /// Claim identity.
        id: String,
        /// The law, or why there is none.
        evidence: AtomEvidence,
        /// Support of the claim's inputs.
        support: AtomSupport,
    },
    /// One answer per graph, with no probabilities (completion counts stay counts).
    GraphDependent(Vec<SuppliedClaim>),
    /// Weighted graph-posterior atoms: credible with a probability on every atom, otherwise a
    /// structural envelope.
    WeightedGraphAtoms(Vec<SuppliedClaim>),
    /// A finite set of supplied scenarios, always a structural envelope.
    FiniteScenarios(Vec<SuppliedClaim>),
}

impl SuppliedClaimSet {
    /// Stable name of how the claims are read.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Point { .. } => "point",
            Self::GraphDependent(_) => "graph_dependent",
            Self::WeightedGraphAtoms(_) => "weighted_graph_posterior",
            Self::FiniteScenarios(_) => "finite_scenarios",
        }
    }

    /// Number of claims.
    pub(crate) const fn len(&self) -> usize {
        match self {
            Self::Point { .. } => 1,
            Self::GraphDependent(claims)
            | Self::WeightedGraphAtoms(claims)
            | Self::FiniteScenarios(claims) => claims.len(),
        }
    }

    /// Adapt the claims to structural atoms under `policy`. Every compiled decision and
    /// legacy robust artifact replay over supplied claims adapts here.
    ///
    /// # Errors
    /// The adapter's refusal when the claims cannot be read under `policy`.
    #[doc(hidden)]
    pub fn adapt(self, policy: StructuralPolicy) -> Result<AdaptedClaims, AdapterError> {
        match self {
            Self::Point { id, evidence, support } => {
                adapt_point_claim(&id, evidence, support, policy)
            }
            Self::GraphDependent(claims) => adapt_graph_dependent_claims(claims, policy),
            Self::WeightedGraphAtoms(claims) => adapt_weighted_graph_atoms(claims, policy),
            Self::FiniteScenarios(claims) => adapt_finite_scenarios(claims, policy),
        }
    }
}

/// The adapted claims and the decision over them.
#[derive(Clone, Debug)]
pub struct SuppliedClaimReadout {
    /// The claims as structural atoms, with their profile and completion counts.
    pub adapted: AdaptedClaims,
    /// The structural result and the robustness verdict.
    pub decision: AdaptedDecision,
}

compiled_terminal! {
    /// A compiled structural decision over supplied claims.
    #[derive(Clone, Debug)]
    CompiledSuppliedClaimDecision {
        contract: AdmissibleDecisionContract,
        claims: SuppliedClaimSet,
    }
}

/// Compile a structural decision over supplied claims, read under the contract's structural
/// policy.
///
/// Nothing is adapted or evaluated at compile; the program's identity covers the contract and
/// every claim.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_supplied_claim_decision(
    contract: AdmissibleDecisionContract,
    claims: SuppliedClaimSet,
    ctx: &ExecutionContext,
) -> Result<CompiledSuppliedClaimDecision, CausalError> {
    admit_compile(ctx)?;
    let question = Declaration::new("antecedent.supplied_claim_decision.v2")
        .field("contract", &contract)
        .field("claims", &claims)
        .finish();
    let decision = decision_program(
        &contract.contract,
        contract.identity().ok(),
        CarrierDemand::IdentifiedSet,
        Some(claims.kind()),
    );
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::SuppliedClaimDecision,
        plan_id: "supplied.supplied_claim_decision.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_structural_claims",
        question,
        evaluator: "decision_adapters::evaluate_adapted",
        reading: Reading::Envelope,
        role: EvidenceRole::JointScenarios,
        composition: Composition::ScenarioSet { scenarios: claims.len() },
        decision: Some(decision),
        claim: Some(antecedent_contract::ClaimKind::PolicyValue),
    })?;
    Ok(CompiledSuppliedClaimDecision { lowered, contract, claims })
}

impl CompiledSuppliedClaimDecision {
    /// The declared contract.
    #[must_use]
    pub const fn contract(&self) -> &AdmissibleDecisionContract {
        &self.contract
    }

    /// The declared claims.
    #[must_use]
    pub const fn claims(&self) -> &SuppliedClaimSet {
        &self.claims
    }

    /// Adapt the claims under the contract's structural policy and decide over them.
    ///
    /// # Errors
    /// The adapter's and the structural evaluation's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SuppliedClaimReadout, TerminalError<AdapterError>> {
        let _scoped = self.enter(ctx)?;
        let policy = self.contract.contract.structural_policy;
        let adapted = self.claims.clone().adapt(policy).map_err(TerminalError::Kernel)?;
        let decision = evaluate_adapted(&self.contract, &adapted).map_err(TerminalError::Kernel)?;
        Ok(SuppliedClaimReadout { adapted, decision })
    }
}

// --- identified sets -------------------------------------------------------------------------

/// One action's identified set of utility, declared directly or by its inputs.
#[derive(Clone, Debug, PartialEq)]
pub enum IdentifiedAction {
    /// The action's utility interval, as supplied.
    Utility(IdentifiedUtility),
    /// One identified set per input of the action; the utility interval is their outer
    /// enclosure through the action's utility expression.
    Inputs {
        /// Semantic action identity.
        action_id: String,
        /// One identified set per declared input of the action, in order.
        inputs: Vec<IdentifiedSet<f64>>,
        /// Hard constraints the action violates.
        hard_exclusions: Vec<String>,
        /// Support of each input.
        support: AtomSupport,
    },
}

impl IdentifiedAction {
    fn action_id(&self) -> &str {
        match self {
            Self::Utility(utility) => &utility.action_id,
            Self::Inputs { action_id, .. } => action_id,
        }
    }
}

/// The identified-set carrier of a decision: one identified set of utility per action.
///
/// A carrier is a structural envelope. It reads no probability and asserts no dependence
/// between actions: the decision compares intervals.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentifiedSetCarrier {
    actions: Vec<IdentifiedAction>,
}

impl IdentifiedSetCarrier {
    /// A carrier of supplied per-action utility intervals.
    #[must_use]
    pub fn from_utilities(utilities: Vec<IdentifiedUtility>) -> Self {
        Self { actions: utilities.into_iter().map(IdentifiedAction::Utility).collect() }
    }

    /// A carrier of per-action declarations.
    #[must_use]
    pub const fn new(actions: Vec<IdentifiedAction>) -> Self {
        Self { actions }
    }

    /// The declared actions, in order.
    #[must_use]
    pub fn actions(&self) -> &[IdentifiedAction] {
        &self.actions
    }
}

compiled_terminal! {
    /// A compiled decision over per-action identified sets of utility.
    #[derive(Clone, Debug)]
    CompiledIdentifiedSetDecision {
        contract: AdmissibleDecisionContract,
        carrier: IdentifiedSetCarrier,
    }
}

/// Compile a decision over an identified-set carrier.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_identified_set_decision(
    contract: AdmissibleDecisionContract,
    carrier: IdentifiedSetCarrier,
    ctx: &ExecutionContext,
) -> Result<CompiledIdentifiedSetDecision, CausalError> {
    admit_compile(ctx)?;
    let question = Declaration::new("antecedent.identified_set_decision.v2")
        .field("contract", &contract)
        .field("carrier", &carrier)
        .finish();
    let decision = decision_program(
        &contract.contract,
        contract.identity().ok(),
        CarrierDemand::IdentifiedSet,
        Some("identified_sets"),
    );
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::IdentifiedSetDecision,
        plan_id: "supplied.identified_set_decision.v1",
        functional: FunctionalKind::IdentifiedSet,
        graph_class: "supplied_identified_sets",
        question,
        evaluator: "decision_adapters::evaluate_identified_sets",
        reading: Reading::Envelope,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision),
        claim: Some(antecedent_contract::ClaimKind::Bounds),
    })?;
    Ok(CompiledIdentifiedSetDecision { lowered, contract, carrier })
}

impl CompiledIdentifiedSetDecision {
    /// The declared contract.
    #[must_use]
    pub const fn contract(&self) -> &AdmissibleDecisionContract {
        &self.contract
    }

    /// The declared carrier.
    #[must_use]
    pub const fn carrier(&self) -> &IdentifiedSetCarrier {
        &self.carrier
    }

    fn utilities_in(&self) -> Result<Vec<IdentifiedUtility>, AdapterError> {
        self.carrier
            .actions
            .iter()
            .map(|action| match action {
                IdentifiedAction::Utility(utility) => Ok(utility.clone()),
                IdentifiedAction::Inputs { action_id, inputs, hard_exclusions, support } => {
                    Ok(IdentifiedUtility {
                        action_id: action_id.clone(),
                        utility: utility_interval(&self.contract.contract, action_id, inputs)?,
                        hard_exclusions: hard_exclusions.clone(),
                        support: support.clone(),
                    })
                }
            })
            .collect()
    }

    /// The utility interval of every declared action: as supplied, or the outer enclosure of
    /// its inputs' identified sets. A carrier need not cover every action of the contract to
    /// be read this way.
    ///
    /// # Errors
    /// The enclosure's own refusals (an unknown action, an input count different from the
    /// action's, an invalid contract, a non-finite or inverted interval), or the compiler's.
    pub fn utilities(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Vec<IdentifiedUtility>, TerminalError<AdapterError>> {
        let _scoped = self.enter(ctx)?;
        self.utilities_in().map_err(TerminalError::Kernel)
    }

    /// Decide over the carrier.
    ///
    /// # Errors
    /// The enclosure's and the identified-set evaluation's own refusals (an action not covered
    /// exactly once, a probability-reading policy or criterion), or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<IdentifiedSetDecision, TerminalError<AdapterError>> {
        let _scoped = self.enter(ctx)?;
        let utilities = self.utilities_in().map_err(TerminalError::Kernel)?;
        evaluate_identified_sets(&self.contract, &utilities).map_err(TerminalError::Kernel)
    }

    /// The identity of the action whose utility the carrier declares at `index`.
    #[must_use]
    pub fn action_id(&self, index: usize) -> Option<&str> {
        self.carrier.actions.get(index).map(IdentifiedAction::action_id)
    }
}

// --- laws and means --------------------------------------------------------------------------

/// What a contract is evaluated on.
#[derive(Clone, Debug, PartialEq)]
#[allow(
    clippy::large_enum_variant,
    reason = "a source is built once per program and moved into it; boxing the law would add an indirection to the public construction"
)]
pub enum ContractSource {
    /// Aligned joint draws of the actions' inputs.
    Law(DistributionArtifact),
    /// One mean per coordinate; answers only an affine expected utility.
    Means(MeanSource),
}

impl ContractSource {
    pub(crate) const fn reading(&self) -> Reading {
        match self {
            Self::Law(_) => Reading::Predictive,
            Self::Means(_) => Reading::Exact,
        }
    }

    pub(crate) const fn carrier(&self) -> CarrierDemand {
        match self {
            Self::Law(_) => CarrierDemand::PredictiveLaw,
            Self::Means(_) => CarrierDemand::Mean,
        }
    }
}

compiled_terminal! {
    /// A compiled evaluation of a decision contract on a declared law or grid of means.
    #[derive(Clone, Debug)]
    CompiledContractDecision {
        contract: DecisionContract,
        source: ContractSource,
    }
}

/// Compile the evaluation of `contract` on `source`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_contract_decision(
    contract: DecisionContract,
    source: ContractSource,
    ctx: &ExecutionContext,
) -> Result<CompiledContractDecision, CausalError> {
    admit_compile(ctx)?;
    let question = Declaration::new("antecedent.contract_decision.v2")
        .field("contract", &contract)
        .field("source", &source)
        .finish();
    let decision = decision_program(&contract, contract.identity().ok(), source.carrier(), None);
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::ContractLawDecision,
        plan_id: "supplied.contract_law_decision.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_law",
        question,
        evaluator: match &source {
            ContractSource::Law(_) => "decision_eval::evaluate_contract",
            ContractSource::Means(_) => "decision_eval::evaluate_contract_on_means",
        },
        reading: source.reading(),
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision),
        claim: Some(antecedent_contract::ClaimKind::PolicyValue),
    })?;
    Ok(CompiledContractDecision { lowered, contract, source })
}

impl CompiledContractDecision {
    /// The declared contract.
    #[must_use]
    pub const fn contract(&self) -> &DecisionContract {
        &self.contract
    }

    /// The declared source.
    #[must_use]
    pub const fn source(&self) -> &ContractSource {
        &self.source
    }

    /// Evaluate the contract on its source.
    ///
    /// # Errors
    /// The evaluator's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DecisionResult, TerminalError<DecisionEvalError>> {
        let _scoped = self.enter(ctx)?;
        match &self.source {
            ContractSource::Law(law) => evaluate_contract(&self.contract, law),
            ContractSource::Means(means) => evaluate_contract_on_means(&self.contract, means),
        }
        .map_err(TerminalError::Kernel)
    }
}

// --- functionals -----------------------------------------------------------------------------

compiled_terminal! {
    /// A compiled functional of one action's utility under a declared law.
    #[derive(Clone, Debug)]
    CompiledDecisionFunctional {
        contract: DecisionContract,
        action_id: String,
        functional: DecisionFunctional,
        law: DistributionArtifact,
    }
}

/// Compile `functional` of `action_id`'s utility under `law`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_decision_functional(
    contract: DecisionContract,
    action_id: &str,
    functional: DecisionFunctional,
    law: DistributionArtifact,
    ctx: &ExecutionContext,
) -> Result<CompiledDecisionFunctional, CausalError> {
    admit_compile(ctx)?;
    let question = Declaration::new("antecedent.decision_functional.v2")
        .field("contract", &contract)
        .field("action", action_id)
        .field("functional", &Wire(&functional))
        .field("law", &law)
        .finish();
    let decision =
        decision_program(&contract, contract.identity().ok(), CarrierDemand::PredictiveLaw, None);
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::DecisionFunctionalValue,
        plan_id: "supplied.decision_functional_value.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "supplied_decision_law",
        question,
        evaluator: "decision_eval::evaluate_functional",
        reading: Reading::Predictive,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: Some(decision),
        claim: Some(antecedent_contract::ClaimKind::Point),
    })?;
    Ok(CompiledDecisionFunctional {
        lowered,
        contract,
        action_id: action_id.to_owned(),
        functional,
        law,
    })
}

impl CompiledDecisionFunctional {
    /// Compute the functional.
    ///
    /// # Errors
    /// The evaluator's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<FunctionalValue, TerminalError<DecisionEvalError>> {
        let _scoped = self.enter(ctx)?;
        evaluate_functional(&self.contract, &self.action_id, self.functional, &self.law)
            .map_err(TerminalError::Kernel)
    }
}

// --- canonical identity of the declared inputs ---------------------------------------------

impl Canonical for SuppliedClaimSet {
    fn encode(&self, out: &mut CanonicalWriter) {
        // Claims are ordered: the adapted atoms and the per-atom results keep their order.
        match self {
            Self::Point { id, evidence, support } => {
                variant(out, "point").text(id);
                evidence.encode(out);
                support.encode(out);
            }
            Self::GraphDependent(claims) => {
                variant(out, "graph_dependent");
                claims.encode(out);
            }
            Self::WeightedGraphAtoms(claims) => {
                variant(out, "weighted_graph_posterior");
                claims.encode(out);
            }
            Self::FiniteScenarios(claims) => {
                variant(out, "finite_scenarios");
                claims.encode(out);
            }
        }
    }
}

impl Canonical for IdentifiedAction {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Self::Utility(utility) => {
                variant(out, "utility");
                utility.encode(out);
            }
            Self::Inputs { action_id, inputs, hard_exclusions, support } => {
                // Inputs are positional: one identified set per declared input, in order.
                variant(out, "inputs").text(action_id);
                inputs.encode(out);
                hard_exclusions.encode(out);
                support.encode(out);
            }
        }
    }
}

impl Canonical for IdentifiedSetCarrier {
    fn encode(&self, out: &mut CanonicalWriter) {
        // Ordered: the enclosed utilities are reported in the carrier's order.
        self.actions.encode(out);
    }
}

impl Canonical for ContractSource {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Self::Law(law) => {
                variant(out, "law");
                law.encode(out);
            }
            Self::Means(means) => {
                // Coordinates and means are parallel, so both stay in order.
                variant(out, "means");
                means.coordinates.encode(out);
                out.reals(&means.means);
                out.text(match means.estimate {
                    MeanEstimate::Posterior => "posterior",
                    MeanEstimate::Frequentist => "frequentist",
                });
                out.text(&means.provider_id)
                    .text(&means.snapshot_id)
                    .text(&means.causal_contract_id)
                    .text(&means.rng_id);
            }
        }
    }
}

#[cfg(test)]
#[path = "claim_decision_programs_site_tests.rs"]
mod site_tests;
