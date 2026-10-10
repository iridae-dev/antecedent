//! The decision contract beside [`CausalContract`].
//!
//! A [`CausalContract`] says what a study identifies and estimates. An
//! [`AdmissibleDecisionContract`] says what a decision does with such claims:
//! stable semantic actions, closed utilities, hard constraints, criterion,
//! target population and horizon, the structural policy, and the admissibility,
//! support and uncertainty requirements. [`BoundDecisionContract`] binds the two
//! by identity so a decision result names the causal target and identification
//! it was asked of, and so a change to either changes the bound identity.
//!
//! The binding does not read the causal contract's quantities: the scientific
//! match between an action's inputs and a claim is checked when the claim is
//! evaluated, and every refusal keeps its registered code and detail.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExternalRefusal;
use antecedent_io::error::IoError;

use super::contract::CausalContract;

pub use antecedent_design::decision_adapters::{
    AdaptedClaims, AdaptedDecision, AdapterError, ClaimKind as DecisionClaimKind, ClaimProbability,
    IdentifiedActionRange, IdentifiedSetDecision, IdentifiedUtility, IdentifiedVerdict,
    SuppliedClaim, adapt_finite_scenarios, adapt_graph_dependent_claims, adapt_point_claim,
    adapt_weighted_graph_atoms, evaluate_adapted, evaluate_identified_sets, utility_interval,
};
pub use antecedent_design::decision_contract::{
    AdmissibilityError, AdmissibilityRules, AdmissibleDecisionContract,
    DeclaredExclusion as DecisionDeclaredExclusion, SupportRule as DecisionSupportRule,
    UncertaintyKind as DecisionUncertaintyKind, UncertaintyRequirement,
};
pub use antecedent_design::decision_robust_artifact::{
    AdmissibleContractArtifact, ExternalCallbackReceipt, ExternalTrustLimit, ReplayReceipt,
    RobustResultArtifact,
};
pub use antecedent_design::decision_robustness::{
    ActionRobustness, AtomSupport, ClaimProfile, InputSupport, RobustDecisionResult, RobustVerdict,
    RobustnessError, SupportShortfall, assess_robustness, evaluate_robust,
};

/// Structured refusal of the decision facade. It carries the registered code and
/// namespaced detail of the underlying error unchanged.
pub type DecisionFacadeRefusal = ExternalRefusal;

/// An admissible decision contract bound to the causal contract it consumes.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundDecisionContract {
    causal_target: String,
    causal_identification: String,
    decision: AdmissibleDecisionContract,
    identity: String,
}

impl BoundDecisionContract {
    /// Bind `decision` to the target and identification identities of `causal`.
    ///
    /// # Errors
    /// An invalid decision contract or rule set refuses with the exact structured
    /// refusal of its validation.
    // The refusal is the cold path of a once-per-declaration check; boxing would not pay.
    #[allow(clippy::result_large_err)]
    pub fn bind(
        causal: &CausalContract,
        decision: AdmissibleDecisionContract,
    ) -> Result<Self, DecisionFacadeRefusal> {
        let decision_identity = decision.identity().map_err(|e| e.to_refusal())?;
        let causal_target = causal.identities.target.to_hex();
        let causal_identification = causal.identities.identification.to_hex();
        let identity = antecedent_design::decision_robust_artifact::bound_decision_identity(
            &decision_identity,
            &causal_target,
            &causal_identification,
        );
        Ok(Self { causal_target, causal_identification, decision, identity })
    }

    /// The admissible decision contract.
    #[must_use]
    pub fn decision(&self) -> &AdmissibleDecisionContract {
        &self.decision
    }

    /// Canonical identity of the decision bound to the causal target and
    /// identification.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Target identity of the causal contract (lowercase hex).
    #[must_use]
    pub fn causal_target(&self) -> &str {
        &self.causal_target
    }

    /// Identification identity of the causal contract (lowercase hex).
    #[must_use]
    pub fn causal_identification(&self) -> &str {
        &self.causal_identification
    }

    /// Evaluate adapted claims (point, graph-dependent, weighted-atom or finite
    /// scenario claims) under the declared structural policy.
    ///
    /// # Errors
    /// A policy different from the one the claims were adapted under, and any
    /// refusal of the structural evaluation or robustness assessment.
    #[allow(clippy::result_large_err)]
    pub fn evaluate_claims(
        &self,
        adapted: &AdaptedClaims,
    ) -> Result<AdaptedDecision, DecisionFacadeRefusal> {
        evaluate_adapted(&self.decision, adapted).map_err(|e| e.to_refusal())
    }

    /// Decide over per-action identified sets (partial identification).
    ///
    /// # Errors
    /// Any refusal of [`evaluate_identified_sets`].
    #[allow(clippy::result_large_err)]
    pub fn evaluate_identified_sets(
        &self,
        utilities: &[IdentifiedUtility],
    ) -> Result<IdentifiedSetDecision, DecisionFacadeRefusal> {
        evaluate_identified_sets(&self.decision, utilities).map_err(|e| e.to_refusal())
    }

    /// The durable contract artifact: base contract plus every rule.
    ///
    /// # Errors
    /// An invalid contract refuses.
    pub fn contract_artifact(&self) -> Result<AdmissibleContractArtifact, IoError> {
        AdmissibleContractArtifact::new(self.decision.clone())
    }

    /// Seal a robust result with the receipts of any external callback behind it.
    ///
    /// # Errors
    /// A result from another contract, or an invalid receipt, refuses.
    pub fn seal_result(
        &self,
        decision: &AdaptedDecision,
        adapted: &AdaptedClaims,
        receipts: Vec<ExternalCallbackReceipt>,
    ) -> Result<RobustResultArtifact, IoError> {
        RobustResultArtifact::new(
            &self.decision,
            decision.robust.clone(),
            &adapted.profile,
            &adapted.atoms,
            receipts,
        )
    }
}
