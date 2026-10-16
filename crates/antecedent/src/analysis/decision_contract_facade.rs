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
//! # Transported responses
//!
//! A value carried to a target population reaches the decision as a
//! [`TransportedResponse`]: [`BoundDecisionContract::evaluate_transported`]
//! applies the contract's admissibility rules (support rules, declared
//! exclusions, the uncertainty requirement) and then the one transported
//! decision evaluation, so the result carries the transport receipt and the
//! bound contract identity and is never presented as a decision on native
//! target data. A transported point answers an affine expected utility and
//! nothing else; a law, a probability, a quantile, a hard constraint and an
//! interval are refused with their own reasons.
//!
//! **Refresh.** A new outcome-only sample of the source population is a new
//! response: the decision is re-evaluated from it, keeping the declared
//! contract, support rules and policy, and recomputing every value and the
//! decision identity. A response about another population, quantity or arm is
//! refused by the coordinate check, never reinterpreted. The decision is not
//! refreshed in place.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ExternalRefusal;
use antecedent_core::{CanonicalWriter, SupportStatus};
use antecedent_design::decision_contract::UncertaintyKind;
use antecedent_io::error::IoError;

use super::contract::CausalContract;

pub use antecedent_design::composition_boundary::SupportPolicy;
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
pub use antecedent_design::decision_transported_source::{
    TRANSPORTED_RESPONSE_INFERENCE, TransportDeclaration, TransportReceipt, TransportedDecision,
    TransportedDemand, TransportedRefusal, TransportedRepresentation, TransportedResponse,
    TransportedRoute, TransportedValue, contract_needs_law, evaluate_transported_decision,
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
    #[allow(
        clippy::result_large_err,
        reason = "DecisionFacadeRefusal carries the decision layer's structured refusal by value on a cold once-per-declaration path; boxing it would change the public error contract"
    )]
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
    #[allow(
        clippy::result_large_err,
        reason = "DecisionFacadeRefusal carries the decision layer's structured refusal by value on a cold once-per-declaration path; boxing it would change the public error contract"
    )]
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
    #[allow(
        clippy::result_large_err,
        reason = "DecisionFacadeRefusal carries the decision layer's structured refusal by value on a cold once-per-declaration path; boxing it would change the public error contract"
    )]
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

    /// Evaluate the contract on a transported response.
    ///
    /// The contract's admissibility rules apply first: a declared exclusion
    /// removes its action, a support rule (or the default weakest support) the
    /// response's receipt does not meet refuses, and an uncertainty requirement
    /// a point cannot meet refuses. The transported evaluation then runs exactly
    /// as [`evaluate_transported_decision`]; the result is bound to this
    /// contract's identity.
    ///
    /// # Errors
    /// A requirement of the rules the transported point cannot satisfy, every
    /// refusal of [`evaluate_transported_decision`] (a law or interval the value
    /// cannot supply, a support label below the policy, a coordinate outside the
    /// target population) with its registered code and detail.
    #[allow(
        clippy::result_large_err,
        reason = "DecisionFacadeRefusal carries the decision layer's structured refusal by value on a cold once-per-declaration path; boxing it would change the public error contract"
    )]
    pub fn evaluate_transported(
        &self,
        response: &TransportedResponse,
        demand: TransportedDemand,
        policy: &SupportPolicy,
    ) -> Result<BoundTransportedDecision, DecisionFacadeRefusal> {
        let rules = &self.decision.rules;
        if !rules.uncertainty.satisfied_by(UncertaintyKind::Point) {
            return Err(ExternalRefusal::new(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                "transported_decision",
                DETAIL_UNCERTAINTY_REQUIREMENT,
            )
            .with_offending(format!("{:?}", rules.uncertainty))
            .with_expected("the representation the contract requires")
            .with_supplied(TRANSPORTED_RESPONSE_INFERENCE)
            .with_remedy(
                "a transported value is a point; relax the uncertainty requirement or supply \
                 native target evidence of the required representation",
            ));
        }
        let supplied = response.receipt().support_status().ok_or_else(|| {
            TransportedRefusal::SupportLabelInflated(response.receipt().support.clone())
                .to_refusal()
        })?;
        let mut contract = self.decision.contract.clone();
        contract.actions.retain(|a| !rules.declared_exclusions.iter().any(|e| e.action_id == a.id));
        if contract.actions.len() < 2 {
            return Err(ExternalRefusal::new(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                "transported_decision",
                DETAIL_EXCLUSIONS,
            )
            .with_expected("at least two admissible actions")
            .with_supplied(contract.actions.len().to_string())
            .with_remedy(
                "a decision compares at least two actions; remove a declared exclusion or \
                 declare another action",
            ));
        }
        for action in &contract.actions {
            for input in 0..action.inputs.len() {
                if let Some(required) = rules.weakest_allowed(&action.id, input) {
                    if supplied.severity() > required.severity() {
                        return Err(support_rule_refusal(&action.id, supplied, required));
                    }
                }
            }
        }
        let decision = evaluate_transported_decision(&contract, response, demand, policy)
            .map_err(|e| e.to_refusal())?;
        let mut writer = CanonicalWriter::new(TAG_BOUND_TRANSPORTED);
        writer.text(&self.identity).text(&decision.identity);
        Ok(BoundTransportedDecision {
            decision,
            bound_identity: self.identity.clone(),
            causal_target: self.causal_target.clone(),
            causal_identification: self.causal_identification.clone(),
            identity: writer.finish_hex(),
        })
    }
}

const TAG_BOUND_TRANSPORTED: &str = "antecedent.decision_contract.bound_transported";
/// Detail: the contract requires an uncertainty representation a point cannot supply.
pub const DETAIL_UNCERTAINTY_REQUIREMENT: &str = "transported_decision.uncertainty_requirement";
/// Detail: declared exclusions leave fewer than two actions to compare.
pub const DETAIL_EXCLUSIONS: &str = "transported_decision.exclusions_leave_no_comparison";
/// Detail: a support rule of the contract is stronger than the transported label.
pub const DETAIL_SUPPORT_RULE: &str = "transported_decision.support_rule_unmet";

fn support_rule_refusal(
    action: &str,
    supplied: SupportStatus,
    required: SupportStatus,
) -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("transport_support_failure"),
        "transported_decision",
        DETAIL_SUPPORT_RULE,
    )
    .with_offending(action.to_owned())
    .with_expected(required.as_str().to_owned())
    .with_supplied(supplied.as_str().to_owned())
    .with_remedy(
        "a transported value is labelled extrapolative at best; weaken the support rule for this \
         action input explicitly or decide on native target evidence",
    )
}

/// A decision on a transported response, bound to the causal contract.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundTransportedDecision {
    decision: TransportedDecision,
    bound_identity: String,
    causal_target: String,
    causal_identification: String,
    identity: String,
}

/// What an inspection of a transported decision shows.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedDecisionInspection {
    /// Always `point_only`.
    pub inference: &'static str,
    /// Always `false`: the value is not native target data.
    pub native_target_data: bool,
    /// The route that carried the value.
    pub route: &'static str,
    /// Populations the value was learned in.
    pub source_populations: Vec<String>,
    /// Population the value was carried to.
    pub target_population: String,
    /// Support label of the value.
    pub support: String,
    /// `(action, expected utility)` of every evaluated action.
    pub actions: Vec<(String, f64)>,
    /// The verdict, rendered.
    pub verdict: String,
    /// Assumptions of the decision followed by the transport's.
    pub assumptions: Vec<String>,
}

impl BoundTransportedDecision {
    /// The decision with its transport receipt.
    #[must_use]
    pub const fn decision(&self) -> &TransportedDecision {
        &self.decision
    }

    /// The transport receipt the value travelled with.
    #[must_use]
    pub const fn receipt(&self) -> &TransportReceipt {
        &self.decision.receipt
    }

    /// Identity of the bound contract the decision was made under.
    #[must_use]
    pub fn bound_identity(&self) -> &str {
        &self.bound_identity
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

    /// Canonical identity of the bound transported decision: the bound
    /// contract, the transport receipt and the lineage of the value.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Plain-data summary: the transport is named beside every number.
    #[must_use]
    pub fn inspect(&self) -> TransportedDecisionInspection {
        let receipt = &self.decision.receipt;
        TransportedDecisionInspection {
            inference: self.decision.inference,
            native_target_data: false,
            route: receipt.route.as_str(),
            source_populations: receipt.source_populations.clone(),
            target_population: receipt.target_population.clone(),
            support: receipt.support.clone(),
            actions: self
                .decision
                .decision
                .outcomes
                .iter()
                .map(|o| (o.id.clone(), o.expected_utility))
                .collect(),
            verdict: format!("{:?}", self.decision.decision.verdict),
            assumptions: self.decision.assumptions.clone(),
        }
    }
}
