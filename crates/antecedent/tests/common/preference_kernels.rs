//! The trade-off kernels a test calls by hand to compare with the compiled route.
//!
//! The facade's direct decision functions are crate-internal: the compiled route is the way in.
//! A test that checks the compiled route against a hand computation calls the design crate's
//! kernels here, exactly as the crate-internal layer does, so both sides of the comparison stay
//! visible.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::native_claims::{
    NativeDecisionInput, NativeDecisionSource, NativeResponseClaim,
};
use antecedent::analysis::preference_decision::{
    AcceptedPreferences, PreferenceContract, PreferenceContractParts, PreferenceEvaluation,
    PreferenceEvaluationOptions, PreferenceMeasure, StructuralPreferenceEvaluation,
    source_requirement,
};
use antecedent::contract::ExternalRefusal;
use antecedent::design::decision_adapters::AdaptedClaims;
use antecedent_design::preference_evaluation::{
    decide_preferences, decide_preferences_on_means, decide_preferences_structural,
};

/// A decision across the preference set together with the analysis input it read.
pub struct ByHand {
    /// What the claim handed the decision.
    pub input: NativeDecisionInput,
    /// The verdict across the preference set.
    pub evaluation: PreferenceEvaluation,
}

/// Decide `contract` on a mean source (exact route) or an aligned joint law (either route).
pub fn decide(
    contract: &PreferenceContract,
    source: &NativeDecisionSource,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
) -> Result<PreferenceEvaluation, ExternalRefusal> {
    match source {
        NativeDecisionSource::Mean(means) => {
            decide_preferences_on_means(contract, means, measure, options)
        }
        NativeDecisionSource::JointLaw(law) => decide_preferences(contract, law, measure, options),
    }
    .map_err(|e| e.to_refusal())
}

/// Ask `claim` for the representation `contract` needs, then decide on it.
pub fn decide_claim(
    contract: &PreferenceContract,
    claim: &NativeResponseClaim,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
) -> Result<ByHand, ExternalRefusal> {
    let requirement = source_requirement(contract)?;
    let input = claim.decision_source(&requirement)?;
    let evaluation = decide(contract, &input.source, measure, options)?;
    Ok(ByHand { input, evaluation })
}

/// Decide with accepted preferences: the contract over the accepted model, with the accepted
/// measure when there is one.
pub fn decide_accepted(
    accepted: &AcceptedPreferences,
    parts: PreferenceContractParts,
    source: &NativeDecisionSource,
    options: PreferenceEvaluationOptions,
) -> Result<PreferenceEvaluation, ExternalRefusal> {
    let contract = accepted.contract(parts);
    decide(&contract, source, accepted.measure(), options)
}

/// Decide `contract` across adapted structures and its preference set together.
pub fn decide_across_structures(
    contract: &PreferenceContract,
    claims: &AdaptedClaims,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
) -> Result<StructuralPreferenceEvaluation, ExternalRefusal> {
    assert_eq!(claims.policy, contract.structural_policy, "claims adapted under the policy");
    decide_preferences_structural(contract, &claims.atoms, measure, options)
        .map_err(|e| e.to_refusal())
}
