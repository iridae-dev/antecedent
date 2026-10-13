//! Artifact kind strings for this crate.
//!
//! The words are the ones already written into manifests. The table exists so
//! a second artifact cannot reuse a discriminator, and so encode paths name
//! one static word.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) const ANALYSIS_RESULT: &str = "analysis_result";
pub(crate) const CAUSAL_PAYLOAD: &str = "causal_payload";
pub(crate) const COMPACT_RUNTIME_EXPORT: &str = "compact_runtime_export_v1";
pub(crate) const EVIDENCE_LEDGER: &str = "evidence_ledger_v1";
pub(crate) const EXTERNAL_CALLBACK: &str = "external_callback_v1";
pub(crate) const EXTERNAL_COLUMN_CLAIM: &str = "external_column_claim_v1";
pub(crate) const EXTERNAL_RESPONSE_CLAIM: &str = "external_response_claim_v1";
pub(crate) const FROZEN_SCORES: &str = "frozen_scores_v1";
pub(crate) const JOINT_DISTRIBUTION: &str = "joint_distribution_v1";
pub(crate) const JOINT_MECHANISM_SENSITIVITY: &str = "joint_mechanism_sensitivity";
pub(crate) const MEASUREMENT_MODEL: &str = "measurement_model_v1";
pub(crate) const PROVIDER_RESULT: &str = "provider_result_v1";
pub(crate) const PROVIDER_VERIFICATION_RECEIPT: &str = "provider_verification_receipt_v1";
pub(crate) const RECALC_RECEIPT: &str = "recalc_receipt_v1";
pub(crate) const SENSITIVITY_DECISION: &str = "sensitivity_decision_v1";
pub(crate) const TEMPORAL_COUNTERFACTUAL: &str = "temporal_counterfactual_v1";
pub(crate) const TRANSPORT_CERTIFICATE: &str = "transport_certificate";
pub(crate) const TRANSPORTED_COUNTERFACTUAL: &str = "transported_counterfactual_v1";

#[cfg(test)]
mod tests {
    use super::{
        ANALYSIS_RESULT, CAUSAL_PAYLOAD, COMPACT_RUNTIME_EXPORT, EVIDENCE_LEDGER, EXTERNAL_CALLBACK,
        EXTERNAL_COLUMN_CLAIM, EXTERNAL_RESPONSE_CLAIM, FROZEN_SCORES, JOINT_DISTRIBUTION,
        JOINT_MECHANISM_SENSITIVITY, MEASUREMENT_MODEL, PROVIDER_RESULT, PROVIDER_VERIFICATION_RECEIPT,
        RECALC_RECEIPT, SENSITIVITY_DECISION, TEMPORAL_COUNTERFACTUAL, TRANSPORT_CERTIFICATE,
        TRANSPORTED_COUNTERFACTUAL,
    };

    #[test]
    fn artifact_kinds_are_unique() {
        let mut kinds = vec![
            ANALYSIS_RESULT,
            CAUSAL_PAYLOAD,
            COMPACT_RUNTIME_EXPORT,
            EVIDENCE_LEDGER,
            EXTERNAL_CALLBACK,
            EXTERNAL_COLUMN_CLAIM,
            EXTERNAL_RESPONSE_CLAIM,
            FROZEN_SCORES,
            JOINT_DISTRIBUTION,
            JOINT_MECHANISM_SENSITIVITY,
            MEASUREMENT_MODEL,
            PROVIDER_RESULT,
            PROVIDER_VERIFICATION_RECEIPT,
            RECALC_RECEIPT,
            SENSITIVITY_DECISION,
            TEMPORAL_COUNTERFACTUAL,
            TRANSPORT_CERTIFICATE,
            TRANSPORTED_COUNTERFACTUAL,
            "categorical_treatment_v1",
            "effect_constancy_v1",
            "latent_class_effects_v1",
            "learned_trial",
            "marginal_sensitivity_model",
            "mechanism_composition_v1",
            "mechanism_discrepancy_v1",
            "nonlinear_mediation_v1",
            "temporal_dependent_interval_internal",
            "temporal_initial_state_marginalization",
            "temporal_new_period_refresh",
            "vector_treatment_v1",
        ];
        let total = kinds.len();
        kinds.sort_unstable();
        kinds.dedup();
        assert_eq!(kinds.len(), total, "duplicate artifact kind: {kinds:?}");
    }
}
