//! Artifact kind strings for this crate.
//!
//! The words match the manifests already shipped. `antecedent-io-core` lists
//! the same words in its uniqueness check.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) const CATEGORICAL_TREATMENT: &str = "categorical_treatment_v1";
pub(crate) const EFFECT_CONSTANCY: &str = "effect_constancy_v1";
pub(crate) const LATENT_CLASS_EFFECTS: &str = "latent_class_effects_v1";
pub(crate) const LEARNED_TRIAL: &str = "learned_trial";
pub(crate) const MARGINAL_SENSITIVITY_MODEL: &str = "marginal_sensitivity_model";
pub(crate) const MECHANISM_DISCREPANCY: &str = "mechanism_discrepancy_v1";
pub(crate) const NONLINEAR_MEDIATION: &str = "nonlinear_mediation_v1";
pub(crate) const SAMPLED_COUNTERFACTUAL_TRANSPORT: &str = "sampled_counterfactual_transport_v1";
pub(crate) const TEMPORAL_DEPENDENT_INTERVAL: &str = "temporal_dependent_interval_internal";
pub(crate) const TEMPORAL_INITIAL_STATE: &str = "temporal_initial_state_marginalization";
pub(crate) const TEMPORAL_NEW_PERIOD_REFRESH: &str = "temporal_new_period_refresh";
pub(crate) const VECTOR_TREATMENT: &str = "vector_treatment_v1";

#[cfg(test)]
mod tests {
    use super::{
        CATEGORICAL_TREATMENT, EFFECT_CONSTANCY, LATENT_CLASS_EFFECTS, LEARNED_TRIAL,
        MARGINAL_SENSITIVITY_MODEL, MECHANISM_DISCREPANCY, NONLINEAR_MEDIATION,
        SAMPLED_COUNTERFACTUAL_TRANSPORT, TEMPORAL_DEPENDENT_INTERVAL, TEMPORAL_INITIAL_STATE,
        TEMPORAL_NEW_PERIOD_REFRESH, VECTOR_TREATMENT,
    };

    #[test]
    fn artifact_kinds_are_unique() {
        let mut kinds = vec![
            CATEGORICAL_TREATMENT,
            EFFECT_CONSTANCY,
            LATENT_CLASS_EFFECTS,
            LEARNED_TRIAL,
            MARGINAL_SENSITIVITY_MODEL,
            MECHANISM_DISCREPANCY,
            NONLINEAR_MEDIATION,
            SAMPLED_COUNTERFACTUAL_TRANSPORT,
            TEMPORAL_DEPENDENT_INTERVAL,
            TEMPORAL_INITIAL_STATE,
            TEMPORAL_NEW_PERIOD_REFRESH,
            VECTOR_TREATMENT,
        ];
        let total = kinds.len();
        kinds.sort_unstable();
        kinds.dedup();
        assert_eq!(kinds.len(), total, "duplicate artifact kind: {kinds:?}");
    }
}
