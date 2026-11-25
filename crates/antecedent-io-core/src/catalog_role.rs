//! Catalog role names shared by a compiled study and a composition-bundle reference.
//!
//! Continuation roles are the payloads a compiled study stores beside its invariant.
//! Producer roles name the artifact identity a bundle reference must carry. The
//! strings are the role ids in `program.catalog` and in `program.body.<role>`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Source table, rebuilt and checked against its content digest.
pub const SOURCE_ROWS: &str = "source_rows";
/// Frozen score table.
pub const FROZEN_SCORES: &str = "frozen_scores";
/// Summary retained after rows were dropped.
pub const RETAINED_SUMMARY: &str = "retained_summary";
/// Callback declaration words. The live caller object is not stored.
pub const CALLBACK_DECLARATION: &str = "callback_declaration";
/// Callback output words. The live caller object is not stored.
pub const CALLBACK_OUTPUT: &str = "callback_output";
/// Posterior draw payload.
pub const POSTERIOR_DRAWS: &str = "posterior_draws";
/// Discovery statistics payload.
pub const DISCOVERY_STATISTICS: &str = "discovery_statistics";
/// Spill frame payload. An `ANTSPILL` file still loads through its own reader.
pub const SPILL: &str = "spill";
/// Opaque provider artifact bytes.
pub const PROVIDER_PAYLOAD: &str = "provider_payload";
/// Joint-law or propagation-chain artifact identity.
pub const PROPAGATION: &str = "propagation";
/// Evidence-ledger artifact identity.
pub const LEDGER: &str = "ledger";
/// Mechanism-composition artifact identity.
pub const COMPOSED_MODEL: &str = "composed_model";
/// Measurement-model artifact identity.
pub const MEASUREMENT: &str = "measurement";
/// Selective-discovery artifact identity.
pub const DISCOVERY: &str = "discovery";
/// Decision-reliability artifact identity.
pub const RELIABILITY: &str = "reliability";

/// Continuation payloads a compiled study stores as catalog roles.
pub const CONTINUATION_ROLES: [&str; 9] = [
    SOURCE_ROWS,
    FROZEN_SCORES,
    RETAINED_SUMMARY,
    CALLBACK_DECLARATION,
    CALLBACK_OUTPUT,
    POSTERIOR_DRAWS,
    DISCOVERY_STATISTICS,
    SPILL,
    PROVIDER_PAYLOAD,
];

/// Producer artifact identities a bundle reference may bind.
pub const PRODUCER_ROLES: [&str; 6] =
    [PROPAGATION, LEDGER, COMPOSED_MODEL, MEASUREMENT, DISCOVERY, RELIABILITY];

/// Whether `role` is a continuation or a producer role.
#[must_use]
pub fn is_role(role: &str) -> bool {
    is_continuation(role) || is_producer(role)
}

/// Whether `role` is a continuation payload.
#[must_use]
pub fn is_continuation(role: &str) -> bool {
    CONTINUATION_ROLES.contains(&role)
}

/// Whether `role` is a producer identity.
#[must_use]
pub fn is_producer(role: &str) -> bool {
    PRODUCER_ROLES.contains(&role)
}

/// One row of an IO kind table, and the catalog role that now carries it when
/// one does.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KindRow {
    /// Artifact kind string, or a legacy magic that is not a CAUSAL kind.
    pub kind: &'static str,
    /// Which inventory the row came from.
    pub table: &'static str,
    /// Continuation role, when this writer’s payload moves into the catalog.
    pub continuation: Option<&'static str>,
    /// Producer role, when a bundle reference binds this artifact’s identity.
    pub producer: Option<&'static str>,
}

/// Both IO kind tables, plus the legacy magics and the producer artifacts that
/// sit beside them. A row with neither role still loads through its existing reader.
pub const KIND_INVENTORY: &[KindRow] = &[
    row("adaptive_monitoring_contract_v1", "io-core", None, None),
    row("analysis_result", "io-core", None, None),
    row("callback_result_v1", "io-core", Some(CALLBACK_DECLARATION), None),
    row("causal_payload", "io-core", None, None),
    row("cohort_decision_source_v1", "io-core", None, None),
    row("compiled_study_v1", "io-core", None, None),
    row("compact_runtime_export_v1", "io-core", None, None),
    row("composition_verification_v1", "io-core", None, None),
    row("counterfactual_transport_v2", "io-core", None, None),
    row("cross_world_execution_v1", "io-core", None, None),
    row("evidence_ledger_v1", "io-core", None, Some(LEDGER)),
    row("external_callback_v1", "io-core", Some(CALLBACK_DECLARATION), None),
    row("external_column_claim_v1", "io-core", None, None),
    row("external_response_claim_v1", "io-core", None, None),
    row("frozen_scores_v1", "io-core", Some(FROZEN_SCORES), None),
    row("interface_contract_v1", "io-core", None, None),
    row("joint_distribution_v1", "io-core", None, None),
    row("joint_law_propagation_v1", "io-core", None, Some(PROPAGATION)),
    row("joint_mechanism_sensitivity", "io-core", None, None),
    row("measurement_model_v1", "io-core", None, Some(MEASUREMENT)),
    row("mechanism_composition_v1", "io-core", None, Some(COMPOSED_MODEL)),
    row("provider_result_v1", "io-core", Some(PROVIDER_PAYLOAD), None),
    row("provider_verification_receipt_v1", "io-core", None, None),
    row("recalc_receipt_v1", "io-core", None, None),
    row("scientific_bridge_v1", "io-core", None, None),
    row("selective_discovery_v1", "io-core", Some(DISCOVERY_STATISTICS), Some(DISCOVERY)),
    row("sensitivity_decision_v1", "io-core", None, None),
    row("staleness_baseline_v1", "io-core", None, None),
    row("temporal_counterfactual_v1", "io-core", None, None),
    row("temporal_counterfactual_general_v1", "io-core", None, None),
    row("transport_certificate", "io-core", None, None),
    row("transported_counterfactual_v1", "io-core", None, None),
    row("causal_posterior", "container-enum", Some(POSTERIOR_DRAWS), None),
    row("adaptive_inference_result_v1", "io", None, None),
    row("categorical_treatment_v1", "io", None, None),
    row("choice_inference_result_v1", "io", None, None),
    row("effect_constancy_v1", "io", None, None),
    row("latent_class_effects_v1", "io", None, None),
    row("learned_trial", "io", None, None),
    row("marginal_sensitivity_model", "io", None, None),
    row("mechanism_discrepancy_v1", "io", None, None),
    row("nonlinear_mediation_v1", "io", None, None),
    row("period_refresh_ledger_v1", "io", None, None),
    row("period_refresh_summary_v1", "io", Some(RETAINED_SUMMARY), None),
    row("predictive_assessment_v1", "io", None, None),
    row("sampled_counterfactual_transport_v1", "io", None, None),
    row("temporal_dependent_interval_internal", "io", None, None),
    row("temporal_initial_state_marginalization", "io", None, None),
    row("temporal_new_period_refresh", "io", None, None),
    row("vector_treatment_v1", "io", None, None),
    row("analysis_result.source_witness", "section", Some(SOURCE_ROWS), None),
    row("ANTSPILL", "magic", Some(SPILL), None),
    row("ANTCALLBACK1", "magic", None, None),
    row("ANTEPROV2", "magic", None, None),
    row("ANTSTALEBASE1", "magic", None, None),
    row("ANTPOLICYPROG", "magic", None, None),
    row("decision_reliability_v1", "design", None, Some(RELIABILITY)),
    row("callback_result_v1.output", "section", Some(CALLBACK_OUTPUT), None),
];

const fn row(
    kind: &'static str,
    table: &'static str,
    continuation: Option<&'static str>,
    producer: Option<&'static str>,
) -> KindRow {
    KindRow { kind, table, continuation, producer }
}
