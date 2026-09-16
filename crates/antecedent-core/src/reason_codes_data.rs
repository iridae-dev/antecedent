//! Generated from `parity/reason_codes.toml`. Do not edit.

#![cfg_attr(rustfmt, rustfmt::skip)]

/// Every registered reason code.
pub const REASON_CODES: &[&str] = &[
    "attested_not_reverifiable",
    "boundary_record",
    "calibration_basis_missing",
    "cancelled_no_claim",
    "construction_not_licensed",
    "data_modality_not_licensed",
    "estimator_grid_not_measured",
    "identification_not_measured",
    "interval_level_not_measured",
    "mechanism_fit_not_converged",
    "no_interval_reported",
    "not_executed",
    "option_not_applicable",
    "population_not_estimable",
    "posterior_draws_below_measured",
    "prior_transfer_not_hydrated",
    "resampling_replicates_below_measured",
    "row_weights_bound_to_snapshot",
    "sample_size_outside_measured_range",
    "stage_stream_unavailable",
    "treatment_support_too_discrete",
    "unidentified_mass_above_measured",
    "validators_not_applicable",
];

/// Codes whose `applies_to` includes `runtime_refusal`.
pub const RUNTIME_REFUSAL_CODES: &[&str] = &[
    "attested_not_reverifiable",
    "cancelled_no_claim",
    "construction_not_licensed",
    "data_modality_not_licensed",
    "mechanism_fit_not_converged",
    "not_executed",
    "option_not_applicable",
    "population_not_estimable",
    "prior_transfer_not_hydrated",
    "row_weights_bound_to_snapshot",
    "stage_stream_unavailable",
    "treatment_support_too_discrete",
    "validators_not_applicable",
];
