//! Plain-data sections of the analysis-result contract.
//!
//! The checked lowering payloads and the four reasoning slots carry no
//! estimator type, so identity digests and structural certificates below the
//! estimators can name them. [`contract_section`] in `antecedent-io` re-exports
//! them with the claim and calibration sections and the consumer.
//!
//! [`contract_section`]: https://docs.rs/antecedent-io
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;
use crate::error::IoError;
use crate::identity::payload_digest;

/// Semantic choices retained by the checked AIPW lowering.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CheckedAipwLoweringWire {
    /// Lowering payload format.
    pub format: u16,
    /// Selected causal functional expression id.
    pub functional: u32,
    /// Treatment variable id.
    pub treatment: u32,
    /// Outcome variable id.
    pub outcome: u32,
    /// Adjustment variable ids in selected-estimand order.
    pub adjustment: Vec<u32>,
    /// Target population name (`all_observed` for this checked route).
    pub population: String,
    /// Licensed procedure: `cross_fitted_logistic_ols` under an untrimmed overlap
    /// policy, `trimmed_logistic_ols` (full-sample common-support refit) under a
    /// trimming one.
    pub procedure: String,
    /// Cross-fitting folds (zero for the trimmed procedure).
    pub folds: u64,
    /// Analytic standard-error kind.
    pub se_kind: String,
    /// Lag for lagged analytic standard-error methods, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub se_lag: Option<u64>,
    /// Bootstrap replicate count (zero means disabled).
    pub bootstrap_replicates: u32,
    /// Propensity trim threshold bits of the retained overlap policy; present
    /// exactly for the `trimmed_logistic_ols` procedure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trim_bits: Option<u64>,
}

/// Data-dependent complete-case rows bound to a checked AIPW lowering.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CheckedAipwRowsWire {
    /// Row-binding payload format.
    pub format: u16,
    /// Source row ids retained by the checked estimator, in source order.
    pub rows: Vec<u64>,
    /// Data snapshot identity these rows came from.
    pub data_snapshot: [u8; 32],
}

/// Semantic roles, procedure, weak-instrument policy, and complete-case scope
/// retained by a checked prepared IV operation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CheckedIvLoweringWire {
    /// Lowering payload format.
    pub format: u16,
    /// Selected causal Wald-ratio functional expression id.
    pub functional: u32,
    /// Endogenous treatment variable id.
    pub treatment: u32,
    /// Outcome variable id.
    pub outcome: u32,
    /// Single instrument variable id.
    pub instrument: u32,
    /// Exogenous adjustment variable ids in selected-estimand order.
    pub adjustment: Vec<u32>,
    /// Queried active treatment value bits.
    pub active_bits: u64,
    /// Queried control treatment value bits.
    pub control_bits: u64,
    /// Instrument's active arm value bits.
    pub instrument_active_bits: u64,
    /// Instrument's control arm value bits.
    pub instrument_control_bits: u64,
    /// `wald` or `two_stage_least_squares`.
    pub procedure: String,
    /// Analytic standard-error method selected by the prepared operation.
    pub se_kind: String,
    /// Weak-instrument confidence-set route, including an explicit withheld state.
    pub weak_instrument_uncertainty: String,
    /// Number of complete-case observations in the prepared design.
    pub complete_case_rows: u64,
}

/// Hash a checked AIPW row binding for inclusion in contract identities.
///
/// # Errors
///
/// CBOR encoding failure.
pub fn checked_aipw_rows_digest(rows: &CheckedAipwRowsWire) -> Result<[u8; 32], IoError> {
    Ok(payload_digest("analysis_result.checked_aipw_rows", &to_cbor(rows)?))
}

/// Optional slot: exactly one of `value` or `unavailable` is required.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SlotSectionWire<T> {
    /// Present value.
    pub value: Option<T>,
    /// Stable unavailable reason.
    pub unavailable: Option<String>,
}

/// Compact identification slot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IdentificationSlotWire {
    /// Identification status.
    pub status: String,
    /// Identified mass.
    pub identified_mass: f64,
    /// Unidentified mass.
    pub unidentified_mass: f64,
    /// Unevaluable mass.
    pub unevaluable_mass: f64,
    /// Incomplete-search mass.
    pub incomplete_search_mass: f64,
    /// Whether reported mass covers the full class.
    pub full_mass_scope: bool,
    /// Search was capped before a determination.
    pub search_capped: bool,
    /// What the masses measure (`posterior_probability`,
    /// `completion_enumeration`, `caller_supplied_class_prior`). Enumeration
    /// weights are not probabilities; absent for a single identified atom.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_basis: Option<String>,
}

/// Compact support slot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SupportSlotWire {
    /// Matrix evidence status.
    pub matrix_status: String,
    /// Matrix coordinate, when classified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix_coordinate: Option<String>,
    /// Empirical support, or `unavailable:<reason>`.
    pub empirical: String,
}

/// One uncertainty component.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UncertaintyComponentWire {
    /// Source name.
    pub source: String,
    /// Reported target.
    pub target: String,
    /// Omitted / unresolved.
    pub omitted: bool,
}

/// Compact uncertainty slot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UncertaintySlotWire {
    /// Declared components.
    pub components: Vec<UncertaintyComponentWire>,
}

/// Compact obligation record (full assumption encodings stay on the body).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObligationSectionWire {
    /// Stable obligation id.
    pub id: String,
    /// Scope name.
    pub scope: String,
    /// Kind name.
    pub kind: String,
    /// Assumption status.
    pub status: String,
}

/// Compact assumption slot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AssumptionSlotWire {
    /// Scoped obligations.
    pub obligations: Vec<ObligationSectionWire>,
}

/// Four reasoning slots.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReasoningSectionWire {
    /// Identification.
    pub identification: SlotSectionWire<IdentificationSlotWire>,
    /// Support.
    pub support: SlotSectionWire<SupportSlotWire>,
    /// Uncertainty.
    pub uncertainty: SlotSectionWire<UncertaintySlotWire>,
    /// Assumptions.
    pub assumptions: SlotSectionWire<AssumptionSlotWire>,
}
