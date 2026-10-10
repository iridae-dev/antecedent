//! 2.3 B3: the source-target mechanism discrepancy diagnostic and its artifact.
//!
//! The facade routes and owns no scientific rule of its own:
//!
//! * [`summarize_sample`] reduces one population's raw rows to the sufficient statistics the
//!   Wald test needs (`n`, `X'X`, `X'y`, `y'y`) through the estimate crate;
//! * [`run_mechanism_discrepancy`] runs the diagnostic on two such summaries and seals it into
//!   a [`MechanismDiscrepancyArtifact`] (shared row identities refuse before anything is
//!   computed, exactly as the raw-data entry of the core does);
//! * [`consume_mechanism_discrepancy`] consumes artifact bytes by recomputation and refuses a
//!   resealed mutation.
//!
//! The diagnostic is a linear-Gaussian-mean Wald test of a single node's mechanism. A
//! non-rejection never certifies invariance (`non_rejection_certifies_invariance` is always
//! `false`), it informs only a selection node on the compared node, and its Type I error and
//! power are unmeasured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

pub use antecedent_estimate::EstimationError;
use antecedent_estimate::mechanism_discrepancy::summarize_population;
pub use antecedent_estimate::mechanism_discrepancy::{
    MECHANISM_DISCREPANCY_ALIGNMENT, MECHANISM_DISCREPANCY_CAVEAT,
    MECHANISM_DISCREPANCY_DEPENDENCE, MECHANISM_DISCREPANCY_INFERENCE_CLAIM,
    MECHANISM_DISCREPANCY_INTERCEPT, MECHANISM_DISCREPANCY_MAX_PARENTS, MECHANISM_DISCREPANCY_NULL,
    MECHANISM_DISCREPANCY_POWER_CAVEAT, MechanismMeasurement, MechanismSummary, ParentSpec,
    PopulationSample,
};
pub use antecedent_io::mechanism_discrepancy_artifact::{
    MeasurementWire, MechanismDiscrepancyArtifact, MechanismDiscrepancyArtifactError,
    MechanismDiscrepancyIdentity, MechanismDiscrepancyReportWire, MechanismDiscrepancyRequestWire,
    ParentWire, PopulationWire,
};

/// One refusal as plain data: registered reason code, namespaced detail and explanation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismDiscrepancyRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced `mechanism_discrepancy.*` detail.
    pub detail: String,
    /// What was refused.
    pub message: String,
}

impl MechanismDiscrepancyRefusal {
    /// The refusal of an artifact error: its own registered refusal when it carries one,
    /// otherwise an invalid-artifact refusal with the error text.
    #[must_use]
    pub fn from_artifact_error(error: &MechanismDiscrepancyArtifactError) -> Self {
        match error.refusal() {
            Some((code, detail, message)) => Self { code, detail, message },
            None => Self {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "mechanism_discrepancy.invalid_artifact".to_owned(),
                message: error.to_string(),
            },
        }
    }

    /// The refusal of a core estimation error (`<detail>: <message>`).
    #[must_use]
    pub fn from_estimation(error: &EstimationError) -> Self {
        match error {
            EstimationError::Refused { code, message }
            | EstimationError::RefusedWithFields { code, message, .. } => {
                let (detail, rest) = message
                    .split_once(": ")
                    .unwrap_or(("mechanism_discrepancy.invalid_request", message.as_str()));
                Self { code, detail: detail.to_owned(), message: rest.to_owned() }
            }
            other => Self {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "mechanism_discrepancy.invalid_request".to_owned(),
                message: other.to_string(),
            },
        }
    }
}

/// Sufficient statistics of one population's raw rows.
///
/// # Errors
///
/// The core refusals of [`summarize_population`] (`too_many_parents`, `sample_too_small`,
/// `row_count_mismatch`, `non_finite_value`).
pub fn summarize_sample(sample: &PopulationSample) -> Result<MechanismSummary, EstimationError> {
    summarize_population(sample)
}

/// The wire declaration of one population's summary.
#[must_use]
pub fn population_wire(summary: &MechanismSummary) -> PopulationWire {
    PopulationWire {
        label: summary.label.clone(),
        measurement: MeasurementWire {
            node: summary.measurement.node.clone(),
            node_unit: summary.measurement.node_unit.clone(),
            parents: summary
                .measurement
                .parents
                .iter()
                .map(|p| ParentWire { name: p.name.clone(), unit: p.unit.clone() })
                .collect(),
            protocol_id: summary.measurement.protocol_id.clone(),
        },
        n: summary.n as u64,
        xtx: summary.xtx.clone(),
        xty: summary.xty.clone(),
        yty: summary.yty,
    }
}

/// Run the diagnostic on a declared request and seal it into an artifact.
///
/// `source_ids` and `target_ids` are optional row identities (empty when not supplied) used
/// only to detect shared units: a common id refuses as
/// `mechanism_discrepancy.dependence_unknown` before anything is computed.
///
/// # Errors
///
/// The core diagnostic's refusals (incomparable measurements, unknown or shared dependence,
/// invalid level or power, rank-deficient or degenerate designs).
pub fn run_mechanism_discrepancy(
    request: &MechanismDiscrepancyRequestWire,
    source_ids: &[String],
    target_ids: &[String],
) -> Result<MechanismDiscrepancyArtifact, MechanismDiscrepancyArtifactError> {
    if !source_ids.is_empty() && !target_ids.is_empty() {
        let ids: BTreeSet<&str> = source_ids.iter().map(String::as_str).collect();
        if target_ids.iter().any(|id| ids.contains(id.as_str())) {
            return Err(MechanismDiscrepancyArtifactError::Refused {
                code: antecedent_core::reason_code!("route_not_supported"),
                message: "mechanism_discrepancy.dependence_unknown: source and target share \
                          units, so their estimates are not independent"
                    .to_owned(),
            });
        }
    }
    MechanismDiscrepancyArtifact::seal(request)
}

/// Consume artifact bytes by recomputation; `expected` is an identity the consumer retained
/// independently of the bytes.
///
/// # Errors
///
/// Corruption, another major version, unsupported semantics, a changed identity, a stored
/// result that does not replay or a core refusal.
pub fn consume_mechanism_discrepancy(
    bytes: &[u8],
    expected: Option<&MechanismDiscrepancyIdentity>,
) -> Result<MechanismDiscrepancyArtifact, MechanismDiscrepancyArtifactError> {
    MechanismDiscrepancyArtifact::from_bytes(bytes, expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire(label: &str, yty: f64) -> PopulationWire {
        PopulationWire {
            label: label.to_owned(),
            measurement: MeasurementWire {
                node: "V".to_owned(),
                node_unit: "mg".to_owned(),
                parents: vec![ParentWire { name: "x".to_owned(), unit: "cm".to_owned() }],
                protocol_id: "protocol-1".to_owned(),
            },
            n: 4,
            xtx: vec![4.0, 6.0, 6.0, 14.0],
            xty: vec![10.0, 19.0],
            yty,
        }
    }

    #[test]
    fn b3_discrepancy_shared_row_identities_refuse_before_the_fit() {
        let request = MechanismDiscrepancyRequestWire {
            source: wire("source", 30.0),
            target: wire("target", 30.0),
            compare_intercept: true,
            alpha: 0.05,
            power: 0.8,
            dependence: "independent".to_owned(),
        };
        let ids = |names: &[&str]| names.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        let error =
            run_mechanism_discrepancy(&request, &ids(&["a", "b"]), &ids(&["b", "c"])).unwrap_err();
        let refusal = MechanismDiscrepancyRefusal::from_artifact_error(&error);
        assert_eq!(refusal.code, "route_not_supported");
        assert_eq!(refusal.detail, "mechanism_discrepancy.dependence_unknown");
    }
}
