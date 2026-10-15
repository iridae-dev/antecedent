//! The calibration key of a composed interval, and its resolution against the
//! shipped coverage records.
//!
//! A composed interval is licensed only by a coverage record the registry
//! holds for exactly its composed coordinate and construction. The caller
//! supplies no record: the key is derived from the sealed claim, and the
//! resolver answers whether a currently attesting record governs it. The
//! calibration harness binds the same key, so a record describes exactly the
//! construction a claim reports.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::PropagationDependence;
use antecedent_core::CalibrationBasis;
use antecedent_io::calibration::{
    CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire, calibration_slot,
};

/// Query axis of every composed-interval record.
pub const COMPOSED_QUERY: &str = "ComposedQuantity";
/// Nominal level the composed constructions are measured at.
pub const COMPOSED_LEVEL: f64 = 0.95;
/// Wire name of the normal approximation to the propagated variance.
pub const COMPOSED_NORMAL_CONSTRUCTION: &str = "normal_approximation";

/// The stable label of a dependence declaration in the calibration key.
#[must_use]
pub fn dependence_label(dependence: &PropagationDependence) -> &'static str {
    match dependence {
        PropagationDependence::DeclaredIndependent => "declared_independent_distinct_evidence",
        PropagationDependence::Covariance { .. } => "supplied_covariance",
        PropagationDependence::AlignedJointLaw => "aligned_joint_law",
        _ => "other",
    }
}

/// The calibration key of a composed interval: the coordinate (operation,
/// route and input roles) is the functional, the route the estimator and the
/// construction the interval method. `evidence_rows` is the declared sample
/// size of the least-supported input's evidence.
#[must_use]
pub fn composed_basis(
    coordinate_id: &str,
    route: &str,
    dependence: &PropagationDependence,
    construction: &str,
    evidence_rows: u64,
) -> CalibrationBasis {
    let a = |text: &str| Arc::<str>::from(text);
    CalibrationBasis::new(
        [
            a(COMPOSED_QUERY),
            a("NotApplicable"),
            a("fixed"),
            a("composed_estimates"),
            a("Frequentist"),
            a(&format!("joint_law.{route}")),
            a(construction),
            a(""),
            a(dependence_label(dependence)),
            a(""),
            a(coordinate_id),
        ],
        COMPOSED_LEVEL,
        a("point"),
        evidence_rows,
        None,
        None,
        0.0,
    )
}

/// What the registry says about a composed key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// No record measured this construction at this level and size.
    NoRecord,
    /// A record matches but does not attest (boundary, outside its range or
    /// no longer current): its identity and the reason.
    NotAttesting {
        /// Record identity.
        record_id: String,
        /// Why it does not attest.
        reason: String,
    },
    /// A currently attesting record governs the key.
    Calibrated {
        /// Record identity.
        record_id: String,
    },
}

/// Resolve a key against the shipped records.
#[must_use]
pub fn resolve(basis: &CalibrationBasis) -> Resolution {
    let wire = CalibrationBasisWire {
        key: CalibrationKeyWire {
            query: basis.query.to_string(),
            graph_class: basis.graph_class.to_string(),
            structure: basis.structure.to_string(),
            modality: basis.modality.to_string(),
            inference: basis.inference.to_string(),
            estimator: basis.estimator.to_string(),
            interval_method: basis.interval_method.to_string(),
            se_kind: basis.se_kind.to_string(),
            dependence: basis.dependence.to_string(),
            posterior: basis.posterior.to_string(),
            functional: basis.functional.to_string(),
            level: basis.level,
            identification: basis.identification.to_string(),
        },
        scope: CalibrationScopeWire {
            row_count: basis.row_count,
            replicates_ok: basis.replicates_ok,
            posterior_draws: basis.posterior_draws,
            unidentified_mass: basis.unidentified_mass,
        },
    };
    let slot = calibration_slot(&wire);
    match (slot.status.as_str(), slot.record_id) {
        ("calibrated", Some(record_id)) => Resolution::Calibrated { record_id },
        (_, Some(record_id)) => Resolution::NotAttesting {
            record_id,
            reason: slot.reason.unwrap_or_else(|| "record_not_attesting".to_owned()),
        },
        _ => Resolution::NoRecord,
    }
}
