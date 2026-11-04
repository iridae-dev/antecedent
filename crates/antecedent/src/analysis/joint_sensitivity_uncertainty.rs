//! Sampling uncertainty of the joint mechanism sensitivity range on the
//! prepared z stage (record `sensitivity.joint_uncertainty`).
//!
//! The interval is the conservative endpoint percentile bootstrap, annotated with the
//! calibration of the measured coordinate (level 0.95, at least 199 replicates), the
//! floor of
//! `cov.classical_transport.admg.frequentist.percentile_bootstrap.l95.z_joint_sensitivity_zero_box`
//! and `...positive_box`. The version 3 range artifact does not attach the interval (its
//! replay would need the bootstrap seed and replicate count); the interval is reported by
//! [`PreparedZTransport::joint_mechanism_sensitivity_interval`] with its calibration slot, and
//! the surface adapter of `antecedent-io-core` reports it per grid point. The artifact's consumer
//! still refuses a forged interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, reason_code};
use antecedent_io::IoError;
use antecedent_validate::{
    JointDeviationSpec, JointSensitivityInterval, joint_sensitivity_bootstrap_interval_internal,
};
use serde::{Deserialize, Serialize};

use super::PreparedZTransport;

/// The sampling descriptor of the version 3 joint sensitivity artifact: the
/// declared method and why its interval is withheld. It lives with this closed
/// route because the interval slot belongs to it; the range artifact
/// (`JointSensitivityBodyWire.sampling`) only carries it withheld.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointSamplingWire {
    /// The one declared composition.
    pub method: String,
    /// Its coverage target.
    pub coverage_target: String,
    /// `withheld`.
    pub status: String,
    /// `not_supported`.
    pub reason_code: String,
    /// Always `None` from a public producer; a consumer refuses a value
    /// (`route_not_supported` / `joint_sensitivity.interval_withheld`).
    pub interval: Option<[f64; 2]>,
}

impl PreparedZTransport {
    /// Conservative endpoint percentile bootstrap of the joint range, with its calibration
    /// slot.
    ///
    /// The interval is `[q_{(1-level)/2}(L*), q_{(1+level)/2}(U*)]`. Replicates inside the
    /// estimator's bounds (2 to 2000) and a level strictly inside `(0, 1)` are accepted;
    /// the measured scope (level 0.95, at least 199 replicates) only decides the slot's
    /// status, never whether the interval is reported.
    ///
    /// # Errors
    ///
    /// The joint-sensitivity contract, including `joint_sensitivity.bounds_exceeded` for a
    /// replicate count or level the estimator cannot evaluate.
    pub fn joint_mechanism_sensitivity_interval(
        &self,
        spec: &JointDeviationSpec,
        replicates: u32,
        level: f64,
        ctx: &ExecutionContext,
    ) -> Result<JointSensitivityInterval, IoError> {
        match joint_sensitivity_bootstrap_interval_internal(
            self.diagram(),
            self.functional(),
            self.data(),
            spec,
            replicates,
            level,
            ctx,
        ) {
            Ok(Ok(interval)) => Ok(interval),
            Ok(Err(reason)) => Err(IoError::Refused {
                code: reason_code!("transport_numerical_failure"),
                message: reason.into(),
            }),
            Err(error) => {
                Err(IoError::Refused { code: error.reason_code(), message: error.to_string() })
            }
        }
    }
}
