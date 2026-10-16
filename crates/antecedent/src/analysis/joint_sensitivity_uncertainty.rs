//! Sampling uncertainty of the joint mechanism sensitivity range on the
//! prepared z stage (record `sensitivity.joint_uncertainty`).
//!
//! The public interval is the conservative endpoint percentile bootstrap at
//! the only measured coordinate: level 0.95 and at least 199 replicates, the
//! floor of
//! `cov.classical_transport.admg.frequentist.percentile_bootstrap.l95.z_joint_sensitivity_zero_box`
//! and `...positive_box`. The version 3 range artifact still withholds a
//! sampling interval and its consumer still refuses a forged one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, reason_code};
use antecedent_io::IoError;
use antecedent_validate::joint_mechanism_sensitivity::JOINT_SENSITIVITY_MAX_BOOTSTRAP;
use antecedent_validate::{
    JointDeviationSpec, JointSensitivityInterval, joint_sensitivity_bootstrap_interval_internal,
};
use serde::{Deserialize, Serialize};

use super::PreparedZTransport;

/// Smallest replicate count the public interval admits.
///
/// Both measured coverage records set `replicates_min` to this value.
const PUBLIC_MIN_REPLICATES: u32 = 199;
/// The only nominal level those records measured.
const PUBLIC_LEVEL: f64 = 0.95;

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
    /// `cell_not_licensed`.
    pub reason_code: String,
    /// Always `None` from a public producer; a consumer refuses a value
    /// (`cell_not_licensed` / `joint_sensitivity.interval_withheld`).
    pub interval: Option<[f64; 2]>,
}

impl PreparedZTransport {
    /// Conservative endpoint percentile bootstrap of the joint range.
    ///
    /// The interval is `[q_{(1-level)/2}(L*), q_{(1+level)/2}(U*)]` at level
    /// 0.95. A request for fewer than 199 or more than 2000 replicates, or any
    /// other level, refuses with `joint_sensitivity.bounds_exceeded`.
    ///
    /// # Errors
    ///
    /// The joint-sensitivity contract, including `joint_sensitivity.bounds_exceeded`.
    pub fn joint_mechanism_sensitivity_interval(
        &self,
        spec: &JointDeviationSpec,
        replicates: u32,
        level: f64,
        ctx: &ExecutionContext,
    ) -> Result<JointSensitivityInterval, IoError> {
        if !(PUBLIC_MIN_REPLICATES..=JOINT_SENSITIVITY_MAX_BOOTSTRAP).contains(&replicates)
            || level != PUBLIC_LEVEL
        {
            return Err(IoError::Refused {
                code: reason_code!("route_not_supported"),
                message: format!(
                    "joint_sensitivity.bounds_exceeded: {replicates} replicates (199-2000) or level {level} (only 0.95)"
                ),
            });
        }
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
