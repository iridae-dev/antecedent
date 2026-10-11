//! The closed sampling-uncertainty route of the joint mechanism sensitivity
//! range on the prepared z stage (record `sensitivity.joint_uncertainty`,
//! calibration unmeasured).
//!
//! The one declared composition, the conservative endpoint percentile
//! bootstrap, has a one-sided coverage target. Its coverage harness
//! (`crates/antecedent/tests/joint_sensitivity_calibration.rs`, a `one_sided`
//! record) is unmeasured, so this route always refuses with
//! `cell_not_licensed`.
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
    /// `cell_not_licensed`.
    pub reason_code: String,
    /// Always `None` from a public producer; a consumer refuses a value
    /// (`cell_not_licensed` / `joint_sensitivity.interval_withheld`).
    pub interval: Option<[f64; 2]>,
}

impl PreparedZTransport {
    /// Conservative endpoint percentile bootstrap of the joint range.
    ///
    /// The interval is `[q_{(1-level)/2}(L*), q_{(1+level)/2}(U*)]`. A request
    /// for fewer than 2 or more than 2000 replicates, or a level outside
    /// `(0, 1)`, refuses.
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
            Err(error) => Err(IoError::Refused {
                code: error.reason_code(),
                message: error.to_string(),
            }),
        }
    }
}
