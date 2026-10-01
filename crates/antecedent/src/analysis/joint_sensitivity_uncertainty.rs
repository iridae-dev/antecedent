//! The closed sampling-uncertainty route of the joint mechanism sensitivity
//! range on the prepared z stage (record `2.2B.X3.joint_sensitivity_uncertainty`,
//! carried forward from 2.2).
//!
//! Sampling uncertainty is not offered in 2.2. The one declared composition,
//! the conservative endpoint percentile bootstrap, has a one-sided coverage
//! target that the shared coverage harness cannot measure yet, so this route
//! always refuses with `cell_not_licensed`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::convert::Infallible;

use antecedent_core::{ExecutionContext, reason_code};
use antecedent_io::IoError;
use antecedent_validate::JointDeviationSpec;
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
    /// The sampling-uncertainty interval of the joint range. Closed for the
    /// 2.2 release: always refuses with `cell_not_licensed`.
    ///
    /// # Errors
    ///
    /// Always `cell_not_licensed` / `joint_sensitivity.interval_withheld`.
    pub fn joint_mechanism_sensitivity_interval(
        &self,
        spec: &JointDeviationSpec,
        ctx: &ExecutionContext,
    ) -> Result<Infallible, IoError> {
        let _ = (spec, ctx);
        Err(IoError::Refused {
            code: reason_code!("cell_not_licensed"),
            message: "joint_sensitivity.interval_withheld: sampling uncertainty is not offered in 2.2; the conservative endpoint percentile bootstrap is not licensed and the assumption range is not a confidence interval".into(),
        })
    }
}
