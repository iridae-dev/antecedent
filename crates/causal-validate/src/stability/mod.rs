//! Discovery stability and validation (DESIGN.md §18.3).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod env_holdout;
mod false_positive;
mod null_calibration;
mod orientation;
mod pcmci_grid;
mod regime;

pub use env_holdout::{EnvironmentHoldout, EnvironmentHoldoutReport};
pub use false_positive::{FalsePositiveCheck, FalsePositiveCheckReport, NullTransform};
pub use null_calibration::{NullCalibrationReport, SyntheticNullCalibration};
pub use orientation::{OrientationStability, OrientationStabilityReport, UndirectedLinkStability};
pub use pcmci_grid::{
    AlphaThresholdSensitivity, BlockBootstrapStability, CiTestSensitivity, DiscoveryStabilityReport,
    LagWindowSensitivity, LinkStability,
};
pub use regime::{RegimeStability, RegimeStabilityReport};
