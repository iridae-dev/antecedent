//! Shared imports for integration targets that need only the known-truth fixtures.
//!
//! Keep the generators and their seed helpers in their existing files so every
//! caller uses the same data-generating processes and calibration seed streams.
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../common/calibration.rs"]
pub mod calibration;
#[path = "../common/fixtures.rs"]
pub mod fixtures;
