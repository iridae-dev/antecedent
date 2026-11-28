//! The calibration state of a result's evidence unit ([`antecedent_io_core::unit_calibration`]).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io_core::unit_calibration::NOT_MEASURED;

use super::unit::UnitRef;

/// The calibration state of `unit`.
#[must_use]
pub fn unit_calibration(unit: &UnitRef) -> &'static str {
    antecedent_io_core::unit_calibration::unit(unit.kind(), &unit.record_id())
}

/// The calibration state of the capability unit `id`.
#[must_use]
pub(crate) fn capability_calibration(id: &str) -> &'static str {
    antecedent_io_core::unit_calibration::capability(id)
}
