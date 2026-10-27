//! Execute shared calibration decision controls once, without measuring coverage.
#![allow(dead_code, reason = "the shared harness also serves calibration consumers")]

#[path = "common/calibration_decision.rs"]
mod decision;
use decision::*;
#[path = "common/calibration_decision_tests.rs"]
mod tests;
