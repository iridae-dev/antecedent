//! Shared numeric assertions for the artifact integration tests.
//!
//! Tolerances stay with the caller. These checks do not read or write an artifact.
//! Each integration test is its own crate and uses one of the two checks.

#![allow(dead_code)]

/// Absolute difference at most `tolerance`.
pub fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() <= tolerance, "{what}: {actual} vs {expected}");
}

/// Absolute difference strictly below `tolerance`.
pub fn close_below(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() < tolerance, "{actual} != {expected}");
}
