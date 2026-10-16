//! Shared helpers for the Antecedent workspace's integration tests.
//!
//! This crate is `publish = false` and is only ever a `[dev-dependencies]` entry of
//! other workspace crates. It depends on `std` and `serde_json` alone and on no
//! Antecedent crate, so it can never introduce a dependency cycle. Helpers that name
//! Antecedent types, and the calibration harness, stay crate-local.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

/// The `conformance/` directory at the repository root joined with `relative`.
///
/// Every workspace crate lives at `crates/<name>`, so this crate's manifest
/// directory resolves `../../conformance` exactly as each test crate's did.
#[must_use]
pub fn conformance_dir(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance").join(relative)
}

/// Parse `expected.json` from a conformance fixture directory.
///
/// # Panics
///
/// Panics if the file is missing or is not valid JSON.
#[must_use]
pub fn load_expected_json(dir: &Path) -> serde_json::Value {
    let raw = std::fs::read_to_string(dir.join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

/// Read a fixture's bytes, panicking with `regenerate_hint` when it is missing.
///
/// # Panics
///
/// Panics, naming the path and `regenerate_hint`, if the file cannot be read.
#[must_use]
pub fn read_fixture_bytes(dir: &Path, name: &str, regenerate_hint: &str) -> Vec<u8> {
    let path = dir.join(name);
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!("missing cross-surface fixture {} ({error}); {regenerate_hint}", path.display())
    })
}

/// `|a - b| <= tol`.
#[must_use]
pub fn within(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// `|a - b| < 1e-12` (strict).
#[must_use]
pub fn near_lt_1e12(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

/// `|a - b| <= 1e-12` (inclusive).
#[must_use]
pub fn near_le_1e12(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12
}

/// `|a - b| < 1e-9` (strict).
#[must_use]
pub fn near_lt_1e9(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// Dot product of two slices, truncated to the shorter.
///
/// Kept here rather than calling `antecedent_kernels::dot`: this crate depends on no Antecedent
/// crate, and a test oracle should not share code with the function it checks.
#[must_use]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The logistic function `1 / (1 + exp(-x))`.
///
/// The textbook formula on purpose: it is the independent oracle the tests compare
/// `antecedent_kernels::sigmoid` and the model code against, and it is accurate on the moderate
/// arguments tests use.
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// A copy of `bytes` with every bit of the final byte flipped.
///
/// # Panics
///
/// Panics if `bytes` is empty.
#[must_use]
pub fn flip_last_byte(bytes: &[u8]) -> Vec<u8> {
    let mut copy = bytes.to_vec();
    *copy.last_mut().expect("non-empty artifact") ^= 0xFF;
    copy
}
