//! Arrow feature boundary and the backends that are outside the contract.
//!
//! Arrow crate types stay in `arrow_adapter.rs` (ADR 0004). The C Data
//! Interface lives in `arrow_ffi.rs` and is compiled only with the optional
//! `arrow` feature, which is also this crate's default. This module names no
//! Arrow type, so it builds with `--no-default-features`.
//!
//! Polars is not a dependency and is not a backend of the internal contract.
//! Nothing here selects a default backend.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![deny(missing_docs)]
#![deny(unsafe_code)]

/// Polars is not a backend of the internal contract.
///
/// ADR 0026 keeps Polars outside this contract. The value stays false until a
/// later ADR adopts it. This constant does not add the crate and does not make
/// a backend the default.
pub const POLARS_IS_INTERNAL_CONTRACT_BACKEND: bool = false;

#[cfg(test)]
mod tests {
    const WORKSPACE_CARGO: &str = include_str!("../../../Cargo.toml");
    const DATA_CARGO: &str = include_str!("../Cargo.toml");

    #[test]
    fn workspace_pins_arrow_array_54_and_pyo3_0_29() {
        assert!(
            WORKSPACE_CARGO.contains("arrow-array = { version = \"54\""),
            "workspace Cargo.toml must pin arrow-array to version 54"
        );
        assert!(
            WORKSPACE_CARGO.contains("arrow-schema = { version = \"54\""),
            "workspace Cargo.toml must pin arrow-schema to version 54"
        );
        assert!(
            WORKSPACE_CARGO.contains("arrow-buffer = { version = \"54\""),
            "workspace Cargo.toml must pin arrow-buffer to version 54"
        );
        assert!(
            WORKSPACE_CARGO.contains("pyo3 = { version = \"0.29\""),
            "workspace Cargo.toml must pin pyo3 to version 0.29"
        );
    }

    #[test]
    fn workspace_has_no_polars_crate_dependency_key() {
        assert!(
            !declares_polars_crate(WORKSPACE_CARGO),
            "workspace Cargo.toml must not declare a polars crate"
        );
    }

    #[test]
    fn data_crate_arrow_feature_is_optional_and_default() {
        assert!(
            DATA_CARGO.contains("default = [\"arrow\"]"),
            "the arrow feature is the antecedent-data default"
        );
        assert!(
            DATA_CARGO.contains("arrow-array = { workspace = true, optional = true }"),
            "arrow-array stays an optional dependency"
        );
        assert!(
            !declares_polars_crate(DATA_CARGO),
            "antecedent-data must not declare a polars crate"
        );
    }

    #[test]
    fn polars_key_detector_matches_a_dependency_key() {
        assert!(line_names_polars_crate("polars = \"0.55\""));
        assert!(line_names_polars_crate("polars = { version = \"0.55\" }"));
        assert!(line_names_polars_crate("[dependencies.polars]"));
        assert!(!line_names_polars_crate("# polars = \"0.55\""));
        assert!(!line_names_polars_crate("arrow-array = { version = \"54\" }"));
        assert!(!line_names_polars_crate("polars-arrow = \"1\""));
    }

    fn declares_polars_crate(manifest: &str) -> bool {
        manifest.lines().any(line_names_polars_crate)
    }

    fn line_names_polars_crate(line: &str) -> bool {
        let code = line.split('#').next().unwrap_or(line).trim();
        if code.is_empty() {
            return false;
        }
        if let Some(header) = code.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')) {
            return header.split('.').any(segment_is_polars);
        }
        let Some((key, _)) = code.split_once('=') else {
            return false;
        };
        let key = key.trim();
        segment_is_polars(key) || key.strip_prefix("polars.").is_some_and(|rest| !rest.is_empty())
    }

    fn segment_is_polars(segment: &str) -> bool {
        segment.trim().trim_matches('"') == "polars"
    }
}
