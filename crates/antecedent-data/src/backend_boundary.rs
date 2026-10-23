//! Arrow feature boundary and the backends that are outside the contract.
//!
//! Arrow crate types stay in `arrow_adapter.rs` (ADR 0004). The C Data
//! Interface lives in `arrow_ffi.rs` and is compiled only with the optional
//! `arrow` feature, which is also this crate's default. This module names no
//! Arrow type, so it builds with `--no-default-features`.
//!
//! Polars is not a dependency and is not a backend of the internal contract.
//! That crate was refused and the empty `polars` feature was retired. See
//! `parity/polars_decision/decision.toml`. File-backed Parquet and Arrow IPC
//! tables are read by `file_source` into the same library-owned columns.
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

/// Preparation path this repository executes.
///
/// The record is `parity/polars_decision/decision.toml`. Direct Arrow is that
/// path. A lazy backend is not part of it.
pub const ACCEPTED_PREPARATION_PATH: &str = "direct_arrow";

/// Whether this crate has adopted a lazy execution backend.
///
/// `lazy_identity` records a view of owned storage. It does not lower a plan.
/// Polars is not linked. Nothing else in this crate executes a lazy plan, so
/// the preparation path does not offer one.
pub const LAZY_BACKEND_ADOPTED: bool = false;

/// Backend a preparation plan may name.
///
/// [`Self::DirectArrow`] is the accepted path. [`Self::Lazy`] is admitted only
/// when [`LAZY_BACKEND_ADOPTED`] is true.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PreparationBackend {
    /// Copy or borrow Arrow into library-owned columns.
    DirectArrow,
    /// Lower the plan onto a lazy backend this crate has adopted.
    Lazy,
}

impl PreparationBackend {
    /// The backend the accepted preparation path uses.
    #[must_use]
    pub const fn accepted() -> Self {
        Self::DirectArrow
    }

    /// Stable name. Direct Arrow spells `direct_arrow`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectArrow => ACCEPTED_PREPARATION_PATH,
            Self::Lazy => "lazy",
        }
    }

    /// Admit `self` as the preparation backend.
    ///
    /// # Errors
    ///
    /// [`PreparationBackend::Lazy`] while [`LAZY_BACKEND_ADOPTED`] is false.
    /// Direct Arrow is admitted.
    pub const fn admit(self) -> Result<Self, LazyBackendRefusal> {
        match self {
            Self::DirectArrow => Ok(self),
            Self::Lazy => {
                if LAZY_BACKEND_ADOPTED {
                    Ok(self)
                } else {
                    Err(LazyBackendRefusal)
                }
            }
        }
    }
}

/// A lazy backend was requested and this crate has not adopted one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LazyBackendRefusal;

impl core::fmt::Display for LazyBackendRefusal {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(
            "preparation refused: lazy backend is not adopted; the accepted path is direct_arrow",
        )
    }
}

impl std::error::Error for LazyBackendRefusal {}

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
        let feature_only = "[features]\ndefault = [\"arrow\"]\npolars = []\n";
        assert!(!declares_polars_crate(feature_only));
    }

    #[test]
    fn decision_record_keeps_direct_arrow_and_an_unlinked_polars_crate() {
        let decision = include_str!("../../../parity/polars_decision/decision.toml");
        for needle in [
            "item = \"polars_refused\"",
            "preparation_path = \"direct_arrow\"",
            "polars_feature = \"retired\"",
            "dependency_added = false",
            "version = \"0.55.2\"",
            "crate_cargo_toml_rust_version = \"absent\"",
            "workspace_cargo_toml_rust_version = \"absent\"",
            "not_linked_because = \"unavailable_rust_version\"",
            "latency = \"not_measured\"",
            "rss = \"not_measured\"",
            "spill = \"not_measured\"",
            "build = \"not_measured\"",
            "msrv = \"not_measured\"",
            "wheel = \"not_measured\"",
        ] {
            assert!(decision.contains(needle), "{needle} missing from the Polars decision");
        }
        assert!(!declares_polars_crate(DATA_CARGO));
        assert!(!declares_polars_crate(WORKSPACE_CARGO));
        assert!(!DATA_CARGO.contains("\npolars = []"), "the empty polars feature is retired");
    }

    #[test]
    fn accepted_path_is_direct_arrow_and_lazy_is_not_adopted() {
        assert_eq!(super::ACCEPTED_PREPARATION_PATH, "direct_arrow");
        const { assert!(!super::LAZY_BACKEND_ADOPTED) };
        assert_eq!(
            super::PreparationBackend::accepted().admit().expect("direct arrow").as_str(),
            "direct_arrow"
        );
        let refused = super::PreparationBackend::Lazy.admit().expect_err("no lazy backend");
        let message = refused.to_string();
        assert!(message.contains("lazy backend is not adopted"), "{message}");
        assert!(message.contains("direct_arrow"), "{message}");
    }

    fn declares_polars_crate(manifest: &str) -> bool {
        let mut in_features = false;
        manifest.lines().any(|line| {
            let code = line.split('#').next().unwrap_or(line).trim();
            if let Some(header) = code.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')) {
                in_features = header == "features";
                return line_names_polars_crate(line);
            }
            if in_features {
                return false;
            }
            line_names_polars_crate(line)
        })
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
