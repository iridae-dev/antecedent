//! Fixed process configuration for a Polars thread pool.
//!
//! A linked Polars pool is process-global. The choices for that kind of pool
//! are admission of one process width, isolation in another process, or a
//! refusal. This repository links no Polars crate, so the configuration is the
//! refusal: a study thread count is not applied, and direct Arrow remains the
//! preparation path. Nothing here resizes a pool, reads a pool width, or names
//! a Polars type. The adapter seam stays a seam.
//!
//! The record is `parity/polars_decision/pool.toml`. The Arrow decision is
//! `parity/polars_decision/decision.toml`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// Private module. Callers are the tests; a study does not obtain a pool from here.
#![allow(dead_code, reason = "process-pool refusal with no Polars crate linked")]

use std::fmt;

/// Preparation path recorded with the pool refusal.
pub const PREPARATION_PATH: &str = "direct_arrow";

/// The `polars` feature is empty. Enabling it links no Polars crate.
pub const POLARS_FEATURE: &str = "polars = []";

/// No Polars pool is linked. A study thread count cannot resize one.
pub const POOL_LINKED: bool = false;

/// What this process does with a Polars thread pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoolDisposition {
    /// No pool is linked. A study thread count is refused.
    Refusal,
}

/// Fixed configuration for the process.
///
/// The value carries no thread count. Constructing it does not start a pool,
/// and dropping it releases nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessConfiguration {
    disposition: PoolDisposition,
}

impl ProcessConfiguration {
    /// The process configuration: a refusal, with no linked pool.
    #[must_use = "the fixed configuration is what a study thread count is checked against"]
    pub const fn fixed() -> Self {
        Self { disposition: PoolDisposition::Refusal }
    }

    /// Admission, isolation, or refusal. This repository records a refusal.
    #[must_use]
    pub const fn disposition(self) -> PoolDisposition {
        self.disposition
    }

    /// Whether a Polars pool is linked into this process.
    #[must_use]
    pub const fn pool_linked(self) -> bool {
        match self.disposition {
            PoolDisposition::Refusal => POOL_LINKED,
        }
    }

    /// Preparation path that stays in force when a study asks for threads.
    #[must_use]
    pub const fn preparation_path(self) -> &'static str {
        match self.disposition {
            PoolDisposition::Refusal => PREPARATION_PATH,
        }
    }

    /// Refuse `threads` as a resize of the process pool.
    ///
    /// Every count is refused, including zero and one. The count is not stored.
    /// A later call with a different count does not change this configuration,
    /// because there is no pool width to update.
    #[must_use = "a study thread count is refused and does not resize a pool"]
    pub const fn refuse_study_threads(self, threads: u32) -> StudyResizeRefusal {
        match self.disposition {
            PoolDisposition::Refusal => StudyResizeRefusal::PerStudyResize { requested: threads },
        }
    }
}

/// A study thread count was not applied to a process pool.
///
/// The process configuration is unchanged. No pool was resized, because no
/// Polars pool is linked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "dropping the refusal would hide that the study thread count was not applied"]
#[non_exhaustive]
pub enum StudyResizeRefusal {
    /// `requested` threads were not applied.
    PerStudyResize {
        /// Thread count the study asked for. It is not a pool width.
        requested: u32,
    },
}

impl fmt::Display for StudyResizeRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PerStudyResize { requested } => write!(
                formatter,
                "a study asked for {requested} threads; that count is not applied to a process \
                 pool; no Polars pool is linked"
            ),
        }
    }
}

impl std::error::Error for StudyResizeRefusal {}

#[cfg(test)]
mod tests {
    use super::*;

    const DECISION: &str = include_str!("../../../parity/polars_decision/decision.toml");
    const POOL: &str = include_str!("../../../parity/polars_decision/pool.toml");
    const DATA_CARGO: &str = include_str!("../Cargo.toml");

    const UNMEASURED: &[&str] = &["latency", "rss", "spill", "build", "msrv", "wheel"];

    #[test]
    fn study_thread_count_does_not_resize_an_unlinked_pool() {
        let fixed = ProcessConfiguration::fixed();
        assert_eq!(fixed.disposition(), PoolDisposition::Refusal);
        assert!(!fixed.pool_linked());
        assert_eq!(fixed.preparation_path(), PREPARATION_PATH);
        assert_eq!(POLARS_FEATURE, "polars = []");

        let eight = fixed.refuse_study_threads(8);
        let one = fixed.refuse_study_threads(1);
        let zero = fixed.refuse_study_threads(0);
        assert_eq!(eight, StudyResizeRefusal::PerStudyResize { requested: 8 });
        assert_eq!(one, StudyResizeRefusal::PerStudyResize { requested: 1 });
        assert_eq!(zero, StudyResizeRefusal::PerStudyResize { requested: 0 });
        assert_ne!(eight, one);
        assert_eq!(fixed, ProcessConfiguration::fixed());
        assert!(!fixed.pool_linked());
        assert_eq!(fixed.preparation_path(), "direct_arrow");

        let message = eight.to_string();
        assert!(message.contains('8'), "{message}");
        assert!(message.contains("not applied"), "{message}");
        assert!(message.contains("no Polars pool is linked"), "{message}");
    }

    #[test]
    fn process_configuration_has_no_default_pool() {
        // `ProcessConfiguration: Default` would be a place to hide a process
        // width. A caller that omits the configuration does not get one.
        assert_no_default::<ProcessConfiguration>();
    }

    #[test]
    fn records_keep_direct_arrow_and_an_unlinked_pool() {
        for needle in [
            "preparation_path = \"direct_arrow\"",
            "polars_feature = \"polars = []\"",
            "dependency_added = false",
            "not_linked_because = \"unavailable_rust_version\"",
            "polars_internal_contract_backend = false",
        ] {
            assert!(DECISION.contains(needle), "{needle} missing from the Arrow decision");
        }
        for needle in [
            "preparation_path = \"direct_arrow\"",
            "scope = \"process\"",
            "configuration = \"fixed\"",
            "disposition = \"refusal\"",
            "pool_linked = false",
            "per_study_resize = \"refused\"",
            "polars_feature = \"polars = []\"",
            "dependency_added = false",
            "decision_record = \"parity/polars_decision/decision.toml\"",
        ] {
            assert!(POOL.contains(needle), "{needle} missing from the pool configuration");
        }
        for name in UNMEASURED {
            let needle = format!("{name} = \"not_measured\"");
            assert!(DECISION.contains(&needle), "{needle} missing from the Arrow decision");
            assert!(POOL.contains(&needle), "{needle} missing from the pool configuration");
        }
        assert!(!POOL.contains("version ="), "the pool record must not pin a crate");
        assert!(!POOL.contains("[dependencies]"), "the pool record must not link a crate");
        assert!(!POOL.contains("polars = \""), "the pool record must not pin a crate");
        assert!(!POOL.contains("polars = {"), "the pool record must not pin a crate");
        assert!(DATA_CARGO.contains("\npolars = []\n"), "{DATA_CARGO}");
        assert!(!DATA_CARGO.contains("polars = {"), "{DATA_CARGO}");
        assert!(!declares_polars_crate(DATA_CARGO), "{DATA_CARGO}");
    }

    fn declares_polars_crate(manifest: &str) -> bool {
        let mut in_features = false;
        manifest.lines().any(|line| {
            let code = line.split('#').next().unwrap_or(line).trim();
            if let Some(header) = code.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')) {
                in_features = header == "features";
                return header.split('.').any(|segment| segment.trim() == "polars");
            }
            if in_features {
                return false;
            }
            let Some((key, _)) = code.split_once('=') else {
                return false;
            };
            key.trim().trim_matches('"') == "polars"
        })
    }

    fn assert_no_default<T: ?Sized>() {
        struct Marker;
        trait Probe<A> {
            fn probe() {}
        }
        impl<T: ?Sized> Probe<()> for T {}
        #[allow(dead_code, reason = "the probe only needs the impl to exist")]
        impl<T: Default> Probe<Marker> for T {}
        <T as Probe<_>>::probe();
    }
}
