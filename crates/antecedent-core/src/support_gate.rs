//! The one gate a `not_supported` route passes through.
//!
//! A route is supported (it runs) or `not_supported` (it refuses with the reason code
//! `not_supported`, a detail and the evidence it lacks). Scientific refusals (not identified,
//! typed n/a, invalid declaration, no such method) are not this gate's business and refuse in
//! every build.
//!
//! A **release build** returns [`NotSupported`] from [`require_supported`]. A **development
//! build** (cargo feature `development`) records a [`WouldRefuse`] entry and returns `Ok(())`,
//! so the route runs and the result states what a release build would have refused. The
//! development path is compiled only under the feature; a release build contains no code that
//! swallows the refusal.
//!
//! Two entry points exist:
//!
//! - [`require_supported`] takes the [`ExecutionContext`]; the entry goes to the log every
//!   clone of that context shares. Prefer it.
//! - [`require_supported_unscoped`] is for a call site with no context (the study builder runs
//!   before one exists). In a development build it records into a **thread-local** log that
//!   [`drain_unscoped`] empties. Limits: the log belongs to the calling thread, so the caller
//!   must drain it on the same thread that recorded (the builder does so at the end of the
//!   build and carries the entries on the study, which copies them onto its results); entries
//!   that nothing drains are dropped with the thread and never reach a result.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::execution::ExecutionContext;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::{Mutex, OnceLock};

/// `"development"` when this build enables the `development` feature, else `"release"`.
pub const BUILD_FLAVOR: &str =
    if cfg!(feature = "development") { "development" } else { "release" };

/// What a development build ran although a release build would have refused it.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct WouldRefuse {
    /// The registered reason code the release build refuses with (`not_supported`).
    pub code: &'static str,
    /// What is not supported.
    pub detail: String,
    /// The evidence that is missing.
    pub missing: Vec<String>,
}

impl WouldRefuse {
    /// An entry with the `not_supported` code.
    #[must_use]
    pub fn not_supported(detail: impl Into<String>, missing: &[&str]) -> Self {
        Self {
            code: crate::reason_code!("not_supported"),
            detail: detail.into(),
            missing: missing.iter().map(|m| (*m).to_owned()).collect(),
        }
    }
}

impl fmt::Display for WouldRefuse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)?;
        if !self.missing.is_empty() {
            write!(f, " (missing: {})", self.missing.join(", "))?;
        }
        Ok(())
    }
}

/// A route that is not supported: the refusal a release build returns.
///
/// Each crate's error enum converts from it (`From<NotSupported>`), added where that enum lives.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NotSupported {
    /// What is not supported.
    pub detail: &'static str,
    /// The evidence that is missing.
    pub missing: Vec<String>,
}

impl NotSupported {
    /// The registered reason code: `not_supported`.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        crate::reason_code!("not_supported")
    }

    /// The wire message `reason=not_supported: <detail> (missing: ...)`, interned so error
    /// types that hold a `&'static str` message can carry it. The set of distinct messages is
    /// bounded by the call sites and the evidence names they pass.
    #[must_use]
    pub fn message(&self) -> &'static str {
        static INTERNED: OnceLock<Mutex<BTreeSet<&'static str>>> = OnceLock::new();
        let text = self.to_string();
        let mut set = INTERNED
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(found) = set.get(text.as_str()) {
            return found;
        }
        let leaked: &'static str = Box::leak(text.into_boxed_str());
        set.insert(leaked);
        leaked
    }
}

impl fmt::Display for NotSupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}: {}", crate::reason_code::PREFIX, self.code(), self.detail)?;
        if !self.missing.is_empty() {
            write!(f, " (missing: {})", self.missing.join(", "))?;
        }
        Ok(())
    }
}

impl std::error::Error for NotSupported {}

#[cfg(not(feature = "development"))]
fn refusal(detail: &'static str, missing: &[&str]) -> NotSupported {
    NotSupported { detail, missing: missing.iter().map(|m| (*m).to_owned()).collect() }
}

/// Require the route to be supported.
///
/// Release: `Err(NotSupported)`. Development: records a [`WouldRefuse`] in `ctx`'s shared log
/// and returns `Ok(())` so the route runs.
///
/// # Errors
///
/// [`NotSupported`] in a release build.
pub fn require_supported(
    ctx: &ExecutionContext,
    detail: &'static str,
    missing: &[&str],
) -> Result<(), NotSupported> {
    #[cfg(feature = "development")]
    {
        ctx.record_would_refuse(WouldRefuse::not_supported(detail, missing));
        Ok(())
    }
    #[cfg(not(feature = "development"))]
    {
        let _ = ctx;
        Err(refusal(detail, missing))
    }
}

/// [`require_supported`] for a call site with no [`ExecutionContext`].
///
/// Release: `Err(NotSupported)`. Development: records into the calling thread's log (see the
/// module docs for the limits) and returns `Ok(())`; [`drain_unscoped`] empties it.
///
/// # Errors
///
/// [`NotSupported`] in a release build.
pub fn require_supported_unscoped(
    detail: &'static str,
    missing: &[&str],
) -> Result<(), NotSupported> {
    #[cfg(feature = "development")]
    {
        UNSCOPED.with(|log| {
            let entry = WouldRefuse::not_supported(detail, missing);
            let mut log = log.borrow_mut();
            if !log.contains(&entry) {
                log.push(entry);
            }
        });
        Ok(())
    }
    #[cfg(not(feature = "development"))]
    {
        Err(refusal(detail, missing))
    }
}

/// Fixture writers call this first: a development build seals artifacts a release build
/// refuses, so it must never write a checked-in fixture.
///
/// # Panics
///
/// In a development build.
pub fn forbid_fixture_write() {
    assert!(
        BUILD_FLAVOR == "release",
        "a development build must not write fixtures: build without `--features development`"
    );
}

/// Empty and return this thread's context-free log. Always empty in a release build.
#[must_use]
pub fn drain_unscoped() -> Vec<WouldRefuse> {
    #[cfg(feature = "development")]
    {
        UNSCOPED.with(|log| std::mem::take(&mut *log.borrow_mut()))
    }
    #[cfg(not(feature = "development"))]
    {
        Vec::new()
    }
}

#[cfg(feature = "development")]
thread_local! {
    static UNSCOPED: std::cell::RefCell<Vec<WouldRefuse>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flavor_follows_the_feature() {
        assert_eq!(BUILD_FLAVOR == "development", cfg!(feature = "development"));
    }

    #[cfg(not(feature = "development"))]
    #[test]
    fn release_refuses_with_the_detail_and_missing_evidence() {
        let ctx = ExecutionContext::for_tests(1);
        let err = require_supported(&ctx, "route x", &["a", "b"]).unwrap_err();
        assert_eq!(err.code(), "not_supported");
        assert_eq!(err.to_string(), "reason=not_supported: route x (missing: a, b)");
        assert!(ctx.would_refuse().is_empty());
        assert!(require_supported_unscoped("route x", &[]).is_err());
        assert!(drain_unscoped().is_empty());
    }

    #[cfg(feature = "development")]
    #[test]
    fn development_runs_and_records_in_the_shared_log() {
        let ctx = ExecutionContext::for_tests(1);
        let clone = ctx.clone();
        assert!(require_supported(&clone, "route x", &["a"]).is_ok());
        assert!(require_supported(&clone, "route x", &["a"]).is_ok());
        let log = ctx.would_refuse();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].code, "not_supported");
        assert_eq!(log[0].missing, vec!["a".to_owned()]);
        assert!(require_supported_unscoped("route y", &[]).is_ok());
        assert_eq!(drain_unscoped().len(), 1);
        assert!(drain_unscoped().is_empty());
    }
}
