//! Compact, copyable identifiers used throughout Antecedent.
//!
//! User-facing names live in schemas and dictionaries, not in hot graph or
//! numerical structures.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use core::num::NonZeroU32;

/// `u32` newtype with `from_raw` / `raw`, `Display` as `{prefix}{raw}`, and
/// optionally `as_usize`. Type names and the stored integer are the macro
/// arguments; both stay source-compatible with the hand-written types.
macro_rules! dense_id {
    (
        $(#[$meta:meta])*
        $name:ident,
        $prefix:literal
    ) => {
        dense_id!($(#[$meta])* $name, $prefix, no_usize);
    };
    (
        $(#[$meta:meta])*
        $name:ident,
        $prefix:literal,
        usize
    ) => {
        dense_id!($(#[$meta])* $name, $prefix, with_usize);
    };
    (
        $(#[$meta:meta])*
        $name:ident,
        $prefix:literal,
        $usize:ident
    ) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name(u32);

        impl $name {
            /// Create from a raw dense index.
            #[must_use]
            pub const fn from_raw(raw: u32) -> Self {
                Self(raw)
            }

            /// Underlying dense index.
            #[must_use]
            pub const fn raw(self) -> u32 {
                self.0
            }

            dense_id!(@usize $usize);
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($prefix, "{}"), self.0)
            }
        }
    };
    (@usize with_usize) => {
        /// Convert to `usize` for indexing.
        #[must_use]
        pub const fn as_usize(self) -> usize {
            self.0 as usize
        }
    };
    (@usize no_usize) => {};
}

dense_id!(
    /// Dense variable index assigned by [`crate::schema::CausalSchema`] construction.
    ///
    /// `from_raw` exists for deserialization and tests. Prefer ids from schema construction.
    VariableId,
    "V",
    usize
);

dense_id!(
    /// Multi-environment partition identifier.
    EnvironmentId,
    "E",
    usize
);

dense_id!(
    /// Opaque handle for a rule-driven temporal intervention schedule.
    ///
    /// Paired with explicit `active_at` offsets on [`crate::intervention::TemporalPolicy::Dynamic`].
    DynamicRuleId,
    "DR",
    usize
);

dense_id!(
    /// Handle to a custom target distribution.
    ///
    /// Resolved against a caller-supplied registry; sampling is deferred.
    DistributionRef,
    "DRef",
    usize
);

dense_id!(
    /// Regime identifier for regime-aware analysis.
    RegimeId,
    "R",
    usize
);

dense_id!(
    /// Temporal lag magnitude.
    ///
    /// [`Lag::CONTEMPORANEOUS`] (`Lag(0)`) is contemporaneous. Historical nodes use
    /// positive lag values internally. Negative-lag conventions are confined to
    /// import/export adapters.
    Lag,
    "L",
    usize
);

impl Lag {
    /// Contemporaneous lag (`0`).
    pub const CONTEMPORANEOUS: Self = Self(0);

    /// Create a positive historical lag.
    #[must_use]
    pub const fn historical(steps: NonZeroU32) -> Self {
        Self(steps.get())
    }

    /// Whether this lag is contemporaneous.
    #[must_use]
    pub const fn is_contemporaneous(self) -> bool {
        self.0 == 0
    }
}

dense_id!(
    /// Stable handle for an immutable categorical domain stored in a schema.
    CategoryDomainId,
    "C",
    usize
);

dense_id!(
    /// Registered causal-query handle in incremental state / design objectives.
    QueryId,
    "Q",
    usize
);

dense_id!(
    /// Registered model handle in design objectives / state model stores.
    ModelId,
    "M",
    usize
);

/// Attribution component (typically a mechanism / node) in change decomposition.
///
/// Distinct from [`VariableId`] at the type level so allocation orders cannot
/// silently mix raw variables with coalition players.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ComponentId(VariableId);

impl ComponentId {
    /// Wrap a variable as an attribution component (mechanism of that node).
    #[must_use]
    pub const fn from_variable(variable: VariableId) -> Self {
        Self(variable)
    }

    /// Underlying variable id.
    #[must_use]
    pub const fn variable(self) -> VariableId {
        self.0
    }

    /// Create from a raw dense variable index.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(VariableId::from_raw(raw))
    }

    /// Underlying dense index.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0.raw()
    }

    /// Convert to `usize` for indexing.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0.as_usize()
    }
}

impl fmt::Display for ComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C{}", self.0.raw())
    }
}

impl From<VariableId> for ComponentId {
    fn from(variable: VariableId) -> Self {
        Self::from_variable(variable)
    }
}

/// Monotonic antecedent-state version.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct StateVersion(u64);

impl StateVersion {
    /// Initial version.
    pub const ZERO: Self = Self(0);

    /// Create from a raw counter.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Underlying counter.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Next version after an event application.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

impl fmt::Display for StateVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn ids_are_copyable_and_compact() {
        assert_eq!(size_of::<VariableId>(), 4);
        assert_eq!(size_of::<EnvironmentId>(), 4);
        assert_eq!(size_of::<DynamicRuleId>(), 4);
        assert_eq!(size_of::<DistributionRef>(), 4);
        assert_eq!(size_of::<RegimeId>(), 4);
        assert_eq!(size_of::<Lag>(), 4);
        assert_eq!(size_of::<CategoryDomainId>(), 4);
        assert_eq!(size_of::<ComponentId>(), 4);
        assert_eq!(size_of::<QueryId>(), 4);
        assert_eq!(size_of::<ModelId>(), 4);
        assert_eq!(size_of::<StateVersion>(), 8);
    }

    #[test]
    fn lag_zero_is_contemporaneous() {
        assert!(Lag::CONTEMPORANEOUS.is_contemporaneous());
        assert!(!Lag::historical(NonZeroU32::new(1).expect("nonzero")).is_contemporaneous());
    }
}
