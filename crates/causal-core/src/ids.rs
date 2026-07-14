//! Compact, copyable identifiers used throughout causal-library.
//!
//! User-facing names live in schemas and dictionaries, not in hot graph or
//! numerical structures (DESIGN.md §4).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use core::num::NonZeroU32;

/// Dense variable index assigned by [`crate::schema::CausalSchema`] construction.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VariableId(u32);

impl VariableId {
    /// Create an identifier from a raw dense index.
    ///
    /// Prefer obtaining IDs from schema construction; this constructor exists
    /// for deserialization and tests.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Underlying dense index.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Convert to `usize` for indexing.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for VariableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V{}", self.0)
    }
}

/// Multi-environment partition identifier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EnvironmentId(u32);

impl EnvironmentId {
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
}

impl fmt::Display for EnvironmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}", self.0)
    }
}

/// Regime identifier for regime-aware analysis.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RegimeId(u32);

impl RegimeId {
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
}

impl fmt::Display for RegimeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "R{}", self.0)
    }
}

/// Temporal lag magnitude.
///
/// [`Lag::CONTEMPORANEOUS`] (`Lag(0)`) is contemporaneous. Historical nodes use
/// positive lag values internally. Negative-lag conventions are confined to
/// import/export adapters (DESIGN.md §4, ADR 0005).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Lag(u32);

impl Lag {
    /// Contemporaneous lag (`0`).
    pub const CONTEMPORANEOUS: Self = Self(0);

    /// Create a lag from a non-negative magnitude.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Create a positive historical lag.
    #[must_use]
    pub const fn historical(steps: NonZeroU32) -> Self {
        Self(steps.get())
    }

    /// Underlying magnitude.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Whether this lag is contemporaneous.
    #[must_use]
    pub const fn is_contemporaneous(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for Lag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.0)
    }
}

/// Attribution component (typically a mechanism / node) in change decomposition.
///
/// Distinct from [`VariableId`] at the type level so allocation orders cannot
/// silently mix raw variables with coalition players (DESIGN.md §17.2).
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

/// Stable handle for an immutable categorical domain stored in a schema.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CategoryDomainId(u32);

impl CategoryDomainId {
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
}

impl fmt::Display for CategoryDomainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C{}", self.0)
    }
}

/// Registered causal-query handle in incremental state / design objectives.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct QueryId(u32);

impl QueryId {
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

    /// Convert to `usize` for indexing.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for QueryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q{}", self.0)
    }
}

/// Registered model handle in design objectives / state model stores.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ModelId(u32);

impl ModelId {
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

    /// Convert to `usize` for indexing.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "M{}", self.0)
    }
}

/// Monotonic causal-state version (DESIGN.md §20).
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
