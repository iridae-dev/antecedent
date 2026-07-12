//! Numeric tolerance classes (ADR 0010 / DESIGN.md §28.5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Declared numeric comparison class for a fixture or assertion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ToleranceClass {
    /// Bitwise / structural equality.
    Exact,
    /// Stable floating-point (default `atol=1e-10`, `rtol=1e-8`).
    StableFloat,
    /// Backend-sensitive linear algebra (`atol=1e-8`, `rtol=1e-6`).
    BackendSensitive,
    /// Compare residuals / fitted values, not raw coefficients.
    ResidualBased,
    /// Monte Carlo summary with documented SE floor.
    MonteCarlo,
    /// Posterior distribution comparison.
    PosteriorDistribution,
}

impl ToleranceClass {
    /// Absolute tolerance default for this class.
    #[must_use]
    pub const fn atol(self) -> f64 {
        match self {
            Self::Exact => 0.0,
            Self::StableFloat => 1e-10,
            Self::BackendSensitive => 1e-8,
            Self::ResidualBased | Self::MonteCarlo | Self::PosteriorDistribution => 1e-6,
        }
    }

    /// Relative tolerance default for this class.
    #[must_use]
    pub const fn rtol(self) -> f64 {
        match self {
            Self::Exact => 0.0,
            Self::StableFloat => 1e-8,
            Self::BackendSensitive => 1e-6,
            Self::ResidualBased | Self::MonteCarlo | Self::PosteriorDistribution => 1e-4,
        }
    }

    /// Whether `actual` matches `expected` under this class.
    #[must_use]
    #[allow(clippy::float_cmp)] // Exact class intentionally uses bitwise equality.
    pub fn close(self, actual: f64, expected: f64) -> bool {
        if self == Self::Exact {
            return actual == expected;
        }
        let diff = (actual - expected).abs();
        diff <= self.atol() || diff <= self.rtol() * expected.abs().max(actual.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_float_accepts_tiny_error() {
        assert!(ToleranceClass::StableFloat.close(2.0, 2.0 + 1e-12));
        assert!(!ToleranceClass::StableFloat.close(2.0, 2.01));
    }
}
