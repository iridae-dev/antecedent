//! The closeness policy of an artifact replay.
//!
//! A consumer recomputes a stored number and compares it with the stored one. Formats have
//! distinct replay contracts: how far apart two numbers may be, what that distance is
//! measured against, and what a non-finite value means. This module names those three
//! choices once so every format states its policy as a constant instead of re-deriving the
//! comparison.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// What a [`ReplayTolerance`] measures the difference against.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayScale {
    /// The tolerance is an absolute bound.
    Absolute,
    /// `max(|replayed|, 1)`: relative to the recomputed value, absolute below one.
    ReplayedOrOne,
    /// `max(|stored|, |replayed|, 1)`: relative to the larger of the two, absolute below one.
    LargerOrOne,
    /// `1 + |replayed|`.
    OnePlusReplayed,
    /// `1 + max(|stored|, |replayed|)`.
    OnePlusLarger,
    /// `1 + |stored| + |replayed|`.
    OnePlusSum,
}

/// How a [`ReplayTolerance`] treats a non-finite value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayNonFinite {
    /// Any non-finite value is a mismatch.
    Refuse,
    /// Two NaNs match (a value the replay legitimately leaves undefined); any other
    /// non-finite value is a mismatch.
    NanMatchesNan,
    /// Two identical bit patterns match, whatever they are; any other non-finite value is
    /// a mismatch.
    SameBits,
}

/// A replay closeness policy: a tolerance, the scale it applies to, and the non-finite rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayTolerance {
    tolerance: f64,
    scale: ReplayScale,
    non_finite: ReplayNonFinite,
}

impl ReplayTolerance {
    /// A policy accepting `|stored - replayed| <= tolerance * scale`.
    #[must_use]
    pub const fn new(tolerance: f64, scale: ReplayScale, non_finite: ReplayNonFinite) -> Self {
        Self { tolerance, scale, non_finite }
    }

    /// The tolerance.
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Whether `stored` reproduces `replayed` under this policy.
    ///
    /// An infinite or NaN operand never passes the arithmetic comparison: an infinite scale
    /// would otherwise accept any finite stored value against an infinite replay.
    #[must_use]
    pub fn close(&self, stored: f64, replayed: f64) -> bool {
        match self.non_finite {
            ReplayNonFinite::Refuse => {}
            ReplayNonFinite::NanMatchesNan if stored.is_nan() && replayed.is_nan() => {
                return true;
            }
            ReplayNonFinite::NanMatchesNan => {}
            ReplayNonFinite::SameBits if stored.to_bits() == replayed.to_bits() => return true,
            ReplayNonFinite::SameBits => {}
        }
        if !(stored.is_finite() && replayed.is_finite()) {
            return false;
        }
        let scale = match self.scale {
            ReplayScale::Absolute => 1.0,
            ReplayScale::ReplayedOrOne => replayed.abs().max(1.0),
            ReplayScale::LargerOrOne => stored.abs().max(replayed.abs()).max(1.0),
            ReplayScale::OnePlusReplayed => 1.0 + replayed.abs(),
            ReplayScale::OnePlusLarger => 1.0 + stored.abs().max(replayed.abs()),
            ReplayScale::OnePlusSum => 1.0 + stored.abs() + replayed.abs(),
        };
        (stored - replayed).abs() <= self.tolerance * scale
    }

    /// Whether every stored value reproduces the replayed one at its position, and the
    /// lengths agree.
    #[must_use]
    pub fn all_close(&self, stored: &[f64], replayed: &[f64]) -> bool {
        stored.len() == replayed.len()
            && stored.iter().zip(replayed).all(|(&s, &r)| self.close(s, r))
    }
}

#[cfg(test)]
mod tests {
    use super::{ReplayNonFinite, ReplayScale, ReplayTolerance};

    #[test]
    fn scales_measure_the_difference_against_their_own_reference() {
        let relative =
            ReplayTolerance::new(1e-3, ReplayScale::ReplayedOrOne, ReplayNonFinite::Refuse);
        assert!(relative.close(1000.5, 1000.0));
        assert!(!relative.close(1002.0, 1000.0));
        assert!(relative.close(0.0005, 0.0), "below one the bound is absolute");
        // The larger operand is the scale for `LargerOrOne`.
        let larger = ReplayTolerance::new(1e-3, ReplayScale::LargerOrOne, ReplayNonFinite::Refuse);
        assert!(larger.close(1001.0, 1000.0));
        assert!(!larger.close(1002.5, 1000.0));
        let absolute = ReplayTolerance::new(1e-9, ReplayScale::Absolute, ReplayNonFinite::Refuse);
        assert!(absolute.close(1e6, 1e6 + 5e-10));
        assert!(!absolute.close(1e6, 1e6 + 1e-3));
    }

    #[test]
    fn non_finite_values_follow_the_declared_rule() {
        let refuse = ReplayTolerance::new(1e-9, ReplayScale::LargerOrOne, ReplayNonFinite::Refuse);
        assert!(!refuse.close(f64::NAN, f64::NAN));
        assert!(!refuse.close(f64::INFINITY, f64::INFINITY));
        assert!(!refuse.close(1.0, f64::INFINITY), "an infinite scale accepts nothing");
        let nan = ReplayTolerance::new(1e-9, ReplayScale::LargerOrOne, ReplayNonFinite::NanMatchesNan);
        assert!(nan.close(f64::NAN, f64::NAN));
        assert!(!nan.close(f64::NAN, 1.0));
        assert!(!nan.close(f64::INFINITY, f64::INFINITY));
        let bits = ReplayTolerance::new(1e-9, ReplayScale::LargerOrOne, ReplayNonFinite::SameBits);
        assert!(bits.close(f64::INFINITY, f64::INFINITY));
        assert!(!bits.close(f64::INFINITY, f64::NEG_INFINITY));
    }

    #[test]
    fn slices_must_agree_in_length_and_position() {
        let policy = ReplayTolerance::new(1e-9, ReplayScale::Absolute, ReplayNonFinite::Refuse);
        assert!(policy.all_close(&[1.0, 2.0], &[1.0, 2.0]));
        assert!(!policy.all_close(&[1.0], &[1.0, 2.0]));
        assert!(!policy.all_close(&[1.0, 2.5], &[1.0, 2.0]));
    }
}
