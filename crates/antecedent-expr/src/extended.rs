//! Extended-range scalar arithmetic and the ratio/sum kernel shared by the evaluators.
//!
//! The compiled evaluator multiplies and sums densities whose individual magnitudes can leave
//! the `f64` range even when the final quotient is perfectly ordinary: a product of many small
//! probabilities underflows to zero, and a later ratio would then report a zero denominator.
//! [`Extended`] keeps a normalized mantissa and a separate integer exponent, so products,
//! quotients and sums of such terms cancel before anything is converted back to `f64`.
//!
//! [`Quantity`] pairs a value with the magnitude its rounding error scales with. That magnitude
//! is propagated through every operation, so a denominator that is a signed `SumOut`, a product
//! containing a difference, or a nested ratio is guarded against cancellation exactly like a
//! plain difference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::numeric::CompensatedSum;

/// A quantity no larger than this many machine epsilons of its error scale is
/// indistinguishable from zero: each contributing term carries at least half an ulp of rounding
/// error.
pub(crate) const CANCELLATION_TOLERANCE: f64 = 8.0 * f64::EPSILON;

/// Terms with a binary exponent inside this window are accumulated directly as `f64`. A term
/// below `2^-SAFE_EXPONENT` loses no precision and the sum of any realistic count of terms
/// below `2^SAFE_EXPONENT` stays finite.
const SAFE_EXPONENT: i64 = 960;
const MAX_EXPONENT_SHIFT: i64 = 5000;
const SUBNORMAL_LIFT: f64 = 18_446_744_073_709_551_616.0;
const EXPONENT_BIAS: i64 = 1023;

/// `2^exponent` for an exponent whose result is a normal `f64` (`-1022..=1023`).
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "the biased exponent of a normal power of two lies in 1..=2046"
)]
fn pow2(exponent: i64) -> f64 {
    debug_assert!((-1022..=1023).contains(&exponent));
    f64::from_bits(((exponent + EXPONENT_BIAS) as u64) << 52)
}

/// `x * 2^exponent`, rounded once at the end.
fn ldexp(mut x: f64, exponent: i64) -> f64 {
    let mut remaining = exponent.clamp(-MAX_EXPONENT_SHIFT, MAX_EXPONENT_SHIFT);
    while remaining > 1000 {
        x *= pow2(1000);
        remaining -= 1000;
        if x == 0.0 || !x.is_finite() {
            return x;
        }
    }
    while remaining < -1000 {
        x *= pow2(-1000);
        remaining += 1000;
        if x == 0.0 || !x.is_finite() {
            return x;
        }
    }
    x * pow2(remaining)
}

/// Split a finite nonzero `x` into `(m, e)` with `|m|` in `[0.5, 1)` and `x = m * 2^e`.
/// Zero and non-finite values are returned unchanged with exponent zero.
#[allow(
    clippy::cast_possible_wrap,
    reason = "the biased exponent field is eleven bits"
)]
fn split(x: f64) -> (f64, i64) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i64;
    if biased == 0 {
        let (mantissa, exponent) = split(x * SUBNORMAL_LIFT);
        return (mantissa, exponent - 64);
    }
    let mantissa = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52));
    (mantissa, biased - 1022)
}

/// A real number `mantissa * 2^exponent` with `|mantissa|` in `[0.5, 1)`, or zero, or a
/// non-finite mantissa carried verbatim.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Extended {
    mantissa: f64,
    exponent: i64,
}

impl Extended {
    pub(crate) const ZERO: Self = Self { mantissa: 0.0, exponent: 0 };
    pub(crate) const ONE: Self = Self { mantissa: 0.5, exponent: 1 };

    fn normalized(mantissa: f64, exponent: i64) -> Self {
        let (mantissa, shift) = split(mantissa);
        if mantissa == 0.0 || !mantissa.is_finite() {
            return Self { mantissa, exponent: 0 };
        }
        Self { mantissa, exponent: exponent + shift }
    }

    pub(crate) fn from_f64(x: f64) -> Self {
        let (mantissa, exponent) = split(x);
        Self { mantissa, exponent }
    }

    /// Nearest `f64`: overflow gives an infinity, underflow rounds to a (signed) zero or a
    /// subnormal.
    pub(crate) fn to_f64(self) -> f64 {
        if self.mantissa == 0.0 || !self.mantissa.is_finite() {
            return self.mantissa;
        }
        ldexp(self.mantissa, self.exponent)
    }

    pub(crate) fn is_zero(self) -> bool {
        self.mantissa == 0.0
    }

    pub(crate) fn abs(self) -> Self {
        Self { mantissa: self.mantissa.abs(), exponent: self.exponent }
    }

    pub(crate) fn negated(self) -> Self {
        Self { mantissa: -self.mantissa, exponent: self.exponent }
    }

    pub(crate) fn product(self, other: Self) -> Self {
        Self::normalized(self.mantissa * other.mantissa, self.exponent + other.exponent)
    }

    pub(crate) fn product_f64(self, factor: f64) -> Self {
        self.product(Self::from_f64(factor))
    }

    pub(crate) fn quotient(self, other: Self) -> Self {
        Self::normalized(self.mantissa / other.mantissa, self.exponent - other.exponent)
    }

    pub(crate) fn plus(self, other: Self) -> Self {
        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }
        if !self.mantissa.is_finite() || !other.mantissa.is_finite() {
            return Self { mantissa: self.mantissa + other.mantissa, exponent: 0 };
        }
        let (big, small) = if self.exponent >= other.exponent { (self, other) } else { (other, self) };
        let lift = ldexp(small.mantissa, small.exponent - big.exponent);
        Self::normalized(big.mantissa + lift, big.exponent)
    }

    /// `|self| <= |other|`; false when either side is `NaN`.
    pub(crate) fn abs_le(self, other: Self) -> bool {
        let (a, b) = (self.mantissa.abs(), other.mantissa.abs());
        if a.is_nan() || b.is_nan() {
            return false;
        }
        if a == 0.0 {
            return true;
        }
        if b == 0.0 {
            return false;
        }
        if !a.is_finite() || !b.is_finite() {
            return a <= b;
        }
        if self.exponent == other.exponent { a <= b } else { self.exponent < other.exponent }
    }
}

/// Compensated accumulator over [`Extended`] terms.
///
/// Terms in the ordinary `f64` range feed a Neumaier accumulator directly. Terms outside it
/// are kept aside and merged against a common exponent when the total is read, so neither a
/// term nor a partial sum overflows or underflows.
#[derive(Clone, Debug, Default)]
pub(crate) struct ExtendedSum {
    direct: CompensatedSum,
    outliers: Vec<Extended>,
    /// Sum of the non-finite terms; stays zero while every term is finite.
    special: f64,
}

impl ExtendedSum {
    pub(crate) fn add(&mut self, term: Extended) {
        if term.mantissa == 0.0 {
            return;
        }
        if !term.mantissa.is_finite() {
            self.special += term.mantissa;
        } else if term.exponent.abs() <= SAFE_EXPONENT {
            self.direct.add(ldexp(term.mantissa, term.exponent));
        } else {
            self.outliers.push(term);
        }
    }

    pub(crate) fn total(&self) -> Extended {
        if self.special != 0.0 {
            return Extended::from_f64(self.special);
        }
        let direct = Extended::from_f64(self.direct.total());
        if self.outliers.is_empty() {
            return direct;
        }
        let reference = self
            .outliers
            .iter()
            .chain(std::iter::once(&direct))
            .filter(|t| !t.is_zero())
            .map(|t| t.exponent)
            .max()
            .unwrap_or(0);
        let mut merged = CompensatedSum::default();
        for term in self.outliers.iter().chain(std::iter::once(&direct)) {
            if !term.is_zero() {
                merged.add(ldexp(term.mantissa, term.exponent - reference));
            }
        }
        Extended::normalized(merged.total(), reference)
    }
}

/// A value together with the magnitude its accumulated rounding error scales with.
///
/// Invariant: `scale >= |value|` once tracked; a quantity that was never tracked carries a zero
/// scale and must not be used for a cancellation test.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Quantity {
    pub(crate) value: Extended,
    pub(crate) scale: Extended,
}

impl Quantity {
    /// Exact multiplicative identity.
    pub(crate) const ONE: Self = Self { value: Extended::ONE, scale: Extended::ZERO };

    /// A directly supplied number: its own magnitude is its error scale.
    pub(crate) fn leaf(x: f64, track: bool) -> Self {
        let value = Extended::from_f64(x);
        Self { value, scale: if track { value.abs() } else { Extended::ZERO } }
    }

    pub(crate) fn times(self, other: Self, track: bool) -> Self {
        let value = self.value.product(other.value);
        let scale = if track {
            other.value.abs().product(self.scale).plus(self.value.abs().product(other.scale))
        } else {
            Extended::ZERO
        };
        Self { value, scale }
    }

    pub(crate) fn difference(self, other: Self, track: bool) -> Self {
        Self {
            value: self.value.plus(other.value.negated()),
            scale: if track { self.scale.plus(other.scale) } else { Extended::ZERO },
        }
    }

    /// `self / denominator`, or `None` when the denominator is zero to working precision:
    /// exactly zero, or no larger than [`CANCELLATION_TOLERANCE`] of its own error scale.
    pub(crate) fn divide(self, denominator: Self, track: bool) -> Option<Self> {
        let floor = denominator.scale.product_f64(CANCELLATION_TOLERANCE);
        if denominator.value.is_zero() || denominator.value.abs_le(floor) {
            return None;
        }
        let value = self.value.quotient(denominator.value);
        let scale = if track {
            self.scale
                .plus(value.abs().product(denominator.scale))
                .quotient(denominator.value.abs())
        } else {
            Extended::ZERO
        };
        Some(Self { value, scale })
    }
}

/// Whether a denominator `den` whose rounding error scales with `scale` is zero to working
/// precision. The `f64` form of [`Quantity::divide`]'s guard, for evaluators that carry plain
/// values: pass `scale == |den|` when the denominator is built only from non-negative terms,
/// which cannot cancel.
pub(crate) fn denominator_is_degenerate(den: f64, scale: f64) -> bool {
    den == 0.0 || den.abs() <= CANCELLATION_TOLERANCE * scale
}

/// Compensated sum of [`Quantity`] terms with an accumulated error scale.
#[derive(Clone, Debug, Default)]
pub(crate) struct QuantitySum {
    value: ExtendedSum,
    scale: ExtendedSum,
    track: bool,
}

impl QuantitySum {
    pub(crate) fn new(track: bool) -> Self {
        Self { value: ExtendedSum::default(), scale: ExtendedSum::default(), track }
    }

    pub(crate) fn add(&mut self, term: Quantity) {
        self.value.add(term.value);
        if self.track {
            self.scale.add(term.scale);
        }
    }

    pub(crate) fn total(&self) -> Quantity {
        Quantity {
            value: self.value.total(),
            scale: if self.track { self.scale.total() } else { Extended::ZERO },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_and_rejoin_round_trip_including_subnormals() {
        for x in [1.0, -3.5, 1e-300, 5e-324, 1.7e308, -2.2250738585072014e-308, 0.1] {
            assert_eq!(Extended::from_f64(x).to_f64().to_bits(), x.to_bits(), "{x}");
        }
        assert_eq!(Extended::from_f64(0.0).to_f64(), 0.0);
        assert!(Extended::from_f64(f64::NAN).to_f64().is_nan());
        assert_eq!(Extended::from_f64(f64::INFINITY).to_f64(), f64::INFINITY);
    }

    #[test]
    fn products_beyond_f64_range_cancel_in_a_quotient() {
        let tiny = Extended::from_f64(1e-200);
        let numerator = tiny.product(tiny).product(Extended::from_f64(3.0));
        let denominator = tiny.product(tiny).product(Extended::from_f64(2.0));
        assert_eq!(numerator.to_f64(), 0.0, "the numerator alone underflows");
        let quotient = numerator.quotient(denominator).to_f64();
        assert!((quotient - 1.5).abs() < 1e-15, "{quotient}");
    }

    #[test]
    fn sum_of_out_of_range_terms_neither_overflows_nor_underflows() {
        let big = Extended::from_f64(1e300).product(Extended::from_f64(1e300));
        let small = Extended::from_f64(1e-300).product(Extended::from_f64(1e-300));
        let mut sum = ExtendedSum::default();
        sum.add(big);
        sum.add(big);
        sum.add(big.negated());
        assert_eq!(sum.total().quotient(big).to_f64(), 1.0);
        let mut tiny = ExtendedSum::default();
        tiny.add(small);
        tiny.add(small);
        assert_eq!(tiny.total().quotient(small).to_f64(), 2.0);
    }

    #[test]
    fn compensated_sum_keeps_digits_a_running_total_drops() {
        let mut sum = ExtendedSum::default();
        for x in [1e16, 1.0, -1e16, 1.0] {
            sum.add(Extended::from_f64(x));
        }
        assert_eq!(sum.total().to_f64(), 2.0);
    }

    #[test]
    fn difference_matches_f64_subtraction() {
        let a = Extended::from_f64(0.9);
        let b = Extended::from_f64(0.4);
        assert_eq!(a.plus(b.negated()).to_f64().to_bits(), (0.9f64 - 0.4f64).to_bits());
    }

    #[test]
    fn guard_catches_cancellation_through_a_signed_sum_and_a_product() {
        let third = Quantity::leaf(0.1 + 0.2, true);
        let other = Quantity::leaf(0.3, true);
        let difference = third.difference(other, true);
        assert!(Quantity::ONE.divide(difference, false).is_none());
        let scaled = difference.times(Quantity::leaf(0.7, true), true);
        assert!(Quantity::ONE.divide(scaled, false).is_none());
        assert!(Quantity::ONE.divide(Quantity::leaf(0.5, true), false).is_some());
    }

    #[test]
    fn ordering_compares_magnitude_across_exponents() {
        assert!(Extended::from_f64(1e-10).abs_le(Extended::from_f64(-1e-9)));
        assert!(!Extended::from_f64(1e-9).abs_le(Extended::from_f64(1e-10)));
        assert!(Extended::ZERO.abs_le(Extended::ZERO));
        assert!(!Extended::from_f64(f64::NAN).abs_le(Extended::ONE));
    }
}
