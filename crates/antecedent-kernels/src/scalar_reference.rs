// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scalar reference and portable masked weighted sum.
//!
//! Both reductions walk terms in index order and add `weight * value` as a
//! separate multiply and add. The absolute tolerance is an explicit `f64`
//! bound: it is not scaled by an FMA rounding term, and neither reduction
//! contracts the product into a fused multiply-add. The same inputs therefore
//! produce the same output bits.
//!
//! The portable loop blocks that same order and finishes with a scalar tail.
//! It is not an AVX2, NEON, or AVX-512 kernel. Those requests are refused
//! until this tolerance passes and N15's measurement is recorded.

// Parent wires this module into the crate root. Later kernels will call it.
#![allow(dead_code, reason = "parent wires this module; later kernels will call it")]

use std::fmt;

/// Terms of `Σ mask · weight · value` in index order.
///
/// A `false` mask drops the term before the weight is read. A zero weight,
/// including `-0.0`, also drops the term, so it does not poison the sum when
/// the value is non-finite.
#[derive(Clone, Copy, Debug)]
pub struct WeightedMask<'a> {
    /// Values in index order.
    pub values: &'a [f64],
    /// Weights in the same order as [`Self::values`].
    pub weights: &'a [f64],
    /// `true` keeps the term. `false` drops it.
    pub mask: &'a [bool],
}

impl<'a> WeightedMask<'a> {
    /// Borrow the three slices. Lengths are checked when the sum runs.
    #[must_use]
    pub const fn new(values: &'a [f64], weights: &'a [f64], mask: &'a [bool]) -> Self {
        Self { values, weights, mask }
    }
}

/// Absolute `f64` distance allowed between the scalar reference and the
/// portable reduction.
///
/// This constant does not depend on FMA. It is not `n * EPSILON`, and it is
/// not widened by a fused multiply-add error term. Both reductions use the
/// same index order and a separate multiply and add, so the allowed distance
/// is zero.
pub const REDUCTION_ABS_TOLERANCE: f64 = 0.0;

/// Portable block length. Three `f64` terms are not an AVX2 lane (4), a NEON
/// lane (2), or an AVX-512 lane (8). A length that is a multiple of this
/// block has an empty tail; any other length runs the tail loop.
pub const PORTABLE_BLOCK_LEN: usize = 3;

/// What must succeed before an architecture-specific kernel can be considered.
pub const NEXT_ATTEMPT_BEFORE_ARCH_PROMOTION: &str =
    "pass N16's tolerance, then N15's measurement, before any promotion";

/// Why a masked weighted sum was not computed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReductionError {
    /// `values`, `weights`, and `mask` do not share one length.
    LengthMismatch {
        /// Length of [`WeightedMask::values`].
        values: usize,
        /// Length of [`WeightedMask::weights`].
        weights: usize,
        /// Length of [`WeightedMask::mask`].
        mask: usize,
    },
}

impl fmt::Display for ReductionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { values, weights, mask } => {
                write!(formatter, "masked weighted sum lengths differ: values {values}")?;
                write!(formatter, ", weights {weights}, mask {mask}")
            }
        }
    }
}

impl std::error::Error for ReductionError {}

/// Instruction set a caller asked to promote. None of these are implemented.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchIsa {
    /// `x86_64` AVX2.
    Avx2,
    /// `AArch64` NEON.
    Neon,
    /// `x86_64` AVX-512.
    Avx512,
}

/// Refusal of an architecture-specific promotion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchRefusal {
    requested: ArchIsa,
    next_attempt: &'static str,
}

impl ArchRefusal {
    /// The ISA that was requested.
    #[must_use]
    pub const fn requested(self) -> ArchIsa {
        self.requested
    }

    /// The next attempt that must pass before that ISA is eligible.
    #[must_use]
    pub const fn next_attempt(self) -> &'static str {
        self.next_attempt
    }
}

impl fmt::Display for ArchRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let isa = match self.requested {
            ArchIsa::Avx2 => "AVX2",
            ArchIsa::Neon => "NEON",
            ArchIsa::Avx512 => "AVX-512",
        };
        let next_attempt = self.next_attempt;
        write!(formatter, "{isa} is refused; next attempt: {next_attempt}")
    }
}

impl std::error::Error for ArchRefusal {}

/// Scalar reference: one left-to-right accumulator.
///
/// Deterministic: the same inputs produce the same `f64` bits. Tolerance
/// against [`portable_masked_weighted_sum`] is [`REDUCTION_ABS_TOLERANCE`].
///
/// # Errors
///
/// [`ReductionError::LengthMismatch`] when the three slices differ in length.
#[must_use = "a length mismatch is an error, not an empty sum"]
pub fn scalar_masked_weighted_sum(inputs: WeightedMask<'_>) -> Result<f64, ReductionError> {
    let n = check_len(inputs)?;
    let mut acc = 0.0;
    let mut i = 0;
    while i < n {
        acc = fold_term(acc, inputs.values[i], inputs.weights[i], inputs.mask[i]);
        i += 1;
    }
    Ok(acc)
}

/// Portable reduction of the same sum.
///
/// Full blocks of [`PORTABLE_BLOCK_LEN`] run first, still in index order, and
/// the remainder is a scalar tail. This is not architecture-specific SIMD and
/// does not use FMA. The same inputs produce the same bits as
/// [`scalar_masked_weighted_sum`].
///
/// # Errors
///
/// [`ReductionError::LengthMismatch`] when the three slices differ in length.
#[must_use = "a length mismatch is an error, not an empty sum"]
pub fn portable_masked_weighted_sum(inputs: WeightedMask<'_>) -> Result<f64, ReductionError> {
    let n = check_len(inputs)?;
    let mut acc = 0.0;
    let mut i = 0;
    while n - i >= PORTABLE_BLOCK_LEN {
        acc = fold_term(acc, inputs.values[i], inputs.weights[i], inputs.mask[i]);
        acc = fold_term(acc, inputs.values[i + 1], inputs.weights[i + 1], inputs.mask[i + 1]);
        acc = fold_term(acc, inputs.values[i + 2], inputs.weights[i + 2], inputs.mask[i + 2]);
        i += PORTABLE_BLOCK_LEN;
    }
    while i < n {
        acc = fold_term(acc, inputs.values[i], inputs.weights[i], inputs.mask[i]);
        i += 1;
    }
    Ok(acc)
}

/// Whether `actual` lies within [`REDUCTION_ABS_TOLERANCE`] of `reference`.
///
/// NaNs match each other. Infinities match only by sign. The comparison is a
/// pure absolute `f64` bound and does not depend on FMA.
#[must_use = "the tolerance decision is the differential result"]
pub const fn within_reduction_tolerance(actual: f64, reference: f64) -> bool {
    if actual.is_nan() || reference.is_nan() {
        return actual.is_nan() && reference.is_nan();
    }
    if actual.is_infinite() || reference.is_infinite() {
        return actual.to_bits() == reference.to_bits();
    }
    (actual - reference).abs() <= REDUCTION_ABS_TOLERANCE
}

/// Refuse AVX2, NEON, and AVX-512.
///
/// This module has no architecture-specific kernel. The error's next attempt
/// is [`NEXT_ATTEMPT_BEFORE_ARCH_PROMOTION`]: pass N16's tolerance, then N15's
/// measurement, before any promotion.
#[must_use = "AVX2, NEON, and AVX-512 stay refused until N16 and N15 pass"]
pub const fn refuse_arch_promotion(requested: ArchIsa) -> Result<(), ArchRefusal> {
    Err(ArchRefusal { requested, next_attempt: NEXT_ATTEMPT_BEFORE_ARCH_PROMOTION })
}

const fn check_len(inputs: WeightedMask<'_>) -> Result<usize, ReductionError> {
    let values = inputs.values.len();
    let weights = inputs.weights.len();
    let mask = inputs.mask.len();
    if values != weights || values != mask {
        return Err(ReductionError::LengthMismatch { values, weights, mask });
    }
    Ok(values)
}

const fn weight_is_zero(weight: f64) -> bool {
    let bits = weight.to_bits();
    bits == 0 || bits == 0x8000_0000_0000_0000
}

#[allow(
    clippy::suboptimal_flops,
    reason = "FMA is outside this tolerance; the product is a separate rounding"
)]
const fn fold_term(acc: f64, value: f64, weight: f64, keep: bool) -> f64 {
    if !keep {
        return acc;
    }
    if weight_is_zero(weight) {
        return acc;
    }
    acc + weight * value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_agree(inputs: WeightedMask<'_>) {
        let scalar = scalar_masked_weighted_sum(inputs).unwrap();
        let portable = portable_masked_weighted_sum(inputs).unwrap();
        assert!(
            within_reduction_tolerance(portable, scalar),
            "portable={portable:?} scalar={scalar:?} tol={REDUCTION_ABS_TOLERANCE}"
        );
    }

    #[test]
    fn aligned_lengths_match_within_f64_tolerance() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = [1.0; 6];
        let mask = [true; 6];
        let inputs = WeightedMask::new(&values, &weights, &mask);
        assert_eq!(values.len() % PORTABLE_BLOCK_LEN, 0);
        let scalar = scalar_masked_weighted_sum(inputs).unwrap();
        assert!(within_reduction_tolerance(scalar, 21.0));
        assert_agree(inputs);

        let empty = WeightedMask::new(&[], &[], &[]);
        assert!(within_reduction_tolerance(scalar_masked_weighted_sum(empty).unwrap(), 0.0));
        assert_agree(empty);
    }

    #[test]
    fn tail_lengths_match_within_f64_tolerance() {
        let lengths = [1usize, 2, 4, 5];
        for n in lengths {
            assert_ne!(n % PORTABLE_BLOCK_LEN, 0);
            let values: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let weights = vec![1.0; n];
            let mask = vec![true; n];
            let inputs = WeightedMask::new(&values, &weights, &mask);
            let expected = (n * (n + 1) / 2) as f64;
            let scalar = scalar_masked_weighted_sum(inputs).unwrap();
            assert!(
                within_reduction_tolerance(scalar, expected),
                "n={n} scalar={scalar} expected={expected}"
            );
            assert_agree(inputs);
        }
    }

    #[test]
    fn mask_drops_values() {
        let values = [1.0, 1_000.0, 3.0, 4.0, 5.0, 6.0];
        let weights = [1.0; 6];
        let mask = [true, false, true, true, true, true];
        let inputs = WeightedMask::new(&values, &weights, &mask);
        let scalar = scalar_masked_weighted_sum(inputs).unwrap();
        assert!(within_reduction_tolerance(scalar, 19.0));
        assert!(!within_reduction_tolerance(scalar, 1_019.0));
        assert_agree(inputs);
    }

    #[test]
    fn nonfinite_value_matches() {
        let values = [1.0, f64::NAN, 2.0];
        let weights = [1.0; 3];
        let mask = [true; 3];
        let inputs = WeightedMask::new(&values, &weights, &mask);
        let scalar = scalar_masked_weighted_sum(inputs).unwrap();
        assert!(scalar.is_nan());
        assert_agree(inputs);

        let dropped = [false, false, true];
        let kept_tail = WeightedMask::new(&values, &weights, &dropped);
        let finite = scalar_masked_weighted_sum(kept_tail).unwrap();
        assert!(within_reduction_tolerance(finite, 2.0));
        assert_agree(kept_tail);
    }

    #[test]
    fn zero_weight_matches() {
        let values = [4.0, 1.0e16, 3.0];
        let weights = [2.0, -0.0, 1.0];
        let mask = [true; 3];
        let inputs = WeightedMask::new(&values, &weights, &mask);
        let scalar = scalar_masked_weighted_sum(inputs).unwrap();
        assert!(within_reduction_tolerance(scalar, 11.0));
        assert_agree(inputs);
    }

    #[test]
    fn cancellation_matches() {
        let values = [1.0, 1.0e16, -1.0e16, 1.0, 1.0e16, -1.0e16];
        let weights = [1.0; 6];
        let mask = [true; 6];
        let inputs = WeightedMask::new(&values, &weights, &mask);
        let scalar = scalar_masked_weighted_sum(inputs).unwrap();
        assert!(within_reduction_tolerance(scalar, 0.0));
        assert_agree(inputs);
    }

    #[test]
    fn same_inputs_produce_the_same_bits() {
        let values = [1.0, 1.0e16, -1.0e16, f64::NAN];
        let weights = [1.0, 1.0, 1.0, 0.0];
        let mask = [true; 4];
        let inputs = WeightedMask::new(&values, &weights, &mask);
        let scalar_a = scalar_masked_weighted_sum(inputs).unwrap();
        let scalar_b = scalar_masked_weighted_sum(inputs).unwrap();
        let portable_a = portable_masked_weighted_sum(inputs).unwrap();
        let portable_b = portable_masked_weighted_sum(inputs).unwrap();
        assert_eq!(scalar_a.to_bits(), scalar_b.to_bits());
        assert_eq!(portable_a.to_bits(), portable_b.to_bits());
        assert_eq!(scalar_a.to_bits(), portable_a.to_bits());
        assert!(within_reduction_tolerance(scalar_a, 0.0));

        let nan = WeightedMask::new(&[f64::NAN], &[1.0], &[true]);
        let scalar_nan = scalar_masked_weighted_sum(nan).unwrap();
        let portable_nan = portable_masked_weighted_sum(nan).unwrap();
        assert_eq!(scalar_nan.to_bits(), portable_nan.to_bits());
        assert_eq!(scalar_nan.to_bits(), scalar_masked_weighted_sum(nan).unwrap().to_bits());
    }

    #[test]
    fn length_mismatch_is_an_error() {
        let inputs = WeightedMask::new(&[1.0], &[1.0, 2.0], &[true]);
        let expected = ReductionError::LengthMismatch { values: 1, weights: 2, mask: 1 };
        assert_eq!(scalar_masked_weighted_sum(inputs), Err(expected));
        assert_eq!(portable_masked_weighted_sum(inputs), Err(expected));
    }

    #[test]
    fn avx2_neon_and_avx512_are_refused() {
        let requests = [ArchIsa::Avx2, ArchIsa::Neon, ArchIsa::Avx512];
        for requested in requests {
            let refusal = refuse_arch_promotion(requested).expect_err("arch promotion is refused");
            assert_eq!(refusal.requested(), requested);
            assert_eq!(refusal.next_attempt(), NEXT_ATTEMPT_BEFORE_ARCH_PROMOTION);
            assert_eq!(
                refusal.next_attempt(),
                "pass N16's tolerance, then N15's measurement, before any promotion"
            );
        }
    }
}
