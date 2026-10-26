//! Neumaier compensated accumulation for the contract's own validation arithmetic.
//!
//! The contract crate sits below the numerical kernels, so it carries this small accumulator
//! rather than depending on them. A plain left-to-right `f64` sum drops low-order bits when
//! terms differ greatly in magnitude or cancel; the correction term keeps them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// A Neumaier compensated running sum.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Compensated {
    total: f64,
    correction: f64,
}

impl Compensated {
    /// Add one term.
    pub(crate) fn add(&mut self, value: f64) {
        let next = self.total + value;
        self.correction += if self.total.abs() >= value.abs() {
            (self.total - next) + value
        } else {
            (value - next) + self.total
        };
        self.total = next;
    }

    /// The compensated total.
    ///
    /// A `NaN` term gives `NaN`. An infinite partial sum makes the correction `NaN`, so a
    /// non-finite compensated total is reported as the plain (uncompensated) running total,
    /// which keeps `±∞` meaningful.
    pub(crate) fn total(self) -> f64 {
        let compensated = self.total + self.correction;
        if compensated.is_finite() || self.total.is_nan() { compensated } else { self.total }
    }
}

/// Compensated sum of `values` in iteration order.
pub(crate) fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut acc = Compensated::default();
    for value in values {
        acc.add(value);
    }
    acc.total()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_a_unit_next_to_a_huge_shift() {
        let values = [1e16, 1.0, -1e16];
        assert_eq!(values.iter().sum::<f64>(), 0.0);
        assert_eq!(compensated_sum(values), 1.0);
    }

    #[test]
    fn infinities_and_nan_propagate() {
        assert_eq!(compensated_sum([1.0, f64::INFINITY]), f64::INFINITY);
        assert!(compensated_sum([1.0, f64::NAN]).is_nan());
        assert_eq!(compensated_sum(std::iter::empty()), 0.0);
    }
}
