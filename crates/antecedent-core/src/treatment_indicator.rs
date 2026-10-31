//! The binarization rule for a validated 0/1 treatment indicator column.
//!
//! Every layer that reads a treatment column back as an arm uses this one predicate, so the
//! fitting side and the scoring side cannot disagree about which rows are treated.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Level above which an indicator value is read as the treated arm.
pub const TREATED_THRESHOLD: f64 = 0.5;

/// Whether an indicator value is read as the treated arm: values above one half are treated,
/// everything else (including a non-finite value) is control.
#[must_use]
pub fn is_treated_indicator(indicator: f64) -> bool {
    indicator > TREATED_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_threshold_is_exclusive_and_non_finite_values_are_control() {
        assert!(is_treated_indicator(1.0));
        assert!(is_treated_indicator(0.5 + 1e-12));
        assert!(!is_treated_indicator(0.5));
        assert!(!is_treated_indicator(0.0));
        assert!(!is_treated_indicator(f64::NAN));
    }
}
