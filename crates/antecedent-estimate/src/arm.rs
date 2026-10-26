//! The binary treatment arm of a validated 0/1 indicator column.
//!
//! Estimators receive treatment as an `f64` column that preparation has already validated as
//! a two-level indicator. Reading that column back with ad-hoc `x > 0.5` and `x == 1.0` tests
//! scatters the binarization rule over the crate; [`TreatmentArm`] is the one place that rule
//! lives, so a row's arm is a typed value and the two spellings cannot disagree.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Level above which an indicator value is read as the treated arm.
const TREATED_THRESHOLD: f64 = 0.5;

/// The arm a binary treatment indicator places a row in.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TreatmentArm {
    /// The control arm (indicator `0`).
    Control,
    /// The treated arm (indicator `1`).
    Treated,
}

impl TreatmentArm {
    /// The arm of a validated indicator value: values above one half are treated.
    #[must_use]
    pub fn of(indicator: f64) -> Self {
        if indicator > TREATED_THRESHOLD { Self::Treated } else { Self::Control }
    }

    /// The arm of a value that must be exactly `0.0` or `1.0`; any other value (including
    /// non-finite ones) is `None`.
    #[must_use]
    pub fn exact(indicator: f64) -> Option<Self> {
        if indicator == 0.0 {
            Some(Self::Control)
        } else if indicator == 1.0 {
            Some(Self::Treated)
        } else {
            None
        }
    }

    /// Whether this is the treated arm.
    #[must_use]
    pub const fn is_treated(self) -> bool {
        matches!(self, Self::Treated)
    }

    /// The indicator value of the arm (`0.0` or `1.0`).
    #[must_use]
    pub const fn indicator(self) -> f64 {
        match self {
            Self::Control => 0.0,
            Self::Treated => 1.0,
        }
    }

    /// The arm of every entry of an indicator column.
    #[must_use]
    pub fn classify(indicators: &[f64]) -> Vec<Self> {
        indicators.iter().map(|&x| Self::of(x)).collect()
    }

    /// Whether an indicator value is read as the treated arm.
    #[must_use]
    pub fn is_treated_indicator(indicator: f64) -> bool {
        Self::of(indicator).is_treated()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indicator_round_trips_through_arm() {
        assert_eq!(TreatmentArm::of(0.0), TreatmentArm::Control);
        assert_eq!(TreatmentArm::of(1.0), TreatmentArm::Treated);
        assert_eq!(TreatmentArm::Treated.indicator(), 1.0);
        assert_eq!(TreatmentArm::Control.indicator(), 0.0);
    }

    #[test]
    fn exact_refuses_other_values() {
        assert_eq!(TreatmentArm::exact(0.0), Some(TreatmentArm::Control));
        assert_eq!(TreatmentArm::exact(-0.0), Some(TreatmentArm::Control));
        assert_eq!(TreatmentArm::exact(1.0), Some(TreatmentArm::Treated));
        assert_eq!(TreatmentArm::exact(0.5), None);
        assert_eq!(TreatmentArm::exact(f64::NAN), None);
    }

    #[test]
    fn classify_reads_a_column() {
        let arms = TreatmentArm::classify(&[0.0, 1.0, 1.0]);
        assert_eq!(arms, [TreatmentArm::Control, TreatmentArm::Treated, TreatmentArm::Treated]);
    }
}
