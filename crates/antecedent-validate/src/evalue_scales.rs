//! E-values on the risk-ratio, odds-ratio, hazard-ratio and standardized-mean-difference
//! scales, with the confidence-limit E-value (`VanderWeele` and Ding 2017).
//!
//! The E-value of a risk ratio `RR ≥ 1` is `E = RR + sqrt(RR·(RR − 1))` (a protective
//! `RR < 1` is inverted first): the minimum strength of association, on the risk-ratio
//! scale, that an unmeasured confounder would need with both the treatment and the outcome,
//! conditional on the measured covariates, to explain the estimate away. Other scales are
//! first converted to an (approximate) risk ratio as `VanderWeele` and Ding (2017, Table 2)
//! and `VanderWeele` (2017, square-root transformation) prescribe:
//!
//! | scale | outcome | risk-ratio conversion |
//! |---|---|---|
//! | odds ratio | rare (< 15%) | `RR ≈ OR` |
//! | odds ratio | common | `RR ≈ sqrt(OR)` |
//! | hazard ratio | rare | `RR ≈ HR` |
//! | hazard ratio | common | `RR ≈ (1 − 0.5^sqrt(HR)) / (1 − 0.5^sqrt(1/HR))` |
//! | standardized mean difference `d` (SE `s`) | continuous | `RR ≈ exp(0.91·d)`, limits `exp(0.91·d ± 1.78·s)` |
//!
//! The confidence-limit E-value is the E-value of the interval limit nearest the null, and
//! `1` when the interval contains the null. [`evalue_range`] reports both as an
//! [`AssumptionRange`] on the declared target, with the risk-ratio strength at which a
//! confounder explains the point estimate and the interval limit away as the two break-even
//! values.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ValidationError;
use crate::assumption_range::{AssumptionRange, RangePoint, SensitivityAssumption};
use crate::target_native::RefutationTarget;

/// Scale of the reported effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EffectScale {
    /// Risk ratio.
    RiskRatio,
    /// Odds ratio; `common_outcome` when the outcome prevalence is 15% or more.
    OddsRatio {
        /// Outcome is common in every stratum.
        common_outcome: bool,
    },
    /// Hazard ratio; `common_outcome` when the outcome is common by the end of follow-up.
    HazardRatio {
        /// Outcome is common by the end of follow-up.
        common_outcome: bool,
    },
    /// Standardized mean difference `d` of a continuous outcome (mean difference divided by
    /// the outcome's within-group standard deviation); its limits come from its standard error.
    StandardizedMeanDifference,
}

impl EffectScale {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RiskRatio => "risk_ratio",
            Self::OddsRatio { common_outcome: false } => "odds_ratio_rare",
            Self::OddsRatio { common_outcome: true } => "odds_ratio_common",
            Self::HazardRatio { common_outcome: false } => "hazard_ratio_rare",
            Self::HazardRatio { common_outcome: true } => "hazard_ratio_common",
            Self::StandardizedMeanDifference => "standardized_mean_difference",
        }
    }

    /// Parse a [`Self::as_str`] name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "risk_ratio" => Self::RiskRatio,
            "odds_ratio_rare" => Self::OddsRatio { common_outcome: false },
            "odds_ratio_common" => Self::OddsRatio { common_outcome: true },
            "hazard_ratio_rare" => Self::HazardRatio { common_outcome: false },
            "hazard_ratio_common" => Self::HazardRatio { common_outcome: true },
            "standardized_mean_difference" => Self::StandardizedMeanDifference,
            _ => return None,
        })
    }
}

/// The reported effect and its interval on [`EffectScale`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EffectInterval {
    /// A ratio estimate with optional lower and upper confidence limits.
    Ratio {
        /// Point estimate (positive).
        estimate: f64,
        /// Lower limit, when reported.
        lower: Option<f64>,
        /// Upper limit, when reported.
        upper: Option<f64>,
    },
    /// A standardized mean difference with its standard error.
    Standardized {
        /// `d`.
        estimate: f64,
        /// Standard error of `d`, when reported.
        standard_error: Option<f64>,
    },
}

/// E-values of a point estimate and its interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EValueBounds {
    /// Point estimate converted to the risk-ratio scale.
    pub risk_ratio: f64,
    /// Limit nearest the null on the risk-ratio scale, when an interval was given and it
    /// excludes the null; `Some(1.0)` when it contains the null.
    pub risk_ratio_limit: Option<f64>,
    /// E-value of the point estimate.
    pub point: f64,
    /// E-value of the interval limit nearest the null (`1` when the interval contains it).
    pub limit: Option<f64>,
}

/// The E-value of a risk ratio.
#[must_use]
pub fn evalue_of_risk_ratio(rr: f64) -> f64 {
    let rr = if rr >= 1.0 { rr } else { 1.0 / rr };
    // sqrt(rr) * sqrt(rr - 1) does not overflow for very large rr.
    rr + rr.sqrt() * (rr - 1.0).sqrt()
}

/// Approximate risk ratio of a ratio estimate on `scale`.
///
/// # Errors
///
/// A non-positive or non-finite ratio, or a standardized scale (use
/// [`smd_to_risk_ratio`]).
pub fn ratio_to_risk_ratio(value: f64, scale: EffectScale) -> Result<f64, ValidationError> {
    if !(value.is_finite() && value > 0.0) {
        return Err(ValidationError::NotApplicable {
            message: "a ratio effect must be positive and finite",
        });
    }
    Ok(match scale {
        EffectScale::RiskRatio
        | EffectScale::OddsRatio { common_outcome: false }
        | EffectScale::HazardRatio { common_outcome: false } => value,
        EffectScale::OddsRatio { common_outcome: true } => value.sqrt(),
        EffectScale::HazardRatio { common_outcome: true } => {
            (1.0 - 0.5_f64.powf(value.sqrt())) / (1.0 - 0.5_f64.powf((1.0 / value).sqrt()))
        }
        EffectScale::StandardizedMeanDifference => {
            return Err(ValidationError::NotApplicable {
                message: "a standardized mean difference converts through smd_to_risk_ratio",
            });
        }
    })
}

/// `exp(0.91·d)`: the approximate risk ratio of a standardized mean difference.
#[must_use]
pub fn smd_to_risk_ratio(d: f64) -> f64 {
    (0.91 * d).exp()
}

/// E-values of `interval` on `scale`.
///
/// # Errors
///
/// A scale and interval of different kinds, a non-positive ratio or limit, limits on the
/// wrong side of the estimate, or a negative standard error.
pub fn evalue_bounds(
    interval: EffectInterval,
    scale: EffectScale,
) -> Result<EValueBounds, ValidationError> {
    let (rr, lower, upper) = match (interval, scale) {
        (
            EffectInterval::Standardized { estimate, standard_error },
            EffectScale::StandardizedMeanDifference,
        ) => {
            if !estimate.is_finite() {
                return Err(ValidationError::NotApplicable {
                    message: "the standardized mean difference must be finite",
                });
            }
            let limits = match standard_error {
                None => (None, None),
                Some(se) if se.is_finite() && se >= 0.0 => (
                    Some(smd_to_risk_ratio(estimate - 1.78 / 0.91 * se)),
                    Some(smd_to_risk_ratio(estimate + 1.78 / 0.91 * se)),
                ),
                Some(_) => {
                    return Err(ValidationError::NotApplicable {
                        message: "the standard error must be finite and non-negative",
                    });
                }
            };
            (smd_to_risk_ratio(estimate), limits.0, limits.1)
        }
        (EffectInterval::Ratio { estimate, lower, upper }, scale)
            if scale != EffectScale::StandardizedMeanDifference =>
        {
            if lower.is_some_and(|l| l > estimate) || upper.is_some_and(|u| u < estimate) {
                return Err(ValidationError::NotApplicable {
                    message: "confidence limits must bracket the estimate",
                });
            }
            (
                ratio_to_risk_ratio(estimate, scale)?,
                lower.map(|l| ratio_to_risk_ratio(l, scale)).transpose()?,
                upper.map(|u| ratio_to_risk_ratio(u, scale)).transpose()?,
            )
        }
        _ => {
            return Err(ValidationError::NotApplicable {
                message: "a ratio scale takes a ratio interval and the standardized scale takes \
                          an estimate with a standard error",
            });
        }
    };
    // The limit nearest the null: the lower limit of a harmful effect, the upper of a
    // protective one; an interval containing 1 has E-value 1.
    let limit = if rr >= 1.0 {
        lower.map(|l| if l <= 1.0 { 1.0 } else { l })
    } else {
        upper.map(|u| if u >= 1.0 { 1.0 } else { u })
    };
    Ok(EValueBounds {
        risk_ratio: rr,
        risk_ratio_limit: limit,
        point: evalue_of_risk_ratio(rr),
        limit: limit.map(evalue_of_risk_ratio),
    })
}

/// Probed confounder strengths (risk-ratio scale, equal with treatment and outcome).
pub(crate) const STRENGTH_GRID: [f64; 10] = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0];

/// The Ding–`VanderWeele` bounding factor of a confounder with risk ratio `strength` on both
/// treatment and outcome: `B = s² / (2s − 1)`. The most the confounder can shift the risk
/// ratio by, so the adjusted risk ratio is at least `RR / B` (harmful effects).
#[must_use]
pub fn bounding_factor(strength: f64) -> f64 {
    strength * strength / (2.0 * strength - 1.0)
}

/// E-values of `interval` on `scale` as an [`AssumptionRange`] on `target`.
///
/// The tested range is the grid of confounder strengths `s` (equal association with
/// treatment and outcome) from `1` to `10`; at each the worst-case adjusted risk ratio
/// `RR / B(s)` (or `RR·B(s)` for a protective effect, capped at the null) and the adjusted
/// limit nearest the null are reported. The break-even values are the point and limit
/// E-values: the strength at which `B(s)` reaches the risk ratio.
///
/// # Errors
///
/// As [`evalue_bounds`].
pub fn evalue_range(
    target: RefutationTarget,
    interval: EffectInterval,
    scale: EffectScale,
) -> Result<(EValueBounds, AssumptionRange), ValidationError> {
    let bounds = evalue_bounds(interval, scale)?;
    let toward_null = |rr: f64, factor: f64| {
        if rr >= 1.0 { (rr / factor).max(1.0) } else { (rr * factor).min(1.0) }
    };
    let tested = STRENGTH_GRID
        .iter()
        .map(|&s| {
            let factor = bounding_factor(s);
            RangePoint {
                parameter: s,
                target: toward_null(bounds.risk_ratio, factor),
                test: bounds.risk_ratio_limit.map(|limit| toward_null(limit, factor)),
            }
        })
        .collect();
    let range = AssumptionRange::new(
        target,
        SensitivityAssumption::UnmeasuredConfoundingRiskRatio,
        "risk_ratio",
        1.0,
        tested,
        Some(bounds.point),
        bounds.limit,
    );
    Ok((bounds, range))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn breastfeeding_example_of_vanderweele_and_ding() {
        // RR 3.9 (95% CI 1.8 to 8.7): E-value 7.26, confidence-limit E-value 3.0.
        let b = evalue_bounds(
            EffectInterval::Ratio { estimate: 3.9, lower: Some(1.8), upper: Some(8.7) },
            EffectScale::RiskRatio,
        )
        .unwrap();
        assert!(close(b.point, 7.26, 0.005), "{}", b.point);
        assert!(close(b.limit.unwrap(), 3.0, 1e-12), "{:?}", b.limit);
    }

    #[test]
    fn smoking_and_lung_cancer_example() {
        // Cornfield's RR 10.73 has E-value 20.95.
        assert!(close(evalue_of_risk_ratio(10.73), 20.95, 0.005));
    }

    #[test]
    fn antidepressant_hazard_ratio_example() {
        // HR 1.59 (1.17 to 2.17), rare outcome: E-value 2.56, limit 1.62.
        let b = evalue_bounds(
            EffectInterval::Ratio { estimate: 1.59, lower: Some(1.17), upper: Some(2.17) },
            EffectScale::HazardRatio { common_outcome: false },
        )
        .unwrap();
        assert!(close(b.point, 2.56, 0.005), "{}", b.point);
        assert!(close(b.limit.unwrap(), 1.62, 0.005), "{:?}", b.limit);
    }

    #[test]
    fn common_outcome_conversions_follow_the_published_transformations() {
        // Square root of a common-outcome odds ratio.
        let rr = ratio_to_risk_ratio(4.0, EffectScale::OddsRatio { common_outcome: true }).unwrap();
        assert!(close(rr, 2.0, 1e-15));
        // The hazard-ratio transform is 1 at 1 and maps 1/HR to 1/RR.
        let scale = EffectScale::HazardRatio { common_outcome: true };
        assert!(close(ratio_to_risk_ratio(1.0, scale).unwrap(), 1.0, 1e-15));
        let up = ratio_to_risk_ratio(2.5, scale).unwrap();
        let down = ratio_to_risk_ratio(0.4, scale).unwrap();
        assert!(close(up * down, 1.0, 1e-12));
        // (1 - 0.5^sqrt(2)) / (1 - 0.5^sqrt(0.5)) for HR = 2.
        let hr2 = (1.0 - 0.5_f64.powf(2.0_f64.sqrt())) / (1.0 - 0.5_f64.powf(0.5_f64.sqrt()));
        assert!(close(ratio_to_risk_ratio(2.0, scale).unwrap(), hr2, 1e-15));
    }

    #[test]
    fn standardized_difference_uses_091_and_178() {
        let b = evalue_bounds(
            EffectInterval::Standardized { estimate: 0.5, standard_error: Some(0.1) },
            EffectScale::StandardizedMeanDifference,
        )
        .unwrap();
        assert!(close(b.risk_ratio, (0.91_f64 * 0.5).exp(), 1e-15));
        assert!(close(b.risk_ratio_limit.unwrap(), (0.91_f64 * 0.5 - 1.78 * 0.1).exp(), 1e-12));
    }

    #[test]
    fn an_interval_covering_the_null_has_limit_evalue_one() {
        let b = evalue_bounds(
            EffectInterval::Ratio { estimate: 0.8, lower: Some(0.6), upper: Some(1.05) },
            EffectScale::RiskRatio,
        )
        .unwrap();
        assert_eq!(b.limit, Some(1.0));
        assert!(close(b.point, evalue_of_risk_ratio(1.25), 1e-12));
    }

    #[test]
    fn the_bounding_factor_at_the_evalue_explains_the_estimate_away() {
        let rr = 3.9;
        let e = evalue_of_risk_ratio(rr);
        assert!(close(bounding_factor(e), rr, 1e-12));
    }
}
