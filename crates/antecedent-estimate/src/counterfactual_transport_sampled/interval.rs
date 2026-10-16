//! The delta-method Wald interval of a sampled counterfactual transport estimate
//! and the calibration basis it is attested against.
//!
//! The interval is `estimate +/- z * se` clamped to `[0, 1]`. It is a candidate
//! construction: a caller may report it as calibrated only when the basis built
//! here resolves to a currently attesting coverage record. Scope outside the
//! measured design is refused with a stable reason instead of being reported.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::CalibrationBasis;
use antecedent_identify::CounterfactualTransportDerivation;

use super::{LABEL_PLUG_IN, SampledCounterfactualTransportEstimate, UnseenCellRule};

/// Nominal level of the measured construction.
pub const SAMPLED_TRANSPORT_INTERVAL_LEVEL: f64 = 0.95;
/// Conditioning mass below which the delta linearization was not measured.
pub const SAMPLED_TRANSPORT_MIN_CONDITIONING_MASS: f64 = 0.02;
/// Interval method name of the measured construction.
pub const SAMPLED_TRANSPORT_INTERVAL_METHOD: &str = "delta_method_wald";
/// Standard-error kind of the measured construction.
pub const SAMPLED_TRANSPORT_SE_KIND: &str = "delta_method_multinomial_central_difference";
/// Estimator name of the measured construction.
pub const SAMPLED_TRANSPORT_ESTIMATOR: &str = "sampled_counterfactual_transport";

/// A clamped delta Wald interval with the basis that governs its calibration.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledTransportInterval {
    /// Nominal two-sided level.
    pub level: f64,
    /// Plug-in estimate.
    pub point: f64,
    /// Lower end, clamped to `[0, 1]`.
    pub lower: f64,
    /// Upper end, clamped to `[0, 1]`.
    pub upper: f64,
    /// `delta_wald_clamped`.
    pub method: &'static str,
    /// Construction and scope facts a coverage record is matched against.
    pub basis: CalibrationBasis,
}

/// Wald interval and calibration basis of a sampled estimate, or the stable
/// reason it is outside the measured construction.
///
/// # Errors
///
/// A level other than the measured one, no point or standard error, an
/// envelope or smoothed estimate, an unseen-cell rule other than refusal,
/// conditioning mass below the measured floor, or unequal table totals.
pub fn sampled_transport_interval(
    derivation: &CounterfactualTransportDerivation,
    estimate: &SampledCounterfactualTransportEstimate,
    level: f64,
) -> Result<SampledTransportInterval, &'static str> {
    if level.to_bits() != SAMPLED_TRANSPORT_INTERVAL_LEVEL.to_bits() {
        return Err("level_not_measured");
    }
    let Some(point) = &estimate.point else {
        return Err("no_point_estimate");
    };
    let Some(se) = &estimate.inference.standard_error else {
        return Err(estimate
            .inference
            .standard_error_withheld_reason
            .unwrap_or("no_standard_error"));
    };
    if estimate.envelope.is_some()
        || point.smoothed
        || estimate.labels.point_estimate != LABEL_PLUG_IN
    {
        return Err("smoothed_or_envelope_not_measured");
    }
    if estimate.unseen_rule != UnseenCellRule::Refuse
        || estimate.tables.iter().any(|t| t.smoothed || t.unseen_cells > 0)
    {
        return Err("unseen_cell_rule_not_measured");
    }
    if point.conditioning_mass.is_some_and(|m| m < SAMPLED_TRANSPORT_MIN_CONDITIONING_MASS) {
        return Err("conditioning_mass_below_measured");
    }
    let low = estimate.tables.iter().map(|t| t.total).min();
    let high = estimate.tables.iter().map(|t| t.total).max();
    let (Some(low), Some(high)) = (low, high) else {
        return Err("no_tables");
    };
    if low != high {
        return Err("unequal_table_totals_not_measured");
    }
    let z = antecedent_stats::special::normal_ppf(0.5 + level / 2.0);
    let functional = derivation.query.canonical_text();
    let basis = CalibrationBasis::new(
        [
            "CounterfactualTransport",
            "Admg",
            "Fixed",
            "Discrete",
            "Frequentist",
            SAMPLED_TRANSPORT_ESTIMATOR,
            SAMPLED_TRANSPORT_INTERVAL_METHOD,
            SAMPLED_TRANSPORT_SE_KIND,
            "independent_units",
            "",
            functional.as_str(),
        ]
        .map(Arc::from),
        level,
        Arc::from("point"),
        low,
        None,
        None,
        0.0,
    );
    Ok(SampledTransportInterval {
        level,
        point: point.probability,
        lower: (point.probability - z * se.value).max(0.0),
        upper: (point.probability + z * se.value).min(1.0),
        method: "delta_wald_clamped",
        basis,
    })
}
