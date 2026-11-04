//! The delta-method Wald interval of a sampled counterfactual transport estimate
//! and the calibration basis it is attested against.
//!
//! The interval is `estimate +/- z * se` clamped to `[0, 1]`. It is always reported when the
//! point and standard error exist; its calibration status is an annotation resolved from the
//! basis built here. A design the coverage records did not measure (another level, smoothed
//! tables, an unseen-cell rule, low conditioning mass, unequal totals) names itself in the
//! basis, so no record matches it and the annotation is `not_measured` or
//! `outside_measured_scope` rather than the interval being withheld.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::CalibrationBasis;
use antecedent_identify::CounterfactualTransportDerivation;

use super::{SampledCounterfactualTransportEstimate, UnseenCellRule};

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

/// Dependence structure of the evidence the interval was built from. Only independent
/// units were measured; any other design names itself, so no coverage record attests it.
fn design_dependence(estimate: &SampledCounterfactualTransportEstimate) -> String {
    let stratified = estimate.tables.iter().any(|t| t.stratified);
    let datasets: BTreeSet<usize> = estimate.tables.iter().map(|t| t.dataset).collect();
    let shared = datasets.len() < estimate.tables.len();
    let mut name = String::from(match (stratified, shared) {
        (false, false) => "independent_units",
        (true, false) => "stratified_fixed_totals",
        (false, true) => "shared_dataset",
        (true, true) => "stratified_shared_dataset",
    });
    // Not measured designs name themselves so that no coverage record matches them.
    let low_mass = estimate
        .point
        .as_ref()
        .and_then(|p| p.conditioning_mass)
        .is_some_and(|m| m < SAMPLED_TRANSPORT_MIN_CONDITIONING_MASS);
    let unequal_totals = estimate.tables.iter().map(|t| t.total).min()
        != estimate.tables.iter().map(|t| t.total).max();
    let smoothed = estimate.tables.iter().any(|t| t.smoothed || t.unseen_cells > 0)
        || estimate.point.as_ref().is_some_and(|p| p.smoothed)
        || estimate.unseen_rule != UnseenCellRule::Refuse;
    for (flag, tag) in [
        (low_mass, "low_conditioning_mass"),
        (unequal_totals, "unequal_table_totals"),
        (smoothed, "unseen_cells_or_smoothed"),
    ] {
        if flag {
            name.push('+');
            name.push_str(tag);
        }
    }
    name
}

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
/// A level outside `(0, 1)`, no point or standard error (an envelope has no single point), or
/// no tables.
pub fn sampled_transport_interval(
    derivation: &CounterfactualTransportDerivation,
    estimate: &SampledCounterfactualTransportEstimate,
    level: f64,
) -> Result<SampledTransportInterval, &'static str> {
    if !(level > 0.0 && level < 1.0) {
        return Err("invalid_interval_level");
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
    let low = estimate.tables.iter().map(|t| t.total).min();
    let Some(low) = low else {
        return Err("no_tables");
    };
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
            design_dependence(estimate).as_str(),
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
    )
    .with_validation("not_applicable");
    Ok(SampledTransportInterval {
        level,
        point: point.probability,
        lower: (point.probability - z * se.value).max(0.0),
        upper: (point.probability + z * se.value).min(1.0),
        method: "delta_wald_clamped",
        basis,
    })
}
