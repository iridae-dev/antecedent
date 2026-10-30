//! The interval of a bridged claim: the source claim's declared sampling uncertainty
//! carried through the bridge's mapping together with the mapping's own estimation
//! uncertainty.
//!
//! A bridged value is a function of the source estimate and of what the mapping was
//! estimated from. Its interval is built from both, by mapping:
//!
//! * identity and declared affine maps are exact transforms of the source interval
//!   (`value = a + b * source`; the endpoints are mapped and ordered); the map is
//!   declared exact, so only the source's standard error enters;
//! * a trial-level least-squares line gives the prediction interval of the terminal
//!   effect of a new trial: variance
//!   `s^2 (1 + 1/k + (x - xbar)^2 / Sxx) + slope^2 se^2`, Student-t with `k - 2`
//!   degrees of freedom, where `s^2` is the residual variance of the `k` validation
//!   trials (treated as exact points, which is what the contract's mapping does) and
//!   `se` is the source standard error;
//! * a binary-misclassification correction propagates the source standard error and the
//!   declared variances of the sensitivity and specificity by the delta method:
//!   `p = (q + Sp - 1) / J`, `Var(p) = [se_q^2 + p^2 Var(Se) + (1 - p)^2 Var(Sp)] / J^2`
//!   with `J = Se + Sp - 1`; the forward map has
//!   `Var(q) = J^2 se_p^2 + p^2 Var(Se) + (1 - p)^2 Var(Sp)`. The interval is clipped to
//!   `[0, 1]`.
//!
//! What it requires: the source claim's standard error, supplied with it (a declared
//! source claim carries a point value only and nothing here estimates it). Without
//! one the request refuses. What it does not carry: the declared assumptions of the
//! bridge (they stay unresolved obligations), invariance of the mapping across
//! populations beyond a declared premise, and any individual-level reading.
//!
//! The interval is published only where a current attesting coverage record measured
//! this construction at this bridge kind, mapping, direction and (for a trial-level
//! line) number of validation trials; the record is resolved from the shipped coverage
//! table and nothing the caller declares opens the route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ExternalRefusal;
use antecedent_contract::bridge::{
    BridgeMapping, BridgedClaim, MappingKind, MeasurementDirection, TrialEffectPair,
};
use antecedent_core::measurement_model::{MeasurementParameters, ParameterEvidence};
use antecedent_core::reason_code;
use antecedent_io_core::calibration::{
    CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire, calibration_slot,
};
use antecedent_io_core::contract_section::CalibrationStatus;
use antecedent_stats::{normal_ppf, student_t_ppf};

/// Query axis name of every bridged-claim interval coverage record.
pub const BRIDGE_INTERVAL_QUERY: &str = "BridgedClaimInterval";
/// Smallest nominal level a request may ask for.
pub const BRIDGE_INTERVAL_MIN_LEVEL: f64 = 0.5;
/// Largest nominal level a request may ask for.
pub const BRIDGE_INTERVAL_MAX_LEVEL: f64 = 0.9999;
/// Smallest Youden index of a measurement correction the delta method is published for.
pub const BRIDGE_INTERVAL_MIN_YOUDEN: f64 = 0.3;
/// Largest standard error of a declared sensitivity or specificity the delta method is
/// published for.
pub const BRIDGE_INTERVAL_MAX_RATE_SD: f64 = 0.05;
/// Assumptions every bridged interval states.
pub const BRIDGE_INTERVAL_ASSUMPTIONS: &str = "the source estimate is approximately normal \
    with the declared standard error; the mapping's evidence is independent of the source \
    estimate; the trial-level relation is linear with exchangeable residuals and the validation \
    effects are exact points; the declared assumptions of the bridge are unresolved obligations \
    the interval does not discharge";

const BRIDGE_INTERVAL_STAGE: &str = "bind";
const BRIDGE_INTERVAL_DETAIL_SOURCE_UNCERTAINTY: &str =
    "scientific_bridge.interval_source_uncertainty_required";
const BRIDGE_INTERVAL_DETAIL_OUTSIDE_MEASURED: &str =
    "scientific_bridge.interval_scope_outside_measured";

/// The interval of a bridged value.
#[derive(Clone, Debug, PartialEq)]
pub struct BridgedClaimInterval {
    /// Lower end.
    pub lower: f64,
    /// Upper end.
    pub upper: f64,
    /// Nominal two-sided level.
    pub level: f64,
    /// The source value that was bridged.
    pub source_value: f64,
    /// The bridged value.
    pub value: f64,
    /// The source standard error supplied with the claim.
    pub source_standard_error: f64,
    /// Standard error of the bridged value: the source's, propagated, together with the
    /// mapping's own.
    pub standard_error: f64,
    /// Degrees of freedom of the Student-t reference; `None` for the normal reference.
    pub degrees_of_freedom: Option<f64>,
    /// The mapping.
    pub mapping: MappingKind,
    /// Construction label.
    pub construction: &'static str,
    /// What licenses the interval: `coverage_record` (a current attesting record measured
    /// the construction) or `exact_transform_of_declared_source_interval` (an identity or
    /// declared affine map of the source interval: the bridge adds no uncertainty and
    /// no estimation, so the interval's level is the declared source interval's own).
    pub licensed_by: &'static str,
    /// Governing coverage record (empty on an unlicensed construction or an exact transform).
    pub coverage_record: String,
    /// Commit the governing record was measured at (empty on an unlicensed construction or an
    /// exact transform).
    pub calibration_sha: String,
    /// Stated assumptions, [`BRIDGE_INTERVAL_ASSUMPTIONS`].
    pub assumptions: &'static str,
}

/// An interval built from the bridged claim with the calibration basis a coverage record
/// would have to match. Unlicensed: `coverage_record` is empty, and the value is not a
/// published claim; [`bridged_claim_interval`] is the only route that publishes.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstructedBridgeInterval {
    /// The interval, with no governing record.
    pub interval: BridgedClaimInterval,
    /// The key and scope a coverage record must equal and cover.
    pub basis: CalibrationBasisWire,
}

fn bridge_interval_not_licensed(detail: &str, remedy: &'static str) -> ExternalRefusal {
    ExternalRefusal::new(reason_code!("cell_not_licensed"), BRIDGE_INTERVAL_STAGE, detail)
        .with_remedy(remedy)
}

fn bridge_interval_scope_refusal(message: String) -> ExternalRefusal {
    bridge_interval_not_licensed(
        BRIDGE_INTERVAL_DETAIL_OUTSIDE_MEASURED,
        "ask for the interval inside a measured design: a bridge kind and mapping with a \
         current attesting coverage record, a validated trial count, rates with a Youden index \
         and standard errors inside the measured range, and level 0.95",
    )
    .with_offending(message)
}

fn bridge_interval_mapping_label(kind: MappingKind) -> &'static str {
    match kind {
        MappingKind::Identity => "identity",
        MappingKind::Affine => "affine",
        MappingKind::TrialLevelLinear => "trial_level_linear",
        MappingKind::MeasurementCorrection => "measurement_correction",
        _ => "unsupported_mapping",
    }
}

fn bridge_interval_trial_moments(trials: &[TrialEffectPair]) -> (f64, f64, f64, f64, f64) {
    let k = trials.len() as f64;
    let mean_x = trials.iter().map(|t| t.surrogate_effect).sum::<f64>() / k;
    let mean_y = trials.iter().map(|t| t.terminal_effect).sum::<f64>() / k;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for t in trials {
        let dx = t.surrogate_effect - mean_x;
        let dy = t.terminal_effect - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    (mean_x, sxx, sxy, syy, k)
}

/// Construct the interval of `bridged` from the source claim's standard error, without
/// resolving a coverage record. A measurement scores this; a caller publishes through
/// [`bridged_claim_interval`].
///
/// # Errors
/// `cell_not_licensed` with `scientific_bridge.interval_source_uncertainty_required` when
/// the source standard error is absent; `cell_not_licensed` with
/// `scientific_bridge.interval_scope_outside_measured` for a level outside
/// `[0.5, 0.9999]`, a negative or non-finite standard error, a Youden index or declared
/// rate standard error outside the measured range, or a degenerate trial-level fit.
#[allow(
    clippy::too_many_lines,
    clippy::result_large_err,
    reason = "one construction per mapping kind; the refusal is the shared external refusal type"
)]
pub fn construct_bridged_claim_interval(
    bridged: &BridgedClaim,
    source_standard_error: Option<f64>,
    level: f64,
) -> Result<ConstructedBridgeInterval, ExternalRefusal> {
    let Some(se) = source_standard_error else {
        return Err(bridge_interval_not_licensed(
            BRIDGE_INTERVAL_DETAIL_SOURCE_UNCERTAINTY,
            "supply the source claim's standard error with its own provenance: a bridged \
             interval carries the source estimate's sampling uncertainty, which the declared \
             point value does not state",
        ));
    };
    if !(se.is_finite() && se >= 0.0) {
        return Err(bridge_interval_scope_refusal(format!(
            "source standard error {se} is not a finite non-negative number"
        )));
    }
    if !(level.is_finite()
        && (BRIDGE_INTERVAL_MIN_LEVEL..=BRIDGE_INTERVAL_MAX_LEVEL).contains(&level))
    {
        return Err(bridge_interval_scope_refusal(format!(
            "level {level} is outside [{BRIDGE_INTERVAL_MIN_LEVEL}, {BRIDGE_INTERVAL_MAX_LEVEL}]"
        )));
    }
    let licensing = bridged.contract.check_licensing()?;
    let tail = 1.0 - (1.0 - level) / 2.0;
    let x0 = bridged.source_value;
    let value = bridged.value;
    let z = normal_ppf(tail);
    let validation_units;
    let (lower, upper, standard_error, degrees_of_freedom, construction, direction, row_count) =
        match &bridged.contract.mapping {
            BridgeMapping::Identity => {
                (value - z * se, value + z * se, se, None, "normal_exact_transform", "none", 1_u64)
            }
            BridgeMapping::Affine { slope, .. } => {
                let half = z * slope.abs() * se;
                (
                    value - half,
                    value + half,
                    slope.abs() * se,
                    None,
                    "normal_exact_transform",
                    "none",
                    1,
                )
            }
            BridgeMapping::TrialLevelLinear(validation) => {
                let (mean_x, sxx, sxy, syy, k) = bridge_interval_trial_moments(&validation.trials);
                if k < 3.0 || sxx <= 0.0 {
                    return Err(bridge_interval_scope_refusal(
                        "the validation trials do not determine a line".to_owned(),
                    ));
                }
                let slope = sxy / sxx;
                let rss = (syy - sxy * sxy / sxx).max(0.0);
                let df = k - 2.0;
                let residual_variance = rss / df;
                let leverage = 1.0 / k + (x0 - mean_x).powi(2) / sxx;
                let variance = residual_variance * (1.0 + leverage) + slope * slope * se * se;
                let sd = variance.sqrt();
                if !(sd.is_finite() && sd > 0.0) {
                    return Err(bridge_interval_scope_refusal(
                        "the trial-level residual variance is zero and the source has no \
                         standard error; no prediction interval is built"
                            .to_owned(),
                    ));
                }
                let t = student_t_ppf(tail, df);
                (
                    value - t * sd,
                    value + t * sd,
                    sd,
                    Some(df),
                    "student_t_prediction_interval",
                    "none",
                    validation.trials.len() as u64,
                )
            }
            BridgeMapping::MeasurementCorrection(model) => {
                let MeasurementParameters::Binary { rates, .. } = &model.parameters else {
                    return Err(bridge_interval_scope_refusal(
                        "the measurement model is not a binary one".to_owned(),
                    ));
                };
                let [shared] = rates.as_slice() else {
                    return Err(bridge_interval_scope_refusal(
                        "the correction needs one shared pair of rates".to_owned(),
                    ));
                };
                let ParameterEvidence::ValidationSample { units, .. } = &model.evidence else {
                    return Err(bridge_interval_scope_refusal(
                        "the rates' sampling variances are propagated only for rates estimated \
                         from a stated validation sample"
                            .to_owned(),
                    ));
                };
                validation_units = *units;
                let j = shared.sensitivity + shared.specificity - 1.0;
                let (var_se, var_sp) = (shared.sensitivity_variance, shared.specificity_variance);
                if j < BRIDGE_INTERVAL_MIN_YOUDEN
                    || var_se.sqrt() > BRIDGE_INTERVAL_MAX_RATE_SD
                    || var_sp.sqrt() > BRIDGE_INTERVAL_MAX_RATE_SD
                {
                    return Err(bridge_interval_scope_refusal(format!(
                        "Youden index {j} or a declared rate standard error lies outside the \
                         measured range (Youden at least {BRIDGE_INTERVAL_MIN_YOUDEN}, rate \
                         standard errors at most {BRIDGE_INTERVAL_MAX_RATE_SD})"
                    )));
                }
                let (variance, label) = match licensing.direction {
                    MeasurementDirection::MeasuredToLatent => (
                        (se * se + value * value * var_se + (1.0 - value) * (1.0 - value) * var_sp)
                            / (j * j),
                        "measured_to_latent",
                    ),
                    MeasurementDirection::LatentToMeasured => (
                        j * j * se * se + x0 * x0 * var_se + (1.0 - x0) * (1.0 - x0) * var_sp,
                        "latent_to_measured",
                    ),
                };
                let sd = variance.sqrt();
                (
                    (value - z * sd).max(0.0),
                    (value + z * sd).min(1.0),
                    sd,
                    None,
                    "normal_delta_method_clipped",
                    label,
                    validation_units,
                )
            }
            _ => {
                return Err(bridge_interval_scope_refusal(
                    "the mapping has no interval construction".to_owned(),
                ));
            }
        };
    let (lo, hi) = if lower <= upper { (lower, upper) } else { (upper, lower) };
    let mapping = bridged.mapping;
    let basis = CalibrationBasisWire {
        key: CalibrationKeyWire {
            validation: "not_applicable".into(),
            query: BRIDGE_INTERVAL_QUERY.to_owned(),
            graph_class: "NotApplicable".to_owned(),
            structure: "fixed".to_owned(),
            modality: "claim_value".to_owned(),
            inference: "Frequentist".to_owned(),
            estimator: format!("scientific_bridge_{}", bridge_interval_mapping_label(mapping)),
            interval_method: construction.to_owned(),
            se_kind: "declared_source_standard_error".to_owned(),
            dependence: "source_independent_of_mapping_evidence".to_owned(),
            posterior: String::new(),
            functional: format!("bridge={};direction={direction}", bridged.kind().as_str()),
            level,
            identification: "point".to_owned(),
        },
        scope: CalibrationScopeWire {
            row_count,
            replicates_ok: None,
            posterior_draws: None,
            unidentified_mass: 0.0,
        },
    };
    Ok(ConstructedBridgeInterval {
        interval: BridgedClaimInterval {
            lower: lo,
            upper: hi,
            level,
            source_value: x0,
            value,
            source_standard_error: se,
            standard_error,
            degrees_of_freedom,
            mapping,
            construction,
            licensed_by: "coverage_record",
            coverage_record: String::new(),
            calibration_sha: String::new(),
            assumptions: BRIDGE_INTERVAL_ASSUMPTIONS,
        },
        basis,
    })
}

/// Build the interval of `bridged` and publish it.
///
/// The record is resolved from the shipped coverage table
/// ([`antecedent_io_core::calibration::calibration_slot`]): the interval is returned only
/// when a current attesting non-boundary record measured exactly this construction.
///
/// # Errors
/// As [`construct_bridged_claim_interval`], and `cell_not_licensed` with
/// `scientific_bridge.interval_scope_outside_measured` when no current attesting coverage
/// record measured this bridge kind, mapping, direction, level and validated trial count.
#[allow(clippy::result_large_err, reason = "the refusal is the shared external refusal type")]
pub fn bridged_claim_interval(
    bridged: &BridgedClaim,
    source_standard_error: Option<f64>,
    level: f64,
) -> Result<BridgedClaimInterval, ExternalRefusal> {
    let ConstructedBridgeInterval { mut interval, basis } =
        construct_bridged_claim_interval(bridged, source_standard_error, level)?;
    if matches!(interval.mapping, MappingKind::Identity | MappingKind::Affine) {
        // An exact monotone map of the declared source interval estimates nothing: its
        // coverage is the source interval's by construction, so there is nothing to measure.
        interval.licensed_by = "exact_transform_of_declared_source_interval";
        return Ok(interval);
    }
    let slot = calibration_slot(&basis);
    match (&slot.status, &slot.record_id, &slot.calibration_sha) {
        (CalibrationStatus::Calibrated, Some(id), Some(sha)) => {
            interval.coverage_record.clone_from(id);
            interval.calibration_sha.clone_from(sha);
            Ok(interval)
        }
        _ => Err(bridge_interval_scope_refusal(format!(
            "no current attesting coverage record measured the {} interval of the {} bridge \
             ({}) at level {level} ({})",
            bridge_interval_mapping_label(bridged.mapping),
            bridged.kind().as_str(),
            basis.key.functional,
            slot.reason.as_deref().unwrap_or("construction_not_measured")
        ))),
    }
}
