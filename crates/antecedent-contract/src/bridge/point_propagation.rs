//! Propagated uncertainty of a bridge of one point value.
//!
//! A bridge whose source is a single number (a point claim and the standard error
//! declared with it) maps that number to the plug-in value `g(x)`. When the mapping
//! carries uncertainty of its own, the plug-in value alone hides it: the variance of
//! the bridged quantity is the source's variance carried through the map, plus the
//! mapping's variance, plus what the declared dependence between them adds. This module
//! forms that variance, by the first-order delta method, for each mapping:
//!
//! * a declared affine map `a + b X` takes the delta method of
//!   [`BridgeContract::propagate_affine_uncertainty`] at the source value and the
//!   squared standard error: `Var(a) + x² Var(b) + b² V + 2 x b c + 2 x Cov(a, b) +
//!   Var(b) V + c²` for the source variance `V` and the declared covariance `c` of the
//!   slope estimator and the source estimator, and the mean `a + b x + c`;
//! * a trial-level line takes the coefficient moments of its least-squares fit,
//!   `Var(b) = s² / Sxx`, `Var(a) = s² (1/k + x̄² / Sxx)` and `Cov(a, b) = -s² x̄ /
//!   Sxx`, through the same formula, where `s²` is the residual variance on `k - 2`
//!   degrees of freedom. This is the variance of the fitted mean line, not of a new
//!   trial's effect: the residual of a new trial is a predictive quantity and enters
//!   only the prediction interval;
//! * a binary-misclassification correction takes the delta method of its rates:
//!   `(V + p² Var(Se) + (1 - p)² Var(Sp)) / J²` for the inversion and
//!   `J² V + x² Var(Se) + (1 - x)² Var(Sp)` for the forward map, `J = Se + Sp - 1`;
//! * a nonlinear map `g` of the bridged quantity takes the second-moment expectation
//!   [`nonlinear_expectation`] of the propagated mean and variance, so the bias
//!   `E[g(X)] - g(E[X])` is reported rather than dropped.
//!
//! The dependence the formula needs is never assumed. A map with declared uncertainty
//! and no declared dependence refuses, and so does a dependence with no uncertainty to
//! depend on. A trial-level line and a correction take their uncertainty from their own
//! evidence and so declare none a second time. The result is an uncalibrated spread: the
//! variance is a first-order propagation that no coverage record measured, and an
//! interval of the bridged value remains the separate, scoped request of the layer
//! that holds the coverage records.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::identity::CanonicalWriter;
use antecedent_core::measurement_model::{MeasurementModel, MeasurementParameters};

use super::super::mapping::{
    BridgeMapping, MeasurementDirection, TrialLevelValidation, TrialStatistics, binary_rates,
    fit_with_statistics,
};
use super::super::refusal::{
    BridgeRefusal, dependence_not_a_covariance, malformed_mapping, mapping_uncertainty_invalid,
    mapping_uncertainty_missing, nonlinear_mean_not_expectation, propagation_overflow,
    source_dependence_invalid, source_dependence_missing, source_variance_invalid,
};
use super::BridgeContract;
use super::propagation::{
    AnalyticLaw, MappingUncertainty, NonlinearMap, SourceMappingDependence, nonlinear_expectation,
    propagate_affine,
};

/// How the propagated variance is read: an uncalibrated spread, never an interval.
pub const POINT_PROPAGATION_LABEL: &str = crate::joint_law_rule::UNCALIBRATED_SPREAD_LABEL;

/// The calibration state of the propagated variance: no coverage record measured it.
pub const POINT_PROPAGATION_CALIBRATION: &str = "unmeasured";

/// `g(mean)` and `E[g(X)]` of a nonlinear map of the propagated law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NonlinearPropagation {
    /// The map.
    pub map: NonlinearMap,
    /// `g` of the propagated mean.
    pub map_of_mean: f64,
    /// `E[g(X)]` under the propagated mean and variance: the bias-corrected mean.
    pub expectation: f64,
}

/// The propagated uncertainty of the bridged value of one point source.
#[derive(Clone, Debug, PartialEq)]
pub struct PointPropagation {
    /// The construction: `affine_delta_method`, `trial_level_mean_line_delta_method` or
    /// `correction_delta_method`.
    pub construction: &'static str,
    /// The source value that was bridged.
    pub source_value: f64,
    /// The source standard error the variance was built from.
    pub source_standard_error: f64,
    /// The plug-in value, `g(x)`: the value of the bridged claim.
    pub plug_in_value: f64,
    /// The propagated mean: the plug-in value plus the declared slope-source covariance
    /// (zero for a correction, whose first-order mean is the plug-in value).
    pub mean: f64,
    /// The propagated variance.
    pub variance: f64,
    /// The mapping uncertainty a declared affine map carried; `None` for a trial-level
    /// line or a correction, whose uncertainty is their own evidence.
    pub mapping_uncertainty: Option<MappingUncertainty>,
    /// The declared dependence the variance was built from.
    pub dependence: SourceMappingDependence,
    /// The nonlinear map of the propagated law, when one was requested.
    pub nonlinear: Option<NonlinearPropagation>,
}

impl PointPropagation {
    /// The propagated mean: the bias-corrected expectation of a nonlinear map when one
    /// was requested, otherwise the mean of the affine propagation.
    #[must_use]
    pub fn propagated_mean(&self) -> f64 {
        self.nonlinear.map_or(self.mean, |nonlinear| nonlinear.expectation)
    }

    /// Standard deviation of the propagated variance.
    #[must_use]
    pub fn standard_deviation(&self) -> f64 {
        self.variance.sqrt()
    }

    /// How the variance is read: [`POINT_PROPAGATION_LABEL`].
    #[must_use]
    pub const fn reading(&self) -> &'static str {
        POINT_PROPAGATION_LABEL
    }

    /// Calibration of the variance: [`POINT_PROPAGATION_CALIBRATION`].
    #[must_use]
    pub const fn calibration(&self) -> &'static str {
        POINT_PROPAGATION_CALIBRATION
    }

    /// Write the propagation, every number as its bit pattern, to an identity stream.
    pub(in crate::bridge) fn write(&self, out: &mut CanonicalWriter) {
        out.text(self.construction)
            .real(self.source_standard_error)
            .real(self.mean)
            .real(self.variance);
        match &self.nonlinear {
            None => {
                out.word(0);
            }
            Some(nonlinear) => {
                out.word(1)
                    .text(nonlinear.map.as_str())
                    .real(nonlinear.map_of_mean)
                    .real(nonlinear.expectation);
            }
        }
    }
}

/// The delta-method variance of `a + b X` for coefficient moments and a source
/// estimate of variance `source_variance` and slope covariance `covariance`.
struct AffineMoments {
    slope: f64,
    intercept_variance: f64,
    slope_variance: f64,
    intercept_slope_covariance: f64,
}

impl AffineMoments {
    fn propagate(
        &self,
        source_value: f64,
        source_variance: f64,
        covariance: f64,
    ) -> Result<(f64, f64), BridgeRefusal> {
        let product = self.slope_variance * source_variance;
        let square = covariance * covariance;
        if !product.is_finite() || !square.is_finite() || square > product {
            return Err(dependence_not_a_covariance(covariance.to_string()));
        }
        let variance = self.intercept_variance
            + source_value * source_value * self.slope_variance
            + self.slope * self.slope * source_variance
            + 2.0 * source_value * self.slope * covariance
            + 2.0 * source_value * self.intercept_slope_covariance
            + self.slope_variance * source_variance
            + covariance * covariance;
        finite_variance(variance)?;
        Ok((covariance, variance))
    }
}

fn finite_variance(variance: f64) -> Result<(), BridgeRefusal> {
    if !variance.is_finite() {
        return Err(propagation_overflow(
            "propagated",
            "the delta method overflowed the finite range",
        ));
    }
    if variance < 0.0 {
        return Err(mapping_uncertainty_invalid(
            "propagated",
            "the delta method did not produce a non-negative variance",
        ));
    }
    Ok(())
}

fn declared_dependence(
    contract: &BridgeContract,
) -> Result<SourceMappingDependence, BridgeRefusal> {
    let Some(dependence) = contract.source_dependence.clone() else {
        return Err(source_dependence_missing());
    };
    dependence.validate()?;
    Ok(dependence)
}

fn uncertainty_declared_once(contract: &BridgeContract) -> Result<(), BridgeRefusal> {
    if contract.mapping_uncertainty.is_some() {
        return Err(mapping_uncertainty_invalid(
            "mapping_uncertainty",
            "this mapping takes its uncertainty from its own evidence; declaring the mapping's uncertainty again would count it twice",
        ));
    }
    Ok(())
}

fn affine_propagation(
    contract: &BridgeContract,
    source_value: f64,
    source_variance: f64,
) -> Result<PointPropagation, BridgeRefusal> {
    let propagated = propagate_affine(contract, source_value, source_variance)?;
    Ok(PointPropagation {
        construction: "affine_delta_method",
        source_value,
        source_standard_error: source_variance.sqrt(),
        plug_in_value: propagated.plug_in_mean,
        mean: propagated.mean,
        variance: propagated.variance,
        mapping_uncertainty: Some(propagated.mapping_uncertainty),
        dependence: propagated.dependence,
        nonlinear: None,
    })
}

/// Coefficient moments of the least-squares line through the validation trials, from the
/// canonical statistics of the fit so a permutation of the trials gives the same bits.
fn trial_moments(statistics: &TrialStatistics, slope: f64) -> Result<AffineMoments, BridgeRefusal> {
    let TrialStatistics { count, mean_x, sxx, residual_sum_squares, .. } = *statistics;
    if count < 3.0 || sxx.is_nan() || sxx <= 0.0 {
        return Err(malformed_mapping(
            "trials",
            "the validation trials do not determine a line with a residual variance",
        ));
    }
    let residual_variance = residual_sum_squares / (count - 2.0);
    Ok(AffineMoments {
        slope,
        intercept_variance: residual_variance * (1.0 / count + mean_x * mean_x / sxx),
        slope_variance: residual_variance / sxx,
        intercept_slope_covariance: -residual_variance * mean_x / sxx,
    })
}

fn trial_propagation(
    contract: &BridgeContract,
    validation: &TrialLevelValidation,
    source_value: f64,
    plug_in: f64,
    source_variance: f64,
) -> Result<PointPropagation, BridgeRefusal> {
    uncertainty_declared_once(contract)?;
    let dependence = declared_dependence(contract)?;
    let (fit, statistics) = fit_with_statistics(validation)?;
    let moments = trial_moments(&statistics, fit.slope)?;
    let (covariance, variance) =
        moments.propagate(source_value, source_variance, dependence.covariance())?;
    let mean = plug_in + covariance;
    if !mean.is_finite() {
        return Err(propagation_overflow(
            "propagated",
            "the delta method overflowed the finite range",
        ));
    }
    Ok(PointPropagation {
        construction: "trial_level_mean_line_delta_method",
        source_value,
        source_standard_error: source_variance.sqrt(),
        plug_in_value: plug_in,
        mean,
        variance,
        mapping_uncertainty: None,
        dependence,
        nonlinear: None,
    })
}

fn correction_propagation(
    contract: &BridgeContract,
    model: &MeasurementModel,
    direction: MeasurementDirection,
    source_value: f64,
    plug_in: f64,
    source_variance: f64,
) -> Result<PointPropagation, BridgeRefusal> {
    uncertainty_declared_once(contract)?;
    let dependence = declared_dependence(contract)?;
    if !matches!(dependence, SourceMappingDependence::Independent { .. }) {
        return Err(source_dependence_invalid(
            "source_dependence",
            "a measurement correction has no slope estimator; declare the source and the validation sample independent",
        ));
    }
    let (sensitivity, specificity, _) = binary_rates(model)?;
    let MeasurementParameters::Binary { rates, .. } = &model.parameters else {
        return Err(malformed_mapping(
            "parameters",
            "the measurement model's parameters are not those of its route",
        ));
    };
    let [shared] = rates.as_slice() else {
        return Err(malformed_mapping(
            "rates",
            "a non-differential model carries one shared pair of rates",
        ));
    };
    let (var_se, var_sp) = (shared.sensitivity_variance, shared.specificity_variance);
    let youden = sensitivity + specificity - 1.0;
    let variance = match direction {
        MeasurementDirection::MeasuredToLatent => {
            (source_variance
                + plug_in * plug_in * var_se
                + (1.0 - plug_in) * (1.0 - plug_in) * var_sp)
                / (youden * youden)
        }
        MeasurementDirection::LatentToMeasured => {
            youden * youden * source_variance
                + source_value * source_value * var_se
                + (1.0 - source_value) * (1.0 - source_value) * var_sp
        }
    };
    finite_variance(variance)?;
    Ok(PointPropagation {
        construction: "correction_delta_method",
        source_value,
        source_standard_error: source_variance.sqrt(),
        plug_in_value: plug_in,
        mean: plug_in,
        variance,
        mapping_uncertainty: None,
        dependence,
        nonlinear: None,
    })
}

/// Propagate the uncertainty of a point source through the contract's mapping.
///
/// Returns `None` when there is nothing to propagate: no standard error was declared
/// with the source, or the contract declares neither mapping uncertainty nor source
/// dependence (an exact map keeps only the interval of the layer that holds the coverage
/// records). A nonlinear map requested with no standard error is the nonlinear map of a
/// mean and is refused.
///
/// # Errors
/// A standard error that is not finite and non-negative; a declared uncertainty with no
/// declared dependence, or a dependence with no uncertainty; a covariance no joint law
/// has; a trial-level line or correction that declares its uncertainty a second time; a
/// correction that declares a slope covariance; a nonlinear map requested of a bridge
/// with nothing propagated; a nonlinear map whose law the propagated moments do not
/// determine; a non-finite result.
pub(in crate::bridge) fn propagate_point(
    contract: &BridgeContract,
    mapping: &BridgeMapping,
    direction: MeasurementDirection,
    source_value: f64,
    plug_in: f64,
    standard_error: Option<f64>,
    nonlinear: Option<NonlinearMap>,
) -> Result<Option<PointPropagation>, BridgeRefusal> {
    let Some(standard_error) = standard_error else {
        if let Some(map) = nonlinear {
            return Err(nonlinear_mean_not_expectation(map.as_str()));
        }
        return Ok(None);
    };
    if !(standard_error.is_finite() && standard_error >= 0.0) {
        return Err(source_variance_invalid(format!("standard error {standard_error}")));
    }
    let declared = contract.mapping_uncertainty.is_some() || contract.source_dependence.is_some();
    if !declared {
        return match nonlinear {
            None => Ok(None),
            Some(_) => Err(point_propagation_unavailable(contract, true)),
        };
    }
    let source_variance = standard_error * standard_error;
    let mut propagation = match mapping {
        BridgeMapping::Identity | BridgeMapping::Affine { .. } => {
            affine_propagation(contract, source_value, source_variance)?
        }
        BridgeMapping::TrialLevelLinear(validation) => {
            trial_propagation(contract, validation, source_value, plug_in, source_variance)?
        }
        BridgeMapping::MeasurementCorrection(model) => correction_propagation(
            contract,
            model,
            direction,
            source_value,
            plug_in,
            source_variance,
        )?,
    };
    propagation.source_standard_error = standard_error;
    if let Some(map) = nonlinear {
        let moment = nonlinear_expectation(
            map,
            &AnalyticLaw::SecondMoment { mean: propagation.mean, variance: propagation.variance },
        )?;
        propagation.nonlinear = Some(NonlinearPropagation {
            map,
            map_of_mean: moment.map_of_mean,
            expectation: moment.expectation,
        });
    }
    Ok(Some(propagation))
}

/// The refusal that names what a bridged claim lacks to carry a propagated variance: the
/// source standard error, the mapping uncertainty or the dependence between the source and
/// the mapping. A missing dependence is never assumed to be independence.
#[must_use]
pub fn point_propagation_unavailable(
    contract: &BridgeContract,
    standard_error_supplied: bool,
) -> BridgeRefusal {
    if !standard_error_supplied {
        return source_variance_invalid("the source claim declared no standard error".to_owned());
    }
    let affine = matches!(contract.mapping, BridgeMapping::Identity | BridgeMapping::Affine { .. });
    match (&contract.mapping_uncertainty, &contract.source_dependence) {
        (None, _) if affine => mapping_uncertainty_missing(),
        _ => source_dependence_missing(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{ContractIdentities, SemanticDigest};

    use super::super::super::claim::{
        BridgeSource, derive_bridged_claim, derive_bridged_claim_propagating,
    };
    use super::super::super::kind::{AssumptionKind, BridgeKind, QuantityClass, ValueKind};
    use super::super::super::mapping::{BridgeMapping, TrialEffectPair, TrialLevelValidation};
    use super::super::{BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeScope};
    use super::{MappingUncertainty, NonlinearMap, SourceMappingDependence};
    use crate::{
        ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, QuantityRole, ReasoningView,
        ScientificQuantity, SlotAvailability,
    };

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::from_bytes([byte; 32])
    }

    fn quantity(variable: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: variable.into(),
            variable_name: variable.into(),
            role: QuantityRole::Outcome,
            units: "probability".into(),
            population_id: "panel".into(),
            regime_id: "observational".into(),
            horizon: 0,
            functional_id: "mean".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    fn claim(value: f64) -> ClaimEnvelope {
        ClaimEnvelope::new(
            digest(8),
            ContractIdentities::new(
                digest(10),
                digest(11),
                Some(digest(12)),
                Some(digest(13)),
                digest(14),
                digest(15),
                digest(16),
            ),
            ClaimKind::Point,
            Some(value),
            Some(Arc::from("probability")),
            ReasoningView::new(
                SlotAvailability::unavailable("fixture"),
                SlotAvailability::unavailable("fixture"),
                SlotAvailability::unavailable("fixture"),
                SlotAvailability::unavailable("fixture"),
            ),
            ClaimDomains::new(
                DomainStatus::Unknown,
                DomainStatus::Unknown,
                DomainStatus::Evaluated,
            ),
            None,
            [Arc::<str>::from("fixture")],
        )
    }

    fn preference(mapping: BridgeMapping) -> BridgeContract {
        BridgeContract {
            kind: BridgeKind::PreferenceRealizedBehavior,
            source: BridgeEndpoint {
                quantity: quantity("survey:take_up"),
                class: QuantityClass::StatedPreference,
            },
            target: BridgeEndpoint {
                quantity: quantity("admin:take_up"),
                class: QuantityClass::RealizedBehavior,
            },
            source_value: ValueKind::Mean,
            target_value: ValueKind::Mean,
            evidence_scope: BridgeScope {
                population_id: "panel".into(),
                regime_id: "observational".into(),
            },
            mapping,
            assumptions: vec![BridgeAssumption {
                id: "hypothetical-bias".into(),
                kind: AssumptionKind::BehavioralLinkage,
                statement: "stated scales realized".into(),
                evidence: Some("validation-panel-2023".into()),
            }],
            transport: None,
            mapping_uncertainty: Some(MappingUncertainty {
                id: "calibration-fit".into(),
                intercept_variance: 0.01,
                slope_variance: 0.04,
                intercept_slope_covariance: -0.01,
            }),
            source_dependence: Some(SourceMappingDependence::Independent {
                evidence: "disjoint-samples".into(),
            }),
            estimand_bridge: None,
        }
    }

    fn derive(
        contract: &BridgeContract,
        value: f64,
        standard_error: Option<f64>,
        nonlinear: Option<NonlinearMap>,
    ) -> Result<super::super::super::claim::BridgedClaim, super::BridgeRefusal> {
        let source = claim(value);
        derive_bridged_claim_propagating(
            contract,
            &BridgeSource { claim: &source, quantity: &contract.source.quantity },
            standard_error,
            nonlinear,
        )
    }

    fn bits(value: f64) -> u64 {
        value.to_bits()
    }

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() <= 1e-12 * (1.0 + left.abs().max(right.abs()))
    }

    #[test]
    fn an_affine_map_with_declared_uncertainty_carries_the_hand_delta_variance() {
        // a = 0.5, b = 2, x = 0.4, se = 0.1; Va = 0.01, Vb = 0.04, Cov(a, b) = -0.01, c = 0.
        let contract = preference(BridgeMapping::Affine { intercept: 0.5, slope: 2.0 });
        let bridged = derive(&contract, 0.4, Some(0.1), None).unwrap();
        let propagation = bridged.propagation.as_ref().unwrap();
        assert_eq!(bits(bridged.value), bits(0.5 + 2.0 * 0.4));
        assert_eq!(bits(propagation.plug_in_value), bits(bridged.value));
        let (v, x, b) = (0.1_f64 * 0.1, 0.4_f64, 2.0_f64);
        let expected = 0.01 + x * x * 0.04 + b * b * v + 2.0 * x * -0.01 + 0.04 * v;
        assert!(close(propagation.variance, expected), "{} vs {expected}", propagation.variance);
        assert_eq!(bits(propagation.mean), bits(propagation.plug_in_value));
        assert_eq!(propagation.reading(), "uncalibrated_spread");
        assert_eq!(propagation.calibration(), "unmeasured");
        // The claim value stays the plug-in value and the interval stays withheld.
        assert!(bridged.request_interval().is_err());
    }

    #[test]
    fn the_sign_of_the_declared_covariance_changes_the_variance_and_the_mean() {
        let mut contract = preference(BridgeMapping::Affine { intercept: 0.5, slope: 2.0 });
        let variance_with = |covariance: f64, contract: &mut BridgeContract| {
            contract.source_dependence = Some(SourceMappingDependence::SlopeCovariance {
                covariance,
                evidence: "shared-rows".into(),
            });
            let bridged = derive(contract, 0.4, Some(0.1), None).unwrap();
            let propagation = bridged.propagation.unwrap();
            (propagation.variance, propagation.mean, propagation.plug_in_value)
        };
        let (positive, mean_positive, plug_in) = variance_with(0.002, &mut contract);
        let (negative, mean_negative, _) = variance_with(-0.002, &mut contract);
        let (independent, ..) = {
            contract.source_dependence =
                Some(SourceMappingDependence::Independent { evidence: "disjoint".into() });
            let propagation = derive(&contract, 0.4, Some(0.1), None).unwrap().propagation.unwrap();
            (propagation.variance, propagation.mean)
        };
        // 2 x b c = 2 * 0.4 * 2 * (+-0.002) and c^2 = 4e-6 on both sides.
        assert!(close(positive - independent, 2.0 * 0.4 * 2.0 * 0.002 + 0.002 * 0.002));
        assert!(close(negative - independent, -2.0 * 0.4 * 2.0 * 0.002 + 0.002 * 0.002));
        assert!(positive > independent && independent > negative);
        assert!(close(mean_positive - plug_in, 0.002) && close(mean_negative - plug_in, -0.002));
    }

    #[test]
    fn a_trial_level_line_carries_the_variance_of_its_fitted_mean_line() {
        let trials = [(0.1, 0.12), (0.3, 0.28), (0.5, 0.55), (0.7, 0.66), (0.9, 0.95), (0.4, 0.37)]
            .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
                surrogate_effect,
                terminal_effect,
            });
        let mut contract = preference(BridgeMapping::TrialLevelLinear(TrialLevelValidation {
            trials: trials.to_vec(),
            min_trials: 3,
            min_r_squared: 0.0,
        }));
        contract.kind = BridgeKind::SurrogateTerminalOutcome;
        contract.source.class = QuantityClass::SurrogateEndpoint;
        contract.target.class = QuantityClass::TerminalOutcome;
        contract.assumptions[0].kind = AssumptionKind::SurrogateValidation;
        contract.mapping_uncertainty = None;
        let bridged = derive(&contract, 0.5, Some(0.05), None).unwrap();
        let propagation = bridged.propagation.as_ref().unwrap();
        let k = trials.len() as f64;
        let mean_x = trials.iter().map(|t| t.surrogate_effect).sum::<f64>() / k;
        let mean_y = trials.iter().map(|t| t.terminal_effect).sum::<f64>() / k;
        let sxx = trials.iter().map(|t| (t.surrogate_effect - mean_x).powi(2)).sum::<f64>();
        let syy = trials.iter().map(|t| (t.terminal_effect - mean_y).powi(2)).sum::<f64>();
        let sxy = trials
            .iter()
            .map(|t| (t.surrogate_effect - mean_x) * (t.terminal_effect - mean_y))
            .sum::<f64>();
        let slope = sxy / sxx;
        let s2 = (syy - sxy * sxy / sxx) / (k - 2.0);
        let x = 0.5;
        let v = 0.05_f64 * 0.05;
        // Var(a) + x^2 Var(b) + 2 x Cov(a, b) = s^2 (1/k + (x - xbar)^2 / Sxx).
        let line = s2 * (1.0 / k + (x - mean_x).powi(2) / sxx);
        let expected = line + slope * slope * v + (s2 / sxx) * v;
        assert!(close(propagation.variance, expected), "{} vs {expected}", propagation.variance);
        assert_eq!(propagation.construction, "trial_level_mean_line_delta_method");
        assert!(propagation.mapping_uncertainty.is_none());
    }

    #[test]
    fn a_nonlinear_square_reports_the_bias_against_the_second_moment_closed_form() {
        let contract = preference(BridgeMapping::Affine { intercept: 0.5, slope: 2.0 });
        let bridged = derive(&contract, 0.4, Some(0.1), Some(NonlinearMap::Square)).unwrap();
        let propagation = bridged.propagation.as_ref().unwrap();
        let nonlinear = propagation.nonlinear.unwrap();
        let (mean, variance) = (propagation.mean, propagation.variance);
        assert_eq!(bits(nonlinear.map_of_mean), bits(mean * mean));
        assert_eq!(bits(nonlinear.expectation), bits(mean * mean + variance));
        assert!(nonlinear.expectation > nonlinear.map_of_mean);
        assert_eq!(bits(propagation.propagated_mean()), bits(nonlinear.expectation));
        // The claim value is still the plug-in value of the linear map.
        assert_eq!(bits(bridged.value), bits(0.5 + 2.0 * 0.4));
    }

    #[test]
    fn a_nonlinear_map_of_a_mean_alone_is_refused() {
        let contract = preference(BridgeMapping::Affine { intercept: 0.5, slope: 2.0 });
        let error = derive(&contract, 0.4, None, Some(NonlinearMap::Square)).unwrap_err();
        assert_eq!(error.detail, "scientific_bridge.nonlinear_mean_not_expectation");
        let error = derive(&contract, 0.4, Some(0.1), Some(NonlinearMap::Exponential)).unwrap_err();
        assert_eq!(error.detail, "scientific_bridge.nonlinear_law_insufficient");
    }

    #[test]
    fn a_declared_uncertainty_without_a_declared_dependence_is_refused_not_assumed_independent() {
        let mut contract = preference(BridgeMapping::Affine { intercept: 0.5, slope: 2.0 });
        contract.source_dependence = None;
        let error = derive(&contract, 0.4, Some(0.1), None).unwrap_err();
        assert_eq!(error.code, "sampling_dependence_unknown");
        assert_eq!(error.detail, "scientific_bridge.source_dependence_missing");
        assert!(error.remedy.is_some());
        // Without a standard error nothing is propagated and the plug-in derivation stands.
        assert!(derive(&contract, 0.4, None, None).unwrap().propagation.is_none());
    }

    #[test]
    fn the_propagation_binds_the_claim_identity_and_an_unpropagated_claim_keeps_its_own() {
        let contract = preference(BridgeMapping::Affine { intercept: 0.5, slope: 2.0 });
        let source = claim(0.4);
        let plain = derive_bridged_claim(
            &contract,
            &BridgeSource { claim: &source, quantity: &contract.source.quantity },
        )
        .unwrap();
        let narrow = derive(&contract, 0.4, Some(0.05), None).unwrap();
        let wide = derive(&contract, 0.4, Some(0.2), None).unwrap();
        assert_eq!(bits(plain.value), bits(narrow.value));
        assert_ne!(plain.claim.claim_id, narrow.claim.claim_id);
        assert_ne!(narrow.claim.claim_id, wide.claim.claim_id);
        assert!(plain.propagation.is_none());
    }
}
