//! Uncertain mappings, source dependence, nonlinear moments and estimand bridges.
//!
//! An affine map of a mean is the mean of the map. A nonlinear map is not:
//! [`nonlinear_expectation`] returns both `g(E[X])` and `E[g(X)]`, and a mean
//! with no further moment is refused. The square uses the second-moment
//! identity `E[X²] = (E[X])² + Var(X)`, which holds for any law with those
//! moments. The exponential uses the normal moment-generating function
//! `E[e^X] = exp(E[X] + Var(X) / 2)`.
//!
//! [`propagate_affine`] is the first-order delta method of a declared affine
//! map `a + b X`. The plug-in `a + b μ` omits the covariance of the slope
//! estimator and the source estimator; the propagated mean adds that
//! covariance, because the expectation of a product is not the product of the
//! expectations. The variance is the variance of `a + β M` for an intercept
//! estimator `a`, a slope estimator `β` and a source estimator `M` with means
//! `a`, `b`, `μ`, taking `M` independent of `a` and writing `c = Cov(β, M)` and
//! `d = Cov(a, β)`:
//! `Var(a) + μ² Var(β) + b² Var(M) + 2 μ b c + 2 μ d + Var(β) Var(M) + c²`.
//! The last two terms are the second-order part of the product; with both
//! covariances declared independent (zero) the result is the exact variance of
//! a product of independent factors plus the intercept variance. The default
//! intercept-slope covariance of zero declares them independent; a regression
//! calibration usually has a negative one (`≈ -x̄ Var(β)`), which this field
//! must carry.
//!
//! An experiment is a particular population, time window, dose, duration and
//! exposure regime. Importing its prior requires an estimand bridge that
//! names each of those on both sides and states the recency and spend
//! differences; the spend difference must agree with the two doses. Mapping uncertainty and source dependence on that prior are
//! kept, and so are the bridge's assumptions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::identity::CanonicalWriter;

use super::super::mapping::BridgeMapping;
use super::super::refusal::{
    BridgeRefusal, dependence_not_a_covariance, estimand_bridge_incomplete,
    estimand_bridge_mismatch, estimand_spend_inconsistent, estimand_units_mismatch,
    experimental_prior_without_estimand_bridge, malformed_mapping, mapping_uncertainty_invalid,
    mapping_uncertainty_missing, nonlinear_law_insufficient, nonlinear_mean_not_expectation,
    propagation_overflow, source_dependence_invalid, source_dependence_missing,
    source_moment_not_finite, source_variance_invalid, uncertain_mapping_not_affine,
};
use super::{BridgeAssumption, BridgeContract, MAX_BRIDGE_TEXT_BYTES, check_text};

/// Section word that opens the identity suffix. Written only when a contract
/// declares mapping uncertainty, source dependence or an estimand bridge, so
/// a contract that declares none keeps the version-1 byte stream.
const EXTENSION_SECTION: u64 = 0xE07;

/// Variance of a declared affine map, apart from the source estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct MappingUncertainty {
    /// Stable identity of this declaration.
    pub id: String,
    /// Variance of the intercept. Zero states that the intercept is exact.
    pub intercept_variance: f64,
    /// Variance of the slope. Zero states that the slope is exact.
    pub slope_variance: f64,
    /// Covariance of the intercept and slope estimators. Zero declares them
    /// independent; its square must be at most the product of the two variances.
    pub intercept_slope_covariance: f64,
}

impl MappingUncertainty {
    pub(super) fn validate(&self) -> Result<(), BridgeRefusal> {
        require_text(
            "mapping_uncertainty.id",
            &self.id,
            "name the mapping uncertainty",
            mapping_uncertainty_invalid,
        )?;
        finite_variance(
            "intercept_variance",
            self.intercept_variance,
            mapping_uncertainty_invalid,
            "state a finite non-negative intercept variance",
        )?;
        finite_variance(
            "slope_variance",
            self.slope_variance,
            mapping_uncertainty_invalid,
            "state a finite non-negative slope variance",
        )?;
        let product = self.intercept_variance * self.slope_variance;
        let square = self.intercept_slope_covariance * self.intercept_slope_covariance;
        if !self.intercept_slope_covariance.is_finite()
            || !product.is_finite()
            || !square.is_finite()
            || square > product
        {
            return Err(mapping_uncertainty_invalid(
                "intercept_slope_covariance",
                "state a finite covariance whose square is at most the product of the intercept and slope variances",
            ));
        }
        Ok(())
    }

    fn write(&self, out: &mut CanonicalWriter) {
        out.text(&self.id)
            .real(self.intercept_variance)
            .real(self.slope_variance)
            .real(self.intercept_slope_covariance + 0.0);
    }
}

/// How the source estimate and the mapping share information.
///
/// Independence is a declaration. A covariance of zero is a different
/// declaration: it states a number, and its evidence is part of the identity.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SourceMappingDependence {
    /// The source estimate and the mapping evidence are declared disjoint.
    Independent {
        /// The study or ledger entry the disjointness rests on.
        evidence: String,
    },
    /// Covariance of the slope estimator and the source estimator.
    SlopeCovariance {
        /// Covariance. Its square must be at most the product of the variances.
        covariance: f64,
        /// The study or ledger entry the covariance rests on.
        evidence: String,
    },
}

impl SourceMappingDependence {
    pub(super) fn validate(&self) -> Result<(), BridgeRefusal> {
        match self {
            Self::Independent { evidence } => require_text(
                "source_dependence.evidence",
                evidence,
                "name the evidence that the source and the mapping are disjoint",
                source_dependence_invalid,
            ),
            Self::SlopeCovariance { covariance, evidence } => {
                if !covariance.is_finite() {
                    return Err(source_dependence_invalid(
                        "covariance",
                        "state a finite covariance of the slope and the source",
                    ));
                }
                require_text(
                    "source_dependence.evidence",
                    evidence,
                    "name the evidence the covariance rests on",
                    source_dependence_invalid,
                )
            }
        }
    }

    pub(super) fn covariance(&self) -> f64 {
        match self {
            Self::Independent { .. } => 0.0,
            Self::SlopeCovariance { covariance, .. } => *covariance,
        }
    }

    fn write(&self, out: &mut CanonicalWriter) {
        match self {
            Self::Independent { evidence } => {
                out.word(1).text(evidence);
            }
            Self::SlopeCovariance { covariance, evidence } => {
                out.word(2).real(*covariance).text(evidence);
            }
        }
    }
}

/// One side of an estimand bridge: the population, time window, dose,
/// duration and exposure regime, each named rather than inferred.
#[derive(Clone, Debug, PartialEq)]
pub struct ExposureSetting {
    /// Population the exposure refers to.
    pub population_id: String,
    /// Start of the time window. A label, not a parsed calendar.
    pub time_start: String,
    /// End of the time window. Equal to the start when the window is a point.
    pub time_end: String,
    /// Dose in `dose_units`.
    pub dose: f64,
    /// Unit identity of `dose`.
    pub dose_units: String,
    /// Duration in `duration_units`. Finite and non-negative.
    pub duration: f64,
    /// Unit identity of `duration`.
    pub duration_units: String,
    /// Exposure regime. `observational` is explicit.
    pub exposure_regime: String,
}

impl ExposureSetting {
    fn validate(&self, prefix: &str) -> Result<(), BridgeRefusal> {
        require_text(
            &format!("{prefix}.population_id"),
            &self.population_id,
            "name the population",
            estimand_bridge_incomplete,
        )?;
        require_text(
            &format!("{prefix}.time_start"),
            &self.time_start,
            "name the start of the time window",
            estimand_bridge_incomplete,
        )?;
        require_text(
            &format!("{prefix}.time_end"),
            &self.time_end,
            "name the end of the time window",
            estimand_bridge_incomplete,
        )?;
        if !self.dose.is_finite() {
            return Err(estimand_bridge_incomplete("dose", "state a finite dose"));
        }
        require_text(
            &format!("{prefix}.dose_units"),
            &self.dose_units,
            "name the dose units",
            estimand_bridge_incomplete,
        )?;
        if !self.duration.is_finite() || self.duration < 0.0 {
            return Err(estimand_bridge_incomplete(
                "duration",
                "state a finite non-negative duration",
            ));
        }
        require_text(
            &format!("{prefix}.duration_units"),
            &self.duration_units,
            "name the duration units",
            estimand_bridge_incomplete,
        )?;
        require_text(
            &format!("{prefix}.exposure_regime"),
            &self.exposure_regime,
            "name the exposure regime",
            estimand_bridge_incomplete,
        )
    }

    fn write(&self, out: &mut CanonicalWriter) {
        out.text(&self.population_id)
            .text(&self.time_start)
            .text(&self.time_end)
            .real(self.dose)
            .text(&self.dose_units)
            .real(self.duration)
            .text(&self.duration_units)
            .text(&self.exposure_regime);
    }
}

/// Translation of an experiment's estimand into the estimand the bridge answers.
///
/// Recency and spend are differences of the target relative to the experiment.
/// Zero is a declaration that they do not differ, not an omitted field.
#[derive(Clone, Debug, PartialEq)]
pub struct EstimandBridge {
    /// Stable identity of this translation.
    pub id: String,
    /// Experiment whose estimand is being translated.
    pub experiment_id: String,
    /// Population, window, dose, duration and regime the experiment estimated.
    pub experiment: ExposureSetting,
    /// Population, window, dose, duration and regime the bridge answers.
    pub target: ExposureSetting,
    /// How much more recent the target is than the experiment, in `recency_units`.
    pub recency_delta: f64,
    /// Unit identity of `recency_delta`.
    pub recency_units: String,
    /// Target spend or exposure minus the experiment's, in `spend_units`. It must
    /// equal the target dose minus the experiment dose, in one shared unit.
    pub spend_delta: f64,
    /// Unit identity of `spend_delta`.
    pub spend_units: String,
}

impl EstimandBridge {
    pub(super) fn validate(&self) -> Result<(), BridgeRefusal> {
        require_text(
            "estimand_bridge.id",
            &self.id,
            "name the estimand bridge",
            estimand_bridge_incomplete,
        )?;
        require_text(
            "estimand_bridge.experiment_id",
            &self.experiment_id,
            "name the experiment",
            estimand_bridge_incomplete,
        )?;
        self.experiment.validate("experiment")?;
        self.target.validate("target")?;
        if !self.recency_delta.is_finite() {
            return Err(estimand_bridge_incomplete(
                "recency_delta",
                "state the recency difference, including zero when the target is not more recent",
            ));
        }
        require_text(
            "estimand_bridge.recency_units",
            &self.recency_units,
            "name the recency units",
            estimand_bridge_incomplete,
        )?;
        if !self.spend_delta.is_finite() {
            return Err(estimand_bridge_incomplete(
                "spend_delta",
                "state the spend or exposure difference, including zero when it is unchanged",
            ));
        }
        require_text(
            "estimand_bridge.spend_units",
            &self.spend_units,
            "name the spend units",
            estimand_bridge_incomplete,
        )
    }

    fn write(&self, out: &mut CanonicalWriter) {
        out.text(&self.id).text(&self.experiment_id);
        self.experiment.write(out);
        self.target.write(out);
        out.real(self.recency_delta)
            .text(&self.recency_units)
            .real(self.spend_delta)
            .text(&self.spend_units);
    }
}

/// A prior taken from an experiment, with the translation that would license it.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentalPrior {
    /// Stable identity of the prior.
    pub prior_id: String,
    /// Experiment the prior was estimated from.
    pub experiment_id: String,
    /// Translation into the target estimand. Absent priors are refused.
    pub estimand_bridge: Option<EstimandBridge>,
    /// Mapping uncertainty carried with the prior, when the map was estimated.
    pub mapping_uncertainty: Option<MappingUncertainty>,
    /// Dependence of the prior's source and its mapping, when declared.
    pub source_dependence: Option<SourceMappingDependence>,
}

/// What an accepted import keeps: the translation, the prior's uncertainty and
/// dependence, and the assumptions the contract already declared.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportedExperimentalPrior {
    /// Prior identity.
    pub prior_id: String,
    /// Experiment identity.
    pub experiment_id: String,
    /// The translation that licensed the import.
    pub estimand_bridge: EstimandBridge,
    /// Mapping uncertainty from the prior, unchanged.
    pub mapping_uncertainty: Option<MappingUncertainty>,
    /// Source dependence from the prior, unchanged.
    pub source_dependence: Option<SourceMappingDependence>,
    /// Contract assumptions in identity order.
    pub assumptions: Vec<BridgeAssumption>,
}

/// A nonlinear map whose expectation has a closed form for a declared law.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NonlinearMap {
    /// `g(x) = x²`.
    Square,
    /// `g(x) = exp(x)`.
    Exponential,
}

impl NonlinearMap {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Square => "square",
            Self::Exponential => "exponential",
        }
    }
}

/// The law a nonlinear expectation is allowed to use.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum AnalyticLaw {
    /// A mean and nothing else. Not enough for a nonlinear expectation.
    MeanOnly {
        /// `E[X]`.
        mean: f64,
    },
    /// Mean and variance of an otherwise unspecified law.
    SecondMoment {
        /// `E[X]`.
        mean: f64,
        /// `Var(X)`, finite and non-negative.
        variance: f64,
    },
    /// A normal law, which determines the exponential moment.
    Normal {
        /// `E[X]`.
        mean: f64,
        /// `Var(X)`, finite and non-negative.
        variance: f64,
    },
}

/// `g(E[X])` and `E[g(X)]` for one map and one law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NonlinearMoment {
    /// The map.
    pub map: NonlinearMap,
    /// `g(E[X])`.
    pub map_of_mean: f64,
    /// `E[g(X)]`.
    pub expectation: f64,
}

/// The map used to carry an interval of support.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SupportMap {
    /// The interval is unchanged.
    Identity,
    /// `intercept + slope * x`, with the ends ordered.
    Affine {
        /// Value at a zero source.
        intercept: f64,
        /// Change per unit of the source.
        slope: f64,
    },
    /// Image of `x²`, including the vertex when the interval contains zero.
    Square,
    /// Image of `exp(x)`.
    Exponential,
}

/// A closed bounded interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValueInterval {
    /// Lower endpoint.
    pub low: f64,
    /// Upper endpoint.
    pub high: f64,
}

/// The image of a support interval together with the assumptions that bind it.
#[derive(Clone, Debug, PartialEq)]
pub struct PreservedSupport {
    /// Assumptions in the same order the contract identity writes them.
    pub assumptions: Vec<BridgeAssumption>,
    /// Lower endpoint of the image.
    pub low: f64,
    /// Upper endpoint of the image.
    pub high: f64,
}

/// Delta-method image of an affine map of an uncertain source.
#[derive(Clone, Debug, PartialEq)]
pub struct PropagatedUncertainty {
    /// `a + b μ`, the map of the source mean.
    pub plug_in_mean: f64,
    /// `a + b μ + Cov(b, μ)`, the expectation of the product.
    pub mean: f64,
    /// First-order variance of the affine map.
    pub variance: f64,
    /// The mapping uncertainty the variance was built from.
    pub mapping_uncertainty: MappingUncertainty,
    /// The dependence the variance was built from.
    pub dependence: SourceMappingDependence,
    /// Contract assumptions in identity order.
    pub assumptions: Vec<BridgeAssumption>,
}

fn require_text(
    field: &str,
    text: &str,
    remedy: &'static str,
    refuse: fn(&str, &'static str) -> BridgeRefusal,
) -> Result<(), BridgeRefusal> {
    if text.trim().is_empty() || text.len() > MAX_BRIDGE_TEXT_BYTES {
        return Err(refuse(field, remedy));
    }
    Ok(())
}

fn finite_variance(
    field: &str,
    value: f64,
    refuse: fn(&str, &'static str) -> BridgeRefusal,
    remedy: &'static str,
) -> Result<(), BridgeRefusal> {
    if value.is_finite() && value >= 0.0 { Ok(()) } else { Err(refuse(field, remedy)) }
}

fn moments(law: &AnalyticLaw) -> Result<(f64, f64), BridgeRefusal> {
    let (mean, variance) = match *law {
        AnalyticLaw::MeanOnly { .. } => {
            return Err(nonlinear_mean_not_expectation("mean"));
        }
        AnalyticLaw::SecondMoment { mean, variance } | AnalyticLaw::Normal { mean, variance } => {
            (mean, variance)
        }
    };
    if !mean.is_finite() {
        return Err(source_moment_not_finite("mean", "state a finite mean"));
    }
    if !(variance.is_finite() && variance >= 0.0) {
        return Err(source_variance_invalid(variance.to_string()));
    }
    Ok((mean, variance))
}

/// `g(E[X])` and `E[g(X)]` for a square or a normal exponential.
///
/// A [`AnalyticLaw::MeanOnly`] source is refused for every nonlinear map.
/// A square accepts any second moment. An exponential accepts only a normal law.
///
/// # Errors
/// The mean is not finite; the variance is not a finite non-negative number;
/// the source is only a mean; an exponential is given a law that is not normal;
/// the expectation overflows.
pub fn nonlinear_expectation(
    map: NonlinearMap,
    law: &AnalyticLaw,
) -> Result<NonlinearMoment, BridgeRefusal> {
    if matches!(law, AnalyticLaw::MeanOnly { .. }) {
        return Err(nonlinear_mean_not_expectation(map.as_str()));
    }
    let (mean, variance) = moments(law)?;
    let (map_of_mean, expectation) = match map {
        NonlinearMap::Square => (mean * mean, mean * mean + variance),
        NonlinearMap::Exponential => {
            if !matches!(law, AnalyticLaw::Normal { .. }) {
                return Err(nonlinear_law_insufficient("second_moment"));
            }
            let map_of_mean = mean.exp();
            let expectation = (mean + 0.5 * variance).exp();
            (map_of_mean, expectation)
        }
    };
    if !map_of_mean.is_finite() || !expectation.is_finite() {
        return Err(propagation_overflow(
            map.as_str(),
            "the nonlinear expectation overflowed; the declared moments are outside the map",
        ));
    }
    Ok(NonlinearMoment { map, map_of_mean, expectation })
}

/// An image endpoint pair, refused as an overflow when a valid source interval
/// maps outside the finite range.
fn image_interval(low: f64, high: f64) -> Result<ValueInterval, BridgeRefusal> {
    if low.is_finite() && high.is_finite() {
        ordered_interval(low, high)
    } else {
        Err(propagation_overflow(
            "support",
            "the image of the support interval overflowed; state a narrower interval or a smaller map",
        ))
    }
}

fn ordered_interval(low: f64, high: f64) -> Result<ValueInterval, BridgeRefusal> {
    if low.is_finite() && high.is_finite() && low <= high {
        Ok(ValueInterval { low, high })
    } else {
        Err(malformed_mapping(
            "support",
            "state a finite support interval with the lower endpoint first",
        ))
    }
}

/// Image of `source` under `map`.
///
/// The square's image includes zero when the source interval contains zero,
/// because the vertex is inside the support. Mapping the endpoints alone
/// drops it.
///
/// # Errors
/// An endpoint is not finite, the interval is reversed, or an affine
/// coefficient is not finite.
pub fn support_image(
    map: SupportMap,
    source: ValueInterval,
) -> Result<ValueInterval, BridgeRefusal> {
    let source = ordered_interval(source.low, source.high)?;
    match map {
        SupportMap::Identity => Ok(source),
        SupportMap::Affine { intercept, slope } => {
            if !intercept.is_finite() || !slope.is_finite() {
                return Err(mapping_uncertainty_invalid(
                    "affine",
                    "state a finite intercept and slope",
                ));
            }
            let left = intercept + slope * source.low;
            let right = intercept + slope * source.high;
            image_interval(left.min(right), left.max(right))
        }
        SupportMap::Square => {
            let left = source.low * source.low;
            let right = source.high * source.high;
            let low = if source.low <= 0.0 && source.high >= 0.0 { 0.0 } else { left.min(right) };
            image_interval(low, left.max(right))
        }
        SupportMap::Exponential => image_interval(source.low.exp(), source.high.exp()),
    }
}

pub(super) fn sorted_assumptions(assumptions: &[BridgeAssumption]) -> Vec<BridgeAssumption> {
    let mut cloned = assumptions.to_vec();
    cloned.sort_by(|left, right| left.id.cmp(&right.id));
    cloned
}

/// Bind `map`'s image of `source` to the contract's assumptions, in identity order.
///
/// # Errors
/// The contract is not licensed ([`BridgeContract::check_licensing`]), or the
/// source interval or the map is not one [`support_image`] accepts.
pub fn preserved_support(
    contract: &BridgeContract,
    map: SupportMap,
    source: ValueInterval,
) -> Result<PreservedSupport, BridgeRefusal> {
    contract.check_licensing()?;
    let image = support_image(map, source)?;
    Ok(PreservedSupport {
        assumptions: sorted_assumptions(&contract.assumptions),
        low: image.low,
        high: image.high,
    })
}

fn affine_coefficients(mapping: &BridgeMapping) -> Result<(f64, f64), BridgeRefusal> {
    match mapping {
        BridgeMapping::Identity => Ok((0.0, 1.0)),
        BridgeMapping::Affine { intercept, slope } => Ok((*intercept, *slope)),
        BridgeMapping::TrialLevelLinear(_) | BridgeMapping::MeasurementCorrection(_) => {
            Err(uncertain_mapping_not_affine(mapping.kind().as_str()))
        }
    }
}

/// Delta-method mean and variance of the contract's affine map.
///
/// The result keeps the declared mapping uncertainty, the declared dependence
/// and the contract's assumptions. A missing dependence is refused rather than
/// treated as zero.
///
/// # Errors
/// The contract is not licensed; the mapping is not affine or identity;
/// mapping uncertainty or source dependence is missing or not well formed; the
/// source moments are not finite and non-negative; the covariance is not
/// feasible; the propagated mean or variance is not finite.
pub(super) fn propagate_affine(
    contract: &BridgeContract,
    source_mean: f64,
    source_variance: f64,
) -> Result<PropagatedUncertainty, BridgeRefusal> {
    contract.check_licensing()?;
    let (intercept, slope) = affine_coefficients(&contract.mapping)?;
    let Some(uncertainty) = contract.mapping_uncertainty.clone() else {
        return Err(mapping_uncertainty_missing());
    };
    uncertainty.validate()?;
    let Some(dependence) = contract.source_dependence.clone() else {
        return Err(source_dependence_missing());
    };
    dependence.validate()?;
    if !source_mean.is_finite() {
        return Err(source_moment_not_finite("source_mean", "state a finite source mean"));
    }
    if !(source_variance.is_finite() && source_variance >= 0.0) {
        return Err(source_variance_invalid(source_variance.to_string()));
    }
    let covariance = dependence.covariance();
    let product = uncertainty.slope_variance * source_variance;
    let square = covariance * covariance;
    if !product.is_finite() || !square.is_finite() || square > product {
        return Err(dependence_not_a_covariance(covariance.to_string()));
    }
    let plug_in_mean = intercept + slope * source_mean;
    let mean = plug_in_mean + covariance;
    let variance = uncertainty.intercept_variance
        + source_mean * source_mean * uncertainty.slope_variance
        + slope * slope * source_variance
        + 2.0 * source_mean * slope * covariance
        + 2.0 * source_mean * uncertainty.intercept_slope_covariance
        + uncertainty.slope_variance * source_variance
        + covariance * covariance;
    if !plug_in_mean.is_finite() || !mean.is_finite() || !variance.is_finite() {
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
    Ok(PropagatedUncertainty {
        plug_in_mean,
        mean,
        variance,
        mapping_uncertainty: uncertainty,
        dependence,
        assumptions: sorted_assumptions(&contract.assumptions),
    })
}

// ---------------------------------------------------------------------------
// Joint propagation of a nonlinear function of the source and the mapping.
// ---------------------------------------------------------------------------

/// Most coordinates a joint law may carry.
const MAX_JOINT_DIMENSION: usize = 32;
/// Fewest and most simulated draws a propagation accepts.
const MIN_SIMULATION_DRAWS: usize = 64;
const MAX_SIMULATION_DRAWS: usize = 4_000_000;
/// Relative step of the finite differences behind the second-order terms, in
/// units of each coordinate's standard deviation.
const DIFFERENCE_STEP: f64 = 1e-2;
/// Variances below this fraction of the largest variance are a degenerate
/// direction in the factorization; negative pivots and off-diagonal residuals
/// beyond [`PSD_SLACK`] of it are refused.
const DEGENERATE_FRACTION: f64 = 1e-13;
const PSD_SLACK: f64 = 1e-9;

/// The joint moment law of every uncertain input of a mapped quantity: the
/// source estimate and each mapping coefficient, with their full covariance.
///
/// The covariance carries the dependence between the source and the mapping
/// that [`SourceMappingDependence`] declares, together with the mapping's own
/// intercept-slope covariance; a nonlinear function of those inputs is then
/// propagated against the joint law, not against a marginal of each.
#[derive(Clone, Debug, PartialEq)]
pub struct JointLaw {
    names: Vec<String>,
    mean: Vec<f64>,
    covariance: Vec<f64>,
}

impl JointLaw {
    /// A joint law from coordinate names, means and a row-major covariance.
    ///
    /// # Errors
    /// The dimension is zero or above the limit; the lengths disagree; a mean
    /// or covariance entry is not finite; the covariance is not symmetric or
    /// not positive semi-definite.
    pub fn new(
        names: Vec<String>,
        mean: Vec<f64>,
        covariance: Vec<f64>,
    ) -> Result<Self, BridgeRefusal> {
        let n = mean.len();
        if n == 0 || n > MAX_JOINT_DIMENSION || names.len() != n || covariance.len() != n * n {
            return Err(malformed_mapping(
                "joint_law",
                "state one name and one mean per coordinate and an n by n covariance, with at most 32 coordinates",
            ));
        }
        if names.iter().any(|name| name.trim().is_empty() || name.len() > MAX_BRIDGE_TEXT_BYTES) {
            return Err(malformed_mapping("joint_law.names", "name every coordinate"));
        }
        if mean.iter().any(|value| !value.is_finite()) {
            return Err(source_moment_not_finite("joint_law.mean", "state finite means"));
        }
        if covariance.iter().any(|value| !value.is_finite()) {
            return Err(mapping_uncertainty_invalid(
                "joint_law.covariance",
                "state a finite joint covariance",
            ));
        }
        let scale = (0..n).map(|index| covariance[index * n + index]).fold(0.0_f64, f64::max);
        for row in 0..n {
            if covariance[row * n + row] < 0.0 {
                return Err(mapping_uncertainty_invalid(
                    "joint_law.covariance",
                    "state non-negative variances on the diagonal",
                ));
            }
            for column in 0..row {
                if (covariance[row * n + column] - covariance[column * n + row]).abs()
                    > PSD_SLACK * scale
                {
                    return Err(mapping_uncertainty_invalid(
                        "joint_law.covariance",
                        "state a symmetric joint covariance",
                    ));
                }
            }
        }
        if cholesky(&covariance, n).is_none() {
            return Err(mapping_uncertainty_invalid(
                "joint_law.covariance",
                "state a positive semi-definite joint covariance: the declared covariances are not jointly feasible",
            ));
        }
        Ok(Self { names, mean, covariance })
    }

    /// The joint law of an affine mapping `a + b M` declared by `uncertainty`
    /// and `dependence`: coordinates `intercept`, `slope`, `source`.
    ///
    /// The source estimator is independent of the intercept estimator; its
    /// covariance with the slope estimator is the declared dependence.
    ///
    /// # Errors
    /// The declarations are malformed, the source moments are not finite and
    /// non-negative, the slope-source covariance exceeds what the two variances
    /// allow, or the three covariances are not jointly feasible.
    pub fn from_affine_declaration(
        intercept: f64,
        slope: f64,
        uncertainty: &MappingUncertainty,
        dependence: &SourceMappingDependence,
        source_mean: f64,
        source_variance: f64,
    ) -> Result<Self, BridgeRefusal> {
        uncertainty.validate()?;
        dependence.validate()?;
        if !source_mean.is_finite() {
            return Err(source_moment_not_finite("source_mean", "state a finite source mean"));
        }
        if !(source_variance.is_finite() && source_variance >= 0.0) {
            return Err(source_variance_invalid(source_variance.to_string()));
        }
        let covariance = dependence.covariance();
        let product = uncertainty.slope_variance * source_variance;
        if !product.is_finite() || covariance * covariance > product {
            return Err(dependence_not_a_covariance(covariance.to_string()));
        }
        Self::new(
            vec!["intercept".into(), "slope".into(), "source".into()],
            vec![intercept, slope, source_mean],
            vec![
                uncertainty.intercept_variance,
                uncertainty.intercept_slope_covariance,
                0.0,
                uncertainty.intercept_slope_covariance,
                uncertainty.slope_variance,
                covariance,
                0.0,
                covariance,
                source_variance,
            ],
        )
    }

    /// Number of coordinates.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// Coordinate names in order.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Coordinate means.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Row-major joint covariance.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    fn factor(&self) -> Vec<f64> {
        // `new` proved the factorization exists.
        cholesky(&self.covariance, self.mean.len()).unwrap_or_default()
    }
}

/// Lower-triangular `L` with `L Lᵀ = covariance`, tolerating zero-variance
/// directions; `None` when the matrix is not positive semi-definite.
fn cholesky(covariance: &[f64], n: usize) -> Option<Vec<f64>> {
    let scale = (0..n).map(|index| covariance[index * n + index]).fold(0.0_f64, f64::max);
    let degenerate = DEGENERATE_FRACTION * scale;
    let slack = PSD_SLACK * scale;
    let mut factor = vec![0.0; n * n];
    for column in 0..n {
        let mut pivot = covariance[column * n + column];
        for k in 0..column {
            pivot -= factor[column * n + k] * factor[column * n + k];
        }
        if pivot < -slack {
            return None;
        }
        if pivot <= degenerate {
            for row in column + 1..n {
                let mut residual = covariance[row * n + column];
                for k in 0..column {
                    residual -= factor[row * n + k] * factor[column * n + k];
                }
                if residual.abs() > slack {
                    return None;
                }
            }
        } else {
            let diagonal = pivot.sqrt();
            factor[column * n + column] = diagonal;
            for row in column + 1..n {
                let mut residual = covariance[row * n + column];
                for k in 0..column {
                    residual -= factor[row * n + k] * factor[column * n + k];
                }
                factor[row * n + column] = residual / diagonal;
            }
        }
    }
    Some(factor)
}

/// How a nonlinear function of a joint law is carried to a mean and a variance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PropagationMethod {
    /// Second-order delta method under the Gaussian moment closure of the
    /// joint law: `g(m) + ½ tr(HΣ)` and `∇ᵀΣ∇ + ½ tr((HΣ)²)`, with the
    /// gradient and Hessian from central differences.
    SecondOrderDelta,
    /// Sigma-point (unscented) transform: `2n + 1` deterministic points chosen
    /// to match the joint mean and covariance, with the fourth-moment
    /// correction `κ = max(3 − n, 0)`.
    Unscented,
    /// Seeded Monte Carlo draws of the joint law. The result carries the
    /// Monte Carlo standard errors of the mean and of the variance.
    Simulation {
        /// Number of draws.
        draws: usize,
        /// Seed of the draw stream.
        seed: u64,
    },
}

impl PropagationMethod {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SecondOrderDelta => "second_order_delta",
            Self::Unscented => "unscented",
            Self::Simulation { .. } => "simulation",
        }
    }
}

/// Mean and variance of `g` over a joint law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointPropagation {
    /// The method that produced the moments.
    pub method: PropagationMethod,
    /// Coordinates of the joint law.
    pub dimension: usize,
    /// `g` at the joint mean: the plug-in value.
    pub plug_in: f64,
    /// `E[g]`.
    pub mean: f64,
    /// `Var(g)`.
    pub variance: f64,
    /// First-order variance `∇ᵀΣ∇`; set by the delta method only.
    pub linearized_variance: Option<f64>,
    /// Monte Carlo standard error of `mean`; set by simulation only.
    pub mean_standard_error: Option<f64>,
    /// Monte Carlo standard error of `variance`; set by simulation only.
    pub variance_standard_error: Option<f64>,
    /// Evaluations of `g` the method used.
    pub evaluations: usize,
}

impl JointPropagation {
    /// `E[g] − g(m)`: what the plug-in value omits.
    #[must_use]
    pub fn bias(&self) -> f64 {
        self.mean - self.plug_in
    }

    /// Standard deviation of `g`.
    #[must_use]
    pub fn standard_deviation(&self) -> f64 {
        self.variance.sqrt()
    }
}

fn checked(value: f64) -> Result<f64, BridgeRefusal> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(propagation_overflow(
            "nonlinear_map",
            "the function produced a value that is not finite inside the joint law; restrict it to its domain or use a smaller spread",
        ))
    }
}

/// Mean and variance of `g(Z)` for `Z` with the joint moments of `law`.
///
/// # Errors
/// The function is not finite at a point the method evaluates; a simulation
/// asks for fewer than 64 or more than four million draws.
pub fn propagate_joint(
    law: &JointLaw,
    g: &dyn Fn(&[f64]) -> f64,
    method: PropagationMethod,
) -> Result<JointPropagation, BridgeRefusal> {
    match method {
        PropagationMethod::SecondOrderDelta => second_order_delta(law, g),
        PropagationMethod::Unscented => unscented(law, g),
        PropagationMethod::Simulation { draws, seed } => simulate(law, g, draws, seed),
    }
}

fn second_order_delta(
    law: &JointLaw,
    g: &dyn Fn(&[f64]) -> f64,
) -> Result<JointPropagation, BridgeRefusal> {
    let n = law.dimension();
    let sigma = law.covariance();
    let center = law.mean();
    let step: Vec<f64> = (0..n).map(|i| DIFFERENCE_STEP * sigma[i * n + i].sqrt()).collect();
    let active: Vec<usize> = (0..n).filter(|&i| sigma[i * n + i] > 0.0).collect();
    let mut evaluations = 0_usize;
    let mut eval = |shifts: &[(usize, f64)]| -> Result<f64, BridgeRefusal> {
        let mut point = center.to_vec();
        for &(index, delta) in shifts {
            point[index] += delta;
        }
        evaluations += 1;
        checked(g(&point))
    };
    let plug_in = eval(&[])?;
    let mut gradient = vec![0.0; n];
    let mut hessian = vec![0.0; n * n];
    for &i in &active {
        let up = eval(&[(i, step[i])])?;
        let down = eval(&[(i, -step[i])])?;
        gradient[i] = (up - down) / (2.0 * step[i]);
        hessian[i * n + i] = (up - 2.0 * plug_in + down) / (step[i] * step[i]);
    }
    for (position, &i) in active.iter().enumerate() {
        for &j in &active[position + 1..] {
            let upper = eval(&[(i, step[i]), (j, step[j])])?;
            let left = eval(&[(i, step[i]), (j, -step[j])])?;
            let right = eval(&[(i, -step[i]), (j, step[j])])?;
            let lower = eval(&[(i, -step[i]), (j, -step[j])])?;
            let mixed = (upper - left - right + lower) / (4.0 * step[i] * step[j]);
            hessian[i * n + j] = mixed;
            hessian[j * n + i] = mixed;
        }
    }
    let mut correction = 0.0;
    let mut linearized = 0.0;
    for i in 0..n {
        for j in 0..n {
            correction += hessian[i * n + j] * sigma[i * n + j];
            linearized += gradient[i] * sigma[i * n + j] * gradient[j];
        }
    }
    // A = H Σ; tr(A²) = Σ_ij A_ij A_ji.
    let mut product = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += hessian[i * n + k] * sigma[k * n + j];
            }
            product[i * n + j] = sum;
        }
    }
    let mut trace_square = 0.0;
    for i in 0..n {
        for j in 0..n {
            trace_square += product[i * n + j] * product[j * n + i];
        }
    }
    let mean = plug_in + 0.5 * correction;
    let variance = (linearized + 0.5 * trace_square).max(0.0);
    if !mean.is_finite() || !variance.is_finite() {
        return Err(propagation_overflow(
            "second_order_delta",
            "the second-order delta method overflowed the finite range",
        ));
    }
    Ok(JointPropagation {
        method: PropagationMethod::SecondOrderDelta,
        dimension: n,
        plug_in,
        mean,
        variance,
        linearized_variance: Some(linearized),
        mean_standard_error: None,
        variance_standard_error: None,
        evaluations,
    })
}

fn unscented(
    law: &JointLaw,
    g: &dyn Fn(&[f64]) -> f64,
) -> Result<JointPropagation, BridgeRefusal> {
    let n = law.dimension();
    let factor = law.factor();
    let center = law.mean();
    let kappa = if n < 3 { (3 - n) as f64 } else { 0.0 };
    let spread = (n as f64 + kappa).sqrt();
    let central_weight = kappa / (n as f64 + kappa);
    let side_weight = 1.0 / (2.0 * (n as f64 + kappa));
    let plug_in = checked(g(center))?;
    let mut values = Vec::with_capacity(2 * n);
    for column in 0..n {
        for sign in [1.0, -1.0] {
            let point: Vec<f64> =
                (0..n).map(|row| center[row] + sign * spread * factor[row * n + column]).collect();
            values.push(checked(g(&point))?);
        }
    }
    let mean = central_weight * plug_in + side_weight * values.iter().sum::<f64>();
    let mut variance = central_weight * (plug_in - mean).powi(2);
    for value in &values {
        variance += side_weight * (value - mean).powi(2);
    }
    if !mean.is_finite() || !variance.is_finite() {
        return Err(propagation_overflow(
            "unscented",
            "the sigma-point transform overflowed the finite range",
        ));
    }
    Ok(JointPropagation {
        method: PropagationMethod::Unscented,
        dimension: n,
        plug_in,
        mean,
        variance,
        linearized_variance: None,
        mean_standard_error: None,
        variance_standard_error: None,
        evaluations: 1 + values.len(),
    })
}

/// SplitMix64 stream: a fixed seed reproduces the draws on every platform.
struct DrawStream(u64);

impl DrawStream {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on the open interval (0, 1).
    fn uniform(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) / 9_007_199_254_740_992.0
    }

    fn normal_pair(&mut self) -> (f64, f64) {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        (radius * angle.cos(), radius * angle.sin())
    }
}

fn simulate(
    law: &JointLaw,
    g: &dyn Fn(&[f64]) -> f64,
    draws: usize,
    seed: u64,
) -> Result<JointPropagation, BridgeRefusal> {
    if !(MIN_SIMULATION_DRAWS..=MAX_SIMULATION_DRAWS).contains(&draws) {
        return Err(malformed_mapping(
            "simulation.draws",
            "state between 64 and 4000000 simulated draws",
        ));
    }
    let n = law.dimension();
    let factor = law.factor();
    let center = law.mean();
    let plug_in = checked(g(center))?;
    let mut stream = DrawStream(seed);
    let mut normals = vec![0.0; n];
    let mut point = vec![0.0; n];
    let mut values = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut index = 0;
        while index < n {
            let (first, second) = stream.normal_pair();
            normals[index] = first;
            if index + 1 < n {
                normals[index + 1] = second;
            }
            index += 2;
        }
        for row in 0..n {
            let mut value = center[row];
            for column in 0..=row {
                value += factor[row * n + column] * normals[column];
            }
            point[row] = value;
        }
        values.push(checked(g(&point))?);
    }
    let count = draws as f64;
    let mean = values.iter().sum::<f64>() / count;
    let mut second = 0.0;
    let mut fourth = 0.0;
    for value in &values {
        let deviation = value - mean;
        second += deviation * deviation;
        fourth += deviation.powi(4);
    }
    let population_variance = second / count;
    let variance = second / (count - 1.0);
    let fourth_moment = fourth / count;
    let mean_standard_error = (variance / count).sqrt();
    let variance_standard_error =
        ((fourth_moment - population_variance * population_variance).max(0.0) / count).sqrt();
    if !mean.is_finite() || !variance.is_finite() {
        return Err(propagation_overflow(
            "simulation",
            "the simulated moments overflowed the finite range",
        ));
    }
    Ok(JointPropagation {
        method: PropagationMethod::Simulation { draws, seed },
        dimension: n,
        plug_in,
        mean,
        variance,
        linearized_variance: None,
        mean_standard_error: Some(mean_standard_error),
        variance_standard_error: Some(variance_standard_error),
        evaluations: draws + 1,
    })
}

/// A nonlinear function of a contract's affine map, with the declarations it
/// was built from.
#[derive(Clone, Debug, PartialEq)]
pub struct ContractNonlinearPropagation {
    /// Mean and variance of the function over the joint law.
    pub propagation: JointPropagation,
    /// The joint law the moments were taken over.
    pub joint_law: JointLaw,
    /// The mapping uncertainty the law was built from.
    pub mapping_uncertainty: MappingUncertainty,
    /// The dependence the law was built from.
    pub dependence: SourceMappingDependence,
    /// Contract assumptions in identity order.
    pub assumptions: Vec<BridgeAssumption>,
}

/// Propagate `g(intercept, slope, source)` through the contract's uncertain
/// affine mapping and its declared source dependence.
///
/// The three inputs are drawn from their joint law: the mapping's intercept
/// and slope estimators with their declared variances and covariance, and the
/// source estimator with its variance and the declared covariance with the
/// slope. `g` is the function of the mapped quantity the consumer needs, for
/// example a utility `exp(intercept + slope * source)`.
///
/// # Errors
/// The contract is not licensed; the mapping is not affine; mapping
/// uncertainty or source dependence is missing or not feasible jointly;
/// [`propagate_joint`] refuses.
pub fn propagate_nonlinear(
    contract: &BridgeContract,
    source_mean: f64,
    source_variance: f64,
    g: &dyn Fn(f64, f64, f64) -> f64,
    method: PropagationMethod,
) -> Result<ContractNonlinearPropagation, BridgeRefusal> {
    contract.check_licensing()?;
    let (intercept, slope) = affine_coefficients(&contract.mapping)?;
    let Some(uncertainty) = contract.mapping_uncertainty.clone() else {
        return Err(mapping_uncertainty_missing());
    };
    let Some(dependence) = contract.source_dependence.clone() else {
        return Err(source_dependence_missing());
    };
    let law = JointLaw::from_affine_declaration(
        intercept,
        slope,
        &uncertainty,
        &dependence,
        source_mean,
        source_variance,
    )?;
    let propagation = propagate_joint(&law, &|point| g(point[0], point[1], point[2]), method)?;
    Ok(ContractNonlinearPropagation {
        propagation,
        joint_law: law,
        mapping_uncertainty: uncertainty,
        dependence,
        assumptions: sorted_assumptions(&contract.assumptions),
    })
}

pub(super) fn estimand_matches(
    contract: &BridgeContract,
    bridge: &EstimandBridge,
) -> Result<(), BridgeRefusal> {
    let evidence = &contract.evidence_scope;
    let target = &contract.target.quantity;
    if bridge.experiment.population_id != evidence.population_id {
        return Err(estimand_bridge_mismatch(
            format!("evidence population `{}`", evidence.population_id),
            format!("experiment population `{}`", bridge.experiment.population_id),
        ));
    }
    if bridge.experiment.exposure_regime != evidence.regime_id {
        return Err(estimand_bridge_mismatch(
            format!("evidence regime `{}`", evidence.regime_id),
            format!("experiment regime `{}`", bridge.experiment.exposure_regime),
        ));
    }
    if bridge.target.population_id != target.population_id {
        return Err(estimand_bridge_mismatch(
            format!("target population `{}`", target.population_id),
            format!("translated population `{}`", bridge.target.population_id),
        ));
    }
    if bridge.target.exposure_regime != target.regime_id {
        return Err(estimand_bridge_mismatch(
            format!("target regime `{}`", target.regime_id),
            format!("translated regime `{}`", bridge.target.exposure_regime),
        ));
    }
    check_spend_consistency(bridge)
}

/// Relative and absolute tolerance on the spend difference.
const SPEND_TOLERANCE: f64 = 1e-9;

/// The spend difference must be the target dose minus the experiment dose, in
/// one unit. Recency labels are not parsed and are not checked here.
fn check_spend_consistency(bridge: &EstimandBridge) -> Result<(), BridgeRefusal> {
    let units = [
        bridge.experiment.dose_units.as_str(),
        bridge.target.dose_units.as_str(),
        bridge.spend_units.as_str(),
    ];
    if units.iter().any(|unit| *unit != units[0]) {
        return Err(estimand_units_mismatch(
            format!("one unit identity (experiment dose `{}`)", units[0]),
            format!("target dose `{}`, spend `{}`", units[1], units[2]),
        ));
    }
    let expected = bridge.target.dose - bridge.experiment.dose;
    let scale = 1.0_f64.max(expected.abs()).max(bridge.spend_delta.abs());
    if !expected.is_finite() || (bridge.spend_delta - expected).abs() > SPEND_TOLERANCE * scale {
        return Err(estimand_spend_inconsistent(
            format!("spend difference {expected}"),
            format!("spend difference {}", bridge.spend_delta),
        ));
    }
    Ok(())
}

/// Accept `prior` only with an estimand bridge, keeping its uncertainty,
/// its dependence and the contract's assumptions.
///
/// The contract is not modified. [`BridgeContract::import_experimental_prior`]
/// stores the result after this returns.
///
/// # Errors
/// The prior has no estimand bridge; an identity is blank; the bridge, the
/// uncertainty or the dependence is not well formed; the bridge's experiment
/// id disagrees with the prior; the translation does not name this contract's
/// evidence scope and target quantity.
pub(super) fn import_prior(
    contract: &BridgeContract,
    prior: &ExperimentalPrior,
) -> Result<ImportedExperimentalPrior, BridgeRefusal> {
    let Some(bridge) = prior.estimand_bridge.clone() else {
        return Err(experimental_prior_without_estimand_bridge());
    };
    check_text("prior.id", &prior.prior_id)?;
    check_text("prior.experiment_id", &prior.experiment_id)?;
    bridge.validate()?;
    if let Some(uncertainty) = &prior.mapping_uncertainty {
        uncertainty.validate()?;
    }
    if let Some(dependence) = &prior.source_dependence {
        dependence.validate()?;
    }
    if bridge.experiment_id != prior.experiment_id {
        return Err(estimand_bridge_mismatch(
            format!("experiment `{}`", prior.experiment_id),
            format!("bridge experiment `{}`", bridge.experiment_id),
        ));
    }
    estimand_matches(contract, &bridge)?;
    Ok(ImportedExperimentalPrior {
        prior_id: prior.prior_id.clone(),
        experiment_id: prior.experiment_id.clone(),
        estimand_bridge: bridge,
        mapping_uncertainty: prior.mapping_uncertainty.clone(),
        source_dependence: prior.source_dependence.clone(),
        assumptions: sorted_assumptions(&contract.assumptions),
    })
}

pub(super) fn write_extension(
    out: &mut CanonicalWriter,
    uncertainty: Option<&MappingUncertainty>,
    dependence: Option<&SourceMappingDependence>,
    estimand: Option<&EstimandBridge>,
) {
    if uncertainty.is_none() && dependence.is_none() && estimand.is_none() {
        return;
    }
    out.word(EXTENSION_SECTION);
    match uncertainty {
        None => {
            out.word(0);
        }
        Some(item) => {
            out.word(1);
            item.write(out);
        }
    }
    match dependence {
        None => {
            out.word(0);
        }
        Some(item) => item.write(out),
    }
    match estimand {
        None => {
            out.word(0);
        }
        Some(item) => {
            out.word(1);
            item.write(out);
        }
    }
}

pub(super) fn validate_finite_affine_mapping(
    contract: &BridgeContract,
    intercepts: &[f64],
    slopes: &[f64],
    source: &[f64],
) -> Result<(), BridgeRefusal> {
    contract.check_licensing()?;
    if source.is_empty()
        || intercepts.len() != source.len()
        || slopes.len() != source.len()
        || source.iter().chain(intercepts).any(|value| !value.is_finite())
        || slopes.iter().any(|value| !value.is_finite() || *value == 0.0)
    {
        return Err(mapping_uncertainty_invalid(
            "finite_joint_slots",
            "retain every finite aligned source/mapping slot with a nonzero slope",
        ));
    }
    let (a, b) = match &contract.mapping {
        BridgeMapping::Identity => (0.0, 1.0),
        BridgeMapping::Affine { intercept, slope } => (*intercept, *slope),
        _ => return Err(uncertain_mapping_not_affine("non_affine")),
    };
    let n = source.len() as f64;
    let mean = |values: &[f64]| values.iter().map(|value| value / n).sum::<f64>();
    let close = |left: f64, right: f64| {
        left.is_finite()
            && right.is_finite()
            && (left - right).abs() <= 1e-10 + 1e-10 * left.abs() + 1e-10 * right.abs()
    };
    if !close(mean(intercepts), a) || !close(mean(slopes), b) {
        return Err(mapping_uncertainty_invalid(
            "finite_mapping_mean",
            "declare the affine means of the retained finite joint mapping law",
        ));
    }
    let covariance = |left: &[f64], right: &[f64]| {
        let lm = mean(left);
        let rm = mean(right);
        left.iter().zip(right).map(|(l, r)| (l - lm) * (r - rm)).sum::<f64>() / n
    };
    let (va, vb, cab) =
        contract.mapping_uncertainty.as_ref().map_or((0.0, 0.0, 0.0), |uncertainty| {
            (
                uncertainty.intercept_variance,
                uncertainty.slope_variance,
                uncertainty.intercept_slope_covariance,
            )
        });
    if !close(covariance(intercepts, intercepts), va)
        || !close(covariance(slopes, slopes), vb)
        || !close(covariance(intercepts, slopes), cab)
    {
        return Err(mapping_uncertainty_invalid(
            "finite_mapping_covariance",
            "declare the moments of the retained finite paired mapping law",
        ));
    }
    if let Some(SourceMappingDependence::SlopeCovariance { covariance: declared, .. }) =
        &contract.source_dependence
    {
        if !close(covariance(slopes, source), *declared) {
            return Err(source_dependence_invalid(
                "finite_source_mapping_covariance",
                "retain the declared finite source/mapping covariance in the paired slots",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{ContractIdentities, SemanticDigest};

    use super::super::super::claim::{BridgeSource, derive_bridged_claim};
    use super::super::super::kind::{AssumptionKind, BridgeKind, QuantityClass, ValueKind};
    use super::super::super::mapping::{BridgeMapping, TrialLevelValidation};
    use super::super::{BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeScope};
    use super::{
        AnalyticLaw, EstimandBridge, ExperimentalPrior, ExposureSetting, MappingUncertainty,
        NonlinearMap, SourceMappingDependence, SupportMap, ValueInterval, nonlinear_expectation,
        support_image,
    };
    use crate::{
        ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, QuantityRole, ReasoningView,
        ScientificQuantity, SlotAvailability,
    };

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::from_bytes([byte; 32])
    }

    fn refused(error: &super::super::super::refusal::BridgeRefusal, code: &str, detail: &str) {
        assert_eq!(error.code, code, "{error:?}");
        assert_eq!(error.detail, detail, "{error:?}");
        assert_eq!(error.validate(), Ok(()), "{error:?}");
        assert!(error.remedy.is_some(), "{error:?}");
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

    fn linkage(id: &str, statement: &str) -> BridgeAssumption {
        BridgeAssumption {
            id: id.into(),
            kind: AssumptionKind::BehavioralLinkage,
            statement: statement.into(),
            evidence: Some("validation-panel-2023".into()),
        }
    }

    fn preference(mapping: BridgeMapping, assumptions: Vec<BridgeAssumption>) -> BridgeContract {
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
            assumptions,
            transport: None,
            mapping_uncertainty: None,
            source_dependence: None,
            estimand_bridge: None,
        }
    }

    #[test]
    fn finite_mapping_moment_overflow_cannot_pass_as_matching_finite_moments() {
        let contract = preference(BridgeMapping::Affine { intercept: 0.0, slope: 1.0 }, vec![]);
        assert!(
            super::validate_finite_affine_mapping(
                &contract,
                &[0.0, 0.0],
                &[f64::MAX, f64::MAX],
                &[0.0, 0.0],
            )
            .is_err()
        );
    }

    fn uncertainty(slope_variance: f64) -> MappingUncertainty {
        MappingUncertainty {
            id: "slope-se".into(),
            intercept_variance: 0.0,
            slope_variance,
            intercept_slope_covariance: 0.0,
        }
    }

    fn setting(dose: f64, duration: f64, start: &str, end: &str) -> ExposureSetting {
        ExposureSetting {
            population_id: "panel".into(),
            time_start: start.into(),
            time_end: end.into(),
            dose,
            dose_units: "usd_per_capita".into(),
            duration,
            duration_units: "week".into(),
            exposure_regime: "observational".into(),
        }
    }

    fn translation(recency: f64, spend: f64) -> EstimandBridge {
        EstimandBridge {
            id: "geo-to-panel".into(),
            experiment_id: "geo-2024".into(),
            experiment: setting(10.0, 12.0, "2024-01", "2024-04"),
            target: setting(25.0, 8.0, "2024-06", "2024-09"),
            recency_delta: recency,
            recency_units: "week".into(),
            spend_delta: spend,
            spend_units: "usd_per_capita".into(),
        }
    }

    fn point_claim(value: f64) -> ClaimEnvelope {
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
            [Arc::<str>::from("prior:geo-2024")],
        )
    }

    #[test]
    fn square_of_the_mean_is_not_the_mean_of_the_square() {
        let points = [(-1.0_f64, 0.5_f64), (1.0, 0.5)];
        let mean = points.iter().map(|(x, p)| x * p).sum::<f64>();
        let expectation = points.iter().map(|(x, p)| x * x * p).sum::<f64>();
        let variance = points.iter().map(|(x, p)| (x - mean).powi(2) * p).sum::<f64>();
        let got = nonlinear_expectation(
            NonlinearMap::Square,
            &AnalyticLaw::SecondMoment { mean, variance },
        )
        .unwrap();
        assert_eq!(got.map_of_mean.to_bits(), (mean * mean).to_bits());
        assert_eq!(got.expectation.to_bits(), expectation.to_bits());
        assert_ne!(got.map_of_mean.to_bits(), got.expectation.to_bits());

        let degenerate = nonlinear_expectation(
            NonlinearMap::Square,
            &AnalyticLaw::SecondMoment { mean: 3.0, variance: 0.0 },
        )
        .unwrap();
        assert_eq!(degenerate.expectation.to_bits(), 9.0_f64.to_bits());
        assert_eq!(degenerate.map_of_mean.to_bits(), degenerate.expectation.to_bits());
    }

    #[test]
    fn normal_exponential_matches_its_moment_generating_function() {
        let got = nonlinear_expectation(
            NonlinearMap::Exponential,
            &AnalyticLaw::Normal { mean: 0.0, variance: 2.0 },
        )
        .unwrap();
        assert_eq!(got.map_of_mean.to_bits(), 1.0_f64.to_bits());
        assert_eq!(got.expectation.to_bits(), 1.0_f64.exp().to_bits());
        assert_ne!(got.map_of_mean.to_bits(), got.expectation.to_bits());
    }

    #[test]
    fn a_nonlinear_map_of_a_mean_is_refused() {
        let error =
            nonlinear_expectation(NonlinearMap::Square, &AnalyticLaw::MeanOnly { mean: 0.5 })
                .unwrap_err();
        refused(
            &error,
            "bridge_mapping_not_licensed",
            "scientific_bridge.nonlinear_mean_not_expectation",
        );
        let error = nonlinear_expectation(
            NonlinearMap::Exponential,
            &AnalyticLaw::SecondMoment { mean: 0.0, variance: 1.0 },
        )
        .unwrap_err();
        refused(
            &error,
            "bridge_mapping_not_licensed",
            "scientific_bridge.nonlinear_law_insufficient",
        );
    }

    #[test]
    fn support_images_match_the_analytic_map() {
        let square =
            support_image(SupportMap::Square, ValueInterval { low: -1.0, high: 2.0 }).unwrap();
        assert_eq!(square.low.to_bits(), 0.0_f64.to_bits());
        assert_eq!(square.high.to_bits(), 4.0_f64.to_bits());
        let affine = support_image(
            SupportMap::Affine { intercept: 1.0, slope: -2.0 },
            ValueInterval { low: 0.0, high: 3.0 },
        )
        .unwrap();
        assert_eq!(affine.low.to_bits(), (-5.0_f64).to_bits());
        assert_eq!(affine.high.to_bits(), 1.0_f64.to_bits());
        let exponential =
            support_image(SupportMap::Exponential, ValueInterval { low: 0.0, high: 1.0 }).unwrap();
        assert_eq!(exponential.low.to_bits(), 1.0_f64.to_bits());
        assert_eq!(exponential.high.to_bits(), 1.0_f64.exp().to_bits());
    }

    #[test]
    fn nonlinear_support_keeps_bridge_assumptions() {
        let mut contract = preference(
            BridgeMapping::Affine { intercept: 0.0, slope: 2.0 },
            vec![linkage("b-link", "b holds"), linkage("a-link", "a holds")],
        );
        let support = contract
            .map_support(SupportMap::Square, ValueInterval { low: -1.0, high: 2.0 })
            .unwrap();
        assert_eq!(support.low.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            support.assumptions.iter().map(|item| item.id.as_str()).collect::<Vec<_>>(),
            ["a-link", "b-link"]
        );
        let before = contract.identity();
        contract.assumptions.iter_mut().find(|item| item.id == "a-link").unwrap().statement =
            "revised".into();
        assert_ne!(before, contract.identity());
        let again = contract
            .map_support(SupportMap::Square, ValueInterval { low: -1.0, high: 2.0 })
            .unwrap();
        assert_eq!(again.assumptions[0].statement, "revised");
        assert_eq!(again.assumptions.len(), 2);
    }

    #[test]
    fn contract_bound_propagation_requires_the_principal_assumption() {
        let mut contract = preference(BridgeMapping::Identity, vec![]);
        contract.mapping_uncertainty = Some(uncertainty(0.25));
        contract.source_dependence =
            Some(SourceMappingDependence::Independent { evidence: "disjoint-rows".into() });
        let expected = contract.check_licensing().unwrap_err();
        assert_eq!(contract.propagate_affine_uncertainty(4.0, 0.25).unwrap_err(), expected);
        assert_eq!(
            contract
                .map_support(SupportMap::Square, ValueInterval { low: 0.0, high: 2.0 })
                .unwrap_err(),
            expected
        );
    }

    #[test]
    fn affine_uncertainty_matches_the_delta_method() {
        let mut contract = preference(
            BridgeMapping::Affine { intercept: 0.0, slope: 2.0 },
            vec![linkage("hypothetical-bias", "stated scales realized")],
        );
        contract.mapping_uncertainty = Some(uncertainty(0.0625));
        contract.source_dependence =
            Some(SourceMappingDependence::Independent { evidence: "disjoint-rows".into() });
        let independent = contract.propagate_affine_uncertainty(4.0, 0.25).unwrap();
        assert_eq!(independent.plug_in_mean.to_bits(), 8.0_f64.to_bits());
        assert_eq!(independent.mean.to_bits(), independent.plug_in_mean.to_bits());
        // 4^2 * 0.0625 + 2^2 * 0.25 + 0.0625 * 0.25 = 1 + 1 + 0.015625.
        assert_eq!(independent.variance.to_bits(), 2.015_625_f64.to_bits());
        assert_eq!(independent.assumptions[0].id, "hypothetical-bias");

        contract.source_dependence = Some(SourceMappingDependence::SlopeCovariance {
            covariance: 0.125,
            evidence: "shared-rows".into(),
        });
        let shared = contract.propagate_affine_uncertainty(4.0, 0.25).unwrap();
        assert_eq!(shared.plug_in_mean.to_bits(), 8.0_f64.to_bits());
        assert_eq!(shared.mean.to_bits(), 8.125_f64.to_bits());
        assert_ne!(shared.plug_in_mean.to_bits(), shared.mean.to_bits());
        // 1 + 1 + 2 * 4 * 2 * 0.125 + 0.0625 * 0.25 + 0.125^2 = 4.03125.
        assert_eq!(shared.variance.to_bits(), 4.03125_f64.to_bits());
        assert_eq!(shared.mapping_uncertainty.slope_variance.to_bits(), 0.0625_f64.to_bits());
        assert!(matches!(
            shared.dependence,
            SourceMappingDependence::SlopeCovariance { covariance, .. }
                if covariance.to_bits() == 0.125_f64.to_bits()
        ));
    }

    #[test]
    fn propagation_without_uncertainty_or_dependence_is_refused() {
        let mut contract = preference(BridgeMapping::Identity, vec![linkage("link", "holds")]);
        let missing = contract.propagate_affine_uncertainty(1.0, 1.0).unwrap_err();
        refused(
            &missing,
            "bridge_assumption_missing",
            "scientific_bridge.mapping_uncertainty_missing",
        );
        contract.mapping_uncertainty = Some(uncertainty(0.25));
        let dependence = contract.propagate_affine_uncertainty(1.0, 1.0).unwrap_err();
        refused(
            &dependence,
            "sampling_dependence_unknown",
            "scientific_bridge.source_dependence_missing",
        );
        contract.source_dependence = Some(SourceMappingDependence::SlopeCovariance {
            covariance: 1.0,
            evidence: "rows".into(),
        });
        let covariance = contract.propagate_affine_uncertainty(1.0, 0.25).unwrap_err();
        refused(
            &covariance,
            "bridge_mapping_not_licensed",
            "scientific_bridge.dependence_not_a_covariance",
        );
        contract.mapping = BridgeMapping::TrialLevelLinear(TrialLevelValidation {
            trials: Vec::new(),
            min_trials: 3,
            min_r_squared: 0.0,
        });
        let affine = contract.propagate_affine_uncertainty(1.0, 0.25).unwrap_err();
        refused(&affine, "scientific_bridge_invalid", "scientific_bridge.mapping_not_allowed");
        contract.mapping = BridgeMapping::Identity;
        contract.mapping_uncertainty = Some(MappingUncertainty {
            id: "slope-se".into(),
            intercept_variance: 0.0,
            slope_variance: -1.0,
            intercept_slope_covariance: 0.0,
        });
        let negative = contract.propagate_affine_uncertainty(1.0, 0.25).unwrap_err();
        refused(
            &negative,
            "scientific_bridge_invalid",
            "scientific_bridge.mapping_uncertainty_invalid",
        );
    }

    #[test]
    fn identity_changes_with_mapping_uncertainty_and_source_dependence() {
        let mut contract = preference(BridgeMapping::Identity, vec![linkage("link", "holds")]);
        let base = contract.identity();
        contract.mapping_uncertainty = Some(uncertainty(0.0625));
        let with_uncertainty = contract.identity();
        assert_ne!(base, with_uncertainty);
        contract.mapping_uncertainty.as_mut().unwrap().slope_variance = 0.25;
        assert_ne!(with_uncertainty, contract.identity());
        contract.mapping_uncertainty = None;
        assert_eq!(base, contract.identity());

        contract.source_dependence =
            Some(SourceMappingDependence::Independent { evidence: "rows-a".into() });
        let independent = contract.identity();
        assert_ne!(base, independent);
        contract.source_dependence = Some(SourceMappingDependence::SlopeCovariance {
            covariance: 0.0,
            evidence: "rows-a".into(),
        });
        let zero = contract.identity();
        assert_ne!(independent, zero);
        contract.source_dependence = Some(SourceMappingDependence::SlopeCovariance {
            covariance: 0.125,
            evidence: "rows-a".into(),
        });
        let covariance = contract.identity();
        assert_ne!(zero, covariance);
        contract.source_dependence = Some(SourceMappingDependence::SlopeCovariance {
            covariance: 0.125,
            evidence: "rows-b".into(),
        });
        assert_ne!(covariance, contract.identity());
    }

    #[test]
    fn an_experimental_prior_without_an_estimand_bridge_is_refused() {
        let mut contract = preference(BridgeMapping::Identity, vec![linkage("link", "holds")]);
        let before = contract.identity();
        let prior = ExperimentalPrior {
            prior_id: "prior-geo".into(),
            experiment_id: "geo-2024".into(),
            estimand_bridge: None,
            mapping_uncertainty: Some(uncertainty(0.0625)),
            source_dependence: Some(SourceMappingDependence::Independent {
                evidence: "disjoint-rows".into(),
            }),
        };
        let error = contract.import_experimental_prior(&prior).unwrap_err();
        refused(
            &error,
            "bridge_assumption_missing",
            "scientific_bridge.experimental_prior_without_estimand_bridge",
        );
        assert_eq!(before, contract.identity());
        assert!(contract.mapping_uncertainty.is_none());
    }

    #[test]
    fn an_imported_experiment_states_every_estimand_dimension() {
        let mut contract = preference(
            BridgeMapping::Affine { intercept: 0.0, slope: 2.0 },
            vec![linkage("hypothetical-bias", "stated scales realized")],
        );
        let mut bridge = translation(20.0, 15.0);
        bridge.experiment.exposure_regime.clear();
        let incomplete = ExperimentalPrior {
            prior_id: "prior-geo".into(),
            experiment_id: "geo-2024".into(),
            estimand_bridge: Some(bridge),
            mapping_uncertainty: None,
            source_dependence: None,
        };
        let error = contract.import_experimental_prior(&incomplete).unwrap_err();
        refused(
            &error,
            "scientific_bridge_invalid",
            "scientific_bridge.estimand_bridge_incomplete",
        );
        assert!(contract.estimand_bridge.is_none());

        let mut mismatched = translation(20.0, 15.0);
        mismatched.experiment.population_id = "geo-east".into();
        let prior = ExperimentalPrior {
            prior_id: "prior-geo".into(),
            experiment_id: "geo-2024".into(),
            estimand_bridge: Some(mismatched),
            mapping_uncertainty: Some(uncertainty(0.0625)),
            source_dependence: Some(SourceMappingDependence::Independent {
                evidence: "disjoint-rows".into(),
            }),
        };
        let before = contract.identity();
        let error = contract.import_experimental_prior(&prior).unwrap_err();
        refused(&error, "bridge_binding_mismatch", "scientific_bridge.estimand_bridge_mismatch");
        assert_eq!(before, contract.identity());

        let prior = ExperimentalPrior { estimand_bridge: Some(translation(20.0, 15.0)), ..prior };
        let imported = contract.import_experimental_prior(&prior).unwrap();
        assert_eq!(imported.prior_id, "prior-geo");
        assert_eq!(imported.experiment_id, "geo-2024");
        assert_eq!(imported.estimand_bridge.experiment.population_id, "panel");
        assert_eq!(imported.estimand_bridge.experiment.time_start, "2024-01");
        assert_eq!(imported.estimand_bridge.experiment.time_end, "2024-04");
        assert_eq!(imported.estimand_bridge.experiment.dose.to_bits(), 10.0_f64.to_bits());
        assert_eq!(imported.estimand_bridge.experiment.duration.to_bits(), 12.0_f64.to_bits());
        assert_eq!(imported.estimand_bridge.experiment.exposure_regime, "observational");
        assert_eq!(imported.estimand_bridge.target.population_id, "panel");
        assert_eq!(imported.estimand_bridge.target.time_start, "2024-06");
        assert_eq!(imported.estimand_bridge.target.dose.to_bits(), 25.0_f64.to_bits());
        assert_eq!(imported.estimand_bridge.target.duration.to_bits(), 8.0_f64.to_bits());
        assert_eq!(imported.estimand_bridge.target.exposure_regime, "observational");
        assert_eq!(imported.estimand_bridge.recency_delta.to_bits(), 20.0_f64.to_bits());
        assert_eq!(imported.estimand_bridge.spend_delta.to_bits(), 15.0_f64.to_bits());
        assert_eq!(
            imported.mapping_uncertainty.as_ref().map(|item| item.id.as_str()),
            Some("slope-se")
        );
        assert!(imported.source_dependence.is_some());
        assert_eq!(imported.assumptions[0].id, "hypothetical-bias");
        let stored = contract.estimand_bridge.as_ref().map(|item| item.spend_delta.to_bits());
        assert_eq!(stored, Some(15.0_f64.to_bits()));
        let imported_identity = contract.identity();
        assert_ne!(before, imported_identity);
        contract.estimand_bridge.as_mut().unwrap().spend_delta = 0.0;
        assert_ne!(imported_identity, contract.identity());
    }

    #[test]
    fn an_inconsistent_spend_or_unit_is_refused() {
        let mut contract = preference(
            BridgeMapping::Affine { intercept: 0.0, slope: 2.0 },
            vec![linkage("hypothetical-bias", "stated scales realized")],
        );
        let prior = |bridge: EstimandBridge| ExperimentalPrior {
            prior_id: "prior-geo".into(),
            experiment_id: "geo-2024".into(),
            estimand_bridge: Some(bridge),
            mapping_uncertainty: None,
            source_dependence: None,
        };
        // Target dose 25 minus experiment dose 10 is +15, not -15.
        let error =
            contract.import_experimental_prior(&prior(translation(20.0, -15.0))).unwrap_err();
        refused(&error, "bridge_binding_mismatch", "scientific_bridge.estimand_spend_inconsistent");
        let mut units = translation(20.0, 15.0);
        units.target.dose_units = "eur_per_capita".into();
        let error = contract.import_experimental_prior(&prior(units)).unwrap_err();
        refused(&error, "bridge_binding_mismatch", "scientific_bridge.estimand_units_mismatch");
        let mut spend_units = translation(20.0, 15.0);
        spend_units.spend_units = "eur_per_capita".into();
        let error = contract.import_experimental_prior(&prior(spend_units)).unwrap_err();
        refused(&error, "bridge_binding_mismatch", "scientific_bridge.estimand_units_mismatch");
        assert!(contract.estimand_bridge.is_none());
        contract.import_experimental_prior(&prior(translation(20.0, 15.0))).unwrap();
    }

    #[test]
    fn intercept_slope_covariance_enters_the_delta_method_variance() {
        let mut contract = preference(
            BridgeMapping::Affine { intercept: 0.0, slope: 2.0 },
            vec![linkage("link", "holds")],
        );
        contract.source_dependence =
            Some(SourceMappingDependence::Independent { evidence: "disjoint-rows".into() });
        // Negatively correlated intercept and slope errors (a regression line pivoting
        // near the source mean) enter through the covariance term:
        // Var(a) + mu^2 Var(b) + 2 mu Cov(a, b) = 0.04 + 25 * 0.04 - 2 * 5 * 0.04 = 0.64.
        contract.mapping_uncertainty = Some(MappingUncertainty {
            id: "calibration".into(),
            intercept_variance: 0.04,
            slope_variance: 0.04,
            intercept_slope_covariance: -0.04,
        });
        let got = contract.propagate_affine_uncertainty(5.0, 0.0).unwrap();
        assert!((got.variance - 0.64).abs() < 1e-12, "{}", got.variance);
        // Ignoring the covariance would give 0.04 + 1.0 = 1.04.
        contract.mapping_uncertainty.as_mut().unwrap().intercept_slope_covariance = 0.0;
        let independent = contract.propagate_affine_uncertainty(5.0, 0.0).unwrap();
        assert!((independent.variance - 1.04).abs() < 1e-12);
        // Independent factors, exact product variance from raw moments:
        // E[b^2] E[M^2] - (b mu)^2 = (4 + 0.04) (25 + 0.01) - 100 = 1.0404.
        contract.mapping_uncertainty.as_mut().unwrap().intercept_variance = 0.0;
        let product = contract.propagate_affine_uncertainty(5.0, 0.01).unwrap();
        assert!((product.variance - 1.0404).abs() < 1e-12, "{}", product.variance);
        // An infeasible covariance is refused.
        contract.mapping_uncertainty = Some(MappingUncertainty {
            id: "calibration".into(),
            intercept_variance: 0.04,
            slope_variance: 0.04,
            intercept_slope_covariance: 0.05,
        });
        let error = contract.propagate_affine_uncertainty(5.0, 0.0).unwrap_err();
        refused(
            &error,
            "scientific_bridge_invalid",
            "scientific_bridge.mapping_uncertainty_invalid",
        );
    }

    #[test]
    fn overflow_and_non_finite_means_have_their_own_refusals() {
        let error = nonlinear_expectation(
            NonlinearMap::Square,
            &AnalyticLaw::SecondMoment { mean: f64::NAN, variance: 1.0 },
        )
        .unwrap_err();
        refused(&error, "scientific_bridge_invalid", "scientific_bridge.source_moment_not_finite");
        let error = nonlinear_expectation(
            NonlinearMap::Exponential,
            &AnalyticLaw::Normal { mean: 1000.0, variance: 1.0 },
        )
        .unwrap_err();
        refused(&error, "scientific_bridge_invalid", "scientific_bridge.propagation_overflow");
        let error =
            support_image(SupportMap::Exponential, ValueInterval { low: 0.0, high: 1000.0 })
                .unwrap_err();
        refused(&error, "scientific_bridge_invalid", "scientific_bridge.propagation_overflow");
        let error =
            support_image(SupportMap::Square, ValueInterval { low: 0.0, high: 1e200 }).unwrap_err();
        refused(&error, "scientific_bridge_invalid", "scientific_bridge.propagation_overflow");
        // A reversed interval is still a malformed mapping.
        let error =
            support_image(SupportMap::Identity, ValueInterval { low: 1.0, high: 0.0 }).unwrap_err();
        refused(&error, "scientific_bridge_invalid", "scientific_bridge.malformed_mapping");
    }

    #[test]
    fn a_derived_claim_retains_mapping_uncertainty_and_source_dependence() {
        let mut contract = preference(
            BridgeMapping::Affine { intercept: 0.0, slope: 2.0 },
            vec![linkage("hypothetical-bias", "stated scales realized")],
        );
        contract.mapping_uncertainty = Some(uncertainty(0.0625));
        contract.source_dependence =
            Some(SourceMappingDependence::Independent { evidence: "disjoint-rows".into() });
        contract.estimand_bridge = Some(translation(20.0, 15.0));
        let claim = point_claim(0.5);
        let quantity = contract.source.quantity.clone();
        let source = BridgeSource { claim: &claim, quantity: &quantity };
        let bridged = derive_bridged_claim(&contract, &source).unwrap();
        assert_eq!(bridged.value.to_bits(), 1.0_f64.to_bits());
        let evidence = bridged.claim.evidence.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        assert!(evidence.contains(&"mapping_uncertainty:slope-se"));
        assert!(evidence.contains(&"source_dependence:independent:disjoint-rows"));
        assert!(evidence.contains(&"estimand_bridge:geo-to-panel"));
        let retained = bridged.receipt.retained.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        assert!(retained.contains(&"mapping_uncertainty"));
        assert!(retained.contains(&"source_dependence"));
        assert!(retained.contains(&"estimand_bridge"));
        assert!(
            bridged
                .receipt
                .unresolved
                .iter()
                .any(|item| item.as_ref() == "assumption:hypothetical-bias")
        );
        let SlotAvailability::Available(slot) = &bridged.claim.reasoning.uncertainty else {
            panic!("mapping uncertainty was dropped");
        };
        assert!(slot.components.iter().any(|item| {
            item.target.as_ref() == "mapping_variance:slope-se"
                && !item.omitted
                && item.source == crate::UncertaintySource::Sampling
        }));
        assert!(
            slot.components
                .iter()
                .any(|item| item.target.as_ref() == "source_dependence" && !item.omitted)
        );
        assert!(matches!(
            bridged.claim.reasoning.support,
            SlotAvailability::Unavailable { ref reason }
                if reason.as_ref() == "bridge_support_not_established"
        ));
        assert_eq!(
            bridged.contract.mapping_uncertainty.as_ref().map(|item| item.id.as_str()),
            Some("slope-se")
        );
        assert!(bridged.contract.source_dependence.is_some());
    }
}

#[cfg(test)]
mod joint_tests {
    use super::super::super::kind::{AssumptionKind, BridgeKind, QuantityClass, ValueKind};
    use super::super::super::mapping::BridgeMapping;
    use super::super::{BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeScope};
    use super::{
        JointLaw, MappingUncertainty, PropagationMethod, SourceMappingDependence, propagate_joint,
        propagate_nonlinear,
    };
    use crate::{QuantityRole, ScientificQuantity};

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

    fn contract(
        intercept: f64,
        slope: f64,
        uncertainty: MappingUncertainty,
        dependence: Option<SourceMappingDependence>,
    ) -> BridgeContract {
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
            mapping: BridgeMapping::Affine { intercept, slope },
            assumptions: vec![BridgeAssumption {
                id: "hypothetical-bias".into(),
                kind: AssumptionKind::BehavioralLinkage,
                statement: "stated scales realized".into(),
                evidence: Some("validation-panel-2023".into()),
            }],
            transport: None,
            mapping_uncertainty: Some(uncertainty),
            source_dependence: dependence,
            estimand_bridge: None,
        }
    }

    fn slope_only(slope_variance: f64) -> MappingUncertainty {
        MappingUncertainty {
            id: "slope-se".into(),
            intercept_variance: 0.0,
            slope_variance,
            intercept_slope_covariance: 0.0,
        }
    }

    fn shared(covariance: f64) -> Option<SourceMappingDependence> {
        Some(SourceMappingDependence::SlopeCovariance { covariance, evidence: "same-panel".into() })
    }

    fn relative(value: f64, reference: f64) -> f64 {
        (value - reference).abs() / reference.abs()
    }

    /// `exp(z0 + z1 + z2)` over a correlated normal vector is lognormal with
    /// `L = z0 + z1 + z2 ~ N(m, v)`: a closed-form reference.
    fn lognormal_law() -> (JointLaw, f64, f64, f64, f64) {
        let covariance = vec![0.01, 0.0025, 0.0, 0.0025, 0.0225, -0.005, 0.0, -0.005, 0.015625];
        let law = JointLaw::new(
            vec!["a".into(), "b".into(), "c".into()],
            vec![0.1, -0.2, 0.3],
            covariance,
        )
        .unwrap();
        let v: f64 = 0.01 + 0.0225 + 0.015625 + 2.0 * (0.0025 + 0.0 - 0.005);
        let m: f64 = 0.2;
        let mean = (m + 0.5 * v).exp();
        let variance = (v.exp() - 1.0) * (2.0 * m + v).exp();
        (law, m, v, mean, variance)
    }

    fn exp_sum(point: &[f64]) -> f64 {
        point.iter().sum::<f64>().exp()
    }

    #[test]
    fn the_second_order_delta_reproduces_the_declared_affine_moments_exactly() {
        let uncertainty = MappingUncertainty {
            id: "line".into(),
            intercept_variance: 0.04,
            slope_variance: 0.0625,
            intercept_slope_covariance: -0.02,
        };
        let contract = contract(0.1, 2.0, uncertainty, shared(0.01));
        let analytic = contract.propagate_affine_uncertainty(4.0, 0.25).unwrap();
        let delta = propagate_nonlinear(
            &contract,
            4.0,
            0.25,
            &|a, b, m| a + b * m,
            PropagationMethod::SecondOrderDelta,
        )
        .unwrap();
        assert!(relative(delta.propagation.mean, analytic.mean) < 1e-7);
        assert!(relative(delta.propagation.variance, analytic.variance) < 1e-6);
        assert!(delta.propagation.linearized_variance.unwrap() < delta.propagation.variance);
        assert_eq!(delta.assumptions.len(), 1);
        assert_eq!(delta.dependence, contract.source_dependence.clone().unwrap());
    }

    #[test]
    fn a_nonlinear_expectation_differs_from_the_map_of_the_mean_on_every_method() {
        let (law, m, v, mean, variance) = lognormal_law();
        let plug_in = m.exp();
        assert!(relative(mean, plug_in) > 0.01);
        let delta = propagate_joint(&law, &exp_sum, PropagationMethod::SecondOrderDelta).unwrap();
        let sigma = propagate_joint(&law, &exp_sum, PropagationMethod::Unscented).unwrap();
        let simulated = propagate_joint(
            &law,
            &exp_sum,
            PropagationMethod::Simulation { draws: 300_000, seed: 17 },
        )
        .unwrap();
        assert!((delta.plug_in - plug_in).abs() < 1e-12);
        for result in [delta, sigma] {
            assert!(result.bias() > 0.0);
            assert!(relative(result.mean, mean) < 2e-3, "{result:?} vs {mean}");
            assert!(relative(result.mean, plug_in) > 0.01);
        }
        // The first-order variance misses the curvature term the second order adds.
        let linear_only = delta.linearized_variance.unwrap();
        assert!(relative(delta.variance, variance) < relative(linear_only, variance));
        assert!(relative(delta.variance, variance) < 2.0 * v);
        assert!(relative(sigma.variance, variance) < 0.08, "{sigma:?} vs {variance}");
        let mean_error = (simulated.mean - mean).abs();
        let variance_error = (simulated.variance - variance).abs();
        assert!(mean_error < 5.0 * simulated.mean_standard_error.unwrap());
        assert!(variance_error < 5.0 * simulated.variance_standard_error.unwrap());
    }

    #[test]
    fn simulation_is_reproducible_and_responds_to_the_seed() {
        let (law, ..) = lognormal_law();
        let run = |seed| {
            propagate_joint(&law, &exp_sum, PropagationMethod::Simulation { draws: 2_000, seed })
                .unwrap()
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5).mean.to_bits(), run(6).mean.to_bits());
    }

    #[test]
    fn joint_source_mapping_dependence_moves_a_nonlinear_variance_as_simulation_shows() {
        // g = exp(slope * source) with no intercept: its variance depends on the
        // sign of the slope-source covariance, which a marginal treatment drops.
        let g = |_: f64, b: f64, m: f64| (b * m).exp();
        let build = |dependence| contract(0.0, 0.5, slope_only(0.0025), dependence);
        let run = |dependence, method| {
            propagate_nonlinear(&build(dependence), 1.0, 0.01, &g, method).unwrap().propagation
        };
        let simulation = PropagationMethod::Simulation { draws: 400_000, seed: 99 };
        let mut variances = Vec::new();
        for dependence in [
            shared(0.004),
            Some(SourceMappingDependence::Independent { evidence: "disjoint-rows".into() }),
            shared(-0.004),
        ] {
            let reference = run(dependence.clone(), simulation);
            let delta = run(dependence.clone(), PropagationMethod::SecondOrderDelta);
            let sigma = run(dependence, PropagationMethod::Unscented);
            assert!(relative(delta.variance, reference.variance) < 0.05, "{delta:?} {reference:?}");
            assert!(relative(sigma.variance, reference.variance) < 0.05, "{sigma:?} {reference:?}");
            assert!(relative(delta.mean, reference.mean) < 5e-3);
            assert!(relative(sigma.mean, reference.mean) < 5e-3);
            variances.push(reference.variance);
        }
        // Positive covariance reinforces the product, negative cancels it.
        assert!(variances[0] > variances[1] && variances[1] > variances[2]);
        assert!(variances[0] > 2.0 * variances[2]);
    }

    #[test]
    fn an_infeasible_joint_covariance_or_missing_declaration_is_refused() {
        let names = || vec!["a".to_owned(), "b".to_owned()];
        let error =
            JointLaw::new(names(), vec![0.0, 0.0], vec![1.0, 2.0, 2.0, 1.0]).unwrap_err();
        assert_eq!(error.detail, "scientific_bridge.mapping_uncertainty_invalid");
        let error =
            JointLaw::new(names(), vec![0.0, 0.0], vec![1.0, 0.5, 0.0, 1.0]).unwrap_err();
        assert_eq!(error.detail, "scientific_bridge.mapping_uncertainty_invalid");
        // Each pairwise covariance is feasible but the three together are not.
        let uncertainty = MappingUncertainty {
            id: "line".into(),
            intercept_variance: 1.0,
            slope_variance: 1.0,
            intercept_slope_covariance: 0.99,
        };
        let infeasible = contract(0.0, 1.0, uncertainty, shared(0.99));
        let error = propagate_nonlinear(
            &infeasible,
            0.0,
            1.0,
            &|a, b, m| a + b * m,
            PropagationMethod::Unscented,
        )
        .unwrap_err();
        assert_eq!(error.detail, "scientific_bridge.mapping_uncertainty_invalid");
        let none = contract(0.0, 1.0, slope_only(1.0), None);
        let error = propagate_nonlinear(
            &none,
            0.0,
            1.0,
            &|a, b, m| a + b * m,
            PropagationMethod::Unscented,
        )
        .unwrap_err();
        assert_eq!(error.detail, "scientific_bridge.source_dependence_missing");
        let (law, ..) = lognormal_law();
        let error = propagate_joint(&law, &exp_sum, PropagationMethod::Simulation { draws: 3, seed: 1 })
            .unwrap_err();
        assert_eq!(error.detail, "scientific_bridge.malformed_mapping");
        let error = propagate_joint(&law, &|_| f64::NAN, PropagationMethod::Unscented).unwrap_err();
        assert_eq!(error.detail, "scientific_bridge.propagation_overflow");
    }
}
