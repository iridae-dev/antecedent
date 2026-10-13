//! The mappings that carry one point value across a bridge.
//!
//! * [`BridgeMapping::Identity`] leaves the value unchanged; it is licensed only
//!   where the declared assumptions make the target quantity *equal* to the
//!   source (a predictive quantity under identification, a stated preference
//!   under a declared behavioral linkage).
//! * [`BridgeMapping::Affine`] is a declared `intercept + slope * value` (a
//!   unit-of-intervention contrast, a linear utility, a declared calibration).
//!   Applied to a mean it is exact for the mean of the mapped quantity: the
//!   expectation of an affine function is the affine function of the
//!   expectation. A nonlinear map is not carried, because the mapped mean of a
//!   law is not the map of its mean.
//! * [`BridgeMapping::TrialLevelLinear`] is the trial-level surrogate
//!   relationship: the terminal-outcome effect is regressed on the surrogate
//!   effect across validation trials by ordinary least squares, and the fit is
//!   applied only inside the range of surrogate effects the trials covered,
//!   and only when the declared minimum number of trials and trial-level
//!   coefficient of determination are met.
//! * [`BridgeMapping::MeasurementCorrection`] is the closed-form inversion (or
//!   forward map) of a binary-misclassification measurement model on a
//!   prevalence or risk: `latent = (observed + specificity - 1) / (sensitivity
//!   + specificity - 1)` and its forward map, for the model's declared rates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::identity::CanonicalWriter;
use antecedent_core::measurement_model::{
    ErrorStructure, MeasurementModel, MeasurementParameters, MeasurementRoute,
};

use super::refusal::{
    BridgeRefusal, degenerate_validation, differential_error, malformed_mapping,
    measurement_route_not_supported, non_finite_value, outside_attainable_range,
    outside_validated_range, validation_r_squared_too_low, validation_trials_too_few,
    youden_too_small,
};

/// Fewest validation trials a trial-level mapping may declare.
pub const MIN_VALIDATION_TRIALS: u32 = 3;
/// Most validation trials one mapping may carry.
pub const MAX_VALIDATION_TRIALS: usize = 4096;

/// The mapping variant, without its numbers.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum MappingKind {
    /// The value is unchanged.
    Identity,
    /// A declared affine map.
    Affine,
    /// A trial-level least-squares fit.
    TrialLevelLinear,
    /// A binary-misclassification measurement correction.
    MeasurementCorrection,
}

impl MappingKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Affine => "affine",
            Self::TrialLevelLinear => "trial_level_linear",
            Self::MeasurementCorrection => "measurement_correction",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Identity, Self::Affine, Self::TrialLevelLinear, Self::MeasurementCorrection]
            .into_iter()
            .find(|kind| kind.as_str() == name)
    }
}

/// Which way a measurement correction runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeasurementDirection {
    /// From the recorded prevalence to the latent prevalence.
    MeasuredToLatent,
    /// From the latent prevalence to the prevalence the indicator records.
    LatentToMeasured,
}

/// The effects of one validation trial on the surrogate and on the terminal
/// outcome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrialEffectPair {
    /// Treatment effect on the surrogate endpoint in this trial.
    pub surrogate_effect: f64,
    /// Treatment effect on the terminal outcome in this trial.
    pub terminal_effect: f64,
}

/// The validation evidence of a trial-level mapping.
#[derive(Clone, Debug, PartialEq)]
pub struct TrialLevelValidation {
    /// One pair per validation trial. Order does not matter.
    pub trials: Vec<TrialEffectPair>,
    /// Fewest trials the mapping accepts (at least [`MIN_VALIDATION_TRIALS`]).
    pub min_trials: u32,
    /// Smallest trial-level coefficient of determination the mapping accepts,
    /// in `[0, 1]`.
    pub min_r_squared: f64,
}

/// The ordinary least-squares fit of terminal effect on surrogate effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrialLevelFit {
    /// Intercept of the terminal effect at zero surrogate effect.
    pub intercept: f64,
    /// Slope of the terminal effect per unit surrogate effect.
    pub slope: f64,
    /// Trial-level coefficient of determination.
    pub r_squared: f64,
    /// Trials fitted.
    pub trials: u32,
    /// Smallest surrogate effect among the trials.
    pub surrogate_min: f64,
    /// Largest surrogate effect among the trials.
    pub surrogate_max: f64,
}

/// A mapping from a source value to a target value.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum BridgeMapping {
    /// The value is unchanged.
    Identity,
    /// `intercept + slope * value`.
    Affine {
        /// Value of the target at a zero source value.
        intercept: f64,
        /// Target change per unit of the source.
        slope: f64,
    },
    /// Trial-level least-squares surrogate relationship.
    TrialLevelLinear(TrialLevelValidation),
    /// Binary-misclassification correction of a prevalence or risk, by the
    /// rates of the measurement model.
    MeasurementCorrection(Box<MeasurementModel>),
}

/// The result of applying a mapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AppliedMapping {
    /// The mapped value.
    pub value: f64,
    /// The trial-level fit, when the mapping fitted one.
    pub fit: Option<TrialLevelFit>,
}

/// Fit the trial-level relationship and check the declared minimums.
///
/// The fit is centered ordinary least squares of terminal effect on surrogate
/// effect, with `r_squared = Sxy^2 / (Sxx * Syy)`.
///
/// # Errors
/// A malformed declaration, fewer trials than declared, a surrogate effect
/// that does not vary, a terminal effect that does not vary, or a trial-level
/// coefficient of determination below the declared minimum.
pub fn fit_trial_level_linear(
    validation: &TrialLevelValidation,
) -> Result<TrialLevelFit, BridgeRefusal> {
    check_validation_declaration(validation)?;
    let n = validation.trials.len();
    if n < usize::try_from(validation.min_trials).unwrap_or(usize::MAX) {
        return Err(validation_trials_too_few(validation.min_trials, n));
    }
    // Sum in a canonical order so a permutation of the trials gives the same bits.
    let mut sorted = validation.trials.clone();
    sorted.sort_by(|a, b| {
        a.surrogate_effect
            .total_cmp(&b.surrogate_effect)
            .then(a.terminal_effect.total_cmp(&b.terminal_effect))
    });
    let count = n as f64;
    let mean_x = sorted.iter().map(|t| t.surrogate_effect).sum::<f64>() / count;
    let mean_y = sorted.iter().map(|t| t.terminal_effect).sum::<f64>() / count;
    let (mut sxx, mut sxy, mut syy) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for trial in &sorted {
        let dx = trial.surrogate_effect - mean_x;
        let dy = trial.terminal_effect - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
        lo = lo.min(trial.surrogate_effect);
        hi = hi.max(trial.surrogate_effect);
    }
    if sxx.is_nan() || sxx <= 0.0 {
        return Err(degenerate_validation(
            "varying surrogate effects",
            "constant surrogate effect",
            "the validation trials must differ in their effect on the surrogate for the relationship to be estimable",
        ));
    }
    if syy.is_nan() || syy <= 0.0 {
        return Err(degenerate_validation(
            "varying terminal effects",
            "constant terminal effect",
            "the validation trials must differ in their effect on the terminal outcome for the trial-level association to be assessed",
        ));
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let r_squared = (sxy * sxy / (sxx * syy)).min(1.0);
    if r_squared < validation.min_r_squared {
        return Err(validation_r_squared_too_low(validation.min_r_squared, r_squared));
    }
    Ok(TrialLevelFit {
        intercept,
        slope,
        r_squared,
        trials: u32::try_from(n).unwrap_or(u32::MAX),
        surrogate_min: lo,
        surrogate_max: hi,
    })
}

fn check_validation_declaration(validation: &TrialLevelValidation) -> Result<(), BridgeRefusal> {
    if validation.min_trials < MIN_VALIDATION_TRIALS {
        return Err(malformed_mapping(
            "min_trials",
            "declare at least three validation trials: two points always fit a line",
        ));
    }
    if !(validation.min_r_squared.is_finite() && (0.0..=1.0).contains(&validation.min_r_squared)) {
        return Err(malformed_mapping(
            "min_r_squared",
            "declare the smallest trial-level coefficient of determination in [0, 1]",
        ));
    }
    if validation.trials.len() > MAX_VALIDATION_TRIALS {
        return Err(malformed_mapping(
            "trials",
            "carry at most the declared bound of validation trials",
        ));
    }
    if validation
        .trials
        .iter()
        .any(|t| !(t.surrogate_effect.is_finite() && t.terminal_effect.is_finite()))
    {
        return Err(non_finite_value("trials", "validation effects must be finite numbers"));
    }
    Ok(())
}

/// The shared rates of a non-differential binary measurement model.
fn binary_rates(model: &MeasurementModel) -> Result<(f64, f64, f64), BridgeRefusal> {
    model.validate().map_err(|error| crate::measurement::model_refusal(&error))?;
    if model.route != MeasurementRoute::BinaryMisclassification {
        return Err(measurement_route_not_supported(
            MeasurementRoute::BinaryMisclassification.as_str(),
            model.route.as_str(),
        ));
    }
    if model.structure != ErrorStructure::NonDifferential {
        return Err(differential_error());
    }
    let MeasurementParameters::Binary { rates, min_youden } = &model.parameters else {
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
    Ok((shared.sensitivity, shared.specificity, *min_youden))
}

impl BridgeMapping {
    /// The variant.
    #[must_use]
    pub const fn kind(&self) -> MappingKind {
        match self {
            Self::Identity => MappingKind::Identity,
            Self::Affine { .. } => MappingKind::Affine,
            Self::TrialLevelLinear(_) => MappingKind::TrialLevelLinear,
            Self::MeasurementCorrection(_) => MappingKind::MeasurementCorrection,
        }
    }

    /// Check the declaration is well formed, without a value.
    ///
    /// # Errors
    /// A non-finite affine coefficient, a malformed validation declaration, or
    /// a measurement model that is not well formed.
    pub fn validate(&self) -> Result<(), BridgeRefusal> {
        match self {
            Self::Identity => Ok(()),
            Self::Affine { intercept, slope } => {
                if intercept.is_finite() && slope.is_finite() {
                    Ok(())
                } else {
                    Err(non_finite_value("affine", "declare a finite intercept and slope"))
                }
            }
            Self::TrialLevelLinear(validation) => check_validation_declaration(validation),
            Self::MeasurementCorrection(model) => {
                model.validate().map_err(|error| crate::measurement::model_refusal(&error))
            }
        }
    }

    /// Apply the mapping to one finite value.
    ///
    /// `direction` is used by the measurement correction only.
    ///
    /// # Errors
    /// A non-finite value; a trial-level fit that does not meet its declared
    /// minimums; a surrogate effect outside the range the validation trials
    /// covered; a recorded prevalence outside the attainable range
    /// `[1 - specificity, sensitivity]`; a latent prevalence outside `[0, 1]`;
    /// Youden's index at or below the model's minimum.
    pub fn apply(
        &self,
        value: f64,
        direction: MeasurementDirection,
    ) -> Result<AppliedMapping, BridgeRefusal> {
        if !value.is_finite() {
            return Err(non_finite_value("value", "bridge a finite point value"));
        }
        let (mapped, fit) = match self {
            Self::Identity => (value, None),
            Self::Affine { intercept, slope } => (intercept + slope * value, None),
            Self::TrialLevelLinear(validation) => {
                let fit = fit_trial_level_linear(validation)?;
                if value < fit.surrogate_min || value > fit.surrogate_max {
                    return Err(outside_validated_range(
                        fit.surrogate_min,
                        fit.surrogate_max,
                        value,
                    ));
                }
                (fit.intercept + fit.slope * value, Some(fit))
            }
            Self::MeasurementCorrection(model) => {
                (apply_correction(model, value, direction)?, None)
            }
        };
        if !mapped.is_finite() {
            return Err(non_finite_value("value", "the mapped value is not finite"));
        }
        Ok(AppliedMapping { value: mapped, fit })
    }

    /// Check the mapping's own evidence is sufficient, without a value: a
    /// trial-level fit meets its declared minimums, and a measurement
    /// correction has one shared pair of rates and an identified Youden index.
    ///
    /// # Errors
    /// The refusals of [`fit_trial_level_linear`] or of the correction's rates.
    pub fn check_ready(&self) -> Result<(), BridgeRefusal> {
        match self {
            Self::Identity | Self::Affine { .. } => Ok(()),
            Self::TrialLevelLinear(validation) => fit_trial_level_linear(validation).map(|_| ()),
            Self::MeasurementCorrection(model) => {
                let (sensitivity, specificity, min_youden) = binary_rates(model)?;
                identified_youden(sensitivity, specificity, min_youden).map(|_| ())
            }
        }
    }

    /// Write the mapping, with every number as its bit pattern, to a
    /// canonical identity stream. A measurement correction enters as the
    /// model's own six-part identity.
    pub fn write_canonical(&self, out: &mut CanonicalWriter) {
        match self {
            Self::Identity => {
                out.word(1);
            }
            Self::Affine { intercept, slope } => {
                out.word(2).real(*intercept).real(*slope);
            }
            Self::TrialLevelLinear(validation) => {
                out.word(3)
                    .word(u64::from(validation.min_trials))
                    .real(validation.min_r_squared)
                    .word(validation.trials.len() as u64);
                // Trial order carries no meaning: write the pairs sorted by bits.
                let mut pairs: Vec<(u64, u64)> = validation
                    .trials
                    .iter()
                    .map(|t| (t.surrogate_effect.to_bits(), t.terminal_effect.to_bits()))
                    .collect();
                pairs.sort_unstable();
                for (surrogate, terminal) in pairs {
                    out.word(surrogate).word(terminal);
                }
            }
            Self::MeasurementCorrection(model) => {
                out.word(4).bytes(model.identities().model.as_bytes());
            }
        }
    }
}

/// Youden's index of the declared rates, when it exceeds the declared minimum.
fn identified_youden(
    sensitivity: f64,
    specificity: f64,
    min_youden: f64,
) -> Result<f64, BridgeRefusal> {
    let youden = sensitivity + specificity - 1.0;
    if youden.is_nan() || youden <= min_youden {
        return Err(youden_too_small(youden, min_youden));
    }
    Ok(youden)
}

fn apply_correction(
    model: &MeasurementModel,
    value: f64,
    direction: MeasurementDirection,
) -> Result<f64, BridgeRefusal> {
    let (sensitivity, specificity, min_youden) = binary_rates(model)?;
    let youden = identified_youden(sensitivity, specificity, min_youden)?;
    match direction {
        MeasurementDirection::MeasuredToLatent => {
            let lower = 1.0 - specificity;
            if value < lower || value > sensitivity {
                return Err(outside_attainable_range(
                    format!("[{lower}, {sensitivity}]"),
                    value,
                    "a recorded prevalence outside [1 - specificity, sensitivity] cannot arise from any latent prevalence under the declared rates; the rates or the recorded value are wrong",
                ));
            }
            Ok((value + specificity - 1.0) / youden)
        }
        MeasurementDirection::LatentToMeasured => {
            if !(0.0..=1.0).contains(&value) {
                return Err(outside_attainable_range(
                    "[0, 1]".to_owned(),
                    value,
                    "a latent prevalence or risk is a probability in [0, 1]",
                ));
            }
            Ok(sensitivity * value + (1.0 - specificity) * (1.0 - value))
        }
    }
}
