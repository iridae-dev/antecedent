//! Estimate conversions for the transportability and randomized-interference
//! wires.
//!
//! Re-exported from [`crate::transport_interference_wire`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::{
    InterferenceEstimate, TransportEffectEstimate, TransportOverlapDiagnostic,
    TransportOverlapReport,
};
use antecedent_io_core::IoError;
use antecedent_io_core::transport_interference_wire::{
    ExposureProbabilityMethodWire, InterferenceEstimateWire, RandomizationContrastWire,
    TransportEffectEstimateWire, TransportOverlapDiagnosticWire,
};
use antecedent_stats::{ExposureProbabilityMethod, RandomizationContrast};

/// Encode an interference estimate.
#[must_use]
pub fn interference_estimate_to_wire(value: &InterferenceEstimate) -> InterferenceEstimateWire {
    InterferenceEstimateWire {
        contrast: contrast_to_wire(value.contrast),
        from_probability_method: probability_method_to_wire(value.from_probability_method),
        to_probability_method: probability_method_to_wire(value.to_probability_method),
        minimum_exposure_probability: value.minimum_exposure_probability,
    }
}

/// Decode an interference estimate without validating its fields.
///
/// Use [`try_interference_estimate_from_wire`] for untrusted wire data.
#[must_use]
pub fn interference_estimate_from_wire(value: &InterferenceEstimateWire) -> InterferenceEstimate {
    InterferenceEstimate {
        contrast: contrast_from_wire(value.contrast),
        from_probability_method: probability_method_from_wire(value.from_probability_method),
        to_probability_method: probability_method_from_wire(value.to_probability_method),
        minimum_exposure_probability: value.minimum_exposure_probability,
    }
}

/// Decode and validate an interference estimate.
///
/// # Errors
/// Returns a conversion error if the wire estimate has invalid numerical fields or methods.
pub fn try_interference_estimate_from_wire(
    value: &InterferenceEstimateWire,
) -> Result<InterferenceEstimate, IoError> {
    antecedent_io_core::causal_artifact::validate_interference_estimate(value)?;
    Ok(interference_estimate_from_wire(value))
}

/// Encode a trial-to-target estimate.
///
/// # Errors
///
/// An extreme-weight count does not fit `u64`.
pub fn transport_effect_to_wire(
    value: &TransportEffectEstimate,
) -> Result<TransportEffectEstimateWire, IoError> {
    Ok(TransportEffectEstimateWire {
        ipw: value.ipw,
        aipw: value.aipw,
        selection_overlap: diagnostic_to_wire(value.overlap.selection)?,
        treatment_overlap: diagnostic_to_wire(value.overlap.treatment)?,
    })
}

/// Decode a trial-to-target estimate.
///
/// # Errors
///
/// An extreme-weight count does not fit `usize`.
pub fn transport_effect_from_wire(
    value: &TransportEffectEstimateWire,
) -> Result<TransportEffectEstimate, IoError> {
    if !value.ipw.is_finite() || value.aipw.is_some_and(|v| !v.is_finite()) {
        return Err(IoError::Convert("transport effect estimates must be finite".into()));
    }
    Ok(TransportEffectEstimate {
        ipw: value.ipw,
        aipw: value.aipw,
        overlap: TransportOverlapReport {
            selection: diagnostic_from_wire(value.selection_overlap)?,
            treatment: diagnostic_from_wire(value.treatment_overlap)?,
        },
    })
}

const fn probability_method_to_wire(
    value: ExposureProbabilityMethod,
) -> ExposureProbabilityMethodWire {
    match value {
        ExposureProbabilityMethod::Exact => ExposureProbabilityMethodWire::Exact,
        ExposureProbabilityMethod::MonteCarlo { draws, seed } => {
            ExposureProbabilityMethodWire::MonteCarlo { draws, seed }
        }
        ExposureProbabilityMethod::SuppliedKnown => ExposureProbabilityMethodWire::SuppliedKnown,
        ExposureProbabilityMethod::SuppliedExternallyEstimated => {
            ExposureProbabilityMethodWire::SuppliedExternallyEstimated
        }
    }
}

const fn probability_method_from_wire(
    value: ExposureProbabilityMethodWire,
) -> ExposureProbabilityMethod {
    match value {
        ExposureProbabilityMethodWire::Exact => ExposureProbabilityMethod::Exact,
        ExposureProbabilityMethodWire::MonteCarlo { draws, seed } => {
            ExposureProbabilityMethod::MonteCarlo { draws, seed }
        }
        ExposureProbabilityMethodWire::SuppliedKnown => ExposureProbabilityMethod::SuppliedKnown,
        ExposureProbabilityMethodWire::SuppliedExternallyEstimated => {
            ExposureProbabilityMethod::SuppliedExternallyEstimated
        }
    }
}

const fn contrast_to_wire(value: RandomizationContrast) -> RandomizationContrastWire {
    RandomizationContrastWire {
        horvitz_thompson: value.horvitz_thompson,
        hajek: value.hajek,
        conservative_variance: value.conservative_variance,
    }
}

const fn contrast_from_wire(value: RandomizationContrastWire) -> RandomizationContrast {
    RandomizationContrast {
        horvitz_thompson: value.horvitz_thompson,
        hajek: value.hajek,
        conservative_variance: value.conservative_variance,
    }
}

fn diagnostic_to_wire(
    value: TransportOverlapDiagnostic,
) -> Result<TransportOverlapDiagnosticWire, IoError> {
    Ok(TransportOverlapDiagnosticWire {
        probability_min: value.probability_min,
        probability_max: value.probability_max,
        effective_sample_size: value.effective_sample_size,
        extreme_weight_count: u64::try_from(value.extreme_weight_count)
            .map_err(|_| IoError::TooLarge)?,
    })
}

fn diagnostic_from_wire(
    value: TransportOverlapDiagnosticWire,
) -> Result<TransportOverlapDiagnostic, IoError> {
    if !value.probability_min.is_finite()
        || !value.probability_max.is_finite()
        || !(0.0..=1.0).contains(&value.probability_min)
        || !(0.0..=1.0).contains(&value.probability_max)
        || value.probability_min > value.probability_max
        || !value.effective_sample_size.is_finite()
        || value.effective_sample_size < 0.0
    {
        return Err(IoError::Convert("invalid transport overlap diagnostic".into()));
    }
    Ok(TransportOverlapDiagnostic {
        probability_min: value.probability_min,
        probability_max: value.probability_max,
        effective_sample_size: value.effective_sample_size,
        extreme_weight_count: usize::try_from(value.extreme_weight_count)
            .map_err(|_| IoError::TooLarge)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_results_round_trip() {
        let interference = InterferenceEstimate {
            contrast: RandomizationContrast {
                horvitz_thompson: 1.0,
                hajek: 0.9,
                conservative_variance: 0.2,
            },
            from_probability_method: ExposureProbabilityMethod::Exact,
            to_probability_method: ExposureProbabilityMethod::MonteCarlo { draws: 50, seed: 7 },
            minimum_exposure_probability: 0.1,
        };
        let back = interference_estimate_from_wire(&interference_estimate_to_wire(&interference));
        assert_eq!(back, interference);

        let transport = TransportEffectEstimate {
            ipw: 2.0,
            aipw: Some(1.9),
            overlap: TransportOverlapReport {
                selection: TransportOverlapDiagnostic {
                    probability_min: 0.2,
                    probability_max: 0.8,
                    effective_sample_size: 10.0,
                    extreme_weight_count: 1,
                },
                treatment: TransportOverlapDiagnostic {
                    probability_min: 0.4,
                    probability_max: 0.6,
                    effective_sample_size: 12.0,
                    extreme_weight_count: 0,
                },
            },
        };
        let wire = transport_effect_to_wire(&transport).unwrap();
        assert_eq!(transport_effect_from_wire(&wire).unwrap(), transport);
    }

    #[test]
    fn transport_effect_rejects_invalid_points_and_overlap() {
        let valid = TransportEffectEstimateWire {
            ipw: 1.0,
            aipw: Some(2.0),
            selection_overlap: TransportOverlapDiagnosticWire {
                probability_min: 0.0,
                probability_max: 1.0,
                effective_sample_size: 0.0,
                extreme_weight_count: 0,
            },
            treatment_overlap: TransportOverlapDiagnosticWire {
                probability_min: 0.2,
                probability_max: 0.8,
                effective_sample_size: 1.0,
                extreme_weight_count: 0,
            },
        };
        assert!(transport_effect_from_wire(&valid).is_ok());
        let mut invalid = valid.clone();
        invalid.ipw = f64::NAN;
        assert!(transport_effect_from_wire(&invalid).is_err());
        invalid = valid.clone();
        invalid.aipw = Some(f64::INFINITY);
        assert!(transport_effect_from_wire(&invalid).is_err());
        for (min, max, ess) in [
            (0.9, 0.1, 1.0),
            (-0.1, 1.0, 1.0),
            (0.0, f64::NAN, 1.0),
            (0.0, 1.0, -1.0),
            (0.0, 1.0, f64::INFINITY),
        ] {
            invalid = valid.clone();
            invalid.treatment_overlap.probability_min = min;
            invalid.treatment_overlap.probability_max = max;
            invalid.treatment_overlap.effective_sample_size = ess;
            assert!(transport_effect_from_wire(&invalid).is_err());
        }
    }

    #[test]
    fn checked_interference_decoder_enforces_wire_domains() {
        let estimate = InterferenceEstimate {
            contrast: RandomizationContrast {
                horvitz_thompson: 1.0,
                hajek: 0.9,
                conservative_variance: 0.2,
            },
            from_probability_method: ExposureProbabilityMethod::Exact,
            to_probability_method: ExposureProbabilityMethod::MonteCarlo { draws: 1, seed: 7 },
            minimum_exposure_probability: 0.1,
        };
        let wire = interference_estimate_to_wire(&estimate);
        assert_eq!(try_interference_estimate_from_wire(&wire).unwrap(), estimate);
        let mut invalid = wire.clone();
        invalid.contrast.hajek = f64::NAN;
        assert!(try_interference_estimate_from_wire(&invalid).is_err());
        invalid = wire.clone();
        invalid.contrast.conservative_variance = -1.0;
        assert!(try_interference_estimate_from_wire(&invalid).is_err());
        invalid = wire.clone();
        invalid.minimum_exposure_probability = 0.0;
        assert!(try_interference_estimate_from_wire(&invalid).is_err());
        invalid = wire;
        invalid.to_probability_method =
            ExposureProbabilityMethodWire::MonteCarlo { draws: 0, seed: 7 };
        assert!(try_interference_estimate_from_wire(&invalid).is_err());
    }

    #[test]
    fn legacy_transport_effect_without_aipw_defaults_to_none() {
        let json = r#"{
            "ipw": 1.5,
            "selection_overlap": {"probability_min":0.2,"probability_max":0.8,"effective_sample_size":4.0,"extreme_weight_count":0},
            "treatment_overlap": {"probability_min":0.4,"probability_max":0.6,"effective_sample_size":4.0,"extreme_weight_count":0}
        }"#;
        let wire: TransportEffectEstimateWire = serde_json::from_str(json).unwrap();
        assert_eq!(wire.aipw, None);
    }
}
