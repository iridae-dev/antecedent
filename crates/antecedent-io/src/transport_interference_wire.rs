//! Durable wire forms for transportability and randomized-interference results.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_estimate::{
    InterferenceEstimate, TransportEffectEstimate, TransportOverlapDiagnostic,
    TransportOverlapReport,
};
use antecedent_identify::{
    NonTransportableCertificate, PopulationFactor, TransportCertificate, TransportFormula,
    TransportIdentification,
};
use antecedent_stats::{ExposureProbabilityMethod, RandomizationContrast};
use serde::{Deserialize, Serialize};

use crate::IoError;

/// Population-labelled symbolic distribution wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PopulationFactorWire {
    /// Population key.
    pub population: String,
    /// Variables in the factor.
    pub variables: Vec<u32>,
    /// Conditioning variables.
    pub conditioned_on: Vec<u32>,
    /// Experimental intervention variables.
    pub interventions: Vec<u32>,
}

/// Symbolic transport formula wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransportFormulaWire {
    /// Direct source response.
    Direct(PopulationFactorWire),
    /// Standardization over a target law.
    Standardize {
        /// Marginalized variables.
        over: Vec<u32>,
        /// Source conditional response.
        source_response: PopulationFactorWire,
        /// Target covariate law.
        target_law: PopulationFactorWire,
    },
    /// Recursive singleton c-component factorization.
    RecursiveFactorization {
        /// Marginalized variables.
        sum_out: Vec<u32>,
        /// Population-labelled factors.
        factors: Vec<PopulationFactorWire>,
    },
}

/// Positive transport certificate wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransportCertificateWire {
    /// Stable rule id.
    pub rule: String,
    /// Selection targets.
    pub selection_targets: Vec<u32>,
    /// Checked premises.
    pub premises: Vec<String>,
}

/// Scoped negative transport certificate wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct NonTransportableCertificateWire {
    /// Stable reason id.
    pub reason: String,
    /// Witness variables.
    pub witness: Vec<u32>,
    /// Scope-preserving message.
    pub message: String,
}

/// Transport identification result wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransportIdentificationWire {
    /// Sound formula and positive certificate.
    Transportable {
        /// Formula.
        formula: Box<TransportFormulaWire>,
        /// Certificate.
        certificate: TransportCertificateWire,
    },
    /// No implemented rule certified a formula; not a general impossibility claim.
    NotCertified(NonTransportableCertificateWire),
}

/// Exposure-probability method wire form.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExposureProbabilityMethodWire {
    /// Exact support enumeration.
    Exact,
    /// Deterministic Monte Carlo.
    MonteCarlo {
        /// Assignment draws.
        draws: u32,
        /// RNG seed.
        seed: u64,
    },
}

/// Randomization contrast wire form.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct RandomizationContrastWire {
    /// HT contrast.
    pub horvitz_thompson: f64,
    /// Hájek contrast.
    pub hajek: f64,
    /// Conservative variance.
    pub conservative_variance: f64,
}

/// Randomized-interference estimate wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterferenceEstimateWire {
    /// Contrast.
    pub contrast: RandomizationContrastWire,
    /// Baseline exposure-probability method.
    pub from_probability_method: ExposureProbabilityMethodWire,
    /// Active exposure-probability method.
    pub to_probability_method: ExposureProbabilityMethodWire,
    /// Minimum exposure probability.
    pub minimum_exposure_probability: f64,
}

/// One transport overlap diagnostic wire form.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct TransportOverlapDiagnosticWire {
    /// Minimum probability.
    pub probability_min: f64,
    /// Maximum probability.
    pub probability_max: f64,
    /// Kish effective sample size.
    pub effective_sample_size: f64,
    /// Extreme-weight count.
    pub extreme_weight_count: u64,
}

/// Trial-to-target effect result wire form.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct TransportEffectEstimateWire {
    /// IPW estimate.
    pub ipw: f64,
    /// AIPW estimate when outcome regressions were supplied.
    #[serde(default)]
    pub aipw: Option<f64>,
    /// Selection overlap.
    pub selection_overlap: TransportOverlapDiagnosticWire,
    /// Treatment overlap.
    pub treatment_overlap: TransportOverlapDiagnosticWire,
}

/// Encode structural transport identification without collapsing `NotCertified`.
#[must_use]
pub fn transport_identification_to_wire(
    value: &TransportIdentification,
) -> TransportIdentificationWire {
    match value {
        TransportIdentification::Transportable { formula, certificate } => {
            TransportIdentificationWire::Transportable {
                formula: Box::new(formula_to_wire(formula)),
                certificate: TransportCertificateWire {
                    rule: certificate.rule.to_string(),
                    selection_targets: certificate
                        .selection_targets
                        .iter()
                        .map(|id| id.raw())
                        .collect(),
                    premises: certificate.premises.iter().map(ToString::to_string).collect(),
                },
            }
        }
        TransportIdentification::NotCertified(certificate) => {
            TransportIdentificationWire::NotCertified(NonTransportableCertificateWire {
                reason: certificate.reason.to_string(),
                witness: certificate.witness.iter().map(|id| id.raw()).collect(),
                message: certificate.message.to_string(),
            })
        }
    }
}

/// Decode structural transport identification.
#[must_use]
pub fn transport_identification_from_wire(
    value: &TransportIdentificationWire,
) -> TransportIdentification {
    match value {
        TransportIdentificationWire::Transportable { formula, certificate } => {
            TransportIdentification::Transportable {
                formula: formula_from_wire(formula),
                certificate: TransportCertificate {
                    rule: Arc::from(certificate.rule.as_str()),
                    selection_targets: certificate
                        .selection_targets
                        .iter()
                        .copied()
                        .map(antecedent_core::VariableId::from_raw)
                        .collect::<Vec<_>>()
                        .into(),
                    premises: certificate
                        .premises
                        .iter()
                        .map(|value| Arc::from(value.as_str()))
                        .collect::<Vec<_>>()
                        .into(),
                },
            }
        }
        TransportIdentificationWire::NotCertified(certificate) => {
            TransportIdentification::NotCertified(NonTransportableCertificate {
                reason: Arc::from(certificate.reason.as_str()),
                witness: certificate
                    .witness
                    .iter()
                    .copied()
                    .map(antecedent_core::VariableId::from_raw)
                    .collect::<Vec<_>>()
                    .into(),
                message: Arc::from(certificate.message.as_str()),
            })
        }
    }
}

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

/// Decode an interference estimate.
#[must_use]
pub fn interference_estimate_from_wire(value: &InterferenceEstimateWire) -> InterferenceEstimate {
    InterferenceEstimate {
        contrast: contrast_from_wire(value.contrast),
        from_probability_method: probability_method_from_wire(value.from_probability_method),
        to_probability_method: probability_method_from_wire(value.to_probability_method),
        minimum_exposure_probability: value.minimum_exposure_probability,
    }
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
    Ok(TransportEffectEstimate {
        ipw: value.ipw,
        aipw: value.aipw,
        overlap: TransportOverlapReport {
            selection: diagnostic_from_wire(value.selection_overlap)?,
            treatment: diagnostic_from_wire(value.treatment_overlap)?,
        },
    })
}

fn factor_to_wire(value: &PopulationFactor) -> PopulationFactorWire {
    PopulationFactorWire {
        population: value.population.to_string(),
        variables: value.variables.iter().map(|id| id.raw()).collect(),
        conditioned_on: value.conditioned_on.iter().map(|id| id.raw()).collect(),
        interventions: value.interventions.iter().map(|id| id.raw()).collect(),
    }
}

fn factor_from_wire(value: &PopulationFactorWire) -> PopulationFactor {
    let vars = |values: &[u32]| {
        values.iter().copied().map(antecedent_core::VariableId::from_raw).collect::<Vec<_>>().into()
    };
    PopulationFactor {
        population: Arc::from(value.population.as_str()),
        variables: vars(&value.variables),
        conditioned_on: vars(&value.conditioned_on),
        interventions: vars(&value.interventions),
    }
}

fn formula_to_wire(value: &TransportFormula) -> TransportFormulaWire {
    match value {
        TransportFormula::Direct(factor) => TransportFormulaWire::Direct(factor_to_wire(factor)),
        TransportFormula::Standardize { over, source_response, target_law } => {
            TransportFormulaWire::Standardize {
                over: over.iter().map(|id| id.raw()).collect(),
                source_response: factor_to_wire(source_response),
                target_law: factor_to_wire(target_law),
            }
        }
        TransportFormula::RecursiveFactorization { sum_out, factors } => {
            TransportFormulaWire::RecursiveFactorization {
                sum_out: sum_out.iter().map(|id| id.raw()).collect(),
                factors: factors.iter().map(factor_to_wire).collect(),
            }
        }
    }
}

fn formula_from_wire(value: &TransportFormulaWire) -> TransportFormula {
    match value {
        TransportFormulaWire::Direct(factor) => TransportFormula::Direct(factor_from_wire(factor)),
        TransportFormulaWire::Standardize { over, source_response, target_law } => {
            TransportFormula::Standardize {
                over: over
                    .iter()
                    .copied()
                    .map(antecedent_core::VariableId::from_raw)
                    .collect::<Vec<_>>()
                    .into(),
                source_response: factor_from_wire(source_response),
                target_law: factor_from_wire(target_law),
            }
        }
        TransportFormulaWire::RecursiveFactorization { sum_out, factors } => {
            TransportFormula::RecursiveFactorization {
                sum_out: sum_out
                    .iter()
                    .copied()
                    .map(antecedent_core::VariableId::from_raw)
                    .collect::<Vec<_>>()
                    .into(),
                factors: factors.iter().map(factor_from_wire).collect::<Vec<_>>().into(),
            }
        }
    }
}

const fn probability_method_to_wire(
    value: ExposureProbabilityMethod,
) -> ExposureProbabilityMethodWire {
    match value {
        ExposureProbabilityMethod::Exact => ExposureProbabilityMethodWire::Exact,
        ExposureProbabilityMethod::MonteCarlo { draws, seed } => {
            ExposureProbabilityMethodWire::MonteCarlo { draws, seed }
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
    fn not_certified_transport_round_trip_preserves_scope() {
        let original = TransportIdentification::NotCertified(NonTransportableCertificate {
            reason: Arc::from("transport.sid.multinode_c_component_not_implemented"),
            witness: Arc::from([antecedent_core::VariableId::from_raw(2)]),
            message: Arc::from("no non-transportability claim is made"),
        });
        let bytes = crate::to_cbor(&transport_identification_to_wire(&original)).unwrap();
        let wire: TransportIdentificationWire = crate::from_cbor(&bytes).unwrap();
        assert_eq!(transport_identification_from_wire(&wire), original);
    }

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
