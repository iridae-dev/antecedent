//! Durable wire forms for transportability and randomized-interference results.
//!
//! The estimate conversions need the estimators and are joined to this path by
//! `antecedent-io`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_identify::{
    MissingEvidenceCertificate, NotCertifiedCertificate, PopulationFactor, TransportCertificate,
    TransportFormula, TransportIdentification,
};
use serde::{Deserialize, Serialize};

/// Population-labelled symbolic distribution wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PopulationFactorWire {
    /// Population key.
    pub population: String,
    /// Supplied evidence regime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regime: Option<u32>,
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
pub struct InterferenceTransportCertificateWire {
    /// Stable rule id.
    pub rule: String,
    /// Selection targets.
    pub selection_targets: Vec<u32>,
    /// Checked premises.
    pub premises: Vec<String>,
}

/// Historical Rust name for [`InterferenceTransportCertificateWire`].
///
/// CBOR and JSON field names are unchanged. The alias keeps existing imports compiling.
pub type TransportCertificateWire = InterferenceTransportCertificateWire;

/// Inconclusive transport certificate wire form; never an impossibility proof.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct NotCertifiedCertificateWire {
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
    NotCertified(NotCertifiedCertificateWire),
    /// A required available source regime is absent. Additive in this slice:
    /// existing `Transportable` / `NotCertified` artifacts still decode.
    MissingEvidence(NotCertifiedCertificateWire),
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
    /// Caller supplied known exposure probabilities.
    SuppliedKnown,
    /// Caller supplied externally estimated probabilities.
    SuppliedExternallyEstimated,
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
                certificate: InterferenceTransportCertificateWire {
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
            TransportIdentificationWire::NotCertified(NotCertifiedCertificateWire {
                reason: certificate.reason.to_string(),
                witness: certificate.witness.iter().map(|id| id.raw()).collect(),
                message: certificate.message.to_string(),
            })
        }
        TransportIdentification::MissingEvidence(certificate) => {
            TransportIdentificationWire::MissingEvidence(NotCertifiedCertificateWire {
                reason: certificate.reason.to_string(),
                witness: certificate.missing.iter().map(|id| id.raw()).collect(),
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
            TransportIdentification::NotCertified(NotCertifiedCertificate {
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
        TransportIdentificationWire::MissingEvidence(certificate) => {
            TransportIdentification::MissingEvidence(MissingEvidenceCertificate {
                reason: Arc::from(certificate.reason.as_str()),
                missing: certificate
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

fn factor_to_wire(value: &PopulationFactor) -> PopulationFactorWire {
    PopulationFactorWire {
        regime: value.regime.map(antecedent_core::RegimeId::raw),
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
        regime: value.regime.map(antecedent_core::RegimeId::from_raw),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_evidence_transport_round_trip_is_not_not_certified() {
        let original = TransportIdentification::MissingEvidence(MissingEvidenceCertificate {
            reason: Arc::from("transport.source_experiment_missing"),
            missing: Arc::from([antecedent_core::VariableId::from_raw(0)]),
            message: Arc::from(
                "the source experiment required by this transport query is unavailable",
            ),
        });
        let bytes = crate::to_cbor(&transport_identification_to_wire(&original)).unwrap();
        let wire: TransportIdentificationWire = crate::from_cbor(&bytes).unwrap();
        assert!(matches!(wire, TransportIdentificationWire::MissingEvidence(_)));
        assert_eq!(transport_identification_from_wire(&wire), original);
        assert_ne!(original.outcome().kind.as_str(), "not_certified");
    }

    #[test]
    fn not_certified_transport_round_trip_preserves_scope() {
        let original = TransportIdentification::NotCertified(NotCertifiedCertificate {
            reason: Arc::from("transport.sid.multinode_c_component_not_implemented"),
            witness: Arc::from([antecedent_core::VariableId::from_raw(2)]),
            message: Arc::from("no non-transportability claim is made"),
        });
        let bytes = crate::to_cbor(&transport_identification_to_wire(&original)).unwrap();
        let wire: TransportIdentificationWire = crate::from_cbor(&bytes).unwrap();
        assert_eq!(transport_identification_from_wire(&wire), original);
    }
}
