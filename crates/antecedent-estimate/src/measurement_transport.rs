//! Measurement models carried to another population under a declared invariance premise.
//!
//! A [`MeasurementModel`] is established for one population: its sensitivity and
//! specificity, confusion matrix or error variance were learned on a validation sample
//! there, and the correction entry points refuse data from any other population
//! (`measurement_binding_mismatch`). Carrying the parameters to another population is a
//! transport claim, and this module makes it only when the caller declares the premise:
//!
//! * [`PopulationErrorStructure::Invariant`] states that the named error parameters are
//!   the same in the source and the target population. Every parameter the model's route
//!   uses must be named, so a premise about specificity alone does not license a
//!   sensitivity-and-specificity correction. The premise is declared, never tested.
//! * [`PopulationErrorStructure::Differential`] is a distinct declared case: the error
//!   differs by population, so nothing is carried. Only a model established for the target
//!   population itself is used, and a missing target model refuses rather than falling
//!   back to another population's rates (`differential_error_not_pooled`).
//!
//! The result, a [`TransportedMeasurementModel`], is the source model re-bound to the
//! target population plus a receipt that binds the source model identity, both
//! populations, the premise and the basis. It is applied through
//! [`TransportedMeasurementModel::apply`], which runs an existing correction entry point on
//! the re-bound model and returns its output with the receipt, so a corrected functional is
//! never presented as native to the target.
//!
//! Only the parameter estimation uncertainty the correction already propagates is
//! reported. The uncertainty of the invariance premise itself is not propagated
//! ([`ParameterUncertainty::NotPropagated`] in the receipt's note), the standard error a
//! correction reports is uncalibrated and conditional on the premise, and no interval is
//! issued.
//!
//! Routes with nothing to carry are refused: replicated measurements re-estimate the error
//! from the target's own replicates, and an external latent posterior is bound to its
//! source snapshot.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::identity::{CanonicalWriter, SemanticDigest};
use antecedent_core::measurement_model::{
    ArmBinding, DifferentialAxis, ErrorStructure, MeasurementModel, MeasurementRoute,
    ParameterEvidence,
};

use crate::measurement::{DataBinding, MeasurementError};

/// Statement of the assumption an invariance declaration makes.
pub const MEASUREMENT_INVARIANCE_ASSUMPTION: &str =
    "measurement_error_parameters_invariant_across_populations";

const TRANSPORT_TAG: &str = "antecedent.measurement_transport.receipt.v1";

/// One error parameter a declaration may state invariant.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ErrorParameter {
    /// `P(observed = 1 | true = 1)` of a binary indicator.
    Sensitivity,
    /// `P(observed = 0 | true = 0)` of a binary indicator.
    Specificity,
    /// The column-stochastic confusion matrix of an assigned categorical label.
    ConfusionMatrix,
    /// The additive error variance of a continuous indicator.
    ErrorVariance,
}

impl ErrorParameter {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sensitivity => "sensitivity",
            Self::Specificity => "specificity",
            Self::ConfusionMatrix => "confusion_matrix",
            Self::ErrorVariance => "error_variance",
        }
    }

    fn required_by(route: MeasurementRoute) -> &'static [Self] {
        match route {
            MeasurementRoute::BinaryMisclassification => &[Self::Sensitivity, Self::Specificity],
            MeasurementRoute::ProbabilisticCategorical => &[Self::ConfusionMatrix],
            MeasurementRoute::ClassicalContinuousError => &[Self::ErrorVariance],
            MeasurementRoute::RepeatedMeasurements | MeasurementRoute::ExternalLatentPosterior => {
                &[]
            }
        }
    }
}

/// The declared premise that error parameters carry from one population to another.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvarianceDeclaration {
    /// Population the model was established for.
    pub source_population: String,
    /// Population the parameters are carried to.
    pub target_population: String,
    /// Who makes the declaration.
    pub declarer: String,
    /// Parameters declared the same in both populations.
    pub invariant: BTreeSet<ErrorParameter>,
}

impl InvarianceDeclaration {
    /// Declare invariance of `invariant` between two populations.
    #[must_use]
    pub fn new(
        source_population: &str,
        target_population: &str,
        declarer: &str,
        invariant: &[ErrorParameter],
    ) -> Self {
        Self {
            source_population: source_population.to_owned(),
            target_population: target_population.to_owned(),
            declarer: declarer.to_owned(),
            invariant: invariant.iter().copied().collect(),
        }
    }
}

/// How the measurement error is declared to relate across populations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PopulationErrorStructure {
    /// The error parameters carry across populations.
    Invariant(InvarianceDeclaration),
    /// The error differs by population. Nothing is carried; each population needs a model
    /// established for it.
    Differential {
        /// Who makes the declaration.
        declarer: String,
    },
}

/// Why a model may be applied to the target population.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransportBasis {
    /// The model was established for the target population: nothing was carried.
    Established,
    /// Parameters carried under a declared invariance premise.
    DeclaredInvariance(InvarianceDeclaration),
    /// A model established for the target was selected under a declared population-differential
    /// error structure; no other population's parameters were used.
    DifferentialEstablished {
        /// Who declared the differential structure.
        declarer: String,
    },
}

/// The audit record of one transport.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportReceipt {
    /// Identity of the model the parameters come from.
    pub source_model: SemanticDigest,
    /// Population of that model.
    pub source_population: String,
    /// Population the parameters are applied to.
    pub target_population: String,
    /// Why the application is licensed.
    pub basis: TransportBasis,
    /// Evidence the parameters carry from their own population.
    pub evidence: ParameterEvidence,
    /// Identity binding all of the above and the re-bound model.
    pub identity: SemanticDigest,
}

impl TransportReceipt {
    /// Whether any parameter was carried from another population.
    #[must_use]
    pub fn carried(&self) -> bool {
        matches!(self.basis, TransportBasis::DeclaredInvariance(_))
    }

    /// Statements that travel with a corrected functional.
    #[must_use]
    pub fn assumptions(&self) -> Vec<String> {
        match &self.basis {
            TransportBasis::DeclaredInvariance(declaration) => vec![
                MEASUREMENT_INVARIANCE_ASSUMPTION.to_owned(),
                format!(
                    "{} invariant between {} and {}, declared by {}",
                    declaration
                        .invariant
                        .iter()
                        .map(|p| p.as_str())
                        .collect::<Vec<_>>()
                        .join(","),
                    declaration.source_population,
                    declaration.target_population,
                    declaration.declarer
                ),
                "invariance uncertainty is not propagated; the standard error is conditional on \
                 the premise and uncalibrated; no interval"
                    .to_owned(),
            ],
            TransportBasis::Established | TransportBasis::DifferentialEstablished { .. } => {
                Vec::new()
            }
        }
    }
}

/// A measurement model usable on a target population, with the receipt that licenses it.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedMeasurementModel {
    model: MeasurementModel,
    receipt: TransportReceipt,
}

/// A correction output with the receipt of the transport it ran under.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedCorrection<T> {
    /// The correction entry point's own output, bound to the re-bound model's identity.
    pub corrected: T,
    /// The transport receipt.
    pub receipt: TransportReceipt,
}

fn refusal(
    code: &'static str,
    detail: &'static str,
    message: impl Into<String>,
) -> MeasurementError {
    MeasurementError::new(code, detail, message)
}

impl TransportedMeasurementModel {
    /// The model re-bound to the target population.
    #[must_use]
    pub fn model(&self) -> &MeasurementModel {
        &self.model
    }

    /// The transport receipt.
    #[must_use]
    pub fn receipt(&self) -> &TransportReceipt {
        &self.receipt
    }

    /// Run a correction entry point on the re-bound model.
    ///
    /// `data` must describe the target population; data from any other population is
    /// refused, so a transported model never silently serves a third population.
    ///
    /// # Errors
    /// A data binding for a population other than the target, or whatever `correct`
    /// refuses.
    pub fn apply<T>(
        &self,
        data: &DataBinding,
        correct: impl FnOnce(&MeasurementModel, &DataBinding) -> Result<T, MeasurementError>,
    ) -> Result<TransportedCorrection<T>, MeasurementError> {
        if data.population_id != self.receipt.target_population {
            return Err(refusal(
                antecedent_core::reason_code!("measurement_binding_mismatch"),
                "measurement_model.population_mismatch",
                format!(
                    "the transported model is for `{}`, the data are `{}`; declare a transport \
                     to this population",
                    self.receipt.target_population, data.population_id
                ),
            ));
        }
        let corrected = correct(&self.model, data)?;
        Ok(TransportedCorrection { corrected, receipt: self.receipt.clone() })
    }
}

fn receipt_identity(
    source: &MeasurementModel,
    rebound: &MeasurementModel,
    basis: &TransportBasis,
) -> SemanticDigest {
    let mut canon = CanonicalWriter::new(TRANSPORT_TAG);
    canon.bytes(source.identities().model.as_bytes());
    canon.bytes(rebound.identities().model.as_bytes());
    match basis {
        TransportBasis::Established => {
            canon.word(1);
        }
        TransportBasis::DeclaredInvariance(declaration) => {
            canon
                .word(2)
                .text(MEASUREMENT_INVARIANCE_ASSUMPTION)
                .text(&declaration.source_population)
                .text(&declaration.target_population)
                .text(&declaration.declarer)
                .word(declaration.invariant.len() as u64);
            for parameter in &declaration.invariant {
                canon.text(parameter.as_str());
            }
        }
        TransportBasis::DifferentialEstablished { declarer } => {
            canon.word(3).text(declarer);
        }
    }
    SemanticDigest::from_bytes(canon.finish())
}

fn established(model: &MeasurementModel, basis: TransportBasis) -> TransportedMeasurementModel {
    let identity = receipt_identity(model, model, &basis);
    TransportedMeasurementModel {
        receipt: TransportReceipt {
            source_model: model.identities().model,
            source_population: model.binding.population_id.clone(),
            target_population: model.binding.population_id.clone(),
            basis,
            evidence: model.evidence.clone(),
            identity,
        },
        model: model.clone(),
    }
}

/// Make a measurement model usable on the target population of `data`.
///
/// `models` are the models established so far, each for its own population. A model
/// established for the target is used as is. Otherwise the declared `structure` decides:
/// invariance carries the declared source's parameters, a declared population-differential
/// error refuses, and no declaration refuses naming the premise that is missing.
///
/// Order of refusal: well-formed models, target model, declaration present, declaration
/// names the source and target, regime unchanged, route has something to carry, error
/// structure poolable across populations, every required parameter declared invariant.
///
/// # Errors
/// A coded [`MeasurementError`] naming the unmet premise; no model is issued.
pub fn transport_measurement_model(
    models: &[MeasurementModel],
    structure: Option<&PopulationErrorStructure>,
    data: &DataBinding,
) -> Result<TransportedMeasurementModel, MeasurementError> {
    if models.is_empty() {
        return Err(refusal(
            antecedent_core::reason_code!("measurement_model_invalid"),
            "measurement_model.no_model",
            "supply the measurement model established for a source population",
        ));
    }
    for model in models {
        model.validate().map_err(|error| crate::measurement::model_error(&error))?;
    }
    if let Some(own) = models.iter().find(|m| m.binding.population_id == data.population_id) {
        let basis = match structure {
            Some(PopulationErrorStructure::Differential { declarer }) => {
                TransportBasis::DifferentialEstablished { declarer: declarer.clone() }
            }
            _ => TransportBasis::Established,
        };
        return Ok(established(own, basis));
    }
    let Some(structure) = structure else {
        return Err(refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            "measurement_model.invariance_premise_required",
            format!(
                "no measurement model is established for population `{}`; declare that the \
                 error parameters are invariant from a named source population, declare the \
                 error differential by population, or establish a validation sample there",
                data.population_id
            ),
        ));
    };
    let declaration = match structure {
        PopulationErrorStructure::Differential { .. } => {
            return Err(refusal(
                antecedent_core::reason_code!("differential_error_not_pooled"),
                "measurement_model.differential_population_unestablished",
                format!(
                    "the error is declared differential by population and no model is \
                     established for `{}`; another population's rates are not borrowed",
                    data.population_id
                ),
            ));
        }
        PopulationErrorStructure::Invariant(declaration) => declaration,
    };
    if declaration.target_population != data.population_id {
        return Err(refusal(
            antecedent_core::reason_code!("transport_not_certified"),
            "measurement_model.invariance_target_mismatch",
            format!(
                "the invariance premise is declared for `{}`, the data are `{}`",
                declaration.target_population, data.population_id
            ),
        ));
    }
    let Some(source) =
        models.iter().find(|m| m.binding.population_id == declaration.source_population)
    else {
        return Err(refusal(
            antecedent_core::reason_code!("transport_not_certified"),
            "measurement_model.invariance_source_unknown",
            format!(
                "the premise names source population `{}`, which has no established model",
                declaration.source_population
            ),
        ));
    };
    if source.binding.regime_id != data.regime_id {
        return Err(refusal(
            antecedent_core::reason_code!("measurement_binding_mismatch"),
            "measurement_model.regime_mismatch",
            format!(
                "the invariance premise covers the population only; the model was established \
                 for regime `{}`, the data are `{}`",
                source.binding.regime_id, data.regime_id
            ),
        ));
    }
    match source.route {
        MeasurementRoute::RepeatedMeasurements => {
            return Err(refusal(
                antecedent_core::reason_code!("route_not_supported"),
                "measurement_model.nothing_to_transport",
                "replicated measurements estimate the error from the analysis sample's own \
                 replicates; correct the target data with its replicates instead of carrying \
                 parameters",
            ));
        }
        MeasurementRoute::ExternalLatentPosterior => {
            return Err(refusal(
                antecedent_core::reason_code!("route_not_supported"),
                "measurement_model.posterior_bound_to_snapshot",
                "an external latent posterior is bound to its source snapshot and request; it \
                 cannot be carried to another population",
            ));
        }
        _ => {}
    }
    match source.structure {
        ErrorStructure::NonDifferential | ErrorStructure::Differential(DifferentialAxis::Arm) => {}
        _ => {
            return Err(refusal(
                antecedent_core::reason_code!("differential_error_not_pooled"),
                "measurement_model.differential_axis_unsupported",
                "the model's error is declared to depend on population or time, so it is not \
                 carried by an invariance premise",
            ));
        }
    }
    let missing: Vec<&'static str> = ErrorParameter::required_by(source.route)
        .iter()
        .filter(|p| !declaration.invariant.contains(p))
        .map(|p| p.as_str())
        .collect();
    if !missing.is_empty() {
        return Err(refusal(
            antecedent_core::reason_code!("transport_not_certified"),
            "measurement_model.invariance_not_declared",
            format!(
                "the correction uses parameters the premise does not declare invariant: {}",
                missing.join(",")
            ),
        ));
    }
    if let ArmBinding::Arm(arm) = &source.binding.arm {
        if data.arm_id.as_deref() != Some(arm.as_str()) {
            return Err(refusal(
                antecedent_core::reason_code!("measurement_binding_mismatch"),
                "measurement_model.arm_mismatch",
                format!("the model is bound to arm `{arm}`; the invariance premise covers the population only"),
            ));
        }
    }
    let mut rebound = source.clone();
    rebound.binding.population_id.clone_from(&declaration.target_population);
    let basis = TransportBasis::DeclaredInvariance(declaration.clone());
    let identity = receipt_identity(source, &rebound, &basis);
    Ok(TransportedMeasurementModel {
        receipt: TransportReceipt {
            source_model: source.identities().model,
            source_population: source.binding.population_id.clone(),
            target_population: declaration.target_population.clone(),
            basis,
            evidence: source.evidence.clone(),
            identity,
        },
        model: rebound,
    })
}

#[cfg(test)]
mod tests {
    use antecedent_core::ExecutionContext;
    use antecedent_core::measurement_model::{
        BinaryArmRates, MeasurementBinding, MeasurementKind, MeasurementParameters,
        MeasurementVariable, RepresentationVersion,
    };

    use super::*;
    use crate::measurement::correct_binary_prevalence;

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(3)
    }

    fn model(population: &str, sensitivity: f64, specificity: f64) -> MeasurementModel {
        MeasurementModel {
            model_id: format!("screen-{population}"),
            route: MeasurementRoute::BinaryMisclassification,
            indicator: MeasurementVariable {
                id: "schema:flag".into(),
                kind: MeasurementKind::NoisyMisclassified,
                units: "binary".into(),
            },
            construct: MeasurementVariable {
                id: "construct:disease".into(),
                kind: MeasurementKind::LatentScientific,
                units: "binary".into(),
            },
            binding: MeasurementBinding {
                population_id: population.into(),
                regime_id: "observational".into(),
                arm: ArmBinding::AllArms,
            },
            structure: ErrorStructure::NonDifferential,
            parameters: MeasurementParameters::Binary {
                rates: vec![BinaryArmRates {
                    arm_id: None,
                    sensitivity,
                    specificity,
                    sensitivity_variance: 0.0,
                    specificity_variance: 0.0,
                }],
                min_youden: 0.1,
            },
            representation: RepresentationVersion {
                coordinate_system_id: "scalar".into(),
                version: 1,
            },
            evidence: ParameterEvidence::ValidationSample {
                sample_id: format!("val-{population}"),
                units: 500,
            },
        }
    }

    fn data(population: &str) -> DataBinding {
        DataBinding {
            population_id: population.into(),
            regime_id: "observational".into(),
            arm_id: None,
            representation: RepresentationVersion {
                coordinate_system_id: "scalar".into(),
                version: 1,
            },
        }
    }

    fn premise(params: &[ErrorParameter]) -> PopulationErrorStructure {
        PopulationErrorStructure::Invariant(InvarianceDeclaration::new(
            "clinic_a", "clinic_b", "analyst", params,
        ))
    }

    const BOTH: [ErrorParameter; 2] = [ErrorParameter::Sensitivity, ErrorParameter::Specificity];

    #[test]
    fn carried_rates_correct_the_target_prevalence() {
        // Rogan-Gladen: (0.52 + 0.8 - 1) / (0.9 + 0.8 - 1) = 0.32 / 0.7.
        let source = model("clinic_a", 0.9, 0.8);
        let transported =
            transport_measurement_model(&[source], Some(&premise(&BOTH)), &data("clinic_b"))
                .unwrap();
        let out = transported
            .apply(&data("clinic_b"), |m, d| correct_binary_prevalence(m, d, 520, 1000, &ctx()))
            .unwrap();
        assert!((out.corrected.result.corrected - 0.32 / 0.7).abs() < 1e-12);
        assert!(out.receipt.carried());
        assert_eq!(out.receipt.target_population, "clinic_b");
        assert!(!out.corrected.record.has_interval());
    }

    #[test]
    fn untransported_model_is_refused_on_another_population() {
        let source = model("clinic_a", 0.9, 0.8);
        let error = correct_binary_prevalence(&source, &data("clinic_b"), 520, 1000, &ctx())
            .unwrap_err();
        assert_eq!(error.detail, "measurement_model.population_mismatch");
    }

    #[test]
    fn missing_premise_names_the_invariance_premise() {
        let source = model("clinic_a", 0.9, 0.8);
        let error = transport_measurement_model(&[source], None, &data("clinic_b")).unwrap_err();
        assert_eq!(error.code, "transport_missing_evidence");
        assert_eq!(error.detail, "measurement_model.invariance_premise_required");
    }

    #[test]
    fn partial_premise_names_the_undeclared_parameter() {
        let source = model("clinic_a", 0.9, 0.8);
        let error = transport_measurement_model(
            &[source],
            Some(&premise(&[ErrorParameter::Specificity])),
            &data("clinic_b"),
        )
        .unwrap_err();
        assert_eq!(error.code, "transport_not_certified");
        assert_eq!(error.detail, "measurement_model.invariance_not_declared");
        assert!(error.message.contains("sensitivity"));
        assert!(!error.message.contains("specificity,"));
    }

    #[test]
    fn differential_population_error_is_not_pooled() {
        let source = model("clinic_a", 0.9, 0.8);
        let differential = PopulationErrorStructure::Differential { declarer: "analyst".into() };
        let error =
            transport_measurement_model(&[source], Some(&differential), &data("clinic_b"))
                .unwrap_err();
        assert_eq!(error.code, "differential_error_not_pooled");
        assert_eq!(error.detail, "measurement_model.differential_population_unestablished");
    }

    #[test]
    fn differential_case_uses_the_target_populations_own_model() {
        // Different rates per population: the target's own 0.7 / 0.95 are used, not 0.9 / 0.8.
        // (0.52 + 0.95 - 1) / (0.7 + 0.95 - 1) = 0.47 / 0.65.
        let models = [model("clinic_a", 0.9, 0.8), model("clinic_b", 0.7, 0.95)];
        let differential = PopulationErrorStructure::Differential { declarer: "analyst".into() };
        let transported =
            transport_measurement_model(&models, Some(&differential), &data("clinic_b")).unwrap();
        assert!(!transported.receipt().carried());
        let out = transported
            .apply(&data("clinic_b"), |m, d| correct_binary_prevalence(m, d, 520, 1000, &ctx()))
            .unwrap();
        assert!((out.corrected.result.corrected - 0.47 / 0.65).abs() < 1e-12);
    }

    #[test]
    fn premise_for_other_populations_or_regime_is_refused() {
        let source = model("clinic_a", 0.9, 0.8);
        let wrong_target = PopulationErrorStructure::Invariant(InvarianceDeclaration::new(
            "clinic_a", "clinic_c", "analyst", &BOTH,
        ));
        let error = transport_measurement_model(
            std::slice::from_ref(&source),
            Some(&wrong_target),
            &data("clinic_b"),
        )
        .unwrap_err();
        assert_eq!(error.detail, "measurement_model.invariance_target_mismatch");
        let unknown_source = PopulationErrorStructure::Invariant(InvarianceDeclaration::new(
            "clinic_z", "clinic_b", "analyst", &BOTH,
        ));
        let error = transport_measurement_model(
            std::slice::from_ref(&source),
            Some(&unknown_source),
            &data("clinic_b"),
        )
        .unwrap_err();
        assert_eq!(error.detail, "measurement_model.invariance_source_unknown");
        let mut other_regime = data("clinic_b");
        other_regime.regime_id = "do(a=1)".into();
        let error =
            transport_measurement_model(&[source], Some(&premise(&BOTH)), &other_regime)
                .unwrap_err();
        assert_eq!(error.detail, "measurement_model.regime_mismatch");
    }

    #[test]
    fn transported_model_does_not_serve_a_third_population() {
        let source = model("clinic_a", 0.9, 0.8);
        let transported =
            transport_measurement_model(&[source], Some(&premise(&BOTH)), &data("clinic_b"))
                .unwrap();
        let error = transported
            .apply(&data("clinic_c"), |m, d| correct_binary_prevalence(m, d, 1, 2, &ctx()))
            .unwrap_err();
        assert_eq!(error.detail, "measurement_model.population_mismatch");
    }

    #[test]
    fn receipt_identity_moves_with_the_declared_parameters() {
        let source = model("clinic_a", 0.9, 0.8);
        let a = transport_measurement_model(
            std::slice::from_ref(&source),
            Some(&premise(&BOTH)),
            &data("clinic_b"),
        )
        .unwrap();
        let mut more = BOTH.to_vec();
        more.push(ErrorParameter::ErrorVariance);
        let b = transport_measurement_model(&[source], Some(&premise(&more)), &data("clinic_b"))
            .unwrap();
        assert_ne!(a.receipt().identity, b.receipt().identity);
        assert_eq!(a.receipt().source_model, b.receipt().source_model);
    }

    #[test]
    fn replicate_route_has_nothing_to_carry() {
        let mut source = model("clinic_a", 0.9, 0.8);
        source.route = MeasurementRoute::RepeatedMeasurements;
        source.parameters = MeasurementParameters::Repeated {
            min_replicated_units: 20,
            min_reliability: 0.1,
        };
        let error = transport_measurement_model(
            &[source],
            Some(&premise(&[ErrorParameter::ErrorVariance])),
            &data("clinic_b"),
        )
        .unwrap_err();
        assert_eq!(error.detail, "measurement_model.nothing_to_transport");
    }
}
