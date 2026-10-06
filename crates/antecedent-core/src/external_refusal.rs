//! Structured refusals for external-object negotiation, verification and binding.
//!
//! Each typed error maps to one registered reason code and a namespaced detail
//! (`family.slot`). The structured fields name the stage, the offending
//! coordinate or probe, expected versus supplied semantics, the missing
//! capability and a remedy when one is known. Nothing here is parsed out of a
//! message; the existing codes are unchanged.

use crate::{
    CheckedCausalContract, ExternalBindingError, ExternalCapability, ExternalContractError,
    ExternalResult, ExternalVerificationError, QuantityMismatch, ScientificQuantity,
    VerificationProbeKind,
};

/// A coded, structured refusal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalRefusal {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stage that refused: `declare`, `negotiate`, `verify` or `bind`.
    pub stage: &'static str,
    /// Namespaced detail, `family.slot`.
    pub detail: String,
    /// Offending coordinate (`coordinate[i]`) or probe, when there is one.
    pub offending: Option<String>,
    /// Expected semantics, when a comparison failed.
    pub expected: Option<String>,
    /// Supplied semantics, when a comparison failed.
    pub supplied: Option<String>,
    /// Missing operation, for a capability refusal.
    pub capability: Option<ExternalCapability>,
    /// Remedy, when one is known.
    pub remedy: Option<&'static str>,
}

fn refusal(code: &'static str, stage: &'static str, detail: &str) -> ExternalRefusal {
    ExternalRefusal {
        code,
        stage,
        detail: detail.to_owned(),
        offending: None,
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    }
}

/// Stable `snake_case` name of an operation.
#[must_use]
pub const fn capability_name(capability: ExternalCapability) -> &'static str {
    match capability {
        ExternalCapability::Sample => "sample",
        ExternalCapability::Cdf => "cdf",
        ExternalCapability::Quantile => "quantile",
        ExternalCapability::LogProbability => "log_probability",
        ExternalCapability::Mean => "mean",
        ExternalCapability::Covariance => "covariance",
        ExternalCapability::Conditional => "conditional",
        ExternalCapability::Intervention => "intervention",
        ExternalCapability::PosteriorPredictive => "posterior_predictive",
        ExternalCapability::Update => "update",
        ExternalCapability::Factor => "factor",
        ExternalCapability::EvaluateUtility => "evaluate_utility",
    }
}

/// Parse the name written by [`capability_name`].
#[must_use]
pub fn capability_from_name(name: &str) -> Option<ExternalCapability> {
    use ExternalCapability as C;
    [
        C::Sample,
        C::Cdf,
        C::Quantile,
        C::LogProbability,
        C::Mean,
        C::Covariance,
        C::Conditional,
        C::Intervention,
        C::PosteriorPredictive,
        C::Update,
        C::Factor,
        C::EvaluateUtility,
    ]
    .into_iter()
    .find(|capability| capability_name(*capability) == name)
}

/// Parse the name written by [`probe_name`].
#[must_use]
pub fn probe_from_name(name: &str) -> Option<VerificationProbeKind> {
    use VerificationProbeKind as P;
    [
        P::Shape,
        P::Normalization,
        P::Moments,
        P::KnownTruth,
        P::SeededBehavior,
        P::Support,
        P::UpdateCoherence,
        P::Monotonicity,
    ]
    .into_iter()
    .find(|kind| probe_name(*kind) == name)
}

/// Stable `snake_case` name of a verification property.
#[must_use]
pub const fn probe_name(kind: VerificationProbeKind) -> &'static str {
    match kind {
        VerificationProbeKind::Shape => "shape",
        VerificationProbeKind::Normalization => "normalization",
        VerificationProbeKind::Moments => "moments",
        VerificationProbeKind::KnownTruth => "known_truth",
        VerificationProbeKind::SeededBehavior => "seeded_behavior",
        VerificationProbeKind::Support => "support",
        VerificationProbeKind::UpdateCoherence => "update_coherence",
        VerificationProbeKind::Monotonicity => "monotonicity",
    }
}

/// Namespaced detail for a coordinate mismatch; each is a literal so the
/// promotion gate can match it against the record's declared refusals.
const fn coordinate_detail(mismatch: QuantityMismatch) -> &'static str {
    match mismatch {
        QuantityMismatch::InvalidCoordinate => {
            "external_response_binding.coordinate_invalid_coordinate"
        }
        QuantityMismatch::Variable => "external_response_binding.coordinate_variable",
        QuantityMismatch::Units => "external_response_binding.coordinate_units",
        QuantityMismatch::Population => "external_response_binding.coordinate_population",
        QuantityMismatch::Regime => "external_response_binding.coordinate_regime",
        QuantityMismatch::Horizon => "external_response_binding.coordinate_horizon",
        QuantityMismatch::Functional => "external_response_binding.coordinate_functional",
        QuantityMismatch::Conditioning => "external_response_binding.coordinate_conditioning",
        QuantityMismatch::Transform => "external_response_binding.coordinate_transform",
    }
}

fn dimension_value(quantity: &ScientificQuantity, mismatch: QuantityMismatch) -> String {
    match mismatch {
        QuantityMismatch::InvalidCoordinate => quantity.variable_id.clone(),
        QuantityMismatch::Variable => format!("{}:{:?}", quantity.variable_id, quantity.role),
        QuantityMismatch::Units => quantity.units.clone(),
        QuantityMismatch::Population => quantity.population_id.clone(),
        QuantityMismatch::Regime => quantity.regime_id.clone(),
        QuantityMismatch::Horizon => quantity.horizon.to_string(),
        QuantityMismatch::Functional => quantity.functional_id.clone(),
        QuantityMismatch::Conditioning => quantity
            .conditioning
            .iter()
            .map(|c| format!("{}={}", c.variable_id, c.value_id))
            .collect::<Vec<_>>()
            .join(","),
        QuantityMismatch::Transform => quantity.transform_id.clone(),
    }
}

impl ExternalContractError {
    /// Structured refusal for a declaration or negotiation failure.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        match self {
            Self::InvalidIdentity => {
                refusal("invalid_argument", "declare", "external_object.invalid_identity")
            }
            Self::InvalidDeclaration => {
                refusal("invalid_argument", "declare", "external_object.invalid_declaration")
            }
            Self::DuplicateDeclaration => {
                refusal("invalid_argument", "declare", "external_object.duplicate_declaration")
            }
            Self::MissingCapability(capability) => ExternalRefusal {
                capability: Some(*capability),
                offending: Some(capability_name(*capability).to_owned()),
                remedy: Some(
                    "declare the exact operation, or license a sampling approximation with a numerical-error and replicate receipt",
                ),
                ..refusal(
                    crate::reason_code!("external_capability_missing"),
                    "negotiate",
                    "provider_capability_negotiation.capability_missing",
                )
            },
            Self::RequestMismatch => refusal(
                crate::reason_code!("external_binding_mismatch"),
                "negotiate",
                "provider_capability_negotiation.request_mismatch",
            ),
        }
    }
}

impl ExternalVerificationError {
    /// Structured refusal for a verification failure.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let code = crate::reason_code!("external_verification_failed");
        match self {
            Self::Contract(error) => {
                let mut value = error.to_refusal();
                value.stage = "verify";
                value
            }
            Self::InvalidProbe(kind) => ExternalRefusal {
                offending: Some(probe_name(*kind).to_owned()),
                ..refusal(
                    "invalid_argument",
                    "verify",
                    "external_object_verification.invalid_probe",
                )
            },
            Self::DuplicateProbe(kind) => ExternalRefusal {
                offending: Some(probe_name(*kind).to_owned()),
                ..refusal(
                    "invalid_argument",
                    "verify",
                    "external_object_verification.duplicate_probe",
                )
            },
            Self::MissingProbe(kind) => ExternalRefusal {
                offending: Some(probe_name(*kind).to_owned()),
                remedy: Some("supply an independently computed probe for this property"),
                ..refusal(code, "verify", "external_object_verification.missing_probe")
            },
            Self::FailedProbe(kind) => ExternalRefusal {
                offending: Some(probe_name(*kind).to_owned()),
                ..refusal(code, "verify", "external_object_verification.failed_probe")
            },
        }
    }
}

impl ExternalBindingError {
    /// Structured refusal for a binding failure, with expected and supplied
    /// semantics read from the contract and result that were presented.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_refusal(
        &self,
        contract: &CheckedCausalContract,
        result: &ExternalResult,
    ) -> ExternalRefusal {
        let binding = crate::reason_code!("external_binding_mismatch");
        let header = match result {
            ExternalResult::Response(r) => &r.header,
            ExternalResult::Distribution(d) => &d.header,
            ExternalResult::Posterior(p) => &p.header,
        };
        let at = |code, detail: &str| refusal(code, "bind", detail);
        match self {
            Self::ContractNotIdentified(status) => ExternalRefusal {
                supplied: Some(status.as_str().to_owned()),
                expected: Some("an identified estimand".to_owned()),
                ..at(
                    crate::reason_code!("effect_not_identified"),
                    "external_response_binding.contract_not_identified",
                )
            },
            Self::InvalidContract => {
                at("invalid_argument", "external_response_binding.invalid_contract")
            }
            Self::Object(error) => {
                let mut value = error.to_refusal();
                value.stage = "bind";
                value
            }
            Self::ObjectKindMismatch => at(binding, "external_response_binding.object_kind"),
            Self::ObjectQuantityMismatch => ExternalRefusal {
                remedy: Some(
                    "label the result with the exact coordinates declared by its provider object",
                ),
                ..at(binding, "external_response_binding.object_quantities")
            },
            Self::GraphMismatch => ExternalRefusal {
                expected: Some(contract.graph_id.clone()),
                supplied: Some(header.graph_id.clone()),
                ..at(binding, "external_response_binding.graph")
            },
            Self::DimensionMismatch => ExternalRefusal {
                expected: Some(contract.estimand.len().to_string()),
                supplied: Some(header.quantities.len().to_string()),
                ..at(binding, "external_response_binding.dimension")
            },
            Self::CoordinateMismatch(index, mismatch) => {
                let expected = contract.estimand.get(*index);
                let supplied = header.quantities.get(*index);
                ExternalRefusal {
                    offending: Some(format!("coordinate[{index}]")),
                    expected: expected.map(|q| dimension_value(q, *mismatch)),
                    supplied: supplied.map(|q| dimension_value(q, *mismatch)),
                    remedy: (*mismatch == QuantityMismatch::Conditioning
                        && supplied.is_some_and(|q| {
                            q.regime_id == crate::external_binding::OBSERVATIONAL_REGIME
                        }))
                    .then_some(
                        "if this observational law is meant to answer the interventional request, provide a checked equivalence for the treatment conditioning",
                    ),
                    ..at(
                        crate::reason_code!("quantity_semantics_mismatch"),
                        coordinate_detail(*mismatch),
                    )
                }
            }
            Self::UncheckedObservationalLaw(index) => ExternalRefusal {
                offending: Some(format!("coordinate[{index}]")),
                expected: contract.estimand.get(*index).map(|q| q.regime_id.clone()),
                supplied: Some(crate::external_binding::OBSERVATIONAL_REGIME.to_owned()),
                remedy: Some(
                    "provide a checked observational-to-interventional equivalence for this graph and regime",
                ),
                ..at(binding, "external_response_binding.unchecked_observational_law")
            },
            Self::NonFiniteValue(index) => ExternalRefusal {
                offending: Some(format!("coordinate[{index}]")),
                ..at("invalid_argument", "external_response_binding.non_finite_value")
            },
            Self::MeaningNotAccepted => ExternalRefusal {
                expected: Some(
                    contract
                        .accepted_meanings
                        .iter()
                        .map(|m| format!("{m:?}"))
                        .collect::<Vec<_>>()
                        .join(","),
                ),
                supplied: match (result, &header.object) {
                    (ExternalResult::Distribution(d), _) => Some(format!("{:?}", d.meaning)),
                    (_, crate::ExternalScientificObject::Law(law)) => {
                        Some(format!("{:?}", law.meaning))
                    }
                    _ => None,
                },
                ..at(
                    crate::reason_code!("distribution_meaning_mismatch"),
                    "external_response_binding.distribution_meaning",
                )
            },
            Self::PosteriorKindMismatch => at(binding, "external_response_binding.posterior_kind"),
            Self::MissingEvidence(id) => ExternalRefusal {
                offending: Some(id.clone()),
                remedy: Some("declare the evidence factor the proof requires"),
                ..at(binding, "external_response_binding.missing_evidence")
            },
            Self::MissingAssumption(id) => ExternalRefusal {
                offending: Some(id.clone()),
                remedy: Some("declare the assumption the identification relies on"),
                ..at(binding, "external_response_binding.missing_assumption")
            },
            Self::TrustMismatch => at(
                crate::reason_code!("external_verification_failed"),
                "external_response_binding.trust",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DistributionMeaning, ExternalResponse, ExternalResultHeader, ExternalScientificObject,
        ExternalTrustState, ExternalUncertaintyMeaning, IdentificationStatus, LawProviderContract,
        ProviderObjectIdentity, QuantityRole, bind_external_result,
    };

    fn quantity(horizon: u32, units: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "schema:y".into(),
            variable_name: "Y".into(),
            role: QuantityRole::Outcome,
            units: units.into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon,
            functional_id: "mean".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    fn fixtures() -> (CheckedCausalContract, ExternalResponse) {
        let quantities = vec![quantity(0, "mmHg"), quantity(1, "mmHg")];
        let object = ExternalScientificObject::Law(LawProviderContract {
            identity: ProviderObjectIdentity {
                provider_id: "lab".into(),
                object_id: "curve".into(),
                version_id: "v1".into(),
                snapshot_id: "s".into(),
                request_id: "r".into(),
            },
            quantities: quantities.clone(),
            meaning: DistributionMeaning::InterventionalPredictive,
            capabilities: vec![ExternalCapability::Mean],
        });
        let contract = CheckedCausalContract {
            graph_id: "g1".into(),
            identification: IdentificationStatus::NonparametricallyIdentified,
            estimand: quantities.clone(),
            accepted_meanings: vec![DistributionMeaning::InterventionalPredictive],
            required_evidence_ids: vec![],
            required_assumption_ids: vec![],
            equivalences: vec![],
        };
        let response = ExternalResponse {
            header: ExternalResultHeader {
                object,
                graph_id: "g1".into(),
                quantities,
                evidence_ids: vec![],
                assumption_ids: vec![],
                trust: ExternalTrustState::ExternallyAttested { attestor: "lab".into() },
            },
            values: vec![1.0, 2.0],
            uncertainty: ExternalUncertaintyMeaning::None,
            point_support: None,
        };
        (contract, response)
    }

    fn refused(contract: &CheckedCausalContract, response: ExternalResponse) -> ExternalRefusal {
        let result = ExternalResult::Response(response);
        bind_external_result(contract, &result).unwrap_err().to_refusal(contract, &result)
    }

    #[test]
    fn wire_names_round_trip() {
        use ExternalCapability as C;
        for capability in [C::Sample, C::Cdf, C::Quantile, C::Update, C::EvaluateUtility] {
            assert_eq!(capability_from_name(capability_name(capability)), Some(capability));
        }
        assert_eq!(capability_from_name("nope"), None);
        for status in [IdentificationStatus::GraphDependent, IdentificationStatus::NotIdentified] {
            assert_eq!(IdentificationStatus::from_name(status.as_str()), Some(status));
        }
        assert_eq!(
            crate::SupportStatus::from_name("missing_evidence").map(crate::SupportStatus::as_str),
            Some("missing_evidence")
        );
        assert_eq!(crate::SupportStatus::from_name("bad"), None);
    }

    #[test]
    fn coordinate_mismatch_names_the_coordinate_and_expected_versus_supplied() {
        let (contract, mut response) = fixtures();
        response.header.quantities[1] = quantity(1, "kPa");
        let value = refused(&contract, response);
        assert_eq!(value.code, "quantity_semantics_mismatch");
        assert_eq!(value.stage, "bind");
        assert_eq!(value.detail, "external_response_binding.coordinate_units");
        assert_eq!(value.offending.as_deref(), Some("coordinate[1]"));
        assert_eq!(value.expected.as_deref(), Some("mmHg"));
        assert_eq!(value.supplied.as_deref(), Some("kPa"));

        let (contract, mut response) = fixtures();
        response.header.quantities[0].horizon = 7;
        let value = refused(&contract, response);
        assert_eq!(value.detail, "external_response_binding.coordinate_horizon");
        assert_eq!(value.expected.as_deref(), Some("0"));
        assert_eq!(value.supplied.as_deref(), Some("7"));
    }

    #[test]
    fn binding_refusals_use_registered_runtime_codes_and_distinct_details() {
        let (contract, base) = fixtures();
        let mut seen = std::collections::BTreeSet::new();
        let mut cases: Vec<(ExternalResponse, &str, &str)> = Vec::new();
        let mut graph = base.clone();
        graph.header.graph_id = "g2".into();
        cases.push((graph, "external_binding_mismatch", "external_response_binding.graph"));
        let mut short = base.clone();
        short.values.pop();
        cases.push((short, "external_binding_mismatch", "external_response_binding.dimension"));
        let mut nan = base.clone();
        nan.values[0] = f64::NAN;
        cases.push((nan, "invalid_argument", "external_response_binding.non_finite_value"));
        let mut native = base.clone();
        native.header.trust = ExternalTrustState::NativeLicensed;
        cases.push((native, "external_verification_failed", "external_response_binding.trust"));
        let mut observational = base.clone();
        for q in &mut observational.header.quantities {
            q.regime_id = "observational".into();
        }
        cases.push((
            observational,
            "external_binding_mismatch",
            "external_response_binding.unchecked_observational_law",
        ));
        for (response, code, detail) in cases {
            let value = refused(&contract, response);
            assert_eq!((value.code, value.detail.as_str()), (code, detail));
            assert!(crate::reason_code::is_runtime_refusal(value.code), "{code}");
            assert!(seen.insert(value.detail));
        }
        let mut unidentified = contract.clone();
        unidentified.identification = IdentificationStatus::NotIdentified;
        let value = refused(&unidentified, base.clone());
        assert_eq!(value.code, "effect_not_identified");
        assert_eq!(value.supplied.as_deref(), Some("not_identified"));
        let mut needs = contract;
        needs.required_evidence_ids = vec!["factor:z".into()];
        let value = refused(&needs, base);
        assert_eq!(value.offending.as_deref(), Some("factor:z"));
        assert!(value.remedy.is_some());
    }

    #[test]
    fn capability_and_verification_refusals_name_the_missing_operation_or_probe() {
        let value = ExternalContractError::MissingCapability(ExternalCapability::Cdf).to_refusal();
        assert_eq!(value.code, "external_capability_missing");
        assert_eq!(value.stage, "negotiate");
        assert_eq!(value.capability, Some(ExternalCapability::Cdf));
        assert_eq!(value.offending.as_deref(), Some("cdf"));
        assert!(value.remedy.is_some());
        assert_eq!(
            ExternalContractError::RequestMismatch.to_refusal().detail,
            "provider_capability_negotiation.request_mismatch"
        );

        let missing = ExternalVerificationError::MissingProbe(VerificationProbeKind::Normalization)
            .to_refusal();
        assert_eq!(missing.code, "external_verification_failed");
        assert_eq!(missing.offending.as_deref(), Some("normalization"));
        assert_eq!(missing.detail, "external_object_verification.missing_probe");
        let failed =
            ExternalVerificationError::FailedProbe(VerificationProbeKind::Moments).to_refusal();
        assert_eq!(failed.detail, "external_object_verification.failed_probe");
        let contract = ExternalVerificationError::Contract(ExternalContractError::RequestMismatch)
            .to_refusal();
        assert_eq!((contract.stage, contract.code), ("verify", "external_binding_mismatch"));
        for r in [value, missing, failed, contract] {
            assert!(crate::reason_code::is_runtime_refusal(r.code), "{}", r.code);
        }
    }
}
