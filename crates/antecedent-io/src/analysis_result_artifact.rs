//! Composite analysis-result artifact.
//!
//! This additive container keeps response, posterior, mediation-grid, and
//! structural-mixture axes together without extending the exhaustive legacy
//! `CausalPayloadKind` enum.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::{
    ArtifactKind, ArtifactManifest, AssumptionRecordWire, CausalQueryWire, CausalResponseWire,
    CompressPolicy, DiagnosticWire, EncodedArtifact, IdentificationResultWire, IoError,
    ProvenanceWire, RefutationReportWire, STABLE_FORMAT, SemanticVersion, from_cbor,
    pack_section_shared, read_and_migrate, to_cbor,
};

const ARTIFACT_KIND: &str = "analysis_result";
const HEADER_SECTION: &str = "analysis_result.header";
const BODY_SECTION: &str = "analysis_result.body";

/// One posterior interval summary in a temporal mediation slice.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MediationPosteriorSummaryWire {
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub standard_deviation: f64,
    /// Lower equal-tail quantile.
    pub q025: f64,
    /// Upper equal-tail quantile.
    pub q975: f64,
}

/// Pointwise uncertainty for one mediation horizon.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TemporalMediationUncertaintyWire {
    /// Frequentist pointwise standard error.
    FrequentistPointwise {
        /// Standard error for the requested contrast.
        standard_error: Option<f64>,
    },
    /// Per-horizon posterior summaries.
    BayesianPointwise {
        /// Requested contrast.
        requested: MediationPosteriorSummaryWire,
        /// Total effect.
        total: MediationPosteriorSummaryWire,
        /// Direct effect.
        direct: MediationPosteriorSummaryWire,
        /// Mediated effect.
        mediated: MediationPosteriorSummaryWire,
        /// Draw count.
        n_draws: u64,
        /// Inference backend.
        backend: String,
    },
    /// No justified uncertainty.
    Unavailable,
}

/// One horizon in a temporal mediation grid.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TemporalMediationSliceWire {
    /// Requested horizon.
    pub horizon: u32,
    /// Identification status at this horizon.
    pub identification_status: crate::IdentificationStatusWire,
    /// Identifier method.
    pub method: String,
    /// Horizon-specific adjustment set.
    pub adjustment: Vec<crate::HorizonAdjustmentNodeWire>,
    /// Requested effect.
    pub effect: f64,
    /// Total effect.
    pub total: Option<f64>,
    /// Direct effect.
    pub direct: Option<f64>,
    /// Mediated effect.
    pub mediated: Option<f64>,
    /// Pointwise uncertainty.
    pub uncertainty: TemporalMediationUncertaintyWire,
    /// Completion-specific structural interval.
    pub identified_set: Option<[f64; 2]>,
    /// Horizon-local diagnostics.
    pub diagnostics: Vec<DiagnosticWire>,
}

/// Durable horizon-indexed temporal mediation result.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TemporalMediationGridWire {
    /// Horizon slices in query order.
    pub slices: Vec<TemporalMediationSliceWire>,
    /// Whether one joint posterior generated all horizons.
    pub joint_posterior: bool,
}

/// Meaning of structural atom weights.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StructuralWeightBasisWire {
    /// Graph-posterior probability.
    PosteriorProbability,
    /// Completion-enumeration weight.
    CompletionEnumeration,
}

/// One graph/completion response atom.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StructuralResponseAtomWire {
    /// Stable atom key.
    pub graph_key: u64,
    /// Raw atom weight.
    pub weight: f64,
    /// Atom identification status.
    pub identification_status: crate::IdentificationStatusWire,
    /// Numerical response when evaluable.
    pub value: Option<crate::ResponseValueWire>,
}

/// Structural-mixture response metadata.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StructuralResponseMixtureWire {
    /// Meaning of atom weights.
    pub weight_basis: StructuralWeightBasisWire,
    /// Examined structural atoms.
    pub atoms: Vec<StructuralResponseAtomWire>,
    /// Evaluable identified mass.
    pub identified_mass: f64,
    /// Structurally unidentified mass.
    pub unidentified_mass: f64,
    /// Identified but numerically unevaluable mass.
    pub unevaluable_mass: f64,
    /// Pointwise identified set.
    pub identified_set: Option<crate::ResponseEnvelopeWire>,
    /// Conditional summary, only for posterior-probability weights.
    pub conditional_on_identified: Option<crate::ResponseValueWire>,
    /// Whether masses cover the full structural support.
    pub full_mass_scope: bool,
    /// Number of capped atom searches.
    pub truncated_atoms: u64,
}

/// A temporal identification certificate with its explicit unfolded variable namespace.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TemporalIdentificationWire {
    /// Requested outcome horizon.
    pub horizon: u32,
    /// Dense identification id indexes this map to a base-schema variable and offset.
    pub variables: Vec<crate::HorizonAdjustmentNodeWire>,
    /// Horizon-specific identification and derivation, using dense ids above.
    pub identification: IdentificationResultWire,
}

/// Composite result body. Every scientific axis is independently optional.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnalysisResultWire {
    /// Original query.
    pub query: CausalQueryWire,
    /// Structural identification and derivation.
    pub identification: IdentificationResultWire,
    /// Namespace for the primary identification; absent means the base schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identification_variables: Option<Vec<crate::HorizonAdjustmentNodeWire>>,
    /// Full horizon-specific certificates and namespaces, when prepared temporally.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub temporal_identification: Vec<TemporalIdentificationWire>,
    /// Scalar estimate when one exists; function-valued results have no scalar placeholder.
    pub estimate: Option<f64>,
    /// Scalar standard error when justified.
    pub standard_error: Option<f64>,
    /// Estimation assumptions.
    pub assumptions: Vec<AssumptionRecordWire>,
    /// Execution and scientific diagnostics.
    pub diagnostics: Vec<DiagnosticWire>,
    /// Validation reports.
    pub refutations: Vec<RefutationReportWire>,
    /// Response axis.
    pub response: Option<CausalResponseWire>,
    /// Nested canonical posterior artifact bytes, including draws when requested.
    pub posterior_artifact: Option<Vec<u8>>,
    /// Horizon-indexed mediation axis.
    pub mediation_grid: Option<TemporalMediationGridWire>,
    /// Structural-mixture response axis.
    pub structural_response: Option<StructuralResponseMixtureWire>,
}

/// Composite artifact header.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnalysisResultHeader {
    /// Variable names in raw-id order.
    pub variable_names: Vec<String>,
}

/// Encode a composite result artifact.
///
/// # Errors
///
/// Returns an error when CBOR/container encoding fails.
pub fn encode_analysis_result_artifact(
    result: &AnalysisResultWire,
    variable_names: Vec<String>,
    artifact_id: &str,
) -> Result<EncodedArtifact, IoError> {
    validate_result(result, &variable_names)?;
    let header = to_cbor(&AnalysisResultHeader { variable_names })?;
    let body = to_cbor(result)?;
    let (header_descriptor, header_section) = pack_section_shared(
        HEADER_SECTION,
        "application/cbor",
        header.into(),
        CompressPolicy::Auto,
    );
    let (body_descriptor, body_section) =
        pack_section_shared(BODY_SECTION, "application/cbor", body.into(), CompressPolicy::Auto);
    Ok(EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: STABLE_FORMAT,
            minimum_reader_version: STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
            library_version: SemanticVersion::from_crate_version(antecedent_core::VERSION)?,
            artifact_id: artifact_id.into(),
            sections: vec![header_descriptor, body_descriptor],
            provenance: ProvenanceWire { note: "composite analysis result".into() },
        },
        sections: vec![header_section, body_section],
    })
}

/// Decode a composite result artifact.
///
/// # Errors
///
/// Returns an error for the wrong artifact kind, missing sections, or invalid CBOR.
pub fn decode_analysis_result_artifact(
    bytes: &[u8],
) -> Result<(EncodedArtifact, AnalysisResultHeader, AnalysisResultWire), IoError> {
    let artifact = read_and_migrate(bytes)?;
    if artifact.manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into()) {
        return Err(IoError::Convert("expected an analysis_result artifact".into()));
    }
    let header = artifact
        .sections
        .iter()
        .find(|section| section.id == HEADER_SECTION)
        .ok_or_else(|| IoError::Convert(format!("missing section `{HEADER_SECTION}`")))?;
    let body = artifact
        .sections
        .iter()
        .find(|section| section.id == BODY_SECTION)
        .ok_or_else(|| IoError::Convert(format!("missing section `{BODY_SECTION}`")))?;
    let decoded_header: AnalysisResultHeader = from_cbor(&header.data)?;
    let decoded_body: AnalysisResultWire = from_cbor(&body.data)?;
    validate_result(&decoded_body, &decoded_header.variable_names)?;
    Ok((artifact, decoded_header, decoded_body))
}

fn validate_result(result: &AnalysisResultWire, variable_names: &[String]) -> Result<(), IoError> {
    use crate::causal_artifact::{
        validate_query_ids, validate_response_result, validate_variable_names,
    };
    validate_variable_names(variable_names)?;
    validate_query_ids(&result.query, variable_names.len())?;
    crate::causal_query_from_wire(&result.query)?;
    let identification_count = match &result.identification_variables {
        Some(variables) => {
            validate_temporal_namespace(variables, variable_names.len())?;
            variables.len()
        }
        None => variable_names.len(),
    };
    validate_identification_namespace(&result.identification, identification_count)?;
    let mut horizons = std::collections::BTreeSet::new();
    for horizon in &result.temporal_identification {
        if horizon.horizon == 0 || !horizons.insert(horizon.horizon) {
            return Err(IoError::Convert(
                "temporal identification horizons must be positive and unique".into(),
            ));
        }
        validate_temporal_namespace(&horizon.variables, variable_names.len())?;
        validate_identification_namespace(&horizon.identification, horizon.variables.len())?;
        crate::identification_from_wire(&horizon.identification)?;
    }
    crate::identification_from_wire(&result.identification)?;
    if result.estimate.is_some_and(|estimate| !estimate.is_finite()) {
        return Err(IoError::Convert(
            "analysis scalar estimate must be finite when present".into(),
        ));
    }
    if result.standard_error.is_some_and(|se| !se.is_finite() || se < 0.0) {
        return Err(IoError::Convert(
            "analysis standard error must be finite and nonnegative".into(),
        ));
    }
    if let Some(response) = &result.response {
        validate_response_result(response, variable_names.len())?;
    }
    if let Some(posterior) = &result.posterior_artifact {
        crate::decode_causal_posterior_bytes(posterior)?;
    }
    Ok(())
}

fn validate_identification_namespace(
    identification: &IdentificationResultWire,
    count: usize,
) -> Result<(), IoError> {
    crate::causal_artifact::validate_query_ids(&identification.query, count)?;
    let ids = identification
        .arena
        .var_sets
        .iter()
        .flatten()
        .copied()
        .chain(identification.arena.interventions.iter().flatten().map(|value| value.variable))
        .chain(
            identification
                .estimands
                .iter()
                .flat_map(|estimand| {
                    estimand
                        .adjustment_set
                        .iter()
                        .chain(&estimand.instruments)
                        .chain(&estimand.mediators)
                })
                .copied(),
        );
    if ids.into_iter().any(|id| id as usize >= count) {
        return Err(IoError::Convert(
            "identification variable is outside its declared namespace".into(),
        ));
    }
    Ok(())
}

fn validate_temporal_namespace(
    variables: &[crate::HorizonAdjustmentNodeWire],
    base_count: usize,
) -> Result<(), IoError> {
    let mut keys = std::collections::BTreeSet::new();
    if variables.is_empty() {
        return Err(IoError::Convert("empty temporal variable namespace".into()));
    }
    for node in variables {
        if node.variable as usize >= base_count || !keys.insert((node.variable, node.offset)) {
            return Err(IoError::Convert(
                "temporal namespace contains an unknown base variable or duplicate node".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> AnalysisResultWire {
        let query = serde_json::json!({"response": {
            "functional": {"average_derivative": {"outcome": 1, "treatment": 0, "weighting": "observed"}},
            "target_population": "all_observed", "observation": "complete", "observation_assumptions": []
        }});
        serde_json::from_value(serde_json::json!({
            "query": query,
            "identification": {"status": "nonparametrically_identified", "query": query,
                "estimands": [], "arena": {"var_sets": [], "interventions": [], "lists": [], "nodes": []},
                "derivation": [], "required_assumptions": [], "diagnostics": [], "candidates_examined": 0, "sets_returned": 0},
            "estimate": 1.0, "standard_error": 0.2, "assumptions": [], "diagnostics": [], "refutations": [],
            "response": null, "posterior_artifact": null, "mediation_grid": null, "structural_response": null
        })).unwrap()
    }

    #[test]
    fn composite_roundtrip_validates_names_ids_and_nested_posterior() {
        let result = fixture();
        let names = vec!["a".into(), "y".into()];
        let artifact = encode_analysis_result_artifact(&result, names.clone(), "review").unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (_, header, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(decoded, result);
        assert_eq!(header.variable_names, names);
        assert!(encode_analysis_result_artifact(&result, vec!["a".into()], "review").is_err());
        assert!(
            encode_analysis_result_artifact(&result, vec!["a".into(), "a".into()], "review")
                .is_err()
        );
        let mut invalid = result;
        invalid.posterior_artifact = Some(vec![1, 2, 3]);
        assert!(encode_analysis_result_artifact(&invalid, names, "review").is_err());
    }
    #[test]
    fn temporal_identification_uses_its_own_namespace_and_preserves_offsets() {
        let mut result = fixture();
        // Base query remains ids 0,1; its unfolded certificate uses ids 4,5.
        let mut wire = serde_json::to_value(&result.identification).unwrap();
        wire["query"]["response"]["functional"]["average_derivative"]["treatment"] = 4.into();
        wire["query"]["response"]["functional"]["average_derivative"]["outcome"] = 5.into();
        result.identification = serde_json::from_value(wire).unwrap();
        let variables: Vec<_> = (-2..=0)
            .flat_map(|offset| {
                (0..2).map(move |variable| crate::HorizonAdjustmentNodeWire { variable, offset })
            })
            .collect();
        result.identification_variables = Some(variables.clone());
        result.temporal_identification.push(TemporalIdentificationWire {
            horizon: 1,
            variables,
            identification: result.identification.clone(),
        });
        let names = vec!["x".into(), "y".into()];
        let artifact = encode_analysis_result_artifact(&result, names.clone(), "temporal").unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(decoded, result);
        result.identification_variables.as_mut().unwrap()[5].variable = 2;
        assert!(encode_analysis_result_artifact(&result, names.clone(), "invalid").is_err());
        result.identification_variables = None;
        assert!(encode_analysis_result_artifact(&result, names, "missing").is_err());
    }
    #[test]
    fn function_valued_result_has_no_scalar_and_survives_json_bridge() {
        let mut result = fixture();
        assert_eq!(result.estimate, Some(1.0)); // Legacy numeric scalar remains readable.
        result.estimate = None;
        result.standard_error = None;
        let json = serde_json::to_value(&result).unwrap();
        assert!(json["estimate"].is_null());
        let restored: AnalysisResultWire = serde_json::from_value(json).unwrap();
        let artifact =
            encode_analysis_result_artifact(&restored, vec!["x".into(), "y".into()], "functional")
                .unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        assert_eq!(decode_analysis_result_artifact(&bytes).unwrap().2, result);
        result.estimate = Some(f64::NAN);
        assert!(
            encode_analysis_result_artifact(&result, vec!["x".into(), "y".into()], "invalid")
                .is_err()
        );
    }
}
