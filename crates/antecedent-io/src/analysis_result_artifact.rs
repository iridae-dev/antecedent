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
    /// Caller-declared mass over class members.
    CallerSuppliedClassPrior,
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
    /// Completion-conditional posterior artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior_artifact: Option<Vec<u8>>,
    /// Full response with conditional sampling uncertainty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<crate::CausalResponseWire>,
}

/// Construction of an [`IdentifiedSetIntervalWire`].
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IdentifiedSetIntervalMethodWire {
    /// Frequentist Imbens–Manski interval from shared circular-block replicates.
    ImbensManskiSharedBlock,
    /// Product-posterior envelope quantiles at Imbens–Manski tails.
    #[serde(alias = "imbens_manski_posterior_draws")]
    ProductPosteriorEnvelopeQuantile,
}

/// Interval for the identified set of a class-aware scalar effect. Frequentist
/// (`imbens_manski_shared_block`): covers the true effect with asymptotic
/// probability at least `level` whenever it is one retained identified
/// completion's effect. Bayesian (`product_posterior_envelope_quantile`): every
/// retained completion's posterior puts at most `1 − Φ(critical_value)` of its
/// mass outside each endpoint. Mirrors [`antecedent_estimate::IdentifiedSetInterval`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct IdentifiedSetIntervalWire {
    /// Nominal coverage of the true (completion-specific) effect.
    pub level: f64,
    /// Lower endpoint of the interval.
    pub lower: f64,
    /// Upper endpoint of the interval.
    pub upper: f64,
    /// Estimated lower bound `min_g θ̂_g`.
    pub bound_lower: f64,
    /// Estimated upper bound `max_g θ̂_g`.
    pub bound_upper: f64,
    /// Endpoint SD: `lower = bound_lower − critical_value · lower_se`.
    pub lower_se: f64,
    /// Endpoint SD: `upper = bound_upper + critical_value · upper_se`.
    pub upper_se: f64,
    /// Imbens–Manski critical value.
    pub critical_value: f64,
    /// Whether the estimated width passed the moment-selection threshold.
    pub width_retained: bool,
    /// Identified completions whose effects enter the set (every fitted
    /// completion, in both constructions).
    pub completions: u64,
    /// Replicates or posterior draws behind the SDs.
    pub replicates: u64,
    /// Construction.
    pub method: IdentifiedSetIntervalMethodWire,
    /// The completion enumeration (or its equivalence audit) was capped: the
    /// set spans retained completions only. Omitted when false; artifacts
    /// written before the field existed decode as `false`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

impl IdentifiedSetIntervalMethodWire {
    /// Wire tag for an in-memory construction.
    ///
    /// # Errors
    ///
    /// A construction this wire version has no tag for.
    pub fn try_from_method(
        method: antecedent_estimate::IdentifiedSetIntervalMethod,
    ) -> Result<Self, IoError> {
        use antecedent_estimate::IdentifiedSetIntervalMethod as Method;
        match method {
            Method::ImbensManskiSharedBlock => Ok(Self::ImbensManskiSharedBlock),
            Method::ProductPosteriorEnvelopeQuantile => Ok(Self::ProductPosteriorEnvelopeQuantile),
            other => Err(IoError::Convert(format!(
                "identified-set interval construction {other:?} has no wire tag"
            ))),
        }
    }

    /// In-memory construction for this wire tag.
    #[must_use]
    pub const fn method(self) -> antecedent_estimate::IdentifiedSetIntervalMethod {
        use antecedent_estimate::IdentifiedSetIntervalMethod as Method;
        match self {
            Self::ImbensManskiSharedBlock => Method::ImbensManskiSharedBlock,
            Self::ProductPosteriorEnvelopeQuantile => Method::ProductPosteriorEnvelopeQuantile,
        }
    }
}

/// Encode an identified-set interval.
///
/// # Errors
///
/// A construction with no wire tag (never silently relabelled).
pub fn identified_set_interval_to_wire(
    interval: &antecedent_estimate::IdentifiedSetInterval,
) -> Result<IdentifiedSetIntervalWire, IoError> {
    Ok(IdentifiedSetIntervalWire {
        level: interval.level,
        lower: interval.lower,
        upper: interval.upper,
        bound_lower: interval.bound_lower,
        bound_upper: interval.bound_upper,
        lower_se: interval.lower_se,
        upper_se: interval.upper_se,
        critical_value: interval.critical_value,
        width_retained: interval.width_retained,
        completions: u64::try_from(interval.completions).unwrap_or(u64::MAX),
        replicates: u64::try_from(interval.replicates).unwrap_or(u64::MAX),
        method: IdentifiedSetIntervalMethodWire::try_from_method(interval.method)?,
        truncated: interval.truncated,
    })
}

/// Decode and validate an identified-set interval.
///
/// # Errors
///
/// Non-finite values, a level outside `(0, 1)`, inverted endpoints or bounds,
/// negative SDs or critical value, no completions, or fewer than two replicates.
pub fn identified_set_interval_from_wire(
    wire: &IdentifiedSetIntervalWire,
) -> Result<antecedent_estimate::IdentifiedSetInterval, IoError> {
    let finite = [
        wire.level,
        wire.lower,
        wire.upper,
        wire.bound_lower,
        wire.bound_upper,
        wire.lower_se,
        wire.upper_se,
        wire.critical_value,
    ]
    .iter()
    .all(|v| v.is_finite());
    if !(finite && wire.level > 0.0 && wire.level < 1.0)
        || wire.lower > wire.upper
        || wire.bound_lower > wire.bound_upper
        || wire.lower_se < 0.0
        || wire.upper_se < 0.0
        || wire.critical_value < 0.0
        || wire.completions == 0
        || wire.replicates < 2
    {
        return Err(IoError::Convert(
            "identified-set interval must be finite with level in (0, 1), ordered endpoints \
             and bounds, nonnegative SDs, at least one completion and two replicates"
                .into(),
        ));
    }
    let count = |v: u64| {
        usize::try_from(v)
            .map_err(|_| IoError::Convert("identified-set interval count overflows".into()))
    };
    Ok(antecedent_estimate::IdentifiedSetInterval {
        level: wire.level,
        lower: wire.lower,
        upper: wire.upper,
        bound_lower: wire.bound_lower,
        bound_upper: wire.bound_upper,
        lower_se: wire.lower_se,
        upper_se: wire.upper_se,
        critical_value: wire.critical_value,
        width_retained: wire.width_retained,
        completions: count(wire.completions)?,
        replicates: count(wire.replicates)?,
        method: wire.method.method(),
        truncated: wire.truncated,
    })
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
    /// Identified mass the Interactive latency tier left out of its graph
    /// subsample and never evaluated. Absent (zero) outside that tier and on
    /// artifacts written before the field existed.
    #[serde(default, skip_serializing_if = "is_zero_mass")]
    pub subsampled_out_mass: f64,
    /// Pointwise identified set.
    pub identified_set: Option<crate::ResponseEnvelopeWire>,
    /// Imbens–Manski interval for a scalar identified set (format 0.5). Absent on
    /// format-0.4 artifacts and whenever no interval was computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identified_set_interval: Option<IdentifiedSetIntervalWire>,
    /// Conditional summary, only for posterior-probability weights.
    pub conditional_on_identified: Option<crate::ResponseValueWire>,
    /// Whether masses cover the full structural support.
    pub full_mass_scope: bool,
    /// Number of capped atom searches.
    pub truncated_atoms: u64,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde's skip_serializing_if passes `&T`.
fn is_zero_mass(mass: &f64) -> bool {
    *mass == 0.0
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
    /// Scalar standard error when justified. Absent when the published interval
    /// is an Anderson–Rubin set (`interval_lower` / `interval_upper`) or when
    /// no scalar SE is licensed.
    pub standard_error: Option<f64>,
    /// Lower endpoint of a published Anderson–Rubin set. May be infinite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_lower: Option<f64>,
    /// Upper endpoint of a published Anderson–Rubin set. May be infinite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_upper: Option<f64>,
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
    /// Per-unit counterfactual effects, when the execution computed them.
    ///
    /// Absent on every other result, so their bodies (and digests) are
    /// unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit_effects: Option<UnitEffectsWire>,
    /// Complete-case-aligned CATE point predictions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cate: Option<Vec<f64>>,
    /// Portable fitted effect, bound into the result body identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fitted_effect: Option<antecedent_estimate::FittedEffect>,
    /// Licensed pointwise CATE standard errors, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cate_se: Option<Vec<f64>>,
    /// Forest leaf-dispersion diagnostic per row. Not a standard error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cate_leaf_dispersion: Option<Vec<f64>>,
    /// Held-out outcome R².
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_oof_r2: Option<f64>,
    /// Held-out treatment probability log loss.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub treatment_oof_logloss: Option<f64>,
    /// Number of nuisance cross-fitting folds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossfit_folds: Option<usize>,
    /// Master seed used for nuisance cross-fitting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossfit_seed: Option<u64>,
    /// Fitted learner (spec, implementation, version), in fit order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub learner_provenance: Vec<(String, String, String)>,
}

/// Per-unit counterfactual effects an execution reported.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UnitEffectsWire {
    /// One effect per unit, in row order.
    pub effects: Vec<f64>,
    /// Every unit carries the same effect by construction of the fitted mechanism.
    pub homogeneous: bool,
    /// Per-unit interval bounds, when the execution formed them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervals: Option<UnitEffectIntervalsWire>,
    /// Per-unit extrapolation flags aligned with [`Self::effects`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extrapolative: Option<Vec<bool>>,
}

/// Level-tagged per-unit interval bounds aligned with [`UnitEffectsWire::effects`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UnitEffectIntervalsWire {
    /// Lower bound per unit.
    pub lower: Vec<f64>,
    /// Upper bound per unit.
    pub upper: Vec<f64>,
    /// Nominal level the bounds were read at.
    pub level: f64,
    /// Construction id.
    pub method: String,
}

/// Composite artifact header.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnalysisResultHeader {
    /// Variable names in raw-id order.
    pub variable_names: Vec<String>,
}

/// Encode a composite result artifact without a contract section.
///
/// # Errors
///
/// Returns an error when CBOR/container encoding fails.
pub fn encode_analysis_result_artifact(
    result: &AnalysisResultWire,
    variable_names: Vec<String>,
    artifact_id: &str,
) -> Result<EncodedArtifact, IoError> {
    encode_analysis_result_artifact_with_contract(result, variable_names, artifact_id, None)
}

/// Encode a composite result, optionally attaching [`crate::CONTRACT_SECTION`].
///
/// The section is additive. Old readers ignore it; this encoder does not raise
/// `minimum_reader_version`. A present contract is validated and bound to the
/// body query and header names before write.
///
/// # Errors
///
/// Returns an error when validation, contract binding, or CBOR encoding fails.
pub fn encode_analysis_result_artifact_with_contract(
    result: &AnalysisResultWire,
    variable_names: Vec<String>,
    artifact_id: &str,
    contract: Option<&crate::AnalysisResultContractWire>,
) -> Result<EncodedArtifact, IoError> {
    validate_result(result, &variable_names)?;
    if let Some(contract) = contract {
        crate::validate_contract_section(contract)?;
        let header = AnalysisResultHeader { variable_names: variable_names.clone() };
        let unresolved = crate::verify_contract_against_body(&header, result, contract);
        if !unresolved.is_empty() {
            return Err(IoError::Convert(format!(
                "contract does not verify against analysis_result body: {}",
                unresolved.join(",")
            )));
        }
    }
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
    let mut sections_desc = vec![header_descriptor, body_descriptor];
    let mut sections = vec![header_section, body_section];
    if let Some(contract) = contract {
        let payload = to_cbor(contract)?;
        let (descriptor, section) = pack_section_shared(
            crate::CONTRACT_SECTION,
            "application/cbor",
            payload.into(),
            CompressPolicy::Auto,
        );
        sections_desc.push(descriptor);
        sections.push(section);
    }
    Ok(EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: STABLE_FORMAT,
            minimum_reader_version: STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
            library_version: SemanticVersion::from_crate_version(antecedent_core::VERSION)?,
            artifact_id: artifact_id.into(),
            sections: sections_desc,
            provenance: ProvenanceWire { note: "composite analysis result".into() },
        },
        sections,
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
    if let Some(model) = &result.fitted_effect {
        // Unsupported optional prediction codecs do not erase the scientific claim.
        // The sealed body still binds their bytes; prediction loading rejects them.
        if model.version == 1 && model.predictor.version == 1 {
            model.validate().map_err(|e| IoError::Convert(e.to_string()))?;
        }
        if model.features.iter().any(|v| *v as usize >= variable_names.len()) {
            return Err(IoError::Convert("fitted effect names an unknown feature".into()));
        }
    }
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
    if result.identification_variables.is_none() && result.identification.query != result.query {
        return Err(IoError::Convert(
            "identification.query does not match the enclosing query".into(),
        ));
    }
    if let crate::CausalQueryWire::TemporalEffect { horizon_steps, .. } = &result.query {
        if !result.temporal_identification.is_empty()
            && !result.temporal_identification.iter().any(|item| item.horizon == *horizon_steps)
        {
            return Err(IoError::Convert(
                "temporal certificate horizon does not match the enclosing query".into(),
            ));
        }
    }
    if result.estimate.is_some_and(|estimate| !estimate.is_finite()) {
        return Err(IoError::Convert(
            "analysis scalar estimate must be finite when present".into(),
        ));
    }
    if result.estimate.is_some() && !licenses_scalar_estimate(&result.identification.status) {
        return Err(IoError::Convert(
            "identification status does not license a scalar estimate".into(),
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
    if let Some(grid) = &result.mediation_grid {
        validate_mediation_grid(grid, variable_names.len())?;
    }
    if let Some(unit_effects) = &result.unit_effects {
        validate_unit_effects(unit_effects)?;
    }
    validate_cate_and_learner_metrics(result)?;
    if let Some(structural) = &result.structural_response {
        validate_structural_response(structural, variable_names.len())?;
    }
    Ok(())
}

fn validate_structural_response(
    structural: &StructuralResponseMixtureWire,
    n_variables: usize,
) -> Result<(), IoError> {
    if let Some(interval) = &structural.identified_set_interval {
        identified_set_interval_from_wire(interval)?;
    }
    if !(0.0..=1.0).contains(&structural.subsampled_out_mass) {
        return Err(IoError::Convert(
            "structural subsampled_out_mass must be a fraction in [0, 1]".into(),
        ));
    }
    let total = structural.identified_mass
        + structural.unidentified_mass
        + structural.unevaluable_mass
        + structural.subsampled_out_mass;
    if !total.is_finite() || (total - 1.0).abs() > 1e-9 {
        return Err(IoError::Convert("structural masses must sum to one".into()));
    }
    if !structural.atoms.is_empty() {
        let weight: f64 = structural.atoms.iter().map(|atom| atom.weight).sum();
        if !weight.is_finite() || (weight - 1.0).abs() > 1e-9 {
            return Err(IoError::Convert("inconsistent atom totals".into()));
        }
    }
    for atom in &structural.atoms {
        if !atom.weight.is_finite() || atom.weight < 0.0 {
            return Err(IoError::Convert(
                "structural atom weight must be finite and nonnegative".into(),
            ));
        }
        if let Some(value) = &atom.value {
            crate::response_value_from_wire(value)?;
        }
        if let Some(response) = &atom.response {
            crate::causal_artifact::validate_response_result(response, n_variables)?;
        }
        if let Some(posterior) = &atom.posterior_artifact {
            crate::decode_causal_posterior_bytes(posterior)?;
        }
    }
    Ok(())
}

/// A point scalar is licensed only when identification does not refuse the estimand.
/// `not_identified` never licenses one; partial / graph-dependent mixtures may still
/// publish a conditional or retained identified-mass summary.
fn licenses_scalar_estimate(status: &str) -> bool {
    status != "not_identified"
}

fn validate_mediation_grid(
    grid: &TemporalMediationGridWire,
    base_variable_count: usize,
) -> Result<(), IoError> {
    if grid.slices.is_empty() {
        return Err(IoError::Convert("mediation grid must contain at least one slice".into()));
    }
    let mut horizons = std::collections::BTreeSet::new();
    for slice in &grid.slices {
        if slice.horizon == 0 || !horizons.insert(slice.horizon) {
            return Err(IoError::Convert(
                "mediation grid horizons must be positive and unique".into(),
            ));
        }
        if slice.method.trim().is_empty() {
            return Err(IoError::Convert("mediation grid method must be non-blank".into()));
        }
        for node in &slice.adjustment {
            if node.variable as usize >= base_variable_count {
                return Err(IoError::Convert(
                    "mediation grid adjustment names an unknown base variable".into(),
                ));
            }
        }
        let optionals = [slice.total, slice.direct, slice.mediated];
        if !slice.effect.is_finite() || optionals.iter().any(|v| v.is_some_and(|x| !x.is_finite()))
        {
            return Err(IoError::Convert(
                "mediation grid effects must be finite when present".into(),
            ));
        }
        if let Some([lower, upper]) = slice.identified_set {
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(IoError::Convert(
                    "mediation grid identified set must be finite and ordered".into(),
                ));
            }
        }
        validate_mediation_uncertainty(&slice.uncertainty)?;
    }
    Ok(())
}

fn validate_mediation_uncertainty(
    uncertainty: &TemporalMediationUncertaintyWire,
) -> Result<(), IoError> {
    match uncertainty {
        TemporalMediationUncertaintyWire::FrequentistPointwise { standard_error } => {
            if standard_error.is_some_and(|se| !se.is_finite() || se < 0.0) {
                return Err(IoError::Convert(
                    "mediation grid standard error must be finite and nonnegative".into(),
                ));
            }
        }
        TemporalMediationUncertaintyWire::BayesianPointwise {
            requested,
            total,
            direct,
            mediated,
            n_draws,
            backend,
        } => {
            for summary in [requested, total, direct, mediated] {
                if ![summary.mean, summary.standard_deviation, summary.q025, summary.q975]
                    .iter()
                    .all(|v| v.is_finite())
                    || summary.standard_deviation < 0.0
                    || summary.q025 > summary.q975
                {
                    return Err(IoError::Convert(
                        "mediation grid posterior summary must be finite and ordered".into(),
                    ));
                }
            }
            if *n_draws == 0 || backend.trim().is_empty() {
                return Err(IoError::Convert(
                    "mediation grid Bayesian uncertainty needs draws and a backend".into(),
                ));
            }
        }
        TemporalMediationUncertaintyWire::Unavailable => {}
    }
    Ok(())
}

fn validate_unit_effects(unit_effects: &UnitEffectsWire) -> Result<(), IoError> {
    if unit_effects.effects.is_empty()
        || unit_effects.effects.iter().any(|effect| !effect.is_finite())
    {
        return Err(IoError::Convert("unit effects must be a non-empty finite vector".into()));
    }
    let n = unit_effects.effects.len();
    if let Some(intervals) = &unit_effects.intervals {
        if intervals.lower.len() != n
            || intervals.upper.len() != n
            || intervals.lower.iter().any(|v| !v.is_finite())
            || intervals.upper.iter().any(|v| !v.is_finite())
            || intervals.lower.iter().zip(&intervals.upper).any(|(lo, hi)| lo > hi)
            || !(0.0..=1.0).contains(&intervals.level)
            || !intervals.level.is_finite()
            || intervals.method.trim().is_empty()
        {
            return Err(IoError::Convert(
                "unit effect intervals must match effects length with finite ordered bounds, \
                 level in [0, 1], and a non-blank method"
                    .into(),
            ));
        }
    }
    if unit_effects.extrapolative.as_ref().is_some_and(|flags| flags.len() != n) {
        return Err(IoError::Convert(
            "unit effect extrapolative flags must match effects length".into(),
        ));
    }
    Ok(())
}

fn validate_cate_and_learner_metrics(result: &AnalysisResultWire) -> Result<(), IoError> {
    if result.cate.is_none() && (result.cate_se.is_some() || result.cate_leaf_dispersion.is_some())
    {
        return Err(IoError::Convert(
            "cate standard errors and leaf dispersion require cate point predictions".into(),
        ));
    }
    if let (Some(cate), Some(dispersion)) = (&result.cate, &result.cate_leaf_dispersion) {
        if dispersion.len() != cate.len() || dispersion.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(IoError::Convert(
                "cate leaf dispersion must match cate length and be finite nonnegative".into(),
            ));
        }
    }
    match (&result.cate, &result.cate_se) {
        (None, _) => {}
        (Some(cate), cate_se) => {
            if cate.is_empty() || cate.iter().any(|v| !v.is_finite()) {
                return Err(IoError::Convert(
                    "cate predictions must be a non-empty finite vector".into(),
                ));
            }
            if let Some(se) = cate_se {
                if se.len() != cate.len() || se.iter().any(|v| !v.is_finite() || *v < 0.0) {
                    return Err(IoError::Convert(
                        "cate standard errors must match cate length and be finite nonnegative"
                            .into(),
                    ));
                }
            }
        }
    }
    if result.outcome_oof_r2.is_some_and(|v| !v.is_finite()) {
        return Err(IoError::Convert("outcome_oof_r2 must be finite when present".into()));
    }
    if result.treatment_oof_logloss.is_some_and(|v| !v.is_finite() || v < 0.0) {
        return Err(IoError::Convert(
            "treatment_oof_logloss must be finite and nonnegative".into(),
        ));
    }
    if result.crossfit_folds.is_some_and(|folds| folds < 2) {
        return Err(IoError::Convert("crossfit_folds must be at least two when present".into()));
    }
    if result.learner_provenance.iter().any(|(spec, implementation, version)| {
        spec.trim().is_empty() || implementation.trim().is_empty() || version.trim().is_empty()
    }) {
        return Err(IoError::Convert("learner provenance entries must be non-blank".into()));
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
        )
        .chain(
            identification
                .estimands
                .iter()
                .filter_map(|estimand| estimand.rd_design.as_ref())
                .map(|design| design.running_variable),
        )
        .chain(identification.hedge.iter().flat_map(crate::HedgeCertificateWire::variables));
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
    fn interval() -> antecedent_estimate::IdentifiedSetInterval {
        antecedent_estimate::IdentifiedSetInterval {
            level: 0.9,
            lower: 0.41,
            upper: 1.12,
            bound_lower: 0.52,
            bound_upper: 0.98,
            lower_se: 0.061,
            upper_se: 0.083,
            critical_value: 1.31,
            width_retained: true,
            completions: 2,
            replicates: 199,
            method: antecedent_estimate::IdentifiedSetIntervalMethod::ImbensManskiSharedBlock,
            truncated: false,
        }
    }

    #[test]
    fn every_identified_set_construction_has_its_own_wire_tag() {
        let mut tags = Vec::new();
        for method in antecedent_estimate::IdentifiedSetIntervalMethod::ALL {
            let original = antecedent_estimate::IdentifiedSetInterval { method, ..interval() };
            let wire = identified_set_interval_to_wire(&original).unwrap();
            assert_eq!(identified_set_interval_from_wire(&wire).unwrap().method, method);
            tags.push(serde_json::to_value(wire.method).unwrap());
        }
        tags.dedup();
        assert_eq!(tags.len(), antecedent_estimate::IdentifiedSetIntervalMethod::ALL.len());
    }

    #[test]
    fn truncated_identified_set_interval_round_trips_and_defaults_to_false() {
        let original = antecedent_estimate::IdentifiedSetInterval { truncated: true, ..interval() };
        let wire = identified_set_interval_to_wire(&original).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["truncated"], true);
        assert_eq!(identified_set_interval_from_wire(&wire).unwrap(), original);
        // An untruncated interval omits the field, and a body without it decodes
        // as untruncated.
        let plain = identified_set_interval_to_wire(&interval()).unwrap();
        let json = serde_json::to_value(&plain).unwrap();
        assert!(json.get("truncated").is_none());
        let decoded: IdentifiedSetIntervalWire = serde_json::from_value(json).unwrap();
        assert!(!identified_set_interval_from_wire(&decoded).unwrap().truncated);
    }

    fn with_structural(interval: Option<IdentifiedSetIntervalWire>) -> AnalysisResultWire {
        let mut result = fixture();
        result.structural_response = Some(StructuralResponseMixtureWire {
            weight_basis: StructuralWeightBasisWire::CompletionEnumeration,
            atoms: Vec::new(),
            identified_mass: 1.0,
            unidentified_mass: 0.0,
            unevaluable_mass: 0.0,
            subsampled_out_mass: 0.0,
            identified_set: None,
            identified_set_interval: interval,
            conditional_on_identified: None,
            full_mass_scope: true,
            truncated_atoms: 0,
        });
        result
    }

    #[test]
    fn subsampled_out_mass_is_optional_on_the_wire_and_round_trips() {
        let names: Vec<String> = vec!["a".into(), "y".into()];
        // Zero is omitted, and a body without the field decodes as zero.
        let plain = with_structural(None);
        let json = serde_json::to_value(&plain).unwrap();
        assert!(json["structural_response"].get("subsampled_out_mass").is_none());
        let decoded: AnalysisResultWire = serde_json::from_value(json).unwrap();
        assert!(decoded.structural_response.unwrap().subsampled_out_mass.abs() < f64::EPSILON);

        let mut result = with_structural(None);
        if let Some(structural) = result.structural_response.as_mut() {
            structural.identified_mass = 0.4;
            structural.subsampled_out_mass = 0.6;
        }
        let artifact = encode_analysis_result_artifact(&result, names.clone(), "sub").unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(decoded, result);

        if let Some(structural) = result.structural_response.as_mut() {
            structural.subsampled_out_mass = 1.5;
        }
        assert!(encode_analysis_result_artifact(&result, names, "bad").is_err());
    }

    #[test]
    fn identified_set_interval_round_trips_and_validates() {
        let original = interval();
        let wire = identified_set_interval_to_wire(&original).unwrap();
        assert_eq!(identified_set_interval_from_wire(&wire).unwrap(), original);
        let result = with_structural(Some(wire.clone()));
        let names = vec!["a".into(), "y".into()];
        let artifact = encode_analysis_result_artifact(&result, names.clone(), "set").unwrap();
        assert_eq!(artifact.manifest.format_version, STABLE_FORMAT);
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(decoded, result);
        let restored = decoded.structural_response.unwrap().identified_set_interval.unwrap();
        assert_eq!(identified_set_interval_from_wire(&restored).unwrap(), original);
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(
            json["structural_response"]["identified_set_interval"]["method"],
            "imbens_manski_shared_block"
        );

        for corrupt in [
            IdentifiedSetIntervalWire { level: 1.0, ..wire.clone() },
            IdentifiedSetIntervalWire { lower: 1.5, ..wire.clone() },
            IdentifiedSetIntervalWire { upper_se: -0.1, ..wire.clone() },
            IdentifiedSetIntervalWire { critical_value: f64::NAN, ..wire.clone() },
            IdentifiedSetIntervalWire { replicates: 1, ..wire.clone() },
            IdentifiedSetIntervalWire { completions: 0, ..wire.clone() },
        ] {
            assert!(identified_set_interval_from_wire(&corrupt).is_err());
            assert!(
                encode_analysis_result_artifact(
                    &with_structural(Some(corrupt)),
                    names.clone(),
                    "bad"
                )
                .is_err()
            );
        }
    }

    #[test]
    fn format_0_4_structural_result_migrates_without_an_interval() {
        // A 0.4 writer had no interval field: the body omits it entirely.
        let result = with_structural(None);
        let body = serde_json::to_value(&result).unwrap();
        assert!(body["structural_response"].get("identified_set_interval").is_none());
        let mut artifact =
            encode_analysis_result_artifact(&result, vec!["a".into(), "y".into()], "old").unwrap();
        artifact.manifest.format_version = crate::FormatVersion { major: 0, minor: 4 };
        artifact.manifest.minimum_reader_version = crate::FormatVersion { major: 0, minor: 4 };
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (migrated, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(migrated.manifest.format_version, STABLE_FORMAT);
        assert_eq!(decoded, result);
        assert!(decoded.structural_response.unwrap().identified_set_interval.is_none());
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

    #[test]
    fn validate_result_refuses_wrong_query_ids_and_keeps_the_valid_counterpart() {
        let names = vec!["a".into(), "y".into()];
        let ok = fixture();
        assert!(encode_analysis_result_artifact(&ok, names.clone(), "ok").is_ok());
        let mut wrong = ok;
        let mut query = serde_json::to_value(&wrong.identification.query).unwrap();
        query["response"]["functional"]["average_derivative"]["treatment"] = 1.into();
        query["response"]["functional"]["average_derivative"]["outcome"] = 0.into();
        wrong.identification.query = serde_json::from_value(query).unwrap();
        let err = encode_analysis_result_artifact(&wrong, names, "wrong").unwrap_err();
        assert!(err.to_string().contains("enclosing query"), "{err}");
    }

    fn pulse_query_wire(horizon: u32) -> crate::CausalQueryWire {
        use crate::query_wire::causal_query_to_wire;
        use antecedent_core::{CausalQuery, TemporalEffectQuery, VariableId};
        causal_query_to_wire(&CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_horizon_steps(horizon),
        ))
        .unwrap()
    }

    #[test]
    fn validate_result_refuses_changed_horizon_and_keeps_the_matching_certificate() {
        let names = vec!["x".into(), "y".into()];
        let mut ok = fixture();
        ok.query = pulse_query_wire(1);
        ok.identification.query = ok.query.clone();
        ok.temporal_identification.push(TemporalIdentificationWire {
            horizon: 1,
            variables: vec![
                crate::HorizonAdjustmentNodeWire { variable: 0, offset: -1 },
                crate::HorizonAdjustmentNodeWire { variable: 1, offset: 0 },
            ],
            identification: ok.identification.clone(),
        });
        assert!(encode_analysis_result_artifact(&ok, names.clone(), "horizon-ok").is_ok());
        let mut stale = ok;
        stale.temporal_identification[0].horizon = 2;
        let err = encode_analysis_result_artifact(&stale, names, "horizon-stale").unwrap_err();
        assert!(err.to_string().contains("horizon"), "{err}");
    }

    #[test]
    fn validate_result_refuses_inconsistent_atom_totals_and_keeps_unit_mass() {
        let names = vec!["a".into(), "y".into()];
        let mut ok = with_structural(None);
        ok.structural_response.as_mut().unwrap().atoms.push(StructuralResponseAtomWire {
            graph_key: 1,
            weight: 1.0,
            identification_status: crate::IdentificationStatusWire::NonparametricallyIdentified,
            value: None,
            posterior_artifact: None,
            response: None,
        });
        assert!(encode_analysis_result_artifact(&ok, names.clone(), "atoms-ok").is_ok());
        let mut bad = ok;
        bad.structural_response.as_mut().unwrap().atoms[0].weight = 0.4;
        let err = encode_analysis_result_artifact(&bad, names, "atoms-bad").unwrap_err();
        assert!(err.to_string().contains("atom totals"), "{err}");
    }

    #[test]
    fn validate_result_refuses_inconsistent_masses_and_keeps_unit_mass() {
        let names = vec!["a".into(), "y".into()];
        let ok = with_structural(None);
        assert!(encode_analysis_result_artifact(&ok, names.clone(), "mass-ok").is_ok());
        let mut bad = ok;
        bad.structural_response.as_mut().unwrap().identified_mass = 0.5;
        let err = encode_analysis_result_artifact(&bad, names, "mass-bad").unwrap_err();
        assert!(err.to_string().contains("masses must sum to one"), "{err}");
    }

    #[test]
    fn validate_result_refuses_scalar_estimate_under_not_identified() {
        let names = vec!["a".into(), "y".into()];
        let mut result = fixture();
        result.identification.status = "not_identified".into();
        assert_eq!(result.estimate, Some(1.0));
        let err = encode_analysis_result_artifact(&result, names.clone(), "uid").unwrap_err();
        assert!(err.to_string().contains("does not license a scalar estimate"), "{err}");
        result.estimate = None;
        result.standard_error = None;
        assert!(encode_analysis_result_artifact(&result, names, "uid-ok").is_ok());
    }

    fn mediation_slice() -> TemporalMediationSliceWire {
        TemporalMediationSliceWire {
            horizon: 1,
            identification_status: crate::IdentificationStatusWire::NonparametricallyIdentified,
            method: "front_door".into(),
            adjustment: vec![],
            effect: 0.5,
            total: Some(0.5),
            direct: Some(0.2),
            mediated: Some(0.3),
            uncertainty: TemporalMediationUncertaintyWire::Unavailable,
            identified_set: None,
            diagnostics: vec![],
        }
    }

    #[test]
    fn validate_result_refuses_malformed_mediation_grid() {
        let names = vec!["a".into(), "y".into()];
        let mut ok = fixture();
        ok.mediation_grid = Some(TemporalMediationGridWire {
            slices: vec![mediation_slice()],
            joint_posterior: false,
        });
        assert!(encode_analysis_result_artifact(&ok, names.clone(), "grid-ok").is_ok());

        let mut bad = ok.clone();
        bad.mediation_grid.as_mut().unwrap().slices[0].effect = f64::NAN;
        let err = encode_analysis_result_artifact(&bad, names.clone(), "grid-nan").unwrap_err();
        assert!(err.to_string().contains("mediation grid"), "{err}");

        let mut inverted = ok;
        inverted.mediation_grid.as_mut().unwrap().slices[0].identified_set = Some([1.0, 0.0]);
        let err = encode_analysis_result_artifact(&inverted, names, "grid-set").unwrap_err();
        assert!(err.to_string().contains("identified set"), "{err}");
    }

    #[test]
    fn validate_result_refuses_malformed_cate() {
        let names = vec!["a".into(), "y".into()];
        let mut ok = fixture();
        ok.cate = Some(vec![0.1, 0.2]);
        ok.cate_se = Some(vec![0.05, 0.04]);
        assert!(encode_analysis_result_artifact(&ok, names.clone(), "cate-ok").is_ok());

        let mut bad_len = ok.clone();
        bad_len.cate_se = Some(vec![0.05]);
        let err = encode_analysis_result_artifact(&bad_len, names.clone(), "cate-len").unwrap_err();
        assert!(err.to_string().contains("cate"), "{err}");

        let mut bad_dispersion = ok.clone();
        bad_dispersion.cate_leaf_dispersion = Some(vec![0.05, -1.0]);
        let err = encode_analysis_result_artifact(&bad_dispersion, names.clone(), "cate-disp")
            .unwrap_err();
        assert!(err.to_string().contains("leaf dispersion"), "{err}");

        let mut bad_nan = ok;
        bad_nan.cate = Some(vec![0.1, f64::NAN]);
        let err = encode_analysis_result_artifact(&bad_nan, names, "cate-nan").unwrap_err();
        assert!(err.to_string().contains("cate"), "{err}");
    }
}
