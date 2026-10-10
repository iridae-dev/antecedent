//! Source-backed projections for a portable composed workflow.
//!
//! These projections replay an original artifact consumer. Their identity binds
//! the whole original source, exact coordinates and affine decision expression.
//! They do not authenticate observations, issue native authority, or license
//! uncertainty. Native unresolved operations require a separate actual executor.

use antecedent_core::{ScientificQuantity, SemanticDigest};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::error::IoError;
use antecedent_io::external_claim_artifact::{
    ExternalClaimArtifact, ExternalClaimMetadata, ExternalClaimTrust,
    MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::reader::ArtifactReader;
use antecedent_io::response_wire::{
    ResponseIdentificationWire, ResponseValueWire, SupportStatusWire,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::composition_boundary::{
    DecisionInput, TrustEvidence, TrustRequirement, evaluate_functional_on_input,
};
use crate::composition_bundle::NodeKind;
use crate::decision_artifact::contract_from_json;
use crate::decision_contract::{DecisionContract, DecisionFunctional};
use crate::decision_eval::{MeanSource, evaluate_contract_on_means};
use crate::source_evidence::SourceEvidence;

/// Maximum envelope size accepted before decode.
pub const MAX_SOURCE_PROJECTION_BYTES: usize = 18 * 1024 * 1024;
const MAGIC: &[u8] = b"ANTE-SOURCE-PROJECTION-1\0";

/// An original scientific source; importing one supplies no live authority.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ProjectionSource {
    /// Original contracted native response, including its unresolved dependencies.
    NativeResponse(Vec<u8>),
    /// Original checked external grid and descriptive attestation/receipt standing.
    ExternalClaim(Vec<u8>),
}

/// A bounded semantic projection using an original source operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceProjection {
    /// Original graph/query/identified execution contract; no recreated executable state.
    CausalContract,
    /// Original external attestation/request verification; never native authority.
    Attestation,
    /// Exact original scientific coordinate set, without unit conversion.
    QuantityCoordinates,
    /// Actual affine mean-to-utility evaluation by the original decision engine.
    AffineDecision {
        /// Original closed decision contract JSON, including caller-declared utility units.
        contract: String,
    },
    /// One actual affine expectation; unrelated unsupported actions remain outside its scope.
    AffineFunctional {
        /// Full original decision declaration, preserving all original scientific inputs.
        contract: String,
        /// Exact selected original semantic action identity.
        action_id: String,
    },
}

/// Scientific information independently reconstructed from the original source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionReport {
    /// Original native container digest, or BLAKE3 of the complete external source.
    pub source_digest: String,
    /// Actual original scientific coordinates, in original value order.
    pub quantities: Vec<ScientificQuantityWire>,
    /// Original source snapshot identity.
    pub snapshot: String,
    /// Original unresolved operations; a portable projection does not resolve them.
    pub unresolved: Vec<String>,
    /// Descriptive original source standing, without an authority upgrade.
    pub trust: String,
    /// Recomputed semantic projection and numerical utility outputs.
    pub output: Value,
    /// Explicit interpretation of this projection's scientific standing.
    pub scope: String,
}

/// Versioned original-source projection with independent consumer replay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceProjectionArtifact {
    version: u16,
    source: ProjectionSource,
    projection: SourceProjection,
    report: ProjectionReport,
    identity: String,
}

#[derive(Clone, Debug)]
struct OriginalSource {
    digest: String,
    quantities: Vec<ScientificQuantityWire>,
    means: Vec<f64>,
    supported: Vec<bool>,
    snapshot: String,
    contract: Value,
    attestation: Option<Value>,
    unresolved: Vec<String>,
    trust: String,
}

impl SourceProjectionArtifact {
    /// Reconstruct the source through its original consumer and evaluate the projection.
    /// This does not resolve any missing original native operation.
    pub fn produce(
        source: ProjectionSource,
        projection: SourceProjection,
    ) -> Result<Self, IoError> {
        let report = reconstruct(&source, &projection)?;
        let mut artifact = Self { version: 1, source, projection, report, identity: String::new() };
        artifact.identity = artifact.recomputed_identity()?;
        Ok(artifact)
    }

    /// Export the complete original source and bound projection.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend(to_cbor(self).map_err(original_refusal)?);
        if bytes.len() > MAX_SOURCE_PROJECTION_BYTES {
            return Err(invalid_projection("source_projection.limits_exceeded"));
        }
        Ok(bytes)
    }

    /// Read under an independently retained identity and replay the original source.
    pub fn consume(bytes: &[u8], expected_identity: &str) -> Result<Self, IoError> {
        let artifact = Self::inspect(bytes)?;
        if artifact.identity != expected_identity {
            return Err(invalid_projection("source_projection.identity_mismatch"));
        }
        Ok(artifact)
    }

    /// Reconstruct portable semantic outputs without supplying an execution resolver.
    /// The resulting report retains every unresolved original operation.
    pub fn inspect(bytes: &[u8]) -> Result<Self, IoError> {
        if bytes.len() > MAX_SOURCE_PROJECTION_BYTES {
            return Err(invalid_projection("source_projection.limits_exceeded"));
        }
        let body = bytes
            .strip_prefix(MAGIC)
            .ok_or_else(|| invalid_projection("source_projection.invalid_artifact"))?;
        let artifact: Self = from_cbor(body)
            .map_err(|_| invalid_projection("source_projection.invalid_artifact"))?;
        if artifact.version != 1 || artifact.identity != artifact.recomputed_identity()? {
            return Err(invalid_projection("source_projection.identity_mismatch"));
        }
        if artifact.report != reconstruct(&artifact.source, &artifact.projection)? {
            return Err(invalid_projection("source_projection.replay_mismatch"));
        }
        Ok(artifact)
    }

    /// Original source bytes for a separately licensed execution resolver.
    #[must_use]
    pub fn original_bytes(&self) -> &[u8] {
        match &self.source {
            ProjectionSource::NativeResponse(bytes) | ProjectionSource::ExternalClaim(bytes) => {
                bytes
            }
        }
    }
    /// Whether the source is an original native response requiring actual execution resolution.
    #[must_use]
    pub fn is_native(&self) -> bool {
        matches!(&self.source, ProjectionSource::NativeResponse(_))
    }

    /// Independently reconstructed report, including unresolved original dependencies.
    #[must_use]
    pub fn report(&self) -> &ProjectionReport {
        &self.report
    }
    /// The bundle node kind whose original source operation this artifact projects.
    #[must_use]
    pub fn node_kind(&self) -> NodeKind {
        match &self.projection {
            SourceProjection::CausalContract => NodeKind::CausalContract,
            SourceProjection::Attestation => NodeKind::Attestation,
            SourceProjection::QuantityCoordinates => NodeKind::QuantityCoordinates,
            SourceProjection::AffineDecision { .. } | SourceProjection::AffineFunctional { .. } => {
                NodeKind::Transformation
            }
        }
    }
    /// The projected semantic operation.
    #[must_use]
    pub fn projection(&self) -> &SourceProjection {
        &self.projection
    }
    /// Whole-source and projection identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }
    fn recomputed_identity(&self) -> Result<String, IoError> {
        let bytes = to_cbor(&(self.version, &self.source, &self.projection, &self.report))
            .map_err(original_refusal)?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

fn reconstruct(
    source: &ProjectionSource,
    projection: &SourceProjection,
) -> Result<ProjectionReport, IoError> {
    let original = match source {
        ProjectionSource::NativeResponse(bytes) => native_source(bytes)?,
        ProjectionSource::ExternalClaim(bytes) => external_source(bytes)?,
    };
    let output = match projection {
        SourceProjection::CausalContract
            if matches!(source, ProjectionSource::NativeResponse(_)) =>
        {
            original.contract.clone()
        }
        SourceProjection::CausalContract => {
            return Err(invalid_projection("source_projection.causal_source_unavailable"));
        }
        SourceProjection::Attestation => original
            .attestation
            .clone()
            .ok_or_else(|| invalid_projection("source_projection.attestation_unavailable"))?,
        SourceProjection::QuantityCoordinates => json!({"quantities":original.quantities}),
        SourceProjection::AffineDecision { contract } => affine_output(&original, contract, None)?,
        SourceProjection::AffineFunctional { contract, action_id } => {
            affine_output(&original, contract, Some(action_id))?
        }
    };
    Ok(ProjectionReport {
        source_digest: original.digest,
        quantities: original.quantities,
        snapshot: original.snapshot,
        unresolved: original.unresolved,
        trust: original.trust,
        output,
        scope: "source-backed semantic/numerical replay; no source authentication, executable authority, unit conversion, distribution or calibration license".into(),
    })
}

fn native_source(bytes: &[u8]) -> Result<OriginalSource, IoError> {
    let evidence = SourceEvidence::from_result(bytes).map_err(original_refusal)?;
    let original = evidence.original_consumption().map_err(original_refusal)?;
    let response = original
        .body
        .response
        .as_ref()
        .ok_or_else(|| invalid_projection("source_projection.response_unavailable"))?;
    let ResponseIdentificationWire::PointIdentified(ResponseValueWire::Surface { mean, .. }) =
        &response.estimate
    else {
        return Err(invalid_projection("source_projection.mean_grid_required"));
    };
    let native_contract = original.contract.as_ref().expect("original checked contract");
    let snapshot = SemanticDigest::from_bytes(native_contract.identities.data_snapshot).to_hex();
    let quantities =
        evidence.coordinates().iter().map(ScientificQuantityWire::from).collect::<Vec<_>>();
    if quantities.len() > 4096
        || mean.len() != quantities.len()
        || mean.iter().any(|value| !value.is_finite())
    {
        return Err(invalid_projection("source_projection.mean_binding_mismatch"));
    }
    Ok(OriginalSource {
        digest: SemanticDigest::from_bytes(original.artifact_digest).to_hex(),
        quantities,
        means: mean.clone(),
        supported: response.support.point_status.as_ref().map_or_else(
            || vec![response.support.status == SupportStatusWire::Supported; mean.len()],
            |statuses| {
                statuses.iter().map(|status| *status == SupportStatusWire::Supported).collect()
            },
        ),
        snapshot,
        contract: json!({"contract":original.contract,"identification":original.body.identification,"query":original.body.query}),
        attestation: None,
        unresolved: evidence.unresolved().iter().map(ToString::to_string).collect(),
        trust: "unverified_imported_native_result".into(),
    })
}

fn external_source(bytes: &[u8]) -> Result<OriginalSource, IoError> {
    if bytes.len() > MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES {
        return Err(invalid_projection("source_projection.limits_exceeded"));
    }
    let mut reader =
        ArtifactReader::open_seek(std::io::Cursor::new(bytes)).map_err(original_refusal)?;
    let declared = reader
        .manifest()
        .sections
        .iter()
        .try_fold(0_u64, |total, section| total.checked_add(section.uncompressed_size))
        .ok_or_else(|| invalid_projection("source_projection.limits_exceeded"))?;
    if declared > MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES as u64 {
        return Err(invalid_projection("source_projection.limits_exceeded"));
    }
    let meta = reader.load_section("external_claim.meta").map_err(original_refusal)?;
    let metadata: ExternalClaimMetadata = from_cbor(meta.as_bytes()).map_err(original_refusal)?;
    let claim =
        ExternalClaimArtifact::from_bytes(bytes, &metadata.identity).map_err(original_refusal)?;
    let identity = &claim.metadata().identity;
    if identity.quantities.len() > 4096 {
        return Err(invalid_projection("source_projection.limits_exceeded"));
    }
    Ok(OriginalSource {
        digest: blake3::hash(bytes).to_hex().to_string(),
        quantities: identity.quantities.clone(),
        means: claim.values().to_vec(),
        supported: identity.point_status.iter().map(|status| status == "supported").collect(),
        snapshot: format!("{}|{}", identity.snapshot_id, identity.causal_contract_id),
        contract: json!({"causal_contract_id":identity.causal_contract_id,"graph_id":identity.graph_id,"identification":identity.identification}),
        attestation: Some(
            json!({"provider_id":identity.provider_id,"object_id":identity.object_id,"request_id":identity.request_id,"provider_fingerprint":identity.provider_fingerprint,"trust":identity.trust,"verification":identity.verification,"native_authority_issued":false}),
        ),
        unresolved: vec![],
        trust: match identity.trust {
            ExternalClaimTrust::ExternallyAttested => "externally_attested",
            ExternalClaimTrust::ExactRequestVerified => "verified_extension",
        }
        .into(),
    })
}

fn affine_output(
    source: &OriginalSource,
    json: &str,
    selected: Option<&str>,
) -> Result<Value, IoError> {
    if json.len() > 1024 * 1024
        || source.quantities.len() > 4096
        || selected.is_some_and(|action| action.len() > 256)
    {
        return Err(invalid_projection("source_projection.limits_exceeded"));
    }
    let contract = contract_from_json(json).map_err(original_refusal)?;
    if contract.actions.len() > 256
        || contract.actions.iter().any(|action| action.inputs.len() > 4096)
    {
        return Err(invalid_projection("source_projection.limits_exceeded"));
    }
    let coordinates = source
        .quantities
        .iter()
        .cloned()
        .map(ScientificQuantity::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(original_refusal)?;
    for quantity in
        contract.actions.iter().filter(|action| selected.is_none_or(|id| action.id == id)).flat_map(
            |action| {
                action
                    .utility
                    .inputs_used()
                    .into_iter()
                    .filter_map(move |index| action.inputs.get(index))
            },
        )
    {
        let index = coordinates
            .iter()
            .position(|source| source.require_same_coordinate(quantity).is_ok())
            .ok_or_else(|| invalid_projection("source_projection.coordinate_mismatch"))?;
        if !source.supported[index] {
            return Err(invalid_projection("source_projection.unsupported_coordinate"));
        }
    }
    let means = MeanSource {
        coordinates,
        means: source.means.clone(),
        provider_id: source.digest.clone(),
        snapshot_id: source.snapshot.clone(),
        causal_contract_id: source.digest.clone(),
        rng_id: "no_sampling".into(),
    };
    if let Some(action_id) = selected {
        return affine_functional_output(&contract, action_id, means, &source.supported);
    }
    let result =
        evaluate_contract_on_means(&contract, &means).map_err(|error| IoError::Refused {
            code: antecedent_core::reason_code!("decision_contract_unsatisfied"),
            message: format!("source_projection.affine_contract_refused: {error:?}"),
        })?;
    Ok(json!({"contract_identity":result.contract_identity,"utility_units":contract.utility_units,
        "actions":result.actions.iter().map(|action| json!({"id":action.id,"expected_utility":action.expected_utility,"criterion_value":action.value,"standard_error":action.standard_error})).collect::<Vec<_>>(),
        "point_only":true,"calibration":"unmeasured","caller_declared_utility_units":true}))
}

fn affine_functional_output(
    contract: &DecisionContract,
    action_id: &str,
    means: MeanSource,
    supported: &[bool],
) -> Result<Value, IoError> {
    let statuses = supported
        .iter()
        .map(|supported| {
            if *supported {
                antecedent_core::SupportStatus::Supported
            } else {
                antecedent_core::SupportStatus::OutsideEmpiricalSupport
            }
        })
        .collect::<Vec<_>>();
    let input = DecisionInput::from_mean_source(
        "original_source",
        means,
        &statuses,
        &TrustEvidence::None,
        TrustRequirement::Unrestricted,
    )
    .map_err(|cause| IoError::Refused {
        code: antecedent_core::reason_code!("decision_contract_unsatisfied"),
        message: format!("source_projection.affine_contract_refused: {cause:?}"),
    })?;
    let result = evaluate_functional_on_input(
        contract,
        action_id,
        DecisionFunctional::Expectation,
        &input,
        antecedent_core::SupportStatus::Supported,
    )
    .map_err(|cause| IoError::Refused {
        code: antecedent_core::reason_code!("decision_contract_unsatisfied"),
        message: format!("source_projection.affine_contract_refused: {cause:?}"),
    })?;
    Ok(
        json!({"contract_identity":contract.identity().map_err(|cause| IoError::Refused { code:antecedent_core::reason_code!("decision_contract_unsatisfied"),message:format!("source_projection.affine_contract_refused: {cause:?}") })?,"utility_units":contract.utility_units,
            "action_id":action_id,"value":result.value,"standard_error":result.standard_error,
            "actions":[{"id":action_id,"expected_utility":result.value,"criterion_value":result.value,"standard_error":result.standard_error}],
            "point_only":true,"calibration":"unmeasured","caller_declared_utility_units":true}),
    )
}

fn invalid_projection(detail: &'static str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: detail.into(),
    }
}
fn original_refusal(error: impl std::fmt::Display) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("source_projection.original_consumer_refused: {error}"),
    }
}
