//! Python bridge for the revalidation of transported claims.
//!
//! Python declares the scene (target, sources with the identity of their evidence, transported
//! claims built from transported responses, decisions resting on claims) and the observed
//! effects; the identities, the classification into `still_valid`, `recompute` and `withdrawn`,
//! the rule that a calibration record is never inherited by another target, the sealed
//! verification and every refusal are Rust's. A refusal comes back as structured JSON for the
//! Python layer to raise as a typed exception.
//!
//! No calibration record is declared or measured here: a record is a digest the caller holds.

use std::collections::BTreeMap;

use antecedent::analysis::transport_revalidation::{
    SceneCalibration, TransportClaimState, TransportRevalidationRefusal,
    TransportRevalidationReport, TransportScene, TransportSceneBuilder, response_coordinate,
    revalidate, transported_claim_node, transported_claim_standing,
};
use antecedent_contract::ProviderTrust;
use antecedent_contract::composition_verification::{
    ObservedEffect, SupportStanding, TrustStanding,
};
use antecedent_contract::evidence_ledger::parse_digest_hex;
use antecedent_core::SemanticDigest;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::{Value as Json, json};

use crate::bridge::{RefusalJson, check_declared_size, external_refusal_json};
use crate::transported_response_api::TransportedResponseHandle;

type Refused<T> = (Option<T>, Option<String>);

const MAX_JSON_BYTES: usize = crate::bridge::MAX_DECLARED_ARTIFACT_BYTES;
const STAGE: &str = "transport_revalidation";

fn check_size(text: &str, what: &str) -> PyResult<()> {
    check_declared_size(text, MAX_JSON_BYTES, what)
}

fn refusal_json(error: &TransportRevalidationRefusal) -> String {
    RefusalJson::new(error.code, STAGE, &error.detail, &error.message).remedy(error.remedy).json()
}

fn digest(value: &str, what: &str) -> PyResult<SemanticDigest> {
    parse_digest_hex(value)
        .ok_or_else(|| crate::value_err(format!("{what} must be 64 hexadecimal characters")))
}

// ---------------------------------------------------------------------------
// Scene.
// ---------------------------------------------------------------------------

/// A declared scene: one target, its sources, and the claims and decisions about it.
#[pyclass(name = "TransportSceneHandle", skip_from_py_object)]
pub(crate) struct TransportSceneHandle {
    inner: TransportScene,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum CalibrationIn {
    PointOnly,
    Required,
    Held {
        record: String,
        #[serde(default)]
        established_at_target: Option<String>,
    },
    Inherited {
        from: String,
        record: String,
        #[serde(default)]
        established_at_target: Option<String>,
    },
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ClaimIn {
    Transported {
        key: String,
        response: usize,
        calibration: CalibrationIn,
        #[serde(default)]
        trust: Option<String>,
    },
    TransportedClaim {
        key: String,
        response: usize,
        calibration: CalibrationIn,
        #[serde(default)]
        provider_trust: Option<String>,
    },
    Decision {
        key: String,
        identity: String,
        quantity: String,
        rests_on: Vec<String>,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SceneIn {
    target_population: String,
    target_data_identity: String,
    #[serde(default)]
    unsupported_target: Option<Vec<String>>,
    sources: Vec<(String, String)>,
    claims: Vec<ClaimIn>,
}

fn calibration_of(wire: CalibrationIn, own_target: SemanticDigest) -> PyResult<SceneCalibration> {
    let at = |value: Option<String>| -> PyResult<SemanticDigest> {
        match value {
            None => Ok(own_target),
            Some(text) => digest(&text, "established_at_target"),
        }
    };
    Ok(match wire {
        CalibrationIn::PointOnly => SceneCalibration::PointOnly,
        CalibrationIn::Required => SceneCalibration::Required,
        CalibrationIn::Held { record, established_at_target } => SceneCalibration::Held {
            record: digest(&record, "calibration record")?,
            established_at_target: at(established_at_target)?,
        },
        CalibrationIn::Inherited { from, record, established_at_target } => {
            SceneCalibration::Inherited {
                from,
                record: digest(&record, "calibration record")?,
                established_at_target: at(established_at_target)?,
            }
        }
    })
}

fn provider_of(name: Option<&str>) -> PyResult<Option<ProviderTrust>> {
    name.map(|name| {
        ProviderTrust::from_wire(name)
            .ok_or_else(|| crate::value_err(format!("unknown provider trust level `{name}`")))
    })
    .transpose()
}

/// The coordinate a transported response is evaluated at outside any scene: a calibration
/// record established for exactly this response at its target is declared at it.
#[pyfunction]
fn transported_response_coordinate(
    response: PyRef<'_, TransportedResponseHandle>,
) -> Refused<String> {
    match response_coordinate(&response.inner) {
        Ok(coordinate) => (Some(coordinate.to_hex()), None),
        Err(error) => (None, Some(refusal_json(&error))),
    }
}

/// The standing of a standalone transported response (support, provenance trust and the
/// declared calibration) as JSON.
#[pyfunction]
#[pyo3(signature = (response, calibration_json, provider_trust=None))]
fn transported_response_standing(
    response: PyRef<'_, TransportedResponseHandle>,
    calibration_json: &str,
    provider_trust: Option<&str>,
) -> PyResult<Refused<String>> {
    let calibration = standalone_calibration(calibration_json)?;
    Ok(
        match transported_claim_standing(
            &response.inner,
            calibration.clone(),
            provider_of(provider_trust)?,
        ) {
            Ok(standing) => {
                let support = match &standing.support {
                    SupportStanding::Supported => json!({"kind": "supported", "coordinates": []}),
                    SupportStanding::Unsupported { coordinates } => {
                        json!({"kind": "unsupported", "coordinates": coordinates})
                    }
                };
                let wire = json!({
                    "support": support,
                    "trust": standing.trust.as_str(),
                    "calibration": crate::composition_verification_api::calibration_wire(&calibration),
                });
                (Some(wire.to_string()), None)
            }
            Err(error) => (None, Some(refusal_json(&error))),
        },
    )
}

/// The composition-verification node of a standalone transported response, in the node
/// declaration form the composition graph reads.
#[pyfunction]
#[pyo3(signature = (key, response, calibration_json, provider_trust=None))]
fn transported_response_node(
    key: &str,
    response: PyRef<'_, TransportedResponseHandle>,
    calibration_json: &str,
    provider_trust: Option<&str>,
) -> PyResult<Refused<String>> {
    let calibration = standalone_calibration(calibration_json)?;
    Ok(
        match transported_claim_node(
            key,
            &response.inner,
            calibration,
            provider_of(provider_trust)?,
        ) {
            Ok(node) => {
                (Some(crate::composition_verification_api::node_wire(&node).to_string()), None)
            }
            Err(error) => (None, Some(refusal_json(&error))),
        },
    )
}

fn standalone_calibration(
    text: &str,
) -> PyResult<antecedent_contract::composition_verification::CalibrationDeclaration> {
    check_size(text, "calibration declaration")?;
    let parsed: Json = serde_json::from_str(text)
        .map_err(|e| crate::value_err(format!("invalid calibration declaration: {e}")))?;
    crate::composition_verification_api::parse_calibration(&parsed)
}

/// The identity a scene gives its target, from the population and the content identity of its
/// declaration and data. A calibration record established at this target names it.
#[pyfunction]
fn transport_scene_target_identity(
    target_population: &str,
    target_data_identity: &str,
) -> PyResult<String> {
    let data = digest(target_data_identity, "target_data_identity")?;
    Ok(TransportSceneBuilder::new(target_population, data).target_identity().to_hex())
}

/// Build a scene from its declaration and the transported responses its claims are about.
#[pyfunction]
fn transport_scene_build(
    scene_json: &str,
    responses: Vec<PyRef<'_, TransportedResponseHandle>>,
) -> PyResult<Refused<TransportSceneHandle>> {
    check_size(scene_json, "transport scene")?;
    let wire: SceneIn = serde_json::from_str(scene_json)
        .map_err(|e| crate::value_err(format!("invalid transport scene: {e}")))?;
    let mut builder = TransportSceneBuilder::new(
        wire.target_population.clone(),
        digest(&wire.target_data_identity, "target_data_identity")?,
    );
    let own_target = builder.target_identity();
    if let Some(coordinates) = wire.unsupported_target {
        builder = builder.target_support(SupportStanding::Unsupported { coordinates });
    }
    for (population, evidence) in wire.sources {
        builder = builder.source(population, digest(&evidence, "source evidence identity")?);
    }
    for claim in wire.claims {
        match claim {
            ClaimIn::Transported { key, response, calibration, trust } => {
                let held = responses.get(response).ok_or_else(|| {
                    crate::value_err(format!("claim `{key}` names response {response}"))
                })?;
                let trust = match trust {
                    None => TrustStanding::NotApplicable,
                    Some(name) => TrustStanding::from_name(&name).ok_or_else(|| {
                        crate::value_err(format!("unknown trust standing `{name}`"))
                    })?,
                };
                builder = builder.transported(
                    key,
                    &held.inner,
                    calibration_of(calibration, own_target)?,
                    trust,
                );
            }
            ClaimIn::TransportedClaim { key, response, calibration, provider_trust } => {
                let held = responses.get(response).ok_or_else(|| {
                    crate::value_err(format!("claim `{key}` names response {response}"))
                })?;
                builder = builder.transported_claim(
                    key,
                    &held.inner,
                    calibration_of(calibration, own_target)?,
                    provider_of(provider_trust.as_deref())?,
                );
            }
            ClaimIn::Decision { key, identity, quantity, rests_on } => {
                let rests: Vec<&str> = rests_on.iter().map(String::as_str).collect();
                builder = builder.decision(
                    key,
                    digest(&identity, "decision identity")?,
                    digest(&quantity, "decision quantity")?,
                    &rests,
                );
            }
        }
    }
    Ok(match builder.build() {
        Ok(inner) => (Some(TransportSceneHandle { inner }), None),
        Err(error) => (None, Some(refusal_json(&error))),
    })
}

#[pymethods]
impl TransportSceneHandle {
    /// The scene's identities and claims as JSON.
    fn summary_json(&self) -> String {
        let scene = &self.inner;
        json!({
            "target_identity": scene.target.identity.to_hex(),
            "target_supported": matches!(scene.target.support, SupportStanding::Supported),
            "selection_diagram": scene.selection_diagram.to_hex(),
            "sources": scene.sources.iter().map(|s| json!({
                "key": s.key, "identity": s.identity.to_hex(),
            })).collect::<Vec<_>>(),
            "claims": scene.claims.iter().map(|c| json!({
                "key": c.key,
                "kind": format!("{:?}", c.kind).to_lowercase(),
                "identity": c.identity.to_hex(),
                "quantity": c.quantity.to_hex(),
                "structure": c.structure.map(|d| d.to_hex()),
                "sources": c.sources,
                "rests_on": c.rests_on,
                "trust": c.trust.as_str(),
            })).collect::<Vec<_>>(),
        })
        .to_string()
    }
}

// ---------------------------------------------------------------------------
// Revalidation.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EffectIn {
    key: String,
    kind: String,
    text: String,
}

fn effects_of(text: Option<&str>) -> PyResult<BTreeMap<String, ObservedEffect>> {
    let mut out = BTreeMap::new();
    let Some(text) = text else { return Ok(out) };
    check_size(text, "observed effects")?;
    let items: Vec<EffectIn> = serde_json::from_str(text)
        .map_err(|e| crate::value_err(format!("invalid observed effects: {e}")))?;
    for item in items {
        let effect = match item.kind.as_str() {
            "recomputed" => ObservedEffect::Recomputed { cause: item.text },
            "withdrawn" => ObservedEffect::Withdrawn { cause: item.text },
            "refused" => ObservedEffect::Refused { detail: item.text },
            other => return Err(crate::value_err(format!("unknown effect kind `{other}`"))),
        };
        if out.insert(item.key, effect).is_some() {
            return Err(crate::value_err("one effect per node"));
        }
    }
    Ok(out)
}

/// A scene's classification against the earlier scene.
#[pyclass(name = "TransportRevalidationHandle", skip_from_py_object)]
pub(crate) struct TransportRevalidationHandle {
    inner: TransportRevalidationReport,
}

fn inspection_json(report: &TransportRevalidationReport) -> Json {
    let inspection = report.inspect();
    json!({
        "claims": inspection.claims.iter().map(|c| json!({
            "key": c.key,
            "kind": c.kind,
            "class": c.class,
            "reasons": c.reasons,
            "remedies": c.remedies,
            "prior_authority": c.prior_authority,
            "calibration": c.calibration,
            "calibration_withdrawn": c.calibration_withdrawn,
        })).collect::<Vec<_>>(),
        "removed": inspection.removed,
        "identity": inspection.identity,
        "verification_identity": report.verification().identity(),
    })
}

#[pymethods]
impl TransportRevalidationHandle {
    /// Every claim's class and reasons in transport terms, as JSON.
    fn inspect_json(&self) -> String {
        inspection_json(&self.inner).to_string()
    }

    /// The claim when it is usable as it stands: `(claim json, refusal json)`.
    fn require_valid(&self, key: &str) -> Refused<String> {
        match self.inner.require_valid(key) {
            Ok(_) => {
                let inspection = inspection_json(&self.inner);
                let claim = inspection["claims"]
                    .as_array()
                    .and_then(|claims| claims.iter().find(|c| c["key"] == key))
                    .cloned();
                (claim.map(|c| c.to_string()), None)
            }
            Err(refusal) => (None, Some(external_refusal_json(&refusal))),
        }
    }

    /// Identity of the classification (hex).
    fn identity(&self) -> String {
        self.inner.identity()
    }

    /// The claim-level verification sealed as `composition_verification_v1`.
    fn export(&self, py: Python<'_>, artifact_id: &str) -> Refused<Py<PyBytes>> {
        match self.inner.export(artifact_id) {
            Ok(bytes) => (Some(PyBytes::new(py, &bytes).unbind()), None),
            Err(error) => (None, Some(refusal_json(&error))),
        }
    }
}

/// Classify every claim of `current` against `previous` (a first verification when absent).
#[pyfunction]
#[pyo3(signature = (previous, current, effects_json=None))]
fn transport_revalidate(
    previous: Option<PyRef<'_, TransportSceneHandle>>,
    current: PyRef<'_, TransportSceneHandle>,
    effects_json: Option<&str>,
) -> PyResult<Refused<TransportRevalidationHandle>> {
    let effects = effects_of(effects_json)?;
    Ok(match revalidate(previous.as_ref().map(|p| &p.inner), &current.inner, &effects) {
        Ok(inner) => (Some(TransportRevalidationHandle { inner }), None),
        Err(error) => (None, Some(refusal_json(&error))),
    })
}

/// The current scene of a set of transported claims, advanced only by a successful
/// revalidation.
#[pyclass(name = "TransportClaimStateHandle")]
pub(crate) struct TransportClaimStateHandle {
    inner: TransportClaimState,
}

#[pymethods]
impl TransportClaimStateHandle {
    #[new]
    fn new() -> Self {
        Self { inner: TransportClaimState::new() }
    }

    /// Classify `current` against the recorded scene and, only on success, record it.
    #[pyo3(signature = (current, effects_json=None))]
    fn observe(
        &mut self,
        current: PyRef<'_, TransportSceneHandle>,
        effects_json: Option<&str>,
    ) -> PyResult<Refused<TransportRevalidationHandle>> {
        let effects = effects_of(effects_json)?;
        Ok(match self.inner.observe(current.inner.clone(), &effects) {
            Ok(inner) => (Some(TransportRevalidationHandle { inner }), None),
            Err(error) => (None, Some(refusal_json(&error))),
        })
    }

    /// The recorded scene, when an observation succeeded.
    fn scene(&self) -> Option<TransportSceneHandle> {
        self.inner.scene().map(|scene| TransportSceneHandle { inner: scene.clone() })
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TransportSceneHandle>()?;
    module.add_class::<TransportRevalidationHandle>()?;
    module.add_class::<TransportClaimStateHandle>()?;
    module.add_function(wrap_pyfunction!(transport_scene_target_identity, module)?)?;
    module.add_function(wrap_pyfunction!(transport_scene_build, module)?)?;
    module.add_function(wrap_pyfunction!(transport_revalidate, module)?)?;
    module.add_function(wrap_pyfunction!(transported_response_coordinate, module)?)?;
    module.add_function(wrap_pyfunction!(transported_response_standing, module)?)?;
    module.add_function(wrap_pyfunction!(transported_response_node, module)?)?;
    Ok(())
}

#[cfg(test)]
mod strict_scene_input_tests {
    use super::{CalibrationIn, ClaimIn};

    #[test]
    fn calibration_and_claim_variants_reject_unknown_keys() {
        for text in [
            r#"{"kind":"point_only"}"#,
            r#"{"kind":"required"}"#,
            r#"{"kind":"held","record":"x","established_at_target":"y"}"#,
            r#"{"kind":"inherited","from":"x","record":"y","established_at_target":"z"}"#,
        ] {
            assert!(serde_json::from_str::<CalibrationIn>(text).is_ok());
        }
        for text in [
            r#"{"kind":"held","record":"x","established_at":"y"}"#,
            r#"{"kind":"inherited","from":"x","record":"y","established_at":"z"}"#,
        ] {
            assert!(serde_json::from_str::<CalibrationIn>(text).is_err());
        }
        for text in [
            r#"{"kind":"transported","key":"x","response":0,"calibration":{"kind":"point_only"},"trust":"verified"}"#,
            r#"{"kind":"transported_claim","key":"x","response":0,"calibration":{"kind":"required"},"provider_trust":"verified"}"#,
            r#"{"kind":"decision","key":"x","identity":"y","quantity":"z","rests_on":[]}"#,
        ] {
            assert!(serde_json::from_str::<ClaimIn>(text).is_ok());
        }
        for text in [
            r#"{"kind":"transported","key":"x","response":0,"calibration":{"kind":"point_only"},"provider_trust":"verified"}"#,
            r#"{"kind":"transported_claim","key":"x","response":0,"calibration":{"kind":"required"},"trust":"verified"}"#,
        ] {
            assert!(serde_json::from_str::<ClaimIn>(text).is_err());
        }
    }
}
