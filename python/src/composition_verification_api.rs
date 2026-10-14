//! Python bridge for composition verification.
//!
//! Python declares the earlier and current claim graph, the observed effects
//! and the shared-data relations it read from its evidence ledger as JSON; the
//! graph checks, the classification, the lineage, the identities, the artifact
//! and every refusal are Rust's. A refusal comes back as structured JSON for
//! the Python layer to raise as a typed exception; a malformed declaration is
//! an input error.
//!
//! No interval, posterior or calibration record is produced.

use std::collections::BTreeMap;

use antecedent_contract::composition_verification::{
    CalibrationDeclaration, CalibrationLineage, ClaimRevalidation, CompositionGraph,
    CompositionNode, CompositionView, DependencyKind, GraphError, LineageEntry, NodeKind,
    ObservedEffect, PriorAuthority, SharedDataRelation, SupportStanding, TrustStanding,
};
use antecedent_contract::evidence_ledger::{SourceId, parse_digest_hex};
use antecedent_core::SemanticDigest;
use antecedent_io::composition_verification_artifact::{
    CompositionVerificationArtifact, CompositionVerificationArtifactError,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Map, Value as Json, json};

use crate::measurement_model_api::{list, member, object, opt_text, parse, present, text};

const STAGE: &str = "composition_verification";
const RELATIONS: [&str; 4] =
    ["shared_data", "shared_prior", "shared_fitted_model", "unknown_dependence"];

// ---------------------------------------------------------------------------
// Refusals as data.
// ---------------------------------------------------------------------------

fn remedy_for(detail: &str) -> &'static str {
    match detail {
        "composition_verification.identity_mismatch" => {
            "supply the artifact whose recomputed identity equals the retained identity, or \
             regenerate it from the intended graphs"
        }
        "composition_verification.replay_mismatch" => {
            "regenerate the artifact from the graphs and observed effects; the stored \
             classification and lineage identities must replay"
        }
        "composition_verification.graph_invalid" => {
            "declare nodes with unique keys, dependencies between declared nodes without a \
             cycle, and a calibration dependency for every inherited calibration"
        }
        "composition_verification.ledger_mismatch" => {
            "verify against the ledger the relations were read from, or regenerate the \
             verification"
        }
        _ => "supply an intact composition_verification_v1 artifact",
    }
}

fn refusal_json(code: &str, detail: &str, message: &str, offending: Option<&str>) -> String {
    json!({
        "code": code,
        "stage": STAGE,
        "detail": detail,
        "message": message,
        "offending": offending,
        "remedy": remedy_for(detail),
    })
    .to_string()
}

fn artifact_refusal(error: &CompositionVerificationArtifactError) -> String {
    match error.refusal() {
        Some((code, detail, message)) => refusal_json(code, &detail, &message, None),
        None => refusal_json(
            antecedent_core::reason_code!("invalid_argument"),
            "composition_verification.artifact_invalid",
            &error.to_string(),
            None,
        ),
    }
}

fn graph_refusal(error: &GraphError) -> String {
    refusal_json(
        antecedent_core::reason_code!("invalid_argument"),
        "composition_verification.graph_invalid",
        &error.to_string(),
        None,
    )
}

fn claim_refusal(claim: &ClaimRevalidation) -> Option<String> {
    claim.refusal().map(|r| {
        json!({
            "code": r.code,
            "stage": STAGE,
            "detail": r.detail,
            "message": format!(
                "{}; {}",
                r.expected.clone().unwrap_or_default(),
                r.supplied.clone().unwrap_or_default()
            ),
            "offending": r.offending,
            "remedy": r.remedy,
        })
        .to_string()
    })
}

// ---------------------------------------------------------------------------
// Declarations.
// ---------------------------------------------------------------------------

fn digest(value: &str, what: &str) -> PyResult<SemanticDigest> {
    parse_digest_hex(value)
        .ok_or_else(|| crate::value_err(format!("{what} must be 64 hexadecimal characters")))
}

fn opt_texts(map: &Map<String, Json>, key: &str) -> PyResult<Vec<String>> {
    match present(map, key) {
        None => Ok(Vec::new()),
        Some(value) => value
            .as_array()
            .ok_or_else(|| crate::value_err(format!("field `{key}` must be a list")))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| crate::value_err(format!("`{key}` holds a non-string")))
            })
            .collect(),
    }
}

fn parse_calibration(value: &Json) -> PyResult<CalibrationDeclaration> {
    let map = object(value, "calibration declaration")?;
    let pair = || -> PyResult<(SemanticDigest, SemanticDigest)> {
        Ok((
            digest(&text(map, "record")?, "calibration record")?,
            digest(&text(map, "coordinate")?, "calibration coordinate")?,
        ))
    };
    Ok(match text(map, "kind")?.as_str() {
        "point_only" => CalibrationDeclaration::PointOnly,
        "required" => CalibrationDeclaration::Required,
        "held" => {
            let (record, coordinate) = pair()?;
            CalibrationDeclaration::Held { record, coordinate }
        }
        "inherited" => {
            let (record, coordinate) = pair()?;
            CalibrationDeclaration::Inherited { from: text(map, "from")?, record, coordinate }
        }
        other => {
            return Err(crate::value_err(format!("unknown calibration kind `{other}`")));
        }
    })
}

fn parse_node(value: &Json) -> PyResult<CompositionNode> {
    let map = object(value, "node")?;
    let kind_name = text(map, "kind")?;
    let kind = NodeKind::from_name(&kind_name)
        .ok_or_else(|| crate::value_err(format!("unknown node kind `{kind_name}`")))?;
    let mut node = CompositionNode::new(
        text(map, "key")?,
        kind,
        digest(&text(map, "identity")?, "node identity")?,
    );
    if let Some(structure) = opt_text(map, "structure")? {
        node.structure = Some(digest(&structure, "structure identity")?);
    }
    node.representation = opt_text(map, "representation")?;
    if let Some(coordinate) = opt_text(map, "coordinate")? {
        node.coordinate = digest(&coordinate, "node coordinate")?;
    }
    if present(map, "unsupported").is_some() {
        node.support = SupportStanding::Unsupported { coordinates: opt_texts(map, "unsupported")? };
    }
    if let Some(trust) = opt_text(map, "trust")? {
        node.trust = TrustStanding::from_name(&trust)
            .ok_or_else(|| crate::value_err(format!("unknown trust standing `{trust}`")))?;
    }
    if let Some(calibration) = present(map, "calibration") {
        node.calibration = parse_calibration(calibration)?;
    }
    node.assumptions = opt_texts(map, "assumptions")?;
    node.transformation = opt_text(map, "transformation")?;
    node.evidence = opt_texts(map, "evidence")?
        .iter()
        .map(|t| {
            SourceId::from_hex(t)
                .ok_or_else(|| crate::value_err("evidence sources must be 64 hexadecimal digits"))
        })
        .collect::<PyResult<_>>()?;
    Ok(node)
}

/// A graph, or the graph error the declaration produced.
fn parse_graph(value: &Json, what: &str) -> PyResult<Result<CompositionGraph, GraphError>> {
    let map = object(value, what)?;
    let mut graph = CompositionGraph::new();
    for node in list(map, "nodes")? {
        if let Err(error) = graph.add_node(parse_node(node)?) {
            return Ok(Err(error));
        }
    }
    for edge in list(map, "edges")? {
        let e = object(edge, "edge")?;
        let kind_name = text(e, "kind")?;
        let kind = DependencyKind::from_name(&kind_name)
            .ok_or_else(|| crate::value_err(format!("unknown dependency kind `{kind_name}`")))?;
        if let Err(error) = graph.add_edge(&text(e, "downstream")?, &text(e, "upstream")?, kind) {
            return Ok(Err(error));
        }
    }
    Ok(graph.validate().map(|()| graph))
}

fn parse_effects(value: Option<&Json>) -> PyResult<BTreeMap<String, ObservedEffect>> {
    let mut out = BTreeMap::new();
    let Some(value) = value else { return Ok(out) };
    let items = value.as_array().ok_or_else(|| crate::value_err("effects must be a list"))?;
    for item in items {
        let m = object(item, "effect")?;
        let (key, body) = (text(m, "key")?, text(m, "text")?);
        let effect = match text(m, "kind")?.as_str() {
            "recomputed" => ObservedEffect::Recomputed { cause: body },
            "withdrawn" => ObservedEffect::Withdrawn { cause: body },
            "refused" => ObservedEffect::Refused { detail: body },
            other => return Err(crate::value_err(format!("unknown effect kind `{other}`"))),
        };
        if out.insert(key, effect).is_some() {
            return Err(crate::value_err("one effect per node"));
        }
    }
    Ok(out)
}

fn parse_shared(value: Option<&Json>) -> PyResult<Vec<SharedDataRelation>> {
    let Some(value) = value else { return Ok(Vec::new()) };
    let items = value.as_array().ok_or_else(|| crate::value_err("shared_data must be a list"))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let m = object(item, "shared-data relation")?;
        let name = text(m, "relation")?;
        let relation = RELATIONS
            .iter()
            .find(|r| **r == name)
            .ok_or_else(|| crate::value_err(format!("unknown relation `{name}`")))?;
        out.push(SharedDataRelation {
            first: text(m, "first")?,
            second: text(m, "second")?,
            relation,
            shared_observations: member(m, "shared_observations")?
                .as_u64()
                .ok_or_else(|| crate::value_err("shared_observations must be an integer"))?,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Results as data.
// ---------------------------------------------------------------------------

fn via(via: Option<&(String, DependencyKind)>) -> (Json, Json) {
    via.map_or((Json::Null, Json::Null), |(n, k)| (json!(n), json!(k.as_str())))
}

fn calibration_json(c: &CalibrationLineage) -> Json {
    json!({
        "state": c.state.as_str(),
        "from": c.from,
        "newly_required": c.newly_required,
        "detail": c.detail,
    })
}

fn claim_json(claim: &ClaimRevalidation) -> Json {
    let numerical: Vec<Json> = claim
        .numerical
        .iter()
        .map(|r| {
            let (node, kind) = via(r.via.as_ref());
            json!({"origin": r.origin, "via_node": node, "via_kind": kind,
                   "cause": r.cause.as_str()})
        })
        .collect();
    let withdrawals: Vec<Json> = claim
        .withdrawals
        .iter()
        .map(|w| {
            let (node, kind) = via(w.via.as_ref());
            json!({"kind": w.kind.as_str(), "origin": w.origin, "via_node": node,
                   "via_kind": kind, "detail": w.detail})
        })
        .collect();
    json!({
        "key": claim.key,
        "kind": claim.kind.as_str(),
        "class": claim.class.as_str(),
        "numerical": numerical,
        "withdrawals": withdrawals,
        "authority_retained": claim.authority == PriorAuthority::Retained,
        "calibration": calibration_json(&claim.calibration),
    })
}

fn shared_json(relations: &[SharedDataRelation]) -> Json {
    Json::Array(
        relations
            .iter()
            .map(|r| {
                json!({"first": r.first, "second": r.second, "relation": r.relation,
                       "shared_observations": r.shared_observations})
            })
            .collect(),
    )
}

fn entry_json(entry: &LineageEntry) -> Json {
    json!({
        "key": entry.key,
        "kind": entry.kind.as_str(),
        "identity": entry.identity.to_hex(),
        "representation": entry.representation,
        "class": entry.class.as_str(),
        "calibration": calibration_json(&entry.calibration),
    })
}

fn view_json(view: &CompositionView) -> Json {
    json!({
        "claim": entry_json(&view.claim),
        "upstream": view.upstream.iter().map(entry_json).collect::<Vec<_>>(),
        "edges": view.edges.iter().map(|e| json!({
            "downstream": e.downstream, "upstream": e.upstream, "kind": e.kind.as_str()
        })).collect::<Vec<_>>(),
        "evidence": view.evidence.iter().map(SourceId::to_hex).collect::<Vec<_>>(),
        "transformations": view.transformations.iter().map(|t| json!({
            "node": t.node, "kind": t.kind.as_str(), "name": t.name
        })).collect::<Vec<_>>(),
        "shared_data": shared_json(&view.shared_data),
        "unsupported": view.unsupported.iter().map(|u| json!({
            "node": u.node, "coordinate": u.coordinate
        })).collect::<Vec<_>>(),
        "assumptions": view.assumptions.iter().map(|a| json!({
            "node": a.node, "statement": a.statement
        })).collect::<Vec<_>>(),
        "versions": view.versions().iter().map(|(n, v)| json!({"node": n, "version": v}))
            .collect::<Vec<_>>(),
        "identity": view.identity().to_hex(),
    })
}

// ---------------------------------------------------------------------------
// Handle and entry points.
// ---------------------------------------------------------------------------

/// A sealed and replayed composition verification.
#[pyclass(name = "CompositionVerificationHandle", skip_from_py_object)]
pub(crate) struct CompositionVerificationHandle {
    inner: CompositionVerificationArtifact,
}

type Made = (Option<CompositionVerificationHandle>, Option<String>);

#[pymethods]
impl CompositionVerificationHandle {
    /// Classification of every claim, identities and declared overlap, as JSON.
    fn summary_json(&self) -> String {
        let r = self.inner.revalidation();
        json!({
            "claims": r.claims().map(claim_json).collect::<Vec<_>>(),
            "removed": r.removed(),
            "previous_identity": r.previous_identity().to_hex(),
            "current_identity": r.current_identity().to_hex(),
            "revalidation_identity": r.identity().to_hex(),
            "identity": self.inner.identity(),
            "has_previous": self.inner.previous().is_some(),
            "ledger_state": self.inner.ledger_state().map(|d| d.to_hex()),
            "shared_data": shared_json(self.inner.shared_data()),
        })
        .to_string()
    }

    /// The lineage report of one claim, or the refusal for an unknown claim.
    fn view_json(&self, key: &str) -> (Option<String>, Option<String>) {
        match self.inner.view(key) {
            Ok(view) => (Some(view_json(&view).to_string()), None),
            Err(error) => (
                None,
                Some(refusal_json(
                    antecedent_core::reason_code!("invalid_argument"),
                    "composition_verification.unknown_claim",
                    &error.to_string(),
                    Some(key),
                )),
            ),
        }
    }

    /// The typed refusal when the claim is not usable as it stands, else `None`.
    fn require_valid(&self, key: &str) -> Option<String> {
        match self.inner.revalidation().claim(key) {
            Some(claim) => claim_refusal(claim),
            None => Some(refusal_json(
                antecedent_core::reason_code!("invalid_argument"),
                "composition_verification.unknown_claim",
                "the claim is not in the current graph",
                Some(key),
            )),
        }
    }

    /// The artifact bytes and identity, or the refusal.
    #[allow(clippy::type_complexity, reason = "native pair of (bytes, identity) or refusal")]
    fn export(
        &self,
        py: Python<'_>,
        artifact_id: &str,
    ) -> (Option<(Py<PyBytes>, String)>, Option<String>) {
        match self.inner.export(artifact_id) {
            Ok(bytes) => {
                (Some((PyBytes::new(py, &bytes).unbind(), self.inner.identity().to_owned())), None)
            }
            Err(error) => (None, Some(artifact_refusal(&error))),
        }
    }
}

/// Classify the current graph against the earlier one and seal the result.
#[pyfunction]
fn composition_verification_verify(request_json: &str) -> PyResult<Made> {
    let parsed = parse(request_json, "composition verification request")?;
    let map = object(&parsed, "request")?;
    let current = match parse_graph(member(map, "current")?, "current graph")? {
        Ok(graph) => graph,
        Err(error) => return Ok((None, Some(graph_refusal(&error)))),
    };
    let previous = match present(map, "previous") {
        None => None,
        Some(value) => match parse_graph(value, "previous graph")? {
            Ok(graph) => Some(graph),
            Err(error) => return Ok((None, Some(graph_refusal(&error)))),
        },
    };
    let effects = parse_effects(present(map, "effects"))?;
    let shared = parse_shared(present(map, "shared_data"))?;
    let state = opt_text(map, "ledger_state")?.map(|t| digest(&t, "ledger state")).transpose()?;
    Ok(
        match CompositionVerificationArtifact::seal(
            previous.as_ref(),
            &current,
            &effects,
            &shared,
            state,
        ) {
            Ok((inner, _)) => (Some(CompositionVerificationHandle { inner }), None),
            Err(error) => (None, Some(artifact_refusal(&error))),
        },
    )
}

/// Consume a composition verification artifact by classifying again, against
/// an artifact identity the consumer retained.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None))]
fn composition_verification_consume(artifact: &[u8], expected_identity: Option<&str>) -> Made {
    match CompositionVerificationArtifact::from_bytes(artifact, expected_identity) {
        Ok(inner) => (Some(CompositionVerificationHandle { inner }), None),
        Err(error) => (None, Some(artifact_refusal(&error))),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CompositionVerificationHandle>()?;
    module.add_function(wrap_pyfunction!(composition_verification_verify, module)?)?;
    module.add_function(wrap_pyfunction!(composition_verification_consume, module)?)?;
    Ok(())
}
