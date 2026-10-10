//! Bounded Python bridge for C3: the portable composed result.
//!
//! Python hands over artifact bytes, references and dependency edges; the bundle
//! graph, the Merkle digests, every per-kind verifier and every refusal are Rust's.
//! A refusal comes back as structured JSON for the Python layer to raise as its own
//! exception type, keeping the `composition_bundle.<stage>` detail. A consumed
//! bundle comes back as JSON: each node verified, reference-unresolved or failed at
//! a named stage, with the facts and numbers read from verified artifacts.

use std::collections::BTreeMap;

use antecedent::analysis::composition::{
    BundleBuilder, BundleError, BundleStage, ClaimLabel, CompositionBundle, ConsumedBundle,
    ConsumedNode, EvidenceRelationship, NodeKind, NodeSource, NodeStatus, ProviderOrData,
    SuppliedSources, consume, describe_artifact, detect_node_kind, export_bundle,
    mean_decision_bytes,
};
use antecedent_core::{ExternalRefusal, reason_code};
use antecedent_design::decision_artifact::contract_from_json_refusal;
use antecedent_io::IoError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::{Value, json};

/// Largest declaration JSON (facts, requirement, supplied sources) accepted.
const MAX_DECLARATION_BYTES: usize = 1024 * 1024;

type Refusal = Option<String>;

fn refusal_json(value: &ExternalRefusal) -> String {
    json!({
        "code": value.code,
        "stage": value.stage,
        "detail": value.detail,
        "offending": value.offending,
        "expected": value.expected,
        "supplied": value.supplied,
        "remedy": value.remedy,
    })
    .to_string()
}

pub(crate) fn error_refusal(error: &BundleError) -> String {
    if let Some(refusal) = error.to_refusal() {
        return refusal_json(&refusal);
    }
    let (code, detail) = match error {
        BundleError::Container(io)
            if matches!(
                **io,
                IoError::UnsupportedVersion { .. } | IoError::UnsupportedFormat { .. }
            ) =>
        {
            (reason_code!("schema_mismatch"), "composition_bundle.incompatible_version")
        }
        _ => (reason_code!("invalid_argument"), "composition_bundle.invalid_container"),
    };
    refusal_json(&ExternalRefusal {
        code,
        stage: "bind",
        detail: detail.to_owned(),
        offending: None,
        expected: None,
        supplied: Some(error.to_string()),
        capability: None,
        remedy: None,
    })
}

fn kind_refusal(name: &str) -> String {
    error_refusal(&BundleError::Refused {
        stage: BundleStage::UnknownNodeKind,
        reason: format!("`{name}` is not a bundle node kind"),
        offending: None,
    })
}

fn check_size(text: &str) -> PyResult<()> {
    if text.len() > MAX_DECLARATION_BYTES {
        return Err(PyValueError::new_err("composition declaration is too large"));
    }
    Ok(())
}

fn requires_json(requires: &ProviderOrData) -> Value {
    serde_json::to_value(requires).unwrap_or(Value::Null)
}

fn numbers_json(values: &[(String, f64)]) -> Value {
    Value::Array(values.iter().map(|(key, value)| json!([key, value])).collect())
}

fn status_json(status: &NodeStatus) -> Value {
    match status {
        NodeStatus::Verified => json!({ "state": "verified" }),
        NodeStatus::ReferenceUnresolved { requires } => json!({
            "state": "reference_unresolved",
            "requires": requires_json(requires),
            "stage": BundleStage::CallbackUnavailable.as_str(),
            "detail": BundleStage::CallbackUnavailable.detail(),
            "code": BundleStage::CallbackUnavailable.code(),
        }),
        NodeStatus::Failed { stage, reason } => json!({
            "state": "failed",
            "stage": stage.as_str(),
            "detail": stage.detail(),
            "code": stage.code(),
            "reason": reason,
        }),
    }
}

fn consumed_node_json(node: &ConsumedNode) -> Value {
    json!({
        "id": node.id,
        "kind": node.kind.as_str(),
        "identity": node.identity,
        "chain_digest": node.chain_digest,
        "status": status_json(&node.status),
        "facts": node.facts,
        "values": numbers_json(&node.values),
        "inspected": numbers_json(&node.inspected),
        "claim_label": node.claim_label.map(ClaimLabel::as_str),
    })
}

pub(crate) fn consumed_json(consumed: &ConsumedBundle) -> String {
    json!({
        "identity": consumed.identity(),
        "all_verified": consumed.all_verified(),
        "nodes": consumed.nodes().iter().map(consumed_node_json).collect::<Vec<_>>(),
        "edges": consumed
            .edges()
            .iter()
            .map(|e| json!({ "from": e.from, "to": e.to, "upstream_digest": e.upstream_digest }))
            .collect::<Vec<_>>(),
    })
    .to_string()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderIn {
    provider_id: String,
    snapshot_id: String,
    request_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DataIn {
    snapshot_id: String,
    digest: String,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct SuppliedIn {
    providers: Vec<ProviderIn>,
    data: Vec<DataIn>,
}

pub(crate) fn supplied_from(text: Option<&str>) -> PyResult<SuppliedSources> {
    let Some(text) = text else {
        return Ok(SuppliedSources::default());
    };
    check_size(text)?;
    let parsed: SuppliedIn = serde_json::from_str(text)
        .map_err(|e| PyValueError::new_err(format!("invalid supplied sources: {e}")))?;
    let mut supplied = SuppliedSources::default();
    for p in &parsed.providers {
        supplied = supplied.with_provider(&p.provider_id, &p.snapshot_id, &p.request_fingerprint);
    }
    for d in &parsed.data {
        supplied = supplied.with_data(&d.snapshot_id, &d.digest);
    }
    Ok(supplied)
}

/// A bundle under construction: artifacts embedded from their bytes, references,
/// dependency edges and evidence relationships.
#[pyclass(name = "CompositionBundleBuilder", skip_from_py_object)]
pub(crate) struct PyBundleBuilder {
    inner: BundleBuilder,
}

#[pymethods]
impl PyBundleBuilder {
    #[new]
    fn new() -> Self {
        Self { inner: BundleBuilder::new() }
    }

    /// Embed an artifact. `kind` is a node kind name, or `auto` to read it from the
    /// container. Returns `(node_id, refusal)`.
    #[pyo3(signature = (kind, data, node_id=None))]
    fn add_artifact(
        &mut self,
        kind: &str,
        data: &[u8],
        node_id: Option<&str>,
    ) -> (Option<String>, Refusal) {
        let added = if kind == "auto" {
            self.inner.add_detected_artifact(node_id, data)
        } else {
            match NodeKind::from_name(kind) {
                Some(parsed) => self.inner.add_artifact(node_id, parsed, data),
                None => return (None, Some(kind_refusal(kind))),
            }
        };
        match added {
            Ok(id) => (Some(id), None),
            Err(error) => (None, Some(error_refusal(&error))),
        }
    }

    /// Add a reference node that needs a supplied provider or data source.
    /// `requires_json` is `{"provider": {...}}` or `{"data": {...}}`; `facts_json`
    /// maps fact names to declared values; `inspected_json` is `[[name, value]]`.
    #[pyo3(signature = (node_id, kind, identity, requires_json, facts_json=None, inspected_json=None))]
    fn add_reference(
        &mut self,
        node_id: &str,
        kind: &str,
        identity: &str,
        requires_json: &str,
        facts_json: Option<&str>,
        inspected_json: Option<&str>,
    ) -> PyResult<Refusal> {
        check_size(requires_json)?;
        let Some(parsed) = NodeKind::from_name(kind) else {
            return Ok(Some(kind_refusal(kind)));
        };
        let requires: ProviderOrData = serde_json::from_str(requires_json)
            .map_err(|e| PyValueError::new_err(format!("invalid requirement: {e}")))?;
        let facts: BTreeMap<String, String> = match facts_json {
            Some(text) => {
                check_size(text)?;
                serde_json::from_str(text)
                    .map_err(|e| PyValueError::new_err(format!("invalid facts: {e}")))?
            }
            None => BTreeMap::new(),
        };
        let inspected: Vec<(String, f64)> = match inspected_json {
            Some(text) => {
                check_size(text)?;
                serde_json::from_str(text)
                    .map_err(|e| PyValueError::new_err(format!("invalid inspected values: {e}")))?
            }
            None => Vec::new(),
        };
        let apply = || -> Result<(), BundleError> {
            self.inner.add_reference(node_id, parsed, identity, requires)?;
            for (key, value) in &facts {
                self.inner.declare_fact(node_id, key, value)?;
            }
            for (key, value) in &inspected {
                self.inner.declare_inspected(node_id, key, *value)?;
            }
            Ok(())
        };
        Ok(apply().err().map(|error| error_refusal(&error)))
    }

    /// Declare that `dependent` was derived from `upstream`.
    fn connect(&mut self, upstream: &str, dependent: &str) -> Refusal {
        self.inner.connect(upstream, dependent).err().map(|e| error_refusal(&e))
    }

    /// Declare how the evidence behind two nodes depends on each other. Returns
    /// `(declaration_node_id, refusal)`.
    fn relate(
        &mut self,
        left: &str,
        right: &str,
        relationship: &str,
    ) -> PyResult<(Option<String>, Refusal)> {
        let Some(parsed) = EvidenceRelationship::from_name(relationship) else {
            return Err(PyValueError::new_err(format!(
                "`{relationship}` is not an evidence relationship"
            )));
        };
        Ok(match self.inner.relate(left, right, parsed) {
            Ok(id) => (Some(id), None),
            Err(error) => (None, Some(error_refusal(&error))),
        })
    }

    /// Seal the graph. Returns `(bundle, refusal)`.
    fn build(&self) -> (Option<PyCompositionBundle>, Refusal) {
        match self.inner.build() {
            Ok(bundle) => (Some(PyCompositionBundle { bundle }), None),
            Err(error) => (None, Some(error_refusal(&error))),
        }
    }
}

/// A sealed composed result.
#[pyclass(name = "CompositionBundle", skip_from_py_object)]
pub(crate) struct PyCompositionBundle {
    bundle: CompositionBundle,
}

#[pymethods]
impl PyCompositionBundle {
    /// The bundle identity (lowercase hex), covering every node, edge and digest.
    #[getter]
    fn identity(&self) -> String {
        self.bundle.identity().to_owned()
    }

    /// Nodes ordered by id, as JSON.
    #[getter]
    fn nodes_json(&self) -> String {
        let nodes: Vec<Value> = self
            .bundle
            .nodes()
            .iter()
            .map(|node| {
                json!({
                    "id": node.id,
                    "kind": node.kind.as_str(),
                    "identity": node.identity,
                    "embedded": matches!(node.source, NodeSource::Embedded { .. }),
                    "chain_digest": self.bundle.chain_digest(&node.id),
                })
            })
            .collect();
        Value::Array(nodes).to_string()
    }

    /// Edges with their upstream Merkle digests, as JSON.
    #[getter]
    fn edges_json(&self) -> String {
        let edges: Vec<Value> = self
            .bundle
            .edges()
            .iter()
            .map(|e| json!({ "from": e.from, "to": e.to, "upstream_digest": e.upstream_digest }))
            .collect();
        Value::Array(edges).to_string()
    }

    /// Serialize through the checksummed container. Returns `(bytes, refusal)`.
    fn export<'py>(
        &self,
        py: Python<'py>,
        artifact_id: &str,
    ) -> (Option<Bound<'py, PyBytes>>, Refusal) {
        match export_bundle(&self.bundle, artifact_id) {
            Ok(bytes) => (Some(PyBytes::new(py, &bytes)), None),
            Err(error) => (None, Some(error_refusal(&error))),
        }
    }
}

/// Consume exported bytes under the identity the consumer retained independently.
/// Returns `(consumed_json, refusal)`.
#[pyfunction]
#[pyo3(signature = (data, expected_identity, supplied_json=None))]
fn consume_composition_bundle(
    data: &[u8],
    expected_identity: &str,
    supplied_json: Option<&str>,
) -> PyResult<(Option<String>, Refusal)> {
    let supplied = supplied_from(supplied_json)?;
    Ok(match consume(data, expected_identity, &supplied) {
        Ok(consumed) => (Some(consumed_json(&consumed)), None),
        Err(error) => (None, Some(error_refusal(&error))),
    })
}

/// Decode one artifact on its own and report `{kind, identity, facts, values}` as
/// JSON; `kind` is a node kind name or `auto`. Returns `(json, refusal)`.
#[pyfunction]
#[pyo3(signature = (kind, data))]
fn composition_describe_artifact(kind: &str, data: &[u8]) -> (Option<String>, Refusal) {
    let parsed = if kind == "auto" { detect_node_kind(data) } else { NodeKind::from_name(kind) };
    let Some(parsed) = parsed else {
        return (None, Some(kind_refusal(kind)));
    };
    match describe_artifact(parsed, data) {
        Ok(described) => (
            Some(
                json!({
                    "kind": described.kind.as_str(),
                    "identity": described.identity,
                    "facts": described.facts,
                    "values": numbers_json(&described.values),
                })
                .to_string(),
            ),
            None,
        ),
        Err(failure) => (
            None,
            Some(error_refusal(&BundleError::Refused {
                stage: failure.stage,
                reason: failure.reason,
                offending: None,
            })),
        ),
    }
}

/// Evaluate a decision contract (declaration JSON) on the means of an external claim
/// container and return the point-only result as container bytes. A functional a mean
/// cannot answer refuses as `unsupported_law`. Returns `(bytes, refusal)`.
#[pyfunction]
fn composition_mean_decision<'py>(
    py: Python<'py>,
    contract_json: &str,
    claim_data: &[u8],
    artifact_id: &str,
) -> PyResult<(Option<Bound<'py, PyBytes>>, Refusal)> {
    check_size(contract_json)?;
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
    };
    Ok(match mean_decision_bytes(&contract, claim_data, artifact_id) {
        Ok(bytes) => (Some(PyBytes::new(py, &bytes)), None),
        Err(failure) => (
            None,
            Some(error_refusal(&BundleError::Refused {
                stage: failure.stage,
                reason: failure.reason,
                offending: None,
            })),
        ),
    })
}

/// The node kind a container's artifact fills, or `None`.
#[pyfunction]
fn composition_detect_kind(data: &[u8]) -> Option<&'static str> {
    detect_node_kind(data).map(NodeKind::as_str)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBundleBuilder>()?;
    m.add_class::<PyCompositionBundle>()?;
    m.add_function(wrap_pyfunction!(consume_composition_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(composition_describe_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(composition_mean_decision, m)?)?;
    m.add_function(wrap_pyfunction!(composition_detect_kind, m)?)
}
