//! Bounded Python bridge for the portable composed result.
//!
//! Python hands over artifact bytes, references and dependency edges; the bundle
//! graph, the Merkle digests, every per-kind verifier and every refusal are Rust's.
//! A refusal comes back as structured JSON for the Python layer to raise as its own
//! exception type, keeping the `composition_bundle.<stage>` detail. A consumed
//! bundle comes back as JSON: each node verified, reference-unresolved or failed at
//! a named stage, with the facts and numbers read from verified artifacts.

use std::collections::BTreeMap;

use crate::bridge::{check_declared_size, external_refusal_json};
use antecedent::analysis::composition::{
    BundleBuilder, BundleEdge, BundleError, BundleLimits, BundleStage, ClaimLabel,
    CompositionBundle, ConsumedBundle, ConsumedNode, EvidenceRelationship, NodeKind, NodeSource,
    NodeStatus, ProviderOrData, SuppliedSources, consume_with_limits, describe_artifact,
    detect_node_kind, export_bundle, mean_decision_bytes, producer_identity,
    refit_compact_bundle_model, refit_compact_model, require_compact_model_refit_inputs,
};
use antecedent::contract::ExternalRefusal;
use antecedent::design::decision_artifact::contract_from_json_refusal;
use antecedent::io::IoError;
use antecedent::reason::reason_code;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::{Value, json};

/// Largest declaration JSON (facts, requirement, supplied sources) accepted.
const MAX_DECLARATION_BYTES: usize = crate::bridge::MAX_CONFIG_DECLARATION_BYTES;

type Refusal = Option<String>;

mod compact_refit;

pub(crate) fn error_refusal(error: &BundleError) -> String {
    if let Some(refusal) = error.to_refusal() {
        return external_refusal_json(&refusal);
    }
    let (code, detail) = match error {
        BundleError::Container(io)
            if matches!(
                **io,
                IoError::UnsupportedVersion { .. } | IoError::UnsupportedFormat { .. }
            ) =>
        {
            (
                reason_code!("schema_mismatch"),
                antecedent::reason::reason_details::composition_bundle::INCOMPATIBLE_VERSION,
            )
        }
        _ => (
            reason_code!("invalid_argument"),
            antecedent::reason::reason_details::composition_bundle::INVALID_CONTAINER,
        ),
    };
    external_refusal_json(&ExternalRefusal {
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
    check_declared_size(text, MAX_DECLARATION_BYTES, "composition")
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

fn consumed_node_json(node: &ConsumedNode, replayed: bool) -> Value {
    json!({
        "id": node.id,
        "kind": node.kind.as_str(),
        "identity": node.identity,
        "chain_digest": node.chain_digest,
        "status": status_json(&node.status),
        "replayed": replayed,
        "facts": node.facts,
        "values": numbers_json(&node.values),
        "inspected": numbers_json(&node.inspected),
        "claim_label": node.claim_label.map(ClaimLabel::as_str),
    })
}

fn edges_json(edges: &[BundleEdge]) -> Value {
    Value::Array(
        edges
            .iter()
            .map(|edge| {
                json!({
                    "from": edge.from,
                    "to": edge.to,
                    "upstream_digest": edge.upstream_digest,
                })
            })
            .collect(),
    )
}

pub(crate) fn consumed_json(consumed: &ConsumedBundle) -> String {
    json!({
        "identity": consumed.identity(),
        "all_verified": consumed.all_verified(),
        "all_replayed": consumed.all_replayed(),
        "nodes": consumed.nodes().iter().map(|node| consumed_node_json(node, consumed.node_replayed(&node.id))).collect::<Vec<_>>(),
        "edges": edges_json(consumed.edges()),
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
        .map_err(|e| crate::value_err(format!("invalid supplied sources: {e}")))?;
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
    pub(crate) inner: BundleBuilder,
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
        crate::catch_ffi(|| {
            check_size(requires_json)?;
            let Some(parsed) = NodeKind::from_name(kind) else {
                return Ok(Some(kind_refusal(kind)));
            };
            let requires: ProviderOrData = serde_json::from_str(requires_json)
                .map_err(|e| crate::value_err(format!("invalid requirement: {e}")))?;
            let facts: BTreeMap<String, String> = match facts_json {
                Some(text) => {
                    check_size(text)?;
                    serde_json::from_str(text)
                        .map_err(|e| crate::value_err(format!("invalid facts: {e}")))?
                }
                None => BTreeMap::new(),
            };
            let inspected: Vec<(String, f64)> = match inspected_json {
                Some(text) => {
                    check_size(text)?;
                    serde_json::from_str(text)
                        .map_err(|e| crate::value_err(format!("invalid inspected values: {e}")))?
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
        })
    }

    /// Add a reference whose digest is the producer identity of `data`.
    /// `role` is a catalog role shared with a compiled study.
    fn add_producer_reference(&mut self, node_id: &str, role: &str, data: &[u8]) -> Refusal {
        self.inner
            .add_producer_reference(node_id, role, data)
            .err()
            .map(|error| error_refusal(&error))
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
        crate::catch_ffi(|| {
            let Some(parsed) = EvidenceRelationship::from_name(relationship) else {
                return Err(crate::value_err(format!(
                    "`{relationship}` is not an evidence relationship"
                )));
            };
            Ok(match self.inner.relate(left, right, parsed) {
                Ok(id) => (Some(id), None),
                Err(error) => (None, Some(error_refusal(&error))),
            })
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
    pub(crate) bundle: CompositionBundle,
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
        edges_json(self.bundle.edges()).to_string()
    }

    /// Serialize through the checksummed container. Returns `(bytes, refusal)`.
    fn export<'py>(
        &self,
        py: Python<'py>,
        artifact_id: &str,
    ) -> (Option<Bound<'py, PyBytes>>, Refusal) {
        match py.detach(|| export_bundle(&self.bundle, artifact_id)) {
            Ok(bytes) => (Some(PyBytes::new(py, &bytes)), None),
            Err(error) => (None, Some(error_refusal(&error))),
        }
    }
}

/// Consume exported bytes under the identity the consumer retained independently and the
/// consumer's own bounds (each omitted bound keeps its default). Returns
/// `(consumed_json, refusal)`.
#[pyfunction]
#[pyo3(signature = (data, expected_identity, supplied_json=None, *, max_bundle_bytes=None, max_nodes=None, max_edges=None, max_embedded_bytes=None, producer_roles=None, producer_payloads=None))]
fn consume_composition_bundle(
    py: Python<'_>,
    data: &[u8],
    expected_identity: &str,
    supplied_json: Option<&str>,
    max_bundle_bytes: Option<usize>,
    max_nodes: Option<usize>,
    max_edges: Option<usize>,
    max_embedded_bytes: Option<usize>,
    producer_roles: Option<Vec<String>>,
    producer_payloads: Option<Vec<Vec<u8>>>,
) -> PyResult<(Option<String>, Refusal)> {
    crate::catch_ffi(|| {
        let mut supplied = supplied_from(supplied_json)?;
        match (producer_roles, producer_payloads) {
            (None, None) => {}
            (Some(roles), Some(payloads)) if roles.len() == payloads.len() => {
                for (role, payload) in roles.into_iter().zip(payloads) {
                    supplied = supplied.with_producer(&role, payload);
                }
            }
            _ => {
                return Err(crate::value_err(
                    "producer_roles and producer_payloads must be the same length",
                ));
            }
        }
        let defaults = BundleLimits::default();
        let limits = BundleLimits {
            max_bundle_bytes: max_bundle_bytes.unwrap_or(defaults.max_bundle_bytes),
            max_nodes: max_nodes.unwrap_or(defaults.max_nodes),
            max_edges: max_edges.unwrap_or(defaults.max_edges),
            max_embedded_bytes: max_embedded_bytes.unwrap_or(defaults.max_embedded_bytes),
        };
        let consumed =
            py.detach(|| consume_with_limits(data, &limits, expected_identity, &supplied));
        Ok(match consumed {
            Ok(consumed) => (Some(consumed_json(&consumed)), None),
            Err(error) => (None, Some(error_refusal(&error))),
        })
    })
}

/// Decode one artifact on its own and report `{kind, identity, facts, values}` as
/// JSON; `kind` is a node kind name or `auto`. Returns `(json, refusal)`.
#[pyfunction]
#[pyo3(signature = (kind, data))]
fn composition_describe_artifact(
    py: Python<'_>,
    kind: &str,
    data: &[u8],
) -> (Option<String>, Refusal) {
    let parsed = if kind == "auto" { detect_node_kind(data) } else { NodeKind::from_name(kind) };
    let Some(parsed) = parsed else {
        return (None, Some(kind_refusal(kind)));
    };
    match py.detach(|| describe_artifact(parsed, data)) {
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
    crate::catch_ffi(|| {
        check_size(contract_json)?;
        let contract = match contract_from_json_refusal(contract_json) {
            Ok(contract) => contract,
            Err(refusal) => return Ok((None, Some(external_refusal_json(&refusal)))),
        };
        Ok(match py.detach(|| mean_decision_bytes(&contract, claim_data, artifact_id)) {
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
    })
}

/// The node kind a container's artifact fills, or `None`.
#[pyfunction]
fn composition_detect_kind(data: &[u8]) -> Option<&'static str> {
    detect_node_kind(data).map(NodeKind::as_str)
}

/// Verify the compact artifact before checking whether its original fit can be replayed.
#[pyfunction]
fn composition_require_compact_refit(py: Python<'_>, data: &[u8]) -> PyResult<Refusal> {
    crate::catch_ffi(|| {
        Ok(py.detach(|| require_compact_model_refit_inputs(data)).err().map(|failure| {
            error_refusal(&BundleError::Refused {
                stage: failure.stage,
                reason: failure.reason,
                offending: None,
            })
        }))
    })
}

/// Refit the native model from its original ordered training data.
#[pyfunction]
#[pyo3(signature = (data, expected_identity, learner, training_json, *, model_node=None, seed=1, threads=None, cancel=None, memory_bytes=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn composition_refit_compact<'py>(
    py: Python<'py>,
    data: &[u8],
    expected_identity: &str,
    learner: &str,
    training_json: &str,
    model_node: Option<&str>,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<(Option<(String, Bound<'py, PyBytes>)>, Refusal)> {
    let ctx = crate::chain_decision_api::context(seed, threads, cancel, memory_bytes);
    let result = crate::detach_catch(py, || {
        let training = match compact_refit::training(training_json, model_node, &ctx) {
            Ok(rows) => rows,
            Err(problem) => return Ok(Err(problem)),
        };
        let replayed = if let Some(node) = model_node {
            refit_compact_bundle_model(data, expected_identity, node, training, learner, &ctx)
        } else {
            refit_compact_model(data, expected_identity, training, learner, &ctx)
        };
        let replayed = match replayed {
            Ok(replayed) => replayed,
            Err(problem) => return Ok(Err(problem)),
        };
        Ok(compact_refit::artifact(&replayed, model_node))
    })?;
    Ok(match result {
        Ok((identity, bytes)) => (Some((identity, PyBytes::new(py, &bytes))), None),
        Err(problem) => (None, Some(error_refusal(&problem))),
    })
}

/// The existing identity of a producer artifact, for a catalog role.
#[pyfunction]
fn producer_artifact_identity(role: &str, data: &[u8]) -> PyResult<String> {
    producer_identity(role, data).map_err(crate::value_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    compact_refit::register(m)?;
    m.add_class::<PyBundleBuilder>()?;
    m.add_class::<PyCompositionBundle>()?;
    m.add_function(wrap_pyfunction!(consume_composition_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(producer_artifact_identity, m)?)?;
    m.add_function(wrap_pyfunction!(composition_describe_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(composition_mean_decision, m)?)?;
    m.add_function(wrap_pyfunction!(composition_require_compact_refit, m)?)?;
    m.add_function(wrap_pyfunction!(composition_refit_compact, m)?)?;
    m.add_function(wrap_pyfunction!(composition_detect_kind, m)?)
}
