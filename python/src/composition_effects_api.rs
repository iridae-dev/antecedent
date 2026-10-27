//! Python bridge for the observed effects a composition verification reads.
//!
//! A composed analysis learns that something upstream moved from four places: a recalculation
//! plan, the evidence ledger, a claim's freshness or contract identities, and the state's
//! invalidation log. Python declares those inputs (stage own-input digests and capabilities,
//! the ledger handle itself, claims with their snapshots, log entries) and Rust builds the
//! composition nodes, decides each effect and classifies the log; this module only parses the
//! declaration and renders the result.
//!
//! No interval, posterior or calibration record is produced.

use antecedent::analysis::composition_verification::{
    contract_effects, freshness_effects, invalidation_effects, ledger_effects, query_revalidation,
    stage_effects, stage_node, stage_node_with_standing,
};
use antecedent::analysis::selective_discovery::revalidate_after_discovery;
use antecedent_contract::claim::ClaimEnvelope;
use antecedent_contract::composition_verification::{
    ClaimStanding, NodeKind, SupportStanding, TrustStanding,
};
use antecedent_contract::evidence_ledger::ResultId;
use antecedent_core::recalc::{ChangeKind, Stage, StageIdentity};
use antecedent_core::{ContractIdentities, ModelId, QueryId, SemanticDigest, StateVersion};
use antecedent_io::recalc_receipt_artifact::{
    CapabilitiesWire, DeclaredStageWire, plan_from_wire, stage_from_label,
};
use antecedent_state::{ChangedInput, InvalidationLog, InvalidationTarget};
use pyo3::prelude::*;
use serde_json::{Map, Value as Json, json};

use crate::composition_verification_api::{
    claim_json, digest, effects_wire, graph_refusal, node_wire, parse_calibration, parse_graph,
    refusal_json,
};
use crate::evidence_ledger_api::EvidenceLedgerHandle;
use crate::measurement_model_api::{member, object, opt_text, parse, present, text};
use crate::scientific_bridge_api::parse_claim;

/// The payload every entry returns: a JSON result or a structured refusal.
type Declared = (Option<String>, Option<String>);

fn invalid(detail: &str, message: &str, offending: Option<&str>) -> String {
    refusal_json(antecedent_core::reason_code!("invalid_argument"), detail, message, offending)
}

fn flag(map: &Map<String, Json>, key: &str) -> PyResult<bool> {
    match present(map, key) {
        None => Ok(false),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| crate::value_err(format!("field `{key}` must be a boolean"))),
    }
}

fn opt_digest(map: &Map<String, Json>, key: &str) -> PyResult<Option<SemanticDigest>> {
    opt_text(map, key)?.map(|t| digest(&t, key)).transpose()
}

fn count(map: &Map<String, Json>, key: &str) -> PyResult<u64> {
    member(map, key)?
        .as_u64()
        .ok_or_else(|| crate::value_err(format!("field `{key}` must be a non-negative integer")))
}

fn small(map: &Map<String, Json>, key: &str) -> PyResult<u32> {
    u32::try_from(count(map, key)?)
        .map_err(|_| crate::value_err(format!("field `{key}` is out of range")))
}

// ---------------------------------------------------------------------------
// Recalculation plan.
// ---------------------------------------------------------------------------

fn standing_of(value: &Json) -> PyResult<ClaimStanding> {
    let map = object(value, "claim standing")?;
    let mut standing = ClaimStanding::native_point();
    if let Some(unsupported) = present(map, "unsupported") {
        let items = unsupported
            .as_array()
            .ok_or_else(|| crate::value_err("`unsupported` must be a list"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| crate::value_err("`unsupported` holds a non-string"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        standing.support = SupportStanding::Unsupported { coordinates: items };
    }
    if let Some(trust) = opt_text(map, "trust")? {
        standing.trust = TrustStanding::from_name(&trust)
            .ok_or_else(|| crate::value_err(format!("unknown trust standing `{trust}`")))?;
    }
    if let Some(calibration) = present(map, "calibration") {
        standing.calibration = parse_calibration(calibration)?;
    }
    Ok(standing)
}

/// Build composition nodes from the stages of a recalculation plan, and read the effects the
/// plan implies.
///
/// `previous_json` and `requested_json` are lists of `{"stage", "own"}`; `capabilities_json` is
/// the capabilities object `plan_recalculation` reads; `bindings_json` is a list of
/// `{"key", "kind", "stage"}` with an optional `"standing"` (unsupported coordinates, trust,
/// calibration). Returns `{"nodes": [...], "effects": [...]}`, or the refusal for a stage the
/// plan does not contain.
#[pyfunction]
fn composition_plan_declaration(
    previous_json: &str,
    requested_json: &str,
    capabilities_json: &str,
    bindings_json: &str,
) -> PyResult<Declared> {
    let previous: Vec<DeclaredStageWire> = decode(previous_json, "previous")?;
    let requested: Vec<DeclaredStageWire> = decode(requested_json, "requested")?;
    let capabilities: CapabilitiesWire = decode(capabilities_json, "capabilities")?;
    let plan = plan_from_wire(&previous, &requested, &capabilities)
        .map_err(|e| crate::value_err(e.to_string()))?;
    let parsed = parse(bindings_json, "stage bindings")?;
    let items = parsed.as_array().ok_or_else(|| crate::value_err("bindings must be a list"))?;
    let mut nodes = Vec::with_capacity(items.len());
    let mut bound: Vec<(String, Stage)> = Vec::with_capacity(items.len());
    for item in items {
        let map = object(item, "stage binding")?;
        let key = text(map, "key")?;
        let kind_name = text(map, "kind")?;
        let kind = NodeKind::from_name(&kind_name)
            .ok_or_else(|| crate::value_err(format!("unknown node kind `{kind_name}`")))?;
        let label = text(map, "stage")?;
        let stage = stage_from_label(&label)
            .ok_or_else(|| crate::value_err(format!("unknown stage `{label}`")))?;
        let node = match present(map, "standing") {
            Some(standing) => {
                stage_node_with_standing(&key, kind, &plan, stage, standing_of(standing)?)
            }
            None => stage_node(&key, kind, &plan, stage),
        };
        let Some(node) = node else {
            return Ok((
                None,
                Some(invalid(
                    "composition_verification.stage_not_in_plan",
                    &format!("the requested workflow does not contain stage `{label}`"),
                    Some(&label),
                )),
            ));
        };
        nodes.push(node_wire(&node));
        bound.push((key, stage));
    }
    let bindings: Vec<(&str, Stage)> = bound.iter().map(|(k, s)| (k.as_str(), *s)).collect();
    let effects = effects_wire(&stage_effects(&plan, &bindings));
    Ok((Some(json!({"nodes": nodes, "effects": effects}).to_string()), None))
}

fn decode<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> PyResult<T> {
    crate::bridge::check_declared_size(
        text,
        crate::bridge::MAX_CONFIG_DECLARATION_BYTES,
        &format!("composition {what}"),
    )?;
    serde_json::from_str(text)
        .map_err(|e| crate::value_err(format!("invalid composition {what} declaration: {e}")))
}

// ---------------------------------------------------------------------------
// Evidence ledger.
// ---------------------------------------------------------------------------

/// The effects of the standings of ledger results bound to node keys: `bindings_json` maps a
/// node key to a result id. A result the ledger does not hold is a refusal.
#[pyfunction]
fn composition_ledger_effects(
    ledger: PyRef<'_, EvidenceLedgerHandle>,
    bindings_json: &str,
) -> PyResult<Declared> {
    let parsed = parse(bindings_json, "ledger bindings")?;
    let map = object(&parsed, "ledger bindings")?;
    let mut bound: Vec<(String, ResultId)> = Vec::with_capacity(map.len());
    for (key, value) in map {
        let id = value
            .as_str()
            .and_then(ResultId::from_hex)
            .ok_or_else(|| crate::value_err("a result id is 64 hexadecimal characters"))?;
        bound.push((key.clone(), id));
    }
    let bindings: Vec<(&str, ResultId)> = bound.iter().map(|(k, id)| (k.as_str(), *id)).collect();
    Ok(match ledger_effects(ledger.ledger(), &bindings) {
        Ok(effects) => (Some(effects_wire(&effects).to_string()), None),
        Err(refusal) => {
            (None, Some(refusal_json(refusal.code, &refusal.detail, &refusal.message, None)))
        }
    })
}

// ---------------------------------------------------------------------------
// Claim freshness and contract identities.
// ---------------------------------------------------------------------------

fn identities_of(value: &Json) -> PyResult<ContractIdentities> {
    let map = object(value, "contract identities")?;
    let hex = |key: &str| digest(&text(map, key)?, &format!("identity `{key}`"));
    Ok(ContractIdentities::new(
        hex("target")?,
        hex("identification")?,
        opt_digest(map, "identification_product")?,
        opt_digest(map, "program")?,
        hex("inference_binding")?,
        hex("observation")?,
        hex("data_snapshot")?,
    ))
}

struct ClaimBinding {
    key: String,
    claim: ClaimEnvelope,
    freshness: antecedent_contract::claim::ClaimFreshness,
    current: Option<ContractIdentities>,
}

fn claim_bindings(bindings_json: &str) -> PyResult<Vec<ClaimBinding>> {
    let parsed = parse(bindings_json, "claim bindings")?;
    let items = parsed.as_array().ok_or_else(|| crate::value_err("bindings must be a list"))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let map = object(item, "claim binding")?;
        let claim = parse_claim(member(map, "claim")?)?;
        let freshness = claim.freshness(
            opt_digest(map, "last_verified_snapshot")?,
            opt_digest(map, "current_snapshot")?,
            opt_digest(map, "superseded_by")?,
            flag(map, "withdrawn")?,
            flag(map, "locally_ready")?,
        );
        let current = present(map, "current").map(identities_of).transpose()?;
        out.push(ClaimBinding { key: text(map, "key")?, claim, freshness, current });
    }
    Ok(out)
}

/// The effects of claim freshness: a withdrawn claim is withdrawn, a superseded claim or one
/// whose snapshot no longer matches recomputes, an offline receiver learns nothing.
#[pyfunction]
fn composition_freshness_effects(bindings_json: &str) -> PyResult<String> {
    let bound = claim_bindings(bindings_json)?;
    let bindings: Vec<(&str, &_)> = bound.iter().map(|b| (b.key.as_str(), &b.freshness)).collect();
    Ok(effects_wire(&freshness_effects(&bindings)).to_string())
}

/// The effects of claim contracts against the identities the receiver now holds: every identity
/// layer is compared and the changed layers are named in the cause.
#[pyfunction]
fn composition_contract_effects(bindings_json: &str) -> PyResult<String> {
    let bound = claim_bindings(bindings_json)?;
    let bindings: Vec<(&str, &ClaimEnvelope, &_, Option<&ContractIdentities>)> = bound
        .iter()
        .map(|b| (b.key.as_str(), &b.claim, &b.freshness, b.current.as_ref()))
        .collect();
    Ok(effects_wire(&contract_effects(&bindings)).to_string())
}

// ---------------------------------------------------------------------------
// Invalidation log.
// ---------------------------------------------------------------------------

fn target_of(value: &Json) -> PyResult<InvalidationTarget> {
    let map = object(value, "invalidation target")?;
    Ok(match text(map, "kind")?.as_str() {
        "query_result" => InvalidationTarget::QueryResult(QueryId::from_raw(small(map, "id")?)),
        "model" => InvalidationTarget::Model(ModelId::from_raw(small(map, "id")?)),
        "all_results" => InvalidationTarget::AllResults,
        other => {
            return Err(crate::value_err(format!("unknown invalidation target `{other}`")));
        }
    })
}

fn log_of(log_json: &str) -> PyResult<InvalidationLog> {
    let parsed = parse(log_json, "invalidation log")?;
    let items = parsed.as_array().ok_or_else(|| crate::value_err("the log must be a list"))?;
    let mut log = InvalidationLog::new();
    for item in items {
        let map = object(item, "invalidation entry")?;
        let at = StateVersion::from_raw(count(map, "at_version")?);
        let target = target_of(member(map, "target")?)?;
        let reason = text(map, "reason")?;
        match present(map, "input") {
            None => log.push(at, target, reason),
            Some(input) => {
                let input = object(input, "changed input")?;
                let label = text(input, "stage")?;
                let stage = stage_from_label(&label)
                    .ok_or_else(|| crate::value_err(format!("unknown stage `{label}`")))?;
                let change = match text(input, "change")?.as_str() {
                    "added" => ChangeKind::Added,
                    "removed" => ChangeKind::Removed,
                    "modified" => ChangeKind::Modified,
                    other => {
                        return Err(crate::value_err(format!("unknown change `{other}`")));
                    }
                };
                let identity = StageIdentity::from_hex(&text(input, "identity")?)
                    .ok_or_else(|| crate::value_err("an input identity is 64 hex characters"))?;
                log.push_input(at, target, reason, ChangedInput { stage, change, identity });
            }
        }
    }
    Ok(log)
}

/// The effects of the invalidations recorded after `since`, one three-state vocabulary: a node
/// bound to a query result (or model) the log invalidated recomputes, a node whose premise the
/// log shows removed is withdrawn, and the most severe entry per node wins.
///
/// `bindings_json` is a list of `{"key", "target"}`.
#[pyfunction]
fn composition_invalidation_effects(
    log_json: &str,
    since: u64,
    bindings_json: &str,
) -> PyResult<String> {
    let log = log_of(log_json)?;
    let parsed = parse(bindings_json, "invalidation bindings")?;
    let items = parsed.as_array().ok_or_else(|| crate::value_err("bindings must be a list"))?;
    let mut bound: Vec<(String, InvalidationTarget)> = Vec::with_capacity(items.len());
    for item in items {
        let map = object(item, "invalidation binding")?;
        bound.push((text(map, "key")?, target_of(member(map, "target")?)?));
    }
    let bindings: Vec<(&str, InvalidationTarget)> =
        bound.iter().map(|(k, t)| (k.as_str(), t.clone())).collect();
    let effects = invalidation_effects(&log, StateVersion::from_raw(since), &bindings);
    Ok(effects_wire(&effects).to_string())
}

/// The three-state class of a query in the log after `since`: `still_valid`, `recompute` or
/// `withdrawn`.
#[pyfunction]
fn composition_query_revalidation(log_json: &str, query: u32, since: u64) -> PyResult<String> {
    let log = log_of(log_json)?;
    Ok(query_revalidation(&log, QueryId::from_raw(query), StateVersion::from_raw(since))
        .as_str()
        .to_owned())
}

// ---------------------------------------------------------------------------
// Discovery.
// ---------------------------------------------------------------------------

/// Classify every claim of the current claim graph against the earlier one after a discovery
/// change. The discovery nodes carry their own identities, so a new dataset, a new threshold, new
/// background knowledge or a changed completion is visible as an identity or structure change; no
/// further observed effect is read. `(classification json, refusal json)`.
#[pyfunction]
fn selective_discovery_revalidate(previous_json: &str, current_json: &str) -> PyResult<Declared> {
    let previous = parse(previous_json, "previous claim graph")?;
    let current = parse(current_json, "current claim graph")?;
    let previous = match parse_graph(&previous, "previous claim graph")? {
        Ok(graph) => graph,
        Err(error) => return Ok((None, Some(graph_refusal(&error)))),
    };
    let current = match parse_graph(&current, "current claim graph")? {
        Ok(graph) => graph,
        Err(error) => return Ok((None, Some(graph_refusal(&error)))),
    };
    Ok(match revalidate_after_discovery(&previous, &current) {
        Ok(result) => (
            Some(
                json!({
                    "claims": result.claims().map(claim_json).collect::<Vec<_>>(),
                    "removed": result.removed(),
                    "previous_identity": result.previous_identity().to_hex(),
                    "current_identity": result.current_identity().to_hex(),
                    "identity": result.identity().to_hex(),
                })
                .to_string(),
            ),
            None,
        ),
        Err(error) => (None, Some(graph_refusal(&error))),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(composition_plan_declaration, module)?)?;
    module.add_function(wrap_pyfunction!(composition_ledger_effects, module)?)?;
    module.add_function(wrap_pyfunction!(composition_freshness_effects, module)?)?;
    module.add_function(wrap_pyfunction!(composition_contract_effects, module)?)?;
    module.add_function(wrap_pyfunction!(composition_invalidation_effects, module)?)?;
    module.add_function(wrap_pyfunction!(composition_query_revalidation, module)?)?;
    module.add_function(wrap_pyfunction!(selective_discovery_revalidate, module)?)?;
    Ok(())
}
