//! Lossless identification records plus named coordinates for Python consumers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Identification;
use antecedent_core::{CausalQuery, TemporalIndexer, TemporalNodeKey, VariableId};
use antecedent_identify::{IdentificationEnvelope, IdentificationResult};
use pyo3::prelude::*;
use serde_json::{Value, json};

fn error(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

fn coordinate(
    id: VariableId,
    indexer: Option<&TemporalIndexer>,
    names: &[String],
) -> PyResult<Value> {
    let key = match indexer {
        Some(indexer) => indexer.key_of(id.raw()).map_err(error)?,
        None => TemporalNodeKey::contemporaneous(id),
    };
    let name = names
        .get(key.variable.as_usize())
        .ok_or_else(|| error("identification variable is outside the supplied schema"))?;
    Ok(json!({"name": name, "variable": key.variable.raw(), "offset": key.offset}))
}

fn record(
    result: &IdentificationResult,
    indexer: Option<&TemporalIndexer>,
    names: &[String],
) -> PyResult<Value> {
    let wire = antecedent_io::identification_to_wire(result).map_err(error)?;
    let adjustments = result
        .estimands
        .iter()
        .map(|e| {
            e.adjustment_set
                .iter()
                .map(|&v| coordinate(v, indexer, names))
                .collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(json!({
        "identification": wire,
        "adjustment_coordinates": adjustments,
        "indexer": indexer.map(|i| json!({"variable_count":i.variable_count(),"history":i.history(),"horizon":i.horizon()})),
    }))
}

fn envelope<G>(
    envelope: &IdentificationEnvelope<G>,
    indexers: &[TemporalIndexer],
    names: &[String],
    graph: impl Fn(&G) -> PyResult<Value>,
) -> PyResult<Value> {
    let cases = envelope
        .cases
        .iter()
        .enumerate()
        .map(|(i, case)| {
            let mut value = record(&case.result, indexers.get(i), names)?;
            value["weight"] = json!(case.weight.0);
            value["graph"] = graph(&case.graph)?;
            Ok(value)
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(json!({
        "cases": cases,
        "identified_weight": envelope.identified_weight.0,
        "unidentified_weight": envelope.unidentified_weight.0,
        "truncated_completions": envelope.truncated_completions,
        "features": envelope.critical_graph_features.iter().map(|f| json!({"kind":f.kind.as_ref(),"detail":f.detail.as_ref()})).collect::<Vec<_>>(),
    }))
}

pub(crate) fn analysis_to_json(
    result: &antecedent::StudyResult,
    names: &[String],
) -> PyResult<Option<String>> {
    let Some(certificate) = &result.certificate else {
        return Ok(None);
    };
    if !matches!(
        certificate.query,
        CausalQuery::AverageEffect(_)
            | CausalQuery::ConditionalEffect(_)
            | CausalQuery::Response(_)
            | CausalQuery::TemporalEffect(_)
    ) {
        return Ok(None);
    }
    to_json(
        &certificate.identification,
        &certificate.query,
        names,
        &format!("{:?}", certificate.graph_class),
    )
    .map(Some)
}

pub(crate) fn to_json(
    identification: &Identification,
    query: &CausalQuery,
    names: &[String],
    graph_class: &str,
) -> PyResult<String> {
    let mut payload = match identification {
        Identification::Point { result, temporal_indexer, .. } => {
            let mut point = record(result, temporal_indexer.as_ref(), names)?;
            point["weight"] = json!(1.0);
            json!({"kind":"point", "cases":[point]})
        }
        Identification::CpdagEnvelope { envelope: e, .. } => envelope(e, &[], names, |g| {
            let text = antecedent_io::dag_to_json(g, Some(names)).map_err(error)?;
            serde_json::from_str(&text).map_err(error)
        })?,
        Identification::Envelope { envelope: e, .. } => envelope(e, &[], names, |g| {
            let text = antecedent_io::pag_to_json(g, Some(names)).map_err(error)?;
            serde_json::from_str(&text).map_err(error)
        })?,
        Identification::TemporalEnvelope { envelope: e, .. } => {
            envelope(&e.envelope, &e.indexers, names, |g| match g {
                antecedent_identify::TemporalCompletionGraph::Dag(dag) => {
                    serde_json::to_value(antecedent_io::temporal_dag_to_wire(dag).map_err(error)?)
                        .map_err(error)
                }
                antecedent_identify::TemporalCompletionGraph::Mag(mag) => {
                    let graph: Value = serde_json::from_str(
                        &antecedent_io::pag_to_json(&mag.as_static_pag_for_alg(), None)
                            .map_err(error)?,
                    )
                    .map_err(error)?;
                    let nodes: Vec<_> = mag
                        .nodes()
                        .iter()
                        .map(|node| match node {
                            antecedent_core::NodeRef::Lagged { variable, lag } => {
                                json!({"variable":variable.raw(),"offset":-i64::from(lag.raw())})
                            }
                            _ => unreachable!("TemporalPag has lagged nodes"),
                        })
                        .collect();
                    Ok(
                        json!({"kind":"temporal_mag","template_graph":graph,"template_coordinates":nodes}),
                    )
                }
            })?
        }
        _ => return Err(error("unsupported identification payload")),
    };
    payload["status"] = json!(format!("{:?}", identification.status()));
    payload["method"] = json!(identification.strategy().as_str());
    payload["graph_class"] = json!(graph_class);
    payload["structure_version"] = json!(identification.structure_version());
    payload["names"] = json!(names);
    payload["witness_query"] =
        serde_json::to_value(antecedent_io::causal_query_to_wire(query).map_err(error)?)
            .map_err(error)?;
    let (targets, outcome) = match query {
        CausalQuery::AverageEffect(q) => (
            vec![TemporalNodeKey::contemporaneous(q.treatment)],
            TemporalNodeKey::contemporaneous(q.outcome),
        ),
        CausalQuery::ConditionalEffect(q) => (
            vec![TemporalNodeKey::contemporaneous(q.inner.treatment)],
            TemporalNodeKey::contemporaneous(q.inner.outcome),
        ),
        CausalQuery::Response(q) => (
            q.functional
                .treatment_ids()
                .into_iter()
                .map(TemporalNodeKey::contemporaneous)
                .collect(),
            TemporalNodeKey::contemporaneous(
                q.functional.primary_pair().ok_or_else(|| error("missing response outcome"))?.1,
            ),
        ),
        CausalQuery::TemporalEffect(q) => (
            q.policy
                .active_offsets()
                .map_err(error)?
                .iter()
                .map(|&offset| TemporalNodeKey { variable: q.treatment, offset })
                .collect(),
            TemporalNodeKey { variable: q.outcome, offset: q.outcome_offset() },
        ),
        _ => return Err(error("query has no adjustment handoff coordinates")),
    };
    let named_key = |key: &TemporalNodeKey| -> PyResult<Value> {
        let name =
            names.get(key.variable.as_usize()).ok_or_else(|| error("unknown query coordinate"))?;
        Ok(json!({"name":name,"variable":key.variable.raw(),"offset":key.offset}))
    };
    payload["treatments"] = json!(targets.iter().map(named_key).collect::<PyResult<Vec<_>>>()?);
    payload["outcome"] = named_key(&outcome)?;
    serde_json::to_string(&payload).map_err(error)
}
