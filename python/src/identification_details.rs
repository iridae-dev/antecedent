//! Lossless identification records plus named coordinates for Python consumers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Identification;
use antecedent_core::{CausalQuery, TemporalIndexer, TemporalNodeKey, VariableId};
use antecedent_identify::{IdentificationEnvelope, IdentificationResult};
use pyo3::prelude::*;
use serde_json::{Value, json};

fn error(e: impl std::fmt::Display) -> PyErr {
    crate::value_err(e.to_string())
}

fn temporal_error(e: antecedent_core::TemporalMaterializationError) -> PyErr {
    crate::py_err(antecedent::CausalError::from(
        antecedent_estimate::EstimationError::TemporalMaterialization(e),
    ))
}

fn resource(message: &str) -> PyErr {
    crate::py_err(antecedent::CausalError::Resource { message: message.into() })
}

fn cancelled() -> PyErr {
    crate::py_err(antecedent::CausalError::Cancelled { stage: "identification metadata" })
}

// Bound the coordinate vector, named JSON values and their serialized representation.
fn admit_targets(
    count: u64,
    names: &[String],
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<antecedent_core::execution::AdmissionLease> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    let max_name = names.iter().map(|name| name.len() as u64).max().unwrap_or(0);
    let width = max_name
        .checked_mul(12)
        .and_then(|n| n.checked_add(512))
        .ok_or_else(|| resource("metadata byte overflow"))?;
    let bytes = count
        .checked_mul(width)
        .and_then(|n| n.checked_add(1024))
        .ok_or_else(|| resource("metadata byte overflow"))?;
    budget
        .charge_work(count.checked_mul(width).ok_or_else(|| resource("metadata work overflow"))?)
        .map_err(temporal_error)?;
    if bytes > budget.limits().max_bytes {
        return Err(resource("metadata byte limit"));
    }
    ctx.acquire_admission(0, bytes).map_err(|e| match e {
        antecedent_core::execution::AdmissionError::Cancelled => cancelled(),
        other => resource(&other.to_string()),
    })
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
    registry: Option<&antecedent_core::PopulationRegistry>,
) -> PyResult<Value> {
    let wire =
        antecedent_io::identification_to_wire_with_registry(result, registry).map_err(error)?;
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
    registry: Option<&antecedent_core::PopulationRegistry>,
    graph: impl Fn(&G) -> PyResult<Value>,
) -> PyResult<Value> {
    let cases = envelope
        .cases
        .iter()
        .enumerate()
        .map(|(i, case)| {
            let mut value = record(&case.result, indexers.get(i), names, registry)?;
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
    ctx: &antecedent_core::ExecutionContext,
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
            | CausalQuery::Mediation(_)
    ) {
        return Ok(None);
    }
    to_json(
        &certificate.identification,
        &certificate.query,
        names,
        &format!("{:?}", certificate.graph_class),
        result.population_registry.as_ref(),
        ctx,
    )
    .map(Some)
}

/// A tier declaration is a known closure or two named scenarios, never a completion mixture.
/// Reuse the same lossless result/query codecs and named coordinate conversion.
pub(crate) fn tiered_to_json(
    result: &IdentificationResult,
    query: &CausalQuery,
    names: &[String],
    background: &antecedent_graph::TieredBackground,
    within_tier: &str,
) -> PyResult<String> {
    let tiers = background
        .tiers
        .iter()
        .map(|tier| {
            tier.iter()
                .map(|&variable| coordinate(variable, None, names))
                .collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;
    let payload = json!({
        "kind": "tiered_identification",
        "status": format!("{:?}", result.status),
        "method": antecedent::IdentifierId::GeneralizedAdjustment.as_str(),
        "graph_class": "tiered_background",
        "within_tier": within_tier,
        "tiers": tiers,
        "scenario_scope": if background.within_tier == antecedent_graph::WithinTier::Unknown {
            "two_declared_orientation_scenarios"
        } else { "known_tier_closure" },
        "names": names,
        "cases": [record(result, None, names, None)?],
        "witness_query": antecedent_io::causal_query_to_wire(query).map_err(error)?,
    });
    serde_json::to_string(&payload).map_err(error)
}

pub(crate) fn to_json(
    identification: &Identification,
    query: &CausalQuery,
    names: &[String],
    graph_class: &str,
    registry: Option<&antecedent_core::PopulationRegistry>,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<String> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    let mut budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    let mut target_lease = None;
    let mut payload = match identification {
        Identification::Point { result, temporal_indexer, .. } => {
            let mut point = record(result, temporal_indexer.as_ref(), names, registry)?;
            point["weight"] = json!(1.0);
            json!({"kind":"point", "cases":[point]})
        }
        Identification::CpdagEnvelope { envelope: e, .. } => {
            let mut payload = envelope(e, &[], names, registry, |g| {
                let text = antecedent_io::dag_to_json(g, Some(names)).map_err(error)?;
                serde_json::from_str(&text).map_err(error)
            })?;
            payload["kind"] = json!("cpdag_envelope");
            payload
        }
        Identification::Envelope { envelope: e, .. } => {
            let mut payload = envelope(e, &[], names, registry, |g| {
                let text = antecedent_io::pag_to_json(g, Some(names)).map_err(error)?;
                serde_json::from_str(&text).map_err(error)
            })?;
            payload["kind"] = json!("envelope");
            payload
        }
        Identification::TemporalEnvelope { envelope: e, .. } => {
            let mut payload = envelope(&e.envelope, &e.indexers, names, registry, |g| match g {
                antecedent_identify::TemporalCompletionGraph::Dag(dag) => {
                    serde_json::to_value(antecedent_io::temporal_dag_to_wire(dag).map_err(error)?)
                        .map_err(error)
                }
                antecedent_identify::TemporalCompletionGraph::Mag(mag) => {
                    let graph: Value = serde_json::from_str(
                        &antecedent_io::pag_to_json(
                            &mag.as_static_pag_for_alg().map_err(error)?,
                            None,
                        )
                        .map_err(error)?,
                    )
                    .map_err(error)?;
                    let nodes = mag
                        .nodes()
                        .iter()
                        .map(|node| match node {
                            antecedent_core::NodeRef::Lagged { variable, lag } => Ok(
                                json!({"variable":variable.raw(),"offset":-i64::from(lag.raw())}),
                            ),
                            _ => Err(error("temporal completion node is not lagged")),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(
                        json!({"kind":"temporal_mag","template_graph":graph,"template_coordinates":nodes}),
                    )
                }
            })?;
            if let Some(cases) = payload.get_mut("cases").and_then(|value| value.as_array_mut()) {
                for (case, completion) in cases.iter_mut().zip(e.envelope.cases.iter()) {
                    case["fingerprint"] = json!(completion.graph.fingerprint());
                }
            }
            payload["completion_keys"] = json!(
                e.envelope.cases.iter().map(|case| case.graph.fingerprint()).collect::<Vec<_>>()
            );
            payload["kind"] = json!("temporal_envelope");
            payload
        }
        _ => return Err(error("unsupported identification payload")),
    };
    payload["status"] = json!(format!("{:?}", identification.status()));
    payload["method"] = json!(identification.strategy().as_str());
    payload["graph_class"] = json!(graph_class);
    payload["structure_version"] = json!(identification.structure_version());
    payload["names"] = json!(names);
    payload["witness_query"] = serde_json::to_value(
        antecedent_io::causal_query_to_wire_with_registry(query, registry).map_err(error)?,
    )
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
        CausalQuery::Response(q) => {
            let targets = if let Some(temporal) = &q.temporal {
                let plan = antecedent_estimate::plan_from_response_query(q, &mut budget, ctx)
                    .map_err(|e| crate::py_err(antecedent::CausalError::from(e)))?;
                if let Some(overlays) = plan.as_ref().and_then(|plan| plan.mechanism_overlays()) {
                    let count = overlays.count() as u64;
                    target_lease = Some(admit_targets(count, names, &mut budget, ctx)?);
                    let mut targets = Vec::new();
                    targets
                        .try_reserve_exact(
                            usize::try_from(count)
                                .map_err(|_| resource("metadata target count"))?,
                        )
                        .map_err(|_| resource("metadata target allocation"))?;
                    for overlay in plan
                        .as_ref()
                        .and_then(|plan| plan.mechanism_overlays())
                        .expect("overlay plan")
                    {
                        if ctx.cancellation.is_cancelled() {
                            return Err(cancelled());
                        }
                        targets.push(TemporalNodeKey {
                            variable: overlay.node.variable,
                            offset: overlay.node.offset,
                        });
                    }
                    targets
                } else {
                    let variables = q.functional.treatment_ids();
                    let count = temporal
                        .policy
                        .offset_count()
                        .map_err(error)?
                        .checked_mul(variables.len() as u64)
                        .ok_or_else(|| resource("metadata target count overflow"))?;
                    target_lease = Some(admit_targets(count, names, &mut budget, ctx)?);
                    let offsets = temporal
                        .policy
                        .active_offsets_within(&mut budget, ctx)
                        .map_err(temporal_error)?;
                    let mut targets = Vec::new();
                    targets
                        .try_reserve_exact(
                            usize::try_from(count)
                                .map_err(|_| resource("metadata target count"))?,
                        )
                        .map_err(|_| resource("metadata target allocation"))?;
                    for variable in variables {
                        for &offset in offsets.iter() {
                            if ctx.cancellation.is_cancelled() {
                                return Err(cancelled());
                            }
                            targets.push(TemporalNodeKey { variable, offset });
                        }
                    }
                    targets
                }
            } else {
                q.functional
                    .treatment_ids()
                    .into_iter()
                    .map(TemporalNodeKey::contemporaneous)
                    .collect()
            };
            let offset = match &q.temporal {
                Some(temporal) => i32::try_from(
                    temporal
                        .horizons
                        .first()
                        .copied()
                        .and_then(|h| h.checked_sub(1))
                        .ok_or_else(|| error("missing or zero temporal horizon"))?,
                )
                .map_err(error)?,
                None => 0,
            };
            (
                targets,
                TemporalNodeKey {
                    variable: q
                        .functional
                        .primary_pair()
                        .ok_or_else(|| error("missing response outcome"))?
                        .1,
                    offset,
                },
            )
        }
        CausalQuery::TemporalEffect(q) => {
            let count = q.policy.offset_count().map_err(error)?;
            target_lease = Some(admit_targets(count, names, &mut budget, ctx)?);
            let offsets =
                q.policy.active_offsets_within(&mut budget, ctx).map_err(temporal_error)?;
            let mut targets = Vec::new();
            targets
                .try_reserve_exact(
                    usize::try_from(count).map_err(|_| resource("metadata target count"))?,
                )
                .map_err(|_| resource("metadata target allocation"))?;
            for &offset in offsets.iter() {
                if ctx.cancellation.is_cancelled() {
                    return Err(cancelled());
                }
                targets.push(TemporalNodeKey { variable: q.treatment, offset });
            }
            (
                targets,
                TemporalNodeKey { variable: q.outcome, offset: q.outcome_offset().map_err(error)? },
            )
        }
        CausalQuery::Mediation(q) => {
            let temporal = matches!(
                identification,
                Identification::TemporalEnvelope { .. }
                    | Identification::Point { temporal_indexer: Some(_), .. }
            );
            (
                vec![TemporalNodeKey {
                    variable: q.treatment,
                    offset: if temporal { -1 } else { 0 },
                }],
                TemporalNodeKey {
                    variable: q.outcome,
                    offset: if temporal {
                        i32::try_from(q.horizons[0].saturating_sub(1)).map_err(error)?
                    } else {
                        0
                    },
                },
            )
        }
        _ => return Err(error("query has no adjustment handoff coordinates")),
    };
    let named_key = |key: &TemporalNodeKey| -> PyResult<Value> {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let name =
            names.get(key.variable.as_usize()).ok_or_else(|| error("unknown query coordinate"))?;
        Ok(json!({"name":name,"variable":key.variable.raw(),"offset":key.offset}))
    };
    payload["treatments"] = json!(targets.iter().map(named_key).collect::<PyResult<Vec<_>>>()?);
    payload["outcome"] = named_key(&outcome)?;
    let output = serde_json::to_string(&payload).map_err(error)?;
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    drop(target_lease);
    Ok(output)
}

#[cfg(test)]
mod temporal_metadata_context_tests {
    use super::*;
    use antecedent_core::{AssumptionSet, ExecutionContext, TemporalEffectQuery, TemporalPolicy};
    use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord};

    #[test]
    fn temporal_to_json_uses_the_supplied_context_for_limits_and_cancellation() {
        Python::initialize();
        Python::attach(|py| {
            let q =
                TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                    .with_policy(TemporalPolicy::pulse(-1));
            let query = CausalQuery::TemporalEffect(q);
            let identification = Identification::Point {
                result: IdentificationResult::not_identified(
                    query.clone(),
                    DerivationTrace::default(),
                    AssumptionSet::new(),
                    IdentificationPerformanceRecord::default(),
                ),
                temporal_indexer: None,
                strategy: antecedent::IdentifierId::TemporalBackdoorUnfolded,
                structure_version: 0,
            };
            let names = vec!["t".to_owned(), "y".to_owned()];
            let original = ExecutionContext::for_tests(1);
            let retained = original.clone();
            let json =
                to_json(&identification, &query, &names, "temporal_dag", None, &retained).unwrap();
            let payload: Value = serde_json::from_str(&json).unwrap();
            assert_eq!(payload["treatments"][0]["offset"], -1);

            let mut limits = original.temporal_materialization_limits();
            limits.max_offsets = 0;
            let limited = original.with_temporal_materialization_limits(limits);
            let refusal = to_json(&identification, &query, &names, "temporal_dag", None, &limited)
                .unwrap_err();
            assert!(refusal.is_instance_of::<crate::errors::CausalResourceError>(py));

            original.cancellation.cancel();
            let refusal = to_json(&identification, &query, &names, "temporal_dag", None, &retained)
                .unwrap_err();
            assert!(refusal.is_instance_of::<crate::errors::CausalCancelledError>(py));
        });
    }
}
