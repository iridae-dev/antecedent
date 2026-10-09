//! Actual bounded proposal-arrival point estimation and raw-provider artifact replay.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::{CausalSerializationError, detach_catch, value_err, with_reason_code};
use antecedent::analysis::proposal_arrival::{ArrivalArtifact, ArrivalRequest, MAX_ARRIVAL_BYTES};
use antecedent_core::{ExecutionContext, reason_code};
use antecedent_io::IoError;
use pyo3::prelude::*;
fn failure(e: IoError) -> PyErr {
    match e {
        IoError::Refused { code, message } => with_reason_code(value_err(message), code),
        other => CausalSerializationError::new_err(other.to_string()),
    }
}
fn invalid(e: serde_json::Error) -> PyErr {
    with_reason_code(
        value_err(format!("proposal_arrival.invalid_request: {e}")),
        reason_code!("invalid_argument"),
    )
}
fn report(a: &ArrivalArtifact) -> PyResult<String> {
    serde_json::to_string(&serde_json::json!({"identity":a.identity,"result":a.result}))
        .map_err(|e| CausalSerializationError::new_err(e.to_string()))
}
fn preserved_catalog(
    catalog: &Bound<'_, PyAny>,
    graph: &crate::graphs::Admg,
    repair: &[u8],
) -> PyResult<antecedent_core::EvidenceCatalog> {
    if graph.names.len() > 12 {
        return Err(with_reason_code(
            value_err("proposal_arrival.bounds_exceeded"),
            reason_code!("route_not_supported"),
        ));
    }
    for (field, maximum) in [("environments", 64), ("regimes", 128), ("bindings", 128)] {
        if catalog.getattr(field)?.len()? > maximum {
            return Err(with_reason_code(
                value_err("proposal_arrival.bounds_exceeded"),
                reason_code!("route_not_supported"),
            ));
        }
    }
    for env in catalog.getattr("environments")?.try_iter()? {
        if env?.getattr("variables")?.len()? > 12 {
            return Err(with_reason_code(
                value_err("proposal_arrival.bounds_exceeded"),
                reason_code!("route_not_supported"),
            ));
        }
    }
    let mut parsed = crate::transport_interference_api::parse_catalog(catalog, graph)?;
    // Public catalogs canonically sort labels; preserve the original artifact's numeric IDs
    // when a new label sorts before an existing label.
    let original = antecedent_design::RepairReportArtifact::from_bytes(repair).map_err(|e| {
        with_reason_code(
            value_err(format!("proposal_arrival.original_consumer_refused: {e}")),
            reason_code!("invalid_argument"),
        )
    })?;
    let base = original
        .data
        .catalog
        .as_ref()
        .ok_or_else(|| {
            with_reason_code(
                value_err("proposal_arrival.base_catalog_missing"),
                reason_code!("invalid_argument"),
            )
        })?
        .to_catalog()
        .map_err(failure)?;
    let mut next = base.regimes.iter().map(|r| r.id.raw()).max().map_or(0, |r| r.saturating_add(1));
    let remap: std::collections::BTreeMap<_, _> = parsed
        .regimes
        .iter()
        .map(|r| {
            let id = base
                .regimes
                .iter()
                .find(|b| b.population == r.population && b.label == r.label)
                .map_or_else(
                    || {
                        let id = antecedent_core::RegimeId::from_raw(next);
                        next = next.saturating_add(1);
                        id
                    },
                    |b| b.id,
                );
            (r.id, id)
        })
        .collect();
    for r in std::sync::Arc::make_mut(&mut parsed.regimes) {
        r.id = remap[&r.id];
    }
    for b in std::sync::Arc::make_mut(&mut parsed.bindings) {
        b.regime = remap[&b.regime];
    }
    Ok(parsed)
}
fn inspect_law_inputs(sources: [&Bound<'_, PyAny>; 2]) -> PyResult<()> {
    for laws in sources {
        if crate::recalc_bounds::sequence_len(laws, "proposal_arrival.invalid_request")? > 64 {
            return Err(with_reason_code(
                value_err("proposal_arrival.bounds_exceeded"),
                reason_code!("route_not_supported"),
            ));
        }
        for i in 0..laws.len()? {
            let law = laws.get_item(i)?;
            for name in ["probabilities", "empirical_counts"] {
                let values = law.getattr(name)?;
                if !values.is_none() && values.len()? > 65_536 {
                    return Err(with_reason_code(
                        value_err("proposal_arrival.bounds_exceeded"),
                        reason_code!("route_not_supported"),
                    ));
                }
            }
            let axes = law.getattr("axes")?;
            if axes.len()? > 12 {
                return Err(with_reason_code(
                    value_err("proposal_arrival.bounds_exceeded"),
                    reason_code!("route_not_supported"),
                ));
            }
            for axis in axes.try_iter()? {
                if axis?.get_item(1)?.len()? > 65_536 {
                    return Err(with_reason_code(
                        value_err("proposal_arrival.bounds_exceeded"),
                        reason_code!("route_not_supported"),
                    ));
                }
            }
        }
    }
    Ok(())
}
#[pyfunction]
#[allow(clippy::too_many_arguments)] // Original artifacts and raw typed providers are independent inputs.
fn estimate_proposal_arrival(
    py: Python<'_>,
    graph: &crate::graphs::Admg,
    catalog: &Bound<'_, PyAny>,
    base_laws: &Bound<'_, PyAny>,
    arrived_laws: &Bound<'_, PyAny>,
    repair: &[u8],
    ranking: &[u8],
    request_json: &str,
    seed: u64,
) -> PyResult<(String, Vec<u8>)> {
    if repair.len() > 8 * 1024 * 1024
        || ranking.len() > 8 * 1024 * 1024
        || request_json.len() > 8 * 1024 * 1024
    {
        return Err(with_reason_code(
            value_err("proposal_arrival.bounds_exceeded"),
            reason_code!("route_not_supported"),
        ));
    }
    let mut request: ArrivalRequest = serde_json::from_str(request_json).map_err(invalid)?;
    if !request.repair_artifact.is_empty() || !request.ranking_artifact.is_empty() {
        return Err(with_reason_code(
            value_err("proposal_arrival.duplicate_original_artifacts"),
            reason_code!("invalid_argument"),
        ));
    }
    let (names, repair_body) = crate::transport_common::unframe_named_artifact(
        crate::repair_api::REPAIR_PREFIX,
        repair,
        "repair",
    )?;
    if names != graph.names {
        return Err(with_reason_code(
            value_err(
                "proposal_arrival.original_consumer_refused: repair coordinate names differ from the supplied graph",
            ),
            reason_code!("invalid_argument"),
        ));
    }
    let parsed = preserved_catalog(catalog, graph, &repair_body)?;
    inspect_law_inputs([base_laws, arrived_laws])?;
    let parse_laws =
        |laws: &Bound<'_, PyAny>| -> PyResult<Vec<antecedent_io::exact_law_wire::ExactLawWire>> {
            laws.try_iter()?
                .map(|law| {
                    let parsed = crate::transport_common::parse_law_table(
                        &law?,
                        &parsed,
                        graph,
                        crate::transport_common::RegimeCheck::Strict,
                    )
                    .map_err(|error| {
                        with_reason_code(
                            value_err(format!(
                                "proposal_arrival.original_consumer_refused: {error}"
                            )),
                            reason_code!("invalid_argument"),
                        )
                    })?;
                    Ok(antecedent_io::exact_law_wire::ExactLawWire::from_law(&parsed))
                })
                .collect()
        };
    request.catalog =
        antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(&parsed);
    request.base_laws = parse_laws(base_laws)?;
    request.arrived_laws = parse_laws(arrived_laws)?;
    request.repair_artifact = repair_body;
    request.ranking_artifact = ranking.to_vec();
    detach_catch(py, move || {
        let artifact = ArrivalArtifact::produce(request, &ExecutionContext::for_tests(seed))
            .map_err(failure)?;
        Ok((report(&artifact)?, artifact.to_bytes().map_err(failure)?))
    })
}
#[pyfunction]
fn consume_proposal_arrival(
    py: Python<'_>,
    artifact: &[u8],
    expected_proposal: &str,
    seed: u64,
) -> PyResult<String> {
    if artifact.len() > MAX_ARRIVAL_BYTES {
        return Err(with_reason_code(
            value_err("proposal_arrival.bounds_exceeded"),
            reason_code!("route_not_supported"),
        ));
    }
    let bytes = artifact.to_vec();
    let expected = expected_proposal.to_owned();
    detach_catch(py, move || {
        let artifact =
            ArrivalArtifact::consume(&bytes, &expected, &ExecutionContext::for_tests(seed))
                .map_err(failure)?;
        report(&artifact)
    })
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_proposal_arrival, m)?)?;
    m.add_function(wrap_pyfunction!(consume_proposal_arrival, m)?)?;
    Ok(())
}
