//! Independent per-population selective discovery.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn per_population_report_json(report: &PerPopulationReport) -> String {
    let populations: Vec<Json> = report
        .populations
        .iter()
        .map(|p| {
            json!({
                "population": p.population,
                "rows": p.data_identity.rows,
                "data_identity": hex(&p.data_identity.digest),
                "columns": p
                    .data_identity
                    .columns
                    .iter()
                    .map(|(variable, digest)| json!({ "variable": variable.raw(), "digest": hex(digest) }))
                    .collect::<Vec<_>>(),
                "report": report_value(&p.report),
                "structure_identity": hex(&p.structure_identity),
                "result_identity": hex(&p.result_identity),
                "retained_statistics": p.retained_statistics,
                "receipts_identity": p.receipts_identity.as_ref().map(hex),
                "search_incomplete": p.search_incomplete,
                "unresolved_edges": p.unresolved_edges,
            })
        })
        .collect();
    let edges: Vec<Json> = report
        .edges
        .iter()
        .map(|edge| {
            let marks: Vec<Json> = edge
                .marks
                .iter()
                .map(|(population, marks)| {
                    json!({
                        "population": population,
                        "marks": marks.as_ref().map(|(at_a, at_b)| {
                            json!({ "at_a": at_a.as_str(), "at_b": at_b.as_str() })
                        }),
                    })
                })
                .collect();
            json!({
                "a": node_json(edge.a),
                "b": node_json(edge.b),
                "marks": marks,
                "agrees": edge.agrees(),
            })
        })
        .collect();
    json!({
        "kind": "descriptive_comparison",
        "populations": populations,
        "edges": edges,
        "differing_edges": report.differing_edges().len(),
        "identity": hex(&report.identity),
    })
    .to_string()
}

/// Per-population discovery: one retained statistic store per population, nothing pooled.
#[pyclass(name = "PerPopulationDiscoveryHandle", skip_from_py_object)]
pub(crate) struct PerPopulationDiscoveryHandle {
    inner: Option<PerPopulationDiscovery>,
}

#[pymethods]
impl PerPopulationDiscoveryHandle {
    #[new]
    fn new() -> Self {
        Self { inner: Some(PerPopulationDiscovery::new()) }
    }

    /// Run the contract on each population's columns separately. `columns[i]` are the
    /// searched columns (in `names` order) of `populations[i]`. Returns the report JSON or a
    /// refusal.
    #[pyo3(signature = (contract_json, populations, names, columns, *, threads=None, cancel=None))]
    #[allow(clippy::too_many_arguments, reason = "one request: contract, labels, data")]
    fn run(
        &mut self,
        py: Python<'_>,
        contract_json: &str,
        populations: Vec<String>,
        names: Vec<String>,
        columns: Vec<Vec<NumericColumn<'_>>>,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> Reply {
        let parsed = parse_json(contract_json)?;
        let map = contract_map(&parsed, "static")?;
        let seed = get_u64(map, "seed")?;
        let contract = static_contract(map, names.len())?;
        if populations.len() != columns.len() {
            return Err(input_err("give one set of columns per population"));
        }
        let batches = columns
            .iter()
            .map(|set| crate::numeric_columns_to_batch(&names, set))
            .collect::<PyResult<Vec<_>>>()?;
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let mut driver = self.inner.take().ok_or_else(|| input_err("handle is unavailable"))?;
        let ran: PyResult<(PerPopulationDiscovery, Result<String, SelectiveDiscoveryError>)> =
            crate::detach_catch(py, move || {
                let mut tables: Vec<TabularData> = Vec::with_capacity(batches.len());
                for batch in &batches {
                    let (data, _): (TabularData, _) =
                        crate::discovery_api::tabular_from_batch(batch)?;
                    tables.push(data);
                }
                let rows: Vec<PopulationRows<'_>> = populations
                    .iter()
                    .zip(&tables)
                    .map(|(population, data)| PopulationRows { population, data })
                    .collect();
                let result = driver.run(&contract, &rows, &ctx).map(per_population_report_json);
                Ok((driver, result))
            });
        match ran {
            Ok((driver, result)) => {
                self.inner = Some(driver);
                reply(result)
            }
            Err(error) => {
                self.inner = Some(PerPopulationDiscovery::new());
                Err(error)
            }
        }
    }

    /// The last report in which every population succeeded, as JSON.
    fn report_json(&self) -> Option<String> {
        self.inner.as_ref().and_then(PerPopulationDiscovery::report).map(per_population_report_json)
    }

    /// Add every population's discovery stages of the last committed report to a composition
    /// graph under `{prefix}.{population}`. Returns `{"graph", "keys"}` JSON, or a refusal of
    /// the graph.
    fn add_claims(&self, graph_json: Option<&str>, prefix: &str) -> Reply {
        let driver = self.inner.as_ref().ok_or_else(|| input_err("handle is unavailable"))?;
        let Some(mut graph) = claim_graph(graph_json)? else {
            return Ok((None, Some(claim_graph_refusal(graph_json)?)));
        };
        match add_per_population_claims(&mut graph, prefix, driver) {
            Ok(keys) => {
                let keyed: Vec<(Option<String>, DiscoveryClaimKeys)> =
                    keys.into_iter().map(|(label, k)| (Some(label), k)).collect();
                Ok((Some(claims_json(&graph, &keyed)), None))
            }
            Err(error) => {
                Ok((None, Some(crate::composition_verification_api::graph_refusal(&error))))
            }
        }
    }

    /// One population's retained outcome as JSON.
    fn outcome_json(&self, population: &str) -> Option<String> {
        self.inner
            .as_ref()
            .and_then(|d| d.adapter(population))
            .and_then(StaticSelectiveDiscovery::outcome)
            .map(outcome_json)
    }

    /// Statistics retained for one population.
    fn retained_statistics(&self, population: &str) -> usize {
        self.inner
            .as_ref()
            .and_then(|d| d.adapter(population))
            .map_or(0, StaticSelectiveDiscovery::retained_statistics)
    }

    /// Refusal of selection targets read from the edges that differ between populations.
    fn request_selection_targets(&self, candidates: Vec<u32>) -> String {
        let Some(report) = self.inner.as_ref().and_then(PerPopulationDiscovery::report) else {
            return refusal_json(
                antecedent::reason::reason_code!("invalid_argument"),
                STAGE,
                antecedent::reason::reason_details::selective_discovery::NOTHING_RETAINED,
                "no per-population discovery has run",
                "run per-population discovery before requesting selection targets",
            );
        };
        let ids: Vec<VariableId> = candidates.into_iter().map(VariableId::from_raw).collect();
        match request_selection_targets_from_differences(report, &ids) {
            Err(refusal) => structure_refusal(&refusal),
            Ok(never) => match never {},
        }
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PerPopulationDiscoveryHandle>()
}
