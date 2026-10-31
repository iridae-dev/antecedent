//! Estimate, retarget, export, and refute clicks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::NumericColumn;

use std::sync::Arc;

use antecedent::Study;
use antecedent_core::{
    AnomalyAttributionQuery, CausalQuery, ChangeAttributionQuery, PopulationSelector,
};
use antecedent_data::TableView;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::{
    ClickControls, FrameInput, PrepareRequest, PyPreparedAnalysis, contract_to_map,
    insert_reuse_identities, map_ate, parse_transform_intent, require_prepared_names,
    response_from_study, transform_report_map, with_graph,
};
use crate::response_api::ResponseAnalysisResult;
use crate::{
    AteAnalysisResult, ate_result_from_analysis, dag_from_named_edges, detach_catch, py_err,
    py_execution_context_ext, tabular_from_arrow_c_objs, tabular_from_numpy,
    tabular_from_py_columns,
};

#[pymethods]
impl PyPreparedAnalysis {
    /// Freeze GCM anomaly scores on a supplied explicit Dag.
    ///
    /// Compile assigns `gcm.parametric` / `gcm.fit`. Do not set identifier or
    /// estimator here.
    #[staticmethod]
    #[pyo3(signature = (names, columns, edges, targets, max_units, *, accepted=false, seed=1,
        threads=None, options=None, reference=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_anomaly_attribution(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        targets: Vec<String>,
        max_units: usize,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
        reference: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("an anomaly attribution query")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let target_ids = targets
                .iter()
                .map(|name| crate::graph_build::schema_var_id(data.schema(), name))
                .collect::<PyResult<Vec<_>>>()?;
            let mut query = AnomalyAttributionQuery::new(target_ids, max_units);
            if let Some((center, scale)) = reference {
                query =
                    query.with_reference(antecedent_core::AnomalyReference::fixed(center, scale));
            }
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let builder =
                with_graph(Study::tabular(data), dag, accepted, req.opts.discovery_algorithm())
                    .query(CausalQuery::AnomalyAttribution(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze GCM distribution-change Shapley on a supplied explicit Dag.
    ///
    /// Compile assigns `gcm.parametric` / `gcm.fit`. Do not set identifier or
    /// estimator here.
    #[staticmethod]
    #[pyo3(signature = (names, columns, edges, outcome, baseline_start, baseline_end,
        comparison_start, comparison_end, *, accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_change_attribution(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        outcome: String,
        baseline_start: usize,
        baseline_end: usize,
        comparison_start: usize,
        comparison_end: usize,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("a change attribution query")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let outcome_id = crate::graph_build::schema_var_id(data.schema(), &outcome)?;
            let query = ChangeAttributionQuery::new(
                outcome_id,
                PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
                PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
            );
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let builder =
                with_graph(Study::tabular(data), dag, accepted, req.opts.discovery_algorithm())
                    .query(CausalQuery::ChangeAttribution(query));
            req.finish_applied(builder, false)
        })
    }

    /// Weighted-mean retarget from the frozen score table. Does not refit.
    #[pyo3(signature = (weights, depends_on, *, seed=1, threads=None))]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "node ids are u32 by construction (DenseNodeId), so node positions fit u32"
    )]
    fn retarget(
        &mut self,
        py: Python<'_>,
        weights: Vec<f64>,
        depends_on: Vec<String>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<AteAnalysisResult> {
        self.refuse_design_retarget()?;
        // Retarget the scores of the execution this call follows: after
        // `estimate(data)` that is the handle rebound on `data`, not the one
        // frozen at prepare.
        let bound = self.last_study.clone().unwrap_or_else(|| Arc::clone(&self.inner));
        let inner = Arc::clone(&bound);
        let out_names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                crate::resolve_user_threads(threads),
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let schema_ids = depends_on
                .iter()
                .map(|name| {
                    out_names
                        .iter()
                        .position(|n| n == name)
                        .map(|i| antecedent_core::VariableId::from_raw(i as u32))
                        .ok_or_else(|| crate::value_err(format!("unknown depends_on {name}")))
                })
                .collect::<PyResult<Vec<_>>>()?;
            let result = inner.retarget(&weights, &schema_ids, &ctx).map_err(py_err)?;
            let mapped = ate_result_from_analysis(&out_names, &result, false, &ctx)?;
            Ok((mapped, result))
        })?;
        self.last_study = Some(bound);
        self.last = Some(Arc::new(result));
        self.last_seed = seed;
        self.last_threads = crate::resolve_user_threads(threads);
        Ok(mapped)
    }

    /// Re-execute the row-weight retarget an exported contract carries.
    ///
    /// Refuses with `row_weights_bound_to_snapshot` unless this handle holds
    /// the data snapshot and score table the weights were bound to.
    #[pyo3(signature = (artifact, *, seed=1, threads=None))]
    fn reexecute_retarget(
        &mut self,
        py: Python<'_>,
        artifact: Vec<u8>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<AteAnalysisResult> {
        self.refuse_design_retarget()?;
        let bound = self.last_study.clone().unwrap_or_else(|| Arc::clone(&self.inner));
        let inner = Arc::clone(&bound);
        let out_names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                crate::resolve_user_threads(threads),
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            crate::measured_inference_api::check_cancelled(&ctx)?;
            let consumed = antecedent_io::consume_analysis_result(&artifact).map_err(py_err)?;
            crate::measured_inference_api::check_cancelled(&ctx)?;
            let section =
                consumed.contract.and_then(|contract| contract.target_weights).ok_or_else(
                    || crate::value_err("artifact carries no row-weight retarget to re-execute"),
                )?;
            let result = inner.reexecute_retarget(&section, &ctx).map_err(py_err)?;
            let mapped = ate_result_from_analysis(&out_names, &result, false, &ctx)?;
            Ok((mapped, result))
        })?;
        self.last_study = Some(bound);
        self.last = Some(Arc::new(result));
        self.last_seed = seed;
        self.last_threads = crate::resolve_user_threads(threads);
        Ok(mapped)
    }

    /// Re-estimate on new columns (same schema) without recompiling.
    #[pyo3(signature = (names, columns, *, seed=1, threads=None, cancel=None, on_progress=None, on_stage=None))]
    #[allow(clippy::too_many_arguments)]
    fn estimate(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        on_stage: Option<Bound<'_, PyAny>>,
    ) -> PyResult<AteAnalysisResult> {
        require_prepared_names(&self.names, &names, "estimate")?;
        let controls = ClickControls::parse(cancel, on_progress.as_ref(), on_stage.as_ref())?;
        let data = FrameInput::Series(tabular_from_py_columns(names, columns)?.0);
        self.click(py, data, seed, threads, controls, map_ate)
    }

    /// Re-estimate a prepared response (same schema) without recompiling.
    #[pyo3(signature = (names, columns, *, seed=1, threads=None, cancel=None, on_progress=None, on_stage=None))]
    #[allow(clippy::too_many_arguments)]
    fn estimate_response(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        on_stage: Option<Bound<'_, PyAny>>,
    ) -> PyResult<ResponseAnalysisResult> {
        require_prepared_names(&self.names, &names, "estimate")?;
        let controls = ClickControls::parse(cancel, on_progress.as_ref(), on_stage.as_ref())?;
        let data = FrameInput::Series(tabular_from_py_columns(names, columns)?.0);
        self.click(py, data, seed, threads, controls, response_from_study)
    }

    /// Export the retained full posterior or response without refitting.
    #[pyo3(signature = (*, artifact_id="prepared-result", payload="result"))]
    fn export_artifact<'py>(
        &self,
        py: Python<'py>,
        artifact_id: &str,
        payload: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let result = self
            .last
            .as_ref()
            .ok_or_else(|| crate::value_err("estimate before exporting an artifact"))?;
        let bytes = if payload == "query" {
            let query = antecedent_io::CausalPayloadWire::Query(Box::new(
                antecedent_io::causal_query_to_wire_with_registry(
                    self.inner.query(),
                    self.inner.population_registry(),
                )
                .map_err(py_err)?,
            ));
            let artifact = antecedent_io::encode_causal_payload_artifact(
                &query,
                self.names.clone(),
                artifact_id,
            )
            .map_err(py_err)?;
            let mut bytes = Vec::new();
            artifact.write_to(&mut bytes).map_err(py_err)?;
            bytes
        } else if payload != "result" {
            return Err(crate::value_err("payload must be 'query' or 'result'"));
        } else if result.mediation_grid().is_some()
            || result.structural_response.is_some()
            || (result.posterior.is_some() && result.response().is_some())
        {
            let wire = result
                .analysis_result_wire_with_context(
                    self.inner.query(),
                    self.inner.population_registry(),
                    self.inner.temporal_identification().as_deref(),
                )
                .map_err(py_err)?;
            let artifact = antecedent_io::encode_analysis_result_artifact(
                &wire,
                self.names.clone(),
                artifact_id,
            )
            .map_err(py_err)?;
            let mut bytes = Vec::new();
            artifact.write_to(&mut bytes).map_err(py_err)?;
            bytes
        } else if let Some(post) = &result.posterior {
            antecedent_io::encode_causal_posterior_bytes(post, artifact_id).map_err(py_err)?
        } else if let Some(response) = result.response() {
            let payload = antecedent_io::CausalPayloadWire::ResponseResult(Box::new(
                antecedent_io::causal_response_to_wire(response).map_err(py_err)?,
            ));
            let artifact = antecedent_io::encode_causal_payload_artifact(
                &payload,
                self.names.clone(),
                artifact_id,
            )
            .map_err(py_err)?;
            let mut bytes = Vec::new();
            artifact.write_to(&mut bytes).map_err(py_err)?;
            bytes
        } else if result.mediation().is_some()
            || result.counterfactual().is_some()
            || result.counterfactual_contrasts().is_some()
        {
            let wire = result
                .static_result_wire(self.inner.query(), self.inner.population_registry())
                .map_err(|e| match e {
                    // An incomplete payload stays the ValueError it always was.
                    antecedent::CausalError::Compile { message } => crate::value_err(message),
                    antecedent::CausalError::Rejected(failure) => {
                        crate::value_err(failure.message())
                    }
                    other => py_err(other),
                })?;
            let payload = antecedent_io::CausalPayloadWire::StaticResult(Box::new(wire));
            let artifact = antecedent_io::encode_causal_payload_artifact(
                &payload,
                self.names.clone(),
                artifact_id,
            )
            .map_err(py_err)?;
            let mut bytes = Vec::new();
            artifact.write_to(&mut bytes).map_err(py_err)?;
            bytes
        } else {
            return Err(crate::value_err(
                "retained result has no posterior or response artifact payload",
            ));
        };
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Export the last estimate as an `analysis_result` with a contract section.
    #[pyo3(signature = (*, artifact_id="prepared-contract", quantities_json=None))]
    fn export_contracted_artifact<'py>(
        &self,
        py: Python<'py>,
        artifact_id: &str,
        quantities_json: Option<&str>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let result = self
            .last
            .as_ref()
            .ok_or_else(|| crate::value_err("estimate before exporting a contracted artifact"))?;
        let ctx = crate::py_execution_context_ext(
            self.last_seed,
            self.last_threads,
            None,
            None,
            Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
        );
        let quantities: Option<Vec<antecedent_contract::ScientificQuantity>> = quantities_json
            .map(|json| {
                let wires: Vec<antecedent_io::quantity_wire::ScientificQuantityWire> =
                    serde_json::from_str(json)
                        .map_err(|error| crate::value_err(error.to_string()))?;
                wires
                    .into_iter()
                    .map(|wire| {
                        antecedent_contract::ScientificQuantity::try_from(wire)
                            .map_err(crate::value_err)
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let labels = quantities
            .as_ref()
            .map(|quantities| {
                let first = quantities
                    .first()
                    .ok_or_else(|| crate::value_err("scientific coordinates cannot be empty"))?;
                Ok::<_, PyErr>(antecedent_contract::ResponseCoordinateLabels {
                    outcome_units: &first.units,
                    population_id: &first.population_id,
                    transform_id: &first.transform_id,
                })
            })
            .transpose()?;
        if let (Some(supplied), Some(labels)) = (&quantities, &labels) {
            let response = result
                .response()
                .ok_or_else(|| crate::value_err("scientific coordinates require a response"))?;
            let expected = antecedent_contract::response_coordinates(
                response,
                &|id| self.names.get(id.as_usize()).cloned(),
                labels,
            )
            .map_err(|error| crate::value_err(format!("{}: {}", error.code, error.detail)))?;
            if supplied.len() != expected.len()
                || supplied
                    .iter()
                    .zip(&expected)
                    .any(|(supplied, expected)| expected.require_same_coordinate(supplied).is_err())
            {
                return Err(crate::value_err(
                    "scientific coordinates do not match the executed native response",
                ));
            }
        }
        let bytes = self
            .last_study
            .as_ref()
            .unwrap_or(&self.inner)
            .encode_contracted_result_with_quantity_labels(
                result,
                artifact_id,
                &ctx,
                labels.as_ref(),
            )
            .map_err(py_err)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Refute against the retained immutable execution and table.
    #[pyo3(signature = (suite, *, seed=1, threads=None, cancel=None))]
    fn refute_bound(
        &mut self,
        py: Python<'_>,
        suite: Bound<'_, PyAny>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<AteAnalysisResult> {
        let prior =
            self.last.clone().ok_or_else(|| crate::value_err("call estimate before refute"))?;
        let refute_suite = crate::suite_from_refute(Some(&suite))?;
        let bound = self.last_study.clone().unwrap_or_else(|| Arc::clone(&self.inner));
        let inner = Arc::clone(&bound);
        let names = self.names.clone();
        let (mapped, result) = crate::detach_catch(py, move || {
            let ctx = crate::py_execution_context_ext(
                seed,
                crate::resolve_user_threads(threads),
                cancel.map(|c| c.inner),
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = crate::interrupt::bind_study_validators(&inner)?
                .refute_bound(&prior, refute_suite, &ctx)
                .map_err(crate::py_err)?;
            let mapped = crate::ate_result_from_analysis(&names, &result, false, &ctx)?;
            Ok((mapped, result))
        })?;
        self.last_study = Some(bound);
        self.last = Some(Arc::new(result));
        self.last_seed = seed;
        self.last_threads = crate::resolve_user_threads(threads);
        Ok(mapped)
    }

    /// Second-click refute against the last estimate (same schema data).
    #[pyo3(signature = (names, columns, suite, *, seed=1, threads=None, cancel=None))]
    fn refute(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<NumericColumn<'_>>,
        suite: Bound<'_, PyAny>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<AteAnalysisResult> {
        require_prepared_names(&self.names, &names, "refute")?;
        let data = tabular_from_numpy(&names, &columns)?;
        drop(columns);
        self.finish_ate_refute(py, data, suite, seed, threads, cancel)
    }

    #[pyo3(signature = (names, columns, suite, *, seed=1, threads=None, cancel=None))]
    fn refute_arrow_c(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        suite: Bound<'_, PyAny>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<AteAnalysisResult> {
        require_prepared_names(&self.names, &names, "refute")?;
        let (data, _) = tabular_from_arrow_c_objs(names, columns)?;
        self.finish_ate_refute(py, data, suite, seed, threads, cancel)
    }

    /// Whether the last retarget's weights changed the target population.
    ///
    /// A constant weight vector re-reports the prepared population, so the
    /// handle is not bound to the retarget's data snapshot.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.names.clone()
    }

    /// Physical-plan highlights retained from prepare (no recompile).
    fn plan_summary(&self) -> std::collections::HashMap<String, String> {
        let rec = &self.inner.plan().record;
        let mut out = std::collections::HashMap::new();
        out.insert("plan_id".into(), rec.plan_id.to_string());
        out.insert("structure_source".into(), self.inner.structure_source().as_str().to_string());
        if let Some(status) = self.inner.support_status() {
            out.insert("evidence_status".into(), status.as_str().to_string());
            if let Some(reason) = status.allowlist_reason() {
                out.insert("allowlist_reason".into(), reason.to_string());
            }
            if let Some(parent) = status.allowlist_parent() {
                out.insert("allowlist_parent".into(), parent.to_string());
            }
        }
        if let Some(b) = rec.estimated_peak_memory_bytes {
            out.insert("estimated_peak_memory_bytes".into(), b.to_string());
        }
        if let Some(b) = rec.workspace_bytes {
            out.insert("workspace_bytes".into(), b.to_string());
        }
        if let Some(b) = rec.batch_size {
            out.insert("batch_size".into(), b.to_string());
        }
        out.insert("worker_threads".into(), rec.worker_threads.to_string());
        out.insert("expected_python_crossings".into(), rec.expected_python_crossings.to_string());
        out.insert("deterministic_reductions".into(), rec.deterministic_reductions.to_string());
        let kernels: Vec<String> =
            rec.kernels.iter().map(|(name, k)| format!("{name}:{k:?}")).collect();
        out.insert("kernels".into(), kernels.join(","));
        out
    }

    /// Cheap inspect: four slots without using cached identification.
    fn inspect(&self) -> PyResult<std::collections::HashMap<String, String>> {
        let contract = self.inner.inspect().map_err(py_err)?;
        Ok(contract_to_map(
            &contract,
            self.inner.schema(),
            self.inner.query(),
            self.inner.population_registry(),
        ))
    }

    /// Domain-separated contract identities and four reasoning slots.
    fn contract(&self) -> PyResult<std::collections::HashMap<String, String>> {
        let contract = self.inner.contract().map_err(py_err)?;
        let mut report = contract_to_map(
            &contract,
            self.inner.schema(),
            self.inner.query(),
            self.inner.population_registry(),
        );
        insert_reuse_identities(
            &mut report,
            self.inner.score_reuse_identity().map_err(py_err)?,
            None,
        );
        Ok(report)
    }

    /// Transformation preview retaining all frozen contract input identities.
    fn preview_transform(
        &self,
        intent: String,
    ) -> PyResult<std::collections::HashMap<String, String>> {
        let intent = parse_transform_intent(&intent)?;
        let report = self.inner.preview_transform(intent).map_err(py_err)?;
        Ok(transform_report_map(&report))
    }
}
