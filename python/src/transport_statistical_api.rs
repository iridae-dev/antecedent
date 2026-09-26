//! Native authority for empirical-table statistical transport.
use crate::graphs::Admg;
use crate::transport_exact_api::ClassicalTransportStage;
use crate::transport_interference_api::parse_catalog;
use antecedent_core::{ExecutionContext, RegimeId, Value};
use antecedent_estimate::{
    EmpiricalTableEstimator, EmpiricalTableOptions, RegimeSample, StatisticalTransportInput,
};
use antecedent_expr::{
    ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData, InterventionAssignment,
};
use antecedent_identify::SidLimits;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::BTreeMap;
use std::sync::Arc;

type StatisticalStagePayload = (Vec<Vec<f64>>, Vec<f64>, String, Vec<String>, String);

use crate::transport_common::{
    RegimeCheck, assignment_from_pairs, error, execution_context, frame_named_artifact,
    parse_law_table, resolve, unframe_named_artifact,
};

const STATISTICAL_PREFIX: &[u8] = b"ANTECEDENT-STATISTICAL-TRANSPORT\x01";

#[pyclass(skip_from_py_object)]
struct PreparedStatisticalStage {
    inner: antecedent::PreparedStudy<antecedent::StatisticalPreparedState>,
    graph: Admg,
    catalog: antecedent_core::EvidenceCatalog,
    last: Option<antecedent::StatisticalStudyResult>,
    memory_bytes: Option<u64>,
    max_support_rows: usize,
    seed: u64,
}

impl PreparedStatisticalStage {
    fn ctx(&self, cancel: Option<crate::PyCancellationToken>) -> ExecutionContext {
        execution_context(self.seed, self.memory_bytes, cancel)
    }
    /// Posterior summary of one execution, shared by the payload and the getter.
    fn bayesian_posterior_json(
        &self,
        result: &antecedent::StatisticalStudyResult,
    ) -> Option<serde_json::Value> {
        result.bayesian_estimate().map(|posterior| {
            serde_json::json!({
                "estimator": posterior.estimator.as_ref(),
                "interval_method": posterior.interval_method.as_ref(),
                "draws_requested": posterior.draws_requested,
                "draws_ok": posterior.draws_ok,
                "draws_failed": posterior.draws_failed,
                "probabilities": posterior.distributions.iter().map(|d| d.probabilities.to_vec()).collect::<Vec<_>>(),
                "atom_intervals": posterior.atom_intervals.to_vec(),
                "mean_intervals": posterior.mean_intervals.iter().map(|(v, lo, hi)| (self.graph.names[v.as_usize()].as_str(), lo, hi)).collect::<Vec<_>>(),
                "calibration_status": posterior.interval_reason.as_deref().unwrap_or(antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED),
            })
        })
    }
    fn payload(
        &self,
        result: &antecedent::StatisticalStudyResult,
    ) -> PyResult<StatisticalStagePayload> {
        let est = result.estimate();
        let uncertainty = serde_json::json!({
            "available": est.uncertainty.is_some() && est.uncertainty_reason.is_none(),
            "reason": est.uncertainty_reason.as_deref(),
            "row": est.uncertainty.as_ref().map(|row| serde_json::json!({
                "estimator": row.estimator.as_ref(),
                "method": row.method.as_ref(),
                "coverage_target": row.coverage_target,
                "interval_scope": row.interval_scope.as_ref(),
                "sample_sizes": row.sample_sizes.iter().map(|(n,s)| (n.as_ref(), s)).collect::<Vec<_>>(),
                "support_regime": row.support_regime.as_ref(),
                "replicates_requested": row.replicates_requested,
                "replicates_ok": row.replicates_ok,
                "replicates_failed": row.replicates_failed,
                "seed": row.seed,
                "calibration_binding": row.calibration_binding.as_deref(),
                "calibration_status": if row.calibration_binding.is_some() { "bound" } else { "not_bound_to_this_execution" },
            })),
            "replicate_ids": est.replicate_ids.as_ref().map(|ids| ids.to_vec()),
            "atom_replicates": est.atom_replicates.as_ref().map(|rows| rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>()),
            "mean_replicates": est.mean_replicates.as_ref().map(|rows| rows.iter().map(|(v,r)| (self.graph.names[v.as_usize()].as_str(),r.to_vec())).collect::<Vec<_>>()),
            "atom_intervals": est.atom_intervals.as_ref().map(|rows| rows.to_vec()),
            "mean_intervals": est.mean_intervals.as_ref().map(|rows| {
                rows.iter().map(|(v, lo, hi)| (self.graph.names[v.as_usize()].as_str(), lo, hi)).collect::<Vec<_>>()
            }),
            "bayesian_posterior": self.bayesian_posterior_json(result),
        });
        Ok((
            result
                .distribution()
                .atoms
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|v| v.as_f64().ok_or_else(|| error("nonnumeric domain")))
                        .collect()
                })
                .collect::<PyResult<Vec<_>>>()?,
            result.distribution().probabilities.to_vec(),
            self.inner.inspect().formula,
            self.inner.rules().into_iter().map(str::to_owned).collect(),
            uncertainty.to_string(),
        ))
    }
}

#[pymethods]
impl PreparedStatisticalStage {
    #[getter]
    fn outcomes(&self) -> Vec<String> {
        self.inner.query().outcomes.iter().map(|v| self.graph.names[v.as_usize()].clone()).collect()
    }
    #[getter]
    fn bayesian_posterior(&self) -> Option<String> {
        self.last
            .as_ref()
            .and_then(|result| self.bayesian_posterior_json(result))
            .map(|value| value.to_string())
    }
    #[pyo3(signature=(execution=None,cancel=None))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        execution: Option<String>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<StatisticalStagePayload> {
        let ctx = self.ctx(cancel);
        let inner = self.inner.clone();
        let execution = execution.unwrap_or_else(|| inner.inspect().identities.execution);
        // A failed execution keeps the previous claim: nothing is cleared until
        // the replacement exists.
        let result = crate::detach_catch(py, move || {
            inner.estimate_checked(&execution, &ctx).map_err(error)
        })?;
        let payload = self.payload(&result)?;
        self.last = Some(result);
        Ok(payload)
    }
    #[pyo3(signature=(assignments,cancel=None))]
    fn estimate_grid(
        &self,
        py: Python<'_>,
        assignments: Vec<BTreeMap<String, f64>>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Vec<Self>> {
        let requests = assignments
            .into_iter()
            .map(|request| assignment_from_pairs(&self.graph.names, request))
            .collect::<PyResult<Vec<_>>>()?;
        let inner = self.inner.clone();
        let ctx = self.ctx(cancel);
        let points =
            crate::detach_catch(py, move || inner.estimate_grid(&requests, &ctx).map_err(error))?;
        Ok(points
            .into_iter()
            .map(|(inner, result)| Self {
                inner,
                graph: Admg { admg: self.graph.admg.clone(), names: self.graph.names.clone() },
                catalog: self.catalog.clone(),
                last: Some(result),
                memory_bytes: self.memory_bytes,
                max_support_rows: self.max_support_rows,
                seed: self.seed,
            })
            .collect())
    }
    #[pyo3(signature=(payload,cancel=None))]
    fn refresh(
        &mut self,
        py: Python<'_>,
        payload: &Bound<'_, PyAny>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<StatisticalStagePayload> {
        let input =
            parse_statistical_input(payload, &self.catalog, &self.graph, self.max_support_rows)?;
        let ctx = self.ctx(cancel);
        let mut candidate = self.inner.clone();
        let (candidate, result) = crate::detach_catch(py, move || {
            let result = candidate.refresh(input, &ctx).map_err(error)?;
            Ok((candidate, result))
        })?;
        self.inner = candidate;
        self.catalog = self.inner.evidence_catalog().clone();
        let out = self.payload(&result)?;
        self.last = Some(result);
        Ok(out)
    }
    #[pyo3(signature=(payload,cancel=None))]
    fn replace_snapshot(
        &mut self,
        py: Python<'_>,
        payload: &Bound<'_, PyAny>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let input =
            parse_statistical_input(payload, &self.catalog, &self.graph, self.max_support_rows)?;
        let ctx = self.ctx(cancel);
        let mut candidate = self.inner.clone();
        let candidate = crate::detach_catch(py, move || {
            candidate.replace_snapshot(input, &ctx).map_err(error)?;
            Ok(candidate)
        })?;
        self.inner = candidate;
        self.catalog = self.inner.evidence_catalog().clone();
        self.last = None;
        Ok(())
    }
    fn inspection_json(&self) -> String {
        let view = self.inner.inspect();
        let factors: Vec<_> = view.factors.iter().map(|f| serde_json::json!({
            "expression": f.expression.raw(),
            "population": f.binding.population.as_ref(),
            "regime": f.binding.regime.map(RegimeId::raw),
            "variables": f.variables.iter().map(|v| self.graph.names[v.as_usize()].as_str()).collect::<Vec<_>>(),
            "conditioned_on": f.conditioned_on.iter().map(|v| self.graph.names[v.as_usize()].as_str()).collect::<Vec<_>>(),
            "interventions": f.interventions.iter().map(|v| self.graph.names[v.as_usize()].as_str()).collect::<Vec<_>>(),
        })).collect();
        let executed = self.last.is_some();
        let findings: Vec<_> = self.last.iter().flat_map(|r| r.distribution().support.iter()).map(|record| serde_json::json!({
            "expression": record.expression.raw(), "status": record.status, "denominator": record.denominator,
            "assignment": record.assignment.iter().map(|(v,x)| (self.graph.names[v.as_usize()].as_str(), x.as_f64())).collect::<Vec<_>>()
        })).collect();
        let uncertainty = if let Some(result) = &self.last {
            if result.bayesian_estimate().is_some() {
                let available = result.reasoning().uncertainty.is_available();
                let reason = match &result.reasoning().uncertainty {
                    antecedent_core::SlotAvailability::Unavailable { reason } => {
                        Some(reason.as_ref())
                    }
                    _ => None,
                };
                serde_json::json!({
                    "available": available,
                    "reason": reason,
                    "summary": if available { "posterior_equal_tail" } else { "unavailable" },
                })
            } else {
                let est = result.estimate();
                serde_json::json!({
                    "available": est.uncertainty.is_some() && est.uncertainty_reason.is_none(),
                    "reason": est.uncertainty_reason.as_deref(),
                    "summary": if est.uncertainty.is_some() && est.uncertainty_reason.is_none() {
                        "percentile_bootstrap"
                    } else {
                        "unavailable"
                    },
                })
            }
        } else {
            let licensed = view.reasoning.uncertainty.is_available();
            let reason = match &view.reasoning.uncertainty {
                antecedent_core::SlotAvailability::Unavailable { reason } => Some(reason.as_ref()),
                _ => None,
            };
            serde_json::json!({
                "available": licensed,
                "reason": reason,
                "summary": if licensed { "percentile_bootstrap" } else { "unavailable" },
            })
        };
        let identification_status = view
            .reasoning
            .identification
            .as_ref()
            .map_or("unavailable", |slot| slot.status.as_str());
        serde_json::json!({
            "identification": {"available": view.reasoning.identification.is_available(), "summary": identification_status, "payload": {
                "formula": view.formula,
                "theorem_scope": view.theorem_scope,
                "rules": self.inner.rules(),
            }},
            "support": {"available": executed, "summary": if executed {"empirical_factor_support_checked"} else {"execution_specific"},
                "payload": {"factor_support": findings, "required_factors": factors, "bindings": view.bindings.iter().map(|b| {
                    serde_json::json!({"population": b.population, "regime": b.regime, "snapshot": b.snapshot, "origin": b.origin})
                }).collect::<Vec<_>>()}},
            "uncertainty": uncertainty,
            "assumptions": {"available": true, "summary": "declared_selection_diagram_and_empirical_tables"},
            "target_id": view.identities.target,
            "observation_id": view.identities.observation,
            "inference_binding_id": view.identities.inference_binding,
            "program_id": view.identities.program,
            "identification_id": view.identities.identification,
            "identification_product_id": view.identities.identification_product,
            "data_snapshot_id": view.identities.snapshot,
            "execution_id": view.identities.execution,
            "supported_operations": ["estimate", "replace_snapshot", "refresh", "export"],
            "formula": view.formula,
            "limitations": [
                "intervals_are_pointwise_not_simultaneous",
                "trial_ipw_does_not_evaluate_recursive_sid",
                "unknown_linked_or_clustered_dependence_withholds_intervals",
            ]
        })
        .to_string()
    }
    fn contrast(&self, reference: PyRef<'_, Self>, outcome: &str) -> PyResult<String> {
        let left = self.last.as_ref().ok_or_else(|| error("transport.no_execution_claim"))?;
        let right = reference.last.as_ref().ok_or_else(|| error("transport.no_execution_claim"))?;
        let contrast = left.contrast(right, resolve(&self.graph.names, outcome)?).map_err(error)?;
        Ok(serde_json::json!({ "estimate": contrast.estimate, "interval": contrast.interval,
            "coverage_target": contrast.coverage_target, "replicates_ok": contrast.replicates_ok,
            "replicates_failed": contrast.replicates_failed, "reason": contrast.reason,
            "method": "percentile_bootstrap", "interval_scope": "pointwise", "calibration_binding": null }).to_string())
    }
    fn last_result(&self) -> PyResult<StatisticalStagePayload> {
        self.payload(self.last.as_ref().ok_or_else(|| error("transport.no_execution_claim"))?)
    }
    fn plan_summary(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::from([
            ("plan_id".into(), self.inner.inspect().identities.execution),
            ("structure_source".into(), "explicit".into()),
            ("deterministic_reductions".into(), "true".into()),
            ("kernels".into(), "empirical_table_joint_outer_bootstrap".into()),
        ])
    }
    fn freeze(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            graph: Admg { admg: self.graph.admg.clone(), names: self.graph.names.clone() },
            catalog: self.catalog.clone(),
            last: self.last.clone(),
            memory_bytes: self.memory_bytes,
            max_support_rows: self.max_support_rows,
            seed: self.seed,
        }
    }
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let result = self.last.as_ref().ok_or_else(|| error("transport.no_execution_claim"))?;
        let artifact = self.inner.export(result).map_err(error)?;
        let framed = frame_named_artifact(STATISTICAL_PREFIX, &self.graph.names, artifact)?;
        Ok(pyo3::types::PyBytes::new(py, &framed).unbind())
    }
    fn preview_transform(
        &self,
        intent: &str,
    ) -> PyResult<std::collections::HashMap<String, String>> {
        let intent = crate::prepared_api::parse_transform_intent(intent)?;
        Ok(crate::prepared_api::transform_report_map(
            &self.inner.preview_transform(intent).map_err(error)?,
        ))
    }
}

pub(crate) fn parse_provider(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<EmpiricalTableEstimator> {
    let Some(value) = value else { return Ok(EmpiricalTableEstimator::Plugin) };
    if let Ok(name) = value.extract::<String>() {
        return match name.as_str() {
            "plugin" => Ok(EmpiricalTableEstimator::Plugin),
            "dirichlet" => Ok(EmpiricalTableEstimator::Dirichlet),
            "empirical_support_bayesian_bootstrap" => {
                Ok(EmpiricalTableEstimator::EmpiricalSupportBayesianBootstrap)
            }
            "state_space_dirichlet" => Ok(EmpiricalTableEstimator::StateSpaceDirichlet),
            _ => Err(error(format!("unknown statistical provider {name}"))),
        };
    }
    let wire = value.call_method0("_wire")?;
    let wire = wire.cast::<pyo3::types::PyDict>()?;
    let kind: String =
        wire.get_item("kind")?.ok_or_else(|| error("missing provider kind"))?.extract()?;
    match kind.as_str() {
        "empirical_table" => Ok(EmpiricalTableEstimator::Plugin),
        "empirical_support_bayesian_bootstrap" => {
            Ok(EmpiricalTableEstimator::EmpiricalSupportBayesianBootstrap)
        }
        "state_space_dirichlet" => Ok(EmpiricalTableEstimator::StateSpaceDirichlet),
        "learned_categorical" => {
            let spec = crate::estimator_config::get_learner(wire, "learner")?
                .ok_or_else(|| error("missing categorical learner"))?;
            Ok(EmpiricalTableEstimator::Learned(spec))
        }
        _ => Err(error("unknown statistical provider")),
    }
}

#[pyfunction]
#[pyo3(signature = (stage, catalog, payload, assignments, *, max_operations=10_000_000, max_depth=256, max_support_rows=1_000_000, memory_bytes=None, cancel=None, bootstrap=crate::transport_defaults::BOOTSTRAP, posterior_draws=199, coverage_level=crate::transport_defaults::COVERAGE_LEVEL, estimator=None, seed=1))]
#[allow(clippy::too_many_arguments)]
fn prepare_statistical_transport(
    py: Python<'_>,
    stage: PyRef<'_, ClassicalTransportStage>,
    catalog: &Bound<'_, PyAny>,
    payload: &Bound<'_, PyAny>,
    assignments: std::collections::BTreeMap<String, f64>,
    max_operations: usize,
    max_depth: usize,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
    bootstrap: u32,
    posterior_draws: u32,
    coverage_level: f64,
    estimator: Option<&Bound<'_, PyAny>>,
    seed: u64,
) -> PyResult<PreparedStatisticalStage> {
    let proof = stage.identified()?;
    let catalog = parse_catalog(catalog, stage.graph())?;
    let input = parse_statistical_input(payload, &catalog, stage.graph(), max_support_rows)?;
    let request = assignment_from_pairs(&stage.graph().names, assignments)?;
    let diagram = stage.diagram();
    let graph = stage.named_graph();
    let estimator = parse_provider(estimator)?;
    crate::detach_catch(py, move || {
        let ctx = execution_context(seed, memory_bytes, cancel);
        let functional = match proof.bind_catalog_with_context(
            &catalog,
            antecedent_identify::SidLimits { steps: max_operations, depth: max_depth },
            &ctx,
        ) {
            Ok(bound) => bound,
            Err(_) => match proof
                .search_catalog(
                    &diagram,
                    &catalog,
                    SidLimits { steps: max_operations, depth: max_depth },
                    &ctx,
                )
                .map_err(error)?
            {
                antecedent_identify::CatalogTransportResult::Identified(bound) => *bound,
                other => {
                    return Err(crate::transport_common::catalog_search_refusal(
                        &other,
                        &graph.names,
                    ));
                }
            },
        };
        let inner = antecedent::StudyBuilder::statistical_transport(
            diagram,
            functional,
            input,
            request,
            ExactEvaluationLimits { operations: max_operations, depth: max_depth },
            EmpiricalTableOptions {
                estimator,
                bootstrap_replicates: bootstrap,
                posterior_draws,
                coverage_level,
                max_joint_cells: max_support_rows,
            },
            &ctx,
        )
        .map_err(error)?;
        Ok(PreparedStatisticalStage {
            inner,
            graph,
            catalog,
            last: None,
            memory_bytes,
            max_support_rows,
            seed,
        })
    })
}

#[pyfunction]
#[pyo3(signature=(bytes,*,max_operations=10_000_000,max_depth=256,max_support_rows=1_000_000,memory_bytes=None,cancel=None,seed=1))]
#[allow(clippy::too_many_arguments)]
fn consume_statistical_transport(
    py: Python<'_>,
    bytes: Vec<u8>,
    max_operations: usize,
    max_depth: usize,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
    seed: u64,
) -> PyResult<PreparedStatisticalStage> {
    crate::detach_catch(py, move || {
        if memory_bytes.is_some_and(|n| bytes.len() as u64 > n) {
            return Err(error("artifact memory budget"));
        }
        let (names, artifact) =
            unframe_named_artifact(STATISTICAL_PREFIX, &bytes, "statistical transport")?;
        let ctx = execution_context(seed, memory_bytes, cancel);
        let (inner, result) =
            antecedent::PreparedStudy::<antecedent::StatisticalPreparedState>::consume(
                &artifact,
                ExactEvaluationLimits { operations: max_operations, depth: max_depth },
                &ctx,
            )
            .map_err(error)?;
        crate::transport_exact_api::validate_artifact_names(
            &names,
            inner.diagram().causal_graph(),
        )?;
        let graph = Admg { admg: inner.diagram().causal_graph().clone(), names };
        let catalog = inner.evidence_catalog().clone();
        let seed = inner.seed();
        Ok(PreparedStatisticalStage {
            inner,
            graph,
            catalog,
            last: Some(result),
            memory_bytes,
            max_support_rows,
            seed,
        })
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PreparedStatisticalStage>()?;
    m.add_function(wrap_pyfunction!(prepare_statistical_transport, m)?)?;
    m.add_function(wrap_pyfunction!(consume_statistical_transport, m)?)?;
    Ok(())
}

/// Parse a `StatisticalTransportData` payload: supplied exact laws plus finite
/// samples. Every sample and law must fit `max_support_rows`; a payload that is
/// not statistical data (for example a bare tuple of exact laws) is refused
/// rather than read as an empty sample set.
pub(crate) fn parse_statistical_input(
    payload: &Bound<'_, PyAny>,
    catalog: &antecedent_core::EvidenceCatalog,
    graph: &Admg,
    max_support_rows: usize,
) -> PyResult<StatisticalTransportInput> {
    let laws = if let Ok(laws) = payload.getattr("laws") {
        parse_optional_laws(&laws, catalog, graph, max_support_rows)?
    } else {
        Vec::new()
    };
    let samples_obj = payload.getattr("samples").map_err(|_| {
        crate::value_err(
            "statistical transport needs StatisticalTransportData with a samples field",
        )
    })?;
    let mut samples = Vec::new();
    for sample in samples_obj.try_iter()? {
        let sample = sample?;
        if sample.hasattr("probabilities")? || !sample.hasattr("columns")? {
            return Err(crate::value_err(
                "statistical transport samples must be RegimeSample values; exact laws belong in StatisticalTransportData.laws",
            ));
        }
        let parsed = parse_sample(&sample, catalog, graph)?;
        if parsed.columns.values().any(|column| column.len() > max_support_rows) {
            return Err(error(format!(
                "sample rows exceed the max_support_rows budget ({max_support_rows})"
            )));
        }
        samples.push(parsed);
    }
    Ok(StatisticalTransportInput { supplied: laws, samples })
}

fn parse_optional_laws(
    laws: &Bound<'_, PyAny>,
    catalog: &antecedent_core::EvidenceCatalog,
    graph: &Admg,
    max_support_rows: usize,
) -> PyResult<Vec<ExactDiscreteLaw>> {
    let mut tables = Vec::new();
    for table in laws.try_iter()? {
        let table = table?;
        if !table.hasattr("probabilities")? {
            continue;
        }
        tables.push(parse_law_table(&table, catalog, graph, RegimeCheck::Lenient)?);
    }
    // The support budget applies to supplied laws as it does to exact data.
    let data = ExactTransportData::try_new(tables, max_support_rows).map_err(error)?;
    Ok(data.laws().to_vec())
}

fn parse_sample(
    sample: &Bound<'_, PyAny>,
    catalog: &antecedent_core::EvidenceCatalog,
    graph: &Admg,
) -> PyResult<RegimeSample> {
    let population: String = sample.getattr("population")?.extract()?;
    let label: String = sample.getattr("regime")?.extract()?;
    let regime = catalog
        .regimes
        .iter()
        .find(|r| r.label.as_deref() == Some(label.as_str()) && r.population.as_ref() == population)
        .ok_or_else(|| error("sample names an unknown population/regime"))?;
    let interventions: Vec<(String, f64)> = sample.getattr("interventions")?.extract()?;
    let interventions = interventions
        .into_iter()
        .map(|(name, value)| {
            Ok(InterventionAssignment {
                variable: resolve(&graph.names, &name)?,
                value: Value::f64(value),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let columns_obj = sample.getattr("columns")?;
    let mapping = columns_obj
        .call_method0("items")?
        .try_iter()?
        .map(|item| item?.extract::<(String, Vec<Option<f64>>)>())
        .collect::<PyResult<Vec<_>>>()?;
    let mut columns = BTreeMap::new();
    let mut n = None;
    for (name, values) in mapping {
        if n.is_some_and(|expected| values.len() != expected) {
            return Err(error("sample columns must have equal length"));
        }
        n = Some(values.len());
        columns.insert(resolve(&graph.names, &name)?, values);
    }
    Ok(RegimeSample {
        population: population.into(),
        regime: regime.id,
        snapshot_identity: Arc::from(sample.getattr("snapshot_identity")?.extract::<String>()?),
        interventions: interventions.into(),
        columns,
    })
}
