//! Python bindings for finite graph/selection transport scenario sets.
use crate::transport_common::{
    RegimeCheck, assignment_from_pairs, error, execution_context, frame_named_artifact, parse_laws,
    resolve, unframe_named_artifact,
};
use crate::{graphs::Admg, transport_interference_api::parse_catalog};
use antecedent_core::{SearchLimits, Value, VariableId};
use antecedent_estimate::transport_scenarios::ScenarioSetReport;
use antecedent_expr::ExactEvaluationLimits;
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    ClassicalTransportQuery, SidLimits,
    sid::scenarios::{TransportScenario, TransportScenarioSet},
};
use antecedent_io::transport_scenario_artifact::{
    TransportScenarioArtifactWire, TransportScenarioConsumeLimits,
};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

const SCENARIO_PREFIX: &[u8] = b"ANTECEDENT-TRANSPORT-SCENARIOS\x01";
const SCOPE: &str = "finite_transport_scenarios_structural_envelope";

/// One Python scenario: `(name, graph, selections, weight)`.
type ScenarioTuple<'py> = (String, PyRef<'py, Admg>, Vec<String>, Option<f64>);

fn report_json(report: &ScenarioSetReport, names: &[String]) -> serde_json::Value {
    let name = |v: &VariableId| names[v.as_usize()].clone();
    serde_json::json!({
        "status": "available",
        "scope": SCOPE,
        "scenarios": report.scenarios.iter().map(|s| {
            let point = s.distribution.as_ref().map(|d| serde_json::json!({
                "outcomes": d.outcomes.iter().map(name).collect::<Vec<_>>(),
                "atoms": d.atoms.iter().map(|row| row.iter().map(Value::as_f64).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "probabilities": d.probabilities.as_ref(),
                "means": d.outcomes.iter().map(|o| (name(o), d.mean(*o).ok())).collect::<BTreeMap<_, _>>(),
            }));
            serde_json::json!({
                "name": s.name.as_ref(),
                "weight": s.weight,
                "status": s.status,
                "detail": s.detail.as_ref().map(|d| crate::transport_common::resolve_variable_ids(d, names)),
                "point": point,
            })
        }).collect::<Vec<_>>(),
        "masses": report.masses.iter().map(|m| serde_json::json!({
            "status": m.status, "count": m.count, "mass": m.mass,
        })).collect::<Vec<_>>(),
        "residual_mass": report.residual_mass,
        "envelope": report.envelope.as_ref().map(|e| serde_json::json!({
            "scenarios": e.scenarios.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "means": e.means.iter().map(|m| serde_json::json!({
                "outcome": name(&m.outcome),
                "lower": m.lower,
                "upper": m.upper,
                "lower_scenario": m.lower_scenario.as_ref(),
                "upper_scenario": m.upper_scenario.as_ref(),
            })).collect::<Vec<_>>(),
            "atoms": e.atoms,
            "interpretation": e.interpretation,
        })),
        "weighted": report.weighted.as_ref().map(|w| serde_json::json!({
            "identified_mass": w.identified_mass,
            "unaccounted_mass": w.unaccounted_mass,
            "identified_weighted_sums": w.identified_weighted_sums.iter().map(|(v, x)| (name(v), *x)).collect::<BTreeMap<_, _>>(),
            "ranges": w.ranges.as_ref().map(|r| r.iter().map(|(v, lo, hi)| serde_json::json!({
                "outcome": name(v), "lower": lo, "upper": hi,
            })).collect::<Vec<_>>()),
            "interpretation": w.interpretation,
        })),
        "receipt": report.receipt.as_ref().map(|r| serde_json::json!({
            "stop": r.stop.code(),
            "operations_limit": r.operations_limit,
            "operations_consumed": r.operations_consumed,
            "explored": r.explored,
            "unevaluated": r.unevaluated,
        })),
    })
}

/// Accept an `ExactTransportData` wrapper or a sequence of laws.
fn law_sequence<'py>(laws: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if laws.hasattr("laws")? { laws.getattr("laws") } else { Ok(laws.clone()) }
}

/// A decided, compiled scenario set; estimation never re-identifies.
#[pyclass(skip_from_py_object)]
struct PreparedTransportScenariosStage {
    inner: antecedent::PreparedTransportScenarios,
    graph: Admg,
    catalog: antecedent_core::EvidenceCatalog,
    last: Option<ScenarioSetReport>,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
}

#[pymethods]
impl PreparedTransportScenariosStage {
    /// Evaluate every compiled scenario; the report keeps every scenario.
    #[pyo3(signature=(*, memory_bytes=None, cancel=None))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = execution_context(0, memory_bytes.or(self.memory_bytes), cancel);
        let inner = self.inner.clone();
        let report = crate::detach_catch(py, move || inner.estimate(&ctx).map_err(error))?;
        let payload = report_json(&report, &self.graph.names).to_string();
        self.last = Some(report);
        Ok(payload)
    }

    /// Recompile identified scenarios against new laws; decisions are kept.
    #[pyo3(signature=(laws, *, memory_bytes=None, cancel=None))]
    fn refresh(
        &mut self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let data = parse_laws(
            &law_sequence(laws)?,
            &self.catalog,
            &self.graph,
            self.max_support_rows,
            RegimeCheck::Strict,
        )?;
        let ctx = execution_context(0, memory_bytes.or(self.memory_bytes), cancel);
        let inner = self.inner.clone();
        self.inner = crate::detach_catch(py, move || inner.refresh(data, &ctx).map_err(error))?;
        self.last = None;
        Ok(())
    }

    /// Each scenario's retained, frozen plan: `compiled`, the provider or support
    /// refusal of an identified scenario, or `not_identified:<status>`.
    fn plan_summary(&self) -> Vec<(String, String)> {
        self.inner.prepared().plan_summary().into_iter().map(|(n, k)| (n.to_string(), k)).collect()
    }

    /// Inference across scenarios sharing data is not licensed; always refuses.
    fn aggregate_interval(&self) -> PyResult<()> {
        self.inner.aggregate_interval().map_err(error)
    }

    /// The last report as a framed, independently consumable artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let report = self.last.as_ref().ok_or_else(|| {
            crate::refusal(
                antecedent_core::reason_code!("not_executed"),
                "transport.no_execution_claim: estimate before exporting a scenario artifact",
            )
        })?;
        let raw = self.inner.export(report).map_err(error)?;
        let framed = frame_named_artifact(SCENARIO_PREFIX, &self.graph.names, raw)?;
        Ok(pyo3::types::PyBytes::new(py, &framed).unbind())
    }
}

/// Decide every scenario once and compile the identified ones.
///
/// `scenarios` are `(name, graph, selections, weight)`; every graph must declare
/// the same variable names. Weights are given for every scenario or none.
#[pyfunction]
#[pyo3(signature=(scenarios, outcomes, treatments, source, target, catalog, laws, assignments, *, max_steps=100_000, max_depth=256, max_scenarios=64, max_operations=10_000_000, max_evaluation_depth=256, max_support_rows=1_000_000, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn prepare_transport_scenarios_stage(
    py: Python<'_>,
    scenarios: Vec<ScenarioTuple<'_>>,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    source: String,
    target: String,
    catalog: &Bound<'_, PyAny>,
    laws: &Bound<'_, PyAny>,
    assignments: BTreeMap<String, f64>,
    max_steps: usize,
    max_depth: usize,
    max_scenarios: usize,
    max_operations: usize,
    max_evaluation_depth: usize,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PreparedTransportScenariosStage> {
    let Some(first) = scenarios.first() else {
        return Err(crate::value_err(
            "scenarios.empty: a scenario set needs at least one scenario",
        ));
    };
    let names = first.1.names.clone();
    let named = Admg { admg: first.1.admg.clone(), names: names.clone() };
    let mut built = Vec::with_capacity(scenarios.len());
    for (name, graph, selections, weight) in &scenarios {
        let mut sorted_names = graph.names.clone();
        sorted_names.sort();
        let mut expected = names.clone();
        expected.sort();
        if sorted_names != expected {
            return Err(crate::value_err(format!(
                "scenarios.coordinate_mismatch: scenario {name} declares different variables"
            )));
        }
        let selections =
            selections.iter().map(|s| resolve(&names, s)).collect::<PyResult<Vec<_>>>()?;
        let diagram = SelectionDiagram::try_new(
            graph.aligned_to_names(&names)?,
            Arc::<[VariableId]>::from(selections),
        )
        .map_err(error)?;
        built.push(TransportScenario { name: Arc::from(name.as_str()), diagram, weight: *weight });
    }
    let set = TransportScenarioSet::try_new(built).map_err(error)?;
    let coordinates = |variables: &[String]| -> PyResult<Arc<[VariableId]>> {
        variables.iter().map(|n| resolve(&names, n)).collect::<PyResult<Vec<_>>>().map(Arc::from)
    };
    let query = ClassicalTransportQuery {
        outcomes: coordinates(&outcomes)?,
        treatments: coordinates(&treatments)?,
        source: source.into(),
        target: target.into(),
    };
    let catalog = parse_catalog(catalog, &named)?;
    let data =
        parse_laws(&law_sequence(laws)?, &catalog, &named, max_support_rows, RegimeCheck::Strict)?;
    let request = assignment_from_pairs(&names, assignments)?;
    let stage_catalog = catalog.clone();
    let inner = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        antecedent::StudyBuilder::transport_scenarios(
            &set,
            query,
            catalog,
            SidLimits { steps: max_steps, depth: max_depth },
            SearchLimits { operations: max_scenarios, depth: 1 },
            data,
            request,
            ExactEvaluationLimits { operations: max_operations, depth: max_evaluation_depth },
            &ctx,
        )
        .map_err(error)
    })?;
    Ok(PreparedTransportScenariosStage {
        inner,
        graph: named,
        catalog: stage_catalog,
        last: None,
        max_support_rows,
        memory_bytes,
    })
}

/// Independently replay a framed scenario artifact under the consumer's limits.
#[pyfunction]
#[pyo3(signature=(artifact, *, max_steps=100_000, max_depth=256, max_scenarios=64, max_operations=10_000_000, max_evaluation_depth=256, max_support_rows=1_000_000, max_laws=256, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn consume_transport_scenarios_artifact(
    py: Python<'_>,
    artifact: &[u8],
    max_steps: usize,
    max_depth: usize,
    max_scenarios: usize,
    max_operations: usize,
    max_evaluation_depth: usize,
    max_support_rows: usize,
    max_laws: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let (names, bytes) = unframe_named_artifact(SCENARIO_PREFIX, artifact, "transport scenario")?;
    let limits = TransportScenarioConsumeLimits {
        identification: SidLimits { steps: max_steps, depth: max_depth },
        scenario_budget: SearchLimits { operations: max_scenarios, depth: 1 },
        evaluation: ExactEvaluationLimits {
            operations: max_operations,
            depth: max_evaluation_depth,
        },
        max_support_rows,
        max_laws,
    };
    crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        let (wire, report) =
            TransportScenarioArtifactWire::consume_with_limits(&bytes, limits, &ctx)
                .map_err(error)?;
        for scenario in &wire.scenarios {
            let graph = antecedent_io::admg_from_wire(&scenario.graph).map_err(error)?;
            crate::transport_exact_api::validate_artifact_names(&names, &graph)?;
        }
        let mut payload = report_json(&report, &names);
        payload["premises_digest"] = wire.premises_digest.clone().into();
        Ok(payload.to_string())
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedTransportScenariosStage>()?;
    module.add_function(wrap_pyfunction!(prepare_transport_scenarios_stage, module)?)?;
    module.add_function(wrap_pyfunction!(consume_transport_scenarios_artifact, module)?)?;
    Ok(())
}
