//! Python bindings for finite graph/selection transport scenario sets.
use crate::transport_common::{
    RegimeCheck, assignment_from_pairs, error, execution_context, frame_named_artifact, parse_laws,
    resolve, serialization_error, unframe_named_artifact,
};
use crate::transport_statistical_api::parse_statistical_input;
use crate::{graphs::Admg, transport_interference_api::parse_catalog};
use antecedent_core::{SearchLimits, Value, VariableDomain, VariableId, reason_code};
use antecedent_estimate::transport_scenarios::ScenarioSetReport;
use antecedent_expr::ExactEvaluationLimits;
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    ClassicalTransportQuery, ConditionalTransportQuery,
    sid::scenarios::{ScenarioCoordinate, TransportScenario, TransportScenarioSet},
};
use antecedent_io::IoError;
use antecedent_io::transport_scenario_artifact::{
    TransportScenarioArtifactWire, TransportScenarioConsumeLimits, scenario_refusal,
};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

const SCENARIO_PREFIX: &[u8] = b"ANTECEDENT-TRANSPORT-SCENARIOS\x01";
const SCOPE: &str = "finite_transport_scenarios_structural_envelope";

/// One declared coordinate: `(name, domain, cardinality, unit)`.
type CoordinateTuple = (String, String, Option<u32>, Option<String>);
/// One Python scenario: `(name, graph, selections, weight, coordinates)`; the
/// coordinates override the set's when given.
type ScenarioTuple<'py> =
    (String, PyRef<'py, Admg>, Vec<String>, Option<f64>, Option<Vec<CoordinateTuple>>);

/// A reason-coded scenario refusal: a schema or argument refusal is a value
/// error carrying its code; any other refusal keeps its transport class.
fn scenario_error(e: IoError) -> PyErr {
    match e {
        IoError::Refused { code, message }
            if code == reason_code!("schema_mismatch")
                || code == reason_code!("invalid_argument") =>
        {
            crate::with_reason_code(crate::value_err(message), code)
        }
        other => error(other),
    }
}

fn schema_mismatch(message: String) -> PyErr {
    crate::with_reason_code(
        crate::value_err(format!("scenarios.coordinate_mismatch: {message}")),
        reason_code!("schema_mismatch"),
    )
}

/// Resolve declared coordinates against the shared variable names.
fn parse_coordinates(
    coordinates: &[CoordinateTuple],
    names: &[String],
    scenario: &str,
) -> PyResult<Arc<[ScenarioCoordinate]>> {
    coordinates
        .iter()
        .map(|(name, domain, cardinality, unit)| {
            let variable = names.iter().position(|n| n == name).ok_or_else(|| {
                schema_mismatch(format!("scenario {scenario} declares unknown variable {name}"))
            })?;
            let domain = match (domain.as_str(), cardinality) {
                ("unspecified", None) => VariableDomain::Unspecified,
                ("continuous", None) => VariableDomain::Continuous,
                ("binary", None) => VariableDomain::Binary,
                ("count", None) => VariableDomain::Count,
                ("categorical", Some(n)) => VariableDomain::Categorical { cardinality: *n },
                _ => return Err(crate::value_err(format!("invalid domain for {name}"))),
            };
            Ok(ScenarioCoordinate {
                variable: VariableId::from_raw(
                    u32::try_from(variable).map_err(|e| crate::value_err(e.to_string()))?,
                ),
                name: Arc::from(name.as_str()),
                domain,
                unit: unit.as_deref().map(Arc::from),
            })
        })
        .collect()
}

/// A scenario row's identification status in the shared transport vocabulary.
///
/// `status` keeps the serialized scenario spellings (`structurally_unidentified`,
/// `unevaluated`); the execution statuses `unsupported_provider` and
/// `support_failure` belong to identified scenarios whose evaluation then failed.
fn scenario_identification_status(status: &str) -> &'static str {
    antecedent_core::TransportOutcomeKind::from_identification_status(status)
        .unwrap_or(antecedent_core::TransportOutcomeKind::Identified)
        .as_str()
}

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
                "identification_status": scenario_identification_status(s.status),
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

/// Supplied exact laws, or `StatisticalTransportData` for the empirical plug-in.
enum Provider {
    Exact(antecedent_expr::ExactTransportData),
    Empirical(antecedent_estimate::StatisticalTransportInput),
}

fn parse_provider(
    laws: &Bound<'_, PyAny>,
    catalog: &antecedent_core::EvidenceCatalog,
    graph: &Admg,
    max_support_rows: usize,
) -> PyResult<Provider> {
    if laws.hasattr("samples")? {
        return parse_statistical_input(laws, catalog, graph, max_support_rows)
            .map(Provider::Empirical);
    }
    parse_laws(&law_sequence(laws)?, catalog, graph, max_support_rows, RegimeCheck::Strict)
        .map(Provider::Exact)
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

    /// Recompile identified scenarios against new laws, or refit the empirical
    /// plug-in from new samples; decisions are kept.
    #[pyo3(signature=(laws, *, memory_bytes=None, cancel=None))]
    fn refresh(
        &mut self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let provider = parse_provider(laws, &self.catalog, &self.graph, self.max_support_rows)?;
        let ctx = execution_context(0, memory_bytes.or(self.memory_bytes), cancel);
        let inner = self.inner.clone();
        let cells = self.max_support_rows;
        self.inner = crate::detach_catch(py, move || {
            match provider {
                Provider::Exact(data) => inner.refresh(data, &ctx),
                Provider::Empirical(input) => inner.refresh_empirical(input, cells, &ctx),
            }
            .map_err(scenario_error)
        })?;
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
/// `scenarios` are `(name, graph, selections, weight, coordinates)`; every graph
/// must declare the same variable names, and each scenario's coordinates (its
/// own, or the set's `coordinates`) must agree. Weights are given for every
/// scenario or none. `laws` are supplied exact laws or `StatisticalTransportData`
/// for empirical plug-in points. A non-empty `conditioned_on` asks the
/// conditional question `P*(y | do(x), w)` of the bounded ADMG conditional row
/// (2.2B B1) in every scenario; it takes exact laws only.
#[pyfunction]
#[pyo3(signature=(scenarios, coordinates, outcomes, treatments, source, target, catalog, laws, assignments, *, conditioned_on=Vec::new(), max_steps=100_000, max_depth=256, max_operations=10_000_000, max_evaluation_depth=256, max_support_rows=1_000_000, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn prepare_transport_scenarios_stage(
    py: Python<'_>,
    scenarios: Vec<ScenarioTuple<'_>>,
    coordinates: Vec<CoordinateTuple>,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    source: String,
    target: String,
    catalog: &Bound<'_, PyAny>,
    laws: &Bound<'_, PyAny>,
    assignments: BTreeMap<String, f64>,
    conditioned_on: Vec<String>,
    max_steps: usize,
    max_depth: usize,
    max_operations: usize,
    max_evaluation_depth: usize,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PreparedTransportScenariosStage> {
    let Some(first) = scenarios.first() else {
        return Err(crate::with_reason_code(
            crate::value_err("scenarios.empty: a scenario set needs at least one scenario"),
            reason_code!("invalid_argument"),
        ));
    };
    let names = first.1.names.clone();
    let named = Admg { admg: first.1.admg.clone(), names: names.clone() };
    let mut expected = names.clone();
    expected.sort();
    let mut built = Vec::with_capacity(scenarios.len());
    for (name, graph, selections, weight, own) in &scenarios {
        let mut sorted_names = graph.names.clone();
        sorted_names.sort();
        if sorted_names != expected {
            return Err(schema_mismatch(format!(
                "scenario {name} declares different variable names"
            )));
        }
        let selections =
            selections.iter().map(|s| resolve(&names, s)).collect::<PyResult<Vec<_>>>()?;
        let diagram = SelectionDiagram::try_new(
            graph.aligned_to_names(&names)?,
            Arc::<[VariableId]>::from(selections),
        )
        .map_err(error)?;
        let coordinates = parse_coordinates(own.as_ref().unwrap_or(&coordinates), &names, name)?;
        built.push(TransportScenario {
            name: Arc::from(name.as_str()),
            diagram,
            weight: *weight,
            coordinates,
        });
    }
    let set = TransportScenarioSet::try_new(built)
        .map_err(|refusal| scenario_error(scenario_refusal(refusal)))?;
    let resolved = |variables: &[String]| -> PyResult<Arc<[VariableId]>> {
        variables.iter().map(|n| resolve(&names, n)).collect::<PyResult<Vec<_>>>().map(Arc::from)
    };
    let query = ClassicalTransportQuery {
        outcomes: resolved(&outcomes)?,
        treatments: resolved(&treatments)?,
        source: source.into(),
        target: target.into(),
    };
    let conditioned =
        (!conditioned_on.is_empty()).then(|| resolved(&conditioned_on)).transpose()?;
    let catalog = parse_catalog(catalog, &named)?;
    let provider = parse_provider(laws, &catalog, &named, max_support_rows)?;
    if conditioned.is_some() && matches!(provider, Provider::Empirical(_)) {
        return Err(crate::refusal(
            reason_code!("cell_not_licensed"),
            "admg_transport.interval_withheld: counted laws are not licensed for a conditional \
             scenario question; the route publishes exact-law points only",
        ));
    }
    let request = assignment_from_pairs(&names, assignments)?;
    let stage_catalog = catalog.clone();
    let inner = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        let budget = SearchLimits { operations: max_steps, depth: max_depth };
        let evaluation =
            ExactEvaluationLimits { operations: max_operations, depth: max_evaluation_depth };
        match (provider, conditioned) {
            (Provider::Exact(data), Some(conditioned_on)) => {
                antecedent::StudyBuilder::conditional_transport_scenarios(
                    &set,
                    ConditionalTransportQuery { base: query, conditioned_on },
                    catalog,
                    budget,
                    data,
                    request,
                    evaluation,
                    &ctx,
                )
            }
            (Provider::Exact(data), None) => antecedent::StudyBuilder::transport_scenarios(
                &set, query, catalog, budget, data, request, evaluation, &ctx,
            ),
            (Provider::Empirical(input), _) => {
                antecedent::StudyBuilder::transport_scenarios_empirical(
                    &set,
                    query,
                    catalog,
                    budget,
                    input,
                    max_support_rows,
                    request,
                    evaluation,
                    &ctx,
                )
            }
        }
        .map_err(scenario_error)
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
#[pyo3(signature=(artifact, *, max_steps=100_000, max_depth=256, max_operations=10_000_000, max_evaluation_depth=256, max_support_rows=1_000_000, max_laws=256, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn consume_transport_scenarios_artifact(
    py: Python<'_>,
    artifact: &[u8],
    max_steps: usize,
    max_depth: usize,
    max_operations: usize,
    max_evaluation_depth: usize,
    max_support_rows: usize,
    max_laws: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let (names, bytes) = unframe_named_artifact(SCENARIO_PREFIX, artifact, "transport scenario")?;
    let limits = TransportScenarioConsumeLimits {
        scenario_budget: SearchLimits { operations: max_steps, depth: max_depth },
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
                .map_err(scenario_error)?;
        for scenario in &wire.scenarios {
            let graph = antecedent_io::admg_from_wire(&scenario.graph).map_err(error)?;
            crate::transport_exact_api::validate_artifact_names(&names, &graph)?;
        }
        // The frame's names must be exactly the ones the identity binds: the
        // report is read with them.
        let bound = wire.coordinates.len() == names.len()
            && wire.coordinates.iter().all(|c| {
                usize::try_from(c.variable).ok().and_then(|i| names.get(i)) == Some(&c.name)
            });
        if !bound {
            return Err(serialization_error(
                "transport scenario artifact names disagree with its coordinate schema",
            ));
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
