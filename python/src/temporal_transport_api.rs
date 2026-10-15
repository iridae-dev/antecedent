//! Python bindings for one finite two-step temporal transport sequence.
use crate::transport_common::{
    RegimeCheck, error, execution_context, frame_named_artifact, parse_laws, resolve,
    serialization_error, unframe_named_artifact,
};
use crate::{graphs::Admg, transport_interference_api::parse_catalog};
use antecedent_core::{SearchLimits, Value, VariableDomain, VariableId, reason_code};
use antecedent_estimate::temporal_transport::TemporalSequenceReport;
use antecedent_expr::ExactEvaluationLimits;
use antecedent_graph::SelectionDiagram;
use antecedent_identify::sid::{
    scenarios::ScenarioCoordinate,
    temporal_sequence::{TemporalSequenceSpec, TemporalSlots},
};
use antecedent_io::IoError;
use antecedent_io::temporal_transport_artifact::{
    TemporalSequenceArtifactWire, TemporalTransportConsumeLimits, temporal_refusal,
};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

const TEMPORAL_PREFIX: &[u8] = b"ANTECEDENT-TEMPORAL-TRANSPORT\x01";
const SCOPE: &str = "finite_two_step_temporal_transport_sequence_point_only";

/// One declared coordinate: `(name, domain, cardinality, unit)`.
type CoordinateTuple = (String, String, Option<u32>, Option<String>);

/// A reason-coded refusal: an argument or specification refusal is a value error
/// carrying its code; any other refusal keeps its transport class.
fn temporal_error(e: IoError) -> PyErr {
    match e {
        IoError::Refused { code, message } if code == reason_code!("invalid_argument") => {
            crate::value_err_code(code, message)
        }
        other => error(other),
    }
}

fn parse_coordinates(
    coordinates: &[CoordinateTuple],
    names: &[String],
) -> PyResult<Vec<ScenarioCoordinate>> {
    coordinates
        .iter()
        .map(|(name, domain, cardinality, unit)| {
            let variable = names.iter().position(|n| n == name).ok_or_else(|| {
                crate::with_reason_code(
                    crate::value_err(format!(
                        "temporal_transport.invalid_spec: coordinate {name} is not a graph variable"
                    )),
                    reason_code!("invalid_argument"),
                )
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

fn report_json(report: &TemporalSequenceReport, names: &[String]) -> serde_json::Value {
    let name = |v: &VariableId| names[v.as_usize()].clone();
    let value = |v: &Value| v.as_f64();
    let d = &report.distribution;
    serde_json::json!({
        "status": "available",
        "scope": SCOPE,
        "horizon": report.horizon,
        "sequence": report.sequence.iter().map(value).collect::<Vec<_>>(),
        "inference_claim": report.inference_claim,
        "point": {
            "outcomes": d.outcomes.iter().map(name).collect::<Vec<_>>(),
            "atoms": d.atoms.iter().map(|row| row.iter().map(Value::as_f64).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "probabilities": d.probabilities.as_ref(),
            "means": d.outcomes.iter().map(|o| (name(o), d.mean(*o).ok())).collect::<BTreeMap<_, _>>(),
        },
        "mean": report.mean,
        "support": {
            "initial_coordinates": report.support.initial_coordinates.iter().map(name).collect::<Vec<_>>(),
            "history_coordinates": report.support.history_coordinates.iter().map(name).collect::<Vec<_>>(),
            "target_law_used": report.support.target_law_used,
            "rows": report.support.rows.iter().map(|r| serde_json::json!({
                "step": r.step,
                "history": r.history.iter().map(value).collect::<Vec<_>>(),
                "target_mass": r.target_mass,
                "status": r.status,
            })).collect::<Vec<_>>(),
        },
        "invariances": report.invariances.iter().map(|i| serde_json::json!({
            "slice": i.slice,
            "variable": name(&i.variable),
            "assumption": i.assumption.name(),
            "borrowed_from_source": i.borrowed_from_source,
        })).collect::<Vec<_>>(),
        "time_varying_confounders": report.time_varying_confounders.iter().map(name).collect::<Vec<_>>(),
        "evidence": report.evidence.iter().map(|e| serde_json::json!({
            "slice": e.slice,
            "population": e.population.as_ref(),
            "regime": e.regime.raw(),
            "interventions": e.interventions.iter().map(name).collect::<Vec<_>>(),
            "measured": e.measured.iter().map(name).collect::<Vec<_>>(),
            "cited_by_derivation": e.cited_by_derivation,
        })).collect::<Vec<_>>(),
    })
}

/// Accept an `ExactTransportData` wrapper or a sequence of laws.
fn law_sequence<'py>(laws: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if laws.hasattr("laws")? { laws.getattr("laws") } else { Ok(laws.clone()) }
}

/// A decided, compiled two-step sequence; estimation never re-identifies.
#[pyclass(skip_from_py_object)]
struct PreparedTemporalTransportStage {
    inner: antecedent::PreparedTemporalTransport,
    graph: Admg,
    catalog: antecedent_core::EvidenceCatalog,
    last: Option<TemporalSequenceReport>,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
}

#[pymethods]
impl PreparedTemporalTransportStage {
    /// Evaluate the whole sequence's history-aware functional; exact and point-only.
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

    /// Re-estimate against new laws of the same measurement window; the proof is
    /// kept. Another window or horizon needs a new preparation.
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
        self.inner =
            crate::detach_catch(py, move || inner.refresh(data, &ctx).map_err(temporal_error))?;
        self.last = None;
        Ok(())
    }

    /// The frozen plan: the sequence, horizon, the derivation's rules, the
    /// populations and regimes its leaves cite, and the history lattice sizes.
    fn plan(&self) -> PyResult<String> {
        use antecedent_identify::sid::temporal_sequence::TemporalOutcome;
        let decision = self.inner.prepared().decision();
        let TemporalOutcome::Identified(bound) = &decision.outcome else {
            return Err(error("a prepared temporal sequence is always identified"));
        };
        let mut cited = bound
            .leaf_factors()
            .into_iter()
            .map(|(_, leaf)| {
                (
                    leaf.binding.population.to_string(),
                    leaf.binding.regime.map(antecedent_core::RegimeId::raw),
                )
            })
            .collect::<Vec<_>>();
        cited.sort();
        cited.dedup();
        Ok(serde_json::json!({
            "compiled": true,
            "horizon": decision.spec.horizon(),
            "sequence": decision.sequence.iter().map(Value::as_f64).collect::<Vec<_>>(),
            "rules": bound.derivation().rules(),
            "cited": cited,
            "histories": {
                "initial": decision.histories.initial.len(),
                "complete": decision.histories.complete.len(),
            },
        })
        .to_string())
    }

    /// An interval is not licensed; always refuses.
    fn interval(&self) -> PyResult<()> {
        self.inner.interval().map_err(error)
    }

    /// The last report as a framed, independently consumable artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let report = self.last.as_ref().ok_or_else(|| {
            crate::refusal(
                antecedent_core::reason_code!("not_executed"),
                "transport.no_execution_claim: estimate before exporting a temporal sequence artifact",
            )
        })?;
        let raw = self.inner.export(report).map_err(error)?;
        let framed = frame_named_artifact(TEMPORAL_PREFIX, &self.graph.names, raw)?;
        Ok(pyo3::types::PyBytes::new(py, &framed).unbind())
    }
}

/// Decide the whole two-step sequence once and compile it against exact laws.
///
/// `slots` is `(baseline, covariates, actions, outcome)`: baseline names, the two
/// steps' covariate names, the two action names and the outcome name.
#[pyfunction]
#[pyo3(signature=(graph, selections, slots, coordinates, horizon, sequence, source, target, catalog, laws, *, max_steps=100_000, max_depth=256, max_operations=10_000_000, max_evaluation_depth=256, max_support_rows=1_000_000, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prepare_temporal_transport_stage(
    py: Python<'_>,
    graph: PyRef<'_, Admg>,
    selections: Vec<String>,
    slots: (Vec<String>, Vec<Vec<String>>, Vec<String>, String),
    coordinates: Vec<CoordinateTuple>,
    horizon: usize,
    sequence: Vec<f64>,
    source: String,
    target: String,
    catalog: &Bound<'_, PyAny>,
    laws: &Bound<'_, PyAny>,
    max_steps: usize,
    max_depth: usize,
    max_operations: usize,
    max_evaluation_depth: usize,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PreparedTemporalTransportStage> {
    let names = graph.names.clone();
    let named = Admg { admg: graph.admg.clone(), names: names.clone() };
    let ids =
        |list: &[String]| list.iter().map(|n| resolve(&names, n)).collect::<PyResult<Vec<_>>>();
    let (baseline, covariates, actions, outcome) = slots;
    let malformed = |message: &str| {
        crate::with_reason_code(
            crate::value_err(format!("temporal_transport.invalid_spec: {message}")),
            reason_code!("invalid_argument"),
        )
    };
    let [first, second] = <[Vec<String>; 2]>::try_from(covariates)
        .map_err(|_| malformed("declare the covariates of exactly two steps"))?;
    let [a1, a2] =
        <[String; 2]>::try_from(actions).map_err(|_| malformed("declare exactly two actions"))?;
    let slots = TemporalSlots {
        baseline: ids(&baseline)?,
        covariates: [ids(&first)?, ids(&second)?],
        actions: [resolve(&names, &a1)?, resolve(&names, &a2)?],
        outcome: resolve(&names, &outcome)?,
    };
    let diagram =
        SelectionDiagram::try_new(named.admg.clone(), Arc::<[VariableId]>::from(ids(&selections)?))
            .map_err(error)?;
    let spec = TemporalSequenceSpec::try_new(
        horizon,
        slots,
        diagram,
        parse_coordinates(&coordinates, &names)?,
    )
    .map_err(|refusal| temporal_error(temporal_refusal(refusal)))?;
    let catalog = parse_catalog(catalog, &named)?;
    let data =
        parse_laws(&law_sequence(laws)?, &catalog, &named, max_support_rows, RegimeCheck::Strict)?;
    let sequence = sequence.into_iter().map(Value::f64).collect::<Vec<_>>();
    let stage_catalog = catalog.clone();
    let inner = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        antecedent::StudyBuilder::temporal_transport_sequence(
            &spec,
            antecedent::SequenceEndpoints { sequence: &sequence, source: &source, target: &target },
            antecedent::EvidenceSearch {
                catalog,
                budget: SearchLimits { operations: max_steps, depth: max_depth },
            },
            data,
            ExactEvaluationLimits { operations: max_operations, depth: max_evaluation_depth },
            &ctx,
        )
        .map_err(temporal_error)
    })?;
    Ok(PreparedTemporalTransportStage {
        inner,
        graph: named,
        catalog: stage_catalog,
        last: None,
        max_support_rows,
        memory_bytes,
    })
}

/// Independently replay a framed temporal sequence artifact under the consumer's limits.
#[pyfunction]
#[pyo3(signature=(artifact, *, max_steps=100_000, max_depth=256, max_operations=10_000_000, max_evaluation_depth=256, max_support_rows=1_000_000, max_laws=256, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn consume_temporal_transport_artifact(
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
    let (names, bytes) = unframe_named_artifact(TEMPORAL_PREFIX, artifact, "temporal transport")?;
    let limits = TemporalTransportConsumeLimits {
        budget: SearchLimits { operations: max_steps, depth: max_depth },
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
            TemporalSequenceArtifactWire::consume_with_limits(&bytes, limits, &ctx)
                .map_err(temporal_error)?;
        let graph = antecedent_io::admg_from_wire(&wire.graph).map_err(error)?;
        crate::transport_exact_api::validate_artifact_names(&names, &graph)?;
        // The frame's names must be exactly the ones the identity binds: the
        // report is read with them.
        let bound = wire.coordinates.len() == names.len()
            && wire.coordinates.iter().all(|c| {
                usize::try_from(c.variable).ok().and_then(|i| names.get(i)) == Some(&c.name)
            });
        if !bound {
            return Err(serialization_error(
                "temporal transport artifact names disagree with its coordinate schema",
            ));
        }
        let mut payload = report_json(&report, &names);
        payload["premises_digest"] = wire.premises_digest.clone().into();
        Ok(payload.to_string())
    })
}

/// Unroll a lagged `TemporalDag` template over the horizon of two (the finite unfolding of the
/// temporal DAG; parents before the first slice are dropped, so the initial state is fixed).
///
/// Template variables are the template's names, numbered by position; they must be exactly the
/// covariates, the action and the outcome. Returns the unrolled named graph and the places of
/// its coordinates in time as JSON; latent confounding is added by naming pairs of unrolled
/// coordinates in `latent_pairs` (a bidirected edge each).
#[pyfunction]
#[pyo3(signature = (template, covariates, action, outcome, baseline, baseline_effects, latent_pairs))]
fn unroll_two_slice_template(
    template: PyRef<'_, crate::graphs::TemporalDag>,
    covariates: Vec<String>,
    action: String,
    outcome: String,
    baseline: Vec<String>,
    baseline_effects: Vec<(String, String)>,
    latent_pairs: Vec<(String, String)>,
) -> PyResult<String> {
    use antecedent_identify::sid::temporal_sequence::{TemplateRoles, unroll_two_slice};

    let invalid = |message: String| {
        crate::value_err_code(
            reason_code!("invalid_argument"),
            format!("temporal_transport.invalid_spec: {message}"),
        )
    };
    let names = &template.names;
    let index_of = |name: &str| -> PyResult<VariableId> {
        names
            .iter()
            .position(|n| n == name)
            .and_then(|i| u32::try_from(i).ok())
            .map(VariableId::from_raw)
            .ok_or_else(|| invalid(format!("`{name}` is not a variable of the template")))
    };
    if names.len() != covariates.len() + 2 {
        return Err(invalid(
            "the template's variables must be exactly the covariates, the action and the outcome"
                .to_owned(),
        ));
    }
    let roles = TemplateRoles {
        covariates: covariates.iter().map(|c| index_of(c)).collect::<PyResult<_>>()?,
        action: index_of(&action)?,
        outcome: index_of(&outcome)?,
    };
    let baseline_index = |name: &str| -> PyResult<usize> {
        baseline
            .iter()
            .position(|b| b == name)
            .ok_or_else(|| invalid(format!("`{name}` is not a declared baseline covariate")))
    };
    let effects = baseline_effects
        .iter()
        .map(|(b, v)| Ok((baseline_index(b)?, index_of(v)?)))
        .collect::<PyResult<Vec<_>>>()?;
    let mut unrolled = unroll_two_slice(&template.dag, &roles, baseline.len(), &effects)
        .map_err(|e| crate::value_err_code(e.code, format!("{}: {}", e.detail, e.message)))?;
    // The name of every unrolled coordinate, by its id: `(period - 1) * n + variable`, then the
    // baseline covariates.
    let n = names.len();
    let mut coordinate_names: Vec<String> = Vec::with_capacity(2 * n + baseline.len());
    for period in 1..=2 {
        coordinate_names.extend(names.iter().map(|name| format!("{name}_{period}")));
    }
    coordinate_names.extend(baseline.iter().cloned());
    let named = |id: antecedent_core::VariableId| coordinate_names[id.as_usize()].clone();
    for (a, b) in &latent_pairs {
        let find = |name: &String| {
            coordinate_names
                .iter()
                .position(|n| n == name)
                .and_then(|i| u32::try_from(i).ok())
                .map(antecedent_graph::DenseNodeId::from_raw)
                .ok_or_else(|| invalid(format!("`{name}` is not an unrolled coordinate")))
        };
        let (a, b) = (find(a)?, find(b)?);
        unrolled.admg.insert_bidirected(a, b).map_err(|e| invalid(e.to_string()))?;
    }
    let mut directed = Vec::new();
    let mut bidirected = Vec::new();
    for node in 0..unrolled.admg.node_count() {
        let from = antecedent_graph::DenseNodeId::try_from_usize(node).map_err(crate::py_err)?;
        for &to in unrolled.admg.children(from) {
            directed.push([
                coordinate_names[from.as_usize()].clone(),
                coordinate_names[to.as_usize()].clone(),
            ]);
        }
        for &to in unrolled.admg.bidirected_neighbors(from) {
            if from < to {
                bidirected.push([
                    coordinate_names[from.as_usize()].clone(),
                    coordinate_names[to.as_usize()].clone(),
                ]);
            }
        }
    }
    let slots = &unrolled.slots;
    Ok(serde_json::json!({
        "names": coordinate_names,
        "directed": directed,
        "bidirected": bidirected,
        "baseline": slots.baseline.iter().copied().map(&named).collect::<Vec<_>>(),
        "covariates": slots.covariates.iter()
            .map(|step| step.iter().copied().map(&named).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        "actions": slots.actions.iter().copied().map(&named).collect::<Vec<_>>(),
        "outcome": named(slots.outcome),
    })
    .to_string())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(unroll_two_slice_template, module)?)?;
    module.add_class::<PreparedTemporalTransportStage>()?;
    module.add_function(wrap_pyfunction!(prepare_temporal_transport_stage, module)?)?;
    module.add_function(wrap_pyfunction!(consume_temporal_transport_artifact, module)?)?;
    Ok(())
}
