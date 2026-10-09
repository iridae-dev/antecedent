//! Python bindings for exact binary observation recovery (2.2B X10):
//! graph-licensed recovery, not MAR/IPCW.
use crate::graphs::Admg;
use crate::transport_common::{
    execution_context, frame_named_artifact, resolve, unframe_named_artifact,
};
use crate::transport_interference_api::parse_catalog;
use crate::transport_z_api::to_py_json;
use antecedent_core::{
    EvidenceCatalog, ExecutionContext, RegimeId, SearchLimits, Value, VariableId,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactDistribution, ExactEvaluationLimits,
    LawTolerance,
};
use antecedent_graph::{DenseNodeId, NodeRef};
use antecedent_identify::recovery::MISSING_LEVEL;
use antecedent_identify::{
    ObservationRecoveryQuery, PartiallyObserved, RECOVERY_DEFAULT_LIMITS, RecoveredEffectQuery,
    RecoveryDecision, RecoveryDerivation, RecoveryDetail, RecoveryError, RecoveryLimits,
    RecoveryWitness, decide_observation_recovery, verify_recovery_witness,
};
use antecedent_io::IoError;
use antecedent_io::recovery_artifact::{RecoveryArtifactWire, RecoveryConsumeLimits};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Magic prefix of a portable recovery artifact: the io crate's versioned CBOR
/// wire, framed with the m-graph's variable names.
const RECOVERY_PREFIX: &[u8] = b"ANTECEDENT-OBSERVATION-RECOVERY\x01";

/// Graph-licensed recovery of the full law, exact laws, point only.
const SCOPE: &str = "graph_licensed_observation_recovery_not_mar_ipcw";

fn recovery_err(error: &RecoveryError) -> PyErr {
    crate::refusal(error.reason_code(), error.to_string())
}

fn io_err(error: &IoError) -> PyErr {
    match error.reason_code() {
        Some(code) => crate::refusal(code, error.to_string()),
        None => crate::transport_common::serialization_error(error),
    }
}

fn name(names: &[String], v: VariableId) -> String {
    names.get(v.as_usize()).cloned().unwrap_or_else(|| format!("v{}", v.raw()))
}

fn raw_names(names: &[String], raw: &[u32]) -> Vec<String> {
    raw.iter().map(|r| name(names, VariableId::from_raw(*r))).collect()
}

fn derivation_json(derivation: &RecoveryDerivation, names: &[String]) -> serde_json::Value {
    let record = derivation.record();
    serde_json::json!({
        "outcome": "recovered",
        "route": "graph-licensed recovery, not MAR/IPCW",
        "rule_version": record.rule_version,
        "population": record.population,
        "premises": record.premises,
        "factors": record.factors.iter().map(|f| serde_json::json!({
            "response": name(names, VariableId::from_raw(f.response)),
            "parents": raw_names(names, &f.parents),
            "conditioning": raw_names(names, &f.conditioning),
        })).collect::<Vec<_>>(),
        "margins": record.margins.iter().map(|m| serde_json::json!({
            "variables": raw_names(names, &m.variables),
            "identity": m.identity,
        })).collect::<Vec<_>>(),
        "effect": record.effect.as_ref().map(|e| serde_json::json!({
            "outcomes": raw_names(names, &e.outcomes),
            "treatments": raw_names(names, &e.treatments),
            "rules": e.rules,
        })),
        "receipt": {
            "operations_limit": record.receipt.operations_limit,
            "depth_limit": record.receipt.depth_limit,
            "memory_bytes": record.receipt.memory_bytes,
            "operations_consumed": record.receipt.operations_consumed,
            "depth_reached": record.receipt.depth_reached,
        },
        "identity": derivation.identity(),
    })
}

fn witness_json(
    witness: &RecoveryWitness,
    graph: &antecedent_graph::Admg,
    query: &ObservationRecoveryQuery,
    names: &[String],
) -> serde_json::Value {
    let model = |mechanisms: &[antecedent_identify::WitnessMechanism]| {
        mechanisms
            .iter()
            .map(|m| {
                serde_json::json!({
                    "node": name(names, VariableId::from_raw(m.node)),
                    "parents": raw_names(names, &m.parents),
                    "p_one_times_60": m.numerators,
                })
            })
            .collect::<Vec<_>>()
    };
    let check = verify_recovery_witness(graph, query, witness).ok();
    serde_json::json!({
        "outcome": "nonrecoverable",
        "reason": RecoveryDetail::NonrecoverableWitness.reason_code(),
        "detail": RecoveryDetail::NonrecoverableWitness.detail(),
        "self_censoring_edge": [name(names, VariableId::from_raw(witness.edge.0)), name(names, VariableId::from_raw(witness.edge.1))],
        "first_model": model(&witness.first),
        "second_model": model(&witness.second),
        "verified": check.as_ref().map(|c| serde_json::json!({
            "observed_cells_equal": c.observed_cells,
            "differing_target_cell": c.differing_cell.iter().map(|(v, l)| (name(names, *v), *l)).collect::<BTreeMap<_, _>>(),
            "target_masses": [c.masses.0.to_string(), c.masses.1.to_string()],
            "denominator": c.denominator.to_string(),
        })),
    })
}

fn level(value: &Value) -> serde_json::Value {
    match value {
        Value::Label(label) => serde_json::json!(label.as_ref()),
        other => serde_json::json!(other.as_f64()),
    }
}

fn distribution_json(point: &ExactDistribution, names: &[String]) -> serde_json::Value {
    serde_json::json!({
        "outcomes": point.outcomes.iter().map(|v| name(names, *v)).collect::<Vec<_>>(),
        "atoms": point.atoms.iter().map(|row| row.iter().map(level).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "probabilities": point.probabilities.as_ref(),
        "means": point.outcomes.iter().map(|v| (name(names, *v), serde_json::json!(point.mean(*v).ok()))).collect::<serde_json::Map<_, _>>(),
    })
}

fn result_json(
    recovered: &antecedent_estimate::RecoveredLaw,
    effects: &[ExactDistribution],
    requests: &[Assignment],
    names: &[String],
) -> serde_json::Value {
    let law = recovered.law();
    serde_json::json!({
        "status": "available",
        "scope": SCOPE,
        "recovered": {
            "variables": law.axes().iter().map(|a| name(names, a.variable)).collect::<Vec<_>>(),
            "levels": law.axes().iter().map(|a| a.values.iter().map(level).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "probabilities": law.probabilities(),
            "origin": "recovered",
            "derivation": recovered.derivation_identity(),
        },
        "effects": requests.iter().zip(effects).map(|(request, point)| {
            let mut entry = distribution_json(point, names);
            entry["assignment"] = request.entries().iter().map(|(v, x)| (name(names, *v), level(x))).collect::<serde_json::Map<_, _>>().into();
            entry
        }).collect::<Vec<_>>(),
        // Exact laws carry no sampling uncertainty: the route is point-only.
        "interval": {"available": false, "status": "point_only"},
    })
}

/// Parse an observed pattern law: axis levels are numbers or the missing level `"?"`.
fn parse_observed(
    law: &Bound<'_, PyAny>,
    names: &[String],
    catalog: &EvidenceCatalog,
) -> PyResult<ExactDiscreteLaw> {
    let population: String = law.getattr("population")?.extract()?;
    let label: String = law.getattr("regime")?.extract()?;
    let regime = regime_of(catalog, &population, &label)?;
    let axes: Vec<(String, Vec<Bound<'_, PyAny>>)> = law.getattr("axes")?.extract()?;
    let axes = axes
        .into_iter()
        .map(|(axis, values)| {
            let values = values
                .into_iter()
                .map(|value| {
                    if let Ok(text) = value.extract::<String>() {
                        if text == MISSING_LEVEL {
                            return Ok(Value::Label(Arc::from(MISSING_LEVEL)));
                        }
                        return Err(crate::value_err(format!(
                            "axis {axis}: the only text level is the missing level \"?\""
                        )));
                    }
                    Ok(Value::f64(value.extract::<f64>()?))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(DiscreteAxis { variable: resolve(names, &axis)?, values: values.into() })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let interventions: Vec<(String, f64)> = law.getattr("interventions")?.extract()?;
    if !interventions.is_empty() {
        return Err(crate::value_err("an observed pattern law has no interventions"));
    }
    let probabilities: Vec<f64> = law.getattr("probabilities")?.extract()?;
    let snapshot: String = law.getattr("snapshot_identity")?.extract()?;
    let absolute: f64 = law.getattr("absolute_tolerance")?.extract()?;
    let relative: f64 = law.getattr("relative_tolerance")?.extract()?;
    let mut exact = ExactDiscreteLaw::try_new(
        population,
        regime,
        Vec::new(),
        axes,
        probabilities,
        snapshot,
        LawTolerance { absolute, relative },
    )
    .map_err(|e| crate::value_err(e.to_string()))?;
    if let Ok(counts) = law.getattr("empirical_counts") {
        if !counts.is_none() {
            let counts: Vec<u64> = counts.extract()?;
            exact =
                exact.with_empirical_counts(counts).map_err(|e| crate::value_err(e.to_string()))?;
        }
    }
    Ok(exact)
}

fn regime_of(catalog: &EvidenceCatalog, population: &str, label: &str) -> PyResult<RegimeId> {
    catalog
        .regimes
        .iter()
        .find(|r| r.label.as_deref() == Some(label) && r.population.as_ref() == population)
        .map(|r| r.id)
        .ok_or_else(|| {
            recovery_err(&RecoveryError::new(
                RecoveryDetail::MissingMargin,
                format!("no regime {label} of population {population} in the catalog"),
            ))
        })
}

fn parse_requests(
    names: &[String],
    requests: Option<Vec<BTreeMap<String, f64>>>,
) -> PyResult<Vec<Assignment>> {
    requests
        .unwrap_or_default()
        .into_iter()
        .map(|pairs| {
            pairs
                .into_iter()
                .map(|(n, x)| Ok((resolve(names, &n)?, Value::f64(x))))
                .collect::<PyResult<Vec<_>>>()
                .map(Assignment::from_pairs)
        })
        .collect()
}

/// The decision of one recovery query: recovered (with its checked formula and
/// downstream effect) or nonrecoverable (with a verified witness).
#[pyclass(skip_from_py_object)]
struct ObservationRecoveryStage {
    decision: RecoveryDecision,
    graph: Admg,
    shared: antecedent_graph::Admg,
    query: ObservationRecoveryQuery,
    catalog: EvidenceCatalog,
    effect: Option<RecoveredEffectQuery>,
    memory_bytes: Option<u64>,
}

#[pymethods]
impl ObservationRecoveryStage {
    /// Internal release-acceptance producer; default wheels omit this method.
    #[cfg(feature = "calibration-internal")]
    #[pyo3(signature=(population, observed_regime, partial, fully, rows, snapshot, replicates, seed))]
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn sampled_candidate(
        &self,
        py: Python<'_>,
        population: String,
        observed_regime: String,
        partial: Vec<(String, String, String)>,
        fully: Vec<String>,
        rows: Vec<(u64, u8, u8, u8)>,
        snapshot: String,
        replicates: usize,
        seed: u64,
    ) -> PyResult<(String, Vec<u8>)> {
        use antecedent_estimate::{
            ObservationPattern, ObservationRow, SampledObservationInput, SampledRecoveryConfig,
            estimate_sampled_recovery,
        };
        use antecedent_io::sampled_recovery_artifact::{
            SampledRecoveryArtifactInput, SampledRecoveryArtifactWire,
        };
        let declared = ObservationRecoveryQuery {
            population: population.into(),
            observed_regime: regime_of(&self.catalog, &self.query.population, &observed_regime)?,
            partially_observed: partial
                .iter()
                .map(|(x, r, p)| {
                    Ok(PartiallyObserved {
                        variable: resolve(&self.graph.names, x)?,
                        response: resolve(&self.graph.names, r)?,
                        proxy: resolve(&self.graph.names, p)?,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?
                .into(),
            fully_observed: fully
                .iter()
                .map(|x| resolve(&self.graph.names, x))
                .collect::<PyResult<Vec<_>>>()?
                .into(),
        };
        let canonical = declared.canonical();
        if canonical != self.query.canonical() {
            return Err(crate::value_err(
                "sampled_recovery.invalid_input: query differs from retained recovery stage",
            ));
        }
        let (k, m) = (partial.len(), fully.len());
        if k + m > 6 || rows.len() > 100_000 {
            return Err(crate::value_err(
                "sampled_recovery.bounds_exceeded: candidate accepts at most six binary variables and 100000 rows",
            ));
        }
        let mask = |n: usize| (1u8 << n) - 1;
        if rows.iter().any(|(_, r, p, f)| r & !mask(k) != 0 || p & !r != 0 || f & !mask(m) != 0) {
            return Err(crate::value_err(
                "sampled_recovery.invalid_input: bits outside declared observation roles",
            ));
        }
        let RecoveryDecision::Recovered(derivation) = &self.decision else {
            return Err(crate::value_err(
                "sampled_recovery.unrecoverable_pattern: no recovered derivation",
            ));
        };
        let effect = self.effect.clone().ok_or_else(|| {
            crate::value_err(
                "sampled_recovery.invalid_input: stage must retain a downstream effect",
            )
        })?;
        let partial_order: Vec<_> = canonical
            .partially_observed
            .iter()
            .map(|p| {
                declared
                    .partially_observed
                    .iter()
                    .position(|s| s == p)
                    .expect("canonicalization preserves roles")
            })
            .collect();
        let full_order: Vec<_> = canonical
            .fully_observed
            .iter()
            .map(|p| {
                declared
                    .fully_observed
                    .iter()
                    .position(|s| s == p)
                    .expect("canonicalization preserves roles")
            })
            .collect();
        let remap = |bits: u8, order: &[usize]| {
            order.iter().enumerate().fold(0u8, |v, (dst, src)| v | (((bits >> src) & 1) << dst))
        };
        let input = SampledObservationInput {
            snapshot_id: snapshot,
            rows: rows
                .into_iter()
                .map(|(id, r, p, f)| ObservationRow {
                    id,
                    pattern: ObservationPattern {
                        responses: remap(r, &partial_order),
                        proxies: remap(p, &partial_order),
                        fully: remap(f, &full_order),
                    },
                })
                .collect(),
        };
        let derivation = derivation.clone();
        let graph = self.shared.clone();
        let catalog = self.catalog.clone();
        let names = self.graph.names.clone();
        let ctx = execution_context(seed, self.memory_bytes, None);
        crate::detach_catch(py, move || {
            let result = estimate_sampled_recovery(
                &derivation,
                &input,
                &SampledRecoveryConfig::new(replicates, seed),
                &ctx,
            )
            .map_err(|e| sampled_candidate_error(e.reason_code(), &e.to_string()))?;
            let wire = SampledRecoveryArtifactWire::checked(&SampledRecoveryArtifactInput {
                graph: &graph,
                effect: &effect,
                derivation: &derivation,
                catalog: &catalog,
                input: &input,
                result: &result,
                variable_names: &names,
            })
            .map_err(|e| sampled_candidate_io_error(&e))?;
            let bytes = wire.export().map_err(|e| sampled_candidate_io_error(&e))?;
            let summary = sampled_candidate_json(&wire);
            Ok((summary, bytes))
        })
    }
    /// `recovered` or `nonrecoverable`.
    #[getter]
    #[doc(hidden)]
    fn outcome(&self) -> &'static str {
        match &self.decision {
            RecoveryDecision::Recovered(_) => "recovered",
            RecoveryDecision::NonRecoverable(_) => "nonrecoverable",
        }
    }

    /// The decision in variable names: the checked formula's propensities and
    /// margins, or the verified witness.
    #[doc(hidden)]
    fn decision(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let names = &self.graph.names;
        let payload = match &self.decision {
            RecoveryDecision::Recovered(derivation) => derivation_json(derivation, names),
            RecoveryDecision::NonRecoverable(witness) => {
                witness_json(witness, &self.shared, &self.query, names)
            }
        };
        to_py_json(py, &payload)
    }

    /// Prepare exact evaluation of the recovered law (and the effect at every
    /// request) from the named observed pattern law. Never decides again.
    #[pyo3(signature=(law, requests=None, *, max_operations=10_000_000, max_depth=256, seed=0, cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_exact(
        &self,
        py: Python<'_>,
        law: &Bound<'_, PyAny>,
        requests: Option<Vec<BTreeMap<String, f64>>>,
        max_operations: usize,
        max_depth: usize,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedObservationRecoveryStage> {
        if let RecoveryDecision::NonRecoverable(witness) = &self.decision {
            return Err(recovery_err(&RecoveryError::new(
                RecoveryDetail::NonrecoverableWitness,
                format!(
                    "self-censoring edge {} -> {}: a verified witness shows the target is not recoverable for every model Markov to the m-graph; see decision()",
                    name(&self.graph.names, VariableId::from_raw(witness.edge.0)),
                    name(&self.graph.names, VariableId::from_raw(witness.edge.1)),
                ),
            )));
        }
        let RecoveryDecision::Recovered(derivation) = &self.decision else {
            unreachable!("a nonrecoverable decision refused above");
        };
        let observed = parse_observed(law, &self.graph.names, &self.catalog)?;
        let requests = parse_requests(&self.graph.names, requests)?;
        // The retained derivation is prepared as decided: never decided again.
        let (shared, catalog, effect, derivation, memory_bytes) = (
            self.shared.clone(),
            self.catalog.clone(),
            self.effect.clone(),
            (**derivation).clone(),
            self.memory_bytes,
        );
        let inner = crate::detach_catch(py, move || {
            let ctx = execution_context(seed, memory_bytes, cancel);
            antecedent::PreparedObservationRecovery::from_derivation(
                shared,
                catalog,
                effect,
                derivation,
                observed,
                requests,
                ExactEvaluationLimits { operations: max_operations, depth: max_depth },
                &ctx,
            )
            .map_err(|e| io_err(&e))
        })?;
        Ok(PreparedObservationRecoveryStage {
            inner,
            graph: self.graph.clone(),
            catalog: self.catalog.clone(),
            last: None,
            memory_bytes: self.memory_bytes,
            seed,
        })
    }

    /// Counted laws are not licensed on this route (a sampled provider is 2.3A):
    /// preparation refuses.
    #[pyo3(signature=(law, requests=None, *, max_operations=10_000_000, max_depth=256, seed=0, cancel=None))]
    #[allow(clippy::too_many_arguments, clippy::unused_self, unused_variables)]
    fn prepare_empirical(
        &self,
        law: &Bound<'_, PyAny>,
        requests: Option<Vec<BTreeMap<String, f64>>>,
        max_operations: usize,
        max_depth: usize,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedObservationRecoveryStage> {
        Err(recovery_err(&RecoveryError::new(
            RecoveryDetail::EmpiricalNotLicensed,
            "counted laws are not licensed on the recovery route: it publishes exact-law points only",
        )))
    }
}

/// A prepared recovery; estimation evaluates the frozen derivation.
#[pyclass(skip_from_py_object)]
struct PreparedObservationRecoveryStage {
    inner: antecedent::PreparedObservationRecovery,
    graph: Admg,
    catalog: EvidenceCatalog,
    last: Option<antecedent::ObservationRecoveryResult>,
    memory_bytes: Option<u64>,
    seed: u64,
}

#[pymethods]
impl PreparedObservationRecoveryStage {
    /// Evaluate the frozen derivation: the recovered law and every effect request.
    /// Exact laws are point-only: `interval["available"]` is always false.
    #[pyo3(signature=(*, memory_bytes=None, cancel=None))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = execution_context(self.seed, memory_bytes.or(self.memory_bytes), cancel);
        let inner = self.inner.clone();
        let result = crate::detach_catch(py, move || inner.estimate(&ctx).map_err(|e| io_err(&e)))?;
        let payload = result_json(
            result.recovered(),
            result.effects(),
            self.inner.requests(),
            &self.graph.names,
        );
        self.last = Some(result);
        Ok(payload.to_string())
    }

    /// Replace the observed pattern law under the snapshot the frozen catalog
    /// binds; the derivation is unchanged and the last claim is cleared.
    #[pyo3(signature=(law, *, memory_bytes=None, cancel=None))]
    fn refresh(
        &mut self,
        py: Python<'_>,
        law: &Bound<'_, PyAny>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let observed = parse_observed(law, &self.graph.names, &self.catalog)?;
        let ctx = execution_context(self.seed, memory_bytes.or(self.memory_bytes), cancel);
        let inner = self.inner.clone();
        self.inner =
            crate::detach_catch(py, move || inner.refresh(observed, &ctx).map_err(|e| io_err(&e)))?;
        self.last = None;
        Ok(())
    }

    /// The last execution as a framed, independently consumable artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let result = self.last.as_ref().ok_or_else(|| {
            crate::refusal(
                antecedent_core::reason_code!("not_executed"),
                "estimate before exporting an observation-recovery artifact",
            )
        })?;
        let raw = result.export_named(&self.inner, &self.graph.names).map_err(|e| io_err(&e))?;
        let framed = frame_named_artifact(RECOVERY_PREFIX, &self.graph.names, raw)?;
        Ok(pyo3::types::PyBytes::new(py, &framed).unbind())
    }

    /// The retained derivation: the checked formula and downstream effect.
    #[doc(hidden)]
    fn plan(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut payload = derivation_json(self.inner.derivation(), &self.graph.names);
        payload["requests"] = self.inner.requests().len().into();
        to_py_json(py, &payload)
    }
}

/// The downstream causal graph over `X ∪ O`, in m-graph variable ids: the given
/// one mapped by name, or the m-graph's causal restriction.
fn effect_graph_for(
    graph: &Admg,
    shared: &antecedent_graph::Admg,
    query: &ObservationRecoveryQuery,
    given: Option<PyRef<'_, Admg>>,
) -> PyResult<antecedent_graph::Admg> {
    let mut out = antecedent_graph::Admg::empty();
    let py_err = |e: antecedent_graph::GraphError| crate::value_err(e.to_string());
    if let Some(effect) = given {
        let mut dense = Vec::new();
        for n in &effect.names {
            let v = resolve(&graph.names, n)?;
            dense.push(out.add_node(NodeRef::Static(v)).map_err(py_err)?);
        }
        for i in 0..effect.admg.node_count() {
            let from = DenseNodeId::from_raw(
                u32::try_from(i).map_err(|_| crate::value_err("graph too large"))?,
            );
            for to in effect.admg.children(from) {
                out.insert_directed(dense[i], dense[to.as_usize()]).map_err(py_err)?;
            }
            for to in effect.admg.bidirected_neighbors(from) {
                if from < *to {
                    out.insert_bidirected(dense[i], dense[to.as_usize()]).map_err(py_err)?;
                }
            }
        }
    } else {
        let substantive = query.substantive();
        let mut dense = BTreeMap::new();
        for v in &substantive {
            dense.insert(*v, out.add_node(NodeRef::Static(*v)).map_err(py_err)?);
        }
        for v in &substantive {
            for child in shared.children(DenseNodeId::from_raw(v.raw())) {
                let c = VariableId::from_raw(child.raw());
                if let Some(to) = dense.get(&c) {
                    out.insert_directed(dense[v], *to).map_err(py_err)?;
                }
            }
        }
    }
    Ok(out)
}

/// Decide an observation-recovery query once.
///
/// `partially_observed` is `(variable, response, proxy)` per partially observed
/// variable; `observed_regime` is the catalog label of the observed pattern law.
/// An effect (`effect_outcomes`, `effect_treatments`) is identified from the
/// recovered law by ordinary target ID on `effect_graph` (default: the m-graph's
/// causal restriction), under the same budget.
#[pyfunction]
#[doc(hidden)]
#[pyo3(signature=(graph, population, observed_regime, partially_observed, fully_observed, catalog, *, effect_outcomes=None, effect_treatments=None, effect_graph=None, max_operations=50_000, max_depth=32, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn identify_observation_recovery_stage(
    py: Python<'_>,
    graph: PyRef<'_, Admg>,
    population: String,
    observed_regime: String,
    partially_observed: Vec<(String, String, String)>,
    fully_observed: Vec<String>,
    catalog: &Bound<'_, PyAny>,
    effect_outcomes: Option<Vec<String>>,
    effect_treatments: Option<Vec<String>>,
    effect_graph: Option<PyRef<'_, Admg>>,
    max_operations: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ObservationRecoveryStage> {
    let named = Admg { admg: graph.admg.clone(), names: graph.names.clone() };
    let catalog = parse_catalog(catalog, &named)?;
    let shared = graph.aligned_to_names(&graph.names)?;
    let names = &graph.names;
    let query = ObservationRecoveryQuery {
        population: Arc::from(population.as_str()),
        observed_regime: regime_of(&catalog, &population, &observed_regime)?,
        partially_observed: partially_observed
            .iter()
            .map(|(x, r, p)| {
                Ok(PartiallyObserved {
                    variable: resolve(names, x)?,
                    response: resolve(names, r)?,
                    proxy: resolve(names, p)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?
            .into(),
        fully_observed: fully_observed
            .iter()
            .map(|o| resolve(names, o))
            .collect::<PyResult<Vec<_>>>()?
            .into(),
    };
    let effect = match (effect_outcomes, effect_treatments) {
        (None, None) => None,
        (Some(outcomes), Some(treatments)) => Some(RecoveredEffectQuery {
            graph: effect_graph_for(&named, &shared, &query, effect_graph)?,
            outcomes: outcomes
                .iter()
                .map(|o| resolve(names, o))
                .collect::<PyResult<Vec<_>>>()?
                .into(),
            treatments: treatments
                .iter()
                .map(|t| resolve(names, t))
                .collect::<PyResult<Vec<_>>>()?
                .into(),
        }),
        _ => {
            return Err(crate::value_err(
                "give both effect_outcomes and effect_treatments, or neither",
            ));
        }
    };
    let limits = RecoveryLimits {
        search: SearchLimits { operations: max_operations, depth: max_depth },
        memory_bytes: RECOVERY_DEFAULT_LIMITS.memory_bytes,
    };
    let (decision_graph, decision_query, decision_catalog, decision_effect) =
        (shared.clone(), query.clone(), catalog.clone(), effect.clone());
    let decision = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        decide_observation_recovery(
            &decision_graph,
            &decision_query,
            &decision_catalog,
            decision_effect.as_ref(),
            limits,
            &ctx,
        )
        .map_err(|e| recovery_err(&e))
    })?;
    Ok(ObservationRecoveryStage {
        decision,
        graph: named,
        shared,
        query,
        catalog,
        effect,
        memory_bytes,
    })
}

/// Independently recheck a framed recovery artifact under the consumer's maxima:
/// re-decide, re-check the formula and recompute every point bit for bit.
#[pyfunction]
#[pyo3(signature=(artifact, *, max_search_operations=50_000, max_search_depth=32, max_operations=10_000_000, max_depth=256, max_requests=64, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn consume_observation_recovery_artifact(
    py: Python<'_>,
    artifact: &[u8],
    max_search_operations: usize,
    max_search_depth: usize,
    max_operations: usize,
    max_depth: usize,
    max_requests: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let (names, bytes) = unframe_named_artifact(RECOVERY_PREFIX, artifact, "observation-recovery")?;
    let limits = RecoveryConsumeLimits {
        decision: RecoveryLimits {
            search: SearchLimits { operations: max_search_operations, depth: max_search_depth },
            memory_bytes: RECOVERY_DEFAULT_LIMITS.memory_bytes,
        },
        evaluation: ExactEvaluationLimits { operations: max_operations, depth: max_depth },
        max_requests,
    };
    crate::detach_catch(py, move || {
        let ctx: ExecutionContext = execution_context(0, memory_bytes, cancel);
        let consumed = RecoveryArtifactWire::consume_with_limits(&bytes, limits, &ctx)
            .map_err(|e| io_err(&e))?;
        let graph = antecedent_io::admg_from_wire(&consumed.wire.graph).map_err(|e| io_err(&e))?;
        crate::transport_exact_api::validate_artifact_names(&names, &graph)?;
        // The frame's names must be the mapping the verified identity binds.
        consumed.wire.check_variable_names(&names).map_err(|e| io_err(&IoError::from(e)))?;
        let requests: Vec<Assignment> = consumed
            .wire
            .requests
            .iter()
            .map(|r| {
                Assignment::from_pairs(
                    r.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
                )
            })
            .collect();
        let mut payload = result_json(&consumed.recovered, &consumed.effects, &requests, &names);
        payload["derivation"] = derivation_json(&consumed.derivation, &names);
        payload["premises_digest"] = consumed.wire.premises_digest.clone().into();
        payload["data_digest"] = consumed.wire.data_digest.clone().into();
        Ok(payload.to_string())
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ObservationRecoveryStage>()?;
    module.add_class::<PreparedObservationRecoveryStage>()?;
    module.add_function(wrap_pyfunction!(identify_observation_recovery_stage, module)?)?;
    module.add_function(wrap_pyfunction!(consume_observation_recovery_artifact, module)?)?;
    #[cfg(feature = "calibration-internal")]
    module.add_function(wrap_pyfunction!(consume_sampled_recovery_candidate, module)?)?;
    Ok(())
}

#[cfg(feature = "calibration-internal")]
fn sampled_candidate_error(code: &str, message: &str) -> PyErr {
    if code == antecedent_core::reason_code!("invalid_argument") {
        crate::with_reason_code(
            crate::value_err(message),
            antecedent_core::reason_code!("invalid_argument"),
        )
    } else {
        crate::refusal(code, message)
    }
}

#[cfg(feature = "calibration-internal")]
fn sampled_candidate_io_error(error: &IoError) -> PyErr {
    match error.reason_code() {
        Some(code) => sampled_candidate_error(code, &error.to_string()),
        None => crate::transport_common::serialization_error(error),
    }
}

#[cfg(feature = "calibration-internal")]
fn sampled_candidate_json(
    wire: &antecedent_io::sampled_recovery_artifact::SampledRecoveryArtifactWire,
) -> String {
    serde_json::json!({"result":wire.result,"snapshot":wire.snapshot_id,"calibration":wire.calibration,"premises_digest":wire.premises_digest,"data_digest":wire.data_digest,"query":wire.query,"names":wire.variable_names,"receipt":wire.receipt}).to_string()
}

/// Candidate consumer reruns the original whole-row method under retained identities.
#[cfg(feature = "calibration-internal")]
#[pyfunction]
#[pyo3(signature=(artifact, premises_digest, data_digest, *, max_rows=100_000, max_replicates=2000, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn consume_sampled_recovery_candidate(
    py: Python<'_>,
    artifact: &[u8],
    premises_digest: String,
    data_digest: String,
    max_rows: usize,
    max_replicates: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    use antecedent_io::sampled_recovery_artifact::{
        SampledRecoveryArtifactWire, SampledRecoveryConsumeLimits, SampledRecoveryExpectation,
    };
    let valid_digest = |value: &str| {
        value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    if artifact.len() > 32 * 1024 * 1024
        || !valid_digest(&premises_digest)
        || !valid_digest(&data_digest)
        || !(1..=100_000).contains(&max_rows)
        || !(1..=2000).contains(&max_replicates)
    {
        return Err(crate::value_err(
            "sampled_recovery.invalid_input: bounded bytes and retained expected identities required",
        ));
    }
    let bytes = artifact.to_vec();
    crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        let expected = SampledRecoveryExpectation {
            premises_digest: Some(premises_digest),
            data_digest: Some(data_digest),
        };
        let consumed = SampledRecoveryArtifactWire::consume_expecting(
            &bytes,
            &expected,
            SampledRecoveryConsumeLimits { max_rows, max_replicates, ..Default::default() },
            &ctx,
        )
        .map_err(|e| sampled_candidate_io_error(&e))?;
        Ok(sampled_candidate_json(&consumed.wire))
    })
}
