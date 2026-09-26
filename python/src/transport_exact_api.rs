//! Native authority for classical identification and exact-law stage evaluation.
use crate::graphs::Admg;
use crate::transport_interference_api::parse_catalog;
use antecedent_core::{ExecutionContext, RegimeId, Value, VariableId};
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    ClassicalTransportQuery, ClassicalTransportResult, SidLimits, identify_classical_transport,
};
use pyo3::prelude::*;
use std::sync::Arc;

type ExactStagePayload = (Vec<Vec<f64>>, Vec<f64>, String, Vec<String>);

#[pyclass(skip_from_py_object)]
pub(crate) struct ClassicalTransportStage {
    result: ClassicalTransportResult,
    graph: Admg,
    diagram: SelectionDiagram,
    certificate: antecedent_io::transport_certificate::TransportCertificateWire,
}
impl ClassicalTransportStage {
    pub(crate) fn identified(&self) -> PyResult<antecedent_identify::ClassicalTransportDerivation> {
        match &self.result {
            ClassicalTransportResult::Identified(proof) => Ok(proof.as_ref().clone()),
            _ => Err(error("statistical evaluation requires a native checked derivation")),
        }
    }
    pub(crate) fn graph(&self) -> &Admg {
        &self.graph
    }
    pub(crate) fn named_graph(&self) -> Admg {
        Admg { admg: self.graph.admg.clone(), names: self.graph.names.clone() }
    }
    pub(crate) fn diagram(&self) -> SelectionDiagram {
        self.diagram.clone()
    }
}
use crate::transport_common::{RegimeCheck, error, parse_laws, resolve};
#[pymethods]
impl ClassicalTransportStage {
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let artifact = self.certificate.export().map_err(error)?;
        let mut bytes = b"ANTECEDENT-TRANSPORT-CERTIFICATE\x01".to_vec();
        bytes.extend(antecedent_io::to_cbor(&(self.graph.names.clone(), artifact)).map_err(error)?);
        Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
    }
    fn certificate_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.certificate).map_err(error)
    }
    #[pyo3(signature=(catalog,*,max_steps=100_000,max_depth=256))]
    fn proof_graph_json(
        &self,
        catalog: &Bound<'_, PyAny>,
        max_steps: usize,
        max_depth: usize,
    ) -> PyResult<String> {
        let antecedent_io::transport_certificate::CertificateOutcome::Identified(proof) =
            &self.certificate.outcome
        else {
            return Err(error("proof graph requires an identified theorem result"));
        };
        let catalog = parse_catalog(catalog, &self.graph)?;
        let query = ClassicalTransportQuery {
            outcomes: self.certificate.outcomes.iter().copied().map(VariableId::from_raw).collect(),
            treatments: self
                .certificate
                .treatments
                .iter()
                .copied()
                .map(VariableId::from_raw)
                .collect(),
            source: Arc::from(self.certificate.source.as_str()),
            target: Arc::from(self.certificate.target.as_str()),
        };
        let view = proof
            .inspect(
                &self.diagram,
                &query,
                &catalog,
                SidLimits { steps: max_steps, depth: max_depth },
                &ExecutionContext::production_default(0),
            )
            .map_err(error)?;
        let steps = view
            .steps
            .iter()
            .map(|step| {
                serde_json::json!({
                    "index": step.index,
                    "rule": step.rule,
                    "children": step.children,
                    "factor_nodes": step.factor_nodes,
                })
            })
            .collect::<Vec<_>>();
        let factors = view.factors.iter().map(|factor| serde_json::json!({
            "expression_node": factor.expression_node,
            "population": factor.population,
            "variables": factor.variables.iter().map(|v| &self.graph.names[v.raw() as usize]).collect::<Vec<_>>(),
            "conditioned_on": factor.conditioned_on.iter().map(|v| &self.graph.names[v.raw() as usize]).collect::<Vec<_>>(),
            "interventions": factor.interventions.iter().map(|v| &self.graph.names[v.raw() as usize]).collect::<Vec<_>>(),
            "supplied_by": factor.supplied_by.map(RegimeId::raw),
            "snapshot_identity": factor.snapshot_identity,
            "binding_failure": factor.binding_failure,
        })).collect::<Vec<_>>();
        Ok(serde_json::json!({
            "root_expression_node":view.root_expression_node,
            "steps":steps,
            "factors":factors,
        })
        .to_string())
    }
    #[getter]
    fn outcome(&self) -> &'static str {
        match &self.result {
            ClassicalTransportResult::Identified(_) => "identified",
            ClassicalTransportResult::ProvenNonTransportable(_) => "proven_non_transportable",
            ClassicalTransportResult::NotCertified => {
                antecedent_core::TransportOutcomeKind::NotCertified.as_str()
            }
        }
    }
    #[getter]
    fn outcomes(&self) -> Vec<String> {
        self.certificate.outcomes.iter().map(|v| self.graph.names[*v as usize].clone()).collect()
    }
    #[getter]
    fn formula(&self) -> Option<String> {
        match &self.result {
            ClassicalTransportResult::Identified(proof) => Some(proof.arena().pretty(proof.root())),
            _ => None,
        }
    }
    #[getter]
    fn rules(&self) -> Vec<String> {
        match &self.result {
            ClassicalTransportResult::Identified(proof) => {
                proof.rules().into_iter().map(str::to_owned).collect()
            }
            _ => Vec::new(),
        }
    }
    #[pyo3(signature=(catalog,*,max_steps=100_000,max_depth=256,memory_bytes=None,cancel=None))]
    fn catalog_search(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        max_steps: usize,
        max_depth: usize,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ClassicalTransportResult::Identified(proof) = &self.result else {
            return Err(error("catalog search requires an identified theorem query"));
        };
        let proof = proof.clone();
        let diagram = self.diagram.clone();
        let catalog = parse_catalog(catalog, &self.graph)?;
        crate::detach_catch(py, move || {
            let mut ctx = ExecutionContext::production_default(0);
            ctx.memory.hard_limit_bytes = memory_bytes;
            crate::apply_cancel(&mut ctx, cancel);
            let result = proof
                .search_catalog(
                    &diagram,
                    &catalog,
                    SidLimits { steps: max_steps, depth: max_depth },
                    &ctx,
                )
                .map_err(error)?;
            let future: Vec<_> = catalog
                .regimes
                .iter()
                .filter(|r| !r.evidence_kind.can_satisfy_factor())
                .map(|r| r.label.as_deref().unwrap_or("unlabelled"))
                .collect();
            let value = match result {
                antecedent_identify::CatalogTransportResult::Identified(bound) => {
                    serde_json::json!({
                    "outcome":"identified", "searched":bound.searched_alternatives().iter().map(AsRef::as_ref).collect::<Vec<_>>(),
                    "missing_factors":[],"future_experiments":future,"exhausted":false,"finite_catalog_complete":false})
                }
                antecedent_identify::CatalogTransportResult::MissingEvidence {
                    searched,
                    obligations,
                } => serde_json::json!({
                    "outcome":"missing_evidence", "searched":searched.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
                    "missing_factors":obligations.iter().map(AsRef::as_ref).collect::<Vec<_>>(),"future_experiments":future,
                    "exhausted":!obligations.iter().any(|note|note.contains("capped")),"finite_catalog_complete":false}),
                antecedent_identify::CatalogTransportResult::NotCertified {
                    searched,
                    obligations,
                } => serde_json::json!({
                    "outcome":antecedent_core::TransportOutcomeKind::NotCertified.as_str(), "searched":searched.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
                    "missing_factors":obligations.iter().map(AsRef::as_ref).collect::<Vec<_>>(),"future_experiments":future,
                    "exhausted":!obligations.iter().any(|note|note.contains("capped")),"finite_catalog_complete":false}),
            };
            Ok(value.to_string())
        })
    }
    #[pyo3(signature = (catalog, laws, assignments, *, max_operations=10_000_000, max_depth=256, max_support_rows=1_000_000, memory_bytes=None))]
    fn evaluate_exact(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        laws: &Bound<'_, PyAny>,
        assignments: std::collections::BTreeMap<String, f64>,
        max_operations: usize,
        max_depth: usize,
        max_support_rows: usize,
        memory_bytes: Option<u64>,
    ) -> PyResult<ExactStagePayload> {
        let mut prepared = self.prepare_exact(
            py,
            catalog,
            laws,
            assignments,
            max_operations,
            max_depth,
            max_support_rows,
            memory_bytes,
            None,
        )?;
        prepared.estimate(py, None, None)
    }
    #[pyo3(signature = (catalog, laws, assignments, *, max_operations=10_000_000, max_depth=256, max_support_rows=1_000_000, memory_bytes=None, cancel=None))]
    fn prepare_exact(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        laws: &Bound<'_, PyAny>,
        assignments: std::collections::BTreeMap<String, f64>,
        max_operations: usize,
        max_depth: usize,
        max_support_rows: usize,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedExactStage> {
        let ClassicalTransportResult::Identified(proof) = &self.result else {
            return Err(error("exact evaluation requires a native checked derivation"));
        };
        let proof = proof.clone();
        let catalog = parse_catalog(catalog, &self.graph)?;
        let data = parse_exact_data(laws, &catalog, &self.graph, max_support_rows)?;
        let request = Assignment::from_pairs(
            assignments
                .into_iter()
                .map(|(name, value)| Ok((resolve(&self.graph.names, &name)?, Value::f64(value))))
                .collect::<PyResult<Vec<_>>>()?,
        );
        let diagram = self.diagram.clone();
        let graph = Admg { admg: self.graph.admg.clone(), names: self.graph.names.clone() };
        crate::detach_catch(py, move || {
            let mut ctx = ExecutionContext::production_default(0);
            ctx.memory.hard_limit_bytes = memory_bytes;
            crate::apply_cancel(&mut ctx, cancel);
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

            let inner = antecedent::StudyBuilder::exact_transport(
                diagram,
                functional,
                data,
                request,
                ExactEvaluationLimits { operations: max_operations, depth: max_depth },
                &ctx,
            )
            .map_err(error)?;
            Ok(PreparedExactStage {
                inner,
                graph,
                catalog,
                last: None,
                memory_bytes,
                max_support_rows,
            })
        })
    }
}

#[pyfunction]
#[pyo3(signature = (graph, selections, source, target, outcomes, treatments, *, max_steps=100_000, max_depth=256, memory_bytes=None, cancel=None))]
fn identify_classical_transport_stage(
    py: Python<'_>,
    graph: PyRef<'_, Admg>,
    selections: Vec<String>,
    source: String,
    target: String,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    max_steps: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ClassicalTransportStage> {
    let coordinates = |variables: Vec<String>| {
        variables.iter().map(|v| resolve(&graph.names, v)).collect::<PyResult<Arc<[VariableId]>>>()
    };
    let query = ClassicalTransportQuery {
        outcomes: coordinates(outcomes)?,
        treatments: coordinates(treatments)?,
        source: source.into(),
        target: target.into(),
    };
    let diagram =
        SelectionDiagram::try_new(graph.aligned_to_names(&graph.names)?, coordinates(selections)?)
            .map_err(error)?;
    let named = Admg { admg: graph.admg.clone(), names: graph.names.clone() };
    crate::detach_catch(py, move || {
        let mut ctx = ExecutionContext::production_default(0);
        ctx.memory.hard_limit_bytes = memory_bytes;
        crate::apply_cancel(&mut ctx, cancel);
        let result = identify_classical_transport(
            &diagram,
            &query,
            SidLimits { steps: max_steps, depth: max_depth },
            &ctx,
        )
        .map_err(error)?;
        let certificate =
            antecedent_io::transport_certificate::TransportCertificateWire::from_result(
                &diagram,
                &query,
                vec![],
                &result,
                SidLimits { steps: max_steps, depth: max_depth },
                &ctx,
            )
            .map_err(error)?;
        Ok(ClassicalTransportStage { result, graph: named, diagram, certificate })
    })
}

#[pyfunction]
#[pyo3(signature=(graph, catalog, target, outcomes, treatments, *, max_steps=100_000,max_depth=256,memory_bytes=None,cancel=None))]
fn identify_meta_transport_stage(
    py: Python<'_>,
    graph: PyRef<'_, Admg>,
    catalog: &Bound<'_, PyAny>,
    target: String,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    max_steps: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ClassicalTransportStage> {
    let catalog = parse_catalog(catalog, &graph)?;
    let coordinates = |names: Vec<String>| {
        names
            .iter()
            .map(|name| resolve(&graph.names, name))
            .collect::<PyResult<Arc<[VariableId]>>>()
    };
    let query = antecedent_identify::MetaTransportQuery::from_catalog(
        coordinates(outcomes)?,
        coordinates(treatments)?,
        target.into(),
        &catalog,
    )
    .map_err(error)?;
    let aligned = graph.aligned_to_names(&graph.names)?;
    let first = query
        .sources
        .iter()
        .min_by(|a, b| a.population.cmp(&b.population))
        .ok_or_else(|| error("meta transport requires a source"))?;
    let diagram = SelectionDiagram::try_new(
        aligned.clone(),
        first.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
    )
    .map_err(error)?;
    let named = Admg { admg: graph.admg.clone(), names: graph.names.clone() };
    crate::detach_catch(py, move || {
        let mut ctx = ExecutionContext::production_default(0);
        ctx.memory.hard_limit_bytes = memory_bytes;
        crate::apply_cancel(&mut ctx, cancel);
        let result = antecedent_identify::identify_meta_transport(
            &aligned,
            &query,
            SidLimits { steps: max_steps, depth: max_depth },
            &ctx,
        )
        .map_err(error)?;
        let mut sources = query.sources.clone();
        sources.sort_by(|a, b| a.population.cmp(&b.population));
        for source in &mut sources {
            source.selections.sort_unstable();
        }
        let classical = ClassicalTransportQuery {
            outcomes: query.outcomes.clone(),
            treatments: query.treatments.clone(),
            source: Arc::from(sources[0].population.as_str()),
            target: query.target.clone(),
        };
        let certificate =
            antecedent_io::transport_certificate::TransportCertificateWire::from_result(
                &diagram,
                &classical,
                sources,
                &result,
                SidLimits { steps: max_steps, depth: max_depth },
                &ctx,
            )
            .map_err(error)?;
        Ok(ClassicalTransportStage { result, graph: named, diagram, certificate })
    })
}

#[pyfunction]
#[pyo3(signature=(artifact,*,max_steps=100_000,max_depth=256,memory_bytes=None,cancel=None))]
fn consume_transport_certificate(
    py: Python<'_>,
    artifact: &[u8],
    max_steps: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ClassicalTransportStage> {
    let bytes = artifact
        .strip_prefix(b"ANTECEDENT-TRANSPORT-CERTIFICATE\x01")
        .ok_or_else(|| error("invalid certificate framing"))?;
    if memory_bytes.is_some_and(|n| bytes.len() as u64 > n) {
        return Err(error("certificate memory budget"));
    }
    let (names, bytes): (Vec<String>, Vec<u8>) = antecedent_io::from_cbor(bytes).map_err(error)?;
    crate::detach_catch(py, move || {
        let mut ctx = ExecutionContext::production_default(0);
        ctx.memory.hard_limit_bytes = memory_bytes;
        crate::apply_cancel(&mut ctx, cancel);
        let limits = SidLimits { steps: max_steps, depth: max_depth };
        let certificate = antecedent_io::transport_certificate::TransportCertificateWire::consume(
            &bytes, limits, &ctx,
        )
        .map_err(error)?;
        let (diagram, _, result) = certificate.check(limits, &ctx).map_err(error)?;
        validate_artifact_names(&names, diagram.causal_graph())?;
        if names.len() != diagram.causal_graph().node_count()
            || names.iter().collect::<std::collections::BTreeSet<_>>().len() != names.len()
        {
            return Err(error("invalid certificate names"));
        }
        Ok(ClassicalTransportStage {
            graph: Admg { admg: diagram.causal_graph().clone(), names },
            diagram,
            result,
            certificate,
        })
    })
}

pub(crate) fn validate_artifact_names(
    names: &[String],
    graph: &antecedent_graph::Admg,
) -> PyResult<()> {
    if names.len() != graph.node_count() || names.iter().collect::<std::collections::BTreeSet<_>>().len() != names.len() || graph.nodes().iter().any(|node| !matches!(node, antecedent_core::NodeRef::Static(v) if v.as_usize() < names.len())) {
        return Err(error("invalid artifact coordinate names"));
    }
    Ok(())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ClassicalTransportStage>()?;
    m.add_function(wrap_pyfunction!(consume_transport_certificate, m)?)?;
    m.add_function(wrap_pyfunction!(identify_meta_transport_stage, m)?)?;
    m.add_class::<PreparedExactStage>()?;
    m.add_function(wrap_pyfunction!(consume_exact_transport, m)?)?;
    m.add_function(wrap_pyfunction!(identify_classical_transport_stage, m)?)?;
    Ok(())
}

pub(crate) fn parse_exact_data(
    laws: &Bound<'_, PyAny>,
    catalog: &antecedent_core::EvidenceCatalog,
    graph: &Admg,
    max_support_rows: usize,
) -> PyResult<ExactTransportData> {
    parse_laws(laws, catalog, graph, max_support_rows, RegimeCheck::Strict)
}

#[pyclass(skip_from_py_object)]
struct PreparedExactStage {
    inner: antecedent::PreparedStudy<antecedent::ExactPreparedState>,
    graph: Admg,
    catalog: antecedent_core::EvidenceCatalog,
    last: Option<antecedent::ExactStudyResult>,
    memory_bytes: Option<u64>,
    max_support_rows: usize,
}
impl PreparedExactStage {
    fn ctx(&self, cancel: Option<crate::PyCancellationToken>) -> ExecutionContext {
        let mut ctx = ExecutionContext::production_default(0);
        ctx.memory.hard_limit_bytes = self.memory_bytes;
        crate::apply_cancel(&mut ctx, cancel);
        ctx
    }

    fn run_mechanism_sensitivity(
        &self,
        py: Python<'_>,
        outcome_values: Vec<f64>,
        parent_cardinalities: Vec<usize>,
        treatment_levels: [usize; 2],
        perturbed_treatment_level: Option<usize>,
        max_fraction: f64,
        source_kernel_regime: u32,
        source_kernel_snapshot: String,
        target_parent_regime: u32,
        target_parent_snapshot: String,
        source_kernel: Vec<(Vec<usize>, Vec<f64>)>,
        source_parent_law: Vec<(Vec<usize>, f64)>,
        target_parent_law: Vec<(usize, Vec<usize>, f64)>,
        decision_threshold: Option<f64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let spec = antecedent_validate::FixedGraphMechanismSensitivitySpec {
            outcome_values,
            parent_cardinalities,
            treatment_levels,
            max_fraction,
            decision_threshold,
            source_kernel_regime: RegimeId::from_raw(source_kernel_regime),
            source_kernel_snapshot,
            target_parent_regime: RegimeId::from_raw(target_parent_regime),
            target_parent_snapshot,
            source_kernel: source_kernel
                .into_iter()
                .map(|(parent_levels, outcome_probabilities)| {
                    antecedent_validate::SourceOutcomeKernelRow {
                        parent_levels,
                        outcome_probabilities,
                    }
                })
                .collect(),
            source_parent_law: source_parent_law
                .into_iter()
                .map(|(parent_levels, probability)| antecedent_validate::SourceParentLawRow {
                    parent_levels,
                    probability,
                })
                .collect(),
            target_parent_law: target_parent_law
                .into_iter()
                .map(|(treatment_level, parent_levels, probability)| {
                    antecedent_validate::TargetParentLawRow {
                        treatment_level,
                        parent_levels,
                        probability,
                    }
                })
                .collect(),
        };
        let ctx = self.ctx(cancel);
        let inner = self.inner.clone();
        let result = crate::detach_catch(py, move || {
            if let Some(level) = perturbed_treatment_level {
                inner.mechanism_sensitivity_at_treatment_level(&spec, level, &ctx).map_err(error)
            } else {
                inner.mechanism_sensitivity(&spec, &ctx).map_err(error)
            }
        })?;
        serde_json::to_string(&serde_json::json!({
            "estimand": result.estimand,
            "outcome": self.graph.names[result.outcome.as_usize()],
            "parents": result.parents.iter().map(|parent| self.graph.names[parent.as_usize()].as_str()).collect::<Vec<_>>(),
            "source_population": result.source_population,
            "target_population": result.target_population,
            "source_kernel_binding": {"regime": result.source_kernel_binding.0.raw(), "snapshot": result.source_kernel_binding.1},
            "target_parent_binding": {"regime": result.target_parent_binding.0.raw(), "snapshot": result.target_parent_binding.1},
            "assumptions": result.assumptions,
            "baseline": result.response.baseline,
            "assumption_range": {"minimum": result.response.minimum, "maximum": result.response.maximum},
            "interval_interpretation": result.response.interval_interpretation,
            "decision_threshold": decision_threshold,
            "perturbed_treatment_level": perturbed_treatment_level,
            "tipping_fraction": result.response.tipping_fraction,
            "optimization_receipt": {
                "minimizing_outcome_by_stratum": result.response.receipt.minimizing_outcome_by_stratum,
                "maximizing_outcome_by_stratum": result.response.receipt.maximizing_outcome_by_stratum,
                "fraction_domain": result.response.receipt.fraction_domain,
                "method": result.response.receipt.method
            }
        })).map_err(error)
    }
    fn payload(&self, result: &antecedent::ExactStudyResult) -> PyResult<ExactStagePayload> {
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
        ))
    }
}
#[pymethods]
impl PreparedExactStage {
    #[pyo3(signature=(outcome_values, parent_cardinalities, treatment_levels, max_fraction, source_kernel_regime, source_kernel_snapshot, target_parent_regime, target_parent_snapshot, source_kernel, source_parent_law, target_parent_law, decision_threshold=None, perturbed_treatment_level=None, cancel=None))]
    fn mechanism_sensitivity(
        &self,
        py: Python<'_>,
        outcome_values: Vec<f64>,
        parent_cardinalities: Vec<usize>,
        treatment_levels: [usize; 2],
        max_fraction: f64,
        source_kernel_regime: u32,
        source_kernel_snapshot: String,
        target_parent_regime: u32,
        target_parent_snapshot: String,
        source_kernel: Vec<(Vec<usize>, Vec<f64>)>,
        source_parent_law: Vec<(Vec<usize>, f64)>,
        target_parent_law: Vec<(usize, Vec<usize>, f64)>,
        decision_threshold: Option<f64>,
        perturbed_treatment_level: Option<usize>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        self.run_mechanism_sensitivity(
            py,
            outcome_values,
            parent_cardinalities,
            treatment_levels,
            perturbed_treatment_level,
            max_fraction,
            source_kernel_regime,
            source_kernel_snapshot,
            target_parent_regime,
            target_parent_snapshot,
            source_kernel,
            source_parent_law,
            target_parent_law,
            decision_threshold,
            cancel,
        )
    }

    #[getter]
    fn outcomes(&self) -> Vec<String> {
        self.inner.query().outcomes.iter().map(|v| self.graph.names[v.as_usize()].clone()).collect()
    }
    #[pyo3(signature=(execution=None,cancel=None))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        execution: Option<String>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<ExactStagePayload> {
        let ctx = self.ctx(cancel);
        let inner = self.inner.clone();
        let execution = execution.unwrap_or_else(|| inner.inspect().identities.execution);
        self.last = None;
        let result = crate::detach_catch(py, move || {
            inner.estimate_checked(&execution, &ctx).map_err(error)
        })?;
        let payload = self.payload(&result)?;
        self.last = Some(result);
        Ok(payload)
    }
    #[pyo3(signature=(laws,cancel=None))]
    fn refresh(
        &mut self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<ExactStagePayload> {
        let data = parse_exact_data(laws, &self.catalog, &self.graph, self.max_support_rows)?;
        let ctx = self.ctx(cancel);
        let mut candidate = self.inner.clone();
        let (candidate, result) = crate::detach_catch(py, move || {
            let result = candidate.refresh(data, &ctx).map_err(error)?;
            Ok((candidate, result))
        })?;
        self.inner = candidate;
        self.catalog = self.inner.evidence_catalog().clone();
        let payload = self.payload(&result)?;
        self.last = Some(result);
        Ok(payload)
    }
    #[pyo3(signature=(laws,cancel=None))]
    fn replace_snapshot(
        &mut self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let data = parse_exact_data(laws, &self.catalog, &self.graph, self.max_support_rows)?;
        let ctx = self.ctx(cancel);
        let mut candidate = self.inner.clone();
        let candidate = crate::detach_catch(py, move || {
            candidate.replace_snapshot(data, &ctx).map_err(error)?;
            Ok(candidate)
        })?;
        self.inner = candidate;
        self.catalog = self.inner.evidence_catalog().clone();
        self.last = None;
        Ok(())
    }
    fn inspection_json(&self) -> String {
        let view = self.inner.inspect();
        let factors:Vec<_> = view.factors.iter().map(|f| serde_json::json!({"expression":f.expression.raw(),"population":f.binding.population.as_ref(),"regime":f.binding.regime.map(antecedent_core::RegimeId::raw),"variables":f.variables.iter().map(|v|self.graph.names[v.as_usize()].as_str()).collect::<Vec<_>>(),"conditioned_on":f.conditioned_on.iter().map(|v|self.graph.names[v.as_usize()].as_str()).collect::<Vec<_>>(),"interventions":f.interventions.iter().map(|v|self.graph.names[v.as_usize()].as_str()).collect::<Vec<_>>()})).collect();
        let executed = self.last.is_some();
        let support:Vec<_>=self.last.iter().flat_map(|r|r.distribution().support.iter()).map(|r|serde_json::json!({
            "expression":r.expression.raw(),"status":r.status,"denominator":r.denominator,
            "assignment":r.assignment.iter().map(|(v,x)|(self.graph.names[v.as_usize()].as_str(),x.as_f64())).collect::<std::collections::BTreeMap<_,_>>()
        })).collect();
        let identification_status = view
            .reasoning
            .identification
            .as_ref()
            .map_or("unavailable", |slot| slot.status.as_str());
        let uncertainty_reason = match &view.reasoning.uncertainty {
            antecedent_core::SlotAvailability::Unavailable { reason } => reason.as_ref(),
            _ => "",
        };
        serde_json::json!({
            "identification":{"available":view.reasoning.identification.is_available(),"summary":identification_status,"payload":{"formula":view.formula,"theorem_scope":view.theorem_scope,"classical_family":view.classical_scope.family.as_str(),"classical_guarantee":view.classical_scope.outcome_guarantees.as_str(),"classical_version":view.classical_scope.reference.version.as_ref(),"catalog_family":view.catalog_scope.family.as_str(),"catalog_guarantee":view.catalog_scope.outcome_guarantees.as_str(),"multi_node_recursion":view.classical_scope.computation_limits.multi_node_c_component_recursion,"rules":self.inner.rules()}},
            // `view.reasoning` comes from `PreparedStudy::inspect`, which always reports
            // support as not-yet-evaluated: it has no visibility into this binding's own
            // retained `last` result, so that stays the actual evaluated/not-evaluated flag.
            "support":{"available":executed,"summary":if executed {"exact_factor_support_checked"} else {"execution_specific"},"payload":{"required_factors":factors,"bindings":view.bindings,"factor_support":support}},
            "uncertainty":{"available":view.reasoning.uncertainty.is_available(),"reason":uncertainty_reason,"summary":"unavailable"},
            "assumptions":{"available":true,"summary":"declared_selection_diagram_and_exact_laws","payload":{"empirically_verified":false,"source":self.inner.query().source.as_ref(),"target":self.inner.query().target.as_ref(),"selection_targets":self.inner.diagram().selection_targets().iter().map(|v|self.graph.names[v.as_usize()].as_str()).collect::<Vec<_>>()}},
            "target_id":view.identities.target,"observation_id":view.identities.observation,"inference_binding_id":view.identities.inference_binding,"program_id":view.identities.program,"identification_id":view.identities.identification,
            "identification_product_id":view.identities.identification_product,"data_snapshot_id":view.identities.snapshot,"execution_id":view.identities.execution,
            "supported_operations":["estimate","mechanism_sensitivity","replace_snapshot","refresh","export"],
            "formula":view.formula
        }).to_string()
    }
    fn last_result(&self) -> PyResult<ExactStagePayload> {
        self.payload(self.last.as_ref().ok_or_else(|| error("transport.no_execution_claim"))?)
    }
    fn plan_summary(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::from([
            ("plan_id".into(), self.inner.inspect().identities.execution),
            ("structure_source".into(), "explicit".into()),
            ("deterministic_reductions".into(), "true".into()),
            ("kernels".into(), "exact_discrete_memoized_elimination".into()),
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
        }
    }
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let result = self.last.as_ref().ok_or_else(|| error("transport.no_execution_claim"))?;
        let artifact = self.inner.export(result).map_err(error)?;
        let bytes = antecedent_io::to_cbor(&(self.graph.names.clone(), artifact)).map_err(error)?;
        let mut framed = b"ANTECEDENT-EXACT-TRANSPORT\x01".to_vec();
        framed.extend(bytes);
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

#[pyfunction]
#[pyo3(signature=(bytes,*,max_operations=10_000_000,max_depth=256,memory_bytes=None,cancel=None))]
fn consume_exact_transport(
    py: Python<'_>,
    bytes: Vec<u8>,
    max_operations: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PreparedExactStage> {
    crate::detach_catch(py, move || {
        if memory_bytes.is_some_and(|n| bytes.len() as u64 > n) {
            return Err(error("artifact memory budget"));
        }
        let payload = bytes
            .strip_prefix(b"ANTECEDENT-EXACT-TRANSPORT\x01")
            .ok_or_else(|| error("invalid exact transport artifact format"))?;
        let (names, artifact): (Vec<String>, Vec<u8>) =
            antecedent_io::from_cbor(payload).map_err(error)?;
        let mut ctx = ExecutionContext::production_default(0);
        ctx.memory.hard_limit_bytes = memory_bytes;
        crate::apply_cancel(&mut ctx, cancel);
        let (inner, result) = antecedent::PreparedStudy::<antecedent::ExactPreparedState>::consume(
            &artifact,
            ExactEvaluationLimits { operations: max_operations, depth: max_depth },
            &ctx,
        )
        .map_err(error)?;
        if names.len() != inner.diagram().causal_graph().node_count()
            || names.iter().collect::<std::collections::BTreeSet<_>>().len() != names.len()
        {
            return Err(error("artifact coordinate names"));
        }
        validate_artifact_names(&names, inner.diagram().causal_graph())?;
        let graph = Admg { admg: inner.diagram().causal_graph().clone(), names };
        let catalog = inner.evidence_catalog().clone();
        Ok(PreparedExactStage {
            inner,
            graph,
            catalog,
            last: Some(result),
            memory_bytes,
            max_support_rows: max_operations,
        })
    })
}
