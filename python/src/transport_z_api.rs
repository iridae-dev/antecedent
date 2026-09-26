//! Python bindings for bounded single-source point-only z-transport.
use crate::transport_common::{
    RegimeCheck, assignment_from_pairs, error, execution_context, frame_named_artifact, parse_laws,
    resolve, serialization_error, unframe_named_artifact,
};
use crate::{graphs::Admg, transport_interference_api::parse_catalog};
use antecedent_core::{EvidenceCatalogDelta, ExecutionContext, Value, VariableId};
use antecedent_design::{
    CandidateDesign, DesignCost, ExperimentPlan, MeasurementPlan, TransportEvidenceCandidate,
    ZTransportCandidateOutcome, ZTransportFailureSnapshot, ZTransportFailureSnapshotWire,
    plan_z_transport_evidence, snapshot_z_transport_failure,
};
use antecedent_expr::ExactEvaluationLimits;
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    SidLimits, TwoSourceZTransportDecision, TwoSourceZTransportQuery, ZTransportDecision,
    ZTransportDerivation, ZTransportMissingEvidence, ZTransportObstruction, ZTransportQuery,
    ZTransportResult, ZTransportSourceSpec, bind_z_transport_catalog,
    decide_two_source_z_transport, decide_z_transport_with_catalog, identify_z_transport_surrogate,
};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Magic prefix of a portable z-transport point artifact: the io crate's
/// versioned CBOR wire, framed with the variable names it was built under.
pub(crate) const Z_TRANSPORT_PREFIX: &[u8] = b"ANTECEDENT-Z-TRANSPORT\x01";

/// Resource limits a stage was identified under; every later call inherits them.
#[derive(Clone, Copy)]
struct StageLimits {
    max_steps: usize,
    max_depth: usize,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
}

/// Keep externally named regime handles stable when a catalog adds a row.
/// `parse_catalog` assigns compact IDs in name order, so inserting a proposed
/// row could otherwise renumber every existing binding in the hypothetical or
/// arriving catalog.
fn align_catalog_regime_ids(
    base: &antecedent_core::EvidenceCatalog,
    catalog: &antecedent_core::EvidenceCatalog,
) -> PyResult<antecedent_core::EvidenceCatalog> {
    use antecedent_core::RegimeId;

    let mut aligned = catalog.clone();
    let mut remap = Vec::<(RegimeId, RegimeId)>::new();
    let mut used = base.regimes.iter().map(|regime| regime.id.raw()).collect::<Vec<_>>();
    let mut next = used.iter().copied().max().map_or(0, |id| id.saturating_add(1));

    for regime in Arc::make_mut(&mut aligned.regimes) {
        let previous = regime.id;
        let target = if let Some(existing) =
            base.regimes.iter().find(|candidate| candidate.label == regime.label)
        {
            let mut comparable = regime.clone();
            comparable.id = existing.id;
            if comparable != *existing {
                return Err(crate::value_err("catalog changed an existing regime contract"));
            }
            existing.id
        } else {
            while used.contains(&next) {
                next =
                    next.checked_add(1).ok_or_else(|| crate::value_err("regime IDs exhausted"))?;
            }
            let id = RegimeId::from_raw(next);
            used.push(next);
            next = next.saturating_add(1);
            id
        };
        remap.push((previous, target));
        regime.id = target;
    }
    for binding in Arc::make_mut(&mut aligned.bindings) {
        binding.regime = remap
            .iter()
            .find_map(|(previous, target)| (*previous == binding.regime).then_some(*target))
            .ok_or_else(|| crate::value_err("catalog binding references an unknown regime"))?;
    }
    aligned.validate().map_err(error)?;
    Ok(aligned)
}

/// JSON for one bounded `TRz` decision, with missing evidence spelled in
/// variable names and as a structured object rather than a debug dump.
fn decision_json(
    decision: &ZTransportDecision,
    catalog: &antecedent_core::EvidenceCatalog,
    names: &[String],
) -> serde_json::Value {
    match decision {
        ZTransportDecision::Identified(proof) => serde_json::json!({
            "outcome": "identified",
            "proof": proof.to_record(),
            "inspection": proof.inspect_proof(catalog),
        }),
        ZTransportDecision::ProvenNonTransportable(obstruction) => serde_json::json!({
            "outcome": "proven_non_transportable",
            "obstruction": obstruction.to_record(),
        }),
        ZTransportDecision::MissingEvidence { missing } => {
            let (kind, detail, structured) = match missing {
                ZTransportMissingEvidence::UnassignedControllable { variable } => {
                    let name = names[variable.as_usize()].as_str();
                    (
                        "unassigned_controllable",
                        format!("the experiment on {name} names no level for line 10 exchange"),
                        serde_json::json!({"kind": "unassigned_controllable", "variable": name}),
                    )
                }
                ZTransportMissingEvidence::CitedFactor { detail } => {
                    let detail = crate::transport_common::resolve_variable_ids(detail, names);
                    (
                        "cited_factor",
                        format!("the cited joint law {detail} is not supplied"),
                        serde_json::json!({"kind": "cited_factor", "detail": detail}),
                    )
                }
            };
            serde_json::json!({
                "outcome": "missing_evidence",
                "reason": "z_transport.missing_evidence",
                "kind": kind,
                "detail": detail,
                "missing": structured,
            })
        }
        ZTransportDecision::NotCertified { reason } => serde_json::json!({
            "outcome": "not_certified",
            "reason": reason,
        }),
    }
}

fn to_py_json(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    Ok(py.import("json")?.call_method1("loads", (value.to_string(),))?.unbind())
}

fn py_bytes(py: Python<'_>, bytes: &[u8]) -> Py<pyo3::types::PyBytes> {
    pyo3::types::PyBytes::new(py, bytes).unbind()
}

#[pyclass(skip_from_py_object)]
struct ZTransportStage {
    result: ZTransportResult,
    graph: Admg,
    diagram: SelectionDiagram,
    query: ZTransportQuery,
    limits: StageLimits,
}

impl ZTransportStage {
    /// The checked derivation, or the stage's own refusal.
    fn identified(&self) -> PyResult<ZTransportDerivation> {
        match &self.result {
            ZTransportResult::Identified(proof) => Ok(proof.as_ref().clone()),
            ZTransportResult::NotCertified { reason } => {
                Err(error(format!("zTR refused: {reason}")))
            }
        }
    }

    /// Prepare the exact or empirical plugin point route on the checked proof.
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        laws: &Bound<'_, PyAny>,
        assignments: BTreeMap<String, f64>,
        empirical: bool,
        max_operations: usize,
        max_depth: usize,
        max_support_rows: Option<usize>,
        memory_bytes: Option<u64>,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedZTransportStage> {
        let proof = self.identified()?;
        let catalog = parse_catalog(catalog, &self.graph)?;
        let max_support_rows = max_support_rows.unwrap_or(self.limits.max_support_rows);
        let memory_bytes = memory_bytes.or(self.limits.memory_bytes);
        let data = parse_laws(laws, &catalog, &self.graph, max_support_rows, RegimeCheck::Strict)?;
        if empirical && data.laws().iter().any(|law| law.empirical_counts().is_none()) {
            return Err(error("z_transport.empirical_counts_required"));
        }
        let request = assignment_from_pairs(&self.graph.names, assignments)?;
        let diagram = self.diagram.clone();
        let query = self.query.clone();
        let named_graph = self.graph.clone();
        crate::detach_catch(py, move || {
            let ctx = execution_context(seed, memory_bytes, cancel);
            let functional =
                bind_z_transport_catalog(&diagram, &query, &proof, &catalog).map_err(error)?;
            PreparedZTransportStage::build(
                diagram,
                functional,
                data,
                request,
                empirical,
                ExactEvaluationLimits { operations: max_operations, depth: max_depth },
                &ctx,
                named_graph,
                catalog,
                max_support_rows,
                memory_bytes,
                seed,
            )
        })
    }
}

#[pymethods]
impl ZTransportStage {
    /// Decide the bounded theorem against the actual complete catalog.
    /// A missing experiment is reported separately from a checked obstruction.
    #[pyo3(signature=(catalog, *, max_steps=None, max_depth=None, memory_bytes=None, cancel=None))]
    fn decide(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        max_steps: Option<usize>,
        max_depth: Option<usize>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyAny>> {
        let catalog = parse_catalog(catalog, &self.graph)?;
        let limits = SidLimits {
            steps: max_steps.unwrap_or(self.limits.max_steps),
            depth: max_depth.unwrap_or(self.limits.max_depth),
        };
        let memory_bytes = memory_bytes.or(self.limits.memory_bytes);
        let diagram = self.diagram.clone();
        let query = self.query.clone();
        let names = self.graph.names.clone();
        let value = crate::detach_catch(py, move || {
            let ctx = execution_context(0, memory_bytes, cancel);
            let decision =
                decide_z_transport_with_catalog(&diagram, &query, &catalog, limits, &ctx)
                    .map_err(error)?;
            Ok(decision_json(&decision, &catalog, &names))
        })?;
        to_py_json(py, &value)
    }

    #[getter]
    fn outcome(&self) -> &'static str {
        match &self.result {
            ZTransportResult::Identified(_) => "identified",
            ZTransportResult::NotCertified { .. } => "not_certified",
        }
    }
    #[getter]
    fn reason(&self) -> Option<&'static str> {
        match &self.result {
            ZTransportResult::Identified(_) => None,
            ZTransportResult::NotCertified { reason } => Some(reason),
        }
    }

    /// Export the exact failure state used when assessing candidate studies.
    fn failure_snapshot(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
    ) -> PyResult<Py<pyo3::types::PyBytes>> {
        let catalog = parse_catalog(catalog, &self.graph)?;
        let snapshot =
            snapshot_z_transport_failure(&self.diagram, &self.query, &catalog).map_err(error)?;
        let wire = snapshot.to_wire().map_err(error)?;
        let bytes = serde_json::to_vec(&wire).map_err(serialization_error)?;
        Ok(py_bytes(py, &bytes))
    }

    /// Inspect every checked rule and source-factor obligation against a catalog.
    fn inspect_proof(&self, py: Python<'_>, catalog: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let proof = self.identified()?;
        let catalog = parse_catalog(catalog, &self.graph)?;
        let inspection =
            serde_json::to_value(proof.inspect_proof(&catalog)).map_err(serialization_error)?;
        to_py_json(py, &inspection)
    }

    #[pyo3(signature=(catalog, laws, assignments, *, max_operations=10_000_000, max_depth=256, max_support_rows=None, memory_bytes=None, seed=0, cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_exact(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        laws: &Bound<'_, PyAny>,
        assignments: BTreeMap<String, f64>,
        max_operations: usize,
        max_depth: usize,
        max_support_rows: Option<usize>,
        memory_bytes: Option<u64>,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedZTransportStage> {
        self.prepare(
            py,
            catalog,
            laws,
            assignments,
            false,
            max_operations,
            max_depth,
            max_support_rows,
            memory_bytes,
            seed,
            cancel,
        )
    }

    /// Prepare an empirical plugin table only when every law has counts.
    #[pyo3(signature=(catalog, laws, assignments, *, max_operations=10_000_000, max_depth=256, max_support_rows=None, memory_bytes=None, seed=0, cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_empirical(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        laws: &Bound<'_, PyAny>,
        assignments: BTreeMap<String, f64>,
        max_operations: usize,
        max_depth: usize,
        max_support_rows: Option<usize>,
        memory_bytes: Option<u64>,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedZTransportStage> {
        self.prepare(
            py,
            catalog,
            laws,
            assignments,
            true,
            max_operations,
            max_depth,
            max_support_rows,
            memory_bytes,
            seed,
            cancel,
        )
    }

    /// Plan proposed studies and return assessment JSON plus verified proposal objects.
    /// Each candidate has `id`, proposed-only `catalog`, `design_kind`, `targets`,
    /// `measured`, `cost`, `sample_budget`, `tag`, `recruitment_sampling`, and
    /// `feasibility_constraints` attributes. Proposals inherit this stage's limits.
    #[pyo3(signature=(catalog, candidates, failure_snapshot=None))]
    fn plan_evidence(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        candidates: Vec<Py<PyAny>>,
        failure_snapshot: Option<Vec<u8>>,
    ) -> PyResult<(String, Vec<Py<ZTransportProposalStage>>)> {
        let base = parse_catalog(catalog, &self.graph)?;
        let expected_snapshot =
            snapshot_z_transport_failure(&self.diagram, &self.query, &base).map_err(error)?;
        let snapshot = if let Some(bytes) = failure_snapshot {
            let wire: ZTransportFailureSnapshotWire =
                serde_json::from_slice(&bytes).map_err(serialization_error)?;
            let supplied = ZTransportFailureSnapshot::from_wire(&wire).map_err(error)?;
            let expected_wire = expected_snapshot.to_wire().map_err(error)?;
            let supplied_json = serde_json::to_vec(&wire).map_err(serialization_error)?;
            let expected_json = serde_json::to_vec(&expected_wire).map_err(serialization_error)?;
            if supplied_json != expected_json {
                return Err(serialization_error(
                    "failure snapshot does not match this stage and catalog",
                ));
            }
            supplied
        } else {
            expected_snapshot
        };
        let mut parsed = Vec::with_capacity(candidates.len());
        for item in candidates {
            let item = item.bind(py);
            let id: String = item.getattr("id")?.extract()?;
            let parsed_proposed = parse_catalog(&item.getattr("catalog")?, &self.graph)?;
            let proposed_catalog = align_catalog_regime_ids(&base, &parsed_proposed)?;
            if proposed_catalog.environments != base.environments
                || proposed_catalog.target_sampling != base.target_sampling
                || base.regimes.iter().any(|r| !proposed_catalog.regimes.contains(r))
                || base.bindings.iter().any(|b| !proposed_catalog.bindings.contains(b))
            {
                return Err(crate::value_err(
                    "candidate catalog must preserve the failure catalog",
                ));
            }
            let proposed_regimes = proposed_catalog
                .regimes
                .iter()
                .filter(|r| r.evidence_kind == antecedent_core::EvidenceKind::Proposed)
                .cloned()
                .collect::<Vec<_>>();
            let delta = EvidenceCatalogDelta::try_new(&base, proposed_regimes).map_err(error)?;
            let design_kind: String = item.getattr("design_kind")?.extract()?;
            let targets: Vec<String> = item.getattr("targets")?.extract()?;
            let measured: Vec<String> = item.getattr("measured")?.extract()?;
            let target_ids = targets
                .iter()
                .map(|name| resolve(&self.graph.names, name))
                .collect::<PyResult<Vec<_>>>()?;
            let measured_ids = measured
                .iter()
                .map(|name| resolve(&self.graph.names, name))
                .collect::<PyResult<Vec<_>>>()?;
            let cost_amount: f64 = item.getattr("cost")?.extract()?;
            let sample_budget: u64 = item.getattr("sample_budget")?.extract()?;
            let cost = DesignCost { amount: cost_amount, sample_budget };
            let tag: u64 = item.getattr("tag")?.extract()?;
            let design = match design_kind.as_str() {
                "intervene" => CandidateDesign::Intervene(ExperimentPlan {
                    targets: target_ids.into(),
                    cost,
                    tag,
                }),
                "measure" => CandidateDesign::Measure(MeasurementPlan {
                    variables: measured_ids.into(),
                    cost,
                    tag,
                }),
                _ => return Err(crate::value_err("design_kind must be 'intervene' or 'measure'")),
            };
            let recruitment_sampling: String = item.getattr("recruitment_sampling")?.extract()?;
            let feasibility_constraints: Vec<String> =
                item.getattr("feasibility_constraints")?.extract()?;
            parsed.push(TransportEvidenceCandidate {
                id: Arc::from(id),
                delta,
                design,
                recruitment_sampling: Arc::from(recruitment_sampling),
                feasibility_constraints: feasibility_constraints
                    .into_iter()
                    .map(Arc::from)
                    .collect::<Vec<_>>()
                    .into(),
                cost,
            });
        }
        let plan = plan_z_transport_evidence(&snapshot, &parsed);
        let mut proposals = Vec::new();
        let mut assessments = Vec::new();
        for assessment in plan.assessments {
            let (outcome, reason) = match assessment.outcome {
                ZTransportCandidateOutcome::VerifiedSufficient(proposal) => {
                    proposals.push(Py::new(
                        py,
                        ZTransportProposalStage {
                            inner: proposal,
                            graph: self.graph.clone(),
                            diagram: self.diagram.clone(),
                            query: self.query.clone(),
                            limits: self.limits,
                        },
                    )?);
                    ("verified_sufficient", None)
                }
                ZTransportCandidateOutcome::Rejected { reason } => {
                    ("rejected", Some(reason.to_string()))
                }
            };
            assessments.push(
                serde_json::json!({"id":assessment.id.as_ref(),"cost":assessment.cost.amount,
                "sample_budget":assessment.cost.sample_budget,"outcome":outcome,"reason":reason}),
            );
        }
        let report = serde_json::json!({"assessments":assessments,
            "ranked_sufficient":plan.ranked_sufficient.iter().map(AsRef::as_ref).collect::<Vec<_>>()})
        .to_string();
        Ok((report, proposals))
    }
}

#[pyclass(skip_from_py_object)]
struct PreparedZTransportStage {
    inner: antecedent::PreparedZTransport,
    graph: Admg,
    diagram: SelectionDiagram,
    catalog: antecedent_core::EvidenceCatalog,
    last: Option<antecedent::ZTransportResult>,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    seed: u64,
}

#[pyclass(skip_from_py_object)]
struct ZTransportProposalStage {
    inner: antecedent_design::ZTransportProposal,
    graph: Admg,
    diagram: SelectionDiagram,
    query: ZTransportQuery,
    limits: StageLimits,
}

#[pymethods]
impl ZTransportProposalStage {
    /// Serialize the frozen failure, candidate delta, and checked derivation.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let wire = self.inner.to_wire().map_err(error)?;
        let bytes = serde_json::to_vec(&wire).map_err(serialization_error)?;
        Ok(py_bytes(py, &bytes))
    }

    /// Recheck that the original snapshot and hypothetical proof still match.
    fn replay(&self) -> PyResult<()> {
        self.inner.replay().map_err(error)
    }

    /// Receive matching study results, reidentify against the actual catalog, and prepare.
    #[pyo3(signature=(catalog, laws, assignments, provider_snapshot, *, empirical=false,
        max_operations=10_000_000, max_depth=256, max_support_rows=None, memory_bytes=None, seed=0, cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn receive(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        laws: &Bound<'_, PyAny>,
        assignments: BTreeMap<String, f64>,
        provider_snapshot: String,
        empirical: bool,
        max_operations: usize,
        max_depth: usize,
        max_support_rows: Option<usize>,
        memory_bytes: Option<u64>,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedZTransportStage> {
        let parsed_catalog = parse_catalog(catalog, &self.graph)?;
        let proposal_wire = self.inner.to_wire().map_err(error)?;
        let base_catalog = proposal_wire.snapshot.catalog.to_catalog().map_err(error)?;
        let catalog = align_catalog_regime_ids(&base_catalog, &parsed_catalog)?;
        let _arrival = self.inner.receive(&catalog, provider_snapshot).map_err(error)?;
        let max_support_rows = max_support_rows.unwrap_or(self.limits.max_support_rows);
        let memory_bytes = memory_bytes.or(self.limits.memory_bytes);
        let data = parse_laws(laws, &catalog, &self.graph, max_support_rows, RegimeCheck::Strict)?;
        if empirical && data.laws().iter().any(|law| law.empirical_counts().is_none()) {
            return Err(error("z_transport.empirical_counts_required"));
        }
        let request = assignment_from_pairs(&self.graph.names, assignments)?;
        let diagram = self.diagram.clone();
        let query = self.query.clone();
        let named_graph = self.graph.clone();
        crate::detach_catch(py, move || {
            let ctx = execution_context(seed, memory_bytes, cancel);
            let proof = match identify_z_transport_surrogate(&diagram, &query).map_err(error)? {
                ZTransportResult::Identified(proof) => proof,
                ZTransportResult::NotCertified { reason } => return Err(error(reason)),
            };
            let functional =
                bind_z_transport_catalog(&diagram, &query, &proof, &catalog).map_err(error)?;
            PreparedZTransportStage::build(
                diagram,
                functional,
                data,
                request,
                empirical,
                ExactEvaluationLimits { operations: max_operations, depth: max_depth },
                &ctx,
                named_graph,
                catalog,
                max_support_rows,
                memory_bytes,
                seed,
            )
        })
    }
}

impl PreparedZTransportStage {
    /// The one constructor behind prepare_exact, prepare_empirical and receive.
    #[allow(clippy::too_many_arguments)]
    fn build(
        diagram: SelectionDiagram,
        functional: antecedent_identify::BoundZTransportFunctional,
        data: antecedent_expr::ExactTransportData,
        request: antecedent_expr::Assignment,
        empirical: bool,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
        graph: Admg,
        catalog: antecedent_core::EvidenceCatalog,
        max_support_rows: usize,
        memory_bytes: Option<u64>,
        seed: u64,
    ) -> PyResult<Self> {
        let sensitivity_diagram = diagram.clone();
        let inner = if empirical {
            antecedent::StudyBuilder::z_transport_empirical(
                diagram, functional, data, request, limits, ctx,
            )
        } else {
            antecedent::StudyBuilder::z_transport(diagram, functional, data, request, limits, ctx)
        }
        .map_err(error)?;
        Ok(Self {
            inner,
            graph,
            diagram: sensitivity_diagram,
            catalog,
            last: None,
            max_support_rows,
            memory_bytes,
            seed,
        })
    }

    fn ctx(
        &self,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> ExecutionContext {
        execution_context(self.seed, memory_bytes.or(self.memory_bytes), cancel)
    }

    fn last_result(&self, what: &str) -> PyResult<&antecedent::ZTransportResult> {
        self.last.as_ref().ok_or_else(|| error(format!("transport.no_execution_claim: {what}")))
    }

    /// The raw io-crate artifact of the last execution.
    fn raw_export(&self, what: &str) -> PyResult<Vec<u8>> {
        self.last_result(what)?.export(&self.inner).map_err(error)
    }

    /// Interval JSON for one execution: availability is the presence of mean
    /// intervals, never a string test on the method name.
    fn interval_json(&self, result: &antecedent::ZTransportResult) -> serde_json::Value {
        let names = &self.graph.names;
        if result.mean_intervals().is_empty() {
            return serde_json::json!({"available": false, "reason": result.interval_reason()});
        }
        serde_json::json!({
            "available": true,
            "method": result.interval_type(),
            "reason": result.interval_reason(),
            "coverage_target": result.coverage_target(),
            "seed": self.seed,
            "mean_intervals": result.mean_intervals().iter().map(|(variable, lower, upper)| serde_json::json!({
                "outcome": &names[variable.as_usize()],
                "lower": lower,
                "upper": upper,
            })).collect::<Vec<_>>(),
        })
    }
}

#[pymethods]
impl PreparedZTransportStage {
    /// Exact outcome-kernel contamination range for the registered surrogate formula.
    /// Other in-bound graphs refuse with `IncompatibleFormula`; call after `estimate`.
    #[pyo3(signature=(max_fraction, decision_threshold=None, *, memory_bytes=None, cancel=None))]
    fn mechanism_sensitivity(
        &self,
        py: Python<'_>,
        max_fraction: f64,
        decision_threshold: Option<f64>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyAny>> {
        let baseline_bytes = self.raw_export("estimate before sensitivity analysis")?;
        let ctx = self.ctx(memory_bytes, cancel);
        let diagram = self.diagram.clone();
        let inner = self.inner.clone();
        let result = crate::detach_catch(py, move || {
            antecedent_validate::z_transport_mechanism_sensitivity(
                &diagram,
                inner.functional(),
                inner.data(),
                max_fraction,
                decision_threshold,
                &ctx,
            )
            .map_err(error)
        })?;
        let baseline_digest = digest_hex("z_transport_sensitivity_baseline", &baseline_bytes);
        let provider_snapshots = self
            .inner
            .data()
            .laws()
            .iter()
            .map(|law| law.snapshot_identity().to_owned())
            .collect::<Vec<_>>();
        let value = serde_json::json!({
            "status":"available",
            "estimand": SENSITIVITY_ESTIMAND,
            "baseline":result.response.baseline,
            "assumption_range":{"minimum":result.response.minimum,"maximum":result.response.maximum},
            "interval_interpretation":result.response.interval_interpretation,
            "delta_domain":result.response.receipt.fraction_domain,
            "decision_threshold":decision_threshold,
            "tipping_fraction":result.response.tipping_fraction,
            "minimizing_outcome_by_stratum":result.response.receipt.minimizing_outcome_by_stratum,
            "maximizing_outcome_by_stratum":result.response.receipt.maximizing_outcome_by_stratum,
            "method":result.response.receipt.method,
            "baseline_binding":{"artifact_digest":baseline_digest,"query":result.query_binding,"source_regime":result.source_regime.raw(),"provider_snapshot":result.provider_snapshot,"provider_snapshots":provider_snapshots}
        });
        to_py_json(py, &value)
    }

    /// Execute the frozen point formula. Cited empirical tables also publish a
    /// nominal interval; `seed` overrides the prepared seed for this execution.
    #[pyo3(signature=(estimator=None, posterior_draws=None, *, memory_bytes=None, cancel=None, seed=None))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        estimator: Option<String>,
        posterior_draws: Option<u32>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
        seed: Option<u64>,
    ) -> PyResult<String> {
        let provider = match estimator.as_deref() {
            None => None,
            Some("empirical_support_bayesian_bootstrap") => {
                Some(antecedent_estimate::BayesianTransportLawProvider::EmpiricalSupport)
            }
            Some("state_space_dirichlet") => {
                Some(antecedent_estimate::BayesianTransportLawProvider::DeclaredStateSpaceDirichlet)
            }
            Some(name) => {
                return Err(crate::value_err(format!("unknown z-transport provider {name}")));
            }
        };
        if provider.is_none() && posterior_draws.is_some() {
            return Err(crate::value_err(
                "posterior_draws requires a Bayesian z-transport provider",
            ));
        }
        if let Some(seed) = seed {
            self.seed = seed;
        }
        let draws = posterior_draws.unwrap_or(199);
        let inner = self.inner.clone();
        let ctx = self.ctx(memory_bytes, cancel);
        let result = crate::detach_catch(py, move || match provider {
            Some(provider) => inner.estimate_bayesian(provider, draws, &ctx).map_err(error),
            None => inner.estimate(&ctx).map_err(error),
        })?;
        let names = &self.graph.names;
        let query = self.inner.functional().derivation().query();
        let distribution = result.distribution();
        let payload = serde_json::json!({
            "status":"available",
            "scope":"single_source_z_transport_cited_joints_sound_incomplete",
            "seed": self.seed,
            "outcomes":query.outcomes.iter().map(|v| &names[v.as_usize()]).collect::<Vec<_>>(),
            "atoms":distribution.atoms.iter().map(|row| row.iter().map(Value::as_f64).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "probabilities":distribution.probabilities.as_ref(),
            "factor_support":distribution.support.iter().map(|s| serde_json::json!({
                "expression":s.expression.raw(), "status":s.status, "denominator":s.denominator,
                "assignment":s.assignment.iter().map(|(v,x)| (names[v.as_usize()].as_str(), x.as_f64())).collect::<BTreeMap<_,_>>()
            })).collect::<Vec<_>>(),
            "interval": self.interval_json(&result)
        })
        .to_string();
        self.last = Some(result);
        Ok(payload)
    }

    /// Rebind fresh laws under the retained catalog and proof; the previous
    /// prepared state survives a failed rebind and the last claim is cleared.
    #[pyo3(signature=(laws, *, memory_bytes=None, cancel=None))]
    fn refresh(
        &mut self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let data = parse_laws(
            laws,
            &self.catalog,
            &self.graph,
            self.max_support_rows,
            RegimeCheck::Strict,
        )?;
        let inner = self.inner.clone();
        let ctx = self.ctx(memory_bytes, cancel);
        self.inner = crate::detach_catch(py, move || inner.refresh(data, &ctx).map_err(error))?;
        self.last = None;
        Ok(())
    }

    /// The last execution as a framed, independently consumable artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let raw = self.raw_export("estimate before exporting a z-transport artifact")?;
        Ok(py_bytes(py, &frame_named_artifact(Z_TRANSPORT_PREFIX, &self.graph.names, raw)?))
    }

    #[pyo3(signature=(max_fraction, decision_threshold=None, *, memory_bytes=None, cancel=None))]
    fn export_sensitivity(
        &self,
        py: Python<'_>,
        max_fraction: f64,
        decision_threshold: Option<f64>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<pyo3::types::PyBytes>> {
        let baseline = self.raw_export("estimate before exporting sensitivity artifact")?;
        let ctx = self.ctx(memory_bytes, cancel);
        let bytes = crate::detach_catch(py, move || {
            antecedent::ZTransportSensitivityArtifactWire::checked(
                baseline,
                max_fraction,
                decision_threshold,
                &ctx,
            )
            .map_err(error)?
            .export()
            .map_err(error)
        })?;
        Ok(py_bytes(py, &bytes))
    }

    #[getter]
    fn interval_type(&self) -> &'static str {
        self.last.as_ref().map_or("no_interval_reported", |result| result.interval_type())
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }
}

const SENSITIVITY_ESTIMAND: &str = "target active-minus-control mean outcome response";

fn digest_hex(domain: &str, bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    antecedent_io::identity::payload_digest(domain, bytes).iter().fold(
        String::new(),
        |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        },
    )
}

/// Independently recheck a framed point artifact and recompute its point.
/// The io crate decides which artifact versions it accepts.
#[pyfunction]
#[pyo3(signature=(artifact, *, memory_bytes=None, cancel=None))]
fn consume_z_transport_artifact(
    py: Python<'_>,
    artifact: &[u8],
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let (names, bytes) = unframe_named_artifact(Z_TRANSPORT_PREFIX, artifact, "z-transport")?;
    crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        let (diagram, result) =
            antecedent::consume_z_transport_artifact(&bytes, &ctx).map_err(error)?;
        crate::transport_exact_api::validate_artifact_names(&names, diagram.causal_graph())?;
        // The rechecked derivation travels with the artifact; report its rules so a
        // loaded result names the same theorem steps as the live one.
        let (_diagram, functional, _data, _request, _limits, _wire) =
            antecedent_io::z_transport_artifact::ZTransportArtifactWire::reconstruct(&bytes)
                .map_err(error)?;
        let distribution = result.distribution();
        let name = |v: &VariableId| names[v.as_usize()].as_str();
        Ok(serde_json::json!({
            "status":"available",
            "scope":"single_source_z_transport_cited_joints_sound_incomplete",
            "proof": functional.derivation().to_record(),
            "outcomes":distribution.outcomes.iter().map(name).collect::<Vec<_>>(),
            "atoms":distribution.atoms.iter().map(|a|a.iter().map(Value::as_f64).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "probabilities":distribution.probabilities.to_vec(),
            "interval":{"available":false,"reason":result.interval_reason()}
        }).to_string())
    })
}

/// Independently recheck a portable sensitivity artifact. The typed result's
/// query binding and provider snapshot are recomputed from the embedded
/// baseline, so the consumed dictionary carries every field `estimate` does.
#[pyfunction]
#[pyo3(signature=(artifact, *, memory_bytes=None, cancel=None))]
fn consume_z_transport_sensitivity_artifact(
    py: Python<'_>,
    artifact: &[u8],
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Py<PyAny>> {
    let bytes = artifact.to_vec();
    let (wire, typed) = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        let wire =
            antecedent::ZTransportSensitivityArtifactWire::consume(&bytes, &ctx).map_err(error)?;
        let (diagram, functional, data, _request, _limits, _point) =
            antecedent_io::z_transport_artifact::ZTransportArtifactWire::reconstruct(
                &wire.baseline_artifact,
            )
            .map_err(error)?;
        let typed = antecedent_validate::z_transport_mechanism_sensitivity(
            &diagram,
            &functional,
            &data,
            wire.max_fraction,
            wire.decision_threshold,
            &ctx,
        )
        .map_err(error)?;
        Ok((wire, typed))
    })?;
    let value = serde_json::json!({
        "status":"available",
        "estimand": SENSITIVITY_ESTIMAND,
        "baseline":wire.baseline,
        "assumption_range":{"minimum":wire.assumption_range[0],"maximum":wire.assumption_range[1]},
        "delta_domain":[0.0,wire.max_fraction],
        "decision_threshold":wire.decision_threshold,
        "tipping_fraction":wire.tipping_fraction,
        "minimizing_outcome_by_stratum":wire.minimizing_outcome_by_stratum,
        "maximizing_outcome_by_stratum":wire.maximizing_outcome_by_stratum,
        "interval_interpretation":wire.interval_interpretation,
        "method":wire.method,
        "baseline_binding":{"artifact_digest":wire.baseline_artifact_digest,"query":typed.query_binding,"source_regime":wire.source_regime,"provider_snapshot":typed.provider_snapshot,"provider_snapshots":wire.provider_snapshots}
    });
    to_py_json(py, &value)
}

#[pyfunction]
fn replay_z_transport_proposal(artifact: &[u8]) -> PyResult<()> {
    let wire: antecedent_design::ZTransportProposalWire =
        serde_json::from_slice(artifact).map_err(serialization_error)?;
    wire.replay().map_err(error)?;
    Ok(())
}

/// Recheck a portable failure snapshot: the status is re-derived from the
/// embedded graph, query and catalog, so an edited status never survives.
#[pyfunction]
fn consume_z_transport_failure_snapshot(py: Python<'_>, artifact: &[u8]) -> PyResult<Py<PyAny>> {
    let wire: ZTransportFailureSnapshotWire =
        serde_json::from_slice(artifact).map_err(serialization_error)?;
    ZTransportFailureSnapshot::from_wire(&wire).map_err(error)?;
    let value = serde_json::json!({
        "version": wire.version,
        "status": serde_json::to_value(&wire.status).map_err(serialization_error)?,
        "obligations": wire.obligations,
        "z_obstruction": wire.z_obstruction,
    });
    to_py_json(py, &value)
}

#[pyfunction]
#[pyo3(signature=(graph, selections, source, target, outcomes, treatments, controllable, experiment_assignment, *, max_steps=100_000, max_depth=256, max_support_rows=1_000_000, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn identify_z_transport_stage(
    py: Python<'_>,
    graph: PyRef<'_, Admg>,
    selections: Vec<String>,
    source: String,
    target: String,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    controllable: Vec<String>,
    experiment_assignment: BTreeMap<String, f64>,
    max_steps: usize,
    max_depth: usize,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ZTransportStage> {
    let coordinates = |variables: Vec<String>| -> PyResult<Arc<[VariableId]>> {
        variables
            .iter()
            .map(|name| resolve(&graph.names, name))
            .collect::<PyResult<Vec<_>>>()
            .map(Arc::from)
    };
    let query = ZTransportQuery {
        outcomes: coordinates(outcomes)?,
        treatments: coordinates(treatments)?,
        controllable: coordinates(controllable)?,
        experiment_assignment: intervention_assignments(&graph.names, experiment_assignment)?,
        source: source.into(),
        target: target.into(),
    };
    let diagram =
        SelectionDiagram::try_new(graph.aligned_to_names(&graph.names)?, coordinates(selections)?)
            .map_err(error)?;
    let named_graph = Admg { admg: graph.admg.clone(), names: graph.names.clone() };
    let limits = StageLimits { max_steps, max_depth, max_support_rows, memory_bytes };
    // The surrogate search is bounded by construction (12 observed, 4 controllable
    // variables) and takes no context; the limits are retained for every catalog
    // decision, preparation and proposal made from this stage.
    let result = crate::detach_catch(py, move || {
        let _ctx = execution_context(0, memory_bytes, cancel);
        identify_z_transport_surrogate(&diagram, &query)
            .map(|result| (result, diagram, query))
            .map_err(error)
    })?;
    let (result, diagram, query) = result;
    Ok(ZTransportStage { result, graph: named_graph, diagram, query, limits })
}

fn intervention_assignments(
    names: &[String],
    assignment: BTreeMap<String, f64>,
) -> PyResult<Arc<[antecedent_core::InterventionAssignment]>> {
    assignment
        .into_iter()
        .map(|(name, value)| {
            Ok(antecedent_core::InterventionAssignment {
                variable: resolve(names, &name)?,
                value: Value::f64(value),
            })
        })
        .collect::<PyResult<Vec<_>>>()
        .map(Arc::from)
}

/// Search two restricted sources separately on one shared graph.
///
/// `sources` are `(population, controllable, experiment_assignment, selections)`
/// in the order of `catalogs`. The result names the identifying source, both
/// line-11 obstructions, or the refusal to combine factors across sources.
#[pyfunction]
#[pyo3(signature=(graph, target, outcomes, treatments, sources, catalogs, *, max_steps=100_000, max_depth=256, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn decide_two_source_z_transport_stage(
    py: Python<'_>,
    graph: PyRef<'_, Admg>,
    target: String,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    sources: Vec<(String, Vec<String>, BTreeMap<String, f64>, Vec<String>)>,
    catalogs: Vec<Bound<'_, PyAny>>,
    max_steps: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Py<PyAny>> {
    if sources.len() != 2 || catalogs.len() != 2 {
        return Err(crate::value_err(
            "two-source z-transport takes exactly two sources and two catalogs",
        ));
    }
    let coordinates = |variables: &[String]| -> PyResult<Arc<[VariableId]>> {
        variables
            .iter()
            .map(|name| resolve(&graph.names, name))
            .collect::<PyResult<Vec<_>>>()
            .map(Arc::from)
    };
    let named = Admg { admg: graph.admg.clone(), names: graph.names.clone() };
    let mut specs = Vec::with_capacity(2);
    for (population, controllable, assignment, selections) in sources {
        specs.push(ZTransportSourceSpec {
            population: population.into(),
            controllable: coordinates(&controllable)?,
            experiment_assignment: intervention_assignments(&graph.names, assignment)?,
            selection_targets: coordinates(&selections)?,
        });
    }
    let parsed = catalogs
        .iter()
        .map(|catalog| parse_catalog(catalog, &named))
        .collect::<PyResult<Vec<_>>>()?;
    let query = TwoSourceZTransportQuery {
        outcomes: coordinates(&outcomes)?,
        treatments: coordinates(&treatments)?,
        target: target.into(),
        sources: [specs[0].clone(), specs[1].clone()],
    };
    let shared = graph.aligned_to_names(&graph.names)?;
    let value = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        let decision = decide_two_source_z_transport(
            &shared,
            &query,
            [&parsed[0], &parsed[1]],
            SidLimits { steps: max_steps, depth: max_depth },
            &ctx,
        )
        .map_err(error)?;
        Ok(match decision {
            TwoSourceZTransportDecision::Identified { source, derivation } => {
                let index = usize::from(query.sources[1].population == source);
                serde_json::json!({
                    "outcome": "identified",
                    "source": source.as_ref(),
                    "proof": derivation.to_record(),
                    "inspection": derivation.inspect_proof(&parsed[index]),
                })
            }
            TwoSourceZTransportDecision::ProvenNonTransportable { obstructions } => {
                serde_json::json!({
                    "outcome": "proven_non_transportable",
                    "obstructions": obstructions.iter().map(ZTransportObstruction::to_record).collect::<Vec<_>>(),
                    "sources": query.sources.iter().map(|s| s.population.as_ref()).collect::<Vec<_>>(),
                })
            }
            TwoSourceZTransportDecision::NotCertified { reason } => serde_json::json!({
                "outcome": "not_certified",
                "reason": reason,
            }),
        })
    })?;
    to_py_json(py, &value)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ZTransportStage>()?;
    module.add_class::<PreparedZTransportStage>()?;
    module.add_class::<ZTransportProposalStage>()?;
    module.add_function(wrap_pyfunction!(identify_z_transport_stage, module)?)?;
    module.add_function(wrap_pyfunction!(decide_two_source_z_transport_stage, module)?)?;
    module.add_function(wrap_pyfunction!(consume_z_transport_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(consume_z_transport_sensitivity_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(consume_z_transport_failure_snapshot, module)?)?;
    module.add_function(wrap_pyfunction!(replay_z_transport_proposal, module)?)?;
    Ok(())
}
