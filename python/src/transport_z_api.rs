//! Python bindings for bounded single-source point-only z-transport.
use crate::{graphs::Admg, transport_interference_api::parse_catalog};
use antecedent_core::{EvidenceCatalogDelta, ExecutionContext, Value, VariableId};
use antecedent_design::{
    CandidateDesign, DesignCost, ExperimentPlan, MeasurementPlan, TransportEvidenceCandidate,
    ZTransportCandidateOutcome, plan_z_transport_evidence, snapshot_z_transport_failure,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment, LawTolerance,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    SidLimits, ZTransportDecision, ZTransportQuery, ZTransportResult, bind_z_transport_catalog,
    decide_z_transport_with_catalog, identify_z_transport_surrogate,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::collections::BTreeMap;
use std::sync::Arc;

fn error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn resolve(names: &[String], name: &str) -> PyResult<VariableId> {
    let index = names
        .iter()
        .position(|candidate| candidate == name)
        .ok_or_else(|| error(format!("unknown variable {name}")))?;
    Ok(VariableId::from_raw(u32::try_from(index).map_err(error)?))
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
                return Err(error("catalog changed an existing regime contract"));
            }
            existing.id
        } else {
            while used.contains(&next) {
                next = next.checked_add(1).ok_or_else(|| error("regime IDs exhausted"))?;
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
            .ok_or_else(|| error("catalog binding references an unknown regime"))?;
    }
    aligned.validate().map_err(error)?;
    Ok(aligned)
}

#[pyclass(skip_from_py_object)]
struct ZTransportStage {
    result: ZTransportResult,
    graph: Admg,
    diagram: SelectionDiagram,
    query: ZTransportQuery,
}

#[pymethods]
impl ZTransportStage {
    /// Decide the bounded theorem against the actual complete catalog.
    /// A missing experiment is reported separately from a checked obstruction.
    fn decide(&self, py: Python<'_>, catalog: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let catalog = parse_catalog(catalog, &self.graph)?;
        let decision = decide_z_transport_with_catalog(
            &self.diagram,
            &self.query,
            &catalog,
            SidLimits::default(),
            &ExecutionContext::for_tests(0),
        )
        .map_err(error)?;
        let value = match decision {
            ZTransportDecision::Identified(proof) => serde_json::json!({
                "outcome": "identified",
                "proof": proof.to_record(),
                "inspection": proof.inspect_proof(&catalog),
            }),
            ZTransportDecision::ProvenNonTransportable(obstruction) => serde_json::json!({
                "outcome": "proven_non_transportable",
                "obstruction": obstruction.to_record(),
            }),
            ZTransportDecision::MissingEvidence { missing } => serde_json::json!({
                "outcome": "missing_evidence",
                "reason": format!("{missing:?}"),
            }),
            ZTransportDecision::NotCertified { reason } => serde_json::json!({
                "outcome": "not_certified",
                "reason": reason,
            }),
        };
        Ok(py.import("json")?.call_method1("loads", (value.to_string(),))?.unbind())
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
        let bytes = serde_json::to_vec(&wire).map_err(error)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
    }

    /// Inspect every checked rule and source-factor obligation against a catalog.
    fn inspect_proof(&self, py: Python<'_>, catalog: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let ZTransportResult::Identified(proof) = &self.result else {
            return Err(error(format!(
                "zTR refused: {}",
                self.reason().unwrap_or("not certified")
            )));
        };
        let catalog = parse_catalog(catalog, &self.graph)?;
        let inspection = proof.inspect_proof(&catalog);
        let json = serde_json::to_string(&inspection).map_err(error)?;
        Ok(py.import("json")?.call_method1("loads", (json,))?.unbind())
    }

    #[pyo3(signature=(catalog, laws, assignments, *, max_operations=10_000_000, max_depth=256, max_support_rows=1_000_000, memory_bytes=None))]
    fn prepare_exact(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        laws: &Bound<'_, PyAny>,
        assignments: BTreeMap<String, f64>,
        max_operations: usize,
        max_depth: usize,
        max_support_rows: usize,
        memory_bytes: Option<u64>,
    ) -> PyResult<PreparedZTransportStage> {
        let ZTransportResult::Identified(proof) = &self.result else {
            return Err(error(format!(
                "zTR refused: {}",
                self.reason().unwrap_or("not certified")
            )));
        };
        let catalog = parse_catalog(catalog, &self.graph)?;
        let data = parse_z_data(laws, &catalog, &self.graph, max_support_rows)?;
        let request = Assignment::from_pairs(
            assignments
                .into_iter()
                .map(|(name, value)| Ok((resolve(&self.graph.names, &name)?, Value::f64(value))))
                .collect::<PyResult<Vec<_>>>()?,
        );
        let diagram = self.diagram.clone();
        let query = self.query.clone();
        let proof = proof.as_ref().clone();
        let named_graph = self.graph.clone();
        crate::detach_catch(py, move || {
            let mut ctx = ExecutionContext::production_default(0);
            ctx.memory.hard_limit_bytes = memory_bytes;
            let functional =
                bind_z_transport_catalog(&diagram, &query, &proof, &catalog).map_err(error)?;
            let sensitivity_diagram = diagram.clone();
            let inner = antecedent::StudyBuilder::z_transport(
                diagram,
                functional,
                data,
                request,
                ExactEvaluationLimits { operations: max_operations, depth: max_depth },
                &ctx,
            )
            .map_err(error)?;
            Ok(PreparedZTransportStage {
                inner,
                graph: named_graph,
                diagram: sensitivity_diagram,
                catalog,
                last: None,
                max_support_rows,
                memory_bytes,
            })
        })
    }

    /// Prepare an empirical plugin table only when every law has counts.
    #[pyo3(signature=(catalog, laws, assignments, *, max_operations=10_000_000, max_depth=256, max_support_rows=1_000_000, memory_bytes=None))]
    fn prepare_empirical(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        laws: &Bound<'_, PyAny>,
        assignments: BTreeMap<String, f64>,
        max_operations: usize,
        max_depth: usize,
        max_support_rows: usize,
        memory_bytes: Option<u64>,
    ) -> PyResult<PreparedZTransportStage> {
        let ZTransportResult::Identified(proof) = &self.result else {
            return Err(error(format!(
                "zTR refused: {}",
                self.reason().unwrap_or("not certified")
            )));
        };
        let catalog = parse_catalog(catalog, &self.graph)?;
        let data = parse_z_data(laws, &catalog, &self.graph, max_support_rows)?;
        if data.laws().iter().any(|law| law.empirical_counts().is_none()) {
            return Err(error("z_transport.empirical_counts_required"));
        }
        let request = Assignment::from_pairs(
            assignments
                .into_iter()
                .map(|(name, value)| Ok((resolve(&self.graph.names, &name)?, Value::f64(value))))
                .collect::<PyResult<Vec<_>>>()?,
        );
        let diagram = self.diagram.clone();
        let query = self.query.clone();
        let proof = proof.as_ref().clone();
        let named_graph = self.graph.clone();
        crate::detach_catch(py, move || {
            let mut ctx = ExecutionContext::production_default(0);
            ctx.memory.hard_limit_bytes = memory_bytes;
            let functional =
                bind_z_transport_catalog(&diagram, &query, &proof, &catalog).map_err(error)?;
            let sensitivity_diagram = diagram.clone();
            let inner = antecedent::StudyBuilder::z_transport_empirical(
                diagram,
                functional,
                data,
                request,
                ExactEvaluationLimits { operations: max_operations, depth: max_depth },
                &ctx,
            )
            .map_err(error)?;
            Ok(PreparedZTransportStage {
                inner,
                graph: named_graph,
                diagram: sensitivity_diagram,
                catalog,
                last: None,
                max_support_rows,
                memory_bytes,
            })
        })
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
}

#[pyclass(skip_from_py_object)]
struct ZTransportProposalStage {
    inner: antecedent_design::ZTransportProposal,
    graph: Admg,
    diagram: SelectionDiagram,
    query: ZTransportQuery,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
}

#[pymethods]
impl ZTransportStage {
    /// Plan proposed studies and return assessment JSON plus verified proposal objects.
    /// Each candidate has `id`, proposed-only `catalog`, `design_kind`, `targets`,
    /// `measured`, `cost`, `sample_budget`, `recruitment_sampling`, and
    /// `feasibility_constraints` attributes.
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
            let wire: antecedent_design::ZTransportFailureSnapshotWire =
                serde_json::from_slice(&bytes).map_err(error)?;
            let supplied =
                antecedent_design::ZTransportFailureSnapshot::from_wire(&wire).map_err(error)?;
            let expected_wire = expected_snapshot.to_wire().map_err(error)?;
            let supplied_json = serde_json::to_vec(&wire).map_err(error)?;
            let expected_json = serde_json::to_vec(&expected_wire).map_err(error)?;
            if supplied_json != expected_json {
                return Err(error("failure snapshot does not match this stage and catalog"));
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
                return Err(error("candidate catalog must preserve the failure catalog"));
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
            let tag: u64 = item.getattr("tag").and_then(|v| v.extract()).unwrap_or(0);
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
                _ => return Err(error("design_kind must be 'intervene' or 'measure'")),
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
                            max_support_rows: 1_000_000,
                            memory_bytes: None,
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
            "ranked_sufficient":plan.ranked_sufficient.iter().map(|id| id.as_ref()).collect::<Vec<_>>()})
        .to_string();
        Ok((report, proposals))
    }
}

#[pymethods]
impl ZTransportProposalStage {
    /// Serialize the frozen failure, candidate delta, and checked derivation.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let wire = self.inner.to_wire().map_err(error)?;
        let bytes = serde_json::to_vec(&wire).map_err(error)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
    }

    /// Recheck that the original snapshot and hypothetical proof still match.
    fn replay(&self) -> PyResult<()> {
        self.inner.replay().map_err(error)
    }

    /// Receive matching study results, reidentify against the actual catalog, and prepare.
    #[pyo3(signature=(catalog, laws, assignments, provider_snapshot, *, empirical=false,
        max_operations=10_000_000, max_depth=256, memory_bytes=None))]
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
        memory_bytes: Option<u64>,
    ) -> PyResult<PreparedZTransportStage> {
        let parsed_catalog = parse_catalog(catalog, &self.graph)?;
        let proposal_wire = self.inner.to_wire().map_err(error)?;
        let base_catalog = proposal_wire.snapshot.catalog.to_catalog().map_err(error)?;
        let catalog = align_catalog_regime_ids(&base_catalog, &parsed_catalog)?;
        let _arrival = self.inner.receive(&catalog, provider_snapshot).map_err(error)?;
        let data = parse_z_data(laws, &catalog, &self.graph, self.max_support_rows)?;
        if empirical && data.laws().iter().any(|law| law.empirical_counts().is_none()) {
            return Err(error("z_transport.empirical_counts_required"));
        }
        let request = Assignment::from_pairs(
            assignments
                .into_iter()
                .map(|(name, value)| Ok((resolve(&self.graph.names, &name)?, Value::f64(value))))
                .collect::<PyResult<Vec<_>>>()?,
        );
        let diagram = self.diagram.clone();
        let query = self.query.clone();
        let named_graph = self.graph.clone();
        let memory = memory_bytes.or(self.memory_bytes);
        crate::detach_catch(py, move || {
            let mut ctx = ExecutionContext::production_default(0);
            ctx.memory.hard_limit_bytes = memory;
            let proof = match identify_z_transport_surrogate(&diagram, &query).map_err(error)? {
                ZTransportResult::Identified(proof) => proof,
                ZTransportResult::NotCertified { reason } => return Err(error(reason)),
            };
            let functional =
                bind_z_transport_catalog(&diagram, &query, &proof, &catalog).map_err(error)?;
            let inner = if empirical {
                antecedent::StudyBuilder::z_transport_empirical(
                    diagram.clone(),
                    functional,
                    data,
                    request,
                    ExactEvaluationLimits { operations: max_operations, depth: max_depth },
                    &ctx,
                )
                .map_err(error)?
            } else {
                antecedent::StudyBuilder::z_transport(
                    diagram.clone(),
                    functional,
                    data,
                    request,
                    ExactEvaluationLimits { operations: max_operations, depth: max_depth },
                    &ctx,
                )
                .map_err(error)?
            };
            Ok(PreparedZTransportStage {
                inner,
                graph: named_graph,
                diagram,
                catalog,
                last: None,
                max_support_rows: 1_000_000,
                memory_bytes: memory,
            })
        })
    }
}

#[pymethods]
impl PreparedZTransportStage {
    #[pyo3(signature=(max_fraction, decision_threshold=None))]
    fn mechanism_sensitivity(
        &self,
        py: Python<'_>,
        max_fraction: f64,
        decision_threshold: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let ctx = ExecutionContext::production_default(0);
        let result = antecedent_validate::z_transport_mechanism_sensitivity(
            &self.diagram,
            self.inner.functional(),
            self.inner.data(),
            max_fraction,
            decision_threshold,
            &ctx,
        )
        .map_err(error)?;
        let point_result =
            self.last.as_ref().ok_or_else(|| error("estimate before sensitivity analysis"))?;
        let baseline_bytes = point_result.export(&self.inner).map_err(error)?;
        let baseline_digest = antecedent_io::identity::payload_digest(
            "z_transport_sensitivity_baseline",
            &baseline_bytes,
        )
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
        let provider_snapshots = self
            .inner
            .data()
            .laws()
            .iter()
            .map(|law| law.snapshot_identity().to_owned())
            .collect::<Vec<_>>();
        let json = serde_json::to_string(&serde_json::json!({
            "status":"available",
            "estimand":"target active-minus-control mean outcome response",
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
        })).map_err(error)?;
        Ok(py.import("json")?.call_method1("loads", (json,))?.unbind())
    }

    #[pyo3(signature=(estimator=None, posterior_draws=None))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        estimator: Option<String>,
        posterior_draws: Option<u32>,
    ) -> PyResult<String> {
        let provider = match estimator.as_deref() {
            None => None,
            Some("empirical_support_bayesian_bootstrap") => {
                Some(antecedent_estimate::BayesianTransportLawProvider::EmpiricalSupport)
            }
            Some("state_space_dirichlet") => {
                Some(antecedent_estimate::BayesianTransportLawProvider::DeclaredStateSpaceDirichlet)
            }
            Some(name) => return Err(error(format!("unknown z-transport provider {name}"))),
        };
        if provider.is_none() && posterior_draws.is_some() {
            return Err(error("posterior_draws requires a Bayesian z-transport provider"));
        }
        let draws = posterior_draws.unwrap_or(199);
        let inner = self.inner.clone();
        let memory = self.memory_bytes;
        let result = crate::detach_catch(py, move || {
            let mut ctx = ExecutionContext::production_default(0);
            ctx.memory.hard_limit_bytes = memory;
            match provider {
                Some(provider) => inner.estimate_bayesian(provider, draws, &ctx).map_err(error),
                None => inner.estimate(&ctx).map_err(error),
            }
        })?;
        let names = &self.graph.names;
        let query = self.inner.functional().derivation().query();
        let distribution = result.distribution();
        let interval = if matches!(
            result.interval_type(),
            antecedent_estimate::PERCENTILE_BOOTSTRAP | antecedent_estimate::POSTERIOR_EQUAL_TAIL
        ) {
            serde_json::json!({
                "available": true,
                "method": result.interval_type(),
                "reason": result.interval_reason(),
                "coverage_target": result.coverage_target(),
                "mean_intervals": result.mean_intervals().iter().map(|(variable, lower, upper)| serde_json::json!({
                    "outcome": &names[variable.as_usize()],
                    "lower": lower,
                    "upper": upper,
                })).collect::<Vec<_>>(),
            })
        } else {
            serde_json::json!({
                "available": false,
                "reason": result.interval_reason(),
            })
        };
        let payload = serde_json::json!({
            "status":"available",
            "scope":"single_source_z_transport_cited_joints_sound_incomplete",
            "outcomes":query.outcomes.iter().map(|v| &names[v.as_usize()]).collect::<Vec<_>>(),
            "atoms":distribution.atoms.iter().map(|row| row.iter().map(Value::as_f64).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "probabilities":distribution.probabilities.as_ref(),
            "factor_support":distribution.support.iter().map(|s| serde_json::json!({
                "expression":s.expression.raw(), "status":s.status, "denominator":s.denominator,
                "assignment":s.assignment.iter().map(|(v,x)| (names[v.as_usize()].as_str(), x.as_f64())).collect::<BTreeMap<_,_>>()
            })).collect::<Vec<_>>(),
            "interval": interval
        }).to_string();
        self.last = Some(result);
        Ok(payload)
    }

    fn refresh(&mut self, py: Python<'_>, laws: &Bound<'_, PyAny>) -> PyResult<()> {
        let data = parse_z_data(laws, &self.catalog, &self.graph, self.max_support_rows)?;
        let inner = self.inner.clone();
        let memory = self.memory_bytes;
        self.inner = crate::detach_catch(py, move || {
            let mut ctx = ExecutionContext::production_default(0);
            ctx.memory.hard_limit_bytes = memory;
            inner.refresh(data, &ctx).map_err(error)
        })?;
        self.last = None;
        Ok(())
    }

    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let result = self
            .last
            .as_ref()
            .ok_or_else(|| error("estimate before exporting a z-transport artifact"))?;
        let bytes = result.export(&self.inner).map_err(error)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
    }

    #[pyo3(signature=(max_fraction, decision_threshold=None))]
    fn export_sensitivity(
        &self,
        py: Python<'_>,
        max_fraction: f64,
        decision_threshold: Option<f64>,
    ) -> PyResult<Py<pyo3::types::PyBytes>> {
        let result = self
            .last
            .as_ref()
            .ok_or_else(|| error("estimate before exporting sensitivity artifact"))?;
        let baseline = result.export(&self.inner).map_err(error)?;
        let mut ctx = ExecutionContext::production_default(0);
        ctx.memory.hard_limit_bytes = self.memory_bytes;
        let wire = antecedent::ZTransportSensitivityArtifactWire::checked(
            baseline,
            max_fraction,
            decision_threshold,
            &ctx,
        )
        .map_err(error)?;
        let bytes = wire.export().map_err(error)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
    }

    #[getter]
    fn interval_type(&self) -> &'static str {
        self.last.as_ref().map_or("no_interval_reported", |result| result.interval_type())
    }
}

#[pyfunction]
fn consume_z_transport_artifact(py: Python<'_>, artifact: &[u8]) -> PyResult<String> {
    let bytes = artifact.to_vec();
    crate::detach_catch(py, move || {
        let ctx = ExecutionContext::production_default(0);
        let (_diagram, result) =
            antecedent::consume_z_transport_artifact(&bytes, &ctx).map_err(error)?;
        let distribution = result.distribution();
        Ok(serde_json::json!({
            "status":"available",
            "outcomes":distribution.outcomes.iter().map(|v|v.raw()).collect::<Vec<_>>(),
            "atoms":distribution.atoms.iter().map(|a|a.iter().map(Value::as_f64).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "probabilities":distribution.probabilities.to_vec(),
            "interval":{"available":false,"reason":"no_interval_reported"}
        }).to_string())
    })
}

#[pyfunction]
fn consume_z_transport_sensitivity_artifact(
    py: Python<'_>,
    artifact: &[u8],
) -> PyResult<Py<PyAny>> {
    let bytes = artifact.to_vec();
    let wire = crate::detach_catch(py, move || {
        let ctx = ExecutionContext::production_default(0);
        antecedent::ZTransportSensitivityArtifactWire::consume(&bytes, &ctx).map_err(error)
    })?;
    let json = serde_json::to_string(&serde_json::json!({
        "status":"available",
        "baseline":wire.baseline,
        "assumption_range":{"minimum":wire.assumption_range[0],"maximum":wire.assumption_range[1]},
        "delta_domain":[0.0,wire.max_fraction],
        "decision_threshold":wire.decision_threshold,
        "tipping_fraction":wire.tipping_fraction,
        "minimizing_outcome_by_stratum":wire.minimizing_outcome_by_stratum,
        "maximizing_outcome_by_stratum":wire.maximizing_outcome_by_stratum,
        "interval_interpretation":wire.interval_interpretation,
        "method":wire.method,
        "baseline_binding":{"artifact_digest":wire.baseline_artifact_digest,"provider_snapshots":wire.provider_snapshots,"source_regime":wire.source_regime}
    })).map_err(error)?;
    Ok(py.import("json")?.call_method1("loads", (json,))?.unbind())
}

#[pyfunction]
fn replay_z_transport_proposal(artifact: &[u8]) -> PyResult<()> {
    let wire: antecedent_design::ZTransportProposalWire =
        serde_json::from_slice(artifact).map_err(error)?;
    wire.replay().map_err(error)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature=(graph, selections, source, target, outcomes, treatments, controllable, experiment_assignment))]
#[allow(clippy::too_many_arguments)]
fn identify_z_transport_stage(
    graph: PyRef<'_, Admg>,
    selections: Vec<String>,
    source: String,
    target: String,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    controllable: Vec<String>,
    experiment_assignment: BTreeMap<String, f64>,
) -> PyResult<ZTransportStage> {
    let coordinates = |variables: Vec<String>| -> PyResult<Arc<[VariableId]>> {
        variables
            .iter()
            .map(|name| resolve(&graph.names, name))
            .collect::<PyResult<Vec<_>>>()
            .map(Arc::from)
    };
    let experiment_assignment = experiment_assignment
        .into_iter()
        .map(|(name, value)| {
            Ok(antecedent_core::InterventionAssignment {
                variable: resolve(&graph.names, &name)?,
                value: Value::f64(value),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let query = ZTransportQuery {
        outcomes: coordinates(outcomes)?,
        treatments: coordinates(treatments)?,
        controllable: coordinates(controllable)?,
        experiment_assignment: experiment_assignment.into(),
        source: source.into(),
        target: target.into(),
    };
    let diagram =
        SelectionDiagram::try_new(graph.aligned_to_names(&graph.names)?, coordinates(selections)?)
            .map_err(error)?;
    let named_graph = Admg { admg: graph.admg.clone(), names: graph.names.clone() };
    let result = identify_z_transport_surrogate(
        &diagram, &query, // Positive paths carry checked recursive rule traces.
    )
    .map_err(error)?;
    Ok(ZTransportStage { result, graph: named_graph, diagram, query })
}

fn parse_z_data(
    laws: &Bound<'_, PyAny>,
    catalog: &antecedent_core::EvidenceCatalog,
    graph: &Admg,
    max_support_rows: usize,
) -> PyResult<ExactTransportData> {
    let mut tables = Vec::new();
    for table in laws.try_iter()? {
        let table = table?;
        let population: String = table.getattr("population")?.extract()?;
        let label: String = table.getattr("regime")?.extract()?;
        let regime = catalog
            .regimes
            .iter()
            .find(|r| {
                r.label.as_deref() == Some(label.as_str()) && r.population.as_ref() == population
            })
            .ok_or_else(|| error("exact zTR law names an unknown population/regime"))?;
        let axes: Vec<(String, Vec<f64>)> = table.getattr("axes")?.extract()?;
        let axes = axes
            .into_iter()
            .map(|(name, values)| {
                Ok(DiscreteAxis {
                    variable: resolve(&graph.names, &name)?,
                    values: values.into_iter().map(Value::f64).collect(),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let interventions: Vec<(String, f64)> = table.getattr("interventions")?.extract()?;
        let interventions = interventions
            .into_iter()
            .map(|(name, value)| {
                Ok(InterventionAssignment {
                    variable: resolve(&graph.names, &name)?,
                    value: Value::f64(value),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let probabilities: Vec<f64> = table.getattr("probabilities")?.extract()?;
        let snapshot: String = table.getattr("snapshot_identity")?.extract()?;
        let absolute: f64 = table.getattr("absolute_tolerance")?.extract()?;
        let relative: f64 = table.getattr("relative_tolerance")?.extract()?;
        let mut law = ExactDiscreteLaw::try_new(
            population,
            regime.id,
            interventions,
            axes,
            probabilities,
            snapshot,
            LawTolerance { absolute, relative },
        )
        .map_err(error)?;
        if let Ok(counts) = table.getattr("empirical_counts") {
            if !counts.is_none() {
                law = law.with_empirical_counts(counts.extract()?).map_err(error)?;
            }
        }
        tables.push(law);
    }
    ExactTransportData::try_new(tables, max_support_rows).map_err(error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ZTransportStage>()?;
    module.add_class::<PreparedZTransportStage>()?;
    module.add_class::<ZTransportProposalStage>()?;
    module.add_function(wrap_pyfunction!(identify_z_transport_stage, module)?)?;
    module.add_function(wrap_pyfunction!(consume_z_transport_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(consume_z_transport_sensitivity_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(replay_z_transport_proposal, module)?)?;
    Ok(())
}
