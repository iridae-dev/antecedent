//! Native authority for retained exact and empirical response grids.
use crate::transport_common::{error, resolve};
use crate::{
    graphs::Admg, transport_exact_api::ClassicalTransportStage,
    transport_interference_api::parse_catalog,
};
use antecedent::analysis::{
    TransportGridData, TransportGridPoint, TransportGridQuery, TransportGridResult,
    TransportGridState,
};
use antecedent_core::{ExecutionContext, Value};
use antecedent_estimate::EmpiricalTableOptions;
use antecedent_expr::{Assignment, ExactEvaluationLimits};
use pyo3::prelude::*;
use std::collections::BTreeMap;
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct PreparedTransportGridStage {
    inner: antecedent::PreparedStudy<TransportGridState>,
    names: Vec<String>,
    last: Option<TransportGridResult>,
    options: Option<EmpiricalTableOptions>,
    memory: Option<u64>,
    max_support_rows: usize,
    /// Stream seed of later estimates: the caller's `seed` at preparation, 0 for a consumed artifact
    /// (its result was computed before this stage existed).
    seed: u64,
}
impl PreparedTransportGridStage {
    fn ctx(&self, cancel: Option<crate::PyCancellationToken>) -> ExecutionContext {
        let mut ctx = ExecutionContext::production_default(self.seed);
        ctx.memory.hard_limit_bytes = self.memory;
        crate::apply_cancel(&mut ctx, cancel);
        ctx
    }
    fn data(&self, data: &Bound<'_, PyAny>) -> PyResult<TransportGridData> {
        let graph = Admg {
            admg: self.inner.query().diagram.causal_graph().clone(),
            names: self.names.clone(),
        };
        let catalog = self.inner.query().functional.catalog();
        if let Some(options) = self.options {
            Ok(TransportGridData::Statistical(
                crate::transport_statistical_api::parse_statistical_input(
                    data,
                    catalog,
                    &graph,
                    self.max_support_rows,
                )?,
                options,
            ))
        } else {
            Ok(TransportGridData::Exact(crate::transport_exact_api::parse_exact_data(
                &data.getattr("laws")?,
                catalog,
                &graph,
                self.max_support_rows,
            )?))
        }
    }
    #[allow(clippy::single_match_else)] // The remaining two variants share a complete distribution.
    fn payload(&self, result: &TransportGridResult) -> String {
        let points:Vec<_>=result.points().iter().zip(&self.inner.query().at).map(|(point,at)|{
            let coordinates:BTreeMap<_,_>=at.entries().iter().map(|(v,x)|(self.names[v.as_usize()].as_str(),x.as_f64())).collect();
            match point {
                TransportGridPoint::Unavailable(failure)=>serde_json::json!({"at":coordinates,"status":failure.kind,"detail":failure.detail,"factor_diagnostic":antecedent_io::transport_grid_wire::TransportGridFailureWire::from_failure(failure)}),
                _=>{
                    let distribution=point.distribution().expect("executable point");
                    let query=self.inner.query().functional.derivation().query();
                    let means:BTreeMap<_,_>=query.outcomes.iter().map(|v|(self.names[v.as_usize()].as_str(),distribution.mean(*v).ok())).collect();
                    let uncertainty=match point {TransportGridPoint::Statistical(_,r)=>{
                        let est=r.estimate();serde_json::json!({"available":est.uncertainty.is_some()&&est.uncertainty_reason.is_none(),"reason":est.uncertainty_reason.as_deref(),"scope":"pointwise","calibration_status":"not_bound_to_this_execution","replicate_ids":est.replicate_ids.as_deref(),"atom_intervals":est.atom_intervals.as_deref(),"mean_intervals":est.mean_intervals.as_ref().map(|rows|rows.iter().map(|(v,l,u)|(self.names[v.as_usize()].as_str(),l,u)).collect::<Vec<_>>()),"replicates_failed":est.uncertainty.as_ref().map(|row|row.replicates_failed)})
                    },_=>serde_json::json!({"available":false,"reason":"exact_supplied_law_no_sampling_uncertainty"})};
                    let support:Vec<_>=distribution.support.iter().map(|s|serde_json::json!({"expression":s.expression.raw(),"status":s.status,"denominator":s.denominator,"assignment":s.assignment.iter().map(|(v,x)|(self.names[v.as_usize()].as_str(),x.as_f64())).collect::<BTreeMap<_,_>>()})).collect();
                    serde_json::json!({"at":coordinates,"status":"available","outcomes":query.outcomes.iter().map(|v|self.names[v.as_usize()].as_str()).collect::<Vec<_>>(),"atoms":distribution.atoms.iter().map(|r|r.iter().map(Value::as_f64).collect::<Vec<_>>()).collect::<Vec<_>>(),"probabilities":distribution.probabilities.as_ref(),"means":means,"uncertainty":uncertainty,"factor_support":support})
                }
            }
        }).collect();
        serde_json::json!({"execution_id":result.identity(),"points":points,"interval_scope":"pointwise","simultaneous_bands":false}).to_string()
    }
    /// Native four-slot reasoning, mirroring [`TransportGridResult::reasoning`] before any
    /// point has been executed (identification is a structural property of the accepted
    /// proof, not of an execution; the other three slots stay unavailable until then).
    fn reasoning(&self) -> antecedent_core::ReasoningView {
        self.last.as_ref().map_or_else(
            || {
                use antecedent_core::{
                    AssumptionSlot, AssumptionSource, AssumptionStatus, IdentificationSlot,
                    IdentificationStatus, ObligationKind, ObligationRecord, ObligationScope,
                    ReasoningView, SlotAvailability,
                };
                ReasoningView::new(
                    SlotAvailability::Available(IdentificationSlot::identified_singleton(
                        IdentificationStatus::NonparametricallyIdentified,
                    )),
                    SlotAvailability::unavailable("execution_specific"),
                    SlotAvailability::unavailable("execution_specific"),
                    SlotAvailability::Available(AssumptionSlot::new(vec![ObligationRecord::new(
                        "transport.population_selection_graph",
                        ObligationScope::Program,
                        AssumptionSource::UserDeclared,
                        ObligationKind::UserAssertion,
                        AssumptionStatus::Declared,
                        "The accepted graph and source-specific mechanism selections describe the declared populations.",
                    )])),
                )
            },
            TransportGridResult::reasoning,
        )
    }
}
#[pymethods]
impl PreparedTransportGridStage {
    fn freeze(&self) -> Self {
        self.clone()
    }
    #[pyo3(signature=(cancel=None))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let inner = self.inner.clone();
        let ctx = self.ctx(cancel);
        let result = crate::detach_catch(py, move || inner.estimate(&ctx).map_err(error))?;
        let payload = self.payload(&result);
        self.last = Some(result);
        Ok(payload)
    }
    #[pyo3(signature=(data,cancel=None))]
    fn refresh(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let data = self.data(data)?;
        let mut candidate = self.inner.clone();
        let ctx = self.ctx(cancel);
        let (candidate, result) = crate::detach_catch(py, move || {
            let result = candidate.refresh(data, &ctx).map_err(error)?;
            Ok((candidate, result))
        })?;
        self.inner = candidate;
        let payload = self.payload(&result);
        self.last = Some(result);
        Ok(payload)
    }
    #[pyo3(signature=(data,cancel=None))]
    fn replace_snapshot(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let input = self.data(data)?;
        let mut candidate = self.inner.clone();
        let ctx = self.ctx(cancel);
        let candidate = crate::detach_catch(py, move || {
            candidate.replace_snapshot(input, &ctx).map_err(error)?;
            Ok(candidate)
        })?;
        self.inner = candidate;
        self.last = None;
        Ok(())
    }
    fn last_result(&self) -> PyResult<String> {
        Ok(self.payload(self.last.as_ref().ok_or_else(|| error("transport.no_execution_claim"))?))
    }
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let bytes = self
            .last
            .as_ref()
            .ok_or_else(|| error("transport.no_execution_claim"))?
            .export()
            .map_err(error)?;
        let mut framed = b"ANTECEDENT-TRANSPORT-GRID\x01".to_vec();
        framed.extend(antecedent_io::to_cbor(&(self.names.clone(), bytes)).map_err(error)?);
        Ok(pyo3::types::PyBytes::new(py, &framed).unbind())
    }
    fn contrast(&self, left: usize, right: usize, outcome: &str) -> PyResult<String> {
        let grid = self.last.as_ref().ok_or_else(|| error("transport.no_execution_claim"))?;
        let contrast = grid.contrast(left, right, resolve(&self.names, outcome)?).map_err(error)?;
        Ok(serde_json::json!({"estimate":contrast.estimate,"interval":contrast.interval,"coverage_target":contrast.coverage_target,"replicates_ok":contrast.replicates_ok,"replicates_failed":contrast.replicates_failed,"reason":contrast.reason,"parent_grid":grid.identity(),"left":left,"right":right,"interval_scope":"pointwise","calibration_status":"not_bound_to_this_execution"}).to_string())
    }
    fn scalar_projection(&self, point: usize, outcome: &str) -> PyResult<String> {
        let result = self.last.as_ref().ok_or_else(|| error("transport.no_execution_claim"))?;
        let (value, receipt) =
            result.scalar_projection(point, resolve(&self.names, outcome)?).map_err(error)?;
        Ok(serde_json::json!({"value":value,"parent_grid":receipt.input.to_hex(),"equivalent_claim":false,"rule":receipt.rule.as_ref(),"omitted":receipt.omitted.iter().map(AsRef::as_ref).collect::<Vec<_>>(),"unavailable_operations":receipt.unavailable_operations.iter().map(AsRef::as_ref).collect::<Vec<_>>()}).to_string())
    }
    fn inspection_json(&self) -> String {
        let query = self.inner.query();
        let proof = query.functional.derivation();
        let points = self.last.as_ref().map(|result| {
            serde_json::from_str::<serde_json::Value>(&self.payload(result)).expect("JSON payload")
        });
        let arena = query.functional.arena();
        let factors: Vec<_> = arena.distribution_leaves(query.functional.root()).into_iter().map(|id| {
            let antecedent_expr::ExprNode::Distribution { variables, conditioned_on, intervention, population, regime, .. } = arena.node(id) else { unreachable!() };
            serde_json::json!({"expression": id.raw(), "population": arena.population(*population), "regime": regime.map(antecedent_core::RegimeId::raw), "variables": arena.var_set(*variables).iter().map(|v|self.names[v.as_usize()].as_str()).collect::<Vec<_>>(), "conditioned_on": arena.var_set(*conditioned_on).iter().map(|v|self.names[v.as_usize()].as_str()).collect::<Vec<_>>(), "interventions": arena.intervention_assignments(*intervention).iter().map(|a|self.names[a.variable.as_usize()].as_str()).collect::<Vec<_>>()})
        }).collect();
        let reasoning = self.reasoning();
        let identification_status =
            reasoning.identification.as_ref().map_or("unavailable", |slot| slot.status.as_str());
        let uncertainty_available = reasoning.uncertainty.is_available();
        serde_json::json!({
            "identification":{"available":reasoning.identification.is_available(),"summary":identification_status,"payload":{"formula":query.functional.arena().pretty(query.functional.root()).replace(":=NaN", ""),"theorem_scope":proof.evidence_setting(),"sources":proof.sources(),"rules":proof.rules(),"required_factors":factors,"required_bindings":query.functional.arena().leaf_bindings(query.functional.root()).iter().map(|b|serde_json::json!({"population":b.population.as_ref(),"regime":b.regime.map(antecedent_core::RegimeId::raw)})).collect::<Vec<_>>()}},
            "support":{"available":self.last.is_some(),"summary":"grid_local_support","payload":{"points":points,"population_positivity":"assumed_not_empirically_proven"}},
            "uncertainty":{"available":uncertainty_available,"summary":if uncertainty_available{"pointwise_bootstrap"}else{"unavailable"},"payload":{"calibration_status":"not_bound_to_this_execution","simultaneous_bands":false}},
            "assumptions":{"available":true,"summary":"declared_population_selections_and_provider_contract","payload":{"empirically_verified":false,"target":proof.query().target.as_ref(),"sources":proof.sources()}},
            "execution_id":self.last.as_ref().map(TransportGridResult::identity),"capabilities":{"storage":true,"proof_verification":true,"numerical_verification":true,"estimator_replay":self.inner.can_reestimate()},"evidence_contract":antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(query.functional.catalog()),"supported_operations":["estimate","replace_snapshot","refresh","export","mean","contrast","scalar_projection"],"formula":query.functional.arena().pretty(query.functional.root()).replace(":=NaN", "")
        }).to_string()
    }
    fn plan_summary(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("kernels".into(), format!("finite_transport_grid:{}", self.inner.query().at.len())),
            ("deterministic_reductions".into(), "true".into()),
        ])
    }
    fn preview_transform(&self, intent: &str) -> PyResult<BTreeMap<String, String>> {
        let intent = crate::prepared_api::parse_transform_intent(intent)?;
        Ok(crate::prepared_api::transform_report_map(
            &self.inner.preview_transform(intent).map_err(error)?,
        )
        .into_iter()
        .collect())
    }
}
#[pyfunction]
#[pyo3(signature=(stage,catalog,data,at,*,statistical=false,estimator=None,bootstrap=crate::transport_defaults::BOOTSTRAP,coverage_level=crate::transport_defaults::COVERAGE_LEVEL,seed=1,max_operations=10_000_000,max_depth=256,max_support_rows=1_000_000,memory_bytes=None,cancel=None))]
#[allow(clippy::too_many_arguments)]
fn prepare_transport_grid(
    py: Python<'_>,
    stage: PyRef<'_, ClassicalTransportStage>,
    catalog: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    at: Vec<BTreeMap<String, f64>>,
    statistical: bool,
    estimator: Option<&Bound<'_, PyAny>>,
    bootstrap: u32,
    coverage_level: f64,
    seed: u64,
    max_operations: usize,
    max_depth: usize,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PreparedTransportGridStage> {
    let proof = stage.identified()?;
    let catalog = parse_catalog(catalog, stage.graph())?;
    let options = statistical.then_some(EmpiricalTableOptions {
        estimator: crate::transport_statistical_api::parse_provider(estimator)?,
        bootstrap_replicates: bootstrap,
        posterior_draws: 199,
        coverage_level,
        max_joint_cells: max_support_rows,
    });
    let input = if let Some(options) = options {
        TransportGridData::Statistical(
            crate::transport_statistical_api::parse_statistical_input(
                data,
                &catalog,
                stage.graph(),
                max_support_rows,
            )?,
            options,
        )
    } else {
        TransportGridData::Exact(crate::transport_exact_api::parse_exact_data(
            &data.getattr("laws")?,
            &catalog,
            stage.graph(),
            max_support_rows,
        )?)
    };
    let requests = at
        .into_iter()
        .map(|at| {
            at.into_iter()
                .map(|(name, value)| Ok((resolve(&stage.graph().names, &name)?, Value::f64(value))))
                .collect::<PyResult<Vec<_>>>()
                .map(Assignment::from_pairs)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let diagram = stage.diagram();
    let names = stage.graph().names.clone();
    crate::detach_catch(py, move || {
        let mut ctx = ExecutionContext::production_default(seed);
        ctx.memory.hard_limit_bytes = memory_bytes;
        crate::apply_cancel(&mut ctx, cancel);
        let functional = match proof.bind_catalog_with_context(
            &catalog,
            antecedent_identify::SidLimits { steps: max_operations, depth: max_depth },
            &ctx,
        ) {
            Ok(f) => f,
            Err(_) => match proof
                .search_catalog(
                    &diagram,
                    &catalog,
                    antecedent_identify::SidLimits { steps: max_operations, depth: max_depth },
                    &ctx,
                )
                .map_err(error)?
            {
                antecedent_identify::CatalogTransportResult::Identified(f) => *f,
                other => {
                    return Err(crate::transport_common::catalog_search_refusal(&other, &names));
                }
            },
        };
        let inner = antecedent::StudyBuilder::transport_grid(
            TransportGridQuery {
                diagram,
                functional,
                at: requests,
                limits: ExactEvaluationLimits { operations: max_operations, depth: max_depth },
            },
            input,
            &ctx,
        )
        .map_err(error)?;
        Ok(PreparedTransportGridStage {
            inner,
            names,
            last: None,
            options,
            memory: memory_bytes,
            max_support_rows,
            seed,
        })
    })
}
#[pyfunction]
#[pyo3(signature=(artifact,*,max_operations=10_000_000,max_depth=256,memory_bytes=None,cancel=None))]
fn consume_transport_grid(
    py: Python<'_>,
    artifact: &[u8],
    max_operations: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PreparedTransportGridStage> {
    let bytes = artifact
        .strip_prefix(b"ANTECEDENT-TRANSPORT-GRID\x01")
        .ok_or_else(|| error("invalid grid framing"))?;
    if memory_bytes.is_some_and(|n| bytes.len() as u64 > n) {
        return Err(error("grid artifact memory budget"));
    }
    let (names, bytes): (Vec<String>, Vec<u8>) = antecedent_io::from_cbor(bytes).map_err(error)?;
    crate::detach_catch(py, move || {
        let mut ctx = ExecutionContext::production_default(0);
        ctx.memory.hard_limit_bytes = memory_bytes;
        crate::apply_cancel(&mut ctx, cancel);
        let (inner, result) = antecedent::PreparedStudy::<TransportGridState>::consume(
            &bytes,
            ExactEvaluationLimits { operations: max_operations, depth: max_depth },
            &ctx,
        )
        .map_err(error)?;
        if names.len() != inner.query().diagram.causal_graph().node_count()
            || names.iter().collect::<std::collections::BTreeSet<_>>().len() != names.len()
        {
            return Err(error("invalid grid coordinate names"));
        }
        crate::transport_exact_api::validate_artifact_names(
            &names,
            inner.query().diagram.causal_graph(),
        )?;
        let options = inner.statistical_options();
        Ok(PreparedTransportGridStage {
            inner,
            names,
            last: Some(result),
            options,
            memory: memory_bytes,
            max_support_rows: max_operations,
            seed: 0,
        })
    })
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PreparedTransportGridStage>()?;
    m.add_function(wrap_pyfunction!(prepare_transport_grid, m)?)?;
    m.add_function(wrap_pyfunction!(consume_transport_grid, m)?)?;
    Ok(())
}
