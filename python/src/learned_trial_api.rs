//! Learner-backed trial transport on the shared prepared lifecycle.
use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_estimate::{TrialAipwInput, TrialAipwOptions};
use pyo3::{prelude::*, types::PyBytes};
use std::sync::Arc;
fn err(e: impl crate::transport_common::TransportPyErr) -> PyErr {
    crate::transport_common::error(e)
}
fn parse_input(data: &Bound<'_, PyAny>, names: &[String]) -> PyResult<TrialAipwInput> {
    if data.hasattr("_buffers")? {
        return crate::transport_buffers::trial(data, names, false);
    }
    let wire: String = data.call_method1("_json", (names.to_vec(),))?.extract()?;
    serde_json::from_str(&wire).map_err(err)
}
// The Python result exposes these fields; per-row predictions and scores remain
// in the frozen native result and portable artifact, available through legacy methods.
fn trial_payload(result: &antecedent::LearnedTrialResult, compact: bool) -> PyResult<String> {
    let estimate = result.estimate();
    if !compact {
        return serde_json::to_string(estimate).map_err(err);
    }
    Ok(serde_json::json!({
        "estimate": estimate.estimate,
        "interval": estimate.interval,
        "uncertainty_reason": estimate.uncertainty_reason,
        "replicates": estimate.replicates,
        "failures": estimate.failures,
        "overlap": estimate.overlap,
        "diagnostics": estimate.diagnostics,
    })
    .to_string())
}
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct PreparedLearnedTrial {
    inner: Option<antecedent::PreparedStudy<antecedent::LearnedTrialState>>,
    names: Vec<String>,
    last: Option<antecedent::LearnedTrialResult>,
    seed: u64,
    memory_bytes: Option<u64>,
}
impl PreparedLearnedTrial {
    fn ctx(&self, cancel: Option<crate::PyCancellationToken>) -> ExecutionContext {
        let mut ctx = ExecutionContext::production_default(self.seed);
        ctx.set_memory_limit(self.memory_bytes);
        crate::apply_cancel(&mut ctx, cancel);
        crate::execution_api::apply(&mut ctx);
        ctx
    }
    fn estimate_payload(
        &mut self,
        py: Python<'_>,
        cancel: Option<crate::PyCancellationToken>,
        compact: bool,
    ) -> PyResult<String> {
        let ctx = self.ctx(cancel);
        let inner = self.inner.clone().ok_or_else(|| err("loaded claims cannot execute"))?;
        let result = crate::detach_catch(py, move || inner.estimate(&ctx).map_err(err))?;
        let payload = trial_payload(&result, compact)?;
        self.last = Some(result);
        Ok(payload)
    }
    /// Native four-slot reasoning: the executed result's own view once estimated, else
    /// the not-yet-executed view of the prepared identification (identification is a
    /// property of the accepted certificate, not of a fitted estimate).
    fn reasoning(&self) -> Option<antecedent_contract::ReasoningView> {
        self.last.as_ref().map(antecedent::LearnedTrialResult::reasoning).or_else(|| {
            self.inner
                .as_ref()
                .map(antecedent::PreparedStudy::<antecedent::LearnedTrialState>::inspect)
        })
    }
}
#[pymethods]
impl PreparedLearnedTrial {
    #[pyo3(signature=(cancel=None))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        self.estimate_payload(py, cancel, false)
    }
    #[pyo3(signature=(cancel=None))]
    fn estimate_summary(
        &mut self,
        py: Python<'_>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        self.estimate_payload(py, cancel, true)
    }
    #[pyo3(signature=(seed=None))]
    fn fork(&self, seed: Option<u64>) -> PyResult<Self> {
        let seed = seed.unwrap_or(self.seed);
        let ctx = self.ctx(None);
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| {
                crate::refusal(
                    "route_not_supported",
                    "loaded trial claims do not restore raw execution",
                )
            })?
            .with_seed(seed, &ctx)
            .map_err(err)?;
        Ok(Self { inner: Some(inner), seed, last: None, ..self.clone() })
    }
    #[pyo3(signature=(data,cancel=None))]
    fn bind_snapshot(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Self> {
        let input = parse_input(data, &self.names)?;
        let ctx = self.ctx(cancel);
        let inner = self.inner.clone().ok_or_else(|| {
            crate::refusal("route_not_supported", "loaded trial claims do not restore preparation")
        })?;
        let inner = crate::detach_catch(py, move || inner.bind_snapshot(input, &ctx).map_err(err))?;
        Ok(Self { inner: Some(inner), last: None, ..self.clone() })
    }
    #[pyo3(signature=(graph, selections, source, target, treatment, outcome, options, *, seed=1, memory_bytes=None, cancel=None))]
    fn reconfigure(
        &self,
        graph: &crate::graphs::Admg,
        selections: Vec<String>,
        source: String,
        target: String,
        treatment: String,
        outcome: String,
        options: String,
        seed: u64,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Self> {
        if graph.names != self.names {
            return Err(err(
                "trial reconfiguration requires the retained variable coordinate schema",
            ));
        }
        let (diagram, query) =
            trial_request(graph, selections, source, target, treatment, outcome)?;
        let options: TrialAipwOptions = serde_json::from_str(&options).map_err(err)?;
        let mut ctx = ExecutionContext::production_default(seed);
        ctx.set_memory_limit(memory_bytes);
        crate::apply_cancel(&mut ctx, cancel);
        crate::execution_api::apply(&mut ctx);
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| {
                crate::refusal(
                    "route_not_supported",
                    "loaded trial claims do not restore preparation",
                )
            })?
            .reconfigure(diagram, query, options, &ctx)
            .map_err(err)?;
        Ok(Self { inner: Some(inner), seed, memory_bytes, last: None, names: self.names.clone() })
    }
    fn last_result_summary(&self) -> PyResult<String> {
        trial_payload(self.last.as_ref().ok_or_else(|| err("execute before reading result"))?, true)
    }
    fn freeze(&self) -> Self {
        self.clone()
    }
    fn export(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let result = self.last.as_ref().ok_or_else(|| err("execute before exporting"))?;
        let mut bytes = b"ANTECEDENT-LEARNED-TRIAL\x01".to_vec();
        bytes.extend(result.export().map_err(err)?);
        Ok(PyBytes::new(py, &bytes).unbind())
    }
    fn inspection_json(&self) -> String {
        let uncertainty = self.last.as_ref().map(antecedent::LearnedTrialResult::estimate);
        let reasoning = self.reasoning();
        let identification_available =
            reasoning.as_ref().is_some_and(|r| r.identification.is_available());
        let identification_status = reasoning
            .as_ref()
            .and_then(|r| r.identification.as_ref())
            .map(|slot| slot.status.as_str());
        serde_json::json!({
            "identification":{"available":identification_available,"summary":identification_status.unwrap_or("unavailable")},
            "support":{"available":self.last.is_some(),"summary":"trial_and_target_overlap_required", "payload": {"overlap":uncertainty.map(|r| r.overlap),"slot":reasoning.as_ref().map(|r| crate::transport_common::support_slot_json(&r.support))}},
            "uncertainty":{"available":uncertainty.is_some_and(|r| r.interval.is_some()),"summary":"joint_outer_bootstrap_uncalibrated","reason":uncertainty.and_then(|r| r.uncertainty_reason.as_deref()),"payload":{"calibration_status":"not_bound_to_this_execution"}},
            "assumptions":{"available":reasoning.as_ref().is_some_and(|r| r.assumptions.is_available()),"summary":"declared_randomization_sampling_and_nuisance_models","payload":{"obligations":reasoning.as_ref().map(|r| crate::transport_common::obligation_ids(&r.assumptions))}},
            "execution_id":self.last.as_ref().map(antecedent::LearnedTrialResult::identity)
        }).to_string()
    }
    fn last_result(&self) -> PyResult<String> {
        serde_json::to_string(
            self.last.as_ref().ok_or_else(|| err("execute before reading result"))?.estimate(),
        )
        .map_err(err)
    }
    fn preview_transform(
        &self,
        intent: &str,
    ) -> PyResult<std::collections::BTreeMap<String, String>> {
        let intent = crate::prepared_api::parse_transform_intent(intent)?;
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| err("loaded claims require preparation for transformation preview"))?;
        Ok(crate::prepared_api::transform_report_map(
            &inner.preview_transform(intent).map_err(err)?,
        )
        .into_iter()
        .collect())
    }
    #[allow(
        clippy::unused_self,
        reason = "exposed as an instance method of the Python class, so it must take self even though the summary is constant"
    )]
    fn plan_summary(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::from([("structure_source".into(), "explicit".into())])
    }
    #[pyo3(signature=(data,cancel=None))]
    fn replace_snapshot(
        &mut self,
        data: &Bound<'_, PyAny>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let input = parse_input(data, &self.names)?;
        let ctx = self.ctx(cancel);
        self.inner
            .as_mut()
            .ok_or_else(|| err("loaded claims cannot refresh"))?
            .replace_snapshot(input, &ctx)
            .map_err(err)?;
        self.last = None;
        Ok(())
    }
}
#[allow(clippy::cast_possible_truncation, reason = "graph node ids fit u32 by construction")]
fn trial_request(
    graph: &crate::graphs::Admg,
    selections: Vec<String>,
    source: String,
    target: String,
    treatment: String,
    outcome: String,
) -> PyResult<(antecedent_graph::SelectionDiagram, TransportQuery)> {
    let resolve = |name: &str| -> PyResult<VariableId> {
        graph
            .names
            .iter()
            .position(|n| n == name)
            .map(|i| VariableId::from_raw(i as u32))
            .ok_or_else(|| err(format!("unknown variable {name}")))
    };
    let treatment = resolve(&treatment)?;
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: resolve(&outcome)?,
        treatment: ContinuousDomain::new(treatment, GridSpec::Values(Arc::from([0.0, 1.0]))),
    });
    let query = TransportQuery::new(response, source, target, [treatment]);
    let diagram = antecedent_graph::SelectionDiagram::try_new(
        graph.admg.clone(),
        selections.iter().map(|s| resolve(s)).collect::<PyResult<Vec<_>>>()?,
    )
    .map_err(err)?;
    Ok((diagram, query))
}
#[pyfunction]
#[pyo3(signature=(graph, selections, source, target, treatment, outcome, data, options, *, seed=1, memory_bytes=None, cancel=None))]
fn prepare_learned_trial(
    graph: &crate::graphs::Admg,
    selections: Vec<String>,
    source: String,
    target: String,
    treatment: String,
    outcome: String,
    data: &Bound<'_, PyAny>,
    options: String,
    seed: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PreparedLearnedTrial> {
    let (diagram, query) = trial_request(graph, selections, source, target, treatment, outcome)?;
    let input = parse_input(data, &graph.names)?;
    let options: TrialAipwOptions = serde_json::from_str(&options).map_err(err)?;
    let mut ctx = ExecutionContext::production_default(seed);
    ctx.set_memory_limit(memory_bytes);
    crate::apply_cancel(&mut ctx, cancel);
    crate::execution_api::apply(&mut ctx);
    let inner =
        antecedent::StudyBuilder::learned_trial_transport(diagram, query, input, options, &ctx)
            .map_err(err)?;
    Ok(PreparedLearnedTrial {
        inner: Some(inner),
        names: graph.names.clone(),
        last: None,
        seed,
        memory_bytes,
    })
}
#[pyfunction]
fn consume_learned_trial(bytes: &[u8]) -> PyResult<PreparedLearnedTrial> {
    let payload = bytes
        .strip_prefix(b"ANTECEDENT-LEARNED-TRIAL\x01")
        .ok_or_else(|| err("invalid learned trial envelope"))?;
    let result = antecedent::LearnedTrialResult::consume(payload).map_err(err)?;
    Ok(PreparedLearnedTrial {
        inner: None,
        names: vec![],
        last: Some(result),
        seed: 0,
        memory_bytes: None,
    })
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PreparedLearnedTrial>()?;
    m.add_function(wrap_pyfunction!(consume_learned_trial, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_learned_trial, m)?)?;
    Ok(())
}
