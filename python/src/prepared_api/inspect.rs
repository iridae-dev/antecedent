//! Checked-target and execution inspection on `PreparedAnalysis`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::{
    ClickControls, FrameInput, PyPreparedAnalysis, contract_to_map, insert_reuse_identities,
    map_ate, require_prepared_names, response_from_study,
};
use crate::response_api::ResponseAnalysisResult;
use crate::{AteAnalysisResult, py_err, py_execution_context_ext};

#[pymethods]
impl PyPreparedAnalysis {
    /// Cached identification product; never identifies or fits again.
    fn identification_json(&self) -> PyResult<String> {
        let mut details = self.inner.identification_details().map_err(crate::py_err)?;
        details["names"] = serde_json::json!(self.names);
        serde_json::to_string(&details).map_err(|error| crate::value_err(error.to_string()))
    }

    /// Inspect the checked conditional target and its bound scalar procedure.
    fn checked_conditional_effect_info(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(info) = self.inner.checked_conditional_effect_info() else {
            return Ok(None);
        };
        let name = |id: antecedent_core::VariableId| {
            self.names.get(id.as_usize()).cloned().unwrap_or_else(|| id.to_string())
        };
        let descriptor = PyDict::new(py);
        descriptor.set_item("query", format!("{:?}", info.query))?;
        descriptor.set_item("treatment", name(info.query.inner.treatment))?;
        descriptor.set_item("outcome", name(info.query.inner.outcome))?;
        descriptor.set_item(
            "modifiers",
            info.query.inner.effect_modifiers.iter().copied().map(&name).collect::<Vec<_>>(),
        )?;
        descriptor.set_item("control", format!("{:?}", info.query.inner.control))?;
        descriptor.set_item("active", format!("{:?}", info.query.inner.active))?;
        descriptor.set_item("functional", format!("{:?}", info.query.inner.outcome_functional))?;
        descriptor.set_item("population", format!("{:?}", info.query.inner.target_population))?;
        descriptor.set_item("identifier", info.identifier.as_str())?;
        descriptor.set_item("estimator", info.estimator.as_str())?;
        descriptor.set_item("procedure", info.procedure.as_ref())?;
        descriptor
            .set_item("validation", info.validation.validation_suite_id().unwrap_or("none"))?;
        descriptor.set_item(
            "design_roles",
            info.design_roles.iter().copied().map(&name).collect::<Vec<_>>(),
        )?;
        descriptor.set_item("source_rows", info.source_rows.as_ref())?;
        Ok(Some(descriptor.unbind()))
    }

    /// Inspect the retained static mediation contrast, graph, and procedure.
    fn checked_static_mediation_info(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(info) = self.inner.checked_static_mediation_info() else {
            return Ok(None);
        };
        let name = |id: antecedent_core::VariableId| {
            self.names.get(id.as_usize()).cloned().unwrap_or_else(|| id.to_string())
        };
        let descriptor = PyDict::new(py);
        descriptor.set_item("query", format!("{:?}", info.query))?;
        descriptor.set_item("treatment", name(info.query.treatment))?;
        descriptor.set_item("outcome", name(info.query.outcome))?;
        descriptor.set_item(
            "mediators",
            info.query.mediators.iter().copied().map(&name).collect::<Vec<_>>(),
        )?;
        descriptor.set_item("contrast", format!("{:?}", info.query.contrast))?;
        descriptor.set_item("control", format!("{:?}", info.query.control))?;
        descriptor.set_item("active", format!("{:?}", info.query.active))?;
        descriptor.set_item("population", format!("{:?}", info.query.target_population))?;
        descriptor.set_item("identifier", info.identifier.as_str())?;
        descriptor.set_item("estimator", info.estimator.as_str())?;
        descriptor
            .set_item("validation", info.validation.validation_suite_id().unwrap_or("none"))?;
        descriptor.set_item("bootstrap_replicates", info.bootstrap_replicates)?;
        descriptor.set_item(
            "adjustment_set",
            info.adjustment_set.iter().copied().map(&name).collect::<Vec<_>>(),
        )?;
        descriptor.set_item("functional_roots", info.functional_roots)?;
        descriptor.set_item("graph_edges", info.graph_edges.as_ref())?;
        Ok(Some(descriptor.unbind()))
    }

    /// Inspect the retained DAG Bayesian ATE target, prior source, and validation.
    fn checked_bayesian_dag_ate_info(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(operation) = self.inner.checked_bayesian_gcomp_operation() else {
            return Ok(None);
        };
        let antecedent::InferenceMode::Bayesian(config) = operation.inference() else {
            return Ok(None);
        };
        let name = |id: antecedent_core::VariableId| {
            self.names.get(id.as_usize()).cloned().unwrap_or_else(|| id.to_string())
        };
        let query = operation.query();
        let descriptor = PyDict::new(py);
        descriptor.set_item("query", format!("{query:?}"))?;
        descriptor.set_item("treatment", name(query.treatment))?;
        descriptor.set_item("outcome", name(query.outcome))?;
        descriptor.set_item("population", format!("{:?}", query.target_population))?;
        descriptor.set_item("identifier", "backdoor.adjustment")?;
        descriptor.set_item("estimator", "bayesian.gcomp")?;
        descriptor.set_item("backend", format!("{:?}", config.backend))?;
        descriptor.set_item("likelihood", format!("{:?}", config.likelihood))?;
        descriptor.set_item("n_draws", config.n_draws)?;
        descriptor.set_item("prior_scale", config.prior_scale)?;
        descriptor.set_item("transferred_prior", config.prior_artifact.is_some())?;
        descriptor.set_item("composed_prior", config.external_compose.is_some())?;
        descriptor.set_item(
            "validation",
            self.inner
                .checked_bayesian_gcomp_validation()
                .and_then(antecedent::RefuteSuite::validation_suite_id)
                .unwrap_or("none"),
        )?;
        descriptor.set_item(
            "adjustment_set",
            operation.estimand().adjustment_set.iter().copied().map(&name).collect::<Vec<_>>(),
        )?;
        Ok(Some(descriptor.unbind()))
    }

    /// Read-only inspection of a retained checked static DAG response operation.
    ///
    /// The descriptor is built from the prepared handle rather than the Python
    /// query object, so callers can inspect the exact functional, procedure,
    /// validation suite, and frozen curve members after discarding their builder.
    fn checked_static_dag_response_info(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(info) = self.inner.checked_static_dag_response_info() else {
            return Ok(None);
        };
        let descriptor = PyDict::new(py);
        descriptor.set_item("query", format!("{:?}", info.query.functional))?;
        descriptor.set_item("identifier", info.identifier.as_str())?;
        descriptor.set_item("estimator", info.estimator.as_str())?;
        descriptor
            .set_item("validation", info.validation.validation_suite_id().unwrap_or("none"))?;
        descriptor.set_item("grid_members", info.grid_members.as_ref())?;
        Ok(Some(descriptor.unbind()))
    }

    /// Inspect frozen DAG graph-posterior atoms and the selected frequentist procedure.
    fn checked_graph_posterior_effect_info(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(info) = self.inner.checked_graph_posterior_effect_info() else {
            return Ok(None);
        };
        let descriptor = PyDict::new(py);
        descriptor.set_item("query", format!("{:?}", info.query))?;
        descriptor.set_item("conditional", info.conditional)?;
        descriptor.set_item("estimator", info.estimator.as_str())?;
        descriptor
            .set_item("validation", info.validation.validation_suite_id().unwrap_or("none"))?;
        descriptor.set_item("graph_keys", info.graph_keys.as_ref())?;
        descriptor.set_item("weights", info.weights.as_ref())?;
        descriptor.set_item(
            "identified",
            info.identified.iter().map(|flag| format!("{flag:?}")).collect::<Vec<_>>(),
        )?;
        descriptor.set_item("bootstrap_replicates", info.bootstrap_replicates)?;
        Ok(Some(descriptor.unbind()))
    }

    /// Inspect the frozen temporal contrast, proof binding, and estimator choice.
    fn checked_temporal_effect_info(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(info) = self.inner.checked_temporal_effect_info() else {
            return Ok(None);
        };
        let descriptor = PyDict::new(py);
        descriptor.set_item("query", format!("{:?}", info.query))?;
        descriptor.set_item("policy", format!("{:?}", info.query.policy))?;
        descriptor.set_item("control", format!("{:?}", info.query.control))?;
        descriptor.set_item("active", format!("{:?}", info.query.active))?;
        descriptor.set_item("horizon_steps", info.query.horizon_steps)?;
        descriptor.set_item("population", format!("{:?}", info.query.target_population))?;
        descriptor.set_item("identifier", info.identifier.as_str())?;
        descriptor.set_item("estimator", info.estimator.as_str())?;
        descriptor
            .set_item("validation", info.validation.validation_suite_id().unwrap_or("none"))?;
        descriptor.set_item("bootstrap_replicates", info.bootstrap_replicates)?;
        descriptor.set_item(
            "adjustment_set",
            info.adjustment_set.iter().map(|key| format!("{key:?}")).collect::<Vec<_>>(),
        )?;
        descriptor.set_item("graph_signature", info.graph_signature.as_ref())?;
        Ok(Some(descriptor.unbind()))
    }

    /// Inspect a fixed temporal DAG response grid and its frozen procedure.
    fn checked_temporal_response_info(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(info) = self.inner.checked_temporal_response_info() else {
            return Ok(None);
        };
        let descriptor = PyDict::new(py);
        descriptor.set_item("query", format!("{:?}", info.query.functional))?;
        descriptor.set_item("identifier", info.identifier.as_str())?;
        descriptor.set_item("estimator", info.estimator.as_str())?;
        descriptor.set_item("uncertainty", info.uncertainty.as_ref())?;
        descriptor.set_item("grid_members", info.grid_members.as_ref())?;
        descriptor.set_item("horizons", info.horizons.as_ref())?;
        Ok(Some(descriptor.unbind()))
    }

    /// Inspection of the frozen execution, using the executed reasoning slots.
    fn execution_contract(&self) -> PyResult<std::collections::HashMap<String, String>> {
        let result = self
            .last
            .as_ref()
            .ok_or_else(|| crate::value_err("estimate before inspecting an execution"))?;
        let inner = self.last_study.as_ref().unwrap_or(&self.inner);
        let ctx = py_execution_context_ext(
            self.last_seed,
            self.last_threads,
            None,
            None,
            Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
        );
        // The contract the execution ran under, with the identities the export
        // carries: a retarget moves the target population and its weighting
        // identity, and a second-click refute moves the validation suite.
        let (mut contract, claim) = inner.execution_contract(result, &ctx).map_err(py_err)?;
        contract.reasoning = claim.reasoning;
        let mut report =
            contract_to_map(&contract, inner.schema(), inner.query(), inner.population_registry());
        insert_reuse_identities(
            &mut report,
            inner.score_reuse_identity().map_err(py_err)?,
            result.row_weights.as_ref().map(|binding| binding.target_weights),
        );
        report.insert("claim_id".into(), claim.claim_id.to_hex());
        report.insert("calibration_status".into(), claim.calibration.status.to_string());
        if let Some(record_id) = &claim.calibration.record_id {
            report.insert("calibration_record_id".into(), record_id.to_string());
        }
        if let Some(reason) = &claim.calibration.reason {
            report.insert("calibration_reason".into(), reason.to_string());
        }
        if let Some(scope_n) = claim.calibration.scope_n {
            report.insert("calibration_scope_n".into(), scope_n.to_string());
        }
        if let Some(dep) = &claim.calibration.scope_dependence {
            report.insert("calibration_scope_dependence".into(), dep.to_string());
        }
        if let Some(sha) = &claim.calibration.calibration_sha {
            report.insert("calibration_sha".into(), sha.to_string());
        }
        Ok(report)
    }

    /// Frozen reference to the last execution, sharing immutable Rust resources.
    fn snapshot(&self) -> PyResult<Self> {
        let last = self
            .last
            .clone()
            .ok_or_else(|| crate::value_err("estimate before taking a result snapshot"))?;
        Ok(Self {
            inner: self.last_study.clone().unwrap_or_else(|| Arc::clone(&self.inner)),
            names: self.names.clone(),
            last: Some(last),
            last_study: self.last_study.clone(),
            last_seed: self.last_seed,
            last_threads: self.last_threads,
            borrowed_bytes: self.borrowed_bytes,
            series: self.series,
        })
    }

    /// Execute the retained data without crossing cell buffers through Python again.
    ///
    /// `cancel` / `on_progress` apply to this click; `on_stage` streams its
    /// stages where the route has them and is refused where it does not.
    #[pyo3(signature = (*, seed=1, threads=None, cancel=None, on_progress=None, on_stage=None))]
    fn estimate_bound(
        &mut self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        on_stage: Option<Bound<'_, PyAny>>,
    ) -> PyResult<AteAnalysisResult> {
        let controls = ClickControls::parse(cancel, on_progress.as_ref(), on_stage.as_ref())?;
        self.click_bound(py, seed, threads, controls, map_ate)
    }

    #[pyo3(signature = (*, seed=1, threads=None, cancel=None, on_progress=None, on_stage=None))]
    fn estimate_response_bound(
        &mut self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        on_stage: Option<Bound<'_, PyAny>>,
    ) -> PyResult<ResponseAnalysisResult> {
        let controls = ClickControls::parse(cancel, on_progress.as_ref(), on_stage.as_ref())?;
        self.click_bound(py, seed, threads, controls, response_from_study)
    }

    /// Execute on a panel, multi-environment, or event frame of the prepared schema.
    ///
    /// `refresh=True` also replaces the retained data. Returns the ATE-shaped
    /// result, or the response result when `response=True`.
    #[pyo3(signature = (
        names, columns, frame, *, response=false, refresh=false, seed=1, threads=None,
        cancel=None, on_progress=None, on_stage=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn estimate_frame(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        frame: Bound<'_, PyDict>,
        response: bool,
        refresh: bool,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        on_stage: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        require_prepared_names(&self.names, &names, "estimate")?;
        let controls = ClickControls::parse(cancel, on_progress.as_ref(), on_stage.as_ref())?;
        let data = FrameInput::ingest(py, &names, columns, Some(&frame))?;
        if response {
            let out =
                self.click(py, data, refresh, seed, threads, controls, response_from_study)?;
            Ok(Py::new(py, out)?.into_any())
        } else {
            let out = self.click(py, data, refresh, seed, threads, controls, map_ate)?;
            Ok(Py::new(py, out)?.into_any())
        }
    }

    /// Whether estimates of this handle stream progressive stages (`on_stage`).
    fn streams_stages(&self) -> bool {
        self.inner.streams_stages()
    }

    /// Re-bind caller custom validators by their attested names.
    ///
    /// The validators are in-process callbacks: a loaded claim carries their
    /// attested results, never the callables, so a rebound study must present
    /// exactly the names its claims attest.
    fn rebind_validators(&mut self, validators: Bound<'_, PyAny>) -> PyResult<()> {
        let parsed = crate::callbacks::parse_validators(Some(&validators))?;
        let mut study = (*self.inner).clone();
        study.rebind_custom_validators(parsed).map_err(py_err)?;
        self.inner = Arc::new(study);
        Ok(())
    }

    /// Attested custom-validator names frozen at prepare, in order.
    fn validator_names(&self) -> Vec<String> {
        self.inner.custom_validator_names().iter().map(|name| (*name).to_string()).collect()
    }
}
