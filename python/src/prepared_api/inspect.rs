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
    /// Independent execution header sharing immutable native prepared storage.
    fn fork(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            names: self.names.clone(),
            last: None,
            last_study: None,
            last_seed: self.last_seed,
            last_threads: self.last_threads,
            borrowed_bytes: self.borrowed_bytes,
            series: self.series,
        }
    }

    /// Requote the same native snapshot under explicit execution controls, without fitting.
    #[pyo3(signature = (*, seed=1, threads=None))]
    fn reconfigure_execution(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Self> {
        let inner = Arc::clone(&self.inner);
        let names = self.names.clone();
        let series = self.series;
        crate::detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                crate::resolve_user_threads(threads),
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let prepared = inner.reconfigure_execution(&ctx).map_err(py_err)?;
            Ok(super::finished_prepared(prepared, names, series))
        })
    }

    /// Branch onto a new owned snapshot; never execute the resulting study.
    #[pyo3(signature = (columns, *, frame=None, seed=1, threads=None))]
    fn bind_snapshot(
        &self,
        py: Python<'_>,
        columns: Vec<Bound<'_, PyAny>>,
        frame: Option<&Bound<'_, PyDict>>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Self> {
        let data = FrameInput::ingest(py, &self.names, columns, frame)?;
        let inner = Arc::clone(&self.inner);
        let names = self.names.clone();
        let series = self.series;
        crate::detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                crate::resolve_user_threads(threads),
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let rebound = match data {
                FrameInput::Series(data) if !series => inner.rebind_data(data, &ctx),
                input => match input.materialize()? {
                    super::FrameData::Series(data) => inner.rebind_series(data, &ctx),
                    super::FrameData::Panel(data) => inner.rebind_panel(data, &ctx),
                    super::FrameData::MultiEnv(data) => inner.rebind_multi_env(data, &ctx),
                    super::FrameData::Events(data, interval, _retention) => {
                        inner.rebind_series(data.align_to_grid(interval).map_err(py_err)?, &ctx)
                    }
                },
            }
            .map_err(py_err)?;
            Ok(super::finished_prepared(rebound, names, series))
        })
    }

    /// Cached identification product; never identifies or fits again.
    fn identification_json(&self) -> PyResult<String> {
        let mut details = self.inner.identification_details().map_err(crate::py_err)?;
        details["names"] = serde_json::json!(self.names);
        serde_json::to_string(&details).map_err(|error| crate::value_err(error.to_string()))
    }

    /// One view over every layer of the prepared program as JSON: the lowered
    /// program, the identification preparation installed, the physical plan and
    /// the sealed operation retained, with that operation's inspection record
    /// under `operation_detail`. Never identifies, fits or scans.
    fn plan_view_json(&self) -> PyResult<String> {
        let view = self.inner.plan_view().map_err(py_err)?;
        let mut json = view.to_json();
        json["names"] = serde_json::json!(self.names);
        json["operation_detail"] = self.operation_detail();
        serde_json::to_string(&json).map_err(|error| crate::value_err(error.to_string()))
    }

    /// Readable report of this handle as Markdown or HTML: the executed claim when an
    /// estimate has run, otherwise the prepared contract, with the plan view of the program.
    /// The Rust renderer reads the records only; it identifies, fits and scans nothing.
    #[pyo3(signature = (format="markdown"))]
    fn report(&self, format: &str) -> PyResult<String> {
        let format = antecedent::report::ReportFormat::from_name(format).ok_or_else(|| {
            crate::value_err(format!(
                "unknown report format {format:?}; use \"markdown\" or \"html\""
            ))
        })?;
        let ctx = py_execution_context_ext(
            self.last_seed,
            self.last_threads,
            None,
            None,
            Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
        );
        // The report reads the handle the last execution ran on, so a retarget or a
        // second-click refute is reported under the contract it executed under.
        let inner = if self.last.is_some() {
            self.last_study.as_ref().unwrap_or(&self.inner)
        } else {
            &self.inner
        };
        inner.report(self.last.as_deref(), &ctx, format).map_err(py_err)
    }

    /// Whether the last execution carries the contract it ran under. A functional-validation
    /// result does not: the handle's contract names a run with no validation, which does not
    /// describe it.
    fn has_execution_contract(&self) -> bool {
        self.last.as_ref().is_some_and(|result| result.executed_contract.is_some())
    }

    /// Capture native contract and evidence at the explicit execution boundary.
    /// Posterior draw payloads and response values are not serialized into this record.
    #[pyo3(signature = (*, outcome_units=None, population="target", transform="identity"))]
    fn execution_record_json(
        &self,
        outcome_units: Option<&str>,
        population: &str,
        transform: &str,
    ) -> PyResult<String> {
        let result = self
            .last
            .as_ref()
            .ok_or_else(|| crate::value_err("estimate before capturing an execution record"))?;
        let inner = self.last_study.as_ref().unwrap_or(&self.inner);
        let ctx = py_execution_context_ext(
            self.last_seed,
            self.last_threads,
            None,
            None,
            Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
        );
        let labels =
            outcome_units.map(|outcome_units| antecedent_contract::ResponseCoordinateLabels {
                outcome_units,
                population_id: population,
                transform_id: transform,
            });
        let (section, body) = inner
            .execution_record_with_quantity_labels(result, &ctx, labels.as_ref())
            .map_err(py_err)?;
        let response = body.response.as_ref().map(|response| {
            serde_json::json!({
                "uncertainty": response.uncertainty,
                "support": response.support,
            })
        });
        let structural = body.structural_response.as_ref().map(|response| {
            serde_json::json!({
                "identified_set_interval": response.identified_set_interval,
            })
        });
        serde_json::to_string(&serde_json::json!({
            "contract": section,
            "evidence": {
                "identification": body.identification,
                "identification_variables": body.identification_variables,
                "assumptions": body.assumptions,
                "refutations": body.refutations,
                "diagnostics": body.diagnostics,
                "standard_error": body.standard_error,
                "unit_effects": body.unit_effects,
                "response": response,
                "structural_response": structural,
            },
        }))
        .map_err(|error| crate::value_err(error.to_string()))
    }

    /// Inspection of the frozen execution, using the executed reasoning slots.
    #[pyo3(signature = (*, outcome_units=None, population="target", transform="identity"))]
    fn execution_contract(
        &self,
        outcome_units: Option<&str>,
        population: &str,
        transform: &str,
    ) -> PyResult<std::collections::HashMap<String, String>> {
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
        let labels =
            outcome_units.map(|outcome_units| antecedent_contract::ResponseCoordinateLabels {
                outcome_units,
                population_id: population,
                transform_id: transform,
            });
        let (mut contract, claim) = inner
            .execution_contract_with_quantity_labels(result, &ctx, labels.as_ref())
            .map_err(py_err)?;
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

    /// Tier overlap and E-value diagnostics read off the last executed tiered result.
    ///
    /// Reads only what the executed result carries; nothing is refit. A refusal is the facade's
    /// own registered code and `tier_diagnostics.*` detail.
    #[pyo3(signature = (*, cancel=None))]
    fn tier_diagnostics_json(
        &self,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        use antecedent::analysis::{TierDesign, tier_diagnostics};
        use antecedent_estimate::Availability;

        let result = self
            .last
            .as_ref()
            .ok_or_else(|| crate::value_err("estimate before reading tier diagnostics"))?;
        let ctx = py_execution_context_ext(
            self.last_seed,
            self.last_threads,
            cancel.map(|c| c.inner),
            None,
            Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
        );
        let name_of = |id: antecedent_core::VariableId| {
            self.names.get(id.as_usize()).cloned().unwrap_or_else(|| id.to_string())
        };
        let unavailable = |u: &antecedent_estimate::Unavailable| {
            serde_json::json!({
                "available": false, "code": u.code, "detail": u.detail, "reason": u.reason,
            })
        };
        let report = tier_diagnostics(result, &ctx).map_err(|error| {
            crate::refusal(error.code, format!("{}: {}", error.detail, error.message))
        })?;
        let design = match &report.design {
            TierDesign::CoDeterminedClosure { adjustment, rows, propensity_scored } => {
                serde_json::json!({
                    "kind": "co_determined_closure",
                    "adjustment": adjustment.iter().copied().map(&name_of).collect::<Vec<_>>(),
                    "rows": rows,
                    "propensity_scored": propensity_scored,
                })
            }
            TierDesign::UnknownScenarios { scenarios } => serde_json::json!({
                "kind": "unknown_scenarios",
                "scenarios": scenarios.iter().map(|s| serde_json::json!({
                    "method": s.method.as_ref(),
                    "adjustment": s.adjustment.iter().copied().map(&name_of).collect::<Vec<_>>(),
                    "effect": s.effect,
                })).collect::<Vec<_>>(),
            }),
        };
        let overlap = match &report.overlap {
            Availability::Available(o) => serde_json::json!({
                "available": true,
                "propensity_min": o.propensity_min,
                "propensity_max": o.propensity_max,
                "ess": o.ess,
                "extreme_weight_count": o.extreme_weight_count,
                "excluded_fraction": o.excluded_fraction,
                "target_population_support": o.target_population_support,
                "excluded_regions": o.excluded_regions.iter()
                    .map(|r| [r.low, r.high]).collect::<Vec<_>>(),
                "clip": o.clip,
                "trim": o.trim,
                "retained_fraction": o.retained_fraction,
                "clip_sensitivity": o.clip_sensitivity.as_ref().map(|c| serde_json::json!({
                    "thresholds": c.thresholds.as_ref(),
                    "ess": c.ess.as_ref(),
                    "treated_ess": c.treated_ess.as_ref(),
                    "control_ess": c.control_ess.as_ref(),
                    "extreme_weight_counts": c.extreme_weight_counts.as_ref(),
                })),
            }),
            Availability::Unavailable(u) => unavailable(u),
        };
        let point = match &report.evalue.point {
            Availability::Available(p) => serde_json::json!({
                "available": true, "value": p.value, "method": p.method.as_ref(),
            }),
            Availability::Unavailable(u) => unavailable(u),
        };
        let payload = serde_json::json!({
            "design": design,
            "overlap": overlap,
            "evalue": { "point": point, "interval": unavailable(&report.evalue.interval) },
        });
        serde_json::to_string(&payload).map_err(|error| crate::value_err(error.to_string()))
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

impl PyPreparedAnalysis {
    fn variable_name(&self, id: antecedent_core::VariableId) -> String {
        self.names.get(id.as_usize()).cloned().unwrap_or_else(|| id.to_string())
    }

    /// The retained operation's inspection record, by its stable name.
    fn operation_detail(&self) -> serde_json::Value {
        use antecedent::analysis::RetainedOperation as R;
        use serde_json::json;
        let operation = self.inner.retained_operation();
        let name = |id: antecedent_core::VariableId| self.variable_name(id);
        let names = |ids: &[antecedent_core::VariableId]| {
            ids.iter().copied().map(|id| self.variable_name(id)).collect::<Vec<_>>()
        };
        let mut detail = match &operation {
            R::CheckedConditional(info) => json!({
                "query": format!("{:?}", info.query),
                "treatment": name(info.query.inner.treatment),
                "outcome": name(info.query.inner.outcome),
                "modifiers": names(&info.query.inner.effect_modifiers[..]),
                "control": format!("{:?}", info.query.inner.control),
                "active": format!("{:?}", info.query.inner.active),
                "functional": format!("{:?}", info.query.inner.outcome_functional),
                "population": format!("{:?}", info.query.inner.target_population),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "procedure": info.procedure.as_ref(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "design_roles": names(&info.design_roles[..]),
                "source_rows": info.source_rows.as_ref(),
            }),
            R::StaticMediation(info) | R::BayesianStaticMediation(info) => json!({
                "query": format!("{:?}", info.query),
                "treatment": name(info.query.treatment),
                "outcome": name(info.query.outcome),
                "mediators": names(&info.query.mediators[..]),
                "contrast": format!("{:?}", info.query.contrast),
                "control": format!("{:?}", info.query.control),
                "active": format!("{:?}", info.query.active),
                "population": format!("{:?}", info.query.target_population),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "bootstrap_replicates": info.bootstrap_replicates,
                "adjustment_set": names(&info.adjustment_set[..]),
                "functional_roots": info.functional_roots,
                "graph_edges": info.graph_edges.as_ref(),
            }),
            R::BayesianGcomp { operation, validation } => {
                let query = operation.query();
                let mut detail = json!({
                    "query": format!("{query:?}"),
                    "treatment": name(query.treatment),
                    "outcome": name(query.outcome),
                    "population": format!("{:?}", query.target_population),
                    "identifier": "backdoor.adjustment",
                    "estimator": "bayesian.gcomp",
                    "validation": validation.validation_suite_id().unwrap_or("none"),
                    "adjustment_set": names(&operation.estimand().adjustment_set[..]),
                });
                if let antecedent::InferenceMode::Bayesian(config) = operation.inference() {
                    detail["backend"] = json!(format!("{:?}", config.backend));
                    detail["likelihood"] = json!(format!("{:?}", config.likelihood));
                    detail["n_draws"] = json!(config.n_draws);
                    detail["prior_scale"] = json!(config.prior_scale);
                    detail["transferred_prior"] = json!(config.prior_artifact.is_some());
                    detail["composed_prior"] = json!(config.external_compose.is_some());
                }
                detail
            }
            R::StaticDagResponse(info) => json!({
                "query": format!("{:?}", info.query.functional),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "grid_members": info.grid_members.as_ref(),
            }),
            R::GraphPosteriorEffect(info) => json!({
                "query": format!("{:?}", info.query),
                "conditional": info.conditional,
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "graph_keys": info.graph_keys.as_ref(),
                "weights": info.weights.as_ref(),
                "identified": info.identified.iter().map(|flag| format!("{flag:?}")).collect::<Vec<_>>(),
                "bootstrap_replicates": info.bootstrap_replicates,
            }),
            R::TemporalDagEffect(info) => json!({
                "query": format!("{:?}", info.query),
                "policy": format!("{:?}", info.query.policy),
                "control": format!("{:?}", info.query.control),
                "active": format!("{:?}", info.query.active),
                "horizon_steps": info.query.horizon_steps,
                "population": format!("{:?}", info.query.target_population),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "bootstrap_replicates": info.bootstrap_replicates,
                "adjustment_set": info.adjustment_set.iter().map(|key| format!("{key:?}")).collect::<Vec<_>>(),
                "graph_signature": info.graph_signature.as_ref(),
            }),
            R::TemporalDagResponse(info) => json!({
                "query": format!("{:?}", info.query.functional),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "uncertainty": info.uncertainty.as_ref(),
                "grid_members": info.grid_members.as_ref(),
                "horizons": info.horizons.as_ref(),
            }),
            _ => json!({}),
        };
        detail["kind"] = json!(operation.name());
        detail
    }
}
