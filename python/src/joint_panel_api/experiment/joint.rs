//! Python inputs for joint conditioning of one retained panel on a retained excursion.
//!
//! Observed contrasts and their sampling covariance come from the retained evidence inside
//! Rust; Python declares only each component's bridge and mapping and the shared evidence.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{facade, future_problem, origin, ranking_json};
use crate::evidence_ledger_api::EvidenceLedgerHandle;
use crate::joint_panel_api::{ContrastFunctional, PyJointPanelResult, context, panel_error};
use antecedent::decision_evidence::{
    JointEvidenceComponent, RetainedJointPanelUpdate, UtilityExpr,
    condition_panel_jointly_on_retained_evidence,
};
use pyo3::{prelude::*, types::PyBytes};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentDeclaration {
    component: String,
    bridge: serde_json::Value,
    functional: ContrastFunctional,
    mapping_dependency: String,
    intercepts: Vec<f64>,
    slopes: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    components: Vec<ComponentDeclaration>,
    model: String,
    experiment: String,
    mapping: String,
    prior_origins: Vec<String>,
}

fn parse(
    declaration_json: &str,
) -> PyResult<(Vec<JointEvidenceComponent>, facade::PanelExperimentEvidence)> {
    crate::bridge::check_declared_size(
        declaration_json,
        crate::bridge::MAX_SPEC_DECLARATION_BYTES,
        "joint panel experiment",
    )?;
    let d: Declaration =
        serde_json::from_str(declaration_json).map_err(|e| crate::value_err(e.to_string()))?;
    let components = d
        .components
        .into_iter()
        .map(|c| {
            let dependency =
                antecedent::contract::evidence_ledger::parse_digest_hex(&c.mapping_dependency)
                    .ok_or_else(|| crate::value_err("invalid mapping dependency digest"))?;
            Ok(JointEvidenceComponent {
                component: c.component,
                bridge: crate::scientific_bridge_api::parse_contract(&c.bridge)?,
                functional: c.functional.native(),
                mapping: facade::PanelExperimentMapping {
                    dependency,
                    intercepts: c.intercepts,
                    slopes: c.slopes,
                },
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let evidence = facade::PanelExperimentEvidence {
        model: origin(&d.model)?,
        experiment: origin(&d.experiment)?,
        mapping: origin(&d.mapping)?,
        prior_origins: d.prior_origins.iter().map(|id| origin(id)).collect::<PyResult<_>>()?,
    };
    Ok((components, evidence))
}

fn expression(json: &str) -> PyResult<UtilityExpr> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| crate::value_err(e.to_string()))?;
    crate::decision_evidence_api::expr(&value)
}

#[pymethods]
impl PyJointPanelResult {
    #[pyo3(signature=(declaration_json,excursion_json,likelihood_assumption_id,ledger,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn condition_joint_excursion(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        excursion_json: &str,
        likelihood_assumption_id: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyJointExperiment> {
        crate::bridge::check_declared_size(
            excursion_json,
            crate::decision_evidence_api::MAX_DECISION_SOURCE_JSON,
            "compiled excursion evidence",
        )?;
        let (components, evidence) = parse(declaration_json)?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let source =
            crate::decision_evidence_api::compiled_excursion_from_json(excursion_json, &ctx)?;
        let ledger = ledger.ledger().clone();
        let inner = crate::detach_catch(py, || {
            condition_panel_jointly_on_retained_evidence(
                &self.inner,
                source,
                components,
                evidence,
                likelihood_assumption_id,
                &ledger,
                &ctx,
            )
            .map_err(panel_error)
        })?;
        Ok(PyJointExperiment { inner })
    }
}

#[pyclass(name = "_JointPanelJointExperiment", frozen)]
pub(crate) struct PyJointExperiment {
    inner: RetainedJointPanelUpdate,
}

#[pymethods]
impl PyJointExperiment {
    fn report(&self, py: Python<'_>, ledger: PyRef<'_, EvidenceLedgerHandle>) -> PyResult<String> {
        let ledger = ledger.ledger().clone();
        crate::detach_catch(py, || {
            let result = self.inner.result();
            // Assumption-bounded evidence has no posterior weights; that is reported, not hidden.
            let weights = result.posterior_weights(&ledger).ok().map(<[f64]>::to_vec);
            Ok(serde_json::json!({
                "scope": result.scope(),
                "likelihood_assumption_id": self.inner.likelihood_assumption_id(),
                "dependency_identity": result.dependency_identity().to_hex(),
                "posterior_weights": weights,
                "quantities": result.informed_quantities().iter()
                    .map(antecedent::io::quantity_wire::ScientificQuantityWire::from)
                    .collect::<Vec<_>>(),
                "calibration": "not_measured",
            })
            .to_string())
        })
    }
    fn derived_quantity(
        &self,
        py: Python<'_>,
        expression_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
    ) -> PyResult<String> {
        let expr = expression(expression_json)?;
        let ledger = ledger.ledger().clone();
        crate::detach_catch(py, || {
            let c = self.inner.result().derived_quantity(&expr, &ledger).map_err(panel_error)?;
            Ok(serde_json::json!({
                "prior_mean": c.prior_mean, "posterior_mean": c.posterior_mean,
                "prior_variance": c.prior_variance, "posterior_variance": c.posterior_variance,
                "scope": "model_conditional", "calibration": "not_measured",
            })
            .to_string())
        })
    }
    fn bias_sensitivity(
        &self,
        py: Python<'_>,
        expression_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
    ) -> PyResult<String> {
        let expr = expression(expression_json)?;
        let ledger = ledger.ledger().clone();
        crate::detach_catch(py, || {
            let s = self.inner.result().bias_sensitivity(&expr, &ledger).map_err(panel_error)?;
            Ok(serde_json::json!({
                "enumerated_minimum": s.enumerated_minimum,
                "enumerated_maximum": s.enumerated_maximum,
                "nominal": s.nominal, "settings": s.settings,
                "interpretation": "assumption_sensitivity_not_a_statistical_interval",
            })
            .to_string())
        })
    }
    #[pyo3(signature=(component,declaration_json,ledger,*,replicates=199,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn rank_next_samples(
        &self,
        py: Python<'_>,
        component: usize,
        declaration_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        replicates: usize,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let (problem, candidates) = future_problem(declaration_json)?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let ledger = ledger.ledger().clone();
        let rows = crate::detach_catch(py, || {
            self.inner
                .result()
                .rank_next_samples(component, &problem, &candidates, replicates, &ledger, &ctx)
                .map_err(panel_error)
        })?;
        Ok(ranking_json(&rows))
    }
    #[pyo3(signature=(ledger,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let ledger = ledger.ledger().clone();
        let bytes = crate::detach_catch(py, || {
            self.inner.result().export(&ledger, &ctx).map_err(panel_error)
        })?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
    #[pyo3(signature=(bytes,component,ledger,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn verify_current(
        &self,
        py: Python<'_>,
        bytes: &[u8],
        component: usize,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let ledger = ledger.ledger().clone();
        crate::detach_catch(py, || {
            self.inner
                .result()
                .consume_current(bytes, component, &ledger, &ctx)
                .map(|_| ())
                .map_err(panel_error)
        })
    }
}

/// Historical replay of an exported joint update from its retained inputs alone.
#[pyfunction]
#[pyo3(signature=(artifact,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn replay_joint_panel_experiment(
    py: Python<'_>,
    artifact: &[u8],
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    crate::detach_catch(py, || {
        let h = facade::PanelJointExperimentResult::replay(artifact, &ctx).map_err(panel_error)?;
        Ok(serde_json::json!({
            "posterior_weights": h.weights(),
            "dependency": h.dependency_identity().to_hex(),
            "grants_current_authority": h.grants_current_authority(),
        })
        .to_string())
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(replay_joint_panel_experiment, module)?)?;
    module.add_class::<PyJointExperiment>()?;
    Ok(())
}
