//! Python experiment inputs execute the native retained joint-law workflow.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{ContrastFunctional, PyJointPanelResult, context, facade, panel_error};
use crate::evidence_ledger_api::EvidenceLedgerHandle;
use antecedent_contract::evidence_ledger::SourceId;
use pyo3::{prelude::*, types::PyBytes};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    bridge: serde_json::Value,
    functional: ContrastFunctional,
    observed_mean: f64,
    observation_variance: f64,
    sample_size: u64,
    mapping_dependency: String,
    intercepts: Vec<f64>,
    slopes: Vec<f64>,
    model: String,
    experiment: String,
    mapping: String,
    prior_origins: Vec<String>,
}
fn origin(s: &str) -> PyResult<SourceId> {
    SourceId::from_hex(s).ok_or_else(|| crate::value_err("invalid evidence origin digest"))
}
fn parse_update(declaration_json: &str) -> PyResult<facade::PanelExperimentUpdate> {
    crate::bridge::check_declared_size(
        declaration_json,
        crate::bridge::MAX_SPEC_DECLARATION_BYTES,
        "panel experiment",
    )?;
    let d: Declaration =
        serde_json::from_str(declaration_json).map_err(|e| crate::value_err(e.to_string()))?;
    let dependency = antecedent_contract::evidence_ledger::parse_digest_hex(&d.mapping_dependency)
        .ok_or_else(|| crate::value_err("invalid mapping dependency digest"))?;
    Ok(facade::PanelExperimentUpdate {
        bridge: crate::scientific_bridge_api::parse_contract(&d.bridge)?,
        functional: d.functional.native(),
        observed_mean: d.observed_mean,
        observation_variance: d.observation_variance,
        sample_size: d.sample_size,
        modular_policy: facade::PanelExperimentModularPolicy::FrozenJointImportance,
        mapping: facade::PanelExperimentMapping {
            dependency,
            intercepts: d.intercepts,
            slopes: d.slopes,
        },
        evidence: facade::PanelExperimentEvidence {
            model: origin(&d.model)?,
            experiment: origin(&d.experiment)?,
            mapping: origin(&d.mapping)?,
            prior_origins: d.prior_origins.iter().map(|id| origin(id)).collect::<PyResult<_>>()?,
        },
    })
}
#[pymethods]
impl PyJointPanelResult {
    fn experiment_quantity(&self, functional_json: &str) -> PyResult<String> {
        crate::bridge::check_declared_size(
            functional_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "experimental response functional",
        )?;
        let functional: ContrastFunctional =
            serde_json::from_str(functional_json).map_err(|e| crate::value_err(e.to_string()))?;
        let quantity = self.inner.experiment_quantity(&functional.native()).map_err(panel_error)?;
        serde_json::to_string(&antecedent_io::quantity_wire::ScientificQuantityWire::from(
            &quantity,
        ))
        .map_err(|e| crate::value_err(e.to_string()))
    }
    #[pyo3(signature=(declaration_json,ledger,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn condition_experiment(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyPanelExperiment> {
        let update = parse_update(declaration_json)?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let ledger = ledger.ledger().clone();
        let inner = crate::detach_catch(py, || {
            self.inner.condition_experiment(update, &ledger, &ctx).map_err(panel_error)
        })?;
        Ok(PyPanelExperiment { inner })
    }
}
#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct Action {
    id: String,
    intercept: f64,
    coefficients: Vec<f64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Future {
    id: String,
    sample_size: u64,
    observation_variance: f64,
    cost_utility: Option<f64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FutureDeclaration {
    actions: Vec<Action>,
    candidates: Vec<Future>,
}
struct Utility;
impl antecedent_design::decision::Utility<Action, facade::PanelExperimentDecisionState>
    for Utility
{
    fn evaluate_batch(
        &self,
        actions: &[Action],
        states: &[facade::PanelExperimentDecisionState],
        out: &mut [f64],
    ) -> Result<(), antecedent_design::DesignError> {
        if out.len() != actions.len() * states.len() {
            return Err(antecedent_design::DesignError::Shape(
                "terminal utility output axes".into(),
            ));
        }
        for (i, action) in actions.iter().enumerate() {
            for (j, state) in states.iter().enumerate() {
                if action.coefficients.len() != state.responses.len() {
                    return Err(antecedent_design::DesignError::Shape(
                        "terminal utility response coordinates".into(),
                    ));
                }
                out[i * states.len() + j] = action.intercept
                    + action
                        .coefficients
                        .iter()
                        .zip(&state.responses)
                        .map(|(c, v)| c * v)
                        .sum::<f64>();
            }
        }
        Ok(())
    }
}
#[pyclass(name = "_JointPanelExperiment", frozen)]
pub(crate) struct PyPanelExperiment {
    inner: facade::PanelExperimentResult,
}
#[pymethods]
impl PyPanelExperiment {
    #[pyo3(signature=(declaration_json,ledger,*,replicates=199,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn rank_next_samples(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        replicates: usize,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        crate::bridge::check_declared_size(
            declaration_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "future-study ranking",
        )?;
        let d: FutureDeclaration =
            serde_json::from_str(declaration_json).map_err(|e| crate::value_err(e.to_string()))?;
        if d.actions.is_empty()
            || d.actions.len() > 512
            || d.actions.iter().any(|a| {
                a.id.trim().is_empty()
                    || !a.intercept.is_finite()
                    || a.coefficients.iter().any(|v| !v.is_finite())
            })
        {
            return Err(crate::value_err("invalid bounded terminal action utility"));
        }
        if d.actions.iter().map(|a| &a.id).collect::<std::collections::BTreeSet<_>>().len()
            != d.actions.len()
        {
            return Err(crate::value_err("terminal action identities must be unique"));
        }
        let problem = antecedent_design::decision::DecisionProblem::new(
            d.actions,
            std::sync::Arc::new(Utility),
            vec![],
        );
        let candidates = d
            .candidates
            .into_iter()
            .map(|c| facade::PanelFutureSample {
                id: c.id,
                sample_size: c.sample_size,
                observation_variance: c.observation_variance,
                cost_utility: c.cost_utility,
            })
            .collect::<Vec<_>>();
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let ledger = ledger.ledger().clone();
        let rows = crate::detach_catch(py, || {
            self.inner
                .rank_next_samples(&problem, &candidates, replicates, &ledger, &ctx)
                .map_err(panel_error)
        })?;
        Ok(serde_json::json!({"scope":"model_conditional","calibration":"unmeasured","candidates":rows.iter().map(|r|serde_json::json!({"id":r.id,"gross_evsi":r.gross_evsi,"standard_error":r.standard_error,"net_value":r.net_value,"dependency_identity":r.dependency_identity.to_hex(),"question_identity":r.question_identity.to_hex(),"admissible_action_identity":r.admissible_action_identity.to_hex()})).collect::<Vec<_>>()}).to_string())
    }
    fn report(&self, ledger: PyRef<'_, EvidenceLedgerHandle>) -> PyResult<String> {
        let (prior, posterior) = self.inner.response_means(ledger.ledger()).map_err(panel_error)?;
        let law = self.inner.numerical_law_source(ledger.ledger()).map_err(panel_error)?;
        Ok(serde_json::json!({"prior_means":prior,"posterior_means":posterior,"posterior_weights":law.metadata().weights,"scope":self.inner.scope(),"quantities":self.inner.quantities(ledger.ledger()).map_err(panel_error)?.iter().map(antecedent_io::quantity_wire::ScientificQuantityWire::from).collect::<Vec<_>>(),"modular_policy":"frozen_joint_importance","calibration":"unmeasured"}).to_string())
    }
    fn decide(
        &self,
        contract_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
    ) -> PyResult<String> {
        let contract = antecedent_design::decision_artifact::contract_from_json(contract_json)
            .map_err(|e| crate::value_err(e.to_string()))?;
        let result = self.inner.decide(&contract, ledger.ledger()).map_err(panel_error)?;
        antecedent_design::decision_artifact::result_to_json(
            &result.result,
            self.inner.numerical_law_source(ledger.ledger()).map_err(panel_error)?,
        )
        .map_err(|e| crate::value_err(e.to_string()))
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
            self.inner.export(&ledger, &ctx).map_err(super::artifact_error)
        })?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
    #[pyo3(signature=(bytes,ledger,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn verify_current(
        &self,
        py: Python<'_>,
        bytes: &[u8],
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
                .consume_current(bytes, &ledger, &ctx)
                .map(|_| ())
                .map_err(super::artifact_error)
        })
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FutureRegimeDeclaration {
    bridge: serde_json::Value,
    functional: ContrastFunctional,
    mapping_dependency: String,
    intercepts: Vec<f64>,
    slopes: Vec<f64>,
    mapping_evidence: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FutureStudyDeclaration {
    id: String,
    sample_size: u64,
    observation_variance: f64,
    cost_utility: Option<f64>,
    regime: Option<FutureRegimeDeclaration>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FutureStudiesDeclaration {
    actions: Vec<Action>,
    studies: Vec<FutureStudyDeclaration>,
}
fn future_regime(d: FutureRegimeDeclaration) -> PyResult<facade::PanelFutureRegime> {
    let dependency = antecedent_contract::evidence_ledger::parse_digest_hex(&d.mapping_dependency)
        .ok_or_else(|| crate::value_err("invalid mapping dependency digest"))?;
    Ok(facade::PanelFutureRegime {
        functional: d.functional.native(),
        bridge: crate::scientific_bridge_api::parse_contract(&d.bridge)?,
        mapping: facade::PanelExperimentMapping {
            dependency,
            intercepts: d.intercepts,
            slopes: d.slopes,
        },
        mapping_evidence: origin(&d.mapping_evidence)?,
    })
}
#[pymethods]
impl PyPanelExperiment {
    /// Rank candidate studies that may run in their own regime, each with its own licensed
    /// bridge, response functional, paired affine mapping and current mapping evidence.
    #[pyo3(signature=(declaration_json,ledger,*,replicates=199,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn rank_next_studies(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        replicates: usize,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        crate::bridge::check_declared_size(
            declaration_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "future-study ranking",
        )?;
        let d: FutureStudiesDeclaration =
            serde_json::from_str(declaration_json).map_err(|e| crate::value_err(e.to_string()))?;
        if d.actions.is_empty()
            || d.actions.len() > 512
            || d.actions.iter().any(|a| {
                a.id.trim().is_empty()
                    || !a.intercept.is_finite()
                    || a.coefficients.iter().any(|v| !v.is_finite())
            })
        {
            return Err(crate::value_err("invalid bounded terminal action utility"));
        }
        if d.actions.iter().map(|a| &a.id).collect::<std::collections::BTreeSet<_>>().len()
            != d.actions.len()
        {
            return Err(crate::value_err("terminal action identities must be unique"));
        }
        let problem = antecedent_design::decision::DecisionProblem::new(
            d.actions,
            std::sync::Arc::new(Utility),
            vec![],
        );
        let mut studies = Vec::with_capacity(d.studies.len());
        for study in d.studies {
            studies.push(facade::PanelFutureStudy {
                sample: facade::PanelFutureSample {
                    id: study.id,
                    sample_size: study.sample_size,
                    observation_variance: study.observation_variance,
                    cost_utility: study.cost_utility,
                },
                regime: study.regime.map(future_regime).transpose()?,
            });
        }
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let ledger = ledger.ledger().clone();
        let rows = crate::detach_catch(py, || {
            self.inner
                .rank_next_studies(&problem, &studies, replicates, &ledger, &ctx)
                .map_err(panel_error)
        })?;
        Ok(serde_json::json!({"scope":"model_conditional","calibration":"unmeasured","candidates":rows.iter().map(|r|serde_json::json!({"id":r.value.id,"gross_evsi":r.value.gross_evsi,"standard_error":r.value.standard_error,"net_value":r.value.net_value,"dependency_identity":r.value.dependency_identity.to_hex(),"question_identity":r.value.question_identity.to_hex(),"admissible_action_identity":r.value.admissible_action_identity.to_hex(),"regime_identity":r.regime_identity.map(|d|d.to_hex())})).collect::<Vec<_>>()}).to_string())
    }
}
#[pymethods]
impl super::designs::PyJointDesignResult {
    fn experiment_quantity(&self, functional_json: &str) -> PyResult<String> {
        crate::bridge::check_declared_size(
            functional_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "experimental response functional",
        )?;
        let functional: ContrastFunctional =
            serde_json::from_str(functional_json).map_err(|e| crate::value_err(e.to_string()))?;
        if !matches!(functional, ContrastFunctional::TerminalMean) {
            return Err(crate::value_err(
                "native design source retains its declared whole-world mean; another time functional requires recompilation",
            ));
        }
        let source = self.inner.posterior_source().map_err(panel_error)?;
        let quantity = facade::JointLawExperimentResult::quantity(&source).map_err(panel_error)?;
        serde_json::to_string(&antecedent_io::quantity_wire::ScientificQuantityWire::from(
            &quantity,
        ))
        .map_err(|e| crate::value_err(e.to_string()))
    }
    #[pyo3(signature=(declaration_json,ledger,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn condition_experiment(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyDesignExperiment> {
        let update = parse_update(declaration_json)?;
        if !matches!(update.functional, facade::JointPanelResponseFunctional::TerminalMean) {
            return Err(crate::value_err(
                "native design source retains its declared whole-world mean",
            ));
        }
        let request = facade::JointLawExperimentRequest {
            bridge: update.bridge,
            observed_mean: update.observed_mean,
            observation_variance: update.observation_variance,
            sample_size: update.sample_size,
            mapping: update.mapping,
            evidence: update.evidence,
        };
        let ledger = ledger.ledger().clone();
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let inner = crate::detach_catch(py, || {
            self.inner.condition_experiment(request, &ledger, &ctx).map_err(panel_error)
        })?;
        Ok(PyDesignExperiment { inner })
    }
}
#[pyclass(name = "_JointDesignExperiment", frozen)]
pub(crate) struct PyDesignExperiment {
    inner: facade::JointLawExperimentResult,
}
#[pymethods]
impl PyDesignExperiment {
    #[pyo3(signature=(declaration_json,ledger,*,replicates=199,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn rank_next_samples(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        replicates: usize,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        crate::bridge::check_declared_size(
            declaration_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "future-study ranking",
        )?;
        let d: FutureDeclaration =
            serde_json::from_str(declaration_json).map_err(|e| crate::value_err(e.to_string()))?;
        if d.actions.is_empty()
            || d.actions.len() > 512
            || d.actions.iter().any(|a| {
                a.id.trim().is_empty()
                    || !a.intercept.is_finite()
                    || a.coefficients.iter().any(|v| !v.is_finite())
            })
        {
            return Err(crate::value_err("invalid bounded terminal action utility"));
        }
        if d.actions.iter().map(|a| &a.id).collect::<std::collections::BTreeSet<_>>().len()
            != d.actions.len()
        {
            return Err(crate::value_err("terminal action identities must be unique"));
        }
        let problem = antecedent_design::decision::DecisionProblem::new(
            d.actions,
            std::sync::Arc::new(Utility),
            vec![],
        );
        let candidates = d
            .candidates
            .into_iter()
            .map(|c| facade::PanelFutureSample {
                id: c.id,
                sample_size: c.sample_size,
                observation_variance: c.observation_variance,
                cost_utility: c.cost_utility,
            })
            .collect::<Vec<_>>();
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let ledger = ledger.ledger().clone();
        let rows = crate::detach_catch(py, || {
            self.inner
                .rank_next_samples(&problem, &candidates, replicates, &ledger, &ctx)
                .map_err(panel_error)
        })?;
        Ok(serde_json::json!({"scope":"model_conditional","calibration":"unmeasured","candidates":rows.iter().map(|r|serde_json::json!({"id":r.id,"gross_evsi":r.gross_evsi,"standard_error":r.standard_error,"net_value":r.net_value,"dependency_identity":r.dependency_identity.to_hex(),"question_identity":r.question_identity.to_hex(),"admissible_action_identity":r.admissible_action_identity.to_hex()})).collect::<Vec<_>>()}).to_string())
    }
    fn report(&self, ledger: PyRef<'_, EvidenceLedgerHandle>) -> PyResult<String> {
        let law = self.inner.numerical_law_source(ledger.ledger()).map_err(panel_error)?;
        let means = (0..law.shape()[1])
            .map(|i| law.mean(i).map_err(|e| crate::value_err(e.to_string())))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(serde_json::json!({"posterior_means":means,"posterior_weights":law.metadata().weights,"quantities":law.quantities(),"scope":"model_conditional_experimental_update;calibration_unmeasured","modular_policy":"frozen_joint_importance","calibration":"unmeasured"}).to_string())
    }
    fn decide(
        &self,
        contract_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
    ) -> PyResult<String> {
        let contract = antecedent_design::decision_artifact::contract_from_json(contract_json)
            .map_err(|e| crate::value_err(e.to_string()))?;
        let result = self.inner.decide(&contract, ledger.ledger()).map_err(panel_error)?;
        antecedent_design::decision_artifact::result_to_json(
            &result.result,
            self.inner.numerical_law_source(ledger.ledger()).map_err(panel_error)?,
        )
        .map_err(|e| crate::value_err(e.to_string()))
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
        let bytes =
            crate::detach_catch(py, || self.inner.export(&ledger, &ctx).map_err(panel_error))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
    #[pyo3(signature=(bytes,ledger,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn verify_current(
        &self,
        py: Python<'_>,
        bytes: &[u8],
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let ledger = ledger.ledger().clone();
        crate::detach_catch(py, || {
            self.inner.consume_current(bytes, &ledger, &ctx).map(|_| ()).map_err(panel_error)
        })
    }
}
#[pymethods]
impl super::designs::PyNativeJointDesignResult {
    fn experiment_quantity(&self, binding_json: &str) -> PyResult<String> {
        crate::bridge::check_declared_size(
            binding_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "native target projection",
        )?;
        let binding: facade::NativeExperimentBinding =
            serde_json::from_str(binding_json).map_err(|e| crate::value_err(e.to_string()))?;
        let quantity = self.inner.experiment_quantity(&binding).map_err(panel_error)?;
        serde_json::to_string(&antecedent_io::quantity_wire::ScientificQuantityWire::from(
            &quantity,
        ))
        .map_err(|e| crate::value_err(e.to_string()))
    }
    #[pyo3(signature=(binding_json,declaration_json,ledger,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn condition_native_experiment(
        &self,
        py: Python<'_>,
        binding_json: &str,
        declaration_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyNativeExperiment> {
        crate::bridge::check_declared_size(
            binding_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "native target projection",
        )?;
        let binding: facade::NativeExperimentBinding =
            serde_json::from_str(binding_json).map_err(|e| crate::value_err(e.to_string()))?;
        let update = parse_update(declaration_json)?;
        if !matches!(update.functional, facade::JointPanelResponseFunctional::TerminalMean) {
            return Err(crate::value_err(
                "native experiment retains the original method-native target; a time redefinition requires recompilation",
            ));
        }
        let request = facade::JointLawExperimentRequest {
            bridge: update.bridge,
            observed_mean: update.observed_mean,
            observation_variance: update.observation_variance,
            sample_size: update.sample_size,
            mapping: update.mapping,
            evidence: update.evidence,
        };
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let ledger = ledger.ledger().clone();
        let inner = crate::detach_catch(py, || {
            self.inner
                .condition_native_experiment(binding, request, &ledger, &ctx)
                .map_err(panel_error)
        })?;
        Ok(PyNativeExperiment { inner })
    }
}
#[pyclass(name = "_NativeJointExperiment", frozen)]
pub(crate) struct PyNativeExperiment {
    inner: facade::NativeJointExperimentResult,
}
#[pymethods]
impl PyNativeExperiment {
    #[pyo3(signature=(declaration_json,ledger,*,replicates=199,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn rank_next_samples(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        replicates: usize,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        crate::bridge::check_declared_size(
            declaration_json,
            crate::bridge::MAX_SPEC_DECLARATION_BYTES,
            "future-study ranking",
        )?;
        let d: FutureDeclaration =
            serde_json::from_str(declaration_json).map_err(|e| crate::value_err(e.to_string()))?;
        if d.actions.is_empty()
            || d.actions.len() > 512
            || d.actions.iter().any(|a| {
                a.id.trim().is_empty()
                    || !a.intercept.is_finite()
                    || a.coefficients.iter().any(|v| !v.is_finite())
            })
        {
            return Err(crate::value_err("invalid bounded terminal action utility"));
        }
        if d.actions.iter().map(|a| &a.id).collect::<std::collections::BTreeSet<_>>().len()
            != d.actions.len()
        {
            return Err(crate::value_err("terminal action identities must be unique"));
        }
        let problem = antecedent_design::decision::DecisionProblem::new(
            d.actions,
            std::sync::Arc::new(Utility),
            vec![],
        );
        let candidates = d
            .candidates
            .into_iter()
            .map(|c| facade::PanelFutureSample {
                id: c.id,
                sample_size: c.sample_size,
                observation_variance: c.observation_variance,
                cost_utility: c.cost_utility,
            })
            .collect::<Vec<_>>();
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let ledger = ledger.ledger().clone();
        let rows = crate::detach_catch(py, || {
            self.inner
                .rank_next_samples(&problem, &candidates, replicates, &ledger, &ctx)
                .map_err(panel_error)
        })?;
        Ok(serde_json::json!({"scope":"model_conditional","calibration":"unmeasured","candidates":rows.iter().map(|r|serde_json::json!({"id":r.id,"gross_evsi":r.gross_evsi,"standard_error":r.standard_error,"net_value":r.net_value,"dependency_identity":r.dependency_identity.to_hex(),"question_identity":r.question_identity.to_hex(),"admissible_action_identity":r.admissible_action_identity.to_hex()})).collect::<Vec<_>>()}).to_string())
    }
    fn report(&self, ledger: PyRef<'_, EvidenceLedgerHandle>) -> PyResult<String> {
        let law = self.inner.numerical_law_source(ledger.ledger()).map_err(panel_error)?;
        let means = (0..law.shape()[1])
            .map(|i| law.mean(i).map_err(|e| crate::value_err(e.to_string())))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(serde_json::json!({"posterior_means":means,"posterior_weights":law.metadata().weights,"quantities":law.quantities(),"scope":"model_conditional_experimental_update;calibration_unmeasured","identification":self.inner.identification(),"local_ratio_regularization":self.inner.local_ratio_regularization(),"modular_policy":"frozen_joint_importance","calibration":"unmeasured"}).to_string())
    }
    fn decide(
        &self,
        contract_json: &str,
        ledger: PyRef<'_, EvidenceLedgerHandle>,
    ) -> PyResult<String> {
        let contract = antecedent_design::decision_artifact::contract_from_json(contract_json)
            .map_err(|e| crate::value_err(e.to_string()))?;
        let result = self.inner.decide(&contract, ledger.ledger()).map_err(panel_error)?;
        antecedent_design::decision_artifact::result_to_json(
            &result.result,
            self.inner.numerical_law_source(ledger.ledger()).map_err(panel_error)?,
        )
        .map_err(|e| crate::value_err(e.to_string()))
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
        let bytes =
            crate::detach_catch(py, || self.inner.export(&ledger, &ctx).map_err(panel_error))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
    #[pyo3(signature=(bytes,ledger,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn verify_current(
        &self,
        py: Python<'_>,
        bytes: &[u8],
        ledger: PyRef<'_, EvidenceLedgerHandle>,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let ledger = ledger.ledger().clone();
        crate::detach_catch(py, || {
            self.inner.consume_current(bytes, &ledger, &ctx).map(|_| ()).map_err(panel_error)
        })
    }
}
#[pyfunction]
fn joint_panel_experiment_acceptance_matrix() -> PyResult<String> {
    serde_json::to_string(&facade::joint_experiment_acceptance_matrix())
        .map_err(|e| crate::value_err(e.to_string()))
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(joint_panel_experiment_acceptance_matrix, module)?)?;
    module.add_class::<PyPanelExperiment>()?;
    module.add_class::<PyDesignExperiment>()?;
    module.add_class::<PyNativeExperiment>()?;
    Ok(())
}
