//! Python access to actual design-native joint identification and posterior laws.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{context, panel_error};
use antecedent::analysis::joint_panel::designs::{
    JointDesignProof, JointDesignResult, PreparedJointDesign, replay_joint_design_artifact,
};
use antecedent_estimate::joint_design::compiled::JointDesignInference;
use pyo3::{prelude::*, types::PyBytes};

fn bounded_json(text: &str, description: &str) -> PyResult<()> {
    crate::bridge::check_declared_size(text, crate::bridge::MAX_SPEC_DECLARATION_BYTES, description)
}
/// Native prepared design retains exact original data and complete program definitions.
#[pyclass(name = "_PreparedJointDesign", frozen)]
pub(super) struct PyPreparedJointDesign {
    pub(super) inner: PreparedJointDesign,
}
#[pymethods]
impl PyPreparedJointDesign {
    #[pyo3(signature=(inference_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        inference_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyJointDesignResult> {
        bounded_json(inference_json, "joint-design statistical procedure")?;
        let inference: JointDesignInference =
            serde_json::from_str(inference_json).map_err(|e| crate::value_err(e.to_string()))?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let inner =
            crate::detach_catch(py, || self.inner.execute(inference, &ctx).map_err(panel_error))?;
        Ok(PyJointDesignResult { inner })
    }
    fn proof(&self) -> PyResult<String> {
        serde_json::to_string(&serde_json::json!({"version":self.inner.proof().version,"recipe":self.inner.proof().recipe,"worlds":self.inner.proof().worlds,"functional":self.inner.proof().recipe.functional(),"dependency":self.inner.dependency().to_hex(),"identified":self.inner.identification().is_identified_claim()})).map_err(|e|crate::value_err(e.to_string()))
    }
    #[pyo3(signature=(artifact,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn consume_current(
        &self,
        py: Python<'_>,
        artifact: &[u8],
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyJointDesignResult> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let inner = crate::detach_catch(py, || {
            self.inner.consume_current(artifact, &ctx).map_err(panel_error)
        })?;
        Ok(PyJointDesignResult { inner })
    }
}
/// Native inference result retains the same scientific proof across statistical modes.
#[pyclass(name = "_JointDesignResult", frozen)]
pub(super) struct PyJointDesignResult {
    pub(super) inner: JointDesignResult,
}
#[pymethods]
impl PyJointDesignResult {
    #[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn export_reference_decomposition<'py>(
        &self,
        py: Python<'py>,
        declaration_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let bytes = crate::detach_catch(py, || {
            population::export_components(&self.inner, declaration_json, &ctx)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[pyo3(signature=(artifact,declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn consume_reference_decomposition_current(
        &self,
        py: Python<'_>,
        artifact: &[u8],
        declaration_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        crate::detach_catch(py, || {
            population::consume_components(&self.inner, artifact, declaration_json, &ctx)
        })
    }
    #[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn decompose_population_reference(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        crate::detach_catch(py, || population::execute(&self.inner, declaration_json, &ctx))
    }
    #[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn export_population_reference<'py>(
        &self,
        py: Python<'py>,
        declaration_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let bytes =
            crate::detach_catch(py, || population::export(&self.inner, declaration_json, &ctx))?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[pyo3(signature=(artifact,declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn consume_population_reference_current(
        &self,
        py: Python<'_>,
        artifact: &[u8],
        declaration_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        crate::detach_catch(py, || {
            population::consume(&self.inner, artifact, declaration_json, &ctx)
        })
    }
    #[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn decompose_reference_components(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Bindings {
            exposure: Vec<u32>,
            saturation: Vec<u32>,
            mechanism: Vec<u32>,
            hybrids: [u32; 16],
            order: [antecedent::analysis::joint_panel::ChangeComponent; 4],
            #[serde(default = "identified_default")]
            require_identified: bool,
        }
        fn identified_default() -> bool {
            true
        }
        bounded_json(declaration_json, "joint-design component decomposition")?;
        let declaration: Bindings =
            serde_json::from_str(declaration_json).map_err(|e| crate::value_err(e.to_string()))?;
        let bindings = antecedent::analysis::joint_panel::ReferenceComponentBindings {
            exposure: declaration
                .exposure
                .into_iter()
                .map(antecedent_core::VariableId::from_raw)
                .collect(),
            saturation: declaration
                .saturation
                .into_iter()
                .map(antecedent_core::VariableId::from_raw)
                .collect(),
            mechanism: declaration
                .mechanism
                .into_iter()
                .map(antecedent_core::VariableId::from_raw)
                .collect(),
            hybrids: declaration.hybrids.map(antecedent_core::ScientificWorldId::from_raw),
            order: declaration.order,
        };
        let scope = if declaration.require_identified {
            antecedent_design::scientific_source::ScientificConsumptionScope::IdentifiedCausal
        } else {
            antecedent_design::scientific_source::ScientificConsumptionScope::ModelConditional
        };
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let result = crate::detach_catch(py, || {
            self.inner.decompose_reference_components(&bindings, scope, &ctx).map_err(panel_error)
        })?;
        serde_json::to_string(&serde_json::json!({"decomposition":{"point":result.decomposition.point,"draws":result.decomposition.draws,"weights":result.decomposition.weights},"full_procedure_bootstrap":!result.procedure_failures.is_empty(),"procedure_failures":result.procedure_failures,"covariance":result.covariance,"joint_uncertainty":result.decomposition.joint_uncertainty(),"question_identity":result.question_identity.to_hex(),"scope":if declaration.require_identified{"identified"}else{"model_conditional"},"dependency":result.source.dependency_identity().to_hex(),"identification":result.source.identification().identity().to_hex()})).map_err(|e|crate::value_err(e.to_string()))
    }
    fn report(&self) -> PyResult<String> {
        serde_json::to_string(&serde_json::json!({"law":self.inner.law(),"dependency":self.inner.dependency_identity().to_hex(),"source_dependency":self.inner.prepared().dependency().to_hex(),"identified":self.inner.prepared().identification().is_identified_claim(),"functional":self.inner.prepared().proof().recipe.functional()})).map_err(|e|crate::value_err(e.to_string()))
    }
    fn export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.export_artifact().map_err(panel_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn posterior_quantities(&self) -> PyResult<String> {
        let source = self.inner.posterior_source().map_err(panel_error)?;
        let antecedent_design::scientific_source::ScientificSourcePayload::Law(law) =
            source.as_numerical_source()
        else {
            return Err(crate::value_err("joint-design posterior source has no law"));
        };
        serde_json::to_string(law.quantities()).map_err(|e| crate::value_err(e.to_string()))
    }
}
#[pyfunction]
#[pyo3(signature=(target_json,proof_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn prepare_joint_design(
    py: Python<'_>,
    target_json: &str,
    proof_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedJointDesign> {
    bounded_json(target_json, "joint-design scientific target")?;
    bounded_json(proof_json, "joint-design proof")?;
    let target: antecedent_io::scientific_target_wire::ScientificTargetWire =
        serde_json::from_str(target_json).map_err(|e| crate::value_err(e.to_string()))?;
    let target = target.to_domain().map_err(|e| crate::value_err(e.to_string()))?;
    let proof: JointDesignProof =
        serde_json::from_str(proof_json).map_err(|e| crate::value_err(e.to_string()))?;
    let ctx = context(seed, threads, max_bytes, cancel)?;
    let inner = crate::detach_catch(py, || {
        PreparedJointDesign::compile(target, proof, &ctx).map_err(panel_error)
    })?;
    Ok(PyPreparedJointDesign { inner })
}
#[pyfunction]
#[pyo3(signature=(panel_result,proof_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn prepare_joint_design_for_panel(
    py: Python<'_>,
    panel_result: &super::PyJointPanelResult,
    proof_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedJointDesign> {
    bounded_json(proof_json, "joint-design proof")?;
    let proof: JointDesignProof =
        serde_json::from_str(proof_json).map_err(|e| crate::value_err(e.to_string()))?;
    let target = panel_result.inner.request().target.clone();
    let ctx = context(seed, threads, max_bytes, cancel)?;
    let inner = crate::detach_catch(py, || {
        PreparedJointDesign::compile(target, proof, &ctx).map_err(panel_error)
    })?;
    Ok(PyPreparedJointDesign { inner })
}
#[pyfunction]
#[pyo3(signature=(artifact,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn replay_joint_design(
    py: Python<'_>,
    artifact: &[u8],
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    let law = crate::detach_catch(py, || {
        replay_joint_design_artifact(artifact, &ctx).map_err(panel_error)
    })?;
    serde_json::to_string(&law).map_err(|e| crate::value_err(e.to_string()))
}
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    population::register(m)?;
    m.add_class::<PyPreparedNativeJointDesign>()?;
    m.add_class::<PyNativeJointDesignResult>()?;
    m.add_function(wrap_pyfunction!(prepare_native_joint_design, m)?)?;
    m.add_function(wrap_pyfunction!(replay_native_joint_design, m)?)?;
    m.add_class::<PyPreparedJointDesign>()?;
    m.add_class::<PyJointDesignResult>()?;
    m.add_function(wrap_pyfunction!(prepare_joint_design, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_joint_design_for_panel, m)?)?;
    m.add_function(wrap_pyfunction!(replay_joint_design, m)?)?;
    Ok(())
}

#[pyclass(name = "_PreparedNativeJointDesign", frozen)]
struct PyPreparedNativeJointDesign {
    inner: antecedent::analysis::joint_panel::designs::PreparedNativeJointDesign,
}
#[pymethods]
impl PyPreparedNativeJointDesign {
    fn identification(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.identification())
            .map_err(|e| crate::value_err(e.to_string()))
    }
    #[pyo3(signature=(inference_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        inference_json: &str,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyNativeJointDesignResult> {
        bounded_json(inference_json, "native joint-design inference")?;
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let _input = ctx
            .acquire_admission(0, (inference_json.len() as u64).saturating_mul(128))
            .map_err(|e| crate::CausalResourceError::new_err(e.to_string()))?;
        let inference: JointDesignInference =
            serde_json::from_str(inference_json).map_err(|e| crate::value_err(e.to_string()))?;
        let inner =
            crate::detach_catch(py, || self.inner.execute(inference, &ctx).map_err(panel_error))?;
        Ok(PyNativeJointDesignResult { inner })
    }
    #[pyo3(signature=(artifact,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn consume_current(
        &self,
        py: Python<'_>,
        artifact: &[u8],
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyNativeJointDesignResult> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let inner = crate::detach_catch(py, || {
            self.inner.consume_current(artifact, &ctx).map_err(panel_error)
        })?;
        Ok(PyNativeJointDesignResult { inner })
    }
}
#[pyclass(name = "_NativeJointDesignResult", frozen)]
pub(super) struct PyNativeJointDesignResult {
    pub(super) inner: antecedent::analysis::joint_panel::designs::NativeJointDesignResult,
}
#[pymethods]
impl PyNativeJointDesignResult {
    fn report(&self) -> PyResult<String> {
        serde_json::to_string(&serde_json::json!({"identification":self.inner.prepared().identification(),"law":self.inner.law(),"dependency":self.inner.dependency_identity().to_hex(),"source_dependency":self.inner.prepared().dependency().to_hex()})).map_err(|e|crate::value_err(e.to_string()))
    }
    fn export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.export_artifact().map_err(panel_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
}
#[pyfunction]
#[pyo3(signature=(recipe_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn prepare_native_joint_design(
    py: Python<'_>,
    recipe_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedNativeJointDesign> {
    bounded_json(recipe_json, "native joint-design source recipe")?;
    let ctx = context(seed, threads, max_bytes, cancel)?;
    let _input = ctx
        .acquire_admission(0, (recipe_json.len() as u64).saturating_mul(128))
        .map_err(|e| crate::CausalResourceError::new_err(e.to_string()))?;
    let recipe: antecedent_estimate::joint_design::compiled::JointDesignRecipe =
        serde_json::from_str(recipe_json).map_err(|e| crate::value_err(e.to_string()))?;
    let inner = crate::detach_catch(py, || {
        antecedent::analysis::joint_panel::designs::PreparedNativeJointDesign::compile(recipe, &ctx)
            .map_err(panel_error)
    })?;
    Ok(PyPreparedNativeJointDesign { inner })
}
#[pyfunction]
#[pyo3(signature=(artifact,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn replay_native_joint_design(
    py: Python<'_>,
    artifact: &[u8],
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    let law = crate::detach_catch(py, || {
        antecedent::analysis::joint_panel::designs::replay_native_joint_design(artifact, &ctx)
            .map_err(panel_error)
    })?;
    serde_json::to_string(&law).map_err(|e| crate::value_err(e.to_string()))
}

mod population;
