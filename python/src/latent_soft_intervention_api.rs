//! Python bridge for native compiler-owned latent soft-intervention discovery.

use antecedent::Study;
use antecedent::analysis::latent_soft_intervention::{
    FittedLatentSoftIntervention, LatentSoftInterventionRequest, PreparedLatentSoftIntervention,
    replay_forecast,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use crate::errors::py_err;
use crate::transport_common::execution_context;
use crate::{PyCancellationToken, detach_catch};

fn encode(value: &impl serde::Serialize) -> PyResult<String> {
    serde_json::to_string(value).map_err(|error| crate::errors::value_err(error.to_string()))
}

#[pyclass(name = "NativePreparedLatentSoftIntervention")]
struct NativePreparedLatentSoftIntervention {
    inner: PreparedLatentSoftIntervention,
}

#[pymethods]
impl NativePreparedLatentSoftIntervention {
    fn inspect(&self) -> String {
        self.inner.plan_view().to_json().to_string()
    }

    fn unit_support(&self) -> Option<String> {
        crate::errors::unit_support_json(&self.inner.plan_view())
    }

    #[pyo3(signature = (memory_bytes=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        memory_bytes: Option<u64>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<NativeFittedLatentSoftIntervention> {
        let ctx = execution_context(0, memory_bytes, cancel);
        let inner = detach_catch(py, || self.inner.execute(&ctx).map_err(py_err))?;
        Ok(NativeFittedLatentSoftIntervention { inner })
    }
}

#[pyclass(name = "NativeFittedLatentSoftIntervention")]
struct NativeFittedLatentSoftIntervention {
    inner: FittedLatentSoftIntervention,
}

#[pymethods]
impl NativeFittedLatentSoftIntervention {
    fn discovery(&self) -> PyResult<String> {
        encode(self.inner.discovery())
    }

    #[pyo3(signature = (context, memory_bytes=None, cancel=None))]
    fn forecast(
        &self,
        py: Python<'_>,
        context: Vec<f64>,
        memory_bytes: Option<u64>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = execution_context(0, memory_bytes, cancel);
        let result = detach_catch(py, || self.inner.forecast(&context, &ctx).map_err(py_err))?;
        encode(&result)
    }

    #[pyo3(signature = (context, memory_bytes=None, cancel=None))]
    fn export_forecast<'py>(
        &self,
        py: Python<'py>,
        context: Vec<f64>,
        memory_bytes: Option<u64>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<(Bound<'py, PyBytes>, String)> {
        let ctx = execution_context(0, memory_bytes, cancel);
        let artifact =
            detach_catch(py, || self.inner.export_forecast(&context, &ctx).map_err(py_err))?;
        Ok((PyBytes::new(py, &artifact.bytes), artifact.expectation))
    }
}

#[pyfunction]
#[pyo3(signature = (request, memory_bytes=None, cancel=None))]
fn latent_soft_intervention_prepare(
    py: Python<'_>,
    request: &str,
    memory_bytes: Option<u64>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<NativePreparedLatentSoftIntervention> {
    if request.len() > 48 * 1024 * 1024 {
        return Err(crate::errors::value_err(
            "latent soft-intervention request exceeds byte limit",
        ));
    }
    let ctx = execution_context(0, memory_bytes, cancel);
    let inner = detach_catch(py, || {
        let request: LatentSoftInterventionRequest = serde_json::from_str(request)
            .map_err(|error| crate::errors::value_err(error.to_string()))?;
        Study::prepare_latent_soft_intervention(request, &ctx).map_err(py_err)
    })?;
    Ok(NativePreparedLatentSoftIntervention { inner })
}

#[pyfunction]
#[pyo3(signature = (data, expected, memory_bytes=None, cancel=None))]
fn latent_soft_intervention_replay(
    py: Python<'_>,
    data: &[u8],
    expected: &str,
    memory_bytes: Option<u64>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<String> {
    let ctx = execution_context(0, memory_bytes, cancel);
    let replayed = detach_catch(py, || replay_forecast(data, expected, &ctx).map_err(py_err))?;
    encode(&replayed)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePreparedLatentSoftIntervention>()?;
    module.add_class::<NativeFittedLatentSoftIntervention>()?;
    module.add_function(wrap_pyfunction!(latent_soft_intervention_prepare, module)?)?;
    module.add_function(wrap_pyfunction!(latent_soft_intervention_replay, module)?)?;
    Ok(())
}
