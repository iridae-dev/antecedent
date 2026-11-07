//! Cyclic evidence handle and artifact consumption.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Bound, MAX_CYCLIC_ARTIFACT_BYTES, Py, PyBytes, PyModule, PyModuleMethods, PyRef, PyResult,
    Python, execution_context, input_admission, operation_error, pyclass, pyfunction, pymethods,
    replay_error, wrap_pyfunction,
};

#[pyclass(name = "_CyclicEvidence")]
pub(super) struct PyCyclicEvidence {
    pub(super) inner: antecedent::analysis::cyclic_evidence::CyclicEvidence,
}
#[pymethods]
impl PyCyclicEvidence {
    fn identity(&self) -> &str {
        self.inner.identity()
    }

    #[pyo3(signature=(*,offset=0,limit=64))]
    fn metadata_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        if limit > 256 {
            return Err(crate::value_err("cyclic evidence metadata page limit exceeds256"));
        }
        let columns = self.inner.columns();
        let end = offset.saturating_add(limit).min(columns.len());
        let page = columns
            .get(offset..end)
            .unwrap_or(&[])
            .iter()
            .map(|column| {
                serde_json::json!({
                    "quantity":column.quantity,"envelope_identity":column.envelope_identity,
                    "port":column.port,"uncertainty_axes":column.uncertainty_axes,
                    "world":column.world,"time":column.time,"sample_count":column.samples.len(),
                    "enclosure_count":column.numerical_enclosures.len(),
                    "mass_bound_count":column.mass_bounds.as_ref().map_or(0,Vec::len),
                })
            })
            .collect::<Vec<_>>();
        Ok(serde_json::json!({"identity":self.inner.identity(),"source_identity":self.inner.source_identity(),"scope":self.inner.scope(),"source_bridge":self.inner.system_bridge(),"recovery_origins":self.inner.report().recovery_origins.as_ref(),"column_count":columns.len(),"offset":offset,"columns":page,"next_offset":(end<columns.len()).then_some(end)}).to_string())
    }

    fn report(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.report())
            .map_err(|e| crate::serialization_err(e.to_string()))
    }
    #[pyo3(signature=(*,max_bytes=None,cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        let ctx = execution_context(max_bytes, cancel);
        let bytes = py.detach(|| self.inner.export(&ctx)).map_err(|e| operation_error(e, &ctx))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
    #[pyo3(signature=(*,others=None,dependence=None,max_bytes=None,cancel=None))]
    fn joint_law(
        &self,
        others: Option<Vec<PyRef<'_, PyCyclicEvidence>>>,
        dependence: Option<&str>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<crate::distribution_api::PyJointDistributionArtifact> {
        let ctx = execution_context(max_bytes, cancel);
        if ctx.cancellation.is_cancelled() {
            return Err(crate::CausalCancelledError::new_err("cancelled"));
        }
        if dependence.is_some_and(|value| !matches!(value, "aligned" | "independent")) {
            return Err(crate::value_err("dependence must be aligned or independent"));
        }
        let others = others.unwrap_or_default();
        if others.len() > 256 {
            return Err(crate::value_err("at most 256 additional evidence products"));
        }
        let references = others.iter().map(|item| &item.inner).collect::<Vec<_>>();
        let samples = std::iter::once(&self.inner)
            .chain(references.iter().copied())
            .flat_map(|item| item.report().columns.iter())
            .try_fold(0_u64, |sum, column| sum.checked_add(column.samples.len() as u64))
            .ok_or_else(|| {
                crate::CausalResourceError::new_err("joint law sample count overflow")
            })?;
        let _lease = antecedent::execution::retain(
            &ctx,
            samples.checked_mul(32).ok_or_else(|| {
                crate::CausalResourceError::new_err("joint law allocation overflow")
            })?,
        )
        .map_err(|error| operation_error(error, &ctx))?;
        let artifact = self
            .inner
            .joint_law_with(&references, dependence)
            .map_err(|error| operation_error(error, &ctx))?;
        Ok(crate::distribution_api::PyJointDistributionArtifact { artifact, source_claim: None })
    }

    #[pyo3(signature=(contract_json, *, max_bytes=None, cancel=None))]
    fn decide_enclosures(
        &self,
        contract_json: &str,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = execution_context(max_bytes, cancel);
        let _input = input_admission(contract_json, &ctx)?;
        let sample_slots = self
            .inner
            .report()
            .columns
            .iter()
            .map(|column| column.samples.len() as u64)
            .sum::<u64>();
        let _workspace = antecedent::execution::retain(
            &ctx,
            sample_slots
                .saturating_mul(32)
                .saturating_add((contract_json.len() as u64).saturating_mul(8)),
        )
        .map_err(|e| operation_error(e, &ctx))?;
        if contract_json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(crate::value_err("decision contract byte limit"));
        }
        let contract = antecedent::design::decision_artifact::contract_from_json(contract_json)
            .map_err(|e| operation_error(e, &ctx))?;
        antecedent::execution::run_admitted(&ctx, || self.inner.decide_enclosures(&contract))
            .map(|v| v.to_string())
            .map_err(|e| operation_error(e, &ctx))
    }
    #[pyo3(signature=(contract_json, *, max_bytes=None, cancel=None))]
    fn decide_means(
        &self,
        contract_json: &str,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = execution_context(max_bytes, cancel);
        let _input = input_admission(contract_json, &ctx)?;
        let sample_slots = self
            .inner
            .report()
            .columns
            .iter()
            .map(|column| column.samples.len() as u64)
            .sum::<u64>();
        let _workspace = antecedent::execution::retain(
            &ctx,
            sample_slots
                .saturating_mul(32)
                .saturating_add((contract_json.len() as u64).saturating_mul(8)),
        )
        .map_err(|e| operation_error(e, &ctx))?;
        if contract_json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(crate::value_err("decision contract byte limit"));
        }
        let contract = antecedent::design::decision_artifact::contract_from_json(contract_json)
            .map_err(|e| operation_error(e, &ctx))?;
        antecedent::execution::run_admitted(&ctx, || self.inner.decide_means(&contract))
            .map(|v| v.to_string())
            .map_err(|e| operation_error(e, &ctx))
    }
}
#[pyfunction]
#[pyo3(signature=(bytes,expected_identity,*,max_bytes=None,cancel=None))]
fn consume_cyclic_evidence(
    py: Python<'_>,
    bytes: &[u8],
    expected_identity: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyCyclicEvidence> {
    let ctx = execution_context(max_bytes, cancel);
    let inner = py
        .detach(|| {
            antecedent::analysis::cyclic_evidence::CyclicEvidence::consume(
                bytes,
                expected_identity,
                &ctx,
            )
        })
        .map_err(|e| replay_error(e, &ctx))?;
    Ok(PyCyclicEvidence { inner })
}

pub(super) fn register_consumer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(consume_cyclic_evidence, m)?)
}
