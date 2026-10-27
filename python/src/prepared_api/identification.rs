//! Native owners for bounded prepared-identification inspection.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::PreparedStudy;
use antecedent::analysis::PreparedIdentificationView;
use pyo3::prelude::*;

use super::PyPreparedAnalysis;

const MAX_PATH_DEPTH: usize = 32;
const MAX_PAGE_SIZE: usize = 256;

/// Immutable identification node sharing its exact prepared native owner.
#[pyclass(name = "PreparedIdentificationView", frozen)]
pub(super) struct PyPreparedIdentificationView {
    owner: Arc<PreparedStudy>,
    path: Vec<usize>,
}

impl PyPreparedIdentificationView {
    fn resolve(&self) -> PyResult<PreparedIdentificationView<'_>> {
        let mut view = self.owner.identification_view();
        for &index in &self.path {
            view = view
                .child(index)
                .ok_or_else(|| crate::value_err("invalid retained identification coordinate"))?;
        }
        Ok(view)
    }

    fn selected_child(&self, index: usize) -> PyResult<Self> {
        if self.path.len() >= MAX_PATH_DEPTH {
            return Err(crate::value_err("identification coordinate exceeds maximum depth"));
        }
        if self.resolve()?.child(index).is_none() {
            return Err(crate::value_err("identification child index is out of range"));
        }
        let mut path = self.path.clone();
        path.push(index);
        Ok(Self { owner: Arc::clone(&self.owner), path })
    }
}

/// Explicit materialization owner. No live preparation or rows are retained.
#[pyclass(name = "_IdentificationMaterialization", frozen)]
pub(super) struct PyIdentificationMaterialization {
    inner: antecedent_io_core::admitted_json::AdmittedJson,
    cancellation: antecedent_core::CancellationToken,
}

impl PyIdentificationMaterialization {
    fn check_cancelled(&self) -> PyResult<()> {
        if self.cancellation.is_cancelled() {
            return Err(crate::py_err(antecedent::CausalError::Cancelled {
                stage: "identification_materialize",
            }));
        }
        Ok(())
    }
}

#[pymethods]
impl PyIdentificationMaterialization {
    /// Requested Python conversion. Keep the native output admission alive
    /// throughout decoding; the Python dict is the caller-owned materialization.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.check_cancelled()?;
        let value = py.import("json")?.call_method1("loads", (self.inner.as_str(),))?;
        self.check_cancelled()?;
        Ok(value.unbind())
    }

    fn retained_bytes(&self) -> usize { self.inner.retained_bytes() }
    fn encoded_bytes(&self) -> usize { self.inner.as_bytes().len() }

    fn __repr__(&self) -> String {
        format!("IdentificationMaterialization(encoded_bytes={}, retained_bytes={})",
            self.inner.as_bytes().len(), self.inner.retained_bytes())
    }
}

#[pymethods]
impl PyPreparedIdentificationView {
    /// Fixed-size retained metadata, without proof conversion or scientific work.
    fn summary_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.resolve()?.summary())
            .map_err(|error| crate::value_err(error.to_string()))
    }

    /// A child borrows the same frozen proof; no siblings are constructed.
    fn child(&self, index: usize) -> PyResult<Self> {
        self.selected_child(index)
    }

    /// Bounded navigation over the native child index; never enumerates before skipping.
    #[pyo3(signature = (*, offset=0, limit=32))]
    fn children(&self, offset: usize, limit: usize) -> PyResult<Vec<Self>> {
        if limit > MAX_PAGE_SIZE {
            return Err(crate::value_err("identification page limit must be at most 256"));
        }
        let count = self.resolve()?.child_count();
        (offset..offset.saturating_add(limit).min(count))
            .map(|index| self.selected_child(index))
            .collect()
    }

    /// Bounded native estimand descriptions on an explicitly selected proof.
    #[pyo3(signature = (*, offset=0, limit=32))]
    fn estimands_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        let page = self.resolve()?.estimands_page(offset, limit).map_err(crate::py_err)?;
        serde_json::to_string(&page).map_err(|error| crate::value_err(error.to_string()))
    }

    /// Native named adjustment coordinates; multiple alternatives require selection.
    #[pyo3(signature = (*, estimand=None, offset=0, limit=32))]
    fn adjustment_json(
        &self,
        estimand: Option<usize>,
        offset: usize,
        limit: usize,
    ) -> PyResult<String> {
        let page =
            self.resolve()?.adjustment_page(estimand, offset, limit).map_err(crate::py_err)?;
        serde_json::to_string(&page).map_err(|error| crate::value_err(error.to_string()))
    }

    /// Ordered native assumptions on this proof or a chosen estimand's exact claim.
    #[pyo3(signature = (*, estimand=None, offset=0, limit=32))]
    fn assumptions_json(
        &self,
        estimand: Option<usize>,
        offset: usize,
        limit: usize,
    ) -> PyResult<String> {
        let page =
            self.resolve()?.assumptions_page(estimand, offset, limit).map_err(crate::py_err)?;
        serde_json::to_string(&page).map_err(|error| crate::value_err(error.to_string()))
    }

    /// Ordered original native proof steps; no full certificate conversion.
    #[pyo3(signature = (*, offset=0, limit=32))]
    fn derivation_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        let page = self.resolve()?.derivation_page(offset, limit).map_err(crate::py_err)?;
        serde_json::to_string(&page).map_err(|error| crate::value_err(error.to_string()))
    }

    /// Explicit full conversion of this proof subtree. Cancellation is checked
    /// between nodes. This is a materialization boundary, never notebook inspection.
    #[pyo3(signature = (*, seed=1, threads=None))]
    fn materialize(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<PyIdentificationMaterialization> {
        let owner = Arc::clone(&self.owner);
        let path = self.path.clone();
        crate::detach_catch(py, move || {
            let context = crate::py_execution_context(seed, crate::resolve_user_threads(threads));
            let mut view = owner.identification_view();
            for index in path {
                view = view.child(index).ok_or_else(|| {
                    crate::value_err("invalid retained identification coordinate")
                })?;
            }
            let inner = view.materialize_admitted(&context).map_err(crate::py_err)?;
            Ok(PyIdentificationMaterialization { inner, cancellation: context.cancellation.clone() })
        })
    }

    /// Explicit schema-name conversion, separate from bounded summary rendering.
    fn schema_names(&self) -> Vec<String> {
        self.owner.schema().variables().iter().map(|variable| variable.name.to_string()).collect()
    }

    fn __repr__(&self) -> PyResult<String> {
        let summary = self.resolve()?.summary();
        Ok(format!(
            "PreparedIdentificationView(kind={}, children={})",
            summary["kind"], summary["children"]
        ))
    }
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Bounded native prepared identity and four-slot projection.
    fn prepared_inspection_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.prepared_inspection().map_err(crate::py_err)?.summary())
            .map_err(|error| crate::value_err(error.to_string()))
    }

    /// Read original obligations, preserving their native order and scope.
    #[pyo3(signature = (*, offset=0, limit=32))]
    fn prepared_obligations_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        let page = self
            .inner
            .prepared_inspection()
            .map_err(crate::py_err)?
            .obligations_page(offset, limit)
            .map_err(crate::py_err)?;
        serde_json::to_string(&page).map_err(|error| crate::value_err(error.to_string()))
    }

    /// Read original uncertainty components without a numerical execution.
    #[pyo3(signature = (*, offset=0, limit=32))]
    fn prepared_uncertainty_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        let page = self
            .inner
            .prepared_inspection()
            .map_err(crate::py_err)?
            .uncertainty_page(offset, limit)
            .map_err(crate::py_err)?;
        serde_json::to_string(&page).map_err(|error| crate::value_err(error.to_string()))
    }

    /// Explicit bounded page of the original proof-derived evidence obligations.
    #[pyo3(signature = (*, offset=0, limit=32))]
    fn prepared_evidence_obligations_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        if limit > 256 {
            return Err(crate::value_err("inspection page limit must be at most 256"));
        }
        let obligations =
            self.inner.prepared_inspection().map_err(crate::py_err)?.evidence_obligations();
        let end = offset.saturating_add(limit).min(obligations.len());
        let items = obligations
            .get(offset..end)
            .unwrap_or(&[])
            .iter()
            .map(|o| crate::repair_api::obligation_json(o, &self.names))
            .collect::<Vec<_>>();
        serde_json::to_string(&serde_json::json!({"offset":offset,"total":obligations.len(),
            "items":items,"next_offset":(end<obligations.len()).then_some(end)}))
        .map_err(|error| crate::value_err(error.to_string()))
    }

    /// Explicit bounded page of original native evidence-gathering actions.
    #[pyo3(signature = (*, offset=0, limit=32))]
    fn prepared_evidence_actions_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        if limit > 256 {
            return Err(crate::value_err("inspection page limit must be at most 256"));
        }
        let actions = self.inner.prepared_inspection().map_err(crate::py_err)?.evidence_actions();
        let end = offset.saturating_add(limit).min(actions.len());
        let items = actions
            .get(offset..end)
            .unwrap_or(&[])
            .iter()
            .map(|a| serde_json::json!({"id":a.id,"description":a.description}))
            .collect::<Vec<_>>();
        serde_json::to_string(&serde_json::json!({"offset":offset,"total":actions.len(),
            "items":items,"next_offset":(end<actions.len()).then_some(end)}))
        .map_err(|error| crate::value_err(error.to_string()))
    }

    /// Retain the native proof owner without compilation, fitting or JSON conversion.
    fn identification_view(&self) -> PyPreparedIdentificationView {
        PyPreparedIdentificationView { owner: Arc::clone(&self.inner), path: Vec::new() }
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIdentificationMaterialization>()?;
    m.add_class::<PyPreparedIdentificationView>()
}
