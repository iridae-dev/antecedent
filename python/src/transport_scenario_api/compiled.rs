//! The compiled finite transport scenario set.
//!
//! [`compile_transport_scenarios_stage`] binds the decided set of a prepared scenario stage into
//! the `TransportScenarioSet` program family (`Study::compile_transport_scenarios`).
//! [`NativeCompiledTransportScenarios::execute`] evaluates the retained plans through the program,
//! which runs the kernel the stage's `estimate` runs, and reports them in the stage's report
//! shape; refreshing with new laws or samples keeps the decisions and compiles the program again
//! over the new source. The export and the refusal of an aggregate interval are the stage's.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use antecedent::analysis::compiled_transport_scenarios::{
    CompiledTransportScenarios, TransportScenariosRun,
};
use pyo3::prelude::*;

use super::{
    PreparedTransportScenariosStage, Provider, SCENARIO_PREFIX, parse_provider, report_json,
    scenario_error,
};
use crate::graphs::Admg;
use crate::transport_common::{error, execution_context, frame_named_artifact};

/// A decided scenario set bound to its lowered program.
#[pyclass(name = "NativeCompiledTransportScenarios", skip_from_py_object)]
pub(crate) struct NativeCompiledTransportScenarios {
    inner: CompiledTransportScenarios,
    graph: Admg,
    catalog: antecedent::query::EvidenceCatalog,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    last: Option<TransportScenariosRun>,
}

#[pymethods]
impl NativeCompiledTransportScenarios {
    /// The plan route of the compiled program.
    #[getter]
    fn route(&self) -> String {
        self.inner.plan_view().route().as_str().to_owned()
    }

    /// Identity of the lowered program, as lowercase hex.
    #[getter]
    fn program_identity(&self) -> String {
        self.inner.lowered().identity().to_hex()
    }

    /// Evaluate every retained plan through the program; the report keeps every scenario and
    /// is the stage's report for the same decided set and laws.
    #[pyo3(signature=(*, memory_bytes=None, cancel=None))]
    fn execute(
        &mut self,
        py: Python<'_>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = execution_context(0, memory_bytes.or(self.memory_bytes), cancel);
        let inner = self.inner.clone();
        let run = crate::detach_catch(py, move || inner.execute(&ctx).map_err(error))?;
        let payload = report_json(&run.report, &self.graph.names)?.to_string();
        self.last = Some(run);
        Ok(payload)
    }

    /// Recompile the identified scenarios against new laws, or refit the empirical plug-in
    /// from new samples, and compile the program again; decisions are kept.
    #[pyo3(signature=(laws, *, memory_bytes=None, cancel=None))]
    fn refresh(
        &self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Self> {
        let provider = parse_provider(laws, &self.catalog, &self.graph, self.max_support_rows)?;
        let ctx = execution_context(0, memory_bytes.or(self.memory_bytes), cancel);
        let prepared = self.inner.prepared().clone();
        let cells = self.max_support_rows;
        let refreshed = crate::detach_catch(py, move || {
            match provider {
                Provider::Exact(data) => prepared.refresh(data, &ctx),
                Provider::Empirical(input) => prepared.refresh_empirical(input, cells, &ctx),
            }
            .map_err(scenario_error)
        })?;
        Ok(Self {
            inner: Study::compile_transport_scenarios(refreshed).map_err(error)?,
            graph: self.graph.clone(),
            catalog: self.catalog.clone(),
            max_support_rows: self.max_support_rows,
            memory_bytes: self.memory_bytes,
            last: None,
        })
    }

    /// Each scenario's retained, frozen plan: `compiled`, the provider or support refusal of an
    /// identified scenario, or `not_identified:<status>`.
    fn plan_summary(&self) -> Vec<(String, String)> {
        self.inner
            .prepared()
            .prepared()
            .plan_summary()
            .into_iter()
            .map(|(n, k)| (n.to_string(), k))
            .collect()
    }

    /// Inference across scenarios sharing data is not supported; always refuses.
    fn aggregate_interval(&self) -> PyResult<()> {
        self.inner.prepared().aggregate_interval().map_err(error)
    }

    /// The last executed report as a framed, independently consumable artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let run = self.last.as_ref().ok_or_else(|| {
            crate::refusal(
                antecedent::reason::reason_code!("not_executed"),
                "transport.no_execution_claim: execute before exporting a scenario artifact",
            )
        })?;
        let raw = self.inner.prepared().export(&run.report).map_err(error)?;
        let framed = frame_named_artifact(SCENARIO_PREFIX, &self.graph.names, raw)?;
        Ok(pyo3::types::PyBytes::new(py, &framed).unbind())
    }
}

/// Bind the decided set of a prepared scenario stage into its compiled program.
#[pyfunction]
pub(super) fn compile_transport_scenarios_stage(
    stage: PyRef<'_, PreparedTransportScenariosStage>,
) -> PyResult<NativeCompiledTransportScenarios> {
    let inner = Study::compile_transport_scenarios(stage.inner.clone()).map_err(error)?;
    Ok(NativeCompiledTransportScenarios {
        inner,
        graph: stage.graph.clone(),
        catalog: stage.catalog.clone(),
        max_support_rows: stage.max_support_rows,
        memory_bytes: stage.memory_bytes,
        last: None,
    })
}
