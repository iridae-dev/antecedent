//! Native retained-source bridge for learned RPCMCI.

use antecedent::analysis::rpcmci_portable::{
    LearnedRpcmciArtifact, LearnedRpcmciRequest, RpcmciReplayError, consume_learned_rpcmci,
};
use antecedent::reason::reason_code;
use antecedent::{CausalError, GraphRef};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::PyCancellationToken;
use crate::bridge::RefusalJson;
use crate::chain_decision_api::context;

type Returned<T> = (Option<T>, Option<String>);

fn refused(error: &RpcmciReplayError) -> String {
    let reason = match error {
        RpcmciReplayError::Cancelled | RpcmciReplayError::Causal(CausalError::Cancelled { .. }) => {
            reason_code!("cancelled_no_claim")
        }
        RpcmciReplayError::Resource(_)
        | RpcmciReplayError::Causal(CausalError::Resource { .. }) => {
            reason_code!("execution_resource_limit")
        }
        _ => reason_code!("invalid_argument"),
    };
    RefusalJson::new(
        reason,
        "rpcmci_learned",
        "rpcmci_learned.portable_refusal",
        &error.to_string(),
    )
    .json()
}

#[pyclass(name = "LearnedRpcmciHandle", skip_from_py_object)]
struct PyLearnedRpcmci(LearnedRpcmciArtifact);

#[pymethods]
impl PyLearnedRpcmci {
    #[getter]
    fn identity(&self) -> &str {
        self.0.identity()
    }

    fn plan_view_json(&self) -> String {
        self.0.plan_view().to_json().to_string()
    }

    fn report_json(&self) -> PyResult<String> {
        serde_json::to_string(self.0.report()).map_err(crate::value_err_display)
    }

    fn export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.0
            .to_bytes()
            .map(|value| PyBytes::new(py, &value))
            .map_err(crate::value_err_display)
    }

    fn regime_scope_json(&self, regime: u32) -> PyResult<Returned<String>> {
        match self.0.regime_scope(regime) {
            Ok(scope) => serde_json::to_string(&scope)
                .map(|value| (Some(value), None))
                .map_err(crate::value_err_display),
            Err(error) => Ok((None, Some(refused(&error)))),
        }
    }

    fn accept_regime(&self, regime: u32) -> Returned<crate::graphs::TemporalCpdag> {
        match self.0.accept_regime(regime) {
            Ok(accepted) => {
                let GraphRef::TemporalCpdag(graph) = accepted.graph_ref() else {
                    return (
                        None,
                        Some(refused(&RpcmciReplayError::Invalid(
                            "review yielded no temporal CPDAG",
                        ))),
                    );
                };
                (
                    Some(crate::graphs::TemporalCpdag {
                        cpdag: graph.clone(),
                        names: self.0.names().to_vec(),
                    }),
                    None,
                )
            }
            Err(error) => (None, Some(refused(&error))),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (request, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
fn learned_rpcmci_run(
    py: Python<'_>,
    request: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Returned<PyLearnedRpcmci>> {
    if request.len() > 32 * 1024 * 1024 {
        return Ok((
            None,
            Some(refused(&RpcmciReplayError::Invalid(
                "source declaration exceeds its byte bound",
            ))),
        ));
    }
    let ctx = context(seed, threads, cancel, memory_bytes);
    crate::detach_catch(py, || {
        let _decode = match ctx.acquire_admission(0, request.len() as u64) {
            Ok(lease) => lease,
            Err(error) => {
                return Ok((
                    None,
                    Some(refused(&if ctx.cancellation.is_cancelled() {
                        RpcmciReplayError::Cancelled
                    } else {
                        RpcmciReplayError::Resource(error.to_string())
                    })),
                ));
            }
        };
        let decoded: LearnedRpcmciRequest =
            serde_json::from_str(request).map_err(crate::value_err_display)?;
        Ok(match decoded.run(&ctx) {
            Ok(result) => (Some(PyLearnedRpcmci(result)), None),
            Err(error) => (None, Some(refused(&error))),
        })
    })
}

#[pyfunction]
#[pyo3(signature = (data, expected_identity, *, threads=None, cancel=None, memory_bytes=None))]
fn learned_rpcmci_consume(
    py: Python<'_>,
    data: &[u8],
    expected_identity: &str,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Returned<PyLearnedRpcmci>> {
    let ctx = context(0, threads, cancel, memory_bytes);
    crate::detach_catch(py, || {
        Ok(
            match consume_learned_rpcmci(data, expected_identity, &ctx) {
                Ok(result) => (Some(PyLearnedRpcmci(result)), None),
                Err(error) => (None, Some(refused(&error))),
            },
        )
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLearnedRpcmci>()?;
    module.add_function(wrap_pyfunction!(learned_rpcmci_run, module)?)?;
    module.add_function(wrap_pyfunction!(learned_rpcmci_consume, module)?)
}
