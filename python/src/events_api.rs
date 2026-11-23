//! Python bridge for compiler-owned event forecasts and world decisions.

use antecedent::Study;
use antecedent::analysis::events::{
    EventBelief, EventError, EventForecastRequest, EventHistory, EventLaw, EventObservation,
    EventStudyRequest, EventWindow, EventWorldRequest, ForecastAssessmentRequest, HawkesParameters,
    PreparedEventForecast, PreparedEventLikelihood, PreparedEventStudy, PreparedEventUpdate,
    PreparedEventWorlds, PreparedForecastAssessment,
};
use antecedent::design::decision_artifact::{contract_from_json, result_to_json};
use antecedent::design::evsi::{CostToUtilityMap, StudyCostSpec};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::provider_api::PyProviderVerificationReceipt;
use crate::transport_common::execution_context;
use crate::{PyCancellationToken, detach_catch};

fn error(value: EventError) -> PyErr {
    match value {
        EventError::Compiler(error) => crate::errors::py_err(error),
        other => crate::errors::value_err(other.to_string()),
    }
}

fn parse<T: serde::de::DeserializeOwned>(text: &str) -> PyResult<T> {
    if text.len() > 64 * 1024 * 1024 {
        return Err(crate::errors::value_err("event request exceeds byte limit"));
    }
    serde_json::from_str(text).map_err(|e| crate::errors::value_err(e.to_string()))
}

fn encode(value: &impl serde::Serialize) -> PyResult<String> {
    serde_json::to_string(value).map_err(|e| crate::errors::value_err(e.to_string()))
}

#[pyclass(name = "NativePreparedEventProgram")]
struct NativePreparedEventProgram {
    inner: Prepared,
}

enum Prepared {
    Forecast(PreparedEventForecast),
    Likelihood(PreparedEventLikelihood),
    Worlds(PreparedEventWorlds),
    Assessment(PreparedForecastAssessment),
    Update(PreparedEventUpdate),
    Study(PreparedEventStudy),
}

fn plan_view_of(prepared: &Prepared) -> antecedent::planner::PlanView {
    match prepared {
        Prepared::Forecast(p) => p.plan_view(),
        Prepared::Likelihood(p) => p.plan_view(),
        Prepared::Worlds(p) => p.plan_view(),
        Prepared::Assessment(p) => p.plan_view(),
        Prepared::Update(p) => p.plan_view(),
        Prepared::Study(p) => p.plan_view(),
    }
}

#[pymethods]
impl NativePreparedEventProgram {
    fn replay_identity(&self) -> PyResult<String> {
        match &self.inner {
            Prepared::Forecast(p) => Ok(p.program_identity().to_hex()),
            Prepared::Worlds(p) => Ok(p.program_identity().to_hex()),
            _ => Err(crate::errors::value_err("replay accepts forecast and world declarations")),
        }
    }

    fn unit_support(&self) -> Option<String> {
        crate::latent_soft_intervention_api::unit_support_json(&plan_view_of(&self.inner))
    }

    fn inspect(&self) -> String {
        plan_view_of(&self.inner).to_json().to_string()
    }

    #[pyo3(signature = (memory_bytes=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        memory_bytes: Option<u64>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<NativeEventResult> {
        let ctx = execution_context(0, memory_bytes, cancel);
        let (law, report, artifact) = detach_catch(py, || -> PyResult<_> {
            match &self.inner {
                Prepared::Forecast(p) => {
                    Ok((Some(p.execute(&ctx).map_err(error)?), Value::Null, None))
                }
                Prepared::Likelihood(p) => {
                    let likelihood = p.execute(&ctx).map_err(error)?;
                    Ok((
                        None,
                        json!({"log_likelihood": likelihood, "zero_density": likelihood == f64::NEG_INFINITY}),
                        None,
                    ))
                }
                Prepared::Worlds(p) => {
                    let w = p.execute(&ctx).map_err(error)?;
                    let decision = w
                        .decision
                        .as_ref()
                        .map(|decision| {
                            let encoded = result_to_json(decision, &w.distribution)
                                .map_err(|e| crate::errors::value_err(e.to_string()))?;
                            parse::<Value>(&encoded)
                        })
                        .transpose()?;
                    let identity = w.distribution.metadata().identity.causal_contract_id.clone();
                    let artifact = w
                        .distribution
                        .to_bytes(&identity)
                        .map_err(|e| crate::errors::value_err(e.to_string()))?;
                    Ok((
                        None,
                        json!({
                            "program": identity, "grid": w.grid, "decision": decision,
                            "outcome_monte_carlo_standard_errors": w.outcome_monte_carlo_standard_errors,
                            "event_meaning": w.event_meaning, "release_supported": w.release_supported,
                            "provider_receipt": w.provider_receipt, "distribution_identity": w.distribution.metadata().identity,
                        }),
                        Some(artifact),
                    ))
                }
                Prepared::Assessment(p) => Ok((None, json!(p.execute(&ctx).map_err(error)?), None)),
                Prepared::Study(p) => Ok((None, json!(p.execute(&ctx).map_err(error)?), None)),
                Prepared::Update(p) => {
                    let (law, belief) = p.execute_with_belief(&ctx).map_err(error)?;
                    Ok((Some(law), json!({"updated_belief": belief}), None))
                }
            }
        })?;
        Ok(NativeEventResult { law, report, artifact })
    }

    #[pyo3(signature = (kind, memory_bytes=None, cancel=None))]
    fn count_interval(
        &self,
        py: Python<'_>,
        kind: usize,
        memory_bytes: Option<u64>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<String> {
        let Prepared::Forecast(forecast) = &self.inner else {
            return Err(crate::errors::value_err("count interval requires a compiled forecast"));
        };
        let ctx = execution_context(0, memory_bytes, cancel);
        let interval = detach_catch(py, || forecast.count_interval(kind, &ctx).map_err(error))?;
        encode(&interval)
    }
}

#[pyclass(name = "NativeEventResult")]
struct NativeEventResult {
    law: Option<EventLaw>,
    report: Value,
    artifact: Option<Vec<u8>>,
}

#[pymethods]
impl NativeEventResult {
    fn report(&self) -> PyResult<String> {
        if let Some(law) = &self.law {
            encode(&json!({
                "program": law.identity().to_hex(), "meaning": law.meaning(), "window": law.window(),
                "path_count": law.paths().len(), "fit": law.fit(), "refit_parameters": law.refit_parameters(),
                "weights": law.weights(), "provider_receipt": law.provider_receipt(),
                "release_supported": law.release_supported(), "operation": self.report,
            }))
        } else {
            encode(&self.report)
        }
    }

    fn paths(&self) -> PyResult<String> {
        let law = self
            .law
            .as_ref()
            .ok_or_else(|| crate::errors::value_err("result is not an event law"))?;
        encode(&law.paths())
    }

    fn distribution<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.artifact
            .as_ref()
            .map(|bytes| PyBytes::new(py, bytes))
            .ok_or_else(|| crate::errors::value_err("result does not retain a world distribution"))
    }
}

#[derive(Deserialize)]
#[allow(
    clippy::large_enum_variant,
    reason = "the request is decoded once and consumed immediately, never stored"
)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Forecast {
        request: EventForecastRequest,
    },
    Likelihood {
        history: EventHistory,
        parameters: HawkesParameters,
    },
    Worlds {
        worlds: EventWorldRequest,
        contract: Option<String>,
    },
    Assessment {
        request: ForecastAssessmentRequest,
    },
    Update {
        prior: EventBelief,
        observation: EventObservation,
        forecast: EventWindow,
        paths: usize,
        max_events: usize,
        seed: u64,
    },
    Study {
        prior: EventBelief,
        request: EventStudyRequest,
        worlds: EventWorldRequest,
        contract: String,
        cost: Option<Cost>,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Cost {
    amount: f64,
    unit: String,
    utility_unit: String,
    #[serde(rename = "utility_per_cost")]
    utility_rate: f64,
}

#[pyfunction]
#[pyo3(signature = (request, source=None, receipt=None, memory_bytes=None, cancel=None))]
fn events_prepare(
    py: Python<'_>,
    request: &str,
    source: Option<PyRef<'_, NativeEventResult>>,
    receipt: Option<PyRef<'_, PyProviderVerificationReceipt>>,
    memory_bytes: Option<u64>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<NativePreparedEventProgram> {
    let law = source.as_ref().and_then(|s| s.law.clone());
    if source.is_some() && law.is_none() {
        return Err(crate::errors::value_err("source result does not retain an event law"));
    }
    let receipt = receipt.as_ref().map(|r| r.inner.clone());
    let ctx = execution_context(0, memory_bytes, cancel);
    let inner = detach_catch(py, || -> PyResult<_> {
        let request: Request = parse(request)?;
        if law.is_some() && !matches!(&request, Request::Worlds { .. }) {
            return Err(crate::errors::value_err(
                "retained event law is only used by world composition",
            ));
        }
        if receipt.is_some() && !matches!(&request, Request::Forecast { .. }) {
            return Err(crate::errors::value_err(
                "provider receipts are only used by external forecasts",
            ));
        }
        let contract = |json: &str| {
            contract_from_json(json).map_err(|e| crate::errors::value_err(e.to_string()))
        };
        Ok(match request {
            Request::Forecast { request } => Prepared::Forecast(
                Study::prepare_event_forecast(request, receipt.as_ref(), &ctx).map_err(error)?,
            ),
            Request::Likelihood { history, parameters } => Prepared::Likelihood(
                Study::prepare_event_likelihood(history, parameters, &ctx).map_err(error)?,
            ),
            Request::Worlds { worlds, contract: declaration } => {
                let law = law.as_ref().ok_or_else(|| {
                    crate::errors::value_err("worlds require a retained compiled forecast")
                })?;
                let declaration = declaration.as_deref().map(contract).transpose()?;
                Prepared::Worlds(
                    Study::prepare_event_worlds(law, worlds, declaration, &ctx).map_err(error)?,
                )
            }
            Request::Assessment { request } => {
                Prepared::Assessment(Study::prepare_event_assessment(request, &ctx).map_err(error)?)
            }
            Request::Update { prior, observation, forecast, paths, max_events, seed } => {
                Prepared::Update(
                    Study::prepare_event_update(
                        prior,
                        observation,
                        forecast,
                        paths,
                        max_events,
                        seed,
                        &ctx,
                    )
                    .map_err(error)?,
                )
            }
            Request::Study { prior, request, worlds, contract: declaration, cost } => {
                let cost = cost.map(|c| {
                    (
                        StudyCostSpec { amount: c.amount, unit: c.unit.clone() },
                        CostToUtilityMap {
                            cost_unit: c.unit,
                            utility_unit: c.utility_unit,
                            utility_per_cost: c.utility_rate,
                        },
                    )
                });
                Prepared::Study(
                    Study::prepare_event_study(
                        prior,
                        request,
                        worlds,
                        contract(&declaration)?,
                        cost,
                        &ctx,
                    )
                    .map_err(error)?,
                )
            }
        })
    })?;
    Ok(NativePreparedEventProgram { inner })
}

#[pyfunction]
fn events_provider_request_id(request: &str) -> PyResult<String> {
    let request: EventForecastRequest = parse(request)?;
    request.provider_request_id().map_err(error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePreparedEventProgram>()?;
    module.add_class::<NativeEventResult>()?;
    module.add_function(wrap_pyfunction!(events_prepare, module)?)?;
    module.add_function(wrap_pyfunction!(events_provider_request_id, module)?)?;
    Ok(())
}
