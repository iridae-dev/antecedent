//! Python bridge for compiler-owned event forecasts and world decisions.

use antecedent::Study;
use antecedent::analysis::events::{
    AcceptedExcitation, BaselineFeature, BridgeContract, Continuation, DeclaredQuantity,
    EventBelief, EventError, EventForecastRequest, EventHistory, EventLaw, EventObservation,
    EventStudyRequest, EventWindow, EventWorldRequest, ExcitationChallenge, ExcitationSubject,
    ExcludedShock, ForecastAssessmentRequest, HawkesParameters, MarkPrototype,
    MarkedExcitationChallenge, MarkedFitSpec, MarkedHistory, MarkedParameters,
    PreparedEventForecast, PreparedEventLikelihood, PreparedEventStudy, PreparedEventUpdate,
    PreparedEventWorlds, PreparedExcitationChallenge, PreparedForecastAssessment,
    PreparedMarkedChallenge, PreparedMarkedFit, PreparedMarkedLikelihood, PreparedShock,
    ReleaseAction, ReleaseDraw, compare_marked_release_times, compare_release_times,
    parameter_bridge,
};
use antecedent::design::decision_artifact::{contract_from_json, result_to_json};
use antecedent::design::evsi::{CostToUtilityMap, StudyCostSpec};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

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

struct QuestionPlan {
    challenge: ChallengePlan,
    bridge: BridgeContract,
    discovery: antecedent::AcceptedGraph,
    action: ReleaseAction,
    draw: ReleaseDraw,
}

enum ChallengePlan {
    Catalog(PreparedExcitationChallenge),
    Marked(PreparedMarkedChallenge),
}

enum Prepared {
    Forecast(PreparedEventForecast),
    Likelihood(PreparedEventLikelihood),
    Worlds(PreparedEventWorlds),
    Assessment(PreparedForecastAssessment),
    Update(PreparedEventUpdate),
    Study(PreparedEventStudy),
    Challenge(PreparedExcitationChallenge),
    MarkedLikelihood(PreparedMarkedLikelihood),
    MarkedFit(PreparedMarkedFit),
    MarkedChallenge(PreparedMarkedChallenge),
    Offspring(HawkesParameters, usize, f64),
    Release(HawkesParameters, PreparedExcitationChallenge, f64, f64, f64),
    MarkedOffspring(MarkedParameters, Vec<MarkPrototype>, usize, f64),
    MarkedRelease(MarkedParameters, Vec<MarkPrototype>, PreparedMarkedChallenge, f64, f64, f64),
    Question(Box<QuestionPlan>),
    Shock(PreparedShock),
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

    fn inspect(&self) -> String {
        let view = match &self.inner {
            Prepared::Forecast(p) => p.plan_view(),
            Prepared::Likelihood(p) => p.plan_view(),
            Prepared::Worlds(p) => p.plan_view(),
            Prepared::Assessment(p) => p.plan_view(),
            Prepared::Update(p) => p.plan_view(),
            Prepared::Study(p) => p.plan_view(),
            Prepared::Challenge(p) => p.plan_view(),
            Prepared::MarkedLikelihood(p) => p.plan_view(),
            Prepared::MarkedFit(p) => p.plan_view(),
            Prepared::MarkedChallenge(p) => p.plan_view(),
            Prepared::Offspring(_, source, horizon) => {
                return json!({"operation": "offspring", "source": source, "horizon": horizon})
                    .to_string();
            }
            Prepared::Release(_, challenge, earlier, later, end) => {
                return json!({
                    "operation": "release",
                    "plan": challenge.plan_view().to_json(),
                    "earlier": earlier,
                    "later": later,
                    "window_end": end,
                })
                .to_string();
            }
            Prepared::MarkedOffspring(_, _, prototype, horizon) => {
                return json!({"operation": "marked_offspring", "prototype": prototype, "horizon": horizon})
                    .to_string();
            }
            Prepared::MarkedRelease(_, _, challenge, earlier, later, end) => {
                return json!({
                    "operation": "marked_release",
                    "plan": challenge.plan_view().to_json(),
                    "earlier": earlier,
                    "later": later,
                    "window_end": end,
                })
                .to_string();
            }
            Prepared::Question(plan) => {
                let evidence = plan.discovery.provenance().map(|item| hex_digest(&item.digest()));
                return json!({"operation": "excitation_question", "evidence": evidence})
                    .to_string();
            }
            Prepared::Shock(plan) => return plan.plan_view().to_json().to_string(),
        };
        view.to_json().to_string()
    }

    #[pyo3(signature = (memory_bytes=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        memory_bytes: Option<u64>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<NativeEventResult> {
        let ctx = execution_context(0, memory_bytes, cancel);
        let (law, report, artifact, accepted) = detach_catch(py, || -> PyResult<_> {
            match &self.inner {
                Prepared::Forecast(p) => {
                    Ok((Some(p.execute(&ctx).map_err(error)?), Value::Null, None, None))
                }
                Prepared::Likelihood(p) => {
                    let likelihood = p.execute(&ctx).map_err(error)?;
                    Ok((
                        None,
                        json!({"log_likelihood": likelihood, "zero_density": likelihood == f64::NEG_INFINITY}),
                        None,
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
                        None,
                    ))
                }
                Prepared::Assessment(p) => {
                    Ok((None, json!(p.execute(&ctx).map_err(error)?), None, None))
                }
                Prepared::Study(p) => {
                    Ok((None, json!(p.execute(&ctx).map_err(error)?), None, None))
                }
                Prepared::Challenge(p) => {
                    Ok((None, json!(p.execute(&ctx).map_err(error)?), None, None))
                }
                Prepared::MarkedLikelihood(p) => {
                    let likelihood = p.execute(&ctx).map_err(error)?;
                    Ok((None, json!({"log_likelihood": likelihood}), None, None))
                }
                Prepared::MarkedFit(p) => {
                    Ok((None, json!(p.execute(&ctx).map_err(error)?), None, None))
                }
                Prepared::MarkedChallenge(p) => {
                    Ok((None, json!(p.execute(&ctx).map_err(error)?), None, None))
                }
                Prepared::Offspring(parameters, source, horizon) => Ok((
                    None,
                    json!(parameters.excitation_offspring(*source, *horizon).map_err(error)?),
                    None,
                    None,
                )),
                Prepared::Release(parameters, challenge, earlier, later, end) => {
                    let reading = challenge.execute(&ctx).map_err(error)?;
                    let comparison = match reading.status {
                        antecedent::analysis::events::DiagnosticStatus::Agreed => Some(
                            compare_release_times(parameters, &reading, *earlier, *later, *end)
                                .map_err(error)?,
                        ),
                        antecedent::analysis::events::DiagnosticStatus::Withheld => None,
                    };
                    Ok((None, json!({"reading": reading, "comparison": comparison}), None, None))
                }
                Prepared::MarkedOffspring(parameters, prototypes, prototype, horizon) => Ok((
                    None,
                    json!(
                        parameters
                            .excitation_offspring(prototypes, *prototype, *horizon)
                            .map_err(error)?
                    ),
                    None,
                    None,
                )),
                Prepared::MarkedRelease(parameters, prototypes, challenge, earlier, later, end) => {
                    let reading = challenge.execute(&ctx).map_err(error)?;
                    let comparison = match reading.status {
                        antecedent::analysis::events::DiagnosticStatus::Agreed => Some(
                            compare_marked_release_times(
                                parameters, prototypes, &reading, *earlier, *later, *end,
                            )
                            .map_err(error)?,
                        ),
                        antecedent::analysis::events::DiagnosticStatus::Withheld => None,
                    };
                    Ok((None, json!({"reading": reading, "comparison": comparison}), None, None))
                }
                Prepared::Update(p) => {
                    let (law, belief) = p.execute_with_belief(&ctx).map_err(error)?;
                    Ok((Some(law), json!({"updated_belief": belief}), None, None))
                }
                Prepared::Shock(plan) => {
                    Ok((None, json!(plan.execute(&ctx).map_err(error)?), None, None))
                }
                Prepared::Question(plan) => execute_question(plan, &ctx),
            }
        })?;
        Ok(NativeEventResult { law, report, artifact, accepted })
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
    accepted: Option<Arc<AcceptedExcitation>>,
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
    Offspring {
        parameters: HawkesParameters,
        source: usize,
        horizon: f64,
    },
    Challenge {
        request: ExcitationChallenge,
    },
    Release {
        parameters: HawkesParameters,
        challenge: ExcitationChallenge,
        earlier: f64,
        later: f64,
        window_end: f64,
    },
    MarkedLikelihood {
        history: MarkedHistory,
        parameters: MarkedParameters,
    },
    MarkedFit {
        history: MarkedHistory,
        fit: MarkedFitSpec,
    },
    MarkedOffspring {
        parameters: MarkedParameters,
        prototypes: Vec<MarkPrototype>,
        prototype: usize,
        horizon: f64,
    },
    MarkedChallenge {
        request: MarkedExcitationChallenge,
    },
    MarkedRelease {
        parameters: MarkedParameters,
        prototypes: Vec<MarkPrototype>,
        challenge: MarkedExcitationChallenge,
        earlier: f64,
        later: f64,
        window_end: f64,
    },
    ExcitationQuestion {
        challenge: ExcitationChallenge,
        bridge: BridgeWire,
        action: ActionWire,
        draw: DrawWire,
    },
    MarkedExcitationQuestion {
        challenge: MarkedExcitationChallenge,
        bridge: BridgeWire,
        action: ActionWire,
        draw: DrawWire,
    },
    Shock {
        history: EventHistory,
        feature: BaselineFeature,
        source: usize,
        target: usize,
        horizon: f64,
        instrument: String,
    },
    ReleaseStudy {
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
#[pyo3(signature = (request, source=None, receipt=None, graph=None, provenance=None, memory_bytes=None, cancel=None))]
fn events_prepare(
    py: Python<'_>,
    request: &str,
    source: Option<PyRef<'_, NativeEventResult>>,
    receipt: Option<PyRef<'_, PyProviderVerificationReceipt>>,
    graph: Option<&Bound<'_, PyAny>>,
    provenance: Option<HashMap<String, String>>,
    memory_bytes: Option<u64>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<NativePreparedEventProgram> {
    let law = source.as_ref().and_then(|item| item.law.clone());
    let accepted = source.as_ref().and_then(|item| item.accepted.clone());
    if source.is_some() && law.is_none() && accepted.is_none() {
        return Err(crate::errors::value_err("source result does not retain an event law"));
    }
    let receipt = receipt.as_ref().map(|r| r.inner.clone());
    let discovery = match (graph, provenance) {
        (Some(graph), Some(provenance)) => Some(accepted_graph(graph, provenance)?),
        (None, None) => None,
        _ => {
            return Err(crate::errors::value_err(
                "an excitation question needs the discovered graph and its provenance",
            ));
        }
    };
    let ctx = execution_context(0, memory_bytes, cancel);
    let inner = detach_catch(py, || -> PyResult<_> {
        let request: Request = parse(request)?;
        if law.is_some()
            && !matches!(&request, Request::Worlds { .. } | Request::ReleaseStudy { .. })
        {
            return Err(crate::errors::value_err(
                "retained event law is only used by world composition",
            ));
        }
        if matches!(&request, Request::ReleaseStudy { .. }) && accepted.is_none() {
            return Err(crate::errors::value_err(
                "a release study needs the excitation the bridge accepted",
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
            Request::Offspring { parameters, source, horizon } => {
                Prepared::Offspring(parameters, source, horizon)
            }
            Request::Challenge { request } => Prepared::Challenge(
                Study::prepare_excitation_challenge(request, &ctx).map_err(error)?,
            ),
            Request::Release { parameters, challenge, earlier, later, window_end } => {
                Prepared::Release(
                    parameters,
                    Study::prepare_excitation_challenge(challenge, &ctx).map_err(error)?,
                    earlier,
                    later,
                    window_end,
                )
            }
            Request::MarkedLikelihood { history, parameters } => Prepared::MarkedLikelihood(
                Study::prepare_marked_likelihood(history, parameters, &ctx).map_err(error)?,
            ),
            Request::MarkedFit { history, fit } => {
                Prepared::MarkedFit(Study::prepare_marked_fit(history, fit, &ctx).map_err(error)?)
            }
            Request::MarkedOffspring { parameters, prototypes, prototype, horizon } => {
                Prepared::MarkedOffspring(parameters, prototypes, prototype, horizon)
            }
            Request::MarkedChallenge { request } => Prepared::MarkedChallenge(
                Study::prepare_marked_challenge(request, &ctx).map_err(error)?,
            ),
            Request::MarkedRelease {
                parameters,
                prototypes,
                challenge,
                earlier,
                later,
                window_end,
            } => Prepared::MarkedRelease(
                parameters,
                prototypes,
                Study::prepare_marked_challenge(challenge, &ctx).map_err(error)?,
                earlier,
                later,
                window_end,
            ),
            Request::ExcitationQuestion { challenge, bridge, action, draw } => {
                let discovery = discovery.clone().ok_or_else(|| {
                    crate::errors::value_err(
                        "an excitation question needs the discovered graph and its provenance",
                    )
                })?;
                Prepared::Question(Box::new(QuestionPlan {
                    challenge: ChallengePlan::Catalog(
                        Study::prepare_excitation_challenge(challenge, &ctx).map_err(error)?,
                    ),
                    bridge: bridge_contract(bridge)?,
                    discovery,
                    action: release_action(action),
                    draw: release_draw(draw),
                }))
            }
            Request::MarkedExcitationQuestion { challenge, bridge, action, draw } => {
                let discovery = discovery.clone().ok_or_else(|| {
                    crate::errors::value_err(
                        "an excitation question needs the discovered graph and its provenance",
                    )
                })?;
                Prepared::Question(Box::new(QuestionPlan {
                    challenge: ChallengePlan::Marked(
                        Study::prepare_marked_challenge(challenge, &ctx).map_err(error)?,
                    ),
                    bridge: bridge_contract(bridge)?,
                    discovery,
                    action: release_action(action),
                    draw: release_draw(draw),
                }))
            }
            Request::Shock { history, feature, source, target, horizon, instrument } => {
                Prepared::Shock(
                    Study::prepare_shock_query(
                        ExcludedShock { history, feature, source, target, horizon, instrument },
                        &ctx,
                    )
                    .map_err(error)?,
                )
            }
            Request::ReleaseStudy { prior, request, worlds, contract: declaration, cost } => {
                let accepted = accepted.clone().ok_or_else(|| {
                    crate::errors::value_err(
                        "a release study needs the excitation the bridge accepted",
                    )
                })?;
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
                    Study::prepare_release_study(
                        &accepted,
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QuantityWire {
    variable_id: String,
    variable_name: String,
    units: String,
    population_id: String,
    regime_id: String,
    horizon: u32,
    functional_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BridgeWire {
    source: QuantityWire,
    target: QuantityWire,
    #[serde(default)]
    intercept: Option<f64>,
    #[serde(default)]
    slope: Option<f64>,
    statement: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ActionWire {
    Now,
    Delay { release: f64 },
    ChangeMark { mark: usize, release: f64 },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DrawWire {
    release: f64,
    window_end: f64,
    paths: usize,
    max_events: usize,
    seed: u64,
}

fn bridge_contract(wire: BridgeWire) -> PyResult<BridgeContract> {
    let declared = |wire: QuantityWire| DeclaredQuantity {
        variable_id: wire.variable_id,
        variable_name: wire.variable_name,
        units: wire.units,
        population_id: wire.population_id,
        regime_id: wire.regime_id,
        horizon: wire.horizon,
        functional_id: wire.functional_id,
    };
    parameter_bridge(
        declared(wire.source),
        declared(wire.target),
        wire.intercept,
        wire.slope,
        &wire.statement,
    )
    .map_err(error)
}

fn release_action(wire: ActionWire) -> ReleaseAction {
    match wire {
        ActionWire::Now => ReleaseAction::Now,
        ActionWire::Delay { release } => ReleaseAction::Delay { release },
        ActionWire::ChangeMark { mark, release } => {
            ReleaseAction::ChangeMark { kind: mark, release }
        }
    }
}

fn release_draw(wire: DrawWire) -> ReleaseDraw {
    ReleaseDraw {
        release: wire.release,
        window_end: wire.window_end,
        paths: wire.paths,
        max_events: wire.max_events,
        seed: wire.seed,
    }
}

type Executed = (Option<EventLaw>, Value, Option<Vec<u8>>, Option<Arc<AcceptedExcitation>>);

fn execute_question(
    plan: &QuestionPlan,
    ctx: &antecedent::prelude::ExecutionContext,
) -> PyResult<Executed> {
    let (reading, subject) = match &plan.challenge {
        ChallengePlan::Catalog(challenge) => {
            (challenge.execute(ctx).map_err(error)?, challenge.kernel(ctx).map_err(error)?)
        }
        ChallengePlan::Marked(challenge) => {
            (challenge.execute(ctx).map_err(error)?, challenge.kernel(ctx).map_err(error)?)
        }
    };
    if reading.status != antecedent::analysis::events::DiagnosticStatus::Agreed {
        return Ok((None, json!({"reading": reading, "accepted": false}), None, None));
    }
    let prepared = Study::prepare_excitation_question(
        reading.clone(),
        subject.clone(),
        plan.bridge.clone(),
        &plan.discovery,
        ctx,
    )
    .map_err(error)?;
    let accepted = prepared.execute(ctx).map_err(error)?;
    let continuation = match &subject {
        ExcitationSubject::Catalog { history, .. } => Continuation::Catalog(history),
        ExcitationSubject::Marked { history, .. } => Continuation::Marked(history),
    };
    let law = accepted.release_law(continuation, plan.action, plan.draw, ctx).map_err(error)?;
    let report = json!({
        "reading": reading,
        "accepted": true,
        "bridged_direct": accepted.bridged_direct,
        "conditional_direct": accepted.conditional_direct,
        "horizon_descendants": accepted.horizon_descendants,
        "lifetime_descendants": accepted.lifetime_descendants,
        "transported": accepted.transported,
    });
    Ok((Some(law), report, None, Some(Arc::new(accepted))))
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    bytes.iter().fold(String::with_capacity(64), |mut out, byte| {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
        out
    })
}

fn accepted_graph(
    graph: &Bound<'_, PyAny>,
    provenance: HashMap<String, String>,
) -> PyResult<antecedent::AcceptedGraph> {
    let provenance = Arc::new(structure_provenance(&provenance)?);
    if let Ok(item) = graph.extract::<crate::graphs::Dag>() {
        return bind(antecedent::AcceptedGraph::dag(item.dag), &item.names, provenance);
    }
    if let Ok(item) = graph.extract::<crate::graphs::Cpdag>() {
        let graph = antecedent::AcceptedGraph::cpdag(item.cpdag)
            .map_err(|err| crate::errors::value_err(err.to_string()))?;
        return bind(graph, &item.names, provenance);
    }
    if let Ok(item) = graph.extract::<crate::graphs::Pag>() {
        return bind(antecedent::AcceptedGraph::pag(item.pag), &item.names, provenance);
    }
    if let Ok(item) = graph.extract::<crate::graphs::Admg>() {
        return bind(antecedent::AcceptedGraph::admg(item.admg), &item.names, provenance);
    }
    if let Ok(item) = graph.extract::<crate::graphs::TemporalDag>() {
        return bind(antecedent::AcceptedGraph::temporal_dag(item.dag), &item.names, provenance);
    }
    if let Ok(item) = graph.extract::<crate::graphs::TemporalCpdag>() {
        let graph = antecedent::AcceptedGraph::temporal_cpdag(item.cpdag)
            .map_err(|err| crate::errors::value_err(err.to_string()))?;
        return bind(graph, &item.names, provenance);
    }
    if let Ok(item) = graph.extract::<crate::graphs::TemporalPag>() {
        return bind(antecedent::AcceptedGraph::temporal_pag(item.pag), &item.names, provenance);
    }
    if let Ok(item) = graph.extract::<crate::graphs::TemporalAdmg>() {
        return bind(antecedent::AcceptedGraph::temporal_admg(item.graph), &item.names, provenance);
    }
    Err(crate::errors::value_err("the discovery graph is not a compiled graph object"))
}

fn bind(
    graph: antecedent::AcceptedGraph,
    names: &[String],
    provenance: Arc<antecedent::StructureProvenance>,
) -> PyResult<antecedent::AcceptedGraph> {
    use antecedent::prelude::CausalSchemaBuilder;
    let mut builder = CausalSchemaBuilder::new();
    for name in names {
        builder = builder.continuous(name.as_str()).context();
    }
    let schema = builder.build().map_err(|err| crate::errors::value_err(err.to_string()))?;
    Ok(graph.with_schema(&schema).with_provenance(provenance))
}

fn structure_provenance(
    fields: &HashMap<String, String>,
) -> PyResult<antecedent::StructureProvenance> {
    let text = |key: &str| {
        fields.get(key).cloned().ok_or_else(|| {
            crate::errors::value_err(format!("discovery provenance is missing {key}"))
        })
    };
    let digest = |key: &str| -> PyResult<[u8; 32]> {
        let text = text(key)?;
        if text.len() != 64 {
            return Err(crate::errors::value_err("provenance digest fields are 32 bytes of hex"));
        }
        let mut out = [0_u8; 32];
        for (index, chunk) in text.as_bytes().chunks(2).enumerate() {
            let pair = std::str::from_utf8(chunk)
                .map_err(|_| crate::errors::value_err("provenance digest is not hex"))?;
            out[index] = u8::from_str_radix(pair, 16)
                .map_err(|_| crate::errors::value_err("provenance digest is not hex"))?;
        }
        Ok(out)
    };
    let pooling = match (
        fields.get("pooling_method"),
        fields.get("pooled_imputations"),
        fields.get("pooling_source"),
    ) {
        (None, None, None) => None,
        (Some(method), Some(imputations), Some(source)) => Some(antecedent::StructurePooling {
            method: method.clone().into(),
            imputations: imputations
                .parse()
                .map_err(|_| crate::errors::value_err("pooled_imputations is not a count"))?,
            source: source.clone().into(),
        }),
        _ => {
            return Err(crate::errors::value_err(
                "pooling provenance needs a method, an imputation count, and a source",
            ));
        }
    };
    Ok(antecedent::StructureProvenance {
        method: text("method")?.into(),
        inference: text("inference")?.into(),
        discovered_class: text("discovered_class")?.into(),
        query: digest("query")?,
        search_bounds: text("search_bounds")?.into(),
        data_snapshot: digest("data_snapshot")?,
        program: digest("program")?,
        claim: digest("claim")?,
        review: text("review")?.into(),
        resolved_marks: text("resolved_marks")?
            .parse()
            .map_err(|_| crate::errors::value_err("resolved_marks is not a count"))?,
        structure_prior: fields.get("structure_prior").cloned().map(Into::into),
        pooling,
    })
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
