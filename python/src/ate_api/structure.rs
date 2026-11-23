//! Structure-only identification and query construction.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn structure_var_id(names: &[String], name: &str) -> PyResult<VariableId> {
    names
        .iter()
        .position(|n| n == name)
        .and_then(|i| u32::try_from(i).ok())
        .map(VariableId::from_raw)
        .ok_or_else(|| crate::value_err(format!("unknown variable {name:?}")))
}

fn structure_query(
    kind: &str,
    names: &[String],
    treatment: &str,
    outcome: &str,
    modifiers: Option<&[String]>,
    policy: Option<&str>,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    window: Option<(i32, i32)>,
    treatments: Option<Vec<String>>,
) -> PyResult<CausalQuery> {
    crate::bridge::check_count(
        "horizon_steps",
        u64::from(horizon_steps),
        crate::bridge::MAX_HORIZON_STEPS,
    )?;
    let t_id = structure_var_id(names, treatment)?;
    let y_id = structure_var_id(names, outcome)?;
    match kind {
        "average" | "average_effect" => {
            Ok(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(t_id, y_id)))
        }
        "intervention_response" => {
            use antecedent::query::{ResponseFunctional, ResponseQuery};
            let targets = treatments.unwrap_or_else(|| vec![treatment.to_string()]);
            let interventions = targets
                .iter()
                .map(|name| {
                    structure_var_id(names, name).map(|id| Intervention::set(id, Value::f64(1.0)))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(CausalQuery::Response(ResponseQuery::new(
                ResponseFunctional::InterventionResponse {
                    outcome: y_id,
                    interventions: interventions.into(),
                },
            )))
        }
        "response" | "response_curve" => {
            use antecedent::query::{
                ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery,
            };
            Ok(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: y_id,
                treatment: ContinuousDomain::new(
                    t_id,
                    GridSpec::Values(std::sync::Arc::from([0.0, 1.0])),
                ),
            })))
        }
        "conditional" | "conditional_effect" => {
            let modifiers = modifiers.filter(|m| !m.is_empty()).ok_or_else(|| {
                crate::value_err("ConditionalEffect identify requires modifiers=")
            })?;
            let w_ids = modifiers
                .iter()
                .map(|name| structure_var_id(names, name))
                .collect::<PyResult<Vec<_>>>()?;
            let inner = AverageEffectQuery::binary_ate(t_id, y_id).with_effect_modifiers(w_ids);
            let cq = ConditionalEffectQuery::try_new(inner)
                .map_err(|e| crate::value_err(e.to_string()))?;
            Ok(CausalQuery::ConditionalEffect(cq))
        }
        "pulse" | "sustained" | "temporal_effect" => {
            let policy = policy.unwrap_or(if kind == "sustained" { "sustained" } else { "pulse" });
            let mut query = crate::temporal_api::temporal_query_from_policy(
                policy,
                t_id,
                y_id,
                treatment_lag,
                horizon_steps,
                active_level,
            )?;
            if let Some((from, until)) = window {
                if !matches!(query.policy, TemporalPolicy::Sustained { .. }) {
                    return Err(crate::value_err("window requires a sustained policy"));
                }
                if treatment_lag != crate::temporal_support::DEFAULT_TREATMENT_LAG {
                    return Err(crate::value_err(
                        "treatment_lag does not apply with a window: the window states the policy's own span",
                    ));
                }
                query.policy = TemporalPolicy::sustained(from, until);
            }
            Ok(CausalQuery::TemporalEffect(query))
        }
        "temporal_response" => {
            use antecedent::query::{
                ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TemporalResponseSpec,
            };
            let mut query = ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: y_id,
                treatment: ContinuousDomain::new(
                    t_id,
                    GridSpec::Values(std::sync::Arc::from([0.0, 1.0])),
                ),
            });
            let temporal_policy =
                crate::temporal_support::policy_at_lag(policy.unwrap_or("pulse"), treatment_lag)?;
            query.temporal = Some(
                TemporalResponseSpec::new(vec![horizon_steps.max(1)], temporal_policy, None)
                    .map_err(|error| crate::value_err(error.to_string()))?,
            );
            Ok(CausalQuery::Response(query))
        }
        "temporal_mediation" => {
            use antecedent::query::{MediationContrast, MediationQuery};
            let mediators = treatments
                .unwrap_or_default()
                .iter()
                .map(|name| structure_var_id(names, name))
                .collect::<PyResult<Vec<_>>>()?;
            let contrast = match policy.unwrap_or("mediated") {
                "total" => MediationContrast::Total,
                "direct" => MediationContrast::Direct,
                "mediated" | "indirect" => MediationContrast::Mediated,
                other => {
                    return Err(crate::value_err(format!(
                        "unknown temporal mediation contrast {other}"
                    )));
                }
            };
            Ok(CausalQuery::Mediation(
                MediationQuery::binary(t_id, y_id, mediators, contrast)
                    .with_horizons([horizon_steps])
                    .map_err(py_msg)?,
            ))
        }
        other => Err(crate::value_err(format!("identify_structure unknown query kind {other:?}"))),
    }
}

fn identification_tuple(
    identification: &antecedent::Identification,
    names: &[String],
    estimand_method: bool,
) -> (String, String, Vec<String>) {
    let status = format!("{:?}", identification.status());
    // The strategy that ran, or (when asked) the method of the estimand that carries the answer.
    let method = if estimand_method {
        identification.estimands().first().map(|e| e.method.to_string())
    } else {
        None
    }
    .unwrap_or_else(|| identification.strategy().as_str().to_string());
    let indexer = match identification {
        antecedent::Identification::Point { temporal_indexer, .. } => temporal_indexer.as_ref(),
        antecedent::Identification::TemporalEnvelope { envelope, .. } => envelope.indexers.first(),
        _ => None,
    };
    let adjustment = identification
        .estimands()
        .first()
        .map(|e| {
            e.adjustment_set
                .iter()
                .filter_map(|id| {
                    let variable =
                        indexer.and_then(|i| i.key_of(id.raw()).ok()).map_or(*id, |k| k.variable);
                    names.get(variable.as_usize()).cloned()
                })
                .collect()
        })
        .unwrap_or_default();
    (status, method, adjustment)
}

/// Identify without data against a typed graph class.
#[pyfunction(name = "_identify_structure")]
#[pyo3(signature = (
    graph,
    query_kind,
    treatment,
    outcome,
    *,
    identifier=None,
    modifiers=None,
    policy=None,
    treatment_lag=1,
    horizon_steps=1,
    active_level=1.0,
    window=None,
    treatments=None,
    include_details=false,
    estimand_method=false,
    horizons=None,
    max_history_lag=None,
    response_grid=None,
    response_steps=None,
))]
#[allow(clippy::too_many_arguments)]
fn identify_structure(
    py: Python<'_>,
    graph: Bound<'_, PyAny>,
    query_kind: String,
    treatment: String,
    outcome: String,
    identifier: Option<String>,
    modifiers: Option<Vec<String>>,
    policy: Option<String>,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    window: Option<(i32, i32)>,
    treatments: Option<Vec<String>>,
    include_details: bool,
    estimand_method: bool,
    horizons: Option<Vec<u32>>,
    max_history_lag: Option<u32>,
    response_grid: Option<Vec<f64>>,
    response_steps: Option<Vec<(String, String, Vec<f64>)>>,
) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    let dag = graph.extract::<graphs::Dag>().ok();
    let admg = graph.extract::<graphs::Admg>().ok();
    let cpdag = graph.extract::<graphs::Cpdag>().ok();
    let pag = graph.extract::<graphs::Pag>().ok();
    let tdag = graph.extract::<graphs::TemporalDag>().ok();
    let tcpdag = graph.extract::<graphs::TemporalCpdag>().ok();
    let tpag = graph.extract::<graphs::TemporalPag>().ok();
    let (summary, details) = detach_catch(py, move || {
        let (structure, names) = if let Some(g) = dag {
            (AcceptedGraph::from(g.dag), g.names)
        } else if let Some(g) = admg {
            // A latent-free ADMG is its directed part: a DAG, so the caller's identifier keeps
            // its meaning; with bidirected edges only general ID reads the structure.
            let aligned = g.aligned_to_names(&g.names)?;
            if aligned.has_bidirected() {
                (AcceptedGraph::from(aligned), g.names)
            } else {
                let mut dag = antecedent::graph::Dag::with_variables(
                    u32::try_from(aligned.node_count())
                        .map_err(|_| crate::value_err("ADMG too large"))?,
                );
                for node in 0..aligned.node_count() {
                    let from =
                        antecedent::graph::DenseNodeId::try_from_usize(node).map_err(py_err)?;
                    for &to in aligned.children(from) {
                        dag.insert_directed(from, to).map_err(py_err)?;
                    }
                }
                (AcceptedGraph::from(dag), g.names)
            }
        } else if let Some(g) = cpdag {
            (AcceptedGraph::from(g.cpdag), g.names)
        } else if let Some(g) = pag {
            (AcceptedGraph::from(g.pag), g.names)
        } else if let Some(g) = tdag {
            (AcceptedGraph::from(g.dag), g.names)
        } else if let Some(g) = tcpdag {
            (AcceptedGraph::from(g.cpdag), g.names)
        } else if let Some(g) = tpag {
            (AcceptedGraph::from(g.pag), g.names)
        } else {
            return Err(crate::value_err(
                "identify_structure requires a Dag, Admg, Cpdag, Pag, TemporalDag, TemporalCpdag, or TemporalPag",
            ));
        };
        let mut query = structure_query(
            &query_kind,
            &names,
            &treatment,
            &outcome,
            modifiers.as_deref(),
            policy.as_deref(),
            treatment_lag,
            horizon_steps,
            active_level,
            window,
            treatments,
        )?;
        match &mut query {
            CausalQuery::TemporalEffect(temporal) => {
                temporal.max_history_lag = max_history_lag;
            }
            CausalQuery::Response(response) if response.temporal.is_some() => {
                let Some(temporal) = response.temporal.as_mut() else {
                    return Err(py_msg("temporal response is missing"));
                };
                if let Some(horizons) = horizons {
                    *temporal = antecedent::query::TemporalResponseSpec::new(
                        horizons,
                        temporal.policy.clone(),
                        max_history_lag,
                    )
                    .map_err(py_msg)?;
                } else {
                    temporal.max_history_lag = max_history_lag;
                }
                if let Some(grid) = response_grid
                    && let antecedent::query::ResponseFunctional::MeanCurve { treatment, .. } =
                        &mut response.functional
                {
                    treatment.grid = antecedent::query::GridSpec::Values(grid.into());
                }
                if let Some(steps) = response_steps {
                    let treatment_ids = steps
                        .iter()
                        .map(|(name, _, _)| structure_var_id(&names, name))
                        .collect::<PyResult<Vec<_>>>()?;
                    let kinds = steps.iter().map(|(_, kind, _)| kind.clone()).collect();
                    let parameters = steps.into_iter().map(|(_, _, values)| values).collect();
                    let functional = crate::response_api::build_functional(
                        "intervention_response",
                        &treatment_ids,
                        &[structure_var_id(&names, &outcome)?],
                        None,
                        None,
                        None,
                        Some(kinds),
                        Some(parameters),
                        1,
                        antecedent::query::DerivativeScale::Identity,
                        antecedent::query::DerivativeWeighting::Observed,
                    )?;
                    response.functional = crate::response_api::wrap_temporal_sequence_steps(
                        functional,
                        temporal.treatment_offset().map_err(py_msg)?,
                    )?;
                }
            }
            CausalQuery::Mediation(mediation) => {
                if let Some(horizons) = horizons {
                    *mediation = mediation.clone().with_horizons(horizons).map_err(py_msg)?;
                }
            }
            _ => {}
        }
        // The compiled study over a table with the graph's variables and no rows: identification
        // reads structure and query only, so the answer is the one a study over the data derives.
        let empty: &[f64] = &[];
        let columns = names.iter().map(|name| (name.as_str(), empty));
        let temporal = matches!(
            structure.class(),
            antecedent::GraphClass::TemporalDag
                | antecedent::GraphClass::TemporalCpdag
                | antecedent::GraphClass::TemporalPag
                | antecedent::GraphClass::TemporalAdmg
        );
        let builder = if temporal {
            Study::series(
                antecedent::prelude::TimeSeriesData::from_f64_columns(columns, 1)
                    .map_err(py_err)?,
            )
        } else {
            Study::tabular(
                antecedent::prelude::TabularData::from_f64_columns(columns).map_err(py_err)?,
            )
        };
        let mut builder = builder.graph(structure.clone()).query(query.clone());
        if let Some(id) = identifier {
            let strategy = id
                .parse::<antecedent::IdentifierId>()
                .map_err(|e| crate::value_err(e.to_string()))?;
            builder = builder.identifier(strategy);
        }
        let ctx = py_execution_context(0, crate::resolve_user_threads(None));
        let identification = builder.identify_structure_within(&ctx).map_err(py_err)?;
        let details = if include_details {
            Some(crate::identification_details::to_json(
                &identification,
                &query,
                &names,
                structure.class().as_str(),
                None,
                &ctx,
            )?)
        } else {
            None
        };
        Ok((identification_tuple(&identification, &names, estimand_method), details))
    })?;
    match details {
        Some(json) => json.into_py_any(py),
        None => summary.into_py_any(py),
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(identify_structure, m)?)
}
