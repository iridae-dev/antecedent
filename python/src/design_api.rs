//! Full [`DesignRanker`] Python surface.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal::design::{
    CandidateDesign, DesignConstraints, DesignCost, DesignEvaluationContext, DesignObjective,
    DesignRankConfig, DesignRanker, EffectWidthContext, EnvironmentGramSpec, EnvironmentPlan,
    ExperimentPlan, InterventionDesignEffect, MeasureColumnSpec, MeasurementPlan,
    ModelLoglikDraws, SamplingPlan, rank_designs as facade_rank_designs,
};
use causal_prob::{GraphIdentFlag, WeightedGraphSamples};
use causal_core::{EnvironmentId, ModelId, QueryId, VariableId};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};

use crate::gcm_api::{DesignConstraintViolation, DesignRanking, RankedDesign};
use crate::{catch_ffi, py_err, py_execution_context, py_msg};

type QueryVarUnlock = (QueryId, Arc<[VariableId]>);
type QueryEnvUnlock = (QueryId, Arc<[EnvironmentId]>);

fn cost_from_dict(d: &Bound<'_, PyDict>) -> PyResult<DesignCost> {
    let amount: f64 = d.get_item("cost")?.map(|v| v.extract()).transpose()?.unwrap_or(0.0);
    let sample_budget: u64 =
        d.get_item("sample_budget")?.map(|v| v.extract()).transpose()?.unwrap_or(0);
    Ok(DesignCost { amount, sample_budget })
}

fn parse_candidate(item: &Bound<'_, PyAny>, default_tag: u64) -> PyResult<CandidateDesign> {
    let d = item
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("each candidate must be a dict with a 'kind' field"))?;
    let kind: String = d
        .get_item("kind")?
        .ok_or_else(|| PyValueError::new_err("candidate missing 'kind'"))?
        .extract()?;
    let tag: u64 = d.get_item("tag")?.map(|v| v.extract()).transpose()?.unwrap_or(default_tag);
    let cost = cost_from_dict(d)?;
    Ok(match kind.as_str() {
        "measure" => {
            let vars: Vec<u32> = d
                .get_item("variables")?
                .ok_or_else(|| PyValueError::new_err("measure requires variables"))?
                .extract()?;
            if vars.is_empty() {
                return Err(PyValueError::new_err("measure variables must be non-empty"));
            }
            CandidateDesign::Measure(MeasurementPlan {
                variables: Arc::from(
                    vars.into_iter().map(VariableId::from_raw).collect::<Vec<_>>(),
                ),
                cost,
                tag,
            })
        }
        "intervene" => {
            let targets: Vec<u32> = d
                .get_item("targets")?
                .ok_or_else(|| PyValueError::new_err("intervene requires targets"))?
                .extract()?;
            if targets.is_empty() {
                return Err(PyValueError::new_err("intervene targets must be non-empty"));
            }
            CandidateDesign::Intervene(ExperimentPlan {
                targets: Arc::from(
                    targets.into_iter().map(VariableId::from_raw).collect::<Vec<_>>(),
                ),
                cost,
                tag,
            })
        }
        "observe_environment" => {
            let environment: u32 = d
                .get_item("environment")?
                .ok_or_else(|| PyValueError::new_err("observe_environment requires environment"))?
                .extract()?;
            let additional_rows: u64 =
                d.get_item("additional_rows")?.map(|v| v.extract()).transpose()?.unwrap_or(0);
            CandidateDesign::ObserveEnvironment(EnvironmentPlan {
                environment: EnvironmentId::from_raw(environment),
                additional_rows,
                cost,
                tag,
            })
        }
        "increase_sampling_rate" | "sampling" => {
            let additional_samples: u64 = d
                .get_item("additional_samples")?
                .ok_or_else(|| {
                    PyValueError::new_err("increase_sampling_rate requires additional_samples")
                })?
                .extract()?;
            CandidateDesign::IncreaseSamplingRate(SamplingPlan { additional_samples, cost, tag })
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown candidate kind `{other}` (expected measure|intervene|observe_environment|increase_sampling_rate)"
            )));
        }
    })
}

fn parse_objective(
    objective: &Bound<'_, PyAny>,
    query_id: Option<u32>,
    model_ids: Option<Vec<u32>>,
    decision_id: Option<u32>,
) -> PyResult<DesignObjective> {
    if let Ok(name) = objective.extract::<String>() {
        return match name.as_str() {
            "reduce_graph_entropy" | "eig" => Ok(DesignObjective::ReduceGraphEntropy),
            "increase_identification_probability" | "id_probability" => {
                let q = query_id.ok_or_else(|| {
                    PyValueError::new_err("increase_identification_probability requires query_id")
                })?;
                Ok(DesignObjective::IncreaseIdentificationProbability {
                    query: QueryId::from_raw(q),
                })
            }
            "reduce_effect_posterior_width" | "effect_width" => {
                let q = query_id.ok_or_else(|| {
                    PyValueError::new_err("reduce_effect_posterior_width requires query_id")
                })?;
                Ok(DesignObjective::ReduceEffectPosteriorWidth { query: QueryId::from_raw(q) })
            }
            "reduce_decision_regret" | "decision_regret" => {
                let d = decision_id.ok_or_else(|| {
                    PyValueError::new_err("reduce_decision_regret requires decision_id")
                })?;
                Ok(DesignObjective::ReduceDecisionRegret {
                    decision: causal::design::DecisionProblemId::from_raw(d),
                })
            }
            "distinguish_models" => {
                let ids = model_ids.ok_or_else(|| {
                    PyValueError::new_err("distinguish_models requires model_ids")
                })?;
                if ids.len() < 2 {
                    return Err(PyValueError::new_err("distinguish_models needs ≥2 model_ids"));
                }
                Ok(DesignObjective::DistinguishModels {
                    models: Arc::from(ids.into_iter().map(ModelId::from_raw).collect::<Vec<_>>()),
                })
            }
            other => Err(PyValueError::new_err(format!("unknown objective `{other}`"))),
        };
    }
    let d = objective
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("objective must be a string or dict"))?;
    let kind: String = d
        .get_item("kind")?
        .ok_or_else(|| PyValueError::new_err("objective dict missing 'kind'"))?
        .extract()?;
    let q = d.get_item("query_id")?.map(|v| v.extract()).transpose()?.or(query_id);
    let mids = d.get_item("model_ids")?.map(|v| v.extract()).transpose()?.or(model_ids);
    let did = d.get_item("decision_id")?.map(|v| v.extract()).transpose()?.or(decision_id);
    // Re-dispatch on kind string without re-binding PyAny.
    match kind.as_str() {
        "reduce_graph_entropy" | "eig" => Ok(DesignObjective::ReduceGraphEntropy),
        "increase_identification_probability" | "id_probability" => {
            let q = q.ok_or_else(|| {
                PyValueError::new_err("increase_identification_probability requires query_id")
            })?;
            Ok(DesignObjective::IncreaseIdentificationProbability { query: QueryId::from_raw(q) })
        }
        "reduce_effect_posterior_width" | "effect_width" => {
            let q = q.ok_or_else(|| {
                PyValueError::new_err("reduce_effect_posterior_width requires query_id")
            })?;
            Ok(DesignObjective::ReduceEffectPosteriorWidth { query: QueryId::from_raw(q) })
        }
        "reduce_decision_regret" | "decision_regret" => {
            let d = did.ok_or_else(|| {
                PyValueError::new_err("reduce_decision_regret requires decision_id")
            })?;
            Ok(DesignObjective::ReduceDecisionRegret {
                decision: causal::design::DecisionProblemId::from_raw(d),
            })
        }
        "distinguish_models" => {
            let ids =
                mids.ok_or_else(|| PyValueError::new_err("distinguish_models requires model_ids"))?;
            if ids.len() < 2 {
                return Err(PyValueError::new_err("distinguish_models needs ≥2 model_ids"));
            }
            Ok(DesignObjective::DistinguishModels {
                models: Arc::from(ids.into_iter().map(ModelId::from_raw).collect::<Vec<_>>()),
            })
        }
        other => Err(PyValueError::new_err(format!("unknown objective `{other}`"))),
    }
}

fn parse_unlock_vars(raw: Option<Bound<'_, PyAny>>) -> PyResult<Option<Vec<QueryVarUnlock>>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let list = raw.cast::<PyList>().map_err(|_| {
        PyValueError::new_err("query_id_unlock must be a list of (query_id, [var_ids])")
    })?;
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let (q, vars): (u32, Vec<u32>) = item.extract()?;
        out.push((
            QueryId::from_raw(q),
            Arc::from(vars.into_iter().map(VariableId::from_raw).collect::<Vec<_>>()),
        ));
    }
    Ok(Some(out))
}

fn parse_unlock_envs(raw: Option<Bound<'_, PyAny>>) -> PyResult<Option<Vec<QueryEnvUnlock>>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let list = raw.cast::<PyList>().map_err(|_| {
        PyValueError::new_err("env_id_unlock must be a list of (query_id, [env_ids])")
    })?;
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let (q, envs): (u32, Vec<u32>) = item.extract()?;
        out.push((
            QueryId::from_raw(q),
            Arc::from(envs.into_iter().map(EnvironmentId::from_raw).collect::<Vec<_>>()),
        ));
    }
    Ok(Some(out))
}

fn parse_measure_columns(cols: &Bound<'_, PyList>) -> PyResult<Arc<[MeasureColumnSpec]>> {
    let mut specs = Vec::with_capacity(cols.len());
    for item in cols.iter() {
        let c = item.cast::<PyDict>()?;
        let variable: u32 = c
            .get_item("variable")?
            .ok_or_else(|| PyValueError::new_err("measure_columns entry needs variable"))?
            .extract()?;
        let cross: Vec<f64> = c
            .get_item("cross")?
            .ok_or_else(|| PyValueError::new_err("measure_columns entry needs cross"))?
            .extract()?;
        let self_dot: f64 = c
            .get_item("self_dot")?
            .ok_or_else(|| PyValueError::new_err("measure_columns entry needs self_dot"))?
            .extract()?;
        let sigma2_after: Option<f64> =
            c.get_item("sigma2_after")?.map(|v| v.extract()).transpose()?;
        specs.push(MeasureColumnSpec {
            variable: VariableId::from_raw(variable),
            cross: Arc::from(cross),
            self_dot,
            sigma2_after,
        });
    }
    Ok(Arc::from(specs))
}

fn parse_environment_grams(grams: &Bound<'_, PyList>) -> PyResult<Arc<[EnvironmentGramSpec]>> {
    let mut specs = Vec::with_capacity(grams.len());
    for item in grams.iter() {
        let c = item.cast::<PyDict>()?;
        let environment: u32 = c
            .get_item("environment")?
            .ok_or_else(|| PyValueError::new_err("environment_grams entry needs environment"))?
            .extract()?;
        let g_xtx: Vec<f64> = c
            .get_item("xtx")?
            .ok_or_else(|| PyValueError::new_err("environment_grams entry needs xtx"))?
            .extract()?;
        let g_n: u64 = c
            .get_item("n")?
            .ok_or_else(|| PyValueError::new_err("environment_grams entry needs n"))?
            .extract()?;
        let g_sigma2: Option<f64> = c.get_item("sigma2")?.map(|v| v.extract()).transpose()?;
        specs.push(EnvironmentGramSpec {
            environment: EnvironmentId::from_raw(environment),
            xtx: Arc::from(g_xtx),
            n: g_n,
            sigma2: g_sigma2,
        });
    }
    Ok(Arc::from(specs))
}

fn parse_effect_width(raw: Option<Bound<'_, PyAny>>) -> PyResult<Option<EffectWidthContext>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let d =
        raw.cast::<PyDict>().map_err(|_| PyValueError::new_err("effect_width must be a dict"))?;
    let xtx: Vec<f64> = d
        .get_item("xtx")?
        .ok_or_else(|| PyValueError::new_err("effect_width requires xtx"))?
        .extract()?;
    let sigma2: f64 = d
        .get_item("sigma2")?
        .ok_or_else(|| PyValueError::new_err("effect_width requires sigma2"))?
        .extract()?;
    let treatment_col: usize = d
        .get_item("treatment_col")?
        .ok_or_else(|| PyValueError::new_err("effect_width requires treatment_col"))?
        .extract()?;
    let n: u64 = d
        .get_item("n")?
        .ok_or_else(|| PyValueError::new_err("effect_width requires n"))?
        .extract()?;
    let measure_columns = if let Some(cols) = d.get_item("measure_columns")? {
        Some(parse_measure_columns(cols.cast::<PyList>()?)?)
    } else {
        None
    };
    let mut intervention_design = None;
    if let Some(iv) = d.get_item("intervention_design")? {
        let c = iv.cast::<PyDict>()?;
        let iv_xtx: Vec<f64> = c
            .get_item("xtx")?
            .ok_or_else(|| PyValueError::new_err("intervention_design needs xtx"))?
            .extract()?;
        let iv_sigma2: f64 = c
            .get_item("sigma2")?
            .ok_or_else(|| PyValueError::new_err("intervention_design needs sigma2"))?
            .extract()?;
        let iv_n: u64 = c
            .get_item("n")?
            .ok_or_else(|| PyValueError::new_err("intervention_design needs n"))?
            .extract()?;
        intervention_design =
            Some(InterventionDesignEffect { xtx: Arc::from(iv_xtx), sigma2: iv_sigma2, n: iv_n });
    }
    let environment_grams = if let Some(grams) = d.get_item("environment_grams")? {
        Some(parse_environment_grams(grams.cast::<PyList>()?)?)
    } else {
        None
    };
    Ok(Some(EffectWidthContext {
        xtx: Arc::from(xtx),
        sigma2,
        treatment_col,
        n,
        measure_columns,
        intervention_design,
        environment_grams,
    }))
}

fn parse_model_loglik(raw: Option<Bound<'_, PyAny>>) -> PyResult<Option<ModelLoglikDraws>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let d =
        raw.cast::<PyDict>().map_err(|_| PyValueError::new_err("model_loglik must be a dict"))?;
    let models: Vec<u32> = d
        .get_item("models")?
        .ok_or_else(|| PyValueError::new_err("model_loglik requires models"))?
        .extract()?;
    let loglik: Vec<f64> = d
        .get_item("loglik")?
        .ok_or_else(|| PyValueError::new_err("model_loglik requires loglik"))?
        .extract()?;
    let n_draws: usize = d
        .get_item("n_draws")?
        .ok_or_else(|| PyValueError::new_err("model_loglik requires n_draws"))?
        .extract()?;
    if models.is_empty() || n_draws == 0 || loglik.len() != models.len() * n_draws {
        return Err(PyValueError::new_err(
            "model_loglik loglik length must equal len(models)*n_draws",
        ));
    }
    Ok(Some(ModelLoglikDraws {
        models: Arc::from(models.into_iter().map(ModelId::from_raw).collect::<Vec<_>>()),
        loglik: Arc::from(loglik),
        n_draws,
    }))
}

fn candidate_kind(c: &CandidateDesign) -> String {
    match c {
        CandidateDesign::Measure(_) => "measure".into(),
        CandidateDesign::Intervene(_) => "intervene".into(),
        CandidateDesign::ObserveEnvironment(_) => "observe_environment".into(),
        CandidateDesign::IncreaseSamplingRate(_) => "increase_sampling_rate".into(),
    }
}

/// Rank candidate designs under a full [`DesignRanker`] objective / context.
#[pyfunction]
#[pyo3(signature = (
    graph_weights,
    identified,
    graph_keys,
    candidates,
    objective=None,
    *,
    query_id=None,
    model_ids=None,
    decision_id=None,
    query_id_unlock=None,
    env_id_unlock=None,
    identified_under_intervention=None,
    graph_features=None,
    effect_width=None,
    model_loglik=None,
    max_cost=None,
    max_sample_budget=None,
    min_batches=2,
    max_batches=64,
    batch_size=8,
    rank_uncertainty_threshold=0.05,
    seed=0,
    threads=1,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rank_designs(
    graph_weights: Vec<f64>,
    identified: Vec<u8>,
    graph_keys: Vec<u64>,
    candidates: Bound<'_, PyList>,
    objective: Option<Bound<'_, PyAny>>,
    query_id: Option<u32>,
    model_ids: Option<Vec<u32>>,
    decision_id: Option<u32>,
    query_id_unlock: Option<Bound<'_, PyAny>>,
    env_id_unlock: Option<Bound<'_, PyAny>>,
    identified_under_intervention: Option<Vec<u8>>,
    graph_features: Option<Vec<u32>>,
    effect_width: Option<Bound<'_, PyAny>>,
    model_loglik: Option<Bound<'_, PyAny>>,
    max_cost: Option<f64>,
    max_sample_budget: Option<u64>,
    min_batches: u32,
    max_batches: u32,
    batch_size: u32,
    rank_uncertainty_threshold: f64,
    seed: u64,
    threads: u32,
) -> PyResult<DesignRanking> {
    catch_ffi(|| {
        let flags: Vec<GraphIdentFlag> = identified
            .into_iter()
            .map(|v| if v == 0 { GraphIdentFlag::Unidentified } else { GraphIdentFlag::Identified })
            .collect();
        let graphs = WeightedGraphSamples::new(graph_weights, flags, graph_keys).map_err(py_msg)?;
        let mut parsed = Vec::with_capacity(candidates.len());
        for (i, item) in candidates.iter().enumerate() {
            parsed.push(parse_candidate(&item, u64::try_from(i).unwrap_or(0))?);
        }
        if parsed.is_empty() {
            return Err(PyValueError::new_err("no candidates"));
        }
        let objective = match objective {
            Some(obj) => parse_objective(&obj, query_id, model_ids, decision_id)?,
            None => DesignObjective::ReduceGraphEntropy,
        };
        let unlock_vars = parse_unlock_vars(query_id_unlock)?;
        let unlock_envs = parse_unlock_envs(env_id_unlock)?;
        let effect_width = parse_effect_width(effect_width)?;
        let model_loglik = parse_model_loglik(model_loglik)?;
        let intervene_flags: Option<Vec<GraphIdentFlag>> =
            identified_under_intervention.map(|v| {
                v.into_iter()
                    .map(|x| {
                        if x == 0 {
                            GraphIdentFlag::Unidentified
                        } else {
                            GraphIdentFlag::Identified
                        }
                    })
                    .collect()
            });

        let ranker = DesignRanker::new()
            .with_config(DesignRankConfig {
                min_batches,
                max_batches,
                batch_size,
                rank_uncertainty_threshold,
            })
            .with_constraints(DesignConstraints { max_cost, max_sample_budget });
        let ctx = py_execution_context(seed, threads);
        let unlock_var_slice = unlock_vars.as_deref();
        let unlock_env_slice = unlock_envs.as_deref();
        let intervene_slice = intervene_flags.as_deref();
        let features_slice = graph_features.as_deref();
        let eval = DesignEvaluationContext::<(), ()> {
            graphs: &graphs,
            effect_width: effect_width.as_ref(),
            model_loglik: model_loglik.as_ref(),
            decisions: None,
            query_id_unlock: unlock_var_slice,
            env_id_unlock: unlock_env_slice,
            identified_under_intervention: intervene_slice,
            graph_features: features_slice,
        };
        let ranking =
            facade_rank_designs(&ranker, &objective, &parsed, &eval, &ctx).map_err(py_err)?;
        let scores: Vec<f64> = ranking.ranked.iter().map(|r| r.score).collect();
        let best = ranking.ranked.first().map_or(0, |r| r.candidate_index);
        let ranked: Vec<RankedDesign> = ranking
            .ranked
            .iter()
            .map(|r| RankedDesign {
                candidate_index: r.candidate_index,
                kind: candidate_kind(&r.candidate),
                tag: r.candidate.tag(),
                score: r.score,
                stderr: r.monte_carlo.stderr,
                rank: r.rank,
                rank_uncertain: r.rank_uncertain,
            })
            .collect();
        let violations: Vec<DesignConstraintViolation> = ranking
            .violations
            .iter()
            .map(|v| DesignConstraintViolation {
                candidate_index: v.candidate_index,
                constraint: v.constraint.to_string(),
                detail: v.detail.to_string(),
            })
            .collect();
        Ok(DesignRanking {
            best_index: best,
            scores,
            mc_samples: ranking.budget.samples,
            early_stopped: ranking.early_stopped,
            ranked,
            violations,
        })
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rank_designs, m)?)?;
    Ok(())
}
