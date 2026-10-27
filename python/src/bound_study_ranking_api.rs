//! Contract-bound study ranking from a finished repair search.
//!
//! Python declares the failed identification (`FailureStage`), the studies that might close
//! it and the posterior ensemble to rank under. The repair search that decides which study
//! is sufficient on its own, the unlock lists derived from it, the licence and identity
//! checks against the prepared study's contract and the ranker are Rust's: a study the
//! search did not verify is reported and never scored.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::design::rank_studies_bound as rank_bound;
use antecedent_core::{QueryId, SearchLimits, VariableId};
use antecedent_design::{RepairLimits, RepairObjective, repair};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};

use crate::design_api::{RankInputs, decision_target_digest, ranker_for, ranking_view};
use crate::evidence_plan_api::FailureStage;
use crate::gcm_api::DesignRanking;
use crate::prepared_api::PyPreparedAnalysis;
use crate::repair_api::{parse_candidates, repair_error};
use crate::transport_common::execution_context;
use crate::{catch_ffi, detach_catch, py_err, py_execution_context, value_err};

/// Rank declared studies through a prepared study's contract.
///
/// Returns the ranking of the verified studies (candidate indices are positions in the
/// declared list) and a JSON record: `verified` indices, `unverified` studies with the
/// checker's classification and reasons, and the `unlock` variables the ranking used.
/// `route` names the failure route whose repair report decides sufficiency; it may be
/// omitted when the failure has exactly one theorem route.
#[pyfunction]
#[pyo3(signature = (
    stage,
    prepared,
    candidates_json,
    repair_objective,
    max_operations,
    max_depth,
    graph_weights,
    identified,
    graph_keys,
    *,
    route=None,
    query_id=0,
    identified_under_intervention=None,
    graph_features=None,
    max_cost=None,
    max_sample_budget=None,
    min_batches=2,
    max_batches=64,
    batch_size=8,
    rank_uncertainty_threshold=0.05,
    seed=0,
    threads=None,
    memory_bytes=None,
    decision_target=None,
    cancel=None,
))]
#[allow(clippy::too_many_arguments)]
fn rank_studies_bound(
    py: Python<'_>,
    stage: PyRef<'_, FailureStage>,
    prepared: PyRef<'_, PyPreparedAnalysis>,
    candidates_json: &str,
    repair_objective: &str,
    max_operations: usize,
    max_depth: usize,
    graph_weights: Vec<f64>,
    identified: Vec<u8>,
    graph_keys: Vec<u64>,
    route: Option<&str>,
    query_id: u32,
    identified_under_intervention: Option<Vec<u8>>,
    graph_features: Option<Vec<u32>>,
    max_cost: Option<f64>,
    max_sample_budget: Option<u64>,
    min_batches: u32,
    max_batches: u32,
    batch_size: u32,
    rank_uncertainty_threshold: f64,
    seed: u64,
    threads: Option<u32>,
    memory_bytes: Option<u64>,
    decision_target: Option<String>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(DesignRanking, String)> {
    let objective = match repair_objective {
        "minimize_cost" => RepairObjective::MinimizeCost,
        "minimize_sample_budget" => RepairObjective::MinimizeSampleBudget,
        other => {
            return Err(value_err(format!(
                "unknown objective {other:?}; use minimize_cost or minimize_sample_budget"
            )));
        }
    };
    let names = stage.names.clone();
    let observed: Vec<VariableId> = stage.observed.clone();
    let candidates = parse_candidates(&names, candidates_json)?;
    let routes = &stage.routes.routes;
    let index = match route {
        Some(wanted) => routes
            .iter()
            .position(|candidate| candidate.family().family_id() == wanted)
            .ok_or_else(|| value_err(format!("the failure has no theorem route {wanted:?}")))?,
        None if routes.len() == 1 => 0,
        None => {
            return Err(value_err(
                "the failure has several theorem routes; name the route that decides sufficiency",
            ));
        }
    };
    let failure_route = routes[index].clone();
    let limits = RepairLimits {
        search: SearchLimits { operations: max_operations, depth: max_depth },
        memory_limit_bytes: memory_bytes
            .unwrap_or_else(|| RepairLimits::default().memory_limit_bytes),
    };
    let study = std::sync::Arc::clone(&prepared.inner);
    let (inputs, ranker, target) = catch_ffi(|| {
        let kind = PyString::new(py, "increase_identification_probability").into_any();
        let inputs = RankInputs::parse(
            graph_weights,
            identified,
            graph_keys,
            None::<&Bound<'_, PyList>>,
            Some(kind),
            Some(query_id),
            None,
            None,
            None,
            None,
            identified_under_intervention,
            graph_features,
            None,
            None,
            None,
        )?;
        let ranker = ranker_for(
            max_cost,
            max_sample_budget,
            min_batches,
            max_batches,
            batch_size,
            rank_uncertainty_threshold,
        );
        Ok((inputs, ranker, decision_target_digest(decision_target.as_deref())?))
    })?;
    let query = QueryId::from_raw(query_id);

    let (ranked, record) = detach_catch(py, move || {
        let repair_ctx = execution_context(0, memory_bytes, cancel);
        let report = repair(failure_route.family(), &candidates, objective, limits, &repair_ctx)
            .map_err(|error| repair_error(&error))?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let contract = study.contract().map_err(py_err)?;
        let eval = inputs.context::<(), ()>(None);
        let ranked = rank_bound(
            &contract,
            &ranker,
            &inputs.objective,
            &candidates,
            &report,
            &observed,
            query,
            &eval,
            &ctx,
            target,
        )
        .map_err(py_err)?;
        let name_of = |variable: &VariableId| {
            names
                .get(variable.as_usize())
                .cloned()
                .unwrap_or_else(|| format!("v{}", variable.raw()))
        };
        let record = serde_json::json!({
            "verified": ranked.verified,
            "unverified": ranked.unverified.iter().map(|study| serde_json::json!({
                "index": study.index,
                "semantic_id": study.semantic_id.as_ref(),
                "classification": study.classification.as_str(),
                "reasons": study.reasons,
            })).collect::<Vec<_>>(),
            "unlock": ranked.unlock.iter().map(name_of).collect::<Vec<_>>(),
        });
        Ok((ranking_view(&ranked.ranking), record.to_string()))
    })?;
    Ok((ranked, record))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rank_studies_bound, m)?)?;
    Ok(())
}
