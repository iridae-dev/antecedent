//! Phase 11 design + incremental state conformance (DESIGN.md §§19–20 / §32).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use causal::{
    CandidateDesign, CausalState, DesignCost, DesignEvaluationContext, DesignObjective,
    DesignRankConfig, DesignRanker, LinearOlsSuffStats, MeasurementPlan, SamplingPlan, StateEvent,
    apply_state_event, new_causal_state, rank_designs,
};
use causal_core::{
    AverageEffectQuery, CacheBudget, CausalQuery, ExecutionContext, QueryId, VariableId,
};
use causal_prob::{GraphIdentFlag, WeightedGraphSamples};
use causal_state::DataBatchRef;
use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/phase11")
        .join(name)
        .join("expected.json")
}

#[test]
fn rank_candidates_eig_conformance() {
    let expected: Value =
        serde_json::from_str(&fs::read_to_string(fixture("rank_candidates_eig")).unwrap()).unwrap();
    let graphs = WeightedGraphSamples::new(
        vec![0.5, 0.3, 0.2],
        vec![
            GraphIdentFlag::Identified,
            GraphIdentFlag::Unidentified,
            GraphIdentFlag::Unidentified,
        ],
        vec![10, 20, 30],
    )
    .unwrap();
    let candidates = vec![
        CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 1,
            cost: DesignCost::zero(),
            tag: 1,
        }),
        CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(2)]),
            cost: DesignCost::zero(),
            tag: 20,
        }),
    ];
    let ranker = DesignRanker::new().with_config(DesignRankConfig {
        min_batches: 4,
        max_batches: 16,
        batch_size: 8,
        rank_uncertainty_threshold: 0.5,
    });
    let ctx = ExecutionContext::for_tests(11);
    let eval = DesignEvaluationContext::<(), ()> {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: None,
        query_id_unlock: None,
    };
    let ranking =
        rank_designs(&ranker, &DesignObjective::ReduceGraphEntropy, &candidates, &eval, &ctx)
            .unwrap();
    assert_eq!(ranking.ranked.len(), 2);
    assert!(ranking.budget.samples >= expected["min_mc_samples"].as_u64().unwrap());
    let best_kind = match &ranking.ranked[0].candidate {
        CandidateDesign::Measure(_) => "measure",
        CandidateDesign::IncreaseSamplingRate(_) => "sampling",
        _ => "other",
    };
    // Measuring with informative tag should rank at least as high as tiny sampling.
    assert!(
        ranking.ranked[0].score + 1e-9 >= ranking.ranked[1].score,
        "scores not ordered: {} vs {}",
        ranking.ranked[0].score,
        ranking.ranked[1].score
    );
    let allowed = expected["acceptable_best_kinds"].as_array().unwrap();
    let ok = allowed.iter().any(|v| v.as_str() == Some(best_kind));
    assert!(ok, "best kind {best_kind} not in {allowed:?}");
}

#[test]
fn incremental_ols_match_conformance() {
    let expected: Value =
        serde_json::from_str(&fs::read_to_string(fixture("incremental_ols_match")).unwrap())
            .unwrap();
    let mut state = new_causal_state(CacheBudget::unlimited());
    let q = state.queries.register(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    )));
    let key: Arc<str> = Arc::from("ols");
    state.suff_stats.ols.insert(Arc::clone(&key), LinearOlsSuffStats::new(2));

    let batch1_rows = [1.0, 0.0, 1.0, 1.0];
    let batch1_y = [1.0, 3.0];
    apply_state_event(
        &mut state,
        StateEvent::AppendData(DataBatchRef { id: Arc::from("b1"), nrows: 2, bytes: 32 }),
    )
    .unwrap();
    state.suff_stats.ols.get_mut(&key).unwrap().append_batch(&batch1_rows, &batch1_y).unwrap();

    let batch2_rows = [1.0, 2.0, 1.0, 3.0];
    let batch2_y = [5.0, 7.0];
    apply_state_event(
        &mut state,
        StateEvent::AppendData(DataBatchRef { id: Arc::from("b2"), nrows: 2, bytes: 32 }),
    )
    .unwrap();
    state.suff_stats.ols.get_mut(&key).unwrap().append_batch(&batch2_rows, &batch2_y).unwrap();

    assert!(state.is_stale(q));
    let beta_inc = state.suff_stats.ols[&key].solve_beta().unwrap();

    let mut full = LinearOlsSuffStats::new(2);
    let mut all_x = batch1_rows.to_vec();
    all_x.extend_from_slice(&batch2_rows);
    let mut all_y = batch1_y.to_vec();
    all_y.extend_from_slice(&batch2_y);
    full.append_batch(&all_x, &all_y).unwrap();
    let beta_full = full.solve_beta().unwrap();

    let tol = expected["stable_float_tol"].as_f64().unwrap();
    assert!((beta_inc[0] - beta_full[0]).abs() < tol);
    assert!((beta_inc[1] - beta_full[1]).abs() < tol);
    assert!((beta_inc[0] - expected["beta0"].as_f64().unwrap()).abs() < tol);
    assert!((beta_inc[1] - expected["beta1"].as_f64().unwrap()).abs() < tol);
    assert_eq!(
        state.version.raw(),
        expected["expected_version_after_two_appends"].as_u64().unwrap()
    );
}

#[test]
fn state_cache_budget_and_stale_queries() {
    let mut state: CausalState = new_causal_state(CacheBudget::new(16));
    let q = state.queries.register(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    )));
    assert!(state.refresh_results(&[(q, 1, 64)]).is_err());
    state.cache_budget = CacheBudget::new(128);
    state.refresh_results(&[(q, 9, 32)]).unwrap();
    assert!(!state.is_stale(q));
    assert_eq!(state.stale_queries(), Vec::<QueryId>::new());
}
