//! design + incremental state conformance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use causal::design::{
    CandidateDesign,
    DesignCost,
    DesignEvaluationContext,
    DesignObjective,
    DesignRankConfig,
    DesignRanker,
    MeasurementPlan,
    SamplingPlan,
    rank_designs,
};
use causal::state::{
    CausalState,
    DataVersion,
    GraphScoreCacheKey,
    GraphScoreData,
    GraphScoreFamily,
    LgssmParams,
    LinearOlsSuffStats,
    LocalScoreCache,
    ParentSetOp,
    ParticleFilterState,
    RollingMechanismDiagnostics,
    StateEvent,
    apply_state_event,
    full_graph_score,
    insert_mechanism_diag,
    new_causal_state,
};
use causal_core::{
    AverageEffectQuery, CacheBudget, CausalQuery, ExecutionContext, QueryId, VariableId,
};
use causal_prob::{GraphIdentFlag, WeightedGraphSamples};
use causal_state::DataBatchRef;
use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/design_state")
        .join(name)
        .join("expected.json")
}

fn idx_f64(i: usize) -> f64 {
    f64::from(u32::try_from(i).expect("test index fits u32"))
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
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
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
    // Informative Measure should rank at least as high as tiny sampling.
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
fn incremental_graph_score_match_conformance() {
    let expected: Value = serde_json::from_str(
        &fs::read_to_string(fixture("incremental_graph_score_match")).unwrap(),
    )
    .unwrap();
    let n = usize::try_from(expected["n_rows"].as_u64().unwrap()).expect("fixture n_rows");
    let mut cols = vec![0.0; 3 * n];
    for i in 0..n {
        let x0 = idx_f64(i) * 0.1 - 2.0;
        let x1 = 2.0 * x0 + 0.01 * (f64::from(u32::try_from(i % 3).expect("mod 3")) - 1.0);
        let x2 = x1 + 0.01 * ((idx_f64(i) * 0.3).sin());
        cols[i] = x0;
        cols[n + i] = x1;
        cols[2 * n + i] = x2;
    }
    let data = GraphScoreData::new(n, 3, Arc::from(cols)).unwrap();
    let mut cache = LocalScoreCache::new(GraphScoreCacheKey {
        data_version: 1,
        family: GraphScoreFamily::GaussianBic,
        var_fingerprint: 3,
        penalty_fingerprint: n as u64,
    });
    for node in 0..3u32 {
        cache.delta_score(&data, ParentSetOp::SetParents { node, parents: Arc::from([]) }).unwrap();
    }
    let s0 = cache.score_graph(&data).unwrap();
    let (_delta, s1) = cache
        .delta_score(&data, ParentSetOp::SetParents { node: 1, parents: Arc::from([0u32]) })
        .unwrap();
    let mut parent_map = std::collections::HashMap::new();
    parent_map.insert(0, Arc::from([]));
    parent_map.insert(1, Arc::from([0u32]));
    parent_map.insert(2, Arc::from([]));
    let full = full_graph_score(&data, GraphScoreFamily::GaussianBic, &parent_map).unwrap();
    let tol = expected["stable_float_tol"].as_f64().unwrap();
    assert!((s1 - full).abs() < tol, "inc={s1} full={full}");
    assert!(s1 > s0 + expected["min_delta"].as_f64().unwrap());
}

#[test]
fn incremental_particle_filter_match_conformance() {
    let expected: Value = serde_json::from_str(
        &fs::read_to_string(fixture("incremental_particle_filter_match")).unwrap(),
    )
    .unwrap();
    let n = usize::try_from(expected["n_obs"].as_u64().unwrap()).expect("fixture n_obs");
    let n_particles =
        usize::try_from(expected["n_particles"].as_u64().unwrap()).expect("fixture n_particles");
    let seed = expected["seed"].as_u64().unwrap();
    let params = LgssmParams {
        a: expected["a"].as_f64().unwrap(),
        process_std: expected["process_std"].as_f64().unwrap(),
        obs_std: expected["obs_std"].as_f64().unwrap(),
    };
    // Deterministic synthetic observations (same generator as unit test).
    let mut rng = causal_core::CausalRng::from_seed(expected["obs_seed"].as_u64().unwrap());
    let mut x = 0.0;
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        x = params.a * x + params.process_std * z;
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        let e = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        ys.push(x + params.obs_std * e);
    }
    let batch = ParticleFilterState::run_batch(&ys, n_particles, params, 1, seed).unwrap();
    let mut step = ParticleFilterState::init(n_particles, params, 1, seed).unwrap();
    for &y in &ys {
        step.step(y).unwrap();
    }
    let tol = expected["stable_float_tol"].as_f64().unwrap();
    assert!((step.weighted_mean() - batch.weighted_mean()).abs() < tol);
    assert!((step.ess() - batch.ess()).abs() < tol);
    assert_eq!(step.n_obs, n as u64);
}

#[test]
fn rolling_mechanism_diag_match_conformance() {
    let expected: Value =
        serde_json::from_str(&fs::read_to_string(fixture("rolling_mechanism_diag_match")).unwrap())
            .unwrap();
    let window = usize::try_from(expected["window"].as_u64().unwrap()).expect("fixture window");
    let n_rows = usize::try_from(expected["n_rows"].as_u64().unwrap()).expect("fixture n_rows");
    let tol = expected["stable_float_tol"].as_f64().unwrap();
    let mut state: CausalState = new_causal_state(CacheBudget::new(1024 * 1024));
    let key: Arc<str> = Arc::from("mech");
    let mut diag = RollingMechanismDiagnostics::new(2, window).unwrap();
    let mut all_rows = Vec::new();
    let mut all_y = Vec::new();
    for i in 0..n_rows {
        let row = [1.0, idx_f64(i)];
        let y =
            expected["beta0"].as_f64().unwrap() + expected["beta1"].as_f64().unwrap() * idx_f64(i);
        all_rows.extend_from_slice(&row);
        all_y.push(y);
        if i == n_rows / 2 {
            apply_state_event(
                &mut state,
                StateEvent::AppendData(DataBatchRef {
                    id: Arc::from("b1"),
                    nrows: (n_rows / 2) as u64,
                    bytes: 64,
                }),
            )
            .unwrap();
        }
        diag.append_row(&row, y).unwrap();
    }
    apply_state_event(
        &mut state,
        StateEvent::AppendData(DataBatchRef {
            id: Arc::from("b2"),
            nrows: (n_rows - n_rows / 2) as u64,
            bytes: 64,
        }),
    )
    .unwrap();
    diag.state_version = state.version;
    diag.data_version = state.data_version().raw();
    diag.refresh_summaries().unwrap();
    insert_mechanism_diag(
        &mut state.suff_stats.mechanism_diags,
        Arc::clone(&key),
        diag.clone(),
        &mut state.cache_budget,
    )
    .unwrap();
    assert!(state.suff_stats.mechanism_diags.contains_key(&key));
    assert_eq!(
        state.version.raw(),
        expected["expected_version_after_two_appends"].as_u64().unwrap()
    );

    let start = n_rows - window;
    let mut batch = LinearOlsSuffStats::new(2);
    batch.append_batch(&all_rows[start * 2..], &all_y[start..]).unwrap();
    let beta = batch.solve_beta().unwrap();
    let slot = &state.suff_stats.mechanism_diags[&key];
    assert!((slot.beta[0] - beta[0]).abs() < tol);
    assert!((slot.beta[1] - beta[1]).abs() < tol);
    assert!((slot.beta[0] - expected["beta0"].as_f64().unwrap()).abs() < tol);
    assert!((slot.beta[1] - expected["beta1"].as_f64().unwrap()).abs() < tol);

    apply_state_event(&mut state, StateEvent::ReplaceData(DataVersion::default().next())).unwrap();
    assert!(state.suff_stats.mechanism_diags.is_empty());
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
