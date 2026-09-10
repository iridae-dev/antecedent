//! 1.5 numeric pins: retarget, exceedance, cell AIPW, tier envelopes, joint IF.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]
#![allow(clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent::{
    BatchStudy, CandidateProcedure, CandidateScreen, EstimatorId, PreparedStudy, RefuteSuite,
    SharedBatchDesign, Study,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, DistributionRef, ExecutionContext,
    IdentificationStatus, Intervention, OutcomeFunctional, PopulationRegistry, ResponseFunctional,
    ResponseQuery, TargetPopulation, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{
    BitSet, Cpdag, Dag, DenseNodeId, GraphWorkspace, Pag, TieredBackground, WithinTier,
};
use antecedent_kernels::standard_normal;

fn cols(pairs: &[(&str, Vec<f64>)]) -> TabularData {
    let borrowed: Vec<(&str, &[f64])> = pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    TabularData::from_f64_columns(borrowed).unwrap()
}

fn confounded_hetero(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery, Vec<f64>, f64) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x15);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        let p = 1.0 / (1.0 + (-(-0.3 + 0.9 * zi)).exp());
        t[i] = f64::from(rng.next_f64() < p);
        y[i] = (1.0 + zi) * t[i] + 0.4 * zi + 0.35 * standard_normal(&mut rng);
        w[i] = (-0.5 * ((zi - 0.6) / 0.7).powi(2)).exp();
    }
    let mass: f64 = w.iter().sum();
    let truth: f64 = w.iter().zip(z.iter()).map(|(wi, zi)| wi * (1.0 + zi)).sum::<f64>() / mass;
    let data = cols(&[("t", t), ("y", y), ("z", z)]);
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, graph, query, w, truth)
}

fn ate_study(
    data: TabularData,
    graph: Dag,
    query: AverageEffectQuery,
    estimator: EstimatorId,
) -> Study {
    Study::tabular(data)
        .graph(graph)
        .query(query)
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

const PIN_ABS: f64 = 1e-12;

fn assert_estimates_pin_eq(
    batch: &antecedent::StudyResult,
    solo: &antecedent::StudyResult,
    what: &str,
) {
    let batch_nan = batch.estimate.ate.is_nan();
    let solo_nan = solo.estimate.ate.is_nan();
    assert_eq!(batch_nan, solo_nan, "{what}: ate NaN mismatch");
    if !batch_nan {
        assert!(
            (batch.estimate.ate - solo.estimate.ate).abs() < PIN_ABS,
            "{what}: ate {} vs {}",
            batch.estimate.ate,
            solo.estimate.ate
        );
    }
    match (&batch.estimate.exceedance_cdf, &solo.estimate.exceedance_cdf) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            assert_eq!(a.len(), b.len(), "{what}: cdf length");
            for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
                assert!((ai - bi).abs() < PIN_ABS, "{what}: cdf[{i}] {ai} vs {bi}");
            }
        }
        _ => panic!("{what}: exceedance_cdf presence mismatch"),
    }
}

fn variance_shift_binary(
    n: usize,
    seed: u64,
) -> (TabularData, Dag, AverageEffectQuery, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(seed);
    let q90 = 1.281_551_565_544_600_4;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        t[i] = f64::from(rng.next_f64() < 0.5);
        y[i] = standard_normal(&mut rng) * (1.0 + 0.4 * t[i]);
    }
    let data = cols(&[("t", t), ("y", y)]);
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let mean_q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let exc_q = mean_q.clone().with_outcome_functional(OutcomeFunctional::exceedance(q90));
    (data, graph, mean_q, exc_q)
}

fn tiered_aipw(
    data: TabularData,
    background: TieredBackground,
    query: AverageEffectQuery,
) -> Study {
    Study::tabular(data)
        .tiered_background(background)
        .unwrap()
        .query(query)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// CoDetermined `{z, u} | {t} | {y}`. Same-tier background siblings add z↔u to the
/// closure ADMG. `u` is not a directed descendant of `t`; a pin with singleton
/// tiers never builds that bidirected edge.
fn codetermined_background_siblings(
    n: usize,
    seed: u64,
) -> (TabularData, TieredBackground, AverageEffectQuery, Vec<f64>, VariableId, VariableId) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(seed);
    let mut z = vec![0.0; n];
    let mut u = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let ui = standard_normal(&mut rng);
        z[i] = zi;
        u[i] = ui;
        let p = 1.0 / (1.0 + (-(-0.3 + 0.9 * zi + 0.5 * ui)).exp());
        t[i] = f64::from(rng.next_f64() < p);
        y[i] = (1.0 + zi) * t[i] + 0.4 * zi + 0.3 * ui + 0.35 * standard_normal(&mut rng);
        w[i] = (-0.5 * ((zi - 0.6) / 0.7).powi(2)).exp();
    }
    let data = cols(&[("t", t), ("y", y), ("z", z), ("u", u)]);
    let schema = data.schema().clone();
    let background = TieredBackground::from_named(
        &schema,
        &[vec!["z", "u"], vec!["t"], vec!["y"]],
        WithinTier::CoDetermined,
    )
    .unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    let z_id = schema.id_of("z").unwrap();
    let u_id = schema.id_of("u").unwrap();
    (data, background, query, w, z_id, u_id)
}

/// CoDetermined `{z} | {t, u} | {y}`. Same-tier treatment peer adds t↔u.
/// `u` is in the certified set and is not a directed descendant of `t`.
fn codetermined_treatment_sibling(
    n: usize,
    seed: u64,
) -> (TabularData, TieredBackground, AverageEffectQuery, Vec<f64>, VariableId) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(seed);
    let mut z = vec![0.0; n];
    let mut u = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let latent = standard_normal(&mut rng);
        z[i] = zi;
        u[i] = latent + 0.35 * standard_normal(&mut rng);
        let p = 1.0 / (1.0 + (-(-0.3 + 0.8 * zi + 0.45 * latent)).exp());
        t[i] = f64::from(rng.next_f64() < p);
        y[i] = (1.0 + zi) * t[i] + 0.4 * zi + 0.3 * u[i] + 0.35 * standard_normal(&mut rng);
        w[i] = (-0.5 * ((u[i] - 0.3) / 0.8).powi(2)).exp();
    }
    let data = cols(&[("t", t), ("y", y), ("z", z), ("u", u)]);
    let schema = data.schema().clone();
    let background = TieredBackground::from_named(
        &schema,
        &[vec!["z"], vec!["t", "u"], vec!["y"]],
        WithinTier::CoDetermined,
    )
    .unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    (data, background, query, w, schema.id_of("u").unwrap())
}

fn assert_bidirected_sibling_not_descendant(
    background: &TieredBackground,
    schema: &antecedent_core::CausalSchema,
    treatment: VariableId,
    sibling: VariableId,
) {
    let admg = background.to_admg(schema).unwrap();
    assert!(admg.has_bidirected(), "same-tier CoDetermined must materialize ↔");
    let t = DenseNodeId::from_raw(treatment.raw());
    let s = DenseNodeId::from_raw(sibling.raw());
    let mut out = BitSet::default();
    let mut ws = GraphWorkspace::default();
    admg.descendants_of(&[t], &mut out, &mut ws);
    assert!(!out.contains(s), "bidirected sibling must not be a directed descendant of treatment");
    assert!(
        admg.bidirected_neighbors(s).iter().any(|&v| v != t)
            || admg.bidirected_neighbors(t).contains(&s),
        "sibling must participate in a bidirected edge (same-tier CoDetermined)"
    );
}

fn retarget_grid_matches_standalone(
    data: &TabularData,
    background: TieredBackground,
    query: AverageEffectQuery,
    weights: &[f64],
    depends_on: &[VariableId],
    ctx: &ExecutionContext,
) -> antecedent::StudyResult {
    let thresholds = [0.0_f64, 0.5, 1.0];
    let mid = thresholds[1];
    let grid_q =
        query.clone().with_outcome_functional(OutcomeFunctional::exceedance_grid(thresholds));
    let exc_q = query.with_outcome_functional(OutcomeFunctional::exceedance(mid));
    let prepared =
        tiered_aipw(data.clone(), background.clone(), grid_q.clone()).prepare(ctx).unwrap();
    assert!(prepared.score_table().is_some(), "CoDetermined AIPW must freeze scores");
    let table = prepared.score_table().unwrap();
    for &v in depends_on {
        assert!(
            table.adjustment_set.iter().any(|&z| z == v),
            "depends_on variable must sit in the certified tier-closure set"
        );
    }
    let retargeted = prepared.retarget(weights, depends_on, ctx).unwrap();
    let cdf = retargeted.estimate.exceedance_cdf.as_ref().expect("retargeted grid F_a(c)");
    assert_eq!(cdf.len(), thresholds.len() * 2);
    assert!(
        retargeted.estimate.ate.is_nan(),
        "grid retarget must not publish a first-threshold ATE"
    );

    // CustomDistribution first-click is licensed for a scalar exceedance (Y → 1{Y>c}),
    // not a multi-threshold grid. Pin the grid contrast at c to that standalone path.
    let exc_retarget = tiered_aipw(data.clone(), background.clone(), exc_q.clone())
        .prepare(ctx)
        .unwrap()
        .retarget(weights, depends_on, ctx)
        .unwrap();
    let mut registry = PopulationRegistry::new();
    let href = DistributionRef::from_raw(1);
    registry.insert_distribution_with_dependence(href, weights.to_vec(), depends_on.to_vec());
    let standalone = Study::tabular(data.clone())
        .tiered_background(background)
        .unwrap()
        .query(exc_q.with_target_population(TargetPopulation::CustomDistribution(href)))
        .population_registry(registry)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(ctx)
        .unwrap();
    assert!(
        (exc_retarget.estimate.ate - standalone.estimate.ate).abs() < PIN_ABS,
        "tiered exceedance retarget {} vs CustomDistribution {}",
        exc_retarget.estimate.ate,
        standalone.estimate.ate
    );
    // columns are (arm0,c), (arm1,c) per threshold; ATE = F_0(c) - F_1(c).
    let grid_contrast = cdf[2] - cdf[3];
    assert!(
        (grid_contrast - exc_retarget.estimate.ate).abs() < PIN_ABS,
        "grid contrast at {mid} {} vs exceedance {}",
        grid_contrast,
        exc_retarget.estimate.ate
    );
    retargeted
}

#[test]
fn retarget_matches_refit_and_covers_kernel_target() {
    let (data, graph, query, weights, truth) = confounded_hetero(2_400, 15);
    let z = VariableId::from_raw(2);
    let ctx = ExecutionContext::for_tests(15);
    let prepared: PreparedStudy =
        ate_study(data.clone(), graph.clone(), query.clone(), EstimatorId::Aipw)
            .prepare(&ctx)
            .unwrap();
    assert!(prepared.score_table().is_some());
    let retargeted = prepared.retarget(&weights, &[z], &ctx).unwrap();
    assert!(
        retargeted.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
        "retarget must reuse prepare-time identification"
    );
    assert!(
        (retargeted.estimate.ate - truth).abs() < 1.96 * retargeted.estimate.se_analytic,
        "retarget ate={} se={} truth={truth}",
        retargeted.estimate.ate,
        retargeted.estimate.se_analytic
    );

    let mut registry = PopulationRegistry::new();
    let href = DistributionRef::from_raw(1);
    registry.insert_distribution_with_dependence(href, weights.clone(), [z]);
    let fresh = Study::tabular(data.clone())
        .graph(graph)
        .query(query.with_target_population(TargetPopulation::CustomDistribution(href)))
        .population_registry(registry)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(
        (fresh.estimate.ate - retargeted.estimate.ate).abs() < 1e-12,
        "fresh={} retarget={}",
        fresh.estimate.ate,
        retargeted.estimate.ate
    );

    let mut refreshed = prepared;
    let click = refreshed.refresh(data, &ctx).unwrap();
    assert!(click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
}

#[test]
fn retarget_refuses_treatment_dependence() {
    let (data, graph, query, weights, _) = confounded_hetero(400, 16);
    let ctx = ExecutionContext::for_tests(16);
    let prepared = ate_study(data, graph, query, EstimatorId::Aipw).prepare(&ctx).unwrap();
    let err = prepared.retarget(&weights, &[VariableId::from_raw(0)], &ctx).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("treatment") || msg.contains("depends_on"), "{msg}");
}

#[test]
fn retarget_weighted_overlap_is_support() {
    let mut rng = ExecutionContext::for_tests(17).rng.stream(1);
    let n = 800usize;
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        t[i] = f64::from(zi < 0.0 && rng.next_f64() < 0.7);
        y[i] = t[i] + 0.2 * standard_normal(&mut rng);
        w[i] = f64::from(zi > 1.2);
    }
    let data = cols(&[("t", t), ("y", y), ("z", z)]);
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let ctx = ExecutionContext::for_tests(17);
    let prepared = ate_study(data, graph, query, EstimatorId::Aipw).prepare(&ctx).unwrap();
    let err = prepared.retarget(&w, &[VariableId::from_raw(2)], &ctx).unwrap_err();
    assert!(matches!(err, antecedent::CausalError::Support { .. }), "{err}");
}

#[test]
fn variance_only_exceedance_covers_and_mean_misses() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/v15_exceedance_variance/expected.json"
    ))
    .unwrap();
    let mean_tol = pin["mean_tolerance"].as_f64().unwrap();
    let exceedance_delta = pin["exceedance_delta"].as_f64().unwrap();
    assert_eq!(pin["estimator"], "aipw");
    let mut rng = ExecutionContext::for_tests(18).rng.stream(2);
    let n = 3_000usize;
    let q90 = 1.281_551_565_544_600_4;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        t[i] = f64::from(rng.next_f64() < 0.5);
        y[i] = standard_normal(&mut rng) * (1.0 + 0.4 * t[i]);
    }
    let data = cols(&[("t", t), ("y", y)]);
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let mean_q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let exc_q = mean_q.clone().with_outcome_functional(OutcomeFunctional::exceedance(q90));
    let ctx = ExecutionContext::for_tests(18);
    let mean = ate_study(data.clone(), graph.clone(), mean_q, EstimatorId::Aipw).run(&ctx).unwrap();
    let exc = ate_study(data, graph, exc_q, EstimatorId::Aipw).run(&ctx).unwrap();
    assert!(
        mean.estimate.ate.abs() < mean_tol,
        "mean path must miss the tail shift, ate={}",
        mean.estimate.ate
    );
    let lo = exc.estimate.ate - 1.96 * exc.estimate.se_analytic;
    let hi = exc.estimate.ate + 1.96 * exc.estimate.se_analytic;
    assert!(
        lo <= exceedance_delta && exceedance_delta <= hi,
        "exceedance must cover +{exceedance_delta}, ate={} se={}",
        exc.estimate.ate,
        exc.estimate.se_analytic
    );
}

#[test]
fn exceedance_grid_on_fresh_estimate_fills_cdf() {
    let (data, graph, query, _, _) = confounded_hetero(1_200, 19);
    let thresholds = [0.0_f64, 0.5, 1.0];
    let grid_q = query.with_outcome_functional(OutcomeFunctional::exceedance_grid(thresholds));
    let ctx = ExecutionContext::for_tests(19);
    let study = ate_study(data.clone(), graph.clone(), grid_q.clone(), EstimatorId::Aipw);
    let fresh = study.run(&ctx).unwrap();
    let cdf = fresh.estimate.exceedance_cdf.as_ref().expect("fresh estimate must publish F_a(c)");
    assert_eq!(cdf.len(), thresholds.len() * 2, "one F_a(c) per (arm, threshold)");
    assert!(cdf.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    assert!(fresh.estimate.score_table.is_some());
    assert!(fresh.estimate.joint_covariance.is_some());
    assert!(
        fresh.estimate.ate.is_nan(),
        "multi-threshold grids must not publish a first-threshold scalar ATE"
    );
    assert!(fresh.estimate.monotone_rearranged);
    assert!(fresh.estimate.simultaneous_interval.is_none());
    assert!(
        fresh
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.functional.grid_scalar_cleared")
    );

    let prepared = ate_study(data.clone(), graph, grid_q, EstimatorId::Aipw).prepare(&ctx).unwrap();
    let click = prepared.estimate(&data, &ctx).unwrap();
    let click_cdf =
        click.estimate.exceedance_cdf.as_ref().expect("prepared estimate must publish F_a(c)");
    assert_eq!(click_cdf.len(), thresholds.len() * 2);
    assert!(click.estimate.score_table.is_some());
    assert!(click.estimate.joint_covariance.is_some());
    assert!(click.estimate.ate.is_nan());
}

fn interaction_dgp(n: usize, seed: u64) -> (TabularData, Dag) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0xAD);
    let mut a = vec![0.0; n];
    let mut d = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        z[i] = standard_normal(&mut rng);
        a[i] = f64::from(rng.next_f64() < 0.5);
        d[i] = f64::from(rng.next_f64() < 0.5);
        y[i] = 1.5 * a[i] * d[i] + 0.25 * z[i] + 0.3 * standard_normal(&mut rng);
    }
    let data = cols(&[("a", a), ("d", d), ("y", y), ("z", z)]);
    let mut graph = Dag::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    (data, graph)
}

fn joint_query(a: f64, d: f64) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(0), Value::f64(a)),
            Intervention::set(VariableId::from_raw(1), Value::f64(d)),
        ]),
    })
}

fn response_value(result: &antecedent::StudyResult) -> f64 {
    result.response.as_ref().map_or(result.estimate.ate, |r| match &r.estimate {
        antecedent_core::ResponseIdentification::PointIdentified(
            antecedent_core::ResponseValue::Scalar(v),
        ) => *v,
        _ => result.estimate.ate,
    })
}

#[test]
fn additive_interaction_is_structurally_zero_cell_aipw_recovers() {
    let (data, graph) = interaction_dgp(2_200, 19);
    let ctx = ExecutionContext::for_tests(19);
    let additive = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Response(joint_query(1.0, 1.0)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(
        additive.response.as_ref().is_some_and(|r| r.interaction_structurally_zero),
        "additive path must disclose structurally zero interaction on the result"
    );
    let mut cells = Vec::new();
    for (a, d) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
        let r = Study::tabular(data.clone())
            .graph(graph.clone())
            .query(CausalQuery::Response(joint_query(a, d)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        cells.push(response_value(&r));
    }
    let additive_ix = (cells[3] - cells[2]) - (cells[1] - cells[0]);
    assert!(additive_ix.abs() < 0.25, "additive interaction={additive_ix}");

    let cell = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(joint_query(1.0, 1.0)))
        .estimator(EstimatorId::CellAipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(cell.response.as_ref().is_some_and(|r| !r.interaction_structurally_zero));
    let lo = cell.estimate.ate - 1.96 * cell.estimate.se_analytic;
    let hi = cell.estimate.ate + 1.96 * cell.estimate.se_analytic;
    assert!(
        lo <= 1.5 && 1.5 <= hi,
        "cell AIPW interaction={} se={}",
        cell.estimate.ate,
        cell.estimate.se_analytic
    );
}

#[test]
fn cell_aipw_k3_and_empty_cell_refuse() {
    let mut rng = ExecutionContext::for_tests(20).rng.stream(3);
    let n = 2_400usize;
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        z[i] = standard_normal(&mut rng);
        a[i] = f64::from(rng.next_f64() < 0.5);
        b[i] = f64::from(rng.next_f64() < 0.5);
        c[i] = f64::from(rng.next_f64() < 0.5);
        y[i] = 0.9 * a[i] * b[i] * c[i] + 0.2 * z[i] + 0.25 * standard_normal(&mut rng);
    }
    let data = cols(&[("a", a), ("b", b), ("c", c), ("y", y), ("z", z)]);
    let mut graph = Dag::with_variables(5);
    graph.insert_directed(DenseNodeId::from_raw(4), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(4), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(4), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(4), DenseNodeId::from_raw(3)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(3)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(3)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(3),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(2), Value::f64(1.0)),
        ]),
    });
    let ctx = ExecutionContext::for_tests(20);
    let ok = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .estimator(EstimatorId::CellAipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(ok.estimate.score_table.as_ref().is_some_and(|t| t.n_columns() == 8));

    let mut rng = ExecutionContext::for_tests(21).rng.stream(4);
    let n = 600usize;
    let mut a = vec![0.0; n];
    let mut d = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        z[i] = standard_normal(&mut rng);
        a[i] = f64::from(rng.next_f64() < 0.5);
        d[i] = f64::from(a[i] < 0.5 && rng.next_f64() < 0.5);
        y[i] = a[i] + 0.2 * z[i];
    }
    let data = cols(&[("a", a), ("d", d), ("y", y), ("z", z)]);
    let mut graph = Dag::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let err = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(joint_query(1.0, 1.0)))
        .estimator(EstimatorId::CellAipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(21))
        .unwrap_err();
    assert!(err.to_string().contains("unsupported cell"), "{err}");
}

#[test]
fn unknown_tier_envelope_straddles_zero() {
    let mut rng = ExecutionContext::for_tests(22).rng.stream(5);
    let n = 2_000usize;
    let mut era = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        era[i] = standard_normal(&mut rng);
        t[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-era[i]).exp()));
        m[i] = t[i] + 0.2 * era[i] + 0.2 * standard_normal(&mut rng);
        y[i] = -1.0 * t[i] + 2.0 * m[i] + 0.2 * era[i] + 0.3 * standard_normal(&mut rng);
    }
    let data = cols(&[("era", era), ("t", t), ("m", m), ("y", y)]);
    let schema = data.schema().clone();
    let background = TieredBackground::from_named(
        &schema,
        &[vec!["era"], vec!["t", "m"], vec!["y"]],
        WithinTier::Unknown,
    )
    .unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    let result = Study::tabular(data)
        .tiered_background(background)
        .unwrap()
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap();
    assert!(result.estimate.ate.is_nan(), "unknown scenarios must not become a single ATE");
    let scenarios = result.estimate.scenario_effects.as_ref().unwrap();
    assert!(scenarios.iter().any(|v| *v < 0.0) && scenarios.iter().any(|v| *v > 0.0));
    assert_eq!(result.estimate.scenario_intervals.as_ref().unwrap().len(), 2);
    assert_eq!(format!("{:?}", result.identification.status), "GraphDependent");
}

#[test]
fn static_cpdag_and_pag_envelope_se_is_finite() {
    let mut rng = ExecutionContext::for_tests(23).rng.stream(6);
    let n = 1_200usize;
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        z[i] = standard_normal(&mut rng);
        t[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-z[i]).exp()));
        y[i] = 2.0 * t[i] + z[i] + 0.4 * standard_normal(&mut rng);
    }
    let data = cols(&[("t", t.clone()), ("y", y.clone()), ("z", z.clone())]);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let mut cpdag = Cpdag::with_variables(3);
    cpdag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_undirected(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    let cpdag_res = Study::tabular(data.clone())
        .graph(cpdag)
        .query(query.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(23))
        .unwrap();
    assert!(
        cpdag_res.estimate.se_analytic.is_finite() && cpdag_res.estimate.se_analytic > 0.0,
        "CPDAG envelope SE must be finite, se={}",
        cpdag_res.estimate.se_analytic
    );
    assert!(
        cpdag_res
            .diagnostics
            .iter()
            .all(|d| { d.code.as_ref() != "estimate.envelope.se_omits_between_atom_variance" })
    );

    let r: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let pag_data = cols(&[("t", t.clone()), ("y", y.clone()), ("z", z.clone()), ("r", r)]);
    let mut pag = Pag::with_variables(4);
    pag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let pag_res = Study::tabular(pag_data)
        .graph(pag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(24))
        .unwrap();
    assert!(
        pag_res.estimate.se_analytic.is_finite() && pag_res.estimate.se_analytic > 0.0,
        "PAG envelope SE must be finite, se={}",
        pag_res.estimate.se_analytic
    );
    assert!(
        pag_res
            .diagnostics
            .iter()
            .all(|d| { d.code.as_ref() != "estimate.envelope.se_omits_between_atom_variance" })
    );
}

#[test]
fn batch_joint_if_and_overlap_does_not_abort() {
    let (data, graph, query, ..) = confounded_hetero(900, 25);
    let q2 = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let ctx = ExecutionContext::production(25, 2);
    let results = BatchStudy::new(data, graph)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .estimate_many(&[query, q2], &ctx)
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(
        results.iter().any(|r| r.diagnostics.iter().any(|d| d.code.as_ref() == "batch.joint_if"))
    );
    assert!(results.iter().any(|r| {
        r.diagnostics.iter().any(|d| d.code.as_ref() == "batch.candidate_selection.unrecorded")
    }));
}

fn retarget_distinct_lever_loop(n: usize, n_levers: u32, seed: u64, max_secs: f64) {
    let (data, graph, query, _, _) = confounded_hetero(n, seed);
    let z = VariableId::from_raw(2);
    let z_col = data.float64_slice(z).unwrap().to_vec();
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = ate_study(data, graph, query, EstimatorId::Aipw).prepare(&ctx).unwrap();
    let denom = f64::from(n_levers.saturating_sub(1).max(1));
    let levers: Vec<Vec<f64>> = (0..n_levers)
        .map(|k| {
            let center = -2.0 + 4.0 * f64::from(k) / denom;
            z_col.iter().map(|&zi| (-0.5 * ((zi - center) / 0.7).powi(2)).exp()).collect()
        })
        .collect();
    let started = std::time::Instant::now();
    for weights in &levers {
        let out = prepared.retarget(weights, &[z], &ctx).unwrap();
        assert!(out.estimate.ate.is_finite());
    }
    let elapsed = started.elapsed();
    assert!(
        elapsed.as_secs_f64() < max_secs,
        "retarget {n}×{n_levers} distinct levers took {elapsed:?}"
    );
}

#[test]
fn retarget_hot_path_smoke_is_bounded() {
    // Default smoke. The licensed 10k×500 ADR 0011 measurement is the ignored
    // timed harness below plus `retarget_alloc_count` (500 distinct levers).
    retarget_distinct_lever_loop(2_000, 50, 26, 2.0);
}

#[test]
#[ignore = "ADR 0011 10k×500 timed harness; allocation contract is retarget_alloc_count"]
fn retarget_hot_path_10k_by_500_is_bounded() {
    retarget_distinct_lever_loop(10_000, 500, 26, 8.0);
}

#[test]
fn joint_response_honors_requested_cell_and_retarget() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/v15_cell_aipw_interaction/expected.json"
    ))
    .unwrap();
    let truth = pin["true_effect"].as_f64().unwrap();
    let tol = pin["tolerance"].as_f64().unwrap();
    assert_eq!(pin["estimator"], "cell.aipw");
    let (data, graph) = interaction_dgp(2400, 81);
    let ctx = ExecutionContext::for_tests(81);
    let make = |a, d| {
        Study::tabular(data.clone())
            .graph(graph.clone())
            .query(CausalQuery::Response(joint_query(a, d)))
            .estimator(EstimatorId::CellAipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    };
    let baseline = make(0.0, 0.0).run(&ctx).unwrap();
    let active = make(1.0, 1.0).run(&ctx).unwrap();
    assert!(response_value(&baseline).abs() < 0.1);
    assert!((response_value(&active) - truth).abs() < tol);
    assert!((response_value(&active) - response_value(&baseline)) > 1.3);
    let prepared = make(1.0, 1.0).prepare(&ctx).unwrap();
    let table = prepared.score_table().unwrap();
    let retargeted = prepared.retarget(&vec![1.0; table.n_rows], &[], &ctx).unwrap();
    assert!((response_value(&retargeted) - response_value(&active)).abs() < 1e-12);
    let invalid = make(2.0, 1.0).run(&ctx).unwrap_err();
    assert!(invalid.to_string().contains("0/1"));
}

#[test]
fn prepared_grid_uses_new_data_and_joint_covariance_roundtrips() {
    let (data, graph, q, _, _) = confounded_hetero(1000, 83);
    let q = q.with_outcome_functional(OutcomeFunctional::exceedance_grid([0.0, 1.0]));
    let ctx = ExecutionContext::for_tests(83);
    let prepared =
        ate_study(data, graph.clone(), q.clone(), EstimatorId::Aipw).prepare(&ctx).unwrap();
    let (new_data, _, _, _, _) = confounded_hetero(1300, 84);
    let click = prepared.estimate(&new_data, &ctx).unwrap();
    let fresh = ate_study(new_data, graph, q, EstimatorId::Aipw).run(&ctx).unwrap();
    assert_eq!(click.estimate.score_table.as_ref().unwrap().n_rows, 1300);
    assert_eq!(click.estimate.exceedance_cdf, fresh.estimate.exceedance_cdf);
    let wire = antecedent_io::effect_estimate_to_wire(&click.estimate);
    let restored = antecedent_io::effect_estimate_from_wire(&wire).unwrap();
    assert_eq!(restored.joint_covariance, click.estimate.joint_covariance);
    assert_eq!(restored.score_inference, click.estimate.score_inference);
    let inference = click.estimate.score_inference.unwrap();
    assert_eq!(inference.lower.len(), 4);
    assert!(inference.critical_value > 1.9);
}

#[test]
fn kernel_target_population_coverage_over_seed_grid() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/v15_retarget_kernel/expected.json"
    ))
    .unwrap();
    let truth = pin["true_effect"].as_f64().unwrap();
    assert!((truth - (1.0 + 0.6 / (1.0 + 0.7_f64.powi(2)))).abs() < 1e-9);
    assert_eq!(pin["estimator"], "aipw");
    let mut covered = 0;
    for seed in 100..180 {
        let (data, graph, q, weights, _) = confounded_hetero(900, seed);
        let ctx = ExecutionContext::for_tests(seed);
        let plan = ate_study(data, graph, q, EstimatorId::Aipw).prepare(&ctx).unwrap();
        let result = plan.retarget(&weights, &[VariableId::from_raw(2)], &ctx).unwrap();
        covered +=
            usize::from((result.estimate.ate - truth).abs() <= 1.96 * result.estimate.se_analytic);
    }
    // For 80 repetitions, 68 is a conservative Monte Carlo lower guard for nominal 95% coverage.
    assert!(covered >= 68, "kernel-target coverage {covered}/80");
}

#[test]
fn cpdag_shared_row_aggregate_coverage_over_seed_grid() {
    let mut covered = 0;
    for seed in 200..280 {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(15);
        let n = 500;
        let mut z = Vec::new();
        let mut t = Vec::new();
        let mut y = Vec::new();
        for _ in 0..n {
            let zi = standard_normal(&mut rng);
            let ti = zi + standard_normal(&mut rng);
            z.push(zi);
            t.push(ti);
            y.push(2.0 * ti + zi + standard_normal(&mut rng));
        }
        let data = cols(&[("t", t), ("y", y), ("z", z)]);
        let mut g = Cpdag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
        g.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let result = Study::tabular(data)
            .graph(g)
            .query(q)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(seed))
            .unwrap();
        // Two equally weighted observational functionals: adjusted beta_T=2,
        // unadjusted beta_T=2+Cov(T,Z)/Var(T)=2.5. Not a posterior over causal truths.
        covered +=
            usize::from((result.estimate.ate - 2.25).abs() <= 1.96 * result.estimate.se_analytic);
    }
    assert!(covered >= 68, "CPDAG frozen-functional coverage {covered}/80");
}

#[test]
fn prepared_score_failure_is_visible_and_refresh_keeps_previous_scores() {
    let (data, graph, query, weights, _) = confounded_hetero(400, 180);
    let ctx = ExecutionContext::for_tests(180);
    let mut plan = ate_study(data.clone(), graph.clone(), query.clone(), EstimatorId::Aipw)
        .prepare(&ctx)
        .unwrap();
    let original = plan.score_table().unwrap().scores.clone();
    let invalid = data.with_replaced_float(VariableId::from_raw(0), vec![0.0; 400].into()).unwrap();
    assert!(plan.refresh(invalid.clone(), &ctx).is_err());
    assert_eq!(plan.score_table().unwrap().scores.as_ref(), original.as_ref());
    assert!(plan.retarget(&weights, &[VariableId::from_raw(2)], &ctx).is_ok());
    assert!(ate_study(invalid, graph, query, EstimatorId::Aipw).prepare(&ctx).is_err());
}

#[test]
fn prepared_and_fresh_scores_honor_configured_propensity_fit() {
    let (data, graph, query, _, _) = confounded_hetero(400, 181);
    let mut estimator = antecedent_estimate::AipwAte::new();
    estimator.bootstrap_replicates = 0;
    estimator.glm_options.max_iter = 0;
    let study = Study::tabular(data)
        .graph(graph)
        .query(query.with_outcome_functional(OutcomeFunctional::exceedance_grid([0.0, 1.0])))
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(181);
    assert!(study.prepare(&ctx).is_err());
    assert!(study.run(&ctx).is_err());
}

#[test]
fn unsupported_grid_cannot_silently_return_only_first_threshold() {
    let (data, graph, query, _, _) = confounded_hetero(100, 182);
    let result = Study::tabular(data)
        .graph(graph)
        .query(query.with_outcome_functional(OutcomeFunctional::exceedance_grid([0.0, 1.0])))
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build();
    assert!(matches!(result, Err(antecedent::CausalError::Unsupported { .. })));
}

#[test]
fn intervention_response_attaches_plugin_if_covariance() {
    let (data, graph) = interaction_dgp(800, 190);
    let ctx = ExecutionContext::for_tests(190);
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(joint_query(1.0, 1.0)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(result.estimate.se_analytic.is_finite() && result.estimate.se_analytic > 0.0);
    assert!(result.estimate.joint_covariance.is_some());
    assert!(result.estimate.influence.is_some());
}

#[test]
fn conditional_exceedance_grid_publishes_per_arm_cdf() {
    let (data, graph, _, _, _) = confounded_hetero(900, 191);
    let inner = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_effect_modifiers([VariableId::from_raw(2)])
        .with_outcome_functional(OutcomeFunctional::exceedance_grid([0.0, 0.5, 1.0]));
    let query = ConditionalEffectQuery::try_new(inner).unwrap();
    let ctx = ExecutionContext::for_tests(191);
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::ConditionalEffect(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let cdf = result.estimate.exceedance_cdf.expect("conditional grid F_a(c)");
    assert_eq!(cdf.len(), 6);
    assert!(cdf.iter().all(|v| v.is_finite()));
    let cov = result.estimate.joint_covariance.expect("per-arm CDF covariance");
    assert_eq!(cov.dim, 6, "joint covariance must be 2K per-arm coordinates, not K contrasts");
    let inf = result.estimate.score_inference.expect("per-arm simultaneous bands");
    assert_eq!(inf.raw_means.len(), 6);
    assert_eq!(inf.lower.len(), 6);
    assert_eq!(inf.upper.len(), 6);
    assert_eq!(inf.threshold_supported.len(), 6);
    assert!(
        inf.threshold_supported.iter().any(|ok| *ok),
        "interior thresholds on this DGP must have tail support"
    );
    assert!(
        inf.lower.iter().zip(&inf.threshold_supported).any(|(lo, ok)| *ok && lo.is_finite()),
        "supported arms must publish a finite raw-score band"
    );
    assert!(
        result.estimate.ate.is_nan(),
        "ConditionalEffect grid must not publish a first-threshold scalar"
    );
    assert!(result.estimate.influence.is_none());
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.functional.grid_scalar_cleared")
    );
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.functional.cdf_inference")
    );
}

#[test]
fn prepared_batch_reuses_plans_and_records_screen() {
    let (data, graph, query, _, _) = confounded_hetero(700, 192);
    let q2 = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let ctx = ExecutionContext::for_tests(192);
    let n = u32::try_from(data.row_count()).unwrap();
    let screen_rows: Vec<u32> = (0..n / 2).collect();
    let estimate_rows: Vec<u32> = (n / 2..n).collect();
    let n_estimate = estimate_rows.len();
    let batch = BatchStudy::new(data.clone(), graph)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .candidate_screen(CandidateScreen {
            screen_id: Arc::from("v15.screen"),
            procedure: CandidateProcedure::BenjaminiHochberg,
            screen_rows: screen_rows.into(),
            estimate_rows: estimate_rows.clone().into(),
        });
    let prepared = batch.prepare(&[query, q2], &ctx).unwrap();
    assert_eq!(prepared.plans().len(), 2);
    let results = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(results.len(), 2);
    let sel = results[0].candidate_selection.as_ref().expect("selection provenance");
    assert_eq!(sel.screen_id.as_ref(), "v15.screen");
    assert_eq!(sel.procedure, CandidateProcedure::BenjaminiHochberg);
    assert!(sel.disjoint);
    let recorded = results[0].estimate.candidate_selection.as_ref().expect("estimate artifact");
    assert_eq!(recorded.screen_id.as_ref(), "v15.screen");
    assert_eq!(recorded.procedure.as_ref(), "bh");
    assert!(!recorded.screen_rows.is_empty());
    assert!(!recorded.estimate_rows.is_empty());
    let wire = antecedent_io::effect_estimate_to_wire(&results[0].estimate);
    let restored = antecedent_io::effect_estimate_from_wire(&wire).unwrap();
    assert_eq!(
        restored.candidate_selection.as_ref().map(|s| s.screen_rows.len()),
        Some(recorded.screen_rows.len())
    );
    assert!(
        results.iter().any(|r| r.diagnostics.iter().any(|d| d.code.as_ref() == "batch.joint_if"))
    );
    assert!(
        results
            .iter()
            .any(|r| r.diagnostics.iter().any(|d| d.code.as_ref() == "batch.candidate_selection"))
    );
    assert!(results.iter().any(|r| {
        r.diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "batch.candidate_selection.ranked_on_reported_family")
    }));
    let inf = results[0].estimate.influence.as_ref().expect("estimate-sample IF");
    assert_eq!(
        inf.len(),
        n_estimate,
        "estimate_rows must restrict the complete-case universe, not only record provenance"
    );
}

#[test]
fn tiered_200_node_certified_set_is_valid_and_evalue_attaches() {
    let n_nodes = 200u32;
    let n = 800;
    let mut names = Vec::with_capacity(n_nodes as usize);
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(n_nodes as usize);
    let mut rng = ExecutionContext::for_tests(200).rng.stream(0xC8);
    let latent: Vec<Vec<f64>> =
        (0..20).map(|_| (0..n).map(|_| standard_normal(&mut rng)).collect()).collect();
    for i in 0..n_nodes as usize {
        names.push(format!("v{i}"));
        let mut col: Vec<f64> =
            (0..n).map(|r| 0.3 * latent[i / 10][r] + standard_normal(&mut rng)).collect();
        if i == 50 {
            for (r, value) in col.iter_mut().enumerate() {
                *value += 0.4 * columns[0][r];
            }
        }
        if i == 199 {
            for (r, value) in col.iter_mut().enumerate() {
                *value = 1.8 * columns[50][r]
                    + 0.4 * columns[0][r]
                    + 0.2 * columns[51][r]
                    + standard_normal(&mut rng);
            }
        }
        columns.push(col);
    }
    let pairs: Vec<(&str, Vec<f64>)> = names.iter().map(String::as_str).zip(columns).collect();
    let borrowed: Vec<(&str, &[f64])> = pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let schema = data.schema();
    let mut tiers = Vec::new();
    for chunk in names.chunks(10) {
        tiers.push(chunk.iter().map(String::as_str).collect::<Vec<_>>());
    }
    let background =
        TieredBackground::from_named(schema, &tiers, WithinTier::CoDetermined).unwrap();
    let t = schema.id_of("v50").unwrap();
    let y = schema.id_of(&names[199]).unwrap();
    let identified =
        antecedent_identify::identify_tiered(&background, &AverageEffectQuery::binary_ate(t, y))
            .unwrap();
    let certified = identified.estimands[0].adjustment_set.clone();
    let expected = background.tier_closure(t, y).unwrap();
    assert_eq!(certified.as_ref(), expected.as_ref());
    assert!(certified.contains(&schema.id_of("v49").unwrap()));
    assert!(!certified.contains(&t));
    assert!(!certified.contains(&y));
    let mut backdoor = antecedent_graph::Admg::with_variables(n_nodes);
    for (k, tier) in background.tiers.iter().enumerate() {
        for later in background.tiers.iter().skip(k + 1) {
            for &u in tier.iter().filter(|&&u| u != t) {
                for &v in later.iter() {
                    backdoor
                        .insert_directed(
                            DenseNodeId::from_raw(u.raw()),
                            DenseNodeId::from_raw(v.raw()),
                        )
                        .unwrap();
                }
            }
        }
        for (i, &u) in tier.iter().enumerate() {
            for &v in tier.iter().skip(i + 1) {
                backdoor
                    .insert_bidirected(
                        DenseNodeId::from_raw(u.raw()),
                        DenseNodeId::from_raw(v.raw()),
                    )
                    .unwrap();
            }
        }
    }
    let z: Vec<_> = certified.iter().map(|v| DenseNodeId::from_raw(v.raw())).collect();
    assert!(
        backdoor
            .is_m_separated(
                DenseNodeId::from_raw(t.raw()),
                DenseNodeId::from_raw(y.raw()),
                &z,
                &mut antecedent_graph::DSeparationWorkspace::default()
            )
            .unwrap()
    );
    let result = Study::tabular(data)
        .tiered_background(background)
        .unwrap()
        .query(AverageEffectQuery::binary_ate(t, y))
        .identifier("backdoor.adjustment".parse().unwrap())
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(200))
        .unwrap();
    assert!((result.estimate.ate - 1.8).abs() < 1.96 * result.estimate.se_analytic);
    assert!(result.estimate.evalue.is_some_and(|e| e.is_finite() && e >= 1.0));
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "tiered.evalue.vanderweele_approx")
    );
    assert!(
        result.diagnostics.iter().any(|d| {
            d.code.as_ref() == "tiered.evalue.vanderweele_approx"
                && d.message.contains("not a tier-identification certificate")
        }),
        "CoDetermined E-value must disclose the VanderWeele approx, not new ID theory"
    );
}

#[test]
fn pag_intervention_response_attaches_shared_row_if() {
    let (data, _, query, _, _) = confounded_hetero(900, 201);
    let mut pag = Pag::with_variables(4);
    pag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let r: Vec<f64> = (0..data.row_count()).map(|i| (i % 2) as f64).collect();
    let pag_data = cols(&[
        ("t", data.float64_values(VariableId::from_raw(0)).unwrap()),
        ("y", data.float64_values(VariableId::from_raw(1)).unwrap()),
        ("z", data.float64_values(VariableId::from_raw(2)).unwrap()),
        ("r", r),
    ]);
    let response = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: query.outcome,
        interventions: Arc::from([Intervention::set(query.treatment, Value::f64(1.0))]),
    });
    let result = Study::tabular(pag_data)
        .graph(pag)
        .query(CausalQuery::Response(response))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(201))
        .unwrap();
    assert!(result.estimate.se_analytic.is_finite());
    assert!(result.estimate.joint_covariance.is_some() || result.estimate.influence.is_some());
}

#[test]
fn cell_aipw_cheap_runs_overlap_and_evalue() {
    let (data, graph) = interaction_dgp(800, 202);
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(joint_query(1.0, 1.0)))
        .estimator(EstimatorId::CellAipw)
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(202))
        .unwrap();
    assert!(result.estimate.ate.is_finite());
    assert!(
        result.refutations.iter().any(|r| r.refuter.as_ref() == "sensitivity.evalue")
            || result.diagnostics.iter().any(|d| d.code.as_ref().contains("evalue"))
    );
}

#[test]
fn continuous_cell_refuses_point_cde() {
    let data = cols(&[
        ("a", vec![0.0, 1.0, 0.0, 1.0]),
        ("d", vec![0.1, 0.4, 0.6, 0.9]),
        ("y", vec![1.0, 2.0, 1.5, 2.5]),
        ("z", vec![0.2, -0.1, 0.0, 0.3]),
    ]);
    let mut graph = Dag::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    });
    let err = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .estimator(EstimatorId::CellAipw)
        .continuous_cell(VariableId::from_raw(1), Arc::from([0.0, 0.5, 1.0]))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap_err();
    assert!(
        err.to_string().contains("do(D=d0)"),
        "point CDE must be a named refuse, not a coarsened grid: {err}"
    );
}

#[test]
fn prepared_batch_cells_reuse_joint_plans() {
    let (data, graph) = interaction_dgp(600, 204);
    let ctx = ExecutionContext::for_tests(204);
    let prepared = BatchStudy::new(data.clone(), graph)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare_cells(&[joint_query(1.0, 1.0), joint_query(1.0, 0.0)], &ctx)
        .unwrap();
    assert_eq!(prepared.plans().len(), 2);
    let results = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.estimate.ate.is_finite()));
}

#[test]
fn linear_plan_has_no_score_table_and_refuses_retarget() {
    let (data, graph, query, weights, _) = confounded_hetero(300, 17);
    let ctx = ExecutionContext::for_tests(17);
    let first =
        ate_study(data.clone(), graph.clone(), query.clone(), EstimatorId::LinearAdjustmentAte)
            .run(&ctx)
            .unwrap();
    assert!(
        first.estimate.score_table.is_none(),
        "first-click linear analyze/run must not invent a score table"
    );
    assert!(!first.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.aipw.crossfit_scores"));
    let prepared =
        ate_study(data, graph, query, EstimatorId::LinearAdjustmentAte).prepare(&ctx).unwrap();
    assert!(prepared.score_table().is_none());
    let err = prepared.retarget(&weights, &[VariableId::from_raw(2)], &ctx).unwrap_err();
    assert!(
        err.to_string().contains("score table"),
        "linear retarget must refuse without inventing AIPW scores: {err}"
    );
}

#[test]
fn class_aware_conditional_grid_mixes_envelope_atoms() {
    let mut rng = ExecutionContext::for_tests(205).rng.stream(0xCD);
    let n = 1_200usize;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut u = vec![0.0; n];
    for i in 0..n {
        let ui = standard_normal(&mut rng);
        let zi = standard_normal(&mut rng);
        u[i] = ui;
        z[i] = zi;
        t[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-ui).exp()));
        y[i] = t[i] + 0.9 * ui + 0.35 * zi + 0.25 * standard_normal(&mut rng);
    }
    let data = cols(&[("t", t), ("y", y), ("z", z), ("u", u)]);
    // t=0, y=1, z=2, u=3. Uncertain T—U orientation; Z is a shared modifier.
    let mut cpdag = Cpdag::with_variables(4);
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(3)).unwrap();
    let mut dag_u_into_t = Dag::with_variables(4);
    dag_u_into_t.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
    dag_u_into_t.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag_u_into_t.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag_u_into_t.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(1)).unwrap();
    let inner = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_effect_modifiers([VariableId::from_raw(2)])
        .with_outcome_functional(OutcomeFunctional::exceedance_grid([0.0, 0.5, 1.0]));
    let query = ConditionalEffectQuery::try_new(inner).unwrap();
    let ctx = ExecutionContext::for_tests(205);
    let mixed = Study::tabular(data.clone())
        .graph(cpdag)
        .query(CausalQuery::ConditionalEffect(query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let primary = Study::tabular(data)
        .graph(dag_u_into_t)
        .query(CausalQuery::ConditionalEffect(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let mixed_cdf = mixed.estimate.exceedance_cdf.expect("class-aware grid F_a(c)");
    let primary_cdf = primary.estimate.exceedance_cdf.expect("primary-atom grid F_a(c)");
    assert_eq!(mixed_cdf.len(), 6);
    assert!(mixed_cdf.iter().all(|v| v.is_finite()));
    assert!(mixed.estimate.joint_covariance.is_some());
    assert!(
        mixed_cdf.iter().zip(primary_cdf.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
        "class-aware grid must mix envelope atoms, not copy the primary estimand"
    );
    assert!(
        mixed.estimate.influence.is_none(),
        "class-aware grids must not keep a mean-CATE or first-threshold IF as the scalar influence"
    );
    assert!(
        mixed.estimate.ate.is_nan(),
        "class-aware grid must not publish a mean-CATE scalar as the grid"
    );
    assert!(
        mixed
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.functional.grid_scalar_cleared")
    );
}

#[test]
fn dag_intervention_response_cheap_runs_overlap_and_evalue() {
    let (data, graph) = interaction_dgp(800, 206);
    let ctx = ExecutionContext::for_tests(206);
    let cell = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Response(joint_query(1.0, 1.0)))
        .estimator(EstimatorId::CellAipw)
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(cell.estimate.ate.is_finite());
    assert!(
        cell.refutations.iter().any(|r| r.refuter.as_ref() == "sensitivity.evalue"),
        "cell.aipw cheap must E-value the cell-versus-control contrast"
    );
    assert!(
        cell.diagnostics.iter().any(|d| d.code.as_ref() == "refute.cell_aipw.contrast"),
        "cell.aipw cheap must disclose the contrast, not the intervention level"
    );
    assert!(!cell.diagnostics.iter().any(|d| d.code.as_ref() == "refute.response.skipped"));

    let plugin = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(joint_query(1.0, 1.0)))
        .estimator(EstimatorId::ResponseInterventionGcomp)
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(plugin.estimate.ate.is_finite());
    assert!(
        plugin.refutations.iter().any(|r| r.refuter.as_ref().contains("overlap")),
        "plugin cheap must still run overlap"
    );
    assert!(
        plugin.diagnostics.iter().any(|d| d.code.as_ref() == "refute.evalue.not_a_contrast"),
        "plugin cheap must refuse E-value on an intervention level"
    );
    assert!(
        !plugin.refutations.iter().any(|r| r.refuter.as_ref() == "sensitivity.evalue"),
        "plugin cheap must not invent an E-value for a cell mean"
    );
    assert!(!plugin.diagnostics.iter().any(|d| d.code.as_ref() == "refute.response.skipped"));
}

#[test]
fn dag_intervention_response_full_runs_effect_refuters() {
    let (data, graph) = interaction_dgp(700, 207);
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(joint_query(1.0, 1.0)))
        .estimator(EstimatorId::CellAipw)
        .refute(RefuteSuite::Full)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(207))
        .unwrap();
    assert!(result.estimate.ate.is_finite());
    assert!(
        result.refutations.len() >= 2,
        "full suite must publish more than the cheap overlap/E-value pair"
    );
}

#[test]
fn plugin_full_omits_contrast_shaped_refuters() {
    let (data, graph) = interaction_dgp(700, 212);
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(joint_query(1.0, 1.0)))
        .estimator(EstimatorId::ResponseInterventionGcomp)
        .refute(RefuteSuite::Full)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(212))
        .unwrap();
    assert!(result.estimate.ate.is_finite());
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "refute.evalue.not_a_contrast"),
        "plugin full must disclose that contrast-shaped refuters are not licensed"
    );
    let names: Vec<_> = result.refutations.iter().map(|r| r.refuter.as_ref()).collect();
    for banned in [
        "placebo",
        "dummy.outcome",
        "dummy_outcome",
        "random.common_cause",
        "unobserved.common_cause",
        "sensitivity.evalue",
        "sensitivity.linear",
        "sensitivity.partial_linear",
        "sensitivity.nonparametric",
        "sensitivity.riesz",
    ] {
        assert!(
            !names.iter().any(|n| n.contains(banned)),
            "plugin full must not run contrast-shaped {banned}: {names:?}"
        );
    }
    assert!(
        names.iter().any(|n| n.contains("overlap")),
        "plugin full must still run overlap: {names:?}"
    );
}

#[test]
fn cell_aipw_refuses_named_interaction_on_exceedance_grid() {
    let (data, graph) = interaction_dgp(800, 208);
    let query = joint_query(1.0, 1.0)
        .with_outcome_functional(OutcomeFunctional::exceedance_grid([0.0, 0.5, 1.0]));
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .estimator(EstimatorId::CellAipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(208))
        .unwrap();
    assert!(
        result.estimate.ate.is_nan(),
        "cell.aipw grids must not publish a first-threshold scalar"
    );
    let table = result.estimate.score_table.as_ref().expect("cell grid scores");
    let err = antecedent_estimate::cell_aipw::contrast_named(table, "interaction").unwrap_err();
    assert!(
        err.to_string().contains("not licensed"),
        "grid interaction must refuse rather than silently use the first threshold: {err}"
    );
}

#[test]
fn candidate_selection_survives_when_joint_if_cannot_form() {
    let (data, graph, q1, _, _) = confounded_hetero(400, 209);
    let q2 = q1.clone().with_target_population(TargetPopulation::Treated);
    let ctx = ExecutionContext::for_tests(209);
    let n = u32::try_from(data.row_count()).unwrap();
    let results = BatchStudy::new(data, graph)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .candidate_screen(CandidateScreen {
            screen_id: Arc::from("v15.no-joint"),
            procedure: CandidateProcedure::BenjaminiHochberg,
            screen_rows: (0..n / 2).collect(),
            estimate_rows: (n / 2..n).collect(),
        })
        .estimate_many(&[q1, q2], &ctx)
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].candidate_selection.as_ref().unwrap().winner_index, None);
    assert!(results[0].candidate_selection.is_some());
    assert!(results[0].estimate.candidate_selection.is_some());
    assert!(results.iter().any(|r| {
        r.diagnostics.iter().any(|d| d.code.as_ref() == "batch.joint_if.unavailable")
    }));
    assert!(
        results
            .iter()
            .any(|r| r.diagnostics.iter().any(|d| d.code.as_ref() == "batch.candidate_selection"))
    );
    assert!(
        results.iter().all(|r| r.estimate.simultaneous_interval.is_none()),
        "batch family max-t must not be invented when joint IF cannot form"
    );
    assert!(
        !results.iter().any(|r| r.diagnostics.iter().any(|d| d.code.as_ref() == "batch.joint_if")),
        "batch family joint IF diagnostic must be absent when the family IF is unavailable"
    );
}

#[test]
fn allobserved_first_click_matches_retarget_ones() {
    let (data, graph, query, _, _) = confounded_hetero(800, 210);
    let ctx = ExecutionContext::for_tests(210);
    let first =
        ate_study(data.clone(), graph.clone(), query.clone(), EstimatorId::Aipw).run(&ctx).unwrap();
    let table = first.estimate.score_table.as_ref().expect("AllObserved iid AIPW must export φ");
    assert!(first.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.aipw.crossfit_scores"));
    assert!(
        !first
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.aipw.full_sample_residualized")
    );
    let ones = vec![1.0; table.n_rows];
    let retargeted = ate_study(data, graph, query, EstimatorId::Aipw)
        .prepare(&ctx)
        .unwrap()
        .retarget(&ones, &[], &ctx)
        .unwrap();
    assert!(
        (first.estimate.ate - retargeted.estimate.ate).abs() < 1e-12,
        "first-click={} retarget(ones)={}; AllObserved iid φ must be the same object",
        first.estimate.ate,
        retargeted.estimate.ate
    );
}

#[test]
fn att_residualized_has_no_scores_and_refuses_retarget() {
    let (data, graph, query, _, _) = confounded_hetero(600, 211);
    let query = query.with_target_population(TargetPopulation::Treated);
    let ctx = ExecutionContext::for_tests(211);
    let first =
        ate_study(data.clone(), graph.clone(), query.clone(), EstimatorId::Aipw).run(&ctx).unwrap();
    assert!(first.estimate.score_table.is_none());
    assert!(
        first
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.aipw.full_sample_residualized")
    );
    assert!(!first.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.aipw.crossfit_scores"));
    let prepared = ate_study(data, graph, query, EstimatorId::Aipw).prepare(&ctx).unwrap();
    assert!(prepared.score_table().is_none());
    let err = prepared.retarget(&[1.0; 8], &[], &ctx).unwrap_err();
    assert!(
        err.to_string().contains("score table"),
        "ATT residualized AIPW must not expose a retarget handle: {err}"
    );
}

#[test]
fn retarget_depends_on_without_dag_refuses_on_prepared_table() {
    let (data, graph, query, _, _) = confounded_hetero(400, 213);
    let ctx = ExecutionContext::for_tests(213);
    let prepared = ate_study(data, graph, query, EstimatorId::Aipw).prepare(&ctx).unwrap();
    let table = prepared.score_table().expect("AllObserved AIPW scores");
    let err =
        antecedent_estimate::check_depends_on(&[VariableId::from_raw(2)], table, None).unwrap_err();
    assert!(
        err.to_string().contains("descendant closure requires a directed graph"),
        "nonempty depends_on without a directed graph must refuse, not skip: {err}"
    );
}

#[test]
fn retarget_succeeds_on_codetermined_tiered_admg() {
    // {z, u} | {t} | {y}: z↔u in the closure ADMG. depends_on=[z]; u is z's
    // bidirected neighbor and is not a directed descendant of t. Identified
    // AIPW via tier-closure {z, u}. ExceedanceGrid is the licensed 1.5 call.
    let (data, background, query, weights, z, u) = codetermined_background_siblings(800, 214);
    let schema = data.schema().clone();
    let t = schema.id_of("t").unwrap();
    assert_bidirected_sibling_not_descendant(&background, &schema, t, u);
    assert!(
        background
            .to_admg(&schema)
            .unwrap()
            .bidirected_neighbors(DenseNodeId::from_raw(z.raw()))
            .contains(&DenseNodeId::from_raw(u.raw())),
        "z must be bidirected-adjacent to u"
    );
    let ctx = ExecutionContext::for_tests(214);
    let out = retarget_grid_matches_standalone(&data, background, query, &weights, &[z], &ctx);
    assert!(out.estimate.exceedance_cdf.is_some());

    // {z} | {t, u} | {y}: t↔u. Walking ↔ as descendants from t would mark u
    // and refuse depends_on=[u]. Directed-only closure must keep this legal.
    let (data, background, query, weights, u) = codetermined_treatment_sibling(800, 214);
    let schema = data.schema().clone();
    let t = schema.id_of("t").unwrap();
    assert_bidirected_sibling_not_descendant(&background, &schema, t, u);
    assert!(
        background
            .to_admg(&schema)
            .unwrap()
            .bidirected_neighbors(DenseNodeId::from_raw(t.raw()))
            .contains(&DenseNodeId::from_raw(u.raw())),
        "treatment-tier peer must be t↔u"
    );
    let out = retarget_grid_matches_standalone(&data, background, query, &weights, &[u], &ctx);
    assert!(out.estimate.exceedance_cdf.is_some());
}

#[test]
fn retarget_nonconstant_weights_require_depends_on() {
    let (data, graph, query, weights, _) = confounded_hetero(400, 215);
    let ctx = ExecutionContext::for_tests(215);
    let prepared = ate_study(data, graph, query, EstimatorId::Aipw).prepare(&ctx).unwrap();
    let err = prepared.retarget(&weights, &[], &ctx).unwrap_err();
    assert!(
        err.to_string().contains("nonempty depends_on is required for nonconstant"),
        "nonconstant w + empty depends_on must refuse: {err}"
    );
}

#[test]
fn batch_prepare_keeps_exceedance_functional() {
    let (data, graph, mean_q, exc_q) = variance_shift_binary(1_200, 216);
    let ctx = ExecutionContext::for_tests(216);
    let queries = [mean_q, exc_q];
    let results = BatchStudy::new(data.clone(), graph.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare(&queries, &ctx)
        .unwrap()
        .estimate(&data, &ctx)
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].estimate.exceedance_cdf.is_none(), "mean sibling must stay a mean");
    assert!(
        (results[1].estimate.ate - results[0].estimate.ate).abs() > 1e-6,
        "exceedance batch member must not silently become the mean sibling"
    );
    for (q, row) in queries.iter().zip(&results) {
        let solo = ate_study(data.clone(), graph.clone(), q.clone(), EstimatorId::Aipw)
            .prepare(&ctx)
            .unwrap()
            .estimate(&data, &ctx)
            .unwrap();
        assert_estimates_pin_eq(row, &solo, "prepared batch vs standalone");
    }
}

#[test]
fn batch_estimate_many_keeps_exceedance_functional() {
    let (data, graph, mean_q, exc_q) = variance_shift_binary(1_200, 218);
    let ctx = ExecutionContext::for_tests(218);
    let queries = [mean_q, exc_q];
    let results = BatchStudy::new(data.clone(), graph.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .estimate_many(&queries, &ctx)
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].estimate.exceedance_cdf.is_none(), "mean sibling must stay a mean");
    assert!(
        (results[1].estimate.ate - results[0].estimate.ate).abs() > 1e-6,
        "analyze_many / estimate_many must not silently become the mean sibling"
    );
    for (q, row) in queries.iter().zip(&results) {
        let solo = ate_study(data.clone(), graph.clone(), q.clone(), EstimatorId::Aipw)
            .prepare(&ctx)
            .unwrap()
            .estimate(&data, &ctx)
            .unwrap();
        assert_estimates_pin_eq(row, &solo, "estimate_many vs standalone");
    }
}

#[test]
fn batch_tiered_codetermined_prepare_and_estimate() {
    // Same-tier {z, u} so the closure ADMG has z↔u. Mean + ExceedanceGrid.
    let (data, background, mean_q, _, _, u) = codetermined_background_siblings(800, 217);
    let schema = data.schema().clone();
    let t = schema.id_of("t").unwrap();
    assert_bidirected_sibling_not_descendant(&background, &schema, t, u);
    let grid_q =
        mean_q.clone().with_outcome_functional(OutcomeFunctional::exceedance_grid([0.0, 0.5]));
    let ctx = ExecutionContext::for_tests(217);
    let queries = [mean_q, grid_q];
    let results = BatchStudy::tiered(data.clone(), background.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare(&queries, &ctx)
        .unwrap()
        .estimate(&data, &ctx)
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].estimate.ate.is_finite());
    assert!(results[0].estimate.exceedance_cdf.is_none(), "mean sibling must stay a mean");
    assert!(
        results[1].estimate.ate.is_nan(),
        "grid batch member must not publish a first-threshold ATE"
    );
    assert!(results[1].estimate.exceedance_cdf.as_ref().is_some_and(|c| c.len() == 4));
    for (q, row) in queries.iter().zip(&results) {
        let solo = tiered_aipw(data.clone(), background.clone(), q.clone())
            .prepare(&ctx)
            .unwrap()
            .estimate(&data, &ctx)
            .unwrap();
        assert_estimates_pin_eq(row, &solo, "tiered batch vs standalone");
    }
}

#[test]
fn plugin_zero_shift_influence_equals_sample_mean_influence() {
    let n = 400usize;
    let mut rng = ExecutionContext::for_tests(301).rng.stream(0x301);
    let t: Vec<_> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let y: Vec<_> = t.iter().map(|v| 1.0 + v + standard_normal(&mut rng)).collect();
    let mean = y.iter().sum::<f64>() / n as f64;
    let expected_se =
        (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n * (n - 1)) as f64).sqrt();
    let data = cols(&[("t", t), ("y", y)]);
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let q = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::shift(VariableId::from_raw(0), Value::f64(0.0))]),
    });
    let result = Study::tabular(data)
        .graph(graph)
        .query(q)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(301))
        .unwrap();
    assert!((response_value(&result) - mean).abs() < 1e-8);
    assert!((result.estimate.se_analytic - expected_se).abs() < 1e-7);
}

#[test]
fn plugin_set_intervals_include_model_fit_uncertainty() {
    let mut covered = 0;
    for seed in 300..340 {
        let n = 300;
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x302);
        let t: Vec<_> = (0..n).map(|_| standard_normal(&mut rng)).collect();
        let y: Vec<_> = (0..n).map(|_| 1.5 + standard_normal(&mut rng)).collect();
        let data = cols(&[("t", t), ("y", y)]);
        let estimator = antecedent_estimate::ContinuousResponseEstimator::new(Arc::from([]));
        let q = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(0.0))]),
        });
        let response = estimator
            .estimate_identified(
                &data,
                &q,
                antecedent_core::IdentificationStatus::NonparametricallyIdentified,
                antecedent_core::AssumptionSet::default(),
            )
            .unwrap();
        let antecedent_core::ResponseUncertainty::Scalar { standard_error, lower, upper, .. } =
            response.uncertainty
        else {
            panic!("missing response interval")
        };
        assert!(
            standard_error > 0.02,
            "constant counterfactual design must retain outcome-model uncertainty"
        );
        covered += usize::from(lower <= 1.5 && upper >= 1.5);
    }
    assert!(covered >= 33, "covered {covered}/40 population response means");
}

#[test]
fn conditional_extreme_threshold_refuses_empty_tail_band() {
    let (data, graph, _, _, _) = confounded_hetero(400, 210);
    let inner = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_effect_modifiers([VariableId::from_raw(2)])
        .with_outcome_functional(OutcomeFunctional::exceedance_grid([0.0, 1.0e6]));
    let query = ConditionalEffectQuery::try_new(inner).unwrap();
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::ConditionalEffect(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(210))
        .unwrap();
    let inf = result.estimate.score_inference.expect("tail support on the object");
    assert_eq!(inf.threshold_supported.len(), 4);
    assert!(!inf.threshold_supported[2] && !inf.threshold_supported[3]);
    assert!(inf.lower[2].is_nan() && inf.upper[2].is_nan());
    assert!(inf.lower[3].is_nan() && inf.upper[3].is_nan());
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.functional.threshold_tail.unsupported")
    );
}

#[test]
fn prepared_batch_shares_fold_object_and_covariate_design() {
    let (data, _graph, query, _, _) = confounded_hetero(500, 211);
    let y2: Vec<f64> = data
        .float64_values(VariableId::from_raw(1))
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, y)| y + 0.15 * ((i % 3) as f64))
        .collect();
    let data = cols(&[
        ("t", data.float64_values(VariableId::from_raw(0)).unwrap()),
        ("y", data.float64_values(VariableId::from_raw(1)).unwrap()),
        ("z", data.float64_values(VariableId::from_raw(2)).unwrap()),
        ("y2", y2),
    ]);
    let mut graph = Dag::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(3)).unwrap();
    let q2 = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(3));
    let ctx = ExecutionContext::for_tests(211);
    let prepared = BatchStudy::new(data.clone(), graph)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare(&[query, q2], &ctx)
        .unwrap();
    let shared: &SharedBatchDesign =
        prepared.shared_design().expect("prepare must freeze a shared design object");
    assert_eq!(shared.n_folds, 5);
    assert!(shared.covariate.is_some(), "same adjustment set must share [1 | Z]");
    let a = prepared.plans()[0].shared_design().expect("plan 0 shared design");
    let b = prepared.plans()[1].shared_design().expect("plan 1 shared design");
    assert!(
        std::ptr::eq(a, b),
        "each query must hold the same SharedBatchDesign object, not an independently rebuilt design"
    );
    let t0 = prepared.plans()[0].score_table().expect("scores");
    let t1 = prepared.plans()[1].score_table().expect("scores");
    assert_eq!(t0.fold_ids.as_ref(), t1.fold_ids.as_ref());
    assert_eq!(t0.fold_ids.as_ref(), shared.folds_for(&t0.row_index).unwrap());
    assert!(t0.nuisance_provenance.contains("batch.shared_design"));
    assert!(t1.nuisance_provenance.contains("batch.shared_design"));
    assert_ne!(
        t0.scores.as_ref(),
        t1.scores.as_ref(),
        "different outcomes must not share residualization"
    );
    let results = prepared.estimate(&data, &ctx).unwrap();
    assert!(
        results
            .iter()
            .any(|r| r.diagnostics.iter().any(|d| d.code.as_ref() == "batch.shared_design"))
    );
}

#[test]
fn pag_multi_atom_response_mixes_aligned_ifs() {
    let n = 1_200usize;
    let mut rng = ExecutionContext::for_tests(212).rng.stream(0xD4);
    let mut r = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        r[i] = standard_normal(&mut rng);
        z[i] = standard_normal(&mut rng);
        t[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-(0.8 * r[i] + 0.5 * z[i])).exp()));
        y[i] = 1.6 * t[i] + 0.7 * z[i] + 0.35 * standard_normal(&mut rng);
    }
    let data = cols(&[("t", t), ("y", y), ("z", z), ("r", r)]);
    // R→T witnesses visibility. Z o→ T is one Markov class with two MAGs (Z→T and Z↔T).
    let mut pag = Pag::with_variables(4);
    pag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
    pag.insert_circle_arrow(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let env = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
        .identify_pag_envelope(
            &pag,
            &AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)),
        )
        .unwrap();
    assert!(
        env.unidentified_weight.0 <= 1e-12,
        "all-identified fixture must not hide unidentified mass"
    );
    let identified: Vec<_> = env
        .cases
        .iter()
        .filter(|c| c.result.status != antecedent_identify::IdentificationStatus::NotIdentified)
        .collect();
    assert!(
        identified.len() >= 2,
        "fixture must have multiple identified MAG completions, got {} of {}",
        identified.len(),
        env.cases.len()
    );
    let zt = DenseNodeId::from_raw(2);
    let tt = DenseNodeId::from_raw(0);
    let marks: Vec<_> =
        identified.iter().map(|c| c.graph.edge_between(zt, tt).map(|e| (e.at_a, e.at_b))).collect();
    assert!(
        marks.windows(2).any(|w| w[0] != w[1]),
        "completions must be distinct MAGs (Z→T vs Z↔T), got {marks:?}"
    );
    let response = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    });
    let mixed = Study::tabular(data)
        .graph(pag)
        .query(CausalQuery::Response(response))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(212))
        .unwrap();
    assert!(mixed.estimate.ate.is_finite());
    assert!(
        mixed.estimate.se_analytic.is_finite() && mixed.estimate.se_analytic > 0.0,
        "all-identified multi-atom mix must publish mixed IF uncertainty, se={}",
        mixed.estimate.se_analytic
    );
    assert!(
        mixed.estimate.joint_covariance.is_some() || mixed.estimate.influence.is_some(),
        "mixed IF / covariance must be attached"
    );
    assert!(
        mixed.diagnostics.iter().any(|d| {
            d.code.as_ref() == "identify.pag.envelope" && d.message.contains("cases=2")
        }),
        "envelope diagnostic must report the multi-atom mix, not a one-completion MAG"
    );
}

#[test]
fn pag_unidentified_completion_does_not_publish_primary_atom_se() {
    let n = 800usize;
    let mut rng = ExecutionContext::for_tests(213).rng.stream(0xD5);
    let mut r = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        r[i] = standard_normal(&mut rng);
        z[i] = standard_normal(&mut rng);
        t[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-(0.7 * r[i] + 0.4 * z[i])).exp()));
        y[i] = 1.2 * t[i] + 0.6 * z[i] + 0.3 * standard_normal(&mut rng);
    }
    let data = cols(&[("t", t), ("y", y), ("z", z), ("r", r)]);
    // Shielded Z—T: two identified MAGs and one unidentified completion in one class.
    let mut pag = Pag::with_variables(4);
    pag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(2)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
    pag.insert_circle_circle(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let env = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
        .identify_pag_envelope(
            &pag,
            &AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)),
        )
        .unwrap();
    assert!(env.cases.len() >= 2);
    assert!(env.unidentified_weight.0 > 0.0, "this fixture must retain unidentified completions");
    assert!(env.identified_weight.0 > 0.0, "this fixture must keep at least one identified atom");
    let response = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    });
    let mixed = Study::tabular(data.clone())
        .graph(pag)
        .query(CausalQuery::Response(response.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(213))
        .unwrap();
    let mut primary_pag = Pag::with_variables(4);
    primary_pag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(2)).unwrap();
    primary_pag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
    primary_pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    primary_pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    primary_pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let primary = Study::tabular(data)
        .graph(primary_pag)
        .query(CausalQuery::Response(response))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(213))
        .unwrap();
    assert!(mixed.estimate.ate.is_finite(), "identified-mass response level is still published");
    assert!(
        !mixed.estimate.se_analytic.is_finite(),
        "unidentified completions must not publish an identified-atom SE as the envelope; mixed se={} primary se={}",
        mixed.estimate.se_analytic,
        primary.estimate.se_analytic
    );
    assert!(
        primary.estimate.se_analytic.is_finite(),
        "the identified MAG still has its own SE; the envelope must not reuse it"
    );
    assert!(
        mixed
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.envelope.se_omits_between_atom_variance"),
        "omitted envelope SE must be disclosed"
    );
}

#[test]
fn quantile_treatment_effect_inverts_aipw_cdf() {
    let n = 2_400usize;
    let mut rng = ExecutionContext::for_tests(215).rng.stream(0x51);
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        z[i] = standard_normal(&mut rng);
        t[i] = f64::from(rng.next_f64() < 0.5);
        y[i] = 2.0 * t[i] + 0.3 * z[i] + 0.4 * standard_normal(&mut rng);
    }
    let data = cols(&[("t", t), ("y", y), ("z", z)]);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_outcome_functional(OutcomeFunctional::quantile(0.5));
    let result = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(215))
        .unwrap();
    assert!(
        (result.estimate.ate - 2.0).abs() < 0.25,
        "median QTE should recover the location shift, ate={}",
        result.estimate.ate
    );
    assert!(result.estimate.se_analytic.is_finite() && result.estimate.se_analytic > 0.0);
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.functional.quantile"),
        "quantile inversion must be disclosed on the result"
    );
}

#[test]
fn pag_front_door_response_uses_general_id() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap();
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (i, name) in columns.iter().enumerate() {
            values[i].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(pairs).unwrap();
    let mut pag = Pag::with_variables(3);
    pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    pag.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    });
    let env = antecedent_identify::identify_pag_response_general(&pag, &query).unwrap();
    assert!(env.identified_weight.0 > 0.0);
    assert_eq!(env.cases[0].result.estimands[0].method.as_ref(), "general.id");
    let result = Study::tabular(data)
        .graph(pag)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(216))
        .unwrap();
    assert!(
        (result.estimate.ate - 0.3).abs() < 1e-9,
        "front-door MAG response must recover the ID functional, ate={}",
        result.estimate.ate
    );
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "identify.response.general_id"),
        "general-ID provenance must be visible"
    );
}

fn same_tier_joint_query(schema: &antecedent_core::CausalSchema) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: schema.id_of("y").unwrap(),
        interventions: Arc::from([
            Intervention::set(schema.id_of("t1").unwrap(), Value::f64(1.0)),
            Intervention::set(schema.id_of("t2").unwrap(), Value::f64(1.0)),
        ]),
    })
}

fn same_tier_joint_dgp(n: usize, seed: u64) -> (TabularData, TieredBackground, ResponseQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0xC0);
    let mut z = vec![0.0; n];
    let mut t1 = vec![0.0; n];
    let mut t2 = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let latent = standard_normal(&mut rng);
        z[i] = zi;
        t1[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-(-0.2 + 0.9 * zi + 0.7 * latent)).exp()));
        t2[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-(-0.1 + 0.8 * zi + 0.65 * latent)).exp()));
        y[i] = 1.2 * t1[i] + 0.8 * t2[i] + 1.5 * t1[i] * t2[i] + 0.55 * zi
            + 0.3 * standard_normal(&mut rng);
    }
    let data = cols(&[("z", z), ("t1", t1), ("t2", t2), ("y", y)]);
    let schema = data.schema().clone();
    let background = TieredBackground::from_named(
        &schema,
        &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
        WithinTier::CoDetermined,
    )
    .unwrap();
    let query = same_tier_joint_query(&schema);
    (data, background, query)
}

#[test]
fn codetermined_same_tier_joint_cell_aipw_matches_closure_admg() {
    let (data, background, query) = same_tier_joint_dgp(2_400, 17);
    let schema = data.schema().clone();
    let z = schema.id_of("z").unwrap();
    let t1_id = schema.id_of("t1").unwrap();
    let t2_id = schema.id_of("t2").unwrap();
    let y = schema.id_of("y").unwrap();
    let admg = background.to_admg(&schema).unwrap();
    let t1 = DenseNodeId::from_raw(t1_id.raw());
    let t2 = DenseNodeId::from_raw(t2_id.raw());
    assert!(admg.has_bidirected(), "same-tier pair must materialize ↔");
    assert!(admg.bidirected_neighbors(t1).contains(&t2));
    let mut descendants = BitSet::default();
    let mut ws = GraphWorkspace::default();
    admg.descendants_of(&[t1], &mut descendants, &mut ws);
    assert!(!descendants.contains(t2), "treating ↔ as a directed path would put t2 in De(t1)");

    let id = antecedent_identify::identify_tiered_joint(&background, &schema, &query).unwrap();
    assert_eq!(id.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(id.estimands[0].adjustment_set.as_ref(), &[z]);
    assert!(
        !id.estimands[0].adjustment_set.iter().any(|&v| v == t2_id),
        "t2 is a treatment; walking ↔ as directed must not put the peer in Z"
    );
    let admg_id = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
        .identify_joint_admg_response(&admg, &query)
        .unwrap();
    assert_eq!(admg_id.estimands[0].adjustment_set, id.estimands[0].adjustment_set);

    let ctx = ExecutionContext::for_tests(17);
    let result = Study::tabular(data.clone())
        .tiered_background(background.clone())
        .unwrap()
        .query(CausalQuery::Response(query.clone()))
        .estimator(EstimatorId::CellAipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(result.estimate.ate.is_finite(), "licensed cell must be finite");
    assert!(result.estimate.se_analytic.is_finite(), "cell-AIPW SE requires IFs");
    let table = result.estimate.score_table.as_ref().expect("cell.aipw must freeze scores");
    let ix = antecedent_estimate::interaction_contrast(table).unwrap();
    assert!(ix.value.is_finite() && ix.se.is_finite(), "interaction={} se={}", ix.value, ix.se);

    let direct = antecedent_estimate::CellSaturatedAipw::new()
        .fit_scores(
            &data,
            &[t1_id, t2_id],
            y,
            id.estimands[0].adjustment_set.as_ref(),
            &OutcomeFunctional::Mean,
            None,
        )
        .unwrap();
    let summary = direct.summarize(None).unwrap();
    let arm11 = direct.columns.iter().position(|c| c.arm == 3).expect("cell 11");
    assert!(
        (result.estimate.ate - summary.means[arm11]).abs() < PIN_ABS,
        "tiered cell.aipw {} must match closure-ADMG cell-AIPW {}",
        result.estimate.ate,
        summary.means[arm11]
    );
    let empty = antecedent_estimate::CellSaturatedAipw::new()
        .fit_scores(&data, &[t1_id, t2_id], y, &[], &OutcomeFunctional::Mean, None)
        .unwrap()
        .summarize(None)
        .unwrap();
    assert!(
        (empty.means[arm11] - summary.means[arm11]).abs() > 1e-4,
        "skipping identification (Z={{}}) would not recover the licensed cell"
    );

    let prepared = BatchStudy::tiered(data.clone(), background)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare_cells(&[query], &ctx)
        .unwrap();
    assert!(
        prepared.plans()[0].score_table().is_some(),
        "prepare_cells must freeze cell-AIPW scores"
    );
    let batch = &prepared.estimate(&data, &ctx).unwrap()[0];
    assert!(batch.estimate.ate.is_finite(), "prepare_cells estimate must be a real cell");
    let batch_table = batch.estimate.score_table.as_ref().expect("batch cell.aipw scores");
    let batch_ix = antecedent_estimate::interaction_contrast(batch_table).unwrap();
    assert!(batch_ix.value.is_finite());
}

#[test]
fn unknown_tier_joint_has_no_single_admg() {
    let (data, _, query) = same_tier_joint_dgp(40, 18);
    let schema = data.schema().clone();
    let background = TieredBackground::from_named(
        &schema,
        &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
        WithinTier::Unknown,
    )
    .unwrap();
    let err = antecedent_identify::identify_tiered_joint(&background, &schema, &query).unwrap_err();
    assert!(err.to_string().contains("no single ADMG"), "{err}");
    let ctx = ExecutionContext::for_tests(18);
    let batch_err = BatchStudy::tiered(data, background)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare_cells(&[query], &ctx)
        .unwrap_err();
    assert!(batch_err.to_string().contains("no single ADMG"), "{batch_err}");
}
