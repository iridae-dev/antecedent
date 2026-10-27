//! The batch (`analyze_many`) routes: shared cross-fitted AIPW nuisances, and
//! the typed boundary of what a batch covers.
//!
//! A batch shares cross-fitted AIPW nuisance fits between queries that would
//! fit the identical nuisance (same rows, treatment coding, adjustment design,
//! fold plan and learner options), and every shared result is bit-identical to
//! the per-query fit. A batch covers average effects (`estimate_many`,
//! `prepare`) and discrete joint `InterventionResponse` cells (`prepare_cells`);
//! any other query is refused with `route_not_supported`, naming `analyze`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    reason = "sharing a nuisance fit must not move a single bit of the estimate"
)]

use std::sync::Arc;

use antecedent::{BatchStudy, EstimatorId, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, DerivativeWeighting, ExecutionContext, Intervention, ResponseFunctional,
    ResponseQuery, StreamDomain, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

const T1: u32 = 0;
const T2: u32 = 1;
const Y1: u32 = 2;
const Y2: u32 = 3;
const Z: u32 = 4;

/// Two binary treatments and two outcomes, all confounded by one `z`.
fn two_by_two(n: usize, seed: u64) -> (TabularData, Dag) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0xB5);
    let (mut t1, mut t2, mut y1, mut y2, mut z) =
        (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        let p1 = 1.0 / (1.0 + (-(-0.2 + 0.8 * zi)).exp());
        let p2 = 1.0 / (1.0 + (-(0.1 - 0.6 * zi)).exp());
        t1[i] = f64::from(rng.next_f64() < p1);
        t2[i] = f64::from(rng.next_f64() < p2);
        y1[i] = 2.0 * t1[i] + 0.5 * t2[i] + zi + 0.3 * standard_normal(&mut rng);
        y2[i] = -t1[i] + 1.5 * t2[i] - 0.5 * zi + 0.3 * standard_normal(&mut rng);
    }
    let pairs = [("t1", t1), ("t2", t2), ("y1", y1), ("y2", y2), ("z", z)];
    let borrowed: Vec<(&str, &[f64])> = pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(5);
    for (from, to) in [(Z, T1), (Z, T2), (Z, Y1), (Z, Y2), (T1, Y1), (T1, Y2), (T2, Y1), (T2, Y2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph)
}

fn ate(treatment: u32, outcome: u32) -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(treatment), VariableId::from_raw(outcome))
}

fn solo(data: &TabularData, graph: &Dag, query: &AverageEffectQuery, seed: u64) -> StudyResult {
    Study::tabular(data.clone())
        .graph(graph.clone())
        .query(query.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap()
}

fn shared_design_field<'a>(result: &'a StudyResult, key: &str) -> &'a str {
    let diagnostic = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "batch.shared_design")
        .expect("every batch member records the shared-design diagnostic");
    diagnostic
        .fields
        .iter()
        .find(|(k, _)| k.as_ref() == key)
        .map_or_else(|| panic!("batch.shared_design has no {key} field"), |(_, v)| v.as_ref())
}

fn assert_bit_identical(batch: &StudyResult, solo: &StudyResult, what: &str) {
    assert_eq!(
        batch.estimate.as_effect().unwrap().ate.to_bits(),
        solo.estimate.as_effect().unwrap().ate.to_bits(),
        "{what}: ate"
    );
    assert_eq!(
        batch.estimate.as_effect().unwrap().se_analytic().to_bits(),
        solo.estimate.as_effect().unwrap().se_analytic().to_bits(),
        "{what}: se_analytic"
    );
    let (Some(a), Some(b)) = (
        &batch.estimate.as_effect().unwrap().scores.influence,
        &solo.estimate.as_effect().unwrap().scores.influence,
    ) else {
        panic!("{what}: both runs publish the influence function");
    };
    assert_eq!(a.len(), b.len(), "{what}: influence length");
    assert!(a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()), "{what}: influence");
}

#[test]
fn batch_shares_matching_nuisances_and_matches_per_query_fits_bit_for_bit() {
    let (data, graph) = two_by_two(600, 41);
    // (t1, y1) and (t1, y2) share the t1 propensity; the repeated (t1, y1) also shares
    // its outcome regressions; (t2, y1) shares nothing with any other member.
    let queries = [ate(T1, Y1), ate(T1, Y2), ate(T2, Y1), ate(T1, Y1)];
    for threads in [1_u32, 4] {
        let mut ctx = ExecutionContext::for_tests(41);
        ctx.parallelism.max_threads = antecedent_core::NonZeroThreadCount::new(threads).unwrap();
        let results = BatchStudy::new(data.clone(), graph.clone())
            .estimator(EstimatorId::Aipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .estimate_many(&queries, &ctx)
            .unwrap();
        assert_eq!(results.len(), queries.len());
        for (i, (result, query)) in results.iter().zip(&queries).enumerate() {
            let alone = solo(&data, &graph, query, 41);
            assert_bit_identical(result, &alone, &format!("threads={threads} query {i}"));
        }
        let shares = |i: usize| {
            (
                shared_design_field(&results[i], "shares_propensity"),
                shared_design_field(&results[i], "shares_outcome_residualization"),
            )
        };
        assert_eq!(shares(0), ("true", "true"), "threads={threads}");
        assert_eq!(shares(1), ("true", "false"), "threads={threads}");
        assert_eq!(shares(2), ("false", "false"), "threads={threads}");
        assert_eq!(shares(3), ("true", "true"), "threads={threads}");
    }
}

#[test]
fn batch_never_shares_nuisances_across_different_seeds() {
    let (data, graph) = two_by_two(400, 43);
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let batch = |seed: u64| {
        BatchStudy::new(data.clone(), graph.clone())
            .estimator(EstimatorId::Aipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .estimate_many(&queries, &ExecutionContext::for_tests(seed))
            .unwrap()
    };
    // Each batch is its own sharing scope: a second batch under another seed (another
    // fold plan) refits, so its members match their own per-query fits, not the first's.
    let first = batch(43);
    let second = batch(44);
    for (i, query) in queries.iter().enumerate() {
        assert_bit_identical(&first[i], &solo(&data, &graph, query, 43), "seed 43");
        assert_bit_identical(&second[i], &solo(&data, &graph, query, 44), "seed 44");
    }
    assert_ne!(
        first[0].estimate.as_effect().unwrap().ate.to_bits(),
        second[0].estimate.as_effect().unwrap().ate.to_bits()
    );
}

#[test]
fn prepared_batch_estimate_shares_nuisances_and_matches_per_query_fits() {
    let (data, graph) = two_by_two(500, 47);
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let ctx = ExecutionContext::for_tests(47);
    let prepared = BatchStudy::new(data.clone(), graph.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare(&queries, &ctx)
        .unwrap();
    let results = prepared.estimate(&data, &ctx).unwrap();
    for (i, query) in queries.iter().enumerate() {
        assert_bit_identical(&results[i], &solo(&data, &graph, query, 47), "prepared");
        assert_eq!(shared_design_field(&results[i], "shares_propensity"), "true");
        assert_eq!(shared_design_field(&results[i], "shares_outcome_residualization"), "false");
    }
}

fn cells(interventions: Vec<Intervention>) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(Y1),
        interventions: Arc::from(interventions),
    })
}

fn set(variable: u32, value: f64) -> Intervention {
    Intervention::set(VariableId::from_raw(variable), Value::f64(value))
}

#[test]
fn prepare_cells_refuses_queries_outside_discrete_joint_cells_naming_analyze() {
    let (data, graph) = two_by_two(300, 53);
    let ctx = ExecutionContext::for_tests(53);
    let joint = cells(vec![set(T1, 1.0), set(T2, 1.0)]);
    let outside = [
        ("single intervention", cells(vec![set(T1, 1.0)])),
        (
            "average derivative",
            ResponseQuery::new(ResponseFunctional::AverageDerivative {
                outcome: VariableId::from_raw(Y1),
                treatment: VariableId::from_raw(T1),
                weighting: DerivativeWeighting::Observed,
            }),
        ),
    ];
    for (what, query) in outside {
        let err = BatchStudy::new(data.clone(), graph.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .prepare_cells(&[joint.clone(), query], &ctx)
            .expect_err(what);
        assert_eq!(err.reason_code(), Some("route_not_supported"), "{what}: {err}");
        assert!(err.to_string().contains("analyze"), "{what} names analyze: {err}");
    }
    // The covered family still prepares.
    BatchStudy::new(data, graph)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare_cells(&[joint, cells(vec![set(T1, 1.0), set(T2, 0.0)])], &ctx)
        .expect("discrete joint cells batch");
}

/// Every `(treatment, confounder)` cell occurs 40 times. `y` and `y_other` are exact linear
/// functions of treatment and confounder, so the linear nuisances recover them exactly.
fn dml_table() -> (TabularData, Dag) {
    let (mut t, mut z, mut y, mut y_other) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..40 {
        for treatment in [0.0, 1.0] {
            for confounder in [-1.0, 1.0] {
                t.push(treatment);
                z.push(confounder);
                y.push(1.0 + confounder + treatment * (2.0 + 0.5 * confounder));
                y_other.push(0.5 - 2.0 * confounder + treatment * 1.25);
            }
        }
    }
    let pairs = [("t", t), ("y", y), ("z", z), ("y_other", y_other)];
    let borrowed: Vec<(&str, &[f64])> = pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(2, 0), (2, 1), (0, 1), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph)
}

fn dml() -> antecedent_estimate::DmlAte {
    antecedent_estimate::DmlAte::new()
        .with_outcome(
            antecedent_learn::LearnerSpec::Linear(antecedent_learn::LinearSpec::default()),
        )
        .with_folds(4)
}

#[test]
fn a_dml_batch_fits_each_learner_nuisance_once_and_matches_independent_runs() {
    use antecedent_learn::fit_counts::count_resolved_fits;
    let (data, graph) = dml_table();
    let ctx = ExecutionContext::for_tests(97);
    let run_batch = |queries: &[AverageEffectQuery]| {
        count_resolved_fits(|| {
            BatchStudy::new(data.clone(), graph.clone())
                .estimator_spec(dml())
                .refute(RefuteSuite::None)
                .estimate_many(queries, &ctx)
                .unwrap()
        })
    };
    let alone = |query: &AverageEffectQuery| {
        count_resolved_fits(|| {
            Study::tabular(data.clone())
                .graph(graph.clone())
                .query(query.clone())
                .estimator(dml())
                .refute(RefuteSuite::None)
                .build()
                .unwrap()
                .run(&ctx)
                .unwrap()
        })
    };
    let same_estimate = |a: &StudyResult, b: &StudyResult, what: &str| {
        let (a, b) = (a.estimate.as_effect().unwrap(), b.estimate.as_effect().unwrap());
        assert_eq!(a.ate.to_bits(), b.ate.to_bits(), "{what}: ate");
        assert_eq!(a.se_analytic().to_bits(), b.se_analytic().to_bits(), "{what}: se");
    };
    let (outcome_one, outcome_two) = (ate(0, 1), ate(0, 3));
    let (single, single_fits) = alone(&outcome_one);
    assert_eq!(single_fits, 12, "three nuisances in four folds");
    // Two identical queries share one bundle: the same number of fits as one query.
    let (twice, twice_fits) = run_batch(&[outcome_one.clone(), outcome_one.clone()]);
    assert_eq!(twice_fits, single_fits, "the second query reads the first's nuisances");
    for result in &twice {
        same_estimate(result, &single, "shared DML bundle");
    }
    // Another outcome column is another key, so it is fit again and still equals its own run.
    let (other, other_fits) = alone(&outcome_two);
    let (mixed, mixed_fits) = run_batch(&[outcome_one, outcome_two]);
    assert_eq!(mixed_fits, single_fits + other_fits, "a changed outcome never borrows a bundle");
    same_estimate(&mixed[1], &other, "separate DML bundle");
}

#[test]
fn a_pool_that_refuses_retention_leaves_every_batch_result_equal_to_its_solo_run() {
    use antecedent_core::execution::AdmissionPool;
    use antecedent_state::retention::RetentionMode;
    let (data, graph) = two_by_two(600, 41);
    let queries = [ate(T1, Y1), ate(T1, Y2), ate(T1, Y1)];
    // The AIPW batch asks the pool for worker slots and for the transient simultaneous-band
    // simulation scratch (about 33 KiB, released before the next query); planners compare
    // against the pool's capacity, not its free bytes. Leaving a small headroom funds that
    // scratch but admits no retained fitted-state entry.
    let capacity = 1_u64 << 30;
    let headroom = 33 * 1024_u64;
    let pool = AdmissionPool::fixed(1, capacity);
    let held = ExecutionContext::for_tests(41)
        .with_admission(pool.clone())
        .acquire_admission(0, capacity - headroom)
        .unwrap();
    let ctx = ExecutionContext::for_tests(41).with_admission(pool.clone());
    let check = |results: &[StudyResult], what: &str| {
        for (i, (result, query)) in results.iter().zip(&queries).enumerate() {
            assert_bit_identical(result, &solo(&data, &graph, query, 41), &format!("{what} {i}"));
            assert_eq!(shared_design_field(result, "shares_propensity"), "false", "{what} {i}");
            assert_eq!(
                shared_design_field(result, "shares_outcome_residualization"),
                "false",
                "{what} {i}"
            );
        }
    };
    let batch = |study: BatchStudy, ctx: &ExecutionContext| {
        study
            .estimator(EstimatorId::Aipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .estimate_many(&queries, ctx)
            .unwrap()
    };
    let refused = batch(BatchStudy::new(data.clone(), graph.clone()), &ctx);
    check(&refused, "pool refuses retention");
    assert_eq!(pool.available().bytes(), headroom, "nothing was retained past the lease");
    // A reduced retention mode builds an empty cache even with a roomy pool.
    let reduced = batch(
        BatchStudy::new(data.clone(), graph.clone()).retention(RetentionMode::SummaryOnly),
        &ExecutionContext::for_tests(41),
    );
    check(&reduced, "summary-only retention");
    drop(held);
    assert_eq!(pool.available().bytes(), capacity);
    // With room, the same batch shares (the contrast that makes the refusals meaningful).
    let shared =
        batch(BatchStudy::new(data.clone(), graph.clone()), &ExecutionContext::for_tests(41));
    assert_eq!(shared_design_field(&shared[0], "shares_propensity"), "true");
}

fn assert_no_batch_statistical_work(report: &antecedent_core::execution_attempt::AttemptReport) {
    use antecedent_core::execution_attempt::Operation;
    for operation in [
        Operation::LeastSquaresSolve,
        Operation::GlmFit,
        Operation::AdjustedFit,
        Operation::LearnerFit,
        Operation::PosteriorFit,
        Operation::PosteriorDraw,
        Operation::ScoreConstruction,
        Operation::FactorConstruction,
        Operation::FactorEvaluation,
        Operation::Integration,
        Operation::Decision,
    ] {
        assert_eq!(report.counts(operation).attempted, 0, "unexpected {operation:?}");
    }
}

fn deferred_batch(
    data: &TabularData,
    graph: &Dag,
    queries: &[AverageEffectQuery],
    ctx: &ExecutionContext,
) -> antecedent::PreparedBatch {
    BatchStudy::new(data.clone(), graph.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare_deferred(queries, ctx)
        .unwrap()
}

#[test]
fn deferred_batch_preparation_and_branches_perform_no_statistical_work() {
    use antecedent_core::execution_attempt::observe_with_context;
    let (data, graph) = two_by_two(500, 71);
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let ctx = ExecutionContext::for_tests(71);
    let prepared = observe_with_context(&ctx, || {
        BatchStudy::new(data.clone(), graph.clone())
            .estimator(EstimatorId::Aipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .prepare_deferred(&queries, &ctx)
    });
    assert_no_batch_statistical_work(&prepared.report);
    let prepared = prepared.result.unwrap();
    assert_eq!(prepared.materialized_plan_count(), 0);
    assert_eq!(prepared.retained_score_table_count(), 0);
    let (other, _) = two_by_two(420, 72);
    let rebound = observe_with_context(&ctx, || prepared.rebind_data(other, &ctx));
    assert_no_batch_statistical_work(&rebound.report);
    let rebound = rebound.result.unwrap();
    assert_eq!(rebound.materialized_plan_count(), 0);
    assert_eq!(rebound.retained_score_table_count(), 0);
    let requoted = observe_with_context(&ctx, || prepared.reconfigure_execution(&ctx));
    assert_no_batch_statistical_work(&requoted.report);
    assert_eq!(requoted.result.unwrap().materialized_plan_count(), 0);
    assert_eq!(prepared.materialized_plan_count(), 0);
}

#[test]
fn completed_batch_shares_actual_fits_and_retains_final_family_evidence() {
    use antecedent_core::execution_attempt::{Operation, observe_with_context};
    let (data, graph) = two_by_two(500, 73);
    let ctx = ExecutionContext::for_tests(73);
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let prepared = deferred_batch(&data, &graph, &queries, &ctx);
    let executed = observe_with_context(&ctx, || prepared.execute_retained(&ctx));
    assert!(executed.report.counts(Operation::GlmFit).completed > 0);
    let completed = executed.result.unwrap();
    assert_eq!(prepared.materialized_plan_count(), 0);
    assert_eq!(prepared.retained_score_table_count(), 0);
    assert_eq!(completed.prepared().materialized_plan_count(), 2);
    assert_eq!(completed.prepared().retained_score_table_count(), 2);
    assert_eq!(completed.scores().source(), antecedent::ScoreSource::Estimated);
    for (index, query) in queries.iter().enumerate() {
        let result = &completed.results()[index];
        assert_bit_identical(result, &solo(&data, &graph, query, 73), "completed batch");
        assert_eq!(shared_design_field(result, "shares_propensity"), "true");
        assert!(result.estimate.as_effect().unwrap().joint.covariance.is_some());
        let (_, body) = completed.prepared().plans()[index]
            .execution_record_with_quantity_labels(result, &ctx, None)
            .unwrap();
        assert!(body.diagnostics.iter().any(|d| d.code == "batch.shared_design"));
    }
    let query = ate(T1, Y1);
    let single = deferred_batch(&data, &graph, std::slice::from_ref(&query), &ctx);
    let repeated = deferred_batch(&data, &graph, &[query.clone(), query], &ctx);
    let one = observe_with_context(&ctx, || single.execute_retained(&ctx));
    let two = observe_with_context(&ctx, || repeated.execute_retained(&ctx));
    one.result.unwrap();
    two.result.unwrap();
    for operation in [Operation::GlmFit, Operation::LeastSquaresSolve, Operation::AdjustedFit] {
        assert_eq!(one.report.counts(operation), two.report.counts(operation), "{operation:?}");
    }
}

#[test]
fn completed_batch_data_branches_preserve_earlier_results_scores_and_contracts() {
    use antecedent_core::execution_attempt::observe_with_context;
    let (data, graph) = two_by_two(500, 79);
    let (other, _) = two_by_two(420, 80);
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let ctx = ExecutionContext::for_tests(79);
    let completed = deferred_batch(&data, &graph, &queries, &ctx).execute_retained(&ctx).unwrap();
    let original_point = completed.results()[0].estimate.as_effect().unwrap().ate.to_bits();
    let original_rows = completed.scores().common_rows().unwrap().unwrap();
    let original_contract = completed.prepared().plans()[0]
        .execution_record_with_quantity_labels(&completed.results()[0], &ctx, None)
        .unwrap()
        .0;
    let branched =
        observe_with_context(&ctx, || completed.prepared().rebind_data(other.clone(), &ctx));
    assert_no_batch_statistical_work(&branched.report);
    let branched = branched.result.unwrap();
    assert_eq!(branched.materialized_plan_count(), 0);
    assert_eq!(branched.retained_score_table_count(), 0);
    let updated = branched.execute_retained(&ctx).unwrap();
    assert_eq!(updated.scores().common_rows().unwrap().unwrap().len(), 420);
    for (index, query) in queries.iter().enumerate() {
        assert_bit_identical(&updated.results()[index], &solo(&other, &graph, query, 79), "branch");
    }
    assert_eq!(completed.results()[0].estimate.as_effect().unwrap().ate.to_bits(), original_point);
    assert_eq!(completed.scores().common_rows().unwrap().unwrap(), original_rows);
    assert_eq!(original_rows.len(), 500);
    let after_contract = completed.prepared().plans()[0]
        .execution_record_with_quantity_labels(&completed.results()[0], &ctx, None)
        .unwrap()
        .0;
    assert_eq!(original_contract, after_contract);
}

#[test]
fn retained_batch_executes_only_the_declared_screen_estimate_rows() {
    use antecedent::{CandidateProcedure, CandidateScreen};
    use antecedent_data::GlobalId;
    let (data, graph) = two_by_two(600, 83);
    let ctx = ExecutionContext::for_tests(83);
    let estimate_rows: Vec<_> = (200..600).map(GlobalId::new).collect();
    let screen = CandidateScreen {
        screen_id: Arc::from("held-out-screen"),
        procedure: CandidateProcedure::MaxT,
        screen_rows: (0..200).map(GlobalId::new).collect::<Vec<_>>().into(),
        estimate_rows: estimate_rows.clone().into(),
    };
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let prepared = BatchStudy::new(data.clone(), graph.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .candidate_screen(screen)
        .prepare_deferred(&queries, &ctx)
        .unwrap();
    let completed = prepared.execute_retained(&ctx).unwrap();
    assert_eq!(
        completed.scores().common_rows().unwrap().unwrap().as_ref(),
        estimate_rows.as_slice()
    );
    let repeated = completed.prepared().execute_retained(&ctx).unwrap();
    assert_eq!(
        repeated.scores().common_rows().unwrap().unwrap().as_ref(),
        estimate_rows.as_slice()
    );
    for (before, after) in completed.results().iter().zip(repeated.results()) {
        assert_bit_identical(before, after, "same screened observations");
    }
}

#[test]
fn new_execution_seed_invalidates_batch_scores_and_matches_its_own_solo_fits() {
    use antecedent_core::execution_attempt::observe_with_context;
    let (data, graph) = two_by_two(500, 89);
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let ctx = ExecutionContext::for_tests(89);
    let completed = deferred_batch(&data, &graph, &queries, &ctx).execute_retained(&ctx).unwrap();
    let mut changed = ctx.clone();
    changed.rng = ExecutionContext::for_tests(90).rng;
    let branch =
        observe_with_context(&changed, || completed.prepared().reconfigure_execution(&changed));
    assert_no_batch_statistical_work(&branch.report);
    let branch = branch.result.unwrap();
    assert_eq!(branch.materialized_plan_count(), 0);
    assert_eq!(branch.retained_score_table_count(), 0);
    assert_eq!(branch.shared_design().unwrap().fold_seed, 90);
    assert_ne!(
        branch.shared_design().unwrap().fold_ids,
        completed.prepared().shared_design().unwrap().fold_ids
    );
    let updated = branch.execute_retained(&changed).unwrap();
    for (index, query) in queries.iter().enumerate() {
        assert_bit_identical(
            &updated.results()[index],
            &solo(&data, &graph, query, 90),
            "new seed",
        );
    }
    assert_eq!(completed.prepared().shared_design().unwrap().fold_seed, 89);
    assert_eq!(completed.prepared().retained_score_table_count(), 2);
}

#[test]
fn retained_batch_state_does_not_cross_admission_ownership_or_cache_policy() {
    use antecedent_core::execution::{AdmissionPool, CachePolicy};
    use antecedent_core::execution_attempt::observe_with_context;
    let (data, graph) = two_by_two(500, 97);
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let ctx = ExecutionContext::for_tests(97).with_admission(AdmissionPool::fixed(1, 1 << 28));
    let completed = deferred_batch(&data, &graph, &queries, &ctx).execute_retained(&ctx).unwrap();
    let same = observe_with_context(&ctx, || completed.prepared().reconfigure_execution(&ctx));
    assert_no_batch_statistical_work(&same.report);
    let same = same.result.unwrap();
    assert_eq!(same.materialized_plan_count(), 2);
    assert!(
        Arc::ptr_eq(
            &same.plans()[0].score_table().unwrap().scores,
            &completed.prepared().plans()[0].score_table().unwrap().scores,
        ),
        "compatible execution branches must share the retained score buffer"
    );
    let separate = ctx.clone().with_admission(AdmissionPool::fixed(1, 1 << 28));
    let other =
        observe_with_context(&separate, || completed.prepared().reconfigure_execution(&separate));
    assert_no_batch_statistical_work(&other.report);
    let other = other.result.unwrap();
    assert_eq!(other.materialized_plan_count(), 0);
    assert_eq!(other.retained_score_table_count(), 0);
    let mut cache_change = ctx.clone();
    cache_change.cache_policy = CachePolicy::enabled(Some(0));
    let changed = observe_with_context(&cache_change, || {
        completed.prepared().reconfigure_execution(&cache_change)
    });
    assert_no_batch_statistical_work(&changed.report);
    assert_eq!(changed.result.unwrap().materialized_plan_count(), 0);
    let bounded = ctx.clone().with_memory_limit(Some(1));
    let refused =
        observe_with_context(&bounded, || completed.prepared().reconfigure_execution(&bounded));
    assert!(matches!(refused.result, Err(antecedent::CausalError::Resource { .. })));
    assert_no_batch_statistical_work(&refused.report);
    let execution_refused =
        observe_with_context(&bounded, || completed.prepared().execute_retained(&bounded));
    assert!(matches!(execution_refused.result, Err(antecedent::CausalError::Resource { .. })));
    assert_no_batch_statistical_work(&execution_refused.report);
    assert_eq!(completed.prepared().materialized_plan_count(), 2);
    assert_eq!(completed.prepared().retained_score_table_count(), 2);
}

#[test]
fn cancelled_batch_execution_never_publishes_into_a_prepared_or_completed_branch() {
    use antecedent_core::execution::CancellationToken;
    use antecedent_core::execution_attempt::{Operation, observe_with_context};
    let (data, graph) = two_by_two(400, 101);
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let ctx = ExecutionContext::for_tests(101);
    let prepared = deferred_batch(&data, &graph, &queries, &ctx);
    let completed = prepared.execute_retained(&ctx).unwrap();
    let original_point = completed.results()[0].estimate.as_effect().unwrap().ate.to_bits();
    let mut cancelled = ctx.clone();
    cancelled.cancellation = CancellationToken::new();
    cancelled.cancellation.cancel();
    let stopped = observe_with_context(&cancelled, || prepared.execute_retained(&cancelled));
    assert!(stopped.result.is_err());
    assert_no_batch_statistical_work(&stopped.report);
    struct CancelOnPoint(CancellationToken);
    impl antecedent::analysis::StageResultSink for CancelOnPoint {
        fn on_stage(&self, event: &antecedent::analysis::StageEvent) {
            if matches!(event, antecedent::analysis::StageEvent::Point { .. }) {
                self.0.cancel();
            }
        }
    }
    let mut stopped = ctx.clone();
    stopped.cancellation = CancellationToken::new();
    let sink: Arc<dyn antecedent::analysis::StageResultSink> =
        Arc::new(CancelOnPoint(stopped.cancellation.clone()));
    let observed = prepared.with_stage_sinks(vec![sink; queries.len()]).unwrap();
    let attempt = observe_with_context(&stopped, || observed.execute_retained(&stopped));
    assert!(attempt.result.is_err());
    assert!(stopped.cancellation.is_requested());
    assert!(attempt.report.counts(Operation::GlmFit).completed > 0);
    assert!(attempt.report.is_complete());
    assert_eq!(prepared.materialized_plan_count(), 0);
    assert_eq!(prepared.retained_score_table_count(), 0);
    assert_eq!(completed.results()[0].estimate.as_effect().unwrap().ate.to_bits(), original_point);
    assert_eq!(completed.prepared().retained_score_table_count(), 2);
    let recovered = prepared.execute_retained(&ctx).unwrap();
    for (before, after) in completed.results().iter().zip(recovered.results()) {
        assert_bit_identical(before, after, "recovery after cancelled execution");
    }
}

#[test]
fn batch_stage_observers_are_member_scoped_and_never_retained_by_completions() {
    use antecedent::analysis::{StageEvent, StageResultSink};
    use antecedent_core::execution_attempt::observe_with_context;
    use std::sync::Mutex;
    struct Recorder(Arc<Mutex<Vec<(usize, &'static str)>>>, usize);
    impl StageResultSink for Recorder {
        fn on_stage(&self, event: &StageEvent) {
            self.0.lock().unwrap().push((self.1, event.stage_id()));
        }
    }
    let (data, graph) = two_by_two(400, 103);
    let ctx = ExecutionContext::for_tests(103);
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let prepared = deferred_batch(&data, &graph, &queries, &ctx);
    let events = Arc::new(Mutex::new(Vec::new()));
    let first = Arc::new(Recorder(Arc::clone(&events), 0));
    let second = Arc::new(Recorder(Arc::clone(&events), 1));
    let weak_first = Arc::downgrade(&first);
    let weak_second = Arc::downgrade(&second);
    let sinks: Vec<Arc<dyn StageResultSink>> = vec![first, second];
    let branch = observe_with_context(&ctx, || prepared.with_stage_sinks(sinks));
    assert_no_batch_statistical_work(&branch.report);
    let branch = branch.result.unwrap();
    let completed = branch.execute_retained(&ctx).unwrap();
    for index in 0..2 {
        let observed: Vec<_> = events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(member, stage)| (*member == index).then_some(*stage))
            .collect();
        assert!(observed.contains(&"estimate_point"));
        assert!(observed.contains(&"uncertainty"));
        assert!(observed.contains(&"validate"));
    }
    drop(branch);
    assert!(weak_first.upgrade().is_none(), "completed study retained member 0's callback");
    assert!(weak_second.upgrade().is_none(), "completed study retained member 1's callback");
    let count = events.lock().unwrap().len();
    let repeated = completed.prepared().execute_retained(&ctx).unwrap();
    assert_eq!(
        events.lock().unwrap().len(),
        count,
        "a completed plan inherited an execution observer"
    );
    for (before, after) in completed.results().iter().zip(repeated.results()) {
        assert_bit_identical(before, after, "observer-free retained execution");
    }
}

#[test]
fn batch_stage_observers_refuse_unsupported_members_before_any_statistical_work() {
    use antecedent::analysis::{StageEvent, StageResultSink};
    use antecedent_core::execution_attempt::observe_with_context;
    struct Unexpected;
    impl StageResultSink for Unexpected {
        fn on_stage(&self, _: &StageEvent) {
            panic!("unsupported stage observers must not execute");
        }
    }
    let (data, graph) = two_by_two(400, 107);
    let ctx = ExecutionContext::for_tests(107);
    let queries = [cells(vec![set(T1, 1.0), set(T2, 1.0)])];
    let prepared = observe_with_context(&ctx, || {
        BatchStudy::new(data, graph)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .prepare_cells_deferred(&queries, &ctx)
    });
    assert_no_batch_statistical_work(&prepared.report);
    let prepared = prepared.result.unwrap();
    let unsupported =
        observe_with_context(&ctx, || prepared.with_stage_sinks(vec![Arc::new(Unexpected)]));
    assert_no_batch_statistical_work(&unsupported.report);
    assert_eq!(unsupported.result.unwrap_err().reason_code(), Some("stage_stream_unavailable"));
    let wrong_count = prepared.with_stage_sinks(Vec::new()).unwrap_err();
    assert!(matches!(wrong_count, antecedent::CausalError::Compile { .. }));
    assert_eq!(prepared.materialized_plan_count(), 0);
}

#[test]
fn scalar_branches_of_completed_batch_members_rebuild_their_row_bound_shared_design() {
    use antecedent_core::execution_attempt::observe_with_context;
    use antecedent_data::TableView;
    let (data, graph) = two_by_two(500, 109);
    let query = ate(T1, Y1);
    let ctx = ExecutionContext::for_tests(109);
    let completed = deferred_batch(&data, &graph, &[query.clone(), ate(T1, Y2)], &ctx)
        .execute_retained(&ctx)
        .unwrap();
    let member = &completed.prepared().plans()[0];
    let old_covariates =
        member.shared_design().unwrap().covariate.as_ref().unwrap().design_matrix.clone();
    let old_point = completed.results()[0].estimate.as_effect().unwrap().ate.to_bits();
    let names = ["t1", "t2", "y1", "y2", "z"];
    let mut columns: Vec<Vec<f64>> =
        (0..5).map(|id| data.float64_values(VariableId::from_raw(id)).unwrap()).collect();
    for (index, z) in columns[Z as usize].iter_mut().enumerate() {
        *z += 0.4 * (index as f64 * 0.17).sin();
    }
    let changed_z = TabularData::from_f64_columns(
        names.iter().zip(&columns).map(|(name, values)| (*name, values.as_slice())),
    )
    .unwrap();
    let (shorter, _) = two_by_two(420, 110);
    for (label, changed) in
        [("changed confounder values", changed_z), ("changed row count", shorter)]
    {
        let branch = observe_with_context(&ctx, || member.rebind_data(changed.clone(), &ctx));
        assert_no_batch_statistical_work(&branch.report);
        let branch = branch.result.unwrap();
        assert!(branch.score_table().is_none());
        let shared = branch.shared_design().unwrap();
        assert_eq!(shared.fold_seed, 109);
        assert_eq!(shared.fold_ids.len(), changed.row_count());
        let covariates = shared.covariate.as_ref().unwrap();
        assert_eq!(covariates.row_index.len(), changed.row_count());
        assert_ne!(covariates.design_matrix, old_covariates, "{label}: stale shared design");
        for (column, variable) in covariates.adjustment_set.iter().enumerate() {
            let offset = (column + 1) * changed.row_count();
            assert_eq!(
                &covariates.design_matrix[offset..offset + changed.row_count()],
                changed.float64_slice(*variable).unwrap(),
                "{label}: shared adjustment column"
            );
        }
        let ready = branch.materialize(&ctx).unwrap();
        let result = ready.estimate_retained(&ctx).unwrap();
        assert_bit_identical(&result, &solo(&changed, &graph, &query, 109), label);
        ready.execution_record_with_quantity_labels(&result, &ctx, None).unwrap();
    }
    assert_eq!(
        member.shared_design().unwrap().covariate.as_ref().unwrap().design_matrix,
        old_covariates
    );
    assert_eq!(member.shared_design().unwrap().fold_ids.len(), 500);
    assert_eq!(completed.results()[0].estimate.as_effect().unwrap().ate.to_bits(), old_point);
}

#[test]
fn staged_checked_aipw_keeps_its_claim_and_performs_each_fit_only_once() {
    use antecedent::analysis::{StageEvent, StageResultSink};
    use antecedent_core::execution_attempt::{Operation, observe_with_context};
    use std::sync::Mutex;
    struct Recorder(Arc<Mutex<Vec<StageEvent>>>);
    impl StageResultSink for Recorder {
        fn on_stage(&self, event: &StageEvent) {
            self.0.lock().unwrap().push(event.clone());
        }
    }
    let (data, graph) = two_by_two(400, 113);
    let ctx = ExecutionContext::for_tests(113);
    for bootstrap in [0, 8] {
        let prepared = BatchStudy::new(data.clone(), graph.clone())
            .estimator(EstimatorId::Aipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(bootstrap)
            .prepare_deferred(&[ate(T1, Y1)], &ctx)
            .unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let observed =
            prepared.with_stage_sinks(vec![Arc::new(Recorder(Arc::clone(&events)))]).unwrap();
        let plain = observe_with_context(&ctx, || prepared.execute_retained(&ctx));
        let staged = observe_with_context(&ctx, || observed.execute_retained(&ctx));
        for operation in [
            Operation::GlmFit,
            Operation::LeastSquaresSolve,
            Operation::AdjustedFit,
            Operation::LearnerFit,
            Operation::ScoreConstruction,
        ] {
            assert_eq!(
                plain.report.counts(operation),
                staged.report.counts(operation),
                "bootstrap={bootstrap}, {operation:?}: staging repeated numerical work"
            );
        }
        let plain = plain.result.unwrap();
        let staged = staged.result.unwrap();
        let plain_result = &plain.results()[0];
        let staged_result = &staged.results()[0];
        assert_bit_identical(plain_result, staged_result, "staged checked AIPW");
        let a = plain_result.estimate.as_effect().unwrap();
        let b = staged_result.estimate.as_effect().unwrap();
        assert_eq!(a.se_bootstrap().map(f64::to_bits), b.se_bootstrap().map(f64::to_bits));
        assert_eq!(a.uncertainty.bootstrap, b.uncertainty.bootstrap);
        assert_eq!(
            plain_result.lowered_program_identity(),
            staged_result.lowered_program_identity()
        );
        let (plain_record, mut plain_body) = plain.prepared().plans()[0]
            .execution_record_with_quantity_labels(plain_result, &ctx, None)
            .unwrap();
        let (staged_record, mut staged_body) = staged.prepared().plans()[0]
            .execution_record_with_quantity_labels(staged_result, &ctx, None)
            .unwrap();
        // result_digest excludes only native execution observations; all scientific
        // metadata, including diagnostic order and provenance, must remain identical.
        plain_body.native_execution = None;
        staged_body.native_execution = None;
        assert_eq!(plain_body, staged_body);
        assert_eq!(plain_record.identities, staged_record.identities);
        assert_eq!(plain_record.claim, staged_record.claim);
        let events = events.lock().unwrap();
        assert_eq!(
            events.iter().map(StageEvent::stage_id).collect::<Vec<_>>(),
            ["identify", "estimate_point", "uncertainty", "validate"]
        );
        let StageEvent::Point { estimate: point } = &events[1] else { unreachable!() };
        let StageEvent::Uncertainty { estimate: uncertain } = &events[2] else { unreachable!() };
        assert_eq!(point.ate.to_bits(), uncertain.ate.to_bits());
        assert!(point.se_bootstrap().is_none(), "bootstrap ran before the point event");
        assert!(point.uncertainty.bootstrap.replicates_ok.is_none());
        if bootstrap > 0 {
            assert_eq!(uncertain.uncertainty.bootstrap.replicates_ok, Some(bootstrap));
            assert!(uncertain.se_bootstrap().is_some());
        }
    }
}

#[test]
fn checked_aipw_point_observer_cancellation_precedes_bootstrap_work() {
    use antecedent::analysis::{StageEvent, StageResultSink};
    use antecedent_core::execution::CancellationToken;
    use antecedent_core::execution_attempt::{Operation, observe_with_context};
    use std::sync::Mutex;
    struct CancelAtPoint(CancellationToken, Arc<Mutex<Vec<&'static str>>>);
    impl StageResultSink for CancelAtPoint {
        fn on_stage(&self, event: &StageEvent) {
            self.1.lock().unwrap().push(event.stage_id());
            if matches!(event, StageEvent::Point { .. }) {
                self.0.cancel();
            }
        }
    }
    let (data, graph) = two_by_two(400, 127);
    let ctx = ExecutionContext::for_tests(127);
    let base = deferred_batch(&data, &graph, &[ate(T1, Y1)], &ctx);
    let point_only = observe_with_context(&ctx, || base.execute_retained(&ctx));
    point_only.result.unwrap();
    let prepared = BatchStudy::new(data, graph)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(8)
        .prepare_deferred(&[ate(T1, Y1)], &ctx)
        .unwrap();
    let mut cancelled = ctx.clone();
    cancelled.cancellation = CancellationToken::new();
    let events = Arc::new(Mutex::new(Vec::new()));
    let observed = prepared
        .with_stage_sinks(vec![Arc::new(CancelAtPoint(
            cancelled.cancellation.clone(),
            Arc::clone(&events),
        ))])
        .unwrap();
    let stopped = observe_with_context(&cancelled, || observed.execute_retained(&cancelled));
    assert!(matches!(stopped.result, Err(antecedent::CausalError::Cancelled { .. })));
    assert_eq!(*events.lock().unwrap(), ["identify", "estimate_point"]);
    for operation in [Operation::GlmFit, Operation::LeastSquaresSolve, Operation::ScoreConstruction]
    {
        assert_eq!(
            point_only.report.counts(operation),
            stopped.report.counts(operation),
            "{operation:?}: cancellation must stop before any bootstrap fit"
        );
    }
    assert_eq!(prepared.materialized_plan_count(), 0);
    assert_eq!(prepared.retained_score_table_count(), 0);
}
