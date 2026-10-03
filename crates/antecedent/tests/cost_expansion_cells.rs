//! Planning-time cost counts of the 2.2 fit routes (cost and cancellation checks): penalized
//! AIPW, clustered DML, the DML score route, batch retarget and the factorized joint cells.
//!
//! Every expectation is a hand count from the declared configuration (grid x (inner folds + 1)
//! per outer fold, folds x strata for the joint cell), and the counted folds are checked
//! against the folds an executed run records.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    BatchRetargetRequest, BatchStudy, EstimatorId, RefuteSuite, RetargetClaim, RetargetContrast,
    Study, estimate_joint_cell_cost,
};
use antecedent_core::{AverageEffectQuery, CausalRng, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{
    AipwAte, ClusterDml, DmlAte, DmlScore, FactorizedJointConfig, LearnerSpec, LinearSpec,
    PropensityNuisance, RidgeTuning,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

const T1: u32 = 0;
const T2: u32 = 1;
const Y1: u32 = 2;
const Z: u32 = 3;

fn stream(seed: u64) -> CausalRng {
    ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0xD5)
}

/// Two binary treatments, one outcome, one confounder.
fn data_and_graph(n: usize, seed: u64) -> (TabularData, Dag) {
    let mut rng = stream(seed);
    let (mut t1, mut t2, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        t1[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (0.2 - 0.8 * zi).exp()));
        t2[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-0.1 + 0.6 * zi).exp()));
        y[i] = 2.0 * t1[i] + 0.5 * t2[i] + zi + 0.3 * standard_normal(&mut rng);
    }
    let data = TabularData::from_f64_columns([
        ("t1", t1.as_slice()),
        ("t2", t2.as_slice()),
        ("y1", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(Z, T1), (Z, T2), (Z, Y1), (T1, Y1), (T2, Y1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph)
}

fn ate(treatment: u32) -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(treatment), VariableId::from_raw(Y1))
}

fn study(
    data: &TabularData,
    graph: &Dag,
    estimator: impl Into<antecedent::EstimatorSpec>,
) -> Study {
    Study::tabular(data.clone())
        .graph(graph.clone())
        .query(ate(T1))
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

fn ridge(grid: &[f64], inner_folds: usize) -> AipwAte {
    AipwAte {
        bootstrap_replicates: 0,
        propensity: PropensityNuisance::ridge_logistic(
            RidgeTuning::new(grid, inner_folds).unwrap(),
        ),
        ..AipwAte::new()
    }
}

/// 3 penalties x (4 inner folds + 1 refit) = 15 propensity fits per outer fold, over 5 folds:
/// 75 propensity and 10 outcome fits. The folds the plan counts are the folds the executed
/// run records, and it records one selected penalty per fold.
#[test]
fn penalized_aipw_cost_is_the_grid_times_inner_folds_per_outer_fold() {
    let (data, graph) = data_and_graph(400, 3);
    let ctx = ExecutionContext::for_tests(3);
    let est = ridge(&[1.0, 10.0, 100.0], 4);
    let study = study(&data, &graph, est.clone());
    let cost = study.prepare(&ctx).unwrap().estimate_cost().unwrap();
    assert_eq!(cost.fit_route, "aipw.ridge_logistic(penalties=3, inner_folds=4)");
    assert_eq!(cost.crossfit_folds, Some(5));
    assert_eq!(cost.propensity_fits_per_pass, Some(5 * 3 * (4 + 1)));
    assert_eq!(cost.outcome_fits_per_pass, Some(10));
    assert_eq!(cost.nuisance_fits_upper_bound, Some(85));
    assert_eq!(cost.propensity_fits_per_pass, Some(5 * est.propensity.planned_fits_per_fold()));
    assert!(cost.seconds.is_none() && !cost.seconds_basis.is_empty());

    let result = study.run(&ctx).unwrap();
    let effect = result.estimate.as_effect().unwrap();
    assert_eq!(effect.crossfit_folds, cost.crossfit_folds.map(|f| usize::try_from(f).unwrap()));
    assert_eq!(effect.learner_provenance.len(), 5, "one selected penalty per counted fold");

    // The count grows with the grid and with the inner folds.
    let fits = |grid: &[f64], inner| {
        let spec = ridge(grid, inner);
        study_cost(&data, &graph, spec).nuisance_fits_upper_bound.unwrap()
    };
    assert!(fits(&[1.0], 3) < fits(&[1.0, 10.0], 3));
    assert!(fits(&[1.0, 10.0], 2) < fits(&[1.0, 10.0], 5));
}

fn study_cost(
    data: &TabularData,
    graph: &Dag,
    estimator: impl Into<antecedent::EstimatorSpec>,
) -> antecedent::CostEstimate {
    let ctx = ExecutionContext::for_tests(3);
    study(data, graph, estimator).prepare(&ctx).unwrap().estimate_cost().unwrap()
}

/// A cluster-DML declaration fits the same logistic and OLS models per fold as plain AIPW,
/// folds by whole clusters, and reports how many distinct clusters it folds by.
#[test]
fn clustered_dml_cost_counts_folds_and_whole_clusters() {
    let (data, graph) = data_and_graph(400, 4);
    let clusters: Vec<u32> = (0..400_u32).map(|i| i / 10).collect();
    let plain = study_cost(&data, &graph, AipwAte { bootstrap_replicates: 0, ..AipwAte::new() });
    let declared = AipwAte {
        bootstrap_replicates: 0,
        cluster_ids: Some(clusters),
        cluster_dml: Some(ClusterDml::new(20).unwrap()),
        ..AipwAte::new()
    };
    let cost = study_cost(&data, &graph, declared);
    assert_eq!(cost.fit_route, "aipw.cluster_dml(unit=cluster)");
    assert_eq!(cost.cluster_labels, Some(40));
    assert_eq!(plain.cluster_labels, None);
    assert_eq!(cost.crossfit_folds, Some(5));
    assert_eq!(cost.nuisance_fits_per_pass, plain.nuisance_fits_per_pass);
    assert_eq!(cost.nuisance_fits_per_pass, Some(15));
}

/// The DML score route counts its own folds: 7 folds of the AIPW score fit 7 propensity and 14
/// outcome models, the partially linear score 7 of each, and the executed run records 7 folds.
#[test]
fn dml_cost_uses_the_declared_folds_and_score() {
    let (data, graph) = data_and_graph(400, 5);
    let aipw = study_cost(&data, &graph, DmlAte::new().with_folds(7));
    assert_eq!(aipw.fit_route, "dml(score=aipw)");
    assert_eq!(aipw.crossfit_folds, Some(7));
    assert_eq!((aipw.propensity_fits_per_pass, aipw.outcome_fits_per_pass), (Some(7), Some(14)));
    let plr = study_cost(
        &data,
        &graph,
        DmlAte::new().with_score(DmlScore::PartiallyLinear).with_folds(7),
    );
    assert_eq!((plr.propensity_fits_per_pass, plr.outcome_fits_per_pass), (Some(7), Some(7)));
    assert!(aipw.seconds.is_none() && plr.seconds.is_none());

    let ctx = ExecutionContext::for_tests(5);
    let result = study(&data, &graph, DmlAte::new().with_folds(7)).run(&ctx).unwrap();
    assert_eq!(result.estimate.as_effect().unwrap().crossfit_folds, Some(7));
}

/// A retarget fits nothing: the cost is the claim, contrast and row counts the reweighting reads.
#[test]
fn batch_retarget_cost_is_zero_fits_and_counts_claims_contrasts_and_rows() {
    let (data, graph) = data_and_graph(300, 6);
    let ctx = ExecutionContext::for_tests(6);
    let prepared = BatchStudy::new(data.clone(), graph)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare(&[ate(T1), ate(T2)], &ctx)
        .unwrap();
    let scores = prepared.prepared_scores();
    let rows = scores.common_rows().unwrap().unwrap();
    let claim = |name: &str, query_index: usize| RetargetClaim {
        name: name.into(),
        query_index,
        weights: vec![1.0; rows.len()],
        depends_on: vec![VariableId::from_raw(Z)],
    };
    let request = |claims: Vec<RetargetClaim>, contrasts: Vec<RetargetContrast>| {
        BatchRetargetRequest { claims, contrasts, expected_snapshot: None }
    };
    let one = scores.estimate_retarget_cost(&request(vec![claim("a", 0)], vec![]));
    let two = scores.estimate_retarget_cost(&request(
        vec![claim("a", 0), claim("b", 1)],
        vec![RetargetContrast {
            name: "d".into(),
            coefficients: vec![("a".into(), 1.0), ("b".into(), -1.0)],
        }],
    ));
    let n = u64::try_from(rows.len()).unwrap();
    assert_eq!((one.claims, one.contrasts, two.claims, two.contrasts), (1, 0, 2, 1));
    assert_eq!(two.snapshot_rows, Some(rows.len()));
    assert_eq!(two.nuisance_fits, 0, "scores are reused, nothing is refit");
    assert_eq!(one.weighted_score_reads, Some(n));
    assert_eq!(two.weighted_score_reads, Some(2 * n));
    assert_eq!(one.covariance_products, Some(n));
    assert_eq!(two.covariance_products, Some(3 * n));
    assert!(two.seconds.is_none() && !two.seconds_basis.is_empty());
    // A retarget of the prepared scores adds no nuisance fits to the batch's own bound.
    let batch = prepared.estimate_cost().unwrap();
    assert_eq!(batch.nuisance_fits_upper_bound, Some(2 * 15));
}

/// Two components under the orderings `[0, 1]` and `[1, 0]`: the distinct conditionals are
/// `P(T0)`, `P(T1)`, `P(T1 | T0 = 0, 1)` and `P(T0 | T1 = 0, 1)`, six strata per fold, so 5
/// folds give 30 conditional fits; the four cells give 20 outcome fits. The ridge route
/// multiplies each conditional by its penalty selection, a declared learner fits it once.
#[test]
fn factorized_joint_cell_cost_follows_folds_orderings_and_route() {
    let orderings = vec![vec![0, 1], vec![1, 0]];
    let tuning = RidgeTuning::new(&[1.0, 10.0], 3).unwrap();
    let mut config = FactorizedJointConfig::new(tuning.clone());
    let ridge_cost = estimate_joint_cell_cost(&config, 2, &orderings, Some(600)).unwrap();
    assert_eq!(ridge_cost.fit_route, "joint_cells.ridge_logistic(penalties=2, inner_folds=3)");
    assert_eq!((ridge_cost.cells, ridge_cost.orderings, ridge_cost.folds), (4, 2, 5));
    assert_eq!(ridge_cost.conditional_strata, 6);
    assert_eq!(ridge_cost.conditional_fits, 30);
    assert_eq!(ridge_cost.propensity_fits, 30 * 2 * (3 + 1));
    assert_eq!(ridge_cost.outcome_fits, 20);
    assert_eq!(ridge_cost.nuisance_fits, 240 + 20);
    assert_eq!(ridge_cost.rows, Some(600));
    assert!(ridge_cost.seconds.is_none() && !ridge_cost.seconds_basis.is_empty());

    config.learner = Some(LearnerSpec::Linear(LinearSpec {}));
    let learner_cost = estimate_joint_cell_cost(&config, 2, &orderings, None).unwrap();
    assert!(
        learner_cost.fit_route.starts_with("joint_cells.learner("),
        "{}",
        learner_cost.fit_route
    );
    assert_eq!(learner_cost.propensity_fits, 30);
    assert_eq!(learner_cost.nuisance_fits, 50);

    // One ordering shares fewer conditionals: P(T0), P(T1 | T0 = 0, 1) is three strata per fold.
    let single = FactorizedJointConfig::new(tuning);
    let one = estimate_joint_cell_cost(&single, 2, &[vec![0, 1]], None).unwrap();
    assert_eq!((one.conditional_strata, one.conditional_fits), (3, 15));
    // Three components under all six orderings: monotone in orderings, components and folds.
    let all: Vec<Vec<usize>> = antecedent_estimate::orderings_for(3, &[0, 1, 2], true).unwrap();
    let fits = |config: &FactorizedJointConfig, k, orderings: &[Vec<usize>]| {
        estimate_joint_cell_cost(config, k, orderings, None).unwrap().nuisance_fits
    };
    assert!(fits(&single, 2, &[vec![0, 1]]) < fits(&single, 2, &orderings));
    assert!(fits(&single, 3, &all[..1]) < fits(&single, 3, &all[..3]));
    assert!(fits(&single, 3, &all[..3]) < fits(&single, 3, &all));
    assert!(fits(&single, 2, &orderings) < fits(&single, 3, &all));
    let more_folds = FactorizedJointConfig { folds: 10, ..single.clone() };
    assert!(fits(&single, 2, &orderings) < fits(&more_folds, 2, &orderings));
}

/// A declaration the fit refuses before any fit is refused by the cost too, never costed.
#[test]
fn a_declaration_the_fit_refuses_is_not_costed() {
    let config = FactorizedJointConfig::new(RidgeTuning::default());
    for orderings in [vec![], vec![vec![0, 1], vec![0, 1]], vec![vec![0, 0]]] {
        let error = estimate_joint_cell_cost(&config, 2, &orderings, None).unwrap_err();
        assert!(error.to_string().contains("joint_cells.ordering"), "{error}");
    }
    let bad = FactorizedJointConfig { folds: 1, ..config };
    let error = estimate_joint_cell_cost(&bad, 2, &[vec![0, 1]], None).unwrap_err();
    assert!(error.to_string().contains("joint_cells.config"), "{error}");
}
