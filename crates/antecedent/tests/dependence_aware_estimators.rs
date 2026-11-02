//! A declared clustered-panel or dependent-time-series design is read by the cross-fitted
//! estimators and the front-door functional, or the study is refused.
//!
//! Double ML, the DR-learner and the causal forest read a clustered panel only through a declared
//! whole-cluster independence unit (and never a series); the front-door functional reads neither.
//! An estimator selected by id alone is resolved to its default configuration and held to the
//! same rule, so selecting it by id is no way around the declaration.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{CausalError, EstimatorId, EstimatorSpec, IdentifierId, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, DeclaredSamplingDesign, ExecutionContext, StreamDomain, VariableId,
};
use antecedent_data::{DataError, TabularData};
use antecedent_estimate::{AnalyticSeKind, CausalForest, ClusterDml, DmlAte, DrLearner};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

const T: u32 = 0;
const Y: u32 = 1;
const Z0: u32 = 2;
const Z1: u32 = 3;
const GROUPS: usize = 40;
const SIZE: usize = 10;

const PANEL: DeclaredSamplingDesign = DeclaredSamplingDesign::ClusteredPanel;
const SERIES: DeclaredSamplingDesign = DeclaredSamplingDesign::DependentTimeSeries;

struct Clustered {
    data: TabularData,
    graph: Dag,
    cluster: Vec<u32>,
}

/// `GROUPS` clusters of `SIZE` rows with a cluster-level shock; back-door adjustment on `z0, z1`.
fn clustered() -> Clustered {
    let mut rng = ExecutionContext::for_tests(31).rng.stream_for(StreamDomain::Test, 0xD3);
    let n = GROUPS * SIZE;
    let (mut t, mut y, mut z0, mut z1) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut cluster = vec![0_u32; n];
    for g in 0..GROUPS {
        let shift = 0.7 * standard_normal(&mut rng);
        let shock = 1.5 * standard_normal(&mut rng);
        for k in 0..SIZE {
            let i = g * SIZE + k;
            z0[i] = shift + standard_normal(&mut rng);
            z1[i] = standard_normal(&mut rng);
            let eta = 0.6 * z0[i] - 0.4 * z1[i];
            t[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-eta).exp()));
            y[i] = 2.0 * t[i] + z0[i] + 0.5 * z1[i] + shock + 0.5 * standard_normal(&mut rng);
            cluster[i] = u32::try_from(g).unwrap();
        }
    }
    let columns = [("t", t), ("y", y), ("z0", z0), ("z1", z1)];
    let borrowed: Vec<(&str, &[f64])> = columns.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(Z0, T), (Z1, T), (Z0, Y), (Z1, Y), (T, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    Clustered { data, graph, cluster }
}

fn ate() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(T), VariableId::from_raw(Y))
}

fn build(
    fixture: &Clustered,
    spec: impl Into<EstimatorSpec>,
    design: DeclaredSamplingDesign,
) -> Result<Study, CausalError> {
    Study::tabular(fixture.data.clone())
        .graph(fixture.graph.clone())
        .query(ate())
        .estimator(spec)
        .refute(RefuteSuite::None)
        .sampling_design(design)
        .build()
}

/// The refusal is typed `option_not_applicable` and names a next attempt.
fn assert_refused_with_next_attempt(error: &CausalError) {
    assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("this estimator reads no"), "{message}");
    assert!(message.contains("Next attempt"), "{message}");
}

/// The estimator ran under its declared units: a cluster sandwich with a Student-t reference.
fn assert_read_the_panel(study: &Study) {
    let result = study.run(&ExecutionContext::for_tests(31)).unwrap();
    let effect = result.estimate().as_effect().expect("an average effect");
    assert_eq!(effect.uncertainty.analytic.kind, Some(AnalyticSeKind::Cluster));
    assert!(effect.uncertainty.analytic.reference_df.is_some(), "a few-cluster t reference");
    assert!(effect.uncertainty.analytic.se.is_finite() && effect.uncertainty.analytic.se > 0.0);
}

#[test]
fn dml_by_id_refuses_a_declared_panel_and_series() {
    let fixture = clustered();
    for design in [PANEL, SERIES] {
        let error = build(&fixture, EstimatorId::Dml, design).expect_err("the id reads no units");
        assert_refused_with_next_attempt(&error);
    }
}

#[test]
fn dml_without_units_refuses_and_with_units_reads_the_panel() {
    let fixture = clustered();
    let error = build(&fixture, DmlAte::new(), PANEL).expect_err("rows stand alone");
    assert_refused_with_next_attempt(&error);
    assert!(error.to_string().contains("with_cluster_dml"), "{error}");

    let units = DmlAte::new()
        .with_cluster_dml(ClusterDml::new(20).unwrap())
        .with_cluster_ids(fixture.cluster.clone());
    // A series is never read by a cluster cross-fit, however the units are declared.
    let error = build(&fixture, units.clone(), SERIES).expect_err("no serial lag is read");
    assert_refused_with_next_attempt(&error);
    assert_read_the_panel(&build(&fixture, units, PANEL).unwrap());
}

#[test]
fn dr_learner_by_id_and_without_units_refuse_and_with_units_reads_the_panel() {
    let fixture = clustered();
    for design in [PANEL, SERIES] {
        let error = build(&fixture, EstimatorId::DrLearner, design).expect_err("id reads none");
        assert_refused_with_next_attempt(&error);
    }
    let error = build(&fixture, DrLearner::new(), PANEL).expect_err("rows stand alone");
    assert_refused_with_next_attempt(&error);

    let units = DrLearner::new()
        .with_cluster_dml(ClusterDml::new(20).unwrap())
        .with_cluster_ids(fixture.cluster.clone());
    assert_read_the_panel(&build(&fixture, units, PANEL).unwrap());
}

#[test]
fn causal_forest_by_id_and_without_units_refuse_and_with_units_reads_the_panel() {
    let fixture = clustered();
    for design in [PANEL, SERIES] {
        let error = build(&fixture, EstimatorId::CausalForest, design).expect_err("id reads none");
        assert_refused_with_next_attempt(&error);
    }
    let error = build(&fixture, CausalForest::new(), PANEL).expect_err("rows stand alone");
    assert_refused_with_next_attempt(&error);

    let units = CausalForest::new()
        .with_n_trees(20)
        .with_cluster_dml(ClusterDml::new(20).unwrap())
        .with_cluster_ids(fixture.cluster.clone());
    assert_read_the_panel(&build(&fixture, units, PANEL).unwrap());
}

#[test]
fn independent_subject_trajectories_declare_no_dependence_to_read() {
    let fixture = clustered();
    for id in [EstimatorId::Dml, EstimatorId::DrLearner, EstimatorId::CausalForest] {
        build(&fixture, id, DeclaredSamplingDesign::IndependentSubjectTrajectories)
            .unwrap_or_else(|error| panic!("{id:?}: {error:?}"));
    }
}

const FRONTDOOR_ROWS: usize = 300;

/// Binary treatment `t`, binary mediator `m`, outcome `y` confounded with `t` by an unmeasured
/// `u`; the graph is the front-door shape `t -> m -> y`.
fn frontdoor() -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(37).rng.stream_for(StreamDomain::Test, 0xFD);
    let n = FRONTDOOR_ROWS;
    let (mut t, mut y, mut m) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let u = f64::from(standard_normal(&mut rng) > 0.25);
        t[i] = f64::from(standard_normal(&mut rng) + 0.8 * u > 0.6);
        m[i] = f64::from(standard_normal(&mut rng) + t[i] > 0.5);
        y[i] = 2.0 * m[i] * (0.5 + u) + 0.5 * u + 0.6 * standard_normal(&mut rng);
    }
    let columns = [("t", t), ("y", y), ("m", m)];
    let borrowed: Vec<(&str, &[f64])> = columns.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    (data, graph, query)
}

fn frontdoor_study(design: DeclaredSamplingDesign, replicates: u32) -> Result<Study, CausalError> {
    let (data, graph, query) = frontdoor();
    Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(IdentifierId::Frontdoor)
        .estimator(EstimatorId::FrontDoorFunctional)
        .bootstrap_replicates(replicates)
        .refute(RefuteSuite::None)
        .sampling_design(design)
        .build()
}

fn is_design_refusal(error: &CausalError) -> bool {
    matches!(
        error,
        CausalError::Data(DataError::SamplingDesign(_))
            | CausalError::Estimate(antecedent_estimate::EstimationError::Data(
                DataError::SamplingDesign(_)
            ))
    )
}

#[test]
fn frontdoor_functional_refuses_a_declared_panel_and_series() {
    for design in [PANEL, SERIES] {
        let error = frontdoor_study(design, 0).expect_err("the functional carries no labels");
        assert_refused_with_next_attempt(&error);
        assert!(error.to_string().contains("front-door functional"), "{error}");
    }
}

#[test]
fn frontdoor_functional_forwards_the_declared_design_to_its_bootstrap() {
    // Rows are the functional's only draw unit, and a declared design never allows a row draw.
    let design = DeclaredSamplingDesign::IndependentSubjectTrajectories;
    let error = frontdoor_study(design, 20)
        .and_then(|study| study.run(&ExecutionContext::for_tests(37)))
        .expect_err("a row bootstrap must not run under a declared trajectory design");
    assert!(is_design_refusal(&error), "unexpected refusal: {error:?}");
    // Without resampling there is nothing to refuse: the analytic error is the published one.
    let result = frontdoor_study(design, 0).unwrap().run(&ExecutionContext::for_tests(37)).unwrap();
    assert!(result.estimate().as_effect().unwrap().uncertainty.analytic.se.is_finite());
}
