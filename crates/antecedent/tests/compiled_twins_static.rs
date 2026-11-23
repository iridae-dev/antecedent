//! The raw static estimators and the class identifiers are the same computation as their
//! compiled routes.
//!
//! A caller-configured estimator handed to `Study::estimator` is used verbatim by the compiled
//! static route: the route prepares it against the identified estimand and fits it with the
//! caller's execution context. Each test fits the estimator directly (prepare, then fit, on the
//! estimand and assumptions the compiled study identified) and compares the point estimate, the
//! analytic standard error and the bootstrap standard error with the compiled result, bit for
//! bit. The interval a study publishes is a function of exactly these values and the recorded
//! standard-error kind, which is compared as well.
//!
//! The identifier tests compare the envelope `GeneralizedAdjustmentIdentifier` returns for a PAG
//! and a CPDAG, and the result `TemporalMediationIdentifier::identify_with_horizon` returns, with
//! the identification the compiled study derives from the same structure and query. The compiled
//! route configures the identifier with the caller's execution context, which bounds its storage
//! and cancellation and does not change the answer.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "the bits are compared exactly and the fixtures build columns from small counts"
)]

use std::sync::Arc;

use antecedent::identify::{
    GeneralizedAdjustmentIdentifier, IdentificationEnvelope, TemporalMediationIdentifier,
};
use antecedent::{Identification, IdentifierId, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec,
    MediationContrast, MediationQuery, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TabularData, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::{
    AipwAte, AipwWorkspace, AnalyticSeKind, CausalForest, DistanceMatching, DmlAte, DrLearner,
    EffectEstimate, EstimationWorkspace, FrontDoorTwoStage, FrontDoorWorkspace, GlmAdjustmentAte,
    GlmAdjustmentWorkspace, LinearAdjustmentAte, PropensityEstimationWorkspace, PropensityMatching,
    PropensityStratification, PropensityWeighting, TwoStageLeastSquares,
    TwoStageLeastSquaresWorkspace, WaldIv,
};
use antecedent_graph::{Cpdag, Dag, DenseNodeId, Pag, TemporalDag, ensure_lagged};

#[path = "fixtures_common/mod.rs"]
mod common;

use common::fixtures::confounded_scm;

fn bits(value: f64) -> u64 {
    value.to_bits()
}

/// The direct fit and the compiled result agree on every number the study publishes.
fn assert_twin(label: &str, direct: &EffectEstimate, compiled: &StudyResult) {
    let compiled = compiled.estimate().as_effect().expect("an effect estimate");
    assert_eq!(
        bits(direct.ate),
        bits(compiled.ate),
        "{label}: point {} vs {}",
        direct.ate,
        compiled.ate
    );
    assert_eq!(
        bits(direct.se_analytic()),
        bits(compiled.se_analytic()),
        "{label}: analytic se {} vs {}",
        direct.se_analytic(),
        compiled.se_analytic()
    );
    assert_eq!(
        direct.se_bootstrap().map(bits),
        compiled.se_bootstrap().map(bits),
        "{label}: bootstrap se"
    );
    assert_eq!(
        direct.uncertainty.analytic.kind, compiled.uncertainty.analytic.kind,
        "{label}: standard-error kind"
    );
    assert_eq!(direct.provenance.n_obs, compiled.provenance.n_obs, "{label}: analysed rows");
}

// --- confounded fixture: adjustment estimators --------------------------------------------

struct Confounded {
    data: TabularData,
    dag: Dag,
    query: AverageEffectQuery,
}

fn confounded() -> Confounded {
    let (data, dag, query) = confounded_scm(400, 21);
    Confounded { data, dag, query }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(73)
}

fn run_confounded(fixture: &Confounded, spec: impl Into<antecedent::EstimatorSpec>) -> StudyResult {
    Study::tabular(fixture.data.clone())
        .graph(fixture.dag.clone())
        .query(fixture.query.clone())
        .estimator(spec)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap()
}

#[test]
fn linear_adjustment_point_fit_equals_the_compiled_route() {
    let fixture = confounded();
    let estimator = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let compiled = run_confounded(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct =
        estimator.fit(&prepared, &mut EstimationWorkspace::default(), &ctx(), assumptions).unwrap();
    assert_twin("linear adjustment", &direct, &compiled);
}

#[test]
fn linear_adjustment_bootstrap_fit_equals_the_compiled_route() {
    let fixture = confounded();
    let estimator = LinearAdjustmentAte::new().with_bootstrap_replicates(40);
    let compiled = run_confounded(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct =
        estimator.fit(&prepared, &mut EstimationWorkspace::default(), &ctx(), assumptions).unwrap();
    assert!(direct.se_bootstrap().is_some(), "the bootstrap ran");
    assert_twin("linear adjustment with bootstrap", &direct, &compiled);
}

#[test]
fn aipw_fit_equals_the_compiled_route() {
    let fixture = confounded();
    let estimator = AipwAte::new().with_bootstrap_replicates(0);
    let compiled = run_confounded(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct =
        estimator.fit(&prepared, &mut AipwWorkspace::default(), &ctx(), assumptions).unwrap();
    assert_twin("aipw", &direct, &compiled);
}

#[test]
fn propensity_weighting_fit_equals_the_compiled_route() {
    let fixture = confounded();
    let estimator = PropensityWeighting::new().with_bootstrap_replicates(0);
    let compiled = run_confounded(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct = estimator
        .fit(&prepared, &mut PropensityEstimationWorkspace::default(), &ctx(), assumptions)
        .unwrap();
    assert_twin("propensity weighting", &direct, &compiled);
}

#[test]
fn propensity_stratification_fit_equals_the_compiled_route() {
    let fixture = confounded();
    let estimator = PropensityStratification::new().with_bootstrap_replicates(0);
    let compiled = run_confounded(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct = estimator
        .fit(&prepared, &mut PropensityEstimationWorkspace::default(), &ctx(), assumptions)
        .unwrap();
    assert_twin("propensity stratification", &direct, &compiled);
}

#[test]
fn propensity_matching_fit_equals_the_compiled_route() {
    let fixture = confounded();
    let estimator = PropensityMatching::new();
    let compiled = run_confounded(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct = estimator
        .fit(&prepared, &mut PropensityEstimationWorkspace::default(), &ctx(), assumptions)
        .unwrap();
    assert_twin("propensity matching", &direct, &compiled);
}

#[test]
fn distance_matching_fit_equals_the_compiled_route() {
    let fixture = confounded();
    let estimator = DistanceMatching::new();
    let compiled = run_confounded(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct = estimator
        .fit(&prepared, &mut PropensityEstimationWorkspace::default(), &ctx(), assumptions)
        .unwrap();
    assert_twin("distance matching", &direct, &compiled);
}

#[test]
fn double_machine_learning_fit_equals_the_compiled_route() {
    let fixture = confounded();
    let estimator = DmlAte::new();
    let compiled = run_confounded(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct = estimator.fit(&prepared, &ctx(), assumptions).unwrap();
    assert_twin("double machine learning", &direct, &compiled);
}

#[test]
fn dr_learner_fit_equals_the_compiled_route() {
    let fixture = confounded();
    let estimator = DrLearner::new();
    let compiled = run_confounded(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct = estimator.fit(&prepared, &ctx(), assumptions).unwrap();
    assert_twin("dr learner", &direct, &compiled);
}

#[test]
fn causal_forest_fit_equals_the_compiled_route() {
    let fixture = confounded();
    let estimator = CausalForest::new().with_n_trees(40);
    let compiled = run_confounded(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct = estimator.fit(&prepared, &ctx(), assumptions).unwrap();
    assert_twin("causal forest", &direct, &compiled);
}

// --- instrumental-variable and front-door fixtures ----------------------------------------

fn table(columns: Vec<(&str, RoleHint, Vec<f64>)>) -> TabularData {
    let n = columns[0].2.len();
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint, _) in &columns {
        builder
            .add_variable(
                *name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(*hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns = columns
        .into_iter()
        .enumerate()
        .map(|(index, (_, _, values))| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(index).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
}

/// Columns t, y, z with a binary instrument.
fn iv_data() -> TabularData {
    let n = 1_600;
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let instrument = (i % 2) as f64;
        let u = ((i * 37 % 101) as f64 - 50.0) / 30.0;
        let treatment = 0.6 * instrument + u;
        z.push(instrument);
        t.push(treatment);
        y.push(2.0 * treatment + u);
    }
    table(vec![
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("z", RoleHint::Context, z),
    ])
}

/// Columns t, m, y with a binary treatment and a mediator that carries the whole effect.
fn frontdoor_data() -> TabularData {
    let n = 2_000;
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let treatment = (i % 2) as f64;
        let u = ((i * 37 % 101) as f64 - 50.0) / 100.0;
        let mediator = 0.7 * treatment + u;
        t.push(treatment);
        m.push(mediator);
        y.push(2.0 * mediator + ((i * 13 % 47) as f64 - 23.0) / 100.0);
    }
    table(vec![
        ("t", RoleHint::TreatmentCandidate, t),
        ("m", RoleHint::Context, m),
        ("y", RoleHint::OutcomeCandidate, y),
    ])
}

fn run_with_identifier(
    data: &TabularData,
    dag: Dag,
    query: AverageEffectQuery,
    identifier: IdentifierId,
    spec: impl Into<antecedent::EstimatorSpec>,
) -> StudyResult {
    Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .identifier(identifier)
        .estimator(spec)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(811))
        .unwrap()
}

fn iv_inputs() -> (TabularData, Dag, AverageEffectQuery) {
    let data = iv_data();
    let dag = Dag::from_named_edges(data.schema(), &[("z", "t"), ("t", "y")]).unwrap();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    (data, dag, query)
}

#[test]
fn wald_iv_fit_equals_the_compiled_route() {
    let (data, dag, query) = iv_inputs();
    let estimator = WaldIv::new().with_se_kind(AnalyticSeKind::Hc1).with_bootstrap_replicates(0);
    let compiled =
        run_with_identifier(&data, dag, query.clone(), IdentifierId::Iv, estimator.clone());
    let prepared = estimator.prepare(&data, &compiled.estimand, &query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct = estimator.fit(&prepared, &ExecutionContext::for_tests(811), assumptions).unwrap();
    assert_twin("wald iv", &direct, &compiled);
}

#[test]
fn two_stage_least_squares_fit_equals_the_compiled_route() {
    let (data, dag, query) = iv_inputs();
    let estimator =
        TwoStageLeastSquares::new().with_se_kind(AnalyticSeKind::Hc1).with_bootstrap_replicates(0);
    let compiled =
        run_with_identifier(&data, dag, query.clone(), IdentifierId::Iv, estimator.clone());
    let prepared = estimator.prepare(&data, &compiled.estimand, &query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct = estimator
        .fit(
            &prepared,
            &mut TwoStageLeastSquaresWorkspace::default(),
            &ExecutionContext::for_tests(811),
            assumptions,
        )
        .unwrap();
    assert_twin("two-stage least squares", &direct, &compiled);
}

#[test]
fn front_door_two_stage_fit_equals_the_compiled_route() {
    let data = frontdoor_data();
    let dag = Dag::from_named_edges(data.schema(), &[("t", "m"), ("m", "y")]).unwrap();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(2), 0.0, 1.0);
    let estimator = FrontDoorTwoStage::new().with_bootstrap_replicates(0);
    let compiled =
        run_with_identifier(&data, dag, query.clone(), IdentifierId::Frontdoor, estimator.clone());
    let prepared = estimator.prepare(&data, &compiled.estimand, &query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct = estimator
        .fit(
            &prepared,
            &mut FrontDoorWorkspace::default(),
            &ExecutionContext::for_tests(811),
            assumptions,
        )
        .unwrap();
    assert_twin("front-door two-stage", &direct, &compiled);
}

// --- binary outcome: logistic adjustment ---------------------------------------------------

#[test]
fn glm_adjustment_fit_equals_the_compiled_route() {
    let n = 600_usize;
    let logistic = |eta: f64| 1.0 / (1.0 + (-eta).exp());
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let covariate = ((i * 37 % 101) as f64 - 50.0) / 25.0;
        let treated = f64::from(((i * 13 % 17) as f64 / 17.0) < logistic(covariate));
        let outcome =
            f64::from(((i * 29 % 19) as f64 / 19.0) < logistic(0.8 * treated + 0.9 * covariate));
        z.push(covariate);
        t.push(treated);
        y.push(outcome);
    }
    let data = table(vec![
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("z", RoleHint::Context, z),
    ]);
    let dag = Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let estimator = GlmAdjustmentAte::new().with_bootstrap_replicates(0);
    let compiled = Study::tabular(data.clone())
        .graph(dag)
        .query(query.clone())
        .estimator(estimator.clone())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap();
    let prepared = estimator.prepare(&data, &compiled.estimand, &query).unwrap();
    let assumptions = compiled.estimate().as_effect().unwrap().provenance.assumptions.clone();
    let direct = estimator
        .fit(&prepared, &mut GlmAdjustmentWorkspace::default(), &ctx(), assumptions)
        .unwrap();
    assert_twin("glm adjustment", &direct, &compiled);
}

// --- class identifiers ---------------------------------------------------------------------

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

/// A table that names `variables` columns and holds no rows.
fn schema_only(variables: usize) -> TabularData {
    let names: Vec<String> = (0..variables).map(|index| format!("v{index}")).collect();
    let empty: &[f64] = &[];
    TabularData::from_f64_columns(names.iter().map(|name| (name.as_str(), empty))).unwrap()
}

fn assert_same_envelope<G>(
    label: &str,
    direct: &IdentificationEnvelope<G>,
    compiled: &IdentificationEnvelope<G>,
) {
    assert_eq!(direct.status, compiled.status, "{label}: status");
    assert_eq!(
        bits(direct.identified_weight.0),
        bits(compiled.identified_weight.0),
        "{label}: identified weight"
    );
    assert_eq!(
        bits(direct.unidentified_weight.0),
        bits(compiled.unidentified_weight.0),
        "{label}: unidentified weight"
    );
    assert_eq!(direct.cases.len(), compiled.cases.len(), "{label}: completions");
    assert_eq!(direct.truncated_completions, compiled.truncated_completions, "{label}: truncated");
    assert_eq!(
        direct.invariant.as_ref().map(|estimand| estimand.adjustment_set.to_vec()),
        compiled.invariant.as_ref().map(|estimand| estimand.adjustment_set.to_vec()),
        "{label}: shared adjustment set"
    );
    for (index, (left, right)) in direct.cases.iter().zip(&compiled.cases).enumerate() {
        assert_eq!(left.result.status, right.result.status, "{label}: case {index} status");
        assert_eq!(
            left.result.estimands.iter().map(|e| e.adjustment_set.to_vec()).collect::<Vec<_>>(),
            right.result.estimands.iter().map(|e| e.adjustment_set.to_vec()).collect::<Vec<_>>(),
            "{label}: case {index} adjustment sets"
        );
    }
}

#[test]
fn pag_envelope_equals_the_compiled_identification() {
    // z -> t, z -> y, t -> y over columns t, y, z: every edge is visible, so z adjusts.
    let mut pag = Pag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        pag.insert_directed(node(from), node(to)).unwrap();
    }
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let direct =
        GeneralizedAdjustmentIdentifier::new().identify_pag_envelope(&pag, &query).unwrap();
    let identification = Study::tabular(schema_only(3))
        .graph(pag)
        .query(query)
        .identify_structure_within(&ExecutionContext::for_tests(5))
        .unwrap();
    let Identification::Envelope { envelope, .. } = identification else {
        panic!("a PAG identifies to a class envelope");
    };
    assert_same_envelope("pag", &direct, &envelope);
}

#[test]
fn cpdag_envelope_equals_the_compiled_identification() {
    // z - t undirected, t -> y, z -> y: both completions identify with the adjustment {z}.
    let mut cpdag = Cpdag::with_variables(3);
    cpdag.insert_undirected(node(2), node(0)).unwrap();
    cpdag.insert_directed(node(0), node(1)).unwrap();
    cpdag.insert_directed(node(2), node(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let direct =
        GeneralizedAdjustmentIdentifier::new().identify_cpdag_envelope(&cpdag, &query).unwrap();
    let identification = Study::tabular(schema_only(3))
        .graph(cpdag)
        .query(query)
        .identify_structure_within(&ExecutionContext::for_tests(5))
        .unwrap();
    let Identification::CpdagEnvelope { envelope, .. } = identification else {
        panic!("a CPDAG identifies to a class envelope");
    };
    assert_same_envelope("cpdag", &direct, &envelope);
}

// --- temporal mediation identifier ---------------------------------------------------------

fn mediation_series() -> TimeSeriesData {
    let n = 241_usize;
    let z: Vec<f64> = (0..n).map(|i| if i % 4 < 2 { 1.0 } else { -1.0 }).collect();
    let t: Vec<f64> =
        (0..n).map(|i| z[i] + if i % 4 == 0 || i % 4 == 2 { 1.0 } else { -1.0 }).collect();
    let mut m = vec![0.0; n];
    let mut y = vec![1.0; n];
    for i in 1..n {
        let noise = (i % 5) as f64 - 2.0;
        m[i] = 0.8 * t[i - 1] + 0.6 * z[i - 1] + 0.15 * noise;
        y[i] = 1.0 + 0.25 * t[i - 1] + 0.55 * m[i] + 5.0 * z[i - 1];
    }
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("m", RoleHint::Context),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns = [t, m, y, z]
        .into_iter()
        .enumerate()
        .map(|(id, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(id).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    TimeSeriesData::try_new(
        OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn mediation_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t0 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    for (from, to) in [(z0, t0), (z1, y0), (z1, m0), (t1, y0), (t1, m0), (m0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

#[test]
fn temporal_mediation_identification_equals_the_compiled_identification() {
    let graph = mediation_graph();
    let query = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    )
    .with_horizons(vec![1_u32])
    .unwrap();
    let identifier = TemporalMediationIdentifier {
        allow_natural_controlled_alias: true,
        ..TemporalMediationIdentifier::new()
    };
    let (direct, _) = identifier.identify_with_horizon(&graph, &query, 1).unwrap();
    let identification = Study::series(mediation_series())
        .graph(graph)
        .query(query)
        .identify_structure_within(&ExecutionContext::for_tests(5))
        .unwrap();
    let Identification::Point { result, .. } = identification else {
        panic!("a fixed temporal graph identifies to a point result");
    };
    assert_eq!(direct.status, result.status, "status");
    assert_eq!(direct.estimands.len(), result.estimands.len(), "estimands");
    for (left, right) in direct.estimands.iter().zip(&result.estimands) {
        assert_eq!(left.method, right.method, "method");
        assert_eq!(left.adjustment_set, right.adjustment_set, "adjustment set");
        assert_eq!(left.mediators, right.mediators, "mediators");
    }
}
