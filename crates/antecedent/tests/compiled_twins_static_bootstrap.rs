//! The strategy-table bootstrap attachment is the uncertainty stage of the compiled static route.
//!
//! `antecedent::strategy_table::attach_static_bootstrap` adds the facade bootstrap standard
//! error to a point estimate without refitting the point model. The compiled static route
//! (`StaticAteDag`) reaches the same function from its uncertainty stage for the weighting,
//! stratification, AIPW, GLM and front-door estimators. Each test runs the compiled study twice
//! on one fixture: once with no bootstrap replicates, giving the point estimate the direct call
//! takes, and once with replicates. Attaching the bootstrap directly to the first result must
//! reproduce the second bit for bit: point, analytic and bootstrap standard errors, the
//! standard-error kind and the interval bounds. Both paths run under the caller's execution
//! context, so the bootstrap draws use the same seed.
//!
//! The direct call returns a bare `EffectEstimate`; the calibration basis and coverage key the
//! interval is recorded under belong to the compiled result, which is the replacement.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]
#![allow(deprecated, reason = "exercises the deprecated direct facade path")]

use antecedent::strategy_table::{
    StaticEstimateInput, StaticEstimateWorkspaces, attach_static_bootstrap,
};
use antecedent::{EstimatorId, RefuteSuite, Study, StudyResult};
use antecedent_core::{AverageEffectQuery, ExecutionContext};
use antecedent_data::TabularData;
use antecedent_estimate::EffectEstimate;
use antecedent_graph::Dag;

#[path = "fixtures_common/mod.rs"]
mod common;

use common::fixtures::confounded_scm;

const REPLICATES: u32 = 40;

fn bits(value: f64) -> u64 {
    value.to_bits()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(73)
}

struct Confounded {
    data: TabularData,
    dag: Dag,
    query: AverageEffectQuery,
}

fn confounded() -> Confounded {
    let (data, dag, query) = confounded_scm(400, 21);
    Confounded { data, dag, query }
}

fn run(fixture: &Confounded, estimator: EstimatorId, replicates: u32) -> StudyResult {
    Study::tabular(fixture.data.clone())
        .graph(fixture.dag.clone())
        .query(fixture.query.clone())
        .estimator(estimator)
        .bootstrap_replicates(replicates)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap()
}

fn assert_twin(label: &str, direct: &EffectEstimate, compiled: &StudyResult) {
    let compiled = compiled.estimate().as_effect().expect("an effect estimate");
    assert_eq!(bits(direct.ate), bits(compiled.ate), "{label}: point");
    assert_eq!(
        bits(direct.se_analytic()),
        bits(compiled.se_analytic()),
        "{label}: analytic standard error"
    );
    assert!(direct.se_bootstrap().is_some(), "{label}: the bootstrap ran");
    assert_eq!(
        direct.se_bootstrap().map(bits),
        compiled.se_bootstrap().map(bits),
        "{label}: bootstrap standard error"
    );
    assert_eq!(
        direct.uncertainty.analytic.kind, compiled.uncertainty.analytic.kind,
        "{label}: standard-error kind"
    );
    assert_eq!(
        format!("{:?}", direct.uncertainty),
        format!("{:?}", compiled.uncertainty),
        "{label}: every uncertainty component, interval bounds included"
    );
    assert_eq!(direct.provenance.n_obs, compiled.provenance.n_obs, "{label}: analysed rows");
}

fn attach_equals_the_compiled_route(estimator: EstimatorId, label: &str) {
    let fixture = confounded();
    let point_study = run(&fixture, estimator, 0);
    let compiled = run(&fixture, estimator, REPLICATES);
    let point = point_study.estimate().as_effect().expect("an effect estimate").clone();
    assert!(point.se_bootstrap().is_none(), "{label}: the point study ran no bootstrap");
    let direct = attach_static_bootstrap(
        estimator,
        StaticEstimateInput {
            data: &fixture.data,
            estimand: &point_study.estimand,
            query: &fixture.query,
            bootstrap_replicates: REPLICATES,
            overlap_policy: None,
            population_registry: None,
            sampling_design: None,
        },
        point,
        &ctx(),
        &mut StaticEstimateWorkspaces::default(),
    )
    .unwrap();
    assert_twin(label, &direct, &compiled);
}

#[test]
fn propensity_weighting_bootstrap_attachment_equals_the_compiled_route() {
    attach_equals_the_compiled_route(EstimatorId::PropensityWeighting, "propensity weighting");
}

#[test]
fn propensity_stratification_bootstrap_attachment_equals_the_compiled_route() {
    attach_equals_the_compiled_route(
        EstimatorId::PropensityStratification,
        "propensity stratification",
    );
}

#[test]
fn aipw_bootstrap_attachment_equals_the_compiled_route() {
    attach_equals_the_compiled_route(EstimatorId::Aipw, "aipw");
}

#[test]
fn an_estimator_without_a_facade_bootstrap_is_refused() {
    let fixture = confounded();
    let point_study = run(&fixture, EstimatorId::PropensityWeighting, 0);
    let point = point_study.estimate().as_effect().unwrap().clone();
    let refused = attach_static_bootstrap(
        EstimatorId::Dml,
        StaticEstimateInput {
            data: &fixture.data,
            estimand: &point_study.estimand,
            query: &fixture.query,
            bootstrap_replicates: REPLICATES,
            overlap_policy: None,
            population_registry: None,
            sampling_design: None,
        },
        point,
        &ctx(),
        &mut StaticEstimateWorkspaces::default(),
    );
    assert!(refused.is_err(), "double ML owns its own uncertainty and is refused");
}
