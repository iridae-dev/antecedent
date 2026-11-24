//! Learner factories declared on a study reach the same factory fit as the direct kernel call.
//!
//! `StudyBuilder::learner_factories` declares the outcome-regression and propensity factories of
//! a study. The compiled static average-effect route over a DAG then prepares the study's
//! configured cross-fitted estimator on the identified estimand and calls its
//! `fit_with_factories` with the caller's execution context. Each twin test fits the estimator
//! directly (prepare, then `fit_with_factories`, on the estimand and the identification
//! assumptions the compiled study derived) and compares the point estimate, the analytic
//! standard error, its kind, the analysed rows, the fold assignment, the learner provenance and
//! the per-row effects with the compiled result, bit for bit. The cross-fit twin checks that the
//! compiled AIPW score table carries exactly the out-of-fold propensities the nuisance cross-fit
//! returns on the compiled fold plan. The known-truth test checks the compiled factory fit
//! against the planted effect of the fixture, and the refusal tests check that a declaration
//! the factory route would not serve is refused at build.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use std::sync::Arc;

use antecedent::analysis::declared_learners::LearnerFactories;
use antecedent::estimate::external_learner::{
    DesignView, IsolatedFactory, LearnerFactory, PredictionTask, cross_fit_aipw_with_factories,
};
use antecedent::{RefuteSuite, Study, StudyResult};
use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext};
use antecedent_data::TabularData;
use antecedent_estimate::{
    AipwAte, CausalForest, DmlAte, DmlScore, DrLearner, EffectEstimate, LearnerSpec, LinearSpec,
    LogisticSpec, PropensityWeighting,
};
use antecedent_graph::Dag;

#[path = "fixtures_common/mod.rs"]
mod common;

use common::fixtures::confounded_scm;

const FACTORY_IDENTITY: &str = "outcome=linear;treatment=logistic";

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(73)
}

struct Fixture {
    data: TabularData,
    dag: Dag,
    query: AverageEffectQuery,
}

fn fixture() -> Fixture {
    let (data, dag, query) = confounded_scm(400, 21);
    Fixture { data, dag, query }
}

fn factories() -> LearnerFactories {
    let outcome: Arc<dyn LearnerFactory> = Arc::from(
        antecedent_learn::resolve_for(
            LearnerSpec::Linear(LinearSpec::default()),
            PredictionTask::Regression,
        )
        .unwrap(),
    );
    let treatment: Arc<dyn LearnerFactory> = Arc::from(
        antecedent_learn::resolve_for(
            LearnerSpec::Logistic(LogisticSpec::default()),
            PredictionTask::BinaryProbability,
        )
        .unwrap(),
    );
    LearnerFactories::new(outcome, treatment, FACTORY_IDENTITY).unwrap()
}

fn run(fixture: &Fixture, spec: impl Into<antecedent::EstimatorSpec>) -> StudyResult {
    Study::tabular(fixture.data.clone())
        .graph(fixture.dag.clone())
        .query(fixture.query.clone())
        .estimator(spec)
        .learner_factories(factories())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap()
}

/// The identification assumptions of the compiled study: the direct call receives exactly
/// these, and the estimator adds its own record of the unverified factories.
fn identification_assumptions(compiled: &StudyResult) -> AssumptionSet {
    compiled.identification.required_assumptions.clone()
}

fn bits(values: Option<&[f64]>) -> Option<Vec<u64>> {
    values.map(|values| values.iter().map(|v| v.to_bits()).collect())
}

fn assert_twin(label: &str, direct: &EffectEstimate, compiled: &StudyResult) {
    let compiled = compiled.estimate().as_effect().expect("an effect estimate");
    assert_eq!(direct.ate.to_bits(), compiled.ate.to_bits(), "{label}: point");
    assert_eq!(
        direct.se_analytic().to_bits(),
        compiled.se_analytic().to_bits(),
        "{label}: analytic standard error"
    );
    assert_eq!(
        direct.uncertainty.analytic.kind, compiled.uncertainty.analytic.kind,
        "{label}: standard-error kind"
    );
    assert_eq!(direct.provenance.n_obs, compiled.provenance.n_obs, "{label}: analysed rows");
    assert_eq!(
        direct.crossfit.fold_assignment, compiled.crossfit.fold_assignment,
        "{label}: fold assignment"
    );
    assert_eq!(
        direct.crossfit.learner_provenance, compiled.crossfit.learner_provenance,
        "{label}: learner provenance"
    );
    assert_eq!(
        bits(direct.heterogeneity.cate.as_deref()),
        bits(compiled.heterogeneity.cate.as_deref()),
        "{label}: per-row effects"
    );
    assert_eq!(
        bits(direct.heterogeneity.cate_se.as_deref()),
        bits(compiled.heterogeneity.cate_se.as_deref()),
        "{label}: per-row effect standard errors"
    );
    assert!(
        compiled.provenance.assumptions.entries.iter().any(|record| matches!(
            &record.assumption,
            antecedent_core::Assumption::Custom { id, .. } if &**id == "dml.caller_factory.unverified"
        )),
        "{label}: the compiled result records the unverified factories"
    );
}

#[test]
fn double_machine_learning_with_declared_factories_equals_the_factory_fit() {
    let fixture = fixture();
    let estimator = DmlAte::new();
    let compiled = run(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let factories = factories();
    let direct = estimator
        .fit_with_factories(
            &prepared,
            &ctx(),
            identification_assumptions(&compiled),
            factories.outcome(),
            factories.treatment(),
        )
        .unwrap();
    assert_twin("double machine learning", &direct, &compiled);
}

#[test]
fn partially_linear_double_machine_learning_with_declared_factories_equals_the_factory_fit() {
    let fixture = fixture();
    let estimator = DmlAte::new().with_score(DmlScore::PartiallyLinear);
    let compiled = run(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let factories = factories();
    let direct = estimator
        .fit_with_factories(
            &prepared,
            &ctx(),
            identification_assumptions(&compiled),
            factories.outcome(),
            factories.treatment(),
        )
        .unwrap();
    assert_twin("partially linear double machine learning", &direct, &compiled);
}

#[test]
fn aipw_with_declared_factories_equals_the_factory_fit() {
    let fixture = fixture();
    let estimator = AipwAte::new().with_bootstrap_replicates(0);
    let compiled = run(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let factories = factories();
    let direct = estimator
        .fit_with_factories(
            &prepared,
            &ctx(),
            identification_assumptions(&compiled),
            factories.outcome(),
            factories.treatment(),
        )
        .unwrap();
    assert_twin("aipw", &direct, &compiled);
}

#[test]
fn dr_learner_with_declared_factories_equals_the_factory_fit() {
    let fixture = fixture();
    let estimator = DrLearner::new();
    let compiled = run(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let factories = factories();
    let direct = estimator
        .fit_with_factories(
            &prepared,
            &ctx(),
            identification_assumptions(&compiled),
            factories.outcome(),
            factories.treatment(),
        )
        .unwrap();
    assert_twin("dr learner", &direct, &compiled);
}

#[test]
fn causal_forest_with_declared_factories_equals_the_factory_fit() {
    let fixture = fixture();
    let estimator = CausalForest::new().with_n_trees(40);
    let compiled = run(&fixture, estimator.clone());
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let factories = factories();
    let direct = estimator
        .fit_with_factories(
            &prepared,
            &ctx(),
            identification_assumptions(&compiled),
            factories.outcome(),
            factories.treatment(),
        )
        .unwrap();
    assert_twin("causal forest", &direct, &compiled);
}

/// The compiled factory fit carries the nuisance cross-fit of `cross_fit_aipw_with_factories`:
/// on the compiled fold plan the direct cross-fit returns exactly the out-of-fold propensities
/// and the learner provenance of the compiled score table.
#[test]
fn aipw_nuisance_cross_fit_equals_the_compiled_score_table() {
    let fixture = fixture();
    let estimator = DmlAte::new();
    let compiled = run(&fixture, estimator.clone());
    let effect = compiled.estimate().as_effect().unwrap();
    let table = effect.scores.table.as_ref().expect("the AIPW score table");
    let folds: Vec<u16> = effect
        .crossfit
        .fold_assignment
        .as_deref()
        .expect("the recorded fold plan")
        .iter()
        .map(|&fold| u16::try_from(fold).unwrap())
        .collect();
    let prepared = estimator.prepare(&fixture.data, &compiled.estimand, &fixture.query).unwrap();
    let design = DesignView::from_column_major(
        &prepared.design_matrix,
        prepared.nrows,
        prepared.design_ncols,
    )
    .unwrap();
    let factories = factories();
    let (_, _, propensity, record) = cross_fit_aipw_with_factories(
        &IsolatedFactory::new(factories.outcome()),
        &IsolatedFactory::new(factories.treatment()),
        design,
        design,
        &prepared.outcome,
        &prepared.treatment,
        (estimator.folds, folds),
        &ctx(),
        None,
    )
    .unwrap();
    let n = prepared.nrows;
    assert_eq!(table.n_rows, n);
    for (row, e) in propensity.iter().enumerate() {
        assert_eq!(table.propensities[n + row].to_bits(), e.to_bits(), "treated propensity {row}");
        assert_eq!(
            table.propensities[row].to_bits(),
            (1.0 - e).to_bits(),
            "control propensity {row}"
        );
    }
    assert_eq!(record.model_provenance, effect.crossfit.learner_provenance);
}

/// Known truth: `y = 2t + z + 0.4e` with `t | z` logistic in `z`, so the declared linear outcome
/// and logistic propensity factories are correctly specified and the compiled AIPW estimate is
/// within four standard errors of the planted effect 2.
#[test]
fn declared_factories_recover_the_planted_effect() {
    let fixture = fixture();
    let compiled = run(&fixture, DmlAte::new());
    let effect = compiled.estimate().as_effect().unwrap();
    let se = effect.se_analytic();
    assert!(se.is_finite() && se > 0.0, "a finite standard error");
    assert!(
        (effect.ate - 2.0).abs() <= 4.0 * se,
        "estimate {} is more than four standard errors {} from the truth 2",
        effect.ate,
        se
    );
}

/// Two studies that differ only in the factories' declared identity have different inference
/// bindings, and a study without factories keeps the identity it had before the declaration.
#[test]
fn the_declared_identity_extends_the_inference_binding() {
    let fixture = fixture();
    let binding = |identity: Option<&str>| {
        let mut builder = Study::tabular(fixture.data.clone())
            .graph(fixture.dag.clone())
            .query(fixture.query.clone())
            .estimator(DmlAte::new())
            .refute(RefuteSuite::None);
        if let Some(identity) = identity {
            let base = factories();
            let declared = LearnerFactories::new(
                Arc::from(
                    antecedent_learn::resolve_for(
                        LearnerSpec::Linear(LinearSpec::default()),
                        PredictionTask::Regression,
                    )
                    .unwrap(),
                ),
                Arc::from(
                    antecedent_learn::resolve_for(
                        LearnerSpec::Logistic(LogisticSpec::default()),
                        PredictionTask::BinaryProbability,
                    )
                    .unwrap(),
                ),
                identity,
            )
            .unwrap();
            assert_eq!(base.identity(), FACTORY_IDENTITY);
            builder = builder.learner_factories(declared);
        }
        builder.build().unwrap().inspect().unwrap().identities.inference_binding
    };
    let first = binding(Some("first"));
    let second = binding(Some("second"));
    let built_in = binding(None);
    assert_ne!(first, second);
    assert_ne!(first, built_in);
}

#[test]
fn factories_on_an_estimator_that_does_not_cross_fit_are_refused_at_build() {
    let fixture = fixture();
    let refused = Study::tabular(fixture.data.clone())
        .graph(fixture.dag.clone())
        .query(fixture.query.clone())
        .estimator(PropensityWeighting::new())
        .learner_factories(factories())
        .refute(RefuteSuite::None)
        .build()
        .unwrap_err();
    assert_eq!(refused.reason_code(), Some("option_not_applicable"), "{refused}");
}

#[test]
fn factories_with_refuters_are_refused_at_build() {
    let fixture = fixture();
    let refused = Study::tabular(fixture.data.clone())
        .graph(fixture.dag.clone())
        .query(fixture.query.clone())
        .estimator(DmlAte::new())
        .learner_factories(factories())
        .refute(RefuteSuite::PlaceboAndRcc)
        .build()
        .unwrap_err();
    assert_eq!(refused.reason_code(), Some("option_not_applicable"), "{refused}");
}

#[test]
fn factories_with_mismatched_tasks_are_refused() {
    let regression: Arc<dyn LearnerFactory> = Arc::from(
        antecedent_learn::resolve_for(
            LearnerSpec::Linear(LinearSpec::default()),
            PredictionTask::Regression,
        )
        .unwrap(),
    );
    let refused =
        LearnerFactories::new(Arc::clone(&regression), regression, FACTORY_IDENTITY).unwrap_err();
    assert_eq!(refused.reason_code(), Some("invalid_argument"), "{refused}");
}
