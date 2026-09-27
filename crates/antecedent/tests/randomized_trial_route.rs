//! Focused coverage for graphless randomized ITT routing.

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{RandomizedEffectQuery, Study};
use antecedent_core::VariableId;
use antecedent_data::TabularData;

#[test]
fn graphless_bernoulli_itt_runs_and_retains_design_units() {
    let outcomes = [3.0, 0.0, 4.0, 1.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let query = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        [true, false, true, false],
        [0.5; 4],
        ["a", "b", "c", "d"].map(Arc::<str>::from),
        ["r0", "r1", "r2", "r3"].map(Arc::<str>::from),
        ("control", "treated"),
    );
    let study = Study::tabular(data.clone()).query(query.clone()).build().unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let result = study.run(&ctx).unwrap();

    let estimate = result.randomized_effect.as_ref().unwrap();
    assert_eq!(estimate.effect, 3.0);
    assert_eq!(estimate.variance_upper_bound, 3.25);
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert_eq!(
        estimate.assignment_units.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        ["a", "b", "c", "d"]
    );
    assert_eq!(
        estimate.outcome_units.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        ["r0", "r1", "r2", "r3"]
    );
    assert_eq!(result.treatment, None);
    assert_eq!(result.support_status, None);

    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
    let refreshed = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(refreshed.randomized_effect.as_ref().unwrap().effect, 3.0);
    assert_eq!(refreshed.treatment, None);
}

#[test]
fn fixed_cuped_bernoulli_itt_adjusts_in_native_retained_execution() {
    let baseline = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    let outcomes = [2.0, 0.0, 6.0, 4.0, 10.0, 8.0, 14.0, 12.0];
    let data = TabularData::from_f64_columns([
        ("outcome", &outcomes[..]), ("baseline", &baseline[..]),
    ]).unwrap();
    let query = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        [true, false, true, false, true, false, true, false],
        [0.5; 8],
        (0..8).map(|i| Arc::<str>::from(format!("u{i}"))).collect::<Vec<_>>(),
        (0..8).map(|i| Arc::<str>::from(format!("y{i}"))).collect::<Vec<_>>(),
        ("control", "treated"),
    ).with_fixed_cuped(VariableId::from_raw(1), 4.0);
    let result = Study::tabular(data).query(query).build().unwrap()
        .run(&ExecutionContext::for_tests(47)).unwrap();
    let adjusted = result.randomized_effect.as_ref().unwrap();
    assert_eq!(adjusted.effect, 2.0);
    assert_eq!(adjusted.variance_upper_bound, 1.0);
    assert_eq!(adjusted.uncertainty.as_ref(), "bernoulli_fixed_cuped_ht_conservative_variance_no_interval");
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "fixed_pre_assignment_cuped"
    )));
}

#[test]
fn graphless_complete_and_stratified_itt_use_neyman_variance() {
    let context = ExecutionContext::for_tests(4);
    let outcomes = [2.0, 0.0, 4.0, 2.0, 6.0, 4.0, 8.0, 6.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let units = (0..8).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect::<Vec<_>>();
    let assignment = [true, false, true, false, true, false, true, false];

    let complete = RandomizedEffectQuery::with_design(
        antecedent_core::RandomizationDesign::Complete { treated_units: 4 },
        VariableId::from_raw(0),
        assignment,
        [0.5; 8],
        units.clone(),
        units.clone(),
        ("control", "treated"),
    );
    let complete_result =
        Study::tabular(data.clone()).query(complete).build().unwrap().run(&context).unwrap();
    let complete_effect = complete_result.randomized_effect.unwrap();
    assert_eq!(complete_effect.effect, 2.0);
    assert!((complete_effect.variance_upper_bound - 10.0 / 3.0).abs() < 1e-12);
    assert_eq!(complete_effect.assignment_design.as_ref(), "complete");
    assert_eq!(complete_effect.control_units, 4);
    assert_eq!(complete_effect.treatment_units, 4);
    assert_eq!(
        complete_effect.uncertainty.as_ref(),
        "complete_neyman_variance_upper_bound_no_interval"
    );

    let blocks = ["north", "north", "north", "north", "south", "south", "south", "south"]
        .map(Arc::<str>::from);
    let stratified = RandomizedEffectQuery::with_design(
        antecedent_core::RandomizationDesign::Stratified {
            blocks: Arc::from(blocks),
            treated_per_row: Arc::from([2; 8]),
        },
        VariableId::from_raw(0),
        assignment,
        [0.5; 8],
        units.clone(),
        units,
        ("control", "treated"),
    );
    let stratified_study = Study::tabular(data.clone()).query(stratified).build().unwrap();
    let prepared = stratified_study.prepare(&context).unwrap();
    let stratified_effect = prepared.estimate(&data, &context).unwrap().randomized_effect.unwrap();
    assert_eq!(stratified_effect.effect, 2.0);
    assert!((stratified_effect.variance_upper_bound - 1.0).abs() < 1e-12);
    assert_eq!(stratified_effect.assignment_design.as_ref(), "stratified");
    assert_eq!(stratified_effect.blocks.len(), 8);
    assert_eq!(
        stratified_effect.uncertainty.as_ref(),
        "stratified_neyman_variance_upper_bound_no_interval"
    );
}

#[test]
fn cluster_randomized_itt_uses_cluster_totals_and_assignment_units() {
    let outcomes = [6.0, 8.0, 4.0, 1.0, 3.0, 2.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let query = RandomizedEffectQuery::with_design(
        antecedent_core::RandomizationDesign::Cluster { treated_clusters: 2 },
        VariableId::from_raw(0),
        [true, true, true, false, false, false],
        [0.5; 6],
        ["a", "a", "b", "c", "c", "d"].map(Arc::<str>::from),
        ["r0", "r1", "r2", "r3", "r4", "r5"].map(Arc::<str>::from),
        ("control", "treated"),
    );
    let context = ExecutionContext::for_tests(37);
    let result =
        Study::tabular(data.clone()).query(query.clone()).build().unwrap().run(&context).unwrap();
    let estimate = result.randomized_effect.as_ref().unwrap();
    assert_eq!(estimate.effect, 4.0);
    assert!((estimate.variance_upper_bound - 104.0 / 9.0).abs() < 1e-12);
    assert_eq!(estimate.control_units, 2);
    assert_eq!(estimate.treatment_units, 2);
    assert_eq!(estimate.assignment_design.as_ref(), "cluster");
    assert_eq!(estimate.uncertainty.as_ref(), "cluster_neyman_variance_upper_bound_no_interval");
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "no_between_cluster_interference"
    )));
    assert_eq!(result.support_status, None);
    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&context).unwrap();
    assert_eq!(
        prepared.estimate(&data, &context).unwrap().randomized_effect,
        result.randomized_effect
    );
}

#[test]
fn cluster_randomized_itt_refuses_inconsistent_assignment_or_sparse_arms() {
    let outcomes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let make = |assignment: Vec<bool>, treated_clusters: usize| {
        RandomizedEffectQuery::with_design(
            antecedent_core::RandomizationDesign::Cluster { treated_clusters },
            VariableId::from_raw(0),
            assignment,
            [0.5; 6],
            ["a", "a", "b", "c", "c", "d"].map(Arc::<str>::from),
            ["r0", "r1", "r2", "r3", "r4", "r5"].map(Arc::<str>::from),
            ("control", "treated"),
        )
    };
    let context = ExecutionContext::for_tests(38);
    assert!(
        Study::tabular(data.clone())
            .query(make(vec![true, false, true, false, false, false], 2))
            .build()
            .unwrap()
            .run(&context)
            .is_err()
    );
    assert!(
        Study::tabular(data)
            .query(make(vec![true, true, false, false, false, false], 1))
            .build()
            .unwrap()
            .run(&context)
            .is_err()
    );
}
