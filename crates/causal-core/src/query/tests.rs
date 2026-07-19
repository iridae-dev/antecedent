//! Query unit tests.

use std::sync::Arc;

use crate::ids::VariableId;
use crate::intervention::Intervention;
use crate::value::Value;

use super::*;

#[test]
fn binary_ate_binds_ids_not_names() {
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let q = AverageEffectQuery::binary_ate(t, y);
    q.validate().unwrap();
    assert_eq!(q.treatment, t);
    assert_eq!(q.outcome, y);
    assert_eq!(q.target_population, TargetPopulation::AllObserved);
    match &q.control {
        Intervention::Set { variable, value } => {
            assert_eq!(*variable, t);
            assert_eq!(*value, Value::f64(0.0));
        }
        other => panic!("expected Set, got {other:?}"),
    }
}

#[test]
fn rejects_treatment_equals_outcome() {
    let id = VariableId::from_raw(0);
    let q = AverageEffectQuery::binary_ate(id, id);
    assert!(matches!(q.validate(), Err(QueryError::TreatmentEqualsOutcome { .. })));
}

#[test]
fn causal_query_static_ate_flag() {
    let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    ));
    assert!(q.is_static_ate());
    assert!(!q.is_temporal_effect());
}

#[test]
fn temporal_pulse_query() {
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let q = TemporalEffectQuery::pulse(t, y, -0.03).with_horizon_steps(2);
    q.validate().unwrap();
    assert_eq!(q.policy, TemporalPolicy::Pulse { at: 0 });
    assert_eq!(q.horizon_steps, 2);
    let cq = CausalQuery::temporal_effect(q);
    assert!(cq.is_temporal_effect());
    cq.validate().unwrap();
}

#[test]
fn rejects_inverted_sustained_window() {
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let q = TemporalEffectQuery::pulse(t, y, 1.0).with_policy(TemporalPolicy::sustained(5, 1));
    assert!(matches!(q.validate(), Err(QueryError::InvalidTemporalWindow { .. })));
}

#[test]
fn rejects_zero_horizon() {
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_horizon_steps(0);
    assert!(matches!(q.validate(), Err(QueryError::NonPositiveHorizon)));
}

#[test]
fn counterfactual_and_anomaly_queries() {
    let y = VariableId::from_raw(1);
    let t = VariableId::from_raw(0);
    let cf = CounterfactualQuery::new(y, [Intervention::set(t, Value::f64(1.0))]);
    cf.validate().unwrap();
    assert!(CausalQuery::counterfactual(cf).is_counterfactual());
    let an = AnomalyAttributionQuery::new([y], 100);
    an.validate().unwrap();
    CausalQuery::anomaly_attribution(an).validate().unwrap();
}

#[test]
fn change_attribution_query_validates() {
    let y = VariableId::from_raw(2);
    let q = ChangeAttributionQuery::new(
        y,
        PopulationSelector::TimeRange { start: 0, end: 10 },
        PopulationSelector::TimeRange { start: 10, end: 20 },
    )
    .with_components(AttributionComponents::All)
    .with_allocation(AllocationMethod::Shapley {
        approximation: ShapleyConfig::monte_carlo(100).with_seed(1),
    });
    q.validate().unwrap();
    CausalQuery::change_attribution(q).validate().unwrap();

    let bad = ChangeAttributionQuery::new(
        y,
        PopulationSelector::TimeRange { start: 5, end: 5 },
        PopulationSelector::All,
    );
    assert!(matches!(bad.validate(), Err(QueryError::InvalidPopulationTimeRange { .. })));
}

#[test]
fn shapley_exact_config_rejects_zero_limit() {
    let cfg = ShapleyConfig::exact().with_max_exact_components(0);
    assert!(matches!(cfg.validate(), Err(QueryError::NonPositiveShapleyLimit)));
}

#[test]
fn interventional_distribution_query_validates() {
    let y = VariableId::from_raw(1);
    let t = VariableId::from_raw(0);
    let q = InterventionalDistributionQuery::new(y, [Intervention::set(t, Value::f64(1.0))]);
    q.validate().unwrap();
    let cq = CausalQuery::distribution(q);
    assert!(cq.is_distribution());
    cq.validate().unwrap();

    let empty = InterventionalDistributionQuery {
        outcomes: Arc::from([]),
        interventions: Arc::from([]),
        conditioning: Arc::from([]),
        target_population: TargetPopulation::AllObserved,
    };
    assert!(matches!(empty.validate(), Err(QueryError::EmptyDistributionOutcomes)));

    let overlap = InterventionalDistributionQuery::new(y, [Intervention::set(t, Value::f64(1.0))])
        .with_conditioning([y]);
    assert!(matches!(
        overlap.validate(),
        Err(QueryError::ConditioningOverlapsOutcomeOrIntervention)
    ));
}

#[test]
fn path_specific_query_validates() {
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(2);
    let m = VariableId::from_raw(1);
    let q = PathSpecificEffectQuery::binary(t, y).with_path_nodes([m]);
    q.validate().unwrap();
    let cq = CausalQuery::path_specific(q);
    assert!(cq.is_path_specific());
    cq.validate().unwrap();

    let bad = PathSpecificEffectQuery::binary(t, y).with_max_paths(0);
    assert!(matches!(bad.validate(), Err(QueryError::NonPositivePathLimit)));

    let overlap = PathSpecificEffectQuery::binary(t, y).with_path_nodes([t]);
    assert!(matches!(
        overlap.validate(),
        Err(QueryError::PathNodeOverlapsTreatmentOrOutcome)
    ));
}

#[test]
fn dynamic_policy_and_planned_populations() {
    use crate::ids::{DistributionRef, DynamicRuleId};

    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let rule = DynamicRuleId::from_raw(7);
    let q = TemporalEffectQuery::pulse(t, y, 1.0).with_policy(TemporalPolicy::dynamic(rule, [0, 3]));
    q.validate().unwrap();
    assert_eq!(q.try_treatment_offset(), Ok(0));
    assert_eq!(q.treatment_offset(), 0);
    let empty = TemporalEffectQuery::pulse(t, y, 1.0).with_policy(TemporalPolicy::dynamic(rule, []));
    assert_eq!(empty.try_treatment_offset(), Err(QueryError::DynamicPolicyHasNoTreatmentOffset));
    assert!(matches!(empty.validate(), Err(_)));

    let named = AverageEffectQuery::binary_ate(t, y)
        .with_target_population(TargetPopulation::Predicate(PredicateExpr::named("cohort_a")));
    named.validate().unwrap();

    let empty_name = AverageEffectQuery::binary_ate(t, y)
        .with_target_population(TargetPopulation::Predicate(PredicateExpr::named("")));
    assert!(matches!(empty_name.validate(), Err(QueryError::EmptyPredicateName)));

    let empty_rows = AverageEffectQuery::binary_ate(t, y).with_target_population(
        TargetPopulation::Predicate(PredicateExpr::rows(Arc::<[usize]>::from([]))),
    );
    assert!(matches!(empty_rows.validate(), Err(QueryError::EmptyPopulationRows)));

    let rows = AverageEffectQuery::binary_ate(t, y)
        .with_target_population(TargetPopulation::Predicate(PredicateExpr::rows([0usize, 2])));
    rows.validate().unwrap();

    let dist = AverageEffectQuery::binary_ate(t, y)
        .with_target_population(TargetPopulation::CustomDistribution(DistributionRef::from_raw(3)));
    dist.validate().unwrap();
}
