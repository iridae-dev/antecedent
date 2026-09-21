//! Target-population license: which declared populations a study estimates.
//!
//! Only an `AverageEffect` estimates a population other than `AllObserved`.
//! Every other population-scoped query kind refuses a declared population at
//! build with the `population_not_estimable` reason code, and
//! `linear.adjustment.ate` refuses one on every graph class instead of
//! reporting its coefficient under the target's name.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{CausalError, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ContinuousDomain, ExecutionContext,
    GridSpec, Intervention, InterventionalDistributionQuery, MediationContrast, MediationQuery,
    PathSpecificEffectQuery, PredicateExpr, ResponseFunctional, ResponseQuery, TargetPopulation,
    TemporalEffectQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Cpdag, Dag, DenseNodeId};

const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);
const Z: VariableId = VariableId::from_raw(2);

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

/// Binary treatment confounded by `z`, with an effect that grows in `z`, so
/// the treated and the all-observed populations have different effects.
fn data(n: usize) -> TabularData {
    let z: Vec<f64> = (0..n).map(|i| (i as f64 / 7.0).sin() * 1.5).collect();
    let t: Vec<f64> =
        (0..n).map(|i| if z[i] + (i as f64 / 3.0).cos() > 0.0 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> =
        (0..n).map(|i| t[i] * (1.0 + 2.0 * z[i]) + z[i] + (i as f64 / 5.0).sin() * 0.1).collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn dag() -> Dag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(d(2), d(0)).unwrap();
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g
}

fn reason(err: &CausalError) -> Option<String> {
    let message = err.to_string();
    antecedent_core::reason_code::split_prefix(&message).map(|(code, _)| code.to_string())
}

/// Every population-scoped kind other than `AverageEffect`, at a given population.
fn scoped_queries(population: &TargetPopulation) -> Vec<(&'static str, CausalQuery)> {
    let mut conditional = AverageEffectQuery::binary_ate(T, Y).with_effect_modifiers([Z]);
    conditional.target_population = population.clone();
    let mut mediation = MediationQuery::binary(T, Y, [Z], MediationContrast::Mediated);
    mediation.target_population = population.clone();
    let curve = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: Y,
        treatment: ContinuousDomain::new(T, GridSpec::Values(Arc::from(vec![0.0, 1.0]))),
    })
    .with_target_population(population.clone());
    vec![
        (
            "conditional",
            CausalQuery::ConditionalEffect(ConditionalEffectQuery::try_new(conditional).unwrap()),
        ),
        ("mediation", CausalQuery::Mediation(mediation)),
        (
            "path_specific",
            CausalQuery::PathSpecific(
                PathSpecificEffectQuery::binary(T, Y).with_target_population(population.clone()),
            ),
        ),
        (
            "distribution",
            CausalQuery::Distribution(
                InterventionalDistributionQuery::new(Y, [Intervention::set(T, Value::f64(1.0))])
                    .with_target_population(population.clone()),
            ),
        ),
        ("response", CausalQuery::Response(curve)),
        (
            "temporal_effect",
            CausalQuery::TemporalEffect(
                TemporalEffectQuery::pulse(T, Y, 1.0).with_target_population(population.clone()),
            ),
        ),
    ]
}

#[test]
fn non_average_kinds_refuse_a_declared_population_by_code() {
    let populations = [
        TargetPopulation::Treated,
        TargetPopulation::Untreated,
        TargetPopulation::Predicate(PredicateExpr::rows(vec![0, 2, 4])),
        TargetPopulation::CustomDistribution(antecedent_core::DistributionRef::from_raw(1)),
    ];
    for population in &populations {
        for (kind, query) in scoped_queries(population) {
            let err = Study::tabular(data(60))
                .graph(dag())
                .query(query)
                .refute(RefuteSuite::None)
                .build()
                .expect_err(kind);
            assert_eq!(
                reason(&err).as_deref(),
                Some("population_not_estimable"),
                "{kind} at {population:?}: {err}"
            );
        }
    }
    // The all-observed population is not a declaration the license refuses.
    for (kind, query) in scoped_queries(&TargetPopulation::AllObserved) {
        let built = Study::tabular(data(60)).graph(dag()).query(query).build();
        if let Err(err) = built {
            let message = format!("{kind}: {err}");
            assert_ne!(reason(&err).as_deref(), Some("population_not_estimable"), "{message}");
        }
    }
}

#[test]
fn average_effect_keeps_its_licensed_populations() {
    let ctx = ExecutionContext::for_tests(3);
    let run = |population: TargetPopulation| {
        Study::tabular(data(400))
            .graph(dag())
            .query(AverageEffectQuery::binary_ate(T, Y).with_target_population(population))
            .estimator(antecedent::EstimatorId::Aipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap()
            .estimate
            .ate
    };
    let (all, treated) = (run(TargetPopulation::AllObserved), run(TargetPopulation::Treated));
    assert!((all - treated).abs() > 0.1, "ATE {all} and ATT {treated} must differ here");
}

#[test]
fn linear_adjustment_refuses_a_population_on_a_cpdag() {
    let mut cpdag = Cpdag::with_variables(3);
    cpdag.insert_directed(d(2), d(1)).unwrap();
    cpdag.insert_directed(d(0), d(1)).unwrap();
    cpdag.insert_undirected(d(2), d(0)).unwrap();
    let ctx = ExecutionContext::for_tests(5);
    let study = |population: TargetPopulation| {
        Study::tabular(data(400))
            .graph(cpdag.clone())
            .query(AverageEffectQuery::binary_ate(T, Y).with_target_population(population))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .and_then(|study| study.run(&ctx))
    };
    assert!(study(TargetPopulation::AllObserved).is_ok());
    for population in [
        TargetPopulation::Treated,
        TargetPopulation::Untreated,
        TargetPopulation::Predicate(PredicateExpr::rows(vec![0, 2, 4, 6])),
    ] {
        let err = study(population.clone()).expect_err("linear adjustment ATT on a CPDAG");
        assert_eq!(reason(&err).as_deref(), Some("population_not_estimable"), "{population:?}");
    }
}
