//! Prepared static derivative-response lifecycle against known linear truth.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study,
};
use antecedent_core::{
    CausalQuery, DerivativeScale, DerivativeWeighting, ExecutionContext, IntervalInterpretation,
    ResponseFunctional as F, ResponseIdentification, ResponseQuery, ResponseUncertainty,
    ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Dag, DenseNodeId};

fn fixture() -> (TabularData, Dag) {
    let a: Vec<_> = (0..800)
        .map(|i| 2.0 + (f64::from(i) * 0.71).sin() + 0.2 * (f64::from(i) * 0.13).cos())
        .collect();
    let b: Vec<_> = (0..800).map(|i| (f64::from(i) * 1.13).cos()).collect();
    let y: Vec<_> = a.iter().zip(&b).map(|(a, b)| 5.0 + 2.0 * a - 0.5 * b).collect();
    let v: Vec<_> = a.iter().zip(&b).map(|(a, b)| 1.0 + 0.25 * a + 1.5 * b).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y", y.as_slice()),
        ("v", v.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for (source, target) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
        graph
            .insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target))
            .unwrap();
    }
    (data, graph)
}

fn ids(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(VariableId::from_raw).collect()
}

#[test]
fn all_frequentist_static_dag_derivative_coordinates_are_sealed_and_dependency_refused() {
    let (data, graph) = fixture();
    let cases = vec![
        (
            "AverageDerivative",
            F::AverageDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                weighting: DerivativeWeighting::Observed,
            },
            vec![2.0],
            EstimatorId::ResponseRieszAde,
        ),
        (
            "PointDerivative/identity",
            F::PointDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                at: 2.0,
                order: 1,
                scale: DerivativeScale::Identity,
            },
            vec![2.0],
            EstimatorId::ResponseKennedyDr,
        ),
        (
            "Elasticity",
            F::PointDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                at: 2.0,
                order: 1,
                scale: DerivativeScale::LogLog,
            },
            vec![4.0 / 9.0],
            EstimatorId::ResponseKennedyDr,
        ),
        (
            "SemiElasticity/treatment-scale",
            F::PointDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                at: 2.0,
                order: 1,
                scale: DerivativeScale::LogTreatment,
            },
            vec![4.0],
            EstimatorId::ResponseKennedyDr,
        ),
        (
            "SemiElasticity/outcome-scale",
            F::PointDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                at: 2.0,
                order: 1,
                scale: DerivativeScale::LogOutcome,
            },
            vec![2.0 / 9.0],
            EstimatorId::ResponseKennedyDr,
        ),
        (
            "DirectionalDerivative",
            F::DirectionalDerivative {
                outcomes: ids(&[2, 3]),
                treatments: ids(&[0, 1]),
                at: Arc::from([2.0, 0.0]),
                direction: Arc::from([1.0, 2.0]),
            },
            vec![1.0, 3.25],
            EstimatorId::ResponseGamDerivative,
        ),
        (
            "ResponseJacobian",
            F::Jacobian {
                outcomes: ids(&[2, 3]),
                treatments: ids(&[0, 1]),
                at: Arc::from([2.0, 0.0]),
                scale: DerivativeScale::Identity,
            },
            vec![2.0, -0.5, 0.25, 1.5],
            EstimatorId::ResponseGamDerivative,
        ),
    ];

    for accepted in [false, true] {
        for (case, functional, expected, expected_estimator) in &cases {
            let query = ResponseQuery::new(functional.clone());
            let base = Study::tabular(data.clone());
            let builder = if accepted {
                base.graph(AcceptedGraph::from(graph.clone()))
            } else {
                base.graph(graph.clone())
            }
            .query(CausalQuery::Response(query.clone()))
            .response_options(ContinuousResponseOptions {
                bandwidth: Some(0.35),
                ..Default::default()
            })
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
            let context = ExecutionContext::for_tests(803);
            let mut prepared =
                builder.prepare(&context).unwrap_or_else(|error| panic!("{case}: {error:?}"));
            let plan = prepared
                .checked_derivative_response_info()
                .unwrap_or_else(|| panic!("{case}: missing retained checked operation"));
            assert_eq!(plan.query, query, "{case}");
            assert_eq!(plan.identifier, IdentifierId::ResponseBackdoor, "{case}");
            assert_eq!(plan.estimator, *expected_estimator, "{case}");
            let expected_cells =
                query.functional.outcome_ids().len() * query.functional.treatment_ids().len();
            assert_eq!(plan.max_derivative_cells, expected_cells, "{case}");
            drop(builder);

            let result = prepared.estimate(&data, &context).unwrap();
            assert!(result.refutations.is_empty(), "{case}");
            let uncertainty = &result.response.as_ref().expect("response payload").uncertainty;
            match (expected.len(), uncertainty) {
                (
                    1,
                    ResponseUncertainty::Scalar {
                        standard_error,
                        interpretation: IntervalInterpretation::Confidence,
                        ..
                    },
                ) => {
                    assert!(standard_error.is_finite() && *standard_error >= 0.0, "{case}");
                    assert_eq!(result.estimate.se_analytic, *standard_error, "{case}");
                    assert!(matches!(
                        *case,
                        "AverageDerivative"
                            | "PointDerivative/identity"
                            | "SemiElasticity/treatment-scale"
                    ));
                }
                (
                    count,
                    ResponseUncertainty::PointwiseBand {
                        lower,
                        upper,
                        interpretation: IntervalInterpretation::Confidence,
                        ..
                    },
                ) if count > 1 => {
                    assert_eq!(lower.len(), count, "{case}");
                    assert_eq!(upper.len(), count, "{case}");
                }
                (1, ResponseUncertainty::None)
                    if matches!(*case, "Elasticity" | "SemiElasticity/outcome-scale") =>
                {
                    assert!(result.estimate.se_analytic.is_nan(), "{case}");
                }
                (count, ResponseUncertainty::None) if count > 1 => {
                    assert!(result.estimate.se_analytic.is_nan(), "{case}");
                }
                (count, other) => panic!(
                    "{case}: expected a confidence scalar or pointwise band for {count} coordinates, got {other:?}"
                ),
            }
            let got = match &result.response.as_ref().expect("response payload").estimate {
                ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => {
                    vec![*value]
                }
                ResponseIdentification::PointIdentified(ResponseValue::Vector(values)) => {
                    values.to_vec()
                }
                ResponseIdentification::PointIdentified(ResponseValue::Jacobian {
                    values, ..
                }) => values.to_vec(),
                other => panic!("{case}: unexpected response identification {other:?}"),
            };
            assert_eq!(got.len(), expected.len(), "{case}");
            for (got, expected) in got.iter().zip(expected) {
                assert!((got - expected).abs() < 0.15, "{case}: {got} != {expected}");
            }

            let refreshed = prepared.refresh(data.clone(), &context).unwrap();
            let refreshed_response = refreshed.response.as_ref().expect("refreshed response");
            let refreshed_values = match &refreshed_response.estimate {
                ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => {
                    vec![*value]
                }
                ResponseIdentification::PointIdentified(ResponseValue::Vector(values)) => {
                    values.to_vec()
                }
                ResponseIdentification::PointIdentified(ResponseValue::Jacobian {
                    values, ..
                }) => values.to_vec(),
                other => panic!("{case}: unexpected refreshed response {other:?}"),
            };
            for (refreshed, original) in refreshed_values.iter().zip(&got) {
                assert!((refreshed - original).abs() < 1e-10, "{case}: refresh changed value");
            }

            let artifact = prepared
                .encode_contracted_result(&refreshed, "checked-derivative-response", &context)
                .unwrap();
            let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
            assert!(
                consumed.acceptance.unresolved.iter().any(|reason| {
                    reason.as_ref() == "dependencies.checked_derivative_response_operation"
                }),
                "{case}"
            );
            assert!(!consumed.acceptance.accepts_as_verified_program(), "{case}");
        }
    }
}

#[test]
fn all_bayesian_static_dag_derivative_coordinates_execute_retained_operations() {
    let (data, graph) = fixture();
    let cases = vec![
        (
            "AverageDerivative",
            F::AverageDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                weighting: DerivativeWeighting::Observed,
            },
            vec![2.0],
            EstimatorId::ResponseRieszAde,
        ),
        (
            "PointDerivative",
            F::PointDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                at: 2.0,
                order: 1,
                scale: DerivativeScale::Identity,
            },
            vec![2.0],
            EstimatorId::ResponseKennedyDr,
        ),
        (
            "Elasticity",
            F::PointDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                at: 2.0,
                order: 1,
                scale: DerivativeScale::LogLog,
            },
            vec![4.0 / 9.0],
            EstimatorId::ResponseKennedyDr,
        ),
        (
            "SemiElasticityTreatment",
            F::PointDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                at: 2.0,
                order: 1,
                scale: DerivativeScale::LogTreatment,
            },
            vec![4.0],
            EstimatorId::ResponseKennedyDr,
        ),
        (
            "SemiElasticityOutcome",
            F::PointDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                at: 2.0,
                order: 1,
                scale: DerivativeScale::LogOutcome,
            },
            vec![2.0 / 9.0],
            EstimatorId::ResponseKennedyDr,
        ),
        (
            "DirectionalDerivative",
            F::DirectionalDerivative {
                outcomes: ids(&[2, 3]),
                treatments: ids(&[0, 1]),
                at: Arc::from([2.0, 0.0]),
                direction: Arc::from([1.0, 2.0]),
            },
            vec![1.0, 3.25],
            EstimatorId::ResponseGamDerivative,
        ),
        (
            "ResponseJacobian",
            F::Jacobian {
                outcomes: ids(&[2, 3]),
                treatments: ids(&[0, 1]),
                at: Arc::from([2.0, 0.0]),
                scale: DerivativeScale::Identity,
            },
            vec![2.0, -0.5, 0.25, 1.5],
            EstimatorId::ResponseGamDerivative,
        ),
    ];

    for accepted in [false, true] {
        for (case, functional, expected, expected_estimator) in &cases {
            let query = ResponseQuery::new(functional.clone());
            let base = Study::tabular(data.clone());
            let builder = if accepted {
                base.graph(AcceptedGraph::from(graph.clone()))
            } else {
                base.graph(graph.clone())
            }
            .query(CausalQuery::Response(query.clone()))
            .response_options(ContinuousResponseOptions {
                bandwidth: Some(0.35),
                ..Default::default()
            })
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
            let context = ExecutionContext::for_tests(804);
            let mut prepared = builder
                .prepare(&context)
                .unwrap_or_else(|error| panic!("{case} accepted={accepted}: {error:?}"));
            let plan = prepared
                .checked_derivative_response_info()
                .unwrap_or_else(|| panic!("{case}: missing retained checked operation"));
            assert_eq!(plan.query, query, "{case}");
            assert_eq!(plan.identifier, IdentifierId::ResponseBackdoor, "{case}");
            assert_eq!(plan.estimator, *expected_estimator, "{case}");
            drop(builder);

            let result = prepared.estimate(&data, &context).unwrap();
            let uncertainty = &result.response.as_ref().unwrap().uncertainty;
            match (expected.len(), uncertainty) {
                (
                    1,
                    ResponseUncertainty::Scalar {
                        standard_error,
                        interpretation: IntervalInterpretation::Credible,
                        ..
                    },
                ) => assert!(standard_error.is_finite() && *standard_error >= 0.0, "{case}"),
                (
                    count,
                    ResponseUncertainty::PointwiseBand {
                        lower,
                        upper,
                        interpretation: IntervalInterpretation::Credible,
                        ..
                    },
                ) if count > 1 => {
                    assert_eq!(lower.len(), count, "{case}");
                    assert_eq!(upper.len(), count, "{case}");
                }
                (count, other) => panic!(
                    "{case}: expected a credible scalar or pointwise band for {count} coordinates, got {other:?}"
                ),
            }
            assert_eq!(result.logical_plan.estimator.as_deref(), Some(expected_estimator.as_str()));
            let values = match &result.response.as_ref().unwrap().estimate {
                ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => {
                    vec![*value]
                }
                ResponseIdentification::PointIdentified(ResponseValue::Vector(values)) => {
                    values.to_vec()
                }
                ResponseIdentification::PointIdentified(ResponseValue::Jacobian {
                    values, ..
                }) => values.to_vec(),
                other => panic!("{case}: unexpected result {other:?}"),
            };
            assert_eq!(values.len(), expected.len(), "{case}");
            for (actual, truth) in values.iter().zip(expected) {
                assert!((actual - truth).abs() < 0.15, "{case}: {actual} != {truth}");
            }

            let refreshed = prepared.refresh(data.clone(), &context).unwrap();
            let artifact = prepared
                .encode_contracted_result(&refreshed, "checked-bayesian-derivative", &context)
                .unwrap();
            let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
            assert!(consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.checked_derivative_response_operation"
            }));
            assert!(!consumed.acceptance.accepts_as_verified_program());
        }
    }
}
