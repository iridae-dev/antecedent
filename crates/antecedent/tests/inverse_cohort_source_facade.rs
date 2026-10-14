//! A fitted effect model's cohort contrast as the forward source of an inverse query, through
//! the facade: actual DR-learner producer, cohort adapter, `inverse_sources::cohort_source`,
//! evaluation, sealing and a consumer's replay.
//!
//! Oracle. The fixture is noiseless: `Y(0) = 1 + z`, `Y(1) = effect + 1 + 1.5 z`, every
//! (treatment, z) cell 40 times, so the linear final stage recovers `tau(z) = effect + 0.5 z`
//! exactly whatever the nuisance fits are. For `effect = 2` the cohort `z = (-1, 0.5, 1)` with
//! weights `(0.2, 0.3, 0.5)` has weighted mean contrast
//! `0.2 * 1.5 + 0.3 * 2.25 + 0.5 * 2.5 = 2.225`. The utilities `2 x - 1` and `4 x - 1.5` are
//! `3.45` and `7.4`; with the target "mean at least 5" only the second meets it. For
//! `effect = 3` the mean contrast is `3 + 0.5 * 0.45 = 3.225` and the second utility `11.4`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::cohort_decision::{
    CohortBinding, CohortColumn, CohortDecisionSource, CohortOptions, CovariateShiftDeclaration,
    FittedEffectModel,
};
use antecedent::analysis::inverse_query;
use antecedent::analysis::inverse_query::{
    Comparison, GridScope, InverseConstraint, InverseQuery, SelectionRule,
};
use antecedent::analysis::inverse_sources::{
    ForwardSourceKind, ForwardSourceRefusal, InverseDemand, TransportedInputPolicy,
    TransportedRoute, cohort_source, evaluate, seal,
};
use antecedent::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_estimate::DrLearner;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_learn::{LearnerSpec, LinearSpec};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(97)
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() < tolerance, "{what}: actual={actual}, expected={expected}");
}

fn columns(effect: f64) -> Vec<(String, Vec<f64>)> {
    let (mut t, mut z, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..40 {
        for treatment in [0.0, 1.0] {
            for confounder in [-1.0, 1.0] {
                t.push(treatment);
                z.push(confounder);
                y.push(1.0 + confounder + treatment * (effect + 0.5 * confounder));
            }
        }
    }
    vec![("t".into(), t), ("y".into(), y), ("z".into(), z)]
}

/// The exported contracted result of a DR-learner fit.
fn fit(effect: f64) -> Vec<u8> {
    let columns = columns(effect);
    let borrowed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .estimator(
            DrLearner::new().with_outcome(LearnerSpec::Linear(LinearSpec::default())).with_folds(4),
        )
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx()).unwrap();
    let result = prepared.estimate(&data, &ctx()).unwrap();
    prepared.encode_contracted_result(&result, "dr-fit", &ctx()).unwrap()
}

fn cohort(population: &str) -> CohortBinding {
    CohortBinding::new(
        "clinic-2026-q1",
        population,
        vec!["patient-0".into(), "patient-1".into(), "patient-2".into()],
        vec!["z".into()],
        vec![CohortColumn::Numeric(vec![-1.0, 0.5, 1.0])],
        vec![0.2, 0.3, 0.5],
    )
    .unwrap()
}

fn source_of(
    effect: f64,
    population: &str,
    shift: Option<CovariateShiftDeclaration>,
) -> CohortDecisionSource {
    FittedEffectModel::load(&fit(effect))
        .unwrap()
        .decision_source(
            &cohort(population),
            "mmHg",
            &CohortOptions { shift, expected_contrast_regime: None },
            &ctx(),
        )
        .unwrap()
}

fn query_on(source: &CohortDecisionSource, constraints: Vec<InverseConstraint>) -> InverseQuery {
    let coordinate = source.quantity().clone();
    let action = |id: &str, slope: f64, offset: f64| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![coordinate.clone()],
        utility: UtilityExpr::sum(
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(slope)),
            UtilityExpr::Const(offset),
        ),
    };
    InverseQuery {
        contract: DecisionContract {
            actions: vec![action("halve", 2.0, -1.0), action("treat", 4.0, -1.5)],
            utility_units: "utility".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: coordinate.population_id.clone(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        },
        grid_order: vec!["halve".into(), "treat".into()],
        grid_scope: GridScope::FiniteEnumeration,
        constraints,
        selection: SelectionRule::FirstInGridOrder,
        tolerance: 1e-12,
        max_evaluations: None,
    }
}

fn mean_at_least(target: f64) -> Vec<InverseConstraint> {
    vec![InverseConstraint::TargetMean { target, comparison: Comparison::AtLeast }]
}

#[test]
fn a_native_cohort_contrast_is_a_point_source_matching_the_hand_calculation() {
    let source = source_of(2.0, "target", None);
    close(source.mean_contrast(), 2.225, 1e-7, "cohort mean contrast");

    let forward = cohort_source(&source, TransportedInputPolicy::NativeTargetOnly).unwrap();
    let origin = forward.origin();
    assert_eq!(origin.kind, ForwardSourceKind::CohortContrast);
    assert_eq!(origin.source_identity, source.identity());
    assert_eq!(origin.support, "supported");
    assert!(origin.transport.is_none());

    let result =
        evaluate(&query_on(&source, mean_at_least(5.0)), &forward, InverseDemand::default())
            .unwrap();
    close(result.result.actions[0].point_values[0].value.unwrap(), 3.45, 1e-7, "halve");
    close(result.result.actions[1].point_values[0].value.unwrap(), 7.4, 1e-7, "treat");
    assert_eq!(result.result.selected.as_deref(), Some("treat"));
    assert_eq!(result.origin.source_identity, source.identity());
}

#[test]
fn the_cohort_source_is_refused_what_a_point_cannot_answer() {
    let source = source_of(2.0, "target", None);
    let forward = cohort_source(&source, TransportedInputPolicy::NativeTargetOnly).unwrap();

    // A quantile of a point is not a statement the source can make.
    let quantile = query_on(
        &source,
        vec![InverseConstraint::TargetQuantile {
            p: 0.5,
            target: 5.0,
            comparison: Comparison::AtLeast,
        }],
    );
    let error = evaluate(&quantile, &forward, InverseDemand::default()).unwrap_err();
    assert_eq!(
        error,
        ForwardSourceRefusal::PointOnlyLawRequested {
            kind: ForwardSourceKind::CohortContrast,
            needed: "quantile",
        }
    );
    assert!(!error.detail().is_empty());

    // An interval region is not licensed for any source.
    let error = evaluate(
        &query_on(&source, mean_at_least(5.0)),
        &forward,
        InverseDemand { interval_region: true },
    )
    .unwrap_err();
    assert_eq!(error, ForwardSourceRefusal::IntervalNotLicensed(ForwardSourceKind::CohortContrast));
    assert_eq!(error.code(), "cell_not_licensed");
}

#[test]
fn a_shifted_cohort_is_read_only_under_a_policy_that_accepts_transported_claims() {
    let shift = CovariateShiftDeclaration::new("target", "clinic_b", &["z"]);
    let source = source_of(2.0, "clinic_b", Some(shift));

    let refused = cohort_source(&source, TransportedInputPolicy::NativeTargetOnly).unwrap_err();
    assert_eq!(refused, ForwardSourceRefusal::TransportedNotAccepted);
    assert_eq!(refused.code(), "cell_not_licensed");

    let forward = cohort_source(&source, TransportedInputPolicy::AcceptsTransported).unwrap();
    assert_eq!(forward.origin().kind, ForwardSourceKind::TransportedResponse);
    assert_eq!(forward.origin().support, "extrapolative");
    let receipt = forward.origin().transport.as_ref().expect("a transport receipt");
    assert_eq!(receipt.route, TransportedRoute::ShiftedCohort);
    assert_eq!(receipt.source_populations, vec!["target".to_owned()]);
    assert_eq!(receipt.target_population, "clinic_b");
    let result =
        evaluate(&query_on(&source, mean_at_least(5.0)), &forward, InverseDemand::default())
            .unwrap();
    close(result.result.actions[1].point_values[0].value.unwrap(), 7.4, 1e-7, "treat");
}

#[test]
fn a_sealed_cohort_query_replays_under_its_identity_and_follows_the_fitted_model() {
    let source = source_of(2.0, "target", None);
    let forward = cohort_source(&source, TransportedInputPolicy::NativeTargetOnly).unwrap();
    let artifact =
        seal(query_on(&source, mean_at_least(5.0)), &forward, InverseDemand::default()).unwrap();
    let bytes = inverse_query::export(&artifact, "cohort-inverse").unwrap();
    let consumed = inverse_query::consume(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.identity(), artifact.identity());
    let origin = consumed.forward_origin().expect("the origin travels with the artifact");
    assert_eq!(origin.source_identity, source.identity());
    assert_eq!(origin.kind, ForwardSourceKind::CohortContrast);

    // A different fitted effect at unchanged shape is another source, another claim and
    // another value: tau(z) = 3 + 0.5 z gives mean contrast 3.225 and utility 11.4.
    let other = source_of(3.0, "target", None);
    assert_ne!(other.identity(), source.identity());
    close(other.mean_contrast(), 3.225, 1e-7, "other mean contrast");
    let other_forward = cohort_source(&other, TransportedInputPolicy::NativeTargetOnly).unwrap();
    assert_ne!(other_forward.origin().claim_digest, forward.origin().claim_digest);
    let result =
        evaluate(&query_on(&other, mean_at_least(5.0)), &other_forward, InverseDemand::default())
            .unwrap();
    close(result.result.actions[1].point_values[0].value.unwrap(), 11.4, 1e-7, "treat");

    // The retained identity of the first artifact refuses the second one's bytes.
    let other_artifact =
        seal(query_on(&other, mean_at_least(5.0)), &other_forward, InverseDemand::default())
            .unwrap();
    let other_bytes = inverse_query::export(&other_artifact, "cohort-inverse").unwrap();
    assert!(inverse_query::consume(&other_bytes, Some(artifact.identity())).is_err());
}
