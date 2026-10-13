//! A cohort-mean contrast source feeds the decision boundary as one affine mean and
//! nothing else: hand-calculated expected utilities, support gating, trust and the
//! refusals of every law-valued request.
//!
//! Oracle. The retained model is `tau(z) = 2 + 0.5 z`; the cohort `z = (-1, 0.5, 1)` with
//! weights `(0.2, 0.3, 0.5)` has weighted mean contrast `2.225`. The utility of `treat` is
//! `4 x - 1.5`, so its expected utility is `4 * 2.225 - 1.5 = 7.4`; `wait` is `0 * x = 0`.

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{ExecutionContext, SupportStatus};
use antecedent_design::cohort_decision_input::{
    cohort_decision_input, cohort_forward_claim, cohort_mean_source, cohort_trust_evidence,
};
use antecedent_design::composition_boundary::{
    ActionStatus, BoundaryError, CalibrationStatus, InputSource, ProviderKind, SupportPolicy,
    SupportedVerdict, TrustEvidence, TrustRequirement, UnevaluatedReason, UnsupportedActionPolicy,
    evaluate_functional_on_input, evaluate_with_support,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, DecisionFunctional,
    StructuralPolicy, Tail, UtilityExpr,
};
use antecedent_design::decision_eval::Verdict;
use antecedent_io_core::cohort_decision::{
    CATE_FUNCTIONAL_ID, CohortBinding, CohortColumn, CohortDecisionSource, CohortOptions,
    CovariateShiftDeclaration, RetainedEffectModel,
};
use antecedent_io_core::compact_export::{
    CompactExport, ExportInput, ExportSpec, InputSupport, TermSpec,
};
use antecedent_io_core::quantity_wire::ScientificQuantityWire;

fn wire(
    id: &str,
    role: &str,
    units: &str,
    functional: &str,
    regime: &str,
) -> ScientificQuantityWire {
    ScientificQuantityWire {
        version: 1,
        variable_id: id.into(),
        variable_name: id.into(),
        role: role.into(),
        units: units.into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn model() -> RetainedEffectModel {
    let export = CompactExport::build(ExportSpec {
        response: wire("y", "outcome", "mmHg", CATE_FUNCTIONAL_ID, "contrast:do(t=1)-do(t=0)"),
        inputs: vec![ExportInput {
            quantity: wire("z", "covariate", "1", "mean", "observed"),
            support: InputSupport::Range { lo: -1.0, hi: 1.0 },
        }],
        terms: vec![TermSpec::Intercept, TermSpec::Linear { quantity: "z".into() }],
        coefficients: vec![2.0, 0.5],
        covariance: vec![0.04, -0.01, -0.01, 0.09],
        mask: vec![],
    })
    .unwrap();
    RetainedEffectModel::from_compact_export(&export).unwrap()
}

fn cohort(population: &str) -> CohortBinding {
    CohortBinding::new(
        "clinic-2026-q1",
        population,
        vec!["r0".into(), "r1".into(), "r2".into()],
        vec!["z".into()],
        vec![CohortColumn::Numeric(vec![-1.0, 0.5, 1.0])],
        vec![0.2, 0.3, 0.5],
    )
    .unwrap()
}

fn source() -> CohortDecisionSource {
    CohortDecisionSource::build(
        &model(),
        &cohort("target"),
        "mmHg",
        &CohortOptions::default(),
        &ExecutionContext::for_tests(3),
    )
    .unwrap()
}

fn contract(
    coordinate: &ScientificQuantity,
    treat: UtilityExpr,
    population: &str,
) -> DecisionContract {
    let action = |id: &str, utility: UtilityExpr| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![coordinate.clone()],
        utility,
    };
    DecisionContract {
        actions: vec![
            action("treat", treat),
            action("wait", UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0))),
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: population.into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

/// `4 x - 1.5`.
fn affine() -> UtilityExpr {
    UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(4.0)),
        UtilityExpr::Const(1.5),
    )
}

#[test]
fn the_affine_expected_utility_is_the_utility_of_the_cohort_mean() {
    let source = source();
    let input = cohort_decision_input("cohort", &source, TrustRequirement::Unrestricted).unwrap();
    let contract = contract(source.quantity(), affine(), "target");
    let decision =
        evaluate_with_support(&contract, &[input], &SupportPolicy::require_all()).unwrap();
    let treat = &decision.outcomes[0];
    assert_eq!(treat.id, "treat");
    assert!((treat.expected_utility - (4.0 * 2.225 - 1.5)).abs() < 1e-12);
    assert!((decision.outcomes[1].expected_utility).abs() < 1e-12);
    // No utility standard error, regret or perfect-information value is inherited.
    for outcome in &decision.outcomes {
        assert_eq!(outcome.standard_error, None);
        assert_eq!(outcome.expected_regret, None);
    }
    assert_eq!(decision.evpi, None);
    assert_eq!(
        decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into()))
    );
    for disposition in &decision.dispositions {
        assert_eq!(disposition.status, ActionStatus::Evaluated);
    }
}

#[test]
fn the_mean_source_names_the_cohort_and_the_parent() {
    let source = source();
    let mean = cohort_mean_source(&source);
    assert_eq!(mean.coordinates.len(), 1);
    assert_eq!(mean.coordinates[0].functional_id, "mean_difference");
    assert_eq!(mean.snapshot_id, cohort("target").digest());
    assert_eq!(mean.causal_contract_id, source.model().parent().claim_id);
    assert!((mean.means[0] - 2.225).abs() < 1e-12);
}

#[test]
fn a_declared_model_is_attested_and_never_native() {
    let source = source();
    assert!(matches!(
        cohort_trust_evidence(&source),
        TrustEvidence::External(antecedent_contract::ExternalTrustState::ExternallyAttested { .. })
    ));
    let input = cohort_decision_input("cohort", &source, TrustRequirement::Unrestricted).unwrap();
    assert_eq!(input.provenance().provider_kind, ProviderKind::ExternalAttested);
    assert_eq!(input.provenance().calibration, CalibrationStatus::PointOnly);
    assert!(matches!(input.source(), InputSource::Mean(_)));
    let refused = cohort_decision_input("cohort", &source, TrustRequirement::Native);
    assert!(matches!(
        refused,
        Err(BoundaryError::NativeRequired { .. } | BoundaryError::MetadataOnlyNativeClaim { .. })
    ));
}

#[test]
fn law_valued_requests_stay_refused_on_the_mean() {
    let source = source();
    let input = cohort_decision_input("cohort", &source, TrustRequirement::Unrestricted).unwrap();
    let affine_contract = contract(source.quantity(), affine(), "target");
    for functional in [
        DecisionFunctional::Variance,
        DecisionFunctional::Probability { threshold: 2.0, tail: Tail::Upper },
        DecisionFunctional::Quantile { p: 0.5 },
        DecisionFunctional::TailExpectation { p: 0.9, tail: Tail::Upper },
    ] {
        let error = evaluate_functional_on_input(
            &affine_contract,
            "treat",
            functional,
            &input,
            SupportStatus::Supported,
        )
        .unwrap_err();
        assert!(matches!(error, BoundaryError::MeanIsNotADistribution { .. }), "{functional:?}");
    }
    // A nonlinear utility is unevaluated, never approximated from the mean.
    let squared = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0));
    let nonlinear = contract(source.quantity(), squared, "target");
    let policy = SupportPolicy {
        unsupported: UnsupportedActionPolicy::CompareSupported,
        weakest_support: SupportStatus::Supported,
    };
    let decision =
        evaluate_with_support(&nonlinear, std::slice::from_ref(&input), &policy).unwrap();
    assert_eq!(
        decision.dispositions[0].status,
        ActionStatus::Unevaluated { reason: UnevaluatedReason::NonAffineUtilityNeedsJointLaw }
    );
    // A probability criterion is a law question.
    let mut probability = contract(source.quantity(), affine(), "target");
    probability.criterion = DecisionCriterion::ThresholdProbability { threshold: 1.0 };
    assert!(
        evaluate_with_support(
            &probability,
            std::slice::from_ref(&input),
            &SupportPolicy::require_all()
        )
        .is_err()
    );
}

#[test]
fn a_contract_must_name_the_exact_cohort_contrast_and_units() {
    let source = source();
    let input = cohort_decision_input("cohort", &source, TrustRequirement::Unrestricted).unwrap();
    let policy = SupportPolicy::compare_supported();
    let evaluate = |mutate: &dyn Fn(&mut ScientificQuantity)| {
        let mut coordinate = source.quantity().clone();
        mutate(&mut coordinate);
        evaluate_with_support(
            &contract(&coordinate, affine(), "target"),
            std::slice::from_ref(&input),
            &policy,
        )
        .unwrap()
        .verdict
    };
    assert_eq!(
        evaluate(&|_| {}),
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into()))
    );
    // Another cohort snapshot, another contrast, other units and an outcome-law functional
    // are different coordinates: nothing is evaluated.
    for mutate in [
        (&|q: &mut ScientificQuantity| q.conditioning[0].value_id = "0".repeat(64))
            as &dyn Fn(&mut ScientificQuantity),
        &|q| q.regime_id = "contrast:do(t=2)-do(t=0)".into(),
        &|q| q.units = "g/dL".into(),
        &|q| q.functional_id = "outcome".into(),
        &|q| q.role = QuantityRole::Treatment,
    ] {
        assert_eq!(evaluate(mutate), SupportedVerdict::NoSupportedAction);
    }
}

#[test]
fn a_declared_shift_is_extrapolative_and_needs_an_explicit_weaker_policy() {
    let options = CohortOptions {
        shift: Some(CovariateShiftDeclaration::new("target", "clinic_b", &["z"])),
        expected_contrast_regime: None,
    };
    let shifted = CohortDecisionSource::build(
        &model(),
        &cohort("clinic_b"),
        "mmHg",
        &options,
        &ExecutionContext::for_tests(3),
    )
    .unwrap();
    assert_eq!(shifted.support_status(), SupportStatus::Extrapolative);
    let input = cohort_decision_input("cohort", &shifted, TrustRequirement::Unrestricted).unwrap();
    let contract = contract(shifted.quantity(), affine(), "clinic_b");
    let default = evaluate_with_support(
        &contract,
        std::slice::from_ref(&input),
        &SupportPolicy::compare_supported(),
    )
    .unwrap();
    assert_eq!(default.verdict, SupportedVerdict::NoSupportedAction);
    let weaker = SupportPolicy {
        unsupported: UnsupportedActionPolicy::RequireAllActions,
        weakest_support: SupportStatus::Extrapolative,
    };
    let accepted = evaluate_with_support(&contract, &[input], &weaker).unwrap();
    assert!((accepted.outcomes[0].expected_utility - 7.4).abs() < 1e-12);
}

#[test]
fn the_cohort_mean_is_a_licensed_forward_source_of_an_inverse_query() {
    use antecedent_design::inverse_query::{
        Comparison, FeasibilityStatus, ForwardEvidence, GridScope, InverseConstraint, InverseQuery,
        SelectionOutcome, SelectionRule, evaluate_inverse_query,
    };
    let source = source();
    let coordinate = source.quantity();
    let action = |id: &str, slope: f64, offset: f64| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![coordinate.clone()],
        utility: UtilityExpr::sum(
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(slope)),
            UtilityExpr::Const(offset),
        ),
    };
    // Utilities at the cohort mean 2.225: halve = 2 * 2.225 - 1 = 3.45, treat = 7.4.
    let contract = DecisionContract {
        actions: vec![action("halve", 2.0, -1.0), action("treat", 4.0, -1.5)],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    };
    let query = InverseQuery {
        contract,
        grid_order: vec!["halve".into(), "treat".into()],
        grid_scope: GridScope::FiniteEnumeration,
        constraints: vec![InverseConstraint::TargetMean {
            target: 5.0,
            comparison: Comparison::AtLeast,
        }],
        selection: SelectionRule::RequireUnique,
        tolerance: 1e-12,
        max_evaluations: None,
    };
    let evidence = ForwardEvidence {
        point: Some(cohort_forward_claim(&source).unwrap()),
        ..ForwardEvidence::default()
    };
    let result = evaluate_inverse_query(&query, &evidence).unwrap();
    assert_eq!(result.actions[0].point, Some(FeasibilityStatus::Infeasible));
    assert_eq!(result.actions[1].point, Some(FeasibilityStatus::Feasible));
    assert!((result.actions[0].point_values[0].value.unwrap() - 3.45).abs() < 1e-12);
    assert!((result.actions[1].point_values[0].value.unwrap() - 7.4).abs() < 1e-12);
    assert_eq!(result.actions[1].point_values[0].standard_error, None);
    assert_eq!(result.selection, SelectionOutcome::Selected);
    assert_eq!(result.selected.as_deref(), Some("treat"));

    // A quantile target is a law question the mean cannot answer.
    let mut quantile = query.clone();
    quantile.constraints = vec![InverseConstraint::TargetQuantile {
        p: 0.5,
        target: 5.0,
        comparison: Comparison::AtLeast,
    }];
    assert!(evaluate_inverse_query(&quantile, &evidence).is_err());

    // A shifted cohort is not offered as a licensed forward source.
    let shifted = CohortDecisionSource::build(
        &model(),
        &cohort("clinic_b"),
        "mmHg",
        &CohortOptions {
            shift: Some(CovariateShiftDeclaration::new("target", "clinic_b", &["z"])),
            expected_contrast_regime: None,
        },
        &ExecutionContext::for_tests(3),
    )
    .unwrap();
    let refusal = cohort_forward_claim(&shifted).unwrap_err();
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, "cohort_decision.inverse_shift_not_licensed");
}
