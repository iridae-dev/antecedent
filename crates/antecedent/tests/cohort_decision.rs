//! A fitted effect model decides for a cohort: actual DR-learner producer, cohort adapter,
//! affine decision contract with hand-calculated cohort means and utilities, replay under a
//! retained identity, and the negative gates.
//!
//! Oracle. The fixture is noiseless: `Y(0) = 1 + z`, `Y(1) = effect + 1 + 1.5 z`, with every
//! (treatment, z) cell occurring 40 times, so the linear final stage recovers
//! `tau(z) = effect + 0.5 z` exactly whatever the nuisance fits are. For `effect = 2` the
//! cohort `z = (-1, 0.5, 1)` with weights `(0.2, 0.3, 0.5)` has weighted mean contrast
//! `0.2 * 1.5 + 0.3 * 2.25 + 0.5 * 2.5 = 2.225`; the utility `4 x - 1.5` has expected
//! utility `7.4` and `10 x - 25` has `-2.75`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::cohort_decision::{
    CATE_FUNCTIONAL_ID, CohortBinding, CohortColumn, CohortExpectation, CohortOptions,
    CovariateShiftDeclaration, FittedEffectModel, LawRequest, cohort_decision_input,
    consume_cohort_decision_source, evaluate_cohort_decision,
};
use antecedent::analysis::compact_export::{
    CompactExport, ExportInput, ExportSpec, InputSupport, TermSpec,
};
use antecedent::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, SupportStatus, VariableId};
use antecedent_data::TabularData;
use antecedent_design::composition_boundary::{
    ProviderKind, SupportPolicy, SupportedVerdict, TrustRequirement, UnsupportedActionPolicy,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::Verdict;
use antecedent_estimate::DrLearner;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_learn::{LearnerSpec, LinearSpec, PredictionMap};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(97)
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() < tolerance, "actual={actual}, expected={expected}");
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

/// The exported analysis result and the in-process fitted effect of a DR-learner fit.
fn fit(effect: f64) -> (Vec<u8>, antecedent_estimate::FittedEffect) {
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
    let fitted = result
        .estimate()
        .as_effect()
        .unwrap()
        .heterogeneity
        .fitted_effect
        .as_ref()
        .expect("a DR-learner fit retains its portable CATE map")
        .as_ref()
        .clone();
    let bytes = prepared.encode_contracted_result(&result, "dr-fit", &ctx()).unwrap();
    (bytes, fitted)
}

fn cohort_of(population: &str, z: &[f64], w: &[f64]) -> CohortBinding {
    CohortBinding::new(
        "clinic-2026-q1",
        population,
        (0..z.len()).map(|i| format!("patient-{i}")).collect(),
        vec!["z".into()],
        vec![CohortColumn::Numeric(z.to_vec())],
        w.to_vec(),
    )
    .unwrap()
}

fn cohort() -> CohortBinding {
    cohort_of("target", &[-1.0, 0.5, 1.0], &[0.2, 0.3, 0.5])
}

fn contract(
    coordinate: &antecedent_contract::ScientificQuantity,
    treat: UtilityExpr,
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
        target_population: coordinate.population_id.clone(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn linear_utility(slope: f64, offset: f64) -> UtilityExpr {
    UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(slope)),
        UtilityExpr::Const(-offset),
    )
}

#[test]
fn producer_to_cohort_to_affine_contract_matches_the_hand_calculation() {
    let (bytes, fitted) = fit(2.0);
    // The producer retains the fitted-row support contract the cohort is checked against.
    let support = fitted.support.as_ref().expect("the producer retains its support contract");
    assert_eq!(support.rows, 160);
    close(support.ranges[0].lo, -1.0, 1e-12);
    close(support.ranges[0].hi, 1.0, 1e-12);

    let model = FittedEffectModel::load(&bytes).unwrap();
    assert_eq!(model.features(), vec!["z".to_owned()]);
    let source =
        model.decision_source(&cohort(), "mmHg", &CohortOptions::default(), &ctx()).unwrap();
    close(source.mean_contrast(), 2.225, 1e-8);
    assert!(source.is_native());
    assert_eq!(source.support_status(), SupportStatus::Supported);
    assert_eq!(source.support().fitted_rows, Some(160));
    assert_eq!(source.quantity().regime_id, "contrast:do(t=1)-do(t=0)");
    assert_eq!(source.quantity().variable_id, "y");
    assert_eq!(source.quantity().functional_id, "mean_difference");
    assert_eq!(source.model().parent().claim_id, model.parent_claim());
    // No standard error is carried for a portable (non-linear-covariance) model.
    assert!(source.uncalibrated_standard_error().is_none());
}

#[test]
fn affine_contracts_on_the_cohort_source_match_hand_calculated_utilities() {
    let (bytes, _) = fit(2.0);
    let model = FittedEffectModel::load(&bytes).unwrap();
    let source =
        model.decision_source(&cohort(), "mmHg", &CohortOptions::default(), &ctx()).unwrap();

    // The native input resolves as native: the numbers were recomputed from the verified model.
    let input = cohort_decision_input("cohort", &source, TrustRequirement::Native).unwrap();
    assert_eq!(input.provenance().provider_kind, ProviderKind::Native);

    let policy = SupportPolicy::require_all();
    let treat = evaluate_cohort_decision(
        &contract(source.quantity(), linear_utility(4.0, -1.5)),
        &source,
        &policy,
    )
    .unwrap();
    close(treat.outcomes[0].expected_utility, 7.4, 1e-7);
    assert_eq!(treat.outcomes[0].standard_error, None);
    assert_eq!(treat.verdict, SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into())));

    let hold = evaluate_cohort_decision(
        &contract(source.quantity(), linear_utility(10.0, -25.0)),
        &source,
        &policy,
    )
    .unwrap();
    close(hold.outcomes[0].expected_utility, -2.75, 1e-7);
    assert_eq!(hold.verdict, SupportedVerdict::Compared(Verdict::UniquelyOptimal("wait".into())));
}

#[test]
fn the_source_replays_in_a_fresh_consumer_under_the_retained_identity() {
    let (bytes, _) = fit(2.0);
    let model = FittedEffectModel::load(&bytes).unwrap();
    let source =
        model.decision_source(&cohort(), "mmHg", &CohortOptions::default(), &ctx()).unwrap();
    let artifact = source.to_bytes("cohort-decision").unwrap();
    let expectation = CohortExpectation {
        identity: source.identity().to_owned(),
        parent_claim: Some(model.parent_claim().to_owned()),
        population_id: Some("target".into()),
        outcome_units: Some("mmHg".into()),
        contrast_regime: Some("contrast:do(t=1)-do(t=0)".into()),
        cohort_digest: Some(cohort().digest()),
    };
    let replayed = consume_cohort_decision_source(&artifact, &expectation).unwrap();
    assert_eq!(replayed.mean_contrast().to_bits(), source.mean_contrast().to_bits());
    assert_eq!(replayed.identity(), source.identity());
    assert!(replayed.is_native());
}

#[test]
fn negative_gates_are_typed_and_actionable() {
    let (bytes, _) = fit(2.0);
    let model = FittedEffectModel::load(&bytes).unwrap();
    let build = |cohort: &CohortBinding, units: &str, options: &CohortOptions| {
        model.decision_source(cohort, units, options, &ctx())
    };
    let default = CohortOptions::default();

    // Outside the fitted support.
    let outside = cohort_of("target", &[0.0, 1.5], &[0.5, 0.5]);
    let error = build(&outside, "mmHg", &default).unwrap_err();
    assert_eq!(error.detail, "cohort_decision.outside_support");
    assert!(error.subject.contains("patient-1"), "{}", error.subject);
    error.to_refusal().validate().unwrap();

    // Missing feature, units and population.
    let renamed = CohortBinding::new(
        "s",
        "target",
        vec!["a".into()],
        vec!["age".into()],
        vec![CohortColumn::Numeric(vec![0.0])],
        vec![1.0],
    )
    .unwrap();
    assert_eq!(
        build(&renamed, "mmHg", &default).unwrap_err().detail,
        "cohort_decision.feature_schema_mismatch"
    );
    assert_eq!(
        build(&cohort(), "", &default).unwrap_err().detail,
        "cohort_decision.outcome_units_required"
    );
    let foreign = cohort_of("clinic_b", &[0.0], &[1.0]);
    let refusal = build(&foreign, "mmHg", &default).unwrap_err();
    assert_eq!(refusal.code, "transport_missing_evidence");
    assert_eq!(refusal.detail, "cohort_decision.shift_declaration_required");

    // A declared shift is point-only evidence under an assumption, never `Supported`.
    let shifted = CohortOptions {
        shift: Some(CovariateShiftDeclaration::new("target", "clinic_b", &["z"])),
        expected_contrast_regime: None,
    };
    let transported = build(&foreign, "mmHg", &shifted).unwrap();
    close(transported.mean_contrast(), 2.0, 1e-8);
    assert_eq!(transported.support_status(), SupportStatus::Extrapolative);
    let weakest = SupportPolicy {
        unsupported: UnsupportedActionPolicy::CompareSupported,
        weakest_support: SupportStatus::Supported,
    };
    let strict = evaluate_cohort_decision(
        &contract(transported.quantity(), linear_utility(1.0, 0.0)),
        &transported,
        &weakest,
    )
    .unwrap();
    assert_eq!(strict.verdict, SupportedVerdict::NoSupportedAction);
    // The shifted cohort must still lie inside the fitted support.
    let beyond = cohort_of("clinic_b", &[0.0, 4.0], &[0.5, 0.5]);
    let error = build(&beyond, "mmHg", &shifted).unwrap_err();
    assert_eq!(error.code, "transport_support_failure");

    // Contrast mismatch.
    let other_contrast = CohortOptions {
        shift: None,
        expected_contrast_regime: Some("contrast:do(t=2)-do(t=0)".into()),
    };
    assert_eq!(
        build(&cohort(), "mmHg", &other_contrast).unwrap_err().detail,
        "cohort_decision.contrast_mismatch"
    );

    // Law-valued requests are refused with a remedy.
    let source = build(&cohort(), "mmHg", &default).unwrap();
    for request in [
        LawRequest::Probability,
        LawRequest::Quantile,
        LawRequest::NonlinearUtility,
        LawRequest::JointLaw,
    ] {
        assert!(!source.law_refusal(request).remedy.is_empty());
    }
    assert_eq!(source.interval_refusal().code, "cell_not_licensed");
}

#[test]
fn changed_rows_weights_cohort_and_parent_are_refused() {
    let (bytes, _) = fit(2.0);
    let (other_bytes, _) = fit(3.0);
    let model = FittedEffectModel::load(&bytes).unwrap();
    let other = FittedEffectModel::load(&other_bytes).unwrap();
    assert_ne!(model.parent_claim(), other.parent_claim());
    let options = CohortOptions::default();
    let source = model.decision_source(&cohort(), "mmHg", &options, &ctx()).unwrap();
    let artifact = source.to_bytes("cohort-decision").unwrap();
    let retained = CohortExpectation::for_identity(source.identity());

    // Another parent claim issues another identity.
    let from_other = other.decision_source(&cohort(), "mmHg", &options, &ctx()).unwrap();
    assert_ne!(from_other.identity(), source.identity());
    close(from_other.mean_contrast(), 3.0 + 0.5 * 0.45, 1e-8);
    let wrong = consume_cohort_decision_source(
        &from_other.to_bytes("cohort-decision").unwrap(),
        &CohortExpectation {
            parent_claim: Some(model.parent_claim().to_owned()),
            ..retained.clone()
        },
    )
    .unwrap_err();
    assert_eq!(wrong.detail, "cohort_decision.wrong_parent_identity");
    assert_eq!(
        FittedEffectModel::load_expecting(&other_bytes, model.parent_claim()).unwrap_err().detail,
        "cohort_decision.wrong_parent_identity"
    );

    // A changed weight, a changed row, a changed identity: different cohort, different source.
    for changed in [
        cohort_of("target", &[-1.0, 0.5, 1.0], &[0.3, 0.2, 0.5]),
        cohort_of("target", &[-1.0, 0.25, 1.0], &[0.2, 0.3, 0.5]),
    ] {
        let source2 = model.decision_source(&changed, "mmHg", &options, &ctx()).unwrap();
        assert_ne!(source2.identity(), source.identity());
        let bytes2 = source2.to_bytes("cohort-decision").unwrap();
        let error = consume_cohort_decision_source(
            &bytes2,
            &CohortExpectation { cohort_digest: Some(cohort().digest()), ..retained.clone() },
        )
        .unwrap_err();
        assert_eq!(error.detail, "cohort_decision.cohort_changed");
        let error = consume_cohort_decision_source(&bytes2, &retained).unwrap_err();
        assert_eq!(error.detail, "cohort_decision.identity_unexpected");
    }

    // A corrupted parent result and a corrupted source are refused.
    let mut corrupted = bytes.clone();
    let middle = corrupted.len() / 2;
    corrupted[middle] ^= 0xFF;
    assert!(FittedEffectModel::load(&corrupted).is_err());
    let mut corrupted = artifact;
    let middle = corrupted.len() / 2;
    corrupted[middle] ^= 0xFF;
    assert!(consume_cohort_decision_source(&corrupted, &retained).is_err());
}

#[test]
fn a_compact_linear_model_carries_the_uncalibrated_variance_diagnostic_only() {
    let (_, fitted) = fit(2.0);
    // Coefficients are the producer's own retained linear map (intercept, slope on z).
    let PredictionMap::Linear { coefficients, .. } = &fitted.predictor.model else {
        panic!("the DR-learner's linear final stage retains a linear map");
    };
    assert_eq!(coefficients.len(), 2);
    let wire = |id: &str, role: &str, units: &str, functional: &str, regime: &str| {
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
    };
    let export = CompactExport::build(ExportSpec {
        response: wire("y", "outcome", "mmHg", CATE_FUNCTIONAL_ID, "contrast:do(t=1)-do(t=0)"),
        inputs: vec![ExportInput {
            quantity: wire("z", "covariate", "1", "mean", "observed"),
            support: InputSupport::Range { lo: -1.0, hi: 1.0 },
        }],
        terms: vec![TermSpec::Intercept, TermSpec::Linear { quantity: "z".into() }],
        coefficients: coefficients.clone(),
        covariance: vec![0.04, -0.01, -0.01, 0.09],
        mask: vec![],
    })
    .unwrap();
    let model = FittedEffectModel::from_compact_export(&export).unwrap();
    let source =
        model.decision_source(&cohort(), "mmHg", &CohortOptions::default(), &ctx()).unwrap();
    close(source.mean_contrast(), 2.225, 1e-8);
    let diagnostic = source.uncalibrated_standard_error().unwrap();
    // phibar = (1, 0.45): 0.04 - 2 * 0.01 * 0.45 + 0.09 * 0.45^2 = 0.049225.
    close(diagnostic.value, 0.049_225_f64.sqrt(), 1e-12);
    assert_eq!(diagnostic.calibration, "calibration unmeasured");
    // A declared model is attested, never native, and exposes no interval.
    assert!(!source.is_native());
    assert!(cohort_decision_input("c", &source, TrustRequirement::Native).is_err());
    assert_eq!(source.interval_refusal().detail, "cohort_decision.interval_requires_native_fit");
    // The decision result inherits no standard error from the diagnostic.
    let decision = evaluate_cohort_decision(
        &contract(source.quantity(), linear_utility(4.0, -1.5)),
        &source,
        &SupportPolicy::require_all(),
    )
    .unwrap();
    assert_eq!(decision.outcomes[0].standard_error, None);
}
