//! C1: a native response feeds decision flows through a typed claim.
//!
//! Hand example. Program: outcome `y` (mmHg) under `do(dose = d)` for the grid
//! `d in {1, 2}` in the target population. Native means: `E[y | do(1)] = 3`,
//! `E[y | do(2)] = 5`. Action A reads the dose-1 mean (utility `m1`), action B
//! reads the dose-2 mean (utility `m2 - 1.5`). Expected utility is A = 3.0 and
//! B = 3.5, so B is uniquely optimal. The joint-draw response retains two
//! aligned draws per coordinate, dose 1 `[2, 4]` and dose 2 `[4, 6]`, whose means
//! are the same 3 and 5; B's per-draw utility is `[2.5, 4.5]`, so
//! `P(utility_B <= 3) = 1/2`.

use std::sync::Arc;

use antecedent::analysis::native_claims::{
    NativeDecisionSource, NativeResponseClaim, NativeResponseContext, native_response_coordinates,
};
use antecedent_core::{
    AssumptionSet, CausalResponse, CausalSchema, CausalSchemaBuilder, ContinuousDomain,
    CredibleDraws, GridSpec, IdentificationStatus, IntervalInterpretation, ProgramBinding,
    QuantityRole, ResponseFunctional, ResponseIdentification, ResponseUncertainty, ResponseValue,
    ScientificQuantity, SupportRegion, SupportReport, SupportStatus,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, DecisionFunctional,
    SourceRepresentation, StructuralPolicy, Tail, UtilityExpr,
};
use antecedent_design::decision_eval::{
    DecisionEvalError, Verdict, evaluate_contract, evaluate_contract_on_means, evaluate_functional,
};
use antecedent_io::distribution_artifact::{
    DistributionCalibration, DistributionTrust, DrawAlignment,
};

fn schema() -> CausalSchema {
    CausalSchemaBuilder::new()
        .continuous("dose")
        .treatment()
        .continuous("y")
        .unit("mmHg")
        .outcome()
        .build()
        .unwrap()
}

fn program() -> ProgramBinding {
    ProgramBinding {
        graph_id: "graph:g".into(),
        contract_id: "contract:c".into(),
        treatment_id: "dose".into(),
        outcome_id: "y".into(),
        population_id: "target".into(),
        intervention_kind: "do".into(),
        horizon: 0,
        dose_grid: vec![1.0, 2.0],
        dose_units: "mg".into(),
        outcome_units: "mmHg".into(),
        functional_id: "mean".into(),
        transform_id: "identity".into(),
    }
}

fn context() -> NativeResponseContext {
    NativeResponseContext {
        program: program(),
        snapshot_id: "snap-1".into(),
        rng_id: "pcg:seed=7".into(),
        calibration: DistributionCalibration::Unmeasured,
    }
}

/// The coordinate written out by hand, independent of the production derivation.
fn quantity(dose: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "target".into(),
        regime_id: format!("do(dose={dose})"),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn response_on(
    grid: &[f64],
    means: &[f64],
    labels: Option<Vec<SupportStatus>>,
    status: SupportStatus,
    uncertainty: ResponseUncertainty,
) -> CausalResponse {
    let schema = schema();
    CausalResponse {
        estimand: ResponseFunctional::MeanCurve {
            outcome: schema.id_of("y").unwrap(),
            treatment: ContinuousDomain::new(
                schema.id_of("dose").unwrap(),
                GridSpec::Values(Arc::from(grid.to_vec())),
            ),
        },
        identification_status: IdentificationStatus::NonparametricallyIdentified,
        estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
            grid: Arc::from(grid.to_vec()),
            dimension: 1,
            mean: Arc::from(means.to_vec()),
        }),
        uncertainty,
        support: SupportReport {
            status,
            query_region: SupportRegion {
                minima: Arc::from(vec![grid[0]]),
                maxima: Arc::from(vec![grid[grid.len() - 1]]),
            },
            diagnostics: vec![],
            warnings: vec![],
            point_status: labels.map(Arc::from),
        },
        assumptions: AssumptionSet::new(),
        provenance_id: "op-1".into(),
        horizon_identification: None,
        interaction_structurally_zero: false,
    }
}

fn supported() -> Vec<SupportStatus> {
    vec![SupportStatus::Supported, SupportStatus::Supported]
}

fn mean_response() -> CausalResponse {
    response_on(
        &[1.0, 2.0],
        &[3.0, 5.0],
        Some(supported()),
        SupportStatus::Supported,
        ResponseUncertainty::None,
    )
}

fn joint_response() -> CausalResponse {
    let draws = CredibleDraws::columns(2, &[vec![2.0, 4.0], vec![4.0, 6.0]]);
    response_on(
        &[1.0, 2.0],
        &[3.0, 5.0],
        Some(supported()),
        SupportStatus::Supported,
        ResponseUncertainty::PointwiseBand {
            level: 0.9,
            lower: Arc::from(vec![2.0, 4.0]),
            upper: Arc::from(vec![4.0, 6.0]),
            interpretation: IntervalInterpretation::Credible,
            draws: Some(draws),
        },
    )
}

fn contract(criterion: DecisionCriterion) -> DecisionContract {
    let action = |id: &str, dose: &str, utility: UtilityExpr| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![quantity(dose)],
        utility,
    };
    DecisionContract {
        actions: vec![
            action("A", "1", UtilityExpr::Input(0)),
            action(
                "B",
                "2",
                UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(1.5)),
            ),
        ],
        utility_units: "util".into(),
        criterion,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-12
}

#[test]
fn c1_native_response_exposes_coordinates_support_and_native_trust() {
    let claim =
        NativeResponseClaim::from_response(&mean_response(), &schema(), &context()).unwrap();
    assert_eq!(claim.coordinates(), &[quantity("1"), quantity("2")][..]);
    assert_eq!(claim.means(), &[3.0, 5.0][..]);
    assert_eq!(claim.point_status(), &[SupportStatus::Supported, SupportStatus::Supported][..]);
    assert_eq!(claim.support_status(), SupportStatus::Supported);
    assert_eq!(claim.trust(), DistributionTrust::NativeLicensed);
    assert_eq!(claim.calibration(), DistributionCalibration::Unmeasured);
    assert_eq!(claim.program_identity(), program().identity());
    assert_eq!(claim.provenance_id(), "op-1");
    assert!(!claim.has_joint_law());
    let getter = native_response_coordinates(&mean_response(), &schema(), &context()).unwrap();
    assert_eq!(getter, claim.coordinates());
}

#[test]
fn c1_native_mean_response_feeds_an_affine_decision_on_hand_values() {
    let claim =
        NativeResponseClaim::from_response(&mean_response(), &schema(), &context()).unwrap();
    let contract = contract(DecisionCriterion::PosteriorExpectedUtility);
    let requirement = contract.source_requirement().unwrap().unwrap();
    let input = claim.into_decision_source(&requirement).unwrap();
    assert_eq!(input.representation, SourceRepresentation::Mean);
    assert_eq!(input.trust, DistributionTrust::NativeLicensed);
    assert!(input.withheld.is_empty());
    assert_eq!(input.coordinates, vec![quantity("1"), quantity("2")]);
    let NativeDecisionSource::Mean(source) = &input.source else {
        panic!("a mean requirement is answered by a mean source");
    };
    assert_eq!(source.means, vec![3.0, 5.0]);
    assert_eq!(source.provider_id, "native:op-1");
    assert_eq!(source.causal_contract_id, "contract:c");
    let result = evaluate_contract_on_means(&contract, source).unwrap();
    assert!(close(result.actions[0].expected_utility, 3.0));
    assert!(close(result.actions[1].expected_utility, 3.5));
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("B".into()));
    assert_eq!(result.source.causal_contract_id, "contract:c");
}

#[test]
fn c1_native_mean_only_result_is_refused_for_probability_and_quantile_sources() {
    let claim =
        NativeResponseClaim::from_response(&mean_response(), &schema(), &context()).unwrap();
    for criterion in [
        DecisionCriterion::Quantile { p: 0.5 },
        DecisionCriterion::ThresholdProbability { threshold: 3.0 },
    ] {
        let requirement = contract(criterion).source_requirement().unwrap().unwrap();
        let refusal = claim.decision_source(&requirement).unwrap_err();
        assert_eq!(refusal.code, "decision_contract_unsatisfied");
        assert_eq!(refusal.detail, "native_claims.source_not_supplied");
        assert_eq!(refusal.expected.as_deref(), Some("joint_draws,marginal_draws"));
        assert_eq!(refusal.supplied.as_deref(), Some("mean"));
        assert!(refusal.validate().is_ok());
    }
    // Draws that are not credible draws of every coordinate are not a joint law.
    let mut partial = joint_response();
    partial.uncertainty = ResponseUncertainty::PointwiseBand {
        level: 0.9,
        lower: Arc::from(vec![2.0, 4.0]),
        upper: Arc::from(vec![4.0, 6.0]),
        interpretation: IntervalInterpretation::Confidence,
        draws: Some(CredibleDraws::columns(2, &[vec![2.0, 4.0], vec![4.0, 6.0]])),
    };
    let claim = NativeResponseClaim::from_response(&partial, &schema(), &context()).unwrap();
    assert!(!claim.has_joint_law());
}

#[test]
fn c1_native_joint_response_supplies_an_aligned_law_for_nonlinear_functionals() {
    let claim =
        NativeResponseClaim::from_response(&joint_response(), &schema(), &context()).unwrap();
    assert!(claim.has_joint_law());
    let contract = contract(DecisionCriterion::Quantile { p: 0.5 });
    let requirement = contract.source_requirement().unwrap().unwrap();
    let input = claim.into_decision_source(&requirement).unwrap();
    assert_eq!(input.representation, SourceRepresentation::JointDraws);
    let NativeDecisionSource::JointLaw(artifact) = &input.source else {
        panic!("a quantile requirement is answered by an aligned joint law");
    };
    assert_eq!(artifact.shape(), [2, 2]);
    // Draw-major rows: draw 0 = (2, 4), draw 1 = (4, 6).
    assert_eq!(artifact.draws(), &[2.0, 4.0, 4.0, 6.0][..]);
    assert_eq!(artifact.metadata().identity.alignment, DrawAlignment::Joint);
    assert_eq!(artifact.metadata().trust, DistributionTrust::NativeLicensed);
    assert_eq!(artifact.metadata().calibration, DistributionCalibration::Unmeasured);
    assert_eq!(artifact.metadata().identity.causal_contract_id, "contract:c");
    assert!(close(artifact.mean(0).unwrap(), 3.0));
    assert!(close(artifact.mean(1).unwrap(), 5.0));
    let expected = contract_expected_utility(&artifact_contract(), artifact);
    assert!(close(expected.0, 3.0) && close(expected.1, 3.5));
    let probability = evaluate_functional(
        &contract,
        "B",
        DecisionFunctional::Probability { threshold: 3.0, tail: Tail::Lower },
        artifact,
    )
    .unwrap();
    assert!(close(probability.value, 0.5));
}

fn artifact_contract() -> DecisionContract {
    contract(DecisionCriterion::PosteriorExpectedUtility)
}

fn contract_expected_utility(
    contract: &DecisionContract,
    artifact: &antecedent_io::distribution_artifact::DistributionArtifact,
) -> (f64, f64) {
    let result = evaluate_contract(contract, artifact).unwrap();
    (result.actions[0].expected_utility, result.actions[1].expected_utility)
}

#[test]
fn c1_native_unsupported_coordinates_are_withheld_not_silently_used() {
    let labels = Some(vec![SupportStatus::Supported, SupportStatus::OutsideEmpiricalSupport]);
    let response = response_on(
        &[1.0, 2.0],
        &[3.0, 5.0],
        labels,
        SupportStatus::OutsideEmpiricalSupport,
        ResponseUncertainty::None,
    );
    let claim = NativeResponseClaim::from_response(&response, &schema(), &context()).unwrap();
    let contract = contract(DecisionCriterion::PosteriorExpectedUtility);
    let requirement = contract.source_requirement().unwrap().unwrap();
    let input = claim.decision_source(&requirement).unwrap();
    assert_eq!(input.coordinates, vec![quantity("1")]);
    assert_eq!(input.point_status, vec![SupportStatus::Supported]);
    assert_eq!(input.withheld.len(), 1);
    assert_eq!(input.withheld[0].index, 1);
    assert_eq!(input.withheld[0].status, SupportStatus::OutsideEmpiricalSupport);
    let NativeDecisionSource::Mean(source) = &input.source else {
        panic!("mean source expected");
    };
    // The action that reads the withheld coordinate is visibly unevaluable.
    let error = evaluate_contract_on_means(&contract, source).unwrap_err();
    assert_eq!(error, DecisionEvalError::QuantityNotFound { action: "B".into(), input: 0 });
    // Nothing supported at all refuses outright.
    let none = response_on(
        &[1.0, 2.0],
        &[3.0, 5.0],
        Some(vec![SupportStatus::MissingEvidence, SupportStatus::MissingEvidence]),
        SupportStatus::MissingEvidence,
        ResponseUncertainty::None,
    );
    let claim = NativeResponseClaim::from_response(&none, &schema(), &context()).unwrap();
    let refusal = claim.decision_source(&requirement).unwrap_err();
    assert_eq!(refusal.detail, "native_claims.no_supported_coordinate");
}

#[test]
fn c1_native_response_for_another_request_is_refused_against_the_program() {
    // A different dose grid.
    let changed = response_on(
        &[1.0, 3.0],
        &[3.0, 6.0],
        Some(supported()),
        SupportStatus::Supported,
        ResponseUncertainty::None,
    );
    let refusal = NativeResponseClaim::from_response(&changed, &schema(), &context()).unwrap_err();
    assert_eq!(refusal.detail, "program_binding.dose_grid_changed");
    // A program that asks for another outcome than the response estimated.
    let mut other = context();
    other.program.outcome_id = "z".into();
    let refusal =
        NativeResponseClaim::from_response(&mean_response(), &schema(), &other).unwrap_err();
    assert_eq!(refusal.detail, "program_binding.treatment_outcome_substitution");
    assert_eq!(refusal.offending.as_deref(), Some("outcome_id"));
    // A schema unit that contradicts the program's units is never converted.
    let mut kpa = context();
    kpa.program.outcome_units = "kPa".into();
    let refusal =
        NativeResponseClaim::from_response(&mean_response(), &schema(), &kpa).unwrap_err();
    assert_eq!(refusal.detail, "native_claims.units_conflict");
    assert_eq!(refusal.expected.as_deref(), Some("kPa"));
    assert_eq!(refusal.supplied.as_deref(), Some("mmHg"));
}

#[test]
fn c1_native_non_point_or_unlabeled_responses_are_refused() {
    let mut partial = mean_response();
    partial.estimate = ResponseIdentification::PartiallyIdentified(ResponseValue::Scalar(1.0));
    let refusal = NativeResponseClaim::from_response(&partial, &schema(), &context()).unwrap_err();
    assert_eq!(
        (refusal.code, refusal.detail.as_str()),
        ("effect_not_identified", "native_claims.response_not_point_identified")
    );
    let mut scalar = mean_response();
    scalar.estimate = ResponseIdentification::PointIdentified(ResponseValue::Scalar(1.0));
    let refusal = NativeResponseClaim::from_response(&scalar, &schema(), &context()).unwrap_err();
    assert_eq!(refusal.detail, "native_claims.unsupported_estimand");
    let mut derivative = mean_response();
    derivative.estimand = ResponseFunctional::AverageDerivative {
        outcome: schema().id_of("y").unwrap(),
        treatment: schema().id_of("dose").unwrap(),
        weighting: antecedent_core::DerivativeWeighting::Observed,
    };
    let refusal =
        NativeResponseClaim::from_response(&derivative, &schema(), &context()).unwrap_err();
    assert_eq!(refusal.detail, "native_claims.unsupported_estimand");
    let unlabeled = response_on(
        &[1.0, 2.0],
        &[3.0, 5.0],
        None,
        SupportStatus::Supported,
        ResponseUncertainty::None,
    );
    let refusal =
        NativeResponseClaim::from_response(&unlabeled, &schema(), &context()).unwrap_err();
    assert_eq!(refusal.detail, "native_claims.point_support_missing");
    let hidden = response_on(
        &[1.0, 2.0],
        &[3.0, 5.0],
        Some(vec![SupportStatus::Supported, SupportStatus::WeakOverlap]),
        SupportStatus::Supported,
        ResponseUncertainty::None,
    );
    let refusal = NativeResponseClaim::from_response(&hidden, &schema(), &context()).unwrap_err();
    assert_eq!(refusal.detail, "coordinate_support.summary_not_worst_label");
    let mut no_snapshot = context();
    no_snapshot.snapshot_id = " ".into();
    let refusal =
        NativeResponseClaim::from_response(&mean_response(), &schema(), &no_snapshot).unwrap_err();
    assert_eq!(refusal.detail, "native_claims.invalid_context");
}
