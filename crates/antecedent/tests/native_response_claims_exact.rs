//! Native response claims on retained joint draws, with every decision value derived by hand.
//!
//! Program: outcome `y` (mmHg) under `do(dose = d)` for d in {1, 2, 3}. The native response
//! retains four equally likely aligned credible draws per coordinate:
//!
//! ```text
//! draw   dose 1   dose 2   dose 3
//!   0       2        8       10
//!   1       4        2        0
//!   2       6        6       10
//!   3       8        4        0
//! ```
//!
//! Means are 5, 5 and 5; dose 3 is outside empirical support (withheld from mean sources).
//! Action `pair` is worth x1 * x2: per-draw 16, 8, 36, 32, so E = 92 / 4 = 23 (the product of
//! the marginal means, 25, is the answer a misaligned pairing would give). `sure` is worth 4 * x1:
//! per-draw 8, 16, 24, 32, so E = 20. `far` reads dose 3 and is worth x3.
//! P(pair <= 10) = 1/4 (only draw 1). With dose 3 unsupported, `far` is left out and the
//! comparison is pair 23 against sure 20, so pair is uniquely optimal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::native_claims::{
    NativeDecisionSource, NativeResponseClaim, NativeResponseContext,
};
use antecedent_contract::{ProgramBinding, QuantityRole, ScientificQuantity};
use antecedent_core::{
    AssumptionSet, CausalResponse, CausalSchema, CausalSchemaBuilder, ContinuousDomain,
    CredibleDraws, GridSpec, IdentificationStatus, IntervalInterpretation, ResponseFunctional,
    ResponseIdentification, ResponseUncertainty, ResponseValue, SupportRegion, SupportReport,
    SupportStatus,
};
use antecedent_design::composition_boundary::{
    ActionStatus, CoordinateIssue, DecisionInput, NativeExecutionRecord, SupportPolicy,
    SupportedVerdict, TrustEvidence, TrustRequirement, UnsupportedReason, evaluate_with_support,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, DecisionFunctional,
    SourceRepresentation, StructuralPolicy, Tail, UtilityExpr,
};
use antecedent_design::decision_eval::{Verdict, evaluate_contract, evaluate_functional};
use antecedent_io::distribution_artifact::DistributionCalibration;

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

fn context() -> NativeResponseContext {
    NativeResponseContext {
        program: ProgramBinding {
            graph_id: "graph:g".into(),
            contract_id: "contract:c".into(),
            treatment_id: "dose".into(),
            outcome_id: "y".into(),
            population_id: "target".into(),
            intervention_kind: "do".into(),
            horizon: 0,
            dose_grid: vec![1.0, 2.0, 3.0],
            dose_units: "mg".into(),
            outcome_units: "mmHg".into(),
            functional_id: "mean".into(),
            transform_id: "identity".into(),
        },
        snapshot_id: "snap-1".into(),
        rng_id: "pcg:seed=7".into(),
        calibration: DistributionCalibration::NotMeasured,
    }
}

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

fn columns() -> Vec<Vec<f64>> {
    vec![vec![2.0, 4.0, 6.0, 8.0], vec![8.0, 2.0, 6.0, 4.0], vec![10.0, 0.0, 10.0, 0.0]]
}

fn response(with_draws: bool) -> CausalResponse {
    let schema = schema();
    let grid = vec![1.0, 2.0, 3.0];
    let uncertainty = if with_draws {
        ResponseUncertainty::PointwiseBand {
            level: 0.9,
            lower: Arc::from(vec![2.0, 2.0, 0.0]),
            upper: Arc::from(vec![8.0, 8.0, 10.0]),
            interpretation: IntervalInterpretation::Credible,
            draws: Some(CredibleDraws::columns(4, &columns())),
        }
    } else {
        ResponseUncertainty::None
    };
    CausalResponse {
        estimand: ResponseFunctional::MeanCurve {
            outcome: schema.id_of("y").unwrap(),
            treatment: ContinuousDomain::new(
                schema.id_of("dose").unwrap(),
                GridSpec::Values(Arc::from(grid.clone())),
            ),
        },
        identification_status: IdentificationStatus::NonparametricallyIdentified,
        estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
            grid: Arc::from(grid),
            dimension: 1,
            mean: Arc::from(vec![5.0, 5.0, 5.0]),
        }),
        uncertainty,
        support: SupportReport {
            status: SupportStatus::OutsideEmpiricalSupport,
            query_region: SupportRegion {
                minima: Arc::from(vec![1.0]),
                maxima: Arc::from(vec![3.0]),
            },
            diagnostics: vec![],
            warnings: vec![],
            point_status: Some(Arc::from(vec![
                SupportStatus::Supported,
                SupportStatus::Supported,
                SupportStatus::OutsideEmpiricalSupport,
            ])),
        },
        assumptions: AssumptionSet::new(),
        provenance_id: "op-1".into(),
        horizon_identification: None,
        interaction_structurally_zero: false,
    }
}

fn contract(with_far: bool) -> DecisionContract {
    let action = |id: &str, dose: &[&str], utility: UtilityExpr| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: dose.iter().map(|d| quantity(d)).collect(),
        utility,
    };
    let mut actions = vec![
        action(
            "pair",
            &["1", "2"],
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
        ),
        action(
            "sure",
            &["1"],
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(4.0)),
        ),
    ];
    if with_far {
        actions.push(action("far", &["3"], UtilityExpr::Input(0)));
    }
    DecisionContract {
        actions,
        utility_units: "util".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn retained_draws_give_the_hand_derived_aligned_expected_utilities_and_probability() {
    let claim = NativeResponseClaim::from_response(&response(true), &schema(), &context()).unwrap();
    assert!(claim.has_joint_law());
    let two = contract(false);
    let requirement = two.source_requirement().unwrap().unwrap();
    let input = claim.decision_source(&requirement).unwrap();
    assert_eq!(input.representation, SourceRepresentation::JointDraws);
    // Dose 3 is masked in the law, not dropped from its columns.
    assert_eq!(input.coordinates, vec![quantity("1"), quantity("2"), quantity("3")]);
    assert_eq!(input.withheld.len(), 1);
    assert_eq!(input.withheld[0].index, 2);
    let NativeDecisionSource::JointLaw(artifact) = &input.source else {
        panic!("a nonlinear utility needs the joint law");
    };
    assert_eq!(artifact.shape(), [4, 3]);
    assert_eq!(
        artifact.draws(),
        &[2.0, 8.0, 10.0, 4.0, 2.0, 0.0, 6.0, 6.0, 10.0, 8.0, 4.0, 0.0][..]
    );

    let result = evaluate_contract(&two, artifact).unwrap();
    assert!(near(result.actions[0].expected_utility, 23.0));
    assert!(near(result.actions[1].expected_utility, 20.0));
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("pair".into()));

    let p = evaluate_functional(
        &two,
        "pair",
        DecisionFunctional::Probability { threshold: 10.0, tail: Tail::Lower },
        artifact,
    )
    .unwrap();
    assert!(near(p.value, 0.25));
}

#[test]
fn a_mean_only_claim_cannot_answer_the_nonlinear_utility() {
    let claim =
        NativeResponseClaim::from_response(&response(false), &schema(), &context()).unwrap();
    assert!(!claim.has_joint_law());
    let requirement = contract(false).source_requirement().unwrap().unwrap();
    let refusal = claim.decision_source(&requirement).unwrap_err();
    assert_eq!(refusal.detail, antecedent_core::reason_details::native_claims::SOURCE_NOT_SUPPLIED);
    assert_eq!(refusal.supplied.as_deref(), Some("mean"));
}

#[test]
fn the_claims_support_labels_decide_which_action_the_composed_decision_scores() {
    let claim = NativeResponseClaim::from_response(&response(true), &schema(), &context()).unwrap();
    let three = contract(true);
    let requirement = three.source_requirement().unwrap().unwrap();
    let input = claim.decision_source(&requirement).unwrap();
    let NativeDecisionSource::JointLaw(artifact) = input.source else {
        panic!("joint law expected");
    };
    // The law is native-supported only with a matching execution record for its provider
    // ("native:antecedent") and snapshot.
    let evidence = TrustEvidence::NativeExecution(NativeExecutionRecord {
        execution_id: "exec-1".into(),
        provider_id: "native:antecedent".into(),
        snapshot_id: "snap-1".into(),
    });
    let composed = DecisionInput::from_distribution_artifact(
        "native",
        &artifact,
        &evidence,
        TrustRequirement::Native,
    )
    .unwrap();
    let decision =
        evaluate_with_support(&three, &[composed], &SupportPolicy::compare_supported()).unwrap();
    let scored: Vec<(&str, f64)> =
        decision.outcomes.iter().map(|o| (o.id.as_str(), o.expected_utility)).collect();
    assert_eq!(scored.len(), 2);
    assert_eq!(scored[0].0, "pair");
    assert!(near(scored[0].1, 23.0));
    assert_eq!(scored[1].0, "sure");
    assert!(near(scored[1].1, 20.0));
    assert_eq!(
        decision.dispositions[2].status,
        ActionStatus::Unsupported {
            reasons: vec![UnsupportedReason {
                input_id: "native".into(),
                coordinate: 0,
                issue: CoordinateIssue::BelowSupport(SupportStatus::OutsideEmpiricalSupport),
            }],
        }
    );
    assert_eq!(
        decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("pair".into()))
    );
}
