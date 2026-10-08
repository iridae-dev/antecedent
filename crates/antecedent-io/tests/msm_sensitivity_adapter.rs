//! 2.3 B3: the marginal sensitivity model result adapted into the F17
//! `SensitivityArtifact`, consumed by its recomputing decision classification.
//!
//! Inputs are the two-stratum example of `antecedent-validate/tests/msm_sensitivity.rs`
//! (identified ATE 0.3; ATE bounds [0.04375, 0.4875] at Lambda = 2, and about
//! [-0.0833, 0.55] at Lambda = 3, derived there by hand). Hand expectations:
//!
//! * invariant: grid Lambda in {1, 1.5, 2}, `treat = ate`, `skip = 0`. The lower bound is
//!   never below 0.04375 > 0, so `treat` is uniquely best under both range vertices at
//!   every point.
//! * switch: grid {1, 2, 3}, `treat = ate - cost`, `skip = 0` with the point quantity
//!   `cost = (0, 0, 2)`. The ATE is at most 1, so at Lambda = 3 `ate - 2 < 0` under both
//!   vertices: `treat` leads at 1 and 2 (lower bound >= 0.04375 > 0), `skip` at 3, so the
//!   switch lies in [2, 3].
//! * no robust action: grid {1, 2, 3}, `treat = ate`, `skip = 0.2`. At 1 the range is the
//!   point 0.3 > 0.2 so `treat` leads; at 2 the range [0.04375, 0.4875] and at 3 the range
//!   [-0.0833, 0.55] both contain 0.2, so the best action depends on where the range
//!   sits: mixed at {2, 3}.

#![allow(clippy::float_cmp, reason = "exact grid endpoints and copied surface values")]

use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_io::error::IoError;
use antecedent_io::msm_sensitivity_adapter::{
    MSM_COORDINATE_ID, MSM_COORDINATE_SCALE, MSM_SOURCE_KIND, MsmOutcomeLaw, MsmSensitivityResult,
    MsmSensitivitySpec, MsmStratum, msm_ate_sensitivity, msm_sensitivity_artifact,
};
use antecedent_io::sensitivity_artifact::{
    ActionUtility, DecisionSwitch, NoRobustReason, SamplingInterval, SamplingStatus,
    SensitivityArtifact, SensitivityOutcome, UtilityTerm,
};

fn binary(rate: f64) -> MsmOutcomeLaw {
    MsmOutcomeLaw { values: vec![0.0, 1.0], probabilities: vec![1.0 - rate, rate] }
}

fn strata() -> Vec<MsmStratum> {
    vec![
        MsmStratum { mass: 0.5, propensity: 0.5, treated: binary(0.8), control: binary(0.4) },
        MsmStratum { mass: 0.5, propensity: 0.25, treated: binary(0.5), control: binary(0.3) },
    ]
}

fn result(lambda_max: f64, grid_points: usize) -> MsmSensitivityResult {
    let mut spec = MsmSensitivitySpec::new(lambda_max).with_threshold(0.0);
    spec.grid_points = grid_points;
    msm_ate_sensitivity(&strata(), &spec).unwrap()
}

fn scientific(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "utils".into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "msm_ate".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn act(id: &str, utility: UtilityTerm) -> ActionUtility {
    ActionUtility { id: id.into(), utility }
}

fn treat_vs(skip: f64) -> Vec<ActionUtility> {
    vec![act("treat", UtilityTerm::quantity("ate")), act("skip", UtilityTerm::Const(skip))]
}

/// `treat = ate - cost` against `skip = 0`.
fn net_of_cost() -> Vec<ActionUtility> {
    let net = UtilityTerm::difference(UtilityTerm::quantity("ate"), UtilityTerm::quantity("cost"));
    vec![act("treat", net), act("skip", UtilityTerm::Const(0.0))]
}

fn artifact(result: &MsmSensitivityResult, actions: Vec<ActionUtility>) -> SensitivityArtifact {
    msm_sensitivity_artifact(result, &scientific("ate"), &[], actions, "checked-contract").unwrap()
}

fn refusal(error: &IoError) -> (&'static str, &str) {
    match error {
        IoError::Refused { code, message } => (*code, message.as_str()),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn b3_msm_adapter_builds_the_lambda_surface_with_distinct_uncertainty_kinds() {
    let result = result(3.0, 5);
    let built = artifact(&result, treat_vs(0.0));
    let coordinate = built.coordinate();
    assert_eq!(coordinate.id, MSM_COORDINATE_ID);
    assert_eq!(coordinate.scale, MSM_COORDINATE_SCALE);
    assert_eq!((coordinate.minimum, coordinate.maximum), (1.0, 3.0));
    assert_eq!(built.grid(), &[1.0, 1.5, 2.0, 2.5, 3.0]);
    let surface = &built.quantities()[0];
    for (index, point) in result.grid.iter().enumerate() {
        assert_eq!(surface.lower[index], point.lower);
        assert_eq!(surface.upper[index], point.upper);
    }
    // Lambda = 1 is the identified point; the range widens from it.
    assert_eq!(surface.lower[0], surface.upper[0]);
    let uncertainty = built.uncertainty();
    assert_eq!(uncertainty.assumption_range.kind, "assumption_range");
    assert!(uncertainty.identified_bound.is_none());
    let SamplingStatus::Withheld { reason_code, detail } = &uncertainty.sampling else {
        panic!("the sampling interval must be withheld");
    };
    assert_eq!(reason_code, "cell_not_licensed");
    assert_eq!(detail, "msm_sensitivity.interval_withheld");
    let provenance = built.provenance();
    assert_eq!(provenance.source_kind, MSM_SOURCE_KIND);
    assert_eq!(provenance.decision_threshold, Some(0.0));
    assert_eq!(provenance.source_tipping.len(), 1);
    assert_eq!(provenance.source_tipping[0].status, "bracketed");
    assert!(provenance.source_tipping[0].lower.is_some());
}

#[test]
fn b3_msm_artifact_reaches_an_invariant_action() {
    let built = artifact(&result(2.0, 3), treat_vs(0.0));
    let expected = SensitivityOutcome::InvariantAction { action: "treat".into() };
    assert_eq!(built.outcome(), Some(&expected));
    assert_eq!(built.decision_outcome(None).unwrap(), expected);
}

#[test]
fn b3_msm_artifact_reaches_an_assumption_dependent_switch() {
    let result = result(3.0, 3);
    let cost = (scientific("cost"), vec![0.0, 0.0, 2.0]);
    let built = msm_sensitivity_artifact(
        &result,
        &scientific("ate"),
        &[cost],
        net_of_cost(),
        "checked-contract",
    )
    .unwrap();
    let expected = SensitivityOutcome::AssumptionDependent {
        switch: DecisionSwitch {
            from: vec!["treat".into()],
            to: vec!["skip".into()],
            lower: 2.0,
            upper: 3.0,
            exact: false,
            interpolated: None,
        },
    };
    assert_eq!(built.decision_outcome(None).unwrap(), expected);
    assert_eq!(built.outcome(), Some(&expected));
}

#[test]
fn b3_msm_artifact_reaches_no_robust_action_when_the_range_straddles() {
    let built = artifact(&result(3.0, 3), treat_vs(0.2));
    let expected = SensitivityOutcome::NoRobustAction {
        reason: NoRobustReason::MixedWithinRange { coordinates: vec![2.0, 3.0] },
        switches: vec![],
    };
    assert_eq!(built.decision_outcome(None).unwrap(), expected);
    assert_eq!(built.outcome(), Some(&expected));
}

#[test]
fn b3_msm_artifact_round_trips_and_replays_its_decision() {
    let built = artifact(&result(3.0, 3), treat_vs(0.2));
    let bytes = built.to_bytes("msm-1").unwrap();
    let restored = SensitivityArtifact::from_bytes(&bytes, Some(built.identity())).unwrap();
    assert_eq!(restored, built);
    assert_eq!(restored.outcome(), built.outcome());
}

#[test]
fn b3_msm_artifact_refuses_a_sampling_composition() {
    let built = artifact(&result(2.0, 3), treat_vs(0.0));
    let mut parts = built.parts().clone();
    let n = parts.grid.len();
    parts.uncertainty.sampling = SamplingStatus::Reported(SamplingInterval {
        quantity: "ate".into(),
        level: 0.95,
        method: "percentile_bootstrap".into(),
        composed: Some("sum_of_interval_and_range".into()),
        lower: vec![0.0; n],
        upper: vec![1.0; n],
    });
    let error = SensitivityArtifact::new(parts).unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "cell_not_licensed");
    assert!(
        message.starts_with("sensitivity_decision_composition.composition_not_licensed"),
        "{message}"
    );
}

#[test]
fn b3_msm_adapter_refuses_malformed_surfaces() {
    let result = result(2.0, 3);
    // A point quantity with the wrong number of grid values.
    let short = (scientific("cost"), vec![0.0, 0.0]);
    let error = msm_sensitivity_artifact(&result, &scientific("ate"), &[short], net_of_cost(), "c")
        .unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "invalid_argument");
    assert!(message.starts_with("sensitivity_decision_composition.invalid_surface"), "{message}");
    // A single action is not a decision.
    let one = vec![act("treat", UtilityTerm::quantity("ate"))];
    let error = msm_sensitivity_artifact(&result, &scientific("ate"), &[], one, "c").unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "decision_contract_unsatisfied");
    assert!(message.starts_with("sensitivity_decision_composition.wrong_contract"), "{message}");
}
