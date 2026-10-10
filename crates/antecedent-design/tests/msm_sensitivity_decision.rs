//! 2.3 B3: a marginal sensitivity model surface consumed by `evaluate_sensitivity_decision`.
//!
//! Same two-stratum inputs and hand expectations as
//! `antecedent-io/tests/msm_sensitivity_adapter.rs` (ATE identified 0.3, bounds
//! [0.04375, 0.4875] at Lambda = 2 and about [-0.0833, 0.55] at Lambda = 3):
//!
//! * invariant: `treat = ate` against `skip = 0` on Lambda in {1, 1.5, 2};
//! * switch: `treat = ate - cost` against `skip = 0`, `cost = (0, 0, 2)` on {1, 2, 3},
//!   switching in [2, 3] because the ATE is at most 1;
//! * no robust action: `treat = ate` against `skip = 0.2` on {1, 2, 3}; mixed at {2, 3}.

use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_contract::StructuralPolicy;
use antecedent_design::decision_structural::StructuralVerdict;
use antecedent_design::sensitivity_decision::{
    ActionUtility, DecisionSwitch, NoRobustReason, SamplingReport, ScenarioCoverage,
    SensitivityArtifact, SensitivityDecisionSpec, SensitivityOutcome, UtilityTerm,
    contract_from_artifact, evaluate_sensitivity_decision,
};
use antecedent_io::msm_sensitivity_adapter::{
    MsmOutcomeLaw, MsmSensitivityResult, MsmSensitivitySpec, MsmStratum, msm_ate_sensitivity,
    msm_sensitivity_artifact,
};

fn binary(rate: f64) -> MsmOutcomeLaw {
    MsmOutcomeLaw { values: vec![0.0, 1.0], probabilities: vec![1.0 - rate, rate] }
}

fn result(lambda_max: f64, grid_points: usize) -> MsmSensitivityResult {
    let strata = vec![
        MsmStratum { mass: 0.5, propensity: 0.5, treated: binary(0.8), control: binary(0.4) },
        MsmStratum { mass: 0.5, propensity: 0.25, treated: binary(0.5), control: binary(0.3) },
    ];
    let mut spec = MsmSensitivitySpec::new(lambda_max);
    spec.grid_points = grid_points;
    msm_ate_sensitivity(&strata, &spec).unwrap()
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

fn evaluate(
    artifact: &SensitivityArtifact,
) -> antecedent_design::sensitivity_decision::SensitivityDecisionResult {
    let contract =
        contract_from_artifact(artifact, StructuralPolicy::RequireInvariantBestAction).unwrap();
    evaluate_sensitivity_decision(&contract, artifact, &SensitivityDecisionSpec::default()).unwrap()
}

fn treat_vs(result: &MsmSensitivityResult, skip: f64) -> SensitivityArtifact {
    let actions =
        vec![act("treat", UtilityTerm::quantity("ate")), act("skip", UtilityTerm::Const(skip))];
    msm_sensitivity_artifact(result, &scientific("ate"), &[], actions, "checked-contract").unwrap()
}

#[test]
fn b3_msm_decision_invariant_action() {
    let built = treat_vs(&result(2.0, 3), 0.0);
    let decided = evaluate(&built);
    assert_eq!(decided.outcome, SensitivityOutcome::InvariantAction { action: "treat".into() });
    assert_eq!(decided.structural.verdict, StructuralVerdict::InvariantBest("treat".into()));
    assert_eq!(decided.coordinates, vec![1.0, 1.5, 2.0]);
    // Two range vertices per grid point, and the utilities are multilinear.
    assert_eq!(decided.structural.atoms.len(), 6);
    assert_eq!(decided.coverage, ScenarioCoverage::VertexCertified);
    assert_eq!(built.decision_outcome(None).unwrap(), decided.outcome);
}

#[test]
fn b3_msm_decision_assumption_dependent_switch() {
    let result = result(3.0, 3);
    let net = UtilityTerm::difference(UtilityTerm::quantity("ate"), UtilityTerm::quantity("cost"));
    let actions = vec![act("treat", net), act("skip", UtilityTerm::Const(0.0))];
    let cost = (scientific("cost"), vec![0.0, 0.0, 2.0]);
    let built =
        msm_sensitivity_artifact(&result, &scientific("ate"), &[cost], actions, "checked-contract")
            .unwrap();
    let decided = evaluate(&built);
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
    assert_eq!(decided.outcome, expected);
    assert_eq!(built.decision_outcome(None).unwrap(), decided.outcome);
}

#[test]
fn b3_msm_decision_no_robust_action() {
    let built = treat_vs(&result(3.0, 3), 0.2);
    let decided = evaluate(&built);
    let expected = SensitivityOutcome::NoRobustAction {
        reason: NoRobustReason::MixedWithinRange { coordinates: vec![2.0, 3.0] },
        switches: vec![],
    };
    assert_eq!(decided.outcome, expected);
    assert_eq!(built.decision_outcome(None).unwrap(), decided.outcome);
}

#[test]
fn b3_msm_decision_keeps_sampling_withheld_and_refuses_composition() {
    let built = treat_vs(&result(2.0, 3), 0.0);
    let decided = evaluate(&built);
    let SamplingReport::Withheld { reason_code, detail } = &decided.sampling else {
        panic!("the sampling interval must stay withheld");
    };
    assert_eq!(reason_code, "cell_not_licensed");
    assert_eq!(detail, "msm_sensitivity.interval_withheld");
    let contract =
        contract_from_artifact(&built, StructuralPolicy::RequireInvariantBestAction).unwrap();
    let spec = SensitivityDecisionSpec {
        sampling_composition: Some("sum_of_interval_and_range".into()),
        ..SensitivityDecisionSpec::default()
    };
    let error = evaluate_sensitivity_decision(&contract, &built, &spec).unwrap_err();
    assert_eq!(error.reason_code(), "cell_not_licensed");
    assert_eq!(error.detail(), "sensitivity_decision_composition.composition_not_licensed");
}
