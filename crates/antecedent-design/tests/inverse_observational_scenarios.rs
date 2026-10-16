//! Observational scenarios compatible with an outcome event, against a hand-enumerated
//! finite law.
//!
//! Eight observed units `(x, z, y)`:
//!
//! ```text
//! row  x  z  y
//!  1   0  0  0
//!  2   0  0  1
//!  3   0  1  0
//!  4   0  1  0
//!  5   1  0  1
//!  6   1  0  1
//!  7   1  1  0
//!  8   1  1  1
//! ```
//!
//! With equal weights the event `Y >= 1` has mass `4/8` (rows 2, 5, 6, 8). The scenario
//! `x = 1` holds rows 5-8 (mass `1/2`), contains the event in rows 5, 6, 8 (joint mass
//! `3/8`), so `P(E | S) = 3/4`, `P(S | E) = 3/4` and the lift is `3/2`. The scenario
//! `x = 0 and z = 1` holds rows 3, 4 (mass `1/4`) and none of the event. The scenario
//! `z = 0` holds rows 1, 2, 5, 6 and the event in rows 2, 5, 6.
//!
//! With weights `(1, 1, 1, 1, 2, 2, 0, 4) / 12` the event mass is `9/12`; `x = 1` has mass
//! `8/12` (row 7 carries no weight), `P(E | x = 1) = 1`, `P(x = 1 | E) = 8/9`, and the Kish
//! effective number of event draws is `9^2 / (1 + 4 + 4 + 16) = 81/25`.

#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "dyadic oracles; one scenario per test"
)]

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_design::inverse_observational_scenarios::{
    DETAIL_CONDITIONS_ON_OUTCOME, DETAIL_COORDINATE_MISSING, DETAIL_COORDINATE_UNSUPPORTED,
    DETAIL_EVENT_MASS, DETAIL_JOINT_LAW, DETAIL_MEANING, DETAIL_PARAMETER, DETAIL_SCENARIOS,
    ObservationalScenarioError, ObservationalScenarioQuery, OutcomeEvent, ScenarioCell,
    ScenarioCondition, evaluate_observational_scenarios, verify_observational_scenarios,
};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

const X: [f64; 8] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
const Z: [f64; 8] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
const Y: [f64; 8] = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];

fn covariate(name: &str) -> ScientificQuantity {
    ScientificQuantity { role: QuantityRole::Covariate, ..common::quantity(name) }
}

fn outcome() -> ScientificQuantity {
    common::quantity("y")
}

struct Build {
    meaning: DistributionMeaningWire,
    alignment: DrawAlignment,
    weights: Option<Vec<f64>>,
    supported: Option<Vec<bool>>,
    calibration: DistributionCalibration,
    y: [f64; 8],
}

impl Default for Build {
    fn default() -> Self {
        Self {
            meaning: DistributionMeaningWire::EmpiricalOutcome,
            alignment: DrawAlignment::Joint,
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            y: Y,
        }
    }
}

fn law(build: &Build) -> DistributionArtifact {
    let columns = [covariate("x"), covariate("z"), outcome()];
    let identity = DistributionIdentity::new(
        build.meaning,
        &columns,
        build.alignment,
        DistributionProvenance {
            source_id: "observed-units".into(),
            provider_id: "enumerated".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "snapshot-1".into(),
            causal_contract_id: "contract-1".into(),
        },
    )
    .unwrap();
    let mut draws = Vec::with_capacity(24);
    for row in 0..8 {
        draws.extend([X[row], Z[row], build.y[row]]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [8, 3],
            weights: build.weights.clone(),
            supported: build.supported.clone(),
            calibration: build.calibration,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn condition(name: &str, lower: f64, upper: f64) -> ScenarioCondition {
    ScenarioCondition { coordinate: covariate(name), lower, upper }
}

fn cell(id: &str, conditions: Vec<ScenarioCondition>) -> ScenarioCell {
    ScenarioCell { id: id.into(), conditions }
}

fn query() -> ObservationalScenarioQuery {
    ObservationalScenarioQuery {
        outcome: outcome(),
        event: OutcomeEvent::AtLeast(1.0),
        scenarios: vec![
            cell("x_is_1", vec![condition("x", 1.0, 1.0)]),
            cell("x0_z1", vec![condition("x", 0.0, 0.0), condition("z", 1.0, 1.0)]),
            cell("z_is_0", vec![condition("z", 0.0, 0.0)]),
        ],
    }
}

#[test]
fn scenario_values_match_the_hand_enumeration() {
    let result = evaluate_observational_scenarios(&query(), &law(&Build::default())).unwrap();
    assert!((result.event_mass - 0.5).abs() < 1e-12);
    assert_eq!(result.n_draws, 8);
    assert!((result.effective_event_draws - 4.0).abs() < 1e-12);
    let x1 = &result.scenarios[0];
    assert_eq!(x1.id, "x_is_1");
    assert!((x1.mass - 0.5).abs() < 1e-12);
    assert!((x1.joint_mass - 0.375).abs() < 1e-12);
    assert!((x1.event_given_scenario.unwrap() - 0.75).abs() < 1e-12);
    assert!((x1.scenario_given_event - 0.75).abs() < 1e-12);
    assert!((x1.lift.unwrap() - 1.5).abs() < 1e-12);
    assert_eq!((x1.scenario_draws, x1.joint_draws), (4, 3));
    let none = &result.scenarios[1];
    assert!((none.mass - 0.25).abs() < 1e-12);
    assert_eq!(none.joint_mass, 0.0);
    assert_eq!(none.event_given_scenario, Some(0.0));
    assert_eq!(none.scenario_given_event, 0.0);
    assert_eq!(none.lift, Some(0.0));
    let z0 = &result.scenarios[2];
    assert!((z0.mass - 0.5).abs() < 1e-12);
    assert!((z0.joint_mass - 0.375).abs() < 1e-12);
    // Rows 2, 5, 6 and 8 are in the event; rows 2, 5, 6 are in a declared scenario and row 8
    // is in `x_is_1`, so the scenarios cover the whole event.
    assert!((result.covered_event_mass - 0.5).abs() < 1e-12);
    assert_eq!(result.uncovered_event_mass, 0.0);
    // An association, not an action: nothing is ranked and nothing is an effect.
    assert!(!result.ranks_actions);
    assert!(!result.is_intervention_effect);
    assert!(result.exact);
    assert_eq!(result.standard_error, Some(0.0));
    verify_observational_scenarios(&result, &query(), &law(&Build::default())).unwrap();
}

#[test]
fn weights_and_a_zero_weight_row_follow_the_oracle() {
    let build = Build {
        weights: Some(vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 4.0]),
        calibration: DistributionCalibration::PointOnly,
        ..Build::default()
    };
    let only_x = ObservationalScenarioQuery {
        scenarios: vec![cell("x_is_1", vec![condition("x", 1.0, 1.0)])],
        ..query()
    };
    let result = evaluate_observational_scenarios(&only_x, &law(&build)).unwrap();
    assert!((result.event_mass - 0.75).abs() < 1e-12);
    let x1 = &result.scenarios[0];
    assert!((x1.mass - 8.0 / 12.0).abs() < 1e-12);
    assert!((x1.event_given_scenario.unwrap() - 1.0).abs() < 1e-12);
    assert!((x1.scenario_given_event - 8.0 / 9.0).abs() < 1e-12);
    assert!((result.effective_event_draws - 81.0 / 25.0).abs() < 1e-12);
    // Row 7 carries no weight: it is not counted as a draw of the scenario.
    assert_eq!(x1.scenario_draws, 3);
    // A sampled law claims no calibrated Monte-Carlo error.
    assert!(!result.exact);
    assert_eq!(result.standard_error, None);
    // Event mass outside every declared scenario is reported, not renormalized away.
    let uncovered = ObservationalScenarioQuery {
        scenarios: vec![cell("x0_z1", vec![condition("x", 0.0, 0.0), condition("z", 1.0, 1.0)])],
        ..query()
    };
    let result = evaluate_observational_scenarios(&uncovered, &law(&build)).unwrap();
    assert_eq!(result.covered_event_mass, 0.0);
    assert!((result.uncovered_event_mass - 0.75).abs() < 1e-12);
}

#[test]
fn an_at_most_event_inverts_the_tail() {
    let query = ObservationalScenarioQuery { event: OutcomeEvent::AtMost(0.0), ..query() };
    let result = evaluate_observational_scenarios(&query, &law(&Build::default())).unwrap();
    // Rows 1, 3, 4, 7 have y = 0.
    assert!((result.event_mass - 0.5).abs() < 1e-12);
    // x = 1: row 7 only, so P(E | x = 1) = 1/4 and P(x = 1 | E) = 1/4.
    let x1 = &result.scenarios[0];
    assert!((x1.event_given_scenario.unwrap() - 0.25).abs() < 1e-12);
    assert!((x1.scenario_given_event - 0.25).abs() < 1e-12);
}

#[test]
fn the_identity_follows_the_numbers_at_an_unchanged_shape() {
    let base = evaluate_observational_scenarios(&query(), &law(&Build::default())).unwrap();
    let mut y = Y;
    y[0] = 1.0;
    let changed =
        evaluate_observational_scenarios(&query(), &law(&Build { y, ..Build::default() })).unwrap();
    assert_ne!(base.identity, changed.identity);
    // A stored result that no longer replays is refused.
    let mut forged = base.clone();
    forged.scenarios[0].event_given_scenario = Some(0.9);
    let error =
        verify_observational_scenarios(&forged, &query(), &law(&Build::default())).unwrap_err();
    assert_eq!(error, ObservationalScenarioError::ResultMismatch);
}

#[test]
fn what_the_law_cannot_answer_is_refused_with_a_typed_reason() {
    let observed = law(&Build::default());
    // Independent marginals carry no pairing between a covariate and the outcome.
    let marginals =
        law(&Build { alignment: DrawAlignment::IndependentMarginals, ..Build::default() });
    let error = evaluate_observational_scenarios(&query(), &marginals).unwrap_err();
    assert_eq!(error, ObservationalScenarioError::JointLawRequired("independent_marginals"));
    assert_eq!(error.code(), "joint_law_required");
    assert_eq!(error.detail(), DETAIL_JOINT_LAW);
    assert!(error.to_refusal().remedy.is_some());
    // An interventional law answers an intervention, not an observed population.
    let interventional = law(&Build {
        meaning: DistributionMeaningWire::InterventionalPredictive,
        ..Build::default()
    });
    let error = evaluate_observational_scenarios(&query(), &interventional).unwrap_err();
    assert_eq!(
        error,
        ObservationalScenarioError::MeaningNotObservational("interventional_predictive")
    );
    assert_eq!(error.code(), "distribution_meaning_mismatch");
    assert_eq!(error.detail(), DETAIL_MEANING);
    // The conditional is undefined where the event has no mass.
    let unreachable = ObservationalScenarioQuery { event: OutcomeEvent::AtLeast(5.0), ..query() };
    let error = evaluate_observational_scenarios(&unreachable, &observed).unwrap_err();
    assert_eq!(error, ObservationalScenarioError::EventHasNoMass);
    assert_eq!(error.detail(), DETAIL_EVENT_MASS);
    // A scenario may not condition on the outcome itself.
    let circular = ObservationalScenarioQuery {
        scenarios: vec![ScenarioCell {
            id: "high_outcome".into(),
            conditions: vec![ScenarioCondition { coordinate: outcome(), lower: 1.0, upper: 1.0 }],
        }],
        ..query()
    };
    let error = evaluate_observational_scenarios(&circular, &observed).unwrap_err();
    assert!(matches!(error, ObservationalScenarioError::ConditionsOnOutcome(_)));
    assert_eq!(error.detail(), DETAIL_CONDITIONS_ON_OUTCOME);
    // A coordinate the law does not have, and one it masks as unsupported.
    let unknown = ObservationalScenarioQuery {
        scenarios: vec![cell("age", vec![condition("age", 0.0, 1.0)])],
        ..query()
    };
    let error = evaluate_observational_scenarios(&unknown, &observed).unwrap_err();
    assert!(matches!(error, ObservationalScenarioError::CoordinateMissing(_)));
    assert_eq!(error.detail(), DETAIL_COORDINATE_MISSING);
    let masked = law(&Build { supported: Some(vec![true, false, true]), ..Build::default() });
    let error = evaluate_observational_scenarios(&query(), &masked).unwrap_err();
    assert!(matches!(error, ObservationalScenarioError::CoordinateUnsupported(_)));
    assert_eq!(error.detail(), DETAIL_COORDINATE_UNSUPPORTED);
}

#[test]
fn malformed_declarations_are_refused_before_the_law_is_read() {
    let observed = law(&Build::default());
    let with = |scenarios: Vec<ScenarioCell>| ObservationalScenarioQuery { scenarios, ..query() };
    for (bad, detail) in [
        (with(vec![]), DETAIL_SCENARIOS),
        (
            with(vec![
                cell("s", vec![condition("x", 0.0, 0.0)]),
                cell("s", vec![condition("z", 0.0, 0.0)]),
            ]),
            DETAIL_SCENARIOS,
        ),
        (with(vec![cell("  ", vec![condition("x", 0.0, 0.0)])]), DETAIL_SCENARIOS),
        (with(vec![cell("empty", vec![])]), DETAIL_SCENARIOS),
        (with(vec![cell("inverted", vec![condition("x", 1.0, 0.0)])]), DETAIL_PARAMETER),
        (with(vec![cell("nan", vec![condition("x", f64::NAN, 1.0)])]), DETAIL_PARAMETER),
        (
            ObservationalScenarioQuery { event: OutcomeEvent::AtLeast(f64::INFINITY), ..query() },
            DETAIL_PARAMETER,
        ),
    ] {
        let error = evaluate_observational_scenarios(&bad, &observed).unwrap_err();
        assert_eq!(error.detail(), detail, "{error}");
        assert_eq!(error.code(), "invalid_argument");
    }
}
