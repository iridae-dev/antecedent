//! F17: an assumption-sensitivity surface consumed by a `DecisionContract`.
//!
//! Every expected value is derived by hand from the declared surfaces:
//!
//! * frozen surface (record `2.3B.F17.sensitivity_decision_composition`): `gamma` in
//!   `{0, 1, 2}`, action A has utility `2 - gamma` (2, 1, 0) and action B has utility 1,
//!   so A leads at 0, they tie at 1 and B leads at 2;
//! * invariant surface: A is `5 - gamma` (5, 4, 3) against B = 1, so A leads everywhere;
//! * closed-form tipping: grid `{0, 0.5, 1.5, 2}`, A = `2 - gamma`, B = 1, so the
//!   difference `1 - gamma` is zero at `gamma = 1`, between the grid points 0.5 and 1.5;
//! * no robust action: an effect whose range is `[-1, 2]` at every `gamma` against a status
//!   quo of 0 (the best action depends on where the range sits), and three actions that
//!   each lead at one of three points.
#![allow(clippy::float_cmp, reason = "exact hand-derived grid values")]

use antecedent_core::{QuantityRole, RegimeId, ScientificQuantity};
use antecedent_design::decision_contract::{DecisionCriterion, StructuralPolicy};
use antecedent_design::decision_structural::StructuralVerdict;
use antecedent_design::sensitivity_decision::{
    ActionUtility, AssumptionCoordinate, AssumptionRangeStatement, DecisionSwitch,
    LICENSED_SAMPLING_COMPOSITIONS, NoRobustReason, PointSupport, SamplingInterval, SamplingReport,
    SamplingStatus, ScenarioCoverage, SensitivityArtifact, SensitivityDecisionSpec,
    SensitivityOutcome, SensitivityParts, SensitivityProvenance, SurfaceQuantity,
    UncertaintyRelationship, UtilityTerm, contract_from_artifact, evaluate_sensitivity_decision,
};
use antecedent_io::error::IoError;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::sensitivity_artifact::{
    AxisTipping, JointFactor, JointFactorBound, JointMechanismSensitivityResult,
    JointSensitivityLimits, JointSensitivityRange, JointSensitivityReceipt,
    JointSensitivityUncertainty, TippingBracket, TippingStatus,
};

fn scientific(name: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "sensitivity_surface".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn surface(name: &str, units: &str, lower: &[f64], upper: &[f64]) -> SurfaceQuantity {
    SurfaceQuantity {
        quantity: ScientificQuantityWire::from(&scientific(name, units)),
        lower: lower.to_vec(),
        upper: upper.to_vec(),
    }
}

fn point(name: &str, values: &[f64]) -> SurfaceQuantity {
    surface(name, "utils", values, values)
}

fn act(id: &str, quantity: &str) -> ActionUtility {
    ActionUtility { id: id.into(), utility: UtilityTerm::quantity(quantity) }
}

fn withheld() -> UncertaintyRelationship {
    UncertaintyRelationship {
        assumption_range: AssumptionRangeStatement {
            kind: "assumption_range".into(),
            interpretation: "assumption range; not a probability".into(),
        },
        identified_bound: None,
        sampling: SamplingStatus::Withheld {
            reason_code: "cell_not_licensed".into(),
            detail: "joint_sensitivity.interval_withheld".into(),
        },
    }
}

fn parts(
    grid: &[f64],
    quantities: Vec<SurfaceQuantity>,
    actions: Vec<ActionUtility>,
) -> SensitivityParts {
    let maximum = grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    SensitivityParts {
        coordinate: AssumptionCoordinate {
            id: "gamma".into(),
            scale: "sensitivity_parameter".into(),
            units: "dimensionless".into(),
            minimum: 0.0,
            maximum,
        },
        grid: grid.to_vec(),
        support: vec![PointSupport::Supported; grid.len()],
        quantities,
        actions,
        uncertainty: withheld(),
        provenance: SensitivityProvenance {
            source_kind: "supplied_surface".into(),
            query_binding: "f17-test".into(),
            provider_snapshot: "snapshot-1".into(),
            source_regime: "regime:1".into(),
            method: "hand-derived surface".into(),
            causal_contract_id: "checked-contract".into(),
            decision_threshold: None,
            source_tipping: vec![],
        },
    }
}

fn frozen_parts() -> SensitivityParts {
    parts(
        &[0.0, 1.0, 2.0],
        vec![point("ua", &[2.0, 1.0, 0.0]), point("ub", &[1.0, 1.0, 1.0])],
        vec![act("A", "ua"), act("B", "ub")],
    )
}

fn invariant_policy() -> StructuralPolicy {
    StructuralPolicy::RequireInvariantBestAction
}

/// Evaluate the artifact's declared actions under the invariance policy.
fn evaluate(
    artifact: &SensitivityArtifact,
) -> antecedent_design::sensitivity_decision::SensitivityDecisionResult {
    let contract = contract_from_artifact(artifact, invariant_policy()).unwrap();
    evaluate_sensitivity_decision(&contract, artifact, &SensitivityDecisionSpec::default()).unwrap()
}

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1e-12, "{actual} vs {expected}");
}

fn refusal(error: &IoError) -> (&'static str, &str) {
    match error {
        IoError::Refused { code, message } => (*code, message.as_str()),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn f17_invariant_action_across_the_whole_assumption_range() {
    // A = 5 - gamma = (5, 4, 3) beats B = 1 at every gamma.
    let artifact = SensitivityArtifact::new(parts(
        &[0.0, 1.0, 2.0],
        vec![point("ua", &[5.0, 4.0, 3.0]), point("ub", &[1.0, 1.0, 1.0])],
        vec![act("A", "ua"), act("B", "ub")],
    ))
    .unwrap();
    let result = evaluate(&artifact);
    assert_eq!(result.outcome, SensitivityOutcome::InvariantAction { action: "A".into() });
    assert_eq!(result.structural.verdict, StructuralVerdict::InvariantBest("A".into()));
    assert_eq!(result.structural.atoms.len(), 3);
    assert_eq!(result.coverage, ScenarioCoverage::PointSurface);
    assert_eq!(result.coordinates, vec![0.0, 1.0, 2.0]);
    // The artifact recomputes the same decision from its stored surface alone.
    assert_eq!(artifact.decision_outcome(None).unwrap(), result.outcome);
    assert_eq!(artifact.outcome(), Some(&result.outcome));
}

#[test]
fn f17_frozen_surface_switches_with_an_exact_tie_at_gamma_one() {
    let artifact = SensitivityArtifact::new(frozen_parts()).unwrap();
    let result = evaluate(&artifact);
    let expected = DecisionSwitch {
        from: vec!["A".into()],
        to: vec!["B".into()],
        lower: 1.0,
        upper: 1.0,
        exact: true,
        interpolated: None,
    };
    assert_eq!(result.outcome, SensitivityOutcome::AssumptionDependent { switch: expected });
    // No invariant unique action over the full range.
    let StructuralVerdict::NoInvariantBest(leaders) = &result.structural.verdict else {
        panic!("expected no invariant best action, got {:?}", result.structural.verdict);
    };
    assert_eq!(
        leaders,
        &vec![
            ("gamma[0]#0".to_owned(), vec!["A".to_owned()]),
            ("gamma[1]#0".to_owned(), vec!["A".to_owned(), "B".to_owned()]),
            ("gamma[2]#0".to_owned(), vec!["B".to_owned()]),
        ]
    );
    assert_eq!(artifact.decision_outcome(None).unwrap(), result.outcome);
}

#[test]
fn f17_tipping_coordinate_is_the_closed_form_crossing() {
    // 1 - gamma changes sign between 0.5 (+0.5) and 1.5 (-0.5); linear, so gamma* = 1.
    let artifact = SensitivityArtifact::new(parts(
        &[0.0, 0.5, 1.5, 2.0],
        vec![point("ua", &[2.0, 1.5, 0.5, 0.0]), point("ub", &[1.0, 1.0, 1.0, 1.0])],
        vec![act("A", "ua"), act("B", "ub")],
    ))
    .unwrap();
    let result = evaluate(&artifact);
    let SensitivityOutcome::AssumptionDependent { switch } = &result.outcome else {
        panic!("expected a switch, got {:?}", result.outcome);
    };
    assert_eq!(switch.from, vec!["A".to_owned()]);
    assert_eq!(switch.to, vec!["B".to_owned()]);
    assert_eq!((switch.lower, switch.upper, switch.exact), (0.5, 1.5, false));
    assert_close(switch.interpolated.unwrap(), 1.0);
    assert_eq!(artifact.decision_outcome(None).unwrap(), result.outcome);
}

#[test]
fn f17_no_robust_action_when_the_range_straddles_at_every_point() {
    // The effect lies in [-1, 2] at every gamma; the status quo is 0. At each gamma the
    // lower vertex favors B and the upper vertex favors A.
    let artifact = SensitivityArtifact::new(parts(
        &[0.0, 1.0, 2.0],
        vec![
            surface("effect", "utils", &[-1.0; 3], &[2.0; 3]),
            point("status_quo", &[0.0, 0.0, 0.0]),
        ],
        vec![act("A", "effect"), act("B", "status_quo")],
    ))
    .unwrap();
    let result = evaluate(&artifact);
    assert_eq!(
        result.outcome,
        SensitivityOutcome::NoRobustAction {
            reason: NoRobustReason::MixedWithinRange { coordinates: vec![0.0, 1.0, 2.0] },
            switches: vec![],
        }
    );
    assert!(matches!(result.structural.verdict, StructuralVerdict::NoInvariantBest(_)));
    // Two vertex scenarios per grid point, both multilinear utilities.
    assert_eq!(result.structural.atoms.len(), 6);
    assert_eq!(result.coverage, ScenarioCoverage::VertexCertified);
    assert_eq!(artifact.decision_outcome(None).unwrap(), result.outcome);
}

#[test]
fn f17_alternating_leaders_give_no_robust_action() {
    // A, B and C each lead at exactly one of the three points: two switches.
    let artifact = SensitivityArtifact::new(parts(
        &[0.0, 1.0, 2.0],
        vec![
            point("qa", &[3.0, 0.0, 0.0]),
            point("qb", &[0.0, 3.0, 0.0]),
            point("qc", &[0.0, 0.0, 3.0]),
        ],
        vec![act("A", "qa"), act("B", "qb"), act("C", "qc")],
    ))
    .unwrap();
    let result = evaluate(&artifact);
    let SensitivityOutcome::NoRobustAction { reason, switches } = &result.outcome else {
        panic!("expected no robust action, got {:?}", result.outcome);
    };
    assert_eq!(reason, &NoRobustReason::MultipleSwitches);
    assert_eq!(switches.len(), 2);
    // A - B is 3 at gamma 0 and -3 at gamma 1: crossing at 0.5; B - C likewise at 1.5.
    assert_close(switches[0].interpolated.unwrap(), 0.5);
    assert_close(switches[1].interpolated.unwrap(), 1.5);
    assert!(matches!(result.structural.verdict, StructuralVerdict::NoInvariantBest(_)));
}

#[test]
fn f17_assumption_range_is_not_a_sampling_interval() {
    // The effect's assumption range is [0.5, 1.5], [0.4, 1.4], [0.3, 1.3] against a status
    // quo of 0, so A leads everywhere. The sampling interval [-2, 3] is far wider and, if
    // it were merged into the range, would flip the decision; it must not.
    let ranged = |uncertainty: UncertaintyRelationship| {
        let mut p = parts(
            &[0.0, 1.0, 2.0],
            vec![
                surface("effect", "utils", &[0.5, 0.4, 0.3], &[1.5, 1.4, 1.3]),
                point("status_quo", &[0.0, 0.0, 0.0]),
            ],
            vec![act("A", "effect"), act("B", "status_quo")],
        );
        p.uncertainty = uncertainty;
        SensitivityArtifact::new(p).unwrap()
    };
    let mut with_interval = withheld();
    with_interval.sampling = SamplingStatus::Reported(SamplingInterval {
        quantity: "effect".into(),
        level: 0.95,
        method: "percentile_bootstrap".into(),
        composed: None,
        lower: vec![-2.0; 3],
        upper: vec![3.0; 3],
    });
    let plain = ranged(withheld());
    let reported = ranged(with_interval.clone());
    let (without, with) = (evaluate(&plain), evaluate(&reported));
    assert_eq!(without.outcome, SensitivityOutcome::InvariantAction { action: "A".into() });
    assert_eq!(with.outcome, without.outcome);
    assert_eq!(with.structural.atoms.len(), without.structural.atoms.len());
    assert_ne!(plain.identity().digest, reported.identity().digest);
    assert!(matches!(without.sampling, SamplingReport::Withheld { .. }));
    assert_eq!(
        with.sampling,
        SamplingReport::SeparateNotComposed {
            quantity: "effect".into(),
            level: 0.95,
            method: "percentile_bootstrap".into(),
            coordinates: vec![0.0, 1.0, 2.0],
            lower: vec![-2.0; 3],
            upper: vec![3.0; 3],
        }
    );

    // Asking for the composition is refused; the licensed list is empty.
    assert!(LICENSED_SAMPLING_COMPOSITIONS.is_empty());
    let contract = contract_from_artifact(&reported, invariant_policy()).unwrap();
    let spec = SensitivityDecisionSpec {
        sampling_composition: Some("conservative_endpoint_percentile_bootstrap".into()),
        ..SensitivityDecisionSpec::default()
    };
    let error = evaluate_sensitivity_decision(&contract, &reported, &spec).unwrap_err();
    assert_eq!(error.reason_code(), "cell_not_licensed");
    assert_eq!(error.detail(), "sensitivity_decision_composition.composition_not_licensed");

    // An artifact that claims a composition is refused at construction.
    let mut claimed = with_interval;
    if let SamplingStatus::Reported(interval) = &mut claimed.sampling {
        interval.composed = Some("conservative_endpoint_percentile_bootstrap".into());
    }
    let mut p = frozen_parts();
    p.uncertainty = claimed;
    p.uncertainty.sampling = match p.uncertainty.sampling {
        SamplingStatus::Reported(interval) => {
            SamplingStatus::Reported(SamplingInterval { quantity: "ua".into(), ..interval })
        }
        other @ SamplingStatus::Withheld { .. } => other,
    };
    let error = SensitivityArtifact::new(p).unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "cell_not_licensed");
    assert!(message.starts_with("sensitivity_decision_composition.composition_not_licensed"));
}

#[test]
fn f17_unsupported_coordinate_refuses_and_a_subrange_avoids_it() {
    let mut p = frozen_parts();
    p.support = vec![PointSupport::Supported, PointSupport::Unsupported, PointSupport::Supported];
    let artifact = SensitivityArtifact::new(p).unwrap();
    assert_eq!(artifact.outcome(), None);
    let contract = contract_from_artifact(&artifact, invariant_policy()).unwrap();

    let error =
        evaluate_sensitivity_decision(&contract, &artifact, &SensitivityDecisionSpec::default())
            .unwrap_err();
    assert_eq!(error.reason_code(), "quantity_semantics_mismatch");
    assert_eq!(error.detail(), "sensitivity_decision_composition.unsupported_coordinate");
    let spec = SensitivityDecisionSpec { range: Some((0.0, 2.0)), ..Default::default() };
    assert!(evaluate_sensitivity_decision(&contract, &artifact, &spec).is_err());

    // Evaluating only gamma = 0, where A = 2 beats B = 1, never touches the bad point.
    let spec = SensitivityDecisionSpec { range: Some((0.0, 0.0)), ..Default::default() };
    let result = evaluate_sensitivity_decision(&contract, &artifact, &spec).unwrap();
    assert_eq!(result.outcome, SensitivityOutcome::InvariantAction { action: "A".into() });

    // A grid point outside the declared range is an unsupported coordinate too.
    let mut outside = frozen_parts();
    outside.grid = vec![0.0, 1.0, 3.0];
    let error = SensitivityArtifact::new(outside).unwrap_err();
    assert_eq!(refusal(&error).0, "quantity_semantics_mismatch");
}

#[test]
fn f17_unevaluated_point_leaves_the_decision_unresolved() {
    let mut p = frozen_parts();
    p.support = vec![PointSupport::Supported, PointSupport::Unevaluated, PointSupport::Supported];
    let artifact = SensitivityArtifact::new(p).unwrap();
    assert_eq!(artifact.outcome(), Some(&SensitivityOutcome::Unresolved { coordinate: 1.0 }));
    let result = evaluate(&artifact);
    assert_eq!(result.outcome, SensitivityOutcome::Unresolved { coordinate: 1.0 });
    assert!(matches!(result.structural.verdict, StructuralVerdict::InsufficientScience(_)));
}

#[test]
fn f17_mixed_estimands_and_foreign_inputs_refuse() {
    // A in dollars against B in quality-adjusted years cannot be compared.
    let mut unlike = frozen_parts();
    unlike.quantities = vec![
        surface("ua", "usd", &[2.0, 1.0, 0.0], &[2.0, 1.0, 0.0]),
        surface("ub", "qaly", &[1.0; 3], &[1.0; 3]),
    ];
    let error = SensitivityArtifact::new(unlike).unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "decision_contract_unsatisfied");
    assert!(message.starts_with("sensitivity_decision_composition.wrong_contract"));
    assert!(message.contains("unlike units"));

    // One utility mixing the two is refused as mixed estimands.
    let mut within = frozen_parts();
    within.quantities = vec![
        surface("ua", "usd", &[2.0, 1.0, 0.0], &[2.0, 1.0, 0.0]),
        surface("ub", "qaly", &[1.0; 3], &[1.0; 3]),
    ];
    within.actions = vec![
        ActionUtility {
            id: "A".into(),
            utility: UtilityTerm::difference(
                UtilityTerm::quantity("ua"),
                UtilityTerm::quantity("ub"),
            ),
        },
        act("B", "ub"),
    ];
    let error = SensitivityArtifact::new(within).unwrap_err();
    assert!(refusal(&error).1.contains("mixed estimands"));

    // A contract with other units, an input the surface lacks or an undefined criterion.
    let artifact = SensitivityArtifact::new(frozen_parts()).unwrap();
    let base = contract_from_artifact(&artifact, invariant_policy()).unwrap();
    let spec = SensitivityDecisionSpec::default();
    let mut units = base.clone();
    units.utility_units = "qaly".into();
    let mut foreign = base.clone();
    foreign.actions[0].inputs[0].regime_id = "do(a=0)".into();
    let mut criterion = base;
    criterion.criterion = DecisionCriterion::Quantile { p: 0.5 };
    for contract in [units, foreign, criterion] {
        let error = evaluate_sensitivity_decision(&contract, &artifact, &spec).unwrap_err();
        assert_eq!(error.reason_code(), "decision_contract_unsatisfied");
        assert_eq!(error.detail(), "sensitivity_decision_composition.wrong_contract");
    }
}

#[test]
fn f17_weights_are_declared_for_point_surfaces_and_never_inferred_from_ranges() {
    let artifact = SensitivityArtifact::new(frozen_parts()).unwrap();
    let bayes = contract_from_artifact(&artifact, StructuralPolicy::BayesOverStructures).unwrap();

    // Bayes weighting without declared genuine weights refuses.
    let none = SensitivityDecisionSpec::default();
    let error = evaluate_sensitivity_decision(&bayes, &artifact, &none).unwrap_err();
    assert_eq!(error.reason_code(), "decision_contract_unsatisfied");

    // Declared weights 0.5, 0.25, 0.25: A is 1.0 + 0.25 + 0 = 1.25, B is 1.0.
    let weighted =
        SensitivityDecisionSpec { weights: Some(vec![0.5, 0.25, 0.25]), ..Default::default() };
    let result = evaluate_sensitivity_decision(&bayes, &artifact, &weighted).unwrap();
    let StructuralVerdict::BayesChoice { action, evaluated_mass } = &result.structural.verdict
    else {
        panic!("expected a Bayes choice, got {:?}", result.structural.verdict);
    };
    assert_eq!(action, "A");
    assert_close(*evaluated_mass, 1.0);
    assert_close(result.structural.actions[0].weighted_value.unwrap(), 1.25);
    assert_close(result.structural.actions[1].weighted_value.unwrap(), 1.0);

    // A weight count that does not match the grid refuses.
    let short = SensitivityDecisionSpec { weights: Some(vec![0.5, 0.5]), ..Default::default() };
    assert!(evaluate_sensitivity_decision(&bayes, &artifact, &short).is_err());

    // An assumption range with weights would turn the range into a probability law.
    let ranged = SensitivityArtifact::new(parts(
        &[0.0, 1.0, 2.0],
        vec![surface("effect", "utils", &[-1.0; 3], &[2.0; 3]), point("status_quo", &[0.0; 3])],
        vec![act("A", "effect"), act("B", "status_quo")],
    ))
    .unwrap();
    let contract = contract_from_artifact(&ranged, StructuralPolicy::BayesOverStructures).unwrap();
    let error = evaluate_sensitivity_decision(&contract, &ranged, &weighted).unwrap_err();
    assert_eq!(error.detail(), "sensitivity_decision_composition.wrong_contract");

    // The worst case over the assumption points needs no weights: A's worst is 0, B's is 1.
    let maximin = contract_from_artifact(&artifact, StructuralPolicy::Maximin).unwrap();
    let result = evaluate_sensitivity_decision(&maximin, &artifact, &none).unwrap();
    assert_eq!(result.structural.verdict, StructuralVerdict::WorstCaseChoice("B".into()));
}

#[test]
fn f17_declaration_order_does_not_change_identity_or_decision() {
    let base = SensitivityArtifact::new(frozen_parts()).unwrap();
    let mut shuffled = frozen_parts();
    shuffled.grid.reverse();
    shuffled.support.reverse();
    for q in &mut shuffled.quantities {
        q.lower.reverse();
        q.upper.reverse();
    }
    shuffled.quantities.reverse();
    shuffled.actions.reverse();
    let permuted = SensitivityArtifact::new(shuffled).unwrap();
    assert_eq!(base.identity(), permuted.identity());
    assert_eq!(base, permuted);
    let first = contract_from_artifact(&base, invariant_policy()).unwrap();
    let second = contract_from_artifact(&permuted, invariant_policy()).unwrap();
    assert_eq!(first.identity().unwrap(), second.identity().unwrap());
    assert_eq!(evaluate(&base).outcome, evaluate(&permuted).outcome);
}

fn joint_result(
    factors: Vec<JointFactorBound>,
    (baseline, minimum, maximum): (f64, f64, f64),
) -> JointMechanismSensitivityResult {
    JointMechanismSensitivityResult {
        query_binding: "binding-1".into(),
        provider_snapshot: "snapshot-1".into(),
        source_regime: RegimeId::from_raw(1),
        factors,
        decision_threshold: Some(0.25),
        baseline,
        range: JointSensitivityRange {
            minimum,
            maximum,
            minimizing_outcome_by_stratum: vec![],
            maximizing_outcome_by_stratum: vec![],
            minimizing_parent_level: None,
            maximizing_parent_level: None,
        },
        axis_tipping: vec![AxisTipping {
            factor: JointFactor::OutcomeKernel,
            analytic: Some(5.0 / 30.0),
            status: TippingStatus::Bracketed,
            bracket: Some(TippingBracket { lower: 0.16, upper: 0.17, iterations: 3 }),
        }],
        frontier: vec![],
        unresolved_detail: None,
        receipt: JointSensitivityReceipt {
            method: "closed-form product-polytope extrema",
            fraction_box: [0.4, 0.0],
            tolerance: 1e-9,
            limits: JointSensitivityLimits::default(),
            memory_limit_bytes: 0,
            operations_consumed: 0,
            depth_reached: 0,
            live_state_bytes: 0,
            stop: None,
            explored: vec![],
            unevaluated: vec![],
        },
        interpretation: "assumption range over the declared joint contamination box; not a confidence interval and not a sampling interval",
        inference_claim: "assumption_range",
        uncertainty: JointSensitivityUncertainty::withheld(),
    }
}

fn adopt_against(status_quo: f64) -> Vec<ActionUtility> {
    vec![
        act("adopt", "effect"),
        ActionUtility { id: "status_quo".into(), utility: UtilityTerm::Const(status_quo) },
    ]
}

#[test]
fn f17_a_2_2_joint_sensitivity_result_feeds_a_decision_contract() {
    // One factor, contamination fraction up to 0.4; baseline 0.5; extrema -0.1 and 1.1.
    // The range is linear in the fraction: lower = 0.5 - 0.6 t, upper = 0.5 + 0.6 t with
    // t = fraction / 0.4, so at 0, 0.1, 0.2, 0.3, 0.4 it is 0.5, 0.35, 0.2, 0.05, -0.1 below
    // and 0.5, 0.65, 0.8, 0.95, 1.1 above.
    let result = joint_result(
        vec![JointFactorBound { factor: JointFactor::OutcomeKernel, max_fraction: 0.4 }],
        (0.5, -0.1, 1.1),
    );
    let effect = scientific("effect", "utils");

    // A status quo of -0.5 is below the whole range: adopting is invariant.
    let artifact =
        SensitivityArtifact::from_joint_result(&result, &effect, 5, adopt_against(-0.5), "checked")
            .unwrap();
    assert_eq!(artifact.coordinate().id, "outcome_kernel");
    assert_eq!(artifact.coordinate().scale, "contamination_fraction");
    let surface = &artifact.quantities()[0];
    for (actual, expected) in surface.lower.iter().zip([0.5, 0.35, 0.2, 0.05, -0.1]) {
        assert_close(*actual, expected);
    }
    for (actual, expected) in surface.upper.iter().zip([0.5, 0.65, 0.8, 0.95, 1.1]) {
        assert_close(*actual, expected);
    }
    assert_eq!(artifact.provenance().source_kind, "joint_mechanism_sensitivity_2_2");
    assert_eq!(artifact.provenance().source_tipping[0].factor, "outcome_kernel");
    let decided = evaluate(&artifact);
    assert_eq!(decided.outcome, SensitivityOutcome::InvariantAction { action: "adopt".into() });
    assert_eq!(decided.coverage, ScenarioCoverage::VertexCertified);
    assert!(matches!(
        decided.sampling,
        SamplingReport::Withheld { ref reason_code, .. } if reason_code.as_str() == "cell_not_licensed"
    ));

    // A status quo of 0.25 sits inside the range once the lower end falls below it,
    // which happens at t = 5/12, i.e. at fraction 1/6, between the grid points 0.1 and 0.2.
    let artifact =
        SensitivityArtifact::from_joint_result(&result, &effect, 5, adopt_against(0.25), "checked")
            .unwrap();
    let decided = evaluate(&artifact);
    let SensitivityOutcome::NoRobustAction { reason, switches } = &decided.outcome else {
        panic!("expected no robust action, got {:?}", decided.outcome);
    };
    assert!(switches.is_empty());
    let NoRobustReason::MixedWithinRange { coordinates } = reason else {
        panic!("expected a mixed range, got {reason:?}");
    };
    assert_eq!(coordinates.len(), 3);
    for (actual, expected) in coordinates.iter().zip([0.2, 0.3, 0.4]) {
        assert_close(*actual, expected);
    }
    let tipping_fraction = 0.4 * 5.0 / 12.0;
    assert!(0.1 < tipping_fraction && tipping_fraction < coordinates[0]);
    assert_eq!(artifact.decision_outcome(None).unwrap(), decided.outcome);

    // Two factors are exact only at the unperturbed point and the declared box.
    let joint = joint_result(
        vec![
            JointFactorBound { factor: JointFactor::OutcomeKernel, max_fraction: 0.3 },
            JointFactorBound { factor: JointFactor::SharedParentMarginal, max_fraction: 0.2 },
        ],
        (1.0, 0.2, 1.8),
    );
    let artifact =
        SensitivityArtifact::from_joint_result(&joint, &effect, 2, adopt_against(0.0), "checked")
            .unwrap();
    assert_eq!(artifact.grid(), &[0.0, 1.0]);
    assert_eq!(artifact.quantities()[0].lower, vec![1.0, 0.2]);
    assert_eq!(artifact.quantities()[0].upper, vec![1.0, 1.8]);
    let error =
        SensitivityArtifact::from_joint_result(&joint, &effect, 5, adopt_against(0.0), "checked")
            .unwrap_err();
    assert!(
        refusal(&error).1.starts_with("sensitivity_decision_composition.unsupported_adaptation")
    );
}
