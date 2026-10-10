//! F17 `SensitivityArtifact`: the portable wire, its recomputing consumer, resealed
//! mutation refusals, version refusal, bounds and the adaptation of a 2.2 result.
//!
//! Expected values are derived by hand from the declared surfaces. The frozen surface has
//! `gamma` in `{0, 1, 2}` with action A at utility `2 - gamma` (2, 1, 0) and action B at
//! utility 1: A leads at 0, they tie at 1 and B leads at 2, so the tipping coordinate is
//! the exact tie at `gamma = 1`.
#![allow(clippy::float_cmp, reason = "exact hand-derived grid values")]

use antecedent_core::{QuantityRole, RegimeId, ScientificQuantity};
use antecedent_io::error::IoError;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::sensitivity_artifact::{
    ActionUtility, AssumptionCoordinate, AssumptionRangeStatement, AxisTipping, DecisionSwitch,
    JointFactor, JointFactorBound, JointMechanismSensitivityResult, JointSensitivityLimits,
    JointSensitivityRange, JointSensitivityReceipt, JointSensitivityUncertainty,
    MAX_SENSITIVITY_ARTIFACT_BYTES, PointSupport, SamplingInterval, SamplingStatus,
    SensitivityArtifact, SensitivityOutcome, SensitivityParts, SensitivityProvenance,
    SurfaceQuantity, TippingBracket, TippingStatus, UncertaintyRelationship, UtilityTerm,
    decode_parts, encode_parts,
};

fn scientific(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "utils".into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "sensitivity_surface".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn surface(name: &str, lower: &[f64], upper: &[f64]) -> SurfaceQuantity {
    SurfaceQuantity {
        quantity: ScientificQuantityWire::from(&scientific(name)),
        lower: lower.to_vec(),
        upper: upper.to_vec(),
    }
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

fn frozen_parts() -> SensitivityParts {
    SensitivityParts {
        coordinate: AssumptionCoordinate {
            id: "gamma".into(),
            scale: "sensitivity_parameter".into(),
            units: "dimensionless".into(),
            minimum: 0.0,
            maximum: 2.0,
        },
        grid: vec![0.0, 1.0, 2.0],
        support: vec![PointSupport::Supported; 3],
        quantities: vec![
            surface("ua", &[2.0, 1.0, 0.0], &[2.0, 1.0, 0.0]),
            surface("ub", &[1.0; 3], &[1.0; 3]),
        ],
        actions: vec![act("A", "ua"), act("B", "ub")],
        uncertainty: withheld(),
        provenance: SensitivityProvenance {
            source_kind: "supplied_surface".into(),
            query_binding: "f17-io-test".into(),
            provider_snapshot: "snapshot-1".into(),
            source_regime: "regime:1".into(),
            method: "hand-derived surface".into(),
            causal_contract_id: "checked-contract".into(),
            decision_threshold: None,
            source_tipping: vec![],
        },
    }
}

fn frozen_switch() -> SensitivityOutcome {
    SensitivityOutcome::AssumptionDependent {
        switch: DecisionSwitch {
            from: vec!["A".into()],
            to: vec!["B".into()],
            lower: 1.0,
            upper: 1.0,
            exact: true,
            interpolated: None,
        },
    }
}

fn set(numbers: &mut [u8], index: usize, value: f64) {
    numbers[index * 8..index * 8 + 8].copy_from_slice(&value.to_le_bytes());
}

fn refusal(error: &IoError) -> (&'static str, &str) {
    match error {
        IoError::Refused { code, message } => (*code, message.as_str()),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

fn assert_wrong_contract(result: Result<SensitivityArtifact, IoError>, needle: &str) {
    let error = result.unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "decision_contract_unsatisfied");
    assert!(message.starts_with("sensitivity_decision_composition.wrong_contract"), "{message}");
    assert!(message.contains(needle), "{message}");
}

#[test]
fn f17_artifact_round_trips_and_its_consumer_replays_the_decision() {
    let artifact = SensitivityArtifact::new(frozen_parts()).unwrap();
    assert_eq!(artifact.outcome(), Some(&frozen_switch()));
    assert_eq!(artifact.decision_outcome(None).unwrap(), frozen_switch());
    let identity = artifact.identity();
    assert_ne!(identity.premises_digest, identity.data_digest);
    assert_eq!(identity.digest.len(), 64);

    let bytes = artifact.to_bytes("sensitivity-1").unwrap();
    let restored = SensitivityArtifact::from_bytes(&bytes, Some(identity)).unwrap();
    assert_eq!(restored, artifact);
    assert_eq!(
        SensitivityArtifact::from_bytes(&bytes, None).unwrap().outcome(),
        artifact.outcome()
    );
    // Serialization is deterministic.
    assert_eq!(restored.to_bytes("sensitivity-1").unwrap(), bytes);
}

#[test]
fn f17_resealed_surface_mutation_is_refused() {
    let artifact = SensitivityArtifact::new(frozen_parts()).unwrap();
    let bytes = artifact.to_bytes("sensitivity-1").unwrap();
    let (meta, mut numbers) = decode_parts(&bytes).unwrap();
    // Numbers: grid (0..3), ua.lower (3..6), ua.upper (6..9), ub.lower, ub.upper.
    // Setting A's utility at gamma = 0 to 0 would make B lead there.
    set(&mut numbers, 3, 0.0);
    set(&mut numbers, 6, 0.0);
    let resealed = encode_parts(&meta, &numbers, "sensitivity-1").unwrap();
    assert_wrong_contract(SensitivityArtifact::from_bytes(&resealed, None), "data digest");

    // A mutation that also recomputes the identity and outcome is self-consistent, so only
    // the identity the consumer retained independently refuses it.
    let mut changed = frozen_parts();
    changed.quantities[0] = surface("ua", &[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0]);
    let forged = SensitivityArtifact::new(changed).unwrap().to_bytes("sensitivity-1").unwrap();
    assert!(SensitivityArtifact::from_bytes(&forged, None).is_ok());
    assert_wrong_contract(
        SensitivityArtifact::from_bytes(&forged, Some(artifact.identity())),
        "retained identity",
    );
}

#[test]
fn f17_resealed_premise_and_result_mutations_are_refused() {
    let artifact = SensitivityArtifact::new(frozen_parts()).unwrap();
    let bytes = artifact.to_bytes("sensitivity-1").unwrap();
    let (meta, numbers) = decode_parts(&bytes).unwrap();
    let reseal = |changed: &antecedent_io::sensitivity_artifact::SensitivityMeta| {
        let bytes = encode_parts(changed, &numbers, "sensitivity-1").unwrap();
        SensitivityArtifact::from_bytes(&bytes, None)
    };

    let mut units = meta.clone();
    units.coordinate.units = "log_odds".into();
    assert_wrong_contract(reseal(&units), "premises digest");

    let mut swapped = meta.clone();
    swapped.actions[0].utility = UtilityTerm::quantity("ub");
    assert_wrong_contract(reseal(&swapped), "premises digest");

    let mut sampling = meta.clone();
    sampling.uncertainty.sampling = SamplingStatus::Withheld {
        reason_code: "route_not_supported".into(),
        detail: "x.y".into(),
    };
    assert_wrong_contract(reseal(&sampling), "premises digest");

    let mut support = meta.clone();
    support.support[1] = PointSupport::Unsupported;
    assert_wrong_contract(reseal(&support), "premises digest");

    let mut digest = meta.clone();
    digest.identity.digest = "0".repeat(64);
    assert_wrong_contract(reseal(&digest), "identity digest");

    // A stored decision that the surface does not replay, with the identity intact.
    let mut outcome = meta.clone();
    outcome.outcome = Some(SensitivityOutcome::InvariantAction { action: "A".into() });
    assert_wrong_contract(reseal(&outcome), "does not replay");

    let mut claim = meta;
    claim.inference_claim = "confidence_interval".into();
    assert_wrong_contract(reseal(&claim), "inference claim");
}

#[test]
fn f17_unknown_versions_and_malformed_payloads_are_refused() {
    let artifact = SensitivityArtifact::new(frozen_parts()).unwrap();
    let bytes = artifact.to_bytes("sensitivity-1").unwrap();
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();

    meta.version = 2;
    let other = encode_parts(&meta, &numbers, "sensitivity-1").unwrap();
    assert!(matches!(
        SensitivityArtifact::from_bytes(&other, None),
        Err(IoError::UnsupportedVersion { version: 2 })
    ));

    meta.version = 1;
    let mut short = numbers.clone();
    short.truncate(short.len() - 8);
    let resealed = encode_parts(&meta, &short, "sensitivity-1").unwrap();
    let error = SensitivityArtifact::from_bytes(&resealed, None).unwrap_err();
    assert!(refusal(&error).1.starts_with("sensitivity_decision_composition.invalid_surface"));

    let mut nan = numbers.clone();
    set(&mut nan, 4, f64::NAN);
    let resealed = encode_parts(&meta, &nan, "sensitivity-1").unwrap();
    assert!(SensitivityArtifact::from_bytes(&resealed, None).is_err());

    let mut oversized = bytes.clone();
    oversized.resize(MAX_SENSITIVITY_ARTIFACT_BYTES + 1, 0);
    assert!(matches!(SensitivityArtifact::from_bytes(&oversized, None), Err(IoError::TooLarge)));
    assert!(SensitivityArtifact::from_bytes(&bytes[..bytes.len() / 2], None).is_err());
    assert!(artifact.to_bytes("  ").is_err());
}

#[test]
fn f17_identity_does_not_depend_on_declaration_order() {
    let base = SensitivityArtifact::new(frozen_parts()).unwrap();
    let mut shuffled = frozen_parts();
    shuffled.grid = vec![2.0, 0.0, 1.0];
    shuffled.support = vec![PointSupport::Supported; 3];
    shuffled.quantities = vec![
        surface("ub", &[1.0; 3], &[1.0; 3]),
        surface("ua", &[0.0, 2.0, 1.0], &[0.0, 2.0, 1.0]),
    ];
    shuffled.actions.reverse();
    let permuted = SensitivityArtifact::new(shuffled).unwrap();
    assert_eq!(permuted.identity(), base.identity());
    assert_eq!(permuted.to_bytes("a").unwrap(), base.to_bytes("a").unwrap());
    assert_eq!(permuted.grid(), &[0.0, 1.0, 2.0]);
    assert_eq!(permuted.outcome(), Some(&frozen_switch()));
}

#[test]
fn f17_range_and_sampling_interval_stay_separate_fields() {
    // Two artifacts with the same assumption range and different sampling intervals share
    // their premises and the surface values; only the numerical data digest differs.
    let build = |interval_lower: f64| {
        let mut p = frozen_parts();
        p.quantities[0] = surface("ua", &[2.5, 1.5, 0.5], &[3.5, 2.5, 1.5]);
        p.uncertainty.sampling = SamplingStatus::Reported(SamplingInterval {
            quantity: "ua".into(),
            level: 0.9,
            method: "percentile_bootstrap".into(),
            composed: None,
            lower: vec![interval_lower; 3],
            upper: vec![9.0; 3],
        });
        SensitivityArtifact::new(p).unwrap()
    };
    let (first, second) = (build(-3.0), build(-2.0));
    assert_eq!(first.identity().premises_digest, second.identity().premises_digest);
    assert_ne!(first.identity().data_digest, second.identity().data_digest);
    assert_eq!(first.quantities(), second.quantities());
    assert_eq!(first.outcome(), second.outcome());

    // The reported interval round trips next to the range, unchanged.
    let restored =
        SensitivityArtifact::from_bytes(&first.to_bytes("a").unwrap(), Some(first.identity()))
            .unwrap();
    let SamplingStatus::Reported(interval) = &restored.uncertainty().sampling else {
        panic!("sampling interval was lost");
    };
    assert_eq!(interval.lower, vec![-3.0; 3]);
    assert_eq!(interval.upper, vec![9.0; 3]);
    assert_eq!(restored.quantities()[0].lower, vec![2.5, 1.5, 0.5]);

    // Composition with the assumption range is refused: no method is licensed.
    let mut composed = frozen_parts();
    composed.uncertainty.sampling = SamplingStatus::Reported(SamplingInterval {
        quantity: "ua".into(),
        level: 0.9,
        method: "percentile_bootstrap".into(),
        composed: Some("conservative_endpoint_percentile_bootstrap".into()),
        lower: vec![0.0; 3],
        upper: vec![9.0; 3],
    });
    let error = SensitivityArtifact::new(composed).unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "cell_not_licensed");
    assert!(message.starts_with("sensitivity_decision_composition.composition_not_licensed"));
}

#[test]
fn f17_bad_coordinates_and_surfaces_are_refused() {
    let refused = |mutate: &dyn Fn(&mut SensitivityParts)| {
        let mut p = frozen_parts();
        mutate(&mut p);
        let error = SensitivityArtifact::new(p).unwrap_err();
        let (code, message) = refusal(&error);
        (code, message.split_once(": ").map(|(detail, _)| detail.to_owned()).unwrap_or_default())
    };
    // A grid point outside the declared assumption range.
    let (code, detail) = refused(&|p| p.grid[2] = 3.0);
    assert_eq!(code, "quantity_semantics_mismatch");
    assert_eq!(detail, "sensitivity_decision_composition.unsupported_coordinate");
    // Duplicate grid points, lower above upper and mismatched lengths.
    let (_, detail) = refused(&|p| p.grid[1] = 0.0);
    assert_eq!(detail, "sensitivity_decision_composition.invalid_surface");
    let (_, detail) = refused(&|p| p.quantities[0] = surface("ua", &[3.0; 3], &[2.0; 3]));
    assert_eq!(detail, "sensitivity_decision_composition.invalid_surface");
    let (_, detail) = refused(&|p| {
        p.support.pop();
    });
    assert_eq!(detail, "sensitivity_decision_composition.invalid_surface");
    // Fewer than two actions, an action reading a missing quantity, mixed populations.
    let (_, detail) = refused(&|p| p.actions.truncate(1));
    assert_eq!(detail, "sensitivity_decision_composition.wrong_contract");
    let (_, detail) = refused(&|p| p.actions[0] = act("A", "missing"));
    assert_eq!(detail, "sensitivity_decision_composition.wrong_contract");
    let (_, detail) = refused(&|p| p.quantities[1].quantity.population_id = "other".into());
    assert_eq!(detail, "sensitivity_decision_composition.wrong_contract");
    // A surface range that is not declared an assumption range.
    let (_, detail) =
        refused(&|p| p.uncertainty.assumption_range.kind = "confidence_interval".into());
    assert_eq!(detail, "sensitivity_decision_composition.wrong_contract");
    // More ranged quantities than the vertex enumeration admits.
    let (code, detail) = refused(&|p| {
        p.quantities = (0..4).map(|k| surface(&format!("q{k}"), &[0.0; 3], &[1.0; 3])).collect();
        p.actions = vec![act("A", "q0"), act("B", "q1")];
    });
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "sensitivity_decision_composition.bounds_exceeded");
}

fn joint_result(
    factors: Vec<JointFactorBound>,
    baseline: f64,
    extrema: (f64, f64),
) -> JointMechanismSensitivityResult {
    JointMechanismSensitivityResult {
        query_binding: "binding-1".into(),
        provider_snapshot: "snapshot-1".into(),
        source_regime: RegimeId::from_raw(1),
        factors,
        decision_threshold: Some(0.25),
        baseline,
        range: JointSensitivityRange {
            minimum: extrema.0,
            maximum: extrema.1,
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
fn f17_a_2_2_result_is_adapted_exactly_and_travels_through_the_wire() {
    // Baseline 0.5, extrema -0.1 and 1.1 at the contamination bound 0.4; the range is linear
    // in the fraction, so it is 0.5 - 0.6 t and 0.5 + 0.6 t with t = fraction / 0.4.
    let result = joint_result(
        vec![JointFactorBound { factor: JointFactor::OutcomeKernel, max_fraction: 0.4 }],
        0.5,
        (-0.1, 1.1),
    );
    let effect = scientific("effect");
    let artifact =
        SensitivityArtifact::from_joint_result(&result, &effect, 5, adopt_against(-0.5), "checked")
            .unwrap();
    let grid = artifact.grid();
    assert_eq!((grid[0], grid[4]), (0.0, 0.4));
    let surface = &artifact.quantities()[0];
    for (actual, expected) in surface.lower.iter().zip([0.5, 0.35, 0.2, 0.05, -0.1]) {
        assert!((actual - expected).abs() <= 1e-12, "{actual} vs {expected}");
    }
    for (actual, expected) in surface.upper.iter().zip([0.5, 0.65, 0.8, 0.95, 1.1]) {
        assert!((actual - expected).abs() <= 1e-12, "{actual} vs {expected}");
    }
    // The endpoints are the result's own values, not blended ones.
    assert_eq!((surface.lower[4], surface.upper[4]), (-0.1, 1.1));
    assert_eq!((surface.lower[0], surface.upper[0]), (0.5, 0.5));
    // The range stays an assumption range and the withheld interval stays withheld.
    assert_eq!(artifact.uncertainty().assumption_range.kind, "assumption_range");
    assert!(matches!(
        &artifact.uncertainty().sampling,
        SamplingStatus::Withheld { detail, .. } if detail == "joint_sensitivity.interval_withheld"
    ));
    assert_eq!(artifact.provenance().decision_threshold, Some(0.25));
    assert_eq!(artifact.provenance().source_tipping[0].analytic, Some(5.0 / 30.0));
    // A status quo of -0.5 lies below the whole range, so adopting is invariant.
    assert_eq!(
        artifact.outcome(),
        Some(&SensitivityOutcome::InvariantAction { action: "adopt".into() })
    );

    let bytes = artifact.to_bytes("adapted").unwrap();
    let restored = SensitivityArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(restored, artifact);

    // A status quo of 0.25 falls inside the range from the first grid point where the lower
    // end drops below it (0.2); at 0 and 0.1 adopting still wins under every vertex.
    let contested =
        SensitivityArtifact::from_joint_result(&result, &effect, 5, adopt_against(0.25), "checked")
            .unwrap();
    assert!(matches!(
        contested.outcome(),
        Some(SensitivityOutcome::NoRobustAction { switches, .. }) if switches.is_empty()
    ));
}

#[test]
fn f17_a_2_2_result_that_cannot_be_filled_exactly_or_is_not_a_range_is_refused() {
    let effect = scientific("effect");
    let two = joint_result(
        vec![
            JointFactorBound { factor: JointFactor::OutcomeKernel, max_fraction: 0.3 },
            JointFactorBound { factor: JointFactor::SharedParentMarginal, max_fraction: 0.2 },
        ],
        1.0,
        (0.2, 1.8),
    );
    let exact =
        SensitivityArtifact::from_joint_result(&two, &effect, 2, adopt_against(0.0), "checked")
            .unwrap();
    assert_eq!(exact.coordinate().id, "box_scale");
    assert_eq!(exact.grid(), &[0.0, 1.0]);
    let error =
        SensitivityArtifact::from_joint_result(&two, &effect, 4, adopt_against(0.0), "checked")
            .unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("sensitivity_decision_composition.unsupported_adaptation"));

    let empty = joint_result(
        vec![JointFactorBound { factor: JointFactor::OutcomeKernel, max_fraction: 0.0 }],
        0.5,
        (0.5, 0.5),
    );
    assert!(
        SensitivityArtifact::from_joint_result(&empty, &effect, 5, adopt_against(0.0), "checked")
            .is_err()
    );

    let mut interval = joint_result(
        vec![JointFactorBound { factor: JointFactor::OutcomeKernel, max_fraction: 0.4 }],
        0.5,
        (-0.1, 1.1),
    );
    interval.inference_claim = "confidence_interval";
    assert_wrong_contract(
        SensitivityArtifact::from_joint_result(
            &interval,
            &effect,
            5,
            adopt_against(0.0),
            "checked",
        ),
        "assumption-range",
    );
}
