//! X8 fixed-population temporal counterfactual artifact: hand-computed two-slice SCM values
//! through the container, a recomputing consumer that replays both worlds, resealed-mutation
//! refusals against a retained identity (action time, unit history, snapshot, mechanism fit)
//! and unknown versions.
#![allow(clippy::float_cmp, reason = "bit-identical replay is the property under test")]
#![allow(clippy::cast_precision_loss, reason = "small deterministic fixtures")]

use antecedent_core::ExecutionContext;
use antecedent_io::temporal_counterfactual_artifact::{
    ActionHistoryWire, FactualUnitWire, FitWire, MechanismWire, TemporalCounterfactualArtifact,
    TemporalCounterfactualArtifactError, TemporalCounterfactualRequestWire, TemporalGraphWire,
    decode_parts, encode_parts,
};

// Known closed-form two-slice SCM:
//   L0 = 1.0 + u0
//   L1 = 0.5 + 0.8 L0 + 1.5 A0 + u1
//   Y  = -1.0 + 0.5 L0 + 0.7 A0 + 2.0 L1 + 1.2 A1 + u2
fn simulate(noise: [f64; 3], a: [f64; 2]) -> [f64; 5] {
    let l0 = 1.0 + noise[0];
    let l1 = 0.5 + 0.8 * l0 + 1.5 * a[0] + noise[1];
    let y = -1.0 + 0.5 * l0 + 0.7 * a[0] + 2.0 * l1 + 1.2 * a[1] + noise[2];
    [l0, a[0], l1, a[1], y]
}

const NOISE: [[f64; 3]; 4] =
    [[0.3, -0.2, 0.5], [-0.4, 0.6, -0.1], [0.0, 0.1, 0.9], [1.1, -0.7, -0.3]];
const FACTUAL_ACTIONS: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
const PLUS: [f64; 2] = [1.0, 1.0];
const MINUS: [f64; 2] = [0.0, 0.0];

fn edges() -> Vec<(String, String)> {
    [
        ("covariate_0", "covariate_1"),
        ("action_0", "covariate_1"),
        ("covariate_0", "outcome"),
        ("action_0", "outcome"),
        ("covariate_1", "outcome"),
        ("action_1", "outcome"),
    ]
    .iter()
    .map(|(p, c)| ((*p).to_owned(), (*c).to_owned()))
    .collect()
}

fn mechanism(node: &str, intercept: f64, parents: &[(&str, f64)]) -> MechanismWire {
    MechanismWire {
        node: node.into(),
        intercept,
        parent_coefficients: parents.iter().map(|(p, c)| ((*p).to_owned(), *c)).collect(),
        noise_halfwidth: None,
    }
}

fn fit() -> FitWire {
    FitWire {
        fit_id: "fit-known-coefficients".into(),
        mechanisms: vec![
            mechanism("covariate_0", 1.0, &[]),
            mechanism("covariate_1", 0.5, &[("covariate_0", 0.8), ("action_0", 1.5)]),
            mechanism(
                "outcome",
                -1.0,
                &[("covariate_0", 0.5), ("action_0", 0.7), ("covariate_1", 2.0), ("action_1", 1.2)],
            ),
        ],
    }
}

fn request() -> TemporalCounterfactualRequestWire {
    let units: Vec<String> = (0..4).map(|i| format!("unit-{i}")).collect();
    TemporalCounterfactualRequestWire {
        graph: TemporalGraphWire { horizon: 2, edges: edges(), latent_confounding: false },
        fit: fit(),
        snapshot: "snapshot-1".into(),
        factual: (0..4)
            .map(|i| FactualUnitWire {
                unit: units[i].clone(),
                history: format!("hist-{i}"),
                times: [0, 1],
                values: simulate(NOISE[i], FACTUAL_ACTIONS[i]),
            })
            .collect(),
        plus: ActionHistoryWire {
            name: "always_treat".into(),
            times: [0, 1],
            actions: PLUS,
            units: units.clone(),
        },
        minus: ActionHistoryWire {
            name: "never_treat".into(),
            times: [0, 1],
            actions: MINUS,
            units,
        },
    }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

fn export(
    request: &TemporalCounterfactualRequestWire,
) -> (TemporalCounterfactualArtifact, Vec<u8>) {
    let artifact = TemporalCounterfactualArtifact::seal(request, &ctx()).unwrap();
    let bytes = artifact.to_bytes("x8-test").unwrap();
    (artifact, bytes)
}

#[test]
fn x8_artifact_matches_hand_abduction_and_replays_both_worlds() {
    let (artifact, bytes) = export(&request());
    let consumed =
        TemporalCounterfactualArtifact::from_bytes(&bytes, Some(artifact.identity()), &ctx())
            .unwrap();
    let report = consumed.report();
    assert_eq!(report.units.len(), 4);
    let (mut sum_plus, mut sum_minus) = (0.0, 0.0);
    for (i, row) in report.units.iter().enumerate() {
        assert_eq!(row.unit, format!("unit-{i}"));
        let (plus, minus) = (simulate(NOISE[i], PLUS)[4], simulate(NOISE[i], MINUS)[4]);
        assert!((row.plus_outcome - plus).abs() < 1e-9, "unit {i} plus");
        assert!((row.minus_outcome - minus).abs() < 1e-9, "unit {i} minus");
        // The shared noise cancels: the per-unit contrast is 0.7 + 1.2 + 2.0 * 1.5 = 4.9.
        assert!(((row.plus_outcome - row.minus_outcome) - 4.9).abs() < 1e-9);
        sum_plus += plus;
        sum_minus += minus;
    }
    // Unit 0 by hand: L0 = 1.3, L1 = 0.5 + 1.04 + 1.5 - 0.2 = 2.84,
    // Y(1,1) = -1 + 0.65 + 0.7 + 5.68 + 1.2 + 0.5 = 7.73, Y(0,0) = -1 + 0.65 + 2.68 + 0.5 = 2.83.
    assert!((report.units[0].plus_outcome - 7.73).abs() < 1e-9);
    assert!((report.units[0].minus_outcome - 2.83).abs() < 1e-9);
    assert!((report.mean_plus - sum_plus / 4.0).abs() < 1e-9);
    assert!((report.mean_minus - sum_minus / 4.0).abs() < 1e-9);
    assert!((report.mean_contrast - 4.9).abs() < 1e-9);
    assert_eq!(report.inference_claim, "point_only");
    assert_eq!(report.mechanism_class, "linear_gaussian_additive_noise");
    assert_eq!(report.receipt.n_worlds, 2);
    assert_eq!(report.receipt.horizon, 2);
    assert_eq!(report.receipt.n_units, 4);
    assert!(report.receipt.shared_by_both_worlds);
    assert_eq!(report.receipt.unit_draws.len(), 4);
    // Fresh consumption reproduces the producer bit for bit.
    assert_eq!(report, artifact.report());
}

#[test]
fn x8_artifact_is_invariant_to_unit_order() {
    let base = TemporalCounterfactualArtifact::seal(&request(), &ctx()).unwrap();
    let mut shuffled = request();
    shuffled.factual.reverse();
    shuffled.plus.units.reverse();
    let again = TemporalCounterfactualArtifact::seal(&shuffled, &ctx()).unwrap();
    assert_eq!(base.report(), again.report());
    assert_eq!(base.identity(), again.identity());
}

#[test]
fn x8_artifact_unpaired_histories_refuse_with_a_witness() {
    let mut unpaired = request();
    unpaired.plus.units.pop();
    let error = TemporalCounterfactualArtifact::seal(&unpaired, &ctx()).unwrap_err();
    let refusal = error.refusal().expect("a typed refusal");
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, "temporal_counterfactual.unpaired_histories");
    let witness = refusal.witness.expect("a retained witness");
    assert_eq!(witness.unit.as_deref(), Some("unit-3"));
    assert_eq!(witness.world.as_deref(), Some("always_treat"));
}

#[test]
fn x8_artifact_refuting_history_retains_node_residual_and_bound() {
    let mut bounded = request();
    for m in &mut bounded.fit.mechanisms {
        m.noise_halfwidth = Some(0.45);
    }
    // Unit 0's outcome noise is 0.5 > 0.45; unit 3's covariate-0 noise 1.1 is larger still.
    let refusal =
        TemporalCounterfactualArtifact::seal(&bounded, &ctx()).unwrap_err().refusal().unwrap();
    assert_eq!(refusal.detail, "temporal_counterfactual.refuting_history");
    let witness = refusal.witness.unwrap();
    assert_eq!(witness.unit.as_deref(), Some("unit-0"));
    assert_eq!(witness.node.as_deref(), Some("outcome"));
    assert!((witness.residual.unwrap() - 0.5).abs() < 1e-9);
    assert_eq!(witness.bound, Some(0.45));
}

fn resealed_mutation_refuses(mutated: &TemporalCounterfactualRequestWire, field: &'static str) {
    let (original, _) = export(&request());
    let (_, mutated_bytes) = export(mutated);
    // Alone the mutated artifact is internally consistent and replays ...
    assert!(TemporalCounterfactualArtifact::from_bytes(&mutated_bytes, None, &ctx()).is_ok());
    // ... but against the identity the consumer retained it is refused.
    let error = TemporalCounterfactualArtifact::from_bytes(
        &mutated_bytes,
        Some(original.identity()),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error, TemporalCounterfactualArtifactError::IdentityMismatch { field }, "{field}");
    let refusal = error.refusal().unwrap();
    assert_eq!(refusal.detail, "temporal_counterfactual.artifact_changed");
    assert_eq!(refusal.offending.as_deref(), Some(field));
}

#[test]
fn x8_artifact_resealed_action_time_is_refused() {
    // Consistent times everywhere so the core accepts it, but they differ from the original.
    let mut moved = request();
    for f in &mut moved.factual {
        f.times = [0, 2];
    }
    moved.plus.times = [0, 2];
    moved.minus.times = [0, 2];
    // Factual times are part of the unit-history digest, which is checked before the worlds.
    resealed_mutation_refuses(&moved, "unit_history");
    // Only the worlds' times differ from a consumer's retained plan when factual times match.
    let (original, _) = export(&request());
    let mut worlds_only = request();
    worlds_only.plus.name = "always_treat_v2".into();
    let (_, bytes) = export(&worlds_only);
    assert_eq!(
        TemporalCounterfactualArtifact::from_bytes(&bytes, Some(original.identity()), &ctx())
            .unwrap_err(),
        TemporalCounterfactualArtifactError::IdentityMismatch { field: "action_history" }
    );
}

#[test]
fn x8_artifact_resealed_action_values_are_refused() {
    let mut changed = request();
    changed.plus.actions = [1.0, 0.0];
    resealed_mutation_refuses(&changed, "action_history");
}

#[test]
fn x8_artifact_resealed_unit_history_is_refused() {
    let mut changed = request();
    // A different but self-consistent trajectory for unit 2: other exogenous noise.
    changed.factual[2].values = simulate([0.4, 0.4, 0.4], FACTUAL_ACTIONS[2]);
    resealed_mutation_refuses(&changed, "unit_history");
    let mut renamed = request();
    renamed.factual[1].history = "hist-other".into();
    resealed_mutation_refuses(&renamed, "unit_history");
}

#[test]
fn x8_artifact_resealed_snapshot_is_refused() {
    let mut changed = request();
    changed.snapshot = "snapshot-2".into();
    resealed_mutation_refuses(&changed, "snapshot");
}

#[test]
fn x8_artifact_resealed_mechanism_fit_is_refused() {
    let mut changed = request();
    changed.fit.fit_id = "fit-refit".into();
    resealed_mutation_refuses(&changed, "mechanism_fit");
    let mut coefficient = request();
    coefficient.fit.mechanisms[2].intercept = -1.0;
    coefficient.fit.mechanisms[2].parent_coefficients[1].1 = 0.7;
    // Identical coefficients are the same fit; a different one is not.
    let (original, _) = export(&request());
    assert_eq!(
        TemporalCounterfactualArtifact::seal(&coefficient, &ctx()).unwrap().identity(),
        original.identity()
    );
    // A changed coefficient has no consistent history under a pinned noise bound, so reseal by
    // changing the covariate-1 mechanism and re-simulating the factual rows consistently.
    let mut refit = request();
    refit.fit.mechanisms[1].parent_coefficients[0].1 = 0.9;
    for (i, f) in refit.factual.iter_mut().enumerate() {
        let l0 = 1.0 + NOISE[i][0];
        let l1 = 0.5 + 0.9 * l0 + 1.5 * FACTUAL_ACTIONS[i][0] + NOISE[i][1];
        f.values[2] = l1;
        f.values[4] = -1.0
            + 0.5 * l0
            + 0.7 * FACTUAL_ACTIONS[i][0]
            + 2.0 * l1
            + 1.2 * FACTUAL_ACTIONS[i][1]
            + NOISE[i][2];
    }
    let (_, bytes) = export(&refit);
    let error =
        TemporalCounterfactualArtifact::from_bytes(&bytes, Some(original.identity()), &ctx())
            .unwrap_err();
    assert!(matches!(error, TemporalCounterfactualArtifactError::IdentityMismatch { .. }));
}

#[test]
fn x8_artifact_unresealed_tampering_does_not_replay() {
    let (artifact, bytes) = export(&request());
    let (meta, values) = decode_parts(&bytes).unwrap();

    // A forged stored counterfactual outcome (plus world of unit 0).
    let mut forged = values.clone();
    forged[40..48].copy_from_slice(&99.0_f64.to_le_bytes());
    let reencoded = encode_parts(&meta, &forged, "forged").unwrap();
    assert_eq!(
        TemporalCounterfactualArtifact::from_bytes(&reencoded, None, &ctx()).unwrap_err(),
        TemporalCounterfactualArtifactError::ResultMismatch("per-unit outcomes")
    );

    // A forged mean contrast.
    let mut forged_meta = meta.clone();
    forged_meta.mean_contrast += 1.0;
    let reencoded = encode_parts(&forged_meta, &values, "forged").unwrap();
    assert_eq!(
        TemporalCounterfactualArtifact::from_bytes(&reencoded, None, &ctx()).unwrap_err(),
        TemporalCounterfactualArtifactError::ResultMismatch("means")
    );

    // A changed factual observation under the stored identity.
    let mut changed = values.clone();
    changed[0..8].copy_from_slice(&5.0_f64.to_le_bytes());
    let reencoded = encode_parts(&meta, &changed, "forged").unwrap();
    assert_eq!(
        TemporalCounterfactualArtifact::from_bytes(&reencoded, None, &ctx()).unwrap_err(),
        TemporalCounterfactualArtifactError::IdentityMismatch { field: "unit_history" }
    );

    // A changed action time in only one world: the core refuses the misalignment.
    let mut forged_meta = meta.clone();
    forged_meta.plus.times = [0, 3];
    let reencoded = encode_parts(&forged_meta, &values, "forged").unwrap();
    let refusal = TemporalCounterfactualArtifact::from_bytes(&reencoded, None, &ctx())
        .unwrap_err()
        .refusal()
        .unwrap();
    assert_eq!(refusal.detail, "temporal_counterfactual.time_misaligned");

    // A tampered per-unit draw digest in the receipt.
    let mut forged_meta = meta.clone();
    forged_meta.receipt.unit_draws[0].digest = "00".repeat(16);
    let reencoded = encode_parts(&forged_meta, &values, "forged").unwrap();
    assert_eq!(
        TemporalCounterfactualArtifact::from_bytes(&reencoded, None, &ctx()).unwrap_err(),
        TemporalCounterfactualArtifactError::ResultMismatch("shared-abduction receipt")
    );

    // The claim cannot be upgraded.
    let mut forged_meta = meta.clone();
    forged_meta.inference_claim = "interval".into();
    let reencoded = encode_parts(&forged_meta, &values, "forged").unwrap();
    assert_eq!(
        TemporalCounterfactualArtifact::from_bytes(&reencoded, None, &ctx()).unwrap_err(),
        TemporalCounterfactualArtifactError::UnsupportedSemantics("inference claim")
    );
    assert!(
        TemporalCounterfactualArtifact::from_bytes(&bytes, Some(artifact.identity()), &ctx())
            .is_ok()
    );
}

#[test]
fn x8_artifact_unknown_major_version_refuses() {
    let (_, bytes) = export(&request());
    let (meta, values) = decode_parts(&bytes).unwrap();
    for version in [0_u16, 2, 9] {
        let mut other = meta.clone();
        other.version = version;
        let reencoded = encode_parts(&other, &values, "other-version").unwrap();
        assert_eq!(
            TemporalCounterfactualArtifact::from_bytes(&reencoded, None, &ctx()).unwrap_err(),
            TemporalCounterfactualArtifactError::UnsupportedVersion { version }
        );
    }
}

#[test]
fn x8_artifact_corrupt_truncated_and_cancelled_inputs_refuse() {
    let (_, bytes) = export(&request());
    let mut corrupt = bytes.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0xff;
    assert!(TemporalCounterfactualArtifact::from_bytes(&corrupt, None, &ctx()).is_err());
    assert!(
        TemporalCounterfactualArtifact::from_bytes(&bytes[..bytes.len() - 5], None, &ctx())
            .is_err()
    );
    let context = ctx();
    context.cancellation.cancel();
    assert_eq!(
        TemporalCounterfactualArtifact::from_bytes(&bytes, None, &context).unwrap_err(),
        TemporalCounterfactualArtifactError::Cancelled
    );
}
