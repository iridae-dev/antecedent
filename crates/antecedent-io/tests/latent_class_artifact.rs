//! B4 latent-class artifact: the hand-derived two-class fixed point of the core tests through
//! the container, a refitting consumer that checks the canonical output bit for bit, label
//! handling, resealed-mutation refusals against a retained identity, the premise records, unknown
//! versions, the row cap and the unmeasured calibration coordinate.
//!
//! Oracle (identical to `antecedent-estimate/tests/latent_class_effects.rs`): class `P` (24
//! rows, `a in {0, 1}`, `x in {-1, 0, 1}`, `y = a + 0.5 x + e`) and class `Q` (16 rows,
//! `a in {0, 1}`, `x in {-1, 1}`, `y = 20 - 2 a - 0.5 x + e`) with within-cell noise
//! `{+0.5, -0.5, +0.25, -0.25}`. So `pi = (0.6, 0.4)`, `tau = (1, -2)`, the canonical (ascending
//! `tau`) order is `Q, P`, and the mixture-average effect is `-0.2`.
#![allow(clippy::float_cmp, reason = "bit-identical replay is the property under test")]

use antecedent_io::latent_class_artifact::{
    LATENT_CLASS_ARTIFACT_MAX_ROWS, LATENT_CLASS_ARTIFACT_VERSION, LatentClassArtifact,
    LatentClassArtifactError, LatentClassConfigWire, LatentClassDataWire, LatentClassRequestWire,
    decode_parts, encode_parts,
};

const NOISE: [f64; 4] = [0.5, -0.5, 0.25, -0.25];

struct Built {
    data: LatentClassDataWire,
    /// 0 for class `P`, 1 for class `Q`.
    class: Vec<u32>,
}

fn push(built: &mut Built, a: f64, x: f64, mean: f64, class: u32) {
    for e in NOISE {
        built.data.outcome.push(mean + e);
        built.data.treatment.push(a);
        built.data.covariates[0].push(x);
        built.class.push(class);
    }
}

fn build() -> Built {
    let mut built = Built {
        data: LatentClassDataWire {
            outcome: vec![],
            treatment: vec![],
            covariate_names: vec!["x".into()],
            covariates: vec![vec![]],
        },
        class: vec![],
    };
    for a in [0.0, 1.0] {
        for x in [-1.0, 0.0, 1.0] {
            push(&mut built, a, x, a + 0.5 * x, 0);
        }
    }
    for a in [0.0, 1.0] {
        for x in [-1.0, 1.0] {
            push(&mut built, a, x, 20.0 - 2.0 * a - 0.5 * x, 1);
        }
    }
    built
}

fn config(seed: u64, bootstrap: u32) -> LatentClassConfigWire {
    LatentClassConfigWire {
        classes: 2,
        seed,
        restarts: 10,
        max_iterations: 500,
        tolerance: 1e-10,
        min_class_weight: 0.05,
        min_separation: 0.7,
        variance_floor: 1e-8,
        bootstrap_replicates: bootstrap,
        initial_labels: None,
        assume_conditional_randomization: true,
    }
}

fn request(seed: u64, bootstrap: u32) -> LatentClassRequestWire {
    LatentClassRequestWire { config: config(seed, bootstrap), data: build().data }
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() <= tolerance, "{what}: {actual} vs {expected}");
}

fn seal(seed: u64, bootstrap: u32) -> (LatentClassArtifact, Vec<u8>) {
    let artifact = LatentClassArtifact::seal(&request(seed, bootstrap)).unwrap();
    let bytes = artifact.to_bytes("b4-latent-test").unwrap();
    (artifact, bytes)
}

fn refusal(error: &LatentClassArtifactError) -> (&'static str, String) {
    let (code, detail, _) = error.refusal().expect("a registered refusal");
    (code, detail)
}

#[test]
fn b4_latent_artifact_round_trips_the_hand_derived_fixed_point() {
    let (artifact, bytes) = seal(11, 0);
    let consumed = LatentClassArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    let meta = consumed.meta();
    assert_eq!(meta.version, LATENT_CLASS_ARTIFACT_VERSION);
    assert_eq!(meta.classes_declared, 2);
    let q = &meta.classes[0];
    let p = &meta.classes[1];
    close(q.effect, -2.0, 1e-8, "tau q");
    close(q.weight, 0.4, 1e-12, "pi q");
    close(q.intercept, 20.0, 1e-8, "alpha q");
    close(p.effect, 1.0, 1e-8, "tau p");
    close(p.weight, 0.6, 1e-12, "pi p");
    close(p.intercept, 0.0, 1e-8, "alpha p");
    close(meta.mixture_average_effect, -0.2, 1e-8, "mixture average");
    close(meta.min_effect_gap, 3.0, 1e-8, "effect gap");
    // Canonical class c came from raw component class_order[c]; the mapping is a permutation.
    let mut sorted = meta.class_order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1]);
    for class in &meta.classes {
        assert_eq!(class.raw_index, meta.class_order[class.index as usize]);
    }
    assert!(q.effect_se.is_none() && meta.mixture_average_se.is_none() && meta.bootstrap.is_none());
    assert_eq!(consumed.meta(), artifact.meta());
}

#[test]
fn b4_latent_artifact_summarizes_responsibilities_exactly() {
    let (artifact, _) = seal(3, 0);
    let summary = &artifact.meta().responsibilities;
    assert_eq!(summary.rows, 40);
    assert_eq!(summary.hard_counts, vec![16, 24]);
    close(summary.class_means[0], 0.4, 1e-12, "mean responsibility q");
    close(summary.class_means[1], 0.6, 1e-12, "mean responsibility p");
    close(summary.separation, 1.0, 1e-12, "separation");
    assert_eq!(summary.digest.len(), 64);
    let report = artifact.report();
    assert_eq!(report.responsibilities.len(), 40);
    let truth = build().class;
    for (i, row) in report.responsibilities.iter().enumerate() {
        close(row.iter().sum::<f64>(), 1.0, 1e-12, "row sum");
        // true class P (0) is canonical 1; true class Q (1) is canonical 0.
        let canonical = 1 - truth[i];
        close(row[canonical as usize], 1.0, 1e-12, "indicator");
        assert_eq!(report.hard_assignment[i], canonical);
    }
}

#[test]
fn b4_latent_artifact_records_trace_premises_and_unmeasured_calibration() {
    let (artifact, _) = seal(5, 0);
    let meta = artifact.meta();
    let likelihood = &meta.likelihood;
    assert!(!likelihood.trace_head.is_empty() && likelihood.trace_head.len() <= 16);
    assert_eq!(likelihood.trace_len, likelihood.iterations);
    for pair in likelihood.trace_head.windows(2) {
        assert!(pair[1] >= pair[0] - 1e-9 * (1.0 + pair[0].abs()), "{pair:?}");
    }
    assert_eq!(likelihood.trace_digest.len(), 64);
    let status =
        |name: &str| meta.premises.iter().find(|p| p.name == name).map(|p| p.status.as_str());
    assert_eq!(status("conditional_randomization_within_class"), Some("declared"));
    assert_eq!(status("gaussian_linear_class_outcome"), Some("declared"));
    assert_eq!(status("class_weight_and_separation"), Some("checked"));
    assert_eq!(meta.calibration, "unmeasured");
    assert_eq!(meta.inference_claim, "point_with_bootstrap_se");
    assert!(meta.caveat.contains("unmeasured"));
}

#[test]
fn b4_latent_artifact_refit_with_bootstrap_is_bit_for_bit() {
    let (artifact, bytes) = seal(7, 20);
    let boot = artifact.meta().bootstrap.as_ref().expect("bootstrap record");
    assert_eq!(boot.requested, 20);
    assert_eq!(boot.succeeded + boot.failed, 20);
    assert!(artifact.meta().classes[0].effect_se.is_some_and(|s| s > 0.0));
    assert!(artifact.meta().mixture_average_se.is_some());
    let replay = LatentClassArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(replay.meta(), artifact.meta(), "the seeded refit reproduces the record");
    assert_eq!(replay.report(), artifact.report());
}

#[test]
fn b4_latent_artifact_label_permutation_keeps_the_canonical_classes() {
    let truth = build().class;
    let swapped: Vec<u32> = truth.iter().map(|&c| 1 - c).collect();
    let mut a = request(1, 0);
    a.config.initial_labels = Some(truth);
    let mut b = request(1, 0);
    b.config.initial_labels = Some(swapped);
    let (a, b) = (LatentClassArtifact::seal(&a).unwrap(), LatentClassArtifact::seal(&b).unwrap());
    for (x, y) in a.meta().classes.iter().zip(&b.meta().classes) {
        close(x.effect, y.effect, 1e-12, "effect");
        close(x.weight, y.weight, 1e-12, "weight");
        close(x.intercept, y.intercept, 1e-12, "intercept");
    }
    close(a.meta().mixture_average_effect, b.meta().mixture_average_effect, 1e-12, "average");
    assert_eq!(a.meta().class_order, vec![1, 0]);
    assert_eq!(b.meta().class_order, vec![0, 1]);
}

#[test]
fn b4_latent_artifact_refuses_changed_data_even_when_resealed() {
    let (artifact, bytes) = seal(11, 0);
    let (meta, mut numbers) = decode_parts(&bytes).unwrap();
    // The outcome column comes first.
    let mut word = [0_u8; 8];
    word.copy_from_slice(&numbers[0..8]);
    numbers[0..8].copy_from_slice(&(f64::from_le_bytes(word) + 0.125).to_le_bytes());
    let tampered = encode_parts(&meta, &numbers, "tampered").unwrap();
    let error = LatentClassArtifact::from_bytes(&tampered, None).unwrap_err();
    assert!(
        matches!(error, LatentClassArtifactError::IdentityMismatch { field: "data" }),
        "{error:?}"
    );

    let mut changed = request(11, 0);
    changed.data.outcome[0] += 0.125;
    let resealed = LatentClassArtifact::seal(&changed).unwrap().to_bytes("resealed").unwrap();
    assert!(LatentClassArtifact::from_bytes(&resealed, None).is_ok());
    let error = LatentClassArtifact::from_bytes(&resealed, Some(artifact.identity())).unwrap_err();
    assert!(matches!(error, LatentClassArtifactError::IdentityMismatch { .. }), "{error:?}");
    assert_eq!(refusal(&error).1, "latent_class.wrong_contract");
}

#[test]
fn b4_latent_artifact_refuses_a_changed_stored_result_or_config() {
    let (_, bytes) = seal(11, 0);
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.classes[0].effect += 0.1;
    let tampered = encode_parts(&meta, &numbers, "tampered").unwrap();
    let error = LatentClassArtifact::from_bytes(&tampered, None).unwrap_err();
    assert!(matches!(error, LatentClassArtifactError::ResultMismatch(_)), "{error:?}");
    assert_eq!(refusal(&error).1, "latent_class.report_replay_mismatch");

    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.config.seed = 12;
    let tampered = encode_parts(&meta, &numbers, "tampered").unwrap();
    let error = LatentClassArtifact::from_bytes(&tampered, None).unwrap_err();
    assert!(
        matches!(error, LatentClassArtifactError::IdentityMismatch { field: "config" }),
        "{error:?}"
    );

    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.config.assume_conditional_randomization = false;
    let tampered = encode_parts(&meta, &numbers, "tampered").unwrap();
    let error = LatentClassArtifact::from_bytes(&tampered, None).unwrap_err();
    assert_eq!(refusal(&error).1, "latent_class.randomization_not_declared");
}

#[test]
fn b4_latent_artifact_seal_refuses_undeclared_weak_and_single_class_fits() {
    let mut undeclared = request(1, 0);
    undeclared.config.assume_conditional_randomization = false;
    let error = LatentClassArtifact::seal(&undeclared).unwrap_err();
    assert_eq!(
        refusal(&error),
        ("required_option_missing", "latent_class.randomization_not_declared".into())
    );

    let mut weak = request(1, 0);
    weak.config.min_class_weight = 0.5;
    let error = LatentClassArtifact::seal(&weak).unwrap_err();
    assert_eq!(refusal(&error), ("population_not_estimable", "latent_class.weak_class".into()));

    let mut single = request(1, 0);
    single.config.classes = 1;
    let error = LatentClassArtifact::seal(&single).unwrap_err();
    assert_eq!(
        refusal(&error),
        ("population_not_estimable", "latent_class.degenerate_class".into())
    );
}

#[test]
fn b4_latent_artifact_refuses_unknown_versions_features_and_corruption() {
    let (_, bytes) = seal(11, 0);
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.version = LATENT_CLASS_ARTIFACT_VERSION + 1;
    let future = encode_parts(&meta, &numbers, "future").unwrap();
    assert!(matches!(
        LatentClassArtifact::from_bytes(&future, None),
        Err(LatentClassArtifactError::UnsupportedVersion { .. })
    ));
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.feature = "something_else".into();
    let foreign = encode_parts(&meta, &numbers, "foreign").unwrap();
    assert!(matches!(
        LatentClassArtifact::from_bytes(&foreign, None),
        Err(LatentClassArtifactError::UnsupportedSemantics(_))
    ));
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.calibration = "measured".into();
    let claimed = encode_parts(&meta, &numbers, "claimed").unwrap();
    assert!(matches!(
        LatentClassArtifact::from_bytes(&claimed, None),
        Err(LatentClassArtifactError::UnsupportedSemantics("calibration"))
    ));
    let mut corrupt = bytes.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0xFF;
    assert!(LatentClassArtifact::from_bytes(&corrupt, None).is_err());
}

#[test]
fn b4_latent_artifact_caps_embedded_rows_and_replicates() {
    let rows = LATENT_CLASS_ARTIFACT_MAX_ROWS + 1;
    let big = LatentClassRequestWire {
        config: config(1, 0),
        data: LatentClassDataWire {
            outcome: vec![0.0; rows],
            treatment: vec![0.0; rows],
            covariate_names: vec![],
            covariates: vec![],
        },
    };
    assert!(matches!(
        LatentClassArtifact::seal(&big),
        Err(LatentClassArtifactError::LimitsExceeded("rows"))
    ));
    let mut replicates = request(1, 0);
    replicates.config.bootstrap_replicates = 1_000_000;
    assert!(matches!(
        LatentClassArtifact::seal(&replicates),
        Err(LatentClassArtifactError::LimitsExceeded("bootstrap replicates"))
    ));
}
