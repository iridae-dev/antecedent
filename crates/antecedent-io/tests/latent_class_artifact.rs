//! Latent-class artifact: the hand-derived two-class fixed point of the core tests through
//! the container, a refitting consumer that checks the canonical output bit for bit, label
//! handling, resealed-mutation refusals against a retained identity, the premise records, unknown
//! versions, the computation, export and replay budgets of a fit with no row ceiling, and the
//! unmeasured calibration coordinate.
//!
//! Oracle (identical to `antecedent-estimate/tests/latent_class_effects.rs`): class `P` (24
//! rows, `a in {0, 1}`, `x in {-1, 0, 1}`, `y = a + 0.5 x + e`) and class `Q` (16 rows,
//! `a in {0, 1}`, `x in {-1, 1}`, `y = 20 - 2 a - 0.5 x + e`) with within-cell noise
//! `{+0.5, -0.5, +0.25, -0.25}`. So `pi = (0.6, 0.4)`, `tau = (1, -2)`, the canonical (ascending
//! `tau`) order is `Q, P`, and the mixture-average effect is `-0.2`.

use antecedent_core::{ExecutionContext, MemoryBudget};
use antecedent_io::latent_class_artifact::{
    LATENT_CLASS_ARTIFACT_VERSION, LatentClassArtifact, LatentClassArtifactError,
    LatentClassConfigWire, LatentClassDataWire, LatentClassRequestWire, decode_parts, encode_parts,
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

#[path = "common/mod.rs"]
mod common;

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    common::close(actual, expected, tolerance, what);
}

fn seal(seed: u64, bootstrap: u32) -> (LatentClassArtifact, Vec<u8>) {
    let artifact = LatentClassArtifact::seal(&request(seed, bootstrap)).unwrap();
    let bytes = artifact.to_bytes("latent-class-test").unwrap();
    (artifact, bytes)
}

fn refusal(error: &LatentClassArtifactError) -> (&'static str, String) {
    let (code, detail, _) = error.refusal().expect("a registered refusal");
    (code, detail)
}

#[test]
fn latent_artifact_round_trips_the_hand_derived_fixed_point() {
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
fn latent_artifact_summarizes_responsibilities_exactly() {
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
fn latent_artifact_records_trace_premises_and_unmeasured_calibration() {
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
fn latent_artifact_refit_with_bootstrap_is_bit_for_bit() {
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
fn latent_artifact_label_permutation_keeps_the_canonical_classes() {
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
fn latent_artifact_refuses_changed_data_even_when_resealed() {
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
fn latent_artifact_refuses_a_changed_stored_result_or_config() {
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
fn latent_artifact_seal_refuses_undeclared_weak_and_single_class_fits() {
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
fn latent_artifact_refuses_unknown_versions_features_and_corruption() {
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
fn latent_artifact_caps_bootstrap_replicates() {
    let mut replicates = request(1, 0);
    replicates.config.bootstrap_replicates = 1_000_000;
    assert!(matches!(
        LatentClassArtifact::seal(&replicates),
        Err(LatentClassArtifactError::LimitsExceeded("bootstrap replicates"))
    ));
}

/// The 40-row oracle dataset repeated `copies` times. Repetition scales every sufficient
/// statistic alike, so the fixed point (`pi = (0.6, 0.4)`, `tau = (1, -2)`) is unchanged.
fn scaled_request(copies: usize) -> LatentClassRequestWire {
    let base = build().data;
    let repeat = |column: &[f64]| -> Vec<f64> {
        column.iter().copied().cycle().take(column.len() * copies).collect()
    };
    LatentClassRequestWire {
        config: config(7, 0),
        data: LatentClassDataWire {
            outcome: repeat(&base.outcome),
            treatment: repeat(&base.treatment),
            covariate_names: base.covariate_names.clone(),
            covariates: base.covariates.iter().map(|c| repeat(c)).collect(),
        },
    }
}

fn assert_fixed_point(artifact: &LatentClassArtifact, rows: u64) {
    let meta = artifact.meta();
    assert_eq!(meta.n_rows, rows);
    assert_eq!(meta.responsibilities.rows, rows);
    close(meta.classes[0].effect, -2.0, 1e-6, "tau q");
    close(meta.classes[0].weight, 0.4, 1e-9, "pi q");
    close(meta.classes[1].effect, 1.0, 1e-6, "tau p");
    close(meta.classes[1].weight, 0.6, 1e-9, "pi p");
    close(meta.mixture_average_effect, -0.2, 1e-6, "mixture average");
}

fn context_with_memory(hard_limit_bytes: u64) -> ExecutionContext {
    let mut ctx = ExecutionContext::production(0, 1);
    ctx.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(hard_limit_bytes) };
    ctx
}

/// Fit, export under a budget, bind by digest and replay a fit of `copies * 40` rows.
fn exercise_budgets(copies: usize) {
    let request = scaled_request(copies);
    let rows = u64::try_from(40 * copies).unwrap();
    let artifact = LatentClassArtifact::seal(&request).unwrap();
    assert_fixed_point(&artifact, rows);

    // Export: the dataset is embedded only inside the budget, and a refusal names the bytes.
    let embedded = artifact.to_bytes("scaled").unwrap();
    assert_eq!(artifact.to_bytes_within("scaled", embedded.len()).unwrap(), embedded);
    let error = artifact.to_bytes_within("scaled", embedded.len() - 1).unwrap_err();
    let (code, detail, text) = error.refusal().expect("a registered refusal");
    assert_eq!(
        (code, detail.as_str()),
        ("route_not_supported", "latent_class.export_budget_exceeded")
    );
    assert!(text.contains(&format!("{} bytes", embedded.len())), "{text}");
    let error = artifact.to_bytes_within("scaled", 1_000).unwrap_err();
    assert_eq!(refusal(&error).1, "latent_class.export_budget_exceeded");
    // The refusal costs the caller nothing: the fit is still complete in memory.
    assert_eq!(artifact.report().responsibilities.len(), request.data.outcome.len());
    assert_fixed_point(&artifact, rows);

    // Replay: the digest-bound record carries no rows and needs the dataset it is bound to.
    let bound = artifact.to_bytes_digest_bound("bound").unwrap();
    assert!(decode_parts(&bound).unwrap().1.is_empty());
    assert!(!decode_parts(&embedded).unwrap().1.is_empty());
    let ctx = ExecutionContext::production(0, 1);
    let identity = artifact.identity();
    let error =
        LatentClassArtifact::from_bytes_with(&bound, Some(identity), None, &ctx).unwrap_err();
    assert_eq!(refusal(&error), ("route_not_supported", "latent_class.data_required".into()));
    let replay =
        LatentClassArtifact::from_bytes_with(&bound, Some(identity), Some(&request.data), &ctx)
            .unwrap();
    assert_eq!(replay.meta(), artifact.meta());
    let mut other = request.data.clone();
    other.outcome[0] += 1.0;
    assert!(matches!(
        LatentClassArtifact::from_bytes_with(&bound, None, Some(&other), &ctx),
        Err(LatentClassArtifactError::IdentityMismatch { field: "data" })
    ));
    // An artifact that embeds its dataset takes none, and replays on its own.
    assert!(matches!(
        LatentClassArtifact::from_bytes_with(&embedded, None, Some(&request.data), &ctx),
        Err(LatentClassArtifactError::UnsupportedSemantics(_))
    ));
    let replay = LatentClassArtifact::from_bytes(&embedded, Some(identity)).unwrap();
    assert_eq!(replay.meta(), artifact.meta());
}

#[test]
fn latent_fit_has_no_row_ceiling_and_export_and_replay_are_budgeted() {
    // 30,000 rows: three times the former embedded-row ceiling.
    exercise_budgets(750);
}

#[test]
#[ignore = "fits 100,000 rows; run with --ignored"]
fn latent_fit_runs_at_one_hundred_thousand_rows() {
    exercise_budgets(2_500);
}

#[test]
fn latent_computation_budget_refuses_before_work_and_honours_cancellation() {
    let request = request(1, 0);
    // 40 rows need about 6 kB of working set.
    let error = LatentClassArtifact::seal_with(&request, &context_with_memory(1_000)).unwrap_err();
    assert_eq!(
        refusal(&error),
        ("route_not_supported", "latent_class.memory_budget_exceeded".into())
    );
    assert!(LatentClassArtifact::seal_with(&request, &context_with_memory(1_000_000)).is_ok());
    let bytes = seal(1, 0).1;
    let error =
        LatentClassArtifact::from_bytes_with(&bytes, None, None, &context_with_memory(1_000))
            .unwrap_err();
    assert_eq!(refusal(&error).1, "latent_class.memory_budget_exceeded");

    let cancelled = ExecutionContext::production(0, 1);
    cancelled.cancellation.cancel();
    let error = LatentClassArtifact::seal_with(&request, &cancelled).unwrap_err();
    assert_eq!(refusal(&error), ("cancelled_no_claim", "latent_class.cancelled".into()));
}
