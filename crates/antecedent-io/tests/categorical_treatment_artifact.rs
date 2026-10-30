//! Categorical-treatment artifact: group-mean closed forms through the container, a
//! recomputing consumer (summary replay for every covariance kind, bit-identical to the
//! row-based fit), the absence of a row ceiling, the computation and export budgets, the digest
//! binding of a summary to its rows, Holm multiplicity over the declared family, the declared
//! monotonicity test and its null, sparse/absent/undeclared-level refusals, resealed-mutation
//! refusals against a retained identity, unresealed tampering and the unmeasured calibration
//! coordinate.
//!
//! Frozen dataset A (rows interleaved on purpose): `a = [1,3]`, `b = [4,6,5]`, `c = [9,7,11,9]`;
//! `RSS = 12`, `sigma^2 = 2`; with reference `a`, `b_b = 3`, `b_c = 7`, and
//! `V = [[5/3, 1], [1, 3/2]]`. Frozen dataset M (ordered `lo < mid < hi`, three rows each):
//! `lo = [3,2,1]`, `mid = [2,1,0]`, `hi = [1,0,-1]`; `sigma^2 = 1`, every adjacent-step variance
//! is `2/3` and every step is `-1`. Calibration (Type I error, coverage, power) is deliberately
//! NOT measured here.

use antecedent_core::ExecutionContext;
use antecedent_estimate::categorical_treatment::{
    CategoricalTreatmentFit, CategoricalTreatmentInput, CategoricalTreatmentSpec, LevelScale,
    fit_categorical_treatment,
};
use antecedent_estimate::vector_treatment::{NamedColumn, VectorCovariance};
use antecedent_io::categorical_treatment_artifact::{
    CATEGORICAL_TREATMENT_ARTIFACT_VERSION, CATEGORICAL_TREATMENT_CALIBRATION, CategoricalSpecWire,
    CategoricalTreatmentArtifact, CategoricalTreatmentArtifactError, CategoricalTreatmentRequest,
    CovarianceKindWire, LevelScaleWire, MonotonicityDirectionWire, decode_parts, encode_parts,
};
use antecedent_io::vector_treatment_artifact::AdjustmentColumnRequest;
use antecedent_kernels::erfc;

#[path = "common/mod.rs"]
mod common;

fn close(actual: f64, expected: f64, what: &str) {
    common::close(actual, expected, 1e-9, what);
}

fn normal_p(z: f64) -> f64 {
    erfc(z.abs() / 2.0_f64.sqrt())
}

fn names(levels: &[&str]) -> Vec<String> {
    levels.iter().map(|l| (*l).to_string()).collect()
}

fn spec(levels: &[&str], ordered: bool, reference: &str) -> CategoricalSpecWire {
    CategoricalSpecWire {
        declared_levels: names(levels),
        scale: if ordered { LevelScaleWire::Ordered } else { LevelScaleWire::Unordered },
        reference: reference.into(),
        min_level_rows: 2,
        pairwise: vec![],
        monotonicity: None,
        covariance: CovarianceKindWire::ModelBased,
    }
}

fn abc_request() -> CategoricalTreatmentRequest {
    let rows = [
        ("c", 9.0),
        ("a", 1.0),
        ("b", 4.0),
        ("c", 7.0),
        ("b", 6.0),
        ("a", 3.0),
        ("c", 11.0),
        ("b", 5.0),
        ("c", 9.0),
    ];
    CategoricalTreatmentRequest {
        outcome: rows.iter().map(|r| r.1).collect(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        levels: rows.iter().map(|r| r.0.to_string()).collect(),
        spec: spec(&["a", "b", "c"], false, "a"),
    }
}

fn ordered_request() -> CategoricalTreatmentRequest {
    let mut rows: Vec<(&str, f64)> = Vec::new();
    for (level, values) in
        [("lo", [3.0, 2.0, 1.0]), ("mid", [2.0, 1.0, 0.0]), ("hi", [1.0, 0.0, -1.0])]
    {
        rows.extend(values.iter().map(|v| (level, *v)));
    }
    CategoricalTreatmentRequest {
        outcome: rows.iter().map(|r| r.1).collect(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        levels: rows.iter().map(|r| r.0.to_string()).collect(),
        spec: CategoricalSpecWire {
            monotonicity: Some(MonotonicityDirectionWire::NonDecreasing),
            ..spec(&["lo", "mid", "hi"], true, "lo")
        },
    }
}

fn export(request: &CategoricalTreatmentRequest) -> (CategoricalTreatmentArtifact, Vec<u8>) {
    let artifact = CategoricalTreatmentArtifact::seal(request).unwrap();
    let bytes = artifact.to_bytes("categorical-test").unwrap();
    (artifact, bytes)
}

#[test]
fn categorical_artifact_group_mean_algebra_survives_the_container() {
    let (artifact, bytes) = export(&abc_request());
    let report = CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity()))
        .unwrap()
        .report();
    assert_eq!(report.replay, "summary");
    let result = &report.result;
    assert_eq!(result.level_order, ["a", "b", "c"]);
    assert_eq!(result.counts.iter().map(|c| c.rows).collect::<Vec<_>>(), [2, 3, 4]);
    assert_eq!(result.coefficients[0].name, "level:b");
    close(result.coefficients[0].estimate, 3.0, "b - a");
    close(result.coefficients[1].estimate, 7.0, "c - a");
    close(result.covariance[0], 5.0 / 3.0, "V_bb");
    close(result.covariance[1], 1.0, "V_bc");
    close(result.covariance[2], 1.0, "V_cb");
    close(result.covariance[3], 1.5, "V_cc");
    assert_eq!(result.level_contrasts[0].level, "b");
    assert_eq!(result.level_contrasts[0].reference, "a");
    close(result.level_contrasts[1].standard_error, 1.5_f64.sqrt(), "se c");
    assert_eq!(report.null, "every level has the reference level's effect");
    assert!(report.monotonicity_null.is_none());
    assert!(result.monotonicity.is_none());
}

#[test]
fn categorical_artifact_omnibus_and_holm_family_match_closed_forms() {
    let mut request = abc_request();
    request.spec.pairwise = vec![("b".into(), "c".into())];
    let report = export(&request).0.report();
    let result = &report.result;
    let statistic = 319.0 / 9.0;
    close(result.omnibus.statistic, statistic, "omnibus");
    assert_eq!(result.omnibus.degrees_of_freedom, 2);
    close(result.omnibus.p_value, (-statistic / 2.0).exp(), "omnibus p");
    // Family: level b, level c and the requested pair b -> c (variance 7/6 off the covariance).
    assert_eq!(result.family_size, 3);
    let pair = &result.pairwise[0];
    close(pair.estimate, 4.0, "c - b");
    close(pair.standard_error.powi(2), 7.0 / 6.0, "pair variance");
    let p_b = normal_p(3.0 / (5.0_f64 / 3.0).sqrt());
    let p_c = normal_p(7.0 / 1.5_f64.sqrt());
    let p_pair = normal_p(4.0 / (7.0_f64 / 6.0).sqrt());
    // Holm step-down in ascending raw order: c, pair, b.
    let holm_c = (3.0 * p_c).min(1.0);
    let holm_pair = holm_c.max((2.0 * p_pair).min(1.0));
    let holm_b = holm_pair.max(p_b);
    close(result.level_contrasts[1].p_holm, holm_c, "Holm c");
    close(pair.p_holm, holm_pair, "Holm pair");
    close(result.level_contrasts[0].p_holm, holm_b, "Holm b");
}

#[test]
fn categorical_artifact_unordered_level_permutation_gives_identical_artifacts() {
    let base = export(&abc_request());
    let mut permuted = abc_request();
    permuted.spec.declared_levels = names(&["c", "a", "b"]);
    let other = export(&permuted);
    assert_eq!(base.0.identity(), other.0.identity());
    assert_eq!(base.0.report().result, other.0.report().result);
    // The permuted artifact is consumable against the original's retained identity.
    assert!(CategoricalTreatmentArtifact::from_bytes(&other.1, Some(base.0.identity())).is_ok());
}

#[test]
fn categorical_artifact_ordered_scale_is_part_of_the_estimand() {
    let ordered = {
        let mut request = abc_request();
        request.spec.scale = LevelScaleWire::Ordered;
        CategoricalTreatmentArtifact::seal(&request).unwrap()
    };
    let unordered = CategoricalTreatmentArtifact::seal(&abc_request()).unwrap();
    assert_ne!(ordered.identity().level_scale_id, unordered.identity().level_scale_id);
    let mut reversed = abc_request();
    reversed.spec.scale = LevelScaleWire::Ordered;
    reversed.spec.declared_levels = names(&["c", "b", "a"]);
    let reversed = CategoricalTreatmentArtifact::seal(&reversed).unwrap();
    assert_ne!(ordered.identity().level_scale_id, reversed.identity().level_scale_id);
    assert_eq!(reversed.report().result.level_order, ["c", "b", "a"]);
}

#[test]
fn categorical_artifact_monotonicity_result_and_null_are_stored_and_replayed() {
    let (artifact, bytes) = export(&ordered_request());
    let report = CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity()))
        .unwrap()
        .report();
    let mono = report.result.monotonicity.as_ref().expect("a declared monotonicity test");
    assert_eq!(mono.direction, "non_decreasing");
    assert_eq!(
        mono.null,
        "every adjacent step of the adjusted level effects is non-negative (non-decreasing)"
    );
    assert_eq!(report.monotonicity_null.as_deref(), Some(mono.null.as_str()));
    assert_eq!(mono.steps.len(), 2);
    for step in &mono.steps {
        close(step.difference, -1.0, "adjacent step");
        close(step.standard_error.powi(2), 2.0 / 3.0, "step variance");
    }
    let z = -(1.5_f64).sqrt();
    close(mono.statistic, z, "oriented minimum z");
    // p = min(1, 2 Phi(z)) = erfc(sqrt(3/4)).
    close(mono.p_value, erfc(0.75_f64.sqrt()), "conservative union-intersection p");
    assert!(mono.conservative);
    assert_eq!(mono.calibration, "unmeasured");
    assert!(report.caveats[2].contains("failing to reject does not prove"), "{:?}", report.caveats);
    // The opposite direction has a different null and the reverse orientation (p = 1 here).
    let mut down = ordered_request();
    down.spec.monotonicity = Some(MonotonicityDirectionWire::NonIncreasing);
    let flipped = export(&down).0.report();
    let flipped = flipped.result.monotonicity.unwrap();
    assert_eq!(flipped.direction, "non_increasing");
    assert_ne!(flipped.null, mono.null);
    close(flipped.statistic, 1.5_f64.sqrt(), "reverse orientation");
    close(flipped.p_value, 1.0, "no evidence against non-increasing");
}

#[test]
fn categorical_artifact_calibration_is_unmeasured_and_says_so() {
    let report = export(&abc_request()).0.report();
    assert_eq!(report.calibration, CATEGORICAL_TREATMENT_CALIBRATION);
    assert_eq!(report.calibration, "unmeasured");
    assert_eq!(report.result.calibration, "unmeasured");
    assert_eq!(report.inference_claim, "asymptotic_wald_calibration_unmeasured");
    assert_eq!(report.version, CATEGORICAL_TREATMENT_ARTIFACT_VERSION);
    assert!(report.caveats[0].contains("calibration is unmeasured"), "{:?}", report.caveats);
}

#[test]
fn categorical_artifact_sparse_absent_and_undeclared_levels_refuse_with_the_level_named() {
    let mut absent = abc_request();
    absent.spec.declared_levels = names(&["a", "b", "c", "d"]);
    let (code, detail, text) =
        CategoricalTreatmentArtifact::seal(&absent).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("arm_not_populated", "categorical_treatment.absent_level")
    );
    assert!(text.contains('d'), "{text}");

    let mut sparse = abc_request();
    sparse.spec.min_level_rows = 3;
    let (code, detail, text) =
        CategoricalTreatmentArtifact::seal(&sparse).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("arm_not_populated", "categorical_treatment.sparse_level")
    );
    assert!(text.contains("`a`"), "{text}");

    let mut undeclared = abc_request();
    undeclared.levels[0] = "zzz".into();
    let (code, detail, text) =
        CategoricalTreatmentArtifact::seal(&undeclared).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("invalid_argument", "categorical_treatment.undeclared_level")
    );
    assert!(text.contains("zzz"), "{text}");

    let mut reference = abc_request();
    reference.spec.reference = "q".into();
    let (_, detail, _) =
        CategoricalTreatmentArtifact::seal(&reference).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "categorical_treatment.unknown_reference");

    let mut unordered_monotone = abc_request();
    unordered_monotone.spec.monotonicity = Some(MonotonicityDirectionWire::NonDecreasing);
    let (code, detail, _) =
        CategoricalTreatmentArtifact::seal(&unordered_monotone).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("route_not_supported", "categorical_treatment.monotonicity_requires_ordered")
    );

    // A covariance kind outside the wire's set cannot be written into a declaration; it is
    // refused when the declaration is decoded.
    assert!(serde_json::from_str::<CovarianceKindWire>("\"cluster\"").is_err());
}

#[test]
fn categorical_artifact_robust_covariance_replays_from_a_summary() {
    let mut hc = abc_request();
    hc.spec.covariance = CovarianceKindWire::Hc1;
    let (artifact, bytes) = export(&hc);
    assert_eq!(artifact.report().replay, "robust_summary");
    // The summary is X'X, X'y, RSS and the meat of a p = 3 design, plus the level counts: no row
    // and no row label is embedded.
    let (meta, numbers) = decode_parts(&bytes).unwrap();
    assert_eq!(numbers.len(), (2 * 9 + 3 + 1) * 8);
    assert!(meta.row_levels.is_empty());
    assert_eq!(meta.counts.iter().map(|c| c.rows).collect::<Vec<_>>(), [2, 3, 4]);
    let consumed =
        CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.report(), artifact.report());
    assert_eq!(consumed.report().result.covariance_kind, "hc1");
    assert_ne!(
        consumed.report().result.covariance,
        CategoricalTreatmentArtifact::seal(&abc_request()).unwrap().report().result.covariance
    );
}

/// Deterministic pseudo-random draws in `[-1, 1)` (a 64-bit LCG).
struct Draws(u64);

impl Draws {
    fn next(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 52) as f64 - 1.0
    }
}

/// `n` rows over the unordered levels `a`, `b`, `c` with one adjustment column `z` and a
/// level-dependent heteroskedastic outcome.
fn synthetic(n: usize, covariance: &str) -> CategoricalTreatmentRequest {
    synthetic_seeded(n, covariance, 0xcafe)
}

/// [`synthetic`] on the stream started from `seed`.
fn synthetic_seeded(n: usize, covariance: &str, seed: u64) -> CategoricalTreatmentRequest {
    let mut draws = Draws(seed);
    let (mut outcome, mut z, mut levels) = (vec![], vec![], vec![]);
    for i in 0..n {
        let zi = draws.next();
        let (level, shift, spread) = [("a", 0.0, 0.5), ("b", 1.0, 1.5), ("c", -2.0, 1.0)][i % 3];
        outcome.push(shift + 0.7 * zi + (spread + zi * zi) * draws.next());
        z.push(zi);
        levels.push(level.to_string());
    }
    CategoricalTreatmentRequest {
        outcome,
        row_snapshot: "snap".into(),
        adjustment: vec![AdjustmentColumnRequest { name: "z".into(), values: z }],
        levels,
        spec: CategoricalSpecWire {
            pairwise: vec![("b".into(), "c".into())],
            covariance: serde_json::from_str(&format!("\"{covariance}\""))
                .expect("a covariance kind"),
            ..spec(&["a", "b", "c"], false, "a")
        },
    }
}

/// The core row-based fit of a request.
fn row_fit(request: &CategoricalTreatmentRequest) -> CategoricalTreatmentFit {
    let covariance = [
        VectorCovariance::ModelBased,
        VectorCovariance::Hc0,
        VectorCovariance::Hc1,
        VectorCovariance::Hc2,
        VectorCovariance::Hc3,
    ]
    .into_iter()
    .find(|kind| CovarianceKindWire::from(*kind) == request.spec.covariance)
    .unwrap();
    let input = CategoricalTreatmentInput {
        outcome: request.outcome.as_slice().into(),
        row_snapshot: request.row_snapshot.as_str().into(),
        adjustment: request
            .adjustment
            .iter()
            .map(|c| NamedColumn::new(c.name.as_str(), c.values.as_slice()))
            .collect(),
        levels: request.levels.as_slice().into(),
    };
    let spec = CategoricalTreatmentSpec {
        declared_levels: request.spec.declared_levels.clone(),
        scale: LevelScale::Unordered,
        reference: request.spec.reference.clone(),
        min_level_rows: usize::try_from(request.spec.min_level_rows).unwrap(),
        pairwise: request.spec.pairwise.clone(),
        monotonicity: None,
        covariance,
    };
    fit_categorical_treatment(&input, &spec).unwrap()
}

#[test]
fn categorical_summary_replay_equals_the_row_based_fit_for_every_robust_estimator() {
    let model_based = CategoricalTreatmentArtifact::seal(&synthetic(450, "model_based")).unwrap();
    for tag in ["hc0", "hc1", "hc2", "hc3"] {
        let request = synthetic(450, tag);
        let artifact = CategoricalTreatmentArtifact::seal(&request).unwrap();
        // Bit for bit: the summary holds the same meat the row-based sandwich accumulates.
        assert_eq!(artifact.fit(), &row_fit(&request), "{tag}");
        let consumed = CategoricalTreatmentArtifact::from_bytes(
            &artifact.to_bytes("t").unwrap(),
            Some(artifact.identity()),
        )
        .unwrap();
        assert_eq!(consumed.fit(), artifact.fit(), "{tag} after the artifact round trip");
        assert_ne!(artifact.fit().covariance, model_based.fit().covariance, "{tag}");
        // Independent of both paths: the textbook sandwich with a dense inverse, on all rows.
        let dummy = |level: &str| -> Vec<f64> {
            request.levels.iter().map(|l| f64::from(u8::from(l == level))).collect()
        };
        let design = vec![
            vec![1.0; request.outcome.len()],
            request.adjustment[0].values.clone(),
            dummy("b"),
            dummy("c"),
        ];
        let full = common::dense_sandwich(&design, &request.outcome, tag);
        common::assert_block_matches(&artifact.fit().covariance, &full, 4, 2, tag);
    }
}

#[test]
fn categorical_robust_fit_has_no_row_ceiling_and_the_artifact_does_not_grow_with_the_rows() {
    let small = CategoricalTreatmentArtifact::seal(&synthetic(300, "hc3")).unwrap();
    let big = synthetic(30_000, "hc3");
    let artifact = CategoricalTreatmentArtifact::seal(&big).unwrap();
    assert_eq!(artifact.meta().n_rows, 30_000);
    assert_eq!(artifact.fit(), &row_fit(&big));
    // Same summary size however many rows were fitted.
    let bytes = artifact.to_bytes("big").unwrap();
    assert!(bytes.len() < 2 * small.to_bytes("small").unwrap().len());
    let consumed =
        CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.fit(), artifact.fit());
}

#[test]
fn categorical_robust_fit_runs_at_one_hundred_thousand_rows() {
    let big = synthetic(100_000, "hc2");
    let artifact = CategoricalTreatmentArtifact::seal(&big).unwrap();
    assert_eq!(artifact.fit(), &row_fit(&big));
    let bytes = artifact.to_bytes("big").unwrap();
    assert!(bytes.len() < 64 * 1024);
    assert!(CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).is_ok());
}

/// The first `rows` rows of a request.
fn head(request: &CategoricalTreatmentRequest, rows: usize) -> CategoricalTreatmentRequest {
    let mut head = request.clone();
    head.outcome.truncate(rows);
    head.levels.truncate(rows);
    for column in &mut head.adjustment {
        column.values.truncate(rows);
    }
    head
}

/// A robust covariance of `kind` at 100,000 rows: the summary fit equals the row-based sandwich
/// bit for bit, the exported summary does not grow with the rows, and on a checkable subsample
/// the covariance equals an independent dense computation from the textbook definitions.
fn assert_robust_covariance_at_100k_rows(kind: &str, seed: u64) {
    let started = std::time::Instant::now();
    let big = synthetic_seeded(100_000, kind, seed);
    let artifact = CategoricalTreatmentArtifact::seal(&big).unwrap();
    assert_eq!(artifact.meta().n_rows, 100_000);
    assert_eq!(artifact.report().result.covariance_kind, kind);
    assert_eq!(artifact.fit(), &row_fit(&big), "{kind}: the summary is the row-based sandwich");
    let bytes = artifact.to_bytes("big").unwrap();
    assert!(bytes.len() < 64 * 1024);
    let consumed =
        CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.fit(), artifact.fit());
    assert!(started.elapsed().as_secs() < 600, "{:?}", started.elapsed());

    // The dense reference: design `[1, z, level:b, level:c]` against reference level `a`.
    let sample = head(&big, 4_000);
    let sealed = CategoricalTreatmentArtifact::seal(&sample).unwrap();
    let dummy = |level: &str| -> Vec<f64> {
        sample.levels.iter().map(|l| f64::from(u8::from(l == level))).collect()
    };
    let design = vec![
        vec![1.0; sample.outcome.len()],
        sample.adjustment[0].values.clone(),
        dummy("b"),
        dummy("c"),
    ];
    let full = common::dense_sandwich(&design, &sample.outcome, kind);
    common::assert_block_matches(&sealed.fit().covariance, &full, 4, 2, kind);
}

#[test]
fn categorical_treatment_hc0_covariance_100k_rows() {
    assert_robust_covariance_at_100k_rows("hc0", 7_840_663);
}

#[test]
fn categorical_treatment_hc1_covariance_100k_rows() {
    assert_robust_covariance_at_100k_rows("hc1", 7_248_023);
}

#[test]
fn categorical_treatment_hc2_covariance_100k_rows() {
    assert_robust_covariance_at_100k_rows("hc2", 1_288_919);
}

#[test]
fn categorical_treatment_hc3_covariance_100k_rows() {
    assert_robust_covariance_at_100k_rows("hc3", 6_336_663);
}

#[test]
fn categorical_summary_is_bound_to_its_rows_by_a_digest() {
    for tag in ["model_based", "hc2"] {
        let request = synthetic(120, tag);
        let artifact = CategoricalTreatmentArtifact::seal(&request).unwrap();
        let digest = artifact.meta().data_digest.clone().expect("a summary records its digest");
        assert_eq!(digest.len(), 64);
        assert_eq!(artifact.report().data_digest.as_deref(), Some(digest.as_str()));
        artifact.verify_data(&request).unwrap();
        // One changed value or one relabelled row fails the binding.
        let mut value = request.clone();
        value.outcome[7] += 1e-9;
        assert_eq!(
            artifact.verify_data(&value),
            Err(CategoricalTreatmentArtifactError::IdentityMismatch { field: "data" })
        );
        let mut relabelled = request.clone();
        relabelled.levels.swap(0, 1);
        assert!(artifact.verify_data(&relabelled).is_err());
        // The digest is part of the design identity: a forged one under the stored identity refuses.
        let (meta, numbers) = decode_parts(&artifact.to_bytes("t").unwrap()).unwrap();
        let mut forged = meta;
        forged.data_digest = Some("0".repeat(64));
        let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
        assert_eq!(
            CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
            CategoricalTreatmentArtifactError::IdentityMismatch { field: "design" }
        );
    }
}

#[test]
fn categorical_budgets_refuse_with_precise_details_and_keep_the_fit() {
    let request = synthetic(2_000, "hc1");
    let mut tight = ExecutionContext::production(0, 1);
    tight.set_memory_limit(Some(10_000));
    let error = CategoricalTreatmentArtifact::seal_with(&request, &tight).unwrap_err();
    let (code, detail, text) = error.refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("route_not_supported", "categorical_treatment.memory_budget_exceeded")
    );
    assert!(text.contains("10000-byte"), "{text}");
    let mut roomy = ExecutionContext::production(0, 1);
    roomy.set_memory_limit(Some(10_000_000));
    let artifact = CategoricalTreatmentArtifact::seal_with(&request, &roomy).unwrap();

    let cancelled = ExecutionContext::production(0, 1);
    cancelled.cancellation.cancel();
    let error = CategoricalTreatmentArtifact::seal_with(&request, &cancelled).unwrap_err();
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!((code, detail.as_str()), ("cancelled_no_claim", "categorical_treatment.cancelled"));

    // Export: an artifact above its budget refuses and names the bytes; the fit survives.
    let bytes = artifact.to_bytes("fit").unwrap();
    assert_eq!(artifact.to_bytes_within("fit", bytes.len()).unwrap(), bytes);
    let error = artifact.to_bytes_within("fit", bytes.len() - 1).unwrap_err();
    let (code, detail, text) = error.refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("route_not_supported", "categorical_treatment.export_budget_exceeded")
    );
    assert!(text.contains(&format!("{} bytes", bytes.len())), "{text}");
    assert_eq!(artifact.meta().n_rows, 2_000);

    // Replay runs under the same kind of context.
    let error =
        CategoricalTreatmentArtifact::from_bytes_with(&bytes, None, &cancelled).unwrap_err();
    assert_eq!(error.refusal().unwrap().1, "categorical_treatment.cancelled");
    assert!(CategoricalTreatmentArtifact::from_bytes_with(&bytes, None, &roomy).is_ok());
}

#[test]
fn categorical_artifact_adjustment_columns_enter_the_design_and_the_identity() {
    let mut adjusted = abc_request();
    adjusted.adjustment = vec![AdjustmentColumnRequest {
        name: "z".into(),
        values: vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    }];
    let (artifact, bytes) = export(&adjusted);
    assert_eq!(artifact.meta().adjustment, ["z"]);
    assert!(CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).is_ok());
    let plain = CategoricalTreatmentArtifact::seal(&abc_request()).unwrap();
    assert_ne!(artifact.identity().adjustment_set_id, plain.identity().adjustment_set_id);
}

/// Seal `mutated` as a consumer-consistent artifact and consume it against the identity of
/// `original`: the mutation is resealed (its own digests are correct) but must still refuse.
fn resealed_mutation_refuses(
    original: &CategoricalTreatmentRequest,
    mutated: &CategoricalTreatmentRequest,
    field: &'static str,
) {
    let (artifact, _) = export(original);
    let (_, mutated_bytes) = export(mutated);
    assert!(CategoricalTreatmentArtifact::from_bytes(&mutated_bytes, None).is_ok(), "{field}");
    let error = CategoricalTreatmentArtifact::from_bytes(&mutated_bytes, Some(artifact.identity()))
        .unwrap_err();
    assert_eq!(error, CategoricalTreatmentArtifactError::IdentityMismatch { field }, "{field}");
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "categorical_treatment.wrong_contract");
}

#[test]
fn categorical_artifact_resealed_identity_changes_are_refused() {
    let original = ordered_request();
    let mut snapshot = original.clone();
    snapshot.row_snapshot = "snap-2".into();
    resealed_mutation_refuses(&original, &snapshot, "row_snapshot");

    let mut reference = original.clone();
    reference.spec.reference = "mid".into();
    resealed_mutation_refuses(&original, &reference, "level_scale");

    let mut order = original.clone();
    order.spec.declared_levels = names(&["hi", "mid", "lo"]);
    order.spec.reference = "lo".into();
    resealed_mutation_refuses(&original, &order, "level_scale");

    let mut data = original.clone();
    data.outcome[0] = 3.5;
    resealed_mutation_refuses(&original, &data, "design");

    let mut kind = original.clone();
    kind.spec.covariance = CovarianceKindWire::Hc0;
    resealed_mutation_refuses(&original, &kind, "design");

    let mut pairs = original.clone();
    pairs.spec.pairwise = vec![("lo".into(), "hi".into())];
    resealed_mutation_refuses(&original, &pairs, "family");

    let mut direction = original.clone();
    direction.spec.monotonicity = Some(MonotonicityDirectionWire::NonIncreasing);
    resealed_mutation_refuses(&original, &direction, "family");

    let mut minimum = original.clone();
    minimum.spec.min_level_rows = 3;
    resealed_mutation_refuses(&original, &minimum, "family");

    let mut dropped_test = original.clone();
    dropped_test.spec.monotonicity = None;
    resealed_mutation_refuses(&original, &dropped_test, "family");
}

#[test]
fn categorical_artifact_unresealed_tampering_does_not_replay() {
    let (artifact, bytes) = export(&ordered_request());
    let (meta, numbers) = decode_parts(&bytes).unwrap();

    let mut forged = meta.clone();
    forged.result.level_contrasts[0].estimate += 1.0;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::ResultMismatch(_)
    ));

    let mut forged = meta.clone();
    forged.result.monotonicity.as_mut().unwrap().p_value = 0.001;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::ResultMismatch(_)
    ));

    let mut forged = meta.clone();
    forged.result.family_size += 1;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::ResultMismatch(_)
    ));

    // A stored monotonicity null that is not the declared direction's.
    let mut forged = meta.clone();
    forged.monotonicity_null = Some("every adjacent step is zero".into());
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::IdentityMismatch { field: "monotonicity_null" }
    );
    let mut forged = meta.clone();
    forged.null = "no level differs".into();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::IdentityMismatch { field: "null" }
    );

    // Level counts that disagree with the embedded Gram matrix.
    let mut forged = meta.clone();
    forged.counts[0].rows += 1;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    let error = CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err();
    assert_eq!(error.refusal().unwrap().1, "categorical_treatment.summary_inconsistent");

    // A changed residual sum of squares (the last embedded number) under the stored identity.
    let mut changed = numbers.clone();
    let last = changed.len() - 8;
    changed[last..].copy_from_slice(&9.0_f64.to_le_bytes());
    let reencoded = encode_parts(&meta, &changed, "forged").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::IdentityMismatch { field: "design" }
    );

    let mut forged = meta.clone();
    forged.calibration = "calibrated".into();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::UnsupportedSemantics("calibration")
    );
    let mut forged = meta.clone();
    forged.caveats.pop();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::UnsupportedSemantics("caveats")
    );
    assert!(CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).is_ok());
}

#[test]
fn categorical_artifact_unknown_major_version_and_corrupt_bytes_refuse() {
    let (_, bytes) = export(&abc_request());
    let (meta, numbers) = decode_parts(&bytes).unwrap();
    for version in [0_u16, 2, 7] {
        let mut other = meta.clone();
        other.version = version;
        let reencoded = encode_parts(&other, &numbers, "other-version").unwrap();
        assert_eq!(
            CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
            CategoricalTreatmentArtifactError::UnsupportedVersion { version }
        );
    }
    let mut other = meta;
    other.feature = "categorical_treatment_v2".into();
    let reencoded = encode_parts(&other, &numbers, "other-feature").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::UnsupportedSemantics("feature marker")
    );
    let mut corrupt = bytes.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0xff;
    assert!(CategoricalTreatmentArtifact::from_bytes(&corrupt, None).is_err());
    assert!(CategoricalTreatmentArtifact::from_bytes(&bytes[..bytes.len() - 5], None).is_err());
    assert!(CategoricalTreatmentArtifact::from_bytes(&[], None).is_err());
}

#[test]
fn the_budgeted_trait_agrees_with_the_inherent_entry_points() {
    use antecedent_io::BudgetedTreatmentArtifact;
    let request = abc_request();
    let artifact =
        <CategoricalTreatmentArtifact as BudgetedTreatmentArtifact>::seal(&request).unwrap();
    let bytes = BudgetedTreatmentArtifact::to_bytes(&artifact, "categorical-test").unwrap();
    assert_eq!(bytes, export(&request).1);
    assert_eq!(
        artifact
            .to_bytes_within("categorical-test", CategoricalTreatmentArtifact::MAX_BYTES)
            .unwrap(),
        bytes
    );
    assert!(artifact.to_bytes_within("categorical-test", bytes.len() - 1).is_err());
    let restored =
        <CategoricalTreatmentArtifact as BudgetedTreatmentArtifact>::from_bytes(&bytes, None)
            .unwrap();
    assert_eq!(restored.identity(), artifact.identity());
}

// ---------------------------------------------------------------------------------------------
// Declared rows: sampling weights, clusters and a serial lag are read, sealed and replayed.
// ---------------------------------------------------------------------------------------------

use antecedent_estimate::categorical_treatment::fit_categorical_treatment_declared;
use antecedent_estimate::vector_treatment::RowStructure;
use antecedent_io::categorical_treatment_artifact::{
    CATEGORICAL_DECLARED_ROWS_FORMAT, CategoricalPosteriorDeclaration, CategoricalRowsRequest,
};

fn declared_request(kind: CovarianceKindWire) -> CategoricalTreatmentRequest {
    let mut request = abc_request();
    request.spec.covariance = kind;
    request.adjustment = vec![AdjustmentColumnRequest {
        name: "z".into(),
        values: vec![0.0, 1.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 2.0],
    }];
    request
}

fn weights() -> Vec<f64> {
    vec![1.0, 2.5, 0.0, 1.0, 4.0, 0.5, 1.0, 3.0, 2.0]
}

fn seal_declared(
    request: &CategoricalTreatmentRequest,
    rows: &CategoricalRowsRequest,
) -> Result<CategoricalTreatmentArtifact, CategoricalTreatmentArtifactError> {
    CategoricalTreatmentArtifact::seal_declared(request, rows, &ExecutionContext::for_tests(0))
}

fn rows_of(
    request: &CategoricalTreatmentRequest,
    rows: &CategoricalRowsRequest,
) -> CategoricalTreatmentFit {
    let input = CategoricalTreatmentInput {
        outcome: request.outcome.as_slice().into(),
        row_snapshot: request.row_snapshot.as_str().into(),
        adjustment: request
            .adjustment
            .iter()
            .map(|c| NamedColumn::new(c.name.as_str(), c.values.as_slice()))
            .collect(),
        levels: request.levels.as_slice().into(),
    };
    let spec = CategoricalTreatmentSpec {
        declared_levels: request.spec.declared_levels.clone(),
        scale: LevelScale::Unordered,
        reference: request.spec.reference.clone(),
        min_level_rows: usize::try_from(request.spec.min_level_rows).unwrap(),
        pairwise: request.spec.pairwise.clone(),
        monotonicity: None,
        covariance: request.spec.covariance.into(),
    };
    let structure = RowStructure {
        sampling_weights: rows.sampling_weights.as_deref(),
        clusters: rows.cluster.as_deref(),
        lag: rows.serial_lag.map(|l| usize::try_from(l).unwrap()),
    };
    fit_categorical_treatment_declared(&input, &spec, &structure).unwrap().0
}

#[test]
fn declared_weights_clusters_and_lags_seal_replay_and_equal_the_row_based_fit() {
    let cases = [
        (
            CovarianceKindWire::Hc1,
            CategoricalRowsRequest { sampling_weights: Some(weights()), ..Default::default() },
        ),
        (
            CovarianceKindWire::ClusterRobust,
            CategoricalRowsRequest {
                cluster: Some(vec![0, 0, 1, 1, 2, 2, 3, 3, 3]),
                ..Default::default()
            },
        ),
        (
            CovarianceKindWire::ClusterRobust,
            CategoricalRowsRequest {
                sampling_weights: Some(weights()),
                cluster: Some(vec![0, 1, 2, 0, 1, 2, 0, 1, 2]),
                ..Default::default()
            },
        ),
        (
            CovarianceKindWire::NeweyWest,
            CategoricalRowsRequest { serial_lag: Some(2), ..Default::default() },
        ),
    ];
    for (kind, rows) in cases {
        let request = declared_request(kind);
        let artifact = seal_declared(&request, &rows).unwrap();
        assert_eq!(artifact.meta().replay, "robust_summary");
        let accounting = artifact.meta().declared_rows.as_ref().expect("accounted");
        assert_eq!(accounting.format, CATEGORICAL_DECLARED_ROWS_FORMAT);
        // The stored fit is the row-based fit of the same declaration, bit for bit.
        assert_eq!(*artifact.fit(), rows_of(&request, &rows), "{kind:?}");
        let bytes = artifact.to_bytes("declared").unwrap();
        let consumed =
            CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
        assert_eq!(consumed.report(), artifact.report(), "{kind:?}");
        // The rows (weights and clusters included) are bound by the digest.
        artifact.verify_data_declared(&request, &rows).unwrap();
        if rows.sampling_weights.is_some() || rows.cluster.is_some() {
            assert!(artifact.verify_data(&request).is_err(), "{kind:?}: plain rows are other rows");
        }
    }
}

#[test]
fn zero_weight_rows_are_deleted_counted_and_the_weights_move_the_identity() {
    let request = declared_request(CovarianceKindWire::Hc1);
    let rows = CategoricalRowsRequest { sampling_weights: Some(weights()), ..Default::default() };
    let artifact = seal_declared(&request, &rows).unwrap();
    let accounting = artifact.meta().declared_rows.as_ref().unwrap();
    assert_eq!((accounting.rows_used, accounting.zero_weight_rows), (8, 1));
    assert!((accounting.weight_sum.unwrap() - 15.0).abs() < 1e-12);
    assert!(accounting.weight_effective_sample_size.unwrap() < 8.0);
    assert_eq!(artifact.meta().n_rows, 8);
    let counts: Vec<u64> = artifact.report().result.counts.iter().map(|c| c.rows).collect();
    assert_eq!(counts, [2, 2, 4], "the zero-weight row of level b is deleted before counting");
    // Another weight vector is another design and another data digest; equal weights at another
    // scale are the same weighted fit but still other declared data.
    let mut other = weights();
    other[4] = 3.5;
    let moved = seal_declared(
        &request,
        &CategoricalRowsRequest { sampling_weights: Some(other), ..Default::default() },
    )
    .unwrap();
    assert_ne!(moved.identity().design_id, artifact.identity().design_id);
    assert!(moved.verify_data_declared(&request, &rows).is_err());
    // Plain rows keep the historical digests: no section, same bytes as the plain entry point.
    let plain = declared_request(CovarianceKindWire::Hc1);
    let a = CategoricalTreatmentArtifact::seal(&plain).unwrap();
    let b = seal_declared(&plain, &CategoricalRowsRequest::default()).unwrap();
    assert_eq!(a.to_bytes("p").unwrap(), b.to_bytes("p").unwrap());
    assert!(a.meta().declared_rows.is_none());
}

#[test]
fn a_resealed_removal_of_the_declaration_or_a_changed_lag_is_refused() {
    let request = declared_request(CovarianceKindWire::NeweyWest);
    let rows = CategoricalRowsRequest { serial_lag: Some(2), ..Default::default() };
    let artifact = seal_declared(&request, &rows).unwrap();
    let bytes = artifact.to_bytes("declared").unwrap();
    let (meta, numbers) = decode_parts(&bytes).unwrap();
    // Dropping the accounting changes the design identity: the stored identity no longer replays.
    let mut stripped = meta.clone();
    stripped.declared_rows = None;
    let reencoded = encode_parts(&stripped, &numbers, "stripped").unwrap();
    assert!(CategoricalTreatmentArtifact::from_bytes(&reencoded, None).is_err());
    let mut longer = meta.clone();
    longer.declared_rows.as_mut().unwrap().serial_lag = Some(3);
    let reencoded = encode_parts(&longer, &numbers, "longer").unwrap();
    assert!(CategoricalTreatmentArtifact::from_bytes(&reencoded, None).is_err());
    // Another lag, sealed honestly, is another covariance identity.
    let other = seal_declared(
        &request,
        &CategoricalRowsRequest { serial_lag: Some(3), ..Default::default() },
    )
    .unwrap();
    assert_ne!(other.identity().covariance_id, artifact.identity().covariance_id);
}

#[test]
fn a_declaration_without_its_covariance_kind_is_refused_by_name_and_never_dropped() {
    let detail = |request: &CategoricalTreatmentRequest, rows: CategoricalRowsRequest| {
        seal_declared(request, &rows).unwrap_err().to_string()
    };
    let hc = declared_request(CovarianceKindWire::Hc1);
    let clusters = vec![0, 0, 1, 1, 2, 2, 3, 3, 3];
    assert!(
        detail(
            &hc,
            CategoricalRowsRequest { cluster: Some(clusters.clone()), ..Default::default() }
        )
        .contains("vector_treatment.cluster_unread")
    );
    assert!(
        detail(&hc, CategoricalRowsRequest { serial_lag: Some(1), ..Default::default() })
            .contains("vector_treatment.lag_unread")
    );
    let model = declared_request(CovarianceKindWire::ModelBased);
    assert!(
        detail(
            &model,
            CategoricalRowsRequest { sampling_weights: Some(weights()), ..Default::default() }
        )
        .contains("weights_need_robust_covariance")
    );
    let cluster_kind = declared_request(CovarianceKindWire::ClusterRobust);
    assert!(
        detail(&cluster_kind, CategoricalRowsRequest::default())
            .contains("vector_treatment.cluster_ids_required")
    );
    let one = vec![1u32; 9];
    assert!(
        detail(&cluster_kind, CategoricalRowsRequest { cluster: Some(one), ..Default::default() })
            .contains("vector_treatment.too_few_clusters")
    );
    let newey = declared_request(CovarianceKindWire::NeweyWest);
    assert!(
        detail(&newey, CategoricalRowsRequest::default()).contains("vector_treatment.lag_required")
    );
    for lag in [0u64, 9, 40] {
        assert!(
            detail(&newey, CategoricalRowsRequest { serial_lag: Some(lag), ..Default::default() })
                .contains("vector_treatment.invalid_lag")
        );
    }
    let mut negative = weights();
    negative[2] = -1.0;
    assert!(
        detail(
            &hc,
            CategoricalRowsRequest { sampling_weights: Some(negative), ..Default::default() }
        )
        .contains("sampling_weights.invalid")
    );
    assert!(
        detail(
            &hc,
            CategoricalRowsRequest { sampling_weights: Some(vec![1.0; 8]), ..Default::default() }
        )
        .contains("categorical_treatment.row_count_mismatch")
    );
}

#[test]
fn the_gaussian_posterior_is_refused_for_declared_rows_with_a_next_attempt() {
    let request = declared_request(CovarianceKindWire::Hc1);
    let artifact = seal_declared(
        &request,
        &CategoricalRowsRequest { sampling_weights: Some(weights()), ..Default::default() },
    )
    .unwrap();
    let declaration = CategoricalPosteriorDeclaration {
        level_precision: 1.0,
        nuisance_precision: 0.01,
        variance_shape: 2.0,
        variance_scale: 2.0,
        credible_level: 0.95,
    };
    let text = artifact.with_posterior(&declaration).unwrap_err().to_string();
    assert!(text.contains("categorical_treatment.posterior_declared_rows"), "{text}");
}

#[test]
fn posterior_declaration_is_bound_to_retained_identity() {
    let plain = CategoricalTreatmentArtifact::seal(&abc_request()).unwrap();
    let plain_identity = plain.identity().clone();
    let plain_bytes = plain.to_bytes("plain").unwrap();
    assert!(plain_identity.posterior_declaration_id.is_none());
    assert!(CategoricalTreatmentArtifact::from_bytes(&plain_bytes, Some(&plain_identity)).is_ok());
    let declaration = CategoricalPosteriorDeclaration {
        level_precision: 1.0,
        nuisance_precision: 1.0,
        variance_shape: 2.0,
        variance_scale: 2.0,
        credible_level: 0.95,
    };
    let original = plain.clone().with_posterior(&declaration).unwrap();
    assert_ne!(original.identity().digest, plain_identity.digest);
    let bytes = original.to_bytes("original").unwrap();
    assert!(CategoricalTreatmentArtifact::from_bytes(&bytes, Some(original.identity())).is_ok());

    let mut variants = vec![declaration; 5];
    variants[0].level_precision = 2.0;
    variants[1].nuisance_precision = 2.0;
    variants[2].variance_shape = 3.0;
    variants[3].variance_scale = 3.0;
    variants[4].credible_level = 0.9;
    for variant in variants {
        let replacement = plain.clone().with_posterior(&variant).unwrap();
        let changed = replacement.to_bytes("changed").unwrap();
        assert!(CategoricalTreatmentArtifact::from_bytes(&changed, None).is_ok());
        assert_eq!(
            CategoricalTreatmentArtifact::from_bytes(&changed, Some(original.identity()))
                .unwrap_err(),
            CategoricalTreatmentArtifactError::IdentityMismatch { field: "posterior_declaration" },
        );
    }

    // Posterior addition and removal must not preserve a retained identity.
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&bytes, Some(&plain_identity)).unwrap_err(),
        CategoricalTreatmentArtifactError::IdentityMismatch { field: "posterior_declaration" },
    );
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&plain_bytes, Some(original.identity()))
            .unwrap_err(),
        CategoricalTreatmentArtifactError::IdentityMismatch { field: "posterior_declaration" },
    );
    // Reattaching a posterior replaces the declaration rather than nesting identities.
    let replaced = original.clone().with_posterior(&variants_for_replacement(declaration)).unwrap();
    let direct = plain.clone().with_posterior(&variants_for_replacement(declaration)).unwrap();
    assert_eq!(replaced.identity(), direct.identity());

    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.posterior.as_mut().unwrap().declaration.level_precision = 2.0;
    let forged = encode_parts(&meta, &numbers, "stale").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&forged, None).unwrap_err(),
        CategoricalTreatmentArtifactError::IdentityMismatch { field: "posterior_declaration" },
    );
    // Historical posterior-bearing v1 identities have no authenticated declaration.
    let (mut legacy, numbers) = decode_parts(&bytes).unwrap();
    legacy.identity = plain_identity;
    let legacy_bytes = encode_parts(&legacy, &numbers, "legacy").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&legacy_bytes, None).unwrap_err(),
        CategoricalTreatmentArtifactError::IdentityMismatch { field: "posterior_declaration" },
    );
}

fn variants_for_replacement(
    mut declaration: CategoricalPosteriorDeclaration,
) -> CategoricalPosteriorDeclaration {
    declaration.level_precision = 2.0;
    declaration
}
