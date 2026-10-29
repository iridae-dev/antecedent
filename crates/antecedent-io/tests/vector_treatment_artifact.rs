//! Joint vector-treatment artifact: hand-derived coefficients and full covariance through the
//! container, a recomputing consumer (summary replay for every covariance kind, bit-identical to
//! the row-based fit and checked against an independent sandwich computation), the absence of a
//! row ceiling, the computation and export budgets, the digest binding of a summary to its rows,
//! resealed-mutation refusals against a retained identity, unresealed tampering, unknown
//! versions and the unmeasured calibration coordinate.
//!
//! Frozen dataset (n = 6, columns `1, t1, t2`): `t1 = [1,0,1,0,0,1]`, `t2 = [1,1,1,0,0,0]`,
//! `y = [3,2,5,1,0,4]`. `b = [2.75, 0.75]`, `RSS = 3.25`, `sigma^2 = 3.25 / 3`, and the treatment
//! block of the model-based covariance is `V = (3.25 / 36) [[9, -3], [-3, 9]]`. Calibration
//! (coverage, Type I error, power) is deliberately NOT measured here.
#![allow(clippy::float_cmp, reason = "bit-identical replay is the property under test")]

use antecedent_core::ExecutionContext;
use antecedent_estimate::vector_treatment::{
    Contrast, NamedColumn, TreatmentColumn, VectorCovariance, VectorTreatmentInput,
    VectorTreatmentOptions, fit_vector_treatment,
};
use antecedent_io::vector_treatment_artifact::{
    AdjustmentColumnRequest, ContrastWire, TreatmentRequest, VECTOR_TREATMENT_ARTIFACT_VERSION,
    VECTOR_TREATMENT_CALIBRATION, VectorTreatmentArtifact, VectorTreatmentArtifactError,
    VectorTreatmentRequest, decode_parts, encode_parts,
};
use antecedent_kernels::erfc;

const X1: [f64; 6] = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
const X2: [f64; 6] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
const W: [f64; 6] = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
const Y: [f64; 6] = [3.0, 2.0, 5.0, 1.0, 0.0, 4.0];
const S: f64 = 3.25 / 36.0;

#[path = "common/mod.rs"]
mod common;

fn close(actual: f64, expected: f64, what: &str) {
    common::close(actual, expected, 1e-9, what);
}

fn treatment(name: &str, values: &[f64], adjustment: &[&str]) -> TreatmentRequest {
    TreatmentRequest {
        name: name.into(),
        values: values.to_vec(),
        adjustment_set: adjustment.iter().map(|a| (*a).to_string()).collect(),
        row_snapshot: "snap".into(),
    }
}

fn contrasts() -> Vec<ContrastWire> {
    vec![
        ContrastWire {
            name: "diff".into(),
            weights: vec![("t1".into(), 1.0), ("t2".into(), -1.0)],
        },
        ContrastWire { name: "sum".into(), weights: vec![("t1".into(), 1.0), ("t2".into(), 1.0)] },
    ]
}

fn request() -> VectorTreatmentRequest {
    VectorTreatmentRequest {
        outcome: Y.to_vec(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        treatments: vec![treatment("t1", &X1, &[]), treatment("t2", &X2, &[])],
        covariance: "model_based".into(),
        contrasts: contrasts(),
    }
}

fn adjusted_request() -> VectorTreatmentRequest {
    VectorTreatmentRequest {
        adjustment: vec![AdjustmentColumnRequest { name: "w".into(), values: W.to_vec() }],
        treatments: vec![treatment("t1", &X1, &["w"]), treatment("t2", &X2, &["w"])],
        ..request()
    }
}

fn export(request: &VectorTreatmentRequest) -> (VectorTreatmentArtifact, Vec<u8>) {
    let artifact = VectorTreatmentArtifact::seal(request).unwrap();
    let bytes = artifact.to_bytes("vector-test").unwrap();
    (artifact, bytes)
}

#[test]
fn vector_artifact_round_trip_matches_hand_algebra_and_full_covariance() {
    let (artifact, bytes) = export(&request());
    let report =
        VectorTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap().report();
    assert_eq!(report.treatments, ["t1", "t2"]);
    assert_eq!(report.replay, "summary");
    assert_eq!(report.covariance_kind, "model_based");
    let result = &report.result;
    assert_eq!(result.coefficients[0].name, "t1");
    close(result.coefficients[0].estimate, 2.75, "b1");
    close(result.coefficients[1].estimate, 0.75, "b2");
    // The FULL covariance, off-diagonals included.
    close(result.covariance[0], 9.0 * S, "V11");
    close(result.covariance[1], -3.0 * S, "V12");
    close(result.covariance[2], -3.0 * S, "V21");
    close(result.covariance[3], 9.0 * S, "V22");
    assert_eq!((result.n_rows, result.residual_df), (6, 3));
    close(result.residual_variance, 3.25 / 3.0, "sigma^2");
    // Contrast variance uses the off-diagonal: 24 s against the naive 18 s.
    let diff = &result.contrasts[0];
    close(diff.estimate, 2.0, "b1 - b2");
    close(diff.standard_error.powi(2), 24.0 * S, "diff variance");
    close(diff.naive_independent_standard_error.powi(2), 18.0 * S, "naive diff variance");
    let sum = &result.contrasts[1];
    close(sum.standard_error.powi(2), 12.0 * S, "sum variance");
}

#[test]
fn vector_artifact_joint_wald_and_holm_match_closed_forms() {
    let report = export(&request()).0.report();
    // Wald = b' V^-1 b = 85.5 / (72 s) = 85.5 / 6.5; df 2 so p = exp(-stat / 2).
    let statistic = 85.5 / 6.5;
    close(report.result.joint_wald.statistic, statistic, "Wald");
    assert_eq!(report.result.joint_wald.degrees_of_freedom, 2);
    close(report.result.joint_wald.p_value, (-statistic / 2.0).exp(), "Wald p");
    assert_eq!(report.null, "all treatment coefficients are zero");
    // Holm over the declared two-contrast family.
    let raw: Vec<f64> = report
        .result
        .contrasts
        .iter()
        .map(|c| erfc((c.estimate / c.standard_error).abs() / 2.0_f64.sqrt()))
        .collect();
    for (c, p) in report.result.contrasts.iter().zip(&raw) {
        close(c.p_value, *p, "raw p");
    }
    let (low, high) = if raw[0] <= raw[1] { (0, 1) } else { (1, 0) };
    let holm_low = (2.0 * raw[low]).min(1.0);
    close(report.result.contrasts[low].p_holm, holm_low, "Holm of the smaller p");
    close(report.result.contrasts[high].p_holm, holm_low.max(raw[high]), "Holm of the larger p");
}

#[test]
fn vector_artifact_calibration_is_unmeasured_and_says_so() {
    let report = export(&request()).0.report();
    assert_eq!(report.calibration, VECTOR_TREATMENT_CALIBRATION);
    assert_eq!(report.calibration, "unmeasured");
    assert_eq!(report.inference_claim, "asymptotic_wald_calibration_unmeasured");
    assert_eq!(report.version, VECTOR_TREATMENT_ARTIFACT_VERSION);
    assert!(report.caveats[0].contains("calibration is unmeasured"), "{:?}", report.caveats);
    assert!(report.caveats[0].contains("no interval"), "{:?}", report.caveats);
    assert!(report.caveats[1].contains("caller owns"), "{:?}", report.caveats);
}

#[test]
fn vector_artifact_summary_replay_is_bit_identical_to_the_core_fit() {
    let artifact = VectorTreatmentArtifact::seal(&adjusted_request()).unwrap();
    let input = VectorTreatmentInput {
        outcome: Y.to_vec().into(),
        row_snapshot: "snap".into(),
        adjustment: vec![NamedColumn::new("w", W.to_vec())],
        treatments: ["t1", "t2"]
            .iter()
            .zip([X1, X2])
            .map(|(name, values)| TreatmentColumn {
                name: (*name).into(),
                values: values.to_vec().into(),
                adjustment_set: vec!["w".into()],
                row_snapshot: "snap".into(),
            })
            .collect(),
    };
    let options = VectorTreatmentOptions {
        covariance: VectorCovariance::ModelBased,
        contrasts: vec![Contrast {
            name: "diff".into(),
            weights: vec![("t1".into(), 1.0), ("t2".into(), -1.0)],
        }],
    };
    let core = fit_vector_treatment(&input, &options).unwrap();
    let mut one = adjusted_request();
    one.contrasts.truncate(1);
    let sealed = VectorTreatmentArtifact::seal(&one).unwrap();
    let fit = sealed.fit();
    assert_eq!(fit.covariance, core.covariance);
    assert_eq!(fit.coefficients, core.coefficients);
    assert_eq!(fit.contrasts, core.contrasts);
    assert_eq!(fit.joint_wald, core.joint_wald);
    assert_eq!(fit.residual_variance, core.residual_variance);
    // The wider request carries the same adjustment declaration.
    assert_eq!(artifact.meta().adjustment, ["w"]);
}

/// Deterministic pseudo-random draws in `[-1, 1)` (a 64-bit LCG), enough to make a heteroskedastic
/// design without a dependency.
struct Draws(u64);

impl Draws {
    fn next(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 52) as f64 - 1.0
    }
}

/// `n` rows of one adjustment column `w`, two treatments and a heteroskedastic outcome.
fn synthetic(n: usize, covariance: &str) -> VectorTreatmentRequest {
    synthetic_seeded(n, covariance, 0x5eed)
}

/// [`synthetic`] on the stream started from `seed`.
fn synthetic_seeded(n: usize, covariance: &str, seed: u64) -> VectorTreatmentRequest {
    let mut draws = Draws(seed);
    let (mut w, mut a, mut b, mut y) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let wi = draws.next();
        let ai = wi + draws.next();
        let bi = 0.5 * ai + draws.next();
        y.push(1.0 + 0.5 * wi + 2.0 * ai - bi + (0.5 + wi * wi) * draws.next());
        w.push(wi);
        a.push(ai);
        b.push(bi);
    }
    VectorTreatmentRequest {
        outcome: y,
        row_snapshot: "snap".into(),
        adjustment: vec![AdjustmentColumnRequest { name: "w".into(), values: w }],
        treatments: vec![treatment("t1", &a, &["w"]), treatment("t2", &b, &["w"])],
        covariance: covariance.into(),
        contrasts: contrasts(),
    }
}

/// The core row-based fit of a request.
fn row_fit(
    request: &VectorTreatmentRequest,
    covariance: VectorCovariance,
) -> antecedent_estimate::vector_treatment::VectorTreatmentFit {
    let input = VectorTreatmentInput {
        outcome: request.outcome.as_slice().into(),
        row_snapshot: request.row_snapshot.as_str().into(),
        adjustment: request
            .adjustment
            .iter()
            .map(|c| NamedColumn::new(c.name.as_str(), c.values.as_slice()))
            .collect(),
        treatments: request
            .treatments
            .iter()
            .map(|t| TreatmentColumn {
                name: t.name.as_str().into(),
                values: t.values.as_slice().into(),
                adjustment_set: t.adjustment_set.iter().map(|a| a.as_str().into()).collect(),
                row_snapshot: t.row_snapshot.as_str().into(),
            })
            .collect(),
    };
    let options = VectorTreatmentOptions {
        covariance,
        contrasts: request
            .contrasts
            .iter()
            .map(|c| Contrast { name: c.name.clone(), weights: c.weights.clone() })
            .collect(),
    };
    fit_vector_treatment(&input, &options).unwrap()
}

const ROBUST: [(&str, VectorCovariance); 4] = [
    ("hc0", VectorCovariance::Hc0),
    ("hc1", VectorCovariance::Hc1),
    ("hc2", VectorCovariance::Hc2),
    ("hc3", VectorCovariance::Hc3),
];

#[test]
fn vector_artifact_robust_covariance_replays_from_a_summary() {
    let mut hc = request();
    hc.covariance = "hc1".into();
    let (artifact, bytes) = export(&hc);
    let report = artifact.report();
    assert_eq!(report.replay, "robust_summary");
    // The summary is X'X, X'y, RSS and the meat of a p = 3 design: no row is embedded.
    let (_, numbers) = decode_parts(&bytes).unwrap();
    assert_eq!(numbers.len(), (2 * 9 + 3 + 1) * 8);
    let consumed = VectorTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.report(), report);
    let core = row_fit(&hc, VectorCovariance::Hc1);
    assert_eq!(consumed.fit().covariance, core.covariance);
    // HC1 differs from the model-based covariance on this design.
    assert_ne!(
        consumed.fit().covariance,
        VectorTreatmentArtifact::seal(&request()).unwrap().fit().covariance
    );
}

#[test]
fn vector_summary_replay_equals_the_row_based_fit_for_every_robust_estimator() {
    for (tag, kind) in ROBUST {
        let request = synthetic(400, tag);
        let artifact = VectorTreatmentArtifact::seal(&request).unwrap();
        let core = row_fit(&request, kind);
        // Bit for bit: the summary holds the same meat the row-based sandwich accumulates.
        assert_eq!(artifact.fit(), &core, "{tag}");
        let consumed = VectorTreatmentArtifact::from_bytes(
            &artifact.to_bytes("t").unwrap(),
            Some(artifact.identity()),
        )
        .unwrap();
        assert_eq!(consumed.fit(), &core, "{tag} after the artifact round trip");
        // The estimator is part of the summary: the meats differ between kinds.
        assert_ne!(
            artifact.fit().covariance,
            VectorTreatmentArtifact::seal(&synthetic(400, "model_based")).unwrap().fit().covariance
        );
    }
}

/// Gauss-Jordan inverse of a `p x p` row-major matrix (the oracle's own solver).
fn invert(matrix: &[f64], p: usize) -> Vec<f64> {
    let mut a: Vec<f64> = matrix.to_vec();
    let mut inverse: Vec<f64> = (0..p * p).map(|k| f64::from(k / p == k % p)).collect();
    for col in 0..p {
        let pivot = (col..p)
            .max_by(|&i, &j| a[i * p + col].abs().total_cmp(&a[j * p + col].abs()))
            .unwrap();
        for k in 0..p {
            a.swap(col * p + k, pivot * p + k);
            inverse.swap(col * p + k, pivot * p + k);
        }
        let scale = a[col * p + col];
        for k in 0..p {
            a[col * p + k] /= scale;
            inverse[col * p + k] /= scale;
        }
        for row in (0..p).filter(|&row| row != col) {
            let factor = a[row * p + col];
            for k in 0..p {
                a[row * p + k] -= factor * a[col * p + k];
                inverse[row * p + k] -= factor * inverse[col * p + k];
            }
        }
    }
    inverse
}

/// Independent HC0 to HC3 sandwich covariance of the full coefficient vector, straight from the
/// textbook definitions with a dense inverse.
fn oracle_covariance(request: &VectorTreatmentRequest, kind: &str) -> Vec<f64> {
    let n = request.outcome.len();
    let columns: Vec<&Vec<f64>> = request
        .adjustment
        .iter()
        .map(|c| &c.values)
        .chain(request.treatments.iter().map(|t| &t.values))
        .collect();
    let p = 1 + columns.len();
    let row = |i: usize| -> Vec<f64> {
        std::iter::once(1.0).chain(columns.iter().map(|c| c[i])).collect()
    };
    let mut gram = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for i in 0..n {
        let x = row(i);
        for a in 0..p {
            xty[a] += x[a] * request.outcome[i];
            for b in 0..p {
                gram[a * p + b] += x[a] * x[b];
            }
        }
    }
    let inverse = invert(&gram, p);
    let beta: Vec<f64> =
        (0..p).map(|a| (0..p).map(|b| inverse[a * p + b] * xty[b]).sum()).collect();
    let mut meat = vec![0.0; p * p];
    for i in 0..n {
        let x = row(i);
        let e = request.outcome[i] - x.iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>();
        let h: f64 =
            (0..p).map(|a| x[a] * (0..p).map(|b| inverse[a * p + b] * x[b]).sum::<f64>()).sum();
        let weight = match kind {
            "hc0" => 1.0,
            "hc1" => n as f64 / (n - p) as f64,
            "hc2" => 1.0 / (1.0 - h),
            _ => 1.0 / ((1.0 - h) * (1.0 - h)),
        };
        for a in 0..p {
            for b in 0..p {
                meat[a * p + b] += weight * e * e * x[a] * x[b];
            }
        }
    }
    let mut left = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            left[a * p + b] = (0..p).map(|c| inverse[a * p + c] * meat[c * p + b]).sum();
        }
    }
    let mut full = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            full[a * p + b] = (0..p).map(|c| left[a * p + c] * inverse[c * p + b]).sum();
        }
    }
    full
}

#[test]
fn vector_robust_summary_matches_an_independent_sandwich_oracle() {
    for (tag, _) in ROBUST {
        let request = synthetic(300, tag);
        let artifact = VectorTreatmentArtifact::seal(&request).unwrap();
        let full = oracle_covariance(&request, tag);
        let p = 4;
        let t0 = 2;
        let covariance = &artifact.fit().covariance;
        for i in 0..2 {
            for j in 0..2 {
                let expected = 0.5 * (full[(t0 + i) * p + t0 + j] + full[(t0 + j) * p + t0 + i]);
                let actual = covariance[i * 2 + j];
                assert!(
                    (actual - expected).abs() <= 1e-9 * expected.abs().max(1e-12),
                    "{tag} V[{i}][{j}]: {actual} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn vector_robust_fit_has_no_row_ceiling_and_the_artifact_does_not_grow_with_the_rows() {
    let small = synthetic(300, "hc3");
    let big = synthetic(30_000, "hc3");
    let small_artifact = VectorTreatmentArtifact::seal(&small).unwrap();
    let artifact = VectorTreatmentArtifact::seal(&big).unwrap();
    assert_eq!(artifact.meta().n_rows, 30_000);
    assert_eq!(artifact.fit(), &row_fit(&big, VectorCovariance::Hc3));
    // Same summary size however many rows were fitted.
    let bytes = artifact.to_bytes("big").unwrap();
    assert!(bytes.len() < 2 * small_artifact.to_bytes("small").unwrap().len());
    let consumed = VectorTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.fit(), artifact.fit());
}

#[test]
fn vector_robust_fit_runs_at_one_hundred_thousand_rows() {
    let big = synthetic(100_000, "hc2");
    let artifact = VectorTreatmentArtifact::seal(&big).unwrap();
    assert_eq!(artifact.fit(), &row_fit(&big, VectorCovariance::Hc2));
    let bytes = artifact.to_bytes("big").unwrap();
    assert!(bytes.len() < 64 * 1024);
    assert!(VectorTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).is_ok());
}

/// The first `rows` rows of a request.
fn head(request: &VectorTreatmentRequest, rows: usize) -> VectorTreatmentRequest {
    let mut head = request.clone();
    head.outcome.truncate(rows);
    for column in &mut head.adjustment {
        column.values.truncate(rows);
    }
    for treatment in &mut head.treatments {
        treatment.values.truncate(rows);
    }
    head
}

/// A robust covariance of `kind` at 100,000 rows: the summary fit equals the row-based sandwich
/// bit for bit, the exported summary does not grow with the rows, and on a checkable subsample
/// the covariance equals the independent dense sandwich of this file.
fn assert_robust_covariance_at_100k_rows(kind: &str, kind_enum: VectorCovariance, seed: u64) {
    let started = std::time::Instant::now();
    let big = synthetic_seeded(100_000, kind, seed);
    let artifact = VectorTreatmentArtifact::seal(&big).unwrap();
    assert_eq!(artifact.meta().n_rows, 100_000);
    assert_eq!(
        artifact.fit(),
        &row_fit(&big, kind_enum),
        "{kind}: the summary is the row sandwich"
    );
    let bytes = artifact.to_bytes("big").unwrap();
    assert!(bytes.len() < 64 * 1024);
    let consumed = VectorTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.fit(), artifact.fit());
    assert!(started.elapsed().as_secs() < 600, "{:?}", started.elapsed());

    let sample = head(&big, 4_000);
    let sealed = VectorTreatmentArtifact::seal(&sample).unwrap();
    let full = oracle_covariance(&sample, kind);
    common::assert_block_matches(&sealed.fit().covariance, &full, 4, 2, kind);
}

#[test]
fn vector_treatment_hc0_covariance_100k_rows() {
    assert_robust_covariance_at_100k_rows("hc0", VectorCovariance::Hc0, 3_162_219);
}

#[test]
fn vector_treatment_hc1_covariance_100k_rows() {
    assert_robust_covariance_at_100k_rows("hc1", VectorCovariance::Hc1, 5_418_411);
}

#[test]
fn vector_treatment_hc2_covariance_100k_rows() {
    assert_robust_covariance_at_100k_rows("hc2", VectorCovariance::Hc2, 8_522_155);
}

#[test]
fn vector_treatment_hc3_covariance_100k_rows() {
    assert_robust_covariance_at_100k_rows("hc3", VectorCovariance::Hc3, 5_919_723);
}

#[test]
fn vector_summary_is_bound_to_its_rows_by_a_digest() {
    for tag in ["model_based", "hc2"] {
        let request = synthetic(120, tag);
        let artifact = VectorTreatmentArtifact::seal(&request).unwrap();
        let digest = artifact.meta().data_digest.clone().expect("a summary records its digest");
        assert_eq!(digest.len(), 64);
        assert_eq!(artifact.report().data_digest.as_deref(), Some(digest.as_str()));
        artifact.verify_data(&request).unwrap();
        // One changed value, a renamed column or a reordered row fails the binding.
        let mut value = request.clone();
        value.outcome[7] += 1e-9;
        assert_eq!(
            artifact.verify_data(&value),
            Err(VectorTreatmentArtifactError::IdentityMismatch { field: "data" })
        );
        let mut renamed = request.clone();
        renamed.treatments[0].name = "u1".into();
        assert_eq!(
            artifact.verify_data(&renamed),
            Err(VectorTreatmentArtifactError::IdentityMismatch { field: "data" })
        );
        let mut reordered = request.clone();
        reordered.outcome.swap(0, 1);
        assert_eq!(
            artifact.verify_data(&reordered),
            Err(VectorTreatmentArtifactError::IdentityMismatch { field: "data" })
        );
        // The digest is part of the design identity: a forged one under the stored identity refuses.
        let (meta, numbers) = decode_parts(&artifact.to_bytes("t").unwrap()).unwrap();
        let mut forged = meta;
        forged.data_digest = Some("0".repeat(64));
        let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
        assert_eq!(
            VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
            VectorTreatmentArtifactError::IdentityMismatch { field: "design" }
        );
    }
}

#[test]
fn vector_budgets_refuse_with_precise_details_and_keep_the_fit() {
    let request = synthetic(2_000, "hc1");
    let mut tight = ExecutionContext::production(0, 1);
    tight.set_memory_limit(Some(10_000));
    let error = VectorTreatmentArtifact::seal_with(&request, &tight).unwrap_err();
    let (code, detail, text) = error.refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("route_not_supported", "vector_treatment.memory_budget_exceeded")
    );
    assert!(text.contains("10000-byte"), "{text}");
    let mut roomy = ExecutionContext::production(0, 1);
    roomy.set_memory_limit(Some(10_000_000));
    let artifact = VectorTreatmentArtifact::seal_with(&request, &roomy).unwrap();

    let cancelled = ExecutionContext::production(0, 1);
    cancelled.cancellation.cancel();
    let error = VectorTreatmentArtifact::seal_with(&request, &cancelled).unwrap_err();
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!((code, detail.as_str()), ("cancelled_no_claim", "vector_treatment.cancelled"));

    // Export: an artifact above its budget refuses and names the bytes; the fit survives.
    let bytes = artifact.to_bytes("fit").unwrap();
    assert_eq!(artifact.to_bytes_within("fit", bytes.len()).unwrap(), bytes);
    let error = artifact.to_bytes_within("fit", bytes.len() - 1).unwrap_err();
    let (code, detail, text) = error.refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("route_not_supported", "vector_treatment.export_budget_exceeded")
    );
    assert!(text.contains(&format!("{} bytes", bytes.len())), "{text}");
    assert_eq!(artifact.fit().n_rows, 2_000);

    // Replay runs under the same kind of context.
    let error = VectorTreatmentArtifact::from_bytes_with(&bytes, None, &cancelled).unwrap_err();
    assert_eq!(error.refusal().unwrap().1, "vector_treatment.cancelled");
    assert!(VectorTreatmentArtifact::from_bytes_with(&bytes, None, &roomy).is_ok());
}

#[test]
fn vector_artifact_incompatible_treatments_refuse_with_the_core_detail() {
    let mut other_set = adjusted_request();
    other_set.treatments[1].adjustment_set = vec![];
    let (code, detail, _) =
        VectorTreatmentArtifact::seal(&other_set).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("route_not_supported", "vector_treatment.adjustment_set_mismatch")
    );

    let mut other_snapshot = request();
    other_snapshot.treatments[1].row_snapshot = "elsewhere".into();
    let (code, detail, _) =
        VectorTreatmentArtifact::seal(&other_snapshot).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("route_not_supported", "vector_treatment.row_snapshot_mismatch")
    );

    let mut short = request();
    short.treatments[1].values.pop();
    let (_, detail, _) = VectorTreatmentArtifact::seal(&short).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "vector_treatment.row_count_mismatch");

    let mut constant = request();
    constant.treatments[1].values = vec![1.0; 6];
    let (code, detail, _) =
        VectorTreatmentArtifact::seal(&constant).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("design_rank_deficient", "vector_treatment.treatment_without_variation")
    );

    let mut collinear = request();
    collinear.treatments[1].values = X1.iter().map(|v| 2.0 * v).collect();
    let (_, detail, _) = VectorTreatmentArtifact::seal(&collinear).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "vector_treatment.collinear_treatments");

    let mut single = request();
    single.treatments.truncate(1);
    let (_, detail, _) = VectorTreatmentArtifact::seal(&single).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "vector_treatment.too_few_treatments");

    let mut unknown = request();
    unknown.contrasts[0].weights[0].0 = "zzz".into();
    let (_, detail, _) = VectorTreatmentArtifact::seal(&unknown).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "vector_treatment.unknown_contrast_coefficient");

    let mut bad_kind = request();
    bad_kind.covariance = "cluster".into();
    let (_, detail, _) = VectorTreatmentArtifact::seal(&bad_kind).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "vector_treatment.invalid_request");
}

/// Seal `mutated` as a consumer-consistent artifact and consume it against the identity of
/// `original`: the mutation is resealed (its own digests are correct) but must still refuse.
fn resealed_mutation_refuses(
    original: &VectorTreatmentRequest,
    mutated: &VectorTreatmentRequest,
    field: &'static str,
) {
    let (artifact, _) = export(original);
    let (_, mutated_bytes) = export(mutated);
    assert!(VectorTreatmentArtifact::from_bytes(&mutated_bytes, None).is_ok(), "{field}");
    let error =
        VectorTreatmentArtifact::from_bytes(&mutated_bytes, Some(artifact.identity())).unwrap_err();
    assert_eq!(error, VectorTreatmentArtifactError::IdentityMismatch { field }, "{field}");
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "vector_treatment.wrong_contract");
}

#[test]
fn vector_artifact_resealed_identity_changes_are_refused() {
    let original = adjusted_request();
    let mut snapshot = original.clone();
    snapshot.row_snapshot = "snap-2".into();
    for t in &mut snapshot.treatments {
        t.row_snapshot = "snap-2".into();
    }
    resealed_mutation_refuses(&original, &snapshot, "row_snapshot");

    let mut renamed = original.clone();
    renamed.adjustment[0].name = "v".into();
    for t in &mut renamed.treatments {
        t.adjustment_set = vec!["v".into()];
    }
    resealed_mutation_refuses(&original, &renamed, "adjustment_set");

    let mut reordered = original.clone();
    reordered.treatments.reverse();
    reordered.contrasts.clear();
    let mut base = original.clone();
    base.contrasts.clear();
    resealed_mutation_refuses(&base, &reordered, "treatment_vector");

    let mut data = original.clone();
    data.outcome[0] = 3.5;
    resealed_mutation_refuses(&original, &data, "design");

    let mut declared = original.clone();
    declared.contrasts.truncate(1);
    resealed_mutation_refuses(&original, &declared, "contrasts");

    let mut kind = original.clone();
    kind.covariance = "hc0".into();
    resealed_mutation_refuses(&original, &kind, "design");
}

#[test]
fn vector_artifact_coefficient_order_is_part_of_the_identity() {
    let forward = VectorTreatmentArtifact::seal(&request()).unwrap();
    let mut swapped = request();
    swapped.treatments.reverse();
    swapped.contrasts.clear();
    let reversed = VectorTreatmentArtifact::seal(&swapped).unwrap();
    assert_ne!(forward.identity().treatment_id, reversed.identity().treatment_id);
    assert_ne!(forward.identity().digest, reversed.identity().digest);
    let r = reversed.report();
    assert_eq!(r.treatments, ["t2", "t1"]);
    close(r.result.coefficients[0].estimate, 0.75, "b2 first");
    close(r.result.covariance[1], -3.0 * S, "V off-diagonal is symmetric under reordering");
}

#[test]
fn vector_artifact_unresealed_tampering_does_not_replay() {
    let (artifact, bytes) = export(&request());
    let (meta, numbers) = decode_parts(&bytes).unwrap();

    let mut forged = meta.clone();
    forged.result.coefficients[0].estimate += 1.0;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::ResultMismatch(_)
    ));

    let mut forged = meta.clone();
    forged.result.covariance[1] = 0.0;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::ResultMismatch(_)
    ));

    let mut forged = meta.clone();
    forged.result.contrasts[0].p_holm *= 0.5;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::ResultMismatch(_)
    ));

    // A changed residual sum of squares (the last embedded number) under the stored identity.
    let mut changed = numbers.clone();
    let last = changed.len() - 8;
    changed[last..].copy_from_slice(&9.0_f64.to_le_bytes());
    let reencoded = encode_parts(&meta, &changed, "forged").unwrap();
    assert_eq!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::IdentityMismatch { field: "design" }
    );
    // A truncated numbers section is malformed.
    let reencoded = encode_parts(&meta, &numbers[..numbers.len() - 8], "forged").unwrap();
    assert!(matches!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::Malformed(_)
    ));

    let mut forged = meta.clone();
    forged.null = "no treatment is non-zero".into();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::IdentityMismatch { field: "null" }
    );
    let mut forged = meta.clone();
    forged.calibration = "calibrated".into();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::UnsupportedSemantics("calibration")
    );
    let mut forged = meta.clone();
    forged.caveats.clear();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::UnsupportedSemantics("caveats")
    );
    // A changed declared contrast under the stored identity.
    let mut forged = meta.clone();
    forged.contrasts[0].weights[1].1 = 2.0;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::IdentityMismatch { field: "contrasts" }
    );
    assert!(VectorTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).is_ok());
}

#[test]
fn vector_artifact_unknown_major_version_and_corrupt_bytes_refuse() {
    let (_, bytes) = export(&request());
    let (meta, numbers) = decode_parts(&bytes).unwrap();
    for version in [0_u16, 2, 7] {
        let mut other = meta.clone();
        other.version = version;
        let reencoded = encode_parts(&other, &numbers, "other-version").unwrap();
        assert_eq!(
            VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
            VectorTreatmentArtifactError::UnsupportedVersion { version }
        );
    }
    let mut other = meta;
    other.feature = "vector_treatment_v2".into();
    let reencoded = encode_parts(&other, &numbers, "other-feature").unwrap();
    assert_eq!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::UnsupportedSemantics("feature marker")
    );
    let mut corrupt = bytes.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0xff;
    assert_eq!(
        format!("{:?}", VectorTreatmentArtifact::from_bytes(&corrupt, None).unwrap_err()),
        r#"Container(ChecksumMismatch { section: "vector_treatment_meta" })"#
    );
    for short in [&bytes[..bytes.len() - 5], &[][..]] {
        assert!(
            matches!(
                VectorTreatmentArtifact::from_bytes(short, None),
                Err(VectorTreatmentArtifactError::Decode(_))
            ),
            "a truncated or empty artifact is a decode refusal"
        );
    }
}

/// The shared shape of the budgeted treatment artifacts: sealed, exported under a byte budget
/// and consumed again through the trait alone.
fn round_trip_through_the_trait<A>(request: &A::Request) -> Vec<u8>
where
    A: antecedent_io::BudgetedTreatmentArtifact,
    A::Error: std::fmt::Debug,
{
    let artifact = A::seal(request).unwrap();
    let bytes = artifact.to_bytes("vector-test").unwrap();
    assert_eq!(artifact.to_bytes_within("vector-test", A::MAX_BYTES).unwrap(), bytes);
    let over_budget = artifact.to_bytes_within("vector-test", bytes.len() - 1).unwrap_err();
    let shown = format!("{over_budget:?}");
    assert!(
        shown.contains("route_not_supported")
            && shown.contains("vector_treatment.export_budget_exceeded"),
        "{shown}"
    );
    A::from_bytes(&bytes, None).unwrap();
    bytes
}

#[test]
fn the_budgeted_trait_agrees_with_the_inherent_entry_points() {
    let request = request();
    let bytes = round_trip_through_the_trait::<VectorTreatmentArtifact>(&request);
    assert_eq!(bytes, export(&request).1);
}
