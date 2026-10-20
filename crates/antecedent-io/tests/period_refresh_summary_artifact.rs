//! The summary-only form of a refreshed least-squares fit: it carries the retained summary and
//! the stated retention, no row, and a consumer verifies the seal and that the stored
//! coefficients and residual variance are what the stored summary gives. The full ledger reader
//! refuses it, so a reduced result is never read as a full one.

use antecedent_io::IoError;
use antecedent_io::period_refresh_artifact::{
    PERIOD_REFRESH_SUMMARY_FEATURE, PERIOD_REFRESH_SUMMARY_KIND, PERIOD_REFRESH_SUMMARY_RETENTION,
    PERIOD_REFRESH_SUMMARY_VERSION, PeriodRefreshArtifactWire, PeriodRefreshSummaryWire,
    SummaryLedgerWire,
};

const X: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
const Y: [f64; 5] = [1.1, 2.9, 5.2, 6.8, 9.1];

fn message(error: &IoError) -> String {
    error.to_string()
}

fn fixture() -> PeriodRefreshSummaryWire {
    let n = X.len();
    let (mut s00, mut s01, mut s11, mut t0, mut t1, mut yty) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (x, y) in X.iter().zip(&Y) {
        s00 += 1.0;
        s01 += x;
        s11 += x * x;
        t0 += y;
        t1 += x * y;
        yty += y * y;
    }
    let det = s00 * s11 - s01 * s01;
    let beta = vec![(s11 * t0 - s01 * t1) / det, (s00 * t1 - s01 * t0) / det];
    let sse: f64 =
        X.iter().zip(&Y).map(|(x, y)| (y - beta[0] - beta[1] * x).powi(2)).sum::<f64>();
    PeriodRefreshSummaryWire {
        version: PERIOD_REFRESH_SUMMARY_VERSION,
        required_features: vec![PERIOD_REFRESH_SUMMARY_FEATURE.into()],
        kind: PERIOD_REFRESH_SUMMARY_KIND.into(),
        retention: PERIOD_REFRESH_SUMMARY_RETENTION.into(),
        source: "line".into(),
        ncols: 2,
        n: n as u64,
        xtx: vec![s00, s01, s01, s11],
        xty: vec![t0, t1],
        yty,
        beta: Some(beta),
        residual_variance: Some(sse / (n as f64 - 2.0)),
        source_digest: "digest-of-the-ordered-rows".into(),
        ledger: SummaryLedgerWire { rows: n as u64, live_rows: n as u64, effective_n: n as u64 },
        seal: String::new(),
    }
}

#[test]
fn a_summary_exports_and_a_consumer_serves_the_same_summary() {
    let sealed = fixture().checked().unwrap();
    let bytes = sealed.export().unwrap();
    drop(sealed);
    let consumed = PeriodRefreshSummaryWire::consume(&bytes).unwrap();
    assert_eq!(consumed, fixture().checked().unwrap());
    assert_eq!(consumed.retention, PERIOD_REFRESH_SUMMARY_RETENTION);
    assert_eq!(consumed.n, 5);
    let beta = consumed.beta.as_ref().unwrap();
    assert!((beta[1] - 2.0).abs() < 0.1 && (beta[0] - 1.0).abs() < 0.2, "{beta:?}");
    assert!(consumed.residual_variance.unwrap() > 0.0);
}

#[test]
fn a_tampered_byte_is_refused() {
    let bytes = fixture().checked().unwrap().export().unwrap();
    for position in 0..bytes.len() {
        let mut changed = bytes.clone();
        changed[position] ^= 0x01;
        if changed == bytes {
            continue;
        }
        // A changed byte either fails to decode or fails the seal and the checks; it is never
        // accepted as the same summary.
        if let Ok(wire) = PeriodRefreshSummaryWire::consume(&changed) {
            assert_eq!(wire, fixture().checked().unwrap(), "byte {position} changed the summary");
        }
    }
}

#[test]
fn a_changed_field_under_the_old_seal_is_refused() {
    let mut wire = fixture().checked().unwrap();
    wire.n += 1;
    wire.ledger.effective_n += 1;
    assert!(message(&wire.verify().unwrap_err()).contains("seal"));
    let mut wire = fixture().checked().unwrap();
    wire.seal = "0".repeat(wire.seal.len());
    assert!(message(&PeriodRefreshSummaryWire::consume(&wire.export().unwrap()).unwrap_err())
        .contains("seal"));
}

#[test]
fn a_resealed_inconsistent_summary_is_refused() {
    let mut wrong_beta = fixture();
    wrong_beta.beta = Some(vec![0.0, 3.0]);
    assert!(message(&wrong_beta.checked().unwrap_err()).contains("least_squares"));

    let mut wrong_variance = fixture();
    wrong_variance.residual_variance = Some(40.0);
    assert!(message(&wrong_variance.checked().unwrap_err()).contains("residual_variance"));

    let mut wrong_ledger = fixture();
    wrong_ledger.ledger.effective_n += 1;
    assert!(message(&wrong_ledger.checked().unwrap_err()).contains("ledger"));

    let mut wrong_retention = fixture();
    wrong_retention.retention = "full reusable state".into();
    assert!(message(&wrong_retention.checked().unwrap_err()).contains("retention"));

    let mut wrong_shape = fixture();
    wrong_shape.xty.pop();
    assert!(message(&wrong_shape.checked().unwrap_err()).contains("summary_shape"));
}

#[test]
fn a_summary_is_not_read_as_the_full_ledger() {
    let bytes = fixture().checked().unwrap().export().unwrap();
    let error = PeriodRefreshArtifactWire::consume(&bytes).unwrap_err();
    let text = message(&error);
    assert!(text.contains("period_refresh.summary_only"), "{text}");
    assert!(text.contains("keeps no rows"), "{text}");
}
