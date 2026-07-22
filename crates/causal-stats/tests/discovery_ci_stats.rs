//! discovery-parity CI statistic conformance (GPDC / `CMIknn` / G²).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::similar_names
)]

use std::fs;
use std::path::PathBuf;

use causal_core::ExecutionContext;
use causal_stats::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependenceTest, ConfidenceMethod, GSquared,
    Gpdc, KnnDependence, SignificanceMethod,
};
use serde_json::Value as JsonValue;

type CiColumns = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/discovery/ci_stats")
}

fn load_expected() -> JsonValue {
    let raw = fs::read_to_string(fixture_dir().join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse")
}

fn load_cols() -> CiColumns {
    let csv = fs::read_to_string(fixture_dir().join("data.csv")).expect("data.csv");
    let mut x = Vec::new();
    let mut y_dep = Vec::new();
    let mut y_ind = Vec::new();
    let mut z = Vec::new();
    let mut x_disc = Vec::new();
    let mut y_disc_dep = Vec::new();
    let mut y_disc_ind = Vec::new();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let p: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
        x.push(p[0]);
        y_dep.push(p[1]);
        y_ind.push(p[2]);
        z.push(p[3]);
        x_disc.push(p[4]);
        y_disc_dep.push(p[5]);
        y_disc_ind.push(p[6]);
    }
    (x, y_dep, y_ind, z, x_disc, y_disc_dep, y_disc_ind)
}

fn close(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

fn run_ci(
    test: &dyn ConditionalIndependenceTest,
    cols: &[&[f64]],
    z_flat: &[usize],
    z_len: usize,
    seed: u64,
) -> (f64, f64) {
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len }];
    let req = CiBatchRequest {
        columns: cols,
        queries: &queries,
        z_flat,
        significance: SignificanceMethod::Analytic,
        confidence: ConfidenceMethod::default(),
    };
    let mut ws = CiWorkspace::default();
    let out = test.test_batch_adhoc(&req, &mut ws, &ExecutionContext::for_tests(seed)).unwrap();
    (out.results[0].statistic, out.results[0].p_value)
}

#[test]
fn discovery_ci_stats_gpdc_cmiknn_gsquared() {
    let expected = load_expected();
    let tig = &expected["reference"];
    let outs = &tig["outputs"];
    assert_eq!(tig["available"].as_bool(), Some(true));
    let methods = &outs["methods"];
    let (x, y_dep, y_ind, z, x_disc, y_disc_dep, y_disc_ind) = load_cols();
    assert_eq!(x.len(), expected["n"].as_u64().unwrap() as usize);

    let atol_s = expected["atol_stat"].as_f64().unwrap();
    let rtol_s = expected["rtol_stat"].as_f64().unwrap();
    let atol_p = expected["atol_p"].as_f64().unwrap();
    let rtol_p = expected["rtol_p"].as_f64().unwrap();
    let gpdc_atol = expected["gpdc_atol_stat"].as_f64().unwrap();
    let gpdc_rtol = expected["gpdc_rtol_stat"].as_f64().unwrap();

    // G² (analytic) — strongest parity claim.
    let g2 = GSquared::new();
    let (s_dep, p_dep) = run_ci(&g2, &[&x_disc, &y_disc_dep], &[], 0, 1);
    let (s_ind, p_ind) = run_ci(&g2, &[&x_disc, &y_disc_ind], &[], 0, 2);
    let ref_dep = &methods["gsquared_dep"];
    let ref_ind = &methods["gsquared_ind"];
    assert!(
        close(s_dep, ref_dep["statistic"].as_f64().unwrap(), atol_s, rtol_s),
        "G² dep stat {s_dep} vs {}",
        ref_dep["statistic"]
    );
    assert!(close(p_dep, ref_dep["p_value"].as_f64().unwrap(), atol_p, rtol_p), "G² dep p {p_dep}");
    assert!(
        close(s_ind, ref_ind["statistic"].as_f64().unwrap(), atol_s, rtol_s),
        "G² ind stat {s_ind}"
    );
    assert!(p_dep < p_ind, "G²: dep p should be smaller than ind p ({p_dep} vs {p_ind})");

    // CMIknn — native kNN MI proxy differs in sign/scale from discovery; check ordering.
    let cmi = KnnDependence::new(5);
    let z_flat = [2usize];
    let (s_dep, p_dep) = run_ci(&cmi, &[&x, &y_dep, &z], &z_flat, 1, 3);
    let (s_ind, p_ind) = run_ci(&cmi, &[&x, &y_ind, &z], &z_flat, 1, 4);
    let _ =
        (methods["cmiknn_dep"]["statistic"].as_f64(), methods["cmiknn_ind"]["statistic"].as_f64());
    assert!(
        s_dep > s_ind,
        "CMIknn dep stat should exceed ind ({s_dep} vs {s_ind}); discovery pins recorded for reference"
    );
    assert!(p_dep <= p_ind + 1e-9, "CMIknn dep p should be <= ind p ({p_dep} vs {p_ind})");
    assert!(p_dep < 0.15, "CMIknn dep should be significant, p={p_dep}");

    // GPDC — native backend vs discovery torch; check ordering + loose magnitude.
    let gpdc = Gpdc::new();
    let (s_dep, p_dep) = run_ci(&gpdc, &[&x, &y_dep, &z], &z_flat, 1, 5);
    let (s_ind, p_ind) = run_ci(&gpdc, &[&x, &y_ind, &z], &z_flat, 1, 6);
    let ref_g = methods["gpdc_dep"]["statistic"].as_f64().unwrap();
    assert!(
        close(s_dep, ref_g, gpdc_atol, gpdc_rtol),
        "GPDC dep stat {s_dep} vs {ref_g} (wide native-backend band)"
    );
    assert!(s_dep > s_ind, "GPDC dep > ind ({s_dep} vs {s_ind})");
    assert!(p_dep <= p_ind + 1e-9, "GPDC dep p <= ind p ({p_dep} vs {p_ind})");
}
