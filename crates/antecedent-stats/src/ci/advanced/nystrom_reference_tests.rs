//! The Nystrom (subset-of-regressors) residuals against a 50-digit reference.
//!
//! `tests/oracle_data/gpdc_nystrom_reference.py` (frozen in the JSON beside it) forms the
//! standardised conditioning columns, the RBF cross kernel on the declared inducing rows, the
//! normal equations `K_Rn K_nR + lambda K_RR` with the relative jitter, and the residual of the
//! centred series, in mpmath. The seeded draw of the inducing rows is a declared input of the
//! reference; the test asserts that the product still draws exactly the frozen rows. Cases cover
//! a rank below the sample size, every row inducing, one conditioning column, a conditioning set
//! at scale 1e4 and offset 1e6, tied conditioning values (a rank-deficient kernel), a six-row
//! sample, a wide length scale (a nearly rank-one kernel) and an empty conditioning set.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!(
        "../../../tests/oracle_data/gpdc_nystrom_reference.json"
    ))
    .unwrap()
    .as_array()
    .unwrap()
    .clone()
}

#[test]
fn nystrom_residuals_match_the_fifty_digit_reference() {
    let ctx = ExecutionContext::for_tests(5);
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let data: Vec<Vec<f64>> = case["columns"].as_array().unwrap().iter().map(floats).collect();
        let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
        let n = cols[0].len();
        let z: Vec<usize> =
            case["z"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let gp = GpdcNystrom {
            length_scale: case["ell"].as_f64().unwrap(),
            ridge: case["ridge"].as_f64().unwrap(),
            inducing_points: case["m"].as_u64().unwrap() as usize,
            ..GpdcNystrom::new()
        };
        if !z.is_empty() {
            let m = gp.inducing_points.min(n);
            let rows: Vec<usize> = case["inducing"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(nystrom_inducing_rows(n, m, gp.inducing_seed, &z), rows, "{name}: draw");
        }
        let fit =
            if z.is_empty() { None } else { Some(nystrom_fit(&cols, &z, n, &gp, &ctx).unwrap()) };
        let [rx, ry] = nystrom_residuals(fit.as_ref(), [cols[0], cols[1]], &ctx).unwrap();
        for (got, want) in [(rx, floats(&case["rx"])), (ry, floats(&case["ry"]))] {
            let scale = want.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
            let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            eprintln!("NYSTROM {name} worst {worst:e} scale {scale:e}");
            assert!(worst <= 1e-6 * scale.max(1.0), "{name}: worst gap {worst} at scale {scale}");
        }
    }
}
