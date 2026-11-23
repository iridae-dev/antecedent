//! The incremental least-squares summary against exact rational least squares
//! (`oracle_data/incremental_ols_reference.py`).
//!
//! Rows are appended one at a time, as one batch, and as two merged windows; the coefficients
//! and the residual variance (for OLS and for externally supplied coefficients) are compared with
//! the rational solution of the normal equations. The designs span a regressor offset of 0 to 1e8,
//! a response offset of 1e8, a through-origin column, a near-collinear pair and minimum rows.

use antecedent_state::LinearOlsSuffStats;
use serde_json::Value;

fn reference() -> Vec<Value> {
    serde_json::from_str(include_str!("oracle_data/incremental_ols_reference.json"))
        .expect("frozen reference")
}

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

fn decimals(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_str().expect("text").parse().expect("number"))
        .collect()
}

fn rows_of(case: &Value) -> (usize, Vec<Vec<f64>>, Vec<f64>) {
    let rows: Vec<Vec<f64>> = case["rows"].as_array().expect("rows").iter().map(numbers).collect();
    (rows[0].len(), rows, numbers(&case["y"]))
}

/// Largest relative error against the scale of the vector, which is what a coefficient near a
/// large offset can be asked for.
fn vector_error(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
    got.iter().zip(want).map(|(g, w)| (g - w).abs() / scale).fold(0.0, f64::max)
}

fn check(stats: &LinearOlsSuffStats, case: &Value, label: &str, tolerance: f64) {
    let name = case["name"].as_str().expect("name");
    let beta = stats.solve_beta().expect("solve");
    let want = decimals(&case["beta"]);
    assert!(vector_error(&beta, &want) <= tolerance, "{name} {label}: beta {beta:?} vs {want:?}");
    let sigma2 = stats.residual_variance(&beta).expect("variance");
    let want: f64 = case["sigma2"].as_str().expect("s2").parse().expect("number");
    assert!(
        (sigma2 - want).abs() <= tolerance * want.abs(),
        "{name} {label}: sigma2 {sigma2} vs {want}"
    );
    let extra = numbers(&case["extra_beta"]);
    let sigma2 = stats.residual_variance(&extra).expect("variance");
    let want: f64 = case["extra_sigma2"].as_str().expect("s2").parse().expect("number");
    assert!(
        (sigma2 - want).abs() <= tolerance * want.abs(),
        "{name} {label}: external sigma2 {sigma2} vs {want}"
    );
}

fn tolerance(name: &str) -> f64 {
    match name {
        // Regressors 1e-4 apart give coefficients near 1e4, and the residual sum of squares is then
        // a difference of terms near 1e8 taken from the scatter, not from the rows: about 1e-5.
        "near_collinear_pair" => 1e-4,
        // The centred scatter itself is accumulated from doubles whose spacing at 1e8 is 1.5e-8
        // against a regressor spread of one, so the slope is good to about 1e-8 there.
        n if n.contains("offset_1e+08") || n.contains("response_offset") => 1e-7,
        n if n.contains("offset_1e+06") => 1e-9,
        _ => 1e-9,
    }
}

#[test]
fn appended_rows_batches_and_merged_windows_equal_exact_least_squares() {
    for case in &reference() {
        let name = case["name"].as_str().expect("name");
        let (p, rows, y) = rows_of(case);
        let tol = tolerance(name);
        let mut single = LinearOlsSuffStats::new(p).expect("stats");
        for (row, response) in rows.iter().zip(&y) {
            single.append_row(row, *response).expect("append");
        }
        check(&single, case, "rows", tol);
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let mut batch = LinearOlsSuffStats::new(p).expect("stats");
        batch.append_batch(&flat, &y).expect("batch");
        check(&batch, case, "batch", tol);
        let cut = rows.len() / 3;
        let (mut left, mut right) =
            (LinearOlsSuffStats::new(p).unwrap(), LinearOlsSuffStats::new(p).unwrap());
        left.append_batch(&flat[..cut * p], &y[..cut]).unwrap();
        right.append_batch(&flat[cut * p..], &y[cut..]).unwrap();
        let mut merged = left.clone();
        merged.merge(&right).expect("merge");
        check(&merged, case, "merged", tol);
        let mut reversed = right.clone();
        reversed.merge(&left).expect("merge");
        check(&reversed, case, "merged-reversed", tol);
    }
}
