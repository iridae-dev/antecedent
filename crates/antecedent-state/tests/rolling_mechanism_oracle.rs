//! Frozen batch-window oracle for rolling mechanism diagnostics.

use antecedent_state::RollingMechanismDiagnostics;
use serde_json::Value;

const FIXTURE: &str =
    include_str!("../../../conformance/state/rolling_mechanism_diagnostics/expected.json");

fn row(index: usize) -> ([f64; 2], f64) {
    let i = index as f64;
    let x = ((index % 23) as f64 - 11.0) / 7.0 + 0.15 * (0.19 * i).sin();
    let shift = if index >= 90 { 0.4 } else { 0.0 };
    let y = 1.5 - 0.7 * x + shift + 0.2 * (0.37 * i).sin() + 0.05 * (0.11 * i).cos();
    ([1.0, x], y)
}

fn close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: actual={actual:.17e}, expected={expected:.17e}"
    );
}

#[test]
fn rolling_summaries_match_independent_batch_windows() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
    let tolerance = fixture["acceptance"]["absolute_tolerance"].as_f64().expect("tolerance");

    for case in fixture["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("case name");
        let window = case["window"].as_u64().expect("window") as usize;
        let checkpoints = case["checkpoints"].as_array().expect("checkpoints");
        let last = checkpoints.last().expect("last checkpoint")["row"].as_u64().expect("row");
        let mut diagnostics = RollingMechanismDiagnostics::new(2, window).expect("diagnostics");
        let mut checkpoint_index = 0;

        for index in 0..=last as usize {
            let (design, response) = row(index);
            diagnostics.append_row(&design, response).expect("append");
            if index as u64 != checkpoints[checkpoint_index]["row"].as_u64().expect("row") {
                continue;
            }
            diagnostics.refresh_summaries().expect("refresh");
            let expected = &checkpoints[checkpoint_index];
            assert_eq!(diagnostics.n, expected["n"].as_u64().expect("n"), "{name}");
            for coefficient in 0..2 {
                close(
                    diagnostics.beta[coefficient],
                    expected["beta"][coefficient].as_f64().expect("beta"),
                    tolerance,
                    &format!("{name} beta[{coefficient}] at row {index}"),
                );
            }
            close(
                diagnostics.residual_sse,
                expected["sse"].as_f64().expect("sse"),
                tolerance,
                &format!("{name} SSE at row {index}"),
            );
            close(
                diagnostics.residual_var.expect("residual variance"),
                expected["residual_variance"].as_f64().expect("residual variance"),
                tolerance,
                &format!("{name} residual variance at row {index}"),
            );
            close(
                diagnostics.mean_abs_residual,
                expected["mean_absolute_residual"].as_f64().expect("mean absolute residual"),
                tolerance,
                &format!("{name} mean absolute residual at row {index}"),
            );
            close(
                diagnostics.max_abs_cusum.expect("CUSUM"),
                expected["max_absolute_cusum"].as_f64().expect("CUSUM"),
                tolerance,
                &format!("{name} CUSUM at row {index}"),
            );
            checkpoint_index += 1;
            if checkpoint_index == checkpoints.len() {
                break;
            }
        }
        assert_eq!(checkpoint_index, checkpoints.len(), "{name}");
    }
}

#[test]
fn row_and_batch_append_replay_identically() {
    let mut rowwise = RollingMechanismDiagnostics::new(2, 17).expect("rowwise");
    let mut batch = RollingMechanismDiagnostics::new(2, 17).expect("batch");
    let mut rows = Vec::new();
    let mut outcomes = Vec::new();
    for index in 0..150 {
        let (design, response) = row(index);
        rowwise.append_row(&design, response).expect("row append");
        rows.extend_from_slice(&design);
        outcomes.push(response);
    }
    batch.append_batch(&rows, &outcomes).expect("batch append");
    rowwise.refresh_summaries().expect("row refresh");
    batch.refresh_summaries().expect("batch refresh");
    assert_eq!(rowwise.beta, batch.beta);
    assert_eq!(rowwise.residual_sse.to_bits(), batch.residual_sse.to_bits());
    assert_eq!(rowwise.residual_var, batch.residual_var);
    assert_eq!(rowwise.mean_abs_residual.to_bits(), batch.mean_abs_residual.to_bits());
    assert_eq!(rowwise.max_abs_cusum, batch.max_abs_cusum);
}
