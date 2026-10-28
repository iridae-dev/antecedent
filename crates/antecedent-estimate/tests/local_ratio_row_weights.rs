//! Row weights of the local-polynomial fuzzy regression-discontinuity and kink ratio.
//!
//! An integer row weight stands for that many copies of the row, so the weighted ratio must
//! equal the ratio of the table in which every row is repeated by its weight; a row of weight
//! zero is a deleted row; rescaling every weight changes neither the ratio nor its first stage;
//! and weights that are negative, non-finite or not aligned with the rows are refused. The
//! fixture is a fuzzy jump with a known local effect, so the unweighted fit is also checked
//! against that effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "the fixture indexes small deterministic literals"
)]

use antecedent_estimate::local_polynomial_ratio::{
    LocalPolynomialRatioFit, fit_local_polynomial_ratio, fit_weighted_local_polynomial_ratio,
};

struct Table {
    running: Vec<f64>,
    outcome: Vec<f64>,
    treatment: Vec<f64>,
}

/// Compliance 1/4 below the cutoff and 3/4 above it, a quadratic trend and a jump of 3 in the
/// outcome per unit of treatment: the fuzzy ratio is `3 (3/4 - 1/4) / (3/4 - 1/4) = 3`.
fn table() -> Table {
    let (mut running, mut outcome, mut treatment) = (vec![], vec![], vec![]);
    for step in 1..60 {
        for (sign, assignments) in [(-1.0, [1.0, 0.0, 0.0, 0.0]), (1.0, [1.0, 1.0, 1.0, 0.0])] {
            let score = sign * f64::from(step) / 60.0;
            for (assigned, noise) in assignments.into_iter().zip([-0.02, -0.01, 0.01, 0.02]) {
                running.push(score);
                treatment.push(assigned);
                outcome.push(1.0 + 2.0 * score + 0.5 * score * score + 3.0 * assigned + noise);
            }
        }
    }
    Table { running, outcome, treatment }
}

fn integer_weights(n: usize) -> Vec<f64> {
    (0..n).map(|i| [1.0, 2.0, 3.0, 1.0, 2.0][i % 5]).collect()
}

fn repeated(table: &Table, weights: &[f64]) -> Table {
    let mut out = Table { running: vec![], outcome: vec![], treatment: vec![] };
    for (i, &w) in weights.iter().enumerate() {
        for _ in 0..(w as usize) {
            out.running.push(table.running[i]);
            out.outcome.push(table.outcome[i]);
            out.treatment.push(table.treatment[i]);
        }
    }
    out
}

fn fit(table: &Table, weights: &[f64], kink: bool) -> Result<LocalPolynomialRatioFit, String> {
    fit_weighted_local_polynomial_ratio(
        &table.running,
        &table.outcome,
        &table.treatment,
        weights,
        0.0,
        1.0,
        kink,
    )
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-8 * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

#[test]
fn integer_row_weights_equal_the_table_with_every_row_repeated() {
    let table = table();
    let weights = integer_weights(table.running.len());
    let replicated = repeated(&table, &weights);
    for kink in [false] {
        let weighted = fit(&table, &weights, kink).unwrap();
        let plain = fit(&replicated, &[], kink).unwrap();
        close(weighted.estimate, plain.estimate, "ratio");
        close(weighted.reduced_form, plain.reduced_form, "reduced form");
        close(weighted.first_stage, plain.first_stage, "first stage");
        let unweighted = fit(&table, &[], kink).unwrap();
        assert!(
            (unweighted.reduced_form - weighted.reduced_form).abs() > 1e-6,
            "the weights must move the fit (kink {kink})"
        );
    }
}

#[test]
fn the_unweighted_fuzzy_ratio_recovers_the_known_local_effect() {
    let table = table();
    let fit = fit_local_polynomial_ratio(
        &table.running,
        &table.outcome,
        &table.treatment,
        0.0,
        1.0,
        false,
    )
    .unwrap();
    close(fit.estimate, 3.0, "known fuzzy effect");
    close(fit.first_stage, 0.5, "known first stage");
    // Unit weights are the unweighted fit.
    let unit = fit_weighted_local_polynomial_ratio(
        &table.running,
        &table.outcome,
        &table.treatment,
        &vec![1.0; table.running.len()],
        0.0,
        1.0,
        false,
    )
    .unwrap();
    close(unit.estimate, fit.estimate, "unit weights");
    close(unit.first_stage, fit.first_stage, "unit weights first stage");
}

#[test]
fn a_zero_weight_row_is_a_deleted_row_and_a_rescaling_changes_nothing() {
    let table = table();
    let n = table.running.len();
    let weights = integer_weights(n);
    // Zero out every seventh row; the same fit is the table without those rows.
    let mut zeroed = weights.clone();
    let kept: Vec<usize> = (0..n).filter(|i| i % 7 != 0).collect();
    for i in (0..n).filter(|i| i % 7 == 0) {
        zeroed[i] = 0.0;
    }
    let reduced = Table {
        running: kept.iter().map(|&i| table.running[i]).collect(),
        outcome: kept.iter().map(|&i| table.outcome[i]).collect(),
        treatment: kept.iter().map(|&i| table.treatment[i]).collect(),
    };
    let reduced_weights: Vec<f64> = kept.iter().map(|&i| weights[i]).collect();
    let with_zeros = fit(&table, &zeroed, false).unwrap();
    let deleted = fit(&reduced, &reduced_weights, false).unwrap();
    close(with_zeros.estimate, deleted.estimate, "zero weight is deletion");
    close(with_zeros.first_stage, deleted.first_stage, "zero weight first stage");

    let scaled: Vec<f64> = weights.iter().map(|w| 7.5 * w).collect();
    let base = fit(&table, &weights, false).unwrap();
    let rescaled = fit(&table, &scaled, false).unwrap();
    close(rescaled.estimate, base.estimate, "rescaled ratio");
    close(rescaled.first_stage, base.first_stage, "rescaled first stage");
}

#[test]
fn invalid_misaligned_or_one_sided_weights_are_refused() {
    let table = table();
    let n = table.running.len();
    let mut negative = vec![1.0; n];
    negative[3] = -1.0;
    let mut nan = vec![1.0; n];
    nan[5] = f64::NAN;
    for (what, weights) in [("negative", negative), ("non-finite", nan), ("short", vec![1.0; n - 1])]
    {
        let error = fit(&table, &weights, false).unwrap_err();
        assert!(error.contains("row weights must be finite, non-negative and row-aligned"), "{what}: {error}");
    }
    // Zero weight on every row left of the cutoff leaves no support on that side.
    let one_sided: Vec<f64> = table.running.iter().map(|&r| if r < 0.0 { 0.0 } else { 1.0 }).collect();
    let error = fit(&table, &one_sided, false).unwrap_err();
    assert!(error.contains("full-rank support"), "{error}");
}
