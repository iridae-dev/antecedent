//! Closed-form reference for the cross-fitted cell AIPW scores.
//!
//! With no adjustment covariates the nuisances have closed forms: the multinomial-logit
//! propensity over cells is the training-fold cell proportion and the per-cell outcome
//! regression is the training-fold cell mean (or cell exceedance proportion). The reference
//! recomputes every score from the table's own fold ids, cells and outcomes by that
//! definition, `psi_ic = mu_c + 1{cell_i = c} (y_i - mu_c) / e_c`, and the contrast value and
//! standard error from those scores, sharing no code with the estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, MeasurementSpec, OrderedFloatBits, OutcomeFunctional, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};

use super::*;
use crate::overlap::OverlapPolicy;

const N: usize = 330;
const FOLDS: usize = 5;
/// The propensity solver stops at a score norm of 1e-8, which moves a propensity of 0.1 by
/// about 1e-7 and a score by a few times that; the closed form is exact.
const SCORE_TOLERANCE: f64 = 2e-6;

/// Unequal cell sizes (roughly 40, 30, 20, 10 percent) and an outcome that depends on the cell
/// nonlinearly, built from deterministic sequences.
fn data() -> (TabularData, Vec<u32>) {
    let mut a = Vec::new();
    let mut d = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();
    for i in 0..N {
        let u = ((i * 37) % 100) as f64 / 100.0;
        let (ai, di) = match u {
            u if u < 0.4 => (0.0, 0.0),
            u if u < 0.7 => (1.0, 0.0),
            u if u < 0.9 => (0.0, 1.0),
            _ => (1.0, 1.0),
        };
        a.push(ai);
        d.push(di);
        // A binary covariate related to the cell and the outcome.
        let zi = f64::from((i * 7 + (ai as usize) * 3 + (di as usize)) % 10 < 4);
        z.push(zi);
        y.push(0.8 * zi + 0.5 + 1.2 * ai - 0.7 * di + 2.0 * ai * di + (0.9 * i as f64).sin() + 0.01 * i as f64);
    }
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("a", RoleHint::TreatmentCandidate),
        ("d", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns = [a, d, y, z]
        .into_iter()
        .enumerate()
        .map(|(k, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(k as u32),
                    Arc::from(v),
                    ValidityBitmap::all_valid(N),
                )
                .unwrap(),
            )
        })
        .collect();
    let folds = (0..N).map(|i| (i % FOLDS) as u32).collect();
    (TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap()), folds)
}

fn fit(functional: &OutcomeFunctional, folds: &[u32]) -> ScoreTable {
    fit_adjusted(functional, folds, &[])
}

fn fit_adjusted(
    functional: &OutcomeFunctional,
    folds: &[u32],
    adjustment: &[VariableId],
) -> ScoreTable {
    let (table_data, _) = data();
    CellSaturatedAipw {
        overlap: OverlapPolicy::RequireDiagnostics { clip: None, trim: None },
        ..CellSaturatedAipw::new()
    }
    .fit_scores_with_assignment(
        &table_data,
        &CellScoreSpec {
            treatments: &[VariableId::from_raw(0), VariableId::from_raw(1)],
            outcome: VariableId::from_raw(2),
            adjustment,
            functional,
            continuous: None,
        },
        SharedCellInputs { fold_ids: Some(folds), design: None },
    )
    .unwrap()
}

/// Reference scores for column `(arm, threshold)` from the table's own cells and outcomes.
fn reference_column(table: &ScoreTable, folds: &[u32], arm: u32, threshold: Option<f64>) -> Vec<f64> {
    let response =
        |i: usize| threshold.map_or(table.observed_outcome[i], |c| f64::from(table.observed_outcome[i] > c));
    let mut out = vec![0.0; table.n_rows];
    for fold in 0..FOLDS as u32 {
        let train: Vec<usize> = (0..table.n_rows).filter(|&i| folds[i] != fold).collect();
        let in_cell: Vec<usize> =
            train.iter().copied().filter(|&i| table.observed_arm[i] == arm).collect();
        let propensity = in_cell.len() as f64 / train.len() as f64;
        let mean = in_cell.iter().map(|&i| response(i)).sum::<f64>() / in_cell.len() as f64;
        for i in (0..table.n_rows).filter(|&i| folds[i] == fold) {
            let indicator = f64::from(table.observed_arm[i] == arm);
            out[i] = mean + indicator * (response(i) - mean) / propensity;
        }
    }
    out
}

fn mean_and_variance_of_mean(scores: &[f64]) -> (f64, f64) {
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n * (n - 1.0));
    (mean, variance)
}

#[test]
fn scores_equal_the_closed_form_for_mean_and_exceedance_functionals() {
    let (_, folds) = data();
    for functional in [
        OutcomeFunctional::Mean,
        OutcomeFunctional::Exceedance(OrderedFloatBits::from_f64(1.0)),
    ] {
        let table = fit(&functional, &folds);
        assert_eq!(table.n_columns(), 4);
        assert_eq!(table.fold_ids.as_ref(), folds.as_slice());
        for (j, column) in table.columns.iter().enumerate() {
            let want = reference_column(&table, &folds, column.arm, column.threshold);
            let got = table.column(j).unwrap();
            for i in 0..N {
                assert!(
                    (got[i] - want[i]).abs() < SCORE_TOLERANCE,
                    "{functional:?} column {j} row {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }
}

#[test]
fn cell_minus_control_and_interaction_match_the_scores_and_their_standard_errors() {
    let (_, folds) = data();
    let table = fit(&OutcomeFunctional::Mean, &folds);
    let column = |arm: u32| {
        reference_column(&table, &folds, arm, None)
    };
    let (psi0, psi1, psi2, psi3) = (column(0), column(1), column(2), column(3));
    // Cell codes are an arbitrary labelling of the two treatments; the table says which.
    let (diff, diff_scores) = cell_minus_control_contrast(&table, 3).unwrap();
    let paired: Vec<f64> = psi3.iter().zip(&psi0).map(|(x, y)| x - y).collect();
    let (value, variance) = mean_and_variance_of_mean(&paired);
    assert!((diff.value - value).abs() < SCORE_TOLERANCE, "{} vs {value}", diff.value);
    assert!((diff.se - variance.sqrt()).abs() < SCORE_TOLERANCE, "{} vs {}", diff.se, variance.sqrt());
    for (a, b) in diff_scores.iter().zip(&paired) {
        assert!((a - b).abs() < SCORE_TOLERANCE);
    }
    let interaction = interaction_contrast(&table).unwrap();
    let combined: Vec<f64> = (0..N).map(|i| psi0[i] - psi1[i] - psi2[i] + psi3[i]).collect();
    let (value, variance) = mean_and_variance_of_mean(&combined);
    assert!((interaction.value - value).abs() < SCORE_TOLERANCE, "{} vs {value}", interaction.value);
    assert!((interaction.se - variance.sqrt()).abs() < SCORE_TOLERANCE);
}

/// With one binary adjustment covariate the design `[1, z]` is saturated in `z`, so the
/// multinomial-logit propensity is the training share of each cell within each `z` stratum and
/// the per-cell least-squares outcome model is the training mean within each (cell, `z`)
/// stratum. This exercises the covariate design, the validation-fold prediction and the
/// fold alignment of the propensity table.
#[test]
fn scores_with_a_binary_covariate_equal_the_stratified_closed_form() {
    let (table_data, folds) = data();
    let z: Vec<f64> = table_data.float64_values(VariableId::from_raw(3)).unwrap();
    let table = fit_adjusted(&OutcomeFunctional::Mean, &folds, &[VariableId::from_raw(3)]);
    for (j, column) in table.columns.iter().enumerate() {
        let arm = column.arm;
        let got = table.column(j).unwrap();
        for fold in 0..FOLDS as u32 {
            let train: Vec<usize> = (0..N).filter(|&i| folds[i] != fold).collect();
            for stratum in [0.0, 1.0] {
                let in_stratum: Vec<usize> =
                    train.iter().copied().filter(|&i| z[i] == stratum).collect();
                let in_cell: Vec<usize> = in_stratum
                    .iter()
                    .copied()
                    .filter(|&i| table.observed_arm[i] == arm)
                    .collect();
                let propensity = in_cell.len() as f64 / in_stratum.len() as f64;
                let mean = in_cell.iter().map(|&i| table.observed_outcome[i]).sum::<f64>()
                    / in_cell.len() as f64;
                for i in (0..N).filter(|&i| folds[i] == fold && z[i] == stratum) {
                    let indicator = f64::from(table.observed_arm[i] == arm);
                    let want = mean + indicator * (table.observed_outcome[i] - mean) / propensity;
                    assert!(
                        (got[i] - want).abs() < SCORE_TOLERANCE * (1.0 + want.abs()),
                        "arm {arm} row {i}: {} vs {want}",
                        got[i]
                    );
                }
            }
        }
    }
}
