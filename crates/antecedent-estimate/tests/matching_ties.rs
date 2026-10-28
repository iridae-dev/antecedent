//! Nearest-neighbour matching averages over every donor tied at the nearest distance.
//!
//! A binary confounder `z` makes the fitted propensity score two-valued, so every treated unit
//! at a `z` ties with every control at that `z`: the whole control cell is its nearest
//! neighbour set. Matching to one arbitrary tied donor (the first in row order) makes the effect
//! depend on how the rows happen to be ordered; the matching estimator of Abadie and Imbens
//! averages over all tied donors, which on this saturated table is exactly the stratified
//! contrast. The table has treated outcomes `[3, 5]` / `[10, 12, 14]` and control outcomes
//! `[0, 2, 4, 6]` / `[4, 6]` at `z = 0` / `z = 1`, so the cell contrasts are 1 and 7:
//!
//! * ATT = (2 * 1 + 3 * 7) / 5 = 23 / 5,
//! * ATC = (4 * 1 + 2 * 7) / 6 = 3,
//! * ATE = (5 * 23 / 5 + 6 * 3) / 11 = 41 / 11.
//!
//! The standard error is the Abadie-Imbens form `sigma^2 sum_j (1 + u_j)^2 / n^2` with the
//! fractional donor usage `u_j`, the weight of the queries unit `j` donates to (each query
//! shares its weight equally over its tied donors), and `sigma^2` the variance of the matched
//! differences divided by the mean of `1 + 1 / (donors of the query)`. The estimate does not
//! move when the rows are shuffled.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "closed forms are compared at tight tolerances; counts are small literals"
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, TargetPopulation, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{EffectEstimate, PropensityEstimationWorkspace, PropensityMatching};
use antecedent_expr::{ExprId, IdentifiedEstimand};

const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);
const Z: VariableId = VariableId::from_raw(2);

/// `(z, treated, outcome)` for every unit of the saturated table.
fn rows() -> Vec<(f64, f64, f64)> {
    let cell = |z: f64, t: f64, ys: &[f64]| ys.iter().map(|&y| (z, t, y)).collect::<Vec<_>>();
    [
        cell(0.0, 1.0, &[3.0, 5.0]),
        cell(0.0, 0.0, &[0.0, 2.0, 4.0, 6.0]),
        cell(1.0, 1.0, &[10.0, 12.0, 14.0]),
        cell(1.0, 0.0, &[4.0, 6.0]),
    ]
    .concat()
}

fn table(rows: &[(f64, f64, f64)]) -> (TabularData, IdentifiedEstimand) {
    let n = rows.len();
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
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
    let column = |id: VariableId, pick: &dyn Fn(&(f64, f64, f64)) -> f64| {
        OwnedColumn::Float64(
            Float64Column::new(
                id,
                Arc::from(rows.iter().map(pick).collect::<Vec<_>>()),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        )
    };
    let columns =
        vec![column(T, &|r| r.1), column(Y, &|r| r.2), column(Z, &|r| r.0)];
    let storage =
        OwnedColumnarStorage::try_new(builder.build().unwrap(), columns, None, None).unwrap();
    let estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([Z]), ExprId::from_raw(0));
    (TabularData::new(storage), estimand)
}

fn fit(rows: &[(f64, f64, f64)], target: TargetPopulation) -> EffectEstimate {
    let (data, estimand) = table(rows);
    let query = AverageEffectQuery::binary_ate(T, Y).with_target_population(target);
    let estimator = PropensityMatching::new().with_bootstrap_replicates(0);
    let problem = estimator.prepare(&data, &estimand, &query).unwrap();
    estimator
        .fit(
            &problem,
            &mut PropensityEstimationWorkspace::default(),
            &ExecutionContext::for_tests(3),
            AssumptionSet::new(),
        )
        .unwrap()
}

#[test]
fn a_two_valued_propensity_matches_every_tied_control_and_recovers_the_stratified_contrasts() {
    let table = rows();
    for (target, want) in [
        (TargetPopulation::Treated, 23.0 / 5.0),
        (TargetPopulation::Untreated, 3.0),
        (TargetPopulation::AllObserved, 41.0 / 11.0),
    ] {
        let got = fit(&table, target.clone()).ate;
        assert!((got - want).abs() < 1e-9, "{target:?}: {got} vs {want}");
    }
}

#[test]
fn the_estimate_does_not_depend_on_the_order_of_the_rows() {
    let table = rows();
    let reversed: Vec<_> = table.iter().rev().copied().collect();
    let interleaved: Vec<_> =
        (0..table.len()).map(|i| table[(i * 7) % table.len()]).collect();
    for target in [TargetPopulation::Treated, TargetPopulation::Untreated, TargetPopulation::AllObserved]
    {
        let base = fit(&table, target.clone());
        for other in [&reversed, &interleaved] {
            let again = fit(other, target.clone());
            assert!((again.ate - base.ate).abs() < 1e-12, "{target:?}: {} vs {}", again.ate, base.ate);
            assert!(
                (again.se_analytic() - base.se_analytic()).abs() < 1e-12,
                "{target:?}: se {} vs {}",
                again.se_analytic(),
                base.se_analytic()
            );
        }
    }
}

#[test]
fn the_standard_error_is_the_abadie_imbens_form_with_fractional_donor_usage() {
    // Matched differences of the 11 units (treated: y - control cell mean; controls, sign
    // flipped: treated cell mean - y), the donor-set size of each, and the usage of each unit.
    let effects = [0.0, 2.0, 5.0, 7.0, 9.0, 4.0, 2.0, 0.0, -2.0, 8.0, 6.0];
    let set_size = [4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0];
    // Treated at z = 0 (2 units) are each used by the 4 controls there with weight 1/2;
    // treated at z = 1 (3 units) by the 2 controls there with weight 1/3; controls at z = 0
    // (4 units) by the 2 treated there with weight 1/4; controls at z = 1 (2 units) by the 3
    // treated there with weight 1/2.
    let usage = [2.0, 2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5];
    let n = effects.len() as f64;
    let mean = effects.iter().sum::<f64>() / n;
    assert!((mean - 41.0 / 11.0).abs() < 1e-12);
    let variance = effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let noise = 1.0 / (set_size.iter().map(|s| 1.0 + 1.0 / s).sum::<f64>() / n);
    let design: f64 = usage.iter().map(|u| (1.0_f64 + u).powi(2)).sum();
    let want = (variance * noise * design / (n * n)).sqrt();
    let got = fit(&rows(), TargetPopulation::AllObserved).se_analytic();
    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
}
