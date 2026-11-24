//! The population do-contrast against its analytic value on a linear-Gaussian structural model.
//!
//! On `x -> y -> z` with the chord `x -> z`, every mechanism linear with additive noise, the
//! interventional mean of a child is affine in the parent's level, so
//! `E[child | do(parent = m + delta/2)] - E[child | do(parent = m - delta/2)]` is the total
//! effect of the parent on the child times `delta`: `(b_xz + b_yz * b_xy) * delta` for `x` on
//! `z`, `b_yz * delta` for `y` on `z` (the intervention cuts `x -> y`), and exactly zero for a
//! child that is not a descendant. The two arms replay the same noise, so every paired
//! difference is that constant and its standard error vanishes. The values are computed here
//! from the coefficients, not from the engine.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_attribution::population_do_contrast;
use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{CompiledCausalModel, CompiledMechanismStore, MechanismSlot};

const B_XY: f64 = 1.7;
const B_XZ: f64 = -0.6;
const B_YZ: f64 = 2.3;
const ROWS: usize = 400;

#[allow(clippy::cast_possible_truncation, reason = "three variables")]
fn data() -> TabularData {
    let mut b = CausalSchemaBuilder::new();
    for name in ["x", "y", "z"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    #[allow(clippy::cast_precision_loss, reason = "small row indices")]
    let x: Vec<f64> = (0..ROWS).map(|i| (i % 23) as f64 * 0.1 - 1.1).collect();
    let y: Vec<f64> = x.iter().map(|x| 0.4 + B_XY * x).collect();
    let z: Vec<f64> = x.iter().zip(&y).map(|(x, y)| -1.0 + B_XZ * x + B_YZ * y).collect();
    let validity = ValidityBitmap::all_valid(ROWS);
    let cols = [x, y, z]
        .into_iter()
        .enumerate()
        .map(|(k, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(k as u32),
                    Arc::from(values),
                    validity.clone(),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
}

/// The fitted-model stand-in: the true linear mechanisms, coefficients in the parent order
/// the compiled gather reads.
fn model() -> CompiledCausalModel {
    let mut g = Dag::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let compiled = CompiledCausalModel::compile(g).unwrap();
    let coefficient = |parent: usize, child: u32| match (parent, child) {
        (0, 1) => B_XY,
        (0, 2) => B_XZ,
        (1, 2) => B_YZ,
        other => panic!("no edge {other:?}"),
    };
    let slot = |child: u32, intercept: f64, sigma: f64| {
        let gather = compiled.gather_for(DenseNodeId::from_raw(child)).unwrap();
        let coeffs: Vec<f64> =
            gather.parents.iter().map(|p| coefficient(p.as_usize(), child)).collect();
        MechanismSlot::LinearGaussian { intercept, coeffs: Arc::from(coeffs), sigma }
    };
    let store = CompiledMechanismStore {
        slots: Arc::from([slot(0, 0.0, 1.0), slot(1, 0.4, 0.8), slot(2, -1.0, 1.2)]),
    };
    compiled.with_mechanisms(store)
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-9 * (1.0 + want.abs()), "{what}: {got} vs {want}");
}

#[test]
fn the_contrast_is_the_total_effect_times_the_width_with_no_monte_carlo_error() {
    let (model, data) = (model(), data());
    let ctx = ExecutionContext::for_tests(17);
    let v = VariableId::from_raw;
    for (parent, child, delta, want) in [
        (0, 2, 0.5, (B_XZ + B_YZ * B_XY) * 0.5),
        (0, 2, -2.0, (B_XZ + B_YZ * B_XY) * -2.0),
        (1, 2, 1.0, B_YZ),
        (0, 1, 0.25, B_XY * 0.25),
        (2, 0, 1.0, 0.0),
        (1, 0, 3.0, 0.0),
    ] {
        let what = format!("do({parent}) on {child} with width {delta}");
        let got =
            population_do_contrast(&model, &data, v(parent), v(child), delta, ROWS, &ctx).unwrap();
        close(got.contrast, want, &what);
        assert!(got.stderr.abs() <= 1e-9, "{what}: paired differences are constant");
    }
}

#[test]
fn a_table_over_the_row_cap_is_refused_not_truncated() {
    let (model, data) = (model(), data());
    let ctx = ExecutionContext::for_tests(17);
    let v = VariableId::from_raw;
    assert!(population_do_contrast(&model, &data, v(0), v(2), 1.0, ROWS - 1, &ctx).is_err());
}
