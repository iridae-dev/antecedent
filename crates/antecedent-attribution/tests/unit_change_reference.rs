//! Per-unit change attribution against closed forms: an additive (linear) game, and a game with
//! interactions whose Shapley values are enumerated over every permutation of the parents.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_attribution::unit_change;
use antecedent_core::{
    AllocationMethod, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    ShapleyConfig, SmallRoleSet, UnitChangeQuery, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    BasisTerm, CompiledCausalModel, CompiledMechanismStore, MechanismSlot, ParentBasis,
};

/// A table of `parents.len()` roots and one outcome, with edges from every root to the outcome.
fn table(parents: &[Vec<f64>], y: Vec<f64>) -> (TabularData, CompiledCausalModel) {
    let n = y.len();
    let mut b = CausalSchemaBuilder::new();
    for k in 0..parents.len() {
        b.add_variable(
            format!("x{k}"),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let validity = ValidityBitmap::all_valid(n);
    let cols = parents
        .iter()
        .cloned()
        .chain(std::iter::once(y))
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
        .collect::<Vec<_>>();
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(parents.len() as u32 + 1);
    for k in 0..parents.len() as u32 {
        g.insert_directed(DenseNodeId::from_raw(k), DenseNodeId::from_raw(parents.len() as u32))
            .unwrap();
    }
    (data, CompiledCausalModel::compile(g).unwrap())
}

fn roots(k: usize) -> Vec<MechanismSlot> {
    (0..k)
        .map(|_| MechanismSlot::LinearGaussian {
            intercept: 0.0,
            coeffs: Arc::from([]),
            sigma: 1.0,
        })
        .collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// With a linear outcome mechanism the unit game is additive, so the Shapley value of
/// parent `j` for a unit is exactly `beta_j (x_j - mean_j)` (the intercept and the abducted
/// noise are common to every coalition and cancel). That holds for the exact value and for
/// any permutation sample, whose standard error is therefore zero.
#[test]
fn linear_mechanism_unit_contributions_are_slope_times_deviation_from_the_mean() {
    let n = 12usize;
    let x1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin() * 3.0 + 10.0).collect();
    let x2: Vec<f64> = (0..n).map(|i| (i as f64 * 1.3).cos() * 2.0 - 4.0).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x1[i] - 0.5 * x2[i] + 0.1 * i as f64).collect();
    let (data, compiled) = table(&[x1.clone(), x2.clone()], y);
    let mut slots = roots(2);
    slots.push(MechanismSlot::LinearGaussian {
        intercept: 1.0,
        coeffs: Arc::from([2.0, -0.5]),
        sigma: 0.3,
    });
    let model = compiled.with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) });
    let (m1, m2) = (mean(&x1), mean(&x2));
    for approximation in [ShapleyConfig::exact(), ShapleyConfig::monte_carlo(7).with_seed(5)] {
        let q = UnitChangeQuery::new(VariableId::from_raw(2), 100)
            .with_allocation(AllocationMethod::Shapley { approximation });
        let result = unit_change(&model, &data, &q, &ExecutionContext::for_tests(2)).unwrap();
        assert_eq!(result.components.len(), 2);
        assert_eq!(result.contributions.len(), 2 * n);
        for i in 0..n {
            let want = [2.0 * (x1[i] - m1), -0.5 * (x2[i] - m2)];
            for j in 0..2 {
                let got = result.contributions[i * 2 + j];
                assert!(
                    (got - want[j]).abs() <= 1e-12 * (1.0 + want[j].abs()),
                    "unit {i} parent {j}: {got} vs {}",
                    want[j]
                );
            }
        }
        assert!(result.mean_contributions.iter().all(|m| m.abs() < 1e-12));
        if let Some(se) = &result.component_mc_stderr {
            assert!(se.iter().all(|s| *s <= 1e-12), "additive game has no permutation variance");
        }
    }
}

/// An outcome with a product of two parents and a separate third parent: the game has
/// interactions, so the value of each parent depends on which others are already in. The
/// reference averages the marginal contribution over all six orders of the three parents,
/// evaluating the mean function directly.
#[test]
fn interacting_parents_receive_the_permutation_average_marginal_contribution() {
    let n = 9usize;
    let x: Vec<Vec<f64>> = (0..3)
        .map(|k| (0..n).map(|i| ((i * (k + 2)) as f64 * 0.61 + k as f64).sin() * 2.0 + k as f64).collect())
        .collect();
    let (centers, scales) = ([1.0_f64, -0.5, 0.25], [2.0_f64, 1.5, 0.5]);
    // z0, z1, z2, z0 * z1, z1 * z2 on standardized parents
    let terms = vec![
        BasisTerm::Power { parent: 0, degree: 1 },
        BasisTerm::Power { parent: 1, degree: 1 },
        BasisTerm::Power { parent: 2, degree: 1 },
        BasisTerm::Product { left: 0, right: 1 },
        BasisTerm::Product { left: 1, right: 2 },
    ];
    let coeffs = [0.8_f64, -1.1, 0.4, 1.7, -2.3];
    let intercept = 0.6;
    let basis = ParentBasis::new(
        3,
        Arc::from(centers),
        Arc::from(scales),
        Arc::from(vec![Arc::<[f64]>::from([]); 3]),
        Arc::from(terms),
    )
    .unwrap();
    let f = |p: [f64; 3]| {
        let z: Vec<f64> = (0..3).map(|k| (p[k] - centers[k]) / scales[k]).collect();
        intercept
            + coeffs[0] * z[0]
            + coeffs[1] * z[1]
            + coeffs[2] * z[2]
            + coeffs[3] * z[0] * z[1]
            + coeffs[4] * z[1] * z[2]
    };
    let y: Vec<f64> = (0..n).map(|i| f([x[0][i], x[1][i], x[2][i]]) + 0.05 * i as f64).collect();
    let (data, compiled) = table(&x, y);
    let mut slots = roots(3);
    slots.push(MechanismSlot::LinearBasis {
        intercept,
        basis,
        coeffs: Arc::from(coeffs),
        sigma: 0.2,
    });
    let model = compiled.with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) });
    let reference = [mean(&x[0]), mean(&x[1]), mean(&x[2])];
    let orders: [[usize; 3]; 6] =
        [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let q = UnitChangeQuery::new(VariableId::from_raw(3), 100)
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
    let result = unit_change(&model, &data, &q, &ExecutionContext::for_tests(2)).unwrap();
    assert_eq!(result.contributions.len(), 3 * n);
    let mut interacting = 0.0_f64;
    for i in 0..n {
        let factual = [x[0][i], x[1][i], x[2][i]];
        let mut want = [0.0_f64; 3];
        for order in orders {
            let mut point = reference;
            let mut previous = f(point);
            for &j in &order {
                point[j] = factual[j];
                let value = f(point);
                want[j] += (value - previous) / 6.0;
                previous = value;
            }
        }
        // The game is not additive: parent 0 alone moves the outcome by a different amount
        // than it contributes on average.
        let mut alone = reference;
        alone[0] = factual[0];
        interacting = interacting.max((f(alone) - f(reference) - want[0]).abs());
        let total: f64 = (0..3).map(|j| result.contributions[i * 3 + j]).sum();
        assert!((total - (f(factual) - f(reference))).abs() <= 1e-11, "efficiency, unit {i}");
        for j in 0..3 {
            let got = result.contributions[i * 3 + j];
            assert!((got - want[j]).abs() <= 1e-11 * (1.0 + want[j].abs()), "unit {i} parent {j}: {got} vs {}", want[j]);
        }
    }
    assert!(interacting > 0.05, "fixture must have interactions, saw {interacting}");
}
