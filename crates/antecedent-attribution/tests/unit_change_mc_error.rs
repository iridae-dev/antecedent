//! The permutation-sampling standard error of a per-unit change attribution against the spread of
//! the estimate it describes.
//!
//! The pooled per-component standard error of `unit_change` is the error of the mean contribution
//! over the units about the Shapley value of the unit games. Each unit draws its own
//! permutations, so the unit estimates are independent and that error is the square root of the
//! summed per-unit variances over the number of units. The fitted model and the data are held
//! fixed; only the permutation seed varies. The exact Shapley value of a three-parent game with
//! interactions is the truth, enumerated over every order by `ShapleyConfig::exact`.
//!
//! * The mean over seeds of the published standard error equals the standard deviation of the
//!   estimate across seeds. The spread of a standard deviation estimated from `SEEDS` independent
//!   draws has relative standard deviation `1 / sqrt(2 (SEEDS - 1))` (the chi-square spread of a
//!   sample variance), so the ratio is held to `1 +- Z / sqrt(2 (SEEDS - 1))`.
//! * The interval `estimate +- 1.96 se` holds the exact value in at least the nominal share of the
//!   seeds, less `Z` binomial standard deviations of that share.
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
#[allow(clippy::cast_possible_truncation, reason = "the fixture uses bounded graph dimensions")]
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


/// Independent permutation seeds.
const SEEDS: u64 = 200;
/// Permutations drawn per unit.
const PERMUTATIONS: usize = 48;
/// Half-width of every acceptance band, in standard deviations of the statistic it bounds.
const Z: f64 = 4.0;

/// A three-parent outcome with interactions (a product of the first two parents and of the
/// last two), fitted mechanisms fixed, over nine units.
#[allow(clippy::needless_range_loop, reason = "the fixture follows the matrix indexing")]
fn interacting_game() -> (CompiledCausalModel, TabularData) {
    let n = 9usize;
    let x: Vec<Vec<f64>> = (0..3)
        .map(|k| {
            (0..n)
                .map(|i| ((i * (k + 2)) as f64 * 0.61 + k as f64).sin() * 2.0 + k as f64)
                .collect()
        })
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
    (model, data)
}

fn query(approximation: ShapleyConfig) -> UnitChangeQuery {
    UnitChangeQuery::new(VariableId::from_raw(3), 100)
        .with_allocation(AllocationMethod::Shapley { approximation })
}

/// Mean contributions and published standard errors over every seed, and the exact values.
struct Replications {
    exact: Vec<f64>,
    estimates: Vec<Vec<f64>>,
    standard_errors: Vec<Vec<f64>>,
}

fn replicate() -> Replications {
    let (model, data) = interacting_game();
    let ctx = ExecutionContext::for_tests(2);
    let exact = unit_change(&model, &data, &query(ShapleyConfig::exact()), &ctx).unwrap();
    assert!(exact.component_mc_stderr.is_none(), "the exact value publishes no sampling error");
    let (mut estimates, mut standard_errors) = (Vec::new(), Vec::new());
    for seed in 0..SEEDS {
        let config = ShapleyConfig::monte_carlo(PERMUTATIONS).with_seed(1_000 + seed);
        let result = unit_change(&model, &data, &query(config), &ctx).unwrap();
        estimates.push(result.mean_contributions.to_vec());
        standard_errors.push(result.component_mc_stderr.as_ref().unwrap().to_vec());
    }
    Replications { exact: exact.mean_contributions.to_vec(), estimates, standard_errors }
}

fn sample_sd(values: &[f64]) -> f64 {
    let m = mean(values);
    (values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (values.len() as f64 - 1.0)).sqrt()
}

#[test]
fn the_published_permutation_error_matches_the_spread_of_the_estimate_across_seeds() {
    let run = replicate();
    let band = Z / (2.0 * (SEEDS as f64 - 1.0)).sqrt();
    let mut spread_seen = 0.0_f64;
    for j in 0..run.exact.len() {
        let estimates: Vec<f64> = run.estimates.iter().map(|e| e[j]).collect();
        let published = mean(&run.standard_errors.iter().map(|s| s[j]).collect::<Vec<_>>());
        let spread = sample_sd(&estimates);
        spread_seen = spread_seen.max(spread);
        let ratio = published / spread;
        assert!(
            (ratio - 1.0).abs() <= band,
            "component {j}: mean published error {published} against spread {spread} (ratio {ratio}, band {band})"
        );
    }
    assert!(spread_seen > 1e-3, "the permutation sample must actually vary, saw {spread_seen}");
}

#[test]
fn the_permutation_interval_holds_the_exact_shapley_value_at_its_nominal_share() {
    let run = replicate();
    let nominal = 0.95_f64;
    let floor = nominal - Z * (nominal * (1.0 - nominal) / SEEDS as f64).sqrt();
    for j in 0..run.exact.len() {
        let held = run
            .estimates
            .iter()
            .zip(&run.standard_errors)
            .filter(|(e, s)| (e[j] - run.exact[j]).abs() <= 1.959_963_984_540_054 * s[j])
            .count();
        let share = held as f64 / SEEDS as f64;
        assert!(share >= floor, "component {j}: the interval held the exact value in {share}, below {floor}");
    }
}
