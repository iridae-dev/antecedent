//! Lagged imputation of a time series: the conditional model reads the neighbouring steps, the
//! parameter draw resamples contiguous blocks, and the assumptions record binds the lagged
//! neighbourhood.
//!
//! The references are computed in this file from the generating process.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "deterministic fixtures: observed cells are compared bit for bit"
)]

use antecedent_core::{CausalRng, DeclaredSamplingDesign, ExecutionContext};
use antecedent_data::{ImputationModel, ImputationSpec};
use antecedent_estimate::multiple_imputation::{
    self, ImputationAssumptions, ImputationColumn, ImputationUnits, Imputer,
};
use antecedent_kernels::standard_normal;

/// Correlation between a series and itself one step later.
fn lag_one_correlation(values: &[f64]) -> f64 {
    let n = values.len() - 1;
    let (a, b) = (&values[..n], &values[1..]);
    let (ma, mb) = (a.iter().sum::<f64>() / n as f64, b.iter().sum::<f64>() / n as f64);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

/// An AR(1) series with coefficient 0.8 and unit variance, an uninformative companion, and
/// `share` of the first series' cells missing completely at random.
fn autoregressive(n: usize, share: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = CausalRng::from_seed(seed);
    let mut x = Vec::with_capacity(n);
    let mut previous = standard_normal(&mut rng);
    for _ in 0..n {
        previous = 0.8 * previous + 0.6 * standard_normal(&mut rng);
        x.push(previous);
    }
    let noise: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let observed =
        x.iter().map(|v| if rng.next_f64() < share { f64::NAN } else { *v }).collect::<Vec<f64>>();
    (observed, noise)
}

fn imputer(x: &[f64], noise: &[f64], spec: ImputationSpec, units: ImputationUnits) -> Imputer {
    Imputer::new(
        vec![
            ImputationColumn::new("x", x.to_vec()),
            ImputationColumn::new("noise", noise.to_vec()),
        ],
        spec,
        units,
    )
    .unwrap()
}

fn mean_lag_one(imputer: &Imputer, tables: u32) -> f64 {
    let ctx = ExecutionContext::for_tests(3);
    (0..tables)
        .map(|j| lag_one_correlation(&imputer.impute(j, &ctx).unwrap().columns()[0]))
        .sum::<f64>()
        / f64::from(tables)
}

#[test]
fn lagged_imputation_keeps_the_autocorrelation_that_exchangeable_rows_destroy() {
    let n = 600;
    let (x, noise) = autoregressive(n, 0.4, 17);
    let spec = ImputationSpec::new().with_imputations(6).with_iterations(8).with_seed(4);
    let rows = imputer(&x, &noise, spec, ImputationUnits::Rows);
    let units = multiple_imputation::bind_series_units(None, n, 9).unwrap();
    let lagged = imputer(&x, &noise, spec, units).with_dynamics(1).unwrap();
    let exchangeable = mean_lag_one(&rows, 6);
    let dynamic = mean_lag_one(&lagged, 6);
    // The true lag-one correlation is 0.8. Drawing each missing cell from its row alone (the
    // companion carries nothing) leaves the filled cells independent of their neighbours.
    assert!(dynamic > 0.6, "lagged imputation autocorrelation {dynamic}");
    assert!(exchangeable < dynamic - 0.1, "rows {exchangeable} against lagged {dynamic}");
    assert_eq!(lagged.lags(), 1);
    assert_eq!(rows.lags(), 0);
}

#[test]
fn a_gap_in_a_smooth_series_is_filled_from_its_neighbours() {
    let n = 300;
    let mut rng = CausalRng::from_seed(9);
    let truth: Vec<f64> = (0..n).map(|t| (0.1 * t as f64).sin()).collect();
    let noise: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let mut observed = truth.clone();
    let gaps = [60_usize, 150, 151, 240];
    for &t in &gaps {
        observed[t] = f64::NAN;
    }
    let spec = ImputationSpec::new().with_imputations(4).with_iterations(10).with_seed(2);
    let units = multiple_imputation::bind_series_units(None, n, 8).unwrap();
    let lagged = imputer(&observed, &noise, spec, units).with_dynamics(1).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    for j in 0..4 {
        let table = lagged.impute(j, &ctx).unwrap();
        let column = &table.columns()[0];
        for &t in &gaps {
            assert!(
                (column[t] - truth[t]).abs() < 0.15,
                "imputation {j} at step {t}: {} against {}",
                column[t],
                truth[t]
            );
        }
        // Observed cells are never altered.
        assert_eq!(column[10], truth[10]);
    }
    // Drawn from the row alone, the same gaps are filled with the series' marginal law.
    let rows = imputer(&observed, &noise, spec, ImputationUnits::Rows);
    let worst = (0..4)
        .map(|j| {
            let table = rows.impute(j, &ctx).unwrap();
            gaps.iter().map(|&t| (table.columns()[0][t] - truth[t]).abs()).fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    assert!(
        worst > 0.15,
        "row imputation reproduced the gaps ({worst}) without reading neighbours"
    );
}

#[test]
fn a_lagged_imputation_is_deterministic_and_its_assumptions_bind_the_neighbourhood() {
    let n = 200;
    let (x, noise) = autoregressive(n, 0.2, 5);
    let spec = ImputationSpec::new().with_imputations(3).with_iterations(4).with_seed(8);
    let build = |lags: usize| {
        let units = multiple_imputation::bind_series_units(None, n, 7).unwrap();
        let base = imputer(&x, &noise, spec, units);
        if lags == 0 { base } else { base.with_dynamics(lags).unwrap() }
    };
    let ctx = ExecutionContext::for_tests(1);
    let (a, b) = (build(2).impute(1, &ctx).unwrap(), build(2).impute(1, &ctx).unwrap());
    assert_eq!(a.columns()[0].to_vec(), b.columns()[0].to_vec(), "same draw, same table");
    let other = build(2).impute(2, &ctx).unwrap();
    assert_ne!(a.columns()[0].to_vec(), other.columns()[0].to_vec(), "tables differ");

    let digest = |lags: usize| ImputationAssumptions::of(&build(lags), None).digest();
    assert_eq!(digest(2), digest(2), "replay");
    assert_ne!(digest(0), digest(1), "the neighbourhood binds");
    assert_ne!(digest(1), digest(2));
    let record = ImputationAssumptions::of(&build(2), None);
    assert_eq!((record.lags, record.unit, record.units), (2, "group", n.div_ceil(7)));
}

#[test]
fn dynamic_imputation_refuses_what_it_cannot_identify() {
    let n = 200;
    let (x, noise) = autoregressive(n, 0.2, 5);
    let units = || multiple_imputation::bind_series_units(None, n, 7).unwrap();
    let chained = ImputationSpec::new();
    let zero = imputer(&x, &noise, chained, units()).with_dynamics(0).unwrap_err();
    assert_eq!(zero.detail(), "multiple_imputation.dynamics_lags");
    let joint = ImputationSpec::new().with_model(ImputationModel::JointNormal);
    let refused = imputer(&x, &noise, joint, units()).with_dynamics(1).unwrap_err();
    assert_eq!(refused.detail(), "multiple_imputation.dynamics_joint_normal");
    assert_eq!(refused.next_attempt().map(|next| next.name()), Some("chained_equations"));

    // Twenty steps cannot identify a model that reads four neighbours on each side.
    let (short_x, short_noise) = autoregressive(20, 0.2, 6);
    let short_units = multiple_imputation::bind_series_units(None, 20, 5).unwrap();
    let short = imputer(&short_x, &short_noise, chained, short_units).with_dynamics(4);
    assert_eq!(short.unwrap_err().detail(), "multiple_imputation.not_identified");
}

#[test]
fn a_series_binds_contiguous_blocks_under_the_dependent_time_series_design_only() {
    let units = multiple_imputation::bind_series_units(None, 10, 4).unwrap();
    let ImputationUnits::Groups(groups) = &units else { panic!("blocks are groups") };
    assert_eq!(&groups[..], &[0_u32, 0, 0, 0, 1, 1, 1, 1, 2, 2][..]);
    assert_eq!(units.count(10), 3);
    assert!(
        multiple_imputation::bind_series_units(
            Some(DeclaredSamplingDesign::DependentTimeSeries),
            10,
            4
        )
        .is_ok()
    );
    for design in [
        DeclaredSamplingDesign::ClusteredPanel,
        DeclaredSamplingDesign::IndependentSubjectTrajectories,
    ] {
        let error = multiple_imputation::bind_series_units(Some(design), 10, 4).unwrap_err();
        assert_eq!(error.detail(), "multiple_imputation.sampling_design", "{design:?}");
    }
}
