//! Declared dependence on the augmented inverse-probability estimator, plain and with a
//! penalized propensity, against closed forms.
//!
//! The cross-fitted AIPW effect is the mean of its influence values `psi_i`. With a Newey-West
//! declaration its standard error is the Bartlett-weighted long-run variance of those values,
//! `(g0 + 2 sum_l (1 - l / (lag + 1)) g_l) / n^2` with `g_l = sum_i (psi_i - mean)(psi_{i-l} - mean)`,
//! computed here from the definition on rows whose disturbance drifts smoothly with the row
//! index. The lag-zero long-run variance is the uncorrected sandwich and the declaration changes
//! only the standard error, never the point estimate. A cluster declaration gives the
//! cluster-summed sandwich `g / (g - 1) sum_g (sum_{i in g} (psi_i - mean))^2 / n^2`. Both hold
//! for the ridge-penalized propensity, which reads the same declarations. A treatment with three
//! levels, or an active level the data never took, is refused by both, naming the observed
//! levels, rather than reduced to a pair or extrapolated to.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "test fixtures index small literals; the oracle is written as the formula reads"
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{
    AipwAte, AipwWorkspace, AnalyticSeKind, EffectEstimate, EstimationError, PropensityNuisance,
    RidgeTuning,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};

const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);
const Z: VariableId = VariableId::from_raw(2);

fn dataset(t: &[f64], y: &[f64], z: &[f64]) -> (TabularData, IdentifiedEstimand) {
    let n = t.len();
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
    let columns = [(T, t), (Y, y), (Z, z)]
        .into_iter()
        .map(|(id, values)| {
            OwnedColumn::Float64(
                Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n)).unwrap(),
            )
        })
        .collect();
    let storage =
        OwnedColumnarStorage::try_new(builder.build().unwrap(), columns, None, None).unwrap();
    let estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([Z]), ExprId::from_raw(0));
    (TabularData::new(storage), estimand)
}

/// Rows whose disturbance drifts smoothly with the row index (so neighbours are dependent) and
/// whose treatment depends on the covariate through a bounded logistic-like rule.
fn rows(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut t, mut y, mut z) = (vec![], vec![], vec![]);
    for i in 0..n {
        let zi = ((i as f64) * 0.71).cos() + 0.3 * ((i * i % 7) as f64) - 0.9;
        let ti = f64::from(((i * 5 + (i / 3) * 2) % 7) < 3 + usize::from(zi > 0.0));
        let drift = 0.8 * ((i as f64) * 0.12).sin();
        t.push(ti);
        z.push(zi);
        y.push(1.0 + 2.0 * ti + 0.5 * zi + drift + 0.2 * ((i as f64) * 1.7).sin());
    }
    (t, y, z)
}

/// The plain logistic propensity and the ridge-penalized one.
fn estimators() -> [(&'static str, AipwAte); 2] {
    [
        ("plain", AipwAte::new().with_bootstrap_replicates(0)),
        (
            "ridge",
            AipwAte {
                propensity: PropensityNuisance::ridge_logistic(
                    RidgeTuning::new(&[0.01, 0.1, 1.0], 3).unwrap(),
                ),
                ..AipwAte::new()
            }
            .with_bootstrap_replicates(0),
        ),
    ]
}

fn fit(
    n: usize,
    base: &AipwAte,
    configure: impl Fn(AipwAte) -> AipwAte,
) -> Result<EffectEstimate, EstimationError> {
    let (t, y, z) = rows(n);
    let (data, estimand) = dataset(&t, &y, &z);
    let estimator = configure(base.clone());
    let prepared = estimator.prepare(&data, &estimand, &AverageEffectQuery::binary_ate(T, Y))?;
    estimator.fit(
        &prepared,
        &mut AipwWorkspace::default(),
        &ExecutionContext::for_tests(41),
        AssumptionSet::new(),
    )
}

/// Standard error of the mean of `psi` with Bartlett weights `1 - l / (lag + 1)`.
fn long_run_se(psi: &[f64], lag: usize) -> f64 {
    let n = psi.len();
    let mean = psi.iter().sum::<f64>() / n as f64;
    let centred: Vec<f64> = psi.iter().map(|v| v - mean).collect();
    let mut total = 0.0;
    for l in 0..=lag.min(n - 1) {
        let weight = if l == 0 { 1.0 } else { 1.0 - l as f64 / (lag as f64 + 1.0) };
        let gamma: f64 = (l..n).map(|i| centred[i] * centred[i - l]).sum();
        total += if l == 0 { gamma } else { 2.0 * weight * gamma };
    }
    total.sqrt() / n as f64
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn aipw_newey_west_se_is_the_bartlett_long_run_variance_of_its_influence_values() {
    let n = 240;
    for (name, base) in estimators() {
        let reference = fit(n, &base, |e| e).unwrap();
        let psi = reference.scores.influence.as_ref().expect("influence values").to_vec();
        assert_eq!(psi.len(), n);
        for lag in [0_usize, 1, 3, 8] {
            let hac = fit(n, &base, |e| e.with_se_kind(AnalyticSeKind::NeweyWest { lag })).unwrap();
            close(hac.ate, reference.ate, 1e-12, "the declaration never moves the point estimate");
            close(
                hac.se_analytic(),
                long_run_se(&psi, lag),
                1e-9,
                &format!("{name} newey-west lag {lag}"),
            );
            if lag > 0 {
                // Autocovariance is present: the long-run error differs from the lag-zero one.
                let zero = long_run_se(&psi, 0);
                assert!((hac.se_analytic() - zero).abs() > 1e-6 * zero, "{name} lag {lag}");
            }
        }
        // The lag-zero long-run variance is the plain (uncorrected) sandwich of the influence.
        let mean = psi.iter().sum::<f64>() / n as f64;
        let plain = psi.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>().sqrt() / n as f64;
        let zero = fit(n, &base, |e| e.with_se_kind(AnalyticSeKind::NeweyWest { lag: 0 })).unwrap();
        close(zero.se_analytic(), plain, 1e-9, "lag zero is the uncorrected sandwich");
    }
}

#[test]
fn aipw_cluster_se_is_the_cluster_summed_sandwich_of_its_influence_values() {
    let n = 240;
    let groups: Vec<u32> = (0..n).map(|i| (i / 8) as u32).collect();
    for (name, base) in estimators() {
        let reference = fit(n, &base, |e| e).unwrap();
        let psi = reference.scores.influence.as_ref().expect("influence values").to_vec();
        let clustered = fit(n, &base, |e| {
            e.with_se_kind(AnalyticSeKind::Cluster).with_source_cluster_ids(groups.clone())
        })
        .unwrap();
        close(clustered.ate, reference.ate, 1e-12, "the declaration never moves the estimate");
        let mean = psi.iter().sum::<f64>() / n as f64;
        let mut totals = std::collections::BTreeMap::<u32, f64>::new();
        for (value, g) in psi.iter().zip(&groups) {
            *totals.entry(*g).or_default() += value - mean;
        }
        let g = totals.len() as f64;
        let expected =
            (g / (g - 1.0) * totals.values().map(|v| v * v).sum::<f64>() / (n as f64).powi(2))
                .sqrt();
        close(clustered.se_analytic(), expected, 1e-10, &format!("{name} cluster sandwich"));
    }
}

#[test]
fn a_three_level_treatment_is_refused_naming_its_levels_by_both_propensity_routes() {
    let n = 240;
    let (_, y, z) = rows(n);
    let t: Vec<f64> = (0..n).map(|i| ((i + i / 5) % 3) as f64).collect();
    let (data, estimand) = dataset(&t, &y, &z);
    let per_level = |level: f64| t.iter().filter(|&&v| v == level).count();
    for (name, estimator) in estimators() {
        let error = estimator
            .prepare(&data, &estimand, &AverageEffectQuery::with_levels(T, Y, 0.0, 2.0))
            .expect_err("a three-level treatment must not be reduced to a pair");
        let EstimationError::Refused { code, message } = error else {
            panic!("{name}: expected a coded refusal, got {error:?}");
        };
        assert_eq!(code, "treatment_level_undeclared", "{name}");
        let listing = format!(
            "observed treatment levels: 0 ({} rows), 1 ({} rows), 2 ({} rows)",
            per_level(0.0),
            per_level(1.0),
            per_level(2.0)
        );
        assert!(message.contains(&listing), "{name}: {message}");
    }
}

#[test]
fn an_active_level_the_data_never_took_is_refused_naming_the_observed_levels() {
    let n = 240;
    let (t, y, z) = rows(n);
    let (data, estimand) = dataset(&t, &y, &z);
    let at = |level: f64| t.iter().filter(|&&v| v == level).count();
    for (name, estimator) in estimators() {
        let error = estimator
            .prepare(&data, &estimand, &AverageEffectQuery::with_levels(T, Y, 0.0, 5.0))
            .expect_err("a level outside the observed treatment must not be extrapolated to");
        let EstimationError::Refused { code, message } = error else {
            panic!("{name}: expected a coded refusal, got {error:?}");
        };
        assert_eq!(code, "treatment_level_undeclared", "{name}");
        assert!(message.contains("active 5, control 0"), "{name}: {message}");
        let listing =
            format!("observed treatment levels: 0 ({} rows), 1 ({} rows)", at(0.0), at(1.0));
        assert!(message.contains(&listing), "{name}: {message}");
    }
}
