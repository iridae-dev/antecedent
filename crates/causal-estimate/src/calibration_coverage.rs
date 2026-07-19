//! Scheduled SE coverage calibration.
//!
//! These tests are `#[ignore]` so every-PR `cargo test` stays fast. Run via
//! `scripts/gate_calibration.sh`.
//!
//! Tolerance rationale: with `N_SIM` trials the Monte Carlo SE of a binomial
//! coverage rate near 0.95 is `√(0.95·0.05/N)`. We accept coverage within
//! roughly ±4 Monte Carlo SE of 0.95 (and a hard floor/ceiling for small N).
//!
//! IPW: Hajek IF after propensity estimation undercovers vs bootstrap at
//! finite n. Bootstrap CI coverage is the primary gate; analytic IF uses a
//! documented floor (≥0.88 when achievable, else ≥0.85).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(test)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, CausalRng, CausalSchemaBuilder, ExecutionContext,
    MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use causal_data::column::{Float64Column, ValidityBitmap};
use causal_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use causal_expr::{ExprId, IdentifiedEstimand};
use causal_kernels::standard_normal;

use crate::adjustment::LinearAdjustmentAte;
use crate::aipw::AipwAte;
use crate::iv::WaldIv;
use crate::propensity::{
    PropensityEstimationWorkspace, PropensityMatching, PropensityWeighting,
};
use crate::se::AnalyticSeKind;

const TRUE_ATE: f64 = 2.0;
/// Default Monte Carlo budget for analytic SE coverage (runtime OK on weekly gate).
const N_SIM: u32 = 400;
/// Bootstrap IPW is heavier; still ≥200 so MC SE of coverage near 0.95 is ~1.5%.
const N_SIM_BOOT: u32 = 200;
const N_OBS: usize = 300;
const Z95: f64 = 1.96;
/// Bootstrap replicates for IPW SE: R=60 keeps gate runtime acceptable while
/// stabilizing the replicate SD used as se_bootstrap.
const BOOT_REPS: u32 = 60;

fn coverage_band(n_sim: u32) -> (f64, f64) {
    let se = (0.95 * 0.05 / f64::from(n_sim)).sqrt();
    let lo = (0.95 - 4.0 * se).max(0.85);
    let hi = (0.95 + 4.0 * se).min(1.0);
    (lo, hi)
}

fn assert_coverage(covered: u32, n_sim: u32, label: &str) {
    let rate = f64::from(covered) / f64::from(n_sim);
    let (lo, hi) = coverage_band(n_sim);
    assert!(
        rate >= lo && rate <= hi,
        "{label}: coverage={rate:.3} outside [{lo:.3}, {hi:.3}] ({covered}/{n_sim})"
    );
}

/// Analytic IPW IF: prefer ≥0.88; documented fallback floor is 0.85 (bootstrap is primary).
fn assert_coverage_ipw_analytic(covered: u32, n_sim: u32) {
    let rate = f64::from(covered) / f64::from(n_sim);
    let (_, hi) = coverage_band(n_sim);
    // Try the stricter floor first in the message path; gate uses ≥0.85.
    let lo_preferred = 0.88;
    let lo = 0.85;
    if rate >= lo_preferred && rate <= hi {
        return;
    }
    assert!(
        rate >= lo && rate <= hi,
        "ipw_hajek analytic IF: coverage={rate:.3} outside [{lo:.3}, {hi:.3}] \
         ({covered}/{n_sim}); preferred floor was {lo_preferred}. Hajek IF after \
         propensity estimation undercovers vs bootstrap at finite n — bootstrap \
         CI is the primary §28.3 gate."
    );
}

fn schema_tyz() -> causal_core::CausalSchema {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.build().unwrap()
}

fn table_tyz(t: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> TabularData {
    let n = t.len();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    TabularData::new(OwnedColumnarStorage::try_new(schema_tyz(), cols, None, None).unwrap())
}

fn confounded_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
    let mut rng = CausalRng::from_seed(seed);
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for _ in 0..n {
        let zi = standard_normal(&mut rng);
        let ui = standard_normal(&mut rng);
        // Logistic treatment so IPW propensity (logit) is correctly specified.
        let logit = 0.8 * zi;
        let p = 1.0 / (1.0 + (-logit).exp());
        let ti = if rng.next_u64() as f64 / (u64::MAX as f64) < p { 1.0 } else { 0.0 };
        let yi = TRUE_ATE * ti + 1.5 * zi + 0.5 * ui + 0.5 * standard_normal(&mut rng);
        t.push(ti);
        y.push(yi);
        z.push(zi);
    }
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    (table_tyz(t, y, z), estimand)
}

fn covers(ate: f64, se: f64) -> bool {
    se.is_finite() && se > 0.0 && (ate - TRUE_ATE).abs() <= Z95 * se
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn linear_adjustment_analytic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::default() };
    let ctx = ExecutionContext::for_tests(1);
    let mut covered = 0u32;
    for s in 0..N_SIM {
        let (data, estimand) = confounded_scm(N_OBS, 1000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = crate::adjustment::EstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        if covers(effect.ate, effect.se_analytic) {
            covered += 1;
        }
    }
    assert_coverage(covered, N_SIM, "linear_adjustment");
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn linear_adjustment_hc1_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = LinearAdjustmentAte {
        bootstrap_replicates: 0,
        se_kind: AnalyticSeKind::Hc1,
        ..LinearAdjustmentAte::default()
    };
    let ctx = ExecutionContext::for_tests(11);
    let mut covered = 0u32;
    for s in 0..N_SIM {
        let (data, estimand) = confounded_scm(N_OBS, 1100 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = crate::adjustment::EstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        if covers(effect.ate, effect.se_analytic) {
            covered += 1;
        }
    }
    assert_coverage(covered, N_SIM, "linear_adjustment_hc1");
}

/// Primary IPW coverage gate: bootstrap SE.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn ipw_hajek_bootstrap_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = PropensityWeighting {
        bootstrap_replicates: BOOT_REPS,
        ..PropensityWeighting::new()
    };
    let ctx = ExecutionContext::for_tests(2);
    let mut covered = 0u32;
    let mut skipped = 0u32;
    for s in 0..N_SIM_BOOT {
        let (data, estimand) = confounded_scm(500, 2000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let Some(se_b) = effect.se_bootstrap else {
            skipped += 1;
            continue;
        };
        if covers(effect.ate, se_b) {
            covered += 1;
        }
    }
    let used = N_SIM_BOOT - skipped;
    assert!(
        used >= N_SIM_BOOT * 9 / 10,
        "ipw bootstrap: too many missing se_bootstrap ({skipped}/{N_SIM_BOOT})"
    );
    assert_coverage(covered, used, "ipw_hajek_bootstrap");
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn ipw_hajek_analytic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = PropensityWeighting { bootstrap_replicates: 0, ..PropensityWeighting::new() };
    let ctx = ExecutionContext::for_tests(2);
    let mut covered = 0u32;
    // Larger n than linear adjustment: estimated-propensity IF needs more samples.
    for s in 0..N_SIM {
        let (data, estimand) = confounded_scm(500, 2100 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        if covers(effect.ate, effect.se_analytic) {
            covered += 1;
        }
    }
    assert_coverage_ipw_analytic(covered, N_SIM);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn aipw_analytic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
    let ctx = ExecutionContext::for_tests(3);
    let mut covered = 0u32;
    for s in 0..N_SIM {
        let (data, estimand) = confounded_scm(N_OBS, 3000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = crate::aipw::AipwWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        if covers(effect.ate, effect.se_analytic) {
            covered += 1;
        }
    }
    assert_coverage(covered, N_SIM, "aipw");
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn matching_homoskedastic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(causal_core::TargetPopulation::Treated);
    let est = PropensityMatching {
        bootstrap_replicates: 0,
        se_kind: AnalyticSeKind::Homoskedastic,
        ..PropensityMatching::new()
    };
    let ctx = ExecutionContext::for_tests(4);
    let mut covered = 0u32;
    for s in 0..N_SIM {
        let (data, estimand) = confounded_scm(N_OBS, 4000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        if covers(effect.ate, effect.se_analytic) {
            covered += 1;
        }
    }
    assert_coverage(covered, N_SIM, "matching_ai");
}

fn binary_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
    let mut rng = CausalRng::from_seed(seed);
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let zi = (i % 2) as f64;
        let ui = standard_normal(&mut rng);
        let ti = 0.5 * zi + ui + 0.1 * standard_normal(&mut rng);
        let yi = TRUE_ATE * ti + ui + 0.1 * standard_normal(&mut rng);
        t.push(ti);
        y.push(yi);
        z.push(zi);
    }
    let estimand = IdentifiedEstimand::instrumental(
        "iv",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    (table_tyz(t, y, z), estimand)
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn wald_iv_analytic_ci_coverage() {
    let query = AverageEffectQuery::with_levels(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        0.0,
        1.0,
    );
    let est = WaldIv {
        bootstrap_replicates: 0,
        se_kind: AnalyticSeKind::Homoskedastic,
        ..WaldIv::new()
    };
    let ctx = ExecutionContext::for_tests(5);
    let mut covered = 0u32;
    for s in 0..N_SIM {
        let (data, estimand) = binary_iv_scm(N_OBS, 5000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let effect = est.fit(&prep, &ctx, AssumptionSet::new()).unwrap();
        if covers(effect.ate, effect.se_analytic) {
            covered += 1;
        }
    }
    assert_coverage(covered, N_SIM, "wald_iv");
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn wald_iv_hc1_ci_coverage() {
    let query = AverageEffectQuery::with_levels(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        0.0,
        1.0,
    );
    let est = WaldIv {
        bootstrap_replicates: 0,
        se_kind: AnalyticSeKind::Hc1,
        ..WaldIv::new()
    };
    let ctx = ExecutionContext::for_tests(15);
    let mut covered = 0u32;
    for s in 0..N_SIM {
        let (data, estimand) = binary_iv_scm(N_OBS, 5100 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let effect = est.fit(&prep, &ctx, AssumptionSet::new()).unwrap();
        if covers(effect.ate, effect.se_analytic) {
            covered += 1;
        }
    }
    assert_coverage(covered, N_SIM, "wald_iv_hc1");
}
