//! Scheduled SE coverage calibration.
//!
//! These tests are `#[ignore]` so every-PR `cargo test` stays fast. Run via
//! `scripts/gate_calibration.sh`.
//!
//! Acceptance band: two-sided `0.95 ± 3·MCSE` with `MCSE = √(0.95·0.05/N)`,
//! the same rule as the 1.9 harness in `crates/antecedent/tests/common/calibration.rs`.
//! At `N = 400` that is `[0.917, 0.983]`, so both an under-covering interval
//! and a conservative (too wide) one fail. There is no floor or ceiling and no
//! estimator-specific exemption; each test prints a `calibration ...` line with
//! the rate, MCSE, band, mean interval length, mean SE, and the Monte Carlo SD
//! of the point estimate.
//!
//! Every DGP here is inside the estimator's stated assumptions (correct
//! nuisance families, the SE kind's variance model). Out-of-assumption probes
//! print their coverage through [`Tally::report`] and are not gated.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalRng, CausalSchemaBuilder, ExecutionContext,
    MeasurementSpec, RoleHint, SmallRoleSet, TargetPopulation, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::standard_normal;

use crate::adjustment::LinearAdjustmentAte;
use crate::aipw::AipwAte;
use crate::frontdoor::{FrontDoorTwoStage, FrontDoorWorkspace};
use crate::iv::{TwoStageLeastSquares, TwoStageLeastSquaresWorkspace, WaldIv};
use crate::propensity::{PropensityEstimationWorkspace, PropensityMatching, PropensityWeighting};
use crate::rd::{RdWorkspace, SharpRegressionDiscontinuity};
use crate::se::AnalyticSeKind;

const TRUE_ATE: f64 = 2.0;
/// Default Monte Carlo budget for analytic SE coverage (runtime OK on weekly gate).
const N_SIM: u32 = 400;
/// Bootstrap IPW is heavier; 200 replicates give a band of `[0.904, 0.996]`.
const N_SIM_BOOT: u32 = 200;
const N_OBS: usize = 300;
const Z95: f64 = 1.96;
/// Bootstrap replicates for IPW SE: R=60 keeps gate runtime acceptable while
/// stabilizing the replicate SD used as `se_bootstrap`.
const BOOT_REPS: u32 = 60;

/// Nominal level of every interval in this file.
const LEVEL: f64 = 0.95;

/// Two-sided acceptance band `LEVEL ± 3·MCSE` (no floor, no cap below 1).
fn coverage_band(n_sim: u32) -> (f64, f64) {
    let mcse = (LEVEL * (1.0 - LEVEL) / f64::from(n_sim)).sqrt();
    ((LEVEL - 3.0 * mcse).max(0.0), (LEVEL + 3.0 * mcse).min(1.0))
}

/// Coverage count plus mean interval length and Monte Carlo spread of the point.
#[derive(Default)]
struct Tally {
    covered: u32,
    scored: u32,
    half_width_sum: f64,
    points: Vec<f64>,
    se_sum: f64,
}

impl Tally {
    /// Score `ate ± Z95·se` against `truth`; a non-finite SE is a miss.
    fn record(&mut self, ate: f64, se: f64, truth: f64) {
        self.scored += 1;
        self.points.push(ate);
        if se.is_finite() && se > 0.0 {
            self.half_width_sum += Z95 * se;
            self.se_sum += se;
            if (ate - truth).abs() <= Z95 * se {
                self.covered += 1;
            }
        }
    }

    fn rate(&self) -> f64 {
        f64::from(self.covered) / f64::from(self.scored.max(1))
    }

    /// Print the `calibration ...` line without gating (out-of-assumption probes).
    fn report(&self, label: &str) {
        let n = f64::from(self.scored.max(1));
        let (lo, hi) = coverage_band(self.scored.max(1));
        let mcse = (LEVEL * (1.0 - LEVEL) / n).sqrt();
        let mean = self.points.iter().sum::<f64>() / n;
        let mc_sd = (self.points.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
            / (n - 1.0).max(1.0))
        .sqrt();
        eprintln!(
            "calibration {label}: nominal={LEVEL:.2} coverage={:.3} mcse={mcse:.4} \
             band=[{lo:.3}, {hi:.3}] mean_length={:.4} mean_se={:.4} mc_sd={mc_sd:.4} \
             mean_point={mean:.4} ({}/{} covered)",
            self.rate(),
            2.0 * self.half_width_sum / n,
            self.se_sum / n,
            self.covered,
            self.scored
        );
    }

    /// Print and gate two-sided nominal coverage.
    fn assert(&self, label: &str) {
        assert!(self.scored > 0, "{label}: no replicates scored");
        self.report(label);
        let (lo, hi) = coverage_band(self.scored);
        let rate = self.rate();
        assert!(
            rate >= lo && rate <= hi,
            "{label}: coverage={rate:.3} outside [{lo:.3}, {hi:.3}] ({}/{})",
            self.covered,
            self.scored
        );
    }
}

/// Columns → `TabularData` (variable ids follow column order).
fn table(columns: &[(&str, &[f64])]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().map(|(name, col)| (*name, *col))).unwrap()
}

fn uniform01(rng: &mut CausalRng) -> f64 {
    rng.next_u64() as f64 / (u64::MAX as f64)
}

fn schema_tyz() -> antecedent_core::CausalSchema {
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

fn backdoor_z() -> IdentifiedEstimand {
    IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    )
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
        let ti = if uniform01(&mut rng) < p { 1.0 } else { 0.0 };
        let yi = TRUE_ATE * ti + 1.5 * zi + 0.5 * ui + 0.5 * standard_normal(&mut rng);
        t.push(ti);
        y.push(yi);
        z.push(zi);
    }
    (table_tyz(t, y, z), backdoor_z())
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn linear_adjustment_analytic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::default() };
    let ctx = ExecutionContext::for_tests(1);
    let mut tally = Tally::default();
    for s in 0..N_SIM {
        let (data, estimand) = confounded_scm(N_OBS, 1000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = crate::adjustment::EstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert("linear_adjustment");
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
    let mut tally = Tally::default();
    for s in 0..N_SIM {
        let (data, estimand) = confounded_scm(N_OBS, 1100 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = crate::adjustment::EstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert("linear_adjustment_hc1");
}

/// IPW bootstrap SE (refits the propensity on every resample).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn ipw_hajek_bootstrap_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = PropensityWeighting { bootstrap_replicates: BOOT_REPS, ..PropensityWeighting::new() };
    let ctx = ExecutionContext::for_tests(2);
    let mut tally = Tally::default();
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
        tally.record(effect.ate, se_b, TRUE_ATE);
    }
    assert!(
        skipped * 20 <= N_SIM_BOOT,
        "ipw bootstrap: too many missing se_bootstrap ({skipped}/{N_SIM_BOOT})"
    );
    tally.assert("ipw_hajek_bootstrap");
}

/// Stacked logistic + weighted-mean sandwich SE (estimated propensity).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn ipw_hajek_analytic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = PropensityWeighting { bootstrap_replicates: 0, ..PropensityWeighting::new() };
    let ctx = ExecutionContext::for_tests(2);
    let mut tally = Tally::default();
    for s in 0..N_SIM {
        let (data, estimand) = confounded_scm(500, 2100 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert("ipw_hajek_analytic");
}

/// The `conformance/estimate/propensity_ipw` SCM at the fixture's `n = 1200`:
/// `Z ~ N(0,1)`, `T ~ Bern(σ(−0.4 + 0.9 Z))`, `Y = 2T + Z + 0.4 ε`.
///
/// Checks the lead that the Hajek analytic SE (0.074 on one fixture draw) is
/// wider than the estimator's Monte Carlo SD (≈0.044): the printed `mean_se`
/// and `mc_sd` are the direct comparison, and the two-sided band fails a
/// conservative SE as surely as an anti-conservative one.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn ipw_hajek_analytic_conformance_scm_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = PropensityWeighting { bootstrap_replicates: 0, ..PropensityWeighting::new() };
    let ctx = ExecutionContext::for_tests(9);
    let mut tally = Tally::default();
    for s in 0..N_SIM {
        let mut rng = ExecutionContext::for_tests(3 + 1000 * u64::from(s)).rng.stream(0x5051_u64);
        let n = 1200;
        let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let p = 1.0 / (1.0 + (-(-0.4 + 0.9 * zi)).exp());
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            let noise = standard_normal(&mut rng) * 0.4;
            z[i] = zi;
            t[i] = ti;
            y[i] = 2.0 * ti + zi + noise;
        }
        let data = table_tyz(t, y, z);
        let prep = est.prepare(&data, &backdoor_z(), &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert("ipw_hajek_analytic_conformance_scm");
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn aipw_analytic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
    let ctx = ExecutionContext::for_tests(3);
    let mut tally = Tally::default();
    for s in 0..N_SIM {
        let (data, estimand) = confounded_scm(N_OBS, 3000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = crate::aipw::AipwWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert("aipw");
}

// ------------------------------------------------ AIPW residualized branch

/// `Z ~ N(0,1)`, `T ~ Bern(σ(0.8 Z))`, `Y = 2T + T·Z + Z + U_g + 0.6 ε` with an
/// optional cluster outcome shock `U_g ~ N(0, cluster_sd²)` shared by the 10
/// rows of each cluster. The per-arm outcome regressions are linear in `Z` and
/// the propensity is logistic in `Z`, so both AIPW nuisances are correctly
/// specified; the effect is heterogeneous, so ATT ≠ ATE ≠ ATC. Columns
/// `t, y, z`; also returns the cluster labels.
fn heterogeneous_binary_scm(n: usize, seed: u64, cluster_sd: f64) -> (TabularData, Vec<u32>) {
    let mut rng = CausalRng::from_seed(seed);
    let (mut t, mut y, mut z, mut g) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0u32; n]);
    let mut cluster_shock = 0.0;
    for i in 0..n {
        if i % 10 == 0 {
            cluster_shock = cluster_sd * standard_normal(&mut rng);
        }
        g[i] = (i / 10) as u32;
        z[i] = standard_normal(&mut rng);
        let p = 1.0 / (1.0 + (-0.8 * z[i]).exp());
        t[i] = if uniform01(&mut rng) < p { 1.0 } else { 0.0 };
        y[i] = 2.0 * t[i] + t[i] * z[i] + z[i] + cluster_shock + 0.6 * standard_normal(&mut rng);
    }
    (table(&[("t", &t), ("y", &y), ("z", &z)]), g)
}

/// `E[Z | T = arm]` for `Z ~ N(0,1)`, `P(T=1|Z) = σ(0.8 Z)`, by quadrature.
fn heterogeneous_arm_mean_z(treated: bool) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for k in 0..18_000 {
        let z = -9.0 + f64::from(k) * 1e-3;
        let p = 1.0 / (1.0 + (-0.8 * z).exp());
        let w = (-0.5 * z * z).exp() * if treated { p } else { 1.0 - p };
        num += w * z;
        den += w;
    }
    num / den
}

fn aipw_residualized_coverage(
    label: &str,
    population: TargetPopulation,
    se_kind: AnalyticSeKind,
    cluster_sd: f64,
    seed: u64,
) {
    let truth = match population {
        TargetPopulation::Treated => 2.0 + heterogeneous_arm_mean_z(true),
        TargetPopulation::Untreated => 2.0 + heterogeneous_arm_mean_z(false),
        _ => 2.0,
    };
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(population);
    let ctx = ExecutionContext::for_tests(seed);
    let mut tally = Tally::default();
    for s in 0..N_SIM {
        let (data, clusters) = heterogeneous_binary_scm(600, seed + u64::from(s), cluster_sd);
        let mut est = AipwAte { bootstrap_replicates: 0, se_kind, ..AipwAte::new() };
        if matches!(se_kind, AnalyticSeKind::Cluster) {
            est = est.with_cluster_ids(clusters);
        }
        let prep = est.prepare(&data, &backdoor_z(), &query).unwrap();
        let mut ws = crate::aipw::AipwWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, truth);
    }
    tally.assert(label);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn aipw_att_hc1_ci_coverage() {
    aipw_residualized_coverage(
        "aipw_att_hc1",
        TargetPopulation::Treated,
        AnalyticSeKind::Hc1,
        0.0,
        31_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn aipw_atc_hc1_ci_coverage() {
    aipw_residualized_coverage(
        "aipw_atc_hc1",
        TargetPopulation::Untreated,
        AnalyticSeKind::Hc1,
        0.0,
        32_000,
    );
}

/// ATE on the residualized branch (HC1 moves it off the cross-fitted score table).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn aipw_ate_hc1_ci_coverage() {
    aipw_residualized_coverage(
        "aipw_ate_hc1",
        TargetPopulation::AllObserved,
        AnalyticSeKind::Hc1,
        0.0,
        33_000,
    );
}

/// Clustered outcome shocks (60 clusters of 10): the cluster-robust IF SE.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn aipw_att_cluster_ci_coverage() {
    aipw_residualized_coverage(
        "aipw_att_cluster",
        TargetPopulation::Treated,
        AnalyticSeKind::Cluster,
        0.8,
        34_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn matching_homoskedastic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(TargetPopulation::Treated);
    let est = PropensityMatching {
        bootstrap_replicates: 0,
        se_kind: AnalyticSeKind::Homoskedastic,
        ..PropensityMatching::new()
    };
    let ctx = ExecutionContext::for_tests(4);
    let mut tally = Tally::default();
    for s in 0..N_SIM {
        let (data, estimand) = confounded_scm(N_OBS, 4000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert("matching_ai");
}

/// Continuous-treatment IV DGP with a **strong** first stage.
///
/// The first-stage coefficient on `Z` must keep Stock–Yogo F well above 10 at
/// `N_OBS` across the calibration seed grid. A weaker `0.5·Z` DGP leaves a
/// non-trivial share of draws with `F < 10`; after `se_if_strong_instrument`
/// those trials publish `se_analytic = NaN` and would be counted as coverage
/// misses even though the procedure correctly refused the SE.
fn binary_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
    let mut rng = CausalRng::from_seed(seed);
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let zi = (i % 2) as f64;
        let ui = standard_normal(&mut rng);
        let ti = 1.5 * zi + ui + 0.1 * standard_normal(&mut rng);
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

fn wald_coverage(label: &str, se_kind: AnalyticSeKind, seed: u64) {
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    let est = WaldIv { bootstrap_replicates: 0, se_kind, ..WaldIv::new() };
    let ctx = ExecutionContext::for_tests(seed);
    let mut tally = Tally::default();
    for s in 0..N_SIM {
        let (data, estimand) = binary_iv_scm(N_OBS, seed * 1000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let effect = est.fit(&prep, &ctx, AssumptionSet::new()).unwrap();
        // Strong-instrument DGP: every draw must publish a finite SE.
        assert!(
            effect.se_analytic.is_finite() && effect.se_analytic > 0.0,
            "{label}: unexpected weak first stage (se_analytic non-finite) on replicate {s}"
        );
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert(label);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn wald_iv_analytic_ci_coverage() {
    wald_coverage("wald_iv", AnalyticSeKind::Homoskedastic, 5);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn wald_iv_hc1_ci_coverage() {
    wald_coverage("wald_iv_hc1", AnalyticSeKind::Hc1, 15);
}

// ------------------------------------------------------------------ 2SLS

/// Over-identified 2SLS with an exogenous covariate: `z1, z2, x, u ~ N(0,1)`,
/// `T = 0.6 z1 + 0.4 z2 + 0.5 x + u + 0.5 e`, `Y = 2T + x + u + s·ε` with
/// `s = 1` (homoskedastic) or `s = 0.4 + 0.8|z1|` (variance moving with the
/// excluded instrument, which invalidates the classical 2SLS SE).
/// Columns `t, y, z1, z2, x`; estimand instruments `{z1, z2}`, adjustment `{x}`.
fn two_sls_scm(n: usize, seed: u64, heteroskedastic: bool) -> (TabularData, IdentifiedEstimand) {
    let mut rng = CausalRng::from_seed(seed);
    let cols: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0; n]);
    let [mut t, mut y, mut z1, mut z2, mut x] = cols;
    for i in 0..n {
        z1[i] = standard_normal(&mut rng);
        z2[i] = standard_normal(&mut rng);
        x[i] = standard_normal(&mut rng);
        let u = standard_normal(&mut rng);
        t[i] = 0.6 * z1[i] + 0.4 * z2[i] + 0.5 * x[i] + u + 0.5 * standard_normal(&mut rng);
        let s = if heteroskedastic { 0.4 + 0.8 * z1[i].abs() } else { 1.0 };
        y[i] = TRUE_ATE * t[i] + x[i] + u + s * standard_normal(&mut rng);
    }
    let estimand = IdentifiedEstimand::new(
        "iv",
        Arc::from([VariableId::from_raw(4)]),
        Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
        Arc::from([]),
        ExprId::from_raw(0),
        None,
    );
    (table(&[("t", &t), ("y", &y), ("z1", &z1), ("z2", &z2), ("x", &x)]), estimand)
}

fn two_sls_coverage(label: &str, se_kind: AnalyticSeKind, heteroskedastic: bool, seed: u64) {
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    let est =
        TwoStageLeastSquares { bootstrap_replicates: 0, se_kind, ..TwoStageLeastSquares::new() };
    let ctx = ExecutionContext::for_tests(seed);
    let mut tally = Tally::default();
    for s in 0..N_SIM {
        let (data, estimand) = two_sls_scm(500, seed + u64::from(s), heteroskedastic);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = TwoStageLeastSquaresWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert(label);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn iv_2sls_analytic_ci_coverage() {
    two_sls_coverage("iv_2sls_analytic", AnalyticSeKind::Homoskedastic, false, 35_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn iv_2sls_hc1_heteroskedastic_ci_coverage() {
    two_sls_coverage("iv_2sls_hc1_heteroskedastic", AnalyticSeKind::Hc1, true, 36_000);
}

// ------------------------------------------------------------- front-door

/// `u ~ N(0,1)` unobserved, `T = u + 0.8 e`, `M = T + 0.7 e`,
/// `Y = 2M + 1.5u + s·ε` with `s = 0.5 + 0.5|T|` (heteroskedastic, so the
/// stacked sandwich rather than a classical SE is required). Columns `t, y, m`.
/// Front-door truth: `β_{T→M}·β_{M→Y} = 1·2 = 2`.
fn frontdoor_scm(n: usize, seed: u64) -> TabularData {
    let mut rng = CausalRng::from_seed(seed);
    let (mut t, mut y, mut m) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let u = standard_normal(&mut rng);
        t[i] = u + 0.8 * standard_normal(&mut rng);
        m[i] = t[i] + 0.7 * standard_normal(&mut rng);
        let s = 0.5 + 0.5 * t[i].abs();
        y[i] = TRUE_ATE * m[i] + 1.5 * u + s * standard_normal(&mut rng);
    }
    table(&[("t", &t), ("y", &y), ("m", &m)])
}

fn frontdoor_coverage(label: &str, se_kind: AnalyticSeKind, seed: u64) {
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    let estimand = IdentifiedEstimand::frontdoor(
        "frontdoor",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    let est = FrontDoorTwoStage { bootstrap_replicates: 0, se_kind, ..FrontDoorTwoStage::new() };
    let ctx = ExecutionContext::for_tests(seed);
    let mut tally = Tally::default();
    for s in 0..N_SIM {
        let data = frontdoor_scm(400, seed + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = FrontDoorWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert(label);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frontdoor_stacked_hc0_ci_coverage() {
    frontdoor_coverage("frontdoor_stacked_hc0", AnalyticSeKind::Hc0, 37_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frontdoor_stacked_hc1_ci_coverage() {
    frontdoor_coverage("frontdoor_stacked_hc1", AnalyticSeKind::Hc1, 38_000);
}

// ------------------------------------------------------------------ sharp RD

/// `R ~ U(cutoff-bandwidth, cutoff+bandwidth)` (so every draw lands inside the RD window),
/// `T = 1{R ≥ cutoff}`, `Y = 1.0 + 0.5(R-c) + TRUE_ATE·T − 0.8·T·(R-c) + s·noise`. The jump at
/// the cutoff is `TRUE_ATE`. `s = 0.3` (homoskedastic) or `s = 0.1 + 0.6·|R − c|` (variance
/// growing away from the cutoff). Reuses `table_tyz` (the treatment column is a
/// required-but-unused placeholder: RD derives treatment from the running variable).
fn rd_scm(
    n: usize,
    seed: u64,
    cutoff: f64,
    bandwidth: f64,
    heteroskedastic: bool,
) -> (TabularData, IdentifiedEstimand) {
    let mut rng = CausalRng::from_seed(seed);
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);
    for _ in 0..n {
        let u = uniform01(&mut rng);
        let ri = cutoff - bandwidth + 2.0 * bandwidth * u;
        let centered = ri - cutoff;
        let ti = if centered >= 0.0 { 1.0 } else { 0.0 };
        let s = if heteroskedastic { 0.1 + 0.6 * centered.abs() } else { 0.3 };
        let yi = 1.0 + 0.5 * centered + TRUE_ATE * ti - 0.8 * ti * centered
            + s * standard_normal(&mut rng);
        t.push(0.0);
        y.push(yi);
        r.push(ri);
    }
    let estimand = IdentifiedEstimand::backdoor("rd.sharp", Arc::from([]), ExprId::from_raw(0));
    (table_tyz(t, y, r), estimand)
}

/// Sharp-RD homoskedastic analytic SE on a homoskedastic DGP (its stated assumption).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn rd_sharp_analytic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = SharpRegressionDiscontinuity {
        bootstrap_replicates: 0,
        ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0)
    };
    let ctx = ExecutionContext::for_tests(6);
    let mut tally = Tally::default();
    for s in 0..N_SIM {
        let (data, estimand) = rd_scm(N_OBS, 6000 + u64::from(s), 0.0, 1.0, false);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = RdWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert("rd_sharp_analytic");
}

/// Sharp-RD HC1 SE on a heteroskedastic DGP (gated). The homoskedastic SE on the
/// same fits is recorded as an out-of-assumption probe (printed, not gated).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn rd_sharp_hc1_heteroskedastic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let hc1 = SharpRegressionDiscontinuity {
        bootstrap_replicates: 0,
        se_kind: AnalyticSeKind::Hc1,
        ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0)
    };
    let classical =
        SharpRegressionDiscontinuity { se_kind: AnalyticSeKind::Homoskedastic, ..hc1.clone() };
    let ctx = ExecutionContext::for_tests(16);
    let mut tally = Tally::default();
    let mut probe = Tally::default();
    for s in 0..N_SIM {
        let (data, estimand) = rd_scm(N_OBS, 6100 + u64::from(s), 0.0, 1.0, true);
        let prep = hc1.prepare(&data, &estimand, &query).unwrap();
        let mut ws = RdWorkspace::default();
        let effect = hc1.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
        let naive = classical.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        probe.record(naive.ate, naive.se_analytic, TRUE_ATE);
    }
    probe.report("rd_sharp_homoskedastic_se_heteroskedastic_probe");
    tally.assert("rd_sharp_hc1_heteroskedastic");
}
