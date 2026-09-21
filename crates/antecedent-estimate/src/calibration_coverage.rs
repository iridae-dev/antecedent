//! SE coverage calibration (measured locally by scripts/measure_calibration.sh).
//!
//! These tests are `#[ignore]` so every-PR `cargo test` stays fast. Run via
//! `scripts/gate_calibration.sh`.
//!
//! The band, the floor, the ceiling, the recheck rule and the normal quantile
//! are the shared harness's (`crates/antecedent/tests/common/calibration.rs`),
//! used here rather than restated: two-sided `0.95 ± 3·MCSE` with
//! `MCSE = √(0.95·0.05/N)` — `[0.917, 0.983]` at `N = 400`, so both an
//! under-covering interval and a conservative one fail; below
//! `PRECISION_N_SIM` replicates a rate more than `RECHECK_SHORTFALL` from the
//! level in either direction prints a `calibration-recheck` line and the gate
//! re-runs the test at `ANTECEDENT_CALIBRATION_NSIM=RECHECK_N_SIM`; from
//! `PRECISION_N_SIM` replicates the rate must also lie in
//! `[0.95 − 2·MCSE, 0.95 + 2·MCSE]` (about `[0.940, 0.960]` at 2000). There is
//! no estimator-specific exemption and no halved replicate count to widen the
//! band; each test prints a `calibration ...` line with the rate, MCSE, band,
//! mean interval length, mean SE, and the Monte Carlo SD of the point estimate.
//!
//! In-assumption DGPs gate the estimators under their stated models. DML,
//! DR-Learner and causal forest each have a reported-level (0.95) cell on
//! [`confounded_scm`]. Adversarial cells (weak IV, weak overlap, curved RD,
//! heteroskedastic matching) live in [`static_dgp`] and as ignored tests
//! below; the gate enrols them after the next full remesurement.
//!
//! **Coverage records.** Each gated test measures a construction the facade
//! reports when a study selects that estimator configuration on a `Dag`
//! (`crates/antecedent/tests/common/estimator_level.rs`, checked against the
//! facade by `crates/antecedent/tests/calibration_binding.rs`). The gate runs
//! through the shared harness (`crates/antecedent/tests/common/calibration.rs`,
//! the one `CoverageTally::for_record` emission path), which prints the
//! `calibration-record` line `scripts/collect_coverage_records.py` collects.
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
    MeasurementSpec, RoleHint, SmallRoleSet, StreamDomain, TargetPopulation, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::standard_normal;

use crate::adjustment::LinearAdjustmentAte;
use crate::aipw::AipwAte;
use crate::causal_forest::CausalForest;
use crate::dml::DmlAte;
use crate::dr::DrLearner;
use crate::frontdoor::{FrontDoorTwoStage, FrontDoorWorkspace};
use crate::frontdoor_functional::{
    ARM_LINEAR_ASSUMPTION_ID, FrontDoorFunctional, SATURATED_ASSUMPTION_ID,
};
use crate::iv::{TwoStageLeastSquares, TwoStageLeastSquaresWorkspace, WaldIv};
use crate::propensity::{PropensityEstimationWorkspace, PropensityMatching, PropensityWeighting};
use crate::rd::{RdWorkspace, SharpRegressionDiscontinuity};
use crate::se::AnalyticSeKind;

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "../../antecedent/tests/common/estimator_level.rs"]
mod estimator_level;
#[path = "../../antecedent/tests/common/static_dgp.rs"]
mod static_dgp;

use calibration::{
    Construction, CoverageTally, RECHECK_N_SIM, REPORTED_LEVEL, RecordKey, SKIP_CAP_DEN,
    SKIP_CAP_NUM, ScopeFacts, Z95, coverage_band, coverage_mcse, grid_n, grid_seed, n_sim,
    needs_recheck, passes_precision, precision_ceiling, precision_floor,
};

const TRUE_ATE: f64 = 2.0;
/// Base sample size of the `N_OBS` designs; each grid point measures
/// [`n_obs`] rows (`SampleGrid::STANDARD`: 150, 300, 600).
const N_OBS: usize = 300;

/// Rows of an `N_OBS` design at this run's sample-size grid point.
fn n_obs() -> usize {
    grid_n(N_OBS)
}
/// Bootstrap replicates for IPW SE: R=60 keeps gate runtime acceptable while
/// stabilizing the replicate SD used as `se_bootstrap`.
const BOOT_REPS: u32 = 60;

/// Nominal level of every interval in this file.
const LEVEL: f64 = 0.95;

/// Coverage count plus mean interval length and Monte Carlo spread of the point.
///
/// Mean length and mean SE are over replicates that produced an interval (a
/// finite positive SE); coverage and the point spread are over every scored
/// replicate. A tally built with [`Tally::for_record`] also scores every
/// replicate on the shared harness's record-keyed tally, which gates
/// ([`Tally::assert`]) and emits the coverage record.
#[derive(Default)]
struct Tally {
    covered: u32,
    scored: u32,
    with_interval: u32,
    half_width_sum: f64,
    points: Vec<f64>,
    se_sum: f64,
    record: Option<(CoverageTally, estimator_level::EstimatorLevelCase)>,
}

impl Tally {
    /// Tally backing the coverage record of estimator-level test `test`, on data
    /// generated by `dgp`.
    #[track_caller]
    fn for_record(test: &'static str, dgp: &'static str) -> Self {
        let case = estimator_level::case_for(test);
        let key = RecordKey { test, dgp, interval: case.interval };
        Self { record: Some((CoverageTally::for_record(key, LEVEL), case)), ..Self::default() }
    }

    /// Bind the replicate about to be scored: `rows` complete rows and, for a
    /// bootstrap SE, the replicates that succeeded.
    fn bind(&mut self, rows: usize, replicates_ok: Option<u32>) {
        if let Some((tally, case)) = self.record.as_mut() {
            tally.bind(
                &case.construction(),
                ScopeFacts {
                    row_count: rows as u64,
                    replicates_ok,
                    posterior_draws: None,
                    unidentified_mass: 0.0,
                },
            );
        }
    }

    /// Score `ate ± Z95·se` against `truth`; a non-finite SE is a miss.
    fn record(&mut self, ate: f64, se: f64, truth: f64) {
        self.scored += 1;
        self.points.push(ate);
        let interval = (se.is_finite() && se > 0.0).then_some((ate - Z95 * se, ate + Z95 * se));
        if let Some((tally, _)) = self.record.as_mut() {
            tally.record(interval, truth);
        }
        if se.is_finite() && se > 0.0 {
            self.with_interval += 1;
            self.half_width_sum += Z95 * se;
            self.se_sum += se;
            if (ate - truth).abs() <= Z95 * se {
                self.covered += 1;
            }
        }
    }

    /// Score an Anderson–Rubin set `(lower, upper)` against `truth`.
    ///
    /// An honestly unbounded endpoint counts as covering on that side. A missing
    /// set (union / withheld) is a miss — never collapsed into a Wald SE.
    fn record_ar(&mut self, ate: f64, interval: Option<(f64, f64)>, truth: f64) {
        self.scored += 1;
        self.points.push(ate);
        let covers = match interval {
            Some((lo, hi)) => {
                let left_ok = !lo.is_finite() || truth >= lo;
                let right_ok = !hi.is_finite() || truth <= hi;
                left_ok && right_ok
            }
            None => false,
        };
        if covers {
            self.covered += 1;
        }
        if let Some((lo, hi)) = interval {
            if lo.is_finite() && hi.is_finite() && lo <= hi {
                self.with_interval += 1;
                self.half_width_sum += 0.5 * (hi - lo);
                self.se_sum += (hi - lo) / (2.0 * Z95);
                if let Some((tally, _)) = self.record.as_mut() {
                    tally.record(Some((lo, hi)), truth);
                }
            } else if covers {
                // Unbounded set that covers: count coverage for the harness with a
                // degenerate finite interval so infinite endpoints are not scored as misses.
                self.with_interval += 1;
                if let Some((tally, _)) = self.record.as_mut() {
                    tally.record(Some((truth, truth)), truth);
                }
            } else if let Some((tally, _)) = self.record.as_mut() {
                tally.record(None, truth);
            }
        } else if let Some((tally, _)) = self.record.as_mut() {
            tally.record(None, truth);
        }
    }

    fn rate(&self) -> f64 {
        f64::from(self.covered) / f64::from(self.scored.max(1))
    }

    /// Print the `calibration ...` line without gating (out-of-assumption probes).
    fn report(&self, label: &str) {
        let n = f64::from(self.scored.max(1));
        let with_interval = f64::from(self.with_interval.max(1));
        let (lo, hi) = coverage_band(self.scored.max(1), LEVEL);
        let mcse = coverage_mcse(self.scored.max(1), LEVEL);
        let mean = self.points.iter().sum::<f64>() / n;
        let mc_sd = (self.points.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
            / (n - 1.0).max(1.0))
        .sqrt();
        eprintln!(
            "calibration {label}: nominal={LEVEL:.2} coverage={:.3} mcse={mcse:.4} \
             band=[{lo:.3}, {hi:.3}] mean_length={:.4} mean_se={:.4} mc_sd={mc_sd:.4} \
             mean_point={mean:.4} ({}/{} covered)",
            self.rate(),
            2.0 * self.half_width_sum / with_interval,
            self.se_sum / with_interval,
            self.covered,
            self.scored
        );
    }

    /// Assert a named boundary cell against its *measured* coverage
    /// (`measured ± 3·MCSE`), and record it as a boundary. For a cell whose
    /// precise measurement sits below the precision floor: the gate guards the
    /// measured level instead of claiming a nominal one.
    ///
    /// # Panics
    ///
    /// When no replicate scored, or coverage leaves the measured band.
    fn assert_boundary(&self, label: &str, measured: f64) {
        assert!(self.scored > 0, "{label}: no replicates scored");
        self.report(label);
        let Some((tally, _)) = self.record.as_ref() else {
            panic!("{label}: a boundary cell needs a record tally")
        };
        tally.assert_boundary(measured);
    }

    /// Per-grid-point named boundary. `Some(m)` holds that point to `m`;
    /// `None` gates it at nominal.
    fn assert_boundary_at(&self, label: &str, measured: [Option<f64>; 3]) {
        assert!(self.scored > 0, "{label}: no replicates scored");
        self.report(label);
        let Some((tally, _)) = self.record.as_ref() else {
            panic!("{label}: a boundary cell needs a record tally")
        };
        tally.assert_boundary_at(measured);
    }

    /// Print and gate nominal coverage: the two-sided band, plus the precision
    /// floor and ceiling from `PRECISION_N_SIM` replicates or a
    /// `calibration-recheck` line below that count.
    fn assert(&self, label: &str) {
        assert!(self.scored > 0, "{label}: no replicates scored");
        self.report(label);
        if let Some((tally, _)) = self.record.as_ref() {
            // Same band, floor, ceiling and recheck rule; also emits the record.
            tally.assert();
            return;
        }
        let (lo, hi) = coverage_band(self.scored, LEVEL);
        let rate = self.rate();
        assert!(
            rate >= lo && rate <= hi,
            "{label}: coverage={rate:.3} outside [{lo:.3}, {hi:.3}] ({}/{})",
            self.covered,
            self.scored
        );
        if let Some(floor) = precision_floor(self.scored, LEVEL) {
            assert!(
                rate >= floor,
                "{label}: coverage={rate:.3} below the precision floor {floor:.3} \
                 (level - 2 MCSE at {} replicates; {}/{})",
                self.scored,
                self.covered,
                self.scored
            );
        }
        if let Some(ceiling) = precision_ceiling(self.scored, LEVEL) {
            assert!(
                rate <= ceiling,
                "{label}: coverage={rate:.3} above the precision ceiling {ceiling:.3} \
                 (level + 2 MCSE at {} replicates; {}/{})",
                self.scored,
                self.covered,
                self.scored
            );
        } else if needs_recheck(self.scored, LEVEL, rate) {
            let side = if rate < LEVEL { "below" } else { "above" };
            eprintln!(
                "calibration-recheck {label}: coverage={rate:.3} is more than \
                 {:.2} {side} {LEVEL:.2} at {} replicates; re-run at \
                 ANTECEDENT_CALIBRATION_NSIM={RECHECK_N_SIM}",
                calibration::RECHECK_SHORTFALL,
                self.scored
            );
        }
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
    let mut rng = CausalRng::from_seed(grid_seed(seed));
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
    let mut tally = Tally::for_record("linear_adjustment_analytic_ci_coverage", "confounded_scm");
    for s in 0..n_sim() {
        let (data, estimand) = confounded_scm(n_obs(), 1000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = crate::adjustment::EstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.bind(n_obs(), None);
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
    let mut tally = Tally::for_record("linear_adjustment_hc1_ci_coverage", "confounded_scm");
    for s in 0..n_sim() {
        let (data, estimand) = confounded_scm(n_obs(), 1100 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = crate::adjustment::EstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.bind(n_obs(), None);
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
    let mut tally = Tally::for_record("ipw_hajek_bootstrap_ci_coverage", "confounded_scm");
    for s in 0..n_sim() {
        // One context per simulation: a shared context would hand every
        // simulation the same bootstrap resample indices.
        let ctx = ExecutionContext::for_tests(2000 + u64::from(s));
        let (data, estimand) = confounded_scm(grid_n(500), 2000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let Some(se_b) = effect.se_bootstrap else {
            // Missing SE is a miss in the coverage rate, not a dropped replicate.
            if let Some((record, _)) = tally.record.as_mut() {
                record.skip();
            }
            continue;
        };
        tally.bind(grid_n(500), effect.bootstrap_replicates_ok);
        tally.record(effect.ate, se_b, TRUE_ATE);
    }
    tally.assert("ipw_hajek_bootstrap");
}

/// Stacked logistic + weighted-mean sandwich SE (estimated propensity).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn ipw_hajek_analytic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = PropensityWeighting { bootstrap_replicates: 0, ..PropensityWeighting::new() };
    let ctx = ExecutionContext::for_tests(2);
    let mut tally = Tally::for_record("ipw_hajek_analytic_ci_coverage", "confounded_scm");
    for s in 0..n_sim() {
        let (data, estimand) = confounded_scm(grid_n(500), 2100 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.bind(grid_n(500), None);
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    // Grid point 0 (n=250) measured 0.939 at 2000 replicates, 0.001 under the
    // precision floor. Points 1 and 2 pass the nominal band.
    tally.assert_boundary_at("ipw_hajek_analytic", [Some(0.939), None, None]);
}

/// `Z ~ N(0,1)`, `T ~ Bern(σ(−0.4 + 0.9 Z))`, `Y = 2T + Z + 0.4 ε` (the
/// `conformance/estimate/propensity_ipw` SCM), `n` rows from `rng`.
fn propensity_ipw_conformance_scm(n: usize, rng: &mut CausalRng) -> TabularData {
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(rng);
        let p = 1.0 / (1.0 + (-(-0.4 + 0.9 * zi)).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let noise = standard_normal(rng) * 0.4;
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + noise;
    }
    table_tyz(t, y, z)
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
    let mut tally = Tally::for_record(
        "ipw_hajek_analytic_conformance_scm_ci_coverage",
        "propensity_ipw_conformance_scm",
    );
    for s in 0..n_sim() {
        let mut rng = ExecutionContext::for_tests(grid_seed(3 + 1000 * u64::from(s)))
            .rng
            .stream_for(StreamDomain::Estimate, 0x5051_u64);
        let n = grid_n(1200);
        let data = propensity_ipw_conformance_scm(n, &mut rng);
        let prep = est.prepare(&data, &backdoor_z(), &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.bind(n, None);
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
    let mut tally = Tally::for_record("aipw_analytic_ci_coverage", "confounded_scm");
    for s in 0..n_sim() {
        let (data, estimand) = confounded_scm(n_obs(), 3000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = crate::aipw::AipwWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.bind(n_obs(), None);
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
    let mut rng = CausalRng::from_seed(grid_seed(seed));
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
    test: &'static str,
    label: &str,
    population: TargetPopulation,
    se_kind: AnalyticSeKind,
    cluster_sd: f64,
    seed: u64,
) {
    aipw_residualized_tally(test, label, population, se_kind, cluster_sd, seed).assert(label);
}

/// The ATC cell measures below the precision floor: see
/// [`aipw_atc_hc1_boundary_within_band`].
fn aipw_residualized_tally(
    test: &'static str,
    label: &str,
    population: TargetPopulation,
    se_kind: AnalyticSeKind,
    cluster_sd: f64,
    seed: u64,
) -> Tally {
    let truth = match population {
        TargetPopulation::Treated => 2.0 + heterogeneous_arm_mean_z(true),
        TargetPopulation::Untreated => 2.0 + heterogeneous_arm_mean_z(false),
        _ => 2.0,
    };
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(population);
    let ctx = ExecutionContext::for_tests(seed);
    let mut tally = Tally::for_record(test, "heterogeneous_binary_scm");
    for s in 0..n_sim() {
        let (data, clusters) =
            heterogeneous_binary_scm(grid_n(600), seed + u64::from(s), cluster_sd);
        let mut est = AipwAte { bootstrap_replicates: 0, se_kind, ..AipwAte::new() };
        if matches!(se_kind, AnalyticSeKind::Cluster) {
            est = est.with_cluster_ids(clusters);
        }
        let prep = est.prepare(&data, &backdoor_z(), &query).unwrap();
        let mut ws = crate::aipw::AipwWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.bind(grid_n(600), None);
        tally.record(effect.ate, effect.se_analytic, truth);
    }
    let _ = label;
    tally
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn aipw_att_hc1_ci_coverage() {
    aipw_residualized_coverage(
        "aipw_att_hc1_ci_coverage",
        "aipw_att_hc1",
        TargetPopulation::Treated,
        AnalyticSeKind::Hc1,
        0.0,
        31_000,
    );
}

/// Measured coverage of the ATC cell at 2000 replicates on the gate's seed
/// stream. The influence-function SE averages 0.0788 against a Monte Carlo SD
/// of 0.0812 on this law, so the interval is about 3% too narrow and covers
/// 0.939, just under the one-sided precision floor (0.940). The ATE and ATT
/// cells on the same law and the same branch cover 0.948 and 0.958, so this is
/// the untreated arm's finite-sample shortfall at n = 600, not the branch's.
const AIPW_ATC_HC1_MEASURED: f64 = 0.939;

/// Boundary cell, not a nominal one: the assertion is the band around
/// [`AIPW_ATC_HC1_MEASURED`], and the record it emits is a boundary, so an
/// execution of this construction reports the measured under-coverage instead
/// of `calibrated`.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn aipw_atc_hc1_boundary_within_band() {
    aipw_residualized_tally(
        "aipw_atc_hc1_boundary_within_band",
        "aipw_atc_hc1",
        TargetPopulation::Untreated,
        AnalyticSeKind::Hc1,
        0.0,
        32_000,
    )
    .assert_boundary("aipw_atc_hc1", AIPW_ATC_HC1_MEASURED);
}

/// ATE on the residualized branch (HC1 moves it off the cross-fitted score table).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn aipw_ate_hc1_ci_coverage() {
    aipw_residualized_coverage(
        "aipw_ate_hc1_ci_coverage",
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
        "aipw_att_cluster_ci_coverage",
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
    let mut tally = Tally::for_record("matching_homoskedastic_ci_coverage", "confounded_scm");
    for s in 0..n_sim() {
        let (data, estimand) = confounded_scm(n_obs(), 4000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.bind(n_obs(), None);
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert("matching_ai");
}

/// Binary IV DGP with a **moderate** first stage so a substantial share of
/// draws at `N_OBS` have Stock–Yogo F < 10. Coverage is scored on the
/// Anderson–Rubin set (not a Wald SE behind an F≥10 pretest).
fn binary_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let zi = (i % 2) as f64;
        let ui = standard_normal(&mut rng);
        let ti = 0.45 * zi + ui + 0.1 * standard_normal(&mut rng);
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

fn wald_coverage(test: &'static str, label: &str, se_kind: AnalyticSeKind, seed: u64) {
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    let est = WaldIv { bootstrap_replicates: 0, se_kind, ..WaldIv::new() };
    let ctx = ExecutionContext::for_tests(seed);
    let mut tally = Tally::for_record(test, "binary_iv_scm");
    let mut weak_f = 0u32;
    for s in 0..n_sim() {
        let (data, estimand) = binary_iv_scm(n_obs(), seed * 1000 + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let effect = est.fit(&prep, &ctx, AssumptionSet::new()).unwrap();
        assert!(
            !effect.se_analytic.is_finite(),
            "{label}: Wald SE must not be published (replicate {s})"
        );
        assert!(effect.se_bootstrap.is_none(), "{label}: bootstrap SE must not be published");
        let diag = effect.first_stage_diagnostics.as_ref().expect("first-stage diagnostics");
        if diag.f_statistic.is_finite() && diag.f_statistic < 10.0 {
            weak_f += 1;
        }
        tally.bind(n_obs(), None);
        if matches!(se_kind, AnalyticSeKind::Homoskedastic) {
            let interval = diag.anderson_rubin.map(|(lo, hi, _)| (lo, hi));
            if interval.is_none()
                && diag.uncertainty_withheld == Some("anderson_rubin_set_is_union")
            {
                // AR acceptance set is a union of rays: not published as one interval,
                // but coverage of the set equals whether AR accepts at the truth.
                let z: Vec<f64> = {
                    let n = prep.nrows;
                    (0..n).map(|r| prep.instruments_matrix[n + r]).collect()
                };
                let mut ws = antecedent_stats::LeastSquaresWorkspace::default();
                let ar_true = antecedent_stats::anderson_rubin_statistic(
                    &prep.outcome,
                    &prep.treatment,
                    &z,
                    prep.nrows,
                    1,
                    &prep.exogenous_matrix,
                    prep.x_ncols,
                    TRUE_ATE,
                    &antecedent_stats::FaerBackend,
                    &mut ws,
                )
                .unwrap();
                let crit = antecedent_stats::anderson_rubin_kf_critical(
                    LEVEL,
                    1,
                    prep.nrows.saturating_sub(1 + prep.x_ncols),
                );
                if ar_true.is_finite() && ar_true <= crit {
                    tally.record_ar(effect.ate, Some((f64::NEG_INFINITY, f64::INFINITY)), TRUE_ATE);
                } else {
                    tally.record_ar(effect.ate, None, TRUE_ATE);
                }
            } else {
                tally.record_ar(effect.ate, interval, TRUE_ATE);
            }
        } else {
            assert!(
                diag.anderson_rubin.is_none(),
                "{label}: non-homoskedastic AR must be withheld"
            );
            assert_eq!(diag.uncertainty_withheld, Some("anderson_rubin_requires_homoskedastic"));
            // No licensed interval product — score as a miss, not a Wald SE.
            tally.record_ar(effect.ate, None, TRUE_ATE);
        }
    }
    if matches!(se_kind, AnalyticSeKind::Homoskedastic) {
        let weak_share = f64::from(weak_f) / f64::from(n_sim().max(1));
        assert!(
            weak_share > 0.05,
            "{label}: DGP must leave a non-trivial F<10 share, got {weak_share}"
        );
        tally.assert(label);
    } else {
        // HC1 / robust: licensed product withheld; do not claim Wald coverage.
        tally.report(label);
    }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn wald_iv_analytic_ci_coverage() {
    wald_coverage("wald_iv_analytic_ci_coverage", "wald_iv", AnalyticSeKind::Homoskedastic, 5);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn wald_iv_hc1_ci_coverage() {
    wald_coverage("wald_iv_hc1_ci_coverage", "wald_iv_hc1", AnalyticSeKind::Hc1, 15);
}

// ------------------------------------------------------------------ 2SLS

/// Over-identified 2SLS with an exogenous covariate: `z1, z2, x, u ~ N(0,1)`,
/// `T = 0.6 z1 + 0.4 z2 + 0.5 x + u + 0.5 e`, `Y = 2T + x + u + s·ε` with
/// `s = 1` (homoskedastic) or `s = 0.4 + 0.8|z1|` (variance moving with the
/// excluded instrument, which invalidates the classical 2SLS SE).
/// Columns `t, y, z1, z2, x`; estimand instruments `{z1, z2}`, adjustment `{x}`.
fn two_sls_scm(n: usize, seed: u64, heteroskedastic: bool) -> (TabularData, IdentifiedEstimand) {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
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

fn two_sls_coverage(
    test: &'static str,
    label: &str,
    se_kind: AnalyticSeKind,
    heteroskedastic: bool,
    seed: u64,
) {
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    let est =
        TwoStageLeastSquares { bootstrap_replicates: 0, se_kind, ..TwoStageLeastSquares::new() };
    let ctx = ExecutionContext::for_tests(seed);
    let mut tally = Tally::for_record(test, "two_sls_scm");
    for s in 0..n_sim() {
        let (data, estimand) = two_sls_scm(grid_n(500), seed + u64::from(s), heteroskedastic);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = TwoStageLeastSquaresWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        assert!(!effect.se_analytic.is_finite(), "{label}: Wald SE must not be published");
        assert!(effect.se_bootstrap.is_none());
        let diag = effect.first_stage_diagnostics.as_ref().expect("first-stage diagnostics");
        tally.bind(grid_n(500), None);
        if matches!(se_kind, AnalyticSeKind::Homoskedastic) {
            let interval = diag.anderson_rubin.map(|(lo, hi, _)| (lo, hi));
            if interval.is_none()
                && diag.uncertainty_withheld == Some("anderson_rubin_set_is_union")
            {
                let z_ncols = prep.z_ncols - 1;
                let mut ws = antecedent_stats::LeastSquaresWorkspace::default();
                let ar_true = antecedent_stats::anderson_rubin_statistic(
                    &prep.outcome,
                    &prep.treatment,
                    &prep.instruments_matrix[prep.nrows..],
                    prep.nrows,
                    z_ncols,
                    &prep.exogenous_matrix,
                    prep.x_ncols,
                    TRUE_ATE,
                    &antecedent_stats::FaerBackend,
                    &mut ws,
                )
                .unwrap();
                let crit = antecedent_stats::anderson_rubin_kf_critical(
                    LEVEL,
                    z_ncols,
                    prep.nrows.saturating_sub(z_ncols + prep.x_ncols),
                );
                if ar_true.is_finite() && ar_true <= crit {
                    tally.record_ar(effect.ate, Some((f64::NEG_INFINITY, f64::INFINITY)), TRUE_ATE);
                } else {
                    tally.record_ar(effect.ate, None, TRUE_ATE);
                }
            } else {
                tally.record_ar(effect.ate, interval, TRUE_ATE);
            }
        } else {
            assert!(diag.anderson_rubin.is_none());
            assert_eq!(diag.uncertainty_withheld, Some("anderson_rubin_requires_homoskedastic"));
            tally.record_ar(effect.ate, None, TRUE_ATE);
        }
    }
    if matches!(se_kind, AnalyticSeKind::Homoskedastic) {
        tally.assert(label);
    } else {
        tally.report(label);
    }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn iv_2sls_analytic_ci_coverage() {
    two_sls_coverage(
        "iv_2sls_analytic_ci_coverage",
        "iv_2sls_analytic",
        AnalyticSeKind::Homoskedastic,
        false,
        35_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn iv_2sls_hc1_heteroskedastic_ci_coverage() {
    two_sls_coverage(
        "iv_2sls_hc1_heteroskedastic_ci_coverage",
        "iv_2sls_hc1_heteroskedastic",
        AnalyticSeKind::Hc1,
        true,
        36_000,
    );
}

// ------------------------------------------------------------- front-door

/// `u ~ N(0,1)` unobserved, `T = u + 0.8 e`, `M = T + 0.7 e`,
/// `Y = 2M + 1.5u + s·ε` with `s = 0.5 + 0.5|T|` (heteroskedastic, so the
/// stacked sandwich rather than a classical SE is required). Columns `t, y, m`.
/// Front-door truth: `β_{T→M}·β_{M→Y} = 1·2 = 2`.
fn frontdoor_scm(n: usize, seed: u64) -> TabularData {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
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

fn frontdoor_coverage(test: &'static str, label: &str, se_kind: AnalyticSeKind, seed: u64) {
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    let estimand = IdentifiedEstimand::frontdoor(
        "frontdoor",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    let est = FrontDoorTwoStage { bootstrap_replicates: 0, se_kind, ..FrontDoorTwoStage::new() };
    let ctx = ExecutionContext::for_tests(seed);
    let mut tally = Tally::for_record(test, "frontdoor_scm");
    for s in 0..n_sim() {
        let data = frontdoor_scm(grid_n(400), seed + u64::from(s));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = FrontDoorWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.bind(grid_n(400), None);
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert(label);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frontdoor_stacked_hc0_ci_coverage() {
    frontdoor_coverage(
        "frontdoor_stacked_hc0_ci_coverage",
        "frontdoor_stacked_hc0",
        AnalyticSeKind::Hc0,
        37_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frontdoor_stacked_hc1_ci_coverage() {
    frontdoor_coverage(
        "frontdoor_stacked_hc1_ci_coverage",
        "frontdoor_stacked_hc1",
        AnalyticSeKind::Hc1,
        38_000,
    );
}

// -------------------------------------------------- front-door functional

/// Effect of `T` on `Y` under the outcome equation shared by both front-door functional
/// designs, `Y = 1 + 2·M·(0.5 + U) + 0.5U + 0.6ε` with `E[U] = 0.4`:
/// `2 · (E[M|do(1)] − E[M|do(0)]) · (0.5 + E[U])`.
const fn frontdoor_functional_truth(mediator_shift: f64) -> f64 {
    2.0 * mediator_shift * 0.9
}

/// `U ~ Bern(0.4)` unobserved, `P(T=1|U) = 0.2 + 0.4U` (arms 0.64 / 0.36), a mediator that
/// depends on `T` only, and an outcome in which `U` both shifts `Y` and modifies the
/// mediator's effect. `E[Y|M,T]` therefore carries a treatment-mediator interaction: the
/// product of coefficients is biased here and only the functional is consistent. The
/// mediator is `Bern(0.25 + 0.5T)` (`discrete`) or `1 + 0.4T + (1 + 0.5T)ε`
/// (heteroskedastic across arms; `E[Y|M,T]` is linear in `M` within an arm). Columns
/// `t, y, m`.
fn frontdoor_functional_scm(n: usize, seed: u64, discrete: bool) -> TabularData {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
    let (mut t, mut y, mut m) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let u = f64::from(uniform01(&mut rng) < 0.4);
        t[i] = f64::from(uniform01(&mut rng) < 0.2 + 0.4 * u);
        m[i] = if discrete {
            f64::from(uniform01(&mut rng) < 0.25 + 0.5 * t[i])
        } else {
            1.0 + 0.4 * t[i] + (1.0 + 0.5 * t[i]) * standard_normal(&mut rng)
        };
        y[i] = 1.0 + 2.0 * m[i] * (0.5 + u) + 0.5 * u + 0.6 * standard_normal(&mut rng);
    }
    table(&[("t", &t), ("y", &y), ("m", &m)])
}

fn frontdoor_functional_coverage(test: &'static str, label: &str, discrete: bool, seed: u64) {
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    let estimand = IdentifiedEstimand::frontdoor(
        "frontdoor",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    // `Auto` is what the facade runs: saturated cells for the binary mediator, the per-arm
    // linear outcome regression for the continuous one.
    let est = FrontDoorFunctional::new().with_bootstrap_replicates(0);
    let (dgp, truth, model) = if discrete {
        (
            "frontdoor_functional_discrete_scm",
            frontdoor_functional_truth(0.5),
            SATURATED_ASSUMPTION_ID,
        )
    } else {
        (
            "frontdoor_functional_continuous_scm",
            frontdoor_functional_truth(0.4),
            ARM_LINEAR_ASSUMPTION_ID,
        )
    };
    let ctx = ExecutionContext::for_tests(seed);
    let mut tally = Tally::for_record(test, dgp);
    for s in 0..n_sim() {
        let data = frontdoor_functional_scm(grid_n(400), seed + u64::from(s), discrete);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let effect = est.fit(&prep, &ctx, AssumptionSet::new()).unwrap();
        let recorded = effect.assumptions.entries.iter().any(|r| match &r.assumption {
            antecedent_core::Assumption::ParametricRestriction(p) => p.id.as_ref() == model,
            antecedent_core::Assumption::Custom { id, .. } => id.as_ref() == model,
            _ => false,
        });
        assert!(recorded, "{test}: the design must exercise {model}");
        tally.bind(grid_n(400), None);
        tally.record(effect.ate, effect.se_analytic, truth);
    }
    tally.assert(label);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frontdoor_functional_saturated_ci_coverage() {
    frontdoor_functional_coverage(
        "frontdoor_functional_saturated_ci_coverage",
        "frontdoor_functional_saturated",
        true,
        39_500,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frontdoor_functional_arm_linear_ci_coverage() {
    frontdoor_functional_coverage(
        "frontdoor_functional_arm_linear_ci_coverage",
        "frontdoor_functional_arm_linear",
        false,
        39_600,
    );
}

// ------------------------------------------------------------------ sharp RD

/// `R ~ U(cutoff-bandwidth, cutoff+bandwidth)` (so every draw lands inside the RD window),
/// `T = 1{R ≥ cutoff}`, `Y = 1.0 + 0.5(R-c) + TRUE_ATE·T − 0.8·T·(R-c) + s·noise`. The jump at
/// the cutoff is `TRUE_ATE`, which is the truth the intervals are scored against: the
/// effect for units at the cutoff, the only effect the design identifies. `s = 0.3`
/// (homoskedastic) or `s = 0.1 + 0.6·|R − c|` (variance growing away from the cutoff).
/// Reuses `table_tyz`; the treatment column is the threshold rule, which the estimator
/// checks row by row.
fn rd_scm(
    n: usize,
    seed: u64,
    cutoff: f64,
    bandwidth: f64,
    heteroskedastic: bool,
) -> (TabularData, IdentifiedEstimand) {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
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
        t.push(ti);
        y.push(yi);
        r.push(ri);
    }
    let estimand = IdentifiedEstimand::backdoor("rd.sharp", Arc::from([]), ExprId::from_raw(0));
    (table_tyz(t, y, r), estimand)
}

/// The query a sharp design on `R` (id 2) at `cutoff` answers: the effect at the cutoff.
fn rd_cutoff_query(cutoff: f64) -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(TargetPopulation::local_at_cutoff(VariableId::from_raw(2), cutoff))
}

/// Sharp-RD homoskedastic analytic SE (explicit opt-in) on a homoskedastic DGP
/// (its stated assumption).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn rd_sharp_analytic_ci_coverage() {
    let query = rd_cutoff_query(0.0);
    let est = SharpRegressionDiscontinuity {
        bootstrap_replicates: 0,
        se_kind: AnalyticSeKind::Homoskedastic,
        ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0)
    };
    let ctx = ExecutionContext::for_tests(6);
    let mut tally = Tally::for_record("rd_sharp_analytic_ci_coverage", "rd_scm");
    for s in 0..n_sim() {
        let (data, estimand) = rd_scm(n_obs(), 6000 + u64::from(s), 0.0, 1.0, false);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = RdWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.bind(n_obs(), None);
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.assert("rd_sharp_analytic");
}

/// A design the two gated cells cannot fail on: `R` has density `2(r + 1)/9` on `[−1, 2]`
/// (`R = 3√U − 1`), the baseline `1 + 0.5r + 0.8r² + r³` is curved, and the effect
/// `τ(r) = 2 + 6r` varies with `R`. Closed forms: effect at the cutoff `τ(0) = 2`; average
/// over an `h`-window `2 + 2h²`; population average `2 + 6·E[R] = 8`. Intervals are scored
/// against `τ(0)`, the only one of the three the design identifies.
fn rd_curved_heterogeneous_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);
    for _ in 0..n {
        let ri = 3.0 * uniform01(&mut rng).sqrt() - 1.0;
        let ti = if ri >= 0.0 { 1.0 } else { 0.0 };
        let yi = 1.0
            + 0.5 * ri
            + 0.8 * ri * ri
            + ri * ri * ri
            + ti * (2.0 + 6.0 * ri)
            + 0.3 * standard_normal(&mut rng);
        t.push(ti);
        y.push(yi);
        r.push(ri);
    }
    let estimand = IdentifiedEstimand::backdoor("rd.sharp", Arc::from([]), ExprId::from_raw(0));
    (table_tyz(t, y, r), estimand)
}

/// Sharp-RD HC1 SE under curvature and effect heterogeneity (printed, not gated, no
/// record). The conventional interval ignores smoothing bias, here about `−0.4h³` from
/// the cubic term, so coverage of the cutoff effect depends on the bandwidth; the gated
/// cells above use an exactly piecewise-linear outcome and cannot show that. It shares
/// the construction key of `rd_sharp_hc1_heteroskedastic_ci_coverage`, so turning it into
/// a record needs a rule for two records under one key.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn rd_sharp_hc1_curved_heterogeneous_probe() {
    const CUTOFF_EFFECT: f64 = 2.0;
    let query = rd_cutoff_query(0.0);
    let ctx = ExecutionContext::for_tests(26);
    for bandwidth in [0.4, 0.8] {
        let est = SharpRegressionDiscontinuity {
            bootstrap_replicates: 0,
            se_kind: AnalyticSeKind::Hc1,
            ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, bandwidth)
        };
        let mut probe = Tally::default();
        for s in 0..n_sim() {
            // Only about `4h/9` of rows fall in the window, so draw ten times `n_obs`.
            let (data, estimand) = rd_curved_heterogeneous_scm(10 * n_obs(), 6200 + u64::from(s));
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            let mut ws = RdWorkspace::default();
            let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
            probe.record(effect.ate, effect.se_analytic, CUTOFF_EFFECT);
        }
        probe.report(&format!("rd_sharp_hc1_curved_heterogeneous_probe_h{bandwidth}"));
    }
}

/// Sharp-RD HC1 SE on a heteroskedastic DGP (gated). The homoskedastic SE on the
/// same fits is recorded as an out-of-assumption probe (printed, not gated).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn rd_sharp_hc1_heteroskedastic_ci_coverage() {
    let query = rd_cutoff_query(0.0);
    let hc1 = SharpRegressionDiscontinuity {
        bootstrap_replicates: 0,
        se_kind: AnalyticSeKind::Hc1,
        ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0)
    };
    let classical =
        SharpRegressionDiscontinuity { se_kind: AnalyticSeKind::Homoskedastic, ..hc1.clone() };
    let ctx = ExecutionContext::for_tests(16);
    let mut tally = Tally::for_record("rd_sharp_hc1_heteroskedastic_ci_coverage", "rd_scm");
    // Out-of-assumption probe (homoskedastic SE on a heteroskedastic law): no record.
    let mut probe = Tally::default();
    for s in 0..n_sim() {
        let (data, estimand) = rd_scm(n_obs(), 6100 + u64::from(s), 0.0, 1.0, true);
        let prep = hc1.prepare(&data, &estimand, &query).unwrap();
        let mut ws = RdWorkspace::default();
        let effect = hc1.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.bind(n_obs(), None);
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
        let naive = classical.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        probe.record(naive.ate, naive.se_analytic, TRUE_ATE);
    }
    probe.report("rd_sharp_homoskedastic_se_heteroskedastic_probe");
    tally.assert("rd_sharp_hc1_heteroskedastic");
}

// ------------------------------------------ DML / DR / causal forest (R13)
// Reported-level (0.95) coverage cells on the well-specified confounded SCM.
// Each scores `ate ± Z95·se_analytic` through [`CoverageTally`], counts fit
// failures as skips (misses, 1% cap), and [`CoverageTally::emit`]s a
// `reported_level` record under the same recheck / floor / ceiling rules.

/// Facade construction key for a Frequentist learner ATE on an explicit Dag.
fn learner_construction(estimator: &'static str) -> Construction {
    Construction {
        query: "AverageEffect".into(),
        graph_class: "Dag".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: estimator.into(),
        interval_method: "analytic_se".into(),
        se_kind: String::new(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "all_observed.mean".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn assert_learner_skip_cap(test: &str, skipped: u32, attempts: u32) {
    assert!(attempts > 0, "{test}: no replicates scored");
    assert!(
        skipped.saturating_mul(SKIP_CAP_DEN) <= attempts.saturating_mul(SKIP_CAP_NUM),
        "{test}: {skipped} of {attempts} replicates skipped (cap {SKIP_CAP_NUM}/{SKIP_CAP_DEN})"
    );
}

fn score_learner_replicate(
    tally: &mut CoverageTally,
    construction: &Construction,
    rows: usize,
    result: Result<crate::adjustment::EffectEstimate, crate::error::EstimationError>,
    skipped: &mut u32,
) {
    match result {
        Ok(effect) => {
            tally.bind(
                construction,
                ScopeFacts {
                    row_count: rows as u64,
                    replicates_ok: None,
                    posterior_draws: None,
                    unidentified_mass: 0.0,
                },
            );
            let interval = (effect.se_analytic.is_finite() && effect.se_analytic > 0.0).then_some(
                (effect.ate - Z95 * effect.se_analytic, effect.ate + Z95 * effect.se_analytic),
            );
            tally.record(interval, TRUE_ATE);
        }
        Err(_) => {
            *skipped += 1;
            tally.skip();
        }
    }
}

/// Cross-fitted DML (AIPW score) analytic SE on [`confounded_scm`].
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn dml_analytic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = DmlAte::new();
    let construction = learner_construction("dml");
    let key = RecordKey {
        test: "dml_analytic_ci_coverage",
        dgp: "confounded_scm",
        interval: "analytic_se",
    };
    let mut tally = CoverageTally::for_record(key, REPORTED_LEVEL).unasserted();
    let mut skipped = 0u32;
    let rows = n_obs();
    for s in 0..n_sim() {
        let ctx = ExecutionContext::for_tests(80_000 + u64::from(s));
        let (data, estimand) = confounded_scm(rows, 80_000 + u64::from(s));
        let result = est
            .prepare(&data, &estimand, &query)
            .and_then(|prep| est.fit(&prep, &ctx, AssumptionSet::new()));
        score_learner_replicate(&mut tally, &construction, rows, result, &mut skipped);
    }
    assert_learner_skip_cap("dml_analytic_ci_coverage", skipped, tally.attempts());
    tally.emit();
}

/// DR-Learner analytic SE (ATE from the DR score) on [`confounded_scm`].
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn dr_learner_analytic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = DrLearner::new();
    let construction = learner_construction("dr.learner");
    let key = RecordKey {
        test: "dr_learner_analytic_ci_coverage",
        dgp: "confounded_scm",
        interval: "analytic_se",
    };
    let mut tally = CoverageTally::for_record(key, REPORTED_LEVEL).unasserted();
    let mut skipped = 0u32;
    let rows = n_obs();
    for s in 0..n_sim() {
        let ctx = ExecutionContext::for_tests(81_000 + u64::from(s));
        let (data, estimand) = confounded_scm(rows, 81_000 + u64::from(s));
        let result = est
            .prepare(&data, &estimand, &query)
            .and_then(|prep| est.fit(&prep, &ctx, AssumptionSet::new()));
        score_learner_replicate(&mut tally, &construction, rows, result, &mut skipped);
    }
    assert_learner_skip_cap("dr_learner_analytic_ci_coverage", skipped, tally.attempts());
    tally.emit();
}

/// Honest causal forest marginal ATE (cross-fitted AIPW SE) on [`confounded_scm`].
///
/// Tree count is capped at 40 so a future remesurement stays tractable; the
/// SE comes from the same orthogonal AIPW score the facade publishes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn causal_forest_analytic_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = CausalForest::new().with_n_trees(40);
    let construction = learner_construction("causal.forest");
    let key = RecordKey {
        test: "causal_forest_analytic_ci_coverage",
        dgp: "confounded_scm",
        interval: "analytic_se",
    };
    let mut tally = CoverageTally::for_record(key, REPORTED_LEVEL).unasserted();
    let mut skipped = 0u32;
    let rows = n_obs();
    for s in 0..n_sim() {
        let ctx = ExecutionContext::for_tests(82_000 + u64::from(s));
        let (data, estimand) = confounded_scm(rows, 82_000 + u64::from(s));
        let result = est
            .prepare(&data, &estimand, &query)
            .and_then(|prep| est.fit(&prep, &ctx, AssumptionSet::new()));
        score_learner_replicate(&mut tally, &construction, rows, result, &mut skipped);
    }
    assert_learner_skip_cap("causal_forest_analytic_ci_coverage", skipped, tally.attempts());
    tally.emit();
}

// ---------------------------------------------------------- adversarial cells
// Fixtures the gate will enrol after the next full remesurement. Each sits
// outside the estimator's comfort zone; they are ignored and not yet listed in
// scripts/gate_calibration.sh so this commit does not start a 15–30 h run.

/// Weak-IV Anderson–Rubin coverage on [`static_dgp::weak_iv_data`].
#[test]
#[ignore = "calibration: adversarial cell; enrol after remesurement"]
fn wald_iv_weak_first_stage_adversarial_ci_coverage() {
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    let est =
        WaldIv { bootstrap_replicates: 0, se_kind: AnalyticSeKind::Homoskedastic, ..WaldIv::new() };
    let ctx = ExecutionContext::for_tests(71);
    let mut tally = Tally::default();
    let mut weak_f = 0u32;
    for s in 0..n_sim() {
        let data = static_dgp::weak_iv_data(n_obs(), 71_000 + u64::from(s));
        let estimand = IdentifiedEstimand::instrumental(
            "iv",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let effect = est.fit(&prep, &ctx, AssumptionSet::new()).unwrap();
        let diag = effect.first_stage_diagnostics.as_ref().expect("first-stage diagnostics");
        if diag.f_statistic.is_finite() && diag.f_statistic < 10.0 {
            weak_f += 1;
        }
        let interval = diag.anderson_rubin.map(|(lo, hi, _)| (lo, hi));
        tally.record_ar(effect.ate, interval, TRUE_ATE);
    }
    let weak_share = f64::from(weak_f) / f64::from(n_sim().max(1));
    assert!(
        weak_share > 0.2,
        "weak_iv adversarial DGP must leave a large F<10 share, got {weak_share}"
    );
    tally.report("wald_iv_weak_first_stage_adversarial");
}

/// IPW analytic SE under [`static_dgp::weak_overlap_data`].
#[test]
#[ignore = "calibration: adversarial cell; enrol after remesurement"]
fn ipw_hajek_weak_overlap_adversarial_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = PropensityWeighting { bootstrap_replicates: 0, ..PropensityWeighting::new() };
    let ctx = ExecutionContext::for_tests(72);
    let mut tally = Tally::default();
    for s in 0..n_sim() {
        let data = static_dgp::weak_overlap_data(grid_n(500), 72_000 + u64::from(s));
        let prep = est.prepare(&data, &backdoor_z(), &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.report("ipw_hajek_weak_overlap_adversarial");
}

/// Sharp-RD HC1 under [`static_dgp::curved_rd_data`] (cubic bias + heterogeneous τ).
#[test]
#[ignore = "calibration: adversarial cell; enrol after remesurement"]
fn rd_sharp_hc1_curved_adversarial_ci_coverage() {
    const CUTOFF_EFFECT: f64 = 2.0;
    let query = rd_cutoff_query(0.0);
    let est = SharpRegressionDiscontinuity {
        bootstrap_replicates: 0,
        se_kind: AnalyticSeKind::Hc1,
        ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 0.8)
    };
    let ctx = ExecutionContext::for_tests(73);
    let mut tally = Tally::default();
    for s in 0..n_sim() {
        let data = static_dgp::curved_rd_data(10 * n_obs(), 73_000 + u64::from(s));
        // Map r → z column expected by table_tyz / rd estimator (ids t,y,r as 0,1,2).
        let estimand = IdentifiedEstimand::backdoor("rd.sharp", Arc::from([]), ExprId::from_raw(0));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = RdWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, CUTOFF_EFFECT);
    }
    tally.report("rd_sharp_hc1_curved_adversarial");
}

/// Matching (homoskedastic SE) under [`static_dgp::heteroskedastic_matching_data`].
#[test]
#[ignore = "calibration: adversarial cell; enrol after remesurement"]
fn matching_heteroskedastic_adversarial_ci_coverage() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(TargetPopulation::Treated);
    let est = PropensityMatching {
        bootstrap_replicates: 0,
        se_kind: AnalyticSeKind::Homoskedastic,
        ..PropensityMatching::new()
    };
    let ctx = ExecutionContext::for_tests(74);
    let mut tally = Tally::default();
    for s in 0..n_sim() {
        let data = static_dgp::heteroskedastic_matching_data(n_obs(), 74_000 + u64::from(s));
        let prep = est.prepare(&data, &backdoor_z(), &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        tally.record(effect.ate, effect.se_analytic, TRUE_ATE);
    }
    tally.report("matching_heteroskedastic_adversarial");
}

// --------------------------------------------- precision-layer unit tests (R13)

/// A reported-level (0.95) rate below the precision floor fails the pass predicate.
#[test]
fn reported_level_below_precision_floor_fails() {
    let floor = precision_floor(RECHECK_N_SIM, REPORTED_LEVEL).expect("floor at recheck n");
    let (lo, hi) = coverage_band(RECHECK_N_SIM, REPORTED_LEVEL);
    let rate = floor - 0.001;
    assert!(rate > lo && rate < hi, "fixture rate {rate} must sit inside [{lo}, {hi}]");
    assert!(
        !passes_precision(RECHECK_N_SIM, REPORTED_LEVEL, rate),
        "rate {rate} must fail below floor {floor}"
    );
    let mut tally = CoverageTally::new("reported_level_below_floor", REPORTED_LEVEL);
    // 0.939 at 2000: inside the ±3·MCSE band, under the floor (~0.940).
    for i in 0..RECHECK_N_SIM {
        let truth = if i < 1878 { 0.5 } else { 2.0 };
        tally.record(Some((0.0, 1.0)), truth);
    }
    assert!((tally.rate() - 0.939).abs() < 1e-12);
    assert!(tally.rate() < floor);
    assert!(std::panic::catch_unwind(|| tally.assert()).is_err());
}

/// A rate above the precision ceiling fails the pass predicate.
#[test]
fn coverage_above_precision_ceiling_fails() {
    let ceiling = precision_ceiling(RECHECK_N_SIM, REPORTED_LEVEL).expect("ceiling at recheck n");
    let (lo, hi) = coverage_band(RECHECK_N_SIM, REPORTED_LEVEL);
    let rate = ceiling + 0.001;
    assert!(rate > lo && rate < hi, "fixture rate {rate} must sit inside [{lo}, {hi}]");
    assert!(
        !passes_precision(RECHECK_N_SIM, REPORTED_LEVEL, rate),
        "rate {rate} must fail above ceiling {ceiling}"
    );
    let mut tally = CoverageTally::new("above_ceiling", REPORTED_LEVEL);
    // 0.961 at 2000: inside the ±3·MCSE band, over the ceiling (~0.960).
    for i in 0..RECHECK_N_SIM {
        let truth = if i < 1922 { 0.5 } else { 2.0 };
        tally.record(Some((0.0, 1.0)), truth);
    }
    assert!((tally.rate() - 0.961).abs() < 1e-12);
    assert!(tally.rate() > ceiling);
    assert!(std::panic::catch_unwind(|| tally.assert()).is_err());
}

/// A skip leaves the coverage denominator and counts as a miss.
#[test]
fn skip_counts_as_coverage_miss() {
    let mut tally = CoverageTally::new("skip_miss", REPORTED_LEVEL);
    tally.record(Some((0.0, 1.0)), 0.5);
    tally.skip();
    assert_eq!(tally.attempts(), 2);
    assert!((tally.rate() - 0.5).abs() < 1e-12, "skip must dilute coverage, got {}", tally.rate());
    assert!(needs_recheck(400, REPORTED_LEVEL, 0.925));
    assert!(needs_recheck(400, REPORTED_LEVEL, 0.975), "recheck is symmetric on the high side");
    assert!(!needs_recheck(400, REPORTED_LEVEL, 0.95));
}
