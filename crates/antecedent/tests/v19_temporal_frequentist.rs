//! 1.9 coverage of dependence-honest Frequentist TemporalDag intervals (R-1, R-2).
//!
//! Plain TemporalDag Pulse / single-step Sustained and temporal mediation
//! (Total, Direct, Mediated) are calibrated under iid noise and under AR(1)
//! outcome residuals. The treatment is persistent through an observed driver
//! `z` (`x_t = Σ_k W[k]·z_{t-k} + u_t`, a graph-certified MA(3)), so the
//! regression scores `x_{t-h}·e_t` are serially correlated whenever the residual
//! is: an iid SE or iid row bootstrap under-covers on these DGPs. No lagged
//! outcome enters the adjustment set, so the residual autocorrelation is not
//! absorbed. (A treatment self-loop would make persistence explicit, but
//! temporal backdoor identification cannot certify a self-looped treatment over
//! a finite window.)
//!
//! Truths are the population values of the reported estimands (linear-Gaussian):
//! Pulse h=1 and single-step Sustained `BETA` (`y_t = BETA·x_{t-1} + e_t`);
//! Pulse h=2 `ALPHA·DELTA`, propagated through an intermediate
//! `w_t = ALPHA·x_{t-1} + ν_t` into `y_t = DELTA·w_{t-1} + e_t`, so the h=2
//! regression residual `DELTA·ν_{t-1} + e_t` carries the intermediate shock;
//! mediation Total `C + A·B`, Direct `C`, Mediated `A·B`. Every graph yields a
//! single temporal-backdoor estimand with an empty adjustment set.
//!
//! Ignored coverage tests run via `scripts/gate_calibration.sh`. The
//! non-ignored tests pin that the circular-block path is the one taken.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

mod common;

use antecedent::estimate::TemporalMediationUncertainty;
use antecedent::{InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, DiagnosticSeverity, ExecutionContext, Lag, MediationContrast, MediationQuery,
    TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};
use common::calibration::{CoverageTally, Z90, ar1_noise, gaussian, n_sim, normal_interval};

/// Driver weights: `x_t = Σ_k W[k]·z_{t-k} + U_SD·u_t` (lag-1 autocorrelation ≈ 0.5).
const W: [f64; 4] = [0.75, 0.375, 0.1875, 0.094];
const U_SD: f64 = 0.35;
/// Effect of `x_{t-1}` on `y_t` (one-step design).
const BETA: f64 = 0.8;
/// Two-step design: `x_{t-1} → w_t` and `w_{t-1} → y_t`.
const ALPHA: f64 = 0.8;
const DELTA: f64 = 0.7;
/// Mediation: `m_t = A·t_{t-1} + …`, `y_t = C·t_{t-1} + B·m_t + e_t`.
const A: f64 = 0.6;
const B: f64 = 0.5;
const C: f64 = 0.4;
/// Bootstrap replicates per fit.
const BOOT: u32 = 100;
/// Burn-in rows discarded so every lag is populated.
const BURN: usize = 8;

#[derive(Clone, Copy)]
struct Scenario {
    label: &'static str,
    rho: f64,
    n: usize,
    seed: u64,
}

const IID_160: Scenario = Scenario { label: "iid n=160", rho: 0.0, n: 160, seed: 100_000 };
const AR05_160: Scenario =
    Scenario { label: "AR(1) rho=0.5 n=160", rho: 0.5, n: 160, seed: 200_000 };
const AR09_400: Scenario =
    Scenario { label: "AR(1) rho=0.9 n=400", rho: 0.9, n: 400, seed: 300_000 };
const AR05_60: Scenario = Scenario { label: "AR(1) rho=0.5 n=60", rho: 0.5, n: 60, seed: 400_000 };

/// `(z, x)`: iid `z`, and the persistent treatment driven by it.
fn driven_treatment(total: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut g = gaussian(seed);
    let z: Vec<f64> = (0..total).map(|_| g()).collect();
    let mut x = vec![0.0; total];
    for t in 0..total {
        let driven: f64 =
            W.iter().enumerate().filter(|(k, _)| *k <= t).map(|(k, w)| w * z[t - k]).sum();
        x[t] = driven + U_SD * g();
    }
    (z, x)
}

/// One-step (`two_step = false`): `y_t = BETA·x_{t-1} + e_t`.
/// Two-step: `w_t = ALPHA·x_{t-1} + 0.6·ν_t`, `y_t = DELTA·w_{t-1} + e_t`.
/// `e` is AR(1)(`rho`) with SD 1. Columns: `x, y, z, w`.
fn pulse_series(s: Scenario, rep: u32, two_step: bool) -> TimeSeriesData {
    let seed = s.seed + u64::from(rep);
    let total = s.n + BURN;
    let (z, x) = driven_treatment(total, seed.wrapping_mul(7919));
    let e = ar1_noise(total, s.rho, 1.0, seed.wrapping_mul(104_729) ^ 0x5A5A);
    let mut g = gaussian(seed.wrapping_mul(15_485_863) ^ 0x3333);
    let mut w = vec![0.0; total];
    let mut y = vec![0.0; total];
    for t in 1..total {
        w[t] = ALPHA * x[t - 1] + 0.6 * g();
        y[t] = if two_step { DELTA * w[t - 1] } else { BETA * x[t - 1] } + e[t];
    }
    TimeSeriesData::from_f64_columns(
        [("x", &x[BURN..]), ("y", &y[BURN..]), ("z", &z[BURN..]), ("w", &w[BURN..])],
        1,
    )
    .unwrap()
}

/// `m_t = A·t_{t-1} + 0.5·ε`, `y_t = C·t_{t-1} + B·m_t + e_t`, `e` AR(1)(`rho`) with SD 0.5.
fn mediation_series(s: Scenario, rep: u32) -> TimeSeriesData {
    let seed = s.seed + u64::from(rep);
    let total = s.n + BURN;
    let (z, t) = driven_treatment(total, seed.wrapping_mul(7919) ^ 0x1111);
    let mut g = gaussian(seed.wrapping_mul(15_485_863));
    let e = ar1_noise(total, s.rho, 0.5, seed.wrapping_mul(104_729) ^ 0x2222);
    let mut m = vec![0.0; total];
    let mut y = vec![0.0; total];
    for i in 1..total {
        m[i] = A * t[i - 1] + 0.5 * g();
        y[i] = C * t[i - 1] + B * m[i] + e[i];
    }
    TimeSeriesData::from_f64_columns(
        [("t", &t[BURN..]), ("m", &m[BURN..]), ("y", &y[BURN..]), ("z", &z[BURN..])],
        1,
    )
    .unwrap()
}

/// Attach `z_{t-k} → x_t` for every driver lag.
fn add_driver(g: &mut TemporalDag, x: VariableId, z: VariableId) {
    let x0 = ensure_lagged(g, x, Lag::CONTEMPORANEOUS).unwrap();
    for k in 0..W.len() {
        let zk = ensure_lagged(g, z, Lag::from_raw(k as u32)).unwrap();
        g.insert_directed(zk, x0).unwrap();
    }
}

/// `z_{t-k} → x_t`, then `x_{t-1} → y_t` (one-step) or
/// `x_{t-1} → w_t`, `w_{t-1} → y_t` (two-step).
fn pulse_dag(two_step: bool) -> TemporalDag {
    let mut g = TemporalDag::empty();
    add_driver(&mut g, VariableId::from_raw(0), VariableId::from_raw(2));
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    if two_step {
        let w0 = ensure_lagged(&mut g, VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
        let w1 = ensure_lagged(&mut g, VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
        g.insert_directed(x1, w0).unwrap();
        g.insert_directed(w1, y0).unwrap();
    } else {
        g.insert_directed(x1, y0).unwrap();
    }
    g
}

/// `z_{t-k} → t_t`, `t_{t-1} → m_t`, `t_{t-1} → y_t`, `m_t → y_t`.
fn mediation_dag() -> TemporalDag {
    let mut g = TemporalDag::empty();
    add_driver(&mut g, VariableId::from_raw(0), VariableId::from_raw(3));
    let t1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(t1, m0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(m0, y0).unwrap();
    g
}

fn pulse(horizon: u32) -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(horizon)
}

fn single_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -1, 1.0)
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1)
}

fn mediation_query(contrast: MediationContrast) -> CausalQuery {
    CausalQuery::Mediation(
        MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            contrast,
        )
        .with_horizons(vec![1])
        .unwrap(),
    )
}

fn run(
    data: TimeSeriesData,
    graph: TemporalDag,
    query: CausalQuery,
    boot: u32,
    seed: u64,
) -> StudyResult {
    Study::series(data)
        .graph(graph)
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(boot)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap()
}

fn has_diagnostic(result: &StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|d| d.code.as_ref() == code)
}

const SHORT_SERIES: &str = "estimate.temporal.circular_block_se.short_series";

/// Coverage of the headline circular-block interval `ate ± z·se_bootstrap`,
/// plus how many replicates carried the short-series warning.
///
/// `seed_offset` separates otherwise identical designs (Pulse h=1 and
/// single-step Sustained fit the same regression) so each gate is independent
/// evidence.
fn effect_coverage(
    s: Scenario,
    what: &str,
    query: &TemporalEffectQuery,
    truth: f64,
    seed_offset: u64,
) -> (Vec<CoverageTally>, u32) {
    let s = Scenario { seed: s.seed + seed_offset, ..s };
    let mut boot =
        CoverageTally::new(format!("{what} [{}] circular-block se_bootstrap", s.label), 0.9);
    let two_step = query.horizon_steps >= 2;
    let mut spread = Spread::default();
    let mut warned = 0;
    for rep in 0..n_sim() {
        let result = run(
            pulse_series(s, rep, two_step),
            pulse_dag(two_step),
            CausalQuery::TemporalEffect(query.clone()),
            BOOT,
            s.seed + u64::from(rep),
        );
        let est = &result.estimate;
        assert_eq!(result.estimand.adjustment_set.len(), 0, "{what}: unexpected adjustment");
        assert!(est.se_analytic.is_nan(), "{what}: no analytic SE is calibrated");
        boot.record(normal_interval(est.ate, est.se_bootstrap, Z90), truth);
        spread.push(est.ate, est.se_bootstrap.unwrap_or(f64::NAN));
        warned += u32::from(has_diagnostic(&result, SHORT_SERIES));
    }
    spread.report(&format!("{what} [{}]", s.label), truth);
    (vec![boot], warned)
}

/// In-assumption gate: nominal coverage must hold (the short-series warning may
/// or may not fire; its rate is reported).
fn gate((tallies, warned): (Vec<CoverageTally>, u32)) {
    eprintln!("short_series warnings: {warned}/{}", n_sim());
    assert_all(&tallies.iter().collect::<Vec<_>>());
}

/// Boundary record: the design sits below the effective-sample floor, so the
/// short-series warning must fire on at least three quarters of replicates (the
/// floor is checked against an estimated lag-1 score autocorrelation, so a
/// minority of draws land above it); coverage is measured and reported, not
/// gated.
fn boundary((tallies, warned): (Vec<CoverageTally>, u32)) {
    for tally in &tallies {
        let (lo, hi) = common::calibration::coverage_band(n_sim(), 0.9);
        eprintln!(
            "calibration-boundary {tally:?}: coverage={:.3} band=[{lo:.3}, {hi:.3}] \
             mean_length={:.4} (not gated; short_series warnings {warned}/{})",
            tally.rate(),
            tally.mean_length(),
            n_sim()
        );
    }
    assert!(
        warned * 4 >= n_sim() * 3,
        "boundary design must carry the short-series warning: {warned}/{}",
        n_sim()
    );
}

/// Print every tally's calibration line before failing on any of them.
fn assert_all(tallies: &[&CoverageTally]) {
    let failures: Vec<String> = tallies
        .iter()
        .filter_map(|tally| {
            std::panic::catch_unwind(|| tally.assert()).err().map(|e| {
                e.downcast_ref::<String>().cloned().unwrap_or_else(|| "coverage failure".into())
            })
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// Monte-Carlo SD of the point estimate against the mean reported SEs.
#[derive(Default)]
struct Spread {
    estimates: Vec<f64>,
    se: Vec<f64>,
}

impl Spread {
    fn push(&mut self, estimate: f64, se: f64) {
        self.estimates.push(estimate);
        self.se.push(se);
    }

    fn report(&self, label: &str, truth: f64) {
        let n = self.estimates.len() as f64;
        let mean = self.estimates.iter().sum::<f64>() / n;
        let sd =
            (self.estimates.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let finite: Vec<f64> = self.se.iter().copied().filter(|v| v.is_finite()).collect();
        let mean_se = finite.iter().sum::<f64>() / finite.len().max(1) as f64;
        eprintln!(
            "spread {label}: bias={:+.4} monte_carlo_sd={sd:.4} mean_se={mean_se:.4} ratio={:.3}",
            mean - truth,
            mean_se / sd
        );
    }
}

/// Coverage of all three contrasts from the shared block replicate of one run.
fn mediation_coverage(s: Scenario) -> (Vec<CoverageTally>, u32) {
    let label = s.label;
    let mut total = CoverageTally::new(format!("temporal mediation Total [{label}]"), 0.9);
    let mut direct = CoverageTally::new(format!("temporal mediation Direct [{label}]"), 0.9);
    let mut mediated = CoverageTally::new(format!("temporal mediation Mediated [{label}]"), 0.9);
    let mut spreads = [Spread::default(), Spread::default(), Spread::default()];
    let mut warned = 0;
    for rep in 0..n_sim() {
        let result = run(
            mediation_series(s, rep),
            mediation_dag(),
            mediation_query(MediationContrast::Mediated),
            BOOT,
            s.seed + u64::from(rep),
        );
        let grid = result.mediation_grid.as_ref().expect("mediation grid");
        let slice = &grid.slices[0];
        let TemporalMediationUncertainty::FrequentistBlockBootstrap { block, .. } =
            &slice.uncertainty
        else {
            panic!("temporal mediation must publish shared circular-block SEs");
        };
        let points = &slice.estimate;
        assert!(result.estimate.se_analytic.is_nan(), "iid analytic SE must be withheld");
        warned += u32::from(has_diagnostic(&result, SHORT_SERIES));
        total.record(normal_interval(points.total.unwrap(), block.total, Z90), C + A * B);
        direct.record(normal_interval(points.direct.unwrap(), block.direct, Z90), C);
        mediated.record(normal_interval(points.mediated.unwrap(), block.mediated, Z90), A * B);
        for (spread, (point, se)) in spreads.iter_mut().zip([
            (points.total, block.total),
            (points.direct, block.direct),
            (points.mediated, block.mediated),
        ]) {
            spread.push(point.unwrap(), se.unwrap_or(f64::NAN));
        }
    }
    for (spread, (name, truth)) in
        spreads.iter().zip([("Total", C + A * B), ("Direct", C), ("Mediated", A * B)])
    {
        spread.report(&format!("temporal mediation {name} [{label}]"), truth);
    }
    (vec![total, direct, mediated], warned)
}

macro_rules! effect_gate {
    ($name:ident, $scenario:expr, $what:expr, $query:expr, $truth:expr, $offset:expr) => {
        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn $name() {
            gate(effect_coverage($scenario, $what, &$query, $truth, $offset));
        }
    };
}

macro_rules! mediation_gate {
    ($name:ident, $scenario:expr) => {
        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn $name() {
            gate(mediation_coverage($scenario));
        }
    };
}

effect_gate!(
    temporal_dag_pulse_iid_n160_nominal_90_coverage,
    IID_160,
    "TemporalDag Pulse h=1",
    pulse(1),
    BETA,
    0
);
effect_gate!(
    temporal_dag_pulse_ar05_n160_nominal_90_coverage,
    AR05_160,
    "TemporalDag Pulse h=1",
    pulse(1),
    BETA,
    0
);
effect_gate!(
    temporal_dag_pulse_ar09_n400_nominal_90_coverage,
    AR09_400,
    "TemporalDag Pulse h=1",
    pulse(1),
    BETA,
    0
);
effect_gate!(
    temporal_dag_pulse_ar05_n60_nominal_90_coverage,
    AR05_60,
    "TemporalDag Pulse h=1",
    pulse(1),
    BETA,
    0
);

effect_gate!(
    temporal_dag_pulse_h2_iid_n160_nominal_90_coverage,
    IID_160,
    "TemporalDag Pulse h=2",
    pulse(2),
    ALPHA * DELTA,
    0
);
effect_gate!(
    temporal_dag_pulse_h2_ar05_n160_nominal_90_coverage,
    AR05_160,
    "TemporalDag Pulse h=2",
    pulse(2),
    ALPHA * DELTA,
    0
);
effect_gate!(
    temporal_dag_pulse_h2_ar09_n400_nominal_90_coverage,
    AR09_400,
    "TemporalDag Pulse h=2",
    pulse(2),
    ALPHA * DELTA,
    0
);
effect_gate!(
    temporal_dag_pulse_h2_ar05_n60_nominal_90_coverage,
    AR05_60,
    "TemporalDag Pulse h=2",
    pulse(2),
    ALPHA * DELTA,
    0
);

effect_gate!(
    temporal_dag_sustained_iid_n160_nominal_90_coverage,
    IID_160,
    "TemporalDag single-step Sustained",
    single_sustained(),
    BETA,
    50_000
);
effect_gate!(
    temporal_dag_sustained_ar05_n160_nominal_90_coverage,
    AR05_160,
    "TemporalDag single-step Sustained",
    single_sustained(),
    BETA,
    50_000
);
effect_gate!(
    temporal_dag_sustained_ar09_n400_nominal_90_coverage,
    AR09_400,
    "TemporalDag single-step Sustained",
    single_sustained(),
    BETA,
    50_000
);
effect_gate!(
    temporal_dag_sustained_ar05_n60_nominal_90_coverage,
    AR05_60,
    "TemporalDag single-step Sustained",
    single_sustained(),
    BETA,
    50_000
);

mediation_gate!(temporal_dag_mediation_iid_n160_nominal_90_coverage, IID_160);
mediation_gate!(temporal_dag_mediation_ar05_n160_nominal_90_coverage, AR05_160);
mediation_gate!(temporal_dag_mediation_ar09_n400_nominal_90_coverage, AR09_400);
mediation_gate!(temporal_dag_mediation_ar05_n60_nominal_90_coverage, AR05_60);

/// Below the effective-sample floor (AR(1) ρ = 0.9 at n = 60 and 160): measured,
/// warned, not gated. The in-assumption gates above carry the nominal claim.
const AR09_160: Scenario =
    Scenario { label: "AR(1) rho=0.9 n=160 (boundary)", rho: 0.9, n: 160, seed: 510_000 };
const AR09_60: Scenario =
    Scenario { label: "AR(1) rho=0.9 n=60 (boundary)", rho: 0.9, n: 60, seed: 500_000 };

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_dag_pulse_ar09_n60_short_series_boundary() {
    boundary(effect_coverage(AR09_60, "TemporalDag Pulse h=1", &pulse(1), BETA, 0));
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_dag_mediation_ar09_n60_short_series_boundary() {
    boundary(mediation_coverage(AR09_60));
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_dag_pulse_ar09_n160_short_series_boundary() {
    boundary(effect_coverage(AR09_160, "TemporalDag Pulse h=1", &pulse(1), BETA, 0));
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_dag_mediation_ar09_n160_short_series_boundary() {
    boundary(mediation_coverage(AR09_160));
}

#[test]
fn temporal_dag_pulse_publishes_circular_block_se_only() {
    let result = run(
        pulse_series(IID_160, 7, false),
        pulse_dag(false),
        CausalQuery::TemporalEffect(pulse(1)),
        24,
        7,
    );
    let diag = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.temporal.circular_block_se")
        .expect("the single-window Frequentist path must record its circular-block SE");
    // 159 lag-aligned rows: ⌈159^(1/3)⌉ = 6 beats the two-slice unfolded window.
    assert!(diag.message.contains("circular-block length 6"), "{}", diag.message);
    assert!(diag.message.contains("n=159 lag-aligned rows"), "{}", diag.message);
    let est = &result.estimate;
    assert!(est.se_bootstrap.is_some_and(|se| se.is_finite() && se > 0.0));
    assert!(est.se_analytic.is_nan(), "the iid OLS SE must not be published on lagged rows");
    assert_eq!(est.bootstrap_replicates_ok, Some(24));
    assert!(!has_diagnostic(&result, "estimate.temporal.circular_block_se.short_series"));

    // Without replicates there is no calibrated SE, and the result says so.
    let bare = run(
        pulse_series(IID_160, 7, false),
        pulse_dag(false),
        CausalQuery::TemporalEffect(pulse(1)),
        0,
        7,
    );
    assert!(bare.estimate.se_bootstrap.is_none());
    assert!(bare.estimate.se_analytic.is_nan());
    assert!((bare.estimate.ate - est.ate).abs() < 1e-12);
    let bare_diag = bare
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.temporal.circular_block_se")
        .unwrap();
    assert!(bare_diag.message.contains("request bootstrap_replicates"), "{}", bare_diag.message);
}

#[test]
fn temporal_mediation_shares_one_block_replicate_across_contrasts() {
    let result = run(
        mediation_series(AR05_160, 3),
        mediation_dag(),
        mediation_query(MediationContrast::Total),
        24,
        3,
    );
    assert!(has_diagnostic(&result, "estimate.temporal.circular_block_se"));
    assert!(result.estimate.se_analytic.is_nan(), "iid Total SE must be withheld by default");
    let slice = &result.mediation_grid.as_ref().unwrap().slices[0];
    let TemporalMediationUncertainty::FrequentistBlockBootstrap { requested, block } =
        &slice.uncertainty
    else {
        panic!("expected shared circular-block uncertainty, got {:?}", slice.uncertainty);
    };
    assert_eq!(*requested, block.total);
    assert_eq!(result.estimate.se_bootstrap, block.total);
    assert_eq!(block.replicates_attempted, 24);
    for se in [block.total, block.direct, block.mediated] {
        assert!(se.is_some_and(|se| se.is_finite() && se > 0.0));
    }

    // No replicates: every iid analytic SE stays NaN and no interval is claimed.
    let none = run(
        mediation_series(AR05_160, 3),
        mediation_dag(),
        mediation_query(MediationContrast::Direct),
        0,
        3,
    );
    assert!(none.estimate.se_analytic.is_nan());
    assert!(none.estimate.se_bootstrap.is_none());
    assert!(matches!(
        none.mediation_grid.as_ref().unwrap().slices[0].uncertainty,
        TemporalMediationUncertainty::FrequentistPointwise { standard_error: None }
    ));
}

#[test]
fn short_series_warns_below_the_block_floor() {
    let short = Scenario { label: "short", rho: 0.5, n: 30, seed: 9 };
    let result = run(
        pulse_series(short, 0, false),
        pulse_dag(false),
        CausalQuery::TemporalEffect(pulse(1)),
        8,
        9,
    );
    let warning = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.temporal.circular_block_se.short_series")
        .expect("a 30-row series must carry the short-series warning");
    assert_eq!(warning.severity, DiagnosticSeverity::Warning);
}
