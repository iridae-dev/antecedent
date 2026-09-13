//! 1.9 coverage of Bayesian temporal Pulse / Sustained credible intervals (R-9).
//!
//! Every replicate runs the staged `Study` path. The treatment and the outcome
//! residual are independent stationary AR(1) processes
//! (`common::calibration::ar1_noise`). The treatment's persistence is latent (the
//! temporal identifier certifies no autoregressive treatment edge), and the
//! residual autocorrelation is latent too, so every DGP sits inside the stated
//! identification assumptions. With a persistent treatment the score `x̃_t e_t`
//! inherits the residual autocorrelation — the case an iid
//! Normal–Inverse-Gamma likelihood understates. The iid-residual regime keeps the
//! persistent treatment, so it checks that the dependence correction does not
//! over-widen when the iid likelihood is right.
//!
//! Ignored tests run via `scripts/gate_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::too_many_lines, clippy::many_single_char_names)]

mod common;

use antecedent::{BayesianConfig, ClassPrior, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalCpdag, TemporalDag, ensure_lagged};
use common::calibration::{CoverageTally, ar1_noise, n_sim, quantile_interval};

/// Effect of `x_{t-1}` on `y_t`.
const BETA1: f64 = 0.8;
/// Effect of `x_{t-2}` on `y_t` in the multi-step DGP (nonzero second lag).
const BETA2: f64 = 0.4;
const DRAWS: usize = 400;
const LEVEL: f64 = 0.9;
/// Treatment persistence used when the outcome residual is iid.
const IID_TREATMENT_RHO: f64 = 0.5;

/// Noise regime: outcome-residual AR(1) coefficient and series length.
#[derive(Clone, Copy, Debug)]
struct Regime {
    rho: f64,
    n: usize,
    seed: u64,
}

impl Regime {
    const IID: Self = Self { rho: 0.0, n: 160, seed: 1_000_000 };
    const RHO05_N160: Self = Self { rho: 0.5, n: 160, seed: 2_000_000 };
    const RHO09_N400: Self = Self { rho: 0.9, n: 400, seed: 3_000_000 };
    const RHO05_N60: Self = Self { rho: 0.5, n: 60, seed: 4_000_000 };

    fn treatment_rho(self) -> f64 {
        if self.rho == 0.0 { IID_TREATMENT_RHO } else { self.rho }
    }

    fn label(self) -> String {
        if self.rho == 0.0 {
            format!("iid residual (treatment rho {IID_TREATMENT_RHO}) n={}", self.n)
        } else {
            format!("AR(1) rho={} n={}", self.rho, self.n)
        }
    }

    /// Stream seed for `rep`. The harness generator uses `seed | 1`, so streams
    /// must differ above bit 0: offsets are odd and ten apart per replicate.
    fn stream(self, rep: u64, k: u64) -> u64 {
        self.seed + 10 * rep + 2 * k + 1
    }
}

/// `y_t = 0.8 x_{t-1} + beta2 x_{t-2} + e_t`, `x` and `e` independent AR(1).
fn series_xy(regime: Regime, rep: u64, beta2: f64) -> TimeSeriesData {
    let x = ar1_noise(regime.n, regime.treatment_rho(), 0.5, regime.stream(rep, 0));
    let e = ar1_noise(regime.n, regime.rho, 0.35, regime.stream(rep, 1));
    let mut y = vec![0.0; regime.n];
    for t in 2..regime.n {
        y[t] = BETA1 * x[t - 1] + beta2 * x[t - 2] + e[t];
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

/// `x_{t-1} → y_t` (Pulse / single-step Sustained DGP has `beta2 = 0`).
fn lag1_dag() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g
}

/// `x_{t-1} → y_t`, `x_{t-2} → y_t` (multi-step DGP).
fn lag12_dag() -> TemporalDag {
    let mut g = lag1_dag();
    let x2 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x2, y0).unwrap();
    g
}

fn pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

fn single_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -1, 1.0)
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1)
}

fn multi_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -2, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1))
        .with_horizon_steps(1)
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(10.0))
}

fn run(
    data: TimeSeriesData,
    graph: impl Into<antecedent::AcceptedGraph>,
    query: TemporalEffectQuery,
    prior: Option<ClassPrior>,
    seed: u64,
) -> StudyResult {
    let mut builder = Study::series(data)
        .graph(graph.into())
        .query(CausalQuery::TemporalEffect(query))
        .inference(bayes());
    if let Some(prior) = prior {
        builder = builder.class_prior(prior);
    }
    builder
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap()
}

fn credible_interval(result: &StudyResult) -> Option<(f64, f64)> {
    let post = result.posterior.as_ref()?;
    let draws = post.draws.column(post.effect_column()?).ok()?;
    quantile_interval(draws, LEVEL)
}

#[derive(Clone, Copy, Debug)]
enum Cell {
    Pulse,
    SingleSustained,
    MultiSustained,
}

impl Cell {
    const fn name(self) -> &'static str {
        match self {
            Self::Pulse => "Pulse",
            Self::SingleSustained => "single-step Sustained",
            Self::MultiSustained => "multi-step Sustained",
        }
    }

    fn query(self) -> TemporalEffectQuery {
        match self {
            Self::Pulse => pulse(),
            Self::SingleSustained => single_sustained(),
            Self::MultiSustained => multi_sustained(),
        }
    }

    const fn beta2(self) -> f64 {
        match self {
            Self::MultiSustained => BETA2,
            Self::Pulse | Self::SingleSustained => 0.0,
        }
    }

    /// True causal contrast of a unit (sustained) treatment change.
    const fn truth(self) -> f64 {
        BETA1 + self.beta2()
    }

    fn graph(self) -> TemporalDag {
        match self {
            Self::MultiSustained => lag12_dag(),
            Self::Pulse | Self::SingleSustained => lag1_dag(),
        }
    }
}

fn coverage(cell: Cell, regime: Regime) {
    let mut tally = CoverageTally::new(
        format!("Bayesian TemporalDag {} {}", cell.name(), regime.label()),
        LEVEL,
    );
    for rep in 0..u64::from(n_sim()) {
        let data = series_xy(regime, rep, cell.beta2());
        let result = run(data, cell.graph(), cell.query(), None, 7 + rep);
        tally.record(credible_interval(&result), cell.truth());
    }
    tally.assert();
}

macro_rules! dag_coverage {
    ($($test:ident => ($cell:expr, $regime:expr);)*) => {$(
        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn $test() {
            coverage($cell, $regime);
        }
    )*};
}

dag_coverage! {
    bayesian_temporal_pulse_iid_nominal_90_coverage => (Cell::Pulse, Regime::IID);
    bayesian_temporal_pulse_ar1_rho05_n160_nominal_90_coverage =>
        (Cell::Pulse, Regime::RHO05_N160);
    bayesian_temporal_pulse_ar1_rho09_n400_nominal_90_coverage =>
        (Cell::Pulse, Regime::RHO09_N400);
    bayesian_temporal_pulse_ar1_rho05_n60_nominal_90_coverage =>
        (Cell::Pulse, Regime::RHO05_N60);
    bayesian_temporal_sustained_single_iid_nominal_90_coverage =>
        (Cell::SingleSustained, Regime::IID);
    bayesian_temporal_sustained_single_ar1_rho05_n160_nominal_90_coverage =>
        (Cell::SingleSustained, Regime::RHO05_N160);
    bayesian_temporal_sustained_single_ar1_rho09_n400_nominal_90_coverage =>
        (Cell::SingleSustained, Regime::RHO09_N400);
    bayesian_temporal_sustained_single_ar1_rho05_n60_nominal_90_coverage =>
        (Cell::SingleSustained, Regime::RHO05_N60);
    bayesian_temporal_sustained_multi_iid_nominal_90_coverage =>
        (Cell::MultiSustained, Regime::IID);
    bayesian_temporal_sustained_multi_ar1_rho05_n160_nominal_90_coverage =>
        (Cell::MultiSustained, Regime::RHO05_N160);
    bayesian_temporal_sustained_multi_ar1_rho09_n400_nominal_90_coverage =>
        (Cell::MultiSustained, Regime::RHO09_N400);
    bayesian_temporal_sustained_multi_ar1_rho05_n60_nominal_90_coverage =>
        (Cell::MultiSustained, Regime::RHO05_N60);
}

/// `t_t = 0.4 z_t + 0.5 u_t`, `y_t = 0.8 t_{t-1} + e_t`; `z`, `u`, `e` independent AR(1).
///
/// `z_{t-1} — t_{t-1}` is undirected in the class. Both completions target the
/// same effect (z has no path to y), so the class-prior mixture functional is
/// `0.8`; the completions differ in their adjustment set and posterior width.
fn series_tyz(regime: Regime, rep: u64) -> TimeSeriesData {
    let z = ar1_noise(regime.n, regime.rho, 1.0, regime.stream(rep, 0));
    let u = ar1_noise(regime.n, regime.rho, 1.0, regime.stream(rep, 1));
    let e = ar1_noise(regime.n, regime.rho, 0.35, regime.stream(rep, 2));
    let t: Vec<f64> = z.iter().zip(&u).map(|(z, u)| 0.4 * z + 0.5 * u).collect();
    let mut y = vec![0.0; regime.n];
    for i in 1..regime.n {
        y[i] = BETA1 * t[i - 1] + e[i];
    }
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

fn tyz_cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_undirected(z1, t1).unwrap();
    g
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_cpdag_class_prior_ar1_rho05_n160_nominal_90_coverage() {
    let regime = Regime::RHO05_N160;
    let prior = ClassPrior::from_ordered([0.5, 0.5]).unwrap();
    let mut tally = CoverageTally::new(
        format!("Bayesian TemporalCpdag Pulse class-prior mixture {}", regime.label()),
        LEVEL,
    );
    for rep in 0..u64::from(n_sim()) {
        let result =
            run(series_tyz(regime, rep), tyz_cpdag(), pulse(), Some(prior.clone()), 11 + rep);
        tally.record(credible_interval(&result), BETA1);
    }
    tally.assert();
}

/// The composed Bayesian refitter resamples lag-aligned rows for
/// `bootstrap.ci_coverage` instead of rebuilding lags on a block-resampled raw
/// series (which pairs outcomes with regressors from unrelated blocks).
#[test]
fn bayesian_multi_step_bootstrap_refuter_resamples_aligned_rows() {
    let result = Study::series(series_xy(Regime::RHO05_N160, 0, BETA2))
        .graph(lag12_dag())
        .query(CausalQuery::TemporalEffect(multi_sustained()))
        .inference(bayes())
        .refute(RefuteSuite::Full)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    let report = result
        .refutations
        .iter()
        .find(|r| r.refuter.as_ref() == "bootstrap.ci_coverage")
        .expect("the composed Bayesian contrast runs bootstrap.ci_coverage");
    assert!(report.replicates > 0 && report.refuted_ate.is_finite(), "{report:?}");
    assert!(
        !result.diagnostics.iter().any(|d| d.code.as_ref() == "refute.validator.not_applicable"
            && d.message.contains("bootstrap")),
        "bootstrap.ci_coverage must not be skipped"
    );
}
