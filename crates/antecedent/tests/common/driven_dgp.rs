//! Single-`TemporalDag` calibration DGPs with a persistent, driven treatment.
//!
//! The treatment is persistent through an observed driver `z`
//! (`x_t = Σ_k W[k]·z_{t-k} + u_t`, a graph-certified MA(3)), so the regression
//! scores `x_{t-h}·e_t` are serially correlated whenever the residual is: an iid
//! SE or iid row bootstrap under-covers on these DGPs. No lagged outcome enters
//! the adjustment set, so the residual autocorrelation is not absorbed. (A
//! treatment self-loop would make persistence explicit, but unfolding cannot
//! certify a self-looped treatment over a finite window, and only a single-step
//! Pulse is then identified, by parent adjustment.)
//!
//! Truths are the population values of the reported estimands (linear-Gaussian):
//! Pulse h=1 and single-step Sustained `BETA` (`y_t = BETA·x_{t-1} + e_t`);
//! Pulse h=2 `ALPHA·DELTA`, propagated through an intermediate
//! `w_t = ALPHA·x_{t-1} + ν_t` into `y_t = DELTA·w_{t-1} + e_t`, so the h=2
//! regression residual `DELTA·ν_{t-1} + e_t` carries the intermediate shock;
//! mediation Total `C + A·B`, Direct `C`, Mediated `A·B`. Every graph yields a
//! single temporal-backdoor estimand with an empty adjustment set.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use antecedent_core::{
    CausalQuery, Lag, MediationContrast, MediationQuery, TemporalEffectQuery, TemporalPolicy,
    VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{ensure_lagged, TemporalDag};

use super::calibration::{ar1_noise, gaussian};

/// Driver weights: `x_t = Σ_k W[k]·z_{t-k} + U_SD·u_t` (lag-1 autocorrelation ≈ 0.5).
pub const W: [f64; 4] = [0.75, 0.375, 0.1875, 0.094];
pub const U_SD: f64 = 0.35;
/// Effect of `x_{t-1}` on `y_t` (one-step design).
pub const BETA: f64 = 0.8;
/// Two-step design: `x_{t-1} → w_t` and `w_{t-1} → y_t`.
pub const ALPHA: f64 = 0.8;
pub const DELTA: f64 = 0.7;
/// Mediation: `m_t = A·t_{t-1} + …`, `y_t = C·t_{t-1} + B·m_t + e_t`.
pub const A: f64 = 0.6;
pub const B: f64 = 0.5;
pub const C: f64 = 0.4;
/// Burn-in rows discarded so every lag is populated.
const BURN: usize = 8;

/// AR(1) persistence, length and seed base of one calibration design.
#[derive(Clone, Copy, Debug)]
pub struct Scenario {
    pub label: &'static str,
    pub rho: f64,
    pub n: usize,
    pub seed: u64,
}

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
#[must_use]
pub fn pulse_series(s: Scenario, rep: u32, two_step: bool) -> TimeSeriesData {
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
#[must_use]
pub fn mediation_series(s: Scenario, rep: u32) -> TimeSeriesData {
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
#[must_use]
pub fn pulse_dag(two_step: bool) -> TemporalDag {
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
#[must_use]
pub fn mediation_dag() -> TemporalDag {
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

#[must_use]
pub fn pulse(horizon: u32) -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(horizon)
}

#[must_use]
pub fn single_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -1, 1.0)
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1)
}

#[must_use]
pub fn mediation_query(contrast: MediationContrast) -> CausalQuery {
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
