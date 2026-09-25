//! Single-`TemporalDag` calibration DGPs with an AR(1)-persistent treatment.
//!
//! Unlike `common::driven_dgp` (an MA(3) treatment, whose regression score
//! forgets within a few lags whatever the residual persistence), the treatment
//! here is AR(1)(`ρ`) along with the residual, so the estimating score
//! `x̃_{t-1}·e_t` is itself close to AR(1)(`ρ²`): its memory grows without bound
//! as `ρ → 1`. These are the designs where the circular-block interval runs out
//! of series first (`v19_short_series_measurement`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code)]

use antecedent_core::{Lag, TemporalEffectQuery, TemporalPolicy, VariableId};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{ensure_lagged, TemporalDag};

use super::calibration::{ar1_noise, gaussian, stream_seed};
use super::driven_dgp::{A, B, C};
use super::fixtures::{B1, B2};

/// Burn-in rows discarded so every lag is populated.
const BURN: usize = 8;

fn lagged(g: &mut TemporalDag, variable: u32, lag: u32) -> antecedent_graph::DenseNodeId {
    ensure_lagged(g, VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap()
}

/// Single-window design on [`super::fixtures::chain_pag_series`] (`t, y, z, m, v`)
/// with `m` and `v` marginalized: `z@1 → t@1`, `t@1 → y`, `z@1 → y` (through
/// `m`). The Pulse adjusts `z[t-1]` and recovers `B1`; the residualized
/// treatment and the residual `CHAIN_D·e_m + u` are both AR(1)(`ρ`).
#[must_use]
pub fn chain_pulse_dag() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let t1 = lagged(&mut g, 0, 1);
    let y0 = lagged(&mut g, 1, 0);
    let z1 = lagged(&mut g, 2, 1);
    g.insert_directed(z1, t1).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g
}

/// Pulse of variable 0 on variable 1 at lag 1, horizon 1.
#[must_use]
pub fn pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

/// Sustained intervention on variable 0 over lags 2..=1, outcome variable 1.
#[must_use]
pub fn multi_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -2, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1))
        .with_horizon_steps(1)
}

/// Mediation with an AR(1)(`ρ`) treatment: `m_t = A·t_{t-1} + 0.5·ε_t`,
/// `y_t = C·t_{t-1} + B·m_t + e_t`, `e` AR(1)(`ρ`) with SD 0.5 (the constants of
/// `common::driven_dgp`, so Total `C + A·B`, Direct `C`, Mediated `A·B`).
/// Columns `t, m, y`; query `common::driven_dgp::mediation_query`.
#[must_use]
pub fn mediation_series(n: usize, rho: f64, seed: u64) -> TimeSeriesData {
    let total = n + BURN;
    let t = ar1_noise(total, rho, 1.0, stream_seed(seed, 0x7A11));
    let e = ar1_noise(total, rho, 0.5, stream_seed(seed, 0x7A12));
    let mut g = gaussian(stream_seed(seed, 0x7A13));
    let mut m = vec![0.0; total];
    let mut y = vec![0.0; total];
    for i in 1..total {
        m[i] = A * t[i - 1] + 0.5 * g();
        y[i] = C * t[i - 1] + B * m[i] + e[i];
    }
    TimeSeriesData::from_f64_columns([("t", &t[BURN..]), ("m", &m[BURN..]), ("y", &y[BURN..])], 1)
        .unwrap()
}

/// `t@1 → m`, `t@1 → y`, `m → y`.
#[must_use]
pub fn mediation_dag() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let (t1, m0, y0) = (lagged(&mut g, 0, 1), lagged(&mut g, 1, 0), lagged(&mut g, 2, 0));
    g.insert_directed(t1, m0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(m0, y0).unwrap();
    g
}

/// Multi-step Sustained design on `super::fixtures::two_lag_dag`:
/// `y_t = B1·x_{t-1} + B2·x_{t-2} + u_t`, `x` and `u` AR(1)(`ρ`); truth `B1 + B2`.
#[must_use]
pub fn sequential_series(n: usize, rho: f64, seed: u64) -> TimeSeriesData {
    let total = n + BURN;
    let x = ar1_noise(total, rho, 1.0, stream_seed(seed, 0x5E01));
    let u = ar1_noise(total, rho, 0.5, stream_seed(seed, 0x5E02));
    let mut y = vec![0.0; total];
    for t in 2..total {
        y[t] = B1 * x[t - 1] + B2 * x[t - 2] + u[t];
    }
    TimeSeriesData::from_f64_columns([("x", &x[BURN..]), ("y", &y[BURN..])], 1).unwrap()
}

/// Autoregressive treatment persistence `t_{t-1} → t_t` in [`autoregressive_pulse_series`].
pub const AR_PHI: f64 = 0.6;
/// Confounder loading on the treatment, `z_{t-1} → t_t`.
pub const AR_GAMMA: f64 = 0.8;
/// Confounder loading on the outcome, `z_{t-2} → y_t` (confounds `t_{t-1} → y_t`).
pub const AR_KAPPA: f64 = 0.9;
/// Pulse effect of `t_{t-1}` on `y_t`: the truth of the lag-one, horizon-one pulse.
pub const AR_BETA: f64 = 0.8;
/// Loading of the treatment-only driver `w_{t-1} → t_t` (`with_driver`).
pub const AR_OMEGA: f64 = 0.7;

/// Pulse design whose treatment is autoregressive and confounded:
/// `t_t = phi·t_{t-1} + AR_GAMMA·z_{t-1} [+ AR_OMEGA·w_{t-1}] + 0.5·u_t`,
/// `y_t = AR_BETA·t_{t-1} + AR_KAPPA·z_{t-2} + 0.5·e_t`, with `z`, `w`, `u`, `e`
/// iid standard normal. The lag-one pulse of `t` on `y` is `AR_BETA`; `z_{t-2}`
/// confounds it. `phi = 0` removes the autoregression. Columns `t, y, z`
/// (then `w` when `with_driver`); graph [`autoregressive_pulse_dag`].
#[must_use]
pub fn autoregressive_pulse_series(
    n: usize,
    phi: f64,
    with_driver: bool,
    seed: u64,
) -> TimeSeriesData {
    let total = n + BURN;
    let mut draw = gaussian(stream_seed(seed, 0xA701));
    let z: Vec<f64> = (0..total).map(|_| draw()).collect();
    let w: Vec<f64> = (0..total).map(|_| if with_driver { draw() } else { 0.0 }).collect();
    let mut t = vec![0.0; total];
    let mut y = vec![0.0; total];
    for i in 2..total {
        t[i] = phi * t[i - 1] + AR_GAMMA * z[i - 1] + AR_OMEGA * w[i - 1] + 0.5 * draw();
        y[i] = AR_BETA * t[i - 1] + AR_KAPPA * z[i - 2] + 0.5 * draw();
    }
    if with_driver {
        TimeSeriesData::from_f64_columns(
            [("t", &t[BURN..]), ("y", &y[BURN..]), ("z", &z[BURN..]), ("w", &w[BURN..])],
            1,
        )
        .unwrap()
    } else {
        TimeSeriesData::from_f64_columns(
            [("t", &t[BURN..]), ("y", &y[BURN..]), ("z", &z[BURN..])],
            1,
        )
        .unwrap()
    }
}

/// `z@1 → t@0`, `z@2 → y@0`, `t@1 → y@0`, plus `t@1 → t@0` when
/// `autoregressive` and `w@1 → t@0` when `with_driver` (variables `t = 0`,
/// `y = 1`, `z = 2`, `w = 3`).
#[must_use]
pub fn autoregressive_pulse_dag(autoregressive: bool, with_driver: bool) -> TemporalDag {
    let mut g = TemporalDag::empty();
    let t1 = lagged(&mut g, 0, 1);
    let t0 = lagged(&mut g, 0, 0);
    let y0 = lagged(&mut g, 1, 0);
    let z1 = lagged(&mut g, 2, 1);
    let z2 = lagged(&mut g, 2, 2);
    g.insert_directed(z1, t0).unwrap();
    g.insert_directed(z2, y0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    if autoregressive {
        g.insert_directed(t1, t0).unwrap();
    }
    if with_driver {
        let w1 = lagged(&mut g, 3, 1);
        g.insert_directed(w1, t0).unwrap();
    }
    g
}
