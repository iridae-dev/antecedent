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

#![allow(dead_code, clippy::many_single_char_names)]

use antecedent_core::{Lag, TemporalEffectQuery, TemporalPolicy, VariableId};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

use super::calibration::{ar1_noise, gaussian};
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
    let t = ar1_noise(total, rho, 1.0, seed ^ 0x7A11);
    let e = ar1_noise(total, rho, 0.5, seed ^ 0x7A12);
    let mut g = gaussian(seed ^ 0x7A13);
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
    let x = ar1_noise(total, rho, 1.0, seed ^ 0x5E01);
    let u = ar1_noise(total, rho, 0.5, seed ^ 0x5E02);
    let mut y = vec![0.0; total];
    for t in 2..total {
        y[t] = B1 * x[t - 1] + B2 * x[t - 2] + u[t];
    }
    TimeSeriesData::from_f64_columns([("x", &x[BURN..]), ("y", &y[BURN..])], 1).unwrap()
}
