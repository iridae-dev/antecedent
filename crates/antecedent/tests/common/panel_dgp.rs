//! Panel DGPs for the panel-route regression and calibration tests.
//!
//! Every unit is a series over `(t, y, z)`:
//!
//! ```text
//! z_t ~ N(0, 1)
//! t_t = φ·t_{t-1} + 0.4·z_t + u_t            u_t ~ N(0, 1)
//! y_t = 1 + β·t_{t-1} + γ·z_{t-1} + e_t      e_t = ρ·e_{t-1} + ε_t
//! ```
//!
//! With `γ = 0` the lag-1 DAG `t@1 → y` is correct and the Pulse effect of a
//! unit step at `t@1` on `y` is `β`. [`confounded_unit`] makes `z@1` a
//! confounder of `t@1 → y` so the two completions of [`cpdag_two`] disagree.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code)]

use std::sync::Arc;

use antecedent_core::{
    ContinuousDomain, GridSpec, Lag, ResponseFunctional, ResponseQuery, TemporalEffectQuery,
    TemporalPolicy, TemporalResponseSpec, VariableId,
};
use antecedent_data::{PanelData, PanelUnit, TimeSeriesData};
use antecedent_graph::{ensure_lagged, TemporalCpdag, TemporalDag};

use super::calibration::gaussian;

/// Treatment column.
pub const T: VariableId = VariableId::from_raw(0);
/// Outcome column.
pub const Y: VariableId = VariableId::from_raw(1);
/// Covariate column.
pub const Z: VariableId = VariableId::from_raw(2);

/// Unit-series parameters.
#[derive(Clone, Copy, Debug)]
pub struct UnitSpec {
    /// Rows.
    pub n: usize,
    /// Lag-1 treatment effect.
    pub beta: f64,
    /// Lag-1 covariate effect on the outcome.
    pub gz: f64,
    /// Treatment persistence.
    pub phi: f64,
    /// Outcome-noise persistence.
    pub rho: f64,
}

impl UnitSpec {
    /// iid noise, non-persistent treatment, correct lag-1 DAG.
    #[must_use]
    pub const fn iid(n: usize, beta: f64) -> Self {
        Self { n, beta, gz: 0.0, phi: 0.0, rho: 0.0 }
    }
}

/// Regularly sampled series from named columns (interval 1 ns).
#[must_use]
pub fn series(t: &[f64], y: &[f64], z: &[f64]) -> TimeSeriesData {
    series_with_interval(t, y, z, 1)
}

/// Regularly sampled series with an explicit sampling interval.
#[must_use]
pub fn series_with_interval(t: &[f64], y: &[f64], z: &[f64], interval_ns: u64) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns([("t", t), ("y", y), ("z", z)], interval_ns).unwrap()
}

/// One unit of the module DGP, keyed by `seed`.
#[must_use]
pub fn unit_series(spec: UnitSpec, seed: u64) -> TimeSeriesData {
    let mut draw = gaussian(seed);
    let UnitSpec { n, beta, gz, phi, rho } = spec;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    // Start both persistent processes at their stationary laws (the treatment
    // innovation 0.4·z + u has variance 1.16).
    let mut prev_t = draw() * (1.16 / (1.0 - phi * phi)).sqrt();
    let mut e = draw() / (1.0 - rho * rho).sqrt();
    for i in 0..n {
        z[i] = draw();
        t[i] = phi * prev_t + 0.4 * z[i] + draw();
        e = rho * e + draw();
        y[i] = if i > 0 { 1.0 + beta * t[i - 1] + gz * z[i - 1] + e } else { 1.0 + e };
        prev_t = t[i];
    }
    series(&t, &y, &z)
}

/// A unit where `z@1` confounds `t@1 → y`: `t_t = 0.9 z_t + u_t`,
/// `y_t = 1 + 2 t_{t-1} + 1.5 z_{t-1} + e_t`.
#[must_use]
pub fn confounded_unit(n: usize, seed: u64) -> TimeSeriesData {
    let mut draw = gaussian(seed);
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        z[i] = draw();
        t[i] = 0.9 * z[i] + draw();
        if i > 0 {
            y[i] = 1.0 + 2.0 * t[i - 1] + 1.5 * z[i - 1] + draw();
        }
    }
    series(&t, &y, &z)
}

/// A unit whose treatment stays within about ±0.3 (`t_t = 0.1·u_t`), so a dose of 1
/// lies outside its lag-aligned treatment range; `y_t = 1 + 0.8 t_{t-1} + e_t`.
#[must_use]
pub fn narrow_treatment_unit(n: usize, seed: u64) -> TimeSeriesData {
    let mut draw = gaussian(seed);
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        z[i] = draw();
        t[i] = 0.1 * draw();
        y[i] = if i > 0 { 1.0 + 0.8 * t[i - 1] + draw() } else { 1.0 };
    }
    series(&t, &y, &z)
}

/// Panel with unit ids `0..units.len()`.
#[must_use]
pub fn panel(units: Vec<TimeSeriesData>) -> PanelData {
    PanelData::try_new(Arc::from(
        units
            .into_iter()
            .enumerate()
            .map(|(i, series)| PanelUnit { unit_id: u32::try_from(i).unwrap(), series })
            .collect::<Vec<_>>(),
    ))
    .unwrap()
}

/// `t@1 → y`.
#[must_use]
pub fn lagged_ty_dag() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let t1 = ensure_lagged(&mut g, T, Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, Y, Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g
}

/// `z@1 → y`, `t@1 → y`, `z@1 — t@1`: two completions, one adjusting for `z@1`.
#[must_use]
pub fn cpdag_two() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(T, Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(Y, Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(Z, Lag::from_raw(1)).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_undirected(z1, t1).unwrap();
    g
}

/// Pulse of a unit step at `t@1` on `y`, horizon 1, history lag 1.
#[must_use]
pub fn pulse_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(T, Y, 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

/// Two-step Sustained window over `t@1, t@0`.
#[must_use]
pub fn sustained_query() -> TemporalEffectQuery {
    pulse_query().with_policy(TemporalPolicy::sustained(-1, 0))
}

/// Mean curve of `y` over treatment doses `{0, 1}` at the given horizons.
#[must_use]
pub fn curve_query(horizons: Vec<u32>) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: Y,
        treatment: ContinuousDomain::new(T, GridSpec::Values(vec![0.0, 1.0].into())),
    })
    .with_temporal(TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(-1), Some(1)).unwrap())
}
