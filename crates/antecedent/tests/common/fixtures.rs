//! Heterogeneous multi-atom calibration fixtures (1.9, R-15 / R-17 / R-18).
//!
//! Every fixture pairs a linear-Gaussian data-generating process with a graph
//! object carrying several atoms. The DBN and `TemporalCpdag` fixtures have
//! identified atoms that *disagree*; the `TemporalPag` and mediation fixtures
//! document why theirs do not (see each). Truths are the probability limits
//! of each atom's estimator under the DGP (`θ_g`), derived analytically from
//! the population covariance of the regressors the atom's estimand adjusts for.
//! They are not the causal effect of the DGP unless every atom is consistent
//! for it. A frozen-weight mixture targets `Σ_g w_g θ_g / Σ_g w_g` over
//! identified atoms; unidentified mass is never mixed.
//!
//! Every exogenous driver has unit marginal variance, and the treatment has
//! unit marginal variance by construction (`A² + SD_X² = 1`), so the formulas
//! below are plain covariances. `rho` applies AR(1) serial correlation to the
//! exogenous confounder `z` and to the outcome innovation; `rho = 0` is iid.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, clippy::many_single_char_names, clippy::similar_names)]

use antecedent_core::{Lag, VariableId};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::{GraphPosterior, set_edge};
use antecedent_graph::{Endpoint, MarkedEdge, MiddleMark, TemporalCpdag, TemporalDag, TemporalPag};
use antecedent_prob::InferenceDiagnostics;

use super::calibration::{ar1_noise, gaussian};

/// `z -> x` loading (contemporaneous confounder of the treatment).
pub const A: f64 = 0.8;
/// Treatment innovation SD; `A² + SD_X² = 1`.
pub const SD_X: f64 = 0.6;
/// Effect of `x[t-1]` on `y[t]`.
pub const B1: f64 = 0.8;
/// Effect of `x[t-2]` on `y[t]` (DBN fixture only).
pub const B2: f64 = 0.5;
/// Effect of `z[t-1]` on `y[t]`.
pub const GAMMA: f64 = 0.6;
/// Outcome innovation marginal SD.
pub const SD_Y: f64 = 0.5;

/// Burn-in rows dropped so every retained row has its full lag history.
const BURN: usize = 4;

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn series(columns: &[(&str, &[f64])]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(columns.iter().map(|(name, col)| (*name, *col)), 1).unwrap()
}

// ---------------------------------------------------------------------------
// Confounded lag DGP shared by the DBN and TemporalCpdag fixtures.
// ---------------------------------------------------------------------------

/// `z[t]` AR(1)(`rho`), `x[t] = A z[t] + SD_X e[t]`,
/// `y[t] = B1 x[t-1] + b2 x[t-2] + GAMMA z[t-1] + u[t]`, `u` AR(1)(`rho`).
///
/// Columns: `x` (0), `y` (1), `z` (2).
#[must_use]
pub fn confounded_series(n: usize, b2: f64, rho: f64, seed: u64) -> TimeSeriesData {
    let total = n + BURN;
    let z = ar1_noise(total, rho, 1.0, seed ^ 0x2A11);
    let u = ar1_noise(total, rho, SD_Y, seed ^ 0x0B0E);
    let mut e = gaussian(seed ^ 0x0E0E);
    let x: Vec<f64> = z.iter().map(|z| A * z + SD_X * e()).collect();
    let mut y = vec![0.0; total];
    for t in 2..total {
        y[t] = B1 * x[t - 1] + b2 * x[t - 2] + GAMMA * z[t - 1] + u[t];
    }
    series(&[("x", &x[BURN..]), ("y", &y[BURN..]), ("z", &z[BURN..])])
}

/// Plim of the treatment coefficient when `y[t]` is regressed on `x[t-1]` alone.
///
/// `Cov(x[t-2], x[t-1]) = A²·rho` and `Cov(z[t-1], x[t-1]) = A`; `Var(x) = 1`.
#[must_use]
pub fn unadjusted_pulse_plim(b2: f64, rho: f64) -> f64 {
    B1 + b2 * A * A * rho + GAMMA * A
}

/// Plim of `b1 + b2` when `y[t]` is regressed on `(x[t-1], x[t-2])` but not `z[t-1]`.
///
/// Omitted-variable bias is `GAMMA · Σ⁻¹ c` with `Σ = [[1, r], [r, 1]]`,
/// `r = A²·rho`, and `c = (Cov(z[t-1], x[t-1]), Cov(z[t-1], x[t-2])) = (A, A·rho)`.
#[must_use]
pub fn unadjusted_two_lag_plim(rho: f64) -> f64 {
    let r = A * A * rho;
    let (c1, c2) = (A, A * rho);
    let det = 1.0 - r * r;
    let k1 = (c1 - r * c2) / det;
    let k2 = (c2 - r * c1) / det;
    B1 + B2 + GAMMA * (k1 + k2)
}

// ---------------------------------------------------------------------------
// Heterogeneous DBN GraphPosterior.
// ---------------------------------------------------------------------------

/// DBN atom weights: two identified atoms, then one unidentified atom.
pub const DBN_WEIGHTS: [f64; 3] = [0.5, 0.3, 0.2];
/// Identified mass of [`heterogeneous_dbn`].
pub const DBN_IDENTIFIED_MASS: f64 = 0.8;
/// Unidentified mass of [`heterogeneous_dbn`].
pub const DBN_UNIDENTIFIED_MASS: f64 = 0.2;
/// Max lag of the DBN template.
pub const DBN_MAX_LAG: u32 = 2;

const P: usize = 3;

fn lag_bit(lag: u32, from: usize, to: usize) -> u64 {
    1u64 << ((lag as usize - 1) * P * P + from * P + to)
}

/// Contemporaneous `z -> x` shared by every DBN atom.
#[must_use]
pub fn dbn_contemporaneous_mask() -> u64 {
    set_edge(0, P, 2, 0, true)
}

/// Atom A (the DGP): `x@1 -> y`, `x@2 -> y`, `z@1 -> y`. Adjusts `z[t-1]`.
#[must_use]
pub fn dbn_atom_a_lag_mask() -> u64 {
    lag_bit(1, 0, 1) | lag_bit(2, 0, 1) | lag_bit(1, 2, 1)
}

/// Atom B: drops `z@1 -> y`, so its backdoor set is empty and it absorbs
/// the `z` path (and, under AR(1), the correlated `x[t-2]`) into its estimate.
#[must_use]
pub fn dbn_atom_b_lag_mask() -> u64 {
    lag_bit(1, 0, 1) | lag_bit(2, 0, 1)
}

/// Atom C: atom A plus `x@1 -> x`. The autoregressive treatment ancestry
/// crosses every finite history boundary, so `TemporalBackdoor` refuses to
/// certify it (`NotCertified`) and its weight stays unidentified.
#[must_use]
pub fn dbn_atom_c_lag_mask() -> u64 {
    dbn_atom_a_lag_mask() | lag_bit(1, 0, 0)
}

fn dbn_posterior(weights: &[f64], lag_masks: &[u64]) -> GraphPosterior {
    let cmask = dbn_contemporaneous_mask();
    let mut lagged = vec![0.0; DBN_MAX_LAG as usize * P * P];
    for (&w, &mask) in weights.iter().zip(lag_masks) {
        for (bit, slot) in lagged.iter_mut().enumerate() {
            if (mask >> bit) & 1 == 1 {
                *slot += w;
            }
        }
    }
    let mut edges = vec![0.0; P * P];
    edges[2 * P] = 1.0;
    GraphPosterior::new(
        P,
        weights.to_vec(),
        vec![cmask; weights.len()],
        edges.clone(),
        edges,
        1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        InferenceDiagnostics::analytic("v19_heterogeneous_dbn"),
        0,
    )
    .unwrap()
    .with_lagged_marginals(DBN_MAX_LAG, lagged)
    .unwrap()
    .with_lag_masks(lag_masks.to_vec())
    .unwrap()
}

/// Three-atom DBN posterior over `(x, y, z)`: atoms A and B identify with
/// different estimands, atom C is unidentified. Weights [`DBN_WEIGHTS`].
#[must_use]
pub fn heterogeneous_dbn() -> GraphPosterior {
    dbn_posterior(
        &DBN_WEIGHTS,
        &[dbn_atom_a_lag_mask(), dbn_atom_b_lag_mask(), dbn_atom_c_lag_mask()],
    )
}

/// Single-atom posterior on atom A (baseline, and the within-atom SE reference).
#[must_use]
pub fn dbn_atom_a_only() -> GraphPosterior {
    dbn_posterior(&[1.0], &[dbn_atom_a_lag_mask()])
}

/// Single-atom posterior on atom B (within-atom SE reference).
#[must_use]
pub fn dbn_atom_b_only() -> GraphPosterior {
    dbn_posterior(&[1.0], &[dbn_atom_b_lag_mask()])
}

/// Atom-level truths `(θ_A, θ_B)` for Pulse / single-step Sustained at lag 1.
///
/// Atom A regresses `y[t]` on `(x[t-1], z[t-1])`; given `z[t-1]`, the omitted
/// `x[t-2]` is uncorrelated with the treatment innovation, so `θ_A = B1` for
/// every `rho`. Atom B regresses on `x[t-1]` alone.
#[must_use]
pub fn dbn_pulse_atom_truths(rho: f64) -> [f64; 2] {
    [B1, unadjusted_pulse_plim(B2, rho)]
}

/// Atom-level truths for multi-step Sustained over lags 2 and 1.
///
/// Sequential g-computation fits `y` on its graph parents and propagates
/// both intervened copies: atom A recovers `B1 + B2`; atom B omits `z[t-1]`.
#[must_use]
pub fn dbn_multistep_atom_truths(rho: f64) -> [f64; 2] {
    [B1 + B2, unadjusted_two_lag_plim(rho)]
}

/// Frozen-weight mixture truth over identified DBN atoms.
#[must_use]
pub fn dbn_mixture_truth(atoms: [f64; 2]) -> f64 {
    (DBN_WEIGHTS[0] * atoms[0] + DBN_WEIGHTS[1] * atoms[1]) / DBN_IDENTIFIED_MASS
}

// ---------------------------------------------------------------------------
// TemporalCpdag with two disagreeing completions.
// ---------------------------------------------------------------------------

/// `t@1 -> y`, `z@1 -> y`, `t@1 — z@1`. Completion `z -> t` makes `z[t-1]` a
/// confounder (adjust, `θ = B1`); completion `t -> z` makes it a descendant
/// of the treatment (empty set, `θ = B1 + GAMMA·A`). Data: [`confounded_series`]
/// with `b2 = 0`.
#[must_use]
pub fn confounded_cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(var(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(var(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(var(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_undirected(z1, t1).unwrap();
    g
}

/// Single-completion CPDAG with the edge oriented `from -> to` (baseline and
/// within-atom SE reference). `z_to_t = true` is the adjusted completion.
#[must_use]
pub fn confounded_cpdag_oriented(z_to_t: bool) -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(var(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(var(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(var(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(z1, y0).unwrap();
    if z_to_t {
        g.insert_directed(z1, t1).unwrap();
    } else {
        g.insert_directed(t1, z1).unwrap();
    }
    g
}

/// Completion truths in envelope enumeration order: `t -> z` (unadjusted)
/// first, then `z -> t` (adjusted). Pinned by
/// `heterogeneous_cpdag_fixture_has_disagreeing_completions`.
#[must_use]
pub fn cpdag_completion_truths(rho: f64) -> [f64; 2] {
    [unadjusted_pulse_plim(0.0, rho), B1]
}

// ---------------------------------------------------------------------------
// TemporalPag with a circle mark: one identified, one unidentified completion.
// ---------------------------------------------------------------------------

/// `r -> t` loading.
pub const PAG_A_RT: f64 = 0.8;
/// Treatment innovation SD in the PAG DGP (`PAG_A_RT² + SD² = 1`).
pub const PAG_SD_T: f64 = 0.6;
/// `t -> z` loading.
pub const PAG_B_TZ: f64 = 0.8;
/// `r -> z` loading.
pub const PAG_C_RZ: f64 = 0.4;

/// `r[t]`, `t[t] = 0.8 r + 0.6 e`, `z[t] = 0.8 t + 0.4 r + 0.5 e`,
/// `y[t] = B1 t[t-1] + GAMMA z[t-1] + SD_Y e`.
///
/// Columns: `t` (0), `y` (1), `z` (2), `r` (3).
#[must_use]
pub fn pag_series(n: usize, seed: u64) -> TimeSeriesData {
    let total = n + BURN;
    let mut e = gaussian(seed ^ 0x9A6);
    let mut r = vec![0.0; total];
    let mut t = vec![0.0; total];
    let mut z = vec![0.0; total];
    let mut y = vec![0.0; total];
    for s in 0..total {
        r[s] = e();
        t[s] = PAG_A_RT * r[s] + PAG_SD_T * e();
        z[s] = PAG_B_TZ * t[s] + PAG_C_RZ * r[s] + 0.5 * e();
        if s > 0 {
            y[s] = B1 * t[s - 1] + GAMMA * z[s - 1] + SD_Y * e();
        }
    }
    series(&[("t", &t[BURN..]), ("y", &y[BURN..]), ("z", &z[BURN..]), ("r", &r[BURN..])])
}

/// Contemporaneous `r -> t`, `r -> z`, `t o-> z`; lagged `t@1 -> y`, `z@1 -> y`.
///
/// `r` makes `t -> y` visible (it points into `t` and is not adjacent to `y`)
/// and shields `t`, so both refinements of the circle keep the same colliders
/// and the class audit accepts two completions (envelope order):
///
/// 0. `t -> z`: the causal path `t -> z -> y` starts with an invisible edge,
///    so the MAG is not adjustment amenable. Unidentified (mass 1/2).
/// 1. `t <-> z`: `z[t-1]` blocks `t <-> z -> y` and `t <- r -> z -> y`; the
///    atom regresses `y[t]` on `(t[t-1], z[t-1])` and recovers `B1`.
///
/// Why not two identified, disagreeing PAG completions: per-completion
/// adjustment sets can differ only through (a) a tail at `t` on a circle edge
/// (a treatment descendant in one completion), whose edge must then be
/// visible, and any visibility witness into `t` turns the other refinement
/// into a new unshielded collider; or (b) collider status at a neighbour `z`
/// of `t` on the shielded triple `t - z - y`, which any witness making
/// `t -> y` visible turns into a discriminating path that fixes the status
/// across the class. This is the closest amenable construction found.
#[must_use]
pub fn circle_pag() -> TemporalPag {
    let mut g = TemporalPag::empty();
    let t0 = g.add_lagged(var(0), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = g.add_lagged(var(2), Lag::CONTEMPORANEOUS).unwrap();
    let r0 = g.add_lagged(var(3), Lag::CONTEMPORANEOUS).unwrap();
    let t1 = g.add_lagged(var(0), Lag::from_raw(1)).unwrap();
    let z1 = g.add_lagged(var(2), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(var(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(r0, t0).unwrap();
    g.insert_directed(r0, z0).unwrap();
    g.insert_marked(MarkedEdge {
        a: t0,
        b: z0,
        at_a: Endpoint::Circle,
        at_b: Endpoint::Arrow,
        middle: MiddleMark::Empty,
    })
    .unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g
}

/// Completion truths in envelope order; `None` marks the unidentified member.
#[must_use]
pub fn pag_completion_truths() -> [Option<f64>; 2] {
    [None, Some(B1)]
}

/// Enumeration mass of the unidentified PAG completion.
pub const PAG_UNIDENTIFIED_MASS: f64 = 0.5;

// ---------------------------------------------------------------------------
// TemporalCpdag mediation graph with two completions.
// ---------------------------------------------------------------------------

/// Mediator equation coefficient on `t[t-1]`.
pub const MED_ALPHA: f64 = 0.6;
/// Mediator equation coefficient on `z[t-1]`.
pub const MED_ETA: f64 = 0.5;
/// Direct `t[t-1] -> y[t]`.
pub const MED_BETA: f64 = 0.4;
/// Mediator `m[t] -> y[t]`.
pub const MED_DELTA: f64 = 0.5;
/// Outcome coefficient on `w[t-1]` in the mediator-outcome-confounded variant.
pub const MED_KAPPA: f64 = 0.5;

/// `z[t]` iid, `w[t] = 0.6 z[t] + 0.8 e`, `t[t]` iid,
/// `m[t] = 0.6 t[t-1] + 0.5 z[t-1] + 0.4 e`,
/// `y[t] = 0.4 t[t-1] + 0.5 m[t] + kappa w[t-1] + 0.4 e`.
///
/// With `kappa != 0`, `m <- z[t-1] -> w[t-1] -> y` confounds the mediator-outcome
/// relation without confounding `t -> y`. Columns: `t` (0), `m` (1), `y` (2),
/// `z` (3), `w` (4).
#[must_use]
pub fn mediation_series(n: usize, kappa: f64, seed: u64) -> TimeSeriesData {
    let total = n + BURN;
    let mut e = gaussian(seed ^ 0x3ED);
    let mut z = vec![0.0; total];
    let mut w = vec![0.0; total];
    let mut t = vec![0.0; total];
    let mut m = vec![0.0; total];
    let mut y = vec![0.0; total];
    for s in 0..total {
        z[s] = e();
        w[s] = 0.6 * z[s] + 0.8 * e();
        t[s] = e();
        if s > 0 {
            m[s] = MED_ALPHA * t[s - 1] + MED_ETA * z[s - 1] + 0.4 * e();
            y[s] = MED_BETA * t[s - 1] + MED_DELTA * m[s] + kappa * w[s - 1] + 0.4 * e();
        }
    }
    series(&[
        ("t", &t[BURN..]),
        ("m", &m[BURN..]),
        ("y", &y[BURN..]),
        ("z", &z[BURN..]),
        ("w", &w[BURN..]),
    ])
}

/// `t@1 -> m`, `t@1 -> y`, `m -> y`, `z@1 -> m`, `w@1 -> y`, `z@1 — w@1`.
///
/// Two completions (`z -> w`, `w -> z`) that necessarily agree: both give
/// `m` and `y` the same parents and the same `t -> y` backdoor set, and a
/// mediation class cannot do otherwise. An undirected edge at `m[t]` or `y[t]`
/// is oriented by Meek's rule from `t[t-1] -> m/y` (the completion sampler
/// enforces this), and an undirected edge at `t[t-1]` makes a parent of
/// `m`/`y` treatment-induced in one completion, which identification refuses
/// for the whole study. Each completion's interval is therefore calibrated
/// against the shared [`mediation_truth`].
#[must_use]
pub fn mediation_cpdag_two() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(var(0), Lag::from_raw(1)).unwrap();
    let m0 = g.add_lagged(var(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = g.add_lagged(var(2), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(var(3), Lag::from_raw(1)).unwrap();
    let w1 = g.add_lagged(var(4), Lag::from_raw(1)).unwrap();
    g.insert_directed(t1, m0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(m0, y0).unwrap();
    g.insert_directed(z1, m0).unwrap();
    g.insert_directed(w1, y0).unwrap();
    g.insert_undirected(z1, w1).unwrap();
    g
}

/// Mediated-effect truth of every completion of [`mediation_cpdag_two`]: the
/// completions are correct for the DGP, so the identified mediated effect is
/// the path product `MED_ALPHA·MED_DELTA` for every `kappa`.
#[must_use]
pub fn mediation_truth() -> f64 {
    MED_ALPHA * MED_DELTA
}

/// Probability limit of an estimator that adjusts only the `t -> y` backdoor
/// set (empty here) and so omits the mediator-outcome confounder: the outcome
/// coefficient on `m` absorbs `kappa·Cov(w[t-1], m | t) / Var(m | t)` with
/// `Cov = 0.6·MED_ETA` and `Var = MED_ETA² + 0.4²`. Recorded for diagnosis only.
#[must_use]
pub fn mediation_backdoor_only_plim(kappa: f64) -> f64 {
    MED_ALPHA * (MED_DELTA + kappa * 0.6 * MED_ETA / (MED_ETA * MED_ETA + 0.16))
}

// ---------------------------------------------------------------------------
// Two-lag TemporalDag (R-18).
// ---------------------------------------------------------------------------

/// `x` iid `N(0, 0.4²)`, `y[t] = B1 x[t-1] + B2 x[t-2] + 0.35 e`.
///
/// Columns: `x` (0), `y` (1). Multi-step Sustained over lags 2 and 1 composes
/// `B1 + B2`; a Pulse at lag 1 is `B1` and at lag 2 is `B2`.
#[must_use]
pub fn two_lag_series(n: usize, seed: u64) -> TimeSeriesData {
    let total = n + BURN;
    let mut e = gaussian(seed ^ 0x2146);
    let x: Vec<f64> = (0..total).map(|_| 0.4 * e()).collect();
    let mut y = vec![0.0; total];
    for t in 2..total {
        y[t] = B1 * x[t - 1] + B2 * x[t - 2] + 0.35 * e();
    }
    series(&[("x", &x[BURN..]), ("y", &y[BURN..])])
}

/// `x@1 -> y`, `x@2 -> y`.
#[must_use]
pub fn two_lag_dag() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let x1 = g.add_lagged(var(0), Lag::from_raw(1)).unwrap();
    let x2 = g.add_lagged(var(0), Lag::from_raw(2)).unwrap();
    let y0 = g.add_lagged(var(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g.insert_directed(x2, y0).unwrap();
    g
}
