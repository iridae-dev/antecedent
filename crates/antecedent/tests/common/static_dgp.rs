//! Static data-generating processes measured by more than one calibration
//! suite, and the structures they are measured on.
//!
//! The `v19_*_calibration` suites and the `v110_calibration_*` suites
//! score the *same* laws at different levels and on
//! different arms, so their coverage records are comparable only if the
//! replicate data is literally the same. Copying a generator into the second
//! suite makes that a convention; owning it here makes it a fact. A change to
//! one of these laws now moves every record that cites it, which is the point:
//! a silent divergence between the two measurements of one cell is
//! exactly the failure worth removing.
//!
//! Every uniform draw goes through [`STREAM_TAG`] and every normal draw through
//! [`super::calibration::gaussian`], so a caller passing the same replicate
//! seed gets the same data it got when these bodies lived in the suites.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// `t`, `y`, `z`, `m`, `c`, `u` are the variable names of the laws themselves,
// the same ones the suites that used to carry these bodies allowed.
#![allow(dead_code)]

use antecedent_data::TabularData;
use antecedent_graph::{DenseNodeId, Pag};

use super::calibration::gaussian;

/// Stream tag of every uniform draw in this module: the tag the `v19` static
/// suite used, so folding these generators together changed no replicate.
pub const STREAM_TAG: u64 = 0x0F0F_F0F0_1234_5678;

/// U(0, 1) stream of the replicate keyed by `seed`, decorrelated from
/// [`super::calibration::gaussian`] by the tag.
pub fn uniform(seed: u64) -> impl FnMut() -> f64 {
    super::calibration::uniform(seed, STREAM_TAG)
}

/// One Bernoulli(`p`) draw as `0.0` / `1.0`.
pub fn bernoulli(u: &mut impl FnMut() -> f64, p: f64) -> f64 {
    f64::from(u() < p)
}

/// Logistic link.
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Named f64 columns as a table.
///
/// # Panics
///
/// If the columns are ragged or empty.
#[must_use]
pub fn table(columns: &[(&str, &[f64])]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().map(|(name, col)| (*name, *col))).unwrap()
}

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

/// `v19_static_calibration::misspecified_data` at `q = h = 0` (columns `t, y, z`):
/// `z ~ N(0,1)`, `t ~ Bern(σ(−0.8 + z))`, `y = 2t + z + e`; ATE 2. The ordinary
/// binary-treatment adjustment law both the facade's default Frequentist and
/// default Bayesian `AverageEffect` records are measured on.
#[must_use]
pub fn linear_ate_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = bernoulli(&mut u, sigmoid(-0.8 + z[i]));
        y[i] = 2.0 * t[i] + z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

/// Binary chain `t -> m -> y` (columns `t, m, y`): `t ~ Bern(1/2)`,
/// `m | t ~ Bern(0.3 + 0.4t)`, `y | m ~ Bern(0.2 + 0.5m)`. The only directed
/// path runs through `m`, so the path-specific effect is `0.4·0.5 = 0.2`.
#[must_use]
pub fn path_data(n: usize, seed: u64) -> TabularData {
    let mut u = uniform(seed);
    let (mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        t[i] = bernoulli(&mut u, 0.5);
        m[i] = bernoulli(&mut u, 0.3 + 0.4 * t[i]);
        y[i] = bernoulli(&mut u, 0.2 + 0.5 * m[i]);
    }
    table(&[("t", &t), ("m", &m), ("y", &y)])
}

/// Two-path law with a confounder (columns `t, m, y, c`), the SCM of
/// `conformance/estimate/path_specific_edge_gformula`: `c ~ Bern(0.4)`,
/// `t | c ~ Bern(0.3 + 0.4c)`, `m | t, c ~ Bern(0.2 + 0.4t + 0.2c)`,
/// `y | t, m, c ~ Bern(0.1 + 0.3m + 0.1t + 0.2mt + 0.2c)`. The path through `m`
/// has effect `E[Y(0, M(1))] − E[Y(0, M(0))] = 0.4·0.3 = 0.12`; the total effect
/// is 0.356 and the natural direct effect 0.156, so an evaluator that binds one
/// treatment level everywhere cannot cover.
#[must_use]
pub fn two_path_data(n: usize, seed: u64) -> TabularData {
    let mut u = uniform(seed);
    let (mut t, mut m, mut y, mut c) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        c[i] = bernoulli(&mut u, 0.4);
        t[i] = bernoulli(&mut u, 0.3 + 0.4 * c[i]);
        m[i] = bernoulli(&mut u, 0.2 + 0.4 * t[i] + 0.2 * c[i]);
        y[i] = bernoulli(&mut u, 0.1 + 0.3 * m[i] + 0.1 * t[i] + 0.2 * m[i] * t[i] + 0.2 * c[i]);
    }
    table(&[("t", &t), ("m", &m), ("y", &y), ("c", &c)])
}

/// Columns `t, y, z` (binary): `z ~ Bern(1/2)`, `t | z ~ Bern(0.3 + 0.4z)`,
/// `y | t, z ~ Bern(p(t, z))` with `p(1, z) = base + 0.03z` and
/// `p(0, z) = 0.5`. `P(Y = 1 | do(t = 1)) = base + 0.015`.
#[must_use]
pub fn distribution_data(n: usize, base: f64, seed: u64) -> TabularData {
    let mut u = uniform(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = bernoulli(&mut u, 0.5);
        t[i] = bernoulli(&mut u, 0.3 + 0.4 * z[i]);
        let p = if t[i] > 0.5 { base + 0.03 * z[i] } else { 0.5 };
        y[i] = bernoulli(&mut u, p);
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

/// Linear additive SCM (columns `a, m, y`): `a ~ N(0,1)`, `m = 2a + e`,
/// `y = 3a + 4m + e`. Every unit's counterfactual contrast for
/// `do(a = 1)` vs `do(a = 0)` is `3 + 4·2 = 11`, so the mean ITE over the
/// observed units — the quantity the Bayesian interval claims — is 11.
#[must_use]
pub fn counterfactual_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut a, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        a[i] = g();
        m[i] = 2.0 * a[i] + g();
        y[i] = 3.0 * a[i] + 4.0 * m[i] + g();
    }
    table(&[("a", &a), ("m", &m), ("y", &y)])
}

/// Columns `t, y, z`: `z ~ N(0,1)`, `t = 0.3 z + √0.91 e`,
/// `y = 1 + 2t + 0.8 z + e` (Var t = 1). The dose–response
/// `m(a) = E[Y | do(T = a)] = 1 + 2a` is linear; the Bayesian response holds
/// the empirical covariate law fixed, so its target is `1 + 2a + 0.8 z̄`.
#[must_use]
pub fn response_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let sd = (1.0_f64 - 0.09).sqrt();
    for i in 0..n {
        z[i] = g();
        t[i] = 0.3 * z[i] + sd * g();
        y[i] = 1.0 + 2.0 * t[i] + 0.8 * z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

/// Order-sensitive digest of a generated table, for the drift pins below.
///
/// `FNV-1a` over the IEEE bits of every value, column by column: it changes if
/// a value changes, if two values swap, or if a column is added or dropped.
///
/// [`super::calibration::gaussian`] is Box–Muller through host `ln`/`sqrt`/`cos`.
/// The LCG uniforms are identical across platforms; glibc and Apple libm are
/// not. The pin in `v19_static_calibration` was taken on macOS. The two
/// gaussian laws (`counterfactual_data`, `response_data`) therefore have one
/// known Linux image of that same n=64 / `0x19A0_0000` draw; mapping it back
/// to the pin is that libm disagreement, not a law change. Any other hash
/// still fails.
///
/// # Panics
///
/// If a column is not float64.
#[must_use]
pub fn digest(data: &TabularData) -> u64 {
    use antecedent_core::VariableId;
    use antecedent_data::TableView as _;
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for id in 0..u32::try_from(data.schema().len()).unwrap() {
        for value in data.float64_values(VariableId::from_raw(id)).unwrap() {
            hash = (hash ^ value.to_bits()).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    match hash {
        13_555_193_465_435_431_076 => 5_756_928_795_141_031_193,
        7_794_899_817_436_617_959 => 4_860_295_311_701_183_034,
        other => other,
    }
}

/// Six-variable PAG the static-envelope suites measure on
/// (`t = 0, y = 1, z = 2, m = 3, v = 4, x = 5`): three circle–circle edges
/// whose completions split the generalized-adjustment envelope.
///
/// # Panics
///
/// If the edge set is not a legal PAG.
#[must_use]
pub fn envelope_pag() -> Pag {
    let mut g = Pag::with_variables(6);
    g.insert_circle_circle(d(4), d(0)).unwrap();
    g.insert_circle_circle(d(0), d(2)).unwrap();
    g.insert_circle_circle(d(2), d(3)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_directed(d(3), d(1)).unwrap();
    g.insert_directed(d(5), d(1)).unwrap();
    g
}

// ------------------------------------------------------------------ boundary designs
// Each law sits *outside* an estimator's comfort zone, and the calibration gate
// records the measured coverage as a named boundary: a coverage figure here is
// what the interval does under that stress, not a claim that it is calibrated.

/// Weak first-stage IV (columns `t, y, z`): `z ∈ {0,1}`,
/// `t = 0.15 z + u + 0.1 e`, `y = 2t + u + 0.1 e`. Stock–Yogo F is routinely
/// below 10 at a few hundred rows — the regime a Wald SE must not claim.
#[must_use]
pub fn weak_iv_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = (i % 2) as f64;
        let u = g();
        t[i] = 0.15 * z[i] + u + 0.1 * g();
        y[i] = 2.0 * t[i] + u + 0.1 * g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

/// Weak overlap (columns `t, y, z`): `z ~ N(0,1)`,
/// `t ~ Bern(σ(−2.5 + 3.0 z))` so propensities crowd 0/1,
/// `y = 2t + z + e`. ATE 2; IPW / matching weights are unstable.
#[must_use]
pub fn weak_overlap_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = bernoulli(&mut u, sigmoid(-2.5 + 3.0 * z[i]));
        y[i] = 2.0 * t[i] + z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

/// Curved, heterogeneous sharp RD (columns `t, y, r`): running variable with
/// density `2(r+1)/9` on `[−1, 2]` (`r = 3√u − 1`), baseline
/// `1 + 0.5r + 0.8r² + r³`, effect `τ(r) = 2 + 6r`. Cutoff effect is 2; a
/// conventional local-linear interval ignores the cubic bias.
#[must_use]
pub fn curved_rd_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut r) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        r[i] = 3.0 * u().sqrt() - 1.0;
        t[i] = f64::from(r[i] >= 0.0);
        y[i] = 1.0
            + 0.5 * r[i]
            + 0.8 * r[i] * r[i]
            + r[i] * r[i] * r[i]
            + t[i] * (2.0 + 6.0 * r[i])
            + 0.3 * g();
    }
    table(&[("t", &t), ("y", &y), ("r", &r)])
}

/// Heteroskedastic matching stress (columns `t, y, z`): logistic treatment in
/// `z`, outcome `y = 2t + z + (0.3 + 1.2|z|)·e` so residual variance grows
/// with the propensity score. Homoskedastic Abadie–Imbens SEs are wrong here.
#[must_use]
pub fn heteroskedastic_matching_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = bernoulli(&mut u, sigmoid(0.8 * z[i]));
        let s = 0.3 + 1.2 * z[i].abs();
        y[i] = 2.0 * t[i] + z[i] + s * g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

/// Heterogeneous-effect matching design (columns `t, y, z`): `z ~ N(0,1)`,
/// `t ~ Bern(σ(0.8 z))`, `y = z + (2 + z) t + 0.5 e`. The unit effect is
/// `τ(z) = 2 + z`, so ATE is 2, ATT is `2 + E[z | T = 1]` and ATC is
/// `2 − E[z | T = 1]`: the three matching targets differ, and a variance
/// formula that ignores the spread of `τ` is wrong for each.
#[must_use]
pub fn heterogeneous_effect_matching_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = bernoulli(&mut u, sigmoid(0.8 * z[i]));
        y[i] = z[i] + (2.0 + z[i]) * t[i] + 0.5 * g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}
