//! Repeated-sampling coverage harness shared by the calibration suites.
//!
//! Every calibration test counts how often a reported interval covers a known
//! truth over `n_sim()` independent datasets, then calls [`CoverageTally::assert`].
//! The acceptance band is two-sided: `level ± 3·MCSE` with
//! `MCSE = sqrt(level·(1-level)/n)`. At the default `n = 400` and a nominal 90%
//! level that is `[0.855, 0.945]`, so both a 75% interval and a near-100%
//! interval fail. Mean interval length is reported so a passing gate also
//! records how wide the interval had to be.
//!
//! `ANTECEDENT_CALIBRATION_NSIM` overrides the replicate count for local smoke
//! runs; the band widens automatically with fewer replicates. The gate script
//! uses the default.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, clippy::cast_precision_loss)]

/// Standard-normal 0.95 quantile (two-sided 90% interval).
pub const Z90: f64 = 1.644_853_626_951_472_2;

/// Default replicate count for 1.9 coverage tests.
pub const DEFAULT_N_SIM: u32 = 400;

/// Replicate count, honoring `ANTECEDENT_CALIBRATION_NSIM` for smoke runs.
#[must_use]
pub fn n_sim() -> u32 {
    std::env::var("ANTECEDENT_CALIBRATION_NSIM")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_N_SIM)
}

/// Two-sided acceptance band for an empirical coverage rate.
#[must_use]
pub fn coverage_band(n_sim: u32, level: f64) -> (f64, f64) {
    let mcse = (level * (1.0 - level) / f64::from(n_sim)).sqrt();
    ((level - 3.0 * mcse).max(0.0), (level + 3.0 * mcse).min(1.0))
}

/// Running coverage count for one calibration target.
#[derive(Debug, Clone)]
pub struct CoverageTally {
    name: String,
    level: f64,
    covered: u32,
    scored: u32,
    skipped: u32,
    length_sum: f64,
}

impl CoverageTally {
    /// Start a tally for `name` at nominal `level` (e.g. `0.9`).
    #[must_use]
    pub fn new(name: impl Into<String>, level: f64) -> Self {
        Self { name: name.into(), level, covered: 0, scored: 0, skipped: 0, length_sum: 0.0 }
    }

    /// Record one replicate's interval `[lo, hi]` against `truth`.
    ///
    /// A non-finite or inverted interval counts as a miss, never as a skip, so
    /// an estimator cannot improve its coverage by failing to report.
    pub fn record(&mut self, interval: Option<(f64, f64)>, truth: f64) {
        self.scored += 1;
        let Some((lo, hi)) =
            interval.filter(|(lo, hi)| lo.is_finite() && hi.is_finite() && lo <= hi)
        else {
            return;
        };
        self.length_sum += hi - lo;
        if truth >= lo && truth <= hi {
            self.covered += 1;
        }
    }

    /// Record a replicate that could not be evaluated for a documented reason
    /// (e.g. a refused fit on a degenerate draw). Skips are capped in [`assert`].
    pub fn skip(&mut self) {
        self.skipped += 1;
    }

    /// Empirical coverage over scored replicates.
    #[must_use]
    pub fn rate(&self) -> f64 {
        if self.scored == 0 { f64::NAN } else { f64::from(self.covered) / f64::from(self.scored) }
    }

    /// Mean interval length over replicates that produced an interval.
    #[must_use]
    pub fn mean_length(&self) -> f64 {
        if self.scored == 0 { f64::NAN } else { self.length_sum / f64::from(self.scored) }
    }

    /// Assert two-sided nominal coverage; at most 5% of replicates may be skipped.
    ///
    /// # Panics
    ///
    /// When coverage falls outside `level ± 3·MCSE` or too many replicates were skipped.
    pub fn assert(&self) {
        let total = self.scored + self.skipped;
        assert!(self.scored > 0, "{}: no replicates scored", self.name);
        assert!(
            self.skipped * 20 <= total,
            "{}: {} of {total} replicates skipped (cap 5%)",
            self.name,
            self.skipped
        );
        let (lo, hi) = coverage_band(self.scored, self.level);
        let rate = self.rate();
        let mcse = (self.level * (1.0 - self.level) / f64::from(self.scored)).sqrt();
        eprintln!(
            "calibration {}: nominal={:.2} coverage={rate:.3} mcse={mcse:.4} band=[{lo:.3}, {hi:.3}] \
             mean_length={:.4} ({}/{} covered, {} skipped)",
            self.name,
            self.level,
            self.mean_length(),
            self.covered,
            self.scored,
            self.skipped
        );
        assert!(
            rate >= lo && rate <= hi,
            "{} {:.0}% coverage={rate:.3} outside [{lo:.3}, {hi:.3}] ({}/{})",
            self.name,
            self.level * 100.0,
            self.covered,
            self.scored
        );
    }
}

/// Normal interval `est ± z·se`, or `None` when `se` is not a positive finite number.
#[must_use]
pub fn normal_interval(est: f64, se: Option<f64>, z: f64) -> Option<(f64, f64)> {
    let se = se.filter(|s| s.is_finite() && *s > 0.0)?;
    est.is_finite().then_some((est - z * se, est + z * se))
}

/// Equal-tailed empirical quantile interval of `draws` at `level`.
#[must_use]
pub fn quantile_interval(draws: &[f64], level: f64) -> Option<(f64, f64)> {
    let mut values: Vec<f64> = draws.iter().copied().filter(|v| v.is_finite()).collect();
    if values.len() < 2 {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let last = (values.len() - 1) as f64;
    let lo_p = (1.0 - level) / 2.0;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let at = |q: f64| values[(last * q).round().clamp(0.0, last) as usize];
    Some((at(lo_p), at(1.0 - lo_p)))
}

/// SplitMix64 finalizer: decorrelates nearby integer seeds.
#[must_use]
pub fn mix_seed(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform `[0, 1)` draw keyed by `seed` (e.g. to pick a structural atom per replicate).
#[must_use]
pub fn unit_uniform(seed: u64) -> f64 {
    (mix_seed(seed) >> 11) as f64 / (1u64 << 53) as f64
}

/// Deterministic standard-normal generator (LCG + Box–Muller), stable across platforms.
///
/// The seed is scrambled before seeding the LCG: a bare `seed | 1` maps the
/// consecutive replicate seeds `2k` and `2k + 1` to the same stream, which
/// silently halves the number of independent calibration datasets.
pub fn gaussian(seed: u64) -> impl FnMut() -> f64 {
    let mut state = mix_seed(seed) | 1;
    move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u1 = ((state >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0);
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u2 = (state >> 33) as f64 / (1u64 << 31) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Stationary AR(1) noise `e_t = rho·e_{t-1} + sd·sqrt(1-rho²)·z_t`, marginal SD `sd`.
///
/// `rho = 0` reproduces iid Gaussian noise with SD `sd`.
#[must_use]
pub fn ar1_noise(n: usize, rho: f64, sd: f64, seed: u64) -> Vec<f64> {
    assert!(rho.abs() < 1.0, "AR(1) coefficient must satisfy |rho| < 1");
    let mut z = gaussian(seed);
    let innovation = sd * (1.0 - rho * rho).sqrt();
    let mut out = Vec::with_capacity(n);
    let mut prev = sd * z();
    for _ in 0..n {
        let e = rho * prev + innovation * z();
        out.push(e);
        prev = e;
    }
    out
}
