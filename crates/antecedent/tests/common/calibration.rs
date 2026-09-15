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
//! The band alone is a tolerance, not a claim that the interval's true
//! coverage is the level: an interval whose true coverage is 0.87 passes a
//! 400-replicate band most of the time. Two further rules make a pass mean
//! something:
//!
//! * **Recheck.** At fewer than [`PRECISION_N_SIM`] replicates, a cell whose
//!   coverage falls below `level − RECHECK_SHORTFALL` (2 points) while still
//!   inside the band passes the band but prints a `calibration-recheck` line.
//!   `scripts/gate_calibration.sh` re-runs such a group at
//!   [`RECHECK_N_SIM`] replicates and takes that run's verdict.
//! * **Precision floor.** At [`PRECISION_N_SIM`] replicates or more, coverage
//!   must also be at least `level − 2·MCSE` (one-sided): 0.887 at 2000
//!   replicates and a 90% level, 0.940 at 95%. A cell that measures 2–4
//!   points low at 400 replicates therefore fails once it is measured
//!   precisely; a cell whose true coverage is the level fails the floor about
//!   2% of the time.
//!
//! "Nominal" in a test name means the cell passes both: the band at the
//! gate's replicate count and, when rechecked, the precision floor. Designs
//! whose precise measurement sits below the floor are named as boundary
//! cells and assert their measured band, not nominal ones.
//!
//! `ANTECEDENT_CALIBRATION_NSIM` overrides the replicate count; the band
//! widens automatically with fewer replicates, and the precision floor applies
//! whenever the count reaches [`PRECISION_N_SIM`]. The gate script uses the
//! default of 400 (the full gate takes about 1.6 h in release at 400; 1000
//! would take about 4 h before any recheck) and rechecks at 2000.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, clippy::cast_precision_loss)]

/// Standard-normal 0.95 quantile (two-sided 90% interval).
pub const Z90: f64 = 1.644_853_626_951_472_2;

/// Default replicate count for 1.9 coverage tests.
pub const DEFAULT_N_SIM: u32 = 400;

/// Replicate count the gate script re-runs a `calibration-recheck` group at.
pub const RECHECK_N_SIM: u32 = 2000;

/// Replicate count from which the one-sided precision floor applies.
pub const PRECISION_N_SIM: u32 = 1000;

/// Shortfall below the level (in coverage points) that asks for a recheck at
/// fewer than [`PRECISION_N_SIM`] replicates.
pub const RECHECK_SHORTFALL: f64 = 0.02;

/// Replicate count, honoring `ANTECEDENT_CALIBRATION_NSIM` for smoke runs.
#[must_use]
pub fn n_sim() -> u32 {
    std::env::var("ANTECEDENT_CALIBRATION_NSIM")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_N_SIM)
}

/// Monte Carlo standard error of an empirical coverage rate at the level.
#[must_use]
pub fn coverage_mcse(n_sim: u32, level: f64) -> f64 {
    (level * (1.0 - level) / f64::from(n_sim)).sqrt()
}

/// Two-sided acceptance band for an empirical coverage rate.
#[must_use]
pub fn coverage_band(n_sim: u32, level: f64) -> (f64, f64) {
    let mcse = coverage_mcse(n_sim, level);
    ((level - 3.0 * mcse).max(0.0), (level + 3.0 * mcse).min(1.0))
}

/// One-sided precision floor `level − 2·MCSE`, in force from [`PRECISION_N_SIM`]
/// replicates; `None` below that count.
#[must_use]
pub fn precision_floor(n_sim: u32, level: f64) -> Option<f64> {
    (n_sim >= PRECISION_N_SIM).then(|| level - 2.0 * coverage_mcse(n_sim, level))
}

/// Whether a rate at `n_sim` replicates asks for a recheck: below
/// `level − RECHECK_SHORTFALL` at fewer than [`PRECISION_N_SIM`] replicates.
#[must_use]
pub fn needs_recheck(n_sim: u32, level: f64, rate: f64) -> bool {
    n_sim < PRECISION_N_SIM && rate < level - RECHECK_SHORTFALL
}

/// Match-key fields for a licensed coverage record.
#[derive(Debug, Clone, Copy)]
pub struct RecordKey {
    pub query: &'static str,
    pub graph_class: &'static str,
    pub inference: &'static str,
    pub estimator: &'static str,
    pub interval_method: &'static str,
    pub se_kind: &'static str,
    pub dgp: &'static str,
    pub n: u64,
    pub dependence: &'static str,
}

/// Running coverage count for one calibration target.
#[derive(Debug, Clone)]
pub struct CoverageTally {
    name: String,
    level: f64,
    covered: u32,
    scored: u32,
    skipped: u32,
    /// Scored replicates that produced a finite, ordered interval.
    with_interval: u32,
    length_sum: f64,
    record: Option<RecordKey>,
}

impl CoverageTally {
    /// Start a tally for `name` at nominal `level` (e.g. `0.9`).
    #[must_use]
    pub fn new(name: impl Into<String>, level: f64) -> Self {
        Self {
            name: name.into(),
            level,
            covered: 0,
            scored: 0,
            skipped: 0,
            with_interval: 0,
            length_sum: 0.0,
            record: None,
        }
    }

    /// Tally that backs a licensed coverage record.
    #[must_use]
    pub fn for_record(key: RecordKey, level: f64) -> Self {
        let id = format!(
            "cov.{}.{}.{}.{}.{}.{}",
            snake(key.query),
            snake(key.graph_class),
            key.inference.to_ascii_lowercase(),
            if key.estimator.is_empty() { "none" } else { key.estimator },
            if key.se_kind.is_empty() { "none" } else { key.se_kind },
            key.dependence
        );
        Self {
            name: id,
            level,
            covered: 0,
            scored: 0,
            skipped: 0,
            with_interval: 0,
            length_sum: 0.0,
            record: Some(key),
        }
    }

    fn emit_record(&self, boundary: bool) {
        let Some(key) = self.record else {
            return;
        };
        eprintln!(
            "calibration-record {}",
            serde_json::json!({
                "id": self.name,
                "query": key.query,
                "graph_class": key.graph_class,
                "inference": key.inference,
                "estimator": key.estimator,
                "interval_method": key.interval_method,
                "se_kind": key.se_kind,
                "dgp": key.dgp,
                "n": key.n,
                "dependence": key.dependence,
                "nominal": self.level,
                "observed": self.rate(),
                "mcse": coverage_mcse(self.scored, self.level),
                "replicates": self.scored,
                "boundary": boundary,
                "test": "",
            })
        );
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
        self.with_interval += 1;
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
        if self.with_interval == 0 {
            f64::NAN
        } else {
            self.length_sum / f64::from(self.with_interval)
        }
    }

    /// Assert nominal coverage; at most 5% of replicates may be skipped.
    ///
    /// The two-sided band `level ± 3·MCSE` always applies. From
    /// [`PRECISION_N_SIM`] replicates the one-sided floor `level − 2·MCSE`
    /// applies as well; below that count a rate more than
    /// [`RECHECK_SHORTFALL`] under the level passes but prints a
    /// `calibration-recheck` line for the gate script to act on.
    ///
    /// # Panics
    ///
    /// When coverage falls outside the band, below the precision floor, or too
    /// many replicates were skipped.
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
        let mcse = coverage_mcse(self.scored, self.level);
        let floor = precision_floor(self.scored, self.level);
        eprintln!(
            "calibration {}: nominal={:.2} coverage={rate:.3} mcse={mcse:.4} band=[{lo:.3}, {hi:.3}]{} \
             mean_length={:.4} ({}/{} covered, {} skipped)",
            self.name,
            self.level,
            floor.map_or(String::new(), |f| format!(" floor={f:.3}")),
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
        if let Some(floor) = floor {
            assert!(
                rate >= floor,
                "{} {:.0}% coverage={rate:.3} below the precision floor {floor:.3} \
                 (level - 2 MCSE at {} replicates; {}/{})",
                self.name,
                self.level * 100.0,
                self.scored,
                self.covered,
                self.scored
            );
        } else if needs_recheck(self.scored, self.level, rate) {
            eprintln!(
                "calibration-recheck {}: coverage={rate:.3} is more than {RECHECK_SHORTFALL:.2} \
                 below {:.2} at {} replicates; re-run at ANTECEDENT_CALIBRATION_NSIM={RECHECK_N_SIM}",
                self.name, self.level, self.scored
            );
        }
        self.emit_record(false);
    }
}

fn snake(name: &str) -> String {
    let mut out = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(ch.to_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

impl CoverageTally {
    /// Assert a named boundary cell against its *measured* coverage: the rate
    /// must lie within `measured ± 3·MCSE` at the replicate count. No recheck
    /// and no precision floor apply, because the cell is documented as
    /// under-covering (the mechanism is named where the test is defined); the
    /// assertion guards against a regression below, or a silent change above,
    /// the measured level.
    ///
    /// # Panics
    ///
    /// When coverage falls outside `measured ± 3·MCSE` or too many replicates
    /// were skipped.
    pub fn assert_boundary(&self, measured: f64) {
        let total = self.scored + self.skipped;
        assert!(self.scored > 0, "{}: no replicates scored", self.name);
        assert!(
            self.skipped * 20 <= total,
            "{}: {} of {total} replicates skipped (cap 5%)",
            self.name,
            self.skipped
        );
        let mcse = coverage_mcse(self.scored, self.level);
        let (lo, hi) = ((measured - 3.0 * mcse).max(0.0), (measured + 3.0 * mcse).min(1.0));
        let rate = self.rate();
        eprintln!(
            "calibration-boundary {}: nominal={:.2} measured={measured:.3} coverage={rate:.3} \
             mcse={mcse:.4} band=[{lo:.3}, {hi:.3}] mean_length={:.4} ({}/{} covered, {} skipped)",
            self.name,
            self.level,
            self.mean_length(),
            self.covered,
            self.scored,
            self.skipped
        );
        assert!(
            rate >= lo && rate <= hi,
            "{} boundary coverage={rate:.3} outside the measured band [{lo:.3}, {hi:.3}] \
             around {measured:.3} ({}/{})",
            self.name,
            self.covered,
            self.scored
        );
        self.emit_record(true);
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

/// `SplitMix64` finalizer: decorrelates nearby integer seeds.
#[must_use]
pub fn mix_seed(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed of the independent stream `stream` of the replicate keyed by `seed`.
///
/// Several noise series of one replicate must not be seeded as `seed ^ tag`
/// with nearby tags: `(seed ^ 0x…1)` for replicate `r` equals `(seed ^ 0x…2)`
/// for replicate `r ^ 3`, so with consecutive replicate seeds one replicate's
/// treatment path is another's residual path. Scrambling the replicate seed
/// first leaves no such algebraic relation between replicates.
#[must_use]
pub fn stream_seed(seed: u64, stream: u64) -> u64 {
    mix_seed(seed).wrapping_add(mix_seed(stream))
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
