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
//! **Coverage records.** A tally built with [`CoverageTally::for_record`]
//! backs a row of `parity/coverage_records.toml`. Each scored replicate's
//! execution is bound with [`CoverageTally::bind`] (for facade executions,
//! `common::calibration_bind::bind`), which takes the construction from the
//! runtime's own calibration match key, so a record describes exactly what
//! the facade reported. [`CoverageTally::assert`] then prints a
//! `calibration-record` line (boundary `false`), [`CoverageTally::assert_boundary`]
//! one flagged boundary, and [`CoverageTally::emit`] one for an
//! [`CoverageTally::unasserted`] second level scored on the same replicates
//! (boundary when it misses the nominal band). `scripts/collect_coverage_records.py`
//! turns those lines into the registry.
//!
//! **Sample-size grid.** A record's scope is a measured range, not one row
//! count. Every record-keyed design draws its sample size through a
//! [`SampleGrid`] (`grid_n(base)` for the standard `n/2, n, 2n`), and the gate
//! runs each group once per grid point with [`GRID_POINT_ENV`] set. Each run
//! prints its `calibration-record` line with its `grid_point`; the collector
//! merges the points of one record into `n_min..n_max` with the coverage
//! measured at every point, and the record is a boundary when any point is.
//! The band, the recheck and the precision floor apply per point, so a
//! failing point is never averaged into a pass. The base point
//! ([`BASE_GRID_POINT`], also the default when the variable is unset)
//! generates exactly the replicate data the design generated before the grid;
//! the other points salt every generator in this module ([`grid_salt`]).
//!
//! `ANTECEDENT_CALIBRATION_NSIM` overrides the replicate count; the band
//! widens automatically with fewer replicates, and the precision floor applies
//! whenever the count reaches [`PRECISION_N_SIM`]. The gate script uses the
//! default of 400 and rechecks at 2000. Independent seeds run across
//! [`map_replicates`] workers (`std::thread::scope` + `available_parallelism`);
//! each seed still builds a serial [`antecedent_core::ExecutionContext::for_tests`]
//! study. Same seed, same interval. The old 1.6 h / overnight figures were
//! one-core loops.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, clippy::cast_precision_loss)]

/// Standard-normal 0.95 quantile (two-sided 90% interval).
pub const Z90: f64 = 1.644_853_626_951_472_2;

/// Standard-normal 0.975 quantile (two-sided 95% interval, the level the
/// runtime reports a standard error at).
pub const Z95: f64 = 1.959_963_984_540_054;

/// Level the runtime reports standard-error and posterior intervals at
/// (`antecedent::result::REPORTED_SE_INTERVAL_LEVEL`).
pub const REPORTED_LEVEL: f64 = 0.95;

/// Default replicate count for 1.9 coverage tests.
pub const DEFAULT_N_SIM: u32 = 400;

/// Replicate count the gate script re-runs a `calibration-recheck` group at.
pub const RECHECK_N_SIM: u32 = 2000;

/// Replicate count from which the one-sided precision floor applies.
pub const PRECISION_N_SIM: u32 = 1000;

/// Shortfall below the level (in coverage points) that asks for a recheck at
/// fewer than [`PRECISION_N_SIM`] replicates.
pub const RECHECK_SHORTFALL: f64 = 0.02;

/// Environment variable selecting the sample-size grid point a run measures
/// (`0`, `1` or `2`; unset is the base point [`BASE_GRID_POINT`]).
pub const GRID_POINT_ENV: &str = "ANTECEDENT_CALIBRATION_GRID_POINT";

/// Number of sample-size grid points every record-keyed tally is measured at.
pub const GRID_POINTS: usize = 3;

/// The grid point whose sample size is the design's base `n` and whose
/// replicate data is exactly the data the design generated before the grid
/// existed (no seed salt).
pub const BASE_GRID_POINT: usize = 1;

/// Sample-size grid of a coverage design: the factor applied to the design's
/// base `n` at each of the [`GRID_POINTS`] points, smallest first. The base
/// point (factor 1) is always [`BASE_GRID_POINT`].
///
/// A coverage record's measured scope is `[n at point 0, n at point 2]`; the
/// runtime labels an execution `calibrated` only inside that range and only
/// when the construction passed at every point
/// (`scripts/collect_coverage_records.py` merges the points; the matcher in
/// `antecedent_io::calibration` never extrapolates beyond them).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleGrid {
    pub name: &'static str,
    /// `(numerator, denominator)` of each point's factor.
    pub factors: [(usize, usize); GRID_POINTS],
}

impl SampleGrid {
    /// `n/2, n, 2n`: the default for every design whose base `n` is a
    /// practical data size (tabular 300–1200 rows, series of 160–400 steps).
    pub const STANDARD: Self = Self { name: "standard", factors: [(1, 2), (1, 1), (2, 1)] };
    /// `3n/4, n, 2n`: short-series designs (base `n` ≤ 100 steps), whose base
    /// point is already the shortest series the construction is licensed for;
    /// halving it would measure a series the runtime warns on, not the design.
    pub const SHORT_SERIES: Self = Self { name: "short_series", factors: [(3, 4), (1, 1), (2, 1)] };
    /// `n/2, n, 3n/2`: designs whose cost grows faster than `n` and that
    /// already run for hours at the base point (Bayesian derivative and
    /// response-Jacobian bands at 1000 rows, the 2500-row counterfactual
    /// designs, the ADMG front-door distribution).
    pub const HEAVY: Self = Self { name: "heavy", factors: [(1, 2), (1, 1), (3, 2)] };

    /// The design's sample size at the current grid point ([`grid_point`]).
    #[must_use]
    pub fn n(self, base: usize) -> usize {
        self.n_at(grid_point(), base)
    }

    /// The design's sample size at grid point `point` (rounded to nearest).
    #[must_use]
    pub fn n_at(self, point: usize, base: usize) -> usize {
        let (num, den) = self.factors[point];
        (base * num + den / 2) / den
    }
}

/// Sample-size grid point of this run, from [`GRID_POINT_ENV`].
///
/// # Panics
///
/// When the variable is set to anything but `0`, `1` or `2`: a typo must not
/// silently re-measure the base point under another point's label.
#[must_use]
pub fn grid_point() -> usize {
    static POINT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *POINT.get_or_init(|| match std::env::var(GRID_POINT_ENV) {
        Err(_) => BASE_GRID_POINT,
        Ok(raw) => raw
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|&point| point < GRID_POINTS)
            .unwrap_or_else(|| panic!("{GRID_POINT_ENV}={raw}: want 0, 1 or 2")),
    })
}

/// Largest base `n` measured on [`SampleGrid::SHORT_SERIES`] by [`grid_n`].
pub const SHORT_SERIES_MAX_BASE: usize = 100;

/// Sample size of a design with base `base` at this run's grid point:
/// [`SampleGrid::SHORT_SERIES`] for a base of at most
/// [`SHORT_SERIES_MAX_BASE`], [`SampleGrid::STANDARD`] otherwise. Heavy designs
/// call [`SampleGrid::HEAVY`] explicitly.
#[must_use]
pub fn grid_n(base: usize) -> usize {
    grid_for(base).n(base)
}

/// The grid [`grid_n`] measures a design with base `base` on.
#[must_use]
pub fn grid_for(base: usize) -> SampleGrid {
    if base <= SHORT_SERIES_MAX_BASE { SampleGrid::SHORT_SERIES } else { SampleGrid::STANDARD }
}

/// Seed salt of this run's grid point: zero at [`BASE_GRID_POINT`] (the data
/// every design generated before the grid), a fixed distinct constant at the
/// other points, so each point measures independent replicate data and every
/// point stays deterministic.
#[must_use]
pub fn grid_salt() -> u64 {
    const SALTS: [u64; GRID_POINTS] = [0x9E6C_63D0_676A_9A98, 0, 0xD1B5_4A32_D192_ED02];
    SALTS[grid_point()]
}

/// A replicate seed for a generator that does not draw through this module
/// (e.g. `CausalRng::from_seed`): unchanged at the base point, salted at the
/// others. Every generator in this module is salted already.
#[must_use]
pub fn grid_seed(seed: u64) -> u64 {
    seed ^ grid_salt()
}

/// Environment variable of a wiring smoke run (`1`): tallies emit their
/// records flagged `"smoke": true` and never gate. A smoke run proves that
/// designs scale with the grid and that the records key and bind; it measures
/// nothing (it is paired with a reduced `ANTECEDENT_CALIBRATION_NSIM`), so
/// `scripts/collect_coverage_records.py` refuses its lines unless it writes a
/// scratch registry with `--smoke`.
pub const SMOKE_ENV: &str = "ANTECEDENT_CALIBRATION_SMOKE";

/// Whether this is a wiring smoke run ([`SMOKE_ENV`]).
#[must_use]
pub fn smoke() -> bool {
    std::env::var(SMOKE_ENV).is_ok_and(|value| value == "1")
}

/// Replicate count, honoring `ANTECEDENT_CALIBRATION_NSIM` for smoke runs.
#[must_use]
pub fn n_sim() -> u32 {
    std::env::var("ANTECEDENT_CALIBRATION_NSIM")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_N_SIM)
}

/// Evaluate `f(0), …, f(n-1)` on `available_parallelism` workers.
///
/// Results come back in seed order so `bind` / `record` stay deterministic.
/// Each call is an independent dataset; `f` must be deterministic in `rep`.
/// Workers are `std::thread::scope` threads — not `rayon`, and not product
/// [`antecedent_core::ExecutionContext`] parallelism. A single replicate
/// still builds its own serial test context.
pub fn map_replicates<T: Send>(n: u32, f: impl Fn(u64) -> T + Sync) -> Vec<T> {
    let n_us = usize::try_from(n).expect("replicate count fits usize");
    if n_us == 0 {
        return Vec::new();
    }
    let threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .clamp(1, n_us);
    if threads == 1 {
        return (0..n).map(|rep| f(u64::from(rep))).collect();
    }
    let mut out: Vec<Option<T>> = (0..n_us).map(|_| None).collect();
    std::thread::scope(|scope| {
        let f = &f;
        let mut rest = out.as_mut_slice();
        let mut start = 0usize;
        for t in 0..threads {
            let take = rest.len().div_ceil(threads - t);
            let (mine, next) = rest.split_at_mut(take);
            let begin = start;
            scope.spawn(move || {
                for (k, slot) in mine.iter_mut().enumerate() {
                    *slot = Some(f((begin + k) as u64));
                }
            });
            rest = next;
            start += take;
            if rest.is_empty() {
                break;
            }
        }
    });
    out.into_iter().map(|slot| slot.expect("every replicate was filled")).collect()
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

/// Provenance of a record-keyed tally: which test measured it, on which data
/// generating process, and which reported interval it scores.
///
/// `test` and `dgp` are function names in the calling file (or
/// `path::fn` for a DGP defined elsewhere, e.g. in `tests/common/`);
/// `interval` is the `IntervalMethod::as_str()` of the interval the test
/// scores. The construction (query, graph axis, estimator, SE kind,
/// dependence, posterior, identification) is never declared by hand: it is
/// bound from the executions themselves with [`CoverageTally::bind`], so a
/// record can only describe a construction the facade actually reported.
#[derive(Debug, Clone, Copy)]
pub struct RecordKey {
    pub test: &'static str,
    pub dgp: &'static str,
    pub interval: &'static str,
}

/// Construction of one reported interval, as the runtime keys it
/// (`antecedent_io::calibration::CalibrationKeyWire` without the level: the
/// record's level is the tally's).
#[derive(Debug, Clone, PartialEq)]
pub struct Construction {
    pub query: String,
    pub graph_class: String,
    pub structure: String,
    pub modality: String,
    pub inference: String,
    pub estimator: String,
    pub interval_method: String,
    pub se_kind: String,
    pub dependence: String,
    pub posterior: String,
    pub functional: String,
    pub identification: String,
    /// Level the runtime reported this interval at. Not part of the record
    /// key (a record's level is its tally's), but an [`CoverageTally::unasserted`]
    /// tally must score exactly this level.
    pub reported_level: f64,
}

/// Execution facts of one replicate that bound a record's scope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScopeFacts {
    pub row_count: u64,
    pub replicates_ok: Option<u32>,
    pub posterior_draws: Option<u32>,
    pub unidentified_mass: f64,
}

/// Construction plus the measured scope aggregated over bound replicates.
#[derive(Debug, Clone)]
struct BoundRecord {
    key: RecordKey,
    file: &'static str,
    label: Option<String>,
    construction: Option<Construction>,
    n_min: u64,
    n_max: u64,
    replicates_min: Option<u32>,
    posterior_draws_min: Option<u32>,
    unidentified_mass_max: f64,
    bound: u32,
    /// Emits without asserting (a second level scored on the same replicates).
    unasserted: bool,
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
    record: Option<BoundRecord>,
}

impl CoverageTally {
    /// Start a tally for `name` at nominal `level` (e.g. `0.9`). Emits no record.
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

    /// Tally that backs a coverage record in `parity/coverage_records.toml`.
    ///
    /// The single emission entry point. Bind every scored replicate's
    /// execution with [`Self::bind`]; [`Self::assert`] /
    /// [`Self::assert_boundary`] (or [`Self::emit`] for an
    /// [`Self::unasserted`] tally) then print one `calibration-record <json>`
    /// line, which `scripts/collect_coverage_records.py` collects.
    #[must_use]
    #[track_caller]
    pub fn for_record(key: RecordKey, level: f64) -> Self {
        let file = std::panic::Location::caller().file();
        let mut tally = Self::new(key.test, level);
        tally.record = Some(BoundRecord {
            key,
            file,
            label: None,
            construction: None,
            n_min: u64::MAX,
            n_max: 0,
            replicates_min: None,
            posterior_draws_min: None,
            unidentified_mass_max: 0.0,
            bound: 0,
            unasserted: false,
        });
        tally
    }

    /// Interval method a record-keyed tally scores.
    #[must_use]
    pub fn record_interval(&self) -> Option<&'static str> {
        self.record.as_ref().map(|record| record.key.interval)
    }

    /// Distinguish several records one test emits for the same interval
    /// (e.g. one per grid point).
    #[must_use]
    pub fn labelled(mut self, label: impl Into<String>) -> Self {
        let label = label.into();
        self.name = format!("{} [{label}]", self.name);
        if let Some(record) = self.record.as_mut() {
            record.label = Some(label);
        }
        self
    }

    /// A tally scored on the same replicates at a second level (the runtime's
    /// reported level when the gate asserts another). It is never asserted:
    /// [`Self::emit`] records its measured coverage, flagged as a boundary
    /// when it falls outside the nominal band or below the precision floor.
    #[must_use]
    pub fn unasserted(mut self) -> Self {
        if let Some(record) = self.record.as_mut() {
            record.unasserted = true;
        }
        self
    }

    /// Bind one replicate's execution: the construction the runtime keyed
    /// the scored interval under, and the execution's scope facts.
    ///
    /// # Panics
    ///
    /// When the construction differs between replicates, when it is not the
    /// interval method the record declared, or on a tally without a record.
    pub fn bind(&mut self, construction: &Construction, scope: ScopeFacts) {
        let name = self.name.clone();
        let record = self
            .record
            .as_mut()
            .unwrap_or_else(|| panic!("{name}: bind on a tally without a record"));
        assert_eq!(
            construction.interval_method, record.key.interval,
            "{name}: the scored interval is not the one the runtime reported as {}",
            construction.interval_method
        );
        if record.unasserted {
            assert!(
                (construction.reported_level - self.level).abs() < 1e-9,
                "{name}: the runtime reported this interval at {}, not at the unasserted tally's {}",
                construction.reported_level,
                self.level
            );
        }
        match &record.construction {
            Some(seen) => assert_eq!(
                seen, construction,
                "{name}: replicates were keyed under different constructions"
            ),
            None => record.construction = Some(construction.clone()),
        }
        record.bound += 1;
        record.n_min = record.n_min.min(scope.row_count);
        record.n_max = record.n_max.max(scope.row_count);
        record.replicates_min = min_opt(record.replicates_min, scope.replicates_ok);
        record.posterior_draws_min = min_opt(record.posterior_draws_min, scope.posterior_draws);
        record.unidentified_mass_max = record.unidentified_mass_max.max(scope.unidentified_mass);
    }

    /// Whether the rate passes the nominal band (and the precision floor at
    /// [`PRECISION_N_SIM`] replicates or more).
    fn passes_nominal(&self) -> bool {
        let (lo, hi) = coverage_band(self.scored, self.level);
        let rate = self.rate();
        rate >= lo
            && rate <= hi
            && precision_floor(self.scored, self.level).is_none_or(|f| rate >= f)
    }

    /// Print an [`Self::unasserted`] tally's line and its record.
    ///
    /// # Panics
    ///
    /// On a tally that is not unasserted, or that bound no execution.
    pub fn emit(&self) {
        assert!(
            self.record.as_ref().is_some_and(|record| record.unasserted),
            "{}: emit is for unasserted record tallies",
            self.name
        );
        eprintln!(
            "calibration {} (recorded, not gated): nominal={:.2} coverage={:.3} mcse={:.4} \
             mean_length={:.4} ({}/{} covered, {} skipped)",
            self.name,
            self.level,
            self.rate(),
            coverage_mcse(self.scored, self.level),
            self.mean_length(),
            self.covered,
            self.scored,
            self.skipped
        );
        self.emit_record(!self.passes_nominal(), "reported_level");
    }

    /// Record a named boundary cell whose coverage the test measures but does
    /// not gate (e.g. a series below its SE family's short-series threshold,
    /// where the runtime warns). The record is always a boundary.
    ///
    /// # Panics
    ///
    /// On a tally without a record, or one that bound no execution.
    pub fn emit_named_boundary(&self) {
        assert!(self.record.is_some(), "{}: emit_named_boundary needs a record tally", self.name);
        eprintln!(
            "calibration-boundary {} (recorded, not gated): nominal={:.2} coverage={:.3} \
             mean_length={:.4} ({}/{} covered, {} skipped)",
            self.name,
            self.level,
            self.rate(),
            self.mean_length(),
            self.covered,
            self.scored,
            self.skipped
        );
        self.emit_record(true, "named_boundary");
    }

    fn emit_record(&self, boundary: bool, role: &str) {
        let Some(record) = self.record.as_ref() else {
            return;
        };
        let construction = record
            .construction
            .as_ref()
            .unwrap_or_else(|| panic!("{}: record tally never bound to an execution", self.name));
        let file = record.file.replace('\\', "/");
        let dgp = if record.key.dgp.contains("::") {
            record.key.dgp.to_string()
        } else {
            format!("{file}::{}", record.key.dgp)
        };
        // A nominal level is in [0, 1], so the rounded percentage is in [0, 100].
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let level_pct = (self.level * 100.0).round() as u32;
        let mut id = format!(
            "cov.{}.{}.{}.{}.l{level_pct}.{}",
            snake(&construction.query),
            snake(&construction.graph_class),
            construction.inference.to_ascii_lowercase(),
            construction.interval_method,
            record.key.test
        );
        if let Some(label) = &record.label {
            id.push('.');
            id.push_str(&sanitize_label(label));
        }
        let rate = self.rate();
        let mut payload = serde_json::json!({
                "id": id,
                "query": construction.query,
                "graph_class": construction.graph_class,
                "structure": construction.structure,
                "modality": construction.modality,
                "inference": construction.inference,
                "estimator": construction.estimator,
                "interval_method": construction.interval_method,
                "se_kind": construction.se_kind,
                "dependence": construction.dependence,
                "posterior": construction.posterior,
                "functional": construction.functional,
                "identification": construction.identification,
                "nominal": self.level,
                "n_min": record.n_min,
                "n_max": record.n_max,
                "replicates_min": record.replicates_min.unwrap_or(0),
                "posterior_draws_min": record.posterior_draws_min.unwrap_or(0),
                "unidentified_mass_max": record.unidentified_mass_max,
                "observed": rate,
                "mcse": (rate * (1.0 - rate) / f64::from(self.scored.max(1))).sqrt(),
                "replicates": self.scored,
                "bound_replicates": record.bound,
                "grid_point": grid_point(),
                "boundary": boundary,
                "role": role,
                "dgp": dgp,
                "test": format!("{file}::{}", record.key.test),
        });
        if smoke() {
            payload["smoke"] = serde_json::Value::Bool(true);
        }
        // One write of the whole line: a whole-file gate group runs its tests on
        // several threads with stdout and stderr in one pipe, and a line written
        // in pieces can be split by another test's `test ... ok`, which leaves
        // the collector an unparseable record.
        let line = format!("calibration-record {payload}\n");
        let _ = std::io::Write::write_all(&mut std::io::stderr().lock(), line.as_bytes());
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
        assert!(
            !self.record.as_ref().is_some_and(|record| record.unasserted),
            "{}: an unasserted tally is emitted, not asserted",
            self.name
        );
        if smoke() {
            self.emit_smoke(!self.passes_nominal(), "gated");
            return;
        }
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
        self.emit_record(false, "gated");
    }
}

fn min_opt(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// One `_` per non-alphanumeric character, so labels that differ only in
/// punctuation (`a=1` and `a=-1`) keep different record ids.
fn sanitize_label(label: &str) -> String {
    let out: String = label
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch.to_ascii_lowercase() } else { '_' })
        .collect();
    out.trim_matches('_').to_string()
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
    /// `measured` is the coverage measured at the base grid point. At the other
    /// sample-size grid points the cell is recorded as a boundary without a
    /// band (nothing was measured there to hold it to); once those points are
    /// measured, [`Self::assert_boundary_at`] holds each point to its own value.
    ///
    /// # Panics
    ///
    /// When coverage falls outside `measured ± 3·MCSE` or too many replicates
    /// were skipped.
    pub fn assert_boundary(&self, measured: f64) {
        if smoke() {
            self.emit_smoke(true, "named_boundary");
            return;
        }
        if grid_point() != BASE_GRID_POINT {
            self.check_skips();
            eprintln!(
                "calibration-boundary {} (grid point {}; measured {measured:.3} at the base point \
                 only, recorded, not gated): nominal={:.2} coverage={:.3} mean_length={:.4} \
                 ({}/{} covered, {} skipped)",
                self.name,
                grid_point(),
                self.level,
                self.rate(),
                self.mean_length(),
                self.covered,
                self.scored,
                self.skipped
            );
            self.emit_record(true, "named_boundary");
            return;
        }
        self.assert_measured_band(measured);
    }

    /// Assert a cell per sample-size grid point: at a point with
    /// `Some(measured)` the cell is a named boundary held to
    /// `measured ± 3·MCSE`; at a point with `None` it must pass the nominal
    /// band (and floor or recheck) like [`Self::assert`]. The record is a
    /// boundary over its whole range when any point is one, with the coverage
    /// measured at each point.
    ///
    /// # Panics
    ///
    /// As [`Self::assert`] or [`Self::assert_boundary`] at this run's point.
    pub fn assert_boundary_at(&self, measured: [Option<f64>; GRID_POINTS]) {
        match measured[grid_point()] {
            Some(value) => self.assert_measured_band(value),
            None => self.assert(),
        }
    }

    /// A smoke run's line: never gated, flagged `"smoke": true` (see [`SMOKE_ENV`]).
    fn emit_smoke(&self, boundary: bool, role: &str) {
        eprintln!(
            "calibration-smoke {} (grid point {}; not gated): nominal={:.2} coverage={:.3} \
             ({}/{} covered, {} skipped)",
            self.name,
            grid_point(),
            self.level,
            self.rate(),
            self.covered,
            self.scored,
            self.skipped
        );
        if self.record.is_some() {
            self.emit_record(boundary, role);
        }
    }

    fn check_skips(&self) {
        assert!(
            !self.record.as_ref().is_some_and(|record| record.unasserted),
            "{}: an unasserted tally is emitted, not asserted",
            self.name
        );
        let total = self.scored + self.skipped;
        assert!(self.scored > 0, "{}: no replicates scored", self.name);
        assert!(
            self.skipped * 20 <= total,
            "{}: {} of {total} replicates skipped (cap 5%)",
            self.name,
            self.skipped
        );
    }

    fn assert_measured_band(&self, measured: f64) {
        assert!(
            !self.record.as_ref().is_some_and(|record| record.unasserted),
            "{}: an unasserted tally is emitted, not asserted",
            self.name
        );
        if smoke() {
            self.emit_smoke(true, "named_boundary");
            return;
        }
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
        self.emit_record(true, "named_boundary");
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
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_add(grid_salt());
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

/// Multiplier of the LCG every calibration suite draws its uniforms from.
pub const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// The one uniform `[0, 1)` stream body, given an already-conditioned `state`.
///
/// Prefer [`uniform`], which conditions the seed correctly. This entry point
/// exists for the suites whose recorded coverage was measured under a
/// different seed conditioning: folding the *stream* must not silently
/// re-generate their data, so their conditioning stays at the call site and is
/// documented there.
pub fn uniform_from_state(state: u64) -> impl FnMut() -> f64 {
    let mut state = grid_seed(state);
    move || {
        state = state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(1);
        (state >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Uniform `[0, 1)` stream of the replicate keyed by `seed`, in stream `tag`.
///
/// The seed is scrambled before it conditions the LCG, for the reason
/// [`gaussian`] gives: a bare `seed | 1` maps the consecutive replicate seeds
/// `2k` and `2k + 1` to the same stream.
pub fn uniform(seed: u64, tag: u64) -> impl FnMut() -> f64 {
    uniform_from_state(mix_seed(seed ^ tag) | 1)
}

/// Deterministic standard-normal generator (LCG + Box–Muller).
///
/// The LCG uniforms are bit-identical across platforms. The `ln`/`sqrt`/`cos`
/// step is host libm, so the last bits of each draw can differ (glibc vs
/// Apple). That is not a law change; see [`super::static_dgp::digest`].
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
