//! Shared-data covariance of scenario-specific estimates over one row sample.
//!
//! Several identified scenarios (graph completions, selection diagrams, ...) can each
//! estimate an effect from the SAME iid complete-row sample. Their estimates are then
//! sampling-dependent, and the dependence cannot be recovered from independent
//! per-scenario resamples: that design always reports a zero off-diagonal. The
//! estimator here resamples whole rows ONCE per replicate and feeds the same row
//! multiset (as per-row multiplicities) to every scenario estimator, so the joint
//! covariance matrix is estimated, not assumed.
//!
//! Two methods share one output:
//!
//! * `shared_row_bootstrap`: a deterministic, seed-derived whole-row nonparametric
//!   bootstrap. Each replicate has one replicate id; the row selection is a function of
//!   that id and the row count only, so it is identical for every scenario and does not
//!   depend on scenario order. A replicate where any estimator fails is dropped for all
//!   scenarios jointly and counted.
//! * `exact_enumeration`: for tiny `n`, every multinomial count vector of the resampled
//!   rows is enumerated with its exact probability, giving the bootstrap distribution's
//!   covariance without Monte Carlo error (a deterministic reference). Failing count
//!   vectors are dropped jointly, their probability mass reported, and the surviving
//!   mass renormalized.
//!
//! The output is a covariance matrix, a point-only claim: no interval is derived from
//! it. Unknown or incompatible dependence (different snapshots, different unit lists,
//! declared-independent samples, duplicate unit identities) is refused with
//! `route_not_supported` (`scenario_covariance.unknown_dependence`); a diagonal-only or
//! zero off-diagonal matrix is never a fallback.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use antecedent_core::ExecutionContext;

use crate::error::EstimationError;
use crate::splitmix::{GOLDEN_GAMMA, mix64, seed_mix, splitmix64};

/// Most scenarios one covariance may span.
pub const MAX_SCENARIOS: usize = 64;
/// Most bootstrap replicates one covariance may use.
pub const MAX_REPLICATES: usize = 2000;
/// Fewest bootstrap replicates (a covariance needs two).
pub const MIN_REPLICATES: usize = 2;
/// Fewest rows (one row cannot vary).
pub const MIN_ROWS: usize = 2;
/// Most rows the exact enumeration accepts (keeps `n^n` exact in `u128`).
pub const MAX_EXACT_ROWS: usize = 24;
/// Largest number of count vectors the exact enumeration will ever visit.
pub const EXACT_ENUMERATION_HARD_CAP: u64 = 4_000_000;
/// How the covariance may be read.
pub const COVARIANCE_INTERPRETATION: &str =
    "joint_sampling_covariance_of_scenario_estimates_point_only_not_an_interval";

/// Relative diagonal jitter allowed when checking positive semidefiniteness.
const PSD_RELATIVE_TOLERANCE: f64 = 1e-9;

/// The boxed per-scenario estimator: a plug-in estimate from per-row multiplicities
/// (`counts[i]` = times row `i` appears in the resampled multiset).
pub type RowEstimatorFn = Box<dyn Fn(&[u32]) -> Result<f64, EstimationError> + Send + Sync>;

/// How a scenario's estimator declares its sample relates to the family's.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RowDependence {
    /// Evaluated on the family's one shared row snapshot.
    SharedRows,
    /// Declared to use its own independent sample: no shared-data covariance exists.
    IndependentSample,
    /// Dependence with the other scenarios is not declared.
    Unknown,
}

/// Which covariance method produced a [`ScenarioCovariance`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CovarianceMethod {
    /// Seeded whole-row bootstrap with one replicate id across scenarios.
    SharedRowBootstrap,
    /// Exact enumeration of the multinomial resampling distribution.
    ExactEnumeration,
}

impl CovarianceMethod {
    /// Wire label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SharedRowBootstrap => "shared_row_bootstrap",
            Self::ExactEnumeration => "exact_enumeration",
        }
    }
}

/// One scenario's estimator over the shared rows, with the identities it is bound to.
pub struct ScenarioRowEstimator {
    id: Arc<str>,
    snapshot_digest: Arc<str>,
    unit_ids: Vec<Arc<str>>,
    dependence: RowDependence,
    estimator: RowEstimatorFn,
}

impl fmt::Debug for ScenarioRowEstimator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScenarioRowEstimator")
            .field("id", &self.id)
            .field("snapshot_digest", &self.snapshot_digest)
            .field("rows", &self.unit_ids.len())
            .field("dependence", &self.dependence)
            .finish_non_exhaustive()
    }
}

impl ScenarioRowEstimator {
    /// A scenario estimator bound to a row snapshot and its ordered unit ids, declared
    /// [`RowDependence::SharedRows`]. `estimator` receives per-row multiplicities
    /// (length = number of unit ids).
    #[must_use]
    pub fn new(
        id: impl Into<Arc<str>>,
        snapshot_digest: impl Into<Arc<str>>,
        unit_ids: Vec<Arc<str>>,
        estimator: impl Fn(&[u32]) -> Result<f64, EstimationError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            snapshot_digest: snapshot_digest.into(),
            unit_ids,
            dependence: RowDependence::SharedRows,
            estimator: Box::new(estimator),
        }
    }

    /// Replace the declared dependence.
    #[must_use]
    pub fn with_dependence(mut self, dependence: RowDependence) -> Self {
        self.dependence = dependence;
        self
    }

    /// Scenario id.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Bound row snapshot digest.
    #[must_use]
    pub fn snapshot_digest(&self) -> &str {
        &self.snapshot_digest
    }

    /// Ordered unit ids of the rows.
    #[must_use]
    pub fn unit_ids(&self) -> &[Arc<str>] {
        &self.unit_ids
    }

    /// Declared dependence.
    #[must_use]
    pub const fn dependence(&self) -> RowDependence {
        self.dependence
    }

    fn estimate(&self, counts: &[u32]) -> Option<f64> {
        (self.estimator)(counts).ok().filter(|v| v.is_finite())
    }
}

/// Options of the seeded shared-row bootstrap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedRowBootstrapOptions {
    /// Replicates drawn (`2..=MAX_REPLICATES`).
    pub replicates: usize,
    /// Seed of the replicate-id stream.
    pub seed: u64,
    /// Largest tolerated fraction of replicates dropped for an estimator failure.
    pub max_failure_fraction: f64,
}

/// Options of the exact multinomial enumeration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactEnumerationOptions {
    /// Declared cap on the number of count vectors (refused above; also bounded by
    /// [`EXACT_ENUMERATION_HARD_CAP`]).
    pub max_compositions: u64,
    /// Largest tolerated probability mass dropped for an estimator failure.
    pub max_failure_mass: f64,
}

/// Joint sampling covariance of the scenario estimates over the shared rows.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioCovariance {
    /// Scenario ids in the order of the matrix rows and columns.
    pub scenario_ids: Vec<Arc<str>>,
    /// Mean of each scenario's estimate over the retained resampling distribution.
    pub means: Vec<f64>,
    /// Row-major `K x K` covariance matrix (symmetric, checked positive semidefinite
    /// within tolerance).
    pub covariance: Vec<f64>,
    /// Replicate ids of the retained bootstrap replicates (empty for exact enumeration).
    pub replicate_ids: Vec<u64>,
    /// Replicate ids of the jointly dropped bootstrap replicates (empty for exact).
    pub failed_replicate_ids: Vec<u64>,
    /// Digest of the method, snapshot, row identities, scenario order, every replicate
    /// id and every replicate's row selection (or every enumerated count vector).
    pub replicate_digest: String,
    /// Digest of the ordered unit ids.
    pub row_identity_digest: String,
    /// The common row snapshot digest.
    pub snapshot_digest: Arc<str>,
    /// Method that produced the matrix.
    pub method: CovarianceMethod,
    /// Number of rows resampled.
    pub n_rows: usize,
    /// Replicates drawn (bootstrap) or count vectors enumerated (exact).
    pub replicates_total: u64,
    /// Replicates (count vectors) retained after joint dropping.
    pub replicates_used: u64,
    /// Replicates (count vectors) dropped because some estimator failed.
    pub failed_replicates: u64,
    /// Probability mass of the dropped count vectors (exact) or the dropped fraction
    /// (bootstrap).
    pub failed_mass: f64,
    /// Seed (bootstrap only).
    pub seed: Option<u64>,
    /// Always [`COVARIANCE_INTERPRETATION`].
    pub interpretation: &'static str,
}

impl ScenarioCovariance {
    /// Number of scenarios.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.scenario_ids.len()
    }

    /// Covariance of scenarios `i` and `j` (`NaN` out of range).
    #[must_use]
    pub fn entry(&self, i: usize, j: usize) -> f64 {
        let k = self.dimension();
        if i < k && j < k { self.covariance[i * k + j] } else { f64::NAN }
    }

    /// Position of a scenario id in the matrix order.
    #[must_use]
    pub fn position(&self, id: &str) -> Option<usize> {
        self.scenario_ids.iter().position(|s| &**s == id)
    }
}

fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

fn unknown_dependence(message: &str) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("route_not_supported"),
        "scenario_covariance.unknown_dependence",
        message,
    )
}

fn invalid(detail: &str, message: &str) -> EstimationError {
    refuse(antecedent_core::reason_code!("invalid_argument"), detail, message)
}

fn over_bound(detail: &str, message: &str) -> EstimationError {
    refuse(antecedent_core::reason_code!("cell_not_licensed"), detail, message)
}

fn numerical(detail: &str, message: &str) -> EstimationError {
    refuse(antecedent_core::reason_code!("transport_numerical_failure"), detail, message)
}

fn check_cancel(ctx: &ExecutionContext) -> Result<(), EstimationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(refuse(
            antecedent_core::reason_code!("cancelled_no_claim"),
            "scenario_covariance.cancelled",
            "cancellation was observed; no covariance is claimed",
        ));
    }
    Ok(())
}

fn widen(i: usize) -> u64 {
    u64::try_from(i).unwrap_or(u64::MAX)
}

/// Two-lane chained digest over 64-bit words (stable, not cryptographic).
struct Digest {
    a: u64,
    b: u64,
}

impl Digest {
    const fn new() -> Self {
        Self { a: 0xcbf2_9ce4_8422_2325, b: 0x9E37_79B9_7F4A_7C15 }
    }

    fn word(&mut self, w: u64) {
        self.a = mix64(self.a ^ w).wrapping_add(GOLDEN_GAMMA);
        self.b = mix64(self.b.rotate_left(17) ^ w ^ GOLDEN_GAMMA);
    }

    fn text(&mut self, s: &str) {
        self.word(widen(s.len()));
        for chunk in s.as_bytes().chunks(8) {
            let mut buf = [0_u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.word(u64::from_le_bytes(buf));
        }
    }

    fn hex(&self) -> String {
        format!("{:016x}{:016x}", self.a, self.b)
    }
}

struct FamilyIdentity {
    n: usize,
    row_digest: String,
    snapshot: Arc<str>,
}

/// Check that the scenarios share one row snapshot, one ordered unit list and a
/// declared shared-row dependence.
///
/// # Errors
/// `invalid_argument` for no scenarios, duplicate scenario ids or too few rows;
/// `cell_not_licensed` (`scenario_covariance.too_many_scenarios`) above
/// [`MAX_SCENARIOS`]; `route_not_supported` (`scenario_covariance.unknown_dependence`)
/// for a different snapshot, a different unit list, duplicate unit ids or a scenario not
/// declared [`RowDependence::SharedRows`].
fn validate_family(scenarios: &[ScenarioRowEstimator]) -> Result<FamilyIdentity, EstimationError> {
    let Some(first) = scenarios.first() else {
        return Err(invalid("scenario_covariance.no_scenarios", "no scenario estimators"));
    };
    if scenarios.len() > MAX_SCENARIOS {
        return Err(over_bound(
            "scenario_covariance.too_many_scenarios",
            &format!("{} scenarios exceed the limit of {MAX_SCENARIOS}", scenarios.len()),
        ));
    }
    let mut seen = HashSet::new();
    for s in scenarios {
        if !seen.insert(s.id()) {
            return Err(invalid(
                "scenario_covariance.duplicate_scenario_id",
                &format!("scenario id '{}' appears more than once", s.id()),
            ));
        }
    }
    for s in scenarios {
        if s.dependence != RowDependence::SharedRows {
            return Err(unknown_dependence(&format!(
                "scenario '{}' does not declare shared rows (dependence {:?}); independence or \
                 unknown dependence is never assumed and no off-diagonal is invented",
                s.id(),
                s.dependence
            )));
        }
        if s.snapshot_digest != first.snapshot_digest {
            return Err(unknown_dependence(&format!(
                "scenario '{}' is bound to a different row snapshot than '{}'",
                s.id(),
                first.id()
            )));
        }
        if s.unit_ids != first.unit_ids {
            return Err(unknown_dependence(&format!(
                "scenario '{}' has a different ordered unit-id list than '{}'",
                s.id(),
                first.id()
            )));
        }
    }
    let mut units = HashSet::new();
    let mut row_digest = Digest::new();
    for u in &first.unit_ids {
        if !units.insert(u) {
            return Err(unknown_dependence(&format!(
                "unit id '{u}' repeats and no cluster map declares the dependence"
            )));
        }
        row_digest.text(u);
    }
    let n = first.unit_ids.len();
    if n < MIN_ROWS {
        return Err(invalid(
            "scenario_covariance.too_few_rows",
            &format!("{n} rows; at least {MIN_ROWS} are needed to resample"),
        ));
    }
    Ok(FamilyIdentity { n, row_digest: row_digest.hex(), snapshot: first.snapshot_digest.clone() })
}

/// Deterministic replicate id of replicate `r` under `seed`.
#[must_use]
pub fn replicate_id(seed: u64, r: u64) -> u64 {
    mix64(seed_mix(seed) ^ r.wrapping_add(1).wrapping_mul(GOLDEN_GAMMA))
}

/// Draw `n` row indices with replacement from a replicate id, filling multiplicities
/// and folding the index list into the digest.
fn draw_counts(id: u64, counts: &mut [u32], digest: &mut Digest) {
    counts.fill(0);
    let n = widen(counts.len());
    let mut state = id;
    for _ in 0..counts.len() {
        let scaled = (u128::from(splitmix64(&mut state)) * u128::from(n)) >> 64;
        let idx = usize::try_from(scaled).unwrap_or(0);
        if let Some(c) = counts.get_mut(idx) {
            *c = c.saturating_add(1);
        }
        digest.word(widen(idx));
    }
}

fn evaluate_all(scenarios: &[ScenarioRowEstimator], counts: &[u32]) -> Option<Vec<f64>> {
    scenarios.iter().map(|s| s.estimate(counts)).collect()
}

/// Weighted mean and covariance of `values` (replicate-major, `k` per replicate).
/// `weights` sum to one; `correction` rescales for the bootstrap's `R / (R - 1)`.
fn moments(values: &[f64], weights: &[f64], k: usize, correction: f64) -> (Vec<f64>, Vec<f64>) {
    let mut means = vec![0.0; k];
    for (row, w) in values.chunks_exact(k).zip(weights) {
        for (m, v) in means.iter_mut().zip(row) {
            *m += w * v;
        }
    }
    let mut cov = vec![0.0; k * k];
    for (row, w) in values.chunks_exact(k).zip(weights) {
        for i in 0..k {
            let di = row[i] - means[i];
            for j in i..k {
                cov[i * k + j] += w * di * (row[j] - means[j]);
            }
        }
    }
    for i in 0..k {
        for j in i..k {
            let v = cov[i * k + j] * correction;
            cov[i * k + j] = v;
            cov[j * k + i] = v;
        }
    }
    (means, cov)
}

/// Cholesky of `cov + jitter * I`; `false` if any pivot is not positive.
fn is_psd(cov: &[f64], k: usize) -> bool {
    let trace: f64 = (0..k).map(|i| cov[i * k + i]).sum();
    let jitter = PSD_RELATIVE_TOLERANCE * (trace / k as f64).max(f64::MIN_POSITIVE);
    let mut l = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut s = cov[i * k + j] + if i == j { jitter } else { 0.0 };
            for m in 0..j {
                s -= l[i * k + m] * l[j * k + m];
            }
            if i == j {
                if !s.is_finite() || s <= 0.0 {
                    return false;
                }
                l[i * k + i] = s.sqrt();
            } else {
                l[i * k + j] = s / l[j * k + j];
            }
        }
    }
    true
}

struct Tally {
    values: Vec<f64>,
    weights: Vec<f64>,
    failed: u64,
    failed_mass: f64,
    total_mass: f64,
}

fn finish(
    scenarios: &[ScenarioRowEstimator],
    identity: &FamilyIdentity,
    tally: &Tally,
    method: CovarianceMethod,
    digest: &Digest,
) -> Result<ScenarioCovariance, EstimationError> {
    let k = scenarios.len();
    let kept_mass: f64 = tally.weights.iter().sum();
    if tally.weights.len() < 2 || kept_mass.is_nan() || kept_mass <= 0.0 {
        return Err(numerical(
            "scenario_covariance.too_many_failed_replicates",
            "fewer than two replicates survived joint dropping",
        ));
    }
    let weights: Vec<f64> = tally.weights.iter().map(|w| w / kept_mass).collect();
    let correction = match method {
        CovarianceMethod::SharedRowBootstrap => {
            let r = tally.weights.len() as f64;
            r / (r - 1.0)
        }
        CovarianceMethod::ExactEnumeration => 1.0,
    };
    let (means, covariance) = moments(&tally.values, &weights, k, correction);
    if covariance.iter().any(|v| !v.is_finite()) || !is_psd(&covariance, k) {
        return Err(numerical(
            "scenario_covariance.not_positive_semidefinite",
            "the estimated covariance is not finite and positive semidefinite within tolerance",
        ));
    }
    Ok(ScenarioCovariance {
        scenario_ids: scenarios.iter().map(|s| s.id.clone()).collect(),
        means,
        covariance,
        replicate_ids: Vec::new(),
        failed_replicate_ids: Vec::new(),
        replicate_digest: digest.hex(),
        row_identity_digest: identity.row_digest.clone(),
        snapshot_digest: identity.snapshot.clone(),
        method,
        n_rows: identity.n,
        replicates_total: 0,
        replicates_used: widen(tally.weights.len()),
        failed_replicates: tally.failed,
        failed_mass: tally.failed_mass / tally.total_mass,
        seed: None,
        interpretation: COVARIANCE_INTERPRETATION,
    })
}

fn seed_digest(scenarios: &[ScenarioRowEstimator], identity: &FamilyIdentity, tag: &str) -> Digest {
    let mut digest = Digest::new();
    digest.text(tag);
    digest.text(&identity.snapshot);
    digest.text(&identity.row_digest);
    digest.word(widen(scenarios.len()));
    for s in scenarios {
        digest.text(s.id());
    }
    digest
}

/// Joint covariance of the scenario estimates by a seeded whole-row bootstrap: one
/// replicate id and one row selection are shared by every scenario estimator.
///
/// The covariance is the sample covariance over the retained replicates (divisor
/// `R - 1`). A replicate where any estimator fails (error or non-finite value) is dropped
/// for all scenarios and counted.
///
/// # Errors
/// The [`validate_family`] refusals; `cell_not_licensed`
/// (`scenario_covariance.too_many_replicates`) above [`MAX_REPLICATES`];
/// `invalid_argument` (`scenario_covariance.too_few_replicates`, `...invalid_failure_bound`);
/// `transport_numerical_failure` (`scenario_covariance.too_many_failed_replicates`,
/// `scenario_covariance.not_positive_semidefinite`); `cancelled_no_claim`
/// (`scenario_covariance.cancelled`).
pub fn shared_row_bootstrap_covariance(
    scenarios: &[ScenarioRowEstimator],
    options: &SharedRowBootstrapOptions,
    ctx: &ExecutionContext,
) -> Result<ScenarioCovariance, EstimationError> {
    if options.replicates > MAX_REPLICATES {
        return Err(over_bound(
            "scenario_covariance.too_many_replicates",
            &format!("{} replicates exceed the limit of {MAX_REPLICATES}", options.replicates),
        ));
    }
    if options.replicates < MIN_REPLICATES {
        return Err(invalid(
            "scenario_covariance.too_few_replicates",
            &format!("{} replicates; at least {MIN_REPLICATES} are needed", options.replicates),
        ));
    }
    if !(0.0..=1.0).contains(&options.max_failure_fraction) {
        return Err(invalid(
            "scenario_covariance.invalid_failure_bound",
            "max_failure_fraction must lie in [0, 1]",
        ));
    }
    let identity = validate_family(scenarios)?;
    let k = scenarios.len();
    let mut digest = seed_digest(scenarios, &identity, "shared_row_bootstrap");
    digest.word(options.seed);
    digest.word(widen(options.replicates));
    let mut counts = vec![0_u32; identity.n];
    let mut tally = Tally {
        values: Vec::with_capacity(options.replicates * k),
        weights: Vec::with_capacity(options.replicates),
        failed: 0,
        failed_mass: 0.0,
        total_mass: options.replicates as f64,
    };
    let (mut kept_ids, mut failed_ids) = (Vec::new(), Vec::new());
    for r in 0..options.replicates {
        check_cancel(ctx)?;
        let id = replicate_id(options.seed, widen(r));
        digest.word(id);
        draw_counts(id, &mut counts, &mut digest);
        if let Some(row) = evaluate_all(scenarios, &counts) {
            tally.values.extend(row);
            tally.weights.push(1.0);
            kept_ids.push(id);
        } else {
            tally.failed += 1;
            tally.failed_mass += 1.0;
            failed_ids.push(id);
        }
    }
    if tally.failed_mass / tally.total_mass > options.max_failure_fraction {
        return Err(numerical(
            "scenario_covariance.too_many_failed_replicates",
            &format!(
                "{} of {} replicates failed in at least one scenario estimator, above the \
                 declared fraction {}",
                tally.failed, options.replicates, options.max_failure_fraction
            ),
        ));
    }
    let mut out =
        finish(scenarios, &identity, &tally, CovarianceMethod::SharedRowBootstrap, &digest)?;
    out.replicate_ids = kept_ids;
    out.failed_replicate_ids = failed_ids;
    out.replicates_total = widen(options.replicates);
    out.seed = Some(options.seed);
    Ok(out)
}

/// Number of multinomial count vectors of `n` rows: `C(2n - 1, n)`.
fn composition_count(n: usize) -> Option<u128> {
    let n = u128::from(widen(n));
    let mut acc: u128 = 1;
    for i in 1..=n {
        acc = acc.checked_mul(n - 1 + i)? / i;
    }
    Some(acc)
}

struct Enumerator<'a> {
    scenarios: &'a [ScenarioRowEstimator],
    ctx: &'a ExecutionContext,
    counts: Vec<u32>,
    fact: Vec<u128>,
    denominator: f64,
    tally: Tally,
    digest: Digest,
}

impl Enumerator<'_> {
    fn probability(&self) -> f64 {
        let mut m = self.fact[self.counts.len()];
        for &c in &self.counts {
            m /= self.fact[c as usize];
        }
        m as f64 / self.denominator
    }

    fn visit(&mut self) -> Result<(), EstimationError> {
        check_cancel(self.ctx)?;
        let p = self.probability();
        for &c in &self.counts {
            self.digest.word(u64::from(c));
        }
        self.tally.total_mass += p;
        if let Some(row) = evaluate_all(self.scenarios, &self.counts) {
            self.digest.word(1);
            self.tally.values.extend(row);
            self.tally.weights.push(p);
        } else {
            self.digest.word(0);
            self.tally.failed += 1;
            self.tally.failed_mass += p;
        }
        Ok(())
    }

    fn walk(&mut self, pos: usize, remaining: u32) -> Result<(), EstimationError> {
        if pos + 1 == self.counts.len() {
            self.counts[pos] = remaining;
            return self.visit();
        }
        for c in 0..=remaining {
            self.counts[pos] = c;
            self.walk(pos + 1, remaining - c)?;
        }
        Ok(())
    }
}

/// Exact covariance of the scenario estimates under the multinomial whole-row
/// resampling distribution (the bootstrap's expectation, with no Monte Carlo error).
///
/// Count vectors where any estimator fails are dropped for all scenarios; the surviving
/// probability mass is renormalized and the dropped mass reported as `failed_mass`. The
/// covariance divisor is the total surviving mass (population covariance of the
/// resampling distribution).
///
/// # Errors
/// The [`validate_family`] refusals; `cell_not_licensed`
/// (`scenario_covariance.exact_enumeration_cap`) when `n` exceeds [`MAX_EXACT_ROWS`] or the
/// number of count vectors exceeds the declared cap or [`EXACT_ENUMERATION_HARD_CAP`];
/// `invalid_argument` (`scenario_covariance.invalid_failure_bound`);
/// `transport_numerical_failure` (`scenario_covariance.too_many_failed_replicates`,
/// `scenario_covariance.not_positive_semidefinite`); `cancelled_no_claim`
/// (`scenario_covariance.cancelled`).
pub fn exact_enumeration_covariance(
    scenarios: &[ScenarioRowEstimator],
    options: &ExactEnumerationOptions,
    ctx: &ExecutionContext,
) -> Result<ScenarioCovariance, EstimationError> {
    if !(0.0..=1.0).contains(&options.max_failure_mass) {
        return Err(invalid(
            "scenario_covariance.invalid_failure_bound",
            "max_failure_mass must lie in [0, 1]",
        ));
    }
    let identity = validate_family(scenarios)?;
    let n = identity.n;
    let cap = options.max_compositions.min(EXACT_ENUMERATION_HARD_CAP);
    let total = if n <= MAX_EXACT_ROWS { composition_count(n) } else { None };
    let Some(total) = total.filter(|t| *t <= u128::from(cap)) else {
        return Err(over_bound(
            "scenario_covariance.exact_enumeration_cap",
            &format!(
                "{n} rows have more multinomial count vectors than the cap {cap} (rows at most \
                 {MAX_EXACT_ROWS}); use the shared-row bootstrap"
            ),
        ));
    };
    let mut fact = vec![1_u128];
    for i in 1..=n {
        let prev = fact[i - 1];
        fact.push(prev * u128::from(widen(i)));
    }
    let denominator = u128::from(widen(n)).pow(u32::try_from(n).unwrap_or(0)) as f64;
    let mut digest = seed_digest(scenarios, &identity, "exact_enumeration");
    digest.word(widen(n));
    let mut walker = Enumerator {
        scenarios,
        ctx,
        counts: vec![0_u32; n],
        fact,
        denominator,
        tally: Tally {
            values: Vec::new(),
            weights: Vec::new(),
            failed: 0,
            failed_mass: 0.0,
            total_mass: 0.0,
        },
        digest,
    };
    walker.walk(0, u32::try_from(n).unwrap_or(0))?;
    let Enumerator { tally, digest, .. } = walker;
    if tally.failed_mass / tally.total_mass > options.max_failure_mass {
        return Err(numerical(
            "scenario_covariance.too_many_failed_replicates",
            &format!(
                "estimator failures cover {} of the resampling mass, above the declared {}",
                tally.failed_mass / tally.total_mass,
                options.max_failure_mass
            ),
        ));
    }
    let mut out =
        finish(scenarios, &identity, &tally, CovarianceMethod::ExactEnumeration, &digest)?;
    out.replicates_total = u64::try_from(total).unwrap_or(u64::MAX);
    Ok(out)
}
