//! Typed view of `conformance/cross_surface/slice_cases.json`, shared by the Rust consumers.
//!
//! Each section names the scientific inputs and the hand-derived values of one slice case. The
//! Python consumer reads the same file, so a value changes in one place.

use serde::Deserialize;

use crate::{conformance_dir, load_json_fixture};

/// The whole fixture.
#[derive(Clone, Debug, Deserialize)]
pub struct SliceCases {
    /// The noiseless line `y = 1 + 2t`.
    pub gaussian: Gaussian,
    /// The binary adjustment over `P(t, z, y)`.
    pub finite_discrete: FiniteDiscrete,
    /// Studies pooled through the evidence ledger.
    pub mixed_evidence: MixedEvidence,
    /// The two-step panel with whole-unit shocks.
    pub temporal: Temporal,
    /// The declared modular feedback policy.
    pub feedback: Feedback,
    /// The recovered observation law.
    pub recovery: Recovery,
    /// Result retention.
    pub retention: Retention,
}

/// The line `y = intercept + effect * t`.
#[derive(Clone, Debug, Deserialize)]
pub struct Gaussian {
    /// Static treatment column.
    pub treatment: Vec<f64>,
    /// Static outcome column.
    pub outcome: Vec<f64>,
    /// Horizon treatment column, one row longer.
    pub horizon_treatment: Vec<f64>,
    /// Horizon outcome column.
    pub horizon_outcome: Vec<f64>,
    /// Line intercept.
    pub intercept: f64,
    /// Line slope and every cell's contrast.
    pub effect: f64,
    /// Conjugate posterior draws.
    pub posterior_draws: usize,
    /// Conjugate prior scale.
    pub prior_scale: f64,
}

/// One posterior draw of the joint table.
#[derive(Clone, Debug, Deserialize)]
pub struct PosteriorDraw {
    /// Joint counts, `t` slowest and `y` fastest.
    pub counts: Vec<u32>,
    /// Mixture weight.
    pub weight: f64,
    /// Interventional contrast of this draw.
    pub contrast: f64,
}

/// The frequency table, its adjustment and the two-draw posterior.
#[derive(Clone, Debug, Deserialize)]
pub struct FiniteDiscrete {
    /// Population name.
    pub population: String,
    /// Snapshot identity of the frequency table.
    pub snapshot: String,
    /// Regime id the static cell cites.
    pub static_regime: u32,
    /// Regime id the horizon cell cites.
    pub horizon_regime: u32,
    /// Joint counts, `t` slowest and `y` fastest.
    pub counts: Vec<u32>,
    /// `E[y | do(t = 1)]`.
    pub active: f64,
    /// `E[y | do(t = 0)]`.
    pub control: f64,
    /// `active - control`.
    pub contrast: f64,
    /// Weighted mean contrast over the posterior draws.
    pub posterior_mean: f64,
    /// The posterior draws.
    pub posterior_draws: Vec<PosteriorDraw>,
    /// Law origins that are not law origins.
    pub refused_origins: Vec<String>,
}

/// An observation set that grows into an overlap with its neighbour.
#[derive(Clone, Debug, Deserialize)]
pub struct Overlap {
    /// Observation range of the grown second study.
    pub grown_range: (u32, u32),
    /// Row-major declared covariance of the two studies.
    pub covariance: Vec<f64>,
    /// Generalized-least-squares pooled estimate.
    pub pooled: f64,
    /// Its variance.
    pub pooled_variance: f64,
    /// Observation slots declared across both studies.
    pub declared_slots: u64,
    /// Observations counted once.
    pub distinct_observations: u64,
    /// Observations the two studies share.
    pub shared: u64,
}

/// Two studies pooled by inverse variance, then an overlapping pair pooled by a declared
/// covariance.
#[derive(Clone, Debug, Deserialize)]
pub struct MixedEvidence {
    /// Study estimates.
    pub estimates: Vec<f64>,
    /// Study variances.
    pub variances: Vec<f64>,
    /// Inverse-variance pooled estimate.
    pub pooled: f64,
    /// Its variance.
    pub pooled_variance: f64,
    /// Disjoint observation ranges of the studies.
    pub observation_ranges: Vec<(u32, u32)>,
    /// The overlapping case.
    pub overlap: Overlap,
}

/// A panel whose unit shocks do not cancel.
#[derive(Clone, Debug, Deserialize)]
pub struct UnevenShocks {
    /// Shock added to every outcome of unit `u`, indexed by `u % len`.
    pub unit_shocks: Vec<f64>,
    /// The marginalized response, shifted by the mean shock.
    pub point: f64,
    /// Standard error of the mean of the per-unit scores.
    pub unit_score_standard_error: f64,
}

/// The two-step panel and its interval design.
#[derive(Clone, Debug, Deserialize)]
pub struct Temporal {
    /// Panel snapshot id.
    pub snapshot: String,
    /// Action sequence.
    pub sequence: (u32, u32),
    /// Snapshot id of the target initial-state law.
    pub target_snapshot: String,
    /// Target initial-state law as `(level, mass)`.
    pub target_law: Vec<(u32, f64)>,
    /// Replicate seed.
    pub seed: u64,
    /// Studentized replicates.
    pub replicates: usize,
    /// Fewest units.
    pub min_units: usize,
    /// Units in the panel.
    pub units: usize,
    /// Two-sided level.
    pub level: f64,
    /// Largest tolerated fraction of failed replicates.
    pub max_failed_fraction: f64,
    /// Shocks that cancel across units.
    pub unit_shocks: Vec<f64>,
    /// The response once the shocks cancel.
    pub point: f64,
    /// Normal quantile at `level`, for the interval a row-independent analysis would report.
    pub row_interval_z: f64,
    /// Shocks that do not cancel.
    pub uneven: UnevenShocks,
}

/// The named modular policy.
#[derive(Clone, Debug, Deserialize)]
pub struct Feedback {
    /// Shared-parameter and prior reconciliation text.
    pub reconciliation: String,
    /// Name of the feedback policy.
    pub specified: String,
    /// Name of the cut policy.
    pub cut: String,
}

/// The recovered observation law.
#[derive(Clone, Debug, Deserialize)]
pub struct Recovery {
    /// Population name.
    pub population: String,
    /// Snapshot id on the Rust surface.
    pub rust_snapshot: String,
    /// Snapshot id on the Python surface.
    pub python_snapshot: String,
}

/// Summary-only retention.
#[derive(Clone, Debug, Deserialize)]
pub struct Retention {
    /// Name of the reduced mode.
    pub reduced: String,
    /// Operations the reduced mode no longer serves, in snake case.
    pub unserved: Vec<String>,
}

/// Load `slice_cases.json`.
///
/// # Panics
///
/// Panics if the committed fixture is missing or malformed.
#[must_use]
pub fn load() -> SliceCases {
    load_json_fixture(
        &conformance_dir("cross_surface"),
        "slice_cases.json",
        "slice_cases.json is a committed cross-surface fixture",
    )
}
