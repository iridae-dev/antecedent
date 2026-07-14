//! Attribution result types (DESIGN.md §17.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{ComponentId, VariableId};

/// Per-component contribution with optional uncertainty.
#[derive(Clone, Debug)]
pub struct ComponentContribution {
    /// Component (usually a node mechanism).
    pub component: ComponentId,
    /// Point contribution (additive under Shapley; path-share under PathBased).
    pub contribution: f64,
    /// Optional posterior / bootstrap standard error.
    pub stderr: Option<f64>,
    /// Optional lower CI.
    pub ci_low: Option<f64>,
    /// Optional upper CI.
    pub ci_high: Option<f64>,
}

/// Pairwise interaction term when allocation is explicitly nonadditive.
#[derive(Clone, Debug)]
pub struct InteractionTerm {
    /// First component.
    pub a: ComponentId,
    /// Second component.
    pub b: ComponentId,
    /// Interaction contribution.
    pub value: f64,
}

/// Path-specific share of the outcome change.
#[derive(Clone, Debug)]
pub struct PathContribution {
    /// Ordered nodes along the directed path (source → … → outcome).
    pub path: Arc<[VariableId]>,
    /// Attributed share.
    pub contribution: f64,
}

/// Compute budget consumed by an attribution run.
#[derive(Clone, Debug, Default)]
pub struct ComputeBudget {
    /// Coalition / path evaluations performed.
    pub evaluations: u64,
    /// Monte Carlo / permutation samples drawn.
    pub samples: u64,
    /// Exact coalitions enumerated (`2^n` path), if any.
    pub exact_coalitions: u64,
}

/// Semantic cache hit/miss statistics.
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    /// Cache hits.
    pub hits: u64,
    /// Cache misses.
    pub misses: u64,
    /// Entries retained at end of run.
    pub entries: u64,
    /// Approximate bytes retained.
    pub bytes: u64,
}

/// Full change-attribution output (DESIGN.md §17.2).
#[derive(Clone, Debug)]
pub struct ChangeAttributionResult {
    /// Outcome variable.
    pub outcome: VariableId,
    /// Total measured change (comparison − baseline summary).
    pub total_change: f64,
    /// Per-component contributions (sum ≈ total under Shapley).
    pub contributions: Arc<[ComponentContribution]>,
    /// Explicit interaction terms when sequential / nonadditive.
    pub interactions: Arc<[InteractionTerm]>,
    /// Path breakdown when using path allocation.
    pub path_breakdown: Arc<[PathContribution]>,
    /// Components left unidentified (e.g. latent / structure-only).
    pub unidentified: Arc<[ComponentId]>,
    /// Graph-sensitivity: contribution std across graph samples (optional).
    pub graph_sensitivity: Option<Arc<[f64]>>,
    /// Compute budget report.
    pub budget: ComputeBudget,
    /// Monte Carlo standard error of the contribution vector (mean over components), if approx.
    pub monte_carlo_stderr: Option<f64>,
    /// Per-component MC stderr when available.
    pub component_mc_stderr: Option<Arc<[f64]>>,
    /// Coalition cache statistics.
    pub cache_stats: CacheStats,
}

impl ChangeAttributionResult {
    /// Sum of point contributions.
    #[must_use]
    pub fn contribution_sum(&self) -> f64 {
        self.contributions.iter().map(|c| c.contribution).sum()
    }
}

/// Mechanism-change detection result for one node (DESIGN.md §17.3).
#[derive(Clone, Debug)]
pub struct MechanismChangeDetection {
    /// Tested node.
    pub variable: VariableId,
    /// Whether a change was detected at the configured significance.
    pub changed: bool,
    /// Test statistic.
    pub statistic: f64,
    /// P-value (two-sided when applicable).
    pub p_value: f64,
    /// Method label (`likelihood_ratio`, `mean_diff`, `classifier_two_sample`, …).
    pub method: Arc<str>,
}

/// Unit-level contribution matrix (units × components).
#[derive(Clone, Debug)]
pub struct UnitChangeResult {
    /// Outcome.
    pub outcome: VariableId,
    /// Unit row indices.
    pub unit_rows: Arc<[usize]>,
    /// Component order matching columns of `contributions`.
    pub components: Arc<[ComponentId]>,
    /// Row-major `n_units * n_components` contributions.
    pub contributions: Arc<[f64]>,
    /// Aggregate (mean over units) contributions.
    pub mean_contributions: Arc<[f64]>,
    /// Budget / cache metadata.
    pub budget: ComputeBudget,
    /// Monte Carlo stderr of mean contributions (if approx).
    pub monte_carlo_stderr: Option<f64>,
    /// Cache stats.
    pub cache_stats: CacheStats,
}

/// Feature relevance under interventions.
#[derive(Clone, Debug)]
pub struct FeatureRelevance {
    /// Feature / parent variable.
    pub feature: VariableId,
    /// Outcome.
    pub outcome: VariableId,
    /// Relevance score (absolute interventional mean shift).
    pub score: f64,
}

/// Root-cause ranking entry.
#[derive(Clone, Debug)]
pub struct RootCauseRank {
    /// Ranked component.
    pub component: ComponentId,
    /// Score (higher = more responsible).
    pub score: f64,
    /// Optional graph-ensemble score std.
    pub graph_std: Option<f64>,
}
