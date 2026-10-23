//! Intrinsic causal influence: how much of a target's variance each upstream noise term
//! (and the target's own) is responsible for.
//!
//! The game is `v(S) = Var(E[Y | U_S])` over the exogenous-noise coordinates `U_S` of the
//! sources in `S`, so `v(∅) = 0` and `v(all) = Var(Y)`, and the Shapley values split the
//! target's variance among the sources (Janzing, Blöbaum, Mastakouri, Faller, Minorics &
//! Budhathoki 2024, "Quantifying intrinsic causal contributions via structure preserving
//! interventions", AISTATS).
//!
//! # Estimating `v(S)`
//!
//! `E[Y | U_S]` is a nested Monte Carlo. A replicate batch draws `m` outer noise vectors and
//! `k` inner draws of every coordinate outside `S` for each. The batch holds two noise tables
//! of `m·k` rows: `fixed`, whose rows `i·k .. (i+1)·k` all carry outer draw `i`, and
//! `resampled`, whose rows are independent. A coalition is one replay of the compiled model
//! that reads the coordinates of `S` from `fixed` and the others from `resampled`
//! ([`NoiseSource::coalition`]); the model is not cloned and no table is copied per coalition.
//!
//! The variance of the `m` inner means overstates `Var(E[Y | U_S])` by
//! `E[Var(Y | U_S)] / k`, so the estimator is the analysis-of-variance form
//!
//! ```text
//! v̂(S) = s²(inner means) − mean(inner sample variances) / k,
//! ```
//!
//! which is unbiased because both `E[s²(means)] = Var(E[Y | U_S]) + E[Var(Y | U_S)]/k` and
//! `E[inner sample variance] = E[Var(Y | U_S)]`. It is not clipped at zero (clipping would
//! bias it), so a small coalition value can be slightly negative.
//!
//! # Uncertainty
//!
//! Three components are kept apart (see [`UncertaintyComponent`]): the Shapley permutation
//! sampling error, the nested simulation error (the spread of independent replicate batches
//! run on the same permutations) and the mechanism-fitting error (a row bootstrap that
//! refits the mechanisms, when data are supplied and the budget allows). A component that
//! was not measured is reported as unquantified, never as zero.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use antecedent_core::{
    ComponentId, ExecutionContext, ShapleyConfig, ShapleyMode, StreamDomain, VariableId, mix64,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Admg, BitSet, DenseNodeId, GraphWorkspace};
use antecedent_model::{
    CompiledCausalModel, MechanismRegistry, MechanismSlot, ModelError, ModelView, NoiseFix,
    NoiseReplayer, NoiseSource, NoiseTable, SelectionPolicy,
};
use antecedent_stats::{Welford, student_t_ppf};

use crate::change_common::stream_tag;
use crate::error::AttributionError;
use crate::population::subset_table;
use crate::result::CacheStats;
use crate::shapley::{CoalitionPayoff, ShapleyEstimate, estimate_shapley};

/// Largest source count for exact Shapley (`2^n` coalition values, each a nested simulation).
/// [`IntrinsicAllocation::Auto`] enumerates up to this bound and samples permutations above it.
pub const INTRINSIC_MAX_EXACT_PLAYERS: usize = 10;

/// Smallest bootstrap size from which a bootstrap standard error is reported.
pub const MIN_FIT_BOOTSTRAP_REPLICATES: u32 = 10;

/// Permutations drawn by [`IntrinsicAllocation::Auto`] when the sources are too many to enumerate.
const DEFAULT_PERMUTATIONS: usize = 200;

/// Stream role of the replicate noise tables.
const NOISE_ROLE: u64 = 0x1171_0153;
/// Stream domain of the bootstrap row resampling.
const FIT_STREAM: u64 = 0x1171_F175;

const GAME: &str = "v(S) = Var(E[Y | U_S]) over exogenous noise coordinates";
const ESTIMATOR: &str = "nested Monte Carlo with the analysis-of-variance correction \
    s2(inner means) - mean(inner variances)/inner (Janzing et al. 2024, AISTATS)";

const REASON_NOT_QUANTIFIED_NO_DATA: &str = "no_data_supplied";
const REASON_NOT_REQUESTED: &str = "bootstrap_not_requested";
const REASON_OVER_BUDGET: &str = "bootstrap_exceeds_draw_budget";
const REASON_REFIT_FAILED: &str = "refit_failed_on_some_replicates";
const REASON_EXACT: &str = "every_coalition_enumerated";

/// The three sources of uncertainty in a sampled variance attribution.
///
/// They are different random mechanisms with different remedies, so every interval of an
/// [`IntrinsicInfluenceResult`] names the one component it covers and none is summed into
/// another. The separate [`SourceInfluence::whole_procedure`] status measures the whole
/// procedure directly, by its spread across full refits.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum UncertaintyComponent {
    /// Sampling of Shapley permutations, given the coalition values.
    PermutationSampling,
    /// Nested Monte Carlo noise inside the coalition values (outer and inner draws), measured
    /// by independent replicate batches.
    Simulation,
    /// Refitting the mechanisms on a finite sample, measured by a row bootstrap.
    MechanismFit,
}

/// What is known about one [`UncertaintyComponent`] of one quantity.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ComponentStatus {
    /// Quantified: a standard error and a Student-t interval at `level` for this component
    /// only.
    Quantified {
        /// Standard error of the component.
        stderr: f64,
        /// Lower end of `estimate − t·stderr`.
        ci_low: f64,
        /// Upper end of `estimate + t·stderr`.
        ci_high: f64,
        /// Nominal two-sided level of the interval.
        level: f64,
        /// Degrees of freedom of the critical value.
        df: f64,
    },
    /// The component does not arise on this execution (for example no permutation is sampled
    /// when every coalition is enumerated).
    NotApplicable {
        /// Why it does not arise.
        reason: &'static str,
    },
    /// The component arises but was not measured, and the result carries no number for it.
    /// This is not zero uncertainty.
    Unquantified {
        /// Why it was not measured.
        reason: &'static str,
    },
}

/// One uncertainty component with its status.
#[derive(Clone, Debug, PartialEq)]
pub struct ComponentUncertainty {
    /// The component the status describes.
    pub component: UncertaintyComponent,
    /// Its status.
    pub status: ComponentStatus,
}

/// A standard error that adds the quantified components in quadrature and says which
/// components it omits.
///
/// This is a convenience for comparing an estimate with a known value, not a total
/// uncertainty: it is a total only when [`Self::excluded`] is empty.
#[derive(Clone, Debug, PartialEq)]
pub struct PartialStderr {
    /// Square root of the summed squared standard errors of the included components.
    pub stderr: f64,
    /// Components included.
    pub included: Vec<UncertaintyComponent>,
    /// Components that arise but were not quantified.
    pub excluded: Vec<UncertaintyComponent>,
}

impl PartialStderr {
    /// Whether every component that arises is included.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.excluded.is_empty()
    }
}

/// Variance contribution of one noise source with its per-component uncertainty.
#[derive(Clone, Debug)]
pub struct SourceInfluence {
    /// The variable whose exogenous noise term is the source.
    pub source: ComponentId,
    /// Shapley share of the target's variance attributed to this source's noise.
    pub contribution: f64,
    /// Permutation-sampling, simulation and mechanism-fit uncertainty of
    /// [`Self::contribution`], in that order.
    pub uncertainty: Arc<[ComponentUncertainty]>,
    /// Uncertainty of the whole procedure under the declared design (independent rows):
    /// the spread of the full attribution (row resample, mechanism refit with family
    /// selection, fresh simulation batches) across the mechanism-fit bootstrap, with the
    /// permutation-sampling variance, held fixed across refits, added. Quantified only when
    /// the bootstrap ran; uncalibrated, and not a licensed interval.
    pub whole_procedure: ComponentStatus,
}

impl SourceInfluence {
    /// Status of `component`, if reported.
    #[must_use]
    pub fn status(&self, component: UncertaintyComponent) -> Option<&ComponentStatus> {
        self.uncertainty.iter().find(|u| u.component == component).map(|u| &u.status)
    }

    /// Standard error of `component`: the quantified value, exactly zero when the component
    /// does not arise (no permutation is sampled when every coalition is enumerated), and
    /// `None` when it arises but was not measured.
    #[must_use]
    pub fn component_stderr(&self, component: UncertaintyComponent) -> Option<f64> {
        match self.status(component)? {
            ComponentStatus::Quantified { stderr, .. } => Some(*stderr),
            ComponentStatus::NotApplicable { .. } => Some(0.0),
            ComponentStatus::Unquantified { .. } => None,
        }
    }

    /// Quantified components added in quadrature, with the components left out named.
    #[must_use]
    pub fn quantified_stderr(&self) -> PartialStderr {
        let mut sum_sq = 0.0;
        let (mut included, mut excluded) = (Vec::new(), Vec::new());
        for u in self.uncertainty.iter() {
            match &u.status {
                ComponentStatus::Quantified { stderr, .. } => {
                    sum_sq += stderr * stderr;
                    included.push(u.component);
                }
                ComponentStatus::Unquantified { .. } => excluded.push(u.component),
                ComponentStatus::NotApplicable { .. } => {}
            }
        }
        PartialStderr { stderr: sum_sq.sqrt(), included, excluded }
    }
}

/// How the Shapley values of an intrinsic-influence run were computed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ShapleyMethod {
    /// Every coalition enumerated.
    Exact,
    /// Random permutations of the sources.
    PermutationSampling {
        /// Permutations drawn.
        n_permutations: usize,
        /// Seed of the permutation stream.
        seed: u64,
    },
}

/// How the mechanism-fit replicates are drawn.
///
/// Both resamplings refit every mechanism on the supplied rows and repeat the whole attribution
/// on each refit; they differ in the law the rows are weighted by. The row bootstrap draws
/// multinomial row counts, reselects the mechanism families on each resample and reports a
/// sampling spread. The Dirichlet row weights are the Bayesian bootstrap: one
/// Dirichlet(1, ..., 1) weight vector per refit, the mechanism families held at the ones
/// selected on the original rows, so the replicates are draws from the posterior over the
/// mechanisms given the empirical row law.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum FitResampling {
    /// Multinomial row counts with the mechanism families reselected on each resample.
    #[default]
    RowBootstrap,
    /// One Dirichlet(1, ..., 1) weight vector per refit over the original rows, with the
    /// mechanism families held at the ones selected on the original rows.
    DirichletRowWeights,
    /// Circular blocks of `block` consecutive rows drawn with replacement, for rows that are
    /// consecutive times of one dependent series; families reselected on each resample.
    BlockBootstrap {
        /// Rows per block (at least 1).
        block: usize,
    },
    /// One Exp(1) weight per block of `block` consecutive rows, shared by every row of the
    /// block (the block Bayesian bootstrap), with the families held at the ones selected on
    /// the original rows.
    DirichletBlockWeights {
        /// Rows per block (at least 1).
        block: usize,
    },
}

impl FitResampling {
    /// Stable name of the resampling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RowBootstrap => "row_bootstrap",
            Self::DirichletRowWeights => "dirichlet_row_weights",
            Self::BlockBootstrap { .. } => "block_bootstrap",
            Self::DirichletBlockWeights { .. } => "dirichlet_block_weights",
        }
    }

    /// Whether the refits are draws of weights with the families held fixed.
    const fn holds_families(self) -> bool {
        matches!(self, Self::DirichletRowWeights | Self::DirichletBlockWeights { .. })
    }

    /// The rows of one resample of `n` rows, for the resamplings that resample rows.
    fn resample_rows(self, n: usize, rng: &mut antecedent_core::CausalRng) -> Vec<usize> {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "next_f64() is in [0, 1), so the scaled draw is a non-negative index below n"
        )]
        let mut draw = |limit: usize| ((rng.next_f64() * limit as f64) as usize).min(limit - 1);
        match self {
            Self::BlockBootstrap { block } => {
                let block = block.clamp(1, n);
                let mut rows = Vec::with_capacity(n + block);
                while rows.len() < n {
                    let start = draw(n);
                    rows.extend((0..block).map(|k| (start + k) % n));
                }
                rows.truncate(n);
                rows
            }
            _ => (0..n).map(|_| draw(n)).collect(),
        }
    }

    /// The weights of one refit of `n` rows, for the resamplings that hold the families.
    fn refit_weights(self, n: usize, rng: &mut antecedent_core::CausalRng) -> Vec<f64> {
        match self {
            Self::DirichletBlockWeights { block } => {
                let block = block.clamp(1, n);
                let mut weights = Vec::with_capacity(n);
                while weights.len() < n {
                    let weight = antecedent_kernels::bayesian_bootstrap_weights(rng, 1, false)[0];
                    let take = block.min(n - weights.len());
                    weights.extend(std::iter::repeat_n(weight, take));
                }
                weights
            }
            _ => antecedent_kernels::bayesian_bootstrap_weights(rng, n, false),
        }
    }
}

/// Method record of an intrinsic-influence run.
#[derive(Clone, Debug, PartialEq)]
pub struct IntrinsicMethod {
    /// Coalition game, `v(S) = Var(E[Y | U_S])`.
    pub game: &'static str,
    /// Estimator of `v(S)`, with its reference.
    pub estimator: &'static str,
    /// How the game was allocated.
    pub shapley: ShapleyMethod,
    /// Outer draws (coalition coordinates) per replicate batch.
    pub outer_draws: usize,
    /// Inner draws per outer draw.
    pub inner_draws: usize,
    /// Independent replicate batches.
    pub replicates: usize,
    /// Bootstrap refits requested for mechanism-fit uncertainty (`0` when none).
    pub fit_bootstrap_replicates: u32,
    /// How those refits weighted the rows.
    pub fit_resampling: FitResampling,
}

/// Compute consumed by an intrinsic-influence run.
#[derive(Clone, Debug, Default)]
pub struct IntrinsicBudget {
    /// Distinct coalition values simulated (each over every replicate batch), whatever the
    /// execution context's coalition-cache policy: the replicate memo never simulates a
    /// coalition twice.
    pub coalition_evaluations: u64,
    /// Rows of the compiled model replayed for those values.
    pub rows_replayed: u64,
    /// Permutations sampled (`0` when every coalition was enumerated).
    pub permutations: u64,
    /// Coalitions enumerated by exact Shapley (`0` when sampled).
    pub exact_coalitions: u64,
    /// Bootstrap refits that completed (each a full attribution on resampled rows).
    pub refits: u32,
}

/// Intrinsic causal influence of upstream noise terms on a target's variance.
#[derive(Clone, Debug)]
pub struct IntrinsicInfluenceResult {
    /// Target variable.
    pub target: VariableId,
    /// Per-source contributions in topological order.
    pub sources: Arc<[SourceInfluence]>,
    /// Variance of the target, estimated directly from the replicate batches.
    pub total_variance: f64,
    /// Simulation standard error of [`Self::total_variance`].
    pub total_variance_stderr: f64,
    /// `total_variance − Σ contributions`. The game has `v(∅) = 0` and `v(all) = Var(Y)`, so
    /// the true value is zero and this is the efficiency gap left by estimation error.
    pub allocation_residual: f64,
    /// Simulation standard error of [`Self::allocation_residual`] (replicate spread).
    pub allocation_residual_stderr: f64,
    /// Nominal level of every interval in [`Self::sources`].
    pub level: f64,
    /// Method record.
    pub method: IntrinsicMethod,
    /// Compute consumed.
    pub budget: IntrinsicBudget,
    /// Coalition cache statistics of the point run.
    pub cache_stats: CacheStats,
    /// The attribution of every completed mechanism refit, one row per refit and one entry per
    /// source in [`Self::sources`] order. Under [`FitResampling::DirichletRowWeights`] these
    /// are draws from the posterior over the mechanisms; empty when no refit ran.
    pub fit_draws: Arc<[Vec<f64>]>,
}

impl IntrinsicInfluenceResult {
    /// Sum of the point contributions.
    #[must_use]
    pub fn contribution_sum(&self) -> f64 {
        self.sources.iter().map(|s| s.contribution).sum()
    }
}

/// How the noise terms of the model are known to relate to one another.
///
/// Intrinsic influence is defined for structural models whose exogenous noises are mutually
/// independent. A compiled model is a DAG and cannot say so by itself, so the caller must
/// declare it; without a declaration the query is refused.
#[derive(Clone, Copy, Debug)]
pub enum NoiseIndependence<'a> {
    /// No declaration was made; the query is refused.
    Undeclared,
    /// The caller declares the model Markovian: every mechanism's noise is independent of
    /// the others' (no latent common cause of two variables).
    Declared,
    /// The structure the model was built from, which may have bidirected edges (latent common
    /// causes). It must share the model's directed edges, and any bidirected edge refuses the
    /// query.
    Structure(&'a Admg),
}

/// How the variance game is allocated among the sources.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntrinsicAllocation {
    /// Exact up to [`INTRINSIC_MAX_EXACT_PLAYERS`] sources, permutation sampling above.
    Auto,
    /// Exact Shapley over every coalition; refused above [`INTRINSIC_MAX_EXACT_PLAYERS`].
    Exact,
    /// Shapley by permutation sampling.
    Permutation {
        /// Permutations to draw (at least 2, so a standard error exists).
        n_permutations: usize,
    },
}

/// Typed intrinsic-causal-influence query.
#[derive(Clone, Debug, PartialEq)]
pub struct IntrinsicInfluenceQuery {
    /// Variable whose variance is attributed.
    pub target: VariableId,
    /// Allocation rule.
    pub allocation: IntrinsicAllocation,
    /// Outer noise draws per replicate batch (at least 2).
    pub outer_draws: usize,
    /// Inner draws per outer draw (at least 2, so the inner variance exists).
    pub inner_draws: usize,
    /// Independent replicate batches (at least 2, so the simulation error exists).
    pub replicates: usize,
    /// Row-bootstrap refits for mechanism-fitting uncertainty; `0` for none, otherwise at
    /// least [`MIN_FIT_BOOTSTRAP_REPLICATES`]. Needs data passed to [`intrinsic_influence`].
    pub fit_bootstrap_replicates: u32,
    /// How the mechanism-fit refits weight the rows.
    pub fit_resampling: FitResampling,
    /// Two-sided level of every reported interval, in `(0, 1)`.
    pub level: f64,
    /// Most sources accepted (hard size guard).
    pub max_sources: usize,
    /// Seed of the noise draws and the permutation stream.
    pub seed: u64,
}

impl IntrinsicInfluenceQuery {
    /// Attribute the variance of `target`, with automatic allocation, 200 outer by 40 inner
    /// draws in 4 replicate batches, no bootstrap and 95% intervals.
    #[must_use]
    pub fn new(target: VariableId) -> Self {
        Self {
            target,
            allocation: IntrinsicAllocation::Auto,
            outer_draws: 200,
            inner_draws: 40,
            replicates: 4,
            fit_bootstrap_replicates: 0,
            fit_resampling: FitResampling::RowBootstrap,
            level: 0.95,
            max_sources: 64,
            seed: 0,
        }
    }

    /// Set the allocation rule.
    #[must_use]
    pub const fn with_allocation(mut self, allocation: IntrinsicAllocation) -> Self {
        self.allocation = allocation;
        self
    }

    /// Set the nested simulation size: outer draws, inner draws and replicate batches.
    #[must_use]
    pub const fn with_simulation(
        mut self,
        outer_draws: usize,
        inner_draws: usize,
        replicates: usize,
    ) -> Self {
        self.outer_draws = outer_draws;
        self.inner_draws = inner_draws;
        self.replicates = replicates;
        self
    }

    /// Request mechanism-fit uncertainty from `replicates` row-bootstrap refits.
    #[must_use]
    pub const fn with_fit_bootstrap(mut self, replicates: u32) -> Self {
        self.fit_bootstrap_replicates = replicates;
        self
    }

    /// Set how the mechanism-fit refits weight the rows.
    #[must_use]
    pub const fn with_fit_resampling(mut self, resampling: FitResampling) -> Self {
        self.fit_resampling = resampling;
        self
    }

    /// Set the interval level.
    #[must_use]
    pub const fn with_level(mut self, level: f64) -> Self {
        self.level = level;
        self
    }

    /// Cap the number of sources.
    #[must_use]
    pub const fn with_max_sources(mut self, max_sources: usize) -> Self {
        self.max_sources = max_sources;
        self
    }

    /// Set the seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validate the sizes, level and permutation count.
    ///
    /// # Errors
    ///
    /// A coded `invalid_argument` refusal.
    pub fn validate(&self) -> Result<(), AttributionError> {
        let bad = |message: &'static str| {
            Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "intrinsic_influence.invalid_settings",
                message,
            ))
        };
        if self.outer_draws < 2 || self.inner_draws < 2 || self.replicates < 2 {
            return bad("outer draws, inner draws and replicate batches must each be at least 2");
        }
        if !(self.level > 0.0 && self.level < 1.0) {
            return bad("level must lie strictly between 0 and 1");
        }
        if self.max_sources == 0 {
            return bad("max_sources must be positive");
        }
        if matches!(self.allocation, IntrinsicAllocation::Permutation { n_permutations } if n_permutations < 2)
        {
            return bad("permutation sampling needs at least 2 permutations");
        }
        if self.fit_bootstrap_replicates != 0
            && self.fit_bootstrap_replicates < MIN_FIT_BOOTSTRAP_REPLICATES
        {
            return bad("a fit bootstrap needs at least 10 replicates");
        }
        Ok(())
    }
}

/// A typed refusal: the registered reason code, the namespaced detail, and the message.
fn refuse(
    code: &'static str,
    detail: &'static str,
    message: impl std::fmt::Display,
) -> AttributionError {
    AttributionError::Message(format!(
        "{}{code}: {detail}: {message}",
        antecedent_core::reason_code::PREFIX
    ))
}

/// Attribute the variance of `query.target` to the exogenous noise terms of its ancestors and
/// itself.
///
/// `noise` states what is known about dependence among the noise terms; `data`, when supplied
/// and `query.fit_bootstrap_replicates > 0`, is the sample the mechanisms were fitted on and is
/// resampled to measure mechanism-fitting uncertainty (each refit selects mechanisms with the
/// standard registry). Without data that component is reported as unquantified.
///
/// # Errors
///
/// Coded refusals for undeclared or correlated noise (bidirected edges, latent common
/// causes), an unfitted mechanism or one whose noise is not independent across rows, a
/// non-continuous target, too many sources or an oversized simulation plan; invalid settings;
/// model or replay failures; [`AttributionError::Cancelled`].
#[allow(
    clippy::too_many_lines,
    reason = "one pipeline from validation through the three uncertainty components to the result record"
)]
pub fn intrinsic_influence(
    model: &CompiledCausalModel,
    noise: NoiseIndependence<'_>,
    data: Option<&TabularData>,
    query: &IntrinsicInfluenceQuery,
    ctx: &ExecutionContext,
) -> Result<IntrinsicInfluenceResult, AttributionError> {
    query.validate()?;
    require_independent_noise(model, noise)?;
    let target = model
        .dense_of(query.target)
        .ok_or_else(|| AttributionError::missing_var("target", query.target))?;
    let (players, player_nodes) = noise_sources(model, target, query.max_sources)?;
    require_noise_support(model, target, &player_nodes)?;

    let (config, shapley) = allocation_config(query, players.len())?;
    let rows_per_batch = query
        .outer_draws
        .checked_mul(query.inner_draws)
        .ok_or_else(|| over_budget("the outer times inner draws overflow"))?;
    let draws_per_coalition = rows_per_batch
        .checked_mul(query.replicates)
        .ok_or_else(|| over_budget("the draws per coalition value overflow"))?;
    plan_within_budget(players.len(), config, draws_per_coalition)?;

    let spec = Game {
        players: &players,
        player_nodes: &player_nodes,
        target: target.as_usize(),
        outer: query.outer_draws,
        inner: query.inner_draws,
        replicates: query.replicates,
        seed: query.seed,
        config,
    };
    let point = run_game(model, &spec, 0, true, ctx)?;
    let Decomposed { pooled, per_replicate, direct_variance, evaluations } = point;
    let cache_stats = pooled.cache_stats.clone();
    let b = query.replicates as f64;

    let sim_se = |series: &[f64]| -> f64 {
        let mut w = Welford::new();
        series.iter().for_each(|&x| w.push(x));
        w.stderr_of_mean()
    };
    let sources_sim: Vec<f64> = (0..players.len())
        .map(|i| sim_se(&per_replicate.iter().map(|phi| phi[i]).collect::<Vec<_>>()))
        .collect();
    let total_variance = direct_variance.iter().sum::<f64>() / b;
    let residuals: Vec<f64> = direct_variance
        .iter()
        .zip(&per_replicate)
        .map(|(total, phi)| total - phi.iter().sum::<f64>())
        .collect();
    let allocation_residual = residuals.iter().sum::<f64>() / b;

    let fit = fit_component(model, data, query, &spec, &pooled, &sources_sim, ctx)?;

    let tail = 0.5 + 0.5 * query.level;
    let sim_df = b - 1.0;
    let sim_t = student_t_ppf(tail, sim_df);
    let perm_df = pooled.budget.samples.saturating_sub(1) as f64;
    let perm_t = student_t_ppf(tail, perm_df);
    let quantified = |estimate: f64, stderr: f64, t: f64, df: f64| ComponentStatus::Quantified {
        stderr,
        ci_low: estimate - t * stderr,
        ci_high: estimate + t * stderr,
        level: query.level,
        df,
    };
    let sources: Vec<SourceInfluence> = pooled
        .players
        .iter()
        .enumerate()
        .map(|(i, &source)| {
            let contribution = pooled.values[i];
            let permutation = match pooled.component_mc_stderr.as_ref() {
                Some(se) => quantified(contribution, se[i], perm_t, perm_df),
                None => ComponentStatus::NotApplicable { reason: REASON_EXACT },
            };
            let (mechanism_fit, whole_procedure) = match &fit {
                FitComponent::Quantified { stderr, whole, t, df, .. } => (
                    quantified(contribution, stderr[i], *t, *df),
                    quantified(contribution, whole[i], *t, *df),
                ),
                FitComponent::Unquantified(reason) => (
                    ComponentStatus::Unquantified { reason },
                    ComponentStatus::Unquantified { reason },
                ),
            };
            SourceInfluence {
                source,
                contribution,
                uncertainty: Arc::from(vec![
                    ComponentUncertainty {
                        component: UncertaintyComponent::PermutationSampling,
                        status: permutation,
                    },
                    ComponentUncertainty {
                        component: UncertaintyComponent::Simulation,
                        status: quantified(contribution, sources_sim[i], sim_t, sim_df),
                    },
                    ComponentUncertainty {
                        component: UncertaintyComponent::MechanismFit,
                        status: mechanism_fit,
                    },
                ]),
                whole_procedure,
            }
        })
        .collect();

    let refits = match &fit {
        FitComponent::Quantified { .. } => query.fit_bootstrap_replicates,
        FitComponent::Unquantified(_) => 0,
    };
    let fit_draws: Arc<[Vec<f64>]> = match fit {
        FitComponent::Quantified { draws, .. } => Arc::from(draws),
        FitComponent::Unquantified(_) => Arc::from(Vec::<Vec<f64>>::new()),
    };
    Ok(IntrinsicInfluenceResult {
        target: query.target,
        sources: Arc::from(sources),
        total_variance,
        total_variance_stderr: sim_se(&direct_variance),
        allocation_residual,
        allocation_residual_stderr: sim_se(&residuals),
        level: query.level,
        method: IntrinsicMethod {
            game: GAME,
            estimator: ESTIMATOR,
            shapley,
            outer_draws: query.outer_draws,
            inner_draws: query.inner_draws,
            replicates: query.replicates,
            fit_bootstrap_replicates: query.fit_bootstrap_replicates,
            fit_resampling: query.fit_resampling,
        },
        budget: IntrinsicBudget {
            coalition_evaluations: evaluations,
            rows_replayed: evaluations
                .saturating_mul(rows_per_batch as u64)
                .saturating_mul(query.replicates as u64),
            permutations: pooled.budget.samples,
            exact_coalitions: pooled.budget.exact_coalitions,
            refits,
        },
        cache_stats,
        fit_draws,
    })
}

fn over_budget(message: &'static str) -> AttributionError {
    refuse(
        antecedent_core::reason_code!("route_not_supported"),
        "intrinsic_influence.draw_budget_exceeded",
        message,
    )
}

/// Refuse a model whose noise terms are not declared independent.
fn require_independent_noise(
    model: &CompiledCausalModel,
    noise: NoiseIndependence<'_>,
) -> Result<(), AttributionError> {
    match noise {
        NoiseIndependence::Undeclared => Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "intrinsic_influence.noise_independence_undeclared",
            "intrinsic influence is defined for independent exogenous noise terms; declare the \
             model Markovian or pass its structure",
        )),
        NoiseIndependence::Declared => Ok(()),
        NoiseIndependence::Structure(admg) => {
            if let Some((a, b)) = admg.bidirected_edge_pairs_sorted().first() {
                return Err(refuse(
                    antecedent_core::reason_code!("route_not_supported"),
                    "intrinsic_influence.correlated_noise",
                    format_args!(
                        "the noise terms of variables {a} and {b} are connected by a bidirected \
                         edge (a latent common cause), so they are not independent and the \
                         variance game is not defined"
                    ),
                ));
            }
            if admg.node_count() != model.graph.node_count()
                || admg.directed_edge_pairs_sorted() != model.graph.directed_edge_pairs_sorted()
            {
                return Err(refuse(
                    antecedent_core::reason_code!("invalid_argument"),
                    "intrinsic_influence.structure_mismatch",
                    "the declared structure does not have the model's directed edges",
                ));
            }
            Ok(())
        }
    }
}

/// The target's ancestors and the target, in topological order, with their dense indices.
fn noise_sources(
    model: &CompiledCausalModel,
    target: DenseNodeId,
    max_sources: usize,
) -> Result<(Vec<ComponentId>, Vec<usize>), AttributionError> {
    let mut ws = GraphWorkspace::default();
    let mut ancestors = BitSet::default();
    model.graph.ancestors_of(&[target], &mut ancestors, &mut ws);
    let mut players = Vec::new();
    let mut nodes = Vec::new();
    for gather in model.parent_gathers.iter() {
        let node = gather.child;
        if node == target || ancestors.contains(node) {
            players
                .push(ComponentId::from_variable(model.output_layout.variables[node.as_usize()]));
            nodes.push(node.as_usize());
        }
    }
    if players.len() > max_sources {
        return Err(AttributionError::SizeLimit {
            kind: "noise sources",
            requested: players.len(),
            max: max_sources,
        });
    }
    Ok((players, nodes))
}

/// Refuse mechanisms whose noise the nested simulation cannot draw as independent rows.
fn require_noise_support(
    model: &CompiledCausalModel,
    target: DenseNodeId,
    player_nodes: &[usize],
) -> Result<(), AttributionError> {
    for &node in player_nodes {
        match model.mechanisms.get(DenseNodeId::from_index(node)) {
            MechanismSlot::Vacant | MechanismSlot::Pending { .. } => {
                return Err(refuse(
                    antecedent_core::reason_code!("route_not_supported"),
                    "intrinsic_influence.mechanism_unfitted",
                    "a source has no fitted mechanism, so its noise cannot be drawn",
                ));
            }
            MechanismSlot::LinearGaussianStateSpace { .. }
            | MechanismSlot::ConditionalLinearGaussianStateSpace { .. } => {
                return Err(refuse(
                    antecedent_core::reason_code!("route_not_supported"),
                    "intrinsic_influence.noise_not_row_independent",
                    "a state-space mechanism's noise is serially dependent across rows, so \
                     outer and inner draws are not independent",
                ));
            }
            MechanismSlot::Discrete { .. }
            | MechanismSlot::DiscreteBasis { .. }
            | MechanismSlot::TreeCategorical { .. }
                if node == target.as_usize() =>
            {
                return Err(refuse(
                    antecedent_core::reason_code!("route_not_supported"),
                    "intrinsic_influence.target_not_continuous",
                    "the variance of a categorical target's level codes is not a meaningful \
                     quantity to attribute",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

/// The Shapley configuration and its record for the query's allocation and source count.
fn allocation_config(
    query: &IntrinsicInfluenceQuery,
    n_players: usize,
) -> Result<(ShapleyConfig, ShapleyMethod), AttributionError> {
    let exact = match query.allocation {
        IntrinsicAllocation::Auto => n_players <= INTRINSIC_MAX_EXACT_PLAYERS,
        IntrinsicAllocation::Exact => {
            if n_players > INTRINSIC_MAX_EXACT_PLAYERS {
                return Err(refuse(
                    antecedent_core::reason_code!("route_not_supported"),
                    "intrinsic_influence.exact_players_exceed_bound",
                    "exact Shapley enumerates 2^n nested simulations; use permutation sampling \
                     above the bound",
                ));
            }
            true
        }
        IntrinsicAllocation::Permutation { .. } => false,
    };
    if exact {
        return Ok((ShapleyConfig::exact().with_seed(query.seed), ShapleyMethod::Exact));
    }
    let n_permutations = match query.allocation {
        IntrinsicAllocation::Permutation { n_permutations } => n_permutations,
        _ => DEFAULT_PERMUTATIONS,
    };
    Ok((
        ShapleyConfig::permutation(n_permutations).with_seed(query.seed),
        ShapleyMethod::PermutationSampling { n_permutations, seed: query.seed },
    ))
}

/// Refuse a plan whose coalition values times draws per value exceed the shared ceiling.
///
/// Coalition values are memoized per mask in [`ReplicateBank`], so a sampled allocation
/// never evaluates more than the `2^k` distinct coalitions of the game: when that is fewer
/// than the permutation plan visits, the exact count is the one charged.
fn plan_within_budget(
    n_players: usize,
    config: ShapleyConfig,
    draws_per_coalition: usize,
) -> Result<(), AttributionError> {
    let sampled_visits = match config.mode {
        ShapleyMode::MonteCarlo { n_samples }
        | ShapleyMode::Permutation { n_permutations: n_samples } => {
            Some((n_samples as u128).saturating_mul(n_players as u128 + 1))
        }
        _ => None,
    };
    let distinct = (n_players < 64).then(|| 1u128 << n_players);
    let config = match (sampled_visits, distinct) {
        (Some(visits), Some(distinct)) if distinct <= visits => ShapleyConfig::exact(),
        _ => config,
    };
    crate::shapley::check_coalition_sample_budget(
        n_players,
        &antecedent_core::AllocationMethod::Shapley { approximation: config },
        draws_per_coalition,
    )
    .map_err(|_| over_budget("planned coalition values times draws exceed the simulation ceiling"))
}

/// Immutable description of one game.
struct Game<'a> {
    players: &'a [ComponentId],
    /// Dense node of each player.
    player_nodes: &'a [usize],
    target: usize,
    outer: usize,
    inner: usize,
    replicates: usize,
    seed: u64,
    config: ShapleyConfig,
}

/// One replicate batch: the two coalition noise tables and the replay scratch.
struct BatchState {
    fixed: NoiseTable,
    resampled: NoiseTable,
    replayer: NoiseReplayer,
    values: Vec<f64>,
}

/// Replicate batches and the memo of coalition values over them.
struct ReplicateBank<'a> {
    spec: &'a Game<'a>,
    view: ModelView<'a>,
    n_nodes: usize,
    batches: Vec<Mutex<BatchState>>,
    /// Coalition mask -> value in each replicate batch.
    memo: HashMap<u64, Vec<f64>>,
    /// Variance of the target over all rows of each batch with no coordinate fixed.
    direct_variance: Option<Vec<f64>>,
    ctx: &'a ExecutionContext,
}

/// The draw index of a replicate's noise table: never `0`, which is the sampling family.
fn noise_draw(seed: u64, replicate: u64, role: u64) -> u64 {
    mix64(mix64(seed ^ NOISE_ROLE) ^ ((replicate << 1) | role)).max(1)
}

impl<'a> ReplicateBank<'a> {
    /// `first_replicate` offsets the global replicate index so bootstrap refits draw noise
    /// independent of the point run's.
    fn new(
        model: &'a CompiledCausalModel,
        spec: &'a Game<'a>,
        first_replicate: u64,
        ctx: &'a ExecutionContext,
    ) -> Result<Self, AttributionError> {
        let view = ModelView::observational(model);
        let n_nodes = model.n_nodes();
        let n_rows = spec.outer * spec.inner;
        let all_nodes: Vec<DenseNodeId> = DenseNodeId::all(n_nodes).collect();
        let player_ids: Vec<DenseNodeId> =
            spec.player_nodes.iter().map(|&n| DenseNodeId::from_index(n)).collect();
        let mut batches = Vec::with_capacity(spec.replicates);
        for b in 0..spec.replicates {
            let replicate = first_replicate + b as u64;
            let base = NoiseTable::sample(&view, n_rows, ctx)?;
            let mut fixed = base.clone();
            fixed.randomize(&view, &all_nodes, noise_draw(spec.seed, replicate, 0), ctx)?;
            let mut resampled = base;
            resampled.randomize(&view, &all_nodes, noise_draw(spec.seed, replicate, 1), ctx)?;
            // Outer draw i is the first row of its group; hold it on every row of the group.
            let mut held = Vec::with_capacity(spec.player_nodes.len() * n_rows);
            for &node in spec.player_nodes {
                let column = fixed.column(node)?;
                for i in 0..spec.outer {
                    held.extend(std::iter::repeat_n(column[i * spec.inner], spec.inner));
                }
            }
            fixed.fix(model, &player_ids, NoiseFix::Columns(&held))?;
            let values = vec![0.0; n_rows * n_nodes];
            batches.push(Mutex::new(BatchState {
                fixed,
                resampled,
                replayer: NoiseReplayer::new(),
                values,
            }));
        }
        Ok(Self { spec, view, n_nodes, batches, memo: HashMap::new(), direct_variance: None, ctx })
    }

    /// `v̂(mask)` in every replicate batch, replaying each batch once on a miss.
    fn values(&mut self, mask: u64) -> Result<&[f64], AttributionError> {
        if self.memo.contains_key(&mask) {
            return Ok(self.memo[&mask].as_slice());
        }
        let mut fixed_nodes = vec![false; self.n_nodes];
        for (bit, &node) in self.spec.player_nodes.iter().enumerate() {
            fixed_nodes[node] = (mask >> bit) & 1 == 1;
        }
        let (outer, inner, target) = (self.spec.outer, self.spec.inner, self.spec.target);
        let n_rows = outer * inner;
        let evaluated = self.ctx.map_indexed(
            self.batches.len(),
            |b, worker| -> Result<(f64, f64), AttributionError> {
                let mut guard = self.batches[b]
                    .lock()
                    .map_err(|_| AttributionError::MissingArtifact("replicate batch poisoned"))?;
                let BatchState { fixed, resampled, replayer, values } = &mut *guard;
                let source = NoiseSource::coalition(fixed, resampled, &fixed_nodes)?;
                let outcome = replayer.replay_into(&self.view, source, 0, values, worker)?;
                if outcome.cancelled {
                    return Err(AttributionError::Cancelled);
                }
                let y = &values[target * n_rows..(target + 1) * n_rows];
                if y.iter().any(|v| !v.is_finite()) {
                    return Err(AttributionError::Model(ModelError::Numerical {
                        message: "the target produced a non-finite value".into(),
                    }));
                }
                let direct = if mask == 0 { variance_of(y) } else { f64::NAN };
                Ok((nested_variance(y, outer, inner), direct))
            },
        )?;
        if mask == 0 {
            self.direct_variance = Some(evaluated.iter().map(|e| e.1).collect());
        }
        let values =
            self.memo.entry(mask).or_insert_with(|| evaluated.into_iter().map(|e| e.0).collect());
        Ok(values.as_slice())
    }
}

/// Bessel-corrected sample variance.
fn variance_of(y: &[f64]) -> f64 {
    let mut w = Welford::new();
    y.iter().for_each(|&x| w.push(x));
    w.sample_variance().unwrap_or(f64::NAN)
}

/// The analysis-of-variance estimate of `Var(E[Y | U_S])` from `outer` groups of `inner`
/// consecutive rows: the sample variance of the group means minus the mean within-group
/// sample variance over `inner`.
fn nested_variance(y: &[f64], outer: usize, inner: usize) -> f64 {
    let mut means = Welford::new();
    let mut within = 0.0;
    for group in y.chunks_exact(inner) {
        let mut w = Welford::new();
        group.iter().for_each(|&x| w.push(x));
        means.push(w.mean());
        within += w.sample_variance().unwrap_or(0.0);
    }
    let between = means.sample_variance().unwrap_or(f64::NAN);
    between - within / outer as f64 / inner as f64
}

/// `v(S)` averaged over all replicate batches.
struct PooledPayoff<'b, 'a>(&'b mut ReplicateBank<'a>);

impl CoalitionPayoff for PooledPayoff<'_, '_> {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        let values = self.0.values(mask)?;
        Ok(values.iter().sum::<f64>() / values.len() as f64)
    }
}

/// `v(S)` from one replicate batch (a lookup after the pooled run has evaluated the mask).
struct BatchPayoff<'b, 'a> {
    bank: &'b mut ReplicateBank<'a>,
    batch: usize,
}

impl CoalitionPayoff for BatchPayoff<'_, '_> {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        Ok(self.bank.values(mask)?[self.batch])
    }
}

/// Shapley values from the pooled coalition values, and from each batch alone on the same
/// permutations.
struct Decomposed {
    pooled: ShapleyEstimate,
    /// Per batch, the Shapley values (empty when not decomposed).
    per_replicate: Vec<Vec<f64>>,
    /// Direct variance of the target in each batch.
    direct_variance: Vec<f64>,
    /// Distinct coalition values simulated.
    evaluations: u64,
}

/// Run the game: pooled Shapley, and when `decompose`, the same Shapley on each batch alone.
///
/// The per-batch runs share the pooled run's permutations and read the memoized coalition
/// values, so their spread is the simulation error with the permutation sample held fixed.
fn run_game(
    model: &CompiledCausalModel,
    spec: &Game<'_>,
    first_replicate: u64,
    decompose: bool,
    ctx: &ExecutionContext,
) -> Result<Decomposed, AttributionError> {
    let mut bank = ReplicateBank::new(model, spec, first_replicate, ctx)?;
    let pooled = estimate_shapley(spec.players, &spec.config, &mut PooledPayoff(&mut bank), ctx)?;
    if bank.direct_variance.is_none() {
        bank.values(0)?;
    }
    let direct_variance = bank.direct_variance.clone().unwrap_or_default();
    let evaluations = bank.memo.len() as u64;
    let mut per_replicate = Vec::new();
    if decompose {
        for batch in 0..spec.replicates {
            let mut payoff = BatchPayoff { bank: &mut bank, batch };
            per_replicate
                .push(estimate_shapley(spec.players, &spec.config, &mut payoff, ctx)?.values);
        }
    }
    Ok(Decomposed { pooled, per_replicate, direct_variance, evaluations })
}

/// What the mechanism-fit bootstrap produced.
enum FitComponent {
    Quantified {
        /// Per-source standard error after removing the Monte Carlo noise of one refit.
        stderr: Vec<f64>,
        /// Per-source standard error of the whole procedure: the raw bootstrap spread with
        /// the permutation variance (fixed across refits) added.
        whole: Vec<f64>,
        t: f64,
        df: f64,
        /// The attribution of every refit, one row per refit.
        draws: Vec<Vec<f64>>,
    },
    Unquantified(&'static str),
}

/// Mechanism-fitting uncertainty: refit the mechanisms on row-bootstrap resamples of `data`
/// (or under Dirichlet row weights, for the posterior over the mechanisms) and repeat the
/// attribution on each refit.
///
/// Under [`FitResampling::DirichletRowWeights`] the standard errors are the posterior standard
/// deviations of the contributions and the interval their normal approximation; the draws
/// themselves are returned for the caller to summarise as a posterior.
///
/// Each refit's contributions carry the same Monte Carlo noise as the point estimate, so the
/// bootstrap variance is reduced by the noise variance of one estimate (the pooled
/// permutation variance when sampled, else the simulation variance) before the square root.
/// Any failed refit leaves the component unquantified rather than dropping the failure.
fn fit_component(
    model: &CompiledCausalModel,
    data: Option<&TabularData>,
    query: &IntrinsicInfluenceQuery,
    spec: &Game<'_>,
    pooled: &ShapleyEstimate,
    sources_sim: &[f64],
    ctx: &ExecutionContext,
) -> Result<FitComponent, AttributionError> {
    let replicates = query.fit_bootstrap_replicates;
    if replicates == 0 {
        return Ok(FitComponent::Unquantified(REASON_NOT_REQUESTED));
    }
    let Some(data) = data else {
        return Ok(FitComponent::Unquantified(REASON_NOT_QUANTIFIED_NO_DATA));
    };
    let per_coalition = spec.outer * spec.inner * spec.replicates;
    let refits = 1 + replicates as usize;
    if plan_within_budget(spec.players.len(), spec.config, per_coalition.saturating_mul(refits))
        .is_err()
    {
        return Ok(FitComponent::Unquantified(REASON_OVER_BUDGET));
    }
    let n_rows = data.row_count();
    if n_rows == 0 {
        return Ok(FitComponent::Unquantified(REASON_NOT_QUANTIFIED_NO_DATA));
    }
    // The posterior over the mechanisms holds the families at the ones selected on the
    // original rows, so they are selected once here.
    let held_assignments = if query.fit_resampling.holds_families() {
        match MechanismRegistry::standard().assign_and_fit_polled(
            model,
            data,
            SelectionPolicy::BestScore,
            &|| ctx.cancellation.is_cancelled(),
        ) {
            Ok((_, assignments)) => Some(assignments),
            Err(ModelError::Cancelled) => return Err(AttributionError::Cancelled),
            Err(_) => return Ok(FitComponent::Unquantified(REASON_REFIT_FAILED)),
        }
    } else {
        None
    };
    let outcomes = ctx.map_indexed(
        replicates as usize,
        |r, worker| -> Result<Option<Vec<f64>>, AttributionError> {
            if worker.cancellation.is_cancelled() {
                return Err(AttributionError::Cancelled);
            }
            let mut rng = worker.rng.stream_for(
                StreamDomain::Attribution,
                stream_tag(FIT_STREAM, antecedent_core::derive_stream(query.seed, r as u64)),
            );
            let attempt = (|| -> Result<Vec<f64>, AttributionError> {
                let store = if let Some(assignments) = &held_assignments {
                    let weights = query.fit_resampling.refit_weights(n_rows, &mut rng);
                    MechanismRegistry::standard().refit_weighted_polled(
                        model,
                        data,
                        assignments,
                        &weights,
                        &|| worker.cancellation.is_cancelled(),
                    )?
                } else {
                    let rows = query.fit_resampling.resample_rows(n_rows, &mut rng);
                    let resample = subset_table(data, &rows)?;
                    MechanismRegistry::standard()
                        .assign_and_fit_polled(
                            model,
                            &resample,
                            SelectionPolicy::BestScore,
                            &|| worker.cancellation.is_cancelled(),
                        )?
                        .0
                };
                let refit = model.clone().with_mechanisms(store);
                require_noise_support(
                    &refit,
                    DenseNodeId::from_index(spec.target),
                    spec.player_nodes,
                )?;
                let first = (r as u64 + 1) * spec.replicates as u64;
                run_game(&refit, spec, first, false, worker).map(|d| d.pooled.values)
            })();
            match attempt {
                Ok(values) => Ok(Some(values)),
                Err(
                    AttributionError::Cancelled | AttributionError::Model(ModelError::Cancelled),
                ) => Err(AttributionError::Cancelled),
                Err(_) => Ok(None),
            }
        },
    )?;
    if outcomes.iter().any(Option::is_none) {
        return Ok(FitComponent::Unquantified(REASON_REFIT_FAILED));
    }
    let draws: Vec<&Vec<f64>> = outcomes.iter().flatten().collect();
    let whole = (0..spec.players.len())
        .map(|i| {
            let mut w = Welford::new();
            draws.iter().for_each(|d| w.push(d[i]));
            let boot_var = w.sample_variance().unwrap_or(f64::NAN);
            let permutation_var =
                pooled.component_mc_stderr.as_ref().map_or(0.0, |se| se[i] * se[i]);
            (boot_var + permutation_var).sqrt()
        })
        .collect();
    let stderr = (0..spec.players.len())
        .map(|i| {
            let mut w = Welford::new();
            draws.iter().for_each(|d| w.push(d[i]));
            let boot_var = w.sample_variance().unwrap_or(f64::NAN);
            let noise_var =
                pooled.component_mc_stderr.as_ref().map_or(sources_sim[i] * sources_sim[i], |se| {
                    (se[i] * se[i]).max(sources_sim[i] * sources_sim[i])
                });
            (boot_var - noise_var).max(0.0).sqrt()
        })
        .collect();
    let df = f64::from(replicates) - 1.0;
    let draws: Vec<Vec<f64>> = draws.into_iter().cloned().collect();
    Ok(FitComponent::Quantified {
        stderr,
        whole,
        t: student_t_ppf(0.5 + 0.5 * query.level, df),
        df,
        draws,
    })
}
