//! Batched Monte Carlo design ranking.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::too_many_lines
)]

use std::sync::Arc;

use antecedent_core::{
    CausalRng, EnvironmentId, ExecutionContext, ModelId, MonteCarloBudget, MonteCarloError,
    QueryId, StreamDomain, VariableId,
};
use antecedent_kernels::sample_categorical;
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};
use antecedent_stats::{Welford, invert_square, normal_ppf};

use crate::candidate::{CandidateDesign, DesignCost};
use crate::decision::DecisionProblem;
use crate::error::DesignError;
use crate::objective::DesignObjective;
use crate::preposterior::{DecisionPrior, DecisionSignal, PreposteriorAnalysis};
use crate::result::{ConstraintViolation, DesignRanking, RankedCandidate, ScoreEvaluation};

/// Hard resource limits applied before scoring (violations are recorded, not silent).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DesignConstraints {
    /// Maximum allowed cost amount (`None` = unbounded).
    pub max_cost: Option<f64>,
    /// Maximum sample-budget consumption (`None` = unbounded).
    pub max_sample_budget: Option<u64>,
}

/// Monte Carlo ranking configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignRankConfig {
    /// Minimum MC batches before considering early stop.
    pub min_batches: u32,
    /// Maximum MC batches (each batch evaluates all active candidates once under shared CRN).
    pub max_batches: u32,
    /// Stop when the sequentially adjusted CI half-width of every top-3 adjacent pair
    /// difference is below this, or when that pair is separated by more than its
    /// adjusted half-width.
    pub rank_uncertainty_threshold: f64,
    /// Batch size (independent CRN replicates per adaptive step).
    pub batch_size: u32,
}

impl Default for DesignRankConfig {
    fn default() -> Self {
        Self { min_batches: 4, max_batches: 64, rank_uncertainty_threshold: 0.05, batch_size: 8 }
    }
}

impl DesignRankConfig {
    /// Batch counts at which the stopping rule is evaluated: every batch from
    /// `min_batches` through `max_batches`.
    fn n_looks(&self) -> u32 {
        let min = self.min_batches.max(1);
        let max = self.max_batches.max(min);
        max - min + 1
    }

    /// Two-sided critical value used at every look, for both the early-stop test and the
    /// reported `rank_uncertain` flag.
    ///
    /// Testing `gap ≥ 1.96·se` after every batch and stopping at the first success is
    /// optional stopping: with dozens of looks the chance that a truly tied pair ever
    /// crosses 1.96 is several times the nominal 5 %. Splitting the 5 % error budget
    /// evenly across the looks (Bonferroni, valid for any dependence between looks)
    /// gives `z = Φ⁻¹(1 − 0.025 / looks)`, which is `1.96` for a single look and ≈ 3.35
    /// for the default 61. The same constant labels the final ranking, so a pair
    /// published as separated is separated at the adjusted level.
    fn critical_value(&self) -> f64 {
        normal_ppf(1.0 - 0.025 / f64::from(self.n_looks()))
    }
}

/// Linear-Gaussian effect-width context for [`DesignObjective::ReduceEffectPosteriorWidth`].
#[derive(Clone, Debug)]
pub struct EffectWidthContext {
    /// Current Gram `XᵀX` (row-major, `p×p`).
    pub xtx: Arc<[f64]>,
    /// Residual variance estimate σ².
    pub sigma2: f64,
    /// Index of the treatment coefficient in the design (for ATE SE).
    pub treatment_col: usize,
    /// Current sample size.
    pub n: u64,
    /// Optional Gram updates when measuring listed variables (from a design analysis).
    pub measure_columns: Option<Arc<[MeasureColumnSpec]>>,
    /// Optional post-intervention Gram / σ² / n from a simulated experiment design.
    pub intervention_design: Option<InterventionDesignEffect>,
    /// Optional per-environment Grams for [`CandidateDesign::ObserveEnvironment`] pooling.
    pub environment_grams: Option<Arc<[EnvironmentGramSpec]>>,
}

/// Gram contribution from observing an additional environment partition.
#[derive(Clone, Debug)]
pub struct EnvironmentGramSpec {
    /// Environment this Gram belongs to.
    pub environment: EnvironmentId,
    /// Environment-local Gram `XᵀX` (row-major, same `p` as baseline).
    pub xtx: Arc<[f64]>,
    /// Environment-local sample size.
    pub n: u64,
    /// Optional residual variance under this environment (`None` = keep baseline σ²).
    pub sigma2: Option<f64>,
}

/// Column that would be added to the OLS design matrix if a variable is measured.
#[derive(Clone, Debug)]
pub struct MeasureColumnSpec {
    /// Variable this column corresponds to.
    pub variable: VariableId,
    /// Cross-products with existing columns: length `p`, entry `j` is `x_new · x_j`.
    pub cross: Arc<[f64]>,
    /// `x_new · x_new`.
    pub self_dot: f64,
    /// Residual variance after including this column (`None` = keep current σ²).
    pub sigma2_after: Option<f64>,
}

/// Simulated post-intervention OLS design used for SE reduction under [`CandidateDesign::Intervene`].
#[derive(Clone, Debug)]
pub struct InterventionDesignEffect {
    /// Gram after the planned intervention design (row-major `p×p`, same `p` as baseline).
    pub xtx: Arc<[f64]>,
    /// Residual variance under the intervention design.
    pub sigma2: f64,
    /// Effective sample size under the intervention design.
    pub n: u64,
}

/// Per-model log-likelihood draws for [`DesignObjective::DistinguishModels`].
///
/// `loglik[model_slot][draw]` — model slots align with `DesignObjective::DistinguishModels.models`.
#[derive(Clone, Debug)]
pub struct ModelLoglikDraws {
    /// Model ids (order matches rows).
    pub models: Arc<[ModelId]>,
    /// Row-major log-likelihood matrix: `models.len() * n_draws`.
    pub loglik: Arc<[f64]>,
    /// Draws per model.
    pub n_draws: usize,
}

/// Decision problems plus the Bayesian model a candidate's data updates, for
/// [`DesignObjective::ReduceDecisionRegret`] (problems keyed by [`crate::DecisionProblemId`]).
pub struct DecisionRegistry<A, O> {
    /// Problems keyed by raw id order (sparse holes allowed via Option).
    pub problems: Vec<Option<DecisionProblem<A, O>>>,
    /// Current belief about the decision state, shared by every problem.
    pub prior: DecisionPrior<O>,
    /// Sampling model of the data each candidate collects about the state.
    pub signal: Arc<dyn DecisionSignal<O>>,
}

/// Inputs shared across candidates for one ranking call.
pub struct DesignEvaluationContext<'a, A = (), O = ()> {
    /// Graph posterior ensemble (normalized preferred).
    pub graphs: &'a WeightedGraphSamples,
    /// Optional effect-width OLS context.
    pub effect_width: Option<&'a EffectWidthContext>,
    /// Optional model log-likelihood draws.
    pub model_loglik: Option<&'a ModelLoglikDraws>,
    /// Optional decision registry.
    pub decisions: Option<&'a DecisionRegistry<A, O>>,
    /// Query → variables that unlock identification when measured / intervened on.
    pub query_id_unlock: Option<&'a [(QueryId, Arc<[VariableId]>)]>,
    /// Query → environments that unlock identification when observed (multi-env ID).
    pub env_id_unlock: Option<&'a [(QueryId, Arc<[EnvironmentId]>)]>,
    /// Per-graph identification after intervention (length `graphs.n_samples`), from running an
    /// identifier on the mutilated / experimental graph. Required for Intervene candidates to
    /// gain identification mass beyond unlock-variable matches.
    pub identified_under_intervention: Option<&'a [GraphIdentFlag]>,
    /// Optional per-graph discrete features for EIG observation models. When `None`, graph keys
    /// are used as categorical labels (soft observation of which posterior atom is true).
    pub graph_features: Option<&'a [u32]>,
}

/// Rank candidate designs under an objective with batched MC and CRN.
pub struct DesignRanker {
    /// Ranking config.
    pub config: DesignRankConfig,
    /// Hard constraints.
    pub constraints: DesignConstraints,
}

impl Default for DesignRanker {
    fn default() -> Self {
        Self::new()
    }
}

impl DesignRanker {
    /// Default ranker.
    #[must_use]
    pub fn new() -> Self {
        Self { config: DesignRankConfig::default(), constraints: DesignConstraints::default() }
    }

    /// Builder: constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DesignConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Builder: config.
    #[must_use]
    pub fn with_config(mut self, config: DesignRankConfig) -> Self {
        self.config = config;
        self
    }

    /// Rank candidates. Higher score is better for every objective (regret negated).
    ///
    /// # Errors
    ///
    /// Empty inputs or invalid config.
    pub fn rank<A, O>(
        &self,
        objective: &DesignObjective,
        candidates: &[CandidateDesign],
        ctx_eval: &DesignEvaluationContext<'_, A, O>,
        ctx: &ExecutionContext,
    ) -> Result<DesignRanking, DesignError>
    where
        A: Clone,
        O: Clone,
    {
        self.validate_rank_inputs(candidates, ctx_eval)?;
        validate_objective_context(objective, ctx_eval)?;
        let analysis = prepare_decision(objective, ctx_eval)?;

        let (violations, active) = self.filter_active_candidates(candidates, analysis.as_ref());
        let decision = analysis
            .map(|analysis| DecisionScoring::new(analysis, candidates, &active))
            .transpose()?;
        let evaluations: Vec<ScoreEvaluation> = active
            .iter()
            .map(|&cand_i| {
                decision
                    .as_ref()
                    .map_or_else(|| static_evaluation(objective), |d| d.evaluation(cand_i))
            })
            .collect();

        let z = self.config.critical_value();
        let (draws, n_samples, budget, early_stopped) = self.run_mc_scoring_loop(
            objective,
            candidates,
            ctx_eval,
            ctx,
            &active,
            decision.as_ref(),
            z,
        )?;

        Ok(Self::assemble_ranking(
            &active,
            &draws,
            &evaluations,
            n_samples,
            candidates,
            violations,
            budget,
            early_stopped,
            objective.implemented_functional(),
            z,
        ))
    }

    /// Reject empty/degenerate `rank()` inputs before any MC work runs.
    fn validate_rank_inputs<A, O>(
        &self,
        candidates: &[CandidateDesign],
        ctx_eval: &DesignEvaluationContext<'_, A, O>,
    ) -> Result<(), DesignError> {
        if candidates.is_empty() {
            return Err(DesignError::EmptyCandidates);
        }
        if ctx_eval.graphs.n_samples == 0 {
            return Err(DesignError::EmptyPosterior);
        }
        if self.config.max_batches == 0 || self.config.batch_size == 0 {
            return Err(DesignError::Config("max_batches and batch_size must be > 0".into()));
        }
        Ok(())
    }

    /// Split `candidates` into hard-constraint violations and the indices that remain
    /// active for MC scoring. Under a decision objective, a candidate the signal has
    /// no sample size for is recorded as unlicensed rather than scored.
    fn filter_active_candidates<O>(
        &self,
        candidates: &[CandidateDesign],
        decision: Option<&PreposteriorAnalysis<'_, O>>,
    ) -> (Vec<ConstraintViolation>, Vec<usize>) {
        let mut violations = Vec::new();
        let mut active: Vec<usize> = Vec::new();
        for (i, c) in candidates.iter().enumerate() {
            if let Some(v) = self.check_constraints(i, c.cost()) {
                violations.push(v);
            } else if let Some(v) = decision.and_then(|d| unmapped_decision_candidate(i, c, d)) {
                violations.push(v);
            } else {
                active.push(i);
            }
        }
        (violations, active)
    }

    /// Adaptive batched Monte Carlo scoring loop with shared CRN draws across active
    /// candidates. Returns every candidate's per-draw scores (draw `i` is the same graph /
    /// decision replicate for all candidates, so pairwise differences are paired) alongside
    /// the realized sample count / budget / early-stop flag.
    #[allow(clippy::too_many_arguments)]
    fn run_mc_scoring_loop<A, O>(
        &self,
        objective: &DesignObjective,
        candidates: &[CandidateDesign],
        ctx_eval: &DesignEvaluationContext<'_, A, O>,
        ctx: &ExecutionContext,
        active: &[usize],
        decision: Option<&DecisionScoring<'_, O>>,
        z: f64,
    ) -> Result<(Vec<Vec<f64>>, u64, MonteCarloBudget, bool), DesignError>
    where
        A: Clone,
        O: Clone,
    {
        let mut draws: Vec<Vec<f64>> = vec![Vec::new(); active.len()];
        let mut n_samples: u64 = 0;
        let mut budget = MonteCarloBudget::default();
        let mut early_stopped = false;
        let mut rng = ctx.rng.stream_for(StreamDomain::Design, 0xD351_0611);
        // Per-ranking constants of the scoring channels, built once rather than per draw.
        let channels = ScoringChannels::new(objective, ctx_eval);

        let min_batches = self.config.min_batches.max(1);
        let max_batches = self.config.max_batches.max(min_batches);

        for batch_i in 0..max_batches {
            for _ in 0..self.config.batch_size {
                // Shared CRN draw index into graph posterior.
                let g_idx = sample_categorical(&mut rng, &ctx_eval.graphs.weights)
                    .ok_or(DesignError::EmptyPosterior)?;
                // Decision replicates share one seed across candidates, so every
                // candidate sees the same sampled state and noise (CRN).
                let crn_seed = decision.map(|_| rng.next_u64());
                for (slot, &cand_i) in active.iter().enumerate() {
                    let score = match (decision, crn_seed) {
                        (Some(d), Some(seed)) => d.score(cand_i, seed)?,
                        _ => score_candidate(
                            objective,
                            &candidates[cand_i],
                            ctx_eval,
                            &channels,
                            g_idx,
                            &mut rng,
                        )?,
                    };
                    draws[slot].push(score);
                    budget.evaluations += 1;
                }
                n_samples += 1;
                budget.samples = n_samples;
            }

            if batch_i + 1 >= min_batches
                && rank_uncertainty_ok(&draws, z, self.config.rank_uncertainty_threshold)
            {
                early_stopped = true;
                break;
            }
            if ctx.cancellation.is_cancelled() {
                break;
            }
        }

        Ok((draws, n_samples, budget, early_stopped))
    }

    /// Turn per-candidate draws into a sorted, CI-annotated [`DesignRanking`].
    ///
    /// `rank_uncertain` compares each candidate with the next-ranked one through the
    /// *paired* difference of their draws (shared-CRN replicates) at the sequentially
    /// adjusted critical value `z`; a gap of exactly zero is a tie and is always uncertain.
    #[allow(clippy::too_many_arguments)]
    fn assemble_ranking(
        active: &[usize],
        draws: &[Vec<f64>],
        evaluations: &[ScoreEvaluation],
        n_samples: u64,
        candidates: &[CandidateDesign],
        violations: Vec<ConstraintViolation>,
        budget: MonteCarloBudget,
        early_stopped: bool,
        implemented_functional: &'static str,
        z: f64,
    ) -> DesignRanking {
        // (candidate index, mean, error, evaluation, slot)
        let mut scored: Vec<(usize, f64, MonteCarloError, ScoreEvaluation, usize)> = active
            .iter()
            .enumerate()
            .map(|(slot, &cand_i)| {
                let mut acc = Welford::new();
                draws[slot].iter().for_each(|&x| acc.push(x));
                let mean = if n_samples > 0 { acc.mean() } else { 0.0 };
                let stderr = match evaluations[slot] {
                    ScoreEvaluation::Exact => 0.0,
                    ScoreEvaluation::MonteCarlo => acc.stderr_of_mean(),
                };
                let err = MonteCarloError { stderr, samples: n_samples };
                (cand_i, mean, err, evaluations[slot], slot)
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
        });

        let mut ranked = Vec::with_capacity(scored.len());
        for (rank, (cand_i, score, mc, evaluation, slot)) in scored.iter().enumerate() {
            let uncertain = scored.get(rank + 1).is_some_and(|next| {
                let exact_pair = matches!(
                    (evaluation, next.3),
                    (ScoreEvaluation::Exact, ScoreEvaluation::Exact)
                );
                let (gap, se) = paired_difference(&draws[*slot], &draws[next.4]);
                let se = if exact_pair { 0.0 } else { se };
                gap.abs() <= z * se
            });
            ranked.push(RankedCandidate {
                candidate_index: *cand_i,
                candidate: candidates[*cand_i].clone(),
                score: *score,
                monte_carlo: *mc,
                rank,
                rank_uncertain: uncertain,
                implemented_functional: Arc::from(implemented_functional),
                evaluation: *evaluation,
            });
        }

        DesignRanking {
            ranked: Arc::from(ranked),
            violations: Arc::from(violations),
            budget,
            early_stopped,
        }
    }

    fn check_constraints(&self, index: usize, cost: DesignCost) -> Option<ConstraintViolation> {
        if let Some(max) = self.constraints.max_cost {
            if cost.amount > max {
                return Some(ConstraintViolation {
                    candidate_index: index,
                    constraint: Arc::from("max_cost"),
                    detail: Arc::from(format!("cost {} exceeds max_cost {max}", cost.amount)),
                });
            }
        }
        if let Some(max) = self.constraints.max_sample_budget {
            if cost.sample_budget > max {
                return Some(ConstraintViolation {
                    candidate_index: index,
                    constraint: Arc::from("max_sample_budget"),
                    detail: Arc::from(format!(
                        "sample_budget {} exceeds max {max}",
                        cost.sample_budget
                    )),
                });
            }
        }
        None
    }
}

/// Mean and standard error of the paired difference `a_i − b_i` (Welford, so the
/// error does not degrade when the scores are large relative to their spread).
fn paired_difference(a: &[f64], b: &[f64]) -> (f64, f64) {
    let mut acc = Welford::new();
    for (x, y) in a.iter().zip(b) {
        acc.push(x - y);
    }
    (acc.mean(), acc.stderr_of_mean())
}

/// Sequential stopping test on the current per-candidate draws.
///
/// The top three candidates by mean are compared pairwise-adjacent, each through the
/// standard error of its paired difference. Early stopping is acceptable for a pair when
/// its half-width `z·se` is already within `threshold`, or when the pair is separated by
/// more than `z·se`; a tie whose half-width is still above the threshold is unresolved.
/// `z` must be the look-adjusted critical value (see `DesignRankConfig::critical_value`),
/// never the fixed-sample 1.96, because this test is repeated after every batch.
fn rank_uncertainty_ok(draws: &[Vec<f64>], z: f64, threshold: f64) -> bool {
    let n = draws.first().map_or(0, Vec::len);
    if draws.len() < 2 || n == 0 {
        return true;
    }
    let means: Vec<f64> = draws
        .iter()
        .map(|d| {
            let mut acc = Welford::new();
            d.iter().for_each(|&x| acc.push(x));
            acc.mean()
        })
        .collect();
    let mut order: Vec<usize> = (0..means.len()).collect();
    order.sort_by(|a, b| means[*b].partial_cmp(&means[*a]).unwrap_or(std::cmp::Ordering::Equal));
    // Check top adjacent pairs.
    let top = order.len().min(3);
    for i in 0..top.saturating_sub(1) {
        let (gap, se) = paired_difference(&draws[order[i]], &draws[order[i + 1]]);
        let half_width = z * se;
        if half_width > threshold && gap.abs() <= half_width {
            return false;
        }
    }
    true
}

fn shannon_entropy(weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    if !(total > 0.0) {
        return 0.0;
    }
    let mut h = 0.0;
    for w in weights {
        let p = w / total;
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

/// Ranking-invariant precomputation for the per-draw scoring channels, built once per
/// `rank()` call instead of once per draw per candidate.
struct ScoringChannels {
    entropy: Option<EntropyChannel>,
    /// `(row_i, row_j)` of every requested model pair present in the log-likelihood table.
    model_pairs: Vec<(usize, usize)>,
}

impl ScoringChannels {
    fn new<A, O>(objective: &DesignObjective, ctx: &DesignEvaluationContext<'_, A, O>) -> Self {
        match objective {
            DesignObjective::ReduceGraphEntropy => Self {
                entropy: EntropyChannel::new(ctx.graphs, ctx.graph_features),
                model_pairs: Vec::new(),
            },
            DesignObjective::DistinguishModels { models } => Self {
                entropy: None,
                model_pairs: ctx
                    .model_loglik
                    .map_or_else(Vec::new, |ll| model_pair_rows(ll, models)),
            },
            _ => Self { entropy: None, model_pairs: Vec::new() },
        }
    }
}

/// Reject malformed per-objective context before any MC work runs. A context that cannot
/// be evaluated is an error, never a silently "exact" zero score.
fn validate_objective_context<A, O>(
    objective: &DesignObjective,
    ctx: &DesignEvaluationContext<'_, A, O>,
) -> Result<(), DesignError> {
    let n = ctx.graphs.n_samples;
    match objective {
        DesignObjective::ReduceGraphEntropy => {
            if let Some(features) = ctx.graph_features {
                if features.len() != n {
                    return Err(DesignError::Shape(format!(
                        "graph_features length {} != graph posterior size {n}",
                        features.len()
                    )));
                }
            }
        }
        DesignObjective::IncreaseIdentificationProbability { .. } => {
            if let Some(flags) = ctx.identified_under_intervention {
                if flags.len() != n {
                    return Err(DesignError::Shape(format!(
                        "identified_under_intervention length {} != graph posterior size {n}",
                        flags.len()
                    )));
                }
            }
        }
        DesignObjective::ReduceEffectPosteriorWidth { .. } => {
            let Some(ew) = ctx.effect_width else {
                return Err(DesignError::Config(
                    "ReduceEffectPosteriorWidth requires effect_width context".into(),
                ));
            };
            baseline_treatment_se(ew)?;
        }
        DesignObjective::DistinguishModels { .. } => {
            let Some(ll) = ctx.model_loglik else {
                return Err(DesignError::Config(
                    "DistinguishModels requires model_loglik context".into(),
                ));
            };
            if ll.loglik.len() != ll.models.len().saturating_mul(ll.n_draws) {
                return Err(DesignError::Shape(format!(
                    "model_loglik has {} values, expected {} models × {} draws",
                    ll.loglik.len(),
                    ll.models.len(),
                    ll.n_draws
                )));
            }
        }
        DesignObjective::ReduceDecisionRegret { .. } => {}
    }
    Ok(())
}

fn score_candidate<A, O>(
    objective: &DesignObjective,
    candidate: &CandidateDesign,
    ctx: &DesignEvaluationContext<'_, A, O>,
    channels: &ScoringChannels,
    graph_idx: usize,
    rng: &mut CausalRng,
) -> Result<f64, DesignError>
where
    A: Clone,
    O: Clone,
{
    match objective {
        DesignObjective::ReduceGraphEntropy => {
            Ok(channels.entropy.as_ref().map_or(0.0, |c| c.draw(candidate, graph_idx, rng)))
        }
        DesignObjective::IncreaseIdentificationProbability { query } => {
            Ok(id_prob_gain(candidate, ctx, *query))
        }
        DesignObjective::ReduceEffectPosteriorWidth { query: _ } => {
            let Some(ew) = ctx.effect_width else {
                return Err(DesignError::Config(
                    "ReduceEffectPosteriorWidth requires effect_width context".into(),
                ));
            };
            effect_width_reduction(candidate, ew)
        }
        DesignObjective::DistinguishModels { models } => {
            let Some(ll) = ctx.model_loglik else {
                return Err(DesignError::Config(
                    "DistinguishModels requires model_loglik context".into(),
                ));
            };
            Ok(model_distinguish_score(candidate, ll, models.len(), &channels.model_pairs, rng))
        }
        DesignObjective::ReduceDecisionRegret { .. } => Err(DesignError::Config(
            "ReduceDecisionRegret is scored through its prepared preposterior analysis".into(),
        )),
    }
}

/// Discrete soft-observation channel over the posterior's graph categories, with the
/// candidate-independent parts (category index per graph, prior entropy) computed once.
struct EntropyChannel {
    weights: Vec<f64>,
    /// Category index of every posterior graph.
    cat_of: Vec<usize>,
    /// Number of distinct categories (≥ 2).
    k: usize,
    prior_h: f64,
}

impl EntropyChannel {
    /// `None` when the posterior is empty, the features do not line up with it, or there
    /// are fewer than two categories — no observation can then change the entropy.
    fn new(graphs: &WeightedGraphSamples, graph_features: Option<&[u32]>) -> Option<Self> {
        let n = graphs.n_samples;
        if n == 0 {
            return None;
        }
        if let Some(feat) = graph_features {
            if feat.len() != n {
                return None;
            }
        }
        let labels: Vec<u64> = if let Some(feat) = graph_features {
            feat.iter().map(|&label| u64::from(label)).collect()
        } else {
            graphs.graph_keys.to_vec()
        };
        let mut categories = labels.clone();
        categories.sort_unstable();
        categories.dedup();
        let k = categories.len();
        if k < 2 {
            return None;
        }
        let cat_of = labels.iter().map(|l| categories.binary_search(l).unwrap_or(0)).collect();
        Some(Self {
            weights: graphs.weights.to_vec(),
            cat_of,
            k,
            prior_h: shannon_entropy(&graphs.weights),
        })
    }

    /// One Monte Carlo draw of information gain for the discrete observation channel:
    /// sample `y ~ P(y | G★, design)`, then return the signed reduction
    /// `H(prior) − H(p(G|y))`.
    ///
    /// Individual draws may be negative when an observation increases posterior entropy;
    /// expectation is taken over draws without pointwise clipping (MM-012).
    fn draw(&self, candidate: &CandidateDesign, graph_idx: usize, rng: &mut CausalRng) -> f64 {
        let k = self.k;
        let reliability = observation_reliability(candidate);
        // A zero-information design must not enter the soft channel: reliability 0 would
        // zero the matched-category likelihood and invent anti-information.
        if reliability <= 0.0 {
            return 0.0;
        }
        let true_cat = self.cat_of[graph_idx];

        // Sample soft observation of the true graph's categorical feature.
        let y = if rng.next_f64() < reliability {
            true_cat
        } else {
            let mut u = (rng.next_f64() * (k - 1) as f64).floor() as usize;
            if u >= true_cat {
                u += 1;
            }
            u.min(k - 1)
        };

        let off = (1.0 - reliability) / (k - 1) as f64;
        let mut post = self.weights.clone();
        for (w, &cat) in post.iter_mut().zip(&self.cat_of) {
            *w *= if cat == y { reliability } else { off };
        }
        self.prior_h - shannon_entropy(&post)
    }
}

/// Single-draw entropy channel for tests that need the raw per-draw value.
#[cfg(test)]
fn eig_graph_entropy(
    candidate: &CandidateDesign,
    graphs: &WeightedGraphSamples,
    graph_idx: usize,
    graph_features: Option<&[u32]>,
    rng: &mut CausalRng,
) -> f64 {
    EntropyChannel::new(graphs, graph_features)
        .map_or(0.0, |channel| channel.draw(candidate, graph_idx, rng))
}

/// Deterministic observation reliability for the discrete graph-feature channel.
///
/// This is `1 − exp(−c · k)` (or a sample-size saturating map), not a likelihood
/// `p(y | G, design)`. Scores that call this are heuristic channel entropy, not EIG.
fn observation_reliability(candidate: &CandidateDesign) -> f64 {
    // Lower bound is 0, not a positive floor: clamping a no-op up to 0.05 made
    // zero-information designs score like weakly informative ones (attr-design-state-5).
    match candidate {
        CandidateDesign::Measure(p) => {
            let k = p.variables.len() as f64;
            (1.0 - (-0.75 * k).exp()).clamp(0.0, 0.99)
        }
        CandidateDesign::Intervene(p) => {
            let k = p.targets.len() as f64;
            (1.0 - (-1.0 * k).exp()).clamp(0.0, 0.99)
        }
        CandidateDesign::ObserveEnvironment(p) => {
            let n = p.additional_rows as f64;
            (1.0 - (1.0 + n / 50.0).recip()).clamp(0.0, 0.95)
        }
        CandidateDesign::IncreaseSamplingRate(p) => {
            let n = p.additional_samples as f64;
            (1.0 - (1.0 + n / 50.0).recip()).clamp(0.0, 0.95)
        }
    }
}

fn evidence_strength(candidate: &CandidateDesign) -> f64 {
    // Used only by the DistinguishModels heuristic (not EIG / ID / SE / decisions).
    observation_reliability(candidate)
}

/// Identification-mass gain of a candidate. `identified_under_intervention`, when present,
/// has already been checked against the posterior size by [`validate_objective_context`].
fn id_prob_gain<A, O>(
    candidate: &CandidateDesign,
    ctx: &DesignEvaluationContext<'_, A, O>,
    query: QueryId,
) -> f64 {
    let baseline = ctx.graphs.identified_mass() / ctx.graphs.total_weight().max(1e-15);
    let unlock = ctx
        .query_id_unlock
        .and_then(|m| m.iter().find(|(q, _)| *q == query).map(|(_, v)| v.as_ref()))
        .unwrap_or(&[]);
    let env_unlock = ctx
        .env_id_unlock
        .and_then(|m| m.iter().find(|(q, _)| *q == query).map(|(_, v)| v.as_ref()))
        .unwrap_or(&[]);

    let intervene_flags = ctx.identified_under_intervention;

    let mut identified = 0.0;
    let mut total = 0.0;
    for i in 0..ctx.graphs.n_samples {
        let w = ctx.graphs.weights[i];
        total += w;
        let mut is_id = ctx.graphs.identified[i] == GraphIdentFlag::Identified;
        if !is_id {
            is_id = candidate_unlocks(candidate, unlock, env_unlock);
        }
        if !is_id {
            if let CandidateDesign::Intervene(_) = candidate {
                if let Some(flags) = intervene_flags {
                    is_id = flags[i] == GraphIdentFlag::Identified;
                }
            }
        }
        if is_id {
            identified += w;
        }
    }
    let post = identified / total.max(1e-15);
    post - baseline
}

fn candidate_unlocks(
    candidate: &CandidateDesign,
    unlock: &[VariableId],
    env_unlock: &[EnvironmentId],
) -> bool {
    match candidate {
        CandidateDesign::Measure(p) => {
            !unlock.is_empty() && p.variables.iter().any(|v| unlock.contains(v))
        }
        CandidateDesign::Intervene(p) => {
            !unlock.is_empty() && p.targets.iter().any(|v| unlock.contains(v))
        }
        CandidateDesign::ObserveEnvironment(p) => {
            !env_unlock.is_empty() && env_unlock.contains(&p.environment)
        }
        CandidateDesign::IncreaseSamplingRate(_) => false,
    }
}

/// Standard error of the treatment coefficient, `sqrt(σ² · [(XᵀX)⁻¹]_tt)`.
///
/// `what` names the Gram in error messages. A Gram that cannot be inverted is an error:
/// reporting it as "no information gain" would rank every candidate as an exact tie.
fn treatment_se(
    xtx: &[f64],
    p: usize,
    treatment_col: usize,
    sigma2: f64,
    what: &str,
) -> Result<f64, DesignError> {
    if p == 0 || xtx.len() != p * p {
        return Err(DesignError::Shape(format!(
            "{what}: Gram must be a non-empty square p×p matrix (got {} entries)",
            xtx.len()
        )));
    }
    if treatment_col >= p {
        return Err(DesignError::Shape(format!(
            "{what}: treatment_col {treatment_col} out of range for p = {p}"
        )));
    }
    if !(sigma2.is_finite() && sigma2 > 0.0) {
        return Err(DesignError::Config(format!(
            "{what}: residual variance must be positive and finite (got {sigma2})"
        )));
    }
    let inv = invert_square(xtx, p)
        .ok_or_else(|| DesignError::Numerical(format!("{what}: Gram is singular")))?;
    Ok((sigma2 * inv[treatment_col * p + treatment_col].max(0.0)).sqrt())
}

fn gram_side_len(xtx: &[f64]) -> usize {
    let n2 = xtx.len();
    let mut k = 0usize;
    while k * k < n2 {
        k += 1;
    }
    if k * k == n2 { k } else { 0 }
}

/// Baseline `(p, se₀)` of the treatment coefficient.
fn baseline_treatment_se(ew: &EffectWidthContext) -> Result<(usize, f64), DesignError> {
    let p = gram_side_len(&ew.xtx);
    let se0 = treatment_se(&ew.xtx, p, ew.treatment_col, ew.sigma2, "baseline")?;
    Ok((p, se0))
}

/// Expand `p×p` Gram by appending one column with given cross-products.
fn expand_gram(xtx: &[f64], p: usize, cross: &[f64], self_dot: f64) -> Option<Vec<f64>> {
    if cross.len() != p || xtx.len() != p * p {
        return None;
    }
    let p1 = p + 1;
    let mut out = vec![0.0; p1 * p1];
    for i in 0..p {
        for j in 0..p {
            out[i * p1 + j] = xtx[i * p + j];
        }
        out[i * p1 + p] = cross[i];
        out[p * p1 + i] = cross[i];
    }
    out[p * p1 + p] = self_dot;
    Some(out)
}

/// Signed reduction `se₀ − se₁` of the treatment-effect standard error.
///
/// A design that *raises* the SE (a collinear covariate, a noisier environment) scores
/// negative and ranks below a neutral one. `0` means the candidate has no modelled effect
/// (no sample increment, no matching design-analysis columns, no environment / intervention
/// Gram supplied); a malformed or singular Gram is an error, not a zero.
fn effect_width_reduction(
    candidate: &CandidateDesign,
    ew: &EffectWidthContext,
) -> Result<f64, DesignError> {
    let (p, se0) = baseline_treatment_se(ew)?;

    match candidate {
        CandidateDesign::IncreaseSamplingRate(s) => {
            if s.additional_samples == 0 {
                return Ok(0.0);
            }
            // XtX scales with n; SE scales as 1/sqrt(n).
            let n1 = (ew.n + s.additional_samples) as f64;
            let n0 = ew.n.max(1) as f64;
            let se1 = se0 * (n0 / n1).sqrt();
            Ok(se0 - se1)
        }
        CandidateDesign::ObserveEnvironment(e) => {
            if let Some(grams) = ew.environment_grams.as_ref() {
                if let Some(spec) = grams.iter().find(|s| s.environment == e.environment) {
                    let p_env = gram_side_len(&spec.xtx);
                    if p_env != p || spec.xtx.len() != ew.xtx.len() {
                        return Err(DesignError::Shape(format!(
                            "environment {:?} Gram has {} entries, expected {}",
                            e.environment,
                            spec.xtx.len(),
                            ew.xtx.len()
                        )));
                    }
                    let mut pooled = ew.xtx.to_vec();
                    for (a, b) in pooled.iter_mut().zip(spec.xtx.iter()) {
                        *a += *b;
                    }
                    let sigma2 = spec.sigma2.unwrap_or(ew.sigma2);
                    let se1 =
                        treatment_se(&pooled, p, ew.treatment_col, sigma2, "pooled environment")?;
                    return Ok(se0 - se1);
                }
            }
            // No env Gram: fall back to additional-row isotropic scaling for SE only.
            if e.additional_rows == 0 {
                return Ok(0.0);
            }
            let n1 = (ew.n + e.additional_rows) as f64;
            let n0 = ew.n.max(1) as f64;
            let se1 = se0 * (n0 / n1).sqrt();
            Ok(se0 - se1)
        }
        CandidateDesign::Measure(plan) => {
            let Some(specs) = ew.measure_columns.as_ref() else {
                // No design-analysis columns → no modelled SE change.
                return Ok(0.0);
            };
            let mut xtx = ew.xtx.to_vec();
            let mut cur_p = p;
            let mut sigma2 = ew.sigma2;
            let mut matched = 0usize;
            for v in plan.variables.iter() {
                let Some(spec) = specs.iter().find(|s| s.variable == *v) else {
                    continue;
                };
                let Some(expanded) = expand_gram(&xtx, cur_p, &spec.cross, spec.self_dot) else {
                    return Err(DesignError::Shape(format!(
                        "measure column for variable {v:?} has {} cross-products, expected {cur_p}",
                        spec.cross.len()
                    )));
                };
                xtx = expanded;
                cur_p += 1;
                if let Some(s2) = spec.sigma2_after {
                    sigma2 = s2;
                }
                matched += 1;
            }
            if matched == 0 {
                return Ok(0.0);
            }
            let se1 = treatment_se(&xtx, cur_p, ew.treatment_col, sigma2, "post-measurement")?;
            Ok(se0 - se1)
        }
        CandidateDesign::Intervene(_) => {
            let Some(design) = ew.intervention_design.as_ref() else {
                return Ok(0.0);
            };
            let p1 = gram_side_len(&design.xtx);
            if p1 != p {
                return Err(DesignError::Shape(format!(
                    "intervention design Gram has {} entries, expected {}",
                    design.xtx.len(),
                    ew.xtx.len()
                )));
            }
            let se1 = treatment_se(
                &design.xtx,
                p1,
                ew.treatment_col,
                design.sigma2,
                "post-intervention",
            )?;
            Ok(se0 - se1)
        }
    }
}

/// Table rows `(i, j)` of every pair drawn from `models` that is present in `ll`.
fn model_pair_rows(ll: &ModelLoglikDraws, models: &[ModelId]) -> Vec<(usize, usize)> {
    let row_of: Vec<Option<usize>> =
        models.iter().map(|m| ll.models.iter().position(|x| x == m)).collect();
    let mut pairs = Vec::new();
    for i in 0..models.len() {
        for j in (i + 1)..models.len() {
            if let (Some(ri), Some(rj)) = (row_of[i], row_of[j]) {
                pairs.push((ri, rj));
            }
        }
    }
    pairs
}

fn model_distinguish_score(
    candidate: &CandidateDesign,
    ll: &ModelLoglikDraws,
    n_models: usize,
    pairs: &[(usize, usize)],
    rng: &mut CausalRng,
) -> f64 {
    if n_models < 2 || ll.n_draws == 0 {
        return 0.0;
    }
    let draw = (rng.next_u64() as usize) % ll.n_draws;
    let strength = evidence_strength(candidate);
    // Expected absolute log-score gap over the requested model pairs, scaled by evidence strength.
    let mut gap = 0.0;
    for &(ri, rj) in pairs {
        let a = ll.loglik[ri * ll.n_draws + draw];
        let b = ll.loglik[rj * ll.n_draws + draw];
        gap += (a - b).abs();
    }
    if pairs.is_empty() { 0.0 } else { strength * gap / pairs.len() as f64 }
}

/// How non-decision objectives are evaluated: deterministic functionals are exact;
/// scores that consume per-draw randomness are Monte Carlo estimates.
const fn static_evaluation(objective: &DesignObjective) -> ScoreEvaluation {
    match objective {
        DesignObjective::IncreaseIdentificationProbability { .. }
        | DesignObjective::ReduceEffectPosteriorWidth { .. } => ScoreEvaluation::Exact,
        DesignObjective::ReduceGraphEntropy
        | DesignObjective::DistinguishModels { .. }
        | DesignObjective::ReduceDecisionRegret { .. } => ScoreEvaluation::MonteCarlo,
    }
}

/// Prepare the preposterior analysis for [`DesignObjective::ReduceDecisionRegret`]
/// (utilities and admissibility evaluated once per ranking).
fn prepare_decision<'a, A, O>(
    objective: &DesignObjective,
    ctx: &DesignEvaluationContext<'a, A, O>,
) -> Result<Option<PreposteriorAnalysis<'a, O>>, DesignError> {
    let DesignObjective::ReduceDecisionRegret { decision } = objective else {
        return Ok(None);
    };
    let Some(registry) = ctx.decisions else {
        return Err(DesignError::Config("ReduceDecisionRegret requires decisions context".into()));
    };
    let Some(Some(problem)) = registry.problems.get(decision.raw() as usize) else {
        return Err(DesignError::Config(format!("decision problem {decision} is not registered")));
    };
    PreposteriorAnalysis::new(problem, &registry.prior, registry.signal.as_ref()).map(Some)
}

fn unmapped_decision_candidate<O>(
    index: usize,
    candidate: &CandidateDesign,
    analysis: &PreposteriorAnalysis<'_, O>,
) -> Option<ConstraintViolation> {
    let signal = analysis.signal();
    signal.sample_size(candidate).is_none().then(|| ConstraintViolation {
        candidate_index: index,
        constraint: Arc::from("unlicensed_candidate"),
        detail: Arc::from(format!(
            "decision signal `{}` declares no sample size for this candidate",
            signal.name()
        )),
    })
}

/// Per-candidate preposterior scoring: the design's sample size and, where
/// available, its exact EVSI (computed once and reused by every replicate).
struct DecisionScoring<'a, O> {
    analysis: PreposteriorAnalysis<'a, O>,
    /// `(n, exact EVSI)` by candidate index; `None` for inactive candidates.
    plans: Vec<Option<(u64, Option<f64>)>>,
}

impl<'a, O> DecisionScoring<'a, O> {
    fn new(
        analysis: PreposteriorAnalysis<'a, O>,
        candidates: &[CandidateDesign],
        active: &[usize],
    ) -> Result<Self, DesignError> {
        let mut plans = vec![None; candidates.len()];
        for &cand_i in active {
            if let Some(n) = analysis.signal().sample_size(&candidates[cand_i]) {
                plans[cand_i] = Some((n, analysis.exact_evsi(n)?));
            }
        }
        Ok(Self { analysis, plans })
    }

    fn evaluation(&self, cand_i: usize) -> ScoreEvaluation {
        match self.plans.get(cand_i).copied().flatten() {
            Some((_, Some(_))) => ScoreEvaluation::Exact,
            _ => ScoreEvaluation::MonteCarlo,
        }
    }

    fn score(&self, cand_i: usize, crn_seed: u64) -> Result<f64, DesignError> {
        match self.plans.get(cand_i).copied().flatten() {
            Some((_, Some(exact))) => Ok(exact),
            Some((n, None)) => self.analysis.sample_evsi(n, &mut CausalRng::from_seed(crn_seed)),
            None => Err(DesignError::Config(format!(
                "candidate {cand_i} has no decision sample size and cannot be scored"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{DesignCost, EnvironmentPlan, MeasurementPlan, SamplingPlan};
    use crate::decision::{DecisionConstraint, DecisionProblemId, Utility};
    use antecedent_core::{CausalRng, EnvironmentId, VariableId};
    use antecedent_prob::GraphIdentFlag;

    fn toy_graphs() -> WeightedGraphSamples {
        WeightedGraphSamples::new(
            vec![0.5, 0.3, 0.2],
            vec![
                GraphIdentFlag::Identified,
                GraphIdentFlag::Unidentified,
                GraphIdentFlag::Unidentified,
            ],
            vec![10, 20, 30],
        )
        .expect("graphs")
    }

    /// MM-012: a mismatched soft observation can raise posterior entropy; the draw
    /// must keep the signed reduction rather than clipping at zero.
    #[test]
    fn entropy_channel_preserves_all_graph_key_bits() {
        let graphs = WeightedGraphSamples::new(
            vec![0.5, 0.5],
            vec![GraphIdentFlag::Identified; 2],
            vec![1, (1u64 << 32) + 1],
        )
        .unwrap();
        let candidate = CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(2)]),
            cost: DesignCost::zero(),
            tag: 20,
        });
        let mut implicit_rng = CausalRng::from_seed(13);
        let mut explicit_rng = implicit_rng.clone();
        let implicit = eig_graph_entropy(&candidate, &graphs, 0, None, &mut implicit_rng);
        let explicit = eig_graph_entropy(&candidate, &graphs, 0, Some(&[0, 1]), &mut explicit_rng);
        assert!(explicit > 0.0);
        assert!((implicit - explicit).abs() < 1e-12);
    }

    #[test]
    fn eig_draw_keeps_negative_entropy_reduction() {
        let graphs = toy_graphs();
        let features = [0_u32, 1, 2];
        let candidate = CandidateDesign::ObserveEnvironment(EnvironmentPlan {
            environment: EnvironmentId::from_raw(0),
            additional_rows: 50, // reliability = 0.5
            cost: DesignCost::zero(),
            tag: 50,
        });
        let mut rng = CausalRng::from_seed(20_260_725);
        let mut saw_negative = false;
        for _ in 0..20_000 {
            let graph_idx = (rng.next_u64() as usize) % graphs.n_samples;
            let delta =
                eig_graph_entropy(&candidate, &graphs, graph_idx, Some(&features), &mut rng);
            assert!(delta.is_finite());
            if delta < -1e-15 {
                saw_negative = true;
                break;
            }
        }
        assert!(
            saw_negative,
            "expected a negative signed entropy reduction under the soft channel"
        );
    }

    #[test]
    fn ranks_measurement_above_noop_sampling_for_entropy() {
        let graphs = toy_graphs();
        // additional_samples = 0 is a true no-op (raw reliability 0). The old
        // clamp(0.05, …) floored it, so a no-op scored like weakly informative
        // sampling; the previous sorted[0] >= sorted[1] check could never fail.
        let noop = CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 0,
            cost: DesignCost::zero(),
            tag: 1,
        });
        let measure = CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(2)]),
            cost: DesignCost::zero(),
            tag: 20,
        });
        let mut rng = CausalRng::from_seed(20_260_921);
        let n_draw = 4_000;
        let mut sum_noop = 0.0;
        let mut sum_measure = 0.0;
        for _ in 0..n_draw {
            let g = (rng.next_u64() as usize) % graphs.n_samples;
            sum_noop += eig_graph_entropy(&noop, &graphs, g, None, &mut rng);
            sum_measure += eig_graph_entropy(&measure, &graphs, g, None, &mut rng);
        }
        let mean_noop = sum_noop / f64::from(n_draw);
        let mean_measure = sum_measure / f64::from(n_draw);
        assert!(
            mean_noop.abs() < 1e-12,
            "no-op must score zero information under the entropy channel, got {mean_noop}"
        );
        assert!(
            mean_measure > mean_noop,
            "measure ({mean_measure}) must strictly beat no-op ({mean_noop})"
        );

        let candidates = vec![noop, measure];
        let ranker = DesignRanker::new().with_config(DesignRankConfig {
            min_batches: 8,
            max_batches: 32,
            batch_size: 8,
            rank_uncertainty_threshold: 0.05,
        });
        let ctx = ExecutionContext::for_tests(7);
        let eval = DesignEvaluationContext::<(), ()> {
            graphs: &graphs,
            effect_width: None,
            model_loglik: None,
            decisions: None,
            query_id_unlock: None,
            env_id_unlock: None,
            identified_under_intervention: None,
            graph_features: None,
        };
        let ranking = ranker
            .rank(&DesignObjective::ReduceGraphEntropy, &candidates, &eval, &ctx)
            .expect("rank");
        assert_eq!(ranking.ranked.len(), 2);
        assert_eq!(ranking.ranked[0].candidate.tag(), 20);
        assert!(ranking.ranked[0].score > ranking.ranked[1].score);
        assert!(ranking.ranked[1].score.abs() < 1e-12);
    }

    #[test]
    fn records_cost_violations_without_silent_drop() {
        let graphs = toy_graphs();
        let candidates = vec![CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(0)]),
            cost: DesignCost { amount: 100.0, sample_budget: 0 },
            tag: 1,
        })];
        let ranker = DesignRanker::new()
            .with_constraints(DesignConstraints { max_cost: Some(10.0), max_sample_budget: None });
        let ctx = ExecutionContext::for_tests(1);
        let eval = DesignEvaluationContext::<(), ()> {
            graphs: &graphs,
            effect_width: None,
            model_loglik: None,
            decisions: None,
            query_id_unlock: None,
            env_id_unlock: None,
            identified_under_intervention: None,
            graph_features: None,
        };
        let ranking = ranker
            .rank(&DesignObjective::ReduceGraphEntropy, &candidates, &eval, &ctx)
            .expect("rank");
        assert_eq!(ranking.violations.len(), 1);
        assert!(ranking.ranked.is_empty());
        assert_eq!(ranking.violations[0].constraint.as_ref(), "max_cost");
    }

    #[test]
    fn identification_prob_increases_when_measuring_unlock_var() {
        let graphs = toy_graphs();
        let q = QueryId::from_raw(0);
        let unlock = [(q, Arc::from([VariableId::from_raw(3)]))];
        let candidates = vec![CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(3)]),
            cost: DesignCost::zero(),
            tag: 1,
        })];
        let ranker = DesignRanker::new().with_config(DesignRankConfig {
            min_batches: 2,
            max_batches: 4,
            batch_size: 4,
            rank_uncertainty_threshold: 1.0,
        });
        let ctx = ExecutionContext::for_tests(3);
        let eval = DesignEvaluationContext::<(), ()> {
            graphs: &graphs,
            effect_width: None,
            model_loglik: None,
            decisions: None,
            query_id_unlock: Some(&unlock),
            env_id_unlock: None,
            identified_under_intervention: None,
            graph_features: None,
        };
        let ranking = ranker
            .rank(
                &DesignObjective::IncreaseIdentificationProbability { query: q },
                &candidates,
                &eval,
                &ctx,
            )
            .expect("rank");
        assert!(ranking.ranked[0].score > 0.0);
    }

    #[test]
    fn intervene_without_identifier_flags_does_not_fabricate_id() {
        use crate::candidate::ExperimentPlan;
        let graphs = toy_graphs();
        let q = QueryId::from_raw(0);
        let candidates = vec![CandidateDesign::Intervene(ExperimentPlan {
            targets: Arc::from([VariableId::from_raw(0)]),
            cost: DesignCost::zero(),
            tag: 1,
        })];
        let ranker = DesignRanker::new().with_config(DesignRankConfig {
            min_batches: 2,
            max_batches: 4,
            batch_size: 4,
            rank_uncertainty_threshold: 1.0,
        });
        let ctx = ExecutionContext::for_tests(3);
        let eval = DesignEvaluationContext::<(), ()> {
            graphs: &graphs,
            effect_width: None,
            model_loglik: None,
            decisions: None,
            query_id_unlock: None,
            env_id_unlock: None,
            identified_under_intervention: None,
            graph_features: None,
        };
        let ranking = ranker
            .rank(
                &DesignObjective::IncreaseIdentificationProbability { query: q },
                &candidates,
                &eval,
                &ctx,
            )
            .expect("rank");
        assert!(ranking.ranked[0].score.abs() < 1e-12);
    }

    #[test]
    fn intervene_uses_identifier_flags() {
        use crate::candidate::ExperimentPlan;
        let graphs = toy_graphs();
        let q = QueryId::from_raw(0);
        let flags =
            [GraphIdentFlag::Identified, GraphIdentFlag::Identified, GraphIdentFlag::Identified];
        let candidates = vec![CandidateDesign::Intervene(ExperimentPlan {
            targets: Arc::from([VariableId::from_raw(0)]),
            cost: DesignCost::zero(),
            tag: 1,
        })];
        let ranker = DesignRanker::new().with_config(DesignRankConfig {
            min_batches: 2,
            max_batches: 4,
            batch_size: 4,
            rank_uncertainty_threshold: 1.0,
        });
        let ctx = ExecutionContext::for_tests(3);
        let eval = DesignEvaluationContext::<(), ()> {
            graphs: &graphs,
            effect_width: None,
            model_loglik: None,
            decisions: None,
            query_id_unlock: None,
            env_id_unlock: None,
            identified_under_intervention: Some(&flags),
            graph_features: None,
        };
        let ranking = ranker
            .rank(
                &DesignObjective::IncreaseIdentificationProbability { query: q },
                &candidates,
                &eval,
                &ctx,
            )
            .expect("rank");
        // Baseline identified mass 0.5 → post 1.0 → gain 0.5
        assert!((ranking.ranked[0].score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn measure_se_reduction_requires_column_spec() {
        // Baseline: X = [1, T] with orthogonal columns.
        let xtx = Arc::from([10.0_f64, 0.0, 0.0, 10.0]);
        let ew_bare = EffectWidthContext {
            xtx: Arc::clone(&xtx),
            sigma2: 1.0,
            treatment_col: 1,
            n: 10,
            measure_columns: None,
            intervention_design: None,
            environment_grams: None,
        };
        let measure = CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(2)]),
            cost: DesignCost::zero(),
            tag: 0,
        });
        assert_eq!(effect_width_reduction(&measure, &ew_bare).unwrap(), 0.0);

        let spec = MeasureColumnSpec {
            variable: VariableId::from_raw(2),
            cross: Arc::from([0.0, 0.0]),
            self_dot: 10.0,
            sigma2_after: Some(0.5),
        };
        let ew = EffectWidthContext {
            xtx,
            sigma2: 1.0,
            treatment_col: 1,
            n: 10,
            measure_columns: Some(Arc::from([spec])),
            intervention_design: None,
            environment_grams: None,
        };
        let red = effect_width_reduction(&measure, &ew).unwrap();
        assert!(red > 0.0, "expected positive SE reduction, got {red}");
    }

    #[test]
    fn eig_is_nonnegative_and_measure_beats_weak_sampling() {
        let graphs = toy_graphs();
        let mut rng = ExecutionContext::for_tests(99).rng.stream_for(StreamDomain::Design, 1);
        let measure = CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(1), VariableId::from_raw(2)]),
            cost: DesignCost::zero(),
            tag: 0,
        });
        let sampling = CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 1,
            cost: DesignCost::zero(),
            tag: 0,
        });
        let mut eig_m = 0.0;
        let mut eig_s = 0.0;
        for g in 0..graphs.n_samples {
            let em = eig_graph_entropy(&measure, &graphs, g, None, &mut rng);
            let es = eig_graph_entropy(&sampling, &graphs, g, None, &mut rng);
            assert!(em >= 0.0 && es >= 0.0);
            eig_m += em;
            eig_s += es;
        }
        assert!(eig_m > eig_s);
    }

    #[test]
    fn observe_environment_unlocks_identification() {
        use crate::candidate::EnvironmentPlan;
        let graphs = toy_graphs();
        let q = QueryId::from_raw(0);
        let env = EnvironmentId::from_raw(7);
        let unlock = [(q, Arc::from([env]))];
        let candidates = vec![CandidateDesign::ObserveEnvironment(EnvironmentPlan {
            environment: env,
            additional_rows: 50,
            cost: DesignCost::zero(),
            tag: 1,
        })];
        let ranker = DesignRanker::new().with_config(DesignRankConfig {
            min_batches: 2,
            max_batches: 4,
            batch_size: 4,
            rank_uncertainty_threshold: 1.0,
        });
        let ctx = ExecutionContext::for_tests(3);
        let eval = DesignEvaluationContext::<(), ()> {
            graphs: &graphs,
            effect_width: None,
            model_loglik: None,
            decisions: None,
            query_id_unlock: None,
            env_id_unlock: Some(&unlock),
            identified_under_intervention: None,
            graph_features: None,
        };
        let ranking = ranker
            .rank(
                &DesignObjective::IncreaseIdentificationProbability { query: q },
                &candidates,
                &eval,
                &ctx,
            )
            .expect("rank");
        // Baseline ID mass 0.5 → post 1.0 → gain 0.5
        assert!((ranking.ranked[0].score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn observe_environment_gram_pooling_differs_from_sampling_rate() {
        use crate::candidate::EnvironmentPlan;
        let env = EnvironmentId::from_raw(1);
        // Baseline orthogonal design; env gram doubles treatment information.
        let xtx = Arc::from([10.0_f64, 0.0, 0.0, 10.0]);
        let env_xtx = Arc::from([10.0_f64, 0.0, 0.0, 40.0]);
        let ew = EffectWidthContext {
            xtx: Arc::clone(&xtx),
            sigma2: 1.0,
            treatment_col: 1,
            n: 10,
            measure_columns: None,
            intervention_design: None,
            environment_grams: Some(Arc::from([EnvironmentGramSpec {
                environment: env,
                xtx: env_xtx,
                n: 10,
                sigma2: None,
            }])),
        };
        let observe = CandidateDesign::ObserveEnvironment(EnvironmentPlan {
            environment: env,
            additional_rows: 10, // same n increment as sampling below
            cost: DesignCost::zero(),
            tag: 0,
        });
        let sampling = CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 10,
            cost: DesignCost::zero(),
            tag: 0,
        });
        let red_obs = effect_width_reduction(&observe, &ew).unwrap();
        let red_samp = effect_width_reduction(&sampling, &ew).unwrap();
        assert!(red_obs > 0.0);
        assert!(red_samp > 0.0);
        assert!(
            (red_obs - red_samp).abs() > 1e-6,
            "env Gram pooling should differ from isotropic n-scaling: obs={red_obs} samp={red_samp}"
        );
    }

    // -- rank_uncertainty_ok (E2) -------------------------------------------------

    /// Draws `mean + amplitude·(±1)` alternating, `n` of them.
    fn alternating(mean: f64, amplitude: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| mean + if i % 2 == 0 { amplitude } else { -amplitude }).collect()
    }

    #[test]
    fn rank_uncertainty_ok_true_when_absolute_ci_narrow() {
        // Paired differences are 0.01 ± 0.001: se = 0.001·sqrt(4/3)/2 ≈ 5.8e-4, so the
        // half-width at z = 1.96 (≈ 1.1e-3) is far under the threshold (0.1).
        let draws = [alternating(10.0, 0.001, 4), vec![9.99; 4]];
        assert!(rank_uncertainty_ok(&draws, 1.96, 0.1));
    }

    /// E2: the top-2 gap (8) exceeds the paired half-width (≈ 1.13) even though that
    /// half-width itself exceeds the absolute threshold (1.0). The dead second branch in
    /// the pre-fix code blocked early stop here regardless of `gap`.
    #[test]
    fn rank_uncertainty_ok_true_when_top_two_well_separated() {
        let draws = [alternating(10.0, 1.0, 4), vec![2.0; 4]];
        assert!(rank_uncertainty_ok(&draws, 1.96, 1.0));
    }

    #[test]
    fn rank_uncertainty_ok_false_when_neither_condition_holds() {
        // Paired differences are 0.5 ± 1: se = sqrt(4/3)/2 ≈ 0.577, half-width ≈ 1.13 > 1.0
        // and the gap (0.5) does not clear it — genuinely uncertain, must not stop.
        let draws = [alternating(10.0, 1.0, 4), vec![9.5; 4]];
        assert!(!rank_uncertainty_ok(&draws, 1.96, 1.0));
    }

    /// Exact ties are unresolved, not "separated by 0 > 0": the gap test is `<=`.
    #[test]
    fn rank_uncertainty_ok_treats_exact_tie_within_threshold_as_resolved_only_by_width() {
        let tied = [vec![3.0; 8], vec![3.0; 8]];
        // Half-width 0 is within any threshold, so stopping is fine (nothing left to learn)...
        assert!(rank_uncertainty_ok(&tied, 1.96, 0.05));
        // ...but a paired difference that is exactly 0 ± noise above the threshold is not.
        let noisy_tie = [alternating(3.0, 1.0, 8), vec![3.0; 8]];
        assert!(!rank_uncertainty_ok(&noisy_tie, 1.96, 0.05));
    }

    /// The default schedule looks 61 times. A gap of 2.5 standard errors clears the
    /// fixed-sample 1.96 but not the look-adjusted boundary, so the sequential test must
    /// keep sampling where the repeated-1.96 rule would have stopped (and, on a truly tied
    /// pair, declared a separation ≈ 28 % of the time at these settings).
    #[test]
    fn early_stop_uses_look_adjusted_boundary() {
        let cfg = DesignRankConfig::default();
        assert_eq!(cfg.n_looks(), 61);
        let z = cfg.critical_value();
        // Independent evaluation of Φ⁻¹(1 − 0.025/61): the normal tail 4.098e-4 sits
        // between z = 3.34 (4.186e-4) and z = 3.35 (4.042e-4), i.e. z ≈ 3.346.
        assert!((z - 3.346).abs() < 5e-3, "z={z}");

        let n = 64usize;
        // Paired differences 0.315 ± 1: se = sqrt(64/63)/8, gap/se ≈ 2.5003.
        let draws = [alternating(0.315, 1.0, n), vec![0.0; n]];
        let se = (n as f64 / (n as f64 - 1.0)).sqrt() / (n as f64).sqrt();
        let ratio = 0.315 / se;
        assert!(ratio > 2.4 && ratio < 2.6, "ratio={ratio}");
        assert!(rank_uncertainty_ok(&draws, 1.96, 1e-3), "fixed-sample rule would stop");
        assert!(!rank_uncertainty_ok(&draws, z, 1e-3), "adjusted rule must keep sampling");
    }

    #[test]
    fn critical_value_is_the_fixed_sample_value_for_a_single_look() {
        let cfg = DesignRankConfig {
            min_batches: 4,
            max_batches: 4,
            batch_size: 4,
            rank_uncertainty_threshold: 0.0,
        };
        assert_eq!(cfg.n_looks(), 1);
        assert!((cfg.critical_value() - 1.959_963_984_540_054).abs() < 1e-9);
    }

    /// The reported flag uses the same adjusted boundary and the paired standard error.
    #[test]
    fn rank_uncertain_uses_adjusted_boundary_and_flags_exact_ties() {
        let graphs = toy_graphs();
        let q = QueryId::from_raw(0);
        let twin = |tag| {
            CandidateDesign::Measure(MeasurementPlan {
                variables: Arc::from([VariableId::from_raw(9)]),
                cost: DesignCost::zero(),
                tag,
            })
        };
        let eval = DesignEvaluationContext::<(), ()> {
            graphs: &graphs,
            effect_width: None,
            model_loglik: None,
            decisions: None,
            query_id_unlock: None,
            env_id_unlock: None,
            identified_under_intervention: None,
            graph_features: None,
        };
        let ranking = DesignRanker::new()
            .rank(
                &DesignObjective::IncreaseIdentificationProbability { query: q },
                &[twin(0), twin(1)],
                &eval,
                &ExecutionContext::for_tests(3),
            )
            .expect("rank");
        // Two identical exact candidates: gap == 0 is a tie, not a certain ordering.
        assert_eq!(ranking.ranked[0].score, ranking.ranked[1].score);
        assert!(ranking.ranked[0].rank_uncertain);
        assert!(!ranking.ranked[1].rank_uncertain);
    }

    // -- mc_stderr numerics -------------------------------------------------------

    /// 512 draws of `1e8 ± 0.5`: exact stderr is `0.5·sqrt(512/511)/sqrt(512)`. The old
    /// `Σx²/n − mean²` form returned ≈ 2× that (0.0885 vs 0.0454 at unit sd) and clamped
    /// to 0 beyond `mean ≈ 1e9`.
    #[test]
    fn score_stderr_survives_large_location() {
        let n = 512usize;
        let offset = 1e8;
        let draws = alternating(offset, 0.5, n);
        let mut acc = Welford::new();
        draws.iter().for_each(|&x| acc.push(x));
        let expected = 0.5 * (n as f64 / (n as f64 - 1.0)).sqrt() / (n as f64).sqrt();
        assert!(
            (acc.stderr_of_mean() / expected - 1.0).abs() < 1e-6,
            "{} vs {expected}",
            acc.stderr_of_mean()
        );
        // The paired difference between two such candidates is equally stable: the
        // differences are exactly ±0.25 alternating.
        let other = alternating(offset, 0.25, n);
        let (gap, se) = paired_difference(&draws, &other);
        let expected_pair = 0.25 * (n as f64 / (n as f64 - 1.0)).sqrt() / (n as f64).sqrt();
        assert!(gap.abs() < 1e-9, "gap={gap}");
        assert!((se / expected_pair - 1.0).abs() < 1e-6, "{se} vs {expected_pair}");
    }

    // -- context validation (malformed context is an error, not an exact 0) ----------

    #[test]
    fn singular_baseline_gram_is_an_error_not_an_exact_zero() {
        let graphs = toy_graphs();
        // Duplicated column: singular Gram.
        let ew = EffectWidthContext {
            xtx: Arc::from([1.0_f64, 1.0, 1.0, 1.0]),
            sigma2: 1.0,
            treatment_col: 1,
            n: 10,
            measure_columns: None,
            intervention_design: None,
            environment_grams: None,
        };
        let eval = DesignEvaluationContext::<(), ()> {
            graphs: &graphs,
            effect_width: Some(&ew),
            model_loglik: None,
            decisions: None,
            query_id_unlock: None,
            env_id_unlock: None,
            identified_under_intervention: None,
            graph_features: None,
        };
        let err = DesignRanker::new()
            .rank(
                &DesignObjective::ReduceEffectPosteriorWidth { query: QueryId::from_raw(0) },
                &[sampling(10, 0), sampling(20, 1)],
                &eval,
                &ExecutionContext::for_tests(1),
            )
            .unwrap_err();
        assert!(matches!(err, DesignError::Numerical(_)), "{err:?}");
    }

    #[test]
    fn mismatched_intervention_flags_are_an_error() {
        let graphs = toy_graphs();
        let flags = [GraphIdentFlag::Identified]; // 1 flag for 3 graphs
        let eval = DesignEvaluationContext::<(), ()> {
            graphs: &graphs,
            effect_width: None,
            model_loglik: None,
            decisions: None,
            query_id_unlock: None,
            env_id_unlock: None,
            identified_under_intervention: Some(&flags),
            graph_features: None,
        };
        let err = DesignRanker::new()
            .rank(
                &DesignObjective::IncreaseIdentificationProbability { query: QueryId::from_raw(0) },
                &[sampling(1, 0)],
                &eval,
                &ExecutionContext::for_tests(1),
            )
            .unwrap_err();
        assert!(matches!(err, DesignError::Shape(_)), "{err:?}");
    }

    /// A design that raises the SE is a negative reduction, not a neutral zero:
    /// orthogonal `[10, 10]` Gram, σ² = 1 → se₀ = sqrt(0.1); adding an orthogonal column
    /// with σ² raised to 4 gives se₁ = sqrt(0.4).
    #[test]
    fn se_increasing_design_scores_negative() {
        let ew = EffectWidthContext {
            xtx: Arc::from([10.0_f64, 0.0, 0.0, 10.0]),
            sigma2: 1.0,
            treatment_col: 1,
            n: 10,
            measure_columns: Some(Arc::from([MeasureColumnSpec {
                variable: VariableId::from_raw(2),
                cross: Arc::from([0.0, 0.0]),
                self_dot: 10.0,
                sigma2_after: Some(4.0),
            }])),
            intervention_design: None,
            environment_grams: None,
        };
        let measure = CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(2)]),
            cost: DesignCost::zero(),
            tag: 0,
        });
        let got = effect_width_reduction(&measure, &ew).unwrap();
        let expected = 0.1_f64.sqrt() - 0.4_f64.sqrt();
        assert!(expected < 0.0);
        assert!((got - expected).abs() < 1e-12, "got={got} expected={expected}");
    }

    #[test]
    fn malformed_candidate_gram_is_an_error() {
        let ew = EffectWidthContext {
            xtx: Arc::from([10.0_f64, 0.0, 0.0, 10.0]),
            sigma2: 1.0,
            treatment_col: 1,
            n: 10,
            measure_columns: Some(Arc::from([MeasureColumnSpec {
                variable: VariableId::from_raw(2),
                cross: Arc::from([0.0]), // length 1, baseline p = 2
                self_dot: 10.0,
                sigma2_after: None,
            }])),
            intervention_design: None,
            environment_grams: None,
        };
        let measure = CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(2)]),
            cost: DesignCost::zero(),
            tag: 0,
        });
        assert!(matches!(effect_width_reduction(&measure, &ew), Err(DesignError::Shape(_))));
    }

    // -- model_distinguish_score ---------------------------------------------------

    #[test]
    fn model_distinguish_score_matches_closed_form_two_models() {
        let m0 = ModelId::from_raw(0);
        let m1 = ModelId::from_raw(1);
        // n_draws = 1 makes the (otherwise random) draw index deterministic:
        // `rng.next_u64() % 1 == 0` always, regardless of RNG state.
        let ll = ModelLoglikDraws {
            models: Arc::from([m0, m1]),
            loglik: Arc::from([-12.0_f64, -9.5]),
            n_draws: 1,
        };
        let candidate = CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(0)]),
            cost: DesignCost::zero(),
            tag: 0,
        });
        // observation_reliability(Measure, k=1) = 1 - exp(-0.75 * 1).
        let expected_strength = 1.0 - (-0.75_f64).exp();
        let expected = expected_strength * (-12.0_f64 - (-9.5)).abs();
        let mut rng = ExecutionContext::for_tests(1).rng.stream_for(StreamDomain::Design, 0);
        let pairs = model_pair_rows(&ll, &[m0, m1]);
        let got = model_distinguish_score(&candidate, &ll, 2, &pairs, &mut rng);
        assert!((got - expected).abs() < 1e-9, "got={got} expected={expected}");
    }

    #[test]
    fn model_distinguish_score_zero_for_single_model() {
        let m0 = ModelId::from_raw(0);
        let ll =
            ModelLoglikDraws { models: Arc::from([m0]), loglik: Arc::from([-1.0_f64]), n_draws: 1 };
        let candidate = CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(0)]),
            cost: DesignCost::zero(),
            tag: 0,
        });
        let mut rng = ExecutionContext::for_tests(1).rng.stream_for(StreamDomain::Design, 0);
        let pairs = model_pair_rows(&ll, &[m0]);
        assert_eq!(model_distinguish_score(&candidate, &ll, 1, &pairs, &mut rng), 0.0);
    }

    // -- ReduceDecisionRegret -------------------------------------------------------

    struct LinearUtility;
    impl Utility<f64, f64> for LinearUtility {
        fn evaluate_batch(
            &self,
            actions: &[f64],
            outcomes: &[f64],
            out: &mut [f64],
        ) -> Result<(), crate::error::DesignError> {
            let n_o = outcomes.len();
            for (ai, a) in actions.iter().enumerate() {
                for (oi, o) in outcomes.iter().enumerate() {
                    out[ai * n_o + oi] = a * o;
                }
            }
            Ok(())
        }
    }

    struct Never;
    impl DecisionConstraint<f64, f64> for Never {
        fn name(&self) -> &str {
            "never"
        }
        fn satisfaction_batch(&self, actions: &[f64], _outcomes: &[f64], out: &mut [f64]) {
            out[..actions.len()].fill(0.0);
        }
    }

    /// Bet on a coin (action 1, pays θ − 0.5) or abstain (action 0, pays 0), with
    /// the coin's success probability θ drawn from three equally likely values.
    fn coin_registry(
        signal: Arc<dyn crate::preposterior::DecisionSignal<f64>>,
    ) -> DecisionRegistry<f64, f64> {
        struct Bet;
        impl Utility<f64, f64> for Bet {
            fn evaluate_batch(
                &self,
                actions: &[f64],
                outcomes: &[f64],
                out: &mut [f64],
            ) -> Result<(), crate::error::DesignError> {
                for (ai, a) in actions.iter().enumerate() {
                    for (oi, theta) in outcomes.iter().enumerate() {
                        out[ai * outcomes.len() + oi] = a * (theta - 0.45);
                    }
                }
                Ok(())
            }
        }
        DecisionRegistry {
            problems: vec![Some(DecisionProblem::new(vec![0.0, 1.0], Arc::new(Bet), vec![]))],
            prior: DecisionPrior::Draws(vec![0.2, 0.5, 0.8]),
            signal,
        }
    }

    fn sampling(n: u64, tag: u64) -> CandidateDesign {
        CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: n,
            cost: DesignCost::zero(),
            tag,
        })
    }

    fn decision_eval<'a>(
        graphs: &'a WeightedGraphSamples,
        registry: &'a DecisionRegistry<f64, f64>,
    ) -> DesignEvaluationContext<'a, f64, f64> {
        DesignEvaluationContext {
            graphs,
            effect_width: None,
            model_loglik: None,
            decisions: Some(registry),
            query_id_unlock: None,
            env_id_unlock: None,
            identified_under_intervention: None,
            graph_features: None,
        }
    }

    fn fixed_ranker() -> DesignRanker {
        DesignRanker::new().with_config(DesignRankConfig {
            min_batches: 4,
            max_batches: 4,
            batch_size: 4,
            rank_uncertainty_threshold: 0.0,
        })
    }

    /// The design's sample size must drive the score: more coin flips are worth
    /// strictly more. The previous score clamped every candidate to the same
    /// resample count and returned the same value for every sample size.
    #[test]
    fn decision_regret_score_increases_with_design_sample_size() {
        let graphs = toy_graphs();
        let registry = coin_registry(Arc::new(crate::preposterior::BinomialSignal));
        let candidates = vec![sampling(1, 0), sampling(10, 1), sampling(100, 2), sampling(0, 3)];
        let ranking = fixed_ranker()
            .rank(
                &DesignObjective::ReduceDecisionRegret { decision: DecisionProblemId::from_raw(0) },
                &candidates,
                &decision_eval(&graphs, &registry),
                &ExecutionContext::for_tests(5),
            )
            .expect("rank");
        let score = |i: usize| {
            ranking.ranked.iter().find(|r| r.candidate_index == i).expect("ranked").score
        };
        assert_eq!(score(3), 0.0);
        assert!(0.0 < score(0) && score(0) < score(1) && score(1) < score(2), "{ranking:?}");
        let problem = registry.problems[0].as_ref().unwrap();
        let analysis =
            PreposteriorAnalysis::new(problem, &registry.prior, registry.signal.as_ref()).unwrap();
        for (i, n) in [(0, 1), (1, 10), (2, 100)] {
            let exact = analysis.exact_evsi(n).unwrap().unwrap();
            assert!((score(i) - exact).abs() < 1e-15);
        }
        assert!(score(2) < analysis.expected_value_of_perfect_information());
        for ranked in ranking.ranked.iter() {
            assert_eq!(ranked.evaluation, ScoreEvaluation::Exact);
            assert_eq!(ranked.monte_carlo.stderr, 0.0);
            assert_eq!(
                ranked.implemented_functional.as_ref(),
                "preposterior_expected_value_of_sample_information"
            );
        }
    }

    #[test]
    fn decision_regret_continuous_signal_is_labelled_monte_carlo() {
        let graphs = toy_graphs();
        let signal = crate::preposterior::GaussianMeanSignal::new(0.25).unwrap();
        let registry = coin_registry(Arc::new(signal));
        let ranking = fixed_ranker()
            .rank(
                &DesignObjective::ReduceDecisionRegret { decision: DecisionProblemId::from_raw(0) },
                &[sampling(4, 0), sampling(0, 1)],
                &decision_eval(&graphs, &registry),
                &ExecutionContext::for_tests(5),
            )
            .expect("rank");
        let top = &ranking.ranked[0];
        assert_eq!(top.candidate_index, 0);
        assert_eq!(top.evaluation, ScoreEvaluation::MonteCarlo);
        assert!(top.monte_carlo.stderr > 0.0 && top.score > 0.0);
        assert_eq!(ranking.ranked[1].evaluation, ScoreEvaluation::Exact);
    }

    #[test]
    fn decision_regret_records_candidates_without_a_sample_size() {
        let graphs = toy_graphs();
        let registry = coin_registry(Arc::new(crate::preposterior::BinomialSignal));
        let measure = CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(0)]),
            cost: DesignCost::zero(),
            tag: 0,
        });
        let ranking = fixed_ranker()
            .rank(
                &DesignObjective::ReduceDecisionRegret { decision: DecisionProblemId::from_raw(0) },
                &[measure, sampling(3, 1)],
                &decision_eval(&graphs, &registry),
                &ExecutionContext::for_tests(5),
            )
            .expect("rank");
        assert_eq!(ranking.ranked.len(), 1);
        assert_eq!(ranking.ranked[0].candidate_index, 1);
        assert_eq!(ranking.violations.len(), 1);
        assert_eq!(ranking.violations[0].candidate_index, 0);
        assert_eq!(ranking.violations[0].constraint.as_ref(), "unlicensed_candidate");
    }

    #[test]
    fn decision_regret_refuses_unknown_and_inadmissible_problems() {
        let graphs = toy_graphs();
        let registry = DecisionRegistry::<f64, f64> {
            problems: vec![
                None,
                Some(DecisionProblem::new(
                    vec![1.0, 2.0],
                    Arc::new(LinearUtility),
                    vec![Arc::new(Never) as Arc<dyn DecisionConstraint<f64, f64>>],
                )),
            ],
            prior: DecisionPrior::Draws(vec![-1.0, 3.0]),
            signal: Arc::new(crate::preposterior::GaussianMeanSignal::new(1.0).unwrap()),
        };
        let rank = |raw: u32| {
            fixed_ranker().rank(
                &DesignObjective::ReduceDecisionRegret {
                    decision: DecisionProblemId::from_raw(raw),
                },
                &[sampling(3, 0)],
                &decision_eval(&graphs, &registry),
                &ExecutionContext::for_tests(5),
            )
        };
        assert!(matches!(rank(0), Err(DesignError::Config(_))));
        assert!(matches!(rank(7), Err(DesignError::Config(_))));
        assert!(matches!(rank(1), Err(DesignError::NoAdmissibleAction(_))));
    }
}
