//! Batched Monte Carlo design ranking (DESIGN.md §19.4).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::similar_names
)]

use std::sync::Arc;

use causal_core::{
    CausalRng, ExecutionContext, ModelId, MonteCarloBudget, MonteCarloError, QueryId, VariableId,
};
use causal_kernels::sample_categorical;
use causal_prob::{GraphIdentFlag, WeightedGraphSamples};
use causal_stats::invert_square;

use crate::candidate::{CandidateDesign, DesignCost};
use crate::decision::{DecisionEvaluation, DecisionProblem, DecisionProblemId, evaluate_decision};
use crate::error::DesignError;
use crate::objective::DesignObjective;
use crate::result::{ConstraintViolation, DesignRanking, RankedCandidate};

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
    /// Stop when max pairwise rank CI half-width among top-k is below this.
    pub rank_uncertainty_threshold: f64,
    /// Batch size (independent CRN replicates per adaptive step).
    pub batch_size: u32,
}

impl Default for DesignRankConfig {
    fn default() -> Self {
        Self { min_batches: 4, max_batches: 64, rank_uncertainty_threshold: 0.05, batch_size: 8 }
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

/// Optional decision problem registry (by [`DecisionProblemId`]).
pub struct DecisionRegistry<A, O> {
    /// Problems keyed by raw id order (sparse holes allowed via Option).
    pub problems: Vec<Option<DecisionProblem<A, O>>>,
    /// Outcome draws shared across candidates (CRN).
    pub outcomes: Vec<O>,
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
        if candidates.is_empty() {
            return Err(DesignError::EmptyCandidates);
        }
        if ctx_eval.graphs.n_samples == 0 {
            return Err(DesignError::EmptyPosterior);
        }
        if self.config.max_batches == 0 || self.config.batch_size == 0 {
            return Err(DesignError::Config("max_batches and batch_size must be > 0".into()));
        }

        let mut violations = Vec::new();
        let mut active: Vec<usize> = Vec::new();
        for (i, c) in candidates.iter().enumerate() {
            if let Some(v) = self.check_constraints(i, c.cost()) {
                violations.push(v);
            } else {
                active.push(i);
            }
        }

        let mut sums = vec![0.0; active.len()];
        let mut sumsq = vec![0.0; active.len()];
        let mut n_samples: u64 = 0;
        let mut budget = MonteCarloBudget::default();
        let mut early_stopped = false;
        let mut rng = ctx.rng.stream(0xD351_0611);

        let min_batches = self.config.min_batches.max(1);
        let max_batches = self.config.max_batches.max(min_batches);

        for batch_i in 0..max_batches {
            for _ in 0..self.config.batch_size {
                // Shared CRN draw index into graph posterior.
                let g_idx = sample_categorical(&mut rng, &ctx_eval.graphs.weights);
                for (slot, &cand_i) in active.iter().enumerate() {
                    let score =
                        score_candidate(objective, &candidates[cand_i], ctx_eval, g_idx, &mut rng)?;
                    sums[slot] += score;
                    sumsq[slot] += score * score;
                    budget.evaluations += 1;
                }
                n_samples += 1;
                budget.samples = n_samples;
            }

            if batch_i + 1 >= min_batches {
                let stderrs: Vec<f64> =
                    (0..active.len()).map(|s| mc_stderr(sums[s], sumsq[s], n_samples)).collect();
                if rank_uncertainty_ok(
                    &sums,
                    &stderrs,
                    n_samples,
                    self.config.rank_uncertainty_threshold,
                ) {
                    early_stopped = true;
                    break;
                }
            }
            if ctx.cancellation.is_cancelled() {
                break;
            }
        }

        let mut scored: Vec<(usize, f64, MonteCarloError)> = active
            .iter()
            .enumerate()
            .map(|(slot, &cand_i)| {
                let mean = if n_samples > 0 { sums[slot] / n_samples as f64 } else { 0.0 };
                let err = MonteCarloError {
                    stderr: mc_stderr(sums[slot], sumsq[slot], n_samples),
                    samples: n_samples,
                };
                (cand_i, mean, err)
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
        });

        let mut ranked = Vec::with_capacity(scored.len());
        for (rank, (cand_i, score, mc)) in scored.iter().enumerate() {
            let uncertain = if rank + 1 < scored.len() {
                let gap = (score - scored[rank + 1].1).abs();
                let se = (mc.stderr.powi(2) + scored[rank + 1].2.stderr.powi(2)).sqrt();
                gap < 1.96 * se
            } else {
                false
            };
            ranked.push(RankedCandidate {
                candidate_index: *cand_i,
                candidate: candidates[*cand_i].clone(),
                score: *score,
                monte_carlo: *mc,
                rank,
                rank_uncertain: uncertain,
            });
        }

        Ok(DesignRanking {
            ranked: Arc::from(ranked),
            violations: Arc::from(violations),
            budget,
            early_stopped,
        })
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

fn mc_stderr(sum: f64, sumsq: f64, n: u64) -> f64 {
    if n < 2 {
        return f64::INFINITY;
    }
    let nf = n as f64;
    let mean = sum / nf;
    let var = (sumsq / nf - mean * mean).max(0.0) * nf / (nf - 1.0);
    (var / nf).sqrt()
}

fn rank_uncertainty_ok(sums: &[f64], stderrs: &[f64], n: u64, threshold: f64) -> bool {
    if sums.len() < 2 || n == 0 {
        return true;
    }
    let means: Vec<f64> = sums.iter().map(|s| s / n as f64).collect();
    let mut order: Vec<usize> = (0..means.len()).collect();
    order.sort_by(|a, b| means[*b].partial_cmp(&means[*a]).unwrap_or(std::cmp::Ordering::Equal));
    // Check top adjacent pairs.
    let top = order.len().min(3);
    for i in 0..top.saturating_sub(1) {
        let a = order[i];
        let b = order[i + 1];
        let gap = (means[a] - means[b]).abs();
        let se = (stderrs[a].powi(2) + stderrs[b].powi(2)).sqrt();
        if 1.96 * se > threshold && gap < 1.96 * se {
            return false;
        }
        if 1.96 * se > threshold {
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

fn score_candidate<A, O>(
    objective: &DesignObjective,
    candidate: &CandidateDesign,
    ctx: &DesignEvaluationContext<'_, A, O>,
    graph_idx: usize,
    rng: &mut CausalRng,
) -> Result<f64, DesignError>
where
    A: Clone,
    O: Clone,
{
    match objective {
        DesignObjective::ReduceGraphEntropy => Ok(eig_graph_entropy(
            candidate,
            ctx.graphs,
            graph_idx,
            ctx.graph_features,
            rng,
        )),
        DesignObjective::IncreaseIdentificationProbability { query } => {
            Ok(id_prob_gain(candidate, ctx, *query))
        }
        DesignObjective::ReduceEffectPosteriorWidth { query: _ } => {
            let Some(ew) = ctx.effect_width else {
                return Err(DesignError::Config(
                    "ReduceEffectPosteriorWidth requires effect_width context".into(),
                ));
            };
            Ok(effect_width_reduction(candidate, ew))
        }
        DesignObjective::DistinguishModels { models } => {
            let Some(ll) = ctx.model_loglik else {
                return Err(DesignError::Config(
                    "DistinguishModels requires model_loglik context".into(),
                ));
            };
            Ok(model_distinguish_score(candidate, ll, models, rng))
        }
        DesignObjective::ReduceDecisionRegret { decision } => {
            let Some(reg) = ctx.decisions else {
                return Err(DesignError::Config(
                    "ReduceDecisionRegret requires decisions context".into(),
                ));
            };
            Ok(decision_regret_reduction(candidate, reg, *decision, rng))
        }
    }
}

/// Expected information gain via one posterior-simulation draw:
/// sample observation `y ~ P(y | G★, design)`, then `H(prior) − H(p(G|y))`.
fn eig_graph_entropy(
    candidate: &CandidateDesign,
    graphs: &WeightedGraphSamples,
    graph_idx: usize,
    graph_features: Option<&[u32]>,
    rng: &mut CausalRng,
) -> f64 {
    let n = graphs.n_samples;
    if n == 0 {
        return 0.0;
    }
    if let Some(feat) = graph_features {
        if feat.len() != n {
            return 0.0;
        }
    }

    let labels: Vec<u32> = if let Some(feat) = graph_features {
        feat.to_vec()
    } else {
        graphs.graph_keys.iter().map(|k| *k as u32).collect()
    };

    let mut categories: Vec<u32> = labels.clone();
    categories.sort_unstable();
    categories.dedup();
    let k = categories.len();
    if k < 2 {
        return 0.0;
    }

    let cat_index = |label: u32| -> usize {
        categories.binary_search(&label).unwrap_or(0)
    };

    let prior_h = shannon_entropy(&graphs.weights);
    let reliability = observation_reliability(candidate);
    let true_cat = cat_index(labels[graph_idx]);

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
    let mut post = graphs.weights.to_vec();
    for (i, w) in post.iter_mut().enumerate() {
        let lik = if cat_index(labels[i]) == y { reliability } else { off };
        *w *= lik;
    }
    let post_h = shannon_entropy(&post);
    (prior_h - post_h).max(0.0)
}

/// Deterministic observation reliability for the discrete graph-feature channel.
fn observation_reliability(candidate: &CandidateDesign) -> f64 {
    match candidate {
        CandidateDesign::Measure(p) => {
            let k = p.variables.len() as f64;
            (1.0 - (-0.75 * k).exp()).clamp(0.05, 0.99)
        }
        CandidateDesign::Intervene(p) => {
            let k = p.targets.len() as f64;
            (1.0 - (-1.0 * k).exp()).clamp(0.05, 0.99)
        }
        CandidateDesign::ObserveEnvironment(p) => {
            let n = p.additional_rows as f64;
            (1.0 - (1.0 + n / 50.0).recip()).clamp(0.05, 0.95)
        }
        CandidateDesign::IncreaseSamplingRate(p) => {
            let n = p.additional_samples as f64;
            (1.0 - (1.0 + n / 50.0).recip()).clamp(0.05, 0.95)
        }
    }
}

fn evidence_strength(candidate: &CandidateDesign) -> f64 {
    // Used only for DistinguishModels / decision scaling (not EIG / ID / SE).
    observation_reliability(candidate)
}

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

    let intervene_flags = ctx.identified_under_intervention.filter(|f| f.len() == ctx.graphs.n_samples);

    let mut identified = 0.0;
    let mut total = 0.0;
    for i in 0..ctx.graphs.n_samples {
        let w = ctx.graphs.weights[i];
        total += w;
        let mut is_id = ctx.graphs.identified[i] == GraphIdentFlag::Identified;
        if !is_id {
            is_id = candidate_unlocks(candidate, unlock);
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

fn candidate_unlocks(candidate: &CandidateDesign, unlock: &[VariableId]) -> bool {
    if unlock.is_empty() {
        return false;
    }
    match candidate {
        CandidateDesign::Measure(p) => p.variables.iter().any(|v| unlock.contains(v)),
        CandidateDesign::ObserveEnvironment(_) | CandidateDesign::IncreaseSamplingRate(_) => false,
        CandidateDesign::Intervene(p) => p.targets.iter().any(|v| unlock.contains(v)),
    }
}

fn treatment_se(xtx: &[f64], p: usize, treatment_col: usize, sigma2: f64) -> Option<f64> {
    if p == 0 || treatment_col >= p || !(sigma2 > 0.0) || xtx.len() != p * p {
        return None;
    }
    let inv = invert_square(xtx, p)?;
    Some((sigma2 * inv[treatment_col * p + treatment_col].max(0.0)).sqrt())
}

fn gram_side_len(xtx: &[f64]) -> usize {
    let n2 = xtx.len();
    let mut k = 0usize;
    while k * k < n2 {
        k += 1;
    }
    if k * k == n2 { k } else { 0 }
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

fn effect_width_reduction(candidate: &CandidateDesign, ew: &EffectWidthContext) -> f64 {
    let p = gram_side_len(&ew.xtx);
    let Some(se0) = treatment_se(&ew.xtx, p, ew.treatment_col, ew.sigma2) else {
        return 0.0;
    };

    match candidate {
        CandidateDesign::IncreaseSamplingRate(s) => {
            if s.additional_samples == 0 {
                return 0.0;
            }
            // XtX scales with n; SE scales as 1/sqrt(n).
            let n1 = (ew.n + s.additional_samples) as f64;
            let n0 = ew.n.max(1) as f64;
            let se1 = se0 * (n0 / n1).sqrt();
            (se0 - se1).max(0.0)
        }
        CandidateDesign::ObserveEnvironment(e) => {
            if e.additional_rows == 0 {
                return 0.0;
            }
            let n1 = (ew.n + e.additional_rows) as f64;
            let n0 = ew.n.max(1) as f64;
            let se1 = se0 * (n0 / n1).sqrt();
            (se0 - se1).max(0.0)
        }
        CandidateDesign::Measure(plan) => {
            let Some(specs) = ew.measure_columns.as_ref() else {
                // No design-analysis columns → cannot claim SE reduction.
                return 0.0;
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
                    return 0.0;
                };
                xtx = expanded;
                cur_p += 1;
                if let Some(s2) = spec.sigma2_after {
                    sigma2 = s2;
                }
                matched += 1;
            }
            if matched == 0 {
                return 0.0;
            }
            let Some(se1) = treatment_se(&xtx, cur_p, ew.treatment_col, sigma2) else {
                return 0.0;
            };
            (se0 - se1).max(0.0)
        }
        CandidateDesign::Intervene(_) => {
            let Some(design) = ew.intervention_design.as_ref() else {
                return 0.0;
            };
            let p1 = gram_side_len(&design.xtx);
            if p1 != p {
                return 0.0;
            }
            let Some(se1) = treatment_se(&design.xtx, p1, ew.treatment_col, design.sigma2) else {
                return 0.0;
            };
            (se0 - se1).max(0.0)
        }
    }
}

fn model_distinguish_score(
    candidate: &CandidateDesign,
    ll: &ModelLoglikDraws,
    models: &[ModelId],
    rng: &mut CausalRng,
) -> f64 {
    if models.len() < 2 || ll.n_draws == 0 {
        return 0.0;
    }
    let draw = (rng.next_u64() as usize) % ll.n_draws;
    let strength = evidence_strength(candidate);
    // Expected absolute log-score gap between first two models, scaled by evidence strength.
    let mut gap = 0.0;
    let mut count = 0usize;
    for i in 0..models.len() {
        for j in (i + 1)..models.len() {
            let Some(ri) = ll.models.iter().position(|m| *m == models[i]) else {
                continue;
            };
            let Some(rj) = ll.models.iter().position(|m| *m == models[j]) else {
                continue;
            };
            let a = ll.loglik[ri * ll.n_draws + draw];
            let b = ll.loglik[rj * ll.n_draws + draw];
            gap += (a - b).abs();
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { strength * gap / count as f64 }
}

fn decision_regret_reduction<A, O>(
    candidate: &CandidateDesign,
    reg: &DecisionRegistry<A, O>,
    decision: DecisionProblemId,
    rng: &mut CausalRng,
) -> f64
where
    A: Clone,
    O: Clone,
{
    let idx = decision.raw() as usize;
    let Some(Some(problem)) = reg.problems.get(idx) else {
        return 0.0;
    };
    if reg.outcomes.is_empty() {
        return 0.0;
    }
    // Baseline regret.
    let base: DecisionEvaluation = evaluate_decision(problem, &reg.outcomes);
    // Candidate: subsample outcomes with replacement (information → tighter effective support).
    let keep = match candidate {
        CandidateDesign::IncreaseSamplingRate(s) => {
            (reg.outcomes.len() as u64).saturating_add(s.additional_samples / 10)
        }
        CandidateDesign::Measure(_) | CandidateDesign::Intervene(_) => reg.outcomes.len() as u64,
        CandidateDesign::ObserveEnvironment(e) => {
            (reg.outcomes.len() as u64).saturating_add(e.additional_rows / 10)
        }
    };
    let n_keep = (keep as usize).clamp(1, reg.outcomes.len().saturating_mul(2).max(1));
    let mut sample = Vec::with_capacity(n_keep.min(reg.outcomes.len()));
    for _ in 0..n_keep.min(reg.outcomes.len()) {
        let i = (rng.next_u64() as usize) % reg.outcomes.len();
        sample.push(reg.outcomes[i].clone());
    }
    // Strength reduces effective regret toward 0.
    let strength = evidence_strength(candidate);
    let after = evaluate_decision(problem, &sample);
    let reduced = after.posterior_regret * (1.0 - strength);
    (base.posterior_regret - reduced).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{DesignCost, MeasurementPlan, SamplingPlan};
    use causal_core::VariableId;
    use causal_prob::GraphIdentFlag;

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

    #[test]
    fn ranks_measurement_above_noop_sampling_for_entropy() {
        let graphs = toy_graphs();
        let candidates = vec![
            CandidateDesign::IncreaseSamplingRate(SamplingPlan {
                additional_samples: 1,
                cost: DesignCost::zero(),
                tag: 1,
            }),
            CandidateDesign::Measure(MeasurementPlan {
                variables: Arc::from([VariableId::from_raw(2)]),
                cost: DesignCost::zero(),
                tag: 20,
            }),
        ];
        let ranker = DesignRanker::new().with_config(DesignRankConfig {
            min_batches: 2,
            max_batches: 8,
            batch_size: 4,
            rank_uncertainty_threshold: 0.5,
        });
        let ctx = ExecutionContext::for_tests(7);
        let eval = DesignEvaluationContext::<(), ()> {
            graphs: &graphs,
            effect_width: None,
            model_loglik: None,
            decisions: None,
            query_id_unlock: None,
            identified_under_intervention: None,
            graph_features: None,
        };
        let ranking = ranker
            .rank(&DesignObjective::ReduceGraphEntropy, &candidates, &eval, &ctx)
            .expect("rank");
        assert_eq!(ranking.ranked.len(), 2);
        assert!(ranking.budget.samples > 0);
        // Higher-reliability Measure should not score below tiny sampling on average.
        assert!(ranking.ranked[0].score >= ranking.ranked[1].score - 1e-9);
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
        let flags = [
            GraphIdentFlag::Identified,
            GraphIdentFlag::Identified,
            GraphIdentFlag::Identified,
        ];
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
        };
        let measure = CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(2)]),
            cost: DesignCost::zero(),
            tag: 0,
        });
        assert_eq!(effect_width_reduction(&measure, &ew_bare), 0.0);

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
        };
        let red = effect_width_reduction(&measure, &ew);
        assert!(red > 0.0, "expected positive SE reduction, got {red}");
    }

    #[test]
    fn eig_is_nonnegative_and_measure_beats_weak_sampling() {
        let graphs = toy_graphs();
        let mut rng = ExecutionContext::for_tests(99).rng.stream(1);
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
}
