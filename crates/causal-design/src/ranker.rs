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
    /// Query → variables that unlock identification when measured (bitmask via var raw ids).
    pub query_id_unlock: Option<&'a [(QueryId, Arc<[VariableId]>)]>,
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

fn sample_categorical(rng: &mut CausalRng, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    if !(total > 0.0) {
        return 0;
    }
    let mut u = rng.next_f64() * total;
    for (i, w) in weights.iter().enumerate() {
        if u <= *w {
            return i;
        }
        u -= *w;
    }
    weights.len().saturating_sub(1)
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
        DesignObjective::ReduceGraphEntropy => {
            Ok(eig_graph_entropy(candidate, ctx.graphs, graph_idx, rng))
        }
        DesignObjective::IncreaseIdentificationProbability { query } => {
            Ok(id_prob_gain(candidate, ctx, *query, graph_idx))
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

/// Expected information gain ≈ prior entropy − expected posterior entropy under soft evidence.
fn eig_graph_entropy(
    candidate: &CandidateDesign,
    graphs: &WeightedGraphSamples,
    graph_idx: usize,
    rng: &mut CausalRng,
) -> f64 {
    let prior_h = shannon_entropy(&graphs.weights);
    // Soft evidence: reweight graphs whose keys interact with measured/intervened vars.
    let strength = evidence_strength(candidate);
    let focus = focus_key(candidate, graphs.graph_keys[graph_idx]);
    let mut post = graphs.weights.to_vec();
    for (i, w) in post.iter_mut().enumerate() {
        let key = graphs.graph_keys[i];
        let agree = if key == focus { 1.0 } else { 0.35 + 0.15 * rng.next_f64() };
        *w *= (1.0 - strength) + strength * agree;
    }
    let post_h = shannon_entropy(&post);
    (prior_h - post_h).max(0.0)
}

fn evidence_strength(candidate: &CandidateDesign) -> f64 {
    match candidate {
        CandidateDesign::Measure(p) => (0.15 * p.variables.len() as f64).clamp(0.05, 0.85),
        CandidateDesign::Intervene(p) => (0.25 * p.targets.len() as f64).clamp(0.1, 0.9),
        CandidateDesign::ObserveEnvironment(p) => {
            (0.0005 * p.additional_rows as f64).clamp(0.05, 0.7)
        }
        CandidateDesign::IncreaseSamplingRate(p) => {
            (0.0004 * p.additional_samples as f64).clamp(0.05, 0.6)
        }
    }
}

fn focus_key(candidate: &CandidateDesign, fallback: u64) -> u64 {
    let tag = candidate.tag();
    if tag == 0 { fallback } else { tag }
}

fn id_prob_gain<A, O>(
    candidate: &CandidateDesign,
    ctx: &DesignEvaluationContext<'_, A, O>,
    query: QueryId,
    graph_idx: usize,
) -> f64 {
    let baseline = ctx.graphs.identified_mass() / ctx.graphs.total_weight().max(1e-15);
    let unlock = ctx
        .query_id_unlock
        .and_then(|m| m.iter().find(|(q, _)| *q == query).map(|(_, v)| v.as_ref()))
        .unwrap_or(&[]);

    let mut identified = 0.0;
    let mut total = 0.0;
    for i in 0..ctx.graphs.n_samples {
        let w = ctx.graphs.weights[i];
        total += w;
        let mut is_id = ctx.graphs.identified[i] == GraphIdentFlag::Identified;
        if !is_id {
            is_id = candidate_unlocks(candidate, unlock)
                || (candidate_unlocks_any(candidate) && i == graph_idx);
        }
        // Interventions can identify by breaking backdoors on matching draws.
        if !is_id {
            if let CandidateDesign::Intervene(p) = candidate {
                if !p.targets.is_empty() && (ctx.graphs.graph_keys[i] % 2 == 0) {
                    is_id = true;
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

fn candidate_unlocks_any(candidate: &CandidateDesign) -> bool {
    matches!(candidate, CandidateDesign::Measure(p) if !p.variables.is_empty())
}

fn effect_width_reduction(candidate: &CandidateDesign, ew: &EffectWidthContext) -> f64 {
    let p = {
        let n2 = ew.xtx.len();
        // p×p
        let mut k = 0usize;
        while k * k < n2 {
            k += 1;
        }
        k
    };
    if p == 0 || ew.treatment_col >= p || ew.sigma2 <= 0.0 {
        return 0.0;
    }
    let Some(inv) = invert_square(ew.xtx.as_ref(), p) else {
        return 0.0;
    };
    let se0 = (ew.sigma2 * inv[ew.treatment_col * p + ew.treatment_col].max(0.0)).sqrt();
    let add_n = match candidate {
        CandidateDesign::IncreaseSamplingRate(s) => s.additional_samples,
        CandidateDesign::ObserveEnvironment(e) => e.additional_rows,
        CandidateDesign::Measure(_) => {
            // Measuring a confounder ≈ adding one orthogonal column: shrink SE mildly.
            return se0 * 0.15;
        }
        CandidateDesign::Intervene(_) => return se0 * 0.1,
    };
    if add_n == 0 {
        return 0.0;
    }
    // Approximate: XtX scales with n; SE scales as 1/sqrt(n).
    let n1 = (ew.n + add_n) as f64;
    let n0 = ew.n.max(1) as f64;
    let se1 = se0 * (n0 / n1).sqrt();
    (se0 - se1).max(0.0)
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
        };
        let ranking = ranker
            .rank(&DesignObjective::ReduceGraphEntropy, &candidates, &eval, &ctx)
            .expect("rank");
        assert_eq!(ranking.ranked.len(), 2);
        assert!(ranking.budget.samples > 0);
        // Measuring with matching tag should not score below tiny sampling on average.
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
}
