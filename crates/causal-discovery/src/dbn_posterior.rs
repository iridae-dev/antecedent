//! Bounded-lag dynamic Bayesian network posterior search.
//!
//! Template = contemporaneous DAG on `p` variables plus optional lag-`ℓ` edges
//! `i → j` for `ℓ ∈ 1..=max_lag`. Scoring uses Gaussian BIC on the aligned
//! lagged design (`t = max_lag .. T-1`).
//!
//! Exact enumeration is used only when `p ≤ `[`DBN_EXACT_MAX_VARS`],
//! `max_lag ≤ `[`DBN_EXACT_MAX_LAG`], and the lag-bit space is small; otherwise
//! the engine falls back to template MCMC automatically (or when
//! `force_mcmc` is set).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

use std::sync::Arc;

use causal_core::{ExecutionContext, VariableId};
use causal_data::{TableView, TimeSeriesData};
use causal_state::{
    GraphScoreCacheKey, GraphScoreData, GraphScoreFamily, LocalScoreCache,
};

use crate::error::DiscoveryError;
use crate::exact_enumeration::enumerate_unique_dags;
use crate::graph_posterior::{
    accumulate_marginals, analytic_graph_diagnostics, has_edge, kish_ess, log_prior_mask,
    mask_is_dag, normalize_log_weights, n_directed_edges, parents_of, set_edge, GraphPosterior,
    GraphPrior, EXACT_ENUM_MAX_NODES,
};

/// Max variables for exact DBN template enumeration (`p ≤ 4`).
pub const DBN_EXACT_MAX_VARS: usize = 4;
/// Max lag for exact DBN enumeration (`max_lag ≤ 2`).
pub const DBN_EXACT_MAX_LAG: u32 = 2;

/// Bounded-lag DBN posterior over a stationary template.
///
/// Exact path: `p ≤ DBN_EXACT_MAX_VARS` and `max_lag ≤ DBN_EXACT_MAX_LAG`
/// (plus a small lag-bit budget). Larger templates use MCMC.
#[derive(Clone, Debug)]
pub struct DbnPosterior {
    /// Maximum lag (`≥ 1`).
    pub max_lag: u32,
    /// Force MCMC even when exact would fit.
    pub force_mcmc: bool,
    /// MCMC chains / warmup / draws when exact is unavailable.
    pub n_chains: u32,
    /// Warmup per chain.
    pub n_warmup: u32,
    /// Draws per chain.
    pub n_draws: u32,
}

impl Default for DbnPosterior {
    fn default() -> Self {
        Self::new(1)
    }
}

impl DbnPosterior {
    /// Construct with `max_lag` (clamped to ≥ 1).
    #[must_use]
    pub fn new(max_lag: u32) -> Self {
        Self {
            max_lag: max_lag.max(1),
            force_mcmc: false,
            n_chains: 2,
            n_warmup: 200,
            n_draws: 400,
        }
    }

    /// Force structure-style MCMC on the template.
    #[must_use]
    pub fn with_force_mcmc(mut self, force: bool) -> Self {
        self.force_mcmc = force;
        self
    }

    /// MCMC schedule for large templates.
    #[must_use]
    pub fn with_mcmc_schedule(mut self, n_chains: u32, n_warmup: u32, n_draws: u32) -> Self {
        self.n_chains = n_chains.max(2);
        self.n_warmup = n_warmup;
        self.n_draws = n_draws.max(4);
        self
    }

    /// Infer DBN template posterior from a time series.
    ///
    /// # Errors
    ///
    /// Short series, unsupported size, score, or empty support.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        prior: &GraphPrior,
        score_family: GraphScoreFamily,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosterior, DiscoveryError> {
        let _ = ctx;
        prior.constraints.validate()?;
        if !matches!(score_family, GraphScoreFamily::GaussianBic) {
            return Err(DiscoveryError::unsupported(
                "DBN posterior currently supports GaussianBic only",
            ));
        }
        let p = variables.len();
        if p == 0 {
            return Err(DiscoveryError::unsupported(
                "DBN posterior requires at least one variable",
            ));
        }
        if n_directed_edges(p) > 63 {
            return Err(DiscoveryError::unsupported(
                "DBN contemporaneous adjacency exceeds 63 directed edges",
            ));
        }
        let max_lag = self.max_lag;
        let n_lag_bits = (max_lag as usize).saturating_mul(p.saturating_mul(p));
        if n_lag_bits > 48 {
            return Err(DiscoveryError::unsupported(
                "DBN lag-edge space too large for this engine",
            ));
        }

        let (score_data, _) = build_lagged_score_data(data, variables, max_lag)?;
        let use_exact = !self.force_mcmc
            && p <= DBN_EXACT_MAX_VARS
            && max_lag <= DBN_EXACT_MAX_LAG
            && p <= EXACT_ENUM_MAX_NODES
            && n_lag_bits <= 16;

        if use_exact {
            self.run_exact(p, max_lag, &score_data, prior, variables, score_family)
        } else {
            self.run_mcmc(p, max_lag, &score_data, prior, variables, score_family, ctx)
        }
    }

    fn run_exact(
        &self,
        p: usize,
        max_lag: u32,
        score_data: &GraphScoreData,
        prior: &GraphPrior,
        variables: &[VariableId],
        score_family: GraphScoreFamily,
    ) -> Result<GraphPosterior, DiscoveryError> {
        let contemp = enumerate_unique_dags(p);
        let n_lag_bits = (max_lag as usize) * p * p;
        let lag_limit = 1u64 << n_lag_bits;
        let mut kept_masks = Vec::new();
        let mut kept_lag = Vec::new();
        let mut kept_log = Vec::new();
        let mut rejected = 0u64;
        let mut cache = LocalScoreCache::new(GraphScoreCacheKey {
            data_version: 1,
            family: score_family,
            var_fingerprint: score_data.n_vars as u64,
            penalty_fingerprint: score_data.n_rows as u64,
        });

        for &cmask in &contemp {
            if log_prior_mask(cmask, p, prior, variables).is_none() {
                rejected += 1;
                continue;
            }
            for lmask in 0..lag_limit {
                match score_dbn_template(cmask, lmask, p, max_lag, score_data, &mut cache, prior, variables)
                {
                    Some(lw) => {
                        kept_masks.push(cmask);
                        kept_lag.push(lmask);
                        kept_log.push(lw);
                    }
                    None => rejected += 1,
                }
            }
        }
        if kept_masks.is_empty() {
            return Err(DiscoveryError::unsupported(
                "no valid DBN templates under prior",
            ));
        }
        let weights = normalize_log_weights(&kept_log)?;
        let ess = kish_ess(&weights);
        let (edge, orient) = accumulate_marginals(p, &weights, &kept_masks);
        let lagged = accumulate_lagged_marginals(p, max_lag, &weights, &kept_lag);
        let diagnostics = analytic_graph_diagnostics(kept_masks.len(), ess);
        GraphPosterior::new(p, weights, kept_masks, edge, orient, ess, diagnostics, rejected)?
            .with_lagged_marginals(max_lag, lagged)?
            .with_lag_masks(kept_lag)
    }

    fn run_mcmc(
        &self,
        p: usize,
        max_lag: u32,
        score_data: &GraphScoreData,
        prior: &GraphPrior,
        variables: &[VariableId],
        score_family: GraphScoreFamily,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosterior, DiscoveryError> {
        use crate::graph_mcmc::{diagnostics_from_traces, GraphMcmcSchedule};
        use std::collections::HashMap;

        let schedule = GraphMcmcSchedule {
            n_chains: self.n_chains,
            n_warmup: self.n_warmup,
            n_draws: self.n_draws,
            thin: 1,
        };
        let (n_chains, n_warmup, n_draws, _) = schedule.as_usize();
        let n_lag_bits = (max_lag as usize) * p * p;
        let n_params = n_directed_edges(p) + n_lag_bits;
        let mut traces = vec![0.0f64; n_chains * n_draws * n_params];
        let mut samples: Vec<Vec<(u64, u64)>> = vec![Vec::new(); n_chains];
        let mut rejected = 0u64;

        for chain in 0..n_chains {
            let mut rng = ctx.rng.stream(3000 + chain as u64);
            let mut cache = LocalScoreCache::new(GraphScoreCacheKey {
                data_version: 1,
                family: score_family,
                var_fingerprint: score_data.n_vars as u64,
                penalty_fingerprint: score_data.n_rows as u64,
            });
            let mut cmask = 0u64;
            let mut lmask = 0u64;
            let mut cur = score_dbn_template(
                cmask, lmask, p, max_lag, score_data, &mut cache, prior, variables,
            )
            .unwrap_or(f64::NEG_INFINITY);
            let total = n_warmup + n_draws;
            let mut kept = 0usize;
            for step in 0..total {
                let (pc, pl, rej) = propose_dbn(cmask, lmask, p, max_lag, &mut rng);
                rejected += rej;
                let prop = score_dbn_template(
                    pc, pl, p, max_lag, score_data, &mut cache, prior, variables,
                );
                let accept = match prop {
                    Some(ps) if cur.is_finite() => {
                        let lr = ps - cur;
                        lr >= 0.0 || rng.next_f64() < lr.exp()
                    }
                    Some(ps) => {
                        cmask = pc;
                        lmask = pl;
                        cur = ps;
                        false
                    }
                    None => false,
                };
                if accept {
                    if let Some(ps) = prop {
                        cmask = pc;
                        lmask = pl;
                        cur = ps;
                    }
                }
                if step >= n_warmup && kept < n_draws {
                    let mut idx = 0;
                    for i in 0..p {
                        for j in 0..p {
                            if i == j {
                                continue;
                            }
                            traces[(chain * n_draws + kept) * n_params + idx] =
                                if has_edge(cmask, p, i, j) { 1.0 } else { 0.0 };
                            idx += 1;
                        }
                    }
                    for b in 0..n_lag_bits {
                        traces[(chain * n_draws + kept) * n_params + idx + b] =
                            if (lmask >> b) & 1 == 1 { 1.0 } else { 0.0 };
                    }
                    samples[chain].push((cmask, lmask));
                    kept += 1;
                }
            }
        }

        let diagnostics = diagnostics_from_traces(
            &schedule,
            &traces,
            n_params,
            true,
            "DBN MCMC diagnostics gate refused posterior",
        )?;

        let mut counts: HashMap<(u64, u64), u64> = HashMap::new();
        for chain in &samples {
            for &key in chain {
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        let total: f64 = counts.values().map(|&c| c as f64).sum();
        if !(total > 0.0) {
            return Err(DiscoveryError::unsupported("DBN MCMC produced no samples"));
        }
        let mut masks = Vec::new();
        let mut lags = Vec::new();
        let mut weights = Vec::new();
        for ((cm, lm), c) in counts {
            masks.push(cm);
            lags.push(lm);
            weights.push(c as f64 / total);
        }
        let ess = kish_ess(&weights);
        let (edge, orient) = accumulate_marginals(p, &weights, &masks);
        let lagged = accumulate_lagged_marginals(p, max_lag, &weights, &lags);
        GraphPosterior::new(p, weights, masks, edge, orient, ess, diagnostics, rejected)?
            .with_lagged_marginals(max_lag, lagged)?
            .with_lag_masks(lags)
    }
}

fn build_lagged_score_data(
    data: &TimeSeriesData,
    variables: &[VariableId],
    max_lag: u32,
) -> Result<(GraphScoreData, usize), DiscoveryError> {
    let p = variables.len();
    let t_len = data.row_count();
    let l = max_lag as usize;
    if t_len <= l + 1 {
        return Err(DiscoveryError::stats_msg(
            "insufficient time points for DBN lag window",
        ));
    }
    let n_rows = t_len - l;
    let n_cols = p * (l + 1);
    let mut flat = vec![0.0; n_cols * n_rows];
    for (vi, &vid) in variables.iter().enumerate() {
        let series = data.float64_values(vid).map_err(DiscoveryError::from)?;
        if series.len() != t_len {
            return Err(DiscoveryError::data_msg("series length mismatch"));
        }
        for lag in 0..=l {
            let col = lag * p + vi;
            for r in 0..n_rows {
                let t = r + l;
                flat[col * n_rows + r] = series[t - lag];
            }
        }
    }
    Ok((GraphScoreData::new(n_rows, n_cols, Arc::from(flat))?, n_rows))
}

fn lag_bit(p: usize, max_lag: u32, lag: u32, from: usize, to: usize) -> u32 {
    debug_assert!(lag >= 1 && lag <= max_lag);
    let block = (lag as usize - 1) * p * p;
    (block + from * p + to) as u32
}

fn has_lag_edge(lmask: u64, p: usize, max_lag: u32, lag: u32, from: usize, to: usize) -> bool {
    let b = lag_bit(p, max_lag, lag, from, to);
    (lmask >> b) & 1 == 1
}

/// Build a [`TemporalDag`] from contemporaneous + lag masks (DBN template atom).
///
/// Contemporaneous edges use lag 0 → lag 0; lag-`ℓ` edges use source lag `ℓ`
/// into contemporaneous targets.
///
/// # Errors
///
/// Invalid contemporaneous DAG or graph mutation failures.
pub fn temporal_dag_from_dbn_masks(
    cmask: u64,
    lmask: u64,
    n_vars: usize,
    max_lag: u32,
    variables: &[VariableId],
) -> Result<causal_graph::TemporalDag, DiscoveryError> {
    use causal_core::Lag;
    use causal_graph::unfold::ensure_lagged;
    use causal_graph::TemporalDag;

    if variables.len() != n_vars {
        return Err(DiscoveryError::data_msg(
            "temporal_dag_from_dbn_masks: variables length != n_vars",
        ));
    }
    if !mask_is_dag(cmask, n_vars) {
        return Err(DiscoveryError::data_msg(
            "temporal_dag_from_dbn_masks: contemporaneous mask is not a DAG",
        ));
    }
    let mut g = TemporalDag::empty();
    // Contemporaneous edges.
    for i in 0..n_vars {
        for j in 0..n_vars {
            if i != j && has_edge(cmask, n_vars, i, j) {
                let from = ensure_lagged(&mut g, variables[i], Lag::CONTEMPORANEOUS)?;
                let to = ensure_lagged(&mut g, variables[j], Lag::CONTEMPORANEOUS)?;
                g.insert_directed(from, to)?;
            }
        }
    }
    // Lagged edges: X_{t-lag} → Y_t.
    for lag in 1..=max_lag {
        for i in 0..n_vars {
            for j in 0..n_vars {
                if has_lag_edge(lmask, n_vars, max_lag, lag, i, j) {
                    let from = ensure_lagged(&mut g, variables[i], Lag::from_raw(lag))?;
                    let to = ensure_lagged(&mut g, variables[j], Lag::CONTEMPORANEOUS)?;
                    g.insert_directed(from, to)?;
                }
            }
        }
    }
    Ok(g)
}

fn score_dbn_template(
    cmask: u64,
    lmask: u64,
    p: usize,
    max_lag: u32,
    data: &GraphScoreData,
    cache: &mut LocalScoreCache,
    prior: &GraphPrior,
    variables: &[VariableId],
) -> Option<f64> {
    if !mask_is_dag(cmask, p) {
        return None;
    }
    let lp = log_prior_mask(cmask, p, prior, variables)?;
    // Lag edges: apply max-parents including lagged.
    let max_pa = prior.constraints.max_parents.unwrap_or(usize::MAX);
    let mut total = lp;
    for j in 0..p {
        let mut pa: Vec<u32> = parents_of(cmask, p, j);
        for lag in 1..=max_lag {
            for i in 0..p {
                if has_lag_edge(lmask, p, max_lag, lag, i, j) {
                    // Column index for (i, lag) in lagged design.
                    pa.push((lag as usize * p + i) as u32);
                }
            }
        }
        if pa.len() > max_pa {
            return None;
        }
        pa.sort_unstable();
        pa.dedup();
        let s = cache
            .local_score(data, j as u32, &Arc::from(pa))
            .ok()?;
        if !s.is_finite() {
            return None;
        }
        total += s;
    }
    // Weak Bernoulli prior on lag edges when edge_inclusion set.
    if let Some(pr) = prior.edge_inclusion {
        let lp_e = pr.ln();
        let lq = (1.0 - pr).ln();
        let n_lag_bits = (max_lag as usize) * p * p;
        for b in 0..n_lag_bits {
            total += if (lmask >> b) & 1 == 1 { lp_e } else { lq };
        }
    }
    Some(total)
}

fn accumulate_lagged_marginals(
    p: usize,
    max_lag: u32,
    weights: &[f64],
    lag_masks: &[u64],
) -> Vec<f64> {
    let cell = (max_lag as usize) * p * p;
    let mut out = vec![0.0; cell];
    for (&w, &lm) in weights.iter().zip(lag_masks.iter()) {
        for b in 0..cell {
            if (lm >> b) & 1 == 1 {
                out[b] += w;
            }
        }
    }
    out
}

fn propose_dbn(
    cmask: u64,
    lmask: u64,
    p: usize,
    max_lag: u32,
    rng: &mut causal_core::CausalRng,
) -> (u64, u64, u64) {
    let mut rejected = 0u64;
    if rng.next_f64() < 0.5 {
        // Flip a contemporaneous directed edge.
        let i = (rng.next_u64() as usize) % p;
        let j = (rng.next_u64() as usize) % p;
        if i == j {
            return (cmask, lmask, 0);
        }
        let on = has_edge(cmask, p, i, j);
        let prop = set_edge(cmask, p, i, j, !on);
        if mask_is_dag(prop, p) {
            (prop, lmask, 0)
        } else {
            rejected += 1;
            (cmask, lmask, rejected)
        }
    } else {
        // Flip a lag edge.
        let n_lag_bits = (max_lag as usize) * p * p;
        let b = (rng.next_u64() as usize) % n_lag_bits.max(1);
        let prop = lmask ^ (1u64 << b);
        (cmask, prop, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        ValidityBitmap,
    };

    fn lag1_series(n_obs: usize) -> (TimeSeriesData, Vec<VariableId>) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "y"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let vars: Vec<_> = (0..2).map(VariableId::from_raw).collect();
        let mut rng = causal_core::CausalRng::from_seed(99);
        let mut x = vec![0.0; n_obs];
        let mut y = vec![0.0; n_obs];
        x[0] = rng.next_f64() * 2.0 - 1.0;
        y[0] = rng.next_f64() * 2.0 - 1.0;
        for t in 1..n_obs {
            x[t] = 0.2 * (rng.next_f64() * 2.0 - 1.0);
            y[t] = 1.4 * x[t - 1] + 0.15 * (rng.next_f64() * 2.0 - 1.0);
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(vars[0], Arc::from(x), ValidityBitmap::all_valid(n_obs))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(vars[1], Arc::from(y), ValidityBitmap::all_valid(n_obs))
                    .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex {
                regularity: SamplingRegularity::Regular { interval_ns: 1 },
                length: n_obs,
            },
        )
        .unwrap();
        (data, vars)
    }

    #[test]
    fn dbn_recovers_lag1_edge() {
        let (data, vars) = lag1_series(200);
        let eng = DbnPosterior::new(1);
        let prior = GraphPrior::uniform().with_constraints(crate::constraints::DiscoveryConstraints {
            temporal: crate::constraints::TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::from_raw(1),
            },
            ..Default::default()
        });
        let post = eng
            .run(
                &data,
                &vars,
                &prior,
                GraphScoreFamily::GaussianBic,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert_eq!(post.max_lag, Some(1));
        let lagged = post.lagged_edge_marginals.as_ref().unwrap();
        // lag-1 bit for from=0 → to=1 is index 0*4 + 0*2 + 1 = 1
        assert!(lagged[1] > 0.4, "P(x_{{t-1}}→y_t)={}", lagged[1]);
    }
}
