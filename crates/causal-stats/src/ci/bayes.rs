//! Bayesian conditional independence diagnostics for conjugate Gaussian models
//!
//! - [`BayesFactorCi`]: log Bayes factor for dependence vs independence after
//!   residualizing on Z under a Normal–Inv-Gamma conjugate model.
//! - [`PosteriorDependenceCi`]: posterior probability of dependence under equal
//!   prior odds (BF / (1 + BF)).
//! - [`PosteriorPredictiveCi`]: posterior-predictive p-value for absolute
//!   residual correlation under the independence null.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use causal_core::{CausalRng, ExecutionContext};
use causal_kernels::{ParCorrMode, ParCorrQuery, partial_correlation_batch, standard_normal};

use super::parcorr::parcorr_mode;
use super::types::{
    CiBatchRequest, CiBatchResult, CiQuery, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, SignificanceMethod,
};
use crate::error::StatsError;
use crate::special::ln_gamma;

/// Default NIG shape/scale (matches `causal-prob` weakly informative InvGamma).
const ALPHA0: f64 = 1e-3;
const BETA0: f64 = 1e-3;
/// Diagonal prior precision on the residual slope (1 / V0); V0 = 100 ⇒ scale 10.
const COEF_PRIOR_PREC: f64 = 0.01;

/// Bayes-factor CI: statistic = log BF₁₀ (dependence vs independence).
///
/// `p_value` is the posterior probability of *independence* under equal prior
/// odds: `1 / (1 + BF₁₀)`. Analytic significance only; block-shuffle is refused.
#[derive(Clone, Copy, Debug, Default)]
pub struct BayesFactorCi;

impl BayesFactorCi {
    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl ConditionalIndependenceTest for BayesFactorCi {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        refuse_block_shuffle(request.significance)?;
        let n = request.nrows()?;
        let nq = request.queries.len();
        let mode = parcorr_mode(&ctx.kernel_policy);
        let mut results = Vec::with_capacity(nq);
        for q in request.queries {
            residualize_one(request, *q, workspace, mode)?;
            let (log_bf, _) = log_bf_dependence_vs_independence(
                &workspace.parcorr.rx[..n],
                &workspace.parcorr.ry[..n],
                n,
            )?;
            let bf = log_bf.exp();
            let p_indep = 1.0 / (1.0 + bf);
            let df = (n as f64) - 2.0 - (q.z_len as f64);
            results.push(CiResult {
                statistic: log_bf,
                p_value: p_indep.clamp(0.0, 1.0),
                df,
                ci: None,
            });
        }
        Ok(CiBatchResult { results })
    }
}

/// Posterior dependence probability under equal prior odds.
///
/// Statistic = `BF₁₀ / (1 + BF₁₀)`; `p_value` = independence posterior mass.
#[derive(Clone, Copy, Debug, Default)]
pub struct PosteriorDependenceCi;

impl PosteriorDependenceCi {
    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl ConditionalIndependenceTest for PosteriorDependenceCi {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        refuse_block_shuffle(request.significance)?;
        let n = request.nrows()?;
        let nq = request.queries.len();
        let mode = parcorr_mode(&ctx.kernel_policy);
        let mut results = Vec::with_capacity(nq);
        for q in request.queries {
            residualize_one(request, *q, workspace, mode)?;
            let (log_bf, _) = log_bf_dependence_vs_independence(
                &workspace.parcorr.rx[..n],
                &workspace.parcorr.ry[..n],
                n,
            )?;
            let bf = log_bf.exp();
            let p_dep = bf / (1.0 + bf);
            let df = (n as f64) - 2.0 - (q.z_len as f64);
            results.push(CiResult {
                statistic: p_dep.clamp(0.0, 1.0),
                p_value: (1.0 - p_dep).clamp(0.0, 1.0),
                df,
                ci: None,
            });
        }
        Ok(CiBatchResult { results })
    }
}

/// Posterior-predictive CI under the conjugate independence null.
///
/// Statistic = observed absolute residual correlation; `p_value` is the fraction
/// of null predictive replicates with `|r| ≥ |r_obs|` (plus one continuity
/// correction). Uses `seed` derived from the execution stream.
#[derive(Clone, Copy, Debug)]
pub struct PosteriorPredictiveCi {
    /// Null predictive replicates.
    pub n_sims: u32,
    /// Base RNG seed (XOR'd with query index).
    pub seed: u64,
}

impl Default for PosteriorPredictiveCi {
    fn default() -> Self {
        Self { n_sims: 199, seed: 0 }
    }
}

impl PosteriorPredictiveCi {
    /// Construct with replicate count.
    #[must_use]
    pub fn new(n_sims: u32) -> Self {
        Self { n_sims: n_sims.max(1), seed: 0 }
    }

    /// Set RNG seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl ConditionalIndependenceTest for PosteriorPredictiveCi {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        refuse_block_shuffle(request.significance)?;
        let n = request.nrows()?;
        if n < 3 {
            return Err(StatsError::Shape { message: "need n >= 3 for PPC CI" });
        }
        let nq = request.queries.len();
        let mode = parcorr_mode(&ctx.kernel_policy);
        if workspace.shuffled.len() < n {
            workspace.shuffled.resize(n, 0.0);
        }

        let mut results = Vec::with_capacity(nq);
        for (i, q) in request.queries.iter().enumerate() {
            residualize_one(request, *q, workspace, mode)?;
            let r_obs = workspace.stats[0]
                .ok_or(StatsError::Shape { message: "residual correlation failed" })?;
            let abs_obs = r_obs.abs();
            // Copy residuals before overwriting scratch.
            let rx: Vec<f64> = workspace.parcorr.rx[..n].to_vec();
            let ry: Vec<f64> = workspace.parcorr.ry[..n].to_vec();
            let (alpha_n, beta_n) = null_nig_posterior(&ry, n)?;

            let mut rng = CausalRng::from_seed(self.seed ^ (i as u64).wrapping_mul(0x9E37_79B9));
            let mut extreme = 1u32; // +1 continuity
            let y_rep = &mut workspace.shuffled[..n];
            for _ in 0..self.n_sims {
                let sigma2 = sample_inv_gamma(alpha_n, beta_n, &mut rng);
                let sigma = sigma2.sqrt();
                for r in 0..n {
                    y_rep[r] = sigma * standard_normal(&mut rng);
                }
                let r_rep = pearson_abs(&rx, y_rep).unwrap_or(0.0);
                if r_rep >= abs_obs {
                    extreme += 1;
                }
            }
            let p = f64::from(extreme) / f64::from(self.n_sims + 1);
            let df = (n as f64) - 2.0 - (q.z_len as f64);
            let _ = ctx;
            results.push(CiResult {
                statistic: abs_obs,
                p_value: p.clamp(0.0, 1.0),
                df,
                ci: None,
            });
        }
        Ok(CiBatchResult { results })
    }
}

fn refuse_block_shuffle(sig: SignificanceMethod) -> Result<(), StatsError> {
    match sig {
        SignificanceMethod::Analytic => Ok(()),
        SignificanceMethod::BlockShuffle { .. } => Err(StatsError::Backend(
            "Bayesian CI tests use conjugate analytic / predictive significance only".into(),
        )),
    }
}

fn residualize_one(
    request: &CiBatchRequest<'_>,
    q: CiQuery,
    workspace: &mut CiWorkspace,
    mode: ParCorrMode,
) -> Result<(), StatsError> {
    let n = request.nrows()?;
    if q.x >= request.columns.len() || q.y >= request.columns.len() {
        return Err(StatsError::Shape { message: "CI query column out of range" });
    }
    workspace.prepare_queries(1);
    workspace.parcorr.prepare(n, q.z_len.max(1));
    if q.z_len == 0 {
        // ParCorr short-circuits to Pearson without writing residuals.
        workspace.parcorr.rx[..n].copy_from_slice(request.columns[q.x]);
        workspace.parcorr.ry[..n].copy_from_slice(request.columns[q.y]);
        let r = causal_kernels::pearson(&workspace.parcorr.rx[..n], &workspace.parcorr.ry[..n])
            .unwrap_or(0.0);
        workspace.stats[0] = Some(r);
        return Ok(());
    }
    let z_end = q.z_start.saturating_add(q.z_len);
    if z_end > request.z_flat.len() {
        return Err(StatsError::Shape { message: "z_flat shorter than query span" });
    }
    for &zi in &request.z_flat[q.z_start..z_end] {
        if zi >= request.columns.len() {
            return Err(StatsError::Shape { message: "conditioning column out of range" });
        }
    }
    let pq = ParCorrQuery { x: q.x, y: q.y, z_start: q.z_start, z_len: q.z_len };
    partial_correlation_batch(
        request.columns,
        &[pq],
        request.z_flat,
        &mut workspace.stats[..1],
        &mut workspace.parcorr,
        mode,
    );
    if workspace.stats[0].is_none() {
        return Err(StatsError::Shape { message: "residual correlation failed" });
    }
    Ok(())
}

/// Log BF₁₀ for simple regression of `ry` on `rx` vs intercept-only (both already residualized).
fn log_bf_dependence_vs_independence(
    rx: &[f64],
    ry: &[f64],
    n: usize,
) -> Result<(f64, f64), StatsError> {
    if rx.len() < n || ry.len() < n || n < 3 {
        return Err(StatsError::Shape { message: "insufficient rows for Bayes factor" });
    }
    // Center so the slope model is comparable to Pearson (mean-zero residuals).
    let (mut xc, mut yc) = (vec![0.0; n], vec![0.0; n]);
    let mut mx = 0.0;
    let mut my = 0.0;
    for i in 0..n {
        mx += rx[i];
        my += ry[i];
    }
    mx /= n as f64;
    my /= n as f64;
    for i in 0..n {
        xc[i] = rx[i] - mx;
        yc[i] = ry[i] - my;
    }
    let log_m0 = log_marginal_null(&yc, n)?;
    let log_m1 = log_marginal_slope(&xc, &yc, n)?;
    let log_bf = log_m1 - log_m0;
    if !log_bf.is_finite() {
        return Err(StatsError::Backend("non-finite Bayes factor".into()));
    }
    Ok((log_bf, log_bf.exp()))
}

fn null_nig_posterior(y: &[f64], n: usize) -> Result<(f64, f64), StatsError> {
    let mut yty = 0.0;
    for i in 0..n {
        yty += y[i] * y[i];
    }
    let alpha_n = ALPHA0 + 0.5 * (n as f64);
    let beta_n = BETA0 + 0.5 * yty;
    if !(alpha_n > 0.0 && beta_n > 0.0) {
        return Err(StatsError::Backend("invalid null NIG posterior".into()));
    }
    Ok((alpha_n, beta_n))
}

fn log_marginal_null(y: &[f64], n: usize) -> Result<f64, StatsError> {
    let (alpha_n, beta_n) = null_nig_posterior(y, n)?;
    let nf = n as f64;
    Ok(-0.5 * nf * (2.0 * std::f64::consts::PI).ln()
        + ALPHA0 * BETA0.ln()
        - alpha_n * beta_n.ln()
        + ln_gamma(alpha_n)
        - ln_gamma(ALPHA0))
}

fn log_marginal_slope(x: &[f64], y: &[f64], n: usize) -> Result<f64, StatsError> {
    // Design: single column x (residualized; no intercept — matches ParCorr residualization).
    let mut xtx = 0.0;
    let mut xty = 0.0;
    let mut yty = 0.0;
    for i in 0..n {
        xtx += x[i] * x[i];
        xty += x[i] * y[i];
        yty += y[i] * y[i];
    }
    let lam0 = COEF_PRIOR_PREC;
    let vn_inv = lam0 + xtx;
    if !(vn_inv > 0.0) {
        return Err(StatsError::Backend("singular slope precision".into()));
    }
    let mn = xty / vn_inv;
    let alpha_n = ALPHA0 + 0.5 * (n as f64);
    let beta_n = BETA0 + 0.5 * (yty - mn * vn_inv * mn);
    if !(beta_n > 0.0) {
        return Err(StatsError::Backend("invalid slope NIG scale".into()));
    }
    let nf = n as f64;
    Ok(-0.5 * nf * (2.0 * std::f64::consts::PI).ln()
        + 0.5 * (lam0.ln() - vn_inv.ln())
        + ALPHA0 * BETA0.ln()
        - alpha_n * beta_n.ln()
        + ln_gamma(alpha_n)
        - ln_gamma(ALPHA0))
}

fn pearson_abs(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if y.len() != n || n < 2 {
        return None;
    }
    let nf = n as f64;
    let mut mx = 0.0;
    let mut my = 0.0;
    for i in 0..n {
        mx += x[i];
        my += y[i];
    }
    mx /= nf;
    my /= nf;
    let mut cxx = 0.0;
    let mut cyy = 0.0;
    let mut cxy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    let denom = (cxx * cyy).sqrt();
    if denom <= f64::EPSILON {
        return Some(0.0);
    }
    Some((cxy / denom).abs())
}

fn sample_inv_gamma(shape: f64, scale: f64, rng: &mut CausalRng) -> f64 {
    let g = sample_gamma(shape, scale, rng);
    1.0 / g.max(f64::MIN_POSITIVE)
}

fn sample_gamma(shape: f64, rate: f64, rng: &mut CausalRng) -> f64 {
    if shape < 1.0 {
        let u = rng.next_f64().max(f64::EPSILON);
        return sample_gamma(shape + 1.0, rate, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x;
        let mut v;
        loop {
            x = standard_normal(rng);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        v = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v / rate;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v / rate;
        }
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::ci::types::{CiPreparationPlan, ConfidenceMethod, SignificanceMethod};

    fn cols_indep(n: usize) -> (Vec<f64>, Vec<f64>) {
        // Deterministic near-independent continuous noise (no shared modular index).
        let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.618_033).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.732_050 + 0.3).cos()).collect();
        (x, y)
    }

    fn cols_dep(n: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 + 0.01).collect();
        (x, y)
    }

    #[test]
    fn bayes_factor_flags_dependence() {
        let n = 80usize;
        let (x, y) = cols_dep(n);
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::None,
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let out = BayesFactorCi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].statistic > 0.0, "log BF={}", out.results[0].statistic);
        assert!(out.results[0].p_value < 0.05);
    }

    #[test]
    fn bayes_factor_independent_not_extreme() {
        let n = 120usize;
        let (x, y) = cols_indep(n);
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::None,
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let out = BayesFactorCi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].p_value > 0.05, "p={}", out.results[0].p_value);
    }

    #[test]
    fn posterior_dependence_high_when_dependent() {
        let n = 60usize;
        let (x, y) = cols_dep(n);
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::None,
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let out = PosteriorDependenceCi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].statistic > 0.9);
    }

    #[test]
    fn ppc_ci_runs_and_bounds_p() {
        let n = 50usize;
        let (x, y) = cols_indep(n);
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::None,
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(4);
        let out = PosteriorPredictiveCi::new(99)
            .with_seed(7)
            .test_batch_adhoc(&req, &mut ws, &ctx)
            .unwrap();
        assert!((0.0..=1.0).contains(&out.results[0].p_value));
    }

    #[test]
    fn prepare_session_compatible() {
        let n = 40usize;
        let (x, y) = cols_dep(n);
        let cols: [&[f64]; 2] = [&x, &y];
        let ctx = ExecutionContext::for_tests(5);
        let plan = CiPreparationPlan {
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::None,
        };
        let prepared = BayesFactorCi::new().prepare(&cols, &plan, &ctx).unwrap();
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::None,
        };
        let mut ws = CiWorkspace::default();
        let out = BayesFactorCi::new().test_batch(&prepared, &req, &mut ws, &ctx).unwrap();
        assert_eq!(out.results.len(), 1);
    }
}
