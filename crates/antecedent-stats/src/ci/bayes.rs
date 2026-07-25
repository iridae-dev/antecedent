//! Bayesian conditional independence diagnostics for conjugate Gaussian models
//!
//! - [`BayesFactorCi`]: log Bayes factor for dependence vs independence under a
//!   Normal–Inv-Gamma conjugate model on full designs `y ~ [1,Z]` vs `y ~ [1,Z,x]`.
//! - [`PosteriorDependenceCi`]: posterior probability of dependence under equal
//!   prior odds (stable logistic of the log BF).
//! - [`PosteriorPredictiveCi`]: posterior-predictive p-value for absolute
//!   residual correlation under the independence null (full M0 refit each sim).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::similar_names, // xtx / xty, cxx / cxy conjugate notation
    clippy::many_single_char_names // Marsaglia gamma / conjugate scalars
)]

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_kernels::standard_normal;

use super::types::{
    CiBatchRequest, CiBatchResult, CiQuery, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, SignificanceMethod,
};
use crate::error::StatsError;
use crate::gram::{chol_log_det, chol_solve, cholesky_spd, form_xtx};
use crate::special::ln_gamma;

/// Default NIG shape/scale (matches `antecedent-prob` weakly informative `InvGamma`).
const ALPHA0: f64 = 1e-3;
const BETA0: f64 = 1e-3;
/// Diagonal prior precision on regression coefficients (1 / V0); V0 = 100 ⇒ scale 10.
const COEF_PRIOR_PREC: f64 = 0.01;

/// Conjugate NIG posterior after observing `(X, y)`.
#[derive(Clone, Debug)]
struct NigPosterior {
    /// Cholesky of `Λₙ = Λ₀ + X'X`.
    chol: Vec<f64>,
    /// Posterior mean `mₙ`.
    m_n: Vec<f64>,
    alpha_n: f64,
    beta_n: f64,
    p: usize,
}

/// Bayes-factor CI: statistic = log BF₁₀ (dependence vs independence).
///
/// `p_value` is the posterior probability of *independence* under equal prior
/// odds. Analytic significance only; block-shuffle is refused.
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
        let _ = (workspace, ctx);
        let mut results = Vec::with_capacity(nq);
        for q in request.queries {
            let log_bf = log_bf10_full(request, *q)?;
            let p_dep = logistic_from_log_bf(log_bf);
            let df = (n as f64) - 2.0 - (q.z_len as f64);
            results.push(CiResult {
                statistic: log_bf,
                p_value: (1.0 - p_dep).clamp(0.0, 1.0),
                df,
                ci: None,
            });
        }
        Ok(CiBatchResult { results })
    }
}

/// Posterior dependence probability under equal prior odds.
///
/// Statistic = `P(M₁ | data)`; `p_value` = independence posterior mass.
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
        let _ = (workspace, ctx);
        let mut results = Vec::with_capacity(nq);
        for q in request.queries {
            let log_bf = log_bf10_full(request, *q)?;
            let p_dep = logistic_from_log_bf(log_bf);
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
/// correction). Each replicate draws `(σ², β)` under `M₀`, regenerates `y`, and
/// refits the same residual-correlation pipeline.
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
        if workspace.shuffled.len() < n {
            workspace.shuffled.resize(n, 0.0);
        }

        let _ = ctx;
        let mut results = Vec::with_capacity(nq);
        for (i, q) in request.queries.iter().enumerate() {
            let prepared_cols = prepare_standardized(request, *q)?;
            // Observed statistic on the same standardized/refit pipeline as replicates.
            let abs_obs = abs_residual_corr_owned(
                &prepared_cols.x,
                &prepared_cols.y,
                &prepared_cols.z_cols,
                n,
            )?;
            let (_, nig0) =
                log_marginal_nig(&prepared_cols.x0, n, prepared_cols.p0, &prepared_cols.y)?;

            let mut rng = CausalRng::from_seed(self.seed ^ (i as u64).wrapping_mul(0x9E37_79B9));
            let mut extreme = 1u32; // +1 continuity
            let y_rep = &mut workspace.shuffled[..n];
            for _ in 0..self.n_sims {
                draw_m0_replicate(&nig0, &prepared_cols.x0, n, &mut rng, y_rep);
                // Full pipeline: re-standardize y_rep, keep x/Z standardization fixed.
                standardize_inplace(y_rep)?;
                let r_rep =
                    abs_residual_corr_owned(&prepared_cols.x, y_rep, &prepared_cols.z_cols, n)?;
                if r_rep >= abs_obs {
                    extreme += 1;
                }
            }
            let p = f64::from(extreme) / f64::from(self.n_sims + 1);
            let df = (n as f64) - 2.0 - (q.z_len as f64);
            results.push(CiResult { statistic: abs_obs, p_value: p.clamp(0.0, 1.0), df, ci: None });
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

/// Stable `P(M₁|data) = logistic(log BF₁₀)` under equal prior odds.
fn logistic_from_log_bf(log_bf: f64) -> f64 {
    if log_bf >= 0.0 {
        1.0 / (1.0 + (-log_bf).exp())
    } else {
        let e = log_bf.exp();
        e / (1.0 + e)
    }
}

struct PreparedQuery {
    x: Vec<f64>,
    y: Vec<f64>,
    z_cols: Vec<Vec<f64>>,
    /// Column-major `X₀ = [1, Z]`.
    x0: Vec<f64>,
    /// Column-major `X₁ = [1, Z, x]`.
    x1: Vec<f64>,
    p0: usize,
    p1: usize,
}

fn prepare_standardized(
    request: &CiBatchRequest<'_>,
    q: CiQuery,
) -> Result<PreparedQuery, StatsError> {
    let n = request.nrows()?;
    if q.x >= request.columns.len() || q.y >= request.columns.len() {
        return Err(StatsError::Shape { message: "CI query column out of range" });
    }
    let x = standardize_col(request.columns[q.x])?;
    let y = standardize_col(request.columns[q.y])?;
    let z_end = q.z_start.saturating_add(q.z_len);
    if z_end > request.z_flat.len() {
        return Err(StatsError::Shape { message: "z_flat shorter than query span" });
    }
    let mut z_cols = Vec::with_capacity(q.z_len);
    for &zi in &request.z_flat[q.z_start..z_end] {
        if zi >= request.columns.len() {
            return Err(StatsError::Shape { message: "conditioning column out of range" });
        }
        z_cols.push(standardize_col(request.columns[zi])?);
    }
    let p0 = 1 + q.z_len;
    let p1 = p0 + 1;
    if n <= p1 {
        return Err(StatsError::Shape {
            message: "need n > columns(X1) for full NIG Bayes factor",
        });
    }
    let x0 = design_intercept_z(n, &z_cols);
    let mut x1 = x0.clone();
    x1.extend_from_slice(&x);
    // Full-rank check via Cholesky of Λₙ for X1.
    let mut xtx = vec![0.0; p1 * p1];
    form_xtx(&x1, n, p1, &mut xtx);
    for i in 0..p1 {
        xtx[i * p1 + i] += COEF_PRIOR_PREC;
    }
    if cholesky_spd(&xtx, p1).is_none() {
        return Err(StatsError::Backend("singular design after standardization".into()));
    }
    Ok(PreparedQuery { x, y, z_cols, x0, x1, p0, p1 })
}

fn design_intercept_z(n: usize, z_cols: &[Vec<f64>]) -> Vec<f64> {
    let p0 = 1 + z_cols.len();
    let mut x0 = vec![0.0; n * p0];
    for r in 0..n {
        x0[r] = 1.0; // intercept column
    }
    for (j, z) in z_cols.iter().enumerate() {
        let base = (j + 1) * n;
        x0[base..base + n].copy_from_slice(z);
    }
    x0
}

fn standardize_col(col: &[f64]) -> Result<Vec<f64>, StatsError> {
    let n = col.len();
    if n < 2 {
        return Err(StatsError::Shape { message: "need n >= 2 to standardize" });
    }
    let nf = n as f64;
    let mut mean = 0.0;
    for &v in col {
        mean += v;
    }
    mean /= nf;
    let mut var = 0.0;
    for &v in col {
        let d = v - mean;
        var += d * d;
    }
    var /= nf; // sample variance with /n (matches unit-variance standardization)
    if var.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(StatsError::Shape { message: "zero-variance column in Bayesian CI" });
    }
    let s = var.sqrt();
    Ok(col.iter().map(|&v| (v - mean) / s).collect())
}

fn standardize_inplace(col: &mut [f64]) -> Result<(), StatsError> {
    let n = col.len();
    if n < 2 {
        return Err(StatsError::Shape { message: "need n >= 2 to standardize" });
    }
    let nf = n as f64;
    let mut mean = 0.0;
    for &v in col.iter() {
        mean += v;
    }
    mean /= nf;
    let mut var = 0.0;
    for &v in col.iter() {
        let d = v - mean;
        var += d * d;
    }
    var /= nf;
    if var.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(StatsError::Shape { message: "zero-variance column in Bayesian CI" });
    }
    let s = var.sqrt();
    for v in col.iter_mut() {
        *v = (*v - mean) / s;
    }
    Ok(())
}

fn log_bf10_full(request: &CiBatchRequest<'_>, q: CiQuery) -> Result<f64, StatsError> {
    let prep = prepare_standardized(request, q)?;
    let n = prep.y.len();
    let (log_m0, _) = log_marginal_nig(&prep.x0, n, prep.p0, &prep.y)?;
    let (log_m1, _) = log_marginal_nig(&prep.x1, n, prep.p1, &prep.y)?;
    let log_bf = log_m1 - log_m0;
    if !log_bf.is_finite() {
        return Err(StatsError::Backend("non-finite Bayes factor".into()));
    }
    Ok(log_bf)
}

/// Log marginal likelihood under shared NIG prior `Λ₀ = λ I`, `α₀`, `β₀`.
fn log_marginal_nig(
    x_cm: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
) -> Result<(f64, NigPosterior), StatsError> {
    if x_cm.len() < n * p || y.len() < n || p == 0 {
        return Err(StatsError::Shape { message: "NIG design shape mismatch" });
    }
    let mut lam_n = vec![0.0; p * p];
    form_xtx(x_cm, n, p, &mut lam_n);
    for i in 0..p {
        lam_n[i * p + i] += COEF_PRIOR_PREC;
    }
    let chol = cholesky_spd(&lam_n, p)
        .ok_or_else(|| StatsError::Backend("Cholesky failed for Λn".into()))?;
    let log_det_ln = chol_log_det(&chol, p);
    let log_det_l0 = (p as f64) * COEF_PRIOR_PREC.ln();

    let mut xty = vec![0.0; p];
    for j in 0..p {
        let col = &x_cm[j * n..(j + 1) * n];
        let mut acc = 0.0;
        for i in 0..n {
            acc += col[i] * y[i];
        }
        xty[j] = acc;
    }
    let m_n = chol_solve(&chol, p, &xty)
        .ok_or_else(|| StatsError::Backend("NIG posterior solve failed".into()))?;
    let mut m_lam_m = 0.0;
    for j in 0..p {
        m_lam_m += m_n[j] * xty[j];
    }
    let mut yty = 0.0;
    for i in 0..n {
        yty += y[i] * y[i];
    }
    let alpha_n = ALPHA0 + 0.5 * (n as f64);
    let beta_n = BETA0 + 0.5 * (yty - m_lam_m);
    if beta_n.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(StatsError::Backend("invalid NIG scale βn".into()));
    }
    let nf = n as f64;
    let log_m = -0.5 * nf * (2.0 * std::f64::consts::PI).ln()
        + 0.5 * (log_det_l0 - log_det_ln)
        + ALPHA0 * BETA0.ln()
        - alpha_n * beta_n.ln()
        + ln_gamma(alpha_n)
        - ln_gamma(ALPHA0);
    if !log_m.is_finite() {
        return Err(StatsError::Backend("non-finite NIG marginal".into()));
    }
    Ok((log_m, NigPosterior { chol, m_n, alpha_n, beta_n, p }))
}

fn draw_m0_replicate(
    nig: &NigPosterior,
    x0_cm: &[f64],
    n: usize,
    rng: &mut CausalRng,
    y_rep: &mut [f64],
) {
    let sigma2 = sample_inv_gamma(nig.alpha_n, nig.beta_n, rng);
    let sigma = sigma2.sqrt();
    // β = m + σ L^{-T} z, z ~ N(0, I)
    let mut z = vec![0.0; nig.p];
    for zi in &mut z {
        *zi = standard_normal(rng);
    }
    // Solve L^T v = z
    let mut v = vec![0.0; nig.p];
    for i in (0..nig.p).rev() {
        let mut acc = z[i];
        for j in (i + 1)..nig.p {
            acc -= nig.chol[j * nig.p + i] * v[j];
        }
        v[i] = acc / nig.chol[i * nig.p + i];
    }
    let mut beta = vec![0.0; nig.p];
    for j in 0..nig.p {
        beta[j] = nig.m_n[j] + sigma * v[j];
    }
    for r in 0..n {
        let mut mean = 0.0;
        for j in 0..nig.p {
            mean += x0_cm[j * n + r] * beta[j];
        }
        y_rep[r] = mean + sigma * standard_normal(rng);
    }
}

fn abs_residual_corr_owned(
    x: &[f64],
    y: &[f64],
    z_cols: &[Vec<f64>],
    n: usize,
) -> Result<f64, StatsError> {
    // OLS residualize x,y on [1,Z] then Pearson |r|.
    let p0 = 1 + z_cols.len();
    let x0 = design_intercept_z(n, z_cols);
    let rx = ols_residuals(&x0, n, p0, x)?;
    let ry = ols_residuals(&x0, n, p0, y)?;
    pearson_abs(&rx, &ry).ok_or(StatsError::Shape { message: "pearson failed in PPC" })
}

fn ols_residuals(x_cm: &[f64], n: usize, p: usize, y: &[f64]) -> Result<Vec<f64>, StatsError> {
    let mut xtx = vec![0.0; p * p];
    form_xtx(x_cm, n, p, &mut xtx);
    // Tiny ridge for numerical stability when Z empty (intercept-only is fine).
    for i in 0..p {
        xtx[i * p + i] += 1e-12;
    }
    let chol = cholesky_spd(&xtx, p)
        .ok_or_else(|| StatsError::Backend("OLS residual Cholesky failed".into()))?;
    let mut xty = vec![0.0; p];
    for j in 0..p {
        let col = &x_cm[j * n..(j + 1) * n];
        let mut acc = 0.0;
        for i in 0..n {
            acc += col[i] * y[i];
        }
        xty[j] = acc;
    }
    let beta = chol_solve(&chol, p, &xty)
        .ok_or_else(|| StatsError::Backend("OLS residual solve failed".into()))?;
    let mut resid = vec![0.0; n];
    for i in 0..n {
        let mut pred = 0.0;
        for j in 0..p {
            pred += x_cm[j * n + i] * beta[j];
        }
        resid[i] = y[i] - pred;
    }
    Ok(resid)
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

/// Independent empty-Z NIG reference via explicit 1×1 / 2×2 algebra (no Cholesky).
#[cfg(test)]
fn log_bf_empty_z_direct(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let xs = standardize_col(x).unwrap();
    let ys = standardize_col(y).unwrap();
    let nf = n as f64;
    let yty: f64 = ys.iter().map(|v| v * v).sum();
    // X0 = ones; X'y = 0 after centering; Λn = λ + n
    let lam0 = COEF_PRIOR_PREC;
    let lam_n0 = lam0 + nf;
    let alpha_n = ALPHA0 + 0.5 * nf;
    let beta_n0 = BETA0 + 0.5 * yty;
    let log_m0 = -0.5 * nf * (2.0 * std::f64::consts::PI).ln()
        + 0.5 * (lam0.ln() - lam_n0.ln())
        + ALPHA0 * BETA0.ln()
        - alpha_n * beta_n0.ln()
        + ln_gamma(alpha_n)
        - ln_gamma(ALPHA0);
    // X1 = [1, x]; orthogonal after centering ⇒ Λn = diag(λ+n, λ+||x||²)
    let xtx: f64 = xs.iter().map(|v| v * v).sum();
    let xty: f64 = xs.iter().zip(ys.iter()).map(|(a, b)| a * b).sum();
    let lam_xx = lam0 + xtx;
    let mn_x = xty / lam_xx;
    let m_lam_m = mn_x * lam_xx * mn_x; // intercept mean 0
    let beta_n1 = BETA0 + 0.5 * (yty - m_lam_m);
    let log_det_l0 = 2.0 * lam0.ln();
    let log_det_ln = lam_n0.ln() + lam_xx.ln();
    let log_m1 = -0.5 * nf * (2.0 * std::f64::consts::PI).ln()
        + 0.5 * (log_det_l0 - log_det_ln)
        + ALPHA0 * BETA0.ln()
        - alpha_n * beta_n1.ln()
        + ln_gamma(alpha_n)
        - ln_gamma(ALPHA0);
    log_m1 - log_m0
}

/// Reference simple-regression log BF (centered, no intercept) — retained for docs.
#[cfg(test)]
#[allow(dead_code)]
fn log_bf_simple_regression_reference(rx: &[f64], ry: &[f64]) -> f64 {
    let n = rx.len();
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
    let mut yty = 0.0;
    for i in 0..n {
        yty += yc[i] * yc[i];
    }
    let alpha_n = ALPHA0 + 0.5 * (n as f64);
    let beta_n0 = BETA0 + 0.5 * yty;
    let nf = n as f64;
    let log_m0 = -0.5 * nf * (2.0 * std::f64::consts::PI).ln() + ALPHA0 * BETA0.ln()
        - alpha_n * beta_n0.ln()
        + ln_gamma(alpha_n)
        - ln_gamma(ALPHA0);
    let mut xtx = 0.0;
    let mut xty = 0.0;
    for i in 0..n {
        xtx += xc[i] * xc[i];
        xty += xc[i] * yc[i];
    }
    let vn_inv = COEF_PRIOR_PREC + xtx;
    let mn = xty / vn_inv;
    let beta_n1 = BETA0 + 0.5 * (yty - mn * vn_inv * mn);
    let log_m1 = -0.5 * nf * (2.0 * std::f64::consts::PI).ln()
        + 0.5 * (COEF_PRIOR_PREC.ln() - vn_inv.ln())
        + ALPHA0 * BETA0.ln()
        - alpha_n * beta_n1.ln()
        + ln_gamma(alpha_n)
        - ln_gamma(ALPHA0);
    log_m1 - log_m0
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::ci::types::{CiPreparationPlan, ConfidenceMethod, SignificanceMethod};
    use crate::gram::chol_log_det;

    fn cols_indep(n: usize) -> (Vec<f64>, Vec<f64>) {
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
    fn logistic_from_log_bf_stays_finite_at_extremes() {
        let near_zero = logistic_from_log_bf(-1000.0);
        let half = logistic_from_log_bf(0.0);
        let near_one = logistic_from_log_bf(1000.0);
        assert!(near_zero.is_finite() && near_zero >= 0.0 && near_zero < 1e-15);
        assert!((half - 0.5).abs() < 1e-15);
        assert!(near_one.is_finite() && near_one <= 1.0 && near_one > 1.0 - 1e-15);
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

    #[test]
    fn empty_z_bf_matches_direct_nig() {
        let n = 100usize;
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
        let ctx = ExecutionContext::for_tests(6);
        let out = BayesFactorCi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        let xref = log_bf_empty_z_direct(&x, &y);
        let got = out.results[0].statistic;
        assert!((got - xref).abs() < 1e-8, "full={got} direct={xref}");
        // Also: intercept-free simple regression agrees in sign / magnitude order.
        let simple = log_bf_simple_regression_reference(&x, &y);
        assert!(got > 0.0 && simple > 0.0);
        assert!((got / simple - 1.0).abs() < 0.5, "full={got} simple={simple}");
    }

    #[test]
    fn irrelevant_z_does_not_inflate_bf() {
        let n = 120usize;
        let (x, y) = cols_indep(n);
        // Irrelevant Z columns independent of both.
        let z1: Vec<f64> = (0..n).map(|i| ((i as f64) * 2.414).sin()).collect();
        let z2: Vec<f64> = (0..n).map(|i| ((i as f64) * 3.7).cos()).collect();
        let cols: [&[f64]; 4] = [&x, &y, &z1, &z2];
        let q0 = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let qz = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 2 }];
        let z_flat = [2usize, 3];
        let ctx = ExecutionContext::for_tests(7);
        let mut ws = CiWorkspace::default();
        let bf0 = BayesFactorCi::new()
            .test_batch_adhoc(
                &CiBatchRequest {
                    columns: &cols,
                    queries: &q0,
                    z_flat: &[],
                    significance: SignificanceMethod::Analytic,
                    confidence: ConfidenceMethod::None,
                },
                &mut ws,
                &ctx,
            )
            .unwrap()
            .results[0]
            .statistic;
        let bfz = BayesFactorCi::new()
            .test_batch_adhoc(
                &CiBatchRequest {
                    columns: &cols,
                    queries: &qz,
                    z_flat: &z_flat,
                    significance: SignificanceMethod::Analytic,
                    confidence: ConfidenceMethod::None,
                },
                &mut ws,
                &ctx,
            )
            .unwrap()
            .results[0]
            .statistic;
        // Full NIG must not manufacture strong evidence from residual-rank inflation.
        assert!(bfz < 5.0, "bf with Z={bfz}");
        assert!((bfz - bf0).abs() < 3.0, "bf0={bf0} bfz={bfz}");
    }

    #[test]
    fn null_gaussian_ppc_calibrated() {
        let n = 80usize;
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
        let ctx = ExecutionContext::for_tests(8);
        let out = PosteriorPredictiveCi::new(199)
            .with_seed(11)
            .test_batch_adhoc(&req, &mut ws, &ctx)
            .unwrap();
        let p = out.results[0].p_value;
        assert!(p > 0.01, "null PPC p={p}");
    }

    #[test]
    fn log_bf_rises_with_signal_and_n() {
        let mut prev = f64::NEG_INFINITY;
        for &(n, noise) in &[(40usize, 2.0), (40, 0.5), (120, 0.5)] {
            let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let y: Vec<f64> = (0..n).map(|i| i as f64 + noise * ((i as f64) * 0.7).sin()).collect();
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
            let ctx = ExecutionContext::for_tests(9);
            let bf = BayesFactorCi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0]
                .statistic;
            assert!(bf > prev, "bf={bf} prev={prev} n={n} noise={noise}");
            prev = bf;
        }
    }

    #[test]
    fn chol_log_det_in_nig_path() {
        let a = [2.0, 0.5, 0.5, 3.0];
        let chol = cholesky_spd(&a, 2).unwrap();
        let ld = chol_log_det(&chol, 2);
        let det: f64 = 2.0 * 3.0 - 0.5 * 0.5;
        assert!((ld - det.ln()).abs() < 1e-12);
    }
}
