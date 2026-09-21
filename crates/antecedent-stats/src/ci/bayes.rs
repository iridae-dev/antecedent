//! Bayesian conditional independence diagnostics for Gaussian linear models
//!
//! - [`BayesFactorCi`]: log Bayes factor for dependence vs independence of `X` and `Y` given
//!   `Z`, under a Zellner g-prior (unit-information, `g = n`) on the partial regression
//!   coefficient and flat / scale-invariant priors on the nuisance parameters. The factor is a
//!   function of the partial correlation `r_{xy·z}`, `n` and `|Z|` only, so it is symmetric in
//!   `(x, y)`.
//! - [`PosteriorDependenceCi`]: posterior probability of dependence under equal prior odds
//!   (stable logistic of the log BF).
//! - [`PosteriorPredictiveCi`]: posterior-predictive p-value for absolute
//!   residual correlation under the independence null (full M0 refit each sim).
//!
//! The first two report `P(M₀ | data)` in [`CiResult::p_value`], which is a posterior
//! probability, not a frequentist p-value: they answer
//! [`ConditionalIndependenceTest::p_value_is_frequentist`] with `false`, and callers must not
//! apply a multiplicity correction to them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::needless_range_loop, clippy::too_many_arguments)]

use antecedent_core::{CausalRng, ExecutionContext, StreamDomain};
use antecedent_kernels::{sample_inv_gamma, standard_normal};

use super::block_shuffle::query_stream_salt;
use super::residualize::{ZDesign, residual_is_uninformative};
use super::types::{
    CiBatchRequest, CiBatchResult, CiQuery, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, SignificanceMethod, permutation_min_p,
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
/// `log BF₁₀ = −½ ln(1+g) − ½ (n−1−|Z|) ln(1 − g r² / (1+g))` with `g = n` and `r` the partial
/// correlation of `x` and `y` given `Z` (Zellner g-prior on the coefficient of `x` in
/// `y ~ [1, Z, x]`, flat priors on the intercept, `Z` coefficients and `σ²`). It depends on the
/// data only through `r`, so swapping `x` and `y` returns the same value. With `n` large the
/// implied decision boundary at `P(independence) = α` is `|t| ≈ sqrt(ln n + 2 ln BF)`
/// (Lindley's paradox): about `|t| > 3.5` at `n = 500`, `α = 0.05`.
///
/// `p_value` is the posterior probability of *independence* under equal prior odds, not a
/// frequentist p-value ([`ConditionalIndependenceTest::p_value_is_frequentist`] is `false`).
/// Analytic significance only; block-shuffle is refused.
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
            let log_bf = log_bf10_partial_corr(request, *q)?;
            let df = (n as f64) - 2.0 - (q.z_len as f64);
            results.push(CiResult {
                statistic: log_bf,
                p_value: logistic_from_log_bf(-log_bf),
                df,
                ci: None,
            });
        }
        Ok(CiBatchResult { results })
    }

    fn p_value_is_frequentist(&self) -> bool {
        false
    }
}

/// Posterior dependence probability under equal prior odds.
///
/// Statistic = `P(M₁ | data)` from the same symmetric g-prior Bayes factor as [`BayesFactorCi`];
/// `p_value` = independence posterior mass (not a frequentist p-value).
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
            let log_bf = log_bf10_partial_corr(request, *q)?;
            let p_dep = logistic_from_log_bf(log_bf);
            let df = (n as f64) - 2.0 - (q.z_len as f64);
            results.push(CiResult {
                statistic: p_dep.clamp(0.0, 1.0),
                p_value: logistic_from_log_bf(-log_bf),
                df,
                ci: None,
            });
        }
        Ok(CiBatchResult { results })
    }

    fn p_value_is_frequentist(&self) -> bool {
        false
    }
}

/// Posterior-predictive CI under the conjugate independence null.
///
/// Statistic = observed absolute residual correlation; `p_value` is the fraction
/// of null predictive replicates with `|r| ≥ |r_obs|` (plus one continuity
/// correction). Each replicate draws `(σ², β)` under `M₀`, regenerates `y`, and
/// refits the same residual-correlation pipeline.
///
/// Replicate draws come from the run's RNG (`ctx.rng`, stream keyed by the query), so the
/// analysis seed governs them and a query's p-value does not depend on its batch position.
/// [`Self::seed`] is an extra salt folded into that stream.
#[derive(Clone, Copy, Debug)]
pub struct PosteriorPredictiveCi {
    /// Null predictive replicates.
    pub n_sims: u32,
    /// Salt XOR'd into the run's stream identifier (`0` by default).
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

    /// Set the stream salt.
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

        let mut results = Vec::with_capacity(nq);
        for q in request.queries {
            let prepared_cols = prepare_standardized(request, *q)?;
            // Observed statistic on the same standardized/refit pipeline as replicates.
            let ry_obs = prepared_cols.design.residuals(&prepared_cols.y)?;
            let abs_obs = pearson_abs(&prepared_cols.rx, &ry_obs)
                .ok_or(StatsError::Shape { message: "pearson failed in PPC" })?;
            let (_, nig0) =
                log_marginal_nig(&prepared_cols.x0, n, prepared_cols.p0, &prepared_cols.y)?;

            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let mut rng = ctx.rng.stream_for(
                StreamDomain::StatsCi,
                0x99C1_u64 ^ self.seed ^ query_stream_salt(&[q.x], &[q.y], z),
            );
            let mut extreme = 1u32; // +1 continuity
            let y_rep = &mut workspace.shuffled[..n];
            for _ in 0..self.n_sims {
                draw_m0_replicate(&nig0, &prepared_cols.x0, n, &mut rng, y_rep);
                // Full pipeline: re-standardize y_rep, keep x/Z standardization fixed.
                standardize_inplace(y_rep)?;
                let ry_rep = prepared_cols.design.residuals(y_rep)?;
                let r_rep = pearson_abs(&prepared_cols.rx, &ry_rep)
                    .ok_or(StatsError::Shape { message: "pearson failed in PPC" })?;
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

    fn min_attainable_p(&self, _significance: SignificanceMethod) -> f64 {
        permutation_min_p(self.n_sims as usize)
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
    /// Standardised `x`, `y`.
    y: Vec<f64>,
    /// `x` residualised on `[1, Z]`; invariant across replicates.
    rx: Vec<f64>,
    /// The fitted `[1, Z]` design, shared by every residualisation of this query.
    design: ZDesign,
    /// Column-major `X₀ = [1, Z]` (standardised Z) for the NIG posterior of the null model.
    x0: Vec<f64>,
    p0: usize,
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
    if n <= p0 + 1 {
        return Err(StatsError::Shape {
            message: "need n > columns(X1) for full NIG Bayes factor",
        });
    }
    let x0 = design_intercept_z(n, &z_cols);
    let z_refs: Vec<&[f64]> = z_cols.iter().map(Vec::as_slice).collect();
    let z_idx: Vec<usize> = (0..z_refs.len()).collect();
    let design = ZDesign::fit(&z_refs, &z_idx, None, n)?;
    let rx = design.residuals(&x)?;
    // The prior ridge makes `Λₙ` positive definite for any design, so collinearity of `x`
    // with `Z` cannot show up there. Test it on the un-ridged residual instead: an `x` the
    // conditioning set determines leaves rounding residue, whose correlation with `y` is noise.
    if residual_is_uninformative(&x, &rx) {
        return Err(StatsError::Shape {
            message: "conditioned variable is collinear with the conditioning set",
        });
    }
    Ok(PreparedQuery { y, rx, design, x0, p0 })
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
    let mut out = col.to_vec();
    standardize_inplace(&mut out)?;
    Ok(out)
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
    var /= nf; // sample variance with /n (matches unit-variance standardization)
    if var.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(StatsError::Shape { message: "zero-variance column in Bayesian CI" });
    }
    let s = var.sqrt();
    for v in col.iter_mut() {
        *v = (*v - mean) / s;
    }
    Ok(())
}

/// Unit-information Zellner g-prior.
fn g_prior_g(n: usize) -> f64 {
    n as f64
}

/// `log BF₁₀` as a function of the partial correlation `r`, sample size and `|Z|`.
///
/// `−½ ln(1+g) − ½ d ln(1 − g r²/(1+g))`, `d = n − 1 − |Z|` the residual dimension of the
/// null model after the intercept and `Z` are projected out.
fn log_bf10_from_partial_corr(r: f64, n: usize, nz: usize) -> f64 {
    let g = g_prior_g(n);
    let d = n as f64 - 1.0 - nz as f64;
    let r2 = (r * r).min(1.0);
    -0.5 * (1.0 + g).ln() - 0.5 * d * (1.0 - g / (1.0 + g) * r2).ln()
}

fn log_bf10_partial_corr(request: &CiBatchRequest<'_>, q: CiQuery) -> Result<f64, StatsError> {
    let n = request.nrows()?;
    if q.x >= request.columns.len() || q.y >= request.columns.len() {
        return Err(StatsError::Shape { message: "CI query column out of range" });
    }
    let z_end = q.z_start.saturating_add(q.z_len);
    if z_end > request.z_flat.len() {
        return Err(StatsError::Shape { message: "z_flat shorter than query span" });
    }
    if n <= q.z_len + 2 {
        return Err(StatsError::Shape { message: "need n > |Z| + 2 for the g-prior Bayes factor" });
    }
    let z = &request.z_flat[q.z_start..z_end];
    let design = ZDesign::fit(request.columns, z, None, n)?;
    let rx = design.residuals(request.columns[q.x])?;
    let ry = design.residuals(request.columns[q.y])?;
    if residual_is_uninformative(request.columns[q.x], &rx)
        || residual_is_uninformative(request.columns[q.y], &ry)
    {
        return Err(StatsError::Shape {
            message: "a conditioned variable is constant or collinear with the conditioning set",
        });
    }
    let r = pearson_abs(&rx, &ry)
        .ok_or(StatsError::Shape { message: "degenerate partial correlation" })?;
    let log_bf = log_bf10_from_partial_corr(r, n, q.z_len);
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

/// `|Pearson r|`, or `None` when either series has no variance (a correlation of nothing is
/// undefined, not zero).
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
    if !(cxx > 0.0 && cyy > 0.0) {
        return None;
    }
    let denom = (cxx * cyy).sqrt();
    if !denom.is_finite() || denom == 0.0 {
        return None;
    }
    Some((cxy / denom).abs().min(1.0))
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
    fn review_pearson_abs_is_scale_invariant() {
        let x = [1.0, 1.0 + 1e-9, 1.0 + 2e-9];
        let y = [2.0, 2.1, 2.2];
        let r1 = pearson_abs(&x, &y).unwrap();
        let scale = 1e-4;
        let xs: Vec<f64> = x.iter().map(|v| v * scale).collect();
        let ys: Vec<f64> = y.iter().map(|v| v * scale).collect();
        let r2 = pearson_abs(&xs, &ys).unwrap();
        assert!((r1 - r2).abs() < 1e-12, "r1={r1} r2={r2}");
    }

    #[test]
    fn logistic_from_log_bf_stays_finite_at_extremes() {
        let near_zero = logistic_from_log_bf(-1000.0);
        let half = logistic_from_log_bf(0.0);
        let near_one = logistic_from_log_bf(1000.0);
        assert!(near_zero.is_finite() && (0.0..1e-15).contains(&near_zero));
        assert!((half - 0.5).abs() < 1e-15);
        assert!(near_one.is_finite() && ((1.0 - 1e-15)..=1.0).contains(&near_one));
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
        let expected = (-out.results[0].statistic).exp();
        assert!(expected > 0.0);
        assert!((out.results[0].p_value / expected - 1.0).abs() < 1e-12);
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

    /// Independent evaluation of the g-prior Bayes factor through sums of squares: with the
    /// intercept projected out, `BF₁₀ = (1+g)^{-1/2} · (RSS₀ / RSS_g)^{d/2}` with
    /// `RSS_g = RSS₀ − g/(1+g) · SSR_x`, `SSR_x = Sxy²/Sxx`, `RSS₀ = Syy`, `d = n − 1`, `g = n`.
    #[test]
    fn empty_z_bf_matches_sum_of_squares_form() {
        let n = 100usize;
        let (x, y) = cols_dep(n);
        let y: Vec<f64> =
            y.iter().enumerate().map(|(i, v)| v + 3.0 * ((i as f64) * 0.9).sin()).collect();
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
        let nf = n as f64;
        let (mx, my) = (x.iter().sum::<f64>() / nf, y.iter().sum::<f64>() / nf);
        let sxx: f64 = x.iter().map(|v| (v - mx) * (v - mx)).sum();
        let syy: f64 = y.iter().map(|v| (v - my) * (v - my)).sum();
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let g = nf;
        let ssr = sxy * sxy / sxx;
        let rss_g = syy - g / (1.0 + g) * ssr;
        let want = -0.5 * (1.0 + g).ln() + 0.5 * (nf - 1.0) * (syy / rss_g).ln();
        let got = out.results[0].statistic;
        assert!((got - want).abs() < 1e-9, "got={got} want={want}");
    }

    /// With one conditioner the partial correlation has the textbook closed form
    /// `(rxy − rxz·ryz) / sqrt((1−rxz²)(1−ryz²))`; the Bayes factor is that value pushed through
    /// the g-prior formula with `d = n − 2`.
    #[test]
    fn bf_with_conditioner_matches_closed_form_partial_correlation() {
        let n = 150usize;
        let z: Vec<f64> =
            (0..n).map(|i| ((i as f64) * 0.37).sin() + 0.2 * ((i * 7 % 5) as f64)).collect();
        let x: Vec<f64> = (0..n).map(|i| z[i] + 0.4 * ((i as f64) * 1.3).cos()).collect();
        let y: Vec<f64> =
            (0..n).map(|i| 0.8 * z[i] + 0.3 * x[i] + 0.5 * ((i as f64) * 2.1).sin()).collect();
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::None,
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(6);
        let got = BayesFactorCi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0]
            .statistic;
        let nf = n as f64;
        let corr = |a: &[f64], b: &[f64]| {
            let (ma, mb) = (a.iter().sum::<f64>() / nf, b.iter().sum::<f64>() / nf);
            let sab: f64 = a.iter().zip(b).map(|(p, q)| (p - ma) * (q - mb)).sum();
            let saa: f64 = a.iter().map(|p| (p - ma) * (p - ma)).sum();
            let sbb: f64 = b.iter().map(|q| (q - mb) * (q - mb)).sum();
            sab / (saa * sbb).sqrt()
        };
        let (rxy, rxz, ryz) = (corr(&x, &y), corr(&x, &z), corr(&y, &z));
        let r = (rxy - rxz * ryz) / ((1.0 - rxz * rxz) * (1.0 - ryz * ryz)).sqrt();
        let g = nf;
        let d = nf - 2.0;
        let want = -0.5 * (1.0 + g).ln() - 0.5 * d * (1.0 - g / (1.0 + g) * r * r).ln();
        assert!((got - want).abs() < 1e-9, "got={got} want={want} r={r}");
    }

    /// `x` is nearly collinear with `z`, `y` depends weakly on `x`'s residual: the earlier
    /// `y ~ [1,Z,x]` vs `y ~ [1,Z]` comparison priced the Occam factor with `x'M_Z x` and gave
    /// different answers for `(x, y)` and `(y, x)` (0.22 vs −2.15 in log BF). The g-prior
    /// factor depends on the partial correlation only, so the two orders agree exactly.
    #[test]
    fn bayes_factor_is_symmetric_in_x_and_y() {
        let n = 500usize;
        let z: Vec<f64> =
            (0..n).map(|i| ((i as f64) * 0.71).sin() + ((i * 13 % 17) as f64) * 0.05).collect();
        let e1: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.9 + 0.4).cos()).collect();
        let e2: Vec<f64> = (0..n).map(|i| ((i as f64) * 2.7 + 1.1).sin()).collect();
        let x: Vec<f64> = (0..n).map(|i| z[i] + 0.1 * e1[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| 0.5 * z[i] + 0.15 * (x[i] - z[i]) + 0.4 * e2[i]).collect();
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let z_flat = [2usize];
        let run = |qx: usize, qy: usize, ci: &dyn ConditionalIndependenceTest| {
            let queries = [CiQuery { x: qx, y: qy, z_start: 0, z_len: 1 }];
            let req = CiBatchRequest {
                columns: &cols,
                queries: &queries,
                z_flat: &z_flat,
                significance: SignificanceMethod::Analytic,
                confidence: ConfidenceMethod::None,
            };
            let mut ws = CiWorkspace::default();
            let ctx = ExecutionContext::for_tests(1);
            ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0]
        };
        for ci in [
            &BayesFactorCi::new() as &dyn ConditionalIndependenceTest,
            &PosteriorDependenceCi::new(),
        ] {
            let a = run(0, 1, ci);
            let b = run(1, 0, ci);
            assert!(
                (a.statistic - b.statistic).abs() < 1e-10,
                "{} vs {}",
                a.statistic,
                b.statistic
            );
            assert!((a.p_value - b.p_value).abs() < 1e-10, "{} vs {}", a.p_value, b.p_value);
        }
    }

    #[test]
    fn bayesian_p_values_are_flagged_as_posterior_probabilities() {
        assert!(!BayesFactorCi::new().p_value_is_frequentist());
        assert!(!PosteriorDependenceCi::new().p_value_is_frequentist());
        assert!(PosteriorPredictiveCi::new(9).p_value_is_frequentist());
    }

    /// The old full-rank check added the prior ridge before factoring, so it could never fail,
    /// and the PPC then correlated ~1e-12 rounding residue. `x` fully determined by `z` is
    /// refused by both paths.
    #[test]
    fn collinear_x_with_conditioner_is_refused() {
        let n = 60usize;
        let z: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin()).collect();
        let x: Vec<f64> = z.iter().map(|v| 3.0 * v + 2.0).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.1).cos() + z[i]).collect();
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::None,
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        assert!(BayesFactorCi::new().test_batch_adhoc(&req, &mut ws, &ctx).is_err());
        assert!(PosteriorPredictiveCi::new(19).test_batch_adhoc(&req, &mut ws, &ctx).is_err());
    }

    /// The replicate stream follows the run's RNG: different analysis seeds give different
    /// Monte Carlo draws, the same seed reproduces them, and a query's p-value does not depend
    /// on its position in the batch.
    #[test]
    fn ppc_uses_the_run_rng_and_is_batch_position_invariant() {
        let n = 60usize;
        let (x, y) = cols_indep(n);
        let w: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.29 + 0.7).sin()).collect();
        let cols: [&[f64]; 3] = [&x, &y, &w];
        let target = CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 };
        let other = CiQuery { x: 0, y: 2, z_start: 0, z_len: 0 };
        let run = |queries: &[CiQuery], seed: u64| {
            let req = CiBatchRequest {
                columns: &cols,
                queries,
                z_flat: &[],
                significance: SignificanceMethod::Analytic,
                confidence: ConfidenceMethod::None,
            };
            let mut ws = CiWorkspace::default();
            let ctx = ExecutionContext::for_tests(seed);
            PosteriorPredictiveCi::new(199).test_batch_adhoc(&req, &mut ws, &ctx).unwrap()
        };
        let alone = run(&[target], 5);
        let second = run(&[other, target], 5);
        assert_eq!(alone.results[0].p_value.to_bits(), second.results[1].p_value.to_bits());
        let again = run(&[target], 5);
        assert_eq!(alone.results[0].p_value.to_bits(), again.results[0].p_value.to_bits());
        let distinct: std::collections::BTreeSet<u64> =
            (1..=8).map(|seed| run(&[target], seed).results[0].p_value.to_bits()).collect();
        assert!(distinct.len() > 1, "PPC p-value ignores the analysis seed");
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
