//! Serial-dependence correction for Bayesian fits on time-ordered rows.
//!
//! The conjugate Normal–Inverse-Gamma likelihood (and the Laplace / HMC Gaussian
//! likelihoods) treat design rows as independent. On a lag-aligned temporal
//! design the outcome residual can be autocorrelated; when the regressor of
//! interest is persistent too, the score of the targeted linear combination
//! `c'β` is autocorrelated and the iid posterior is too narrow by the
//! long-run-variance ratio
//!
//! ```text
//! κ = LRV(w e) / Var(w e),    w = X (X'X)⁻¹ c
//! ```
//!
//! (`w_t e_t` is the influence of row `t` on `c'β̂`).
//! [`SerialDependence::LongRunTempering`] replaces the likelihood `L(θ)` by the
//! power likelihood `L(θ)^{1/κ̂}` (every row weighted `1/κ̂`, i.e. an effective
//! sample of `n/κ̂` rows). The result is a generalized (power) posterior, not
//! the posterior of a correctly specified dependent-data model: the prior keeps
//! its full weight, the posterior scale of every coefficient is widened by
//! `√κ̂` (calibrated for the targeted combination), and heteroskedasticity or a
//! misspecified mean are **not** corrected.
//!
//! With [`DependenceScope::Levels`] one fit targets several combinations (every grid
//! cell of a temporal response at one horizon), and with
//! [`DependenceScope::MediationPaths`] the path coefficients a linear mediation
//! mechanism contributes to; `κ̂` is then the largest of their ratios.
//!
//! `κ̂ = max(κ̂_HAC · f_b², κ̂_AR)`:
//!
//! - `κ̂_HAC` prewhitens the score, applies a Bartlett (Newey–West) kernel with
//!   the rule-of-thumb bandwidth `M = ⌊4 (n/100)^{2/9}⌋` to the prewhitened
//!   series and recolours it (Andrews & Monahan 1992). Two prewhitening filters
//!   are evaluated and the larger ratio kept: AR(1) with the Kendall
//!   small-sample bias correction of `ρ̂` (recolouring `1/(1 − ρ̂)²`), and, when
//!   BIC selects an order `q ≥ 2` (`q ≤ 4`, Yule–Walker), AR(q) (recolouring
//!   `1/(1 − Σφ̂)²`). `f_b` is the Kiefer–Vogelsang fixed-b factor of bandwidth
//!   `M + 1` ([`crate::temporal_block::fixed_b_scale`]), the same correction the
//!   Frequentist circular-block SEs carry. The kernel only mops up what the
//!   prewhitening filter leaves, so the estimate is as good as the filter's fit:
//!   it is **not** robust to arbitrary dependence (long memory, or
//!   autocorrelation beyond what an AR(4) captures, is under-corrected).
//! - `κ̂_AR = w'Γ̂w / (γ̂₀ w'w)` is the variance ratio of `c'β̂` given the realized
//!   design when the residual follows the fitted autoregression: AR(1) with the
//!   Kendall-corrected `ρ̂` of the OLS residuals, and the BIC-selected AR(q),
//!   `q ≥ 2`, when there is one (larger ratio kept). It is far less noisy and
//!   accounts for the realized regressor path, but assumes the residual's
//!   autocorrelation is independent of that path; the AR(q) term is therefore
//!   bounded by three times the fixed-b-scaled `κ̂_HAC`, so residual structure
//!   that the regressors themselves generate (and that never reaches the score)
//!   cannot inflate `κ̂` on its own.
//!
//! The larger of the two never narrows the robust estimate, and the AR(q)
//! terms only ever add to the AR(1) ones. In the 1.9 calibration (AR(1)
//! residuals, n = 60–400) the prewhitened ratio alone had log-SD 0.2–0.4 and
//! left nominal-90% intervals at 0.848–0.860 for n ≤ 160; the AR(1)
//! combination put them at 0.877–0.922. On AR(2)(0.3, 0.5) treatment and
//! residual it left them at 0.76–0.79 (n = 160–1000); the AR(q) terms raise
//! that to ≈ 0.86 at n = 160 and ≈ 0.89 at n ≥ 400 while leaving the AR(1) and
//! iid regimes unchanged. Short series with higher-order dependence stay
//! under-covered (≈ 0.81 at n = 60), because BIC rarely selects the order.
//! `κ̂` is floored at `1` (the correction never narrows the iid posterior) and
//! capped so at least `ncols + 2` effective rows remain. An exact fit (residual
//! sum of squares at most `1e-20` of the outcome's centred sum of squares) keeps
//! `κ̂ = 1`: the residuals are rounding error, whose apparent autocorrelation
//! would otherwise drive `κ̂` to the cap.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use std::sync::Arc;

use antecedent_stats::{CompiledDesign, DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::error::EstimationError;

/// Stable prefix of the inference-diagnostics note recording the tempering factor.
pub const DEPENDENCE_NOTE_PREFIX: &str = "serial_dependence.long_run_tempering";

/// Assumption id recorded on posteriors fitted under [`SerialDependence::LongRunTempering`].
pub const DEPENDENCE_ASSUMPTION_ID: &str = "bayes.temporal.long_run_tempering";

/// Largest absolute AR(1) prewhitening coefficient (keeps the recolouring finite).
const MAX_PREWHITEN_RHO: f64 = 0.97;

/// Fewest rows on which a long-run variance is estimated; shorter designs keep `κ = 1`.
const MIN_ROWS: usize = 8;

/// Residual sum of squares, relative to the outcome's centred sum of squares, at or
/// below which the fit is exact and `κ̂` stays at `1`.
const EXACT_FIT_RELATIVE_SS: f64 = 1e-20;

/// Largest autoregressive order the BIC search considers for the AR(q) terms.
const MAX_AR_ORDER: usize = 4;

/// Largest factor by which the AR(q) residual quadratic form may exceed the
/// fixed-b-scaled score HAC ratio. On the 1.9 AR(2) calibration designs the bound
/// never binds (coverage identical to the unbounded term); on a deterministic
/// period-4 treatment with an omitted lag it stops a residual quadratic form of
/// ≈145 against a score ratio of ≈0.2.
const AR_QUADRATIC_HAC_BOUND: f64 = 3.0;

/// Row-dependence model for a Bayesian likelihood.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum SerialDependence {
    /// Rows are exchangeable; the iid likelihood is the stated model.
    #[default]
    Iid,
    /// Rows are time-ordered; temper the likelihood by the long-run-variance ratio of
    /// the targeted linear combination.
    LongRunTempering(DependenceScope),
}

/// Which linear combination `c'β` of the design coefficients sets the tempering factor.
#[derive(Clone, Debug, PartialEq)]
pub enum DependenceScope {
    /// `c = e_T`: the treatment coefficient (single-equation g-computation, effect `β_T Δ`).
    Treatment,
    /// `c` = the gradient of a composed contrast with respect to this mechanism's
    /// coefficients (sequential g-computation), in design-column order.
    Direction(Arc<[f64]>),
    /// Several combinations sharing one fit, e.g. the g-computed level `w(a)'β` of every
    /// grid cell of a temporal response at one horizon (`w(a)` = the design-column
    /// averages with the treatment column at `a`). `κ̂` is the largest of their
    /// long-run-variance ratios, so every cell's interval carries at least its own
    /// correction.
    Levels(Arc<[Arc<[f64]>]>),
    /// The path combinations one linear mediation mechanism contributes to: the
    /// treatment→mediator slope `a`, or the direct `c'`, mediated `b` and total
    /// `c' + â b` gradients of the outcome mechanism (`â` the plug-in path
    /// coefficient). `κ̂` is the largest of their ratios, so every published
    /// decomposition quantity carries at least its own correction.
    MediationPaths(Arc<[Arc<[f64]>]>),
}

impl DependenceScope {
    const fn label(&self) -> &'static str {
        match self {
            Self::Treatment => "treatment",
            Self::Direction(_) => "contrast_gradient",
            Self::Levels(_) => "response_levels",
            Self::MediationPaths(_) => "mediation_paths",
        }
    }
}

/// Estimated tempering factor for one design.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemperingFactor {
    /// Applied factor `κ̂` (rows weighted `1/κ̂`), after the floor and cap.
    pub kappa: f64,
    /// Combined ratio before the floor / cap:
    /// `max(hac_ratio · fixed_b², ar_ratio)`.
    pub raw_ratio: f64,
    /// Prewhitened Bartlett long-run-variance ratio of the targeted score (the
    /// larger of the AR(1) and the BIC-selected AR(q) prewhitening).
    pub hac_ratio: f64,
    /// Kiefer–Vogelsang fixed-b SE factor of the Bartlett bandwidth
    /// ([`crate::temporal_block::fixed_b_scale`]); `hac_ratio` is scaled by its square.
    pub fixed_b: f64,
    /// Autoregressive-residual quadratic-form ratio conditional on the realized
    /// design (the larger of the AR(1) and the BIC-selected AR(q) residual model).
    pub ar_ratio: f64,
    /// BIC-selected autoregressive order of the driving score (`0..=4`); orders
    /// `≥ 2` add the AR(q)-prewhitened ratio.
    pub score_ar_order: usize,
    /// BIC-selected autoregressive order of the OLS residuals (`0..=4`); orders
    /// `≥ 2` add the AR(q) quadratic-form ratio.
    pub residual_ar_order: usize,
    /// Bartlett bandwidth on the prewhitened score.
    pub bandwidth: usize,
    /// Design rows.
    pub nrows: usize,
    /// Whether the cap bound `κ̂` (the correction is then incomplete).
    pub capped: bool,
    /// Whether the long-run-variance ratio could not be estimated (`n` below
    /// `max(8, p+2)`); `κ = 1` then leaves the iid posterior in force.
    pub inestimable: bool,
    /// Label of the scope that chose the combination.
    pub scope: &'static str,
}

impl TemperingFactor {
    /// Effective number of rows `n / κ̂`.
    #[must_use]
    pub fn effective_rows(&self) -> f64 {
        self.nrows as f64 / self.kappa
    }

    /// Machine-readable inference-diagnostics note (see [`DEPENDENCE_NOTE_PREFIX`]).
    #[must_use]
    pub fn note(&self) -> Arc<str> {
        Arc::from(format!(
            "{DEPENDENCE_NOTE_PREFIX} kappa={:.6} raw_ratio={:.6} hac_ratio={:.6} fixed_b={:.6} \
             ar_ratio={:.6} score_ar_order={} residual_ar_order={} n={} n_eff={:.3} \
             bandwidth={} scope={} capped={} inestimable={}",
            self.kappa,
            self.raw_ratio,
            self.hac_ratio,
            self.fixed_b,
            self.ar_ratio,
            self.score_ar_order,
            self.residual_ar_order,
            self.nrows,
            self.effective_rows(),
            self.bandwidth,
            self.scope,
            self.capped,
            self.inestimable
        ))
    }

    /// Human-readable assumption description.
    #[must_use]
    pub fn description(&self) -> Arc<str> {
        Arc::from(format!(
            "generalized (power) posterior with a serial-dependence correction: the Gaussian \
             likelihood on {} time-ordered rows is tempered by 1/kappa with kappa = {:.4}, the \
             larger of the autoregressive-prewhitened Newey-West long-run-variance ratio of the \
             {} score (bandwidth {}, BIC score order {}) scaled by the squared fixed-b factor \
             {:.4}, and the autoregressive-residual variance ratio of the same combination given \
             the design (BIC residual order {}); floored at 1{}; effective rows {:.1}; the prior \
             keeps full weight; calibrated for autoregressive dependence up to order 4, not for \
             long memory; heteroskedasticity and mean misspecification are not corrected",
            self.nrows,
            self.kappa,
            self.scope,
            self.bandwidth,
            self.score_ar_order,
            self.fixed_b,
            self.residual_ar_order,
            if self.inestimable {
                "; long-run-variance ratio inestimable on this short design — iid posterior \
                 retained and is likely too narrow"
            } else if self.capped {
                ", capped; the correction is then incomplete and the posterior is still too narrow"
            } else {
                ""
            },
            self.effective_rows()
        ))
    }
}

fn note_flag(notes: &[Arc<str>], key: &str) -> bool {
    notes.iter().any(|note| {
        note.strip_prefix(DEPENDENCE_NOTE_PREFIX)
            .and_then(|rest| rest.split_whitespace().find_map(|kv| kv.strip_prefix(key)))
            == Some("true")
    })
}

/// Parse the tempering factor recorded on posterior inference notes.
#[must_use]
pub fn tempering_kappa_from_notes(notes: &[Arc<str>]) -> Option<f64> {
    notes.iter().find_map(|note| {
        let rest = note.strip_prefix(DEPENDENCE_NOTE_PREFIX)?;
        let value = rest.split_whitespace().find_map(|kv| kv.strip_prefix("kappa="))?;
        value.parse::<f64>().ok()
    })
}

/// Whether any tempering note recorded that the `n/(p+2)` cap bound `κ̂`.
#[must_use]
pub fn tempering_capped_from_notes(notes: &[Arc<str>]) -> bool {
    note_flag(notes, "capped=")
}

/// Whether any tempering note recorded that the LRV ratio could not be estimated.
#[must_use]
pub fn tempering_inestimable_from_notes(notes: &[Arc<str>]) -> bool {
    note_flag(notes, "inestimable=")
}

/// Estimate the long-run-variance tempering factor for `design` over `scope`.
///
/// Rows must be in time order.
///
/// # Errors
///
/// Rank-deficient design (the OLS residuals are undefined), a missing treatment
/// column for [`DependenceScope::Treatment`], or a direction of the wrong length.
#[allow(clippy::too_many_lines)]
pub fn long_run_tempering_factor(
    design: &CompiledDesign,
    scope: &DependenceScope,
) -> Result<TemperingFactor, EstimationError> {
    let n = design.nrows;
    let p = design.ncols;
    let bandwidth = newey_west_bandwidth(n);
    let floor = TemperingFactor {
        kappa: 1.0,
        raw_ratio: 1.0,
        hac_ratio: 1.0,
        fixed_b: 1.0,
        ar_ratio: 1.0,
        score_ar_order: 0,
        residual_ar_order: 0,
        bandwidth,
        nrows: n,
        capped: false,
        inestimable: false,
        scope: scope.label(),
    };
    let check_width = |direction: &[f64]| {
        if direction.len() == p {
            Ok(direction.to_vec())
        } else {
            Err(EstimationError::stats_msg(format!(
                "long-run tempering direction has {} entries for {p} design columns",
                direction.len()
            )))
        }
    };
    let directions: Vec<Vec<f64>> = match scope {
        DependenceScope::Treatment => {
            let t = design.treatment_column().ok_or_else(|| {
                EstimationError::stats_msg("long-run tempering needs a treatment column")
            })?;
            let mut c = vec![0.0; p];
            c[t] = 1.0;
            vec![c]
        }
        DependenceScope::Direction(direction) => vec![check_width(direction)?],
        DependenceScope::Levels(directions) | DependenceScope::MediationPaths(directions) => {
            directions.iter().map(|direction| check_width(direction)).collect::<Result<_, _>>()?
        }
    };
    // A zero or non-finite combination carries no score; it neither tempers nor refuses.
    let directions: Vec<Vec<f64>> = directions
        .into_iter()
        .filter(|c| c.iter().all(|v| v.is_finite()) && c.iter().any(|v| *v != 0.0))
        .collect();
    if n < MIN_ROWS.max(p + 2) {
        return Ok(TemperingFactor { inestimable: true, ..floor });
    }
    if directions.is_empty() {
        return Ok(floor);
    }
    let x = &design.matrix[..n * p];
    let residuals = FaerBackend
        .least_squares(x, n, p, &design.outcome, &mut LeastSquaresWorkspace::default())?
        .residuals;
    // An exact fit leaves only rounding error, whose autocorrelation is an artefact of the
    // arithmetic: there is no residual dependence to correct.
    let outcome_mean = design.outcome.iter().sum::<f64>() / n as f64;
    let centred_ss: f64 = design.outcome.iter().map(|y| (y - outcome_mean).powi(2)).sum();
    let residual_ss: f64 = residuals.iter().map(|e| e * e).sum();
    if residual_ss <= EXACT_FIT_RELATIVE_SS * centred_ss {
        return Ok(floor);
    }
    // w = X (X'X)⁻¹ c: row influence weights of c'β̂.
    let mut xtx = vec![0.0; p * p];
    for a in 0..p {
        for b in a..p {
            let dot: f64 =
                x[a * n..(a + 1) * n].iter().zip(&x[b * n..(b + 1) * n]).map(|(u, v)| u * v).sum();
            xtx[a * p + b] = dot;
            xtx[b * p + a] = dot;
        }
    }
    let fixed_b = crate::temporal_block::fixed_b_scale(bandwidth + 1, n);
    let residual_ar = bic_autoregression(&residuals);
    // The direction with the largest combined ratio drives κ̂; its components are
    // reported alongside it.
    let (mut raw_ratio, mut hac_ratio, mut ar_ratio) = (f64::NAN, f64::NAN, f64::NAN);
    let mut score_ar_order = 0;
    for c in &directions {
        let v = crate::util::solve_spd(&xtx, c, p)
            .ok_or_else(|| EstimationError::stats_msg("long-run tempering: singular X'X"))?;
        // w_t = x_t'(X'X)⁻¹c: the weight of row t in c'β̂; w_t·ê_t is its influence.
        let weights: Vec<f64> =
            (0..n).map(|t| (0..p).map(|k| x[k * n + t] * v[k]).sum::<f64>()).collect();
        let influence: Vec<f64> = weights.iter().zip(&residuals).map(|(w, e)| w * e).collect();
        let score_ar = bic_autoregression(&influence);
        let mut hac = long_run_variance_ratio(&influence, bandwidth);
        if score_ar.phi.len() >= 2 {
            hac = hac.max(ar_prewhitened_variance_ratio(&influence, &score_ar.phi, bandwidth));
        }
        let mut ar = ar1_quadratic_ratio(&weights, &residuals);
        if residual_ar.phi.len() >= 2 {
            // The quadratic form assumes the residual's autocorrelation is independent of
            // the regressor path. Residual structure that is a function of the regressors
            // (an omitted seasonal lag of a seasonal treatment) does not reach the score,
            // so the AR(q) term may exceed the score-based ratio by at most a bounded factor.
            let bound = AR_QUADRATIC_HAC_BOUND * hac * fixed_b * fixed_b;
            ar = ar.max(ar_quadratic_ratio(&weights, &residual_ar).min(bound));
        }
        if !hac.is_finite() || !ar.is_finite() {
            continue;
        }
        let combined = (hac * fixed_b * fixed_b).max(ar);
        if raw_ratio.is_nan() || combined > raw_ratio {
            raw_ratio = combined;
            hac_ratio = hac;
            ar_ratio = ar;
            score_ar_order = score_ar.phi.len();
        }
    }
    if !raw_ratio.is_finite() {
        return Ok(floor);
    }
    let cap = (n as f64 / (p + 2) as f64).max(1.0);
    let kappa = raw_ratio.clamp(1.0, cap);
    Ok(TemperingFactor {
        kappa,
        raw_ratio,
        hac_ratio,
        fixed_b,
        ar_ratio,
        score_ar_order,
        residual_ar_order: residual_ar.phi.len(),
        capped: raw_ratio > cap,
        ..floor
    })
}

/// Yule–Walker autoregression of a series, at a BIC-selected order.
#[derive(Clone, Debug, Default)]
struct Autoregression {
    /// Coefficients `φ₁ … φ_q` (empty for `q = 0`).
    phi: Vec<f64>,
    /// Uncentred autocorrelations `r_0 = 1, r_1 … r_q` of the series.
    autocorrelation: Vec<f64>,
}

/// Yule–Walker AR(q) fit of `s` from its uncentred autocovariances (the same
/// convention as [`kendall_rho`]; OLS residuals and scores have mean zero), with
/// `q ≤ MAX_AR_ORDER` minimizing `n ln σ̂²_q + q ln n` (Levinson–Durbin). The
/// Toeplitz autocovariance is positive definite, so every fitted model is
/// stationary.
fn bic_autoregression(s: &[f64]) -> Autoregression {
    let n = s.len();
    let max_order = MAX_AR_ORDER.min(n.saturating_sub(2));
    let gamma: Vec<f64> = (0..=max_order)
        .map(|k| s[k..].iter().zip(s).map(|(a, b)| a * b).sum::<f64>() / n as f64)
        .collect();
    if gamma[0] <= 0.0 || !gamma[0].is_finite() {
        return Autoregression::default();
    }
    let nf = n as f64;
    let mut phi: Vec<f64> = Vec::new();
    let mut variance = gamma[0];
    let (mut best_bic, mut best) = (nf * variance.ln(), Vec::new());
    for order in 1..=max_order {
        let acc = gamma[order]
            - phi.iter().enumerate().map(|(j, f)| f * gamma[order - 1 - j]).sum::<f64>();
        let reflection = acc / variance;
        if !reflection.is_finite() || reflection.abs() >= 1.0 {
            break;
        }
        let mut next: Vec<f64> =
            (0..order - 1).map(|j| phi[j] - reflection * phi[order - 2 - j]).collect();
        next.push(reflection);
        phi = next;
        variance *= 1.0 - reflection * reflection;
        if variance <= 0.0 || variance.is_nan() {
            break;
        }
        let bic = nf * variance.ln() + order as f64 * nf.ln();
        if bic < best_bic {
            best_bic = bic;
            best.clone_from(&phi);
        }
    }
    let autocorrelation = gamma[..=best.len()].iter().map(|g| g / gamma[0]).collect();
    Autoregression { phi: best, autocorrelation }
}

/// AR(q)-prewhitened Bartlett long-run variance of `s` divided by its variance:
/// `u_t = s_t − Σ φ_j s_{t−j}`, recoloured by `1/(1 − Σφ)²` (floored like the
/// AR(1) filter's `1 − ρ̂`).
fn ar_prewhitened_variance_ratio(s: &[f64], phi: &[f64], bandwidth: usize) -> f64 {
    let n = s.len();
    let q = phi.len();
    let gamma0 = s.iter().map(|v| v * v).sum::<f64>() / n as f64;
    if gamma0 <= 0.0 || !gamma0.is_finite() || n <= q + 1 {
        return 1.0;
    }
    let u: Vec<f64> = (q..n)
        .map(|t| s[t] - phi.iter().enumerate().map(|(j, f)| f * s[t - 1 - j]).sum::<f64>())
        .collect();
    let recolour = (1.0 - phi.iter().sum::<f64>()).max(1.0 - MAX_PREWHITEN_RHO);
    bartlett_long_run_variance(&u, bandwidth) / recolour.powi(2) / gamma0
}

/// Variance ratio of `Σ w_t e_t` when the residual follows the fitted AR(q):
/// `1 + 2 Σ_k ρ_k Σ_t w_t w_{t−k} / w'w`, with the model autocorrelation `ρ_k`
/// (equal to the sample one for `k ≤ q`, the AR recursion beyond), truncated
/// once it is negligible.
fn ar_quadratic_ratio(weights: &[f64], model: &Autoregression) -> f64 {
    let n = weights.len();
    let q = model.phi.len();
    let norm: f64 = weights.iter().map(|w| w * w).sum();
    if norm <= 0.0 || q == 0 {
        return 1.0;
    }
    let mut rho = model.autocorrelation.clone();
    let mut cross = 0.0;
    for k in 1..n {
        if k > q {
            let next: f64 = model.phi.iter().enumerate().map(|(j, f)| f * rho[k - 1 - j]).sum();
            rho.push(next);
            if rho[k + 1 - q..=k].iter().all(|r| r.abs() < 1e-12) {
                break;
            }
        }
        cross += rho[k] * weights[k..].iter().zip(weights).map(|(a, b)| a * b).sum::<f64>();
    }
    1.0 + 2.0 * cross / norm
}

/// Bias-corrected lag-1 autocorrelation of `s` (Kendall 1954:
/// `E[ρ̂] ≈ ρ − (1 + 3ρ)/n`), clamped to `±MAX_PREWHITEN_RHO`.
fn kendall_rho(s: &[f64]) -> f64 {
    let (num, den) =
        s.windows(2).fold((0.0, 0.0), |(num, den), w| (num + w[1] * w[0], den + w[0] * w[0]));
    let rho_hat = if den > 0.0 { num / den } else { 0.0 };
    (rho_hat + (1.0 + 3.0 * rho_hat) / s.len() as f64).clamp(-MAX_PREWHITEN_RHO, MAX_PREWHITEN_RHO)
}

/// Variance ratio of `Σ w_t e_t` under AR(1) residuals, conditional on the
/// realized weights: `w'Γw / (γ₀ w'w) = 1 + 2 Σ_{s<t} ρ̂^{t−s} w_s w_t / w'w`,
/// with `ρ̂` the [`kendall_rho`] of the OLS residuals.
fn ar1_quadratic_ratio(weights: &[f64], residuals: &[f64]) -> f64 {
    let rho = kendall_rho(residuals);
    let (mut carry, mut cross, mut norm) = (0.0, 0.0, 0.0);
    for (t, &w) in weights.iter().enumerate() {
        if t > 0 {
            carry = rho * (carry + weights[t - 1]);
        }
        cross += w * carry;
        norm += w * w;
    }
    if norm > 0.0 { 1.0 + 2.0 * cross / norm } else { 1.0 }
}

/// Newey–West rule-of-thumb bandwidth `⌊4 (n/100)^{2/9}⌋`.
fn newey_west_bandwidth(n: usize) -> usize {
    (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor().max(0.0) as usize
}

/// Bartlett (Newey–West) long-run variance of `u` (uncentred, divided by its length).
fn bartlett_long_run_variance(u: &[f64], bandwidth: usize) -> f64 {
    let m = u.len();
    let lags = bandwidth.min(m.saturating_sub(1));
    let autocov = |lag: usize| u[lag..].iter().zip(&u[..m - lag]).map(|(a, b)| a * b).sum::<f64>();
    let mut lrv = autocov(0);
    for lag in 1..=lags {
        let weight = 1.0 - lag as f64 / (lags + 1) as f64;
        lrv += 2.0 * weight * autocov(lag);
    }
    (lrv / m as f64).max(0.0)
}

/// AR(1)-prewhitened Bartlett long-run variance of `s` divided by its variance.
fn long_run_variance_ratio(s: &[f64], bandwidth: usize) -> f64 {
    let n = s.len();
    let gamma0 = s.iter().map(|v| v * v).sum::<f64>() / n as f64;
    if gamma0 <= 0.0 || !gamma0.is_finite() {
        return 1.0;
    }
    let rho = kendall_rho(s);
    let u: Vec<f64> = s.windows(2).map(|w| w[1] - rho * w[0]).collect();
    let lrv = bartlett_long_run_variance(&u, bandwidth) / (1.0 - rho).powi(2);
    lrv / gamma0
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{CausalRng, VariableId};
    use antecedent_kernels::standard_normal;

    fn ar1(n: usize, rho: f64, rng: &mut CausalRng) -> Vec<f64> {
        let mut prev = standard_normal(rng);
        (0..n)
            .map(|_| {
                prev = rho * prev + (1.0 - rho * rho).sqrt() * standard_normal(rng);
                prev
            })
            .collect()
    }

    fn design(n: usize, rho: f64, seed: u64) -> CompiledDesign {
        let mut rng = CausalRng::from_seed(seed);
        let x = ar1(n, rho, &mut rng);
        let e = ar1(n, rho, &mut rng);
        let z: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
        let y: Vec<f64> =
            x.iter().zip(&e).zip(&z).map(|((x, e), z)| 0.8 * x + 0.3 * z + e).collect();
        CompiledDesign::linear_adjustment(&x, &[(VariableId::from_raw(2), z.as_slice())], &y, &[])
            .unwrap()
    }

    #[test]
    fn iid_rows_centre_on_one() {
        let (mut hac, mut ar) = (Vec::new(), Vec::new());
        for seed in 0..40 {
            let f = long_run_tempering_factor(&design(400, 0.0, seed), &DependenceScope::Treatment)
                .unwrap();
            assert!(f.kappa >= 1.0);
            assert!(f.fixed_b > 1.0 && f.fixed_b < 1.05, "fixed-b at n = 400: {}", f.fixed_b);
            let combined = (f.hac_ratio * f.fixed_b * f.fixed_b).max(f.ar_ratio);
            assert!((f.raw_ratio - combined).abs() < 1e-12);
            hac.push(f.hac_ratio);
            ar.push(f.ar_ratio);
        }
        for (name, ratios) in [("prewhitened HAC", hac), ("AR(1) quadratic", ar)] {
            let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
            assert!((mean - 1.0).abs() < 0.12, "iid {name} ratio should centre on 1, got {mean}");
        }
    }

    #[test]
    fn ar1_quadratic_ratio_matches_the_closed_form() {
        // Constant-sign weights under ρ: 1 + 2 Σ_{k≥1} (1 − k/n) ρ^k.
        let n = 50;
        let weights = vec![1.0; n];
        let mut residuals = Vec::with_capacity(n);
        let mut e = 0.0;
        for t in 0..n {
            e = 0.6 * e + if t % 3 == 0 { 1.0 } else { -0.4 };
            residuals.push(e);
        }
        let rho = kendall_rho(&residuals);
        let expected = 1.0
            + 2.0 * (1..n).map(|k| (1.0 - k as f64 / n as f64) * rho.powf(k as f64)).sum::<f64>();
        assert!((ar1_quadratic_ratio(&weights, &residuals) - expected).abs() < 1e-9);
        // Alternating weights flip the sign of every odd lag.
        let alternating: Vec<f64> = (0..n).map(|t| if t % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!(ar1_quadratic_ratio(&alternating, &residuals) < 1.0 || rho <= 0.0);
    }

    #[test]
    fn ar1_score_ratio_tracks_the_analytic_long_run_factor() {
        // Product of two independent AR(1)(rho) series is AR(1)-correlated with rho²:
        // LRV / Var = (1 + rho²) / (1 − rho²).
        let rho: f64 = 0.7;
        let truth = (1.0 + rho * rho) / (1.0 - rho * rho);
        let mut ratios = Vec::new();
        for seed in 0..40 {
            let f = long_run_tempering_factor(
                &design(2_000, rho, 100 + seed),
                &DependenceScope::Treatment,
            )
            .unwrap();
            ratios.push(f.kappa);
            assert!(
                tempering_kappa_from_notes(&[f.note()]).is_some_and(|k| (k - f.kappa).abs() < 1e-5)
            );
        }
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        assert!((mean / truth - 1.0).abs() < 0.15, "ratio {mean} vs analytic {truth}");
        // The treatment direction written explicitly is the same combination.
        let d = design(500, rho, 7);
        let a = long_run_tempering_factor(&d, &DependenceScope::Treatment).unwrap();
        let b =
            long_run_tempering_factor(&d, &DependenceScope::Direction(Arc::from([0.0, 1.0, 0.0])))
                .unwrap();
        assert!((a.raw_ratio - b.raw_ratio).abs() < 1e-9);
    }

    #[test]
    fn levels_scope_takes_the_largest_ratio_of_its_combinations() {
        let d = design(800, 0.7, 11);
        let slope = long_run_tempering_factor(&d, &DependenceScope::Treatment).unwrap();
        let intercept =
            long_run_tempering_factor(&d, &DependenceScope::Direction(Arc::from([1.0, 0.0, 0.0])))
                .unwrap();
        let levels = long_run_tempering_factor(
            &d,
            &DependenceScope::Levels(Arc::from([
                Arc::from([0.0, 1.0, 0.0]),
                Arc::from([1.0, 0.0, 0.0]),
                Arc::from([0.0, 0.0, 0.0]),
            ])),
        )
        .unwrap();
        assert_eq!(levels.scope, "response_levels");
        assert!((levels.raw_ratio - slope.raw_ratio.max(intercept.raw_ratio)).abs() < 1e-12);
        let wrong_width = long_run_tempering_factor(
            &d,
            &DependenceScope::Levels(Arc::from([Arc::from([1.0, 0.0])])),
        );
        assert!(wrong_width.is_err());
    }

    #[test]
    fn short_or_irrelevant_designs_do_not_temper() {
        let f = long_run_tempering_factor(&design(6, 0.9, 1), &DependenceScope::Treatment).unwrap();
        assert!((f.kappa - 1.0).abs() < f64::EPSILON);
        assert!(f.inestimable);
        let zero = long_run_tempering_factor(
            &design(200, 0.9, 1),
            &DependenceScope::Direction(Arc::from([0.0, 0.0, 0.0])),
        )
        .unwrap();
        assert!((zero.kappa - 1.0).abs() < f64::EPSILON);
        assert!(tempering_kappa_from_notes(&[Arc::from("unrelated")]).is_none());
    }

    /// Stationary AR(2) with unit innovations after a burn-in.
    fn ar2(n: usize, phi: [f64; 2], rng: &mut CausalRng) -> Vec<f64> {
        let (mut a, mut b) = (0.0, 0.0);
        let mut out = Vec::with_capacity(n);
        for t in 0..n + 500 {
            let next = phi[0] * a + phi[1] * b + standard_normal(rng);
            b = a;
            a = next;
            if t >= 500 {
                out.push(next);
            }
        }
        out
    }

    fn ar2_design(n: usize, phi: [f64; 2], seed: u64) -> CompiledDesign {
        let mut rng = CausalRng::from_seed(seed);
        let x = ar2(n, phi, &mut rng);
        let e = ar2(n, phi, &mut rng);
        let y: Vec<f64> = x.iter().zip(&e).map(|(x, e)| 0.8 * x + e).collect();
        CompiledDesign::linear_adjustment(&x, &[], &y, &[]).unwrap()
    }

    #[test]
    fn bic_autoregression_recovers_the_order_and_coefficients() {
        let mut rng = CausalRng::from_seed(5);
        let fit = bic_autoregression(&ar2(4_000, [0.3, 0.5], &mut rng));
        assert_eq!(fit.phi.len(), 2, "{fit:?}");
        assert!((fit.phi[0] - 0.3).abs() < 0.05 && (fit.phi[1] - 0.5).abs() < 0.05, "{fit:?}");
        assert!((fit.autocorrelation[1] - 0.6).abs() < 0.05, "{fit:?}");
        // AR(1) and white noise do not trigger the higher-order terms.
        let mut picked_high = 0;
        for seed in 0..40 {
            let mut rng = CausalRng::from_seed(100 + seed);
            let x = ar1(400, 0.7, &mut rng);
            picked_high += usize::from(bic_autoregression(&x).phi.len() >= 2);
        }
        assert!(picked_high <= 4, "BIC picked q >= 2 on {picked_high}/40 AR(1) series");
        // An AR(1) model reproduces the AR(1) quadratic form.
        let model = Autoregression { phi: vec![0.6], autocorrelation: vec![1.0, 0.6] };
        let weights: Vec<f64> = (0..50).map(|t| 1.0 + 0.1 * f64::from(t)).collect();
        let expected = {
            let (mut carry, mut cross, mut norm) = (0.0, 0.0, 0.0);
            for (t, &w) in weights.iter().enumerate() {
                if t > 0 {
                    carry = 0.6 * (carry + weights[t - 1]);
                }
                cross += w * carry;
                norm += w * w;
            }
            1.0 + 2.0 * cross / norm
        };
        assert!((ar_quadratic_ratio(&weights, &model) - expected).abs() < 1e-9);
    }

    #[test]
    fn ar2_score_ratio_tracks_the_analytic_long_run_factor() {
        // x and e independent AR(2)(0.3, 0.5): the score x e has autocorrelation ρ_k²,
        // so LRV / Var = 1 + 2 Σ ρ_k².
        let phi = [0.3, 0.5];
        let mut rho = vec![1.0, phi[0] / (1.0 - phi[1])];
        for k in 2..400 {
            rho.push(phi[0] * rho[k - 1] + phi[1] * rho[k - 2]);
        }
        let truth = 1.0 + 2.0 * rho[1..].iter().map(|r| r * r).sum::<f64>();
        let (mut kappas, mut ar1_only) = (Vec::new(), Vec::new());
        for seed in 0..30 {
            let d = ar2_design(2_000, phi, 300 + seed);
            let f = long_run_tempering_factor(&d, &DependenceScope::Treatment).unwrap();
            assert!(f.score_ar_order >= 2 && f.residual_ar_order >= 2, "{f:?}");
            kappas.push(f.kappa);
            // The AR(1)-only combination, for comparison.
            let x = &d.matrix[..2 * d.nrows];
            let residuals = FaerBackend
                .least_squares(x, d.nrows, 2, &d.outcome, &mut LeastSquaresWorkspace::default())
                .unwrap()
                .residuals;
            let centred: Vec<f64> = {
                let xs = &x[d.nrows..];
                let mean = xs.iter().sum::<f64>() / d.nrows as f64;
                let ss: f64 = xs.iter().map(|v| (v - mean).powi(2)).sum();
                xs.iter().map(|v| (v - mean) / ss).collect()
            };
            let influence: Vec<f64> = centred.iter().zip(&residuals).map(|(w, e)| w * e).collect();
            ar1_only.push(
                (long_run_variance_ratio(&influence, f.bandwidth) * f.fixed_b * f.fixed_b)
                    .max(ar1_quadratic_ratio(&centred, &residuals)),
            );
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        assert!(
            (mean(&kappas) / truth - 1.0).abs() < 0.15,
            "AR(q) kappa {} vs analytic {truth}",
            mean(&kappas)
        );
        assert!(
            mean(&ar1_only) < 0.8 * truth,
            "the AR(1)-only ratio under-corrects AR(2) dependence: {} vs {truth}",
            mean(&ar1_only)
        );
    }

    #[test]
    fn regressor_generated_residual_structure_does_not_inflate_kappa() {
        // Period-4 treatment with its second lag omitted: the residual is 3·t_{t-2}, a
        // perfectly periodic series BIC fits as AR(q), but its product with the lag-1
        // regressor is identically zero, so the score carries no dependence.
        let n = 401;
        let t: Vec<f64> = (0..=n)
            .map(|i| match i % 4 {
                1 => 1.0,
                3 => -1.0,
                _ => 0.0,
            })
            .collect();
        let x: Vec<f64> = t[1..].to_vec();
        let y: Vec<f64> =
            (0..n).map(|i| 2.0 * t[i + 1] + 3.0 * t[i] + 0.03 * (0.73 * i as f64).sin()).collect();
        let d = CompiledDesign::linear_adjustment(&x, &[], &y, &[]).unwrap();
        let f = long_run_tempering_factor(&d, &DependenceScope::Treatment).unwrap();
        assert!(f.residual_ar_order >= 2, "{f:?}");
        assert!(f.kappa < 2.0, "the bounded AR(q) term must not blow kappa up: {f:?}");
    }

    #[test]
    fn an_exact_fit_keeps_kappa_at_one() {
        // Noise-free outcome on a slowly drifting treatment: the residuals are rounding
        // error, whose structure must not be read as serial dependence.
        let n = 800;
        let t: Vec<f64> =
            (0..=n).map(|i| 0.3 + 0.4 * (i % 2) as f64 + 0.05 * (0.017 * i as f64).sin()).collect();
        let x: Vec<f64> = t[1..].to_vec();
        let lag: Vec<f64> = t[..n].to_vec();
        let y: Vec<f64> = x.iter().zip(&lag).map(|(a, b)| 1.0 + 2.0 * a + 3.0 * b).collect();
        let d = CompiledDesign::linear_adjustment(
            &x,
            &[(VariableId::from_raw(2), lag.as_slice())],
            &y,
            &[],
        )
        .unwrap();
        let f =
            long_run_tempering_factor(&d, &DependenceScope::Direction(Arc::from([0.0, 1.0, 1.0])))
                .unwrap();
        assert!((f.kappa - 1.0).abs() < f64::EPSILON, "{f:?}");
    }
}
