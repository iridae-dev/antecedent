//! Serial-dependence correction for Bayesian fits on time-ordered rows.
//!
//! The conjugate Normal–Inverse-Gamma likelihood (and the Laplace / HMC Gaussian
//! likelihoods) treat design rows as independent. On a lag-aligned temporal
//! design the outcome residual can be autocorrelated; when the regressor of
//! interest is persistent too, the iid posterior of the targeted linear
//! combination `c'β` is too narrow. [`SerialDependence::LongRunTempering`]
//! replaces the likelihood `L(θ)` by the power likelihood `L(θ)^{1/κ̂}` (every row
//! weighted `1/κ̂`, i.e. an effective sample of `n/κ̂` rows). The result is a
//! generalized (power) posterior, not the posterior of a correctly specified
//! dependent-data model: the prior keeps its full weight, the posterior scale of
//! every coefficient is widened by `√κ̂` (calibrated for the targeted
//! combination), and heteroskedasticity or a misspecified mean are **not**
//! corrected.
//!
//! With [`DependenceScope::Levels`] one fit targets several combinations (every
//! grid cell of a temporal response at one horizon), and with
//! [`DependenceScope::MediationPaths`] the path coefficients a linear mediation
//! mechanism contributes to; `κ̂` is then the largest of their factors.
//!
//! # Derivation
//!
//! Let the residual follow a stationary Gaussian AR(q) with correlation matrix
//! `R` and let `w = X (X'X)⁻¹ c` be the row weights of `c'β̂`. Two quantities
//! separate the iid posterior of `c'β` from its sampling law:
//!
//! 1. **Variance ratio.** `Var(c'β̂) = σ² w'Rw`, against the iid `σ² w'w`:
//!    the ratio `κ_R = w'Rw / w'w` is exact conditional on the realized design
//!    (no long-run approximation, no kernel bandwidth).
//! 2. **Residual scale.** The tempered posterior estimates `σ²` from the OLS
//!    residual sum of squares as if `n/κ̂` rows were independent, but
//!    `E[RSS] = σ² (n − tr(HR))` with `H = X(X'X)⁻¹X'`: a persistent residual
//!    loses `d = tr(HR)` rows of scale to the projection (`d = p` when iid;
//!    on a 59-row AR(2)(0.3, 0.5) design with intercept and one persistent
//!    regressor `d ≈ 17`). The factor `n / (n − d)` restores the scale.
//!
//! The tempering factor is therefore
//!
//! ```text
//! κ̂ = [w'R̂w / w'w] · n / (n − tr(H R̂)) · exp(τ̂² / 2)
//! ```
//!
//! where `R̂` is the correlation matrix of an AR(q) fitted to the residual by
//! **restricted maximum likelihood** (REML; `q ≤ 4` by BIC on the REML
//! likelihood, `−2ℓ_R + q ln(n − p)`). REML maximizes the likelihood of the
//! residual contrasts, so the projection that decorrelates the OLS residuals is
//! accounted for exactly (Cheang & Reinsel 2000); a Yule–Walker or Burg fit to
//! the OLS residuals is biased toward independence by the same `tr(HR)`
//! mechanism, and no small-sample bias correction of that fit recovers what the
//! projection removed. `τ̂` is the delta-method standard deviation of `log κ̂`
//! from the observed REML information (the AR parameters are the only estimated
//! inputs of `κ̂`; `w` and `H` are fixed by the design): integrating the
//! coefficient posterior over a lognormal `κ` with that spread has variance
//! `κ̂ exp(τ̂²/2) · Var_iid`, so the posterior mean of `κ` is applied rather
//! than its plug-in. The AR(q) is parameterized by its partial autocorrelations
//! (bounded at `±0.97`), so every fitted model is stationary.
//!
//! The factor is floored at `1` (the correction never narrows the iid
//! posterior), the scale denominator `n − d` is floored at `p + 2`, and `κ̂` is
//! capped so at least `p + 2` effective rows remain. Two guards keep the
//! parametric model honest:
//!
//! - **Regressor-generated residual structure.** The AR model assumes the
//!   residual autocorrelation is independent of the regressor path. Residual
//!   structure that is a function of the regressors (an omitted seasonal lag of
//!   a seasonal treatment) never reaches the score `w ⊙ ê`, so `κ̂` may exceed
//!   three times the autoregressive-prewhitened Newey–West long-run-variance
//!   ratio of that score (AR(1) with the Kendall bias correction, or a
//!   BIC-selected AR(q) prewhitening; bandwidth `⌊4 (n/100)^{2/9}⌋`; scaled by
//!   the squared Kiefer–Vogelsang fixed-b factor of bandwidth `M + 1`,
//!   [`crate::temporal_block::fixed_b_scale`]) by at most that bound. On the
//!   calibration designs the bound never binds.
//! - **Exact fits.** A residual sum of squares at most `1e-20` of the outcome's
//!   centred sum of squares keeps `κ̂ = 1`: the residuals are rounding error,
//!   whose apparent autocorrelation would otherwise drive `κ̂` to the cap.
//!
//! The model is calibrated for short-memory dependence an AR(4) captures: in
//! calibration (nominal 90%, 400 replicates, `n = 60–400`) AR(1),
//! ARMA(1,1), MA(2) and AR(2)(0.3, 0.5) treatment-and-residual designs cover
//! 0.87–0.92 for Pulse, Sustained and temporal mediation; the previous
//! kernel-and-AR(1) rule left AR(2) at 0.81 (`n = 60`) and 0.87 (`n = 160`).
//! Long memory, or autocorrelation beyond what an AR(4) captures, is
//! under-corrected.
//!
//! # Cost
//!
//! Everything that depends only on the design — the OLS residual, the REML
//! fit at every candidate order, `X'R̂X` and `tr(HR̂)` on the delta-method
//! stencil, the observed information — is computed once per design
//! ([`long_run_tempering_factor`] keeps a small content-keyed cache), and each
//! REML likelihood evaluation is O(q²p²) from lagged cross-products of the
//! augmented design accumulated once, so the fit costs about the same at any
//! `n`. Each combination then adds O(p²) quadratic forms and, only while it can
//! still drive `κ̂`, the O(n) score HAC bound.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::sync::Arc;

use antecedent_stats::{CompiledDesign, DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::ar_kernel::{MAX_AR_ORDER, MAX_AR_RHO, bic_autoregression, dot, kendall_rho};
use crate::error::EstimationError;
use crate::util::gram;

/// Stable prefix of the inference-diagnostics note recording the tempering factor.
pub const DEPENDENCE_NOTE_PREFIX: &str = "serial_dependence.long_run_tempering";

/// Assumption id recorded on posteriors fitted under [`SerialDependence::LongRunTempering`].
pub const DEPENDENCE_ASSUMPTION_ID: &str = "bayes.temporal.long_run_tempering";

/// Largest absolute partial autocorrelation of the fitted AR(q), and largest
/// absolute AR(1) prewhitening coefficient (keeps the recolouring finite).
const MAX_PARTIAL_AUTOCORRELATION: f64 = MAX_AR_RHO;

/// Fewest rows on which a tempering factor is estimated; shorter designs keep `κ = 1`.
const MIN_ROWS: usize = 8;

/// Residual sum of squares, relative to the outcome's centred sum of squares, at or
/// below which the fit is exact and `κ̂` stays at `1`.
const EXACT_FIT_RELATIVE_SS: f64 = 1e-20;

/// Largest factor by which the autoregressive factor may exceed the
/// fixed-b-scaled score HAC ratio. On the calibration designs the bound
/// never binds; on a deterministic period-4 treatment with an omitted lag it
/// stops a residual quadratic form of ≈145 against a score ratio of ≈0.2.
const AR_QUADRATIC_HAC_BOUND: f64 = 3.0;

/// Largest delta-method standard deviation of `log κ̂` that is propagated
/// (`exp(1/2)` is the largest inflation an ill-conditioned information can add).
const MAX_KAPPA_LOG_SD: f64 = 1.0;

/// Finite-difference step (partial-autocorrelation coordinates) of the
/// delta-method gradient and the observed-information Hessian.
const DELTA_STEP: f64 = 1e-3;

/// Nelder–Mead initial simplex edge in the unbounded partial-autocorrelation
/// coordinates `z = artanh(r / 0.97)`.
const SIMPLEX_STEP: f64 = 0.2;

/// Nelder–Mead convergence: objective range and simplex extent.
const SIMPLEX_F_TOL: f64 = 1e-12;
const SIMPLEX_X_TOL: f64 = 1e-8;

/// Nelder–Mead iteration budget per coordinate.
const SIMPLEX_ITERATIONS_PER_DIM: usize = 500;

/// Newton polish of each order's simplex optimum: at most this many
/// fixed-Hessian steps, stopping once a step moves no coordinate by more than
/// `NEWTON_POLISH_TOL`; a step is accepted only if it does not lower the
/// likelihood by more than `NEWTON_POLISH_SLACK` of its magnitude (rounding).
const NEWTON_POLISH_STEPS: usize = 4;
const NEWTON_POLISH_TOL: f64 = 1e-11;
const NEWTON_POLISH_SLACK: f64 = 1e-10;

/// Row-dependence model for a Bayesian likelihood.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum SerialDependence {
    /// Rows are exchangeable; the iid likelihood is the stated model.
    #[default]
    Iid,
    /// Rows are time-ordered; temper the likelihood by the variance ratio of the
    /// targeted linear combination under an autoregressive residual.
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
    /// factors, so every cell's interval carries at least its own correction.
    Levels(Arc<[Arc<[f64]>]>),
    /// The path combinations one linear mediation mechanism contributes to: the
    /// treatment→mediator slope `a`, or the direct `c'`, mediated `b` and total
    /// `c' + â b` gradients of the outcome mechanism (`â` the plug-in path
    /// coefficient). `κ̂` is the largest of their factors, so every published
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
    /// Combined factor before the floor / cap:
    /// `min(ar_ratio · n/(n − df_loss) · exp(kappa_log_sd²/2), 3 · hac_ratio · fixed_b²)`.
    pub raw_ratio: f64,
    /// Variance ratio `w'R̂w / w'w` of the driving combination under the REML AR(q)
    /// residual, conditional on the realized design.
    pub ar_ratio: f64,
    /// Rows of residual scale lost to the projection, `tr(H R̂)` (`p` when iid).
    pub df_loss: f64,
    /// Delta-method standard deviation of `log κ̂` from the observed REML information.
    pub kappa_log_sd: f64,
    /// Autoregressive-prewhitened Bartlett long-run-variance ratio of the driving
    /// score (the larger of the AR(1) and the BIC-selected AR(q) prewhitening); with
    /// `fixed_b` it bounds the autoregressive factor.
    pub hac_ratio: f64,
    /// Kiefer–Vogelsang fixed-b SE factor of the Bartlett bandwidth
    /// ([`crate::temporal_block::fixed_b_scale`]); `hac_ratio` is scaled by its square.
    pub fixed_b: f64,
    /// BIC-selected autoregressive order of the driving score (`0..=4`) in the
    /// bounding HAC ratio.
    pub score_ar_order: usize,
    /// BIC-selected REML order of the residual AR(q) (`0..=4`).
    pub residual_ar_order: usize,
    /// Bartlett bandwidth on the prewhitened score.
    pub bandwidth: usize,
    /// Design rows.
    pub nrows: usize,
    /// Whether the score HAC bound held the autoregressive factor down.
    pub bounded: bool,
    /// Whether the cap bound `κ̂` (the correction is then incomplete).
    pub capped: bool,
    /// Whether the factor could not be estimated (`n` below `max(8, p+2)`);
    /// `κ = 1` then leaves the iid posterior in force.
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
            "{DEPENDENCE_NOTE_PREFIX} kappa={:.6} raw_ratio={:.6} ar_ratio={:.6} df_loss={:.6} \
             kappa_log_sd={:.6} hac_ratio={:.6} fixed_b={:.6} score_ar_order={} \
             residual_ar_order={} n={} n_eff={:.3} bandwidth={} scope={} bounded={} capped={} \
             inestimable={}",
            self.kappa,
            self.raw_ratio,
            self.ar_ratio,
            self.df_loss,
            self.kappa_log_sd,
            self.hac_ratio,
            self.fixed_b,
            self.score_ar_order,
            self.residual_ar_order,
            self.nrows,
            self.effective_rows(),
            self.bandwidth,
            self.scope,
            self.bounded,
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
             variance ratio of the {} combination given the design under a REML-fitted \
             autoregressive residual (BIC order {}; ratio {:.4}), times the residual-scale \
             factor n/(n - tr(HR)) for the {:.2} rows the projection removes, times \
             exp(tau^2/2) for the delta-method spread tau = {:.4} of log kappa; bounded by \
             three times the prewhitened Newey-West long-run-variance ratio of the score \
             (bandwidth {}, BIC score order {}, fixed-b factor {:.4}){}; floored at 1{}; \
             effective rows {:.1}; the prior keeps full weight; calibrated for autoregressive \
             dependence up to order 4, not for long memory; heteroskedasticity and mean \
             misspecification are not corrected",
            self.nrows,
            self.kappa,
            self.scope,
            self.residual_ar_order,
            self.ar_ratio,
            self.df_loss,
            self.kappa_log_sd,
            self.bandwidth,
            self.score_ar_order,
            self.fixed_b,
            if self.bounded { ", which bound it" } else { "" },
            if self.inestimable {
                "; factor inestimable on this short design — iid posterior retained and is \
                 likely too narrow"
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

/// Whether any tempering note recorded that the factor could not be estimated.
#[must_use]
pub fn tempering_inestimable_from_notes(notes: &[Arc<str>]) -> bool {
    note_flag(notes, "inestimable=")
}

/// Estimate the tempering factor for `design` over `scope`.
///
/// Rows must be in time order. The residual REML fit (the AR order, its
/// correlation operator, the projected trace `tr(HR̂)` and the observed
/// information) depends only on the design, so it is computed once per design
/// and shared by every combination of `scope`; repeated calls on the same rows
/// (every horizon of a temporal response is its own design, but the class atoms
/// of one adjustment set, a Study's estimate and its checks, or a re-run on the
/// same series are not) reuse it through a small content-keyed cache
/// ([`residual_fit`]). The fit is a pure function of the design, so a cache hit
/// and a recomputation are indistinguishable.
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
        ar_ratio: 1.0,
        df_loss: p as f64,
        kappa_log_sd: 0.0,
        hac_ratio: 1.0,
        fixed_b: 1.0,
        score_ar_order: 0,
        residual_ar_order: 0,
        bandwidth,
        nrows: n,
        bounded: false,
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
    let fit = residual_fit(design)?;
    // An exact fit leaves only rounding error, whose autocorrelation is an artefact of the
    // arithmetic: there is no residual dependence to correct.
    if fit.exact {
        return Ok(floor);
    }
    let x = &design.matrix[..n * p];
    let q = fit.model.phi.len();
    // Every combination's autoregressive factor: v = (X'X)⁻¹ c gives the row weights
    // w = X v, and w'Rw / w'w = v'(X'RX)v / v'(X'X)v from the design-level quadratic forms.
    let mut factors: Vec<(usize, f64, f64, f64)> = Vec::with_capacity(directions.len());
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(directions.len());
    for (i, c) in directions.iter().enumerate() {
        let v = crate::util::solve_spd(&fit.xtx, c, p)
            .ok_or_else(|| EstimationError::stats_msg("long-run tempering: singular X'X"))?;
        let norm = quadratic_form(&fit.xtx, &v, p);
        let ratios: Vec<f64> = fit
            .quadratic
            .iter()
            .map(|m| match m {
                Some(m) if norm > 0.0 => quadratic_form(m, &v, p) / norm,
                _ => 1.0,
            })
            .collect();
        let log_kappa = |k: usize| (ratios[k] * fit.scale_factor(k)).ln();
        let tau = match &fit.info_chol {
            Some(l) if q > 0 => {
                let g: Vec<f64> = (0..q)
                    .map(|i| (log_kappa(1 + 2 * i) - log_kappa(2 + 2 * i)) / (2.0 * DELTA_STEP))
                    .collect();
                let u = cholesky_solve(l, &g, q);
                let var: f64 = g.iter().zip(&u).map(|(a, b)| a * b).sum();
                if var.is_finite() && var > 0.0 { var.sqrt().min(MAX_KAPPA_LOG_SD) } else { 0.0 }
            }
            _ => 0.0,
        };
        let ar = ratios[0];
        let unbounded = fit.scale_factor(0) * ar * (0.5 * tau * tau).exp();
        factors.push((i, unbounded, ar, tau));
        weights.push(v);
    }
    // The combination with the largest factor drives κ̂; its components are reported.
    // The score HAC bound is only needed for a combination whose unbounded factor can
    // still beat the best combined factor, so it is taken in descending order and a
    // combination that cannot win skips the bound (ties keep the first in scope order).
    factors.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut best: Option<(usize, TemperingFactor)> = None;
    for &(i, unbounded, ar, tau) in &factors {
        if !unbounded.is_finite() {
            continue;
        }
        if let Some((best_i, b)) = &best {
            match unbounded.total_cmp(&b.raw_ratio) {
                std::cmp::Ordering::Less => break,
                std::cmp::Ordering::Equal if i > *best_i => break,
                _ => {}
            }
        }
        let v = &weights[i];
        let w: Vec<f64> =
            (0..n).map(|t| (0..p).map(|k| x[k * n + t] * v[k]).sum::<f64>()).collect();
        let influence: Vec<f64> = w.iter().zip(&fit.residuals).map(|(w, e)| w * e).collect();
        let score_ar = bic_autoregression(&influence);
        let mut hac = long_run_variance_ratio(&influence, bandwidth);
        if score_ar.phi.len() >= 2 {
            hac = hac.max(ar_prewhitened_variance_ratio(&influence, &score_ar.phi, bandwidth));
        }
        let bound = AR_QUADRATIC_HAC_BOUND * hac * fit.fixed_b * fit.fixed_b;
        if !bound.is_finite() {
            continue;
        }
        let combined = unbounded.min(bound);
        let wins = match &best {
            None => true,
            Some((best_i, b)) => match combined.total_cmp(&b.raw_ratio) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Equal => i < *best_i,
                std::cmp::Ordering::Less => false,
            },
        };
        if wins {
            best = Some((
                i,
                TemperingFactor {
                    raw_ratio: combined,
                    ar_ratio: ar,
                    df_loss: fit.df_loss[0],
                    kappa_log_sd: tau,
                    hac_ratio: hac,
                    fixed_b: fit.fixed_b,
                    score_ar_order: score_ar.phi.len(),
                    residual_ar_order: q,
                    bounded: bound < unbounded,
                    ..floor
                },
            ));
        }
    }
    let best = best.map(|(_, factor)| factor);
    let Some(factor) = best else {
        return Ok(floor);
    };
    let cap = (n as f64 / (p + 2) as f64).max(1.0);
    Ok(TemperingFactor {
        kappa: factor.raw_ratio.clamp(1.0, cap),
        capped: factor.raw_ratio > cap,
        ..factor
    })
}

/// `v'Mv` for a row-major `p × p` matrix `M`.
fn quadratic_form(m: &[f64], v: &[f64], p: usize) -> f64 {
    (0..p).map(|a| v[a] * (0..p).map(|b| m[a * p + b] * v[b]).sum::<f64>()).sum()
}

/// Design-level part of the tempering factor: the OLS residual, the Gram
/// matrix, the REML AR(q) residual model and everything of `κ̂` that does not
/// depend on the targeted combination.
#[derive(Debug)]
struct ResidualFit {
    n: usize,
    p: usize,
    /// OLS residuals `y − Xβ̂`.
    residuals: Vec<f64>,
    /// `X'X`, row-major.
    xtx: Vec<f64>,
    /// Residual sum of squares at most `EXACT_FIT_RELATIVE_SS` of the centred outcome.
    exact: bool,
    /// REML AR(q) residual model at the BIC order.
    model: ArErrorModel,
    /// `X'RX` (row-major `p × p`) under the correlation `R` of the fitted model
    /// (index `0`) and of the models at `z ± DELTA_STEP e_i` (indices `1 + 2i`
    /// and `2 + 2i`, the delta-method stencil; absent for `q = 0`). `None` when
    /// the model is not positive definite or the projected trace is not
    /// solvable: the identity is used, so every ratio is `1`.
    quadratic: Vec<Option<Vec<f64>>>,
    /// `tr(H R)` at each stencil point (`p` for the identity).
    df_loss: Vec<f64>,
    /// Cholesky factor of the observed REML information in the `z` coordinates;
    /// `None` for `q = 0` or when the finite-difference Hessian is not positive
    /// definite (the spread is then zero).
    info_chol: Option<Vec<f64>>,
    /// Kiefer–Vogelsang fixed-b factor of the bounding score HAC.
    fixed_b: f64,
}

impl ResidualFit {
    /// Residual-scale factor `n / (n − tr(HR))` at operator `k`, denominator floored at `p + 2`.
    fn scale_factor(&self, k: usize) -> f64 {
        self.n as f64 / (self.n as f64 - self.df_loss[k]).max((self.p + 2) as f64)
    }

    /// Fit the design (at least `max(8, p + 2)` rows).
    fn new(design: &CompiledDesign) -> Result<Self, EstimationError> {
        let n = design.nrows;
        let p = design.ncols;
        let x = &design.matrix[..n * p];
        let y = &design.outcome[..n];
        let residuals =
            FaerBackend.least_squares(x, n, p, y, &mut LeastSquaresWorkspace::default())?.residuals;
        let outcome_mean = y.iter().sum::<f64>() / n as f64;
        let centred_ss: f64 = y.iter().map(|v| (v - outcome_mean).powi(2)).sum();
        let residual_ss: f64 = residuals.iter().map(|e| e * e).sum();
        let xtx = gram(x, n, p);
        let bandwidth = newey_west_bandwidth(n);
        let fixed_b = crate::temporal_block::fixed_b_scale(bandwidth + 1, n);
        let mut fit = Self {
            n,
            p,
            residuals,
            xtx,
            exact: residual_ss <= EXACT_FIT_RELATIVE_SS * centred_ss,
            model: ArErrorModel::default(),
            quadratic: Vec::new(),
            df_loss: Vec::new(),
            info_chol: None,
            fixed_b,
        };
        if fit.exact {
            return Ok(fit);
        }
        let kernel = RemlKernel::new(x, &fit.residuals, n, p);
        fit.model = reml_autoregression(&kernel);
        let q = fit.model.z.len();
        // Quadratic forms at z and on the delta-method stencil, each with its projected trace.
        let mut stencil: Vec<Vec<f64>> = vec![fit.model.z.clone()];
        for i in 0..q {
            for s in [DELTA_STEP, -DELTA_STEP] {
                let mut z = fit.model.z.clone();
                z[i] += s;
                stencil.push(z);
            }
        }
        for z in &stencil {
            let (m, loss) = fit.projected_trace(x, &z_to_phi(z));
            fit.quadratic.push(m);
            fit.df_loss.push(loss);
        }
        fit.info_chol = (q > 0).then(|| observed_information(&fit.model.z, &kernel)).flatten();
        Ok(fit)
    }

    /// `X'RX` under the correlation `R` of the AR model `phi`, and
    /// `tr(HR) = Σ_j [(X'X)⁻¹ X'R x_j]_j`; `(None, p)` when the model is not
    /// positive definite or the trace is not solvable.
    fn projected_trace(&self, x: &[f64], phi: &[f64]) -> (Option<Vec<f64>>, f64) {
        let (n, p) = (self.n, self.p);
        let Some(op) = Whitener::new(phi) else {
            return (None, p as f64);
        };
        let mut xrx = vec![0.0; p * p];
        let mut df_loss = 0.0;
        for j in 0..p {
            let rx = op.correlate(&x[j * n..(j + 1) * n]);
            let col: Vec<f64> = (0..p)
                .map(|a| x[a * n..(a + 1) * n].iter().zip(&rx).map(|(u, v)| u * v).sum())
                .collect();
            match crate::util::solve_spd(&self.xtx, &col, p) {
                Some(v) => df_loss += v[j],
                None => return (None, p as f64),
            }
            for a in 0..p {
                xrx[a * p + j] = col[a];
            }
        }
        (Some(xrx), df_loss)
    }
}

/// Cached design-level fits, most recently used first: at most this many
/// designs, and at most `FIT_CACHE_MAX_ELEMENTS` retained `f64` values across
/// them (the cache keeps each design's rows alive to verify a hit by content,
/// so a few large designs evict the rest; the most recent always stays).
const FIT_CACHE_CAPACITY: usize = 16;
const FIT_CACHE_MAX_ELEMENTS: usize = 1 << 21;

struct CachedFit {
    n: usize,
    p: usize,
    hash: u64,
    matrix: Arc<[f64]>,
    outcome: Arc<[f64]>,
    fit: Arc<ResidualFit>,
}

static FIT_CACHE: std::sync::Mutex<Vec<CachedFit>> = std::sync::Mutex::new(Vec::new());

/// Content fingerprint of the fitted rows (verified by full comparison on a hit).
fn design_hash(x: &[f64], y: &[f64]) -> u64 {
    x.iter()
        .chain(y)
        .fold(0xcbf2_9ce4_8422_2325_u64, |h, v| (h ^ v.to_bits()).wrapping_mul(0x0100_0000_01b3))
}

/// The design-level fit of `design`, from the cache when the same rows were fitted
/// before (a process-wide cache of [`FIT_CACHE_CAPACITY`] designs, keyed on the
/// row contents so the result is a pure function of the design).
fn residual_fit(design: &CompiledDesign) -> Result<Arc<ResidualFit>, EstimationError> {
    let n = design.nrows;
    let p = design.ncols;
    let x = &design.matrix[..n * p];
    let y = &design.outcome[..n];
    let hash = design_hash(x, y);
    let mut cache = FIT_CACHE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(i) = cache.iter().position(|c| {
        c.n == n && c.p == p && c.hash == hash && &c.matrix[..n * p] == x && &c.outcome[..n] == y
    }) {
        let hit = cache.remove(i);
        let fit = Arc::clone(&hit.fit);
        cache.insert(0, hit);
        return Ok(fit);
    }
    drop(cache);
    let fit = Arc::new(ResidualFit::new(design)?);
    let mut cache = FIT_CACHE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    cache.insert(
        0,
        CachedFit {
            n,
            p,
            hash,
            matrix: Arc::clone(&design.matrix),
            outcome: Arc::clone(&design.outcome),
            fit: Arc::clone(&fit),
        },
    );
    cache.truncate(FIT_CACHE_CAPACITY);
    let retained = |c: &CachedFit| c.matrix.len() + c.outcome.len() + c.fit.residuals.len();
    let mut total: usize = cache.iter().map(retained).sum();
    while total > FIT_CACHE_MAX_ELEMENTS && cache.len() > 1 {
        total -= cache.pop().map_or(0, |c| retained(&c));
    }
    Ok(fit)
}

/// REML AR(q) fit of the residual process.
#[derive(Clone, Debug, Default)]
struct ArErrorModel {
    /// Coefficients `φ₁ … φ_q` (empty for `q = 0`).
    phi: Vec<f64>,
    /// Unbounded partial-autocorrelation coordinates `z` (`r = 0.97 tanh z`).
    z: Vec<f64>,
}

/// Fixed-capacity AR coefficient vector (`q ≤ MAX_AR_ORDER`).
type ArCoefficients = [f64; MAX_AR_ORDER];

/// Partial autocorrelations `r` (`|r_k| < 1`) to AR coefficients (Levinson step-up).
fn pacf_to_phi(r: &[f64]) -> Vec<f64> {
    let mut phi = [0.0; MAX_AR_ORDER];
    pacf_to_phi_into(r, &mut phi);
    phi[..r.len()].to_vec()
}

fn pacf_to_phi_into(r: &[f64], phi: &mut ArCoefficients) {
    debug_assert!(r.len() <= MAX_AR_ORDER);
    let mut next = [0.0; MAX_AR_ORDER];
    for (m, &k) in r.iter().enumerate() {
        for j in 0..m {
            next[j] = phi[j] - k * phi[m - 1 - j];
        }
        next[m] = k;
        phi[..=m].copy_from_slice(&next[..=m]);
    }
}

fn z_to_phi(z: &[f64]) -> Vec<f64> {
    let r: Vec<f64> = z.iter().map(|v| MAX_PARTIAL_AUTOCORRELATION * v.tanh()).collect();
    pacf_to_phi(&r)
}

/// Whitening transform `L` of a stationary AR(q) with unit innovations
/// (`Γ⁻¹ = L'L`): rows `t < q` are the inverse Cholesky factor of the
/// stationary `q × q` autocovariance, rows `t ≥ q` the AR filter.
#[derive(Debug)]
struct Whitener {
    q: usize,
    phi: ArCoefficients,
    /// Lower Cholesky factor `C` of the `q × q` stationary autocovariance (row-major).
    chol: [f64; MAX_AR_ORDER * MAX_AR_ORDER],
    /// `log|Γ| = 2 Σ log diag(C)`.
    log_det: f64,
    /// Marginal variance `γ₀` of the process with unit innovations.
    gamma0: f64,
}

impl Whitener {
    /// `None` when the implied autocovariance is not positive definite.
    fn new(phi: &[f64]) -> Option<Self> {
        let q = phi.len();
        if q > MAX_AR_ORDER {
            return None;
        }
        let mut coefficients = [0.0; MAX_AR_ORDER];
        coefficients[..q].copy_from_slice(phi);
        Self::from_coefficients(q, coefficients)
    }

    /// The whitener of the model at partial-autocorrelation coordinates `z`.
    fn from_z(z: &[f64]) -> Option<Self> {
        let q = z.len();
        if q > MAX_AR_ORDER {
            return None;
        }
        let mut r = [0.0; MAX_AR_ORDER];
        for (r, v) in r.iter_mut().zip(z) {
            *r = MAX_PARTIAL_AUTOCORRELATION * v.tanh();
        }
        let mut phi = [0.0; MAX_AR_ORDER];
        pacf_to_phi_into(&r[..q], &mut phi);
        Self::from_coefficients(q, phi)
    }

    fn from_coefficients(q: usize, phi: ArCoefficients) -> Option<Self> {
        let mut chol = [0.0; MAX_AR_ORDER * MAX_AR_ORDER];
        if q == 0 {
            return Some(Self { q, phi, chol, log_det: 0.0, gamma0: 1.0 });
        }
        // Yule–Walker: ρ_k = Σ_j φ_j ρ_{|k−j|}, k = 1..q, with ρ_0 = 1.
        let mut a = [0.0; MAX_AR_ORDER * MAX_AR_ORDER];
        let mut rho = [0.0; MAX_AR_ORDER];
        for k in 1..=q {
            for j in 1..=q {
                let idx = k.abs_diff(j);
                if idx == 0 {
                    rho[k - 1] -= phi[j - 1];
                } else {
                    a[(k - 1) * q + idx - 1] += phi[j - 1];
                }
            }
            a[(k - 1) * q + k - 1] -= 1.0;
        }
        if !solve_dense_in_place(&mut a, &mut rho, q) {
            return None;
        }
        let gamma0 = 1.0 / (1.0 - phi[..q].iter().zip(&rho).map(|(f, r)| f * r).sum::<f64>());
        if !gamma0.is_finite() || gamma0 <= 0.0 {
            return None;
        }
        let mut cov = [0.0; MAX_AR_ORDER * MAX_AR_ORDER];
        for i in 0..q {
            for j in 0..q {
                let lag = i.abs_diff(j);
                cov[i * q + j] = gamma0 * if lag == 0 { 1.0 } else { rho[lag - 1] };
            }
        }
        if !cholesky_into(&cov, &mut chol, q) {
            return None;
        }
        let log_det = 2.0 * (0..q).map(|i| chol[i * q + i].ln()).sum::<f64>();
        Some(Self { q, phi, chol, log_det, gamma0 })
    }

    fn order(&self) -> usize {
        self.q
    }

    fn coefficients(&self) -> &[f64] {
        &self.phi[..self.q]
    }

    /// `u = L v`.
    #[cfg(test)]
    fn apply(&self, v: &[f64]) -> Vec<f64> {
        let q = self.order();
        let n = v.len();
        let mut u = vec![0.0; n];
        // Forward substitution C u_top = v_top.
        for i in 0..q.min(n) {
            let s: f64 = (0..i).map(|j| self.chol[i * q + j] * u[j]).sum();
            u[i] = (v[i] - s) / self.chol[i * q + i];
        }
        for t in q..n {
            u[t] = v[t]
                - self
                    .coefficients()
                    .iter()
                    .enumerate()
                    .map(|(j, f)| f * v[t - 1 - j])
                    .sum::<f64>();
        }
        u
    }

    /// `v = L⁻¹ u`.
    #[cfg(test)]
    fn solve(&self, u: &[f64]) -> Vec<f64> {
        let mut v = vec![0.0; u.len()];
        self.solve_into(u, &mut v);
        v
    }

    /// `v = L⁻¹ u`, written into `v`.
    fn solve_into(&self, u: &[f64], v: &mut [f64]) {
        let q = self.order();
        let n = u.len();
        for (i, vi) in v.iter_mut().enumerate().take(q.min(n)) {
            *vi = (0..=i).map(|j| self.chol[i * q + j] * u[j]).sum();
        }
        // v_t = u_t + Σ_j φ_j v_{t−1−j} for t ≥ q.
        match q {
            0 => v[..n].copy_from_slice(u),
            1 => recolour::<1>(&self.phi, u, v),
            2 => recolour::<2>(&self.phi, u, v),
            3 => recolour::<3>(&self.phi, u, v),
            _ => recolour::<4>(&self.phi, u, v),
        }
    }

    /// `z = L'⁻¹ x`, written into `z`.
    fn solve_transpose_into(&self, x: &[f64], z: &mut [f64]) {
        let q = self.order();
        let n = x.len();
        // z_t = x_t + Σ_j φ_j z_{t+1+j} over the filter rows t + 1 + j ∈ [q, n).
        match q {
            0 => z[..n].copy_from_slice(x),
            1 => recolour_transpose::<1>(&self.phi, x, z),
            2 => recolour_transpose::<2>(&self.phi, x, z),
            3 => recolour_transpose::<3>(&self.phi, x, z),
            _ => recolour_transpose::<4>(&self.phi, x, z),
        }
        let top = q.min(n);
        let mut rhs = [0.0; MAX_AR_ORDER];
        for (t, r) in rhs.iter_mut().enumerate().take(top) {
            let mut s = x[t];
            for (j, f) in self.coefficients().iter().enumerate() {
                let row = t + 1 + j;
                if row >= q && row < n {
                    s += f * z[row];
                }
            }
            *r = s;
        }
        // z_top = C' rhs.
        for (i, zi) in z.iter_mut().enumerate().take(top) {
            *zi = (i..top).map(|j| self.chol[j * q + i] * rhs[j]).sum();
        }
    }

    /// `R v = Γ v / γ₀` with `R` the correlation matrix of the process.
    fn correlate(&self, v: &[f64]) -> Vec<f64> {
        let mut tmp = vec![0.0; v.len()];
        self.solve_transpose_into(v, &mut tmp);
        let mut out = vec![0.0; v.len()];
        self.solve_into(&tmp, &mut out);
        for o in &mut out {
            *o /= self.gamma0;
        }
        out
    }
}

/// Forward AR(Q) recolouring `v_t = u_t + Σ_{j<Q} φ_j v_{t−1−j}` for `t ≥ Q`
/// (rows `t < Q` of `v` are inputs).
fn recolour<const Q: usize>(phi: &ArCoefficients, u: &[f64], v: &mut [f64]) {
    let n = u.len();
    for t in Q..n {
        let mut s = u[t];
        for j in 0..Q {
            s += phi[j] * v[t - 1 - j];
        }
        v[t] = s;
    }
}

/// Backward AR(Q) recolouring `z_t = x_t + Σ_{j<Q, t+1+j<n} φ_j z_{t+1+j}` for
/// `t ≥ Q`, in descending `t`.
fn recolour_transpose<const Q: usize>(phi: &ArCoefficients, x: &[f64], z: &mut [f64]) {
    let n = x.len();
    // Tail rows with fewer than Q leads available.
    for t in (Q.max(n.saturating_sub(Q))..n).rev() {
        let mut s = x[t];
        for j in 0..Q.min(n - t - 1) {
            s += phi[j] * z[t + 1 + j];
        }
        z[t] = s;
    }
    for t in (Q..n.saturating_sub(Q)).rev() {
        let mut s = x[t];
        for j in 0..Q {
            s += phi[j] * z[t + 1 + j];
        }
        z[t] = s;
    }
}

/// Lower Cholesky factor (row-major) of a symmetric positive-definite `m × m` matrix.
fn cholesky(a: &[f64], m: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; m * m];
    cholesky_into(a, &mut l, m).then_some(l)
}

/// [`cholesky`] into a caller-provided factor (`false` when not positive definite).
fn cholesky_into(a: &[f64], l: &mut [f64], m: usize) -> bool {
    for i in 0..m {
        for j in 0..=i {
            let mut s = a[i * m + j];
            for k in 0..j {
                s -= l[i * m + k] * l[j * m + k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return false;
                }
                l[i * m + i] = s.sqrt();
            } else {
                l[i * m + j] = s / l[j * m + j];
            }
        }
    }
    true
}

/// Solve `L L' x = b` from a Cholesky factor.
fn cholesky_solve(l: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    let mut x = vec![0.0; m];
    cholesky_solve_into(l, b, &mut x, m);
    x
}

/// [`cholesky_solve`] into `x` (used as the forward-substitution scratch).
fn cholesky_solve_into(l: &[f64], b: &[f64], x: &mut [f64], m: usize) {
    for i in 0..m {
        let s: f64 = (0..i).map(|k| l[i * m + k] * x[k]).sum();
        x[i] = (b[i] - s) / l[i * m + i];
    }
    for i in (0..m).rev() {
        let s: f64 = (i + 1..m).map(|k| l[k * m + i] * x[k]).sum();
        x[i] = (x[i] - s) / l[i * m + i];
    }
}

/// Gaussian elimination with partial pivoting for a small dense system, in
/// place: `a` is destroyed and `b` receives the solution (`false` when singular
/// or non-finite).
fn solve_dense_in_place(a: &mut [f64], b: &mut [f64], m: usize) -> bool {
    for col in 0..m {
        let Some(pivot) = (col..m).max_by(|&i, &j| {
            a[i * m + col]
                .abs()
                .partial_cmp(&a[j * m + col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            return false;
        };
        if a[pivot * m + col].abs() < 1e-300 {
            return false;
        }
        if pivot != col {
            for k in 0..m {
                a.swap(col * m + k, pivot * m + k);
            }
            b.swap(col, pivot);
        }
        for row in col + 1..m {
            let f = a[row * m + col] / a[col * m + col];
            for k in col..m {
                a[row * m + k] -= f * a[col * m + k];
            }
            b[row] -= f * b[col];
        }
    }
    for i in (0..m).rev() {
        let s: f64 = (i + 1..m).map(|k| a[i * m + k] * b[k]).sum();
        b[i] = (b[i] - s) / a[i * m + i];
    }
    b[..m].iter().all(|v| v.is_finite())
}

/// Lagged cross-products of the augmented design `A = [X | e]` (`e` the OLS
/// residual), from which the whitened Gram `Ã'Ã = A'L'LA` of any AR(q ≤ Q)
/// whitener follows without touching the rows again.
///
/// For rows `t ≥ q` the whitener is the filter `ã_t = Σ_{j=0}^{q} ψ_j a_{t−j}`
/// with `ψ = (1, −φ₁, …, −φ_q)`, so their Gram is `Σ_{j,k} ψ_j ψ_k S_jk` with
/// `S_jk = Σ_{t ≥ Q} a_{t−j} a_{t−k}'`. Each `S_jk` is the full-range lag
/// `k − j` product `T_d = Σ_{s ≥ d} a_s a_{s−d}'` minus at most `Q` boundary
/// rows at either end, so only `Q + 1` blocks are accumulated over the rows;
/// the rows `q ≤ t < Q` and the `q` Cholesky-whitened leading rows are added
/// per evaluation from the stored leading rows. The residual `e` stands in for
/// `y`: the GLS residual sum of squares of `e` on `X̃` equals that of `y`
/// (`Xβ̂` lies in the column space), and `ẽ'ẽ − β̃'X̃'ẽ` does not cancel the
/// way `ỹ'ỹ − β̃'X̃'ỹ` does on a well-fitting design.
struct RemlKernel {
    n: usize,
    p: usize,
    /// Augmented width `p + 1`.
    m: usize,
    /// Largest order served (`min(4, n − p − 2)`).
    max_order: usize,
    /// Rows `t < max_order` of `A`, row-major `max_order × m`.
    top: Vec<f64>,
    /// `S_jk` for `0 ≤ j ≤ k ≤ max_order`, each an `m × m` row-major block at
    /// `block(j, k)`.
    lagged: Vec<f64>,
    /// Per-evaluation scratch: the whitened Gram (`m × m`), its `p × p`
    /// leading block, the factor, `X̃'ẽ` and `β̃`.
    scratch: std::cell::RefCell<Vec<f64>>,
}

impl RemlKernel {
    fn new(x: &[f64], residuals: &[f64], n: usize, p: usize) -> Self {
        let m = p + 1;
        let big_q = MAX_AR_ORDER.min((n - p).saturating_sub(2));
        let column = |c: usize| if c < p { &x[c * n..(c + 1) * n] } else { residuals };
        let mut top = vec![0.0; big_q * m];
        for t in 0..big_q {
            for c in 0..m {
                top[t * m + c] = column(c)[t];
            }
        }
        // Full-range lag products T_d[a][b] = Σ_{s ≥ d} a_s[a] a_{s−d}[b].
        let mut lag_products = vec![0.0; (big_q + 1) * m * m];
        for d in 0..=big_q {
            for a in 0..m {
                let ca = &column(a)[d..n];
                for b in 0..m {
                    let cb = &column(b)[..n - d];
                    lag_products[(d * m + a) * m + b] =
                        ca.iter().zip(cb).map(|(u, v)| u * v).sum::<f64>();
                }
            }
        }
        // S_jk = Σ_{t ≥ Q} a_{t−j} a_{t−k}' = T_{k−j} − Σ_{s < Q − j} − Σ_{s ≥ n − j}
        // (both boundary sums over s with s − d ≥ 0, i.e. s ≥ d).
        let blocks = (big_q + 1) * (big_q + 2) / 2;
        let mut lagged = vec![0.0; blocks * m * m];
        for j in 0..=big_q {
            for k in j..=big_q {
                let d = k - j;
                let base = Self::block(j, k, big_q) * m * m;
                lagged[base..base + m * m]
                    .copy_from_slice(&lag_products[d * m * m..(d + 1) * m * m]);
                let boundary: Vec<usize> = (d..big_q - j).chain(n - j..n).collect();
                for s in boundary {
                    for a in 0..m {
                        let u = column(a)[s];
                        for b in 0..m {
                            lagged[base + a * m + b] -= u * column(b)[s - d];
                        }
                    }
                }
            }
        }
        let scratch = std::cell::RefCell::new(vec![0.0; m * m + 2 * p * p + 2 * p]);
        Self { n, p, m, max_order: big_q, top, lagged, scratch }
    }

    /// Index of block `(j, k)`, `j ≤ k ≤ q`, in the packed upper triangle.
    const fn block(j: usize, k: usize, q: usize) -> usize {
        j * (q + 1) - j * (j + 1) / 2 + k
    }

    /// Whitened Gram `Ã'Ã` (row-major `m × m`) of the AR model `w`, into `g`.
    fn whitened_gram(&self, w: &Whitener, g: &mut [f64]) {
        let (m, q, big_q) = (self.m, w.order(), self.max_order);
        let mut psi = [0.0; MAX_AR_ORDER + 1];
        psi[0] = 1.0;
        for (j, f) in w.coefficients().iter().enumerate() {
            psi[j + 1] = -f;
        }
        g.fill(0.0);
        for j in 0..=q {
            for k in j..=q {
                let c = psi[j] * psi[k];
                let base = Self::block(j, k, big_q) * m * m;
                let block = &self.lagged[base..base + m * m];
                for a in 0..m {
                    for b in 0..m {
                        g[a * m + b] += c * block[a * m + b];
                        if j != k {
                            g[a * m + b] += c * block[b * m + a];
                        }
                    }
                }
            }
        }
        // Filter rows q ≤ t < Q from the stored leading rows.
        for t in q..big_q {
            for a in 0..m {
                let ra: f64 = (0..=q).map(|j| psi[j] * self.top[(t - j) * m + a]).sum();
                for b in 0..m {
                    let rb: f64 = (0..=q).map(|j| psi[j] * self.top[(t - j) * m + b]).sum();
                    g[a * m + b] += ra * rb;
                }
            }
        }
        // Cholesky-whitened rows t < q: U = C⁻¹ A_top, G += U'U.
        if q > 0 {
            let mut u = [0.0; MAX_AR_ORDER];
            let mut top = vec![0.0; q * m];
            for c in 0..m {
                for i in 0..q {
                    let s: f64 = (0..i).map(|j| w.chol[i * q + j] * u[j]).sum();
                    u[i] = (self.top[i * m + c] - s) / w.chol[i * q + i];
                    top[i * m + c] = u[i];
                }
            }
            for i in 0..q {
                for a in 0..m {
                    for b in 0..m {
                        g[a * m + b] += top[i * m + a] * top[i * m + b];
                    }
                }
            }
        }
    }
}

/// Profile REML log-likelihood of the AR(q) error model at coordinates `z`:
/// `−½ log|Γ| − ½ log|X̃'X̃| − ½ (n − p) log(RSS̃ / (n − p))` with `X̃ = LX`,
/// `ỹ = Ly` (`−∞` when the model is not positive definite).
fn reml_log_likelihood(z: &[f64], kernel: &RemlKernel) -> f64 {
    let Some(w) = Whitener::from_z(z) else {
        return f64::NEG_INFINITY;
    };
    let (n, p, m) = (kernel.n, kernel.p, kernel.m);
    let mut scratch = kernel.scratch.borrow_mut();
    let (g, rest) = scratch.split_at_mut(m * m);
    let (gxx, rest) = rest.split_at_mut(p * p);
    let (l, rest) = rest.split_at_mut(p * p);
    let (xy, beta) = rest.split_at_mut(p);
    kernel.whitened_gram(&w, g);
    for a in 0..p {
        gxx[a * p..(a + 1) * p].copy_from_slice(&g[a * m..a * m + p]);
        xy[a] = g[a * m + p];
    }
    if !cholesky_into(gxx, l, p) {
        return f64::NEG_INFINITY;
    }
    cholesky_solve_into(l, xy, beta, p);
    let rss = g[p * m + p] - beta.iter().zip(&*xy).map(|(b, v)| b * v).sum::<f64>();
    if rss <= 0.0 || rss.is_nan() {
        return f64::NEG_INFINITY;
    }
    let log_det_g = 2.0 * (0..p).map(|i| l[i * p + i].ln()).sum::<f64>();
    let dof = (n - p) as f64;
    -0.5 * w.log_det - 0.5 * log_det_g - 0.5 * dof * (rss / dof).ln()
}

/// A Nelder–Mead vertex: coordinates (`dim ≤ MAX_AR_ORDER`) and objective.
type Vertex = (ArCoefficients, f64);

/// Nelder–Mead minimization of `f` from `x0` (initial simplex edge `SIMPLEX_STEP`).
fn nelder_mead(f: impl Fn(&[f64]) -> f64, x0: &[f64]) -> (Vec<f64>, f64) {
    let dim = x0.len();
    debug_assert!(dim <= MAX_AR_ORDER);
    let eval = |v: &ArCoefficients| f(&v[..dim]);
    let mut origin = [0.0; MAX_AR_ORDER];
    origin[..dim].copy_from_slice(x0);
    let mut simplex: Vec<Vertex> = Vec::with_capacity(dim + 1);
    simplex.push((origin, eval(&origin)));
    for i in 0..dim {
        let mut v = origin;
        v[i] += SIMPLEX_STEP;
        let fv = eval(&v);
        simplex.push((v, fv));
    }
    let order = |s: &mut Vec<Vertex>| {
        s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    };
    order(&mut simplex);
    for _ in 0..SIMPLEX_ITERATIONS_PER_DIM * dim {
        let f_range = simplex[dim].1 - simplex[0].1;
        let x_range = simplex[1..]
            .iter()
            .flat_map(|(v, _)| {
                v[..dim].iter().zip(&simplex[0].0[..dim]).map(|(a, b)| (a - b).abs())
            })
            .fold(0.0_f64, f64::max);
        if f_range.abs() <= SIMPLEX_F_TOL && x_range <= SIMPLEX_X_TOL {
            break;
        }
        let mut centroid = [0.0; MAX_AR_ORDER];
        for k in 0..dim {
            centroid[k] = simplex[..dim].iter().map(|(v, _)| v[k]).sum::<f64>() / dim as f64;
        }
        let worst = simplex[dim];
        let point = |coef: f64| -> ArCoefficients {
            let mut out = [0.0; MAX_AR_ORDER];
            for k in 0..dim {
                out[k] = centroid[k] + coef * (centroid[k] - worst.0[k]);
            }
            out
        };
        let reflected = point(1.0);
        let fr = eval(&reflected);
        if fr < simplex[0].1 {
            let expanded = point(2.0);
            let fe = eval(&expanded);
            simplex[dim] = if fe < fr { (expanded, fe) } else { (reflected, fr) };
        } else if fr < simplex[dim - 1].1 {
            simplex[dim] = (reflected, fr);
        } else {
            let contracted = if fr < worst.1 { point(0.5) } else { point(-0.5) };
            let fc = eval(&contracted);
            if fc < fr.min(worst.1) {
                simplex[dim] = (contracted, fc);
            } else {
                let best = simplex[0].0;
                for entry in simplex.iter_mut().skip(1) {
                    for (v, b) in entry.0.iter_mut().zip(&best).take(dim) {
                        *v = b + 0.5 * (*v - b);
                    }
                    entry.1 = eval(&entry.0);
                }
            }
        }
        order(&mut simplex);
    }
    let (v, fv) = simplex[0];
    (v[..dim].to_vec(), fv)
}

/// REML AR(q) fit of the residual process at the BIC-selected order
/// (`−2ℓ_R + q ln(n − p)`, `q ≤ 4`), each order warm-started from the previous
/// and polished by [`newton_polish`].
fn reml_autoregression(kernel: &RemlKernel) -> ArErrorModel {
    let dof = (kernel.n - kernel.p) as f64;
    let mut best = ArErrorModel::default();
    let mut best_bic = -2.0 * reml_log_likelihood(&[], kernel);
    let mut prev: Vec<f64> = Vec::new();
    for q in 1..=kernel.max_order {
        let mut start = prev.clone();
        start.push(0.0);
        let (mut z, mut neg_ll) = nelder_mead(|z| -reml_log_likelihood(z, kernel), &start);
        if !neg_ll.is_finite() {
            break;
        }
        newton_polish(&mut z, &mut neg_ll, kernel);
        prev.clone_from(&z);
        let bic = 2.0 * neg_ll + q as f64 * dof.ln();
        if bic < best_bic {
            best_bic = bic;
            best = ArErrorModel { phi: z_to_phi(&z), z };
        }
    }
    best
}

/// Newton polish of a REML optimum found by the simplex.
///
/// The simplex stops on a likelihood range of `SIMPLEX_F_TOL`, which on a
/// likelihood of curvature `~n` leaves `z` uncertain to `~√(2·10⁻¹²/n)`
/// (`10⁻⁷` at `n = 400`) and `κ̂` to the same relative order: any change in
/// the rounding of the likelihood then lands the simplex elsewhere in that
/// basin. Fixed-Hessian Newton steps (the observed information at the simplex
/// point, and a Richardson-extrapolated central-difference gradient whose
/// truncation error is `O(h⁴)`) contract the remaining error by the relative
/// error of the finite-difference Hessian (`~10⁻⁶`) per step, so two steps pin
/// `z`, and with it `κ̂`, to rounding. A step is taken only while it improves
/// the likelihood beyond its own rounding; on a boundary or a non-definite
/// Hessian the simplex point stands.
fn newton_polish(z: &mut Vec<f64>, neg_ll: &mut f64, kernel: &RemlKernel) {
    let q = z.len();
    if q == 0 {
        return;
    }
    let Some(l) = cholesky(&information_matrix(z, kernel), q) else {
        return;
    };
    let ll = |z: &[f64]| reml_log_likelihood(z, kernel);
    let mut shifted = z.clone();
    for _ in 0..NEWTON_POLISH_STEPS {
        let mut gradient = vec![0.0; q];
        for (i, g) in gradient.iter_mut().enumerate() {
            let mut central = |h: f64| {
                shifted.clone_from(z);
                shifted[i] += h;
                let plus = ll(&shifted);
                shifted[i] -= 2.0 * h;
                (plus - ll(&shifted)) / (2.0 * h)
            };
            let coarse = central(DELTA_STEP);
            let fine = central(0.5 * DELTA_STEP);
            *g = (4.0 * fine - coarse) / 3.0;
        }
        if gradient.iter().any(|g| !g.is_finite()) {
            return;
        }
        let step = cholesky_solve(&l, &gradient, q);
        let size = step.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
        if !size.is_finite() || size > SIMPLEX_STEP {
            return;
        }
        for (s, d) in shifted.iter_mut().zip(z.iter().zip(&step)) {
            *s = d.0 + d.1;
        }
        let candidate = -ll(&shifted);
        if !candidate.is_finite()
            || candidate > *neg_ll + NEWTON_POLISH_SLACK * neg_ll.abs().max(1.0)
        {
            return;
        }
        z.clone_from(&shifted);
        *neg_ll = candidate;
        if size <= NEWTON_POLISH_TOL {
            return;
        }
    }
}

/// Cholesky factor of the observed REML information at `z` (central second
/// differences, step `DELTA_STEP`, symmetrized); `None` when not positive definite.
fn observed_information(z: &[f64], kernel: &RemlKernel) -> Option<Vec<f64>> {
    cholesky(&information_matrix(z, kernel), z.len())
}

/// Observed REML information `−∇²ℓ_R` at `z` (central second differences,
/// step `DELTA_STEP`, symmetrized), row-major `q × q`.
fn information_matrix(z: &[f64], kernel: &RemlKernel) -> Vec<f64> {
    let q = z.len();
    let h = DELTA_STEP;
    let ll = |z: &[f64]| reml_log_likelihood(z, kernel);
    let mut info = vec![0.0; q * q];
    for i in 0..q {
        for j in 0..q {
            let mut shifted = z.to_vec();
            let mut eval = |si: f64, sj: f64| {
                shifted.copy_from_slice(z);
                shifted[i] += si;
                shifted[j] += sj;
                ll(&shifted)
            };
            let second = (eval(h, h) - eval(h, -h) - eval(-h, h) + eval(-h, -h)) / (4.0 * h * h);
            info[i * q + j] = -second;
        }
    }
    // Symmetrize the finite-difference Hessian before factoring.
    for i in 0..q {
        for j in 0..i {
            let s = 0.5 * (info[i * q + j] + info[j * q + i]);
            info[i * q + j] = s;
            info[j * q + i] = s;
        }
    }
    info
}

/// AR(q)-prewhitened Bartlett long-run variance of `s` divided by its variance:
/// `u_t = s_t − Σ φ_j s_{t−j}`, recoloured by `1/(1 − Σφ)²` (floored like the
/// AR(1) filter's `1 − ρ̂`).
fn ar_prewhitened_variance_ratio(s: &[f64], phi: &[f64], bandwidth: usize) -> f64 {
    let n = s.len();
    let q = phi.len();
    let gamma0 = dot(s, s) / n as f64;
    if gamma0 <= 0.0 || !gamma0.is_finite() || n <= q + 1 {
        return 1.0;
    }
    let u: Vec<f64> = (q..n)
        .map(|t| s[t] - phi.iter().enumerate().map(|(j, f)| f * s[t - 1 - j]).sum::<f64>())
        .collect();
    let recolour = (1.0 - phi.iter().sum::<f64>()).max(1.0 - MAX_PARTIAL_AUTOCORRELATION);
    bartlett_long_run_variance(&u, bandwidth) / recolour.powi(2) / gamma0
}

/// Standard error of the sample mean of a serially dependent series: `√(γ₀ · r / n)` with
/// `γ₀` the variance and `r` the Bartlett long-run-variance ratio, the larger of the
/// AR(1)-prewhitened and (when BIC selects order 2 or more) the AR(q)-prewhitened
/// readings, floored at 1 so a persistence-blind mean is never narrower than the iid one
/// (the same floor as the tempering factor). `0` for fewer than three finite values or a
/// constant series.
#[must_use]
pub(crate) fn mean_standard_error(series: &[f64]) -> f64 {
    let n = series.len();
    if n < 3 || series.iter().any(|v| !v.is_finite()) {
        return 0.0;
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    let s: Vec<f64> = series.iter().map(|v| v - mean).collect();
    let gamma0 = dot(&s, &s) / n as f64;
    if !gamma0.is_finite() || gamma0 <= 0.0 {
        return 0.0;
    }
    let bandwidth = newey_west_bandwidth(n);
    let mut ratio = long_run_variance_ratio(&s, bandwidth);
    let model = bic_autoregression(&s);
    if model.phi.len() >= 2 {
        ratio = ratio.max(ar_prewhitened_variance_ratio(&s, &model.phi, bandwidth));
    }
    (gamma0 * ratio.max(1.0) / n as f64).sqrt()
}

/// Newey–West rule-of-thumb bandwidth `⌊4 (n/100)^{2/9}⌋`.
fn newey_west_bandwidth(n: usize) -> usize {
    (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor().max(0.0) as usize
}

/// Bartlett (Newey–West) long-run variance of `u` (uncentred, divided by its length).
fn bartlett_long_run_variance(u: &[f64], bandwidth: usize) -> f64 {
    let m = u.len();
    let lags = bandwidth.min(m.saturating_sub(1));
    let autocov = |lag: usize| dot(&u[lag..], &u[..m - lag]);
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
    let gamma0 = dot(s, s) / n as f64;
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

    /// Dense Toeplitz correlation of an AR model, for checking the banded operator.
    fn dense_correlation(phi: &[f64], n: usize) -> Vec<f64> {
        let w = Whitener::new(phi).unwrap();
        let mut r = vec![0.0; n * n];
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let col = w.correlate(&e);
            for i in 0..n {
                r[i * n + j] = col[i];
            }
        }
        r
    }

    /// The standard error of a mean is `√(γ₀ (1 + ρ)/(1 − ρ) / n)` for an AR(1); the
    /// iid reading is a floor, so a white series reads `√(γ₀/n)` up to the estimated ratio.
    #[test]
    fn mean_standard_error_reads_the_long_run_variance_of_the_mean() {
        let (n, rho) = (6000usize, 0.8_f64);
        let mut rng = CausalRng::from_seed(101);
        let persistent = ar1(n, rho, &mut rng);
        let mean = persistent.iter().sum::<f64>() / n as f64;
        let gamma0 = persistent.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let truth = (gamma0 * (1.0 + rho) / (1.0 - rho) / n as f64).sqrt();
        let se = mean_standard_error(&persistent);
        assert!((se / truth - 1.0).abs() < 0.25, "se {se} vs closed form {truth}");
        let white = ar1(n, 0.0, &mut rng);
        let white_mean = white.iter().sum::<f64>() / n as f64;
        let white_gamma0 = white.iter().map(|v| (v - white_mean).powi(2)).sum::<f64>() / n as f64;
        let iid = (white_gamma0 / n as f64).sqrt();
        let white_se = mean_standard_error(&white);
        assert!(white_se >= iid - 1e-15, "never narrower than the iid reading");
        assert!(white_se < 1.25 * iid, "white se {white_se} vs iid {iid}");
        assert_eq!(mean_standard_error(&[1.0, 2.0]), 0.0);
        assert_eq!(mean_standard_error(&[3.0; 40]), 0.0);
        assert_eq!(mean_standard_error(&[1.0, f64::NAN, 2.0, 4.0]), 0.0);
    }

    #[test]
    fn whitener_reproduces_the_autoregressive_correlation() {
        // AR(2)(0.3, 0.5): ρ₁ = 0.6, ρ₂ = 0.68, ρ_k = 0.3 ρ_{k−1} + 0.5 ρ_{k−2}.
        let n = 30;
        let r = dense_correlation(&[0.3, 0.5], n);
        let mut rho = vec![1.0, 0.6, 0.68];
        for k in 3..n {
            rho.push(0.3 * rho[k - 1] + 0.5 * rho[k - 2]);
        }
        for i in 0..n {
            for j in 0..n {
                assert!((r[i * n + j] - rho[i.abs_diff(j)]).abs() < 1e-10, "({i}, {j})");
            }
        }
        // L'L = Γ⁻¹: whitening a correlated draw and colouring it back is the identity.
        let w = Whitener::new(&[0.3, 0.5]).unwrap();
        let v: Vec<f64> = (0..n).map(|t| (0.37 * t as f64).sin()).collect();
        let back = w.solve(&w.apply(&v));
        assert!(v.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-10));
        let identity = Whitener::new(&[]).unwrap();
        assert!(identity.correlate(&v).iter().zip(&v).all(|(a, b)| (a - b).abs() < 1e-12));
    }

    /// Explicit row-wise REML likelihood (whitened rows, dense Gram, explicit residuals).
    fn explicit_reml_log_likelihood(z: &[f64], d: &CompiledDesign) -> f64 {
        let (n, p) = (d.nrows, d.ncols);
        let x = &d.matrix[..n * p];
        let w = Whitener::new(&z_to_phi(z)).unwrap();
        let xw: Vec<Vec<f64>> = (0..p).map(|k| w.apply(&x[k * n..(k + 1) * n])).collect();
        let yw = w.apply(&d.outcome[..n]);
        let mut g = vec![0.0; p * p];
        let mut xy = vec![0.0; p];
        for a in 0..p {
            xy[a] = xw[a].iter().zip(&yw).map(|(u, v)| u * v).sum();
            for b in 0..p {
                g[a * p + b] = xw[a].iter().zip(&xw[b]).map(|(u, v)| u * v).sum();
            }
        }
        let l = cholesky(&g, p).unwrap();
        let beta = cholesky_solve(&l, &xy, p);
        let rss: f64 =
            (0..n).map(|t| (yw[t] - (0..p).map(|k| xw[k][t] * beta[k]).sum::<f64>()).powi(2)).sum();
        let log_det_g = 2.0 * (0..p).map(|i| l[i * p + i].ln()).sum::<f64>();
        let dof = (n - p) as f64;
        -0.5 * w.log_det - 0.5 * log_det_g - 0.5 * dof * (rss / dof).ln()
    }

    #[test]
    fn lagged_kernel_reproduces_the_row_wise_likelihood() {
        for (n, rho, seed) in [(60, 0.7, 1), (233, 0.2, 2), (800, 0.9, 3)] {
            let d = design(n, rho, seed);
            let fit = ResidualFit::new(&d).unwrap();
            let kernel = RemlKernel::new(&d.matrix[..n * d.ncols], &fit.residuals, n, d.ncols);
            for z in [
                vec![],
                vec![0.3],
                vec![-0.8, 0.4],
                vec![0.5, 0.2, -0.1],
                vec![0.9, -0.3, 0.2, 0.4],
                vec![2.0, 1.5, -1.0, 0.7],
            ] {
                let fast = reml_log_likelihood(&z, &kernel);
                let slow = explicit_reml_log_likelihood(&z, &d);
                assert!(
                    (fast - slow).abs() <= 1e-9 * slow.abs().max(1.0),
                    "n={n} z={z:?}: {fast} vs {slow}"
                );
            }
        }
    }

    #[test]
    fn the_design_fit_is_cached_by_content() {
        let d = design(300, 0.7, 21);
        let copy = CompiledDesign::linear_adjustment(
            &d.matrix[300..600],
            &[(VariableId::from_raw(2), &d.matrix[600..900])],
            &d.outcome,
            &[],
        )
        .unwrap();
        let first = residual_fit(&d).unwrap();
        let second = residual_fit(&copy).unwrap();
        assert!(Arc::ptr_eq(&first, &second), "identical rows share one fit");
        let other = residual_fit(&design(300, 0.7, 22)).unwrap();
        assert!(!Arc::ptr_eq(&first, &other));
        let direct = ResidualFit::new(&d).unwrap();
        assert_eq!(direct.model.z, first.model.z);
        assert_eq!(direct.df_loss, first.df_loss);
    }

    /// The simplex alone leaves `z` uncertain to `~√(2 SIMPLEX_F_TOL / n)`; the
    /// Newton polish makes the optimum a function of the design, not of the path
    /// the simplex took to it.
    #[test]
    fn the_polished_reml_optimum_does_not_depend_on_the_simplex_path() {
        for (n, rho, seed) in [(200, 0.7, 11), (400, 0.9, 12), (2000, 0.5, 13)] {
            let d = design(n, rho, seed);
            let fit = ResidualFit::new(&d).unwrap();
            let kernel = RemlKernel::new(&d.matrix[..n * d.ncols], &fit.residuals, n, d.ncols);
            assert!(!fit.model.z.is_empty(), "n={n}: {:?}", fit.model);
            for shift in [0.15, -0.3] {
                let start: Vec<f64> = fit.model.z.iter().map(|z| z + shift).collect();
                let (mut z, mut neg_ll) = nelder_mead(|z| -reml_log_likelihood(z, &kernel), &start);
                let unpolished = z.clone();
                newton_polish(&mut z, &mut neg_ll, &kernel);
                for ((a, b), raw) in z.iter().zip(&fit.model.z).zip(&unpolished) {
                    assert!(
                        (a - b).abs() < 1e-10,
                        "n={n} shift={shift}: polished {z:?} vs fitted {:?} (simplex {raw})",
                        fit.model.z
                    );
                }
            }
        }
    }

    #[test]
    fn iid_rows_centre_on_one() {
        let mut ratios = Vec::new();
        for seed in 0..40 {
            let f = long_run_tempering_factor(&design(400, 0.0, seed), &DependenceScope::Treatment)
                .unwrap();
            assert!(f.kappa >= 1.0);
            assert!(f.fixed_b > 1.0 && f.fixed_b < 1.05, "fixed-b at n = 400: {}", f.fixed_b);
            ratios.push(f.ar_ratio);
            // With no residual dependence the scale loss is the column count.
            assert!(f.residual_ar_order > 0 || (f.df_loss - 3.0).abs() < 1e-9, "{f:?}");
        }
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        assert!((mean - 1.0).abs() < 0.05, "iid variance ratio should centre on 1, got {mean}");
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
        assert!((levels.raw_ratio - slope.raw_ratio.max(intercept.raw_ratio)).abs() < 1e-9);
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
    fn reml_recovers_the_order_and_coefficients() {
        let d = ar2_design(4_000, [0.3, 0.5], 5);
        let model = ResidualFit::new(&d).unwrap().model;
        assert_eq!(model.phi.len(), 2, "{model:?}");
        assert!(
            (model.phi[0] - 0.3).abs() < 0.05 && (model.phi[1] - 0.5).abs() < 0.05,
            "{model:?}"
        );
        // AR(1) and white noise do not trigger higher orders.
        let mut picked_high = 0;
        for seed in 0..40 {
            let d = design(400, 0.7, 100 + seed);
            let f = long_run_tempering_factor(&d, &DependenceScope::Treatment).unwrap();
            picked_high += usize::from(f.residual_ar_order >= 2);
        }
        assert!(picked_high <= 4, "REML picked q >= 2 on {picked_high}/40 AR(1) designs");
    }

    #[test]
    fn ar2_factor_tracks_the_design_conditional_truth() {
        // x and e independent AR(2)(0.3, 0.5): the variance ratio of the slope given the
        // design is w'Rw / w'w with R the AR(2) correlation, and the residual scale loses
        // tr(HR) rows. The estimated factor should centre on that oracle, and the
        // previous AR(1)-only components fall short of it.
        let phi = [0.3, 0.5];
        let (mut kappas, mut oracles) = (Vec::new(), Vec::new());
        for seed in 0..30 {
            let d = ar2_design(200, phi, 300 + seed);
            let f = long_run_tempering_factor(&d, &DependenceScope::Treatment).unwrap();
            assert!(f.residual_ar_order >= 2, "{f:?}");
            assert!(!f.bounded, "the score bound must not bind on AR(2): {f:?}");
            kappas.push(f.kappa);
            let (n, p) = (d.nrows, d.ncols);
            let xtx = gram(&d.matrix, n, p);
            let v = crate::util::solve_spd(&xtx, &[0.0, 1.0], p).unwrap();
            let w: Vec<f64> =
                (0..n).map(|t| (0..p).map(|k| d.matrix[k * n + t] * v[k]).sum::<f64>()).collect();
            let oracle = Whitener::new(&phi).unwrap();
            let rw = oracle.correlate(&w);
            let ratio = w.iter().zip(&rw).map(|(a, b)| a * b).sum::<f64>()
                / w.iter().map(|a| a * a).sum::<f64>();
            let mut loss = 0.0;
            for j in 0..p {
                let rx = oracle.correlate(&d.matrix[j * n..(j + 1) * n]);
                let col: Vec<f64> = (0..p)
                    .map(|a| d.matrix[a * n..(a + 1) * n].iter().zip(&rx).map(|(u, v)| u * v).sum())
                    .collect();
                loss += crate::util::solve_spd(&xtx, &col, p).unwrap()[j];
            }
            oracles.push(ratio * n as f64 / (n as f64 - loss));
        }
        let geo = |v: &[f64]| (v.iter().map(|k| k.ln()).sum::<f64>() / v.len() as f64).exp();
        assert!(
            (geo(&kappas) / geo(&oracles) - 1.0).abs() < 0.12,
            "kappa {} vs design-conditional oracle {}",
            geo(&kappas),
            geo(&oracles)
        );
    }

    #[test]
    fn kappa_uncertainty_shrinks_with_the_series_length() {
        let short =
            long_run_tempering_factor(&ar2_design(80, [0.3, 0.5], 9), &DependenceScope::Treatment)
                .unwrap();
        let long = long_run_tempering_factor(
            &ar2_design(2_000, [0.3, 0.5], 9),
            &DependenceScope::Treatment,
        )
        .unwrap();
        assert!(short.kappa_log_sd > 0.1, "{short:?}");
        assert!(long.kappa_log_sd < 0.1 && long.kappa_log_sd > 0.0, "{long:?}");
        assert!(short.df_loss > 2.0 && short.df_loss < short.nrows as f64, "{short:?}");
    }

    #[test]
    fn regressor_generated_residual_structure_does_not_inflate_kappa() {
        // Period-4 treatment with its second lag omitted: the residual is 3·t_{t-2}, a
        // perfectly periodic series the AR model fits, but its product with the lag-1
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
        assert!(f.kappa < 2.0, "the bounded autoregressive factor must not blow kappa up: {f:?}");
        assert!(f.bounded, "{f:?}");
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
