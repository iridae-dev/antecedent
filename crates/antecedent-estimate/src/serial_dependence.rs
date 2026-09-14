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
//!   1.9 calibration designs the bound never binds.
//! - **Exact fits.** A residual sum of squares at most `1e-20` of the outcome's
//!   centred sum of squares keeps `κ̂ = 1`: the residuals are rounding error,
//!   whose apparent autocorrelation would otherwise drive `κ̂` to the cap.
//!
//! The model is calibrated for short-memory dependence an AR(4) captures: in
//! the 1.9 calibration (nominal 90%, 400 replicates, `n = 60–400`) AR(1),
//! ARMA(1,1), MA(2) and AR(2)(0.3, 0.5) treatment-and-residual designs cover
//! 0.87–0.92 for Pulse, Sustained and temporal mediation; the previous
//! kernel-and-AR(1) rule left AR(2) at 0.81 (`n = 60`) and 0.87 (`n = 160`).
//! Long memory, or autocorrelation beyond what an AR(4) captures, is
//! under-corrected.
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

/// Largest absolute partial autocorrelation of the fitted AR(q), and largest
/// absolute AR(1) prewhitening coefficient (keeps the recolouring finite).
const MAX_PARTIAL_AUTOCORRELATION: f64 = 0.97;

/// Fewest rows on which a tempering factor is estimated; shorter designs keep `κ = 1`.
const MIN_ROWS: usize = 8;

/// Residual sum of squares, relative to the outcome's centred sum of squares, at or
/// below which the fit is exact and `κ̂` stays at `1`.
const EXACT_FIT_RELATIVE_SS: f64 = 1e-20;

/// Largest autoregressive order the BIC search considers.
const MAX_AR_ORDER: usize = 4;

/// Largest factor by which the autoregressive factor may exceed the
/// fixed-b-scaled score HAC ratio. On the 1.9 calibration designs the bound
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
    let x = &design.matrix[..n * p];
    let y = &design.outcome[..n];
    let residuals =
        FaerBackend.least_squares(x, n, p, y, &mut LeastSquaresWorkspace::default())?.residuals;
    // An exact fit leaves only rounding error, whose autocorrelation is an artefact of the
    // arithmetic: there is no residual dependence to correct.
    let outcome_mean = y.iter().sum::<f64>() / n as f64;
    let centred_ss: f64 = y.iter().map(|v| (v - outcome_mean).powi(2)).sum();
    let residual_ss: f64 = residuals.iter().map(|e| e * e).sum();
    if residual_ss <= EXACT_FIT_RELATIVE_SS * centred_ss {
        return Ok(floor);
    }
    let xtx = gram(x, n, p);
    // w = X (X'X)⁻¹ c: row influence weights of c'β̂ for every combination.
    let weights: Vec<Vec<f64>> = directions
        .iter()
        .map(|c| {
            let v = crate::util::solve_spd(&xtx, c, p)
                .ok_or_else(|| EstimationError::stats_msg("long-run tempering: singular X'X"))?;
            Ok((0..n).map(|t| (0..p).map(|k| x[k * n + t] * v[k]).sum::<f64>()).collect())
        })
        .collect::<Result<_, EstimationError>>()?;
    let fixed_b = crate::temporal_block::fixed_b_scale(bandwidth + 1, n);
    // REML AR(q) residual model at the BIC order, and its delta-method spread.
    let design_ref = DesignRef { x, y, n, p, xtx: &xtx };
    let model = reml_autoregression(&design_ref);
    let kappa_terms = KappaTerms::at(&model.z, &design_ref, &weights);
    let log_sd = kappa_log_sd(&model, &design_ref, &weights);
    // The combination with the largest factor drives κ̂; its components are reported.
    let mut best: Option<TemperingFactor> = None;
    for (i, w) in weights.iter().enumerate() {
        let influence: Vec<f64> = w.iter().zip(&residuals).map(|(w, e)| w * e).collect();
        let score_ar = bic_autoregression(&influence);
        let mut hac = long_run_variance_ratio(&influence, bandwidth);
        if score_ar.phi.len() >= 2 {
            hac = hac.max(ar_prewhitened_variance_ratio(&influence, &score_ar.phi, bandwidth));
        }
        let ar = kappa_terms.ratios[i];
        let tau = log_sd[i];
        let unbounded = kappa_terms.scale_factor() * ar * (0.5 * tau * tau).exp();
        let bound = AR_QUADRATIC_HAC_BOUND * hac * fixed_b * fixed_b;
        if !unbounded.is_finite() || !bound.is_finite() {
            continue;
        }
        let combined = unbounded.min(bound);
        if best.is_none_or(|b| combined > b.raw_ratio) {
            best = Some(TemperingFactor {
                raw_ratio: combined,
                ar_ratio: ar,
                df_loss: kappa_terms.df_loss,
                kappa_log_sd: tau,
                hac_ratio: hac,
                fixed_b,
                score_ar_order: score_ar.phi.len(),
                residual_ar_order: model.phi.len(),
                bounded: bound < unbounded,
                ..floor
            });
        }
    }
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

/// Column-major design with its Gram matrix.
struct DesignRef<'a> {
    x: &'a [f64],
    y: &'a [f64],
    n: usize,
    p: usize,
    xtx: &'a [f64],
}

/// `X'X` (row-major `p × p`) of a column-major design.
fn gram(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut xtx = vec![0.0; p * p];
    for a in 0..p {
        for b in a..p {
            let dot: f64 =
                x[a * n..(a + 1) * n].iter().zip(&x[b * n..(b + 1) * n]).map(|(u, v)| u * v).sum();
            xtx[a * p + b] = dot;
            xtx[b * p + a] = dot;
        }
    }
    xtx
}

/// REML AR(q) fit of the residual process.
#[derive(Clone, Debug, Default)]
struct ArErrorModel {
    /// Coefficients `φ₁ … φ_q` (empty for `q = 0`).
    phi: Vec<f64>,
    /// Unbounded partial-autocorrelation coordinates `z` (`r = 0.97 tanh z`).
    z: Vec<f64>,
}

/// Partial autocorrelations `r` (`|r_k| < 1`) to AR coefficients (Levinson step-up).
fn pacf_to_phi(r: &[f64]) -> Vec<f64> {
    let mut phi: Vec<f64> = Vec::with_capacity(r.len());
    for &k in r {
        let m = phi.len();
        let mut next: Vec<f64> = (0..m).map(|j| phi[j] - k * phi[m - 1 - j]).collect();
        next.push(k);
        phi = next;
    }
    phi
}

fn z_to_phi(z: &[f64]) -> Vec<f64> {
    let r: Vec<f64> = z.iter().map(|v| MAX_PARTIAL_AUTOCORRELATION * v.tanh()).collect();
    pacf_to_phi(&r)
}

/// Whitening transform `L` of a stationary AR(q) with unit innovations
/// (`Γ⁻¹ = L'L`): rows `t < q` are the inverse Cholesky factor of the
/// stationary `q × q` autocovariance, rows `t ≥ q` the AR filter.
struct Whitener {
    phi: Vec<f64>,
    /// Lower Cholesky factor `C` of the `q × q` stationary autocovariance (row-major).
    chol: Vec<f64>,
    /// `log|Γ| = 2 Σ log diag(C)`.
    log_det: f64,
    /// Marginal variance `γ₀` of the process with unit innovations.
    gamma0: f64,
}

impl Whitener {
    /// `None` when the implied autocovariance is not positive definite.
    fn new(phi: &[f64]) -> Option<Self> {
        let q = phi.len();
        if q == 0 {
            return Some(Self { phi: Vec::new(), chol: Vec::new(), log_det: 0.0, gamma0: 1.0 });
        }
        // Yule–Walker: ρ_k = Σ_j φ_j ρ_{|k−j|}, k = 1..q, with ρ_0 = 1.
        let mut a = vec![0.0; q * q];
        let mut b = vec![0.0; q];
        for k in 1..=q {
            for j in 1..=q {
                let idx = k.abs_diff(j);
                if idx == 0 {
                    b[k - 1] -= phi[j - 1];
                } else {
                    a[(k - 1) * q + idx - 1] += phi[j - 1];
                }
            }
            a[(k - 1) * q + k - 1] -= 1.0;
        }
        let rho = solve_dense(&a, &b, q)?;
        let gamma0 = 1.0 / (1.0 - phi.iter().zip(&rho).map(|(f, r)| f * r).sum::<f64>());
        if !gamma0.is_finite() || gamma0 <= 0.0 {
            return None;
        }
        let mut cov = vec![0.0; q * q];
        for i in 0..q {
            for j in 0..q {
                let lag = i.abs_diff(j);
                cov[i * q + j] = gamma0 * if lag == 0 { 1.0 } else { rho[lag - 1] };
            }
        }
        let chol = cholesky(&cov, q)?;
        let log_det = 2.0 * (0..q).map(|i| chol[i * q + i].ln()).sum::<f64>();
        Some(Self { phi: phi.to_vec(), chol, log_det, gamma0 })
    }

    fn order(&self) -> usize {
        self.phi.len()
    }

    /// `u = L v`.
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
            u[t] = v[t] - self.phi.iter().enumerate().map(|(j, f)| f * v[t - 1 - j]).sum::<f64>();
        }
        u
    }

    /// `v = L⁻¹ u`.
    fn solve(&self, u: &[f64]) -> Vec<f64> {
        let q = self.order();
        let n = u.len();
        let mut v = vec![0.0; n];
        for (i, vi) in v.iter_mut().enumerate().take(q.min(n)) {
            *vi = (0..=i).map(|j| self.chol[i * q + j] * u[j]).sum();
        }
        for t in q..n {
            v[t] = u[t] + self.phi.iter().enumerate().map(|(j, f)| f * v[t - 1 - j]).sum::<f64>();
        }
        v
    }

    /// `z = L'⁻¹ x`.
    fn solve_transpose(&self, x: &[f64]) -> Vec<f64> {
        let q = self.order();
        let n = x.len();
        let mut z = vec![0.0; n];
        let lead = |z: &[f64], t: usize| {
            self.phi
                .iter()
                .enumerate()
                .filter(|(j, _)| t + 1 + j < n && t + 1 + j >= q)
                .map(|(j, f)| f * z[t + 1 + j])
                .sum::<f64>()
        };
        for t in (q..n).rev() {
            z[t] = x[t] + lead(&z, t);
        }
        let rhs: Vec<f64> = (0..q.min(n)).map(|t| x[t] + lead(&z, t)).collect();
        // z_top = C' rhs.
        for (i, zi) in z.iter_mut().enumerate().take(q.min(n)) {
            *zi = (i..q.min(n)).map(|j| self.chol[j * q + i] * rhs[j]).sum();
        }
        z
    }

    /// `R v = Γ v / γ₀` with `R` the correlation matrix of the process.
    fn correlate(&self, v: &[f64]) -> Vec<f64> {
        let mut out = self.solve(&self.solve_transpose(v));
        for o in &mut out {
            *o /= self.gamma0;
        }
        out
    }
}

/// Lower Cholesky factor (row-major) of a symmetric positive-definite `m × m` matrix.
fn cholesky(a: &[f64], m: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..=i {
            let mut s = a[i * m + j];
            for k in 0..j {
                s -= l[i * m + k] * l[j * m + k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i * m + i] = s.sqrt();
            } else {
                l[i * m + j] = s / l[j * m + j];
            }
        }
    }
    Some(l)
}

/// Solve `L L' x = b` from a Cholesky factor.
fn cholesky_solve(l: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    let mut y = vec![0.0; m];
    for i in 0..m {
        let s: f64 = (0..i).map(|k| l[i * m + k] * y[k]).sum();
        y[i] = (b[i] - s) / l[i * m + i];
    }
    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let s: f64 = (i + 1..m).map(|k| l[k * m + i] * x[k]).sum();
        x[i] = (y[i] - s) / l[i * m + i];
    }
    x
}

/// Gaussian elimination with partial pivoting for a small dense system.
fn solve_dense(a: &[f64], b: &[f64], m: usize) -> Option<Vec<f64>> {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    for col in 0..m {
        let pivot = (col..m).max_by(|&i, &j| {
            a[i * m + col]
                .abs()
                .partial_cmp(&a[j * m + col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if a[pivot * m + col].abs() < 1e-300 {
            return None;
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
    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let s: f64 = (i + 1..m).map(|k| a[i * m + k] * x[k]).sum();
        x[i] = (b[i] - s) / a[i * m + i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Profile REML log-likelihood of the AR(q) error model at coordinates `z`:
/// `−½ log|Γ| − ½ log|X̃'X̃| − ½ (n − p) log(RSS̃ / (n − p))` with `X̃ = LX`,
/// `ỹ = Ly` (`−∞` when the model is not positive definite).
fn reml_log_likelihood(z: &[f64], d: &DesignRef<'_>) -> f64 {
    let Some(w) = Whitener::new(&z_to_phi(z)) else {
        return f64::NEG_INFINITY;
    };
    let (n, p) = (d.n, d.p);
    let xw: Vec<Vec<f64>> = (0..p).map(|k| w.apply(&d.x[k * n..(k + 1) * n])).collect();
    let yw = w.apply(d.y);
    let mut g = vec![0.0; p * p];
    let mut xy = vec![0.0; p];
    for a in 0..p {
        xy[a] = xw[a].iter().zip(&yw).map(|(u, v)| u * v).sum();
        for b in a..p {
            let dot: f64 = xw[a].iter().zip(&xw[b]).map(|(u, v)| u * v).sum();
            g[a * p + b] = dot;
            g[b * p + a] = dot;
        }
    }
    let Some(l) = cholesky(&g, p) else {
        return f64::NEG_INFINITY;
    };
    let beta = cholesky_solve(&l, &xy, p);
    let rss: f64 = (0..n)
        .map(|t| {
            let fit: f64 = (0..p).map(|k| xw[k][t] * beta[k]).sum();
            (yw[t] - fit).powi(2)
        })
        .sum();
    if rss <= 0.0 || rss.is_nan() {
        return f64::NEG_INFINITY;
    }
    let log_det_g = 2.0 * (0..p).map(|i| l[i * p + i].ln()).sum::<f64>();
    let dof = (n - p) as f64;
    -0.5 * w.log_det - 0.5 * log_det_g - 0.5 * dof * (rss / dof).ln()
}

/// Nelder–Mead minimization of `f` from `x0` (initial simplex edge `SIMPLEX_STEP`).
fn nelder_mead(f: impl Fn(&[f64]) -> f64, x0: &[f64]) -> (Vec<f64>, f64) {
    let dim = x0.len();
    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(dim + 1);
    simplex.push((x0.to_vec(), f(x0)));
    for i in 0..dim {
        let mut v = x0.to_vec();
        v[i] += SIMPLEX_STEP;
        let fv = f(&v);
        simplex.push((v, fv));
    }
    let order = |s: &mut Vec<(Vec<f64>, f64)>| {
        s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    };
    order(&mut simplex);
    for _ in 0..SIMPLEX_ITERATIONS_PER_DIM * dim {
        let f_range = simplex[dim].1 - simplex[0].1;
        let x_range = simplex[1..]
            .iter()
            .flat_map(|(v, _)| v.iter().zip(&simplex[0].0).map(|(a, b)| (a - b).abs()))
            .fold(0.0_f64, f64::max);
        if f_range.abs() <= SIMPLEX_F_TOL && x_range <= SIMPLEX_X_TOL {
            break;
        }
        let centroid: Vec<f64> = (0..dim)
            .map(|k| simplex[..dim].iter().map(|(v, _)| v[k]).sum::<f64>() / dim as f64)
            .collect();
        let worst = simplex[dim].clone();
        let point = |coef: f64| -> Vec<f64> {
            centroid.iter().zip(&worst.0).map(|(c, w)| c + coef * (c - w)).collect()
        };
        let reflected = point(1.0);
        let fr = f(&reflected);
        if fr < simplex[0].1 {
            let expanded = point(2.0);
            let fe = f(&expanded);
            simplex[dim] = if fe < fr { (expanded, fe) } else { (reflected, fr) };
        } else if fr < simplex[dim - 1].1 {
            simplex[dim] = (reflected, fr);
        } else {
            let (contracted, fc) = if fr < worst.1 {
                let c = point(0.5);
                let fc = f(&c);
                (c, fc)
            } else {
                let c = point(-0.5);
                let fc = f(&c);
                (c, fc)
            };
            if fc < fr.min(worst.1) {
                simplex[dim] = (contracted, fc);
            } else {
                let best = simplex[0].0.clone();
                for entry in simplex.iter_mut().skip(1) {
                    let v: Vec<f64> =
                        entry.0.iter().zip(&best).map(|(a, b)| b + 0.5 * (a - b)).collect();
                    entry.1 = f(&v);
                    entry.0 = v;
                }
            }
        }
        order(&mut simplex);
    }
    simplex.swap_remove(0)
}

/// REML AR(q) fit of the residual process at the BIC-selected order
/// (`−2ℓ_R + q ln(n − p)`, `q ≤ 4`), each order warm-started from the previous.
fn reml_autoregression(d: &DesignRef<'_>) -> ArErrorModel {
    let dof = (d.n - d.p) as f64;
    let max_order = MAX_AR_ORDER.min((d.n - d.p).saturating_sub(2));
    let mut best = ArErrorModel::default();
    let mut best_bic = -2.0 * reml_log_likelihood(&[], d);
    let mut prev: Vec<f64> = Vec::new();
    for q in 1..=max_order {
        let mut start = prev.clone();
        start.push(0.0);
        let (z, neg_ll) = nelder_mead(|z| -reml_log_likelihood(z, d), &start);
        if !neg_ll.is_finite() {
            break;
        }
        prev.clone_from(&z);
        let bic = 2.0 * neg_ll + q as f64 * dof.ln();
        if bic < best_bic {
            best_bic = bic;
            best = ArErrorModel { phi: z_to_phi(&z), z };
        }
    }
    best
}

/// Variance ratios of every combination and the projection scale loss at one
/// AR model.
struct KappaTerms {
    /// `w'Rw / w'w` per combination.
    ratios: Vec<f64>,
    /// `tr(H R)`.
    df_loss: f64,
    n: usize,
    p: usize,
}

impl KappaTerms {
    fn at(z: &[f64], d: &DesignRef<'_>, weights: &[Vec<f64>]) -> Self {
        let (n, p) = (d.n, d.p);
        let identity = |weights: &[Vec<f64>]| Self {
            ratios: vec![1.0; weights.len()],
            df_loss: p as f64,
            n,
            p,
        };
        let Some(w) = Whitener::new(&z_to_phi(z)) else {
            return identity(weights);
        };
        let ratios = weights
            .iter()
            .map(|wt| {
                let rw = w.correlate(wt);
                let norm: f64 = wt.iter().map(|v| v * v).sum();
                let quad: f64 = wt.iter().zip(&rw).map(|(a, b)| a * b).sum();
                if norm > 0.0 { quad / norm } else { 1.0 }
            })
            .collect();
        // tr((X'X)⁻¹ X'RX) = Σ_j [(X'X)⁻¹ (X'R x_j)]_j.
        let mut df_loss = 0.0;
        for j in 0..p {
            let rx = w.correlate(&d.x[j * n..(j + 1) * n]);
            let col: Vec<f64> = (0..p)
                .map(|a| d.x[a * n..(a + 1) * n].iter().zip(&rx).map(|(u, v)| u * v).sum())
                .collect();
            match crate::util::solve_spd(d.xtx, &col, p) {
                Some(v) => df_loss += v[j],
                None => return identity(weights),
            }
        }
        Self { ratios, df_loss, n, p }
    }

    /// Residual-scale factor `n / (n − tr(HR))`, denominator floored at `p + 2`.
    fn scale_factor(&self) -> f64 {
        self.n as f64 / (self.n as f64 - self.df_loss).max((self.p + 2) as f64)
    }

    fn log_kappa(&self, i: usize) -> f64 {
        (self.ratios[i] * self.scale_factor()).ln()
    }
}

/// Delta-method standard deviation of `log κ̂` per combination: gradient of
/// `log κ` in the AR coordinates against the observed REML information
/// (central differences, step `DELTA_STEP`). Zero for `q = 0` or when the
/// information is not positive definite; capped at `MAX_KAPPA_LOG_SD`.
fn kappa_log_sd(model: &ArErrorModel, d: &DesignRef<'_>, weights: &[Vec<f64>]) -> Vec<f64> {
    let q = model.z.len();
    let m = weights.len();
    if q == 0 {
        return vec![0.0; m];
    }
    let h = DELTA_STEP;
    let shifted = |i: usize, s: f64| {
        let mut z = model.z.clone();
        z[i] += s;
        z
    };
    let mut gradients = vec![vec![0.0; q]; m];
    for i in 0..q {
        let plus = KappaTerms::at(&shifted(i, h), d, weights);
        let minus = KappaTerms::at(&shifted(i, -h), d, weights);
        for (k, g) in gradients.iter_mut().enumerate() {
            g[i] = (plus.log_kappa(k) - minus.log_kappa(k)) / (2.0 * h);
        }
    }
    let ll = |z: &[f64]| reml_log_likelihood(z, d);
    let mut info = vec![0.0; q * q];
    for i in 0..q {
        for j in 0..q {
            let mut z = model.z.clone();
            let mut eval = |si: f64, sj: f64| {
                z.clone_from(&model.z);
                z[i] += si;
                z[j] += sj;
                ll(&z)
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
    let Some(l) = cholesky(&info, q) else {
        return vec![0.0; m];
    };
    gradients
        .iter()
        .map(|g| {
            let v = cholesky_solve(&l, g, q);
            let var: f64 = g.iter().zip(&v).map(|(a, b)| a * b).sum();
            if var.is_finite() && var > 0.0 { var.sqrt().min(MAX_KAPPA_LOG_SD) } else { 0.0 }
        })
        .collect()
}

/// Yule–Walker autoregression of a series, at a BIC-selected order (the
/// prewhitening filter of the bounding score HAC ratio).
#[derive(Clone, Debug, Default)]
struct Autoregression {
    /// Coefficients `φ₁ … φ_q` (empty for `q = 0`).
    phi: Vec<f64>,
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
    Autoregression { phi: best }
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
    let recolour = (1.0 - phi.iter().sum::<f64>()).max(1.0 - MAX_PARTIAL_AUTOCORRELATION);
    bartlett_long_run_variance(&u, bandwidth) / recolour.powi(2) / gamma0
}

/// Bias-corrected lag-1 autocorrelation of `s` (Kendall 1954:
/// `E[ρ̂] ≈ ρ − (1 + 3ρ)/n`), clamped to `±MAX_PARTIAL_AUTOCORRELATION`.
fn kendall_rho(s: &[f64]) -> f64 {
    let (num, den) =
        s.windows(2).fold((0.0, 0.0), |(num, den), w| (num + w[1] * w[0], den + w[0] * w[0]));
    let rho_hat = if den > 0.0 { num / den } else { 0.0 };
    (rho_hat + (1.0 + 3.0 * rho_hat) / s.len() as f64)
        .clamp(-MAX_PARTIAL_AUTOCORRELATION, MAX_PARTIAL_AUTOCORRELATION)
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
        let xtx = gram(&d.matrix, d.nrows, d.ncols);
        let model = reml_autoregression(&DesignRef {
            x: &d.matrix,
            y: &d.outcome,
            n: d.nrows,
            p: d.ncols,
            xtx: &xtx,
        });
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
