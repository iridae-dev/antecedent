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
//! cell of a temporal response at one horizon); `κ̂` is then the largest of their
//! ratios.
//!
//! `κ̂ = max(κ̂_HAC · f_b², κ̂_AR)`:
//!
//! - `κ̂_HAC` uses AR(1) prewhitening of the score (with the Kendall small-sample
//!   bias correction of `ρ̂`) followed by a Bartlett (Newey–West) kernel with the
//!   rule-of-thumb bandwidth `M = ⌊4 (n/100)^{2/9}⌋`, then recolouring by
//!   `1/(1 − ρ̂)²` (Andrews & Monahan 1992). It is robust to the form of the
//!   dependence but noisy in short series, and a posterior scaled by a noisy
//!   variance ratio under-covers; `f_b` is the Kiefer–Vogelsang fixed-b factor
//!   of bandwidth `M + 1` ([`crate::temporal_block::fixed_b_scale`]), the same
//!   correction the Frequentist circular-block SEs carry.
//! - `κ̂_AR = w'Γ̂w / (γ̂₀ w'w)` is the variance ratio of `c'β̂` given the realized
//!   design when the residual is AR(1) with the Kendall-corrected `ρ̂` of the OLS
//!   residuals. It is far less noisy and accounts for the realized regressor
//!   path, but assumes AR(1)-type residual dependence.
//!
//! The larger of the two never narrows the robust estimate. In the 1.9
//! calibration (AR(1) residuals, n = 60–400) the prewhitened ratio alone had
//! log-SD 0.2–0.4 and left nominal-90% intervals at 0.848–0.860 for n ≤ 160;
//! the combination put them at 0.877–0.922. `κ̂` is floored at `1` (the
//! correction never narrows the iid posterior) and capped so at least
//! `ncols + 2` effective rows remain.
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
}

impl DependenceScope {
    const fn label(&self) -> &'static str {
        match self {
            Self::Treatment => "treatment",
            Self::Direction(_) => "contrast_gradient",
            Self::Levels(_) => "response_levels",
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
    /// AR(1)-prewhitened Bartlett long-run-variance ratio of the targeted score.
    pub hac_ratio: f64,
    /// Kiefer–Vogelsang fixed-b SE factor of the Bartlett bandwidth
    /// ([`crate::temporal_block::fixed_b_scale`]); `hac_ratio` is scaled by its square.
    pub fixed_b: f64,
    /// AR(1)-residual quadratic-form ratio conditional on the realized design.
    pub ar_ratio: f64,
    /// Bartlett bandwidth on the prewhitened score.
    pub bandwidth: usize,
    /// Design rows.
    pub nrows: usize,
    /// Whether the cap bound `κ̂` (the correction is then incomplete).
    pub capped: bool,
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
             ar_ratio={:.6} n={} n_eff={:.3} bandwidth={} scope={} capped={}",
            self.kappa,
            self.raw_ratio,
            self.hac_ratio,
            self.fixed_b,
            self.ar_ratio,
            self.nrows,
            self.effective_rows(),
            self.bandwidth,
            self.scope,
            self.capped
        ))
    }

    /// Human-readable assumption description.
    #[must_use]
    pub fn description(&self) -> Arc<str> {
        Arc::from(format!(
            "generalized (power) posterior with a serial-dependence correction: the Gaussian \
             likelihood on {} time-ordered rows is tempered by 1/kappa with kappa = {:.4}, the \
             larger of the AR(1)-prewhitened Newey-West long-run-variance ratio of the {} score \
             (bandwidth {}) scaled by the squared fixed-b factor {:.4}, and the AR(1)-residual \
             variance ratio of the same combination given the design; floored at 1{}; \
             effective rows {:.1}; the prior keeps full weight; heteroskedasticity and mean \
             misspecification are not corrected",
            self.nrows,
            self.kappa,
            self.scope,
            self.bandwidth,
            self.fixed_b,
            if self.capped { ", capped" } else { "" },
            self.effective_rows()
        ))
    }
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

/// Estimate the long-run-variance tempering factor for `design` over `scope`.
///
/// Rows must be in time order.
///
/// # Errors
///
/// Rank-deficient design (the OLS residuals are undefined), a missing treatment
/// column for [`DependenceScope::Treatment`], or a direction of the wrong length.
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
        bandwidth,
        nrows: n,
        capped: false,
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
        DependenceScope::Levels(levels) => {
            levels.iter().map(|direction| check_width(direction)).collect::<Result<_, _>>()?
        }
    };
    // A zero or non-finite combination carries no score; it neither tempers nor refuses.
    let directions: Vec<Vec<f64>> = directions
        .into_iter()
        .filter(|c| c.iter().all(|v| v.is_finite()) && c.iter().any(|v| *v != 0.0))
        .collect();
    if n < MIN_ROWS.max(p + 2) || directions.is_empty() {
        return Ok(floor);
    }
    let x = &design.matrix[..n * p];
    let residuals = FaerBackend
        .least_squares(x, n, p, &design.outcome, &mut LeastSquaresWorkspace::default())?
        .residuals;
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
    // The direction with the largest combined ratio drives κ̂; its components are
    // reported alongside it.
    let (mut raw_ratio, mut hac_ratio, mut ar_ratio) = (f64::NAN, f64::NAN, f64::NAN);
    for c in &directions {
        let v = solve_spd(&xtx, c, p)
            .ok_or_else(|| EstimationError::stats_msg("long-run tempering: singular X'X"))?;
        // w_t = x_t'(X'X)⁻¹c: the weight of row t in c'β̂; w_t·ê_t is its influence.
        let weights: Vec<f64> =
            (0..n).map(|t| (0..p).map(|k| x[k * n + t] * v[k]).sum::<f64>()).collect();
        let influence: Vec<f64> = weights.iter().zip(&residuals).map(|(w, e)| w * e).collect();
        let hac = long_run_variance_ratio(&influence, bandwidth);
        let ar = ar1_quadratic_ratio(&weights, &residuals);
        if !hac.is_finite() || !ar.is_finite() {
            continue;
        }
        let combined = (hac * fixed_b * fixed_b).max(ar);
        if raw_ratio.is_nan() || combined > raw_ratio {
            raw_ratio = combined;
            hac_ratio = hac;
            ar_ratio = ar;
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
        capped: raw_ratio > cap,
        ..floor
    })
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

/// Solve `A v = b` for symmetric positive-definite `A` (row-major, `p × p`) by Cholesky.
fn solve_spd(a: &[f64], b: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut sum = a[i * p + j];
            for k in 0..j {
                sum -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if sum <= 0.0 || sum.is_nan() {
                    return None;
                }
                l[i * p + i] = sum.sqrt();
            } else {
                l[i * p + j] = sum / l[j * p + j];
            }
        }
    }
    let mut y = vec![0.0; p];
    for i in 0..p {
        let s: f64 = (0..i).map(|k| l[i * p + k] * y[k]).sum();
        y[i] = (b[i] - s) / l[i * p + i];
    }
    let mut v = vec![0.0; p];
    for i in (0..p).rev() {
        let s: f64 = (i + 1..p).map(|k| l[k * p + i] * v[k]).sum();
        v[i] = (y[i] - s) / l[i * p + i];
    }
    Some(v)
}

/// Newey–West rule-of-thumb bandwidth `⌊4 (n/100)^{2/9}⌋`.
fn newey_west_bandwidth(n: usize) -> usize {
    (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor().max(0.0) as usize
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
    let m = u.len();
    let lags = bandwidth.min(m.saturating_sub(1));
    let autocov = |lag: usize| u[lag..].iter().zip(&u[..m - lag]).map(|(a, b)| a * b).sum::<f64>();
    let mut lrv = autocov(0);
    for lag in 1..=lags {
        let weight = 1.0 - lag as f64 / (lags + 1) as f64;
        lrv += 2.0 * weight * autocov(lag);
    }
    let lrv = (lrv / m as f64).max(0.0) / (1.0 - rho).powi(2);
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
        let zero = long_run_tempering_factor(
            &design(200, 0.9, 1),
            &DependenceScope::Direction(Arc::from([0.0, 0.0, 0.0])),
        )
        .unwrap();
        assert!((zero.kappa - 1.0).abs() < f64::EPSILON);
        assert!(tempering_kappa_from_notes(&[Arc::from("unrelated")]).is_none());
    }
}
