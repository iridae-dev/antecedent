//! External prior bank: power-prior / mixture composition.
//!
//! Heterogeneous sources are composed into a single [`PriorSet`] usable as a
//! Bayesian coefficient prior. Priors never upgrade nonparametric identification.
//!
//! ## Three distinct effective-sample-size conventions
//!
//! [`ComposedPrior`] reports **prior-strength ESS**: the sample size implied by
//! how much precision a source (or the composed prior) contributes, via
//! `α · ess`. This is *not* interchangeable with:
//!
//! * MCMC / autocorrelation ESS (`crate::mcmc_stats`, `InferenceDiagnostics`) —
//!   how many effectively independent draws a chain produced.
//! * Kish importance-weighting ESS ([`kish_ess`]) — how concentrated an
//!   importance/trust weight vector is, `(Σw)² / Σw²`.
//!
//! Conflating any of the three misreports how much evidence a posterior or
//! prior actually carries; keep them in separate fields with separate names.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::PriorAssumption;

use crate::error::ProbError;
use crate::prior::{GaussianCoefficientPrior, PriorSet, PriorSpec};

/// Floor on conjugate-scale variance after composition.
const COMPOSE_VAR_FLOOR: f64 = 1e-12;

/// Kish (1965) effective sample size: `(Σw)² / Σw²`.
///
/// A concentration-of-trust diagnostic over an importance/trust weight
/// vector — distinct from MCMC/autocorrelation ESS and from the
/// **prior-strength ESS** reported on [`ComposedPrior`] (precision-based, not
/// weight-based; see the module docs). Guards `Σw² > 0`, returning `0.0`
/// otherwise (e.g. all-zero weights).
#[must_use]
pub fn kish_ess(weights: &[f64]) -> f64 {
    let sum: f64 = weights.iter().sum();
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    if sum_sq > 0.0 { (sum * sum) / sum_sq } else { 0.0 }
}

/// Per-source trust knobs for external prior composition.
///
/// `alpha` is the power-prior exponent (precision scale on the Gaussian approx).
/// `mixture_weight` is optional; when any source sets it, all must, with
/// `Σ w_k ≤ 1` and leftover mass on the baseline prior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExternalPriorWeight {
    /// Power-prior exponent in `[0, 1]`.
    pub alpha: f64,
    /// Optional mixture weight; `None` selects the pure precision-add path.
    pub mixture_weight: Option<f64>,
}

impl ExternalPriorWeight {
    /// Construct with validation.
    ///
    /// # Errors
    ///
    /// `alpha` outside `[0, 1]`, non-finite values, or negative mixture weight.
    pub fn new(alpha: f64, mixture_weight: Option<f64>) -> Result<Self, ProbError> {
        let w = Self { alpha, mixture_weight };
        w.validate()?;
        Ok(w)
    }

    /// Power-prior only (`mixture_weight = None`).
    ///
    /// # Errors
    ///
    /// Invalid `alpha`.
    pub fn power(alpha: f64) -> Result<Self, ProbError> {
        Self::new(alpha, None)
    }

    /// Power-prior with an explicit mixture weight.
    ///
    /// # Errors
    ///
    /// Invalid `alpha` or mixture weight.
    pub fn power_mixture(alpha: f64, mixture_weight: f64) -> Result<Self, ProbError> {
        Self::new(alpha, Some(mixture_weight))
    }

    /// Validate finite `alpha ∈ [0, 1]` and non-negative finite mixture weight.
    ///
    /// # Errors
    ///
    /// Invalid parameters.
    pub fn validate(self) -> Result<(), ProbError> {
        if !self.alpha.is_finite() || !(0.0..=1.0).contains(&self.alpha) {
            return Err(ProbError::InvalidPrior {
                message: "external prior alpha must be finite and in [0, 1]",
            });
        }
        if let Some(w) = self.mixture_weight {
            if !w.is_finite() || w < 0.0 {
                return Err(ProbError::InvalidPrior {
                    message: "mixture weight must be finite and >= 0",
                });
            }
        }
        Ok(())
    }
}

/// One hydrated external source plus trust weights.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalPriorSource {
    /// Caller-stable artifact / catalog id.
    pub id: Arc<str>,
    /// Already-mapped coefficient prior (e.g. from `hydrate_prior`).
    pub prior: PriorSet,
    /// Power / mixture weights for this source.
    pub weight: ExternalPriorWeight,
    /// Caller-declared **prior-strength ESS** for this source (e.g. the
    /// original study's sample size, or an effective N after a design
    /// discount). `None` when the caller has no such figure. Distinct from
    /// MCMC ESS and from Kish importance-weighting ESS — see the module docs.
    pub ess: Option<f64>,
}

impl ExternalPriorSource {
    /// Validate the trust weight and optional prior-strength `ess`.
    ///
    /// # Errors
    ///
    /// Invalid `weight`, or `ess` that is non-finite or negative.
    pub fn validate(&self) -> Result<(), ProbError> {
        self.weight.validate()?;
        if let Some(ess) = self.ess {
            if !ess.is_finite() || ess < 0.0 {
                return Err(ProbError::InvalidPrior {
                    message: "external prior source ess must be finite and >= 0",
                });
            }
        }
        Ok(())
    }
}

/// Result of composing external sources with a baseline prior.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposedPrior {
    /// Composed coefficient prior (usable as `BayesianConfig::prior`).
    pub prior: PriorSet,
    /// Source ids in composition order.
    pub source_ids: Arc<[Arc<str>]>,
    /// Caller-requested alphas (before conflict shrink).
    pub alphas_requested: Arc<[f64]>,
    /// Alphas actually used in composition.
    pub alphas_applied: Arc<[f64]>,
    /// Mixture weights (mirrors inputs; `None` entries mean power-only path).
    pub mixture_weights: Arc<[Option<f64>]>,
    /// Per-source **prior-strength ESS** after α discount: `α_applied · ess`
    /// (`None` when that source declared no `ess`). Forced to `0.0` for
    /// sources dropped from composition (`α == 0` on the power path; `α <= 0`
    /// or the mixture weight `<= 0` on the mixture path) even when a nonzero
    /// `ess` was declared, so this always agrees with the arithmetic that
    /// actually ran.
    pub effective_ess: Arc<[Option<f64>]>,
    /// Composed **prior-strength ESS** — path-dependent, unlike
    /// `effective_ess` above:
    ///
    /// * **Power path**: precision genuinely adds (`Λ = Λ₀ + Σ αₖΛₖ`), so
    ///   `Σ αₖ · essₖ` over contributing sources (`α > 0`) is sound. Reported
    ///   as `Some` only when *every* contributing source declared an `ess`;
    ///   otherwise `None` — a partial sum would misrepresent it as complete.
    /// * **Mixture path**: always `None`. The result is moment-matched and
    ///   its variance includes between-component spread (`second − μ²`), so
    ///   the composed prior is *weaker* than a precision-sum would imply;
    ///   summing source ESS here would overstate composed strength.
    pub composed_ess: Option<f64>,
    /// Kish concentration-of-trust diagnostic ([`kish_ess`]) over the weight
    /// vector actually used in composition: **applied alphas** on the power
    /// path, **mixture weights** on the mixture path — both with dropped
    /// sources zeroed, matching the arithmetic that ran. `None` only when
    /// there are no sources to diagnose. An importance-weighting diagnostic,
    /// distinct from the prior-strength ESS fields above and from MCMC ESS.
    pub kish_ess: Option<f64>,
}

impl ComposedPrior {
    /// Borrow the composed [`PriorSet`].
    #[must_use]
    pub fn as_prior_set(&self) -> &PriorSet {
        &self.prior
    }

    /// Consume into the composed [`PriorSet`].
    #[must_use]
    pub fn into_prior_set(self) -> PriorSet {
        self.prior
    }
}

/// Compose external Gaussian coefficient priors with a baseline.
///
/// * **Power path** (all `mixture_weight` are `None`):  
///   `Λ = Λ₀ + Σ α_k Λ_k`, mean from precision-weighted average.
/// * **Mixture path** (all weights set): moment-match  
///   `Σ w_k N(μ_k, V_k/α_k) + (1−Σw) · baseline`. Sources with `α_k = 0` are
///   dropped and their weight folds into leftover baseline mass.
///
/// Uses `weight.alpha` as both requested and applied. For conflict shrink,
/// mutate source alphas (or call [`compose_external_priors_with_alphas`]) before
/// composing.
///
/// # Errors
///
/// Invalid weights, missing Gaussians, or dimension mismatch.
pub fn compose_external_priors(
    sources: &[ExternalPriorSource],
    baseline: &PriorSet,
) -> Result<ComposedPrior, ProbError> {
    let alphas: Vec<f64> = sources.iter().map(|s| s.weight.alpha).collect();
    compose_external_priors_with_alphas(sources, &alphas, &alphas, baseline)
}

/// Compose with explicit requested / applied alpha vectors (conflict path).
///
/// `alphas_applied[k]` overrides `sources[k].weight.alpha` for the math while
/// preserving mixture-weight mode selection from the source weights.
///
/// # Errors
///
/// Length mismatch, invalid weights, missing Gaussians, or dimension mismatch.
pub fn compose_external_priors_with_alphas(
    sources: &[ExternalPriorSource],
    alphas_requested: &[f64],
    alphas_applied: &[f64],
    baseline: &PriorSet,
) -> Result<ComposedPrior, ProbError> {
    if sources.len() != alphas_requested.len() || sources.len() != alphas_applied.len() {
        return Err(ProbError::Shape {
            message: "compose_external_priors: alpha vector length mismatch",
        });
    }
    for &a in alphas_requested.iter().chain(alphas_applied.iter()) {
        if !a.is_finite() || !(0.0..=1.0).contains(&a) {
            return Err(ProbError::InvalidPrior {
                message: "external prior alpha must be finite and in [0, 1]",
            });
        }
    }
    for src in sources {
        src.validate()?;
    }
    validate_mixture_weights(sources)?;

    let base_coef = baseline.gaussian_coefficients().ok_or(ProbError::InvalidPrior {
        message: "baseline prior missing GaussianCoefficients",
    })?;
    base_coef.validate()?;
    let n = base_coef.len();

    for src in sources {
        let coef = src.prior.gaussian_coefficients().ok_or(ProbError::InvalidPrior {
            message: "external source prior missing GaussianCoefficients",
        })?;
        coef.validate()?;
        if coef.len() != n {
            return Err(ProbError::Shape {
                message: "compose_external_priors: coefficient dimension mismatch",
            });
        }
    }

    let use_mixture = sources.iter().any(|s| s.weight.mixture_weight.is_some());
    let composed_coef = if use_mixture {
        compose_mixture(base_coef, sources, alphas_applied)?
    } else {
        compose_power_add(base_coef, sources, alphas_applied)?
    };

    let mut prior = PriorSet {
        specs: Vec::new(),
        contrast: baseline.contrast,
        categorical: baseline.categorical.clone(),
        restrictions: Vec::new(),
    };
    prior.push(PriorSpec::GaussianCoefficients(composed_coef));
    if let Some(ig) = baseline.residual_inv_gamma() {
        prior.push(PriorSpec::ResidualInvGamma(ig));
    } else if let Some(v) = baseline.known_residual_variance() {
        prior.push(PriorSpec::KnownResidualVariance(v));
    }
    for r in &baseline.restrictions {
        prior.restrictions.push(r.clone());
    }
    for src in sources {
        for r in &src.prior.restrictions {
            prior.restrictions.push(r.clone());
        }
    }
    prior.restrictions.push(composition_assumption(sources, alphas_requested, alphas_applied));
    prior.validate()?;

    let source_ids: Vec<Arc<str>> = sources.iter().map(|s| Arc::clone(&s.id)).collect();
    let mixture_weights: Vec<Option<f64>> =
        sources.iter().map(|s| s.weight.mixture_weight).collect();
    let effective_ess = effective_ess_per_source(sources, alphas_applied, use_mixture);
    let composed_ess = if use_mixture { None } else { power_composed_ess(sources, alphas_applied) };
    let kish_ess_diag = if sources.is_empty() {
        None
    } else {
        Some(kish_ess(&kish_weights_for_composition(sources, alphas_applied, use_mixture)))
    };

    Ok(ComposedPrior {
        prior,
        source_ids: Arc::from(source_ids),
        alphas_requested: Arc::from(alphas_requested.to_vec()),
        alphas_applied: Arc::from(alphas_applied.to_vec()),
        mixture_weights: Arc::from(mixture_weights),
        effective_ess: Arc::from(effective_ess),
        composed_ess,
        kish_ess: kish_ess_diag,
    })
}

/// Per-source **prior-strength ESS** after α discount (`α_applied · ess`);
/// `None` when a source declared no `ess`. Zeroed for sources dropped from
/// composition so this always agrees with the arithmetic that actually ran —
/// see [`ComposedPrior::effective_ess`].
fn effective_ess_per_source(
    sources: &[ExternalPriorSource],
    alphas_applied: &[f64],
    use_mixture: bool,
) -> Vec<Option<f64>> {
    sources
        .iter()
        .zip(alphas_applied.iter())
        .map(|(src, &alpha)| {
            let dropped = if use_mixture {
                alpha <= 0.0 || src.weight.mixture_weight.unwrap_or(0.0) <= 0.0
            } else {
                alpha == 0.0
            };
            src.ess.map(|ess| if dropped { 0.0 } else { alpha * ess })
        })
        .collect()
}

/// `Σ αₖ · essₖ` over power-path sources that actually contribute (`α > 0`);
/// `None` unless every contributing source declared an `ess` — see
/// [`ComposedPrior::composed_ess`].
fn power_composed_ess(sources: &[ExternalPriorSource], alphas_applied: &[f64]) -> Option<f64> {
    let mut total = 0.0;
    for (src, &alpha) in sources.iter().zip(alphas_applied.iter()) {
        if alpha == 0.0 {
            continue;
        }
        total += alpha * src.ess?;
    }
    Some(total)
}

/// Weight vector actually used in composition, for the [`kish_ess`]
/// concentration diagnostic: applied alphas on the power path, mixture
/// weights (zeroed for `α <= 0`) on the mixture path.
fn kish_weights_for_composition(
    sources: &[ExternalPriorSource],
    alphas_applied: &[f64],
    use_mixture: bool,
) -> Vec<f64> {
    if use_mixture {
        sources
            .iter()
            .zip(alphas_applied.iter())
            .map(
                |(src, &alpha)| {
                    if alpha <= 0.0 { 0.0 } else { src.weight.mixture_weight.unwrap_or(0.0) }
                },
            )
            .collect()
    } else {
        alphas_applied.to_vec()
    }
}

fn validate_mixture_weights(sources: &[ExternalPriorSource]) -> Result<(), ProbError> {
    if sources.is_empty() {
        return Ok(());
    }
    let any = sources.iter().any(|s| s.weight.mixture_weight.is_some());
    let all = sources.iter().all(|s| s.weight.mixture_weight.is_some());
    if any && !all {
        return Err(ProbError::InvalidPrior {
            message: "mixture weights must be set on all sources or none",
        });
    }
    if !any {
        return Ok(());
    }
    let sum: f64 = sources.iter().map(|s| s.weight.mixture_weight.unwrap_or(0.0)).sum();
    if !sum.is_finite() || sum > 1.0 + 1e-12 {
        return Err(ProbError::InvalidPrior { message: "sum of mixture weights must be <= 1" });
    }
    Ok(())
}

fn compose_power_add(
    baseline: &GaussianCoefficientPrior,
    sources: &[ExternalPriorSource],
    alphas: &[f64],
) -> Result<GaussianCoefficientPrior, ProbError> {
    let n = baseline.len();
    let mut lam = baseline.precision();
    let mut num = vec![0.0; n];
    for i in 0..n {
        num[i] = lam[i] * baseline.mean[i];
    }
    for (src, &alpha) in sources.iter().zip(alphas.iter()) {
        if alpha == 0.0 {
            continue;
        }
        let coef = src.prior.gaussian_coefficients().expect("validated");
        let prec = coef.precision();
        for i in 0..n {
            let a_lam = alpha * prec[i];
            lam[i] += a_lam;
            num[i] += a_lam * coef.mean[i];
        }
    }
    let mut mean = vec![0.0; n];
    let mut variance = vec![0.0; n];
    for i in 0..n {
        if !(lam[i] > 0.0) || !lam[i].is_finite() {
            return Err(ProbError::Numerical {
                message: "compose_external_priors: non-positive composed precision".into(),
            });
        }
        mean[i] = num[i] / lam[i];
        variance[i] = (1.0 / lam[i]).max(COMPOSE_VAR_FLOOR);
    }
    let out = GaussianCoefficientPrior { mean: Arc::from(mean), variance: Arc::from(variance) };
    out.validate()?;
    Ok(out)
}

fn compose_mixture(
    baseline: &GaussianCoefficientPrior,
    sources: &[ExternalPriorSource],
    alphas: &[f64],
) -> Result<GaussianCoefficientPrior, ProbError> {
    let n = baseline.len();
    // (weight, mean_i, var_i) accumulated per active component; leftover on baseline.
    let mut active_w = 0.0;
    let mut comps: Vec<(f64, &GaussianCoefficientPrior, f64)> = Vec::new();
    for (src, &alpha) in sources.iter().zip(alphas.iter()) {
        let w = src.weight.mixture_weight.unwrap_or(0.0);
        if alpha <= 0.0 || w <= 0.0 {
            // Dropped mass folds into leftover baseline.
            continue;
        }
        let coef = src.prior.gaussian_coefficients().expect("validated");
        comps.push((w, coef, alpha));
        active_w += w;
    }
    let leftover = (1.0 - active_w).max(0.0);
    if leftover > 0.0 {
        comps.push((leftover, baseline, 1.0));
    }
    if comps.is_empty() {
        return Err(ProbError::InvalidPrior {
            message: "compose_external_priors: mixture has no positive-mass components",
        });
    }

    let mut mean = vec![0.0; n];
    let mut variance = vec![0.0; n];
    for i in 0..n {
        let mut mu = 0.0;
        let mut second = 0.0;
        for &(w, coef, alpha) in &comps {
            let m = coef.mean[i];
            // Power-scale: precision α Λ ⇒ variance V/α.
            let v = (coef.variance[i] / alpha).max(COMPOSE_VAR_FLOOR);
            mu += w * m;
            second += w * (v + m * m);
        }
        mean[i] = mu;
        variance[i] = (second - mu * mu).max(COMPOSE_VAR_FLOOR);
    }
    let out = GaussianCoefficientPrior { mean: Arc::from(mean), variance: Arc::from(variance) };
    out.validate()?;
    Ok(out)
}

fn composition_assumption(
    sources: &[ExternalPriorSource],
    alphas_requested: &[f64],
    alphas_applied: &[f64],
) -> PriorAssumption {
    let mut parts = Vec::with_capacity(sources.len());
    for (i, src) in sources.iter().enumerate() {
        let w = src.weight.mixture_weight.map_or_else(|| "none".to_string(), |x| format!("{x}"));
        parts.push(format!(
            "{}:alpha_req={},alpha_app={},w={}",
            src.id, alphas_requested[i], alphas_applied[i], w
        ));
    }
    PriorAssumption {
        id: Arc::from("external_composed_prior"),
        description: Arc::from(format!(
            "External power-prior / mixture composition [{}]",
            parts.join("; ")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prior::GaussianCoefficientPrior;

    fn gauss(mean: f64, var: f64) -> PriorSet {
        let mut p = PriorSet::new();
        p.push(PriorSpec::GaussianCoefficients(
            GaussianCoefficientPrior::shared(1, mean, var).unwrap(),
        ));
        p
    }

    #[test]
    fn rejects_alpha_out_of_range() {
        assert!(ExternalPriorWeight::power(-0.1).is_err());
        assert!(ExternalPriorWeight::power(1.1).is_err());
        assert!(ExternalPriorWeight::power(f64::NAN).is_err());
    }

    #[test]
    fn rejects_mixture_weight_sum_gt_one() {
        let baseline = PriorSet::weakly_informative(1);
        let sources = [
            ExternalPriorSource {
                id: Arc::from("a"),
                prior: gauss(1.0, 1.0),
                weight: ExternalPriorWeight::power_mixture(1.0, 0.7).unwrap(),
                ess: None,
            },
            ExternalPriorSource {
                id: Arc::from("b"),
                prior: gauss(2.0, 1.0),
                weight: ExternalPriorWeight::power_mixture(1.0, 0.5).unwrap(),
                ess: None,
            },
        ];
        let err = compose_external_priors(&sources, &baseline).unwrap_err();
        assert!(matches!(err, ProbError::InvalidPrior { .. }));
    }

    #[test]
    fn power_prior_precision_add_analytic() {
        // baseline: mean 0, V0=4 ⇒ Λ0=0.25
        // old: mean 2, V=1 ⇒ Λ=1; α=0.5 ⇒ αΛ=0.5
        // composed Λ=0.75, μ=(0 + 0.5*2)/0.75 = 4/3, V=4/3
        let baseline = gauss(0.0, 4.0);
        let sources = [ExternalPriorSource {
            id: Arc::from("old"),
            prior: gauss(2.0, 1.0),
            weight: ExternalPriorWeight::power(0.5).unwrap(),
            ess: None,
        }];
        let composed = compose_external_priors(&sources, &baseline).unwrap();
        let coef = composed.prior.gaussian_coefficients().unwrap();
        let lam = 1.0 / coef.variance[0];
        assert!((lam - 0.75).abs() < 1e-12);
        assert!((coef.mean[0] - (4.0 / 3.0)).abs() < 1e-12);
        assert!(composed.prior.restrictions.iter().any(|r| &*r.id == "external_composed_prior"));
    }

    #[test]
    fn mixture_preserves_leftover_baseline_mass() {
        // w=0.4 on source, leftover 0.6 on baseline mean 0 var 100
        // source mean 10, var 1, α=1
        let baseline = gauss(0.0, 100.0);
        let sources = [ExternalPriorSource {
            id: Arc::from("s"),
            prior: gauss(10.0, 1.0),
            weight: ExternalPriorWeight::power_mixture(1.0, 0.4).unwrap(),
            ess: None,
        }];
        let composed = compose_external_priors(&sources, &baseline).unwrap();
        let coef = composed.prior.gaussian_coefficients().unwrap();
        // μ = 0.4*10 + 0.6*0 = 4
        assert!((coef.mean[0] - 4.0).abs() < 1e-10);
        // second = 0.4*(1+100) + 0.6*(100+0) = 40.4 + 60 = 100.4
        // var = 100.4 - 16 = 84.4
        assert!((coef.variance[0] - 84.4).abs() < 1e-10);
    }

    #[test]
    fn applied_alpha_override() {
        let baseline = gauss(0.0, 4.0);
        let sources = [ExternalPriorSource {
            id: Arc::from("old"),
            prior: gauss(2.0, 1.0),
            weight: ExternalPriorWeight::power(1.0).unwrap(),
            ess: None,
        }];
        let composed =
            compose_external_priors_with_alphas(&sources, &[1.0], &[0.0], &baseline).unwrap();
        let coef = composed.prior.gaussian_coefficients().unwrap();
        // α'=0 → identical to baseline
        assert!((coef.mean[0] - 0.0).abs() < 1e-12);
        assert!((coef.variance[0] - 4.0).abs() < 1e-12);
        assert_eq!(&*composed.alphas_requested, &[1.0]);
        assert_eq!(&*composed.alphas_applied, &[0.0]);
    }

    #[test]
    fn rejects_mixed_mixture_mode() {
        let baseline = PriorSet::weakly_informative(1);
        let sources = [
            ExternalPriorSource {
                id: Arc::from("a"),
                prior: gauss(1.0, 1.0),
                weight: ExternalPriorWeight::power(1.0).unwrap(),
                ess: None,
            },
            ExternalPriorSource {
                id: Arc::from("b"),
                prior: gauss(2.0, 1.0),
                weight: ExternalPriorWeight::power_mixture(1.0, 0.5).unwrap(),
                ess: None,
            },
        ];
        assert!(compose_external_priors(&sources, &baseline).is_err());
    }

    #[test]
    fn power_prior_ess_sums_over_contributing_sources() {
        // Same numbers as `power_prior_precision_add_analytic`, plus a declared
        // source ess=40: effective_ess = α·ess = 0.5*40 = 20; single positive
        // weight ⇒ kish_ess = 1.
        let baseline = gauss(0.0, 4.0);
        let sources = [ExternalPriorSource {
            id: Arc::from("old"),
            prior: gauss(2.0, 1.0),
            weight: ExternalPriorWeight::power(0.5).unwrap(),
            ess: Some(40.0),
        }];
        let composed = compose_external_priors(&sources, &baseline).unwrap();
        assert_eq!(composed.effective_ess.len(), 1);
        assert!((composed.effective_ess[0].unwrap() - 20.0).abs() < 1e-12);
        assert!((composed.composed_ess.unwrap() - 20.0).abs() < 1e-12);
        assert!((composed.kish_ess.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn power_prior_composed_ess_none_without_full_ess_coverage() {
        // Source `b` contributes (α=0.25 > 0) but declares no ess: a partial
        // sum over only `a` would misrepresent composed strength, so
        // composed_ess must be None even though `a`'s own effective_ess is Some.
        let baseline = gauss(0.0, 4.0);
        let sources = [
            ExternalPriorSource {
                id: Arc::from("a"),
                prior: gauss(2.0, 1.0),
                weight: ExternalPriorWeight::power(0.5).unwrap(),
                ess: Some(40.0),
            },
            ExternalPriorSource {
                id: Arc::from("b"),
                prior: gauss(3.0, 1.0),
                weight: ExternalPriorWeight::power(0.25).unwrap(),
                ess: None,
            },
        ];
        let composed = compose_external_priors(&sources, &baseline).unwrap();
        assert!((composed.effective_ess[0].unwrap() - 20.0).abs() < 1e-12);
        assert!(composed.effective_ess[1].is_none());
        assert!(composed.composed_ess.is_none());
        // kish_ess over alphas_applied=[0.5, 0.25]: (0.75)^2 / (0.25+0.0625) = 1.8.
        assert!((composed.kish_ess.unwrap() - 1.8).abs() < 1e-12);
    }

    #[test]
    fn power_prior_dropped_source_contributes_no_ess() {
        // `b` is dropped (α=0) despite declaring a large ess; its effective_ess
        // must report 0, and it must not appear in composed_ess at all (so a
        // missing ess on a dropped source cannot poison the sum).
        let baseline = gauss(0.0, 4.0);
        let sources = [
            ExternalPriorSource {
                id: Arc::from("a"),
                prior: gauss(2.0, 1.0),
                weight: ExternalPriorWeight::power(0.5).unwrap(),
                ess: Some(40.0),
            },
            ExternalPriorSource {
                id: Arc::from("b"),
                prior: gauss(5.0, 1.0),
                weight: ExternalPriorWeight::power(0.0).unwrap(),
                ess: Some(999.0),
            },
        ];
        let composed = compose_external_priors(&sources, &baseline).unwrap();
        assert!((composed.effective_ess[0].unwrap() - 20.0).abs() < 1e-12);
        assert!((composed.effective_ess[1].unwrap() - 0.0).abs() < 1e-12);
        assert!((composed.composed_ess.unwrap() - 20.0).abs() < 1e-12);
        // kish_ess over [0.5, 0.0]: (0.5)^2 / (0.25) = 1.0.
        assert!((composed.kish_ess.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mixture_composed_ess_is_none_but_effective_ess_reported_per_source() {
        // Same numbers as `mixture_preserves_leftover_baseline_mass`. The
        // moment-matched result is weaker than a precision sum implies, so
        // composed_ess must be None even though this source declares an ess
        // and its own effective_ess is reported.
        let baseline = gauss(0.0, 100.0);
        let sources = [ExternalPriorSource {
            id: Arc::from("s"),
            prior: gauss(10.0, 1.0),
            weight: ExternalPriorWeight::power_mixture(1.0, 0.4).unwrap(),
            ess: Some(50.0),
        }];
        let composed = compose_external_priors(&sources, &baseline).unwrap();
        assert!(composed.composed_ess.is_none());
        assert!((composed.effective_ess[0].unwrap() - 50.0).abs() < 1e-12);
        // kish_ess over mixture weights [0.4]: 1.0 (single positive weight).
        assert!((composed.kish_ess.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mixture_dropped_source_contributes_no_effective_ess() {
        // `b` has mixture_weight=0 (dropped: folds into leftover baseline mass)
        // despite declaring a large ess; effective_ess must report 0 for it.
        let baseline = gauss(0.0, 100.0);
        let sources = [
            ExternalPriorSource {
                id: Arc::from("a"),
                prior: gauss(10.0, 1.0),
                weight: ExternalPriorWeight::power_mixture(1.0, 0.4).unwrap(),
                ess: Some(50.0),
            },
            ExternalPriorSource {
                id: Arc::from("b"),
                prior: gauss(20.0, 1.0),
                weight: ExternalPriorWeight::power_mixture(1.0, 0.0).unwrap(),
                ess: Some(999.0),
            },
        ];
        let composed = compose_external_priors(&sources, &baseline).unwrap();
        assert!((composed.effective_ess[0].unwrap() - 50.0).abs() < 1e-12);
        assert!((composed.effective_ess[1].unwrap() - 0.0).abs() < 1e-12);
        assert!(composed.composed_ess.is_none());
        // kish_ess over mixture weights [0.4, 0.0]: (0.4)^2 / (0.16) = 1.0.
        assert!((composed.kish_ess.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sources_without_ess_report_none_effective_and_composed() {
        // Neither source declares an ess: every effective_ess entry is None
        // and composed_ess is None, but kish_ess (a weight-only diagnostic)
        // is still reported.
        let baseline = gauss(0.0, 4.0);
        let sources = [
            ExternalPriorSource {
                id: Arc::from("a"),
                prior: gauss(2.0, 1.0),
                weight: ExternalPriorWeight::power(0.5).unwrap(),
                ess: None,
            },
            ExternalPriorSource {
                id: Arc::from("b"),
                prior: gauss(3.0, 1.0),
                weight: ExternalPriorWeight::power(0.3).unwrap(),
                ess: None,
            },
        ];
        let composed = compose_external_priors(&sources, &baseline).unwrap();
        assert!(composed.effective_ess.iter().all(Option::is_none));
        assert!(composed.composed_ess.is_none());
        let kish = composed.kish_ess.unwrap();
        assert!(kish.is_finite() && kish > 0.0);
    }

    #[test]
    fn rejects_negative_ess() {
        let sources = [ExternalPriorSource {
            id: Arc::from("a"),
            prior: gauss(1.0, 1.0),
            weight: ExternalPriorWeight::power(1.0).unwrap(),
            ess: Some(-1.0),
        }];
        let err = sources[0].validate().unwrap_err();
        assert!(matches!(err, ProbError::InvalidPrior { .. }));
        let baseline = PriorSet::weakly_informative(1);
        assert!(compose_external_priors(&sources, &baseline).is_err());
    }

    #[test]
    fn kish_ess_matches_transport_adjustment_formula() {
        // Free function agrees with TransportAdjustment::kish_ess for the same
        // weights (the latter now delegates to this one).
        use crate::transport::TransportAdjustment;
        let adj = TransportAdjustment::new([1.0, 2.0, 3.0], [0.5, 0.25, 0.25]).unwrap();
        assert!((kish_ess(&[0.5, 0.25, 0.25]) - adj.kish_ess()).abs() < 1e-12);
    }
}
