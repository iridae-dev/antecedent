//! External prior bank: power-prior / mixture composition.
//!
//! Heterogeneous sources are composed into a single [`PriorSet`] usable as a
//! Bayesian coefficient prior. Priors never upgrade nonparametric identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::PriorAssumption;

use crate::error::ProbError;
use crate::prior::{GaussianCoefficientPrior, PriorSet, PriorSpec};

/// Floor on conjugate-scale variance after composition.
const COMPOSE_VAR_FLOOR: f64 = 1e-12;

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
        src.weight.validate()?;
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

    Ok(ComposedPrior {
        prior,
        source_ids: Arc::from(source_ids),
        alphas_requested: Arc::from(alphas_requested.to_vec()),
        alphas_applied: Arc::from(alphas_applied.to_vec()),
        mixture_weights: Arc::from(mixture_weights),
    })
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
            },
            ExternalPriorSource {
                id: Arc::from("b"),
                prior: gauss(2.0, 1.0),
                weight: ExternalPriorWeight::power_mixture(1.0, 0.5).unwrap(),
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
            },
            ExternalPriorSource {
                id: Arc::from("b"),
                prior: gauss(2.0, 1.0),
                weight: ExternalPriorWeight::power_mixture(1.0, 0.5).unwrap(),
            },
        ];
        assert!(compose_external_priors(&sources, &baseline).is_err());
    }
}
