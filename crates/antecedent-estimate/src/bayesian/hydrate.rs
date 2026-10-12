//! Prior hydration from posterior artifacts and summaries.

use std::sync::Arc;

use antecedent_core::PriorAssumption;
use antecedent_prob::{
    EffectPrior, GaussianCoefficientPrior, PosteriorQuantityKind, PriorSet, PriorSpec,
};

use crate::error::EstimationError;

#[cfg(test)]
use super::CausalPosterior;

/// Minimum coefficient prior variance when hydrating from a posterior (numerical floor).
const HYDRATE_VAR_FLOOR: f64 = 1e-12;
/// Smallest `|Δ|` accepted for the identity-link `ATE → β_T` mapping.
const HYDRATE_CONTRAST_FLOOR: f64 = 1e-12;

fn identity_ate_to_slope(mean: f64, sd: f64, delta: f64) -> Result<(f64, f64), EstimationError> {
    if !delta.is_finite() || delta.abs() <= HYDRATE_CONTRAST_FLOOR {
        return Err(EstimationError::stats_msg(
            "hydrate_prior: EffectFunctional identity-link ATE→β_T mapping requires a finite nonzero source contrast Δ (active − control)",
        ));
    }
    Ok((mean / delta, sd / delta.abs()))
}

fn effect_functional_bind_col(
    names: &[Arc<str>],
    treatment_col: Option<usize>,
) -> Result<usize, EstimationError> {
    if let Some(idx) = names.iter().position(|n| n.as_ref() == "treatment_at_mean_modifier") {
        return Ok(idx);
    }
    treatment_col.ok_or_else(|| {
        EstimationError::stats_msg(
            "hydrate_prior: EffectFunctional requires treatment_col, or a CATE `treatment_at_mean_modifier` coefficient",
        )
    })
}

/// Build a Gaussian coefficient [`PriorSet`] from posterior quantity summaries.
///
/// Uses coefficient-column posterior means and SDs (index-aligned). Effect
/// columns are ignored. Coefficient SDs are **absolute**; they are converted
/// into conjugate scale `V0` via [`GaussianCoefficientPrior::from_absolute_variance`]
/// using the source residual-variance posterior mean. A source without a
/// [`PosteriorQuantityKind::ResidualVariance`] column keeps the absolute
/// variances, marked with [`PriorSet::mark_absolute_coefficient_scale`]; the
/// target fit converts them with its own residual-variance estimate (see
/// [`residual_sigma2_for_hydrate`]).
///
/// Summaries carry no covariance, so this hydration is **diagonal**:
/// off-diagonal posterior covariance is dropped, and a transferred coefficient
/// prior can be tighter than the source marginal on linear combinations. Use
/// [`hydrate_prior_from_coefficient_moments`] with the source coefficient
/// covariance (or [`hydrate_prior_from_posterior`], which reads it from the
/// draws) for the dense `V0` that sequential updating needs.
///
/// # Errors
///
/// No coefficient columns, non-finite summaries, non-contiguous indices, or
/// dimension mismatch vs `expected_n_coef`.
#[cfg(test)]
pub(crate) fn hydrate_prior_from_quantity_summaries(
    quantities: &[PosteriorQuantityKind],
    mean: &[f64],
    sd: &[f64],
    expected_n_coef: Option<usize>,
) -> Result<PriorSet, EstimationError> {
    hydrate_prior_from_coefficient_moments(quantities, mean, sd, None, expected_n_coef)
}

/// Restriction id recorded on a prior hydrated with the source's full
/// coefficient covariance (dense `V0`).
pub(crate) const HYDRATED_COEFFICIENT_COVARIANCE_ID: &str = "hydrated_coefficient_covariance";

/// Restriction id recorded when a supplied source coefficient covariance could
/// not be used (not positive definite) and hydration fell back to the diagonal.
pub(crate) const HYDRATED_COVARIANCE_DROPPED_ID: &str = "hydrated_coefficient_covariance_dropped";

/// Build a Gaussian coefficient [`PriorSet`] from posterior coefficient moments.
///
/// As [`hydrate_prior_from_quantity_summaries`], plus the source's absolute
/// coefficient covariance `Σ` (row-major `p × p`, coefficient-index order;
/// see [`coefficient_covariance_from_draws`]). With it the hydrated prior is
/// dense: the per-coefficient `V0` diagonal is converted from the summary SDs
/// exactly as before, and the correlation of `Σ` is carried as a
/// [`PriorSpec::CoefficientCorrelation`], so `V0 = Σ / σ²`. For a known-σ²
/// conjugate Gaussian source this makes sequential updating reproduce the
/// pooled posterior up to the Monte Carlo error of `Σ`. Without `Σ` (or when
/// every off-diagonal correlation is zero) the prior is diagonal. A `Σ` whose
/// correlation is not positive definite (e.g. a coefficient with constant
/// draws) is dropped to the diagonal and recorded as
/// [`HYDRATED_COVARIANCE_DROPPED_ID`].
///
/// # Errors
///
/// As [`hydrate_prior_from_quantity_summaries`], or a covariance of the wrong
/// size or with non-finite entries.
pub fn hydrate_prior_from_coefficient_moments(
    quantities: &[PosteriorQuantityKind],
    mean: &[f64],
    sd: &[f64],
    coefficient_covariance: Option<&[f64]>,
    expected_n_coef: Option<usize>,
) -> Result<PriorSet, EstimationError> {
    if mean.len() != quantities.len() || sd.len() != quantities.len() {
        return Err(EstimationError::stats_msg(
            "hydrate_prior: mean/sd length must match quantities",
        ));
    }
    let mut coef_cols: Vec<(usize, usize)> = quantities
        .iter()
        .enumerate()
        .filter_map(|(col, q)| match q {
            PosteriorQuantityKind::Coefficient { index, .. } => Some((*index, col)),
            _ => None,
        })
        .collect();
    coef_cols.sort_by_key(|(index, _)| *index);
    let n_coef = coef_cols.len();
    if n_coef == 0 {
        return Err(EstimationError::stats_msg(
            "hydrate_prior_from_posterior: no coefficient columns in posterior",
        ));
    }
    if let Some(expected) = expected_n_coef {
        if n_coef != expected {
            return Err(EstimationError::PriorDimensionMismatch {
                posterior: n_coef,
                design: expected,
            });
        }
    }
    for (i, (index, _)) in coef_cols.iter().enumerate() {
        if *index != i {
            return Err(EstimationError::stats_msg(format!(
                "posterior coefficient indices are not contiguous (expected {i}, got {index})"
            )));
        }
    }
    let source_sigma2 = residual_sigma2_for_hydrate(quantities, mean)?;
    let sigma2 = source_sigma2.unwrap_or(1.0);
    let mut means = Vec::with_capacity(n_coef);
    let mut abs_vars = Vec::with_capacity(n_coef);
    for (_, col) in &coef_cols {
        let m = mean[*col];
        let s = sd[*col];
        if !m.is_finite() || !s.is_finite() {
            return Err(EstimationError::stats_msg(
                "posterior coefficient summary is non-finite; cannot hydrate prior",
            ));
        }
        means.push(m);
        abs_vars.push((s * s).max(HYDRATE_VAR_FLOOR * sigma2));
    }
    let coef = GaussianCoefficientPrior::from_absolute_variance(
        Arc::from(means),
        Arc::from(abs_vars),
        sigma2,
    )
    .map_err(EstimationError::from)?;
    let mut prior = PriorSet {
        specs: vec![PriorSpec::GaussianCoefficients(coef)],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    };
    if let Some(cov) = coefficient_covariance {
        attach_coefficient_correlation(&mut prior, cov, n_coef)?;
    }
    if source_sigma2.is_none() {
        prior.mark_absolute_coefficient_scale(&(0..n_coef).collect::<Vec<_>>());
    }
    Ok(prior)
}

/// Carry the correlation of the absolute source covariance `cov` on `prior`.
fn attach_coefficient_correlation(
    prior: &mut PriorSet,
    cov: &[f64],
    n_coef: usize,
) -> Result<(), EstimationError> {
    if cov.len() != n_coef.saturating_mul(n_coef) {
        return Err(EstimationError::stats_msg(format!(
            "hydrate_prior: coefficient covariance has {} entries, expected {n_coef}²",
            cov.len()
        )));
    }
    if cov.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::stats_msg(
            "hydrate_prior: coefficient covariance is non-finite",
        ));
    }
    match antecedent_prob::CoefficientCorrelation::from_covariance(cov, n_coef) {
        Ok(Some(corr)) => {
            prior.push(PriorSpec::CoefficientCorrelation(corr));
            prior.restrictions.push(PriorAssumption {
                id: Arc::from(HYDRATED_COEFFICIENT_COVARIANCE_ID),
                description: Arc::from(
                    "dense V0: the source posterior coefficient correlation is carried with \
                     the per-coefficient variances (V0 = posterior covariance / source \
                     sigma^2), so sequential updating keeps the source's joint uncertainty",
                ),
            });
        }
        Ok(None) => {}
        Err(_) => prior.restrictions.push(PriorAssumption {
            id: Arc::from(HYDRATED_COVARIANCE_DROPPED_ID),
            description: Arc::from(
                "the source posterior coefficient covariance is not positive definite; the \
                 hydrated prior is diagonal (off-diagonal covariance dropped) and can be \
                 tighter than the source on linear combinations",
            ),
        }),
    }
    Ok(())
}

/// Sample covariance of a posterior's coefficient columns.
///
/// `draws_colmajor` holds `n_draws` values per quantity of `quantities`
/// (column-major, as in `PosteriorDraws` and the posterior artifact). The
/// result is the row-major `p × p` covariance (denominator `n − 1`) in
/// coefficient-index order, or `None` when the draws are absent (a summary-only
/// artifact) or too few (`n_draws ≤ p`) to give a positive-definite estimate.
///
/// # Errors
///
/// Draw buffer length inconsistent with `quantities × n_draws`.
pub fn coefficient_covariance_from_draws(
    quantities: &[PosteriorQuantityKind],
    draws_colmajor: &[f64],
    n_draws: usize,
) -> Result<Option<Vec<f64>>, EstimationError> {
    if draws_colmajor.is_empty() {
        return Ok(None);
    }
    if draws_colmajor.len() != quantities.len().saturating_mul(n_draws) {
        return Err(EstimationError::stats_msg(
            "hydrate_prior: posterior draws do not match the quantity schema",
        ));
    }
    let mut coef_cols: Vec<(usize, usize)> = quantities
        .iter()
        .enumerate()
        .filter_map(|(col, q)| match q {
            PosteriorQuantityKind::Coefficient { index, .. } => Some((*index, col)),
            _ => None,
        })
        .collect();
    coef_cols.sort_by_key(|(index, _)| *index);
    let p = coef_cols.len();
    if p == 0 || n_draws <= p {
        return Ok(None);
    }
    let cols: Vec<&[f64]> = coef_cols
        .iter()
        .map(|(_, col)| &draws_colmajor[col * n_draws..(col + 1) * n_draws])
        .collect();
    let means: Vec<f64> = cols.iter().map(|c| c.iter().sum::<f64>() / n_draws as f64).collect();
    let mut cov = vec![0.0; p * p];
    for i in 0..p {
        for j in i..p {
            let mut acc = 0.0;
            for d in 0..n_draws {
                acc += (cols[i][d] - means[i]) * (cols[j][d] - means[j]);
            }
            let v = acc / (n_draws - 1) as f64;
            cov[i * p + j] = v;
            cov[j * p + i] = v;
        }
    }
    Ok(Some(cov))
}

/// Source `σ²` for absolute→`V0` hydrate: the residual-variance posterior mean.
///
/// A known-σ² Gaussian fit records its σ² as a constant
/// [`PosteriorQuantityKind::ResidualVariance`] column, so its mean is that σ².
/// `None` when the source posterior has no such column (a GLM, a draws-free
/// summary artifact, or a known-σ² artifact written before the column was
/// recorded): it carries no scale to convert its
/// absolute coefficient variances into the conjugate `V0` a Gaussian target
/// prior is written in, and assuming `σ² = 1` would mis-scale the prior variance
/// by the target's σ². The hydrated prior then keeps the **absolute** variances
/// under [`PriorSet::mark_absolute_coefficient_scale`], and the target fit plugs
/// in its own residual-variance estimate (`V0 = absolute / σ̂²_target`, recorded
/// as a diagnostic note); a GLM target reads `V0` at `σ² ≡ 1`, where absolute
/// and `V0` coincide.
fn residual_sigma2_for_hydrate(
    quantities: &[PosteriorQuantityKind],
    mean: &[f64],
) -> Result<Option<f64>, EstimationError> {
    for (i, q) in quantities.iter().enumerate() {
        if matches!(q, PosteriorQuantityKind::ResidualVariance) {
            let s2 = mean[i];
            if s2 <= 0.0 || !s2.is_finite() {
                return Err(EstimationError::stats_msg(
                    "hydrate_prior: residual_variance mean must be finite and > 0",
                ));
            }
            return Ok(Some(s2));
        }
    }
    Ok(None)
}

/// Build a Gaussian coefficient [`PriorSet`] from a fitted posterior (sequential Bayes).
///
/// Dense: the coefficient covariance is read from the posterior draws
/// ([`coefficient_covariance_from_draws`]) and carried as the prior's
/// coefficient correlation (see [`hydrate_prior_from_coefficient_moments`]).
///
/// # Errors
///
/// See [`hydrate_prior_from_coefficient_moments`].
#[cfg(test)]
pub(crate) fn hydrate_prior_from_posterior(
    posterior: &CausalPosterior,
    expected_n_coef: Option<usize>,
) -> Result<PriorSet, EstimationError> {
    let quantities = &posterior.draws.schema.quantities;
    let cov = coefficient_covariance_from_draws(
        quantities,
        &posterior.draws.values,
        posterior.draws.n_draws,
    )?;
    hydrate_prior_from_coefficient_moments(
        quantities,
        &posterior.summaries.mean,
        &posterior.summaries.sd,
        cov.as_deref(),
        expected_n_coef,
    )
}

/// Bridge from a banked posterior into a target design's coefficient prior.
///
/// Mirrors `antecedent_io::PriorMapping` without depending on `antecedent-io` (avoids a
/// cycle). Convert at the facade.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum HydrateMapping {
    /// Identical coefficient subspace (P1-C sequential Bayes).
    IdenticalCoefficientSubspace,
    /// Effect-functional transfer via a named source quantity (e.g. `"ate"`).
    EffectFunctional {
        /// Source effect / quantity name.
        source_quantity: String,
    },
    /// Explicit source→target quantity name pairs.
    NamedParameters {
        /// `(source_name, target_name)` pairs.
        pairs: Vec<(String, String)>,
    },
}

/// A source posterior as hydration sees it: quantity schema, per-quantity moments, and
/// the optional absolute coefficient covariance and treatment contrast.
#[derive(Clone, Copy, Debug)]
pub struct HydrateSource<'a> {
    /// Source posterior quantity kinds (artifact column schema).
    pub quantities: &'a [PosteriorQuantityKind],
    /// Source posterior means, aligned to [`Self::quantities`].
    pub mean: &'a [f64],
    /// Source posterior SDs, aligned to [`Self::quantities`].
    pub sd: &'a [f64],
    /// Source's absolute coefficient covariance (row-major `n_coef²`), when its draws
    /// were retained. `None` hydrates the diagonal prior only.
    pub coefficient_covariance: Option<&'a [f64]>,
    /// Source treatment contrast `active − control`, needed by
    /// [`HydrateMapping::EffectFunctional`].
    pub source_contrast: Option<f64>,
}

impl<'a> HydrateSource<'a> {
    /// A summaries-only source: moments with no covariance or contrast.
    #[must_use]
    pub fn summaries(
        quantities: &'a [PosteriorQuantityKind],
        mean: &'a [f64],
        sd: &'a [f64],
    ) -> Self {
        Self { quantities, mean, sd, coefficient_covariance: None, source_contrast: None }
    }

    /// Carry the source's absolute coefficient covariance.
    #[must_use]
    pub fn with_coefficient_covariance(mut self, covariance: Option<&'a [f64]>) -> Self {
        self.coefficient_covariance = covariance;
        self
    }

    /// Record the source treatment contrast `active − control`.
    #[must_use]
    pub fn with_source_contrast(mut self, contrast: Option<f64>) -> Self {
        self.source_contrast = contrast;
        self
    }
}

/// The model a hydrated prior is built for: its baseline prior and coefficient layout.
#[derive(Clone, Copy, Debug)]
pub struct HydrateTarget<'a> {
    /// Baseline prior; unmapped dimensions and residual specs are kept from it.
    pub baseline: &'a PriorSet,
    /// Target coefficient names in design-column order.
    pub coef_names: &'a [Arc<str>],
    /// Treatment coefficient column, when the target has one.
    pub treatment_col: Option<usize>,
}

/// Build a coefficient [`PriorSet`] under a declared [`HydrateMapping`].
///
/// - [`HydrateMapping::IdenticalCoefficientSubspace`]: full coef hydrate; hard-errors
///   when source coef count ≠ baseline length. Carries the source's coefficient
///   covariance (correlation, dense `V0`; see
///   [`hydrate_prior_from_coefficient_moments`]) when
///   [`HydrateSource::coefficient_covariance`] is set, and is diagonal otherwise.
/// - [`HydrateMapping::EffectFunctional`]: identity-link ATE → slope bridge
///   `β ~ N(μ_ATE / Δ, (σ_ATE / Δ)²)` onto the treatment coefficient, or onto
///   `treatment_at_mean_modifier` when that CATE coefficient is present.
///   [`HydrateSource::source_contrast`] is the source `active − control`. Missing or
///   ~0 Δ is a typed refuse of the mapping. Other dims keep the baseline. Remains
///   diagonal.
/// - [`HydrateMapping::NamedParameters`]: maps named source moments onto named
///   target coefficients; unmapped dims keep the baseline. A complete one-to-one map
///   of named coefficient columns also carries the full covariance, permuted into
///   target order; partial or non-coefficient mappings remain diagonal.
///
/// Records `external_effect_prior` / `external_named_prior` on
/// [`PriorSet::restrictions`].
///
/// # Errors
///
/// Dimension mismatch, missing effect column, unknown names, missing/zero Δ,
/// a covariance of the wrong size, or invalid baseline.
#[allow(
    clippy::too_many_lines,
    reason = "one arm per hydration mapping (identical subspace, effect functional, named coefficients), each assembling its prior and restriction record from the validated source and target"
)]
pub fn hydrate_prior(
    mapping: &HydrateMapping,
    source: &HydrateSource<'_>,
    target: &HydrateTarget<'_>,
) -> Result<PriorSet, EstimationError> {
    let HydrateSource { quantities, mean, sd, coefficient_covariance, source_contrast } = *source;
    let HydrateTarget { baseline, coef_names: target_coef_names, treatment_col } = *target;
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::PriorConstruction,
        || {
            if mean.len() != quantities.len() || sd.len() != quantities.len() {
                return Err(EstimationError::stats_msg(
                    "hydrate_prior: mean/sd length must match quantities",
                ));
            }
            let n_target = target_coef_names.len();
            let base_coef = baseline.gaussian_coefficients().ok_or_else(|| {
                EstimationError::stats_msg("hydrate_prior: baseline missing GaussianCoefficients")
            })?;
            if base_coef.len() != n_target {
                return Err(EstimationError::stats_msg(format!(
                    "hydrate_prior: baseline n_coef {} != target_coef_names {}",
                    base_coef.len(),
                    n_target
                )));
            }

            match mapping {
                HydrateMapping::IdenticalCoefficientSubspace => {
                    let mut prior = hydrate_prior_from_coefficient_moments(
                        quantities,
                        mean,
                        sd,
                        coefficient_covariance,
                        Some(n_target),
                    )?;
                    // Preserve residual specs from baseline when present.
                    merge_baseline_residuals(&mut prior, baseline);
                    Ok(prior)
                }
                HydrateMapping::EffectFunctional { source_quantity } => {
                    let t_col = effect_functional_bind_col(target_coef_names, treatment_col)?;
                    if t_col >= n_target {
                        return Err(EstimationError::stats_msg(format!(
                            "hydrate_prior: treatment_col {t_col} out of range for {n_target} coefs"
                        )));
                    }
                    let delta = source_contrast.ok_or_else(|| {
                EstimationError::stats_msg(
                    "hydrate_prior: EffectFunctional identity-link ATE→β_T mapping requires a finite nonzero source contrast Δ (active − control)",
                )
            })?;
                    let (m, s) = quantity_moments(quantities, mean, sd, source_quantity.as_str())?;
                    let (slope_mean, slope_sd) = identity_ate_to_slope(m, s, delta)?;
                    let effect =
                        EffectPrior::new(slope_mean, slope_sd.max(HYDRATE_VAR_FLOOR.sqrt()))
                            .map_err(EstimationError::from)?;
                    let source_sigma2 = residual_sigma2_for_hydrate(quantities, mean)?;
                    let sigma2 = source_sigma2.unwrap_or(1.0);
                    let mut means: Vec<f64> = base_coef.mean.to_vec();
                    let mut vars: Vec<f64> = base_coef.variance.to_vec();
                    means[t_col] = effect.mean;
                    let abs_var = (effect.sd * effect.sd).max(HYDRATE_VAR_FLOOR * sigma2);
                    vars[t_col] = abs_var / sigma2;
                    let coef = GaussianCoefficientPrior {
                        mean: Arc::from(means),
                        variance: Arc::from(vars),
                    };
                    coef.validate().map_err(EstimationError::from)?;
                    let target_name = target_coef_names[t_col].as_ref();
                    let implied = effect.mean * delta;
                    let mut prior = PriorSet {
                        specs: vec![PriorSpec::GaussianCoefficients(coef)],
                        contrast: baseline.contrast,
                        categorical: baseline.categorical.clone(),
                        restrictions: vec![PriorAssumption {
                            id: Arc::from("external_effect_prior"),
                            description: Arc::from(format!(
                                "external effect-functional prior: identity-link ATE→β via N(μ/Δ, (σ/Δ)²) from `{source_quantity}` (Δ={delta}) onto {target_name}; implied NDE/ATE mean {implied}; {}",
                                scale_conversion_note(source_sigma2)
                            )),
                        }],
                    };
                    if source_sigma2.is_none() {
                        prior.mark_absolute_coefficient_scale(&[t_col]);
                    }
                    merge_baseline_residuals(&mut prior, baseline);
                    Ok(prior)
                }
                HydrateMapping::NamedParameters { pairs } => {
                    if pairs.is_empty() {
                        return Err(EstimationError::stats_msg(
                            "hydrate_prior: NamedParameters requires at least one pair",
                        ));
                    }
                    let mut used_sources = std::collections::HashSet::new();
                    let mut used_targets = std::collections::HashSet::new();
                    if pairs.iter().any(|(source, target)| {
                        source.trim().is_empty()
                            || target.trim().is_empty()
                            || !used_sources.insert(source)
                            || !used_targets.insert(target)
                    }) {
                        return Err(EstimationError::stats_msg(
                            "hydrate_prior: NamedParameters requires distinct nonblank source and target names",
                        ));
                    }
                    let target_name_set: std::collections::HashSet<&str> =
                        target_coef_names.iter().map(AsRef::as_ref).collect();
                    if target_name_set.len() != n_target {
                        return Err(EstimationError::stats_msg(
                            "hydrate_prior: target coefficient names must be unique",
                        ));
                    }
                    for (source, _) in pairs {
                        let count = quantities
                            .iter()
                            .filter(|quantity| match quantity {
                                PosteriorQuantityKind::Coefficient { name: Some(name), .. }
                                | PosteriorQuantityKind::Effect { name }
                                | PosteriorQuantityKind::Scalar { name } => {
                                    name.as_ref() == source.as_str()
                                }
                                PosteriorQuantityKind::ResidualVariance => {
                                    source == "residual_variance"
                                }
                                PosteriorQuantityKind::Coefficient { name: None, .. } => false,
                            })
                            .count();
                        if count != 1 {
                            return Err(EstimationError::stats_msg(format!(
                                "hydrate_prior: source quantity `{source}` must be uniquely named"
                            )));
                        }
                    }
                    let mut means: Vec<f64> = base_coef.mean.to_vec();
                    let mut vars: Vec<f64> = base_coef.variance.to_vec();
                    let name_index: std::collections::HashMap<&str, usize> = target_coef_names
                        .iter()
                        .enumerate()
                        .map(|(i, n)| (n.as_ref(), i))
                        .collect();
                    let source_sigma2 = residual_sigma2_for_hydrate(quantities, mean)?;
                    let sigma2 = source_sigma2.unwrap_or(1.0);
                    let mut mapped = Vec::with_capacity(pairs.len());
                    for (src, tgt) in pairs {
                        let (m, s) = quantity_moments(quantities, mean, sd, src)?;
                        let Some(&idx) = name_index.get(tgt.as_str()) else {
                            return Err(EstimationError::stats_msg(format!(
                                "hydrate_prior: unknown target coefficient name `{tgt}`"
                            )));
                        };
                        means[idx] = m;
                        let abs_var = (s * s).max(HYDRATE_VAR_FLOOR * sigma2);
                        vars[idx] = abs_var / sigma2;
                        mapped.push(idx);
                    }
                    let coef = GaussianCoefficientPrior {
                        mean: Arc::from(means),
                        variance: Arc::from(vars),
                    };
                    coef.validate().map_err(EstimationError::from)?;
                    let pair_desc = pairs
                        .iter()
                        .map(|(a, b)| format!("{a}->{b}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let mut prior = PriorSet {
                        specs: vec![PriorSpec::GaussianCoefficients(coef)],
                        contrast: baseline.contrast,
                        categorical: baseline.categorical.clone(),
                        restrictions: vec![PriorAssumption {
                            id: Arc::from("external_named_prior"),
                            description: Arc::from(match source_sigma2 {
                                Some(sigma2) => format!(
                                    "external named-parameter prior ({pair_desc}); absolute posterior Var converted to V0 with source σ²={sigma2}; diagonal only (off-diagonal covariance dropped)"
                                ),
                                None => format!(
                                    "external named-parameter prior ({pair_desc}); {}; diagonal only (off-diagonal covariance dropped)",
                                    scale_conversion_note(None)
                                ),
                            }),
                        }],
                    };
                    if source_sigma2.is_none() {
                        prior.mark_absolute_coefficient_scale(&mapped);
                    }
                    if let Some(cov) = coefficient_covariance {
                        if let Some(permuted) = permute_full_named_coefficient_covariance(
                            quantities,
                            pairs,
                            target_coef_names,
                            cov,
                        )? {
                            attach_coefficient_correlation(&mut prior, &permuted, n_target)?;
                            if prior.coefficient_correlation().is_some() {
                                prior.restrictions[0].description = Arc::from(format!(
                                    "external named-parameter prior ({pair_desc}); {}; full coefficient covariance retained under explicit named permutation",
                                    scale_conversion_note(source_sigma2)
                                ));
                            }
                        }
                    }
                    merge_baseline_residuals(&mut prior, baseline);
                    Ok(prior)
                }
            }
        },
    )
}

/// Return a target-ordered covariance only for a complete named coefficient
/// bijection. An ordinary partial named prior keeps its baseline dimensions.
fn permute_full_named_coefficient_covariance(
    quantities: &[PosteriorQuantityKind],
    pairs: &[(String, String)],
    target_names: &[Arc<str>],
    covariance: &[f64],
) -> Result<Option<Vec<f64>>, EstimationError> {
    let mut source: Vec<_> = quantities
        .iter()
        .filter_map(|quantity| match quantity {
            PosteriorQuantityKind::Coefficient { index, name: Some(name) } => {
                Some((*index, name.as_ref()))
            }
            _ => None,
        })
        .collect();
    let n = target_names.len();
    if source.len() != n || pairs.len() != n {
        return Ok(None);
    }
    source.sort_by_key(|(index, _)| *index);
    if source
        .iter()
        .enumerate()
        .any(|(position, (index, name))| *index != position || name.trim().is_empty())
    {
        return Ok(None);
    }
    if covariance.len() != n.saturating_mul(n) {
        return Err(EstimationError::stats_msg(
            "hydrate_prior: named coefficient covariance dimension mismatch",
        ));
    }
    let mut target_by_source = Vec::with_capacity(n);
    let mut seen_targets = std::collections::HashSet::new();
    for (_, source_name) in &source {
        let matched: Vec<_> = pairs.iter().filter(|(name, _)| name == source_name).collect();
        if matched.len() != 1 {
            return Ok(None);
        }
        let matched_target: Vec<_> = target_names
            .iter()
            .enumerate()
            .filter(|(_, name)| name.as_ref() == matched[0].1.as_str())
            .collect();
        if matched_target.len() != 1 || !seen_targets.insert(matched_target[0].0) {
            return Ok(None);
        }
        target_by_source.push(matched_target[0].0);
    }
    let mut permuted = vec![0.0; covariance.len()];
    for source_i in 0..n {
        for source_j in 0..n {
            permuted[target_by_source[source_i] * n + target_by_source[source_j]] =
                covariance[source_i * n + source_j];
        }
    }
    Ok(Some(permuted))
}

/// How a hydrated absolute variance reached the target prior's scale.
fn scale_conversion_note(source_sigma2: Option<f64>) -> String {
    match source_sigma2 {
        Some(sigma2) => format!("absolute Var(β) converted to V0 with source σ²={sigma2}"),
        None => "absolute Var(β) kept on the absolute scale (the source records no residual \
                 variance); the target fit converts it to V0 with its own residual-variance \
                 estimate"
            .to_owned(),
    }
}

fn merge_baseline_residuals(prior: &mut PriorSet, baseline: &PriorSet) {
    for spec in &baseline.specs {
        match spec {
            PriorSpec::ResidualInvGamma(_) | PriorSpec::KnownResidualVariance(_) => {
                if !prior.specs.iter().any(|s| {
                    matches!(
                        s,
                        PriorSpec::ResidualInvGamma(_) | PriorSpec::KnownResidualVariance(_)
                    )
                }) {
                    prior.specs.push(spec.clone());
                }
            }
            // The hydrated coefficient block (and its correlation) replaces the baseline's.
            PriorSpec::GaussianCoefficients(_) | PriorSpec::CoefficientCorrelation(_) => {}
        }
    }
}

fn quantity_moments(
    quantities: &[PosteriorQuantityKind],
    mean: &[f64],
    sd: &[f64],
    name: &str,
) -> Result<(f64, f64), EstimationError> {
    for (i, q) in quantities.iter().enumerate() {
        let q_name = match q {
            PosteriorQuantityKind::Effect { name: n }
            | PosteriorQuantityKind::Scalar { name: n } => Some(n.as_ref()),
            PosteriorQuantityKind::Coefficient { name: n, .. } => {
                n.as_ref().map(std::convert::AsRef::as_ref)
            }
            PosteriorQuantityKind::ResidualVariance => Some("residual_variance"),
        };
        if q_name == Some(name) {
            let m = mean[i];
            let s = sd[i];
            if !m.is_finite() || !s.is_finite() {
                return Err(EstimationError::stats_msg(format!(
                    "hydrate_prior: non-finite summary for quantity `{name}`"
                )));
            }
            return Ok((m, s.max(HYDRATE_VAR_FLOOR.sqrt())));
        }
    }
    Err(EstimationError::stats_msg(format!("hydrate_prior: missing quantity `{name}`")))
}
