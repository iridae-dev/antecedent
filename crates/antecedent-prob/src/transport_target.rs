//! Row-weighted target populations and kernel transport weights.
//!
//! A tag-defined target population compares population labels
//! ([`crate::transport`]). A *weighted target* is declared differently: the
//! target is a weighting over the source rows themselves, for example a
//! geometry neighbourhood given as per-row weights, or a kernel over a
//! covariate distance. Transport compatibility then reads the weighting's
//! support and overlap with the source rows (positivity, Kish effective sample
//! size, concentration) instead of tag equality, and refuses with a
//! next-attempt message when the overlap fails.
//!
//! The tag path is untouched: nothing here is reached from
//! [`crate::transport::apply_transport`].
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Write as _;
use std::sync::Arc;

use antecedent_core::PriorAssumption;

use crate::error::ProbError;
use crate::external_prior::{
    ComposedPrior, ExternalPriorSource, compose_external_priors_with_alphas, kish_ess,
};
use crate::prior::PriorSet;
use crate::transport::{
    TRANSPORT_MIN_KISH_ESS, TransportAdjustment, TransportError, TransportPolicy,
    replace_coef_moments, source_residual_sigma2,
};

/// Refusal code when a weighted target does not overlap the source rows.
pub(crate) const TARGET_OVERLAP_CODE: &str = "transport_target_overlap_failed";

/// Stable assumption id recorded for a weighted-target transport claim.
pub(crate) const TARGET_ASSUMPTION_ID: &str = "external_transport_target_weighting";

/// Default floor on the Kish ESS as a fraction of the source rows.
///
/// A weighting whose effective rows are fewer than this share of the source
/// rows rests on a thin slice of the source. It applies on top of the absolute
/// [`TRANSPORT_MIN_KISH_ESS`] floor, which is never relaxed.
pub const TARGET_MIN_ESS_FRACTION: f64 = 0.05;

/// Kernel family turning a scaled unit distance into a weight.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum KernelFamily {
    /// `exp(-u^2 / 2)` with `u = d / h`.
    Gaussian,
    /// `1 - u^2` for `u < 1`, else 0.
    Epanechnikov,
    /// `(1 - u^3)^3` for `u < 1`, else 0.
    Tricube,
}

impl KernelFamily {
    /// Stable string id for assumptions / wire formats.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Epanechnikov => "epanechnikov",
            Self::Tricube => "tricube",
        }
    }

    /// Parse from a wire / Python string.
    ///
    /// # Errors
    ///
    /// Unknown family name.
    pub fn parse(s: &str) -> Result<Self, TransportError> {
        match s {
            "gaussian" => Ok(Self::Gaussian),
            "epanechnikov" => Ok(Self::Epanechnikov),
            "tricube" => Ok(Self::Tricube),
            _ => Err(TransportError::InvalidWeights {
                message: "unknown kernel family; use gaussian, epanechnikov or tricube",
            }),
        }
    }
}

/// A declared kernel turning caller-supplied unit distances into transport
/// weights. The family and bandwidth are the whole declaration; both enter the
/// prior identity through [`TransportKernel::assumption`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransportKernel {
    family: KernelFamily,
    bandwidth: f64,
}

impl TransportKernel {
    /// Declare a kernel.
    ///
    /// # Errors
    ///
    /// Non-finite or non-positive bandwidth.
    pub fn new(family: KernelFamily, bandwidth: f64) -> Result<Self, TransportError> {
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(TransportError::InvalidWeights {
                message: "kernel bandwidth must be finite and > 0",
            });
        }
        Ok(Self { family, bandwidth })
    }

    /// Kernel family.
    #[must_use]
    pub const fn family(&self) -> KernelFamily {
        self.family
    }

    /// Declared bandwidth.
    #[must_use]
    pub const fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// Kernel value at a non-negative distance.
    #[must_use]
    pub fn weight(&self, distance: f64) -> f64 {
        let u = distance / self.bandwidth;
        match self.family {
            KernelFamily::Gaussian => (-0.5 * u * u).exp(),
            KernelFamily::Epanechnikov => {
                if u < 1.0 {
                    1.0 - u * u
                } else {
                    0.0
                }
            }
            KernelFamily::Tricube => {
                if u < 1.0 {
                    let c = 1.0 - u * u * u;
                    c * c * c
                } else {
                    0.0
                }
            }
        }
    }

    /// Weights for a slice of unit distances.
    ///
    /// # Errors
    ///
    /// Empty input, or a distance that is not finite and non-negative.
    pub fn weights(&self, distances: &[f64]) -> Result<Vec<f64>, TransportError> {
        if distances.is_empty() {
            return Err(TransportError::InvalidWeights {
                message: "kernel weights require at least one unit distance",
            });
        }
        if distances.iter().any(|d| !d.is_finite() || *d < 0.0) {
            return Err(TransportError::InvalidWeights {
                message: "unit distances must be finite and >= 0",
            });
        }
        Ok(distances.iter().map(|&d| self.weight(d)).collect())
    }

    /// Assumption recording the kernel. The id carries the family and the exact
    /// bandwidth bits so the prior identity changes with either.
    #[must_use]
    pub fn assumption(&self) -> PriorAssumption {
        PriorAssumption {
            id: Arc::from(format!(
                "transport_kernel:{}:bandwidth_bits={:016x}",
                self.family.id(),
                self.bandwidth.to_bits()
            )),
            description: Arc::from(format!(
                "transport weights are the {} kernel of the declared unit distances at \
                 bandwidth {}",
                self.family.id(),
                self.bandwidth
            )),
        }
    }
}

/// Support and overlap of a target weighting with the source rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetOverlap {
    /// Source rows carrying a weight.
    pub n_rows: usize,
    /// Rows with strictly positive weight (the weighting's support).
    pub n_support: usize,
    /// `n_support / n_rows`.
    pub support_fraction: f64,
    /// Kish effective sample size of the weights.
    pub kish_ess: f64,
    /// `kish_ess / n_rows`.
    pub ess_fraction: f64,
    /// Largest single-row share of the total weight.
    pub max_weight_share: f64,
}

impl TargetOverlap {
    /// Overlap diagnostics for validated (finite, non-negative, positive mass)
    /// weights.
    #[must_use]
    pub fn from_weights(weights: &[f64]) -> Self {
        let n_rows = weights.len();
        let n_support = weights.iter().filter(|&&w| w > 0.0).count();
        let mass: f64 = weights.iter().sum();
        let max = weights.iter().copied().fold(0.0_f64, f64::max);
        let ess = kish_ess(weights);
        let n = n_rows as f64;
        Self {
            n_rows,
            n_support,
            support_fraction: n_support as f64 / n,
            kish_ess: ess,
            ess_fraction: ess / n,
            max_weight_share: if mass > 0.0 { max / mass } else { 0.0 },
        }
    }

    /// The overlap check: positivity, the absolute Kish floor, and the
    /// fraction floor. Returns the next-attempt message when it fails.
    #[must_use]
    pub fn failure(&self, min_ess_fraction: f64) -> Option<String> {
        if self.n_support == 0
            || self.kish_ess < TRANSPORT_MIN_KISH_ESS
            || self.ess_fraction < min_ess_fraction
        {
            Some(self.message(min_ess_fraction))
        } else {
            None
        }
    }

    fn message(&self, min_ess_fraction: f64) -> String {
        format!(
            "{TARGET_OVERLAP_CODE}: the target weighting has Kish ESS {:.3} over {} source rows \
             (support {} of {} rows, largest single-row share {:.3}); the floor is {} effective \
             rows and an ESS fraction of at least {}. Next attempt: widen the kernel bandwidth or \
             the neighbourhood, move the target toward where the source rows are, or add source \
             rows that cover the target",
            self.kish_ess,
            self.n_rows,
            self.n_support,
            self.n_rows,
            self.max_weight_share,
            TRANSPORT_MIN_KISH_ESS,
            min_ess_fraction,
        )
    }
}

/// The overlap check of a bare row weighting: positivity, the absolute Kish floor and the
/// fraction floor of `min_ess_fraction` of the rows.
///
/// This is the one gate every weighted target passes, whether it transports a prior, a
/// standardized effect or a trial contrast.
///
/// # Errors
///
/// Weights that are empty, non-finite or negative; a fraction outside `[0, 1]`; or
/// `transport_target_overlap_failed` with a next-attempt message when the check fails.
pub fn check_weights_overlap(
    weights: &[f64],
    min_ess_fraction: f64,
) -> Result<TargetOverlap, ProbError> {
    if weights.is_empty() || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(TransportError::InvalidWeights {
            message: "target weights must be non-empty, finite and >= 0",
        }
        .into());
    }
    if !(0.0..=1.0).contains(&min_ess_fraction) {
        return Err(TransportError::InvalidWeights {
            message: "min_ess_fraction must lie in [0, 1]",
        }
        .into());
    }
    let overlap = TargetOverlap::from_weights(weights);
    match overlap.failure(min_ess_fraction) {
        Some(message) => Err(overlap_refusal(message)),
        None => Ok(overlap),
    }
}

/// A target population declared as a weighting over the source rows.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedTarget {
    adjustment: TransportAdjustment,
    kernel: Option<TransportKernel>,
    min_ess_fraction: f64,
}

fn overlap_refusal(message: String) -> ProbError {
    ProbError::Refused { code: TARGET_OVERLAP_CODE, message }
}

impl WeightedTarget {
    /// Target given as per-row weights (for example a geometry neighbourhood).
    ///
    /// # Errors
    ///
    /// Invalid effects or weights (see [`TransportAdjustment::new`]).
    pub fn from_row_weights(
        unit_effects: impl Into<Arc<[f64]>>,
        row_weights: impl Into<Arc<[f64]>>,
    ) -> Result<Self, TransportError> {
        Ok(Self {
            adjustment: TransportAdjustment::new(unit_effects, row_weights)?,
            kernel: None,
            min_ess_fraction: TARGET_MIN_ESS_FRACTION,
        })
    }

    /// Target given as a declared kernel over per-row distances to the target.
    ///
    /// # Errors
    ///
    /// Invalid effects or distances, or [`ProbError::Refused`]
    /// (`transport_target_overlap_failed`) when the kernel gives every row zero
    /// weight, so the target has no support among the source rows.
    pub fn from_kernel(
        unit_effects: impl Into<Arc<[f64]>>,
        distances: &[f64],
        kernel: TransportKernel,
    ) -> Result<Self, ProbError> {
        let weights = kernel.weights(distances)?;
        if weights.iter().sum::<f64>() <= 0.0 {
            return Err(overlap_refusal(TargetOverlap::from_weights(&weights).message(0.0)));
        }
        let adjustment = TransportAdjustment::new(unit_effects, weights)?;
        Ok(Self { adjustment, kernel: Some(kernel), min_ess_fraction: TARGET_MIN_ESS_FRACTION })
    }

    /// Declare the ESS-fraction floor (the absolute Kish floor still applies).
    ///
    /// # Errors
    ///
    /// A fraction outside `[0, 1]`.
    pub fn with_min_ess_fraction(mut self, fraction: f64) -> Result<Self, TransportError> {
        if !(0.0..=1.0).contains(&fraction) {
            return Err(TransportError::InvalidWeights {
                message: "min_ess_fraction must lie in [0, 1]",
            });
        }
        self.min_ess_fraction = fraction;
        Ok(self)
    }

    /// The row weights.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.adjustment.target_weights
    }

    /// The unit-level effects, aligned with [`Self::weights`].
    #[must_use]
    pub fn unit_effects(&self) -> &[f64] {
        &self.adjustment.unit_effects
    }

    /// The declared ESS-fraction floor.
    #[must_use]
    pub const fn min_ess_fraction(&self) -> f64 {
        self.min_ess_fraction
    }

    /// The declared kernel, when the weights came from one.
    #[must_use]
    pub const fn kernel(&self) -> Option<TransportKernel> {
        self.kernel
    }

    /// Support and overlap diagnostics.
    #[must_use]
    pub fn overlap(&self) -> TargetOverlap {
        TargetOverlap::from_weights(&self.adjustment.target_weights)
    }

    /// The compatibility check: `Ok` when the weighting overlaps the source.
    ///
    /// # Errors
    ///
    /// `transport_target_overlap_failed` with a next-attempt message.
    pub fn check_overlap(&self) -> Result<TargetOverlap, ProbError> {
        let overlap = self.overlap();
        match overlap.failure(self.min_ess_fraction) {
            Some(message) => Err(overlap_refusal(message)),
            None => Ok(overlap),
        }
    }
}

/// Transport sources toward a weighted target.
///
/// Refuses when the weighting fails the overlap check. Otherwise each source's
/// chosen coefficient takes the weighted mean and is widened by the sampling
/// variance of that mean, exactly as the tag path does for caller weights, and
/// the claim (and kernel, when declared) is recorded as assumptions.
///
/// # Errors
///
/// Overlap refusal, or an invalid coefficient index / prior.
pub fn apply_weighted_target_transport(
    sources: &[ExternalPriorSource],
    policy: TransportPolicy,
    target: &WeightedTarget,
    coef_index: Option<usize>,
) -> Result<(Vec<ExternalPriorSource>, TargetOverlap), ProbError> {
    let overlap = target.check_overlap()?;
    let adj = &target.adjustment;
    let (mean, _het_var) = adj.weighted_moments();
    let mut out = Vec::with_capacity(sources.len());
    for src in sources {
        let mut prepared = src.clone();
        let coef =
            prepared.prior.gaussian_coefficients().ok_or(TransportError::InvalidWeights {
                message: "source prior missing GaussianCoefficients for transport reweight",
            })?;
        let n_coef = coef.len();
        let idx = coef_index.unwrap_or(n_coef.saturating_sub(1));
        if idx >= n_coef {
            return Err(TransportError::CoefIndexOutOfRange { index: idx, n_coef }.into());
        }
        let sigma2 = source_residual_sigma2(&prepared.prior);
        let reweight_v0 = adj.reweighted_mean_variance() / sigma2;
        let source_v0 = coef.variance[idx];
        let v0 = source_v0 + reweight_v0;
        replace_coef_moments(&mut prepared.prior, idx, mean, v0)?;
        let mut extra = String::new();
        let _ = write!(
            extra,
            "TransportPolicy {} for source={} target=row_weighting; reweighted mean={mean:.6} \
             source_v0={source_v0:.6} reweight_v0={reweight_v0:.6} ess={:.3} support={}/{}",
            policy.id(),
            src.id,
            overlap.kish_ess,
            overlap.n_support,
            overlap.n_rows,
        );
        prepared.prior.restrictions.push(PriorAssumption {
            id: Arc::from(TARGET_ASSUMPTION_ID),
            description: Arc::from(extra),
        });
        if let Some(kernel) = target.kernel {
            prepared.prior.restrictions.push(kernel.assumption());
        }
        out.push(prepared);
    }
    Ok((out, overlap))
}

/// Transport toward a weighted target, then compose.
///
/// # Errors
///
/// Overlap refusal, transport failures, or composition failures.
pub fn compose_with_weighted_target_transport(
    sources: &[ExternalPriorSource],
    baseline: &PriorSet,
    policy: TransportPolicy,
    target: &WeightedTarget,
    coef_index: Option<usize>,
) -> Result<(ComposedPrior, TargetOverlap), ProbError> {
    let (prepared, overlap) = apply_weighted_target_transport(sources, policy, target, coef_index)?;
    let alphas: Vec<f64> = prepared.iter().map(|s| s.weight.alpha).collect();
    let composed = compose_external_priors_with_alphas(&prepared, &alphas, &alphas, baseline)?;
    Ok((composed, overlap))
}

/// One bandwidth of a kernel sensitivity report.
#[derive(Clone, Debug, PartialEq)]
pub struct BandwidthSensitivityRow {
    /// Declared bandwidth.
    pub bandwidth: f64,
    /// Kernel-weighted mean of the unit effects (always reported when any row
    /// carries weight).
    pub weighted_mean: Option<f64>,
    /// Sampling variance of that weighted mean.
    pub mean_variance: Option<f64>,
    /// Support and overlap at this bandwidth.
    pub overlap: Option<TargetOverlap>,
    /// Whether the overlap check passes, so a transport at this bandwidth runs.
    pub overlap_ok: bool,
    /// Next-attempt message when the overlap check fails.
    pub refusal: Option<String>,
}

/// How the transported estimate moves across a set of declared bandwidths.
///
/// A bandwidth whose weighting fails the overlap check is reported with its
/// refusal message rather than dropped, so the report shows where transport
/// stops being supported.
///
/// # Errors
///
/// Empty bandwidth set, an invalid bandwidth, or invalid effects / distances.
pub fn kernel_bandwidth_sensitivity(
    family: KernelFamily,
    bandwidths: &[f64],
    distances: &[f64],
    unit_effects: &[f64],
    min_ess_fraction: f64,
) -> Result<Vec<BandwidthSensitivityRow>, TransportError> {
    if bandwidths.is_empty() {
        return Err(TransportError::InvalidWeights {
            message: "bandwidth sensitivity requires at least one bandwidth",
        });
    }
    if unit_effects.len() != distances.len() {
        return Err(TransportError::InvalidWeights {
            message: "unit_effects and distances length mismatch",
        });
    }
    let mut rows = Vec::with_capacity(bandwidths.len());
    for &bandwidth in bandwidths {
        let kernel = TransportKernel::new(family, bandwidth)?;
        let weights = kernel.weights(distances)?;
        let overlap = TargetOverlap::from_weights(&weights);
        if weights.iter().sum::<f64>() <= 0.0 {
            rows.push(BandwidthSensitivityRow {
                bandwidth,
                weighted_mean: None,
                mean_variance: None,
                overlap: Some(overlap),
                overlap_ok: false,
                refusal: Some(overlap.message(min_ess_fraction)),
            });
            continue;
        }
        let adj = TransportAdjustment::new(unit_effects.to_vec(), weights)?;
        let failure = overlap.failure(min_ess_fraction);
        rows.push(BandwidthSensitivityRow {
            bandwidth,
            weighted_mean: Some(adj.weighted_moments().0),
            mean_variance: Some(adj.reweighted_mean_variance()),
            overlap: Some(overlap),
            overlap_ok: failure.is_none(),
            refusal: failure,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::external_prior::ExternalPriorWeight;
    use crate::prior::{GaussianCoefficientPrior, PriorSpec};

    fn source(mean: f64) -> ExternalPriorSource {
        let mut p = PriorSet::new();
        p.push(PriorSpec::GaussianCoefficients(
            GaussianCoefficientPrior::shared(1, mean, 1.0).unwrap(),
        ));
        ExternalPriorSource {
            id: Arc::from("s"),
            prior: p,
            weight: ExternalPriorWeight::power(0.5).unwrap(),
            ess: None,
        }
    }

    #[test]
    fn kernels_match_hand_values() {
        let g = TransportKernel::new(KernelFamily::Gaussian, 2.0).unwrap();
        assert!((g.weight(2.0) - (-0.5_f64).exp()).abs() < 1e-15);
        let e = TransportKernel::new(KernelFamily::Epanechnikov, 2.0).unwrap();
        assert!((e.weight(1.0) - 0.75).abs() < 1e-15);
        assert_eq!(e.weight(2.0), 0.0);
        let t = TransportKernel::new(KernelFamily::Tricube, 2.0).unwrap();
        assert!((t.weight(1.0) - (1.0 - 0.125_f64).powi(3)).abs() < 1e-15);
    }

    #[test]
    fn kernel_identity_enters_the_assumption_id() {
        let a = TransportKernel::new(KernelFamily::Gaussian, 1.0).unwrap().assumption();
        let b = TransportKernel::new(KernelFamily::Gaussian, 1.5).unwrap().assumption();
        let c = TransportKernel::new(KernelFamily::Tricube, 1.0).unwrap().assumption();
        assert_ne!(a.id, b.id);
        assert_ne!(a.id, c.id);
    }

    #[test]
    fn thin_overlap_is_refused_with_a_next_attempt() {
        let effects: Vec<f64> = (0..100).map(f64::from).collect();
        let dist: Vec<f64> = (0..100).map(f64::from).collect();
        let k = TransportKernel::new(KernelFamily::Gaussian, 0.5).unwrap();
        let target = WeightedTarget::from_kernel(effects, &dist, k).unwrap();
        let err = apply_weighted_target_transport(
            &[source(0.0)],
            TransportPolicy::InvariantConditionalOutcome,
            &target,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("Next attempt"));
    }

    #[test]
    fn broad_overlap_reweights_the_mean() {
        let effects: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
        let weights: Vec<f64> = (0..100).map(|i| if i < 50 { 1.0 } else { 0.0 }).collect();
        let target = WeightedTarget::from_row_weights(effects.clone(), weights).unwrap();
        let (out, overlap) = apply_weighted_target_transport(
            &[source(0.0)],
            TransportPolicy::InvariantConditionalOutcome,
            &target,
            None,
        )
        .unwrap();
        assert!((overlap.kish_ess - 50.0).abs() < 1e-12);
        let mean = out[0].prior.gaussian_coefficients().unwrap().mean[0];
        let hand = effects[..50].iter().sum::<f64>() / 50.0;
        assert!((mean - hand).abs() < 1e-12);
    }
}
