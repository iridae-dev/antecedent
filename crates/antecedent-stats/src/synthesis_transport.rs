//! Numerical synthesis of transported claims with native target evidence.
//!
//! A claim carried to a target population is not native target data: it has
//! the uncertainty of the source estimate and, on top of it, the uncertainty
//! of the carry itself (an unverified invariance, an estimated transport
//! weight). This module puts that extra uncertainty into the synthesis exactly
//! once per transport, and refuses to treat one transport applied to several
//! claims as several independent ones.
//!
//! # Model
//!
//! A summary `i` of origin *native* is `y_i = theta + e_i`, `Var e_i = v_i`.
//! A summary of origin *transported through* `t` is
//! `y_i = theta + b_t + e_i`, where the transport error `b_t` has variance
//! `tau_t^2`, **shared by every summary carried through the same transport
//! `t`**. Hence
//!
//! ```text
//! Var y_i      = v_i + tau_t^2
//! Cov(y_i,y_j) = Cov(e_i, e_j) + tau_t^2     (same t, i != j)
//! ```
//!
//! and the fixed-effect estimate is generalized least squares under that
//! covariance. Counting `tau_t^2` as independent per summary would understate
//! the variance of the pooled value; dropping it would treat the carry as
//! free. Neither is done.
//!
//! The routes are point-and-variance only. The variance is the model variance
//! under the declared covariance; it is not a calibrated interval, and the
//! dependence between source estimates and native estimates is the caller's
//! declaration (the evidence ledger decides whether the *data* overlap).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::synthesis::{
    HierarchicalPosterior, MeanPrior, PooledEstimate, SynthesisError, TauPrior, fixed_effect_gls,
    hierarchical_normal_borrowing,
};

/// Version of the transported-synthesis semantics.
pub const TRANSPORTED_SYNTHESIS_VERSION: u32 = 1;
/// Most summaries one transported synthesis may combine.
pub const MAX_TRANSPORTED_SUMMARIES: usize = 256;

const DIAGONAL_RELATIVE_TOLERANCE: f64 = 1e-9;

/// Where a summary's value comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum SummaryOrigin {
    /// Estimated from the target population's own evidence.
    Native,
    /// Carried to the target through one transport.
    Transported {
        /// Identity of the transport (equal keys are one transport and share
        /// one transport error).
        transport_key: String,
        /// Variance `tau^2` of the transport error `b_t` (finite, `>= 0`). A
        /// transport with no declared uncertainty is not zero; it is not a
        /// summary of this kind and the caller refuses it upstream.
        transport_variance: f64,
    },
}

/// One estimate to synthesize.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportAwareSummary {
    /// Point estimate of the common target quantity.
    pub estimate: f64,
    /// Sampling variance of the estimate (finite, `> 0`).
    pub sampling_variance: f64,
    /// Native or transported.
    pub origin: SummaryOrigin,
}

/// Why a transported synthesis was refused.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum TransportedSynthesisError {
    /// The underlying synthesis routine refused its input.
    #[error("synthesis refused: {0}")]
    Synthesis(#[from] SynthesisError),
    /// A transport variance is negative or not finite.
    #[error("transport variance of summary {index} must be finite and nonnegative")]
    InvalidTransportVariance {
        /// Index of the summary.
        index: usize,
    },
    /// Summaries carried through one transport disagree on its variance.
    #[error("transport `{key}` is declared with more than one variance")]
    InconsistentTransportVariance {
        /// The transport key.
        key: String,
    },
    /// One transport serves several summaries but the route cannot carry the
    /// shared error.
    #[error("transport `{key}` serves several summaries; pool them with the shared covariance")]
    SharedTransportNeedsCovariance {
        /// The transport key.
        key: String,
    },
    /// The supplied sampling covariance disagrees with the declared sampling
    /// variances on the diagonal.
    #[error("sampling covariance diagonal disagrees with the sampling variance of summary {index}")]
    SamplingCovarianceDiagonalMismatch {
        /// Index of the summary.
        index: usize,
    },
    /// The prior route needs a transported prior and native likelihood
    /// summaries.
    #[error("{0}")]
    PriorRoleMismatch(&'static str),
}

fn validate(summaries: &[TransportAwareSummary]) -> Result<(), TransportedSynthesisError> {
    if summaries.is_empty() {
        return Err(SynthesisError::TooFewStudies { needed: 1, got: 0 }.into());
    }
    if summaries.len() > MAX_TRANSPORTED_SUMMARIES {
        return Err(
            SynthesisError::Shape("more summaries than the transported-synthesis bound").into()
        );
    }
    let mut variance_of_key: BTreeMap<&str, f64> = BTreeMap::new();
    for (index, s) in summaries.iter().enumerate() {
        if !s.estimate.is_finite() {
            return Err(SynthesisError::NonFinite("estimate").into());
        }
        if !s.sampling_variance.is_finite() || s.sampling_variance <= 0.0 {
            return Err(SynthesisError::NonPositiveVariance("sampling variance").into());
        }
        if let SummaryOrigin::Transported { transport_key, transport_variance } = &s.origin {
            if !transport_variance.is_finite() || *transport_variance < 0.0 {
                return Err(TransportedSynthesisError::InvalidTransportVariance { index });
            }
            if let Some(previous) = variance_of_key.insert(transport_key, *transport_variance) {
                if previous.to_bits() != transport_variance.to_bits() {
                    return Err(TransportedSynthesisError::InconsistentTransportVariance {
                        key: transport_key.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Covariance of the summaries with the extra transport uncertainty counted
/// once per transport: `diag += tau_t^2` for each transported summary and
/// `off-diagonal += tau_t^2` for each pair carried through the same `t`.
///
/// `sampling_covariance`, when supplied, is the `k x k` covariance of the
/// sampling errors (its diagonal must equal the declared sampling variances);
/// otherwise the sampling errors are independent.
///
/// # Errors
/// Shape, finiteness, variance or transport-variance refusals.
pub fn transport_aware_covariance(
    summaries: &[TransportAwareSummary],
    sampling_covariance: Option<&[f64]>,
) -> Result<Vec<f64>, TransportedSynthesisError> {
    validate(summaries)?;
    let k = summaries.len();
    let mut cov = vec![0.0_f64; k * k];
    match sampling_covariance {
        Some(base) => {
            if base.len() != k * k {
                return Err(SynthesisError::Shape("sampling covariance is not k x k").into());
            }
            if base.iter().any(|v| !v.is_finite()) {
                return Err(SynthesisError::NonFinite("sampling covariance").into());
            }
            for (index, s) in summaries.iter().enumerate() {
                let d = base[index * k + index];
                if (d - s.sampling_variance).abs()
                    > DIAGONAL_RELATIVE_TOLERANCE * s.sampling_variance
                {
                    return Err(TransportedSynthesisError::SamplingCovarianceDiagonalMismatch {
                        index,
                    });
                }
            }
            cov.copy_from_slice(base);
        }
        None => {
            for (i, s) in summaries.iter().enumerate() {
                cov[i * k + i] = s.sampling_variance;
            }
        }
    }
    for (i, a) in summaries.iter().enumerate() {
        let SummaryOrigin::Transported { transport_key: ka, transport_variance: tau_a } = &a.origin
        else {
            continue;
        };
        cov[i * k + i] += tau_a;
        for (j, b) in summaries.iter().enumerate().skip(i + 1) {
            if let SummaryOrigin::Transported { transport_key: kb, .. } = &b.origin {
                if ka == kb {
                    cov[i * k + j] += tau_a;
                    cov[j * k + i] += tau_a;
                }
            }
        }
    }
    Ok(cov)
}

/// Fixed-effect pooling of native and transported summaries.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedPooled {
    /// The generalized-least-squares pool under the full covariance.
    pub pooled: PooledEstimate,
    /// The covariance used (row-major, `k x k`).
    pub covariance: Vec<f64>,
    /// Variance of the pool had every transport error been zero.
    pub variance_without_transport: f64,
    /// `pooled.variance - variance_without_transport` (nonnegative): what the
    /// transports cost, counted once.
    pub transport_variance_added: f64,
    /// Distinct transports that contributed, sorted.
    pub transport_keys: Vec<String>,
}

/// Pool native and transported summaries with the transport error counted
/// once per transport.
///
/// # Errors
/// As [`transport_aware_covariance`] and
/// [`crate::synthesis::fixed_effect_gls`] (a singular covariance is refused).
pub fn pool_transported(
    summaries: &[TransportAwareSummary],
    sampling_covariance: Option<&[f64]>,
) -> Result<TransportedPooled, TransportedSynthesisError> {
    let covariance = transport_aware_covariance(summaries, sampling_covariance)?;
    let estimates: Vec<f64> = summaries.iter().map(|s| s.estimate).collect();
    let pooled = fixed_effect_gls(&estimates, &covariance)?;

    let stripped: Vec<TransportAwareSummary> = summaries
        .iter()
        .map(|s| TransportAwareSummary { origin: SummaryOrigin::Native, ..s.clone() })
        .collect();
    let baseline = transport_aware_covariance(&stripped, sampling_covariance)?;
    let without = fixed_effect_gls(&estimates, &baseline)?;
    let keys: BTreeSet<&str> = summaries
        .iter()
        .filter_map(|s| match &s.origin {
            SummaryOrigin::Transported { transport_key, .. } => Some(transport_key.as_str()),
            SummaryOrigin::Native => None,
        })
        .collect();
    Ok(TransportedPooled {
        transport_variance_added: (pooled.variance - without.variance).max(0.0),
        variance_without_transport: without.variance,
        transport_keys: keys.into_iter().map(str::to_owned).collect(),
        covariance,
        pooled,
    })
}

/// Hierarchical borrowing over native and transported summaries, each
/// transported summary's variance inflated by its transport variance.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedBorrowing {
    /// Posterior of the common mean and of each summary's effect.
    pub posterior: HierarchicalPosterior,
    /// The variances the borrowing used (`v_i + tau_t^2` for transported).
    pub effective_variances: Vec<f64>,
}

/// Hierarchical normal-normal borrowing across native and transported
/// summaries.
///
/// Each transported summary enters with variance `v_i + tau_t^2`. The
/// hierarchical model treats the summaries as conditionally independent, so a
/// transport that serves more than one summary is refused
/// ([`TransportedSynthesisError::SharedTransportNeedsCovariance`]): its shared
/// error would be counted once per summary. Pool those with
/// [`pool_transported`] instead.
///
/// # Errors
/// As [`hierarchical_normal_borrowing`], and the transport refusals above.
pub fn borrow_transported(
    summaries: &[TransportAwareSummary],
    mean_prior: MeanPrior,
    tau_prior: &TauPrior,
) -> Result<TransportedBorrowing, TransportedSynthesisError> {
    validate(summaries)?;
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for s in summaries {
        if let SummaryOrigin::Transported { transport_key, .. } = &s.origin {
            if !seen.insert(transport_key) {
                return Err(TransportedSynthesisError::SharedTransportNeedsCovariance {
                    key: transport_key.clone(),
                });
            }
        }
    }
    let effective_variances: Vec<f64> = summaries
        .iter()
        .map(|s| match &s.origin {
            SummaryOrigin::Native => s.sampling_variance,
            SummaryOrigin::Transported { transport_variance, .. } => {
                s.sampling_variance + transport_variance
            }
        })
        .collect();
    let estimates: Vec<f64> = summaries.iter().map(|s| s.estimate).collect();
    let posterior =
        hierarchical_normal_borrowing(&estimates, &effective_variances, mean_prior, tau_prior)?;
    Ok(TransportedBorrowing { posterior, effective_variances })
}

/// Posterior of a normal-normal update whose prior is a transported claim.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedPriorUpdate {
    /// Posterior mean.
    pub mean: f64,
    /// Posterior variance.
    pub variance: f64,
    /// Prior variance including the transport variance (counted once).
    pub prior_variance: f64,
    /// Share of the posterior precision contributed by the prior.
    pub prior_weight: f64,
    /// Precision contributed by the native likelihood summaries.
    pub likelihood_precision: f64,
}

/// Update native likelihood summaries with a transported claim as the prior.
///
/// The prior is `N(estimate, v + tau_t^2)`; the native summaries are pooled by
/// inverse variance as independent likelihood terms. The prior's data must be
/// disjoint from the likelihood's: that is the evidence ledger's check, made
/// before this is called.
///
/// # Errors
/// [`TransportedSynthesisError::PriorRoleMismatch`] when `prior` is native or
/// a likelihood summary is transported, plus the variance refusals.
pub fn transported_prior_update(
    prior: &TransportAwareSummary,
    likelihood: &[TransportAwareSummary],
) -> Result<TransportedPriorUpdate, TransportedSynthesisError> {
    let SummaryOrigin::Transported { transport_variance, .. } = &prior.origin else {
        return Err(TransportedSynthesisError::PriorRoleMismatch(
            "the prior of this route is a transported claim; a native prior is a native source",
        ));
    };
    if likelihood.is_empty() {
        return Err(SynthesisError::TooFewStudies { needed: 1, got: 0 }.into());
    }
    if likelihood.iter().any(|s| !matches!(s.origin, SummaryOrigin::Native)) {
        return Err(TransportedSynthesisError::PriorRoleMismatch(
            "the likelihood of the prior route is native target evidence",
        ));
    }
    let mut all = Vec::with_capacity(likelihood.len() + 1);
    all.push(prior.clone());
    all.extend(likelihood.iter().cloned());
    validate(&all)?;

    let prior_variance = prior.sampling_variance + transport_variance;
    let prior_precision = 1.0 / prior_variance;
    let mut precision = 0.0;
    let mut weighted = 0.0;
    for s in likelihood {
        let p = 1.0 / s.sampling_variance;
        precision += p;
        weighted += p * s.estimate;
    }
    let total = prior_precision + precision;
    Ok(TransportedPriorUpdate {
        mean: (prior_precision * prior.estimate + weighted) / total,
        variance: 1.0 / total,
        prior_variance,
        prior_weight: prior_precision / total,
        likelihood_precision: precision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native(estimate: f64, variance: f64) -> TransportAwareSummary {
        TransportAwareSummary {
            estimate,
            sampling_variance: variance,
            origin: SummaryOrigin::Native,
        }
    }

    fn carried(estimate: f64, variance: f64, key: &str, tau2: f64) -> TransportAwareSummary {
        TransportAwareSummary {
            estimate,
            sampling_variance: variance,
            origin: SummaryOrigin::Transported {
                transport_key: key.to_owned(),
                transport_variance: tau2,
            },
        }
    }

    fn near(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn one_shared_transport_is_counted_once() {
        // Cov = [[1,0,0],[0,1.5,.5],[0,.5,1.5]]; the transported pair has
        // precision 1 together (inverse of the 2x2 block times ones is
        // (.5,.5)), the native one has 1: weights (.5,.25,.25).
        let s = [native(1.0, 1.0), carried(2.0, 1.0, "t", 0.5), carried(3.0, 1.0, "t", 0.5)];
        let pooled = pool_transported(&s, None).unwrap();
        near(pooled.pooled.estimate, 1.75);
        near(pooled.pooled.variance, 0.5);
        near(pooled.pooled.weights[0], 0.5);
        near(pooled.pooled.weights[1], 0.25);
        // Ignoring transport entirely would give 1/3; counting the shared
        // error as two independent ones would give 3/7. Neither is it.
        near(pooled.variance_without_transport, 1.0 / 3.0);
        near(pooled.transport_variance_added, 0.5 - 1.0 / 3.0);
        assert_eq!(pooled.transport_keys, vec!["t".to_owned()]);
    }

    #[test]
    fn distinct_transports_add_independent_errors() {
        let s = [native(1.0, 1.0), carried(2.0, 1.0, "a", 0.5), carried(3.0, 1.0, "b", 0.5)];
        let pooled = pool_transported(&s, None).unwrap();
        // Diagonal [1, 1.5, 1.5]: precisions 1, 2/3, 2/3.
        near(pooled.pooled.variance, 3.0 / 7.0);
        near(pooled.pooled.estimate, 13.0 / 7.0);
        assert_eq!(pooled.transport_keys.len(), 2);
    }

    #[test]
    fn zero_transport_variance_is_the_native_pool() {
        let s = [native(1.0, 1.0), carried(2.0, 1.0, "t", 0.0)];
        let pooled = pool_transported(&s, None).unwrap();
        near(pooled.pooled.estimate, 1.5);
        near(pooled.pooled.variance, 0.5);
        near(pooled.transport_variance_added, 0.0);
    }

    #[test]
    fn borrowing_with_a_point_mass_tau_is_inverse_variance_on_inflated_variances() {
        let s = [native(1.0, 1.0), carried(2.0, 1.0, "a", 0.5), carried(3.0, 1.0, "b", 0.5)];
        let tau = TauPrior { support: vec![0.0], probabilities: vec![1.0] };
        let b = borrow_transported(&s, MeanPrior::Flat, &tau).unwrap();
        near(b.posterior.mean, 13.0 / 7.0);
        near(b.posterior.variance, 3.0 / 7.0);
        assert_eq!(b.effective_variances, vec![1.0, 1.5, 1.5]);
    }

    #[test]
    fn borrowing_refuses_a_transport_that_serves_two_summaries() {
        let s = [native(1.0, 1.0), carried(2.0, 1.0, "t", 0.5), carried(3.0, 1.0, "t", 0.5)];
        let tau = TauPrior { support: vec![0.0], probabilities: vec![1.0] };
        let err = borrow_transported(&s, MeanPrior::Flat, &tau).unwrap_err();
        assert_eq!(
            err,
            TransportedSynthesisError::SharedTransportNeedsCovariance { key: "t".to_owned() }
        );
    }

    #[test]
    fn transported_prior_carries_its_extra_variance_once() {
        // Prior N(2, 1 + 0.5); one native summary y=1, v=1:
        // precision 1/1.5 + 1 = 5/3; variance 0.6; mean 0.6 (2/1.5 + 1) = 1.4.
        let prior = carried(2.0, 1.0, "t", 0.5);
        let update = transported_prior_update(&prior, &[native(1.0, 1.0)]).unwrap();
        near(update.mean, 1.4);
        near(update.variance, 0.6);
        near(update.prior_variance, 1.5);
        near(update.prior_weight, (1.0 / 1.5) / (5.0 / 3.0));
        near(update.likelihood_precision, 1.0);
    }

    #[test]
    fn roles_and_inputs_are_checked() {
        let prior = native(2.0, 1.0);
        assert!(matches!(
            transported_prior_update(&prior, &[native(1.0, 1.0)]),
            Err(TransportedSynthesisError::PriorRoleMismatch(_))
        ));
        let transported_likelihood = carried(1.0, 1.0, "u", 0.1);
        assert!(matches!(
            transported_prior_update(&carried(2.0, 1.0, "t", 0.5), &[transported_likelihood]),
            Err(TransportedSynthesisError::PriorRoleMismatch(_))
        ));
        assert!(matches!(
            pool_transported(&[native(1.0, 1.0), carried(2.0, 1.0, "t", -0.5)], None),
            Err(TransportedSynthesisError::InvalidTransportVariance { index: 1 })
        ));
        assert!(matches!(
            pool_transported(&[carried(1.0, 1.0, "t", 0.5), carried(2.0, 1.0, "t", 0.25)], None),
            Err(TransportedSynthesisError::InconsistentTransportVariance { .. })
        ));
        assert!(matches!(
            pool_transported(&[native(f64::NAN, 1.0)], None),
            Err(TransportedSynthesisError::Synthesis(SynthesisError::NonFinite(_)))
        ));
    }

    #[test]
    fn a_supplied_sampling_covariance_is_kept_and_checked() {
        // Native pair with sampling covariance 0.5 and one transported summary.
        let s = [native(1.0, 1.0), native(2.0, 1.0), carried(3.0, 1.0, "t", 1.0)];
        let base = [1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0];
        let cov = transport_aware_covariance(&s, Some(&base)).unwrap();
        near(cov[1], 0.5);
        near(cov[8], 2.0);
        let bad = [2.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(matches!(
            transport_aware_covariance(&s, Some(&bad)),
            Err(TransportedSynthesisError::SamplingCovarianceDiagonalMismatch { index: 0 })
        ));
    }
}
