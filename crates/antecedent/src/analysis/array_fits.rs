//! The treatment-propensity nuisance product: a retained, sealed fit with a replay artifact.
//!
//! A prepared average-effect study knows its treatment and certified adjustment set;
//! [`crate::PreparedStudy::treatment_score_posterior`] fits the Laplace posterior of the logistic
//! treatment model on those columns and returns it as an [`ArrayFit`]. The product is a posterior
//! of a nuisance, never of an effect, it is unlicensed with calibration unmeasured, and the
//! result and the artifact say so.
//!
//! [`ArrayFit::consume`] refits from the exported artifact and refuses any resealed change of
//! request, array or stored value. The artifact is a different kind from the cross-fitted
//! effect-score table, which does not read it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io::array_fit_artifact::{
    ARRAY_FIT_ARTIFACT_FEATURE, ARRAY_FIT_ARTIFACT_VERSION, ARRAY_FIT_CALIBRATION,
    ARRAY_FIT_LICENSING, ArrayFitArtifactWire, ArrayFitConsumeLimits, ArrayFitError,
    ArrayFitRequestWire, ArrayFitResultWire, ID_BAYESIAN_PROPENSITY,
    ID_BAYESIAN_TEMPORAL_PROPENSITY, ID_TEMPORAL_PROPENSITY, NamedValueWire, PenalizedScoreWire,
    PropensityDesignWire, PropensityPosteriorWire, PropensityScorePosteriorWire,
};

use antecedent_estimate::EstimationError;
use std::sync::{Arc, Mutex};
use antecedent_core::{ExecutionContext, execution::{AdmissionLease, AdmissionError}};

use crate::error::CausalError;

/// A retained, sealed nuisance product.
#[derive(Clone, Debug)]
pub struct ArrayFit {
    artifact: Arc<ArrayFitArtifactWire>,
    encoding: Arc<Mutex<Option<Arc<RetainedArrayFitEncoding>>>>,
}

/// An explicitly materialized encoding with the exact native byte reservation retained.
#[derive(Debug)]
pub struct RetainedArrayFitEncoding {
    bytes: Vec<u8>,
    _lease: Arc<AdmissionLease>,
}
impl RetainedArrayFitEncoding {
    /// Borrow immutable encoded bytes without another serialization or ownership copy.
    #[must_use]
    pub fn bytes(&self) -> &[u8] { &self.bytes }
}

impl PartialEq for ArrayFit {
    fn eq(&self, other: &Self) -> bool { self.artifact == other.artifact }
}

impl ArrayFit {
    /// Fit a declared request and seal it with its digest.
    ///
    /// # Errors
    /// A refused design or prior, a request above the default consumer bounds, or a Laplace fit
    /// that does not publish a posterior (`array_fit.fit_refused` or the estimator's own detail).
    pub fn fit(request: ArrayFitRequestWire) -> Result<Self, ArrayFitError> {
        Ok(Self { artifact: Arc::new(ArrayFitArtifactWire::seal(request)?), encoding: Arc::default() })
    }

    /// Consume an exported artifact by refitting from its embedded arrays.
    ///
    /// # Errors
    /// Corruption, an unknown version, a changed digest or request, or a stored result that
    /// does not replay.
    pub fn consume(bytes: &[u8], limits: ArrayFitConsumeLimits) -> Result<Self, ArrayFitError> {
        Ok(Self { artifact: Arc::new(ArrayFitArtifactWire::consume_typed(bytes, limits)?), encoding: Arc::default() })
    }

    /// Export the artifact.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, ArrayFitError> {
        Ok(self.encoded()?.bytes().to_vec())
    }

    /// Independently replay borrowed immutable bytes under an explicit producer expectation.
    /// The resulting request/result owns its decoded storage; it retains no live study.
    ///
    /// # Errors
    /// The independent consumer's errors or a producer expectation mismatch.
    pub fn consume_expected(bytes: &[u8], limits: ArrayFitConsumeLimits, expected: Option<&str>) -> Result<Self, ArrayFitError> {
        Ok(Self { artifact: Arc::new(ArrayFitArtifactWire::consume_typed_expected(bytes,limits,expected)?), encoding: Arc::default() })
    }

    /// Explicit serialization under unlimited default native resource policy.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn encoded(&self) -> Result<Arc<RetainedArrayFitEncoding>, ArrayFitError> {
        self.encoded_with_context(&ExecutionContext::production(0,1))
    }

    /// Materialize once after reserving the exact serialized native byte count.
    /// Failed admission is never cached: a later eligible policy may retry.
    ///
    /// # Errors
    /// Cancellation, actual byte admission, size overflow or encoding failure.
    pub fn encoded_with_context(&self, ctx: &ExecutionContext) -> Result<Arc<RetainedArrayFitEncoding>, ArrayFitError> {
        use std::io::Write;
        if ctx.cancellation.is_cancelled() { return Err(AdmissionError::Cancelled.into()); }
        let mut cache=self.encoding.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(encoded)=&*cache { return Ok(Arc::clone(encoded)); }
        // Both passes observe cancellation on every write, so a long encoding stops promptly.
        struct Checked<'a, W> { inner: W, written: usize, ctx: &'a ExecutionContext }
        impl<W: Write> Write for Checked<'_, W> {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.ctx.cancellation.is_cancelled() {
                    return Err(std::io::Error::other("array-fit encoding cancelled"));
                }
                let n=self.inner.write(bytes)?;
                self.written=self.written.checked_add(n).ok_or_else(|| std::io::Error::other("array-fit encoding size overflow"))?;
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
        }
        let encode=|writer: &mut dyn Write| -> Result<usize, ArrayFitError> {
            let mut checked=Checked { inner:writer, written:0, ctx };
            match self.artifact.export_into(&mut checked) {
                Ok(()) => Ok(checked.written),
                Err(_) if ctx.cancellation.is_cancelled() => Err(AdmissionError::Cancelled.into()),
                Err(error) => Err(ArrayFitError::Undecodable(error.to_string())),
            }
        };
        let counted=encode(&mut std::io::sink())?;
        let bytes=u64::try_from(counted).map_err(|_|ArrayFitError::Admission(AdmissionError::Overflow))?;
        let lease=Arc::new(ctx.acquire_admission(0,bytes)?);
        let mut encoding=Vec::with_capacity(counted);
        encode(&mut encoding)?;
        if ctx.cancellation.is_cancelled() { return Err(AdmissionError::Cancelled.into()); }
        let encoded=Arc::new(RetainedArrayFitEncoding { bytes:encoding,_lease:lease });
        *cache=Some(Arc::clone(&encoded));
        Ok(encoded)
    }

    /// Whether an explicit native encoding is retained.
    #[must_use]
    pub fn has_encoding(&self) -> bool { self.encoding.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_some() }

    /// The estimator id.
    #[must_use]
    pub fn estimator_id(&self) -> &str {
        &self.artifact.result.estimator_id
    }

    /// The stored request, arrays included.
    #[must_use]
    pub fn request(&self) -> &ArrayFitRequestWire {
        &self.artifact.request
    }

    /// The stored result.
    #[must_use]
    pub fn result(&self) -> &ArrayFitResultWire {
        &self.artifact.result
    }

    /// One named real of the result.
    #[must_use]
    pub fn value(&self, name: &str) -> Option<f64> {
        self.artifact.result.values.iter().find(|v| v.name == name).map(|v| v.value)
    }

    /// Posterior draws of the target; empty when the result is not a posterior of the target.
    #[must_use]
    pub fn draws(&self) -> &[f64] {
        &self.artifact.result.draws
    }

    /// The request digest.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.artifact.request_digest
    }

    /// The underlying artifact.
    #[must_use]
    pub fn artifact(&self) -> &ArrayFitArtifactWire {
        &self.artifact
    }
}

impl From<ArrayFitError> for CausalError {
    fn from(error: ArrayFitError) -> Self {
        if let ArrayFitError::Admission(error)=error {
            return if matches!(error,AdmissionError::Cancelled) { CausalError::Cancelled { stage:"array_fit" } }
                else { CausalError::Resource { message:error.to_string() } };
        }
        let (code, detail) = error.refusal();
        let explanation = match &error {
            ArrayFitError::Refused { message, .. } => message.clone(),
            other => other.to_string(),
        };
        CausalError::Estimate(EstimationError::refused_detail(code, detail, explanation))
    }
}

/// How a prepared study's treatment propensity is summarized as a nuisance product.
///
/// The prior is `N(0, prior_sd^2)` on every logit coefficient, the intercept included, and the
/// product is a posterior of that nuisance, never of an effect.
#[derive(Clone, Debug, PartialEq)]
pub struct TreatmentScoreSpec {
    /// Standard deviation of the Gaussian prior on every coefficient.
    pub prior_sd: f64,
    /// Posterior draws.
    pub draws: usize,
    /// Seed of the draw stream.
    pub seed: u64,
    /// Positivity band `(clip, 1 - clip)` of the outside-band probability.
    pub clip: f64,
    /// Rows of the prepared data whose score posterior is reported.
    pub summary_rows: Vec<usize>,
}

/// The treatment model fitted on the independent-subject rows of a prepared panel study.
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalScoreModel {
    /// Ridge-penalized logistic fit; the intercept is unpenalized and the score is the inverse
    /// link at the penalized mode, with no uncertainty.
    Penalized {
        /// Penalty on half the squared norm of the slopes.
        penalty: f64,
    },
    /// Laplace posterior under independent `N(0, sd^2)` priors on the intercept and the slopes.
    Posterior {
        /// Prior standard deviation of the intercept.
        intercept_sd: f64,
        /// Prior standard deviation of each slope.
        slope_sd: f64,
        /// Posterior draws.
        draws: usize,
        /// Seed of the draw stream.
        seed: u64,
    },
}

/// How a prepared panel study's temporal treatment propensity is summarized as a nuisance
/// product: the model and the lag histories whose scores are reported.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalTreatmentScoreSpec {
    /// The treatment model.
    pub model: TemporalScoreModel,
    /// Lag histories, one value per adjustment lag column, whose scores are reported.
    pub histories: Vec<Vec<f64>>,
}

#[cfg(test)]
mod native_owner_regression {
    use super::*;
    use antecedent_core::execution::AdmissionPool;
    fn product()->ArrayFit {
        let mut x=vec![1.0;200]; x.extend((0..200).map(|i|if i>=100 {1.0}else{0.0}));
        let treatment=(0..200).map(|i|if i<30 || (100..160).contains(&i) {1.0}else{0.0}).collect();
        ArrayFit::fit(ArrayFitRequestWire::PropensityPosterior(PropensityPosteriorWire {x_colmajor:x,treatment,nrows:200,prior_sd:2.0,draws:8,seed:13,rows:vec![vec![1.0,0.0]],clip:0.05})).unwrap()
    }
    #[test]
    fn cloned_products_share_native_arrays_and_exact_encoding_lease_until_last_owner() {
        let fit=product(); let branch=fit.clone();
        assert!(Arc::ptr_eq(&fit.artifact,&branch.artifact));
        assert!(Arc::ptr_eq(&fit.encoding,&branch.encoding));
        assert!(!fit.has_encoding());
        let zero=ExecutionContext::for_tests(13).with_memory_limit(Some(0));
        assert!(matches!(fit.encoded_with_context(&zero),Err(ArrayFitError::Admission(_))));
        assert!(!fit.has_encoding());
        let pool=AdmissionPool::fixed(1,1_000_000);
        let ctx=ExecutionContext::for_tests(13).with_admission(pool.clone());
        let encoded=fit.encoded_with_context(&ctx).unwrap();
        let retained=branch.encoded_with_context(&ctx).unwrap();
        assert!(Arc::ptr_eq(&encoded,&retained));
        let length=u64::try_from(encoded.bytes().len()).unwrap();
        assert_eq!(pool.available().bytes(),1_000_000-length);
        let loaded=ArrayFit::consume_expected(encoded.bytes(),ArrayFitConsumeLimits::default(),Some(fit.digest())).unwrap();
        assert_eq!(loaded,fit);
        assert!(loaded.draws().is_empty());
        assert_eq!(loaded.result().target,"propensity_score_nuisance");
        drop(fit);drop(branch);drop(encoded);
        assert_eq!(pool.available().bytes(),1_000_000-length);
        drop(retained);
        assert_eq!(pool.available().bytes(),1_000_000);
    }
}
