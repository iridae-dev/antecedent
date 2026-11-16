//! Weighted target populations: a target declared as row weights or as a kernel over a
//! covariate distance, resolved against the analysed table.
//!
//! The declaration enters the study through [`crate::StudyBuilder::weighted_target`]. At build
//! the weights are read from the table, pass the overlap gate (positivity, the Kish effective
//! sample size floor, the fraction floor), and are registered as the study's custom target
//! distribution over the declared covariates, so the weighted standardization, the
//! `depends_on` check against the certified adjustment set, the per-arm weighted overlap gate
//! and the target-weights identity are those of that distribution. A study that declares no
//! weighted target runs none of this.
//!
//! The same declaration weights a trial-to-target transport ([`transport_trial_to_weighted_target`]):
//! the target is then the non-participant rows reweighted by it, behind the same gate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, DistributionRef, VariableId};
use antecedent_data::TableView;
use antecedent_prob::{
    KernelFamily, ProbError, TARGET_MIN_ESS_FRACTION, TargetOverlap, TransportError,
    TransportKernel, check_weights_overlap,
};

use crate::error::CausalError;

fn invalid(message: &'static str) -> CausalError {
    CausalError::from(ProbError::from(TransportError::InvalidWeights { message }))
}

/// How the weights of a target are given.
#[derive(Clone, Debug, PartialEq)]
pub enum WeightedTargetSpec {
    /// One finite non-negative weight per analysed row, in table order.
    RowWeights(Arc<[f64]>),
    /// A kernel over the scaled Euclidean distance of the covariates to a center.
    Kernel {
        /// Declared kernel family and bandwidth.
        kernel: TransportKernel,
        /// Center coordinates, one per covariate.
        center: Arc<[f64]>,
        /// Positive scales dividing the coordinate differences, one per covariate.
        scale: Arc<[f64]>,
    },
}

/// A declared weighted target population.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedTargetDeclaration {
    spec: WeightedTargetSpec,
    covariates: Arc<[VariableId]>,
    min_ess_fraction: f64,
}

impl WeightedTargetDeclaration {
    /// Target given as per-row weights that are a function of `depends_on`.
    ///
    /// # Errors
    ///
    /// Empty weights or `depends_on`, or a weight that is not finite and non-negative.
    pub fn row_weights(
        weights: impl Into<Arc<[f64]>>,
        depends_on: impl Into<Arc<[VariableId]>>,
    ) -> Result<Self, CausalError> {
        let weights = weights.into();
        let covariates = depends_on.into();
        if weights.is_empty() || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(invalid("target weights must be non-empty, finite and >= 0"));
        }
        if covariates.is_empty() {
            return Err(invalid("target weights must name the covariates they depend on"));
        }
        Ok(Self {
            spec: WeightedTargetSpec::RowWeights(weights),
            covariates,
            min_ess_fraction: TARGET_MIN_ESS_FRACTION,
        })
    }

    /// Target given as a kernel over the scaled distance of `covariates` to `center`.
    ///
    /// `scale` defaults to one per covariate.
    ///
    /// # Errors
    ///
    /// Empty covariates, mismatched lengths, a non-finite center, or a scale that is not
    /// finite and positive.
    pub fn kernel(
        kernel: TransportKernel,
        covariates: impl Into<Arc<[VariableId]>>,
        center: impl Into<Arc<[f64]>>,
        scale: Option<Arc<[f64]>>,
    ) -> Result<Self, CausalError> {
        let covariates = covariates.into();
        let center = center.into();
        if covariates.is_empty() {
            return Err(invalid("a kernel target needs at least one covariate"));
        }
        if center.len() != covariates.len() || center.iter().any(|c| !c.is_finite()) {
            return Err(invalid(
                "a kernel target needs one finite center coordinate per covariate",
            ));
        }
        let scale = scale.unwrap_or_else(|| vec![1.0; covariates.len()].into());
        if scale.len() != covariates.len() || scale.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(invalid("a kernel target needs one finite positive scale per covariate"));
        }
        Ok(Self {
            spec: WeightedTargetSpec::Kernel { kernel, center, scale },
            covariates,
            min_ess_fraction: TARGET_MIN_ESS_FRACTION,
        })
    }

    /// Declare the ESS-fraction floor; the absolute Kish floor of 20 rows still applies.
    ///
    /// # Errors
    ///
    /// A fraction outside `[0, 1]`.
    pub fn with_min_ess_fraction(mut self, fraction: f64) -> Result<Self, CausalError> {
        if !(0.0..=1.0).contains(&fraction) {
            return Err(invalid("min_ess_fraction must lie in [0, 1]"));
        }
        self.min_ess_fraction = fraction;
        Ok(self)
    }

    /// The declared weighting.
    #[must_use]
    pub const fn spec(&self) -> &WeightedTargetSpec {
        &self.spec
    }

    /// The covariates the weights are a function of.
    #[must_use]
    pub fn covariates(&self) -> &[VariableId] {
        &self.covariates
    }

    /// The declared ESS-fraction floor.
    #[must_use]
    pub const fn min_ess_fraction(&self) -> f64 {
        self.min_ess_fraction
    }

    /// The per-row weights of this declaration over `table`.
    ///
    /// # Errors
    ///
    /// A covariate that is not a numeric column of the table, a missing or non-finite
    /// covariate value, or explicit weights of another length than the table.
    pub fn weights<T: TableView + ?Sized>(&self, table: &T) -> Result<Vec<f64>, CausalError> {
        let rows = table.row_count();
        match &self.spec {
            WeightedTargetSpec::RowWeights(weights) => {
                if weights.len() != rows {
                    return Err(invalid("target weights need one entry per analysed row"));
                }
                Ok(weights.to_vec())
            }
            WeightedTargetSpec::Kernel { kernel, center, scale } => {
                let mut squared = vec![0.0_f64; rows];
                for ((&id, &centre), &scale) in
                    self.covariates.iter().zip(center.iter()).zip(scale.iter())
                {
                    let values = table.float64_cow(id).map_err(CausalError::from)?;
                    if values.len() != rows || values.iter().any(|v| !v.is_finite()) {
                        return Err(invalid(
                            "a kernel target covariate has missing or non-finite values; \
                             complete it or declare explicit row weights",
                        ));
                    }
                    for (acc, value) in squared.iter_mut().zip(values.iter()) {
                        let z = (value - centre) / scale;
                        *acc += z * z;
                    }
                }
                let distances: Vec<f64> = squared.iter().map(|s| s.sqrt()).collect();
                Ok(kernel
                    .weights(&distances)
                    .map_err(|error| CausalError::from(ProbError::from(error)))?)
            }
        }
    }

    /// Resolve the declaration against `table`: the weights, the overlap gate and the report.
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::weights`], or `transport_target_overlap_failed` with a
    /// next-attempt message when the weighting has no support among the rows, a Kish effective
    /// sample size below the floor, or a smaller share of the rows than declared.
    pub fn resolve<T: TableView + ?Sized>(
        &self,
        table: &T,
    ) -> Result<ResolvedWeightedTarget, CausalError> {
        let weights = self.weights(table)?;
        let overlap =
            check_weights_overlap(&weights, self.min_ess_fraction).map_err(
                |error| match error {
                    ProbError::Refused {
                        code: code @ "transport_target_overlap_failed",
                        message,
                    } => CausalError::Estimate(antecedent_estimate::EstimationError::refused(
                        code,
                        format!("weighted_target.overlap_failed: {message}"),
                    )),
                    other => CausalError::from(other),
                },
            )?;
        let report = self.report(self.identity(&weights), overlap, None);
        Ok(ResolvedWeightedTarget { weights, report })
    }

    /// The report of this declaration under a resolved identity and overlap.
    fn report(
        &self,
        identity: String,
        overlap: TargetOverlap,
        trial_overlap: Option<TargetOverlap>,
    ) -> WeightedTargetReport {
        let raw = u32::from_str_radix(&identity[..8], 16).unwrap_or(0) & 0x7FFF_FFFF;
        WeightedTargetReport {
            kernel: match &self.spec {
                WeightedTargetSpec::Kernel { kernel, .. } => Some(*kernel),
                WeightedTargetSpec::RowWeights(_) => None,
            },
            covariates: Arc::clone(&self.covariates),
            center: match &self.spec {
                WeightedTargetSpec::Kernel { center, .. } => Some(Arc::clone(center)),
                WeightedTargetSpec::RowWeights(_) => None,
            },
            scale: match &self.spec {
                WeightedTargetSpec::Kernel { scale, .. } => Some(Arc::clone(scale)),
                WeightedTargetSpec::RowWeights(_) => None,
            },
            overlap,
            trial_overlap,
            min_ess_fraction: self.min_ess_fraction,
            identity,
            distribution: DistributionRef::from_raw(raw),
        }
    }

    /// Resolve the declaration as the target of a trial-to-target transport over `table`.
    ///
    /// The target is the non-participant rows (`trial == 0`) reweighted by the declaration.
    /// Two gates apply, the shared overlap gate: one over the weights on the target rows, one
    /// over the weights `odds * v` the trial rows carry toward them, where `odds` is
    /// `(1 - s) / s` of the selection probability `s`.
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::weights`]; a trial or selection column of another length than
    /// the weights, or with a non-finite entry; or `transport_target_overlap_failed` from
    /// either gate.
    pub fn resolve_for_transport<T: TableView + ?Sized>(
        &self,
        table: &T,
        trial: VariableId,
        selection_probability: VariableId,
    ) -> Result<ResolvedWeightedTarget, CausalError> {
        let weights = self.weights(table)?;
        let trial = table.float64_cow(trial).map_err(CausalError::from)?;
        let selection = table.float64_cow(selection_probability).map_err(CausalError::from)?;
        if trial.len() != weights.len()
            || selection.len() != weights.len()
            || trial.iter().chain(selection.iter()).any(|v| !v.is_finite())
        {
            return Err(invalid(
                "the trial and selection columns must be finite and cover every row",
            ));
        }
        let mut target_rows = Vec::new();
        let mut trial_rows = Vec::new();
        for ((&member, &s), &v) in trial.iter().zip(selection.iter()).zip(weights.iter()) {
            if member == 0.0 {
                target_rows.push(v);
            } else if s > 0.0 && s < 1.0 {
                trial_rows.push((1.0 - s) / s * v);
            }
        }
        let gate = |rows: &[f64]| {
            check_weights_overlap(rows, self.min_ess_fraction).map_err(|error| match error {
                ProbError::Refused { code: code @ "transport_target_overlap_failed", message } => {
                    CausalError::Estimate(antecedent_estimate::EstimationError::refused(
                        code,
                        format!("weighted_target.overlap_failed: {message}"),
                    ))
                }
                other => CausalError::from(other),
            })
        };
        let overlap = gate(&target_rows)?;
        let trial_overlap = gate(&trial_rows)?;
        let report = self.report(self.identity(&weights), overlap, Some(trial_overlap));
        Ok(ResolvedWeightedTarget { weights, report })
    }

    /// Identity of the declaration and the exact weight bits.
    fn identity_digest(&self, weights: &[f64]) -> [u8; 32] {
        let mut stream = CanonicalWriter::new("antecedent.weighted_target.v1");
        match &self.spec {
            WeightedTargetSpec::RowWeights(_) => {
                stream.word(1);
            }
            WeightedTargetSpec::Kernel { kernel, center, scale } => {
                stream.word(2).text(kernel.family().id()).real(kernel.bandwidth());
                stream.reals(center).reals(scale);
            }
        }
        let covariates: Vec<u64> = self.covariates.iter().map(|v| u64::from(v.raw())).collect();
        stream.words(&covariates).real(self.min_ess_fraction).reals(weights);
        stream.finish()
    }

    /// [`Self::identity_digest`] as lowercase hex.
    fn identity(&self, weights: &[f64]) -> String {
        antecedent_core::to_hex(&self.identity_digest(weights))
    }

    /// The declaration as the transport query carries it: the identity of the declaration and
    /// the exact weights, the covariates, the floor and the kernel.
    #[must_use]
    pub fn transport_target(&self, weights: &[f64]) -> antecedent_core::TransportWeightedTarget {
        antecedent_core::TransportWeightedTarget {
            identity: self.identity_digest(weights),
            covariates: Arc::clone(&self.covariates),
            min_ess_fraction_bits: self.min_ess_fraction.to_bits(),
            kernel: match &self.spec {
                WeightedTargetSpec::Kernel { kernel, center, scale } => {
                    Some(antecedent_core::TransportTargetKernel {
                        family: Arc::from(kernel.family().id()),
                        bandwidth_bits: kernel.bandwidth().to_bits(),
                        center_bits: center.iter().map(|c| c.to_bits()).collect(),
                        scale_bits: scale.iter().map(|c| c.to_bits()).collect(),
                    })
                }
                WeightedTargetSpec::RowWeights(_) => None,
            },
        }
    }
}

/// The resolved weights of a weighted target of a trial-to-target transport study, with the
/// floor its overlap gate applies, frozen at build.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WeightedTransportWeights {
    /// One weight per table row.
    pub(crate) weights: Arc<[f64]>,
    /// The ESS-fraction floor the gate applies.
    pub(crate) min_ess_fraction: f64,
}

/// The weights of a resolved target and the report on them.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedWeightedTarget {
    /// Weights in table row order.
    pub weights: Vec<f64>,
    /// The declaration and overlap diagnostics.
    pub report: WeightedTargetReport,
}

/// The weighted target population a result is about, with its overlap diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedTargetReport {
    kernel: Option<TransportKernel>,
    covariates: Arc<[VariableId]>,
    center: Option<Arc<[f64]>>,
    scale: Option<Arc<[f64]>>,
    overlap: TargetOverlap,
    trial_overlap: Option<TargetOverlap>,
    min_ess_fraction: f64,
    identity: String,
    distribution: DistributionRef,
}

impl WeightedTargetReport {
    /// The declared kernel; `None` for explicit row weights.
    #[must_use]
    pub const fn kernel(&self) -> Option<TransportKernel> {
        self.kernel
    }

    /// Kernel family, when a kernel was declared.
    #[must_use]
    pub fn family(&self) -> Option<KernelFamily> {
        self.kernel.map(|k| k.family())
    }

    /// The covariates the weights depend on.
    #[must_use]
    pub fn covariates(&self) -> &[VariableId] {
        &self.covariates
    }

    /// Kernel center, when a kernel was declared.
    #[must_use]
    pub fn center(&self) -> Option<&[f64]> {
        self.center.as_deref()
    }

    /// Kernel scales, when a kernel was declared.
    #[must_use]
    pub fn scale(&self) -> Option<&[f64]> {
        self.scale.as_deref()
    }

    /// Support, Kish effective sample size and largest single-row weight share.
    #[must_use]
    pub const fn overlap(&self) -> &TargetOverlap {
        &self.overlap
    }

    /// For a trial-to-target transport, the overlap of the weights the trial rows carry
    /// toward the target; `None` for a target of an ordinary average effect.
    #[must_use]
    pub const fn trial_overlap(&self) -> Option<&TargetOverlap> {
        self.trial_overlap.as_ref()
    }

    /// The ESS-fraction floor the gate applied.
    #[must_use]
    pub const fn min_ess_fraction(&self) -> f64 {
        self.min_ess_fraction
    }

    /// Digest of the declaration and the exact weight bits.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The custom target distribution the weights are registered under.
    #[must_use]
    pub const fn distribution(&self) -> DistributionRef {
        self.distribution
    }

    /// The assumption statement the target adds to a result's assumptions.
    #[must_use]
    pub fn assumption(&self) -> String {
        let what = self.kernel.map_or_else(
            || "the declared row weighting".to_owned(),
            |k| {
                format!(
                    "the {} kernel neighbourhood (bandwidth {}) of {:?} (scale {:?})",
                    k.family().id(),
                    k.bandwidth(),
                    self.center().unwrap_or(&[]),
                    self.scale().unwrap_or(&[]),
                )
            },
        );
        format!(
            "weighted target: the target population is {what} over covariates {:?}; the effect \
             is standardized to it, assuming it differs from the analysed rows only through \
             those covariates (identity {})",
            self.covariates.iter().map(|v| v.raw()).collect::<Vec<_>>(),
            self.identity,
        )
    }
}

/// A trial-to-target contrast for a weighted target, with the report on its weights.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedTrialTransport {
    /// The transported contrast, standard errors and the two overlap gates.
    pub estimate: antecedent_estimate::WeightedTargetTransportEstimate,
    /// The declaration and overlap of the target weights over the table.
    pub report: WeightedTargetReport,
}

/// Transport a randomized trial contrast to a weighted target population.
///
/// `table` holds the trial and the target rows (the covariates the declaration reads are its
/// columns); `trial[i]` marks a trial row. The target is the non-participant rows reweighted
/// by the declaration, behind the overlap gate over those rows and the gate over the weights
/// the trial rows carry toward them.
///
/// # Errors
///
/// The refusals of [`WeightedTargetDeclaration::resolve`] and of
/// [`antecedent_estimate::trial_to_weighted_target_effect`], including
/// `transport_target_overlap_failed`.
#[allow(
    clippy::too_many_arguments,
    reason = "the inputs of the trial-to-target contrast and the declared target"
)]
pub fn transport_trial_to_weighted_target<T: TableView + ?Sized>(
    declaration: &WeightedTargetDeclaration,
    table: &T,
    identification: &antecedent_identify::TransportIdentification,
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
    outcome_regressions: Option<(&[f64], &[f64])>,
) -> Result<WeightedTrialTransport, CausalError> {
    let weights = declaration.weights(table)?;
    let identity = declaration.identity(&weights);
    let estimate = antecedent_estimate::trial_to_weighted_target_effect(
        identification,
        outcome,
        treatment,
        trial,
        selection_probability,
        treatment_probability,
        outcome_regressions,
        &weights,
        declaration.min_ess_fraction(),
    )?;
    let report =
        declaration.report(identity, estimate.target_overlap, Some(estimate.trial_overlap));
    Ok(WeightedTrialTransport { estimate, report })
}
