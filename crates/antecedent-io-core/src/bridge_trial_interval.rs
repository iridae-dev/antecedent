//! Whole-trial pairs percentile intervals for a trial-level terminal effect at a source mean.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::IoError;
use crate::bridge_distribution::{
    bridge_draws_cancellable, shared_trial_bootstrap_source_cancellable,
};
use crate::bridge_joint_outcomes::JointOutcomeBridgeError;
use crate::calibration::{
    CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire, calibration_slot,
};
use crate::contract_section::{CalibrationSlotWire, CalibrationStatus};
use crate::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use crate::quantity_wire::DistributionMeaningWire;
use antecedent_contract::bridge::{
    BridgeContract, BridgeMapping, SourceMappingDependence, ValueKind, draws_unusable,
    fit_trial_level_linear,
};
use antecedent_core::{
    CanonicalWriter, CausalRng, ExecutionContext, SemanticDigest, derive_stream,
};
use antecedent_stats::{QuantileRule, quantile_sorted};
use serde::{Deserialize, Serialize};

/// Sampling design for the mean at which the estimated trial-level map is evaluated.
#[derive(Clone, Debug)]
pub enum TrialMeanStudy {
    /// The source mean is the mean surrogate effect on the same validation trials.
    Shared,
    /// The source mean is estimated from a separately sampled series of trial effects.
    Independent {
        /// Independent source study or evidence identity.
        study_id: String,
        /// One observed surrogate treatment effect per independently sampled source trial.
        surrogate_effects: Vec<f64>,
    },
}

/// Caller-declared data-generating design of a release calibration record.
/// The observed-range checks cannot verify its distributional or independence premises.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialBridgeReleaseDesign {
    /// IID validation trials: surrogate Uniform(-1, 1) and terminal effect
    /// 0.1 + 0.9 × surrogate + independent Gaussian noise (SD 0.35).
    /// The same trials estimate the source mean.
    SharedUniformGaussianV1,
    /// Same validation law, with a separate IID Uniform(-0.25, 0.75) source sample.
    /// Both samples have the same number of trials.
    IndependentShiftedUniformV1,
}

impl TrialBridgeReleaseDesign {
    /// Stable wire name of the source-fixed design.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SharedUniformGaussianV1 => "shared_uniform_gaussian_v1",
            Self::IndependentShiftedUniformV1 => "independent_shifted_uniform_v1",
        }
    }
}

/// Scientific declaration of a joint refit experiment. The contract retains the validation trials.
#[derive(Clone, Debug)]
pub struct TrialBridgeExperiment {
    /// Trial-level line with its premises and source/validation dependence declaration.
    pub contract: BridgeContract,
    /// Ownership of the source observations.
    pub study: TrialMeanStudy,
    /// Number of paired source and validation refits.
    pub replicates: u32,
    /// Interval confidence level.
    pub level: f64,
    /// Stream for drawing validation pairs and, separately, independent source observations.
    pub seed: u64,
    /// Release calibration applies only under the explicitly declared design.
    pub release_design: Option<TrialBridgeReleaseDesign>,
}

/// The whole-procedure construction scored during calibration.
#[derive(Clone, Debug)]
pub struct ConstructedTrialBridgeInterval {
    /// Target coordinate of the point and interval endpoints.
    pub quantity: antecedent_contract::ScientificQuantity,
    /// Original fitted line evaluated at the original source sample mean.
    pub point: f64,
    /// Equal-tailed type-7 quantiles across mapped source replicates.
    pub endpoints: [f64; 2],
    /// Complete paired bootstrap law, preserving replicate meaning.
    pub bootstrap: DistributionArtifact,
    /// Repeated-experiment coordinate used when requesting a publication receipt.
    pub basis: CalibrationBasisWire,
    /// Identity binds both samples, the confidence level, stream, and actual mapped draw table.
    pub identity: SemanticDigest,
}

/// A sampling interval with the coverage record applicable to the experiment that produced it.
#[derive(Clone, Debug)]
pub struct TrialBridgeConfidenceInterval {
    /// Construction independently reproducible from raw trials.
    pub interval: ConstructedTrialBridgeInterval,
    /// Measured coverage of the complete source and validation refit.
    pub calibration: CalibrationSlotWire,
}

impl TrialBridgeExperiment {
    /// Construct the statistic and its bootstrap endpoints for measurement against known truth.
    /// # Errors
    /// Invalid trial data, incompatible source design, unavailable fit, or resource refusal.
    pub fn construct(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ConstructedTrialBridgeInterval, JointOutcomeBridgeError> {
        self.contract.check_draws_support()?;
        let BridgeMapping::TrialLevelLinear(validation) = &self.contract.mapping else {
            return Err(
                draws_unusable("mapping", "supply an estimated trial-level regression").into()
            );
        };
        if self.contract.source_value != ValueKind::Bootstrap
            || self.contract.target_value != ValueKind::Bootstrap
            || !(0. < self.level && self.level < 1.)
            || self.replicates < 2
        {
            return Err(draws_unusable(
                "interval experiment",
                "supply a bootstrap contract, at least two draws and a level in (0,1)",
            )
            .into());
        }
        let source_size = match &self.study {
            TrialMeanStudy::Shared => validation.trials.len(),
            TrialMeanStudy::Independent { surrogate_effects, .. } => surrogate_effects.len(),
        };
        let n = validation.trials.len();
        if let Some(design) = self.release_design {
            self.check_release_design(design, validation, source_size)?;
        }
        if source_size < 2
            || (n.saturating_add(source_size) as u64).saturating_mul(u64::from(self.replicates))
                > antecedent_contract::bridge::MAX_PROPAGATION_WORK
        {
            return Err(draws_unusable(
                "trial refit work",
                "supply at least two source trials and a refit design within the work budget",
            )
            .into());
        }
        let _lease = ctx
            .acquire_serial_numerical_admission(
                (n.saturating_add(source_size) as u64)
                    .saturating_mul(128)
                    .saturating_add(u64::from(self.replicates).saturating_mul(224)),
            )
            .map_err(IoError::from)?;
        let mut key = CanonicalWriter::new("antecedent.bridge.trial_percentile_interval.v1");
        key.bytes(self.contract.identity().as_bytes())
            .word(self.level.to_bits())
            .word(u64::from(self.replicates))
            .word(self.seed)
            .opt_text(self.release_design.map(TrialBridgeReleaseDesign::as_str));
        let (design,source_mean,source) = match (&self.study,&self.contract.source_dependence) {
            (TrialMeanStudy::Shared,Some(SourceMappingDependence::SharedTrialMean{..})) => {
                key.text("shared");
                let mut ordered: Vec<_> = validation.trials.iter().map(|t|t.surrogate_effect).collect();
                ordered.sort_by(f64::total_cmp);
                ("shared_trial_mean",ordered.iter().sum::<f64>()/n as f64,shared_trial_bootstrap_source_cancellable(&self.contract,self.replicates as usize,self.seed,&ctx.cancellation)?)
            },
            (TrialMeanStudy::Independent{study_id,surrogate_effects},Some(SourceMappingDependence::Independent { .. })) if !study_id.trim().is_empty() && surrogate_effects.iter().all(|v|v.is_finite()) => {
                let mut ordered = surrogate_effects.clone();
                ordered.sort_by(f64::total_cmp);
                let mut evidence=CanonicalWriter::new("antecedent.bridge.independent_source_trials.v1");
                evidence.text(study_id).word(source_size as u64);
                for v in &ordered {evidence.word(v.to_bits());}
                let snapshot=evidence.finish_hex();
                key.text("independent").text(&snapshot);
                const SOURCE_STREAM:u64=0xa1ff_1e02_0001;
                let mut rng=CausalRng::from_seed(derive_stream(self.seed,SOURCE_STREAM));
                let mut samples=Vec::with_capacity(self.replicates as usize);
                for _ in 0..self.replicates {
                    if ctx.cancellation.is_cancelled(){ return Err(antecedent_contract::bridge::propagation_cancelled().into()); }
                    let mean=(0..source_size).map(|_|ordered[antecedent_kernels::unbiased_index(&mut rng,source_size)]).sum::<f64>()/source_size as f64;
                    samples.push(mean);
                }
                let identity = DistributionIdentity::new(DistributionMeaningWire::Bootstrap,std::slice::from_ref(&self.contract.source.quantity),DrawAlignment::Joint,
                    DistributionProvenance {source_id:study_id.clone(),provider_id:"scientific_bridge.independent_trial_mean".into(),rng_id:format!("source-trials:{}:{SOURCE_STREAM}",self.seed),
                        causal_contract_id:self.contract.identity().to_hex(),snapshot_id:snapshot})?;
                let source=DistributionArtifact::new(DistributionMetadata {version:1,identity,axes:["draw".into(),"quantity".into()],shape:[self.replicates as usize,1],weights:None,supported:None,calibration:DistributionCalibration::NotMeasured,trust:DistributionTrust::Unverified,legacy_posterior:None,legacy_bindings:None},samples)?;
                ("independent_trial_mean",ordered.iter().sum::<f64>()/source_size as f64,source)
            }
            _=>return Err(draws_unusable("source study","declare whether source and validation are the same trials or separate studies, and retain finite source effects").into())
        };
        let fit = fit_trial_level_linear(validation)?;
        let point = fit.intercept + fit.slope * source_mean;
        if !point.is_finite() || source_mean < fit.surrogate_min || source_mean > fit.surrogate_max
        {
            return Err(draws_unusable(
                "source mean",
                "the observed source mean must lie inside the validation range",
            )
            .into());
        }
        let mapped =
            bridge_draws_cancellable(&self.contract, &source, None, self.seed, &ctx.cancellation)?;
        let bootstrap = mapped.distribution().clone();
        let mut values = bootstrap.draws().to_vec();
        values.sort_by(f64::total_cmp);
        let tail = (1. - self.level) / 2.;
        let endpoints = [
            quantile_sorted(&values, tail, QuantileRule::Interpolated),
            quantile_sorted(&values, 1. - tail, QuantileRule::Interpolated),
        ];
        key.bytes(&bootstrap.to_bytes("trial-bootstrap")?);
        let sample_design = if n == source_size {
            "equal_source_and_validation_counts".to_owned()
        } else {
            format!("unequal_source_n={source_size};validation_n={n}")
        };
        let basis = CalibrationBasisWire {
            key: CalibrationKeyWire {
                validation: "not_applicable".into(),
                query: "TrialBridgePercentileInterval".into(),
                graph_class: "NotApplicable".into(),
                structure: "fixed".into(),
                modality: "trial_effect_pairs".into(),
                inference: "Frequentist".into(),
                estimator: "trial_ols_at_source_mean".into(),
                interval_method: "trial_pairs_percentile_type7".into(),
                se_kind: "bootstrap".into(),
                dependence: design.into(),
                posterior: String::new(),
                functional: format!(
                    "release_design={};source_design={sample_design};refits={};min_r_squared={:016x};min_trials={}",
                    self.release_design.map_or("undeclared", TrialBridgeReleaseDesign::as_str),
                    self.replicates,
                    validation.min_r_squared.to_bits(),
                    validation.min_trials
                ),
                level: self.level,
                identification: "point".into(),
            },
            scope: CalibrationScopeWire {
                row_count: n as u64,
                replicates_ok: Some(self.replicates),
                posterior_draws: None,
                unidentified_mass: 0.,
            },
        };
        Ok(ConstructedTrialBridgeInterval {
            quantity: self.contract.target.quantity.clone(),
            point,
            endpoints,
            bootstrap,
            basis,
            identity: SemanticDigest::from_bytes(key.finish()),
        })
    }

    fn check_release_design(
        &self,
        design: TrialBridgeReleaseDesign,
        validation: &antecedent_contract::bridge::TrialLevelValidation,
        source_size: usize,
    ) -> Result<(), JointOutcomeBridgeError> {
        let source_matches = match (design, &self.study) {
            (TrialBridgeReleaseDesign::SharedUniformGaussianV1, TrialMeanStudy::Shared) => true,
            (
                TrialBridgeReleaseDesign::IndependentShiftedUniformV1,
                TrialMeanStudy::Independent { surrogate_effects, .. },
            ) => surrogate_effects.iter().all(|value| (-0.25..=0.75).contains(value)),
            _ => false,
        };
        let validation_matches =
            validation.trials.iter().all(|pair| (-1.0..=1.0).contains(&pair.surrogate_effect));
        if !source_matches
            || !validation_matches
            || source_size != validation.trials.len()
            || self.level.to_bits() != 0.95_f64.to_bits()
            || self.replicates != 399
            || validation.min_r_squared.to_bits() != 0.0_f64.to_bits()
            || validation.min_trials != 3
        {
            return Err(draws_unusable(
                "trial release design",
                "match the declared trial design, effect support and equal sample counts; use 399 refits at 0.95 with min_trials=3 and min_r_squared=0",
            ).into());
        }
        Ok(())
    }

    /// Publish only inside the design measured by a current coverage record.
    /// # Errors
    /// A construction refusal or a missing measured coverage record.
    pub fn interval(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TrialBridgeConfidenceInterval, JointOutcomeBridgeError> {
        let interval = self.construct(ctx)?;
        let calibration = calibration_slot(&interval.basis);
        if self.release_design.is_none()
            || calibration.status != CalibrationStatus::Calibrated
            || calibration.record_id.is_none()
            || calibration.calibration_sha.is_none()
        {
            return Err(antecedent_contract::ExternalRefusal::new(
                antecedent_core::reason_code!("route_not_supported"),
                "bind",
                antecedent_core::reason_details::scientific_bridge::INTERVAL_SCOPE_OUTSIDE_MEASURED,
            )
            .with_remedy(
                "request the validated trial refit design or consume its transformed bootstrap law",
            )
            .into());
        }
        Ok(TrialBridgeConfidenceInterval { interval, calibration })
    }
}
