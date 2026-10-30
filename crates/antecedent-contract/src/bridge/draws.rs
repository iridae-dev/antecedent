//! Bridges that carry distributions.
//!
//! A bridge of a mean maps one number. A source that is a posterior or a
//! predictive law is a set of aligned draws, and the bridged claim is the same
//! set of rows with the bridged column in place of the source column: row `i`
//! of the output is the image of row `i` of the input, so every other column of
//! the source table stays paired with it and the alignment identity of the
//! table is the source's own.
//!
//! This module holds the rules, not the arithmetic of sampling:
//!
//! * [`BridgeContract::check_draws_licensing`] is [`BridgeContract::check_licensing`]
//!   for a draws-valued source, plus the declaration of what the mapping adds to
//!   the spread of the draws. A map with uncertain coefficients declares how its
//!   estimators depend on the source estimate; a trial-level line and a
//!   measurement correction take their uncertainty from their own evidence and
//!   so declare none a second time, which would count it twice.
//! * [`DrawMapping`] is the joint law of the mapping's coefficients, stated as
//!   moments: the declared (or trial-fitted) intercept and slope with their
//!   variances and covariance, their covariance with the source estimate, and,
//!   for a measurement correction, the rates and their variances.
//! * [`AffineDrawLaw::innovation`] turns those moments into the conditional
//!   draw of the coefficients given one source draw. The source variance comes
//!   only from the source draws and the mapping variance only from the mapping,
//!   so neither is counted twice; the covariance between them is the declared
//!   dependence and nothing else.
//! * [`DrawsPlan`] binds a licensed contract to a source of draws and
//!   [`DrawsPlan::finish`] produces the derived claim once the mapped draws
//!   exist: its own identity (binding the contract, the source and the mapped
//!   draws), provenance and a receipt that says what was not established.
//!
//! The interval of a bridged distribution is not built here. It is requested
//! through the same measured-coverage gate as the interval of a bridged mean,
//! and a distribution without a measured record is read as an uncalibrated
//! spread.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::assumption::{AssumptionSource, AssumptionStatus};
use antecedent_core::identity::{CanonicalWriter, ContractIdentities, SemanticDigest};

use super::super::claim::{
    BRIDGE_CLAIM_IDENTITY_VERSION, bridged_identities, mismatch_name, node, obligation,
};
use super::super::kind::{BridgeKind, ValueKind};
use super::super::mapping::{
    BridgeMapping, MappingKind, MeasurementDirection, TrialLevelValidation, TrialStatistics,
    binary_rates, fit_with_statistics, identified_youden,
};
use super::super::refusal::{
    BridgeRefusal, dependence_not_a_covariance, draws_meaning_mismatch, draws_not_aligned,
    draws_unusable, interval_not_licensed, malformed_mapping, mapping_uncertainty_invalid,
    mapping_uncertainty_missing, provenance_failed, quantity_mismatch, source_dependence_invalid,
    source_dependence_missing, source_not_draws, source_variance_invalid,
    validation_trials_too_few, wrong_value_kind,
};
use super::{BridgeContract, Licensing, SourceMappingDependence};
use crate::claim::{
    ClaimDomains, ClaimEnvelope, ClaimFreshness, ClaimKind, DomainStatus, HandoffReceipt,
};
use crate::obligation::{ObligationKind, ObligationRecord, ObligationScope};
use crate::provenance::ProvenanceGraph;
use crate::quantity::ScientificQuantity;
use crate::reasoning::{
    AssumptionSlot, ReasoningView, SlotAvailability, UncertaintyComponent, UncertaintySlot,
    UncertaintySource,
};
use antecedent_core::measurement_model::MeasurementParameters;

const TAG_DRAWS_CLAIM: &str = "antecedent.scientific_bridge.draws_claim";

/// Fewest validation trials a trial-level line may have when its coefficients
/// are drawn: with `k` trials the coefficients follow a Student-t law with
/// `k - 2` degrees of freedom, whose variance is finite only above two.
pub const MIN_DRAW_TRIALS: u32 = 5;

/// Relative tolerance of the feasibility checks on a declared covariance.
const FEASIBILITY_TOLERANCE: f64 = 1e-10;

// ------------------------------------------------------------------ mapping laws

/// The joint law of the coefficients of an affine map `a + b x`, as moments.
///
/// The coefficients are drawn together with the source estimate `M`, taking `M`
/// independent of the intercept and writing `c = Cov(b, M)` and
/// `d = Cov(a, b)`: the trivariate covariance of `(a, b, M)` is
/// `[[Va, d, 0], [d, Vb, c], [0, c, V]]`, the same one
/// [`BridgeContract::propagate_affine_uncertainty`] takes its delta-method
/// variance from.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineDrawLaw {
    /// Mean of the intercept.
    pub intercept_mean: f64,
    /// Mean of the slope.
    pub slope_mean: f64,
    /// Variance of the intercept; zero states it exact.
    pub intercept_variance: f64,
    /// Variance of the slope; zero states it exact.
    pub slope_variance: f64,
    /// Covariance of the intercept and slope estimators.
    pub intercept_slope_covariance: f64,
    /// Covariance of the slope estimator and the source estimate.
    pub source_slope_covariance: f64,
    /// Degrees of freedom of the Student-t law the coefficients follow when
    /// they were fitted from validation trials; `None` is the Gaussian law of a
    /// declared mapping uncertainty.
    pub degrees_of_freedom: Option<f64>,
    /// Residual variance of a trial-level line: the variance a new trial adds
    /// to the mean line. Zero for a declared map.
    pub residual_variance: f64,
    /// The range of source values the map is licensed for, when it has one (the
    /// validated surrogate range of a trial-level line).
    pub domain: Option<(f64, f64)>,
}

/// The draw of the coefficients given one source draw, in standard-normal
/// loadings.
///
/// With `z1`, `z2` independent standard normals, a scale factor `g` (one for a
/// Gaussian law, `sqrt(nu / chi2_nu)` for a Student-t law) and a source draw
/// `m` of mean `mu`:
///
/// * `slope = slope_mean + slope_loading * (m - mu) + g * slope_sd * z1`;
/// * `intercept = intercept_mean + g * (intercept_on_slope * z1 + intercept_sd * z2)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineInnovation {
    /// `Cov(b, M) / Var(M)`: the part of the slope explained by the source draw.
    pub slope_loading: f64,
    /// Standard deviation of the slope given the source draw.
    pub slope_sd: f64,
    /// Loading of the intercept on the slope's standard-normal innovation.
    pub intercept_on_slope: f64,
    /// Standard deviation of the intercept given the slope.
    pub intercept_sd: f64,
}

impl AffineDrawLaw {
    /// The conditional draw of the coefficients given the source draws, whose
    /// variance is `source_variance`.
    ///
    /// # Errors
    /// A source variance that is not finite and non-negative; a covariance with
    /// the source declared for draws that do not vary; a declared covariance
    /// that no joint law of the intercept, the slope and the source has (the
    /// square of the slope-source covariance exceeds the product of the slope
    /// and source variances, or the intercept-slope covariance is not feasible
    /// given it).
    pub fn innovation(&self, source_variance: f64) -> Result<AffineInnovation, BridgeRefusal> {
        if !(source_variance.is_finite() && source_variance >= 0.0) {
            return Err(source_variance_invalid(source_variance.to_string()));
        }
        for (name, value) in [
            ("intercept_variance", self.intercept_variance),
            ("slope_variance", self.slope_variance),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(mapping_uncertainty_invalid(
                    name,
                    "state a finite nonnegative variance",
                ));
            }
        }
        if !self.intercept_slope_covariance.is_finite() {
            return Err(mapping_uncertainty_invalid(
                "intercept_slope_covariance",
                "state a finite covariance",
            ));
        }
        if !self.source_slope_covariance.is_finite() {
            return Err(source_dependence_invalid(
                "source_slope_covariance",
                "state a finite covariance",
            ));
        }
        let covariance = self.source_slope_covariance;
        let (explained, slope_loading) = if covariance == 0.0 {
            (0.0, 0.0)
        } else if source_variance > 0.0 {
            let scaled = covariance / source_variance.sqrt();
            (scaled * scaled, covariance / source_variance)
        } else {
            return Err(draws_unusable(
                "source_variance",
                "a covariance with the source can be declared only for draws that vary",
            ));
        };
        if !explained.is_finite() || !slope_loading.is_finite() {
            return Err(dependence_not_a_covariance(covariance.to_string()));
        }
        let slack = FEASIBILITY_TOLERANCE * self.slope_variance.max(explained);
        let remaining = self.slope_variance - explained;
        if remaining < -slack {
            return Err(dependence_not_a_covariance(covariance.to_string()));
        }
        let slope_sd = remaining.max(0.0).sqrt();
        let covariance_ab = self.intercept_slope_covariance;
        let intercept_on_slope = if slope_sd > 0.0 {
            covariance_ab / slope_sd
        } else if covariance_ab.abs()
            <= FEASIBILITY_TOLERANCE * self.intercept_variance.sqrt() * self.slope_variance.sqrt()
        {
            0.0
        } else {
            return Err(dependence_not_a_covariance(covariance_ab.to_string()));
        };
        let residual = self.intercept_variance - intercept_on_slope * intercept_on_slope;
        if !intercept_on_slope.is_finite() || !residual.is_finite() {
            return Err(dependence_not_a_covariance(covariance_ab.to_string()));
        }
        if residual < -FEASIBILITY_TOLERANCE * self.intercept_variance {
            return Err(dependence_not_a_covariance(covariance_ab.to_string()));
        }
        Ok(AffineInnovation {
            slope_loading,
            slope_sd,
            intercept_on_slope,
            intercept_sd: residual.max(0.0).sqrt(),
        })
    }
}

/// The rates of a binary-misclassification correction, with the variances of
/// their estimates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectionDrawLaw {
    /// Sensitivity.
    pub sensitivity: f64,
    /// Specificity.
    pub specificity: f64,
    /// Variance of the sensitivity estimate; zero states it exact.
    pub sensitivity_variance: f64,
    /// Variance of the specificity estimate; zero states it exact.
    pub specificity_variance: f64,
    /// Smallest Youden index the model licenses.
    pub min_youden: f64,
    /// Which way the correction runs.
    pub direction: MeasurementDirection,
}

/// What a draws bridge adds to the spread of its source.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DrawMapping {
    /// An identity, declared or trial-fitted affine map with its coefficient law.
    Affine(AffineDrawLaw),
    /// A binary-misclassification correction with its rates.
    Correction(CorrectionDrawLaw),
}

fn check_rate_variance(name: &str, rate: f64, variance: f64) -> Result<(), BridgeRefusal> {
    if variance == 0.0 {
        return Ok(());
    }
    if variance > 0.0 && variance.is_finite() && variance < rate * (1.0 - rate) {
        return Ok(());
    }
    Err(mapping_uncertainty_invalid(
        name,
        "a rate with a variance is drawn from a Beta law of that mean and variance, which exists only below the Bernoulli variance mean * (1 - mean)",
    ))
}

fn correction_law(
    model: &antecedent_core::measurement_model::MeasurementModel,
    direction: MeasurementDirection,
) -> Result<CorrectionDrawLaw, BridgeRefusal> {
    let (sensitivity, specificity, min_youden) = binary_rates(model)?;
    identified_youden(sensitivity, specificity, min_youden)?;
    let MeasurementParameters::Binary { rates, .. } = &model.parameters else {
        return Err(malformed_mapping(
            "parameters",
            "the measurement model's parameters are not those of its route",
        ));
    };
    let [shared] = rates.as_slice() else {
        return Err(malformed_mapping(
            "rates",
            "a non-differential model carries one shared pair of rates",
        ));
    };
    check_rate_variance("sensitivity_variance", sensitivity, shared.sensitivity_variance)?;
    check_rate_variance("specificity_variance", specificity, shared.specificity_variance)?;
    Ok(CorrectionDrawLaw {
        sensitivity,
        specificity,
        sensitivity_variance: shared.sensitivity_variance,
        specificity_variance: shared.specificity_variance,
        min_youden,
        direction,
    })
}

/// The law of the coefficients of a trial-level line: the sampling covariance
/// of the least-squares coefficients, `s^2 (X'X)^-1`, with `s^2` the residual
/// variance on `k - 2` degrees of freedom. Under the reference prior this is the
/// posterior of the coefficients, a Student-t law.
fn trial_law(
    validation: &TrialLevelValidation,
    source_slope_covariance: f64,
) -> Result<AffineDrawLaw, BridgeRefusal> {
    let (fit, statistics) = fit_with_statistics(validation)?;
    let trials = validation.trials.len();
    if trials < MIN_DRAW_TRIALS as usize {
        return Err(validation_trials_too_few(MIN_DRAW_TRIALS, trials));
    }
    // The fit's own canonical statistics: the residual sum of squares is a second pass over the
    // residuals, so a near-exact line is not lost to cancellation and an exact one is exactly 0.
    let TrialStatistics { count, mean_x, sxx, residual_sum_squares, .. } = statistics;
    let degrees_of_freedom = count - 2.0;
    let residual_variance = residual_sum_squares / degrees_of_freedom;
    Ok(AffineDrawLaw {
        intercept_mean: fit.intercept,
        slope_mean: fit.slope,
        intercept_variance: residual_variance * (1.0 / count + mean_x * mean_x / sxx),
        slope_variance: residual_variance / sxx,
        intercept_slope_covariance: -residual_variance * mean_x / sxx,
        source_slope_covariance,
        degrees_of_freedom: Some(degrees_of_freedom),
        residual_variance,
        domain: Some((fit.surrogate_min, fit.surrogate_max)),
    })
}

fn declared_affine_law(contract: &BridgeContract, intercept: f64, slope: f64) -> AffineDrawLaw {
    let (intercept_variance, slope_variance, intercept_slope_covariance) =
        contract.mapping_uncertainty.as_ref().map_or((0.0, 0.0, 0.0), |uncertainty| {
            (
                uncertainty.intercept_variance,
                uncertainty.slope_variance,
                uncertainty.intercept_slope_covariance,
            )
        });
    AffineDrawLaw {
        intercept_mean: intercept,
        slope_mean: slope,
        intercept_variance,
        slope_variance,
        intercept_slope_covariance,
        source_slope_covariance: contract
            .source_dependence
            .as_ref()
            .map_or(0.0, SourceMappingDependence::covariance),
        degrees_of_freedom: None,
        residual_variance: 0.0,
        domain: None,
    }
}

/// Check the declaration of what the mapping adds to the draws it carries.
///
/// An identity or declared affine map with uncertain coefficients declares its
/// dependence on the source (and an exact map declares neither); a trial-level
/// line draws its coefficients from its validation trials and a measurement
/// correction draws its rates from their variances, so neither declares the
/// mapping's uncertainty again, and both declare how they depend on the source
/// estimate (a correction has no slope estimator, so only independence).
pub(super) fn check_draws_declaration(contract: &BridgeContract) -> Result<(), BridgeRefusal> {
    match &contract.mapping {
        BridgeMapping::Identity | BridgeMapping::Affine { .. } => {
            match (&contract.mapping_uncertainty, &contract.source_dependence) {
                (Some(_), None) => Err(source_dependence_missing()),
                (None, Some(_)) => Err(mapping_uncertainty_missing()),
                _ => Ok(()),
            }
        }
        BridgeMapping::TrialLevelLinear(_) => {
            if contract.mapping_uncertainty.is_some() {
                return Err(mapping_uncertainty_invalid(
                    "mapping_uncertainty",
                    "a trial-level line draws its coefficients from its validation trials; declaring the mapping's uncertainty again would count it twice",
                ));
            }
            if contract.source_dependence.is_none() {
                return Err(source_dependence_missing());
            }
            Ok(())
        }
        BridgeMapping::MeasurementCorrection(_) => {
            if contract.mapping_uncertainty.is_some() {
                return Err(mapping_uncertainty_invalid(
                    "mapping_uncertainty",
                    "a measurement correction draws its rates from the variances of the measurement model; declaring the mapping's uncertainty again would count it twice",
                ));
            }
            match &contract.source_dependence {
                None => Err(source_dependence_missing()),
                Some(SourceMappingDependence::Independent { .. }) => Ok(()),
                Some(SourceMappingDependence::SlopeCovariance { .. }) => {
                    Err(source_dependence_invalid(
                        "source_dependence",
                        "a measurement correction has no slope estimator; declare the source and the validation sample independent",
                    ))
                }
            }
        }
    }
}

fn draw_mapping(
    contract: &BridgeContract,
    direction: MeasurementDirection,
) -> Result<DrawMapping, BridgeRefusal> {
    Ok(match &contract.mapping {
        BridgeMapping::Identity => DrawMapping::Affine(declared_affine_law(contract, 0.0, 1.0)),
        BridgeMapping::Affine { intercept, slope } => {
            DrawMapping::Affine(declared_affine_law(contract, *intercept, *slope))
        }
        BridgeMapping::TrialLevelLinear(validation) => {
            let covariance = contract
                .source_dependence
                .as_ref()
                .map_or(0.0, SourceMappingDependence::covariance);
            DrawMapping::Affine(trial_law(validation, covariance)?)
        }
        BridgeMapping::MeasurementCorrection(model) => {
            DrawMapping::Correction(correction_law(model, direction)?)
        }
    })
}

// ------------------------------------------------------------------------ source

/// The draws being bridged and the identities they carry.
///
/// The draws themselves are not part of it: the contract layer states the rule
/// and the layer that holds the draw table samples. `claim_id` is the identity
/// of the draws object (or of the claim that carries it), `alignment_identity`
/// the pairing identity of its table.
#[derive(Clone, Copy, Debug)]
pub struct BridgeDrawsSource<'a> {
    /// The quantity the bridged column of the table is about.
    pub quantity: &'a ScientificQuantity,
    /// What the draws are: a posterior or a predictive law. `None` for draws of
    /// another meaning (a sampling law, a bootstrap, an empirical law).
    pub value_kind: Option<ValueKind>,
    /// The name of the draws' meaning, for the refusal of one that is neither.
    pub meaning: &'a str,
    /// The kind of the claim that carries the draws, when one does.
    pub claim_kind: ClaimKind,
    /// Identity of the source claim or draws object.
    pub claim_id: SemanticDigest,
    /// The identity layers the bridged claim inherits.
    pub identities: &'a ContractIdentities,
    /// Execution lineage of the source, when it has one.
    pub execution: Option<SemanticDigest>,
    /// Evidence references the source carries.
    pub evidence: &'a [Arc<str>],
    /// Evaluated-domain status of the source.
    pub evaluated: DomainStatus,
    /// Whether the rows of the table are paired joint realizations.
    pub joint: bool,
    /// The pairing identity of the table, when it is joint.
    pub alignment_identity: &'a str,
}

/// A licensed contract bound to a source of draws: the mapping law and
/// everything the derived claim needs, before any draw is mapped.
#[derive(Clone, Debug)]
pub struct DrawsPlan {
    contract: BridgeContract,
    contract_identity: SemanticDigest,
    licensing: Licensing,
    mapping: DrawMapping,
    source_claim_id: SemanticDigest,
    source_kind: ValueKind,
    source_identities: ContractIdentities,
    source_execution: Option<SemanticDigest>,
    source_evidence: Vec<Arc<str>>,
    source_evaluated: DomainStatus,
    alignment: String,
    stream: u64,
}

impl BridgeContract {
    /// Bind this contract to a source of draws.
    ///
    /// Nothing is planned unless the contract licenses a draws-valued bridge
    /// ([`Self::check_draws_licensing`]) and the source is the aligned draws of
    /// the contract's source quantity, of the value kind the contract declares.
    /// `stream` names the random stream the mapping's coefficients are drawn
    /// from; it enters the identity of the derived claim.
    ///
    /// # Errors
    /// A contract that does not license its bridge; a source that is an
    /// associational, refused, bounded or incomplete claim; draws of another
    /// quantity; draws that are not the declared value kind or are independent
    /// marginals.
    pub fn plan_distribution(
        &self,
        source: &BridgeDrawsSource<'_>,
        stream: u64,
    ) -> Result<DrawsPlan, BridgeRefusal> {
        let licensing = self.check_draws_licensing()?;
        if let Err(mismatch) = source.quantity.require_same_coordinate(&self.source.quantity) {
            return Err(quantity_mismatch(
                mismatch_name(mismatch),
                self.source.quantity.canonical_identity(),
                source.quantity.canonical_identity(),
            ));
        }
        if matches!(
            source.claim_kind,
            ClaimKind::Associational
                | ClaimKind::Refusal
                | ClaimKind::Bounds
                | ClaimKind::Incomplete
        ) {
            return Err(source_not_draws(source.claim_kind.as_str()));
        }
        let Some(held) = source.value_kind else {
            return Err(draws_meaning_mismatch(self.source_value, source.meaning));
        };
        if held != self.source_value {
            return Err(wrong_value_kind(self.source_value, held));
        }
        if !source.joint {
            return Err(draws_not_aligned());
        }
        let mapping = draw_mapping(self, licensing.direction)?;
        Ok(DrawsPlan {
            contract: self.clone(),
            contract_identity: self.identity(),
            licensing,
            mapping,
            source_claim_id: source.claim_id,
            source_kind: held,
            source_identities: *source.identities,
            source_execution: source.execution,
            source_evidence: source.evidence.to_vec(),
            source_evaluated: source.evaluated,
            alignment: source.alignment_identity.to_owned(),
            stream,
        })
    }
}

impl DrawsPlan {
    /// The contract.
    #[must_use]
    pub const fn contract(&self) -> &BridgeContract {
        &self.contract
    }

    /// The law the mapping draws its coefficients from.
    #[must_use]
    pub const fn mapping(&self) -> &DrawMapping {
        &self.mapping
    }

    /// What the licensing check determined.
    #[must_use]
    pub const fn licensing(&self) -> Licensing {
        self.licensing
    }

    /// The random stream the coefficients are drawn from.
    #[must_use]
    pub const fn stream(&self) -> u64 {
        self.stream
    }

    /// Whether each mapped draw also carries the residual of the trial-level
    /// line. A predictive law of the surrogate effect carried through a line is
    /// the law of a new trial's terminal effect, which includes the trial-level
    /// residual; a posterior of the mean does not.
    #[must_use]
    pub fn includes_residual(&self) -> bool {
        self.source_kind == ValueKind::PredictiveLaw
            && matches!(self.contract.mapping, BridgeMapping::TrialLevelLinear(_))
    }

    /// The identity of the contract.
    #[must_use]
    pub const fn contract_identity(&self) -> SemanticDigest {
        self.contract_identity
    }

    /// Identity of the source claim or draws object.
    #[must_use]
    pub const fn source_claim_id(&self) -> SemanticDigest {
        self.source_claim_id
    }

    /// Build the derived claim from the mapped draws' summary.
    ///
    /// # Errors
    /// A derived claim that would carry the identity of its own source.
    pub fn finish(
        self,
        summary: &MappedDrawsSummary,
    ) -> Result<BridgedDistributionClaim, BridgeRefusal> {
        let claim_id = bridged_distribution_claim_id(
            self.contract_identity,
            self.source_claim_id,
            self.stream,
            summary.draws_digest,
        );
        let identities =
            bridged_identities(&self.contract, self.contract_identity, &self.source_identities);
        let mut evidence = self.source_evidence.clone();
        let mut push = |item: String| {
            if !evidence.iter().any(|existing| **existing == *item) {
                evidence.push(Arc::from(item));
            }
        };
        push(format!("alignment:{}", self.alignment));
        push(format!("bridge:{}", self.contract_identity.to_hex()));
        push(format!("bridge_source:{}", self.source_claim_id.to_hex()));
        push(format!("bridge_kind:{}", self.contract.kind.as_str()));
        push(format!("bridge_draws:{}", summary.draws_digest.to_hex()));
        push(format!("bridge_stream:{}", self.stream));
        push(format!("bridge_reading:{}", crate::joint_law_rule::UNCALIBRATED_SPREAD_LABEL));
        if let Some(uncertainty) = &self.contract.mapping_uncertainty {
            push(format!("mapping_uncertainty:{}", uncertainty.id));
        }
        if let Some(dependence) = &self.contract.source_dependence {
            push(match dependence {
                SourceMappingDependence::Independent { evidence: name } => {
                    format!("source_dependence:independent:{name}")
                }
                SourceMappingDependence::SlopeCovariance { evidence: name, .. } => {
                    format!("source_dependence:covariance:{name}")
                }
            });
        }
        if let Some(bridge) = &self.contract.estimand_bridge {
            push(format!("estimand_bridge:{}", bridge.id));
        }
        let envelope = ClaimEnvelope::new(
            claim_id,
            identities,
            ClaimKind::Mixture,
            Some(summary.mean),
            Some(Arc::from(self.contract.target.quantity.units.as_str())),
            self.reasoning(),
            ClaimDomains::new(DomainStatus::Unknown, DomainStatus::Unknown, self.source_evaluated),
            self.source_execution,
            evidence,
        );

        let mut provenance = ProvenanceGraph::new();
        let source_hex: Arc<str> = Arc::from(self.source_claim_id.to_hex());
        let operation = format!("scientific_bridge_draws:{}", self.contract.kind.as_str());
        let record_failed = |_| provenance_failed();
        provenance
            .try_push(node(self.source_claim_id, "source_draws", &[]))
            .map_err(record_failed)?;
        provenance
            .try_push(node(claim_id, &operation, std::slice::from_ref(&source_hex)))
            .map_err(record_failed)?;
        let receipt = self.receipt(claim_id, operation);
        Ok(BridgedDistributionClaim {
            claim: envelope,
            contract: self.contract.clone(),
            contract_identity: self.contract_identity,
            source_claim_id: self.source_claim_id,
            value_kind: self.contract.target_value,
            mapping: self.contract.mapping.kind(),
            transported: self.licensing.transported,
            alignment_identity: self.alignment,
            stream: self.stream,
            n_draws: summary.n_draws,
            mean: summary.mean,
            variance: summary.variance,
            draws_digest: summary.draws_digest,
            provenance,
            receipt,
        })
    }

    fn reasoning(&self) -> ReasoningView {
        let mut obligations: Vec<ObligationRecord> =
            self.contract.assumptions.iter().map(obligation).collect();
        if let Some(premise) = &self.contract.transport {
            let mut record = ObligationRecord::new(
                format!("bridge_transport:{}", premise.id),
                ObligationScope::Program,
                AssumptionSource::UserDeclared,
                ObligationKind::UserAssertion,
                AssumptionStatus::Declared,
                premise.statement.as_str(),
            );
            record.evidence = premise.evidence.as_deref().map(Arc::from);
            obligations.push(record);
        }
        let source = if self.source_kind == ValueKind::Posterior {
            UncertaintyComponent::new(UncertaintySource::Parameter, "source_draws", false)
        } else {
            UncertaintyComponent::new(UncertaintySource::Sampling, "source_draws", false)
        };
        let mut components = vec![source];
        match (&self.contract.mapping, &self.contract.mapping_uncertainty) {
            (BridgeMapping::TrialLevelLinear(_), _) => components.push(UncertaintyComponent::new(
                UncertaintySource::Sampling,
                "trial_level_fit",
                false,
            )),
            (BridgeMapping::MeasurementCorrection(_), _) => components.push(
                UncertaintyComponent::new(UncertaintySource::Measurement, "rate_variances", false),
            ),
            (_, Some(uncertainty)) => components.push(UncertaintyComponent::new(
                UncertaintySource::Sampling,
                format!("mapping_variance:{}", uncertainty.id),
                false,
            )),
            _ => {}
        }
        if self.contract.source_dependence.is_some() {
            components.push(UncertaintyComponent::new(
                UncertaintySource::Sampling,
                "source_dependence",
                false,
            ));
        }
        ReasoningView::new(
            SlotAvailability::unavailable("bridge_assumptions_declared"),
            SlotAvailability::unavailable("bridge_support_not_established"),
            SlotAvailability::Available(UncertaintySlot::new(components)),
            SlotAvailability::Available(AssumptionSlot::new(obligations)),
        )
    }

    fn receipt(&self, claim_id: SemanticDigest, operation: String) -> HandoffReceipt {
        let mut unresolved: Vec<Arc<str>> = self
            .contract
            .assumptions
            .iter()
            .map(|assumption| Arc::<str>::from(format!("assumption:{}", assumption.id)))
            .collect();
        if let Some(premise) = &self.contract.transport {
            unresolved.push(Arc::from(format!("transportability:{}", premise.id)));
        }
        // The coefficients are drawn given the source draw alone; pairing the bridged column
        // with another column of the table assumes the mapping's evidence is independent of
        // that column's, which the bridge cannot check.
        unresolved.push(Arc::from("mapping_independent_of_sibling_columns"));
        let mut retained = vec![
            Arc::from("claim.envelope"),
            Arc::from("parents"),
            Arc::from("provenance"),
            Arc::from("joint_draws"),
            Arc::from("replicate_ids"),
        ];
        for (declared, name) in [
            (self.contract.mapping_uncertainty.is_some(), "mapping_uncertainty"),
            (self.contract.source_dependence.is_some(), "source_dependence"),
            (self.contract.estimand_bridge.is_some(), "estimand_bridge"),
        ] {
            if declared {
                retained.push(Arc::from(name));
            }
        }
        HandoffReceipt::new(
            self.source_claim_id,
            Some(claim_id),
            "scientific_bridge",
            operation,
            retained,
            [Arc::from("identification"), Arc::from("support")],
            unresolved,
            [Arc::from("calibrated_interval"), Arc::from("accept_as_identified")],
        )
    }
}

/// The summary of the mapped draws the derived claim binds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MappedDrawsSummary {
    /// Rows mapped.
    pub n_draws: usize,
    /// Weighted mean of the bridged column.
    pub mean: f64,
    /// Weighted variance of the bridged column.
    pub variance: f64,
    /// Digest of the mapped draw table.
    pub draws_digest: SemanticDigest,
}

/// The identity of the claim a contract derives from a source of draws.
///
/// It binds the contract identity, the source identity, the random stream of the
/// mapping's coefficients and the digest of the mapped draws, so the derived
/// claim is exactly one realization of the bridge.
#[must_use]
pub fn bridged_distribution_claim_id(
    contract: SemanticDigest,
    source: SemanticDigest,
    stream: u64,
    draws: SemanticDigest,
) -> SemanticDigest {
    let mut out = CanonicalWriter::new(TAG_DRAWS_CLAIM);
    out.word(BRIDGE_CLAIM_IDENTITY_VERSION)
        .bytes(contract.as_bytes())
        .bytes(source.as_bytes())
        .word(stream)
        .bytes(draws.as_bytes());
    SemanticDigest::from_bytes(out.finish())
}

/// A claim derived across a bridge from the draws of a posterior or a
/// predictive law.
///
/// The source is not modified. The claim's value is the mean of the bridged
/// draws, a summary: the claim is the aligned draws, whose spread is read as
/// `uncalibrated_spread` unless a measured coverage record licenses an interval.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct BridgedDistributionClaim {
    /// The derived claim envelope.
    pub claim: ClaimEnvelope,
    /// The contract the claim was derived under.
    pub contract: BridgeContract,
    /// Identity of the contract.
    pub contract_identity: SemanticDigest,
    /// Identity of the source draws.
    pub source_claim_id: SemanticDigest,
    /// The kind of object the bridged draws are.
    pub value_kind: ValueKind,
    /// The mapping that carried the draws.
    pub mapping: MappingKind,
    /// Whether the bridge carries evidence established elsewhere than the
    /// claim's population or regime, under a declared premise.
    pub transported: bool,
    /// The pairing identity of the draw table, unchanged by the bridge.
    pub alignment_identity: String,
    /// The random stream the mapping's coefficients were drawn from.
    pub stream: u64,
    /// Rows mapped.
    pub n_draws: usize,
    /// Mean of the bridged column.
    pub mean: f64,
    /// Variance of the bridged column.
    pub variance: f64,
    /// Digest of the mapped draw table.
    pub draws_digest: SemanticDigest,
    /// Ancestry: the source draws and this derivation.
    pub provenance: ProvenanceGraph,
    /// Derivation receipt: what was retained, omitted and left unresolved.
    pub receipt: HandoffReceipt,
}

impl BridgedDistributionClaim {
    /// The bridge kind.
    #[must_use]
    pub const fn kind(&self) -> BridgeKind {
        self.contract.kind
    }

    /// Check the claim carries the kind of object a consumer requires.
    ///
    /// # Errors
    /// `bridge_substitution_refused` naming the held and requested kinds: the
    /// bridged draws are a posterior or a predictive law, never a mean, an
    /// identified set or a joint outcome law.
    pub fn require_value_kind(&self, requested: ValueKind) -> Result<(), BridgeRefusal> {
        if self.value_kind == requested {
            Ok(())
        } else {
            Err(wrong_value_kind(requested, self.value_kind))
        }
    }

    /// Freshness of the bridged claim given the freshness of its source: a
    /// withdrawn or superseded source makes it inapplicable, and the snapshot is
    /// the source's.
    #[must_use]
    pub fn freshness(&self, source: &ClaimFreshness) -> ClaimFreshness {
        let retired = source.withdrawn || source.superseded_by.is_some();
        ClaimFreshness {
            historically_valid: source.historically_valid,
            currently_applicable: if retired { Some(false) } else { source.currently_applicable },
            locally_ready: source.locally_ready,
            superseded_by: None,
            withdrawn: false,
            as_of_snapshot: source.as_of_snapshot,
        }
    }

    /// An unscoped request for an interval of the bridged draws.
    ///
    /// # Errors
    /// Always `cell_not_licensed` with `scientific_bridge.interval_not_licensed`:
    /// name a level and ask for the interval through the measured-coverage gate.
    pub fn request_interval(&self) -> Result<(), BridgeRefusal> {
        Err(interval_not_licensed())
    }
}

#[cfg(test)]
#[path = "draws_tests.rs"]
mod tests;
