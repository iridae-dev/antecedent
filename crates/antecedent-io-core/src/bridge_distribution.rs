//! Bridges of draws: a posterior or predictive law carried across a scientific
//! bridge as aligned draws.
//!
//! The source is one column of an aligned joint draw table
//! ([`DistributionArtifact`]). The bridged result is the same table with that
//! column replaced by the bridged quantity: row `i` of the output is the image of
//! row `i` of the input, every other column keeps its place, and the table's
//! alignment identity is unchanged, so the bridged column still pairs by index
//! with every column it was sampled with. A bridge that destroyed the pairing
//! would silently turn a joint law into marginals.
//!
//! # What is drawn
//!
//! * An identity or declared affine map maps each draw exactly, unless the
//!   contract declares uncertain coefficients ([`MappingUncertainty`]). Then each
//!   row also draws the intercept and slope from their declared joint law,
//!   conditionally on that row's source draw and the declared dependence
//!   ([`SourceMappingDependence`]): `Cov(slope, source)` is the only link between
//!   the mapping and the source, the source's own spread comes only from the
//!   draws and the mapping's only from its variances, so neither is counted twice.
//!   The mean and variance of the result agree with the first-order delta method
//!   of [`BridgeContract::propagate_affine_uncertainty`], which for a normal
//!   mapping law is exact.
//! * A trial-level line draws its coefficients from the Student-t law of the
//!   least-squares coefficients (`k - 2` degrees of freedom), as a scale mixture:
//!   one `sqrt(nu / chi2_nu)` factor per row scales the coefficients' innovation
//!   and, for a predictive law, the new trial's residual. Every draw must lie in
//!   the validated surrogate range; a law is neither truncated nor extrapolated.
//! * A measurement correction draws sensitivity and specificity from Beta laws
//!   of their declared means and variances and applies the correction row by
//!   row. A recorded prevalence is possible only for rates with
//!   `1 - specificity <= recorded <= sensitivity` and an identified Youden index,
//!   so each row redraws its rates (up to a bound) until the pair is compatible
//!   with its recorded value: the rates are drawn from their declared laws
//!   conditioned on the recorded value being attainable. A row no admissible
//!   rates produce is refused, not clipped.
//!
//! The coefficients are drawn from a deterministic stream of `seed`, so the
//! mapped draws are a function of the contract, the source table and the seed
//! and a consumer replays them bit for bit.
//!
//! # What is not claimed
//!
//! The bridged distribution is read as an uncalibrated spread. An interval is
//! requested through the measured-coverage gate (`antecedent_io::bridge_interval`
//! and its distribution counterpart) and is otherwise unlicensed. The bridge's
//! declared assumptions stay unresolved obligations, and pairing the bridged
//! column with another column of the table assumes the mapping's evidence is
//! independent of the evidence behind that column.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::bridge::{
    AffineDrawLaw, BridgeContract, BridgeDrawsSource, BridgeRefusal, BridgedDistributionClaim,
    CorrectionDrawLaw, DrawMapping, DrawsPlan, MappedDrawsSummary, MeasurementDirection,
    SourceMappingDependence, ValueKind, chain_stage_not_affine, draws_outside_domain,
    draws_unusable,
};
use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{LawMeaning, StageKind};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, ExternalRefusal, ReasoningView,
    ScientificQuantity, SlotAvailability,
};
use antecedent_core::{
    CanonicalWriter, CausalRng, ContractIdentities, SemanticDigest, derive_stream,
};
use antecedent_kernels::{sample_gamma, standard_normal};
use std::sync::Arc;

use crate::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionTrust, DrawAlignment,
};
use crate::propagation_chain_record::{
    ChainCoefficientWire, ChainOpWire, ChainSourceWire, ChainStageWire,
};
use crate::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};

/// Domain word of the random stream the mapping's coefficients are drawn from.
/// The stream of a bridge is `derive_stream(seed, BRIDGE_DRAWS_STREAM)`.
pub const BRIDGE_DRAWS_STREAM: u64 = 0x5C1B_D2A7_0000_0001;

const TAG_DRAWS: &str = "antecedent.scientific_bridge.draws_table";
const TAG_SOURCE: &str = "antecedent.scientific_bridge.draws_source";

/// The coefficient draws of an affine bridge, one per row.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineColumns {
    /// The drawn slope of each row.
    pub slopes: Vec<f64>,
    /// The drawn intercept of each row.
    pub intercepts: Vec<f64>,
}

/// A distribution carried across a bridge.
#[derive(Clone, Debug)]
pub struct BridgedDistribution {
    claim: BridgedDistributionClaim,
    distribution: DistributionArtifact,
    source: DistributionArtifact,
    source_quantity: ScientificQuantity,
    column: usize,
    source_mean: f64,
    source_variance: f64,
    affine: Option<AffineColumns>,
    residual: bool,
    alignment: String,
}

impl BridgedDistribution {
    /// The derived claim.
    #[must_use]
    pub const fn claim(&self) -> &BridgedDistributionClaim {
        &self.claim
    }

    /// The derived claim envelope.
    #[must_use]
    pub const fn envelope(&self) -> &ClaimEnvelope {
        &self.claim.claim
    }

    /// The bridged draw table: the source table with the bridged column replaced.
    #[must_use]
    pub const fn distribution(&self) -> &DistributionArtifact {
        &self.distribution
    }

    /// The source draw table, unchanged.
    #[must_use]
    pub const fn source(&self) -> &DistributionArtifact {
        &self.source
    }

    /// The source table as a claim envelope: a mixture of its draws whose value is
    /// their mean, with the identity layers the table's provenance gives it. It is
    /// the source a bridge artifact records, rebuilt identically from the table.
    #[must_use]
    pub fn source_envelope(&self) -> ClaimEnvelope {
        ClaimEnvelope::new(
            self.claim.source_claim_id,
            table_identities(&self.source, &self.source_quantity),
            ClaimKind::Mixture,
            Some(self.source_mean),
            Some(Arc::from(self.source_quantity.units.as_str())),
            ReasoningView::new(
                SlotAvailability::unavailable("replayed_source"),
                SlotAvailability::unavailable("replayed_source"),
                SlotAvailability::unavailable("replayed_source"),
                SlotAvailability::unavailable("replayed_source"),
            ),
            ClaimDomains::new(
                DomainStatus::Unknown,
                DomainStatus::Unknown,
                DomainStatus::Evaluated,
            ),
            None,
            Vec::<Arc<str>>::new(),
        )
    }

    /// The column of both tables the bridge replaced.
    #[must_use]
    pub const fn column(&self) -> usize {
        self.column
    }

    /// The pairing identity both tables share.
    #[must_use]
    pub fn alignment_identity(&self) -> &str {
        &self.alignment
    }

    /// Weighted mean of the source column.
    #[must_use]
    pub const fn source_mean(&self) -> f64 {
        self.source_mean
    }

    /// Weighted variance of the source column: the source standard error is its
    /// square root.
    #[must_use]
    pub const fn source_variance(&self) -> f64 {
        self.source_variance
    }

    /// The drawn coefficients of an affine bridge.
    #[must_use]
    pub const fn affine_columns(&self) -> Option<&AffineColumns> {
        self.affine.as_ref()
    }

    /// The bridged column, one value per row.
    #[must_use]
    pub fn mapped_column(&self) -> Vec<f64> {
        column_of(&self.distribution, self.column)
    }

    /// The source column, one value per row.
    #[must_use]
    pub fn source_column(&self) -> Vec<f64> {
        column_of(&self.source, self.column)
    }

    /// The bridge as a propagation-chain measurement-mapping stage.
    ///
    /// The stage reads three columns of one aligned draw table, the source
    /// draws, the drawn slope and the drawn intercept, and applies
    /// `source * slope + intercept` with the slope and the intercept as the
    /// stage's own inputs (coefficients that are inputs, not constants). A chain
    /// that reads the stage's output in a transport or a utility therefore
    /// propagates the source, the mapping's coefficients and every later stage
    /// jointly, row by row. The declared dependence of the source and the
    /// mapping enters the chain as a shared factor, and the stage brings the
    /// contract as its own.
    ///
    /// # Errors
    /// A bridge that is not an affine map of its inputs: a measurement
    /// correction, or a predictive law carried through a trial-level line, whose
    /// per-row residual is not a coefficient. Deriving the bridged draws and
    /// supplying them to the chain as a source carries those.
    pub fn chain_stage(&self, stage_id: &str) -> Result<BridgeChainStage, BridgeRefusal> {
        let Some(affine) = &self.affine else {
            return Err(chain_stage_not_affine(self.claim.mapping.as_str()));
        };
        if self.residual {
            return Err(chain_stage_not_affine("predictive law through a trial-level line"));
        }
        let contract = self.claim.contract_identity.to_hex();
        let source_id = format!("{stage_id}.source");
        let slope_id = format!("{stage_id}.slope");
        let intercept_id = format!("{stage_id}.intercept");
        let meaning = LawMeaning::Posterior;
        let mapping_factor = format!("bridge_mapping:{contract}");
        let mut source =
            ChainSourceWire::draws(&source_id, meaning, &self.alignment, &self.source_column());
        let mut slope = ChainSourceWire::draws(&slope_id, meaning, &self.alignment, &affine.slopes)
            .with_factor(SharedFactorKind::Parameter, &mapping_factor);
        let intercept =
            ChainSourceWire::draws(&intercept_id, meaning, &self.alignment, &affine.intercepts)
                .with_factor(SharedFactorKind::Parameter, &mapping_factor);
        if let Some(SourceMappingDependence::SlopeCovariance { evidence, .. }) =
            &self.claim.contract.source_dependence
        {
            // A declared covariance is a shared factor: the slope was drawn given the source draw.
            let shared = format!("bridge_dependence:{evidence}");
            source = source.with_factor(SharedFactorKind::Model, &shared);
            slope = slope.with_factor(SharedFactorKind::Model, &shared);
        }
        let stage = ChainStageWire::new(
            stage_id,
            StageKind::MeasurementMapping,
            &[source_id.as_str(), slope_id.as_str(), intercept_id.as_str()],
            ChainOpWire::affine(ChainCoefficientWire::input(1), ChainCoefficientWire::input(2)),
        )
        .with_own_factor(SharedFactorKind::Model, &format!("bridge_contract:{contract}"));
        Ok(BridgeChainStage { sources: vec![source, slope, intercept], stage })
    }
}

/// A bridge expressed as a stage of a propagation chain, with the sources it
/// reads.
#[derive(Clone, Debug)]
pub struct BridgeChainStage {
    /// The source draws, the slope draws and the intercept draws, columns of one
    /// aligned draw table.
    pub sources: Vec<ChainSourceWire>,
    /// The measurement-mapping stage, whose coefficients are inputs.
    pub stage: ChainStageWire,
}

fn column_of(table: &DistributionArtifact, column: usize) -> Vec<f64> {
    let width = table.shape()[1];
    table.draws().chunks_exact(width).map(|row| row[column]).collect()
}

fn refusal_from_text(
    code: &'static str,
    detail: &str,
    expected: String,
    supplied: String,
    remedy: &'static str,
) -> ExternalRefusal {
    ExternalRefusal::new(code, "bind", detail)
        .with_expected(expected)
        .with_supplied(supplied)
        .with_remedy(remedy)
}

/// Digest of a draw table: its alignment, width and every cell.
fn table_digest(alignment: &str, width: usize, draws: &[f64]) -> SemanticDigest {
    let mut writer = CanonicalWriter::new(TAG_DRAWS);
    writer.text(alignment).word(width as u64).reals(draws);
    SemanticDigest::from_bytes(writer.finish())
}

fn text_digest<S: AsRef<str>>(role: &str, parts: &[S]) -> SemanticDigest {
    let mut writer = CanonicalWriter::new(TAG_SOURCE);
    writer.text(role);
    for part in parts {
        writer.text(part.as_ref());
    }
    SemanticDigest::from_bytes(writer.finish())
}

/// What a distribution's meaning says about the kind of value its draws are.
fn value_kind_of(meaning: DistributionMeaningWire) -> (Option<ValueKind>, &'static str) {
    match meaning {
        DistributionMeaningWire::ParameterPosterior => {
            (Some(ValueKind::Posterior), "parameter_posterior")
        }
        DistributionMeaningWire::CausalFunctionalPosterior => {
            (Some(ValueKind::Posterior), "causal_functional_posterior")
        }
        DistributionMeaningWire::PosteriorPredictive => {
            (Some(ValueKind::PredictiveLaw), "posterior_predictive")
        }
        DistributionMeaningWire::InterventionalPredictive => {
            (Some(ValueKind::PredictiveLaw), "interventional_predictive")
        }
        DistributionMeaningWire::EstimatorSampling => (None, "estimator_sampling"),
        DistributionMeaningWire::Bootstrap => (None, "bootstrap"),
        DistributionMeaningWire::EmpiricalOutcome => (None, "empirical_outcome"),
    }
}

/// The identity layers a table of draws carries as a claim source: digests of
/// the strings its distribution identity binds, so two bridges of one table share
/// evidence and a table of another snapshot does not.
fn table_identities(
    source: &DistributionArtifact,
    quantity: &ScientificQuantity,
) -> ContractIdentities {
    let identity = &source.metadata().identity;
    ContractIdentities::new(
        text_digest("target", &[&quantity.canonical_identity()]),
        text_digest("identification", &[&identity.causal_contract_id]),
        None,
        None,
        text_digest("inference_binding", &[&identity.provider_id, &identity.rng_id]),
        text_digest("observation", &[&identity.source_id]),
        text_digest("data_snapshot", &[&identity.snapshot_id]),
    )
}

/// Weights of the rows, normalized to sum to one.
fn row_weights(source: &DistributionArtifact) -> Vec<f64> {
    let rows = source.n_draws();
    match &source.metadata().weights {
        Some(weights) => {
            let total: f64 = weights.iter().sum();
            weights.iter().map(|w| w / total).collect()
        }
        None => vec![1.0 / rows as f64; rows],
    }
}

fn weighted_moments(values: &[f64], weights: &[f64]) -> (f64, f64) {
    let mean: f64 = values.iter().zip(weights).map(|(v, w)| v * w).sum();
    let variance: f64 = values.iter().zip(weights).map(|(v, w)| w * (v - mean) * (v - mean)).sum();
    (mean, variance)
}

/// A Beta draw of the given mean and variance; the mean itself when the rate is exact.
fn beta_draw(mean: f64, variance: f64, rng: &mut CausalRng) -> f64 {
    if variance == 0.0 {
        return mean;
    }
    let concentration = mean * (1.0 - mean) / variance - 1.0;
    let first = sample_gamma(mean * concentration, 1.0, rng);
    let second = sample_gamma((1.0 - mean) * concentration, 1.0, rng);
    first / (first + second)
}

struct Mapped {
    values: Vec<f64>,
    affine: Option<AffineColumns>,
}

fn map_affine(
    law: &AffineDrawLaw,
    source: &[f64],
    mean: f64,
    variance: f64,
    residual: bool,
    rng: &mut CausalRng,
) -> Result<Mapped, BridgeRefusal> {
    if let Some((low, high)) = law.domain {
        let outside: Vec<f64> =
            source.iter().copied().filter(|value| *value < low || *value > high).collect();
        if let Some(first) = outside.first() {
            return Err(draws_outside_domain(outside.len(), source.len(), *first));
        }
    }
    let innovation = law.innovation(variance)?;
    let residual_sd = law.residual_variance.sqrt();
    let mut values = Vec::with_capacity(source.len());
    let mut slopes = Vec::with_capacity(source.len());
    let mut intercepts = Vec::with_capacity(source.len());
    for &draw in source {
        let scale = match law.degrees_of_freedom {
            Some(df) => (df / sample_gamma(df / 2.0, 0.5, rng)).sqrt(),
            None => 1.0,
        };
        let z_slope = standard_normal(rng);
        let z_intercept = standard_normal(rng);
        let slope = law.slope_mean
            + innovation.slope_loading * (draw - mean)
            + scale * innovation.slope_sd * z_slope;
        let intercept = law.intercept_mean
            + scale
                * (innovation.intercept_on_slope * z_slope + innovation.intercept_sd * z_intercept);
        let mut mapped = intercept + slope * draw;
        if residual {
            mapped += scale * residual_sd * standard_normal(rng);
        }
        values.push(mapped);
        slopes.push(slope);
        intercepts.push(intercept);
    }
    Ok(Mapped { values, affine: Some(AffineColumns { slopes, intercepts }) })
}

/// Most times a row redraws its rates looking for a pair the recorded value is
/// compatible with.
const MAX_RATE_ATTEMPTS: usize = 64;

fn map_correction(
    law: &CorrectionDrawLaw,
    source: &[f64],
    rng: &mut CausalRng,
) -> Result<Mapped, BridgeRefusal> {
    let exact = law.sensitivity_variance == 0.0 && law.specificity_variance == 0.0;
    let mut values = Vec::with_capacity(source.len());
    let mut outside = 0_usize;
    let mut first = None;
    for &draw in source {
        let mut mapped = None;
        for _ in 0..MAX_RATE_ATTEMPTS {
            let sensitivity = beta_draw(law.sensitivity, law.sensitivity_variance, rng);
            let specificity = beta_draw(law.specificity, law.specificity_variance, rng);
            let youden = sensitivity + specificity - 1.0;
            match law.direction {
                MeasurementDirection::MeasuredToLatent => {
                    if youden > law.min_youden && draw >= 1.0 - specificity && draw <= sensitivity {
                        mapped = Some((draw + specificity - 1.0) / youden);
                    }
                }
                MeasurementDirection::LatentToMeasured => {
                    if !(0.0..=1.0).contains(&draw) {
                        // No rate makes a latent prevalence outside [0, 1] a probability.
                        break;
                    }
                    if youden > law.min_youden {
                        mapped = Some(sensitivity * draw + (1.0 - specificity) * (1.0 - draw));
                    }
                }
            }
            if mapped.is_some() || exact {
                break;
            }
        }
        if let Some(value) = mapped {
            values.push(value);
        } else {
            outside += 1;
            first.get_or_insert(draw);
            values.push(f64::NAN);
        }
    }
    if let Some(first) = first {
        return Err(draws_outside_domain(outside, source.len(), first));
    }
    Ok(Mapped { values, affine: None })
}

/// Bridge the draws of `source` across `contract`.
///
/// `source` is the aligned table whose column about the contract's source
/// quantity is bridged. Its identity layers are digests of the provenance its
/// distribution identity binds, so a bridge of the table is replayable from the
/// table alone. `claim_kind`, when the draws come from a claim, is the kind of
/// that claim: an associational, refused, bounded or incomplete claim is never a
/// bridge source. `seed` selects the random stream of the mapping's coefficients.
///
/// # Errors
/// A contract that does not license a draws bridge, a source of draws that is not
/// a posterior or a predictive law of the contract's quantity or has no row
/// pairing, a source claim that is not about draws, draws outside the range the
/// mapping is licensed for, a covariance no joint law has, or a mapped draw that
/// is not finite.
#[allow(
    clippy::too_many_lines,
    reason = "one linear derivation: locate the column, plan, draw per mapping, assemble the table"
)]
pub fn bridge_draws(
    contract: &BridgeContract,
    source: &DistributionArtifact,
    claim_kind: Option<ClaimKind>,
    seed: u64,
) -> Result<BridgedDistribution, BridgeRefusal> {
    let identity = &source.metadata().identity;
    // The column about the contract's source quantity.
    let mut column = None;
    for (index, wire) in identity.quantities.iter().enumerate() {
        let Ok(quantity) = ScientificQuantity::try_from(wire.clone()) else {
            return Err(draws_unusable("quantities", "the draw table's quantities do not convert"));
        };
        if quantity.require_same_coordinate(&contract.source.quantity).is_ok() {
            column = Some((index, quantity));
        }
    }
    let Some((column, quantity)) = column else {
        return Err(refusal_from_text(
            antecedent_core::reason_code!("bridge_binding_mismatch"),
            "scientific_bridge.quantity_mismatch",
            contract.source.quantity.canonical_identity(),
            identity
                .quantities
                .iter()
                .map(|wire| wire.variable_id.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            "bridge a draw table that carries a column about the contract's source quantity",
        ));
    };
    if source.metadata().supported.as_ref().is_some_and(|mask| !mask[column]) {
        return Err(draws_unusable("column", "the source column is masked as unsupported"));
    }
    let alignment_state = identity.alignment == DrawAlignment::Joint;
    let alignment = if alignment_state {
        source.alignment_identity().map_err(|_| {
            draws_unusable("alignment", "the draw table's pairing identity could not be encoded")
        })?
    } else {
        String::new()
    };
    let (value_kind, meaning) = value_kind_of(identity.semantic);
    let table_claim_id = table_digest(&alignment, source.shape()[1], source.draws());
    let table_ids = table_identities(source, &quantity);
    let planned: DrawsPlan = {
        let bridge_source = BridgeDrawsSource {
            quantity: &quantity,
            value_kind,
            meaning,
            claim_kind: claim_kind.unwrap_or(ClaimKind::Mixture),
            claim_id: table_claim_id,
            identities: &table_ids,
            execution: None,
            evidence: &[],
            evaluated: DomainStatus::Evaluated,
            joint: alignment_state,
            alignment_identity: &alignment,
        };
        contract.plan_distribution(&bridge_source, seed)?
    };

    let values = column_of(source, column);
    let weights = row_weights(source);
    let (mean, variance) = weighted_moments(&values, &weights);
    if !(mean.is_finite() && variance.is_finite()) {
        return Err(draws_unusable("source", "the source draws have no finite mean and variance"));
    }
    let mut rng = CausalRng::from_seed(derive_stream(seed, BRIDGE_DRAWS_STREAM));
    let residual = planned.includes_residual();
    let mapped = match planned.mapping() {
        DrawMapping::Affine(law) => map_affine(law, &values, mean, variance, residual, &mut rng)?,
        DrawMapping::Correction(law) => map_correction(law, &values, &mut rng)?,
        _ => {
            return Err(draws_unusable("mapping", "the mapping has no draws propagation"));
        }
    };
    if mapped.values.iter().any(|value| !value.is_finite()) {
        return Err(draws_unusable(
            "mapped draw",
            "the mapped draws overflowed the finite range; state a smaller map",
        ));
    }

    // The source table with the bridged column replaced, row for row.
    let width = source.shape()[1];
    let mut table = source.draws().to_vec();
    for (row, value) in mapped.values.iter().enumerate() {
        table[row * width + column] = *value;
    }
    let mut metadata = source.metadata().clone();
    metadata.identity.quantities[column] = ScientificQuantityWire::from(&contract.target.quantity);
    metadata.calibration = DistributionCalibration::Unmeasured;
    metadata.trust = DistributionTrust::Unverified;
    let distribution = DistributionArtifact::new(metadata, table.clone()).map_err(|error| {
        draws_unusable(
            "target",
            if matches!(error, crate::IoError::Convert(_)) {
                "the target quantity repeats a coordinate of the draw table"
            } else {
                "the bridged draw table is not a valid distribution"
            },
        )
    })?;
    let (mapped_mean, mapped_variance) = weighted_moments(&mapped.values, &weights);
    let summary = MappedDrawsSummary {
        n_draws: source.n_draws(),
        mean: mapped_mean,
        variance: mapped_variance,
        draws_digest: table_digest(&alignment, width, &table),
    };
    let claim = planned.finish(&summary)?;
    Ok(BridgedDistribution {
        claim,
        distribution,
        source: source.clone(),
        source_quantity: quantity,
        column,
        source_mean: mean,
        source_variance: variance,
        affine: mapped.affine,
        residual,
        alignment,
    })
}

#[cfg(test)]
#[path = "bridge_distribution_tests.rs"]
mod tests;
