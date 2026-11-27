//! Equal-tailed credible intervals from a supplied joint posterior of an affine map and its input.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::IoError;
use crate::bridge_joint_outcomes::JointOutcomeBridgeError;
use crate::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionTrust, DrawAlignment,
};
use crate::quantity_wire::DistributionMeaningWire;
use antecedent_contract::ScientificQuantity;
use antecedent_contract::bridge::{
    BridgeContract, BridgeMapping, FixedBridgeDomain, SupportMap, ValueKind, draws_not_aligned,
    draws_outside_domain, draws_unusable, support_image,
};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_stats::joint_law::{MAX_MARGINAL_ATOMS, Marginal};

/// The source table has three distinct named columns: the input, and posterior deviations
/// from the intercept and slope in the declared affine contract.
#[derive(Clone, Debug)]
pub struct PosteriorAffineBridge {
    /// Reference map, scientific endpoints, scope, and identifying assumptions.
    pub contract: BridgeContract,
    /// Posterior shift of the intercept, measured in target units.
    pub intercept_shift: ScientificQuantity,
    /// Posterior shift of the slope, in units of target per unit of source.
    pub slope_shift: ScientificQuantity,
    /// Valid source range of the reference map and its coefficient shifts.
    pub domain: FixedBridgeDomain,
}

/// A credible interval of the mapped posterior, with its attained mass in the supplied table.
#[derive(Clone, Debug)]
pub struct PosteriorBridgeInterval {
    /// Target coordinate of the interval.
    pub quantity: ScientificQuantity,
    /// The target column replaces the source column. Every sibling row retains its pairing.
    pub posterior: DistributionArtifact,
    /// Requested posterior probability.
    pub level: f64,
    /// Lower and upper generalized-inverse quantiles at equal tail probabilities.
    pub endpoints: [f64; 2],
    /// Actual probability of the closed interval; an atom at an endpoint can increase this.
    pub attained_mass: f64,
    /// Digest of the contract, source posterior, coefficient bindings, domain and request.
    pub identity: SemanticDigest,
}

impl PosteriorAffineBridge {
    /// Calculate the joint posterior pushforward and take quantiles of the resulting measure.
    ///
    /// # Errors
    /// A pair of marginals, an unsupported or ambiguous coordinate, a second coefficient
    /// uncertainty model, an uncovered source row, or an inadmissible request.
    pub fn interval(
        &self,
        source: &DistributionArtifact,
        level: f64,
        ctx: &ExecutionContext,
    ) -> Result<PosteriorBridgeInterval, JointOutcomeBridgeError> {
        let (intercept, slope) = self.checked_map(source, level)?;
        if source.n_draws() > MAX_MARGINAL_ATOMS {
            return Err(IoError::TooLarge.into());
        }
        let bytes = (source.draws().len() as u64)
            .saturating_mul(32)
            .saturating_add((source.n_draws() as u64).saturating_mul(96));
        let _lease = ctx.acquire_serial_numerical_admission(bytes).map_err(IoError::from)?;
        let [x, a, b] = self.coordinates(source)?;
        let domain = support_image(SupportMap::Identity, self.domain.0)?;
        let mut draws = source.draws().to_vec();
        let mut values = Vec::with_capacity(source.n_draws());
        for row in draws.chunks_exact_mut(source.shape()[1]) {
            if ctx.cancellation.is_cancelled() {
                return Err(antecedent_contract::bridge::propagation_cancelled().into());
            }
            if row[x] < domain.low || row[x] > domain.high {
                return Err(draws_outside_domain(1, source.n_draws(), row[x]).into());
            }
            let value = (intercept + row[a]) + (slope + row[b]) * row[x];
            if !value.is_finite() {
                return Err(draws_unusable(
                    "posterior row",
                    "mapped posterior values must be finite",
                )
                .into());
            }
            row[x] = value;
            values.push(value);
        }
        let law = match &source.metadata().weights {
            Some(weights) => {
                let total: f64 = weights.iter().sum();
                let masses: Vec<_> = weights.iter().map(|w| w / total).collect();
                Marginal::from_weighted(&values, &masses)
            }
            None => Marginal::from_draws(&values),
        }
        .map_err(|_| draws_unusable("posterior weights", "supply a finite positive total mass"))?;
        let tail = (1. - level) * 0.5;
        let [Ok(lower), Ok(upper)] = [tail, 1. - tail].map(|p| law.quantile(p)) else {
            return Err(draws_unusable(
                "posterior quantile",
                "supply a representable tail probability",
            )
            .into());
        };
        let endpoints = [lower, upper];
        let attained_mass = law.mass_at_most(endpoints[1]) - law.mass_below(endpoints[0]);
        let mut key = CanonicalWriter::new("antecedent.bridge.affine_joint_posterior.v1");
        key.bytes(self.contract.identity().as_bytes())
            .text(&self.intercept_shift.canonical_identity())
            .text(&self.slope_shift.canonical_identity())
            .word(domain.low.to_bits())
            .word(domain.high.to_bits())
            .word(level.to_bits())
            .bytes(&source.to_bytes("joint-posterior")?);
        let mut metadata = source.metadata().clone();
        metadata.identity.quantities[x] = (&self.contract.target.quantity).into();
        metadata.calibration = DistributionCalibration::NotMeasured;
        metadata.trust = DistributionTrust::Unverified;
        metadata.legacy_posterior = None;
        metadata.legacy_bindings = None;
        let posterior = DistributionArtifact::new(metadata, draws)?;
        Ok(PosteriorBridgeInterval {
            quantity: self.contract.target.quantity.clone(),
            posterior,
            level,
            endpoints,
            attained_mass,
            identity: SemanticDigest::from_bytes(key.finish()),
        })
    }

    fn checked_map(
        &self,
        source: &DistributionArtifact,
        level: f64,
    ) -> Result<(f64, f64), JointOutcomeBridgeError> {
        self.contract.check_draws_support()?;
        let BridgeMapping::Affine { intercept, slope } = self.contract.mapping else {
            return Err(draws_unusable(
                "posterior map",
                "supply an affine reference for the joint posterior shifts",
            )
            .into());
        };
        if self.contract.source_value != ValueKind::Posterior
            || self.contract.target_value != ValueKind::Posterior
            || self.contract.mapping_uncertainty.is_some()
            || self.contract.source_dependence.is_some()
            || !matches!(
                source.semantic(),
                DistributionMeaningWire::ParameterPosterior
                    | DistributionMeaningWire::CausalFunctionalPosterior
            )
        {
            return Err(draws_unusable("posterior source", "bind all coefficient uncertainty in one joint posterior over source and coefficient shifts").into());
        }
        if source.metadata().identity.alignment != DrawAlignment::Joint {
            return Err(draws_not_aligned().into());
        }
        if !(0.0 < level && level < 1.0) {
            return Err(draws_unusable(
                "credible mass",
                "supply a posterior probability strictly between zero and one",
            )
            .into());
        }
        Ok((intercept, slope))
    }

    fn coordinates(
        &self,
        source: &DistributionArtifact,
    ) -> Result<[usize; 3], JointOutcomeBridgeError> {
        let coefficient_units = format!(
            "{}/{}",
            self.contract.target.quantity.units, self.contract.source.quantity.units
        );
        if self.intercept_shift.units != self.contract.target.quantity.units
            || self.slope_shift.units != coefficient_units
        {
            return Err(draws_unusable("coefficient units", "declare the intercept shift in target units and the slope shift in target/source units").into());
        }
        let find = |wanted: &ScientificQuantity| -> Result<usize, JointOutcomeBridgeError> {
            source
                .quantities()
                .iter()
                .position(|wire| {
                    ScientificQuantity::try_from(wire.clone())
                        .is_ok_and(|q| q.require_same_coordinate(wanted).is_ok())
                })
                .filter(|i| source.metadata().supported.as_ref().is_none_or(|mask| mask[*i]))
                .ok_or_else(|| {
                    draws_unusable(
                        "posterior coordinate",
                        "supply a supported posterior column for each declared binding",
                    )
                    .into()
                })
        };
        let x = find(&self.contract.source.quantity)?;
        let a = find(&self.intercept_shift)?;
        let b = find(&self.slope_shift)?;
        if x == a || x == b || a == b {
            return Err(draws_unusable(
                "posterior coordinates",
                "bind three distinct columns of the same joint posterior",
            )
            .into());
        }
        Ok([x, a, b])
    }
}
