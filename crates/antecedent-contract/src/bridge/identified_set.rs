//! Exact images of scalar identified sets under declared, fixed mappings.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, IdentifiedSet, SemanticDigest};

use super::refusal::{mapping_uncertainty_invalid, outside_validated_range, quantity_mismatch};
use super::{
    BridgeContract, BridgeMapping, BridgeRefusal, SupportMap, ValueInterval, ValueKind,
    support_image,
};
use crate::ScientificQuantity;

/// Source set and the retained identity of the evidence establishing it.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentifiedSetSource {
    /// The coordinate of both endpoints.
    pub quantity: ScientificQuantity,
    /// All values consistent with the identifying premises.
    pub set: IdentifiedSet<f64>,
    /// Identity of the source claim from which the set was read.
    pub claim_identity: SemanticDigest,
}

/// The interval of input values on which a fixed bridge is declared valid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedBridgeDomain(pub ValueInterval);

impl FixedBridgeDomain {
    /// Check the declaration and both extrema of a source set before evaluating the mapping.
    pub(super) fn covers(self, values: ValueInterval) -> Result<(), BridgeRefusal> {
        let range = support_image(SupportMap::Identity, self.0)?;
        let source = support_image(SupportMap::Identity, values)?;
        for endpoint in [source.low, source.high] {
            if endpoint < range.low || endpoint > range.high {
                return Err(outside_validated_range(range.low, range.high, endpoint));
            }
        }
        Ok(())
    }
}

/// Image under the contract and source identified by this result.
#[derive(Clone, Debug, PartialEq)]
pub struct BridgedIdentifiedSet {
    /// Target quantity, including its population, units and scale.
    pub quantity: ScientificQuantity,
    /// Sharp image of the source interval.
    pub set: IdentifiedSet<f64>,
    /// Digest of the contract, original claim, source bounds and valid input range.
    pub identity: SemanticDigest,
}

impl BridgeContract {
    /// Admit a deterministic affine map on the named input and output coordinates.
    pub(super) fn fixed_map_for(&self, kind: ValueKind) -> Result<SupportMap, BridgeRefusal> {
        self.check_support_carrying(&[kind])?;
        if self.mapping_uncertainty.is_some() || self.source_dependence.is_some() {
            return Err(mapping_uncertainty_invalid(
                "mapping",
                "declare a fixed mapping to compute its exact image",
            ));
        }
        match self.mapping {
            BridgeMapping::Identity => Ok(SupportMap::Identity),
            BridgeMapping::Affine { intercept, slope } => {
                Ok(SupportMap::Affine { intercept, slope })
            }
            _ => Err(mapping_uncertainty_invalid(
                "mapping",
                "supply a fixed mapping with certified extrema on the declared domain",
            )),
        }
    }

    /// Coefficients of a fixed map contributing one named input to a joint outcome.
    ///
    /// # Errors
    /// Checks both endpoint coordinates, assumptions and the joint-outcome value kind.
    pub fn joint_outcome_coefficients(&self) -> Result<(f64, f64), BridgeRefusal> {
        match self.fixed_map_for(ValueKind::JointOutcomeLaw)? {
            SupportMap::Identity => Ok((0.0, 1.0)),
            SupportMap::Affine { intercept, slope } => Ok((intercept, slope)),
            _ => Err(mapping_uncertainty_invalid("mapping", "declare a fixed affine contribution")),
        }
    }

    /// Transform a source identified set on the full, declared valid input interval.
    ///
    /// # Errors
    /// Returns a typed refusal for a mismatched coordinate, missing bridge premise,
    /// uncertain map, or source extending beyond the declared domain.
    pub fn identified_set_image(
        &self,
        source: &IdentifiedSetSource,
        domain: FixedBridgeDomain,
    ) -> Result<BridgedIdentifiedSet, BridgeRefusal> {
        let map = self.fixed_map_for(ValueKind::IdentifiedSet)?;
        self.source.quantity.require_same_coordinate(&source.quantity).map_err(|_| {
            quantity_mismatch(
                "source quantity",
                self.source.quantity.canonical_identity(),
                source.quantity.canonical_identity(),
            )
        })?;
        let bounds = ValueInterval { low: source.set.lower, high: source.set.upper };
        domain.covers(bounds)?;
        let image = support_image(map, bounds)?;
        let mut identity = CanonicalWriter::new("antecedent.scientific_bridge.identified_set.v1");
        identity.bytes(self.identity().as_bytes()).bytes(source.claim_identity.as_bytes());
        for value in [bounds.low, bounds.high, domain.0.low, domain.0.high] {
            identity.word(value.to_bits());
        }
        Ok(BridgedIdentifiedSet {
            quantity: self.target.quantity.clone(),
            set: IdentifiedSet { lower: image.low, upper: image.high },
            identity: SemanticDigest::from_bytes(identity.finish()),
        })
    }
}
