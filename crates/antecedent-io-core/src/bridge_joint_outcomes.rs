//! Row-wise vector maps of one observed or predictive joint outcome law.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::IoError;
use crate::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIntakeLimits, DistributionTrust,
};
use crate::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use antecedent_contract::bridge::{
    BridgeContract, BridgeRefusal, FixedBridgeDomain, SupportMap, ValueInterval,
    draws_outside_domain, draws_unusable, support_image,
};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use std::collections::BTreeSet;

/// A named contribution, with its units and input domain supplied by its bridge.
#[derive(Clone, Debug, PartialEq)]
pub struct JointOutcomeTerm {
    /// Maps one input onto the scale of its output.
    pub contract: BridgeContract,
    /// Valid values of the input coordinate.
    pub domain: FixedBridgeDomain,
}

/// Each output is the sum of its declared affine contributions to one target coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct JointOutcomeMap {
    /// Outputs in the requested order. Terms within an output are accumulated in declaration order.
    pub outputs: Vec<Vec<JointOutcomeTerm>>,
}

/// The mapped joint distribution and the identity of the source and mapping that produced it.
#[derive(Clone, Debug)]
pub struct JointOutcomeImage {
    /// Row weights, pairing identity and distribution meaning are inherited from the source.
    pub distribution: DistributionArtifact,
    /// Digest binding the complete source table, all term declarations and their domains.
    pub identity: SemanticDigest,
}

/// Failure to check the scientific declaration or admit its computation.
#[derive(Debug, thiserror::Error)]
pub enum JointOutcomeBridgeError {
    /// Coordinate, range, pairing or mapping refusal.
    #[error("joint outcome mapping: {0:?}")]
    Contract(Box<BridgeRefusal>),
    /// Source or result exceeds intake or runtime limits, or fails artifact validation.
    #[error(transparent)]
    Artifact(#[from] IoError),
}

impl From<BridgeRefusal> for JointOutcomeBridgeError {
    fn from(value: BridgeRefusal) -> Self {
        Self::Contract(Box::new(value))
    }
}

impl JointOutcomeMap {
    /// Transform complete joint rows. All outputs read the same input row.
    ///
    /// # Errors
    /// A missing coordinate, unpaired marginals, unsupported quantity, ambiguous output,
    /// out-of-domain row or resource stop refuses the entire derived law.
    pub fn apply(
        &self,
        source: &DistributionArtifact,
        ctx: &ExecutionContext,
    ) -> Result<JointOutcomeImage, JointOutcomeBridgeError> {
        let limits = DistributionIntakeLimits::default();
        if source.metadata().identity.alignment
            != crate::distribution_artifact::DrawAlignment::Joint
        {
            return Err(antecedent_contract::bridge::draws_not_aligned().into());
        }
        if !matches!(
            source.semantic(),
            DistributionMeaningWire::EmpiricalOutcome
                | DistributionMeaningWire::InterventionalPredictive
                | DistributionMeaningWire::PosteriorPredictive
        ) {
            return Err(draws_unusable(
                "source law",
                "supply observed or predictive joint outcome rows",
            )
            .into());
        }
        let n_outputs = self.outputs.len();
        let output_bytes = source
            .n_draws()
            .checked_mul(n_outputs)
            .and_then(|n| n.checked_mul(8))
            .ok_or(IoError::TooLarge)?;
        let term_count = self
            .outputs
            .iter()
            .try_fold(0usize, |n, terms| n.checked_add(terms.len()))
            .ok_or(IoError::TooLarge)?;
        if n_outputs == 0
            || n_outputs > limits.max_coordinates
            || output_bytes > limits.max_bytes
            || (term_count as u64).saturating_mul(source.n_draws() as u64)
                > antecedent_contract::bridge::MAX_PROPAGATION_WORK
        {
            return Err(IoError::TooLarge.into());
        }
        let workspace = (output_bytes as u64)
            .saturating_add((source.draws().len() as u64).saturating_mul(24))
            .saturating_add((term_count as u64).saturating_mul(64));
        let _lease = ctx.acquire_serial_numerical_admission(workspace).map_err(IoError::from)?;
        let mut identity = CanonicalWriter::new("antecedent.bridge.joint_outcome_image.v1");
        identity
            .bytes(&CanonicalWriter::digest_bytes(
                "antecedent.bridge.joint_outcome_source.v1",
                &source.to_bytes("joint-outcome-source")?,
            ))
            .word(n_outputs as u64);
        let mut coordinates = Vec::with_capacity(n_outputs);
        let mut columns = Vec::with_capacity(n_outputs);
        let source_quantities = source
            .quantities()
            .iter()
            .cloned()
            .map(antecedent_contract::ScientificQuantity::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                draws_unusable("source", "declare the full scientific coordinate for every input")
            })?;
        let mut seen_targets = BTreeSet::new();
        for terms in &self.outputs {
            let first = terms.first().ok_or_else(|| {
                draws_unusable("output", "bind at least one source coordinate per output")
            })?;
            let target = &first.contract.target.quantity;
            if !seen_targets.insert(target.canonical_identity()) {
                return Err(draws_unusable("output", "declare each output coordinate once").into());
            }
            coordinates.push(ScientificQuantityWire::from(target));
            identity.word(terms.len() as u64);
            columns.push(output_column(terms, target, source, &source_quantities, &mut identity)?);
        }
        let values = map_rows(source, &columns, output_bytes / 8, ctx)?;
        let mut metadata = source.metadata().clone();
        metadata.identity.quantities = coordinates;
        metadata.shape[1] = n_outputs;
        metadata.supported = None;
        metadata.legacy_bindings = None;
        metadata.legacy_posterior = None;
        metadata.calibration = DistributionCalibration::NotMeasured;
        metadata.trust = DistributionTrust::Unverified;
        Ok(JointOutcomeImage {
            distribution: DistributionArtifact::new(metadata, values)?,
            identity: SemanticDigest::from_bytes(identity.finish()),
        })
    }
}

fn output_column(
    terms: &[JointOutcomeTerm],
    target: &antecedent_contract::ScientificQuantity,
    source: &DistributionArtifact,
    source_quantities: &[antecedent_contract::ScientificQuantity],
    identity: &mut CanonicalWriter,
) -> Result<Vec<(usize, f64, f64, ValueInterval)>, JointOutcomeBridgeError> {
    let mut named = BTreeSet::new();
    let mut column = Vec::with_capacity(terms.len());
    for term in terms {
        let (offset, slope) = term.contract.joint_outcome_coefficients()?;
        if target.require_same_coordinate(&term.contract.target.quantity).is_err() {
            return Err(draws_unusable(
                "output",
                "all contributions must use the same output coordinate and units",
            )
            .into());
        }
        let index = source_quantities
            .iter()
            .position(|q| q.require_same_coordinate(&term.contract.source.quantity).is_ok())
            .ok_or_else(|| {
                draws_unusable(
                    "input",
                    "supply the named source coordinate with its declared units",
                )
            })?;
        if !named.insert(index)
            || source.metadata().supported.as_ref().is_some_and(|supported| !supported[index])
        {
            return Err(draws_unusable(
                "input",
                "use each supported input coordinate once per output",
            )
            .into());
        }
        let domain = support_image(SupportMap::Identity, term.domain.0)?;
        identity
            .bytes(term.contract.identity().as_bytes())
            .word(domain.low.to_bits())
            .word(domain.high.to_bits());
        column.push((index, offset, slope, domain));
    }
    Ok(column)
}

fn map_rows(
    source: &DistributionArtifact,
    columns: &[Vec<(usize, f64, f64, ValueInterval)>],
    capacity: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, JointOutcomeBridgeError> {
    let mut values = Vec::with_capacity(capacity);
    for row in source.draws().chunks_exact(source.shape()[1]) {
        if ctx.cancellation.is_cancelled() {
            return Err(IoError::StreamCancelled.into());
        }
        for terms in columns {
            let mut sum = 0.0;
            for &(index, offset, slope, domain) in terms {
                let x = row[index];
                if x < domain.low || x > domain.high {
                    return Err(draws_outside_domain(1, source.n_draws(), x).into());
                }
                sum += offset + slope * x;
            }
            if !sum.is_finite() {
                return Err(draws_unusable("output", "the mapped value must be finite").into());
            }
            values.push(sum);
        }
    }
    Ok(values)
}
