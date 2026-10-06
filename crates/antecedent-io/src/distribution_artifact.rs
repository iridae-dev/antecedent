//! Bounded, portable aligned-draw artifact for the frozen 2.3 F1 cell.
//!
//! The v1 layout has two named axes, `draw` then `quantity`, and f64 values
//! stored draw-major: draw `i` occupies `i * n_quantities..(i + 1) * n_quantities`.
//! A row is one joint realization only when alignment is explicitly `Joint`.
//! The caller must supply an independently retained expected identity on load;
//! an artifact's own checksum cannot establish its causal premises after resealing.

use antecedent_core::ScientificQuantity;
use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;
use crate::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// Maximum artifact bytes accepted before any decode allocation.
pub const MAX_DISTRIBUTION_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
/// Maximum number of draws.
pub const MAX_DISTRIBUTION_DRAWS: usize = 100_000;
/// Maximum number of scalar coordinates.
pub const MAX_DISTRIBUTION_COORDINATES: usize = 1_024;
const ARTIFACT_KIND: &str = "joint_distribution_v1";
const META_SECTION: &str = "distribution.meta";
const DRAW_SECTION: &str = "distribution.draws";

/// Whether equal draw indices are a paired joint realization.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrawAlignment {
    /// Each row is one joint realization across every coordinate.
    Joint,
    /// The columns are independently sampled marginals; row pairing is invalid.
    IndependentMarginals,
}

/// Claim carried by a finite draw set; this is not itself a calibration result.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistributionCalibration {
    /// Exact finite law supplied by a checked provider.
    Exact,
    /// Descriptive point summaries only.
    PointOnly,
    /// A separately cited, coordinate-specific calibration record exists.
    Measured,
    /// Inferential calibration is absent.
    Unmeasured,
}

/// Source trust label preserved without upgrading it during load.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistributionTrust {
    /// Native route separately licensed at its exact coordinate.
    NativeLicensed,
    /// External provider attestation only.
    ExternalAttested,
    /// One external object verified for an exact request fingerprint.
    VerifiedExtension,
    /// No provider verification claim.
    Unverified,
}

/// Source and execution lineage bound to every distribution object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DistributionProvenance {
    /// Stable source or study ID.
    pub source_id: String,
    /// Stable provider object ID.
    pub provider_id: String,
    /// RNG algorithm, seed and stream ID, or `deterministic_exact`.
    pub rng_id: String,
    /// Data snapshot or exact-law ID.
    pub snapshot_id: String,
    /// Independently checked causal-contract ID.
    pub causal_contract_id: String,
}

/// Scientific identity a producer and independent consumer agree on.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionIdentity {
    /// Random-quantity meaning of each draw.
    pub semantic: DistributionMeaningWire,
    /// Quantities in exact artifact column order.
    pub quantities: Vec<ScientificQuantityWire>,
    /// Whether rows are genuinely paired joint realizations.
    pub alignment: DrawAlignment,
    /// Stable source/study ID.
    pub source_id: String,
    /// Stable provider object ID.
    pub provider_id: String,
    /// RNG algorithm, seed and stream identity, or `deterministic_exact`.
    pub rng_id: String,
    /// Data snapshot or exact-law identity.
    pub snapshot_id: String,
    /// Independently checked causal-contract identity.
    pub causal_contract_id: String,
}

impl DistributionIdentity {
    /// Build a portable identity from checked core quantity coordinates.
    ///
    /// # Errors
    /// Empty/duplicate coordinates or missing source identities are refused.
    pub fn new(
        semantic: DistributionMeaningWire,
        quantities: &[ScientificQuantity],
        alignment: DrawAlignment,
        provenance: DistributionProvenance,
    ) -> Result<Self, IoError> {
        for quantity in quantities {
            quantity.validate().map_err(|_| IoError::Convert("invalid quantity".into()))?;
        }
        let value = Self {
            semantic,
            quantities: quantities.iter().map(ScientificQuantityWire::from).collect(),
            alignment,
            source_id: provenance.source_id,
            provider_id: provenance.provider_id,
            rng_id: provenance.rng_id,
            snapshot_id: provenance.snapshot_id,
            causal_contract_id: provenance.causal_contract_id,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), IoError> {
        if self.quantities.is_empty() || self.quantities.len() > MAX_DISTRIBUTION_COORDINATES {
            return Err(IoError::TooLarge);
        }
        for quantity in &self.quantities {
            ScientificQuantity::try_from(quantity.clone())
                .map_err(|error| IoError::Convert(error.into()))?;
        }
        let mut ids = std::collections::HashSet::new();
        if self.quantities.iter().any(|quantity| {
            !ids.insert((
                &quantity.variable_id,
                &quantity.population_id,
                &quantity.regime_id,
                quantity.horizon,
                &quantity.functional_id,
                &quantity.conditioning,
                &quantity.transform_id,
            ))
        }) {
            return Err(IoError::Convert("duplicate quantity coordinate".into()));
        }
        if [
            &self.source_id,
            &self.provider_id,
            &self.rng_id,
            &self.snapshot_id,
            &self.causal_contract_id,
        ]
        .into_iter()
        .any(|identity| identity.trim().is_empty())
        {
            return Err(IoError::Convert("missing distribution provenance identity".into()));
        }
        Ok(())
    }
}

/// Stored metadata; the numerical section is separate f64 LE draw-major bytes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionMetadata {
    /// Metadata format version.
    pub version: u16,
    /// Complete scientific and provider identity.
    pub identity: DistributionIdentity,
    /// Named axes in physical order.
    pub axes: [String; 2],
    /// Shape in physical order: `[n_draws, n_quantities]`.
    pub shape: [usize; 2],
    /// Optional nonnegative per-draw weights; normalized on evaluation.
    pub weights: Option<Vec<f64>>,
    /// Per-coordinate support; absent means every coordinate is supported.
    pub supported: Option<Vec<bool>>,
    /// Declared inferential calibration status.
    pub calibration: DistributionCalibration,
    /// Provider trust status; loading never upgrades it.
    pub trust: DistributionTrust,
}

/// A validated finite draw set plus its bound metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct DistributionArtifact {
    metadata: DistributionMetadata,
    draws: Vec<f64>,
}

impl DistributionArtifact {
    /// Validate and own one finite distribution artifact.
    ///
    /// # Errors
    /// Shape, values, weights, masks, identities and payload bounds are checked.
    pub fn new(metadata: DistributionMetadata, draws: Vec<f64>) -> Result<Self, IoError> {
        validate(&metadata, &draws)?;
        Ok(Self { metadata, draws })
    }

    /// Bound metadata, including semantic tag, axes and provenance.
    #[must_use]
    pub fn metadata(&self) -> &DistributionMetadata {
        &self.metadata
    }

    /// Read-only draw-major numerical buffer.
    #[must_use]
    pub fn draws(&self) -> &[f64] {
        &self.draws
    }

    /// Serialize through the existing checksummed sectioned container.
    ///
    /// # Errors
    /// A payload larger than the F1 bound or invalid metadata is refused.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        validate(&self.metadata, &self.draws)?;
        if artifact_id.trim().is_empty() {
            return Err(IoError::Convert("missing artifact id".into()));
        }
        let meta_bytes = to_cbor(&self.metadata)?;
        let mut draw_bytes = Vec::with_capacity(self.draws.len() * 8);
        for value in &self.draws {
            draw_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let encoded = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: crate::migrate::STABLE_FORMAT,
                minimum_reader_version: crate::migrate::STABLE_FORMAT,
                artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
                library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))?,
                artifact_id: artifact_id.into(),
                sections: vec![
                    section_descriptor(META_SECTION, "application/cbor", &meta_bytes),
                    section_descriptor(DRAW_SECTION, "application/octet-stream", &draw_bytes),
                ],
                provenance: ProvenanceWire { note: "aligned_joint_distribution".into() },
            },
            sections: vec![
                SectionBytes::new(META_SECTION, meta_bytes),
                SectionBytes::new(DRAW_SECTION, draw_bytes),
            ],
        };
        let mut bytes = Vec::new();
        encoded.write_to(&mut bytes)?;
        if bytes.len() > MAX_DISTRIBUTION_ARTIFACT_BYTES {
            return Err(IoError::TooLarge);
        }
        Ok(bytes)
    }

    /// Load and check against an independently retained scientific identity.
    ///
    /// The expected identity must come from the consumer's own causal contract,
    /// not from the artifact under inspection. A resealed change in population,
    /// regime, axes, quantity order, provider or snapshot then fails.
    ///
    /// # Errors
    /// Oversized, truncated, corrupt, incompatible or semantically changed
    /// artifacts are refused before any numerical operation.
    pub fn from_bytes(bytes: &[u8], expected: &DistributionIdentity) -> Result<Self, IoError> {
        if bytes.len() > MAX_DISTRIBUTION_ARTIFACT_BYTES {
            return Err(IoError::TooLarge);
        }
        expected.validate()?;
        let mut reader = ArtifactReader::open_seek(std::io::Cursor::new(bytes))?;
        let manifest = reader.manifest();
        if manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into())
            || manifest.sections.len() != 2
            || manifest.sections[0].id != META_SECTION
            || manifest.sections[1].id != DRAW_SECTION
        {
            return Err(IoError::Convert("unsupported distribution artifact layout".into()));
        }
        let declared = manifest.sections.iter().try_fold(0u64, |total, section| {
            total.checked_add(section.uncompressed_size).ok_or(IoError::TooLarge)
        })?;
        if declared > MAX_DISTRIBUTION_ARTIFACT_BYTES as u64 {
            return Err(IoError::TooLarge);
        }
        let meta = reader.load_section(META_SECTION)?;
        let metadata: DistributionMetadata = from_cbor(meta.as_bytes())?;
        if metadata.identity != *expected {
            return Err(IoError::Refused {
                code: antecedent_core::reason_code!("quantity_semantics_mismatch"),
                message: "aligned_joint_draws.identity_expected: distribution identity differs from expected causal contract".into(),
            });
        }
        validate_metadata(&metadata)?;
        let draw = reader.load_section(DRAW_SECTION)?;
        if draw.as_bytes().len() != metadata.shape[0] * metadata.shape[1] * 8 {
            return Err(IoError::Convert("distribution draw payload length mismatch".into()));
        }
        let draws = draw
            .as_bytes()
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
            .collect();
        Self::new(metadata, draws)
    }

    /// Finite weighted mean of one supported coordinate.
    ///
    /// # Errors
    /// Out-of-range or masked coordinates are refused.
    pub fn mean(&self, coordinate: usize) -> Result<f64, IoError> {
        self.require_coordinate(coordinate)?;
        let width = self.metadata.shape[1];
        let value: f64 = self
            .weights()
            .enumerate()
            .map(|(draw, weight)| weight * self.draws[draw * width + coordinate])
            .sum();
        if !value.is_finite() {
            return Err(IoError::Convert("distribution mean overflow".into()));
        }
        Ok(value)
    }

    /// Finite weighted population covariance of aligned joint coordinates.
    ///
    /// # Errors
    /// Independent marginals and unsupported coordinates are refused.
    pub fn covariance(&self, left: usize, right: usize) -> Result<f64, IoError> {
        self.require_joint(left, right)?;
        let (left_mean, right_mean) = (self.mean(left)?, self.mean(right)?);
        let width = self.metadata.shape[1];
        let value: f64 = self
            .weights()
            .enumerate()
            .map(|(draw, weight)| {
                weight
                    * (self.draws[draw * width + left] - left_mean)
                    * (self.draws[draw * width + right] - right_mean)
            })
            .sum();
        if !value.is_finite() {
            return Err(IoError::Convert("distribution covariance overflow".into()));
        }
        Ok(value)
    }

    /// Expected value of a two-coordinate nonlinear function on aligned draws.
    ///
    /// # Errors
    /// Independent marginals, masked coordinates and non-finite function
    /// outputs are refused.
    pub fn joint_expectation(
        &self,
        left: usize,
        right: usize,
        function: impl Fn(f64, f64) -> f64,
    ) -> Result<f64, IoError> {
        self.require_joint(left, right)?;
        let width = self.metadata.shape[1];
        let mut value = 0.0;
        for (draw, weight) in self.weights().enumerate() {
            let output =
                function(self.draws[draw * width + left], self.draws[draw * width + right]);
            if !output.is_finite() {
                return Err(IoError::Convert(
                    "nonlinear function returned a non-finite value".into(),
                ));
            }
            value += weight * output;
        }
        if !value.is_finite() {
            return Err(IoError::Convert("nonlinear expectation overflow".into()));
        }
        Ok(value)
    }

    /// Probability of an interventional outcome exceeding a threshold.
    ///
    /// An observational conditional distribution requires a separately checked
    /// causal-equivalence proof before it can be bound to an intervention.
    ///
    /// # Errors
    /// Wrong distribution meaning, regime or masked coordinate is refused.
    pub fn interventional_probability_above(
        &self,
        coordinate: usize,
        threshold: f64,
    ) -> Result<f64, IoError> {
        self.require_coordinate(coordinate)?;
        if !threshold.is_finite() {
            return Err(IoError::Convert("non-finite outcome threshold".into()));
        }
        if self.metadata.identity.semantic != DistributionMeaningWire::InterventionalPredictive {
            return Err(IoError::Refused {
                code: antecedent_core::reason_code!("distribution_meaning_mismatch"),
                message: "distribution_meaning.incompatible_operation: interventional outcome threshold requires interventional predictive draws".into(),
            });
        }
        if self.metadata.identity.quantities[coordinate].regime_id == "observational"
            || self.metadata.identity.quantities[coordinate].role != "outcome"
            || self.metadata.identity.quantities[coordinate].functional_id != "outcome"
        {
            return Err(IoError::Refused {
                code: antecedent_core::reason_code!("quantity_semantics_mismatch"),
                message: "aligned_joint_draws.outcome_coordinate: outcome probability requires an interventional outcome coordinate".into(),
            });
        }
        let width = self.metadata.shape[1];
        Ok(self
            .weights()
            .enumerate()
            .filter(|(draw, _)| self.draws[draw * width + coordinate] > threshold)
            .map(|(_, weight)| weight)
            .sum())
    }

    fn require_coordinate(&self, coordinate: usize) -> Result<(), IoError> {
        if coordinate >= self.metadata.shape[1] {
            return Err(IoError::Convert("distribution coordinate out of range".into()));
        }
        if self.metadata.supported.as_ref().is_some_and(|mask| !mask[coordinate]) {
            return Err(IoError::Convert("distribution coordinate unsupported".into()));
        }
        Ok(())
    }

    fn require_joint(&self, left: usize, right: usize) -> Result<(), IoError> {
        self.require_coordinate(left)?;
        self.require_coordinate(right)?;
        if self.metadata.identity.alignment != DrawAlignment::Joint {
            return Err(IoError::Refused {
                code: antecedent_core::reason_code!("joint_law_required"),
                message: "aligned_joint_draws.marginals_not_joint: independent marginals do not define covariance or nonlinear joint expectations".into(),
            });
        }
        Ok(())
    }

    fn weights(&self) -> impl Iterator<Item = f64> + '_ {
        let sum = self
            .metadata
            .weights
            .as_ref()
            .map_or(self.metadata.shape[0] as f64, |weights| weights.iter().sum());
        (0..self.metadata.shape[0]).map(move |index| {
            self.metadata.weights.as_ref().map_or(1.0, |weights| weights[index]) / sum
        })
    }
}

fn validate_metadata(meta: &DistributionMetadata) -> Result<(), IoError> {
    if meta.version != 1 {
        return Err(IoError::UnsupportedVersion { version: u32::from(meta.version) });
    }
    meta.identity.validate()?;
    if meta.axes != ["draw", "quantity"]
        || meta.shape[0] == 0
        || meta.shape[0] > MAX_DISTRIBUTION_DRAWS
        || meta.shape[1] != meta.identity.quantities.len()
    {
        return Err(IoError::Convert("invalid distribution axes or shape".into()));
    }
    let cells = meta.shape[0].checked_mul(meta.shape[1]).ok_or(IoError::TooLarge)?;
    if cells.checked_mul(8).ok_or(IoError::TooLarge)? > MAX_DISTRIBUTION_ARTIFACT_BYTES {
        return Err(IoError::TooLarge);
    }
    if let Some(weights) = &meta.weights {
        let sum: f64 = weights.iter().sum();
        if weights.len() != meta.shape[0]
            || weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
            || !sum.is_finite()
            || sum <= 0.0
        {
            return Err(IoError::Convert("invalid distribution weights".into()));
        }
    }
    if meta.supported.as_ref().is_some_and(|mask| mask.len() != meta.shape[1]) {
        return Err(IoError::Convert("distribution support mask length mismatch".into()));
    }
    Ok(())
}

fn validate(meta: &DistributionMetadata, draws: &[f64]) -> Result<(), IoError> {
    validate_metadata(meta)?;
    if draws.len() != meta.shape[0] * meta.shape[1] || draws.iter().any(|value| !value.is_finite())
    {
        return Err(IoError::Convert("invalid distribution draw values or length".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use antecedent_core::{QuantityRole, ScientificQuantity};

    use super::*;

    fn quantity(id: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: format!("schema:{id}"),
            variable_name: id.into(),
            role: QuantityRole::Outcome,
            units: "dimensionless".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    fn fixture(alignment: DrawAlignment) -> DistributionArtifact {
        let identity = DistributionIdentity::new(
            DistributionMeaningWire::InterventionalPredictive,
            &[quantity("x"), quantity("y")],
            alignment,
            DistributionProvenance {
                source_id: "study".into(),
                provider_id: "provider".into(),
                rng_id: "deterministic_exact".into(),
                snapshot_id: "snapshot".into(),
                causal_contract_id: "checked-contract".into(),
            },
        )
        .unwrap();
        DistributionArtifact::new(
            DistributionMetadata {
                version: 1,
                identity,
                axes: ["draw".into(), "quantity".into()],
                shape: [2, 2],
                weights: None,
                supported: None,
                calibration: DistributionCalibration::Exact,
                trust: DistributionTrust::Unverified,
            },
            vec![0.0, 0.0, 1.0, 2.0],
        )
        .unwrap()
    }

    #[test]
    fn enumerated_joint_values_round_trip_with_independent_expected_identity() {
        let original = fixture(DrawAlignment::Joint);
        let bytes = original.to_bytes("joint-fixture").unwrap();
        let loaded = DistributionArtifact::from_bytes(&bytes, &original.metadata.identity).unwrap();
        assert_eq!(loaded.metadata(), original.metadata());
        assert_eq!(loaded.draws(), original.draws());
        assert!((loaded.mean(0).unwrap() - 0.5).abs() < 1e-12);
        assert!((loaded.mean(1).unwrap() - 1.0).abs() < 1e-12);
        assert!((loaded.covariance(0, 1).unwrap() - 0.5).abs() < 1e-12);
        assert!((loaded.joint_expectation(0, 1, |x, y| x * y).unwrap() - 1.0).abs() < 1e-12);
        assert!((loaded.interventional_probability_above(1, 1.0).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn marginal_draws_and_wrong_semantics_refuse_before_value_evaluation() {
        let marginal = fixture(DrawAlignment::IndependentMarginals);
        assert_eq!(
            marginal.covariance(0, 1).unwrap_err().reason_code(),
            Some("joint_law_required")
        );
        assert_eq!(
            marginal.joint_expectation(0, 1, |x, y| x * y).unwrap_err().reason_code(),
            Some("joint_law_required")
        );
        let mut meta = fixture(DrawAlignment::Joint).metadata;
        meta.identity.semantic = DistributionMeaningWire::CausalFunctionalPosterior;
        let posterior = DistributionArtifact::new(meta, vec![0.0, 0.0, 1.0, 2.0]).unwrap();
        assert_eq!(
            posterior.interventional_probability_above(1, 1.0).unwrap_err().reason_code(),
            Some("distribution_meaning_mismatch")
        );
    }

    #[test]
    fn expected_contract_rejects_resealed_identity_and_corruption() {
        let original = fixture(DrawAlignment::Joint);
        let bytes = original.to_bytes("joint-fixture").unwrap();
        let mut changed = original.metadata.clone();
        changed.identity.snapshot_id = "other-snapshot".into();
        let resealed = DistributionArtifact::new(changed, original.draws.clone())
            .unwrap()
            .to_bytes("joint-fixture")
            .unwrap();
        assert_eq!(
            DistributionArtifact::from_bytes(&resealed, &original.metadata.identity)
                .unwrap_err()
                .reason_code(),
            Some("quantity_semantics_mismatch")
        );
        let mut corrupt = bytes.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(DistributionArtifact::from_bytes(&corrupt, &original.metadata.identity).is_err());
        assert!(
            DistributionArtifact::from_bytes(
                &bytes[..bytes.len() - 1],
                &original.metadata.identity
            )
            .is_err()
        );
        assert!(
            DistributionArtifact::from_bytes(
                &vec![0u8; MAX_DISTRIBUTION_ARTIFACT_BYTES + 1],
                &original.metadata.identity
            )
            .is_err()
        );
    }

    #[test]
    fn every_quantity_dimension_is_checked_against_the_consumer_contract() {
        let original = fixture(DrawAlignment::Joint);
        for changed_dimension in 0..8 {
            let mut changed = original.metadata.clone();
            let quantity = &mut changed.identity.quantities[0];
            match changed_dimension {
                0 => quantity.variable_id = "schema:other".into(),
                1 => quantity.units = "kg".into(),
                2 => quantity.population_id = "source".into(),
                3 => quantity.regime_id = "observational".into(),
                4 => quantity.horizon = 1,
                5 => quantity.functional_id = "mean".into(),
                6 => quantity.conditioning.push(crate::quantity_wire::QuantityConditionWire {
                    variable_id: "schema:z".into(),
                    value_id: "high".into(),
                }),
                7 => quantity.transform_id = "log".into(),
                _ => unreachable!(),
            }
            let resealed = DistributionArtifact::new(changed, original.draws.clone())
                .unwrap()
                .to_bytes("changed-coordinate")
                .unwrap();
            assert!(
                DistributionArtifact::from_bytes(&resealed, &original.metadata.identity).is_err(),
                "changed dimension {changed_dimension} was accepted"
            );
        }
        let mut newer = original.metadata.clone();
        newer.version = 2;
        assert!(DistributionArtifact::new(newer, original.draws.clone()).is_err());
    }

    #[test]
    fn weights_mask_and_reordered_draws_are_preserved() {
        let original = fixture(DrawAlignment::Joint);
        let mut meta = original.metadata.clone();
        meta.weights = Some(vec![1.0, 3.0]);
        meta.supported = Some(vec![true, false]);
        let weighted = DistributionArtifact::new(meta, vec![1.0, 2.0, 0.0, 0.0]).unwrap();
        let bytes = weighted.to_bytes("weighted").unwrap();
        let loaded = DistributionArtifact::from_bytes(&bytes, &weighted.metadata.identity).unwrap();
        assert!((loaded.mean(0).unwrap() - 0.25).abs() < 1e-12);
        assert!(loaded.mean(1).is_err());
        assert!(loaded.covariance(0, 1).is_err());

        let reversed =
            DistributionArtifact::new(original.metadata.clone(), vec![0.0, 2.0, 1.0, 0.0]).unwrap();
        assert!((reversed.covariance(0, 1).unwrap() + 0.5).abs() < 1e-12);
        assert!(reversed.joint_expectation(0, 1, |x, y| x * y).unwrap().abs() < 1e-12);
    }
}
