//! Portable staleness baseline: the summary of a plan's data that new data is compared with.
//!
//! The baseline is written once by the producer of a plan (or of an exported result) and read
//! by a consumer that holds only these bytes and new data. It carries no rows except a bounded
//! uniform subsample of the numeric covariates, so a consumer can test for drift without the
//! producer and without the original data. Everything a staleness diagnostic reads is here:
//! the schema, the plan's roles, per-column moments, quantile bins, level counts, missingness
//! counts, the joint moments and subsample of the covariates, the propensity overlap summary,
//! the recorded assumption checks, and (for a Bayesian plan) the sufficient statistics of the
//! outcome regression.
//!
//! The seal is a digest of every other field. [`StalenessBaselineWire::consume`] refuses bytes
//! whose seal, marker, version, feature or kind does not verify, so a tampered or truncated
//! baseline is never read as a valid one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::IoError;
use serde::{Deserialize, Serialize};

/// Byte marker ahead of the encoded baseline.
pub const STALENESS_BASELINE_MAGIC: &[u8] = b"ANTSTALEBASE1";
/// Largest encoded baseline accepted before decoding.
pub const STALENESS_BASELINE_MAX_BYTES: usize = 64_000_000;
/// Wire format version of a baseline whose missingness patterns fit one 64-bit word.
pub const STALENESS_BASELINE_VERSION: u32 = 1;
/// Wire format version of a baseline that stores wide missingness patterns (more than 64 plan
/// columns). A reader of version 1 refuses it; the narrow encoding is unchanged, so every
/// baseline that fits one word keeps its bytes and its seal.
pub const STALENESS_BASELINE_WIDE_VERSION: u32 = 2;
/// Required feature of a staleness baseline.
pub const STALENESS_BASELINE_FEATURE: &str = "staleness_baseline_v1";
/// Most plan columns whose missingness patterns are stored as one 64-bit word.
pub const NARROW_PATTERN_COLUMNS: usize = 64;

/// Numeric summary of one column over its observed rows.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NumericProfileWire {
    /// Mean of the observed values.
    pub mean: f64,
    /// Centered sum of squares of the observed values.
    pub m2: f64,
    /// Smallest observed value.
    pub min: f64,
    /// Largest observed value.
    pub max: f64,
    /// Interior quantile bin edges, ascending and distinct. `edges.len() + 1` bins.
    pub edges: Vec<f64>,
    /// Observed count per bin (a value equal to an edge falls in the upper bin).
    pub bin_counts: Vec<u64>,
}

/// One categorical level and its observed count.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LevelCountWire {
    /// Level label.
    pub label: String,
    /// Observed rows at this level.
    pub count: u64,
}

/// Profile of one schema column.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StalenessColumnWire {
    /// Column name.
    pub name: String,
    /// Declared value type label (`continuous`, `binary`, `categorical`, ...).
    pub value_type: String,
    /// Storage kind label (`float64`, `int64`, `boolean`, `categorical`, ...).
    pub storage: String,
    /// Role in the plan (`treatment`, `outcome`, `adjustment`, `period` or `other`).
    pub role: String,
    /// Observed rows.
    pub n_valid: u64,
    /// Missing rows.
    pub n_missing: u64,
    /// Numeric summary, for a column read as numbers.
    pub numeric: Option<NumericProfileWire>,
    /// Level counts, for a categorical column.
    pub levels: Option<Vec<LevelCountWire>>,
}

/// Joint moments and a uniform subsample of the numeric covariates (complete rows only).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct JointProfileWire {
    /// Covariates, in order.
    pub columns: Vec<String>,
    /// Complete rows summarized.
    pub n: u64,
    /// Mean vector.
    pub mean: Vec<f64>,
    /// Row-major centered scatter matrix.
    pub m2: Vec<f64>,
    /// Row-major subsample, `subsample_rows × columns.len()`.
    pub subsample: Vec<f64>,
    /// Rows in the subsample.
    pub subsample_rows: u64,
}

/// Counts of the missingness patterns over the plan's columns.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MissingPatternWire {
    /// Columns the pattern bits refer to (bit `i` set means column `i` is missing).
    pub columns: Vec<String>,
    /// `(pattern, count)` in ascending pattern order, for at most [`NARROW_PATTERN_COLUMNS`]
    /// columns (empty otherwise).
    pub patterns: Vec<(u64, u64)>,
    /// `(pattern words, count)` in ascending pattern order, for more than
    /// [`NARROW_PATTERN_COLUMNS`] columns: word `w` holds the bits of columns `64 w ..`, least
    /// significant bit first, and trailing zero words are omitted. Absent (and not encoded) for a
    /// narrow plan.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub wide_patterns: Vec<(Vec<u64>, u64)>,
}

impl MissingPatternWire {
    /// Every pattern as trimmed words with its count, narrow or wide, in ascending order.
    #[must_use]
    pub fn word_patterns(&self) -> Vec<(Vec<u64>, u64)> {
        let mut out: Vec<(Vec<u64>, u64)> = self
            .patterns
            .iter()
            .map(|(p, c)| (if *p == 0 { Vec::new() } else { vec![*p] }, *c))
            .collect();
        out.extend(self.wide_patterns.iter().map(|(words, c)| {
            let mut words = words.clone();
            while words.last() == Some(&0) {
                words.pop();
            }
            (words, *c)
        }));
        out
    }
}

/// Propensity overlap of the treatment on the stored adjustment set.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OverlapSummaryWire {
    /// Complete rows the propensity was fitted on.
    pub n: u64,
    /// Fraction of rows whose propensity lies outside `[epsilon, 1 - epsilon]`.
    pub extreme_fraction: f64,
    /// Effective sample size of the inverse-propensity weights divided by `n`.
    pub ess_ratio: f64,
    /// Smallest fitted propensity.
    pub min_propensity: f64,
    /// Largest fitted propensity.
    pub max_propensity: f64,
    /// Largest inverse-propensity weight.
    pub max_weight: f64,
    /// Largest absolute weighted standardized mean difference of the adjustment set.
    pub max_weighted_smd: f64,
    /// Positivity margin the extreme fraction was counted with.
    pub epsilon: f64,
}

/// One assumption check recorded on the plan's data, with the threshold it was judged by.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RecordedCheckWire {
    /// Check name.
    pub name: String,
    /// What a failure invalidates: `identification`, `design` or `target`.
    pub level: String,
    /// Statistic on the plan's data.
    pub statistic: f64,
    /// Threshold the statistic was judged by.
    pub threshold: f64,
    /// `at_most` (pass when the statistic is at most the threshold) or `at_least`.
    pub direction: String,
    /// Whether the check passed on the plan's data.
    pub passed: bool,
}

/// Sufficient statistics of the outcome regression of a Bayesian plan.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OutcomeModelWire {
    /// Design columns after the intercept (treatment, then the adjustment set).
    pub design: Vec<String>,
    /// Outcome column.
    pub outcome: String,
    /// Complete rows summarized.
    pub n: u64,
    /// Row-major `XᵀX` with the intercept first.
    pub xtx: Vec<f64>,
    /// `Xᵀy`.
    pub xty: Vec<f64>,
    /// `yᵀy`.
    pub yty: f64,
}

/// The sealed baseline of one plan's data snapshot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StalenessBaselineWire {
    /// Format version.
    pub version: u32,
    /// Required features.
    pub required_features: Vec<String>,
    /// Artifact kind.
    pub kind: String,
    /// Identity of the plan the baseline describes.
    pub plan_identity: String,
    /// Content identity of the data snapshot the baseline summarizes.
    pub snapshot: String,
    /// Identity of the policy the baseline was built under (bins, subsample, margins).
    pub policy: String,
    /// Treatment column, when the plan has one.
    pub treatment: Option<String>,
    /// Outcome column, when the plan has one.
    pub outcome: Option<String>,
    /// Stored adjustment set.
    pub adjustment: Vec<String>,
    /// Whether the plan is Bayesian.
    pub bayesian: bool,
    /// Column whose values define observation periods, when declared.
    pub period_column: Option<String>,
    /// Rows of the snapshot.
    pub rows: u64,
    /// Rows complete on the plan's columns.
    pub complete_rows: u64,
    /// Treatment arms (levels of a treatment with few distinct values) and their counts over
    /// the complete rows; empty for a continuous treatment or a plan without one.
    pub arms: Vec<LevelCountWire>,
    /// Column profiles in schema order.
    pub columns: Vec<StalenessColumnWire>,
    /// Joint covariate moments and subsample.
    pub joint: Option<JointProfileWire>,
    /// Missingness patterns over the plan's columns.
    pub missing: Option<MissingPatternWire>,
    /// Propensity overlap summary.
    pub overlap: Option<OverlapSummaryWire>,
    /// Recorded assumption checks.
    pub checks: Vec<RecordedCheckWire>,
    /// Outcome regression statistics of a Bayesian plan.
    pub outcome_model: Option<OutcomeModelWire>,
    /// Digest of every other field.
    pub seal: String,
}

/// Artifact kind of a staleness baseline.
pub const STALENESS_BASELINE_KIND: &str = crate::artifact_kinds::STALENESS_BASELINE;

impl StalenessBaselineWire {
    /// The seal this baseline should carry: the digest of every field but the seal.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn expected_seal(&self) -> Result<String, IoError> {
        let mut unsealed = self.clone();
        unsealed.seal.clear();
        let digest = crate::identity::payload_digest_of(STALENESS_BASELINE_KIND, &unsealed)?;
        Ok(antecedent_core::SemanticDigest::from_bytes(digest).to_hex())
    }

    /// Stamp version, feature, kind and seal.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn sealed(mut self) -> Result<Self, IoError> {
        self.version = if self.missing.as_ref().is_some_and(|m| !m.wide_patterns.is_empty()) {
            STALENESS_BASELINE_WIDE_VERSION
        } else {
            STALENESS_BASELINE_VERSION
        };
        self.required_features = vec![STALENESS_BASELINE_FEATURE.into()];
        self.kind = STALENESS_BASELINE_KIND.into();
        self.seal = self.expected_seal()?;
        Ok(self)
    }

    /// Check version, feature, kind and seal.
    ///
    /// # Errors
    /// The first field that does not verify.
    pub fn verify(&self) -> Result<(), IoError> {
        let wide = self.missing.as_ref().is_some_and(|m| !m.wide_patterns.is_empty());
        let expected =
            if wide { STALENESS_BASELINE_WIDE_VERSION } else { STALENESS_BASELINE_VERSION };
        if self.version != expected {
            return Err(IoError::Convert(format!(
                "staleness baseline version {} is not {expected}",
                self.version
            )));
        }
        if self.required_features != [STALENESS_BASELINE_FEATURE] {
            return Err(IoError::Convert("staleness baseline requires an unknown feature".into()));
        }
        if self.kind != STALENESS_BASELINE_KIND {
            return Err(IoError::Convert(format!(
                "artifact kind {} is not a staleness baseline",
                self.kind
            )));
        }
        if self.seal != self.expected_seal()? {
            return Err(IoError::Convert("staleness baseline seal does not verify".into()));
        }
        Ok(())
    }

    /// Encode a sealed baseline.
    ///
    /// # Errors
    /// The baseline does not verify, encoding fails, or the bytes exceed the limit.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        self.verify()?;
        let mut bytes = STALENESS_BASELINE_MAGIC.to_vec();
        ciborium::into_writer(self, &mut bytes).map_err(IoError::cbor_encode)?;
        if bytes.len() > STALENESS_BASELINE_MAX_BYTES {
            return Err(IoError::Convert("staleness baseline exceeds the byte limit".into()));
        }
        Ok(bytes)
    }

    /// Decode and verify a baseline.
    ///
    /// # Errors
    /// Oversize input, a missing marker, a corrupt body, or a field that does not verify.
    pub fn consume(bytes: &[u8]) -> Result<Self, IoError> {
        if bytes.len() > STALENESS_BASELINE_MAX_BYTES {
            return Err(IoError::Convert("staleness baseline exceeds the byte limit".into()));
        }
        let body = bytes
            .strip_prefix(STALENESS_BASELINE_MAGIC)
            .ok_or_else(|| IoError::Convert("bytes are not a staleness baseline".into()))?;
        let mut reader = std::io::Cursor::new(body);
        let wire: Self = ciborium::from_reader(&mut reader).map_err(IoError::cbor_decode)?;
        if reader.position() != body.len() as u64 {
            return Err(IoError::Convert("staleness baseline has trailing bytes".into()));
        }
        wire.verify()?;
        Ok(wire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal() -> StalenessBaselineWire {
        StalenessBaselineWire {
            version: 0,
            required_features: Vec::new(),
            kind: String::new(),
            plan_identity: "plan".into(),
            snapshot: "snap".into(),
            policy: "policy".into(),
            treatment: Some("t".into()),
            outcome: Some("y".into()),
            adjustment: vec!["z".into()],
            bayesian: false,
            period_column: None,
            rows: 3,
            complete_rows: 3,
            arms: Vec::new(),
            columns: vec![StalenessColumnWire {
                name: "z".into(),
                value_type: "continuous".into(),
                storage: "float64".into(),
                role: "adjustment".into(),
                n_valid: 3,
                n_missing: 0,
                numeric: Some(NumericProfileWire {
                    mean: 1.0,
                    m2: 2.0,
                    min: 0.0,
                    max: 2.0,
                    edges: vec![1.0],
                    bin_counts: vec![1, 2],
                }),
                levels: None,
            }],
            joint: None,
            missing: None,
            overlap: None,
            checks: Vec::new(),
            outcome_model: None,
            seal: String::new(),
        }
    }

    #[test]
    fn narrow_patterns_keep_version_one_and_wide_patterns_need_version_two() {
        let mut narrow = minimal();
        narrow.missing = Some(MissingPatternWire {
            columns: vec!["z".into()],
            patterns: vec![(0, 2), (1, 1)],
            wide_patterns: Vec::new(),
        });
        let narrow = narrow.sealed().unwrap();
        assert_eq!(narrow.version, STALENESS_BASELINE_VERSION);
        assert_eq!(StalenessBaselineWire::consume(&narrow.export().unwrap()).unwrap(), narrow);

        let mut wide = minimal();
        wide.missing = Some(MissingPatternWire {
            columns: (0..70).map(|k| format!("c{k}")).collect(),
            patterns: Vec::new(),
            wide_patterns: vec![(Vec::new(), 2), (vec![0, 1 << 5], 1)],
        });
        let wide = wide.sealed().unwrap();
        assert_eq!(wide.version, STALENESS_BASELINE_WIDE_VERSION);
        let back = StalenessBaselineWire::consume(&wide.export().unwrap()).unwrap();
        assert_eq!(back, wide);
        let patterns = back.missing.as_ref().unwrap().word_patterns();
        assert_eq!(patterns, vec![(Vec::new(), 2), (vec![0, 1 << 5], 1)]);
        // A wide baseline stamped with the narrow version does not verify.
        let mut wrong = wide.clone();
        wrong.version = STALENESS_BASELINE_VERSION;
        assert!(wrong.verify().is_err());
    }

    #[test]
    fn a_sealed_baseline_round_trips_and_a_tampered_one_is_refused() {
        let sealed = minimal().sealed().expect("seal");
        let bytes = sealed.export().expect("export");
        assert_eq!(StalenessBaselineWire::consume(&bytes).expect("consume"), sealed);

        let mut tampered = sealed.clone();
        tampered.rows = 4;
        assert!(tampered.export().is_err(), "an edited field breaks the seal");
        let mut raw = STALENESS_BASELINE_MAGIC.to_vec();
        ciborium::into_writer(&tampered, &mut raw).expect("encode");
        assert!(StalenessBaselineWire::consume(&raw).is_err());
        assert!(StalenessBaselineWire::consume(b"not a baseline").is_err());
        assert!(StalenessBaselineWire::consume(&bytes[..bytes.len() - 3]).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(StalenessBaselineWire::consume(&trailing).is_err());
        trailing.extend_from_slice(&bytes);
        assert!(StalenessBaselineWire::consume(&trailing).is_err());
    }
}
