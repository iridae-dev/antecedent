//! Domain-separated semantic identities for compiled causal contracts.
//!
//! Digests are computed in `antecedent-io` from canonical wire encodings.
//! This module owns the scientific record: which layer a digest belongs to,
//! and that layers are not interchangeable.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use std::sync::Arc;

/// Lowercase hex encoding of `bytes` (two characters per byte).
#[must_use]
pub fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[usize::from(b >> 4)] as char);
        out.push(HEX[usize::from(b & 0x0f)] as char);
    }
    out
}

/// Streaming length-prefixed canonical writer hashed with BLAKE3.
///
/// Framing: every field is either a little-endian `u64` word or a `u64`
/// little-endian length followed by raw bytes. [`Self::new`] writes the tag as
/// [`Self::text`], so distinct tags separate distinct encodings. Reals enter as
/// their IEEE-754 bit pattern, so `-0.0` and `0.0` differ and `NaN` payloads are
/// preserved.
///
/// The byte stream is identical to the buffered `Canon` builders previously
/// private to `antecedent-io`, so a digest computed by either agrees.
#[derive(Clone, Debug)]
pub struct CanonicalWriter {
    hasher: blake3::Hasher,
}

impl CanonicalWriter {
    /// Start a stream with a domain `tag` (written as length-prefixed text).
    #[must_use]
    pub fn new(tag: &str) -> Self {
        let mut writer = Self { hasher: blake3::Hasher::new() };
        writer.text(tag);
        writer
    }

    /// One `u64` word, little-endian.
    pub fn word(&mut self, word: u64) -> &mut Self {
        self.hasher.update(&word.to_le_bytes());
        self
    }

    /// Length-prefixed UTF-8 text.
    pub fn text(&mut self, text: &str) -> &mut Self {
        self.bytes(text.as_bytes())
    }

    /// A real as its `to_bits` word.
    pub fn real(&mut self, value: f64) -> &mut Self {
        self.word(value.to_bits())
    }

    /// Length-prefixed raw bytes.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.word(bytes.len() as u64);
        self.hasher.update(bytes);
        self
    }

    /// The 32-byte BLAKE3 digest of everything written so far (non-consuming).
    #[must_use]
    pub fn finish(&self) -> [u8; 32] {
        *self.hasher.finalize().as_bytes()
    }
}

/// Identity encoding format. Bump only when the canonical payload changes.
///
/// Version 2 canonicalizes graph edge order, replaces debug-rendered estimator
/// configuration with structured wires, and widens each layer's coverage (target
/// population in the program, posterior atom weights, prior contents, execution
/// budgets). No version-1 digest is comparable with a version-2 digest.
pub const IDENTITY_FORMAT: u16 = 2;

/// Format tag recorded beside every digest.
pub const IDENTITY_FORMAT_TAG: &str = "antecedent.identity.v2";

/// Scientific layer a digest identifies.
///
/// One overloaded hash would make reuse and audit misleading. Each variant
/// has its own BLAKE3 derive-key domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum IdentityDomain {
    /// Query, population, interventions, outcome functional, temporal
    /// policy/horizons, and variable-name bindings.
    Target,
    /// Population-free target question plus accepted structural semantics
    /// (including posterior atoms and weights, class priors, selection
    /// diagrams) and the observation/evidence contract. Identifier
    /// configuration is not this layer.
    Identification,
    /// Concrete identification products (status, estimands, arena, derivation).
    IdentificationProduct,
    /// Target (including its population), premises, identification
    /// products, and licensed inferential commitments. A different graph or a
    /// different target population is a different program.
    Program,
    /// Resolved prior contents, backend and likelihood, numeric
    /// configuration, validation, and dependence. Independent of structure.
    InferenceBinding,
    /// Schema and observation contract, not row contents.
    Observation,
    /// Observation plus typed contents, masks, weights, and ordered temporal/unit partitions.
    DataSnapshot,
    /// Seeds, backend and kernel policy, determinism, adaptive budgets, and
    /// implementation versions.
    Execution,
    /// Portable claim envelope over a program and one result.
    Claim,
    /// Score-table, fold, nuisance, and shared-draw reuse. Stricter than
    /// identification: matching names and shapes are not this layer.
    ScoreReuse,
    /// Row-weight retarget payload bound to one data snapshot.
    TargetWeights,
    /// Whole learner-backed trial-transport claim: graph, query, raw inputs, frozen
    /// options, seed, and the retained score and bootstrap evidence.
    LearnedTrial,
    /// Whole structural transport certificate outcome (positive proof or refusal).
    TransportCertificate,
}

impl IdentityDomain {
    /// Closed set of domains. A new variant fails dictionary tests until listed.
    pub const ALL: [IdentityDomain; 13] = [
        Self::Target,
        Self::Identification,
        Self::IdentificationProduct,
        Self::Program,
        Self::InferenceBinding,
        Self::Observation,
        Self::DataSnapshot,
        Self::Execution,
        Self::Claim,
        Self::ScoreReuse,
        Self::TargetWeights,
        Self::LearnedTrial,
        Self::TransportCertificate,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Target => "target",
            Self::Identification => "identification",
            Self::IdentificationProduct => "identification_product",
            Self::Program => "program",
            Self::InferenceBinding => "inference_binding",
            Self::Observation => "observation",
            Self::DataSnapshot => "data_snapshot",
            Self::Execution => "execution",
            Self::Claim => "claim",
            Self::ScoreReuse => "score_reuse",
            Self::TargetWeights => "target_weights",
            Self::LearnedTrial => "learned_trial",
            Self::TransportCertificate => "transport_certificate",
        }
    }

    /// BLAKE3 derive-key context for this domain.
    #[must_use]
    pub const fn derive_key(self) -> &'static str {
        match self {
            Self::Target => "antecedent.identity.target.v1",
            Self::Identification => "antecedent.identity.identification.v1",
            Self::IdentificationProduct => "antecedent.identity.identification_product.v1",
            Self::Program => "antecedent.identity.program.v1",
            Self::InferenceBinding => "antecedent.identity.inference_binding.v1",
            Self::Observation => "antecedent.identity.observation.v1",
            Self::DataSnapshot => "antecedent.identity.data_snapshot.v1",
            Self::Execution => "antecedent.identity.execution.v1",
            Self::Claim => "antecedent.identity.claim.v1",
            Self::ScoreReuse => "antecedent.identity.score_reuse.v1",
            Self::TargetWeights => "antecedent.identity.target_weights.v1",
            Self::LearnedTrial => "antecedent.identity.learned_trial.v1",
            Self::TransportCertificate => "antecedent.identity.transport_certificate.v1",
        }
    }
}

impl fmt::Display for IdentityDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 32-byte domain-separated digest. Not a Rust `Hash` of a `Debug` string.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct SemanticDigest {
    bytes: [u8; 32],
}

impl SemanticDigest {
    /// Construct from a finalized BLAKE3 output.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self { bytes }
    }

    /// Raw bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    /// Lowercase hex encoding (64 characters).
    #[must_use]
    pub fn to_hex(&self) -> String {
        to_hex(&self.bytes)
    }
}

impl fmt::Display for SemanticDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Digest plus the domain it was computed under.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct IdentityRef {
    /// Layer this digest identifies.
    pub domain: IdentityDomain,
    /// Domain-separated digest.
    pub digest: SemanticDigest,
}

impl IdentityRef {
    /// Bind a digest to its domain.
    #[must_use]
    pub const fn new(domain: IdentityDomain, digest: SemanticDigest) -> Self {
        Self { domain, digest }
    }
}

/// Domain-separated identities for one compiled contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct ContractIdentities {
    /// Causal target.
    pub target: SemanticDigest,
    /// Identification premises (target + structure + observation).
    pub identification: SemanticDigest,
    /// Cached identification products, when preparation produced them.
    pub identification_product: Option<SemanticDigest>,
    /// Program: target + premises + products + licensed inferential commitments.
    ///
    /// `None` on a cheap structural inspection: it runs no identification, so
    /// the products the program covers do not exist yet and the program
    /// identity is unavailable rather than a digest of a different program.
    pub program: Option<SemanticDigest>,
    /// Inference binding (priors, numeric knobs, validation, dependence).
    pub inference_binding: SemanticDigest,
    /// Schema / observation contract.
    pub observation: SemanticDigest,
    /// Data snapshot identity, including content digests and ordered partitions.
    pub data_snapshot: SemanticDigest,
}

impl ContractIdentities {
    /// Bind every identity layer.
    #[must_use]
    pub const fn new(
        target: SemanticDigest,
        identification: SemanticDigest,
        identification_product: Option<SemanticDigest>,
        program: Option<SemanticDigest>,
        inference_binding: SemanticDigest,
        observation: SemanticDigest,
        data_snapshot: SemanticDigest,
    ) -> Self {
        Self {
            target,
            identification,
            identification_product,
            program,
            inference_binding,
            observation,
            data_snapshot,
        }
    }

    /// All present identities as domain-tagged refs, in layer order.
    #[must_use]
    pub fn refs(&self) -> Arc<[IdentityRef]> {
        let mut out = vec![
            IdentityRef::new(IdentityDomain::Target, self.target),
            IdentityRef::new(IdentityDomain::Identification, self.identification),
        ];
        if let Some(product) = self.identification_product {
            out.push(IdentityRef::new(IdentityDomain::IdentificationProduct, product));
        }
        if let Some(program) = self.program {
            out.push(IdentityRef::new(IdentityDomain::Program, program));
        }
        out.extend([
            IdentityRef::new(IdentityDomain::InferenceBinding, self.inference_binding),
            IdentityRef::new(IdentityDomain::Observation, self.observation),
            IdentityRef::new(IdentityDomain::DataSnapshot, self.data_snapshot),
        ]);
        Arc::from(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domains_are_distinct_and_named() {
        let domains = IdentityDomain::ALL;
        let mut keys = std::collections::BTreeSet::new();
        let mut names = std::collections::BTreeSet::new();
        for domain in domains {
            assert!(keys.insert(domain.derive_key()));
            assert!(names.insert(domain.as_str()));
            assert!(domain.derive_key().contains(domain.as_str()) || domain.as_str().contains('_'));
        }
        assert_eq!(keys.len(), IdentityDomain::ALL.len());
    }

    #[test]
    fn digest_hex_is_stable_and_lowercase() {
        let mut bytes = [0u8; 32];
        bytes[0] = 0xab;
        bytes[31] = 0xcd;
        let digest = SemanticDigest::from_bytes(bytes);
        let hex = digest.to_hex();
        assert_eq!(hex.len(), 64);
        assert!(hex.starts_with("ab"));
        assert!(hex.ends_with("cd"));
        assert!(hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
    }

    /// The old `antecedent-io` `Canon` buffered every field and hashed once; the
    /// streaming writer must produce the same digest.
    #[test]
    fn canonical_writer_matches_buffered_canon_framing() {
        fn word(buf: &mut Vec<u8>, w: u64) {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        fn text(buf: &mut Vec<u8>, t: &str) {
            word(buf, t.len() as u64);
            buf.extend_from_slice(t.as_bytes());
        }
        let mut old = Vec::new();
        text(&mut old, "antecedent.test.canon");
        word(&mut old, 42);
        text(&mut old, "label");
        word(&mut old, (-0.5_f64).to_bits());
        text(&mut old, "");
        let expected = blake3::hash(&old);

        let mut writer = CanonicalWriter::new("antecedent.test.canon");
        writer.word(42).text("label").real(-0.5).text("");
        assert_eq!(writer.finish(), *expected.as_bytes());
        assert_eq!(to_hex(&writer.finish()), expected.to_hex().to_string());
        assert_eq!(
            to_hex(&writer.finish()),
            "77a467e6a59c1f6dca516663cb868433147eb4c625ddbdbe07207fdf3e6a11f5"
        );

        // `bytes` is the same framing as `text` over the same octets.
        let mut a = CanonicalWriter::new("t");
        a.text("xy");
        let mut b = CanonicalWriter::new("t");
        b.bytes(b"xy");
        assert_eq!(a.finish(), b.finish());
    }
}
