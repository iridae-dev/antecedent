//! Domain-separated semantic identities for compiled causal contracts.
//!
//! Digests are computed in `antecedent-io` from canonical wire encodings.
//! This module owns the scientific record: which layer a digest belongs to,
//! and that layers are not interchangeable.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use std::sync::Arc;

/// Identity encoding format. Bump only when the canonical payload changes.
pub const IDENTITY_FORMAT: u16 = 1;

/// Format tag recorded beside every digest.
pub const IDENTITY_FORMAT_TAG: &str = "antecedent.identity.v1";

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
    /// Target plus accepted structural semantics and observation/evidence
    /// contract. Identifier configuration is not this layer.
    Identification,
    /// Concrete identification products (status, estimands, arena, derivation).
    IdentificationProduct,
    /// Target, premises, identification products, and licensed inferential
    /// commitments. A different graph is a different program.
    Program,
    /// Resolved prior, numeric configuration, validation, and dependence.
    InferenceBinding,
    /// Schema and observation contract, not row contents.
    Observation,
    /// Observation plus typed contents, masks, weights, and ordered temporal/unit partitions.
    DataSnapshot,
    /// Seeds, backend, budgets, and implementation versions.
    Execution,
    /// Portable claim envelope over a program and one result.
    Claim,
    /// Score-table, fold, nuisance, and shared-draw reuse. Stricter than
    /// identification: matching names and shapes are not this layer.
    ScoreReuse,
    /// Row-weight retarget payload bound to one data snapshot.
    TargetWeights,
}

impl IdentityDomain {
    /// Closed set of domains. A new variant fails dictionary tests until listed.
    pub const ALL: [IdentityDomain; 11] = [
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
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for b in self.bytes {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
        out
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
    /// Program: premises + products + licensed inferential commitments.
    pub program: SemanticDigest,
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
        program: SemanticDigest,
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
        out.extend([
            IdentityRef::new(IdentityDomain::Program, self.program),
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
}
