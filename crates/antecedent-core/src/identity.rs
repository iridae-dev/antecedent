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

/// FNV-1a 64-bit offset basis.
pub const FNV1A_64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
pub const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Streaming 64-bit FNV-1a for in-memory cache keys.
///
/// Not collision resistant and not a persisted identity: use
/// [`CanonicalWriter`] for anything that is stored or compared across
/// processes. [`Self::byte`]/[`Self::bytes`]/[`Self::word`] are standard
/// FNV-1a (per-byte xor then multiply). [`Self::fold`] xors a whole `u64`
/// before one multiply; it is cheaper and weaker, kept so an existing
/// in-memory key keeps its value.
#[derive(Clone, Copy, Debug)]
pub struct Fnv1a64(u64);

impl Default for Fnv1a64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv1a64 {
    /// Start at the FNV offset basis.
    #[must_use]
    pub const fn new() -> Self {
        Self(FNV1A_64_OFFSET)
    }

    /// Fold one byte (xor, then multiply by the prime).
    pub fn byte(&mut self, byte: u8) -> &mut Self {
        self.0 ^= u64::from(byte);
        self.0 = self.0.wrapping_mul(FNV1A_64_PRIME);
        self
    }

    /// Fold bytes in order, per byte.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        for &byte in bytes {
            self.byte(byte);
        }
        self
    }

    /// Fold a `u64` as its eight little-endian bytes, per byte.
    pub fn word(&mut self, word: u64) -> &mut Self {
        self.bytes(&word.to_le_bytes())
    }

    /// Xor a whole `u64` in, then multiply once (word-at-a-time variant).
    pub fn fold(&mut self, word: u64) -> &mut Self {
        self.0 ^= word;
        self.0 = self.0.wrapping_mul(FNV1A_64_PRIME);
        self
    }

    /// The current 64-bit hash.
    #[must_use]
    pub const fn finish(&self) -> u64 {
        self.0
    }
}

/// Standard 64-bit FNV-1a of `bytes`.
#[must_use]
pub fn fnv1a_u64(bytes: &[u8]) -> u64 {
    Fnv1a64::new().bytes(bytes).finish()
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

    /// Write the same length-prefixed bytes while checking the caller's budget
    /// or cancellation before each bounded hash update, including empty input.
    ///
    /// # Errors
    /// Propagates the check's error. Discard this writer after an error because
    /// the field may have been partially written.
    pub fn try_bytes<E>(
        &mut self,
        bytes: &[u8],
        mut check: impl FnMut() -> Result<(), E>,
    ) -> Result<&mut Self, E> {
        check()?;
        self.word(bytes.len() as u64);
        for chunk in bytes.chunks(4096) {
            check()?;
            self.hasher.update(chunk);
        }
        Ok(self)
    }

    /// A count-prefixed sequence of reals: the length as a word, then each `to_bits` word.
    pub fn reals(&mut self, values: &[f64]) -> &mut Self {
        self.word(values.len() as u64);
        for &value in values {
            self.real(value);
        }
        self
    }

    /// A count-prefixed sequence of `u32` values, each widened to one word.
    pub fn u32s(&mut self, values: &[u32]) -> &mut Self {
        self.word(values.len() as u64);
        for &value in values {
            self.word(u64::from(value));
        }
        self
    }

    /// A count-prefixed sequence of `u64` words.
    pub fn words(&mut self, values: &[u64]) -> &mut Self {
        self.word(values.len() as u64);
        for &value in values {
            self.word(value);
        }
        self
    }

    /// A count-prefixed sequence of texts: the length as a word, then each as [`Self::text`].
    pub fn texts<S: AsRef<str>>(&mut self, values: &[S]) -> &mut Self {
        self.word(values.len() as u64);
        for value in values {
            self.text(value.as_ref());
        }
        self
    }

    /// An optional text: a presence word (`0` or `1`), then the text when present.
    pub fn opt_text(&mut self, value: Option<&str>) -> &mut Self {
        match value {
            Some(text) => self.word(1).text(text),
            None => self.word(0),
        }
    }

    /// An optional word: a presence word (`0` or `1`), then the value when present.
    pub fn opt_word(&mut self, value: Option<u64>) -> &mut Self {
        match value {
            Some(word) => self.word(1).word(word),
            None => self.word(0),
        }
    }

    /// An optional real: a presence word (`0` or `1`), then its bit pattern when present.
    pub fn opt_real(&mut self, value: Option<f64>) -> &mut Self {
        match value {
            Some(real) => self.word(1).real(real),
            None => self.word(0),
        }
    }

    /// The 32-byte BLAKE3 digest of everything written so far (non-consuming).
    #[must_use]
    pub fn finish(&self) -> [u8; 32] {
        *self.hasher.finalize().as_bytes()
    }

    /// [`Self::finish`] as 64 lowercase hex characters.
    #[must_use]
    pub fn finish_hex(&self) -> String {
        to_hex(&self.finish())
    }

    /// Digest of one opaque byte string (a stored artifact or CBOR body) under
    /// the domain `tag`: [`Self::new`] then [`Self::bytes`].
    #[must_use]
    pub fn digest_bytes(tag: &str, bytes: &[u8]) -> [u8; 32] {
        Self::new(tag).bytes(bytes).finish()
    }

    /// [`Self::digest_bytes`] as 64 lowercase hex characters.
    #[must_use]
    pub fn digest_bytes_hex(tag: &str, bytes: &[u8]) -> String {
        to_hex(&Self::digest_bytes(tag, bytes))
    }
}

/// Unframed BLAKE3 digest of `bytes` as 64 lowercase hex characters.
///
/// For content addressing of an opaque byte string (an artifact container, a
/// CBOR body) where the digest must equal what `b3sum` reports. Anything built
/// from several fields goes through [`CanonicalWriter`] instead, which frames
/// and domain-separates them.
#[must_use]
pub fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// BLAKE3 derive-key digest of `parts` fed in order under the key-derivation `context`.
///
/// The parts are concatenated with no framing of their own: callers that need length
/// prefixes or a format word pass those as explicit parts. The byte stream hashed is
/// exactly `context`-keyed BLAKE3 over `parts[0] || parts[1] || ...`.
#[must_use]
pub fn derive_key_digest(context: &str, parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(context);
    for part in parts {
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
}

/// Parse 64 hex characters, lowercase or uppercase, into 32 bytes.
#[must_use]
pub fn digest_from_hex(text: &str) -> Option<[u8; 32]> {
    parse_hex32(text, true)
}

/// Parse 64 lowercase hex characters into 32 bytes; uppercase is refused.
#[must_use]
pub fn digest_from_lower_hex(text: &str) -> Option<[u8; 32]> {
    parse_hex32(text, false)
}

fn parse_hex32(text: &str, accept_upper: bool) -> Option<[u8; 32]> {
    let bytes = text.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let nibble = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' if accept_upper => Some(c - b'A' + 10),
        _ => None,
    };
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *slot = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Some(out)
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
    /// Declared `cyclic.solution.v1` semantics of a feedback program: family,
    /// solve schedule, certificate demand and selection policy. Comparing it does
    /// not solve the system.
    CyclicSolution,
    /// Recovery of a linear feedback law from clamp experiments: the experiment
    /// data snapshot, the declared regime contract and the recovery derivation.
    CyclicRecovery,
}

impl IdentityDomain {
    /// Closed set of domains. A new variant fails dictionary tests until listed.
    pub const ALL: [IdentityDomain; 15] = [
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
        Self::CyclicSolution,
        Self::CyclicRecovery,
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
            Self::CyclicSolution => "cyclic_solution",
            Self::CyclicRecovery => "cyclic_recovery",
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
            Self::CyclicSolution => "antecedent.identity.cyclic_solution.v1",
            Self::CyclicRecovery => "antecedent.identity.cyclic_recovery.v1",
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

    /// Digest of a real vector under `tag`: its length, then every `to_bits` word.
    ///
    /// Equality is bitwise, so a single changed bit anywhere in `values` changes the digest.
    #[must_use]
    pub fn of_reals(tag: &str, values: &[f64]) -> Self {
        Self::from_bytes(CanonicalWriter::new(tag).reals(values).finish())
    }

    /// Digest of a `u32` vector under `tag`: its length, then every element.
    #[must_use]
    pub fn of_u32s(tag: &str, values: &[u32]) -> Self {
        Self::from_bytes(CanonicalWriter::new(tag).u32s(values).finish())
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

/// One input a reusable product's key must cover.
///
/// A product keyed without an axis it reads would be served to a request that differs
/// only on that axis. [`ReusableProduct::required_axes`] states what each product reads, and
/// a key reports what it covers, so an omission is a checked difference rather than a review
/// comment.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum DependencyAxis {
    /// Scientific target: functional, thresholds, contrast.
    Target,
    /// Schema variables and observation contract.
    Schema,
    /// Graph, structure class, prior and tier background.
    Model,
    /// Numeric design matrix and its shape.
    Design,
    /// Row identities of the complete cases.
    Rows,
    /// Realized fold assignment, not only its seed.
    Folds,
    /// Outcome values the fit is trained to.
    Outcome,
    /// Treatment values and coding.
    Treatment,
    /// Sampling or target weights.
    Weights,
    /// Provider identity and version.
    Provider,
    /// Learner, GLM and estimator options.
    Options,
    /// Randomness the fit consumes.
    Seed,
    /// Inference mode.
    Inference,
    /// Numerical policy and named tolerances.
    Numerical,
}

impl DependencyAxis {
    /// Every axis, in declaration order.
    pub const ALL: [Self; 14] = [
        Self::Target,
        Self::Schema,
        Self::Model,
        Self::Design,
        Self::Rows,
        Self::Folds,
        Self::Outcome,
        Self::Treatment,
        Self::Weights,
        Self::Provider,
        Self::Options,
        Self::Seed,
        Self::Inference,
        Self::Numerical,
    ];
}

/// A product that is computed once and served again.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum ReusableProduct {
    /// Identification retained with a lowered program.
    RetainedIdentification,
    /// Cross-fitted propensity scores of a generalized linear model.
    PropensityFit,
    /// Cross-fitted arm-specific outcome regressions.
    OutcomeFit,
    /// Learner-backed cross-fitted nuisance bundle (DML and doubly robust).
    LearnerBundle,
    /// Blocked out-of-fold nuisances of a sequential contrast.
    SequentialNuisance,
    /// Per-split multi-action propensity and arm outcome means of an observational policy value.
    PolicyNuisance,
}

impl ReusableProduct {
    /// Every product, in declaration order.
    pub const ALL: [Self; 6] = [
        Self::RetainedIdentification,
        Self::PropensityFit,
        Self::OutcomeFit,
        Self::LearnerBundle,
        Self::SequentialNuisance,
        Self::PolicyNuisance,
    ];

    /// Axes this product reads. A key for it must cover every one.
    #[must_use]
    pub const fn required_axes(self) -> &'static [DependencyAxis] {
        use DependencyAxis as A;
        match self {
            Self::RetainedIdentification => {
                &[A::Target, A::Schema, A::Model, A::Options, A::Inference]
            }
            Self::PropensityFit => &[
                A::Design,
                A::Rows,
                A::Folds,
                A::Treatment,
                A::Weights,
                A::Provider,
                A::Options,
                A::Numerical,
            ],
            Self::OutcomeFit => &[
                A::Target,
                A::Design,
                A::Rows,
                A::Folds,
                A::Outcome,
                A::Treatment,
                A::Weights,
                A::Provider,
                A::Options,
                A::Numerical,
            ],
            Self::LearnerBundle | Self::PolicyNuisance => &[
                A::Design,
                A::Rows,
                A::Folds,
                A::Outcome,
                A::Treatment,
                A::Weights,
                A::Provider,
                A::Options,
                A::Seed,
                A::Numerical,
            ],
            Self::SequentialNuisance => &[
                A::Target,
                A::Design,
                A::Rows,
                A::Folds,
                A::Outcome,
                A::Treatment,
                A::Weights,
                A::Provider,
                A::Options,
                A::Seed,
                A::Numerical,
            ],
        }
    }

    /// Axes `product` reads that `covered` does not name, in axis order.
    #[must_use]
    pub fn omitted_axes(self, covered: &[DependencyAxis]) -> Vec<DependencyAxis> {
        self.required_axes().iter().copied().filter(|axis| !covered.contains(axis)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_hashing_preserves_framing_and_stops_between_chunks() {
        use crate::recalc::StageIdentity;
        for size in [0, 1, 4096, 4097, 20_000] {
            let bytes = vec![0xab; size];
            let mut expected = CanonicalWriter::new("checked");
            expected.bytes(&bytes);
            let mut actual = CanonicalWriter::new("checked");
            actual.try_bytes(&bytes, || Ok::<_, ()>(())).unwrap();
            assert_eq!(actual.finish(), expected.finish());
            let parts: &[&[u8]] = &[&[], &bytes, b"tail"];
            assert_eq!(
                StageIdentity::try_of("label", parts, || Ok::<_, ()>(())).unwrap(),
                StageIdentity::of("label", parts)
            );
        }
        let mut checks = 0;
        let mut writer = CanonicalWriter::new("checked");
        let result = writer.try_bytes(&[0; 20_000], || {
            checks += 1;
            if checks == 3 { Err("cancelled") } else { Ok(()) }
        });
        assert_eq!(result.err(), Some("cancelled"));
        assert_eq!(checks, 3);
        let mut partial = CanonicalWriter::new("checked");
        partial.word(20_000);
        partial.hasher.update(&[0; 4096]);
        assert_eq!(writer.finish(), partial.finish());
    }

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
    fn one_shot_byte_digest_is_the_framed_tag_then_bytes() {
        let mut stream = CanonicalWriter::new("test.tag");
        stream.bytes(b"payload");
        assert_eq!(CanonicalWriter::digest_bytes("test.tag", b"payload"), stream.finish());
        assert_eq!(
            CanonicalWriter::digest_bytes_hex("test.tag", b"payload"),
            "4f754574b2b5f7d97cf0f17422e58bfd9a06e4a82f51522d5d372018766a4148"
        );
        assert_ne!(
            CanonicalWriter::digest_bytes_hex("test.tag", b"payload"),
            CanonicalWriter::digest_bytes_hex("test.tag2", b"payload")
        );
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
    fn fnv1a_matches_reference_vectors() {
        assert_eq!(fnv1a_u64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_u64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_u64(b"foobar"), 0x8594_4171_f739_67e8);
        let mut by_word = Fnv1a64::new();
        by_word.word(0x0102_0304_0506_0708);
        assert_eq!(by_word.finish(), fnv1a_u64(&0x0102_0304_0506_0708_u64.to_le_bytes()));
        let mut folded = Fnv1a64::new();
        folded.fold(7);
        assert_eq!(folded.finish(), (FNV1A_64_OFFSET ^ 7).wrapping_mul(FNV1A_64_PRIME));
    }

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
        assert_eq!(writer.finish_hex(), expected.to_hex().to_string());
        assert_eq!(
            writer.finish_hex(),
            "77a467e6a59c1f6dca516663cb868433147eb4c625ddbdbe07207fdf3e6a11f5"
        );

        // Optional and sequence fields are word-framed and separate absent from empty.
        let mut some = CanonicalWriter::new("t");
        some.opt_text(Some("")).reals(&[1.0]).texts(&["a"]).opt_word(None).opt_real(Some(0.5));
        let mut none = CanonicalWriter::new("t");
        none.opt_text(None).reals(&[1.0]).texts(&["a"]).opt_word(None).opt_real(Some(0.5));
        assert_ne!(some.finish(), none.finish());
        assert_eq!(some.finish_hex().len(), 64);
        assert_eq!(blake3_hex(b"abc"), blake3::hash(b"abc").to_hex().to_string());

        // `bytes` is the same framing as `text` over the same octets.
        let mut a = CanonicalWriter::new("t");
        a.text("xy");
        let mut b = CanonicalWriter::new("t");
        b.bytes(b"xy");
        assert_eq!(a.finish(), b.finish());
    }

    #[test]
    fn derive_key_digest_feeds_parts_unframed_in_order() {
        let mut hasher = blake3::Hasher::new_derive_key("antecedent.test.derive");
        hasher.update(&7_u16.to_le_bytes());
        hasher.update(b"payload");
        let expected = *hasher.finalize().as_bytes();
        let parts: [&[u8]; 2] = [&7_u16.to_le_bytes(), b"payload"];
        assert_eq!(derive_key_digest("antecedent.test.derive", &parts), expected);
        // Part boundaries carry no framing: the same octets split elsewhere agree.
        let joined: [&[u8]; 1] = [b"\x07\x00payload"];
        assert_eq!(derive_key_digest("antecedent.test.derive", &joined), expected);
        assert_ne!(derive_key_digest("antecedent.test.other", &joined), expected);
    }

    #[test]
    fn hex_parsers_round_trip_and_differ_only_on_case() {
        let bytes: [u8; 32] = core::array::from_fn(|i| {
            u8::try_from(i).expect("index below 32").wrapping_mul(37).wrapping_add(11)
        });
        let lower = to_hex(&bytes);
        let upper = lower.to_uppercase();
        assert_eq!(digest_from_hex(&lower), Some(bytes));
        assert_eq!(digest_from_hex(&upper), Some(bytes));
        assert_eq!(digest_from_lower_hex(&lower), Some(bytes));
        assert_eq!(digest_from_lower_hex(&upper), None);
        assert_eq!(digest_from_hex(&lower[..62]), None);
        assert_eq!(digest_from_hex(&format!("{}g", &lower[..63])), None);
    }

    #[test]
    fn real_vector_digest_tracks_every_bit_length_and_tag() {
        let base = SemanticDigest::of_reals("t", &[1.0, 2.0, 3.0]);
        assert_eq!(base, SemanticDigest::of_reals("t", &[1.0, 2.0, 3.0]));
        assert_ne!(
            base,
            SemanticDigest::of_reals("t", &[1.0, 2.0, f64::from_bits(3.0f64.to_bits() + 1)])
        );
        assert_ne!(base, SemanticDigest::of_reals("t", &[1.0, 2.0]));
        assert_ne!(base, SemanticDigest::of_reals("u", &[1.0, 2.0, 3.0]));
        assert_ne!(SemanticDigest::of_reals("t", &[0.0]), SemanticDigest::of_reals("t", &[-0.0]));
        assert_ne!(SemanticDigest::of_u32s("t", &[1, 2]), SemanticDigest::of_u32s("t", &[2, 1]));
    }

    #[test]
    fn every_reusable_product_reads_axes_and_omissions_are_reported() {
        for product in ReusableProduct::ALL {
            let required = product.required_axes();
            assert!(!required.is_empty());
            assert!(product.omitted_axes(required).is_empty());
            for axis in required {
                let without: Vec<_> = required.iter().copied().filter(|a| a != axis).collect();
                assert_eq!(product.omitted_axes(&without), vec![*axis]);
            }
        }
        for axis in DependencyAxis::ALL {
            assert!(
                ReusableProduct::ALL.iter().any(|p| p.required_axes().contains(&axis)),
                "{axis:?} is read by no product"
            );
        }
    }
}
