//! Queryable lineage for 2.3 compositions.
//!
//! Each link names a stage (causal contract, evidence, external provider,
//! distribution artifact, transformation, decision contract, sensitivity input,
//! study-ranking provider, or a claim) and the links it was derived from.
//! Parents must precede their children, so a chain is acyclic by construction.
//! A consumer can ask which stages and identities stand behind any reported
//! number, and can require that particular stages are present.

use std::collections::{BTreeSet, HashMap};

/// Maximum links in one chain.
pub const MAX_CHAIN_LINKS: usize = 4_096;

/// What kind of object a link stands for.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompositionStage {
    /// An identified or compiled causal contract.
    CausalContract,
    /// An evidence factor or observed-data snapshot's factor.
    Evidence,
    /// A data snapshot.
    Data,
    /// A foreign provider object at an exact request.
    ExternalProvider,
    /// A portable distribution artifact.
    DistributionArtifact,
    /// A named transformation or checked equivalence.
    Transformation,
    /// A compiled decision contract.
    DecisionContract,
    /// An input to a sensitivity analysis.
    SensitivityInput,
    /// A provider used to rank candidate studies.
    StudyRankingProvider,
    /// A reported claim derived from the links above.
    Claim,
}

impl CompositionStage {
    /// Stable `snake_case` name for wires.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CausalContract => "causal_contract",
            Self::Evidence => "evidence",
            Self::Data => "data",
            Self::ExternalProvider => "external_provider",
            Self::DistributionArtifact => "distribution_artifact",
            Self::Transformation => "transformation",
            Self::DecisionContract => "decision_contract",
            Self::SensitivityInput => "sensitivity_input",
            Self::StudyRankingProvider => "study_ranking_provider",
            Self::Claim => "claim",
        }
    }

    /// Parse the wire spelling.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [
            Self::CausalContract,
            Self::Evidence,
            Self::Data,
            Self::ExternalProvider,
            Self::DistributionArtifact,
            Self::Transformation,
            Self::DecisionContract,
            Self::SensitivityInput,
            Self::StudyRankingProvider,
            Self::Claim,
        ]
        .into_iter()
        .find(|stage| stage.as_str() == name)
    }
}

/// One derivation step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionLink {
    /// Stable identity, unique within the chain.
    pub id: String,
    /// What this link stands for.
    pub stage: CompositionStage,
    /// Identities of the links it was derived from; each must appear earlier.
    pub parents: Vec<String>,
    /// The digests this link believes its parents have, in parent order. A
    /// wire-supplied chain states them so a changed predecessor refuses; chains
    /// built in process leave this `None`.
    pub declared_parent_digests: Option<Vec<String>>,
}

/// Why a chain is invalid or a query cannot be answered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProvenanceChainError {
    /// Blank identity or too many links.
    InvalidLink,
    /// Two links share an identity.
    DuplicateLink(String),
    /// A parent is unknown or does not precede its child.
    UnresolvedParent {
        /// Child link.
        child: String,
        /// Missing or later parent.
        parent: String,
    },
    /// The queried identity is not in the chain.
    UnknownLink(String),
    /// A required stage is absent from the queried lineage.
    MissingStage(CompositionStage),
    /// A carried or expected digest differs from the recomputed one.
    DigestMismatch {
        /// Link whose digest or declared predecessor digests differ.
        link: String,
        /// The digest(s) the chain computes; several are joined with `,`.
        expected: String,
        /// The digest(s) that were declared or supplied; several are joined with `,`.
        supplied: String,
    },
}

fn refusal(code: &'static str, detail: &str, link: &str) -> crate::ExternalRefusal {
    crate::ExternalRefusal {
        code,
        stage: "provenance",
        detail: detail.to_owned(),
        offending: Some(link.to_owned()),
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    }
}

impl ProvenanceChainError {
    /// Structured refusal with a `composition_provenance` detail.
    #[must_use]
    pub fn to_refusal(&self) -> crate::ExternalRefusal {
        let mismatch = crate::reason_code!("external_binding_mismatch");
        let invalid = crate::reason_code!("invalid_argument");
        match self {
            Self::InvalidLink => crate::ExternalRefusal {
                offending: None,
                ..refusal(invalid, "composition_provenance.invalid_link", "")
            },
            Self::DuplicateLink(id) => {
                refusal(invalid, "composition_provenance.duplicate_link", id)
            }
            Self::UnresolvedParent { child, parent } => crate::ExternalRefusal {
                expected: Some(parent.clone()),
                ..refusal(mismatch, "composition_provenance.unresolved_parent", child)
            },
            Self::UnknownLink(id) => refusal(mismatch, "composition_provenance.unknown_link", id),
            Self::MissingStage(stage) => crate::ExternalRefusal {
                offending: Some(stage.as_str().to_owned()),
                ..refusal(mismatch, "composition_provenance.missing_stage", "")
            },
            Self::DigestMismatch { link, expected, supplied } => crate::ExternalRefusal {
                expected: Some(expected.clone()),
                supplied: Some(supplied.clone()),
                remedy: Some("rebuild the lineage from the retained digests of every predecessor"),
                ..refusal(mismatch, "composition_provenance.digest_mismatch", link)
            },
        }
    }
}

/// Digest of one link: BLAKE3 over the little-endian `u64` length and bytes
/// of the id, the little-endian `u64` length and bytes of the stage wire
/// name, then the lowercase hex digest bytes of each parent in declared order.
fn link_digest(id: &str, stage: CompositionStage, parent_digests: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    let name = stage.as_str();
    hasher.update(&(id.len() as u64).to_le_bytes());
    hasher.update(id.as_bytes());
    hasher.update(&(name.len() as u64).to_le_bytes());
    hasher.update(name.as_bytes());
    for parent in parent_digests {
        hasher.update(parent.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

/// A validated, acyclic derivation chain with a Merkle digest per link.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenanceChain {
    links: Vec<CompositionLink>,
    index: HashMap<String, usize>,
    digests: Vec<String>,
}

impl ProvenanceChain {
    /// Validate links in dependency order and compute each link's digest.
    ///
    /// # Errors
    /// Blank or duplicate identities, oversize chains, and parents that are
    /// unknown or not earlier in the list refuse. A link whose declared
    /// parent digests differ from the recomputed ones refuses with
    /// `DigestMismatch`.
    pub fn new(links: Vec<CompositionLink>) -> Result<Self, ProvenanceChainError> {
        if links.len() > MAX_CHAIN_LINKS {
            return Err(ProvenanceChainError::InvalidLink);
        }
        let mut index = HashMap::with_capacity(links.len());
        let mut digests: Vec<String> = Vec::with_capacity(links.len());
        for (position, link) in links.iter().enumerate() {
            if link.id.trim().is_empty() || link.parents.iter().any(|p| p.trim().is_empty()) {
                return Err(ProvenanceChainError::InvalidLink);
            }
            if let Some(parent) = link.parents.iter().find(|p| !index.contains_key(*p)) {
                return Err(ProvenanceChainError::UnresolvedParent {
                    child: link.id.clone(),
                    parent: parent.clone(),
                });
            }
            if index.insert(link.id.clone(), position).is_some() {
                return Err(ProvenanceChainError::DuplicateLink(link.id.clone()));
            }
            let parent_digests: Vec<&str> =
                link.parents.iter().map(|p| digests[index[p]].as_str()).collect();
            if let Some(declared) = &link.declared_parent_digests {
                let same = declared.len() == parent_digests.len()
                    && declared.iter().zip(&parent_digests).all(|(d, p)| d == p);
                if !same {
                    return Err(ProvenanceChainError::DigestMismatch {
                        link: link.id.clone(),
                        expected: parent_digests.join(","),
                        supplied: declared.join(","),
                    });
                }
            }
            let digest = link_digest(&link.id, link.stage, &parent_digests);
            digests.push(digest);
        }
        Ok(Self { links, index, digests })
    }

    /// Links in dependency order.
    #[must_use]
    pub fn links(&self) -> &[CompositionLink] {
        &self.links
    }

    /// The Merkle digest of a link, lowercase hex.
    ///
    /// # Errors
    /// An identity outside the chain refuses.
    pub fn digest_of(&self, id: &str) -> Result<&str, ProvenanceChainError> {
        self.index
            .get(id)
            .map(|position| self.digests[*position].as_str())
            .ok_or_else(|| ProvenanceChainError::UnknownLink(id.to_owned()))
    }

    /// Require that a link has exactly the digest a reader retained.
    ///
    /// # Errors
    /// An unknown identity or a different digest refuses.
    pub fn verify_digest(&self, id: &str, expected: &str) -> Result<(), ProvenanceChainError> {
        let actual = self.digest_of(id)?;
        if actual == expected {
            Ok(())
        } else {
            Err(ProvenanceChainError::DigestMismatch {
                link: id.to_owned(),
                expected: actual.to_owned(),
                supplied: expected.to_owned(),
            })
        }
    }

    /// The link and all its ancestors, parents before children.
    ///
    /// # Errors
    /// An identity outside the chain refuses.
    pub fn lineage(&self, id: &str) -> Result<Vec<&CompositionLink>, ProvenanceChainError> {
        let start =
            *self.index.get(id).ok_or_else(|| ProvenanceChainError::UnknownLink(id.to_owned()))?;
        let mut keep = vec![false; self.links.len()];
        keep[start] = true;
        for position in (0..=start).rev() {
            if keep[position] {
                for parent in &self.links[position].parents {
                    keep[self.index[parent]] = true;
                }
            }
        }
        Ok(self.links.iter().zip(keep).filter_map(|(link, kept)| kept.then_some(link)).collect())
    }

    /// Stages present in the lineage of `id`, including its own.
    ///
    /// # Errors
    /// An identity outside the chain refuses.
    pub fn stages_behind(
        &self,
        id: &str,
    ) -> Result<BTreeSet<CompositionStage>, ProvenanceChainError> {
        Ok(self.lineage(id)?.into_iter().map(|link| link.stage).collect())
    }

    /// Require every named stage somewhere in the lineage of `id`.
    ///
    /// # Errors
    /// Reports the first missing stage, or an unknown identity.
    pub fn require_stages(
        &self,
        id: &str,
        required: &[CompositionStage],
    ) -> Result<(), ProvenanceChainError> {
        let present = self.stages_behind(id)?;
        match required.iter().find(|stage| !present.contains(stage)) {
            Some(stage) => Err(ProvenanceChainError::MissingStage(*stage)),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(id: &str, stage: CompositionStage, parents: &[&str]) -> CompositionLink {
        CompositionLink {
            id: id.into(),
            stage,
            parents: parents.iter().map(|p| (*p).to_owned()).collect(),
            declared_parent_digests: None,
        }
    }

    #[test]
    fn digests_are_stable_distinct_and_verifiable() {
        let chain = chain();
        let claim = chain.digest_of("claim").unwrap().to_owned();
        assert_eq!(claim.len(), 64);
        assert_eq!(chain.verify_digest("claim", &claim), Ok(()));
        assert!(matches!(
            chain.verify_digest("claim", "00"),
            Err(ProvenanceChainError::DigestMismatch { .. })
        ));
        assert_ne!(chain.digest_of("data").unwrap(), chain.digest_of("contract").unwrap());
    }

    fn chain() -> ProvenanceChain {
        use CompositionStage as S;
        ProvenanceChain::new(vec![
            link("contract", S::CausalContract, &[]),
            link("data", S::Data, &[]),
            link("evidence", S::Evidence, &["data"]),
            link("provider", S::ExternalProvider, &["evidence"]),
            link("unrelated", S::SensitivityInput, &[]),
            link("claim", S::Claim, &["contract", "provider"]),
        ])
        .unwrap()
    }

    #[test]
    fn lineage_names_exactly_the_ancestors_in_dependency_order() {
        let chain = chain();
        let ids: Vec<_> =
            chain.lineage("claim").unwrap().into_iter().map(|l| l.id.as_str()).collect();
        assert_eq!(ids, ["contract", "data", "evidence", "provider", "claim"]);
        assert!(!ids.contains(&"unrelated"));
        assert_eq!(chain.lineage("data").unwrap().len(), 1);
        assert_eq!(chain.lineage("nope"), Err(ProvenanceChainError::UnknownLink("nope".into())));
    }

    #[test]
    fn required_stages_must_stand_behind_the_queried_number() {
        use CompositionStage as S;
        let chain = chain();
        assert_eq!(
            chain.require_stages("claim", &[S::CausalContract, S::ExternalProvider, S::Data]),
            Ok(())
        );
        assert_eq!(
            chain.require_stages("claim", &[S::DecisionContract]),
            Err(ProvenanceChainError::MissingStage(S::DecisionContract))
        );
        // A sibling's stage does not stand behind a different link.
        assert_eq!(
            chain.require_stages("provider", &[S::CausalContract]),
            Err(ProvenanceChainError::MissingStage(S::CausalContract))
        );
    }

    #[test]
    fn invalid_chains_refuse() {
        use CompositionStage as S;
        assert_eq!(
            ProvenanceChain::new(vec![link("a", S::Data, &["b"]), link("b", S::Data, &[])]),
            Err(ProvenanceChainError::UnresolvedParent { child: "a".into(), parent: "b".into() })
        );
        assert_eq!(
            ProvenanceChain::new(vec![link("a", S::Data, &[]), link("a", S::Claim, &[])]),
            Err(ProvenanceChainError::DuplicateLink("a".into()))
        );
        assert_eq!(
            ProvenanceChain::new(vec![link(" ", S::Data, &[])]),
            Err(ProvenanceChainError::InvalidLink)
        );
        for stage in [S::Claim, S::StudyRankingProvider, S::DecisionContract] {
            assert_eq!(CompositionStage::from_name(stage.as_str()), Some(stage));
        }
    }
}
