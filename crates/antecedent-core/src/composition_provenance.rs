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
}

/// A validated, acyclic derivation chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenanceChain {
    links: Vec<CompositionLink>,
    index: HashMap<String, usize>,
}

impl ProvenanceChain {
    /// Validate links in dependency order.
    ///
    /// # Errors
    /// Blank or duplicate identities, oversize chains, and parents that are
    /// unknown or not earlier in the list refuse.
    pub fn new(links: Vec<CompositionLink>) -> Result<Self, ProvenanceChainError> {
        if links.len() > MAX_CHAIN_LINKS {
            return Err(ProvenanceChainError::InvalidLink);
        }
        let mut index = HashMap::with_capacity(links.len());
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
        }
        Ok(Self { links, index })
    }

    /// Links in dependency order.
    #[must_use]
    pub fn links(&self) -> &[CompositionLink] {
        &self.links
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
        }
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
