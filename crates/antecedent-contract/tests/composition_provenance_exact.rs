//! Provenance chain: which links a change reaches, derived by hand from the declared graph.
//!
//! A link's digest covers its own id and stage and the digests of its parents, so a change to
//! one link changes exactly that link and its descendants. Renaming the contract below reaches
//! the data snapshot, the provider and the decision contract (its children) and, through the
//! decision contract, the claim; the distribution, the transformation and the unrelated data link
//! share no ancestor with the contract and keep their digests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_contract::{CompositionLink, CompositionStage as S, ProvenanceChain};

type Row = (&'static str, S, Vec<&'static str>);

fn chain(contract: &'static str) -> (ProvenanceChain, Vec<&'static str>) {
    let rows: Vec<Row> = vec![
        (contract, S::CausalContract, vec![]),
        ("data:snapshot-1", S::Data, vec![contract]),
        ("provider:lab/curve@v1", S::ExternalProvider, vec!["data:snapshot-1"]),
        ("distribution:d1", S::DistributionArtifact, vec![]),
        ("transform:joint", S::Transformation, vec!["distribution:d1"]),
        ("decision:u1", S::DecisionContract, vec![contract]),
        ("data:other", S::Data, vec![]),
        ("claim", S::Claim, vec!["transform:joint", "decision:u1"]),
    ];
    let ids = rows.iter().map(|r| r.0).collect();
    let links = rows
        .iter()
        .map(|(id, stage, parents)| CompositionLink {
            id: (*id).to_owned(),
            stage: *stage,
            parents: parents.iter().map(|p| (*p).to_owned()).collect(),
            declared_parent_digests: None,
        })
        .collect();
    (ProvenanceChain::new(links).unwrap(), ids)
}

#[test]
fn a_changed_link_changes_exactly_itself_and_its_descendants() {
    let (before, ids) = chain("contract:c");
    let (after, renamed) = chain("contract:c2");
    let changed: Vec<usize> = (0..ids.len())
        .filter(|&i| before.digest_of(ids[i]).unwrap() != after.digest_of(renamed[i]).unwrap())
        .collect();
    // contract (0), data snapshot (1), provider (2), decision contract (5) and claim (7).
    assert_eq!(changed, vec![0, 1, 2, 5, 7]);
}

/// The lineage of the claim is exactly its ancestors: the transformation, the distribution, the
/// decision contract and the contract behind it. The data snapshot, the provider and the
/// unrelated link are not behind the claim.
#[test]
fn the_lineage_of_the_claim_is_exactly_its_hand_derived_ancestors() {
    let (chain, _) = chain("contract:c");
    let lineage: BTreeSet<&str> =
        chain.lineage("claim").unwrap().iter().map(|l| l.id.as_str()).collect();
    assert_eq!(
        lineage,
        BTreeSet::from([
            "claim",
            "transform:joint",
            "distribution:d1",
            "decision:u1",
            "contract:c"
        ])
    );
    let stages: BTreeSet<S> = chain.stages_behind("claim").unwrap().into_iter().collect();
    assert_eq!(
        stages,
        BTreeSet::from([
            S::Claim,
            S::Transformation,
            S::DistributionArtifact,
            S::DecisionContract,
            S::CausalContract
        ])
    );
}
