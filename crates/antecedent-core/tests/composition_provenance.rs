//! F24 `composition_provenance`: predecessor digests in the provenance chain.

use antecedent_core::{
    CompositionLink, CompositionStage as S, ProvenanceChain, ProvenanceChainError,
};

type Row = (&'static str, S, &'static [&'static str]);

/// contract -> data snapshot -> provider -> distribution -> transformation,
/// and contract -> decision contract; the claim derives from the last of each.
const ROWS: [Row; 7] = [
    ("contract:c", S::CausalContract, &[]),
    ("data:snapshot-1", S::Data, &["contract:c"]),
    ("provider:lab/curve@v1", S::ExternalProvider, &["data:snapshot-1"]),
    ("distribution:d1", S::DistributionArtifact, &["provider:lab/curve@v1"]),
    ("transform:joint", S::Transformation, &["distribution:d1"]),
    ("decision:u1", S::DecisionContract, &["contract:c"]),
    ("claim", S::Claim, &["transform:joint", "decision:u1"]),
];

fn links(rows: &[Row], declared: Option<&[Vec<String>]>) -> Vec<CompositionLink> {
    rows.iter()
        .enumerate()
        .map(|(i, (id, stage, parents))| CompositionLink {
            id: (*id).to_owned(),
            stage: *stage,
            parents: parents.iter().map(|p| (*p).to_owned()).collect(),
            declared_parent_digests: declared.map(|d| d[i].clone()),
        })
        .collect()
}

/// BLAKE3 of the documented preimage, recomputed independently of the crate.
fn expected_digest(id: &str, stage: S, parent_digests: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(id.len() as u64).to_le_bytes());
    hasher.update(id.as_bytes());
    hasher.update(&(stage.as_str().len() as u64).to_le_bytes());
    hasher.update(stage.as_str().as_bytes());
    for parent in parent_digests {
        hasher.update(parent.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn producer() -> ProvenanceChain {
    ProvenanceChain::new(links(&ROWS, None)).unwrap()
}

/// What each link believes its parents' digests are, from the producer.
fn declared_from(chain: &ProvenanceChain) -> Vec<Vec<String>> {
    chain
        .links()
        .iter()
        .map(|l| l.parents.iter().map(|p| chain.digest_of(p).unwrap().to_owned()).collect())
        .collect()
}

#[test]
fn f24_every_score_traces_contract_snapshot_provider_artifact_transform_and_decision_in_order() {
    let chain = producer();
    let lineage = chain.lineage("claim").unwrap();
    let seen: Vec<(&str, S)> = lineage.iter().map(|l| (l.id.as_str(), l.stage)).collect();
    let want: Vec<(&str, S)> = ROWS.iter().map(|(id, stage, _)| (*id, *stage)).collect();
    assert_eq!(seen, want);
    assert_eq!(chain.stages_behind("claim").unwrap().into_iter().collect::<Vec<_>>(), {
        let mut stages: Vec<S> = ROWS.iter().map(|r| r.1).collect();
        stages.sort();
        stages
    });

    let mut digests: Vec<String> = Vec::new();
    for (id, stage, parents) in ROWS {
        let parent_digests: Vec<&str> = parents
            .iter()
            .map(|p| {
                let at = ROWS.iter().position(|r| r.0 == *p).unwrap();
                digests[at].as_str()
            })
            .collect();
        let digest = expected_digest(id, stage, &parent_digests);
        assert_eq!(chain.digest_of(id).unwrap(), digest, "{id}");
        assert_eq!(chain.verify_digest(id, &digest), Ok(()));
        assert_eq!(digest.len(), 64);
        digests.push(digest);
    }
    let distinct: std::collections::BTreeSet<&String> = digests.iter().collect();
    assert_eq!(distinct.len(), digests.len());
}

#[test]
fn f24_missing_or_changed_predecessor_digest_refuses_chain_validation() {
    // A missing predecessor.
    let mut rows = ROWS;
    rows[2] = ("provider:lab/curve@v1", S::ExternalProvider, &["data:missing"]);
    let error = ProvenanceChain::new(links(&rows, None)).unwrap_err();
    assert_eq!(
        error,
        ProvenanceChainError::UnresolvedParent {
            child: "provider:lab/curve@v1".into(),
            parent: "data:missing".into()
        }
    );
    let refusal = error.to_refusal();
    assert_eq!(refusal.code, "external_binding_mismatch");
    assert_eq!(refusal.stage, "provenance");
    assert_eq!(refusal.detail, "composition_provenance.unresolved_parent");
    assert_eq!(refusal.offending.as_deref(), Some("provider:lab/curve@v1"));

    // A changed declared predecessor digest.
    let good = producer();
    let mut declared = declared_from(&good);
    let real = declared[2][0].clone();
    declared[2][0] = "0".repeat(64);
    let error = ProvenanceChain::new(links(&ROWS, Some(&declared))).unwrap_err();
    assert_eq!(
        error,
        ProvenanceChainError::DigestMismatch {
            link: "provider:lab/curve@v1".into(),
            expected: real.clone(),
            supplied: "0".repeat(64)
        }
    );
    let refusal = error.to_refusal();
    assert_eq!(refusal.code, "external_binding_mismatch");
    assert_eq!(refusal.detail, "composition_provenance.digest_mismatch");
    assert_eq!(refusal.offending.as_deref(), Some("provider:lab/curve@v1"));
    assert_eq!(refusal.expected.as_deref(), Some(real.as_str()));
    assert_eq!(refusal.supplied.as_deref(), Some("0".repeat(64).as_str()));

    // A dropped declared digest is a changed predecessor list too.
    let mut short = declared_from(&good);
    short[6].pop();
    assert!(matches!(
        ProvenanceChain::new(links(&ROWS, Some(&short))),
        Err(ProvenanceChainError::DigestMismatch { .. })
    ));
    // The declared digests of the untouched chain are accepted.
    let ok = ProvenanceChain::new(links(&ROWS, Some(&declared_from(&good)))).unwrap();
    assert_eq!(ok.digest_of("claim").unwrap(), good.digest_of("claim").unwrap());

    // Changing an upstream link id changes every downstream digest.
    let mut renamed = ROWS;
    renamed[0] = ("contract:other", S::CausalContract, &[]);
    renamed[1] = ("data:snapshot-1", S::Data, &["contract:other"]);
    renamed[5] = ("decision:u1", S::DecisionContract, &["contract:other"]);
    let changed = ProvenanceChain::new(links(&renamed, None)).unwrap();
    for (id, _, _) in ROWS {
        // A renamed link no longer exists under its old id; every surviving link
        // downstream of the rename has a different digest.
        let Ok(after) = changed.digest_of(id) else {
            continue;
        };
        assert_ne!(good.digest_of(id).unwrap(), after, "{id}");
    }
    assert!(changed.digest_of("contract:c").is_err());
    assert!(changed.digest_of("claim").is_ok());
    // Against the retained claim digest, the renamed chain refuses.
    let retained = good.digest_of("claim").unwrap().to_owned();
    let error = changed.verify_digest("claim", &retained).unwrap_err();
    assert_eq!(error.to_refusal().detail, "composition_provenance.digest_mismatch");
    assert_eq!(
        changed.verify_digest("nope", &retained).unwrap_err().to_refusal().detail,
        "composition_provenance.unknown_link"
    );
    assert_eq!(
        ProvenanceChain::new(links(&ROWS[..1], None))
            .unwrap()
            .require_stages("contract:c", &[S::Claim])
            .unwrap_err()
            .to_refusal()
            .detail,
        "composition_provenance.missing_stage"
    );
}

#[test]
fn f24_fresh_reader_reconstructs_the_lineage_from_digests_not_labels() {
    let producer = producer();
    // Retained independently of the artifact bytes.
    let retained_claim = producer.digest_of("claim").unwrap().to_owned();
    let declared = declared_from(&producer);

    // The reader sees only (id, stage name, parents, declared parent digests).
    let wire: Vec<(String, String, Vec<String>, Vec<String>)> = producer
        .links()
        .iter()
        .zip(&declared)
        .map(|(l, d)| (l.id.clone(), l.stage.as_str().to_owned(), l.parents.clone(), d.clone()))
        .collect();
    let rebuild = |wire: &[(String, String, Vec<String>, Vec<String>)]| {
        ProvenanceChain::new(
            wire.iter()
                .map(|(id, stage, parents, digests)| CompositionLink {
                    id: id.clone(),
                    stage: antecedent_core::CompositionStage::from_name(stage).unwrap(),
                    parents: parents.clone(),
                    declared_parent_digests: Some(digests.clone()),
                })
                .collect(),
        )
    };
    let reader = rebuild(&wire).unwrap();
    assert_eq!(reader.verify_digest("claim", &retained_claim), Ok(()));
    let ids: Vec<&str> = reader.lineage("claim").unwrap().iter().map(|l| l.id.as_str()).collect();
    assert_eq!(ids, ROWS.map(|r| r.0));

    // Relabel a display id everywhere but keep the carried digests: refused.
    let relabel = |from: &str, to: &str| {
        let swap = |s: &String| if s == from { to.to_owned() } else { s.clone() };
        wire.iter()
            .map(|(id, stage, parents, digests)| {
                (swap(id), stage.clone(), parents.iter().map(swap).collect(), digests.clone())
            })
            .collect::<Vec<_>>()
    };
    let relabelled = relabel("provider:lab/curve@v1", "provider:friendly-name");
    let error = rebuild(&relabelled).unwrap_err();
    assert!(matches!(error, ProvenanceChainError::DigestMismatch { .. }));
    assert_eq!(error.to_refusal().detail, "composition_provenance.digest_mismatch");

    // Relabelling and re-declaring the digests still cannot reach the retained claim digest.
    let consistent = ProvenanceChain::new(links(
        &[
            ROWS[0],
            ROWS[1],
            ("provider:friendly-name", S::ExternalProvider, &["data:snapshot-1"]),
            ("distribution:d1", S::DistributionArtifact, &["provider:friendly-name"]),
            ROWS[4],
            ROWS[5],
            ROWS[6],
        ],
        None,
    ))
    .unwrap();
    assert!(matches!(
        consistent.verify_digest("claim", &retained_claim),
        Err(ProvenanceChainError::DigestMismatch { .. })
    ));
}
