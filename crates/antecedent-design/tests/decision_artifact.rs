//! Independent-process consumption and replay of decision artifacts.

use antecedent_contract::ScientificQuantity;
use antecedent_design::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, source_digest,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{Verdict, evaluate_contract};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::error::IoError;
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

fn quantity(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity { regime_id: regime.into(), ..common::quantity(variable) }
}

/// The message of a typed decision refusal; any other error kind fails the test.
fn refusal_message<T: std::fmt::Debug>(result: Result<T, IoError>) -> String {
    match result.expect_err("a typed refusal") {
        IoError::Refused { code, message } => {
            assert_eq!(code, "decision_contract_unsatisfied");
            message
        }
        other => panic!("not a typed decision refusal: {other:?}"),
    }
}

fn columns() -> Vec<ScientificQuantity> {
    vec![quantity("p", "do(a=1)"), quantity("q", "do(a=1)"), quantity("safe", "do(a=0)")]
}

fn source_identity() -> DistributionIdentity {
    DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns(),
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration-1".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap()
}

fn source() -> DistributionArtifact {
    let p = [1.0, 3.0, 2.0, 0.0];
    let q = [4.0, 0.0, 2.0, 6.0];
    let mut draws = Vec::new();
    for i in 0..4 {
        draws.extend([p[i], q[i], 3.0]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity: source_identity(),
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 3],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

/// The consumer re-declares the contract from constants; it never trusts the
/// producer's bytes for what the decision was.
fn contract() -> DecisionContract {
    let cols = columns();
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "risky".into(),
                kind: ActionKind::Intervention,
                inputs: vec![cols[0].clone(), cols[1].clone()],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            },
            DecisionAction {
                id: "safe".into(),
                kind: ActionKind::Policy,
                inputs: vec![cols[2].clone()],
                utility: UtilityExpr::maximum(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
            },
        ],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![HardConstraint {
            id: "q-cap".into(),
            expr: UtilityExpr::Input(1),
            bound: 5.0,
            min_probability: 0.75,
            units: "units".into(),
            applies_to: vec!["risky".into()],
        }],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

#[test]
fn contract_artifact_round_trips_and_binds_its_identity() {
    let artifact = DecisionContractArtifact::new(contract()).unwrap();
    let bytes = artifact.to_bytes("decision-contract").unwrap();
    let loaded = DecisionContractArtifact::from_bytes(&bytes, artifact.identity()).unwrap();
    assert_eq!(loaded.contract(), &contract());

    // Reordering unordered actions keeps the identity the consumer retained.
    let mut reordered = contract();
    reordered.actions.reverse();
    let reordered = DecisionContractArtifact::new(reordered).unwrap();
    assert_eq!(reordered.identity(), artifact.identity());
    assert!(
        DecisionContractArtifact::from_bytes(
            &reordered.to_bytes("decision-contract").unwrap(),
            artifact.identity()
        )
        .is_ok()
    );

    // Any semantic edit, even resealed with its own identity, is refused.
    let mut edited = contract();
    edited.constraints[0].bound = 6.0;
    let resealed = DecisionContractArtifact::new(edited).unwrap();
    assert!(
        refusal_message(DecisionContractArtifact::from_bytes(
            &resealed.to_bytes("decision-contract").unwrap(),
            artifact.identity()
        ))
        .starts_with("decision_artifact.identity_expected")
    );
    // Truncation and oversize refuse before decoding.
    assert!(matches!(
        DecisionContractArtifact::from_bytes(&bytes[..bytes.len() - 5], artifact.identity())
            .unwrap_err(),
        IoError::Io(_)
    ));
    let huge = vec![0_u8; 9 * 1024 * 1024];
    assert!(matches!(
        DecisionContractArtifact::from_bytes(&huge, artifact.identity()).unwrap_err(),
        IoError::TooLarge
    ));
}

#[test]
fn result_replays_from_its_inputs_and_refuses_any_changed_number() {
    let c = contract();
    let s = source();
    let result = evaluate_contract(&c, &s).unwrap();
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("safe".into()));
    let artifact = DecisionResultArtifact::new(result, &s);
    artifact.replay(&c, &s).unwrap();

    let bytes = artifact.to_bytes("decision-result").unwrap();
    let identity = c.identity().unwrap();
    let digest = source_digest(&s);
    let loaded = DecisionResultArtifact::from_bytes(&bytes, &identity, &digest).unwrap();
    assert_eq!(loaded.result(), artifact.result());
    loaded.replay(&c, &s).unwrap();

    // Another contract or another source is not the stored decision.
    assert!(
        refusal_message(DecisionResultArtifact::from_bytes(
            &bytes,
            "0".repeat(64).as_str(),
            &digest
        ))
        .starts_with("decision_artifact.contract_expected")
    );
    assert!(
        refusal_message(DecisionResultArtifact::from_bytes(
            &bytes,
            &identity,
            "0".repeat(64).as_str()
        ))
        .starts_with("decision_artifact.source_expected")
    );
    let mut other = c.clone();
    other.actions[1].utility = UtilityExpr::Input(0);
    assert!(
        refusal_message(loaded.replay(&other, &s)).starts_with("decision_artifact.replay_contract")
    );

    // A stored number edited and resealed under the same identities does not replay.
    let mut tampered = artifact.result().clone();
    tampered.actions[0].expected_utility += 0.5;
    let tampered = DecisionResultArtifact::new(tampered, &s);
    let reloaded = DecisionResultArtifact::from_bytes(
        &tampered.to_bytes("decision-result").unwrap(),
        &identity,
        &digest,
    )
    .unwrap();
    assert!(
        refusal_message(reloaded.replay(&c, &s)).starts_with("decision_artifact.replay_mismatch")
    );
    let mut flipped = artifact.result().clone();
    flipped.verdict = Verdict::UniquelyOptimal("risky".into());
    assert!(
        refusal_message(DecisionResultArtifact::new(flipped, &s).replay(&c, &s))
            .starts_with("decision_artifact.replay_mismatch")
    );
    // The stored expected utility of the risky action is E[P Q] = (4 + 0 + 4 + 0) / 4.
    assert!((artifact.result().actions[0].expected_utility - 2.0).abs() < 1e-12);
    assert!((artifact.result().actions[1].expected_utility - 3.0).abs() < 1e-12);
}

#[test]
fn fresh_process_consumer_replays_the_decision_without_the_producer() {
    const PATH_ENV: &str = "ANTECEDENT_23_DECISION_TEST_DIR";
    if let Ok(dir) = std::env::var(PATH_ENV) {
        let dir = std::path::PathBuf::from(dir);
        let read = |name: &str| std::fs::read(dir.join(name)).unwrap();
        // Everything the consumer trusts is built here from constants.
        let consumer_contract = contract();
        let identity = consumer_contract.identity().unwrap();
        let contract_artifact =
            DecisionContractArtifact::from_bytes(&read("contract.bin"), &identity).unwrap();
        let loaded_source =
            DistributionArtifact::from_bytes(&read("source.bin"), &source_identity()).unwrap();
        let digest = source_digest(&loaded_source);
        let result =
            DecisionResultArtifact::from_bytes(&read("result.bin"), &identity, &digest).unwrap();
        result.replay(contract_artifact.contract(), &loaded_source).unwrap();
        assert_eq!(result.result().verdict, Verdict::UniquelyOptimal("safe".into()));
        // EVPI: E[max(PQ, 3)] - 3 with PQ = 4, 0, 4, 0.
        assert!((result.result().evpi.unwrap() - 0.5).abs() < 1e-12);
        return;
    }
    let c = contract();
    let s = source();
    let result = DecisionResultArtifact::new(evaluate_contract(&c, &s).unwrap(), &s);
    let unique =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir =
        std::env::temp_dir().join(format!("antecedent-decision-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("contract.bin"),
        DecisionContractArtifact::new(c).unwrap().to_bytes("c").unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("source.bin"), s.to_bytes("s").unwrap()).unwrap();
    std::fs::write(dir.join("result.bin"), result.to_bytes("r").unwrap()).unwrap();
    // The child must select this very test: under a suite crate it is a module member, so its
    // libtest name carries the module path (a bare name would match nothing and "pass").
    let prefix =
        module_path!().split_once("::").map_or(String::new(), |(_, rest)| format!("{rest}::"));
    let test_name =
        format!("{prefix}fresh_process_consumer_replays_the_decision_without_the_producer");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg(&test_name)
        .env(PATH_ENV, &dir)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
        "the child selected no test: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn source_with_p(p: [f64; 4]) -> DistributionArtifact {
    let q = [4.0, 0.0, 2.0, 6.0];
    let mut draws = Vec::new();
    for i in 0..4 {
        draws.extend([p[i], q[i], 3.0]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity: source_identity(),
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 3],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

#[test]
fn result_names_the_exact_stages_and_identities_behind_it() {
    use antecedent_contract::{CompositionStage as S, ProvenanceChainError};
    let c = contract();
    let s = source();
    let artifact = DecisionResultArtifact::new(evaluate_contract(&c, &s).unwrap(), &s);
    let chain = artifact.provenance_chain().unwrap();
    let digest = source_digest(&s);
    let identity = c.identity().unwrap();
    let links: Vec<(String, S, Vec<String>)> =
        chain.links().iter().map(|l| (l.id.clone(), l.stage, l.parents.clone())).collect();
    assert_eq!(
        links,
        vec![
            ("contract:checked-contract".to_owned(), S::CausalContract, vec![]),
            (
                "provider:exact-law#enumeration-1".to_owned(),
                S::ExternalProvider,
                vec!["contract:checked-contract".to_owned()]
            ),
            (
                format!("distribution:{digest}"),
                S::DistributionArtifact,
                vec!["provider:exact-law#enumeration-1".to_owned()]
            ),
            (format!("decision:{identity}"), S::DecisionContract, vec![]),
            (
                "decision_result".to_owned(),
                S::Claim,
                vec![format!("distribution:{digest}"), format!("decision:{identity}")]
            ),
        ]
    );
    assert_eq!(
        chain.require_stages(
            DecisionResultArtifact::CLAIM_LINK_ID,
            &[S::DecisionContract, S::DistributionArtifact, S::ExternalProvider, S::CausalContract]
        ),
        Ok(())
    );
    // No evidence factor stands behind this result.
    assert_eq!(
        chain.require_stages(DecisionResultArtifact::CLAIM_LINK_ID, &[S::Evidence]),
        Err(ProvenanceChainError::MissingStage(S::Evidence))
    );
    // The decision contract does not derive from the data.
    assert_eq!(
        chain.require_stages(&format!("decision:{identity}"), &[S::DistributionArtifact]),
        Err(ProvenanceChainError::MissingStage(S::DistributionArtifact))
    );
}

#[test]
fn a_different_source_digest_is_a_different_distribution_link() {
    let c = contract();
    let first = source();
    let second = source_with_p([2.0, 3.0, 2.0, 0.0]);
    assert_ne!(source_digest(&first), source_digest(&second));
    let ids = |s: &DistributionArtifact| -> Vec<String> {
        DecisionResultArtifact::new(evaluate_contract(&c, s).unwrap(), s)
            .provenance_chain()
            .unwrap()
            .links()
            .iter()
            .map(|l| l.id.clone())
            .collect()
    };
    let (a, b) = (ids(&first), ids(&second));
    assert_ne!(a[2], b[2]);
    assert_eq!(a[2], format!("distribution:{}", source_digest(&first)));
    // Contract, provider and decision links do not depend on the draws.
    assert_eq!((&a[0], &a[1], &a[3]), (&b[0], &b[1], &b[3]));
}

#[test]
fn result_json_carries_the_lineage_and_replay_still_compares_like_with_like() {
    let c = contract();
    let s = source();
    let result = evaluate_contract(&c, &s).unwrap();
    let json: serde_json::Value = serde_json::from_str(
        &antecedent_design::decision_artifact::result_to_json(&result, &s).unwrap(),
    )
    .unwrap();
    let lineage = json["lineage"].as_array().unwrap();
    let stages: Vec<&str> = lineage.iter().map(|l| l["stage"].as_str().unwrap()).collect();
    assert_eq!(
        stages,
        [
            "causal_contract",
            "external_provider",
            "distribution_artifact",
            "decision_contract",
            "claim"
        ]
    );
    assert_eq!(lineage[4]["id"], "decision_result");
    assert_eq!(lineage[4]["parents"].as_array().unwrap().len(), 2);
    // A stored artifact loads and replays; the lineage is re-derived, not trusted.
    let artifact = DecisionResultArtifact::new(result, &s);
    let bytes = artifact.to_bytes("r").unwrap();
    let loaded =
        DecisionResultArtifact::from_bytes(&bytes, &c.identity().unwrap(), &source_digest(&s))
            .unwrap();
    loaded.replay(&c, &s).unwrap();
    assert_eq!(loaded.provenance_chain().unwrap(), artifact.provenance_chain().unwrap());
}

#[test]
fn result_lineage_carries_merkle_digests_that_change_with_the_source() {
    use antecedent_contract::ProvenanceChainError;
    let c = contract();
    let first = source();
    let second = source_with_p([2.0, 3.0, 2.0, 0.0]);
    let lineage_json = |s: &DistributionArtifact| -> Vec<serde_json::Value> {
        let result = evaluate_contract(&c, s).unwrap();
        let json: serde_json::Value = serde_json::from_str(
            &antecedent_design::decision_artifact::result_to_json(&result, s).unwrap(),
        )
        .unwrap();
        json["lineage"].as_array().unwrap().clone()
    };
    let (a, b) = (lineage_json(&first), lineage_json(&second));
    let digest = |l: &serde_json::Value| l["digest"].as_str().unwrap().to_owned();
    assert!(a.iter().all(|l| digest(l).len() == 64));
    // The claim records the digests of its distribution and decision parents.
    assert_eq!(a[4]["parent_digests"], serde_json::json!([digest(&a[2]), digest(&a[3])]));
    assert_eq!(a[0]["parent_digests"], serde_json::json!([]));
    // Upstream links agree; the distribution and everything derived from it differ.
    for shared in [0, 1, 3] {
        assert_eq!(digest(&a[shared]), digest(&b[shared]));
    }
    assert_ne!(digest(&a[2]), digest(&b[2]));
    assert_ne!(digest(&a[4]), digest(&b[4]));

    // The chain's own digests are the ones written to the wire; a reader holding
    // a different retained digest for the result refuses.
    let chain = DecisionResultArtifact::new(evaluate_contract(&c, &first).unwrap(), &first)
        .provenance_chain()
        .unwrap();
    let claim = DecisionResultArtifact::CLAIM_LINK_ID;
    assert_eq!(chain.digest_of(claim).unwrap(), digest(&a[4]));
    assert!(matches!(
        chain.verify_digest(claim, &digest(&b[4])),
        Err(ProvenanceChainError::DigestMismatch { .. })
    ));
}

/// A digest of artifact bytes, to pin them compactly.
fn bytes_digest(bytes: &[u8]) -> String {
    antecedent_core::CanonicalWriter::new("antecedent.test.artifact_bytes")
        .bytes(bytes)
        .finish_hex()
}

#[test]
fn contract_and_result_bytes_are_pinned() {
    use UtilityExpr as U;
    let every_operator = HardConstraint {
        id: "every-operator".into(),
        expr: U::Max(
            Box::new(U::Min(
                Box::new(U::Add(Box::new(U::Input(0)), Box::new(U::Const(-0.25)))),
                Box::new(U::Mul(Box::new(U::Const(3.0)), Box::new(U::Input(1)))),
            )),
            Box::new(U::Sub(Box::new(U::Neg(Box::new(U::Input(1)))), Box::new(U::Const(1.5)))),
        ),
        bound: 2.0,
        min_probability: 0.5,
        units: "units".into(),
        applies_to: vec!["risky".into()],
    };
    let mut found = Vec::new();
    let criteria = [
        DecisionCriterion::PosteriorExpectedUtility,
        DecisionCriterion::PosteriorExpectedLoss,
        DecisionCriterion::ThresholdProbability { threshold: 1.25 },
        DecisionCriterion::Quantile { p: 0.3 },
        DecisionCriterion::MinimaxOverIdentifiedSet,
        DecisionCriterion::MaximinOverStructures,
        DecisionCriterion::Regret,
        DecisionCriterion::ExpectedRegret,
    ];
    let policies = [
        StructuralPolicy::RequireInvariantBestAction,
        StructuralPolicy::Maximin,
        StructuralPolicy::BayesOverStructures,
        StructuralPolicy::ReportOnly,
    ];
    let kinds = [ActionKind::Regime, ActionKind::Study, ActionKind::External];
    for (i, criterion) in criteria.into_iter().enumerate() {
        let mut c = contract();
        c.criterion = criterion;
        c.structural_policy = policies[i % policies.len()];
        c.actions[1].kind = kinds[i % kinds.len()];
        c.constraints.push(every_operator.clone());
        let artifact = DecisionContractArtifact::new(c.clone()).unwrap();
        found.push(bytes_digest(&artifact.to_bytes("pinned").unwrap()));
        found.push(bytes_digest(
            antecedent_design::decision_artifact::contract_to_json(&c).unwrap().as_bytes(),
        ));
        if let Ok(result) = evaluate_contract(&c, &source()) {
            let artifact = DecisionResultArtifact::new(result, &source());
            found.push(bytes_digest(&artifact.to_bytes("pinned").unwrap()));
        }
    }
    assert_eq!(found, PINNED_BYTES);
}

/// Digests of the bytes these cases write; any change to a wire changes them.
const PINNED_BYTES: &[&str] = &[
    "9caa07ed21d7c12f051da502d1a3179a08e293ab12f3478bb2ff835342206ac4",
    "afe81a73463d1c34afd3593dfe3223f8731398ec67d59bc752beb093d37a05b3",
    "808c2e82c70fc401020939f1808a4c2bb8046efad1ede1bb875a1fafad1820aa",
    "2d8e260c48ade442a831cee1bf22467fc0cfada067a807931ac6f6df5b464d7c",
    "b8028c9f40922a85d9d0fe586bff75d3f8b915dda2d31a91b39beaf8ef168640",
    "7776008fdcf0166148cef6914ea709762a17d855131048737015d57f09c942ef",
    "a9ee75b8a7af612b4c4f9727850464dc2445d507efa548d0d2f370a8ed267e19",
    "a224139c71dc0fa82bdc9cd7ddae9e5f5e3dd54562c24e9b4a1df3bfe6152523",
    "45b3e5178b6cf41c23188e6a977701e2e5709d8ea862f17074367ba723a83357",
    "3e678e476f310d700850aa0515bc1f7dd2543d1c519d76508c227ee50913d3d3",
    "1963aa661674f82672bda548f6dfa223bca934db0351306a9f37f9a954d6f3c5",
    "3af1d170fceee48ef06cac8004b1dbde2dafb03c52aa70ad462db8662d67189b",
    "418cd451ee34a800f720f587a530152335847e628170dab3c45274246e2e096e",
    "069728b9d4336a56ca1d731e0988006dd77104f20a280f598b57c65c7e5c6baa",
    "f5b07c09bd0ef2f512d0d6ebef7b503b9c2a3624d387a853ceb17eeb22239259",
    "c179e41e5afa97bff714d5ddd76aead19c4cbad955e4d5e12f0d2cb4876c9dbb",
    "2f4fc1716219fb1a4e53b4eeab0e0db9ae054cca091a16e5870ffbbe9112f105",
    "b13805cbc08e5dfacf46d671bb93131f00031d56ae737e37d87942865e59b87a",
    "d14129be9235ba44637caf9876bc04b107255b81cda6dfad16005984a46f336f",
    "aed3ece2ec46605efcd078d0fc84ab608da92d1aa8a8b01ff07001d1286f7f4d",
    "354df6020340354644696e6acf1ec066281a70ba0639ed01cd2650027a663065",
    "fab5b59b5efc493a9919be643694d9424ecc60f527cb022985beef57f4877f7b",
];
