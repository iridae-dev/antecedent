//! Portable structural decision results: retention, refusal, replay, fresh reader.

use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_structural::{
    AtomEvidence, AtomStatus, StructuralAtom, StructuralVerdict, evaluate_structural,
};
use antecedent_design::decision_structural_artifact::StructuralResultArtifact;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn quantity(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

/// One structure's exact law: action A and B payoffs, two equally likely rows.
fn law(a: f64, b: f64) -> Box<DistributionArtifact> {
    let columns = [quantity("a", "do(a=1)"), quantity("b", "do(a=0)")];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "structure".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration".into(),
            causal_contract_id: "checked".into(),
        },
    )
    .unwrap();
    Box::new(
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
                legacy_posterior: None,
                legacy_bindings: None,
            },
            vec![a, b, a, b],
        )
        .unwrap(),
    )
}

fn atom(id: &str, p: Option<f64>, evidence: AtomEvidence) -> StructuralAtom {
    StructuralAtom { id: id.into(), probability: p, evidence }
}

fn evaluated(id: &str, p: Option<f64>, a: f64, b: f64) -> StructuralAtom {
    atom(id, p, AtomEvidence::Evaluated(law(a, b)))
}

fn contract(policy: StructuralPolicy) -> DecisionContract {
    let action = |id: &str, variable: &str, regime: &str| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![quantity(variable, regime)],
        utility: UtilityExpr::Input(0),
    };
    DecisionContract {
        actions: vec![action("A", "a", "do(a=1)"), action("B", "b", "do(a=0)")],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: policy,
    }
}

/// The frozen F6 structures: A=(2,-1), B=(0,1), no atom probabilities.
fn frozen_atoms() -> Vec<StructuralAtom> {
    vec![evaluated("graph-1", None, 2.0, 0.0), evaluated("graph-2", None, -1.0, 1.0)]
}

/// Probabilities (0.5, 0.3, 0.2); the last structure is not identified.
fn weighted_atoms() -> Vec<StructuralAtom> {
    vec![
        evaluated("graph-1", Some(0.5), 2.0, 0.0),
        evaluated("graph-2", Some(0.3), -1.0, 1.0),
        atom("graph-3", Some(0.2), AtomEvidence::Unidentified),
    ]
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

fn retained(atoms: &[StructuralAtom]) -> Vec<(String, Option<String>)> {
    StructuralResultArtifact::new(
        evaluate_structural(&contract(StructuralPolicy::Maximin), atoms).unwrap(),
        atoms,
    )
    .atom_digests()
    .to_vec()
}

#[test]
fn f6_artifact_retains_atom_ids_unidentified_mass_and_declared_policy() {
    let c = contract(StructuralPolicy::Maximin);
    let identity = c.identity().unwrap();
    let atoms = frozen_atoms();
    let result = evaluate_structural(&c, &atoms).unwrap();
    assert_eq!(result.verdict, StructuralVerdict::WorstCaseChoice("B".into()));
    let artifact = StructuralResultArtifact::new(result, &atoms);
    let bytes = artifact.to_bytes("structural-result").unwrap();
    let loaded =
        StructuralResultArtifact::from_bytes(&bytes, &identity, artifact.atom_digests()).unwrap();
    assert_eq!(loaded.result(), artifact.result());
    assert_eq!(loaded.result().policy, StructuralPolicy::Maximin);
    let ids: Vec<&str> = loaded.result().atoms.iter().map(|a| a.id.as_str()).collect();
    assert_eq!(ids, ["graph-1", "graph-2"]);
    assert!(loaded.result().atoms.iter().all(|a| a.probability.is_none()));
    assert!(loaded.result().unidentified_mass.is_none());
    assert_eq!(loaded.result().actions[0].range, Some((-1.0, 2.0)));
    assert_eq!(loaded.result().actions[1].range, Some((0.0, 1.0)));
    assert!(loaded.atom_digests().iter().all(|(_, d)| d.is_some()));
    loaded.replay(&c, &atoms).unwrap();

    // With genuine probabilities the unidentified mass is kept, not normalized away.
    let weighted = weighted_atoms();
    let bayes = contract(StructuralPolicy::BayesOverStructures);
    let result = evaluate_structural(&bayes, &weighted).unwrap();
    let artifact = StructuralResultArtifact::new(result, &weighted);
    let bytes = artifact.to_bytes("structural-weighted").unwrap();
    let loaded = StructuralResultArtifact::from_bytes(
        &bytes,
        &bayes.identity().unwrap(),
        artifact.atom_digests(),
    )
    .unwrap();
    assert_eq!(loaded.result(), artifact.result());
    assert!(near(loaded.result().unidentified_mass.unwrap(), 0.2));
    assert!(near(loaded.result().evaluated_mass.unwrap(), 0.8));
    assert!(near(loaded.result().unevaluated_mass.unwrap(), 0.0));
    assert_eq!(loaded.result().atoms[2].status, AtomStatus::Unidentified);
    assert_eq!(loaded.result().atoms[2].probability, Some(0.2));
    assert_eq!(loaded.atom_digests()[2], ("graph-3".to_owned(), None));
    assert_eq!(loaded.result().policy, StructuralPolicy::BayesOverStructures);
    assert!(matches!(loaded.result().verdict, StructuralVerdict::InsufficientScience(_)));
    loaded.replay(&bayes, &weighted).unwrap();

    // An unevaluated reason string survives too.
    let reasoned = vec![
        evaluated("graph-1", None, 2.0, 0.0),
        atom("graph-2", None, AtomEvidence::Unevaluated("budget_exhausted".into())),
    ];
    let result = evaluate_structural(&c, &reasoned).unwrap();
    let artifact = StructuralResultArtifact::new(result, &reasoned);
    let bytes = artifact.to_bytes("structural-reason").unwrap();
    let loaded =
        StructuralResultArtifact::from_bytes(&bytes, &identity, artifact.atom_digests()).unwrap();
    assert_eq!(loaded.result().atoms[1].status, AtomStatus::Unevaluated("budget_exhausted".into()));
    assert!(matches!(loaded.result().verdict, StructuralVerdict::InsufficientScience(_)));
}

#[test]
fn f6_artifact_refuses_other_contract_other_atom_ids_changed_digests_and_tampering() {
    let c = contract(StructuralPolicy::Maximin);
    let identity = c.identity().unwrap();
    let atoms = frozen_atoms();
    let artifact = StructuralResultArtifact::new(evaluate_structural(&c, &atoms).unwrap(), &atoms);
    let bytes = artifact.to_bytes("structural-result").unwrap();
    let digests = artifact.atom_digests().to_vec();
    assert!(StructuralResultArtifact::from_bytes(&bytes, &identity, &digests).is_ok());

    // Another contract identity.
    assert!(
        StructuralResultArtifact::from_bytes(&bytes, "0".repeat(64).as_str(), &digests).is_err()
    );
    let other = contract(StructuralPolicy::ReportOnly).identity().unwrap();
    assert!(StructuralResultArtifact::from_bytes(&bytes, &other, &digests).is_err());

    // A renamed atom id.
    let mut renamed = digests.clone();
    renamed[0].0 = "graph-x".into();
    assert!(StructuralResultArtifact::from_bytes(&bytes, &identity, &renamed).is_err());

    // A changed or dropped draws digest.
    let mut changed = digests.clone();
    changed[1].1 = Some("0".repeat(64));
    assert!(StructuralResultArtifact::from_bytes(&bytes, &identity, &changed).is_err());
    let mut dropped = digests.clone();
    dropped[1].1 = None;
    assert!(StructuralResultArtifact::from_bytes(&bytes, &identity, &dropped).is_err());
    assert!(StructuralResultArtifact::from_bytes(&bytes, &identity, &digests[..1]).is_err());

    // Truncation, corruption and oversize refuse.
    assert!(
        StructuralResultArtifact::from_bytes(&bytes[..bytes.len() - 5], &identity, &digests)
            .is_err()
    );
    let mut corrupt = bytes.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0xff;
    assert!(StructuralResultArtifact::from_bytes(&corrupt, &identity, &digests).is_err());
    let huge = vec![0_u8; 9 * 1024 * 1024];
    assert!(StructuralResultArtifact::from_bytes(&huge, &identity, &digests).is_err());
    assert!(artifact.to_bytes("  ").is_err());
}

#[test]
fn f6_replay_recomputes_and_refuses_a_changed_verdict_or_number() {
    let c = contract(StructuralPolicy::Maximin);
    let atoms = frozen_atoms();
    let result = evaluate_structural(&c, &atoms).unwrap();
    assert_eq!(result.verdict, StructuralVerdict::WorstCaseChoice("B".into()));
    let artifact = StructuralResultArtifact::new(result.clone(), &atoms);
    artifact.replay(&c, &atoms).unwrap();

    // A verdict flipped and resealed under the same identities does not replay.
    let mut flipped = result.clone();
    flipped.verdict = StructuralVerdict::WorstCaseChoice("A".into());
    let flipped = StructuralResultArtifact::new(flipped, &atoms);
    let reloaded = StructuralResultArtifact::from_bytes(
        &flipped.to_bytes("r").unwrap(),
        &c.identity().unwrap(),
        flipped.atom_digests(),
    )
    .unwrap();
    assert!(reloaded.replay(&c, &atoms).is_err());

    // So does an edited number.
    let mut edited = result.clone();
    edited.actions[0].range = Some((-1.0, 3.0));
    assert!(StructuralResultArtifact::new(edited, &atoms).replay(&c, &atoms).is_err());
    let mut mass = result;
    mass.unidentified_mass = Some(0.25);
    assert!(StructuralResultArtifact::new(mass, &atoms).replay(&c, &atoms).is_err());

    // Another contract or other atoms are not the stored decision.
    let other_contract = contract(StructuralPolicy::ReportOnly);
    assert!(artifact.replay(&other_contract, &atoms).is_err());
    let other_atoms =
        vec![evaluated("graph-1", None, 3.0, 0.0), evaluated("graph-2", None, -1.0, 1.0)];
    assert!(artifact.replay(&c, &other_atoms).is_err());
    let renamed = vec![evaluated("graph-1", None, 2.0, 0.0), evaluated("graph-z", None, -1.0, 1.0)];
    assert!(artifact.replay(&c, &renamed).is_err());
    assert!(artifact.replay(&c, &atoms[..1]).is_err());
}

#[test]
fn f6_fresh_process_consumer_rebuilds_atoms_from_constants_and_replays() {
    const PATH_ENV: &str = "ANTECEDENT_23_STRUCTURAL_TEST_DIR";
    if let Ok(dir) = std::env::var(PATH_ENV) {
        let dir = std::path::PathBuf::from(dir);
        // Everything the consumer trusts is built here from constants.
        let consumer_contract = contract(StructuralPolicy::Maximin);
        let identity = consumer_contract.identity().unwrap();
        let consumer_atoms = frozen_atoms();
        let digests = retained(&consumer_atoms);
        let bytes = std::fs::read(dir.join("structural.bin")).unwrap();
        let loaded = StructuralResultArtifact::from_bytes(&bytes, &identity, &digests).unwrap();
        loaded.replay(&consumer_contract, &consumer_atoms).unwrap();
        let result = loaded.result();
        assert_eq!(result.verdict, StructuralVerdict::WorstCaseChoice("B".into()));
        assert_eq!(result.actions[0].range, Some((-1.0, 2.0)));
        assert_eq!(result.actions[1].range, Some((0.0, 1.0)));
        assert_eq!(result.policy, StructuralPolicy::Maximin);
        return;
    }
    let c = contract(StructuralPolicy::Maximin);
    let atoms = frozen_atoms();
    let artifact = StructuralResultArtifact::new(evaluate_structural(&c, &atoms).unwrap(), &atoms);
    let unique =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir =
        std::env::temp_dir().join(format!("antecedent-structural-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("structural.bin"), artifact.to_bytes("s").unwrap()).unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("f6_fresh_process_consumer_rebuilds_atoms_from_constants_and_replays")
        .env(PATH_ENV, &dir)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
