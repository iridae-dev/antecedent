//! Independent finite source-state, terminal decision and ranking acceptance.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::analysis::design_ranking::{DesignRankingRequestWire, evaluate};
use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_design::composition_bundle::{
    BundleLimits, BundleNode, CompositionBundle, NodeKind, SuppliedSources,
};
use antecedent_design::composition_verifiers::{describe_artifact, standard_consumer};
use antecedent_design::decision_artifact::source_digest;
use antecedent_design::rollout_artifact::{
    RolloutArtifact, RolloutExpectation, RolloutSource, StateInterpretation, consume,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;
use serde_json::json;

fn pin(key: &str) -> serde_json::Value {
    let value: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/composition/rollout/expected.json"
    ))
    .unwrap();
    value[key].clone()
}
fn quantity(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "dimensionless".into(),
        population_id: "target".into(),
        regime_id: "do(t=1)".into(),
        horizon: 0,
        functional_id: "state".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}
fn source(posterior: bool) -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        if posterior {
            DistributionMeaningWire::CausalFunctionalPosterior
        } else {
            DistributionMeaningWire::InterventionalPredictive
        },
        &[quantity("theta"), quantity("other")],
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "finite-known-law".into(),
            provider_id: "declared-model".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "law-1".into(),
            causal_contract_id: "checked-source".into(),
        },
    )
    .unwrap();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [2, 2],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Unmeasured,
            trust: if posterior {
                DistributionTrust::NativeLicensed
            } else {
                DistributionTrust::ExternalAttested
            },
            legacy_posterior: None,
            legacy_bindings: None,
        },
        vec![0.25, 1., 0.75, 2.],
    )
    .unwrap()
}
fn fixture(posterior: bool) -> RolloutArtifact {
    let source = source(posterior);
    let digest = source_digest(&source);
    let q = source.quantities()[0].clone();
    let request: DesignRankingRequestWire = serde_json::from_value(json!({
        "decision": {"contract_identity":"bet-terminal", "utility_unit":"utility", "action_ids":["wait","bet"],
            "intercepts":[0.,-0.5],"slopes":[0.,1.],"prior":{"kind":"draws","states":pin("state_draws")}},
        "candidates":[{"semantic_id":"sample", "sample_size":2,"cost":{"amount":0.,"unit":"utility"},
            "signal":{"prior_id":"law-1","state_quantity":q,"observation_quantity":q,"rng_seed":3,
                "evidence_lineage":["law-1"],"conditional_independence":"iid_given_state"}, "provider":{"kind":"binomial"}}],
        "rng_seed":3,"source_digests":[digest]
    })).unwrap();
    let ranked = evaluate(&request).unwrap();
    let wire = ranked.artifact();
    let expected = RolloutExpectation {
        source: RolloutSource {
            identity: source.metadata().identity.clone(),
            trust: source.metadata().trust,
            calibration: source.metadata().calibration,
        },
        state: q,
        interpretation: if posterior {
            StateInterpretation::PosteriorState
        } else {
            StateInterpretation::InterventionalState
        },
        source_digest: digest,
        decision: wire.decision.clone(),
        ranking_identity: wire.digest.clone(),
    };
    RolloutArtifact::new(source.to_bytes("law").unwrap(), ranked.export("rank").unwrap(), expected)
        .unwrap()
}
#[test]
fn rollout_finite_source_and_ranking_replay_match_independent_bet_truth() {
    for posterior in [false, true] {
        let artifact = fixture(posterior);
        let bytes = artifact.to_bytes("rollout").unwrap();
        let decoded = consume(&bytes, &artifact.binding).unwrap();
        assert_eq!(decoded.digest, artifact.digest);
        let ranking = antecedent_design::design_ranking_artifact::consume(
            &decoded.ranking_artifact,
            &antecedent_design::design_ranking_artifact::ConsumeExpectation::default(),
        )
        .unwrap();
        assert!(
            (antecedent_design::design_ranking_artifact::DesignRankingArtifactWire::from_bytes(
                &decoded.ranking_artifact
            )
            .unwrap()
            .prior_expected_utility
                - pin("prior_expected_utility").as_f64().unwrap())
            .abs()
                < 1e-12
        );
        assert!((ranking.ranking.entries[0].evsi - pin("evsi").as_f64().unwrap()).abs() < 1e-12);
        assert!((ranking.ranking.entries[0].evpi - pin("evpi").as_f64().unwrap()).abs() < 1e-12);
        assert_eq!(decoded.binding.source.trust, artifact.binding.source.trust);
        assert_eq!(decoded.binding.source.calibration, DistributionCalibration::Unmeasured);
    }
}
#[test]
fn rollout_resealed_prior_terminal_coordinate_and_standing_mutations_refuse() {
    let artifact = fixture(false);
    for edit in 0..8 {
        let mut changed = artifact.clone();
        match edit {
            0 => changed.binding.decision.action_ids[0] = "other-action".into(),
            1 => changed.binding.decision.admissible[0] = false,
            2 => changed.binding.decision.utility_unit = "other-unit".into(),
            3 => changed.binding.state.units = "other-units".into(),
            4 => changed.binding.source.trust = DistributionTrust::NativeLicensed,
            5 => changed.binding.source.calibration = DistributionCalibration::Measured,
            6 => {
                if let antecedent_design::design_ranking_artifact::PriorWire::Draws { states } =
                    &mut changed.binding.decision.prior
                {
                    states.reverse();
                }
            }
            _ => {
                if let antecedent_design::design_ranking_artifact::UtilityWire::Table { rows } =
                    &mut changed.binding.decision.utility
                {
                    rows[1][0] = 999.;
                }
            }
        }
        changed.seal().unwrap();
        assert!(changed.verify(&artifact.binding).is_err(), "expected boundary edit {edit}");
        assert!(changed.verify(&changed.binding).is_err(), "scientific boundary edit {edit}");
    }
}
#[test]
fn rollout_changed_source_values_and_draw_order_cannot_retain_original_prior() {
    let artifact = fixture(false);
    let old = source(false);
    for draws in [vec![0.75, 2., 0.25, 1.], vec![0.25, 999., 0.75, 2.], vec![0.5, 1., 0.75, 2.]] {
        let replacement = DistributionArtifact::new(old.metadata().clone(), draws).unwrap();
        let mut changed = artifact.clone();
        changed.source_artifact = replacement.to_bytes("law").unwrap();
        changed.seal().unwrap();
        assert!(changed.verify(&artifact.binding).is_err());
        changed.binding.source_digest = source_digest(&replacement);
        changed.seal().unwrap();
        assert!(changed.verify(&changed.binding).is_err());
    }
}
#[test]
fn rollout_meaning_support_weighting_and_marginal_pairing_refuse() {
    let original = fixture(false);
    let old = source(false);
    for edit in 0..5 {
        let mut metadata = old.metadata().clone();
        match edit {
            0 => metadata.weights = Some(vec![0.5, 0.5]),
            1 => metadata.supported = Some(vec![false, true]),
            2 => metadata.identity.alignment = DrawAlignment::IndependentMarginals,
            3 => metadata.identity.semantic = DistributionMeaningWire::EmpiricalOutcome,
            _ => metadata.identity.quantities[0].functional_id = "outcome".into(),
        }
        let source = DistributionArtifact::new(metadata, old.draws().to_vec()).unwrap();
        let mut changed = original.clone();
        changed.source_artifact = source.to_bytes("changed-law").unwrap();
        changed.binding.source.identity = source.metadata().identity.clone();
        changed.binding.state = source.quantities()[0].clone();
        changed.binding.source_digest = source_digest(&source);
        changed.seal().unwrap();
        assert!(changed.verify(&changed.binding).is_err(), "edit {edit}");
    }
}
fn bundle(
    artifact: &RolloutArtifact,
    source_bytes: Vec<u8>,
    rank_bytes: Vec<u8>,
) -> CompositionBundle {
    let law = describe_artifact(NodeKind::Distribution, &source_bytes).unwrap();
    let rollout_bytes = artifact.to_bytes("rollout").unwrap();
    let rank = describe_artifact(NodeKind::StudyRanking, &rank_bytes).unwrap();
    CompositionBundle::new(
        vec![
            BundleNode::embedded("law", NodeKind::Distribution, &law.identity, source_bytes),
            BundleNode::embedded("rollout", NodeKind::Rollout, &artifact.digest, rollout_bytes),
            BundleNode::embedded("ranking", NodeKind::StudyRanking, &rank.identity, rank_bytes),
        ],
        &[
            ("law".into(), "rollout".into()),
            ("law".into(), "ranking".into()),
            ("rollout".into(), "ranking".into()),
        ],
        &BundleLimits::default(),
    )
    .unwrap()
}
#[test]
fn rollout_bundle_verifies_source_rollout_ranking_chain_and_refuses_swapped_metadata() {
    let artifact = fixture(false);
    let good =
        bundle(&artifact, artifact.source_artifact.clone(), artifact.ranking_artifact.clone());
    let bytes = good.to_bytes("bundle", &BundleLimits::default()).unwrap();
    let consumed = standard_consumer()
        .consume(&bytes, &BundleLimits::default(), good.identity(), &SuppliedSources::default())
        .unwrap();
    assert!(consumed.all_verified());
    let old = source(false);
    let mut metadata = old.metadata().clone();
    metadata.identity.provider_id = "another-provider".into();
    let swapped = DistributionArtifact::new(metadata, old.draws().to_vec()).unwrap();
    let bad =
        bundle(&artifact, swapped.to_bytes("swapped").unwrap(), artifact.ranking_artifact.clone());
    let bytes = bad.to_bytes("bundle", &BundleLimits::default()).unwrap();
    let consumed = standard_consumer()
        .consume(&bytes, &BundleLimits::default(), bad.identity(), &SuppliedSources::default())
        .unwrap();
    assert!(!consumed.all_verified());
}
#[test]
fn rollout_truncation_corruption_and_nonfinite_prior_refuse() {
    let artifact = fixture(false);
    let bytes = artifact.to_bytes("rollout").unwrap();
    assert!(consume(&bytes[..bytes.len() / 2], &artifact.binding).is_err());
    let mut changed = artifact.clone();
    if let antecedent_design::design_ranking_artifact::PriorWire::Draws { states } =
        &mut changed.binding.decision.prior
    {
        states[0] = f64::NAN;
    }
    changed.seal().unwrap();
    assert!(changed.verify(&changed.binding).is_err());
}
