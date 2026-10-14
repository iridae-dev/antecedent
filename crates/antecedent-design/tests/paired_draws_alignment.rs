//! Row pairing of separate artifacts needs one alignment identity.
//!
//! Two joint laws sampled from different studies are each aligned within
//! themselves, but row `r` of one has no relation to row `r` of the other. The
//! pairing identity covers source, provider, rng, snapshot, contract and draw
//! count, so a declared-independent pair of different studies is refused as an
//! alignment mismatch rather than paired by index.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{ExternalTrustState, QuantityRole, ScientificQuantity};
use antecedent_design::composition_boundary::{
    BoundaryError, DecisionInput, EvidenceRelation, PairRelation, TrustEvidence, TrustRequirement,
    check_paired_draws,
};
use antecedent_design::decision_eval::DecisionEvalError;
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

fn quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: "do(a=0)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn joint(provider: &str, snapshot: &str) -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &[quantity()],
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: format!("study-{snapshot}"),
            provider_id: provider.into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: snapshot.into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 1],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap()
}

fn input(id: &str, provider: &str, snapshot: &str) -> DecisionInput {
    let evidence =
        TrustEvidence::External(ExternalTrustState::ExternallyAttested { attestor: "lab".into() });
    DecisionInput::from_distribution_artifact(
        id,
        &joint(provider, snapshot),
        &evidence,
        TrustRequirement::Unrestricted,
    )
    .unwrap()
}

#[test]
fn independent_separate_studies_are_not_paired_by_index() {
    let pair = [input("left", "engine", "snap-a"), input("right", "lab", "snap-b")];
    let relation = PairRelation {
        left: "left".into(),
        right: "right".into(),
        relation: EvidenceRelation::IndependentSources,
        route: None,
    };
    let error = check_paired_draws(&pair, &[relation]).unwrap_err();
    assert_eq!(
        error,
        BoundaryError::Eval(DecisionEvalError::JointLawRequired {
            action: None,
            supplied_alignment: "alignment_mismatch",
        })
    );
    assert_eq!(error.reason_code(), "joint_law_required");
}
