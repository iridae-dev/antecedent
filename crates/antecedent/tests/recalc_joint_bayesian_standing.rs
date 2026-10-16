//! A joint Bayesian recalculation session declares calibration and the alignment of its draws.
//!
//! Oracles (by construction of the dependency table):
//!
//! - the calibration record is declared unmeasured and the draws are rows of one aligned joint
//!   posterior, keyed by the source units, the draw count and the seed;
//! - both enter the law and the decision only; the fit and everything that reads only data keep
//!   the identities of the same request with the inputs removed;
//! - a fresh state recomputes every declared stage, and a changed draw count or seed is a
//!   different alignment input.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "calibration-internal")]
use antecedent::analysis::recalc_bayesian::PosteriorSummarySpec;
use antecedent::analysis::recalc_joint_bayesian::{JointBayesianRequest, JointBayesianSession};
use antecedent::analysis::recalc_receipt::UtilitySpec;
use antecedent_core::recalc::{
    CalibrationInput, RecalcCapabilities, RecalcPlan, RetargetSupport, Stage, StageStatus,
};
use antecedent_core::{
    ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JointPriors, JointTransportModel, JointTransportOptions,
    PriorProvenance, SourceData, SourceDependence, SourceSharing, TargetData, TransportGraphClass,
    VaryingBlock,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};

fn request() -> JointBayesianRequest {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(vec![0.0, 1.0].into()),
            ),
        }),
        "source",
        "target",
        vec![VariableId::from_raw(0)],
    );
    let sources = ["a", "b"]
        .into_iter()
        .map(|id| {
            let treatment = (0..128).map(|row| row % 2 == 1).collect::<Vec<_>>();
            let outcome = treatment.iter().map(|t| if *t { 3.0 } else { 1.0 }).collect();
            SourceData {
                id: id.into(),
                identity: DataIdentity {
                    snapshot_digest: format!("snapshot-{id}"),
                    datum_ids: (0..128).map(|row| format!("{id}-{row}")).collect(),
                },
                treatment,
                outcome,
                covariates: vec![],
                noise_variance: 1.0,
            }
        })
        .collect();
    JointBayesianRequest {
        diagram: SelectionDiagram::try_new(graph, vec![]).unwrap(),
        query,
        model: JointTransportModel {
            graph: TransportGraphClass::FixedDag,
            features: vec![],
            varying: VaryingBlock::Intercept,
            sharing: SourceSharing::SharedVaryingBlock,
            dependence: SourceDependence::IndependentSamples,
            priors: JointPriors {
                invariant: GaussianPrior::isotropic(1, 0.0, 100.0, PriorProvenance::Declared),
                varying: GaussianPrior::isotropic(1, 0.0, 4.0, PriorProvenance::Declared),
            },
            max_unsupported_mass: 0.0,
            conflict_z_threshold: 3.0,
        },
        sources,
        target: TargetData {
            identity: DataIdentity {
                snapshot_digest: "target".into(),
                datum_ids: (0..40).map(|row| format!("target-{row}")).collect(),
            },
            rows: 40,
            covariates: vec![],
        },
        options: JointTransportOptions { draws: 1024, seed: 19 },
        summary: PosteriorSummarySpec::default(),
        utility: UtilitySpec { benefit_per_unit: 1.0, cost: 0.0 },
    }
}

fn recomputed(plan: &RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Recomputed { .. }))
}

#[test]
fn a_fresh_joint_plan_declares_calibration_and_alignment_and_moves_only_law_and_decision() {
    let request = request();
    let ids = request.identities();
    assert_eq!(ids.own(Stage::Calibration), CalibrationInput::Unmeasured.identity());
    assert!(ids.contains(Stage::Coupling));
    assert!(!ids.contains(Stage::Representation));

    let mut bare = ids.clone();
    bare.remove(Stage::Calibration);
    bare.remove(Stage::Coupling);
    let with = ids.effective(RetargetSupport::NotDeclared);
    let without = bare.effective(RetargetSupport::NotDeclared);
    for stage in [
        Stage::Graph,
        Stage::Query,
        Stage::DataSnapshot,
        Stage::Identification,
        Stage::ScoreArtifact,
    ] {
        assert_eq!(with[&stage], without[&stage], "{stage} reads neither standing input");
    }
    for stage in [Stage::Law, Stage::Decision] {
        assert_ne!(with[&stage], without[&stage], "{stage} reads the standing inputs");
    }

    // A fresh state recomputes every declared stage.
    let caps = RecalcCapabilities::in_process(RetargetSupport::NotDeclared);
    let fresh = RecalcPlan::plan(JointBayesianSession::new().identities(), &ids, &caps);
    for stage in
        [Stage::Calibration, Stage::Coupling, Stage::ScoreArtifact, Stage::Law, Stage::Decision]
    {
        assert!(recomputed(&fresh, stage), "{stage}: {:?}", fresh.status(stage));
    }

    // The alignment is the draw count, the seed and the source units: a different draw count is
    // a different coupling input.
    let mut other = request.clone();
    other.options.draws = 2048;
    assert_ne!(ids.own(Stage::Coupling), other.identities().own(Stage::Coupling));
    let mut reseeded = request.clone();
    reseeded.options.seed = 20;
    assert_ne!(ids.own(Stage::Coupling), reseeded.identities().own(Stage::Coupling));
    assert_eq!(ids.own(Stage::Coupling), request.identities().own(Stage::Coupling));
}
