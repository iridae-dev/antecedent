//! A Bayesian recalculation session declares the standing its posterior statement rests on.
//!
//! The posterior probabilities and quantiles state an uncertainty claim with no calibration
//! record, and the single posterior forms no joint statement with another stream. Both are
//! declared in the stage identities, so they enter the effective identity of the law and the
//! decision and of nothing that reads only data or the fit.
//!
//! Oracles (by construction of the dependency table):
//!
//! - the law and the decision depend on the calibration and coupling inputs; the graph, query,
//!   data, identification and score artifact do not, so their effective identities are those of
//!   the same request with the inputs removed;
//! - a fresh state has no earlier identity, so every declared stage (the standing inputs
//!   included) is recomputed;
//! - against an earlier state that declared the same request, moving only the calibration
//!   record recomputes the law and the decision and reuses the fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::BayesianConfig;
use antecedent::analysis::recalc_bayesian::{
    BayesianModel, BayesianRequest, BayesianSession, PosteriorSummarySpec,
};
use antecedent::analysis::recalc_receipt::{
    RecalcSessionKind, StandingInputDeclaration, UtilitySpec,
};
use antecedent_core::recalc::{
    CalibrationInput, CouplingInput, RecalcCapabilities, RecalcPlan, RetargetSupport, Stage,
    StageStatus,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_prob::{GaussianCoefficientPrior, PriorSet, PriorSpec};

fn request() -> BayesianRequest {
    let mut t = Vec::new();
    let mut y = Vec::new();
    for _ in 0..16 {
        for action in [0.0, 1.0] {
            for noise in [-0.4, 0.4] {
                t.push(action);
                y.push(1.0 + 2.0 * action + noise);
            }
        }
    }
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(2, 10.0)));
    prior.push(PriorSpec::KnownResidualVariance(1.0));
    BayesianRequest {
        columns: vec![("t".into(), t), ("y".into(), y)],
        edges: vec![(0, 1)],
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(1),
        model: BayesianModel::Gaussian,
        inference: BayesianConfig::conjugate().n_draws(1_024).prior(prior),
        summary: PosteriorSummarySpec::default(),
        utility: UtilitySpec { benefit_per_unit: 1.0, cost: 0.0 },
    }
}

fn recomputed(plan: &RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Recomputed { .. }))
}

fn reused(plan: &RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Reused { .. }))
}

#[test]
fn a_fresh_bayesian_plan_declares_calibration_and_coupling_and_moves_only_law_and_decision() {
    let ctx = ExecutionContext::for_tests(41);
    let request = request();
    let ids = request.identities(&ctx);

    // The declared inputs are the typed identities, and no representation is declared: the
    // session consumes no measurement model.
    assert_eq!(ids.own(Stage::Calibration), CalibrationInput::Unmeasured.identity());
    assert_eq!(ids.own(Stage::Coupling), CouplingInput::None.identity());
    assert!(!ids.contains(Stage::Representation));

    // They enter the law and the decision and nothing else.
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

    // A fresh state recomputes every declared stage, the standing inputs among them.
    let plan = BayesianSession::new().plan(&request, &ctx);
    assert!(plan.is_executable());
    for stage in
        [Stage::Calibration, Stage::Coupling, Stage::ScoreArtifact, Stage::Law, Stage::Decision]
    {
        assert!(recomputed(&plan, stage), "{stage}: {:?}", plan.status(stage));
    }
    assert!(plan.status(Stage::Representation).is_none());

    // Against an earlier state of the same request, a moved calibration record recomputes the
    // law and the decision and reuses the fit.
    let mut moved = ids.clone();
    moved.set_calibration(&CalibrationInput::PointOnly);
    let caps = RecalcCapabilities::in_process(RetargetSupport::NotDeclared);
    let unchanged = RecalcPlan::plan(&ids, &ids, &caps);
    assert!(unchanged.recomputed().is_empty(), "{:?}", unchanged.recomputed());
    let plan = RecalcPlan::plan(&ids, &moved, &caps);
    assert!(reused(&plan, Stage::ScoreArtifact));
    assert!(recomputed(&plan, Stage::Law));
    assert!(recomputed(&plan, Stage::Decision));
}

#[test]
fn the_session_table_states_what_each_family_declares_and_why_the_others_do_not() {
    for kind in [RecalcSessionKind::Bayesian, RecalcSessionKind::JointBayesian] {
        let standing = kind.standing();
        assert_eq!(standing.calibration, StandingInputDeclaration::Declared);
        assert_eq!(standing.coupling, StandingInputDeclaration::Declared);
        assert!(matches!(
            standing.representation,
            StandingInputDeclaration::Undeclared { finding, remedy }
                if finding == "recalc.standing_undeclared.representation_version_not_held"
                    && !remedy.is_empty()
        ));
    }
    for kind in [
        RecalcSessionKind::Receipt,
        RecalcSessionKind::Adjusted,
        RecalcSessionKind::Cell,
        RecalcSessionKind::Design,
        RecalcSessionKind::DoublyRobust,
        RecalcSessionKind::Static,
        RecalcSessionKind::Temporal,
    ] {
        let standing = kind.standing();
        assert!(matches!(standing.calibration, StandingInputDeclaration::NotApplicable { .. }));
        assert!(matches!(standing.coupling, StandingInputDeclaration::NotApplicable { .. }));
    }
    for kind in [RecalcSessionKind::External, RecalcSessionKind::Composite] {
        let standing = kind.standing();
        assert!(matches!(standing.calibration, StandingInputDeclaration::Undeclared { .. }));
        assert!(matches!(standing.coupling, StandingInputDeclaration::Undeclared { .. }));
    }
}
