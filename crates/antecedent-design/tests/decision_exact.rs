//! Decision evaluation, admissibility and structural robustness on weighted finite laws whose
//! expected utilities, regrets and ranges are worked out by hand from the rows written here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_design::decision_contract::{
    ActionKind, AdmissibilityRules, AdmissibleDecisionContract, DecisionAction, DecisionContract,
    DecisionCriterion, HardConstraint, StructuralPolicy, UncertaintyKind, UtilityExpr,
};
use antecedent_design::decision_eval::{Verdict, evaluate_contract};
use antecedent_design::decision_robustness::{ClaimProfile, RobustVerdict, evaluate_robust};
use antecedent_design::decision_structural::{AtomEvidence, StructuralAtom};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12
}

fn quantity(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity { regime_id: regime.into(), ..common::quantity(variable) }
}

/// Columns x and y under do(a=1) (action A) and k under do(a=0) (action B).
fn columns() -> Vec<ScientificQuantity> {
    vec![quantity("x", "do(a=1)"), quantity("y", "do(a=1)"), quantity("k", "do(a=0)")]
}

/// A weighted exact joint law; each row is `(x, y, k)`.
fn law(weights: Vec<f64>, rows: &[[f64; 3]]) -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns(),
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration".into(),
            causal_contract_id: "checked".into(),
        },
    )
    .unwrap();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows.len(), 3],
            weights: Some(weights),
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        rows.iter().flatten().copied().collect(),
    )
    .unwrap()
}

/// Action A has utility x*y, action B has utility k.
fn base(policy: StructuralPolicy, constraints: Vec<HardConstraint>) -> DecisionContract {
    let c = columns();
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "A".into(),
                kind: ActionKind::Intervention,
                inputs: vec![c[0].clone(), c[1].clone()],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            },
            DecisionAction {
                id: "B".into(),
                kind: ActionKind::Intervention,
                inputs: vec![c[2].clone()],
                utility: UtilityExpr::Input(0),
            },
        ],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints,
        target_population: "target".into(),
        horizon: 0,
        structural_policy: policy,
    }
}

fn cap_y(bound: f64, min_probability: f64) -> HardConstraint {
    HardConstraint {
        id: "y-cap".into(),
        expr: UtilityExpr::Input(1),
        bound,
        min_probability,
        units: "units".into(),
        applies_to: vec!["A".into()],
    }
}

/// Rows (x, y, k) with weights .1 .2 .3 .4: (1,4,6) (2,3,2) (3,2,4) (4,1,5).
/// U_A = x*y = 4, 6, 6, 4 so E[U_A] = .4 + 1.2 + 1.8 + 1.6 = 5.0.
/// U_B = k so E[U_B] = .6 + .4 + 1.2 + 2.0 = 4.2.
/// The row-wise best is 6, 6, 6, 5 so E[max] = .6 + 1.2 + 1.8 + 2.0 = 5.6 and EVPI = 0.6.
/// Regret of A: .1*2 + .2*0 + .3*0 + .4*1 = 0.6; regret of B: .1*0 + .2*4 + .3*2 + .4*0 = 1.4.
fn source() -> DistributionArtifact {
    law(
        vec![0.1, 0.2, 0.3, 0.4],
        &[[1.0, 4.0, 6.0], [2.0, 3.0, 2.0], [3.0, 2.0, 4.0], [4.0, 1.0, 5.0]],
    )
}

#[test]
fn weighted_nonlinear_expected_utilities_regrets_and_evpi_match_the_hand_values() {
    let result = evaluate_contract(&base(StructuralPolicy::ReportOnly, vec![]), &source()).unwrap();
    assert!(near(result.actions[0].expected_utility, 5.0));
    assert!(near(result.actions[1].expected_utility, 4.2));
    assert!(near(result.actions[0].expected_regret.unwrap(), 0.6));
    assert!(near(result.actions[1].expected_regret.unwrap(), 1.4));
    assert!(near(result.evpi.unwrap(), 0.6));
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("A".into()));
}

/// P(y <= 2) = P(rows 3, 4) = .3 + .4 = 0.7. A constraint demanding 0.6 keeps A admissible; one
/// demanding 0.75 excludes A without any penalty, so B (worth 4.2 < 5.0) is chosen and A keeps
/// its own expected utility 5.0.
#[test]
fn a_probability_constraint_excludes_an_action_exactly_when_its_hand_probability_falls_short() {
    let kept = evaluate_contract(
        &base(StructuralPolicy::ReportOnly, vec![cap_y(2.0, 0.6)]),
        &source(),
    )
    .unwrap();
    assert!(kept.actions[0].admissible);
    assert_eq!(kept.verdict, Verdict::UniquelyOptimal("A".into()));

    let dropped = evaluate_contract(
        &base(StructuralPolicy::ReportOnly, vec![cap_y(2.0, 0.75)]),
        &source(),
    )
    .unwrap();
    assert!(!dropped.actions[0].admissible);
    assert_eq!(dropped.actions[0].exclusions.len(), 1);
    assert!(near(dropped.actions[0].exclusions[0].probability, 0.7));
    assert!(near(dropped.actions[0].exclusions[0].required, 0.75));
    assert!(near(dropped.actions[0].expected_utility, 5.0));
    assert!(near(dropped.actions[1].expected_utility, 4.2));
    assert_eq!(dropped.verdict, Verdict::UniquelyOptimal("B".into()));
}

fn structure(id: &str, weights: Vec<f64>, rows: &[[f64; 3]]) -> StructuralAtom {
    StructuralAtom {
        id: id.into(),
        probability: None,
        evidence: AtomEvidence::Evaluated(Box::new(law(weights, rows))),
    }
}

fn robust(atoms: &[StructuralAtom]) -> antecedent_design::decision_robustness::RobustDecisionResult {
    let contract = AdmissibleDecisionContract {
        contract: base(StructuralPolicy::RequireInvariantBestAction, vec![]),
        rules: AdmissibilityRules::default(),
    };
    let claims = ClaimProfile {
        uncertainty: UncertaintyKind::StructuralEnvelope,
        support: Default::default(),
    };
    evaluate_robust(&contract, atoms, &claims).unwrap().1
}

/// Structure s1: rows (1,2,3) (3,2,1) (2,4,2) with weights .25 .25 .5 gives
/// E[U_A] = .25*2 + .25*6 + .5*8 = 6.0 and E[U_B] = .75 + .25 + 1.0 = 2.0.
/// Structure s2: rows (1,1,2) (3,3,3) with weights .5 .5 gives E[U_A] = .5 + 4.5 = 5.0 and
/// E[U_B] = 1.0 + 1.5 = 2.5. A leads in both, with ranges [5, 6] and [2, 2.5].
fn s1() -> StructuralAtom {
    structure("s1", vec![0.25, 0.25, 0.5], &[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 4.0, 2.0]])
}
fn s2() -> StructuralAtom {
    structure("s2", vec![0.5, 0.5], &[[1.0, 1.0, 2.0], [3.0, 3.0, 3.0]])
}

#[test]
fn a_leader_in_every_structure_is_robust_with_ranges_computed_from_the_laws() {
    let result = robust(&[s1(), s2()]);
    assert_eq!(result.verdict, RobustVerdict::StructurallyRobust("A".into()));
    let (a_lo, a_hi) = result.actions[0].range.unwrap();
    let (b_lo, b_hi) = result.actions[1].range.unwrap();
    assert!(near(a_lo, 5.0) && near(a_hi, 6.0));
    assert!(near(b_lo, 2.0) && near(b_hi, 2.5));
}

/// Structure s3: rows (1,1,10) (1,1,8) with weights .5 .5 gives E[U_A] = 1 and E[U_B] = 9, so B
/// leads there and the choice depends on the structure; no action is selected.
#[test]
fn a_structure_where_another_action_leads_makes_the_choice_graph_dependent() {
    let s3 = structure("s3", vec![0.5, 0.5], &[[1.0, 1.0, 10.0], [1.0, 1.0, 8.0]]);
    let result = robust(&[s1(), s2(), s3]);
    assert_eq!(
        result.verdict,
        RobustVerdict::GraphDependentChoice(vec![
            ("s1".to_owned(), vec!["A".to_owned()]),
            ("s2".to_owned(), vec!["A".to_owned()]),
            ("s3".to_owned(), vec!["B".to_owned()]),
        ])
    );
    assert_eq!(result.verdict.selected_action(), None);
}
