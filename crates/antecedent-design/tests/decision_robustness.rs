//! Admissibility rules and robustness states for decisions under structural
//! uncertainty. Expected values are enumerated by hand in each test.

use antecedent_core::{QuantityRole, ScientificQuantity, SupportStatus};
use antecedent_design::decision_contract::{
    ActionKind, AdmissibilityError, AdmissibilityRules, AdmissibleDecisionContract, DecisionAction,
    DecisionContract, DecisionCriterion, DeclaredExclusion, HardConstraint, StructuralPolicy,
    SupportRule, UncertaintyKind, UncertaintyRequirement, UtilityExpr,
};
use antecedent_design::decision_robustness::{
    AtomSupport, ClaimProfile, InputSupport, RobustDecisionResult, RobustVerdict, RobustnessError,
    SupportShortfall, assess_robustness, evaluate_robust,
};
use antecedent_design::decision_structural::{
    AtomEvidence, StructuralAtom, StructuralVerdict, evaluate_structural,
};
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

/// One structure's exact law: action A and B payoffs on two equally likely rows.
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

fn base(
    criterion: DecisionCriterion,
    policy: StructuralPolicy,
    constraints: Vec<HardConstraint>,
) -> DecisionContract {
    let action = |id: &str, variable: &str, regime: &str| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![quantity(variable, regime)],
        utility: UtilityExpr::Input(0),
    };
    DecisionContract {
        actions: vec![action("A", "a", "do(a=1)"), action("B", "b", "do(a=0)")],
        utility_units: "units".into(),
        criterion,
        constraints,
        target_population: "target".into(),
        horizon: 0,
        structural_policy: policy,
    }
}

fn contract(policy: StructuralPolicy, rules: AdmissibilityRules) -> AdmissibleDecisionContract {
    AdmissibleDecisionContract {
        contract: base(DecisionCriterion::PosteriorExpectedUtility, policy, vec![]),
        rules,
    }
}

fn require_supported() -> AdmissibilityRules {
    AdmissibilityRules {
        default_weakest_support: Some(SupportStatus::Supported),
        ..AdmissibilityRules::default()
    }
}

fn profile(entries: Vec<(&str, AtomSupport)>) -> ClaimProfile {
    ClaimProfile {
        uncertainty: UncertaintyKind::StructuralEnvelope,
        support: entries.into_iter().map(|(id, s)| (id.to_owned(), s)).collect(),
    }
}

fn run(
    c: &AdmissibleDecisionContract,
    atoms: &[StructuralAtom],
    claims: &ClaimProfile,
) -> RobustDecisionResult {
    evaluate_robust(c, atoms, claims).unwrap().1
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

fn leaders(pairs: &[(&str, Vec<&str>)]) -> Vec<(String, Vec<String>)> {
    pairs
        .iter()
        .map(|(id, ids)| ((*id).to_owned(), ids.iter().map(|s| (*s).to_owned()).collect()))
        .collect()
}

#[test]
fn f6_robustness_invariant_best_is_structurally_robust() {
    let c = contract(StructuralPolicy::RequireInvariantBestAction, require_supported());
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 6.0, 2.0)];
    let claims = profile(vec![("s1", AtomSupport::supported()), ("s2", AtomSupport::supported())]);
    let result = run(&c, &atoms, &claims);
    assert_eq!(result.verdict, RobustVerdict::StructurallyRobust("A".into()));
    assert!(result.shortfalls.is_empty());
    assert!(result.unsupported_atoms.is_empty());
    assert_eq!(result.verdict.selected_action(), Some("A"));
    // The ranges are the hand-set payoffs: A in {5, 6}, B in {2, 3}.
    assert_eq!(result.actions[0].range, Some((5.0, 6.0)));
    assert_eq!(result.actions[1].range, Some((2.0, 3.0)));
    assert_eq!(result.contract_identity, c.identity().unwrap());
    assert_eq!(result.base_contract_identity, c.contract.identity().unwrap());
}

#[test]
fn f6_robustness_graph_dependent_choice_lists_each_structures_leader() {
    let c = contract(StructuralPolicy::RequireInvariantBestAction, AdmissibilityRules::default());
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 1.0, 4.0)];
    let claims = profile(vec![]);
    let (structural, robust) = evaluate_robust(&c, &atoms, &claims).unwrap();
    assert!(matches!(structural.verdict, StructuralVerdict::NoInvariantBest(_)));
    assert_eq!(
        robust.verdict,
        RobustVerdict::GraphDependentChoice(leaders(&[("s1", vec!["A"]), ("s2", vec!["B"])]))
    );
    assert_eq!(robust.verdict.selected_action(), None);
}

#[test]
fn f6_robustness_support_robust_when_the_unsupported_structure_agrees() {
    let c = contract(StructuralPolicy::RequireInvariantBestAction, require_supported());
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 6.0, 2.0)];
    let claims = profile(vec![
        ("s1", AtomSupport::supported()),
        ("s2", AtomSupport::uniform(SupportStatus::Extrapolative)),
    ]);
    let result = run(&c, &atoms, &claims);
    // s2 fails the rule for both actions, so it is set aside, and A still wins
    // with and without it.
    assert_eq!(result.verdict, RobustVerdict::SupportRobust("A".into()));
    assert_eq!(result.unsupported_atoms, vec!["s2".to_owned()]);
    assert_eq!(result.shortfalls.len(), 2);
    assert_eq!(
        result.shortfalls[0],
        SupportShortfall {
            atom: "s2".into(),
            action: "A".into(),
            input: 0,
            status: SupportStatus::Extrapolative,
            weakest_allowed: SupportStatus::Supported,
        }
    );
    assert_eq!(result.actions[0].supported_range, Some((5.0, 5.0)));
    assert_eq!(result.actions[0].unsupported_in, vec!["s2".to_owned()]);
}

#[test]
fn f6_robustness_support_dependent_when_ignoring_support_changes_the_choice() {
    let c = contract(StructuralPolicy::RequireInvariantBestAction, require_supported());
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 1.0, 4.0)];
    let claims = profile(vec![
        ("s1", AtomSupport::supported()),
        ("s2", AtomSupport::uniform(SupportStatus::OutsideEmpiricalSupport)),
    ]);
    let result = run(&c, &atoms, &claims);
    assert_eq!(
        result.verdict,
        RobustVerdict::SupportDependent { supported_choice: "A".into(), unrestricted_choice: None }
    );
    // The set-aside structure keeps its identity and the mass it would carry.
    assert_eq!(result.unsupported_atoms, vec!["s2".to_owned()]);
    assert_eq!(result.unsupported_mass, None);
}

#[test]
fn f6_robustness_per_input_support_removes_one_action_in_one_structure() {
    let c = contract(StructuralPolicy::RequireInvariantBestAction, require_supported());
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 1.0, 4.0)];
    let s2 = AtomSupport {
        overall: SupportStatus::Supported,
        per_input: vec![InputSupport {
            action_id: "B".into(),
            input: 0,
            status: SupportStatus::Extrapolative,
        }],
    };
    let claims = profile(vec![("s1", AtomSupport::supported()), ("s2", s2)]);
    let result = run(&c, &atoms, &claims);
    // In s2 only A is licensed, so A leads there; B's raw lead in s2 is not used.
    assert_eq!(
        result.verdict,
        RobustVerdict::SupportDependent { supported_choice: "A".into(), unrestricted_choice: None }
    );
    assert_eq!(result.actions[1].unsupported_in, vec!["s2".to_owned()]);
    assert!(result.actions[0].unsupported_in.is_empty());
    assert_eq!(result.actions[1].range, Some((3.0, 4.0)));
    assert_eq!(result.actions[1].supported_range, Some((3.0, 3.0)));
    assert!(result.unsupported_atoms.is_empty());
}

#[test]
fn f6_robustness_unsupported_extrapolation_when_no_structure_supports_an_action() {
    let c = contract(StructuralPolicy::RequireInvariantBestAction, require_supported());
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 6.0, 2.0)];
    let claims = profile(vec![
        ("s1", AtomSupport::uniform(SupportStatus::Extrapolative)),
        ("s2", AtomSupport::uniform(SupportStatus::OutsideEmpiricalSupport)),
    ]);
    let result = run(&c, &atoms, &claims);
    assert_eq!(result.verdict, RobustVerdict::UnsupportedExtrapolation);
    assert_eq!(result.unsupported_atoms, vec!["s1".to_owned(), "s2".to_owned()]);
    // Unassessed support counts as missing evidence once a rule needs it.
    let unassessed = run(&c, &atoms, &profile(vec![]));
    assert_eq!(unassessed.verdict, RobustVerdict::UnsupportedExtrapolation);
    assert!(unassessed.shortfalls.iter().all(|s| s.status == SupportStatus::MissingEvidence));
}

#[test]
fn f6_robustness_insufficient_claims_when_a_structure_is_unresolved() {
    let c = contract(StructuralPolicy::RequireInvariantBestAction, AdmissibilityRules::default());
    for unresolved in [AtomEvidence::Unidentified, AtomEvidence::Unevaluated("budget".into())] {
        let atoms = [evaluated("s1", None, 5.0, 3.0), atom("s2", None, unresolved)];
        let result = run(&c, &atoms, &profile(vec![]));
        assert!(
            matches!(result.verdict, RobustVerdict::InsufficientClaims(_)),
            "{:?}",
            result.verdict
        );
    }
    // No structure evaluated at all.
    let none = [atom("s1", None, AtomEvidence::Unidentified)];
    let result = run(&c, &none, &profile(vec![]));
    assert!(matches!(result.verdict, RobustVerdict::InsufficientClaims(_)));
}

#[test]
fn f6_robustness_no_admissible_action_from_declared_exclusions_and_hard_constraints() {
    let both = AdmissibilityRules {
        declared_exclusions: vec![
            DeclaredExclusion { action_id: "A".into(), reason: "legal".into() },
            DeclaredExclusion { action_id: "B".into(), reason: "ethics".into() },
        ],
        ..AdmissibilityRules::default()
    };
    let c = contract(StructuralPolicy::RequireInvariantBestAction, both);
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 6.0, 2.0)];
    let result = run(&c, &atoms, &profile(vec![]));
    assert_eq!(result.verdict, RobustVerdict::NoAdmissibleAction);

    // A hard constraint that every action violates in every structure.
    let impossible = HardConstraint {
        id: "floor".into(),
        expr: UtilityExpr::Input(0),
        bound: -100.0,
        min_probability: 1.0,
        units: "units".into(),
        applies_to: vec![],
    };
    let hard = AdmissibleDecisionContract {
        contract: base(
            DecisionCriterion::PosteriorExpectedUtility,
            StructuralPolicy::RequireInvariantBestAction,
            vec![impossible],
        ),
        rules: AdmissibilityRules::default(),
    };
    let (structural, robust) = evaluate_robust(&hard, &atoms, &profile(vec![])).unwrap();
    assert_eq!(structural.verdict, StructuralVerdict::NoAdmissibleAction);
    assert_eq!(robust.verdict, RobustVerdict::NoAdmissibleAction);
}

#[test]
fn f4_admissibility_declared_exclusion_and_hard_constraint_only_remove_actions() {
    // A has the higher payoff in both structures but is declared inadmissible;
    // B wins, and A keeps its raw range rather than a penalized one.
    let rules = AdmissibilityRules {
        declared_exclusions: vec![DeclaredExclusion {
            action_id: "A".into(),
            reason: "legal".into(),
        }],
        ..AdmissibilityRules::default()
    };
    let c = contract(StructuralPolicy::RequireInvariantBestAction, rules);
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 6.0, 2.0)];
    let result = run(&c, &atoms, &profile(vec![]));
    assert_eq!(result.verdict, RobustVerdict::StructurallyRobust("B".into()));
    assert_eq!(result.actions[0].declared_exclusion.as_deref(), Some("legal"));
    assert_eq!(result.actions[0].range, Some((5.0, 6.0)));
    assert_eq!(result.actions[1].range, Some((2.0, 3.0)));

    // A hard constraint on A alone: the excluded action has no criterion range
    // and B's value is untouched.
    let cap = HardConstraint {
        id: "cap".into(),
        expr: UtilityExpr::Input(0),
        bound: 4.0,
        min_probability: 1.0,
        units: "units".into(),
        applies_to: vec!["A".into()],
    };
    let hard = AdmissibleDecisionContract {
        contract: base(
            DecisionCriterion::PosteriorExpectedUtility,
            StructuralPolicy::RequireInvariantBestAction,
            vec![cap],
        ),
        rules: AdmissibilityRules::default(),
    };
    let atoms = [evaluated("s1", None, 9.0, 3.0), evaluated("s2", None, 8.0, 2.0)];
    let (structural, robust) = evaluate_robust(&hard, &atoms, &profile(vec![])).unwrap();
    assert_eq!(structural.verdict, StructuralVerdict::InvariantBest("B".into()));
    assert_eq!(robust.verdict, RobustVerdict::StructurallyRobust("B".into()));
    assert_eq!(robust.actions[0].range, None);
    assert_eq!(robust.actions[1].range, Some((2.0, 3.0)));
}

#[test]
fn f4_admissibility_uncertainty_requirement_is_enforced_by_kind() {
    use UncertaintyKind as K;
    use UncertaintyRequirement as R;
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 6.0, 2.0)];
    let with = |required: UncertaintyRequirement, supplied: UncertaintyKind| {
        let c = contract(
            StructuralPolicy::RequireInvariantBestAction,
            AdmissibilityRules { uncertainty: required, ..AdmissibilityRules::default() },
        );
        let claims = ClaimProfile { uncertainty: supplied, support: vec![] };
        run(&c, &atoms, &claims).verdict
    };
    let ok = RobustVerdict::StructurallyRobust("A".into());
    assert_eq!(with(R::None, K::Point), ok);
    assert_eq!(with(R::None, K::Credible), ok);
    assert_eq!(with(R::PointOnly, K::Point), ok);
    assert_eq!(with(R::StructuralEnvelope, K::StructuralEnvelope), ok);
    assert_eq!(with(R::Credible, K::Credible), ok);
    for (required, supplied) in [
        (R::PointOnly, K::StructuralEnvelope),
        (R::PointOnly, K::Credible),
        (R::StructuralEnvelope, K::Point),
        (R::StructuralEnvelope, K::Credible),
        (R::Credible, K::Point),
        (R::Credible, K::StructuralEnvelope),
    ] {
        assert!(
            matches!(with(required, supplied), RobustVerdict::InsufficientClaims(_)),
            "{required:?} accepted {supplied:?}"
        );
    }
}

#[test]
fn f6_robustness_bayes_keeps_unresolved_and_unsupported_mass() {
    // A = 0.5*5 + 0.5*1 = 3.0 and B = 0.5*3 + 0.5*4 = 3.5.
    let atoms = [evaluated("s1", Some(0.5), 5.0, 3.0), evaluated("s2", Some(0.5), 1.0, 4.0)];
    let unruled = contract(StructuralPolicy::BayesOverStructures, AdmissibilityRules::default());
    let result = run(&unruled, &atoms, &profile(vec![]));
    match result.verdict {
        RobustVerdict::BayesChoice { ref action, evaluated_mass } => {
            assert_eq!(action, "B");
            assert!(near(evaluated_mass, 1.0));
        }
        ref other => panic!("{other:?}"),
    }

    // The half of the mass that sits on an unsupported structure is reported
    // and the choice is not made on the remainder.
    let ruled = contract(StructuralPolicy::BayesOverStructures, require_supported());
    let claims = profile(vec![
        ("s1", AtomSupport::supported()),
        ("s2", AtomSupport::uniform(SupportStatus::Extrapolative)),
    ]);
    let result = run(&ruled, &atoms, &claims);
    assert_eq!(result.verdict, RobustVerdict::UnsupportedExtrapolation);
    assert!(near(result.unsupported_mass.unwrap(), 0.5));

    // An unidentified structure's mass is carried and never renormalized.
    let partial = [
        evaluated("s1", Some(0.5), 5.0, 3.0),
        evaluated("s2", Some(0.3), 1.0, 4.0),
        atom("s3", Some(0.2), AtomEvidence::Unidentified),
    ];
    let result = run(&unruled, &partial, &profile(vec![]));
    assert!(near(result.unidentified_mass.unwrap(), 0.2));
    assert!(near(result.evaluated_mass.unwrap(), 0.8));
    assert!(matches!(result.verdict, RobustVerdict::InsufficientClaims(_)));
}

#[test]
fn f6_robustness_worst_case_choice_changes_when_support_is_ignored() {
    // Worst case over {s1, s2}: A = min(5, 1) = 1 and B = min(3, 4) = 3, so B.
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 1.0, 4.0)];
    let unruled = contract(StructuralPolicy::Maximin, AdmissibilityRules::default());
    let result = run(&unruled, &atoms, &profile(vec![]));
    assert_eq!(result.verdict, RobustVerdict::WorstCaseChoice("B".into()));

    // With s2 unsupported only s1 remains: A = 5 beats B = 3.
    let ruled = contract(StructuralPolicy::Maximin, require_supported());
    let claims = profile(vec![
        ("s1", AtomSupport::supported()),
        ("s2", AtomSupport::uniform(SupportStatus::Extrapolative)),
    ]);
    let result = run(&ruled, &atoms, &claims);
    assert_eq!(
        result.verdict,
        RobustVerdict::SupportDependent {
            supported_choice: "A".into(),
            unrestricted_choice: Some("B".into()),
        }
    );
}

#[test]
fn f6_robustness_report_only_never_chooses() {
    let c = contract(StructuralPolicy::ReportOnly, require_supported());
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 1.0, 4.0)];
    let result = run(&c, &atoms, &profile(vec![]));
    assert_eq!(result.verdict, RobustVerdict::ReportOnly);
    assert_eq!(result.verdict.selected_action(), None);
}

#[test]
fn f6_robustness_refuses_a_foreign_result_and_an_unknown_profile_entry() {
    let c = contract(StructuralPolicy::RequireInvariantBestAction, AdmissibilityRules::default());
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 6.0, 2.0)];
    let structural = evaluate_structural(&c.contract, &atoms).unwrap();

    let mut other = c.clone();
    other.contract.criterion = DecisionCriterion::PosteriorExpectedLoss;
    let error = assess_robustness(&other, &structural, &profile(vec![])).unwrap_err();
    assert_eq!(error, RobustnessError::ContractMismatch);
    let refusal = error.to_refusal();
    assert_eq!(refusal.detail, "decision_robustness.contract_mismatch");
    assert_eq!(refusal.code, "decision_contract_unsatisfied");

    let ghost = profile(vec![("ghost", AtomSupport::supported())]);
    let error = assess_robustness(&c, &structural, &ghost).unwrap_err();
    assert!(matches!(error, RobustnessError::InvalidProfile(_)));
    assert_eq!(error.to_refusal().detail, "decision_robustness.invalid_profile");

    let twice = profile(vec![("s1", AtomSupport::supported()), ("s1", AtomSupport::supported())]);
    assert!(matches!(
        assess_robustness(&c, &structural, &twice),
        Err(RobustnessError::InvalidProfile(_))
    ));

    // An invalid rule set surfaces through the same entry point.
    let mut bad = c;
    bad.rules.declared_exclusions =
        vec![DeclaredExclusion { action_id: "ghost".into(), reason: "x".into() }];
    let error = assess_robustness(&bad, &structural, &profile(vec![])).unwrap_err();
    assert_eq!(
        error,
        RobustnessError::Admissibility(AdmissibilityError::UnknownAction("ghost".into()))
    );
    assert_eq!(error.to_refusal().detail, "decision_admissibility.unknown_action");
}

fn rich_rules() -> AdmissibilityRules {
    AdmissibilityRules {
        default_weakest_support: Some(SupportStatus::WeakOverlap),
        support_rules: vec![
            SupportRule {
                action_id: "A".into(),
                input: 0,
                weakest_allowed: SupportStatus::Supported,
            },
            SupportRule {
                action_id: "B".into(),
                input: 0,
                weakest_allowed: SupportStatus::Extrapolative,
            },
        ],
        declared_exclusions: vec![DeclaredExclusion {
            action_id: "A".into(),
            reason: "legal".into(),
        }],
        uncertainty: UncertaintyRequirement::StructuralEnvelope,
    }
}

#[test]
fn f4_admissibility_identity_ignores_order_but_not_semantics() {
    type Edit = Box<dyn Fn(&mut AdmissibleDecisionContract)>;
    let c = contract(StructuralPolicy::Maximin, rich_rules());
    let id = c.identity().unwrap();
    // The wrapper's identity is its own, not the base contract's.
    assert_ne!(id, c.contract.identity().unwrap());

    let mut reordered = c.clone();
    reordered.contract.actions.reverse();
    reordered.rules.support_rules.reverse();
    reordered.rules.declared_exclusions.reverse();
    assert_eq!(reordered.identity().unwrap(), id);

    let mut two_exclusions = c.clone();
    two_exclusions
        .rules
        .declared_exclusions
        .push(DeclaredExclusion { action_id: "B".into(), reason: "ethics".into() });
    let two_id = two_exclusions.identity().unwrap();
    two_exclusions.rules.declared_exclusions.reverse();
    assert_eq!(two_exclusions.identity().unwrap(), two_id);

    let edits: Vec<Edit> = vec![
        Box::new(|c| c.rules.default_weakest_support = Some(SupportStatus::Supported)),
        Box::new(|c| c.rules.default_weakest_support = None),
        Box::new(|c| c.rules.support_rules[0].weakest_allowed = SupportStatus::WeakOverlap),
        Box::new(|c| c.rules.support_rules[1].weakest_allowed = SupportStatus::MissingEvidence),
        Box::new(|c| {
            c.rules.support_rules.pop();
        }),
        Box::new(|c| c.rules.declared_exclusions[0].reason = "ethics".into()),
        Box::new(|c| c.rules.declared_exclusions[0].action_id = "B".into()),
        Box::new(|c| c.rules.declared_exclusions.clear()),
        Box::new(|c| c.rules.uncertainty = UncertaintyRequirement::Credible),
        Box::new(|c| c.rules.uncertainty = UncertaintyRequirement::PointOnly),
        Box::new(|c| c.rules.uncertainty = UncertaintyRequirement::None),
        Box::new(|c| c.contract.criterion = DecisionCriterion::PosteriorExpectedLoss),
        Box::new(|c| c.contract.structural_policy = StructuralPolicy::ReportOnly),
        Box::new(|c| c.contract.actions[1].utility = UtilityExpr::Const(1.0)),
    ];
    let mut seen = vec![id];
    for edit in edits {
        let mut changed = c.clone();
        edit(&mut changed);
        // Some edits (a constant-only utility) are invalid contracts; they
        // must refuse rather than share an identity.
        match changed.identity() {
            Ok(changed_id) => {
                assert!(!seen.contains(&changed_id), "an edit left the identity unchanged");
                seen.push(changed_id);
            }
            Err(error) => assert!(matches!(error, AdmissibilityError::Contract(_))),
        }
    }
    // Default rules are a distinct, valid declaration.
    let plain = contract(StructuralPolicy::Maximin, AdmissibilityRules::default());
    assert!(!seen.contains(&plain.identity().unwrap()));
}

#[test]
fn f4_admissibility_malformed_rules_refuse_with_registered_details() {
    let refuse = |edit: &dyn Fn(&mut AdmissibleDecisionContract)| {
        let mut c = contract(StructuralPolicy::Maximin, AdmissibilityRules::default());
        edit(&mut c);
        c.validate().unwrap_err()
    };
    let rule = |action: &str, input: usize| SupportRule {
        action_id: action.into(),
        input,
        weakest_allowed: SupportStatus::Supported,
    };
    let error = refuse(&|c| c.rules.support_rules = vec![rule("ghost", 0)]);
    assert_eq!(error, AdmissibilityError::UnknownAction("ghost".into()));
    assert_eq!(error.to_refusal().detail, "decision_admissibility.unknown_action");

    let error = refuse(&|c| c.rules.support_rules = vec![rule("A", 3)]);
    assert_eq!(error, AdmissibilityError::UnknownInput { action: "A".into(), input: 3 });
    assert_eq!(error.to_refusal().detail, "decision_admissibility.unknown_input");

    let error = refuse(&|c| c.rules.support_rules = vec![rule("A", 0), rule("A", 0)]);
    assert_eq!(error, AdmissibilityError::DuplicateRule("A[0]".into()));
    assert_eq!(error.to_refusal().detail, "decision_admissibility.duplicate_rule");

    let blank = DeclaredExclusion { action_id: "A".into(), reason: "  ".into() };
    let error = refuse(&|c| c.rules.declared_exclusions = vec![blank.clone()]);
    assert!(matches!(error, AdmissibilityError::InvalidRule(_)));
    assert_eq!(error.to_refusal().detail, "decision_admissibility.invalid_rule");

    let twice = DeclaredExclusion { action_id: "A".into(), reason: "legal".into() };
    let error = refuse(&|c| c.rules.declared_exclusions = vec![twice.clone(), twice.clone()]);
    assert_eq!(error, AdmissibilityError::DuplicateRule("A".into()));

    let error = refuse(&|c| {
        c.contract.actions.pop();
    });
    assert!(matches!(error, AdmissibilityError::Contract(_)));
    assert_eq!(error.to_refusal().detail, "decision_contract.invalid_declaration");
}

#[test]
fn f4_admissibility_support_status_orders_by_severity() {
    let rules = AdmissibilityRules {
        default_weakest_support: Some(SupportStatus::WeakOverlap),
        support_rules: vec![SupportRule {
            action_id: "B".into(),
            input: 0,
            weakest_allowed: SupportStatus::Extrapolative,
        }],
        ..AdmissibilityRules::default()
    };
    assert!(rules.support_met("A", 0, SupportStatus::Supported));
    assert!(rules.support_met("A", 0, SupportStatus::WeakOverlap));
    assert!(!rules.support_met("A", 0, SupportStatus::Extrapolative));
    assert!(rules.support_met("B", 0, SupportStatus::Extrapolative));
    assert!(!rules.support_met("B", 0, SupportStatus::OutsideEmpiricalSupport));
    assert!(!rules.support_met("B", 0, SupportStatus::MissingEvidence));
    // Without a default or rule an input is unchecked.
    let open = AdmissibilityRules::default();
    assert!(open.support_met("A", 0, SupportStatus::MissingEvidence));
    assert_eq!(open.weakest_allowed("A", 0), None);
}
