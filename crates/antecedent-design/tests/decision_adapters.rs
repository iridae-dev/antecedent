//! Adapters from 2.2 claim types to decision inputs under a declared policy.
//! Expected values are enumerated by hand in each test.

use antecedent_core::{IdentifiedSet, QuantityRole, ScientificQuantity, SupportStatus};
use antecedent_design::decision_adapters::{
    AdapterError, ClaimKind, ClaimProbability, IdentifiedUtility, IdentifiedVerdict, SuppliedClaim,
    adapt_finite_scenarios, adapt_graph_dependent_claims, adapt_point_claim,
    adapt_weighted_graph_atoms, evaluate_adapted, evaluate_identified_sets, utility_interval,
};
use antecedent_design::decision_contract::{
    ActionKind, AdmissibilityRules, AdmissibleDecisionContract, DecisionAction, DecisionContract,
    DecisionCriterion, DeclaredExclusion, StructuralPolicy, UncertaintyKind,
    UncertaintyRequirement, UtilityExpr,
};
use antecedent_design::decision_robustness::{
    AtomSupport, InputSupport, RobustVerdict, SupportShortfall,
};
use antecedent_design::decision_structural::{AtomEvidence, AtomStatus, StructuralVerdict};
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

fn claim(id: &str, probability: ClaimProbability, evidence: AtomEvidence) -> SuppliedClaim {
    SuppliedClaim { id: id.into(), probability, evidence, support: AtomSupport::supported() }
}

fn exact(id: &str, probability: ClaimProbability, a: f64, b: f64) -> SuppliedClaim {
    claim(id, probability, AtomEvidence::Evaluated(law(a, b)))
}

fn base(criterion: DecisionCriterion, policy: StructuralPolicy) -> DecisionContract {
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
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: policy,
    }
}

fn contract(
    criterion: DecisionCriterion,
    policy: StructuralPolicy,
    rules: AdmissibilityRules,
) -> AdmissibleDecisionContract {
    AdmissibleDecisionContract { contract: base(criterion, policy), rules }
}

fn plain(policy: StructuralPolicy) -> AdmissibleDecisionContract {
    contract(DecisionCriterion::PosteriorExpectedUtility, policy, AdmissibilityRules::default())
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

fn interval(id: &str, lower: f64, upper: f64) -> IdentifiedUtility {
    IdentifiedUtility {
        action_id: id.into(),
        utility: IdentifiedSet::try_new(lower, upper).unwrap(),
        hard_exclusions: vec![],
        support: AtomSupport::supported(),
    }
}

fn require_supported() -> AdmissibilityRules {
    AdmissibilityRules {
        default_weakest_support: Some(SupportStatus::Supported),
        ..AdmissibilityRules::default()
    }
}

#[test]
fn f6_adapter_partial_identification_gives_an_interval_per_action_with_conflicting_leaders() {
    // A in [2, 8], B in [4, 6]. The lower bound favors B (4 > 2), the upper bound
    // favors A (8 > 6); neither interval lies wholly above the other.
    let sets = [interval("A", 2.0, 8.0), interval("B", 4.0, 6.0)];
    let invariant = plain(StructuralPolicy::RequireInvariantBestAction);
    let result = evaluate_identified_sets(&invariant, &sets).unwrap();
    assert_eq!(result.actions[0].utility, (2.0, 8.0));
    assert_eq!(result.actions[1].utility, (4.0, 6.0));
    assert_eq!(result.lower_leader.as_deref(), Some("B"));
    assert_eq!(result.upper_leader.as_deref(), Some("A"));
    assert!(result.conflicting_leaders);
    assert!(result.actions.iter().all(|a| a.possibly_optimal && !a.necessarily_optimal));
    assert!(result.actions.iter().all(|a| a.dominated_by.is_empty()));
    assert_eq!(
        result.verdict,
        IdentifiedVerdict::NoNecessarilyBest { possibly_optimal: vec!["A".into(), "B".into()] }
    );
    // Regret over the sets: A loses at most 6 - 2 = 4 to B, B at most 8 - 4 = 4 to A.
    assert!(near(result.actions[0].max_regret.unwrap(), 4.0));
    assert!(near(result.actions[1].max_regret.unwrap(), 4.0));

    // Maximin: worst cases are 2 and 4, so B.
    let maximin = plain(StructuralPolicy::Maximin);
    let result = evaluate_identified_sets(&maximin, &sets).unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::WorstCaseChoice("B".into()));

    // Minimax regret ties at 4 and 4, so no action is named.
    let regret = contract(
        DecisionCriterion::Regret,
        StructuralPolicy::Maximin,
        AdmissibilityRules::default(),
    );
    let result = evaluate_identified_sets(&regret, &sets).unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::Tied(vec!["A".into(), "B".into()]));

    // Widen B to [3, 6]: B's regret is 8 - 3 = 5 and A's is 6 - 2 = 4, so A.
    let wider = [interval("A", 2.0, 8.0), interval("B", 3.0, 6.0)];
    let result = evaluate_identified_sets(&regret, &wider).unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::MinimaxRegretChoice("A".into()));
    assert!(near(result.actions[1].max_regret.unwrap(), 5.0));

    // Report only returns the intervals.
    let report = plain(StructuralPolicy::ReportOnly);
    let result = evaluate_identified_sets(&report, &sets).unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::ReportOnly);
}

#[test]
fn f6_adapter_identified_sets_with_a_dominating_action_have_a_necessarily_best() {
    // A in [5, 8] beats B in [1, 4] for every value: 5 > 4.
    let sets = [interval("A", 5.0, 8.0), interval("B", 1.0, 4.0)];
    let result =
        evaluate_identified_sets(&plain(StructuralPolicy::RequireInvariantBestAction), &sets)
            .unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::NecessarilyBest("A".into()));
    assert!(result.actions[0].necessarily_optimal);
    assert_eq!(result.actions[1].dominated_by, vec!["A".to_owned()]);
    assert!(!result.actions[1].possibly_optimal);
    assert!(!result.conflicting_leaders);
    // A cannot regret choosing A: max(0, 4 - 5).
    assert!(near(result.actions[0].max_regret.unwrap(), 0.0));
    // B can regret up to 8 - 1 = 7.
    assert!(near(result.actions[1].max_regret.unwrap(), 7.0));
    // Touching intervals do not dominate strictly: A in [4, 8] vs B in [1, 4].
    let touching = [interval("A", 4.0, 8.0), interval("B", 1.0, 4.0)];
    let result =
        evaluate_identified_sets(&plain(StructuralPolicy::RequireInvariantBestAction), &touching)
            .unwrap();
    assert!(matches!(result.verdict, IdentifiedVerdict::NoNecessarilyBest { .. }));
}

#[test]
fn f6_adapter_identified_losses_are_minimized_at_their_worst_case() {
    // Losses A in [2, 8] and B in [4, 6]: the worst losses are 8 and 6, so B.
    let sets = [interval("A", 2.0, 8.0), interval("B", 4.0, 6.0)];
    let by_loss = contract(
        DecisionCriterion::PosteriorExpectedLoss,
        StructuralPolicy::Maximin,
        AdmissibilityRules::default(),
    );
    let result = evaluate_identified_sets(&by_loss, &sets).unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::WorstCaseChoice("B".into()));
    // The supplied intervals are reported as given, not flipped.
    assert_eq!(result.actions[0].utility, (2.0, 8.0));
    let minimax = contract(
        DecisionCriterion::MinimaxOverIdentifiedSet,
        StructuralPolicy::ReportOnly,
        AdmissibilityRules::default(),
    );
    let result = evaluate_identified_sets(&minimax, &sets).unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::WorstCaseChoice("B".into()));
}

#[test]
fn f6_adapter_identified_sets_respect_constraints_declarations_and_support() {
    let invariant = |rules| {
        contract(
            DecisionCriterion::PosteriorExpectedUtility,
            StructuralPolicy::RequireInvariantBestAction,
            rules,
        )
    };
    // A hard constraint removes A; B alone is then the only choice.
    let mut sets = [interval("A", 5.0, 8.0), interval("B", 1.0, 4.0)];
    sets[0].hard_exclusions = vec!["cap".into()];
    let result =
        evaluate_identified_sets(&invariant(AdmissibilityRules::default()), &sets).unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::NecessarilyBest("B".into()));
    assert!(!result.actions[0].eligible);
    assert_eq!(result.actions[0].max_regret, None);
    assert_eq!(result.actions[0].hard_exclusions, vec!["cap".to_owned()]);

    // Both removed by constraints: no admissible action.
    sets[1].hard_exclusions = vec!["cap".into()];
    let result =
        evaluate_identified_sets(&invariant(AdmissibilityRules::default()), &sets).unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::NoAdmissibleAction);

    // A declaration removes A without touching its interval.
    let sets = [interval("A", 5.0, 8.0), interval("B", 1.0, 4.0)];
    let declared = AdmissibilityRules {
        declared_exclusions: vec![DeclaredExclusion {
            action_id: "A".into(),
            reason: "legal".into(),
        }],
        ..AdmissibilityRules::default()
    };
    let result = evaluate_identified_sets(&invariant(declared), &sets).unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::NecessarilyBest("B".into()));
    assert_eq!(result.actions[0].declared_exclusion.as_deref(), Some("legal"));
    assert_eq!(result.actions[0].utility, (5.0, 8.0));

    // Support below the rule removes an action in the set; all removed is an
    // unsupported extrapolation, not a missing admissible action.
    let mut weak = [interval("A", 5.0, 8.0), interval("B", 1.0, 4.0)];
    weak[0].support = AtomSupport::uniform(SupportStatus::Extrapolative);
    let result = evaluate_identified_sets(&invariant(require_supported()), &weak).unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::NecessarilyBest("B".into()));
    assert_eq!(result.actions[0].support_shortfalls, vec![(0, SupportStatus::Extrapolative)]);
    weak[1].support = AtomSupport::uniform(SupportStatus::OutsideEmpiricalSupport);
    let result = evaluate_identified_sets(&invariant(require_supported()), &weak).unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::UnsupportedExtrapolation);

    // An envelope is not a credible law.
    let credible = AdmissibilityRules {
        uncertainty: UncertaintyRequirement::Credible,
        ..AdmissibilityRules::default()
    };
    let result = evaluate_identified_sets(&invariant(credible), &sets).unwrap();
    assert!(matches!(result.verdict, IdentifiedVerdict::InsufficientClaims(_)));
}

#[test]
fn f6_adapter_identified_set_refusals() {
    let sets = [interval("A", 2.0, 8.0), interval("B", 4.0, 6.0)];
    let error =
        evaluate_identified_sets(&plain(StructuralPolicy::BayesOverStructures), &sets).unwrap_err();
    assert_eq!(error, AdapterError::BayesOverIdentifiedSet);
    assert_eq!(error.to_refusal().detail, "decision_adapters.bayes_over_identified_set");
    assert_eq!(error.to_refusal().code, "decision_contract_unsatisfied");

    let quantile = contract(
        DecisionCriterion::Quantile { p: 0.5 },
        StructuralPolicy::Maximin,
        AdmissibilityRules::default(),
    );
    let error = evaluate_identified_sets(&quantile, &sets).unwrap_err();
    assert_eq!(error, AdapterError::CriterionNotLicensedForSet);
    assert_eq!(error.to_refusal().detail, "decision_adapters.criterion_not_licensed_for_set");

    let c = plain(StructuralPolicy::Maximin);
    let error = evaluate_identified_sets(&c, &sets[..1]).unwrap_err();
    assert_eq!(error, AdapterError::MissingAction("B".into()));
    assert_eq!(error.to_refusal().detail, "decision_adapters.missing_action");

    let twice = [interval("A", 2.0, 8.0), interval("A", 4.0, 6.0), interval("B", 1.0, 2.0)];
    let error = evaluate_identified_sets(&c, &twice).unwrap_err();
    assert_eq!(error, AdapterError::DuplicateAction("A".into()));
    assert_eq!(error.to_refusal().detail, "decision_adapters.duplicate_action");

    let ghost = [interval("A", 2.0, 8.0), interval("B", 4.0, 6.0), interval("C", 0.0, 1.0)];
    let error = evaluate_identified_sets(&c, &ghost).unwrap_err();
    assert_eq!(error, AdapterError::UnknownAction("C".into()));
    assert_eq!(error.to_refusal().detail, "decision_adapters.unknown_action");

    // Fields are public, so an inverted or non-finite interval can be built.
    let mut inverted = [interval("A", 2.0, 8.0), interval("B", 4.0, 6.0)];
    inverted[0].utility = IdentifiedSet { lower: 9.0, upper: 1.0 };
    let error = evaluate_identified_sets(&c, &inverted).unwrap_err();
    assert_eq!(error, AdapterError::InvalidInterval("A".into()));
    assert_eq!(error.to_refusal().detail, "decision_adapters.invalid_interval");
    inverted[0].utility = IdentifiedSet { lower: f64::NAN, upper: 1.0 };
    assert!(evaluate_identified_sets(&c, &inverted).is_err());

    // A contract that is invalid refuses before any interval is read.
    let mut bad = c;
    bad.contract.actions.pop();
    assert!(matches!(evaluate_identified_sets(&bad, &sets), Err(AdapterError::Admissibility(_))));
}

/// Two actions: `T` reads (benefit, cost) as `benefit * cost - 1`; `W` reads
/// benefit alone.
fn interval_contract(utility: UtilityExpr) -> DecisionContract {
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "T".into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("benefit", "do(a=1)"), quantity("cost", "do(a=1)")],
                utility,
            },
            DecisionAction {
                id: "W".into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("benefit", "do(a=0)")],
                utility: UtilityExpr::Input(0),
            },
        ],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::Maximin,
    }
}

#[test]
fn f6_adapter_utility_interval_encloses_the_nonlinear_utility() {
    let product_minus_one = UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
        UtilityExpr::Const(1.0),
    );
    let c = interval_contract(product_minus_one);
    let set = |lower, upper| IdentifiedSet::try_new(lower, upper).unwrap();

    // benefit in [1, 2], cost in [3, 4]: products 3, 4, 6, 8 so [3, 8]; minus 1.
    let got = utility_interval(&c, "T", &[set(1.0, 2.0), set(3.0, 4.0)]).unwrap();
    assert_eq!(got, set(2.0, 7.0));
    // benefit in [-1, 2]: products -3, -4, 6, 8 so [-4, 8]; minus 1 gives [-5, 7].
    let got = utility_interval(&c, "T", &[set(-1.0, 2.0), set(3.0, 4.0)]).unwrap();
    assert_eq!(got, set(-5.0, 7.0));

    // max(benefit * cost - 1, 0) over the same inputs clips the lower end: [0, 7].
    let clipped = UtilityExpr::maximum(
        UtilityExpr::difference(
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            UtilityExpr::Const(1.0),
        ),
        UtilityExpr::Const(0.0),
    );
    let c = interval_contract(clipped);
    let got = utility_interval(&c, "T", &[set(-1.0, 2.0), set(3.0, 4.0)]).unwrap();
    assert_eq!(got, set(0.0, 7.0));

    // min(benefit, -cost) over benefit [1, 2], cost [3, 4]: min of [1, 2] and
    // [-4, -3] is [-4, -3].
    let min_expr = UtilityExpr::minimum(
        UtilityExpr::Input(0),
        UtilityExpr::Neg(Box::new(UtilityExpr::Input(1))),
    );
    let c = interval_contract(min_expr);
    let got = utility_interval(&c, "T", &[set(1.0, 2.0), set(3.0, 4.0)]).unwrap();
    assert_eq!(got, set(-4.0, -3.0));

    // The enclosure feeds the decision: W in [0, 1] against T in [-4, -3] makes
    // W necessarily best.
    let w = interval("W", 0.0, 1.0);
    let t = IdentifiedUtility {
        action_id: "T".into(),
        utility: got,
        hard_exclusions: vec![],
        support: AtomSupport::supported(),
    };
    let mut invariant =
        AdmissibleDecisionContract { contract: c, rules: AdmissibilityRules::default() };
    invariant.contract.structural_policy = StructuralPolicy::RequireInvariantBestAction;
    let result = evaluate_identified_sets(&invariant, &[t, w]).unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::NecessarilyBest("W".into()));

    // Wrong counts and unknown actions refuse.
    assert!(matches!(
        utility_interval(&invariant.contract, "T", &[set(1.0, 2.0)]),
        Err(AdapterError::InvalidInterval(_))
    ));
    assert!(matches!(
        utility_interval(&invariant.contract, "ghost", &[set(1.0, 2.0)]),
        Err(AdapterError::UnknownAction(_))
    ));
}

#[test]
fn f6_adapter_invariant_best_action_across_graph_dependent_claims() {
    let claims = vec![
        exact("g1", ClaimProbability::Unspecified, 5.0, 3.0),
        exact("g2", ClaimProbability::Unspecified, 6.0, 2.0),
    ];
    let adapted =
        adapt_graph_dependent_claims(claims, StructuralPolicy::RequireInvariantBestAction).unwrap();
    assert_eq!(adapted.kind, ClaimKind::GraphDependent);
    assert_eq!(adapted.profile.uncertainty, UncertaintyKind::StructuralEnvelope);
    let c = plain(StructuralPolicy::RequireInvariantBestAction);
    let decision = evaluate_adapted(&c, &adapted).unwrap();
    assert_eq!(decision.structural.verdict, StructuralVerdict::InvariantBest("A".into()));
    assert_eq!(decision.robust.verdict, RobustVerdict::StructurallyRobust("A".into()));
    // The structural spread is kept, not averaged: A in {5, 6}.
    assert_eq!(decision.structural.actions[0].range, Some((5.0, 6.0)));
}

#[test]
fn f6_adapter_graph_dependent_claims_name_each_graphs_leader() {
    let claims = vec![
        exact("g1", ClaimProbability::Unspecified, 5.0, 3.0),
        exact("g2", ClaimProbability::Unspecified, 1.0, 4.0),
    ];
    let adapted =
        adapt_graph_dependent_claims(claims, StructuralPolicy::RequireInvariantBestAction).unwrap();
    let decision =
        evaluate_adapted(&plain(StructuralPolicy::RequireInvariantBestAction), &adapted).unwrap();
    assert_eq!(
        decision.robust.verdict,
        RobustVerdict::GraphDependentChoice(vec![
            ("g1".into(), vec!["A".into()]),
            ("g2".into(), vec!["B".into()]),
        ])
    );
    // Graph-dependent claims carry no weights.
    assert!(decision.structural.actions.iter().all(|a| a.weighted_value.is_none()));
    assert!(decision.structural.evaluated_mass.is_none());
}

#[test]
fn f6_adapter_graph_dependent_claims_refuse_probabilities() {
    let claims = vec![
        exact("g1", ClaimProbability::Genuine(0.5), 5.0, 3.0),
        exact("g2", ClaimProbability::Genuine(0.5), 1.0, 4.0),
    ];
    let error = adapt_graph_dependent_claims(claims, StructuralPolicy::RequireInvariantBestAction)
        .unwrap_err();
    assert_eq!(error, AdapterError::GraphClaimsCarryNoProbability("g1".into()));
    assert_eq!(error.to_refusal().detail, "decision_adapters.graph_claims_carry_no_probability");
}

#[test]
fn f6_adapter_weighted_graph_atoms_never_renormalize_unidentified_mass() {
    let claims = vec![
        exact("g1", ClaimProbability::Genuine(0.5), 5.0, 3.0),
        exact("g2", ClaimProbability::Genuine(0.3), 1.0, 4.0),
        claim("g3", ClaimProbability::Genuine(0.2), AtomEvidence::Unidentified),
    ];
    let adapted =
        adapt_weighted_graph_atoms(claims, StructuralPolicy::BayesOverStructures).unwrap();
    assert_eq!(adapted.profile.uncertainty, UncertaintyKind::Credible);
    let decision =
        evaluate_adapted(&plain(StructuralPolicy::BayesOverStructures), &adapted).unwrap();
    let structural = &decision.structural;
    assert!(near(structural.unidentified_mass.unwrap(), 0.2));
    assert!(near(structural.evaluated_mass.unwrap(), 0.8));
    // A: 0.5 * 5 + 0.3 * 1 = 2.8 and B: 0.5 * 3 + 0.3 * 4 = 2.7, over 0.8 of the mass.
    assert!(near(structural.actions[0].weighted_value.unwrap(), 2.8));
    assert!(near(structural.actions[1].weighted_value.unwrap(), 2.7));
    assert!(matches!(decision.robust.verdict, RobustVerdict::InsufficientClaims(_)));
    assert!(near(decision.robust.unidentified_mass.unwrap(), 0.2));
    assert!(matches!(structural.atoms[2].status, AtomStatus::Unidentified));

    // With the whole mass evaluated, Bayes chooses. Weights 0.6 and 0.4 with
    // payoffs (5, 3) and (1, 2): A = 3.0 + 0.4 = 3.4 and B = 1.8 + 0.8 = 2.6.
    let claims = vec![
        exact("g1", ClaimProbability::Genuine(0.6), 5.0, 3.0),
        exact("g2", ClaimProbability::Genuine(0.4), 1.0, 2.0),
    ];
    let adapted =
        adapt_weighted_graph_atoms(claims, StructuralPolicy::BayesOverStructures).unwrap();
    let decision =
        evaluate_adapted(&plain(StructuralPolicy::BayesOverStructures), &adapted).unwrap();
    assert!(near(decision.structural.actions[0].weighted_value.unwrap(), 3.4));
    assert!(near(decision.structural.actions[1].weighted_value.unwrap(), 2.6));
    match decision.robust.verdict {
        RobustVerdict::BayesChoice { ref action, evaluated_mass } => {
            assert_eq!(action, "A");
            assert!(near(evaluated_mass, 1.0));
        }
        ref other => panic!("{other:?}"),
    }
    // Mass where each action leads: A at g1 (0.6); at g2 the payoffs are A = 1
    // and B = 2, so B leads there (0.4).
    assert!(near(decision.structural.actions[0].mass_where_best.unwrap(), 0.6));
    assert!(near(decision.structural.actions[1].mass_where_best.unwrap(), 0.4));
}

#[test]
fn f6_adapter_weighted_atoms_bayes_refusal_without_genuine_probabilities() {
    // Counts are never probabilities.
    let counts = vec![
        exact("g1", ClaimProbability::CompletionCount(3), 5.0, 3.0),
        exact("g2", ClaimProbability::CompletionCount(1), 1.0, 4.0),
    ];
    let error = adapt_weighted_graph_atoms(counts.clone(), StructuralPolicy::BayesOverStructures)
        .unwrap_err();
    assert_eq!(error, AdapterError::CompletionCountNotProbability("g1".into()));
    let refusal = error.to_refusal();
    assert_eq!(refusal.detail, "decision_adapters.completion_count_not_probability");
    assert_eq!(refusal.code, "decision_contract_unsatisfied");

    // Neither is the absence of any weight.
    let none = vec![
        exact("g1", ClaimProbability::Unspecified, 5.0, 3.0),
        exact("g2", ClaimProbability::Unspecified, 1.0, 4.0),
    ];
    let error = adapt_weighted_graph_atoms(none.clone(), StructuralPolicy::BayesOverStructures)
        .unwrap_err();
    assert_eq!(error, AdapterError::ProbabilitiesRequired);
    assert_eq!(error.to_refusal().detail, "decision_adapters.probabilities_required");
    let error =
        adapt_finite_scenarios(none.clone(), StructuralPolicy::BayesOverStructures).unwrap_err();
    assert_eq!(error, AdapterError::ProbabilitiesRequired);
    let error =
        adapt_graph_dependent_claims(none, StructuralPolicy::BayesOverStructures).unwrap_err();
    assert_eq!(error, AdapterError::ProbabilitiesRequired);
    let error = adapt_point_claim(
        "p",
        AtomEvidence::Evaluated(law(5.0, 3.0)),
        AtomSupport::supported(),
        StructuralPolicy::BayesOverStructures,
    )
    .unwrap_err();
    assert_eq!(error, AdapterError::ProbabilitiesRequired);

    // Other policies accept counts but drop them from the weights and keep them.
    let adapted = adapt_weighted_graph_atoms(counts, StructuralPolicy::Maximin).unwrap();
    assert_eq!(adapted.completion_counts, vec![("g1".to_owned(), 3), ("g2".to_owned(), 1)]);
    assert!(adapted.atoms.iter().all(|a| a.probability.is_none()));
    assert_eq!(adapted.profile.uncertainty, UncertaintyKind::StructuralEnvelope);
    let decision = evaluate_adapted(&plain(StructuralPolicy::Maximin), &adapted).unwrap();
    // Worst case: A = min(5, 1) = 1 and B = min(3, 4) = 3, so B; counts never weighted.
    assert_eq!(decision.robust.verdict, RobustVerdict::WorstCaseChoice("B".into()));
    assert!(decision.structural.actions.iter().all(|a| a.weighted_value.is_none()));
    assert_eq!(decision.completion_counts.len(), 2);

    // Probabilities for some claims and not others is malformed.
    let mixed = vec![
        exact("g1", ClaimProbability::Genuine(0.5), 5.0, 3.0),
        exact("g2", ClaimProbability::Unspecified, 1.0, 4.0),
    ];
    let error = adapt_weighted_graph_atoms(mixed, StructuralPolicy::Maximin).unwrap_err();
    assert!(matches!(error, AdapterError::InvalidClaims(_)));
    assert_eq!(error.to_refusal().detail, "decision_adapters.invalid_claims");
}

#[test]
fn f6_adapter_finite_scenarios_with_declared_weights() {
    // Declared weights 0.6 and 0.4 as probabilities: A = 3 + 0.4 = 3.4 and
    // B = 1.8 + 0.8 = 2.6.
    let claims = vec![
        exact("s1", ClaimProbability::Genuine(0.6), 5.0, 3.0),
        exact("s2", ClaimProbability::Genuine(0.4), 1.0, 2.0),
    ];
    let adapted = adapt_finite_scenarios(claims, StructuralPolicy::BayesOverStructures).unwrap();
    assert_eq!(adapted.kind, ClaimKind::FiniteScenarios);
    // Scenarios are an envelope, never a posterior.
    assert_eq!(adapted.profile.uncertainty, UncertaintyKind::StructuralEnvelope);
    let decision =
        evaluate_adapted(&plain(StructuralPolicy::BayesOverStructures), &adapted).unwrap();
    assert!(matches!(
        decision.robust.verdict,
        RobustVerdict::BayesChoice { ref action, .. } if action == "A"
    ));

    let credible = contract(
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::BayesOverStructures,
        AdmissibilityRules {
            uncertainty: UncertaintyRequirement::Credible,
            ..AdmissibilityRules::default()
        },
    );
    let decision = evaluate_adapted(&credible, &adapted).unwrap();
    assert!(matches!(decision.robust.verdict, RobustVerdict::InsufficientClaims(_)));

    // Unevaluated scenario mass is carried: 0.5 stays unevaluated.
    let truncated = vec![
        exact("s1", ClaimProbability::Genuine(0.5), 5.0, 3.0),
        claim("s2", ClaimProbability::Genuine(0.5), AtomEvidence::Unevaluated("budget".into())),
    ];
    let adapted = adapt_finite_scenarios(truncated, StructuralPolicy::BayesOverStructures).unwrap();
    let decision =
        evaluate_adapted(&plain(StructuralPolicy::BayesOverStructures), &adapted).unwrap();
    assert!(near(decision.structural.unevaluated_mass.unwrap(), 0.5));
    assert!(near(decision.structural.evaluated_mass.unwrap(), 0.5));
    assert!(matches!(decision.robust.verdict, RobustVerdict::InsufficientClaims(_)));
}

#[test]
fn f6_adapter_point_claim_is_a_point_not_an_envelope() {
    let adapted = adapt_point_claim(
        "p",
        AtomEvidence::Evaluated(law(5.0, 3.0)),
        AtomSupport::supported(),
        StructuralPolicy::RequireInvariantBestAction,
    )
    .unwrap();
    assert_eq!(adapted.kind, ClaimKind::Point);
    assert_eq!(adapted.profile.uncertainty, UncertaintyKind::Point);
    let point_only = contract(
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::RequireInvariantBestAction,
        AdmissibilityRules {
            uncertainty: UncertaintyRequirement::PointOnly,
            ..AdmissibilityRules::default()
        },
    );
    let decision = evaluate_adapted(&point_only, &adapted).unwrap();
    assert_eq!(decision.robust.verdict, RobustVerdict::StructurallyRobust("A".into()));
    let envelope = contract(
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::RequireInvariantBestAction,
        AdmissibilityRules {
            uncertainty: UncertaintyRequirement::StructuralEnvelope,
            ..AdmissibilityRules::default()
        },
    );
    let decision = evaluate_adapted(&envelope, &adapted).unwrap();
    assert!(matches!(decision.robust.verdict, RobustVerdict::InsufficientClaims(_)));
}

#[test]
fn f6_adapter_unsupported_atom_keeps_its_support_status_beyond_unevaluated() {
    let s2 = SuppliedClaim {
        id: "g2".into(),
        probability: ClaimProbability::Unspecified,
        evidence: AtomEvidence::Evaluated(law(6.0, 2.0)),
        support: AtomSupport {
            overall: SupportStatus::Supported,
            per_input: vec![InputSupport {
                action_id: "B".into(),
                input: 0,
                status: SupportStatus::Extrapolative,
            }],
        },
    };
    let claims = vec![exact("g1", ClaimProbability::Unspecified, 5.0, 3.0), s2];
    let adapted =
        adapt_finite_scenarios(claims, StructuralPolicy::RequireInvariantBestAction).unwrap();
    let rules = AdmissibilityRules {
        default_weakest_support: Some(SupportStatus::WeakOverlap),
        ..AdmissibilityRules::default()
    };
    let c = contract(
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::RequireInvariantBestAction,
        rules,
    );
    let decision = evaluate_adapted(&c, &adapted).unwrap();
    // Both atoms were evaluated; the support shortfall is retained separately.
    assert!(decision.structural.atoms.iter().all(|a| matches!(a.status, AtomStatus::Evaluated(_))));
    assert_eq!(
        decision.robust.shortfalls,
        vec![SupportShortfall {
            atom: "g2".into(),
            action: "B".into(),
            input: 0,
            status: SupportStatus::Extrapolative,
            weakest_allowed: SupportStatus::WeakOverlap,
        }]
    );
    assert_eq!(decision.robust.actions[1].unsupported_in, vec!["g2".to_owned()]);
    assert!(decision.robust.actions[0].unsupported_in.is_empty());
    // A leads everywhere B is licensed to compete, and everywhere it is not.
    assert_eq!(decision.robust.verdict, RobustVerdict::SupportRobust("A".into()));
}

#[test]
fn f6_adapter_no_admissible_action_when_a_declaration_removes_every_action() {
    let claims = vec![exact("g1", ClaimProbability::Unspecified, 5.0, 3.0)];
    let adapted =
        adapt_graph_dependent_claims(claims, StructuralPolicy::RequireInvariantBestAction).unwrap();
    let rules = AdmissibilityRules {
        declared_exclusions: vec![
            DeclaredExclusion { action_id: "A".into(), reason: "legal".into() },
            DeclaredExclusion { action_id: "B".into(), reason: "cost".into() },
        ],
        ..AdmissibilityRules::default()
    };
    let c = contract(
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::RequireInvariantBestAction,
        rules,
    );
    let decision = evaluate_adapted(&c, &adapted).unwrap();
    assert_eq!(decision.robust.verdict, RobustVerdict::NoAdmissibleAction);
}

#[test]
fn f6_adapter_policy_and_claim_shape_mismatches_refuse() {
    let claims = vec![
        exact("g1", ClaimProbability::Unspecified, 5.0, 3.0),
        exact("g2", ClaimProbability::Unspecified, 1.0, 4.0),
    ];
    let adapted = adapt_graph_dependent_claims(claims.clone(), StructuralPolicy::Maximin).unwrap();
    let error = evaluate_adapted(&plain(StructuralPolicy::RequireInvariantBestAction), &adapted)
        .unwrap_err();
    assert_eq!(error, AdapterError::PolicyMismatch);
    assert_eq!(error.to_refusal().detail, "decision_adapters.policy_mismatch");

    assert!(matches!(
        adapt_graph_dependent_claims(vec![], StructuralPolicy::Maximin),
        Err(AdapterError::InvalidClaims(_))
    ));
    let mut dup = claims.clone();
    dup[1].id = "g1".into();
    assert!(matches!(
        adapt_finite_scenarios(dup, StructuralPolicy::Maximin),
        Err(AdapterError::InvalidClaims(_))
    ));
    let mut blank = claims;
    blank[0].id = " ".into();
    assert!(matches!(
        adapt_finite_scenarios(blank, StructuralPolicy::Maximin),
        Err(AdapterError::InvalidClaims(_))
    ));
}
