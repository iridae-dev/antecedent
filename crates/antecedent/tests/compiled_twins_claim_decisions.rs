//! The compiled decisions over supplied claims, identified sets and declared laws equal the
//! direct kernel calls they replace.
//!
//! Each compiled program (`SuppliedClaimDecision`, `IdentifiedSetDecision`,
//! `ContractLawDecision`, `DecisionFunctionalValue`) calls the same kernel function as the direct
//! entry on the same inputs, so every value and verdict agrees bit for bit (compared both by
//! `PartialEq` and by the `Debug` rendering, which distinguishes every non-NaN bit pattern). Each
//! test also reaches a hand-derived value of the unit's own truth fixtures through the compiled
//! route, and checks the route and unit the program runs under.
//!
//! These decisions have no sampling interval, so there is no calibration basis to compare.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use antecedent::analysis::claim_decision_programs::{
    ContractSource, IdentifiedAction, IdentifiedSetCarrier, SuppliedClaimSet,
    compile_contract_decision, compile_decision_functional, compile_identified_set_decision,
    compile_supplied_claim_decision,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_core::IdentifiedSet;
use antecedent_design::DecisionFunctional;
use antecedent_design::decision_adapters::{
    ClaimProbability, IdentifiedUtility, IdentifiedVerdict, SuppliedClaim, adapt_finite_scenarios,
    adapt_graph_dependent_claims, adapt_point_claim, adapt_weighted_graph_atoms, evaluate_adapted,
    evaluate_identified_sets, utility_interval,
};
use antecedent_design::decision_contract::{
    ActionKind, AdmissibilityRules, AdmissibleDecisionContract, DecisionAction, DecisionContract,
    DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{
    MeanEstimate, MeanSource, Verdict, evaluate_contract, evaluate_contract_on_means,
    evaluate_functional,
};
use antecedent_design::decision_robustness::{AtomSupport, RobustVerdict};
use antecedent_design::decision_structural::{AtomEvidence, StructuralVerdict};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn quantity(variable: &str, regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

/// An exact law over `columns` with equally likely `rows`.
fn law(columns: &[ScientificQuantity], rows: &[&[f64]]) -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        columns,
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
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows.len(), columns.len()],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        rows.iter().flat_map(|row| row.iter().copied()).collect(),
    )
    .unwrap()
}

fn action(id: &str, inputs: Vec<ScientificQuantity>, utility: UtilityExpr) -> DecisionAction {
    DecisionAction { id: id.into(), kind: ActionKind::Intervention, inputs, utility }
}

fn contract_of(
    actions: Vec<DecisionAction>,
    criterion: DecisionCriterion,
    policy: StructuralPolicy,
) -> DecisionContract {
    DecisionContract {
        actions,
        utility_units: "units".into(),
        criterion,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: policy,
    }
}

/// Two actions whose payoffs are the two columns of a structure's law.
fn ab_columns() -> [ScientificQuantity; 2] {
    [quantity("a", "do(a=1)", "outcome"), quantity("b", "do(a=0)", "outcome")]
}

fn ab_contract(
    criterion: DecisionCriterion,
    policy: StructuralPolicy,
) -> AdmissibleDecisionContract {
    let [a, b] = ab_columns();
    AdmissibleDecisionContract {
        contract: contract_of(
            vec![
                action("A", vec![a], UtilityExpr::Input(0)),
                action("B", vec![b], UtilityExpr::Input(0)),
            ],
            criterion,
            policy,
        ),
        rules: AdmissibilityRules::default(),
    }
}

/// One structure's exact law: A pays `a` and B pays `b` on both of two equally likely rows.
fn exact(id: &str, probability: ClaimProbability, a: f64, b: f64) -> SuppliedClaim {
    SuppliedClaim {
        id: id.into(),
        probability,
        evidence: AtomEvidence::Evaluated(Box::new(law(&ab_columns(), &[&[a, b], &[a, b]]))),
        support: AtomSupport::supported(),
    }
}

fn assert_same<T: PartialEq + std::fmt::Debug>(compiled: &T, direct: &T) {
    assert_eq!(compiled, direct);
    assert_eq!(format!("{compiled:?}"), format!("{direct:?}"));
}

fn assert_runs_under(lowered: &antecedent::planner::lowered::LoweredProgram, route: &str) {
    assert_eq!(lowered.route(), route);
}

#[test]
fn compiled_supplied_claims_equal_the_direct_adapters_and_evaluation() {
    let ctx = ExecutionContext::for_tests(7);
    let policy = StructuralPolicy::RequireInvariantBestAction;
    let contract = ab_contract(DecisionCriterion::PosteriorExpectedUtility, policy);
    let graphs = || {
        vec![
            exact("g1", ClaimProbability::Unspecified, 5.0, 3.0),
            exact("g2", ClaimProbability::Unspecified, 6.0, 2.0),
        ]
    };

    let program = compile_supplied_claim_decision(
        contract.clone(),
        SuppliedClaimSet::GraphDependent(graphs()),
        &ctx,
    )
    .unwrap();
    assert_runs_under(program.lowered(), "SuppliedClaimDecision");
    let unit = program.plan_view().unit_support().expect("the family is bound to a unit");
    assert_eq!(unit.unit(), "decision.structural_robustness");
    let compiled = program.execute(&ctx).unwrap();
    let adapted = adapt_graph_dependent_claims(graphs(), policy).unwrap();
    let direct = evaluate_adapted(&contract, &adapted).unwrap();
    assert_same(&compiled.decision, &direct);
    assert_eq!(format!("{:?}", compiled.adapted), format!("{adapted:?}"));

    // Hand-set payoffs: A pays 5 and 6, B pays 3 and 2, so A leads under both graphs and the
    // structural spread of A is kept as {5, 6}, never averaged.
    assert_eq!(compiled.decision.structural.verdict, StructuralVerdict::InvariantBest("A".into()));
    assert_eq!(compiled.decision.robust.verdict, RobustVerdict::StructurallyRobust("A".into()));
    assert_eq!(compiled.decision.structural.actions[0].range, Some((5.0, 6.0)));
}

#[test]
fn every_claim_reading_compiles_to_the_same_adapter_and_evaluation() {
    let ctx = ExecutionContext::for_tests(7);
    let policy = StructuralPolicy::Maximin;
    let contract = ab_contract(DecisionCriterion::PosteriorExpectedUtility, policy);
    let weighted = || {
        vec![
            exact("s1", ClaimProbability::Genuine(0.75), 5.0, 3.0),
            exact("s2", ClaimProbability::Genuine(0.25), 1.0, 4.0),
        ]
    };
    let cases: [(SuppliedClaimSet, _); 3] = [
        (
            SuppliedClaimSet::WeightedGraphAtoms(weighted()),
            adapt_weighted_graph_atoms(weighted(), policy).unwrap(),
        ),
        (
            SuppliedClaimSet::FiniteScenarios(weighted()),
            adapt_finite_scenarios(weighted(), policy).unwrap(),
        ),
        (
            SuppliedClaimSet::Point {
                id: "only".into(),
                evidence: exact("only", ClaimProbability::Unspecified, 2.0, 1.0).evidence,
                support: AtomSupport::supported(),
            },
            adapt_point_claim(
                "only",
                exact("only", ClaimProbability::Unspecified, 2.0, 1.0).evidence,
                AtomSupport::supported(),
                policy,
            )
            .unwrap(),
        ),
    ];
    for (claims, adapted) in cases {
        let program = compile_supplied_claim_decision(contract.clone(), claims, &ctx).unwrap();
        let compiled = program.execute(&ctx).unwrap();
        assert_same(&compiled.decision, &evaluate_adapted(&contract, &adapted).unwrap());
        assert_eq!(format!("{:?}", compiled.adapted), format!("{adapted:?}"));
    }
}

#[test]
fn a_claim_refusal_is_the_adapters_own_error() {
    // Graph-dependent claims never carry probabilities; the compiled program refuses with the
    // adapter's error, exactly as the direct adapter does.
    let ctx = ExecutionContext::for_tests(7);
    let policy = StructuralPolicy::RequireInvariantBestAction;
    let claims = || vec![exact("g1", ClaimProbability::Genuine(0.5), 5.0, 3.0)];
    let program = compile_supplied_claim_decision(
        ab_contract(DecisionCriterion::PosteriorExpectedUtility, policy),
        SuppliedClaimSet::GraphDependent(claims()),
        &ctx,
    )
    .unwrap();
    let compiled = program.execute(&ctx).unwrap_err();
    let direct = adapt_graph_dependent_claims(claims(), policy).unwrap_err();
    assert_eq!(
        format!("{:?}", compiled.kernel().expect("a kernel refusal")),
        format!("{direct:?}")
    );
}

fn interval(id: &str, lower: f64, upper: f64) -> IdentifiedUtility {
    IdentifiedUtility {
        action_id: id.into(),
        utility: IdentifiedSet::try_new(lower, upper).unwrap(),
        hard_exclusions: vec![],
        support: AtomSupport::supported(),
    }
}

#[test]
fn compiled_identified_sets_equal_evaluate_identified_sets() {
    let ctx = ExecutionContext::for_tests(7);
    let sets = || vec![interval("A", 2.0, 8.0), interval("B", 4.0, 6.0)];
    for policy in [
        StructuralPolicy::RequireInvariantBestAction,
        StructuralPolicy::Maximin,
        StructuralPolicy::ReportOnly,
    ] {
        let contract = ab_contract(DecisionCriterion::PosteriorExpectedUtility, policy);
        let program = compile_identified_set_decision(
            contract.clone(),
            IdentifiedSetCarrier::from_utilities(sets()),
            &ctx,
        )
        .unwrap();
        assert_runs_under(program.lowered(), "IdentifiedSetDecision");
        let unit = program.plan_view().unit_support().expect("the family is bound to a unit");
        assert_eq!(unit.unit(), "decision.structural_robustness");
        let compiled = program.execute(&ctx).unwrap();
        assert_same(&compiled, &evaluate_identified_sets(&contract, &sets()).unwrap());
    }

    // A in [2, 8], B in [4, 6]: the lower bound favours B, the upper bound A, so the leaders
    // conflict and neither is necessarily best; the worst cases 2 and 4 make maximin pick B; the
    // maximum regrets are 6 - 2 = 4 and 8 - 4 = 4.
    let invariant = ab_contract(
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::RequireInvariantBestAction,
    );
    let carrier = IdentifiedSetCarrier::from_utilities(sets());
    let result =
        compile_identified_set_decision(invariant, carrier, &ctx).unwrap().execute(&ctx).unwrap();
    assert_eq!(result.lower_leader.as_deref(), Some("B"));
    assert_eq!(result.upper_leader.as_deref(), Some("A"));
    assert!(result.conflicting_leaders);
    assert_eq!(
        result.verdict,
        IdentifiedVerdict::NoNecessarilyBest { possibly_optimal: vec!["A".into(), "B".into()] }
    );
    assert_eq!(result.actions[0].max_regret, Some(4.0));
    assert_eq!(result.actions[1].max_regret, Some(4.0));
    let maximin =
        ab_contract(DecisionCriterion::PosteriorExpectedUtility, StructuralPolicy::Maximin);
    let result = compile_identified_set_decision(
        maximin,
        IdentifiedSetCarrier::from_utilities(sets()),
        &ctx,
    )
    .unwrap()
    .execute(&ctx)
    .unwrap();
    assert_eq!(result.verdict, IdentifiedVerdict::WorstCaseChoice("B".into()));
}

/// T pays `utility` of (benefit, cost) under `do(a=1)`; W pays the benefit under `do(a=0)`.
fn enclosure_contract(utility: UtilityExpr) -> AdmissibleDecisionContract {
    AdmissibleDecisionContract {
        contract: contract_of(
            vec![
                action(
                    "T",
                    vec![
                        quantity("benefit", "do(a=1)", "outcome"),
                        quantity("cost", "do(a=1)", "outcome"),
                    ],
                    utility,
                ),
                action("W", vec![quantity("benefit", "do(a=0)", "outcome")], UtilityExpr::Input(0)),
            ],
            DecisionCriterion::PosteriorExpectedUtility,
            StructuralPolicy::RequireInvariantBestAction,
        ),
        rules: AdmissibilityRules::default(),
    }
}

#[test]
fn a_carrier_declared_by_inputs_encloses_through_utility_interval() {
    let ctx = ExecutionContext::for_tests(7);
    let set = |lower, upper| IdentifiedSet::try_new(lower, upper).unwrap();
    let product_minus_one = UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
        UtilityExpr::Const(1.0),
    );
    let contract = enclosure_contract(product_minus_one);
    let inputs = || vec![set(1.0, 2.0), set(3.0, 4.0)];
    let carrier = IdentifiedSetCarrier::new(vec![IdentifiedAction::Inputs {
        action_id: "T".into(),
        inputs: inputs(),
        hard_exclusions: vec![],
        support: AtomSupport::supported(),
    }]);
    let program = compile_identified_set_decision(contract.clone(), carrier, &ctx).unwrap();
    let utilities = program.utilities(&ctx).unwrap();
    let direct = utility_interval(&contract.contract, "T", &inputs()).unwrap();
    assert_same(&utilities[0].utility, &direct);
    // benefit in [1, 2] and cost in [3, 4]: the products 3, 4, 6, 8 give [3, 8], minus 1.
    assert_eq!(utilities[0].utility, set(2.0, 7.0));

    // min(benefit, -cost) encloses to [-4, -3], so W in [0, 1] is necessarily best.
    let min_expr = UtilityExpr::minimum(
        UtilityExpr::Input(0),
        UtilityExpr::Neg(Box::new(UtilityExpr::Input(1))),
    );
    let contract = enclosure_contract(min_expr);
    let carrier = IdentifiedSetCarrier::new(vec![
        IdentifiedAction::Inputs {
            action_id: "T".into(),
            inputs: inputs(),
            hard_exclusions: vec![],
            support: AtomSupport::supported(),
        },
        IdentifiedAction::Utility(interval("W", 0.0, 1.0)),
    ]);
    let compiled = compile_identified_set_decision(contract.clone(), carrier, &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    let t = IdentifiedUtility {
        utility: utility_interval(&contract.contract, "T", &inputs()).unwrap(),
        ..interval("T", 0.0, 0.0)
    };
    let direct = evaluate_identified_sets(&contract, &[t, interval("W", 0.0, 1.0)]).unwrap();
    assert_same(&compiled, &direct);
    assert_eq!(compiled.verdict, IdentifiedVerdict::NecessarilyBest("W".into()));

    // A wrong input count refuses with the enclosure's own error.
    let carrier = IdentifiedSetCarrier::new(vec![IdentifiedAction::Inputs {
        action_id: "T".into(),
        inputs: vec![set(1.0, 2.0)],
        hard_exclusions: vec![],
        support: AtomSupport::supported(),
    }]);
    let refused = compile_identified_set_decision(contract.clone(), carrier, &ctx)
        .unwrap()
        .utilities(&ctx)
        .unwrap_err();
    let direct = utility_interval(&contract.contract, "T", &[set(1.0, 2.0)]).unwrap_err();
    assert_eq!(format!("{:?}", refused.kernel().expect("a kernel refusal")), format!("{direct:?}"));
}

/// x and y on two equally likely rows (0, 2) and (2, 0), and a constant k = 5.
fn xyk() -> [ScientificQuantity; 3] {
    [
        quantity("x", "do(t=1)", "outcome"),
        quantity("y", "do(t=1)", "outcome"),
        quantity("k", "do(t=0)", "outcome"),
    ]
}

fn xyk_law() -> DistributionArtifact {
    law(&xyk(), &[&[0.0, 2.0, 5.0], &[2.0, 0.0, 5.0]])
}

fn xyk_contract() -> DecisionContract {
    let [x, y, k] = xyk();
    contract_of(
        vec![
            action(
                "A",
                vec![x.clone(), y],
                UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            ),
            action(
                "B",
                vec![k],
                UtilityExpr::sum(
                    UtilityExpr::Const(1.0),
                    UtilityExpr::product(UtilityExpr::Const(0.0), UtilityExpr::Input(0)),
                ),
            ),
            action("X", vec![x], UtilityExpr::Input(0)),
        ],
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::ReportOnly,
    )
}

#[test]
fn compiled_contract_on_a_law_equals_evaluate_contract() {
    let ctx = ExecutionContext::for_tests(7);
    let program =
        compile_contract_decision(xyk_contract(), ContractSource::Law(xyk_law()), &ctx).unwrap();
    assert_runs_under(program.lowered(), "ContractLawDecision");
    let unit = program.plan_view().unit_support().expect("the family is bound to a unit");
    assert_eq!(unit.unit(), "decision.contract");
    let compiled = program.execute(&ctx).unwrap();
    assert_same(&compiled, &evaluate_contract(&xyk_contract(), &xyk_law()).unwrap());

    // A pays x*y = 0 on both rows; B the constant 1; X pays x, 0 or 2, so 1 in expectation and
    // ties B. Only the joint rows give A's 0: independent marginals would give E[x]E[y] = 1.
    assert_eq!(compiled.actions[0].expected_utility, 0.0);
    assert_eq!(compiled.actions[1].expected_utility, 1.0);
    assert_eq!(compiled.actions[2].expected_utility, 1.0);
}

#[test]
fn compiled_contract_on_means_equals_evaluate_contract_on_means() {
    let ctx = ExecutionContext::for_tests(7);
    let coordinate = |regime: &str| quantity("y", regime, "mean");
    let means = MeanSource {
        coordinates: vec![coordinate("do(a=0)"), coordinate("do(a=1)"), coordinate("do(a=2)")],
        means: vec![1.0, 3.0, 5.0],
        provider_id: "lab".into(),
        snapshot_id: "snap-9".into(),
        causal_contract_id: "checked-contract".into(),
        rng_id: "none:mean_grid".into(),
        estimate: MeanEstimate::Frequentist,
    };
    // 2 x - 1 of the mean at a = 0 (wait) and a = 2 (treat).
    let affine = || {
        UtilityExpr::difference(
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(2.0)),
            UtilityExpr::Const(1.0),
        )
    };
    let contract = contract_of(
        vec![
            action("wait", vec![coordinate("do(a=0)")], affine()),
            action("treat", vec![coordinate("do(a=2)")], affine()),
        ],
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::ReportOnly,
    );
    let program =
        compile_contract_decision(contract.clone(), ContractSource::Means(means.clone()), &ctx)
            .unwrap();
    let compiled = program.execute(&ctx).unwrap();
    assert_same(&compiled, &evaluate_contract_on_means(&contract, &means).unwrap());
    // 2 * 1 - 1 = 1 and 2 * 5 - 1 = 9.
    assert_eq!(compiled.actions[0].expected_utility, 1.0);
    assert_eq!(compiled.actions[1].expected_utility, 9.0);
    assert_eq!(compiled.verdict, Verdict::UniquelyOptimal("treat".into()));
}

#[test]
fn compiled_decision_functional_equals_evaluate_functional() {
    let ctx = ExecutionContext::for_tests(7);
    for functional in [DecisionFunctional::Expectation, DecisionFunctional::Variance] {
        let program =
            compile_decision_functional(xyk_contract(), "X", functional.clone(), xyk_law(), &ctx)
                .unwrap();
        assert_runs_under(program.lowered(), "DecisionFunctionalValue");
        let unit = program.plan_view().unit_support().expect("the family is bound to a unit");
        assert_eq!(unit.unit(), "decision.functionals");
        let compiled = program.execute(&ctx).unwrap();
        let direct = evaluate_functional(&xyk_contract(), "X", functional.clone(), &xyk_law());
        assert_same(&compiled, &direct.unwrap());
    }
    // X pays x in {0, 2} with equal weight: E[x] = 1 and Var[x] = 2 - 1 = 1.
    let value = |functional| {
        compile_decision_functional(xyk_contract(), "X", functional, xyk_law(), &ctx)
            .unwrap()
            .execute(&ctx)
            .unwrap()
            .value
    };
    assert_eq!(value(DecisionFunctional::Expectation), 1.0);
    assert_eq!(value(DecisionFunctional::Variance), 1.0);
}

#[test]
fn programs_over_different_declarations_have_different_identities() {
    let ctx = ExecutionContext::for_tests(7);
    let sets = |upper| vec![interval("A", 2.0, upper), interval("B", 4.0, 6.0)];
    let contract =
        ab_contract(DecisionCriterion::PosteriorExpectedUtility, StructuralPolicy::Maximin);
    let identity = |upper| {
        compile_identified_set_decision(
            contract.clone(),
            IdentifiedSetCarrier::from_utilities(sets(upper)),
            &ctx,
        )
        .unwrap()
        .identity()
    };
    assert_eq!(identity(8.0), identity(8.0));
    assert_ne!(identity(8.0), identity(9.0));
}
