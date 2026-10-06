//! Acceptance evidence for the frozen records F4 (decision contract), F5
//! (decision functionals) and F6 (structural decision policy).
//!
//! Every expected number below is enumerated by hand from the frozen text of
//! `parity/promotion_2_3.toml`, never read back from the code under test.

use std::path::{Path, PathBuf};

use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, source_digest,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionContractError, DecisionCriterion,
    DecisionFunctional, HardConstraint, SourceRepresentation, StructuralPolicy, Tail, UtilityExpr,
};
use antecedent_design::decision_eval::{
    DecisionEvalError, DecisionResult, MeanSource, Verdict, evaluate_contract,
    evaluate_contract_on_means, evaluate_functional,
};
use antecedent_design::decision_structural::{
    AtomEvidence, AtomStatus, StructuralAtom, StructuralError, StructuralVerdict,
    evaluate_structural,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

// ---------------------------------------------------------------------------
// Shared builders
// ---------------------------------------------------------------------------

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

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

fn identity(columns: &[ScientificQuantity], alignment: DrawAlignment) -> DistributionIdentity {
    DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        columns,
        alignment,
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

/// An exact finite law: each row is one realization of every column.
fn law(
    columns: &[ScientificQuantity],
    alignment: DrawAlignment,
    weights: Option<Vec<f64>>,
    rows: &[&[f64]],
) -> DistributionArtifact {
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity: identity(columns, alignment),
            axes: ["draw".into(), "quantity".into()],
            shape: [rows.len(), columns.len()],
            weights,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        rows.concat(),
    )
    .unwrap()
}

fn action(id: &str, inputs: Vec<ScientificQuantity>, utility: UtilityExpr) -> DecisionAction {
    DecisionAction { id: id.into(), kind: ActionKind::Intervention, inputs, utility }
}

fn contract_of(
    actions: Vec<DecisionAction>,
    criterion: DecisionCriterion,
    constraints: Vec<HardConstraint>,
    structural_policy: StructuralPolicy,
) -> DecisionContract {
    DecisionContract {
        actions,
        utility_units: "units".into(),
        criterion,
        constraints,
        target_population: "target".into(),
        horizon: 0,
        structural_policy,
    }
}

fn constraint(id: &str, expr: UtilityExpr, bound: f64, applies_to: &[&str]) -> HardConstraint {
    HardConstraint {
        id: id.into(),
        expr,
        bound,
        min_probability: 1.0,
        units: "units".into(),
        applies_to: applies_to.iter().map(|a| (*a).to_owned()).collect(),
    }
}

// ---------------------------------------------------------------------------
// Fresh-process consumer plumbing
// ---------------------------------------------------------------------------

const CONSUMER_DIR: &str = "ANTECEDENT_23_ACCEPTANCE_DIR";

fn consumer_dir() -> Option<PathBuf> {
    std::env::var(CONSUMER_DIR).ok().map(PathBuf::from)
}

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap()
}

fn mark_consumed(dir: &Path) {
    std::fs::write(dir.join("consumed.ok"), b"ok").unwrap();
}

/// Write `files`, re-run this very test in a new process that reads only them,
/// and require that the consumer branch ran to its end.
fn run_fresh_consumer(test: &str, files: &[(String, Vec<u8>)]) {
    let unique =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir()
        .join(format!("antecedent-acceptance-{test}-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, bytes) in files {
        std::fs::write(dir.join(name), bytes).unwrap();
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg(test)
        .env(CONSUMER_DIR, &dir)
        .output()
        .unwrap();
    let consumed = dir.join("consumed.ok").exists();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(consumed, "the fresh-process consumer branch did not run");
}

// ---------------------------------------------------------------------------
// F4: actions A and B, joint states (0,2) and (2,0), utility x*y for A, 1 for B
// ---------------------------------------------------------------------------

fn f4_columns() -> Vec<ScientificQuantity> {
    vec![
        quantity("x", "do(a=A)", "outcome"),
        quantity("y", "do(a=A)", "outcome"),
        quantity("k", "do(a=B)", "outcome"),
    ]
}

/// Rows `(x, y)` = `(0, 2)` and `(2, 0)`, equally weighted; `k` is an arbitrary
/// 5 that B's constant utility ignores.
fn f4_source() -> DistributionArtifact {
    law(&f4_columns(), DrawAlignment::Joint, None, &[&[0.0, 2.0, 5.0], &[2.0, 0.0, 5.0]])
}

fn f4_contract(constraints: Vec<HardConstraint>) -> DecisionContract {
    let c = f4_columns();
    contract_of(
        vec![
            action(
                "A",
                vec![c[0].clone(), c[1].clone()],
                UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            ),
            // The constant 1: `1 + 0 * k` is 1 whatever k is.
            action(
                "B",
                vec![c[2].clone()],
                UtilityExpr::sum(
                    UtilityExpr::Const(1.0),
                    UtilityExpr::product(UtilityExpr::Const(0.0), UtilityExpr::Input(0)),
                ),
            ),
        ],
        DecisionCriterion::PosteriorExpectedUtility,
        constraints,
        StructuralPolicy::ReportOnly,
    )
}

#[test]
fn f4_two_actions_nonlinear_utility_joint_rows_choose_b() {
    let result = evaluate_contract(&f4_contract(vec![]), &f4_source()).unwrap();
    // A: x*y = 0*2 = 0 and 2*0 = 0, so E[U_A] = 0. B: the constant 1.
    assert!(near(result.actions[0].expected_utility, 0.0));
    assert!(near(result.actions[1].expected_utility, 1.0));
    assert!(result.actions.iter().all(|a| a.admissible && a.exclusions.is_empty()));
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("B".into()));
    // B is best in every row, so perfect information adds nothing: 1 - 1 = 0.
    assert!(near(result.evpi.unwrap(), 0.0));
    // A loses 1 in each row; B never loses.
    assert!(near(result.actions[0].expected_regret.unwrap(), 1.0));
    assert!(near(result.actions[1].expected_regret.unwrap(), 0.0));

    // The same marginals paired by independent means would give E[x]E[y] = 1 and
    // a tie; only the joint rows give 0.
    let marginal_mean_product = ((0.0 + 2.0) / 2.0) * ((2.0 + 0.0) / 2.0);
    assert!(near(marginal_mean_product, 1.0));
    assert!(!near(result.actions[0].expected_utility, marginal_mean_product));
}

#[test]
fn f4_wrong_contract_hard_constraint_excludes_without_penalty_and_empty_set_is_explicit() {
    // A constraint on B (k = 5 is never <= 4) excludes B although B's expected
    // utility 1 beats A's 0: the exclusion is not a penalty and B's number stays.
    let excluded_b = f4_contract(vec![constraint("k-cap", UtilityExpr::Input(0), 4.0, &["B"])]);
    let result = evaluate_contract(&excluded_b, &f4_source()).unwrap();
    let b = &result.actions[1];
    assert!(!b.admissible);
    assert_eq!(b.exclusions.len(), 1);
    assert_eq!(b.exclusions[0].constraint_id, "k-cap");
    assert!(near(b.exclusions[0].probability, 0.0));
    assert!(near(b.exclusions[0].required, 1.0));
    assert!(near(b.expected_utility, 1.0));
    assert!(b.expected_regret.is_none());
    assert!(near(result.actions[0].expected_utility, 0.0));
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("A".into()));

    // y <= 1 holds in one of two equally likely rows, so it holds with 1/2 < 1.
    let excluded_a = f4_contract(vec![constraint("y-cap", UtilityExpr::Input(1), 1.0, &["A"])]);
    let result = evaluate_contract(&excluded_a, &f4_source()).unwrap();
    assert!(!result.actions[0].admissible);
    assert!(near(result.actions[0].exclusions[0].probability, 0.5));
    assert!(near(result.actions[0].expected_utility, 0.0));
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("B".into()));

    // A bound nothing satisfies, applied to every action: no admissible action
    // is an explicit verdict, with no perfect-information value.
    let none = f4_contract(vec![constraint("floor", UtilityExpr::Input(0), -100.0, &[])]);
    let result = evaluate_contract(&none, &f4_source()).unwrap();
    assert_eq!(result.verdict, Verdict::NoAdmissibleAction);
    assert!(result.evpi.is_none());
    assert!(result.actions.iter().all(|a| !a.admissible));
}

fn f4_cap() -> HardConstraint {
    // y <= 2 holds in both rows, so A stays admissible and the receipt is kept.
    constraint("y-cap", UtilityExpr::Input(1), 2.0, &["A"])
}

#[test]
fn f4_artifact_fresh_process_consumer_recomputes_result_from_receipts() {
    if let Some(dir) = consumer_dir() {
        // The consumer re-declares the contract and the source identity from
        // constants; it trusts none of the producer's declarations.
        let contract = f4_contract(vec![f4_cap()]);
        let contract_id = contract.identity().unwrap();
        let contract_artifact =
            DecisionContractArtifact::from_bytes(&read(&dir, "contract.bin"), &contract_id)
                .unwrap();
        assert_eq!(contract_artifact.contract(), &contract);
        let source = DistributionArtifact::from_bytes(
            &read(&dir, "source.bin"),
            &identity(&f4_columns(), DrawAlignment::Joint),
        )
        .unwrap();
        let stored = DecisionResultArtifact::from_bytes(
            &read(&dir, "result.bin"),
            &contract_id,
            &source_digest(&source),
        )
        .unwrap();
        stored.replay(contract_artifact.contract(), &source).unwrap();
        let result = stored.result();
        assert!(near(result.actions[0].expected_utility, 0.0));
        assert!(near(result.actions[1].expected_utility, 1.0));
        assert_eq!(result.verdict, Verdict::UniquelyOptimal("B".into()));
        assert!(result.actions[0].admissible);
        mark_consumed(&dir);
        return;
    }
    let contract = f4_contract(vec![f4_cap()]);
    let source = f4_source();
    let result =
        DecisionResultArtifact::new(evaluate_contract(&contract, &source).unwrap(), &source);
    run_fresh_consumer(
        "f4_artifact_fresh_process_consumer_recomputes_result_from_receipts",
        &[
            (
                "contract.bin".to_owned(),
                DecisionContractArtifact::new(contract).unwrap().to_bytes("c").unwrap(),
            ),
            ("source.bin".to_owned(), source.to_bytes("s").unwrap()),
            ("result.bin".to_owned(), result.to_bytes("r").unwrap()),
        ],
    );
}

#[test]
fn f4_artifact_refuses_a_stored_number_or_constraint_changed_under_the_same_identities() {
    let contract = f4_contract(vec![f4_cap()]);
    let source = f4_source();
    let result = evaluate_contract(&contract, &source).unwrap();
    let artifact = DecisionResultArtifact::new(result.clone(), &source);
    artifact.replay(&contract, &source).unwrap();

    let mut tampered: DecisionResult = result.clone();
    tampered.actions[1].expected_utility = 2.0;
    assert!(DecisionResultArtifact::new(tampered, &source).replay(&contract, &source).is_err());
    let mut flipped = result;
    flipped.verdict = Verdict::UniquelyOptimal("A".into());
    assert!(DecisionResultArtifact::new(flipped, &source).replay(&contract, &source).is_err());

    // A different constraint is a different contract: the stored result is not its answer.
    let tighter = f4_contract(vec![constraint("y-cap", UtilityExpr::Input(1), 1.0, &["A"])]);
    assert_ne!(tighter.identity().unwrap(), contract.identity().unwrap());
    assert!(artifact.replay(&tighter, &source).is_err());
}

// ---------------------------------------------------------------------------
// F5: finite law, mass 1/4 at 0 and 3/4 at 2
// ---------------------------------------------------------------------------

fn f5_columns(functional: &str) -> Vec<ScientificQuantity> {
    vec![quantity("y", "do(a=A)", functional), quantity("k", "do(a=B)", functional)]
}

/// `P(Y = 0) = 1/4` and `P(Y = 2) = 3/4`, as raw weights 1 and 3.
fn f5_source() -> DistributionArtifact {
    law(
        &f5_columns("outcome"),
        DrawAlignment::Joint,
        Some(vec![1.0, 3.0]),
        &[&[0.0, 1.0], &[2.0, 1.0]],
    )
}

/// `Y`, `Y * Y` (for the variance identity) and a reference action.
fn f5_actions(functional: &str) -> Vec<DecisionAction> {
    let c = f5_columns(functional);
    vec![
        action("Y", vec![c[0].clone()], UtilityExpr::Input(0)),
        action(
            "Y2",
            vec![c[0].clone()],
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0)),
        ),
        action("ref", vec![c[1].clone()], UtilityExpr::Input(0)),
    ]
}

fn f5_contract(criterion: DecisionCriterion) -> DecisionContract {
    contract_of(f5_actions("outcome"), criterion, vec![], StructuralPolicy::ReportOnly)
}

fn f5_value(criterion: DecisionCriterion) -> f64 {
    evaluate_contract(&f5_contract(criterion), &f5_source()).unwrap().actions[0].value
}

#[test]
fn f5_positive_expectation_and_variance_identity_on_the_finite_law() {
    let result =
        evaluate_contract(&f5_contract(DecisionCriterion::PosteriorExpectedUtility), &f5_source())
            .unwrap();
    // E[Y] = 0 * 1/4 + 2 * 3/4 = 3/2 and E[Y^2] = 0 * 1/4 + 4 * 3/4 = 3.
    let mean = result.actions[0].expected_utility;
    let second_moment = result.actions[1].expected_utility;
    assert!(near(mean, 1.5));
    assert!(near(second_moment, 3.0));
    // The identity route: Var(Y) = E[Y^2] - E[Y]^2 = 3 - 9/4 = 3/4.
    let identity_variance = second_moment - mean * mean;
    assert!(near(identity_variance, 0.75));
    // An exact law carries no sampling error.
    assert!(near(result.actions[0].standard_error.unwrap(), 0.0));

    // The direct evaluators agree with the hand enumeration and the identity.
    let contract = f5_contract(DecisionCriterion::PosteriorExpectedUtility);
    let expectation =
        evaluate_functional(&contract, "Y", DecisionFunctional::Expectation, &f5_source()).unwrap();
    assert!(near(expectation.value, 1.5));
    assert!(near(expectation.standard_error.unwrap(), 0.0));
    assert_eq!(expectation.source_mode.representation, SourceRepresentation::JointDraws);
    let utility =
        evaluate_functional(&contract, "Y", DecisionFunctional::ExpectedUtility, &f5_source())
            .unwrap();
    assert!(near(utility.value, 1.5));
    let variance =
        evaluate_functional(&contract, "Y", DecisionFunctional::Variance, &f5_source()).unwrap();
    assert!(near(variance.value, 0.75));
    assert!(near(variance.value, identity_variance));
    // E[Y^2] = 3 through the direct route as well.
    let squared =
        evaluate_functional(&contract, "Y2", DecisionFunctional::Expectation, &f5_source())
            .unwrap();
    assert!(near(squared.value, 3.0));
}

#[test]
fn f5_positive_threshold_probability_on_the_finite_law() {
    // The criterion is P(Y >= 1); the law has no mass at 1, so it equals
    // P(Y > 1) = P(Y = 2) = 3/4.
    assert!(near(f5_value(DecisionCriterion::ThresholdProbability { threshold: 1.0 }), 0.75));
    // Mass at 0 only: P(Y >= -1) = 1; nothing reaches 3: P(Y >= 3) = 0.
    assert!(near(f5_value(DecisionCriterion::ThresholdProbability { threshold: -1.0 }), 1.0));
    assert!(near(f5_value(DecisionCriterion::ThresholdProbability { threshold: 3.0 }), 0.0));

    // The direct functional: P(Y >= 1) = P(Y >= 1.5) = P(Y > 1) = 3/4 and
    // P(Y <= 1) = 1/4 (the law has no mass strictly between 0 and 2).
    let contract = f5_contract(DecisionCriterion::PosteriorExpectedUtility);
    let probability = |threshold: f64, tail: Tail| {
        evaluate_functional(
            &contract,
            "Y",
            DecisionFunctional::Probability { threshold, tail },
            &f5_source(),
        )
        .unwrap()
        .value
    };
    assert!(near(probability(1.0, Tail::Upper), 0.75));
    assert!(near(probability(1.5, Tail::Upper), 0.75));
    assert!(near(probability(1.0, Tail::Lower), 0.25));
    // The atom at the threshold is included on both sides.
    assert!(near(probability(2.0, Tail::Upper), 0.75));
    assert!(near(probability(0.0, Tail::Lower), 0.25));
    assert!(near(probability(2.0, Tail::Lower), 1.0));
}

#[test]
fn f5_positive_quantile_is_the_left_inverse_of_the_cdf() {
    // F(0) = 1/4 and F(2) = 1, so F^-1(u) = inf{x : F(x) >= u} is 0 for u <= 1/4
    // and 2 above it.
    assert!(near(f5_value(DecisionCriterion::Quantile { p: 0.5 }), 2.0));
    assert!(near(f5_value(DecisionCriterion::Quantile { p: 0.25 }), 0.0));
    assert!(near(f5_value(DecisionCriterion::Quantile { p: 0.26 }), 2.0));
    assert!(near(f5_value(DecisionCriterion::Quantile { p: 0.999 }), 2.0));

    // The direct functional returns the same left inverse; at the atom boundary
    // F(0) = 1/4 reaches u = 1/4 exactly, so the quantile is exactly 0.
    let contract = f5_contract(DecisionCriterion::PosteriorExpectedUtility);
    let quantile = |p: f64| {
        evaluate_functional(&contract, "Y", DecisionFunctional::Quantile { p }, &f5_source())
            .unwrap()
            .value
    };
    assert!(near(quantile(0.5), 2.0));
    assert_eq!(quantile(0.25).to_bits(), 0.0_f64.to_bits());
    assert!(near(quantile(0.26), 2.0));
    assert!(near(quantile(0.999), 2.0));
    assert!(near(quantile(0.01), 0.0));
}

#[test]
fn f5_positive_functionals_without_an_evaluator_state_only_their_source_requirements() {
    use SourceRepresentation as S;
    let y = UtilityExpr::Input(0);
    let y_squared = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0));

    // The selected tail rule is the fractional-boundary tail mean: the lowest
    // (or highest) mass `p`, with the boundary atom split, divided by `p`.
    let contract = f5_contract(DecisionCriterion::PosteriorExpectedUtility);
    let tail_mean = |p: f64, tail: Tail| {
        evaluate_functional(
            &contract,
            "Y",
            DecisionFunctional::TailExpectation { p, tail },
            &f5_source(),
        )
        .unwrap()
        .value
    };
    // Mass 1/4 at 0 and 3/4 at 2.
    assert!(near(tail_mean(0.25, Tail::Lower), 0.0));
    assert!(near(tail_mean(0.25, Tail::Upper), 2.0));
    // Lower 1/2 = all of the atom at 0 plus 1/4 of the atom at 2: (0 + 0.5) / 0.5.
    assert!(near(tail_mean(0.5, Tail::Lower), 1.0));
    assert!(near(tail_mean(0.5, Tail::Upper), 2.0));
    // Lower 3/4 = (0.25 * 0 + 0.5 * 2) / 0.75 = 4/3.
    assert!(near(tail_mean(0.75, Tail::Lower), 4.0 / 3.0));

    // What each source must supply.
    let variance = DecisionFunctional::Variance;
    assert_eq!(
        variance.requirement(&y).any_of,
        vec![S::MeanAndCovariance, S::JointDraws, S::MarginalDraws]
    );
    assert_eq!(variance.requirement(&y_squared).any_of, vec![S::JointDraws, S::MarginalDraws]);
    let tail = DecisionFunctional::TailExpectation { p: 0.25, tail: Tail::Upper };
    assert_eq!(tail.requirement(&y).any_of, vec![S::JointDraws, S::MarginalDraws]);
    assert!(tail.requirement(&y).sampled_needs_error_receipt);
    let lower = DecisionFunctional::TailExpectation { p: 0.25, tail: Tail::Lower };
    assert_eq!(lower.requirement(&y), tail.requirement(&y));

    // The evaluated functionals and their criteria.
    let probability = DecisionFunctional::Probability { threshold: 1.0, tail: Tail::Upper };
    assert_eq!(probability.requirement(&y).any_of, vec![S::Cdf, S::JointDraws, S::MarginalDraws]);
    let quantile = DecisionFunctional::Quantile { p: 0.5 };
    assert_eq!(
        quantile.requirement(&y).any_of,
        vec![S::QuantileFunction, S::Cdf, S::JointDraws, S::MarginalDraws]
    );
    assert_eq!(DecisionCriterion::Quantile { p: 0.5 }.functional(), Some(quantile));
    assert_eq!(
        DecisionCriterion::ThresholdProbability { threshold: 1.0 }.functional(),
        Some(probability)
    );
    assert_eq!(DecisionCriterion::Regret.functional(), None);

    // A mean alone satisfies neither the quantile nor the probability requirement.
    assert!(matches!(
        quantile.requirement(&y).check(&[S::Mean]),
        Err(DecisionContractError::MissingSource { .. })
    ));
    assert!(matches!(
        probability.requirement(&y).check(&[S::Mean]),
        Err(DecisionContractError::MissingSource { .. })
    ));
}

#[test]
fn f5_evaluate_functional_refuses_unknown_action_bad_parameters_and_marginals() {
    let contract = f5_contract(DecisionCriterion::PosteriorExpectedUtility);
    let source = f5_source();
    let error = evaluate_functional(&contract, "ghost", DecisionFunctional::Expectation, &source)
        .unwrap_err();
    assert_eq!(
        error,
        DecisionEvalError::Contract(DecisionContractError::UnknownAction("ghost".into()))
    );
    for functional in [
        DecisionFunctional::Quantile { p: 0.0 },
        DecisionFunctional::Quantile { p: 1.0 },
        DecisionFunctional::Quantile { p: f64::NAN },
        DecisionFunctional::TailExpectation { p: 1.5, tail: Tail::Lower },
        DecisionFunctional::Probability { threshold: f64::INFINITY, tail: Tail::Upper },
    ] {
        assert_eq!(
            evaluate_functional(&contract, "Y", functional, &source).unwrap_err(),
            DecisionEvalError::Contract(DecisionContractError::InvalidParameter("functional"))
        );
    }
    // Independent marginals cannot answer a functional of a joint utility.
    let c = f5_columns("outcome");
    let rows: [&[f64]; 2] = [&[0.0, 1.0], &[2.0, 1.0]];
    let marginals = law(&c, DrawAlignment::IndependentMarginals, None, &rows);
    let error =
        evaluate_functional(&contract, "Y", DecisionFunctional::Variance, &marginals).unwrap_err();
    assert!(matches!(error, DecisionEvalError::JointLawRequired { .. }));
}

#[test]
fn f5_exact_law_has_zero_standard_error_and_a_sample_does_not() {
    let contract = f5_contract(DecisionCriterion::PosteriorExpectedUtility);
    let exact =
        evaluate_functional(&contract, "Y", DecisionFunctional::Expectation, &f5_source()).unwrap();
    assert!(near(exact.standard_error.unwrap(), 0.0));
    let exact_variance =
        evaluate_functional(&contract, "Y", DecisionFunctional::Variance, &f5_source()).unwrap();
    assert!(near(exact_variance.standard_error.unwrap(), 0.0));

    // The same rows as a draw sample (calibration unmeasured), repeated so the
    // empirical law is the finite law: mass 1/4 at 0 and 3/4 at 2.
    let columns = f5_columns("outcome");
    let rows: [&[f64]; 4] = [&[0.0, 1.0], &[2.0, 1.0], &[2.0, 1.0], &[2.0, 1.0]];
    let sampled = DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity: identity(&columns, DrawAlignment::Joint),
            axes: ["draw".into(), "quantity".into()],
            shape: [rows.len(), columns.len()],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        rows.concat(),
    )
    .unwrap();
    let mean =
        evaluate_functional(&contract, "Y", DecisionFunctional::Expectation, &sampled).unwrap();
    assert!(near(mean.value, 1.5));
    // Population variance 3/4 over 4 effective draws: se = sqrt(3/16).
    assert!(near(mean.standard_error.unwrap(), (0.75_f64 / 4.0).sqrt()));
    assert!(mean.standard_error.unwrap() > 0.0);
    let variance =
        evaluate_functional(&contract, "Y", DecisionFunctional::Variance, &sampled).unwrap();
    assert!(near(variance.value, 0.75));
    assert_eq!(variance.standard_error, None);
}

fn mean_contract(criterion: DecisionCriterion) -> DecisionContract {
    let c = f5_columns("mean");
    contract_of(
        vec![
            action("Y", vec![c[0].clone()], UtilityExpr::Input(0)),
            action("ref", vec![c[1].clone()], UtilityExpr::Input(0)),
        ],
        criterion,
        vec![],
        StructuralPolicy::ReportOnly,
    )
}

fn mean_source() -> MeanSource {
    MeanSource {
        coordinates: f5_columns("mean"),
        // E[Y] = 3/2, as the finite law's mean; the reference mean is 1.
        means: vec![1.5, 1.0],
        provider_id: "lab".into(),
        snapshot_id: "snap".into(),
        causal_contract_id: "checked-contract".into(),
        rng_id: "none:mean_grid".into(),
    }
}

#[test]
fn f5_negative_mean_only_source_cannot_answer_quantile_or_threshold_probability() {
    // Control: the same mean source answers an expectation of an affine utility.
    let result = evaluate_contract_on_means(
        &mean_contract(DecisionCriterion::PosteriorExpectedUtility),
        &mean_source(),
    )
    .unwrap();
    assert!(near(result.actions[0].expected_utility, 1.5));

    for criterion in [
        DecisionCriterion::Quantile { p: 0.5 },
        DecisionCriterion::ThresholdProbability { threshold: 1.0 },
    ] {
        let error =
            evaluate_contract_on_means(&mean_contract(criterion), &mean_source()).unwrap_err();
        let DecisionEvalError::MeanSourceInsufficient { needed } = &error else {
            panic!("expected MeanSourceInsufficient, got {error:?}");
        };
        assert!(needed.contains(&SourceRepresentation::JointDraws));
        let refusal = error.to_refusal();
        assert_eq!(refusal.code, "decision_contract_unsatisfied");
        assert_eq!(refusal.stage, "evaluate");
        assert_eq!(refusal.detail, "decision_evaluation.mean_source_insufficient");
        assert_eq!(refusal.supplied.as_deref(), Some("mean"));
        assert!(refusal.remedy.is_some());
    }
}

#[test]
fn f5_negative_independent_marginals_cannot_answer_a_nonlinear_joint_utility() {
    let c = vec![
        quantity("y1", "do(a=A)", "outcome"),
        quantity("y2", "do(a=A)", "outcome"),
        quantity("k", "do(a=B)", "outcome"),
    ];
    let contract = contract_of(
        vec![
            action(
                "Y1Y2",
                vec![c[0].clone(), c[1].clone()],
                UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            ),
            action("ref", vec![c[2].clone()], UtilityExpr::Input(0)),
        ],
        DecisionCriterion::PosteriorExpectedUtility,
        vec![],
        StructuralPolicy::ReportOnly,
    );
    // Rows (0,2) and (2,0): as paired realizations E[Y1*Y2] = 0, while the
    // product of the marginal means would be 1 * 1 = 1.
    let rows: [&[f64]; 2] = [&[0.0, 2.0, 1.0], &[2.0, 0.0, 1.0]];
    let joint = evaluate_contract(&contract, &law(&c, DrawAlignment::Joint, None, &rows)).unwrap();
    assert!(near(joint.actions[0].expected_utility, 0.0));

    let marginals = law(&c, DrawAlignment::IndependentMarginals, None, &rows);
    let error = evaluate_contract(&contract, &marginals).unwrap_err();
    assert_eq!(
        error,
        DecisionEvalError::JointLawRequired {
            action: Some("Y1Y2".into()),
            supplied_alignment: "independent_marginals",
        }
    );
    assert_eq!(error.reason_code(), "joint_law_required");
    let refusal = error.to_refusal();
    assert_eq!(refusal.code, "joint_law_required");
    assert_eq!(refusal.stage, "evaluate");
    assert_eq!(refusal.detail, "decision_evaluation.joint_law_required");
    assert_eq!(refusal.offending.as_deref(), Some("Y1Y2"));
    assert_eq!(refusal.expected.as_deref(), Some("joint"));
    assert_eq!(refusal.supplied.as_deref(), Some("independent_marginals"));
    assert!(refusal.remedy.is_some());
}

#[test]
fn f5_artifact_fresh_process_consumer_retains_functional_convention_law_and_units() {
    let cases = [
        ("eu", DecisionCriterion::PosteriorExpectedUtility, 1.5),
        ("threshold", DecisionCriterion::ThresholdProbability { threshold: 1.0 }, 0.75),
        ("quantile", DecisionCriterion::Quantile { p: 0.5 }, 2.0),
    ];
    if let Some(dir) = consumer_dir() {
        let source = DistributionArtifact::from_bytes(
            &read(&dir, "source.bin"),
            &identity(&f5_columns("outcome"), DrawAlignment::Joint),
        )
        .unwrap();
        // The input law is retained: the raw weights 1 and 3 are 1/4 and 3/4.
        assert_eq!(source.metadata().weights, Some(vec![1.0, 3.0]));
        assert_eq!(source.metadata().shape, [2, 2]);
        for (name, criterion, expected) in cases {
            let contract = f5_contract(criterion);
            let contract_id = contract.identity().unwrap();
            let contract_artifact = DecisionContractArtifact::from_bytes(
                &read(&dir, &format!("{name}.contract.bin")),
                &contract_id,
            )
            .unwrap();
            // Functional kind and its threshold or quantile level survive.
            assert_eq!(contract_artifact.contract().criterion, criterion);
            assert_eq!(contract_artifact.contract().utility_units, "units");
            let stored = DecisionResultArtifact::from_bytes(
                &read(&dir, &format!("{name}.result.bin")),
                &contract_id,
                &source_digest(&source),
            )
            .unwrap();
            stored.replay(contract_artifact.contract(), &source).unwrap();
            assert_eq!(stored.result().criterion, criterion);
            assert!(near(stored.result().actions[0].value, expected), "{name}");
        }
        mark_consumed(&dir);
        return;
    }
    let source = f5_source();
    let mut files = vec![("source.bin".to_owned(), source.to_bytes("s").unwrap())];
    let mut identities = Vec::new();
    for (name, criterion, _) in cases {
        let contract = f5_contract(criterion);
        identities.push(contract.identity().unwrap());
        let result =
            DecisionResultArtifact::new(evaluate_contract(&contract, &source).unwrap(), &source);
        files.push((
            format!("{name}.contract.bin"),
            DecisionContractArtifact::new(contract).unwrap().to_bytes("c").unwrap(),
        ));
        files.push((format!("{name}.result.bin"), result.to_bytes("r").unwrap()));
    }
    // The functional is part of the contract identity.
    identities.sort();
    identities.dedup();
    assert_eq!(identities.len(), 3);
    run_fresh_consumer(
        "f5_artifact_fresh_process_consumer_retains_functional_convention_law_and_units",
        &files,
    );
}

// ---------------------------------------------------------------------------
// F6: two structural atoms, utilities A = (2, -1) and B = (0, 1), no probabilities
// ---------------------------------------------------------------------------

fn f6_columns() -> Vec<ScientificQuantity> {
    vec![quantity("a", "do(a=1)", "outcome"), quantity("b", "do(a=0)", "outcome")]
}

/// One structure's exact law: A's and B's payoffs in two equal rows.
fn f6_law(a: f64, b: f64) -> Box<DistributionArtifact> {
    Box::new(law(&f6_columns(), DrawAlignment::Joint, None, &[&[a, b], &[a, b]]))
}

fn f6_atom(id: &str, probability: Option<f64>, evidence: AtomEvidence) -> StructuralAtom {
    StructuralAtom { id: id.into(), probability, evidence }
}

fn f6_evaluated(id: &str, probability: Option<f64>, a: f64, b: f64) -> StructuralAtom {
    f6_atom(id, probability, AtomEvidence::Evaluated(f6_law(a, b)))
}

/// Atom `s1` has utilities `(A, B) = (2, 0)`, atom `s2` has `(-1, 1)`.
fn f6_atoms() -> [StructuralAtom; 2] {
    [f6_evaluated("s1", None, 2.0, 0.0), f6_evaluated("s2", None, -1.0, 1.0)]
}

fn f6_contract(criterion: DecisionCriterion, policy: StructuralPolicy) -> DecisionContract {
    let c = f6_columns();
    contract_of(
        vec![
            action("A", vec![c[0].clone()], UtilityExpr::Input(0)),
            action("B", vec![c[1].clone()], UtilityExpr::Input(0)),
        ],
        criterion,
        vec![],
        policy,
    )
}

#[test]
fn f6_positive_maximin_chooses_b_with_worst_cases_zero_against_minus_one() {
    let policies = [
        (DecisionCriterion::PosteriorExpectedUtility, StructuralPolicy::Maximin),
        (DecisionCriterion::MaximinOverStructures, StructuralPolicy::ReportOnly),
    ];
    for (criterion, policy) in policies {
        let result = evaluate_structural(&f6_contract(criterion, policy), &f6_atoms()).unwrap();
        assert_eq!(result.verdict, StructuralVerdict::WorstCaseChoice("B".into()));
        // A: (2, -1) has minimum -1; B: (0, 1) has minimum 0.
        assert_eq!(result.actions[0].per_atom, vec![Some(2.0), Some(-1.0)]);
        assert_eq!(result.actions[1].per_atom, vec![Some(0.0), Some(1.0)]);
        assert_eq!(result.actions[0].range, Some((-1.0, 2.0)));
        assert_eq!(result.actions[1].range, Some((0.0, 1.0)));
        // No genuine probabilities: no weighted value and no masses.
        assert!(result.actions.iter().all(|a| a.weighted_value.is_none()));
        assert!(result.evaluated_mass.is_none() && result.unevaluated_mass.is_none());
    }
}

#[test]
fn f6_positive_report_only_shows_graph_dependence_and_no_weighted_winner() {
    let contract =
        f6_contract(DecisionCriterion::PosteriorExpectedUtility, StructuralPolicy::ReportOnly);
    let result = evaluate_structural(&contract, &f6_atoms()).unwrap();
    assert_eq!(result.verdict, StructuralVerdict::ReportOnly);
    assert!(result.actions.iter().all(|a| a.weighted_value.is_none()));
    assert!(result.actions.iter().all(|a| a.mass_where_best.is_none()));
    // The structures disagree: each prefers the other's action.
    let leader = |index: usize| -> Verdict {
        let AtomStatus::Evaluated(inner) = &result.atoms[index].status else {
            panic!("atom {index} was not evaluated");
        };
        inner.verdict.clone()
    };
    assert_eq!(leader(0), Verdict::UniquelyOptimal("A".into()));
    assert_eq!(leader(1), Verdict::UniquelyOptimal("B".into()));

    // Requiring an invariant best action finds none.
    let invariant = f6_contract(
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::RequireInvariantBestAction,
    );
    let result = evaluate_structural(&invariant, &f6_atoms()).unwrap();
    assert_eq!(
        result.verdict,
        StructuralVerdict::NoInvariantBest(vec![
            ("s1".into(), vec!["A".into()]),
            ("s2".into(), vec!["B".into()]),
        ])
    );
}

#[test]
fn f6_negative_bayes_over_structures_refuses_without_atom_probabilities() {
    let bayes = f6_contract(
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::BayesOverStructures,
    );
    let error = evaluate_structural(&bayes, &f6_atoms()).unwrap_err();
    assert_eq!(error, StructuralError::ProbabilitiesRequired);
    assert_eq!(error.reason_code(), "decision_contract_unsatisfied");
    let refusal = error.to_refusal();
    assert_eq!(refusal.stage, "structural");
    assert_eq!(refusal.detail, "decision_structural.probabilities_required");
    assert!(refusal.remedy.is_some());

    // Probabilities for some atoms only are not genuine probabilities either.
    let partial = [f6_evaluated("s1", Some(0.5), 2.0, 0.0), f6_evaluated("s2", None, -1.0, 1.0)];
    assert_eq!(
        evaluate_structural(&bayes, &partial).unwrap_err(),
        StructuralError::InvalidProbabilities
    );
}

#[test]
fn f6_negative_unevaluated_mass_is_reported_and_never_normalized_away() {
    let atoms = [
        f6_evaluated("s1", Some(0.5), 2.0, 0.0),
        f6_evaluated("s2", Some(0.25), -1.0, 1.0),
        f6_atom("s3", Some(0.25), AtomEvidence::Unevaluated("budget".into())),
    ];
    let bayes = f6_contract(
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::BayesOverStructures,
    );
    let result = evaluate_structural(&bayes, &atoms).unwrap();
    assert!(near(result.evaluated_mass.unwrap(), 0.75));
    assert!(near(result.unevaluated_mass.unwrap(), 0.25));
    assert!(near(result.unidentified_mass.unwrap(), 0.0));
    // A: 0.5 * 2 + 0.25 * (-1) = 3/4; B: 0.5 * 0 + 0.25 * 1 = 1/4. Dividing by the
    // evaluated mass would give 1 and 1/3; the values stay unnormalized.
    assert!(near(result.actions[0].weighted_value.unwrap(), 0.75));
    assert!(near(result.actions[1].weighted_value.unwrap(), 0.25));
    assert!(near(result.actions[0].mass_where_best.unwrap(), 0.5));
    assert!(near(result.actions[1].mass_where_best.unwrap(), 0.25));
    match &result.verdict {
        StructuralVerdict::BayesChoice { action, evaluated_mass } => {
            assert_eq!(action, "A");
            assert!(near(*evaluated_mass, 0.75));
        }
        other => panic!("expected BayesChoice, got {other:?}"),
    }

    // A worst case over structures needs every structure; it does not skip one.
    let maximin =
        f6_contract(DecisionCriterion::PosteriorExpectedUtility, StructuralPolicy::Maximin);
    let result = evaluate_structural(&maximin, &atoms).unwrap();
    assert!(matches!(result.verdict, StructuralVerdict::InsufficientScience(_)));
}

#[test]
fn f6_contract_artifact_retains_the_declared_robustness_policy_only() {
    // There is no artifact for a structural result yet: atom identities and
    // unidentified mass are not retained by any wire. The contract artifact
    // does retain the declared policy and criterion, which is all this asserts.
    let cases = [
        (DecisionCriterion::PosteriorExpectedUtility, StructuralPolicy::Maximin),
        (DecisionCriterion::PosteriorExpectedUtility, StructuralPolicy::ReportOnly),
        (DecisionCriterion::PosteriorExpectedUtility, StructuralPolicy::BayesOverStructures),
        (DecisionCriterion::MaximinOverStructures, StructuralPolicy::ReportOnly),
    ];
    let mut identities = Vec::new();
    for (criterion, policy) in cases {
        let contract = f6_contract(criterion, policy);
        let artifact = DecisionContractArtifact::new(contract.clone()).unwrap();
        let loaded = DecisionContractArtifact::from_bytes(
            &artifact.to_bytes("f6").unwrap(),
            artifact.identity(),
        )
        .unwrap();
        assert_eq!(loaded.contract(), &contract);
        assert_eq!(loaded.contract().structural_policy, policy);
        assert_eq!(loaded.contract().criterion, criterion);
        identities.push(artifact.identity().to_owned());
    }
    identities.sort();
    identities.dedup();
    assert_eq!(identities.len(), cases.len());
}
