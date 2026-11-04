//! External provider results consumed by production computation, against hand-derived values.
//!
//! * A bound response grid (risk of a binary outcome at three doses) drives the point-only
//!   decision engine; each action's utility is worked out by hand from the bound values.
//! * A Bernoulli(1/4) law supplied by a provider is consumed as a finite weighted law by the
//!   library's mean, probability, quantile, tail and information functionals, and as a seeded
//!   sample; every expected value is the hand value of the functional of the declared law.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    CheckedCausalContract, DistributionMeaning, ExternalCapability, ExternalCapabilityRequest,
    ExternalContractError, ExternalResponse, ExternalResult, ExternalResultHeader,
    ExternalScientificObject, ExternalTrustState, ExternalUncertaintyMeaning, LawProviderContract,
    ProviderObjectIdentity, QuantityRole, ScientificQuantity, bind_external_result,
};
use antecedent_core::{CausalRng, IdentificationStatus, SupportStatus};
use antecedent_design::composition_verifiers::mean_source_of;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, DecisionFunctional,
    StructuralPolicy, Tail, UtilityExpr,
};
use antecedent_design::decision_eval::{
    DecisionEvalError, Verdict, evaluate_contract, evaluate_contract_on_means, evaluate_functional,
};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::external_claim_artifact::ExternalClaimArtifact;
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

fn near(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-12, "{what}: {actual} vs {expected}");
}

// ---------------------------------------------------------------------------------------------
// Response binding
// ---------------------------------------------------------------------------------------------

fn risk(dose: u32) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:y".into(),
        variable_name: "Y".into(),
        role: QuantityRole::Outcome,
        units: "probability".into(),
        population_id: "target".into(),
        regime_id: format!("do(a={dose})"),
        horizon: 0,
        functional_id: "risk".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

/// Bind the risk grid `values` on doses 0, 1, 2 and return the claim.
fn bound_grid(values: [f64; 3]) -> ExternalClaimArtifact {
    let quantities: Vec<ScientificQuantity> = (0..3).map(risk).collect();
    let identity = ProviderObjectIdentity {
        provider_id: "lab".into(),
        object_id: "risk-grid".into(),
        version_id: "v1".into(),
        snapshot_id: "snap-1".into(),
        request_id: "req-1".into(),
    };
    let object = ExternalScientificObject::Law(LawProviderContract {
        identity,
        quantities: quantities.clone(),
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![ExternalCapability::Mean],
    });
    let contract = CheckedCausalContract {
        graph_id: "graph:binary".into(),
        identification: IdentificationStatus::NonparametricallyIdentified,
        estimand: quantities.clone(),
        accepted_meanings: vec![DistributionMeaning::InterventionalPredictive],
        required_evidence_ids: vec![],
        required_assumption_ids: vec![],
        equivalences: vec![],
    };
    let response = ExternalResponse {
        header: ExternalResultHeader {
            object,
            graph_id: "graph:binary".into(),
            quantities,
            evidence_ids: vec![],
            assumption_ids: vec![],
            trust: ExternalTrustState::ExternallyAttested { attestor: "lab".into() },
        },
        values: values.to_vec(),
        uncertainty: ExternalUncertaintyMeaning::None,
        point_support: Some(vec![SupportStatus::Supported; 3]),
    };
    let bound = bind_external_result(&contract, &ExternalResult::Response(response)).unwrap();
    ExternalClaimArtifact::from_bound_claim(&bound, "binary-contract").unwrap()
}

fn action(id: &str, inputs: Vec<ScientificQuantity>, utility: UtilityExpr) -> DecisionAction {
    DecisionAction { id: id.into(), kind: ActionKind::Intervention, inputs, utility }
}

/// `scale * x - cost` of the action's first input.
fn linear(scale: f64, cost: f64) -> UtilityExpr {
    UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(scale)),
        UtilityExpr::Const(cost),
    )
}

fn grid_decision() -> DecisionContract {
    DecisionContract {
        actions: vec![
            action("none", vec![risk(0)], linear(80.0, 10.0)),
            action("full", vec![risk(1)], linear(80.0, 30.0)),
            action("half", vec![risk(2)], linear(80.0, 12.0)),
            // 100 * (risk under do(a=1) - risk under do(a=0)) - 5: a contrast of two bound doses.
            action(
                "switch",
                vec![risk(1), risk(0)],
                UtilityExpr::difference(
                    UtilityExpr::product(
                        UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Input(1)),
                        UtilityExpr::Const(100.0),
                    ),
                    UtilityExpr::Const(5.0),
                ),
            ),
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

/// The bound grid P(Y=1 | do(a)) = 1/4, 3/4, 1/2 on doses 0, 1, 2. By hand the utilities are
///
/// * none   = 80 * 1/4 - 10          = 10
/// * full   = 80 * 3/4 - 30          = 30
/// * half   = 80 * 1/2 - 12          = 28
/// * switch = 100 * (3/4 - 1/4) - 5  = 45
///
/// so the contrast of two bound doses wins. Changing only the dose-1 risk to 1/2 gives
/// full = 10 and switch = 100 * (1/2 - 1/4) - 5 = 20, so half (28) wins.
#[test]
fn a_bound_response_grid_gives_the_hand_derived_decision() {
    let claim = bound_grid([0.25, 0.75, 0.5]);
    assert_eq!(claim.values(), &[0.25, 0.75, 0.5]);
    near(claim.values()[1] - claim.values()[0], 0.5, "risk difference");

    let source = mean_source_of(&claim).unwrap();
    let result = evaluate_contract_on_means(&grid_decision(), &source).unwrap();
    let utility = |id: &str| result.actions.iter().find(|a| a.id == id).unwrap().expected_utility;
    near(utility("none"), 10.0, "none");
    near(utility("full"), 30.0, "full");
    near(utility("half"), 28.0, "half");
    near(utility("switch"), 45.0, "switch");
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("switch".into()));
    assert_eq!(result.evpi, None);
    assert_eq!(result.source.provider_id, "lab");
    assert_eq!(result.source.snapshot_id, "snap-1");
    assert_eq!(result.source.causal_contract_id, "binary-contract");

    let shifted = bound_grid([0.25, 0.5, 0.5]);
    let source = mean_source_of(&shifted).unwrap();
    let result = evaluate_contract_on_means(&grid_decision(), &source).unwrap();
    let utility = |id: &str| result.actions.iter().find(|a| a.id == id).unwrap().expected_utility;
    near(utility("none"), 10.0, "none");
    near(utility("full"), 10.0, "full");
    near(utility("half"), 28.0, "half");
    near(utility("switch"), 20.0, "switch");
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("half".into()));
}

/// A dose the grid did not bind has no value: an action reading do(a=3) is refused naming the
/// action and the input, never answered by extending or interpolating the bound doses.
#[test]
fn a_dose_outside_the_bound_grid_is_not_answered() {
    let source = mean_source_of(&bound_grid([0.25, 0.75, 0.5])).unwrap();
    let mut contract = grid_decision();
    contract.actions[2] = action("extrapolate", vec![risk(3)], linear(80.0, 12.0));
    let error = evaluate_contract_on_means(&contract, &source).unwrap_err();
    assert!(matches!(
        error,
        DecisionEvalError::QuantityNotFound { ref action, input: 0 } if action == "extrapolate"
    ));
}

// ---------------------------------------------------------------------------------------------
// Scientific providers
// ---------------------------------------------------------------------------------------------

fn provider_identity() -> ProviderObjectIdentity {
    ProviderObjectIdentity {
        provider_id: "lab".into(),
        object_id: "bernoulli".into(),
        version_id: "v1".into(),
        snapshot_id: "enumeration".into(),
        request_id: "req-1".into(),
    }
}

fn outcome() -> ScientificQuantity {
    ScientificQuantity { regime_id: "do(a=1)".into(), ..common::quantity("y") }
}

fn bernoulli_artifact(
    weights: Option<Vec<f64>>,
    draws: Vec<f64>,
    calibration: DistributionCalibration,
) -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &[outcome()],
        DrawAlignment::Joint,
        DistributionProvenance::from_provider(
            &provider_identity(),
            "study",
            "deterministic_exact",
            "checked",
        ),
    )
    .unwrap();
    let rows = draws.len();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows, 1],
            weights,
            supported: None,
            calibration,
            trust: DistributionTrust::ExternalAttested,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

/// Action A pays `4y - 1`, action B pays `1 - y`.
fn payoff() -> DecisionContract {
    DecisionContract {
        actions: vec![
            action("A", vec![outcome()], linear(4.0, 1.0)),
            action(
                "B",
                vec![outcome()],
                UtilityExpr::difference(UtilityExpr::Const(1.0), UtilityExpr::Input(0)),
            ),
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

/// The provider declares Bernoulli(1/4) with sample and mean only. Negotiation admits those two
/// and refuses CDF and quantile; the enumerated law {0: 3/4, 1: 1/4} it stands for is then consumed
/// by the library's functionals.
///
/// By hand, A's utility `4y - 1` is -1 with probability 3/4 and 3 with probability 1/4:
///
/// * E = -3/4 + 3/4 = 0 and Var = 3/4 * 1 + 1/4 * 9 = 3;
/// * P(U <= -1) = 3/4 and P(U >= 3) = 1/4;
/// * quantiles `inf {x : F(x) >= p}`: -1 at p = 1/2 and p = 3/4, 3 at p = 0.76;
/// * the upper-half tail mean takes the whole mass 1/4 at 3 and 1/4 of the mass at -1:
///   (1/4 * 3 + 1/4 * -1) / (1/2) = 1, the lower-half tail mean is -1;
///
/// B's expected utility is 1 - 1/4 = 3/4. Row-wise the better action pays 1 (y = 0) and 3
/// (y = 1), so E[max] = 3/4 + 3/4 = 3/2, EVPI = 3/2 - 3/4 = 3/4, and Kish ESS of the weights is
/// 1 / (9/16 + 1/16) = 8/5.
#[test]
fn a_negotiated_bernoulli_law_is_consumed_by_the_library_functionals() {
    let law = ExternalScientificObject::Law(LawProviderContract {
        identity: provider_identity(),
        quantities: vec![outcome()],
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![ExternalCapability::Sample, ExternalCapability::Mean],
    });
    let ask =
        |operation| ExternalCapabilityRequest { expected_identity: provider_identity(), operation };
    assert!(law.negotiate(&ask(ExternalCapability::Sample)).is_ok());
    assert!(law.negotiate(&ask(ExternalCapability::Mean)).is_ok());
    for missing in [ExternalCapability::Cdf, ExternalCapability::Quantile] {
        assert_eq!(
            law.negotiate(&ask(missing)),
            Err(ExternalContractError::MissingCapability(missing))
        );
    }

    let exact =
        bernoulli_artifact(Some(vec![0.75, 0.25]), vec![0.0, 1.0], DistributionCalibration::Exact);
    near(exact.mean(0).unwrap(), 0.25, "mean of the law");
    near(exact.interventional_probability_above(0, 0.0).unwrap(), 0.25, "P(Y > 0)");
    near(exact.interventional_probability_above(0, 1.0).unwrap(), 0.0, "P(Y > 1)");

    let contract = payoff();
    let value = |functional| {
        let result = evaluate_functional(&contract, "A", functional, &exact).unwrap();
        assert_eq!(result.standard_error, Some(0.0));
        result.value
    };
    near(value(DecisionFunctional::Expectation), 0.0, "E[U]");
    near(value(DecisionFunctional::Variance), 3.0, "Var[U]");
    near(
        value(DecisionFunctional::Probability { threshold: -1.0, tail: Tail::Lower }),
        0.75,
        "P(U <= -1)",
    );
    near(
        value(DecisionFunctional::Probability { threshold: 3.0, tail: Tail::Upper }),
        0.25,
        "P(U >= 3)",
    );
    near(value(DecisionFunctional::Quantile { p: 0.5 }), -1.0, "median");
    near(value(DecisionFunctional::Quantile { p: 0.75 }), -1.0, "quantile at an atom boundary");
    near(value(DecisionFunctional::Quantile { p: 0.76 }), 3.0, "quantile above the boundary");
    near(
        value(DecisionFunctional::TailExpectation { p: 0.5, tail: Tail::Upper }),
        1.0,
        "upper tail mean",
    );
    near(
        value(DecisionFunctional::TailExpectation { p: 0.5, tail: Tail::Lower }),
        -1.0,
        "lower tail mean",
    );

    let decision = evaluate_contract(&contract, &exact).unwrap();
    near(decision.actions[0].expected_utility, 0.0, "A");
    near(decision.actions[1].expected_utility, 0.75, "B");
    near(decision.actions[0].expected_regret.unwrap(), 1.5, "regret of A");
    near(decision.actions[1].expected_regret.unwrap(), 0.75, "regret of B");
    near(decision.evpi.unwrap(), 0.75, "EVPI");
    near(decision.effective_draws, 1.6, "Kish ESS");
    assert_eq!(decision.verdict, Verdict::UniquelyOptimal("B".into()));
    assert_eq!(decision.source.provider_id, provider_identity().provider_id);
    assert_eq!(decision.source.snapshot_id, provider_identity().snapshot_id);
}

/// The provider's seeded sample (the declared `sample` operation) is consumed as an equally
/// weighted law. Its mean, `P(Y > 0)` and expected utility of A estimate the hand values 1/4, 1/4
/// and 0; with n = 20000 draws the standard deviation of the mean is `sqrt(3/16 / n)`, and the
/// band is five of them (and 4 times five for A, whose utility is 4y - 1).
#[test]
fn a_seeded_sample_from_the_provider_estimates_the_hand_values_within_five_sigma() {
    let n = 20_000_usize;
    let mut rng = CausalRng::from_seed(11);
    let draws: Vec<f64> = (0..n).map(|_| f64::from(u8::from(rng.next_f64() >= 0.75))).collect();
    let sample = bernoulli_artifact(None, draws, DistributionCalibration::Unmeasured);
    let band = 5.0 * (3.0 / 16.0 / n as f64).sqrt();
    let mean = sample.mean(0).unwrap();
    assert!((mean - 0.25).abs() <= band, "sample mean {mean}");
    let above = sample.interventional_probability_above(0, 0.0).unwrap();
    assert!((above - 0.25).abs() <= band, "P(Y>0) {above}");
    let expectation =
        evaluate_functional(&payoff(), "A", DecisionFunctional::Expectation, &sample).unwrap();
    assert!(expectation.value.abs() <= 4.0 * band, "E[U_A] {}", expectation.value);
    assert_eq!(expectation.standard_error, None);
    // Another seed is another stream.
    let mut other = CausalRng::from_seed(12);
    let again: Vec<f64> = (0..n).map(|_| f64::from(u8::from(other.next_f64() >= 0.75))).collect();
    assert_ne!(sample.draws(), again.as_slice());
}
