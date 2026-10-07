//! 2.3 C4: the multi-source acceptance story through the Rust facades.
//!
//! This is the Rust half of `python/tests/test_c4_multi_source.py`, with the same hand
//! derivation, so both surfaces assert the same numbers. Program: outcome `y` (mmHg) under
//! `do(a = d)` for `d` in `{0, 1, 2}` mg in the `target` population. Sources: a native mean
//! response `N`, an attested point-mean study `E1` and a second study `E2` that is bound as
//! point means and carries two equally likely aligned joint draws.
//!
//! Mean grids `(m0, m1, m2)`: `N = (2, 4, 9)` with dose 2 outside empirical support (withheld),
//! `E1 = (1, 3, 5.5)`, `E2 = (2, 3.5, 5)` (the means of its two joint rows). Actions: `wait` is
//! worth `m0`, `treat` is `m1 - 1` and `extend` is `m2 - 3`.
//!
//! * `N` alone: wait 2, treat `4 - 1 = 3`, extend unsupported (no coordinate), so treat.
//! * `N` then `E1`: wait 2 (N), treat 3 (N), extend `5.5 - 3 = 2.5` (E1): treat is uniquely
//!   optimal at 3 and no regret or EVPI exists across sources.
//!
//! `E2` rows `(y0, y1, y2, theta)`: `(2, 6, 3, 1/4)` and `(2, 1, 7, 3/4)`, so the per-row
//! utilities are `(2, 5, 0)` and `(2, 0, 4)`. `E[U] = (2, 2.5, 2)`, `E[max U] = (5 + 4) / 2 = 4.5`
//! and `EVPI = 4.5 - 2.5 = 2`. Expected regrets: wait `(3 + 2) / 2 = 2.5`, treat
//! `(0 + 4) / 2 = 2`, extend `(5 + 0) / 2 = 2.5`. `P(U_treat >= 5) = 1/2`; the 3/4-quantile of
//! `U_treat` over `{0, 5}` is 5 (`F(0) = 1/2 < 3/4`). With dose 2 masked unsupported only wait and
//! treat are scored: `E[max(2, U_treat)] = (5 + 2) / 2 = 3.5`, `EVPI = 3.5 - 2.5 = 1`.
//!
//! Study ranking. The follow-up decision bets on the response rate `theta` (the law's state
//! column, prior draws `{1/4, 3/4}`): abstain pays 0 and bet pays `theta - 1/2`, both worth 0
//! now, and `EVPI = E[max(0, theta - 1/2)] = (0 + 1/4) / 2 = 1/8`. Two Bernoulli trials give
//! `P(k) = (10, 12, 10) / 32` and `E[theta | k] = (0.3, 0.5, 0.7)`; only `k = 2` makes the bet
//! worth taking, so `EVSI = 10/32 * 0.2 = 1/16`. A native binomial signal and an equivalent
//! attested posterior share it; with study costs 0.02 and 0.05 the net values are `0.0425` and
//! `0.0125`.
//!
//! The native claim enters composition with a native execution record here (Rust can state
//! one); the Python half cannot, which is why it carries no native label.

use std::sync::Arc;

use antecedent::analysis::composition::{
    BundleBuilder, BundleStage, ClaimLabel, ConsumedBundle, EvidenceRelationship, FACT_LAW,
    LAW_JOINT, LAW_MEAN_ONLY, NodeKind, NodeStatus, ProviderOrData, SuppliedSources, consume,
    export_bundle, mean_result_to_bytes, mean_source_of,
};
use antecedent::analysis::design_ranking::{self, DesignRankingRequestWire};
use antecedent::analysis::native_claims::{
    NativeDecisionSource, NativeResponseClaim, NativeResponseContext,
};
use antecedent_core::{
    AssumptionSet, CausalResponse, CausalSchema, CausalSchemaBuilder, CheckedCausalContract,
    ContinuousDomain, DistributionMeaning, ExternalCapability, ExternalResponse, ExternalResult,
    ExternalResultHeader, ExternalScientificObject, ExternalTrustState, ExternalUncertaintyMeaning,
    GridSpec, IdentificationStatus, LawProviderContract, ProgramBinding, ProviderObjectIdentity,
    QuantityRole, ResponseFunctional, ResponseIdentification, ResponseUncertainty, ResponseValue,
    ScientificQuantity, SupportRegion, SupportReport, SupportStatus, bind_external_result,
};
use antecedent_design::composition_boundary::{
    ActionStatus, BoundaryError, CompositionOperation, CoordinateIssue, DecisionInput,
    DependenceRoute, EvidenceRelation, NativeExecutionRecord, PairRelation, ProviderKind,
    RouteKind, SupportPolicy, SupportedDecision, SupportedVerdict, TrustEvidence, TrustRequirement,
    UnsupportedReason, check_composition, evaluate_functional_on_input, evaluate_with_support,
};
use antecedent_design::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, source_digest,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, DecisionFunctional,
    StructuralPolicy, Tail, UtilityExpr,
};
use antecedent_design::decision_eval::{
    MeanSource, Verdict, evaluate_contract, evaluate_contract_on_means,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::external_claim_artifact::ExternalClaimArtifact;
use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use serde_json::{Value, json};

const CONTRACT: &str = "contract:c";
const NATIVE_SNAPSHOT: &str = "snap-n";
const NATIVE_DIGEST: &str = "digest-snap-n";
const IDS: [&str; 3] = ["wait", "treat", "extend"];
const LAW_ROWS: [[f64; 4]; 2] = [[2.0, 6.0, 3.0, 0.25], [2.0, 1.0, 7.0, 0.75]];

fn close(left: f64, right: f64) {
    assert!((left - right).abs() < 1e-9, "{left} != {right}");
}

// ---------------------------------------------------------------------- coordinates

fn quantity(dose: u32, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "target".into(),
        regime_id: format!("do(a={dose})"),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn state_quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "response_rate".into(),
        variable_name: "response_rate".into(),
        role: QuantityRole::Outcome,
        units: "dimensionless".into(),
        population_id: "target".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "state".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn observation_quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "trial_successes".into(),
        functional_id: "count".into(),
        ..state_quantity()
    }
}

// ------------------------------------------------------------------------- contracts

fn action(id: &str, dose: u32, functional: &str, utility: UtilityExpr) -> DecisionAction {
    DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![quantity(dose, functional)],
        utility,
    }
}

fn contract_of(actions: Vec<DecisionAction>, units: &str) -> DecisionContract {
    DecisionContract {
        actions,
        utility_units: units.into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

/// wait: `x0`; treat: `x0 - 1`; extend: `x0 - 3`, over `functional` coordinates.
fn contract(functional: &str) -> DecisionContract {
    let minus = |c: f64| UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(c));
    contract_of(
        vec![
            action("wait", 0, functional, UtilityExpr::Input(0)),
            action("treat", 1, functional, minus(1.0)),
            action("extend", 2, functional, minus(3.0)),
        ],
        "util",
    )
}

fn rollout_contract() -> DecisionContract {
    let state = |utility| DecisionAction {
        id: String::new(),
        kind: ActionKind::Intervention,
        inputs: vec![state_quantity()],
        utility,
    };
    let abstain = DecisionAction {
        id: "abstain".into(),
        ..state(UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)))
    };
    let bet = DecisionAction {
        id: "bet".into(),
        ..state(UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(0.5)))
    };
    contract_of(vec![abstain, bet], "utility")
}

// ----------------------------------------------------------------------- native claim

fn schema() -> CausalSchema {
    CausalSchemaBuilder::new()
        .continuous("a")
        .treatment()
        .continuous("y")
        .unit("mmHg")
        .outcome()
        .build()
        .unwrap()
}

fn program() -> ProgramBinding {
    ProgramBinding {
        graph_id: "graph:g".into(),
        contract_id: CONTRACT.into(),
        treatment_id: "a".into(),
        outcome_id: "y".into(),
        population_id: "target".into(),
        intervention_kind: "do".into(),
        horizon: 0,
        dose_grid: vec![0.0, 1.0, 2.0],
        dose_units: "mg".into(),
        outcome_units: "mmHg".into(),
        functional_id: "mean".into(),
        transform_id: "identity".into(),
    }
}

fn native_response() -> CausalResponse {
    let schema = schema();
    let grid = [0.0, 1.0, 2.0];
    CausalResponse {
        estimand: ResponseFunctional::MeanCurve {
            outcome: schema.id_of("y").unwrap(),
            treatment: ContinuousDomain::new(
                schema.id_of("a").unwrap(),
                GridSpec::Values(Arc::from(grid.to_vec())),
            ),
        },
        identification_status: IdentificationStatus::NonparametricallyIdentified,
        estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
            grid: Arc::from(grid.to_vec()),
            dimension: 1,
            mean: Arc::from(vec![2.0, 4.0, 9.0]),
        }),
        uncertainty: ResponseUncertainty::None,
        support: SupportReport {
            status: SupportStatus::OutsideEmpiricalSupport,
            query_region: SupportRegion {
                minima: Arc::from(vec![0.0]),
                maxima: Arc::from(vec![2.0]),
            },
            diagnostics: vec![],
            warnings: vec![],
            point_status: Some(Arc::from(vec![
                SupportStatus::Supported,
                SupportStatus::Supported,
                SupportStatus::OutsideEmpiricalSupport,
            ])),
        },
        assumptions: AssumptionSet::new(),
        provenance_id: "op-n".into(),
        horizon_identification: None,
        interaction_structurally_zero: false,
    }
}

fn native_context() -> NativeResponseContext {
    NativeResponseContext {
        program: program(),
        snapshot_id: NATIVE_SNAPSHOT.into(),
        rng_id: "none:point_estimate".into(),
        calibration: DistributionCalibration::PointOnly,
    }
}

fn attested(attestor: &str) -> TrustEvidence {
    TrustEvidence::External(ExternalTrustState::ExternallyAttested { attestor: attestor.into() })
}

fn mean_source(provider: &str, snapshot: &str, means: &[f64]) -> MeanSource {
    MeanSource {
        coordinates: (0..3).map(|dose| quantity(dose, "mean")).collect(),
        means: means.to_vec(),
        provider_id: provider.into(),
        snapshot_id: snapshot.into(),
        causal_contract_id: CONTRACT.into(),
        rng_id: "none:mean_grid".into(),
    }
}

/// The native claim as a composition input: supported coordinates only, with a native record.
fn native_input() -> DecisionInput {
    let claim =
        NativeResponseClaim::from_response(&native_response(), &schema(), &native_context())
            .unwrap();
    let requirement = contract("mean").source_requirement().unwrap().unwrap();
    let input = claim.into_decision_source(&requirement).unwrap();
    let NativeDecisionSource::Mean(source) = input.source else {
        panic!("a mean requirement is answered by a mean source");
    };
    let record = TrustEvidence::NativeExecution(NativeExecutionRecord {
        execution_id: "exec-n".into(),
        provider_id: source.provider_id.clone(),
        snapshot_id: source.snapshot_id.clone(),
    });
    DecisionInput::from_mean_source(
        "native",
        source,
        &input.point_status,
        &record,
        TrustRequirement::Native,
    )
    .unwrap()
}

fn mean_input(id: &str, provider: &str, snapshot: &str, means: &[f64]) -> DecisionInput {
    DecisionInput::from_mean_source(
        id,
        mean_source(provider, snapshot, means),
        &[SupportStatus::Supported; 3],
        &attested(provider),
        TrustRequirement::Unrestricted,
    )
    .unwrap()
}

fn e1_input() -> DecisionInput {
    mean_input("e1", "lab-1", "snap-e1", &[1.0, 3.0, 5.5])
}

fn e2_mean_input() -> DecisionInput {
    mean_input("e2mean", "lab-2", "snap-e2", &[2.0, 3.5, 5.0])
}

// -------------------------------------------------------------------------- the law

fn law_columns() -> Vec<ScientificQuantity> {
    vec![quantity(0, "outcome"), quantity(1, "outcome"), quantity(2, "outcome"), state_quantity()]
}

fn law(provider: &str, snapshot: &str, mask: Option<Vec<bool>>) -> DistributionArtifact {
    let columns = law_columns();
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: format!("study-{snapshot}"),
            provider_id: provider.into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: snapshot.into(),
            causal_contract_id: CONTRACT.into(),
        },
    )
    .unwrap();
    let draws: Vec<f64> = LAW_ROWS.iter().flatten().copied().collect();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [2, 4],
            weights: None,
            supported: mask,
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn e2_law_input(mask: Option<Vec<bool>>) -> DecisionInput {
    DecisionInput::from_distribution_artifact(
        "e2law",
        &law("lab-2", "snap-e2", mask),
        &attested("lab-2"),
        TrustRequirement::Unrestricted,
    )
    .unwrap()
}

fn law_contract() -> DecisionContract {
    contract("outcome")
}

fn compare(contract: &DecisionContract, inputs: &[DecisionInput]) -> SupportedDecision {
    evaluate_with_support(contract, inputs, &SupportPolicy::compare_supported()).unwrap()
}

// ----------------------------------------------------------- tests: the native claim

#[test]
fn c4_acceptance_native_claim_is_native_and_withholds_the_unsupported_dose() {
    let claim =
        NativeResponseClaim::from_response(&native_response(), &schema(), &native_context())
            .unwrap();
    assert_eq!(claim.means(), &[2.0, 4.0, 9.0][..]);
    assert_eq!(claim.support_status(), SupportStatus::OutsideEmpiricalSupport);
    assert_eq!(claim.trust(), DistributionTrust::NativeLicensed);
    assert_eq!(claim.program_identity(), program().identity());
    let expected: Vec<ScientificQuantity> = (0..3).map(|dose| quantity(dose, "mean")).collect();
    assert_eq!(claim.coordinates(), &expected[..]);
    assert!(!claim.has_joint_law());

    let requirement = contract("mean").source_requirement().unwrap().unwrap();
    let input = claim.into_decision_source(&requirement).unwrap();
    assert_eq!(input.coordinates, vec![quantity(0, "mean"), quantity(1, "mean")]);
    assert_eq!(input.withheld.len(), 1);
    assert_eq!(input.withheld[0].index, 2);
    assert_eq!(input.withheld[0].status, SupportStatus::OutsideEmpiricalSupport);

    let composed = native_input();
    assert_eq!(composed.provenance().provider_kind, ProviderKind::Native);
    assert_eq!(composed.provenance().trust, DistributionTrust::NativeLicensed);
}

// ------------------------------------------------- tests: per-action support

#[test]
fn c4_acceptance_an_unsupported_action_is_reported_and_the_rest_are_compared() {
    let decision = compare(&contract("mean"), &[native_input()]);
    assert_eq!(decision.outcomes.len(), 2);
    close(decision.outcomes[0].expected_utility, 2.0);
    close(decision.outcomes[1].expected_utility, 4.0 - 1.0);
    assert_eq!(
        decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into()))
    );
    assert_eq!(
        decision.dispositions[2].status,
        ActionStatus::Unsupported {
            reasons: vec![UnsupportedReason {
                input_id: "native".into(),
                coordinate: 0,
                issue: CoordinateIssue::NotInSource,
            }],
        }
    );
    let strict =
        evaluate_with_support(&contract("mean"), &[native_input()], &SupportPolicy::require_all())
            .unwrap_err();
    assert_eq!(strict, BoundaryError::UnsupportedActions { actions: vec!["extend".into()] });
}

#[test]
fn c4_acceptance_another_source_answers_the_action_the_native_claim_cannot() {
    let decision = compare(&contract("mean"), &[native_input(), e1_input()]);
    close(decision.outcomes[0].expected_utility, 2.0);
    close(decision.outcomes[1].expected_utility, 3.0);
    close(decision.outcomes[2].expected_utility, 5.5 - 3.0);
    assert_eq!(
        decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into()))
    );
    let answered: Vec<Option<&str>> =
        decision.dispositions.iter().map(|d| d.input_id.as_deref()).collect();
    assert_eq!(answered, vec![Some("native"), Some("native"), Some("e1")]);
    assert_eq!(decision.evpi, None, "sources are compared by value, never state by state");
    assert!(decision.outcomes.iter().all(|o| o.expected_regret.is_none()));

    let alone = compare(&contract("mean"), &[e1_input()]);
    close(alone.outcomes[0].expected_utility, 1.0);
    close(alone.outcomes[1].expected_utility, 2.0);
    close(alone.outcomes[2].expected_utility, 2.5);
    assert_eq!(
        alone.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("extend".into()))
    );
    let means = compare(&contract("mean"), &[e2_mean_input()]);
    close(means.outcomes[1].expected_utility, 2.5);
}

#[test]
fn c4_acceptance_all_actions_unsupported_is_a_state_not_an_error() {
    let blind = DecisionInput::from_mean_source(
        "blind",
        mean_source("lab-1", "snap-e1", &[1.0, 3.0, 5.5]),
        &[SupportStatus::MissingEvidence; 3],
        &attested("lab-1"),
        TrustRequirement::Unrestricted,
    )
    .unwrap();
    let decision = compare(&contract("mean"), &[blind]);
    assert_eq!(decision.verdict, SupportedVerdict::NoSupportedAction);
    assert!(decision.outcomes.is_empty());
    assert_eq!(decision.evpi, None);
}

// ----------------------------------------------- tests: the joint-law decision

#[test]
fn c4_acceptance_the_joint_law_decision_matches_the_hand_derived_values() {
    let artifact = law("lab-2", "snap-e2", None);
    let result = evaluate_contract(&law_contract(), &artifact).unwrap();
    close(result.actions[0].expected_utility, 2.0);
    close(result.actions[1].expected_utility, (5.0 + 0.0) / 2.0);
    close(result.actions[2].expected_utility, (0.0 + 4.0) / 2.0);
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("treat".into()));
    close(result.evpi.unwrap(), 4.5 - 2.5);
    close(result.actions[0].expected_regret.unwrap(), 2.5);
    close(result.actions[1].expected_regret.unwrap(), 2.0);
    close(result.actions[2].expected_regret.unwrap(), 2.5);

    let composed = compare(&law_contract(), &[e2_law_input(None)]);
    close(composed.evpi.unwrap(), 2.0);
    assert_eq!(composed.results.len(), 1);

    // With dose 2 masked unsupported only wait and treat are scored: E[max] = 3.5.
    let masked = compare(&law_contract(), &[e2_law_input(Some(vec![true, true, false, true]))]);
    assert_eq!(
        masked.dispositions[2].status,
        ActionStatus::Unsupported {
            reasons: vec![UnsupportedReason {
                input_id: "e2law".into(),
                coordinate: 0,
                issue: CoordinateIssue::BelowSupport(SupportStatus::OutsideEmpiricalSupport),
            }],
        }
    );
    close(masked.evpi.unwrap(), 3.5 - 2.5);
}

#[test]
fn c4_acceptance_probability_and_quantile_come_only_from_the_joint_law() {
    let law = e2_law_input(None);
    let value_of = |action: &str, which: DecisionFunctional| {
        evaluate_functional_on_input(&law_contract(), action, which, &law, SupportStatus::Supported)
            .unwrap()
            .value
    };
    let upper = DecisionFunctional::Probability { threshold: 5.0, tail: Tail::Upper };
    close(value_of("treat", upper), 0.5);
    for (action, expected) in IDS.iter().zip([2.0, 5.0, 4.0]) {
        close(value_of(action, DecisionFunctional::Quantile { p: 0.75 }), expected);
    }

    // The same study's point means agree on the expectation and refuse everything else.
    let mean = e2_mean_input();
    let expectation = evaluate_functional_on_input(
        &contract("mean"),
        "treat",
        DecisionFunctional::Expectation,
        &mean,
        SupportStatus::Supported,
    )
    .unwrap();
    close(expectation.value, 3.5 - 1.0);
    assert_eq!(expectation.standard_error, None);
    for refused in [
        DecisionFunctional::Probability { threshold: 2.0, tail: Tail::Upper },
        DecisionFunctional::Quantile { p: 0.5 },
    ] {
        let error = evaluate_functional_on_input(
            &contract("mean"),
            "treat",
            refused,
            &mean,
            SupportStatus::Supported,
        )
        .unwrap_err();
        assert_eq!(error, BoundaryError::MeanIsNotADistribution { action: "treat".into() });
    }
}

// ----------------------------------------------------- tests: overlapping evidence

fn shared(left: &str, right: &str, route: Option<DependenceRoute>) -> PairRelation {
    PairRelation {
        left: left.into(),
        right: right.into(),
        relation: EvidenceRelation::SharedData { ids: vec!["registry-7".into()] },
        route,
    }
}

fn independent(left: &str, right: &str) -> PairRelation {
    PairRelation {
        left: left.into(),
        right: right.into(),
        relation: EvidenceRelation::IndependentSources,
        route: None,
    }
}

#[test]
fn c4_acceptance_shared_data_refuses_pooling_and_a_licensed_route_lifts_it() {
    let inputs = [native_input(), e1_input(), e2_law_input(None)];
    let relations =
        [independent("native", "e1"), independent("e1", "e2law"), shared("native", "e2law", None)];
    let pooled =
        check_composition(&inputs, &relations, Some(CompositionOperation::StatisticalPooling))
            .unwrap_err();
    assert_eq!(
        pooled,
        BoundaryError::SharedEvidenceNotIndependent {
            left: "native".into(),
            right: "e2law".into(),
        }
    );
    let pair = [e1_input(), e2_law_input(None)];
    let receipt = check_composition(
        &pair,
        &[independent("e1", "e2law")],
        Some(CompositionOperation::StatisticalPooling),
    )
    .unwrap();
    assert!(receipt.independence_assumed && receipt.shared_evidence.is_empty());

    for operation in [CompositionOperation::EvidenceReuse, CompositionOperation::CausalTransport] {
        let receipt = check_composition(&inputs, &relations, Some(operation)).unwrap();
        assert_eq!(receipt.shared_evidence, vec!["registry-7".to_owned()]);
        assert!(!receipt.independence_assumed);
    }

    let route = DependenceRoute {
        kind: RouteKind::JointLaw,
        id: "jl-1".into(),
        source_input: "e2law".into(),
    };
    let licensed = [native_input(), e2_law_input(None)];
    let receipt = check_composition(
        &licensed,
        &[shared("native", "e2law", Some(route))],
        Some(CompositionOperation::StatisticalPooling),
    )
    .unwrap();
    assert_eq!(receipt.routes, vec!["jl-1".to_owned()]);
    assert!(!receipt.independence_assumed);

    let unknown = check_composition(&pair, &[], Some(CompositionOperation::EvidenceReuse));
    assert!(matches!(unknown, Err(BoundaryError::UnknownDependence { .. })));
}

// ------------------------------------------------------------------ study ranking

fn quantity_json(quantity: &ScientificQuantity) -> Value {
    serde_json::to_value(ScientificQuantityWire::from(quantity)).unwrap()
}

fn signal_json() -> Value {
    json!({
        "prior_id": "prior:e2-state",
        "state_quantity": quantity_json(&state_quantity()),
        "observation_quantity": quantity_json(&observation_quantity()),
        "rng_seed": 3,
        "evidence_lineage": ["snapshot:snap-e2"],
        "conditional_independence": "iid_given_state",
    })
}

fn candidates_json() -> Value {
    json!([
        {
            "semantic_id": "native-trial",
            "sample_size": 2,
            "cost": {"amount": 0.02, "unit": "utility"},
            "signal": signal_json(),
            "provider": {"kind": "binomial"},
        },
        {
            "semantic_id": "external-trial",
            "sample_size": 2,
            "cost": {"amount": 0.05, "unit": "utility"},
            "signal": signal_json(),
            "provider": {
                "kind": "external",
                "provider_id": "lab-2",
                "object_id": "trial-2",
                "version_id": "v1",
                "snapshot_id": "snap-e2",
                "attestor": "lab-qa",
                "law": {
                    "mode": "posterior",
                    "states": [0.25, 0.75],
                    "statistics": [0.0, 1.0, 2.0],
                    "predictive": [0.3125, 0.375, 0.3125],
                    "posterior": [[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]],
                },
            },
        },
    ])
}

fn ranking_request() -> DesignRankingRequestWire {
    let digest = source_digest(&law("lab-2", "snap-e2", None));
    let wire = json!({
        "decision": {
            "contract_identity": rollout_contract().identity().unwrap(),
            "utility_unit": "utility",
            "action_ids": ["abstain", "bet"],
            "intercepts": [0.0, -0.5],
            "slopes": [0.0, 1.0],
            "prior": {"kind": "draws", "states": [0.25, 0.75]},
        },
        "candidates": candidates_json(),
        "cost_map": {"cost_unit": "utility", "utility_unit": "utility", "utility_per_cost": 1.0},
        "prior_observation_ids": ["obs:e2-registry"],
        "source_digests": [digest],
        "rng_seed": 5,
    });
    serde_json::from_value(wire).unwrap()
}

#[test]
fn c4_acceptance_the_ranking_matches_the_hand_derived_evsi_and_net_values() {
    let evaluation = design_ranking::evaluate(&ranking_request()).unwrap();
    let report = evaluation.report();
    let ids: Vec<&str> = report.candidates.iter().map(|c| c.semantic_id.as_str()).collect();
    assert_eq!(ids, vec!["native-trial", "external-trial"]);
    close(report.evpi, 1.0 / 8.0);
    close(report.prior_expected_utility, 0.0);
    for candidate in &report.candidates {
        close(candidate.evsi, 1.0 / 16.0);
        close(candidate.evsi, 10.0 / 32.0 * 0.2);
        close(candidate.evpi, 1.0 / 8.0);
    }
    close(report.candidates[0].study_cost_utility.unwrap(), 0.02);
    close(report.candidates[0].net_value.unwrap(), 1.0 / 16.0 - 0.02);
    close(report.candidates[1].net_value.unwrap(), 1.0 / 16.0 - 0.05);
    close(report.candidates[0].net_value.unwrap(), 0.0425);
    close(report.candidates[1].net_value.unwrap(), 0.0125);

    // The follow-up decision over the law's own state column has the same EVPI.
    let rollout = evaluate_contract(&rollout_contract(), &law("lab-2", "snap-e2", None)).unwrap();
    close(rollout.evpi.unwrap(), 1.0 / 8.0);
}

// ------------------------------------------------------------------ the bundle

struct ClaimSpec<'a> {
    provider: &'a str,
    snapshot: &'a str,
    request: &'a str,
    evidence: &'a str,
    values: [f64; 3],
}

fn claim(spec: &ClaimSpec<'_>) -> ExternalClaimArtifact {
    let quantities: Vec<ScientificQuantity> = (0..3).map(|dose| quantity(dose, "mean")).collect();
    let object = ExternalScientificObject::Law(LawProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: spec.provider.into(),
            object_id: "curve".into(),
            version_id: "v1".into(),
            snapshot_id: spec.snapshot.into(),
            request_id: spec.request.into(),
        },
        quantities: quantities.clone(),
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![ExternalCapability::Mean],
    });
    let contract = CheckedCausalContract {
        graph_id: "graph:g".into(),
        identification: IdentificationStatus::NonparametricallyIdentified,
        estimand: quantities.clone(),
        accepted_meanings: vec![DistributionMeaning::InterventionalPredictive],
        required_evidence_ids: vec![],
        required_assumption_ids: vec!["ignorability".into()],
        equivalences: vec![],
    };
    let response = ExternalResponse {
        header: ExternalResultHeader {
            object,
            graph_id: "graph:g".into(),
            quantities,
            evidence_ids: vec![spec.evidence.into()],
            assumption_ids: vec!["ignorability".into()],
            trust: ExternalTrustState::ExternallyAttested { attestor: spec.provider.into() },
        },
        values: spec.values.to_vec(),
        uncertainty: ExternalUncertaintyMeaning::None,
        point_support: Some(vec![SupportStatus::Supported; 3]),
    };
    let bound = bind_external_result(&contract, &ExternalResult::Response(response)).unwrap();
    ExternalClaimArtifact::from_bound_claim(&bound, CONTRACT).unwrap()
}

fn e1_claim() -> ExternalClaimArtifact {
    claim(&ClaimSpec {
        provider: "lab-1",
        snapshot: "snap-e1",
        request: "req-e1",
        evidence: "study:e1",
        values: [1.0, 3.0, 5.5],
    })
}

fn e2_claim(evidence: &str) -> ExternalClaimArtifact {
    claim(&ClaimSpec {
        provider: "lab-2",
        snapshot: "snap-e2",
        request: "req-e2",
        evidence,
        values: [2.0, 3.5, 5.0],
    })
}

fn add(builder: &mut BundleBuilder, id: &str, bytes: &[u8]) {
    builder.add_detected_artifact(Some(id), bytes).unwrap();
}

fn composed_builder(e2_evidence: &str) -> (BundleBuilder, String) {
    let artifact = law("lab-2", "snap-e2", None);
    let contract = law_contract();
    let result = evaluate_contract(&contract, &artifact).unwrap();
    let ranking = design_ranking::evaluate(&ranking_request()).unwrap().export("ranking").unwrap();
    let mut builder = BundleBuilder::new();
    builder
        .add_reference(
            "program",
            NodeKind::CausalContract,
            &program().identity(),
            ProviderOrData::Data {
                snapshot_id: NATIVE_SNAPSHOT.into(),
                digest: NATIVE_DIGEST.into(),
            },
        )
        .unwrap();
    builder.declare_inspected("program", "native.mean.do(a=1)", 4.0).unwrap();
    add(&mut builder, "e1", &e1_claim().to_bytes("e1").unwrap());
    add(&mut builder, "e2claim", &e2_claim(e2_evidence).to_bytes("e2").unwrap());
    let relation = builder.relate("e1", "e2claim", EvidenceRelationship::Independent).unwrap();
    let contract_bytes = DecisionContractArtifact::new(contract).unwrap().to_bytes("c").unwrap();
    add(&mut builder, "contract_j", &contract_bytes);
    add(&mut builder, "e2law", &artifact.to_bytes("law").unwrap());
    let result_bytes = DecisionResultArtifact::new(result, &artifact).to_bytes("r").unwrap();
    add(&mut builder, "result_j", &result_bytes);
    add(&mut builder, "ranking", &ranking);
    for (from, to) in [
        ("program", "e1"),
        ("program", "e2claim"),
        ("contract_j", "result_j"),
        ("e2law", "result_j"),
        ("e2law", "ranking"),
    ] {
        builder.connect(from, to).unwrap();
    }
    (builder, relation)
}

fn consumed(builder: &BundleBuilder, supplied: &SuppliedSources) -> ConsumedBundle {
    let bundle = builder.build().unwrap();
    let bytes = export_bundle(&bundle, "composed").unwrap();
    consume(&bytes, bundle.identity(), supplied).unwrap()
}

fn failed_stage(consumed: &ConsumedBundle, id: &str) -> BundleStage {
    match &consumed.node(id).unwrap().status {
        NodeStatus::Failed { stage, .. } => *stage,
        other => panic!("node `{id}` is {other:?}"),
    }
}

#[test]
fn c4_acceptance_the_composed_bundle_is_consumed_independently_with_the_hand_values() {
    let (builder, relation) = composed_builder("study:e2");
    let bundle = consumed(&builder, &SuppliedSources::default());
    assert!(matches!(
        bundle.node("program").unwrap().status,
        NodeStatus::ReferenceUnresolved { requires: ProviderOrData::Data { .. } }
    ));
    for id in ["e1", "e2claim", relation.as_str(), "contract_j", "e2law", "result_j", "ranking"] {
        assert_eq!(bundle.node(id).unwrap().status, NodeStatus::Verified, "{id}");
    }
    close(bundle.value("result_j", "wait.expected_utility").unwrap(), 2.0);
    close(bundle.value("result_j", "treat.expected_utility").unwrap(), 2.5);
    close(bundle.value("result_j", "extend.expected_utility").unwrap(), 2.0);
    close(bundle.value("ranking", "native-trial.net_value").unwrap(), 0.0425);
    close(bundle.value("ranking", "external-trial.net_value").unwrap(), 0.0125);
    close(bundle.value("ranking", "native-trial.evsi").unwrap(), 0.0625);
    assert_eq!(bundle.node("result_j").unwrap().claim_label, Some(ClaimLabel::JointDraw));
    let law_fact = |id: &str| bundle.node(id).unwrap().facts.get(FACT_LAW).map(String::as_str);
    assert_eq!(law_fact("e2law"), Some(LAW_JOINT));
    assert_eq!(law_fact("e1"), Some(LAW_MEAN_ONLY));
    assert_eq!(
        bundle.node("program").unwrap().inspected,
        vec![("native.mean.do(a=1)".to_owned(), 4.0)]
    );

    let supplied = SuppliedSources::default().with_data(NATIVE_SNAPSHOT, NATIVE_DIGEST);
    consumed(&builder, &supplied).require_verified().unwrap();
    let other = SuppliedSources::default().with_data(NATIVE_SNAPSHOT, "another-digest");
    let changed = consumed(&builder, &other);
    assert_eq!(failed_stage(&changed, "program"), BundleStage::GraphOrSnapshotMismatch);
    close(changed.value("result_j", "treat.expected_utility").unwrap(), 2.5);
}

#[test]
fn c4_acceptance_overlapping_evidence_fails_the_declared_independence() {
    let (builder, relation) = composed_builder("study:e1");
    let supplied = SuppliedSources::default().with_data(NATIVE_SNAPSHOT, NATIVE_DIGEST);
    let bundle = consumed(&builder, &supplied);
    assert_eq!(failed_stage(&bundle, &relation), BundleStage::SwappedEvidence);
    assert_eq!(bundle.node("result_j").unwrap().status, NodeStatus::Verified);
}

#[test]
fn c4_acceptance_the_point_only_decision_is_an_attested_result_with_the_hand_values() {
    let e1 = e1_claim();
    let source = mean_source_of(&e1).unwrap();
    let contract = contract("mean");
    let result = evaluate_contract_on_means(&contract, &source).unwrap();
    close(result.actions[0].expected_utility, 1.0);
    close(result.actions[1].expected_utility, 3.0 - 1.0);
    close(result.actions[2].expected_utility, 5.5 - 3.0);
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("extend".into()));

    let mut builder = BundleBuilder::new();
    add(&mut builder, "claim", &e1.to_bytes("claim").unwrap());
    let contract_bytes = DecisionContractArtifact::new(contract).unwrap().to_bytes("c").unwrap();
    add(&mut builder, "contract_m", &contract_bytes);
    add(&mut builder, "result_m", &mean_result_to_bytes(&result, &source, "m").unwrap());
    builder.connect("claim", "result_m").unwrap();
    builder.connect("contract_m", "result_m").unwrap();
    let bundle = consumed(&builder, &SuppliedSources::default());
    bundle.require_verified().unwrap();
    assert_eq!(bundle.node("result_m").unwrap().claim_label, Some(ClaimLabel::PointOnlyAttested));
    close(bundle.value("result_m", "wait.expected_utility").unwrap(), 1.0);
    close(bundle.value("result_m", "treat.expected_utility").unwrap(), 2.0);
    close(bundle.value("result_m", "extend.expected_utility").unwrap(), 2.5);
}
