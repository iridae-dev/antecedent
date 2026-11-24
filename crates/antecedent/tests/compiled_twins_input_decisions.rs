//! The compiled programs over declared inputs and the direct kernel calls they replace give the
//! same result bit for bit, and each compiled route reproduces a hand-derived value.
//!
//! Every program executes the kernel the direct entry calls, on the same inputs: the supported
//! input decision and functional (`composition_boundary`), the sensitivity decision, the inverse
//! query and its finite-enumeration baseline, the ranking statements over decision draws, the
//! prior adapted to a future signal, the evidence obligations of a failed identification, the
//! finite decision problem under state draws and the adaptive policy search. Each test asserts
//! the compiled result equals the direct one (by value and by its full rendering, which
//! distinguishes every float bit pattern but NaN payloads), that the program runs under its
//! family's route and support unit, and one value derived by hand in the unit record's truth
//! fixture, reached through the compiled route.
//!
//! These results carry no sampling interval, so there is no calibration basis to compare.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use std::sync::Arc;

use antecedent::analysis::input_decision_programs::{
    compile_adaptive_policy_search, compile_decision_draws_ranking, compile_evidence_obligations,
    compile_finite_action_decision, compile_finite_enumeration_baseline, compile_input_functional,
    compile_inverse_query, compile_prior_signal_adaptation, compile_sensitivity_decision,
    compile_supported_input_decision,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, ExecutionContext, SupportStatus, VariableId,
};
use antecedent_design::adaptive_policy::{AdaptivePolicyFamily, PolicyFeature, PolicyRule};
use antecedent_design::adaptive_policy_search::{
    AdaptivePolicySearchError, DrawSplit, HeldOutDesign, PolicyConstraints, PolicyObjective,
    PolicyPath, PolicyRolloutSource, PolicySearchOptions, PreparedAdaptivePolicySearch,
    ResponseStanding,
};
use antecedent_design::composition_boundary::{
    DecisionInput, NativeExecutionRecord, SupportPolicy, TrustEvidence, TrustRequirement,
    evaluate_functional_on_input, evaluate_with_support,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, DecisionFunctional,
    StructuralPolicy, Tail, UtilityExpr,
};
use antecedent_design::decision_reliability::{
    BatchPlan, CandidateCoupling, CandidateDraws, CandidateOrigin, DecisionDraws, DrawSegment,
    RankingRequest, SourceParadigm, StreamId, StreamPurpose, WithinStreamDependence, rank,
};
use antecedent_design::inverse_query::{
    Comparison, EnumeratedForwardValue, FeasibilityStatus, ForwardClaim, ForwardEvidence,
    GridScope, InverseConstraint, InverseQuery, SelectionRule, evaluate_inverse_query,
    finite_enumeration_baseline,
};
use antecedent_design::obligation_adapters::identification_failure_obligations;
use antecedent_design::prior_signal::{
    CandidateSignalSource, NativeSignalFamily, PriorSignalRequest, PriorSourceInput,
    SourceToTarget, adapt_prior_to_signal,
};
use antecedent_design::sensitivity_decision::{
    ActionUtility, AssumptionCoordinate, AssumptionRangeStatement, PointSupport, SamplingStatus,
    SensitivityArtifact, SensitivityDecisionSpec, SensitivityOutcome, SensitivityParts,
    SensitivityProvenance, SurfaceQuantity, UncertaintyRelationship, UtilityTerm,
    contract_from_artifact, evaluate_sensitivity_decision,
};
use antecedent_design::{DecisionPrior, DecisionProblem, DesignError, Utility, evaluate_decision};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord, IdentificationResult};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use antecedent_io_core::{
    DesignVariableRole, DesignVariableSummary, EstimandFingerprint, PriorCatalog, PriorSourceMeta,
    PriorSourceRef, TargetDesign,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

fn same<T: std::fmt::Debug>(compiled: &T, direct: &T) {
    assert_eq!(format!("{compiled:?}"), format!("{direct:?}"));
}

/// The route the program runs under and the unit its support is read from.
fn routed(route: &str, unit: &str, lowered_route: &str, view: &antecedent::planner::PlanView) {
    assert_eq!(lowered_route, route);
    let support = view.unit_support().expect("the route is bound to a unit");
    assert_eq!(support.unit(), unit);
}

fn q(regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
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

/// Aligned equally weighted rows of `columns`, one quantity per column.
fn law(
    columns: &[(ScientificQuantity, Vec<f64>)],
    trust: DistributionTrust,
    calibration: DistributionCalibration,
    supported: Option<Vec<bool>>,
) -> DistributionArtifact {
    let quantities: Vec<ScientificQuantity> = columns.iter().map(|(q, _)| q.clone()).collect();
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "study-snap-n".into(),
            provider_id: "engine".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "snap-n".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    let rows = columns[0].1.len();
    let mut draws = Vec::with_capacity(rows * columns.len());
    for row in 0..rows {
        for (_, values) in columns {
            draws.push(values[row]);
        }
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows, columns.len()],
            weights: None,
            supported,
            calibration,
            trust,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn action(id: &str, regime: &str, functional: &str) -> DecisionAction {
    DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![q(regime, functional)],
        utility: UtilityExpr::Input(0),
    }
}

fn contract(actions: Vec<DecisionAction>) -> DecisionContract {
    DecisionContract {
        actions,
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

// ---- supported inputs (composition.boundary) ----------------------------------------------

/// `x0 = [1, 2, 3, 2]` (mean 2) and `x1 = [4, 0, 2, 6]` (mean 3) under `do(a=0)` and
/// `do(a=1)`; the `do(a=2)` coordinate is masked when `mask` says so.
fn boundary_law(mask: Option<Vec<bool>>) -> DistributionArtifact {
    law(
        &[
            (q("do(a=0)", "outcome"), vec![1.0, 2.0, 3.0, 2.0]),
            (q("do(a=1)", "outcome"), vec![4.0, 0.0, 2.0, 6.0]),
            (q("do(a=2)", "outcome"), vec![0.0; 4]),
        ],
        DistributionTrust::NativeSupported,
        DistributionCalibration::Exact,
        mask,
    )
}

fn boundary_contract() -> DecisionContract {
    contract(vec![
        action("wait", "do(a=0)", "outcome"),
        action("treat", "do(a=1)", "outcome"),
        action("extend", "do(a=2)", "outcome"),
    ])
}

#[test]
fn compiled_supported_input_decision_equals_evaluate_with_support() {
    let ctx = ctx();
    let input = DecisionInput::from_distribution_artifact(
        "native",
        &boundary_law(Some(vec![true, true, false])),
        &TrustEvidence::NativeExecution(NativeExecutionRecord {
            execution_id: "exec-1".into(),
            provider_id: "engine".into(),
            snapshot_id: "snap-n".into(),
        }),
        TrustRequirement::Native,
    )
    .unwrap();
    let policy = SupportPolicy::compare_supported();
    let direct =
        evaluate_with_support(&boundary_contract(), std::slice::from_ref(&input), &policy).unwrap();
    let program =
        compile_supported_input_decision(boundary_contract(), vec![input], policy, &ctx).unwrap();
    routed(
        "SupportedInputDecision",
        "composition.boundary",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    // By hand: E[wait] = 2, E[treat] = 3, E[max(x0, x1)] = (4 + 2 + 3 + 6) / 4 = 3.75, EVPI 0.75.
    assert!((compiled.outcomes[0].expected_utility - 2.0).abs() < 1e-12);
    assert!((compiled.outcomes[1].expected_utility - 3.0).abs() < 1e-12);
    assert!((compiled.evpi.unwrap() - 0.75).abs() < 1e-12);
}

#[test]
fn compiled_input_functional_equals_evaluate_functional_on_input() {
    let ctx = ctx();
    let two =
        contract(vec![action("wait", "do(a=0)", "outcome"), action("treat", "do(a=1)", "outcome")]);
    let input = DecisionInput::from_distribution_artifact(
        "law",
        &boundary_law(None),
        &TrustEvidence::None,
        TrustRequirement::Unrestricted,
    )
    .unwrap();
    let functional = DecisionFunctional::Probability { threshold: 2.0, tail: Tail::Upper };
    let direct = evaluate_functional_on_input(
        &two,
        "treat",
        functional.clone(),
        &input,
        SupportStatus::Supported,
    )
    .unwrap();
    let program =
        compile_input_functional(two, "treat", functional, input, SupportStatus::Supported, &ctx)
            .unwrap();
    routed(
        "SupportedInputDecision",
        "composition.boundary",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    assert_eq!(compiled.value.to_bits(), direct.value.to_bits());
    // P(x1 >= 2) over [4, 0, 2, 6] is 3 / 4.
    assert!((compiled.value - 0.75).abs() < 1e-12);
}

// ---- sensitivity decision (decision.sensitivity_composition) ------------------------------

fn surface(name: &str, values: &[f64]) -> SurfaceQuantity {
    let scientific = ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "utils".into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "sensitivity_surface".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    };
    SurfaceQuantity {
        quantity: ScientificQuantityWire::from(&scientific),
        lower: values.to_vec(),
        upper: values.to_vec(),
    }
}

/// A = 5 - gamma = (5, 4, 3) against B = 1 on gamma in {0, 1, 2}: A leads everywhere.
fn invariant_artifact() -> SensitivityArtifact {
    SensitivityArtifact::new(SensitivityParts {
        coordinate: AssumptionCoordinate {
            id: "gamma".into(),
            scale: "sensitivity_parameter".into(),
            units: "dimensionless".into(),
            minimum: 0.0,
            maximum: 2.0,
        },
        grid: vec![0.0, 1.0, 2.0],
        support: vec![PointSupport::Supported; 3],
        quantities: vec![surface("ua", &[5.0, 4.0, 3.0]), surface("ub", &[1.0, 1.0, 1.0])],
        actions: vec![
            ActionUtility { id: "A".into(), utility: UtilityTerm::quantity("ua") },
            ActionUtility { id: "B".into(), utility: UtilityTerm::quantity("ub") },
        ],
        uncertainty: UncertaintyRelationship {
            assumption_range: AssumptionRangeStatement {
                kind: "assumption_range".into(),
                interpretation: "assumption range; not a probability".into(),
            },
            identified_bound: None,
            sampling: SamplingStatus::Withheld {
                reason_code: "uncertainty_representation_insufficient".into(),
                detail: antecedent_core::reason_details::joint_sensitivity::NO_INTERVAL_METHOD
                    .into(),
            },
        },
        provenance: SensitivityProvenance {
            source_kind: "supplied_surface".into(),
            query_binding: "sensitivity-test".into(),
            provider_snapshot: "snapshot-1".into(),
            source_regime: "regime:1".into(),
            method: "hand-derived surface".into(),
            causal_contract_id: "checked-contract".into(),
            decision_threshold: None,
            source_tipping: vec![],
        },
    })
    .unwrap()
}

#[test]
fn compiled_sensitivity_decision_equals_evaluate_sensitivity_decision() {
    let ctx = ctx();
    let artifact = invariant_artifact();
    let contract =
        contract_from_artifact(&artifact, StructuralPolicy::RequireInvariantBestAction).unwrap();
    let spec = SensitivityDecisionSpec::default();
    let direct = evaluate_sensitivity_decision(&contract, &artifact, &spec).unwrap();
    let program = compile_sensitivity_decision(contract, artifact, spec, &ctx).unwrap();
    routed(
        "SensitivityDecision",
        "decision.sensitivity_composition",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    assert_eq!(compiled.outcome, SensitivityOutcome::InvariantAction { action: "A".into() });
}

// ---- inverse queries (decision.functional_inverse) ----------------------------------------

const REGIMES: [&str; 3] = ["do(a=0)", "do(a=1)", "do(a=2)"];
const IDS: [&str; 3] = ["a0", "a1", "a2"];

/// Bernoulli(1/4, 1/2, 3/4) as four equiprobable aligned rows.
fn bernoulli_law() -> DistributionArtifact {
    law(
        &[
            (q(REGIMES[0], "outcome"), vec![1.0, 0.0, 0.0, 0.0]),
            (q(REGIMES[1], "outcome"), vec![1.0, 1.0, 0.0, 0.0]),
            (q(REGIMES[2], "outcome"), vec![1.0, 1.0, 1.0, 0.0]),
        ],
        DistributionTrust::Unverified,
        DistributionCalibration::Exact,
        None,
    )
}

fn inverse_query() -> InverseQuery {
    InverseQuery {
        contract: contract(
            IDS.iter().zip(REGIMES).map(|(id, regime)| action(id, regime, "outcome")).collect(),
        ),
        grid_order: IDS.iter().map(|id| (*id).to_owned()).collect(),
        grid_scope: GridScope::FiniteEnumeration,
        constraints: vec![InverseConstraint::TargetMean {
            target: 0.5,
            comparison: Comparison::AtLeast,
        }],
        selection: SelectionRule::FirstInGridOrder,
        tolerance: 1e-12,
        max_evaluations: None,
    }
}

fn forward() -> ForwardEvidence {
    ForwardEvidence {
        point: Some(ForwardClaim::Law(Box::new(bernoulli_law()))),
        ..ForwardEvidence::default()
    }
}

#[test]
fn compiled_inverse_query_equals_evaluate_inverse_query() {
    let ctx = ctx();
    let direct = evaluate_inverse_query(&inverse_query(), &forward()).unwrap();
    let program = compile_inverse_query(inverse_query(), forward(), &ctx).unwrap();
    routed(
        "InverseQueryDecision",
        "decision.functional_inverse",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    // Means by hand: 1/4, 2/4, 3/4; the least action reaching 1/2 is a1.
    for (action, expected) in compiled.actions.iter().zip([0.25, 0.5, 0.75]) {
        assert!((action.point_values[0].value.unwrap() - expected).abs() < 1e-9);
    }
    assert_eq!(compiled.selected.as_deref(), Some("a1"));
}

#[test]
fn compiled_finite_enumeration_baseline_equals_the_direct_baseline() {
    let ctx = ctx();
    let points: Vec<EnumeratedForwardValue> = IDS
        .iter()
        .zip([0.25, 0.5, 0.75])
        .map(|(id, value)| EnumeratedForwardValue {
            id: (*id).into(),
            value: Some(value),
            supported: true,
        })
        .collect();
    let direct = finite_enumeration_baseline(&points, 0.5, Comparison::AtLeast, 1e-12).unwrap();
    let program =
        compile_finite_enumeration_baseline(points, 0.5, Comparison::AtLeast, 1e-12, &ctx).unwrap();
    routed(
        "InverseQueryDecision",
        "decision.functional_inverse",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    let statuses: Vec<FeasibilityStatus> = compiled.iter().map(|(_, status)| *status).collect();
    assert_eq!(
        statuses,
        vec![
            FeasibilityStatus::Infeasible,
            FeasibilityStatus::Feasible,
            FeasibilityStatus::Feasible
        ]
    );
}

// ---- ranking statements (decision_reliability.ranking_statement) --------------------------

fn carrier(columns: &[(&str, Vec<f64>)]) -> DecisionDraws {
    let n = columns[0].1.len();
    let shared =
        DrawSegment::whole(StreamId::new(20_240, StreamPurpose::Selection, 0, "test_stream"), n);
    DecisionDraws {
        utility_unit: "utility".into(),
        producer: "test".into(),
        source: SourceParadigm::KnownLaw,
        coupling: CandidateCoupling::SharedSimulation,
        within_stream: WithinStreamDependence::Independent,
        weights: None,
        repeated_evaluations: 0,
        origin: CandidateOrigin::Enumerated,
        search_record: None,
        candidates: columns
            .iter()
            .map(|(id, utilities)| CandidateDraws {
                id: (*id).to_owned(),
                utilities: utilities.clone(),
                segment: shared.clone(),
            })
            .collect(),
    }
}

#[test]
fn compiled_decision_draws_ranking_equals_rank() {
    let ctx = ctx();
    // Four equally likely states; the last is a three-way tie.
    let draws = carrier(&[
        ("A", vec![3.0, 3.0, 2.0, 2.0]),
        ("B", vec![2.0, 1.0, 3.0, 2.0]),
        ("C", vec![1.0, 2.0, 1.0, 2.0]),
    ]);
    let request = RankingRequest {
        baseline: Some("B".into()),
        top_k: vec![2],
        batch_plan: BatchPlan::SquareRoot,
    };
    let direct = rank(&draws, &request, &ctx).unwrap();
    let program = compile_decision_draws_ranking(draws, request, &ctx).unwrap();
    routed(
        "DecisionDrawsRanking",
        "decision_reliability.ranking_statement",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    // P(A best) = (1 + 1 + 0 + 1/3) / 4 = 7/12 with the tie shared.
    let a = compiled.statements.iter().find(|s| s.id == "A").unwrap();
    assert!((a.probability_best.value - 7.0 / 12.0).abs() < 1e-12);
    assert_eq!(compiled.best_by_expected_utility, "A");
}

// ---- prior adapted to a future signal (design.prior_signal_planning) ----------------------

fn meta(id: &str, outcome: &str) -> PriorSourceMeta {
    PriorSourceMeta::new(
        id,
        EstimandFingerprint::new("ate", "t", outcome),
        "NonparametricallyIdentified",
    )
    .with_design(vec![
        DesignVariableSummary::new("t", DesignVariableRole::Treatment),
        DesignVariableSummary::new("y", DesignVariableRole::Outcome),
    ])
}

fn source(id: &str, draws: &[f64], observations: &[&str]) -> PriorSourceInput {
    PriorSourceInput {
        artifact_id: id.into(),
        source_population: "target".into(),
        draws: draws.to_vec(),
        weight: 1.0,
        prior_strength: 1.0,
        conflict_shrinkage: 0.0,
        lineage: vec![format!("lineage:{id}")],
        observation_ids: observations.iter().map(|s| (*s).to_owned()).collect(),
        source_to_target: SourceToTarget::Identity,
    }
}

fn signal_request() -> PriorSignalRequest {
    PriorSignalRequest {
        prior_id: "prior-1".into(),
        sources: vec![
            source("A", &[0.2, 0.4], &["obs-a1", "obs-a2"]),
            source("B", &[0.6, 0.8], &["obs-b1"]),
        ],
        signal: CandidateSignalSource::Native(NativeSignalFamily::Binomial),
        target_population: "target".into(),
        transport: None,
        candidate_observation_ids: vec!["future-1".into()],
        resolution: 4,
    }
}

#[test]
fn compiled_prior_signal_adaptation_equals_adapt_prior_to_signal() {
    let ctx = ctx();
    let catalog = PriorCatalog::from_sources(vec![
        PriorSourceRef::from_meta(meta("A", "y")),
        PriorSourceRef::from_meta(meta("B", "y")),
    ]);
    let target = TargetDesign::new(EstimandFingerprint::new("ate", "t", "y"), ["t", "y"]);
    let direct = adapt_prior_to_signal(&catalog, &target, &signal_request()).unwrap();
    let program = compile_prior_signal_adaptation(catalog, target, signal_request(), &ctx).unwrap();
    routed(
        "PriorSignalAdaptation",
        "design.prior_signal_planning",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    same(&compiled, &direct);
    // Quotas 2 and 2: A contributes 0.2, 0.4 and B 0.6, 0.8; the pooled mean is 0.5.
    let DecisionPrior::Draws(rows) = &compiled.prior else { panic!("expected draws") };
    let expected = [0.2, 0.4, 0.6, 0.8];
    assert!(rows.iter().zip(expected).all(|(x, y)| (x - y).abs() < 1e-12));
    assert!((compiled.diagnostics.pooled_mean - 0.5).abs() < 1e-12);
}

// ---- evidence obligations (design.evidence_obligations) -----------------------------------

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

#[test]
fn compiled_evidence_obligations_equal_identification_failure_obligations() {
    let ctx = ctx();
    // T=0, Y=1; Z=2 and W=3 confound T and Y; I=4 causes Z only; D=5 is a child of T.
    let mut dag = Dag::with_variables(6);
    for (a, b) in [(0, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 2), (0, 5)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let graph = Admg::from_dag(&dag).unwrap();
    let failed = IdentificationResult::not_identified(
        CausalQuery::average_effect(AverageEffectQuery::binary_ate(v(0), v(1))),
        DerivationTrace::default(),
        AssumptionSet::new(),
        IdentificationPerformanceRecord::default(),
    );
    let observed = vec![v(0), v(1)];
    let direct =
        identification_failure_obligations(&failed, &graph, &observed, "pop", None).unwrap();
    let program = compile_evidence_obligations(failed, graph, observed, "pop", None, &ctx).unwrap();
    routed(
        "EvidenceObligations",
        "design.evidence_obligations",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    // The minimal back-door set is {Z, W}: the joint law owed is over (T, Y, Z, W).
    let backdoor: Vec<_> =
        compiled.iter().filter(|o| o.provenance.family.as_ref() == "backdoor").collect();
    assert_eq!(backdoor.len(), 1);
    assert_eq!(
        backdoor[0].provenance.proof_step.as_deref(),
        Some("backdoor.adjustment_set:0,1,2,3")
    );
}

// ---- a finite decision problem (design.evsi) ----------------------------------------------

/// `U(a, theta) = a * theta`.
struct Linear;

impl Utility<f64, f64> for Linear {
    fn evaluate_batch(&self, a: &[f64], o: &[f64], out: &mut [f64]) -> Result<(), DesignError> {
        for (i, ai) in a.iter().enumerate() {
            for (j, oj) in o.iter().enumerate() {
                out[i * o.len() + j] = ai * oj;
            }
        }
        Ok(())
    }
}

fn problem() -> DecisionProblem<f64, f64> {
    DecisionProblem::new(vec![1.0, 2.0], Arc::new(Linear), vec![])
}

#[test]
fn compiled_finite_action_decision_equals_evaluate_decision() {
    let ctx = ctx();
    let direct = evaluate_decision(&problem(), &[-1.0, 3.0]).unwrap();
    let program = compile_finite_action_decision(problem(), vec![-1.0, 3.0], &ctx).unwrap();
    routed("FiniteActionDecision", "design.evsi", program.lowered().route(), &program.plan_view());
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    // Outcomes [-1, 3], U = a * theta, a in {1, 2}: E[U] = 1 and 2, the Bayes action is a = 2
    // and the expected regret is the EVPI (max(-1, -2) + max(3, 6)) / 2 - 2 = 0.5.
    assert!((compiled.expected_utility - 2.0).abs() < 1e-15);
    assert!((compiled.posterior_regret - 0.5).abs() < 1e-15);
}

// ---- adaptive policy search ----------------------------------------------------------------

/// A noiseless one-step source: the path utility is `-(u - 0.3)^2` for the action `u`, so the
/// best fixed action is 0.3 with value 0.
struct Quadratic;

impl PolicyRolloutSource for Quadratic {
    fn identity(&self) -> String {
        "quadratic-source".into()
    }
    fn n_draws(&self, _split: DrawSplit) -> usize {
        4
    }
    fn horizon(&self) -> usize {
        1
    }
    fn standing(&self) -> ResponseStanding {
        ResponseStanding { identified: false, basis: "declared simulator".into() }
    }
    fn heldout_design(&self) -> HeldOutDesign {
        HeldOutDesign::IndependentDraws
    }
    fn rollout(
        &self,
        family: &AdaptivePolicyFamily,
        theta: &[f64],
        split: DrawSplit,
        _ctx: &ExecutionContext,
    ) -> Result<Vec<PolicyPath>, AdaptivePolicySearchError> {
        (0..self.n_draws(split))
            .map(|_| {
                let u = family
                    .action(theta, 0, &[0.0])
                    .map_err(|e| AdaptivePolicySearchError::Source(e.to_string()))?;
                Ok(PolicyPath {
                    utility: -(u - 0.3) * (u - 0.3),
                    actions: vec![u],
                    step_costs: vec![u],
                    step_exposures: vec![u],
                })
            })
            .collect()
    }
}

fn prepared_search() -> PreparedAdaptivePolicySearch {
    let schedule = AdaptivePolicyFamily::new(
        "schedule",
        vec![PolicyFeature::Current(VariableId::from_raw(0))],
        &[(-4.0, 4.0)],
        PolicyRule::Schedule { steps: 1 },
        (0.0, 1.0),
        (-1.0, 1.0),
    )
    .unwrap();
    PreparedAdaptivePolicySearch::compile(
        vec![schedule],
        None,
        PolicyConstraints::default(),
        PolicyObjective::ExpectedUtility,
        PolicySearchOptions { seed: 11, max_evaluations: 600, ..PolicySearchOptions::default() },
    )
    .unwrap()
}

#[test]
fn compiled_adaptive_policy_search_equals_the_prepared_search() {
    let ctx = ExecutionContext::for_tests(5);
    let direct = prepared_search().execute(&Quadratic, &ctx).unwrap();
    let program = compile_adaptive_policy_search(prepared_search(), &ctx).unwrap();
    routed(
        "AdaptivePolicySearch",
        "design.adaptive_policy_search",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&Quadratic, &ctx).unwrap();
    same(&compiled, &direct);
    assert_eq!(compiled.record_identity, direct.record_identity);
    // By hand: the optimum of -(u - 0.3)^2 on [0, 1] is u = 0.3 with value 0.
    let family = &compiled.families[0];
    assert!((family.theta[0] - 0.3).abs() < 0.05, "theta {:?}", family.theta);
    assert!(family.in_sample.mean_utility > -0.0025);
}
