//! Identity-site laws of the compiled programs over declared inputs (`formal/identity_sites.toml`):
//! each test compiles the production entry and reads the program identity, so the laws run on
//! the question digest every family writes.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{AssumptionSet, AverageEffectQuery, CausalQuery};
use antecedent_design::adaptive_policy::{AdaptivePolicyFamily, PolicyFeature, PolicyRule};
use antecedent_design::adaptive_policy_search::{
    PolicyConstraints, PolicyObjective, PolicySearchOptions,
};
use antecedent_design::composition_boundary::{TrustEvidence, TrustRequirement};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionCriterion, StructuralPolicy, Tail, UtilityExpr,
};
use antecedent_design::decision_reliability::{
    BatchPlan, CandidateCoupling, CandidateDraws, CandidateOrigin, DrawSegment, JointLawStanding,
    ProducerOutput, RolloutNoise, RolloutUtilityDraws, SourceParadigm, StreamId, StreamPurpose,
    WithinStreamDependence, reach_carrier,
};
use antecedent_design::decision_structural::AtomEvidence;
use antecedent_design::sensitivity_decision::{
    ActionUtility, AssumptionCoordinate, AssumptionRangeStatement, PointSupport, SamplingStatus,
    SensitivityParts, SensitivityProvenance, SurfaceQuantity, UncertaintyRelationship, UtilityTerm,
    contract_from_artifact,
};
use antecedent_design::{DecisionConstraint, Utility};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord};
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use antecedent_io_core::{
    DesignVariableRole, DesignVariableSummary, EstimandFingerprint, PriorSourceMeta,
};

use crate::identity_laws::{
    ADVERSARIAL_TEXT, all_distinct, bit_exact, injective, ordered_laws, set_laws,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

fn q(regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
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

fn action(id: &str, regime: &str) -> DecisionAction {
    DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![q(regime)],
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

fn two_actions() -> DecisionContract {
    contract(vec![action("wait", "do(a=0)"), action("treat", "do(a=1)")])
}

/// A mean input over `(regime, mean)` coordinates, every coordinate supported.
fn mean_input(id: &str, coordinates: &[(&str, f64)]) -> DecisionInput {
    let source = MeanSource {
        coordinates: coordinates.iter().map(|(regime, _)| q(regime)).collect(),
        means: coordinates.iter().map(|(_, mean)| *mean).collect(),
        estimate: MeanEstimate::Posterior,
        provider_id: "engine".into(),
        snapshot_id: "snap".into(),
        causal_contract_id: "checked".into(),
        rng_id: "rng".into(),
    };
    let statuses = vec![SupportStatus::Supported; coordinates.len()];
    DecisionInput::from_mean_source(
        id,
        source,
        &statuses,
        &TrustEvidence::None,
        TrustRequirement::Unrestricted,
    )
    .expect("a mean input")
}

#[test]
fn site_input_decision_programs_supported_inputs_and_coordinates_are_ordered() {
    let ctx = ctx();
    let decide = |inputs: Vec<DecisionInput>, policy: SupportPolicy| {
        Some(
            compile_supported_input_decision(two_actions(), inputs, policy, &ctx)
                .expect("compiles")
                .identity(),
        )
    };
    let coordinates = [("do(a=0)", 1.0), ("do(a=1)", 2.0)];
    // Actions are assigned to inputs in order and values are reported per input.
    ordered_laws(&["m1", "m2", "m3"], |ids| {
        decide(
            ids.iter().map(|id| mean_input(id, &coordinates)).collect(),
            SupportPolicy::compare_supported(),
        )
    });
    // Means align with coordinates by position, and a coordinate's support is its first entry.
    ordered_laws(&[("do(a=0)", 1.0), ("do(a=1)", 2.0), ("do(a=2)", 3.0)], |coordinates| {
        decide(vec![mean_input("m", coordinates)], SupportPolicy::compare_supported())
    });
    all_distinct(&[
        decide(vec![mean_input("m", &coordinates)], SupportPolicy::compare_supported()),
        decide(vec![mean_input("m", &coordinates)], SupportPolicy::require_all()),
        decide(vec![mean_input("n", &coordinates)], SupportPolicy::compare_supported()),
    ]);
    bit_exact(|mean| {
        decide(
            vec![mean_input("m", &[("do(a=0)", mean), ("do(a=1)", 2.0)])],
            SupportPolicy::compare_supported(),
        )
    });
}

#[test]
fn site_input_decision_programs_input_functional_reads_every_part() {
    let ctx = ctx();
    let functional = |action: &str, threshold: f64, weakest: SupportStatus| {
        compile_input_functional(
            two_actions(),
            action,
            DecisionFunctional::Probability { threshold, tail: Tail::Upper },
            mean_input("m", &[("do(a=0)", 1.0), ("do(a=1)", 2.0)]),
            weakest,
            &ctx,
        )
        .expect("compiles")
        .identity()
    };
    assert_eq!(
        functional("treat", 2.0, SupportStatus::Supported),
        functional("treat", 2.0, SupportStatus::Supported)
    );
    all_distinct(&[
        functional("treat", 2.0, SupportStatus::Supported),
        functional("wait", 2.0, SupportStatus::Supported),
        functional("treat", 3.0, SupportStatus::Supported),
        functional("treat", 2.0, SupportStatus::MissingEvidence),
    ]);
}

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

fn sensitivity_artifact() -> SensitivityArtifact {
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
    .expect("a sensitivity artifact")
}

#[test]
fn site_input_decision_programs_sensitivity_weights_are_ordered() {
    let ctx = ctx();
    let decide = |spec: SensitivityDecisionSpec| {
        let artifact = sensitivity_artifact();
        let contract =
            contract_from_artifact(&artifact, StructuralPolicy::RequireInvariantBestAction)
                .expect("a contract");
        Some(
            compile_sensitivity_decision(contract, artifact, spec, &ctx)
                .expect("compiles")
                .identity(),
        )
    };
    // One weight per evaluated grid point, in ascending coordinate order.
    ordered_laws(&[0.5, 0.3, 0.2], |weights| {
        decide(SensitivityDecisionSpec {
            weights: Some(weights.to_vec()),
            ..SensitivityDecisionSpec::default()
        })
    });
    all_distinct(&[
        decide(SensitivityDecisionSpec::default()),
        decide(SensitivityDecisionSpec {
            range: Some((0.0, 1.0)),
            ..SensitivityDecisionSpec::default()
        }),
        decide(SensitivityDecisionSpec {
            sampling_composition: Some("method".into()),
            ..SensitivityDecisionSpec::default()
        }),
    ]);
}

fn inverse_query(grid: &[&str], targets: &[f64], tolerance: f64) -> InverseQuery {
    InverseQuery {
        contract: contract(vec![
            action("a0", "do(a=0)"),
            action("a1", "do(a=1)"),
            action("a2", "do(a=2)"),
        ]),
        grid_order: grid.iter().map(|id| (*id).to_owned()).collect(),
        grid_scope: GridScope::FiniteEnumeration,
        constraints: targets
            .iter()
            .map(|&target| InverseConstraint::TargetMean {
                target,
                comparison: Comparison::AtLeast,
            })
            .collect(),
        selection: SelectionRule::FirstInGridOrder,
        tolerance,
        max_evaluations: None,
    }
}

fn members(ids: &[&str]) -> ForwardEvidence {
    ForwardEvidence {
        identified_set: Some(MemberSet {
            members: ids
                .iter()
                .map(|id| StructuralAtom {
                    id: (*id).into(),
                    probability: None,
                    evidence: AtomEvidence::Unidentified,
                })
                .collect(),
            exhaustive: true,
        }),
        ..ForwardEvidence::default()
    }
}

#[test]
fn site_input_decision_programs_inverse_query_grid_constraints_and_members_are_ordered() {
    let ctx = ctx();
    let answer = |query: InverseQuery, evidence: ForwardEvidence| {
        Some(compile_inverse_query(query, evidence, &ctx).expect("compiles").identity())
    };
    let grid = ["a0", "a1", "a2"];
    // The declared grid is the tie-rule order.
    ordered_laws(&grid, |grid| answer(inverse_query(grid, &[0.5], 1e-12), members(&["s1"])));
    // Values are reported per constraint in declaration order.
    ordered_laws(&[0.25, 0.5, 0.75], |targets| {
        answer(inverse_query(&grid, targets, 1e-12), members(&["s1"]))
    });
    // Members are reported in declaration order.
    ordered_laws(&["s1", "s2", "s3"], |ids| {
        answer(inverse_query(&grid, &[0.5], 1e-12), members(ids))
    });
    bit_exact(|tolerance| answer(inverse_query(&grid, &[0.5], tolerance), members(&["s1"])));
}

#[test]
fn site_input_decision_programs_enumerated_points_are_ordered() {
    let ctx = ctx();
    let classify = |points: &[(&str, f64)], target: f64| {
        let points = points
            .iter()
            .map(|(id, value)| EnumeratedForwardValue {
                id: (*id).into(),
                value: Some(*value),
                supported: true,
            })
            .collect();
        Some(
            compile_finite_enumeration_baseline(points, target, Comparison::AtLeast, 1e-12, &ctx)
                .expect("compiles")
                .identity(),
        )
    };
    ordered_laws(&[("a0", 0.25), ("a1", 0.5), ("a2", 0.75)], |points| classify(points, 0.5));
    bit_exact(|value| classify(&[("a0", value)], 0.5));
    bit_exact(|target| classify(&[("a0", 0.25)], target));
}

fn carrier(columns: &[(&str, [f64; 4])]) -> DecisionDraws {
    let shared =
        DrawSegment::whole(StreamId::new(20_240, StreamPurpose::Selection, 0, "test_stream"), 4);
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
                utilities: utilities.to_vec(),
                segment: shared.clone(),
            })
            .collect(),
    }
}

#[test]
fn site_input_decision_programs_draws_candidates_are_a_set_and_top_k_ordered() {
    let ctx = ctx();
    let rank_with = |columns: &[(&str, [f64; 4])], top_k: Vec<u32>| {
        let draws = carrier(columns);
        // A repeated candidate id is refused by the carrier's own validation.
        draws.validate().ok()?;
        let request = RankingRequest { baseline: None, top_k, batch_plan: BatchPlan::SquareRoot };
        Some(compile_decision_draws_ranking(draws, request, &ctx).expect("compiles").identity())
    };
    let candidates =
        [("A", [3.0, 3.0, 2.0, 2.0]), ("B", [2.0, 1.0, 3.0, 2.0]), ("C", [1.0, 2.0, 1.0, 2.0])];
    set_laws(&candidates, |columns| rank_with(columns, vec![1]));
    // Top-k sizes are reported in the declared order.
    ordered_laws(&[1, 2, 3], |top_k| rank_with(&candidates, top_k.to_vec()));
}

fn rollout_carrier(
    columns: &[(&str, [f64; 4])],
    noise: RolloutNoise,
) -> Option<antecedent_design::decision_reliability::ProducerCarrier> {
    let policies =
        columns.iter().map(|(id, utilities)| ((*id).to_owned(), utilities.to_vec())).collect();
    reach_carrier(ProducerOutput::Rollout(RolloutUtilityDraws {
        model_identity: "model".into(),
        stream_scheme: "keyed".into(),
        seed: 3,
        utility_unit: "utility".into(),
        purpose: StreamPurpose::Selection,
        replicate: 0,
        noise,
        policies,
        reported_means: Vec::new(),
        max_standard_errors: 4.0,
    }))
    .ok()
}

#[test]
fn site_input_decision_programs_carrier_ranking_candidates_are_a_set_and_top_k_ordered() {
    let ctx = ctx();
    let rank_with = |columns: &[(&str, [f64; 4])], top_k: Vec<u32>| {
        // A repeated policy id is refused by the carrier's own validation.
        let carrier = rollout_carrier(columns, RolloutNoise::StructuralExogenous)?;
        let request = RankingRequest { baseline: None, top_k, batch_plan: BatchPlan::SquareRoot };
        Some(compile_producer_carrier_ranking(carrier, request, &ctx).expect("compiles").identity())
    };
    let candidates =
        [("A", [3.0, 3.0, 2.0, 2.0]), ("B", [2.0, 1.0, 3.0, 2.0]), ("C", [1.0, 2.0, 1.0, 2.0])];
    set_laws(&candidates, |columns| rank_with(columns, vec![1]));
    ordered_laws(&[1, 2, 3], |top_k| rank_with(&candidates, top_k.to_vec()));
    // The same draws under the two rollout noises differ only in joint-law standing.
    let shared = rollout_carrier(&candidates, RolloutNoise::StructuralExogenous).unwrap();
    let marginal = rollout_carrier(&candidates, RolloutNoise::ImplementationKeyed).unwrap();
    assert_eq!(shared.joint_law(), JointLawStanding::OneSharedLaw);
    assert!(matches!(marginal.joint_law(), JointLawStanding::MarginalOnly { .. }));
    assert_eq!(shared.draws().digest_hex(), marginal.draws().digest_hex());
    let request =
        RankingRequest { baseline: None, top_k: vec![1], batch_plan: BatchPlan::SquareRoot };
    assert_ne!(
        compile_producer_carrier_ranking(shared, request.clone(), &ctx).unwrap().identity(),
        compile_producer_carrier_ranking(marginal, request, &ctx).unwrap().identity(),
    );
}

fn meta(id: &str) -> PriorSourceMeta {
    PriorSourceMeta::new(
        id,
        EstimandFingerprint::new("ate", "t", "y"),
        "NonparametricallyIdentified",
    )
    .with_design(vec![
        DesignVariableSummary::new("t", DesignVariableRole::Treatment),
        DesignVariableSummary::new("y", DesignVariableRole::Outcome),
    ])
}

fn prior_source(
    id: &str,
    population: &str,
    lineage: &[&str],
    observations: &[&str],
) -> PriorSourceInput {
    PriorSourceInput {
        artifact_id: id.into(),
        source_population: population.into(),
        draws: vec![0.2, 0.4],
        weight: 1.0,
        prior_strength: 1.0,
        conflict_shrinkage: 0.0,
        lineage: lineage.iter().map(|s| (*s).to_owned()).collect(),
        observation_ids: observations.iter().map(|s| (*s).to_owned()).collect(),
        source_to_target: SourceToTarget::Identity,
    }
}

fn signal_request(
    sources: Vec<PriorSourceInput>,
    candidate: &[&str],
    signal: CandidateSignalSource,
) -> PriorSignalRequest {
    PriorSignalRequest {
        prior_id: "prior-1".into(),
        sources,
        signal,
        target_population: "target".into(),
        transport: None,
        candidate_observation_ids: candidate.iter().map(|s| (*s).to_owned()).collect(),
        resolution: 4,
    }
}

fn binomial() -> CandidateSignalSource {
    CandidateSignalSource::Native(NativeSignalFamily::Binomial)
}

#[test]
fn site_input_decision_programs_prior_signal_sets_and_sources() {
    let ctx = ctx();
    let adapt = |variables: &[&str], request: PriorSignalRequest| {
        let catalog = PriorCatalog::from_sources(vec![
            PriorSourceRef::from_meta(meta("A")),
            PriorSourceRef::from_meta(meta("B")),
        ]);
        let target =
            TargetDesign::new(EstimandFingerprint::new("ate", "t", "y"), variables.iter().copied());
        Some(
            compile_prior_signal_adaptation(catalog, target, request, &ctx)
                .expect("compiles")
                .identity(),
        )
    };
    let one = |lineage: &[&str], observations: &[&str]| {
        vec![prior_source("A", "target", lineage, observations)]
    };
    let ty = ["t", "y"];
    // The candidate's observation ids are checked for overlap as a set.
    set_laws(&["f1", "f2", "f3"], |ids| {
        adapt(&ty, signal_request(one(&["l"], &["o"]), ids, binomial()))
    });
    // Lineage is unioned and observation ids are checked for overlap as sets.
    set_laws(&["l1", "l2", "l3"], |lineage| {
        adapt(&ty, signal_request(one(lineage, &["o"]), &["f"], binomial()))
    });
    set_laws(&["o1", "o2", "o3"], |observations| {
        adapt(&ty, signal_request(one(&["l"], observations), &["f"], binomial()))
    });
    // The target's variables are a set (a BTreeSet).
    set_laws(&["t", "y", "z"], |variables| {
        adapt(variables, signal_request(one(&["l"], &["o"]), &["f"], binomial()))
    });
    // Sources are pooled in declaration order.
    ordered_laws(&["A", "B"], |ids| {
        let sources = ids.iter().map(|id| prior_source(id, "target", &["l"], &["o"])).collect();
        adapt(&ty, signal_request(sources, &["f"], binomial()))
    });
    ordered_laws(&["u1", "u2", "u3"], |unlocks| {
        let signal = CandidateSignalSource::UnlockListOnly(
            unlocks.iter().map(|u| (*u).to_owned()).collect(),
        );
        adapt(&ty, signal_request(one(&["l"], &["o"]), &["f"], signal))
    });
    // Adjacent texts of a source are framed.
    injective(
        &ADVERSARIAL_TEXT,
        |parts| *parts,
        |(id, population)| {
            adapt(
                &ty,
                signal_request(
                    vec![prior_source(id, population, &["l"], &["o"])],
                    &["f"],
                    binomial(),
                ),
            )
        },
    );
}

fn obligation_graph() -> Admg {
    let mut dag = Dag::with_variables(4);
    for (a, b) in [(0, 1), (2, 0), (2, 1), (3, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
            .expect("an acyclic edge");
    }
    Admg::from_dag(&dag).expect("an admg")
}

#[test]
fn site_input_decision_programs_observed_variables_are_a_set() {
    let ctx = ctx();
    let owe = |observed: &[u32], population: &str| {
        let failed = IdentificationResult::not_identified(
            CausalQuery::average_effect(AverageEffectQuery::binary_ate(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
            )),
            DerivationTrace::default(),
            AssumptionSet::new(),
            IdentificationPerformanceRecord::default(),
        );
        let observed = observed.iter().map(|&raw| VariableId::from_raw(raw)).collect();
        Some(
            compile_evidence_obligations(
                failed,
                obligation_graph(),
                observed,
                population,
                None,
                &ctx,
            )
            .expect("compiles")
            .identity(),
        )
    };
    set_laws(&[0, 1, 3], |observed| owe(observed, "pop"));
    all_distinct(&[owe(&[0, 1], "pop"), owe(&[0, 1], "other")]);
}

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

/// A constraint every action satisfies, known by its name.
struct Named(String);

impl DecisionConstraint<f64, f64> for Named {
    fn name(&self) -> &str {
        &self.0
    }

    fn satisfaction_batch(&self, _actions: &[f64], _outcomes: &[f64], out: &mut [f64]) {
        out.fill(1.0);
    }
}

#[test]
fn site_input_decision_programs_finite_action_constraints_are_ordered() {
    let ctx = ctx();
    let decide = |names: &[&str], threshold: f64| {
        let constraints: Vec<Arc<dyn DecisionConstraint<f64, f64>>> = names
            .iter()
            .map(|name| {
                Arc::new(Named((*name).to_owned())) as Arc<dyn DecisionConstraint<f64, f64>>
            })
            .collect();
        let mut problem = DecisionProblem::new(vec![1.0, 2.0], Arc::new(Linear), constraints);
        problem.chance_threshold = threshold;
        Some(
            compile_finite_action_decision(problem, vec![-1.0, 3.0], &ctx)
                .expect("compiles")
                .identity(),
        )
    };
    // Satisfaction probabilities are reported per constraint in declaration order.
    ordered_laws(&["c1", "c2", "c3"], |names| decide(names, 1.0));
    all_distinct(&[decide(&[], 1.0), decide(&["c1"], 1.0), decide(&["c1", "c1"], 1.0)]);
    bit_exact(|threshold| decide(&["c1"], threshold));
}

fn prepared_search(seed: u64) -> PreparedAdaptivePolicySearch {
    let schedule = AdaptivePolicyFamily::new(
        "schedule",
        vec![PolicyFeature::Current(VariableId::from_raw(0))],
        &[(-4.0, 4.0)],
        PolicyRule::Schedule { steps: 1 },
        (0.0, 1.0),
        (-1.0, 1.0),
    )
    .expect("a family");
    PreparedAdaptivePolicySearch::compile(
        vec![schedule],
        None,
        PolicyConstraints::default(),
        PolicyObjective::ExpectedUtility,
        PolicySearchOptions { seed, max_evaluations: 600, ..PolicySearchOptions::default() },
    )
    .expect("a prepared search")
}

#[test]
fn site_input_decision_programs_policy_search_wraps_the_search_identity() {
    let ctx = ctx();
    let search = |seed: u64| {
        compile_adaptive_policy_search(prepared_search(seed), &ctx).expect("compiles").identity()
    };
    assert_eq!(search(11), search(11));
    all_distinct(&[search(11), search(12)]);
}
