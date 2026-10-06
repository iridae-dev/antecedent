//! X6 study planning end to end: declared studies compiled to hypothetical
//! deltas, the bounded plan, arrival of the real data re-identified through the
//! public X1/X9 route and executed against enumerated structural truth, and the
//! independently replayed `study_plan_v1` artifact.
//!
//! Fixtures: R-443 Figure 1(c,d) (X1, shared with the estimate-crate tests) and
//! the front-door surrogate-trial model (X9).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-estimate/tests/common/mod.rs"]
mod common;
#[path = "../../antecedent-estimate/tests/common/mixed_fixture.rs"]
mod mixed_fixture;

use std::sync::Arc;

use antecedent::design::{
    StudyArrivalDecision, StudyCandidate, StudyCost, StudyPlanArtifactWire, StudyPlanConsumeLimits,
    StudyPlanError, StudyPlanLimits, StudyPlanResult, StudyPlanRoute, plan_studies,
};
use antecedent_core::{
    DependenceGroup, DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ExecutionContext, InterventionAssignment, LawOrigin, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, SamplingSelection, SearchLimits, Value,
};
use antecedent_estimate::{evaluate_exact_mixed_source, evaluate_exact_mz_transport};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment as ExprInterventionAssignment, LawTolerance,
};
use antecedent_identify::{
    MixedSourceDecision, MzTransportDecision, bind_mixed_source_catalog, bind_mz_transport_catalog,
};
use common::mz_fixture::{
    X, Y, Z1, Z2, evidence, graph, query, source_a_scm, source_b_scm, sources, target_scm,
};
use common::z_scm::{Scm, risk_of, vid};

// ------------------------------------------------------------------ X1 (mz)

fn assign(variable: usize, level: bool) -> InterventionAssignment {
    InterventionAssignment { variable: vid(variable), value: Value::Bool(level) }
}

fn study(
    id: &str,
    population: &str,
    on: &[usize],
    levels: Option<Vec<Vec<(usize, bool)>>>,
    measured: &[usize],
    cost: u64,
) -> StudyCandidate {
    StudyCandidate {
        id: Arc::from(id),
        population: Arc::from(population),
        interventions: on.iter().map(|v| vid(*v)).collect(),
        levels: levels.map(|levels| {
            levels
                .into_iter()
                .map(|level| level.into_iter().map(|(v, l)| assign(v, l)).collect())
                .collect()
        }),
        measured: measured.iter().map(|v| vid(*v)).collect(),
        recruitment: Arc::from("independent recruitment, one arm per level"),
        cost: StudyCost { units: cost, sample_budget: 500 },
        requires: Arc::from([]),
        conflicts: Arc::from([]),
        feasibility_constraints: Arc::from([Arc::from("ethics approval in hand")]),
    }
}

/// The target's observational law of Figure 1(c,d) alone (regime 0), with the
/// fixture's environments.
fn mz_base() -> EvidenceCatalog {
    let (catalog, _) = evidence();
    EvidenceCatalog::try_new(
        Arc::clone(&catalog.environments),
        catalog.regimes.iter().filter(|r| r.id.raw() == 0).cloned().collect::<Vec<_>>(),
        catalog.bindings.iter().filter(|b| b.regime.raw() == 0).cloned().collect::<Vec<_>>(),
        None,
    )
    .unwrap()
}

fn mz_candidates() -> Vec<StudyCandidate> {
    vec![
        study(
            "a_do_z2",
            "a",
            &[Z2],
            Some(vec![vec![(Z2, false)], vec![(Z2, true)]]),
            &[Z1, X, Y],
            3,
        ),
        study("b_do_z1", "b", &[Z1], Some(vec![vec![(Z1, false)]]), &[X, Z2, Y], 2),
        study("b_observe", "b", &[], None, &[Z1, X, Z2, Y], 1),
    ]
}

fn mz_route() -> StudyPlanRoute {
    StudyPlanRoute::Mz(query(sources()))
}

fn mz_plan() -> StudyPlanResult {
    let ctx = ExecutionContext::for_tests(3);
    plan_studies(
        &graph(),
        &mz_route(),
        &mz_base(),
        &mz_candidates(),
        StudyPlanLimits::default(),
        &ctx,
    )
    .unwrap()
}

fn scm_for(population: &str) -> Scm {
    match population {
        "a" => source_a_scm(),
        "b" => source_b_scm(),
        _ => target_scm(),
    }
}

/// The arriving catalog: the base plus every proposed regime as available
/// evidence bound to `snapshot`, with each law enumerated from its population's
/// structural model. `tamper` may rewrite a law's probabilities.
fn arrive(
    base: &EvidenceCatalog,
    base_data: Option<&ExactTransportData>,
    regimes: &[EvidenceRegime],
    snapshot: &str,
    scm: &dyn Fn(&str) -> Scm,
    tamper: &dyn Fn(RegimeId, Vec<f64>) -> Vec<f64>,
) -> (EvidenceCatalog, ExactTransportData) {
    let mut all = base.regimes.to_vec();
    let mut bindings = base.bindings.to_vec();
    let mut laws = base_data.map(|d| d.laws().to_vec()).unwrap_or_default();
    for proposed in regimes {
        let mut regime = proposed.clone();
        regime.evidence_kind = EvidenceKind::Available;
        bindings.push(RegimeBinding {
            dataset_identity: None,
            regime: regime.id,
            snapshot_identity: Arc::from(snapshot),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        });
        let measured = regime.measured.iter().map(|v| v.as_usize()).collect::<Vec<_>>();
        let axes = measured
            .iter()
            .map(|i| DiscreteAxis {
                variable: vid(*i),
                values: Arc::from([Value::Bool(false), Value::Bool(true)]),
            })
            .collect::<Vec<_>>();
        let levels: Vec<Vec<(usize, u8)>> = if regime.intervention_values.is_empty() {
            let on = regime.interventions.iter().map(|v| v.as_usize()).collect::<Vec<_>>();
            (0..(1usize << on.len()))
                .map(|mask| {
                    on.iter()
                        .enumerate()
                        .map(|(b, v)| (*v, u8::from((mask >> b) & 1 == 1)))
                        .collect()
                })
                .collect()
        } else {
            vec![
                regime
                    .intervention_values
                    .iter()
                    .map(|a| (a.variable.as_usize(), u8::from(a.value == Value::Bool(true))))
                    .collect(),
            ]
        };
        let model = scm(&regime.population);
        for level in levels {
            let probabilities = tamper(regime.id, model.law(&level, &measured));
            laws.push(
                ExactDiscreteLaw::try_new(
                    regime.population.as_ref(),
                    regime.id,
                    level
                        .iter()
                        .map(|(v, l)| {
                            ExprInterventionAssignment::concrete(vid(*v), Value::Bool(*l == 1))
                        })
                        .collect::<Vec<_>>(),
                    axes.clone(),
                    probabilities,
                    snapshot,
                    LawTolerance::default(),
                )
                .unwrap(),
            );
        }
        all.push(regime);
    }
    (
        EvidenceCatalog::try_new(Arc::clone(&base.environments), all, bindings, None).unwrap(),
        ExactTransportData::try_new(laws, 4096).unwrap(),
    )
}

fn mz_base_data() -> ExactTransportData {
    let (_, data) = evidence();
    ExactTransportData::try_new(
        data.laws().iter().filter(|l| l.regime().raw() == 0).cloned().collect::<Vec<_>>(),
        4096,
    )
    .unwrap()
}

fn request(x: bool) -> Assignment {
    Assignment::from_pairs([(vid(X), Value::Bool(x))])
}

#[test]
fn the_sufficient_pair_s_arriving_data_identify_and_match_the_enumerated_truth() {
    let plan = mz_plan();
    let top = plan.proposal(0).expect("the complementary pair is sufficient");
    assert_eq!(top.proposal().candidates, [Arc::from("a_do_z2"), Arc::from("b_do_z1")]);
    assert!(plan.plan().minimal);
    // The regimes the arrival must deliver: a's do(Z2 = 0/1) and b's do(Z1 = 0).
    let regimes = top.delta().proposed_regimes.to_vec();
    assert_eq!(regimes.iter().map(|r| r.id.raw()).collect::<Vec<_>>(), [1, 2, 3]);
    let (actual, data) =
        arrive(&mz_base(), Some(&mz_base_data()), &regimes, "arrival-1", &scm_for, &|_, p| p);
    let ctx = ExecutionContext::for_tests(3);
    let arrival = top.receive(&actual, "arrival-1", &ctx).unwrap();
    let StudyArrivalDecision::Mz(decision) = arrival.decision else {
        panic!("an mz plan re-identifies through the mz route");
    };
    let MzTransportDecision::Identified { derivation, cited } = *decision else {
        panic!("the arriving data identify");
    };
    assert_eq!(cited.iter().map(|r| r.raw()).collect::<Vec<_>>(), [1, 2, 3]);
    let bound = bind_mz_transport_catalog(&graph(), &derivation, &actual).unwrap();
    let truth = target_scm();
    for x in [false, true] {
        let point = risk_of(
            &evaluate_exact_mz_transport(
                &bound,
                data.clone(),
                request(x),
                ExactEvaluationLimits::default(),
                &ctx,
            )
            .unwrap(),
        );
        let expected = truth.risk(&[(X, u8::from(x))], Y);
        assert!((point - expected).abs() < 1e-12, "do(X={x}): {point} vs truth {expected}");
    }
    // The fixture is not trivial: the target's observational conditional differs.
    let joint = truth.law(&[], &[X, Y]);
    let naive = joint[3] / (joint[2] + joint[3]);
    assert!((naive - truth.risk(&[(X, 1)], Y)).abs() > 1e-3);
}

#[test]
fn arriving_data_of_another_shape_or_without_support_are_refused() {
    let plan = mz_plan();
    let top = plan.proposal(0).unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let regimes = top.delta().proposed_regimes.to_vec();
    let refuse = |actual: &EvidenceCatalog, snapshot: &str| {
        let error = top.receive(actual, snapshot, &ctx).unwrap_err();
        (error.code, error.detail)
    };
    let mismatch = ("invalid_argument", "study_plan.arrival_mismatch");
    // Another margin than proposed.
    let mut narrow = regimes.clone();
    narrow[2].measured = Arc::from([vid(X), vid(Y)]);
    let (actual, _) = arrive(&mz_base(), None, &narrow, "arrival-1", &scm_for, &|_, p| p);
    assert_eq!(refuse(&actual, "arrival-1"), mismatch);
    // One level missing.
    let (actual, _) = arrive(&mz_base(), None, &regimes[..2], "arrival-1", &scm_for, &|_, p| p);
    assert_eq!(refuse(&actual, "arrival-1"), mismatch);
    // Bound to another provider snapshot than the one claimed.
    let (actual, _) = arrive(&mz_base(), None, &regimes, "arrival-1", &scm_for, &|_, p| p);
    assert_eq!(refuse(&actual, "arrival-2"), mismatch);
    // The right shape, but a selected-sample law or a fitted-model artifact:
    // not the measured whole-population law the plan verified.
    let mut selected = regimes.clone();
    selected[2].selection = SamplingSelection::SelectedOn { variables: Arc::from([vid(Y)]) };
    let (actual, _) = arrive(&mz_base(), None, &selected, "arrival-1", &scm_for, &|_, p| p);
    assert_eq!(refuse(&actual, "arrival-1"), mismatch);
    let mut fitted = regimes.clone();
    fitted[2].origin = LawOrigin::ModelArtifact { artifact: Arc::from("posterior-7") };
    let (actual, _) = arrive(&mz_base(), None, &fitted, "arrival-1", &scm_for, &|_, p| p);
    assert_eq!(refuse(&actual, "arrival-1"), mismatch);
    // The right shape with no mass at X = 1 in b's trial: arrival re-identifies
    // (support is not structural), and the ordinary evaluator refuses the point.
    let (actual, data) =
        arrive(&mz_base(), Some(&mz_base_data()), &regimes, "arrival-1", &scm_for, &|id, p| {
            if id.raw() != 3 {
                return p;
            }
            // Axes are (X, Z2, Y), X the most significant bit: rows 4..8 have X = 1.
            let kept = p[..4].iter().sum::<f64>();
            p.iter().enumerate().map(|(i, q)| if i >= 4 { 0.0 } else { q / kept }).collect()
        });
    let arrival = top.receive(&actual, "arrival-1", &ctx).unwrap();
    let StudyArrivalDecision::Mz(decision) = arrival.decision else { panic!() };
    let MzTransportDecision::Identified { derivation, .. } = *decision else { panic!() };
    let bound = bind_mz_transport_catalog(&graph(), &derivation, &actual).unwrap();
    let error = evaluate_exact_mz_transport(
        &bound,
        data,
        request(true),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap_err();
    assert!(error.to_string().contains("support") || error.to_string().contains("zero"), "{error}");
}

fn to_bytes(wire: &StudyPlanArtifactWire) -> Vec<u8> {
    serde_json::to_vec(wire).unwrap()
}

fn from_bytes(bytes: &[u8]) -> StudyPlanArtifactWire {
    serde_json::from_slice(bytes).unwrap()
}

#[test]
fn the_exported_plan_is_replayed_by_an_independent_consumer() {
    let produced = mz_plan();
    let bytes = to_bytes(&produced.to_artifact().unwrap());
    drop(produced);
    let wire = from_bytes(&bytes);
    let ctx = ExecutionContext::for_tests(9);
    let consumed = wire.consume_with_limits(StudyPlanConsumeLimits::default(), &ctx).unwrap();
    // The consumer's plan is the producer's, bit for bit on the wire.
    assert_eq!(to_bytes(&consumed.to_artifact().unwrap()), bytes);
    assert_eq!(consumed.plan().proposals[0].cost_units, 5);
    assert_eq!(wire.plan.proposals[0].repairs.len(), 3);
    assert!(wire.plan.minimal);
    assert_eq!(wire.premises.ranking, "x6.ranking.v1");
    // Unknown fields are refused at every level.
    let mut json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    json["plan"]["proposals"][0]["extra"] = serde_json::json!(1);
    assert!(serde_json::from_value::<StudyPlanArtifactWire>(json.clone()).is_err());
    let mut json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    json["premises"]["candidates"][0]["probability_of_success"] = serde_json::json!(0.9);
    assert!(serde_json::from_value::<StudyPlanArtifactWire>(json).is_err());
    // The consumed plan still accepts the arrival.
    let top = consumed.proposal(0).unwrap();
    let (actual, _) =
        arrive(&mz_base(), None, &top.delta().proposed_regimes, "arrival-1", &scm_for, &|_, p| p);
    assert!(top.receive(&actual, "arrival-1", &ctx).is_ok());
}

fn consume(wire: &StudyPlanArtifactWire) -> Result<StudyPlanResult, StudyPlanError> {
    wire.consume_with_limits(StudyPlanConsumeLimits::default(), &ExecutionContext::for_tests(9))
}

fn detail(result: Result<StudyPlanResult, StudyPlanError>) -> (&'static str, &'static str) {
    let error = result.unwrap_err();
    (error.code, error.detail)
}

#[test]
fn a_mutated_plan_artifact_fails_consumption_for_the_right_reason() {
    let original = mz_plan().to_artifact().unwrap();
    let invalid = ("transport_not_certified", "study_plan.invalid_artifact");
    // Unsealed edits: the digests catch them.
    let mut edited = original.clone();
    edited.plan.minimal = false;
    assert_eq!(detail(consume(&edited)), invalid);
    let mut lineage = original.clone();
    lineage.data.catalog.bindings.clear();
    assert_eq!(detail(consume(&lineage)), invalid);
    let mut version = original.clone();
    version.version = 2;
    assert_eq!(detail(consume(&version)), invalid);
    // Re-sealed semantic edits keep valid digests and fail the full replay.
    let resealed = |edit: &dyn Fn(&mut StudyPlanArtifactWire)| {
        let mut wire = original.clone();
        edit(&mut wire);
        let wire = wire.sealed().unwrap();
        assert_ne!(wire.plan_digest, original.plan_digest, "the edit must change the artifact");
        let error = consume(&wire).unwrap_err();
        (error.code, error.detail, error.message)
    };
    let replay_differs = |edit: &dyn Fn(&mut StudyPlanArtifactWire)| {
        let (code, detail, message) = resealed(edit);
        assert_eq!((code, detail), invalid);
        assert!(message.contains("replayed plan differs"), "{message}");
    };
    replay_differs(&|w| w.plan.minimal = false);
    replay_differs(&|w| w.plan.proposals[0].cost_units = 4);
    replay_differs(&|w| {
        w.plan.proposals[0].repairs[0].required_margin.pop();
    });
    replay_differs(&|w| w.plan.failure.detail = "mz_transport.search_incomplete".into());
    replay_differs(&|w| w.plan.subsets[0].status = "sufficient".into());
    replay_differs(&|w| w.plan.subsets.swap(0, 1));
    replay_differs(&|w| w.plan.operations_consumed += 1);
    replay_differs(&|w| {
        w.plan.stop = Some(antecedent_design::StudyPlanStopWire {
            kind: "proposal_cap".into(),
            stop: None,
            operations_limit: None,
            depth_limit: None,
            memory_limit_bytes: None,
            operations_consumed: None,
            depth_reached: None,
            explored: Vec::new(),
            unevaluated: Vec::new(),
        });
    });
    // Premises: a cost, a margin and the limits change what the replay produces.
    replay_differs(&|w| w.premises.candidates[0].cost_units = 1);
    replay_differs(&|w| w.premises.limits.operations -= 1);
    let (code, detail, _) = resealed(&|w| w.premises.ranking = "x6.ranking.v0".into());
    assert_eq!((code, detail), invalid);
    let (code, detail, _) = resealed(&|w| w.premises.rule_set = "x9.rules.v1".into());
    assert_eq!((code, detail), invalid);
    // A candidate outside the controllables no longer compiles on replay.
    let (code, detail, message) = resealed(&|w| w.premises.candidates[0].interventions = vec![0]);
    assert_eq!((code, detail), invalid, "{message}");
    // Stored limits above the consumer's refuse before any work: even a
    // cancelled consumer reports the limits, not a stop.
    let cancelled = ExecutionContext::for_tests(9);
    cancelled.cancellation.cancel();
    let small = StudyPlanConsumeLimits {
        search: SearchLimits { operations: 1000, depth: 24 },
        ..StudyPlanConsumeLimits::default()
    };
    let error = original.consume_with_limits(small, &cancelled).unwrap_err();
    assert_eq!((error.code, error.detail), ("route_not_supported", "study_plan.bounds_exceeded"));
    // Lineage: re-sealed with every base binding (snapshot) cleared, or with a
    // base regime bound only to a planning placeholder, the plan no longer
    // replays (a plan needs provider lineage for every available base regime).
    let mut unbound = original.clone();
    unbound.data.catalog.bindings.clear();
    let unbound = unbound.sealed().unwrap();
    assert_ne!(unbound.data_digest, original.data_digest);
    let (code, detail, message) = {
        let error = consume(&unbound).unwrap_err();
        (error.code, error.detail, error.message)
    };
    assert_eq!((code, detail), invalid);
    assert!(message.contains("does not replay") && message.contains("lineage"), "{message}");
    let (code, detail, message) = resealed(&|w| {
        w.data.catalog.bindings[0].snapshot_identity = "hypothetical:0".into();
    });
    assert_eq!((code, detail), invalid);
    assert!(message.contains("lineage"), "{message}");
    // What replay cannot tell: a base binding renamed to another real snapshot
    // is a correct plan of that stated lineage; `receive` then requires the
    // arriving catalog to keep exactly that binding.
}

#[test]
fn a_changed_base_catalog_forces_a_re_plan() {
    let plan = mz_plan();
    let top = plan.proposal(0).unwrap();
    let ctx = ExecutionContext::for_tests(3);
    // The arriving catalog changed a base regime: not the frozen base.
    let (mut actual, _) =
        arrive(&mz_base(), None, &top.delta().proposed_regimes, "arrival-1", &scm_for, &|_, p| p);
    let mut regimes = actual.regimes.to_vec();
    regimes[0].study = Some(Arc::from("relabelled"));
    actual.regimes = regimes.into();
    let error = top.receive(&actual, "arrival-1", &ctx).unwrap_err();
    assert_eq!((error.code, error.detail), ("invalid_argument", "study_plan.arrival_mismatch"));
    // An artifact whose base now already holds b's trial (re-sealed): the
    // replayed plan differs (b's trial alone is no longer a candidate gap), so
    // the stored plan is refused and a new plan is needed.
    let original = plan.to_artifact().unwrap();
    let mut changed = original.clone();
    let (with_trial, _) = arrive(
        &mz_base(),
        None,
        &[top.delta().proposed_regimes[2].clone()],
        "b-trial",
        &scm_for,
        &|_, p| p,
    );
    let mut with_trial = with_trial;
    let mut shifted = with_trial.regimes.to_vec();
    shifted[1].id = RegimeId::from_raw(40);
    shifted[1].label = Some(Arc::from("b-trial"));
    let mut bindings = with_trial.bindings.to_vec();
    bindings[1].regime = RegimeId::from_raw(40);
    with_trial =
        EvidenceCatalog::try_new(Arc::clone(&with_trial.environments), shifted, bindings, None)
            .unwrap();
    changed.data.catalog = antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(
        &with_trial.canonicalized().unwrap(),
    );
    let changed = changed.sealed().unwrap();
    let error = consume(&changed).unwrap_err();
    assert_eq!(
        (error.code, error.detail),
        ("transport_not_certified", "study_plan.invalid_artifact")
    );
    // Planning afresh on the changed base gives a different, cheaper plan.
    let fresh = plan_studies(
        &graph(),
        &mz_route(),
        &with_trial,
        &mz_candidates(),
        StudyPlanLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(fresh.plan().proposals[0].candidates, [Arc::from("a_do_z2")]);
}

#[test]
fn impossible_designs_do_not_compile() {
    let ctx = ExecutionContext::for_tests(3);
    let invalid = ("invalid_argument", "study_plan.invalid_candidate");
    let refuse = |candidate: StudyCandidate| {
        let error = plan_studies(
            &graph(),
            &mz_route(),
            &mz_base(),
            &[candidate],
            StudyPlanLimits::default(),
            &ctx,
        )
        .unwrap_err();
        (error.code, error.detail)
    };
    // Outside a's controllables, a target experiment, a level outside the binary
    // domain, levels on an observational study, an overlapping margin, a
    // variable outside the graph, and no recruitment declaration.
    assert_eq!(refuse(study("a_do_z1", "a", &[Z1], None, &[X, Z2, Y], 1)), invalid);
    assert_eq!(refuse(study("target_do_x", "target", &[X], None, &[Z1, Z2, Y], 1)), invalid);
    let mut bad_level = study("a_do_z2", "a", &[Z2], None, &[Z1, X, Y], 1);
    bad_level.levels = Some(Arc::from([Arc::from([InterventionAssignment {
        variable: vid(Z2),
        value: Value::f64(2.0),
    }])]));
    assert_eq!(refuse(bad_level), invalid);
    assert_eq!(refuse(study("obs", "b", &[], Some(vec![vec![(Z1, false)]]), &[X], 1)), invalid);
    assert_eq!(refuse(study("overlap", "a", &[Z2], None, &[Z2, Y], 1)), invalid);
    assert_eq!(refuse(study("outside", "b", &[], None, &[9], 1)), invalid);
    let mut silent = study("silent", "b", &[], None, &[X], 1);
    silent.recruitment = Arc::from(" ");
    assert_eq!(refuse(silent), invalid);
    let mut free = study("free", "b", &[], None, &[X], 1);
    free.cost.units = 0;
    assert_eq!(refuse(free), invalid);
}

// --------------------------------------------------------------- X9 (mixed)

use mixed_fixture::{X as FX, Y as FY, Z as FZ, build, frontdoor_scm, study as mixed_study};

/// The front door `X -> Z -> Y`, `X <-> Y`: an observational (X, Z) study, and a
/// surrogate trial on Z that published only separate marginals of X and Y.
fn surrogate_base() -> (antecedent_graph::Admg, EvidenceCatalog, ExactTransportData) {
    let g = mixed_fixture::graph(3, &[(0, 1), (1, 2)], &[(0, 2)]);
    let (catalog, data) = build(&frontdoor_scm(), &[mixed_study("observational", &[], &[FX, FZ])]);
    let mut marginals = EvidenceRegime::try_new(
        RegimeId::from_raw(2),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [vid(FZ)],
        [],
        [vid(FX), vid(FY)],
        "target",
        DistributionAvailability::SeparateMarginals { variables: Arc::from([vid(FX), vid(FY)]) },
    )
    .unwrap();
    marginals.study = Some(Arc::from("trial-marginals"));
    let mut regimes = catalog.regimes.to_vec();
    regimes.push(marginals);
    let mut bindings = catalog.bindings.to_vec();
    bindings.push(RegimeBinding {
        dataset_identity: None,
        regime: RegimeId::from_raw(2),
        snapshot_identity: Arc::from("trial-marginals"),
        schema_names: Arc::from([]),
        sampling: SamplingDesign::Independent,
        weights: None,
        dependence: DependenceGroup::IndependentStudies,
    });
    let catalog =
        EvidenceCatalog::try_new(Arc::clone(&catalog.environments), regimes, bindings, None)
            .unwrap();
    (g, catalog, data)
}

fn surrogate_route() -> StudyPlanRoute {
    StudyPlanRoute::Mixed(mixed_fixture::query(FY, FX))
}

#[test]
fn a_declared_margin_repairs_the_named_missing_joint() {
    let (g, base, base_data) = surrogate_base();
    let route = surrogate_route();
    let candidates = vec![
        study("trial_xy", "target", &[FZ], None, &[FX, FY], 3),
        study("trial_y", "target", &[FZ], None, &[FY], 2),
        study("full_observational", "target", &[], None, &[FX, FZ, FY], 5),
    ];
    let ctx = ExecutionContext::for_tests(5);
    let plan =
        plan_studies(&g, &route, &base, &candidates, StudyPlanLimits::for_route(&route), &ctx)
            .unwrap();
    // The frozen failure names the exact joint the trial published only as marginals.
    let failure = &plan.plan().failure;
    assert_eq!(
        (failure.code, failure.detail),
        ("transport_missing_evidence", "mixed_search.missing_joint")
    );
    assert!(failure.facts[0].starts_with("missing_joint:regime=2"));
    // The sub-margin trial (only Y measured) is cheapest and sufficient; the
    // wider trial is sufficient too; the full observational joint is sufficient
    // through the named target-first sID route.
    let ranked = plan
        .plan()
        .proposals
        .iter()
        .map(|p| (p.candidates[0].to_string(), p.cost_units, p.stage.clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        ranked,
        [
            ("trial_y".to_owned(), 2, "mixed_source:rule_search".to_owned()),
            ("trial_xy".to_owned(), 3, "mixed_source:rule_search".to_owned()),
            ("full_observational".to_owned(), 5, "mixed_source:named:target_first_sid".to_owned()),
        ]
    );
    assert!(plan.plan().minimal);
    // The repaired proof step and the margin the proof reads from the trial.
    let top = &plan.plan().proposals[0];
    assert_eq!(top.repairs.len(), 1);
    let repair = &top.repairs[0];
    assert!(!repair.proof_steps.is_empty());
    assert_eq!(repair.required_margin, [vid(FY)]);
    assert_eq!(repair.intervened, [vid(FZ)]);
    // The trial's data arrive: the public route identifies and the point is the truth.
    let proposal = plan.proposal(0).unwrap();
    let (actual, data) = arrive(
        &base,
        Some(&base_data),
        &proposal.delta().proposed_regimes,
        "trial-y",
        &|_| frontdoor_scm(),
        &|_, p| p,
    );
    let arrival = proposal.receive(&actual, "trial-y", &ctx).unwrap();
    let StudyArrivalDecision::Mixed(decision) = arrival.decision else { panic!() };
    let MixedSourceDecision::Identified { derivation, .. } = *decision else {
        panic!("the trial's joint identifies through the rule search");
    };
    let bound = bind_mixed_source_catalog(&g, &derivation, &actual).unwrap();
    let truth = frontdoor_scm();
    for x in [false, true] {
        let point = risk_of(
            &evaluate_exact_mixed_source(
                &bound,
                data.clone(),
                Assignment::from_pairs([(vid(FX), Value::Bool(x))]),
                ExactEvaluationLimits::default(),
                &ctx,
            )
            .unwrap(),
        );
        let expected = truth.risk(&[(FX, u8::from(x))], FY);
        assert!((point - expected).abs() < 1e-12, "do(X={x}): {point} vs {expected}");
    }
}

#[test]
fn a_value_restricted_study_is_refused_on_the_mixed_route() {
    let (g, base, _) = surrogate_base();
    let route = surrogate_route();
    let ctx = ExecutionContext::for_tests(5);
    let restricted = study("trial_z1", "target", &[FZ], Some(vec![vec![(FZ, true)]]), &[FY], 1);
    let error =
        plan_studies(&g, &route, &base, &[restricted], StudyPlanLimits::for_route(&route), &ctx)
            .unwrap_err();
    assert_eq!((error.code, error.detail), ("invalid_argument", "study_plan.invalid_candidate"));
    assert!(error.message.contains("restricts its levels"), "{}", error.message);
}

// ------------------------------------------------------- stops and bounds

struct CancelAfter {
    token: antecedent_core::CancellationToken,
    reports: std::sync::Mutex<usize>,
    after: usize,
}

impl antecedent_core::ProgressSink for CancelAfter {
    fn report(&self, _fraction: f64, _stage: &str) {
        let mut reports = self.reports.lock().unwrap();
        *reports += 1;
        if *reports == self.after {
            self.token.cancel();
        }
    }
}

fn cancelling_after(reports: usize) -> ExecutionContext {
    let mut ctx = ExecutionContext::for_tests(9);
    ctx.progress = Some(Arc::new(CancelAfter {
        token: ctx.cancellation.clone(),
        reports: std::sync::Mutex::new(0),
        after: reports,
    }));
    ctx
}

const BUDGET: (&str, &str) = ("transport_budget_cancel", "study_plan.budget");

#[test]
fn a_cancelled_replay_is_a_budget_stop_never_a_verdict_on_the_artifact() {
    let wire = mz_plan().to_artifact().unwrap();
    // Cancelled before the replay starts (the stored limits are affordable).
    let cancelled = ExecutionContext::for_tests(9);
    cancelled.cancellation.cancel();
    let error =
        wire.consume_with_limits(StudyPlanConsumeLimits::default(), &cancelled).unwrap_err();
    assert_eq!((error.code, error.detail), BUDGET, "{}", error.message);
    // Cancelled in the middle of the replay, after two subsets were decided.
    let error = wire
        .consume_with_limits(StudyPlanConsumeLimits::default(), &cancelling_after(2))
        .unwrap_err();
    assert_eq!((error.code, error.detail), BUDGET, "{}", error.message);
    assert_eq!(error.receipt.map(|r| r.stop), Some(antecedent_core::SearchStop::Cancelled));
    // The same artifact replays once nothing cancels it.
    assert!(consume(&wire).is_ok());
    // A plan that cancellation stopped cannot be exported: no consumer could
    // replay it.
    let stopped = plan_studies(
        &graph(),
        &mz_route(),
        &mz_base(),
        &mz_candidates(),
        StudyPlanLimits::default(),
        &cancelling_after(2),
    )
    .unwrap();
    assert!(stopped.plan().receipt().is_some());
    let error = stopped.to_artifact().unwrap_err();
    assert_eq!((error.code, error.detail), BUDGET);
}

#[test]
fn an_arrival_the_public_route_cannot_finish_is_a_budget_stop() {
    let plan = mz_plan();
    let top = plan.proposal(0).unwrap();
    let (actual, _) =
        arrive(&mz_base(), None, &top.delta().proposed_regimes, "arrival-1", &scm_for, &|_, p| p);
    // Cancellation.
    let cancelled = ExecutionContext::for_tests(3);
    cancelled.cancellation.cancel();
    let error = top.receive(&actual, "arrival-1", &cancelled).unwrap_err();
    assert_eq!((error.code, error.detail), BUDGET, "{}", error.message);
    assert_eq!(error.receipt.map(|r| r.stop), Some(antecedent_core::SearchStop::Cancelled));
    // Exhaustion: the context's hard memory limit stops the public route.
    let mut tight = ExecutionContext::for_tests(3);
    tight.memory =
        antecedent_core::MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(64) };
    let error = top.receive(&actual, "arrival-1", &tight).unwrap_err();
    assert_eq!((error.code, error.detail), BUDGET, "{}", error.message);
    assert_eq!(error.receipt.map(|r| r.stop), Some(antecedent_core::SearchStop::Memory));
    // The mixed route: the same, never arrival_not_identified.
    let (g, base, _) = surrogate_base();
    let route = surrogate_route();
    let candidates = vec![study("trial_y", "target", &[FZ], None, &[FY], 2)];
    let ctx = ExecutionContext::for_tests(5);
    let mixed =
        plan_studies(&g, &route, &base, &candidates, StudyPlanLimits::for_route(&route), &ctx)
            .unwrap();
    let proposal = mixed.proposal(0).unwrap();
    let (actual, _) = arrive(
        &base,
        None,
        &proposal.delta().proposed_regimes,
        "trial-y",
        &|_| frontdoor_scm(),
        &|_, p| p,
    );
    for (ctx, stop) in [
        (&cancelled, antecedent_core::SearchStop::Cancelled),
        (&tight, antecedent_core::SearchStop::Memory),
    ] {
        let error = proposal.receive(&actual, "trial-y", ctx).unwrap_err();
        assert_eq!((error.code, error.detail), BUDGET, "{}", error.message);
        assert_eq!(error.receipt.map(|r| r.stop), Some(stop));
    }
    assert!(proposal.receive(&actual, "trial-y", &ctx).is_ok());
}

#[test]
fn a_placeholder_snapshot_never_poses_as_delivered_data() {
    let plan = mz_plan();
    let top = plan.proposal(0).unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let regimes = top.delta().proposed_regimes.to_vec();
    // Every arriving regime bound to a placeholder-namespace snapshot.
    let (actual, _) = arrive(&mz_base(), None, &regimes, "hypothetical:1", &scm_for, &|_, p| p);
    let error = top.receive(&actual, "hypothetical:1", &ctx).unwrap_err();
    assert_eq!((error.code, error.detail), ("invalid_argument", "study_plan.arrival_mismatch"));
    assert!(error.message.contains("planning preview"), "{}", error.message);
    // The planner's own placeholder bindings are in that namespace.
    let preview = top.delta().preview_catalog(&mz_base()).unwrap();
    let placeholders = top.delta().with_placeholder_bindings(&preview).unwrap();
    let placeholder = placeholders
        .bindings
        .iter()
        .find(|b| b.regime == regimes[0].id)
        .map(|b| b.snapshot_identity.clone())
        .unwrap();
    let error = top.receive(&placeholders, placeholder, &ctx).unwrap_err();
    assert_eq!(error.detail, "study_plan.arrival_mismatch");
    assert!(error.message.contains("planning preview"), "{}", error.message);
    // A real provider snapshot is accepted.
    let (actual, _) = arrive(&mz_base(), None, &regimes, "arrival-1", &scm_for, &|_, p| p);
    assert!(top.receive(&actual, "arrival-1", &ctx).is_ok());
}

#[test]
fn bounds_are_checked_before_any_candidate_compiles() {
    let ctx = ExecutionContext::for_tests(3);
    let refuse = |candidates: &[StudyCandidate], limits: StudyPlanLimits| {
        let error =
            plan_studies(&graph(), &mz_route(), &mz_base(), candidates, limits, &ctx).unwrap_err();
        (error.code, error.detail)
    };
    let bound = ("route_not_supported", "study_plan.bounds_exceeded");
    // A candidate that cannot compile (no recruitment declaration), sorted first.
    let mut broken = study("a_broken", "b", &[], None, &[X], 1);
    broken.recruitment = Arc::from(" ");
    assert_eq!(
        refuse(&[broken.clone()], StudyPlanLimits::default()),
        ("invalid_argument", "study_plan.invalid_candidate")
    );
    // Seventeen candidates: the count refuses before the broken one compiles.
    let mut many = vec![broken.clone()];
    many.extend((0..16).map(|i| study(&format!("obs{i:02}"), "b", &[], None, &[X], 1)));
    assert_eq!(refuse(&many, StudyPlanLimits::default()), bound);
    // Nine level combinations in one candidate, likewise.
    let wide = study("z_wide", "a", &[Z2], Some(vec![vec![(Z2, false)]; 9]), &[Z1, X, Y], 1);
    assert_eq!(refuse(&[broken.clone(), wide], StudyPlanLimits::default()), bound);
    // Plan limits above the caps, likewise.
    let over = StudyPlanLimits {
        search: SearchLimits { operations: 200_001, depth: 24 },
        ..StudyPlanLimits::default()
    };
    assert_eq!(refuse(&[broken], over), bound);
}

#[test]
fn a_candidate_label_that_a_base_regime_already_uses_is_refused() {
    let base = mz_base();
    let mut regimes = base.regimes.to_vec();
    regimes[0].label = Some(Arc::from("b_do_z1#0"));
    let base = EvidenceCatalog::try_new(
        Arc::clone(&base.environments),
        regimes,
        base.bindings.to_vec(),
        None,
    )
    .unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let error = plan_studies(
        &graph(),
        &mz_route(),
        &base,
        &mz_candidates(),
        StudyPlanLimits::default(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!((error.code, error.detail), ("invalid_argument", "study_plan.invalid_candidate"));
    assert!(error.message.contains("b_do_z1#0"), "{}", error.message);
}

#[test]
fn a_consumer_refuses_a_stored_memory_cap_above_its_own_or_the_context_s_before_work() {
    use antecedent_identify::STUDY_PLAN_MEMORY_BYTES;
    let wire = mz_plan().to_artifact().unwrap();
    assert_eq!(wire.premises.limits.memory_limit_bytes, STUDY_PLAN_MEMORY_BYTES);
    let bound = ("route_not_supported", "study_plan.bounds_exceeded");
    // A cancelled consumer: a refusal here is taken before any replay work.
    let cancelled = || {
        let ctx = ExecutionContext::for_tests(9);
        ctx.cancellation.cancel();
        ctx
    };
    let lower = StudyPlanConsumeLimits {
        memory_limit_bytes: STUDY_PLAN_MEMORY_BYTES - 1,
        ..StudyPlanConsumeLimits::default()
    };
    let error = wire.consume_with_limits(lower, &cancelled()).unwrap_err();
    assert_eq!((error.code, error.detail), bound);
    let mut hard = cancelled();
    hard.memory = antecedent_core::MemoryBudget {
        soft_limit_bytes: None,
        hard_limit_bytes: Some(STUDY_PLAN_MEMORY_BYTES - 1),
    };
    let error = wire.consume_with_limits(StudyPlanConsumeLimits::default(), &hard).unwrap_err();
    assert_eq!((error.code, error.detail), bound);
    // At exactly the stored cap the replay proceeds (here to the cancellation).
    let error =
        wire.consume_with_limits(StudyPlanConsumeLimits::default(), &cancelled()).unwrap_err();
    assert_eq!((error.code, error.detail), BUDGET);
}

/// Compatibility: another artifact kind or format version is refused, sealed or
/// not, so a reader never replays a plan whose format it does not know.
#[test]
fn another_artifact_kind_or_version_is_refused_even_when_resealed() {
    let original = mz_plan().to_artifact().unwrap();
    let invalid = ("transport_not_certified", "study_plan.invalid_artifact");
    let mut kind = original.clone();
    kind.kind = "study_plan_v2".into();
    assert_eq!(detail(consume(&kind)), invalid);
    let mut version = original.clone();
    version.version += 1;
    assert_eq!(detail(consume(&version)), invalid);
    for edit in [
        &(|w: &mut StudyPlanArtifactWire| w.kind = "study_plan_v2".into())
            as &dyn Fn(&mut StudyPlanArtifactWire),
        &|w: &mut StudyPlanArtifactWire| w.version += 1,
    ] {
        let mut wire = original.clone();
        edit(&mut wire);
        let refused = wire.sealed().map_or_else(|e| (e.code, e.detail), |w| detail(consume(&w)));
        assert_eq!(refused, invalid);
    }
}
