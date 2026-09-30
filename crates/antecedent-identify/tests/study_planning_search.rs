//! X6 study planning core: bounded, cost-ordered subset search over declared
//! candidate studies, re-deciding the X1 (mz) route on each preview catalog
//! under one shared search budget.
//!
//! The mz fixture is R-443 Figure 1(c,d) (`Z1 -> X -> Z2 -> Y`, `Z1 <-> X`,
//! `Z1 <-> Z2`, `Z1 <-> Y`): source `a` changes Z1 and Z2 and can run do(Z2);
//! source `b` changes Z1 and Y and can run do(Z1 = 0). The target supplies its
//! observational law.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex};

use antecedent_core::{
    CancellationToken, DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog,
    EvidenceCatalogDelta, EvidenceKind, EvidenceRegime, ExecutionContext, InterventionAssignment,
    MemoryBudget, ProgressSink, RegimeBinding, RegimeId, RegimeKind, SamplingDesign, SearchLimits,
    SearchStop, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    MixedSourceQuery, MzTransportQuery, STUDY_PLAN_DEFAULT_LIMITS, StudyPlan, StudyPlanCandidate,
    StudyPlanLimits, StudyPlanRefusal, StudyPlanRoute, StudyPlanStop, StudyProposalDerivation,
    StudySubsetOutcome, ZTransportSourceSpec, plan_study_additions,
};

const Z1: u32 = 0;
const X: u32 = 1;
const Z2: u32 = 2;
const Y: u32 = 3;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn fig_1_graph() -> Admg {
    let mut g = Admg::with_variables(4);
    for (a, b) in [(Z1, X), (X, Z2), (Z2, Y)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for (a, b) in [(Z1, X), (Z1, Z2), (Z1, Y)] {
        g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    g
}

fn source(population: &str, controllable: &[u32], selection: &[u32]) -> ZTransportSourceSpec {
    ZTransportSourceSpec {
        population: Arc::from(population),
        controllable: controllable.iter().copied().map(v).collect::<Vec<_>>().into(),
        experiment_assignment: Arc::from([]),
        selection_targets: selection.iter().copied().map(v).collect::<Vec<_>>().into(),
    }
}

fn fig_1_sources() -> Vec<ZTransportSourceSpec> {
    let mut b = source("b", &[Z1], &[Z1, Y]);
    b.experiment_assignment =
        Arc::from([InterventionAssignment { variable: v(Z1), value: Value::Bool(false) }]);
    vec![source("a", &[Z2], &[Z1, Z2]), b]
}

fn mz(sources: Vec<ZTransportSourceSpec>) -> StudyPlanRoute {
    StudyPlanRoute::Mz(MzTransportQuery {
        outcomes: Arc::from([v(Y)]),
        treatments: Arc::from([v(X)]),
        target: Arc::from("target"),
        sources: sources.into(),
    })
}

fn regime(id: u32, kind: EvidenceKind, population: &str, on: &[(u32, bool)]) -> EvidenceRegime {
    let intervened = on.iter().map(|(x, _)| *x).collect::<Vec<_>>();
    let measured = (0..4).filter(|i| !intervened.contains(i)).map(v).collect::<Vec<_>>();
    EvidenceRegime::try_new(
        RegimeId::from_raw(id),
        if on.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
        kind,
        intervened.iter().copied().map(v).collect::<Vec<_>>(),
        on.iter()
            .map(|(x, level)| InterventionAssignment {
                variable: v(*x),
                value: Value::Bool(*level),
            })
            .collect::<Vec<_>>(),
        measured,
        population,
        DistributionAvailability::Joint,
    )
    .unwrap()
}

fn catalog(regimes: Vec<EvidenceRegime>) -> EvidenceCatalog {
    let bindings = regimes
        .iter()
        .map(|r| RegimeBinding {
            dataset_identity: None,
            regime: r.id,
            snapshot_identity: Arc::from(format!("snapshot-{}", r.id.raw())),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        })
        .collect::<Vec<_>>();
    let coordinates = (0..4)
        .map(|i| VariableCoordinate { variable: v(i), domain: VariableDomain::Binary, unit: None })
        .collect::<Vec<_>>();
    let environments = ["target", "a", "b"]
        .into_iter()
        .map(|p| Environment::try_new(p, coordinates.clone(), []).unwrap())
        .collect::<Vec<_>>();
    EvidenceCatalog::try_new(environments, regimes, bindings, None).unwrap()
}

/// The target's observational law only.
fn base() -> EvidenceCatalog {
    catalog(vec![regime(0, EvidenceKind::Available, "target", &[])])
}

fn candidate(
    base: &EvidenceCatalog,
    id: &str,
    cost: u64,
    regimes: Vec<EvidenceRegime>,
) -> StudyPlanCandidate {
    StudyPlanCandidate {
        id: Arc::from(id),
        delta: EvidenceCatalogDelta::try_new(base, regimes).unwrap(),
        cost_units: cost,
        sample_budget: 0,
        requires: Arc::from([]),
        conflicts: Arc::from([]),
    }
}

fn proposed(id: u32, population: &str, on: &[(u32, bool)]) -> EvidenceRegime {
    regime(id, EvidenceKind::Proposed, population, on)
}

/// a's do(Z2) at both levels, b's do(Z1 = 0), and an observational b study
/// that repairs nothing.
fn fig_1_candidates(base: &EvidenceCatalog) -> Vec<StudyPlanCandidate> {
    vec![
        candidate(
            base,
            "a_do_z2",
            3,
            vec![proposed(1, "a", &[(Z2, false)]), proposed(2, "a", &[(Z2, true)])],
        ),
        candidate(base, "b_do_z1", 2, vec![proposed(3, "b", &[(Z1, false)])]),
        candidate(base, "b_observe", 1, vec![proposed(4, "b", &[])]),
    ]
}

fn plan(
    route: &StudyPlanRoute,
    base: &EvidenceCatalog,
    candidates: &[StudyPlanCandidate],
    limits: StudyPlanLimits,
    ctx: &ExecutionContext,
) -> Result<StudyPlan, StudyPlanRefusal> {
    plan_study_additions(&fig_1_graph(), route, base, candidates, limits, ctx)
}

fn default_plan() -> StudyPlan {
    let ctx = ExecutionContext::for_tests(1);
    let base = base();
    plan(&mz(fig_1_sources()), &base, &fig_1_candidates(&base), StudyPlanLimits::default(), &ctx)
        .unwrap()
}

fn statuses(plan: &StudyPlan) -> Vec<(String, &'static str)> {
    plan.subsets.iter().map(|s| (s.label(), s.outcome.status())).collect()
}

#[test]
fn complementary_sources_are_sufficient_only_as_a_pair() {
    let plan = default_plan();
    // The frozen failure is the X1 decision on the target law alone.
    assert_eq!(plan.failure.code, "transport_missing_evidence");
    assert_eq!(plan.failure.detail, "mz_transport.missing_joint_regime");
    assert!(!plan.failure.facts.is_empty());
    assert_eq!(
        statuses(&plan),
        [
            ("subset:[b_observe]".to_owned(), "insufficient"),
            ("subset:[b_do_z1]".to_owned(), "insufficient"),
            ("subset:[a_do_z2]".to_owned(), "insufficient"),
            ("subset:[b_do_z1,b_observe]".to_owned(), "insufficient"),
            ("subset:[a_do_z2,b_observe]".to_owned(), "insufficient"),
            ("subset:[a_do_z2,b_do_z1]".to_owned(), "sufficient"),
            ("subset:[a_do_z2,b_do_z1,b_observe]".to_owned(), "dominated"),
        ]
    );
    let top = &plan.proposals[0];
    assert_eq!(plan.proposals.len(), 1);
    assert_eq!(top.candidates, [Arc::from("a_do_z2"), Arc::from("b_do_z1")]);
    assert_eq!(top.cost_units, 5);
    assert_eq!(top.stage, "mz_transport:combined");
    assert!(top.uncited_candidates.is_empty());
    // Both sources' experiments are cited, each factor by its own regime.
    assert_eq!(top.cited_regimes.iter().map(|r| r.raw()).collect::<Vec<_>>(), [1, 2, 3]);
    let repaired =
        top.repairs.iter().map(|r| (r.candidate.as_ref(), r.regime.raw())).collect::<Vec<_>>();
    assert_eq!(repaired, [("a_do_z2", 1), ("a_do_z2", 2), ("b_do_z1", 3)]);
    for repair in &top.repairs {
        assert!(!repair.factors.is_empty(), "every cited proposed regime names its factor");
        assert!(repair.required_margin.iter().all(|x| repair.declared_margin.contains(x)));
        assert!(!repair.required_margin.is_empty());
    }
    let StudyProposalDerivation::Mz(record) = &top.derivation else {
        panic!("an mz proposal carries the X1 derivation record");
    };
    assert_eq!(record.route, "combined");
    assert_eq!(record.populations, ["a", "b"]);
    // Every strictly cheaper subset was decided: the pair is certified cost-minimal.
    assert!(plan.minimal);
    assert!(plan.stop.is_none());
    assert_eq!(plan.status(), None);
    assert!(top.decision_operations > 0 && top.decision_operations <= 4096);
}

#[test]
fn a_replayed_obstruction_is_theorem_limited() {
    // R-443 Fig. 2(a,b) analogue: Z -> X -> Y with X <-> Y; a controls X (S_a
    // into Y), b controls Z (S_b into X). The obstruction concerns the declared
    // controllable sets, so no study inside them is evaluated.
    let mut g = Admg::with_variables(3);
    for (a, b) in [(0, 1), (1, 2)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    g.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let assigned = |mut s: ZTransportSourceSpec, x: u32| {
        s.experiment_assignment =
            Arc::from([InterventionAssignment { variable: v(x), value: Value::f64(0.0) }]);
        s
    };
    let route = StudyPlanRoute::Mz(MzTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(1)]),
        target: Arc::from("target"),
        sources: vec![assigned(source("a", &[1], &[2]), 1), assigned(source("b", &[0], &[1]), 0)]
            .into(),
    });
    let observed = |id, population: &str, on: &[u32]| {
        EvidenceRegime::try_new(
            RegimeId::from_raw(id),
            if on.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
            EvidenceKind::Available,
            on.iter().copied().map(v).collect::<Vec<_>>(),
            [],
            (0..3).filter(|i| !on.contains(i)).map(v).collect::<Vec<_>>(),
            population,
            DistributionAvailability::Joint,
        )
        .unwrap()
    };
    let base = catalog(vec![observed(0, "target", &[])]);
    let mut study = observed(1, "a", &[1]);
    study.evidence_kind = EvidenceKind::Proposed;
    let candidates = vec![candidate(&base, "a_do_x", 1, vec![study])];
    let ctx = ExecutionContext::for_tests(1);
    let refusal =
        plan_study_additions(&g, &route, &base, &candidates, StudyPlanLimits::default(), &ctx)
            .unwrap_err();
    assert_eq!(refusal.code, "transport_proven_non_transportable");
    assert_eq!(refusal.detail, "study_plan.theorem_limited");
    assert!(refusal.receipt.is_none());
}

#[test]
fn no_sufficient_subset_is_none_certified_never_impossible() {
    let base = base();
    // Observational source studies and a single-level a trial cannot repair it.
    let candidates = vec![
        candidate(&base, "a_observe", 1, vec![proposed(1, "a", &[])]),
        candidate(&base, "b_observe", 1, vec![proposed(2, "b", &[])]),
        candidate(&base, "a_do_z2_low", 2, vec![proposed(3, "a", &[(Z2, false)])]),
    ];
    let ctx = ExecutionContext::for_tests(1);
    let plan =
        plan(&mz(fig_1_sources()), &base, &candidates, StudyPlanLimits::default(), &ctx).unwrap();
    assert!(plan.proposals.is_empty());
    assert_eq!(plan.status(), Some(("transport_not_certified", "study_plan.none_certified")));
    assert_eq!(plan.subsets.len(), 7);
    assert!(plan.subsets.iter().all(|s| s.outcome.status() == "insufficient"));
    assert!(plan.stop.is_none() && !plan.minimal);
}

/// Operations the full Figure 1 plan charges.
fn full_operations() -> usize {
    default_plan().operations_consumed
}

#[test]
fn a_stopped_plan_keeps_verified_proposals_and_names_unevaluated_subsets() {
    let full = full_operations();
    let base = base();
    let candidates = fig_1_candidates(&base);
    let ctx = ExecutionContext::for_tests(1);
    // One budget bounds everything: one operation short of the full plan stops
    // it, after the pair was verified (the dominated triple is never entered).
    let limits = |operations| StudyPlanLimits {
        search: SearchLimits { operations, depth: 24 },
        ..StudyPlanLimits::default()
    };
    let short = plan(&mz(fig_1_sources()), &base, &candidates, limits(full - 1), &ctx).unwrap();
    let Some(StudyPlanStop::Budget(receipt)) = &short.stop else {
        panic!("one operation short of the full plan must stop: {:?}", short.stop);
    };
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(receipt.operations_limit, full - 1);
    assert_eq!(receipt.operations_consumed, Some(full - 1));
    assert!(receipt.explored.first().is_some_and(|r| r == "stage:base"));
    assert!(!receipt.unevaluated.is_empty());
    assert!(short.subsets.iter().any(|s| s.outcome == StudySubsetOutcome::Unevaluated));
    // A stop before the pair keeps nothing and says so, never a verdict.
    let pair_start = {
        let mut probe = full - 1;
        loop {
            let p = plan(&mz(fig_1_sources()), &base, &candidates, limits(probe), &ctx).unwrap();
            if p.proposals.is_empty() {
                break p;
            }
            probe -= 1;
        }
    };
    assert_eq!(pair_start.status(), Some(("transport_budget_cancel", "study_plan.budget")));
    let receipt = pair_start.receipt().unwrap();
    assert!(receipt.unevaluated.contains(&"subset:[a_do_z2,b_do_z1]".to_owned()));
    assert!(!receipt.explored.contains(&"subset:[a_do_z2,b_do_z1]".to_owned()));
    // The explored and unevaluated regions are disjoint.
    assert!(receipt.explored.iter().all(|r| !receipt.unevaluated.contains(r)));
    // A budget that cannot finish the base decision refuses with its receipt.
    let refusal = plan(&mz(fig_1_sources()), &base, &candidates, limits(1), &ctx).unwrap_err();
    assert_eq!((refusal.code, refusal.detail), ("transport_budget_cancel", "study_plan.budget"));
    let receipt = refusal.receipt.unwrap();
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(receipt.unevaluated.first().map(String::as_str), Some("stage:base"));
    assert_eq!(receipt.unevaluated.len(), 1 + 7);
}

#[test]
fn a_truncated_plan_keeps_its_minimality_when_every_cheaper_subset_was_decided() {
    let base = base();
    let mut candidates = fig_1_candidates(&base);
    // An expensive, useless study after the pair: the plan must still decide it.
    candidates.push(candidate(&base, "a_observe", 10, vec![proposed(5, "a", &[])]));
    let ctx = ExecutionContext::for_tests(1);
    let full =
        plan(&mz(fig_1_sources()), &base, &candidates, StudyPlanLimits::default(), &ctx).unwrap();
    assert!(full.stop.is_none() && full.minimal);
    // One operation short: the stop falls after the pair, on a costlier subset,
    // so every subset cheaper than the pair was decided and the claim stands.
    let limits = StudyPlanLimits {
        search: SearchLimits { operations: full.operations_consumed - 1, depth: 24 },
        ..StudyPlanLimits::default()
    };
    let stopped = plan(&mz(fig_1_sources()), &base, &candidates, limits, &ctx).unwrap();
    let receipt = stopped.receipt().expect("one operation short stops the plan");
    assert_eq!(receipt.stop, SearchStop::Operations);
    let shape = |plan: &StudyPlan| {
        plan.proposals
            .iter()
            .map(|p| (p.candidates.clone(), p.cost_units, p.stage.clone(), p.repairs.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(shape(&stopped), shape(&full));
    assert!(stopped.minimal);
    let top = stopped.proposals[0].cost_units;
    assert!(stopped.subsets.iter().filter(|s| s.cost_units < top).all(|s| s.outcome.conclusive()));
    assert!(stopped.subsets.iter().any(|s| s.cost_units > top && !s.outcome.conclusive()));
}

struct CancelAfter {
    token: CancellationToken,
    reports: Mutex<usize>,
    after: usize,
}

impl ProgressSink for CancelAfter {
    fn report(&self, _fraction: f64, _stage: &str) {
        let mut reports = self.reports.lock().unwrap();
        *reports += 1;
        if *reports == self.after {
            self.token.cancel();
        }
    }
}

#[test]
fn depth_memory_and_cancellation_stop_the_one_plan_budget() {
    let base = base();
    let candidates = fig_1_candidates(&base);
    // Depth: a depth-1 budget cannot run the base decision.
    let ctx = ExecutionContext::for_tests(1);
    let shallow = StudyPlanLimits {
        search: SearchLimits { operations: 200_000, depth: 1 },
        ..StudyPlanLimits::default()
    };
    let refusal = plan(&mz(fig_1_sources()), &base, &candidates, shallow, &ctx).unwrap_err();
    assert_eq!(refusal.receipt.as_ref().map(|r| r.stop), Some(SearchStop::Depth));
    assert_eq!(refusal.receipt.as_ref().map(|r| r.depth_limit), Some(1));
    // Memory: the context's hard limit is the effective cap, observed per charge.
    let mut tight = ExecutionContext::for_tests(1);
    tight.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(64) };
    let refusal =
        plan(&mz(fig_1_sources()), &base, &candidates, StudyPlanLimits::default(), &tight)
            .unwrap_err();
    let receipt = refusal.receipt.unwrap();
    assert_eq!(receipt.stop, SearchStop::Memory);
    assert_eq!(receipt.memory_limit_bytes, Some(64));
    // Cancellation before entry: nothing is charged.
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    let refusal =
        plan(&mz(fig_1_sources()), &base, &candidates, StudyPlanLimits::default(), &cancelled)
            .unwrap_err();
    let receipt = refusal.receipt.unwrap();
    assert_eq!(receipt.stop, SearchStop::Cancelled);
    assert_eq!(receipt.operations_consumed, None);
    // Cancellation in the middle of the plan: the next charge observes it.
    let mut mid = ExecutionContext::for_tests(1);
    mid.progress = Some(Arc::new(CancelAfter {
        token: mid.cancellation.clone(),
        reports: Mutex::new(0),
        after: 2,
    }));
    let stopped =
        plan(&mz(fig_1_sources()), &base, &candidates, StudyPlanLimits::default(), &mid).unwrap();
    let receipt = stopped.receipt().expect("a mid-plan cancellation is a receipt");
    assert_eq!(receipt.stop, SearchStop::Cancelled);
    assert_eq!(stopped.subsets.iter().filter(|s| s.outcome.conclusive()).count(), 2);
    assert!(stopped.proposals.is_empty() && !stopped.minimal);
}

#[test]
fn one_budget_is_shared_by_the_base_and_every_subset_decision() {
    let plan = default_plan();
    // The plan's total is the base decision plus every entered subset (one entry
    // charge each) plus every re-run decision, on one budget.
    let decisions = plan.proposals.iter().map(|p| p.decision_operations).sum::<usize>();
    let entered = plan
        .subsets
        .iter()
        .filter(|s| s.outcome.conclusive() && s.outcome.status() != "dominated")
        .count();
    assert_eq!(entered, 6);
    assert!(plan.operations_consumed > decisions + entered);
    assert!(plan.operations_consumed <= STUDY_PLAN_DEFAULT_LIMITS.operations);
    assert_eq!(plan.limits.search, STUDY_PLAN_DEFAULT_LIMITS);
    // Exactly the plan's total completes it; one fewer stops the same budget.
    let base = base();
    let ctx = ExecutionContext::for_tests(1);
    let exact = |operations| StudyPlanLimits {
        search: SearchLimits { operations, depth: 24 },
        ..StudyPlanLimits::default()
    };
    let enough = crate::plan(
        &mz(fig_1_sources()),
        &base,
        &fig_1_candidates(&base),
        exact(plan.operations_consumed),
        &ctx,
    )
    .unwrap();
    assert!(enough.stop.is_none());
    assert_eq!(enough.operations_consumed, plan.operations_consumed);
    let short = crate::plan(
        &mz(fig_1_sources()),
        &base,
        &fig_1_candidates(&base),
        exact(plan.operations_consumed - 1),
        &ctx,
    )
    .unwrap();
    assert_eq!(short.receipt().map(|r| r.stop), Some(SearchStop::Operations));
}

#[test]
fn ties_rank_deterministically_and_declaration_order_is_irrelevant() {
    let base = base();
    let mut candidates = fig_1_candidates(&base);
    // A second b trial at the same cost as b_do_z1: both pairs are sufficient,
    // equal in cost; the tie breaks on the sorted ids.
    candidates.push(candidate(&base, "b_do_z1_again", 2, vec![proposed(5, "b", &[(Z1, false)])]));
    let ctx = ExecutionContext::for_tests(1);
    let forward =
        plan(&mz(fig_1_sources()), &base, &candidates, StudyPlanLimits::default(), &ctx).unwrap();
    let ranked = forward.proposals.iter().map(|p| p.candidates.clone()).collect::<Vec<_>>();
    assert_eq!(
        ranked,
        [
            vec![Arc::from("a_do_z2"), Arc::from("b_do_z1")],
            vec![Arc::from("a_do_z2"), Arc::from("b_do_z1_again")],
        ]
    );
    assert_eq!(forward.proposals[0].cost_units, forward.proposals[1].cost_units);
    let mut reversed = candidates.clone();
    reversed.reverse();
    let mut sources = fig_1_sources();
    sources.reverse();
    let backward = plan(&mz(sources), &base, &reversed, StudyPlanLimits::default(), &ctx).unwrap();
    assert_eq!(forward, backward);
}

#[test]
fn bounds_and_impossible_designs_refuse_by_detail() {
    let base = base();
    let ctx = ExecutionContext::for_tests(1);
    let route = mz(fig_1_sources());
    let refuse = |candidates: Vec<StudyPlanCandidate>, limits: StudyPlanLimits| {
        let refusal = plan(&route, &base, &candidates, limits, &ctx).unwrap_err();
        (refusal.code, refusal.detail)
    };
    let invalid = ("invalid_argument", "study_plan.invalid_candidate");
    let bound = ("route_not_supported", "study_plan.bounds_exceeded");
    // Seventeen candidates exceed the universe bound; sixteen do not.
    let many = |n: u32| {
        (0..n)
            .map(|i| candidate(&base, &format!("c{i:02}"), 1, vec![proposed(10 + i, "b", &[])]))
            .collect::<Vec<_>>()
    };
    assert_eq!(refuse(many(17), StudyPlanLimits::default()), bound);
    assert!(plan(&route, &base, &many(16), StudyPlanLimits::default(), &ctx).is_ok());
    // Nine regimes in one candidate exceed the per-candidate bound; eight do not.
    let levels = |n: u32| (0..n).map(|i| proposed(20 + i, "b", &[])).collect::<Vec<_>>();
    assert_eq!(
        refuse(vec![candidate(&base, "wide", 1, levels(9))], StudyPlanLimits::default()),
        bound
    );
    assert!(
        plan(
            &route,
            &base,
            &[candidate(&base, "wide", 1, levels(8))],
            StudyPlanLimits::default(),
            &ctx
        )
        .is_ok()
    );
    // Plan limits above the declared caps.
    let over = |operations, depth| StudyPlanLimits {
        search: SearchLimits { operations, depth },
        ..StudyPlanLimits::default()
    };
    let one = || vec![candidate(&base, "b_do_z1", 1, vec![proposed(3, "b", &[(Z1, false)])])];
    assert_eq!(refuse(one(), over(200_001, 24)), bound);
    assert_eq!(refuse(one(), over(200_000, 25)), bound);
    // Impossible designs.
    let target_trial = candidate(&base, "t", 1, vec![proposed(3, "target", &[(X, true)])]);
    assert_eq!(refuse(vec![target_trial], StudyPlanLimits::default()), invalid);
    let outside = candidate(&base, "a_do_z1", 1, vec![proposed(3, "a", &[(Z1, false)])]);
    assert_eq!(refuse(vec![outside], StudyPlanLimits::default()), invalid);
    let elsewhere = StudyPlanCandidate {
        delta: EvidenceCatalogDelta { proposed_regimes: Arc::from([proposed(3, "c", &[])]) },
        ..candidate(&base, "c_obs", 1, vec![proposed(3, "b", &[])])
    };
    assert_eq!(refuse(vec![elsewhere], StudyPlanLimits::default()), invalid);
    let mut free = one();
    free[0].cost_units = 0;
    assert_eq!(refuse(free, StudyPlanLimits::default()), invalid);
    let mut duplicate = one();
    duplicate.push(duplicate[0].clone());
    assert_eq!(refuse(duplicate, StudyPlanLimits::default()), invalid);
    let mut bad_id = one();
    bad_id[0].id = Arc::from("b do z1");
    assert_eq!(refuse(bad_id, StudyPlanLimits::default()), invalid);
    let mut unknown = one();
    unknown[0].requires = Arc::from([Arc::from("nobody")]);
    assert_eq!(refuse(unknown, StudyPlanLimits::default()), invalid);
    let clash = vec![
        candidate(&base, "x1", 1, vec![proposed(3, "b", &[])]),
        candidate(&base, "x2", 1, vec![proposed(3, "a", &[])]),
    ];
    assert_eq!(refuse(clash, StudyPlanLimits::default()), invalid);
    // A value-restricted study on the mixed route, and a depth above its 16.
    let mixed = StudyPlanRoute::Mixed(MixedSourceQuery {
        outcomes: Arc::from([v(Y)]),
        treatments: Arc::from([v(X)]),
        target: Arc::from("target"),
        sources: Arc::from([]),
    });
    let restricted = candidate(&base, "trial", 1, vec![proposed(3, "target", &[(Z2, true)])]);
    let refusal =
        plan(&mixed, &base, &[restricted], StudyPlanLimits::for_route(&mixed), &ctx).unwrap_err();
    assert_eq!((refusal.code, refusal.detail), invalid);
    let family = {
        let mut r = proposed(3, "target", &[(Z2, true)]);
        r.intervention_values = Arc::from([]);
        candidate(&base, "trial", 1, vec![r])
    };
    let refusal = plan(&mixed, &base, &[family], over(200_000, 17), &ctx).unwrap_err();
    assert_eq!((refusal.code, refusal.detail), bound);
}

#[test]
fn a_base_that_already_identifies_has_no_failure_to_repair() {
    let complete = catalog(vec![
        regime(0, EvidenceKind::Available, "target", &[]),
        regime(1, EvidenceKind::Available, "a", &[(Z2, false)]),
        regime(2, EvidenceKind::Available, "a", &[(Z2, true)]),
        regime(3, EvidenceKind::Available, "b", &[(Z1, false)]),
    ]);
    let candidates = vec![candidate(&complete, "b_observe", 1, vec![proposed(4, "b", &[])])];
    let ctx = ExecutionContext::for_tests(1);
    let refusal =
        plan(&mz(fig_1_sources()), &complete, &candidates, StudyPlanLimits::default(), &ctx)
            .unwrap_err();
    assert_eq!(
        (refusal.code, refusal.detail),
        ("invalid_argument", "study_plan.no_failure_to_repair")
    );
}
