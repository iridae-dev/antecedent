//! Discovery, refutation, decision, refresh and program-search procedures at horizon one.
//!
//! These procedures read a response, a sample or a state, not a causal graph of their own. At
//! horizon one the temporal member runs the same numerical engine as its static twin on the same
//! quantities; what the temporal member adds (time-ordered orientation, a pre-trend refuter, a
//! transport premise, a mechanism standing across periods) is pinned as the difference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use super::*;
use antecedent::period_refresh::{PeriodRefresh, PeriodRefreshOutcome, PeriodRefreshSpec};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::CausalRng;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, ProgramObjective,
    ProgramSearchContract, ProgramSearchSpace, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{
    MeanEstimate, MeanSource, Verdict, evaluate_contract_on_means,
};
use antecedent_design::program_search::{
    FiniteProgramDomain, PreparedFiniteProgramSearch, ProgramCandidate, ProgramControl,
    ProgramFeasibility, ProgramSearchScope, ProgramSearchSource,
};
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, EvaluationLimits,
    FiniteSequentialModel, HistoryKey,
};
use antecedent_design::transported_policy::{
    MechanismStatus, ModelMechanism, SelectionDiagram, TransportedPolicyRequest,
    evaluate_transported_policy,
};
use antecedent_discovery::{
    DiscoveryConstraints, DiscoveryWorkspace, Pc, Pcmci, TemporalConstraints,
};
use antecedent_estimate::temporal_refresh::{
    MechanismStanding, ObservationCadence, ObservationPeriod, TemporalWindowIdentity,
};
use antecedent_kernels::{shuffle, standard_normal, unbiased_index};
use antecedent_state::{ConjugateOlsPrior, OlsRefresh, RefreshRow};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
use antecedent_validate::target_native::{
    RefutationTarget, TargetFamily, TargetPerturbation, TargetQuadrant, TargetRefit,
    TargetRefitError, TargetRefutationConfig, TargetRefutationPlan, TargetRefuterKind, TargetSuite,
    run_target_plan,
};

// ---------------------------------------------------------------------------
// Discovery: PC on aligned rows against PCMCI at one lag.
// ---------------------------------------------------------------------------

const DISCOVERY_ROWS: usize = 600;

struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// outcome: equivalent adjacency; `not_equivalent` orientation (time order is an assumption).
///
/// `x_{t-1} -> y_t` with `x` serially independent. PCMCI restricted to lag one finds exactly that
/// link, oriented by time. Static PC on the lag-aligned pairs `(x_{t-1}, y_t)` finds the same
/// adjacency but a single undirected edge: two variables with no collider cannot be oriented by
/// independence tests, and rows carry no clock. The orientation of the temporal graph is the
/// time-order background knowledge, not something the statistics recover.
#[test]
fn horizon_one_discovery_finds_the_static_adjacency_and_adds_only_the_time_orientation() {
    let mut rng = Lcg(77);
    let x: Vec<f64> = (0..DISCOVERY_ROWS).map(|_| rng.normal()).collect();
    let mut y = vec![0.0; DISCOVERY_ROWS];
    for t in 1..DISCOVERY_ROWS {
        y[t] = 0.8 * x[t - 1] + 0.5 * rng.normal();
    }
    let variables = [id(0), id(1)];
    let constraints = |temporal: TemporalConstraints| DiscoveryConstraints {
        temporal,
        alpha: 0.001,
        max_cond_size: 2,
        ..DiscoveryConstraints::default()
    };

    let temporal = Pcmci::new()
        .with_fdr(false)
        .with_constraints(constraints(TemporalConstraints {
            max_lag: Lag::from_raw(1),
            min_lag: Lag::from_raw(1),
        }))
        .run(
            &series(&[("x", &x), ("y", &y)]),
            &variables,
            &mut DiscoveryWorkspace::default(),
            &ctx(),
        )
        .unwrap();
    let links: BTreeSet<(u32, u32, u32, u32)> = temporal
        .evidence
        .links
        .iter()
        .map(|scored| {
            (
                scored.link.source.raw(),
                scored.link.source_lag.raw(),
                scored.link.target.raw(),
                scored.link.target_lag.raw(),
            )
        })
        .collect();
    assert_eq!(links, BTreeSet::from([(0, 1, 1, 0)]), "x at lag one causes y");

    let mut static_constraints = Pc::new().constraints;
    static_constraints.alpha = 0.001;
    static_constraints.max_cond_size = 2;
    let static_result = Pc::new()
        .with_fdr(false)
        .with_constraints(static_constraints)
        .run(
            &table(&[("x_lag", &x[..DISCOVERY_ROWS - 1]), ("y", &y[1..])]),
            &variables,
            &mut DiscoveryWorkspace::default(),
            &ctx(),
        )
        .unwrap();
    let edges = static_result.evidence.graph.edges();
    assert_eq!(edges.len(), 1, "the same adjacency: one edge between the two variables");
    assert!(edges[0].is_undirected(), "static PC cannot orient an unshielded pair");
}

// ---------------------------------------------------------------------------
// Refutation: the same target-native engine in the static and the temporal quadrant.
// ---------------------------------------------------------------------------

const LEVELS: [f64; 3] = [0.0, 1.0, 2.0];

struct Simulated {
    t: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
}

fn simulate(n: usize, seed: u64) -> Simulated {
    let mut rng = CausalRng::from_seed(seed);
    let mut sim = Simulated { t: Vec::new(), x: Vec::new(), y: Vec::new() };
    for _ in 0..n {
        let x = standard_normal(&mut rng);
        let t = if x + 0.5 * standard_normal(&mut rng) > 0.5 {
            2.0
        } else if x > -0.5 {
            1.0
        } else {
            0.0
        };
        let y = 1.0 + 0.5 * t + x + 0.5 * standard_normal(&mut rng);
        sim.t.push(t);
        sim.x.push(x);
        sim.y.push(y);
    }
    sim
}

/// World values `E[Y | do(T = level)]` from the regression `Y ~ 1 + T + X (+ U)`.
fn world_values(t: &[f64], x: &[f64], y: &[f64], extra: Option<&[f64]>) -> Vec<f64> {
    let n = y.len();
    let mut design = vec![1.0; n];
    design.extend_from_slice(t);
    design.extend_from_slice(x);
    let mut p = 3;
    if let Some(u) = extra {
        design.extend_from_slice(u);
        p += 1;
    }
    let fit =
        FaerBackend.least_squares(&design, n, p, y, &mut LeastSquaresWorkspace::default()).unwrap();
    let b = &fit.coefficients;
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_u = extra.map_or(0.0, |u| u.iter().sum::<f64>() / n as f64);
    LEVELS
        .iter()
        .map(|level| b[0] + b[1] * level + b[2] * mean_x + if p == 4 { b[3] * mean_u } else { 0.0 })
        .collect()
}

/// One regression procedure, declared in either quadrant: the refits are the same function.
struct RegressionRefit {
    sim: Simulated,
    target: RefutationTarget,
    original: Vec<f64>,
}

impl RegressionRefit {
    fn new(sim: Simulated, quadrant: TargetQuadrant) -> Self {
        let target = RefutationTarget::new(
            TargetFamily::WorldValued,
            quadrant,
            ["do(T=0)", "do(T=1)", "do(T=2)"],
            [3; 32],
        )
        .unwrap();
        let original = world_values(&sim.t, &sim.x, &sim.y, None);
        Self { sim, target, original }
    }

    fn rows(&self, rows: &[usize]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            rows.iter().map(|&i| self.sim.t[i]).collect(),
            rows.iter().map(|&i| self.sim.x[i]).collect(),
            rows.iter().map(|&i| self.sim.y[i]).collect(),
        )
    }
}

impl TargetRefit for RegressionRefit {
    fn target(&self) -> &RefutationTarget {
        &self.target
    }

    fn original(&self) -> &[f64] {
        &self.original
    }

    fn refit(
        &self,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        _ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let n = self.sim.y.len();
        let s = &self.sim;
        Ok(match perturbation {
            TargetPerturbation::PlaceboTreatment { .. } => {
                let mut t = s.t.clone();
                shuffle(rng, &mut t);
                world_values(&t, &s.x, &s.y, None)
            }
            TargetPerturbation::PlaceboOutcome { .. } => {
                let mut y = s.y.clone();
                shuffle(rng, &mut y);
                world_values(&s.t, &s.x, &y, None)
            }
            TargetPerturbation::RandomCommonCause { .. } => {
                let u: Vec<f64> = (0..n).map(|_| standard_normal(rng)).collect();
                world_values(&s.t, &s.x, &s.y, Some(&u))
            }
            TargetPerturbation::Subset { fraction, .. } => {
                let mut rows: Vec<usize> = (0..n).collect();
                shuffle(rng, &mut rows);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a subset fraction in (0, 1) keeps between zero and n rows"
                )]
                let keep = (fraction * n as f64) as usize;
                rows.truncate(keep);
                let (t, x, y) = self.rows(&rows);
                world_values(&t, &x, &y, None)
            }
            TargetPerturbation::Bootstrap { .. } => {
                let rows: Vec<usize> = (0..n).map(|_| unbiased_index(rng, n)).collect();
                let (t, x, y) = self.rows(&rows);
                world_values(&t, &x, &y, None)
            }
            _ => {
                return Err(TargetRefitError::Unsupported {
                    reason: Arc::from("no construction is declared for this refuter"),
                    next_attempt: Arc::from("declare the control or the pre-period it needs"),
                });
            }
        })
    }
}

/// outcome: equivalent reports for every refuter the two quadrants share; the temporal catalogue
/// adds refuters a one-decision sample cannot construct.
///
/// The refutation engine draws its random streams from the context seed, the refuter and the
/// replicate only, never from the quadrant, so the same procedure refuted in the static and the
/// temporal quadrant gives identical original and refuted vectors, joint statistics and verdicts
/// for the placebo, common-cause, subset and bootstrap refuters. The temporal quadrant also owes a
/// pre-trend refuter, which needs a lead of the treatment that one decision does not have: the
/// route records it as refused with the next attempt rather than passing it.
#[test]
fn horizon_one_refutation_gives_identical_reports_in_both_quadrants_and_refuses_the_pre_trend() {
    let configuration =
        TargetRefutationConfig { replicates: 30, ..TargetRefutationConfig::default() };
    let run_in = |quadrant: TargetQuadrant| {
        let refit = RegressionRefit::new(simulate(300, 41), quadrant);
        let plan = TargetRefutationPlan::for_target(refit.target().clone(), TargetSuite::Full);
        run_target_plan(&plan, &refit, &configuration, &ExecutionContext::for_tests(9)).unwrap()
    };
    let static_run = run_in(TargetQuadrant::STATIC_FREQUENTIST);
    let temporal_run = run_in(TargetQuadrant::TEMPORAL_FREQUENTIST);

    let shared = [
        TargetRefuterKind::PlaceboTreatment,
        TargetRefuterKind::PlaceboOutcome,
        TargetRefuterKind::RandomCommonCause,
        TargetRefuterKind::SubsetStability,
        TargetRefuterKind::BootstrapStability,
    ];
    for kind in shared {
        let left = static_run.reports.iter().find(|report| report.kind == kind).unwrap();
        let right = temporal_run.reports.iter().find(|report| report.kind == kind).unwrap();
        assert_eq!(left.original, right.original, "{kind:?}");
        assert_eq!(left.refuted, right.refuted, "{kind:?}");
        assert_eq!(left.statistic, right.statistic, "{kind:?}");
        assert_eq!(
            (left.passed, left.informative, left.replicates),
            (right.passed, right.informative, right.replicates)
        );
        assert_ne!(
            left.target.identity(),
            right.target.identity(),
            "the quadrant is part of the target"
        );
    }
    // Independent of the equivalence: the generating world is E[Y | do(T = l)] = 1 + 0.5 l
    // (E[X] = 0), the original estimate recovers it (observed error 0.015), every shared
    // refuter passes, and the placebo-treatment world is flat in the level while the real one
    // rises by 1.0 (observed spread 0.02).
    for run in [&static_run, &temporal_run] {
        for kind in shared {
            let report = run.reports.iter().find(|report| report.kind == kind).unwrap();
            assert!(report.passed && report.informative, "{kind:?}");
            for (value, level) in report.original.iter().zip(LEVELS) {
                assert!((value - (1.0 + 0.5 * level)).abs() < 0.05, "{kind:?}: {value} at {level}");
            }
        }
        let placebo = run
            .reports
            .iter()
            .find(|report| report.kind == TargetRefuterKind::PlaceboTreatment)
            .unwrap();
        let spread = placebo.refuted[2] - placebo.refuted[0];
        assert!(spread.abs() < 0.1, "a shuffled treatment has no effect: spread {spread}");
    }
    assert!(static_run.reports.iter().all(|report| report.kind != TargetRefuterKind::PreTrend));
    let refused: Vec<TargetRefuterKind> = temporal_run.refused.iter().map(|r| r.kind).collect();
    assert!(refused.contains(&TargetRefuterKind::PreTrend), "{refused:?}");
    assert!(temporal_run.refused.iter().all(|r| !r.next_attempt.is_empty()));
    assert!(static_run.refused.iter().all(|r| r.kind != TargetRefuterKind::PreTrend));
}

// ---------------------------------------------------------------------------
// Decision value: a one-step transported policy against the static expected-utility decision.
// ---------------------------------------------------------------------------

fn commit(label: &str, immediate: [f64; 2]) -> ActionSpec {
    ActionSpec {
        label: label.into(),
        role: ActionRole::Commit,
        utility_cost: 0.0,
        resource_use: 0.0,
        transition: vec![1.0, 0.0, 0.0, 1.0],
        channel: None,
        immediate_utility: immediate.to_vec(),
        delayed: None,
    }
}

/// One decision: ship (`+10` sound, `-10` faulty) or hold (zero) a unit that is faulty with
/// probability `1 - initial[0]`.
fn one_decision_model(initial: [f64; 2]) -> FiniteSequentialModel {
    FiniteSequentialModel {
        label: "inspection".into(),
        population: "plant_a".into(),
        regime: "operations".into(),
        period: "2026".into(),
        state_variable: "unit_condition".into(),
        states: vec!["sound".into(), "faulty".into()],
        initial: initial.to_vec(),
        horizon: 1,
        actions: vec![commit("ship", [10.0, -10.0]), commit("hold", [0.0, 0.0])],
        terminal_utility: vec![0.0, 0.0],
        censoring: CensoringRule::Refuse,
        resource_budget: None,
        stopping: Vec::new(),
        support: ActionSupport::everywhere(2),
    }
}

fn faulty_share(regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "faulty".into(),
        variable_name: "faulty".into(),
        role: QuantityRole::Outcome,
        units: "share".into(),
        population_id: "plant_b".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

/// outcome: equivalent action values and choice; `not_equivalent` evidence requirements.
///
/// With one commit decision the transported policy value is the expected immediate utility under
/// the target initial law, `10 - 20 p` for shipping and zero for holding, which is the static
/// expected utility of the same two actions under the same mean state share. The temporal route
/// will not produce that number without the selection diagram that licenses carrying the source
/// model to the target population; the static mean-source decision takes the share as given.
#[test]
fn horizon_one_transported_policy_value_is_the_static_expected_utility_but_needs_a_selection_diagram()
 {
    let source = one_decision_model([0.5, 0.5]);
    let diagram = SelectionDiagram {
        source_population: "plant_a".into(),
        target_population: "plant_b".into(),
        mechanisms: vec![
            (ModelMechanism::InitialState, MechanismStatus::Shifted),
            (ModelMechanism::Utility, MechanismStatus::Invariant),
            (ModelMechanism::Transition { action: 0 }, MechanismStatus::Invariant),
            (ModelMechanism::Transition { action: 1 }, MechanismStatus::Invariant),
        ],
    };
    let ship = ContingentPolicy::new("ship").decide(HistoryKey::root(), 0);
    let hold = ContingentPolicy::new("hold").decide(HistoryKey::root(), 1);
    let static_contract = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "ship".into(),
                kind: ActionKind::Regime,
                inputs: vec![faulty_share("ship")],
                utility: UtilityExpr::difference(
                    UtilityExpr::Const(10.0),
                    UtilityExpr::product(UtilityExpr::Const(20.0), UtilityExpr::Input(0)),
                ),
            },
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::Regime,
                inputs: vec![faulty_share("hold")],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
            },
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "plant_b".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    };

    for faulty in [0.1, 0.3, 0.7, 0.9] {
        let law = [1.0 - faulty, faulty];
        let value_of = |policy: &ContingentPolicy| {
            let request = TransportedPolicyRequest {
                source: &source,
                policy,
                target_population: "plant_b",
                target_initial: Some(&law),
                diagram: Some(&diagram),
                target_utility: None,
                limits: EvaluationLimits::default(),
            };
            evaluate_transported_policy(&request, &ctx()).unwrap().target_value
        };
        let (shipped, held) = (value_of(&ship), value_of(&hold));
        assert!((shipped - (10.0 - 20.0 * faulty)).abs() < 1e-12, "{shipped}");
        assert!(held.abs() < 1e-12);

        let means = MeanSource {
            coordinates: vec![faulty_share("ship"), faulty_share("hold")],
            means: vec![faulty, faulty],
            estimate: MeanEstimate::Frequentist,
            provider_id: "inspection".into(),
            snapshot_id: "plant_b".into(),
            causal_contract_id: "state_share".into(),
            rng_id: "none".into(),
        };
        let decision = evaluate_contract_on_means(&static_contract, &means).unwrap();
        assert!((decision.actions[0].expected_utility - shipped).abs() < 1e-12);
        assert!((decision.actions[1].expected_utility - held).abs() < 1e-12);
        let best = if shipped > held { "ship" } else { "hold" };
        assert_eq!(decision.verdict, Verdict::UniquelyOptimal(best.into()));
    }

    // The temporal route refuses to move the model to another population without the diagram.
    let law = [0.8, 0.2];
    let request = TransportedPolicyRequest {
        source: &source,
        policy: &ship,
        target_population: "plant_b",
        target_initial: Some(&law),
        diagram: None,
        target_utility: None,
        limits: EvaluationLimits::default(),
    };
    let refusal = evaluate_transported_policy(&request, &ctx()).unwrap_err();
    assert_eq!(refusal.detail, "transported_policy.selection_diagram_missing");
}

// ---------------------------------------------------------------------------
// Model refresh: an autoregressive mechanism of one lag against the static least squares.
// ---------------------------------------------------------------------------

const REFRESH_ROWS: usize = 400;
const REFRESH_SPLIT: usize = 200;

fn refresh_window(start: i64, end: i64, snapshot: &str) -> TemporalWindowIdentity {
    TemporalWindowIdentity {
        graph_id: "ar_mechanism".into(),
        horizon: 1,
        lag_alignment: vec![("t".into(), 1)],
        intervention_history: vec!["a1=1".into()],
        selection_targets: BTreeSet::from(["s_t1".to_owned()]),
        regimes: BTreeSet::from(["observational".to_owned()]),
        period: ObservationPeriod { start, end },
        unit_ids: BTreeSet::from(["u1".to_owned()]),
        snapshot_id: snapshot.into(),
        proof_id: "proof_1".into(),
    }
}

fn pair_rows(values: &[f64], targets: std::ops::Range<usize>) -> Vec<RefreshRow> {
    targets.map(|i| RefreshRow::new(i as u64, vec![1.0, values[i - 1]], values[i])).collect()
}

fn assert_same_fit(left: &OlsRefresh, right: &OlsRefresh, label: &str) {
    let (a, b) = (left.snapshot(), right.snapshot());
    assert_eq!(a.n, b.n, "{label}");
    let (beta_a, beta_b) = (a.beta.as_ref().unwrap(), b.beta.as_ref().unwrap());
    for (x, y) in beta_a.iter().zip(beta_b.iter()) {
        assert!((x - y).abs() < 1e-9, "{label}: {x} vs {y}");
    }
    assert!((a.residual_variance.unwrap() - b.residual_variance.unwrap()).abs() < 1e-9, "{label}");
    assert!((a.yty - b.yty).abs() < 1e-8 * (1.0 + a.yty.abs()), "{label}");
}

/// outcome: equivalent summary; `not_equivalent` standing of the mechanism across periods.
///
/// An autoregressive mechanism of one lag is the least-squares summary of the pairs
/// `(v_{t-1}, v_t)`: the period refresh equals a static refresh fitted on those rows, and
/// extending it by a later period in place equals appending the same rows to the static summary,
/// with no rebuild. Only the temporal member carries a declaration that the mechanism is the same
/// in the next period: without it the old rows are dropped and the new period alone is fitted,
/// which differs from the pooled static summary.
#[test]
fn horizon_one_period_refresh_is_the_static_least_squares_refresh_with_a_standing_declaration() {
    let values: Vec<f64> = {
        let mut v = vec![0.0; REFRESH_ROWS];
        for t in 1..REFRESH_ROWS {
            v[t] = 0.6 * v[t - 1] + disturbance(t);
        }
        v
    };
    let prior = ConjugateOlsPrior::isotropic(2, 1.0, 4.0, 1.5).unwrap();
    let spec = |standing| PeriodRefreshSpec {
        series_id: "t",
        cadence: ObservationCadence::new(1).unwrap(),
        initial_law: "law_1",
        standing,
        lags: 1,
    };
    let first = refresh_window(0, REFRESH_SPLIT as i64, "snap_1");
    let second = refresh_window(REFRESH_SPLIT as i64, REFRESH_ROWS as i64, "snap_2");
    let cadence = || ObservationCadence::new(1).unwrap();

    let mut temporal = PeriodRefresh::start(
        &spec(MechanismStanding::DeclaredInvariant),
        &first,
        &prior,
        &values[..REFRESH_SPLIT],
    )
    .unwrap();
    let mut static_fit =
        OlsRefresh::from_rows("series", 2, pair_rows(&values, 1..REFRESH_SPLIT)).unwrap();
    assert_same_fit(temporal.fit(), &static_fit, "first period");
    // Independent of both: the simple-regression normal equations on the pairs (v_{t-1}, v_t).
    let simple_ols = |targets: std::ops::Range<usize>| -> [f64; 2] {
        let n = targets.len() as f64;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for i in targets {
            let (x, y) = (values[i - 1], values[i]);
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        [(sy - slope * sx) / n, slope]
    };
    let check_beta = |fit: &OlsRefresh, want: [f64; 2], label: &str| {
        let beta = fit.snapshot().beta.as_ref().unwrap().to_vec();
        for (got, want) in beta.iter().zip(want) {
            assert!((got - want).abs() < 1e-9, "{label}: {got} vs {want}");
        }
    };
    check_beta(temporal.fit(), simple_ols(1..REFRESH_SPLIT), "first period closed form");

    let outcome = temporal.refresh(&second, cadence(), "law_1", &values[REFRESH_SPLIT..]).unwrap();
    assert_eq!(outcome, PeriodRefreshOutcome::Extended { contiguous: true });
    static_fit.append_window("series", &pair_rows(&values, REFRESH_SPLIT..REFRESH_ROWS)).unwrap();
    assert_same_fit(temporal.fit(), &static_fit, "extended");
    check_beta(temporal.fit(), simple_ols(1..REFRESH_ROWS), "extended closed form");
    assert_eq!(temporal.fit().recomputes(), 0);
    assert_eq!(static_fit.recomputes(), 0);
    temporal.check_against_fresh().unwrap();

    // Without the declaration the temporal member restarts on the new period alone.
    let mut undeclared = PeriodRefresh::start(
        &spec(MechanismStanding::Undeclared),
        &first,
        &prior,
        &values[..REFRESH_SPLIT],
    )
    .unwrap();
    let restarted =
        undeclared.refresh(&second, cadence(), "law_1", &values[REFRESH_SPLIT..]).unwrap();
    assert!(matches!(restarted, PeriodRefreshOutcome::Restarted { .. }));
    let new_period_alone =
        OlsRefresh::from_rows("series", 2, pair_rows(&values, REFRESH_SPLIT + 1..REFRESH_ROWS))
            .unwrap();
    assert_same_fit(undeclared.fit(), &new_period_alone, "restarted");
    check_beta(
        undeclared.fit(),
        simple_ols(REFRESH_SPLIT + 1..REFRESH_ROWS),
        "restarted closed form",
    );
    let pooled = static_fit.snapshot();
    let restarted_fit = undeclared.fit().snapshot();
    assert_ne!(pooled.n, restarted_fit.n, "the restart forgets the first period");
}

// ---------------------------------------------------------------------------
// Program search: the same finite search over a one-step response and a static one.
// ---------------------------------------------------------------------------

fn program_quantity(regime: &str, horizon: u32) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "USD".into(),
        population_id: "observed".into(),
        regime_id: regime.into(),
        horizon,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn finite_search(horizon: u32) -> (PreparedFiniteProgramSearch, MeanSource) {
    let contract = ProgramSearchContract {
        space: ProgramSearchSpace { id: "one channel".into(), variables: vec!["response".into()] },
        objective: ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)),
        constraints: vec![],
    };
    let domain = FiniteProgramDomain {
        controls: vec![ProgramControl {
            variable: id(0),
            offset: 0,
            lower: 0.0,
            upper: 4.0,
            unit_cost: 1.0,
        }],
        budget: Some(3.0),
        restrictions: vec![],
    };
    let candidates: Vec<ProgramCandidate> = [("reference", 0.0), ("low", 2.0), ("high", 4.0)]
        .into_iter()
        .map(|(name, level)| ProgramCandidate {
            id: name.into(),
            controls: vec![level],
            inputs: vec![program_quantity(name, horizon)],
        })
        .collect();
    let source = MeanSource {
        coordinates: candidates.iter().map(|c| c.inputs[0].clone()).collect(),
        means: vec![0.0, 3.0, 8.0],
        estimate: MeanEstimate::Frequentist,
        provider_id: "compiled-model".into(),
        snapshot_id: "source".into(),
        causal_contract_id: "target".into(),
        rng_id: "no-sampling".into(),
    };
    let search = PreparedFiniteProgramSearch::compile(
        contract,
        domain,
        candidates,
        ProgramSearchScope {
            population: "observed".into(),
            horizon,
            utility_units: "USD".into(),
            max_candidates: 32,
        },
        &ctx(),
    )
    .unwrap();
    (search, source)
}

/// outcome: equivalent ranking, feasibility and best program.
///
/// The finite program search reads only the response quantities it is given and the operational
/// limits of the controls. Run over the same response means labelled as a static target (horizon
/// zero) and as a one-step temporal window (horizon one), it checks the same budget, ranks the
/// same programs, picks the same best and certifies the same finite set under the same
/// reliability statement; the horizon is part of each quantity's identity, not of the procedure.
#[test]
fn horizon_one_finite_program_search_ranks_exactly_as_the_static_search() {
    let (static_search, static_source) = finite_search(0);
    let (temporal_search, temporal_source) = finite_search(1);
    let static_result =
        static_search.execute(ProgramSearchSource::Means(&static_source), &ctx()).unwrap();
    let temporal_result =
        temporal_search.execute(ProgramSearchSource::Means(&temporal_source), &ctx()).unwrap();

    assert_eq!(static_result.rows, temporal_result.rows);
    // Hand values: the objective is the mean, the cost is the control level, and the budget of 3
    // excludes the control at 4 before it is valued.
    let hand: Vec<(&str, f64, Option<f64>)> =
        vec![("reference", 0.0, Some(0.0)), ("low", 2.0, Some(3.0)), ("high", 4.0, None)];
    for (row, (id, cost, value)) in static_result.rows.iter().zip(hand) {
        assert_eq!((row.id.as_str(), row.cost, row.value), (id, cost, value));
    }
    assert_eq!(static_result.best, vec!["low".to_owned()]);
    assert_eq!(temporal_result.best, static_result.best);
    assert_eq!(
        (static_result.exhaustive, static_result.exact_finite_law, static_result.reliability),
        (temporal_result.exhaustive, temporal_result.exact_finite_law, temporal_result.reliability)
    );
    assert!(matches!(
        &static_result.rows[2].feasibility,
        ProgramFeasibility::Operational(reason) if reason == "budget"
    ));
    assert_eq!(static_result.search_identity, temporal_result.search_identity);
    assert_eq!(static_result.continuous_gap, None);
}
