//! Sequential evaluation of a supplied policy against hand-computed values.
//!
//! The decision problem is a small inspection tree: a unit is `sound` or
//! `faulty` (prior one half each). Actions:
//!
//! | # | action  | role     | cost | notes |
//! |---|---------|----------|------|-------|
//! | 0 | wait    | continue | 0    | the unit may degrade: sound -> faulty w.p. 1/4 |
//! | 1 | inspect | continue | 1/4  | alarm w.p. 3/4 when faulty, 1/4 when sound; uses 1 resource |
//! | 2 | ship    | commit   | 0    | +10 if sound, -10 if faulty |
//! | 3 | repair  | commit   | 4    | outcome one decision time later: faulty is sound w.p. 1/2, then +10/-10 |
//! | 4 | pilot   | continue | 1/4  | an outcome [3, -1] two decision times later |
//!
//! Horizon three, terminal utility [2, -2], resource budget two, and two
//! declared stopping boundaries on `P(faulty)`: at least 0.85 forces `repair`,
//! at most 0.15 forces `ship`.
//!
//! Oracles (all written here, independent of the library):
//!
//! * hand values worked out in the comments of each test;
//! * a forward joint-probability enumeration (`oracle`) that never forms a
//!   conditional belief or a conditional expectation of a delayed outcome;
//! * a full logged-data generator (`log_under`) enumerating the latent state,
//!   the observations and the post-commit evolution under a logging policy.

#![allow(
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::type_complexity,
    reason = "the test oracles index two-state arrays by state and compare exact dyadic constants"
)]

use std::collections::BTreeMap;

use antecedent_contract::evidence_ledger::EvidenceLedger;
use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, DifferentialAxis, ErrorStructure, MeasurementBinding,
    MeasurementKind, MeasurementModel, MeasurementParameters, MeasurementRoute,
    MeasurementVariable, ParameterEvidence, RepresentationVersion,
};
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, BeliefTracker, CensoringRule, ComparisonKind,
    ComparisonSpec, ContingentPolicy, DelayedOutcome, EvaluationLimits, FiniteSequentialModel,
    HistoryKey, HistoryStep, LoggedStep, LoggedTrajectory, NodeKind, ObservationChannel,
    PolicySpec, RolloutConfig, SequentialRequest, StoppingBoundary, StreamRole, VoiRequest,
    compare_policies, evaluate_off_policy, evaluate_policy, request_interval, rollout_policy,
    stream_uniform, value_of_collecting_evidence,
};
use antecedent_design::sequential_evaluation_artifact::{
    ExpectedEvaluation, OffPolicyInput, SequentialEvaluationArtifact, consume,
};

const WAIT: usize = 0;
const INSPECT: usize = 1;
const SHIP: usize = 2;
const REPAIR: usize = 3;
const PILOT: usize = 4;
const NEGATIVE: usize = 0;
const ALARM: usize = 1;

type History = Vec<(usize, usize)>;
type Table = BTreeMap<History, Vec<(usize, f64)>>;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

// ---------------------------------------------------------------------------
// The problem, as plain arrays
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Act {
    name: &'static str,
    commit: bool,
    cost: f64,
    res: f64,
    trans: [[f64; 2]; 2],
    imm: [f64; 2],
    delayed: Option<(usize, [f64; 2])>,
    /// `lik[symbol][state]`.
    lik: Option<[[f64; 2]; 2]>,
}

#[derive(Clone)]
struct Boundary {
    states: Vec<usize>,
    upper: Option<f64>,
    lower: Option<f64>,
    stop: usize,
}

#[derive(Clone)]
struct Spec {
    init: [f64; 2],
    horizon: usize,
    acts: Vec<Act>,
    terminal: [f64; 2],
    budget: Option<f64>,
    boundaries: Vec<Boundary>,
    forfeit: bool,
    swapped: bool,
}

const IDENTITY: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];

fn base_spec(with_boundaries: bool) -> Spec {
    let acts = vec![
        Act {
            name: "wait",
            commit: false,
            cost: 0.0,
            res: 0.0,
            trans: [[0.75, 0.25], [0.0, 1.0]],
            imm: [0.0, 0.0],
            delayed: None,
            lik: None,
        },
        Act {
            name: "inspect",
            commit: false,
            cost: 0.25,
            res: 1.0,
            trans: IDENTITY,
            imm: [0.0, 0.0],
            delayed: None,
            lik: Some([[0.75, 0.25], [0.25, 0.75]]),
        },
        Act {
            name: "ship",
            commit: true,
            cost: 0.0,
            res: 0.0,
            trans: IDENTITY,
            imm: [10.0, -10.0],
            delayed: None,
            lik: None,
        },
        Act {
            name: "repair",
            commit: true,
            cost: 4.0,
            res: 0.0,
            trans: [[1.0, 0.0], [0.5, 0.5]],
            imm: [0.0, 0.0],
            delayed: Some((1, [10.0, -10.0])),
            lik: None,
        },
        Act {
            name: "pilot",
            commit: false,
            cost: 0.25,
            res: 0.0,
            trans: IDENTITY,
            imm: [0.0, 0.0],
            delayed: Some((2, [3.0, -1.0])),
            lik: None,
        },
    ];
    let boundaries = if with_boundaries {
        vec![
            Boundary { states: vec![1], upper: Some(0.85), lower: None, stop: REPAIR },
            Boundary { states: vec![1], upper: None, lower: Some(0.15), stop: SHIP },
        ]
    } else {
        Vec::new()
    };
    Spec {
        init: [0.5, 0.5],
        horizon: 3,
        acts,
        terminal: [2.0, -2.0],
        budget: Some(2.0),
        boundaries,
        forfeit: false,
        swapped: false,
    }
}

/// The same problem with the two latent states relabelled.
fn swap_states(spec: &Spec) -> Spec {
    let swap2 = |v: [f64; 2]| [v[1], v[0]];
    let swap_matrix = |m: [[f64; 2]; 2]| [[m[1][1], m[1][0]], [m[0][1], m[0][0]]];
    let mut out = spec.clone();
    out.init = swap2(spec.init);
    out.terminal = swap2(spec.terminal);
    for act in &mut out.acts {
        act.trans = swap_matrix(act.trans);
        act.imm = swap2(act.imm);
        act.delayed = act.delayed.map(|(d, u)| (d, swap2(u)));
        // rows are symbols (unchanged); columns are states (swapped)
        act.lik = act.lik.map(|l| [swap2(l[0]), swap2(l[1])]);
    }
    for boundary in &mut out.boundaries {
        boundary.states = boundary.states.iter().map(|s| 1 - s).collect();
    }
    out.swapped = !spec.swapped;
    out
}

fn state_names(swapped: bool) -> Vec<String> {
    if swapped {
        vec!["faulty".into(), "sound".into()]
    } else {
        vec!["sound".into(), "faulty".into()]
    }
}

fn measurement(
    route: MeasurementRoute,
    kind: MeasurementKind,
    parameters: MeasurementParameters,
) -> MeasurementModel {
    MeasurementModel {
        model_id: "inspection_alarm".into(),
        route,
        indicator: MeasurementVariable { id: "alarm".into(), kind, units: "category".into() },
        construct: MeasurementVariable {
            id: "unit_condition".into(),
            kind: MeasurementKind::LatentScientific,
            units: "category".into(),
        },
        binding: MeasurementBinding {
            population_id: "plant_a".into(),
            regime_id: "operations".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters,
        representation: RepresentationVersion {
            coordinate_system_id: "alarm_panel".into(),
            version: 1,
        },
        evidence: ParameterEvidence::Declared { declarer: "reliability_engineering".into() },
    }
}

fn confusion(lik: [[f64; 2]; 2], states: &[String]) -> MeasurementModel {
    measurement(
        MeasurementRoute::ProbabilisticCategorical,
        MeasurementKind::ProbabilisticFeatureAssignment,
        MeasurementParameters::Categorical {
            levels: states.to_vec(),
            confusion: vec![lik[0][0], lik[0][1], lik[1][0], lik[1][1]],
            column_counts: None,
            max_condition: 100.0,
        },
    )
}

fn binary(sensitivity: f64, specificity: f64) -> MeasurementModel {
    measurement(
        MeasurementRoute::BinaryMisclassification,
        MeasurementKind::NoisyMisclassified,
        MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity,
                specificity,
                sensitivity_variance: 0.0,
                specificity_variance: 0.0,
            }],
            min_youden: 0.1,
        },
    )
}

fn build(spec: &Spec) -> FiniteSequentialModel {
    let states = state_names(spec.swapped);
    FiniteSequentialModel {
        label: "inspection".into(),
        population: "plant_a".into(),
        regime: "operations".into(),
        period: "2026".into(),
        state_variable: "unit_condition".into(),
        states: states.clone(),
        initial: spec.init.to_vec(),
        horizon: spec.horizon,
        actions: spec
            .acts
            .iter()
            .map(|act| ActionSpec {
                label: act.name.into(),
                role: if act.commit { ActionRole::Commit } else { ActionRole::Continue },
                utility_cost: act.cost,
                resource_use: act.res,
                transition: act.trans.iter().flatten().copied().collect(),
                channel: act
                    .lik
                    .map(|lik| ObservationChannel { measurement: confusion(lik, &states) }),
                immediate_utility: act.imm.to_vec(),
                delayed: act
                    .delayed
                    .map(|(delay, utility)| DelayedOutcome { delay, utility: utility.to_vec() }),
            })
            .collect(),
        terminal_utility: spec.terminal.to_vec(),
        censoring: if spec.forfeit { CensoringRule::Forfeit } else { CensoringRule::Refuse },
        resource_budget: spec.budget,
        stopping: spec
            .boundaries
            .iter()
            .enumerate()
            .map(|(i, b)| StoppingBoundary {
                label: format!("boundary_{i}"),
                states: b.states.clone(),
                upper: b.upper,
                lower: b.lower,
                stop_action: b.stop,
            })
            .collect(),
        support: ActionSupport::everywhere(spec.acts.len()),
    }
}

// ---------------------------------------------------------------------------
// Policies as plain tables
// ---------------------------------------------------------------------------

fn det(entries: &[(&[(usize, usize)], usize)]) -> Table {
    entries.iter().map(|(history, action)| (history.to_vec(), vec![(*action, 1.0)])).collect()
}

fn key(history: &[(usize, usize)]) -> HistoryKey {
    HistoryKey::from_steps(
        history.iter().map(|&(action, observation)| HistoryStep { action, observation }),
    )
}

fn policy_of(version: &str, table: &Table) -> ContingentPolicy {
    let mut policy = ContingentPolicy::new(version);
    for (history, distribution) in table {
        policy = policy.decide_stochastic(key(history), distribution.clone()).unwrap();
    }
    policy
}

fn act_now_ship() -> Table {
    det(&[(&[], SHIP)])
}

fn act_now_repair() -> Table {
    det(&[(&[], REPAIR)])
}

fn inspect_then_act() -> Table {
    det(&[(&[], INSPECT), (&[(INSPECT, ALARM)], REPAIR), (&[(INSPECT, NEGATIVE)], SHIP)])
}

fn inspect_twice() -> Table {
    det(&[
        (&[], INSPECT),
        (&[(INSPECT, ALARM)], INSPECT),
        (&[(INSPECT, NEGATIVE)], INSPECT),
        (&[(INSPECT, ALARM), (INSPECT, NEGATIVE)], REPAIR),
        (&[(INSPECT, NEGATIVE), (INSPECT, ALARM)], REPAIR),
    ])
}

fn pilot_inspect_act() -> Table {
    det(&[
        (&[], PILOT),
        (&[(PILOT, 0)], INSPECT),
        (&[(PILOT, 0), (INSPECT, ALARM)], REPAIR),
        (&[(PILOT, 0), (INSPECT, NEGATIVE)], SHIP),
    ])
}

fn wait_three() -> Table {
    det(&[(&[], WAIT), (&[(WAIT, 0)], WAIT), (&[(WAIT, 0), (WAIT, 0)], WAIT)])
}

fn pilot_three() -> Table {
    det(&[(&[], PILOT), (&[(PILOT, 0)], PILOT), (&[(PILOT, 0), (PILOT, 0)], PILOT)])
}

fn stochastic_root() -> Table {
    let mut table = det(&[(&[(INSPECT, NEGATIVE)], SHIP), (&[(INSPECT, ALARM)], REPAIR)]);
    table.insert(Vec::new(), vec![(INSPECT, 0.5), (REPAIR, 0.5)]);
    table
}

// ---------------------------------------------------------------------------
// Independent oracle: forward joint-probability enumeration
// ---------------------------------------------------------------------------

/// Expected total utility, accumulating `joint probability x realized utility`
/// over every observed history. `joint[s]` is `P(history, state = s)`.
fn walk(
    spec: &Spec,
    table: &Table,
    hist: &History,
    joint: [f64; 2],
    pending: &[(usize, usize)],
) -> f64 {
    let t = hist.len();
    let mass = joint[0] + joint[1];
    let mut total = 0.0;
    for &(a, due) in pending {
        if due == t {
            let u = spec.acts[a].delayed.unwrap().1;
            total += joint[0] * u[0] + joint[1] * u[1];
        }
    }
    let remaining: Vec<(usize, usize)> =
        pending.iter().copied().filter(|&(_, due)| due != t).collect();
    if t == spec.horizon {
        assert!(remaining.is_empty() || spec.forfeit, "an outcome is censored by the horizon");
        return total + joint[0] * spec.terminal[0] + joint[1] * spec.terminal[1];
    }
    let post = [joint[0] / mass, joint[1] / mass];
    let mut forced = None;
    for b in &spec.boundaries {
        let m: f64 = b.states.iter().map(|&s| post[s]).sum();
        if b.upper.is_some_and(|u| m >= u) || b.lower.is_some_and(|l| m <= l) {
            forced = Some(vec![(b.stop, 1.0)]);
            break;
        }
    }
    let distribution = forced.unwrap_or_else(|| {
        table.get(hist).unwrap_or_else(|| panic!("no decision for {hist:?}")).clone()
    });
    for (a, pi) in distribution {
        let act = &spec.acts[a];
        let mut branch = joint[0] * act.imm[0] + joint[1] * act.imm[1] - mass * act.cost;
        if act.commit {
            let mut items = remaining.clone();
            if let Some((delay, _)) = act.delayed {
                items.push((a, t + delay));
            }
            let last = items.iter().map(|&(_, due)| due).max().unwrap_or(t);
            let mut state = joint;
            for step in (t + 1)..=last {
                let mut next = [0.0; 2];
                for from in 0..2 {
                    for to in 0..2 {
                        next[to] += state[from] * act.trans[from][to];
                    }
                }
                state = next;
                for &(ia, due) in &items {
                    if due == step {
                        let u = spec.acts[ia].delayed.unwrap().1;
                        branch += state[0] * u[0] + state[1] * u[1];
                    }
                }
            }
        } else {
            let mut next_pending = remaining.clone();
            if let Some((delay, _)) = act.delayed {
                next_pending.push((a, t + delay));
            }
            let symbols = if act.lik.is_some() { 2 } else { 1 };
            for symbol in 0..symbols {
                let mut next = [0.0; 2];
                for from in 0..2 {
                    for to in 0..2 {
                        let lik = act.lik.map_or(1.0, |l| l[symbol][to]);
                        next[to] += joint[from] * act.trans[from][to] * lik;
                    }
                }
                if next[0] + next[1] > 0.0 {
                    let mut child = hist.clone();
                    child.push((a, symbol));
                    branch += walk(spec, table, &child, next, &next_pending);
                }
            }
        }
        total += pi * branch;
    }
    total
}

fn oracle(spec: &Spec, table: &Table) -> f64 {
    walk(spec, table, &Vec::new(), spec.init, &[])
}

fn value(spec: &Spec, table: &Table) -> f64 {
    evaluate_policy(&build(spec), &policy_of("v", table), EvaluationLimits::default(), &ctx())
        .unwrap()
        .value
}

// ---------------------------------------------------------------------------
// Exact values
// ---------------------------------------------------------------------------

#[test]
fn exact_values_match_hand_computation_and_the_independent_oracle() {
    let spec = base_spec(true);
    // (policy, hand value, how it was worked out)
    let cases: Vec<(&str, Table, f64)> = vec![
        // ship now: 0.5 * 10 + 0.5 * (-10)
        ("act_now_ship", act_now_ship(), 0.0),
        // repair now: sound 10 - 4 = 6, faulty (0.5*10 + 0.5*(-10)) - 4 = -4: 0.5*6 + 0.5*(-4)
        ("act_now_repair", act_now_repair(), 1.0),
        // inspect (cost 1/4), then repair after an alarm (P(faulty) = 3/4:
        // 6/4 - 4*3/4 = -1.5) and ship after no alarm (P(faulty) = 1/4:
        // 10*3/4 - 10/4 = 5): -0.25 + 0.5 * (-1.5) + 0.5 * 5
        ("inspect_then_act", inspect_then_act(), 1.5),
        // two inspections (cost 1/2). AA: P = 0.3125, P(faulty) = 0.9 -> forced
        // repair 6*0.1 - 4*0.9 = -3. NN: P = 0.3125, P(faulty) = 0.1 -> forced
        // ship 10*0.9 - 10*0.1 = 8. Mixed: P = 0.375, P(faulty) = 0.5 -> repair 1.
        // -0.5 + 0.3125*(-3) + 0.3125*8 + 0.375*1
        ("inspect_twice", inspect_twice(), 1.4375),
        // pilot (1/4), inspect (1/4), the pilot outcome falls due at time two:
        // 3 - 4 P(faulty): 0 after an alarm, 2 after none -> expected 1; then
        // repair after an alarm (-1.5) / ship after none (5): expected 1.75.
        ("pilot_inspect_act", pilot_inspect_act(), 2.25),
        // pilot, then ship before the outcome is due: ship is worth 0 and the
        // pending outcome is evolved one step under ship's transition (the
        // identity): 3*0.5 - 0.5 = 1. -0.25 + 0 + 1
        ("pilot_then_ship", det(&[(&[], PILOT), (&[(PILOT, 0)], SHIP)]), 0.75),
        // pilot, then repair: step -4, own outcome one step later: belief
        // (0.75, 0.25) -> 10*0.75 - 10*0.25 = 5; the pending outcome one step
        // later: 3*0.75 - 0.25 = 2. -0.25 + (-4 + 5 + 2)
        ("pilot_then_repair", det(&[(&[], PILOT), (&[(PILOT, 0)], REPAIR)]), 2.75),
        // wait to the horizon: P(faulty) 0.5 -> 0.625 -> 0.71875 -> 0.7890625,
        // terminal 2 - 4 * 0.7890625
        ("wait_three", wait_three(), -1.15625),
        // wait once, then repair at P(faulty) = 0.625: 6*0.375 - 4*0.625
        ("wait_then_repair", det(&[(&[], WAIT), (&[(WAIT, 0)], REPAIR)]), -0.25),
        // a coin between inspect-then-act (1.5) and repair now (1.0)
        ("stochastic_root", stochastic_root(), 1.25),
    ];
    for (name, table, hand) in cases {
        let library = value(&spec, &table);
        let independent = oracle(&spec, &table);
        assert!((library - hand).abs() < 1e-12, "{name}: library {library} vs hand {hand}");
        assert!((independent - hand).abs() < 1e-12, "{name}: oracle {independent} vs hand {hand}");
        assert!((library - independent).abs() < 1e-12, "{name}: library vs oracle");
    }
}

#[test]
fn forfeited_and_refused_outcomes_beyond_the_horizon() {
    // pilot x3: outcomes due at 2 and 3 are realized (each 3*0.5 - 0.5 = 1), the
    // one due at 4 is past the horizon. Terminal 2*0.5 - 2*0.5 = 0; cost 3/4.
    let mut spec = base_spec(true);
    let err = evaluate_policy(
        &build(&spec),
        &policy_of("pilot", &pilot_three()),
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.code, "invalid_argument");
    assert_eq!(err.detail, "sequential_evaluation.delayed_outcome_beyond_horizon");

    spec.forfeit = true;
    let evaluation = evaluate_policy(
        &build(&spec),
        &policy_of("pilot", &pilot_three()),
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert!((evaluation.value - 1.25).abs() < 1e-12, "{}", evaluation.value);
    assert!((oracle(&spec, &pilot_three()) - 1.25).abs() < 1e-12);
    let horizon = evaluation.histories.last().unwrap();
    assert_eq!(horizon.kind, NodeKind::Horizon);
    assert_eq!(horizon.forfeited_outcomes, 1);
}

#[test]
fn history_and_decision_receipts_account_for_the_value() {
    let spec = base_spec(true);
    let evaluation = evaluate_policy(
        &build(&spec),
        &policy_of("inspect_twice_v1", &inspect_twice()),
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert_eq!(evaluation.policy_version, "inspect_twice_v1");
    assert_eq!(evaluation.inference, "point_only");
    // root, two after one inspection, four after two.
    assert_eq!(evaluation.histories.len(), 7);
    assert_eq!(evaluation.decisions.len(), 7);
    let accounted: f64 =
        evaluation.histories.iter().map(|h| h.reach_probability * h.realized_utility).sum();
    assert!((accounted - evaluation.value).abs() < 1e-12);
    assert!(evaluation.accounting_residual < 1e-12);

    let find = |history: &[(usize, usize)]| {
        evaluation.histories.iter().find(|h| h.history == key(history)).unwrap()
    };
    let aa = find(&[(INSPECT, ALARM), (INSPECT, ALARM)]);
    assert!((aa.reach_probability - 0.3125).abs() < 1e-15);
    assert!((aa.belief[1] - 0.9).abs() < 1e-12 && (aa.belief[0] - 0.1).abs() < 1e-12);
    assert_eq!(aa.kind, NodeKind::ForcedStop { boundary: 0 });
    // forced repair at P(faulty) = 0.9: -3
    assert!((aa.continuation_value + 3.0).abs() < 1e-12);
    let nn = find(&[(INSPECT, NEGATIVE), (INSPECT, NEGATIVE)]);
    assert_eq!(nn.kind, NodeKind::ForcedStop { boundary: 1 });
    assert!((nn.continuation_value - 8.0).abs() < 1e-12);
    let mixed = find(&[(INSPECT, ALARM), (INSPECT, NEGATIVE)]);
    assert_eq!(mixed.kind, NodeKind::Decision);
    assert!((mixed.reach_probability - 0.1875).abs() < 1e-15);
    assert!((mixed.belief[1] - 0.5).abs() < 1e-12);

    let forced: Vec<_> = evaluation.decisions.iter().filter(|d| d.forced_by.is_some()).collect();
    assert_eq!(forced.len(), 2);
    let root = &evaluation.decisions[0];
    assert!(root.history.is_empty() && root.forced_by.is_none());
    assert_eq!(root.actions, vec![(INSPECT, 1.0)]);
    // the channel's measurement identity is bound
    assert_eq!(evaluation.measurement_identities.len(), 1);
    assert_eq!(evaluation.measurement_identities[0].0, INSPECT);
    assert_eq!(evaluation.measurement_identities[0].1.len(), 64);
}

#[test]
fn value_is_invariant_to_relabelling_the_latent_states() {
    let spec = base_spec(true);
    let swapped = swap_states(&spec);
    let base_model = build(&spec);
    let swapped_model = build(&swapped);
    assert_ne!(base_model.identity_hex(), swapped_model.identity_hex());
    for table in [inspect_twice(), pilot_inspect_act(), wait_three(), stochastic_root()] {
        let a = value(&spec, &table);
        let b = value(&swapped, &table);
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        assert!((oracle(&swapped, &table) - b).abs() < 1e-12);
    }
    // the display label is not part of the identity; the laws are
    let mut relabelled = base_model.clone();
    relabelled.label = "another display label".into();
    assert_eq!(relabelled.identity_hex(), base_model.identity_hex());
    let mut changed = base_model.clone();
    changed.terminal_utility = vec![2.0, -2.5];
    assert_ne!(changed.identity_hex(), base_model.identity_hex());
}

#[test]
fn a_binary_channel_gives_the_same_likelihood_as_the_equivalent_confusion() {
    let spec = base_spec(true);
    let mut model = build(&spec);
    model.actions[INSPECT].channel = Some(ObservationChannel { measurement: binary(0.75, 0.75) });
    let policy = policy_of("v", &inspect_twice());
    let binary_value =
        evaluate_policy(&model, &policy, EvaluationLimits::default(), &ctx()).unwrap().value;
    assert!((binary_value - 1.4375).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn policy_that_takes_an_unsupported_action_is_refused() {
    let spec = base_spec(true);
    let mut model = build(&spec);
    // repair has no support at the first decision time.
    model.support = ActionSupport::everywhere(5)
        .with_history(HistoryKey::root(), vec![true, true, true, false, true]);
    let err = evaluate_policy(
        &model,
        &policy_of("v", &act_now_repair()),
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.code, "history_support_violated");
    assert_eq!(err.detail, "sequential_evaluation.action_not_supported");
    assert!(err.message.contains("repair"));
    // a supported action in the same history is evaluated
    assert!(
        evaluate_policy(
            &model,
            &policy_of("v", &act_now_ship()),
            EvaluationLimits::default(),
            &ctx(),
        )
        .is_ok()
    );
    // the refusal is structured
    assert_eq!(err.to_refusal().code, "history_support_violated");
}

#[test]
fn incomplete_policy_and_resource_budget_are_refused() {
    let spec = base_spec(true);
    let model = build(&spec);
    // no decision for the mixed histories
    let mut incomplete = inspect_twice();
    incomplete.remove(&vec![(INSPECT, ALARM), (INSPECT, NEGATIVE)]);
    let err =
        evaluate_policy(&model, &policy_of("v", &incomplete), EvaluationLimits::default(), &ctx())
            .unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.policy_incomplete");

    // a third inspection exceeds the budget of two (boundary-forced histories
    // never reach it; the mixed ones do)
    let mut greedy = inspect_twice();
    greedy.insert(vec![(INSPECT, ALARM), (INSPECT, NEGATIVE)], vec![(INSPECT, 1.0)]);
    greedy.insert(vec![(INSPECT, NEGATIVE), (INSPECT, ALARM)], vec![(INSPECT, 1.0)]);
    let err =
        evaluate_policy(&model, &policy_of("v", &greedy), EvaluationLimits::default(), &ctx())
            .unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.resource_budget_exceeded");
}

#[test]
fn static_score_tables_and_policy_learning_are_not_evaluation() {
    let table = PolicySpec::ReweightedStaticScores {
        scores: vec![1.0, 0.0, 1.5],
        weights: vec![0.2, 0.3, 0.5],
    };
    let err = table.into_contingent().unwrap_err();
    assert_eq!(err.code, "route_not_supported");
    assert_eq!(err.detail, "sequential_evaluation.static_score_table");
    let learned = PolicySpec::LearnedFromEvaluationData { rule: "fitted_q_iteration".into() };
    let err = learned.into_contingent().unwrap_err();
    assert_eq!(err.code, "route_not_supported");
    assert_eq!(err.detail, "sequential_evaluation.policy_change_refused");
    // a fixed policy passes through unchanged
    let fixed = PolicySpec::Contingent(policy_of("v", &act_now_ship()));
    assert_eq!(fixed.into_contingent().unwrap().version(), "v");

    // and the same through a comparison
    let model = build(&base_spec(true));
    let err = compare_policies(
        &model,
        vec![ComparisonSpec {
            label: "score_table".into(),
            kind: ComparisonKind::ActNow,
            policy: PolicySpec::ReweightedStaticScores { scores: vec![1.0], weights: vec![1.0] },
        }],
        None,
        &SequentialRequest::default(),
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.static_score_table");
}

#[test]
fn transport_and_intervals_are_typed_refusals() {
    let model = build(&base_spec(true));
    assert!(SequentialRequest::default().check(&model).is_ok());
    let same = SequentialRequest {
        target_population: Some("plant_a".into()),
        target_period: Some("2026".into()),
    };
    assert!(same.check(&model).is_ok());
    let other_population =
        SequentialRequest { target_population: Some("plant_b".into()), target_period: None };
    let err = other_population.check(&model).unwrap_err();
    assert_eq!(err.code, "transport_missing_evidence");
    assert_eq!(err.detail, "sequential_evaluation.transported_policy_contract_missing");
    assert!(err.remedy.contains("selection diagram"));
    let other_period =
        SequentialRequest { target_population: None, target_period: Some("2027".into()) };
    let err = other_period.check(&model).unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.temporal_transport_contract_missing");

    let err = request_interval().unwrap_err();
    assert_eq!(err.code, "cell_not_licensed");
    assert_eq!(err.detail, "sequential_evaluation.interval_not_licensed");
}

#[test]
fn observation_channels_pass_through_the_measurement_model_checks() {
    let spec = base_spec(true);
    let check = |edit: &dyn Fn(&mut MeasurementModel)| {
        let mut model = build(&spec);
        let channel = model.actions[INSPECT].channel.as_mut().unwrap();
        edit(&mut channel.measurement);
        model.validate().unwrap_err()
    };
    // a different construct
    let err = check(&|m| m.construct.id = "something_else".into());
    assert_eq!(err.code, "measurement_binding_mismatch");
    assert_eq!(err.detail, "sequential_evaluation.channel_construct_mismatch");
    // the measured column offered as the state variable
    let mut model = build(&spec);
    model.state_variable = "alarm".into();
    model.actions[INSPECT].channel.as_mut().unwrap().measurement.construct.id = "alarm".into();
    assert_eq!(model.validate().unwrap_err().code, "proxy_offered_as_construct");
    // another population or regime
    let err = check(&|m| m.binding.population_id = "plant_b".into());
    assert_eq!(err.code, "measurement_binding_mismatch");
    let err = check(&|m| m.binding.regime_id = "trial".into());
    assert_eq!(err.detail, "sequential_evaluation.channel_binding_mismatch");
    // differential error is never pooled
    let err = check(&|m| {
        m.structure = ErrorStructure::Differential(DifferentialAxis::Arm);
    });
    // the measurement layer refuses first: its parameter layout is shared-rate
    assert!(
        err.code == "differential_error_not_pooled" || err.code == "measurement_model_invalid",
        "{}",
        err.code
    );
    // a malformed model
    let err = check(&|m| m.indicator.id = String::new());
    assert_eq!(err.code, "measurement_model_invalid");
    // confusion levels that are not the model's states
    let err = check(&|m| {
        if let MeasurementParameters::Categorical { levels, .. } = &mut m.parameters {
            levels.swap(0, 1);
        }
    });
    assert_eq!(err.detail, "sequential_evaluation.channel_levels_mismatch");
    // a route without a finite likelihood
    let err = check(&|m| {
        m.route = MeasurementRoute::ClassicalContinuousError;
        m.indicator.kind = MeasurementKind::NoisyMisclassified;
        m.parameters = MeasurementParameters::ClassicalError {
            error_variance: 1.0,
            error_variance_standard_error: 0.0,
            min_reliability: 0.1,
        };
    });
    assert_eq!(err.detail, "sequential_evaluation.channel_route");
}

// ---------------------------------------------------------------------------
// Belief updates and the once-only record
// ---------------------------------------------------------------------------

#[test]
fn belief_updates_record_each_observation_and_action_once() {
    let model = build(&base_spec(true));
    let mut ledger = EvidenceLedger::new();
    let mut tracker = BeliefTracker::new(&model, "unit_17").unwrap();
    assert_eq!(tracker.belief(), &[0.5, 0.5]);
    let receipt = tracker.observe(&mut ledger, INSPECT, ALARM, "inspection_1").unwrap().clone();
    // P(alarm) = 0.5 * 0.25 + 0.5 * 0.75 = 0.5; P(faulty | alarm) = 0.375 / 0.5
    assert!((receipt.probability - 0.5).abs() < 1e-15);
    assert!((receipt.belief[1] - 0.75).abs() < 1e-15);
    assert!(receipt.measurement_identity.is_some());
    assert!(!receipt.parameter_uncertainty_propagated);
    assert_eq!(ledger.sources().len(), 1);
    assert_eq!(tracker.history(), &key(&[(INSPECT, ALARM)]));
    assert_eq!(tracker.time(), 1);

    // the same record again is double counting, and changes nothing
    let before = ledger.state_digest();
    let err = tracker.observe(&mut ledger, INSPECT, ALARM, "inspection_1").unwrap_err();
    assert_eq!(err.code, "evidence_double_counted");
    assert_eq!(ledger.state_digest(), before);
    assert_eq!(tracker.time(), 1);

    // a second observation: odds 3 * 1/3 -> back to one half
    let receipt = tracker.observe(&mut ledger, INSPECT, NEGATIVE, "inspection_2").unwrap().clone();
    assert!((receipt.belief[1] - 0.5).abs() < 1e-15);
    // the budget of two inspections is exhausted
    let err = tracker.observe(&mut ledger, INSPECT, ALARM, "inspection_3").unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.resource_budget_exceeded");
    // a commit action ends the episode and has no observation
    let err = tracker.observe(&mut ledger, SHIP, 0, "ship_1").unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.commit_has_no_observation");
    // a symbol outside the channel
    let mut other = BeliefTracker::new(&model, "unit_18").unwrap();
    let err = other.observe(&mut ledger, INSPECT, 5, "inspection_1").unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.observation_impossible");
    // the same record identity in another session is a different record
    other.observe(&mut ledger, INSPECT, ALARM, "inspection_1").unwrap();
    assert_eq!(ledger.sources().len(), 3);
}

#[test]
fn an_observation_the_model_forbids_is_refused_not_conditioned_on() {
    // The unit is known to be sound, and a sound unit never alarms.
    let mut spec = base_spec(true);
    spec.init = [1.0, 0.0];
    spec.acts[INSPECT].lik = Some([[1.0, 0.25], [0.0, 0.75]]);
    let model = build(&spec);
    let mut ledger = EvidenceLedger::new();
    let mut tracker = BeliefTracker::new(&model, "s").unwrap();
    let err = tracker.observe(&mut ledger, INSPECT, ALARM, "r1").unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.observation_impossible");
    assert!(ledger.sources().is_empty());
    assert_eq!(tracker.time(), 0);
    // the possible observation updates the belief (and keeps it on sound)
    let ok = tracker.observe(&mut ledger, INSPECT, NEGATIVE, "r1").unwrap();
    assert!((ok.belief[0] - 1.0).abs() < 1e-15);
}

// ---------------------------------------------------------------------------
// Comparisons and the value of collecting evidence
// ---------------------------------------------------------------------------

#[test]
fn comparisons_score_collecting_evidence_through_the_design_ranker() {
    // Without boundaries the cross-check scope applies.
    let spec = base_spec(false);
    let model = build(&spec);
    let specs = vec![
        ComparisonSpec {
            label: "repair_now".into(),
            kind: ComparisonKind::ActNow,
            policy: PolicySpec::Contingent(policy_of("v_now", &act_now_repair())),
        },
        ComparisonSpec {
            label: "wait_then_repair".into(),
            kind: ComparisonKind::Wait,
            policy: PolicySpec::Contingent(policy_of(
                "v_wait",
                &det(&[(&[], WAIT), (&[(WAIT, 0)], REPAIR)]),
            )),
        },
        ComparisonSpec {
            label: "inspect_then_repair".into(),
            kind: ComparisonKind::CollectEvidence,
            policy: PolicySpec::Contingent(policy_of(
                "v_collect",
                &det(&[
                    (&[], INSPECT),
                    (&[(INSPECT, ALARM)], REPAIR),
                    (&[(INSPECT, NEGATIVE)], REPAIR),
                ]),
            )),
        },
        ComparisonSpec {
            label: "inspect_then_choose".into(),
            kind: ComparisonKind::ChooseAfterObservation,
            policy: PolicySpec::Contingent(policy_of("v_choose", &inspect_then_act())),
        },
    ];
    let voi = VoiRequest { collect_action: INSPECT, commit_actions: vec![SHIP, REPAIR] };
    let report = compare_policies(
        &model,
        specs,
        Some(&voi),
        &SequentialRequest::default(),
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    let values: Vec<f64> = report.rows.iter().map(|r| r.value).collect();
    // repair now 1; wait then repair -0.25; inspect then always repair
    // -0.25 + 0.5 * (-1.5) + 0.5 * 3.5 = 0.75; inspect then choose 1.5.
    for (got, want) in values.iter().zip([1.0, -0.25, 0.75, 1.5]) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
    assert_eq!(report.best, 3);
    assert!((report.rows[3].difference_from_first - 0.5).abs() < 1e-12);
    assert_eq!(report.inference, "point_only");

    // The ranker's value of the observation: prior value 1 (repair), posterior
    // value 0.5*(-1.5) + 0.5*5 = 1.75, so EVSI = 0.75; cost 1/4; net 0.5.
    let check = report.value_of_information.unwrap();
    assert!((check.prior_value - 1.0).abs() < 1e-12);
    assert!((check.expected_value_of_sample_information - 0.75).abs() < 1e-12);
    assert!((check.net_value_of_collecting - 0.5).abs() < 1e-12);
    assert!((check.predicted_policy_value - 1.5).abs() < 1e-12);
    assert!((check.enumerated_policy_value - 1.5).abs() < 1e-12);
    assert_eq!(check.ranking_evaluation, "exact");
    assert_eq!(check.implemented_functional, "preposterior_expected_value_of_sample_information");
    // the supplied choose-after-observation policy is the Bayes policy
    assert_eq!(
        check.bayes_after_observation.distribution(&key(&[(INSPECT, ALARM)])).unwrap(),
        [(REPAIR, 1.0)]
    );
    assert_eq!(
        check.bayes_after_observation.distribution(&key(&[(INSPECT, NEGATIVE)])).unwrap(),
        [(SHIP, 1.0)]
    );

    // the same value through the direct entry point
    let direct =
        value_of_collecting_evidence(&model, &voi, EvaluationLimits::default(), &ctx()).unwrap();
    assert_eq!(
        direct.expected_value_of_sample_information.to_bits(),
        check.expected_value_of_sample_information.to_bits()
    );
}

#[test]
fn a_comparison_must_implement_its_declared_kind() {
    let model = build(&base_spec(false));
    let mismatched = compare_policies(
        &model,
        vec![ComparisonSpec {
            label: "not_now".into(),
            kind: ComparisonKind::ActNow,
            policy: PolicySpec::Contingent(policy_of("v", &inspect_then_act())),
        }],
        None,
        &SequentialRequest::default(),
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(mismatched.detail, "sequential_evaluation.comparison_kind_mismatch");
    // "choose differently" with the same decision after both observations
    let same = compare_policies(
        &model,
        vec![ComparisonSpec {
            label: "same".into(),
            kind: ComparisonKind::ChooseAfterObservation,
            policy: PolicySpec::Contingent(policy_of(
                "v",
                &det(&[
                    (&[], INSPECT),
                    (&[(INSPECT, ALARM)], REPAIR),
                    (&[(INSPECT, NEGATIVE)], REPAIR),
                ]),
            )),
        }],
        None,
        &SequentialRequest::default(),
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(same.detail, "sequential_evaluation.comparison_kind_mismatch");
    // the cross-check does not cover a model with boundaries
    let bounded = build(&base_spec(true));
    let err = value_of_collecting_evidence(
        &bounded,
        &VoiRequest { collect_action: INSPECT, commit_actions: vec![SHIP, REPAIR] },
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.code, "route_not_supported");
    assert_eq!(err.detail, "sequential_evaluation.voi_scope");
    // an action with its own utility is not pure information
    let err = value_of_collecting_evidence(
        &model,
        &VoiRequest { collect_action: PILOT, commit_actions: vec![SHIP, REPAIR] },
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.voi_collect_action");
}

// ---------------------------------------------------------------------------
// Rollouts on common random streams
// ---------------------------------------------------------------------------

#[test]
fn rollouts_agree_with_the_enumeration_within_the_declared_tolerance() {
    let spec = base_spec(true);
    let model = build(&spec);
    for (table, exact) in [(inspect_twice(), 1.4375), (pilot_inspect_act(), 2.25)] {
        let policy = policy_of("v", &table);
        let estimate = rollout_policy(
            &model,
            &policy,
            RolloutConfig { seed: 20_260_101, rollouts: 20_000 },
            &ctx(),
        )
        .unwrap();
        // predeclared: within five Monte Carlo standard errors
        assert!(estimate.monte_carlo_standard_error > 0.0);
        assert!(
            (estimate.mean - exact).abs() <= 5.0 * estimate.monte_carlo_standard_error,
            "mean {} vs {exact} (se {})",
            estimate.mean,
            estimate.monte_carlo_standard_error
        );
        assert_eq!(estimate.error_label, "uncalibrated_diagnostic");
        assert_eq!(estimate.inference, "point_only");
        // replay is bit-identical
        let again = rollout_policy(
            &model,
            &policy,
            RolloutConfig { seed: 20_260_101, rollouts: 20_000 },
            &ctx(),
        )
        .unwrap();
        assert_eq!(again.mean.to_bits(), estimate.mean.to_bits());
    }
}

#[test]
fn stream_identities_are_semantic_and_policies_share_them() {
    // a pure function of (seed, rollout, time, role)
    let u = stream_uniform(7, 3, 2, StreamRole::Transition);
    assert_eq!(u.to_bits(), stream_uniform(7, 3, 2, StreamRole::Transition).to_bits());
    assert!((0.0..1.0).contains(&u));
    assert_ne!(u.to_bits(), stream_uniform(7, 3, 2, StreamRole::Observation).to_bits());
    assert_ne!(u.to_bits(), stream_uniform(7, 4, 2, StreamRole::Transition).to_bits());
    assert_ne!(u.to_bits(), stream_uniform(7, 3, 3, StreamRole::Transition).to_bits());
    assert_ne!(u.to_bits(), stream_uniform(8, 3, 2, StreamRole::Transition).to_bits());

    // a policy that differs only at a history it never reaches sees the same
    // streams, so its rollouts are bit-identical
    let model = build(&base_spec(true));
    let base = policy_of("a", &inspect_twice());
    let mut extended = inspect_twice();
    extended.insert(vec![(WAIT, 0), (WAIT, 0), (WAIT, 0)], vec![(SHIP, 1.0)]);
    let extended = policy_of("b", &extended);
    let config = RolloutConfig { seed: 5, rollouts: 2_000 };
    let a = rollout_policy(&model, &base, config, &ctx()).unwrap();
    let b = rollout_policy(&model, &extended, config, &ctx()).unwrap();
    assert_eq!(a.mean.to_bits(), b.mean.to_bits());
    assert_eq!(a.monte_carlo_standard_error.to_bits(), b.monte_carlo_standard_error.to_bits());
    // a different policy on the same streams gives a different estimate
    let c = rollout_policy(&model, &policy_of("c", &act_now_repair()), config, &ctx()).unwrap();
    assert_ne!(a.mean.to_bits(), c.mean.to_bits());
}

#[test]
fn rollouts_observe_cancellation_and_bounds() {
    let model = build(&base_spec(true));
    let policy = policy_of("v", &act_now_ship());
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let err = rollout_policy(&model, &policy, RolloutConfig { seed: 1, rollouts: 10 }, &cancelled)
        .unwrap_err();
    assert_eq!(err.code, "cancelled_no_claim");
    let err = rollout_policy(&model, &policy, RolloutConfig { seed: 1, rollouts: 0 }, &ctx())
        .unwrap_err();
    assert_eq!(err.code, "invalid_argument");
    let err = evaluate_policy(
        &model,
        &policy_of("v", &inspect_twice()),
        EvaluationLimits { max_nodes: 3, max_receipts: 100 },
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.limit_exceeded");
}

// ---------------------------------------------------------------------------
// Off-policy evaluation from logged finite data
// ---------------------------------------------------------------------------

fn mu(history: &History) -> Vec<(usize, f64)> {
    let inspections = history.iter().filter(|(a, _)| *a == INSPECT).count();
    if inspections >= 2 {
        vec![(SHIP, 0.5), (REPAIR, 0.5)]
    } else {
        vec![(INSPECT, 0.5), (SHIP, 0.25), (REPAIR, 0.25)]
    }
}

/// Unnormalized `P(history, state)` by the forward recursion.
fn forward(spec: &Spec, hist: &History) -> [f64; 2] {
    let mut joint = spec.init;
    for &(a, symbol) in hist {
        let act = &spec.acts[a];
        let mut next = [0.0; 2];
        for from in 0..2 {
            for to in 0..2 {
                let lik = act.lik.map_or(1.0, |l| l[symbol][to]);
                next[to] += joint[from] * act.trans[from][to] * lik;
            }
        }
        joint = next;
    }
    joint
}

type Step = (usize, Option<usize>, f64);

#[allow(clippy::too_many_arguments, reason = "an explicit recursion state for the data generator")]
fn emit(
    spec: &Spec,
    hist: &History,
    state: usize,
    steps: &[Step],
    prob: f64,
    outcome: f64,
    pending: &[(usize, usize)],
    out: &mut Vec<(Vec<Step>, f64, f64)>,
) {
    let t = hist.len();
    let mut outcome = outcome;
    let mut remaining = Vec::new();
    for &(a, due) in pending {
        if due == t {
            outcome += spec.acts[a].delayed.unwrap().1[state];
        } else {
            remaining.push((a, due));
        }
    }
    if t == spec.horizon {
        out.push((steps.to_vec(), outcome + spec.terminal[state], prob));
        return;
    }
    let joint = forward(spec, hist);
    let mass = joint[0] + joint[1];
    let post = [joint[0] / mass, joint[1] / mass];
    let mut distribution = mu(hist);
    for b in &spec.boundaries {
        let m: f64 = b.states.iter().map(|&s| post[s]).sum();
        if b.upper.is_some_and(|u| m >= u) || b.lower.is_some_and(|l| m <= l) {
            distribution = vec![(b.stop, 1.0)];
            break;
        }
    }
    for (a, p) in distribution {
        let act = &spec.acts[a];
        let here = outcome + act.imm[state];
        if act.commit {
            let mut items = remaining.clone();
            if let Some((delay, _)) = act.delayed {
                items.push((a, t + delay));
            }
            let last = items.iter().map(|&(_, due)| due).max().unwrap_or(t);
            let mut extended = steps.to_vec();
            extended.push((a, None, p));
            expand(spec, act, state, t, last, &items, prob * p, here, &extended, out);
        } else {
            let mut next_pending = remaining.clone();
            if let Some((delay, _)) = act.delayed {
                next_pending.push((a, t + delay));
            }
            let symbols = if act.lik.is_some() { 2 } else { 1 };
            for to in 0..2 {
                for symbol in 0..symbols {
                    let lik = act.lik.map_or(1.0, |l| l[symbol][to]);
                    let pl = act.trans[state][to] * lik;
                    if pl <= 0.0 {
                        continue;
                    }
                    let mut child = hist.clone();
                    child.push((a, symbol));
                    let mut extended = steps.to_vec();
                    extended.push((a, Some(symbol), p));
                    emit(spec, &child, to, &extended, prob * p * pl, here, &next_pending, out);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments, reason = "an explicit recursion state for the data generator")]
fn expand(
    spec: &Spec,
    act: &Act,
    state: usize,
    step: usize,
    last: usize,
    items: &[(usize, usize)],
    prob: f64,
    outcome: f64,
    steps: &[Step],
    out: &mut Vec<(Vec<Step>, f64, f64)>,
) {
    if step == last {
        out.push((steps.to_vec(), outcome, prob));
        return;
    }
    for to in 0..2 {
        let pt = act.trans[state][to];
        if pt <= 0.0 {
            continue;
        }
        let mut o = outcome;
        for &(ia, due) in items {
            if due == step + 1 {
                o += spec.acts[ia].delayed.unwrap().1[to];
            }
        }
        expand(spec, act, to, step + 1, last, items, prob * pt, o, steps, out);
    }
}

const LOG_SIZE: f64 = 65_536.0;

/// The full finite population of trajectories under the logging policy, with
/// each distinct (history, outcome) repeated in proportion to its exact
/// probability. Every probability here is a dyadic rational.
fn log_under(spec: &Spec) -> Vec<LoggedTrajectory> {
    let mut leaves = Vec::new();
    for state in 0..2 {
        emit(spec, &Vec::new(), state, &[], spec.init[state], 0.0, &[], &mut leaves);
    }
    let mut groups: BTreeMap<String, (Vec<Step>, f64, f64)> = BTreeMap::new();
    for (steps, outcome, prob) in leaves {
        let id = format!("{steps:?}|{}", outcome.to_bits());
        groups.entry(id).or_insert((steps, outcome, 0.0)).2 += prob;
    }
    let mut log = Vec::new();
    for (group, (steps, outcome, prob)) in groups.values().enumerate() {
        let count = prob * LOG_SIZE;
        assert!((count - count.round()).abs() < 1e-9, "non-dyadic group probability {prob}");
        for copy in 0..(count.round() as usize) {
            log.push(LoggedTrajectory {
                id: format!("t{group:04}-{copy:05}"),
                steps: steps
                    .iter()
                    .map(|&(action, observation, p)| LoggedStep {
                        action,
                        observation,
                        assignment_probability: p,
                    })
                    .collect(),
                outcome_utility: *outcome,
            });
        }
    }
    log
}

fn behaviour_policy() -> ContingentPolicy {
    let mut table: Table = BTreeMap::new();
    let mut frontier: Vec<History> = vec![Vec::new()];
    while let Some(history) = frontier.pop() {
        let distribution = mu(&history);
        table.insert(history.clone(), distribution.clone());
        for (action, _) in distribution {
            if action == INSPECT && history.iter().filter(|(a, _)| *a == INSPECT).count() < 2 {
                for symbol in [NEGATIVE, ALARM] {
                    let mut child = history.clone();
                    child.push((INSPECT, symbol));
                    frontier.push(child);
                }
            }
        }
    }
    policy_of("logging_v1", &table)
}

#[test]
fn off_policy_weighting_recovers_the_enumerated_value_and_the_log_is_counted_once() {
    let spec = base_spec(true);
    let model = build(&spec);
    let log = log_under(&spec);
    assert_eq!(log.len(), 65_536);
    let target = policy_of("inspect_then_act_v1", &inspect_then_act());
    let behaviour = behaviour_policy();
    let mut ledger = EvidenceLedger::new();
    let estimate = evaluate_off_policy(
        &model,
        &target,
        &behaviour,
        &log,
        &mut ledger,
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    // The log is the whole population, so the importance-weighted mean is the
    // g-formula value (hand value 1.5) up to rounding.
    assert!((estimate.ordinary - 1.5).abs() < 1e-9, "{}", estimate.ordinary);
    assert!((estimate.self_normalized - 1.5).abs() < 1e-9);
    assert!((estimate.ordinary - oracle(&spec, &inspect_then_act())).abs() < 1e-9);
    // weights are 8 on the consistent quarter... mass: P(inspect then the
    // targeted commit) = 0.5 * 0.25 = 1/8, so a fraction 1/8 of trajectories
    // carry weight 8: ESS = N/8.
    assert!((estimate.max_weight - 8.0).abs() < 1e-12);
    assert!((estimate.effective_sample_size - LOG_SIZE / 8.0).abs() < 1e-6);
    assert_eq!(estimate.inference, "point_only");
    assert!(estimate.uncalibrated_standard_error.unwrap() > 0.0);
    assert_eq!(estimate.trajectories, 65_536);
    assert!(!estimate.reused_recorded_log);
    assert!(estimate.records_counted > 65_536);

    // evaluating the same log again re-uses the recorded observations
    let state = ledger.state_digest();
    let again = evaluate_off_policy(
        &model,
        &policy_of("another_policy", &act_now_ship()),
        &behaviour,
        &log,
        &mut ledger,
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert!(again.reused_recorded_log);
    assert_eq!(ledger.state_digest(), state);
    // ship now: the log's own value of "ship at the root" is 0
    assert!((again.ordinary - 0.0).abs() < 1e-9);

    // permutation invariance: a reversed log is the same estimate, bit for bit
    let mut reversed = log.clone();
    reversed.reverse();
    let mut fresh = EvidenceLedger::new();
    let permuted = evaluate_off_policy(
        &model,
        &target,
        &behaviour,
        &reversed,
        &mut fresh,
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert_eq!(permuted.ordinary.to_bits(), estimate.ordinary.to_bits());
    assert_eq!(permuted.ledger_state, estimate.ledger_state);
}

#[test]
fn off_policy_refusals_positivity_duplicates_and_inconsistent_logs() {
    let spec = base_spec(true);
    let model = build(&spec);
    let full = log_under(&spec);
    // a small sample that still contains consistent trajectories
    let log: Vec<LoggedTrajectory> =
        full.iter().filter(|t| t.steps[0].action == INSPECT).step_by(40).cloned().collect();
    assert!(log.len() > 50);
    let target = policy_of("inspect_then_act_v1", &inspect_then_act());
    let behaviour = behaviour_policy();
    let run = |model: &FiniteSequentialModel,
               target: &ContingentPolicy,
               behaviour: &ContingentPolicy,
               log: &[LoggedTrajectory],
               ledger: &mut EvidenceLedger| {
        evaluate_off_policy(
            model,
            target,
            behaviour,
            log,
            ledger,
            EvaluationLimits::default(),
            &ctx(),
        )
    };
    run(&model, &target, &behaviour, &log, &mut EvidenceLedger::new()).unwrap();

    // positivity: the target takes `pilot`, which the logging policy never takes
    let pilot_target = policy_of("pilot_first", &det(&[(&[], PILOT), (&[(PILOT, 0)], SHIP)]));
    let err = run(&model, &pilot_target, &behaviour, &log, &mut EvidenceLedger::new()).unwrap_err();
    assert_eq!(err.code, "history_support_violated");
    assert_eq!(err.detail, "sequential_evaluation.behaviour_support_missing");

    // a trajectory identity repeated
    let mut duplicated = log.clone();
    duplicated.push(log[0].clone());
    let err =
        run(&model, &target, &behaviour, &duplicated, &mut EvidenceLedger::new()).unwrap_err();
    assert_eq!(err.code, "evidence_double_counted");
    assert_eq!(err.detail, "sequential_evaluation.trajectory_recorded_twice");

    // a log whose records partly overlap what the ledger already holds
    let mut ledger = EvidenceLedger::new();
    run(&model, &target, &behaviour, &log, &mut ledger).unwrap();
    let before = ledger.state_digest();
    let mut revised = log.clone();
    revised[0].outcome_utility += 1.0;
    let err = run(&model, &target, &behaviour, &revised, &mut ledger).unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.log_partially_counted");
    assert_eq!(ledger.state_digest(), before);

    // a logged probability that is not the logging policy's
    let mut wrong = log.clone();
    wrong[0].steps[0].assignment_probability = 0.25;
    let err = run(&model, &target, &behaviour, &wrong, &mut EvidenceLedger::new()).unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.logged_trajectory_inconsistent");

    // a trajectory that keeps inspecting after a declared boundary was crossed
    let mut ignoring = log[0].clone();
    ignoring.id = "ignores_boundary".into();
    ignoring.steps = vec![
        LoggedStep { action: INSPECT, observation: Some(ALARM), assignment_probability: 0.5 },
        LoggedStep { action: INSPECT, observation: Some(ALARM), assignment_probability: 0.5 },
        LoggedStep { action: INSPECT, observation: Some(ALARM), assignment_probability: 0.5 },
    ];
    let err =
        run(&model, &target, &behaviour, &[ignoring], &mut EvidenceLedger::new()).unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.logged_trajectory_inconsistent");

    // no logged trajectory is consistent with the target: no overlap
    let nothing: Vec<LoggedTrajectory> =
        full.iter().filter(|t| t.steps[0].action == SHIP).take(10).cloned().collect();
    let err = run(&model, &target, &behaviour, &nothing, &mut EvidenceLedger::new()).unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.no_overlap");
}

#[test]
fn off_policy_standard_error_preserves_small_spread_at_large_location() {
    let model = build(&base_spec(false));
    let policy = policy_of("ship_only", &act_now_ship());
    let log: Vec<LoggedTrajectory> = [1e9 - 1.0, 1e9 + 1.0]
        .iter()
        .enumerate()
        .map(|(i, &outcome)| LoggedTrajectory {
            id: format!("row_{i}"),
            steps: vec![LoggedStep {
                action: SHIP,
                observation: None,
                assignment_probability: 1.0,
            }],
            outcome_utility: outcome,
        })
        .collect();
    let estimate = evaluate_off_policy(
        &model,
        &policy,
        &policy,
        &log,
        &mut EvidenceLedger::new(),
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert!((estimate.uncalibrated_standard_error.unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn off_policy_refuses_overflowed_returns_without_recording_the_log() {
    let spec = base_spec(false);
    let mut model = build(&spec);
    model.actions[SHIP].utility_cost = f64::MAX;
    let policy = policy_of("ship_only", &act_now_ship());
    let log = [LoggedTrajectory {
        id: "overflowing_return".into(),
        steps: vec![LoggedStep { action: SHIP, observation: None, assignment_probability: 1.0 }],
        outcome_utility: -f64::MAX,
    }];
    let mut ledger = EvidenceLedger::new();
    let initial_state = ledger.state_digest();
    let error = evaluate_off_policy(
        &model,
        &policy,
        &policy,
        &log,
        &mut ledger,
        EvaluationLimits::default(),
        &ctx(),
    )
    .expect_err("a finite-input net return that overflows must be refused");
    assert_eq!(error.detail, "sequential_evaluation.return_overflow");
    assert_eq!(ledger.state_digest(), initial_state);
}

// ---------------------------------------------------------------------------
// The replayable artifact
// ---------------------------------------------------------------------------

fn sealed_artifact() -> SequentialEvaluationArtifact {
    let spec = base_spec(true);
    let model = build(&spec);
    let policy = policy_of("inspect_then_act_v1", &inspect_then_act());
    let trajectories: Vec<LoggedTrajectory> =
        log_under(&spec).into_iter().filter(|t| t.steps[0].action == INSPECT).step_by(97).collect();
    let off = OffPolicyInput { behaviour: behaviour_policy(), trajectories };
    SequentialEvaluationArtifact::new(
        &model,
        &policy,
        Some(&off),
        Some(RolloutConfig { seed: 9, rollouts: 500 }),
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap()
}

fn expected_of(artifact: &SequentialEvaluationArtifact) -> ExpectedEvaluation {
    let identities = artifact.identities();
    ExpectedEvaluation {
        model: Some(identities.model.clone()),
        policy: Some(identities.policy.clone()),
        evaluation: Some(identities.evaluation.clone()),
    }
}

/// Apply a change, reseal it consistently, and ask a consumer to accept it.
fn tampered(
    bytes: &[u8],
    change: impl FnOnce(&mut SequentialEvaluationArtifact),
) -> Result<SequentialEvaluationArtifact, String> {
    let mut artifact = SequentialEvaluationArtifact::from_bytes(bytes).unwrap();
    change(&mut artifact);
    artifact.seal().unwrap();
    let resealed = artifact.to_bytes("resealed").unwrap();
    consume(&resealed, None, &ctx()).map_err(|error| error.to_string())
}

#[test]
fn artifact_replays_bit_for_bit_and_carries_receipts_for_a_fresh_consumer() {
    let artifact = sealed_artifact();
    assert!((artifact.value() - 1.5).abs() < 1e-12);
    let bytes = artifact.to_bytes("sequential-evaluation-1").unwrap();
    let consumed = consume(&bytes, Some(&expected_of(&artifact)), &ctx()).unwrap();
    assert_eq!(consumed.value().to_bits(), artifact.value().to_bits());
    assert_eq!(consumed.body, artifact.body);
    assert_eq!(consumed.body.inference, "point_only");
    assert_eq!(consumed.body.policy.version, "inspect_then_act_v1");
    // history and decision receipts: root, alarm, no alarm
    assert_eq!(consumed.body.exact.histories.len(), 3);
    assert_eq!(consumed.body.exact.decisions.len(), 3);
    assert!(consumed.body.off_policy.is_some());
    assert!(consumed.body.rollout.is_some());
    // a policy-only artifact carries neither estimate
    let spec = base_spec(true);
    let plain = SequentialEvaluationArtifact::new(
        &build(&spec),
        &policy_of("inspect_twice_v1", &inspect_twice()),
        None,
        None,
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert!((plain.value() - 1.4375).abs() < 1e-12);
    assert!(plain.body.off_policy.is_none() && plain.body.rollout.is_none());
    let plain_bytes = plain.to_bytes("plain").unwrap();
    consume(&plain_bytes, Some(&expected_of(&plain)), &ctx()).unwrap();
}

#[test]
fn semantic_mutations_of_a_resealed_artifact_are_refused() {
    let artifact = sealed_artifact();
    let bytes = artifact.to_bytes("sequential-evaluation-1").unwrap();

    let refused = |what: &str, change: &dyn Fn(&mut SequentialEvaluationArtifact), detail: &str| {
        let error = tampered(&bytes, |a| change(a)).unwrap_err();
        assert!(error.contains(detail), "{what}: {error}");
    };
    // a stored value
    refused("value", &|a| a.body.exact.value_bits ^= 1, "value_mismatch");
    // a history receipt
    refused(
        "history",
        &|a| a.body.exact.histories[0].realized_bits ^= 1,
        "history_receipt_mismatch",
    );
    // a decision receipt
    refused(
        "decision",
        &|a| a.body.exact.decisions[0].actions[0].action = SHIP as u64,
        "decision_receipt_mismatch",
    );
    // a policy entry: after an alarm the policy now ships (decisions are in
    // canonical history order: root, no alarm, alarm), which no longer replays
    refused(
        "policy",
        &|a| a.body.policy.decisions[2].actions[0].action = SHIP as u64,
        "value_mismatch",
    );
    // a model law that matters
    refused(
        "model law",
        &|a| a.body.model.actions[SHIP].immediate_bits[0] = 11.0_f64.to_bits(),
        "value_mismatch",
    );
    // a model field that does not change this value still changes the identity
    refused(
        "model identity",
        &|a| a.body.model.terminal_bits[0] = 3.0_f64.to_bits(),
        "identity_mismatch",
    );
    // the off-policy estimate and its inputs
    refused(
        "off-policy estimate",
        &|a| a.body.off_policy.as_mut().unwrap().ordinary_bits ^= 1,
        "off_policy_mismatch",
    );
    refused(
        "logged outcome",
        &|a| a.body.off_policy.as_mut().unwrap().trajectories[0].outcome_bits = 99.0_f64.to_bits(),
        "off_policy_mismatch",
    );
    // the rollout diagnostic
    refused("rollout", &|a| a.body.rollout.as_mut().unwrap().mean_bits ^= 1, "rollout_mismatch");
    // the stated claim and the identities
    refused("claim", &|a| a.body.inference = "interval".into(), "version");
    refused(
        "evaluation identity",
        &|a| a.body.identities.evaluation = "0".repeat(64),
        "identity_mismatch",
    );
}

#[test]
fn a_retained_identity_refuses_another_evaluation_and_corruption_is_refused() {
    let artifact = sealed_artifact();
    let bytes = artifact.to_bytes("one").unwrap();
    let other = SequentialEvaluationArtifact::new(
        &build(&base_spec(true)),
        &policy_of("act_now_repair_v1", &act_now_repair()),
        None,
        None,
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    let other_bytes = other.to_bytes("two").unwrap();
    // the other artifact is internally consistent ...
    consume(&other_bytes, None, &ctx()).unwrap();
    // ... but is not the evaluation the consumer retained
    let error = consume(&other_bytes, Some(&expected_of(&artifact)), &ctx()).unwrap_err();
    assert!(error.to_string().contains("identity_mismatch"), "{error}");
    let error = consume(&bytes, Some(&expected_of(&other)), &ctx()).unwrap_err();
    assert!(error.to_string().contains("identity_mismatch"), "{error}");

    // a byte flipped inside the container
    let mut corrupted = bytes.clone();
    let middle = corrupted.len() / 2;
    corrupted[middle] ^= 0x5a;
    assert_eq!(
        consume(&corrupted, None, &ctx()).unwrap_err().to_string(),
        "checksum mismatch for section `policy_evaluation.body`"
    );
    // not an artifact at all
    assert_eq!(
        consume(b"not an artifact", None, &ctx()).unwrap_err().to_string(),
        "bad artifact magic"
    );
    // an edit that was not resealed cannot be written
    let mut unsealed = SequentialEvaluationArtifact::from_bytes(&bytes).unwrap();
    unsealed.body.exact.value_bits ^= 1;
    assert_eq!(
        unsealed.to_bytes("unsealed").unwrap_err().to_string(),
        "reason=route_not_supported: sequential_evaluation.identity_mismatch: body digest changed"
    );
}

#[test]
fn deep_single_branch_horizon_uses_bounded_native_stack() {
    let mut spec = base_spec(false);
    spec.horizon = 1_024;
    spec.acts[WAIT].trans = IDENTITY;
    spec.acts[WAIT].imm = [1.0, 1.0];
    spec.terminal = [2.0, 2.0];
    let model = build(&spec);
    let mut policy = ContingentPolicy::new("deep_wait");
    let mut history = HistoryKey::root();
    for _ in 0..spec.horizon {
        policy = policy.decide(history.clone(), WAIT);
        history = history.extended(WAIT, 0);
    }
    let evaluation = std::thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn(move || {
            evaluate_policy(&model, &policy, EvaluationLimits::default(), &ctx()).unwrap()
        })
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(evaluation.value, 1_026.0);
    assert_eq!(evaluation.nodes_visited, 1_025);
    assert_eq!(evaluation.histories.len(), 1_025);
    assert_eq!(evaluation.decisions.len(), 1_024);
    assert_eq!(evaluation.accounting_residual, 0.0);
    for (time, receipt) in evaluation.histories.iter().enumerate() {
        assert_eq!(receipt.time, time);
        assert_eq!(receipt.continuation_value, (1_026 - time) as f64);
    }
}
