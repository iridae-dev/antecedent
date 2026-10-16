//! Comparing supplied policies and valuing the collection of evidence, each
//! against its own hand-solved value.
//!
//! Inspection tree: a unit is `sound` or `faulty` (one half each). `inspect`
//! costs `inspect_cost`, raises an alarm with probability 3/4 when faulty and
//! 1/4 when sound; `ship` pays +10 / -10; `repair` costs 4, and one decision
//! time later a faulty unit is sound with probability 1/2, then +10 / -10.
//! Hand values: ship now 0, repair now `0.5 * 6 + 0.5 * (-4) = 1`.
//! After an alarm `P(faulty) = 3/4`: ship `-5`, repair `-3/2`; after no alarm
//! `P(faulty) = 1/4`: ship `5`, repair `7/2`; `P(alarm) = 1/2`. Choosing after
//! the observation is worth `-cost + 1/2 (-3/2) + 1/2 (5) = 7/4 - cost`, so the
//! expected value of the sample information is `7/4 - 1 = 3/4`.

use std::collections::BTreeMap;

use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    ArmBinding, ErrorStructure, MeasurementBinding, MeasurementKind, MeasurementModel,
    MeasurementParameters, MeasurementRoute, MeasurementVariable, ParameterEvidence,
    RepresentationVersion,
};
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ComparisonKind, ComparisonSpec,
    ContingentPolicy, DelayedOutcome, EvaluationLimits, FiniteSequentialModel, HistoryKey,
    HistoryStep, ObservationChannel, PolicySpec, SequentialRequest, VoiRequest, compare_policies,
    value_of_collecting_evidence,
};

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

fn alarm_panel() -> MeasurementModel {
    let states = vec!["sound".to_owned(), "faulty".to_owned()];
    MeasurementModel {
        model_id: "inspection_alarm".into(),
        route: MeasurementRoute::ProbabilisticCategorical,
        indicator: MeasurementVariable {
            id: "alarm".into(),
            kind: MeasurementKind::ProbabilisticFeatureAssignment,
            units: "category".into(),
        },
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
        parameters: MeasurementParameters::Categorical {
            levels: states,
            // rows are symbols (no alarm, alarm), columns the states.
            confusion: vec![0.75, 0.25, 0.25, 0.75],
            column_counts: None,
            max_condition: 100.0,
        },
        representation: RepresentationVersion {
            coordinate_system_id: "alarm_panel".into(),
            version: 1,
        },
        evidence: ParameterEvidence::Declared { declarer: "reliability_engineering".into() },
    }
}

fn identity() -> Vec<f64> {
    vec![1.0, 0.0, 0.0, 1.0]
}

fn model(inspect_cost: f64) -> FiniteSequentialModel {
    let action = |label: &str,
                  role: ActionRole,
                  cost: f64,
                  resource: f64,
                  transition: Vec<f64>,
                  channel: Option<ObservationChannel>,
                  immediate: [f64; 2],
                  delayed: Option<DelayedOutcome>| ActionSpec {
        label: label.into(),
        role,
        utility_cost: cost,
        resource_use: resource,
        transition,
        channel,
        immediate_utility: immediate.to_vec(),
        delayed,
    };
    FiniteSequentialModel {
        label: "inspection".into(),
        population: "plant_a".into(),
        regime: "operations".into(),
        period: "2026".into(),
        state_variable: "unit_condition".into(),
        states: vec!["sound".into(), "faulty".into()],
        initial: vec![0.5, 0.5],
        horizon: 3,
        actions: vec![
            action(
                "wait",
                ActionRole::Continue,
                0.0,
                0.0,
                vec![0.75, 0.25, 0.0, 1.0],
                None,
                [0.0, 0.0],
                None,
            ),
            action(
                "inspect",
                ActionRole::Continue,
                inspect_cost,
                1.0,
                identity(),
                Some(ObservationChannel { measurement: alarm_panel() }),
                [0.0, 0.0],
                None,
            ),
            action("ship", ActionRole::Commit, 0.0, 0.0, identity(), None, [10.0, -10.0], None),
            action(
                "repair",
                ActionRole::Commit,
                4.0,
                0.0,
                vec![1.0, 0.0, 0.5, 0.5],
                None,
                [0.0, 0.0],
                Some(DelayedOutcome { delay: 1, utility: vec![10.0, -10.0] }),
            ),
            action(
                "pilot",
                ActionRole::Continue,
                0.25,
                0.0,
                identity(),
                None,
                [0.0, 0.0],
                Some(DelayedOutcome { delay: 2, utility: vec![3.0, -1.0] }),
            ),
        ],
        terminal_utility: vec![2.0, -2.0],
        censoring: CensoringRule::Refuse,
        resource_budget: Some(2.0),
        stopping: Vec::new(),
        support: ActionSupport::everywhere(5),
    }
}

fn key(history: &[(usize, usize)]) -> HistoryKey {
    HistoryKey::from_steps(
        history.iter().map(|&(action, observation)| HistoryStep { action, observation }),
    )
}

fn policy(version: &str, entries: &[(&[(usize, usize)], usize)]) -> ContingentPolicy {
    let table: Table =
        entries.iter().map(|(history, action)| (history.to_vec(), vec![(*action, 1.0)])).collect();
    let mut policy = ContingentPolicy::new(version);
    for (history, distribution) in table {
        policy = policy.decide_stochastic(key(&history), distribution).unwrap();
    }
    policy
}

fn spec(label: &str, kind: ComparisonKind, policy: ContingentPolicy) -> ComparisonSpec {
    ComparisonSpec { label: label.into(), kind, policy: PolicySpec::Contingent(policy) }
}

#[test]
fn a_comparison_ranks_supplied_policies_by_their_exact_values_without_a_collection_check() {
    let model = model(0.25);
    let report = compare_policies(
        &model,
        vec![
            spec("ship_now", ComparisonKind::ActNow, policy("v_ship", &[(&[], SHIP)])),
            spec(
                "inspect_then_choose",
                ComparisonKind::ChooseAfterObservation,
                policy(
                    "v_choose",
                    &[
                        (&[], INSPECT),
                        (&[(INSPECT, ALARM)], REPAIR),
                        (&[(INSPECT, NEGATIVE)], SHIP),
                    ],
                ),
            ),
            spec("repair_now", ComparisonKind::ActNow, policy("v_repair", &[(&[], REPAIR)])),
        ],
        None,
        &SequentialRequest::default(),
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    let values: Vec<f64> = report.rows.iter().map(|r| r.value).collect();
    for (got, want) in values.iter().zip([0.0, 1.5, 1.0]) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
    assert_eq!(report.best, 1);
    assert!((report.rows[1].difference_from_first - 1.5).abs() < 1e-12);
    assert!((report.rows[2].difference_from_first - 1.0).abs() < 1e-12);
    assert_eq!(report.inference, "point_only");
    // No collection request, so no value-of-information cross-check is made.
    assert!(report.value_of_information.is_none());
}

#[test]
fn the_value_of_collecting_evidence_is_the_gain_over_the_best_commit_less_the_cost() {
    let request = VoiRequest { collect_action: INSPECT, commit_actions: vec![SHIP, REPAIR] };
    // Cost 1/4: prior value 1 (repair), posterior value 7/4, EVSI 3/4, net 1/2.
    let cheap =
        value_of_collecting_evidence(&model(0.25), &request, EvaluationLimits::default(), &ctx())
            .unwrap();
    assert!((cheap.prior_value - 1.0).abs() < 1e-12);
    assert!((cheap.expected_value_of_sample_information - 0.75).abs() < 1e-12);
    assert!((cheap.net_value_of_collecting - 0.5).abs() < 1e-12);
    assert_eq!(cheap.ranking_evaluation, "exact");
    // The cost moves the net and leaves the information value alone: at
    // cost 3/4 the net is exactly zero.
    let dear =
        value_of_collecting_evidence(&model(0.75), &request, EvaluationLimits::default(), &ctx())
            .unwrap();
    assert!((dear.expected_value_of_sample_information - 0.75).abs() < 1e-12);
    assert!(dear.net_value_of_collecting.abs() < 1e-12);
    // An action with its own utility is not pure information.
    let err = value_of_collecting_evidence(
        &model(0.25),
        &VoiRequest { collect_action: PILOT, commit_actions: vec![SHIP, REPAIR] },
        EvaluationLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.detail, "sequential_evaluation.voi_collect_action");
}
