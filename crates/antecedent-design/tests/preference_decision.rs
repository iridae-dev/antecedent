//! Exact verdicts across a preference set: hand-derived switch points, every preference
//! policy, measures, hard constraints, mean sources, degenerate sets and refusals.
//!
//! The standard law has `E[benefit, cost] = (10, 4)` under `A` and `(5.5, 1)` under `B`, and
//! the value is `benefit - w * cost`, so `v_A = 10 - 4w`, `v_B = 5.5 - w` and the switch point is
//! `w* = (c_B0 - c_A0) / (c_A1 - c_B1) = (5.5 - 10) / (-4 + 1) = 1.5`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::reason_details;
use antecedent_design::decision_contract::{
    ActionKind, DecisionCriterion, HardConstraint, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{MeanEstimate, MeanSource};
use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferencePolicy,
};
use antecedent_design::preference_decision::{
    CertifiedRouteReason, MeasureOrigin, PreferenceDecisionError as E, PreferenceDecisionOptions,
    PreferenceDecisionResult, PreferenceMeasure, PreferenceVerdict, evaluate_preference_contract,
    evaluate_preference_contract_on_means,
};
use antecedent_design::preference_model::{
    Attribute, LinearRelation, PreferenceExpr as P, PreferenceParameter, PreferenceSet,
    RelationOrigin, RelationSense, ValueModel, shape,
};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

fn quantity(variable: &str, regime: &str, units: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        units: units.into(),
        regime_id: regime.into(),
        functional_id: functional.into(),
        ..common::quantity(variable)
    }
}

fn regime(action: usize) -> String {
    format!("do(a={action})")
}

fn parameter(id: &str, lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: id.into(),
        units: "events/usd".into(),
        lower,
        upper,
        assumption_id: format!("{id}-judgment"),
    }
}

fn attributes() -> Vec<Attribute> {
    vec![
        Attribute { id: "benefit".into(), units: "events".into() },
        Attribute { id: "cost".into(), units: "usd".into() },
    ]
}

fn model_with(parameters: Vec<PreferenceParameter>, value: P) -> ValueModel {
    ValueModel {
        attributes: attributes(),
        preferences: PreferenceSet::new(parameters, vec![]),
        value,
        utility_units: "events".into(),
    }
}

/// `benefit - w * cost`, `w` in `[lower, upper]`.
fn model(lower: f64, upper: f64) -> ValueModel {
    model_with(
        vec![parameter("w", lower, upper)],
        P::difference(shape::rate(0), P::product(P::Parameter(0), shape::rate(1))),
    )
}

/// Actions in the order given; action `a` reads every attribute under regime `do(a=a)`.
fn contract_for(
    model: ValueModel,
    ids: &[&str],
    functional: &str,
    policy: PreferencePolicy,
) -> PreferenceContract {
    let actions: Vec<PreferenceAction> = ids
        .iter()
        .enumerate()
        .map(|(a, id)| PreferenceAction {
            id: (*id).into(),
            kind: ActionKind::Intervention,
            inputs: model
                .attributes
                .iter()
                .map(|attribute| {
                    ActionInput::Quantity(quantity(
                        &attribute.id,
                        &regime(a),
                        &attribute.units,
                        functional,
                    ))
                })
                .collect(),
        })
        .collect();
    PreferenceContract {
        model,
        actions,
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: policy,
    }
}

/// Joint draws: `arms[a][k]` holds the draws of attribute `k` under action `a`.
fn law(
    model: &ValueModel,
    arms: &[Vec<Vec<f64>>],
    weights: Option<Vec<f64>>,
) -> DistributionArtifact {
    let mut columns = Vec::new();
    for a in 0..arms.len() {
        for attribute in &model.attributes {
            columns.push(quantity(&attribute.id, &regime(a), &attribute.units, "outcome"));
        }
    }
    let n = arms[0][0].len();
    let mut draws = Vec::new();
    for i in 0..n {
        for arm in arms {
            for attribute in arm {
                draws.push(attribute[i]);
            }
        }
    }
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration-1".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [n, columns.len()],
            weights,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

/// `A` and `B` on three weighted draws; `E[benefit, cost]` is `(10, 4)` and `(5.5, 1)`.
fn two_arms() -> Vec<Vec<Vec<f64>>> {
    vec![
        vec![vec![8.0, 12.0, 10.0], vec![3.0, 5.0, 4.0]],
        vec![vec![9.0, 5.0, 4.0], vec![1.0, 1.0, 1.0]],
    ]
}

/// A third action `C` with `v_C = 4 - 2w`, below `B` for every `w >= 0`.
fn three_arms() -> Vec<Vec<Vec<f64>>> {
    let mut arms = two_arms();
    arms.push(vec![vec![4.0, 4.0, 4.0], vec![2.0, 2.0, 2.0]]);
    arms
}

fn weights() -> Vec<f64> {
    vec![1.0, 1.0, 2.0]
}

fn options() -> PreferenceDecisionOptions {
    PreferenceDecisionOptions::default()
}

fn evaluate(
    contract: &PreferenceContract,
    arms: &[Vec<Vec<f64>>],
    measure: Option<&PreferenceMeasure>,
) -> Result<PreferenceDecisionResult, E> {
    let source = law(&contract.model, arms, Some(weights()));
    evaluate_preference_contract(contract, &source, measure, options())
}

fn run(lower: f64, upper: f64, policy: PreferencePolicy) -> PreferenceDecisionResult {
    evaluate(&contract_for(model(lower, upper), &["A", "B"], "outcome", policy), &two_arms(), None)
        .unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

fn close_pair(got: Option<(f64, f64)>, want: (f64, f64)) -> bool {
    got.is_some_and(|(lo, hi)| close(lo, want.0) && close(hi, want.1))
}

fn declared_measure(points: Vec<Vec<f64>>, weights: Option<Vec<f64>>) -> PreferenceMeasure {
    PreferenceMeasure {
        points,
        weights,
        labels: None,
        origin: MeasureOrigin::Declared { assumption_id: "panel-weights".into() },
        unresolved: vec![],
    }
}

#[test]
fn left_of_the_switch_point_the_first_action_is_necessarily_best() {
    let result = run(0.0, 1.0, PreferencePolicy::RequireInvariantBestAction);
    assert_eq!(result.verdict, PreferenceVerdict::NecessarilyBest("A".into()));
    assert_eq!(result.parameter_ids, vec!["w".to_string()]);
    let (a, b) = (&result.actions[0], &result.actions[1]);
    let coefficients = a.coefficients.as_ref().unwrap();
    assert!(close(coefficients[0], 10.0) && close(coefficients[1], -4.0), "{coefficients:?}");
    let coefficients = b.coefficients.as_ref().unwrap();
    assert!(close(coefficients[0], 5.5) && close(coefficients[1], -1.0), "{coefficients:?}");
    assert!(close_pair(a.value_range, (6.0, 10.0)), "{:?}", a.value_range);
    assert!(close_pair(b.value_range, (4.5, 5.5)), "{:?}", b.value_range);
    assert!(a.necessarily_best && !b.necessarily_best);
    assert!(a.uniquely_best && !b.uniquely_best);
    assert_eq!(b.dominated_by, vec!["A".to_string()]);
    assert!(a.dominated_by.is_empty());
    assert_eq!((a.possibly_best, b.possibly_best), (Some(true), Some(false)));
    // Regret of B is the largest margin of A over it: 4.5 - 3w at w = 0.
    assert!(close(a.worst_case_regret.unwrap(), 0.0));
    assert!(close(b.worst_case_regret.unwrap(), 4.5));
    assert!(result.switches.is_empty());
    // Per draw at the vertices w = 0 and w = 1: the first draw (weight 1/4) has
    // A = 8 - 3w < B = 9 - w at w = 0; the other two draws keep A ahead at both vertices.
    assert!(close(result.draw_share_best_everywhere.unwrap(), 0.75));
    assert_eq!(result.n_draws, 3);
    assert_eq!(result.source.provider_id, "exact-law");
    assert_eq!(result.assumptions, vec!["w-judgment".to_string()]);
    assert_eq!(result.unresolved_measure_share, None);
    assert!(result.actions.iter().all(|o| o.measure_share.is_none()));
}

#[test]
fn right_of_the_switch_point_the_second_action_is_necessarily_best() {
    let result = run(2.0, 3.0, PreferencePolicy::RequireInvariantBestAction);
    assert_eq!(result.verdict, PreferenceVerdict::NecessarilyBest("B".into()));
    assert!(close(result.actions[0].worst_case_regret.unwrap(), 4.5));
    assert!(result.switches.is_empty());
    // Every draw keeps B at least as good at w = 2 and w = 3 (the third ties at w = 2).
    assert!(close(result.draw_share_best_everywhere.unwrap(), 1.0));
}

#[test]
fn around_the_switch_point_the_choice_depends_on_the_preference() {
    let result = run(0.0, 3.0, PreferencePolicy::RequireInvariantBestAction);
    assert_eq!(
        result.verdict,
        PreferenceVerdict::PreferenceDependent { possibly_best: vec!["A".into(), "B".into()] }
    );
    assert_eq!(result.switches.len(), 1);
    let switch = &result.switches[0];
    assert_eq!((switch.first.as_str(), switch.second.as_str()), ("A", "B"));
    assert!(close(switch.margin_coefficients[0], 4.5));
    assert!(close(switch.margin_coefficients[1], -3.0));
    assert!(close(switch.margin_range.0, -4.5) && close(switch.margin_range.1, 4.5));
    // Witnesses lie on the right sides of w* = 1.5.
    assert!(switch.favours_first_at[0] < 1.5 && close(switch.favours_first_at[0], 0.0));
    assert!(switch.favours_second_at[0] > 1.5 && close(switch.favours_second_at[0], 3.0));
    assert!(result.actions.iter().all(|o| !o.necessarily_best));
    assert!(result.actions.iter().all(|o| o.dominated_by.is_empty()));
    assert_eq!(result.draw_share_best_everywhere, None);
    // The margin 4.5 - 3w depends on w alone and is zero at w = 1.5; A leads below it.
    let threshold = switch.threshold.as_ref().unwrap();
    assert_eq!(threshold.parameter_id, "w");
    assert!(close(threshold.value, 1.5));
    assert!(!threshold.first_leads_above);
    assert_eq!(result.parameter_ranges.len(), 1);
    assert!(close_pair(Some(result.parameter_ranges[0]), (0.0, 3.0)));
    assert_eq!(
        result.statement(),
        "The preferred action depends on the declared trade-offs: A leads when w is below 1.5 \
         and B leads above it; the admissible range of w is 0 to 3."
    );
}

#[test]
fn a_dominated_action_is_never_possibly_best() {
    let c = contract_for(
        model(0.0, 3.0),
        &["A", "B", "C"],
        "outcome",
        PreferencePolicy::RequireInvariantBestAction,
    );
    let result = evaluate(&c, &three_arms(), None).unwrap();
    assert_eq!(result.actions[2].possibly_best, Some(false));
    assert_eq!(
        result.verdict,
        PreferenceVerdict::PreferenceDependent { possibly_best: vec!["A".into(), "B".into()] }
    );
    // Only A and B change order; A - C = 6 - 2w only touches zero at w = 3.
    let pairs: Vec<(&str, &str)> =
        result.switches.iter().map(|s| (s.first.as_str(), s.second.as_str())).collect();
    assert_eq!(pairs, vec![("A", "B")]);
    // C's regret is the largest of A - C = 6 - 2w and B - C = 1.5 + w: 6 at w = 0.
    assert!(close(result.actions[2].worst_case_regret.unwrap(), 6.0));
}

#[test]
fn a_relation_that_cuts_the_range_removes_the_switch() {
    let mut m = model(0.0, 3.0);
    m.preferences.relations.push(LinearRelation {
        id: "cap".into(),
        terms: vec![(0, 1.0)],
        sense: RelationSense::AtMost,
        bound: 1.0,
        origin: RelationOrigin::Declared,
    });
    let c = contract_for(m, &["A", "B"], "outcome", PreferencePolicy::RequireInvariantBestAction);
    let result = evaluate(&c, &two_arms(), None).unwrap();
    assert_eq!(result.verdict, PreferenceVerdict::NecessarilyBest("A".into()));
    assert!(result.switches.is_empty());
    assert_eq!(result.parameter_ranges.len(), 1);
    assert!(close_pair(Some(result.parameter_ranges[0]), (0.0, 1.0)));
    assert!(close_pair(result.actions[0].value_range, (6.0, 10.0)));
    assert_eq!(
        result.assumptions,
        vec!["declared relation cap".to_string(), "w-judgment".to_string()]
    );
}

#[test]
fn relations_that_contradict_the_ranges_leave_an_empty_set() {
    let mut m = model(0.0, 3.0);
    m.preferences.relations.push(LinearRelation {
        id: "impossible".into(),
        terms: vec![(0, 1.0)],
        sense: RelationSense::AtLeast,
        bound: 5.0,
        origin: RelationOrigin::Declared,
    });
    let set = m.preferences.clone();
    let c = contract_for(m, &["A", "B"], "outcome", PreferencePolicy::RequireInvariantBestAction);
    let refused = evaluate(&c, &two_arms(), None).unwrap_err();
    assert_eq!(refused, E::EmptyPreferenceSet);
    assert_eq!(refused.detail(), reason_details::preference_decision::PREFERENCE_SET_EMPTY);
    assert_eq!(declared_measure(vec![vec![1.0]], None).validate(&set), Err(E::EmptyPreferenceSet));
}

#[test]
fn maximin_and_minimax_regret_match_hand_values() {
    // On [0, 2]: A in [2, 10], B in [3.5, 5.5]; regret of A is 3w - 4.5 at w = 2 (1.5),
    // regret of B is 4.5 - 3w at w = 0 (4.5).
    let maximin = run(0.0, 2.0, PreferencePolicy::Maximin);
    assert_eq!(maximin.verdict, PreferenceVerdict::WorstCaseChoice("B".into()));
    assert!(close_pair(maximin.actions[0].value_range, (2.0, 10.0)));
    assert!(close_pair(maximin.actions[1].value_range, (3.5, 5.5)));
    let regret = run(0.0, 2.0, PreferencePolicy::MinimaxRegret);
    assert_eq!(regret.verdict, PreferenceVerdict::MinimaxRegretChoice("A".into()));
    assert!(close(regret.actions[0].worst_case_regret.unwrap(), 1.5));
    assert!(close(regret.actions[1].worst_case_regret.unwrap(), 4.5));
    assert_eq!(
        maximin.statement(),
        "Action B has the best worst-case value over the admissible values of the declared \
         trade-offs."
    );
    assert_eq!(
        regret.statement(),
        "Action A has the smallest worst-case regret over the admissible values of the declared \
         trade-offs."
    );
    let report = run(0.0, 2.0, PreferencePolicy::ReportOnly);
    assert_eq!(report.verdict, PreferenceVerdict::ReportOnly);
    assert_eq!(
        report.statement(),
        "No action is chosen: the comparison across the declared trade-offs is reported only."
    );
    assert_eq!(report.switches, regret.switches);
    assert_eq!(report.actions, regret.actions);
}

#[test]
fn a_weighted_policy_chooses_by_a_three_point_measure() {
    let c = contract_for(model(0.0, 3.0), &["A", "B"], "outcome", PreferencePolicy::Weighted);
    let mut measure =
        declared_measure(vec![vec![0.5], vec![1.0], vec![2.5]], Some(vec![1.0, 1.0, 2.0]));
    measure.labels = Some(vec!["low".into(), "mid".into(), "high".into()]);
    let result = evaluate(&c, &two_arms(), Some(&measure)).unwrap();
    // A leads at w = 0.5 and 1 (weight 1/2), B at w = 2.5 (weight 1/2).
    // A: (8 + 6) / 4 + 0 = 3.5; B: (5 + 4.5) / 4 + 3 / 2 = 3.875.
    assert_eq!(result.verdict, PreferenceVerdict::MeasureChoice { action: "B".into(), share: 0.5 });
    assert_eq!(
        result.statement(),
        "Action B has the best value weighted by the supplied measure over the declared \
         trade-offs; it leads at measure points carrying a weight of 0.5."
    );
    assert!(close(result.actions[0].measure_share.unwrap(), 0.5));
    assert!(close(result.actions[1].measure_share.unwrap(), 0.5));
    assert!(close(result.actions[0].measure_value.unwrap(), 3.5));
    assert!(close(result.actions[1].measure_value.unwrap(), 3.875));
    assert_eq!(result.unresolved_measure_share, Some(0.0));
    assert!(result.assumptions.contains(&"panel-weights".to_string()));

    // The measure is reported whatever the policy.
    let worst_case =
        contract_for(model(0.0, 3.0), &["A", "B"], "outcome", PreferencePolicy::Maximin);
    let reported = evaluate(&worst_case, &two_arms(), Some(&measure)).unwrap();
    assert!(close(reported.actions[1].measure_share.unwrap(), 0.5));
    assert!(matches!(reported.verdict, PreferenceVerdict::WorstCaseChoice(_)));
}

#[test]
fn a_weighted_policy_refuses_without_a_complete_measure() {
    let c = contract_for(model(0.0, 3.0), &["A", "B"], "outcome", PreferencePolicy::Weighted);
    let refused = evaluate(&c, &two_arms(), None).unwrap_err();
    assert_eq!(refused, E::MeasureRequired);
    assert_eq!(refused.detail(), reason_details::preference_decision::MEASURE_REQUIRED);
    let mut partial = declared_measure(vec![vec![1.0]], None);
    partial.unresolved = vec![0];
    let refused = evaluate(&c, &two_arms(), Some(&partial)).unwrap_err();
    assert_eq!(refused, E::MeasureIncomplete { parameters: vec!["w".into()] });
    assert_eq!(refused.detail(), reason_details::preference_decision::MEASURE_INCOMPLETE);
}

#[test]
fn a_point_counts_only_where_its_leader_holds_over_the_unresolved_parameters() {
    // `u * benefit - w * cost`, u in [0.5, 1.5], w in [0, 1]; margin of A over B is
    // 4.5u - 3w. At u = 1 and u = 1.5 A leads for every w; at u = 0.5 the margin spans
    // [-0.75, 2.25], so that point's weight is unresolved.
    let m = model_with(
        vec![parameter("u", 0.5, 1.5), parameter("w", 0.0, 1.0)],
        P::difference(
            P::product(P::Parameter(0), shape::rate(0)),
            P::product(P::Parameter(1), shape::rate(1)),
        ),
    );
    let c = contract_for(m, &["A", "B"], "outcome", PreferencePolicy::RequireInvariantBestAction);
    let mut measure = declared_measure(vec![vec![1.0, 0.0], vec![0.5, 0.0], vec![1.5, 0.0]], None);
    measure.unresolved = vec![1];
    let result = evaluate(&c, &two_arms(), Some(&measure)).unwrap();
    // The margin 4.5u - 3w moves with both parameters: no single threshold.
    assert_eq!(result.switches.len(), 1);
    assert_eq!(result.switches[0].threshold, None);
    assert_eq!(
        result.statement(),
        "The preferred action depends on the declared trade-offs: A and B are each preferred \
         somewhere in the admissible set, and the boundary between them depends on more than one \
         trade-off."
    );
    assert!(close(result.actions[0].measure_share.unwrap(), 2.0 / 3.0));
    assert!(close(result.actions[1].measure_share.unwrap(), 0.0));
    assert!(close(result.unresolved_measure_share.unwrap(), 1.0 / 3.0));
    assert!(result.actions.iter().all(|o| o.measure_value.is_none()));
    assert_eq!(
        result.verdict,
        PreferenceVerdict::PreferenceDependent { possibly_best: vec!["A".into(), "B".into()] }
    );
}

#[test]
fn a_hard_constraint_excludes_an_action_from_every_comparison() {
    let budget = |bound: f64| HardConstraint {
        id: "budget".into(),
        expr: UtilityExpr::Input(1),
        bound,
        min_probability: 1.0,
        units: "usd".into(),
        applies_to: vec![],
    };
    let mut c = contract_for(model(0.0, 1.0), &["A", "B"], "outcome", PreferencePolicy::Maximin);
    // A costs 5 on a draw of positive weight.
    c.constraints = vec![budget(4.5)];
    let result = evaluate(&c, &two_arms(), None).unwrap();
    assert_eq!(result.verdict, PreferenceVerdict::WorstCaseChoice("B".into()));
    let a = &result.actions[0];
    assert!(!a.admissible);
    assert_eq!(a.exclusions.len(), 1);
    assert_eq!(a.exclusions[0].constraint_id, "budget");
    assert_eq!((a.coefficients.as_ref(), a.value_range, a.worst_case_regret), (None, None, None));
    assert_eq!(a.possibly_best, Some(false));
    assert!(result.actions[1].necessarily_best);
    assert!(result.switches.is_empty());
    assert!(close(result.draw_share_best_everywhere.unwrap(), 1.0));

    assert!(result.actions[1].uniquely_best);
    assert!(!a.necessarily_best && !a.uniquely_best && a.dominated_by.is_empty());

    c.constraints = vec![budget(0.5)];
    let result = evaluate(&c, &two_arms(), None).unwrap();
    assert_eq!(result.verdict, PreferenceVerdict::NoAdmissibleAction);
    assert_eq!(
        result.statement(),
        "No action is admissible: every action fails a hard constraint."
    );
}

#[test]
fn a_model_without_parameters_gives_the_ordinary_verdict() {
    let m = model_with(vec![], P::difference(shape::rate(0), shape::rate(1)));
    let c = contract_for(m, &["A", "B"], "outcome", PreferencePolicy::RequireInvariantBestAction);
    let result = evaluate(&c, &two_arms(), None).unwrap();
    assert_eq!(result.verdict, PreferenceVerdict::NecessarilyBest("A".into()));
    assert!(result.parameter_ids.is_empty() && result.parameter_ranges.is_empty());
    assert_eq!(result.statement(), "Action A is preferred under the declared value model.");
    assert!(close_pair(result.actions[0].value_range, (6.0, 6.0)));
    assert!(close_pair(result.actions[1].value_range, (4.5, 4.5)));
    assert_eq!(result.actions[0].coefficients.as_ref().map(Vec::len), Some(1));
    assert!(result.switches.is_empty());
    // Per draw: 5 < 8 on the first draw; 7 > 4 and 6 > 3 on the others.
    assert!(close(result.draw_share_best_everywhere.unwrap(), 0.75));
    assert!(result.assumptions.is_empty());
}

fn mean_source() -> MeanSource {
    let m = model(0.0, 3.0);
    let mut coordinates = Vec::new();
    for a in 0..2 {
        for attribute in &m.attributes {
            coordinates.push(quantity(&attribute.id, &regime(a), &attribute.units, "mean"));
        }
    }
    MeanSource {
        coordinates,
        means: vec![10.0, 4.0, 5.5, 1.0],
        estimate: MeanEstimate::Posterior,
        provider_id: "lab".into(),
        snapshot_id: "snap-1".into(),
        causal_contract_id: "checked-contract".into(),
        rng_id: "none:mean_grid".into(),
    }
}

#[test]
fn a_mean_source_answers_an_affine_value() {
    let c = contract_for(
        model(0.0, 3.0),
        &["A", "B"],
        "mean",
        PreferencePolicy::RequireInvariantBestAction,
    );
    let result =
        evaluate_preference_contract_on_means(&c, &mean_source(), None, options()).unwrap();
    assert_eq!(
        result.verdict,
        PreferenceVerdict::PreferenceDependent { possibly_best: vec!["A".into(), "B".into()] }
    );
    assert!(close(result.switches[0].margin_range.0, -4.5));
    assert!(close(result.switches[0].margin_range.1, 4.5));
    assert_eq!(result.n_draws, 0);
    assert_eq!(result.draw_share_best_everywhere, None);
    assert_eq!(result.source.provider_id, "lab");

    // A part that is not affine in the attributes needs a law, not means.
    let kinked = model_with(
        vec![parameter("w", 0.0, 3.0)],
        P::difference(shape::excess(0, 7.0), P::product(P::Parameter(0), shape::rate(1))),
    );
    let c = contract_for(kinked, &["A", "B"], "mean", PreferencePolicy::ReportOnly);
    let refused = evaluate_preference_contract_on_means(&c, &mean_source(), None, options());
    let Err(E::Eval(inner)) = &refused else { panic!("{refused:?}") };
    assert_eq!(refused.as_ref().unwrap_err().detail(), inner.to_refusal().detail);

    // A hard constraint is refused as the mean evaluator refuses it.
    let mut c = contract_for(model(0.0, 3.0), &["A", "B"], "mean", PreferencePolicy::ReportOnly);
    c.constraints = vec![HardConstraint {
        id: "budget".into(),
        expr: UtilityExpr::Input(1),
        bound: 4.5,
        min_probability: 1.0,
        units: "usd".into(),
        applies_to: vec![],
    }];
    let refused = evaluate_preference_contract_on_means(&c, &mean_source(), None, options());
    assert!(matches!(refused, Err(E::Eval(_))), "{refused:?}");
}

#[test]
fn an_expected_loss_is_read_smaller_is_better() {
    let mut c = contract_for(
        model(0.0, 1.0),
        &["A", "B"],
        "outcome",
        PreferencePolicy::RequireInvariantBestAction,
    );
    c.criterion = DecisionCriterion::PosteriorExpectedLoss;
    let result = evaluate(&c, &two_arms(), None).unwrap();
    assert_eq!(result.verdict, PreferenceVerdict::NecessarilyBest("B".into()));
    // Ranges stay in loss units; coefficients are in larger-is-better orientation.
    assert!(close_pair(result.actions[0].value_range, (6.0, 10.0)));
    let coefficients = result.actions[0].coefficients.as_ref().unwrap();
    assert!(close(coefficients[0], -10.0) && close(coefficients[1], 4.0));
    assert!(close(result.actions[0].worst_case_regret.unwrap(), 4.5));
    assert!(close(result.actions[1].worst_case_regret.unwrap(), 0.0));
}

#[test]
fn expected_regret_gives_the_expected_utility_verdict_without_amounts() {
    for (lower, upper) in [(0.0, 1.0), (2.0, 3.0), (0.0, 3.0)] {
        let utility = run(lower, upper, PreferencePolicy::RequireInvariantBestAction);
        let mut c = contract_for(
            model(lower, upper),
            &["A", "B"],
            "outcome",
            PreferencePolicy::RequireInvariantBestAction,
        );
        c.criterion = DecisionCriterion::ExpectedRegret;
        let regret = evaluate(&c, &two_arms(), None).unwrap();
        assert_eq!(regret.verdict, utility.verdict);
        assert_eq!(regret.switches, utility.switches);
        assert!(regret.actions.iter().all(|o| o.value_range.is_none()));
    }
    let mut c = contract_for(model(0.0, 3.0), &["A", "B"], "outcome", PreferencePolicy::Maximin);
    c.criterion = DecisionCriterion::ExpectedRegret;
    let refused = evaluate(&c, &two_arms(), None).unwrap_err();
    assert_eq!(
        refused,
        E::CertifiedRouteRequired { reason: CertifiedRouteReason::WorstCaseOfExpectedRegret }
    );
    assert_eq!(
        refused.detail(),
        reason_details::preference_decision::WORST_CASE_OF_EXPECTED_REGRET
    );
}

#[test]
fn values_the_exact_route_cannot_answer_need_the_certified_route() {
    let mut c = contract_for(model(0.0, 3.0), &["A", "B"], "outcome", PreferencePolicy::ReportOnly);
    c.criterion = DecisionCriterion::Quantile { p: 0.5 };
    let refused = evaluate(&c, &two_arms(), None).unwrap_err();
    assert_eq!(
        refused,
        E::CertifiedRouteRequired {
            reason: CertifiedRouteReason::Criterion(DecisionCriterion::Quantile { p: 0.5 })
        }
    );
    assert_eq!(
        refused.detail(),
        reason_details::preference_decision::CRITERION_NEEDS_CERTIFIED_ROUTE
    );
    c.criterion = DecisionCriterion::ThresholdProbability { threshold: 1.0 };
    assert!(matches!(
        evaluate(&c, &two_arms(), None),
        Err(E::CertifiedRouteRequired { reason: CertifiedRouteReason::Criterion(_) })
    ));

    let shaped = model_with(
        vec![parameter("w", 0.0, 1.0)],
        P::difference(P::product(P::exp(P::Parameter(0)), shape::rate(0)), shape::rate(1)),
    );
    let c = contract_for(shaped, &["A", "B"], "outcome", PreferencePolicy::ReportOnly);
    let refused = evaluate(&c, &two_arms(), None).unwrap_err();
    assert_eq!(
        refused,
        E::CertifiedRouteRequired {
            reason: CertifiedRouteReason::ShapeParameters { parameters: vec!["w".into()] }
        }
    );
    assert_eq!(
        refused.detail(),
        reason_details::preference_decision::SHAPE_PARAMETERS_NEED_CERTIFIED_ROUTE
    );

    let curved = model_with(
        vec![parameter("w", 0.0, 1.0)],
        P::difference(P::product(P::Parameter(0), P::exp(shape::rate(0))), shape::rate(1)),
    );
    let c = contract_for(curved, &["A", "B"], "outcome", PreferencePolicy::ReportOnly);
    let refused = evaluate(&c, &two_arms(), None).unwrap_err();
    assert_eq!(
        refused,
        E::CertifiedRouteRequired { reason: CertifiedRouteReason::PartNotLowerable }
    );
    assert_eq!(refused.detail(), reason_details::preference_decision::PART_NOT_LOWERABLE);
}

#[test]
fn actions_equal_everywhere_are_tied() {
    let mut arms = three_arms();
    arms[1] = arms[0].clone();
    let c = contract_for(
        model(0.0, 1.0),
        &["A", "B", "C"],
        "outcome",
        PreferencePolicy::RequireInvariantBestAction,
    );
    let result = evaluate(&c, &arms, None).unwrap();
    assert_eq!(result.verdict, PreferenceVerdict::Tied(vec!["A".into(), "B".into()]));
    // A and B are each never beaten (each ties the other everywhere); none is unique.
    let never: Vec<bool> = result.actions.iter().map(|o| o.necessarily_best).collect();
    assert_eq!(never, vec![true, true, false]);
    assert!(result.actions.iter().all(|o| !o.uniquely_best));
    // A - C = 6 - 2w and B - C likewise lie in [4, 6]: C is dominated by both.
    assert_eq!(result.actions[2].dominated_by, vec!["A".to_string(), "B".to_string()]);
    assert!(result.switches.is_empty());
    assert_eq!(result.draw_share_best_everywhere, None);
    assert_eq!(
        result.statement(),
        "Actions A and B are equal for every admissible value of the declared trade-offs; no \
         single action is chosen."
    );
}

#[test]
fn the_result_does_not_depend_on_declaration_order() {
    let forward = contract_for(
        model(0.0, 3.0),
        &["A", "B", "C"],
        "outcome",
        PreferencePolicy::RequireInvariantBestAction,
    );
    let mut backward = forward.clone();
    backward.actions.reverse();
    let source = law(&forward.model, &three_arms(), Some(weights()));
    let one = evaluate_preference_contract(&forward, &source, None, options()).unwrap();
    let two = evaluate_preference_contract(&backward, &source, None, options()).unwrap();
    // Results are reported in declaration order, so the order is part of the contract identity
    // (deliberately bound); every verdict below is unchanged by it.
    assert_ne!(one.contract_identity, two.contract_identity);
    for outcome in &one.actions {
        let other = two.actions.iter().find(|o| o.id == outcome.id).unwrap();
        assert_eq!(outcome.possibly_best, other.possibly_best);
        assert_eq!(outcome.necessarily_best, other.necessarily_best);
        assert_eq!(outcome.uniquely_best, other.uniquely_best);
        let mut dominated = other.dominated_by.clone();
        dominated.reverse();
        assert_eq!(outcome.dominated_by, dominated, "declaration order of the dominating rivals");
        let (a, b) = (outcome.value_range.unwrap(), other.value_range.unwrap());
        assert!(close(a.0, b.0) && close(a.1, b.1));
        assert!(close(outcome.worst_case_regret.unwrap(), other.worst_case_regret.unwrap()));
    }
    assert_eq!(
        two.verdict,
        PreferenceVerdict::PreferenceDependent { possibly_best: vec!["B".into(), "A".into()] }
    );
    let switch = &two.switches[0];
    assert_eq!((switch.first.as_str(), switch.second.as_str()), ("B", "A"));
    assert!(close(switch.margin_range.0, -4.5) && close(switch.favours_first_at[0], 3.0));
    // Swapping the pair flips which action leads above the threshold, not where it lies.
    let (ahead, behind) =
        (one.switches[0].threshold.as_ref().unwrap(), switch.threshold.as_ref().unwrap());
    assert!(close(ahead.value, behind.value));
    assert_ne!(ahead.first_leads_above, behind.first_leads_above);
    assert_eq!(one.statement(), two.statement());
    // Repeated evaluation is identical.
    assert_eq!(evaluate_preference_contract(&forward, &source, None, options()).unwrap(), one);
}

#[test]
#[allow(clippy::too_many_lines, reason = "one table listing every measure refusal")]
fn every_measure_refusal_names_its_cause() {
    let set = model(0.0, 3.0).preferences;
    let base = || declared_measure(vec![vec![1.0], vec![2.0]], Some(vec![1.0, 3.0]));
    assert_eq!(base().validate(&set), Ok(()));
    #[allow(clippy::type_complexity, reason = "a table of edits and their refusals")]
    let cases: Vec<(Box<dyn Fn(&mut PreferenceMeasure)>, E)> = vec![
        (Box::new(|m| m.points.clear()), E::MeasureEmpty),
        (
            Box::new(|m| m.points[1] = vec![1.0, 2.0]),
            E::MeasurePointDimension { index: 1, expected: 1, found: 2 },
        ),
        (Box::new(|m| m.points[0] = vec![f64::NAN]), E::MeasurePointNonFinite { index: 0 }),
        (Box::new(|m| m.points[1] = vec![3.5]), E::MeasurePointOutsideSet { index: 1 }),
        (
            Box::new(|m| m.weights = Some(vec![1.0])),
            E::MeasureWeightCount { expected: 2, found: 1 },
        ),
        (Box::new(|m| m.weights = Some(vec![1.0, -1.0])), E::MeasureWeightInvalid { index: 1 }),
        (Box::new(|m| m.weights = Some(vec![0.0, 0.0])), E::MeasureWeightTotalZero),
        (
            Box::new(|m| m.labels = Some(vec!["one".into()])),
            E::MeasureLabelCount { expected: 2, found: 1 },
        ),
        (
            Box::new(|m| m.labels = Some(vec!["one".into(), " ".into()])),
            E::MeasureLabelBlank { index: 1 },
        ),
        (
            Box::new(|m| m.labels = Some(vec!["one".into(), "one".into()])),
            E::MeasureLabelDuplicate { label: "one".into() },
        ),
        (
            Box::new(|m| {
                m.origin = MeasureOrigin::ChoiceEstimate { estimate_identity: String::new() }
            }),
            E::MeasureOriginBlank,
        ),
        (Box::new(|m| m.unresolved = vec![1]), E::MeasureUnresolvedUnknown { index: 1 }),
        (Box::new(|m| m.unresolved = vec![0, 0]), E::MeasureUnresolvedRepeated { index: 0 }),
    ];
    let mut details = std::collections::BTreeSet::new();
    for (edit, expected) in cases {
        let mut measure = base();
        edit(&mut measure);
        let refused = measure.validate(&set).unwrap_err();
        assert_eq!(refused, expected);
        details.insert(refused.detail());
        // The evaluation refuses the same measure the same way.
        let c = contract_for(model(0.0, 3.0), &["A", "B"], "outcome", PreferencePolicy::ReportOnly);
        assert_eq!(evaluate(&c, &two_arms(), Some(&measure)), Err(expected));
    }
    assert_eq!(details.len(), 13);
    assert!(details.iter().all(|d| d.starts_with("preference_decision.measure_")));
    // A point with an unresolved coordinate is inside when some admissible value fits.
    let mut free = base();
    free.points[1] = vec![99.0];
    free.unresolved = vec![0];
    assert_eq!(free.validate(&set), Ok(()));
}

#[test]
fn invalid_options_and_contracts_refuse() {
    let c = contract_for(model(0.0, 3.0), &["A", "B"], "outcome", PreferencePolicy::ReportOnly);
    let source = law(&c.model, &two_arms(), Some(weights()));
    let bad = PreferenceDecisionOptions { tie_tolerance: -1.0, ..options() };
    let refused = evaluate_preference_contract(&c, &source, None, bad).unwrap_err();
    assert_eq!(refused, E::InvalidOptions { option: "tie_tolerance" });
    assert_eq!(refused.detail(), reason_details::preference_decision::OPTIONS_INVALID);
    let mut invalid = c.clone();
    invalid.actions.truncate(1);
    let refused = evaluate_preference_contract(&invalid, &source, None, options()).unwrap_err();
    assert!(matches!(refused, E::Contract(_)));
    assert_eq!(refused.detail(), reason_details::preference_contract::TOO_FEW_ACTIONS);
    // A source that lacks B's coordinates is surfaced as a source refusal.
    let other = law(&c.model, &two_arms()[..1], Some(weights()));
    let refused = evaluate_preference_contract(&c, &other, None, options()).unwrap_err();
    assert!(matches!(refused, E::Eval(_)), "{refused:?}");
}

#[test]
fn an_action_tied_at_the_edge_of_the_set_is_never_beaten_but_not_unique() {
    // On [0, 1.5] the margin of A over B, 4.5 - 3w, lies in [0, 4.5]: zero at w = 1.5 only.
    let c = contract_for(
        model(0.0, 1.5),
        &["A", "B", "C"],
        "outcome",
        PreferencePolicy::RequireInvariantBestAction,
    );
    let result = evaluate(&c, &three_arms(), None).unwrap();
    assert_eq!(
        result.verdict,
        PreferenceVerdict::NeverBeaten { action: "A".into(), ties_with: vec!["B".into()] }
    );
    let (a, b, c_outcome) = (&result.actions[0], &result.actions[1], &result.actions[2]);
    assert!(a.necessarily_best && !a.uniquely_best);
    assert!(!b.necessarily_best && !b.uniquely_best);
    // A ties B at w = 1.5, so it does not dominate B; C trails both everywhere
    // (A - C = 6 - 2w in [3, 6], B - C = 1.5 + w in [1.5, 3]).
    assert!(b.dominated_by.is_empty());
    assert_eq!(c_outcome.dominated_by, vec!["A".to_string(), "B".to_string()]);
    assert_eq!(b.possibly_best, Some(true));
    // A margin range that only touches zero is not a switch.
    assert!(result.switches.is_empty());
    // At the vertices w = 0 and w = 1.5 the first draw (weight 1/4) has A = 8 < B = 9 at
    // w = 0; the other draws keep A at least as good at both vertices.
    assert!(close(result.draw_share_best_everywhere.unwrap(), 0.75));
    assert_eq!(
        result.statement(),
        "Action A is never beaten for any admissible value of the declared trade-offs, and ties \
         with B at the edge of the admissible set."
    );
}

#[test]
fn an_action_best_nowhere_need_not_be_dominated_by_any_single_rival() {
    // v_C = 7 - 2.5w on [0, 3]: A - C = 3 - 1.5w and B - C = 1.5w - 1.5 each change sign, but
    // C would need w >= 2 to reach A and w <= 1 to reach B, so it is best nowhere.
    let mut arms = two_arms();
    arms.push(vec![vec![7.0, 7.0, 7.0], vec![2.5, 2.5, 2.5]]);
    let c = contract_for(
        model(0.0, 3.0),
        &["A", "B", "C"],
        "outcome",
        PreferencePolicy::RequireInvariantBestAction,
    );
    let result = evaluate(&c, &arms, None).unwrap();
    let c_outcome = &result.actions[2];
    assert_eq!(c_outcome.possibly_best, Some(false));
    assert!(c_outcome.dominated_by.is_empty());
    assert!(!c_outcome.necessarily_best);
    assert_eq!(
        result.verdict,
        PreferenceVerdict::PreferenceDependent { possibly_best: vec!["A".into(), "B".into()] }
    );
    let pairs: Vec<(&str, &str)> =
        result.switches.iter().map(|s| (s.first.as_str(), s.second.as_str())).collect();
    assert_eq!(pairs, vec![("A", "B"), ("A", "C"), ("B", "C")]);
    let thresholds: Vec<f64> =
        result.switches.iter().map(|s| s.threshold.as_ref().unwrap().value).collect();
    assert!(close(thresholds[0], 1.5) && close(thresholds[1], 2.0) && close(thresholds[2], 1.0));
    // Only the switch between the possibly-best actions describes the choice.
    assert_eq!(
        result.statement(),
        "The preferred action depends on the declared trade-offs: A leads when w is below 1.5 \
         and B leads above it; the admissible range of w is 0 to 3."
    );
}

#[test]
fn a_fixed_parameter_does_not_stop_a_single_threshold() {
    // `u * benefit - w * cost` with u fixed at 1: the margin 4.5u - 3w depends on w alone.
    let m = model_with(
        vec![parameter("u", 1.0, 1.0), parameter("w", 0.0, 3.0)],
        P::difference(
            P::product(P::Parameter(0), shape::rate(0)),
            P::product(P::Parameter(1), shape::rate(1)),
        ),
    );
    let c = contract_for(m, &["A", "B"], "outcome", PreferencePolicy::RequireInvariantBestAction);
    let result = evaluate(&c, &two_arms(), None).unwrap();
    let threshold = result.switches[0].threshold.as_ref().unwrap();
    assert_eq!(threshold.parameter_id, "w");
    assert!(close(threshold.value, 1.5) && !threshold.first_leads_above);
    assert!(close_pair(Some(result.parameter_ranges[0]), (1.0, 1.0)));
    assert!(close_pair(Some(result.parameter_ranges[1]), (0.0, 3.0)));
}

#[test]
fn the_statement_quotes_the_range_the_relations_leave() {
    // A relation `w <= 2` cuts the declared [0, 3]; the switch at 1.5 remains.
    let mut m = model(0.0, 3.0);
    m.preferences.relations.push(LinearRelation {
        id: "cap".into(),
        terms: vec![(0, 1.0)],
        sense: RelationSense::AtMost,
        bound: 2.0,
        origin: RelationOrigin::Choice { record_id: "r1".into(), rejected_option: "b".into() },
    });
    let c = contract_for(m, &["A", "B"], "outcome", PreferencePolicy::RequireInvariantBestAction);
    let result = evaluate(&c, &two_arms(), None).unwrap();
    assert!(close_pair(Some(result.parameter_ranges[0]), (0.0, 2.0)));
    assert_eq!(
        result.statement(),
        "The preferred action depends on the declared trade-offs: A leads when w is below 1.5 \
         and B leads above it; the admissible range of w is 0 to 2."
    );
}

#[test]
fn statements_never_call_a_range_or_share_a_probability() {
    for (lower, upper) in [(0.0, 1.0), (0.0, 1.5), (0.0, 3.0), (2.0, 3.0)] {
        for policy in [
            PreferencePolicy::RequireInvariantBestAction,
            PreferencePolicy::Maximin,
            PreferencePolicy::MinimaxRegret,
            PreferencePolicy::Weighted,
            PreferencePolicy::ReportOnly,
        ] {
            let c = contract_for(model(lower, upper), &["A", "B", "C"], "outcome", policy);
            let inside =
                declared_measure(vec![vec![lower], vec![0.5 * (lower + upper)], vec![upper]], None);
            let result = evaluate(&c, &three_arms(), Some(&inside)).unwrap();
            let statement = result.statement();
            let lowered = statement.to_lowercase();
            for word in ["probab", "optimal", "robust", "likely", "confiden"] {
                assert!(!lowered.contains(word), "{statement}");
            }
            assert!(statement.ends_with('.'), "{statement}");
            let again = evaluate(&c, &three_arms(), Some(&inside)).unwrap();
            assert_eq!(statement, again.statement(), "deterministic");
        }
    }
}

#[test]
fn a_margin_below_the_rounding_of_large_expected_values_is_a_tie() {
    // E[benefit | A] = 1e9 + 2.5 exactly; B's fixed benefit lies four representable steps above
    // it, far below the rounding of expected values of this size.
    let value = P::sum(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1)));
    let decide = |mean_of_a: [f64; 4], fixed_b: f64| {
        let model = model_with(vec![parameter("w", 0.0, 1.0)], value.clone());
        let mut contract = contract_for(
            model,
            &["A", "B"],
            "outcome",
            PreferencePolicy::RequireInvariantBestAction,
        );
        contract.actions[1].inputs[0] =
            ActionInput::Fixed { value: fixed_b, units: "events".into() };
        let arms = vec![vec![mean_of_a.to_vec(), vec![1.0; 4]], vec![vec![0.0; 4], vec![1.0; 4]]];
        let source = law(&contract.model, &arms, None);
        evaluate_preference_contract(&contract, &source, None, options()).unwrap()
    };
    let mean: f64 = 1e9 + 2.5;
    let above = mean.next_up().next_up().next_up().next_up();
    let large = decide([1e9 + 1.0, 1e9 + 2.0, 1e9 + 3.0, 1e9 + 4.0], above);
    assert_eq!(large.verdict, PreferenceVerdict::Tied(vec!["A".into(), "B".into()]));
    assert!(large.actions.iter().all(|a| !a.uniquely_best && a.dominated_by.is_empty()));
    // The same lead at unit scale is a lead.
    let unit = decide([1.0, 2.0, 3.0, 4.0], 2.5 + 1e-6);
    assert_eq!(unit.verdict, PreferenceVerdict::NecessarilyBest("B".into()));
}
