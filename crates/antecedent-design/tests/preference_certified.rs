//! Certified verdicts across a preference set: switch points known in closed form, rigor
//! against dense parameter grids, relations, structures, measures and every refusal.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::reason_details;
use antecedent_design::decision_contract::{
    ActionKind, DecisionCriterion, HardConstraint, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::evaluate_contract;
use antecedent_design::decision_structural::{AtomEvidence, StructuralAtom};
use antecedent_design::preference_certified::{
    CertifiedActionOutcome, CertifiedAtomStatus, CertifiedOptions, CertifiedPreferenceResult,
    CertifiedVerdict, PreferenceCertifiedError as E, RegionStatus, StructuralCertifiedVerdict,
    evaluate_preference_certified, evaluate_preference_certified_structural,
};
use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferenceContractError, PreferencePolicy,
};
use antecedent_design::preference_decision::{MeasureOrigin, PreferenceMeasure};
use antecedent_design::preference_model::{
    Attribute, LinearRelation, PreferenceExpr as P, PreferenceParameter, PreferenceSet,
    RelationOrigin, RelationSense, ValueModel,
};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

const UNITS: &str = "u";

fn quantity(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        regime_id: format!("do({name}=1)"),
        units: UNITS.into(),
        ..common::quantity(name)
    }
}

/// An exact finite law over the named columns; `rows[i]` is draw `i`.
fn law(names: &[&str], rows: &[&[f64]], weights: Option<Vec<f64>>) -> DistributionArtifact {
    let columns: Vec<ScientificQuantity> = names.iter().map(|name| quantity(name)).collect();
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
    let draws: Vec<f64> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows.len(), names.len()],
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

fn parameter(id: &str, lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: id.into(),
        units: "u/u".into(),
        lower,
        upper,
        assumption_id: format!("{id}-judgment"),
    }
}

fn model(
    attributes: usize,
    parameters: Vec<PreferenceParameter>,
    relations: Vec<LinearRelation>,
    value: P,
) -> ValueModel {
    ValueModel {
        attributes: (0..attributes)
            .map(|k| Attribute { id: format!("x{k}"), units: UNITS.into() })
            .collect(),
        preferences: PreferenceSet { parameters, relations },
        value,
        utility_units: UNITS.into(),
    }
}

fn q(name: &str) -> ActionInput {
    ActionInput::Quantity(quantity(name))
}

fn fixed(value: f64) -> ActionInput {
    ActionInput::Fixed { value, units: UNITS.into() }
}

fn action(id: &str, inputs: Vec<ActionInput>) -> PreferenceAction {
    PreferenceAction { id: id.into(), kind: ActionKind::Intervention, inputs }
}

fn contract(
    model: ValueModel,
    actions: Vec<PreferenceAction>,
    criterion: DecisionCriterion,
    policy: PreferencePolicy,
) -> PreferenceContract {
    PreferenceContract {
        model,
        actions,
        criterion,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: policy,
    }
}

fn budget(max_boxes: usize) -> CertifiedOptions {
    CertifiedOptions { max_boxes, min_width_fraction: 0.0, tie_tolerance: 1e-9 }
}

fn orientation(criterion: DecisionCriterion) -> f64 {
    match criterion {
        DecisionCriterion::PosteriorExpectedLoss
        | DecisionCriterion::ExpectedRegret
        | DecisionCriterion::Regret => -1.0,
        _ => 1.0,
    }
}

fn inside(lower: &[f64], upper: &[f64], point: &[f64]) -> bool {
    point.iter().zip(lower.iter().zip(upper)).all(|(p, (l, u))| l <= p && p <= u)
}

fn grid(lower: f64, upper: f64, steps: u32) -> Vec<Vec<f64>> {
    (0..=steps).map(|i| vec![lower + (upper - lower) * f64::from(i) / f64::from(steps)]).collect()
}

/// Rigor: no certified region contains a grid point of the set at which a rival is strictly
/// better under the decision layer's own evaluation of the bound contract. Returns how many
/// grid points fell in certified regions.
fn assert_sound(
    contract: &PreferenceContract,
    source: &DistributionArtifact,
    result: &CertifiedPreferenceResult,
    points: &[Vec<f64>],
) -> usize {
    let sign = orientation(contract.criterion);
    let mut covered = 0;
    for point in points.iter().filter(|p| contract.model.preferences.contains(p, 0.0)) {
        let decision = evaluate_contract(&contract.bind(point).unwrap(), source).unwrap();
        let value = |id: &str| {
            let outcome = decision.actions.iter().find(|o| o.id == id).unwrap();
            outcome.admissible.then_some(sign * outcome.value)
        };
        for region in &result.regions {
            let RegionStatus::Best(best) = &region.status else { continue };
            if !inside(&region.lower, &region.upper, point) {
                continue;
            }
            covered += 1;
            let leader = value(best.as_str()).unwrap();
            for rival in &contract.actions {
                if let Some(other) = value(rival.id.as_str()) {
                    assert!(
                        other <= leader + 1e-12,
                        "{} beats certified {best} at {point:?}",
                        rival.id
                    );
                }
            }
        }
    }
    covered
}

fn certified(result: &CertifiedPreferenceResult, id: &str) -> Vec<(f64, f64)> {
    result
        .regions
        .iter()
        .filter(|r| matches!(&r.status, RegionStatus::Best(a) if a == id))
        .map(|r| (r.lower[0], r.upper[0]))
        .collect()
}

fn outcome<'r>(result: &'r CertifiedPreferenceResult, id: &str) -> &'r CertifiedActionOutcome {
    result.actions.iter().find(|a| a.id == id).unwrap()
}

/// Contains `value`.
fn encloses(bounds: Option<(f64, f64)>, value: f64) -> bool {
    bounds.is_some_and(|(l, h)| l <= value && value <= h)
}

/// Encloses `[lo, hi]` and is within `slack` of it.
fn tight(bounds: Option<(f64, f64)>, lo: f64, hi: f64, slack: f64) -> bool {
    bounds.is_some_and(|(l, h)| l <= lo && h >= hi && lo - l <= slack && h - hi <= slack)
}

// -------------------------------------------------------------------------------------------
// Discount factor: value = x0 + d x1 + d^2 x2. `soon` pays E[x1] = 10 next period, `late`
// pays E[x2] = 14 the period after, so soon leads iff 10 d > 14 d^2, i.e. d < 5/7.
// -------------------------------------------------------------------------------------------

fn discount_contract(policy: PreferencePolicy, swapped: bool) -> PreferenceContract {
    let value = P::sum(
        P::sum(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1))),
        P::product(P::product(P::Parameter(0), P::Parameter(0)), P::Attribute(2)),
    );
    let mut actions = vec![
        action("soon", vec![fixed(0.0), q("soon_next"), fixed(0.0)]),
        action("late", vec![fixed(0.0), fixed(0.0), q("late_later")]),
    ];
    if swapped {
        actions.reverse();
    }
    contract(
        model(3, vec![parameter("d", 0.5, 1.0)], vec![], value),
        actions,
        DecisionCriterion::PosteriorExpectedUtility,
        policy,
    )
}

fn discount_law() -> DistributionArtifact {
    law(&["soon_next", "late_later"], &[&[8.0, 12.0], &[12.0, 16.0]], None)
}

#[test]
fn discount_factor_switch_is_certified_on_each_side_and_shrinks_with_the_budget() {
    let contract = discount_contract(PreferencePolicy::RequireInvariantBestAction, false);
    let source = discount_law();
    let switch = 5.0 / 7.0;
    let mut previous = f64::INFINITY;
    let mut fractions = Vec::new();
    for max_boxes in [1, 9, 33, 129, 513] {
        let result =
            evaluate_preference_certified(&contract, &source, None, budget(max_boxes)).unwrap();
        assert!(result.boxes_examined <= max_boxes);
        for (_, upper) in certified(&result, "soon") {
            assert!(upper <= switch + 1e-12, "soon certified past the switch: {upper}");
        }
        for (lower, _) in certified(&result, "late") {
            assert!(lower >= switch - 1e-12, "late certified before the switch: {lower}");
        }
        assert!(result.undecided_fraction <= previous + 1e-15);
        previous = result.undecided_fraction;
        fractions.push(result.undecided_fraction);
        let total: f64 =
            result.actions.iter().map(|a| a.best_fraction).sum::<f64>() + result.undecided_fraction;
        assert!((total - 1.0).abs() < 1e-12);
        // Hand value: soon is best on (0.5, 5/7), a fraction 3/7 of the range.
        let soon = outcome(&result, "soon").best_fraction;
        assert!(soon <= 3.0 / 7.0 + 1e-12 && soon >= 3.0 / 7.0 - result.undecided_fraction);
        assert_sound(&contract, &source, &result, &grid(0.5, 1.0, 400));
    }
    assert!((fractions[0] - 1.0).abs() < 1e-15, "one box cannot be decided");
    assert!(fractions[4] < fractions[1] && fractions[4] < 0.02, "{fractions:?}");
    let result = evaluate_preference_certified(&contract, &source, None, budget(513)).unwrap();
    assert_eq!(
        result.verdict,
        CertifiedVerdict::PreferenceDependent {
            certified_somewhere: vec!["soon".into(), "late".into()]
        }
    );
    let coarse = evaluate_preference_certified(&contract, &source, None, budget(1)).unwrap();
    assert_eq!(
        coarse.verdict,
        CertifiedVerdict::Undetermined {
            candidates: vec!["soon".into(), "late".into()],
            undecided_fraction: 1.0
        }
    );
    assert_eq!(result.parameter_ids, vec!["d".to_owned()]);
    assert_eq!(result.assumptions, vec!["d-judgment".to_owned()]);
    assert_eq!(result.n_draws, 2);
}

#[test]
fn subdivision_is_deterministic_and_independent_of_action_order() {
    let source = discount_law();
    let first = discount_contract(PreferencePolicy::ReportOnly, false);
    let second = discount_contract(PreferencePolicy::ReportOnly, true);
    let a = evaluate_preference_certified(&first, &source, None, budget(65)).unwrap();
    let again = evaluate_preference_certified(&first, &source, None, budget(65)).unwrap();
    assert_eq!(a, again);
    let b = evaluate_preference_certified(&second, &source, None, budget(65)).unwrap();
    assert_eq!(a.regions.len(), b.regions.len());
    for (x, y) in a.regions.iter().zip(&b.regions) {
        assert_eq!((&x.lower, &x.upper), (&y.lower, &y.upper));
        match (&x.status, &y.status) {
            (RegionStatus::Best(p), RegionStatus::Best(r)) => assert_eq!(p, r),
            (
                RegionStatus::Undecided { candidates: p },
                RegionStatus::Undecided { candidates: r },
            ) => {
                let mut p = p.clone();
                let mut r = r.clone();
                p.sort();
                r.sort();
                assert_eq!(p, r);
            }
            other => panic!("status differs: {other:?}"),
        }
    }
    for id in ["soon", "late"] {
        assert!((outcome(&a, id).best_fraction - outcome(&b, id).best_fraction).abs() < 1e-15);
    }
    assert_eq!(a.verdict, CertifiedVerdict::ReportOnly);
    let sorted = a.regions.windows(2).all(|w| w[0].lower[0] < w[1].lower[0]);
    assert!(sorted, "regions are in lexicographic order of the lower corner");
}

// -------------------------------------------------------------------------------------------
// Breakpoint location: value = max(x0 - k, 0). `steady` pays 6 surely, `spread` pays 0 or 10.
// steady leads iff 6 - k > 5 - k / 2, i.e. k < 2; spread leads on (2, 9].
// -------------------------------------------------------------------------------------------

#[test]
fn breakpoint_location_switch_is_certified_on_each_side() {
    let value = P::maximum(P::difference(P::Attribute(0), P::Parameter(0)), P::Const(0.0));
    let contract = contract(
        model(1, vec![parameter("k", 0.0, 9.0)], vec![], value),
        vec![action("steady", vec![q("steady")]), action("spread", vec![q("spread")])],
        DecisionCriterion::PosteriorExpectedUtility,
        PreferencePolicy::RequireInvariantBestAction,
    );
    let source = law(&["steady", "spread"], &[&[6.0, 0.0], &[6.0, 10.0]], None);
    let result = evaluate_preference_certified(&contract, &source, None, budget(257)).unwrap();
    for (_, upper) in certified(&result, "steady") {
        assert!(upper <= 2.0 + 1e-12);
    }
    for (lower, _) in certified(&result, "spread") {
        assert!(lower >= 2.0 - 1e-12);
    }
    assert!(outcome(&result, "spread").best_fraction > 0.75);
    assert!(outcome(&result, "steady").best_fraction > 0.2);
    assert!(result.undecided_fraction < 0.01);
    assert!(assert_sound(&contract, &source, &result, &grid(0.0, 9.0, 450)) > 400);
    assert!(matches!(result.verdict, CertifiedVerdict::PreferenceDependent { .. }));
}

// -------------------------------------------------------------------------------------------
// Risk attitude: value = -exp(-r x0). `sure` pays 4, `gamble` pays 0 or 10 with equal weight.
// sure leads iff exp(-4 r) < (1 + exp(-10 r)) / 2; the indifference point lies in
// (0.08, 0.085) (by hand: g(0.08) < 0 < g(0.085) for g = (1 + e^{-10r})/2 - e^{-4r}).
// -------------------------------------------------------------------------------------------

fn risk_contract(policy: PreferencePolicy) -> PreferenceContract {
    let value = P::negate(P::exp(P::negate(P::product(P::Parameter(0), P::Attribute(0)))));
    contract(
        model(1, vec![parameter("r", 0.01, 1.0)], vec![], value),
        vec![action("sure", vec![q("sure")]), action("gamble", vec![q("gamble")])],
        DecisionCriterion::PosteriorExpectedUtility,
        policy,
    )
}

fn risk_law() -> DistributionArtifact {
    law(&["sure", "gamble"], &[&[4.0, 0.0], &[4.0, 10.0]], None)
}

fn sure_minus_gamble(r: f64) -> f64 {
    0.5 * (1.0 + (-10.0 * r).exp()) - (-4.0 * r).exp()
}

#[test]
fn exponential_risk_attitude_brackets_the_indifference_point() {
    assert!(sure_minus_gamble(0.08) < 0.0 && sure_minus_gamble(0.085) > 0.0);
    let contract = risk_contract(PreferencePolicy::RequireInvariantBestAction);
    let source = risk_law();
    let result = evaluate_preference_certified(&contract, &source, None, budget(257)).unwrap();
    for (_, upper) in certified(&result, "gamble") {
        assert!(upper <= 0.085, "gamble certified at r = {upper}");
    }
    for (lower, _) in certified(&result, "sure") {
        assert!(lower >= 0.08, "sure certified at r = {lower}");
    }
    // Closed-form rigor on a dense grid: the certified action is never strictly worse.
    for i in 0..=2000 {
        let r = 0.01 + 0.99 * f64::from(i) / 2000.0;
        for region in &result.regions {
            if inside(&region.lower, &region.upper, &[r]) {
                match &region.status {
                    RegionStatus::Best(a) if a == "sure" => assert!(sure_minus_gamble(r) >= 0.0),
                    RegionStatus::Best(_) => assert!(sure_minus_gamble(r) <= 0.0),
                    RegionStatus::Undecided { .. } => {}
                }
            }
        }
    }
    assert_eq!(
        result.verdict,
        CertifiedVerdict::PreferenceDependent {
            certified_somewhere: vec!["sure".into(), "gamble".into()]
        }
    );
    assert!(result.undecided_fraction < 0.01);
}

#[test]
fn weighted_policy_evaluates_measure_points_directly_when_the_value_is_not_lowerable() {
    let contract = risk_contract(PreferencePolicy::Weighted);
    let points = [0.02, 0.5, 0.9];
    let measure = PreferenceMeasure {
        points: points.iter().map(|r| vec![*r]).collect(),
        weights: None,
        labels: Some(vec!["near-neutral".into(), "averse".into(), "very-averse".into()]),
        origin: MeasureOrigin::Declared { assumption_id: "panel-view".into() },
        unresolved: vec![],
    };
    let result =
        evaluate_preference_certified(&contract, &risk_law(), Some(&measure), budget(64)).unwrap();
    let sure: f64 = points.iter().map(|r| -(-4.0 * r).exp()).sum::<f64>() / 3.0;
    let gamble: f64 = points.iter().map(|r| -0.5 * (1.0 + (-10.0 * r).exp())).sum::<f64>() / 3.0;
    assert!((outcome(&result, "sure").measure_value.unwrap() - sure).abs() < 1e-12);
    assert!((outcome(&result, "gamble").measure_value.unwrap() - gamble).abs() < 1e-12);
    match &result.verdict {
        CertifiedVerdict::MeasureChoice { action, share } => {
            assert_eq!(action, "sure");
            assert!((share - 2.0 / 3.0).abs() < 1e-12);
        }
        other => panic!("{other:?}"),
    }
    assert!((outcome(&result, "gamble").measure_share.unwrap() - 1.0 / 3.0).abs() < 1e-12);
    assert!(result.regions.is_empty());
    assert_eq!(result.boxes_examined, 0);
    assert_eq!(result.assumptions, vec!["panel-view".to_owned(), "r-judgment".to_owned()]);
}

// -------------------------------------------------------------------------------------------
// Affine model: value = x0 - w x1. E[A] = 10 - 4 w, E[B] = 6 - w; A - B = 4 - 3 w.
// -------------------------------------------------------------------------------------------

fn affine_contract(
    upper: f64,
    criterion: DecisionCriterion,
    policy: PreferencePolicy,
) -> PreferenceContract {
    let value = P::difference(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1)));
    contract(
        model(2, vec![parameter("w", 0.0, upper)], vec![], value),
        vec![
            action("A", vec![q("a_gain"), q("a_cost")]),
            action("B", vec![q("b_gain"), q("b_cost")]),
        ],
        criterion,
        policy,
    )
}

fn affine_law() -> DistributionArtifact {
    law(
        &["a_gain", "a_cost", "b_gain", "b_cost"],
        &[&[9.0, 3.0, 6.0, 0.0], &[11.0, 5.0, 6.0, 2.0]],
        None,
    )
}

#[test]
fn affine_model_agrees_with_hand_computed_margins() {
    let source = affine_law();
    let utility = DecisionCriterion::PosteriorExpectedUtility;
    let invariant = affine_contract(1.0, utility, PreferencePolicy::RequireInvariantBestAction);
    let result = evaluate_preference_certified(&invariant, &source, None, budget(64)).unwrap();
    assert_eq!(result.verdict, CertifiedVerdict::NecessarilyBest("A".into()));
    assert_eq!(result.boxes_examined, 1, "an exact affine margin decides the declared box");
    assert!((outcome(&result, "A").best_fraction - 1.0).abs() < 1e-15);
    assert!(tight(outcome(&result, "A").value_bounds, 6.0, 10.0, 1e-9));
    assert!(tight(outcome(&result, "B").value_bounds, 5.0, 6.0, 1e-9));
    // One box: the worst value is bounded below over the box and above at its centre.
    let (low, high) = outcome(&result, "A").worst_case_bounds.unwrap();
    assert!(low <= 6.0 && low > 6.0 - 1e-9 && (high - 8.0).abs() < 1e-9);
    let (low, high) = outcome(&result, "B").worst_case_bounds.unwrap();
    assert!(low <= 5.0 && low > 5.0 - 1e-9 && (high - 5.5).abs() < 1e-9);
    assert!(tight(outcome(&result, "A").worst_case_regret_bounds, 0.0, 0.0, 1e-9));
    let (low, high) = outcome(&result, "B").worst_case_regret_bounds.unwrap();
    assert!(encloses(Some((low, high)), 4.0) && high < 4.0 + 1e-9 && (low - 2.5).abs() < 1e-9);

    let maximin = affine_contract(1.0, utility, PreferencePolicy::Maximin);
    let result = evaluate_preference_certified(&maximin, &source, None, budget(64)).unwrap();
    assert_eq!(result.verdict, CertifiedVerdict::WorstCaseChoice("A".into()));
    let regret = affine_contract(1.0, utility, PreferencePolicy::MinimaxRegret);
    let result = evaluate_preference_certified(&regret, &source, None, budget(64)).unwrap();
    assert_eq!(result.verdict, CertifiedVerdict::MinimaxRegretChoice("A".into()));

    // A loss criterion reverses the order: B has the smaller expected loss on [0, 1].
    let loss = affine_contract(
        1.0,
        DecisionCriterion::PosteriorExpectedLoss,
        PreferencePolicy::RequireInvariantBestAction,
    );
    let result = evaluate_preference_certified(&loss, &source, None, budget(64)).unwrap();
    assert_eq!(result.verdict, CertifiedVerdict::NecessarilyBest("B".into()));
    assert!(tight(outcome(&result, "B").value_bounds, 5.0, 6.0, 1e-9));
    // The worst loss of B is its largest, 6, at w = 0.
    let (low, high) = outcome(&result, "B").worst_case_bounds.unwrap();
    assert!(encloses(Some((low, high)), 6.0) && high < 6.0 + 1e-9 && (low - 5.5).abs() < 1e-9);

    // Expected regret ranks like expected utility; per-draw regret of A is zero on [0, 1].
    let expected_regret = affine_contract(
        1.0,
        DecisionCriterion::ExpectedRegret,
        PreferencePolicy::RequireInvariantBestAction,
    );
    let result =
        evaluate_preference_certified(&expected_regret, &source, None, budget(64)).unwrap();
    assert_eq!(result.verdict, CertifiedVerdict::NecessarilyBest("A".into()));
    assert!(tight(outcome(&result, "A").value_bounds, 0.0, 0.0, 1e-9));
    assert!(tight(outcome(&result, "B").value_bounds, 1.0, 4.0, 1e-9));
}

#[test]
fn affine_switch_is_located_and_sound() {
    let source = affine_law();
    let contract = affine_contract(
        2.0,
        DecisionCriterion::PosteriorExpectedUtility,
        PreferencePolicy::RequireInvariantBestAction,
    );
    let result = evaluate_preference_certified(&contract, &source, None, budget(129)).unwrap();
    for (_, upper) in certified(&result, "A") {
        assert!(upper <= 4.0 / 3.0 + 1e-12);
    }
    for (lower, _) in certified(&result, "B") {
        assert!(lower >= 4.0 / 3.0 - 1e-12);
    }
    assert!(assert_sound(&contract, &source, &result, &grid(0.0, 2.0, 300)) > 290);
    assert_eq!(
        result.verdict,
        CertifiedVerdict::PreferenceDependent { certified_somewhere: vec!["A".into(), "B".into()] }
    );
}

#[test]
fn weighted_policy_reads_measure_points_through_the_bound_contract() {
    let contract = affine_contract(
        2.0,
        DecisionCriterion::PosteriorExpectedUtility,
        PreferencePolicy::Weighted,
    );
    let measure = PreferenceMeasure {
        points: vec![vec![0.2], vec![0.5], vec![1.8]],
        weights: Some(vec![1.0, 1.0, 2.0]),
        labels: None,
        origin: MeasureOrigin::ChoiceEstimate { estimate_identity: "estimate-1".into() },
        unresolved: vec![],
    };
    let result =
        evaluate_preference_certified(&contract, &affine_law(), Some(&measure), budget(8)).unwrap();
    // Hand values: A = 0.25 * 9.2 + 0.25 * 8 + 0.5 * 2.8, B = 0.25 * 5.8 + 0.25 * 5.5 + 0.5 * 4.2.
    assert!((outcome(&result, "A").measure_value.unwrap() - 5.7).abs() < 1e-12);
    assert!((outcome(&result, "B").measure_value.unwrap() - 4.925).abs() < 1e-12);
    match &result.verdict {
        CertifiedVerdict::MeasureChoice { action, share } => {
            assert_eq!(action, "A");
            assert!((share - 0.5).abs() < 1e-12);
        }
        other => panic!("{other:?}"),
    }
}

// -------------------------------------------------------------------------------------------
// Non-expectation criteria on a two-draw law. U_A = (5 - 2w, 2), U_B = (3.2 - w/2, 1).
// -------------------------------------------------------------------------------------------

fn small_law_contract(
    criterion: DecisionCriterion,
    policy: PreferencePolicy,
) -> PreferenceContract {
    affine_contract(2.0, criterion, policy)
}

fn small_law() -> DistributionArtifact {
    law(
        &["a_gain", "a_cost", "b_gain", "b_cost"],
        &[&[5.0, 2.0, 3.2, 0.5], &[2.0, 0.0, 1.0, 0.0]],
        None,
    )
}

#[test]
fn threshold_probability_certificates_hold_on_a_dense_grid() {
    // P(U_A >= 3) = 0.5 on w <= 1, else 0; P(U_B >= 3) = 0.5 on w <= 0.4, else 0.
    let contract = small_law_contract(
        DecisionCriterion::ThresholdProbability { threshold: 3.0 },
        PreferencePolicy::RequireInvariantBestAction,
    );
    let source = small_law();
    let result = evaluate_preference_certified(&contract, &source, None, budget(257)).unwrap();
    assert!(certified(&result, "B").is_empty());
    let a = certified(&result, "A");
    assert!(!a.is_empty());
    for (lower, upper) in a {
        assert!(lower >= 0.4 - 1e-12 && upper <= 1.0 + 1e-12);
    }
    assert!(assert_sound(&contract, &source, &result, &grid(0.0, 2.0, 800)) > 200);
    assert!(matches!(result.verdict, CertifiedVerdict::Undetermined { .. }));
}

#[test]
fn quantile_certificates_hold_on_a_dense_grid() {
    // Q_0.75 is the larger draw: A = max(5 - 2w, 2), B = 3.2 - w/2; A leads iff w < 1.2.
    let contract = small_law_contract(
        DecisionCriterion::Quantile { p: 0.75 },
        PreferencePolicy::RequireInvariantBestAction,
    );
    let source = small_law();
    let result = evaluate_preference_certified(&contract, &source, None, budget(257)).unwrap();
    for (_, upper) in certified(&result, "A") {
        assert!(upper <= 1.2 + 1e-12);
    }
    for (lower, _) in certified(&result, "B") {
        assert!(lower >= 1.2 - 1e-12);
    }
    assert!(assert_sound(&contract, &source, &result, &grid(0.0, 2.0, 800)) > 700);
    assert_eq!(
        result.verdict,
        CertifiedVerdict::PreferenceDependent { certified_somewhere: vec!["A".into(), "B".into()] }
    );
}

#[test]
fn largest_regret_certificates_hold_on_a_dense_grid() {
    // Regret_A = max(0, 1.5w - 1.8), Regret_B = max(1, 1.8 - 1.5w); A leads iff w < 28/15.
    let contract =
        small_law_contract(DecisionCriterion::Regret, PreferencePolicy::RequireInvariantBestAction);
    let source = small_law();
    let result = evaluate_preference_certified(&contract, &source, None, budget(257)).unwrap();
    let switch = 28.0 / 15.0;
    for (_, upper) in certified(&result, "A") {
        assert!(upper <= switch + 1e-12);
    }
    for (lower, _) in certified(&result, "B") {
        assert!(lower >= switch - 1e-12);
    }
    assert!(assert_sound(&contract, &source, &result, &grid(0.0, 2.0, 800)) > 700);
    // Regret of B never falls below 1; its value enclosure says so.
    let (low, _) = outcome(&result, "B").value_bounds.unwrap();
    assert!(low <= 1.0 && low > 1.0 - 1e-9);
    assert_eq!(
        result.verdict,
        CertifiedVerdict::PreferenceDependent { certified_somewhere: vec!["A".into(), "B".into()] }
    );
}

// -------------------------------------------------------------------------------------------
// Relations
// -------------------------------------------------------------------------------------------

fn relation(
    id: &str,
    terms: Vec<(usize, f64)>,
    sense: RelationSense,
    bound: f64,
) -> LinearRelation {
    LinearRelation { id: id.into(), terms, sense, bound, origin: RelationOrigin::Declared }
}

fn two_weight_contract(relations: Vec<LinearRelation>) -> PreferenceContract {
    // value = a x0 + b x1; A pays (1, 0) and B pays (0, 1), so A leads iff a > b.
    let value = P::sum(
        P::product(P::Parameter(0), P::Attribute(0)),
        P::product(P::Parameter(1), P::Attribute(1)),
    );
    contract(
        model(2, vec![parameter("a", 0.0, 1.0), parameter("b", 0.0, 1.0)], relations, value),
        vec![action("A", vec![q("one"), fixed(0.0)]), action("B", vec![fixed(0.0), q("one")])],
        DecisionCriterion::PosteriorExpectedUtility,
        PreferencePolicy::RequireInvariantBestAction,
    )
}

#[test]
fn relations_drop_boxes_outside_and_flag_boxes_on_the_boundary() {
    let contract = two_weight_contract(vec![relation(
        "budget",
        vec![(0, 1.0), (1, 1.0)],
        RelationSense::AtMost,
        1.0,
    )]);
    let source = law(&["one"], &[&[1.0]], None);
    let result = evaluate_preference_certified(&contract, &source, None, budget(401)).unwrap();
    let mut kept_volume = 0.0;
    let mut flagged = 0;
    for region in &result.regions {
        assert!(region.lower[0] + region.lower[1] <= 1.0 + 1e-12, "a box wholly outside is kept");
        if region.touches_relation_boundary {
            flagged += 1;
        } else {
            assert!(region.upper[0] + region.upper[1] <= 1.0 + 1e-12);
        }
        kept_volume += (region.upper[0] - region.lower[0]) * (region.upper[1] - region.lower[1]);
    }
    assert!(flagged > 0);
    assert!(kept_volume < 0.9 && kept_volume >= 0.5, "{kept_volume}");
    let total: f64 =
        result.actions.iter().map(|a| a.best_fraction).sum::<f64>() + result.undecided_fraction;
    assert!((total - 1.0).abs() < 1e-12);
    assert_eq!(
        result.verdict,
        CertifiedVerdict::PreferenceDependent { certified_somewhere: vec!["A".into(), "B".into()] }
    );
    let points: Vec<Vec<f64>> = (0..=40)
        .flat_map(|i| (0..=40).map(move |j| vec![f64::from(i) / 40.0, f64::from(j) / 40.0]))
        .collect();
    assert!(assert_sound(&contract, &source, &result, &points) > 0);
}

#[test]
fn a_relation_that_orders_the_weights_makes_one_action_necessarily_best() {
    let contract = two_weight_contract(vec![relation(
        "a-at-least-b-plus",
        vec![(0, 1.0), (1, -1.0)],
        RelationSense::AtLeast,
        0.1,
    )]);
    let source = law(&["one"], &[&[1.0]], None);
    let result = evaluate_preference_certified(&contract, &source, None, budget(4001)).unwrap();
    assert!(certified(&result, "B").is_empty());
    assert_eq!(result.verdict, CertifiedVerdict::NecessarilyBest("A".into()));
    assert!(result.boxes_examined < 4001);
}

#[test]
fn a_fixed_parameter_contributes_no_dimension() {
    // value = x0 - w x1 + c x1 with c fixed at zero.
    let value = P::sum(
        P::difference(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1))),
        P::product(P::Parameter(1), P::Attribute(1)),
    );
    let contract = contract(
        model(2, vec![parameter("w", 0.0, 2.0), parameter("c", 0.0, 0.0)], vec![], value),
        vec![
            action("A", vec![q("a_gain"), q("a_cost")]),
            action("B", vec![q("b_gain"), q("b_cost")]),
        ],
        DecisionCriterion::PosteriorExpectedUtility,
        PreferencePolicy::RequireInvariantBestAction,
    );
    let result = evaluate_preference_certified(&contract, &affine_law(), None, budget(65)).unwrap();
    assert!(result.regions.iter().all(|r| r.lower[1].abs() < 1e-300 && r.upper[1].abs() < 1e-300));
    let a = outcome(&result, "A").best_fraction;
    assert!(a <= 2.0 / 3.0 + 1e-12 && a >= 2.0 / 3.0 - result.undecided_fraction - 1e-12);
}

// -------------------------------------------------------------------------------------------
// Hard constraints
// -------------------------------------------------------------------------------------------

fn cap(bound: f64) -> HardConstraint {
    HardConstraint {
        id: "cap".into(),
        expr: UtilityExpr::Input(1),
        bound,
        min_probability: 1.0,
        units: UNITS.into(),
        applies_to: vec![],
    }
}

#[test]
fn hard_constraints_exclude_actions_from_every_comparison() {
    let mut contract = affine_contract(
        2.0,
        DecisionCriterion::PosteriorExpectedUtility,
        PreferencePolicy::RequireInvariantBestAction,
    );
    contract.constraints = vec![cap(3.0)];
    let result = evaluate_preference_certified(&contract, &affine_law(), None, budget(8)).unwrap();
    let a = outcome(&result, "A");
    assert!(!a.admissible && a.value_bounds.is_none());
    assert_eq!(a.exclusions[0].constraint_id, "cap");
    assert!((a.exclusions[0].probability - 0.5).abs() < 1e-15);
    assert_eq!(result.verdict, CertifiedVerdict::NecessarilyBest("B".into()));
    contract.constraints = vec![cap(-1.0)];
    let result = evaluate_preference_certified(&contract, &affine_law(), None, budget(8)).unwrap();
    assert_eq!(result.verdict, CertifiedVerdict::NoAdmissibleAction);
    assert!(result.regions.is_empty());
}

#[test]
fn constraints_on_a_value_that_cannot_be_lowered_are_evaluated_on_the_draws() {
    let mut contract = risk_contract(PreferencePolicy::RequireInvariantBestAction);
    contract.constraints = vec![HardConstraint {
        id: "floor".into(),
        expr: UtilityExpr::Neg(Box::new(UtilityExpr::Input(0))),
        bound: -1.0,
        min_probability: 0.75,
        units: UNITS.into(),
        applies_to: vec![],
    }];
    let result = evaluate_preference_certified(&contract, &risk_law(), None, budget(8)).unwrap();
    let gamble = outcome(&result, "gamble");
    assert!(!gamble.admissible);
    assert_eq!(gamble.exclusions[0].constraint_id, "floor");
    assert!((gamble.exclusions[0].probability - 0.5).abs() < 1e-15);
    assert!(outcome(&result, "sure").admissible);
    assert_eq!(result.verdict, CertifiedVerdict::NecessarilyBest("sure".into()));
}

// -------------------------------------------------------------------------------------------
// Structures
// -------------------------------------------------------------------------------------------

fn atom(id: &str, probability: Option<f64>, evidence: AtomEvidence) -> StructuralAtom {
    StructuralAtom { id: id.into(), probability, evidence }
}

fn evaluated(source: DistributionArtifact) -> AtomEvidence {
    AtomEvidence::Evaluated(Box::new(source))
}

/// Under structure two, E[A] = 7 - w and E[B] = 8 - 1.5 w, so B leads on [0, 1].
fn second_structure() -> DistributionArtifact {
    law(&["a_gain", "a_cost", "b_gain", "b_cost"], &[&[7.0, 1.0, 8.0, 1.5]], None)
}

fn structural(policy: StructuralPolicy) -> PreferenceContract {
    let mut contract = affine_contract(
        1.0,
        DecisionCriterion::PosteriorExpectedUtility,
        PreferencePolicy::RequireInvariantBestAction,
    );
    contract.structural_policy = policy;
    contract
}

#[test]
fn structural_policies_combine_two_structures() {
    let atoms = |p: Option<f64>| {
        vec![atom("s1", p, evaluated(affine_law())), atom("s2", p, evaluated(second_structure()))]
    };
    let options = budget(64);
    let invariant = structural(StructuralPolicy::RequireInvariantBestAction);
    let result =
        evaluate_preference_certified_structural(&invariant, &atoms(None), None, options).unwrap();
    assert!(result.combined.is_none());
    let StructuralCertifiedVerdict::StructuresDisagree(verdicts) = &result.verdict else {
        panic!("{:?}", result.verdict)
    };
    assert_eq!(verdicts[0].verdict, CertifiedVerdict::NecessarilyBest("A".into()));
    assert_eq!(verdicts[1].verdict, CertifiedVerdict::NecessarilyBest("B".into()));
    let agreeing =
        vec![atom("s1", None, evaluated(affine_law())), atom("s1b", None, evaluated(affine_law()))];
    let result =
        evaluate_preference_certified_structural(&invariant, &agreeing, None, options).unwrap();
    assert_eq!(
        result.verdict,
        StructuralCertifiedVerdict::Combined(CertifiedVerdict::NecessarilyBest("A".into()))
    );

    // Worst case over structures: min(10 - 4w, 7 - w) - (6 - w) = 1 on [0, 1].
    let maximin = structural(StructuralPolicy::Maximin);
    let result =
        evaluate_preference_certified_structural(&maximin, &atoms(None), None, options).unwrap();
    assert_eq!(
        result.verdict,
        StructuralCertifiedVerdict::Combined(CertifiedVerdict::NecessarilyBest("A".into()))
    );
    let combined = result.combined.unwrap();
    assert!(tight(outcome(&combined, "A").value_bounds, 6.0, 7.0, 1e-9));
    assert_eq!(combined.n_draws, 3);

    // Probability-weighted: 8.5 - 2.5 w against 7 - 1.25 w.
    let bayes = structural(StructuralPolicy::BayesOverStructures);
    let result =
        evaluate_preference_certified_structural(&bayes, &atoms(Some(0.5)), None, options).unwrap();
    assert_eq!(
        result.verdict,
        StructuralCertifiedVerdict::Combined(CertifiedVerdict::NecessarilyBest("A".into()))
    );
    assert!(tight(outcome(result.combined.as_ref().unwrap(), "A").value_bounds, 6.0, 8.5, 1e-9));
    assert_eq!(result.evaluated_mass, Some(1.0));

    let report = structural(StructuralPolicy::ReportOnly);
    let result =
        evaluate_preference_certified_structural(&report, &atoms(None), None, options).unwrap();
    assert_eq!(result.verdict, StructuralCertifiedVerdict::ReportOnly);
    assert!(result.combined.is_none());
    assert!(result.atoms.iter().all(|a| matches!(a.status, CertifiedAtomStatus::Evaluated(_))));
}

#[test]
fn unresolved_structures_leave_claims_insufficient() {
    let options = budget(16);
    let partial = vec![
        atom("s1", Some(0.5), evaluated(affine_law())),
        atom("s2", Some(0.5), AtomEvidence::Unidentified),
    ];
    let bayes = structural(StructuralPolicy::BayesOverStructures);
    let result = evaluate_preference_certified_structural(&bayes, &partial, None, options).unwrap();
    assert!(matches!(result.verdict, StructuralCertifiedVerdict::InsufficientScience(_)));
    assert_eq!(result.unidentified_mass, Some(0.5));
    assert!(result.combined.is_some());

    let missing_column = law(&["a_gain", "a_cost"], &[&[1.0, 1.0]], None);
    let atoms = vec![
        atom("s1", None, evaluated(affine_law())),
        atom("s2", None, evaluated(missing_column)),
    ];
    let maximin = structural(StructuralPolicy::Maximin);
    let result = evaluate_preference_certified_structural(&maximin, &atoms, None, options).unwrap();
    assert!(matches!(result.atoms[1].status, CertifiedAtomStatus::Unevaluated(_)));
    assert!(matches!(result.verdict, StructuralCertifiedVerdict::InsufficientScience(_)));
    let invariant = structural(StructuralPolicy::RequireInvariantBestAction);
    let result =
        evaluate_preference_certified_structural(&invariant, &atoms, None, options).unwrap();
    assert!(matches!(result.verdict, StructuralCertifiedVerdict::InsufficientScience(_)));
}

// -------------------------------------------------------------------------------------------
// Refusals
// -------------------------------------------------------------------------------------------

fn refusal(contract: &PreferenceContract, measure: Option<&PreferenceMeasure>) -> E {
    evaluate_preference_certified(contract, &affine_law(), measure, budget(16)).unwrap_err()
}

fn weighted_with(edit: impl FnOnce(&mut PreferenceMeasure)) -> E {
    let contract = affine_contract(
        2.0,
        DecisionCriterion::PosteriorExpectedUtility,
        PreferencePolicy::Weighted,
    );
    let mut measure = PreferenceMeasure {
        points: vec![vec![0.5]],
        weights: None,
        labels: None,
        origin: MeasureOrigin::Declared { assumption_id: "view".into() },
        unresolved: vec![],
    };
    edit(&mut measure);
    refusal(&contract, Some(&measure))
}

#[test]
fn every_contract_option_and_measure_refusal_is_typed() {
    let utility = DecisionCriterion::PosteriorExpectedUtility;
    let mut one_action = affine_contract(1.0, utility, PreferencePolicy::ReportOnly);
    one_action.actions.truncate(1);
    let error = refusal(&one_action, None);
    assert!(matches!(error, E::Contract(PreferenceContractError::TooFewActions { count: 1 })));
    assert_eq!(error.detail(), reason_details::preference_contract::TOO_FEW_ACTIONS);

    let structures = affine_contract(
        1.0,
        DecisionCriterion::MaximinOverStructures,
        PreferencePolicy::ReportOnly,
    );
    assert_eq!(refusal(&structures, None), E::StructureInputsRequired);

    let plain = affine_contract(1.0, utility, PreferencePolicy::ReportOnly);
    let law = affine_law();
    for (options, name) in [
        (CertifiedOptions { max_boxes: 0, ..budget(1) }, "max_boxes"),
        (CertifiedOptions { min_width_fraction: 2.0, ..budget(1) }, "min_width_fraction"),
        (CertifiedOptions { tie_tolerance: -1.0, ..budget(1) }, "tie_tolerance"),
    ] {
        let error = evaluate_preference_certified(&plain, &law, None, options).unwrap_err();
        assert_eq!(error, E::InvalidOptions { option: name });
        assert_eq!(error.detail(), reason_details::preference_certified::OPTIONS_INVALID);
    }

    let quantile_regret = affine_contract(
        1.0,
        DecisionCriterion::Quantile { p: 0.5 },
        PreferencePolicy::MinimaxRegret,
    );
    assert_eq!(refusal(&quantile_regret, None), E::MinimaxRegretNeedsExpectation);

    let weighted = affine_contract(1.0, utility, PreferencePolicy::Weighted);
    assert_eq!(refusal(&weighted, None), E::MeasureRequired);
    assert_eq!(weighted_with(|m| m.points.clear()), E::MeasureEmpty);
    assert_eq!(weighted_with(|m| m.unresolved = vec![4]), E::MeasureUnknownParameter { index: 4 });
    assert_eq!(
        weighted_with(|m| m.unresolved = vec![0]),
        E::MeasureIncomplete { parameters: vec!["w".into()] }
    );
    assert_eq!(
        weighted_with(|m| m.points.push(vec![0.1, 0.2])),
        E::MeasurePointDimension { index: 1, expected: 1, found: 2 }
    );
    assert_eq!(weighted_with(|m| m.points[0][0] = f64::NAN), E::MeasureNonFinitePoint { index: 0 });
    assert_eq!(weighted_with(|m| m.points[0][0] = 3.0), E::MeasurePointOutsideSet { index: 0 });
    assert_eq!(weighted_with(|m| m.weights = Some(vec![-1.0])), E::MeasureWeightsInvalid);
    assert_eq!(weighted_with(|m| m.weights = Some(vec![1.0, 1.0])), E::MeasureWeightsInvalid);
    assert_eq!(weighted_with(|m| m.labels = Some(vec![])), E::MeasureLabelsInvalid);

    let missing = law_missing_b();
    let error = evaluate_preference_certified(&plain, &missing, None, budget(4)).unwrap_err();
    assert!(matches!(error, E::Eval(_)));
    assert_eq!(error.detail(), reason_details::preference_certified::SOURCE_REFUSED);
}

fn law_missing_b() -> DistributionArtifact {
    law(&["a_gain", "a_cost"], &[&[1.0, 1.0]], None)
}

#[test]
fn every_enclosure_and_set_refusal_is_typed() {
    let empty = two_weight_contract(vec![relation(
        "impossible",
        vec![(0, 1.0), (1, 1.0)],
        RelationSense::AtLeast,
        3.0,
    )]);
    let one = law(&["one"], &[&[1.0]], None);
    let error = evaluate_preference_certified(&empty, &one, None, budget(16)).unwrap_err();
    assert_eq!(error, E::EmptyPreferenceSet);
    assert_eq!(error.detail(), reason_details::preference_certified::PREFERENCE_SET_EMPTY);

    // ln(x0 + k) with x0 = 2 and k down to -5 is undefined on part of the declared range.
    let ln = contract(
        model(
            1,
            vec![parameter("k", -5.0, 1.0)],
            vec![],
            P::ln(P::sum(P::Attribute(0), P::Parameter(0))),
        ),
        vec![action("sure", vec![q("sure")]), action("gamble", vec![q("gamble")])],
        DecisionCriterion::PosteriorExpectedUtility,
        PreferencePolicy::ReportOnly,
    );
    let source = law(&["sure", "gamble"], &[&[2.0, 3.0]], None);
    let error = evaluate_preference_certified(&ln, &source, None, budget(16)).unwrap_err();
    assert_eq!(
        error,
        E::LogarithmDomain { action: "sure".into(), lower: vec![-5.0], upper: vec![1.0] }
    );
    assert_eq!(error.detail(), reason_details::preference_certified::LOGARITHM_DOMAIN);

    let overflow = contract(
        model(
            1,
            vec![parameter("k", 0.0, 1000.0)],
            vec![],
            P::exp(P::product(P::Parameter(0), P::Attribute(0))),
        ),
        vec![action("sure", vec![q("sure")]), action("gamble", vec![q("gamble")])],
        DecisionCriterion::PosteriorExpectedUtility,
        PreferencePolicy::ReportOnly,
    );
    let source = law(&["sure", "gamble"], &[&[1000.0, 1.0]], None);
    let error = evaluate_preference_certified(&overflow, &source, None, budget(16)).unwrap_err();
    assert!(matches!(error, E::NonFiniteEnclosure { ref subject, .. } if subject == "sure"));
    assert_eq!(error.detail(), reason_details::preference_certified::ENCLOSURE_NON_FINITE);

    // A measure point at which the value overflows: -exp(-r x) with x = -1000 and r = 1.
    let risky = risk_contract(PreferencePolicy::Weighted);
    let source = law(&["sure", "gamble"], &[&[-1000.0, 1.0]], None);
    let measure = PreferenceMeasure {
        points: vec![vec![1.0]],
        weights: None,
        labels: None,
        origin: MeasureOrigin::Declared { assumption_id: "view".into() },
        unresolved: vec![],
    };
    let error =
        evaluate_preference_certified(&risky, &source, Some(&measure), budget(16)).unwrap_err();
    assert!(matches!(error, E::Model(_)));
    assert_eq!(error.detail(), reason_details::preference_model::VALUE_NON_FINITE);
}

#[test]
fn every_structural_refusal_is_typed() {
    let options = budget(8);
    let invariant = structural(StructuralPolicy::RequireInvariantBestAction);
    let error =
        evaluate_preference_certified_structural(&invariant, &[], None, options).unwrap_err();
    assert_eq!(error, E::InvalidAtoms);
    assert_eq!(error.detail(), reason_details::preference_certified::ATOMS_INVALID);
    let repeated =
        vec![atom("s", None, evaluated(affine_law())), atom("s", None, evaluated(affine_law()))];
    assert_eq!(
        evaluate_preference_certified_structural(&invariant, &repeated, None, options).unwrap_err(),
        E::InvalidAtoms
    );
    let partial = vec![
        atom("s1", Some(0.5), evaluated(affine_law())),
        atom("s2", None, evaluated(affine_law())),
    ];
    let error =
        evaluate_preference_certified_structural(&invariant, &partial, None, options).unwrap_err();
    assert_eq!(error, E::InvalidProbabilities);
    assert_eq!(error.detail(), reason_details::preference_certified::ATOM_PROBABILITIES_INVALID);

    let bayes = structural(StructuralPolicy::BayesOverStructures);
    let unweighted = vec![atom("s1", None, evaluated(affine_law()))];
    let error =
        evaluate_preference_certified_structural(&bayes, &unweighted, None, options).unwrap_err();
    assert_eq!(error, E::ProbabilitiesRequired);
    assert_eq!(error.detail(), reason_details::preference_certified::ATOM_PROBABILITIES_REQUIRED);

    let mut quantile = bayes;
    quantile.criterion = DecisionCriterion::Quantile { p: 0.5 };
    let weighted = vec![atom("s1", Some(1.0), evaluated(affine_law()))];
    let error =
        evaluate_preference_certified_structural(&quantile, &weighted, None, options).unwrap_err();
    assert_eq!(error, E::MixtureCriterionUnsupported { criterion: "quantile" });
    assert_eq!(error.detail(), reason_details::preference_certified::MIXTURE_CRITERION_UNSUPPORTED);
}

#[test]
fn the_remaining_detail_ids_are_stable() {
    let cases = [
        (
            E::StructureInputsRequired,
            reason_details::preference_certified::STRUCTURE_INPUTS_REQUIRED,
        ),
        (
            E::MinimaxRegretNeedsExpectation,
            reason_details::preference_certified::MINIMAX_REGRET_NEEDS_EXPECTATION,
        ),
        (E::MeasureRequired, reason_details::preference_certified::MEASURE_REQUIRED),
        (E::MeasureEmpty, reason_details::preference_certified::MEASURE_EMPTY),
        (
            E::MeasureUnknownParameter { index: 0 },
            reason_details::preference_certified::MEASURE_UNKNOWN_PARAMETER,
        ),
        (
            E::MeasureIncomplete { parameters: vec![] },
            reason_details::preference_certified::MEASURE_INCOMPLETE,
        ),
        (
            E::MeasurePointDimension { index: 0, expected: 1, found: 2 },
            reason_details::preference_certified::MEASURE_POINT_DIMENSION,
        ),
        (
            E::MeasureNonFinitePoint { index: 0 },
            reason_details::preference_certified::MEASURE_POINT_NON_FINITE,
        ),
        (
            E::MeasurePointOutsideSet { index: 0 },
            reason_details::preference_certified::MEASURE_POINT_OUTSIDE_SET,
        ),
        (E::MeasureWeightsInvalid, reason_details::preference_certified::MEASURE_WEIGHTS_INVALID),
        (E::MeasureLabelsInvalid, reason_details::preference_certified::MEASURE_LABELS_INVALID),
    ];
    for (error, detail) in cases {
        assert_eq!(error.detail(), detail);
    }
}
