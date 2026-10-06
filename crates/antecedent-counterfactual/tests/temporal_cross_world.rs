//! X8 fixed-population temporal counterfactual and the closed transported gate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::ExecutionContext;
use antecedent_counterfactual::temporal_cross_world::{
    FactualUnitHistory, HistoryId, NamedActionHistory, NodeMechanism, SnapshotId,
    TemporalCounterfactualError, TemporalCounterfactualSpec, TemporalGraph,
    TemporalMechanismFit, TemporalNode, UnitId, evaluate_temporal_counterfactual,
};
use antecedent_counterfactual::transported_gate::{
    PopulationRole, RegimeFactorKey, TransportedCounterfactualGate,
    TransportedCounterfactualPrerequisites, refuse_transported_counterfactual,
};

use TemporalNode::{Action0, Action1, Covariate0, Covariate1, Outcome};

// Known closed-form two-slice SCM:
//   L0 = 1.0 + u0
//   L1 = 0.5 + 0.8 L0 + 1.5 A0 + u1
//   Y  = -1.0 + 0.5 L0 + 0.7 A0 + 2.0 L1 + 1.2 A1 + u2
fn simulate(noise: [f64; 3], a: [f64; 2]) -> [f64; 5] {
    let l0 = 1.0 + noise[0];
    let l1 = 0.5 + 0.8 * l0 + 1.5 * a[0] + noise[1];
    let y = -1.0 + 0.5 * l0 + 0.7 * a[0] + 2.0 * l1 + 1.2 * a[1] + noise[2];
    [l0, a[0], l1, a[1], y]
}

const NOISE: [[f64; 3]; 4] =
    [[0.3, -0.2, 0.5], [-0.4, 0.6, -0.1], [0.0, 0.1, 0.9], [1.1, -0.7, -0.3]];
const FACTUAL_ACTIONS: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
const PLUS: [f64; 2] = [1.0, 1.0];
const MINUS: [f64; 2] = [0.0, 0.0];

fn graph() -> TemporalGraph {
    TemporalGraph {
        horizon: 2,
        edges: vec![
            (Covariate0, Covariate1),
            (Action0, Covariate1),
            (Covariate0, Outcome),
            (Action0, Outcome),
            (Covariate1, Outcome),
            (Action1, Outcome),
        ],
        latent_confounding: false,
    }
}

fn fit(halfwidth: Option<f64>) -> TemporalMechanismFit {
    TemporalMechanismFit {
        fit_id: "fit-known-coefficients".to_owned(),
        mechanisms: vec![
            NodeMechanism {
                node: Covariate0,
                intercept: 1.0,
                parent_coefficients: vec![],
                noise_halfwidth: halfwidth,
            },
            NodeMechanism {
                node: Covariate1,
                intercept: 0.5,
                parent_coefficients: vec![(Covariate0, 0.8), (Action0, 1.5)],
                noise_halfwidth: halfwidth,
            },
            NodeMechanism {
                node: Outcome,
                intercept: -1.0,
                parent_coefficients: vec![
                    (Covariate0, 0.5),
                    (Action0, 0.7),
                    (Covariate1, 2.0),
                    (Action1, 1.2),
                ],
                noise_halfwidth: halfwidth,
            },
        ],
    }
}

fn unit(i: usize) -> UnitId {
    UnitId::new(format!("unit-{i}"))
}

fn spec() -> TemporalCounterfactualSpec {
    let factual = (0..4)
        .map(|i| FactualUnitHistory {
            unit: unit(i),
            history: HistoryId::new(format!("hist-{i}")),
            times: [0, 1],
            values: simulate(NOISE[i], FACTUAL_ACTIONS[i]),
        })
        .collect();
    let units: Vec<UnitId> = (0..4).map(unit).collect();
    TemporalCounterfactualSpec {
        graph: graph(),
        fit: fit(None),
        snapshot: SnapshotId::new("snapshot-1"),
        factual,
        plus: NamedActionHistory {
            name: "always_treat".to_owned(),
            times: [0, 1],
            actions: PLUS,
            units: units.clone(),
        },
        minus: NamedActionHistory {
            name: "never_treat".to_owned(),
            times: [0, 1],
            actions: MINUS,
            units,
        },
    }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

fn detail(error: &TemporalCounterfactualError) -> &str {
    &error.refusal().expect("a typed refusal").detail
}

#[test]
fn x8_temporal_shared_scm_matches_hand_abduction_and_replay() {
    let result = evaluate_temporal_counterfactual(&spec(), &ctx()).unwrap();
    assert_eq!(result.units.len(), 4);
    let (mut sum_plus, mut sum_minus) = (0.0, 0.0);
    for (i, row) in result.units.iter().enumerate() {
        let truth_plus = simulate(NOISE[i], PLUS)[4];
        let truth_minus = simulate(NOISE[i], MINUS)[4];
        assert!((row.plus_outcome - truth_plus).abs() < 1e-9, "unit {i} plus");
        assert!((row.minus_outcome - truth_minus).abs() < 1e-9, "unit {i} minus");
        assert!((row.factual_outcome - simulate(NOISE[i], FACTUAL_ACTIONS[i])[4]).abs() < 1e-9);
        sum_plus += truth_plus;
        sum_minus += truth_minus;
    }
    assert!((result.mean_plus - sum_plus / 4.0).abs() < 1e-9);
    assert!((result.mean_minus - sum_minus / 4.0).abs() < 1e-9);
    assert!((result.mean_contrast - (sum_plus - sum_minus) / 4.0).abs() < 1e-9);
    // The linear effect of (1,1) over (0,0): 0.7 + 1.2 + 2.0 * 1.5 = 4.9, and the
    // shared noise cancels in every unit's contrast.
    for row in &result.units {
        assert!(((row.plus_outcome - row.minus_outcome) - 4.9).abs() < 1e-9);
    }
    assert_eq!(result.receipt.n_worlds, 2);
    assert_eq!(result.receipt.horizon, 2);
    assert_eq!(result.receipt.n_units, 4);
    assert!(result.receipt.shared_by_both_worlds);
}

#[test]
fn x8_temporal_shared_scm_shared_draw_differs_from_independent_noise() {
    let result = evaluate_temporal_counterfactual(&spec(), &ctx()).unwrap();
    // Independent fresh noise per world: the minus world uses another unit's draw.
    let mut differing = 0;
    for (i, row) in result.units.iter().enumerate() {
        let fresh_minus = simulate(NOISE[(i + 1) % 4], MINUS)[4];
        if (row.minus_outcome - fresh_minus).abs() > 1e-3 {
            differing += 1;
        }
        // The shared-draw contrast is constant (4.9); the fresh-draw contrast is not.
        let fresh_unit_contrast = row.plus_outcome - fresh_minus;
        assert!((fresh_unit_contrast - 4.9).abs() > 1e-3, "unit {i} coupling must matter");
    }
    assert_eq!(differing, 4);
    // A single fresh draw reused for the minus world also moves the mean contrast.
    let independent_mean = result
        .units
        .iter()
        .map(|r| r.plus_outcome - simulate([0.9, 0.9, 0.9], MINUS)[4])
        .sum::<f64>()
        / 4.0;
    assert!((independent_mean - result.mean_contrast).abs() > 1e-3);
}

#[test]
fn x8_temporal_shared_scm_unit_order_does_not_change_the_answer() {
    let base = evaluate_temporal_counterfactual(&spec(), &ctx()).unwrap();
    let mut shuffled = spec();
    shuffled.factual.reverse();
    shuffled.plus.units.reverse();
    let again = evaluate_temporal_counterfactual(&shuffled, &ctx()).unwrap();
    assert_eq!(base.units, again.units);
    assert_eq!(base.receipt.digest, again.receipt.digest);
}

#[test]
fn x8_temporal_shared_scm_cancelled_context_stops() {
    let context = ctx();
    context.cancellation.cancel();
    let error = evaluate_temporal_counterfactual(&spec(), &context).unwrap_err();
    assert_eq!(error, TemporalCounterfactualError::Cancelled);
}

#[test]
fn x8_temporal_shared_scm_bounds_are_enforced() {
    let mut deep = spec();
    deep.graph.horizon = 3;
    let error = evaluate_temporal_counterfactual(&deep, &ctx()).unwrap_err();
    let refusal = error.refusal().unwrap();
    assert_eq!(refusal.code, "invalid_argument");
    assert_eq!(refusal.detail, "temporal_counterfactual.horizon_exceeded");

    let mut latent = spec();
    latent.graph.latent_confounding = true;
    let error = evaluate_temporal_counterfactual(&latent, &ctx()).unwrap_err();
    assert_eq!(error.refusal().unwrap().code, "route_not_supported");
    assert_eq!(detail(&error), "temporal_counterfactual.latent_confounding");

    let mut mismatched = spec();
    mismatched.fit.mechanisms[2].parent_coefficients.pop();
    let error = evaluate_temporal_counterfactual(&mismatched, &ctx()).unwrap_err();
    assert_eq!(detail(&error), "temporal_counterfactual.fit_mismatch");
}

#[test]
fn x8_unpaired_history_missing_unit_in_a_world_refuses_with_witness() {
    let mut s = spec();
    s.minus.units.retain(|u| *u != unit(2));
    let error = evaluate_temporal_counterfactual(&s, &ctx()).unwrap_err();
    let refusal = error.refusal().unwrap();
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, "temporal_counterfactual.unpaired_histories");
    assert_eq!(refusal.offending.as_deref(), Some("unit-2"));
    let witness = error.witness().unwrap();
    assert_eq!(witness.unit, Some(unit(2)));
    assert_eq!(witness.history, Some(HistoryId::new("hist-2")));
    assert_eq!(witness.world.as_deref(), Some("never_treat"));
}

#[test]
fn x8_unpaired_history_extra_world_unit_and_duplicate_factual_refuse() {
    let mut extra = spec();
    extra.plus.units.push(UnitId::new("stranger"));
    let error = evaluate_temporal_counterfactual(&extra, &ctx()).unwrap_err();
    assert_eq!(detail(&error), "temporal_counterfactual.unpaired_histories");
    let witness = error.witness().unwrap();
    assert_eq!(witness.unit, Some(UnitId::new("stranger")));
    assert_eq!(witness.world.as_deref(), Some("always_treat"));

    let mut duplicate = spec();
    let mut again = duplicate.factual[1].clone();
    again.history = HistoryId::new("hist-1-second");
    duplicate.factual.push(again);
    let error = evaluate_temporal_counterfactual(&duplicate, &ctx()).unwrap_err();
    assert_eq!(detail(&error), "temporal_counterfactual.unpaired_histories");
    assert_eq!(error.witness().unwrap().world.as_deref(), Some("factual"));
}

#[test]
fn x8_unpaired_history_refuting_witness_is_retained() {
    let mut s = spec();
    s.fit = fit(Some(1.5));
    // Unit 3's factual outcome carries a residual far outside the declared support.
    s.factual[3].values[4] += 5.0;
    let error = evaluate_temporal_counterfactual(&s, &ctx()).unwrap_err();
    let refusal = error.refusal().unwrap();
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, "temporal_counterfactual.refuting_history");
    let witness = error.witness().unwrap();
    assert_eq!(witness.unit, Some(unit(3)));
    assert_eq!(witness.history, Some(HistoryId::new("hist-3")));
    assert_eq!(witness.node, Some(Outcome));
    assert_eq!(witness.bound, Some(1.5));
    assert!((witness.residual.unwrap() - (NOISE[3][2] + 5.0)).abs() < 1e-9);
    // The same data is accepted without a declared support bound.
    let mut open = spec();
    open.factual[3].values[4] += 5.0;
    assert!(evaluate_temporal_counterfactual(&open, &ctx()).is_ok());
}

#[test]
fn x8_unpaired_history_time_misaligned_and_missing_histories_refuse() {
    let mut s = spec();
    s.plus.times = [0, 2];
    let error = evaluate_temporal_counterfactual(&s, &ctx()).unwrap_err();
    assert_eq!(detail(&error), "temporal_counterfactual.time_misaligned");
    assert_eq!(error.witness().unwrap().world.as_deref(), Some("always_treat"));

    let mut empty = spec();
    empty.factual.clear();
    let error = evaluate_temporal_counterfactual(&empty, &ctx()).unwrap_err();
    assert_eq!(error.refusal().unwrap().code, "route_not_supported");
    assert_eq!(detail(&error), "temporal_counterfactual.shared_history_missing");
}

#[test]
fn x8_temporal_world_replay_digests_recompute_bit_identically() {
    let a = evaluate_temporal_counterfactual(&spec(), &ctx()).unwrap();
    let b = evaluate_temporal_counterfactual(&spec(), &ctx()).unwrap();
    assert_eq!(a.receipt, b.receipt);
    assert_eq!(a.receipt.digest.len(), 32);
    assert!(a.receipt.is_self_consistent());
    assert_eq!(a.receipt.digest, a.receipt.recompute_digest());
    assert_eq!(a.receipt.unit_draws.len(), 4);
    let mut tampered = a.receipt.clone();
    tampered.unit_draws[0].digest = "0".repeat(32);
    assert!(!tampered.is_self_consistent());
}

#[test]
fn x8_temporal_world_replay_any_identity_change_changes_the_digest() {
    let base = evaluate_temporal_counterfactual(&spec(), &ctx()).unwrap().receipt.digest;
    let digest_of = |s: &TemporalCounterfactualSpec| {
        evaluate_temporal_counterfactual(s, &ctx()).unwrap().receipt.digest
    };
    let mut changes: Vec<TemporalCounterfactualSpec> = Vec::new();

    let mut snapshot = spec();
    snapshot.snapshot = SnapshotId::new("snapshot-2");
    changes.push(snapshot);

    let mut history_id = spec();
    history_id.factual[0].history = HistoryId::new("hist-0-renamed");
    changes.push(history_id);

    let mut unit_value = spec();
    unit_value.factual[1].values[2] += 0.25;
    changes.push(unit_value);

    let mut action_time = spec();
    for u in &mut action_time.factual {
        u.times = [0, 2];
    }
    action_time.plus.times = [0, 2];
    action_time.minus.times = [0, 2];
    changes.push(action_time);

    let mut action_value = spec();
    action_value.plus.actions = [1.0, 0.5];
    changes.push(action_value);

    let mut fit_identity = spec();
    fit_identity.fit.fit_id = "fit-other".to_owned();
    changes.push(fit_identity);

    let mut coefficient = spec();
    coefficient.fit.mechanisms[2].parent_coefficients[3].1 = 1.25;
    changes.push(coefficient);

    let mut graph_change = spec();
    // Drop an edge and its coefficient together.
    graph_change.graph.edges.retain(|e| *e != (Covariate0, Outcome));
    graph_change.fit.mechanisms[2].parent_coefficients.retain(|(p, _)| *p != Covariate0);
    changes.push(graph_change);

    let mut seen = vec![base.clone()];
    for changed in &changes {
        let digest = digest_of(changed);
        assert!(!seen.contains(&digest), "a changed identity must change the digest");
        seen.push(digest);
    }
}

fn factor(role: PopulationRole, regime: &str) -> RegimeFactorKey {
    RegimeFactorKey { role, regime: regime.to_owned() }
}

fn all_gates() -> TransportedCounterfactualPrerequisites {
    TransportedCounterfactualPrerequisites {
        transport_license: true,
        fixed_population_license: true,
        cross_population_assumptions: true,
    }
}

#[test]
fn x8_transported_gate_reports_exact_missing_gates_and_route_frozen() {
    let required = [factor(PopulationRole::Source, "obs"), factor(PopulationRole::Target, "obs")];
    let supplied: BTreeMap<RegimeFactorKey, String> =
        required.iter().cloned().map(|k| (k, "evidence".to_owned())).collect();

    let none = TransportedCounterfactualPrerequisites::default();
    let refusal = refuse_transported_counterfactual(&none, &required, &supplied);
    assert_eq!(
        refusal.missing_gates,
        vec![
            TransportedCounterfactualGate::TransportLicenseMissing,
            TransportedCounterfactualGate::FixedPopulationTemporalLicenseMissing,
            TransportedCounterfactualGate::CrossPopulationAssumptionsMissing,
        ]
    );
    assert_eq!(refusal.refusal.code, "cell_not_licensed");
    assert_eq!(refusal.refusal.detail, "transported_counterfactual.route_frozen");
    assert_eq!(refusal.refusal.offending.as_deref(), Some("transport_license_missing"));
    assert!(refusal.refusal.validate().is_ok());

    let neighbour_only = TransportedCounterfactualPrerequisites {
        transport_license: true,
        fixed_population_license: false,
        cross_population_assumptions: true,
    };
    let refusal = refuse_transported_counterfactual(&neighbour_only, &required, &supplied);
    assert_eq!(
        refusal.missing_gates,
        vec![TransportedCounterfactualGate::FixedPopulationTemporalLicenseMissing]
    );
    assert_eq!(refusal.refusal.detail, "transported_counterfactual.route_frozen");

    // Every prerequisite and factor present: the joint theorem has still not passed.
    let refusal = refuse_transported_counterfactual(&all_gates(), &required, &supplied);
    assert!(refusal.missing_gates.is_empty());
    assert!(refusal.missing_factors.is_empty());
    assert_eq!(refusal.refusal.code, "cell_not_licensed");
    assert_eq!(refusal.refusal.detail, "transported_counterfactual.route_frozen");
}

#[test]
fn x8_transported_gate_absent_regime_factor_is_factor_missing() {
    let required = [factor(PopulationRole::Source, "obs"), factor(PopulationRole::Target, "do_x")];
    let mut supplied: BTreeMap<RegimeFactorKey, String> = BTreeMap::new();
    supplied.insert(required[0].clone(), "evidence".to_owned());
    let refusal = refuse_transported_counterfactual(&all_gates(), &required, &supplied);
    assert_eq!(refusal.refusal.code, "transport_missing_evidence");
    assert_eq!(refusal.refusal.detail, "transported_counterfactual.factor_missing");
    assert_eq!(refusal.refusal.offending.as_deref(), Some("target:do_x"));
    assert_eq!(refusal.missing_factors, vec![required[1].clone()]);
    assert!(refusal.missing_gates.is_empty());
    assert!(refusal.refusal.validate().is_ok());

    // A missing gate takes precedence, with the missing factor still retained.
    let gated = TransportedCounterfactualPrerequisites {
        transport_license: false,
        ..all_gates()
    };
    let refusal = refuse_transported_counterfactual(&gated, &required, &supplied);
    assert_eq!(refusal.refusal.detail, "transported_counterfactual.route_frozen");
    assert_eq!(refusal.missing_factors, vec![required[1].clone()]);
}
