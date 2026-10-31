//! Maintained end-to-end decision-evidence workflow: a cluster-randomized experiment with a
//! limited number of independent clusters, three related outcomes, a nonlinear margin, and the
//! decision classified as robust, assumption-dependent or unresolved.
#![allow(clippy::cast_precision_loss, reason = "small deterministic fixtures")]

use antecedent::decision_evidence::*;

const ROWS_PER_CLUSTER: usize = 12;

/// Deterministic cluster-level shocks shared by every outcome of a cluster, so the outcomes
/// are dependent through the clusters (a cluster that buys more also buys larger baskets).
fn shock(cluster: usize, salt: f64) -> f64 {
    ((cluster as f64 + 1.0) * 12.9898 + salt).sin() * 0.5
}

struct Fixture {
    totals: [Vec<f64>; 3],
    assigned: Vec<bool>,
    rows: usize,
}

fn fixture(clusters: usize) -> Fixture {
    let assigned: Vec<bool> = (0..clusters).map(|c| c % 2 == 0).collect();
    let mut totals = [vec![], vec![], vec![]];
    for (c, treated) in assigned.iter().enumerate() {
        let u = shock(c, 0.0);
        let (mut f, mut v, mut k) = (0.0, 0.0, 0.0);
        for r in 0..ROWS_PER_CLUSTER {
            let row_noise = ((c * 31 + r * 17) as f64 * 0.7).sin() * 0.2;
            f += 2.0 + if *treated { 0.3 } else { 0.0 } + 0.6 * u + row_noise;
            v += 10.0 + if *treated { 1.0 } else { 0.0 } + 2.0 * u + 0.5 * row_noise;
            k += 6.0 + 0.1 * shock(c, 3.0);
        }
        totals[0].push(f);
        totals[1].push(v);
        totals[2].push(k);
    }
    Fixture { totals, assigned, rows: clusters * ROWS_PER_CLUSTER }
}

fn components() -> Vec<OutcomeComponent> {
    [("frequency", "orders"), ("value", "usd"), ("cost", "usd")]
        .iter()
        .map(|(n, u)| OutcomeComponent {
            name: (*n).into(),
            variable_id: format!("outcome.{n}"),
            units: (*u).into(),
        })
        .collect()
}

fn evidence(clusters: usize) -> JointExperimentEvidence {
    let fx = fixture(clusters);
    let slices: Vec<&[f64]> = fx.totals.iter().map(Vec::as_slice).collect();
    let contrast = complete_cluster_joint_contrast(&slices, &fx.assigned, fx.rows).unwrap();
    evidence_from_cluster_contrast(
        &contrast,
        "exp-shipping-2026q3",
        "active_customers",
        ("free_shipping", "standard_shipping"),
        4,
        components(),
        fx.rows,
        vec![],
        Exposure::NoBetweenUnitInterference {
            assumption_id: "assumption.no_cross_customer_spillover".into(),
        },
    )
    .unwrap()
}

fn request() -> Request {
    Request {
        population_id: "active_customers".into(),
        treated_regime_id: "free_shipping".into(),
        control_regime_id: "standard_shipping".into(),
        horizon: 4,
        target: Target::ExperimentalContrast,
    }
}

/// margin per customer = orders * (basket value - unit cost)
fn margin(cost: InputBinding) -> DerivedQuantity {
    DerivedQuantity {
        name: "margin_per_customer".into(),
        units: "usd".into(),
        expr: UtilityExpr::product(
            UtilityExpr::Input(0),
            UtilityExpr::difference(UtilityExpr::Input(1), UtilityExpr::Input(2)),
        ),
        inputs: vec![
            InputBinding::Measured("frequency".into()),
            InputBinding::Measured("value".into()),
            cost,
        ],
    }
}

fn frame(threshold: f64) -> DecisionFrame {
    DecisionFrame {
        treated_action: "launch_free_shipping".into(),
        control_action: "keep_standard".into(),
        threshold,
    }
}

#[test]
fn threshold_agrees_with_the_estimator() {
    assert_eq!(MIN_UNITS_PER_ARM_FOR_INTERVAL, 30);
    let fx = fixture(60);
    let slices: Vec<&[f64]> = fx.totals.iter().map(Vec::as_slice).collect();
    assert!(
        complete_cluster_joint_contrast(&slices, &fx.assigned, fx.rows).unwrap().interval_supported
    );
    let fx = fixture(40);
    let slices: Vec<&[f64]> = fx.totals.iter().map(Vec::as_slice).collect();
    assert!(
        !complete_cluster_joint_contrast(&slices, &fx.assigned, fx.rows)
            .unwrap()
            .interval_supported
    );
}

#[test]
fn nonlinear_margin_from_a_clustered_experiment_keeps_dependence_and_inherits_no_license() {
    let ev = evidence(80);
    let q = margin(InputBinding::Measured("cost".into()));
    let r = derive(&ev, &request(), &q).unwrap();
    let s = r.sampling.as_ref().expect("40 clusters per arm support an interval");
    assert!(r.contrast.unwrap() > 0.0 && s.standard_error > 0.0);
    assert_eq!((r.license, r.calibration), ("none", "unmeasured"));

    // Dependence is retained: zeroing the cross-component covariance changes the answer.
    let mut independent = ev.clone();
    if let Dependence::Joint { treated, control } = &mut independent.dependence {
        for m in [treated, control] {
            for i in 0..3 {
                for j in 0..3 {
                    if i != j {
                        m[i * 3 + j] = 0.0;
                    }
                }
            }
        }
    }
    let naive = derive(&independent, &request(), &q).unwrap().sampling.unwrap().standard_error;
    assert!((naive - s.standard_error).abs() > 1e-4, "{naive} vs {}", s.standard_error);

    // Marginals alone cannot answer a question about the margin.
    let mut marginals = ev.clone();
    if let Dependence::Joint { treated, control } = &ev.dependence {
        marginals.dependence = Dependence::MarginalsOnly {
            treated: (0..3).map(|i| treated[i * 3 + i]).collect(),
            control: (0..3).map(|i| control[i * 3 + i]).collect(),
        };
    }
    assert_eq!(
        derive(&marginals, &request(), &q).unwrap().unresolved,
        vec![UnresolvedReason::JointLawMissing]
    );
}

#[test]
fn the_decision_is_robust_assumption_dependent_or_unresolved_and_the_reasons_are_named() {
    let ev = evidence(80);
    let measured = margin(InputBinding::Measured("cost".into()));
    let assumed = margin(InputBinding::Assumed {
        lower: 4.0,
        upper: 12.0,
        assumption_id: "assumption.unit_cost_range".into(),
        shared_across_arms: true,
    });

    let robust = decide(&ev, &request(), &measured, &frame(0.0)).unwrap();
    assert_eq!(robust.verdict, Verdict::Robust { action: "launch_free_shipping".into() });

    // Unit cost was not measured: the range [4, 12] moves the contrast from about 3.5 to 1.1.
    let dependent = decide(&ev, &request(), &assumed, &frame(2.5)).unwrap();
    let Verdict::AssumptionDependent { assumption_ids, at_lower, at_upper } = &dependent.verdict
    else {
        panic!("{:?}", dependent.verdict);
    };
    assert_eq!(assumption_ids, &["assumption.unit_cost_range".to_owned()]);
    assert_ne!(at_lower, at_upper);
    let range = dependent.derived.assumption_range.as_ref().unwrap();
    assert!(range.exact);
    assert!(
        dependent.derived.sampling.is_some(),
        "sampling and assumption range are reported apart"
    );

    // Few independent clusters, many rows: no interval, however many rows there are.
    let few = evidence(24);
    let unresolved = decide(&few, &request(), &measured, &frame(0.0)).unwrap();
    let Verdict::Unresolved { reasons } = &unresolved.verdict else {
        panic!("{:?}", unresolved.verdict)
    };
    assert!(matches!(
        reasons[0],
        UnresolvedReason::TooFewIndependentUnits { treated: 12, control: 12, required: 30 }
    ));

    // Unit cost missing altogether.
    let missing = decide(
        &ev,
        &request(),
        &margin(InputBinding::Missing { what: "unit cost".into() }),
        &frame(0.0),
    )
    .unwrap();
    assert!(matches!(missing.verdict, Verdict::Unresolved { .. }));
}

#[test]
fn out_of_scope_requests_are_typed_refusals_and_deployment_needs_a_bridge() {
    let ev = evidence(80);
    let q = margin(InputBinding::Measured("cost".into()));
    let mut r = request();
    r.population_id = "all_customers".into();
    let e = derive(&ev, &r, &q).unwrap_err();
    assert_eq!(e.reason_code(), "decision_evidence.incompatible_population");
    let mut r = request();
    r.target = Target::Deployment { bridge_assumption_id: None };
    assert_eq!(derive(&ev, &r, &q).unwrap_err(), EvidenceRefusal::DeploymentBridgeMissing);
    r.target = Target::Deployment {
        bridge_assumption_id: Some("assumption.rollout_matches_trial".into()),
    };
    let bridged = derive(&ev, &r, &q).unwrap();
    assert!(bridged.sampling.is_some());
}

#[test]
fn planning_reuse_and_withdrawal_follow_the_decision() {
    let ev = evidence(80);
    let assumed = margin(InputBinding::Assumed {
        lower: 4.0,
        upper: 12.0,
        assumption_id: "assumption.unit_cost_range".into(),
        shared_across_arms: true,
    });
    let bar = frame(2.5);
    let plan = plan_evidence(
        &ev,
        &request(),
        &assumed,
        &bar,
        &[
            Candidate::ResolveAssumption {
                id: "audit_unit_cost".into(),
                assumption_id: "assumption.unit_cost_range".into(),
                narrowed: (5.5, 6.5),
                cost: None,
            },
            Candidate::MoreUnits {
                id: "double_clusters".into(),
                additional_units_per_arm: 40,
                cost: Some(0.01),
            },
        ],
    )
    .unwrap();
    // Pinning unit cost alone does not settle this decision: sampling uncertainty over so few
    // clusters still straddles the bar, so the plan says the bottleneck is clusters, not cost.
    assert!(
        matches!(
            &plan.resolution_rows[0].value,
            PlanValue::Resolution {
                resolves: false,
                verdict: Verdict::Unresolved { reasons },
            } if reasons == &[UnresolvedReason::SamplingStraddlesThreshold]
        ),
        "{:?}",
        plan.resolution_rows[0].value
    );
    let PlanValue::Evsi { gross, .. } = plan.evsi_rows[0].value else { panic!() };
    assert!(gross > 0.0);

    // A decision-rule change reuses the estimate and the derived quantity.
    let before = StageDigests::of(&ev, &assumed, &bar);
    let after = StageDigests::of(&ev, &assumed, &frame(3.0));
    let receipt = reuse_receipt(&before, &after);
    assert_eq!(receipt.status(Stage::Derived), StageStatus::Reused);
    assert_eq!(receipt.status(Stage::Decision), StageStatus::Invalidated);

    // Withdrawing the experiment withdraws the decision's support.
    let mut ledger = EvidenceLedger::new();
    ledger.record(&ev.evidence_id);
    assert!(ledger.unsupported_parts(&ev.evidence_id).is_empty());
    ledger.withdraw(&ev.evidence_id, "assignment log found corrupted");
    assert_eq!(ledger.unsupported_parts(&ev.evidence_id).len(), 1);
}

#[test]
fn centered_adjustment_and_arm_spillover_compose_for_a_nonlinear_decision() {
    let mut fx = fixture(80);
    let sizes = vec![ROWS_PER_CLUSTER; fx.assigned.len()];
    let pre: Vec<f64> =
        (0..sizes.len()).map(|c| ROWS_PER_CLUSTER as f64 * (1.0 + shock(c, 0.0))).collect();
    fx.totals[0] = centered_fixed_cuped_totals(&fx.totals[0], &pre, &sizes, 0.4).unwrap();
    let slices: Vec<&[f64]> = fx.totals.iter().map(Vec::as_slice).collect();
    let contrast = complete_cluster_joint_contrast(&slices, &fx.assigned, fx.rows).unwrap();
    let ev = evidence_from_cluster_contrast(
        &contrast,
        "exp-shipping-centered",
        "active_customers",
        ("free_shipping", "standard_shipping"),
        4,
        components(),
        fx.rows,
        vec![ComponentAdjustment {
            component: "frequency".into(),
            kind: AdjustmentKind::CenteredFixedCuped {
                covariate: "pre_frequency".into(),
                coefficient: 0.4,
            },
        }],
        Exposure::BoundedArmSpillover {
            assumption_id: "arm-spillover".into(),
            treated_bias_bounds: vec![0.1, 0.2, 0.0],
            control_bias_bounds: vec![0.1, 0.2, 0.0],
        },
    )
    .unwrap();
    let answer =
        decide(&ev, &request(), &margin(InputBinding::Measured("cost".into())), &frame(0.0))
            .unwrap();
    assert!(answer.derived.sampling.is_some());
    let range = answer.derived.assumption_range.unwrap();
    assert!(
        range.lower < answer.derived.contrast.unwrap()
            && range.upper > answer.derived.contrast.unwrap()
    );
    assert_eq!(range.assumption_ids, vec!["arm-spillover"]);
}
