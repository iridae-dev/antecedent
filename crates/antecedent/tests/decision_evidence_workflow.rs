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
    // clusters still straddles the bar. More-units EVSI is refused while the cost range remains.
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
    assert!(matches!(plan.evsi_rows[0].value, PlanValue::Refused(_)));

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

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the test keeps its existing bounded fixture and exact scientific assertions together"
)]
fn matched_pairs_preserve_arm_dependence_and_improve_precision() {
    let pairs = 40;
    let design = compile_matched_pair_design(MatchedPairDesign {
        cluster_ids: (0..2 * pairs).map(|i| format!("cluster-{i}")).collect(),
        pairs: (0..pairs).map(|i| (2 * i, 2 * i + 1)).collect(),
        matching_variables: vec!["pretreatment_level".into()],
        treatment_probabilities: vec![0.5; 2 * pairs],
        cluster_sizes: vec![1; 2 * pairs],
        preassignment_assumption_id: "pairing-fixed".into(),
        independent_pairs_assumption_id: "independent-pairs".into(),
        consistency_assumption_id: "consistency".into(),
    })
    .unwrap();
    let assigned: Vec<bool> = (0..2 * pairs).map(|i| i % 2 == 0).collect();
    let totals: Vec<f64> = (0..2 * pairs)
        .map(|i| {
            (i / 2) as f64 + if assigned[i] { 1.0 + 0.01 * ((i / 2) as f64).sin() } else { 0.0 }
        })
        .collect();
    let ev = evidence_from_matched_pairs(
        &design,
        &[&totals],
        &assigned,
        "matched-trial",
        "pop",
        ("t", "c"),
        1,
        vec![OutcomeComponent {
            name: "outcome".into(),
            variable_id: "v".into(),
            units: "u".into(),
        }],
        Exposure::NoBetweenUnitInterference { assumption_id: "no-spill".into() },
    )
    .unwrap();
    let request = Request {
        population_id: "pop".into(),
        treated_regime_id: "t".into(),
        control_regime_id: "c".into(),
        horizon: 1,
        target: Target::ExperimentalContrast,
    };
    let q = DerivedQuantity {
        name: "effect".into(),
        units: "u".into(),
        expr: UtilityExpr::Input(0),
        inputs: vec![InputBinding::Measured("outcome".into())],
    };
    let result = derive(&ev, &request, &q).unwrap();
    let truth = 1.0 + (0..pairs).map(|i| 0.01 * (i as f64).sin()).sum::<f64>() / pairs as f64;
    assert!((result.contrast.unwrap() - truth).abs() < 1e-10);
    assert!(result.sampling.unwrap().standard_error < 0.01);
    let constant: Vec<f64> =
        (0..2 * pairs).map(|i| (i / 2) as f64 + if assigned[i] { 1.0 } else { 0.0 }).collect();
    let unstable = evidence_from_matched_pairs(
        &design,
        &[&constant],
        &assigned,
        "constant",
        "pop",
        ("t", "c"),
        1,
        vec![OutcomeComponent {
            name: "outcome".into(),
            variable_id: "v".into(),
            units: "u".into(),
        }],
        Exposure::NoBetweenUnitInterference { assumption_id: "no-spill".into() },
    )
    .unwrap();
    assert!(matches!(
        derive(&unstable, &request, &q).unwrap().unresolved.as_slice(),
        [UnresolvedReason::SamplingUnavailable]
    ));
    assert_eq!(ev.design.treated_units, pairs);
    let mut changed_assignment = ev.clone();
    if let Some(DesignDetails::MatchedPairs { realized_assignment, .. }) =
        &mut changed_assignment.design_details
    {
        realized_assignment.swap(0, 1);
    }
    assert_eq!(ev.scientific_identity(), changed_assignment.scientific_identity());
    assert_ne!(ev.estimate_identity(), changed_assignment.estimate_identity());
    let mut changed_matching = ev.clone();
    if let Some(DesignDetails::MatchedPairs { matching_variables, .. }) =
        &mut changed_matching.design_details
    {
        matching_variables.push("new-premise".into());
    }
    assert_ne!(ev.scientific_identity(), changed_matching.scientific_identity());
    let independent = complete_cluster_joint_contrast(&[&totals], &assigned, 2 * pairs).unwrap();
    assert!(independent.effect_covariance[0] > 1.0);
    let mut few_design = design.design.clone();
    few_design.cluster_ids.truncate(58);
    few_design.pairs.truncate(29);
    few_design.treatment_probabilities.truncate(58);
    few_design.cluster_sizes = vec![1_000; 58];
    let few_design = compile_matched_pair_design(few_design).unwrap();
    let few_totals: Vec<f64> = totals[..58].iter().map(|v| v * 1_000.0).collect();
    let few = evidence_from_matched_pairs(
        &few_design,
        &[&few_totals],
        &assigned[..58],
        "few",
        "pop",
        ("t", "c"),
        1,
        vec![OutcomeComponent {
            name: "outcome".into(),
            variable_id: "v".into(),
            units: "u".into(),
        }],
        Exposure::NoBetweenUnitInterference { assumption_id: "no-spill".into() },
    )
    .unwrap();
    assert!(matches!(
        derive(&few, &request, &q).unwrap().unresolved.as_slice(),
        [UnresolvedReason::TooFewIndependentUnits { .. }]
    ));
    let mut inconsistent = ev.clone();
    inconsistent.design.treated_units += 1;
    assert!(derive(&inconsistent, &request, &q).is_err());
    let mut bad = assigned.clone();
    bad[1] = true;
    assert!(
        evidence_from_matched_pairs(
            &design,
            &[&totals],
            &bad,
            "e",
            "pop",
            ("t", "c"),
            1,
            vec![OutcomeComponent {
                name: "outcome".into(),
                variable_id: "v".into(),
                units: "u".into()
            }],
            Exposure::NoBetweenUnitInterference { assumption_id: "a".into() }
        )
        .is_err()
    );
}

#[test]
fn proxy_bridge_requires_a_scoped_effect_assumption_and_retains_lineage() {
    let ev = evidence(80);
    let request = request();
    let bridge = ProxyBridge {
        proxy_component: "frequency".into(),
        target_variable_id: "profit_effect".into(),
        target_units: "currency".into(),
        population_id: request.population_id.clone(),
        treated_regime_id: request.treated_regime_id.clone(),
        control_regime_id: request.control_regime_id.clone(),
        horizon: request.horizon,
        mapping: ProxyMapping::CausalEffectBounds {
            multiplier: (1.0, 1.5),
            residual: (0.0, 0.0),
            assumption_id: "proxy-causal-map".into(),
            mapping_evidence_id: "proxy-validation".into(),
        },
    };
    let result = decide_with_proxy(&ev, &request, &bridge, &frame(0.0)).unwrap();
    assert!(matches!(result.verdict, Verdict::Robust { .. }));
    assert_eq!(result.evidence_ids, vec!["exp-shipping-2026q3", "proxy-validation"]);
    assert!(result.sampling_envelope.is_some());
    let mut ledger = EvidenceLedger::new();
    ledger.record("exp-shipping-2026q3");
    ledger.record("proxy-validation");
    assert!(result.unsupported_evidence(&ledger).is_empty());
    assert!(ledger.withdraw("proxy-validation", "mapping withdrawn"));
    assert_eq!(result.unsupported_evidence(&ledger)[0].0, "proxy-validation");
    let mut association = bridge.clone();
    association.mapping = ProxyMapping::PredictiveAssociation;
    assert!(decide_with_proxy(&ev, &request, &association, &frame(0.0)).is_err());
    let mut unsupported = bridge.clone();
    unsupported.population_id = "other".into();
    assert!(decide_with_proxy(&ev, &request, &unsupported, &frame(0.0)).is_err());
    let mut wide = bridge.clone();
    wide.mapping = ProxyMapping::CausalEffectBounds {
        multiplier: (-2.0, 2.0),
        residual: (-1.0, 1.0),
        assumption_id: "weak".into(),
        mapping_evidence_id: "validation".into(),
    };
    assert!(matches!(
        decide_with_proxy(&ev, &request, &wide, &frame(0.0)).unwrap().verdict,
        Verdict::AssumptionDependent { .. }
    ));
    assert!(matches!(
        decide_with_proxy(&evidence(24), &request, &bridge, &frame(0.0)).unwrap().verdict,
        Verdict::Unresolved { .. }
    ));
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the test keeps its existing bounded fixture and exact scientific assertions together"
)]
fn switchback_uses_independent_sequences_and_preserves_period_identity() {
    let nseq = 40;
    let periods = 4;
    let design = compile_joint_switchback(JointSwitchbackDesign {
        sequence_ids: (0..nseq)
            .flat_map(|s| (0..periods).map(move |_| format!("seq-{s}")))
            .collect(),
        period_ids: (0..nseq).flat_map(|_| (0..periods).map(|p| format!("period-{p}"))).collect(),
        treatment_probabilities: vec![0.5; nseq * periods],
        no_carryover_assumption_id: "carryover-zero".into(),
        washout_rule_id: "washout-zero".into(),
        washout_periods: 0,
        independent_sequences_assumption_id: "sequences-independent".into(),
        consistency_assumption_id: "consistency".into(),
    })
    .unwrap();
    let assigned: Vec<bool> =
        (0..nseq * periods).map(|i| (i + (i / periods) % 3) % 2 == 0).collect();
    let y0: Vec<f64> = (0..nseq * periods)
        .map(|i| {
            2.0 + (i / periods) as f64 * 0.02
                + if assigned[i] { 0.3 + 0.02 * ((i / periods) as f64).sin() } else { 0.0 }
        })
        .collect();
    let y1: Vec<f64> = (0..nseq * periods)
        .map(|i| 4.0 + (i / periods) as f64 * 0.01 + if assigned[i] { 0.5 } else { 0.0 })
        .collect();
    let components = ["frequency", "value"]
        .into_iter()
        .map(|n| OutcomeComponent { name: n.into(), variable_id: n.into(), units: "u".into() })
        .collect();
    let ev = evidence_from_joint_switchback(
        &design,
        &[&y0, &y1],
        &assigned,
        "switchback",
        "pop",
        ("t", "c"),
        1,
        components,
        Exposure::NoBetweenUnitInterference { assumption_id: "no-interference".into() },
    )
    .unwrap();
    assert_eq!(ev.design.treated_units, nseq);
    let Dependence::JointArms { covariance } = &ev.dependence else {
        panic!("joint arms required")
    };
    // Independent arithmetic reference: sequence-level HT contrast scores and their sample
    // cross covariance. This checks the polarization-based joint estimator, not only a diagonal.
    let score = |y: &[f64], s: usize| {
        (0..periods)
            .map(|p| {
                let i = s * periods + p;
                if assigned[i] { 2.0 * y[i] } else { -2.0 * y[i] }
            })
            .sum::<f64>()
            / periods as f64
    };
    let scores0: Vec<_> = (0..nseq).map(|s| score(&y0, s)).collect();
    let scores1: Vec<_> = (0..nseq).map(|s| score(&y1, s)).collect();
    let mean0 = scores0.iter().sum::<f64>() / nseq as f64;
    let mean1 = scores1.iter().sum::<f64>() / nseq as f64;
    let reference =
        scores0.iter().zip(&scores1).map(|(a, b)| (a - mean0) * (b - mean1)).sum::<f64>()
            / (nseq * (nseq - 1)) as f64;
    let actual = covariance[1] - covariance[3] - covariance[9] + covariance[11];
    assert!((reference - actual).abs() < 1e-10);
    let req = Request {
        population_id: "pop".into(),
        treated_regime_id: "t".into(),
        control_regime_id: "c".into(),
        horizon: 1,
        target: Target::ExperimentalContrast,
    };
    let q = DerivedQuantity {
        name: "margin".into(),
        units: "u".into(),
        expr: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
        inputs: vec![
            InputBinding::Measured("frequency".into()),
            InputBinding::Measured("value".into()),
        ],
    };
    assert!(derive(&ev, &req, &q).unwrap().sampling.is_some());
    let single = DerivedQuantity {
        name: "frequency-effect".into(),
        units: "u".into(),
        expr: UtilityExpr::Input(0),
        inputs: vec![InputBinding::Measured("frequency".into())],
    };
    let joint = derive(&ev, &req, &single).unwrap();
    let ids: Vec<&str> = design.design.sequence_ids.iter().map(String::as_str).collect();
    let marginal = antecedent_estimate::switchback::switchback_itt(
        &y0,
        &assigned,
        &design.design.treatment_probabilities,
        &ids,
    )
    .unwrap();
    assert!((joint.contrast.unwrap() - marginal.effect).abs() < 1e-10);
    let interval = joint.sampling.unwrap();
    assert!((interval.standard_error.powi(2) - marginal.variance).abs() < 1e-10);
    let marginal_interval = marginal.interval_95.unwrap();
    assert!((interval.lower - marginal_interval[0]).abs() < 1e-10);
    assert!((interval.upper - marginal_interval[1]).abs() < 1e-10);
    let mut bad = design.design.clone();
    bad.period_ids[1] = bad.period_ids[0].clone();
    assert!(compile_joint_switchback(bad).is_err());
    let mut nonzero_washout = design.design.clone();
    nonzero_washout.washout_periods = 1;
    assert!(compile_joint_switchback(nonzero_washout).is_err());
    let mut sparse = design.design.clone();
    sparse.sequence_ids.truncate(20 * periods);
    sparse.period_ids.truncate(20 * periods);
    sparse.treatment_probabilities.truncate(20 * periods);
    assert!(compile_joint_switchback(sparse).is_err());
}

#[test]
#[allow(
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    reason = "the test keeps its existing bounded fixture and exact scientific assertions together"
)]
fn joint_lagged_excursion_reuses_existing_scores_and_keeps_estimand_scope() {
    use antecedent_estimate::temporal_design::contrasts::two_arm_scores;
    use antecedent_estimate::temporal_design::{
        ClusterScores, PanelRows, lagged_pairs, sampling_summary,
    };
    let units = 40;
    let periods = 5;
    let n = units * periods;
    let design = JointExcursionDesign {
        unit_ids: (0..units).flat_map(|u| (0..periods).map(move |_| format!("geo-{u}"))).collect(),
        periods: (0..units).flat_map(|_| (0..periods).map(|p| p as i64)).collect(),
        treatment_probabilities: vec![0.5; n],
        lag: 1,
        min_probability: 0.1,
        assignment_law_id: "weekly-randomization".into(),
        independent_units_assumption_id: "geos-independent".into(),
        administrative_followup_assumption_id: "end-of-study".into(),
        consistency_assumption_id: "consistency".into(),
    };
    let prepared = compile_joint_excursion(design.clone()).unwrap();
    assert_eq!(prepared.paired_decisions, units * (periods - 1));
    let assigned: Vec<bool> = (0..n).map(|i| ((i * 73 + i / periods * 19) % 11) < 5).collect();
    // Treatment at t affects t+1; the lagged excursion explicitly allows this carryover.
    let y: Vec<f64> = (0..n)
        .map(|i| {
            2.0 + 0.2 * ((i / periods) as f64 * 0.17).sin()
                + if i % periods > 0 && assigned[i - 1] { 1.0 } else { 0.0 }
        })
        .collect();
    let z: Vec<f64> = (0..n)
        .map(|i| {
            5.0 + 0.4 * ((i / periods) as f64 * 0.17).sin()
                + if i % periods > 0 && assigned[i - 1] { 0.6 } else { 0.0 }
        })
        .collect();
    let components = ["frequency", "value"]
        .into_iter()
        .map(|v| OutcomeComponent { name: v.into(), variable_id: v.into(), units: "u".into() })
        .collect();
    let ev = evidence_from_joint_excursion(
        &prepared,
        &[&y, &z],
        &assigned,
        "pulse-study",
        "pop",
        ("on", "off"),
        4,
        components,
        Exposure::NoBetweenUnitInterference { assumption_id: "no-spill".into() },
    )
    .unwrap();
    assert_eq!(ev.design.treated_units, units);
    let req = Request {
        population_id: "pop".into(),
        treated_regime_id: "on".into(),
        control_regime_id: "off".into(),
        horizon: 4,
        target: Target::Excursion { lag: 1, assignment_law_id: "weekly-randomization".into() },
    };
    let q = DerivedQuantity {
        name: "effect".into(),
        units: "u".into(),
        expr: UtilityExpr::Input(0),
        inputs: vec![InputBinding::Measured("frequency".into())],
    };
    let derived = derive(&ev, &req, &q).unwrap();
    let dense: Vec<usize> = (0..n).map(|i| i / periods).collect();
    let pairs = lagged_pairs(PanelRows { unit: &dense, period: &design.periods }, 1).unwrap();
    let scores =
        two_arm_scores(&pairs, &y, &assigned, &design.treatment_probabilities, &dense, 0.1)
            .unwrap();
    let summary = sampling_summary(
        &ClusterScores::accumulate(&scores.scores, 3, &scores.cluster_of_pair).unwrap(),
    )
    .unwrap();
    assert!((derived.contrast.unwrap() - summary.estimate[0]).abs() < 1e-10);
    assert!((derived.sampling.unwrap().standard_error - summary.standard_error[0]).abs() < 1e-10);
    // Independent arithmetic reference: reconstruct HT arm means and cluster residuals
    // directly from decision and outcome rows, including cross-outcome covariance.
    let mut sums = vec![[0.0; 4]; units];
    for t in 0..n {
        if t % periods + 1 < periods {
            let g = t / periods;
            let o = t + 1;
            let a = assigned[t];
            if a {
                sums[g][0] += 2.0 * y[o];
                sums[g][1] += 2.0 * z[o];
            } else {
                sums[g][2] += 2.0 * y[o];
                sums[g][3] += 2.0 * z[o];
            }
        }
    }
    let m: Vec<f64> = (0..4)
        .map(|j| sums.iter().map(|v| v[j]).sum::<f64>() / prepared.paired_decisions as f64)
        .collect();
    let cross = sums
        .iter()
        .map(|v| (v[0] - (periods - 1) as f64 * m[0]) * (v[1] - (periods - 1) as f64 * m[1]))
        .sum::<f64>()
        * units as f64
        / ((units - 1) as f64 * (prepared.paired_decisions as f64).powi(2));
    let Dependence::JointArms { covariance } = &ev.dependence else {
        panic!("joint covariance required")
    };
    assert!((covariance[1] - cross).abs() < 1e-10);
    assert!((derived.contrast.unwrap() - (m[0] - m[2])).abs() < 1e-10);
    let nonlinear = DerivedQuantity {
        name: "net".into(),
        units: "u".into(),
        expr: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
        inputs: vec![
            InputBinding::Measured("frequency".into()),
            InputBinding::Measured("value".into()),
        ],
    };
    assert!(derive(&ev, &req, &nonlinear).unwrap().sampling.is_some());
    // Independent normal reference at the decision threshold: the posterior-mean
    // standard deviation is se*sqrt(m/(n+m)), and E[max(0,Z)] = sd/sqrt(2*pi).
    let sampled = derive(&ev, &req, &q).unwrap().sampling.unwrap();
    let plan = plan_evidence(
        &ev,
        &req,
        &q,
        &frame(sampled.estimate),
        &[
            Candidate::MoreIndependentUnits {
                id: "more-geos".into(),
                additional_units: 40,
                same_design_assumption_id: "same-schedule-and-variance".into(),
                cost: Some(0.0),
            },
            Candidate::MoreIndependentUnits {
                id: "missing-assumption".into(),
                additional_units: 40,
                same_design_assumption_id: " ".into(),
                cost: None,
            },
            Candidate::MoreUnits {
                id: "wrong-unit".into(),
                additional_units_per_arm: 40,
                cost: None,
            },
        ],
    )
    .unwrap();
    let row = plan.evsi_rows.iter().find(|r| r.id == "more-geos").unwrap();
    assert_eq!(row.planning_assumption_id.as_deref(), Some("same-schedule-and-variance"));
    let PlanValue::Evsi { gross, .. } = row.value else { panic!("expected EVSI") };
    let expected =
        sampled.standard_error * (40.0_f64 / 80.0).sqrt() / (2.0 * std::f64::consts::PI).sqrt();
    assert!((gross - expected).abs() < 1e-10);
    assert!(
        plan.evsi_rows
            .iter()
            .filter(|r| r.id != "more-geos")
            .all(|r| matches!(r.value, PlanValue::Refused(_)))
    );
    let bridge = TemporalEffectBridge {
        source: TemporalEstimand::Excursion {
            lag: 1,
            assignment_law_id: "weekly-randomization".into(),
        },
        target: TemporalEstimand::FixedSchedule {
            active_schedule_id: "sustained-program".into(),
            reference_schedule_id: "status-quo".into(),
            horizon: 12,
        },
        source_component: "frequency".into(),
        target_variable_id: "frequency".into(),
        target_units: "u".into(),
        mapping: ProxyMapping::CausalEffectBounds {
            multiplier: (0.8, 1.2),
            residual: (-0.1, 0.1),
            assumption_id: "pulse-to-schedule".into(),
            mapping_evidence_id: "bridge-study".into(),
        },
    };
    let bridged = decide_with_temporal_bridge(&ev, &req, &bridge, &frame(0.0)).unwrap();
    assert!(bridged.decision.sampling_envelope.is_some());
    assert_eq!(bridged.decision.evidence_ids, vec!["pulse-study", "bridge-study"]);
    let mut unsupported = bridge.clone();
    unsupported.mapping = ProxyMapping::PredictiveAssociation;
    assert!(decide_with_temporal_bridge(&ev, &req, &unsupported, &frame(0.0)).is_err());
    let mut wrong = bridge;
    wrong.source =
        TemporalEstimand::Excursion { lag: 2, assignment_law_id: "weekly-randomization".into() };
    assert!(decide_with_temporal_bridge(&ev, &req, &wrong, &frame(0.0)).is_err());
    let mut bad = req.clone();
    bad.target = Target::ExperimentalContrast;
    assert!(derive(&ev, &bad, &q).is_err());
    bad.target = Target::Excursion { lag: 2, assignment_law_id: "weekly-randomization".into() };
    assert!(derive(&ev, &bad, &q).is_err());
    bad.target = Target::Deployment { bridge_assumption_id: Some("unjustified".into()) };
    assert!(derive(&ev, &bad, &q).is_err());
    let mut gap = design.clone();
    gap.periods[1] = 99;
    assert!(compile_joint_excursion(gap).is_err());
    let mut low = design;
    let take = 20 * periods;
    low.unit_ids.truncate(take);
    low.periods.truncate(take);
    low.treatment_probabilities.truncate(take);
    let low = compile_joint_excursion(low).unwrap();
    let ev = evidence_from_joint_excursion(
        &low,
        &[&y[..take]],
        &assigned[..take],
        "small",
        "pop",
        ("on", "off"),
        4,
        vec![OutcomeComponent {
            name: "frequency".into(),
            variable_id: "frequency".into(),
            units: "u".into(),
        }],
        Exposure::NoBetweenUnitInterference { assumption_id: "no-spill".into() },
    )
    .unwrap();
    assert!(matches!(
        derive(&ev, &req, &q).unwrap().unresolved.as_slice(),
        [UnresolvedReason::TooFewIndependentUnits { .. }]
    ));
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the test keeps its existing bounded fixture and exact scientific assertions together"
)]
fn observational_whole_regime_reuses_gformula_and_keeps_missing_joint_uncertainty() {
    use antecedent_estimate::temporal_design::{
        dense_labels,
        regime::{RegimeMethod, RegimePanel, RegimeRows, StaticRegimes, regime_point_estimate},
    };
    let units = 120;
    let periods = 2;
    let n = units * periods;
    let unit_ids: Vec<String> =
        (0..units).flat_map(|u| (0..periods).map(move |_| format!("u-{u}"))).collect();
    let period: Vec<i64> = (0..units).flat_map(|_| [0, 1]).collect();
    let assignment: Vec<bool> = (0..units).flat_map(|u| [u % 2 == 0, (u / 2) % 2 == 0]).collect();
    let confounder: Vec<f64> = (0..n)
        .map(|i| ((i / periods) as f64 * (0.13 + 0.04 * (i % periods) as f64)).sin())
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let u = i / periods;
            1.0 + 0.5 * confounder[i]
                + 0.03 * ((u * 17 + i % periods * 7) % 13) as f64
                + if assignment[u * 2] { 1.0 } else { 0.0 }
                + if i % periods == 1 && assignment[u * 2 + 1] { 2.0 } else { 0.0 }
        })
        .collect();
    let design = JointStaticRegimeDesign {
        unit_ids: unit_ids.clone(),
        periods: period.clone(),
        realized_assignment: assignment.clone(),
        confounder_ids: vec!["history".into()],
        active_path: vec![true, true],
        reference_path: vec![false, false],
        method: "g_formula".into(),
        min_probability: 0.05,
        positivity_assumption_id: "positivity".into(),
        exchangeability_assumption_id: "exchangeability".into(),
        consistency_assumption_id: "consistency".into(),
        followup_assumption_id: "complete".into(),
        model_assumption_id: "specified-model".into(),
    };
    let compiled = compile_joint_static_regime(design).unwrap();
    let components = vec![OutcomeComponent {
        name: "value".into(),
        variable_id: "value".into(),
        units: "u".into(),
    }];
    let ev = evidence_from_joint_static_regime(
        &compiled,
        &[&y],
        std::slice::from_ref(&confounder),
        "history-study",
        "pop",
        ("active", "reference"),
        components,
        Exposure::NoBetweenUnitInterference { assumption_id: "no-spill".into() },
    )
    .unwrap();
    let request = Request {
        population_id: "pop".into(),
        treated_regime_id: "active".into(),
        control_regime_id: "reference".into(),
        horizon: 2,
        target: Target::Regime {
            periods: 2,
            exchangeability_assumption_id: "exchangeability".into(),
        },
    };
    let q = DerivedQuantity {
        name: "regime-value".into(),
        units: "u".into(),
        expr: UtilityExpr::Input(0),
        inputs: vec![InputBinding::Measured("value".into())],
    };
    let result = derive(&ev, &request, &q).unwrap();
    let ctx = antecedent_core::ExecutionContext::for_tests(1);
    let retained = compile_and_evidence_from_joint_static_regime(
        &compiled,
        &[&y],
        std::slice::from_ref(&confounder),
        "history-study",
        "pop",
        ("active", "reference"),
        vec![OutcomeComponent {
            name: "value".into(),
            variable_id: "value".into(),
            units: "u".into(),
        }],
        Exposure::NoBetweenUnitInterference { assumption_id: "no-spill".into() },
        &ctx,
    )
    .unwrap();
    assert_eq!(retained.numerical_evidence(), &ev);
    let artifact = retained.source().export_artifact().unwrap();
    assert_eq!(
        retained.source().prepared().consume_current(&artifact, &ctx).unwrap().law(),
        retained.source().law()
    );
    assert!(result.sampling.is_none());
    assert_eq!(result.unresolved, vec![UnresolvedReason::JointLawMissing]);
    assert!((result.contrast.unwrap() - 3.0).abs() < 0.1);
    let (dense, _) = dense_labels(&unit_ids);
    let panel = RegimePanel::new(RegimeRows {
        unit: &dense,
        period: &period,
        outcome: &y,
        treated: &assignment,
        confounders: std::slice::from_ref(&confounder),
    })
    .unwrap();
    let (reference, _) = regime_point_estimate(
        &panel,
        StaticRegimes { active: &[true, true], reference: &[false, false] },
        RegimeMethod::GFormula,
        0.05,
    )
    .unwrap();
    assert!((result.contrast.unwrap() - reference[0]).abs() < 1e-10);
    assert!(
        matches!(&ev.design_details, Some(DesignDetails::TemporalRegime { observed_minimum_probabilities, .. }) if observed_minimum_probabilities == &[None])
    );
    for (name, method) in [
        ("inverse_probability_weighting", RegimeMethod::InverseProbability),
        ("doubly_robust", RegimeMethod::DoublyRobust),
    ] {
        let mut spec = compiled.design.clone();
        spec.method = name.into();
        let prepared = compile_joint_static_regime(spec).unwrap();
        let estimate = evidence_from_joint_static_regime(
            &prepared,
            &[&y],
            std::slice::from_ref(&confounder),
            "history-study",
            "pop",
            ("active", "reference"),
            vec![OutcomeComponent {
                name: "value".into(),
                variable_id: "value".into(),
                units: "u".into(),
            }],
            Exposure::NoBetweenUnitInterference { assumption_id: "no-spill".into() },
        )
        .unwrap();
        let (expected, _) = regime_point_estimate(
            &panel,
            StaticRegimes { active: &[true, true], reference: &[false, false] },
            method,
            0.05,
        )
        .unwrap();
        assert!(
            (derive(&estimate, &request, &q).unwrap().contrast.unwrap() - expected[0]).abs()
                < 1e-10
        );
        assert!(
            matches!(&estimate.design_details,Some(DesignDetails::TemporalRegime{observed_minimum_probabilities,..}) if observed_minimum_probabilities[0].is_some())
        );
    }
    let mut bad = request.clone();
    bad.target = Target::Excursion { lag: 1, assignment_law_id: "law".into() };
    assert!(derive(&ev, &bad, &q).is_err());
    let decision = decide(&ev, &request, &q, &frame(0.0)).unwrap();
    assert!(matches!(decision.verdict, Verdict::Unresolved { .. }));
    let bridge = TemporalEffectBridge {
        source: TemporalEstimand::StaticRegime {
            active_regime_id: "active".into(),
            reference_regime_id: "reference".into(),
            horizon: 2,
            exchangeability_assumption_id: "exchangeability".into(),
        },
        target: TemporalEstimand::ModelResponse {
            model_id: "response-model-v1".into(),
            active_schedule_id: "active-path".into(),
            reference_schedule_id: "reference-path".into(),
            horizon: 2,
        },
        source_component: "value".into(),
        target_variable_id: "value".into(),
        target_units: "u".into(),
        mapping: ProxyMapping::CausalEffectBounds {
            multiplier: (1.0, 1.0),
            residual: (0.0, 0.0),
            assumption_id: "model-bridge".into(),
            mapping_evidence_id: "mapping-evidence".into(),
        },
    };
    let bridged = decide_with_temporal_bridge(&ev, &request, &bridge, &frame(0.0)).unwrap();
    assert!(matches!(bridged.decision.verdict, Verdict::Unresolved { .. }));
    assert_eq!(bridged.decision.evidence_ids, vec!["history-study", "mapping-evidence"]);
}

#[test]
#[allow(
    clippy::cast_possible_wrap,
    clippy::items_after_statements,
    clippy::too_many_lines,
    reason = "the test keeps its existing bounded fixture and exact scientific assertions together"
)]
fn joint_excursion_matches_independent_rational_oracle() {
    let units = 32;
    let periods = 4;
    let n = units * periods;
    let assignments: Vec<bool> =
        (0..n).map(|i| ((i / periods * 7 + i % periods * 5) % 11) < 5).collect();
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for u in 0..units {
        for p in 0..periods - 1 {
            let treated = assignments[u * periods + p];
            y[u * periods + p + 1] =
                2.0 + (u % 7) as f64 / 5.0 + p as f64 / 4.0 + if treated { 1.5 } else { 0.0 };
            z[u * periods + p + 1] =
                4.0 + (u % 5) as f64 / 3.0 - p as f64 / 5.0 + if treated { 2.0 / 3.0 } else { 0.0 };
        }
    }
    let design = compile_joint_excursion(JointExcursionDesign {
        unit_ids: (0..units).flat_map(|u| (0..periods).map(move |_| format!("u{u}"))).collect(),
        periods: (0..units).flat_map(|_| (0..periods).map(|p| p as i64)).collect(),
        treatment_probabilities: vec![0.5; n],
        lag: 1,
        min_probability: 0.1,
        assignment_law_id: "half".into(),
        independent_units_assumption_id: "independent".into(),
        administrative_followup_assumption_id: "administrative".into(),
        consistency_assumption_id: "consistency".into(),
    })
    .unwrap();
    let components = ["y", "z"]
        .into_iter()
        .map(|v| OutcomeComponent { name: v.into(), variable_id: v.into(), units: "u".into() })
        .collect();
    let evidence = evidence_from_joint_excursion(
        &design,
        &[&y, &z],
        &assignments,
        "study",
        "pop",
        ("on", "off"),
        3,
        components,
        Exposure::NoBetweenUnitInterference { assumption_id: "isolation".into() },
    )
    .unwrap();
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("oracle_data/joint_excursion_reference.json")).unwrap();
    let means: Vec<_> =
        evidence.treated_means.iter().chain(&evidence.control_means).copied().collect();
    let Dependence::JointArms { covariance } = evidence.dependence else {
        panic!("joint law required")
    };
    for (actual, expected) in means.iter().zip(expected["means"].as_array().unwrap()) {
        assert!((actual - expected.as_f64().unwrap()).abs() < 1e-11);
    }
    for (actual, expected) in covariance.iter().zip(expected["covariance"].as_array().unwrap()) {
        assert!((actual - expected.as_f64().unwrap()).abs() < 1e-11);
    }
    use antecedent::analysis::joint_panel::designs::PreparedNativeJointDesign;
    use antecedent_estimate::joint_design::{
        compiled::{JointDesignInference, JointDesignRecipe, JointNativeDesignLaw},
        excursion::{ExcursionOutcome, JointExcursionRecipe},
    };
    let source = JointDesignRecipe::Excursion {
        recipe: JointExcursionRecipe {
            population: "pop".into(),
            treated_regime_id: "on".into(),
            control_regime_id: "off".into(),
            unit_ids: design.design.unit_ids.clone(),
            periods: design.design.periods.clone(),
            treatment_probabilities: vec![0.5; n],
            assignments,
            outcomes: vec![
                ExcursionOutcome {
                    component_name: "y".into(),
                    variable_id: "y".into(),
                    units: "u".into(),
                    values: y,
                },
                ExcursionOutcome {
                    component_name: "z".into(),
                    variable_id: "z".into(),
                    units: "u".into(),
                    values: z,
                },
            ],
            lag: 1,
            horizon: 3,
            min_probability: 0.1,
            assignment_law: "half".into(),
            independence: "independent".into(),
            followup: "administrative".into(),
            consistency: "versions".into(),
            isolation: "isolation".into(),
        },
    };
    let ctx = antecedent_core::ExecutionContext::for_tests(42);
    let prepared = PreparedNativeJointDesign::compile(source.clone(), &ctx).unwrap();
    assert_eq!(prepared.identification().target_kind, "sequential_randomized_lagged_excursion");
    let result = prepared.execute(JointDesignInference::Frequentist, &ctx).unwrap();
    let JointNativeDesignLaw::Excursion(law) = result.law() else {
        panic!("excursion law required")
    };
    for (actual, expected) in law.means.iter().zip(expected["means"].as_array().unwrap()) {
        assert!((actual - expected.as_f64().unwrap()).abs() < 1e-11);
    }
    for (actual, expected) in law.covariance.iter().zip(expected["covariance"].as_array().unwrap())
    {
        assert!((actual - expected.as_f64().unwrap()).abs() < 1e-11);
    }
    let bound =
        evidence_from_compiled_excursion(result.clone(), "study", ("on", "off"), 3).unwrap();
    let request = Request {
        population_id: "pop".into(),
        treated_regime_id: "on".into(),
        control_regime_id: "off".into(),
        horizon: 3,
        target: Target::Excursion { lag: 1, assignment_law_id: "half".into() },
    };
    let quantity = DerivedQuantity {
        name: "y".into(),
        units: "u".into(),
        expr: UtilityExpr::Input(0),
        inputs: vec![InputBinding::Measured("y".into())],
    };
    let derived = derive(bound.numerical_evidence(), &request, &quantity).unwrap();
    assert!((derived.contrast.unwrap() - (law.means[0] - law.means[2])).abs() < 1e-11);
    assert_eq!(bound.source().dependency_identity(), result.dependency_identity());
    let artifact = result.export_artifact().unwrap();
    assert_eq!(prepared.consume_current(&artifact, &ctx).unwrap().law(), result.law());
    let mut changed = source;
    let JointDesignRecipe::Excursion { recipe } = &mut changed else { unreachable!() };
    recipe.assignment_law = "different-law".into();
    let other = PreparedNativeJointDesign::compile(changed, &ctx).unwrap();
    assert!(other.consume_current(&artifact, &ctx).is_err());
    let cancelled = antecedent_core::ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    assert!(matches!(
        compiled_decision_refusal(
            PreparedNativeJointDesign::compile(prepared.recipe().clone(), &cancelled).unwrap_err()
        ),
        EvidenceRefusal::Cancelled
    ));
    let constrained = antecedent_core::ExecutionContext::for_tests(1)
        .with_admission(antecedent_core::execution::AdmissionPool::fixed(1, 1));
    assert!(matches!(
        compiled_decision_refusal(
            PreparedNativeJointDesign::compile(prepared.recipe().clone(), &constrained)
                .unwrap_err()
        ),
        EvidenceRefusal::Resource(_)
    ));
}

#[test]
fn compiled_cluster_matches_existing_estimator_and_retains_source() {
    use antecedent_estimate::joint_design::{
        cluster_decision::JointClusterRecipe, excursion::ExcursionOutcome,
        experiment::ExperimentExposure,
    };
    let fx = fixture(80);
    let original: Vec<_> = fx.totals.iter().map(Vec::as_slice).collect();
    let reference = complete_cluster_joint_contrast(&original, &fx.assigned, fx.rows).unwrap();
    let source = JointClusterRecipe {
        population: "active_customers".into(),
        treated_regime_id: "free_shipping".into(),
        control_regime_id: "standard_shipping".into(),
        horizon: 4,
        cluster_ids: (0..80).map(|i| format!("cluster-{i}")).collect(),
        cluster_sizes: vec![ROWS_PER_CLUSTER; 80],
        assignment_law: "complete-randomization".into(),
        treated_clusters: 40,
        consistency: "consistency".into(),
        assignments: fx.assigned,
        outcomes: components()
            .into_iter()
            .zip(fx.totals)
            .map(|(c, values)| ExcursionOutcome {
                component_name: c.name,
                variable_id: c.variable_id,
                units: c.units,
                values,
            })
            .collect(),
        adjustments: vec![],
        exposure: ExperimentExposure::NoInterference {
            assumption: "assumption.no_cross_customer_spillover".into(),
        },
    };
    let ctx = antecedent_core::ExecutionContext::for_tests(0);
    let bound = compile_and_evidence_from_cluster(source, "exp-shipping-2026q3", &ctx).unwrap();
    let ev = bound.numerical_evidence();
    assert_eq!(ev.treated_means, reference.treated_means);
    assert_eq!(ev.control_means, reference.control_means);
    let Dependence::Joint { treated, control } = &ev.dependence else {
        panic!("arm covariance required")
    };
    assert_eq!(treated, &reference.treated_covariance);
    assert_eq!(control, &reference.control_covariance);
    assert!(
        derive(ev, &request(), &margin(InputBinding::Measured("cost".into())))
            .unwrap()
            .sampling
            .is_some()
    );
    let bytes = bound.source().export_artifact().unwrap();
    assert_eq!(
        bound.source().prepared().consume_current(&bytes, &ctx).unwrap().law(),
        bound.source().law()
    );
    assert!(
        evidence_from_compiled_cluster(
            bound.source().clone(),
            "e",
            ("different", "standard_shipping"),
            4
        )
        .is_err()
    );
}

#[test]
#[allow(
    clippy::items_after_statements,
    reason = "the test keeps its existing bounded fixture and exact scientific assertions together"
)]
fn static_regime_bootstrap_refits_outcomes_with_the_same_subject_weights() {
    let n = 120;
    let assigned: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.29).sin()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 2.0 + x[i] + if assigned[i] { 0.7 } else { 0.0 } + 0.1 * ((i * 7 % 13) as f64))
        .collect();
    let z: Vec<f64> = y.iter().map(|v| 2.0 * v).collect();
    let design = compile_joint_static_regime(JointStaticRegimeDesign {
        unit_ids: (0..n).map(|i| format!("u{i}")).collect(),
        periods: vec![0; n],
        realized_assignment: assigned,
        confounder_ids: vec!["x".into()],
        active_path: vec![true],
        reference_path: vec![false],
        method: "g_formula".into(),
        min_probability: 0.05,
        positivity_assumption_id: "positivity".into(),
        exchangeability_assumption_id: "exchangeability".into(),
        consistency_assumption_id: "consistency".into(),
        followup_assumption_id: "followup".into(),
        model_assumption_id: "specified-model".into(),
    })
    .unwrap();
    let components = ["y", "z"]
        .into_iter()
        .map(|id| OutcomeComponent { name: id.into(), variable_id: id.into(), units: "u".into() })
        .collect();
    let ctx = antecedent_core::ExecutionContext::for_tests(7);
    let bound = compile_and_evidence_from_joint_static_regime_bootstrap(
        &design,
        &[&y, &z],
        std::slice::from_ref(&x),
        "study",
        "pop",
        ("on", "off"),
        components,
        Exposure::NoBetweenUnitInterference { assumption_id: "isolation".into() },
        99,
        123,
        &ctx,
    )
    .unwrap();
    use antecedent_estimate::joint_design::compiled::JointNativeDesignLaw;
    let JointNativeDesignLaw::StaticRegimeBootstrap(law) = bound.source().law() else {
        panic!("bootstrap law required")
    };
    assert_eq!(law.realizations.len(), 99);
    assert!(law.failures.iter().all(Option::is_none));
    for row in law.realizations.iter().flatten() {
        assert!((row[1] - 2.0 * row[0]).abs() < 1e-9);
        assert!((row[3] - 2.0 * row[2]).abs() < 1e-9);
    }
    let covariance = law.covariance.as_ref().unwrap();
    let rows: Vec<_> = law.realizations.iter().map(|v| v.as_ref().unwrap()).collect();
    let means: Vec<f64> =
        (0..4).map(|i| rows.iter().map(|row| row[i]).sum::<f64>() / rows.len() as f64).collect();
    for i in 0..4 {
        for j in 0..4 {
            let reference =
                rows.iter().map(|row| (row[i] - means[i]) * (row[j] - means[j])).sum::<f64>()
                    / (rows.len() - 1) as f64;
            assert!((covariance[4 * i + j] - reference).abs() < 1e-12);
        }
    }
    assert!((covariance[1] - 2.0 * covariance[0]).abs() < 1e-10);
    assert!((covariance[5] - 4.0 * covariance[0]).abs() < 1e-10);
    let request = Request {
        population_id: "pop".into(),
        treated_regime_id: "on".into(),
        control_regime_id: "off".into(),
        horizon: 1,
        target: Target::Regime {
            periods: 1,
            exchangeability_assumption_id: "exchangeability".into(),
        },
    };
    let quantity = DerivedQuantity {
        name: "y".into(),
        units: "u".into(),
        expr: UtilityExpr::Input(0),
        inputs: vec![InputBinding::Measured("y".into())],
    };
    let result = derive(bound.numerical_evidence(), &request, &quantity).unwrap();
    assert!(result.sampling.is_some());
    assert_eq!((result.license, result.calibration), ("none", "unmeasured"));
    let artifact = bound.source().export_artifact().unwrap();
    assert_eq!(
        bound.source().prepared().consume_current(&artifact, &ctx).unwrap().law(),
        bound.source().law()
    );
    assert!(
        decision_interval_calibration_basis(bound.numerical_evidence(), &request, &quantity)
            .is_ok()
    );
}

#[test]
#[allow(
    clippy::items_after_statements,
    reason = "the test keeps its existing bounded fixture and exact scientific assertions together"
)]
fn regime_bootstrap_does_not_drop_unsupported_refits() {
    let design = compile_joint_static_regime(JointStaticRegimeDesign {
        unit_ids: vec!["a".into(), "b".into()],
        periods: vec![0, 0],
        realized_assignment: vec![true, false],
        confounder_ids: vec!["x".into()],
        active_path: vec![true],
        reference_path: vec![false],
        method: "g_formula".into(),
        min_probability: 0.05,
        positivity_assumption_id: "positivity".into(),
        exchangeability_assumption_id: "exchangeability".into(),
        consistency_assumption_id: "consistency".into(),
        followup_assumption_id: "followup".into(),
        model_assumption_id: "specified-model".into(),
    })
    .unwrap();
    let ctx = antecedent_core::ExecutionContext::for_tests(4);
    let bound = compile_and_evidence_from_joint_static_regime_bootstrap(
        &design,
        &[&[1.0, 2.0]],
        &[vec![0.0, 1.0]],
        "e",
        "pop",
        ("on", "off"),
        vec![OutcomeComponent { name: "y".into(), variable_id: "y".into(), units: "u".into() }],
        Exposure::NoBetweenUnitInterference { assumption_id: "isolation".into() },
        99,
        11,
        &ctx,
    )
    .unwrap();
    use antecedent_estimate::joint_design::compiled::JointNativeDesignLaw;
    let JointNativeDesignLaw::StaticRegimeBootstrap(law) = bound.source().law() else {
        panic!("bootstrap law required")
    };
    assert!(law.failures.iter().any(Option::is_some));
    assert_eq!(law.failures.len(), 99);
    assert!(law.covariance.is_none());
    assert!(matches!(bound.numerical_evidence().dependence, Dependence::PointOnly));
}

#[test]
fn compiled_matched_pair_covariance_matches_rational_oracle() {
    let n = 60;
    let assignment: Vec<_> = (0..n).map(|i| i % 2 == (i / 2) % 2).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 + (i % 7) as f64 / 5.0 + (i / 2) as f64 / 10.0).collect();
    let z: Vec<f64> = (0..n).map(|i| 4.0 + (i % 5) as f64 / 3.0 - (i / 2) as f64 / 20.0).collect();
    let design = compile_matched_pair_design(MatchedPairDesign {
        cluster_ids: (0..n).map(|i| format!("c{i}")).collect(),
        pairs: (0..30).map(|i| (2 * i, 2 * i + 1)).collect(),
        matching_variables: vec!["pre".into()],
        treatment_probabilities: vec![0.5; n],
        cluster_sizes: vec![1; n],
        preassignment_assumption_id: "fixed".into(),
        independent_pairs_assumption_id: "independent".into(),
        consistency_assumption_id: "consistency".into(),
    })
    .unwrap();
    let components = ["y", "z"]
        .into_iter()
        .map(|v| OutcomeComponent { name: v.into(), variable_id: v.into(), units: "u".into() })
        .collect();
    let ctx = antecedent_core::ExecutionContext::for_tests(0);
    let bound = compile_and_evidence_from_matched_pairs(
        &design,
        &[&y, &z],
        &assignment,
        "e",
        "pop",
        ("on", "off"),
        1,
        components,
        Exposure::NoBetweenUnitInterference { assumption_id: "isolation".into() },
        &ctx,
    )
    .unwrap();
    let evidence = bound.numerical_evidence();
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("oracle_data/joint_matched_pair_reference.json"))
            .unwrap();
    for (actual, expected) in evidence
        .treated_means
        .iter()
        .chain(&evidence.control_means)
        .zip(reference["means"].as_array().unwrap())
    {
        assert!((actual - expected.as_f64().unwrap()).abs() < 1e-11);
    }
    let Dependence::JointArms { covariance } = &evidence.dependence else {
        panic!("joint covariance required")
    };
    for (actual, expected) in covariance.iter().zip(reference["covariance"].as_array().unwrap()) {
        assert!((actual - expected.as_f64().unwrap()).abs() < 1e-11);
    }
    let bytes = bound.source().export_artifact().unwrap();
    assert_eq!(
        bound.source().prepared().consume_current(&bytes, &ctx).unwrap().law(),
        bound.source().law()
    );
}

#[test]
fn weighted_regime_refit_matches_exact_rational_reference() {
    use antecedent_estimate::temporal_design::regime::{
        RegimeMethod, RegimePanel, RegimeRows, StaticRegimes, regime_point_estimate_weighted,
    };
    let unit: Vec<usize> = (0..8).collect();
    let period = vec![0; 8];
    let x = vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
    let treatment = vec![false, false, true, true, false, false, true, true];
    let weights = vec![1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0];
    let y: Vec<f64> = (0..8)
        .map(|i| 2.0 + x[i] / 2.0 + (if treatment[i] { 0.75 } else { 0.0 }) + (i % 3) as f64 / 5.0)
        .collect();
    let panel = RegimePanel::new(RegimeRows {
        unit: &unit,
        period: &period,
        outcome: &y,
        treated: &treatment,
        confounders: std::slice::from_ref(&x),
    })
    .unwrap();
    let (values, _) = regime_point_estimate_weighted(
        &panel,
        StaticRegimes { active: &[true], reference: &[false] },
        RegimeMethod::GFormula,
        0.05,
        &weights,
    )
    .unwrap();
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("oracle_data/weighted_regime_reference.json")).unwrap();
    assert!((values[1] - expected["active"].as_f64().unwrap()).abs() < 1e-10);
    assert!((values[2] - expected["reference"].as_f64().unwrap()).abs() < 1e-10);
}

#[test]
fn adaptive_regime_compiler_keeps_history_joint_covariance_and_target() {
    use antecedent_estimate::joint_design::adaptive_regime::{
        AdaptiveAction, AdaptiveEpisode, AdaptivePolicy, JointAdaptiveRegimeRecipe,
    };
    let mut episodes = Vec::new();
    for a in 0..2 {
        for state in 0..2 {
            for b in 0..2 {
                for repeat in 0..if a == state { 6 } else { 2 } {
                    let y = (2 * a + 3 * b + 4 * state + a * b) as f64;
                    episodes.push(AdaptiveEpisode {
                        id: format!("{a}/{state}/{b}/{repeat}"),
                        states: vec![0, state],
                        actions: vec![a, b],
                        outcomes: vec![y, 2.0 * y],
                    });
                }
            }
        }
    }
    let policy = |id: &str, first: usize, adaptive: bool| AdaptivePolicy {
        id: id.into(),
        rules: vec![
            AdaptiveAction { states: vec![0], past_actions: vec![], action: first },
            AdaptiveAction { states: vec![0, 0], past_actions: vec![first], action: 0 },
            AdaptiveAction {
                states: vec![0, 1],
                past_actions: vec![first],
                action: usize::from(adaptive),
            },
        ],
    };
    let recipe = JointAdaptiveRegimeRecipe {
        population: "pop".into(),
        state_variable_id: "state".into(),
        action_variable_id: "action".into(),
        outcomes: vec![("y".into(), "y".into(), "u".into()), ("z".into(), "z".into(), "u".into())],
        policies: [policy("adaptive", 1, true), policy("reference", 0, false)],
        episodes,
        policy_prespecification: "fixed before evaluation".into(),
        positivity: "positivity".into(),
        exchangeability: "sequential-exchangeability".into(),
        consistency: "consistency".into(),
        followup: "complete".into(),
        isolation: "independent".into(),
    };
    let ctx = antecedent_core::ExecutionContext::for_tests(1);
    let bound = compile_and_evidence_from_adaptive_regime(recipe, "adaptive-study", &ctx).unwrap();
    let ev = bound.numerical_evidence();
    assert_eq!(ev.treated_means, vec![8.0, 16.0]);
    assert_eq!(ev.control_means, vec![1.0, 2.0]);
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("oracle_data/adaptive_regime_reference.json")).unwrap();
    let Dependence::JointArms { covariance } = &ev.dependence else {
        panic!("joint covariance required")
    };
    for (actual, expected) in covariance.iter().zip(expected["covariance"].as_array().unwrap()) {
        assert!((actual - expected.as_f64().unwrap()).abs() < 1e-11);
    }
    let req = Request {
        population_id: "pop".into(),
        treated_regime_id: "adaptive".into(),
        control_regime_id: "reference".into(),
        horizon: 2,
        target: Target::AdaptiveRegime {
            periods: 2,
            exchangeability_assumption_id: "sequential-exchangeability".into(),
        },
    };
    let q = DerivedQuantity {
        name: "y".into(),
        units: "u".into(),
        expr: UtilityExpr::Input(0),
        inputs: vec![InputBinding::Measured("y".into())],
    };
    let result = derive(ev, &req, &q).unwrap();
    assert_eq!(result.contrast, Some(7.0));
    assert!(result.sampling.is_some());
    assert_eq!(result.license, "none");
    assert_eq!(result.calibration, "unmeasured");
    let mut wrong = req.clone();
    wrong.target = Target::Regime {
        periods: 2,
        exchangeability_assumption_id: "sequential-exchangeability".into(),
    };
    assert!(derive(ev, &wrong, &q).is_err());
    let basis = decision_interval_calibration_basis(ev, &req, &q).unwrap();
    assert_eq!(basis.row_count, ev.design.treated_units as u64);
}
