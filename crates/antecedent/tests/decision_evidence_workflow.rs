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
