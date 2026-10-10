//! Real downstream F18 consumers retain original evidence and point-only limitations.
use antecedent::analysis::effect_constancy::{
    DependenceWire, EffectConstancy, EffectConstancyRequestWire, EffectEstimandWire, FamilyWire,
    PartitionWire,
};
use antecedent::analysis::effect_constancy_consumers::{ConstancyConsumers, PriorPartitionBinding};
use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_io::prior_bank::{
    DesignVariableRole, DesignVariableSummary, EstimandFingerprint, PriorCatalog, PriorSourceMeta,
    PriorSourceRef, TargetDesign,
};

fn oracle() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/population_time/effect_constancy_consumers/expected.json"
    ))
    .unwrap()
}

fn estimand() -> EffectEstimandWire {
    EffectEstimandWire {
        estimand: "ate_difference".into(),
        units: "outcome_units".into(),
        regime: "treat_vs_control".into(),
        population: "all_observed_h2".into(),
    }
}
fn original(effects: &[f64]) -> EffectConstancy {
    EffectConstancy::evaluate(&EffectConstancyRequestWire {
        partitions: effects
            .iter()
            .enumerate()
            .map(|(i, &effect)| PartitionWire {
                label: format!("p{i}"),
                coordinate: format!("region:{i}"),
                support: "supported".into(),
                estimand: estimand(),
                effect,
                standard_error: 0.5,
            })
            .collect(),
        dependence: DependenceWire { kind: "independent".into(), covariance: vec![] },
        family: FamilyWire { kind: "all_pairs".into(), reference: None },
        alpha: 0.05,
    })
    .unwrap()
}
fn consumer(original: &EffectConstancy) -> ConstancyConsumers {
    ConstancyConsumers::consume(&original.export("f18-downstream").unwrap(), original.identity())
        .unwrap()
}
fn effect() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y_effect".into(),
        variable_name: "Y effect".into(),
        role: QuantityRole::Outcome,
        units: "outcome_units".into(),
        population_id: "all_observed_h2".into(),
        regime_id: "treat_vs_control".into(),
        horizon: 0,
        functional_id: "ate_difference".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}
fn contract() -> DecisionContract {
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "baseline".into(),
                kind: ActionKind::Regime,
                inputs: vec![effect()],
                utility: UtilityExpr::product(UtilityExpr::Const(0.0), UtilityExpr::Input(0)),
            },
            DecisionAction {
                id: "treat".into(),
                kind: ActionKind::Intervention,
                inputs: vec![effect()],
                utility: UtilityExpr::Input(0),
            },
        ],
        utility_units: "outcome_units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "all_observed_h2".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}
#[test]
fn original_covariance_aware_contrast_replays_and_does_not_license_transport() {
    let mut req = EffectConstancyRequestWire {
        partitions: vec![
            PartitionWire {
                label: "a".into(),
                coordinate: "region:a".into(),
                support: "supported".into(),
                estimand: estimand(),
                effect: 1.0,
                standard_error: 0.5,
            },
            PartitionWire {
                label: "b".into(),
                coordinate: "region:b".into(),
                support: "supported".into(),
                estimand: estimand(),
                effect: 2.0,
                standard_error: 0.5,
            },
        ],
        dependence: DependenceWire {
            kind: "covariance".into(),
            covariance: vec![0.25, 0.125, 0.125, 0.25],
        },
        family: FamilyWire { kind: "all_pairs".into(), reference: None },
        alpha: 0.05,
    };
    let original = EffectConstancy::evaluate(&req).unwrap();
    let diagnostic = consumer(&original).transport_diagnostic("a", "b").unwrap();
    // Var(a-b)=.25+.25-2*.125=.25, not the independent .5.
    assert!((diagnostic.contrast.difference + 1.0).abs() < 1e-12);
    assert!((diagnostic.contrast.standard_error - 0.5).abs() < 1e-12);
    assert!(
        (diagnostic.contrast.p_holm
            - oracle()["transport"]["holm_p_single_contrast"].as_f64().unwrap())
        .abs()
            < 1e-9
    );
    assert!(diagnostic.separate_transport_identification_required);
    assert_eq!(diagnostic.evidence.calibration, "unmeasured");
    assert!(consumer(&original).transport_diagnostic("b", "a").is_err());
    req.partitions[0].effect = 1.1;
    let changed = EffectConstancy::evaluate(&req).unwrap();
    assert!(
        ConstancyConsumers::consume(&changed.export("changed").unwrap(), original.identity())
            .is_err()
    );
}
#[test]
fn actual_bank_filtering_and_effect_ranking_remain_data_dependent_advice() {
    let fingerprint = EstimandFingerprint::new("ate", "A", "Y");
    let sources = ["far", "near", "wrong"].map(|id| {
        let mut meta = PriorSourceMeta::new(
            id,
            if id == "wrong" {
                EstimandFingerprint::new("ate", "A", "Z")
            } else {
                fingerprint.clone()
            },
            "nonparametrically_identified",
        );
        meta.design = vec![
            DesignVariableSummary::new("A", DesignVariableRole::Treatment),
            DesignVariableSummary::new(
                if id == "wrong" { "Z" } else { "Y" },
                DesignVariableRole::Outcome,
            ),
        ];
        PriorSourceRef::from_meta(meta)
    });
    let bank = PriorCatalog::from_sources(sources.into());
    let target = TargetDesign::new(fingerprint, Vec::<String>::new());
    let c = consumer(&original(&[1.0, 2.0, 2.1]));
    let bindings = vec![
        PriorPartitionBinding { artifact_id: "far".into(), partition: "p0".into() },
        PriorPartitionBinding { artifact_id: "near".into(), partition: "p1".into() },
        PriorPartitionBinding { artifact_id: "wrong".into(), partition: "p2".into() },
    ];
    let ranking = c.rank_prior_sources(&bank, &target, "p2", &bindings).unwrap();
    assert_eq!(
        ranking
            .ranked
            .iter()
            .map(antecedent_io::CompatibilityReport::artifact_id)
            .collect::<Vec<_>>(),
        ["near", "far"]
    );
    assert!((ranking.scores[0].1 - oracle()["prior"]["scores"][0].as_f64().unwrap()).abs() < 1e-12);
    assert!(ranking.data_dependent_selection);
    assert!(!ranking.posterior_transfer_licensed);
    assert!(c.rank_prior_sources(&bank, &target, "p2", &bindings[..2]).is_err());
}
#[test]
fn actual_affine_policy_engine_retains_ties_and_empty_generalization_intersection() {
    let disagreement =
        consumer(&original(&[-1.0, 2.0])).policy_review(&contract(), &effect()).unwrap();
    assert!(disagreement.common_leaders.is_empty());
    assert!(!disagreement.generalization_guarantee);
    assert!((disagreement.partitions[0].result.actions[1].value + 1.0).abs() < 1e-12);
    assert!((disagreement.partitions[1].result.actions[1].value - 2.0).abs() < 1e-12);
    let ties = consumer(&original(&[0.0, 0.0])).policy_review(&contract(), &effect()).unwrap();
    assert_eq!(ties.common_leaders, ["baseline", "treat"]);
    assert_ne!(ties.partitions[0].source.snapshot_id, ties.partitions[1].source.snapshot_id);
    let same = consumer(&original(&[1.0, 2.0])).policy_review(&contract(), &effect()).unwrap();
    assert_eq!(same.common_leaders, ["treat"]);
    assert!(
        same.partitions.iter().all(|p| p.result.evpi.is_none()
            && p.result.actions.iter().all(|a| a.standard_error.is_none()))
    );
    let mut mismatch = effect();
    mismatch.units = "wrong".into();
    assert!(consumer(&original(&[1.0, 2.0])).policy_review(&contract(), &mismatch).is_err());
}
