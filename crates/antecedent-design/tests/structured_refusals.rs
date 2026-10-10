//! Acceptance evidence for record F25 (structured refusals).
//!
//! A refusal is data: the stage that refused, the offending action or coordinate,
//! expected versus supplied semantics, the missing capability and a remedy, under a
//! registered code and a namespaced detail. Four different mistakes (wrong
//! population, wrong action, wrong unit, missing capability) must stay four
//! distinguishable refusals, and a malformed refusal envelope is itself refused.

use std::collections::BTreeSet;

use antecedent_core::{
    DistributionMeaning, ExternalCapability, ExternalCapabilityRequest, ExternalRefusal,
    ExternalScientificObject, LawProviderContract, ProviderObjectIdentity, QuantityRole,
    ScientificQuantity,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{DecisionEvalError, evaluate_contract};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::external_binding_wire::{ContractWire, ResponseWire, bind_response};
use antecedent_io::quantity_wire::DistributionMeaningWire;
use serde_json::{Value, json};

fn quantity(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn action(id: &str, inputs: Vec<ScientificQuantity>, utility: UtilityExpr) -> DecisionAction {
    DecisionAction { id: id.into(), kind: ActionKind::Intervention, inputs, utility }
}

/// Action A1 needs the joint law of two outcomes; A2 reads one.
fn contract() -> DecisionContract {
    DecisionContract {
        actions: vec![
            action(
                "A1",
                vec![quantity("p", "do(a=1)"), quantity("q", "do(a=1)")],
                UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            ),
            action("A2", vec![quantity("s", "do(a=0)")], UtilityExpr::Input(0)),
        ],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn independent_marginals() -> DistributionArtifact {
    let columns = [quantity("p", "do(a=1)"), quantity("q", "do(a=1)"), quantity("s", "do(a=0)")];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::IndependentMarginals,
        DistributionProvenance {
            source_id: "marginals".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "snap".into(),
            causal_contract_id: "checked".into(),
        },
    )
    .unwrap();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [2, 3],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        vec![0.0, 0.0, 1.0, 1.0, 2.0, 1.0],
    )
    .unwrap()
}

fn pair(refusal: &ExternalRefusal) -> (String, String) {
    (refusal.code.to_owned(), refusal.detail.clone())
}

/// F25 frozen boundary: a decision requests a joint outcome utility from
/// independent marginals at action A1; the refusal keeps the stage, the action, the
/// reason, the supplied alignment and the remedy.
#[test]
fn f25_decision_refusal_carries_stage_action_reason_supplied_meaning_and_remedy() {
    let error = evaluate_contract(&contract(), &independent_marginals()).unwrap_err();
    assert!(matches!(
        error,
        DecisionEvalError::JointLawRequired { action: Some(ref id), .. } if id == "A1"
    ));
    let refusal = error.to_refusal();
    assert_eq!(refusal.stage, "evaluate");
    assert_eq!(refusal.code, "joint_law_required");
    assert_eq!(refusal.detail, "decision_evaluation.joint_law_required");
    assert_eq!(refusal.offending.as_deref(), Some("A1"));
    assert_eq!(refusal.expected.as_deref(), Some("joint"));
    assert_eq!(refusal.supplied.as_deref(), Some("independent_marginals"));
    assert!(refusal.remedy.is_some_and(|r| r.contains("joint draws")));
    assert_eq!(refusal.validate(), Ok(()));
}

fn wire_contract() -> ContractWire {
    let q = |dose: u32, units: &str| {
        json!({
            "version": 1, "variable_id": "y", "variable_name": "y", "role": "outcome",
            "units": units, "population_id": "target", "regime_id": format!("do(a={dose})"),
            "horizon": 0, "functional_id": "mean", "conditioning": [], "transform_id": "identity"
        })
    };
    serde_json::from_value(json!({
        "graph_id": "g", "identification": "nonparametrically_identified",
        "estimand": [q(0, "mmHg"), q(1, "mmHg")],
        "accepted_meanings": ["interventional_predictive"],
        "required_evidence_ids": [], "required_assumption_ids": [], "equivalences": []
    }))
    .unwrap()
}

fn wire_response(units: &str) -> ResponseWire {
    let q = |dose: u32, units: &str| -> Value {
        json!({
            "version": 1, "variable_id": "y", "variable_name": "y", "role": "outcome",
            "units": units, "population_id": "target", "regime_id": format!("do(a={dose})"),
            "horizon": 0, "functional_id": "mean", "conditioning": [], "transform_id": "identity"
        })
    };
    serde_json::from_value(json!({
        "provider": {
            "provider_id": "lab", "object_id": "o", "version_id": "v", "snapshot_id": "s",
            "request_id": "r", "meaning": "interventional_predictive", "capabilities": ["mean"]
        },
        "graph_id": "g", "quantities": [q(0, "mmHg"), q(1, units)], "values": [1.0, 2.0],
        "evidence_ids": [], "assumption_ids": [], "attestor": "lab", "probes": null,
        "uncertainty_method": null, "point_support": null
    }))
    .unwrap()
}

/// F25 frozen boundary: wrong population, action, unit and capability each retain a
/// distinct, stable reason and detail, and each is a well-formed envelope.
#[test]
fn f25_wrong_population_action_unit_and_capability_stay_four_distinct_refusals() {
    let mut population = contract();
    population.actions[0].inputs[1].population_id = "other".into();
    let population = population.validate().unwrap_err().to_refusal();

    let mut unknown = contract();
    unknown.constraints.push(HardConstraint {
        id: "cap".into(),
        expr: UtilityExpr::Input(0),
        bound: 1.0,
        min_probability: 1.0,
        units: "units".into(),
        applies_to: vec!["A9".into()],
    });
    let action = unknown.validate().unwrap_err().to_refusal();

    let unit = bind_response(&wire_contract(), &wire_response("kPa"), "c").unwrap_err();
    let unit = ExternalRefusal {
        code: antecedent_core::reason_code!("quantity_semantics_mismatch"),
        stage: "bind",
        detail: unit.detail.clone(),
        offending: unit.offending.clone(),
        expected: unit.expected.clone(),
        supplied: unit.supplied.clone(),
        capability: None,
        remedy: None,
    };
    assert_eq!(unit.detail, "external_response_binding.coordinate_units");
    assert_eq!(unit.expected.as_deref(), Some("mmHg"));
    assert_eq!(unit.supplied.as_deref(), Some("kPa"));

    let law = ExternalScientificObject::Law(LawProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "o".into(),
            version_id: "v".into(),
            snapshot_id: "s".into(),
            request_id: "r".into(),
        },
        quantities: vec![quantity("y", "do(a=1)")],
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![ExternalCapability::Mean],
    });
    let capability = law
        .negotiate(&ExternalCapabilityRequest {
            expected_identity: law.identity().clone(),
            operation: ExternalCapability::Cdf,
        })
        .unwrap_err()
        .to_refusal();

    let refusals = [&population, &action, &unit, &capability];
    assert_eq!(
        pair(&population),
        (
            "quantity_semantics_mismatch".to_owned(),
            "decision_contract.outside_decision_scope".to_owned()
        )
    );
    assert_eq!(population.offending.as_deref(), Some("A1[1]"));
    assert_eq!(
        pair(&action),
        ("invalid_argument".to_owned(), "decision_contract.unknown_action".to_owned())
    );
    assert_eq!(
        pair(&capability),
        (
            "external_capability_missing".to_owned(),
            "provider_capability_negotiation.capability_missing".to_owned()
        )
    );
    assert_eq!(capability.capability, Some(ExternalCapability::Cdf));
    let distinct: BTreeSet<_> = refusals.iter().map(|r| pair(r)).collect();
    assert_eq!(distinct.len(), 4, "four mistakes, four distinct (code, detail) pairs");
    for refusal in refusals {
        assert_eq!(refusal.validate(), Ok(()), "{}", refusal.detail);
    }
}

/// F25: a refusal that is itself malformed is refused, with its own detail.
#[test]
fn f25_malformed_refusal_envelopes_are_themselves_refused() {
    let good = evaluate_contract(&contract(), &independent_marginals()).unwrap_err().to_refusal();

    let mut unregistered = good.clone();
    unregistered.code = "not_a_registered_code";
    let error = unregistered.validate().unwrap_err();
    assert_eq!(
        (error.code, error.detail.as_str()),
        ("invalid_argument", "structured_refusals.unregistered_code")
    );
    assert_eq!(error.offending.as_deref(), Some("not_a_registered_code"));

    for bad in ["no_namespace", "a.b.c", "Upper.case", "ns.", ".slot", ""] {
        let mut malformed = good.clone();
        malformed.detail = bad.to_owned();
        let error = malformed.validate().unwrap_err();
        assert_eq!(error.detail, "structured_refusals.malformed_detail", "{bad}");
        assert_eq!(error.offending.as_deref(), Some(bad));
    }

    let mut stageless = good;
    stageless.stage = "  ";
    let error = stageless.validate().unwrap_err();
    assert_eq!(error.detail, "structured_refusals.missing_stage");
    assert_eq!(error.validate(), Ok(()), "the refusal about a refusal is itself well formed");
}

/// A contract declared as JSON (the form every host language sends) refuses with the
/// same typed code and detail a Rust caller gets, not a generic declaration error.
#[test]
fn f25_json_declared_contract_keeps_its_typed_refusal() {
    use antecedent_design::decision_artifact::{contract_from_json_refusal, contract_to_json};
    let good = contract_to_json(&contract()).unwrap();
    assert!(contract_from_json_refusal(&good).is_ok());

    let mut wrong_population: Value = serde_json::from_str(&good).unwrap();
    wrong_population["actions"][0]["inputs"][1]["population_id"] = json!("other");
    let refusal = contract_from_json_refusal(&wrong_population.to_string()).unwrap_err();
    assert_eq!(
        pair(&refusal),
        (
            "quantity_semantics_mismatch".to_owned(),
            "decision_contract.outside_decision_scope".to_owned()
        )
    );
    assert_eq!(refusal.offending.as_deref(), Some("A1[1]"));

    let mut unknown_action: Value = serde_json::from_str(&good).unwrap();
    unknown_action["constraints"] = json!([{
        "id": "cap", "expr": {"input": 0}, "bound": 1.0, "min_probability": 1.0,
        "units": "units", "applies_to": ["A9"]
    }]);
    let refusal = contract_from_json_refusal(&unknown_action.to_string()).unwrap_err();
    assert_eq!(
        pair(&refusal),
        ("invalid_argument".to_owned(), "decision_contract.unknown_action".to_owned())
    );

    let refusal = contract_from_json_refusal("{ not json").unwrap_err();
    assert_eq!(refusal.detail, "decision_contract.invalid_declaration");
    assert_eq!(refusal.validate(), Ok(()));
}
