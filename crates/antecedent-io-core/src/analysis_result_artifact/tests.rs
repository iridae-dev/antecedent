//! Analysis-result artifact tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn fixture() -> AnalysisResultWire {
    let query = serde_json::json!({"response": {
        "functional": {"average_derivative": {"outcome": 1, "treatment": 0, "weighting": "observed"}},
        "target_population": "all_observed", "observation": "complete", "observation_assumptions": []
    }});
    serde_json::from_value(serde_json::json!({
            "query": query,
            "identification": {"status": "nonparametrically_identified", "query": query,
                "estimands": [], "arena": {"var_sets": [], "interventions": [], "lists": [], "nodes": []},
                "derivation": [], "required_assumptions": [], "diagnostics": [], "candidates_examined": 0, "sets_returned": 0},
            "estimate": 1.0, "standard_error": 0.2, "assumptions": [], "diagnostics": [], "refutations": [],
            "response": null, "posterior_artifact": null, "mediation_grid": null, "structural_response": null
        })).unwrap()
}

fn distribution_fixture() -> AnalysisResultWire {
    let query = antecedent_core::CausalQuery::Distribution(
        antecedent_core::InterventionalDistributionQuery::new(
            antecedent_core::VariableId::from_raw(1),
            [antecedent_core::Intervention::set(
                antecedent_core::VariableId::from_raw(0),
                antecedent_core::Value::f64(1.0),
            )],
        ),
    );
    let query = crate::causal_query_to_wire(&query).unwrap();
    let mut result = fixture();
    result.query = query.clone();
    result.identification.query = query;
    result.estimate = Some(0.7);
    result.interventional_distribution = Some(InterventionalDistributionWire {
        atoms: vec![
            DistributionAtomWire {
                outcomes: vec![(1, crate::ValueWire::Float64(0.0))],
                conditioning: Vec::new(),
                probability: 0.3,
            },
            DistributionAtomWire {
                outcomes: vec![(1, crate::ValueWire::Float64(1.0))],
                conditioning: Vec::new(),
                probability: 0.7,
            },
        ],
    });
    result
}

#[test]
fn panel_did_result_artifact_round_trips_design_uncertainty_semantics() {
    let domain = antecedent_core::PanelDidQuery::new(
        antecedent_core::VariableId::from_raw(1),
        [true, true, true, true, false, false, false, false],
        [false, true, false, true, false, true, false, true],
        ["a", "a", "b", "b", "c", "c", "d", "d"].map(std::sync::Arc::<str>::from),
        ["x", "x", "y", "y", "z", "z", "w", "w"].map(std::sync::Arc::<str>::from),
    );
    let query =
        crate::causal_query_to_wire(&antecedent_core::CausalQuery::PanelDid(domain)).unwrap();
    let mut result = fixture();
    result.query = query.clone();
    result.identification.query = query;
    result.estimate = Some(2.0);
    result.standard_error = None;
    result.panel_did = Some(PanelDidWire {
        effect: 2.0,
        standard_error: Some(0.4),
        interval_95: None,
        treated_subjects: 2,
        comparison_subjects: 2,
        clusters: 4,
        uncertainty: "cluster_robust_standard_error_no_interval".into(),
        graphless_support_status: None,
        event_time_effects: vec![],
        event_time_intervals_95: vec![],
        augmented: None,
    });
    let encoded = encode_analysis_result_artifact(
        &result,
        vec!["treatment".into(), "outcome".into()],
        "panel-did-result",
    )
    .unwrap();
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes).unwrap();
    let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(decoded, result);
}

#[test]
fn repeated_cross_section_did_result_round_trips_without_interval() {
    let domain = antecedent_core::PanelDidQuery::repeated_cross_section(
        antecedent_core::VariableId::from_raw(1),
        [false, false, false, false, true, true, true, true],
        [false, false, true, true, false, false, true, true],
        (0..8).map(|i| std::sync::Arc::<str>::from(format!("s{i}"))).collect::<Vec<_>>(),
        (0..8).map(|i| std::sync::Arc::<str>::from(format!("c{i}"))).collect::<Vec<_>>(),
    );
    let query =
        crate::causal_query_to_wire(&antecedent_core::CausalQuery::PanelDid(domain)).unwrap();
    let mut result = fixture();
    result.query = query.clone();
    result.identification.query = query;
    result.estimate = Some(2.0);
    result.standard_error = None;
    result.panel_did = Some(PanelDidWire {
        effect: 2.0,
        standard_error: Some((16.0_f64 / 7.0).sqrt()),
        interval_95: None,
        treated_subjects: 4,
        comparison_subjects: 4,
        clusters: 8,
        uncertainty: "cluster_robust_standard_error_no_interval".into(),
        graphless_support_status: None,
        event_time_effects: vec![],
        event_time_intervals_95: vec![],
        augmented: None,
    });
    let encoded = encode_analysis_result_artifact(
        &result,
        vec!["treatment".into(), "outcome".into()],
        "rcs-did-result",
    )
    .unwrap();
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes).unwrap();
    let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(decoded, result);
    result.interval_lower = Some(0.0);
    assert!(
        encode_analysis_result_artifact(
            &result,
            vec!["treatment".into(), "outcome".into()],
            "rcs-invalid-interval"
        )
        .is_err()
    );
}

#[test]
fn staggered_did_result_round_trips_selected_cohort_counts_without_interval() {
    let mut units = Vec::new();
    let mut clusters = Vec::new();
    let mut periods = Vec::new();
    let mut cohorts = Vec::new();
    for unit in 0..10 {
        for period in 1..=4 {
            units.push(std::sync::Arc::<str>::from(format!("u{unit}")));
            clusters.push(std::sync::Arc::<str>::from(format!("c{unit}")));
            periods.push(period);
            cohorts.push(if unit < 4 {
                0
            } else if unit < 8 {
                3
            } else {
                4
            });
        }
    }
    let query = crate::causal_query_to_wire(&antecedent_core::CausalQuery::PanelDid(
        antecedent_core::PanelDidQuery::staggered_group_time(
            antecedent_core::VariableId::from_raw(1),
            units,
            clusters,
            periods,
            cohorts,
            3,
            4,
        ),
    ))
    .unwrap();
    let mut result = fixture();
    result.query = query.clone();
    result.identification.query = query;
    result.estimate = Some(4.0);
    result.standard_error = None;
    result.panel_did = Some(PanelDidWire {
        effect: 4.0,
        standard_error: Some(0.5),
        interval_95: None,
        treated_subjects: 4,
        comparison_subjects: 4,
        clusters: 8,
        uncertainty: "cluster_robust_standard_error_no_interval".into(),
        graphless_support_status: None,
        event_time_effects: vec![],
        event_time_intervals_95: vec![],
        augmented: None,
    });
    let encoded = encode_analysis_result_artifact(
        &result,
        vec!["treatment".into(), "outcome".into()],
        "staggered-did-result",
    )
    .unwrap();
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes).unwrap();
    let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(decoded, result);
}

#[test]
fn distribution_atoms_are_bound_to_query_and_reported_mean() {
    let mut result = distribution_fixture();
    validate_interventional_distribution(&result).unwrap();

    result.interventional_distribution.as_mut().unwrap().atoms[0].probability = 0.2;
    assert!(validate_interventional_distribution(&result).is_err());

    let mut result = distribution_fixture();
    result.interventional_distribution.as_mut().unwrap().atoms[1].outcomes[0].0 = 0;
    assert!(validate_interventional_distribution(&result).is_err());

    let mut result = distribution_fixture();
    result.estimate = Some(0.6);
    assert!(validate_interventional_distribution(&result).is_err());

    let mut result = distribution_fixture();
    result.interventional_distribution.as_mut().unwrap().atoms[1].probability = f64::NAN;
    assert!(validate_interventional_distribution(&result).is_err());
}

#[test]
fn composite_roundtrip_validates_names_ids_and_nested_posterior() {
    let result = fixture();
    let names = vec!["a".into(), "y".into()];
    let artifact = encode_analysis_result_artifact(&result, names.clone(), "review").unwrap();
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes).unwrap();
    let (_, header, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(decoded, result);
    assert_eq!(header.variable_names, names);
    assert!(encode_analysis_result_artifact(&result, vec!["a".into()], "review").is_err());
    assert!(
        encode_analysis_result_artifact(&result, vec!["a".into(), "a".into()], "review").is_err()
    );
    let mut invalid = result;
    invalid.posterior_artifact = Some(vec![1, 2, 3]);
    assert!(encode_analysis_result_artifact(&invalid, names, "review").is_err());
}
#[test]
fn temporal_identification_uses_its_own_namespace_and_preserves_offsets() {
    let mut result = fixture();
    // Base query remains ids 0,1; its unfolded certificate uses ids 4,5.
    let mut wire = serde_json::to_value(&result.identification).unwrap();
    wire["query"]["response"]["functional"]["average_derivative"]["treatment"] = 4.into();
    wire["query"]["response"]["functional"]["average_derivative"]["outcome"] = 5.into();
    result.identification = serde_json::from_value(wire).unwrap();
    let variables: Vec<_> = (-2..=0)
        .flat_map(|offset| {
            (0..2).map(move |variable| crate::HorizonAdjustmentNodeWire { variable, offset })
        })
        .collect();
    result.identification_variables = Some(variables.clone());
    result.temporal_identification.push(TemporalIdentificationWire {
        horizon: 1,
        variables,
        identification: result.identification.clone(),
    });
    let names = vec!["x".into(), "y".into()];
    let artifact = encode_analysis_result_artifact(&result, names.clone(), "temporal").unwrap();
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes).unwrap();
    let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(decoded, result);
    result.identification_variables.as_mut().unwrap()[5].variable = 2;
    assert!(encode_analysis_result_artifact(&result, names.clone(), "invalid").is_err());
    result.identification_variables = None;
    assert!(encode_analysis_result_artifact(&result, names, "missing").is_err());
}
fn interval() -> antecedent_stats::IdentifiedSetInterval {
    antecedent_stats::IdentifiedSetInterval {
        level: 0.9,
        lower: 0.41,
        upper: 1.12,
        bound_lower: 0.52,
        bound_upper: 0.98,
        lower_se: 0.061,
        upper_se: 0.083,
        critical_value: 1.31,
        width_retained: true,
        completions: 2,
        replicates: 199,
        method: antecedent_stats::IdentifiedSetIntervalMethod::ImbensManskiSharedBlock,
        truncated: false,
    }
}

#[test]
fn every_identified_set_construction_has_its_own_wire_tag() {
    let mut tags = Vec::new();
    for method in antecedent_stats::IdentifiedSetIntervalMethod::ALL {
        let original = antecedent_stats::IdentifiedSetInterval { method, ..interval() };
        let wire = identified_set_interval_to_wire(&original).unwrap();
        assert_eq!(identified_set_interval_from_wire(&wire).unwrap().method, method);
        tags.push(serde_json::to_value(wire.method).unwrap());
    }
    tags.dedup();
    assert_eq!(tags.len(), antecedent_stats::IdentifiedSetIntervalMethod::ALL.len());
}

#[test]
fn truncated_identified_set_interval_round_trips_and_defaults_to_false() {
    let original = antecedent_stats::IdentifiedSetInterval { truncated: true, ..interval() };
    let wire = identified_set_interval_to_wire(&original).unwrap();
    let json = serde_json::to_value(&wire).unwrap();
    assert_eq!(json["truncated"], true);
    assert_eq!(identified_set_interval_from_wire(&wire).unwrap(), original);
    // An untruncated interval omits the field, and a body without it decodes
    // as untruncated.
    let plain = identified_set_interval_to_wire(&interval()).unwrap();
    let json = serde_json::to_value(&plain).unwrap();
    assert!(json.get("truncated").is_none());
    let decoded: IdentifiedSetIntervalWire = serde_json::from_value(json).unwrap();
    assert!(!identified_set_interval_from_wire(&decoded).unwrap().truncated);
}

fn with_structural(interval: Option<IdentifiedSetIntervalWire>) -> AnalysisResultWire {
    let mut result = fixture();
    result.structural_response = Some(StructuralResponseMixtureWire {
        weight_basis: StructuralWeightBasisWire::CompletionEnumeration,
        atoms: Vec::new(),
        identified_mass: 1.0,
        unidentified_mass: 0.0,
        unevaluable_mass: 0.0,
        subsampled_out_mass: 0.0,
        identified_set: None,
        identified_set_interval: interval,
        conditional_on_identified: None,
        full_mass_scope: true,
        truncated_atoms: 0,
    });
    result
}

#[test]
fn subsampled_out_mass_is_optional_on_the_wire_and_round_trips() {
    let names: Vec<String> = vec!["a".into(), "y".into()];
    // Zero is omitted, and a body without the field decodes as zero.
    let plain = with_structural(None);
    let json = serde_json::to_value(&plain).unwrap();
    assert!(json["structural_response"].get("subsampled_out_mass").is_none());
    let decoded: AnalysisResultWire = serde_json::from_value(json).unwrap();
    assert!(decoded.structural_response.unwrap().subsampled_out_mass.abs() < f64::EPSILON);

    let mut result = with_structural(None);
    if let Some(structural) = result.structural_response.as_mut() {
        structural.identified_mass = 0.4;
        structural.subsampled_out_mass = 0.6;
    }
    let artifact = encode_analysis_result_artifact(&result, names.clone(), "sub").unwrap();
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes).unwrap();
    let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(decoded, result);

    if let Some(structural) = result.structural_response.as_mut() {
        structural.subsampled_out_mass = 1.5;
    }
    assert!(encode_analysis_result_artifact(&result, names, "bad").is_err());
}

#[test]
fn identified_set_interval_round_trips_and_validates() {
    let original = interval();
    let wire = identified_set_interval_to_wire(&original).unwrap();
    assert_eq!(identified_set_interval_from_wire(&wire).unwrap(), original);
    let result = with_structural(Some(wire.clone()));
    let names = vec!["a".into(), "y".into()];
    let artifact = encode_analysis_result_artifact(&result, names.clone(), "set").unwrap();
    assert_eq!(artifact.manifest.format_version, STABLE_FORMAT);
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes).unwrap();
    let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(decoded, result);
    let restored = decoded.structural_response.unwrap().identified_set_interval.unwrap();
    assert_eq!(identified_set_interval_from_wire(&restored).unwrap(), original);
    let json = serde_json::to_value(&result).unwrap();
    assert_eq!(
        json["structural_response"]["identified_set_interval"]["method"],
        "imbens_manski_shared_block"
    );

    for corrupt in [
        IdentifiedSetIntervalWire { level: 1.0, ..wire.clone() },
        IdentifiedSetIntervalWire { lower: 1.5, ..wire.clone() },
        IdentifiedSetIntervalWire { upper_se: -0.1, ..wire.clone() },
        IdentifiedSetIntervalWire { critical_value: f64::NAN, ..wire.clone() },
        IdentifiedSetIntervalWire { replicates: 1, ..wire.clone() },
        IdentifiedSetIntervalWire { completions: 0, ..wire.clone() },
    ] {
        assert!(identified_set_interval_from_wire(&corrupt).is_err());
        assert!(
            encode_analysis_result_artifact(&with_structural(Some(corrupt)), names.clone(), "bad")
                .is_err()
        );
    }
}

#[test]
fn format_0_4_structural_result_migrates_without_an_interval() {
    // A 0.4 writer had no interval field: the body omits it entirely.
    let result = with_structural(None);
    let body = serde_json::to_value(&result).unwrap();
    assert!(body["structural_response"].get("identified_set_interval").is_none());
    let mut artifact =
        encode_analysis_result_artifact(&result, vec!["a".into(), "y".into()], "old").unwrap();
    artifact.manifest.format_version = crate::FormatVersion { major: 0, minor: 4 };
    artifact.manifest.minimum_reader_version = crate::FormatVersion { major: 0, minor: 4 };
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes).unwrap();
    let (migrated, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(migrated.manifest.format_version, STABLE_FORMAT);
    assert_eq!(decoded, result);
    assert!(decoded.structural_response.unwrap().identified_set_interval.is_none());
}

#[test]
fn function_valued_result_has_no_scalar_and_survives_json_bridge() {
    let mut result = fixture();
    assert_eq!(result.estimate, Some(1.0)); // Legacy numeric scalar remains readable.
    result.estimate = None;
    result.standard_error = None;
    let json = serde_json::to_value(&result).unwrap();
    assert!(json["estimate"].is_null());
    let restored: AnalysisResultWire = serde_json::from_value(json).unwrap();
    let artifact =
        encode_analysis_result_artifact(&restored, vec!["x".into(), "y".into()], "functional")
            .unwrap();
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes).unwrap();
    assert_eq!(decode_analysis_result_artifact(&bytes).unwrap().2, result);
    result.estimate = Some(f64::NAN);
    assert!(
        encode_analysis_result_artifact(&result, vec!["x".into(), "y".into()], "invalid").is_err()
    );
}

#[test]
fn validate_result_refuses_wrong_query_ids_and_keeps_the_valid_counterpart() {
    let names = vec!["a".into(), "y".into()];
    let ok = fixture();
    assert!(encode_analysis_result_artifact(&ok, names.clone(), "ok").is_ok());
    let mut wrong = ok;
    let mut query = serde_json::to_value(&wrong.identification.query).unwrap();
    query["response"]["functional"]["average_derivative"]["treatment"] = 1.into();
    query["response"]["functional"]["average_derivative"]["outcome"] = 0.into();
    wrong.identification.query = serde_json::from_value(query).unwrap();
    let err = encode_analysis_result_artifact(&wrong, names, "wrong").unwrap_err();
    assert!(err.to_string().contains("enclosing query"), "{err}");
}

fn pulse_query_wire(horizon: u32) -> crate::CausalQueryWire {
    use crate::query_wire::causal_query_to_wire;
    use antecedent_core::{CausalQuery, TemporalEffectQuery, VariableId};
    causal_query_to_wire(&CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_horizon_steps(horizon),
    ))
    .unwrap()
}

#[test]
fn validate_result_refuses_changed_horizon_and_keeps_the_matching_certificate() {
    let names = vec!["x".into(), "y".into()];
    let mut ok = fixture();
    ok.query = pulse_query_wire(1);
    ok.identification.query = ok.query.clone();
    ok.temporal_identification.push(TemporalIdentificationWire {
        horizon: 1,
        variables: vec![
            crate::HorizonAdjustmentNodeWire { variable: 0, offset: -1 },
            crate::HorizonAdjustmentNodeWire { variable: 1, offset: 0 },
        ],
        identification: ok.identification.clone(),
    });
    assert!(encode_analysis_result_artifact(&ok, names.clone(), "horizon-ok").is_ok());
    let mut stale = ok;
    stale.temporal_identification[0].horizon = 2;
    let err = encode_analysis_result_artifact(&stale, names, "horizon-stale").unwrap_err();
    assert!(err.to_string().contains("horizon"), "{err}");
}

#[test]
fn validate_result_refuses_inconsistent_atom_totals_and_keeps_unit_mass() {
    let names = vec!["a".into(), "y".into()];
    let mut ok = with_structural(None);
    ok.structural_response.as_mut().unwrap().atoms.push(StructuralResponseAtomWire {
        graph_key: 1,
        weight: 1.0,
        identification_status: crate::IdentificationStatusWire::NonparametricallyIdentified,
        value: None,
        posterior_artifact: None,
        response: None,
    });
    assert!(encode_analysis_result_artifact(&ok, names.clone(), "atoms-ok").is_ok());
    let mut bad = ok;
    bad.structural_response.as_mut().unwrap().atoms[0].weight = 0.4;
    let err = encode_analysis_result_artifact(&bad, names, "atoms-bad").unwrap_err();
    assert!(err.to_string().contains("atom totals"), "{err}");
}

#[test]
fn validate_result_refuses_inconsistent_masses_and_keeps_unit_mass() {
    let names = vec!["a".into(), "y".into()];
    let ok = with_structural(None);
    assert!(encode_analysis_result_artifact(&ok, names.clone(), "mass-ok").is_ok());
    let mut bad = ok;
    bad.structural_response.as_mut().unwrap().identified_mass = 0.5;
    let err = encode_analysis_result_artifact(&bad, names.clone(), "mass-bad").unwrap_err();
    assert!(err.to_string().contains("masses must sum to 1"), "{err}");
}

#[test]
fn validate_result_refuses_out_of_range_masses_that_sum_to_one() {
    let names = vec!["a".into(), "y".into()];
    let mut bad = with_structural(None);
    let structural = bad.structural_response.as_mut().unwrap();
    structural.identified_mass = 1.5;
    structural.unidentified_mass = -0.5;
    structural.unevaluable_mass = 0.0;
    structural.subsampled_out_mass = 0.0;
    let err = encode_analysis_result_artifact(&bad, names, "mass-range").unwrap_err();
    assert!(err.to_string().contains("inside [0, 1]"), "{err}");
}

#[test]
fn validate_result_refuses_standard_error_without_estimate() {
    let names = vec!["a".into(), "y".into()];
    let mut result = fixture();
    assert!(result.estimate.is_some() && result.standard_error.is_some());
    result.estimate = None;
    let err = encode_analysis_result_artifact(&result, names.clone(), "se-only").unwrap_err();
    assert!(err.to_string().contains("requires the estimate"), "{err}");
    result.standard_error = None;
    assert!(encode_analysis_result_artifact(&result, names, "se-none").is_ok());
}

#[test]
fn validate_result_refuses_scalar_estimate_under_not_identified() {
    let names = vec!["a".into(), "y".into()];
    let mut result = fixture();
    result.identification.status = "not_identified".into();
    assert_eq!(result.estimate, Some(1.0));
    let err = encode_analysis_result_artifact(&result, names.clone(), "uid").unwrap_err();
    assert!(err.to_string().contains("does not license a scalar estimate"), "{err}");
    result.estimate = None;
    result.standard_error = None;
    assert!(encode_analysis_result_artifact(&result, names, "uid-ok").is_ok());
}

fn mediation_slice() -> TemporalMediationSliceWire {
    TemporalMediationSliceWire {
        horizon: 1,
        identification_status: crate::IdentificationStatusWire::NonparametricallyIdentified,
        method: "front_door".into(),
        adjustment: vec![],
        effect: 0.5,
        total: Some(0.5),
        direct: Some(0.2),
        mediated: Some(0.3),
        uncertainty: TemporalMediationUncertaintyWire::Unavailable,
        identified_set: None,
        diagnostics: vec![],
    }
}

#[test]
fn validate_result_refuses_malformed_mediation_grid() {
    let names = vec!["a".into(), "y".into()];
    let mut ok = fixture();
    ok.mediation_grid =
        Some(TemporalMediationGridWire { slices: vec![mediation_slice()], joint_posterior: false });
    assert!(encode_analysis_result_artifact(&ok, names.clone(), "grid-ok").is_ok());

    let mut bad = ok.clone();
    bad.mediation_grid.as_mut().unwrap().slices[0].effect = f64::NAN;
    let err = encode_analysis_result_artifact(&bad, names.clone(), "grid-nan").unwrap_err();
    assert!(err.to_string().contains("mediation grid"), "{err}");

    let mut inverted = ok;
    inverted.mediation_grid.as_mut().unwrap().slices[0].identified_set = Some([1.0, 0.0]);
    let err = encode_analysis_result_artifact(&inverted, names, "grid-set").unwrap_err();
    assert!(err.to_string().contains("identified set"), "{err}");
}

#[test]
fn validate_result_refuses_malformed_cate() {
    let names = vec!["a".into(), "y".into()];
    let mut ok = fixture();
    ok.cate = Some(vec![0.1, 0.2].into());
    ok.cate_se = Some(vec![0.05, 0.04].into());
    assert!(encode_analysis_result_artifact(&ok, names.clone(), "cate-ok").is_ok());

    let mut bad_len = ok.clone();
    bad_len.cate_se = Some(vec![0.05].into());
    let err = encode_analysis_result_artifact(&bad_len, names.clone(), "cate-len").unwrap_err();
    assert!(err.to_string().contains("cate"), "{err}");

    let mut bad_dispersion = ok.clone();
    bad_dispersion.cate_leaf_dispersion = Some(vec![0.05, -1.0].into());
    let err =
        encode_analysis_result_artifact(&bad_dispersion, names.clone(), "cate-disp").unwrap_err();
    assert!(err.to_string().contains("leaf dispersion"), "{err}");

    let mut bad_nan = ok;
    bad_nan.cate = Some(vec![0.1, f64::NAN].into());
    let err = encode_analysis_result_artifact(&bad_nan, names, "cate-nan").unwrap_err();
    assert!(err.to_string().contains("cate"), "{err}");
}

#[test]
fn portable_family_summaries_roundtrip_without_changing_historical_absence() {
    let legacy = fixture();
    let old_json = serde_json::to_value(&legacy).unwrap();
    for field in [
        "joint_covariance",
        "simultaneous_interval",
        "adjusted_p_values",
        "family_contrast",
        "family_contrast_interval",
    ] {
        assert!(
            old_json.get(field).is_none(),
            "absent optional {field} changed historical payloads"
        );
    }
    let mut body = legacy;
    body.joint_covariance =
        Some(crate::JointCovarianceWire { dim: 2, values: vec![1.0, 0.25, 0.25, 2.0] });
    body.simultaneous_interval = Some((-1.0, 3.0, 0.95));
    body.adjusted_p_values = Some((0.1, 0.2));
    body.family_contrast = Some((0.5, 0.3));
    body.family_contrast_interval = Some((-0.1, 1.1, 0.95));
    let artifact =
        encode_analysis_result_artifact(&body, vec!["a".into(), "y".into()], "family").unwrap();
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes).unwrap();
    let (_, _, restored) = decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(restored, body);
}

#[test]
fn malformed_portable_family_covariance_and_bands_are_refused() {
    let names = vec!["a".into(), "y".into()];
    for covariance in [
        crate::JointCovarianceWire { dim: 2, values: vec![1.0] },
        crate::JointCovarianceWire { dim: 2, values: vec![1.0, 0.25, 0.5, 1.0] },
        crate::JointCovarianceWire { dim: 2, values: vec![1.0, 2.0, 2.0, 1.0] },
        crate::JointCovarianceWire { dim: 1, values: vec![-1.0] },
        crate::JointCovarianceWire { dim: 1, values: vec![f64::NAN] },
    ] {
        let mut body = fixture();
        body.joint_covariance = Some(covariance);
        let error =
            encode_analysis_result_artifact(&body, names.clone(), "bad-family").unwrap_err();
        assert!(error.to_string().contains("family inference"), "{error}");
    }
    let mut missing_covariance = fixture();
    missing_covariance.simultaneous_interval = Some((0.0, 2.0, 0.95));
    assert!(
        encode_analysis_result_artifact(&missing_covariance, names.clone(), "band-without-family")
            .is_err()
    );
    let mut missing_contrast = fixture();
    missing_contrast.joint_covariance =
        Some(crate::JointCovarianceWire { dim: 1, values: vec![1.0] });
    missing_contrast.family_contrast_interval = Some((0.0, 2.0, 0.95));
    assert!(
        encode_analysis_result_artifact(&missing_contrast, names, "band-without-contrast").is_err()
    );
}
