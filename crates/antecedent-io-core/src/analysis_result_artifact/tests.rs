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

/// The refusal text of a rejected value; the call must fail and name `detail`.
fn refused<T: std::fmt::Debug>(result: Result<T, impl std::fmt::Display>, detail: &str) {
    let error = result.expect_err("the value must be refused").to_string();
    assert!(error.contains(detail), "expected `{detail}` in `{error}`");
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
    refused(
        encode_analysis_result_artifact(
            &result,
            vec!["treatment".into(), "outcome".into()],
            "rcs-invalid-interval",
        ),
        "invalid panel DiD payload or fabricated interval",
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
    refused(
        validate_interventional_distribution(&result),
        "atom masses must sum to one per conditioning assignment",
    );

    let mut result = distribution_fixture();
    result.interventional_distribution.as_mut().unwrap().atoms[1].outcomes[0].0 = 0;
    refused(
        validate_interventional_distribution(&result),
        "atom does not match its query or has invalid mass",
    );

    let mut result = distribution_fixture();
    result.estimate = Some(0.6);
    refused(validate_interventional_distribution(&result), "atoms disagree with the reported mean");

    let mut result = distribution_fixture();
    result.interventional_distribution.as_mut().unwrap().atoms[1].probability = f64::NAN;
    refused(
        validate_interventional_distribution(&result),
        "atom does not match its query or has invalid mass",
    );
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
    refused(
        encode_analysis_result_artifact(&result, vec!["a".into()], "review"),
        "variable id 1 is outside causal payload header bounds (len=1)",
    );
    refused(
        encode_analysis_result_artifact(&result, vec!["a".into(), "a".into()], "review"),
        "duplicate causal payload variable name `a`",
    );
    let mut invalid = result;
    invalid.posterior_artifact = Some(vec![1, 2, 3]);
    refused(encode_analysis_result_artifact(&invalid, names, "review"), "io error");
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
    refused(
        encode_analysis_result_artifact(&result, names.clone(), "invalid"),
        "temporal namespace contains an unknown base variable or duplicate node",
    );
    result.identification_variables = None;
    refused(
        encode_analysis_result_artifact(&result, names, "missing"),
        "variable id 5 is outside causal payload header bounds (len=2)",
    );
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
    refused(
        encode_analysis_result_artifact(&result, names, "bad"),
        "mixture masses must be finite and inside [0, 1]",
    );
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
        let detail = "identified-set interval must be finite with level in (0, 1), ordered endpoints and bounds, nonnegative SDs, at least one completion and two replicates";
        refused(identified_set_interval_from_wire(&corrupt), detail);
        refused(
            encode_analysis_result_artifact(&with_structural(Some(corrupt)), names.clone(), "bad"),
            detail,
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
    refused(
        encode_analysis_result_artifact(&result, vec!["x".into(), "y".into()], "invalid"),
        "analysis scalar estimate must be finite when present",
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
        effect: Some(0.5),
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
    bad.mediation_grid.as_mut().unwrap().slices[0].effect = Some(f64::NAN);
    let err = encode_analysis_result_artifact(&bad, names.clone(), "grid-nan").unwrap_err();
    assert!(err.to_string().contains("mediation grid"), "{err}");

    let mut bounded = ok.clone();
    let slice = &mut bounded.mediation_grid.as_mut().unwrap().slices[0];
    slice.effect = None;
    slice.total = None;
    slice.direct = None;
    slice.mediated = None;
    slice.identification_status = crate::IdentificationStatusWire::GraphDependent;
    slice.identified_set = Some([0.2, 0.7]);
    let artifact = encode_analysis_result_artifact(&bounded, names.clone(), "grid-bounds").unwrap();
    assert_eq!(artifact.manifest.format_version, WITHHELD_MEDIATION_FORMAT);
    assert_eq!(artifact.manifest.minimum_reader_version, WITHHELD_MEDIATION_FORMAT);
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes).unwrap();
    let read = crate::read_and_migrate(bytes.as_slice()).unwrap();
    assert_eq!(read.manifest.format_version, WITHHELD_MEDIATION_FORMAT);
    assert_eq!(read.manifest.minimum_reader_version, WITHHELD_MEDIATION_FORMAT);
    let (_, _, decoded) = decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(decoded.mediation_grid, bounded.mediation_grid);
    assert_eq!(to_cbor(&Some(0.5)).unwrap(), to_cbor(&0.5_f64).unwrap());
    let mut foreign = artifact.clone();
    foreign.manifest.artifact_kind = crate::ArtifactKind::Other("other".into());
    assert!(crate::migrate_artifact(foreign).is_err());
    let mut inconsistent = artifact.clone();
    inconsistent.manifest.minimum_reader_version = STABLE_FORMAT;
    assert!(crate::migrate_artifact(inconsistent).is_err());
    let mut mislabeled = artifact;
    mislabeled.manifest.format_version = STABLE_FORMAT;
    mislabeled.manifest.minimum_reader_version = STABLE_FORMAT;
    let mut old_bytes = Vec::new();
    mislabeled.write_to(&mut old_bytes).unwrap();
    assert!(decode_analysis_result_artifact(&old_bytes).is_err());
    bounded.mediation_grid.as_mut().unwrap().slices[0].total = Some(0.5);
    assert!(encode_analysis_result_artifact(&bounded, names.clone(), "grid-bad-bounds").is_err());

    let mut false_identified = ok.clone();
    false_identified.mediation_grid.as_mut().unwrap().slices[0].effect = None;
    assert!(
        encode_analysis_result_artifact(&false_identified, names.clone(), "grid-missing").is_err()
    );
    let mut false_point = ok.clone();
    false_point.mediation_grid.as_mut().unwrap().slices[0].identification_status =
        crate::IdentificationStatusWire::NotIdentified;
    assert!(
        encode_analysis_result_artifact(&false_point, names.clone(), "grid-false-point").is_err()
    );

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
    refused(
        encode_analysis_result_artifact(&missing_covariance, names.clone(), "band-without-family"),
        "simultaneous interval requires valid covariance and bounds",
    );
    let mut missing_contrast = fixture();
    missing_contrast.joint_covariance =
        Some(crate::JointCovarianceWire { dim: 1, values: vec![1.0] });
    missing_contrast.family_contrast_interval = Some((0.0, 2.0, 0.95));
    refused(
        encode_analysis_result_artifact(&missing_contrast, names, "band-without-contrast"),
        "contrast interval requires its contrast and covariance",
    );
}

mod source_witness_cases {
    use std::sync::Arc;

    use antecedent_core::{
        CausalSchema, CausalSchemaBuilder, MeasurementSpec, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::column::{Int64Column, ValidityBitmap};
    use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData};

    use super::*;
    use crate::{
        DataPartitionIdentityWire, DataSnapshotIdentityWire, SOURCE_WITNESS_FORMAT,
        SOURCE_WITNESS_SECTION, SourceColumnWire, SourceWitnessError, SourceWitnessLimits,
        SourceWitnessWire, verify_source_witness,
    };

    fn names(count: usize) -> Vec<String> {
        ["x", "y", "unused", "extra"].iter().take(count).map(|name| (*name).into()).collect()
    }

    fn columns() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0, 0.0, 1.0, 1.0],
            vec![0.5, 2.25, -1.0, 3.5, 4.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
        ]
    }

    /// The witness written out by hand, not through `from_storage`.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the value is bounded by its validated numerical or fixture domain"
    )]
    fn witness(columns: &[Vec<f64>]) -> SourceWitnessWire {
        SourceWitnessWire {
            format: SOURCE_WITNESS_FORMAT,
            row_count: columns.first().map_or(0, Vec::len) as u64,
            columns: columns
                .iter()
                .enumerate()
                .map(|(index, values)| SourceColumnWire {
                    variable: index as u32,
                    values: values.clone(),
                })
                .collect(),
        }
    }

    /// Snapshot whose single partition digest comes from the public data
    /// constructors, independent of the witness code.
    fn snapshot(names: &[String], columns: &[Vec<f64>]) -> DataSnapshotIdentityWire {
        let data = TabularData::from_f64_columns(
            names.iter().zip(columns).map(|(name, values)| (name.as_str(), values.as_slice())),
        )
        .unwrap();
        let rows = data.storage().columns()[0].len() as u64;
        DataSnapshotIdentityWire {
            format: antecedent_core::IDENTITY_FORMAT,
            observation: [7; 32],
            modality: "tabular".into(),
            regularity: None,
            row_count: rows,
            unit_count: None,
            partitions: vec![DataPartitionIdentityWire {
                content: data.storage().content_digest(),
                row_count: rows,
                unit_id: None,
                regularity: None,
            }],
            interference: None,
            distribution_factor_laws: None,
            nested_counterfactual_fit: None,
            linear_fit_moments: None,
            cross_product_moments: None,
            participants: None,
            missing: None,
            sampling_design: None,
        }
    }

    fn schema(count: usize) -> CausalSchema {
        let mut builder = CausalSchemaBuilder::new();
        for name in names(count) {
            builder
                .add_variable(
                    name.as_str(),
                    ValueType::Continuous,
                    SmallRoleSet::empty(),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        builder.build().unwrap()
    }

    fn float(id: u32, values: Vec<f64>) -> OwnedColumn {
        let rows = values.len();
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(id),
                Arc::<[f64]>::from(values),
                ValidityBitmap::all_valid(rows),
            )
            .unwrap(),
        )
    }

    fn refusal(storage: Result<OwnedColumnarStorage, antecedent_data::DataError>) -> String {
        SourceWitnessWire::from_storage(&storage.unwrap()).unwrap_err().to_string()
    }

    #[test]
    fn source_witness_round_trips_through_cbor_and_storage() {
        let table = columns();
        let data = TabularData::from_f64_columns(
            names(3).iter().zip(&table).map(|(name, values)| (name.as_str(), values.as_slice())),
        )
        .unwrap();
        let carried = SourceWitnessWire::from_storage(data.storage()).unwrap();
        assert_eq!(carried, witness(&table));
        let decoded: SourceWitnessWire = from_cbor(&to_cbor(&carried).unwrap()).unwrap();
        assert_eq!(decoded, carried);
        let rebuilt = decoded.to_storage(&names(3), &SourceWitnessLimits::default()).unwrap();
        assert_eq!(rebuilt.content_digest(), data.storage().content_digest());
        assert_eq!(rebuilt.columns().len(), 3);
    }

    #[test]
    fn source_witness_from_storage_refuses_tables_it_cannot_carry_exactly() {
        let rows = 3;
        let plain = || vec![float(0, vec![1.0, 2.0, 3.0]), float(1, vec![4.0, 5.0, 6.0])];
        let masked = OwnedColumnarStorage::try_new(
            schema(2),
            plain(),
            Some(ValidityBitmap::all_valid(rows)),
            None,
        );
        assert!(refusal(masked).contains("analysis mask"));
        let weighted = OwnedColumnarStorage::try_new(
            schema(2),
            plain(),
            None,
            Some(Arc::<[f64]>::from(vec![1.0; rows])),
        );
        assert!(refusal(weighted).contains("weights"));
        let missing = OwnedColumnarStorage::try_new(
            schema(2),
            vec![float(0, vec![1.0, f64::NAN, 3.0]), float(1, vec![4.0, 5.0, 6.0])],
            None,
            None,
        );
        assert!(refusal(missing).contains("invalid"));
        let integer = OwnedColumnarStorage::try_new(
            schema(2),
            vec![
                float(0, vec![1.0, 2.0, 3.0]),
                OwnedColumn::Int64(
                    Int64Column::new(
                        VariableId::from_raw(1),
                        Arc::<[i64]>::from(vec![1, 2, 3]),
                        ValidityBitmap::all_valid(rows),
                    )
                    .unwrap(),
                ),
            ],
            None,
            None,
        );
        assert!(refusal(integer).contains("not Float64"));
        let infinite = OwnedColumnarStorage::try_new(
            schema(2),
            vec![float(0, vec![1.0, f64::INFINITY, 3.0]), float(1, vec![4.0, 5.0, 6.0])],
            None,
            None,
        );
        assert!(refusal(infinite).contains("non-finite"));
    }

    #[test]
    fn source_witness_to_storage_checks_declared_sizes_before_allocating() {
        let one_column = |row_count: u64| SourceWitnessWire {
            format: SOURCE_WITNESS_FORMAT,
            row_count,
            columns: vec![SourceColumnWire { variable: 0, values: Vec::new() }],
        };
        let limits = SourceWitnessLimits::default();
        // No value is present: a size check that ran after allocation or the length
        // check would report something other than TooLarge.
        assert_eq!(
            one_column(u64::MAX).to_storage(&names(1), &limits).unwrap_err(),
            SourceWitnessError::TooLarge
        );
        assert_eq!(
            one_column(limits.max_rows + 1).to_storage(&names(1), &limits).unwrap_err(),
            SourceWitnessError::TooLarge
        );
        let small_bytes =
            SourceWitnessLimits { max_bytes: 1024, max_rows: u64::MAX, max_columns: 8 };
        assert_eq!(
            one_column(1 << 20).to_storage(&names(1), &small_bytes).unwrap_err(),
            SourceWitnessError::TooLarge
        );
        let unbounded =
            SourceWitnessLimits { max_bytes: u64::MAX, max_rows: u64::MAX, max_columns: u64::MAX };
        let two_columns = SourceWitnessWire {
            format: SOURCE_WITNESS_FORMAT,
            row_count: u64::MAX,
            columns: (0..2)
                .map(|variable| SourceColumnWire { variable, values: Vec::new() })
                .collect(),
        };
        assert_eq!(
            two_columns.to_storage(&names(2), &unbounded).unwrap_err(),
            SourceWitnessError::TooLarge,
            "rows x columns x 8 overflows u64"
        );
        let few_columns = SourceWitnessLimits { max_columns: 2, ..SourceWitnessLimits::default() };
        assert_eq!(
            witness(&columns()).to_storage(&names(3), &few_columns).unwrap_err(),
            SourceWitnessError::TooLarge
        );
    }

    #[test]
    fn source_witness_to_storage_refuses_malformed_tables() {
        let limits = SourceWitnessLimits::default();
        let malformed = |wire: &SourceWitnessWire, count: usize| {
            matches!(wire.to_storage(&names(count), &limits), Err(SourceWitnessError::Malformed(_)))
        };
        let table = columns();
        assert!(!malformed(&witness(&table), 3));
        let mut descending = witness(&table[..2]);
        descending.columns[0].variable = 1;
        descending.columns[1].variable = 0;
        assert!(malformed(&descending, 2));
        let mut duplicate = witness(&table[..2]);
        duplicate.columns[1].variable = 0;
        assert!(malformed(&duplicate, 2));
        let mut gap = witness(&table[..2]);
        gap.columns[1].variable = 2;
        assert!(malformed(&gap, 2));
        assert!(malformed(&witness(&table), 2), "names must cover exactly the columns");
        let mut ragged = witness(&table);
        ragged.columns[1].values.pop();
        assert!(malformed(&ragged, 3));
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut cell = witness(&table);
            cell.columns[2].values[1] = bad;
            assert!(malformed(&cell, 3));
        }
        let mut unknown_format = witness(&table);
        unknown_format.format = SOURCE_WITNESS_FORMAT + 1;
        assert!(malformed(&unknown_format, 3));
        let repeated_names = vec!["x".to_owned(); 3];
        assert!(matches!(
            witness(&table).to_storage(&repeated_names, &limits),
            Err(SourceWitnessError::Malformed(_))
        ));
        assert!(malformed(&witness(&[]), 0), "a table without columns is not a witness");
    }

    #[test]
    fn verify_source_witness_binds_the_exact_table() {
        let limits = SourceWitnessLimits::default();
        let table = columns();
        let variables = names(3);
        let bound = snapshot(&variables, &table);
        verify_source_witness(&witness(&table), &variables, &bound, &limits)
            .expect("the true table, with its unused column, verifies");

        let mismatch = |wire: &SourceWitnessWire, snapshot: &DataSnapshotIdentityWire| {
            verify_source_witness(wire, &variables, snapshot, &limits).unwrap_err()
        };

        let mut changed = table.clone();
        changed[1][2] += 1.0;
        assert_eq!(
            mismatch(&witness(&changed), &bound),
            SourceWitnessError::Mismatch("content digest")
        );

        let mut swapped = table.clone();
        for column in &mut swapped {
            column.swap(0, 1);
        }
        assert_eq!(
            mismatch(&witness(&swapped), &bound),
            SourceWitnessError::Mismatch("content digest")
        );

        let mut unused_changed = table.clone();
        unused_changed[2][4] = 51.0;
        assert_eq!(
            mismatch(&witness(&unused_changed), &bound),
            SourceWitnessError::Mismatch("content digest"),
            "a column the analysis did not use is still bound"
        );

        assert!(matches!(
            verify_source_witness(&witness(&table[..2]), &variables, &bound, &limits),
            Err(SourceWitnessError::Malformed(_))
        ));
        assert!(matches!(
            verify_source_witness(&witness(&table[..2]), &variables[..2], &bound, &limits),
            Err(SourceWitnessError::Mismatch(_))
        ));

        let mut longer = witness(&table);
        longer.row_count += 1;
        assert!(verify_source_witness(&longer, &variables, &bound, &limits).is_err());
        let mut other_rows = bound.clone();
        other_rows.partitions[0].row_count += 1;
        assert_eq!(
            mismatch(&witness(&table), &other_rows),
            SourceWitnessError::Mismatch("row count")
        );

        let mut other_digest = bound.clone();
        other_digest.partitions[0].content[0] ^= 1;
        assert_eq!(
            mismatch(&witness(&table), &other_digest),
            SourceWitnessError::Mismatch("content digest")
        );

        let mut two = bound.clone();
        two.partitions.push(two.partitions[0].clone());
        assert_eq!(
            mismatch(&witness(&table), &two),
            SourceWitnessError::Mismatch("snapshot must have exactly one partition")
        );
    }

    fn sourced(source: Option<&SourceWitnessWire>) -> EncodedArtifact {
        encode_analysis_result_artifact_with_contract_and_source(
            &fixture(),
            vec!["a".into(), "y".into()],
            "witnessed",
            None,
            source,
        )
        .unwrap()
    }

    fn two_column_witness() -> SourceWitnessWire {
        witness(&columns()[..2])
    }

    fn ids(artifact: &EncodedArtifact) -> Vec<&str> {
        artifact.sections.iter().map(|section| section.id.as_str()).collect()
    }

    #[test]
    fn delegating_encoder_adds_no_source_section() {
        let compact = encode_analysis_result_artifact_with_contract(
            &fixture(),
            vec!["a".into(), "y".into()],
            "witnessed",
            None,
        )
        .unwrap();
        assert_eq!(ids(&compact), [HEADER_SECTION, BODY_SECTION]);
        let described: Vec<&str> =
            compact.manifest.sections.iter().map(|section| section.id.as_str()).collect();
        assert_eq!(described, [HEADER_SECTION, BODY_SECTION]);
        let explicit = sourced(None);
        assert_eq!(ids(&explicit), ids(&compact));
        for (left, right) in explicit.sections.iter().zip(&compact.sections) {
            assert_eq!(left.data, right.data);
        }
        assert_eq!(explicit.manifest.sections, compact.manifest.sections);
        assert_eq!(
            decode_source_witness_section(&compact, &SourceWitnessLimits::default()).unwrap(),
            None
        );
    }

    #[test]
    fn sourced_artifact_decodes_unchanged_and_carries_the_witness() {
        let carried = two_column_witness();
        let compact = sourced(None);
        let artifact = sourced(Some(&carried));
        assert_eq!(ids(&artifact), [HEADER_SECTION, BODY_SECTION, SOURCE_WITNESS_SECTION]);
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let mut compact_bytes = Vec::new();
        compact.write_to(&mut compact_bytes).unwrap();

        let (read_back, header, body) = decode_analysis_result_artifact(&bytes).unwrap();
        let (_, compact_header, compact_body) =
            decode_analysis_result_artifact(&compact_bytes).unwrap();
        assert_eq!(header, compact_header);
        assert_eq!(body, compact_body);
        assert_eq!(
            artifact.manifest.minimum_reader_version,
            compact.manifest.minimum_reader_version
        );
        assert_eq!(
            decode_source_witness_section(&read_back, &SourceWitnessLimits::default()).unwrap(),
            Some(carried)
        );
    }

    #[test]
    fn source_section_errors_are_typed_and_checked_before_decoding() {
        let carried = two_column_witness();
        let limits = SourceWitnessLimits::default();

        let mut duplicated = sourced(Some(&carried));
        let extra = duplicated.sections[2].clone();
        let descriptor = duplicated.manifest.sections[2].clone();
        duplicated.sections.push(extra);
        duplicated.manifest.sections.push(descriptor);
        refused(decode_source_witness_section(&duplicated, &limits), "duplicate section");

        // Bytes that are not CBOR: only a length check ahead of decoding reports the size.
        let mut oversized = sourced(Some(&carried));
        oversized.sections[2].data = Arc::from(vec![0xff_u8; 4096]);
        let tight = SourceWitnessLimits { max_bytes: 1024, ..limits };
        refused(decode_source_witness_section(&oversized, &tight), "exceeds its size limits");
        refused(decode_source_witness_section(&oversized, &limits), "cbor error");

        // Declared sizes are re-checked after decoding.
        let artifact = sourced(Some(&carried));
        let few_rows = SourceWitnessLimits { max_rows: 2, ..limits };
        refused(decode_source_witness_section(&artifact, &few_rows), "exceeds its size limits");

        let mut ragged_names = carried.clone();
        ragged_names.columns.pop();
        refused(
            encode_analysis_result_artifact_with_contract_and_source(
                &fixture(),
                vec!["a".into(), "y".into()],
                "witnessed",
                None,
                Some(&ragged_names),
            ),
            "column count differs",
        );
        let mut unknown = carried;
        unknown.format = SOURCE_WITNESS_FORMAT + 1;
        refused(
            encode_analysis_result_artifact_with_contract_and_source(
                &fixture(),
                vec!["a".into(), "y".into()],
                "witnessed",
                None,
                Some(&unknown),
            ),
            "unsupported format",
        );
    }
}

fn temporal_design_result() -> AnalysisResultWire {
    let query: crate::CausalQueryWire = serde_json::from_value(serde_json::json!({
        "temporal_design": {
            "outcome": 1,
            "units": ["a", "a", "b", "b"],
            "periods": [0, 1, 0, 1],
            "lag": 0,
            "assignment": {"two_arm": {
                "assignment": [true, false, true, false],
                "probability": [0.5, 0.5, 0.5, 0.5]
            }},
            "min_probability": 0.1
        }
    }))
    .unwrap();
    let mut result = fixture();
    result.query = query.clone();
    result.identification.query = query;
    result.estimate = None;
    result.standard_error = None;
    result.temporal_design = Some(TemporalDesignWire {
        design: "two_arm".into(),
        functional: "mean".into(),
        coordinates: vec!["effect".into(), "second".into()],
        estimate: vec![1.0, 2.0],
        standard_error: vec![0.5, 0.5],
        lower: vec![0.0, 1.0],
        upper: vec![2.0, 3.0],
        covariance: Some(vec![0.25, 0.1, 0.1, 0.25]),
        band_coordinates: vec![0, 1],
        band_critical_value: Some(2.2),
        band_lower: Some(vec![-0.1, 0.9]),
        band_upper: Some(vec![2.1, 3.1]),
        holm_adjusted_p: None,
        smoothing_bias_bound: None,
        pairs: 4,
        clusters: 2,
        unpaired: 0,
        minimum_probability: Some(0.5),
        replicates: Some([10, 0]),
        draws: None,
        uncertainty: "cluster_bootstrap".into(),
        calibration: "unmeasured".into(),
        licensed: false,
    });
    result
}

#[test]
fn temporal_design_section_fields_are_linked_to_the_query_and_each_other() {
    let validate = |edit: &dyn Fn(&mut TemporalDesignWire)| {
        let mut result = temporal_design_result();
        edit(result.temporal_design.as_mut().unwrap());
        super::temporal_design::validate_temporal_design(&result)
    };
    validate(&|_| {}).expect("the consistent section validates");
    refused(validate(&|s| s.functional = "quantile:0.5".into()), "functional");
    validate(&|s| s.functional = "mean".into()).expect("a mean functional");
    refused(validate(&|s| s.clusters = 5), "exceeds the rows");
    refused(validate(&|s| s.unpaired = 5), "exceeds the rows");
    refused(validate(&|s| s.minimum_probability = Some(0.0)), "minimum assignment probability");
    refused(
        validate(&|s| s.minimum_probability = Some(f64::NAN)),
        "minimum assignment probability",
    );
    refused(validate(&|s| s.draws = Some(0)), "posterior has draws");
    refused(validate(&|s| s.smoothing_bias_bound = Some(vec![0.0, -1.0])), "smoothing bias");
    refused(validate(&|s| s.covariance = Some(vec![0.25, 0.1, 0.2, 0.25])), "covariance");
    refused(validate(&|s| s.covariance = Some(vec![0.25, 0.9, 0.9, 0.25])), "covariance");
    refused(validate(&|s| s.covariance = Some(vec![f64::NAN, 0.0, 0.0, 0.25])), "covariance");
    refused(validate(&|s| s.band_lower = Some(vec![f64::NAN, 0.9])), "band is not finite");
    refused(validate(&|s| s.band_coordinates = vec![1, 0]), "strictly ascending");
}

#[test]
fn quasi_rollout_intervals_bands_and_diagnostics_must_be_finite() {
    let mut query = crate::causal_query_to_wire(&antecedent_core::CausalQuery::PanelDid(
        antecedent_core::PanelDidQuery::new(
            antecedent_core::VariableId::from_raw(1),
            [true, true, false, false],
            [false, true, false, true],
            ["a", "a", "b", "b"].map(std::sync::Arc::<str>::from),
            ["x", "x", "y", "y"].map(std::sync::Arc::<str>::from),
        ),
    ))
    .unwrap();
    if let crate::CausalQueryWire::PanelDid(panel) = &mut query {
        panel.rollout = Some(crate::rollout_query_wire::RolloutDidWire {
            design: "staggered_heterogeneity_robust".into(),
            estimator: "callaway_santanna".into(),
            comparison: "never_treated".into(),
            nuisance: "outcome_regression".into(),
            base_period: "varying".into(),
            aggregation: "simple".into(),
            anticipation: 0,
            covariates: Vec::new(),
            observation: None,
            observation_shift_bound: None,
        });
    }
    let q = super::quasi_rollout::NORMAL_Q975;
    let build = |edit: &dyn Fn(&mut QuasiRolloutWire)| {
        let mut result = fixture();
        result.query = query.clone();
        result.identification.query = query.clone();
        result.estimate = Some(1.0);
        result.standard_error = None;
        let mut rollout = QuasiRolloutWire {
            design: "staggered_did".into(),
            method: "callaway_santanna.simple".into(),
            effects: vec![DesignEffectWire {
                kind: "overall".into(),
                group: None,
                period: None,
                event_time: None,
                unit: None,
                estimate: 1.0,
                standard_error: Some(0.5),
                interval: Some([1.0 - q * 0.5, 1.0 + q * 0.5]),
                band: Some([1.0 - 2.2 * 0.5, 1.0 + 2.2 * 0.5]),
                treated: 2,
                comparison: 2,
            }],
            uncertainty: "cluster_bootstrap".into(),
            calibration: "unmeasured".into(),
            replicates_requested: 10,
            replicates_ok: 10,
            replicates_failed: 0,
            critical_value: Some(2.2),
            diagnostics: vec![("pre_trend_p".into(), 0.4)],
            licensed: false,
        };
        edit(&mut rollout);
        result.quasi_rollout = Some(rollout);
        super::quasi_rollout::validate_quasi_rollout(&result)
    };
    build(&|_| {}).expect("the consistent section validates");
    refused(build(&|r| r.critical_value = Some(f64::INFINITY)), "critical value");
    refused(build(&|r| r.critical_value = Some(f64::NAN)), "critical value");
    refused(build(&|r| r.diagnostics[0].1 = f64::NAN), "diagnostic is not finite");
    refused(build(&|r| r.effects[0].standard_error = Some(f64::INFINITY)), "standard error");
    refused(build(&|r| r.effects[0].interval = Some([f64::NEG_INFINITY, 2.0])), "not finite");
    refused(build(&|r| r.effects[0].band = Some([0.0, f64::NAN])), "not finite");
    refused(build(&|r| r.effects[0].estimate = f64::NAN), "quasi rollout section");
    let off_centre = |r: &mut QuasiRolloutWire| {
        r.effects[0].interval = Some([0.0, 2.5]);
        r.effects[0].band = None;
        r.critical_value = None;
    };
    refused(build(&off_centre), "normal interval");
    build(&|r| {
        off_centre(r);
        r.uncertainty = "rbc_hc0_delta_normal_fixed_bandwidth".into();
    })
    .expect("the robust bias-corrected interval is not centred on the conventional estimate");
}

#[test]
fn a_contrast_mean_must_be_the_mean_of_its_unit_effects() {
    let contrast = |mean: f64| OutcomeContrastWire {
        outcome: 1,
        mean,
        standard_error: None,
        unit_effects: UnitEffectsWire {
            effects: vec![1.0, 2.0, 3.0],
            homogeneous: false,
            intervals: None,
            extrapolative: None,
        },
    };
    super::counterfactual_contrasts::validate_outcome_contrasts(&[contrast(2.0)], &[1], 3)
        .expect("the mean of the effects");
    refused(
        super::counterfactual_contrasts::validate_outcome_contrasts(&[contrast(2.5)], &[1], 3),
        "mean of its unit effects",
    );
}
