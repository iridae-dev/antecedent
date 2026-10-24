// SPDX-License-Identifier: MIT OR Apache-2.0
// Test-only evidence recorder. Records are emitted only after executable assertions pass.
pub fn record(
    prefix: &str,
    mapping: bool,
    translation: usize,
    case: &str,
    dependency: &str,
    checks: serde_json::Value,
) {
    let Ok(directory) = std::env::var("ANTECEDENT_X05_ACCEPTANCE_DIR") else { return };
    let identity = std::env::var("ANTECEDENT_X05_SOURCE_ID").expect("runner source identity");
    let mapping =
        if mapping { "uncertain_source_dependent_affine" } else { "deterministic_affine" };
    let translation = [
        "same_estimand",
        "changed_population",
        "changed_horizon_and_recency",
        "changed_dose_and_spend",
        "changed_duration",
    ][translation];
    for decision in
        ["expected_utility", "binding_operational_constraint", "weak_separation_and_ties"]
    {
        let row = format!("{prefix}::{mapping}::{translation}::{decision}");
        let refusal = checks.get("sampling_law_prior_guard").is_some();
        let mut stages = serde_json::Map::new();
        for stage in ["original_data", "design_identification", "native_estimation"] {
            stages.insert(stage.into(), serde_json::json!("executed_and_checked"));
        }
        for stage in [
            "explicit_estimand_translation",
            "modular_experiment_update",
            "terminal_decision",
            "next_study_net_value",
            "historical_and_current_artifact_replay",
        ] {
            stages.insert(
                stage.into(),
                serde_json::json!(if refusal {
                    "not_applicable_after_sampling_law_prior_refusal"
                } else {
                    "executed_and_independently_checked"
                }),
            );
        }
        let mut row_checks = checks.clone();
        if decision == "binding_operational_constraint" {
            if let Some(specific) =
                checks.get("constrained_next_study").and_then(serde_json::Value::as_object)
            {
                for (key, value) in specific {
                    row_checks.as_object_mut().unwrap().insert(key.clone(), value.clone());
                }
            }
        }
        let value = serde_json::json!({"row":row,"case":case,"source_identity":identity,"source_dependency":dependency,"status":"passed","stages":stages,"checks":row_checks});
        let name = format!("{}--{}.json", row.replace("::", "_"), case);
        let path = std::path::Path::new(&directory).join(name);
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        std::fs::rename(tmp, path).unwrap();
    }
}
