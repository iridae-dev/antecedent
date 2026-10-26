//! Acceptance obligations generated from actual design and bridge/decision advertisements.
// SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_estimate::joint_design::capabilities::JOINT_DESIGN_CAPABILITIES;

/// A single required experiment-to-decision acceptance obligation.
#[derive(Clone, Debug, serde::Serialize)]
pub struct JointExperimentAcceptanceRow {
    /// Stable semantic case identity, independent of which cases currently pass.
    pub id: String,
    /// Original scientific design family.
    pub design: String,
    /// Actual original statistical procedure meaning.
    pub inference: String,
    /// Actual estimand, never a local-to-population target substitution.
    pub target: String,
    /// Explicit paired mapping law.
    pub mapping: String,
    /// Population/time/dose/duration setting that must be translated.
    pub translation: String,
    /// Terminal utility/feasibility case.
    pub decision: String,
    /// Expected scoped result or scientific refusal. This is an obligation, not a test result.
    pub required_outcome: String,
    /// Outstanding implementation promises, distinct from a scientific refusal.
    pub implementation_gaps: Vec<String>,
    /// Declared statistical limits, kept separate from implementation obligations.
    pub statistical_limitations: Vec<String>,
    /// Stable-candidate measurement status; passing implementation tests grants no license.
    pub calibration: String,
    /// Complete procedure stages that the positive case must execute and independently check.
    pub stages: Vec<String>,
}
/// Generated matrix containing every advertised design and bridge/decision combination.
#[derive(Clone, Debug, serde::Serialize)]
pub struct JointExperimentAcceptanceMatrix {
    /// Version of the generated protocol; not an artifact trust or calibration license.
    pub version: u16,
    /// Explicit update semantics shared by all positive rows.
    pub modular_policy: String,
    /// Every required row; no filtering to existing passing combinations.
    pub rows: Vec<JointExperimentAcceptanceRow>,
}
/// Generate machine-readable acceptance obligations from the authoritative X07 catalogue.
///
/// A missing implementation remains in the matrix with its original promise. A
/// frequentist sampling law used as a Bayesian prior is a scientific refusal;
/// absence of a Bayesian native likelihood or compiler port is an implementation
/// gap. Neither is represented as a passing test result by this generator.
#[must_use]
pub fn joint_experiment_acceptance_matrix() -> JointExperimentAcceptanceMatrix {
    let mappings = ["deterministic_affine", "uncertain_source_dependent_affine"];
    let translations = [
        "same_estimand",
        "changed_population",
        "changed_horizon_and_recency",
        "changed_dose_and_spend",
        "changed_duration",
    ];
    let decisions =
        ["expected_utility", "binding_operational_constraint", "weak_separation_and_ties"];
    let stages = [
        "original_data",
        "design_identification",
        "native_estimation",
        "explicit_estimand_translation",
        "modular_experiment_update",
        "terminal_decision",
        "next_study_net_value",
        "historical_and_current_artifact_replay",
    ];
    let mut rows = Vec::new();
    for capability in &JOINT_DESIGN_CAPABILITIES {
        for inference in ["frequentist", "bayesian"] {
            for mapping in mappings {
                for translation in translations {
                    for decision in decisions {
                        let mut gaps = capability
                            .implementation_gaps
                            .iter()
                            .map(|s| (*s).to_owned())
                            .collect::<Vec<_>>();
                        if !capability.compiled_world_port && !capability.compiled_native_port {
                            gaps.push("experiment bridge needs the actual typed design target/source compiler port; an incompatible world-mean facade does not satisfy this row".into());
                        }
                        if inference == "bayesian" && capability.bayesian_owner.is_none() {
                            gaps.push("proper native design likelihood/prior and aligned posterior law must be implemented".into());
                        }
                        let required_outcome = if inference == "frequentist" {
                            "scoped_refusal_sampling_distribution_is_not_a_bayesian_prior"
                        } else {
                            "conditional_full_chain_execution_and_independent_stage_references"
                        };
                        rows.push(JointExperimentAcceptanceRow {
                            id: format!(
                                "{}::{inference}::{mapping}::{translation}::{decision}",
                                capability.design
                            ),
                            design: capability.design.into(),
                            inference: inference.into(),
                            target: capability.target.into(),
                            mapping: mapping.into(),
                            translation: translation.into(),
                            decision: decision.into(),
                            required_outcome: required_outcome.into(),
                            implementation_gaps: gaps,
                            statistical_limitations: capability
                                .statistical_limitations
                                .iter()
                                .map(|s| (*s).into())
                                .collect(),
                            calibration: capability.calibration.into(),
                            stages: stages.iter().map(|s| (*s).into()).collect(),
                        });
                    }
                }
            }
        }
    }
    append_retained_panel_obligations(&mut rows, &mappings, &translations, &decisions, &stages);
    JointExperimentAcceptanceMatrix {
        version: 1,
        modular_policy: "frozen_joint_importance; not a joint refit".into(),
        rows,
    }
}

fn append_retained_panel_obligations(
    rows: &mut Vec<JointExperimentAcceptanceRow>,
    mappings: &[&str],
    translations: &[&str],
    decisions: &[&str],
    stages: &[&str],
) {
    // The separately advertised model-conditional panel workflow supports three
    // retained time projections. These rows complement the design catalogue.
    for functional in ["terminal_mean", "window_average", "weighted_sum"] {
        for &mapping in mappings {
            for &translation in translations {
                for &decision in decisions {
                    rows.push(JointExperimentAcceptanceRow {
                        id: format!(
                            "retained_panel::{functional}::{mapping}::{translation}::{decision}"
                        ),
                        design: "retained_panel_model_conditional".into(),
                        inference: "bayesian".into(),
                        target: format!("exact retained joint-program {functional} contrast"),
                        mapping: mapping.into(),
                        translation: translation.into(),
                        decision: decision.into(),
                        required_outcome:
                            "conditional_full_chain_execution_and_independent_stage_references"
                                .into(),
                        implementation_gaps: vec![],
                        statistical_limitations: vec![
                            "frozen modular conditional update; not a joint refit".into(),
                        ],
                        calibration: "registered_unmeasured_candidate_bound".into(),
                        stages: stages.iter().map(|s| (*s).into()).collect(),
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_includes_every_advertised_combination_and_preserves_implementation_gaps() {
        let matrix = joint_experiment_acceptance_matrix();
        if let Ok(directory) = std::env::var("ANTECEDENT_X05_ACCEPTANCE_DIR") {
            std::fs::write(
                std::path::Path::new(&directory).join("protocol.json"),
                serde_json::to_vec_pretty(&matrix).unwrap(),
            )
            .unwrap();
        }
        assert_eq!(
            matrix.rows.len(),
            JOINT_DESIGN_CAPABILITIES.len() * 2 * 2 * 5 * 3 + 3 * 2 * 5 * 3
        );
        let ids = matrix.rows.iter().map(|r| &r.id).collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ids.len(), matrix.rows.len());
        for capability in &JOINT_DESIGN_CAPABILITIES {
            let rows =
                matrix.rows.iter().filter(|r| r.design == capability.design).collect::<Vec<_>>();
            assert_eq!(rows.len(), 60);
            for row in rows {
                for gap in capability.implementation_gaps {
                    assert!(row.implementation_gaps.iter().any(|s| s == gap));
                }
                if row.inference == "frequentist" {
                    assert!(row.required_outcome.starts_with("scoped_refusal_sampling"));
                }
                assert_eq!(row.stages.len(), 8);
            }
        }
        assert!(serde_json::to_string(&matrix).unwrap().contains("frozen_joint_importance"));
    }
}
