//! Experimental evidence for decision-relevant outcomes.
//!
//! The estimator side ([`antecedent_estimate::complete_cluster_joint_contrast`]) and the
//! decision side ([`antecedent_design::decision_evidence`]) are separate layers; this module is
//! the one place they meet, so the design crate stays independent of the estimators.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_design::decision_evidence::*;
pub use antecedent_estimate::randomized_joint::{
    JointClusterContrast, complete_cluster_joint_contrast,
};

/// Wrap a joint cluster estimate as decision evidence.
///
/// The covariance of the arm-mean estimators travels with the means; nothing is recomputed or
/// widened. The unit of analysis stays the assignment cluster.
///
/// # Errors
///
/// [`EvidenceRefusal`] from [`JointExperimentEvidence::new`].
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the evidence record; every field is required"
)]
pub fn evidence_from_cluster_contrast(
    contrast: &JointClusterContrast,
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    horizon: u32,
    components: Vec<OutcomeComponent>,
    outcome_rows: usize,
    adjustments: Vec<ComponentAdjustment>,
    exposure: Exposure,
) -> Result<JointExperimentEvidence, EvidenceRefusal> {
    if components.len() != contrast.outcomes {
        return Err(EvidenceRefusal::Invalid(
            "components must align with the estimated outcomes".into(),
        ));
    }
    JointExperimentEvidence::new(
        evidence_id,
        population_id,
        regimes,
        horizon,
        components,
        (contrast.treated_means.clone(), contrast.control_means.clone()),
        Dependence::Joint {
            treated: contrast.treated_covariance.clone(),
            control: contrast.control_covariance.clone(),
        },
        DesignRecord {
            assignment_design: "cluster".into(),
            treated_units: contrast.treated_clusters,
            control_units: contrast.control_clusters,
            outcome_rows,
        },
        adjustments,
        exposure,
    )
}
